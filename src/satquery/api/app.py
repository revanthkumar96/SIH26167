"""FastAPI application.

Serves the whole system from one process: uploads, natural-language queries with a
live execution-trace stream, evidence artefacts, benchmark runs, and the web UI.

The backend is loaded lazily on first use. Starting the app must never require a
GPU or a model download -- that is what keeps the echo backend a usable
development and demo-fallback mode.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from satquery.agent.controller import Controller
from satquery.agent.tools import default_registry
from satquery.api.jobs import Job, JobStore, step_event
from satquery.api.settings import Settings
from satquery.eval.backends import BackendConfig, build_backend
from satquery.eval.datasets import BenchmarkConfig, load_benchmark
from satquery.eval.report import append_results, comparison_table, results_dashboard
from satquery.eval.runner import run_benchmark
from satquery.geo.raster import read_info, render_preview
from satquery.models import (
    DownloadProgress,
    describe_catalog,
    ensure_model,
    needs_model,
)
from satquery.schema import jsonable

STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2"}

#: Bundled real scenes, produced by scripts/fetch_sentinel_samples.py. Each set
#: exercises one of the mandatory input configurations with genuine imagery.
SAMPLE_SETS: list[dict[str, Any]] = [
    {
        "id": "mumbai_single",
        "title": "Mumbai — single scene",
        "subtitle": "Sentinel-2 L2A · 12 bands · 0% cloud",
        "config": "single image",
        "files": ["mumbai_optical_S2_20240206.tif"],
        "query": "Describe the land cover and major objects visible in this image.",
    },
    {
        "id": "mumbai_crossmodal",
        "title": "Mumbai — optical + SAR",
        "subtitle": "Sentinel-2 & Sentinel-1 RTC, same day",
        "config": "cross-modal pair",
        "files": [
            "mumbai_optical_S2_20240206.tif",
            "mumbai_sar_S1_VV_20240206.tif",
        ],
        "query": (
            "Use the optical and SAR images together to identify built-up and "
            "water-covered regions."
        ),
    },
    {
        "id": "ujani_bitemporal",
        "title": "Ujani reservoir — before / after",
        "subtitle": "Sentinel-2, dry season vs post-monsoon",
        "config": "bi-temporal pair",
        "files": ["ujani_before_20240518.tif", "ujani_after_20241129.tif"],
        "query": "What changed between these two dates, and where did the change occur?",
    },
]


def _sample_files_exist(config: Settings, sample: dict[str, Any]) -> bool:
    return all((config.samples_dir / name).exists() for name in sample["files"])


class QueryRequest(BaseModel):
    query: str = Field(default="", max_length=2000)
    image_ids: list[str] = Field(min_length=1, max_length=2)


class BenchmarkRequest(BaseModel):
    configs: list[str] = Field(min_length=1)
    limit: int | None = Field(default=32, ge=1, le=100_000)
    models: list[ModelSpec] | None = None


class ModelSpec(BaseModel):
    backend: str = Field(min_length=1, max_length=32)
    model: str = Field(min_length=1, max_length=256)


class ModelPullRequest(BaseModel):
    model: str = Field(min_length=1, max_length=256)
    revision: str | None = None


class DatasetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    with_images: bool = False
    shards: int = Field(default=2, ge=1, le=50)


def _data_present(benchmark: BenchmarkConfig) -> bool:
    """Whether every annotation file a benchmark needs is actually on disk.

    RSVQA splits its annotations across separate files named in ``extra``, so
    checking only ``annotations`` would report it as ready when nothing has been
    downloaded -- and the Benchmarks tab would invite a run that then fails.
    """
    required: list[Path] = []
    if benchmark.annotations:
        required.append(benchmark.annotation_path)
    for key in ("questions", "answers"):
        relative = benchmark.extra.get(key)
        if relative:
            required.append(benchmark.root / str(relative))
    return bool(required) and all(path.exists() for path in required)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Accepts injected settings so tests can isolate."""
    config = settings or Settings()
    config.ensure_dirs()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        """Fetch weights at startup so the first query does not pay for them.

        Runs in the background: the server answers /api/health and serves the UI
        while a multi-gigabyte download is still in flight, and the UI shows the
        progress rather than appearing hung.
        """
        if config.preload and needs_model(config.backend):
            app.state.preload_task = asyncio.create_task(preload_model())
        else:
            app.state.model_status = DownloadProgress(
                state="skipped",
                model=config.model,
                detail=f"backend '{config.backend}' needs no weights",
            )
        yield
        task = getattr(app.state, "preload_task", None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="SatQuery AI", version="0.1.0", lifespan=lifespan)
    app.state.settings = config
    app.state.jobs = JobStore()
    app.state.registry = default_registry()
    app.state.backend = None
    app.state.controller = None
    app.state.backend_lock = asyncio.Lock()
    app.state.model_status = DownloadProgress(state="idle", model=config.model)
    app.state.preload_task = None
    # Background tasks are held here: a bare create_task reference can be
    # garbage collected mid-run, cancelling the job silently.
    app.state.tasks = set()

    def spawn(coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        return task

    # -- weights and backend --------------------------------------------

    async def fetch_weights() -> DownloadProgress:
        """Download the model if it is not already on disk."""
        loop = asyncio.get_running_loop()

        def on_update(progress: DownloadProgress) -> None:
            # Called from the download thread; the status object is replaced
            # wholesale so a reader never sees a half-updated record.
            loop.call_soon_threadsafe(
                setattr, app.state, "model_status", replace(progress)
            )

        status = await asyncio.to_thread(
            ensure_model,
            config.model,
            config.revision,
            on_update,
            config.allow_download,
            config.models_dir,
        )
        app.state.model_status = status
        return status

    async def preload_model() -> None:
        status = await fetch_weights()
        if status.state != "ready":
            return
        with contextlib.suppress(Exception):
            # A backend that fails to construct must not take the server down;
            # the error surfaces on the first query instead, with a real message.
            await get_controller()

    async def get_controller() -> Controller:
        if app.state.controller is None:
            async with app.state.backend_lock:
                if app.state.controller is None:
                    # Weights live in a plain directory, so the backend is
                    # pointed at the resolved path rather than the repo id --
                    # otherwise it would re-resolve through the hub cache and
                    # download a second copy.
                    model_ref = config.model
                    if needs_model(config.backend):
                        status = app.state.model_status
                        if status.state != "ready":
                            status = await fetch_weights()
                        if status.state != "ready":
                            raise HTTPException(
                                503,
                                f"model '{config.model}' is unavailable: "
                                f"{status.detail}",
                            )
                        model_ref = status.path or config.model
                    try:
                        backend = await asyncio.to_thread(
                            build_backend,
                            config.backend,
                            BackendConfig(
                                model=model_ref,
                                dtype=config.dtype,
                                max_side=config.max_side,
                            ),
                        )
                    except Exception as exc:
                        # A missing dependency or an unloadable checkpoint is an
                        # operator problem, not a bug: report it as such rather
                        # than returning a 500 with a stack trace.
                        raise HTTPException(
                            503,
                            f"backend '{config.backend}' could not load "
                            f"'{model_ref}': {type(exc).__name__}: {exc}",
                        ) from exc
                    app.state.backend = backend
                    app.state.controller = Controller(
                        backend,
                        registry=app.state.registry,
                        workroot=config.artifacts_dir,
                    )
        return app.state.controller

    # -- introspection ---------------------------------------------------

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "settings": config.describe(),
            "backend_loaded": app.state.controller is not None,
            "model": app.state.model_status.as_dict(),
            "tools": len(app.state.registry),
        }

    @app.get("/api/model")
    async def model_status() -> dict[str, Any]:
        """Weight download state, polled by the UI while a fetch is in flight."""
        return app.state.model_status.as_dict()

    @app.get("/api/models")
    async def list_models() -> dict[str, Any]:
        """Recommended bake-off catalog with local presence."""
        return {
            "active": {"backend": config.backend, "model": config.model},
            "catalog": describe_catalog(config.models_dir),
        }

    @app.post("/api/models/pull")
    async def pull_model(request: ModelPullRequest) -> dict[str, Any]:
        """Download model weights in the background."""
        job = app.state.jobs.create("dataset", model=request.model)
        loop = asyncio.get_running_loop()
        last = {"percent": -1.0}

        def on_update(progress: DownloadProgress) -> None:
            percent = progress.percent or 0.0
            if progress.state == "downloading" and percent - last["percent"] < 1.0:
                return
            last["percent"] = percent
            loop.call_soon_threadsafe(
                job.emit,
                {"type": "progress", "progress": progress.as_dict()},
            )

        async def execute() -> None:
            job.status = "running"
            job.emit({"type": "start", "model": request.model})
            status = await asyncio.to_thread(
                ensure_model,
                request.model,
                request.revision,
                on_update,
                config.allow_download,
                config.models_dir,
            )
            job.result = status.as_dict()
            if status.state == "ready":
                job.status = "done"
                job.emit({"type": "complete", "progress": job.result})
            else:
                job.status = "error"
                job.error = status.detail
                job.emit({"type": "error", "message": status.detail})

        spawn(execute())
        return {"run_id": job.id}

    @app.get("/api/results")
    async def results() -> dict[str, Any]:
        """Bake-off comparison from the shared results CSV."""
        return results_dashboard(config.results_csv)

    @app.get("/api/tools")
    async def tools() -> dict[str, Any]:
        """The predefined registry the controller selects from."""
        return {"tools": app.state.registry.describe()}

    # -- uploads ---------------------------------------------------------

    @app.post("/api/upload")
    async def upload(files: list[UploadFile]) -> dict[str, Any]:
        if not files:
            raise HTTPException(400, "no files supplied")
        if len(files) > 2:
            raise HTTPException(
                400,
                "at most two images: one scene, a bi-temporal pair, or a "
                "co-registered optical-SAR pair",
            )

        uploaded: list[dict[str, Any]] = []
        for upload_file in files:
            suffix = Path(upload_file.filename or "").suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise HTTPException(
                    400,
                    f"unsupported format '{suffix or upload_file.filename}'. "
                    f"Use GeoTIFF/TIFF for geospatial inputs; PNG and JPEG are "
                    f"accepted for benchmark imagery.",
                )

            image_id = uuid.uuid4().hex[:12]
            destination = config.uploads_dir / f"{image_id}{suffix}"
            with destination.open("wb") as handle:
                shutil.copyfileobj(upload_file.file, handle)

            size_mb = destination.stat().st_size / (1024 * 1024)
            if size_mb > config.max_upload_mb:
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    413, f"{upload_file.filename} exceeds {config.max_upload_mb} MB"
                )

            try:
                info = await asyncio.to_thread(read_info, destination)
                preview = config.previews_dir / f"{image_id}.png"
                await asyncio.to_thread(render_preview, destination, preview)
            except Exception as exc:
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    400,
                    f"could not read {upload_file.filename}: {exc}. "
                    "GeoTIFF inputs need the geo extra: pip install -e '.[geo]'",
                ) from exc

            uploaded.append(
                {
                    "id": image_id,
                    "filename": upload_file.filename,
                    "preview": f"/previews/{image_id}.png",
                    "info": info.as_dict(),
                }
            )

        return {"images": uploaded}

    @app.get("/api/samples")
    async def samples() -> dict[str, Any]:
        """Bundled scenes, so the UI can demonstrate itself without a file picker."""
        return {"samples": [s for s in SAMPLE_SETS if _sample_files_exist(config, s)]}

    @app.post("/api/samples/{sample_id}/load")
    async def load_sample(sample_id: str) -> dict[str, Any]:
        """Register a bundled scene as if it had been uploaded."""
        chosen = next((s for s in SAMPLE_SETS if s["id"] == sample_id), None)
        if chosen is None or not _sample_files_exist(config, chosen):
            raise HTTPException(404, f"sample '{sample_id}' is not available")

        loaded: list[dict[str, Any]] = []
        for filename in chosen["files"]:
            source = config.samples_dir / filename
            image_id = uuid.uuid4().hex[:12]
            destination = config.uploads_dir / f"{image_id}{source.suffix}"
            shutil.copyfile(source, destination)

            try:
                info = await asyncio.to_thread(read_info, destination)
                preview = config.previews_dir / f"{image_id}.png"
                await asyncio.to_thread(render_preview, destination, preview)
            except Exception as exc:
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    400,
                    f"could not read {filename}: {exc}. "
                    "GeoTIFF inputs need the geo extra: pip install -e '.[geo]'",
                ) from exc
            loaded.append(
                {
                    "id": image_id,
                    "filename": filename,
                    "preview": f"/previews/{image_id}.png",
                    "info": info.as_dict(),
                }
            )
        return {"images": loaded, "query": chosen.get("query", "")}

    def resolve_image(image_id: str) -> Path:
        matches = sorted(config.uploads_dir.glob(f"{image_id}.*"))
        if not matches:
            raise HTTPException(404, f"unknown image id '{image_id}'")
        return matches[0]

    # -- queries ---------------------------------------------------------

    @app.post("/api/query")
    async def query(request: QueryRequest) -> dict[str, Any]:
        paths = [resolve_image(image_id) for image_id in request.image_ids]
        controller = await get_controller()

        job = app.state.jobs.create(
            "query", query=request.query, image_ids=request.image_ids
        )
        loop = asyncio.get_running_loop()

        def on_step(step: Any) -> None:
            # Called from the worker thread; hop back to the loop before touching
            # subscriber queues.
            loop.call_soon_threadsafe(job.emit, step_event(step))

        async def execute() -> None:
            job.status = "running"
            job.emit({"type": "start", "query": request.query})
            try:
                trace = await asyncio.to_thread(
                    controller.run, request.query, paths, on_step, job.id
                )
                job.result = trace.to_dict()
                job.status = "done"
                job.emit({"type": "complete", "trace": job.result})
            except Exception as exc:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.emit({"type": "error", "message": job.error})

        spawn(execute())
        return {"run_id": job.id}

    # -- benchmarks ------------------------------------------------------

    @app.get("/api/benchmarks")
    async def benchmarks() -> dict[str, Any]:
        """Benchmark configs on disk, with whether their data is present."""
        found: list[dict[str, Any]] = []
        for path in sorted(config.bench_config_dir.glob("*.yaml")):
            entry: dict[str, Any] = {"config": str(path), "name": path.stem}
            try:
                benchmark = BenchmarkConfig.from_yaml(path)
                entry.update(
                    {
                        "name": benchmark.name,
                        "task": str(benchmark.task),
                        "adapter": benchmark.adapter,
                        "data_present": _data_present(benchmark),
                    }
                )
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            found.append(entry)
        return {"benchmarks": found}

    @app.get("/api/datasets")
    async def datasets() -> dict[str, Any]:
        """The prescribed benchmarks, their official sources, and what is on disk."""
        from satquery.data import describe_all

        return {"datasets": describe_all(config.data_root)}

    @app.post("/api/datasets/pull")
    async def pull_dataset(request: DatasetRequest) -> dict[str, Any]:
        """Download a benchmark from the source named in the problem statement."""
        from satquery.data import SOURCES, DataProgress, pull

        if request.name not in SOURCES:
            raise HTTPException(404, f"unknown dataset '{request.name}'")

        job = app.state.jobs.create("dataset", dataset=request.name)
        loop = asyncio.get_running_loop()
        last = {"percent": -1.0}

        def on_update(progress: DataProgress) -> None:
            # Downloads emit per megabyte; forward whole percent steps and every
            # state change, so a websocket is not flooded on a 3.8 GB archive.
            percent = progress.percent or 0.0
            if progress.state == "downloading" and percent - last["percent"] < 1.0:
                return
            last["percent"] = percent
            loop.call_soon_threadsafe(
                job.emit, {"type": "progress", "progress": progress.as_dict()}
            )

        async def execute() -> None:
            job.status = "running"
            job.emit({"type": "start", "dataset": request.name})
            status = await asyncio.to_thread(
                pull,
                request.name,
                config.data_root,
                on_update,
                request.with_images,
                request.shards,
            )
            job.result = status.as_dict()
            if status.state == "ready":
                job.status = "done"
                job.emit({"type": "complete", "progress": job.result})
            else:
                job.status = "error"
                job.error = status.detail
                job.emit({"type": "error", "message": status.detail})

        spawn(execute())
        return {"run_id": job.id}

    @app.post("/api/benchmarks/run")
    async def run_benchmarks(request: BenchmarkRequest) -> dict[str, Any]:
        configs: list[BenchmarkConfig] = []
        for entry in request.configs:
            path = Path(entry)
            if not path.exists():
                raise HTTPException(404, f"benchmark config not found: {entry}")
            benchmark = BenchmarkConfig.from_yaml(path)
            if request.limit is not None:
                benchmark.limit = request.limit
            configs.append(benchmark)

        model_specs: list[ModelSpec]
        if request.models:
            model_specs = request.models
        else:
            controller = await get_controller()
            model_specs = [
                ModelSpec(
                    backend=config.backend,
                    model=controller.backend.config.model,
                )
            ]

        job = app.state.jobs.create(
            "benchmark",
            configs=[c.name for c in configs],
            limit=request.limit,
            models=[m.model_dump() for m in model_specs],
        )
        loop = asyncio.get_running_loop()

        def progress(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(job.emit, event)

        def work() -> list[Any]:
            all_results = []
            for spec in model_specs:
                progress(
                    {
                        "type": "model_start",
                        "backend": spec.backend,
                        "model": spec.model,
                    }
                )
                model_ref = spec.model
                if needs_model(spec.backend):
                    status = ensure_model(
                        spec.model,
                        config.revision,
                        allow_download=config.allow_download,
                        models_dir=config.models_dir,
                    )
                    if status.state != "ready":
                        raise RuntimeError(
                            f"model '{spec.model}' unavailable: {status.detail}"
                        )
                    model_ref = status.path or spec.model

                with build_backend(
                    spec.backend,
                    BackendConfig(
                        model=model_ref,
                        dtype=config.dtype,
                        max_side=config.max_side,
                    ),
                ) as backend:
                    for benchmark in configs:
                        progress(
                            {
                                "type": "benchmark_start",
                                "name": benchmark.name,
                                "model": spec.model,
                                "backend": spec.backend,
                            }
                        )
                        slug = spec.model.replace("/", "__")
                        result = run_benchmark(
                            load_benchmark(benchmark),
                            backend,
                            output_dir=config.workspace
                            / "bench"
                            / job.id
                            / slug
                            / benchmark.name,
                            progress_every=0,
                        )
                        all_results.append(result)
                        progress(
                            {
                                "type": "benchmark_result",
                                "name": benchmark.name,
                                "task": result.task,
                                "metrics": result.metrics,
                                "num_samples": result.num_samples,
                                "duration_s": round(result.duration_s, 2),
                                "model": spec.model,
                                "backend": spec.backend,
                            }
                        )
                progress(
                    {
                        "type": "model_complete",
                        "backend": spec.backend,
                        "model": spec.model,
                    }
                )
            append_results(all_results, config.results_csv)
            return all_results

        async def execute() -> None:
            job.status = "running"
            job.emit(
                {
                    "type": "start",
                    "configs": [c.name for c in configs],
                    "models": [m.model_dump() for m in model_specs],
                }
            )
            try:
                results = await asyncio.to_thread(work)
                job.result = {
                    "table": comparison_table(results),
                    "results": [r.summary() for r in results],
                    "dashboard": results_dashboard(config.results_csv),
                }
                job.status = "done"
                job.emit({"type": "complete", "result": job.result})
            except Exception as exc:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.emit({"type": "error", "message": job.error})

        spawn(execute())
        return {"run_id": job.id}

    # -- job access ------------------------------------------------------

    @app.get("/api/runs")
    async def runs() -> dict[str, Any]:
        return {"runs": app.state.jobs.list()}

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str) -> JSONResponse:
        job = app.state.jobs.get(run_id)
        if job is None:
            raise HTTPException(404, f"unknown run '{run_id}'")
        return JSONResponse(jsonable(job.detail()))

    @app.websocket("/ws/runs/{run_id}")
    async def run_stream(websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        job: Job | None = app.state.jobs.get(run_id)
        if job is None:
            await websocket.send_json({"type": "error", "message": "unknown run"})
            await websocket.close()
            return
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            async for event in app.state.jobs.stream(job):
                await websocket.send_json(jsonable(event))
        with contextlib.suppress(RuntimeError):
            await websocket.close()

    # -- files -----------------------------------------------------------

    @app.get("/artifacts/{run_id}/{filename}")
    async def artifact(run_id: str, filename: str) -> FileResponse:
        path = (config.artifacts_dir / run_id / filename).resolve()
        root = config.artifacts_dir.resolve()
        if root not in path.parents or not path.is_file():
            raise HTTPException(404, "artifact not found")
        return FileResponse(path)

    @app.get("/previews/{filename}")
    async def preview(filename: str) -> FileResponse:
        path = (config.previews_dir / filename).resolve()
        root = config.previews_dir.resolve()
        if root not in path.parents or not path.is_file():
            raise HTTPException(404, "preview not found")
        return FileResponse(path)

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

    return app


app = create_app()
