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
from satquery.eval.report import append_results, comparison_table
from satquery.eval.runner import run_benchmark
from satquery.geo.raster import read_info, render_preview
from satquery.schema import jsonable

STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2"}


class QueryRequest(BaseModel):
    query: str = Field(default="", max_length=2000)
    image_ids: list[str] = Field(min_length=1, max_length=2)


class BenchmarkRequest(BaseModel):
    configs: list[str] = Field(min_length=1)
    limit: int | None = Field(default=32, ge=1, le=100_000)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Accepts injected settings so tests can isolate."""
    config = settings or Settings()
    config.ensure_dirs()

    app = FastAPI(title="SatQuery AI", version="0.1.0")
    app.state.settings = config
    app.state.jobs = JobStore()
    app.state.registry = default_registry()
    app.state.backend = None
    app.state.controller = None
    app.state.backend_lock = asyncio.Lock()
    # Background tasks are held here: a bare create_task reference can be
    # garbage collected mid-run, cancelling the job silently.
    app.state.tasks = set()

    def spawn(coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        return task

    # -- backend, loaded on first use -----------------------------------

    async def get_controller() -> Controller:
        if app.state.controller is None:
            async with app.state.backend_lock:
                if app.state.controller is None:
                    backend = await asyncio.to_thread(
                        build_backend,
                        config.backend,
                        BackendConfig(
                            model=config.model,
                            dtype=config.dtype,
                            max_side=config.max_side,
                        ),
                    )
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
            "tools": len(app.state.registry),
        }

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
                    400, f"could not read {upload_file.filename}: {exc}"
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
                        "data_present": benchmark.annotation_path.exists()
                        or bool(benchmark.extra.get("questions")),
                    }
                )
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            found.append(entry)
        return {"benchmarks": found}

    @app.post("/api/benchmarks/run")
    async def run_benchmarks(request: BenchmarkRequest) -> dict[str, Any]:
        controller = await get_controller()
        backend = app.state.backend

        configs: list[BenchmarkConfig] = []
        for entry in request.configs:
            path = Path(entry)
            if not path.exists():
                raise HTTPException(404, f"benchmark config not found: {entry}")
            benchmark = BenchmarkConfig.from_yaml(path)
            if request.limit is not None:
                benchmark.limit = request.limit
            configs.append(benchmark)

        job = app.state.jobs.create(
            "benchmark",
            configs=[c.name for c in configs],
            limit=request.limit,
            model=controller.backend.config.model,
        )
        loop = asyncio.get_running_loop()

        def progress(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(job.emit, event)

        def work() -> list[Any]:
            results = []
            for benchmark in configs:
                progress({"type": "benchmark_start", "name": benchmark.name})
                result = run_benchmark(
                    load_benchmark(benchmark),
                    backend,
                    output_dir=config.workspace / "bench" / job.id / benchmark.name,
                    progress_every=0,
                )
                results.append(result)
                progress(
                    {
                        "type": "benchmark_result",
                        "name": benchmark.name,
                        "task": result.task,
                        "metrics": result.metrics,
                        "num_samples": result.num_samples,
                        "duration_s": round(result.duration_s, 2),
                    }
                )
            append_results(results, config.results_csv)
            return results

        async def execute() -> None:
            job.status = "running"
            job.emit({"type": "start", "configs": [c.name for c in configs]})
            try:
                results = await asyncio.to_thread(work)
                job.result = {
                    "table": comparison_table(results),
                    "results": [r.summary() for r in results],
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
