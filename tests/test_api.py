"""API surface: uploads, queries, trace streaming, benchmarks and artefacts.

Runs against the echo backend through the real ASGI app, so the whole request
path is exercised without a GPU.
"""

import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from satquery.api.app import create_app
from satquery.api.jobs import JobStore
from satquery.api.settings import Settings


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        backend="echo",
        model="echo",
        workspace=tmp_path / "workspace",
        bench_config_dir=tmp_path / "bench",
    )
    settings.bench_config_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(settings)) as test_client:
        test_client.settings = settings
        yield test_client


def _image_bytes(tmp_path, name, size=(64, 64), colour=(20, 100, 50)):
    path = tmp_path / name
    Image.new("RGB", size, colour).save(path)
    return path.read_bytes()


def _upload(client, tmp_path, names):
    files = [
        ("files", (name, _image_bytes(tmp_path, name), "image/png")) for name in names
    ]
    response = client.post("/api/upload", files=files)
    assert response.status_code == 200, response.text
    return [image["id"] for image in response.json()["images"]]


# -- health and introspection ----------------------------------------------


def test_health_reports_settings_without_loading_a_model(client):
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["settings"]["backend"] == "echo"
    assert payload["backend_loaded"] is False
    assert payload["tools"] > 0


def test_registry_is_exposed_with_permitted_params(client):
    tools = client.get("/api/tools").json()["tools"]
    names = {tool["name"] for tool in tools}
    assert {"change_mask", "sar_indices", "vlm_vqa"} <= names

    change = next(t for t in tools if t["name"] == "change_mask")
    assert "threshold" in change["allowed_params"]
    assert change["accepts"] == "bitemporal_pair"


# -- uploads ---------------------------------------------------------------


def test_upload_returns_preview_and_metadata(client, tmp_path):
    response = client.post(
        "/api/upload",
        files=[
            ("files", ("scene.png", _image_bytes(tmp_path, "scene.png"), "image/png"))
        ],
    )
    assert response.status_code == 200
    image = response.json()["images"][0]
    assert image["info"]["size"] == [64, 64]
    assert image["info"]["bands"] == 3
    assert client.get(image["preview"]).status_code == 200


def test_upload_rejects_unsupported_format(client):
    response = client.post(
        "/api/upload", files=[("files", ("notes.txt", b"hello", "text/plain"))]
    )
    assert response.status_code == 400
    assert "unsupported format" in response.json()["detail"]


def test_upload_rejects_more_than_two_images(client, tmp_path):
    files = [
        ("files", (f"{i}.png", _image_bytes(tmp_path, f"{i}.png"), "image/png"))
        for i in range(3)
    ]
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "at most two images" in response.json()["detail"]


def test_query_with_unknown_image_id_is_404(client):
    response = client.post("/api/query", json={"query": "x", "image_ids": ["nope"]})
    assert response.status_code == 404


# -- query end to end ------------------------------------------------------


def test_single_image_query_streams_then_completes(client, tmp_path):
    ids = _upload(client, tmp_path, ["scene.png"])
    run_id = client.post(
        "/api/query", json={"query": "Is there water?", "image_ids": ids}
    ).json()["run_id"]

    events = []
    with client.websocket_connect(f"/ws/runs/{run_id}") as socket:
        while True:
            event = socket.receive_json()
            events.append(event)
            if event["type"] in {"complete", "error"}:
                break

    kinds = [e["type"] for e in events]
    assert kinds[0] == "start"
    assert "step" in kinds
    assert kinds[-1] == "complete"

    trace = events[-1]["trace"]
    assert trace["routed_task"] == "vqa"
    assert trace["answer"] == "yes"
    assert trace["steps"][0]["tool"] == "vlm_vqa"
    assert trace["routing_rule"]


def test_bitemporal_query_runs_specialist_first_and_serves_the_mask(client, tmp_path):
    ids = _upload(client, tmp_path, ["t1.png", "t2.png"])
    run_id = client.post(
        "/api/query", json={"query": "What changed?", "image_ids": ids}
    ).json()["run_id"]

    with client.websocket_connect(f"/ws/runs/{run_id}") as socket:
        while True:
            event = socket.receive_json()
            if event["type"] in {"complete", "error"}:
                break

    assert event["type"] == "complete", event
    trace = event["trace"]
    assert [s["tool"] for s in trace["steps"]] == ["change_mask", "vlm_change_caption"]

    mask = next(e for e in trace["evidence"] if e["type"] == "mask")
    assert client.get(f"/{mask['uri']}").status_code == 200


def test_run_detail_is_available_after_completion(client, tmp_path):
    ids = _upload(client, tmp_path, ["scene.png"])
    run_id = client.post(
        "/api/query", json={"query": "Describe this", "image_ids": ids}
    ).json()["run_id"]

    with client.websocket_connect(f"/ws/runs/{run_id}") as socket:
        while socket.receive_json()["type"] not in {"complete", "error"}:
            pass

    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["status"] == "done"
    assert detail["result"]["routed_task"] == "caption"
    assert json.dumps(detail)  # the whole payload must be JSON-safe

    assert any(r["id"] == run_id for r in client.get("/api/runs").json()["runs"])


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/missing").status_code == 404


def test_websocket_for_unknown_run_reports_an_error(client):
    with client.websocket_connect("/ws/runs/missing") as socket:
        assert socket.receive_json()["type"] == "error"


# -- artefact path safety --------------------------------------------------


def test_artifact_traversal_is_refused(client):
    response = client.get("/artifacts/..%2f..%2fsettings/x.png")
    assert response.status_code == 404


# -- benchmarks ------------------------------------------------------------


def test_benchmarks_listing_flags_missing_data(client):
    config = client.settings.bench_config_dir / "demo.yaml"
    config.write_text(
        "name: demo\nadapter: vrsbench_vqa\ntask: vqa\n"
        "root: data/nowhere\nannotations: missing.json\n",
        encoding="utf-8",
    )
    benchmarks = client.get("/api/benchmarks").json()["benchmarks"]
    demo = next(b for b in benchmarks if b["name"] == "demo")
    assert demo["data_present"] is False
    assert demo["task"] == "vqa"


def test_benchmark_run_scores_and_streams(client, tmp_path):
    root = tmp_path / "bench_data"
    (root / "images").mkdir(parents=True)
    (root / "vqa.json").write_text(
        json.dumps(
            [
                {
                    "image_id": f"img{i}.png",
                    "question": "Is there water?",
                    "ground_truth": "yes" if i % 2 == 0 else "no",
                    "type": "presence",
                }
                for i in range(8)
            ]
        ),
        encoding="utf-8",
    )
    config = client.settings.bench_config_dir / "demo.yaml"
    config.write_text(
        f"name: demo_vqa\nadapter: vrsbench_vqa\ntask: vqa\n"
        f"root: {root.as_posix()}\nannotations: vqa.json\nimage_dir: images\n",
        encoding="utf-8",
    )

    run_id = client.post(
        "/api/benchmarks/run", json={"configs": [str(config)], "limit": 8}
    ).json()["run_id"]

    results = []
    with client.websocket_connect(f"/ws/runs/{run_id}") as socket:
        while True:
            event = socket.receive_json()
            if event["type"] == "benchmark_result":
                results.append(event)
            if event["type"] in {"complete", "error"}:
                break

    assert event["type"] == "complete", event
    assert results[0]["metrics"]["oa"] == pytest.approx(0.5)
    assert client.settings.results_csv.exists()


def test_multi_file_benchmark_is_not_reported_ready_when_files_are_absent(client):
    """RSVQA splits annotations across files named in `extra`; check them all.

    Checking only `annotations` reported RSVQA as ready with nothing downloaded,
    so the Benchmarks tab invited a run that then failed.
    """
    config = client.settings.bench_config_dir / "rsvqa.yaml"
    config.write_text(
        "\n".join(
            [
                "name: rsvqa_lr",
                "adapter: rsvqa",
                "task: vqa",
                "root: data/nowhere",
                "image_dir: LR/Images_LR",
                "extra:",
                "  questions: LR/q.json",
                "  answers: LR/a.json",
            ]
        ),
        encoding="utf-8",
    )
    benchmarks = client.get("/api/benchmarks").json()["benchmarks"]
    entry = next(b for b in benchmarks if b["name"] == "rsvqa_lr")
    assert entry["data_present"] is False


def test_benchmark_run_with_missing_config_is_404(client):
    response = client.post(
        "/api/benchmarks/run", json={"configs": ["nowhere/absent.yaml"]}
    )
    assert response.status_code == 404


def test_models_catalog_lists_echo(client):
    payload = client.get("/api/models").json()
    assert "catalog" in payload
    assert any(m["id"] == "echo" for m in payload["catalog"])
    assert payload["active"]["backend"] == "echo"


def test_results_endpoint_reads_csv(client, tmp_path):
    csv_path = client.settings.results_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "timestamp,benchmark,task,backend,model,num_samples,metric,value,"
        "duration_s,prompt_version,config_hash,git_sha\n"
        "2026-01-01T00:00:00+00:00,demo,vqa,echo,echo,10,oa,0.500000,"
        "1.00,1.0.0,abc,def\n",
        encoding="utf-8",
    )
    payload = client.get("/api/results").json()
    assert payload["total_rows"] == 1
    assert payload["models"][0]["scores"]["demo"] == pytest.approx(0.5)


def test_benchmark_run_accepts_model_list(client, tmp_path):
    root = tmp_path / "bench_data"
    (root / "images").mkdir(parents=True)
    (root / "vqa.json").write_text(
        json.dumps(
            [
                {
                    "image_id": "img0.png",
                    "question": "Is there water?",
                    "ground_truth": "yes",
                    "type": "presence",
                }
            ]
        ),
        encoding="utf-8",
    )
    config = client.settings.bench_config_dir / "tiny.yaml"
    config.write_text(
        f"name: tiny\nadapter: vrsbench_vqa\ntask: vqa\n"
        f"root: {root.as_posix()}\nannotations: vqa.json\nimage_dir: images\n",
        encoding="utf-8",
    )

    run_id = client.post(
        "/api/benchmarks/run",
        json={
            "configs": [str(config)],
            "limit": 1,
            "models": [{"backend": "echo", "model": "echo"}],
        },
    ).json()["run_id"]

    with client.websocket_connect(f"/ws/runs/{run_id}") as socket:
        while True:
            event = socket.receive_json()
            if event["type"] in {"complete", "error"}:
                break

    assert event["type"] == "complete"
    assert client.get("/api/results").json()["total_rows"] >= 1


# -- job store -------------------------------------------------------------


def test_job_store_evicts_oldest():
    store = JobStore(max_jobs=3)
    for _ in range(5):
        store.create("query")
    assert len(store.list()) == 3


def test_job_emit_survives_a_full_subscriber_queue():
    """A stalled websocket must not be able to block the run producing events."""
    import asyncio

    store = JobStore()
    job = store.create("query")
    queue = asyncio.Queue(maxsize=1)
    job._subscribers.add(queue)

    job.emit({"type": "step"})
    job.emit({"type": "step"})  # queue is full; must not raise

    assert len(job.events) == 2
