"""Weight presence, download orchestration and the startup preload.

No network: the hub call is stubbed, because what matters here is that the server
behaves correctly around the download, not that Hugging Face works.
"""

import pytest
from fastapi.testclient import TestClient

from satquery import models
from satquery.api.app import create_app
from satquery.api.settings import Settings
from satquery.models import DownloadProgress, ensure_model, needs_model


def test_only_model_backends_need_weights():
    assert needs_model("hf")
    assert needs_model("vllm")
    assert not needs_model("echo")


def test_progress_percent_is_none_without_a_total():
    assert DownloadProgress().percent is None
    assert DownloadProgress(downloaded_bytes=50, total_bytes=200).percent == 25.0


def test_local_directory_counts_as_present(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    status = ensure_model(str(tmp_path))
    assert status.state == "ready"
    assert status.detail == "already present"
    assert status.path == str(tmp_path)


def test_cached_model_is_not_downloaded(monkeypatch, tmp_path):
    """A model already on disk must never trigger a fetch."""
    import huggingface_hub

    monkeypatch.setattr(models, "cached_path", lambda *a, **k: tmp_path)

    def explode(*args, **kwargs):
        raise AssertionError("should not download a model already on disk")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", explode)

    status = ensure_model("some/model")
    assert status.state == "ready"
    assert status.path == str(tmp_path)


def test_download_disabled_reports_why(monkeypatch):
    monkeypatch.setattr(models, "cached_path", lambda *a, **k: None)
    status = ensure_model("some/model", allow_download=False)
    assert status.state == "error"
    assert "downloads are disabled" in status.detail


def test_offline_mode_points_at_the_pull_command(monkeypatch):
    monkeypatch.setattr(models, "cached_path", lambda *a, **k: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    status = ensure_model("some/model")
    assert status.state == "error"
    assert "satquery models pull" in status.detail


def test_download_failure_is_returned_not_raised(monkeypatch):
    """A failed fetch must leave the server up and able to explain itself."""
    monkeypatch.setattr(models, "cached_path", lambda *a, **k: None)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    import huggingface_hub

    def boom(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)

    status = ensure_model("some/model")
    assert status.state == "error"
    assert "network unreachable" in status.detail


def test_progress_callback_sees_each_state(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "cached_path", lambda *a, **k: None)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path)
    )

    seen: list[str] = []
    status = ensure_model("some/model", on_update=lambda p: seen.append(p.state))

    assert status.state == "ready"
    assert seen[0] == "checking"
    assert "downloading" in seen


# -- server integration ----------------------------------------------------


def _client(tmp_path, **overrides):
    options = {
        "backend": "echo",
        "model": "echo",
        "workspace": tmp_path / "workspace",
        "bench_config_dir": tmp_path / "bench",
        **overrides,
    }
    settings = Settings(**options)
    settings.bench_config_dir.mkdir(parents=True, exist_ok=True)
    return settings


def test_echo_backend_skips_the_fetch_entirely(tmp_path):
    with TestClient(create_app(_client(tmp_path))) as client:
        model = client.get("/api/model").json()
    assert model["state"] == "skipped"
    assert "needs no weights" in model["detail"]


def test_startup_fetches_weights_for_a_model_backend(tmp_path, monkeypatch):
    """The whole point: weights arrive at startup, not on the first query."""
    calls: list[str] = []

    def fake_ensure(
        model, revision=None, on_update=None, allow_download=True, models_dir=None
    ):
        calls.append(model)
        progress = DownloadProgress(
            state="ready", model=model, detail="downloaded", path="/fake"
        )
        if on_update:
            on_update(progress)
        return progress

    monkeypatch.setattr("satquery.api.app.ensure_model", fake_ensure)
    # Constructing a real backend would import torch; the fetch is what is under
    # test, so the backend build is allowed to fail behind its suppression.
    monkeypatch.setattr(
        "satquery.api.app.build_backend",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no torch here")),
    )

    settings = _client(tmp_path, backend="hf", model="some/model")
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health").json()

    assert calls == ["some/model"]
    assert health["model"]["state"] == "ready"
    assert health["settings"]["preload"] is True


def test_preload_can_be_disabled(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        "satquery.api.app.ensure_model",
        lambda model, *a, **k: calls.append(model) or DownloadProgress(state="ready"),
    )

    settings = _client(tmp_path, backend="hf", model="some/model", preload=False)
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/model").json()["state"] == "skipped"
    assert calls == []


def test_query_reports_503_when_weights_are_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "satquery.api.app.ensure_model",
        lambda *a, **k: DownloadProgress(
            state="error", model="some/model", detail="network unreachable"
        ),
    )

    settings = _client(tmp_path, backend="hf", model="some/model")
    with TestClient(create_app(settings)) as client:
        from PIL import Image

        path = tmp_path / "scene.png"
        Image.new("RGB", (32, 32)).save(path)
        upload = client.post(
            "/api/upload",
            files=[("files", ("scene.png", path.read_bytes(), "image/png"))],
        )
        image_id = upload.json()["images"][0]["id"]

        response = client.post(
            "/api/query", json={"query": "test", "image_ids": [image_id]}
        )

    assert response.status_code == 503
    assert "network unreachable" in response.json()["detail"]


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_flag_parsing_accepts_common_spellings(value, monkeypatch):
    monkeypatch.setenv("SATQUERY_PRELOAD", value)
    assert Settings().preload is True


def test_flag_parsing_rejects_other_values(monkeypatch):
    monkeypatch.setenv("SATQUERY_PRELOAD", "0")
    assert Settings().preload is False
