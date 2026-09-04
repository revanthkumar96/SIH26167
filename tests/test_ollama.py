"""Ollama backend: host resolution, presence checks and error surfacing.

Network is stubbed. What matters is that the two failures an operator actually
hits -- no server, and a model that is not pulled -- are named precisely, and
that Ollama's own diagnosis is not thrown away behind a bare status code.
"""

import pytest

from satquery.eval.backends import ollama


class _Response:
    def __init__(self, payload=None, status=200, text=""):
        self._payload = payload or {}
        self.status_code = status
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError("should not be called; body is read instead")


def test_host_defaults_and_normalises(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert ollama.host_url() == "http://127.0.0.1:11434"

    # The CLI accepts a bare host:port, so we accept it too.
    monkeypatch.setenv("OLLAMA_HOST", "myhost:11434")
    assert ollama.host_url() == "http://myhost:11434"

    monkeypatch.setenv("OLLAMA_HOST", "https://gpu.box:443/")
    assert ollama.host_url() == "https://gpu.box:443"


def test_blank_host_falls_back(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "   ")
    assert ollama.host_url() == "http://127.0.0.1:11434"


def test_latest_tag_is_matched_when_omitted(monkeypatch):
    monkeypatch.setattr(ollama, "installed_models", lambda *a, **k: ["qwen3-vl:latest"])
    assert ollama.model_present("qwen3-vl")
    assert not ollama.model_present("qwen3-vl:2b-instruct")


def test_exact_tag_match(monkeypatch):
    monkeypatch.setattr(
        ollama, "installed_models", lambda *a, **k: ["qwen3-vl:2b-instruct"]
    )
    assert ollama.model_present("qwen3-vl:2b-instruct")


def test_unreachable_server_is_reported_not_raised(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("requests.get", boom)
    reachable, detail = ollama.server_reachable()
    assert reachable is False
    assert "connection refused" in detail


def test_construction_refuses_without_a_server(monkeypatch):
    monkeypatch.setattr(ollama, "server_reachable", lambda *a, **k: (False, "refused"))
    from satquery.eval.backends.base import BackendConfig

    with pytest.raises(RuntimeError, match="ollama serve"):
        ollama.OllamaBackend(BackendConfig(model="qwen3-vl:2b-instruct"))


def test_construction_refuses_when_the_model_is_not_pulled(monkeypatch):
    monkeypatch.setattr(ollama, "server_reachable", lambda *a, **k: (True, "0.33.2"))
    monkeypatch.setattr(ollama, "installed_models", lambda *a, **k: ["llama3.2:1b"])
    from satquery.eval.backends.base import BackendConfig

    with pytest.raises(RuntimeError, match="ollama pull qwen3-vl:2b-instruct"):
        ollama.OllamaBackend(BackendConfig(model="qwen3-vl:2b-instruct"))


# -- error surfacing -------------------------------------------------------


def test_nested_error_body_is_unwrapped():
    """Ollama nests a JSON document inside the error string; unwrap one level.

    This is the message a model without a vision projector actually returns,
    and it is the difference between a usable report and "500 Server Error".
    """
    body = (
        '{"error":"{\\"error\\":{\\"code\\":500,\\"message\\":\\"image input is '
        "not supported - hint: if this is unexpected, you may need to provide "
        'the mmproj\\",\\"type\\":\\"server_error\\"}}"}'
    )
    assert "image input is not supported" in ollama._error_message(body)


def test_flat_error_body_is_read():
    assert ollama._error_message('{"error":"model not found"}') == "model not found"


def test_non_json_body_is_passed_through():
    assert "gateway" in ollama._error_message("502 bad gateway")


def test_generate_raises_with_the_server_message(monkeypatch, tmp_path):
    from PIL import Image

    from satquery.eval.backends.base import BackendConfig
    from satquery.schema import GenerationRequest

    monkeypatch.setattr(ollama, "server_reachable", lambda *a, **k: (True, "0.33.2"))
    monkeypatch.setattr(ollama, "installed_models", lambda *a, **k: ["m:1b"])

    backend = ollama.OllamaBackend(BackendConfig(model="m:1b"))
    monkeypatch.setattr(
        backend._session,
        "post",
        lambda *a, **k: _Response(status=500, text='{"error":"no vision projector"}'),
    )

    scene = tmp_path / "s.png"
    Image.new("RGB", (32, 32)).save(scene)

    with pytest.raises(RuntimeError, match="no vision projector"):
        backend.generate([GenerationRequest("t", "hi", (scene,), 16)])


def test_paired_images_go_in_one_message(monkeypatch, tmp_path):
    """The paired tasks need both images in a single turn."""
    from PIL import Image

    from satquery.eval.backends.base import BackendConfig
    from satquery.schema import GenerationRequest

    monkeypatch.setattr(ollama, "server_reachable", lambda *a, **k: (True, "0.33.2"))
    monkeypatch.setattr(ollama, "installed_models", lambda *a, **k: ["m:1b"])
    backend = ollama.OllamaBackend(BackendConfig(model="m:1b"))

    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(json)
        return _Response({"message": {"content": "ok"}})

    monkeypatch.setattr(backend._session, "post", fake_post)

    paths = []
    for name in ("a.png", "b.png"):
        path = tmp_path / name
        Image.new("RGB", (32, 32)).save(path)
        paths.append(path)

    backend.generate([GenerationRequest("t", "compare", tuple(paths), 16)])

    messages = captured["messages"]
    assert len(messages) == 1
    assert len(messages[0]["images"]) == 2
    assert captured["stream"] is False
    assert captured["options"]["num_predict"] == 16
