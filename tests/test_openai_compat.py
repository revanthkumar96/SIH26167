"""OpenAI-compatible backend: endpoint resolution, preflight and error surfacing.

Network is stubbed. What matters is that the failures an operator actually hits
on the AWS path -- no server, a misspelled adapter name, a context overflow --
are named precisely rather than surfacing as a bare status code, and that the
prompt sent is the one ``build_messages()`` produced rather than a reformatted
copy. A prompt-format difference between the backend that measured the baseline
and the one that measures the adapted model would silently invalidate the
comparison the whole programme exists to produce.
"""

from __future__ import annotations

import pytest

from satquery.eval.backends import BACKENDS, openai_compat, runtime_status
from satquery.eval.backends.base import BackendConfig
from satquery.schema import GenerationRequest


class _Response:
    def __init__(self, payload=None, status=200, text=""):
        self._payload = payload or {}
        self.status_code = status
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")


def _completion(content="a lake", finish="stop", tokens=7):
    return {
        "choices": [
            {"message": {"content": content}, "finish_reason": finish},
        ],
        "usage": {"completion_tokens": tokens},
    }


@pytest.fixture
def served(monkeypatch):
    """A reachable server offering the base model and one adapter."""
    monkeypatch.setattr(
        openai_compat,
        "server_reachable",
        lambda *a, **k: (True, "qwen3-vl-base, qwen3-vl-satquery"),
    )
    monkeypatch.setattr(
        openai_compat,
        "served_models",
        lambda *a, **k: ["qwen3-vl-base", "qwen3-vl-satquery"],
    )


# -- endpoint resolution -------------------------------------------------


def test_base_url_defaults(monkeypatch):
    monkeypatch.delenv("SATQUERY_VLM_BASE_URL", raising=False)
    assert openai_compat.base_url() == "http://127.0.0.1:8000/v1"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://box:8000/v1", "http://box:8000/v1"),
        # The /v1 suffix is easy to forget, and omitting it 404s every request.
        ("http://box:8000", "http://box:8000/v1"),
        ("http://box:8000/", "http://box:8000/v1"),
        ("box:8000", "http://box:8000/v1"),
        ("https://gpu.example:443/v1/", "https://gpu.example:443/v1"),
        ("   ", "http://127.0.0.1:8000/v1"),
    ],
)
def test_base_url_normalises(monkeypatch, value, expected):
    monkeypatch.setenv("SATQUERY_VLM_BASE_URL", value)
    assert openai_compat.base_url() == expected


def test_api_key_falls_back_to_the_vllm_placeholder(monkeypatch):
    """vLLM started without --api-key accepts any bearer token, but needs one."""
    monkeypatch.delenv("SATQUERY_VLM_API_KEY", raising=False)
    assert openai_compat.api_key() == "EMPTY"

    monkeypatch.setenv("SATQUERY_VLM_API_KEY", "secret")
    assert openai_compat.api_key() == "secret"


def test_timeout_override_rejects_nonsense(monkeypatch):
    monkeypatch.setenv("SATQUERY_VLM_TIMEOUT", "900")
    assert openai_compat.request_timeout() == 900.0

    for bad in ("nonsense", "-1", "0", ""):
        monkeypatch.setenv("SATQUERY_VLM_TIMEOUT", bad)
        assert openai_compat.request_timeout() == float(
            openai_compat.DEFAULT_REQUEST_TIMEOUT
        )


# -- preflight -----------------------------------------------------------


def test_registered_as_a_backend():
    assert "openai_compat" in BACKENDS


def test_unreachable_server_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(
        "requests.get", lambda *a, **k: (_ for _ in ()).throw(OSError("refused"))
    )
    reachable, detail = openai_compat.server_reachable()
    assert reachable is False
    assert "refused" in detail


def test_construction_refuses_without_a_server(monkeypatch):
    monkeypatch.setattr(
        openai_compat, "server_reachable", lambda *a, **k: (False, "refused")
    )
    with pytest.raises(RuntimeError, match="SATQUERY_VLM_BASE_URL"):
        openai_compat.OpenAICompatBackend(BackendConfig(model="qwen3-vl-base"))


def test_unknown_adapter_fails_at_construction_not_mid_sweep(served):
    """A misspelled adapter must not waste the GPU hours already spent.

    vLLM exposes each --lora-modules entry as its own model name, so the check
    is the same one that catches a typo in the base model.
    """
    with pytest.raises(RuntimeError, match="lora-modules"):
        openai_compat.OpenAICompatBackend(BackendConfig(model="qwen3-vl-satqeury"))


def test_runtime_status_reports_a_missing_server(monkeypatch):
    monkeypatch.setattr(
        openai_compat, "server_reachable", lambda *a, **k: (False, "refused")
    )
    status = runtime_status("openai_compat")
    assert status["available"] is False
    assert "SATQUERY_VLM_BASE_URL" in str(status["detail"])


def test_runtime_status_reports_what_is_served(monkeypatch):
    monkeypatch.setattr(
        openai_compat, "server_reachable", lambda *a, **k: (True, "qwen3-vl-satquery")
    )
    status = runtime_status("openai_compat")
    assert status["available"] is True
    assert "qwen3-vl-satquery" in str(status["detail"])


# -- generation ----------------------------------------------------------


def test_generate_sends_openai_messages_and_returns_text(served, monkeypatch, tmp_path):
    """The payload carries build_messages() output unmodified."""
    image = tmp_path / "scene.png"
    from PIL import Image

    Image.new("RGB", (8, 8), "green").save(image)

    captured: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        captured["headers"] = headers
        return _Response(_completion("a lake", tokens=3))

    backend = openai_compat.OpenAICompatBackend(BackendConfig(model="qwen3-vl-base"))
    monkeypatch.setattr(backend._session, "post", fake_post)

    request = GenerationRequest(
        sample_id="s1", prompt="What is this?", images=(image,), max_new_tokens=32
    )
    (text, meta) = backend.generate_with_meta([request])[0]

    assert text == "a lake"
    assert meta == {"finish_reason": "stop", "tokens": 3}
    assert captured["url"].endswith("/chat/completions")
    assert captured["payload"]["model"] == "qwen3-vl-base"
    assert captured["payload"]["max_tokens"] == 32

    # One user turn, image parts before the text part, image inlined as a data
    # URL -- the shape build_messages() emits for every other backend.
    content = captured["payload"]["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1]["text"] == "What is this?"
    assert captured["headers"]["Authorization"].startswith("Bearer ")


def test_truncation_is_surfaced_rather_than_hidden(served, monkeypatch, tmp_path):
    """A reply cut off at the budget is a defect the trace reports."""
    from PIL import Image

    image = tmp_path / "s.png"
    Image.new("RGB", (8, 8), "blue").save(image)

    backend = openai_compat.OpenAICompatBackend(BackendConfig(model="qwen3-vl-base"))
    monkeypatch.setattr(
        backend._session,
        "post",
        lambda *a, **k: _Response(_completion("a partial sen", finish="length")),
    )

    _, meta = backend.generate_with_meta(
        [GenerationRequest("s", "q", (image,), max_new_tokens=4)]
    )[0]
    assert meta["finish_reason"] == "length"


def test_server_error_body_is_surfaced(served, monkeypatch, tmp_path):
    """A context overflow names the limit; a bare status code would not."""
    import json as json_module

    from PIL import Image

    image = tmp_path / "s.png"
    Image.new("RGB", (8, 8), "red").save(image)

    body = json_module.dumps(
        {"error": {"message": "This model's maximum context length is 8192 tokens"}}
    )
    backend = openai_compat.OpenAICompatBackend(BackendConfig(model="qwen3-vl-base"))
    monkeypatch.setattr(
        backend._session, "post", lambda *a, **k: _Response(status=400, text=body)
    )

    with pytest.raises(RuntimeError, match="maximum context length is 8192"):
        backend.generate([GenerationRequest("s", "q", (image,), max_new_tokens=8)])


def test_empty_choices_raises_rather_than_returning_blank(
    served, monkeypatch, tmp_path
):
    from PIL import Image

    image = tmp_path / "s.png"
    Image.new("RGB", (8, 8), "red").save(image)

    backend = openai_compat.OpenAICompatBackend(BackendConfig(model="qwen3-vl-base"))
    monkeypatch.setattr(
        backend._session, "post", lambda *a, **k: _Response({"choices": []})
    )

    with pytest.raises(RuntimeError, match="no choices"):
        backend.generate([GenerationRequest("s", "q", (image,), max_new_tokens=8)])
