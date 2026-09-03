"""Backend runtime preflight.

Two failures were being conflated and both were reaching the user raw: a backend
whose Python runtime is not installed, and a model too large for the RAM on hand.
Both now refuse up front with something the operator can act on.
"""

import sys
import types

import pytest

from satquery.eval import backends
from satquery.eval.backends import (
    BackendConfig,
    BackendUnavailableError,
    build_backend,
    missing_modules,
    runtime_status,
)


def test_echo_needs_no_runtime():
    status = runtime_status("echo")
    assert status["available"] is True
    assert missing_modules("echo") == []


def test_missing_runtime_names_the_install_command(monkeypatch):
    """`No module named 'torch'` alone tells an operator nothing to do."""
    monkeypatch.setattr(backends, "missing_modules", lambda name: ["torch"])

    status = runtime_status("hf")
    assert status["available"] is False
    assert status["missing"] == ["torch"]
    assert 'pip install -e ".[hf]"' in status["detail"]
    assert "torch" in status["detail"]


def test_plural_reads_correctly(monkeypatch):
    monkeypatch.setattr(
        backends, "missing_modules", lambda name: ["torch", "transformers"]
    )
    assert "are not installed" in runtime_status("hf")["detail"]


def test_singular_reads_correctly(monkeypatch):
    monkeypatch.setattr(backends, "missing_modules", lambda name: ["vllm"])
    assert "is not installed" in runtime_status("vllm")["detail"]


def test_build_refuses_before_importing_a_missing_runtime(monkeypatch):
    monkeypatch.setattr(backends, "missing_modules", lambda name: ["torch"])
    with pytest.raises(BackendUnavailableError, match=r"pip install"):
        build_backend("hf", BackendConfig(model="whatever"))


def test_unknown_backend_lists_the_real_ones():
    with pytest.raises(KeyError, match="Available:"):
        build_backend("telepathy", BackendConfig(model="x"))


def test_echo_still_builds_without_any_extras():
    assert build_backend("echo").name == "echo"


def test_unknown_backend_runtime_status():
    status = runtime_status("telepathy")
    assert status["available"] is False
    assert "unknown backend" in status["detail"]


# -- host memory preflight -------------------------------------------------


def _install_fake(monkeypatch, name, module):
    monkeypatch.setitem(sys.modules, name, module)


def _fake_torch(cuda: bool):
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    return module


def _fake_psutil(available_bytes: int):
    module = types.ModuleType("psutil")
    module.virtual_memory = lambda: types.SimpleNamespace(available=available_bytes)
    return module


def _weights(tmp_path, gigabytes: float):
    path = tmp_path / "model"
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"\0" * int(gigabytes * 1e9))
    return path


def test_memory_check_skipped_on_gpu(monkeypatch, tmp_path):
    """VRAM is a different budget; this check is about host RAM only."""
    _install_fake(monkeypatch, "torch", _fake_torch(cuda=True))
    _install_fake(monkeypatch, "psutil", _fake_psutil(1))
    from satquery.eval.backends.hf import check_host_memory

    assert check_host_memory(str(_weights(tmp_path, 0.01))) is None


def test_memory_check_skipped_without_psutil(monkeypatch, tmp_path):
    """A missing optional dependency must not block a load that would work."""
    _install_fake(monkeypatch, "torch", _fake_torch(cuda=False))
    monkeypatch.setitem(sys.modules, "psutil", None)
    from satquery.eval.backends.hf import check_host_memory

    assert check_host_memory(str(_weights(tmp_path, 0.01))) is None


def test_memory_check_passes_when_it_fits(monkeypatch, tmp_path):
    _install_fake(monkeypatch, "torch", _fake_torch(cuda=False))
    _install_fake(monkeypatch, "psutil", _fake_psutil(8_000_000_000))
    from satquery.eval.backends.hf import check_host_memory

    assert check_host_memory(str(_weights(tmp_path, 0.02))) is None


def test_memory_check_refuses_when_it_does_not_fit(monkeypatch, tmp_path):
    _install_fake(monkeypatch, "torch", _fake_torch(cuda=False))
    _install_fake(monkeypatch, "psutil", _fake_psutil(200_000_000))
    from satquery.eval.backends.hf import check_host_memory

    message = check_host_memory(str(_weights(tmp_path, 1.0)))
    assert message is not None
    assert "GB of RAM" in message
    assert "GPU host" in message


def test_remote_repo_id_is_not_size_checked(monkeypatch):
    """Only a local directory has a size we can measure up front."""
    _install_fake(monkeypatch, "torch", _fake_torch(cuda=False))
    _install_fake(monkeypatch, "psutil", _fake_psutil(1))
    from satquery.eval.backends.hf import check_host_memory

    assert check_host_memory("Qwen/Qwen2.5-VL-3B-Instruct") is None
