"""Backend factory.

Imports are deferred so that installing torch or vLLM is only required for the
backend actually being used -- the echo backend must stay importable in CI.

Availability is checked *before* construction rather than letting an import blow
up mid-run. A benchmark sweep that dies on ``No module named 'torch'`` tells the
operator nothing about what to do next; the error here names the install command.
"""

from __future__ import annotations

from importlib.util import find_spec

from satquery.eval.backends.base import BackendConfig, VLMBackend

#: "echo" is a test double, not a product surface. It stays importable so CI
#: and the unit suite can exercise the whole pipeline with no weights, but it
#: is absent from the model catalog and is never offered in the UI.
BACKENDS = ("ollama", "hf", "vllm", "echo")

#: Modules each backend needs, and how to get them.
RUNTIME_REQUIREMENTS: dict[str, tuple[tuple[str, ...], str]] = {
    "echo": ((), ""),
    # Ollama is a service, not a library: the import is trivial, the server
    # is the real dependency. Reachability is checked separately.
    "ollama": (("requests",), "install Ollama from https://ollama.com"),
    # torchvision is not optional: Qwen2.5-VL's processor imports it eagerly.
    "hf": (("torch", "torchvision", "transformers"), 'pip install -e ".[hf]"'),
    "vllm": (("vllm",), 'pip install -e ".[vllm]"'),
}


class BackendUnavailableError(RuntimeError):
    """The backend's runtime is not installed in this interpreter."""


def missing_modules(name: str) -> list[str]:
    """Required modules that are not importable, without importing them."""
    required, _ = RUNTIME_REQUIREMENTS.get(name, ((), ""))
    missing: list[str] = []
    for module in required:
        try:
            if find_spec(module) is None:
                missing.append(module)
        except (ImportError, ValueError):
            missing.append(module)
    return missing


def runtime_status(name: str) -> dict[str, object]:
    """Whether a backend can run here, and what is missing if not."""
    if name not in RUNTIME_REQUIREMENTS:
        return {
            "backend": name,
            "available": False,
            "detail": f"unknown backend '{name}'",
        }

    missing = missing_modules(name)
    _, install = RUNTIME_REQUIREMENTS[name]
    if not missing and name == "ollama":
        # A library check is not enough here. The dependency that actually fails
        # is a server that is not running, and saying "runtime installed" while
        # every request errors would be worse than saying nothing.
        from satquery.eval.backends.ollama import host_url, server_reachable

        reachable, detail = server_reachable()
        if reachable:
            return {
                "backend": name,
                "available": True,
                "detail": f"Ollama {detail} at {host_url()}",
            }
        return {
            "backend": name,
            "available": False,
            "missing": ["ollama server"],
            "install": "ollama serve",
            "detail": (
                f"no Ollama server at {host_url()}. Start it with 'ollama serve', "
                f"or set OLLAMA_HOST."
            ),
        }
    if not missing:
        return {"backend": name, "available": True, "detail": "runtime installed"}
    return {
        "backend": name,
        "available": False,
        "missing": missing,
        "install": install,
        "detail": (
            f"{name} backend needs {', '.join(missing)}, which "
            f"{'is' if len(missing) == 1 else 'are'} not installed. "
            f"Install with: {install}"
        ),
    }


def require_runtime(name: str) -> None:
    """Raise with an actionable message when a backend cannot run here."""
    status = runtime_status(name)
    if not status["available"]:
        raise BackendUnavailableError(str(status["detail"]))


def build_backend(name: str, config: BackendConfig | None = None) -> VLMBackend:
    """Instantiate a backend by name."""
    if name not in BACKENDS:
        raise KeyError(f"unknown backend '{name}'. Available: {', '.join(BACKENDS)}")

    require_runtime(name)

    if name == "echo":
        from satquery.eval.backends.echo import EchoBackend

        return EchoBackend(config)

    if config is None:
        raise ValueError(f"backend '{name}' requires a BackendConfig")

    if name == "ollama":
        from satquery.eval.backends.ollama import OllamaBackend

        return OllamaBackend(config)

    if name == "hf":
        from satquery.eval.backends.hf import HFBackend

        return HFBackend(config)

    from satquery.eval.backends.vllm_backend import VLLMBackend

    return VLLMBackend(config)


__all__ = [
    "BACKENDS",
    "RUNTIME_REQUIREMENTS",
    "BackendConfig",
    "BackendUnavailableError",
    "VLMBackend",
    "build_backend",
    "missing_modules",
    "require_runtime",
    "runtime_status",
]
