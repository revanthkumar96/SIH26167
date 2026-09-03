"""Backend factory.

Imports are deferred so that installing torch or vLLM is only required for the
backend actually being used -- the echo backend must stay importable in CI.
"""

from __future__ import annotations

from satquery.eval.backends.base import BackendConfig, VLMBackend

BACKENDS = ("echo", "hf", "vllm")


def build_backend(name: str, config: BackendConfig | None = None) -> VLMBackend:
    """Instantiate a backend by name."""
    if name == "echo":
        from satquery.eval.backends.echo import EchoBackend

        return EchoBackend(config)

    if config is None:
        raise ValueError(f"backend '{name}' requires a BackendConfig")

    if name == "hf":
        from satquery.eval.backends.hf import HFBackend

        return HFBackend(config)

    if name == "vllm":
        from satquery.eval.backends.vllm_backend import VLLMBackend

        return VLLMBackend(config)

    raise KeyError(f"unknown backend '{name}'. Available: {', '.join(BACKENDS)}")


__all__ = ["BACKENDS", "BackendConfig", "VLMBackend", "build_backend"]
