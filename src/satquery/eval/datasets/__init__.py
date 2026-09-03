"""Benchmark adapters.

Importing this package registers every adapter, so ``get_adapter`` resolves by name
without the caller knowing which module defines it.
"""

from satquery.eval.datasets import cdvqa, rsvqa, vrsbench  # noqa: F401
from satquery.eval.datasets.base import (
    BenchmarkConfig,
    BenchmarkDataset,
    available_adapters,
    get_adapter,
    register,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkDataset",
    "available_adapters",
    "get_adapter",
    "register",
]


def load_benchmark(config: BenchmarkConfig) -> BenchmarkDataset:
    """Instantiate the adapter named by a config."""
    return get_adapter(config.adapter)(config)
