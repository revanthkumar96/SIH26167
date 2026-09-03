"""Benchmark evaluation harness.

Offline and independent of the API and controller: it imports the same prompts and
backends the serving path uses, but nothing in it depends on a running system.
"""

from satquery.eval.datasets import BenchmarkConfig, load_benchmark
from satquery.eval.runner import EvalResult, run_benchmark

__all__ = ["BenchmarkConfig", "EvalResult", "load_benchmark", "run_benchmark"]
