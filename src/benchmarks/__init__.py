"""
src/benchmarks/__init__.py

Benchmarking framework for Deep CFR training performance analysis.

Provides:
- BenchmarkResult: Data structure for benchmark results
- BenchmarkSuite: Framework for running and organizing benchmarks
- network_bench: Neural network performance benchmarks
- traversal_bench: Single-worker traversal benchmarks
- worker_scaling: Worker scaling benchmarks
- memory_bench: Memory profiling benchmarks
- e2e_bench: End-to-end training step benchmarks
- reporting: Result formatting and visualization
"""

from .runner import BenchmarkResult, BenchmarkSuite
from . import network_bench
from . import traversal_bench
from . import worker_scaling
from . import memory_bench
from . import e2e_bench
from . import reporting

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "network_bench",
    "traversal_bench",
    "worker_scaling",
    "memory_bench",
    "e2e_bench",
    "reporting",
]
