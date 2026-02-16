"""
src/benchmarks/memory_bench.py

Memory profiling benchmark for Deep CFR buffers and workers.
Measures reservoir buffer memory usage and estimates worker memory footprint.
"""

import logging
import multiprocessing
import os
from typing import List, Optional

import numpy as np
import psutil

from ..encoding import INPUT_DIM, NUM_ACTIONS
from ..reservoir import ReservoirBuffer, ReservoirSample
from ..config import load_config
from ..networks import AdvantageNetwork
from ..cfr.deep_worker import run_deep_cfr_worker
from .runner import BenchmarkResult

logger = logging.getLogger(__name__)


def _get_process_memory_mb() -> float:
    """Get current process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def _worker_memory_test(worker_args):
    """Run a single traversal and measure memory."""
    result = run_deep_cfr_worker(worker_args)
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_memory(
    buffer_capacities: Optional[List[int]] = None,
    worker_counts: Optional[List[int]] = None,
    config_path: Optional[str] = None,
    device: str = "cpu",
) -> BenchmarkResult:
    """
    Profile memory usage for buffers and workers.

    Part A: Measures memory usage of ReservoirBuffer at different capacities
    Part B: Estimates per-worker memory and max safe worker count

    Args:
        buffer_capacities: List of buffer sizes to test
        worker_counts: List of worker counts for estimation
        config_path: Path to config file
        device: Device for network initialization

    Returns:
        BenchmarkResult with memory profiling data
    """
    buffer_capacities = buffer_capacities or [100_000, 500_000, 1_000_000, 2_000_000]
    worker_counts = worker_counts or [1, 4, 8, 16, 23, 32]
    config_path = config_path or "/workspace/config/parallel.config.yaml"

    logger.info("Starting memory benchmark")

    # --- Part A: Buffer Memory Profiling ---
    logger.info("Part A: Profiling reservoir buffer memory usage")
    buffer_memory_results = []

    for capacity in buffer_capacities:
        logger.info("Testing buffer capacity: %d", capacity)

        # Measure baseline memory
        import gc
        gc.collect()
        baseline_mb = _get_process_memory_mb()

        # Create and fill buffer using batch generation for speed
        buffer = ReservoirBuffer(capacity=capacity)

        # Generate all samples in batches for efficiency
        batch_gen_size = min(50_000, capacity)
        filled = 0
        while filled < capacity:
            gen_count = min(batch_gen_size, capacity - filled)
            features_batch = np.random.randn(gen_count, INPUT_DIM).astype(np.float32)
            targets_batch = np.random.randn(gen_count, NUM_ACTIONS).astype(np.float32)
            masks_batch = np.random.randint(0, 2, (gen_count, NUM_ACTIONS)).astype(bool)

            for j in range(gen_count):
                sample = ReservoirSample(
                    features=features_batch[j],
                    target=targets_batch[j],
                    action_mask=masks_batch[j],
                    iteration=filled + j,
                )
                buffer.add(sample)

            filled += gen_count
            if filled % 200_000 == 0:
                logger.info("  Filled %d/%d samples", filled, capacity)

        # Measure memory after filling
        filled_mb = _get_process_memory_mb()
        delta_mb = filled_mb - baseline_mb

        buffer_memory_results.append(
            {
                "capacity": capacity,
                "memory_mb": delta_mb,
                "mb_per_sample": delta_mb / capacity if capacity > 0 else 0,
            }
        )

        logger.info(
            "  Capacity %d: %.2f MB (%.4f MB/sample)",
            capacity,
            delta_mb,
            delta_mb / capacity if capacity > 0 else 0,
        )

        # Clean up
        del buffer
        gc.collect()

    # --- Part B: Worker Memory Estimation ---
    logger.info("Part B: Estimating worker memory usage")

    config = load_config(config_path)
    if not config:
        raise RuntimeError(f"Failed to load config from {config_path}")

    # Create dummy network weights
    net = AdvantageNetwork(input_dim=INPUT_DIM, hidden_dim=256, output_dim=NUM_ACTIONS)
    network_weights = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
    network_config = {
        "input_dim": INPUT_DIM,
        "hidden_dim": 256,
        "output_dim": NUM_ACTIONS,
    }

    # Spawn a worker and measure its memory
    worker_args = (
        0,  # iteration
        config,
        network_weights,
        network_config,
        None,  # progress_queue
        None,  # archive_queue
        0,  # worker_id
        "/tmp/bench_logs",
        "bench",
    )

    logger.info("Spawning test worker to measure memory...")
    with multiprocessing.Pool(processes=1) as pool:
        worker_mem_result = pool.apply(_worker_memory_test, (worker_args,))

    worker_memory_mb = worker_mem_result
    logger.info("Worker memory: %.2f MB", worker_memory_mb)

    # Estimate max safe workers for 31GB system (leaving 8GB for OS/buffers)
    available_memory_gb = 31 - 8  # 23 GB for workers
    available_memory_mb = available_memory_gb * 1024
    max_safe_workers = int(available_memory_mb / worker_memory_mb)

    logger.info(
        "Estimated max safe workers for 31GB system: %d (%.2f MB per worker)",
        max_safe_workers,
        worker_memory_mb,
    )

    # Build worker memory projections
    worker_memory_projections = []
    for worker_count in worker_counts:
        projected_memory_mb = worker_count * worker_memory_mb
        projected_memory_gb = projected_memory_mb / 1024
        worker_memory_projections.append(
            {
                "workers": worker_count,
                "memory_mb": projected_memory_mb,
                "memory_gb": projected_memory_gb,
            }
        )

    # Assemble results
    metrics = {
        "buffer_memory": buffer_memory_results,
        "worker_memory_mb": worker_memory_mb,
        "max_safe_workers_31gb": max_safe_workers,
        "worker_memory_projections": worker_memory_projections,
    }

    logger.info("Memory benchmark complete")

    return BenchmarkResult(
        name="memory_profiling",
        config={
            "buffer_capacities": buffer_capacities,
            "worker_counts": worker_counts,
            "config_path": config_path,
        },
        timings={},
        metrics=metrics,
        metadata={
            "device": device,
            "system_memory_gb": psutil.virtual_memory().total / (1024**3),
        },
    )
