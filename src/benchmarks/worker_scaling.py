"""
src/benchmarks/worker_scaling.py

Worker scaling benchmark for Deep CFR parallel traversals.
Measures throughput and efficiency as worker count increases.
"""

import logging
import multiprocessing
import time
from typing import List, Optional

import numpy as np

from ..config import load_config
from ..encoding import INPUT_DIM, NUM_ACTIONS
from ..networks import AdvantageNetwork
from ..cfr.deep_worker import run_deep_cfr_worker
from .runner import BenchmarkResult

logger = logging.getLogger(__name__)


def benchmark_worker_scaling(
    worker_counts: Optional[List[int]] = None,
    traversals_per_test: int = 100,
    config_path: Optional[str] = None,
    device: str = "cpu",
) -> BenchmarkResult:
    """
    Measure throughput scaling with worker count.

    For each worker count, runs a fixed number of traversals in parallel and
    measures total throughput and efficiency relative to linear scaling.

    Args:
        worker_counts: List of worker counts to test (defaults to [1,2,4,8,16,23,32])
        traversals_per_test: Number of traversals to run for each test
        config_path: Path to config file
        device: Device for network initialization

    Returns:
        BenchmarkResult with scaling metrics
    """
    worker_counts = worker_counts or [1, 2, 4, 8, 16, 23, 32]
    config_path = config_path or "/workspace/config/parallel.config.yaml"

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

    logger.info(
        "Starting worker scaling benchmark: %s workers, %d traversals each",
        worker_counts,
        traversals_per_test,
    )

    scaling_results = []
    baseline_throughput = None

    for worker_count in worker_counts:
        logger.info("Testing with %d workers...", worker_count)

        # Prepare worker args for all traversals
        worker_args_list = []
        for i in range(traversals_per_test):
            worker_args_list.append(
                (
                    i,  # iteration
                    config,
                    network_weights,
                    network_config,
                    None,  # progress_queue
                    None,  # archive_queue
                    i % worker_count,  # worker_id
                    "/tmp/bench_logs",
                    "bench",
                )
            )

        start_time = time.time()

        if worker_count == 1:
            # Sequential execution
            results = []
            for args in worker_args_list:
                result = run_deep_cfr_worker(args)
                results.append(result)
        else:
            # Parallel execution
            with multiprocessing.Pool(processes=worker_count) as pool:
                results = pool.map(run_deep_cfr_worker, worker_args_list)

        wall_time = time.time() - start_time

        # Calculate metrics
        throughput = traversals_per_test / wall_time
        total_nodes = sum(r.stats.nodes_visited for r in results if r)

        if baseline_throughput is None:
            baseline_throughput = throughput
            efficiency = 1.0
        else:
            # Efficiency = actual speedup / ideal speedup
            ideal_speedup = worker_count
            actual_speedup = throughput / baseline_throughput
            efficiency = actual_speedup / ideal_speedup

        scaling_results.append(
            {
                "workers": worker_count,
                "wall_time": wall_time,
                "throughput": throughput,
                "efficiency": efficiency,
                "total_nodes": total_nodes,
            }
        )

        logger.info(
            "  %d workers: %.2f trav/s (%.1f%% efficiency)",
            worker_count,
            throughput,
            efficiency * 100,
        )

    # Build metrics dict
    metrics = {
        "baseline_throughput": baseline_throughput,
        "scaling_results": scaling_results,
    }

    # Add per-worker-count breakdowns
    timings = {}
    for result in scaling_results:
        wc = result["workers"]
        timings[f"wall_time_{wc}w"] = result["wall_time"]
        metrics[f"throughput_{wc}w"] = result["throughput"]
        metrics[f"efficiency_{wc}w"] = result["efficiency"]

    logger.info("Worker scaling benchmark complete")

    return BenchmarkResult(
        name="worker_scaling",
        config={
            "worker_counts": worker_counts,
            "traversals_per_test": traversals_per_test,
            "config_path": config_path,
        },
        timings=timings,
        metrics=metrics,
        metadata={
            "device": device,
            "max_workers_tested": max(worker_counts),
        },
    )
