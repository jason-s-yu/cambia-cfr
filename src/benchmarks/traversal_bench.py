"""
src/benchmarks/traversal_bench.py

Single-worker traversal performance benchmark for Deep CFR.
Measures traversal throughput, nodes visited, and sample generation rates.
"""

import logging
import time
from typing import Optional, List

import numpy as np

from ..config import load_config
from ..encoding import INPUT_DIM, NUM_ACTIONS
from ..networks import AdvantageNetwork
from ..cfr.deep_worker import run_deep_cfr_worker
from .runner import BenchmarkResult

logger = logging.getLogger(__name__)


def benchmark_traversal(
    num_traversals: int = 20,
    config_path: Optional[str] = None,
    device: str = "cpu",
) -> BenchmarkResult:
    """
    Measure single-worker traversal performance.

    Runs a series of sequential traversals with dummy network weights and
    measures per-traversal timing, nodes visited, and sample generation rates.

    Args:
        num_traversals: Number of traversals to run
        config_path: Path to config file (defaults to parallel.config.yaml)
        device: Device for network initialization (cpu/cuda)

    Returns:
        BenchmarkResult with timing and throughput metrics
    """
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

    logger.info("Starting traversal benchmark: %d traversals", num_traversals)

    traversal_times: List[float] = []
    total_nodes = 0
    total_adv_samples = 0
    total_strat_samples = 0
    total_errors = 0

    start_time = time.time()

    for i in range(num_traversals):
        worker_args = (
            i,  # iteration
            config,
            network_weights,
            network_config,
            None,  # progress_queue
            None,  # archive_queue
            0,  # worker_id
            "/tmp/bench_logs",  # run_log_dir
            "bench",  # run_timestamp
        )

        trav_start = time.time()
        result = run_deep_cfr_worker(worker_args)
        trav_end = time.time()

        trav_time = trav_end - trav_start
        traversal_times.append(trav_time)

        if result:
            total_nodes += result.stats.nodes_visited
            total_adv_samples += len(result.advantage_samples)
            total_strat_samples += len(result.strategy_samples)
            total_errors += result.stats.error_count

        if (i + 1) % 5 == 0:
            logger.info(
                "Completed %d/%d traversals (%.2fs each)",
                i + 1,
                num_traversals,
                np.mean(traversal_times[-5:]),
            )

    total_time = time.time() - start_time

    # Calculate metrics
    avg_traversal_time = np.mean(traversal_times)
    std_traversal_time = np.std(traversal_times)
    traversals_per_sec = num_traversals / total_time
    avg_nodes_per_traversal = total_nodes / num_traversals
    nodes_per_sec = total_nodes / total_time
    avg_adv_samples = total_adv_samples / num_traversals
    avg_strat_samples = total_strat_samples / num_traversals

    logger.info(
        "Traversal benchmark complete: %.2f trav/s, %.0f nodes/s",
        traversals_per_sec,
        nodes_per_sec,
    )

    return BenchmarkResult(
        name="traversal_performance",
        config={
            "num_traversals": num_traversals,
            "config_path": config_path,
        },
        timings={
            "total_time": total_time,
            "avg_traversal_time": avg_traversal_time,
            "std_traversal_time": std_traversal_time,
            "min_traversal_time": float(np.min(traversal_times)),
            "max_traversal_time": float(np.max(traversal_times)),
        },
        metrics={
            "traversals_per_sec": traversals_per_sec,
            "nodes_per_sec": nodes_per_sec,
            "avg_nodes_per_traversal": avg_nodes_per_traversal,
            "avg_advantage_samples": avg_adv_samples,
            "avg_strategy_samples": avg_strat_samples,
            "total_nodes_visited": total_nodes,
            "total_errors": total_errors,
        },
        metadata={
            "device": device,
            "hidden_dim": 256,
        },
    )
