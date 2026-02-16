"""
src/benchmarks/e2e_bench.py

End-to-end training step benchmark for Deep CFR.
Measures time breakdown across all phases of a training iteration.
"""

import logging
import multiprocessing
import time
from typing import Optional

import numpy as np
import torch

from ..config import load_config
from ..encoding import INPUT_DIM, NUM_ACTIONS
from ..networks import AdvantageNetwork, StrategyNetwork
from ..reservoir import ReservoirBuffer
from ..cfr.deep_worker import run_deep_cfr_worker
from .runner import BenchmarkResult

logger = logging.getLogger(__name__)


def benchmark_e2e(
    num_steps: int = 2,
    device: str = "cpu",
    num_workers: int = 4,
    traversals_per_step: int = 50,
    train_steps_per_iteration: int = 100,
    config_path: Optional[str] = None,
) -> BenchmarkResult:
    """
    End-to-end training step benchmark with detailed time breakdown.

    Simulates the full Deep CFR training loop for a specified number of steps,
    measuring time spent in each phase:
    1. Weight serialization
    2. Traversal (worker dispatch + collection)
    3. Sample buffer append
    4. Advantage network training
    5. Strategy network training
    6. Checkpoint save (optional)

    Args:
        num_steps: Number of training steps to benchmark
        device: Device for network training ("cpu" or "cuda")
        num_workers: Number of parallel workers for traversals
        traversals_per_step: Traversals to run per training step
        train_steps_per_iteration: SGD steps per network training phase
        config_path: Path to config file

    Returns:
        BenchmarkResult with detailed timing breakdown
    """
    config_path = config_path or "/workspace/config/parallel.config.yaml"
    config = load_config(config_path)
    if not config:
        raise RuntimeError(f"Failed to load config from {config_path}")

    # Initialize networks
    device_obj = torch.device(device)
    adv_net = AdvantageNetwork(
        input_dim=INPUT_DIM, hidden_dim=256, output_dim=NUM_ACTIONS
    ).to(device_obj)
    strat_net = StrategyNetwork(
        input_dim=INPUT_DIM, hidden_dim=256, output_dim=NUM_ACTIONS
    ).to(device_obj)

    adv_optimizer = torch.optim.Adam(adv_net.parameters(), lr=1e-3)
    strat_optimizer = torch.optim.Adam(strat_net.parameters(), lr=1e-3)

    # Initialize buffers
    adv_buffer = ReservoirBuffer(capacity=100_000)
    strat_buffer = ReservoirBuffer(capacity=100_000)

    # Network config for workers
    network_config = {
        "input_dim": INPUT_DIM,
        "hidden_dim": 256,
        "output_dim": NUM_ACTIONS,
    }

    logger.info(
        "Starting e2e benchmark: %d steps, %d workers, %d trav/step, device=%s",
        num_steps,
        num_workers,
        traversals_per_step,
        device,
    )

    # Timing accumulators
    total_serialization_time = 0.0
    total_traversal_time = 0.0
    total_buffer_append_time = 0.0
    total_adv_train_time = 0.0
    total_strat_train_time = 0.0

    step_times = []

    for step in range(num_steps):
        step_start = time.time()
        logger.info("Step %d/%d", step + 1, num_steps)

        # --- Phase 1: Weight Serialization ---
        ser_start = time.time()
        network_weights = {k: v.cpu().numpy() for k, v in adv_net.state_dict().items()}
        ser_time = time.time() - ser_start
        total_serialization_time += ser_time

        # --- Phase 2: Traversal ---
        trav_start = time.time()

        worker_args_list = []
        for i in range(traversals_per_step):
            worker_args_list.append(
                (
                    step * traversals_per_step + i,  # iteration
                    config,
                    network_weights,
                    network_config,
                    None,  # progress_queue
                    None,  # archive_queue
                    i % num_workers,  # worker_id
                    "/tmp/bench_logs",
                    "bench",
                )
            )

        if num_workers == 1:
            results = [run_deep_cfr_worker(args) for args in worker_args_list]
        else:
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = pool.map(run_deep_cfr_worker, worker_args_list)

        trav_time = time.time() - trav_start
        total_traversal_time += trav_time

        # --- Phase 3: Buffer Append ---
        append_start = time.time()
        for result in results:
            if result:
                for sample in result.advantage_samples:
                    adv_buffer.add(sample)
                for sample in result.strategy_samples:
                    strat_buffer.add(sample)
        append_time = time.time() - append_start
        total_buffer_append_time += append_time

        # --- Phase 4: Advantage Network Training ---
        adv_train_start = time.time()
        adv_net.train()
        for _ in range(train_steps_per_iteration):
            batch = adv_buffer.sample_batch(2048)
            if not batch:
                break

            features_batch = np.stack([s.features for s in batch])
            targets_batch = np.stack([s.target for s in batch])
            masks_batch = np.stack([s.action_mask for s in batch])
            iterations_batch = np.array([s.iteration for s in batch], dtype=np.float32)

            features_t = torch.from_numpy(features_batch).float().to(device_obj)
            targets_t = torch.from_numpy(targets_batch).float().to(device_obj)
            masks_t = torch.from_numpy(masks_batch).bool().to(device_obj)
            iterations_t = torch.from_numpy(iterations_batch).float().to(device_obj)

            weights = (iterations_t + 1.0).pow(1.5) / (iterations_t + 1.0).pow(1.5).mean()

            predictions = adv_net(features_t, masks_t)
            masked_preds = predictions * masks_t.float()
            masked_targets = targets_t * masks_t.float()

            num_legal = masks_t.float().sum(dim=1).clamp(min=1.0)
            per_sample_mse = ((masked_preds - masked_targets) ** 2).sum(dim=1) / num_legal
            loss = (weights * per_sample_mse).mean()

            adv_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adv_net.parameters(), max_norm=1.0)
            adv_optimizer.step()

        adv_train_time = time.time() - adv_train_start
        total_adv_train_time += adv_train_time

        # --- Phase 5: Strategy Network Training ---
        strat_train_start = time.time()
        strat_net.train()
        for _ in range(train_steps_per_iteration):
            batch = strat_buffer.sample_batch(2048)
            if not batch:
                break

            features_batch = np.stack([s.features for s in batch])
            targets_batch = np.stack([s.target for s in batch])
            masks_batch = np.stack([s.action_mask for s in batch])
            iterations_batch = np.array([s.iteration for s in batch], dtype=np.float32)

            features_t = torch.from_numpy(features_batch).float().to(device_obj)
            targets_t = torch.from_numpy(targets_batch).float().to(device_obj)
            masks_t = torch.from_numpy(masks_batch).bool().to(device_obj)
            iterations_t = torch.from_numpy(iterations_batch).float().to(device_obj)

            weights = (iterations_t + 1.0).pow(1.5) / (iterations_t + 1.0).pow(1.5).mean()

            predictions = strat_net(features_t, masks_t)
            masked_preds = predictions * masks_t.float()
            masked_targets = targets_t * masks_t.float()

            num_legal = masks_t.float().sum(dim=1).clamp(min=1.0)
            per_sample_mse = ((masked_preds - masked_targets) ** 2).sum(dim=1) / num_legal
            loss = (weights * per_sample_mse).mean()

            strat_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(strat_net.parameters(), max_norm=1.0)
            strat_optimizer.step()

        strat_train_time = time.time() - strat_train_start
        total_strat_train_time += strat_train_time

        step_time = time.time() - step_start
        step_times.append(step_time)

        logger.info(
            "  Step %d: %.2fs total (trav=%.2fs, adv_train=%.2fs, strat_train=%.2fs)",
            step + 1,
            step_time,
            trav_time,
            adv_train_time,
            strat_train_time,
        )

    # Calculate averages
    avg_step_time = np.mean(step_times)
    avg_serialization = total_serialization_time / num_steps
    avg_traversal = total_traversal_time / num_steps
    avg_append = total_buffer_append_time / num_steps
    avg_adv_train = total_adv_train_time / num_steps
    avg_strat_train = total_strat_train_time / num_steps

    logger.info("E2E benchmark complete: %.2fs per step average", avg_step_time)

    return BenchmarkResult(
        name="e2e_training_step",
        config={
            "num_steps": num_steps,
            "device": device,
            "num_workers": num_workers,
            "traversals_per_step": traversals_per_step,
            "train_steps_per_iteration": train_steps_per_iteration,
            "config_path": config_path,
        },
        timings={
            "avg_step_time": avg_step_time,
            "total_time": sum(step_times),
            "avg_serialization": avg_serialization,
            "avg_traversal": avg_traversal,
            "avg_buffer_append": avg_append,
            "avg_advantage_training": avg_adv_train,
            "avg_strategy_training": avg_strat_train,
            "total_serialization": total_serialization_time,
            "total_traversal": total_traversal_time,
            "total_buffer_append": total_buffer_append_time,
            "total_advantage_training": total_adv_train_time,
            "total_strategy_training": total_strat_train_time,
        },
        metrics={
            "steps_per_hour": 3600 / avg_step_time if avg_step_time > 0 else 0,
            "traversals_per_sec": traversals_per_step / avg_traversal
            if avg_traversal > 0
            else 0,
            "buffer_sizes": {
                "advantage": len(adv_buffer),
                "strategy": len(strat_buffer),
            },
            "time_breakdown_pct": {
                "serialization": (avg_serialization / avg_step_time * 100)
                if avg_step_time > 0
                else 0,
                "traversal": (avg_traversal / avg_step_time * 100)
                if avg_step_time > 0
                else 0,
                "buffer_append": (avg_append / avg_step_time * 100)
                if avg_step_time > 0
                else 0,
                "advantage_training": (avg_adv_train / avg_step_time * 100)
                if avg_step_time > 0
                else 0,
                "strategy_training": (avg_strat_train / avg_step_time * 100)
                if avg_step_time > 0
                else 0,
            },
        },
        metadata={
            "device": device,
            "cuda_available": torch.cuda.is_available(),
        },
    )
