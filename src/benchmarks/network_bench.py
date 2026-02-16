"""
src/benchmarks/network_bench.py

Neural network performance benchmarks for Deep CFR.

Benchmarks:
- Forward pass throughput across batch sizes
- Backward pass (training step) throughput
- GPU vs CPU comparison
- Batch size sweep for optimal throughput
"""

import logging
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..encoding import INPUT_DIM, NUM_ACTIONS
from ..networks import AdvantageNetwork, StrategyNetwork
from .runner import BenchmarkResult

logger = logging.getLogger(__name__)

# Try to import psutil for CPU memory tracking (optional)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available, CPU memory profiling disabled")


def _generate_batch(batch_size: int, device: str) -> tuple:
    """
    Generate realistic random batch data for benchmarking.

    Returns:
        (features, action_mask) tensors on the specified device
    """
    # Features in [0, 1] (representing one-hot encodings and normalized values)
    features = torch.rand(batch_size, INPUT_DIM, device=device)

    # Action masks: ~10 legal actions per sample (realistic for game states)
    # Create sparse masks
    action_mask = torch.zeros(batch_size, NUM_ACTIONS, dtype=torch.bool, device=device)
    for i in range(batch_size):
        num_legal = np.random.randint(5, 15)  # 5-15 legal actions
        legal_indices = np.random.choice(NUM_ACTIONS, num_legal, replace=False)
        action_mask[i, legal_indices] = True

    return features, action_mask


def benchmark_forward_pass(
    device: str = "cpu",
    batch_sizes: Optional[List[int]] = None,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> BenchmarkResult:
    """
    Benchmark forward pass throughput for AdvantageNetwork across batch sizes.

    Args:
        device: Device to run on ("cpu" or "cuda")
        batch_sizes: List of batch sizes to test (default: [512, 1024, 2048, 4096, 8192, 16384])
        num_warmup: Number of warmup iterations
        num_iters: Number of timed iterations per batch size

    Returns:
        BenchmarkResult with timing and throughput metrics
    """
    if batch_sizes is None:
        batch_sizes = [512, 1024, 2048, 4096, 8192, 16384]

    logger.info("Running forward pass benchmark on %s", device)

    # Create network
    network = AdvantageNetwork(
        input_dim=INPUT_DIM,
        hidden_dim=256,
        output_dim=NUM_ACTIONS,
        dropout=0.1,
    ).to(device)
    network.eval()

    timings = {}
    metrics = {}

    with torch.no_grad():
        for batch_size in batch_sizes:
            logger.info("Testing batch size %d", batch_size)

            # Generate batch
            features, action_mask = _generate_batch(batch_size, device)

            # Warmup
            for _ in range(num_warmup):
                _ = network(features, action_mask)

            # Synchronize for accurate timing on GPU
            if device == "cuda":
                torch.cuda.synchronize()

            # Timed runs
            durations = []
            for _ in range(num_iters):
                start = time.perf_counter()
                _ = network(features, action_mask)

                if device == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()
                durations.append(end - start)

            # Statistics
            avg_duration = np.mean(durations)
            std_duration = np.std(durations)
            throughput = batch_size / avg_duration

            timings[f"batch_{batch_size}_avg_s"] = avg_duration
            timings[f"batch_{batch_size}_std_s"] = std_duration
            metrics[f"batch_{batch_size}_throughput"] = throughput
            metrics[f"batch_{batch_size}_avg_ms"] = avg_duration * 1000
            metrics[f"batch_{batch_size}_std_ms"] = std_duration * 1000

            logger.info(
                "Batch %d: %.4f ms (± %.4f ms), %.2f samples/s",
                batch_size, avg_duration * 1000, std_duration * 1000, throughput
            )

    result = BenchmarkResult(
        name="forward_pass",
        config={
            "batch_sizes": batch_sizes,
            "num_warmup": num_warmup,
            "num_iters": num_iters,
            "network": "AdvantageNetwork",
        },
        timings=timings,
        metrics=metrics,
        metadata={
            "device": device,
            "network_params": sum(p.numel() for p in network.parameters()),
        },
    )

    return result


def benchmark_backward_pass(
    device: str = "cpu",
    batch_sizes: Optional[List[int]] = None,
    num_warmup: int = 5,
    num_iters: int = 50,
) -> BenchmarkResult:
    """
    Benchmark full training step (forward + backward + optimizer) for AdvantageNetwork.

    Includes:
    - Forward pass
    - Weighted MSE loss computation (matching deep_trainer.py)
    - Backward pass
    - Gradient clipping
    - Optimizer step

    Args:
        device: Device to run on ("cpu" or "cuda")
        batch_sizes: List of batch sizes to test
        num_warmup: Number of warmup iterations
        num_iters: Number of timed iterations per batch size

    Returns:
        BenchmarkResult with timing and throughput metrics
    """
    if batch_sizes is None:
        batch_sizes = [512, 1024, 2048, 4096, 8192]

    logger.info("Running backward pass benchmark on %s", device)

    # Create network and optimizer
    network = AdvantageNetwork(
        input_dim=INPUT_DIM,
        hidden_dim=256,
        output_dim=NUM_ACTIONS,
        dropout=0.1,
    ).to(device)
    network.train()

    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    alpha = 1.5  # Weighting exponent matching deep_trainer

    timings = {}
    metrics = {}

    for batch_size in batch_sizes:
        logger.info("Testing batch size %d", batch_size)

        # Generate batch
        features, action_mask = _generate_batch(batch_size, device)
        # Generate random target values (regrets)
        targets = torch.randn(batch_size, NUM_ACTIONS, device=device)
        # Generate random iteration weights
        iterations = torch.randint(1, 100, (batch_size,), dtype=torch.float32, device=device)
        weights = (iterations + 1.0).pow(alpha)
        weights = weights / weights.mean()

        # Warmup
        for _ in range(num_warmup):
            optimizer.zero_grad()
            predictions = network(features, action_mask)
            masked_preds = predictions * action_mask.float()
            masked_targets = targets * action_mask.float()
            num_legal = action_mask.float().sum(dim=1).clamp(min=1.0)
            per_sample_mse = ((masked_preds - masked_targets) ** 2).sum(dim=1) / num_legal
            loss = (weights * per_sample_mse).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

        # Synchronize for accurate timing on GPU
        if device == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        durations = []
        for _ in range(num_iters):
            start = time.perf_counter()

            optimizer.zero_grad()
            predictions = network(features, action_mask)
            masked_preds = predictions * action_mask.float()
            masked_targets = targets * action_mask.float()
            num_legal = action_mask.float().sum(dim=1).clamp(min=1.0)
            per_sample_mse = ((masked_preds - masked_targets) ** 2).sum(dim=1) / num_legal
            loss = (weights * per_sample_mse).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            durations.append(end - start)

        # Statistics
        avg_duration = np.mean(durations)
        std_duration = np.std(durations)
        throughput = batch_size / avg_duration

        timings[f"batch_{batch_size}_avg_s"] = avg_duration
        timings[f"batch_{batch_size}_std_s"] = std_duration
        metrics[f"batch_{batch_size}_throughput"] = throughput
        metrics[f"batch_{batch_size}_avg_ms"] = avg_duration * 1000
        metrics[f"batch_{batch_size}_std_ms"] = std_duration * 1000

        logger.info(
            "Batch %d: %.4f ms (± %.4f ms), %.2f samples/s",
            batch_size, avg_duration * 1000, std_duration * 1000, throughput
        )

    result = BenchmarkResult(
        name="backward_pass",
        config={
            "batch_sizes": batch_sizes,
            "num_warmup": num_warmup,
            "num_iters": num_iters,
            "network": "AdvantageNetwork",
            "includes": "forward + loss + backward + grad_clip + optimizer",
        },
        timings=timings,
        metrics=metrics,
        metadata={
            "device": device,
            "network_params": sum(p.numel() for p in network.parameters()),
        },
    )

    return result


def benchmark_gpu_vs_cpu(
    batch_sizes: Optional[List[int]] = None,
    num_iters: int = 50,
) -> BenchmarkResult:
    """
    Compare GPU vs CPU performance for forward and backward passes.

    Runs both forward and backward benchmarks on CPU and GPU (if available),
    then computes speedup ratios.

    Args:
        batch_sizes: List of batch sizes to test
        num_iters: Number of iterations per test

    Returns:
        BenchmarkResult with comparison metrics and speedup ratios
    """
    if batch_sizes is None:
        batch_sizes = [1024, 2048, 4096]

    logger.info("Running GPU vs CPU comparison")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping GPU comparison")
        return BenchmarkResult(
            name="gpu_vs_cpu",
            config={"batch_sizes": batch_sizes},
            timings={},
            metrics={"error": "CUDA not available"},
            metadata={"cuda_available": False},
        )

    # Run CPU benchmarks
    cpu_forward = benchmark_forward_pass(
        device="cpu",
        batch_sizes=batch_sizes,
        num_iters=num_iters,
    )
    cpu_backward = benchmark_backward_pass(
        device="cpu",
        batch_sizes=batch_sizes,
        num_iters=num_iters,
    )

    # Run GPU benchmarks
    gpu_forward = benchmark_forward_pass(
        device="cuda",
        batch_sizes=batch_sizes,
        num_iters=num_iters,
    )
    gpu_backward = benchmark_backward_pass(
        device="cuda",
        batch_sizes=batch_sizes,
        num_iters=num_iters,
    )

    # Compute speedup ratios
    metrics = {}
    for batch_size in batch_sizes:
        cpu_forward_time = cpu_forward.timings[f"batch_{batch_size}_avg_s"]
        gpu_forward_time = gpu_forward.timings[f"batch_{batch_size}_avg_s"]
        forward_speedup = cpu_forward_time / gpu_forward_time

        cpu_backward_time = cpu_backward.timings[f"batch_{batch_size}_avg_s"]
        gpu_backward_time = gpu_backward.timings[f"batch_{batch_size}_avg_s"]
        backward_speedup = cpu_backward_time / gpu_backward_time

        metrics[f"batch_{batch_size}_forward_speedup"] = forward_speedup
        metrics[f"batch_{batch_size}_backward_speedup"] = backward_speedup
        metrics[f"batch_{batch_size}_cpu_forward_ms"] = cpu_forward_time * 1000
        metrics[f"batch_{batch_size}_gpu_forward_ms"] = gpu_forward_time * 1000
        metrics[f"batch_{batch_size}_cpu_backward_ms"] = cpu_backward_time * 1000
        metrics[f"batch_{batch_size}_gpu_backward_ms"] = gpu_backward_time * 1000

        logger.info(
            "Batch %d: Forward speedup %.2fx, Backward speedup %.2fx",
            batch_size, forward_speedup, backward_speedup
        )

    result = BenchmarkResult(
        name="gpu_vs_cpu",
        config={
            "batch_sizes": batch_sizes,
            "num_iters": num_iters,
        },
        timings={},
        metrics=metrics,
        metadata={
            "cuda_available": True,
            "cuda_device_name": torch.cuda.get_device_name(0),
        },
    )

    return result


def benchmark_batch_size_sweep(
    device: str = "cpu",
    batch_sizes: Optional[List[int]] = None,
) -> BenchmarkResult:
    """
    Find optimal batch size by sweeping batch sizes and measuring throughput + memory.

    Tests full training loop throughput at each batch size and measures peak memory usage.

    Args:
        device: Device to run on ("cpu" or "cuda")
        batch_sizes: List of batch sizes to sweep (default: power-of-2 from 256 to 16384)

    Returns:
        BenchmarkResult with throughput curve and memory curve
    """
    if batch_sizes is None:
        batch_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]

    logger.info("Running batch size sweep on %s", device)

    # Create network and optimizer
    network = AdvantageNetwork(
        input_dim=INPUT_DIM,
        hidden_dim=256,
        output_dim=NUM_ACTIONS,
        dropout=0.1,
    ).to(device)
    network.train()

    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    alpha = 1.5

    metrics = {}
    timings = {}

    for batch_size in batch_sizes:
        logger.info("Testing batch size %d", batch_size)

        try:
            # Generate batch
            features, action_mask = _generate_batch(batch_size, device)
            targets = torch.randn(batch_size, NUM_ACTIONS, device=device)
            iterations = torch.randint(1, 100, (batch_size,), dtype=torch.float32, device=device)
            weights = (iterations + 1.0).pow(alpha)
            weights = weights / weights.mean()

            # Reset memory tracking
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            elif HAS_PSUTIL:
                process = psutil.Process()
                mem_before = process.memory_info().rss

            # Warmup
            for _ in range(5):
                optimizer.zero_grad()
                predictions = network(features, action_mask)
                masked_preds = predictions * action_mask.float()
                masked_targets = targets * action_mask.float()
                num_legal = action_mask.float().sum(dim=1).clamp(min=1.0)
                per_sample_mse = ((masked_preds - masked_targets) ** 2).sum(dim=1) / num_legal
                loss = (weights * per_sample_mse).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                optimizer.step()

            # Synchronize
            if device == "cuda":
                torch.cuda.synchronize()

            # Timed run
            num_iters = 50
            start = time.perf_counter()

            for _ in range(num_iters):
                optimizer.zero_grad()
                predictions = network(features, action_mask)
                masked_preds = predictions * action_mask.float()
                masked_targets = targets * action_mask.float()
                num_legal = action_mask.float().sum(dim=1).clamp(min=1.0)
                per_sample_mse = ((masked_preds - masked_targets) ** 2).sum(dim=1) / num_legal
                loss = (weights * per_sample_mse).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            avg_duration = (end - start) / num_iters
            throughput = batch_size / avg_duration

            timings[f"batch_{batch_size}_avg_s"] = avg_duration
            metrics[f"batch_{batch_size}_throughput"] = throughput

            # Measure memory
            if device == "cuda":
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                metrics[f"batch_{batch_size}_memory_mb"] = peak_memory_mb
                logger.info(
                    "Batch %d: %.2f samples/s, %.2f MB GPU memory",
                    batch_size, throughput, peak_memory_mb
                )
            elif HAS_PSUTIL:
                mem_after = process.memory_info().rss
                mem_delta_mb = (mem_after - mem_before) / (1024 ** 2)
                metrics[f"batch_{batch_size}_memory_mb"] = mem_delta_mb
                logger.info(
                    "Batch %d: %.2f samples/s, %.2f MB CPU memory delta",
                    batch_size, throughput, mem_delta_mb
                )
            else:
                logger.info("Batch %d: %.2f samples/s", batch_size, throughput)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("Batch size %d: OOM, stopping sweep", batch_size)
                metrics[f"batch_{batch_size}_error"] = "OOM"
                break
            else:
                raise

    # Find optimal batch size (highest throughput)
    throughput_items = [
        (k, v) for k, v in metrics.items()
        if k.endswith("_throughput")
    ]
    if throughput_items:
        optimal_key, optimal_throughput = max(throughput_items, key=lambda x: x[1])
        optimal_batch = int(optimal_key.split("_")[1])
        metrics["optimal_batch_size"] = optimal_batch
        metrics["optimal_throughput"] = optimal_throughput

    result = BenchmarkResult(
        name="batch_size_sweep",
        config={
            "batch_sizes": batch_sizes,
            "device": device,
        },
        timings=timings,
        metrics=metrics,
        metadata={
            "device": device,
            "network_params": sum(p.numel() for p in network.parameters()),
        },
    )

    return result


def benchmark_network_performance(
    device: str = "cpu",
    batch_sizes: Optional[List[int]] = None,
    **kwargs,
) -> BenchmarkResult:
    """
    Run all network benchmarks and combine into a single result.

    This is the main entry point for the network benchmark suite.
    Runs forward pass, backward pass, and batch size sweep on the specified device.
    If GPU is available and device is 'both', also runs GPU vs CPU comparison.

    Args:
        device: Device to run on ("cpu", "cuda", or "both")
        batch_sizes: List of batch sizes to test
        **kwargs: Extra arguments (ignored, for compatibility with BenchmarkSuite)

    Returns:
        BenchmarkResult with combined metrics from all network benchmarks
    """
    if batch_sizes is None:
        batch_sizes = [512, 1024, 2048, 4096, 8192]

    all_timings = {}
    all_metrics = {}

    # Forward pass benchmark
    fwd = benchmark_forward_pass(device=device, batch_sizes=batch_sizes)
    for k, v in fwd.timings.items():
        all_timings[f"forward_{k}"] = v
    for k, v in fwd.metrics.items():
        all_metrics[f"forward_{k}"] = v

    # Backward pass benchmark
    bwd = benchmark_backward_pass(device=device, batch_sizes=batch_sizes)
    for k, v in bwd.timings.items():
        all_timings[f"backward_{k}"] = v
    for k, v in bwd.metrics.items():
        all_metrics[f"backward_{k}"] = v

    # Batch size sweep
    sweep = benchmark_batch_size_sweep(device=device, batch_sizes=batch_sizes)
    for k, v in sweep.timings.items():
        all_timings[f"sweep_{k}"] = v
    for k, v in sweep.metrics.items():
        all_metrics[f"sweep_{k}"] = v

    # GPU vs CPU comparison (only if GPU available)
    if torch.cuda.is_available() and device != "cpu":
        try:
            comparison = benchmark_gpu_vs_cpu(batch_sizes=[1024, 2048, 4096])
            for k, v in comparison.metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics[f"gpu_vs_cpu_{k}"] = v
        except Exception as e:
            logger.warning("GPU vs CPU comparison failed: %s", e)

    return BenchmarkResult(
        name="network_performance",
        config={
            "device": device,
            "batch_sizes": batch_sizes,
        },
        timings=all_timings,
        metrics=all_metrics,
        metadata={
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "network_params": sum(p.numel() for p in AdvantageNetwork().parameters()),
        },
    )
