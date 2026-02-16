"""src/cli.py - Typer-based CLI for Cambia CFR Training Suite."""

import sys
import signal
import multiprocessing
from pathlib import Path
from typing import Optional

import typer

# Main app
app = typer.Typer(
    name="cambia",
    help="Cambia CFR Training Suite",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Train subcommand group
train_app = typer.Typer(
    help="Train a CFR agent",
    no_args_is_help=True,
)
app.add_typer(train_app, name="train")


def setup_multiprocessing():
    """Set up multiprocessing start method for stability."""
    try:
        preferred_method = "forkserver" if sys.platform != "win32" else "spawn"
        available_methods = multiprocessing.get_all_start_methods()
        current_method = multiprocessing.get_start_method(allow_none=True)

        method_to_set = None
        if preferred_method in available_methods:
            method_to_set = preferred_method
        elif "spawn" in available_methods:
            method_to_set = "spawn"
        elif "fork" in available_methods and sys.platform != "win32":
            method_to_set = "fork"

        if method_to_set and (current_method is None or current_method != method_to_set):
            force_set = current_method is None
            multiprocessing.set_start_method(method_to_set, force=force_set)
    except RuntimeError:
        pass


@train_app.command("tabular", help="Train using tabular CFR+")
def train_tabular(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    iterations: Optional[int] = typer.Option(
        None,
        "--iterations",
        "-n",
        help="Number of iterations to run (overrides config)",
    ),
    load: bool = typer.Option(
        False,
        "--load",
        help="Load existing agent data before training",
    ),
    save_path: Optional[Path] = typer.Option(
        None,
        "--save-path",
        "-s",
        help="Override save path for agent data",
    ),
):
    """Train a tabular CFR+ agent."""
    from .config import load_config
    from .main_train import create_infrastructure, run_tabular_training, handle_sigint

    setup_multiprocessing()
    signal.signal(signal.SIGINT, handle_sigint)

    cfg = load_config(str(config))
    if not cfg:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        raise typer.Exit(1)

    total_iterations = (
        iterations if iterations is not None else cfg.cfr_training.num_iterations
    )

    try:
        infra = create_infrastructure(cfg, total_iterations)
        exit_code = run_tabular_training(
            cfg,
            infra,
            iterations=iterations,
            load=load,
            save_path=str(save_path) if save_path else None,
        )
    except Exception as e:
        print(f"FATAL: Error during training: {e}", file=sys.stderr)
        raise typer.Exit(1)

    raise typer.Exit(exit_code)


@train_app.command("deep", help="Train using Deep CFR")
def train_deep(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    steps: Optional[int] = typer.Option(
        None,
        "--steps",
        "-n",
        help="Number of training steps to run",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        help="Resume from checkpoint",
        exists=True,
    ),
    save_path: Optional[Path] = typer.Option(
        None,
        "--save-path",
        "-s",
        help="Override checkpoint save path",
    ),
    # Deep CFR overrides
    lr: Optional[float] = typer.Option(
        None,
        "--lr",
        help="Learning rate",
        rich_help_panel="Deep CFR Overrides",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Training batch size",
        rich_help_panel="Deep CFR Overrides",
    ),
    train_steps: Optional[int] = typer.Option(
        None,
        "--train-steps",
        help="SGD steps per iteration",
        rich_help_panel="Deep CFR Overrides",
    ),
    traversals: Optional[int] = typer.Option(
        None,
        "--traversals",
        help="Traversals per training step",
        rich_help_panel="Deep CFR Overrides",
    ),
    alpha: Optional[float] = typer.Option(
        None,
        "--alpha",
        help="Iteration weighting exponent",
        rich_help_panel="Deep CFR Overrides",
    ),
    buffer_capacity: Optional[int] = typer.Option(
        None,
        "--buffer-capacity",
        help="Reservoir buffer capacity",
        rich_help_panel="Deep CFR Overrides",
    ),
    gpu: Optional[bool] = typer.Option(
        None,
        "--gpu/--no-gpu",
        help="Use GPU if available",
        rich_help_panel="Deep CFR Overrides",
    ),
):
    """Train a Deep CFR agent."""
    from .config import load_config
    from .main_train import create_infrastructure, run_deep_training, handle_sigint
    from .cfr.deep_trainer import DeepCFRConfig

    setup_multiprocessing()
    signal.signal(signal.SIGINT, handle_sigint)

    cfg = load_config(str(config))
    if not cfg:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        raise typer.Exit(1)

    # Build overrides dict from CLI options
    overrides = {}
    if lr is not None:
        overrides["learning_rate"] = lr
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if train_steps is not None:
        overrides["train_steps_per_iteration"] = train_steps
    if traversals is not None:
        overrides["traversals_per_step"] = traversals
    if alpha is not None:
        overrides["alpha"] = alpha
    if buffer_capacity is not None:
        overrides["advantage_buffer_capacity"] = buffer_capacity
        overrides["strategy_buffer_capacity"] = buffer_capacity
    if gpu is not None:
        overrides["use_gpu"] = gpu

    # Bridge config.py DeepCfrConfig -> deep_trainer.py DeepCFRConfig with overrides
    dcfr_config = DeepCFRConfig.from_yaml_config(cfg, **overrides)

    total_steps = steps if steps is not None else 100

    try:
        infra = create_infrastructure(cfg, total_steps)
        exit_code = run_deep_training(
            cfg,
            dcfr_config,
            infra,
            steps=steps,
            checkpoint=str(checkpoint) if checkpoint else None,
            save_path=str(save_path) if save_path else None,
        )
    except Exception as e:
        print(f"FATAL: Error during training: {e}", file=sys.stderr)
        raise typer.Exit(1)

    raise typer.Exit(exit_code)


@app.command("resume", help="Resume training from a checkpoint")
def resume(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to checkpoint file",
        exists=True,
    ),
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    steps: Optional[int] = typer.Option(
        None,
        "--steps",
        "-n",
        help="Additional steps/iterations to run",
    ),
):
    """Resume training from a checkpoint (auto-detects type)."""
    from .config import load_config
    from .main_train import (
        create_infrastructure,
        run_tabular_training,
        run_deep_training,
        handle_sigint,
    )

    setup_multiprocessing()
    signal.signal(signal.SIGINT, handle_sigint)

    cfg = load_config(str(config))
    if not cfg:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        raise typer.Exit(1)

    suffix = checkpoint.suffix.lower()

    try:
        if suffix == ".pt":
            # Deep CFR checkpoint
            from .cfr.deep_trainer import DeepCFRConfig

            dcfr_config = DeepCFRConfig.from_yaml_config(cfg)
            total_steps = steps if steps is not None else 100
            infra = create_infrastructure(cfg, total_steps)
            exit_code = run_deep_training(
                cfg,
                dcfr_config,
                infra,
                steps=steps,
                checkpoint=str(checkpoint),
            )
        elif suffix == ".joblib":
            # Tabular CFR checkpoint
            total_iterations = (
                steps if steps is not None else cfg.cfr_training.num_iterations
            )
            infra = create_infrastructure(cfg, total_iterations)
            exit_code = run_tabular_training(
                cfg,
                infra,
                iterations=steps,
                load=True,
                save_path=str(checkpoint),
            )
        else:
            print(f"ERROR: Unknown checkpoint type: {suffix}", file=sys.stderr)
            print(
                "Expected .pt (Deep CFR) or .joblib (Tabular CFR)", file=sys.stderr
            )
            raise typer.Exit(1)
    except Exception as e:
        print(f"FATAL: Error during training: {e}", file=sys.stderr)
        raise typer.Exit(1)

    raise typer.Exit(exit_code)


@app.command("info", help="Display checkpoint metadata")
def info(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to checkpoint file",
        exists=True,
    ),
):
    """Display checkpoint metadata."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    suffix = checkpoint.suffix.lower()

    if suffix == ".pt":
        import torch

        try:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

            table = Table(title=f"Deep CFR Checkpoint: {checkpoint.name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Training Step", str(ckpt.get("training_step", "N/A")))
            table.add_row(
                "Total Traversals", str(ckpt.get("total_traversals", "N/A"))
            )

            if "dcfr_config" in ckpt:
                dcfr = ckpt["dcfr_config"]
                table.add_row("Learning Rate", str(dcfr.get("learning_rate", "N/A")))
                table.add_row("Batch Size", str(dcfr.get("batch_size", "N/A")))
                table.add_row("Hidden Dim", str(dcfr.get("hidden_dim", "N/A")))
                table.add_row("Alpha", str(dcfr.get("alpha", "N/A")))

            adv_history = ckpt.get("advantage_loss_history", [])
            if adv_history:
                _, last_loss = adv_history[-1]
                table.add_row("Last Advantage Loss", f"{last_loss:.6f}")

            strat_history = ckpt.get("strategy_loss_history", [])
            if strat_history:
                _, last_loss = strat_history[-1]
                table.add_row("Last Strategy Loss", f"{last_loss:.6f}")

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error loading checkpoint:[/red] {e}")
            raise typer.Exit(1)

    elif suffix == ".joblib":
        import joblib

        try:
            data = joblib.load(checkpoint)

            table = Table(title=f"Tabular CFR Checkpoint: {checkpoint.name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            if isinstance(data, dict):
                table.add_row("Iteration", str(data.get("iteration", "N/A")))
                table.add_row(
                    "Infoset Count", str(len(data.get("regret_sum", {})))
                )

                if "exploitability_history" in data:
                    history = data["exploitability_history"]
                    if history:
                        table.add_row(
                            "Recent Exploitability", f"{history[-1]:.6f}"
                        )
            else:
                table.add_row("Type", str(type(data).__name__))

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error loading checkpoint:[/red] {e}")
            raise typer.Exit(1)

    else:
        console.print(f"[red]ERROR: Unknown checkpoint type:[/red] {suffix}")
        console.print("Expected .pt (Deep CFR) or .joblib (Tabular CFR)")
        raise typer.Exit(1)


# Benchmark subcommand group
benchmark_app = typer.Typer(
    help="Run performance benchmarks",
    no_args_is_help=True,
)
app.add_typer(benchmark_app, name="benchmark")


@benchmark_app.command("all")
def benchmark_all(
    output_dir: Path = typer.Option(
        "/workspace/benchmarks",
        "--output-dir",
        "-o",
        help="Output directory for benchmark results",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Device to run on (cpu/cuda)",
    ),
    config: Path = typer.Option(
        "/workspace/config/parallel.config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Also save raw JSON output",
    ),
):
    """Run all benchmarks."""
    from .benchmarks.runner import BenchmarkSuite
    from .benchmarks import network_bench
    from .benchmarks.traversal_bench import benchmark_traversal
    from .benchmarks.worker_scaling import benchmark_worker_scaling
    from .benchmarks.memory_bench import benchmark_memory
    from .benchmarks.e2e_bench import benchmark_e2e

    suite = BenchmarkSuite()
    suite.register(network_bench.benchmark_network_performance, "network")
    suite.register(benchmark_traversal, "traversal")
    suite.register(benchmark_worker_scaling, "scaling")
    suite.register(benchmark_memory, "memory")
    suite.register(benchmark_e2e, "e2e")

    results = suite.run_all(
        output_dir=str(output_dir),
        device=device,
        config_path=str(config),
    )

    print(f"\nBenchmark suite complete. Results saved to {output_dir}")


@benchmark_app.command("network")
def benchmark_network_cmd(
    output_dir: Path = typer.Option(
        "/workspace/benchmarks",
        "--output-dir",
        "-o",
        help="Output directory for benchmark results",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Device to run on (cpu/cuda)",
    ),
    batch_sizes: Optional[str] = typer.Option(
        None,
        "--batch-sizes",
        help="Comma-separated list of batch sizes (e.g., 256,512,1024)",
    ),
):
    """Run network performance benchmarks."""
    from datetime import datetime
    from .benchmarks import network_bench
    from .benchmarks.reporting import print_result
    import json

    batch_size_list = None
    if batch_sizes:
        batch_size_list = [int(x.strip()) for x in batch_sizes.split(",")]

    result = network_bench.benchmark_network_performance(
        device=device, batch_sizes=batch_size_list
    )

    print_result(result)

    # Save to timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = output_dir / timestamp / "network"
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {json_path}")


@benchmark_app.command("traversal")
def benchmark_traversal_cmd(
    output_dir: Path = typer.Option(
        "/workspace/benchmarks",
        "--output-dir",
        "-o",
        help="Output directory for benchmark results",
    ),
    num_traversals: int = typer.Option(
        20,
        "--num-traversals",
        "-n",
        help="Number of traversals to run",
    ),
    config: Path = typer.Option(
        "/workspace/config/parallel.config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    """Run traversal benchmarks."""
    from datetime import datetime
    from .benchmarks.traversal_bench import benchmark_traversal
    from .benchmarks.reporting import print_result
    import json

    result = benchmark_traversal(
        num_traversals=num_traversals,
        config_path=str(config),
        device="cpu",
    )

    print_result(result)

    # Save to timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = output_dir / timestamp / "traversal"
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {json_path}")


@benchmark_app.command("scaling")
def benchmark_scaling_cmd(
    output_dir: Path = typer.Option(
        "/workspace/benchmarks",
        "--output-dir",
        "-o",
        help="Output directory for benchmark results",
    ),
    worker_counts: Optional[str] = typer.Option(
        None,
        "--worker-counts",
        help="Comma-separated list of worker counts (e.g., 1,2,4,8)",
    ),
    traversals: int = typer.Option(
        100,
        "--traversals",
        "-t",
        help="Traversals per test",
    ),
    config: Path = typer.Option(
        "/workspace/config/parallel.config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    """Run worker scaling benchmarks."""
    from datetime import datetime
    from .benchmarks.worker_scaling import benchmark_worker_scaling
    from .benchmarks.reporting import print_result
    import json

    worker_count_list = None
    if worker_counts:
        worker_count_list = [int(x.strip()) for x in worker_counts.split(",")]

    result = benchmark_worker_scaling(
        worker_counts=worker_count_list,
        traversals_per_test=traversals,
        config_path=str(config),
        device="cpu",
    )

    print_result(result)

    # Save to timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = output_dir / timestamp / "scaling"
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {json_path}")


@benchmark_app.command("memory")
def benchmark_memory_cmd(
    output_dir: Path = typer.Option(
        "/workspace/benchmarks",
        "--output-dir",
        "-o",
        help="Output directory for benchmark results",
    ),
    config: Path = typer.Option(
        "/workspace/config/parallel.config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    """Run memory profiling benchmarks."""
    from datetime import datetime
    from .benchmarks.memory_bench import benchmark_memory
    from .benchmarks.reporting import print_result
    import json

    result = benchmark_memory(
        config_path=str(config),
        device="cpu",
    )

    print_result(result)

    # Save to timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = output_dir / timestamp / "memory"
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {json_path}")


@benchmark_app.command("e2e")
def benchmark_e2e_cmd(
    output_dir: Path = typer.Option(
        "/workspace/benchmarks",
        "--output-dir",
        "-o",
        help="Output directory for benchmark results",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Device to run on (cpu/cuda)",
    ),
    num_steps: int = typer.Option(
        2,
        "--num-steps",
        "-n",
        help="Number of training steps to benchmark",
    ),
    num_workers: int = typer.Option(
        4,
        "--num-workers",
        "-w",
        help="Number of parallel workers",
    ),
    config: Path = typer.Option(
        "/workspace/config/parallel.config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    """Run end-to-end training step benchmark."""
    from datetime import datetime
    from .benchmarks.e2e_bench import benchmark_e2e
    from .benchmarks.reporting import print_result
    import json

    result = benchmark_e2e(
        num_steps=num_steps,
        device=device,
        num_workers=num_workers,
        config_path=str(config),
    )

    print_result(result)

    # Save to timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = output_dir / timestamp / "e2e"
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    app()
