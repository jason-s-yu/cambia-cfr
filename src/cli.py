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


if __name__ == "__main__":
    app()
