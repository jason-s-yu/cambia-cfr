"""
src/benchmarks/reporting.py

Formatting and reporting utilities for benchmark results.

Provides Rich-based console output and file export functionality.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from .runner import BenchmarkResult

logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Examples:
        0.001234 -> "1.23 ms"
        0.123456 -> "123.46 ms"
        1.234567 -> "1.23 s"
        123.456  -> "2.06 min"
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = seconds / 60
        return f"{minutes:.2f} min"


def format_throughput(samples_per_sec: float) -> str:
    """
    Format throughput in samples/sec to human-readable string.

    Examples:
        123.45 -> "123.45 samples/s"
        1234.5 -> "1.23 K samples/s"
        123456 -> "123.46 K samples/s"
    """
    if samples_per_sec < 1000:
        return f"{samples_per_sec:.2f} samples/s"
    elif samples_per_sec < 1_000_000:
        return f"{samples_per_sec / 1000:.2f} K samples/s"
    else:
        return f"{samples_per_sec / 1_000_000:.2f} M samples/s"


def print_result(result: "BenchmarkResult"):
    """
    Print a BenchmarkResult to console using Rich tables.

    Displays:
    - Benchmark name and timestamp
    - Configuration parameters
    - Timing measurements
    - Derived metrics
    - Metadata (device, etc.)
    """
    console = Console()

    console.print(f"\n[bold cyan]{result.name}[/bold cyan]")
    console.print(f"[dim]Timestamp: {result.timestamp}[/dim]\n")

    # Configuration table
    if result.config:
        config_table = Table(title="Configuration", show_header=True)
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="white")

        for key, value in result.config.items():
            config_table.add_row(key, str(value))

        console.print(config_table)

    # Timings table
    if result.timings:
        timing_table = Table(title="Timings", show_header=True)
        timing_table.add_column("Measurement", style="cyan")
        timing_table.add_column("Duration", style="green")

        for key, seconds in result.timings.items():
            timing_table.add_row(key, format_duration(seconds))

        console.print(timing_table)

    # Metrics table
    if result.metrics:
        metrics_table = Table(title="Metrics", show_header=True)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="yellow")

        for key, value in result.metrics.items():
            # Format throughput specially
            if "throughput" in key.lower() or "samples_per_sec" in key:
                formatted = format_throughput(value)
            elif isinstance(value, float):
                formatted = f"{value:.4f}"
            else:
                formatted = str(value)
            metrics_table.add_row(key, formatted)

        console.print(metrics_table)

    # Metadata table
    if result.metadata:
        meta_table = Table(title="Metadata", show_header=True)
        meta_table.add_column("Key", style="cyan")
        meta_table.add_column("Value", style="dim")

        for key, value in result.metadata.items():
            meta_table.add_row(key, str(value))

        console.print(meta_table)


def print_comparison(results: List["BenchmarkResult"]):
    """
    Print side-by-side comparison of multiple benchmark results.

    Useful for comparing different configurations or devices.
    """
    if not results:
        return

    console = Console()
    console.print("\n[bold cyan]Benchmark Comparison[/bold cyan]\n")

    # Build comparison table
    table = Table(show_header=True)
    table.add_column("Benchmark", style="cyan")

    # Find common metrics across all results
    all_metrics = set()
    for result in results:
        all_metrics.update(result.metrics.keys())

    for metric in sorted(all_metrics):
        table.add_column(metric, style="yellow")

    # Add rows
    for result in results:
        row_values = [result.name]
        for metric in sorted(all_metrics):
            value = result.metrics.get(metric, "N/A")
            if value != "N/A":
                if "throughput" in metric.lower() or "samples_per_sec" in metric:
                    row_values.append(format_throughput(value))
                elif isinstance(value, float):
                    row_values.append(f"{value:.4f}")
                else:
                    row_values.append(str(value))
            else:
                row_values.append(value)

        table.add_row(*row_values)

    console.print(table)


def save_json(result: "BenchmarkResult", path: str):
    """
    Save a BenchmarkResult to a JSON file.

    Args:
        result: BenchmarkResult to save
        path: Destination file path
    """
    import json

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info("Saved benchmark result to %s", output_path)


def save_summary(results: List["BenchmarkResult"], path: str):
    """
    Save a text summary of all benchmark results.

    Args:
        results: List of BenchmarkResults
        path: Destination file path
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Capture Rich console output to string
    console = Console(file=open(output_path, "w"), width=120)

    console.print("[bold cyan]Benchmark Suite Summary[/bold cyan]\n")

    for result in results:
        console.print(f"\n{'=' * 80}")
        console.print(f"[bold]{result.name}[/bold]")
        console.print(f"Timestamp: {result.timestamp}\n")

        if result.config:
            console.print("[cyan]Configuration:[/cyan]")
            for key, value in result.config.items():
                console.print(f"  {key}: {value}")
            console.print()

        if result.timings:
            console.print("[cyan]Timings:[/cyan]")
            for key, seconds in result.timings.items():
                console.print(f"  {key}: {format_duration(seconds)}")
            console.print()

        if result.metrics:
            console.print("[cyan]Metrics:[/cyan]")
            for key, value in result.metrics.items():
                if "throughput" in key.lower() or "samples_per_sec" in key:
                    formatted = format_throughput(value)
                elif isinstance(value, float):
                    formatted = f"{value:.4f}"
                else:
                    formatted = str(value)
                console.print(f"  {key}: {formatted}")
            console.print()

    console.file.close()
    logger.info("Saved summary to %s", output_path)
