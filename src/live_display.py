# src/live_display.py
"""Manages the Rich-based live console display for training progress."""

import logging
from collections import deque
from typing import Deque, Dict, Optional, Tuple, Union, List

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from .utils import WorkerStats, format_large_number

# Define worker status type alias used internally by the display
WorkerDisplayStatus = Union[str, WorkerStats, Tuple[str, int, int, int]]


class LiveDisplayManager:
    """Manages the creation and updating of the Rich live display."""

    def __init__(self, num_workers: int, total_iterations: int, console: Console):
        self.num_workers = num_workers
        self.total_iterations = total_iterations
        self.console = console

        # Internal state tracking
        self._worker_statuses: Dict[int, WorkerDisplayStatus] = {
            i: "Initializing" for i in range(num_workers)
        }
        self._current_iteration = 0
        self._last_exploitability = "N/A"
        self._total_infosets = "0"
        self._last_iter_time = "N/A"
        self._log_records: Deque[logging.LogRecord] = deque(maxlen=15)

        # Rich components
        self.progress = self._create_progress_bar()
        self.iteration_task_id = self.progress.add_task(
            "Overall Progress", total=total_iterations, status_text="Initializing..."
        )
        self.layout = self._create_layout()
        self.live: Optional[Live] = None

    def _create_progress_bar(self) -> Progress:
        """Creates the Rich Progress instance."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn("• Iter: {task.completed}/{task.total}"),
            TextColumn("• Elapsed:"),
            TimeElapsedColumn(),
            TextColumn("• Remaining:"),
            TimeRemainingColumn(),
            TextColumn("• {task.fields[status_text]}"),
            console=self.console,
            transient=False,
        )

    def _create_layout(self) -> Layout:
        """Creates the Rich Layout structure."""
        layout = Layout()
        layout.split(
            Layout(name="header", size=1),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=7),
        )
        layout["main"].split_row(
            Layout(name="workers", ratio=2), Layout(name="logs", ratio=3)
        )
        layout["footer"].split(
            Layout(name="progress_bar", size=3), Layout(name="padding", size=4)
        )
        return layout

    def _generate_worker_table(self) -> Table:
        """Generates the table displaying worker status."""
        table = Table(
            title=f"Worker Status (Iteration {self._current_iteration})",
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        table.add_column("ID", style="dim", width=4, justify="right")
        table.add_column("Status", style="cyan", justify="left", no_wrap=True)
        table.add_column("Depth", style="green", width=10, justify="right")
        table.add_column("Nodes", style="blue", width=12, justify="right")
        table.add_column("Warn", style="yellow", width=5, justify="right")
        table.add_column("Err", style="red", width=5, justify="right")

        for i in range(self.num_workers):
            status_info = self._worker_statuses.get(i, "Unknown")
            status_str, depth_str, nodes_str, warn_str, err_str = (
                "Unknown",
                "N/A",
                "N/A",
                "0",
                "0",
            )

            if isinstance(status_info, WorkerStats):
                status_str = "[green]Done[/green]"
                depth_str = str(status_info.max_depth)
                nodes_str = format_large_number(status_info.nodes_visited)
                warn_str = str(status_info.warning_count)
                err_str = str(status_info.error_count)
            elif isinstance(status_info, tuple) and len(status_info) == 4:
                state, cur_d, max_d, nodes = status_info
                status_str = state
                depth_str = f"{cur_d}/{max_d}"
                nodes_str = format_large_number(nodes)
                warn_str, err_str = "-", "-"
            elif isinstance(status_info, str):
                if "Error" in status_info or "Fail" in status_info:
                    status_str = f"[bold red]{status_info}[/]"
                elif "Queued" in status_info or "Starting" in status_info:
                    status_str = f"[yellow]{status_info}[/]"
                else:
                    status_str = status_info
                depth_str, nodes_str, warn_str, err_str = "-", "-", "-", "-"

            table.add_row(str(i), status_str, depth_str, nodes_str, warn_str, err_str)

        return table

    def _generate_log_panel(self) -> Panel:
        """Generates the panel displaying recent log messages."""
        log_texts: List[Text] = []  # Store Text objects
        handler = RichHandler(show_level=True, markup=True)  # For level styling

        for record in list(self._log_records):  # Iterate over a copy
            try:
                level_text = handler.get_level_text(record)  # Get styled level
                message_text = self.format_log_message(record)  # Get formatted message
                # Combine level and message into a single Text line, add newline
                log_line = Text.assemble(level_text, " ", message_text, "\n")
                log_texts.append(log_line)
            except Exception as fmt_exc:
                # Avoid crashing display if a single log record fails formatting
                log_texts.append(Text(f"Log Format Error: {fmt_exc}\n", style="red"))

        # Combine all Text lines into one renderable Text object
        # Use Group for potentially better vertical overflow handling
        log_content = Group(*log_texts) if log_texts else Text("")

        return Panel(
            log_content,
            title="Recent Logs",
            border_style="dim",
            expand=True,
        )

    def format_log_message(self, record: logging.LogRecord) -> str:
        """Formats a log record message, handling potential % formatting."""
        try:
            msg = record.getMessage()
            # Optional: Simplify further?
            # msg = msg.replace(f"[{record.processName}]", "").strip()
            # msg = msg.replace(f"- {record.name} -", "").strip()
            return msg.strip()
        except Exception as e:
            return f"Log format error: {e}"

    def _generate_header_text(self) -> Text:
        """Generates the header text."""
        header = (
            f" Cambia CFR+ Training :: Iter: {self._current_iteration}/{self.total_iterations} "
            f":: Infosets: {self._total_infosets} :: Expl: {self._last_exploitability} "
            f":: Last Iter: {self._last_iter_time} "
        )
        return Text(header, style="bold white on blue", justify="center")

    def _generate_renderable(self) -> Layout:
        """Builds the complete layout with updated components."""
        try:
            self.layout["header"].update(self._generate_header_text())
            self.layout["workers"].update(
                Panel(self._generate_worker_table(), title="Workers")
            )
            self.layout["logs"].update(self._generate_log_panel())
            self.layout["progress_bar"].update(self.progress)
            self.layout["padding"].update("")
            return self.layout
        except Exception as e:
            # Fallback if rendering fails during update
            logging.error(f"Error generating display layout: {e}", exc_info=True)
            return Text(f"Error generating display layout: {e}", style="bold red")

    # --- Public Update Methods ---
    def add_log_record(self, record: logging.LogRecord):
        """Adds a log record to the display queue."""
        self._log_records.append(record)

    def update_worker_status(self, worker_id: int, status: WorkerDisplayStatus):
        """Updates the status of a specific worker."""
        if 0 <= worker_id < self.num_workers:
            self._worker_statuses[worker_id] = status
        else:
            logging.error(
                f"LiveDisplay: Invalid worker ID {worker_id} for status update."
            )

    def update_overall_progress(self, completed: int):
        """Updates the main iteration progress bar."""
        self._current_iteration = completed
        try:
            self.progress.update(self.iteration_task_id, completed=completed)
        except Exception as e:
            logging.error(f"Error updating progress bar: {e}")

    def update_stats(
        self,
        iteration: int,
        infosets: str,
        exploitability: str,
        last_iter_time: Optional[float] = None,
    ):
        """Updates the displayed overall statistics."""
        self._current_iteration = iteration
        self._total_infosets = infosets
        self._last_exploitability = exploitability
        self._last_iter_time = (
            f"{last_iter_time:.2f}s" if last_iter_time is not None else "N/A"
        )
        status_text = f"Infosets: {infosets} | Expl: {exploitability} | Last T: {self._last_iter_time}"
        try:
            self.progress.update(
                self.iteration_task_id, status_text=status_text, advance=0
            )
        except Exception as e:
            logging.error(f"Error updating progress bar stats: {e}")

    def start(self):
        """Starts the Rich Live display."""
        if not self.live:
            try:
                self.live = Live(
                    self._generate_renderable(),
                    console=self.console,
                    refresh_per_second=4,
                    transient=False,
                    vertical_overflow="visible",
                )
                self.live.start(refresh=True)
                logging.debug("Rich Live display started.")
            except Exception as e:
                logging.error(f"Failed to start Rich Live display: {e}", exc_info=True)
                self.live = None

    def stop(self):
        """Stops the Rich Live display."""
        if self.live:
            try:
                self.live.stop()
                self.console.print()
                logging.debug("Rich Live display stopped.")
            except Exception as e:
                logging.error(f"Error stopping Rich Live display: {e}", exc_info=True)
            finally:
                self.live = None

    def run(self, func, *args, **kwargs):
        """Runs a function within the Live context."""
        try:
            if not self.live:
                with Live(
                    self._generate_renderable(),
                    console=self.console,
                    refresh_per_second=4,
                    transient=False,
                    vertical_overflow="visible",
                ) as live_context:
                    self.live = live_context
                    try:
                        result = func(*args, **kwargs)
                        return result
                    finally:
                        self.live = None  # Ensure live is cleared before context exits
            else:
                # Already live (shouldn't happen)
                result = func(*args, **kwargs)
                return result
        except Exception:
            if self.live:
                self.stop()  # Ensure display is stopped on error
            raise
