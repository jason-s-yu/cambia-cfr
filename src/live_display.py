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

        # Store previous renderables to compare for changes
        self._last_header_text_str: Optional[str] = None
        self._last_worker_table_title: Optional[str] = None
        self._last_worker_statuses_repr: Optional[str] = None
        self._last_log_panel_content_str: Optional[str] = None

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
            TextColumn("• [i]({task.fields[status_text]})[/i]"),
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
            title=f"Worker Status (Iter {self._current_iteration})",
            show_header=True,
            header_style="bold magenta",
            expand=True,
            padding=(0, 1),
        )
        table.add_column("ID", style="dim", width=4, justify="right")
        table.add_column(
            "Status", style="cyan", justify="left", no_wrap=True, min_width=10
        )
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
                if status_info.error_count > 0:
                    status_str = "[bold red]Error[/bold red]"
                elif status_info.warning_count > 0:
                    status_str = "[yellow]Warn[/yellow]"
                else:
                    status_str = "[green]Done[/green]"
                depth_str = str(status_info.max_depth)
                nodes_str = format_large_number(status_info.nodes_visited)
                warn_str = str(status_info.warning_count)
                err_str = str(status_info.error_count)
            elif isinstance(status_info, tuple) and len(status_info) == 4:
                state, cur_d, max_d, nodes = status_info
                status_str = state if state else "Running"
                depth_str = f"{cur_d}/{max_d}"
                nodes_str = format_large_number(nodes)
                warn_str, err_str = "-", "-"
            elif isinstance(status_info, str):
                if "Error" in status_info or "Fail" in status_info:
                    status_str = f"[bold red]{status_info}[/]"
                elif (
                    "Queued" in status_info
                    or "Starting" in status_info
                    or "Initializing" in status_info
                ):
                    status_str = f"[yellow]{status_info}[/]"
                elif "Finished" in status_info or "Done" in status_info:
                    status_str = "[green]Done[/green]"
                elif "Idle" in status_info:
                    status_str = "[dim]Idle[/dim]"
                else:
                    status_str = status_info
                depth_str, nodes_str, warn_str, err_str = "-", "-", "-", "-"
            table.add_row(str(i), status_str, depth_str, nodes_str, warn_str, err_str)
        return table

    def _generate_log_panel_content_str(self) -> str:
        """Generates the string content for the log panel, for comparison."""
        log_texts_plain: List[str] = []
        # Ensure RichHandler uses the same console for consistent formatting, if needed.
        handler = RichHandler(
            show_level=True,
            markup=True,
            rich_tracebacks=False,
            show_path=False,
            console=self.console,
        )
        for record in list(self._log_records):
            try:
                render_result = handler.render_message(
                    record, message=record.getMessage()
                )
                if render_result:
                    log_texts_plain.append(
                        render_result.plain
                        if isinstance(render_result, Text)
                        else str(render_result)
                    )
            except Exception:  # pylint: disable=broad-except
                log_texts_plain.append(f"Log Format Error: {record.getMessage()}\n")
        return "".join(log_texts_plain)

    def _generate_log_panel(self) -> Panel:
        """Generates the panel displaying recent log messages."""
        log_texts: List[Text] = []
        handler = RichHandler(
            show_level=True,
            markup=True,
            rich_tracebacks=False,
            show_path=False,
            console=self.console,
        )
        for record in list(self._log_records):
            try:
                render_result = handler.render_message(
                    record, message=record.getMessage()
                )
                if render_result:
                    if isinstance(
                        render_result, Text
                    ) and not render_result.plain.endswith("\n"):
                        render_result.append("\n")
                    log_texts.append(render_result)
            except Exception as fmt_exc:  # pylint: disable=broad-except
                log_texts.append(Text(f"Log Format Error: {fmt_exc}\n", style="red"))
        log_content = Group(*log_texts) if log_texts else Text("")
        return Panel(log_content, title="Recent Logs", border_style="dim", expand=True)

    def _generate_header_text_obj(self) -> Text:
        """Generates the Rich Text object for the header."""
        header_str = (
            f" Cambia CFR+ Training :: Iter: {self._current_iteration}/{self.total_iterations} "
            f":: Infosets: {self._total_infosets} :: Expl: {self._last_exploitability} "
            f":: Last Iter: {self._last_iter_time} "
        )
        return Text(header_str, style="bold white on blue", justify="center")

    # Method name matches user's traceback context
    def _update_layout_if_changed(self) -> bool:
        """
        Updates individual components of the layout if their content has changed.
        Returns True if any component was updated.
        """
        layout_updated = False

        # Header
        current_header_text_obj = self._generate_header_text_obj()
        current_header_str = current_header_text_obj.plain
        if self._last_header_text_str != current_header_str:
            self.layout["header"].update(current_header_text_obj)
            self._last_header_text_str = current_header_str
            layout_updated = True

        # Worker Table
        current_worker_table_title = f"Worker Status (Iter {self._current_iteration})"
        current_worker_statuses_repr = repr(sorted(self._worker_statuses.items()))
        if (
            self._last_worker_table_title != current_worker_table_title
            or self._last_worker_statuses_repr != current_worker_statuses_repr
        ):
            self.layout["workers"].update(
                Panel(self._generate_worker_table(), title="Workers", border_style="blue")
            )
            self._last_worker_table_title = current_worker_table_title
            self._last_worker_statuses_repr = current_worker_statuses_repr
            layout_updated = True

        # Log Panel
        current_log_str = self._generate_log_panel_content_str()
        if self._last_log_panel_content_str != current_log_str:
            self.layout["logs"].update(self._generate_log_panel())
            self._last_log_panel_content_str = current_log_str
            layout_updated = True

        try:
            pb_layout_region = self.layout.get("progress_bar")
            if pb_layout_region is not None:
                if pb_layout_region.renderable is not self.progress:
                    pb_layout_region.update(self.progress)
            else:
                logging.error(
                    "Critical: Layout region 'progress_bar' not found during update processing."
                )
        except Exception as e_pb_update:
            logging.error(
                f"Error accessing/updating 'progress_bar' layout region: {e_pb_update}",
                exc_info=True,
            )

        self.layout["padding"].update("")
        return layout_updated

    def refresh(self):
        """Updates the Live display if necessary."""
        if self.live:
            try:
                content_changed = self._update_layout_if_changed()
                self.live.update(self.layout, refresh=content_changed)
            except Exception as e:
                logging.error(
                    "Error explicitly refreshing Live display: %s", e, exc_info=True
                )

    def add_log_record(self, record: logging.LogRecord):
        """Adds a log record to the display queue and refreshes."""
        self._log_records.append(record)
        self.refresh()

    def update_worker_status(self, worker_id: int, status: WorkerDisplayStatus):
        """Updates the status of a specific worker and refreshes."""
        if 0 <= worker_id < self.num_workers:
            if self._worker_statuses.get(worker_id) != status:
                self._worker_statuses[worker_id] = status
                self.refresh()  # Refresh when worker status changes
        else:
            logging.warning(
                "LiveDisplay: Invalid worker ID %s for status update.", worker_id
            )

    def update_overall_progress(self, completed: int):
        """Updates the main iteration progress bar and refreshes if iteration changed."""
        needs_refresh_for_header = False
        if self._current_iteration != completed:
            self._current_iteration = completed
            needs_refresh_for_header = (
                True  # Iteration number change affects header & table title
            )
        try:
            self.progress.update(self.iteration_task_id, completed=completed)
            if needs_refresh_for_header:
                self.refresh()
        except Exception as e:
            logging.error("Error updating progress bar completion: %s", e, exc_info=True)

    def update_stats(
        self,
        iteration: int,
        infosets: str,
        exploitability: str,
        last_iter_time: Optional[float] = None,
    ):
        """Updates the displayed overall statistics and refreshes if header data changed."""
        needs_refresh_for_header = False
        if self._current_iteration != iteration:
            self._current_iteration = iteration
            needs_refresh_for_header = True
        if self._total_infosets != infosets:
            self._total_infosets = infosets
            needs_refresh_for_header = True
        if self._last_exploitability != exploitability:
            self._last_exploitability = exploitability
            needs_refresh_for_header = True

        new_last_iter_time_str = (
            f"{last_iter_time:.2f}s" if last_iter_time is not None else "N/A"
        )
        if self._last_iter_time != new_last_iter_time_str:
            self._last_iter_time = new_last_iter_time_str
            needs_refresh_for_header = True

        status_text_for_progress_bar = f"Infosets: {infosets} | Expl: {exploitability} | Last T: {self._last_iter_time}"
        try:
            self.progress.update(
                self.iteration_task_id, status_text=status_text_for_progress_bar
            )
            if needs_refresh_for_header:
                self.refresh()
        except Exception as e:
            logging.error("Error updating progress bar stats field: %s", e, exc_info=True)

    def start(self):
        """Starts the Rich Live display."""
        if not self.live:
            try:
                self._update_layout_if_changed()
                self.live = Live(
                    self.layout,
                    console=self.console,
                    refresh_per_second=2,
                    transient=False,
                    vertical_overflow="visible",
                )
                self.live.start(refresh=True)
                logging.debug("Rich Live display started.")
            except Exception as e:
                logging.error("Failed to start Rich Live display: %s", e, exc_info=True)
                self.live = None

    def stop(self):
        """Stops the Rich Live display."""
        if self.live:
            try:
                self.live.stop()
                self.console.print()
                logging.debug("Rich Live display stopped.")
            except Exception as e:
                logging.error("Error stopping Rich Live display: %s", e, exc_info=True)
            finally:
                self.live = None

    def run(self, func, *args, **kwargs):
        """Runs a function within the Live context."""
        if self.live:
            current_live_instance = self.live
            self.live = None
            try:
                current_live_instance.stop()
            except Exception as e:  # pylint: disable=broad-except
                logging.warning("Error stopping existing live instance in run(): %s", e)

        try:
            self._update_layout_if_changed()
            with Live(
                self.layout,
                console=self.console,
                refresh_per_second=2,
                transient=False,
                vertical_overflow="visible",
            ) as live_context:
                self.live = live_context
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.live = None
        except Exception:
            if self.live:
                try:
                    self.live.stop()
                except Exception:  # pylint: disable=broad-except
                    pass
                finally:
                    self.live = None
            logging.error("Error occurred within Live context run:", exc_info=True)
            raise
