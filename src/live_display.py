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

    def __init__(
        self,
        num_workers: int,
        total_iterations: int,
        console: Console,
        console_log_level_value: int = logging.ERROR,  # Default to ERROR
    ):
        self.num_workers = num_workers
        self.total_iterations = total_iterations
        self.console = console
        self.console_log_level_value = console_log_level_value

        # Internal state tracking
        self._worker_statuses: Dict[int, WorkerDisplayStatus] = {
            i: "Initializing" for i in range(num_workers)
        }
        self._current_iteration = 0
        self._last_exploitability = "N/A"
        self._total_infosets = "0"
        self._last_iter_time = "N/A"
        self._min_worker_nodes_str: str = "N/A"
        self._total_log_size_str: str = "Calculating..."
        # Log records will be filtered by RichHandler's level, so deque can be larger if needed
        # for internal history, but display will be filtered.
        self._log_records: Deque[logging.LogRecord] = deque(
            maxlen=50
        )  # Max items in internal deque

        # Store previous renderables to compare for changes
        self._last_header_text_str: Optional[str] = None
        self._last_worker_table_title: Optional[str] = None
        self._last_worker_statuses_repr: Optional[str] = None
        self._last_log_panel_content_str: Optional[str] = None
        self._last_log_summary_str: Optional[str] = None

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
            Layout(name="main_content", ratio=1),  # Renamed for clarity
            Layout(name="footer", size=7),
        )
        # Main content now has two rows: main_row (workers/logs), and log_summary_panel
        layout["main_content"].split_column(
            Layout(name="main_row", ratio=1),
            Layout(name="log_summary_panel", size=3),
        )
        layout["main_row"].split_row(  # This was layout["main"] before
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
            level=self.console_log_level_value,  # Apply console log level here
            console=self.console,
        )
        for record in list(self._log_records):
            try:
                # RichHandler.level should filter records before render_message is even effectively used by Live.
                # However, if we were manually iterating and deciding to render, we'd check record.levelno
                if (
                    record.levelno >= handler.level
                ):  # Explicit check matching handler's behavior
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
            level=self.console_log_level_value,  # Apply console log level here
            console=self.console,
        )
        for record in list(
            self._log_records
        ):  # Iterate over a copy for thread safety if needed
            try:
                # RichHandler.level handles filtering. Records not meeting the level won't be rendered.
                # The RichHandler will internally filter based on its configured level.
                # We just feed it all records from our deque that LiveLogHandler (which also filters) gave us.
                render_result = handler.render_message(
                    record, message=record.getMessage()
                )  # This will return None or empty if filtered by level
                if render_result:  # Only add if RichHandler decided to render it
                    if isinstance(
                        render_result, Text
                    ) and not render_result.plain.endswith("\n"):
                        render_result.append("\n")
                    log_texts.append(render_result)
            except Exception as fmt_exc:  # pylint: disable=broad-except
                # Only append error if the record itself was at or above the display level
                if record.levelno >= handler.level:
                    log_texts.append(Text(f"Log Format Error: {fmt_exc}\n", style="red"))

        log_content = Group(*log_texts) if log_texts else Text("")
        return Panel(log_content, title="Recent Logs", border_style="dim", expand=True)

    def _generate_header_text_obj(self) -> Text:
        """Generates the Rich Text object for the header."""
        header_str = (
            f" Cambia CFR+ :: Iter: {self._current_iteration}/{self.total_iterations} "
            f":: Infosets: {self._total_infosets} :: Expl: {self._last_exploitability} "
            f":: Min Nodes: {self._min_worker_nodes_str} "
            f":: Last Iter: {self._last_iter_time} "
        )
        return Text(header_str, style="bold white on blue", justify="center")

    def _generate_log_summary_panel(self) -> Panel:
        """Generates the panel displaying log size summary."""
        return Panel(
            Text(self._total_log_size_str, justify="center"),
            title="Log Sizes",
            border_style="dim",
            expand=True,
        )

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
        # The comparison string should also be generated using the level-filtered RichHandler
        current_log_str = self._generate_log_panel_content_str()
        if self._last_log_panel_content_str != current_log_str:
            self.layout["logs"].update(self._generate_log_panel())
            self._last_log_panel_content_str = current_log_str
            layout_updated = True

        # Log Summary Panel
        current_log_summary_str = self._total_log_size_str  # Direct comparison
        if self._last_log_summary_str != current_log_summary_str:
            self.layout["log_summary_panel"].update(self._generate_log_summary_panel())
            self._last_log_summary_str = current_log_summary_str
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
                # Only refresh the live object if content actually changed or if progress bar itself needs update.
                # Rich's Progress bar updates independently if it's part of the layout.
                # The explicit self.live.update is more for the Panels.
                if content_changed:
                    self.live.update(self.layout, refresh=True)
                # If only progress bar updated, Rich handles it.
                # We could potentially call self.live.refresh() to force it, but usually not needed.
            except Exception as e:
                logging.error(
                    "Error explicitly refreshing Live display: %s", e, exc_info=True
                )

    def add_log_record(self, record: logging.LogRecord):
        """Adds a log record to the display queue and refreshes."""
        # LiveLogHandler already filters by its level.
        # We add it to our internal deque.
        self._log_records.append(record)
        # The refresh will cause _generate_log_panel to be called,
        # which uses RichHandler that filters by *its* level.
        self.refresh()

    def update_worker_status(self, worker_id: int, status: WorkerDisplayStatus):
        """Updates the status of a specific worker and refreshes."""
        changed = False
        if 0 <= worker_id < self.num_workers:
            if self._worker_statuses.get(worker_id) != status:
                self._worker_statuses[worker_id] = status
                changed = True
        else:
            logging.warning(
                "LiveDisplay: Invalid worker ID %s for status update.", worker_id
            )

        if changed:
            # Update min worker nodes if any worker status changed
            min_nodes = float("inf")
            found_stats = False
            for i in range(self.num_workers):
                s_info = self._worker_statuses.get(i)
                if isinstance(s_info, WorkerStats):
                    min_nodes = min(min_nodes, s_info.nodes_visited)
                    found_stats = True
                elif (
                    isinstance(s_info, tuple) and len(s_info) == 4
                ):  # Running status with node count
                    # s_info[3] is nodes_visited
                    min_nodes = min(min_nodes, s_info[3])
                    found_stats = True

            if found_stats and min_nodes != float("inf"):
                new_min_nodes_str = format_large_number(
                    int(min_nodes)
                )  # Ensure it's int for format_large_number
                if self._min_worker_nodes_str != new_min_nodes_str:
                    self._min_worker_nodes_str = new_min_nodes_str
                    # The refresh call below will pick up header change
            elif not found_stats:  # Or if all are inf (e.g. init)
                if (
                    self._min_worker_nodes_str != "N/A"
                ):  # Avoid unnecessary updates if already N/A
                    self._min_worker_nodes_str = "N/A"
            self.refresh()

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
                self.refresh()  # Refresh if header elements changed
            # No explicit refresh needed just for progress bar numbers, Rich handles that.
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
        if (
            self._current_iteration != iteration
        ):  # Though update_overall_progress usually handles this
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

        # Min worker nodes string is updated by update_worker_status, so header will pick it up
        # No need to explicitly check self._min_worker_nodes_str here for refresh trigger,
        # as any change to it would have already called self.refresh() via update_worker_status.

        status_text_for_progress_bar = f"Infosets: {infosets} | Expl: {exploitability} | Last T: {self._last_iter_time}"
        try:
            self.progress.update(
                self.iteration_task_id, status_text=status_text_for_progress_bar
            )
            if needs_refresh_for_header:
                self.refresh()  # Refresh if header elements changed
            # No explicit refresh needed just for progress bar status text, Rich handles that.
        except Exception as e:
            logging.error("Error updating progress bar stats field: %s", e, exc_info=True)

    def update_log_summary_display(
        self, current_logs_bytes: int, archived_logs_bytes: int
    ):
        """Updates the total log size string and refreshes the display."""
        total_bytes = current_logs_bytes + archived_logs_bytes

        # Use format_large_number for individual components as it's more aligned with existing display
        current_formatted = format_large_number(current_logs_bytes)
        archived_formatted = format_large_number(archived_logs_bytes)

        if total_bytes < 1024:
            size_str = f"{total_bytes} B"
        elif total_bytes < 1024**2:
            size_str = f"{total_bytes / 1024:.2f} KB"
        elif total_bytes < 1024**3:
            size_str = f"{total_bytes / (1024**2):.2f} MB"
        else:
            size_str = f"{total_bytes / (1024**3):.2f} GB"

        new_display_str = f"Active Logs: {current_formatted}B | Archived: {archived_formatted}B | Total Est: {size_str}"
        if self._total_log_size_str != new_display_str:
            self._total_log_size_str = new_display_str
            self.refresh()

    def start(self):
        """Starts the Rich Live display."""
        if not self.live:
            try:
                self._update_layout_if_changed()  # Ensure layout is current before starting
                self.live = Live(
                    self.layout,
                    console=self.console,
                    refresh_per_second=2,  # Lowered refresh rate might help with perceived flicker too
                    transient=False,  # Keep display after exit
                    vertical_overflow="visible",
                )
                self.live.start(refresh=True)  # Initial refresh
                logging.debug("Rich Live display started.")
            except Exception as e:
                logging.error("Failed to start Rich Live display: %s", e, exc_info=True)
                self.live = None

    def stop(self):
        """Stops the Rich Live display."""
        if self.live:
            try:
                self.live.stop()
                # self.console.print() # Avoid extra print if transient=False
                logging.debug("Rich Live display stopped.")
            except Exception as e:
                logging.error("Error stopping Rich Live display: %s", e, exc_info=True)
            finally:
                self.live = None

    def run(self, func, *args, **kwargs):
        """Runs a function within the Live context."""
        # If a Live instance is already active from a previous self.start(), stop it.
        if self.live:
            current_live_instance = self.live
            self.live = None  # Nullify self.live before stopping the old one
            try:
                current_live_instance.stop()
                # self.console.print() # Avoid extra print if transient=False
            except Exception as e:  # pylint: disable=broad-except
                logging.warning("Error stopping existing live instance in run(): %s", e)

        # Create a new Live context for the duration of `func`
        try:
            self._update_layout_if_changed()  # Ensure layout is current before starting new Live
            with Live(
                self.layout,
                console=self.console,
                refresh_per_second=2,
                transient=False,
                vertical_overflow="visible",
            ) as live_context:
                self.live = live_context  # Store the new active Live instance
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # self.live is implicitly stopped by exiting the 'with' block.
                    # We should nullify self.live here so self.stop() doesn't try to stop it again.
                    self.live = None
        except Exception:
            # If an error occurs setting up or during the Live context for func
            if self.live:  # If self.live was set but 'with' block failed or func failed
                try:
                    self.live.stop()  # Attempt to stop it if it's still around
                except Exception:  # pylint: disable=broad-except
                    pass  # Best effort
                finally:
                    self.live = None  # Ensure it's nullified
            logging.error("Error occurred within Live context run:", exc_info=True)
            raise
