"""src/cfr/training_loop_mixin.py"""

import logging
import multiprocessing
import multiprocessing.pool
import os
import threading
import time
import queue
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..utils import (
    WorkerResult,
    WorkerStats,
    format_large_number,
    format_infoset_count,
)
from .exceptions import GracefulShutdownException
from .worker import run_cfr_simulation_worker


# Use TYPE_CHECKING to avoid circular import for type hints where possible
if TYPE_CHECKING:
    # Keep types only needed for hints here
    from src.persistence import ReachProbDict
    from ..analysis_tools import AnalysisTools
    from ..live_display import LiveDisplayManager
    from ..log_archiver import LogArchiver
    from ..utils import PolicyDict, LogQueue as ProgressQueue
    from ..config import Config

    ArchiveQueueWorker = Any  # Placeholder type alias
    # WorkerDisplayStatus needs definition or import if complex
    WorkerDisplayStatus = Any  # Placeholder


logger = logging.getLogger(__name__)


class CFRTrainingLoopMixin:
    """Handles the CFR+ training loop, parallelism, progress, and scheduling."""

    # --- Type Hinting for Attributes Expected from Main Class ---
    # Use string forward references if needed to avoid direct imports at top level
    config: "Config"
    analysis: "AnalysisTools"
    shutdown_event: "threading.Event"
    current_iteration: int
    exploitability_results: List[Tuple[int, float]]
    regret_sum: "PolicyDict"
    strategy_sum: "PolicyDict"
    reach_prob_sum: "ReachProbDict"
    progress_queue: Optional["ProgressQueue"]
    archive_queue: Optional["ArchiveQueueWorker"]
    run_log_dir: Optional[str]
    run_timestamp: Optional[str]
    live_display_manager: Optional["LiveDisplayManager"]
    log_archiver_global_ref: Optional["LogArchiver"]
    # Methods expected
    load_data: Callable[..., Any]
    save_data: Callable[..., Any]
    _merge_local_updates: Callable[[List[Optional["WorkerResult"]]], None]
    compute_average_strategy: Callable[..., Optional["PolicyDict"]]
    # Internal display state
    _last_exploit_str: str = "N/A"
    _total_infosets_str: str = "0"
    # Timing / Periodic updates
    _total_run_time_start: float = 0.0
    _last_log_size_update_time: float = 0.0
    _last_exploitability_calc_time: float = 0.0
    # Use Any for WorkerDisplayStatus if definition is complex or elsewhere
    _worker_statuses: Dict[int, Any] = {}

    def __init__(self, *args, **kwargs):
        """Initializes attributes specific to the training loop."""
        self._last_log_size_update_time = 0.0
        self._total_run_time_start = 0.0
        self._last_exploitability_calc_time = 0.0  # Initialize time tracking
        self._worker_statuses = {}

    def _shutdown_pool(self, pool: Optional[multiprocessing.pool.Pool]):
        """Gracefully shuts down the multiprocessing pool."""
        if pool:
            # Check if pool is still running before interacting
            # Note: _state attribute is internal, but commonly checked
            pool_running = getattr(pool, "_state", -1) == multiprocessing.pool.RUN

            if pool_running:
                logger.info("Terminating worker pool...")
                try:
                    pool.terminate()
                    # Add a small delay to allow worker processes to potentially react to SIGTERM
                    time.sleep(0.5)  # 500ms delay
                    pool.join()  # Join AFTER terminate, no timeout argument needed or accepted
                    logger.info("Worker pool terminated and joined.")
                except ValueError:
                    logger.warning(
                        "Attempted to terminate/join an already closed pool state."
                    )
                except (
                    Exception
                ) as e_pool_shutdown:  # JUSTIFIED: Pool shutdown errors during cleanup; log and continue
                    logger.error(
                        "Exception during pool shutdown: %s",
                        e_pool_shutdown,
                        exc_info=True,
                    )
            else:
                logger.info("Worker pool already terminated or closed.")

    def _try_update_log_size_display(self):
        """Attempts to update the log size display periodically."""
        # Add checks for attribute existence before accessing
        if not hasattr(self, "live_display_manager") or not self.live_display_manager:
            return
        if (
            not hasattr(self, "log_archiver_global_ref")
            or not self.log_archiver_global_ref
        ):
            return
        if not hasattr(self, "config") or not hasattr(self.config, "logging"):
            return
        if (
            not hasattr(self.config.logging, "log_size_update_interval_sec")
            or self.config.logging.log_size_update_interval_sec <= 0
        ):
            return

        current_time = time.time()
        if (
            current_time - self._last_log_size_update_time
            > self.config.logging.log_size_update_interval_sec
        ):
            try:
                current_size, archived_size = (
                    self.log_archiver_global_ref.get_total_log_size_info()
                )
                # Check if manager still exists and has the method
                if hasattr(self.live_display_manager, "update_log_summary_display"):
                    self.live_display_manager.update_log_summary_display(
                        current_size, archived_size
                    )
                self._last_log_size_update_time = current_time
            except AttributeError as e_attr:
                logger.debug(
                    "Skipping log size update due to missing attribute: %s", e_attr
                )
            except (
                Exception
            ) as e_log_size:  # JUSTIFIED: Non-critical display update; must not crash training
                logger.error(
                    "Error updating log size display periodically: %s",
                    e_log_size,
                    exc_info=True,
                )

    def _calculate_exploitability_if_needed(self, iteration: int):
        """Calculates exploitability if the iteration or time interval is met."""
        # Check if methods exist before calling
        if not (
            hasattr(self, "compute_average_strategy")
            and callable(self.compute_average_strategy)
            and hasattr(self, "analysis")
            and hasattr(self.analysis, "calculate_exploitability")
            and callable(self.analysis.calculate_exploitability)
        ):
            logger.error("Missing methods for exploitability calculation.")
            return

        # Update main process status via display manager
        display = getattr(self, "live_display_manager", None)
        if display and hasattr(display, "update_main_process_status"):
            display.update_main_process_status("Computing average strategy...")

        try:
            # Check for shutdown BEFORE computing average strategy
            if self.shutdown_event.is_set():
                logger.warning(
                    "Shutdown detected before computing average strategy for exploitability."
                )
                raise GracefulShutdownException("Shutdown during exploitability prep")

            current_avg_strategy = self.compute_average_strategy()

            if current_avg_strategy is not None:
                logger.info("Calculating exploitability at iteration %d...", iteration)
                # Update status before starting calculation
                if display and hasattr(display, "update_main_process_status"):
                    display.update_main_process_status("Calculating exploitability...")

                # Check for shutdown BEFORE starting long calculation
                if self.shutdown_event.is_set():
                    logger.warning(
                        "Shutdown detected before starting exploitability calculation."
                    )
                    raise GracefulShutdownException("Shutdown during exploitability prep")

                exploit_start_time = time.time()
                try:
                    # Pass display manager to exploitability calculation
                    exploit = self.analysis.calculate_exploitability(
                        current_avg_strategy, self.config, display  # Pass display manager
                    )
                    if hasattr(self, "exploitability_results") and isinstance(
                        self.exploitability_results, list
                    ):
                        self.exploitability_results.append((iteration, exploit))
                    self._last_exploit_str = (
                        f"{exploit:.4f}" if exploit != float("inf") else "N/A"
                    )
                    logger.info(
                        "Exploitability: %s (took %.2fs)",
                        self._last_exploit_str,
                        time.time() - exploit_start_time,
                    )
                    self._last_exploitability_calc_time = time.time()  # Update timestamp
                except (
                    GracefulShutdownException
                ):  # Catch if raised internally by calculate_exploitability
                    raise  # Re-raise to be caught by main loop handler
                except (
                    Exception
                ):  # JUSTIFIED: Exploitability is diagnostic only; must not crash training
                    logger.exception(
                        "Error calculating exploitability at iter %d.", iteration
                    )
                    self._last_exploit_str = "Calc FAILED"
                finally:
                    # Reset status after calculation finishes or fails
                    if display and hasattr(display, "update_main_process_status"):
                        display.update_main_process_status("Idle / Waiting...")
            else:
                logger.warning(
                    "Could not compute avg strategy for exploitability at iter %d.",
                    iteration,
                )
                self._last_exploit_str = "N/A (Avg Err)"
                if display and hasattr(display, "update_main_process_status"):
                    display.update_main_process_status("Idle / Waiting...")

        except GracefulShutdownException:  # Catch if raised during avg strategy calc
            raise  # Re-raise to be caught by main loop handler
        except (
            Exception
        ) as e_avg_strat:  # JUSTIFIED: Exploitability is diagnostic only; must not crash training
            logger.exception(
                "Error computing average strategy for exploitability: %s", e_avg_strat
            )
            self._last_exploit_str = "N/A (Avg Err)"
            if display and hasattr(display, "update_main_process_status"):
                display.update_main_process_status("Idle / Waiting...")

    def train(self, num_iterations: Optional[int] = None):
        """Runs the main CFR+ training loop, potentially in parallel."""

        # Safely access config attributes
        cfr_config = getattr(self.config, "cfr_training", None)
        if not cfr_config:
            logger.critical("CFRTrainingConfig not found in main config. Cannot train.")
            return

        total_iterations_to_run = (
            num_iterations
            if num_iterations is not None
            else getattr(cfr_config, "num_iterations", 0)
        )
        self.current_iteration = getattr(
            self, "current_iteration", 0
        )  # Ensure it's initialized
        last_completed_iteration = self.current_iteration
        start_iter_num = last_completed_iteration + 1
        end_iter_num = last_completed_iteration + total_iterations_to_run

        exploitability_iter_interval = getattr(cfr_config, "exploitability_interval", 0)
        exploitability_time_interval = getattr(
            cfr_config, "exploitability_interval_seconds", 0
        )
        num_workers = getattr(cfr_config, "num_workers", 1)
        save_interval = getattr(cfr_config, "save_interval", 0)
        progress_q_local = getattr(self, "progress_queue", None)
        archive_q_local = getattr(self, "archive_queue", None)
        run_log_dir_local = getattr(self, "run_log_dir", None)
        run_timestamp_local = getattr(self, "run_timestamp", None)
        display = getattr(self, "live_display_manager", None)
        log_sim_traces = getattr(self.config.logging, "log_simulation_traces", False)

        logger.info(
            "Starting CFR+ training from iteration %d up to %d (%d workers).",
            start_iter_num,
            end_iter_num,
            num_workers,
        )
        logger.info(
            "Exploitability Check: Iter=%d, Time=%ds",
            exploitability_iter_interval,
            exploitability_time_interval,
        )
        if total_iterations_to_run <= 0:
            logger.warning(
                "Number of iterations to run (%d) must be positive. Exiting.",
                total_iterations_to_run,
            )
            return

        if not display:
            logger.warning(
                "LiveDisplayManager not provided. Console output will be minimal."
            )
        if not run_log_dir_local or not run_timestamp_local:
            logger.warning(
                "Run log directory/timestamp not set. Worker logging might fail."
            )

        self._total_run_time_start = time.time()
        self._last_log_size_update_time = time.time()
        # Initialize last calc time to start to allow first calc if interval is set
        self._last_exploitability_calc_time = (
            0.0 if exploitability_time_interval > 0 else time.time()
        )
        pool: Optional[multiprocessing.pool.Pool] = None
        self._worker_statuses = {i: "Initializing" for i in range(num_workers)}

        # Initial display update
        if (
            display
            and hasattr(display, "update_worker_status")
            and hasattr(display, "update_stats")
            and hasattr(display, "update_overall_progress")
            and hasattr(display, "update_main_process_status")  # Check for new method
        ):
            for i in range(num_workers):
                display.update_worker_status(i, "Idle")
            self._try_update_log_size_display()
            # Ensure regret_sum exists before calculating length
            regret_sum_len = len(self.regret_sum) if hasattr(self, "regret_sum") else 0
            self._total_infosets_str = format_infoset_count(regret_sum_len)
            display.update_stats(
                iteration=self.current_iteration,
                infosets=self._total_infosets_str,
                exploitability=self._last_exploit_str,
            )
            display.update_overall_progress(self.current_iteration)
            display.update_main_process_status("Idle / Waiting...")  # Initial status

        try:
            for t in range(start_iter_num, end_iter_num + 1):
                # Check shutdown at start of loop
                if self.shutdown_event.is_set():
                    logger.warning("Shutdown detected before starting iteration %d.", t)
                    raise GracefulShutdownException("Shutdown detected before iteration")

                iter_start_time = time.time()
                self.current_iteration = t
                min_nodes_this_iter_overall = float("inf")
                self._try_update_log_size_display()

                # Update display for start of iteration
                if (
                    display
                    and hasattr(display, "update_worker_status")
                    and hasattr(display, "update_stats")
                    and hasattr(display, "update_overall_progress")
                    and hasattr(display, "update_main_process_status")
                ):
                    for i in range(num_workers):
                        status_to_set: Any = (
                            "Queued" if num_workers > 1 else "Starting",
                            0,
                            0,
                            0,
                            0,
                        )  # Use Any for WorkerDisplayStatus
                        self._worker_statuses[i] = status_to_set
                        display.update_worker_status(i, self._worker_statuses[i])
                    regret_sum_len = (
                        len(self.regret_sum) if hasattr(self, "regret_sum") else 0
                    )
                    current_total_infosets_str = format_infoset_count(regret_sum_len)
                    if self._total_infosets_str != current_total_infosets_str:
                        self._total_infosets_str = current_total_infosets_str
                    display.update_overall_progress(t)
                    display.update_stats(
                        iteration=t,
                        infosets=self._total_infosets_str,
                        exploitability=self._last_exploit_str,
                    )
                    display.update_main_process_status(
                        "Running Simulations..."
                    )  # Status during sims
                else:  # Still update internal status
                    for i in range(num_workers):
                        self._worker_statuses[i] = (
                            "Queued" if num_workers > 1 else "Starting",
                            0,
                            0,
                            0,
                            0,
                        )

                # Prepare snapshot
                try:
                    regret_snapshot = (
                        dict(self.regret_sum) if hasattr(self, "regret_sum") else {}
                    )
                except Exception as e_snap:
                    logger.exception(
                        "Failed to prepare regret snapshot for iter %d: %s", t, e_snap
                    )
                    raise RuntimeError(
                        f"Could not prepare regret snapshot for iter {t}"
                    ) from e_snap

                worker_args_list = []
                if run_log_dir_local and run_timestamp_local:
                    for i in range(num_workers):
                        worker_args_list.append(
                            (
                                t,
                                self.config,
                                regret_snapshot,
                                progress_q_local,
                                archive_q_local,
                                i,  # worker_id
                                run_log_dir_local,
                                run_timestamp_local,
                            )
                        )
                else:  # Should not happen if setup succeeded
                    logger.error("run_log_dir or run_timestamp missing for worker args!")
                    raise RuntimeError("Missing log dir/timestamp for workers")

                results: List[Optional[WorkerResult]] = [None] * num_workers
                sim_failed_count = 0
                successful_sim_count = 0

                # --- Execute simulations ---
                if num_workers == 1:
                    # Sequential execution
                    logger.debug("Running simulation sequentially for iter %d...", t)
                    try:
                        # Check shutdown BEFORE running simulation
                        if self.shutdown_event.is_set():
                            raise GracefulShutdownException(
                                "Shutdown before sequential worker"
                            )
                        results[0] = run_cfr_simulation_worker(worker_args_list[0])
                        if results[0] and isinstance(results[0], WorkerResult):
                            self._worker_statuses[0] = results[0].stats
                            min_nodes_this_iter_overall = min(
                                min_nodes_this_iter_overall,
                                results[0].stats.nodes_visited,
                            )
                            if results[0].stats.error_count > 0:
                                sim_failed_count += 1
                            else:
                                successful_sim_count += 1
                            if log_sim_traces and self.analysis:
                                self.analysis.log_simulation_trace(
                                    {
                                        "metadata": {
                                            "iteration": t,
                                            "worker_id": 0,
                                            "final_utility": results[0].final_utility,
                                            "stats": results[0].stats.__dict__,
                                        },
                                        "history": results[0].simulation_nodes,
                                    }
                                )
                        else:
                            self._worker_statuses[0] = "Execution Failed"
                            sim_failed_count += 1
                    except GracefulShutdownException:  # Catch shutdown during run
                        raise  # Re-raise
                    except (
                        Exception
                    ) as e_seq:  # JUSTIFIED: Worker failures are tracked; training can continue with failed worker count
                        logger.exception(
                            "Sequential simulation worker failed iter %d: %s", t, e_seq
                        )
                        self._worker_statuses[0] = "Execution Failed"
                        sim_failed_count += 1
                else:
                    # Parallel execution
                    if not pool:
                        logger.info("Creating worker pool (size %d)...", num_workers)
                        pool = multiprocessing.Pool(processes=num_workers)

                    async_results = pool.map_async(
                        run_cfr_simulation_worker, worker_args_list
                    )
                    try:
                        timeout_interval = 0.5  # seconds
                        while not async_results.ready():
                            # Check shutdown frequently while waiting
                            if self.shutdown_event.is_set():
                                logger.warning(
                                    "Shutdown detected while waiting for workers iter %d. Terminating pool.",
                                    t,
                                )
                                self._shutdown_pool(pool)  # Terminate pool on shutdown
                                pool = None  # Mark pool as stopped
                                raise GracefulShutdownException(
                                    "Shutdown during worker execution"
                                )

                            async_results.wait(
                                timeout=timeout_interval
                            )  # Wait with timeout

                            # --- Progress Queue Processing ---
                            if progress_q_local:
                                while True:  # Drain queue
                                    try:
                                        update = progress_q_local.get_nowait()
                                        worker_id_upd, cur_d, max_d, nodes, min_d_bt = (
                                            update
                                        )
                                        if 0 <= worker_id_upd < num_workers:
                                            current_status = self._worker_statuses.get(
                                                worker_id_upd
                                            )
                                            status_str = "Running"
                                            if isinstance(current_status, tuple):
                                                status_str = current_status[
                                                    0
                                                ]  # Preserve state string

                                            new_status = (
                                                status_str,
                                                cur_d,
                                                max_d,
                                                nodes,
                                                min_d_bt,
                                            )
                                            # Update only if different to avoid excessive refreshes
                                            if current_status != new_status:
                                                self._worker_statuses[worker_id_upd] = (
                                                    new_status
                                                )
                                                if display:
                                                    display.update_worker_status(
                                                        worker_id_upd, new_status
                                                    )
                                                min_nodes_this_iter_overall = min(
                                                    min_nodes_this_iter_overall, nodes
                                                )

                                    except queue.Empty:
                                        break  # No more updates currently
                                    except (
                                        Exception
                                    ) as pqe:  # JUSTIFIED: Progress queue errors are display-only; must not crash training
                                        logger.debug(
                                            "Error processing progress queue: %s", pqe
                                        )
                        # Get final results ONLY if shutdown wasn't triggered
                        if not self.shutdown_event.is_set():
                            results = async_results.get()
                        else:
                            # If shutdown was triggered, we already raised GracefulShutdownException
                            # Results are irrelevant. Ensure loop terminates.
                            # This path should technically not be reached due to the raise above.
                            logger.warning(
                                "Shutdown occurred before getting worker results."
                            )
                            raise GracefulShutdownException(
                                "Shutdown while waiting for results"
                            )

                        # Process final results from workers
                        for i, res in enumerate(results):
                            if isinstance(res, WorkerResult):
                                self._worker_statuses[i] = res.stats
                                if res.stats.error_count == 0:
                                    successful_sim_count += 1
                                    min_nodes_this_iter_overall = min(
                                        min_nodes_this_iter_overall,
                                        res.stats.nodes_visited,
                                    )
                                    if log_sim_traces and self.analysis:
                                        self.analysis.log_simulation_trace(
                                            {
                                                "metadata": {
                                                    "iteration": t,
                                                    "worker_id": i,
                                                    "final_utility": res.final_utility,
                                                    "stats": res.stats.__dict__,
                                                },
                                                "history": res.simulation_nodes,
                                            }
                                        )
                                else:  # Error occurred
                                    sim_failed_count += 1
                                    logger.warning(
                                        "Worker %d reported %d errors.",
                                        i,
                                        res.stats.error_count,
                                    )
                            elif (
                                res is None
                            ):  # Worker might have crashed or returned None
                                self._worker_statuses[i] = "Failed (None Result)"
                                sim_failed_count += 1
                                logger.error(
                                    "Worker %d returned None result for iter %d.", i, t
                                )
                            else:  # Unexpected return type
                                self._worker_statuses[i] = (
                                    f"Failed (Type {type(res).__name__})"
                                )
                                sim_failed_count += 1
                                logger.error(
                                    "Worker %d returned unexpected result type %s for iter %d.",
                                    i,
                                    type(res).__name__,
                                    t,
                                )

                    except (
                        GracefulShutdownException,
                        KeyboardInterrupt,
                    ):  # Catch shutdown here too
                        logger.warning(
                            "Shutdown/Interrupt received during worker pool execution iter %d. Ensuring pool shutdown.",
                            t,
                        )
                        self._shutdown_pool(pool)
                        pool = None
                        raise  # Re-raise
                    except (
                        Exception
                    ) as e_pool:  # JUSTIFIED: Pool errors are critical infrastructure failures; re-raised after cleanup
                        logger.exception(
                            "Error during worker pool execution iter %d: %s", t, e_pool
                        )
                        self._shutdown_pool(pool)
                        pool = None
                        raise  # Propagate error up

                # --- End Simulation Execution ---

                # Check shutdown AFTER simulations
                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown after worker simulations")

                # Update display min nodes
                if display and hasattr(display, "_min_worker_nodes_overall_str"):
                    if min_nodes_this_iter_overall != float("inf"):
                        display._min_worker_nodes_overall_str = format_large_number(
                            int(min_nodes_this_iter_overall)
                        )
                    else:
                        display._min_worker_nodes_overall_str = "N/A"

                # Check simulation results and merge
                if sim_failed_count == num_workers and num_workers > 0:
                    logger.error(
                        "All %d simulations failed for iteration %d. Skipping merge.",
                        num_workers,
                        t,
                    )
                elif sim_failed_count > 0:
                    logger.warning(
                        "%d/%d simulations reported errors for iter %d.",
                        sim_failed_count,
                        num_workers,
                        t,
                    )

                valid_results_for_merge = [
                    res
                    for res in results
                    if isinstance(res, WorkerResult) and res.stats.error_count == 0
                ]
                if (
                    not valid_results_for_merge
                    and successful_sim_count == 0
                    and num_workers > 0
                ):
                    logger.warning(
                        "No successful worker results for iter %d. Merge will process no data.",
                        t,
                    )

                # Check shutdown BEFORE merge
                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before merge")

                if valid_results_for_merge:
                    logger.debug(
                        "Merging results from %d successful workers for iter %d...",
                        len(valid_results_for_merge),
                        t,
                    )
                    # Update status BEFORE merging
                    if display and hasattr(display, "update_main_process_status"):
                        display.update_main_process_status("Merging Worker Results...")

                    merge_start_time = time.time()
                    try:
                        if hasattr(self, "_merge_local_updates") and callable(
                            self._merge_local_updates
                        ):
                            self._merge_local_updates(valid_results_for_merge)
                            logger.debug(
                                "Iter %d merge took %.3fs",
                                t,
                                time.time() - merge_start_time,
                            )
                        else:
                            logger.error(
                                "Merge required but _merge_local_updates method missing!"
                            )
                    except (
                        Exception
                    ):  # JUSTIFIED: Merge errors are critical but recoverable; skip iteration and continue
                        logger.exception("Error merging results iter %d.", t)
                        if display and hasattr(display, "update_stats"):
                            display.update_stats(
                                iteration=t,
                                infosets=self._total_infosets_str,
                                exploitability="Merge FAILED",
                                last_iter_time=time.time() - iter_start_time,
                            )
                        # Reset status after merge error
                        if display and hasattr(display, "update_main_process_status"):
                            display.update_main_process_status("Idle / Waiting...")
                        continue  # Skip rest of iteration on merge failure
                    finally:
                        # Reset status after merge (even if successful)
                        if display and hasattr(display, "update_main_process_status"):
                            display.update_main_process_status("Idle / Waiting...")

                # Post-iteration updates
                iter_time = time.time() - iter_start_time
                regret_sum_len = (
                    len(self.regret_sum) if hasattr(self, "regret_sum") else 0
                )
                self._total_infosets_str = format_infoset_count(regret_sum_len)

                # Check shutdown BEFORE exploitability
                if self.shutdown_event.is_set():
                    raise GracefulShutdownException(
                        "Shutdown before exploitability check"
                    )

                # --- Exploitability Calculation (Iteration and Time based) ---
                should_calculate_exploitability = False
                trigger_reason = ""

                # Check iteration interval
                if (
                    exploitability_iter_interval > 0
                    and t % exploitability_iter_interval == 0
                ):
                    should_calculate_exploitability = True
                    trigger_reason = "iteration interval"

                # Check time interval (if not already triggered by iteration)
                current_time = time.time()
                if (
                    not should_calculate_exploitability
                    and exploitability_time_interval > 0
                    and (current_time - self._last_exploitability_calc_time)
                    >= exploitability_time_interval
                ):
                    should_calculate_exploitability = True
                    trigger_reason = "time interval"

                if should_calculate_exploitability:
                    logger.info(
                        "Triggering exploitability calculation at iteration %d (Reason: %s)",
                        t,
                        trigger_reason,
                    )
                    if hasattr(self, "_calculate_exploitability_if_needed"):
                        self._calculate_exploitability_if_needed(
                            t
                        )  # This now raises GracefulShutdownException
                    else:
                        logger.error("Cannot calculate exploitability: method missing.")
                # --- End Exploitability Calculation ---

                # Check shutdown AFTER exploitability
                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown after exploitability check")

                # Update display
                if display and hasattr(display, "update_stats"):
                    display.update_stats(
                        iteration=t,
                        infosets=self._total_infosets_str,
                        exploitability=self._last_exploit_str,
                        last_iter_time=iter_time,
                    )
                    if hasattr(display, "update_worker_status"):
                        for i_disp in range(num_workers):
                            final_status_iter_end = self._worker_statuses.get(i_disp)
                            if final_status_iter_end:
                                display.update_worker_status(
                                    i_disp, final_status_iter_end
                                )

                # Check shutdown BEFORE save
                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before save interval")

                # Periodic save
                if save_interval > 0 and t % save_interval == 0:
                    logger.info("Saving progress at iteration %d...", t)
                    if hasattr(self, "save_data") and callable(self.save_data):
                        self.save_data()
                    else:
                        logger.error("Cannot save data: save_data method missing.")

                last_completed_iteration = (
                    t  # Update last successfully completed iteration
                )

            logger.info("Training loop completed %d iterations.", total_iterations_to_run)

        except (GracefulShutdownException, KeyboardInterrupt) as shutdown_exc:
            exception_type = type(shutdown_exc).__name__
            logger.warning(
                "Shutdown/Interrupt (%s) caught in train loop. Saving progress...",
                exception_type,
            )
            if hasattr(self, "shutdown_event") and not self.shutdown_event.is_set():
                self.shutdown_event.set()
            self._perform_emergency_save(
                last_completed_iteration,  # Pass last known good iteration
                start_iter_num,
                exception_type,
                pool_to_shutdown=pool,
            )
            raise GracefulShutdownException(
                f"{exception_type} processed in train loop"
            ) from shutdown_exc

        except Exception as main_loop_e:
            logger.exception("Unhandled exception in main training loop:")
            self._perform_emergency_save(
                last_completed_iteration,  # Pass last known good iteration
                start_iter_num,
                type(main_loop_e).__name__,
                pool_to_shutdown=pool,
            )
            raise

        # --- Normal Completion ---
        # Check if shutdown event was set *just* as the loop finished
        if hasattr(self, "shutdown_event") and not self.shutdown_event.is_set():
            end_time = time.time()
            total_completed_in_run = self.current_iteration - (
                start_iter_num - 1
            )  # Use start_iter_num
            logger.info(
                "Training loop finished normally after %d iterations.",
                total_completed_in_run,
            )
            if self._total_run_time_start > 0:
                logger.info(
                    "Total training time this run: %.2f seconds.",
                    end_time - self._total_run_time_start,
                )
            logger.info("Final iteration count: %d", self.current_iteration)

            # Final exploitability calculation (if not just calculated)
            final_calc_needed = True
            if (
                exploitability_iter_interval > 0
                and self.current_iteration > 0
                and self.current_iteration % exploitability_iter_interval == 0
            ):
                final_calc_needed = False
            if not final_calc_needed and exploitability_time_interval > 0:
                if (
                    time.time() - self._last_exploitability_calc_time
                ) < 1.0:  # Allow small tolerance
                    final_calc_needed = False

            if final_calc_needed:
                logger.info("Performing final exploitability calculation...")
                if hasattr(self, "_calculate_exploitability_if_needed"):
                    self._calculate_exploitability_if_needed(self.current_iteration)
                else:
                    logger.error("Cannot calculate final exploitability: method missing.")

            # Final display update
            regret_sum_len_final = (
                len(self.regret_sum) if hasattr(self, "regret_sum") else 0
            )
            self._total_infosets_str = format_infoset_count(regret_sum_len_final)
            if display and hasattr(display, "update_stats"):
                display.update_stats(
                    iteration=self.current_iteration,
                    infosets=self._total_infosets_str,
                    exploitability=self._last_exploit_str,
                    last_iter_time=None,
                )
                if hasattr(display, "update_worker_status"):
                    for i_final_disp in range(num_workers):
                        final_status = self._worker_statuses.get(i_final_disp, "Finished")
                        display.update_worker_status(i_final_disp, final_status)

            # Final save & summary
            logger.info("Performing final save...")
            self._try_update_log_size_display()
            if hasattr(self, "save_data") and callable(self.save_data):
                self.save_data()
            else:
                logger.error("Cannot perform final save: save_data method missing.")
            logger.info("Final data saved.")
            if hasattr(self, "_write_run_summary") and callable(self._write_run_summary):
                self._write_run_summary()
        else:  # Shutdown event was set during normal completion
            logger.warning(
                "Shutdown signal received during normal completion. Final save/summary skipped (emergency save should cover)."
            )

    def _perform_emergency_save(
        self,
        last_completed_iteration: int,
        start_iter_num_this_run: int,  # Keep this arg for consistency
        reason: str,
        pool_to_shutdown: Optional[multiprocessing.pool.Pool],
    ):
        """Attempts to save state after an interruption or error."""
        logger.warning("Attempting emergency save due to: %s", reason)
        self._shutdown_pool(pool_to_shutdown)  # Ensure workers stopped

        # Save based on the last fully completed iteration recorded before interruption
        iter_to_save_as_completed = last_completed_iteration

        logger.info(
            "Attempting to save data reflecting completion of iteration %d.",
            iter_to_save_as_completed,
        )
        # Temporarily set current_iteration for the save_data method
        original_current_iter_val = self.current_iteration
        self.current_iteration = iter_to_save_as_completed

        try:
            if hasattr(self, "save_data") and callable(self.save_data):
                self._try_update_log_size_display()
                self.save_data()
                logger.info(
                    "Emergency save completed for iteration %d.", self.current_iteration
                )
            else:
                logger.error("Emergency save failed: save_data method not found.")
        except (
            Exception
        ) as e_save:  # JUSTIFIED: Emergency save is last-ditch effort; log failure and continue cleanup
            logger.exception("Emergency save failed: %s", e_save)
        finally:
            # Restore actual current iteration number (might be inaccurate if interrupt mid-iter)
            self.current_iteration = original_current_iter_val
            # Write summary regardless of save success
            if hasattr(self, "_write_run_summary") and callable(self._write_run_summary):
                logger.info("Writing run summary after emergency save attempt.")
                self._write_run_summary()

    def _write_run_summary(self):
        """Writes a summary log at the end of a training run."""
        # Safe access to attributes
        run_log_dir = getattr(self, "run_log_dir", None)
        run_timestamp = getattr(self, "run_timestamp", None)
        config = getattr(self, "config", None)
        exploit_results = getattr(self, "exploitability_results", [])
        regret_sum = getattr(self, "regret_sum", {})
        worker_statuses = getattr(self, "_worker_statuses", {})
        last_exploit_str = getattr(self, "_last_exploit_str", "N/A")
        current_iter = getattr(self, "current_iteration", -1)
        total_run_start = getattr(self, "_total_run_time_start", 0)
        num_workers = (
            getattr(config.cfr_training, "num_workers", 0)
            if config and hasattr(config, "cfr_training")
            else 0
        )

        if not run_log_dir or not run_timestamp or not config:
            logger.error(
                "Cannot write run summary: Missing required attributes (run_log_dir, run_timestamp, config)."
            )
            return

        summary_file_path = os.path.join(
            run_log_dir,
            f"{config.logging.log_file_prefix}_run_{run_timestamp}-summary.log",
        )
        logger.info("Writing run summary to: %s", summary_file_path)

        try:
            with open(summary_file_path, "w", encoding="utf-8") as f:
                f.write("--- Cambia CFR+ Training Run Summary ---\n")
                f.write(f"Run Timestamp: {run_timestamp}\n")
                f.write(f"Run Directory: {run_log_dir}\n")
                config_path = getattr(config, "_source_path", "N/A")
                f.write(f"Config File: {config_path}\n")
                f.write(f"Last Attempted/Reached Iteration: {max(0, current_iter)}\n")
                total_time = time.time() - total_run_start if total_run_start > 0 else 0
                f.write(f"Total Run Time (this execution): {total_time:.2f} seconds\n")
                f.write(f"Final Exploitability (last calculated): {last_exploit_str}\n")
                f.write(f"Total Unique Infosets Reached: {len(regret_sum):,}\n")
                f.write(f"Number of Workers: {num_workers}\n")
                f.write("\n--- Worker Summary (Final Status) ---\n")

                if num_workers > 0 and worker_statuses:
                    # (Ensure WorkerStats is imported or use Any if needed)
                    total_nodes = 0
                    max_depth_overall = 0
                    min_depth_bt_overall = float("inf")
                    total_warnings = 0
                    total_errors = 0
                    completed_workers_with_stats = 0
                    failed_workers_marked = 0
                    for i in range(num_workers):
                        status_info = worker_statuses.get(i)
                        status_line = f"Worker {i}: "
                        if isinstance(status_info, WorkerStats):  # Use imported type
                            status_line += f"Completed (MaxD:{status_info.max_depth}, Nodes:{status_info.nodes_visited:,}, MinDBT:{status_info.min_depth_after_bottom_out}, Warn:{status_info.warning_count}, Err:{status_info.error_count})"
                            total_nodes += status_info.nodes_visited
                            max_depth_overall = max(
                                max_depth_overall, status_info.max_depth
                            )
                            if status_info.min_depth_after_bottom_out > 0:
                                min_depth_bt_overall = min(
                                    min_depth_bt_overall,
                                    status_info.min_depth_after_bottom_out,
                                )
                            total_warnings += status_info.warning_count
                            total_errors += status_info.error_count
                            if status_info.error_count == 0:
                                completed_workers_with_stats += 1
                            else:
                                failed_workers_marked += 1
                        elif isinstance(status_info, str):
                            status_line += status_info
                            if "Error" in status_info or "Fail" in status_info:
                                failed_workers_marked += 1
                        elif isinstance(status_info, tuple) and len(status_info) == 5:
                            state, cur_d, max_d, nodes, min_d_bt = status_info
                            status_line += f"Stopped (State:{state}, CurD:{cur_d}, MaxD:{max_d}, Nodes:{nodes:,}, MinDBT:{min_d_bt})"
                            failed_workers_marked += 1  # Assume stopped is failed
                        else:
                            status_line += (
                                f"Unknown Final State ({type(status_info).__name__})"
                            )
                            failed_workers_marked += 1
                        f.write(status_line + "\n")

                    f.write("\n--- Aggregated Worker Stats ---\n")
                    f.write(f"Successful Workers: {completed_workers_with_stats}\n")
                    f.write(f"Failed/Stopped Workers: {failed_workers_marked}\n")
                    f.write(f"Total Nodes Visited (Successful): {total_nodes:,}\n")
                    f.write(f"Overall Max Depth Reached: {max_depth_overall}\n")
                    min_dbt_str = (
                        str(int(min_depth_bt_overall))
                        if min_depth_bt_overall != float("inf")
                        else "N/A"
                    )
                    f.write(f"Overall Min Depth After Bottom Out: {min_dbt_str}\n")
                    f.write(f"Total Warnings (Successful): {total_warnings}\n")
                    f.write(f"Total Errors (Successful): {total_errors}\n")
                else:
                    f.write(
                        "Worker status information not available or 0 workers configured.\n"
                    )

                f.write("\n--- Exploitability History ---\n")
                if exploit_results:
                    for iter_num_exploit, exploit_val in exploit_results:
                        f.write(f"Iteration {iter_num_exploit}: {exploit_val:.6f}\n")
                else:
                    f.write("No exploitability data recorded.\n")
                f.write("\n--- End Summary ---\n")
        except IOError as e_io:
            logger.error(
                "Failed to write run summary file %s: %s", summary_file_path, e_io
            )
        except (
            Exception
        ) as e_summary:  # JUSTIFIED: Summary write is post-training cleanup; must not crash shutdown
            logger.exception("Unexpected error writing run summary: %s", e_summary)
