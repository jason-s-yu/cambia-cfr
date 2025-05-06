"""src/cfr/training_loop_mixin.py"""

import logging
import multiprocessing
import multiprocessing.pool
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..analysis_tools import AnalysisTools
from ..config import Config
from ..live_display import LiveDisplayManager, WorkerDisplayStatus
from ..log_archiver import LogArchiver


from ..utils import (
    LogQueue as ProgressQueue,
    PolicyDict,
    WorkerResult,
    WorkerStats,
    format_large_number,
    format_infoset_count,
)
from .exceptions import GracefulShutdownException
from .worker import run_cfr_simulation_worker

logger = logging.getLogger(__name__)

# Constants for pool shutdown
POOL_TERMINATE_TIMEOUT_SECONDS = 10  # Time to wait for pool.join() after terminate
POOL_JOIN_TIMEOUT_SECONDS = (
    5  # Time to wait for pool.join() if terminate wasn't called first
)


class CFRTrainingLoopMixin:
    """Handles the overall CFR+ training loop, progress, and scheduling, supporting parallelism."""

    # Attributes expected from main class
    config: Config
    analysis: "AnalysisTools"
    shutdown_event: "threading.Event"
    current_iteration: int
    exploitability_results: List[Tuple[int, float]]
    regret_sum: PolicyDict
    progress_queue: Optional[ProgressQueue]
    run_log_dir: Optional[str]
    run_timestamp: Optional[str]
    live_display_manager: Optional[LiveDisplayManager]
    # Attributes from CFRDataManagerMixin
    load_data: Callable[..., Any]
    save_data: Callable[..., Any]
    _merge_local_updates: Callable[[List[Optional[WorkerResult]]], None]
    compute_average_strategy: Callable[..., Optional[PolicyDict]]
    # Internal state
    _last_exploit_str: str = "N/A"
    _total_infosets_str: str = "0"
    _total_run_time_start: float = 0.0
    _worker_statuses: Dict[int, WorkerDisplayStatus] = {}
    log_archiver: Optional[LogArchiver] = None

    def __init__(self, *args, **kwargs):
        if hasattr(self, "config") and hasattr(self, "run_log_dir") and self.run_log_dir:
            if self.config.logging.log_archive_enabled:
                self.log_archiver = LogArchiver(self.config, self.run_log_dir)
        else:
            pass

    def _shutdown_pool(self, pool: Optional[multiprocessing.pool.Pool]):
        """Gracefully shuts down the multiprocessing pool."""
        if pool:
            logger.info("Terminating worker pool...")
            try:
                pool.terminate()  # Send SIGTERM to all worker processes
                pool.join(timeout=POOL_TERMINATE_TIMEOUT_SECONDS)  # Wait for them to exit
                logger.info("Worker pool terminated and joined.")
            except Exception as e:
                logger.error("Exception during pool shutdown: %s", e, exc_info=True)

    def train(self, num_iterations: Optional[int] = None):
        """Runs the main CFR+ training loop, potentially in parallel."""
        if (
            self.config.logging.log_archive_enabled
            and not self.log_archiver
            and self.run_log_dir
        ):
            self.log_archiver = LogArchiver(self.config, self.run_log_dir)
            logger.info("LogArchiver initialized in train method.")

        total_iterations_to_run = (
            num_iterations or self.config.cfr_training.num_iterations
        )
        last_completed_iteration = self.current_iteration
        start_iter_num = last_completed_iteration + 1
        end_iter_num = last_completed_iteration + total_iterations_to_run

        exploitability_interval = self.config.cfr_training.exploitability_interval
        num_workers = self.config.cfr_training.num_workers
        progress_queue = self.progress_queue
        run_log_dir = self.run_log_dir
        run_timestamp = self.run_timestamp
        display = self.live_display_manager

        logger.info(
            "Starting CFR+ training from iteration %d up to %d (%d workers).",
            start_iter_num,
            end_iter_num,
            num_workers,
        )
        if total_iterations_to_run <= 0:
            logger.warning("Number of iterations to run must be positive.")
            return

        if not display:
            logger.warning(
                "LiveDisplayManager not provided. Console output will be minimal."
            )

        self._total_run_time_start = time.time()
        pool: Optional[multiprocessing.pool.Pool] = None
        self._worker_statuses = {i: "Initializing" for i in range(num_workers)}
        if display:
            for i in range(num_workers):
                display.update_worker_status(i, "Idle")

        self._total_infosets_str = format_infoset_count(len(self.regret_sum))
        if display:
            display.update_stats(
                iteration=self.current_iteration,
                infosets=self._total_infosets_str,
                exploitability=self._last_exploit_str,
            )
            display.update_overall_progress(self.current_iteration)

        try:
            for t in range(start_iter_num, end_iter_num + 1):
                if self.shutdown_event.is_set():
                    logger.warning("Shutdown detected before starting iteration %d.", t)
                    raise GracefulShutdownException("Shutdown detected before iteration")

                iter_start_time = time.time()
                self.current_iteration = t
                # initial_live_status = ("Starting", 0, 0, 0) # Not used

                if display:
                    for i in range(num_workers):
                        status_to_set: WorkerDisplayStatus = (
                            "Queued" if num_workers > 1 else "Starting",
                            0,
                            0,
                            0,
                        )
                        self._worker_statuses[i] = status_to_set
                        display.update_worker_status(i, self._worker_statuses[i])

                    self._total_infosets_str = format_infoset_count(len(self.regret_sum))
                    display.update_overall_progress(t)
                    display.update_stats(
                        iteration=t,
                        infosets=self._total_infosets_str,
                        exploitability=self._last_exploit_str,
                    )
                else:  # No display
                    for i in range(num_workers):
                        status_to_set_no_disp: WorkerDisplayStatus = (
                            "Queued" if num_workers > 1 else "Starting",
                            0,
                            0,
                            0,
                        )
                        self._worker_statuses[i] = status_to_set_no_disp

                try:
                    # Snapshot is created once per iteration, before worker dispatch
                    regret_snapshot = dict(self.regret_sum)
                except Exception as e:
                    logger.error(
                        "Failed to prepare regret snapshot for iter %d: %s",
                        t,
                        e,
                        exc_info=True,
                    )
                    raise RuntimeError(
                        f"Could not prepare regret snapshot for iter {t}"
                    ) from e

                worker_base_args = (
                    t,
                    self.config,
                    regret_snapshot,  # Pass the snapshot
                    progress_queue,
                )
                results: List[Optional[WorkerResult]] = [None] * num_workers
                sim_failed_count = 0
                completed_worker_count = 0

                if num_workers == 1:
                    running_status_tuple: WorkerDisplayStatus = ("Running", 0, 0, 0)
                    if display:
                        display.update_worker_status(0, running_status_tuple)
                    self._worker_statuses[0] = running_status_tuple
                    try:
                        worker_args_seq = worker_base_args + (
                            0,
                            run_log_dir,
                            run_timestamp,
                        )
                        result: Optional[WorkerResult] = run_cfr_simulation_worker(
                            worker_args_seq
                        )
                        results[0] = result
                        completed_worker_count = 1
                        if result is None or not isinstance(result, WorkerResult):
                            logger.error(
                                "Worker 0 returned unexpected type or None: %s",
                                type(result),
                            )
                            self._worker_statuses[0] = (
                                f"Error (Type: {type(result).__name__})"
                            )
                        elif isinstance(result, WorkerResult):
                            self._worker_statuses[0] = result.stats
                            if result.stats.error_count > 0:
                                sim_failed_count += 1
                        if display:  # Update display after processing result
                            display.update_worker_status(0, self._worker_statuses[0])

                    except Exception as e:
                        logger.exception("Error during sequential simulation iter %d.", t)
                        self._worker_statuses[0] = "Error (Exception)"
                        if display:
                            display.update_worker_status(0, self._worker_statuses[0])
                        sim_failed_count += 1
                    finally:
                        # Drain progress queue for sequential mode
                        if progress_queue:
                            try:
                                while True:
                                    prog_w_id, prog_cur_d, prog_max_d, prog_n = (
                                        progress_queue.get_nowait()
                                    )
                                    if (
                                        prog_w_id == 0
                                    ):  # Ensure it's for the current worker
                                        new_status_tuple_seq: WorkerDisplayStatus = (
                                            "Running",
                                            prog_cur_d,
                                            prog_max_d,
                                            prog_n,
                                        )
                                        # Only update if current status is a tuple (i.e., not final)
                                        if isinstance(
                                            self._worker_statuses.get(0), tuple
                                        ):
                                            self._worker_statuses[0] = (
                                                new_status_tuple_seq
                                            )
                                            if display:
                                                display.update_worker_status(
                                                    prog_w_id, new_status_tuple_seq
                                                )
                            except queue.Empty:
                                pass  # Queue is empty, normal
                            except Exception as prog_e:
                                logger.error(
                                    "Error draining progress queue (sequential): %s",
                                    prog_e,
                                )
                        # Ensure final status is displayed
                        if display and isinstance(
                            self._worker_statuses.get(0), (str, WorkerStats)
                        ):
                            display.update_worker_status(0, self._worker_statuses[0])

                else:  # Parallel execution
                    worker_args_list = [
                        worker_base_args + (worker_id, run_log_dir, run_timestamp)
                        for worker_id in range(num_workers)
                    ]
                    async_results: Dict[int, multiprocessing.pool.AsyncResult] = {}
                    pool = None  # Define pool here to ensure it's in scope for finally
                    try:
                        pool = multiprocessing.Pool(processes=num_workers)
                        for worker_id, args in enumerate(worker_args_list):
                            async_results[worker_id] = pool.apply_async(
                                run_cfr_simulation_worker, (args,)
                            )
                            # Initial status for display
                            status_disp_par: WorkerDisplayStatus = ("Running", 0, 0, 0)
                            self._worker_statuses[worker_id] = status_disp_par
                            if display:
                                display.update_worker_status(worker_id, status_disp_par)
                        pool.close()  # No more tasks will be submitted

                        # Main loop to collect results and update display
                        active_workers_for_iteration = num_workers
                        while active_workers_for_iteration > 0:
                            if self.shutdown_event.is_set():
                                logger.warning(
                                    "Shutdown event detected while waiting for workers iter %d.",
                                    t,
                                )
                                raise GracefulShutdownException(
                                    "Shutdown during worker result collection"
                                )

                            # Process progress queue for live updates
                            if progress_queue:
                                try:
                                    while not progress_queue.empty():  # Check before get
                                        prog_w_id, prog_cur_d, prog_max_d, prog_n = (
                                            progress_queue.get_nowait()
                                        )
                                        # Only update if worker hasn't finished/errored
                                        current_internal_status = (
                                            self._worker_statuses.get(prog_w_id)
                                        )
                                        is_still_running_tuple = isinstance(
                                            current_internal_status, tuple
                                        ) and (
                                            current_internal_status[0]
                                            in ["Running", "Queued", "Starting"]
                                        )

                                        if is_still_running_tuple:
                                            new_prog_status: WorkerDisplayStatus = (
                                                "Running",
                                                prog_cur_d,
                                                prog_max_d,
                                                prog_n,
                                            )
                                            self._worker_statuses[prog_w_id] = (
                                                new_prog_status
                                            )
                                            if display:
                                                display.update_worker_status(
                                                    prog_w_id, new_prog_status
                                                )
                                except queue.Empty:
                                    pass  # Normal, queue is empty
                                except Exception as prog_e:
                                    logger.error(
                                        "Error reading progress queue (parallel): %s",
                                        prog_e,
                                    )

                            # Check for completed workers
                            for worker_id_done, async_res in list(
                                async_results.items()
                            ):  # Iterate copy
                                if async_res.ready():
                                    try:
                                        result_par: Optional[WorkerResult] = (
                                            async_res.get(timeout=0.01)
                                        )  # Short timeout
                                        results[worker_id_done] = result_par

                                        final_status_for_worker: WorkerDisplayStatus
                                        if result_par is None or not isinstance(
                                            result_par, WorkerResult
                                        ):
                                            sim_failed_count += 1
                                            final_status_for_worker = f"Failed (Type: {type(result_par).__name__})"
                                        elif isinstance(result_par, WorkerResult):
                                            final_status_for_worker = result_par.stats
                                            if result_par.stats.error_count > 0:
                                                sim_failed_count += 1
                                        self._worker_statuses[worker_id_done] = (
                                            final_status_for_worker
                                        )
                                        if display:
                                            display.update_worker_status(
                                                worker_id_done, final_status_for_worker
                                            )

                                    except multiprocessing.TimeoutError:
                                        continue  # Should not happen if ready() is true
                                    except Exception as pool_e:
                                        logger.error(
                                            "Error fetching result worker %d iter %d: %s",
                                            worker_id_done,
                                            t,
                                            pool_e,
                                            exc_info=False,
                                        )
                                        results[worker_id_done] = None
                                        sim_failed_count += 1
                                        err_fetch_status: WorkerDisplayStatus = (
                                            "Error (Fetch Fail)"
                                        )
                                        self._worker_statuses[worker_id_done] = (
                                            err_fetch_status
                                        )
                                        if display:
                                            display.update_worker_status(
                                                worker_id_done, err_fetch_status
                                            )
                                    finally:
                                        active_workers_for_iteration -= 1
                                        async_results.pop(
                                            worker_id_done, None
                                        )  # Remove processed item
                            time.sleep(
                                0.05
                            )  # Short sleep to avoid busy-waiting if no results ready

                    except GracefulShutdownException:
                        logger.warning("Graceful shutdown initiated for iter %d pool.", t)
                        self._shutdown_pool(pool)  # Ensure pool is terminated
                        raise  # Re-raise to be caught by outer handler
                    except Exception as e:
                        logger.exception(
                            "Error during parallel pool management iter %d.", t
                        )
                        sim_failed_count = (
                            num_workers  # Assume all failed if pool management breaks
                        )
                        self._shutdown_pool(pool)  # Attempt to clean up
                        if display:
                            for i_err in range(num_workers):
                                if i_err not in self._worker_statuses or not isinstance(
                                    self._worker_statuses[i_err], (WorkerStats, str)
                                ):
                                    status_pool_mgmt_err: WorkerDisplayStatus = (
                                        "Error (Pool Mgmt)"
                                    )
                                    self._worker_statuses[i_err] = status_pool_mgmt_err
                                    display.update_worker_status(
                                        i_err, status_pool_mgmt_err
                                    )
                    finally:
                        if (
                            pool and not self.shutdown_event.is_set()
                        ):  # If not already shut down by exception
                            self._shutdown_pool(pool)

                if sim_failed_count == num_workers and num_workers > 0:
                    logger.error(
                        "All simulations failed for iteration %d. Skipping merge.", t
                    )
                elif sim_failed_count > 0:
                    logger.warning(
                        "%d/%d worker sims failed for iter %d.",
                        sim_failed_count,
                        num_workers,
                        t,
                    )

                valid_results = [res for res in results if isinstance(res, WorkerResult)]
                if (
                    not valid_results and num_workers > 0
                ):  # Only log error if workers were expected
                    logger.error(
                        "No valid worker results for iter %d. Skipping merge.", t
                    )
                elif valid_results:
                    logger.debug(
                        "Merging results from %d valid workers for iter %d...",
                        len(valid_results),
                        t,
                    )
                    merge_start_time = time.time()
                    try:
                        self._merge_local_updates(valid_results)
                        logger.debug(
                            "Iter %d merge took %.3fs", t, time.time() - merge_start_time
                        )
                    except Exception as merge_err:
                        logger.exception("Error merging results iter %d.", t)
                        if display:
                            display.update_stats(
                                iteration=t,
                                infosets=self._total_infosets_str,
                                exploitability="Merge FAILED",
                                last_iter_time=time.time() - iter_start_time,
                            )
                        continue  # Skip to next iteration

                iter_time = time.time() - iter_start_time
                self._total_infosets_str = format_infoset_count(len(self.regret_sum))

                if exploitability_interval > 0 and t % exploitability_interval == 0:
                    logger.info("Calculating exploitability at iteration %d...", t)
                    exploit_start_time = time.time()
                    current_avg_strategy = self.compute_average_strategy()
                    if current_avg_strategy:
                        exploit = self.analysis.calculate_exploitability(
                            current_avg_strategy, self.config
                        )
                        self.exploitability_results.append((t, exploit))
                        self._last_exploit_str = (
                            f"{exploit:.4f}" if exploit != float("inf") else "N/A"
                        )
                        logger.info(
                            "Exploitability: %.4f (took %.2fs)",
                            exploit,
                            time.time() - exploit_start_time,
                        )
                    else:
                        logger.warning(
                            "Could not compute avg strategy for exploitability at iter %d.",
                            t,
                        )
                        self._last_exploit_str = "N/A (Avg Err)"

                if display:
                    display.update_stats(
                        iteration=t,
                        infosets=self._total_infosets_str,
                        exploitability=self._last_exploit_str,
                        last_iter_time=iter_time,
                    )
                    for i_disp in range(
                        num_workers
                    ):  # Update final worker statuses after merge
                        final_status_iter_end = self._worker_statuses.get(i_disp)
                        if final_status_iter_end:
                            display.update_worker_status(i_disp, final_status_iter_end)

                if t % self.config.cfr_training.save_interval == 0:
                    logger.info("Saving progress at iteration %d...", t)
                    self.save_data()  # Save data for completed iteration t
                    if self.log_archiver:
                        logger.info("Attempting log archival after save interval...")
                        try:
                            self.log_archiver.scan_and_archive_worker_logs()
                        except Exception as arch_e:
                            logger.error(
                                "Error during periodic log archival: %s",
                                arch_e,
                                exc_info=True,
                            )

            # End of main training loop (for t in range...)
            # If loop completes normally (no exception/shutdown)
            logger.info("Training loop completed %d iterations.", total_iterations_to_run)

        except (GracefulShutdownException, KeyboardInterrupt) as shutdown_exc:
            exception_type = type(shutdown_exc).__name__
            logger.warning(
                "Shutdown/Interrupt (%s) caught in train loop. Saving progress...",
                exception_type,
            )
            # self.current_iteration is the one that was *attempted* or *partially completed*
            # We save data for the *last fully completed* iteration.
            # If shutdown_event was set, train() should exit and main will handle final save.
            # If KeyboardInterrupt is caught directly here, this is the point to save.
            if (
                not self.shutdown_event.is_set()
            ):  # If this was a direct KeyboardInterrupt not via SIGINT handler
                self.shutdown_event.set()

            # Perform emergency save
            self._perform_emergency_save(
                last_completed_iteration,
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
                last_completed_iteration,
                start_iter_num,
                type(main_loop_e).__name__,
                pool_to_shutdown=pool,
            )
            raise  # Re-raise the original exception

        # --- Post-Loop (Normal Completion) ---
        if not self.shutdown_event.is_set():  # Only if not already shutting down
            end_time = time.time()
            total_completed_in_run = self.current_iteration - last_completed_iteration
            logger.info("Training loop finished %d iterations.", total_completed_in_run)
            logger.info(
                "Total training time this run: %.2f seconds.",
                end_time - self._total_run_time_start,
            )
            logger.info(
                "Current iteration count (last completed): %d", self.current_iteration
            )

            logger.info("Computing final average strategy...")
            final_avg_strategy = self.compute_average_strategy()
            if final_avg_strategy:
                logger.info("Calculating final exploitability...")
                final_exploit = self.analysis.calculate_exploitability(
                    final_avg_strategy, self.config
                )
                self.exploitability_results.append(
                    (self.current_iteration, final_exploit)
                )
                logger.info("Final exploitability: %.4f", final_exploit)
                self._last_exploit_str = (
                    f"{final_exploit:.4f}" if final_exploit != float("inf") else "N/A"
                )
            else:
                logger.warning("Could not compute final average strategy.")
                self._last_exploit_str = "N/A (Avg Err)"

            self._total_infosets_str = format_infoset_count(len(self.regret_sum))
            if display:
                display.update_stats(
                    iteration=self.current_iteration,
                    infosets=self._total_infosets_str,
                    exploitability=self._last_exploit_str,
                    last_iter_time=None,
                )
                for i_final_disp in range(num_workers):
                    final_status = self._worker_statuses.get(i_final_disp, "Finished")
                    display.update_worker_status(i_final_disp, final_status)

            logger.info("Performing final save and log archival...")
            self.save_data()  # Save for the last completed iteration
            if self.log_archiver:
                try:
                    self.log_archiver.scan_and_archive_worker_logs()
                except Exception as arch_e:
                    logger.error(
                        "Error during final log archival: %s", arch_e, exc_info=True
                    )
            logger.info("Final average strategy and data saved.")
            self._write_run_summary()  # Write summary on normal completion

    def _perform_emergency_save(
        self,
        last_completed_iteration_before_run: int,
        start_iter_num_this_run: int,
        reason: str,
        pool_to_shutdown: Optional[multiprocessing.pool.Pool],
    ):
        """Handles saving data during an interruption or error."""
        logger.warning(f"Attempting emergency save due to {reason}...")
        if pool_to_shutdown:
            self._shutdown_pool(pool_to_shutdown)

        # Determine the iteration number for which to save data
        # self.current_iteration is the iteration that was *in progress* or just started
        # We want to save data for the *last fully completed* iteration.
        iter_to_save = (
            self.current_iteration - 1
        )  # If current_iteration was processing T, T-1 was last completed.

        # If current_iteration is the very first one that was attempted in this run (start_iter_num_this_run),
        # then iter_to_save would be start_iter_num_this_run - 1.
        # This should correctly point to last_completed_iteration_before_run.
        if iter_to_save < last_completed_iteration_before_run:
            # This case occurs if shutdown happened before even one new iteration's data merge completed
            iter_to_save = last_completed_iteration_before_run

        if (
            iter_to_save >= 0 and iter_to_save >= last_completed_iteration_before_run
        ):  # Ensure we are saving valid progress
            original_current_iter_val = (
                self.current_iteration
            )  # Store for restoration if needed
            self.current_iteration = (
                iter_to_save  # Set to the iteration number being saved
            )
            logger.info(
                f"Saving data for last completed iteration: {self.current_iteration}"
            )
            try:
                self.save_data()
                if self.log_archiver:
                    logger.info("Attempting log archival during emergency save...")
                    self.log_archiver.scan_and_archive_worker_logs()
            except Exception as save_e:
                logger.error(f"Emergency save/archive failed: {save_e}", exc_info=True)
            finally:
                self.current_iteration = (
                    original_current_iter_val  # Restore actual current iter
                )
        else:
            logger.warning(
                f"Shutdown ({reason}) before first new iteration completed relative to loaded data. "
                f"No new progress to save beyond iteration {last_completed_iteration_before_run}."
            )

        self._write_run_summary()  # Attempt to write summary even on error/interrupt

    def _write_run_summary(self):
        """Writes a summary of the training run to the summary log file."""
        if not self.run_log_dir or not self.run_timestamp:
            logger.warning(
                "Cannot write run summary: Missing run_log_dir or run_timestamp."
            )
            return

        summary_file_path = os.path.join(
            self.run_log_dir,
            f"{self.config.logging.log_file_prefix}_run_{self.run_timestamp}-summary.log",
        )
        logger.info("Writing run summary to: %s", summary_file_path)

        try:
            with open(summary_file_path, "w", encoding="utf-8") as f:
                f.write("--- Cambia CFR+ Training Run Summary ---\n")
                f.write(f"Run Timestamp: {self.run_timestamp}\n")
                f.write(f"Run Directory: {self.run_log_dir}\n")
                config_path = getattr(self.config, "_source_path", "N/A")
                f.write(f"Config File: {config_path}\n")
                f.write(
                    f"Target/Attempted Iterations in this run (up to): {self.current_iteration}\n"
                )
                # Note: self.current_iteration reflects the iteration number that was *in progress* or *last attempted*.
                # The actual *last completed and saved* iteration might be current_iteration - 1 or less if interrupted.
                # For simplicity, we report based on the state of current_iteration at time of summary.

                total_time = (
                    time.time() - self._total_run_time_start
                    if self._total_run_time_start > 0
                    else 0
                )
                f.write(f"Total Run Time (this execution): {total_time:.2f} seconds\n")
                f.write(
                    f"Final Exploitability (if calculated): {self._last_exploit_str}\n"
                )
                f.write(f"Total Unique Infosets Reached: {len(self.regret_sum):,}\n")
                f.write(f"Number of Workers: {self.config.cfr_training.num_workers}\n")
                f.write("\n--- Worker Summary (Final Status) ---\n")

                num_workers = self.config.cfr_training.num_workers
                if num_workers > 0 and self._worker_statuses:
                    total_nodes = 0
                    max_depth_overall = 0
                    total_warnings = 0
                    total_errors = 0
                    completed_workers_with_stats = 0  # Workers that returned WorkerStats
                    failed_workers_marked = (
                        0  # Workers marked as "Failed" or "Error" strings
                    )

                    for i in range(num_workers):
                        status_info = self._worker_statuses.get(i)
                        status_line = f"Worker {i}: "
                        if isinstance(status_info, WorkerStats):
                            nodes_fmt = format_large_number(status_info.nodes_visited)
                            status_line += (
                                f"Completed (Max Depth: {status_info.max_depth}, "
                                f"Nodes Visited: {nodes_fmt}, "
                                f"Warnings: {status_info.warning_count}, "
                                f"Errors: {status_info.error_count})"
                            )
                            total_nodes += status_info.nodes_visited
                            max_depth_overall = max(
                                max_depth_overall, status_info.max_depth
                            )
                            total_warnings += status_info.warning_count
                            total_errors += (
                                status_info.error_count
                            )  # Add errors from stats
                            completed_workers_with_stats += 1
                        elif isinstance(status_info, str):
                            status_line += status_info
                            if "Fail" in status_info or "Error" in status_info:
                                failed_workers_marked += 1
                        elif (
                            isinstance(status_info, tuple) and len(status_info) == 4
                        ):  # (state, cur_d, max_d, nodes)
                            state, cur_d, max_d, nodes_val = status_info
                            nodes_fmt_tuple = format_large_number(nodes_val)
                            status_line += f"Stopped ({state}, D:{cur_d}/{max_d}, N:{nodes_fmt_tuple})"
                            failed_workers_marked += 1  # Assume stopped tuple means abnormal termination for summary
                        else:
                            status_line += (
                                f"Unknown Final State ({type(status_info).__name__})"
                            )
                            failed_workers_marked += 1
                        f.write(status_line + "\n")

                    f.write("\n--- Aggregated Worker Stats ---\n")
                    f.write(
                        f"Workers that returned full stats: {completed_workers_with_stats}\n"
                    )
                    f.write(
                        f"Workers explicitly marked as Failed/Error (string status): {failed_workers_marked}\n"
                    )
                    f.write(
                        f"Total Nodes Visited (Sum from WorkerStats): {format_large_number(total_nodes)} ({total_nodes:,})\n"
                    )
                    f.write(
                        f"Max Depth Reached (Overall from WorkerStats): {max_depth_overall}\n"
                    )
                    f.write(
                        f"Total Warnings Reported (Sum from WorkerStats): {total_warnings}\n"
                    )
                    f.write(
                        f"Total Errors Reported (Sum from WorkerStats): {total_errors}\n"
                    )

                elif num_workers == 1 and self._worker_statuses:  # Sequential specific
                    f.write("Sequential execution (1 worker).\n")
                    status_info_seq = self._worker_statuses.get(0)
                    if isinstance(status_info_seq, WorkerStats):
                        nodes_fmt_seq = format_large_number(status_info_seq.nodes_visited)
                        f.write("  Final Status: Completed (Returned Stats)\n")
                        f.write(f"  Max Depth: {status_info_seq.max_depth}\n")
                        f.write(
                            f"  Nodes Visited: {nodes_fmt_seq} ({status_info_seq.nodes_visited:,})\n"
                        )
                        f.write(f"  Warnings: {status_info_seq.warning_count}\n")
                        f.write(f"  Errors: {status_info_seq.error_count}\n")
                    elif isinstance(status_info_seq, str):
                        f.write(f"  Final Status: {status_info_seq}\n")
                    # other tuple case handled by general logic above if it were to occur for seq
                    else:
                        f.write(
                            f"  Final Status: Unknown ({type(status_info_seq).__name__})\n"
                        )
                else:
                    f.write(
                        "Worker status information not available or not applicable (0 workers).\n"
                    )

                f.write("\n--- Exploitability History ---\n")
                if self.exploitability_results:
                    for iter_num_exploit, exploit_val in self.exploitability_results:
                        f.write(f"Iteration {iter_num_exploit}: {exploit_val:.6f}\n")
                else:
                    f.write("No exploitability data recorded.\n")

                f.write("\n--- End Summary ---\n")
        except IOError as e:
            logger.error("Failed to write run summary file: %s", e)
        except Exception as e:  # Catch-all for unexpected summary errors
            logger.exception("Unexpected error writing run summary:")
