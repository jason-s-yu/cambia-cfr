"""src/cfr/training_loop_mixin.py"""

import copy
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

# Import the new formatting utility
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
    # Define _worker_statuses here for summary writing, even if display manager handles live updates
    _worker_statuses: Dict[int, WorkerDisplayStatus] = {}

    def train(self, num_iterations: Optional[int] = None):
        """Runs the main CFR+ training loop, potentially in parallel."""
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
        # Initialize worker statuses internally for summary, and update display manager
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

        # Outer try block for the entire training duration
        try:
            # Main iteration loop
            for t in range(start_iter_num, end_iter_num + 1):
                if self.shutdown_event.is_set():
                    logger.warning("Shutdown detected before starting iteration %d.", t)
                    raise GracefulShutdownException("Shutdown detected before iteration")

                iter_start_time = time.time()
                self.current_iteration = t
                # Reset internal statuses for summary tracking at start of iter
                self._worker_statuses = {
                    i: ("Starting", 0, 0, 0) for i in range(num_workers)
                }
                if display:
                    for i in range(num_workers):
                        display.update_worker_status(i, ("Starting", 0, 0, 0))
                    self._total_infosets_str = format_infoset_count(len(self.regret_sum))
                    display.update_overall_progress(t)
                    display.update_stats(
                        iteration=t,
                        infosets=self._total_infosets_str,
                        exploitability=self._last_exploit_str,
                    )

                # Prepare snapshot
                try:
                    regret_sum_copy = copy.deepcopy(self.regret_sum)
                    regret_snapshot = dict(regret_sum_copy)
                except Exception as e:
                    logger.error(
                        "Failed to prepare regret snapshot: %s", e, exc_info=True
                    )
                    raise RuntimeError(
                        f"Could not prepare regret snapshot for iter {t}"
                    ) from e

                # Worker execution logic
                worker_base_args = (t, self.config, regret_snapshot, progress_queue)
                results: List[Optional[WorkerResult]] = [None] * num_workers
                sim_failed_count = 0
                completed_worker_count = 0

                if num_workers == 1:
                    # Sequential execution
                    if display:
                        display.update_worker_status(0, ("Running", 0, 0, 0))
                    self._worker_statuses[0] = (
                        "Running",
                        0,
                        0,
                        0,
                    )  # Update internal status
                    try:
                        worker_args = worker_base_args + (0, run_log_dir, run_timestamp)
                        result: Optional[WorkerResult] = run_cfr_simulation_worker(
                            worker_args
                        )
                        results[0] = result
                        completed_worker_count = 1
                        if result is None or not isinstance(result, WorkerResult):
                            logger.error(
                                "Worker 0 returned unexpected type or None: %s",
                                type(result),
                            )
                            self._worker_statuses[0] = "Error (Return Type)"
                            if display:
                                display.update_worker_status(0, self._worker_statuses[0])
                            sim_failed_count += 1
                        elif isinstance(result, WorkerResult):
                            self._worker_statuses[0] = result.stats  # Store final stats
                            if display:
                                display.update_worker_status(0, result.stats)
                            if result.stats.error_count > 0:
                                sim_failed_count += 1
                    except Exception as e:
                        logger.exception("Error during sequential simulation iter %d.", t)
                        self._worker_statuses[0] = "Error (Exception)"
                        if display:
                            display.update_worker_status(0, self._worker_statuses[0])
                        sim_failed_count += 1
                else:
                    # Parallel execution
                    worker_args_list = [
                        worker_base_args + (worker_id, run_log_dir, run_timestamp)
                        for worker_id in range(num_workers)
                    ]
                    async_results: Dict[int, multiprocessing.pool.AsyncResult] = {}
                    pool = None
                    try:
                        if display:
                            for i in range(num_workers):
                                display.update_worker_status(i, ("Queued", 0, 0, 0))
                        self._worker_statuses = {
                            i: ("Queued", 0, 0, 0) for i in range(num_workers)
                        }  # Update internal

                        pool = multiprocessing.Pool(processes=num_workers)
                        for worker_id, args in enumerate(worker_args_list):
                            async_results[worker_id] = pool.apply_async(
                                run_cfr_simulation_worker, (args,)
                            )
                            self._worker_statuses[worker_id] = (
                                "Running",
                                0,
                                0,
                                0,
                            )  # Update internal
                            if display:
                                display.update_worker_status(
                                    worker_id, self._worker_statuses[worker_id]
                                )

                        pool.close()

                        # Result and Progress Collection Loop
                        while completed_worker_count < num_workers:
                            if self.shutdown_event.is_set():
                                raise GracefulShutdownException(
                                    "Shutdown during worker execution"
                                )

                            if (
                                progress_queue
                            ):  # Process progress independent of display manager existence
                                try:
                                    while True:
                                        prog_w_id, prog_cur_d, prog_max_d, prog_n = (
                                            progress_queue.get_nowait()
                                        )
                                        # Update internal status *first*
                                        current_internal_status = (
                                            self._worker_statuses.get(prog_w_id)
                                        )
                                        if (
                                            isinstance(current_internal_status, tuple)
                                            and current_internal_status[0] == "Running"
                                        ):
                                            self._worker_statuses[prog_w_id] = (
                                                "Running",
                                                prog_cur_d,
                                                prog_max_d,
                                                prog_n,
                                            )
                                            # Then update display if available
                                            if display:
                                                display.update_worker_status(
                                                    prog_w_id,
                                                    self._worker_statuses[prog_w_id],
                                                )
                                except queue.Empty:
                                    pass
                                except Exception as prog_e:
                                    logger.error(
                                        "Error reading progress queue: %s", prog_e
                                    )

                            ready_workers = [
                                wid
                                for wid, res in async_results.items()
                                if res and res.ready()
                            ]
                            for worker_id in ready_workers:
                                if worker_id not in async_results:
                                    continue
                                async_res = async_results.pop(worker_id, None)
                                if async_res is None:
                                    continue
                                try:
                                    result: Optional[WorkerResult] = async_res.get(
                                        timeout=0.1
                                    )
                                    results[worker_id] = result
                                    completed_worker_count += 1
                                    if result is None or not isinstance(
                                        result, WorkerResult
                                    ):
                                        sim_failed_count += 1
                                        self._worker_statuses[worker_id] = (
                                            f"Failed (Type: {type(result).__name__})"
                                        )
                                        if display:
                                            display.update_worker_status(
                                                worker_id,
                                                self._worker_statuses[worker_id],
                                            )
                                    elif isinstance(result, WorkerResult):
                                        self._worker_statuses[worker_id] = (
                                            result.stats
                                        )  # Store final stats
                                        if display:
                                            display.update_worker_status(
                                                worker_id, result.stats
                                            )
                                        if result.stats.error_count > 0:
                                            sim_failed_count += 1
                                except multiprocessing.TimeoutError:
                                    async_results[worker_id] = async_res
                                    continue
                                except Exception as pool_e:
                                    logger.error(
                                        "Error fetching result worker %d iter %d: %s",
                                        worker_id,
                                        t,
                                        pool_e,
                                        exc_info=False,
                                    )
                                    results[worker_id] = None
                                    completed_worker_count += 1
                                    sim_failed_count += 1
                                    self._worker_statuses[worker_id] = (
                                        "Error (Fetch Fail)"
                                    )
                                    if display:
                                        display.update_worker_status(
                                            worker_id, self._worker_statuses[worker_id]
                                        )
                            if completed_worker_count < num_workers:
                                time.sleep(0.1)
                    except GracefulShutdownException:
                        raise
                    except Exception as e:
                        logger.exception(
                            "Error during parallel pool management iter %d.", t
                        )
                        sim_failed_count = num_workers
                        if display:
                            for i in range(num_workers):
                                if i not in async_results:
                                    self._worker_statuses[i] = (
                                        "Error (Pool Mgmt)"  # Update internal
                                    )
                                    display.update_worker_status(
                                        i, self._worker_statuses[i]
                                    )
                        raise e
                    finally:
                        if pool is not None:
                            pool.terminate()
                            pool.join()

                # --- Process Results (Merge) ---
                if sim_failed_count == num_workers and num_workers > 0:
                    logger.error(
                        "All simulations failed for iteration %d. Skipping merge.", t
                    )
                    if display:
                        display.update_stats(
                            iteration=t,
                            infosets=self._total_infosets_str,
                            exploitability=self._last_exploit_str,
                            last_iter_time=time.time() - iter_start_time,
                        )
                        for i in range(num_workers):
                            current_status = self._worker_statuses.get(
                                i
                            )  # Use internal status
                            if not isinstance(current_status, WorkerStats):
                                if (
                                    not isinstance(current_status, str)
                                    or "Fail" not in current_status
                                    and "Error" not in current_status
                                ):
                                    self._worker_statuses[i] = (
                                        "Failed (Iter End)"  # Update internal
                                    )
                                    display.update_worker_status(
                                        i, self._worker_statuses[i]
                                    )
                    continue

                if sim_failed_count > 0:
                    logger.warning(
                        "%d/%d worker sims failed for iter %d.",
                        sim_failed_count,
                        num_workers,
                        t,
                    )
                valid_results = [res for res in results if isinstance(res, WorkerResult)]
                if not valid_results or all(
                    r.regret_updates is None for r in valid_results
                ):
                    logger.error(
                        "All simulations failed or returned no update data for iteration %d. Skipping merge.",
                        t,
                    )
                    if display:
                        display.update_stats(
                            iteration=t,
                            infosets=self._total_infosets_str,
                            exploitability=self._last_exploit_str,
                            last_iter_time=time.time() - iter_start_time,
                        )
                    continue

                logger.debug("Merging results for iteration %d...", t)
                merge_start_time = time.time()
                try:
                    self._merge_local_updates(valid_results)
                    merge_time = time.time() - merge_start_time
                    logger.debug("Iter %d merge took %.3fs", t, merge_time)
                except Exception:
                    logger.exception("Error merging results iter %d.", t)
                    if display:
                        display.update_stats(
                            iteration=t,
                            infosets=self._total_infosets_str,
                            exploitability="Merge FAILED",
                            last_iter_time=time.time() - iter_start_time,
                        )
                    continue

                # --- Iteration Completed ---
                iter_time = time.time() - iter_start_time
                self._total_infosets_str = format_infoset_count(len(self.regret_sum))

                # --- Exploitability Calculation ---
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
                        exploit_calc_time = time.time() - exploit_start_time
                        logger.info(
                            "Exploitability: %.4f (took %.2fs)",
                            exploit,
                            exploit_calc_time,
                        )
                    else:
                        logger.warning(
                            "Could not compute avg strategy for exploitability."
                        )
                        self._last_exploit_str = "N/A (Avg Err)"

                # --- Final Update for Iteration ---
                if display:
                    display.update_stats(
                        iteration=t,
                        infosets=self._total_infosets_str,
                        exploitability=self._last_exploit_str,
                        last_iter_time=iter_time,
                    )

                # --- Save Interval ---
                if t % self.config.cfr_training.save_interval == 0:
                    logger.info("Saving progress at iteration %d...", t)
                    save_start_time = time.time()
                    self.save_data()
                    logger.info(
                        "Save complete (took %.2fs).", time.time() - save_start_time
                    )
            # --- END OF for t in range(...) LOOP ---

        # --- Main Loop Exception Handling ---
        except GracefulShutdownException as shutdown_exc:
            exception_type = type(shutdown_exc).__name__
            logger.warning(
                "Shutdown/Interrupt (%s) caught in train loop. Terminating pool and saving progress...",
                exception_type,
            )
            if num_workers > 1 and pool is not None:
                logger.warning("Terminating worker pool due to %s...", exception_type)
                pool.terminate()
                try:
                    pool.join()
                except Exception as join_e:
                    logger.error(
                        "Error joining pool after %s: %s", exception_type, join_e
                    )
                pool = None
            # Save logic
            completed_iter_to_save = (
                self.current_iteration - 1
                if hasattr(self, "current_iteration")
                and self.current_iteration > last_completed_iteration
                else last_completed_iteration
            )
            if completed_iter_to_save >= 0:
                saved_iter_num = self.current_iteration
                self.current_iteration = completed_iter_to_save
                try:
                    self.save_data()
                except Exception as save_e:
                    logger.error(
                        "Failed to save progress during shutdown (%s): %s",
                        exception_type,
                        save_e,
                    )
                finally:
                    self.current_iteration = saved_iter_num
            else:
                logger.warning(
                    "Shutdown (%s) before first new iteration completed or loaded. No progress to save.",
                    exception_type,
                )
            raise

        except KeyboardInterrupt as interrupt_exc:
            exception_type = type(interrupt_exc).__name__
            logger.warning(
                "Shutdown/Interrupt (%s) caught in train loop. Terminating pool and saving progress...",
                exception_type,
            )
            if num_workers > 1 and pool is not None:
                logger.warning("Terminating worker pool due to %s...", exception_type)
                pool.terminate()
                try:
                    pool.join()
                except Exception as join_e:
                    logger.error(
                        "Error joining pool after %s: %s", exception_type, join_e
                    )
                pool = None
            # Save logic
            completed_iter_to_save = (
                self.current_iteration - 1
                if hasattr(self, "current_iteration")
                and self.current_iteration > last_completed_iteration
                else last_completed_iteration
            )
            if completed_iter_to_save >= 0:
                saved_iter_num = self.current_iteration
                self.current_iteration = completed_iter_to_save
                try:
                    self.save_data()
                except Exception as save_e:
                    logger.error(
                        "Failed to save progress during shutdown (%s): %s",
                        exception_type,
                        save_e,
                    )
                finally:
                    self.current_iteration = saved_iter_num
            else:
                logger.warning(
                    "Shutdown (%s) before first new iteration completed or loaded. No progress to save.",
                    exception_type,
                )
            raise GracefulShutdownException(
                f"{exception_type} processed"
            ) from interrupt_exc

        except Exception as main_loop_e:
            logger.exception("Unhandled exception in main training loop:")
            if num_workers > 1 and pool is not None:
                logger.warning("Terminating worker pool due to unhandled error...")
                pool.terminate()
                try:
                    pool.join()
                except Exception as join_e:
                    logger.error(
                        "Error joining worker pool during error handling: %s", join_e
                    )
                pool = None
            logger.warning("Attempting emergency save after unhandled exception...")
            # Save logic
            completed_iter_to_save = (
                self.current_iteration - 1
                if hasattr(self, "current_iteration")
                and self.current_iteration > last_completed_iteration
                else last_completed_iteration
            )
            if completed_iter_to_save >= 0:
                saved_iter_num = self.current_iteration
                self.current_iteration = completed_iter_to_save
                try:
                    self.save_data()
                except Exception as save_e:
                    logger.error("Emergency save failed: %s", save_e)
                finally:
                    self.current_iteration = saved_iter_num
            raise main_loop_e

        finally:
            # Final pool cleanup
            if num_workers > 1 and pool is not None:
                is_active = False
                try:
                    is_active = True
                except Exception:
                    is_active = False
                if is_active:
                    logger.warning(
                        "Terminating/Joining pool in main finally block (unexpected loop exit?)."
                    )
                    pool.terminate()
                    try:
                        pool.join()
                    except Exception as final_join_e:
                        logger.error(
                            "Error joining pool in main finally block: %s", final_join_e
                        )
                pool = None
            # Display manager stopping handled by main_train.py

        # --- Training Loop Finished Normally ---
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
            if (
                not self.exploitability_results
                or self.exploitability_results[-1][0] != self.current_iteration
            ):
                self.exploitability_results.append(
                    (self.current_iteration, final_exploit)
                )
            else:
                self.exploitability_results[-1] = (self.current_iteration, final_exploit)
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
            for i in range(num_workers):
                # Ensure final status shows completion or last known stats
                if not isinstance(self._worker_statuses.get(i), WorkerStats):
                    if (
                        self._worker_statuses.get(i) is not None
                        and "Fail" not in str(self._worker_statuses.get(i))
                        and "Error" not in str(self._worker_statuses.get(i))
                    ):
                        display.update_worker_status(i, "Finished")
                    # Otherwise, keep the Error/Failed status

        self.save_data()
        logger.info("Final average strategy and data saved.")

    def _write_run_summary(self):
        """Writes a summary of the training run to the summary log file."""
        if not self.run_log_dir or not self.run_timestamp:
            logger.warning(
                "Cannot write run summary: Missing run_log_dir or run_timestamp."
            )
            return

        # Construct summary file path using the stored run details
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
                f.write(f"Total Iterations Completed: {self.current_iteration}\n")
                total_time = (
                    time.time() - self._total_run_time_start
                    if self._total_run_time_start > 0
                    else 0
                )
                f.write(f"Total Run Time: {total_time:.2f} seconds\n")
                f.write(f"Final Exploitability: {self._last_exploit_str}\n")
                f.write(f"Total Unique Infosets: {len(self.regret_sum):,}\n")
                f.write("\n--- Worker Summary ---\n")

                num_workers = self.config.cfr_training.num_workers
                if num_workers > 0:
                    total_nodes = 0
                    max_depth_overall = 0
                    total_warnings = 0
                    total_errors = 0
                    completed_workers = 0
                    failed_workers = 0
                    error_in_stats_workers = 0

                    # Use the internal _worker_statuses which holds final state
                    for i in range(num_workers):
                        status_info = self._worker_statuses.get(i)
                        status_line = f"Worker {i}: "
                        if isinstance(status_info, WorkerStats):
                            nodes_fmt = format_large_number(status_info.nodes_visited)
                            status_line += f"Completed (Max Depth: {status_info.max_depth}, Nodes Visited: {nodes_fmt}, Warnings: {status_info.warning_count}, Errors: {status_info.error_count})"
                            total_nodes += status_info.nodes_visited
                            max_depth_overall = max(
                                max_depth_overall, status_info.max_depth
                            )
                            total_warnings += status_info.warning_count
                            total_errors += status_info.error_count
                            if status_info.error_count > 0:
                                error_in_stats_workers += 1
                            completed_workers += 1
                        elif isinstance(status_info, str):
                            status_line += status_info
                            if "Fail" in status_info or "Error" in status_info:
                                failed_workers += 1
                                total_errors += 1
                        elif isinstance(status_info, tuple) and len(status_info) == 4:
                            state, cur_d, max_d, nodes = status_info
                            nodes_fmt = format_large_number(nodes)
                            status_line += (
                                f"Stopped ({state}, D:{cur_d}/{max_d}, N:{nodes_fmt})"
                            )
                            failed_workers += 1
                            total_errors += 1
                        else:
                            status_line += (
                                f"Unknown Final State ({type(status_info).__name__})"
                            )
                            failed_workers += 1
                            total_errors += 1
                        f.write(status_line + "\n")

                    f.write("\n--- Aggregated Worker Stats ---\n")
                    f.write(f"Completed Workers (Returned Stats): {completed_workers}\n")
                    f.write(
                        f"Failed/Error Workers (String/Tuple Status): {failed_workers}\n"
                    )
                    f.write(f"Workers with Errors in Stats: {error_in_stats_workers}\n")
                    f.write(
                        f"Total Nodes Visited (Sum Across Completed Workers): {format_large_number(total_nodes)} ({total_nodes:,})\n"
                    )
                    f.write(
                        f"Max Depth Reached (Overall Across Completed Workers): {max_depth_overall}\n"
                    )
                    f.write(
                        f"Total Warnings Logged (Sum Across Completed Workers): {total_warnings}\n"
                    )
                    f.write(
                        f"Total Errors Logged (Sum Across Completed Workers + Failed): {total_errors}\n"
                    )
                else:  # Sequential
                    f.write("Sequential execution (1 worker).\n")
                    status_info = self._worker_statuses.get(0)
                    if isinstance(status_info, WorkerStats):
                        nodes_fmt = format_large_number(status_info.nodes_visited)
                        f.write(f"  Final Status: Completed\n")
                        f.write(f"  Max Depth: {status_info.max_depth}\n")
                        f.write(
                            f"  Nodes Visited: {nodes_fmt} ({status_info.nodes_visited:,})\n"
                        )
                        f.write(f"  Warnings: {status_info.warning_count}\n")
                        f.write(f"  Errors: {status_info.error_count}\n")
                    elif isinstance(status_info, str):
                        f.write(f"  Final Status: {status_info}\n")
                    elif isinstance(status_info, tuple) and len(status_info) == 4:
                        state, cur_d, max_d, nodes = status_info
                        nodes_fmt = format_large_number(nodes)
                        f.write(
                            f"  Final Status: Stopped ({state}, D:{cur_d}/{max_d}, N:{nodes_fmt})\n"
                        )

                f.write("\n--- End Summary ---\n")
        except IOError as e:
            logger.error("Failed to write run summary file: %s", e)
        except Exception as e:
            logger.exception("Unexpected error writing run summary:")
