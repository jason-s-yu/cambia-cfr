# src/cfr/training_loop_mixin.py
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

    def __init__(
        self, *args, **kwargs
    ):  # Add dummy __init__ for mixin pattern if not defined in base
        # This is a bit of a hack if the base class (CFRTrainer) doesn't call super().__init__()
        # or if attributes are not set before train() is called.
        # For this specific case, CFRTrainer calls its __init__ which sets these attributes.
        # The primary purpose here is to initialize self.log_archiver.
        # It relies on self.config and self.run_log_dir being set by CFRTrainer's init.
        if hasattr(self, "config") and hasattr(self, "run_log_dir") and self.run_log_dir:
            if self.config.logging.log_archive_enabled:
                self.log_archiver = LogArchiver(self.config, self.run_log_dir)
        else:
            # This state might occur if this mixin is used without proper prior initialization
            # For now, we assume CFRTrainer handles it.
            pass

    def train(self, num_iterations: Optional[int] = None):
        """Runs the main CFR+ training loop, potentially in parallel."""
        # Initialize log_archiver here if not done in __init__ or if CFRTrainer.__init__ is complex
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
                initial_live_status = ("Starting", 0, 0, 0)

                if display:
                    for i in range(num_workers):
                        status_to_set = "Queued" if num_workers > 1 else "Starting"
                        self._worker_statuses[i] = (
                            status_to_set,
                            0,
                            0,
                            0,
                        )
                        display.update_worker_status(i, self._worker_statuses[i])

                    self._total_infosets_str = format_infoset_count(len(self.regret_sum))
                    display.update_overall_progress(t)
                    display.update_stats(
                        iteration=t,
                        infosets=self._total_infosets_str,
                        exploitability=self._last_exploit_str,
                    )
                else:
                    for i in range(num_workers):
                        status_to_set = "Queued" if num_workers > 1 else "Starting"
                        self._worker_statuses[i] = (status_to_set, 0, 0, 0)

                try:
                    regret_snapshot = dict(self.regret_sum)
                except Exception as e:
                    logger.error(
                        "Failed to prepare regret snapshot: %s", e, exc_info=True
                    )
                    raise RuntimeError(
                        f"Could not prepare regret snapshot for iter {t}"
                    ) from e

                worker_base_args = (
                    t,
                    self.config,
                    regret_snapshot,
                    progress_queue,
                )
                results: List[Optional[WorkerResult]] = [None] * num_workers
                sim_failed_count = 0
                completed_worker_count = 0

                if num_workers == 1:
                    running_status_tuple = ("Running", 0, 0, 0)
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
                            if display:
                                display.update_worker_status(0, self._worker_statuses[0])
                        elif isinstance(result, WorkerResult):
                            self._worker_statuses[0] = result.stats
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
                    finally:
                        if progress_queue:
                            try:
                                while True:
                                    prog_w_id, prog_cur_d, prog_max_d, prog_n = (
                                        progress_queue.get_nowait()
                                    )
                                    if prog_w_id == 0:
                                        new_status_tuple = (
                                            "Running",
                                            prog_cur_d,
                                            prog_max_d,
                                            prog_n,
                                        )
                                        if isinstance(
                                            self._worker_statuses.get(0), tuple
                                        ):
                                            self._worker_statuses[0] = new_status_tuple
                                            if display:
                                                display.update_worker_status(
                                                    prog_w_id, new_status_tuple
                                                )
                            except queue.Empty:
                                pass
                            except Exception as prog_e:
                                logger.error(
                                    "Error draining progress queue (sequential): %s",
                                    prog_e,
                                )
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
                    pool = None
                    try:
                        pool = multiprocessing.Pool(processes=num_workers)
                        for worker_id, args in enumerate(worker_args_list):
                            async_results[worker_id] = pool.apply_async(
                                run_cfr_simulation_worker, (args,)
                            )
                            running_status_tuple = ("Running", 0, 0, 0)
                            self._worker_statuses[worker_id] = running_status_tuple
                            if display:
                                display.update_worker_status(
                                    worker_id, running_status_tuple
                                )
                        pool.close()

                        while completed_worker_count < num_workers:
                            if self.shutdown_event.is_set():
                                raise GracefulShutdownException(
                                    "Shutdown during worker execution"
                                )

                            if progress_queue:
                                try:
                                    while True:
                                        prog_w_id, prog_cur_d, prog_max_d, prog_n = (
                                            progress_queue.get_nowait()
                                        )
                                        current_internal_status = (
                                            self._worker_statuses.get(prog_w_id)
                                        )
                                        is_running_tuple = isinstance(
                                            current_internal_status, tuple
                                        ) and (
                                            current_internal_status[0] == "Running"
                                            or current_internal_status[0] == "Queued"
                                            or current_internal_status[0] == "Starting"
                                        )

                                        if is_running_tuple:
                                            new_status_tuple = (
                                                "Running",
                                                prog_cur_d,
                                                prog_max_d,
                                                prog_n,
                                            )
                                            self._worker_statuses[prog_w_id] = (
                                                new_status_tuple
                                            )
                                            if display:
                                                display.update_worker_status(
                                                    prog_w_id, new_status_tuple
                                                )
                                except queue.Empty:
                                    pass
                                except Exception as prog_e:
                                    logger.error(
                                        "Error reading progress queue: %s", prog_e
                                    )

                            ready_workers_processed_this_cycle = False
                            for worker_id_done, async_res in list(async_results.items()):
                                if async_res.ready():
                                    ready_workers_processed_this_cycle = True
                                    try:
                                        result: Optional[WorkerResult] = async_res.get(
                                            timeout=0.01
                                        )
                                        results[worker_id_done] = result
                                        if result is None or not isinstance(
                                            result, WorkerResult
                                        ):
                                            sim_failed_count += 1
                                            fail_status = (
                                                f"Failed (Type: {type(result).__name__})"
                                            )
                                            self._worker_statuses[worker_id_done] = (
                                                fail_status
                                            )
                                        elif isinstance(result, WorkerResult):
                                            self._worker_statuses[worker_id_done] = (
                                                result.stats
                                            )
                                            if result.stats.error_count > 0:
                                                sim_failed_count += 1

                                        if display:
                                            display.update_worker_status(
                                                worker_id_done,
                                                self._worker_statuses[worker_id_done],
                                            )

                                    except multiprocessing.TimeoutError:
                                        continue
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
                                        error_status_str = "Error (Fetch Fail)"
                                        self._worker_statuses[worker_id_done] = (
                                            error_status_str
                                        )
                                        if display:
                                            display.update_worker_status(
                                                worker_id_done, error_status_str
                                            )
                                    finally:
                                        completed_worker_count += 1
                                        async_results.pop(worker_id_done, None)

                            if not ready_workers_processed_this_cycle:
                                time.sleep(0.05)

                    except GracefulShutdownException:
                        raise
                    except Exception as e:
                        logger.exception(
                            "Error during parallel pool management iter %d.", t
                        )
                        sim_failed_count = num_workers
                        if display:
                            for i in range(num_workers):
                                if i not in self._worker_statuses or not isinstance(
                                    self._worker_statuses[i], (WorkerStats, str)
                                ):
                                    error_status_pool = "Error (Pool Mgmt)"
                                    self._worker_statuses[i] = error_status_pool
                                    display.update_worker_status(i, error_status_pool)
                    finally:
                        if pool is not None:
                            pool.terminate()
                            pool.join()

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
                            current_w_status = self._worker_statuses.get(i)
                            if not isinstance(current_w_status, (WorkerStats, str)) or (
                                isinstance(current_w_status, str)
                                and "Fail" not in current_w_status
                                and "Error" not in current_w_status
                            ):
                                fail_status_iter_end = "Failed (Iter End)"
                                self._worker_statuses[i] = fail_status_iter_end
                                display.update_worker_status(i, fail_status_iter_end)
                    continue

                if sim_failed_count > 0:
                    logger.warning(
                        "%d/%d worker sims failed for iter %d.",
                        sim_failed_count,
                        num_workers,
                        t,
                    )

                valid_results = [res for res in results if isinstance(res, WorkerResult)]
                if not valid_results:
                    logger.error(
                        "No valid worker results found for iteration %d after filtering. Skipping merge.",
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

                logger.debug(
                    "Merging results from %d valid workers for iteration %d...",
                    len(valid_results),
                    t,
                )
                merge_start_time = time.time()
                try:
                    self._merge_local_updates(valid_results)
                    merge_time = time.time() - merge_start_time
                    logger.debug("Iter %d merge took %.3fs", t, merge_time)
                except Exception as merge_err:
                    logger.exception("Error merging results iter %d.", t)
                    if display:
                        display.update_stats(
                            iteration=t,
                            infosets=self._total_infosets_str,
                            exploitability="Merge FAILED",
                            last_iter_time=time.time() - iter_start_time,
                        )
                    continue

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

                if display:
                    display.update_stats(
                        iteration=t,
                        infosets=self._total_infosets_str,
                        exploitability=self._last_exploit_str,
                        last_iter_time=iter_time,
                    )
                    for i in range(num_workers):
                        final_status_for_worker = self._worker_statuses.get(i)
                        if final_status_for_worker:
                            display.update_worker_status(i, final_status_for_worker)

                if t % self.config.cfr_training.save_interval == 0:
                    logger.info("Saving progress at iteration %d...", t)
                    save_start_time = time.time()
                    self.save_data()
                    logger.info(
                        "Save complete (took %.2fs).", time.time() - save_start_time
                    )
                    # After saving, attempt to archive logs if enabled
                    if self.log_archiver:
                        logger.info("Attempting log archival after save interval...")
                        try:
                            self.log_archiver.scan_and_archive_worker_logs()
                        except Exception as arch_e:
                            logger.error(
                                "Error during log archival: %s", arch_e, exc_info=True
                            )

        except GracefulShutdownException as shutdown_exc:
            exception_type = type(shutdown_exc).__name__
            logger.warning(
                "Shutdown/Interrupt (%s) caught in train loop. Terminating pool and saving progress...",
                exception_type,
            )
            completed_iter_to_save = (
                self.current_iteration - 1
                if self.current_iteration > last_completed_iteration
                and self.current_iteration > start_iter_num
                else last_completed_iteration
            )
            if completed_iter_to_save >= 0:
                saved_iter_num = self.current_iteration
                self.current_iteration = completed_iter_to_save
                try:
                    self.save_data()
                    if self.log_archiver:  # Attempt archive on shutdown save
                        self.log_archiver.scan_and_archive_worker_logs()
                except Exception as save_e:
                    logger.error(
                        "Failed to save/archive progress during shutdown (%s): %s",
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
            completed_iter_to_save = (
                self.current_iteration - 1
                if self.current_iteration > last_completed_iteration
                and self.current_iteration > start_iter_num
                else last_completed_iteration
            )
            if completed_iter_to_save >= 0:
                saved_iter_num = self.current_iteration
                self.current_iteration = completed_iter_to_save
                try:
                    self.save_data()
                    if self.log_archiver:  # Attempt archive on shutdown save
                        self.log_archiver.scan_and_archive_worker_logs()
                except Exception as save_e:
                    logger.error(
                        "Failed to save/archive progress during shutdown (%s): %s",
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
            logger.warning("Attempting emergency save after unhandled exception...")
            completed_iter_to_save = (
                self.current_iteration - 1
                if self.current_iteration > last_completed_iteration
                and self.current_iteration > start_iter_num
                else last_completed_iteration
            )
            if completed_iter_to_save >= 0:
                saved_iter_num = self.current_iteration
                self.current_iteration = completed_iter_to_save
                try:
                    self.save_data()
                    if self.log_archiver:  # Attempt archive on emergency save
                        self.log_archiver.scan_and_archive_worker_logs()
                except Exception as save_e:
                    logger.error("Emergency save/archive failed: %s", save_e)
                finally:
                    self.current_iteration = saved_iter_num
            raise main_loop_e

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
                final_status = self._worker_statuses.get(i)
                if isinstance(final_status, WorkerStats):
                    display.update_worker_status(i, final_status)
                elif isinstance(final_status, str) and (
                    "Error" in final_status or "Fail" in final_status
                ):
                    display.update_worker_status(i, final_status)
                else:
                    display.update_worker_status(i, "Finished")

        logger.info("Performing final save and log archival...")
        self.save_data()
        if self.log_archiver:  # Final archive attempt
            try:
                self.log_archiver.scan_and_archive_worker_logs()
            except Exception as arch_e:
                logger.error("Error during final log archival: %s", arch_e, exc_info=True)
        logger.info("Final average strategy and data saved.")

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
                f.write(f"Total Iterations Completed: {self.current_iteration}\n")
                total_time = (
                    time.time() - self._total_run_time_start
                    if self._total_run_time_start > 0
                    else 0
                )
                f.write(f"Total Run Time: {total_time:.2f} seconds\n")
                f.write(f"Final Exploitability: {self._last_exploit_str}\n")
                f.write(f"Total Unique Infosets Reached: {len(self.regret_sum):,}\n")
                f.write(f"Number of Workers: {self.config.cfr_training.num_workers}\n")
                f.write("\n--- Worker Summary (Final Status) ---\n")

                num_workers = self.config.cfr_training.num_workers
                if num_workers > 0 and self._worker_statuses:
                    total_nodes = 0
                    max_depth_overall = 0
                    total_warnings = 0
                    total_errors = 0
                    completed_workers = 0
                    failed_workers = 0
                    error_in_stats_workers = 0

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
                            elif status_info == "Finished":
                                completed_workers += 1
                            elif status_info not in [
                                "Done",
                                "Initializing",
                            ]:
                                failed_workers += 1

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
                    f.write(
                        f"Workers Successfully Completed (Returned Stats or 'Finished'): {completed_workers}\n"
                    )
                    f.write(f"Workers Failed/Stopped Abnormally: {failed_workers}\n")
                    f.write(
                        f"Workers with Errors Reported in Stats: {error_in_stats_workers}\n"
                    )
                    f.write(
                        f"Total Nodes Visited (Sum Across Completed Workers with Stats): {format_large_number(total_nodes)} ({total_nodes:,})\n"
                    )
                    f.write(
                        f"Max Depth Reached (Overall Across Completed Workers with Stats): {max_depth_overall}\n"
                    )
                    f.write(
                        f"Total Warnings Reported (Sum Across Completed Workers with Stats): {total_warnings}\n"
                    )
                    f.write(
                        f"Total Errors Reported (Sum Across Stats + Explicit Errors): {total_errors}\n"
                    )

                elif num_workers == 1 and self._worker_statuses:
                    f.write("Sequential execution (1 worker).\n")
                    status_info = self._worker_statuses.get(0)
                    if isinstance(status_info, WorkerStats):
                        nodes_fmt = format_large_number(status_info.nodes_visited)
                        f.write("  Final Status: Completed\n")
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
                    else:
                        f.write(
                            f"  Final Status: Unknown ({type(status_info).__name__})\n"
                        )
                else:
                    f.write("Worker status information not available.\n")

                f.write("\n--- Exploitability History ---\n")
                if self.exploitability_results:
                    for iter_num, exploit in self.exploitability_results:
                        f.write(f"Iteration {iter_num}: {exploit:.6f}\n")
                else:
                    f.write("No exploitability data recorded.\n")

                f.write("\n--- End Summary ---\n")
        except IOError as e:
            logger.error("Failed to write run summary file: %s", e)
        except Exception as e:
            logger.exception("Unexpected error writing run summary:")
