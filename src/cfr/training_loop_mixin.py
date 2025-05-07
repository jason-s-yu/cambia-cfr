"""src/cfr/training_loop_mixin.py"""

import logging
import multiprocessing
import multiprocessing.pool
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.persistence import ReachProbDict


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
from .worker import (
    run_cfr_simulation_worker,
    ArchiveQueueWorker,
)

logger = logging.getLogger(__name__)

POOL_TERMINATE_TIMEOUT_SECONDS = 10


class CFRTrainingLoopMixin:
    """Handles the overall CFR+ training loop, progress, and scheduling, supporting parallelism."""

    config: Config
    analysis: "AnalysisTools"
    shutdown_event: "threading.Event"
    current_iteration: int
    exploitability_results: List[Tuple[int, float]]
    regret_sum: PolicyDict
    strategy_sum: PolicyDict
    reach_prob_sum: ReachProbDict
    progress_queue: Optional[ProgressQueue]
    archive_queue: Optional[ArchiveQueueWorker]
    run_log_dir: Optional[str]
    run_timestamp: Optional[str]
    live_display_manager: Optional[LiveDisplayManager]
    log_archiver_global_ref: Optional[LogArchiver]

    load_data: Callable[..., Any]
    save_data: Callable[..., Any]
    _merge_local_updates: Callable[[List[Optional[WorkerResult]]], None]
    compute_average_strategy: Callable[..., Optional[PolicyDict]]

    _last_exploit_str: str = "N/A"
    _total_infosets_str: str = "0"
    _total_run_time_start: float = 0.0
    _worker_statuses: Dict[int, WorkerDisplayStatus] = {}
    _last_log_size_update_time: float = 0.0

    def __init__(self, *args, **kwargs):
        if hasattr(self, "archive_queue"):
            logger.debug("CFRTrainingLoopMixin initialized with archive_queue.")
        else:
            self.archive_queue = None
            logger.debug("CFRTrainingLoopMixin initialized without archive_queue.")
        self._last_log_size_update_time = 0.0

    def _shutdown_pool(self, pool: Optional[multiprocessing.pool.Pool]):
        """Gracefully shuts down the multiprocessing pool."""
        if pool:
            logger.info("Terminating worker pool...")
            try:
                pool.terminate()
                pool.join(timeout=POOL_TERMINATE_TIMEOUT_SECONDS)
                logger.info("Worker pool terminated and joined.")
            except Exception as e:
                logger.error("Exception during pool shutdown: %s", e, exc_info=True)

    def _try_update_log_size_display(self):
        """Attempts to update the log size display if conditions are met."""
        if (
            self.live_display_manager
            and hasattr(self, "log_archiver_global_ref")
            and self.log_archiver_global_ref
            and self.config.logging.log_size_update_interval_sec
            > 0  # Check if interval is positive
        ):
            current_time = time.time()
            if (
                current_time - self._last_log_size_update_time
                > self.config.logging.log_size_update_interval_sec
            ):
                try:
                    current_size, archived_size = (
                        self.log_archiver_global_ref.get_total_log_size_info()
                    )
                    self.live_display_manager.update_log_summary_display(
                        current_size, archived_size
                    )
                    self._last_log_size_update_time = current_time
                except AttributeError:
                    logger.debug(
                        "Skipping periodic log size update: log_archiver_global_ref not available on self."
                    )
                except Exception as log_size_e:
                    logger.error(
                        "Error updating log size display periodically: %s",
                        log_size_e,
                        exc_info=True,
                    )

    def train(self, num_iterations: Optional[int] = None):
        """Runs the main CFR+ training loop, potentially in parallel."""

        total_iterations_to_run = (
            num_iterations
            if num_iterations is not None
            else self.config.cfr_training.num_iterations
        )
        last_completed_iteration = self.current_iteration
        start_iter_num = last_completed_iteration + 1
        end_iter_num = last_completed_iteration + total_iterations_to_run

        exploitability_interval = self.config.cfr_training.exploitability_interval
        num_workers = self.config.cfr_training.num_workers
        progress_q_local = self.progress_queue
        archive_q_local = self.archive_queue
        run_log_dir_local = self.run_log_dir
        run_timestamp_local = self.run_timestamp
        display = self.live_display_manager

        logger.info(
            "Starting CFR+ training from iteration %d up to %d (%d workers).",
            start_iter_num,
            end_iter_num,
            num_workers,
        )
        if total_iterations_to_run <= 0:
            logger.warning(
                "Number of iterations to run must be positive. Current: %d, Target: %d",
                self.current_iteration,
                end_iter_num,
            )
            return

        if not display:
            logger.warning(
                "LiveDisplayManager not provided. Console output will be minimal."
            )

        self._total_run_time_start = time.time()
        self._last_log_size_update_time = time.time()  # Initialize for the first check
        pool: Optional[multiprocessing.pool.Pool] = None
        self._worker_statuses = {i: "Initializing" for i in range(num_workers)}
        if display:
            for i in range(num_workers):
                display.update_worker_status(i, "Idle")
            self._try_update_log_size_display()  # Initial log size update

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
                self._try_update_log_size_display()  # Periodic check

                if self.shutdown_event.is_set():
                    logger.warning("Shutdown detected before starting iteration %d.", t)
                    raise GracefulShutdownException("Shutdown detected before iteration")

                iter_start_time = time.time()
                self.current_iteration = t
                min_nodes_this_iter_overall = float("inf")

                if display:
                    for i in range(num_workers):
                        status_to_set: WorkerDisplayStatus = (
                            "Queued" if num_workers > 1 else "Starting",
                            0,
                            0,
                            0,
                            0,
                        )
                        self._worker_statuses[i] = status_to_set
                        display.update_worker_status(i, self._worker_statuses[i])

                    current_total_infosets_str = format_infoset_count(
                        len(self.regret_sum)
                    )
                    if self._total_infosets_str != current_total_infosets_str:
                        self._total_infosets_str = current_total_infosets_str

                    display.update_overall_progress(t)
                    display.update_stats(
                        iteration=t,
                        infosets=self._total_infosets_str,
                        exploitability=self._last_exploit_str,
                    )
                else:
                    for i in range(num_workers):
                        status_to_set_no_disp: WorkerDisplayStatus = (
                            "Queued" if num_workers > 1 else "Starting",
                            0,
                            0,
                            0,
                            0,
                        )
                        self._worker_statuses[i] = status_to_set_no_disp

                try:
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
                    regret_snapshot,
                    progress_q_local,
                    archive_q_local,
                )
                results: List[Optional[WorkerResult]] = [None] * num_workers
                sim_failed_count = 0
                successful_sim_count = 0

                if num_workers == 1:
                    running_status_tuple: WorkerDisplayStatus = (
                        "Running",
                        0,
                        0,
                        0,
                        0,
                    )
                    if display:
                        display.update_worker_status(0, running_status_tuple)
                    self._worker_statuses[0] = running_status_tuple
                    try:
                        worker_args_seq = worker_base_args + (
                            0,
                            run_log_dir_local,
                            run_timestamp_local,
                        )
                        result: Optional[WorkerResult] = run_cfr_simulation_worker(
                            worker_args_seq
                        )
                        results[0] = result
                        if result and isinstance(result, WorkerResult):
                            self._worker_statuses[0] = result.stats
                            min_nodes_this_iter_overall = min(
                                min_nodes_this_iter_overall, result.stats.nodes_visited
                            )
                            if result.stats.error_count > 0:
                                sim_failed_count += 1
                            else:
                                successful_sim_count += 1
                        else:
                            logger.error(
                                "Worker 0 returned unexpected type or None: %s",
                                type(result),
                            )
                            self._worker_statuses[0] = (
                                f"Error (Type: {type(result).__name__})"
                            )
                            sim_failed_count += 1
                        if display:
                            display.update_worker_status(0, self._worker_statuses[0])

                    except Exception:
                        logger.exception("Error during sequential simulation iter %d.", t)
                        self._worker_statuses[0] = "Error (Exception)"
                        if display:
                            display.update_worker_status(0, self._worker_statuses[0])
                        sim_failed_count += 1
                    finally:
                        if progress_q_local:
                            try:
                                while True:
                                    (
                                        prog_w_id,
                                        prog_cur_d,
                                        prog_max_d,
                                        prog_n,
                                        prog_min_d_bt,
                                    ) = progress_q_local.get_nowait()
                                    if prog_w_id == 0:
                                        new_status_tuple_seq: WorkerDisplayStatus = (
                                            "Running",
                                            prog_cur_d,
                                            prog_max_d,
                                            prog_n,
                                            prog_min_d_bt,
                                        )
                                        if isinstance(
                                            self._worker_statuses.get(0),
                                            tuple,
                                        ):
                                            self._worker_statuses[0] = (
                                                new_status_tuple_seq
                                            )
                                            if display:
                                                display.update_worker_status(
                                                    prog_w_id, new_status_tuple_seq
                                                )
                                    self._try_update_log_size_display()  # Check during drain
                            except queue.Empty:
                                pass
                            except Exception as prog_e:
                                logger.error(
                                    "Error draining progress queue (sequential): %s",
                                    prog_e,
                                )
                        if display and isinstance(
                            self._worker_statuses.get(0),
                            (
                                str,
                                WorkerStats,
                            ),
                        ):
                            display.update_worker_status(0, self._worker_statuses[0])
                else:
                    worker_args_list = [
                        worker_base_args
                        + (worker_id, run_log_dir_local, run_timestamp_local)
                        for worker_id in range(num_workers)
                    ]
                    async_results_map: Dict[int, multiprocessing.pool.AsyncResult] = {}
                    pool = None
                    try:
                        pool = multiprocessing.Pool(processes=num_workers)
                        for worker_id, args_for_worker in enumerate(worker_args_list):
                            async_results_map[worker_id] = pool.apply_async(
                                run_cfr_simulation_worker, (args_for_worker,)
                            )
                            status_disp_par: WorkerDisplayStatus = (
                                "Running",
                                0,
                                0,
                                0,
                                0,
                            )
                            self._worker_statuses[worker_id] = status_disp_par
                            if display:
                                display.update_worker_status(worker_id, status_disp_par)
                        pool.close()

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
                            self._try_update_log_size_display()  # Check while waiting

                            if progress_q_local:
                                try:
                                    while not progress_q_local.empty():
                                        (
                                            prog_w_id,
                                            prog_cur_d,
                                            prog_max_d,
                                            prog_n,
                                            prog_min_d_bt,
                                        ) = progress_q_local.get_nowait()
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
                                                prog_min_d_bt,
                                            )
                                            self._worker_statuses[prog_w_id] = (
                                                new_prog_status
                                            )
                                            if display:
                                                display.update_worker_status(
                                                    prog_w_id, new_prog_status
                                                )
                                        self._try_update_log_size_display()  # Check during drain
                                except queue.Empty:
                                    pass
                                except Exception as prog_e:
                                    logger.error(
                                        "Error reading progress queue (parallel): %s",
                                        prog_e,
                                    )

                            for worker_id_done, async_res in list(
                                async_results_map.items()
                            ):
                                if async_res.ready():
                                    try:
                                        result_par: Optional[WorkerResult] = (
                                            async_res.get(timeout=0.01)
                                        )
                                        results[worker_id_done] = result_par

                                        final_status_for_worker: WorkerDisplayStatus
                                        if result_par and isinstance(
                                            result_par, WorkerResult
                                        ):
                                            final_status_for_worker = result_par.stats
                                            min_nodes_this_iter_overall = min(
                                                min_nodes_this_iter_overall,
                                                result_par.stats.nodes_visited,
                                            )
                                            if result_par.stats.error_count > 0:
                                                sim_failed_count += 1
                                            else:
                                                successful_sim_count += 1
                                        else:
                                            sim_failed_count += 1
                                            final_status_for_worker = f"Failed (Type: {type(result_par).__name__})"

                                        self._worker_statuses[worker_id_done] = (
                                            final_status_for_worker
                                        )
                                        if display:
                                            display.update_worker_status(
                                                worker_id_done, final_status_for_worker
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
                                        async_results_map.pop(worker_id_done, None)
                            time.sleep(0.05)
                    except GracefulShutdownException:
                        logger.warning("Graceful shutdown initiated for iter %d pool.", t)
                        self._shutdown_pool(pool)
                        raise
                    except Exception:
                        logger.exception(
                            "Error during parallel pool management iter %d.", t
                        )
                        sim_failed_count = num_workers
                        self._shutdown_pool(pool)
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
                            pool
                            and not self.shutdown_event.is_set()
                            and hasattr(pool, "_state")
                            and pool._state != multiprocessing.pool.TERMINATE
                        ):
                            self._shutdown_pool(pool)

                if display:
                    if min_nodes_this_iter_overall != float("inf"):
                        display._min_worker_nodes_overall_str = format_large_number(
                            int(min_nodes_this_iter_overall)
                        )
                    else:
                        display._min_worker_nodes_overall_str = "N/A"

                if sim_failed_count == num_workers and num_workers > 0:
                    logger.error(
                        "All %d worker simulations reported errors for iteration %d. Skipping merge.",
                        num_workers,
                        t,
                    )
                elif sim_failed_count > 0:
                    logger.warning(
                        "%d/%d worker simulations reported errors for iter %d.",
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
                    and num_workers > 0
                    and successful_sim_count == 0
                ):
                    logger.warning(
                        "No successful worker simulation results for iter %d. Merge will process no data.",
                        t,
                    )

                logger.debug(
                    "Merging results from %d successful workers (out of %d total) for iter %d...",
                    len(valid_results_for_merge),
                    num_workers,
                    t,
                )
                merge_start_time = time.time()
                try:
                    self._merge_local_updates(valid_results_for_merge)
                    logger.debug(
                        "Iter %d merge took %.3fs", t, time.time() - merge_start_time
                    )
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

                iter_time = time.time() - iter_start_time
                self._total_infosets_str = format_infoset_count(len(self.regret_sum))

                if exploitability_interval > 0 and t % exploitability_interval == 0:
                    logger.info("Calculating exploitability at iteration %d...", t)
                    exploit_start_time = time.time()
                    current_avg_strategy = self.compute_average_strategy()
                    if current_avg_strategy is not None:
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
                    for i_disp in range(num_workers):
                        final_status_iter_end = self._worker_statuses.get(i_disp)
                        if final_status_iter_end:
                            display.update_worker_status(i_disp, final_status_iter_end)

                if t % self.config.cfr_training.save_interval == 0:
                    logger.info("Saving progress at iteration %d...", t)
                    self.save_data()

            logger.info("Training loop completed %d iterations.", total_iterations_to_run)

        except (GracefulShutdownException, KeyboardInterrupt) as shutdown_exc:
            exception_type = type(shutdown_exc).__name__
            logger.warning(
                "Shutdown/Interrupt (%s) caught in train loop. Saving progress...",
                exception_type,
            )
            if not self.shutdown_event.is_set():
                self.shutdown_event.set()
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
            raise

        if not self.shutdown_event.is_set():
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
            if final_avg_strategy is not None:
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

            logger.info("Performing final save...")
            self._try_update_log_size_display()  # One last update before saving
            self.save_data()
            logger.info("Final average strategy and data saved.")
            self._write_run_summary()

    def _perform_emergency_save(
        self,
        last_completed_iteration_before_run: int,
        start_iter_num_this_run: int,
        reason: str,
        pool_to_shutdown: Optional[multiprocessing.pool.Pool],
    ):
        logger.warning("Attempting emergency save due to %s...", reason)
        if pool_to_shutdown:
            self._shutdown_pool(pool_to_shutdown)

        iter_to_save_as_completed = self.current_iteration
        if iter_to_save_as_completed >= start_iter_num_this_run:
            iter_to_save_as_completed = self.current_iteration - 1
        else:
            iter_to_save_as_completed = last_completed_iteration_before_run

        if iter_to_save_as_completed < last_completed_iteration_before_run:
            iter_to_save_as_completed = last_completed_iteration_before_run

        if iter_to_save_as_completed >= 0:
            original_current_iter_val_for_save = self.current_iteration
            self.current_iteration = iter_to_save_as_completed
            logger.info(
                "Saving data for last fully completed iteration: %d",
                self.current_iteration,
            )
            try:
                self._try_update_log_size_display()  # Attempt one last update
                self.save_data()
            except Exception as save_e:
                logger.error("Emergency save failed: %s", save_e, exc_info=True)
            finally:
                self.current_iteration = original_current_iter_val_for_save
        else:
            logger.warning(
                "Shutdown (%s) before first new iteration completed relative to loaded data. "
                "No new progress to save beyond iteration %d.",
                reason,
                last_completed_iteration_before_run,
            )

        self._write_run_summary()

    def _write_run_summary(self):
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
                    min_depth_bt_overall = float("inf")
                    total_warnings = 0
                    total_errors = 0
                    completed_workers_with_stats = 0
                    failed_workers_marked = 0

                    for i in range(num_workers):
                        status_info = self._worker_statuses.get(i)
                        status_line = f"Worker {i}: "
                        if isinstance(status_info, WorkerStats):
                            nodes_fmt = format_large_number(status_info.nodes_visited)
                            min_d_bt_fmt = (
                                str(int(status_info.min_depth_after_bottom_out))
                                if status_info.min_depth_after_bottom_out != float("inf")
                                else "N/A"
                            )
                            status_line += (
                                f"Completed (Max Depth: {status_info.max_depth}, "
                                f"Min Depth (BT): {min_d_bt_fmt}, "
                                f"Nodes Visited: {nodes_fmt}, "
                                f"Warnings: {status_info.warning_count}, "
                                f"Errors: {status_info.error_count})"
                            )
                            total_nodes += status_info.nodes_visited
                            max_depth_overall = max(
                                max_depth_overall, status_info.max_depth
                            )
                            if status_info.min_depth_after_bottom_out != float("inf"):
                                min_depth_bt_overall = min(
                                    min_depth_bt_overall,
                                    status_info.min_depth_after_bottom_out,
                                )
                            total_warnings += status_info.warning_count
                            total_errors += status_info.error_count
                            completed_workers_with_stats += 1
                        elif isinstance(status_info, str):
                            status_line += status_info
                            if "Fail" in status_info or "Error" in status_info:
                                failed_workers_marked += 1
                        elif isinstance(status_info, tuple) and len(status_info) == 5:
                            state, cur_d, max_d, nodes_val, min_depth_val_bt = status_info
                            nodes_fmt_tuple = format_large_number(nodes_val)
                            min_d_bt_tuple_fmt = (
                                str(min_depth_val_bt)
                                if min_depth_val_bt != float("inf")
                                else "N/A"
                            )

                            min_depth_diff_str_summary = "N/A"
                            if min_depth_val_bt != float("inf") and min_depth_val_bt > 0:
                                diff = cur_d - min_depth_val_bt
                                min_depth_diff_str_summary = f"{min_depth_val_bt} ({'+' if diff >=0 else ''}{diff})"

                            status_line += f"Stopped ({state}, D:{cur_d}/{max_d}, MinD(BT):{min_d_bt_tuple_fmt}, MinD(Diff):{min_depth_diff_str_summary}, N:{nodes_fmt_tuple})"
                            failed_workers_marked += 1
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
                        f"Workers explicitly marked as Failed/Error/Stopped (string/tuple status): {failed_workers_marked}\n"
                    )
                    f.write(
                        f"Total Nodes Visited (Sum from WorkerStats): {format_large_number(total_nodes)} ({total_nodes:,})\n"
                    )
                    f.write(
                        f"Max Depth Reached (Overall from WorkerStats): {max_depth_overall}\n"
                    )
                    min_d_bt_overall_fmt = (
                        str(int(min_depth_bt_overall))
                        if min_depth_bt_overall != float("inf")
                        else "N/A"
                    )
                    f.write(
                        f"Min Depth (Backtrack) (Overall from WorkerStats): {min_d_bt_overall_fmt}\n"
                    )
                    f.write(
                        f"Total Warnings Reported (Sum from WorkerStats): {total_warnings}\n"
                    )
                    f.write(
                        f"Total Errors Reported (Sum from WorkerStats): {total_errors}\n"
                    )

                elif num_workers == 1 and self._worker_statuses:
                    f.write("Sequential execution (1 worker).\n")
                    status_info_seq = self._worker_statuses.get(0)
                    if isinstance(status_info_seq, WorkerStats):
                        nodes_fmt_seq = format_large_number(status_info_seq.nodes_visited)
                        min_d_bt_seq_fmt = (
                            str(int(status_info_seq.min_depth_after_bottom_out))
                            if status_info_seq.min_depth_after_bottom_out != float("inf")
                            else "N/A"
                        )
                        current_depth_for_diff_seq = status_info_seq.max_depth
                        min_d_bt_val_seq = status_info_seq.min_depth_after_bottom_out
                        min_depth_diff_str_seq = "N/A"
                        if min_d_bt_val_seq != float("inf") and min_d_bt_val_seq > 0:
                            diff_seq = current_depth_for_diff_seq - int(min_d_bt_val_seq)
                            min_depth_diff_str_seq = f"{int(min_d_bt_val_seq)} ({'+' if diff_seq >=0 else ''}{diff_seq})"

                        f.write("  Final Status: Completed (Returned Stats)\n")
                        f.write(f"  Max Depth: {status_info_seq.max_depth}\n")
                        f.write(f"  Min Depth (BT): {min_d_bt_seq_fmt}\n")
                        f.write(f"  Min Depth (Diff): {min_depth_diff_str_seq}\n")
                        f.write(
                            f"  Nodes Visited: {nodes_fmt_seq} ({status_info_seq.nodes_visited:,})\n"
                        )
                        f.write(f"  Warnings: {status_info_seq.warning_count}\n")
                        f.write(f"  Errors: {status_info_seq.error_count}\n")
                    elif isinstance(status_info_seq, str):
                        f.write(f"  Final Status: {status_info_seq}\n")
                    else:
                        f.write(
                            f"  Final Status: Unknown ({type(status_info_seq).__name__})\n"
                        )
                else:
                    f.write(
                        "Worker status information not available or not applicable (e.g. 0 workers configured).\n"
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
        except Exception as e:
            logger.exception("Unexpected error writing run summary: %s", e)
