"""src/cfr/training_loop_mixin.py"""

import copy
import logging
import multiprocessing
import multiprocessing.pool
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from ..analysis_tools import AnalysisTools
from ..config import Config

# Import the new formatting utility
from ..utils import (
    LogQueue as ProgressQueue,
    PolicyDict,
    WorkerResult,
    WorkerStats,
    format_large_number,
)
from .exceptions import GracefulShutdownException
from .worker import run_cfr_simulation_worker

logger = logging.getLogger(__name__)

# Define a type for worker status storage - extended tuple for live updates
WorkerStatusInfo = Union[
    str, WorkerStats, Tuple[str, int, int, int]
]  # state, cur_d, max_d, nodes


def format_infoset_count(count: int) -> str:
    """Formats a large number with k/M suffixes for the main progress bar."""
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count // 1000}k"
    else:
        return f"{count / 1_000_000:.1f}M"


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
    # Attributes from CFRDataManagerMixin
    load_data: Callable[..., Any]
    save_data: Callable[..., Any]
    _merge_local_updates: Callable[[List[Optional[WorkerResult]]], None]
    compute_average_strategy: Callable[..., Optional[PolicyDict]]
    # Internal state
    _last_exploit_str: str = "N/A"
    _total_infosets_str: str = "0"  # Initialize without 'k'
    _worker_statuses: Dict[int, WorkerStatusInfo] = {}
    _total_run_time_start: float = 0.0  # Track start time

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

        logger.info(
            "Starting CFR+ training from iteration %d up to %d (%d workers).",
            start_iter_num,
            end_iter_num,
            num_workers,
        )
        if total_iterations_to_run <= 0:
            logger.warning("Number of iterations to run must be positive.")
            return

        self._total_run_time_start = time.time()
        pool: Optional[multiprocessing.pool.Pool] = None
        self._worker_statuses = {i: "Idle" for i in range(num_workers)}
        # Update total infosets display initially
        self._total_infosets_str = format_infoset_count(len(self.regret_sum))
        progress_bar: Optional[tqdm] = None

        try:
            progress_bar = tqdm(
                range(start_iter_num, end_iter_num + 1),
                desc=f"CFR+ Training ({num_workers}W)",
                total=total_iterations_to_run,
                unit="iter",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                position=0,
                leave=True,
                file=sys.stderr,
            )
            # Use Infosets in initial postfix
            progress_bar.set_postfix(
                {
                    "LastT": "N/A",
                    "Expl": self._last_exploit_str,
                    "Infosets": self._total_infosets_str,
                },
                refresh=False,
            )

            for t in progress_bar:
                if self.shutdown_event.is_set():
                    logger.warning("Shutdown detected before starting iteration %d.", t)
                    raise GracefulShutdownException("Shutdown detected before iteration")

                iter_start_time = time.time()
                self.current_iteration = t
                self._worker_statuses = {
                    i: ("Starting", 0, 0, 0)
                    for i in range(num_workers)  # state, cur_d, max_d, nodes
                }
                # Update total infoset count for display before starting iteration logic
                self._total_infosets_str = format_infoset_count(len(self.regret_sum))
                self._update_status_display(progress_bar, t, end_iter_num)

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
                    # Sequential execution
                    self._worker_statuses[0] = ("Running", 0, 0, 0)
                    self._update_status_display(progress_bar, t, end_iter_num)
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
                            sim_failed_count += 1
                            # Add error to stats if possible from a potential partial return
                            if isinstance(result, WorkerResult) and isinstance(
                                result.stats, WorkerStats
                            ):
                                result.stats.error_count += 1
                            elif isinstance(
                                self._worker_statuses[0], WorkerStats
                            ):  # Should not happen
                                self._worker_statuses[0].error_count += 1
                        elif isinstance(result, WorkerResult):
                            self._worker_statuses[0] = result.stats
                    except Exception as e:
                        logger.exception("Error during sequential simulation iter %d.", t)
                        sim_failed_count += 1
                        # Try to capture stats with error count if status was already WorkerStats
                        if isinstance(self._worker_statuses[0], WorkerStats):
                            self._worker_statuses[0].error_count += 1
                            self._worker_statuses[0] = (
                                f"Error ({type(e).__name__})"  # Overwrite with error string
                            )
                        else:
                            self._worker_statuses[0] = (
                                "Error (Exception)"  # Default error message
                            )
                    self._update_status_display(progress_bar, t, end_iter_num)

                else:
                    # Parallel execution
                    worker_args_list = [
                        worker_base_args + (worker_id, run_log_dir, run_timestamp)
                        for worker_id in range(num_workers)
                    ]
                    async_results: Dict[int, multiprocessing.pool.AsyncResult] = {}
                    pool = None
                    try:
                        self._worker_statuses = {
                            i: ("Queued", 0, 0, 0) for i in range(num_workers)
                        }
                        self._update_status_display(progress_bar, t, end_iter_num)

                        pool = multiprocessing.Pool(processes=num_workers)
                        for worker_id, args in enumerate(worker_args_list):
                            async_results[worker_id] = pool.apply_async(
                                run_cfr_simulation_worker, (args,)
                            )
                            self._worker_statuses[worker_id] = ("Running", 0, 0, 0)

                        pool.close()
                        self._update_status_display(progress_bar, t, end_iter_num)

                        # Result and Progress Collection Loop
                        while completed_worker_count < num_workers:
                            if self.shutdown_event.is_set():
                                raise GracefulShutdownException(
                                    "Shutdown during worker execution"
                                )

                            # Check for Progress Updates
                            if progress_queue:
                                try:
                                    while True:
                                        # Expect worker_id, current_depth, max_depth, nodes
                                        prog_w_id, prog_cur_d, prog_max_d, prog_n = (
                                            progress_queue.get_nowait()
                                        )
                                        current_status = self._worker_statuses.get(
                                            prog_w_id
                                        )
                                        # Only update if worker is still 'Running'
                                        if (
                                            isinstance(current_status, tuple)
                                            and current_status[0] == "Running"
                                        ):
                                            self._worker_statuses[prog_w_id] = (
                                                "Running",
                                                prog_cur_d,
                                                prog_max_d,
                                                prog_n,
                                            )
                                except queue.Empty:
                                    pass
                                except Exception as prog_e:
                                    logger.error(
                                        "Error reading progress queue: %s", prog_e
                                    )

                            # Check for Completed Workers
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
                                        # Try to get stats if it's a WorkerResult without data
                                        if isinstance(
                                            result, WorkerResult
                                        ) and isinstance(result.stats, WorkerStats):
                                            self._worker_statuses[worker_id] = (
                                                result.stats
                                            )  # Keep stats
                                            if (
                                                self._worker_statuses[
                                                    worker_id
                                                ].error_count
                                                == 0
                                            ):
                                                self._worker_statuses[
                                                    worker_id
                                                ].error_count = (
                                                    1  # Ensure error is counted
                                                )
                                        else:
                                            self._worker_statuses[worker_id] = (
                                                f"Failed (Iter {t}, Type: {type(result).__name__})"
                                            )
                                    elif isinstance(result, WorkerResult):
                                        self._worker_statuses[worker_id] = result.stats
                                except multiprocessing.TimeoutError:
                                    async_results[worker_id] = async_res  # Put back
                                    continue
                                except Exception as pool_e:
                                    logger.error(
                                        "Error fetching result worker %d iter %d: %s",
                                        worker_id,
                                        t,
                                        pool_e,
                                        exc_info=False,
                                    )
                                    results[worker_id] = None  # Indicate failure
                                    completed_worker_count += 1
                                    sim_failed_count += 1
                                    # Create basic stats with error count for summary
                                    error_stats = WorkerStats(error_count=1)
                                    self._worker_statuses[worker_id] = error_stats

                            self._update_status_display(progress_bar, t, end_iter_num)
                            if completed_worker_count < num_workers:
                                time.sleep(0.1)

                    except GracefulShutdownException:
                        raise
                    except Exception as e:
                        logger.exception(
                            "Error during parallel pool management iter %d.", t
                        )
                        sim_failed_count = num_workers
                        raise e

                # --- Process Results (Merge) ---
                if sim_failed_count == num_workers and num_workers > 0:
                    progress_bar.set_postfix(
                        {
                            "LastT": "N/A",
                            "Expl": self._last_exploit_str,
                            "Infosets": self._total_infosets_str,
                        },
                        refresh=False,
                    )
                    progress_bar.set_postfix_str("All Sims FAILED", refresh=True)
                    logger.error(
                        "All simulations failed for iteration %d. Skipping merge.", t
                    )
                    continue
                if sim_failed_count > 0:
                    logger.warning(
                        "%d/%d worker sims failed for iter %d.",
                        sim_failed_count,
                        num_workers,
                        t,
                    )
                    # Filter results to only include successful WorkerResult objects for merging
                    valid_results = [
                        res for res in results if isinstance(res, WorkerResult)
                    ]
                    if not valid_results or all(
                        r.regret_updates is None for r in valid_results
                    ):  # Check if data is actually present
                        progress_bar.set_postfix(
                            {
                                "LastT": "N/A",
                                "Expl": self._last_exploit_str,
                                "Infosets": self._total_infosets_str,
                            },
                            refresh=False,
                        )
                        progress_bar.set_postfix_str(
                            "All Sims FAILED (No Data)", refresh=True
                        )
                        logger.error(
                            "All simulations failed or returned no update data for iteration %d. Skipping merge.",
                            t,
                        )
                        continue
                else:  # No failures
                    valid_results = [
                        res for res in results if isinstance(res, WorkerResult)
                    ]

                progress_bar.set_postfix(
                    {
                        "LastT": "N/A",
                        "Expl": self._last_exploit_str,
                        "Infosets": self._total_infosets_str,
                    },
                    refresh=False,
                )
                progress_bar.set_postfix_str("Merging...", refresh=False)
                merge_start_time = time.time()
                try:
                    self._merge_local_updates(valid_results)  # Pass only valid results
                    merge_time = time.time() - merge_start_time
                    logger.debug("Iter %d merge took %.3fs", t, merge_time)
                except Exception:
                    logger.exception("Error merging results iter %d.", t)
                    progress_bar.set_postfix(
                        {
                            "LastT": "N/A",
                            "Expl": self._last_exploit_str,
                            "Infosets": self._total_infosets_str,
                        },
                        refresh=False,
                    )
                    progress_bar.set_postfix_str("Merge FAILED", refresh=True)
                    continue

                # --- Iteration Completed ---
                iter_time = time.time() - iter_start_time
                # Update total infoset count string AFTER merge for the postfix
                self._total_infosets_str = format_infoset_count(len(self.regret_sum))

                exploit_calc_time = 0.0
                if exploitability_interval > 0 and t % exploitability_interval == 0:
                    progress_bar.set_postfix(
                        {
                            "LastT": f"{iter_time:.2f}s",
                            "Expl": self._last_exploit_str,
                            "Infosets": self._total_infosets_str,
                        },
                        refresh=False,
                    )
                    progress_bar.set_postfix_str("Avg Strat...", refresh=False)
                    exploit_start_time = time.time()
                    logger.info("Calculating exploitability at iteration %d...", t)
                    current_avg_strategy = self.compute_average_strategy()
                    if current_avg_strategy:
                        progress_bar.set_postfix(
                            {
                                "LastT": f"{iter_time:.2f}s",
                                "Expl": self._last_exploit_str,
                                "Infosets": self._total_infosets_str,
                            },
                            refresh=False,
                        )
                        progress_bar.set_postfix_str("Exploit...", refresh=False)
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

                # Update postfix with final iter time and potentially new exploitability
                postfix_dict = {
                    "LastT": f"{iter_time:.2f}s",
                    "Expl": self._last_exploit_str,
                    "Infosets": self._total_infosets_str,
                }
                progress_bar.set_postfix(postfix_dict, refresh=True)
                # Final status display update for this iteration
                self._update_status_display(progress_bar, t, end_iter_num)

                if t % self.config.cfr_training.save_interval == 0:
                    progress_bar.set_postfix(postfix_dict, refresh=False)
                    progress_bar.set_postfix_str("Saving...", refresh=True)
                    self.save_data()  # type: ignore
                    progress_bar.set_postfix(
                        postfix_dict, refresh=True
                    )  # Restore normal postfix

        # --- Main Loop Exception Handling ---
        except (GracefulShutdownException, KeyboardInterrupt) as shutdown_exc:
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
                    logger.debug("Worker pool joined after %s.", exception_type)
                except Exception as join_e:
                    logger.error(
                        "Error joining pool after %s: %s", exception_type, join_e
                    )
                pool = None
            completed_iter_to_save = (
                self.current_iteration - 1
                if hasattr(self, "current_iteration")
                and self.current_iteration > last_completed_iteration
                else last_completed_iteration
            )
            if completed_iter_to_save >= 0:
                saved_iter = self.current_iteration
                self.current_iteration = completed_iter_to_save
                try:
                    self.save_data()
                    logger.info("Progress saved successfully after %s (state as of iter %d completion).", exception_type, self.current_iteration)  # type: ignore
                except Exception as save_e:
                    logger.error(
                        "Failed to save progress during shutdown (%s): %s",
                        exception_type,
                        save_e,
                    )
                finally:
                    self.current_iteration = (
                        saved_iter  # Restore for potential summary write
                    )
            else:
                logger.warning(
                    "Shutdown (%s) before first new iteration completed or loaded. No progress to save.",
                    exception_type,
                )
            raise GracefulShutdownException(
                f"{exception_type} processed"
            ) from shutdown_exc  # Re-raise specific exception
        except Exception as main_loop_e:
            logger.exception("Unhandled exception in main training loop:")
            if num_workers > 1 and pool is not None:
                logger.warning("Terminating worker pool due to unhandled error...")
                pool.terminate()
                try:
                    pool.join()
                    logger.debug("Worker pool joined after error.")
                except Exception as join_e:
                    logger.error(
                        "Error joining worker pool during error handling: %s", join_e
                    )
                pool = None
            logger.warning("Attempting emergency save after unhandled exception...")
            completed_iter_to_save = (
                self.current_iteration - 1
                if hasattr(self, "current_iteration")
                and self.current_iteration > last_completed_iteration
                else last_completed_iteration
            )
            if completed_iter_to_save >= 0:
                saved_iter = self.current_iteration
                self.current_iteration = completed_iter_to_save
                try:
                    self.save_data()
                    logger.info("Emergency save completed for iteration %d.", self.current_iteration)  # type: ignore
                except Exception as save_e:
                    logger.error("Emergency save failed: %s", save_e)
                finally:
                    self.current_iteration = (
                        saved_iter  # Restore for potential summary write
                    )
            raise main_loop_e
        finally:
            # Final pool cleanup
            if num_workers > 1 and pool is not None:
                logger.warning(
                    "Terminating/Joining pool in main finally block (unexpected loop exit?)."
                )
                pool.terminate()
                try:
                    pool.join()
                    logger.debug("Pool joined in main finally block.")
                except Exception as final_join_e:
                    logger.error(
                        "Error joining pool in main finally block: %s", final_join_e
                    )
                pool = None
            # Cleanup progress bar
            if progress_bar:
                progress_bar.close()
            # Write summary file AFTER potential save in finally block
            self._write_run_summary()

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
                # Update last entry if calculation happened at the very end
                self.exploitability_results[-1] = (self.current_iteration, final_exploit)
            logger.info("Final exploitability: %.4f", final_exploit)
            self._last_exploit_str = (
                f"{final_exploit:.4f}" if final_exploit != float("inf") else "N/A"
            )
        else:
            logger.warning("Could not compute final average strategy.")
            self._last_exploit_str = "N/A (Avg Err)"

        # Update final Infosets count
        self._total_infosets_str = format_infoset_count(len(self.regret_sum))

        # Save final data and write summary (Summary write now happens in finally too)
        self.save_data()  # type: ignore
        logger.info("Final average strategy and data saved.")

        final_status_msg = f"\nFinished. Iter: {self.current_iteration} | Infosets: {self._total_infosets_str} | Expl: {self._last_exploit_str}"
        print(final_status_msg, file=sys.stderr)

    def _update_status_display(
        self, progress_bar: Optional[tqdm], current_iter: int, total_iter: int
    ):
        """Updates the main progress bar postfix and prints worker statuses above it."""
        if not progress_bar:
            return

        # --- Update postfix for the main bar (using Infosets) ---
        last_t_value = "N/A"
        if isinstance(progress_bar.postfix, dict):
            last_t_value = progress_bar.postfix.get("LastT", "N/A")
        elif isinstance(progress_bar.postfix, str):
            last_t_value = "N/A"  # Should not happen with dict usage
        # Recalculate Infosets string based on current regret_sum size
        self._total_infosets_str = format_infoset_count(len(self.regret_sum))
        current_postfix_dict = {
            "LastT": last_t_value,
            "Expl": self._last_exploit_str,
            "Infosets": self._total_infosets_str,
        }
        progress_bar.set_postfix(current_postfix_dict, refresh=False)

        # --- Prepare status lines for workers (using new formatting) ---
        status_lines = []
        num_workers = self.config.cfr_training.num_workers
        if num_workers > 0:
            header = f"--- Iteration {current_iter}/{total_iter} Worker Status ---"
            status_lines.append(header)
            max_len = len(header)
            for i in range(num_workers):
                status_info = self._worker_statuses.get(i, "Unknown")
                status_str = ""
                if isinstance(status_info, WorkerStats):
                    # Format final stats: Use format_large_number for nodes
                    nodes_fmt = format_large_number(status_info.nodes_visited)
                    status_str = f"Done (D:{status_info.max_depth}, N:{nodes_fmt}, W:{status_info.warning_count}, E:{status_info.error_count})"
                elif isinstance(status_info, tuple) and len(status_info) == 4:
                    # Format live stats: Use format_large_number for nodes, show current/max depth
                    state, current_depth, max_depth, nodes = status_info
                    nodes_fmt = format_large_number(nodes)
                    status_str = f"{state} (D:{current_depth}/{max_depth}, N:{nodes_fmt})"
                elif isinstance(status_info, str):
                    status_str = status_info
                else:
                    status_str = f"Unknown State ({type(status_info).__name__})"

                line = f" W{i:<2}: {status_str}"
                status_lines.append(line)
                max_len = max(max_len, len(line))
            status_lines.append("-" * max_len)

            clear_lines = num_workers + 2
            if sys.stderr.isatty():
                # Move cursor up, clear lines, move cursor up again
                tqdm.write(
                    f"\033[{clear_lines}F\033[J\033[{clear_lines}F",
                    file=sys.stderr,
                    end="",
                )

            status_block = "\n".join(status_lines)
            tqdm.write(status_block, file=sys.stderr, end="\n")  # Write block and newline

        progress_bar.refresh()

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
                # Attempt to get config path if stored (might need modification)
                config_path = getattr(
                    self.config, "_source_path", "N/A"
                )  # Example attribute name
                f.write(f"Config File: {config_path}\n")
                f.write(f"Total Iterations Completed: {self.current_iteration}\n")
                total_time = (
                    time.time() - self._total_run_time_start
                    if self._total_run_time_start > 0
                    else 0
                )
                f.write(f"Total Run Time: {total_time:.2f} seconds\n")
                f.write(f"Final Exploitability: {self._last_exploit_str}\n")
                # Write total infosets with full number in summary file
                f.write(f"Total Unique Infosets: {len(self.regret_sum):,}\n")
                f.write("\n--- Worker Summary ---\n")

                num_workers = self.config.cfr_training.num_workers
                if num_workers > 0:
                    total_nodes = 0
                    max_depth_overall = 0
                    total_warnings = 0
                    total_errors = 0
                    completed_workers = 0
                    failed_workers = (
                        0  # Workers returning non-WorkerStats or with error status string
                    )
                    error_in_stats_workers = (
                        0  # Workers returning WorkerStats but error_count > 0
                    )

                    for i in range(num_workers):
                        status_info = self._worker_statuses.get(i)
                        status_line = f"Worker {i}: "
                        if isinstance(status_info, WorkerStats):
                            # Use format_large_number for node display in summary too
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
                            status_line += status_info  # e.g., "Failed", "Error", "Idle"
                            if "Fail" in status_info or "Error" in status_info:
                                failed_workers += 1
                                total_errors += (
                                    1  # Count failure/error string as an error
                                )
                        # Handle live status tuple in summary (should ideally be WorkerStats or str)
                        elif isinstance(status_info, tuple) and len(status_info) == 4:
                            state, cur_d, max_d, nodes = status_info
                            nodes_fmt = format_large_number(nodes)
                            status_line += (
                                f"Stopped ({state}, D:{cur_d}/{max_d}, N:{nodes_fmt})"
                            )
                            failed_workers += (
                                1  # Treat stopped mid-run as a failure/incomplete state
                            )
                            total_errors += 1
                        else:
                            status_line += (
                                f"Unknown Final State ({type(status_info).__name__})"
                            )
                            failed_workers += 1  # Count unknown as failure
                            total_errors += 1
                        f.write(status_line + "\n")

                    f.write("\n--- Aggregated Worker Stats ---\n")
                    f.write(f"Completed Workers (Returned Stats): {completed_workers}\n")
                    f.write(f"Failed/Error Workers (String Status): {failed_workers}\n")
                    f.write(f"Workers with Errors in Stats: {error_in_stats_workers}\n")
                    # Format aggregated numbers nicely in summary
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
                    # Handle tuple case in summary if sequential run interrupted
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
