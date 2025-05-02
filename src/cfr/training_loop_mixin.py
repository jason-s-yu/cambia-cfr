# src/cfr/training_loop_mixin.py
"""Mixin class for orchestrating the CFR+ training loop."""

import logging
import sys
import threading
import time
import copy
import multiprocessing
import multiprocessing.pool
import queue  # For queue.Empty
from typing import Any, Callable, Optional, List, Tuple, Dict, Union

from tqdm import tqdm

from ..analysis_tools import AnalysisTools

# Adjust import if GenericQueue alias is used elsewhere
from ..utils import WorkerResult, PolicyDict, LogQueue as GenericQueue, WorkerStats
from ..config import Config
from .exceptions import GracefulShutdownException
from .worker import run_cfr_simulation_worker


logger = logging.getLogger(__name__)

# Timeout for waiting on worker pool join during shutdown (seconds)
# Currently not used as pool.join doesn't support timeout
# POOL_JOIN_TIMEOUT = 10.0

# Define a type for worker status storage
WorkerStatusInfo = Union[str, WorkerStats, Tuple[str, int, int]]


class CFRTrainingLoopMixin:
    """Handles the overall CFR+ training loop, progress, and scheduling, supporting parallelism."""

    # Attributes expected to be initialized in the main class's __init__
    config: Config
    analysis: "AnalysisTools"
    shutdown_event: "threading.Event"
    current_iteration: int
    exploitability_results: List[Tuple[int, float]]
    regret_sum: PolicyDict
    log_queue: Optional[GenericQueue]
    progress_queue: Optional[GenericQueue]
    # Attributes expected from CFRDataManagerMixin
    load_data: Callable[..., Any]
    save_data: Callable[..., Any]
    _merge_local_updates: Callable[[List[Optional[WorkerResult]]], None]
    compute_average_strategy: Callable[..., Optional[PolicyDict]]
    # Internal state for display
    _last_exploit_str: str = "N/A"
    _total_infosets_str: str = "0"
    # Store worker status: str, WorkerStats (final), or Tuple[str, depth, nodes] (live)
    _worker_statuses: Dict[int, WorkerStatusInfo] = {}

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
        log_queue = self.log_queue  # Get queue from instance
        progress_queue = self.progress_queue  # Get progress queue

        logger.info(
            "Starting CFR+ training from iteration %d up to %d (%d workers).",
            start_iter_num,
            end_iter_num,
            num_workers,
        )
        if total_iterations_to_run <= 0:
            logger.warning("Number of iterations to run must be positive.")
            return

        loop_start_time = time.time()
        pool: Optional[multiprocessing.pool.Pool] = None  # Keep track of the pool object
        self._worker_statuses = {i: "Idle" for i in range(num_workers)}
        progress_bar: Optional[tqdm] = None  # Initialize progress bar reference

        try:
            # Use a single tqdm bar for overall progress
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
            progress_bar.set_postfix(
                {
                    "LastT": "N/A",
                    "Expl": self._last_exploit_str,
                    "Nodes": self._total_infosets_str,
                },
                refresh=False,
            )

            for t in progress_bar:
                if self.shutdown_event.is_set():
                    logger.warning("Shutdown detected before starting iteration %d.", t)
                    raise GracefulShutdownException("Shutdown detected before iteration")

                iter_start_time = time.time()
                self.current_iteration = t
                # Reset worker statuses: Use tuple ("Starting", 0, 0) for live update structure
                self._worker_statuses = {
                    i: ("Starting", 0, 0) for i in range(num_workers)
                }
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

                # Pass progress_queue to worker args
                worker_base_args = (
                    t,
                    self.config,
                    regret_snapshot,
                    log_queue,
                    progress_queue,
                )
                results: List[Optional[WorkerResult]] = [None] * num_workers
                sim_failed_count = 0
                completed_worker_count = 0

                if num_workers == 1:
                    self._worker_statuses[0] = ("Running", 0, 0)  # Initial running status
                    self._update_status_display(progress_bar, t, end_iter_num)
                    try:
                        worker_args = worker_base_args + (0,)  # worker_id = 0
                        # No easy way to get live updates in sequential mode without major refactor
                        # We'll just show the final stats after completion
                        result: Optional[WorkerResult] = run_cfr_simulation_worker(
                            worker_args
                        )
                        results[0] = result
                        completed_worker_count = 1
                        if result is None:
                            sim_failed_count += 1
                            self._worker_statuses[0] = "Failed"
                        elif isinstance(result, WorkerResult):
                            self._worker_statuses[0] = result.stats  # Final stats
                        else:
                            logger.error(
                                "Worker 0 returned unexpected type: %s", type(result)
                            )
                            self._worker_statuses[0] = "Error (Return Type)"
                            sim_failed_count += 1
                    except Exception:
                        logger.exception("Error during sequential simulation iter %d.", t)
                        sim_failed_count += 1
                        self._worker_statuses[0] = "Error"
                    self._update_status_display(progress_bar, t, end_iter_num)

                else:  # Parallel execution
                    worker_args_list = [
                        worker_base_args + (worker_id,)
                        for worker_id in range(num_workers)
                    ]
                    async_results: Dict[int, multiprocessing.pool.AsyncResult] = {}
                    pool = None
                    try:
                        self._worker_statuses = {
                            i: ("Queued", 0, 0) for i in range(num_workers)
                        }
                        self._update_status_display(progress_bar, t, end_iter_num)

                        pool = multiprocessing.Pool(processes=num_workers)

                        for worker_id, args in enumerate(worker_args_list):
                            async_results[worker_id] = pool.apply_async(
                                run_cfr_simulation_worker, (args,)
                            )
                            self._worker_statuses[worker_id] = (
                                "Running",
                                0,
                                0,
                            )  # Initial running status

                        pool.close()
                        self._update_status_display(progress_bar, t, end_iter_num)

                        # --- Result and Progress Collection Loop ---
                        while completed_worker_count < num_workers:
                            if self.shutdown_event.is_set():
                                raise GracefulShutdownException(
                                    "Shutdown during worker execution"
                                )

                            # --- Check for Progress Updates ---
                            if progress_queue:
                                try:
                                    while True:  # Process all available updates
                                        prog_w_id, prog_d, prog_n = (
                                            progress_queue.get_nowait()
                                        )
                                        # Update status only if worker is still considered running
                                        current_status = self._worker_statuses.get(
                                            prog_w_id
                                        )
                                        if (
                                            isinstance(current_status, tuple)
                                            and current_status[0] == "Running"
                                        ):
                                            self._worker_statuses[prog_w_id] = (
                                                "Running",
                                                prog_d,
                                                prog_n,
                                            )
                                except queue.Empty:
                                    pass  # No more progress updates for now
                                except Exception as prog_e:
                                    logger.error(
                                        "Error reading progress queue: %s", prog_e
                                    )

                            # --- Check for Completed Workers ---
                            ready_workers = [
                                worker_id
                                for worker_id, res in async_results.items()
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
                                    if result is None:
                                        sim_failed_count += 1
                                        self._worker_statuses[worker_id] = (
                                            f"Failed (Iter {t})"
                                        )
                                    elif isinstance(result, WorkerResult):
                                        # Store final WorkerStats object upon completion
                                        self._worker_statuses[worker_id] = result.stats
                                    else:
                                        logger.error(
                                            "Worker %d returned unexpected type: %s",
                                            worker_id,
                                            type(result),
                                        )
                                        self._worker_statuses[worker_id] = (
                                            "Error (Return Type)"
                                        )
                                        sim_failed_count += 1
                                except multiprocessing.TimeoutError:
                                    async_results[worker_id] = async_res  # Put back
                                    continue
                                except Exception as pool_e:
                                    logger.error(
                                        "!!! Exception type '%s' fetching result worker %d iter %d.",
                                        type(pool_e).__name__,
                                        worker_id,
                                        t,
                                    )
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
                                        f"Error ({type(pool_e).__name__})"
                                    )

                            # Update display after checking progress and completion
                            self._update_status_display(progress_bar, t, end_iter_num)

                            if completed_worker_count < num_workers:
                                time.sleep(0.1)  # Avoid busy-waiting

                    except GracefulShutdownException:
                        raise
                    except Exception as e:
                        logger.exception(
                            "Error during parallel pool management iter %d.", t
                        )
                        sim_failed_count = num_workers
                        raise e

                # --- Process Results (Merge) ---
                # (Merge logic remains the same)
                if sim_failed_count == num_workers and num_workers > 0:
                    progress_bar.set_postfix(
                        {
                            "LastT": "N/A",
                            "Expl": self._last_exploit_str,
                            "Nodes": self._total_infosets_str,
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
                    valid_results = [res for res in results if res is not None]
                    if not valid_results:
                        progress_bar.set_postfix(
                            {
                                "LastT": "N/A",
                                "Expl": self._last_exploit_str,
                                "Nodes": self._total_infosets_str,
                            },
                            refresh=False,
                        )
                        progress_bar.set_postfix_str("All Sims FAILED", refresh=True)
                        logger.error(
                            "All simulations failed for iteration %d. Skipping merge.", t
                        )
                        continue
                else:
                    valid_results = results

                progress_bar.set_postfix(
                    {
                        "LastT": "N/A",
                        "Expl": self._last_exploit_str,
                        "Nodes": self._total_infosets_str,
                    },
                    refresh=False,
                )
                progress_bar.set_postfix_str("Merging...", refresh=False)
                merge_start_time = time.time()
                try:
                    self._merge_local_updates(valid_results)
                    merge_time = time.time() - merge_start_time
                    logger.debug("Iter %d merge took %.3fs", t, merge_time)
                except Exception:
                    logger.exception("Error merging results iter %d.", t)
                    progress_bar.set_postfix(
                        {
                            "LastT": "N/A",
                            "Expl": self._last_exploit_str,
                            "Nodes": self._total_infosets_str,
                        },
                        refresh=False,
                    )
                    progress_bar.set_postfix_str("Merge FAILED", refresh=True)
                    continue

                # --- Iteration Completed ---
                # (Rest of iteration completion logic remains the same)
                iter_time = time.time() - iter_start_time
                self._total_infosets_str = f"{len(self.regret_sum):,}"

                exploit_calc_time = 0.0
                if exploitability_interval > 0 and t % exploitability_interval == 0:
                    progress_bar.set_postfix(
                        {
                            "LastT": f"{iter_time:.2f}s",
                            "Expl": self._last_exploit_str,
                            "Nodes": self._total_infosets_str,
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
                                "Nodes": self._total_infosets_str,
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

                postfix_dict = {
                    "LastT": f"{iter_time:.2f}s",
                    "Expl": self._last_exploit_str,
                    "Nodes": self._total_infosets_str,
                }
                progress_bar.set_postfix(postfix_dict, refresh=True)
                self._update_status_display(
                    progress_bar, t, end_iter_num
                )  # Final update for iter

                if t % self.config.cfr_training.save_interval == 0:
                    progress_bar.set_postfix(postfix_dict, refresh=False)
                    progress_bar.set_postfix_str("Saving...", refresh=True)
                    self.save_data()
                    progress_bar.set_postfix(postfix_dict, refresh=True)

        # --- Main Loop Exception Handling ---
        # (Exception handling logic remains the same, including pool termination)
        except (
            GracefulShutdownException,
            KeyboardInterrupt,
        ) as shutdown_exc:
            exception_type = type(shutdown_exc).__name__
            logger.warning(
                "Shutdown/Interrupt (%s) caught in train loop. Terminating pool and saving progress...",
                exception_type,
            )
            if num_workers > 1 and pool is not None:
                logger.warning("Terminating worker pool due to %s...", exception_type)
                pool.terminate()
                try:
                    logger.debug(
                        "Joining worker pool after %s terminate...", exception_type
                    )
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
                    logger.info(
                        "Progress saved successfully after %s (state as of iter %d completion).",
                        exception_type,
                        self.current_iteration,
                    )
                except Exception as save_e:
                    logger.error(
                        "Failed to save progress during shutdown (%s): %s",
                        exception_type,
                        save_e,
                    )
                finally:
                    self.current_iteration = saved_iter
            else:
                logger.warning(
                    "Shutdown (%s) before first new iteration completed or loaded. No progress to save.",
                    exception_type,
                )
            raise GracefulShutdownException(
                f"{exception_type} processed"
            ) from shutdown_exc
        except Exception as main_loop_e:
            logger.exception("Unhandled exception in main training loop:")
            if num_workers > 1 and pool is not None:
                logger.warning("Terminating worker pool due to unhandled error...")
                pool.terminate()
                try:
                    logger.debug("Joining worker pool after error terminate...")
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
                    logger.info(
                        "Emergency save completed for iteration %d.",
                        self.current_iteration,
                    )
                except Exception as save_e:
                    logger.error("Emergency save failed: %s", save_e)
                finally:
                    self.current_iteration = saved_iter
            raise main_loop_e
        finally:
            # Final pool cleanup just in case
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
            num_status_lines = num_workers + 2 if num_workers > 0 else 0
            if sys.stderr.isatty() and num_status_lines > 0:
                pass

        # --- Training Loop Finished Normally ---
        # (Final summary, exploitability, save logic remains the same)
        end_time = time.time()
        total_completed_in_run = self.current_iteration - last_completed_iteration
        logger.info("Training loop finished %d iterations.", total_completed_in_run)
        logger.info(
            "Total training time this run: %.2f seconds.", end_time - loop_start_time
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
        final_status_msg = f"\nFinished. Iter: {self.current_iteration} | Nodes: {self._total_infosets_str} | Expl: {self._last_exploit_str}"
        print(final_status_msg, file=sys.stderr)
        self.save_data()
        logger.info("Final average strategy and data saved.")

    def _update_status_display(
        self, progress_bar: Optional[tqdm], current_iter: int, total_iter: int
    ):
        """Updates the main progress bar postfix and prints worker statuses above it."""
        if not progress_bar:
            return

        # --- Update postfix for the main bar ---
        last_t_value = "N/A"
        if isinstance(progress_bar.postfix, dict):
            last_t_value = progress_bar.postfix.get("LastT", "N/A")
        elif isinstance(progress_bar.postfix, str):
            last_t_value = "N/A"

        current_postfix_dict = {
            "LastT": last_t_value,
            "Expl": self._last_exploit_str,
            "Nodes": self._total_infosets_str,
        }
        progress_bar.set_postfix(current_postfix_dict, refresh=False)

        # --- Prepare status lines for workers ---
        status_lines = []
        num_workers = self.config.cfr_training.num_workers
        if num_workers > 0:
            header = f"--- Iteration {current_iter}/{total_iter} Worker Status ---"
            status_lines.append(header)
            max_len = len(header)
            for i in range(num_workers):
                status_info = self._worker_statuses.get(i, "Unknown")
                status_str = ""
                # --- NEW: Format based on status_info type ---
                if isinstance(status_info, WorkerStats):
                    # Final stats after completion
                    status_str = f"Done (D:{status_info.max_depth}, N:{status_info.nodes_visited:,})"
                elif isinstance(status_info, tuple) and len(status_info) == 3:
                    # Live update: ("Running", depth, nodes)
                    state, depth, nodes = status_info
                    status_str = f"{state} (D:{depth}, N:{nodes:,})"
                elif isinstance(status_info, str):
                    # Simple string status (Idle, Failed, Error, Queued, Starting)
                    status_str = status_info
                else:
                    status_str = f"Unknown State ({type(status_info).__name__})"
                # --- END NEW ---

                line = f" W{i:<2}: {status_str}"
                status_lines.append(line)
                max_len = max(max_len, len(line))
            status_lines.append("-" * max_len)

            clear_lines = num_workers + 2
            if sys.stderr.isatty():
                tqdm.write(
                    f"\033[{clear_lines}F\033[J\033[{clear_lines}F",
                    file=sys.stderr,
                    end="",
                )

            status_block = "\n".join(status_lines)
            tqdm.write(status_block, file=sys.stderr, end="\n")

        progress_bar.refresh()
