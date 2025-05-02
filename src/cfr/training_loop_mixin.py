# src/cfr/training_loop_mixin.py
"""Mixin class for orchestrating the CFR+ training loop."""

import logging
import sys
import threading
import time
import copy
import multiprocessing
import multiprocessing.pool
from typing import Any, Callable, Optional, List, Tuple, Dict, Union

from tqdm import tqdm

from ..analysis_tools import AnalysisTools
from ..utils import WorkerResult, PolicyDict, LogQueue, WorkerStats
from ..config import Config
from .exceptions import GracefulShutdownException
from .worker import run_cfr_simulation_worker


logger = logging.getLogger(__name__)

# Timeout for waiting on worker pool join during shutdown (seconds)
# Currently not used as pool.join doesn't support timeout
# POOL_JOIN_TIMEOUT = 10.0


class CFRTrainingLoopMixin:
    """Handles the overall CFR+ training loop, progress, and scheduling, supporting parallelism."""

    # Attributes expected to be initialized in the main class's __init__
    config: Config
    analysis: "AnalysisTools"
    shutdown_event: "threading.Event"
    current_iteration: int
    exploitability_results: List[Tuple[int, float]]
    regret_sum: PolicyDict
    log_queue: Optional[LogQueue]
    # Attributes expected from CFRDataManagerMixin
    load_data: Callable[..., Any]
    save_data: Callable[..., Any]
    _merge_local_updates: Callable[[List[Optional[WorkerResult]]], None]
    compute_average_strategy: Callable[..., Optional[PolicyDict]]
    # Internal state for display
    _last_exploit_str: str = "N/A"
    _total_infosets_str: str = "0"
    # Store worker status strings or WorkerStats objects
    _worker_statuses: Dict[int, Union[str, WorkerStats]] = {}

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

        # Use a single tqdm bar for overall progress
        # Worker status will be printed above using tqdm.write
        progress_bar = tqdm(
            range(start_iter_num, end_iter_num + 1),
            desc=f"CFR+ Training ({num_workers}W)",
            total=total_iterations_to_run,
            unit="iter",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            position=0,  # Main bar at position 0
            leave=True,
        )
        # Initialize postfix early to prevent AttributeError
        progress_bar.set_postfix(
            {
                "LastT": "N/A",
                "Expl": self._last_exploit_str,
                "Nodes": self._total_infosets_str,
            },
            refresh=False,
        )

        try:
            # 't' is the iteration number currently being executed
            for t in progress_bar:
                # --- Check for Shutdown Signal EARLY ---
                if self.shutdown_event.is_set():
                    logger.warning("Shutdown detected before starting iteration %d.", t)
                    raise GracefulShutdownException("Shutdown detected before iteration")

                iter_start_time = time.time()
                self.current_iteration = t
                # Reset worker statuses at the start of the iteration
                self._worker_statuses = {i: "Starting" for i in range(num_workers)}
                self._update_status_display(progress_bar, t, end_iter_num)

                # --- Prepare for Simulation(s) ---
                try:
                    # Deep copy might be slow for very large dicts, but necessary for snapshot isolation
                    regret_sum_copy = copy.deepcopy(self.regret_sum)
                    # Convert back to dict for pickling if needed, but workers handle defaultdicts
                    regret_snapshot = dict(regret_sum_copy)
                except Exception as e:
                    logger.error(
                        "Failed to prepare regret snapshot: %s", e, exc_info=True
                    )
                    raise RuntimeError(
                        f"Could not prepare regret snapshot for iter {t}"
                    ) from e

                # Worker arguments now include the log_queue
                worker_base_args = (
                    t,
                    self.config,
                    regret_snapshot,
                    log_queue,  # Pass queue
                )
                # Initialize results list with Nones
                results: List[Optional[WorkerResult]] = [None] * num_workers
                sim_failed_count = 0
                completed_worker_count = 0

                # --- Run Simulation(s) ---
                if num_workers == 1:
                    self._worker_statuses[0] = "Running"
                    self._update_status_display(progress_bar, t, end_iter_num)
                    try:
                        worker_args = worker_base_args + (0,)
                        # Worker returns WorkerResult object or None
                        result: Optional[WorkerResult] = run_cfr_simulation_worker(
                            worker_args
                        )
                        results[0] = result  # Store result object
                        completed_worker_count = 1
                        if result is None:
                            sim_failed_count += 1
                            self._worker_statuses[0] = "Failed"
                        elif isinstance(result, WorkerResult):
                            # Store WorkerStats object for display
                            self._worker_statuses[0] = result.stats
                    except Exception:
                        logger.exception("Error during sequential simulation iter %d.", t)
                        sim_failed_count += 1
                        self._worker_statuses[0] = "Error"
                    self._update_status_display(progress_bar, t, end_iter_num)

                else:
                    # --- Parallel Execution with Improved Shutdown ---
                    worker_args_list = [
                        worker_base_args + (worker_id,)
                        for worker_id in range(num_workers)
                    ]
                    # Store AsyncResult objects indexed by worker_id
                    async_results: Dict[int, multiprocessing.pool.AsyncResult] = {}
                    try:
                        self._worker_statuses = {i: "Queued" for i in range(num_workers)}
                        self._update_status_display(progress_bar, t, end_iter_num)

                        # Create pool within the try block for proper cleanup
                        pool = multiprocessing.Pool(processes=num_workers)

                        # Submit tasks and store AsyncResult objects indexed by worker_id
                        for worker_id, args in enumerate(worker_args_list):
                            async_results[worker_id] = pool.apply_async(
                                run_cfr_simulation_worker, (args,)
                            )
                            self._worker_statuses[worker_id] = "Running"

                        pool.close()  # No more tasks will be submitted
                        self._update_status_display(
                            progress_bar, t, end_iter_num
                        )  # Show all as running

                        # Collect results with timeout and update status
                        # start_collection_time = time.time()
                        while completed_worker_count < num_workers:
                            if self.shutdown_event.is_set():
                                logger.warning(
                                    "Shutdown detected during worker execution for iter %d. Terminating pool.",
                                    t,
                                )
                                raise GracefulShutdownException(
                                    "Shutdown during worker execution"
                                )

                            # Check which results are ready without blocking indefinitely
                            ready_workers = [
                                worker_id
                                for worker_id, res in async_results.items()
                                if res and res.ready()
                            ]

                            for worker_id in ready_workers:
                                if worker_id not in async_results:
                                    continue  # Already processed
                                async_res = async_results.pop(
                                    worker_id, None
                                )  # Remove from dict to avoid re-processing
                                if async_res is None:
                                    continue  # Already processed

                                try:
                                    # Get the actual result (WorkerResult object or None)
                                    result: Optional[WorkerResult] = async_res.get()
                                    results[worker_id] = (
                                        result  # Store result at correct index
                                    )
                                    completed_worker_count += 1
                                    if result is None:
                                        sim_failed_count += 1
                                        self._worker_statuses[worker_id] = (
                                            f"Failed (Iter {t})"
                                        )
                                    elif isinstance(result, WorkerResult):
                                        # Store WorkerStats object for display
                                        self._worker_statuses[worker_id] = result.stats
                                except Exception as pool_e:
                                    # Log more specific error here before the generic message
                                    logger.error(
                                        "!!! Exception type '%s' encountered fetching result for worker %d iter %d.",
                                        type(pool_e).__name__,
                                        worker_id,
                                        t,
                                    )
                                    # Log the generic message which includes the exception string
                                    logger.error(
                                        "Error fetching result for worker %d iter %d: %s",
                                        worker_id,
                                        t,
                                        pool_e,
                                        exc_info=False,  # Avoid double traceback if logger propagates
                                    )
                                    results[worker_id] = None  # Mark as failed worker
                                    completed_worker_count += 1
                                    sim_failed_count += 1
                                    self._worker_statuses[worker_id] = (
                                        f"Error ({type(pool_e).__name__})"
                                    )

                            # Update display immediately after checking ready workers
                            self._update_status_display(progress_bar, t, end_iter_num)

                            if completed_worker_count < num_workers:
                                time.sleep(0.2)  # Short sleep to avoid busy-waiting

                    except (
                        GracefulShutdownException
                    ):  # Re-raise to be caught by outer handler
                        raise
                    except Exception:
                        logger.exception(
                            "Error during parallel simulation pool iter %d.", t
                        )
                        sim_failed_count = (
                            num_workers  # Assume all failed if pool error occurs
                        )
                    finally:
                        if pool:
                            # Terminate remaining workers forcefully if shutdown was requested or error occurred
                            if self.shutdown_event.is_set() or sim_failed_count > 0:
                                logger.warning("Terminating worker pool...")
                                pool.terminate()
                            try:
                                # Join without timeout
                                pool.join()
                            except Exception as join_e:
                                logger.error("Error joining worker pool: %s", join_e)
                            pool = None  # Ensure pool is cleaned up reference

                # --- Process Results (Merge) ---
                if sim_failed_count == num_workers and num_workers > 0:
                    # Ensure postfix is a dict before setting str
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
                    continue  # Skip merge, save, exploitability

                if sim_failed_count > 0:
                    logger.warning(
                        "%d/%d worker sims failed for iter %d.",
                        sim_failed_count,
                        num_workers,
                        t,
                    )
                    # Filter out Nones before merging
                    valid_results = [res for res in results if res is not None]
                    if not valid_results:
                        # Ensure postfix is a dict before setting str
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
                            "All simulations failed for iteration %d. Skipping merge.",
                            t,
                        )
                        continue
                else:
                    # All results should be valid WorkerResult objects
                    valid_results = results  # Keep existing type hint logic

                # Ensure postfix is a dict before setting str
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
                    # Pass only the list of valid WorkerResult objects
                    self._merge_local_updates(valid_results)  # type: ignore
                    merge_time = time.time() - merge_start_time
                    logger.debug("Iter %d merge took %.3fs", t, merge_time)
                except Exception:
                    logger.exception("Error merging results iter %d.", t)
                    # Ensure postfix is a dict before setting str
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

                # --- Iteration Completed Successfully ---
                iter_time = time.time() - iter_start_time
                self._total_infosets_str = f"{len(self.regret_sum):,}"

                # Calculate Exploitability Periodically
                exploit_calc_time = 0.0
                if exploitability_interval > 0 and t % exploitability_interval == 0:
                    # Ensure postfix is a dict before setting str
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
                        # Ensure postfix is a dict before setting str
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

                # Update progress bar postfix
                postfix_dict = {
                    "LastT": f"{iter_time:.2f}s",
                    "Expl": self._last_exploit_str,
                    "Nodes": self._total_infosets_str,
                }
                progress_bar.set_postfix(postfix_dict, refresh=True)
                # Update worker status display to show final state for the iteration
                self._update_status_display(progress_bar, t, end_iter_num)

                # Save progress periodically
                if t % self.config.cfr_training.save_interval == 0:
                    # Ensure postfix is a dict before setting str
                    progress_bar.set_postfix(postfix_dict, refresh=False)
                    progress_bar.set_postfix_str("Saving...", refresh=True)
                    self.save_data()  # Saves state after completing iteration 't'
                    progress_bar.set_postfix(
                        postfix_dict, refresh=True
                    )  # Restore normal postfix

        except GracefulShutdownException:
            logger.warning(
                "Graceful shutdown exception caught in train loop. Saving progress..."
            )
            # Ensure pool is terminated if it was active
            if pool:
                logger.warning("Terminating worker pool due to shutdown...")
                pool.terminate()
                try:
                    # Join without timeout
                    pool.join()
                except Exception as join_e:
                    logger.error("Error joining worker pool during shutdown: %s", join_e)
                pool = None
            # Save state as of the *last successfully completed* iteration before the exception
            completed_iter_to_save = (
                self.current_iteration
                - 1  # Iteration t was interrupted or not fully processed
                if hasattr(self, "current_iteration")
                and self.current_iteration > last_completed_iteration
                else last_completed_iteration  # Save last known good iteration if crash was early
            )
            if (
                completed_iter_to_save >= 0
            ):  # Only save if at least one new iter completed or loaded data exists
                temp_iter = self.current_iteration
                self.current_iteration = completed_iter_to_save
                try:
                    self.save_data()
                    logger.info(
                        "Progress saved successfully (state as of iteration %d completion).",
                        self.current_iteration,
                    )
                except Exception as save_e:
                    logger.error("Failed to save progress during shutdown: %s", save_e)
                # Restore potentially inaccurate current_iteration for debugging info? Or leave as saved iter?
                # self.current_iteration = temp_iter
            else:
                logger.warning(
                    "Shutdown before first new iteration completed or loaded. No progress to save."
                )
            # Re-raise as KeyboardInterrupt for main's finally block
            raise KeyboardInterrupt(
                "Graceful shutdown processed"
            ) from GracefulShutdownException()

        except (
            KeyboardInterrupt
        ) as exc:  # Catch direct KeyboardInterrupt if not through GracefulShutdownException
            logger.warning(
                "KeyboardInterrupt caught directly in train loop. Saving progress..."
            )
            # Logic similar to GracefulShutdownException handling
            if pool:
                logger.warning("Terminating worker pool due to KeyboardInterrupt...")
                pool.terminate()
                try:
                    pool.join()
                except Exception as join_e:
                    logger.error("Error joining pool: %s", join_e)
                pool = None
            completed_iter_to_save = (
                self.current_iteration - 1
                if self.current_iteration > last_completed_iteration
                else last_completed_iteration
            )
            if completed_iter_to_save >= 0:
                temp_iter = self.current_iteration
                self.current_iteration = completed_iter_to_save
                try:
                    self.save_data()
                    logger.info(
                        "Progress saved after interrupt (iter %d).",
                        self.current_iteration,
                    )
                except Exception as save_e:
                    logger.error("Failed save after interrupt: %s", save_e)
            raise KeyboardInterrupt("KeyboardInterrupt processed") from exc

        except Exception as main_loop_e:
            logger.exception("Unhandled exception in main training loop:")
            # Ensure pool is terminated if it was active
            if pool:
                logger.warning("Terminating worker pool due to error...")
                pool.terminate()
                try:
                    # Join without timeout
                    pool.join()
                except Exception as join_e:
                    logger.error(
                        "Error joining worker pool during error handling: %s", join_e
                    )
                pool = None
            logger.warning("Attempting emergency save...")
            completed_iter_to_save = (
                self.current_iteration - 1
                if hasattr(self, "current_iteration")
                and self.current_iteration > last_completed_iteration
                else last_completed_iteration
            )
            if (
                completed_iter_to_save >= 0
            ):  # Allow saving even if iter 0 failed but data existed
                temp_iter = self.current_iteration
                self.current_iteration = completed_iter_to_save
                try:
                    self.save_data()
                    logger.info(
                        "Emergency save completed for iteration %d.",
                        self.current_iteration,
                    )
                except Exception as save_e:
                    logger.error("Emergency save failed: %s", save_e)
                # self.current_iteration = temp_iter # Restore
            raise main_loop_e  # Re-raise the original error

        finally:
            # Clean up the progress bar display area
            progress_bar.close()
            # Clear worker status lines by writing empty lines or a final summary
            if self.config.cfr_training.num_workers > 0:
                # Attempt to clear previous status lines, might not work perfectly everywhere
                # Write enough newlines to cover the potential status block height
                num_status_lines = self.config.cfr_training.num_workers + 2
                # Only write clearing characters if likely in a terminal
                if sys.stderr.isatty():
                    tqdm.write("\n" * num_status_lines, file=sys.stderr, end="")

        # --- Training Loop Finished Normally ---
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
            # Check if last iteration's exploitability needs adding/updating
            if (
                not self.exploitability_results
                or self.exploitability_results[-1][0] != self.current_iteration
            ):
                self.exploitability_results.append(
                    (self.current_iteration, final_exploit)
                )
            else:  # Update if last iter exploitability was calculated mid-loop
                self.exploitability_results[-1] = (self.current_iteration, final_exploit)
            logger.info("Final exploitability: %.4f", final_exploit)
            self._last_exploit_str = (
                f"{final_exploit:.4f}" if final_exploit != float("inf") else "N/A"
            )
        else:
            logger.warning("Could not compute final average strategy.")
            self._last_exploit_str = "N/A (Avg Err)"

        final_status_msg = f"\nFinished. Iter: {self.current_iteration} | Nodes: {self._total_infosets_str} | Expl: {self._last_exploit_str}"
        tqdm.write(final_status_msg, file=sys.stderr)

        # Final save
        self.save_data()
        logger.info("Final average strategy and data saved.")

    def _update_status_display(
        self, progress_bar: tqdm, current_iter: int, total_iter: int
    ):
        """Updates the main progress bar postfix and prints worker statuses above it."""
        # --- Update postfix for the main bar ---
        last_t_value = "N/A"
        # Check the type of postfix before trying to access it
        if isinstance(progress_bar.postfix, dict):
            last_t_value = progress_bar.postfix.get("LastT", "N/A")
        elif isinstance(progress_bar.postfix, str):
            # If postfix is a string (from set_postfix_str), we can't get LastT.
            # Keep the default "N/A" or decide if the string itself should be preserved somehow.
            # For simplicity, let's reset to "N/A" if it was a string.
            last_t_value = "N/A"

        # Construct the postfix dictionary we *want* to set
        current_postfix_dict = {
            "LastT": last_t_value,
            "Expl": self._last_exploit_str,
            "Nodes": self._total_infosets_str,
        }
        # Set the postfix to the dictionary format, ensuring it's not a string
        progress_bar.set_postfix(
            current_postfix_dict, refresh=False
        )  # Don't refresh main bar yet

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
                if isinstance(status_info, WorkerStats):
                    # Format WorkerStats object
                    status_str = f"Done (Depth:{status_info.max_depth}, Nodes:{status_info.nodes_visited:,})"
                elif isinstance(status_info, str):
                    # Use the string status directly (e.g., "Running", "Failed")
                    status_str = status_info
                else:
                    status_str = "Unknown State"

                line = f" W{i:<2}: {status_str}"
                status_lines.append(line)
                max_len = max(max_len, len(line))
            status_lines.append("-" * max_len)  # Footer line matching max width

            # Use tqdm.write to print the status block.
            # Clear previous lines using ANSI codes if in a suitable terminal.
            clear_lines = num_workers + 2
            if sys.stderr.isatty():
                # Move cursor up, clear lines from cursor down, move cursor up again
                # This seems slightly more robust on some terminals than clearing line by line up.
                tqdm.write(
                    f"\033[{clear_lines}F\033[J\033[{clear_lines}F",
                    file=sys.stderr,
                    end="",
                )

            status_block = "\n".join(status_lines)
            tqdm.write(status_block, file=sys.stderr, end="\n")

        # Refresh the main progress bar now *after* writing status lines
        # and ensuring postfix is a dictionary
        progress_bar.refresh()
