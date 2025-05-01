# src/cfr/training_loop_mixin.py
"""Mixin class for orchestrating the CFR+ training loop."""

import logging
import sys
import threading
import time
import copy
import multiprocessing
import multiprocessing.pool
from typing import Any, Callable, Optional, List, Tuple

from tqdm import tqdm

from ..analysis_tools import AnalysisTools
from ..utils import WorkerResult, PolicyDict, LogQueue
from ..config import Config
from .exceptions import GracefulShutdownException
from .worker import run_cfr_simulation_worker


logger = logging.getLogger(__name__)


class CFRTrainingLoopMixin:
    """Handles the overall CFR+ training loop, progress, and scheduling, supporting parallelism."""

    # Attributes expected to be initialized in the main class's __init__
    config: Config
    analysis: "AnalysisTools"
    shutdown_event: "threading.Event"
    current_iteration: int
    exploitability_results: List[Tuple[int, float]]
    regret_sum: PolicyDict
    log_queue: Optional[LogQueue]  # ADDED
    # Attributes expected from CFRDataManagerMixin
    load_data: Callable[..., Any]
    save_data: Callable[..., Any]
    _merge_local_updates: Callable[[List[Optional[WorkerResult]]], None]
    compute_average_strategy: Callable[..., Optional[PolicyDict]]
    # Internal state for display
    _last_exploit_str: str = "N/A"
    _total_infosets_str: str = "0"

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

        status_bar = tqdm(
            total=0, position=0, bar_format="{desc}", desc="Initializing status..."
        )
        progress_bar = tqdm(
            range(start_iter_num, end_iter_num + 1),
            desc=f"CFR+ Training ({num_workers}W)",
            total=total_iterations_to_run,
            unit="iter",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            position=1,
            leave=False,
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

                status_bar.set_description_str(
                    f"Iter {self.current_iteration}/{end_iter_num} | Nodes:{self._total_infosets_str} | Expl:{self._last_exploit_str}"
                )

                # --- Prepare for Simulation(s) ---
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

                # Worker arguments now include the log_queue
                worker_base_args = (
                    t,
                    self.config,
                    regret_snapshot,
                    log_queue,
                )  # Pass queue
                results: List[Optional[WorkerResult]] = []
                sim_failed = False

                # --- Run Simulation(s) ---
                if num_workers == 1:
                    progress_bar.set_postfix_str("Running sim...", refresh=False)
                    try:
                        worker_args = worker_base_args + (0,)  # Add worker_id 0
                        result = run_cfr_simulation_worker(worker_args)
                        results = [result]
                        if result is None:
                            sim_failed = True
                    except Exception:
                        logger.exception("Error during sequential simulation iter %d.", t)
                        sim_failed = True
                else:
                    # --- Parallel Execution with Improved Shutdown ---
                    worker_args_list = [
                        worker_base_args + (worker_id,)
                        for worker_id in range(num_workers)
                    ]
                    async_results = []
                    try:
                        progress_bar.set_postfix_str(
                            f"Starting {num_workers} sims...", refresh=False
                        )
                        # Create pool within the try block for proper cleanup
                        pool = multiprocessing.Pool(processes=num_workers)
                        # Use imap_unordered for better responsiveness and shutdown checking
                        result_iterator = pool.imap_unordered(
                            run_cfr_simulation_worker, worker_args_list
                        )
                        pool.close()  # No more tasks will be submitted

                        num_completed = 0
                        while num_completed < num_workers:
                            if self.shutdown_event.is_set():
                                logger.warning(
                                    "Shutdown detected during worker execution for iter %d. Terminating pool.",
                                    t,
                                )
                                raise GracefulShutdownException(
                                    "Shutdown during worker execution"
                                )
                            try:
                                # Get next result with a timeout to check shutdown event
                                result = next(result_iterator)
                                results.append(result)
                                num_completed += 1
                                progress_bar.set_postfix_str(
                                    f"Sims Running ({num_completed}/{num_workers})...",
                                    refresh=True,
                                )
                            except StopIteration:
                                # Should not happen if num_completed < num_workers
                                logger.error("Pool iterator stopped unexpectedly.")
                                break
                            except (
                                multiprocessing.TimeoutError
                            ):  # Import TimeoutError if needed (not standard)
                                # No result within timeout, loop continues to check shutdown_event
                                continue
                            except Exception as pool_e:
                                logger.error(
                                    "Error fetching result from pool iter %d: %s",
                                    t,
                                    pool_e,
                                )
                                # Consider how to handle partial failures - maybe append None?
                                results.append(None)  # Mark as failed worker
                                num_completed += 1  # Count as completed (failed)

                        # Check for worker failures indicated by None results
                        if None in results:
                            failed_count = results.count(None)
                            logger.warning(
                                "%d/%d worker sims failed for iter %d.",
                                failed_count,
                                num_workers,
                                t,
                            )
                            results = [
                                res for res in results if res is not None
                            ]  # Filter out Nones
                            if not results:
                                sim_failed = True  # All failed
                    except (
                        GracefulShutdownException
                    ):  # Re-raise to be caught by outer handler
                        raise
                    except Exception as e:
                        logger.exception(
                            "Error during parallel simulation pool iter %d.", t
                        )
                        sim_failed = True
                    finally:
                        if pool:
                            # Terminate remaining workers forcefully if shutdown was requested
                            if self.shutdown_event.is_set():
                                logger.warning("Terminating worker pool...")
                                pool.terminate()
                            pool.join()  # Wait for pool to finish/terminate
                            pool = None  # Ensure pool is cleaned up

                # --- Process Results (Merge) ---
                if sim_failed:
                    progress_bar.set_postfix_str("Sim FAILED", refresh=True)
                    continue  # Skip merge, save, exploitability

                if (
                    not results and num_workers > 0
                ):  # Handle case where all workers failed
                    progress_bar.set_postfix_str("All Sims FAILED", refresh=True)
                    logger.error(
                        "All simulations failed for iteration %d. Skipping merge.", t
                    )
                    continue

                progress_bar.set_postfix_str("Merging...", refresh=False)
                merge_start_time = time.time()
                try:
                    self._merge_local_updates(results)
                    merge_time = time.time() - merge_start_time
                    logger.debug("Iter %d merge took %.3fs", t, merge_time)
                except Exception as e:
                    logger.exception("Error merging results iter %d.", t)
                    progress_bar.set_postfix_str("Merge FAILED", refresh=True)
                    continue

                # --- Iteration Completed Successfully ---
                iter_time = time.time() - iter_start_time
                self._total_infosets_str = f"{len(self.regret_sum):,}"

                # Calculate Exploitability Periodically
                exploit_calc_time = 0.0
                if exploitability_interval > 0 and t % exploitability_interval == 0:
                    exploit_start_time = time.time()
                    logger.info("Calculating exploitability at iteration %d...", t)
                    current_avg_strategy = self.compute_average_strategy()
                    if current_avg_strategy:
                        exploit = self.analysis.calculate_exploitability(
                            current_avg_strategy, self.config
                        )
                        self.exploitability_results.append((t, exploit))
                        self._last_exploit_str = (
                            f"{exploit:.3f}" if exploit != float("inf") else "N/A"
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
                status_bar.set_description_str(
                    f"Iter {t}/{end_iter_num} done | Nodes:{self._total_infosets_str} | Expl:{self._last_exploit_str}"
                )

                # Save progress periodically
                if t % self.config.cfr_training.save_interval == 0:
                    self.save_data()  # Saves state after completing iteration 't'

        except GracefulShutdownException:
            logger.warning(
                "Graceful shutdown exception caught in train loop. Saving progress..."
            )
            # Ensure pool is terminated if it was active
            if pool:
                logger.warning("Terminating worker pool due to shutdown...")
                pool.terminate()
                pool.join()
                pool = None
            # Save state as of the *last successfully completed* iteration before the exception
            completed_iter_to_save = (
                self.current_iteration - 1
            )  # Iteration t was interrupted or not fully processed
            if (
                completed_iter_to_save > last_completed_iteration
            ):  # Only save if at least one new iter completed
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
                self.current_iteration = temp_iter  # Restore for potential debugging
            else:
                logger.warning(
                    "Shutdown before first new iteration completed. No progress to save."
                )
            raise KeyboardInterrupt("Graceful shutdown initiated")  # Re-raise for main

        except Exception as main_loop_e:
            logger.exception("Unhandled exception in main training loop:")
            # Ensure pool is terminated if it was active
            if pool:
                logger.warning("Terminating worker pool due to error...")
                pool.terminate()
                pool.join()
                pool = None
            logger.warning("Attempting emergency save...")
            completed_iter_to_save = self.current_iteration - 1
            if completed_iter_to_save > last_completed_iteration:
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
                self.current_iteration = temp_iter  # Restore
            raise main_loop_e  # Re-raise the original error

        # --- Training Loop Finished Normally ---
        status_bar.close()
        progress_bar.close()
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
                f"{final_exploit:.3f}" if final_exploit != float("inf") else "N/A"
            )
        else:
            logger.warning("Could not compute final average strategy.")
            self._last_exploit_str = "N/A (Avg Err)"

        tqdm.write(
            f"Final State (Iter {self.current_iteration}) | Nodes:{self._total_infosets_str} | Expl:{self._last_exploit_str}",
            file=sys.stderr,
        )

        # Final save
        self.save_data()
        logger.info("Final average strategy and data saved.")
