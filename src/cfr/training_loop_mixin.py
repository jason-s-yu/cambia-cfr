# src/cfr/training_loop_mixin.py
"""Mixin class for orchestrating the CFR+ training loop."""

import logging
import sys
import time
import copy  # For deep copying regret snapshot
import multiprocessing
from typing import Any, Callable, Optional, List, Tuple

from tqdm import tqdm

# Ensure relative imports work correctly
from ..utils import WorkerResult, PolicyDict  # Import WorkerResult and PolicyDict
from ..config import Config  # Import Config for type`` hinting
from .exceptions import GracefulShutdownException
from .worker import run_cfr_simulation_worker  # Import the worker function


logger = logging.getLogger(__name__)


class CFRTrainingLoopMixin:
    """Handles the overall CFR+ training loop, progress, and scheduling, supporting parallelism."""

    # Attributes expected to be initialized in the main class's __init__
    config: Config
    analysis: "AnalysisTools"  # Forward reference if AnalysisTools is imported elsewhere
    shutdown_event: "threading.Event"
    current_iteration: int
    exploitability_results: List[Tuple[int, float]]
    regret_sum: PolicyDict
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
        last_completed_iteration = self.current_iteration  # Loaded by load_data
        start_iter_num = last_completed_iteration + 1
        end_iter_num = last_completed_iteration + total_iterations_to_run

        exploitability_interval = self.config.cfr_training.exploitability_interval

        # Determine number of workers based on parsed config value
        num_workers = self.config.parsed_num_workers
        logger.info("Using %d worker process(es) for training.", num_workers)

        if total_iterations_to_run <= 0:
            logger.warning("Number of iterations to run must be positive.")
            return
        logger.info(
            "Starting CFR+ training loop from iteration %d up to %d...",
            start_iter_num,
            end_iter_num,
        )
        loop_start_time = time.time()

        # Status bar (top) - Less detailed now, only total nodes/exploit
        status_bar = tqdm(
            total=0, position=0, bar_format="{desc}", desc="Initializing status..."
        )
        # Main progress bar (bottom)
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
                if self.shutdown_event.is_set():
                    logger.warning(
                        "Shutdown detected before starting iteration %d. Stopping.", t
                    )
                    raise GracefulShutdownException("Shutdown detected before iteration")

                iter_start_time = time.time()
                self.current_iteration = t  # Mark start of processing iteration t

                # Update status bar (simplified)
                status_bar.set_description_str(
                    f"Iter {self.current_iteration}/{end_iter_num} | Nodes:{self._total_infosets_str} | Expl:{self._last_exploit_str}"
                )

                # --- Prepare for Simulation(s) ---
                # Create a snapshot of regrets for workers to use for strategy calculation
                # Deep copy to avoid race conditions if shared memory isn't used
                try:
                    regret_snapshot = copy.deepcopy(self.regret_sum)
                except Exception as e:
                    logger.error("Failed to deepcopy regret_sum: %s", e, exc_info=True)
                    # Decide how to handle: maybe try shallow copy or fail?
                    # For now, let's try to proceed cautiously with a reference (risky)
                    logger.warning(
                        "Proceeding with shallow copy of regret_sum due to deepcopy error."
                    )
                    regret_snapshot = self.regret_sum  # Risky fallback

                worker_args = (t, self.config, regret_snapshot)
                results: List[Optional[WorkerResult]] = []
                sim_failed = False

                # --- Run Simulation(s) ---
                if num_workers == 1:
                    # Sequential execution
                    try:
                        # status_bar update removed from worker, update here?
                        progress_bar.set_postfix_str("Running sim...", refresh=False)
                        result = run_cfr_simulation_worker(worker_args)
                        results = [result]
                        if result is None:
                            sim_failed = True
                            logger.error(
                                "Sequential simulation failed for iteration %d.", t
                            )
                    except Exception:
                        logger.exception(
                            "Error during sequential simulation on iteration %d.", t
                        )
                        sim_failed = True
                else:
                    # Parallel execution
                    worker_args_list = [
                        worker_args
                    ] * num_workers  # Pass same args to all workers
                    try:
                        progress_bar.set_postfix_str(
                            f"Running {num_workers} sims...", refresh=False
                        )
                        # Use context manager for the pool
                        with multiprocessing.Pool(processes=num_workers) as pool:
                            results = pool.map(
                                run_cfr_simulation_worker, worker_args_list
                            )
                        # Check for worker failures indicated by None results
                        if None in results:
                            failed_count = results.count(None)
                            logger.warning(
                                "%d out of %d worker simulations failed for iteration %d.",
                                failed_count,
                                num_workers,
                                t,
                            )
                            # Continue with successful results, but log the issue.
                            # Filter out None values before merging
                            results = [res for res in results if res is not None]
                            if not results:  # All workers failed
                                sim_failed = True
                    except Exception as e:
                        logger.exception(
                            "Error during parallel simulation pool execution on iteration %d.",
                            t,
                        )
                        sim_failed = True

                if sim_failed:
                    progress_bar.set_postfix_str("Sim FAILED", refresh=True)
                    # Optionally implement retry logic or halt training
                    continue  # Skip merge, save, exploitability for this failed iter

                # --- Merge Results ---
                progress_bar.set_postfix_str("Merging...", refresh=False)
                merge_start_time = time.time()
                try:
                    # Ensure _merge_local_updates exists and call it
                    if hasattr(self, "_merge_local_updates") and callable(
                        self._merge_local_updates
                    ):
                        self._merge_local_updates(results)
                    else:
                        logger.error(
                            "Merge function _merge_local_updates not found in DataManagerMixin!"
                        )
                        # Halt training if merge isn't possible
                        raise RuntimeError("Merge function missing")
                    merge_time = time.time() - merge_start_time
                    logger.debug("Iter %d merge took %.3fs", t, merge_time)
                except Exception as e:
                    logger.exception("Error during result merging on iteration %d.", t)
                    progress_bar.set_postfix_str("Merge FAILED", refresh=True)
                    # If merge fails, state might be corrupt. Decide whether to halt.
                    continue  # Skip exploit/save

                # --- Iteration Completed Successfully ---
                iter_time = time.time() - iter_start_time
                self._total_infosets_str = (
                    f"{len(self.regret_sum):,}"  # Update after merge
                )

                # Calculate Exploitability Periodically
                exploit_calc_time = 0.0
                if (
                    exploitability_interval > 0
                    and self.current_iteration % exploitability_interval == 0
                ):
                    exploit_start_time = time.time()
                    logger.info(
                        "Calculating exploitability at iteration %d...",
                        self.current_iteration,
                    )
                    current_avg_strategy = self.compute_average_strategy()
                    if current_avg_strategy:
                        # Ensure analysis object has calculate_exploitability
                        if hasattr(
                            self.analysis, "calculate_exploitability"
                        ) and callable(self.analysis.calculate_exploitability):
                            exploit = self.analysis.calculate_exploitability(
                                current_avg_strategy, self.config
                            )
                            self.exploitability_results.append(
                                (self.current_iteration, exploit)
                            )
                            self._last_exploit_str = (
                                f"{exploit:.3f}" if exploit != float("inf") else "N/A"
                            )
                            exploit_calc_time = time.time() - exploit_start_time
                            logger.info(
                                "Exploitability calculated: %.4f (took %.2fs)",
                                exploit,
                                exploit_calc_time,
                            )
                        else:
                            logger.error(
                                "Exploitability calculation method not found on analysis object!"
                            )
                            self._last_exploit_str = "N/A (Error)"
                    else:
                        logger.warning(
                            "Could not compute average strategy for exploitability calculation."
                        )
                        self._last_exploit_str = "N/A (Avg Strat Error)"

                # Update progress bar postfix
                postfix_dict = {
                    "LastT": f"{iter_time:.2f}s",
                    "Expl": self._last_exploit_str,
                    "Nodes": self._total_infosets_str,
                    "MergeT": f"{merge_time:.2f}s" if merge_time else "N/A",
                    "ExplT": f"{exploit_calc_time:.1f}s" if exploit_calc_time else "N/A",
                }
                progress_bar.set_postfix(postfix_dict, refresh=True)

                # Update status bar description (simplified)
                status_bar.set_description_str(
                    f"Iter {self.current_iteration}/{end_iter_num} done | Nodes:{self._total_infosets_str} | Expl:{self._last_exploit_str}"
                )

                # Save progress periodically
                if self.current_iteration % self.config.cfr_training.save_interval == 0:
                    # Ensure save_data exists and call it
                    if hasattr(self, "save_data") and callable(self.save_data):
                        self.save_data()
                    else:
                        logger.error("Save function save_data not found!")
                        # Halt? For now, just log.

        except GracefulShutdownException:
            logger.warning(
                "Graceful shutdown exception caught in train loop. Saving progress..."
            )
            completed_iter_to_save = self.current_iteration - 1
            if completed_iter_to_save >= 0:
                temp_iter = self.current_iteration
                self.current_iteration = (
                    completed_iter_to_save  # Set to last completed for saving
                )
                try:
                    # Ensure save_data exists and call it
                    if hasattr(self, "save_data") and callable(self.save_data):
                        self.save_data()
                        logger.info(
                            "Progress saved successfully (state as of iteration %d completion).",
                            self.current_iteration,
                        )
                    else:
                        logger.error("Save function save_data not found during shutdown!")

                except Exception as save_e:
                    logger.error("Failed to save progress during shutdown: %s", save_e)
                self.current_iteration = temp_iter  # Restore current iteration number
            else:
                logger.warning(
                    "Shutdown occurred before first iteration completed. No progress to save."
                )
            raise KeyboardInterrupt("Graceful shutdown initiated")  # Re-raise for main

        except Exception as main_loop_e:
            logger.exception("Unhandled exception in main training loop:")
            # Attempt to save progress before exiting
            logger.warning("Attempting emergency save...")
            completed_iter_to_save = self.current_iteration - 1
            if completed_iter_to_save >= 0:
                temp_iter = self.current_iteration
                self.current_iteration = completed_iter_to_save
                try:
                    if hasattr(self, "save_data") and callable(self.save_data):
                        self.save_data()
                        logger.info(
                            "Emergency save completed for iteration %d.",
                            self.current_iteration,
                        )
                    else:
                        logger.error(
                            "Save function save_data not found during emergency save!"
                        )
                except Exception as save_e:
                    logger.error("Emergency save failed: %s", save_e)
                self.current_iteration = temp_iter  # Restore for potential debugging
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
            if hasattr(self.analysis, "calculate_exploitability") and callable(
                self.analysis.calculate_exploitability
            ):
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
                else:  # Update if last iteration was already calculated
                    self.exploitability_results[-1] = (
                        self.current_iteration,
                        final_exploit,
                    )
                logger.info("Final exploitability: %.4f", final_exploit)
                self._last_exploit_str = (
                    f"{final_exploit:.3f}" if final_exploit != float("inf") else "N/A"
                )
            else:
                logger.error(
                    "Exploitability calculation method not found on analysis object!"
                )
                self._last_exploit_str = "N/A (Error)"
        else:
            logger.warning("Could not compute final average strategy.")
            self._last_exploit_str = "N/A (Avg Strat Error)"

        # Final status update to console
        tqdm.write(
            f"Final State (Iter {self.current_iteration}) | Nodes:{self._total_infosets_str} | Expl:{self._last_exploit_str}",
            file=sys.stderr,
        )

        # Final save
        if hasattr(self, "save_data") and callable(self.save_data):
            self.save_data()
            logger.info("Final average strategy and data saved.")
        else:
            logger.error("Save function save_data not found at end of training!")
