# src/cfr/training_loop_mixin.py
"""Mixin class for orchestrating the CFR+ training loop."""

import logging
import sys
import time
import traceback
from typing import Optional

from tqdm import tqdm
import numpy as np

from ..agent_state import AgentState
from ..game.engine import CambiaGameState

from .exceptions import GracefulShutdownException

logger = logging.getLogger(__name__)


class CFRTrainingLoopMixin:
    """Handles the overall CFR+ training loop, progress, and scheduling."""

    # Attributes expected to be initialized in the main class's __init__
    # self.config: Config
    # self.analysis: AnalysisTools
    # self.shutdown_event: threading.Event
    # self.current_iteration: int
    # self.exploitability_results: List[Tuple[int, float]]
    # self._last_exploit_str: str
    # self._total_infosets_str: str
    # self.max_depth_this_iter: int
    # self.num_players: int

    # Methods expected to be provided by other mixins
    # self.load_data()
    # self.save_data()
    # self._cfr_recursive()
    # self.compute_average_strategy()

    def train(self, num_iterations: Optional[int] = None):
        """Runs the main CFR+ training loop."""
        total_iterations_to_run = (
            num_iterations or self.config.cfr_training.num_iterations
        )
        last_completed_iteration = self.current_iteration  # Loaded by load_data
        start_iter_num = last_completed_iteration + 1
        end_iter_num = last_completed_iteration + total_iterations_to_run

        exploitability_interval = self.config.cfr_training.exploitability_interval
        if total_iterations_to_run <= 0:
            logger.warning("Number of iterations to run must be positive.")
            return
        logger.info(
            "Starting CFR+ training loop from iteration %d up to %d...",
            start_iter_num,
            end_iter_num,
        )
        loop_start_time = time.time()

        # Status bar (top)
        status_bar = tqdm(
            total=0, position=0, bar_format="{desc}", desc="Initializing status..."
        )
        # Main progress bar (bottom)
        progress_bar = tqdm(
            range(start_iter_num, end_iter_num + 1),
            desc="CFR+ Training",
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
                self.max_depth_this_iter = 0

                # Update status bar
                status_bar.set_description_str(
                    f"Iter {self.current_iteration}/{end_iter_num} | Depth:0 | MaxD:0 | Nodes:{self._total_infosets_str} | Expl:{self._last_exploit_str}"
                )

                # Initialize game state and agent states for this iteration
                game_state = None
                initial_agent_states = []
                action_log_for_game = []
                game_failed = False

                try:
                    game_state = CambiaGameState(house_rules=self.config.cambia_rules)
                    initial_hands_for_log = [list(p.hand) for p in game_state.players]
                    initial_peeks_for_log = [
                        p.initial_peek_indices for p in game_state.players
                    ]

                    if not game_state.is_terminal():
                        # Create initial observation (needs access to _create_observation)
                        initial_obs = self._create_observation(
                            None, None, game_state, -1, [], None
                        )
                        for i in range(self.num_players):
                            agent = AgentState(
                                player_id=i,
                                opponent_id=game_state.get_opponent_index(i),
                                memory_level=self.config.agent_params.memory_level,
                                time_decay_turns=self.config.agent_params.time_decay_turns,
                                initial_hand_size=len(initial_hands_for_log[i]),
                                config=self.config,
                            )
                            agent.initialize(
                                initial_obs,
                                initial_hands_for_log[i],
                                initial_peeks_for_log[i],
                            )
                            initial_agent_states.append(agent)
                    else:
                        logger.error(
                            "Game is terminal immediately after initialization on iteration %d. Skipping.",
                            self.current_iteration,
                        )
                        progress_bar.set_postfix_str("Error game terminal init, skipping")
                        game_failed = True

                    if not initial_agent_states and not game_failed:
                        logger.error(
                            "Failed to initialize agent states on iteration %d. Skipping.",
                            self.current_iteration,
                        )
                        progress_bar.set_postfix_str("Error init agents, skipping")
                        game_failed = True

                except Exception as e:
                    logger.exception(
                        "Error initializing game/agents state on iteration %d: %s",
                        self.current_iteration,
                        e,
                    )
                    progress_bar.set_postfix_str("Error init game/agents, skipping")
                    game_failed = True

                if game_failed:
                    continue  # Skip this iteration

                # Run CFR recursion
                final_utilities = None
                try:
                    # Pass the shutdown event down
                    final_utilities = self._cfr_recursive(
                        game_state,
                        initial_agent_states,
                        np.ones(
                            self.num_players, dtype=np.float64
                        ),  # Initial reach probs
                        self.current_iteration,
                        action_log_for_game,
                        status_bar,
                        depth=0,
                        shutdown_event=self.shutdown_event,
                    )
                    # Log history only if game completed without shutdown
                    game_details = self.analysis.format_game_details_for_log(
                        game_state=game_state,
                        iteration=self.current_iteration,
                        initial_hands=initial_hands_for_log,
                        action_sequence=action_log_for_game,
                    )
                    self.analysis.log_game_history(game_details)

                except GracefulShutdownException as shutdown_exc:
                    logger.warning(
                        "Graceful shutdown triggered during iteration %d.",
                        self.current_iteration,
                    )
                    raise shutdown_exc  # Re-raise to be caught by outer handler

                except RecursionError:
                    logger.error(
                        "Iter %d, MaxDepth %d: Recursion depth exceeded! Saving progress and stopping.",
                        self.current_iteration,
                        self.max_depth_this_iter,
                    )
                    logger.error("State at RecursionError (approx): %s", game_state)
                    if action_log_for_game:
                        logger.error(
                            "Last actions:\n%s",
                            "\n".join([f"  {e}" for e in action_log_for_game[-10:]]),
                        )
                    logger.error("Traceback:\n%s", traceback.format_exc())
                    progress_bar.set_postfix_str("RecursionError!")
                    game_failed = True
                    temp_iter = self.current_iteration
                    self.current_iteration = (
                        temp_iter - 1
                    )  # Save state from previous iteration
                    self.save_data()
                    self.current_iteration = temp_iter
                    raise  # Re-raise to stop training

                except Exception as e:
                    logger.exception(
                        "Error during CFR recursion on iteration %d: %s",
                        self.current_iteration,
                        e,
                    )
                    logger.error("State at Error: %s", game_state)
                    progress_bar.set_postfix_str(f"CFRError! {type(e).__name__}")
                    game_failed = True
                    # Optionally save progress from previous iteration here too

                if game_failed:
                    continue  # Skip saving/exploitability

                # --- Iteration Completed Successfully ---
                iter_time = time.time() - iter_start_time

                # Calculate Exploitability Periodically
                if (
                    exploitability_interval > 0
                    and self.current_iteration % exploitability_interval == 0
                ):
                    exploit_start_time = time.time()
                    logger.info(
                        "Calculating exploitability at iteration %d...",
                        self.current_iteration,
                    )
                    current_avg_strategy = (
                        self.compute_average_strategy()
                    )  # Provided by DataManagerMixin
                    if current_avg_strategy:
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
                        logger.warning(
                            "Could not compute average strategy for exploitability calculation."
                        )
                        self._last_exploit_str = "N/A"

                self._total_infosets_str = (
                    f"{len(self.regret_sum):,}"  # Access regret_sum
                )

                # Update progress bar postfix
                postfix_dict = {
                    "LastT": f"{iter_time:.2f}s",
                    "DepthMax": f"{self.max_depth_this_iter}",
                    "Expl": self._last_exploit_str,
                    "TotalNodes": self._total_infosets_str,
                }
                progress_bar.set_postfix(postfix_dict, refresh=True)

                # Update status bar description
                status_bar.set_description_str(
                    f"Iter {self.current_iteration}/{end_iter_num} complete | Depth:N/A | MaxD:{self.max_depth_this_iter} | Nodes:{self._total_infosets_str} | Expl:{self._last_exploit_str}"
                )

                # Save progress periodically (save_data provided by DataManagerMixin)
                if self.current_iteration % self.config.cfr_training.save_interval == 0:
                    self.save_data()

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
                    self.save_data()
                    logger.info(
                        "Progress saved successfully (state as of iteration %d completion).",
                        self.current_iteration,
                    )
                except Exception as save_e:
                    logger.error("Failed to save progress during shutdown: %s", save_e)
                self.current_iteration = temp_iter  # Restore current iteration number
            else:
                logger.warning(
                    "Shutdown occurred before first iteration completed. No progress to save."
                )
            raise KeyboardInterrupt("Graceful shutdown initiated")  # Re-raise for main

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
        final_avg_strategy = (
            self.compute_average_strategy()
        )  # Provided by DataManagerMixin
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
            else:  # Update if last iteration was already calculated
                self.exploitability_results[-1] = (self.current_iteration, final_exploit)
            logger.info("Final exploitability: %.4f", final_exploit)
            self._last_exploit_str = (
                f"{final_exploit:.3f}" if final_exploit != float("inf") else "N/A"
            )
        else:
            logger.warning("Could not compute final average strategy.")
            self._last_exploit_str = "N/A"

        # Final status update to console
        tqdm.write(
            f"Final State (Iter {self.current_iteration}) | MaxDepth:{self.max_depth_this_iter} | Nodes:{self._total_infosets_str} | Expl:{self._last_exploit_str}",
            file=sys.stderr,
        )

        # Final save (save_data provided by DataManagerMixin)
        self.save_data()
        logger.info("Final average strategy and data saved.")
