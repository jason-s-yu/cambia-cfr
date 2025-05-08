"""src/analysis_tools.py"""

import logging
import json
import os
import copy
import queue
import time
import multiprocessing
import multiprocessing.pool
import traceback
from typing import Optional, List, Set
from dataclasses import asdict, is_dataclass
import numpy as np

from .game.engine import CambiaGameState
from .card import Card
from .agent_state import AgentState, AgentObservation
from .constants import (
    ActionReplace,
    GameAction,
    DecisionContext,
    NUM_PLAYERS,
    ActionDiscard,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingSwapDecision,
    ActionSnapOpponentMove,
)
from .config import Config
from .utils import InfosetKey, PolicyDict, normalize_probabilities, SimulationTrace
from .game.helpers import serialize_card

# Conditional imports for type hinting
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .live_display import LiveDisplayManager

logger = logging.getLogger(__name__)


# --- Top-level functions for parallel BR calculation ---


def _br_action_worker(
    game_state_copy: CambiaGameState,
    br_agent_state_copy: AgentState,
    opp_view_agent_state_copy: AgentState,
    action: GameAction,
    opponent_avg_strategy: PolicyDict,
    br_player: int,
    depth: int,
) -> float:
    """
    Target function for the BR pool. Applies one action and calls node logic.
    """
    try:
        # Apply the action to the copied state
        state_delta, undo_info = game_state_copy.apply_action(action)
        if not callable(undo_info):
            logger.error(
                "BR ActionWorker(D%d): Action %s returned invalid undo.", depth, action
            )
            return -float("inf")  # Indicate failure

        # Create observation AFTER action (use static helper)
        obs_after_action = AnalysisTools._create_observation_for_br(
            game_state_copy, action, br_agent_state_copy.player_id
        )
        if obs_after_action is None:
            # Undo is not possible here as it's in a different process
            logger.error(
                "BR ActionWorker(D%d): Failed to create observation after action %s.",
                depth,
                action,
            )
            return -float("inf")  # Indicate failure

        # Update agent states using static helpers
        br_obs_filtered = AnalysisTools._filter_observation_for_br(
            obs_after_action, br_player
        )
        opp_view_obs_filtered = AnalysisTools._filter_observation_for_br(
            obs_after_action, br_agent_state_copy.opponent_id
        )
        br_agent_state_copy.update(br_obs_filtered)
        opp_view_agent_state_copy.update(opp_view_obs_filtered)

        # Recursive call to node logic (serially within this worker)
        action_value = AnalysisTools._best_response_node_logic(
            game_state_copy,
            opponent_avg_strategy,
            br_player,
            br_agent_state_copy,
            opp_view_agent_state_copy,
            depth + 1,
            pool=None,  # No pool for recursive calls within worker
        )
        # No undo needed as we work on copies
        return action_value
    except Exception as e:
        logger.error(
            "BR ActionWorker(D%d): Error processing action %s: %s\n%s",
            depth,
            action,
            e,
            traceback.format_exc(),
            exc_info=False,
        )
        return -float("inf")  # Indicate failure


def _best_response_recursive_entry(
    game_state: CambiaGameState,
    opponent_avg_strategy: PolicyDict,
    br_player: int,
    br_agent_state: AgentState,
    opp_view_agent_state: AgentState,
    pool: Optional[multiprocessing.pool.Pool],  # Pool for parallelizing actions
    depth: int = 0,
) -> float:
    """Entry point for the recursive best response calculation."""
    return AnalysisTools._best_response_node_logic(
        game_state,
        opponent_avg_strategy,
        br_player,
        br_agent_state,
        opp_view_agent_state,
        depth,
        pool,
    )


def _run_br_calculation_process(
    avg_strat: PolicyDict,
    config: Config,
    br_player: int,
    result_queue: multiprocessing.Queue,
):
    """
    Function executed by each of the two main BR calculation processes.
    Sets up a local pool and calls the BR entry point.
    """
    try:
        # Basic logging setup for this process (optional, could log to specific file)
        # configure_logging_for_process(f"br_process_p{br_player}")

        logger.info("BR Process P%d: Starting...", br_player)
        exploit_workers = config.analysis.exploitability_num_workers
        logger.info(
            "BR Process P%d: Using %d workers for action evaluation.",
            br_player,
            exploit_workers,
        )

        pool: Optional[multiprocessing.pool.Pool] = None
        if exploit_workers > 1:
            pool = multiprocessing.Pool(processes=exploit_workers)
            logger.debug("BR Process P%d: Worker pool created.", br_player)

        # Initialize game state and agent states *within this process*
        game_state = CambiaGameState(house_rules=config.cambia_rules)
        opponent_player = 1 - br_player

        initial_obs = AnalysisTools._create_observation_for_br(game_state, None, -1)
        if initial_obs is None:
            raise RuntimeError("BR Process P%d: Failed to create initial observation.")

        br_agent_state = AgentState(
            player_id=br_player,
            opponent_id=opponent_player,
            memory_level=config.agent_params.memory_level,
            time_decay_turns=config.agent_params.time_decay_turns,
            initial_hand_size=len(game_state.players[br_player].hand),
            config=config,
        )
        br_agent_state.initialize(
            initial_obs,
            game_state.players[br_player].hand,
            game_state.players[br_player].initial_peek_indices,
        )

        opp_view_agent_state = AgentState(
            player_id=opponent_player,
            opponent_id=br_player,
            memory_level=config.agent_params.memory_level,
            time_decay_turns=config.agent_params.time_decay_turns,
            initial_hand_size=len(game_state.players[opponent_player].hand),
            config=config,
        )
        opp_view_agent_state.initialize(
            initial_obs,
            game_state.players[opponent_player].hand,
            game_state.players[opponent_player].initial_peek_indices,
        )
        logger.debug("BR Process P%d: Game and Agent states initialized.", br_player)

        # Call the entry point for BR calculation
        br_value = _best_response_recursive_entry(
            game_state,
            avg_strat,
            br_player,
            br_agent_state,
            opp_view_agent_state,
            pool,  # Pass the pool
            depth=0,
        )
        logger.info(
            "BR Process P%d: Calculation complete. Value: %.6f", br_player, br_value
        )

        # Put result on the queue
        result_queue.put((br_player, br_value))

    except Exception as e:
        logger.error(
            "BR Process P%d: Unhandled exception: %s\n%s",
            br_player,
            e,
            traceback.format_exc(),
            exc_info=False,
        )
        result_queue.put((br_player, float("inf")))  # Signal failure
    finally:
        # Clean up the local pool
        if pool:
            try:
                logger.debug("BR Process P%d: Closing worker pool...", br_player)
                pool.close()
                pool.join()
                logger.debug("BR Process P%d: Worker pool closed and joined.", br_player)
            except Exception as e_pool_close:
                logger.error(
                    "BR Process P%d: Error closing pool: %s", br_player, e_pool_close
                )


# Helper function for default serialization in JSON dump
def default_serializer(obj):
    """Default JSON serializer for objects not directly serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.void)):
        return None
    if isinstance(obj, InfosetKey):
        return obj.astuple()  # Use the tuple representation
    if isinstance(obj, Card):
        return serialize_card(obj)
    # Handle GameAction NamedTuples and other dataclasses explicitly
    if hasattr(obj, "_asdict") and callable(obj._asdict):  # Check for NamedTuple
        action_dict = obj._asdict()
        serialized_dict = {}
        for k, v in action_dict.items():
            # Recursive call for nested objects/cards
            serialized_dict[k] = default_serializer(v)
        return {type(obj).__name__: serialized_dict}  # Include type name
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)  # Use dataclasses.asdict for regular dataclasses
    # Fallback for other types
    try:
        # If it's a simple object instance, return its class name
        if hasattr(obj, "__class__") and not isinstance(obj, type):
            return obj.__class__.__name__
        return str(obj)  # Try string representation
    except TypeError:
        return repr(obj)  # Final fallback


class AnalysisTools:
    """Provides tools for analyzing CFR training progress and game history."""

    def __init__(
        self,
        config: Config,
        log_dir: Optional[str] = None,
        log_file_prefix: Optional[str] = None,
    ):
        self.config = config
        self.delta_log_file_path = None  # (Deprecated?) Path for detailed delta logs
        self.simulation_trace_log_path = None  # Path for simulation traces

        if log_dir and log_file_prefix:
            try:
                os.makedirs(log_dir, exist_ok=True)
                # Setup path for the delta log file (keep for now if needed elsewhere)
                self.delta_log_file_path = os.path.join(
                    log_dir, f"{log_file_prefix}_game_deltas.jsonl"
                )
                # Setup path for the new simulation trace log file
                sim_trace_prefix = getattr(
                    config.logging,
                    "simulation_trace_filename_prefix",
                    "simulation_traces",
                )
                self.simulation_trace_log_path = os.path.join(
                    log_dir, f"{sim_trace_prefix}_simulation_traces.jsonl"
                )
                logger.info(
                    "AnalysisTools: Simulation trace log path: %s",
                    self.simulation_trace_log_path,
                )
            except OSError as e_mkdir:
                logger.error(
                    "AnalysisTools: Failed to create log directory '%s': %s",
                    log_dir,
                    e_mkdir,
                )
            except AttributeError as e_attr:
                logger.error(
                    "AnalysisTools: Missing config attribute for logging setup: %s",
                    e_attr,
                )
        else:
            logger.warning(
                "AnalysisTools: Log directory/prefix not provided. Trace/Delta logging disabled."
            )

    def calculate_exploitability(
        self,
        average_strategy: PolicyDict,
        config: Config,
        live_display_manager: Optional["LiveDisplayManager"] = None,
    ) -> float:
        """Calculates the exploitability of the agent's average strategy using parallel processes."""
        if not average_strategy:
            logger.warning("Cannot calculate exploitability: Average strategy is empty.")
            return float("inf")

        exploitability = float("inf")  # Default to infinity
        start_time = time.time()

        # Update display status
        if live_display_manager and hasattr(
            live_display_manager, "update_main_process_status"
        ):
            live_display_manager.update_main_process_status(
                "Calculating exploitability..."
            )

        try:
            logger.info("Starting parallel exploitability calculation...")
            # Use a standard multiprocessing Queue
            result_queue = multiprocessing.Queue()
            processes: List[multiprocessing.Process] = []

            # Spawn two processes, one for each BR player
            for br_player in range(NUM_PLAYERS):
                logger.info(
                    "Spawning Best Response calculation process for Player %d...",
                    br_player,
                )
                process = multiprocessing.Process(
                    target=_run_br_calculation_process,
                    args=(average_strategy, config, br_player, result_queue),
                    name=f"BR_Calc_P{br_player}",
                )
                processes.append(process)
                process.start()

            # Wait for both processes to finish and collect results
            results_dict = {}
            processes_completed = 0
            while processes_completed < NUM_PLAYERS:
                try:
                    # Use timeout to allow periodic checks if needed (e.g., for shutdown signals)
                    p_id, value = result_queue.get(timeout=1.0)
                    results_dict[p_id] = value
                    processes_completed += 1
                    logger.info("BR Process P%d finished. Result: %.6f", p_id, value)
                except queue.Empty:
                    # Check if any process terminated unexpectedly
                    for i, p in enumerate(processes):
                        if not p.is_alive() and i not in results_dict:
                            logger.error(
                                "BR Process P%d terminated unexpectedly. Assigning inf.",
                                i,
                            )
                            results_dict[i] = float("inf")
                            processes_completed += 1
                    continue  # Continue waiting if not all processes finished

            # Join processes after collecting results
            for process in processes:
                try:
                    process.join(timeout=5.0)
                    if process.is_alive():
                        logger.warning(
                            "Process %s did not join within timeout. Terminating.",
                            process.name,
                        )
                        process.terminate()
                        process.join(timeout=1.0)  # Wait briefly after terminate
                except Exception as e_join:
                    logger.error("Error joining process %s: %s", process.name, e_join)

            # Calculate final exploitability
            br_value_p0 = results_dict.get(0, float("inf"))
            br_value_p1 = results_dict.get(1, float("inf"))

            if br_value_p0 == float("inf") or br_value_p1 == float("inf"):
                logger.warning(
                    "Exploitability calculation resulted in infinity (BR failed for at least one player)."
                )
                exploitability = float("inf")
            else:
                exploitability = (br_value_p0 + br_value_p1) / 2.0
                logger.info("Calculated Exploitability: %.6f", exploitability)

        except Exception as e_exploit:
            logger.exception(
                "Error during parallel exploitability calculation setup/coordination: %s",
                e_exploit,
            )
            exploitability = float("inf")  # Indicate error
        finally:
            # Ensure status is reset even on error
            if live_display_manager and hasattr(
                live_display_manager, "update_main_process_status"
            ):
                live_display_manager.update_main_process_status("Idle / Waiting...")
            logger.info(
                "Total Exploitability calculation time: %.2f seconds",
                time.time() - start_time,
            )

        return exploitability

    # Deprecated: _compute_best_response_value removed, logic moved to _run_br_calculation_process

    @staticmethod
    def _best_response_node_logic(
        game_state: CambiaGameState,
        opponent_avg_strategy: PolicyDict,
        br_player: int,
        br_agent_state: AgentState,
        opp_view_agent_state: AgentState,
        depth: int,
        pool: Optional[multiprocessing.pool.Pool],  # Pool for parallelizing actions
    ) -> float:
        """Recursive logic for a node in the Best Response calculation."""
        try:
            if game_state.is_terminal():
                return game_state.get_utility(br_player)

            acting_player = game_state.get_acting_player()
            if acting_player == -1:
                logger.error(
                    "BR NodeLogic(D%d): Invalid acting player. State: %s",
                    depth,
                    game_state,
                )
                return 0.0

            opponent_player = 1 - br_player
            legal_actions_set: Set[GameAction] = game_state.get_legal_actions()
            # Sort for deterministic behavior, helpful for debugging
            legal_actions: List[GameAction] = sorted(list(legal_actions_set), key=repr)
            num_actions = len(legal_actions)

            if num_actions == 0:
                if not game_state.is_terminal():
                    logger.error(
                        "BR NodeLogic(D%d): No legal actions but non-terminal! State: %s",
                        depth,
                        game_state,
                    )
                return game_state.get_utility(br_player)  # Return current utility

            current_context = AnalysisTools._get_decision_context(game_state)
            if current_context is None:
                logger.error(
                    "BR NodeLogic(D%d): Could not determine decision context. State: %s",
                    depth,
                    game_state,
                )
                return 0.0  # Error case

            # --- Node Logic ---
            if acting_player == br_player:
                # Maximize value for BR player

                # Parallel execution for BR player's actions
                if pool and num_actions > 1:
                    tasks = []
                    for action in legal_actions:
                        # Create deep copies of game and agent states for the worker
                        # Note: deepcopy can be slow, potential optimization point if needed
                        try:
                            game_state_copy = copy.deepcopy(game_state)
                            br_agent_state_copy = br_agent_state.clone()
                            opp_view_agent_state_copy = opp_view_agent_state.clone()
                        except Exception as e_copy:
                            logger.error(
                                "BR NodeLogic(D%d): Error deep copying states for parallel task: %s",
                                depth,
                                e_copy,
                            )
                            # Fallback to serial execution? Or return error? Fallback for now.
                            pool = None  # Disable pool for this node
                            break  # Break from action loop, will go to serial logic below

                        tasks.append(
                            (
                                game_state_copy,
                                br_agent_state_copy,
                                opp_view_agent_state_copy,
                                action,
                                opponent_avg_strategy,
                                br_player,
                                depth,  # Pass depth for logging within worker
                            )
                        )

                    if pool:  # Check if pool wasn't disabled by copy error
                        try:
                            results = pool.starmap(_br_action_worker, tasks)
                            valid_results = [r for r in results if r != -float("inf")]
                            if not valid_results:
                                logger.warning(
                                    "BR NodeLogic(D%d): BR player P%d (parallel) had no successful action paths. Returning 0.",
                                    depth,
                                    br_player,
                                )
                                return 0.0
                            return max(valid_results)
                        except Exception as e_starmap:
                            logger.error(
                                "BR NodeLogic(D%d): Error in pool.starmap: %s",
                                depth,
                                e_starmap,
                            )
                            # Fallback to serial execution on starmap error
                            pool = None  # Disable pool

                # Serial execution for BR player's actions (if pool=None, 1 action, or error fallback)
                max_value = -float("inf")
                actions_attempted_serially = 0
                for action in legal_actions:
                    state_delta, undo_info = game_state.apply_action(action)
                    if not callable(undo_info):
                        logger.error(
                            "BR NodeLogic(D%d): BR Action %s returned invalid undo. State:%s",
                            depth,
                            action,
                            game_state,
                        )
                        continue  # Skip this action path

                    obs_after_action = AnalysisTools._create_observation_for_br(
                        game_state, action, acting_player
                    )
                    if obs_after_action is None:
                        undo_info()
                        continue

                    next_br_agent_state = br_agent_state.clone()
                    next_opp_view_agent_state = opp_view_agent_state.clone()
                    try:
                        br_obs_filtered = AnalysisTools._filter_observation_for_br(
                            obs_after_action, br_player
                        )
                        next_br_agent_state.update(br_obs_filtered)
                        opp_obs_filtered = AnalysisTools._filter_observation_for_br(
                            obs_after_action, opponent_player
                        )
                        next_opp_view_agent_state.update(opp_obs_filtered)
                    except Exception as e_update:
                        logger.error(
                            "BR NodeLogic(D%d): Error updating agent states after BR action %s: %s",
                            depth,
                            action,
                            e_update,
                            exc_info=False,  # Reduce noise
                        )
                        undo_info()
                        continue  # Skip this action path

                    action_value = AnalysisTools._best_response_node_logic(
                        game_state,
                        opponent_avg_strategy,
                        br_player,
                        next_br_agent_state,
                        next_opp_view_agent_state,
                        depth + 1,
                        pool=None,  # Always serial in recursive calls
                    )
                    undo_info()  # Restore state
                    max_value = max(max_value, action_value)
                    actions_attempted_serially += 1

                # If no actions were successfully explored, return 0?
                if max_value == -float("inf") and actions_attempted_serially == 0:
                    logger.warning(
                        "BR NodeLogic(D%d): BR player P%d (serial) had no successful action paths. Returning 0.",
                        depth,
                        br_player,
                    )
                    return 0.0
                return max_value

            else:  # Opponent's turn (always serial)
                # Use Opponent's view state to get the key
                try:
                    base_infoset_tuple = opp_view_agent_state.get_infoset_key()
                    if not isinstance(base_infoset_tuple, tuple):
                        raise TypeError("Infoset key not tuple")
                    infoset_key = InfosetKey(*base_infoset_tuple, current_context.value)
                except Exception as e_key:
                    logger.error(
                        "BR NodeLogic(D%d): Error getting Opponent P%d infoset key: %s. OppView State: %s",
                        depth,
                        acting_player,
                        e_key,
                        opp_view_agent_state,
                        exc_info=False,  # Reduce noise
                    )
                    return 0.0  # Error case

                opponent_strategy = opponent_avg_strategy.get(infoset_key)
                strategy_was_missing = opponent_strategy is None
                dim_mismatch = False

                if opponent_strategy is None:
                    opponent_strategy = (
                        np.ones(num_actions) / num_actions
                        if num_actions > 0
                        else np.array([])
                    )
                elif len(opponent_strategy) != num_actions:
                    logger.warning(
                        "BR NodeLogic(D%d): Dim mismatch Opp P%d strategy at OppView key %s. Have %d, need %d. Using uniform.",
                        depth,
                        acting_player,
                        infoset_key,
                        len(opponent_strategy),
                        num_actions,
                    )
                    opponent_strategy = (
                        np.ones(num_actions) / num_actions
                        if num_actions > 0
                        else np.array([])
                    )
                    dim_mismatch = True

                expected_value = 0.0
                strategy_sum = (
                    opponent_strategy.sum() if opponent_strategy is not None else 0.0
                )

                if num_actions > 0 and strategy_sum > 1e-9:
                    if not np.isclose(strategy_sum, 1.0):
                        if (
                            not strategy_was_missing and not dim_mismatch
                        ):  # Only warn if strategy was present but unnormalized
                            logger.warning(
                                "BR NodeLogic(D%d): Normalizing opponent strategy key %s (Sum: %f)",
                                depth,
                                infoset_key,
                                strategy_sum,
                            )
                        opponent_strategy = normalize_probabilities(opponent_strategy)
                        if len(opponent_strategy) == 0 or not np.isclose(
                            opponent_strategy.sum(), 1.0
                        ):
                            logger.error(
                                "BR NodeLogic(D%d): Failed to normalize opp strategy for %s. Using uniform.",
                                depth,
                                infoset_key,
                            )
                            opponent_strategy = (
                                np.ones(num_actions) / num_actions
                                if num_actions > 0
                                else np.array([])
                            )

                    for i, action in enumerate(legal_actions):
                        action_prob = opponent_strategy[i]
                        if action_prob < 1e-9:
                            continue

                        state_delta, undo_info = game_state.apply_action(action)
                        if not callable(undo_info):
                            logger.error(
                                "BR NodeLogic(D%d): Opponent Action %s returned invalid undo. State:%s",
                                depth,
                                action,
                                game_state,
                            )
                            continue  # Assume 0 value for this path

                        obs_after_action = AnalysisTools._create_observation_for_br(
                            game_state, action, acting_player
                        )
                        if obs_after_action is None:
                            undo_info()
                            continue

                        next_br_agent_state = br_agent_state.clone()
                        next_opp_view_agent_state = opp_view_agent_state.clone()
                        try:
                            br_obs_filtered = AnalysisTools._filter_observation_for_br(
                                obs_after_action, br_player
                            )
                            next_br_agent_state.update(br_obs_filtered)
                            opp_obs_filtered = AnalysisTools._filter_observation_for_br(
                                obs_after_action, opponent_player
                            )
                            next_opp_view_agent_state.update(opp_obs_filtered)
                        except Exception as e_update:
                            logger.error(
                                "BR NodeLogic(D%d): Error updating agent states after Opp action %s: %s",
                                depth,
                                action,
                                e_update,
                                exc_info=False,  # Reduce noise
                            )
                            undo_info()
                            continue  # Assume 0 value for this path

                        recursive_value = AnalysisTools._best_response_node_logic(
                            game_state,
                            opponent_avg_strategy,
                            br_player,
                            next_br_agent_state,
                            next_opp_view_agent_state,
                            depth + 1,
                            pool=None,  # Always serial in recursive calls
                        )
                        undo_info()
                        expected_value += action_prob * recursive_value
                else:
                    # This path should only be reached if num_actions > 0 but strategy sum is ~0.
                    if num_actions > 0:
                        logger.warning(
                            "BR NodeLogic(D%d): Opponent P%d zero strategy sum at OppView infoset %s.",
                            depth,
                            acting_player,
                            infoset_key,
                        )
                    # If opponent effectively cannot act according to strategy, BR player gets utility of current state?
                    # This interpretation seems reasonable for BR.
                    return game_state.get_utility(br_player)

                return expected_value

        except Exception as e_br_rec:
            logger.exception(
                "BR NodeLogic(D%d): Unhandled error in recursion: %s. State: %s",
                depth,
                e_br_rec,
                game_state,
            )
            return 0.0  # Return neutral value on unhandled error

    @staticmethod
    def _get_decision_context(game_state: CambiaGameState) -> Optional[DecisionContext]:
        """Helper to determine DecisionContext robustly."""
        try:
            if game_state.snap_phase_active:
                return DecisionContext.SNAP_DECISION
            pending = game_state.pending_action
            if pending:
                # Use isinstance for robust type checking
                if isinstance(pending, ActionDiscard):
                    return DecisionContext.POST_DRAW
                if isinstance(
                    pending,
                    (
                        ActionAbilityPeekOwnSelect,
                        ActionAbilityPeekOtherSelect,
                        ActionAbilityBlindSwapSelect,
                        ActionAbilityKingLookSelect,
                        ActionAbilityKingSwapDecision,
                    ),
                ):
                    return DecisionContext.ABILITY_SELECT
                if isinstance(pending, ActionSnapOpponentMove):
                    return DecisionContext.SNAP_MOVE
                logger.warning(
                    "BR Context: Unknown pending action type: %s", type(pending).__name__
                )
                return DecisionContext.START_TURN  # Fallback
            if game_state.is_terminal():
                return DecisionContext.TERMINAL
            return DecisionContext.START_TURN
        except AttributeError as e_attr:
            logger.error(
                "Error determining decision context due to missing attribute: %s", e_attr
            )
            return None
        except Exception as e_ctx:
            logger.error("Error determining decision context: %s", e_ctx, exc_info=True)
            return None

    @staticmethod
    def _create_observation_for_br(
        game_state: CambiaGameState,
        action: Optional[GameAction],
        acting_player: int,
    ) -> Optional[AgentObservation]:
        """Helper to create a basic observation for BR agent updates."""
        try:
            # BR calculation needs accurate drawn_card info for the acting agent's state update,
            # especially for replace actions where the agent needs to know the card.
            # Let's refine this to pass the drawn card if it's relevant.
            drawn_card_for_obs = None
            if isinstance(action, ActionDiscard):
                # The card just discarded *was* the drawn card. It's now the top of discard.
                drawn_card_for_obs = game_state.get_discard_top()
            elif isinstance(action, ActionReplace):
                # The card just placed into the hand *was* the drawn card.
                # Need to pass this for the acting player's AgentState update.
                if acting_player != -1 and 0 <= action.target_hand_index < len(
                    game_state.players[acting_player].hand
                ):
                    drawn_card_for_obs = game_state.players[acting_player].hand[
                        action.target_hand_index
                    ]
                else:  # Log error if index is bad
                    logger.error(
                        "BR Create Obs: ActionReplace index %d invalid for actor %d hand size %d",
                        action.target_hand_index,
                        acting_player,
                        (
                            len(game_state.players[acting_player].hand)
                            if acting_player != -1
                            else -1
                        ),
                    )

            # Peeked cards are generally not needed for public state update in BR,
            # as BR agent has perfect info implicitly. Set to None.
            peeked_cards_for_obs = None

            obs = AgentObservation(
                acting_player=acting_player,
                action=action,
                discard_top_card=game_state.get_discard_top(),
                player_hand_sizes=[
                    game_state.get_player_card_count(i) for i in range(NUM_PLAYERS)
                ],
                stockpile_size=game_state.get_stockpile_size(),
                drawn_card=drawn_card_for_obs,  # Pass the potentially determined drawn card
                peeked_cards=peeked_cards_for_obs,
                snap_results=copy.deepcopy(game_state.snap_results_log),
                did_cambia_get_called=game_state.cambia_caller_id is not None,
                who_called_cambia=game_state.cambia_caller_id,
                is_game_over=game_state.is_terminal(),
                current_turn=game_state.get_turn_number(),
            )
            return obs
        except Exception as e_obs:
            logger.error("Error creating observation for BR: %s", e_obs, exc_info=True)
            return None

    @staticmethod
    def _filter_observation_for_br(
        obs: AgentObservation, observer_id: int
    ) -> AgentObservation:
        """
        Filters observation for BR agent state updates.
        Crucially, keeps drawn_card info *if* the observer is the acting player
        and the action was Replace/Discard, as the agent needs this for state update.
        """
        filtered_obs = copy.copy(obs)

        # Keep drawn_card only if observer is actor AND action requires it
        if obs.drawn_card and obs.acting_player == observer_id:
            if not isinstance(obs.action, (ActionDiscard, ActionReplace)):
                filtered_obs.drawn_card = None  # Not needed for other actions
        elif obs.drawn_card and obs.acting_player != observer_id:
            filtered_obs.drawn_card = None  # Observer is not actor

        # BR doesn't use peeked_cards for state update (assumes perfect info implicitly)
        filtered_obs.peeked_cards = None

        return filtered_obs

    # Simulation Trace Logging

    def log_simulation_trace(self, trace_data: SimulationTrace):
        """Logs the detailed trace of a worker simulation to a JSON Lines file."""
        if not self.simulation_trace_log_path:
            # Log warning only if tracing is enabled in config
            if getattr(self.config.logging, "log_simulation_traces", False):
                logger.warning(
                    "Simulation trace logging enabled but path not set. Skipping."
                )
            return

        try:
            # Ensure trace_data conforms to the expected structure
            if (
                not isinstance(trace_data, dict)
                or "metadata" not in trace_data
                or "history" not in trace_data
            ):
                logger.warning(
                    "Attempted to log invalid simulation trace data structure: %s",
                    trace_data,
                )
                return

            with open(self.simulation_trace_log_path, "a", encoding="utf-8") as f:
                # Use the defined default_serializer helper function
                json_record = json.dumps(trace_data, default=default_serializer)
                f.write(json_record + "\n")
        except IOError as e_io:
            logger.error(
                "Error writing simulation trace to %s: %s",
                self.simulation_trace_log_path,
                e_io,
            )
        except TypeError as e_type:
            logger.error(
                "Error serializing simulation trace details to JSON: %s.", e_type
            )
            # Attempt to log problematic parts safely
            try:
                problematic_part = {
                    k: repr(v)[:200] for k, v in trace_data.get("metadata", {}).items()
                }
                problematic_part["history_len"] = len(trace_data.get("history", []))
                logger.debug("Problematic trace data (repr): %s", problematic_part)
            except Exception:
                pass  # Avoid errors during error logging
        except Exception as e_log:
            logger.error(
                "Unexpected error logging simulation trace: %s", e_log, exc_info=True
            )
