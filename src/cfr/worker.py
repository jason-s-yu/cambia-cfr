"""
src/cfr/worker.py

Implements the worker process logic for CFR+ training using External Sampling Monte Carlo CFR (ESMCFR).
Each worker simulates games based on a strategy snapshot provided by the main process,
accumulates local updates, and returns them. Uses Outcome Sampling for action selection.
"""

import copy
import logging
import multiprocessing
import os
import queue
import sys
import traceback
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, TypeAlias, Union, Any

import numpy as np

from ..agent_state import AgentObservation, AgentState
from ..config import Config
from .exceptions import (
    GameStateError,
    ActionApplicationError,
    UndoFailureError,
    AgentStateError,
    ObservationUpdateError,
    EncodingError,
    InfosetEncodingError,
    ActionEncodingError,
    TraversalError,
)

from ..card import Card
from ..constants import (
    NUM_PLAYERS,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionAbilityPeekOtherSelect,
    ActionAbilityPeekOwnSelect,
    ActionDiscard,
    ActionReplace,
    ActionSnapOpponentMove,
    DecisionContext,
    GameAction,
)
from ..game.engine import CambiaGameState
from ..game.types import StateDelta, UndoInfo
from ..serial_rotating_handler import SerialRotatingFileHandler
from ..utils import (
    InfosetKey,
    LocalReachProbUpdateDict,
    LocalRegretUpdateDict,
    LocalStrategyUpdateDict,
    WorkerResult,
    WorkerStats,
    get_rm_plus_strategy,
    normalize_probabilities,
    SimulationNodeData,
)
from ..game.helpers import serialize_card

# Type Aliases
RegretSnapshotDict: TypeAlias = Dict[InfosetKey, np.ndarray]
ProgressQueueWorker: TypeAlias = queue.Queue
ArchiveQueueWorker: TypeAlias = Union[queue.Queue, "multiprocessing.Queue"]

# Tuned progress update interval
PROGRESS_UPDATE_NODE_INTERVAL = 2500

logger = logging.getLogger(__name__)  # Get logger instance at module level


def _serialize_action_for_history(action: GameAction) -> Any:
    """Simple serialization for history log."""
    if hasattr(action, "_asdict"):
        # Attempt to serialize card objects within the action dict if possible
        action_dict = action._asdict()
        serialized_dict = {}
        for k, v in action_dict.items():
            # Use the imported Card class directly for the check
            if isinstance(v, Card):
                serialized_dict[k] = serialize_card(v)
            else:
                # Basic serialization for other types
                serialized_dict[k] = (
                    repr(v) if not isinstance(v, (int, float, bool, str)) else v
                )
        return {type(action).__name__: serialized_dict}  # Include type name
    elif isinstance(action, Card):  # Check against Card directly
        return serialize_card(action)
    elif action is None:
        return None
    else:  # Fallback for simple actions or unexpected types
        return type(action).__name__


def _traverse_game_for_worker(
    game_state: CambiaGameState,
    agent_states: List[AgentState],
    reach_probs: np.ndarray,
    iteration: int,
    updating_player: int,  # The player whose regret/strategy is being updated this iteration
    weight: float,  # Weight for this iteration (e.g., for CFR+)
    regret_sum_snapshot: RegretSnapshotDict,
    config: Config,
    local_regret_updates: LocalRegretUpdateDict,
    local_strategy_sum_updates: LocalStrategyUpdateDict,
    local_reach_prob_updates: LocalReachProbUpdateDict,
    depth: int,
    worker_stats: WorkerStats,
    progress_queue: Optional[ProgressQueueWorker],
    worker_id: int,
    min_depth_after_bottom_out_tracker: List[float],
    has_bottomed_out_tracker: List[bool],
    simulation_nodes: List[SimulationNodeData],
) -> np.ndarray:
    """
    Recursive traversal logic for ESMCFR using Outcome Sampling.
    Samples a single action based on the current strategy and recurses.
    Applies importance-weighted regret updates.
    """
    # Use the module-level logger
    logger_traverse = logging.getLogger(__name__)

    worker_stats.nodes_visited += 1
    worker_stats.max_depth = max(worker_stats.max_depth, depth)

    if has_bottomed_out_tracker[0]:
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )

    if progress_queue and (
        worker_stats.nodes_visited % PROGRESS_UPDATE_NODE_INTERVAL == 0
    ):
        try:
            progress_update = (
                worker_id,
                depth,
                worker_stats.max_depth,
                worker_stats.nodes_visited,
                (
                    int(min_depth_after_bottom_out_tracker[0])
                    if min_depth_after_bottom_out_tracker[0] != float("inf")
                    else 0
                ),
            )
            progress_queue.put_nowait(progress_update)
        except queue.Full:
            pass  # Ignore if queue is full
        except Exception as pq_e:
            logger_traverse.error(
                "W%d D%d: Error putting progress on queue: %s", worker_id, depth, pq_e
            )
            worker_stats.error_count += 1

    if game_state.is_terminal():
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        return np.array(
            [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
        )

    if depth >= config.system.recursion_limit:
        logger_traverse.error(
            "W%d D%d: Max recursion depth reached. Returning 0.", worker_id, depth
        )
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Determine context
    if game_state.snap_phase_active:
        current_context = DecisionContext.SNAP_DECISION
    elif game_state.pending_action:
        pending = game_state.pending_action
        if isinstance(pending, ActionDiscard):
            current_context = DecisionContext.POST_DRAW
        elif isinstance(
            pending,
            (
                ActionAbilityPeekOwnSelect,
                ActionAbilityPeekOtherSelect,
                ActionAbilityBlindSwapSelect,
                ActionAbilityKingLookSelect,
                ActionAbilityKingSwapDecision,
            ),
        ):
            current_context = DecisionContext.ABILITY_SELECT
        elif isinstance(pending, ActionSnapOpponentMove):
            current_context = DecisionContext.SNAP_MOVE
        else:
            logger_traverse.warning(
                "W%d D%d: Unknown pending action type (%s) for context.",
                worker_id,
                depth,
                type(pending).__name__,
            )
            worker_stats.warning_count += 1
            current_context = DecisionContext.START_TURN
    else:
        current_context = DecisionContext.START_TURN

    player = game_state.get_acting_player()
    if player == -1:
        logger_traverse.error(
            "W%d D%d: Could not determine acting player. State: %s Context: %s",
            worker_id,
            depth,
            game_state,
            current_context.name,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    opponent = 1 - player
    try:
        current_agent_state = agent_states[player]
        if not callable(current_agent_state.get_infoset_key):
            logger_traverse.error(
                "W%d D%d: Agent state P%d missing get_infoset_key. State: %s",
                worker_id,
                depth,
                player,
                current_agent_state,
            )
            worker_stats.error_count += 1
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        base_infoset_tuple = current_agent_state.get_infoset_key()
        if not isinstance(base_infoset_tuple, tuple):
            logger_traverse.error(
                "W%d D%d: get_infoset_key did not return a tuple for P%d. Got %s.",
                worker_id,
                depth,
                player,
                type(base_infoset_tuple).__name__,
            )
            worker_stats.error_count += 1
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        infoset_key = InfosetKey(*base_infoset_tuple, current_context.value)
        infoset_key_tuple = infoset_key.astuple()
    except (AgentStateError, EncodingError, InfosetEncodingError) as e_key:
        logger_traverse.warning(
            "W%d D%d: Agent/encoding error getting infoset key P%d: %s. Context: %s",
            worker_id,
            depth,
            player,
            e_key,
            current_context.name,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)
    except Exception as e_key:  # JUSTIFIED: worker resilience - workers must not crash the training pool
        logger_traverse.error(
            "W%d D%d: Error getting infoset key P%d: %s. AgentState: %s Context: %s",
            worker_id,
            depth,
            player,
            e_key,
            current_agent_state,
            current_context.name,
            exc_info=True,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    try:
        legal_actions_set = game_state.get_legal_actions()
        # Sort actions for deterministic mapping between strategy index and action
        legal_actions = sorted(list(legal_actions_set), key=repr)
    except GameStateError as e_legal:
        logger_traverse.warning(
            "W%d D%d: Game state error getting legal actions P%d: %s. Context: %s",
            worker_id,
            depth,
            player,
            e_legal,
            current_context.name,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)
    except Exception as e_legal:  # JUSTIFIED: worker resilience - workers must not crash the training pool
        logger_traverse.error(
            "W%d D%d: Error getting legal actions P%d: %s. State: %s Context: %s",
            worker_id,
            depth,
            player,
            e_legal,
            game_state,
            current_context.name,
            exc_info=True,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    num_actions = len(legal_actions)

    if num_actions == 0:
        if not game_state.is_terminal():
            logger_traverse.error(
                "W%d D%d: No legal actions P%d, but state non-terminal! State: %s Context: %s",
                worker_id,
                depth,
                player,
                game_state,
                current_context.name,
            )
            worker_stats.error_count += 1
            # Log stall node
            stall_node = SimulationNodeData(
                depth=depth,
                player=player,
                infoset_key=infoset_key_tuple,
                context=current_context.name,
                strategy=[],
                chosen_action="STALLED_NO_LEGAL_ACTIONS",
                state_delta=[],
            )
            if config.logging.log_simulation_traces:
                simulation_nodes.append(stall_node)
            return np.zeros(NUM_PLAYERS, dtype=np.float64)
        else:  # Terminal due to no legal actions
            has_bottomed_out_tracker[0] = True
            min_depth_after_bottom_out_tracker[0] = min(
                min_depth_after_bottom_out_tracker[0], float(depth)
            )
            return np.array(
                [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
            )

    # Get strategy from regrets
    current_regrets: Optional[np.ndarray] = regret_sum_snapshot.get(infoset_key)
    strategy = np.array([])
    if current_regrets is not None and len(current_regrets) == num_actions:
        strategy = get_rm_plus_strategy(current_regrets)
    else:
        if current_regrets is not None:
            logger_traverse.warning(
                "W%d D%d: Regret dim mismatch key %s. Snap:%d Need:%d. Using uniform.",
                worker_id,
                depth,
                infoset_key,
                len(current_regrets),
                num_actions,
            )
            worker_stats.warning_count += 1
        strategy = np.ones(num_actions) / num_actions if num_actions > 0 else np.array([])
        # Initialize local updates if key is new or has wrong dimension
        if (
            infoset_key not in local_regret_updates
            or len(local_regret_updates.get(infoset_key, np.array([])))
            != num_actions  # Check against np.array([])
        ):
            local_regret_updates[infoset_key] = np.zeros(num_actions, dtype=np.float64)
        if (
            infoset_key not in local_strategy_sum_updates
            or len(local_strategy_sum_updates.get(infoset_key, np.array([])))
            != num_actions  # Check against np.array([])
        ):
            local_strategy_sum_updates[infoset_key] = np.zeros(
                num_actions, dtype=np.float64
            )

    # --- Strategy Sum Update (Common to all CFR variants) ---
    player_reach = reach_probs[player]
    if weight > 0 and player_reach > 1e-9:
        if len(strategy) == num_actions:
            # Ensure local update entry exists and has correct dimension
            if (
                len(local_strategy_sum_updates.get(infoset_key, np.array([])))
                != num_actions
            ):
                local_strategy_sum_updates[infoset_key] = np.zeros(
                    num_actions, dtype=np.float64
                )
            local_strategy_sum_updates[infoset_key] += weight * player_reach * strategy
            local_reach_prob_updates[infoset_key] += weight * player_reach
        else:
            logger_traverse.error(
                "W%d D%d: Strategy len %d != num_actions %d for key %s. Skip strat update.",
                worker_id,
                depth,
                len(strategy),
                num_actions,
                infoset_key,
            )
            worker_stats.error_count += 1

    # --- Outcome Sampling: Sample one action and recurse ---
    node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)
    chosen_action_index: Optional[int] = None
    chosen_action: Optional[GameAction] = None
    sampling_prob_chosen = 0.0

    if num_actions > 0 and len(strategy) == num_actions and np.sum(strategy) > 1e-9:
        # Normalize strategy before sampling just in case
        if not np.isclose(np.sum(strategy), 1.0):
            strategy = normalize_probabilities(strategy)

        try:
            chosen_action_index = np.random.choice(num_actions, p=strategy)
            chosen_action = legal_actions[chosen_action_index]
            sampling_prob_chosen = strategy[chosen_action_index]
        except (
            ValueError
        ) as e_choice:  # Catch potential errors from invalid probabilities
            logger_traverse.error(
                "W%d D%d P%d: Error sampling action with strategy %s: %s. Using uniform.",
                worker_id,
                depth,
                player,
                strategy,
                e_choice,
            )
            worker_stats.error_count += 1
            if num_actions > 0:
                chosen_action_index = np.random.choice(num_actions)
                chosen_action = legal_actions[chosen_action_index]
                sampling_prob_chosen = 1.0 / num_actions
            else:  # Should not happen if num_actions > 0
                chosen_action_index = None
                chosen_action = None
                sampling_prob_chosen = 0.0

    elif num_actions > 0:  # Strategy was invalid (e.g., all zeros or length mismatch)
        logger_traverse.warning(
            "W%d D%d P%d: Invalid strategy %s for %d actions. Sampling uniformly.",
            worker_id,
            depth,
            player,
            strategy,
            num_actions,
        )
        worker_stats.warning_count += 1
        chosen_action_index = np.random.choice(num_actions)
        chosen_action = legal_actions[chosen_action_index]
        sampling_prob_chosen = 1.0 / num_actions
        # Use uniform strategy for regret update as well
        strategy = np.ones(num_actions) / num_actions if num_actions > 0 else np.array([])

    # If an action was chosen (either via strategy or fallback uniform)
    if chosen_action is not None and chosen_action_index is not None:
        # Log the decision node (including the sampled action)
        node_data = SimulationNodeData(
            depth=depth,
            player=player,
            infoset_key=infoset_key_tuple,
            context=current_context.name,
            strategy=strategy.tolist() if strategy.size > 0 else [],
            chosen_action=_serialize_action_for_history(
                chosen_action
            ),  # Log sampled action
            state_delta=[],  # Will be set after apply_action
        )
        if config.logging.log_simulation_traces:
            simulation_nodes.append(node_data)

        # Calculate reach probability for the next state (pass unchanged reach probs down)
        next_reach_probs = reach_probs.copy()

        # Apply action and recurse
        # Capture king swap indices before apply_action clears pending_action_data
        pre_apply_king_swap_indices = None
        if (
            isinstance(chosen_action, ActionAbilityKingSwapDecision)
            and chosen_action.perform_swap
            and game_state.pending_action_data
        ):
            pad = game_state.pending_action_data
            if "own_idx" in pad and "opp_idx" in pad:
                pre_apply_king_swap_indices = (pad["own_idx"], pad["opp_idx"])

        state_delta: Optional[StateDelta] = None
        undo_info: Optional[UndoInfo] = None
        apply_success = False
        try:
            state_delta, undo_info = game_state.apply_action(chosen_action)
            if callable(undo_info):
                apply_success = True
                # Update node_data with delta if tracing
                if (
                    config.logging.log_simulation_traces
                    and simulation_nodes
                    and simulation_nodes[-1] is node_data
                ):
                    node_data["state_delta"] = [list(d) for d in state_delta]

            else:
                logger_traverse.error(
                    "W%d D%d P%d: apply_action for sampled %s returned invalid undo_info. State:%s",
                    worker_id,
                    depth,
                    player,
                    chosen_action,
                    game_state,
                )
                worker_stats.error_count += 1
        except ActionApplicationError as apply_err:
            logger_traverse.warning(
                "W%d D%d P%d: Action application error for %s: %s. Context:%s",
                worker_id,
                depth,
                player,
                chosen_action,
                apply_err,
                current_context.name,
            )
            worker_stats.error_count += 1
            # Update node_data with error if tracing
            if (
                config.logging.log_simulation_traces
                and simulation_nodes
                and simulation_nodes[-1] is node_data
            ):
                node_data["state_delta"] = [("apply_error", str(apply_err))]
        except Exception as apply_err:  # JUSTIFIED: worker resilience - workers must not crash the training pool
            logger_traverse.error(
                "W%d D%d P%d: Error applying sampled action %s: %s. State:%s Context:%s",
                worker_id,
                depth,
                player,
                chosen_action,
                apply_err,
                game_state,
                current_context.name,
                exc_info=True,
            )
            worker_stats.error_count += 1
            # Update node_data with error if tracing
            if (
                config.logging.log_simulation_traces
                and simulation_nodes
                and simulation_nodes[-1] is node_data
            ):
                node_data["state_delta"] = [("apply_error", str(apply_err))]

        if apply_success:
            # Create observation
            observation = _create_observation(
                None,
                chosen_action,
                game_state,
                player,
                game_state.snap_results_log,  # Pass CURRENT snap log
                king_swap_indices=pre_apply_king_swap_indices,
            )
            next_agent_states = []
            agent_update_failed = False
            player_specific_obs_for_log = None
            if observation is None:  # Check if observation creation failed
                logger_traverse.error(
                    "W%d D%d: Failed to create observation after action %s.",
                    worker_id,
                    depth,
                    chosen_action,
                )
                agent_update_failed = True  # Mark as failed to prevent recursion
                worker_stats.error_count += 1
                if undo_info:
                    try:
                        undo_info()
                    except UndoFailureError as undo_e:
                        logger_traverse.warning(
                            "W%d D%d: Undo failure after obs create fail: %s",
                            worker_id,
                            depth,
                            undo_e,
                        )
                    except Exception as undo_e:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                        logger_traverse.error(
                            "W%d D%d: Error undoing after obs create fail: %s",
                            worker_id,
                            depth,
                            undo_e,
                        )
            else:
                try:
                    for agent_idx, agent_state in enumerate(agent_states):
                        cloned_agent = agent_state.clone()
                        player_specific_obs = _filter_observation(observation, agent_idx)
                        if agent_idx == player:
                            player_specific_obs_for_log = player_specific_obs
                        cloned_agent.update(player_specific_obs)
                        next_agent_states.append(cloned_agent)
                except (AgentStateError, ObservationUpdateError) as e_update:
                    failed_agent_idx = agent_idx  # Capture index where failure occurred
                    logger_traverse.warning(
                        "W%d D%d: Agent state update error P%d after action %s: %s",
                        worker_id,
                        depth,
                        failed_agent_idx,
                        chosen_action,
                        e_update,
                    )
                    worker_stats.error_count += 1
                    agent_update_failed = True
                    if undo_info:
                        try:
                            undo_info()
                        except UndoFailureError as undo_e:
                            logger_traverse.warning(
                                "W%d D%d: Undo failure after agent update fail: %s",
                                worker_id,
                                depth,
                                undo_e,
                            )
                            worker_stats.error_count += 1
                        except Exception as undo_e:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                            logger_traverse.error(
                                "W%d D%d: Error undoing after agent update fail: %s",
                                worker_id,
                                depth,
                                undo_e,
                                exc_info=True,
                            )
                            worker_stats.error_count += 1
                except Exception as e_update:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                    failed_agent_idx = agent_idx  # Capture index where failure occurred
                    logger_traverse.error(
                        "W%d D%d: Error updating agent P%d after action %s: %s. State(post-action):%s FilteredObs:%s",
                        worker_id,
                        depth,
                        failed_agent_idx,
                        chosen_action,
                        e_update,
                        game_state,
                        player_specific_obs_for_log,  # May be None if error happened before P0 update
                        exc_info=True,
                    )
                    worker_stats.error_count += 1
                    agent_update_failed = True
                    if undo_info:
                        try:
                            undo_info()
                        except UndoFailureError as undo_e:
                            logger_traverse.warning(
                                "W%d D%d: Undo failure after agent update fail: %s",
                                worker_id,
                                depth,
                                undo_e,
                            )
                            worker_stats.error_count += 1
                        except Exception as undo_e:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                            logger_traverse.error(
                                "W%d D%d: Error undoing after agent update fail: %s",
                                worker_id,
                                depth,
                                undo_e,
                                exc_info=True,
                            )
                            worker_stats.error_count += 1

            if not agent_update_failed:
                try:
                    # Single recursive call for the sampled action
                    node_value = _traverse_game_for_worker(
                        game_state,
                        next_agent_states,
                        next_reach_probs,  # Pass original reach probs down
                        iteration,
                        updating_player,
                        weight,
                        regret_sum_snapshot,
                        config,
                        local_regret_updates,
                        local_strategy_sum_updates,
                        local_reach_prob_updates,
                        depth + 1,
                        worker_stats,
                        progress_queue,
                        worker_id,
                        min_depth_after_bottom_out_tracker,
                        has_bottomed_out_tracker,
                        simulation_nodes,
                    )
                except TraversalError as recursive_err:
                    logger_traverse.warning(
                        "W%d D%d: Traversal error in recursive call after action %s: %s",
                        worker_id,
                        depth,
                        chosen_action,
                        recursive_err,
                    )
                    worker_stats.error_count += 1
                    node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)
                    if undo_info:
                        try:
                            undo_info()
                        except UndoFailureError as undo_e:
                            logger_traverse.warning(
                                "W%d D%d: Undo failure after traversal error: %s",
                                worker_id,
                                depth,
                                undo_e,
                            )
                            worker_stats.error_count += 1
                        except Exception as undo_e:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                            logger_traverse.error(
                                "W%d D%d: Error undoing after traversal error: %s",
                                worker_id,
                                depth,
                                undo_e,
                                exc_info=True,
                            )
                            worker_stats.error_count += 1
                except Exception as recursive_err:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                    logger_traverse.error(
                        "W%d D%d: Error in recursive call after action %s: %s. State:%s Context:%s",
                        worker_id,
                        depth,
                        chosen_action,
                        recursive_err,
                        game_state,
                        current_context.name,
                        exc_info=True,
                    )
                    worker_stats.error_count += 1
                    node_value = np.zeros(
                        NUM_PLAYERS, dtype=np.float64
                    )  # Set to zero on error
                    if undo_info:
                        try:
                            undo_info()
                        except UndoFailureError as undo_e:
                            logger_traverse.warning(
                                "W%d D%d: Undo failure after recursion error: %s",
                                worker_id,
                                depth,
                                undo_e,
                            )
                            worker_stats.error_count += 1
                        except Exception as undo_e:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                            logger_traverse.error(
                                "W%d D%d: Error undoing after recursion error: %s",
                                worker_id,
                                depth,
                                undo_e,
                                exc_info=True,
                            )
                            worker_stats.error_count += 1

                # Undo action after recursion returns
                if undo_info:
                    try:
                        undo_info()
                    except UndoFailureError as undo_e:
                        logger_traverse.error(
                            "W%d D%d P%d: Undo failure for action %s: %s. State likely corrupt. Returning zero.",
                            worker_id,
                            depth,
                            player,
                            chosen_action,
                            undo_e,
                            exc_info=True,
                        )
                        worker_stats.error_count += 1
                        return np.zeros(NUM_PLAYERS, dtype=np.float64)
                    except Exception as undo_e:  # JUSTIFIED: worker resilience - must not crash on undo, state likely corrupt
                        logger_traverse.error(
                            "W%d D%d P%d: Error undoing action %s: %s. State likely corrupt. Returning zero.",
                            worker_id,
                            depth,
                            player,
                            chosen_action,
                            undo_e,
                            exc_info=True,
                        )
                        worker_stats.error_count += 1
                        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # --- Outcome Sampling Regret Update ---
    if player == updating_player and chosen_action_index is not None:
        if len(strategy) == num_actions:  # Check strategy validity
            sampled_utility = node_value[player]
            if sampling_prob_chosen > 1e-9:
                utility_estimate = sampled_utility / sampling_prob_chosen

                instantaneous_regrets = np.zeros(num_actions, dtype=np.float64)
                for a_idx in range(num_actions):
                    action_value_estimate = (
                        1.0 if a_idx == chosen_action_index else 0.0
                    ) * utility_estimate
                    instantaneous_regrets[a_idx] = (
                        action_value_estimate - strategy[a_idx] * utility_estimate
                    )

                # Ensure local regret update entry exists and has correct dimension
                if (
                    len(local_regret_updates.get(infoset_key, np.array([])))
                    != num_actions
                ):
                    local_regret_updates[infoset_key] = np.zeros(
                        num_actions, dtype=np.float64
                    )

                opponent_reach = reach_probs[opponent]
                update_weight = opponent_reach * weight  # Include iteration weight
                local_regret_updates[infoset_key] += update_weight * instantaneous_regrets

            else:  # Sampling probability near zero
                logger_traverse.debug(
                    "W%d D%d P%d: Sampling prob %.3e near zero for action %s at key %s. Skipping regret update.",
                    worker_id,
                    depth,
                    player,
                    sampling_prob_chosen,
                    chosen_action,
                    infoset_key,
                )
        else:  # Strategy was invalid when sampling occurred
            logger_traverse.warning(
                "W%d D%d P%d: Cannot perform regret update, strategy was invalid during sampling.",
                worker_id,
                depth,
                player,
            )

    return node_value  # Return the utility vector obtained from the single sampled path


def run_cfr_simulation_worker(
    worker_args: Tuple[
        int,
        Config,
        RegretSnapshotDict,
        Optional[ProgressQueueWorker],
        Optional[ArchiveQueueWorker],
        int,
        str,
        str,
    ],
) -> Optional[WorkerResult]:
    """Top-level function executed by each worker process. Sets up per-worker logging."""
    # Initialize logger_instance to None
    logger_instance: Optional[logging.Logger] = None
    worker_stats = WorkerStats()
    (
        iteration,
        config,
        regret_sum_snapshot,
        progress_queue,
        archive_queue,
        worker_id,
        run_log_dir,
        run_timestamp,
    ) = worker_args

    # Store worker ID in stats
    worker_stats.worker_id = worker_id

    # Initialize trace list for this simulation
    simulation_nodes_this_sim: List[SimulationNodeData] = []
    final_utility_value: Optional[np.ndarray] = None

    worker_root_logger = logging.getLogger()
    try:
        # Clear existing handlers for this process
        for handler in worker_root_logger.handlers[:]:
            worker_root_logger.removeHandler(handler)
            if hasattr(handler, "close"):
                try:
                    handler.close()
                except Exception:
                    pass

        # Set root level BEFORE adding handlers
        worker_root_logger.setLevel(logging.DEBUG)

        # Add NullHandler to prevent defaults and stop propagation
        null_handler = logging.NullHandler()
        worker_root_logger.addHandler(null_handler)
        worker_root_logger.propagate = False  # Prevent logs reaching main process root

        # Setup per-worker logging to file
        worker_log_dir = os.path.join(run_log_dir, f"w{worker_id}")
        os.makedirs(worker_log_dir, exist_ok=True)
        log_pattern = os.path.join(
            worker_log_dir,
            f"{config.logging.log_file_prefix}_run_{run_timestamp}-w{worker_id}",
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)-8s - [%(processName)-20s] - %(name)-25s - %(message)s"
        )
        file_handler = SerialRotatingFileHandler(
            filename_pattern=log_pattern,
            maxBytes=config.logging.log_max_bytes,
            backupCount=config.logging.log_backup_count,
            encoding="utf-8",
            archive_queue=archive_queue,
            logging_config_snapshot=config.logging,
        )
        worker_log_level_str = config.logging.get_worker_log_level(
            worker_id, config.cfr_training.num_workers
        )
        file_log_level = getattr(logging, worker_log_level_str.upper(), logging.DEBUG)
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)
        worker_root_logger.addHandler(file_handler)  # Add specific file handler

        # Now get the logger instance for this module AFTER setup
        logger_instance = logging.getLogger(__name__)
        logger_instance.info(
            "Worker %d logging initialized (dir: %s, file_level: %s). Root handlers: %s",
            worker_id,
            worker_log_dir,
            logging.getLevelName(file_log_level),
            [type(h).__name__ for h in worker_root_logger.handlers],
        )

    except Exception as log_setup_e:
        # Fallback logging if setup fails
        print(
            f"!!! CRITICAL Error setting up logging W{worker_id}: {log_setup_e} !!!",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        worker_stats.error_count += 1
        # Ensure the root logger doesn't propagate setup errors if handlers fail
        if not worker_root_logger.hasHandlers():
            worker_root_logger.addHandler(logging.NullHandler())
        logger_instance = logging.getLogger(__name__)  # Still try to get logger

    # Main simulation logic
    try:
        # --- Game and Agent State Initialization ---
        try:
            game_state = CambiaGameState(house_rules=config.cambia_rules)
        except GameStateError as game_init_e:
            if logger_instance:
                logger_instance.warning(
                    "W%d Iter %d: Game state initialization error: %s",
                    worker_id,
                    iteration,
                    game_init_e,
                )
            worker_stats.error_count += 1
            return WorkerResult(
                stats=worker_stats,
                simulation_nodes=simulation_nodes_this_sim,
                final_utility=None,
            )
        except Exception as game_init_e:  # JUSTIFIED: worker resilience - workers must not crash the training pool
            if logger_instance:
                logger_instance.error(
                    "W%d Iter %d: Failed GameState init: %s",
                    worker_id,
                    iteration,
                    game_init_e,
                    exc_info=True,
                )
            worker_stats.error_count += 1
            # Return minimal result indicating failure
            return WorkerResult(
                stats=worker_stats,
                simulation_nodes=simulation_nodes_this_sim,
                final_utility=None,
            )

        initial_agent_states = []
        if not game_state.is_terminal():
            try:
                # Create observation needed for AgentState initialization
                initial_obs = _create_observation(
                    None, None, game_state, -1, []
                )  # No explicit card needed
                if initial_obs is None:
                    raise ValueError("Failed to create initial observation.")

                initial_hands = [list(p.hand) for p in game_state.players]
                initial_peeks = [p.initial_peek_indices for p in game_state.players]
                for i in range(NUM_PLAYERS):
                    agent = AgentState(
                        player_id=i,
                        opponent_id=1 - i,
                        memory_level=config.agent_params.memory_level,
                        time_decay_turns=config.agent_params.time_decay_turns,
                        initial_hand_size=len(initial_hands[i]),
                        config=config,
                    )
                    agent.initialize(initial_obs, initial_hands[i], initial_peeks[i])
                    initial_agent_states.append(agent)
            except (AgentStateError, ObservationUpdateError, EncodingError) as agent_init_e:
                if logger_instance:
                    logger_instance.warning(
                        "W%d Iter %d: Agent state initialization error: %s",
                        worker_id,
                        iteration,
                        agent_init_e,
                    )
                worker_stats.error_count += 1
                return WorkerResult(
                    stats=worker_stats,
                    simulation_nodes=simulation_nodes_this_sim,
                    final_utility=None,
                )
            except Exception as agent_init_e:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                if logger_instance:
                    logger_instance.error(
                        "W%d Iter %d: Failed AgentStates init: %s. GameState: %s",
                        worker_id,
                        iteration,
                        agent_init_e,
                        game_state,
                        exc_info=True,
                    )
                worker_stats.error_count += 1
                return WorkerResult(
                    stats=worker_stats,
                    simulation_nodes=simulation_nodes_this_sim,
                    final_utility=None,
                )
        else:  # Game terminal at start
            if logger_instance:
                logger_instance.warning(
                    "W%d Iter %d: Game terminal at init. State: %s",
                    worker_id,
                    iteration,
                    game_state,
                )
            final_utility_value = np.array(
                [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
            )
            return WorkerResult(
                stats=worker_stats,
                simulation_nodes=simulation_nodes_this_sim,
                final_utility=final_utility_value.tolist(),
            )

        if len(initial_agent_states) != NUM_PLAYERS:
            if logger_instance:
                logger_instance.error(
                    "W%d Iter %d: Incorrect agent states initialized (%d).",
                    worker_id,
                    iteration,
                    len(initial_agent_states),
                )
            worker_stats.error_count += 1
            return WorkerResult(
                stats=worker_stats,
                simulation_nodes=simulation_nodes_this_sim,
                final_utility=None,
            )

        # --- Traversal ---
        updating_player = iteration % NUM_PLAYERS
        weight = (
            float(max(0, (iteration + 1) - (config.cfr_plus_params.averaging_delay + 1)))
            if config.cfr_plus_params.weighted_averaging_enabled
            else 1.0
        )
        local_regret_updates: LocalRegretUpdateDict = defaultdict(
            lambda: np.array([], dtype=np.float64)
        )
        local_strategy_sum_updates: LocalStrategyUpdateDict = defaultdict(
            lambda: np.array([], dtype=np.float64)
        )
        local_reach_prob_updates: LocalReachProbUpdateDict = defaultdict(float)
        min_depth_after_bottom_out_tracker = [float("inf")]
        has_bottomed_out_tracker = [False]

        # Run the ESMCFR traversal (Outcome Sampling)
        final_utility_value = _traverse_game_for_worker(
            game_state=game_state,
            agent_states=initial_agent_states,
            reach_probs=np.ones(NUM_PLAYERS, dtype=np.float64),
            iteration=iteration,
            updating_player=updating_player,
            weight=weight,
            regret_sum_snapshot=regret_sum_snapshot,
            config=config,
            local_regret_updates=local_regret_updates,
            local_strategy_sum_updates=local_strategy_sum_updates,
            local_reach_prob_updates=local_reach_prob_updates,
            depth=0,
            worker_stats=worker_stats,
            progress_queue=progress_queue,
            worker_id=worker_id,
            min_depth_after_bottom_out_tracker=min_depth_after_bottom_out_tracker,
            has_bottomed_out_tracker=has_bottomed_out_tracker,
            simulation_nodes=simulation_nodes_this_sim,
        )

        # Note: final_utility_value here is the return from the recursive call,
        # which represents the utility obtained from the single sampled path.
        if final_utility_value is None or len(final_utility_value) != NUM_PLAYERS:
            if logger_instance:
                logger_instance.error(
                    "W%d Iter %d: Traversal returned invalid utility: %s. Setting final to zero.",
                    worker_id,
                    iteration,
                    final_utility_value,
                )
            worker_stats.error_count += 1
            final_utility_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

        worker_stats.min_depth_after_bottom_out = (
            int(min_depth_after_bottom_out_tracker[0])
            if min_depth_after_bottom_out_tracker[0] != float("inf")
            else 0
        )

        # --- Return Result ---
        return WorkerResult(
            regret_updates=dict(local_regret_updates),
            strategy_updates=dict(local_strategy_sum_updates),
            reach_prob_updates=dict(local_reach_prob_updates),
            stats=worker_stats,
            simulation_nodes=simulation_nodes_this_sim,
            final_utility=(
                final_utility_value.tolist() if final_utility_value is not None else None
            ),
        )

    except KeyboardInterrupt:
        if logger_instance:
            logger_instance.warning(
                "W%d Iter %d received KeyboardInterrupt.", worker_id, iteration
            )
        worker_stats.error_count += 1
        # Return partial results if interrupted
        return WorkerResult(
            stats=worker_stats,
            simulation_nodes=simulation_nodes_this_sim,
            final_utility=None,
        )
    except Exception as e_inner:  # JUSTIFIED: worker resilience - top-level worker catch to prevent pool crash
        worker_stats.error_count += 1
        if logger_instance:
            logger_instance.critical(
                "!!! Unhandled Error W%d Iter %d simulation: %s !!!",
                worker_id,
                iteration,
                e_inner,
                exc_info=True,
            )
        # Also print to stderr for visibility if logging fails
        print(
            f"!!! FATAL WORKER ERROR W{worker_id} Iter {iteration}: {e_inner} !!!",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        return WorkerResult(
            stats=worker_stats,
            simulation_nodes=simulation_nodes_this_sim,
            final_utility=None,
        )
    finally:
        # Ensure logs are flushed and handlers closed on worker exit
        if logger_instance:
            for handler in logger_instance.handlers[:]:
                if hasattr(handler, "flush"):
                    try:
                        handler.flush()
                    except Exception:
                        pass
                if hasattr(handler, "close"):
                    try:
                        handler.close()
                    except Exception:
                        pass


# --- Observation Helpers ---
# (Keep these helpers here as they are used by the worker's traversal logic)
def _create_observation(
    prev_state: Optional[
        CambiaGameState
    ],  # Kept for signature consistency if needed elsewhere
    action: Optional[GameAction],
    next_state: CambiaGameState,
    acting_player: int,
    snap_results: List[Dict],
    king_swap_indices: Optional[tuple] = None,  # (own_idx, opp_idx) for king swap
) -> Optional[AgentObservation]:
    """Creates the AgentObservation object based on the state *after* the action."""
    logger_obs = logging.getLogger(__name__)  # Use module logger
    try:
        discard_top = next_state.get_discard_top()
        hand_sizes = [next_state.get_player_card_count(i) for i in range(NUM_PLAYERS)]
        stock_size = next_state.get_stockpile_size()
        cambia_called = next_state.cambia_caller_id is not None
        who_called = next_state.cambia_caller_id
        game_over = next_state.is_terminal()
        turn_num = next_state.get_turn_number()

        # Determine drawn card based on action and NEXT state
        drawn_card_for_obs = None
        if isinstance(action, ActionDiscard):
            # The card just discarded *was* the drawn card. It's now the top of discard.
            drawn_card_for_obs = next_state.get_discard_top()
            if drawn_card_for_obs is None:
                logger_obs.error("Create Obs: ActionDiscard but discard pile empty?")
        elif isinstance(action, ActionReplace):
            # The card just placed into the hand *was* the drawn card.
            if acting_player != -1 and action.target_hand_index < len(
                next_state.players[acting_player].hand
            ):
                drawn_card_for_obs = next_state.players[acting_player].hand[
                    action.target_hand_index
                ]
            else:
                logger_obs.error(
                    "Create Obs: ActionReplace index %d invalid for actor %d hand size %d",
                    action.target_hand_index if action else -1,
                    acting_player,
                    (
                        len(next_state.players[acting_player].hand)
                        if acting_player != -1
                        else -1
                    ),
                )

        # Populate peeked cards *only* if the action caused a peek
        peeked_cards_dict = None
        if isinstance(action, ActionAbilityPeekOwnSelect) and acting_player != -1:
            hand = next_state.get_player_hand(acting_player)
            target_idx = action.target_hand_index
            if hand and 0 <= target_idx < len(hand):
                card = hand[target_idx]
                peeked_cards_dict = {(acting_player, target_idx): card}
            else:
                logger_obs.warning(
                    "Create Obs: PeekOwn index %d invalid for hand size %d.",
                    target_idx,
                    len(hand) if hand else 0,
                )
        elif isinstance(action, ActionAbilityPeekOtherSelect) and acting_player != -1:
            opp_idx = next_state.get_opponent_index(acting_player)
            opp_hand = next_state.get_player_hand(opp_idx)
            target_opp_idx = action.target_opponent_hand_index
            if opp_hand and 0 <= target_opp_idx < len(opp_hand):
                card = opp_hand[target_opp_idx]
                peeked_cards_dict = {(opp_idx, target_opp_idx): card}
            else:
                logger_obs.warning(
                    "Create Obs: PeekOther index %d invalid for opp hand size %d.",
                    target_opp_idx,
                    len(opp_hand) if opp_hand else 0,
                )
        elif isinstance(action, ActionAbilityKingLookSelect) and acting_player != -1:
            # Re-fetch cards looked at for the observation
            own_idx, opp_look_idx = action.own_hand_index, action.opponent_hand_index
            opp_real_idx = next_state.get_opponent_index(acting_player)
            own_hand = next_state.get_player_hand(acting_player)
            opp_hand = next_state.get_player_hand(opp_real_idx)
            card1, card2 = None, None
            if own_hand and 0 <= own_idx < len(own_hand):
                card1 = own_hand[own_idx]
            if opp_hand and 0 <= opp_look_idx < len(opp_hand):
                card2 = opp_hand[opp_look_idx]
            if card1 and card2:
                peeked_cards_dict = {
                    (acting_player, own_idx): card1,
                    (opp_real_idx, opp_look_idx): card2,
                }
            else:
                logger_obs.warning(
                    "Create Obs: KingLook indices invalid/cards missing. Own %s/%s, Opp %s/%s",
                    own_idx,
                    len(own_hand) if own_hand else "N/A",
                    opp_look_idx,
                    len(opp_hand) if opp_hand else "N/A",
                )

        final_snap_results = snap_results if snap_results else []

        # Populate king_swap_indices if this action is a performed king swap
        obs_king_swap_indices = king_swap_indices
        if (
            obs_king_swap_indices is None
            and isinstance(action, ActionAbilityKingSwapDecision)
            and action.perform_swap
        ):
            # Try to read from next_state pending_action_data (may already be cleared)
            pad = next_state.pending_action_data
            if pad and "own_idx" in pad and "opp_idx" in pad:
                obs_king_swap_indices = (pad["own_idx"], pad["opp_idx"])

        obs = AgentObservation(
            acting_player=acting_player,
            action=action,
            discard_top_card=discard_top,
            player_hand_sizes=hand_sizes,
            stockpile_size=stock_size,
            drawn_card=drawn_card_for_obs,  # Determined from next_state based on action
            peeked_cards=peeked_cards_dict,
            snap_results=final_snap_results,
            did_cambia_get_called=cambia_called,
            who_called_cambia=who_called,
            is_game_over=game_over,
            current_turn=turn_num,
            king_swap_indices=obs_king_swap_indices,
        )
        logger_obs.debug("Created observation: %s", obs)
        return obs
    except GameStateError as e:
        logger_obs.warning("Game state error creating observation: %s", e)
        return None
    except Exception as e:  # JUSTIFIED: worker resilience - observation creation must not crash worker
        logger_obs.error("Error creating observation: %s", e, exc_info=True)
        return None


def _filter_observation(obs: AgentObservation, observer_id: int) -> AgentObservation:
    """Creates a player-specific view of the observation, masking private info."""
    # Shallow copy is usually sufficient as AgentState doesn't modify observation fields
    filtered_obs = copy.copy(obs)

    # Mask drawn card unless observer is the actor AND the action requires the drawn card info
    if obs.drawn_card and obs.acting_player != observer_id:
        filtered_obs.drawn_card = None
    elif obs.drawn_card and obs.acting_player == observer_id:
        # Keep drawn_card only if the action is Discard or Replace (when agent needs it)
        if not isinstance(obs.action, (ActionDiscard, ActionReplace)):
            filtered_obs.drawn_card = None

    # Filter peeked cards - only pass if observer was the actor performing the peek/look
    if obs.peeked_cards:
        is_observer_peek_action = (
            (
                isinstance(obs.action, ActionAbilityPeekOwnSelect)
                and obs.acting_player == observer_id
            )
            or (
                isinstance(obs.action, ActionAbilityPeekOtherSelect)
                and obs.acting_player == observer_id
            )
            or (
                isinstance(obs.action, ActionAbilityKingLookSelect)
                and obs.acting_player == observer_id
            )
        )
        if not is_observer_peek_action:
            filtered_obs.peeked_cards = None
        # Note: KingSwapDecision doesn't reveal peek info in *this* observation's action
    else:
        filtered_obs.peeked_cards = None

    # Snap results are public information derived from game state deltas
    filtered_obs.snap_results = obs.snap_results

    return filtered_obs
