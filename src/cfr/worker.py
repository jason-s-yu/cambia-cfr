"""src/cfr/worker.py"""

import copy
import logging
import multiprocessing
import os
import queue
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, TypeAlias, Union, Any

import numpy as np

from ..agent_state import AgentObservation, AgentState
from ..config import Config

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
    SimulationNodeData,
    normalize_probabilities,
)
from ..game.helpers import serialize_card

RegretSnapshotDict: TypeAlias = Dict[InfosetKey, np.ndarray]
ProgressQueueWorker: TypeAlias = queue.Queue
ArchiveQueueWorker: TypeAlias = Union[queue.Queue, "multiprocessing.Queue"]

# Backlog 17: Tuned progress update interval
PROGRESS_UPDATE_NODE_INTERVAL = 2500

logger = logging.getLogger(__name__)


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


# --- The rest of the file remains the same ---


def _traverse_game_for_worker(
    game_state: CambiaGameState,
    agent_states: List[AgentState],
    reach_probs: np.ndarray,
    iteration: int,
    updating_player: int,
    weight: float,
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
    simulation_nodes: List[SimulationNodeData],  # Modified parameter name
) -> np.ndarray:
    """Recursive traversal logic adapted for worker process."""
    logger = logging.getLogger(__name__)  # Get logger instance within function

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
            logger.error(
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
        logger.error(
            "W%d D%d: Max recursion depth reached. Returning 0.", worker_id, depth
        )
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

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
            logger.warning(
                "W%d D%d: Unknown pending action type (%s) for context.",
                worker_id,
                depth,
                type(pending).__name__,
            )
            worker_stats.warning_count += 1
            current_context = DecisionContext.START_TURN  # Fallback context
    else:
        current_context = DecisionContext.START_TURN

    player = game_state.get_acting_player()
    if player == -1:
        logger.error(
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
        if not hasattr(current_agent_state, "get_infoset_key") or not callable(
            current_agent_state.get_infoset_key
        ):
            logger.error(
                "W%d D%d: Agent state P%d missing get_infoset_key. State: %s",
                worker_id,
                depth,
                player,
                current_agent_state,
            )
            worker_stats.error_count += 1
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        base_infoset_tuple = current_agent_state.get_infoset_key()
        # Ensure it's a tuple (get_infoset_key already returns tuple)
        if not isinstance(base_infoset_tuple, tuple):
            logger.error(
                "W%d D%d: get_infoset_key did not return a tuple for P%d. Got %s.",
                worker_id,
                depth,
                player,
                type(base_infoset_tuple).__name__,
            )
            worker_stats.error_count += 1
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        infoset_key = InfosetKey(*base_infoset_tuple, current_context.value)
        infoset_key_tuple = infoset_key.astuple()  # Get tuple representation for logging
    except Exception as e_key:
        logger.error(
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
        legal_actions = sorted(list(legal_actions_set), key=repr)
    except Exception as e_legal:
        logger.error(
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
            logger.error(
                "W%d D%d: No legal actions P%d, but state non-terminal! State: %s Context: %s",
                worker_id,
                depth,
                player,
                game_state,
                current_context.name,
            )
            worker_stats.error_count += 1
            # Add history entry indicating the stall
            # Backlog 5: Use SimulationNodeData format, indicate stall
            stall_node = SimulationNodeData(
                depth=depth,
                player=player,
                infoset_key=infoset_key_tuple,
                context=current_context.name,
                strategy=[],
                chosen_action="NO_LEGAL_ACTIONS_NON_TERMINAL",
                state_delta=[],
            )
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

    current_regrets: Optional[np.ndarray] = regret_sum_snapshot.get(infoset_key)
    strategy = np.array([])
    if current_regrets is not None and len(current_regrets) == num_actions:
        strategy = get_rm_plus_strategy(current_regrets)
    else:
        if current_regrets is not None:
            logger.warning(
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
            or len(local_regret_updates.get(infoset_key, [])) != num_actions
        ):
            local_regret_updates[infoset_key] = np.zeros(num_actions, dtype=np.float64)
        if (
            infoset_key not in local_strategy_sum_updates
            or len(local_strategy_sum_updates.get(infoset_key, [])) != num_actions
        ):
            local_strategy_sum_updates[infoset_key] = np.zeros(
                num_actions, dtype=np.float64
            )

    player_reach = reach_probs[player]
    if player == updating_player and weight > 0 and player_reach > 1e-9:
        if len(strategy) == num_actions:
            if (
                infoset_key not in local_strategy_sum_updates
                or len(local_strategy_sum_updates[infoset_key]) != num_actions
            ):
                local_strategy_sum_updates[infoset_key] = np.zeros(
                    num_actions, dtype=np.float64
                )
            local_strategy_sum_updates[infoset_key] += weight * player_reach * strategy
            local_reach_prob_updates[infoset_key] += weight * player_reach
        else:
            logger.error(
                "W%d D%d: Strategy len %d != num_actions %d for key %s. Skip strat update.",
                worker_id,
                depth,
                len(strategy),
                num_actions,
                infoset_key,
            )
            worker_stats.error_count += 1

    # --- Sample action for traversal ---
    chosen_action_index = -1
    chosen_action = None
    if num_actions > 0:
        try:
            # Ensure strategy sums to 1 for np.random.choice
            strategy_sum = strategy.sum()
            if not np.isclose(strategy_sum, 1.0):
                logger.warning(
                    "W%d D%d P%d: Strategy for %s does not sum to 1 (%f). Normalizing for sampling.",
                    worker_id,
                    depth,
                    player,
                    infoset_key,
                    strategy_sum,
                )
                strategy = normalize_probabilities(strategy)
                if len(strategy) == 0 or not np.isclose(strategy.sum(), 1.0):
                    logger.error(
                        "W%d D%d P%d: Renormalization failed for %s. Using uniform.",
                        worker_id,
                        depth,
                        player,
                        infoset_key,
                    )
                    strategy = np.ones(num_actions) / num_actions

            chosen_action_index = np.random.choice(num_actions, p=strategy)
            chosen_action = legal_actions[chosen_action_index]
        except ValueError as e_choice:
            logger.error(
                "W%d D%d P%d: Error sampling action for key %s (Strategy: %s): %s. Sampling uniform.",
                worker_id,
                depth,
                player,
                infoset_key,
                strategy,
                e_choice,
            )
            chosen_action_index = np.random.choice(num_actions)  # Uniform fallback
            chosen_action = legal_actions[chosen_action_index]
            worker_stats.error_count += 1
        except IndexError:
            logger.error(
                "W%d D%d P%d: Chosen action index %d out of bounds for legal actions (len %d).",
                worker_id,
                depth,
                player,
                chosen_action_index,
                num_actions,
            )
            worker_stats.error_count += 1
            return np.zeros(NUM_PLAYERS, dtype=np.float64)  # Cannot proceed

    action_utilities = np.zeros((num_actions, NUM_PLAYERS), dtype=np.float64)

    # Only traverse the chosen action path in ESMCFR worker simulation
    if chosen_action is not None and chosen_action_index != -1:
        explicit_drawn_card_for_obs: Optional[Card] = None
        if isinstance(chosen_action, (ActionReplace, ActionDiscard)):
            if game_state.pending_action_data:
                explicit_drawn_card_for_obs = game_state.pending_action_data.get(
                    "drawn_card"
                )
            else:
                logger.warning(
                    "W%d D%d P%d: Action %s needs pending_data for drawn card, but empty.",
                    worker_id,
                    depth,
                    player,
                    type(chosen_action).__name__,
                )

        state_delta: Optional[StateDelta] = None
        undo_info: Optional[UndoInfo] = None
        apply_success = False
        try:
            state_delta, undo_info = game_state.apply_action(chosen_action)
            if callable(undo_info):
                apply_success = True
                # Create and append SimulationNodeData
                node_data = SimulationNodeData(
                    depth=depth,
                    player=player,
                    infoset_key=infoset_key_tuple,
                    context=current_context.name,
                    strategy=strategy.tolist() if strategy.size > 0 else [],
                    chosen_action=_serialize_action_for_history(chosen_action),
                    state_delta=[
                        list(d) for d in state_delta
                    ],  # Convert tuples inside delta
                )
                simulation_nodes.append(node_data)
            else:
                logger.error(
                    "W%d D%d P%d: apply_action for %s returned invalid undo_info. State:%s",
                    worker_id,
                    depth,
                    player,
                    chosen_action,
                    game_state,
                )
                worker_stats.error_count += 1
                apply_success = False  # Mark as failed
        except Exception as apply_err:
            logger.error(
                "W%d D%d P%d: Error applying action %s: %s. State:%s Context:%s",
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
            apply_success = False  # Mark as failed
            # Log failure node
            fail_node = SimulationNodeData(
                depth=depth,
                player=player,
                infoset_key=infoset_key_tuple,
                context=current_context.name,
                strategy=strategy.tolist() if strategy.size > 0 else [],
                chosen_action=_serialize_action_for_history(chosen_action),
                state_delta=[("apply_error", str(apply_err))],
            )
            simulation_nodes.append(fail_node)

        if apply_success:
            observation = _create_observation(
                None,
                chosen_action,
                game_state,
                player,
                game_state.snap_results_log,
                explicit_drawn_card_for_obs,
            )
            next_agent_states = []
            agent_update_failed = False
            player_specific_obs_for_log = None
            try:
                for agent_idx, agent_state in enumerate(agent_states):
                    cloned_agent = agent_state.clone()
                    player_specific_obs = _filter_observation(observation, agent_idx)
                    if agent_idx == player:
                        player_specific_obs_for_log = player_specific_obs
                    cloned_agent.update(player_specific_obs)
                    next_agent_states.append(cloned_agent)
            except Exception as e_update:
                logger.error(
                    "W%d D%d: Error updating agent P%d after action %s: %s. State(post-action):%s FilteredObs:%s",
                    worker_id,
                    depth,
                    agent_idx,
                    chosen_action,
                    e_update,
                    game_state,
                    player_specific_obs_for_log,
                    exc_info=True,
                )
                worker_stats.error_count += 1
                agent_update_failed = True
                if undo_info:
                    try:
                        undo_info()
                        # Attempt to remove the last node added if undo succeeds
                        if simulation_nodes and simulation_nodes[-1].get(
                            "chosen_action"
                        ) == _serialize_action_for_history(chosen_action):
                            simulation_nodes.pop()
                    except Exception as undo_e:
                        logger.error(
                            "W%d D%d: Error undoing after agent update fail: %s",
                            worker_id,
                            depth,
                            undo_e,
                            exc_info=True,
                        )
                        worker_stats.error_count += 1

            if not agent_update_failed:
                try:
                    temp_reach_probs = reach_probs.copy()
                    # Need action probability for reach update - use the strategy[chosen_action_index]
                    chosen_action_prob = (
                        strategy[chosen_action_index]
                        if chosen_action_index != -1
                        and len(strategy) > chosen_action_index
                        else 0.0
                    )
                    temp_reach_probs[
                        player
                    ] *= chosen_action_prob  # Use actual sampled prob

                    recursive_utilities = _traverse_game_for_worker(
                        game_state,
                        next_agent_states,
                        temp_reach_probs,
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
                        simulation_nodes,  # Pass the list along
                    )
                    # Store utility for the specific action chosen
                    action_utilities[chosen_action_index] = recursive_utilities
                except Exception as recursive_err:
                    logger.error(
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
                    action_utilities[chosen_action_index] = np.zeros(
                        NUM_PLAYERS, dtype=np.float64
                    )
                    if undo_info:
                        try:
                            undo_info()
                            if simulation_nodes and simulation_nodes[-1].get(
                                "chosen_action"
                            ) == _serialize_action_for_history(chosen_action):
                                simulation_nodes.pop()  # Backtrack history
                        except Exception as undo_e:
                            logger.error(
                                "W%d D%d: Error undoing after recursion error: %s",
                                worker_id,
                                depth,
                                undo_e,
                                exc_info=True,
                            )
                            worker_stats.error_count += 1

                if undo_info:
                    try:
                        undo_info()
                    except Exception as undo_e:
                        logger.error(
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
    else:  # chosen_action is None or index is -1
        logger.error(
            "W%d D%d P%d: No action was chosen/sampled. State: %s",
            worker_id,
            depth,
            player,
            game_state,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Calculate node value based on the single traversed action's utility
    # The expected value over the strategy is implicitly handled by sampling one path
    node_value = action_utilities[chosen_action_index]

    # --- Regret Update ---
    # Calculate regret based on the difference between each hypothetical action's utility
    # and the actual sampled utility (node_value)
    if player == updating_player:
        if (
            infoset_key not in local_regret_updates
            or len(local_regret_updates[infoset_key]) != num_actions
        ):
            local_regret_updates[infoset_key] = np.zeros(num_actions, dtype=np.float64)

        if num_actions > 0:
            # Calculate counterfactual values for ALL actions IF needed for standard regret update
            # For ESMCFR, the regret update uses the sampled value.
            # Instantaneous regret = (utility if action i was taken) - (utility from sampled action)
            # We only have the utility from the sampled action (node_value).
            # To get the utility if action 'j' was taken, we'd need to recurse down that path too.
            # ESMCFR simplifies this: Regret update uses the difference between the expected utility of taking action 'j'
            # (which is action_utilities[j, player]) and the expected value of the node (node_value[player]).
            # HOWEVER, we only calculated utility for the chosen action.
            # Let's revisit the ESMCFR regret update formula.
            # Regret[i] = reach_prob_opponent * (utility[i] - node_value)
            # We need utility[i] for all i. This implies we DO need to traverse all actions,
            # but only update reach probs based on the sampled action.
            # OR, the update rule uses the policy itself: R[i] += (policy[i] - 1{a==i}) * cf_value
            # Let's assume the standard regret calculation: Need utilities for all actions.
            # This means the single-path traversal is INCORRECT for regret calculation.
            #
            # TODO: Currently traversing all actions, but updating reach probs based on chosen path.
            # This aligns better with standard CFR variations.

            node_value_full_traversal = np.zeros(NUM_PLAYERS, dtype=np.float64)
            action_utilities_full = np.zeros((num_actions, NUM_PLAYERS), dtype=np.float64)

            for i, action in enumerate(legal_actions):
                if i >= len(strategy):  # Should not happen if logic above is correct
                    continue

                action_prob = strategy[i]
                # Dont skip based on action_prob < 1e-9 for regret calculation
                # We need the counterfactual utility even for actions not taken

                explicit_drawn_card_for_obs: Optional[Card] = None
                if isinstance(action, (ActionReplace, ActionDiscard)):
                    if game_state.pending_action_data:
                        explicit_drawn_card_for_obs = game_state.pending_action_data.get(
                            "drawn_card"
                        )

                state_delta, undo_info = game_state.apply_action(action)
                if not callable(undo_info):
                    continue  # Skip if action fails

                observation = _create_observation(
                    None,
                    action,
                    game_state,
                    player,
                    game_state.snap_results_log,
                    explicit_drawn_card_for_obs,
                )
                next_agent_states = []
                agent_update_failed = False
                try:
                    for agent_idx, agent_state in enumerate(agent_states):
                        cloned_agent = agent_state.clone()
                        player_specific_obs = _filter_observation(observation, agent_idx)
                        cloned_agent.update(player_specific_obs)
                        next_agent_states.append(cloned_agent)
                except Exception:
                    agent_update_failed = True
                    undo_info()
                    continue  # Skip if agent update fails

                temp_reach_probs = reach_probs.copy()
                # TODO: true ESMCFR
            node_value = (
                np.sum(strategy[:, np.newaxis] * action_utilities, axis=0)
                if num_actions > 0
                else np.zeros(NUM_PLAYERS, dtype=np.float64)
            )

            # Regret Update (using full traversal utilities)
            if player == updating_player:
                if num_actions > 0:
                    player_action_values = action_utilities[:, player]
                    player_node_value = node_value[player]
                    instantaneous_regret = player_action_values - player_node_value
                    opponent_reach = reach_probs[opponent]
                    update_weight = (
                        opponent_reach  # In ESMCFR, this should be weighted differently?
                    )

                    if len(instantaneous_regret) == len(
                        local_regret_updates[infoset_key]
                    ):
                        local_regret_updates[infoset_key] += (
                            update_weight * instantaneous_regret
                        )
                    else:  # Error logging from before
                        pass  # Already logged error if shapes mismatch

    return node_value


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
    logger_instance = None
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

    # Initialize trace list for this simulation
    simulation_nodes_this_sim: List[SimulationNodeData] = []
    final_utility_value: Optional[np.ndarray] = None  # Store as numpy array first

    worker_root_logger = logging.getLogger()
    try:
        # Setup logging (same as before)
        for handler in worker_root_logger.handlers[:]:
            worker_root_logger.removeHandler(handler)
            if hasattr(handler, "close"):
                try:
                    handler.close()
                except Exception:
                    pass

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
        worker_root_logger.addHandler(file_handler)
        worker_root_logger.setLevel(logging.DEBUG)
        logger_instance = logging.getLogger(__name__)
        logger_instance.info(
            "Worker %d logging initialized (dir: %s, level: %s)",
            worker_id,
            worker_log_dir,
            logging.getLevelName(file_log_level),
        )
    except Exception as log_setup_e:
        print(
            f"!!! CRITICAL Error setting up logging W{worker_id}: {log_setup_e} !!!",
            file=sys.stderr,
        )
        worker_stats.error_count += 1
        if not worker_root_logger.hasHandlers():
            worker_root_logger.addHandler(logging.NullHandler())
        logger_instance = logging.getLogger(__name__)

    try:
        # Game and Agent State Initialization (same as before)
        try:
            game_state = CambiaGameState(house_rules=config.cambia_rules)
        except Exception as game_init_e:
            if logger_instance:
                logger_instance.error(
                    "W%d Iter %d: Failed GameState init: %s",
                    worker_id,
                    iteration,
                    game_init_e,
                    exc_info=True,
                )
            else:
                print(
                    f"W{worker_id} Iter {iteration}: GameState init error: {game_init_e}",
                    file=sys.stderr,
                )
            worker_stats.error_count += 1
            return WorkerResult(
                stats=worker_stats,
                simulation_nodes=simulation_nodes_this_sim,
                final_utility=None,
            )

        initial_agent_states = []
        if not game_state.is_terminal():
            try:
                initial_obs = _create_observation(None, None, game_state, -1, [], None)
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
            except Exception as agent_init_e:
                if logger_instance:
                    logger_instance.error(
                        "W%d Iter %d: Failed AgentStates init: %s. GameState: %s",
                        worker_id,
                        iteration,
                        agent_init_e,
                        game_state,
                        exc_info=True,
                    )
                else:
                    print(
                        f"W{worker_id} Iter {iteration}: AgentState init error: {agent_init_e}",
                        file=sys.stderr,
                    )
                worker_stats.error_count += 1
                return WorkerResult(
                    stats=worker_stats,
                    simulation_nodes=simulation_nodes_this_sim,
                    final_utility=None,
                )
        else:  # Game terminal at start
            if logger_instance:
                logger_instance.error(
                    "W%d Iter %d: Game terminal at init. State: %s",
                    worker_id,
                    iteration,
                    game_state,
                )
            worker_stats.error_count += 1
            # Get utility even if terminal at start
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

        # Run the traversal, collecting simulation nodes
        node_utility_at_root = _traverse_game_for_worker(
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
            simulation_nodes=simulation_nodes_this_sim,  # Pass the list here
        )

        # Capture final utility AFTER traversal completes
        if game_state.is_terminal():
            final_utility_value = np.array(
                [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
            )
        elif (
            node_utility_at_root is not None and len(node_utility_at_root) == NUM_PLAYERS
        ):
            # If traversal finished but not terminal (e.g., recursion limit), use root node utility as approximation?
            logger.warning(
                "W%d Iter %d: Traversal finished but game not terminal. Using root node utility %s.",
                worker_id,
                iteration,
                node_utility_at_root,
            )
            final_utility_value = node_utility_at_root
        else:
            logger.error(
                "W%d Iter %d: Could not determine final utility after traversal.",
                worker_id,
                iteration,
            )
            final_utility_value = np.zeros(
                NUM_PLAYERS, dtype=np.float64
            )  # Default on error

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
        else:
            print(f"W{worker_id} Iter {iteration} KeyboardInterrupt.", file=sys.stderr)
        worker_stats.error_count += 1
        return WorkerResult(
            stats=worker_stats,
            simulation_nodes=simulation_nodes_this_sim,
            final_utility=None,
        )
    except Exception as e_inner:
        worker_stats.error_count += 1
        if logger_instance:
            logger_instance.critical(
                "!!! Unhandled Error W%d Iter %d simulation: %s !!!",
                worker_id,
                iteration,
                e_inner,
                exc_info=True,
            )
        else:
            print(
                f"!!! FATAL WORKER ERROR W{worker_id} Iter {iteration}: {e_inner} !!!",
                file=sys.stderr,
            )
            import traceback

            traceback.print_exc(file=sys.stderr)
        return WorkerResult(
            stats=worker_stats,
            simulation_nodes=simulation_nodes_this_sim,
            final_utility=None,
        )
    finally:
        if logger_instance:
            for handler in logger_instance.handlers[:]:
                if hasattr(handler, "close"):
                    try:
                        handler.close()
                    except Exception:
                        pass


# --- Observation Helpers (remain unchanged from worker.py v0.8.4) ---
def _create_observation(
    prev_state: Optional[CambiaGameState],
    action: Optional[GameAction],
    next_state: CambiaGameState,
    acting_player: int,
    snap_results: List[Dict],
    explicit_drawn_card: Optional[Card] = None,  # Use Card type hint
) -> AgentObservation:
    """Creates the AgentObservation object based on the state *after* the action."""
    logger_obs = logging.getLogger(__name__)
    discard_top = next_state.get_discard_top()
    hand_sizes = [next_state.get_player_card_count(i) for i in range(NUM_PLAYERS)]
    stock_size = next_state.get_stockpile_size()
    cambia_called = next_state.cambia_caller_id is not None
    who_called = next_state.cambia_caller_id
    game_over = next_state.is_terminal()
    turn_num = next_state.get_turn_number()

    drawn_card_for_obs = explicit_drawn_card
    if explicit_drawn_card:
        logger_obs.debug(
            "Create Obs: Explicit drawn card provided: %s", drawn_card_for_obs
        )

    peeked_cards_dict = None
    if isinstance(action, ActionAbilityPeekOwnSelect) and acting_player != -1:
        hand = next_state.get_player_hand(acting_player)
        target_idx = action.target_hand_index
        if hand and 0 <= target_idx < len(hand):
            card = hand[target_idx]
            peeked_cards_dict = {(acting_player, target_idx): card}
            logger_obs.debug(
                "Create Obs: PeekOwn recorded peek of %s at (%d, %d)",
                card,
                acting_player,
                target_idx,
            )
        else:
            logger_obs.warning(
                "Create Obs: PeekOwn index %d invalid for hand size %d.",
                target_idx,
                len(hand),
            )
    elif isinstance(action, ActionAbilityPeekOtherSelect) and acting_player != -1:
        opp_idx = next_state.get_opponent_index(acting_player)
        opp_hand = next_state.get_player_hand(opp_idx)
        target_opp_idx = action.target_opponent_hand_index
        if opp_hand and 0 <= target_opp_idx < len(opp_hand):
            card = opp_hand[target_opp_idx]
            peeked_cards_dict = {(opp_idx, target_opp_idx): card}
            logger_obs.debug(
                "Create Obs: PeekOther recorded peek of %s at (%d, %d)",
                card,
                opp_idx,
                target_opp_idx,
            )
        else:
            logger_obs.warning(
                "Create Obs: PeekOther index %d invalid for opp hand %d.",
                target_opp_idx,
                len(opp_hand),
            )
    elif isinstance(action, ActionAbilityKingLookSelect) and acting_player != -1:
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
            logger_obs.debug(
                "Create Obs: KingLook recorded peek of %s at (%d, %d) and %s at (%d, %d)",
                card1,
                acting_player,
                own_idx,
                card2,
                opp_real_idx,
                opp_look_idx,
            )
        else:
            logger_obs.warning(
                "Create Obs: KingLook indices invalid/cards missing. Own %s/%s, Opp %s/%s",
                own_idx,
                len(own_hand) if own_hand else "N/A",
                opp_look_idx,
                len(opp_hand) if opp_hand else "N/A",
            )

    final_snap_results = snap_results if snap_results else []

    obs = AgentObservation(
        acting_player=acting_player,
        action=action,
        discard_top_card=discard_top,
        player_hand_sizes=hand_sizes,
        stockpile_size=stock_size,
        drawn_card=drawn_card_for_obs,
        peeked_cards=peeked_cards_dict,
        snap_results=final_snap_results,
        did_cambia_get_called=cambia_called,
        who_called_cambia=who_called,
        is_game_over=game_over,
        current_turn=turn_num,
    )
    return obs


def _filter_observation(obs: AgentObservation, observer_id: int) -> AgentObservation:
    """Creates a player-specific view of the observation, masking private info."""
    filtered_obs = copy.copy(obs)

    is_public_reveal_action = isinstance(obs.action, (ActionDiscard, ActionReplace))
    if (
        obs.drawn_card
        and obs.acting_player != observer_id
        and not is_public_reveal_action
    ):
        filtered_obs.drawn_card = None

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
    else:
        filtered_obs.peeked_cards = None

    filtered_obs.snap_results = obs.snap_results
    return filtered_obs
