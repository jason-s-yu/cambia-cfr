# src/cfr/worker.py
"""
Functions executed by worker processes for parallel CFR training.
Based on External Sampling Monte Carlo CFR.
"""

import copy
import logging
import logging.handlers
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, TypeAlias

import numpy as np

# Relative imports from parent directories
from ..agent_state import AgentState, AgentObservation
from ..config import Config
from ..constants import (
    NUM_PLAYERS,
    DecisionContext,
    GameAction,
    ActionDiscard,
    ActionReplace,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionSnapOpponentMove,
    CardObject,
)
from ..game.engine import CambiaGameState
from ..game.types import StateDelta, UndoInfo
from ..utils import (
    InfosetKey,
    get_rm_plus_strategy,
    LocalRegretUpdateDict,
    LocalStrategyUpdateDict,
    LocalReachProbUpdateDict,
    WorkerResult,
    LogQueue,
)

# Logger setup is now simpler - just get logger. Configuration happens in main.
logger = logging.getLogger(__name__)

RegretSnapshotDict: TypeAlias = Dict[InfosetKey, np.ndarray]


# --- Traversal Logic (largely unchanged, ensure logging uses logger.) ---
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
) -> np.ndarray:
    """Recursive traversal logic adapted for worker process."""

    # Base Case
    if game_state.is_terminal():
        return np.array(
            [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
        )

    # Determine Context (ensure logger calls work)
    if game_state.snap_phase_active:
        current_context = DecisionContext.SNAP_DECISION
    # ... (rest of context logic unchanged, but ensure logger calls use logger.)
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
                "Worker: Unknown pending action type (%s) for context.",
                type(pending).__name__,
                exc_info=True,
            )  # Use logger
            current_context = DecisionContext.START_TURN
    else:
        current_context = DecisionContext.START_TURN

    # Get Acting Player and Infoset Key (ensure logger calls work)
    player = game_state.get_acting_player()
    if player == -1:
        logger.error(
            "Worker: Could not determine acting player depth %d.", depth
        )  # Use logger
        return np.zeros(NUM_PLAYERS, dtype=np.float64)
    current_agent_state = agent_states[player]
    opponent = 1 - player
    try:
        # Basic validation moved here for clarity
        if not hasattr(current_agent_state, "own_hand") or not hasattr(
            current_agent_state, "opponent_belief"
        ):
            logger.error(
                "Worker: Agent state P%d invalid for key gen depth %d.", player, depth
            )  # Use logger
            return np.zeros(NUM_PLAYERS, dtype=np.float64)
        base_infoset_tuple = current_agent_state.get_infoset_key()
        infoset_key = InfosetKey(*base_infoset_tuple, current_context.value)
    except Exception as e_key:
        logger.error(
            "Worker: Error getting infoset key P%d depth %d: %s",
            player,
            depth,
            e_key,
            exc_info=True,
        )  # Use logger
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Get Legal Actions (ensure logger calls work)
    try:
        legal_actions_set = game_state.get_legal_actions()
        legal_actions = sorted(list(legal_actions_set), key=repr)
    except Exception as e_legal:
        logger.error(
            "Worker: Error getting legal actions P%d depth %d: %s",
            player,
            depth,
            e_legal,
            exc_info=True,
        )  # Use logger
        return np.zeros(NUM_PLAYERS, dtype=np.float64)
    num_actions = len(legal_actions)

    # Handle No Legal Actions (ensure logger calls work)
    if num_actions == 0:
        if not game_state.is_terminal():
            logger.warning(
                "Worker: No legal actions P%d depth %d, non-terminal.", player, depth
            )  # Use logger
            return np.zeros(NUM_PLAYERS, dtype=np.float64)
        else:
            return np.array(
                [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
            )

    # Get Strategy from Snapshot (ensure logger calls work)
    current_regrets: Optional[np.ndarray] = regret_sum_snapshot.get(infoset_key)
    if current_regrets is None or len(current_regrets) != num_actions:
        if current_regrets is not None:
            logger.warning(
                "Worker: Regret dim mismatch key %s. Snap:%d Need:%d. Using uniform.",
                infoset_key,
                len(current_regrets),
                num_actions,
            )  # Use logger
        strategy = np.ones(num_actions) / num_actions if num_actions > 0 else np.array([])
        if infoset_key not in local_regret_updates:
            local_regret_updates[infoset_key] = np.zeros(num_actions, dtype=np.float64)
        if infoset_key not in local_strategy_sum_updates:
            local_strategy_sum_updates[infoset_key] = np.zeros(
                num_actions, dtype=np.float64
            )
    else:
        strategy = get_rm_plus_strategy(current_regrets)

    player_reach = reach_probs[player]
    opponent_reach = reach_probs[opponent]

    # Accumulate Strategy Sum Update (ensure logger calls work)
    if player == updating_player and weight > 0 and player_reach > 1e-9:
        if (
            infoset_key not in local_strategy_sum_updates
            or len(local_strategy_sum_updates[infoset_key]) != num_actions
        ):
            if infoset_key in local_strategy_sum_updates:  # Log only if re-initializing
                logger.warning(
                    "Worker: StratSum dim mismatch key %s accum. Have:%d Need:%d. Re-init.",
                    infoset_key,
                    len(local_strategy_sum_updates[infoset_key]),
                    num_actions,
                )  # Use logger
            local_strategy_sum_updates[infoset_key] = np.zeros(
                num_actions, dtype=np.float64
            )
        if len(strategy) == num_actions:
            local_strategy_sum_updates[infoset_key] += weight * player_reach * strategy
        else:
            logger.error(
                "Worker: Strategy len %d mismatch num_actions %d key %s. Skip strat update.",
                len(strategy),
                num_actions,
                infoset_key,
            )  # Use logger
        local_reach_prob_updates[infoset_key] += weight * player_reach

    # Iterate Through Actions (ensure logger calls work)
    action_utilities = np.zeros((num_actions, NUM_PLAYERS), dtype=np.float64)
    node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

    for i, action in enumerate(legal_actions):
        if i >= len(strategy):
            logger.error(
                "Worker: Action index %d OOB strategy len %d.", i, len(strategy)
            )  # Use logger
            break
        action_prob = strategy[i]
        if action_prob < 1e-9:
            continue

        # Apply Action (ensure logger calls work)
        state_delta: Optional[StateDelta] = None
        undo_info: Optional[UndoInfo] = None
        drawn_card_this_step: Optional[CardObject] = None
        try:
            if isinstance(
                action, (ActionReplace, ActionDiscard)
            ) and game_state.pending_action_data.get("drawn_card"):
                drawn_card_this_step = game_state.pending_action_data["drawn_card"]
            state_delta, undo_info = game_state.apply_action(action)
        except Exception as apply_err:
            logger.error(
                "Worker: Error applying action %s P%d depth %d: %s",
                action,
                player,
                depth,
                apply_err,
                exc_info=True,
            )  # Use logger
            continue

        # Create observation and update agent states (ensure logger calls work)
        observation = _create_observation(
            None,
            action,
            game_state,
            player,
            game_state.snap_results_log,
            drawn_card_this_step,
        )
        next_agent_states = []
        agent_update_failed = False
        for agent_idx, agent_state in enumerate(agent_states):
            cloned_agent = agent_state.clone()
            try:
                player_specific_obs = _filter_observation(observation, agent_idx)
                cloned_agent.update(player_specific_obs)
            except Exception as e_update:
                logger.error(
                    "Worker: Error updating agent state %d action %s depth %d: %s",
                    agent_idx,
                    action,
                    depth,
                    e_update,
                    exc_info=True,
                )  # Use logger
                agent_update_failed = True
                break
            next_agent_states.append(cloned_agent)

        if agent_update_failed:
            if undo_info:
                try:
                    undo_info()
                except Exception as undo_e:
                    logger.error(
                        "Worker: Error undoing after agent update fail: %s",
                        undo_e,
                        exc_info=True,
                    )  # Use logger
            continue

        # Calculate next reach probabilities
        temp_reach_probs = reach_probs.copy()
        temp_reach_probs[player] *= action_prob

        # Recursive call (ensure logger calls work)
        try:
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
            )
            action_utilities[i] = recursive_utilities
        except Exception as recursive_err:
            logger.error(
                "Worker: Error in recursive call action %s depth %d: %s",
                action,
                depth,
                recursive_err,
                exc_info=True,
            )  # Use logger
            action_utilities[i] = np.zeros(NUM_PLAYERS, dtype=np.float64)

        # Undo action (ensure logger calls work)
        if undo_info:
            try:
                undo_info()
            except Exception as undo_e:
                logger.error(
                    "Worker: Error undoing action %s depth %d: %s. State corrupt.",
                    action,
                    depth,
                    undo_e,
                    exc_info=True,
                )  # Use logger
                return np.zeros(NUM_PLAYERS, dtype=np.float64)
        else:
            logger.error(
                "Worker: Missing undo info action %s depth %d. State corrupt.",
                action,
                depth,
            )  # Use logger
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Calculate Node Value & Accumulate Regret Update (ensure logger calls work)
    node_value = np.sum(strategy[:, np.newaxis] * action_utilities, axis=0)
    if player == updating_player:
        if (
            infoset_key not in local_regret_updates
            or len(local_regret_updates[infoset_key]) != num_actions
        ):
            if infoset_key in local_regret_updates:  # Log only if re-initializing
                logger.warning(
                    "Worker: Regret dim mismatch key %s accum. Have:%d Need:%d. Re-init.",
                    infoset_key,
                    len(local_regret_updates[infoset_key]),
                    num_actions,
                )  # Use logger
            local_regret_updates[infoset_key] = np.zeros(num_actions, dtype=np.float64)
        player_action_values = action_utilities[:, player]
        player_node_value = node_value[player]
        instantaneous_regret = player_action_values - player_node_value
        update_weight = opponent_reach
        local_regret_updates[infoset_key] += update_weight * instantaneous_regret

    return node_value


def run_cfr_simulation_worker(
    worker_args: Tuple[
        int,
        Config,
        RegretSnapshotDict,
        Optional[LogQueue],
        int,  # Pass LogQueue, worker_id
    ],
) -> Optional[WorkerResult]:
    """Top-level function executed by each worker process."""
    iteration, config, regret_sum_snapshot, log_queue, worker_id = worker_args

    # --- Setup Logging for this worker using QueueHandler ---
    if log_queue:
        worker_root_logger = logging.getLogger()
        # Remove any handlers that might be inherited (shouldn't be needed with spawn/forkserver)
        for handler in worker_root_logger.handlers[:]:
            worker_root_logger.removeHandler(handler)
        # Add the queue handler
        queue_handler = logging.handlers.QueueHandler(log_queue)
        worker_root_logger.addHandler(queue_handler)
        # Set worker logger level to DEBUG to send everything to the queue
        worker_root_logger.setLevel(logging.DEBUG)
        # Optionally add a NullHandler to prevent "No handlers found" warnings
        # worker_root_logger.addHandler(logging.NullHandler())
        logger.info(
            "Worker %d logging initialized via QueueHandler.", worker_id
        )  # This goes to queue
    else:
        # Fallback or error handling if queue is not provided
        # For now, just disable logging effectively in the worker if no queue
        logging.getLogger().addHandler(logging.NullHandler())
        # print(f"[Worker {worker_id}] Error: Log queue not provided.", file=sys.stderr)

    try:
        logger.debug(
            "Worker %d starting iteration %d.", worker_id, iteration
        )  # Use logger

        # --- Initialize Game and Agent States ---
        game_state = CambiaGameState(house_rules=config.cambia_rules)
        initial_agent_states = []
        initial_hands_for_log = [list(p.hand) for p in game_state.players]
        initial_peeks_for_log = [p.initial_peek_indices for p in game_state.players]

        if not game_state.is_terminal():
            initial_obs = _create_observation(None, None, game_state, -1, [], None)
            for i in range(NUM_PLAYERS):
                agent = AgentState(
                    player_id=i,
                    opponent_id=game_state.get_opponent_index(i),
                    memory_level=config.agent_params.memory_level,
                    time_decay_turns=config.agent_params.time_decay_turns,
                    initial_hand_size=len(initial_hands_for_log[i]),
                    config=config,
                )
                agent.initialize(
                    initial_obs, initial_hands_for_log[i], initial_peeks_for_log[i]
                )
                initial_agent_states.append(agent)
        else:
            logger.error("Worker %d: Game terminal at init.", worker_id)  # Use logger
            return None

        if not initial_agent_states:
            logger.error("Worker %d: Failed init agent states.", worker_id)  # Use logger
            return None

        # --- Determine Updating Player & Iteration Weight ---
        updating_player = iteration % NUM_PLAYERS
        if config.cfr_plus_params.weighted_averaging_enabled:
            delay = config.cfr_plus_params.averaging_delay
            weight = float(max(0, iteration - delay))
        else:
            weight = 1.0

        # --- Initialize Local Update Dictionaries ---
        local_regret_updates: LocalRegretUpdateDict = defaultdict(
            lambda: np.array([], dtype=np.float64)
        )
        local_strategy_sum_updates: LocalStrategyUpdateDict = defaultdict(
            lambda: np.array([], dtype=np.float64)
        )
        local_reach_prob_updates: LocalReachProbUpdateDict = defaultdict(float)

        # --- Run Traversal ---
        _traverse_game_for_worker(
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
        )

        logger.debug(
            "Worker %d finished iteration %d.", worker_id, iteration
        )  # Use logger
        return (
            dict(local_regret_updates),
            dict(local_strategy_sum_updates),
            dict(local_reach_prob_updates),
        )

    except Exception as e:
        # Log exception using the configured queue handler
        logger.error(
            "Error in worker %d iter %d: %s", worker_id, iteration, e, exc_info=True
        )  # Use logger
        return None


# --- Helper functions needed by _traverse_game_for_worker ---
# (Implementation unchanged from v0.7.1, copied below for completeness)
def _create_observation(
    prev_state: Optional[CambiaGameState],
    action: Optional[GameAction],
    next_state: CambiaGameState,
    acting_player: int,
    snap_results: List[Dict],
    explicit_drawn_card: Optional[CardObject] = None,
) -> AgentObservation:
    """Creates the AgentObservation object based on the state *after* the action."""
    discard_top = next_state.get_discard_top()
    hand_sizes = [next_state.get_player_card_count(i) for i in range(NUM_PLAYERS)]
    stock_size = next_state.get_stockpile_size()
    cambia_called = next_state.cambia_caller_id is not None
    who_called = next_state.cambia_caller_id
    game_over = next_state.is_terminal()
    turn_num = next_state.get_turn_number()

    drawn_card_for_obs = None
    if isinstance(action, (ActionDiscard, ActionReplace)):
        drawn_card_for_obs = explicit_drawn_card
    elif explicit_drawn_card is not None and acting_player != -1:
        drawn_card_for_obs = explicit_drawn_card

    peeked_cards_dict = None
    if isinstance(action, ActionAbilityPeekOwnSelect) and acting_player != -1:
        hand = next_state.get_player_hand(acting_player)
        target_idx = action.target_hand_index
        if hand and 0 <= target_idx < len(hand):
            peeked_cards_dict = {(acting_player, target_idx): hand[target_idx]}
    elif isinstance(action, ActionAbilityPeekOtherSelect) and acting_player != -1:
        opp_idx = next_state.get_opponent_index(acting_player)
        opp_hand = next_state.get_player_hand(opp_idx)
        target_opp_idx = action.target_opponent_hand_index
        if opp_hand and 0 <= target_opp_idx < len(opp_hand):
            peeked_cards_dict = {(opp_idx, target_opp_idx): opp_hand[target_opp_idx]}
    elif isinstance(action, ActionAbilityKingLookSelect) and acting_player != -1:
        own_idx, opp_look_idx = action.own_hand_index, action.opponent_hand_index
        opp_real_idx = next_state.get_opponent_index(acting_player)
        own_hand, opp_hand = next_state.get_player_hand(
            acting_player
        ), next_state.get_player_hand(opp_real_idx)
        card1, card2 = (
            own_hand[own_idx] if own_hand and 0 <= own_idx < len(own_hand) else None
        ), (
            opp_hand[opp_look_idx]
            if opp_hand and 0 <= opp_look_idx < len(opp_hand)
            else None
        )
        if card1 and card2:
            peeked_cards_dict = {
                (acting_player, own_idx): card1,
                (opp_real_idx, opp_look_idx): card2,
            }

    final_snap_results = snap_results
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
    """Creates a player-specific view of the observation."""
    filtered_obs = copy.copy(obs)
    if obs.drawn_card and obs.acting_player != observer_id:
        if not isinstance(obs.action, (ActionDiscard, ActionReplace)):
            filtered_obs.drawn_card = None
    if obs.peeked_cards:
        new_peeked = {}
        if (
            isinstance(obs.action, ActionAbilityPeekOwnSelect)
            and obs.acting_player == observer_id
        ):
            new_peeked = obs.peeked_cards
        elif (
            isinstance(obs.action, ActionAbilityPeekOtherSelect)
            and obs.acting_player == observer_id
        ):
            new_peeked = obs.peeked_cards
        elif (
            isinstance(obs.action, ActionAbilityKingLookSelect)
            and obs.acting_player == observer_id
        ):
            new_peeked = obs.peeked_cards
        filtered_obs.peeked_cards = new_peeked if new_peeked else None
    else:
        filtered_obs.peeked_cards = None
    filtered_obs.snap_results = obs.snap_results
    return filtered_obs
    