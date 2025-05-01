# src/cfr/worker.py
"""
Functions executed by worker processes for parallel CFR training.
Based on External Sampling Monte Carlo CFR.
"""

import copy
import logging
import sys
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
    # PolicyDict, # No longer directly needed as snapshot is dict
    get_rm_plus_strategy,
    LocalRegretUpdateDict,
    LocalStrategyUpdateDict,
    LocalReachProbUpdateDict,
    WorkerResult,
)

# Logger setup will happen inside run_cfr_simulation_worker
logger = logging.getLogger(__name__)

# Type alias for the snapshot (regular dict)
RegretSnapshotDict: TypeAlias = Dict[InfosetKey, np.ndarray]


# Placeholder for the recursive traversal function, to be defined below
def _traverse_game_for_worker(
    game_state: CambiaGameState,
    agent_states: List[AgentState],
    reach_probs: np.ndarray,
    iteration: int,
    updating_player: int,
    weight: float,
    regret_sum_snapshot: RegretSnapshotDict,  # Snapshot is now a regular dict
    config: Config,  # Pass config for rules etc.
    # --- Local Accumulators ---
    local_regret_updates: LocalRegretUpdateDict,
    local_strategy_sum_updates: LocalStrategyUpdateDict,
    local_reach_prob_updates: LocalReachProbUpdateDict,
    depth: int,
) -> np.ndarray:  # Returns utility vector
    """Recursive traversal logic adapted for worker process."""
    # Implementation will adapt logic from CFRRecursionMixin._cfr_recursive
    # Key differences:
    # 1. Reads strategy from regret_sum_snapshot using get_rm_plus_strategy.
    # 2. Accumulates updates into local_* dictionaries instead of modifying shared state.
    # 3. Does not interact with tqdm status bars.
    # 4. No pruning logic needed for basic external sampling (can be added later if desired).
    # 5. Does not need access to self.analysis or log game history here.

    # --- Base Case: Terminal Node ---
    if game_state.is_terminal():
        return np.array(
            [game_state.get_utility(i) for i in range(NUM_PLAYERS)],
            dtype=np.float64,
        )

    # --- Determine Node Type and Context ---
    # (Same logic as in recursion_mixin)
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
            # Log locally if needed, main process handles global logging
            logger.warning(
                "Worker: Unknown pending action type (%s) when determining context.",
                type(pending).__name__,
                exc_info=True,
            )
            current_context = DecisionContext.START_TURN
    else:
        current_context = DecisionContext.START_TURN

    # --- Get Acting Player and Infoset ---
    player = game_state.get_acting_player()
    if player == -1:
        logger.error("Worker: Could not determine acting player at depth %d.", depth)
        return np.zeros(NUM_PLAYERS, dtype=np.float64)
    current_agent_state = agent_states[player]
    opponent = 1 - player
    try:
        if not hasattr(current_agent_state, "own_hand") or not hasattr(
            current_agent_state, "opponent_belief"
        ):
            logger.error(
                "Worker: Agent state P%d invalid for key generation at depth %d.",
                player,
                depth,
            )
            return np.zeros(NUM_PLAYERS, dtype=np.float64)
        base_infoset_tuple = current_agent_state.get_infoset_key()
        infoset_key = InfosetKey(*base_infoset_tuple, current_context.value)
    except Exception as e_key:
        logger.error(
            "Worker: Error getting infoset key for P%d at depth %d: %s",
            player,
            depth,
            e_key,
            exc_info=True,
        )
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # --- Get Legal Actions ---
    try:
        legal_actions_set = game_state.get_legal_actions()
        legal_actions = sorted(list(legal_actions_set), key=repr)
    except Exception as e_legal:
        logger.error(
            "Worker: Error getting legal actions for P%d at depth %d: %s",
            player,
            depth,
            e_legal,
            exc_info=True,
        )
        return np.zeros(NUM_PLAYERS, dtype=np.float64)
    num_actions = len(legal_actions)

    # --- Handle No Legal Actions ---
    if num_actions == 0:
        # logger.warning(...)
        # No need to force end check here, just return 0 utility if state isn't terminal
        if not game_state.is_terminal():
            logger.warning(
                "Worker: No legal actions for P%d at depth %d, non-terminal state.",
                player,
                depth,
            )
            return np.zeros(NUM_PLAYERS, dtype=np.float64)
        else:
            return np.array(
                [game_state.get_utility(i) for i in range(NUM_PLAYERS)],
                dtype=np.float64,
            )

    # --- Get Current Strategy from Regret Snapshot ---
    # Snapshot is now a dict, use .get() with a default value (None).
    current_regrets: Optional[np.ndarray] = regret_sum_snapshot.get(infoset_key)

    if current_regrets is None or len(current_regrets) != num_actions:
        if current_regrets is not None:  # Log dimension mismatch
            logger.warning(
                "Worker: Regret dim mismatch for key %s. Snapshot: %d, Need: %d. Using uniform.",
                infoset_key,
                len(current_regrets),
                num_actions,
            )
        # If infoset is new or mismatched in snapshot, use uniform strategy for traversal
        strategy = np.ones(num_actions) / num_actions if num_actions > 0 else np.array([])
        # Initialize local updates arrays if needed, assuming they might be needed later
        if infoset_key not in local_regret_updates:
            local_regret_updates[infoset_key] = np.zeros(num_actions, dtype=np.float64)
        if infoset_key not in local_strategy_sum_updates:
            local_strategy_sum_updates[infoset_key] = np.zeros(
                num_actions, dtype=np.float64
            )
        # local_reach_prob_updates is a float, defaultdict handles it
    else:
        strategy = get_rm_plus_strategy(current_regrets)

    # --- External Sampling: Only update the specific player for this simulation ---
    # We traverse according to the fixed strategy profile `strategy` for *all* players.
    # However, regret/strategy sum updates are only accumulated for the designated `updating_player`.
    player_reach = reach_probs[player]
    opponent_reach = reach_probs[opponent]

    # --- Accumulate Strategy Sum Update (if current node belongs to updating_player) ---
    if player == updating_player and weight > 0 and player_reach > 1e-9:
        # Ensure local arrays are initialized with correct size
        if (
            infoset_key not in local_strategy_sum_updates
            or len(local_strategy_sum_updates[infoset_key]) != num_actions
        ):
            # Re-initialize if dimensions changed between reading snapshot and accumulating here
            if (
                infoset_key in local_strategy_sum_updates
                and len(local_strategy_sum_updates[infoset_key]) != num_actions
            ):
                logger.warning(
                    "Worker: StratSum dim mismatch for key %s during accumulation. Have: %d, Need: %d. Re-initializing.",
                    infoset_key,
                    len(local_strategy_sum_updates[infoset_key]),
                    num_actions,
                )
            local_strategy_sum_updates[infoset_key] = np.zeros(
                num_actions, dtype=np.float64
            )

        # Ensure strategy has the same length (should match num_actions)
        if len(strategy) == num_actions:
            local_strategy_sum_updates[infoset_key] += weight * player_reach * strategy
        else:
            logger.error(
                "Worker: Strategy length (%d) mismatch with num_actions (%d) for key %s. Skipping strategy update.",
                len(strategy),
                num_actions,
                infoset_key,
            )

        local_reach_prob_updates[infoset_key] += weight * player_reach

    # --- Iterate Through Actions ---
    action_utilities = np.zeros((num_actions, NUM_PLAYERS), dtype=np.float64)
    node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

    for i, action in enumerate(legal_actions):
        if i >= len(strategy):  # Safety check if strategy length somehow mismatched
            logger.error(
                "Worker: Action index %d out of bounds for strategy length %d. Breaking loop.",
                i,
                len(strategy),
            )
            break
        action_prob = strategy[i]
        if action_prob < 1e-9:  # Skip negligible probability branches
            continue

        # --- Apply Action and Recurse ---
        state_delta: Optional[StateDelta] = None
        undo_info: Optional[UndoInfo] = None
        drawn_card_this_step: Optional[CardObject] = None
        try:
            # Store drawn card *before* applying action if applicable
            if isinstance(
                action, (ActionReplace, ActionDiscard)
            ) and game_state.pending_action_data.get("drawn_card"):
                drawn_card_this_step = game_state.pending_action_data["drawn_card"]
            state_delta, undo_info = game_state.apply_action(action)
        except Exception as apply_err:
            logger.error(
                "Worker: Error applying action %s for P%d at depth %d: %s",
                action,
                player,
                depth,
                apply_err,
                exc_info=True,
            )
            # In worker, error applying action should likely stop this simulation branch
            continue  # Skip to next action

        # Create observation and update agent states (local copies)
        # Use helper functions similar to _create_observation and _filter_observation
        # These helpers need to be available or reimplemented in this scope
        # observation = _create_observation_for_worker(...) # Needs implementation/import
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
                    "Worker: Error updating agent state %d after action %s at depth %d: %s",
                    agent_idx,
                    action,
                    depth,
                    e_update,
                    exc_info=True,
                )
                agent_update_failed = True
                break
            next_agent_states.append(cloned_agent)

        if agent_update_failed:
            if undo_info:
                try:
                    undo_info()
                except Exception as undo_e:
                    logger.error(
                        "Worker: Error undoing action after agent update failure: %s",
                        undo_e,
                        exc_info=True,
                    )
            continue  # Skip to next action

        # Calculate next reach probabilities
        next_reach_probs = reach_probs.copy()
        # The player acting used their reach probability *weighted by the action prob*
        # The opponent's reach remains the same for this step down.
        # CFR reach prob calculation: pi(h) = product of player probabilities along path h
        # pi(ha) = pi(h) * sigma(I(h), a)
        # For ESMCFR, we need reach for both players to correctly weight regret.
        # Reach_probs passed down should represent pi_i(h) and pi_-i(h)
        # Let's adjust reach prob calculation based on standard CFR update:
        temp_reach_probs = reach_probs.copy()
        temp_reach_probs[
            player
        ] *= action_prob  # Update reach for the acting player based on action taken

        # Recursive call
        try:
            recursive_utilities = _traverse_game_for_worker(
                game_state,
                next_agent_states,
                temp_reach_probs,  # Pass the updated reach probs
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
                "Worker: Error in recursive call after action %s at depth %d: %s",
                action,
                depth,
                recursive_err,
                exc_info=True,
            )
            action_utilities[i] = np.zeros(
                NUM_PLAYERS, dtype=np.float64
            )  # Assign 0 utility on error

        # Undo action
        if undo_info:
            try:
                undo_info()
            except Exception as undo_e:
                logger.error(
                    "Worker: Error undoing action %s at depth %d: %s. State may be corrupt.",
                    action,
                    depth,
                    undo_e,
                    exc_info=True,
                )
                # If undo fails, the state is corrupt for this worker, probably stop
                return np.zeros(NUM_PLAYERS, dtype=np.float64)
        else:
            logger.error(
                "Worker: Missing undo info for action %s at depth %d. State corrupt.",
                action,
                depth,
            )
            return np.zeros(NUM_PLAYERS, dtype=np.float64)  # State corrupt

    # --- Calculate Node Value & Accumulate Regret Update ---
    node_value = np.sum(strategy[:, np.newaxis] * action_utilities, axis=0)

    # Accumulate regret update (if current node belongs to updating_player)
    if player == updating_player:
        # Ensure local regret array is initialized and has correct size
        if (
            infoset_key not in local_regret_updates
            or len(local_regret_updates[infoset_key]) != num_actions
        ):
            if (
                infoset_key in local_regret_updates
                and len(local_regret_updates[infoset_key]) != num_actions
            ):
                logger.warning(
                    "Worker: Regret dim mismatch for key %s during accumulation. Have: %d, Need: %d. Re-initializing.",
                    infoset_key,
                    len(local_regret_updates[infoset_key]),
                    num_actions,
                )
            local_regret_updates[infoset_key] = np.zeros(num_actions, dtype=np.float64)

        player_action_values = action_utilities[:, player]
        player_node_value = node_value[player]
        instantaneous_regret = player_action_values - player_node_value

        # Weight regret update by opponent's reach probability
        # NOTE: In External Sampling, we weight by the reach prob of the SAMPLER,
        # which corresponds to the reach prob of players *not* being updated.
        # Here, `opponent_reach` is the reach prob pi_-i(h) passed down.
        update_weight = opponent_reach  # Correct for ESMCFR
        local_regret_updates[infoset_key] += update_weight * instantaneous_regret

    return node_value


def _setup_worker_logging(log_level: int, worker_id: int):
    """Configures basic logging for a worker process."""
    # Use a unique format for worker logs
    log_format = (
        f"[Worker {worker_id}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    formatter = logging.Formatter(log_format)

    # Get the root logger
    worker_logger = logging.getLogger()

    # Remove any handlers potentially inherited? (Shouldn't happen with multiprocessing)
    for handler in worker_logger.handlers[:]:
        worker_logger.removeHandler(handler)

    # Add a StreamHandler to stderr
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    worker_logger.addHandler(stream_handler)

    # Set the logger level
    worker_logger.setLevel(log_level)
    # Optionally suppress logs from libraries within worker if needed
    # logging.getLogger("some_library").setLevel(logging.WARNING)


def run_cfr_simulation_worker(
    worker_args: Tuple[
        int, Config, RegretSnapshotDict, int, int
    ],  # Added log_level, worker_id
) -> Optional[WorkerResult]:
    """
    Top-level function executed by each worker process.
    Initializes game state, agent states, and runs one simulation traversal.

    Args:
        worker_args: A tuple containing (iteration, config, regret_sum_snapshot, log_level, worker_id).

    Returns:
        A WorkerResult tuple containing the locally accumulated update dictionaries,
        or None if the simulation failed critically.
    """
    iteration, config, regret_sum_snapshot, log_level, worker_id = worker_args

    # --- Setup Logging for this worker ---
    _setup_worker_logging(log_level, worker_id)

    try:
        logger.debug("Worker %d starting iteration %d.", worker_id, iteration)

        # --- Initialize Game and Agent States ---
        game_state = CambiaGameState(house_rules=config.cambia_rules)
        initial_agent_states = []
        initial_hands_for_log = [list(p.hand) for p in game_state.players]
        initial_peeks_for_log = [p.initial_peek_indices for p in game_state.players]

        if not game_state.is_terminal():
            initial_obs = _create_observation(  # Use local helper
                None, None, game_state, -1, [], None
            )
            for i in range(NUM_PLAYERS):
                agent = AgentState(
                    player_id=i,
                    opponent_id=game_state.get_opponent_index(i),
                    memory_level=config.agent_params.memory_level,
                    time_decay_turns=config.agent_params.time_decay_turns,
                    initial_hand_size=len(initial_hands_for_log[i]),
                    config=config,
                )
                # Use initialize method of AgentState
                agent.initialize(
                    initial_obs,
                    initial_hands_for_log[i],
                    initial_peeks_for_log[i],
                )
                initial_agent_states.append(agent)
        else:
            logger.error("Game terminal at init.")
            return None  # Critical failure

        if not initial_agent_states:
            logger.error("Failed init agent states.")
            return None  # Critical failure

        # --- Determine Updating Player & Iteration Weight ---
        # In standard ESMCFR, updates for all players are gathered in each traversal.
        # The original spec mentioned alternating updates, which is slightly different.
        # Let's follow the spec: update only one player per iteration.
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
            reach_probs=np.ones(NUM_PLAYERS, dtype=np.float64),  # Initial reach
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

        # Return the accumulated local updates
        # Convert defaultdicts back to regular dicts for pickling consistency if needed
        # (Defaultdicts are generally pickleable, but explicit conversion is safer)
        logger.debug("Worker %d finished iteration %d.", worker_id, iteration)
        return (
            dict(local_regret_updates),
            dict(local_strategy_sum_updates),
            dict(local_reach_prob_updates),
        )

    except Exception as e:
        # Log the exception traceback from within the worker
        # Logger is configured now
        logger.error(
            "Error in worker process during iteration %d: %s", iteration, e, exc_info=True
        )
        # Optionally return None or re-raise to signal failure to the main process
        return None


# --- Helper functions needed by _traverse_game_for_worker ---
# These are adapted from CFRRecursionMixin or AgentState
# We place them here to avoid complex dependencies or passing instance methods


def _create_observation(
    prev_state: Optional[CambiaGameState],  # Not used currently
    action: Optional[GameAction],
    next_state: CambiaGameState,
    acting_player: int,
    snap_results: List[Dict],
    explicit_drawn_card: Optional[CardObject] = None,
) -> AgentObservation:
    """Creates the AgentObservation object based on the state *after* the action."""
    # --- Reimplementation of CFRRecursionMixin._create_observation ---
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
    # --- Reimplementation of CFRRecursionMixin._filter_observation ---
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
