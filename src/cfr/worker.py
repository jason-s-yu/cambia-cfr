# src/cfr/worker.py
"""
Functions executed by worker processes for parallel CFR training.
Based on External Sampling Monte Carlo CFR.
"""

import copy
import logging
import logging.handlers
from collections import defaultdict
import sys
from typing import Dict, List, Optional, Tuple, TypeAlias
import queue  # For queue.Empty, queue.Full exceptions

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
    LogQueue as GenericQueue,  # Rename for clarity
    WorkerStats,
)

# Logger setup is now simpler - just get logger. Configuration happens in main.
# Ensure logger is configured *after* queue handler is added in run_cfr_simulation_worker
# logger = logging.getLogger(__name__) # Get logger inside the function after setup

RegretSnapshotDict: TypeAlias = Dict[InfosetKey, np.ndarray]
ProgressQueue: TypeAlias = GenericQueue  # Use same type alias for progress queue
PROGRESS_UPDATE_NODE_INTERVAL = 1000  # Send update every N nodes


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
    worker_stats: WorkerStats,
    progress_queue: Optional[ProgressQueue],
    worker_id: int,
) -> np.ndarray:
    """Recursive traversal logic adapted for worker process."""
    logger = logging.getLogger(__name__)  # Get logger instance for this function scope

    # Update stats
    worker_stats.nodes_visited += 1
    worker_stats.max_depth = max(worker_stats.max_depth, depth)

    # --- Send Progress Update Periodically ---
    if progress_queue and (
        worker_stats.nodes_visited % PROGRESS_UPDATE_NODE_INTERVAL == 0
    ):
        try:
            progress_update = (
                worker_id,
                worker_stats.max_depth,
                worker_stats.nodes_visited,
            )
            progress_queue.put_nowait(progress_update)
        except queue.Full:
            # If the queue is full, just skip the update for now
            # logger.debug("Progress queue full for worker %d, skipping update.", worker_id)
            pass
        except Exception as pq_e:
            # Log error but don't crash the worker
            logger.error(
                "Error putting progress update on queue for worker %d: %s",
                worker_id,
                pq_e,
            )

    # Base Case
    if game_state.is_terminal():
        return np.array(
            [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
        )

    # Check depth limit (prevent infinite recursion)
    if depth >= config.system.recursion_limit:
        logger.error(
            "Worker %d: Max recursion depth (%d) reached. Returning 0.", worker_id, depth
        )
        # Treat as terminal with neutral utility
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Determine Context (ensure logger calls work)
    if game_state.snap_phase_active:
        current_context = DecisionContext.SNAP_DECISION
    # ... (rest of context logic unchanged, but ensure logger calls use logger.)
    elif game_state.pending_action:
        pending = game_state.pending_action
        if isinstance(pending, ActionDiscard):  # After draw, choosing discard/replace
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
        ):  # Choosing target/decision for ability
            current_context = DecisionContext.ABILITY_SELECT
        elif isinstance(
            pending, ActionSnapOpponentMove
        ):  # Choosing card to move after SnapOpponent
            current_context = DecisionContext.SNAP_MOVE
        else:
            logger.warning(
                "Worker %d: Unknown pending action type (%s) for context.",
                worker_id,
                type(pending).__name__,
            )
            current_context = DecisionContext.START_TURN  # Fallback
    else:  # Normal start of turn
        current_context = DecisionContext.START_TURN

    # Get Acting Player and Infoset Key (ensure logger calls work)
    player = game_state.get_acting_player()
    if player == -1:
        logger.error(
            "Worker %d: Could not determine acting player depth %d. State: %s",
            worker_id,
            depth,
            game_state,
        )
        return np.zeros(NUM_PLAYERS, dtype=np.float64)
    current_agent_state = agent_states[player]
    opponent = 1 - player
    try:
        # Basic validation moved here for clarity
        if not hasattr(current_agent_state, "own_hand") or not hasattr(
            current_agent_state, "opponent_belief"
        ):
            logger.error(
                "Worker %d: Agent state P%d invalid for key gen depth %d. State: %s",
                worker_id,
                player,
                depth,
                current_agent_state,
            )
            return np.zeros(NUM_PLAYERS, dtype=np.float64)
        base_infoset_tuple = current_agent_state.get_infoset_key()
        # Ensure context is valid Enum member before accessing value
        if not isinstance(current_context, DecisionContext):
            logger.error(
                "Worker %d: Invalid context type '%s' for infoset key.",
                worker_id,
                type(current_context),
            )
            return np.zeros(NUM_PLAYERS, dtype=np.float64)
        infoset_key = InfosetKey(*base_infoset_tuple, current_context.value)
    except Exception as e_key:
        logger.error(
            "Worker %d: Error getting infoset key P%d depth %d: %s. State: %s, Context: %s",
            worker_id,
            player,
            depth,
            e_key,
            current_agent_state,
            (
                current_context.name
                if isinstance(current_context, DecisionContext)
                else current_context
            ),
            exc_info=True,
        )
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Get Legal Actions (ensure logger calls work)
    try:
        legal_actions_set = game_state.get_legal_actions()
        # Sort for deterministic iteration order (important for strategy array alignment)
        legal_actions = sorted(list(legal_actions_set), key=repr)
    except Exception as e_legal:
        logger.error(
            "Worker %d: Error getting legal actions P%d depth %d: %s. State: %s",
            worker_id,
            player,
            depth,
            e_legal,
            game_state,
            exc_info=True,
        )
        return np.zeros(NUM_PLAYERS, dtype=np.float64)
    num_actions = len(legal_actions)

    # Handle No Legal Actions (ensure logger calls work)
    if num_actions == 0:
        # This check should ideally happen within get_legal_actions or is_terminal
        # If we reach here and it's not terminal, it's the bug we need to fix.
        if not game_state.is_terminal():
            # This log might still appear if the underlying cause in the engine isn't fixed yet.
            logger.error(  # Keep as ERROR
                "Worker %d: No legal actions P%d depth %d, but state non-terminal! State: %s Context: %s",
                worker_id,
                player,
                depth,
                game_state,
                (
                    current_context.name
                    if isinstance(current_context, DecisionContext)
                    else current_context
                ),
            )
            # To avoid crashing, return neutral utility, but this highlights the bug.
            return np.zeros(NUM_PLAYERS, dtype=np.float64)
        else:  # Correctly terminal
            return np.array(
                [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
            )

    # Get Strategy from Snapshot (ensure logger calls work)
    current_regrets: Optional[np.ndarray] = regret_sum_snapshot.get(infoset_key)
    strategy = np.array([])  # Initialize strategy

    # Check if snapshot contains the key and dimensions match
    if current_regrets is not None and len(current_regrets) == num_actions:
        strategy = get_rm_plus_strategy(current_regrets)
    else:
        # Handle missing key or dimension mismatch
        if current_regrets is not None:  # Dimension mismatch
            logger.warning(
                "Worker %d: Regret dim mismatch key %s. Snap:%d Need:%d. Using uniform.",
                worker_id,
                infoset_key,
                len(current_regrets),
                num_actions,
            )
        # else: Key not found, expected if infoset is new

        # Use uniform strategy
        strategy = np.ones(num_actions) / num_actions if num_actions > 0 else np.array([])

        # Initialize local updates for this new/mismatched infoset if needed
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
    opponent_reach = reach_probs[opponent]

    # Accumulate Strategy Sum Update (ensure logger calls work)
    # Ensure strategy has the correct dimension before updating sums
    if player == updating_player and weight > 0 and player_reach > 1e-9:
        if len(strategy) == num_actions:
            # Ensure the local update dict entry exists and has correct dimension
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
                "Worker %d: Strategy len %d != num_actions %d for key %s. Skip strat update.",
                worker_id,
                len(strategy),
                num_actions,
                infoset_key,
            )

    # Iterate Through Actions (ensure logger calls work)
    action_utilities = np.zeros((num_actions, NUM_PLAYERS), dtype=np.float64)
    node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

    for i, action in enumerate(legal_actions):
        if i >= len(strategy):
            logger.error(
                "Worker %d: Action index %d OOB for strategy len %d. Key: %s",
                worker_id,
                i,
                len(strategy),
                infoset_key,
            )
            continue  # Skip this action to prevent crash
        action_prob = strategy[i]
        # Skip actions with near-zero probability to save computation
        # Note: RBP pruning could be added here based on regret_sum_snapshot
        if action_prob < 1e-9:
            continue

        # Apply Action (ensure logger calls work)
        state_delta: Optional[StateDelta] = None
        undo_info: Optional[UndoInfo] = None
        drawn_card_this_step: Optional[CardObject] = None
        try:
            # Check if the action involves consuming the drawn card from pending data
            if isinstance(
                action, (ActionReplace, ActionDiscard)
            ) and game_state.pending_action_data.get("drawn_card"):
                drawn_card_this_step = game_state.pending_action_data["drawn_card"]

            state_delta, undo_info = game_state.apply_action(action)
            if not callable(undo_info):
                logger.error(
                    "Worker %d: apply_action for %s returned invalid undo_info. State:%s",
                    worker_id,
                    action,
                    game_state,
                )
                # Treat as failure for this branch
                action_utilities[i] = np.zeros(NUM_PLAYERS, dtype=np.float64)
                continue  # Skip recursion for this action

        except Exception as apply_err:
            logger.error(
                "Worker %d: Error applying action %s P%d depth %d: %s. State:%s",
                worker_id,
                action,
                player,
                depth,
                apply_err,
                game_state,
                exc_info=True,
            )
            continue  # Skip this action branch

        # Create observation and update agent states (ensure logger calls work)
        observation = _create_observation(
            None,
            action,
            game_state,  # State *after* action
            player,  # Player who took the action
            game_state.snap_results_log,  # Log *after* action (might include snap results)
            drawn_card_this_step,  # Pass card involved if Discard/Replace
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
                    "Worker %d: Error updating agent state P%d after action %s depth %d: %s. State(post-action):%s Obs:%s",
                    worker_id,
                    agent_idx,
                    action,
                    depth,
                    e_update,
                    game_state,
                    observation,
                    exc_info=True,
                )
                agent_update_failed = True
                break  # Stop processing agents for this action branch
            next_agent_states.append(cloned_agent)

        # If agent update failed for any player, undo and skip recursion
        if agent_update_failed:
            if undo_info:
                try:
                    undo_info()
                except Exception as undo_e:
                    logger.error(
                        "Worker %d: Error undoing after agent update fail: %s",
                        worker_id,
                        undo_e,
                        exc_info=True,
                    )
            continue  # Skip recursion for this action

        # Calculate next reach probabilities
        temp_reach_probs = reach_probs.copy()
        temp_reach_probs[player] *= action_prob

        # Recursive call (ensure logger calls work)
        try:
            recursive_utilities = _traverse_game_for_worker(
                game_state,  # Pass the state *after* the action
                next_agent_states,  # Pass the updated agent states
                temp_reach_probs,  # Pass updated reach probs
                iteration,
                updating_player,
                weight,
                regret_sum_snapshot,
                config,
                local_regret_updates,
                local_strategy_sum_updates,
                local_reach_prob_updates,
                depth + 1,  # Increment depth
                worker_stats,  # Pass stats object along
                progress_queue,  # Pass progress queue along
                worker_id,  # Pass worker id along
            )
            action_utilities[i] = recursive_utilities
        except Exception as recursive_err:
            logger.error(
                "Worker %d: Error in recursive call action %s depth %d: %s. State:%s",
                worker_id,
                action,
                depth,
                recursive_err,
                game_state,
                exc_info=True,
            )
            action_utilities[i] = np.zeros(NUM_PLAYERS, dtype=np.float64)

        # Undo action (ensure logger calls work)
        if undo_info:
            try:
                undo_info()
                # Sanity check: Ensure state after undo matches state before apply_action (requires cloning state before loop)
                # This is computationally expensive, rely on logging/testing for now.
            except Exception as undo_e:
                logger.error(
                    "Worker %d: Error undoing action %s depth %d: %s. State likely corrupt. Returning zero.",
                    worker_id,
                    action,
                    depth,
                    undo_e,
                    exc_info=True,
                )
                # If undo fails, state is corrupt, safest to return 0 and stop traversal down this path
                return np.zeros(NUM_PLAYERS, dtype=np.float64)
        # else: Already logged error when undo_info was missing/invalid after apply_action call

    # Calculate Node Value & Accumulate Regret Update (ensure logger calls work)
    # Ensure strategy used has the same length as action_utilities collected
    valid_strategy_for_value_calc = (
        strategy
        if len(strategy) == num_actions
        else (np.ones(num_actions) / num_actions if num_actions > 0 else np.array([]))
    )
    if len(valid_strategy_for_value_calc) > 0:
        node_value = np.sum(
            valid_strategy_for_value_calc[:, np.newaxis] * action_utilities, axis=0
        )
    else:  # No actions, node value is 0
        node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

    if player == updating_player:
        # Ensure local update entry exists and matches dimension
        if (
            infoset_key not in local_regret_updates
            or len(local_regret_updates[infoset_key]) != num_actions
        ):
            local_regret_updates[infoset_key] = np.zeros(num_actions, dtype=np.float64)

        if num_actions > 0:  # Only calculate regret if actions exist
            player_action_values = action_utilities[:, player]
            player_node_value = node_value[player]
            instantaneous_regret = player_action_values - player_node_value
            update_weight = opponent_reach
            # Ensure instantaneous regret has same shape as local update array
            if len(instantaneous_regret) == len(local_regret_updates[infoset_key]):
                local_regret_updates[infoset_key] += update_weight * instantaneous_regret
            else:
                logger.error(
                    "Worker %d: Regret calculation shape mismatch. Inst: %s, Local: %s. Key: %s",
                    worker_id,
                    instantaneous_regret.shape,
                    local_regret_updates[infoset_key].shape,
                    infoset_key,
                )

    return node_value


def run_cfr_simulation_worker(
    worker_args: Tuple[
        int,
        Config,
        RegretSnapshotDict,
        Optional[GenericQueue],
        Optional[ProgressQueue],
        int,
    ],
) -> Optional[WorkerResult]:  # Return the WorkerResult dataclass
    """Top-level function executed by each worker process."""
    # !!! IMPORTANT: Setup logging FIRST using the queue !!!
    iteration, config, regret_sum_snapshot, log_queue, progress_queue, worker_id = (
        worker_args  # Unpack args
    )
    logger = None  # Initialize logger variable
    try:
        worker_root_logger = logging.getLogger()  # Get root logger for this process
        if log_queue:
            # Avoid adding handler if it already exists
            if not any(
                isinstance(h, logging.handlers.QueueHandler) and h.queue is log_queue
                for h in worker_root_logger.handlers
            ):
                # Remove any handlers inherited via fork (if any)
                for handler in worker_root_logger.handlers[:]:
                    worker_root_logger.removeHandler(handler)
                # Add the queue handler
                queue_handler = logging.handlers.QueueHandler(log_queue)
                worker_root_logger.addHandler(queue_handler)
                # Set worker logger level to DEBUG to send everything to the queue
                worker_root_logger.setLevel(logging.DEBUG)
                # Now safe to get and use child loggers
                logger = logging.getLogger(__name__)  # Assign logger here
                # logger.info("Worker %d logging initialized via QueueHandler.", worker_id) # Reduced noise
        else:
            # Fallback or error handling if queue is not provided
            if (
                not worker_root_logger.hasHandlers()
            ):  # Add NullHandler only if no other handlers exist
                worker_root_logger.addHandler(logging.NullHandler())
            logger = logging.getLogger(__name__)  # Get logger even if queue fails
            logger.warning(
                "Worker %d running without queue logging.", worker_id
            )  # Log a warning if possible

        # --- Now wrap the main worker logic in a try...except block ---
        # This inner try...except catches errors within the simulation logic
        try:
            worker_stats = WorkerStats()  # Initialize stats for this run
            # logger.debug("Worker %d starting iteration %d.", worker_id, iteration) # Reduced noise

            # --- Initialize Game and Agent States ---
            game_state = CambiaGameState(house_rules=config.cambia_rules)
            initial_agent_states = []
            initial_hands_for_log = [list(p.hand) for p in game_state.players]
            initial_peeks_for_log = [p.initial_peek_indices for p in game_state.players]

            if not game_state.is_terminal():
                # Create initial observation *before* initializing agents
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
                    # Pass the single initial observation to all agents
                    agent.initialize(
                        initial_obs, initial_hands_for_log[i], initial_peeks_for_log[i]
                    )
                    initial_agent_states.append(agent)
            else:
                logger.error(
                    "Worker %d: Game terminal at init. State: %s", worker_id, game_state
                )
                return None  # Cannot start traversal

            if len(initial_agent_states) != NUM_PLAYERS:
                logger.error(
                    "Worker %d: Failed to initialize all agent states.", worker_id
                )
                return None

            # --- Determine Updating Player & Iteration Weight ---
            updating_player = iteration % NUM_PLAYERS
            if config.cfr_plus_params.weighted_averaging_enabled:
                delay = config.cfr_plus_params.averaging_delay
                # Iteration weight starts from 1 at iter d+1
                weight = float(max(0, (iteration + 1) - (delay + 1)))
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
                reach_probs=np.ones(
                    NUM_PLAYERS, dtype=np.float64
                ),  # Initial reach is 1 for both
                iteration=iteration,
                updating_player=updating_player,
                weight=weight,
                regret_sum_snapshot=regret_sum_snapshot,
                config=config,
                local_regret_updates=local_regret_updates,
                local_strategy_sum_updates=local_strategy_sum_updates,
                local_reach_prob_updates=local_reach_prob_updates,
                depth=0,
                worker_stats=worker_stats,  # Pass stats object
                progress_queue=progress_queue,  # Pass progress queue
                worker_id=worker_id,  # Pass worker id
            )

            # logger.debug("Worker %d finished iteration %d. MaxDepth: %d, Nodes: %d", worker_id, iteration, worker_stats.max_depth, worker_stats.nodes_visited) # Reduced noise
            # Return WorkerResult dataclass instance
            return WorkerResult(
                regret_updates=dict(local_regret_updates),
                strategy_updates=dict(local_strategy_sum_updates),
                reach_prob_updates=dict(local_reach_prob_updates),
                stats=worker_stats,
            )

        except Exception as e_inner:
            # Log exception using the configured queue handler *inside the worker*
            # Use the logger instance obtained after setup
            if logger:  # Check if logger was initialized
                logger.error(
                    "!!! Unhandled Error in worker %d iter %d simulation: %s !!!",
                    worker_id,
                    iteration,
                    e_inner,
                    exc_info=True,  # Include traceback in the log sent via queue
                )
            else:  # Fallback print if logger failed
                print(
                    f"!!! FATAL WORKER ERROR (Worker {worker_id}, Iter {iteration}): {e_inner} !!!",
                    file=sys.stderr,
                )
            # Return None to indicate failure *without* pickling the exception
            return None

    # This outer try...except catches errors during logger setup itself
    except Exception as e_outer:
        # Cannot use logger here as setup might have failed
        print(
            f"!!! CRITICAL Error during worker {worker_id} init (Iter {iteration}): {e_outer} !!!",
            file=sys.stderr,
        )
        # Print traceback manually if possible
        import traceback

        traceback.print_exc(file=sys.stderr)
        return None  # Indicate failure


# --- Helper functions needed by _traverse_game_for_worker ---
# (Implementation unchanged from v0.7.4, but ensure consistency)
def _create_observation(
    prev_state: Optional[
        CambiaGameState
    ],  # Not currently used but kept for potential future use
    action: Optional[GameAction],  # Action that LED to next_state
    next_state: CambiaGameState,  # State AFTER action was applied
    acting_player: int,  # Player who took the action
    snap_results: List[Dict],  # Snap results that occurred *during* this action step
    explicit_drawn_card: Optional[
        CardObject
    ] = None,  # Card drawn if action was Draw... or Discard/Replace
) -> AgentObservation:
    """Creates the AgentObservation object based on the state *after* the action."""
    logger = logging.getLogger(__name__)  # Get logger instance
    discard_top = next_state.get_discard_top()
    hand_sizes = [next_state.get_player_card_count(i) for i in range(NUM_PLAYERS)]
    stock_size = next_state.get_stockpile_size()
    cambia_called = next_state.cambia_caller_id is not None
    who_called = next_state.cambia_caller_id
    game_over = next_state.is_terminal()
    turn_num = next_state.get_turn_number()

    # Determine the drawn card relevant to this observation context
    drawn_card_for_obs = None
    if isinstance(action, (ActionDiscard, ActionReplace)):
        # If action was Discard or Replace, the 'explicit_drawn_card' is the one
        # that was originally drawn and now involved in this action.
        # This is the card that becomes 'public knowledge' potentially.
        drawn_card_for_obs = explicit_drawn_card
        # logger.debug(f"## OBS CREATE ## Action {type(action).__name__}, explicit_drawn_card: {explicit_drawn_card}, set drawn_card_for_obs: {drawn_card_for_obs}")

    elif explicit_drawn_card is not None and acting_player != -1:
        # If an explicit card is passed (e.g., from ActionDrawStockpile)
        # it's relevant for the actor's observation immediately.
        drawn_card_for_obs = explicit_drawn_card
        # logger.debug(f"## OBS CREATE ## Action {type(action).__name__}, explicit_drawn_card: {explicit_drawn_card} (actor {acting_player}), set drawn_card_for_obs: {drawn_card_for_obs}")

    # Determine peeked cards relevant to this observation
    peeked_cards_dict = None
    # Peek results are associated with the *Select* or *Look* action that generated them.
    # KingSwapDecision action itself doesn't generate new peek info in the observation.
    if isinstance(action, ActionAbilityPeekOwnSelect) and acting_player != -1:
        # We need the state *after* the action to confirm the peeked card
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
        # Re-fetch cards looked at from the state *after* the look action
        own_idx, opp_look_idx = action.own_hand_index, action.opponent_hand_index
        opp_real_idx = next_state.get_opponent_index(acting_player)
        own_hand, opp_hand = next_state.get_player_hand(
            acting_player
        ), next_state.get_player_hand(opp_real_idx)
        card1 = own_hand[own_idx] if own_hand and 0 <= own_idx < len(own_hand) else None
        card2 = (
            opp_hand[opp_look_idx]
            if opp_hand and 0 <= opp_look_idx < len(opp_hand)
            else None
        )
        if card1 and card2:
            peeked_cards_dict = {
                (acting_player, own_idx): card1,
                (opp_real_idx, opp_look_idx): card2,
            }

    # Use the snap results passed in (should reflect results from this specific step)
    final_snap_results = snap_results if snap_results else []

    obs = AgentObservation(
        acting_player=acting_player,
        action=action,
        discard_top_card=discard_top,
        player_hand_sizes=hand_sizes,
        stockpile_size=stock_size,
        drawn_card=drawn_card_for_obs,  # Pass potentially visible drawn card
        peeked_cards=peeked_cards_dict,  # Pass peek info caused by this action
        snap_results=final_snap_results,  # Pass snap results from this step
        did_cambia_get_called=cambia_called,
        who_called_cambia=who_called,
        is_game_over=game_over,
        current_turn=turn_num,
    )
    # logger.debug(f"## OBS CREATE ## Created observation: {obs}")
    return obs


def _filter_observation(obs: AgentObservation, observer_id: int) -> AgentObservation:
    """Creates a player-specific view of the observation, masking private info."""
    logger = logging.getLogger(__name__)  # Get logger instance
    # Start with a shallow copy, then modify fields that need filtering
    filtered_obs = copy.copy(obs)

    # --- Filter Drawn Card ---
    # Observer only sees the drawn card if:
    # 1. They were the acting player who drew it (ActionDraw...).
    # 2. The action was Discard or Replace (making the drawn card public knowledge via discard pile).
    is_public_reveal = isinstance(obs.action, (ActionDiscard, ActionReplace))

    if obs.drawn_card and obs.acting_player != observer_id and not is_public_reveal:
        # If observer didn't act and it wasn't a public reveal action, hide the card.
        # logger.debug(f"## OBS FILTER ## P{observer_id} filtering drawn_card from P{obs.acting_player}'s action {type(obs.action).__name__}")
        filtered_obs.drawn_card = None
    # else: logger.debug(f"## OBS FILTER ## P{observer_id} keeps drawn_card (Actor:{obs.acting_player==observer_id}, PublicReveal:{is_public_reveal})")

    # --- Filter Peeked Cards ---
    # Observer only sees peek results if they were the actor performing the peek/look.
    if obs.peeked_cards:
        # Check if the action type corresponds to a peek/look by the observer
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
            # If the observer didn't perform the peek/look action, hide the results.
            # logger.debug(f"## OBS FILTER ## P{observer_id} filtering peeked_cards from P{obs.acting_player}'s action {type(obs.action).__name__}")
            filtered_obs.peeked_cards = None
        # else: logger.debug(f"## OBS FILTER ## P{observer_id} keeps peeked_cards.")
    else:
        filtered_obs.peeked_cards = None  # Ensure it's None if originally None

    # Snap results are considered public information
    filtered_obs.snap_results = obs.snap_results  # Already copied by shallow copy

    # logger.debug(f"## OBS FILTER ## Filtered for P{observer_id}: {filtered_obs}")
    return filtered_obs
