"""src/cfr/worker.py"""

import copy
import logging
import os
import queue
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, TypeAlias

import numpy as np

# Relative imports from parent directories
from ..agent_state import AgentObservation, AgentState
from ..config import Config
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
    CardObject,
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
)

RegretSnapshotDict: TypeAlias = Dict[InfosetKey, np.ndarray]
ProgressQueue: TypeAlias = queue.Queue


PROGRESS_UPDATE_NODE_INTERVAL = 1000  # Send update every N nodes


# --- Traversal Logic ---
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
    # Get logger instance AFTER setup in run_cfr_simulation_worker
    logger = logging.getLogger(__name__)

    # Update stats
    worker_stats.nodes_visited += 1
    worker_stats.max_depth = max(worker_stats.max_depth, depth)

    # --- Send Progress Update Periodically ---
    if progress_queue and (
        worker_stats.nodes_visited % PROGRESS_UPDATE_NODE_INTERVAL == 0
    ):
        try:
            # Send current depth, max depth for this worker, and nodes visited
            progress_update = (
                worker_id,
                depth,  # Current recursion depth
                worker_stats.max_depth,  # Max depth seen by this worker so far
                worker_stats.nodes_visited,  # Total nodes visited by this worker
            )
            # Use put_nowait for manager queue if progress_queue comes from manager
            progress_queue.put_nowait(progress_update)
        except queue.Full:
            pass
        except Exception as pq_e:
            logger.error(
                "Error putting progress update on queue for worker %d: %s",
                worker_id,
                pq_e,
            )
            worker_stats.error_count += 1

    # Base Case
    if game_state.is_terminal():
        return np.array(
            [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
        )

    # Check depth limit
    if depth >= config.system.recursion_limit:
        logger.error(
            "Worker %d: Max recursion depth (%d) reached. Returning 0.", worker_id, depth
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # --- Determine Context, Acting Player, Infoset Key ---
    # Determine Context (ensure logger calls work)
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
                "Worker %d: Unknown pending action type (%s) for context.",
                worker_id,
                type(pending).__name__,
            )
            worker_stats.warning_count += 1
            current_context = DecisionContext.START_TURN  # Fallback
    else:  # Normal start of turn
        current_context = DecisionContext.START_TURN

    player = game_state.get_acting_player()
    if player == -1:
        logger.error(
            "Worker %d: Could not determine acting player depth %d. State: %s",
            worker_id,
            depth,
            game_state,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)
    current_agent_state = agent_states[player]
    opponent = 1 - player
    try:
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
            worker_stats.error_count += 1
            return np.zeros(NUM_PLAYERS, dtype=np.float64)
        base_infoset_tuple = current_agent_state.get_infoset_key()
        if not isinstance(current_context, DecisionContext):
            logger.error(
                "Worker %d: Invalid context type '%s' for infoset key.",
                worker_id,
                type(current_context),
            )
            worker_stats.error_count += 1
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
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # --- Get Legal Actions ---
    try:
        legal_actions_set = game_state.get_legal_actions()
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
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)
    num_actions = len(legal_actions)

    if num_actions == 0:
        if not game_state.is_terminal():
            logger.error(
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
            worker_stats.error_count += 1
            # Non-terminal state bug candidate (Backlog #1)
            return np.zeros(NUM_PLAYERS, dtype=np.float64)
        else:  # Correctly terminal
            return np.array(
                [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
            )

    # --- Get Strategy from Snapshot ---
    current_regrets: Optional[np.ndarray] = regret_sum_snapshot.get(infoset_key)
    strategy = np.array([])  # Initialize strategy
    if current_regrets is not None and len(current_regrets) == num_actions:
        strategy = get_rm_plus_strategy(current_regrets)
    else:
        if current_regrets is not None:  # Dimension mismatch
            logger.warning(
                "Worker %d: Regret dim mismatch key %s. Snap:%d Need:%d. Using uniform.",
                worker_id,
                infoset_key,
                len(current_regrets),
                num_actions,
            )
            worker_stats.warning_count += 1
        strategy = np.ones(num_actions) / num_actions if num_actions > 0 else np.array([])
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

    # --- Accumulate Updates & Iterate Through Actions ---
    player_reach = reach_probs[player]
    opponent_reach = reach_probs[opponent]
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
                "Worker %d: Strategy len %d != num_actions %d for key %s. Skip strat update.",
                worker_id,
                len(strategy),
                num_actions,
                infoset_key,
            )
            worker_stats.error_count += 1

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
            worker_stats.error_count += 1
            continue
        action_prob = strategy[i]
        if action_prob < 1e-9:
            continue

        state_delta: Optional[StateDelta] = None
        undo_info: Optional[UndoInfo] = None
        drawn_card_this_step: Optional[CardObject] = None
        try:
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
                worker_stats.error_count += 1
                action_utilities[i] = np.zeros(NUM_PLAYERS, dtype=np.float64)
                continue

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
            worker_stats.error_count += 1
            continue

        # Create observation and update agent states
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
                worker_stats.error_count += 1
                agent_update_failed = True
                break
            next_agent_states.append(cloned_agent)

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
                    worker_stats.error_count += 1
            continue

        temp_reach_probs = reach_probs.copy()
        temp_reach_probs[player] *= action_prob

        # Recursive call
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
                worker_stats,
                progress_queue,
                worker_id,
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
            worker_stats.error_count += 1
            action_utilities[i] = np.zeros(NUM_PLAYERS, dtype=np.float64)

        # Undo action
        if undo_info:
            try:
                undo_info()
            except Exception as undo_e:
                logger.error(
                    "Worker %d: Error undoing action %s depth %d: %s. State likely corrupt. Returning zero.",
                    worker_id,
                    action,
                    depth,
                    undo_e,
                    exc_info=True,
                )
                worker_stats.error_count += 1
                return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # --- Calculate Node Value & Accumulate Regret Update ---
    valid_strategy_for_value_calc = (
        strategy
        if len(strategy) == num_actions
        else (np.ones(num_actions) / num_actions if num_actions > 0 else np.array([]))
    )
    if len(valid_strategy_for_value_calc) > 0:
        node_value = np.sum(
            valid_strategy_for_value_calc[:, np.newaxis] * action_utilities, axis=0
        )
    else:
        node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

    if player == updating_player:
        if (
            infoset_key not in local_regret_updates
            or len(local_regret_updates[infoset_key]) != num_actions
        ):
            local_regret_updates[infoset_key] = np.zeros(num_actions, dtype=np.float64)

        if num_actions > 0:
            player_action_values = action_utilities[:, player]
            player_node_value = node_value[player]
            instantaneous_regret = player_action_values - player_node_value
            update_weight = opponent_reach
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
                worker_stats.error_count += 1

    return node_value


def run_cfr_simulation_worker(
    worker_args: Tuple[
        int,  # iteration
        Config,
        RegretSnapshotDict,
        Optional[ProgressQueue],  # progress_queue
        int,  # worker_id
        str,  # run_log_dir
        str,  # run_timestamp
    ],
) -> Optional[WorkerResult]:
    """Top-level function executed by each worker process. Sets up per-worker logging."""
    logger = None  # Initialize logger variable
    worker_stats = WorkerStats()  # Initialize stats early for exception handling
    (
        iteration,
        config,
        regret_sum_snapshot,
        progress_queue,
        worker_id,
        run_log_dir,
        run_timestamp,
    ) = worker_args

    try:
        # --- Per-Worker Logging Setup ---
        worker_root_logger = logging.getLogger()  # Get root logger for this process
        # Remove any inherited handlers
        for handler in worker_root_logger.handlers[:]:
            worker_root_logger.removeHandler(handler)

        try:
            # Create worker-specific directory
            worker_log_dir = os.path.join(run_log_dir, f"w{worker_id}")
            os.makedirs(worker_log_dir, exist_ok=True)

            # Construct base log pattern (handler adds _NNN.log)
            log_pattern = os.path.join(
                worker_log_dir,
                f"{config.logging.log_file_prefix}_run_{run_timestamp}-w{worker_id}",
            )

            # Create and add the SerialRotatingFileHandler for this worker
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)-8s - [%(processName)-20s] - %(name)-25s - %(message)s"
            )
            file_handler = SerialRotatingFileHandler(
                filename_pattern=log_pattern,
                maxBytes=config.logging.log_max_bytes,
                backupCount=config.logging.log_backup_count,
                encoding="utf-8",
            )
            # Use the file log level from config for workers too
            file_log_level = getattr(
                logging, config.logging.log_level_file.upper(), logging.DEBUG
            )
            file_handler.setLevel(file_log_level)
            file_handler.setFormatter(formatter)
            worker_root_logger.addHandler(file_handler)

            # Set worker's root logger level (e.g., DEBUG to capture everything for the file)
            worker_root_logger.setLevel(logging.DEBUG)
            logger = logging.getLogger(__name__)  # Now safe to get child logger
            logger.info(
                "Worker %d logging initialized to directory %s", worker_id, worker_log_dir
            )

        except Exception as log_setup_e:
            # If logging setup fails, print error and continue without logging
            print(
                f"!!! CRITICAL Error setting up logging for worker {worker_id}: {log_setup_e} !!!",
                file=sys.stderr,
            )
            worker_stats.error_count += 1  # Count logging setup error
            # Add a NullHandler to prevent "No handler found" warnings
            if not worker_root_logger.hasHandlers():
                worker_root_logger.addHandler(logging.NullHandler())
            logger = logging.getLogger(__name__)  # Get logger anyway

        # --- Now wrap the main worker logic ---
        try:
            # logger.debug("Worker %d starting iteration %d.", worker_id, iteration) # Reduced noise

            # Initialize Game and Agent States
            game_state = CambiaGameState(house_rules=config.cambia_rules)
            initial_agent_states = []
            if not game_state.is_terminal():
                initial_obs = _create_observation(None, None, game_state, -1, [], None)
                initial_hands_for_log = [list(p.hand) for p in game_state.players]
                initial_peeks_for_log = [
                    p.initial_peek_indices for p in game_state.players
                ]
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
                logger.error(
                    "Worker %d: Game terminal at init. State: %s", worker_id, game_state
                )
                worker_stats.error_count += 1
                return WorkerResult(stats=worker_stats)  # Return stats even on error

            if len(initial_agent_states) != NUM_PLAYERS:
                logger.error(
                    "Worker %d: Failed to initialize all agent states.", worker_id
                )
                worker_stats.error_count += 1
                return WorkerResult(stats=worker_stats)  # Return stats even on error

            # Determine Updating Player & Iteration Weight
            updating_player = iteration % NUM_PLAYERS
            if config.cfr_plus_params.weighted_averaging_enabled:
                delay = config.cfr_plus_params.averaging_delay
                weight = float(max(0, (iteration + 1) - (delay + 1)))
            else:
                weight = 1.0

            # Initialize Local Update Dictionaries
            local_regret_updates: LocalRegretUpdateDict = defaultdict(
                lambda: np.array([], dtype=np.float64)
            )
            local_strategy_sum_updates: LocalStrategyUpdateDict = defaultdict(
                lambda: np.array([], dtype=np.float64)
            )
            local_reach_prob_updates: LocalReachProbUpdateDict = defaultdict(float)

            # Run Traversal
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
                worker_stats=worker_stats,
                progress_queue=progress_queue,
                worker_id=worker_id,
            )

            # logger.debug("Worker %d finished iteration %d.", worker_id, iteration) # Reduced noise
            return WorkerResult(
                regret_updates=dict(local_regret_updates),
                strategy_updates=dict(local_strategy_sum_updates),
                reach_prob_updates=dict(local_reach_prob_updates),
                stats=worker_stats,
            )

        except Exception as e_inner:
            worker_stats.error_count += 1  # Count inner simulation error
            if logger:  # Log if logger was successfully set up
                logger.error(
                    "!!! Unhandled Error in worker %d iter %d simulation: %s !!!",
                    worker_id,
                    iteration,
                    e_inner,
                    exc_info=True,
                )
            else:  # Fallback print
                print(
                    f"!!! FATAL WORKER ERROR (Worker {worker_id}, Iter {iteration}): {e_inner} !!!",
                    file=sys.stderr,
                )
            # Return only stats on inner error
            return WorkerResult(stats=worker_stats)

    # This outer try...except catches errors during initial setup before logger might exist
    except Exception as e_outer:
        worker_stats.error_count += 1  # Count outer setup error
        print(
            f"!!! CRITICAL Error during worker {worker_id} init (Iter {iteration}): {e_outer} !!!",
            file=sys.stderr,
        )
        import traceback

        traceback.print_exc(file=sys.stderr)
        # Return only stats on outer error
        return WorkerResult(stats=worker_stats)

    finally:
        # Explicitly shutdown logging for the worker process?
        # logging.shutdown() # Maybe needed depending on start method / OS
        pass


# --- Helper functions _create_observation, _filter_observation ---
def _create_observation(
    prev_state: Optional[CambiaGameState],
    action: Optional[GameAction],
    next_state: CambiaGameState,
    acting_player: int,
    snap_results: List[Dict],
    explicit_drawn_card: Optional[CardObject] = None,
) -> AgentObservation:
    """Creates the AgentObservation object based on the state *after* the action."""
    logger_obs = logging.getLogger(__name__)  # Get logger instance for this helper
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
        # logger_obs.debug(f"## OBS CREATE ## Action {type(action).__name__}, explicit_drawn_card: {explicit_drawn_card}, set drawn_card_for_obs: {drawn_card_for_obs}")
    elif explicit_drawn_card is not None and acting_player != -1:
        drawn_card_for_obs = explicit_drawn_card
        # logger_obs.debug(f"## OBS CREATE ## Action {type(action).__name__}, explicit_drawn_card: {explicit_drawn_card} (actor {acting_player}), set drawn_card_for_obs: {drawn_card_for_obs}")

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
    # logger_obs.debug(f"## OBS CREATE ## Created observation: {obs}")
    return obs


def _filter_observation(obs: AgentObservation, observer_id: int) -> AgentObservation:
    """Creates a player-specific view of the observation, masking private info."""
    logger_filt = logging.getLogger(__name__)  # Get logger instance for this helper
    filtered_obs = copy.copy(obs)

    is_public_reveal = isinstance(obs.action, (ActionDiscard, ActionReplace))
    if obs.drawn_card and obs.acting_player != observer_id and not is_public_reveal:
        # logger_filt.debug(f"## OBS FILTER ## P{observer_id} filtering drawn_card from P{obs.acting_player}'s action {type(obs.action).__name__}")
        filtered_obs.drawn_card = None
    # else: logger_filt.debug(f"## OBS FILTER ## P{observer_id} keeps drawn_card (Actor:{obs.acting_player==observer_id}, PublicReveal:{is_public_reveal})")

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
            # logger_filt.debug(f"## OBS FILTER ## P{observer_id} filtering peeked_cards from P{obs.acting_player}'s action {type(obs.action).__name__}")
            filtered_obs.peeked_cards = None
        # else: logger_filt.debug(f"## OBS FILTER ## P{observer_id} keeps peeked_cards.")
    else:
        filtered_obs.peeked_cards = None

    filtered_obs.snap_results = obs.snap_results
    # logger_filt.debug(f"## OBS FILTER ## Filtered for P{observer_id}: {filtered_obs}")
    return filtered_obs
