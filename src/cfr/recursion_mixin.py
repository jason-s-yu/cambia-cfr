# src/cfr/recursion_mixin.py
"""Mixin class for the core CFR+ recursive traversal logic."""

import logging
import copy
import threading
from collections import deque
from typing import Callable, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from ..agent_state import AgentState, AgentObservation
from ..constants import (
    ActionDiscard,
    ActionReplace,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionSnapOpponentMove,
    DecisionContext,
    GameAction,
    CardObject,
)
from ..game.engine import CambiaGameState
from ..game.types import StateDelta, UndoInfo
from ..utils import InfosetKey, get_rm_plus_strategy

from .exceptions import GracefulShutdownException

logger = logging.getLogger(__name__)


class CFRRecursionMixin:
    """Handles the recursive CFR+ game tree traversal and updates."""

    # Attributes expected to be initialized in the main class's __init__
    # self.config: Config
    # self.num_players: int
    # self.regret_sum: PolicyDict
    # self.strategy_sum: PolicyDict
    # self.reach_prob_sum: ReachProbDict
    # self.shutdown_event: threading.Event
    # self.max_depth_this_iter: int
    # self._last_exploit_str: str

    def _cfr_recursive(
        self,
        game_state: CambiaGameState,
        agent_states: List[AgentState],
        reach_probs: np.ndarray,
        iteration: int,  # Iteration number currently being run
        action_log: List[Dict],
        status_bar: tqdm,
        depth: int,
        shutdown_event: threading.Event,
    ) -> np.ndarray:
        """Performs the recursive CFR+ calculation for a given game state."""

        # --- Check for Shutdown Request ---
        if shutdown_event.is_set():
            raise GracefulShutdownException("Shutdown requested during recursion")

        self.max_depth_this_iter = max(self.max_depth_this_iter, depth)

        # --- Update Status Bar ---
        total_infosets = len(self.regret_sum)
        status_desc = f"Iter {iteration} | Depth:{depth} | MaxD:{self.max_depth_this_iter} | Nodes:{total_infosets:,} | Expl:{self._last_exploit_str}"
        status_bar.set_description_str(status_desc)

        # --- Base Case: Terminal Node ---
        if game_state.is_terminal():
            return np.array(
                [game_state.get_utility(i) for i in range(self.num_players)],
                dtype=np.float64,
            )

        # --- Determine Node Type and Context ---
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
                    "Iter %d, Depth %d: Unhandled pending action type %s for context determination.",
                    iteration,
                    depth,
                    type(pending),
                )
                current_context = DecisionContext.START_TURN
        else:
            current_context = DecisionContext.START_TURN

        # --- Get Acting Player and Infoset ---
        player = game_state.get_acting_player()
        if player == -1:
            logger.error(
                "Iter %d, Depth %d: Could not determine acting player. State: %s. Context: %s",
                iteration,
                depth,
                game_state,
                current_context.name,
            )
            return np.zeros(self.num_players, dtype=np.float64)
        current_agent_state = agent_states[player]
        opponent = 1 - player
        try:
            if not hasattr(current_agent_state, "own_hand") or not hasattr(
                current_agent_state, "opponent_belief"
            ):
                logger.error(
                    "Iter %d, P%d, Depth %d: AgentState uninitialized. State: %s",
                    iteration,
                    player,
                    depth,
                    current_agent_state,
                )
                return np.zeros(self.num_players, dtype=np.float64)
            base_infoset_tuple = current_agent_state.get_infoset_key()
            infoset_key = InfosetKey(*base_infoset_tuple, current_context.value)
        except Exception:
            logger.exception(
                "Iter %d, P%d, Depth %d: Error getting infoset key. AgentState: %s. GameState: %s. Context: %s",
                iteration,
                player,
                depth,
                current_agent_state,
                game_state,
                current_context.name,
            )
            return np.zeros(self.num_players, dtype=np.float64)

        # --- Get Legal Actions ---
        try:
            legal_actions_set = game_state.get_legal_actions()
            # Sort actions by representation for consistency
            legal_actions = sorted(list(legal_actions_set), key=repr)
        except Exception as e:
            logger.exception(
                "Iter %d, P%d, Depth %d: Error getting legal actions. State: %s. InfosetKey: %s. Context: %s: %s",
                iteration,
                player,
                depth,
                game_state,
                infoset_key,
                current_context.name,
                e,
            )
            return np.zeros(self.num_players, dtype=np.float64)
        num_actions = len(legal_actions)

        # --- Handle No Legal Actions ---
        if num_actions == 0:
            if not game_state.is_terminal():
                logger.warning(
                    "Iter %d, P%d, Depth %d: No legal actions found at infoset %s in non-terminal state %s (Context: %s). Forcing end check.",
                    iteration,
                    player,
                    depth,
                    infoset_key,
                    game_state,
                    current_context.name,
                )
                local_undo_stack: deque[Callable[[], None]] = deque()
                local_delta_list: StateDelta = []
                # Use _check_game_end which should be available via self (likely from GameState via mixin)
                game_state._check_game_end(local_undo_stack, local_delta_list)
                # Undo any changes made by _check_game_end if it modified state
                while local_undo_stack:
                    local_undo_stack.popleft()()
                if game_state.is_terminal():
                    return np.array(
                        [game_state.get_utility(i) for i in range(self.num_players)],
                        dtype=np.float64,
                    )
                else:
                    logger.error(
                        "Iter %d, P%d, Depth %d: State still non-terminal after no-action check. Logic error.",
                        iteration,
                        player,
                        depth,
                    )
                    return np.zeros(self.num_players, dtype=np.float64)
            else:  # Already terminal, should have been caught earlier
                return np.array(
                    [game_state.get_utility(i) for i in range(self.num_players)],
                    dtype=np.float64,
                )

        # --- Initialize/Retrieve Regret and Strategy Sums ---
        current_regrets = self.regret_sum.get(infoset_key)
        if current_regrets is None or len(current_regrets) != num_actions:
            if current_regrets is not None:
                # Log dimension mismatch (potential issue with legal actions consistency or infoset key)
                logger.warning(
                    "## TEMP DEBUG ## Iter %d, P%d, Depth %d, Infoset %s: Regret dimension mismatch! Existing: %d, Expected: %d. Context: %s. Re-initializing.",
                    iteration,
                    player,
                    depth,
                    infoset_key,
                    len(current_regrets),
                    num_actions,
                    current_context.name,
                )
            # Initialize with zeros if key is new or dimension is wrong
            self.regret_sum[infoset_key] = np.zeros(num_actions, dtype=np.float64)
            current_regrets = self.regret_sum[infoset_key]

        current_strategy_sum = self.strategy_sum.get(infoset_key)
        if current_strategy_sum is None or len(current_strategy_sum) != num_actions:
            if current_strategy_sum is not None:
                logger.warning(
                    "## TEMP DEBUG ## Iter %d, P%d, Depth %d, Infoset %s: Strategy sum dimension mismatch! Existing: %d, Expected: %d. Context: %s. Re-initializing.",
                    iteration,
                    player,
                    depth,
                    infoset_key,
                    len(current_strategy_sum),
                    num_actions,
                    current_context.name,
                )
            self.strategy_sum[infoset_key] = np.zeros(num_actions, dtype=np.float64)

        if infoset_key not in self.reach_prob_sum:
            self.reach_prob_sum[infoset_key] = 0.0

        # --- Calculate Current Strategy & Update Strategy Sum ---
        strategy = get_rm_plus_strategy(current_regrets)
        player_reach = reach_probs[player]

        # Update strategy sum contribution from this player's visit
        if player_reach > 1e-9:  # Only update if the node is reachable by the player
            if self.config.cfr_plus_params.weighted_averaging_enabled:
                delay = self.config.cfr_plus_params.averaging_delay
                weight = float(max(0, iteration - delay))
            else:
                weight = 1.0

            if weight > 0:
                # Accumulate weighted reach probability sum for this infoset
                self.reach_prob_sum[infoset_key] += weight * player_reach
                # Accumulate weighted strategy sum for this infoset
                self.strategy_sum[infoset_key] += weight * player_reach * strategy

        # --- Iterate Through Actions ---
        action_utilities = np.zeros((num_actions, self.num_players), dtype=np.float64)
        node_value = np.zeros(self.num_players, dtype=np.float64)
        use_pruning = self.config.cfr_training.pruning_enabled
        pruning_threshold = self.config.cfr_training.pruning_threshold

        for i, action in enumerate(legal_actions):
            action_prob = strategy[i]

            # --- Regret Pruning ---
            node_positive_regret_sum = np.sum(np.maximum(0.0, current_regrets))
            should_prune = (
                use_pruning
                and player_reach > 1e-6  # Only prune if reasonably reachable
                and current_regrets[i] <= pruning_threshold
                and node_positive_regret_sum
                > pruning_threshold  # Avoid pruning if all regrets are zero
                and iteration
                > self.config.cfr_plus_params.averaging_delay
                + 10  # Prune only after averaging stabilizes
            )

            if should_prune:
                # Assign 0 utility if pruned, effectively ignoring this branch for current node value
                action_utilities[i] = np.zeros(self.num_players, dtype=np.float64)
                continue  # Skip recursion for this action

            if (
                action_prob < 1e-9 and not should_prune
            ):  # Also skip negligible probabilities unless pruning active
                action_utilities[i] = np.zeros(self.num_players, dtype=np.float64)
                continue

            # --- Apply Action and Recurse ---
            action_log_entry = {
                "player": player,
                "turn": game_state.get_turn_number(),
                "depth": depth,
                "context": current_context.name,
                "state_desc_before": str(game_state),
                "infoset_key": infoset_key,
                "action": action,
                "action_prob": action_prob,
                "reach_prob": player_reach,
            }
            state_before_this_action_str = action_log_entry["state_desc_before"]
            state_delta: Optional[StateDelta] = None
            undo_info: Optional[UndoInfo] = None
            state_after_action_desc = "ERROR"
            drawn_card_this_step: Optional[CardObject] = None

            try:
                # Store drawn card *before* applying action if applicable
                if isinstance(
                    action, (ActionReplace, ActionDiscard)
                ) and game_state.pending_action_data.get("drawn_card"):
                    drawn_card_this_step = game_state.pending_action_data["drawn_card"]
                state_delta, undo_info = game_state.apply_action(action)
                state_after_action_desc = str(game_state)
            except Exception as apply_err:
                logger.exception(
                    "Iter %d, P%d, Depth %d: Error applying action %s in state %s at infoset %s: %s",
                    iteration,
                    player,
                    depth,
                    action,
                    state_before_this_action_str,
                    infoset_key,
                    apply_err,
                )
                action_log_entry["outcome"] = f"ERROR applying action: {apply_err}"
                action_log.append(action_log_entry)
                action_utilities[i] = np.zeros(
                    self.num_players, dtype=np.float64
                )  # Assign 0 utility on error
                undo_info = None  # Ensure undo is not called
                continue  # Move to next action

            # Create observation based on the state *after* the action
            observation = self._create_observation(
                None,
                action,
                game_state,
                player,
                game_state.snap_results_log,
                drawn_card_this_step,
            )

            # Update agent states for the next step
            next_agent_states = []
            agent_update_failed = False
            for agent_idx, agent_state in enumerate(agent_states):
                cloned_agent = agent_state.clone()
                try:
                    player_specific_obs = self._filter_observation(observation, agent_idx)
                    cloned_agent.update(player_specific_obs)
                except Exception as e:
                    logger.exception(
                        "Iter %d, P%d, Depth %d: Error updating AgentState %d for action %s. Infoset: %s. Obs: %s: %s",
                        iteration,
                        player,
                        depth,
                        agent_idx,
                        action,
                        infoset_key,
                        observation,
                        e,
                    )
                    agent_update_failed = True
                    break
                next_agent_states.append(cloned_agent)

            if agent_update_failed:
                action_log_entry["outcome"] = "ERROR updating agent state"
                action_log.append(action_log_entry)
                if undo_info:
                    undo_info()  # Try to undo game state change
                action_utilities[i] = np.zeros(self.num_players, dtype=np.float64)
                continue  # Move to next action

            # Calculate next reach probabilities
            next_reach_probs = reach_probs.copy()
            next_reach_probs[player] *= action_prob

            # Recursive call
            try:
                recursive_utilities = self._cfr_recursive(
                    game_state,
                    next_agent_states,
                    next_reach_probs,
                    iteration,
                    action_log,
                    status_bar,
                    depth + 1,
                    shutdown_event=shutdown_event,
                )
                action_utilities[i] = recursive_utilities
            except GracefulShutdownException as shutdown_exc:
                logger.debug(
                    "Iter %d, P%d, Depth %d: Graceful shutdown caught during recursion for action %s.",
                    iteration,
                    player,
                    depth,
                    action,
                )
                if undo_info:
                    try:
                        undo_info()
                    except Exception as undo_e:
                        logger.error(
                            "Error undoing action during shutdown propagation: %s", undo_e
                        )
                raise shutdown_exc  # Propagate up
            except RecursionError as rec_err:
                logger.error(
                    "Iter %d, P%d, Depth %d: Recursion depth exceeded during action: %s. State: %s",
                    iteration,
                    player,
                    depth,
                    action,
                    game_state,
                )
                action_log_entry["outcome"] = f"ERROR: RecursionError - {rec_err}"
                action_log.append(action_log_entry)
                if undo_info:
                    undo_info()
                raise  # Propagate recursion error up
            except Exception as recursive_err:
                logger.exception(
                    "Iter %d, P%d, Depth %d: Error in recursive call for action %s. Infoset: %s: %s",
                    iteration,
                    player,
                    depth,
                    action,
                    infoset_key,
                    recursive_err,
                )
                action_log_entry["outcome"] = f"ERROR in recursion: {recursive_err}"
                action_log.append(action_log_entry)
                if undo_info:
                    undo_info()
                action_utilities[i] = np.zeros(self.num_players, dtype=np.float64)
                continue  # Move to next action

            # Log action outcome after successful recursion
            action_log_entry["outcome_utilities"] = action_utilities[i].tolist()
            action_log_entry["state_desc_after"] = state_after_action_desc
            # Append to log passed by reference (done implicitly by caller if no exception)

            # Undo action to backtrack
            if undo_info:
                try:
                    undo_info()
                except Exception as undo_e:
                    logger.exception(
                        "FATAL: Iter %d, P%d, Depth %d: Error undoing action %s. State corrupt: %s",
                        iteration,
                        player,
                        depth,
                        action,
                        undo_e,
                    )
                    return np.zeros(
                        self.num_players, dtype=np.float64
                    )  # Avoid using bad values
            else:
                logger.error(
                    "Iter %d, P%d, Depth %d: Missing undo info for action %s after successful recursion. State corrupt.",
                    iteration,
                    player,
                    depth,
                    action,
                )
                return np.zeros(self.num_players, dtype=np.float64)

        # --- Calculate Node Value & Update Regrets ---
        node_value = np.sum(strategy[:, np.newaxis] * action_utilities, axis=0)
        opponent_reach = reach_probs[opponent]

        # Update regrets for the current player
        if player_reach > 1e-9:
            player_action_values = action_utilities[:, player]
            player_node_value = node_value[player]
            instantaneous_regret = player_action_values - player_node_value

            # Weight regret update by opponent's reach probability
            update_weight = opponent_reach
            current_regrets_before_update = np.copy(self.regret_sum[infoset_key])

            try:
                updated_regrets = (
                    current_regrets_before_update + update_weight * instantaneous_regret
                )
                # Apply Regret Matching+: Ensure regrets are non-negative
                self.regret_sum[infoset_key] = np.maximum(0.0, updated_regrets)
            except ValueError as e_regret:  # Catch potential dimension mismatches
                logger.exception(
                    "Iter %d, P%d, Depth %d, Infoset %s, Context %s: ValueError during regret update: %s. CurrentRegrets: %s, InstantRegret: %s",
                    iteration,
                    player,
                    depth,
                    infoset_key,
                    current_context.name,
                    e_regret,
                    current_regrets_before_update,
                    instantaneous_regret,
                )
            except Exception as e_regret:
                logger.exception(
                    "Iter %d, P%d, Depth %d, Infoset %s, Context %s: Unexpected error during regret update: %s",
                    iteration,
                    player,
                    depth,
                    infoset_key,
                    current_context.name,
                    e_regret,
                )

        return node_value

    def _filter_observation(
        self, obs: AgentObservation, observer_id: int
    ) -> AgentObservation:
        """Creates a player-specific view of the observation, masking private info."""
        filtered_obs = copy.copy(obs)  # Shallow copy first

        # Mask drawn card unless observer is the actor OR it was just publicly revealed
        if obs.drawn_card and obs.acting_player != observer_id:
            # Public reveal happens on Discard/Replace actions
            if not isinstance(obs.action, (ActionDiscard, ActionReplace)):
                filtered_obs.drawn_card = None

        # Filter peeked cards
        if obs.peeked_cards:
            new_peeked = {}
            # Only include peeks relevant to the observer
            if (
                isinstance(obs.action, ActionAbilityPeekOwnSelect)
                and obs.acting_player == observer_id
            ):
                new_peeked = obs.peeked_cards  # Observer sees their own peek result
            elif (
                isinstance(obs.action, ActionAbilityPeekOtherSelect)
                and obs.acting_player == observer_id
            ):
                new_peeked = obs.peeked_cards  # Observer sees their opponent peek result
            elif (
                isinstance(obs.action, ActionAbilityKingLookSelect)
                and obs.acting_player == observer_id
            ):
                new_peeked = obs.peeked_cards  # Observer sees both cards they looked at
            # King Swap Decision doesn't reveal peeked cards in *this* observation
            # Agent relies on memory from the LookSelect step's observation.

            filtered_obs.peeked_cards = new_peeked if new_peeked else None
        else:
            filtered_obs.peeked_cards = None

        # Snap results are public
        filtered_obs.snap_results = obs.snap_results

        return filtered_obs

    def _create_observation(
        self,
        prev_state: Optional[CambiaGameState],  # Not currently used
        action: Optional[GameAction],
        next_state: CambiaGameState,
        acting_player: int,  # Player who took the action
        snap_results: List[Dict],  # Snap results during this step
        explicit_drawn_card: Optional[
            CardObject
        ] = None,  # Card drawn if action was Draw...
    ) -> AgentObservation:
        """Creates the AgentObservation object based on the state *after* the action."""
        discard_top = next_state.get_discard_top()
        hand_sizes = [
            next_state.get_player_card_count(i) for i in range(self.num_players)
        ]
        stock_size = next_state.get_stockpile_size()
        cambia_called = next_state.cambia_caller_id is not None
        who_called = next_state.cambia_caller_id
        game_over = next_state.is_terminal()
        turn_num = next_state.get_turn_number()

        drawn_card_for_obs = None
        if isinstance(action, (ActionDiscard, ActionReplace)):
            # Drawn card is the one passed explicitly (it was either discarded or used for replace)
            drawn_card_for_obs = explicit_drawn_card
        elif explicit_drawn_card is not None and acting_player != -1:
            # Drawn card passed explicitly from a Draw action
            drawn_card_for_obs = explicit_drawn_card

        peeked_cards_dict = None
        # Populate peeked cards *only* if the action was a peek/look type by the actor
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
            # Re-fetch cards looked at for the observation (ideally passed from engine)
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

        # Use the provided snap results for this specific step
        final_snap_results = snap_results

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
        return obs
