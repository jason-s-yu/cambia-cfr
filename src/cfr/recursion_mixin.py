# src/cfr/recursion_mixin.py
"""
Mixin class originally intended for CFR+ recursive traversal logic.
The core recursion is now handled by src/cfr/worker.py for parallelization.
This mixin retains helper methods related to observation creation/filtering.
"""

import logging
import copy
from typing import Dict, List, Optional

from ..agent_state import AgentObservation
from ..constants import (
    ActionDiscard,
    ActionReplace,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityKingLookSelect,
    GameAction,
    CardObject,
    NUM_PLAYERS,
)

from ..game.engine import CambiaGameState

logger = logging.getLogger(__name__)


class CFRRecursionMixin:
    """
    Provides helper methods related to game state observation creation
    and filtering, previously used by the core CFR recursion.
    The main recursive traversal logic is now in worker.py.
    """

    # Attributes expected to be initialized in the main class's __init__
    # self.config: Config
    # self.num_players: int
    # self.regret_sum: PolicyDict
    # self.strategy_sum: PolicyDict
    # self.reach_prob_sum: ReachProbDict
    # self.shutdown_event: threading.Event
    # self.max_depth_this_iter: int
    # self._last_exploit_str: str

    # --- Observation Helper Methods (Retained for potential reuse) ---

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
        # Use NUM_PLAYERS constant instead of self.num_players if available
        hand_sizes = [next_state.get_player_card_count(i) for i in range(NUM_PLAYERS)]
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
