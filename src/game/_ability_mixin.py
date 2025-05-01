"""
src/game/_ability_mixin.py

This module implements card ability mixins for the game engine
"""

import logging
import copy
from typing import Set, Deque, Optional

# Use relative imports
from .types import StateDelta
from .helpers import card_has_discard_ability, serialize_card
from ..card import Card
from ..constants import (
    GameAction,
    KING,
    QUEEN,
    JACK,
    NINE,
    TEN,
    SEVEN,
    EIGHT,
    ActionDiscard,
    ActionReplace,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionSnapOpponentMove,
)

logger = logging.getLogger(__name__)


class AbilityMixin:
    """Mixin handling pending actions and card abilities for CambiaGameState."""

    # --- Pending Action Legal Actions ---

    def _get_legal_pending_actions(self, acting_player: int) -> Set[GameAction]:
        """Calculates legal actions when a pending action exists for the acting player."""
        legal_actions: Set[GameAction] = set()

        if not self.pending_action:
            logger.error(
                "Called _get_legal_pending_actions when no pending action exists."
            )
            return legal_actions

        if acting_player != self.pending_action_player:
            logger.error(
                "Legal actions requested for P%d but pending action is for P%d", acting_player, self.pending_action_player
            )
            return legal_actions  # Wrong player

        # Validate player object (should be guaranteed by get_acting_player)
        if not (
            0 <= acting_player < len(self.players)
            and hasattr(self.players[acting_player], "hand")
        ):
            logger.error(
                "Pending action legal actions: Acting player P%d invalid or missing hand.", acting_player
            )
            return legal_actions

        action_type = (
            self.pending_action
        )  # The *type* of action we are waiting for resolution on
        player = acting_player

        # --- Determine actions based on the *pending* action type ---
        if isinstance(
            action_type, ActionDiscard
        ):  # Waiting for Post-Draw Choice (Discard/Replace)
            legal_actions.add(
                ActionDiscard(use_ability=False)
            )  # Always allow simple discard
            drawn_card = self.pending_action_data.get("drawn_card")
            if drawn_card and card_has_discard_ability(
                drawn_card
            ):  # Check if ability can be used
                legal_actions.add(ActionDiscard(use_ability=True))
            # Allow replacing any card in hand
            for i in range(
                self.get_player_card_count(player)
            ):  # get_player_card_count from QueryMixin/base
                legal_actions.add(ActionReplace(target_hand_index=i))

        elif isinstance(
            action_type, ActionAbilityPeekOwnSelect
        ):  # Waiting for 7/8 Peek Choice
            for i in range(self.get_player_card_count(player)):
                legal_actions.add(ActionAbilityPeekOwnSelect(target_hand_index=i))

        elif isinstance(
            action_type, ActionAbilityPeekOtherSelect
        ):  # Waiting for 9/T Peek Choice
            opp_idx = self.get_opponent_index(player)  # Method from QueryMixin/base
            for i in range(self.get_player_card_count(opp_idx)):
                legal_actions.add(
                    ActionAbilityPeekOtherSelect(target_opponent_hand_index=i)
                )

        elif isinstance(
            action_type, ActionAbilityBlindSwapSelect
        ):  # Waiting for J/Q Swap Choice
            own_count = self.get_player_card_count(player)
            opp_idx = self.get_opponent_index(player)
            opp_count = self.get_player_card_count(opp_idx)
            for i in range(own_count):
                for j in range(opp_count):
                    legal_actions.add(
                        ActionAbilityBlindSwapSelect(
                            own_hand_index=i, opponent_hand_index=j
                        )
                    )

        elif isinstance(
            action_type, ActionAbilityKingLookSelect
        ):  # Waiting for K Look Choice
            own_count = self.get_player_card_count(player)
            opp_idx = self.get_opponent_index(player)
            opp_count = self.get_player_card_count(opp_idx)
            for i in range(own_count):
                for j in range(opp_count):
                    legal_actions.add(
                        ActionAbilityKingLookSelect(
                            own_hand_index=i, opponent_hand_index=j
                        )
                    )

        elif isinstance(
            action_type, ActionAbilityKingSwapDecision
        ):  # Waiting for K Swap Decision
            legal_actions.add(ActionAbilityKingSwapDecision(perform_swap=True))
            legal_actions.add(ActionAbilityKingSwapDecision(perform_swap=False))

        elif isinstance(
            action_type, ActionSnapOpponentMove
        ):  # Waiting for Snap Opponent Move Choice
            snapper_idx = player
            target_slot = self.pending_action_data.get("target_empty_slot_index")
            if target_slot is None:
                logger.error(
                    "Missing target_empty_slot_index for legal SnapOpponentMove actions."
                )
            else:
                # Player can move any of their current cards to the empty slot
                for i in range(self.get_player_card_count(snapper_idx)):
                    legal_actions.add(
                        ActionSnapOpponentMove(
                            own_card_to_move_hand_index=i,
                            target_empty_slot_index=target_slot,
                        )
                    )

        else:
            logger.error(
                "Unknown pending action type (%s) encountered for legal actions.", action_type
            )

        return legal_actions

    # --- Pending Action / Ability Processing ---

    def _handle_pending_action(
        self,
        action: GameAction,
        acting_player: int,
        undo_stack: Deque,
        delta_list: StateDelta,
    ) -> Optional[Card]:
        """
        Processes an action intended to resolve a pending game state (e.g., discard/replace, ability choice).
        Modifies state via _add_change.
        Returns the card that was effectively discarded *this turn* (for snap checks), or None.
        Returns None if action is invalid for the pending state or player.
        """
        if not self.pending_action:
            logger.error("_handle_pending_action called when no action is pending.")
            return None
        if acting_player != self.pending_action_player:
            logger.error(
                "Action %s received from P%d but pending action is for P%d", action, acting_player, self.pending_action_player
            )
            return None  # Action ignored

        pending_type = self.pending_action  # The type of action we were waiting for
        player = self.pending_action_player
        card_just_discarded_for_snap_check: Optional[Card] = (
            None  # Card ending up on discard pile *now*
        )

        # --- Handle Post-Draw Choices (Discard/Replace) ---
        # Check if the current pending state expects a discard/replace decision
        if isinstance(pending_type, ActionDiscard):
            drawn_card = self.pending_action_data.get("drawn_card")
            if not drawn_card:
                logger.error(
                    "Pending post-draw choice but no drawn_card in pending data!"
                )
                self._clear_pending_action(undo_stack, delta_list)  # Clear invalid state
                return None

            if isinstance(
                action, ActionDiscard
            ):  # Player chose to discard the drawn card
                logger.debug(
                    "Player %s discards drawn %s. Use ability: %s", player, drawn_card, action.use_ability
                )
                # --- State Change: Add drawn card to discard ---
                original_discard_len = len(self.discard_pile)

                def change_discard():
                    self.discard_pile.append(drawn_card)

                def undo_discard():
                    self.discard_pile.pop()  # Assumes it was last added

                self._add_change(
                    change_discard,
                    undo_discard,
                    (
                        "list_append",
                        "discard_pile",
                        original_discard_len,
                        serialize_card(drawn_card),
                    ),
                    undo_stack,
                    delta_list,
                )
                # --- End State Change ---
                card_just_discarded_for_snap_check = drawn_card
                use_ability = action.use_ability and card_has_discard_ability(drawn_card)

                # Clear pending state *before* potentially triggering ability (which sets a new pending state)
                self._clear_pending_action(undo_stack, delta_list)

                if use_ability:
                    # This might set a *new* pending action state (e.g., for King look)
                    self._trigger_discard_ability(
                        player, drawn_card, undo_stack, delta_list
                    )
                # If ability sets a new pending state, the turn doesn't advance yet.
                # If no ability or ability doesn't require choice, turn *should* advance (or snap check happens).

            elif isinstance(action, ActionReplace):  # Player chose to replace a hand card
                target_idx = action.target_hand_index
                hand = self.players[player].hand
                if 0 <= target_idx < len(hand):
                    replaced_card = hand[target_idx]  # The card currently in hand
                    logger.debug(
                        "Player %d replaces card at index %d (%s) with drawn %s.", player, target_idx, replaced_card, drawn_card
                    )
                    # --- State Change: Swap card in hand, discard old one ---
                    original_card_in_hand = hand[
                        target_idx
                    ]  # Should be same as replaced_card
                    original_discard_len = len(self.discard_pile)

                    def change_replace():
                        self.players[player].hand[target_idx] = drawn_card
                        self.discard_pile.append(replaced_card)

                    def undo_replace():
                        popped_discard = (
                            self.discard_pile.pop()
                        )  # Should be replaced_card
                        self.players[player].hand[target_idx] = original_card_in_hand
                        if popped_discard is not replaced_card and serialize_card(
                            popped_discard
                        ) != serialize_card(replaced_card):
                            logger.error(
                                "Undo Replace Mismatch: Popped %s, expected %s", popped_discard, replaced_card
                            )

                    delta_replace = (
                        "replace_discard",
                        player,
                        target_idx,
                        serialize_card(drawn_card),
                        serialize_card(replaced_card),
                    )
                    self._add_change(
                        change_replace,
                        undo_replace,
                        delta_replace,
                        undo_stack,
                        delta_list,
                    )
                    # --- End State Change ---
                    card_just_discarded_for_snap_check = (
                        replaced_card  # This is the card hitting the discard pile
                    )
                    self._clear_pending_action(
                        undo_stack, delta_list
                    )  # Clear the draw/replace pending state
                else:
                    logger.error(
                        "Invalid REPLACE action index: %d for hand size %d", target_idx, len(hand)
                    )
                    # Don't clear pending state? Or clear and penalize? Let's clear and log.
                    self._clear_pending_action(undo_stack, delta_list)
                    return None  # Indicate invalid action choice

            else:  # Received action doesn't match expected Discard/Replace
                logger.warning(
                    "Received action %s while expecting post-draw Discard/Replace.", action
                )
                return None  # Invalid action for this state

        # --- Handle Ability Selections ---
        elif isinstance(pending_type, ActionAbilityPeekOwnSelect) and isinstance(
            action, ActionAbilityPeekOwnSelect
        ):
            target_idx = action.target_hand_index
            hand = self.players[player].hand
            peeked_card_str = "ERROR"
            if 0 <= target_idx < len(hand):
                peeked_card = hand[target_idx]
                if isinstance(peeked_card, Card):
                    peeked_card_str = serialize_card(peeked_card)
                    logger.info(
                        "P%d uses 7/8 ability, peeks own card %d: %s", player, target_idx, peeked_card_str
                    )
                else:
                    logger.error(
                        "Peek Own Error: Item at index %d is not Card: %s", target_idx, peeked_card
                    )
            else:
                logger.error(
                    "Invalid PEEK_OWN index %d for hand size %d", target_idx, len(hand)
                )
            # Log the peek event for informational purposes (no state *change*, but part of game flow)
            delta_list.append(
                ("peek_own", player, target_idx, peeked_card_str)
            )  # Delta type 'peek_own'
            card_just_discarded_for_snap_check = (
                self.get_discard_top()
            )  # Ability triggered by this card
            self._clear_pending_action(
                undo_stack, delta_list
            )  # Ability resolution complete

        elif isinstance(pending_type, ActionAbilityPeekOtherSelect) and isinstance(
            action, ActionAbilityPeekOtherSelect
        ):
            opp_idx = self.get_opponent_index(player)
            target_opp_idx = action.target_opponent_hand_index
            peeked_card_str = "ERROR"
            if 0 <= opp_idx < len(self.players) and hasattr(
                self.players[opp_idx], "hand"
            ):
                opp_hand = self.players[opp_idx].hand
                if 0 <= target_opp_idx < len(opp_hand):
                    peeked_card = opp_hand[target_opp_idx]
                    if isinstance(peeked_card, Card):
                        peeked_card_str = serialize_card(peeked_card)
                        logger.info(
                            "P%d uses 9/T ability, peeks opponent card %d: %s", player, target_opp_idx, peeked_card_str
                        )
                    else:
                        logger.error(
                            "Peek Other Error: Item at opponent index %d is not Card: %s", target_opp_idx, peeked_card
                        )
                else:
                    logger.error(
                        "Invalid PEEK_OTHER index %d for opponent hand size %d", target_opp_idx, len(opp_hand)
                    )
            else:
                logger.error(
                    "Peek Other Error: Opponent %d invalid or missing hand.", opp_idx
                )
            delta_list.append(
                ("peek_other", player, opp_idx, target_opp_idx, peeked_card_str)
            )  # Delta type 'peek_other'
            card_just_discarded_for_snap_check = self.get_discard_top()
            self._clear_pending_action(undo_stack, delta_list)

        elif isinstance(pending_type, ActionAbilityBlindSwapSelect) and isinstance(
            action, ActionAbilityBlindSwapSelect
        ):
            own_h_idx, opp_h_idx = action.own_hand_index, action.opponent_hand_index
            opp_idx = self.get_opponent_index(player)
            hand = self.players[player].hand
            # valid_swap = False
            if 0 <= opp_idx < len(self.players) and hasattr(
                self.players[opp_idx], "hand"
            ):
                opp_hand = self.players[opp_idx].hand
                if 0 <= own_h_idx < len(hand) and 0 <= opp_h_idx < len(opp_hand):
                    original_own_card = hand[own_h_idx]
                    original_opp_card = opp_hand[opp_h_idx]
                    # Basic validation
                    if isinstance(original_own_card, Card) and isinstance(
                        original_opp_card, Card
                    ):
                        # --- State Change: Swap cards ---
                        def change_blind_swap():
                            hand[own_h_idx], opp_hand[opp_h_idx] = (
                                opp_hand[opp_h_idx],
                                hand[own_h_idx],
                            )

                        def undo_blind_swap():
                            hand[own_h_idx], opp_hand[opp_h_idx] = (
                                original_own_card,
                                original_opp_card,
                            )  # Restore original objects

                        delta_blind_swap = (
                            "swap_blind",
                            player,
                            own_h_idx,
                            opp_idx,
                            opp_h_idx,
                            serialize_card(original_own_card),
                            serialize_card(original_opp_card),
                        )
                        self._add_change(
                            change_blind_swap,
                            undo_blind_swap,
                            delta_blind_swap,
                            undo_stack,
                            delta_list,
                        )
                        # --- End State Change ---
                        logger.info(
                            "P%d uses J/Q ability, blind swaps own %d (%s) with opp %d (%s).", player, own_h_idx, original_own_card, opp_h_idx, original_opp_card
                        )
                        # valid_swap = True
                    else:
                        logger.error(
                            "Blind Swap Error: Cards involved are not valid Card objects (%s, %s)", original_own_card, original_opp_card
                        )
                else:
                    logger.error(
                        "Invalid BLIND_SWAP indices: own %d (max %d), opp %d (max %d)", own_h_idx, len(hand)-1, opp_h_idx, len(opp_hand)-1
                    )
            else:
                logger.error(
                    "Blind Swap Error: Opponent %d invalid or missing hand.", opp_idx
                )

            card_just_discarded_for_snap_check = self.get_discard_top()
            self._clear_pending_action(
                undo_stack, delta_list
            )  # Clear pending state regardless of swap success

        elif isinstance(pending_type, ActionAbilityKingLookSelect) and isinstance(
            action, ActionAbilityKingLookSelect
        ):
            # This action *selects* which cards to look at, leading to the *next* pending state (swap decision).
            own_h_idx, opp_h_idx = action.own_hand_index, action.opponent_hand_index
            opp_idx = self.get_opponent_index(player)
            hand = self.players[player].hand
            can_proceed = False
            card1_str, card2_str = "ERROR", "ERROR"
            card1, card2 = None, None

            if 0 <= opp_idx < len(self.players) and hasattr(
                self.players[opp_idx], "hand"
            ):
                opp_hand = self.players[opp_idx].hand
                if 0 <= own_h_idx < len(hand) and 0 <= opp_h_idx < len(opp_hand):
                    card1, card2 = hand[own_h_idx], opp_hand[opp_h_idx]
                    # Basic validation
                    if isinstance(card1, Card) and isinstance(card2, Card):
                        card1_str, card2_str = serialize_card(card1), serialize_card(
                            card2
                        )
                        logger.info(
                            "P%d uses K ability, looks at own %d (%s) and opp %d (%s). Waiting for swap decision.", player, own_h_idx, card1_str, opp_h_idx, card2_str
                        )
                        can_proceed = True
                    else:
                        logger.error(
                            "King Look Error: Cards involved are not valid Card objects (%s, %s)", card1, card2
                        )
                else:
                    logger.error(
                        "Invalid KING_LOOK indices: own %d (max %d), opp %d (max %d)", own_h_idx, len(hand)-1, opp_h_idx, len(opp_hand)-1
                    )
            else:
                logger.error(
                    "King Look Error: Opponent %s invalid or missing hand.", opp_idx
                )

            if can_proceed:
                # --- State Change: Set next pending action (Swap Decision) ---
                original_pending = (
                    self.pending_action,
                    self.pending_action_player,
                    copy.deepcopy(self.pending_action_data),
                )
                new_pending_data = {
                    "own_idx": own_h_idx,
                    "opp_idx": opp_h_idx,
                    "card1": card1,
                    "card2": card2,
                }  # Store actual card objects looked at
                next_pending_action_type = ActionAbilityKingSwapDecision(
                    perform_swap=False
                )  # Placeholder type

                def change_king_pending():
                    self.pending_action = next_pending_action_type
                    self.pending_action_player = player
                    self.pending_action_data = new_pending_data

                def undo_king_pending():
                    (
                        self.pending_action,
                        self.pending_action_player,
                        self.pending_action_data,
                    ) = original_pending

                # Serialize cards only for the delta log
                delta_king_pending = (
                    "set_pending_action",
                    type(next_pending_action_type).__name__,
                    player,
                    {
                        "own_idx": own_h_idx,
                        "opp_idx": opp_h_idx,
                        "card1": card1_str,
                        "card2": card2_str,
                    },
                    (type(original_pending[0]).__name__ if original_pending[0] else None),
                    original_pending[1],
                    original_pending[2],
                )
                self._add_change(
                    change_king_pending,
                    undo_king_pending,
                    delta_king_pending,
                    undo_stack,
                    delta_list,
                )
                # --- End State Change ---
                # Also log the 'look' itself as an informational delta
                delta_list.append(
                    ("king_look", player, own_h_idx, opp_h_idx, card1_str, card2_str)
                )
                # Return None for discard check, as the turn isn't finished yet.
                return None
            else:
                # Ability fizzles, clear the pending state
                card_just_discarded_for_snap_check = self.get_discard_top()
                self._clear_pending_action(undo_stack, delta_list)

        elif isinstance(pending_type, ActionAbilityKingSwapDecision) and isinstance(
            action, ActionAbilityKingSwapDecision
        ):
            # This action makes the swap decision based on the cards looked at previously.
            perform_swap = action.perform_swap
            look_data = self.pending_action_data
            own_h_idx, opp_h_idx = look_data.get("own_idx"), look_data.get("opp_idx")
            card1, card2 = look_data.get("card1"), look_data.get(
                "card2"
            )  # Card objects stored from look phase

            if own_h_idx is None or opp_h_idx is None or card1 is None or card2 is None:
                logger.error("Missing data for King Swap decision. Ability fizzles.")
            else:
                opp_idx = self.get_opponent_index(player)
                hand = self.players[player].hand
                # valid_context = False
                if 0 <= opp_idx < len(self.players) and hasattr(
                    self.players[opp_idx], "hand"
                ):
                    opp_hand = self.players[opp_idx].hand
                    # --- Sanity Check: Verify cards are still where they should be ---
                    own_idx_valid = 0 <= own_h_idx < len(hand)
                    opp_idx_valid = 0 <= opp_h_idx < len(opp_hand)
                    card_at_own_idx = hand[own_h_idx] if own_idx_valid else None
                    card_at_opp_idx = opp_hand[opp_h_idx] if opp_idx_valid else None
                    # Use 'is' for object identity comparison - did intervening actions change the hand?
                    own_card_match = own_idx_valid and (card_at_own_idx is card1)
                    opp_card_match = opp_idx_valid and (card_at_opp_idx is card2)

                    if own_card_match and opp_card_match:
                        # valid_context = True
                        if perform_swap:
                            # --- State Change: Perform the swap ---
                            def change_king_swap():
                                hand[own_h_idx], opp_hand[opp_h_idx] = card2, card1

                            def undo_king_swap():
                                hand[own_h_idx], opp_hand[opp_h_idx] = card1, card2

                            delta_king_swap = (
                                "swap_king",
                                player,
                                own_h_idx,
                                opp_idx,
                                opp_h_idx,
                                serialize_card(card1),
                                serialize_card(card2),
                            )
                            self._add_change(
                                change_king_swap,
                                undo_king_swap,
                                delta_king_swap,
                                undo_stack,
                                delta_list,
                            )
                            # --- End State Change ---
                            logger.info(
                                "P%d King ability: Swapped own %d (%s) with opp %d (%s).", player, own_h_idx, card1, opp_h_idx, card2
                            )
                        else:
                            logger.info(
                                "P%d King ability: Chose not to swap cards.", player
                            )
                    else:
                        # Log detailed reason if cards changed or indices became invalid
                        reason = "Swap cancelled:"
                        if not own_idx_valid:
                            reason += f" Own index {own_h_idx} invalid (Hand size: {len(hand)})."
                        elif not own_card_match:
                            reason += f" Card at own index {own_h_idx} changed (Expected: {card1}, Found: {card_at_own_idx})."
                        if not opp_idx_valid:
                            reason += f" Opponent index {opp_h_idx} invalid (Hand size: {len(opp_hand)})."
                        elif not opp_card_match:
                            reason += f" Card at opponent index {opp_h_idx} changed (Expected: {card2}, Found: {card_at_opp_idx})."
                        logger.warning("King Swap Error: %s Ability fizzles.", reason)
                else:
                    logger.error(
                        "King Swap Error: Opponent %d invalid or missing hand.", opp_idx
                    )

            card_just_discarded_for_snap_check = (
                self.get_discard_top()
            )  # The original King discard triggered this chain
            self._clear_pending_action(
                undo_stack, delta_list
            )  # Clear the swap decision pending state

        elif isinstance(pending_type, ActionSnapOpponentMove) and isinstance(
            action, ActionSnapOpponentMove
        ):
            # This resolves the pending state created by a successful ActionSnapOpponent
            snapper_idx = player
            own_card_idx = action.own_card_to_move_hand_index
            # Target slot must match the one stored in pending data
            target_slot_idx = self.pending_action_data.get("target_empty_slot_index")
            if target_slot_idx is None:
                logger.error(
                    "Snap Opponent Move Error: Missing target_empty_slot_index in pending data."
                )
                self._clear_pending_action(
                    undo_stack, delta_list
                )  # Clear inconsistent state
                # Turn should advance here? Let main loop handle it.
                return None

            # Verify action matches the required target slot
            if action.target_empty_slot_index != target_slot_idx:
                logger.error(
                    "Snap Opponent Move Error: Action target slot (%d) does not match pending data target (%d).", action.target_empty_slot_index, target_slot_idx
                )
                # Do not clear pending action - player needs to submit correct action.
                return None  # Invalid action

            hand = self.players[snapper_idx].hand
            opp_idx = self.get_opponent_index(snapper_idx)

            if 0 <= opp_idx < len(self.players) and hasattr(
                self.players[opp_idx], "hand"
            ):
                opp_hand = self.players[opp_idx].hand
                if 0 <= own_card_idx < len(hand):
                    card_to_move = hand[own_card_idx]
                    # Check target slot is valid for insertion (should be, as it was a valid index before card removal)
                    if 0 <= target_slot_idx <= len(opp_hand):  # Allow insert at end
                        # --- State Change: Move card ---
                        original_card = hand[own_card_idx]  # Should be card_to_move

                        def change_move():
                            moved_card = self.players[snapper_idx].hand.pop(own_card_idx)
                            self.players[opp_idx].hand.insert(target_slot_idx, moved_card)

                        def undo_move():
                            # This assumes card was inserted correctly at target_slot_idx
                            card = self.players[opp_idx].hand.pop(target_slot_idx)
                            self.players[snapper_idx].hand.insert(own_card_idx, card)
                            if card is not original_card:
                                logger.warning("Undo SnapOpponentMove mismatch")

                        delta_move = (
                            "snap_opponent_move",
                            snapper_idx,
                            own_card_idx,
                            opp_idx,
                            target_slot_idx,
                            serialize_card(card_to_move),
                        )
                        self._add_change(
                            change_move, undo_move, delta_move, undo_stack, delta_list
                        )
                        # --- End State Change ---
                        logger.info(
                            "P%d completes Snap Opponent: Moves %s (from own idx %d) to opp idx %d.", snapper_idx, card_to_move, own_card_idx, target_slot_idx
                        )
                        self._clear_pending_action(undo_stack, delta_list)
                        # Turn advances *after* this move is complete (handled in main apply_action loop)
                        # The discard that *triggered* the original snap is the relevant one for snap checks, not this move.
                        # However, no snap check happens after this move action resolves.
                        return None  # No card discarded *now*
                    else:
                        logger.error(
                            "Invalid target slot index %d for SnapOpponentMove (Opp Hand Size: %d). Should not happen.", target_slot_idx, len(opp_hand)
                        )
                else:
                    logger.error(
                        "Invalid own card index %d for SnapOpponentMove (Own Hand Size: %d).", own_card_idx, len(hand)
                    )
            else:
                logger.error("Snap Opponent Move Error: Opponent %d invalid.", opp_idx)

            # If move failed validation, clear pending state to avoid getting stuck? Risky.
            # Let's assume for now if validation fails, it returns None and state remains pending.
            # self._clear_pending_action(undo_stack, delta_list)
            return None

        else:  # Mismatch between pending state and action received
            logger.warning(
                "Unhandled pending action (%s) vs received action (%s)", pending_type, action
            )
            # Should we clear pending state? Probably safest to avoid getting stuck.
            self._clear_pending_action(undo_stack, delta_list)
            return None  # Indicate action was not processed as expected

        # Return the card that was discarded *if* this action completed a discard/replace cycle.
        return card_just_discarded_for_snap_check

    # --- Ability Triggering and State Management ---

    def _trigger_discard_ability(
        self,
        player_index: int,
        discarded_card: Card,
        undo_stack: Deque,
        delta_list: StateDelta,
    ):
        """
        Checks if a discarded card has an ability that requires further player choice.
        If so, sets the pending action state accordingly using _add_change.
        """
        rank = discarded_card.rank
        logger.debug(
            "Player %d triggering ability of discarded %s", player_index, discarded_card
        )
        ability_requires_choice = False
        next_pending_action: Optional[GameAction] = (
            None  # The *type* of action needed next
        )

        if rank in [SEVEN, EIGHT]:
            next_pending_action = ActionAbilityPeekOwnSelect(
                target_hand_index=-1
            )  # Placeholder index
            ability_requires_choice = True
        elif rank in [NINE, TEN]:
            next_pending_action = ActionAbilityPeekOtherSelect(
                target_opponent_hand_index=-1
            )  # Placeholder index
            ability_requires_choice = True
        elif rank in [JACK, QUEEN]:
            next_pending_action = ActionAbilityBlindSwapSelect(
                own_hand_index=-1, opponent_hand_index=-1
            )  # Placeholders
            ability_requires_choice = True
        elif rank == KING:
            # King ability has two steps: Look selection, then Swap decision
            next_pending_action = ActionAbilityKingLookSelect(
                own_hand_index=-1, opponent_hand_index=-1
            )  # Placeholder for look
            ability_requires_choice = True
        # Add other abilities here if needed

        if ability_requires_choice and next_pending_action is not None:
            # --- State Change: Set Pending Action for Ability Choice ---
            original_pending_action = self.pending_action
            original_pending_player = self.pending_action_player
            original_pending_data = copy.deepcopy(self.pending_action_data)
            # Store the card that triggered the ability for context if needed
            new_pending_data = {"ability_card": discarded_card}

            def change_ability_pending():
                self.pending_action = next_pending_action
                self.pending_action_player = player_index
                self.pending_action_data = new_pending_data

            def undo_ability_pending():
                self.pending_action = original_pending_action
                self.pending_action_player = original_pending_player
                self.pending_action_data = original_pending_data

            # Serialize complex objects only for delta log
            prev_pending_type_name = (
                type(original_pending_action).__name__
                if original_pending_action
                else None
            )
            serialized_orig_data = {
                k: serialize_card(v) if isinstance(v, Card) else v
                for k, v in original_pending_data.items()
            }
            serialized_new_data = {"ability_card": serialize_card(discarded_card)}
            delta_ability = (
                "set_pending_action",
                type(next_pending_action).__name__,
                player_index,
                serialized_new_data,
                prev_pending_type_name,
                original_pending_player,
                serialized_orig_data,
            )
            self._add_change(
                change_ability_pending,
                undo_ability_pending,
                delta_ability,
                undo_stack,
                delta_list,
            )
            # --- End State Change ---
            logger.debug(
                "Set pending action for P%d to %s due to %s ability.", player_index, type(next_pending_action).__name__, discarded_card
            )
        else:
            logger.debug(
                "Card %s has no discard ability requiring further choice.", discarded_card
            )

    def _clear_pending_action(self, undo_stack: Deque, delta_list: StateDelta):
        """Resets the pending action state, adding undo operation and delta."""
        if (
            self.pending_action is None
            and self.pending_action_player is None
            and not self.pending_action_data
        ):
            return  # Nothing to clear

        # --- State Change: Clear Pending Action ---
        original_pending_action = self.pending_action
        original_pending_player = self.pending_action_player
        original_pending_data = copy.deepcopy(
            self.pending_action_data
        )  # Crucial deep copy

        def change_clear():
            self.pending_action = None
            self.pending_action_player = None
            self.pending_action_data = {}

        def undo_clear():
            self.pending_action = original_pending_action
            self.pending_action_player = original_pending_player
            self.pending_action_data = original_pending_data  # Restore the deep copy

        # Serialize complex objects only for delta log
        orig_pending_type_name = (
            type(original_pending_action).__name__ if original_pending_action else None
        )
        serialized_orig_data = {
            k: serialize_card(v) if isinstance(v, Card) else v
            for k, v in original_pending_data.items()
        }
        delta_clear = (
            "clear_pending_action",
            orig_pending_type_name,
            original_pending_player,
            serialized_orig_data,
        )
        self._add_change(change_clear, undo_clear, delta_clear, undo_stack, delta_list)
        # --- End State Change ---
        logger.debug(
            "Cleared pending action state (was %s for P%d).", orig_pending_type_name, original_pending_player
        )

    def _get_pending_action_name(self) -> str:
        """Helper for __str__."""
        if self.pending_action:
            return type(self.pending_action).__name__
        return "N/A"
