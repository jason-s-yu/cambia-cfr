"""
src/game/_ability_mixin.py

This module implements card ability mixins for the game engine
"""

import logging
import copy
from typing import Set, Deque, Optional

from .types import StateDelta
from .helpers import card_has_discard_ability, serialize_card
from ..card import Card
from ..constants import (
    KING,
    QUEEN,
    JACK,
    NINE,
    TEN,
    SEVEN,
    EIGHT,
    GameAction,
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

        # Ensure self has necessary attributes before proceeding (basic safety)
        if not all(
            hasattr(self, attr)
            for attr in [
                "pending_action",
                "pending_action_player",
                "players",
                "get_player_card_count",
                "get_opponent_index",
            ]
        ):
            logger.error(
                "AbilityMixin: Missing required attributes for _get_legal_pending_actions."
            )
            return legal_actions

        if not self.pending_action:
            logger.error(
                "Called _get_legal_pending_actions when no pending action exists."
            )
            return legal_actions
        if acting_player != self.pending_action_player:
            logger.error(
                "Legal actions requested for P%d but pending action is for P%d",
                acting_player,
                self.pending_action_player,
            )
            return legal_actions
        if not (
            0 <= acting_player < len(self.players)
            and hasattr(self.players[acting_player], "hand")
        ):
            logger.error(
                "Pending action legal actions: P%d invalid or missing hand.",
                acting_player,
            )
            return legal_actions

        action_type = self.pending_action
        player = acting_player
        try:
            player_hand_count = self.get_player_card_count(
                player
            )  # Method from QueryMixin/base
            opponent_id = self.get_opponent_index(player)
            opponent_hand_count = self.get_player_card_count(opponent_id)
        except Exception as e_count:
            logger.error(
                "Error getting hand counts for legal pending actions P%d: %s",
                player,
                e_count,
            )
            return legal_actions  # Cannot proceed without counts

        try:
            if isinstance(action_type, ActionDiscard):  # Post-Draw Choice
                legal_actions.add(ActionDiscard(use_ability=False))
                drawn_card = self.pending_action_data.get("drawn_card")
                if drawn_card and card_has_discard_ability(drawn_card):
                    legal_actions.add(ActionDiscard(use_ability=True))
                for i in range(player_hand_count):
                    legal_actions.add(ActionReplace(target_hand_index=i))

            elif isinstance(action_type, ActionAbilityPeekOwnSelect):  # 7/8 Peek Choice
                for i in range(player_hand_count):
                    legal_actions.add(ActionAbilityPeekOwnSelect(target_hand_index=i))

            elif isinstance(action_type, ActionAbilityPeekOtherSelect):  # 9/T Peek Choice
                for i in range(opponent_hand_count):
                    legal_actions.add(
                        ActionAbilityPeekOtherSelect(target_opponent_hand_index=i)
                    )

            elif isinstance(action_type, ActionAbilityBlindSwapSelect):  # J/Q Swap Choice
                # Add action only if both hands are non-empty
                if player_hand_count > 0 and opponent_hand_count > 0:
                    for i in range(player_hand_count):
                        for j in range(opponent_hand_count):
                            legal_actions.add(
                                ActionAbilityBlindSwapSelect(
                                    own_hand_index=i, opponent_hand_index=j
                                )
                            )
                else:
                    logger.debug(
                        "Cannot generate BlindSwap actions: P%d has %d cards, Opponent has %d cards.",
                        player,
                        player_hand_count,
                        opponent_hand_count,
                    )

            elif isinstance(action_type, ActionAbilityKingLookSelect):  # K Look Choice
                # Add action only if both hands are non-empty
                if player_hand_count > 0 and opponent_hand_count > 0:
                    for i in range(player_hand_count):
                        for j in range(opponent_hand_count):
                            legal_actions.add(
                                ActionAbilityKingLookSelect(
                                    own_hand_index=i, opponent_hand_index=j
                                )
                            )
                else:
                    logger.debug(
                        "Cannot generate KingLook actions: P%d has %d cards, Opponent has %d cards.",
                        player,
                        player_hand_count,
                        opponent_hand_count,
                    )

            elif isinstance(
                action_type, ActionAbilityKingSwapDecision
            ):  # K Swap Decision
                legal_actions.add(ActionAbilityKingSwapDecision(perform_swap=True))
                legal_actions.add(ActionAbilityKingSwapDecision(perform_swap=False))

            elif isinstance(
                action_type, ActionSnapOpponentMove
            ):  # Snap Opponent Move Choice
                target_slot = self.pending_action_data.get("target_empty_slot_index")
                if target_slot is None or not isinstance(target_slot, int):
                    logger.error(
                        "Missing/invalid target_empty_slot_index for legal SnapOpponentMove actions."
                    )
                # Check if player has cards to move
                elif player_hand_count > 0:
                    for i in range(player_hand_count):
                        legal_actions.add(
                            ActionSnapOpponentMove(
                                own_card_to_move_hand_index=i,
                                target_empty_slot_index=target_slot,
                            )
                        )
                else:
                    logger.debug(
                        "Cannot generate SnapMove actions: P%d has no cards to move.",
                        player,
                    )

            else:
                logger.error(
                    "Unknown pending action type (%s) encountered for legal actions.",
                    type(action_type).__name__,
                )

        except IndexError as e_idx:
            logger.error(
                "Index error generating legal pending actions for %s: %s",
                type(action_type).__name__,
                e_idx,
                exc_info=True,
            )
        except Exception as e_legal:
            logger.error(
                "Unexpected error generating legal pending actions for %s: %s",
                type(action_type).__name__,
                e_legal,
                exc_info=True,
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
        Processes an action resolving a pending state. Modifies state via _add_change.
        Returns the card discarded *this step* for snap checks, or None.
        Returns None if action is invalid for the pending state or player.
        """
        # Ensure self has necessary attributes before proceeding
        if not all(
            hasattr(self, attr)
            for attr in [
                "pending_action",
                "pending_action_player",
                "players",
                "discard_pile",
                "_add_change",
                "_clear_pending_action",
                "_trigger_discard_ability",
                "get_opponent_index",
                "get_discard_top",
            ]
        ):
            logger.critical(
                "AbilityMixin: Missing required attributes for _handle_pending_action."
            )
            return None  # Cannot proceed

        if not self.pending_action:
            logger.error("_handle_pending_action called with no pending action.")
            return None
        if acting_player != self.pending_action_player:
            logger.error(
                "Action %s from P%d but pending action is for P%d",
                action,
                acting_player,
                self.pending_action_player,
            )
            return None

        pending_type = self.pending_action
        player = self.pending_action_player
        card_just_discarded_for_snap_check: Optional[Card] = None

        try:  # Wrap main logic in try-except
            # --- Handle Post-Draw Choices (Discard/Replace) ---
            if isinstance(pending_type, ActionDiscard):
                drawn_card = self.pending_action_data.get("drawn_card")
                if not drawn_card or not isinstance(drawn_card, Card):  # Basic type check
                    logger.error(
                        "Pending post-draw choice but invalid/missing drawn_card in data! Data: %s",
                        self.pending_action_data,
                    )
                    self._clear_pending_action(undo_stack, delta_list)
                    return None

                if isinstance(action, ActionDiscard):
                    logger.debug(
                        "P%d discards drawn %s. Use ability: %s",
                        player,
                        drawn_card,
                        action.use_ability,
                    )
                    original_discard_len = len(self.discard_pile)

                    def change_discard():
                        self.discard_pile.append(drawn_card)

                    def undo_discard():
                        # Check if discard pile ends with the expected card before popping
                        if self.discard_pile and self.discard_pile[-1] is drawn_card:
                            self.discard_pile.pop()
                        else:
                            logger.warning(
                                "Undo Discard mismatch/pile empty. Discard Top: %s, Expected: %s",
                                self.discard_pile[-1] if self.discard_pile else "Empty",
                                drawn_card,
                            )

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

                    card_just_discarded_for_snap_check = drawn_card
                    use_ability = action.use_ability and card_has_discard_ability(
                        drawn_card
                    )
                    self._clear_pending_action(
                        undo_stack, delta_list
                    )  # Clear before ability trigger

                    if use_ability:
                        self._trigger_discard_ability(
                            player, drawn_card, undo_stack, delta_list
                        )

                elif isinstance(action, ActionReplace):
                    target_idx = action.target_hand_index
                    hand = self.players[player].hand
                    if 0 <= target_idx < len(hand):
                        replaced_card = hand[target_idx]
                        if not isinstance(replaced_card, Card):  # Basic type check
                            logger.error(
                                "Replace target index %d holds non-Card object: %s",
                                target_idx,
                                replaced_card,
                            )
                            self._clear_pending_action(
                                undo_stack, delta_list
                            )  # Clear invalid state
                            return None

                        logger.debug(
                            "P%d replaces card at index %d (%s) with drawn %s.",
                            player,
                            target_idx,
                            replaced_card,
                            drawn_card,
                        )
                        original_card_in_hand = hand[
                            target_idx
                        ]  # Should be same object as replaced_card
                        original_discard_len = len(self.discard_pile)

                        def change_replace():
                            self.players[player].hand[target_idx] = drawn_card
                            self.discard_pile.append(replaced_card)

                        def undo_replace():
                            # Check discard pile top matches replaced_card
                            if (
                                self.discard_pile
                                and self.discard_pile[-1] is replaced_card
                            ):
                                self.discard_pile.pop()
                            else:
                                logger.warning(
                                    "Undo Replace discard pop mismatch. Discard Top: %s, Expected: %s",
                                    (
                                        self.discard_pile[-1]
                                        if self.discard_pile
                                        else "Empty"
                                    ),
                                    replaced_card,
                                )
                            # Check hand card matches drawn_card before restoring original
                            if (
                                0 <= target_idx < len(self.players[player].hand)
                                and self.players[player].hand[target_idx] is drawn_card
                            ):
                                self.players[player].hand[
                                    target_idx
                                ] = original_card_in_hand
                            else:
                                logger.warning(
                                    "Undo Replace hand restore mismatch. Hand[%d]: %s, Expected: %s",
                                    target_idx,
                                    (
                                        self.players[player].hand[target_idx]
                                        if 0
                                        <= target_idx
                                        < len(self.players[player].hand)
                                        else "OOB"
                                    ),
                                    drawn_card,
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
                        card_just_discarded_for_snap_check = replaced_card
                        self._clear_pending_action(undo_stack, delta_list)
                    else:
                        logger.error(
                            "Invalid REPLACE action index: %d for hand size %d",
                            target_idx,
                            len(hand),
                        )
                        return None  # Indicate invalid action choice

                else:
                    logger.warning(
                        "Received action %s while expecting post-draw Discard/Replace.",
                        action,
                    )
                    return None

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
                            "P%d uses 7/8 ability, peeks own card %d: %s",
                            player,
                            target_idx,
                            peeked_card_str,
                        )
                    else:
                        logger.error(
                            "Peek Own Error: Item at index %d is not Card: %s",
                            target_idx,
                            peeked_card,
                        )
                else:
                    logger.error(
                        "Invalid PEEK_OWN index %d for hand size %d",
                        target_idx,
                        len(hand),
                    )
                delta_list.append(("peek_own", player, target_idx, peeked_card_str))
                card_just_discarded_for_snap_check = (
                    self.get_discard_top()
                )  # Use discard top as trigger card
                self._clear_pending_action(undo_stack, delta_list)

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
                                "P%d uses 9/T ability, peeks opponent card %d: %s",
                                player,
                                target_opp_idx,
                                peeked_card_str,
                            )
                        else:
                            logger.error(
                                "Peek Other Error: Item at opponent index %d is not Card: %s",
                                target_opp_idx,
                                peeked_card,
                            )
                    else:
                        logger.error(
                            "Invalid PEEK_OTHER index %d for opponent hand size %d",
                            target_opp_idx,
                            len(opp_hand),
                        )
                else:
                    logger.error(
                        "Peek Other Error: Opponent %d invalid or missing hand.", opp_idx
                    )
                delta_list.append(
                    ("peek_other", player, opp_idx, target_opp_idx, peeked_card_str)
                )
                card_just_discarded_for_snap_check = self.get_discard_top()
                self._clear_pending_action(undo_stack, delta_list)

            elif isinstance(pending_type, ActionAbilityBlindSwapSelect) and isinstance(
                action, ActionAbilityBlindSwapSelect
            ):
                own_h_idx, opp_h_idx = action.own_hand_index, action.opponent_hand_index
                opp_idx = self.get_opponent_index(player)
                hand = self.players[player].hand
                swap_successful = False
                original_own_card, original_opp_card = None, None

                if 0 <= opp_idx < len(self.players) and hasattr(
                    self.players[opp_idx], "hand"
                ):
                    opp_hand = self.players[opp_idx].hand
                    if 0 <= own_h_idx < len(hand) and 0 <= opp_h_idx < len(opp_hand):
                        original_own_card = hand[own_h_idx]
                        original_opp_card = opp_hand[opp_h_idx]
                        if isinstance(original_own_card, Card) and isinstance(
                            original_opp_card, Card
                        ):
                            # Capture for undo closure
                            captured_own_card = original_own_card
                            captured_opp_card = original_opp_card

                            def change_blind_swap():
                                hand[own_h_idx], opp_hand[opp_h_idx] = (
                                    captured_opp_card,
                                    captured_own_card,
                                )

                            def undo_blind_swap():
                                if 0 <= own_h_idx < len(hand) and 0 <= opp_h_idx < len(
                                    opp_hand
                                ):
                                    # Check identity before restoring
                                    if (
                                        hand[own_h_idx] is captured_opp_card
                                        and opp_hand[opp_h_idx] is captured_own_card
                                    ):
                                        hand[own_h_idx], opp_hand[opp_h_idx] = (
                                            captured_own_card,
                                            captured_opp_card,
                                        )
                                    else:
                                        logger.warning(
                                            "Undo BlindSwap mismatch: cards changed unexpectedly."
                                        )
                                else:
                                    logger.warning(
                                        "Undo BlindSwap failed: index out of bounds."
                                    )

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
                            swap_successful = True
                        else:
                            logger.error(
                                "Blind Swap Error: Cards involved not valid Card objects (%s, %s)",
                                original_own_card,
                                original_opp_card,
                            )
                    else:
                        logger.error(
                            "Invalid BLIND_SWAP indices: own %d (max %d), opp %d (max %d)",
                            own_h_idx,
                            len(hand) - 1,
                            opp_h_idx,
                            len(opp_hand) - 1,
                        )
                else:
                    logger.error(
                        "Blind Swap Error: Opponent %d invalid or missing hand.", opp_idx
                    )

                if swap_successful:
                    logger.info(
                        "P%d uses J/Q ability, blind swaps own %d (%s) with opp %d (%s).",
                        player,
                        own_h_idx,
                        original_own_card,
                        opp_h_idx,
                        original_opp_card,
                    )

                card_just_discarded_for_snap_check = self.get_discard_top()
                self._clear_pending_action(undo_stack, delta_list)

            elif isinstance(pending_type, ActionAbilityKingLookSelect) and isinstance(
                action, ActionAbilityKingLookSelect
            ):
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
                        if isinstance(card1, Card) and isinstance(card2, Card):
                            card1_str, card2_str = serialize_card(card1), serialize_card(
                                card2
                            )
                            can_proceed = True
                        else:
                            logger.error(
                                "King Look Error: Cards involved not valid Card objects (%s, %s)",
                                card1,
                                card2,
                            )
                    else:
                        logger.error(
                            "Invalid KING_LOOK indices: own %d (max %d), opp %d (max %d)",
                            own_h_idx,
                            len(hand) - 1,
                            opp_h_idx,
                            len(opp_hand) - 1,
                        )
                else:
                    logger.error(
                        "King Look Error: Opponent %s invalid or missing hand.", opp_idx
                    )

                if can_proceed:
                    logger.info(
                        "P%d uses K ability, looks at own %d (%s) and opp %d (%s). Waiting for swap decision.",
                        player,
                        own_h_idx,
                        card1_str,
                        opp_h_idx,
                        card2_str,
                    )
                    original_pending = (
                        self.pending_action,
                        self.pending_action_player,
                        copy.deepcopy(self.pending_action_data),
                    )
                    # Store actual card objects for swap decision phase
                    new_pending_data = {
                        "own_idx": own_h_idx,
                        "opp_idx": opp_h_idx,
                        "card1": card1,
                        "card2": card2,
                    }
                    next_pending_action_type = ActionAbilityKingSwapDecision(
                        perform_swap=False
                    )

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
                        (
                            type(original_pending[0]).__name__
                            if original_pending[0]
                            else None
                        ),
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
                    delta_list.append(
                        ("king_look", player, own_h_idx, opp_h_idx, card1_str, card2_str)
                    )
                    return None  # Turn isn't finished yet
                else:
                    card_just_discarded_for_snap_check = self.get_discard_top()
                    self._clear_pending_action(undo_stack, delta_list)

            elif isinstance(pending_type, ActionAbilityKingSwapDecision) and isinstance(
                action, ActionAbilityKingSwapDecision
            ):
                perform_swap = action.perform_swap
                look_data = self.pending_action_data
                own_h_idx, opp_h_idx = look_data.get("own_idx"), look_data.get("opp_idx")
                card1, card2 = look_data.get("card1"), look_data.get("card2")

                if (
                    own_h_idx is None
                    or opp_h_idx is None
                    or not isinstance(card1, Card)
                    or not isinstance(card2, Card)
                ):
                    logger.error(
                        "Missing/invalid data for King Swap decision. Data: %s. Ability fizzles.",
                        look_data,
                    )
                else:
                    opp_idx = self.get_opponent_index(player)
                    hand = self.players[player].hand
                    swap_context_valid = False
                    reason = "Swap context invalid:"

                    if 0 <= opp_idx < len(self.players) and hasattr(
                        self.players[opp_idx], "hand"
                    ):
                        opp_hand = self.players[opp_idx].hand
                        own_idx_valid = 0 <= own_h_idx < len(hand)
                        opp_idx_valid = 0 <= opp_h_idx < len(opp_hand)
                        card_at_own_idx = hand[own_h_idx] if own_idx_valid else None
                        card_at_opp_idx = opp_hand[opp_h_idx] if opp_idx_valid else None
                        own_card_match = own_idx_valid and (card_at_own_idx is card1)
                        opp_card_match = opp_idx_valid and (card_at_opp_idx is card2)

                        if own_card_match and opp_card_match:
                            swap_context_valid = True
                        else:
                            if not own_idx_valid:
                                reason += (
                                    f" Own idx {own_h_idx} invalid (size {len(hand)})."
                                )
                            elif not own_card_match:
                                reason += f" Own card changed (Expected {card1}, Found {card_at_own_idx})."
                            if not opp_idx_valid:
                                reason += f" Opp idx {opp_h_idx} invalid (size {len(opp_hand)})."
                            elif not opp_card_match:
                                reason += f" Opp card changed (Expected {card2}, Found {card_at_opp_idx})."
                    else:
                        reason += f" Opponent P{opp_idx} invalid."

                    if swap_context_valid:
                        if perform_swap:
                            # Capture card objects for closure
                            captured_card1, captured_card2 = card1, card2

                            def change_king_swap():
                                hand[own_h_idx], opp_hand[opp_h_idx] = (
                                    captured_card2,
                                    captured_card1,
                                )

                            def undo_king_swap():
                                if 0 <= own_h_idx < len(hand) and 0 <= opp_h_idx < len(
                                    opp_hand
                                ):
                                    # Check identity before restoring
                                    if (
                                        hand[own_h_idx] is captured_card2
                                        and opp_hand[opp_h_idx] is captured_card1
                                    ):
                                        hand[own_h_idx], opp_hand[opp_h_idx] = (
                                            captured_card1,
                                            captured_card2,
                                        )
                                    else:
                                        logger.warning(
                                            "Undo KingSwap mismatch: cards changed unexpectedly."
                                        )
                                else:
                                    logger.warning(
                                        "Undo KingSwap failed: index out of bounds."
                                    )

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
                            logger.info(
                                "P%d King ability: Swapped own %d (%s) with opp %d (%s).",
                                player,
                                own_h_idx,
                                card1,
                                opp_h_idx,
                                card2,
                            )
                        else:
                            logger.info(
                                "P%d King ability: Chose not to swap cards.", player
                            )
                            delta_list.append(
                                (
                                    "king_swap_decision",
                                    player,
                                    own_h_idx,
                                    opp_h_idx,
                                    False,
                                    serialize_card(card1),
                                    serialize_card(card2),
                                )
                            )
                    else:
                        logger.warning("King Swap Error: %s Ability fizzles.", reason)

                card_just_discarded_for_snap_check = self.get_discard_top()
                self._clear_pending_action(undo_stack, delta_list)

            elif isinstance(pending_type, ActionSnapOpponentMove) and isinstance(
                action, ActionSnapOpponentMove
            ):
                snapper_idx = player
                own_card_idx = action.own_card_to_move_hand_index
                target_slot_idx = self.pending_action_data.get("target_empty_slot_index")

                if target_slot_idx is None or not isinstance(target_slot_idx, int):
                    logger.error(
                        "SnapOpponentMove Error: Missing/invalid target_empty_slot_index. Data: %s",
                        self.pending_action_data,
                    )
                    self._clear_pending_action(undo_stack, delta_list)
                    return None
                if action.target_empty_slot_index != target_slot_idx:
                    logger.error(
                        "SnapOpponentMove Error: Action target slot (%d) != pending data target (%d).",
                        action.target_empty_slot_index,
                        target_slot_idx,
                    )
                    return None  # Invalid action, wait

                hand = self.players[snapper_idx].hand
                opp_idx = self.get_opponent_index(snapper_idx)
                move_successful = False
                card_to_move = None

                if 0 <= opp_idx < len(self.players) and hasattr(
                    self.players[opp_idx], "hand"
                ):
                    opp_hand = self.players[opp_idx].hand
                    if 0 <= own_card_idx < len(hand):
                        card_to_move = hand[own_card_idx]
                        if not isinstance(card_to_move, Card):
                            logger.error(
                                "SnapOpponentMove Error: Item at own index %d is not Card: %s",
                                own_card_idx,
                                card_to_move,
                            )
                        elif 0 <= target_slot_idx <= len(opp_hand):  # Allow insert at end
                            original_card = card_to_move  # Capture for closure/undo

                            def change_move():
                                moved_card = self.players[snapper_idx].hand.pop(
                                    own_card_idx
                                )
                                self.players[opp_idx].hand.insert(
                                    target_slot_idx, moved_card
                                )

                            def undo_move():
                                if 0 <= target_slot_idx < len(self.players[opp_idx].hand):
                                    moved_back_card = self.players[opp_idx].hand.pop(
                                        target_slot_idx
                                    )
                                    if moved_back_card is original_card:
                                        # Check original index still valid before inserting
                                        if (
                                            0
                                            <= own_card_idx
                                            <= len(self.players[snapper_idx].hand)
                                        ):
                                            self.players[snapper_idx].hand.insert(
                                                own_card_idx, moved_back_card
                                            )
                                        else:
                                            logger.warning(
                                                "Undo SnapOpponentMove insert failed: own index %d OOB",
                                                own_card_idx,
                                            )
                                    else:
                                        logger.warning(
                                            "Undo SnapOpponentMove mismatch: card identity changed."
                                        )
                                else:
                                    logger.warning(
                                        "Undo SnapOpponentMove pop failed: target index %d invalid for opp hand %s.",
                                        target_slot_idx,
                                        self.players[opp_idx].hand,
                                    )

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
                            move_successful = True
                        else:
                            logger.error(
                                "Invalid target slot index %d for SnapOpponentMove (Opp Hand Size: %d).",
                                target_slot_idx,
                                len(opp_hand),
                            )
                    else:
                        logger.error(
                            "Invalid own card index %d for SnapOpponentMove (Own Hand Size: %d).",
                            own_card_idx,
                            len(hand),
                        )
                else:
                    logger.error(
                        "Snap Opponent Move Error: Opponent %d invalid.", opp_idx
                    )

                if move_successful:
                    logger.info(
                        "P%d completes Snap Opponent: Moves %s (from own idx %d) to opp idx %d.",
                        snapper_idx,
                        card_to_move,
                        own_card_idx,
                        target_slot_idx,
                    )
                    self._clear_pending_action(undo_stack, delta_list)
                    return None  # No card discarded now
                else:
                    return None  # Failed validation, require valid action

            else:  # Mismatch between pending state and action received
                logger.warning(
                    "Unhandled pending action type %s vs received action %s",
                    type(pending_type).__name__,
                    type(action).__name__,
                )
                return None  # Indicate invalid action for this state

        except Exception as e_handle:
            logger.exception(
                "Error handling pending action %s for P%d: %s",
                action,
                acting_player,
                e_handle,
            )
            self._clear_pending_action(undo_stack, delta_list)  # Attempt recovery
            return None

        return card_just_discarded_for_snap_check

    # --- Ability Triggering and State Management ---

    def _trigger_discard_ability(
        self,
        player_index: int,
        discarded_card: Card,
        undo_stack: Deque,
        delta_list: StateDelta,
    ):
        """Checks discard ability, sets pending action state if choice needed."""
        # Ensure self has necessary attributes
        if not all(
            hasattr(self, attr)
            for attr in [
                "pending_action",
                "pending_action_player",
                "pending_action_data",
                "_add_change",
            ]
        ):
            logger.critical(
                "AbilityMixin: Missing required attributes for _trigger_discard_ability."
            )
            return

        rank = discarded_card.rank
        logger.debug(
            "P%d triggering ability of discarded %s", player_index, discarded_card
        )
        next_pending_action: Optional[GameAction] = None

        if rank == SEVEN or rank == EIGHT:
            next_pending_action = ActionAbilityPeekOwnSelect(-1)
        elif rank == NINE or rank == TEN:
            next_pending_action = ActionAbilityPeekOtherSelect(-1)
        elif rank == JACK or rank == QUEEN:
            next_pending_action = ActionAbilityBlindSwapSelect(-1, -1)
        elif rank == KING:
            next_pending_action = ActionAbilityKingLookSelect(-1, -1)

        if next_pending_action:
            original_pending_action = self.pending_action
            original_pending_player = self.pending_action_player
            original_pending_data = copy.deepcopy(self.pending_action_data)
            new_pending_data = {
                "ability_card": discarded_card
            }  # Store card for context if needed

            def change_ability_pending():
                self.pending_action = next_pending_action
                self.pending_action_player = player_index
                self.pending_action_data = new_pending_data

            def undo_ability_pending():
                self.pending_action = original_pending_action
                self.pending_action_player = original_pending_player
                self.pending_action_data = original_pending_data  # Restore deepcopy

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
            logger.debug(
                "Set pending action for P%d to %s due to %s ability.",
                player_index,
                type(next_pending_action).__name__,
                discarded_card,
            )
        else:
            logger.debug(
                "Card %s has no discard ability requiring further choice.", discarded_card
            )

    def _clear_pending_action(self, undo_stack: Deque, delta_list: StateDelta):
        """Resets the pending action state, adding undo operation and delta."""
        # Ensure self has necessary attributes
        if not all(
            hasattr(self, attr)
            for attr in [
                "pending_action",
                "pending_action_player",
                "pending_action_data",
                "_add_change",
            ]
        ):
            logger.critical(
                "AbilityMixin: Missing required attributes for _clear_pending_action."
            )
            return

        if (
            self.pending_action is None
            and self.pending_action_player is None
            and not self.pending_action_data
        ):
            return  # Nothing to clear

        original_pending_action = self.pending_action
        original_pending_player = self.pending_action_player
        original_pending_data = copy.deepcopy(
            self.pending_action_data
        )  # Deepcopy essential

        def change_clear():
            self.pending_action = None
            self.pending_action_player = None
            self.pending_action_data = {}

        def undo_clear():
            self.pending_action = original_pending_action
            self.pending_action_player = original_pending_player
            self.pending_action_data = original_pending_data  # Restore deepcopy

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
        logger.debug(
            "Cleared pending action state (was %s for P%s).",
            orig_pending_type_name,
            original_pending_player,
        )

    def _get_pending_action_name(self) -> str:
        """Helper for __str__."""
        pending = getattr(self, "pending_action", None)
        return type(pending).__name__ if pending else "N/A"
