"""
src/game/_ability_mixin.py

Implements card ability logic mixin for the Cambia game engine.
Handles pending actions related to ability resolution.
"""

import logging
import copy
from typing import Set, Deque, Optional, TYPE_CHECKING

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
from src.cfr.exceptions import ActionApplicationError

# Use TYPE_CHECKING for CambiaGameState hint to avoid circular import
if TYPE_CHECKING:
    from .engine import CambiaGameState


logger = logging.getLogger(__name__)


class AbilityMixin:
    """Mixin handling pending actions and card abilities for CambiaGameState."""

    # --- Pending Action Legal Actions ---

    def _get_legal_pending_actions(
        self: "CambiaGameState", acting_player: int
    ) -> Set[GameAction]:
        """Calculates legal actions when a pending action exists for the acting player."""
        legal_actions: Set[GameAction] = set()

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
            player_hand_count = self.get_player_card_count(player)
            opponent_id = self.get_opponent_index(player)
            opponent_hand_count = self.get_player_card_count(opponent_id)
        except Exception as e_count:
            logger.error(
                "Error getting hand counts for legal pending actions P%d: %s",
                player,
                e_count,
            )
            return legal_actions

        try:
            if isinstance(action_type, ActionDiscard):  # Post-Draw Choice
                legal_actions.add(ActionDiscard(use_ability=False))
                drawn_card = self.pending_action_data.get("drawn_card")
                if drawn_card and card_has_discard_ability(drawn_card):
                    # Check if ability *can* be used (e.g., non-empty hands for swap/peek)
                    rank = drawn_card.rank
                    can_use = True
                    if rank in [SEVEN, EIGHT] and player_hand_count == 0:
                        can_use = False
                    if rank in [NINE, TEN] and opponent_hand_count == 0:
                        can_use = False
                    if rank in [JACK, QUEEN, KING] and (
                        player_hand_count == 0 or opponent_hand_count == 0
                    ):
                        can_use = False

                    if can_use:
                        legal_actions.add(ActionDiscard(use_ability=True))
                    else:
                        logger.debug(
                            "Ability card %s drawn, but cannot use ability due to empty hand(s).",
                            drawn_card,
                        )

                # Add Replace actions only if hand has cards
                if player_hand_count > 0:
                    for i in range(player_hand_count):
                        legal_actions.add(ActionReplace(target_hand_index=i))

            elif isinstance(action_type, ActionAbilityPeekOwnSelect):  # 7/8 Peek Choice
                if player_hand_count > 0:
                    for i in range(player_hand_count):
                        legal_actions.add(ActionAbilityPeekOwnSelect(target_hand_index=i))
                else:
                    logger.debug(
                        "Cannot generate PeekOwn actions: P%d has 0 cards.", player
                    )
                    # If no actions possible, this pending state is stuck. Engine needs to handle?
                    # Add a Pass action? No, ability should fizzle.
                    pass

            elif isinstance(action_type, ActionAbilityPeekOtherSelect):  # 9/T Peek Choice
                if opponent_hand_count > 0:
                    for i in range(opponent_hand_count):
                        legal_actions.add(
                            ActionAbilityPeekOtherSelect(target_opponent_hand_index=i)
                        )
                else:
                    logger.debug(
                        "Cannot generate PeekOther actions: Opponent has 0 cards."
                    )
                    # Ability fizzles

            elif isinstance(action_type, ActionAbilityBlindSwapSelect):  # J/Q Swap Choice
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
                    # Ability fizzles

            elif isinstance(action_type, ActionAbilityKingLookSelect):  # K Look Choice
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
                    # Ability fizzles

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
                    # Ability fizzles / Snap state error
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
                    # Snap state error - player should have had cards to initiate snap

            else:
                logger.error(
                    "Unknown pending action type (%s) encountered for legal actions.",
                    type(action_type).__name__,
                )

        except (IndexError, KeyError) as e_idx:
            # JUSTIFIED: Index/key errors during legal action generation indicate state corruption
            logger.error(
                "Index/key error generating legal pending actions for %s: %s",
                type(action_type).__name__,
                e_idx,
                exc_info=True,
            )
        except Exception as e_legal:
            # JUSTIFIED: Catch all other errors to prevent legal action calculation from crashing
            logger.error(
                "Unexpected error generating legal pending actions for %s: %s",
                type(action_type).__name__,
                e_legal,
                exc_info=True,
            )

        # If no actions generated (e.g. 0 cards for required action), handle gracefully
        if not legal_actions and self.pending_action:
            logger.warning(
                "No legal actions generated for P%d in pending state %s (Likely 0 cards?). Ability may fizzle.",
                player,
                type(action_type).__name__,
            )
            # Return empty set - the caller (_handle_pending_action) should handle this
            # by clearing the pending state (fizzling the ability).

        return legal_actions

    # --- Pending Action / Ability Processing ---

    def _handle_pending_action(
        self: "CambiaGameState",
        action: GameAction,
        acting_player: int,
        undo_stack: Deque,
        delta_list: StateDelta,
    ) -> Optional[Card]:
        """
        Processes an action resolving a pending state. Modifies state via _add_change.
        Returns the card discarded *this step* for snap checks, or None if no discard occurred.
        Returns None if action is invalid for the pending state or player.
        """
        if not all(
            hasattr(self, attr)
            for attr in [
                "pending_action",
                "pending_action_player",
                "pending_action_data",
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
            return None

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

        # Check if action is legal for current pending state
        legal_pending_actions = self._get_legal_pending_actions(acting_player)
        if action not in legal_pending_actions:
            # Handle cases where no legal actions exist (ability fizzles)
            if not legal_pending_actions:
                logger.warning(
                    "Ability/Pending action for P%d (%s) has no legal actions. Fizzling.",
                    player,
                    type(pending_type).__name__,
                )
                # Return the card discarded that *triggered* this pending state, if available
                # This allows snap checks even if the ability fizzles.
                card_that_triggered = self.pending_action_data.get("ability_card")
                if not isinstance(card_that_triggered, Card):
                    card_that_triggered = self.get_discard_top()  # Last resort fallback

                self._clear_pending_action(undo_stack, delta_list)
                return (
                    card_that_triggered  # Return card that triggered this for snap check
                )
            else:
                logger.warning(
                    "Invalid action %s for P%d pending state %s. Legal: %s. Waiting.",
                    action,
                    player,
                    type(pending_type).__name__,
                    legal_pending_actions,
                )
                return None  # Wait for a valid action

        try:
            # --- Handle Post-Draw Choices (Discard/Replace) ---
            if isinstance(pending_type, ActionDiscard):
                drawn_card = self.pending_action_data.get("drawn_card")
                if not isinstance(drawn_card, Card):
                    logger.error(
                        "Pending post-draw choice but invalid/missing drawn_card in data! Data: %s",
                        self.pending_action_data,
                    )
                    self._clear_pending_action(
                        undo_stack, delta_list
                    )  # Clear invalid state
                    return None

                if isinstance(action, ActionDiscard):
                    logger.debug(
                        "P%d discards drawn %s. Use ability: %s",
                        player,
                        drawn_card,
                        action.use_ability,
                    )
                    original_discard_len = len(self.discard_pile)
                    original_discard_top = self.get_discard_top()

                    def change_discard():
                        self.discard_pile.append(drawn_card)

                    def undo_discard():
                        # Assert preconditions for undo
                        assert (
                            self.discard_pile and self.discard_pile[-1] is drawn_card
                        ), f"Undo Discard: Top card mismatch (Expected {drawn_card}, Got {self.discard_pile[-1] if self.discard_pile else 'Empty'})"
                        self.discard_pile.pop()
                        # Assert postconditions
                        assert len(self.discard_pile) == original_discard_len
                        assert self.get_discard_top() is original_discard_top

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
                    # Ensure hand and index are valid before proceeding
                    if not (0 <= target_idx < len(hand)):
                        logger.error(
                            "Invalid REPLACE action index: %d for hand size %d",
                            target_idx,
                            len(hand),
                        )
                        return None  # Invalid choice

                    replaced_card = hand[target_idx]
                    if not isinstance(replaced_card, Card):
                        logger.error(
                            "Replace target index %d holds non-Card object: %s",
                            target_idx,
                            replaced_card,
                        )
                        self._clear_pending_action(undo_stack, delta_list)
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
                    ]  # Capture original card *at the index*
                    original_hand_state = list(
                        hand
                    )  # Capture full hand state for undo check
                    original_discard_len = len(self.discard_pile)
                    original_discard_top = self.get_discard_top()

                    def change_replace():
                        # Check index valid and card is still the same before modifying
                        if (
                            0 <= target_idx < len(self.players[player].hand)
                            and self.players[player].hand[target_idx]
                            is original_card_in_hand
                        ):
                            self.players[player].hand[target_idx] = drawn_card
                            self.discard_pile.append(replaced_card)
                        else:
                            logger.error(
                                "Change Replace Error: Hand state changed unexpectedly before replace. Hand[%d] is %s, expected %s",
                                target_idx,
                                (
                                    self.players[player].hand[target_idx]
                                    if 0 <= target_idx < len(self.players[player].hand)
                                    else "OOB"
                                ),
                                original_card_in_hand,
                            )
                            # Avoid modifying state if precondition fails

                    def undo_replace():
                        # Assert preconditions for undo
                        assert (
                            self.discard_pile and self.discard_pile[-1] is replaced_card
                        ), f"Undo Replace: Discard top mismatch (Expected {replaced_card}, Got {self.discard_pile[-1] if self.discard_pile else 'Empty'})"
                        popped_discard = self.discard_pile.pop()

                        assert (
                            0 <= target_idx < len(self.players[player].hand)
                            and self.players[player].hand[target_idx] is drawn_card
                        ), f"Undo Replace: Hand content mismatch at index {target_idx} (Expected {drawn_card}, Got {self.players[player].hand[target_idx]})"
                        self.players[player].hand[
                            target_idx
                        ] = original_card_in_hand  # Restore original card

                        # Assert postconditions
                        assert len(self.discard_pile) == original_discard_len
                        assert self.get_discard_top() is original_discard_top
                        assert (
                            self.players[player].hand == original_hand_state
                        ), f"Undo Replace: Hand state mismatch! Expected {original_hand_state}, Got {self.players[player].hand}"

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
                    # This path is now unreachable due to the legal action check at the start
                    logger.error(
                        "INTERNAL ERROR: Reached unexpected action %s handling Post-Draw state.",
                        action,
                    )
                    return None

            # --- Handle Ability Selections ---
            elif isinstance(pending_type, ActionAbilityPeekOwnSelect):
                # Action guaranteed to be ActionAbilityPeekOwnSelect by initial check
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
                        delta_list.append(
                            ("peek_own", player, target_idx, peeked_card_str)
                        )
                    else:  # Should not happen if hand state is correct
                        logger.error(
                            "Peek Own Error: Item at index %d is not Card: %s",
                            target_idx,
                            peeked_card,
                        )
                        delta_list.append(("peek_own_fail", player, target_idx))
                else:  # Should not happen if legal actions generated correctly
                    logger.error(
                        "Invalid PEEK_OWN index %d for hand size %d",
                        target_idx,
                        len(hand),
                    )
                    delta_list.append(("peek_own_fail", player, target_idx))
                card_just_discarded_for_snap_check = self.get_discard_top()
                self._clear_pending_action(undo_stack, delta_list)

            elif isinstance(pending_type, ActionAbilityPeekOtherSelect):
                # Action guaranteed to be ActionAbilityPeekOtherSelect
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
                            delta_list.append(
                                (
                                    "peek_other",
                                    player,
                                    opp_idx,
                                    target_opp_idx,
                                    peeked_card_str,
                                )
                            )
                        else:  # Should not happen
                            logger.error(
                                "Peek Other Error: Item at opponent index %d is not Card: %s",
                                target_opp_idx,
                                peeked_card,
                            )
                            delta_list.append(
                                ("peek_other_fail", player, opp_idx, target_opp_idx)
                            )
                    else:  # Should not happen if legal actions OK
                        logger.error(
                            "Invalid PEEK_OTHER index %d for opponent hand size %d",
                            target_opp_idx,
                            len(opp_hand),
                        )
                        delta_list.append(
                            ("peek_other_fail", player, opp_idx, target_opp_idx)
                        )
                else:  # Should not happen
                    logger.error(
                        "Peek Other Error: Opponent %d invalid or missing hand.", opp_idx
                    )
                    delta_list.append(
                        ("peek_other_fail", player, opp_idx, target_opp_idx)
                    )
                card_just_discarded_for_snap_check = self.get_discard_top()
                self._clear_pending_action(undo_stack, delta_list)

            elif isinstance(pending_type, ActionAbilityBlindSwapSelect):
                # Action guaranteed to be ActionAbilityBlindSwapSelect
                own_h_idx, opp_h_idx = action.own_hand_index, action.opponent_hand_index
                opp_idx = self.get_opponent_index(player)
                hand = self.players[player].hand
                swap_successful = False
                original_own_card, original_opp_card = None, None
                original_own_hand_state = list(hand)  # For undo check
                original_opp_hand_state = list(
                    self.players[opp_idx].hand
                )  # For undo check

                # Validation (should pass if action was legal)
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
                            captured_own_card = original_own_card  # For closure
                            captured_opp_card = original_opp_card  # For closure

                            def change_blind_swap():
                                # Check indices again before swap, state might have changed? No, rely on caller sync.
                                if 0 <= own_h_idx < len(
                                    self.players[player].hand
                                ) and 0 <= opp_h_idx < len(self.players[opp_idx].hand):
                                    # Check cards haven't changed unexpectedly
                                    if (
                                        self.players[player].hand[own_h_idx]
                                        is captured_own_card
                                        and self.players[opp_idx].hand[opp_h_idx]
                                        is captured_opp_card
                                    ):
                                        (
                                            self.players[player].hand[own_h_idx],
                                            self.players[opp_idx].hand[opp_h_idx],
                                        ) = (
                                            captured_opp_card,
                                            captured_own_card,
                                        )
                                    else:
                                        logger.error(
                                            "Change BlindSwap error: Card identity mismatch at indices."
                                        )
                                else:
                                    logger.error("Change BlindSwap error: Index OOB.")

                            def undo_blind_swap():
                                # Assert preconditions
                                assert (
                                    0 <= own_h_idx < len(self.players[player].hand)
                                ), f"Undo BlindSwap: Own index {own_h_idx} OOB"
                                assert (
                                    0 <= opp_h_idx < len(self.players[opp_idx].hand)
                                ), f"Undo BlindSwap: Opp index {opp_h_idx} OOB"
                                assert (
                                    self.players[player].hand[own_h_idx]
                                    is captured_opp_card
                                ), f"Undo BlindSwap: Own card mismatch (Expected {captured_opp_card}, Got {self.players[player].hand[own_h_idx]})"
                                assert (
                                    self.players[opp_idx].hand[opp_h_idx]
                                    is captured_own_card
                                ), f"Undo BlindSwap: Opp card mismatch (Expected {captured_own_card}, Got {self.players[opp_idx].hand[opp_h_idx]})"

                                # Perform undo
                                (
                                    self.players[player].hand[own_h_idx],
                                    self.players[opp_idx].hand[opp_h_idx],
                                ) = (
                                    captured_own_card,
                                    captured_opp_card,
                                )
                                # Assert postconditions
                                assert (
                                    self.players[player].hand == original_own_hand_state
                                ), "Undo BlindSwap: Own hand mismatch post-undo"
                                assert (
                                    self.players[opp_idx].hand == original_opp_hand_state
                                ), "Undo BlindSwap: Opp hand mismatch post-undo"

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
                                "Blind Swap Error: Cards involved not valid Card objects"
                            )
                    else:
                        logger.error("Invalid BLIND_SWAP indices")
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
                self._clear_pending_action(
                    undo_stack, delta_list
                )  # Clear state even if swap failed (ability fizzles)

            elif isinstance(pending_type, ActionAbilityKingLookSelect):
                # Action guaranteed to be ActionAbilityKingLookSelect
                own_h_idx, opp_h_idx = action.own_hand_index, action.opponent_hand_index
                opp_idx = self.get_opponent_index(player)
                hand = self.players[player].hand
                can_proceed = False
                card1_str, card2_str = "ERROR", "ERROR"
                card1, card2 = None, None

                # Validation (should pass if action legal)
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
                                "King Look Error: Cards involved not valid Card objects"
                            )
                    else:
                        logger.error("Invalid KING_LOOK indices")
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
                    original_pending_tuple = (  # Keep using tuple for undo state capture
                        self.pending_action,
                        self.pending_action_player,
                        dict(self.pending_action_data),
                    )
                    # Store actual cards and indices for decision phase
                    new_pending_data = {
                        "own_idx": own_h_idx,
                        "opp_idx": opp_h_idx,
                        "card1": card1,
                        "card2": card2,
                        # Propagate original ability card from *current* step's data
                        "ability_card": self.pending_action_data.get("ability_card"),
                    }
                    next_pending_action_type = ActionAbilityKingSwapDecision(
                        perform_swap=False
                    )

                    def change_king_pending():
                        self.pending_action = next_pending_action_type
                        self.pending_action_player = player
                        self.pending_action_data = new_pending_data

                    def undo_king_pending():
                        # Assert preconditions (optional, but good practice)
                        assert self.pending_action is next_pending_action_type
                        assert self.pending_action_player == player
                        # Restore previous state
                        (
                            self.pending_action,
                            self.pending_action_player,
                            self.pending_action_data,
                        ) = original_pending_tuple  # Use captured tuple
                        logger.debug("Undo King Look -> Pending Swap.")

                    prev_pending_type_name = (
                        type(original_pending_tuple[0]).__name__
                        if original_pending_tuple[0]
                        else None
                    )
                    # Serialize data for logging/delta (handle potential None for ability_card)
                    serialized_orig_data = {
                        k: serialize_card(v) if isinstance(v, Card) else v
                        for k, v in original_pending_tuple[2].items()
                    }
                    serialized_new_data = {
                        "own_idx": own_h_idx,
                        "opp_idx": opp_h_idx,
                        "card1": card1_str,
                        "card2": card2_str,
                        "ability_card": serialize_card(new_pending_data["ability_card"]),
                    }
                    delta_king_pending = (
                        "set_pending_action",
                        type(next_pending_action_type).__name__,
                        player,
                        serialized_new_data,
                        prev_pending_type_name,
                        original_pending_tuple[1],  # Original player
                        serialized_orig_data,
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
                else:  # Look failed (invalid index/hand)
                    card_just_discarded_for_snap_check = self.get_discard_top()
                    self._clear_pending_action(
                        undo_stack, delta_list
                    )  # Clear pending state

            elif isinstance(pending_type, ActionAbilityKingSwapDecision):
                # Action guaranteed to be ActionAbilityKingSwapDecision
                perform_swap = action.perform_swap
                look_data = self.pending_action_data
                own_h_idx = look_data.get("own_idx")
                opp_h_idx = look_data.get("opp_idx")
                card1 = look_data.get("card1")  # Own card peeked
                card2 = look_data.get("card2")  # Opp card peeked

                # --- Pre-action Validation ---
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
                    # Clear pending state even on error
                    card_just_discarded_for_snap_check = (
                        self.pending_action_data.get("ability_card")
                        or self.get_discard_top()
                    )
                    self._clear_pending_action(undo_stack, delta_list)
                    return card_just_discarded_for_snap_check  # Allow snap check

                opp_idx = self.get_opponent_index(player)
                hand = self.players[player].hand
                opp_hand = self.players[opp_idx].hand  # Assume valid opponent index

                # Check context validity (indices in bounds, cards still match)
                own_idx_valid = 0 <= own_h_idx < len(hand)
                opp_idx_valid = 0 <= opp_h_idx < len(opp_hand)
                card_at_own_idx = hand[own_h_idx] if own_idx_valid else None
                card_at_opp_idx = opp_hand[opp_h_idx] if opp_idx_valid else None
                own_card_match = own_idx_valid and (card_at_own_idx is card1)
                opp_card_match = opp_idx_valid and (card_at_opp_idx is card2)
                context_valid = own_card_match and opp_card_match
                # --- End Pre-action Validation ---

                if context_valid:
                    if perform_swap:
                        captured_card1, captured_card2 = card1, card2  # For closure
                        original_own_hand_state = list(hand)  # For undo check
                        original_opp_hand_state = list(opp_hand)  # For undo check

                        def change_king_swap():
                            # Check context again just before swap? Maybe overkill if action application is atomic.
                            if 0 <= own_h_idx < len(
                                self.players[player].hand
                            ) and 0 <= opp_h_idx < len(self.players[opp_idx].hand):
                                (
                                    self.players[player].hand[own_h_idx],
                                    self.players[opp_idx].hand[opp_h_idx],
                                ) = (
                                    captured_card2,
                                    captured_card1,
                                )
                            else:
                                logger.error("Change KingSwap error: Index OOB.")

                        def undo_king_swap():
                            # Assert preconditions
                            assert (
                                0 <= own_h_idx < len(self.players[player].hand)
                            ), f"Undo KingSwap: Own index {own_h_idx} OOB"
                            assert (
                                0 <= opp_h_idx < len(self.players[opp_idx].hand)
                            ), f"Undo KingSwap: Opp index {opp_h_idx} OOB"
                            assert (
                                self.players[player].hand[own_h_idx] is captured_card2
                            ), f"Undo KingSwap: Own card mismatch (Expected {captured_card2}, Got {self.players[player].hand[own_h_idx]})"
                            assert (
                                self.players[opp_idx].hand[opp_h_idx] is captured_card1
                            ), f"Undo KingSwap: Opp card mismatch (Expected {captured_card1}, Got {self.players[opp_idx].hand[opp_h_idx]})"

                            # Perform undo
                            (
                                self.players[player].hand[own_h_idx],
                                self.players[opp_idx].hand[opp_h_idx],
                            ) = (
                                captured_card1,
                                captured_card2,
                            )
                            logger.debug("Undo King Swap successful.")
                            # Assert postconditions
                            assert (
                                self.players[player].hand == original_own_hand_state
                            ), "Undo KingSwap: Own hand mismatch post-undo"
                            assert (
                                self.players[opp_idx].hand == original_opp_hand_state
                            ), "Undo KingSwap: Opp hand mismatch post-undo"

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
                        logger.info("P%d King ability: Chose not to swap cards.", player)
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
                else:  # Context was invalid
                    reason = f"King Swap Context Invalid: own_idx_ok={own_idx_valid}, opp_idx_ok={opp_idx_valid}, own_card_match={own_card_match}, opp_card_match={opp_card_match}"
                    logger.warning("%s. Ability fizzles.", reason)
                    delta_list.append(
                        ("king_swap_fail", player, own_h_idx, opp_h_idx, reason)
                    )

                # Ability sequence ends here, clear pending state
                card_just_discarded_for_snap_check = (
                    self.pending_action_data.get("ability_card") or self.get_discard_top()
                )
                self._clear_pending_action(undo_stack, delta_list)
                return (
                    card_just_discarded_for_snap_check  # Return original discarded King
                )

            elif isinstance(pending_type, ActionSnapOpponentMove):
                # Action guaranteed to be ActionSnapOpponentMove
                snapper_idx = player
                own_card_idx = action.own_card_to_move_hand_index
                target_slot_idx = self.pending_action_data.get("target_empty_slot_index")

                # Pre-action validation
                if target_slot_idx is None or not isinstance(target_slot_idx, int):
                    logger.error(
                        "SnapOpponentMove Error: Missing/invalid target_empty_slot_index."
                    )
                    self._clear_pending_action(
                        undo_stack, delta_list
                    )  # Clear broken state
                    return None
                if action.target_empty_slot_index != target_slot_idx:
                    logger.error(
                        "SnapOpponentMove Error: Action target slot (%d) != pending data target (%d).",
                        action.target_empty_slot_index,
                        target_slot_idx,
                    )
                    return None  # Invalid action choice, wait for correct one

                hand = self.players[snapper_idx].hand
                opp_idx = self.get_opponent_index(snapper_idx)
                move_successful = False
                card_to_move = None

                if not (
                    0 <= opp_idx < len(self.players)
                    and hasattr(self.players[opp_idx], "hand")
                ):
                    logger.error(
                        "Snap Opponent Move Error: Opponent %d invalid.", opp_idx
                    )
                elif not (0 <= own_card_idx < len(hand)):
                    logger.error(
                        "Invalid own card index %d for SnapOpponentMove (Own Hand Size: %d).",
                        own_card_idx,
                        len(hand),
                    )
                else:
                    card_to_move = hand[own_card_idx]
                    if not isinstance(card_to_move, Card):
                        logger.error(
                            "SnapOpponentMove Error: Item at own index %d is not Card: %s",
                            own_card_idx,
                            card_to_move,
                        )
                    else:
                        opp_hand = self.players[opp_idx].hand
                        if not (
                            0 <= target_slot_idx <= len(opp_hand)
                        ):  # Allow insert at end
                            logger.error(
                                "Invalid target slot index %d for SnapOpponentMove (Opp Hand Size: %d).",
                                target_slot_idx,
                                len(opp_hand),
                            )
                        else:
                            original_card = card_to_move  # Capture for closure
                            original_snapper_hand = list(hand)
                            original_opponent_hand = list(opp_hand)

                            def change_move():
                                # Check indices and card identity again
                                if (
                                    0
                                    <= own_card_idx
                                    < len(self.players[snapper_idx].hand)
                                    and self.players[snapper_idx].hand[own_card_idx]
                                    is original_card
                                ):
                                    moved_card = self.players[snapper_idx].hand.pop(
                                        own_card_idx
                                    )
                                    if (
                                        0
                                        <= target_slot_idx
                                        <= len(self.players[opp_idx].hand)
                                    ):
                                        self.players[opp_idx].hand.insert(
                                            target_slot_idx, moved_card
                                        )
                                    else:
                                        logger.error(
                                            "SnapMove change error: Target index %d OOB",
                                            target_slot_idx,
                                        )
                                else:
                                    logger.error(
                                        "SnapMove change error: Source index %d OOB or card mismatch.",
                                        own_card_idx,
                                    )

                            def undo_move():
                                # Assert preconditions
                                assert (
                                    0 <= target_slot_idx < len(self.players[opp_idx].hand)
                                ), f"Undo SnapMove: Target index {target_slot_idx} OOB (Opp size {len(self.players[opp_idx].hand)})"
                                assert (
                                    self.players[opp_idx].hand[target_slot_idx]
                                    is original_card
                                ), f"Undo SnapMove: Target card mismatch (Expected {original_card}, Got {self.players[opp_idx].hand[target_slot_idx]})"

                                moved_back_card = self.players[opp_idx].hand.pop(
                                    target_slot_idx
                                )

                                assert (
                                    0
                                    <= own_card_idx
                                    <= len(self.players[snapper_idx].hand)
                                ), f"Undo SnapMove: Source index {own_card_idx} OOB (Own size {len(self.players[snapper_idx].hand)})"
                                self.players[snapper_idx].hand.insert(
                                    own_card_idx, moved_back_card
                                )
                                # Assert postconditions
                                assert (
                                    self.players[snapper_idx].hand
                                    == original_snapper_hand
                                ), "Undo SnapMove: Snapper hand mismatch"
                                assert (
                                    self.players[opp_idx].hand == original_opponent_hand
                                ), "Undo SnapMove: Opponent hand mismatch"

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

                if move_successful:
                    logger.info(
                        "P%d completes Snap Opponent: Moves %s (from own idx %d) to opp idx %d.",
                        snapper_idx,
                        card_to_move,
                        own_card_idx,
                        target_slot_idx,
                    )
                    self._clear_pending_action(undo_stack, delta_list)
                    # Turn advances after pending action resolved and no snap phase active
                    return None  # No card discarded this step
                else:
                    # Failure in validation or index means invalid action was chosen
                    # The initial legal action check should prevent this path.
                    logger.error(
                        "SnapOpponentMove handler reached failure state unexpectedly."
                    )
                    return None

            else:  # Mismatch between pending state and action received
                # This path is now unreachable due to the legal action check at the start
                logger.error(
                    "INTERNAL ERROR: Unhandled pending action type %s vs received action %s",
                    type(pending_type).__name__,
                    type(action).__name__,
                )
                return None

        except ActionApplicationError:
            # Re-raise action application errors
            raise
        except Exception as e_handle:
            logger.exception(
                "Error handling pending action %s for P%d: %s",
                action,
                acting_player,
                e_handle,
            )
            self._clear_pending_action(undo_stack, delta_list)  # Attempt recovery
            raise ActionApplicationError(f"Pending action handling failed for {action}") from e_handle

        return card_just_discarded_for_snap_check  # Return card discarded this step (for snap check) or None

    # --- Ability Triggering and State Management ---

    def _trigger_discard_ability(
        self: "CambiaGameState",
        player_index: int,
        discarded_card: Card,
        undo_stack: Deque,
        delta_list: StateDelta,
    ):
        """Checks discard ability, sets pending action state if choice needed."""
        if not all(
            hasattr(self, attr)
            for attr in [
                "pending_action",
                "pending_action_player",
                "pending_action_data",
                "_add_change",
                "get_player_card_count",
                "get_opponent_index",
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

        # Check if ability requires action based on current game state
        player_hand_count = self.get_player_card_count(player_index)
        opponent_hand_count = self.get_player_card_count(
            self.get_opponent_index(player_index)
        )
        can_peek_own = player_hand_count > 0
        can_peek_opp = opponent_hand_count > 0
        can_swap = can_peek_own and can_peek_opp

        if rank in [SEVEN, EIGHT] and can_peek_own:
            next_pending_action = ActionAbilityPeekOwnSelect(-1)
        elif rank in [NINE, TEN] and can_peek_opp:
            next_pending_action = ActionAbilityPeekOtherSelect(-1)
        elif rank in [JACK, QUEEN] and can_swap:
            next_pending_action = ActionAbilityBlindSwapSelect(-1, -1)
        elif rank == KING and can_swap:
            next_pending_action = ActionAbilityKingLookSelect(-1, -1)
        else:
            logger.debug(
                "Card %s ability requires no action or cannot be performed (Hand sizes: P%d=%d, P%d=%d). Fizzles.",
                discarded_card,
                player_index,
                player_hand_count,
                self.get_opponent_index(player_index),
                opponent_hand_count,
            )

        if next_pending_action:
            # Store original state *before* setting pending action
            original_pending_action = self.pending_action
            original_pending_player = self.pending_action_player
            original_pending_data = dict(self.pending_action_data)
            new_pending_data = {"ability_card": discarded_card}  # Store card for context

            def change_ability_pending():
                self.pending_action = next_pending_action
                self.pending_action_player = player_index
                self.pending_action_data = new_pending_data

            def undo_ability_pending():
                # Assert preconditions (optional)
                assert self.pending_action is next_pending_action
                assert self.pending_action_player == player_index
                # Restore previous state
                self.pending_action = original_pending_action
                self.pending_action_player = original_pending_player
                self.pending_action_data = original_pending_data
                logger.debug("Undo set ability pending.")

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

    def _clear_pending_action(
        self: "CambiaGameState", undo_stack: Deque, delta_list: StateDelta
    ):
        """Resets the pending action state, adding undo operation and delta."""
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
            logger.debug("Clear pending action called, but nothing to clear.")
            return  # Nothing to clear

        original_pending_action = self.pending_action
        original_pending_player = self.pending_action_player
        original_pending_data = dict(self.pending_action_data)

        def change_clear():
            self.pending_action = None
            self.pending_action_player = None
            self.pending_action_data = {}

        def undo_clear():
            # Assert preconditions (optional)
            assert self.pending_action is None
            assert self.pending_action_player is None
            # Restore previous state
            self.pending_action = original_pending_action
            self.pending_action_player = original_pending_player
            self.pending_action_data = original_pending_data
            logger.debug("Undo clear pending action.")

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

    def _get_pending_action_name(self: "CambiaGameState") -> str:
        """Helper for __str__."""
        pending = getattr(self, "pending_action", None)
        return type(pending).__name__ if pending else "N/A"
