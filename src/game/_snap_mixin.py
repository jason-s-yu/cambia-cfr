"""
src/game/_snap_mixin.py

This module implements snapping mixins for the game engine
"""

import logging
import copy
from typing import Set, Optional, Deque, Tuple

from .types import StateDelta
from .helpers import serialize_card
from ..card import Card
from ..constants import (
    GameAction,
    ActionPassSnap,
    ActionSnapOwn,
    ActionSnapOpponent,
    ActionSnapOpponentMove,
)

logger = logging.getLogger(__name__)


class SnapLogicMixin:
    """Mixin handling the Snap phase logic for CambiaGameState."""

    # --- Snap Phase Legal Actions ---

    def _get_legal_snap_actions(self, acting_player: int) -> Set[GameAction]:
        """Calculates legal actions during the snap phase for the acting player."""
        legal_actions: Set[GameAction] = set()

        if not self.snap_phase_active:
            logger.error(
                "Snap Logic: _get_legal_snap_actions called when snap phase inactive."
            )
            return legal_actions
        if not (
            0 <= acting_player < len(self.players)
            and hasattr(self.players[acting_player], "hand")
        ):
            logger.error(
                "Snap Logic: Acting player P%d invalid for legal actions.", acting_player
            )
            return legal_actions

        snapper_hand = self.players[acting_player].hand
        if self.snap_discarded_card is None or not isinstance(
            self.snap_discarded_card, Card
        ):
            logger.error(
                "Snap Logic: Snap phase active but snap_discarded_card is invalid: %s.",
                self.snap_discarded_card,
            )
            return legal_actions

        target_rank = self.snap_discarded_card.rank

        # Basic validation of snapper's hand
        if not all(isinstance(card, Card) for card in snapper_hand):
            logger.error(
                "Snap Logic: P%d hand contains non-Card objects: %s",
                acting_player,
                snapper_hand,
            )
            return legal_actions  # Cannot determine legal actions

        # Always possible to pass
        legal_actions.add(ActionPassSnap())

        # Check for own snaps
        for i, card in enumerate(snapper_hand):
            if isinstance(card, Card) and card.rank == target_rank:
                legal_actions.add(ActionSnapOwn(own_card_hand_index=i))

        # Check for opponent snaps if allowed
        if self.house_rules.allowOpponentSnapping:
            opponent_idx = self.get_opponent_index(
                acting_player
            )  # Method from QueryMixin/base
            if not (
                0 <= opponent_idx < len(self.players)
                and hasattr(self.players[opponent_idx], "hand")
            ):
                logger.warning(
                    "Snap Logic: Opponent P%d invalid, cannot check for SnapOpponent.",
                    opponent_idx,
                )
            else:
                opponent_hand = self.players[opponent_idx].hand
                if not all(isinstance(card, Card) for card in opponent_hand):
                    logger.error(
                        "Snap Logic: Opponent P%d hand contains non-Card objects: %s",
                        opponent_idx,
                        opponent_hand,
                    )
                else:
                    for i, card in enumerate(opponent_hand):
                        if isinstance(card, Card) and card.rank == target_rank:
                            legal_actions.add(
                                ActionSnapOpponent(opponent_target_hand_index=i)
                            )

        return legal_actions

    # --- Snap Phase Action Processing ---

    def _handle_snap_action(
        self,
        action: GameAction,
        acting_player: int,
        undo_stack: Deque,
        delta_list: StateDelta,
    ) -> bool:
        """
        Processes an action during the snap phase. Modifies state via _add_change.
        Returns True if action processed, False on critical error (e.g., wrong player).
        """
        if not self.snap_phase_active:
            logger.error("Snap Logic: _handle_snap_action called when inactive.")
            return False
        if not (0 <= self.snap_current_snapper_idx < len(self.snap_potential_snappers)):
            logger.error(
                "Snap Logic: Invalid snap_current_snapper_idx %d (potential: %d)",
                self.snap_current_snapper_idx,
                len(self.snap_potential_snappers),
            )
            self._end_snap_phase(undo_stack, delta_list)  # Attempt cleanup
            return True  # Handled error state

        expected_player = self.snap_potential_snappers[self.snap_current_snapper_idx]
        if acting_player != expected_player:
            logger.error(
                "Snap Logic: Action %s from P%d, expected P%d. Ignoring.",
                action,
                acting_player,
                expected_player,
            )
            return False  # Indicate action ignored

        if self.snap_discarded_card is None or not isinstance(
            self.snap_discarded_card, Card
        ):
            logger.error(
                "Snap Logic: Cannot process snap action, snap_discarded_card invalid: %s.",
                self.snap_discarded_card,
            )
            self._end_snap_phase(undo_stack, delta_list)
            return True  # Handled error state

        logger.debug(
            "Snap Phase (T%d): P%d acting with %s",
            self._turn_number,
            acting_player,
            action,
        )
        target_rank = self.snap_discarded_card.rank
        snap_success = False
        snap_penalty = False
        attempted_card_str: Optional[str] = None  # For logging failed attempts
        action_type_str = type(action).__name__

        # Ensure _add_change and _apply_penalty are available
        if not hasattr(self, "_add_change") or not callable(self._add_change):
            logger.critical("Snap Logic: _add_change method not found on self!")
            return False  # Cannot proceed without state change mechanism
        if not hasattr(self, "_apply_penalty") or not callable(self._apply_penalty):
            logger.critical("Snap Logic: _apply_penalty method not found on self!")
            return False  # Cannot proceed without penalty mechanism

        # Helper for logging snap results (uses _add_change implicitly via self)
        def log_snap_result(details_dict):
            original_log = list(self.snap_results_log)

            def change():
                self.snap_results_log.append(details_dict)

            def undo():
                self.snap_results_log = original_log

            self._add_change(
                change, undo, ("snap_log_append", details_dict), undo_stack, delta_list
            )

        # Process Specific Snap Actions
        try:
            if isinstance(action, ActionPassSnap):
                logger.debug("P%d passes snap.", acting_player)
                log_snap_result(
                    {
                        "snapper": acting_player,
                        "action_type": action_type_str,
                        "target_rank": target_rank,
                        "success": False,
                        "penalty": False,
                        "details": "Passed",
                        "snapped_card": None,
                    }
                )

            elif isinstance(action, ActionSnapOwn):
                snap_idx = action.own_card_hand_index
                hand = self.players[acting_player].hand
                removed_card_info: Optional[Tuple[int, int, Card]] = None

                if 0 <= snap_idx < len(hand):
                    attempted_card = hand[snap_idx]
                    if not isinstance(attempted_card, Card):
                        logger.error(
                            "SnapOwn Error: P%d Item at index %d is not Card: %s.",
                            acting_player,
                            snap_idx,
                            attempted_card,
                        )
                        snap_penalty = True
                        attempted_card_str = repr(attempted_card)  # Log representation
                    elif attempted_card.rank == target_rank:
                        card_to_remove = attempted_card
                        original_hand = list(hand)  # Copy for undo verification
                        original_discard = list(self.discard_pile)

                        def change_snap_own():
                            removed = self.players[acting_player].hand.pop(snap_idx)
                            self.discard_pile.append(removed)

                        def undo_snap_own():
                            # Check if discard pile ends with the expected card before popping
                            if (
                                self.discard_pile
                                and self.discard_pile[-1] is card_to_remove
                            ):
                                popped = self.discard_pile.pop()
                                # Check if hand index is still valid before inserting
                                if 0 <= snap_idx <= len(self.players[acting_player].hand):
                                    self.players[acting_player].hand.insert(
                                        snap_idx, popped
                                    )
                                else:
                                    logger.warning(
                                        "Undo SnapOwn failed: Index %d invalid for hand size %d.",
                                        snap_idx,
                                        len(self.players[acting_player].hand),
                                    )
                            else:
                                logger.warning(
                                    "Undo SnapOwn mismatch: discard top is %s, expected %s",
                                    (
                                        self.discard_pile[-1]
                                        if self.discard_pile
                                        else "Empty"
                                    ),
                                    card_to_remove,
                                )

                        delta_snap_own = (
                            "snap_own_success",
                            acting_player,
                            snap_idx,
                            serialize_card(card_to_remove),
                        )
                        self._add_change(
                            change_snap_own,
                            undo_snap_own,
                            delta_snap_own,
                            undo_stack,
                            delta_list,
                        )
                        snap_success = True
                        removed_card_info = (acting_player, snap_idx, card_to_remove)
                        logger.info(
                            "P%d snaps own %s (Rank %s) from index %d. Hand size: %d",
                            acting_player,
                            card_to_remove,
                            target_rank,
                            snap_idx,
                            len(self.players[acting_player].hand),
                        )
                    else:  # Card rank doesn't match
                        snap_penalty = True
                        attempted_card_str = serialize_card(attempted_card)
                else:  # Invalid index
                    snap_penalty = True
                    attempted_card_str = f"Invalid Index {snap_idx}"

                if snap_penalty:
                    logger.warning(
                        "P%d invalid Snap Own: %s (Target: %s, Attempted: %s). Penalty.",
                        acting_player,
                        action,
                        target_rank,
                        attempted_card_str,
                    )
                    penalty_deltas = self._apply_penalty(
                        acting_player, self.house_rules.penaltyDrawCount, undo_stack
                    )
                    delta_list.extend(penalty_deltas)

                log_snap_result(
                    {
                        "snapper": acting_player,
                        "action_type": action_type_str,
                        "target_rank": target_rank,
                        "success": snap_success,
                        "penalty": snap_penalty,
                        "removed_own_index": (
                            removed_card_info[1] if removed_card_info else None
                        ),
                        "snapped_card": (
                            serialize_card(removed_card_info[2])
                            if removed_card_info
                            else None
                        ),
                        "attempted_card_str": (
                            attempted_card_str if snap_penalty else None
                        ),
                    }
                )

            elif isinstance(action, ActionSnapOpponent):
                snapped_opponent_card_info: Optional[Tuple[int, int, Card]] = None
                if not self.house_rules.allowOpponentSnapping:
                    logger.warning(
                        "Invalid Action: SnapOpponent attempted by P%d but disallowed by rules.",
                        acting_player,
                    )
                    snap_penalty = True
                    penalty_deltas = self._apply_penalty(
                        acting_player, self.house_rules.penaltyDrawCount, undo_stack
                    )
                    delta_list.extend(penalty_deltas)
                    attempted_card_str = "Disallowed Action"
                else:
                    opp_idx = self.get_opponent_index(acting_player)
                    if not (
                        0 <= opp_idx < len(self.players)
                        and hasattr(self.players[opp_idx], "hand")
                    ):
                        logger.error("SnapOpponent Error: Opponent P%d invalid.", opp_idx)
                        snap_penalty = True  # Penalize if opponent state is bad? Or just fail? Penalize for now.
                        penalty_deltas = self._apply_penalty(
                            acting_player, self.house_rules.penaltyDrawCount, undo_stack
                        )
                        delta_list.extend(penalty_deltas)
                        attempted_card_str = f"Invalid Opponent {opp_idx}"
                    else:
                        opp_hand = self.players[opp_idx].hand
                        target_opp_hand_idx = action.opponent_target_hand_index
                        if 0 <= target_opp_hand_idx < len(opp_hand):
                            attempted_card = opp_hand[target_opp_hand_idx]
                            if not isinstance(attempted_card, Card):
                                logger.error(
                                    "SnapOpponent Error: P%d target P%d index %d holds non-Card: %s.",
                                    acting_player,
                                    opp_idx,
                                    target_opp_hand_idx,
                                    attempted_card,
                                )
                                snap_penalty = True
                                attempted_card_str = repr(attempted_card)
                            elif attempted_card.rank == target_rank:
                                card_to_remove = attempted_card
                                original_opp_hand = list(
                                    opp_hand
                                )  # For undo verification

                                def change_snap_opp_remove():
                                    self.players[opp_idx].hand.pop(target_opp_hand_idx)

                                def undo_snap_opp_remove():
                                    # Check index validity before inserting
                                    if (
                                        0
                                        <= target_opp_hand_idx
                                        <= len(self.players[opp_idx].hand)
                                    ):
                                        self.players[opp_idx].hand.insert(
                                            target_opp_hand_idx, card_to_remove
                                        )
                                    else:
                                        logger.warning(
                                            "Undo SnapOpponentRemove failed: index %d invalid for opp hand size %d",
                                            target_opp_hand_idx,
                                            len(self.players[opp_idx].hand),
                                        )

                                delta_snap_opp_remove = (
                                    "snap_opponent_remove",
                                    opp_idx,
                                    target_opp_hand_idx,
                                    serialize_card(card_to_remove),
                                )
                                self._add_change(
                                    change_snap_opp_remove,
                                    undo_snap_opp_remove,
                                    delta_snap_opp_remove,
                                    undo_stack,
                                    delta_list,
                                )

                                snap_success = True
                                snapped_opponent_card_info = (
                                    opp_idx,
                                    target_opp_hand_idx,
                                    card_to_remove,
                                )
                                logger.info(
                                    "P%d snaps opponent P%d's %s at index %d. Requires move.",
                                    acting_player,
                                    opp_idx,
                                    card_to_remove,
                                    target_opp_hand_idx,
                                )

                                # Set pending action for the MOVE step
                                original_pending = (
                                    self.pending_action,
                                    self.pending_action_player,
                                    copy.deepcopy(self.pending_action_data),
                                )
                                original_snap_active = (
                                    self.snap_phase_active
                                )  # Should be True
                                next_pending_action_type = ActionSnapOpponentMove(
                                    own_card_to_move_hand_index=-1,
                                    target_empty_slot_index=-1,
                                )  # Placeholder type
                                new_pending_data = {
                                    "target_empty_slot_index": target_opp_hand_idx
                                }

                                def change_pending_move():
                                    self.pending_action = next_pending_action_type
                                    self.pending_action_player = (
                                        acting_player  # Snapper makes move choice
                                    )
                                    self.pending_action_data = new_pending_data
                                    self.snap_phase_active = (
                                        False  # Move happens outside snap phase
                                    )

                                def undo_pending_move():
                                    (
                                        self.pending_action,
                                        self.pending_action_player,
                                        self.pending_action_data,
                                    ) = original_pending
                                    self.snap_phase_active = original_snap_active

                                delta_pending = (
                                    "set_pending_action",
                                    "ActionSnapOpponentMove",
                                    acting_player,
                                    new_pending_data,
                                    (
                                        type(original_pending[0]).__name__
                                        if original_pending[0]
                                        else None
                                    ),
                                    original_pending[1],
                                    original_pending[2],
                                )
                                delta_snap_active = (
                                    "set_attr",
                                    "snap_phase_active",
                                    False,
                                    original_snap_active,
                                )
                                self._add_change(
                                    change_pending_move,
                                    undo_pending_move,
                                    delta_pending,
                                    undo_stack,
                                    delta_list,
                                )
                                delta_list.append(delta_snap_active)

                            else:  # Card rank doesn't match
                                snap_penalty = True
                                attempted_card_str = serialize_card(attempted_card)
                        else:  # Invalid index
                            snap_penalty = True
                            attempted_card_str = f"Invalid Index {target_opp_hand_idx}"

                        if snap_penalty:
                            logger.warning(
                                "P%d invalid Snap Opponent: %s (Target: %s, Attempted: %s). Penalty.",
                                acting_player,
                                action,
                                target_rank,
                                attempted_card_str,
                            )
                            penalty_deltas = self._apply_penalty(
                                acting_player,
                                self.house_rules.penaltyDrawCount,
                                undo_stack,
                            )
                            delta_list.extend(penalty_deltas)

                # Log snap result (success or failure)
                log_snap_result(
                    {
                        "snapper": acting_player,
                        "action_type": action_type_str,
                        "target_rank": target_rank,
                        "success": snap_success,
                        "penalty": snap_penalty,
                        "removed_opponent_index": (
                            snapped_opponent_card_info[1]
                            if snapped_opponent_card_info
                            else None
                        ),
                        "snapped_card": (
                            serialize_card(snapped_opponent_card_info[2])
                            if snapped_opponent_card_info
                            else None
                        ),
                        "attempted_card_str": (
                            attempted_card_str if snap_penalty else None
                        ),
                    }
                )

                # Return immediately if waiting for MOVE action
                if snap_success:
                    return True

            else:  # Invalid action type during snap phase
                logger.error(
                    "Invalid action type %s received during snap phase processing from P%d.",
                    type(action).__name__,
                    acting_player,
                )
                log_snap_result(
                    {
                        "snapper": acting_player,
                        "action_type": "InvalidAction",
                        "target_rank": target_rank,
                        "success": False,
                        "penalty": False,
                        "details": f"Received {type(action).__name__}",
                    }
                )
                # Don't apply penalty for engine error, just advance snap turn

        except Exception as e_snap_handle:
            logger.exception(
                "Error handling snap action %s for P%d: %s",
                action,
                acting_player,
                e_snap_handle,
            )
            # Attempt to advance snap turn to avoid getting stuck
            # State changes might be partial, undo stack might be affected

        # --- Advance Snap Turn or End Phase ---
        # (Do not advance if ActionSnapOpponent succeeded, as it awaits move)
        if not (isinstance(action, ActionSnapOpponent) and snap_success):
            try:
                original_snap_idx_local = self.snap_current_snapper_idx
                next_snap_idx = self.snap_current_snapper_idx + 1

                def change_snap_idx():
                    self.snap_current_snapper_idx = next_snap_idx

                def undo_snap_idx():
                    self.snap_current_snapper_idx = original_snap_idx_local

                self._add_change(
                    change_snap_idx,
                    undo_snap_idx,
                    (
                        "set_attr",
                        "snap_current_snapper_idx",
                        next_snap_idx,
                        original_snap_idx_local,
                    ),
                    undo_stack,
                    delta_list,
                )

                # Check if snap phase ends
                if self.snap_current_snapper_idx >= len(self.snap_potential_snappers):
                    self._end_snap_phase(undo_stack, delta_list)
                # else: Next snapper acts.
            except Exception as e_advance_snap:
                logger.exception("Error advancing snap turn index: %s", e_advance_snap)
                # Attempt to end phase if advancing failed
                self._end_snap_phase(undo_stack, delta_list)

        return True  # Indicate action was processed (even if pass/penalty/error handled)

    # --- Snap Phase Initiation and Termination ---

    def _initiate_snap_phase(
        self, discarded_card: Card, undo_stack: Deque, delta_list: StateDelta
    ) -> bool:
        """Checks if discard triggers snap phase, sets up state if so."""
        if not isinstance(discarded_card, Card):
            logger.error(
                "Cannot initiate snap phase: discarded_card is not a Card object (%s).",
                discarded_card,
            )
            return False

        potential_indices = []
        target_rank = discarded_card.rank
        discarder_player = (
            self.current_player_index
        )  # Player whose action led to this discard

        for p_idx in range(self.num_players):
            if p_idx == self.cambia_caller_id:
                continue  # Cambia caller cannot snap

            if not (
                0 <= p_idx < len(self.players) and hasattr(self.players[p_idx], "hand")
            ):
                logger.warning("Initiate Snap Check: P%d invalid. Skipping.", p_idx)
                continue

            hand = self.players[p_idx].hand
            if not all(isinstance(card, Card) for card in hand):
                logger.error(
                    "Initiate Snap Check: P%d hand invalid: %s. Skipping.", p_idx, hand
                )
                continue

            can_snap_own = any(card.rank == target_rank for card in hand)
            can_snap_opponent = False
            if self.house_rules.allowOpponentSnapping:
                opp_idx = self.get_opponent_index(p_idx)
                if opp_idx != self.cambia_caller_id:  # Opponent also cannot be caller
                    if 0 <= opp_idx < len(self.players) and hasattr(
                        self.players[opp_idx], "hand"
                    ):
                        opp_hand = self.players[opp_idx].hand
                        if not all(isinstance(card, Card) for card in opp_hand):
                            logger.error(
                                "Initiate Snap Check: Opponent P%d hand invalid. Skipping snap-opp for P%d.",
                                opp_idx,
                                p_idx,
                            )
                        else:
                            can_snap_opponent = any(
                                card.rank == target_rank for card in opp_hand
                            )
                    else:
                        logger.warning(
                            "Initiate Snap Check: Opponent P%d invalid for P%d checking SnapOpponent.",
                            opp_idx,
                            p_idx,
                        )

            if can_snap_own or can_snap_opponent:
                potential_indices.append(p_idx)

        if not potential_indices:
            logger.debug(
                "No potential snappers found for discard of %s (Rank %s).",
                discarded_card,
                target_rank,
            )
            return False

        # Determine snap order: Start from player *after* the discarder.
        ordered_snappers = []
        for i in range(1, self.num_players):
            check_p_idx = (discarder_player + i) % self.num_players
            if check_p_idx in potential_indices:
                ordered_snappers.append(check_p_idx)
        # Discarder snaps last (if eligible and not Cambia caller)
        if (
            discarder_player in potential_indices
            and discarder_player != self.cambia_caller_id
        ):
            ordered_snappers.append(discarder_player)

        if not ordered_snappers:
            logger.debug(
                "Potential snappers list empty after ordering/filtering. No snap phase."
            )
            return False

        # Start Snap Phase
        logger.info(
            "Snap phase started. Discard: %s. Snappers (ordered): %s.",
            discarded_card,
            ordered_snappers,
        )
        original_snap_phase = self.snap_phase_active
        original_snap_card = self.snap_discarded_card
        original_snap_potentials = list(self.snap_potential_snappers)
        original_snap_idx = self.snap_current_snapper_idx
        original_snap_log = list(self.snap_results_log)

        def change_snap_start():
            self.snap_phase_active = True
            self.snap_discarded_card = discarded_card
            self.snap_potential_snappers = ordered_snappers
            self.snap_current_snapper_idx = 0
            self.snap_results_log = []  # Clear log for this new phase

        def undo_snap_start():
            self.snap_phase_active = original_snap_phase
            self.snap_discarded_card = original_snap_card
            self.snap_potential_snappers = original_snap_potentials
            self.snap_current_snapper_idx = original_snap_idx
            self.snap_results_log = original_snap_log

        delta_snap_start = (
            "start_snap_phase",
            serialize_card(discarded_card),
            ordered_snappers,
        )
        self._add_change(
            change_snap_start, undo_snap_start, delta_snap_start, undo_stack, delta_list
        )
        return True

    def _end_snap_phase(self, undo_stack: Deque, delta_list: StateDelta):
        """Cleans up snap phase state and advances the main game turn."""
        if not self.snap_phase_active:
            return

        logger.debug("Ending snap phase.")
        original_snap_phase = self.snap_phase_active
        original_snap_card = self.snap_discarded_card
        original_snap_potentials = list(self.snap_potential_snappers)
        original_snap_idx = self.snap_current_snapper_idx
        # Keep snap_results_log until next snap starts

        def change_snap_end():
            self.snap_phase_active = False
            self.snap_discarded_card = None
            self.snap_potential_snappers = []
            self.snap_current_snapper_idx = 0

        def undo_snap_end():
            self.snap_phase_active = original_snap_phase
            self.snap_discarded_card = original_snap_card
            self.snap_potential_snappers = original_snap_potentials
            self.snap_current_snapper_idx = original_snap_idx

        delta_snap_end = ("end_snap_phase",)
        self._add_change(
            change_snap_end, undo_snap_end, delta_snap_end, undo_stack, delta_list
        )

        # Crucially, the turn advances *after* the snap phase concludes.
        # Ensure _advance_turn is available via self (from engine)
        if hasattr(self, "_advance_turn") and callable(self._advance_turn):
            self._advance_turn(undo_stack, delta_list)
        else:
            logger.error(
                "Snap Logic: Cannot advance turn after ending snap phase - _advance_turn missing."
            )

    def _get_snap_target_rank_str(self) -> str:
        """Helper for __str__."""
        if self.snap_phase_active and self.snap_discarded_card:
            # Ensure it's a card object before accessing rank
            return str(getattr(self.snap_discarded_card, "rank", "Invalid"))
        return "N/A"
