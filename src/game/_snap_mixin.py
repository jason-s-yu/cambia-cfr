"""
src/game/_snap_mixin.py

This module implements snapping mixins for the game engine
"""

import logging
import copy
from typing import Set, Dict, Any, Optional, Deque, Tuple

# Use relative imports
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
            logger.error("Called _get_legal_snap_actions when snap phase is not active.")
            return legal_actions

        if not (
            0 <= acting_player < len(self.players)
            and hasattr(self.players[acting_player], "hand")
        ):
            logger.error(
                "Snap phase legal actions: Acting player %s object invalid or missing hand.",
                acting_player,
            )
            return legal_actions  # Should not happen if get_acting_player is correct

        snapper_hand = self.players[acting_player].hand
        if self.snap_discarded_card is None:
            logger.error("Snap phase active but snap_discarded_card is None.")
            # Attempt to gracefully end snap phase? Or just return no actions?
            return legal_actions

        target_rank = self.snap_discarded_card.rank

        # Basic validation of hand contents
        if not all(isinstance(card, Card) for card in snapper_hand):
            logger.error(
                "Snap phase legal actions: Player %s's hand contains non-Card objects: %s",
                acting_player,
                snapper_hand,
            )
            # Treat as error state, no valid snaps possible
            return legal_actions

        # Always possible to pass
        legal_actions.add(ActionPassSnap())

        # Check for own snaps
        for i, card in enumerate(snapper_hand):
            if card.rank == target_rank:
                legal_actions.add(ActionSnapOwn(own_card_hand_index=i))

        # Check for opponent snaps if allowed
        if self.house_rules.allowOpponentSnapping:
            opponent_idx = self.get_opponent_index(
                acting_player
            )  # Method from QueryMixin/base
            if 0 <= opponent_idx < len(self.players) and hasattr(
                self.players[opponent_idx], "hand"
            ):
                opponent_hand = self.players[opponent_idx].hand
                if not all(isinstance(card, Card) for card in opponent_hand):
                    logger.error(
                        "Snap phase legal actions: Opponent %s's hand contains non-Card objects: %s",
                        opponent_idx,
                        opponent_hand,
                    )
                    # Cannot check opponent hand, don't add actions
                else:
                    for i, card in enumerate(opponent_hand):
                        if card.rank == target_rank:
                            legal_actions.add(
                                ActionSnapOpponent(opponent_target_hand_index=i)
                            )
            else:
                # This might happen if opponent object is somehow invalid
                logger.warning(
                    "Snap phase legal actions: Opponent object %s invalid or missing hand, cannot check for SnapOpponent.",
                    opponent_idx,
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
        Processes an action taken during the snap phase. Modifies state via _add_change.
        Returns True if the action was successfully processed (even if it was a pass or penalty),
        False if there was a fundamental error (e.g., wrong player).
        """
        if not self.snap_phase_active:
            logger.error("_handle_snap_action called when snap phase not active.")
            return False  # Should not happen

        # Verify the action is coming from the correct player for this point in the snap sequence
        if acting_player != self.snap_potential_snappers[self.snap_current_snapper_idx]:
            logger.error(
                f"Snap Action {action} received from P{acting_player}, but expected action from P{self.snap_potential_snappers[self.snap_current_snapper_idx]}. Ignoring."
            )
            return False  # Indicate action was ignored

        logger.debug(
            f"Snap Phase (Turn {self._turn_number}): Player {acting_player} choosing action: {action}"
        )

        if self.snap_discarded_card is None:
            logger.error("Cannot process snap action: snap_discarded_card is None.")
            self._end_snap_phase(undo_stack, delta_list)  # Try to clean up
            return True  # Handled the error state by ending phase

        target_rank = self.snap_discarded_card.rank
        snap_success = False
        snap_penalty = False
        removed_card_info: Optional[Tuple[int, int, Card]] = (
            None  # (player_idx, hand_idx, card)
        )
        snapped_opponent_card_info: Optional[Tuple[int, int, Card]] = (
            None  # (player_idx, hand_idx, card)
        )
        attempted_card: Optional[Card] = None
        action_type_str = type(action).__name__
        snap_details: Dict[str, Any] = {}

        # Helper for logging snap results (uses _add_change implicitly via self)
        def log_snap_result(details_dict):
            original_log = list(self.snap_results_log)

            def change():
                self.snap_results_log.append(details_dict)

            def undo():
                self.snap_results_log = original_log

            # _add_change is expected to be available via self (from base class or another mixin)
            self._add_change(
                change, undo, ("snap_log_append", details_dict), undo_stack, delta_list
            )

        # --- Process Specific Snap Actions ---
        if isinstance(action, ActionPassSnap):
            logger.debug(f"Player {acting_player} passes snap.")
            snap_details = {
                "snapper": acting_player,
                "action_type": action_type_str,
                "target_rank": target_rank,
                "success": False,
                "penalty": False,
                "details": "Passed",
                "snapped_card": None,
            }
            log_snap_result(snap_details)

        elif isinstance(action, ActionSnapOwn):
            snap_idx = action.own_card_hand_index
            hand = self.players[acting_player].hand
            if 0 <= snap_idx < len(hand):
                attempted_card = hand[snap_idx]
                if not isinstance(attempted_card, Card):
                    logger.error(
                        f"SnapOwn Error: Item at index {snap_idx} is not a Card: {attempted_card}."
                    )
                    snap_penalty = True
                elif attempted_card.rank == target_rank:
                    card_to_remove = hand[snap_idx]

                    # --- State Change: Remove card, add to discard ---
                    def change_snap_own():
                        removed = self.players[acting_player].hand.pop(snap_idx)
                        self.discard_pile.append(removed)

                    def undo_snap_own():
                        popped_discard = (
                            self.discard_pile.pop()
                        )  # Assumes it's the one just added
                        self.players[acting_player].hand.insert(snap_idx, popped_discard)
                        if popped_discard is not card_to_remove:
                            logger.warning("Undo SnapOwn mismatch")  # Basic sanity check

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
                    # --- End State Change ---
                    snap_success = True
                    removed_card_info = (acting_player, snap_idx, card_to_remove)
                    logger.info(
                        f"Player {acting_player} snaps own {card_to_remove} (matching {target_rank}) from index {snap_idx}. Hand size: {len(self.players[acting_player].hand)}"
                    )
                else:  # Card rank doesn't match
                    snap_penalty = True
            else:  # Invalid index
                snap_penalty = True

            if snap_penalty:
                logger.warning(
                    f"Player {acting_player} attempted invalid Snap Own: {action} (Target Rank: {target_rank}, Attempted Card: {attempted_card}). Applying penalty."
                )
                # _apply_penalty is expected to be available via self
                penalty_deltas = self._apply_penalty(
                    acting_player, self.house_rules.penaltyDrawCount, undo_stack
                )
                delta_list.extend(penalty_deltas)

            snap_details = {
                "snapper": acting_player,
                "action_type": action_type_str,
                "target_rank": target_rank,
                "success": snap_success,
                "penalty": snap_penalty,
                "removed_own_index": removed_card_info[1] if removed_card_info else None,
                "snapped_card": (
                    serialize_card(removed_card_info[2]) if removed_card_info else None
                ),
                "attempted_card": (
                    serialize_card(attempted_card)
                    if snap_penalty and attempted_card
                    else None
                ),
            }
            log_snap_result(snap_details)

        elif isinstance(action, ActionSnapOpponent):
            if not self.house_rules.allowOpponentSnapping:
                logger.error(
                    "Invalid Action: SnapOpponent attempted but rule disallowed."
                )
                snap_penalty = True
                penalty_deltas = self._apply_penalty(
                    acting_player, self.house_rules.penaltyDrawCount, undo_stack
                )
                delta_list.extend(penalty_deltas)
                attempted_card = (
                    None  # Cannot determine attempted card if rule disallowed
                )
            else:
                opp_idx = self.get_opponent_index(acting_player)
                if not (
                    0 <= opp_idx < len(self.players)
                    and hasattr(self.players[opp_idx], "hand")
                ):
                    logger.error(
                        f"SnapOpponent Error: Opponent object {opp_idx} invalid or missing hand."
                    )
                    snap_penalty = True
                    penalty_deltas = self._apply_penalty(
                        acting_player, self.house_rules.penaltyDrawCount, undo_stack
                    )
                    delta_list.extend(penalty_deltas)
                    attempted_card = None
                else:
                    opp_hand = self.players[opp_idx].hand
                    target_opp_hand_idx = action.opponent_target_hand_index
                    if 0 <= target_opp_hand_idx < len(opp_hand):
                        attempted_card = opp_hand[target_opp_hand_idx]
                        if not isinstance(attempted_card, Card):
                            logger.error(
                                f"SnapOpponent Error: Item at opponent index {target_opp_hand_idx} is not a Card: {attempted_card}."
                            )
                            snap_penalty = True
                        elif attempted_card.rank == target_rank:
                            card_to_remove = opp_hand[target_opp_hand_idx]

                            # --- State Change: Remove opponent card ---
                            def change_snap_opp_remove():
                                self.players[opp_idx].hand.pop(target_opp_hand_idx)

                            def undo_snap_opp_remove():
                                self.players[opp_idx].hand.insert(
                                    target_opp_hand_idx, card_to_remove
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
                            # --- End State Change ---
                            snap_success = True
                            snapped_opponent_card_info = (
                                opp_idx,
                                target_opp_hand_idx,
                                card_to_remove,
                            )
                            logger.info(
                                f"Player {acting_player} snaps opponent's {card_to_remove} at index {target_opp_hand_idx}. Requires move."
                            )

                            # --- State Change: Set pending action for the MOVE ---
                            original_pending = (
                                self.pending_action,
                                self.pending_action_player,
                                copy.deepcopy(self.pending_action_data),
                            )
                            original_snap_active = (
                                self.snap_phase_active
                            )  # Should be True

                            def change_pending_move():
                                # Use placeholder type; specific indices come from next action
                                self.pending_action = ActionSnapOpponentMove(
                                    own_card_to_move_hand_index=-1,
                                    target_empty_slot_index=-1,
                                )
                                self.pending_action_player = (
                                    acting_player  # The snapper makes the move choice
                                )
                                self.pending_action_data = {
                                    "target_empty_slot_index": target_opp_hand_idx
                                }  # Store target slot
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
                                {"target_empty_slot_index": target_opp_hand_idx},
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
                            delta_list.append(
                                delta_snap_active
                            )  # Log snap active change separately
                            # --- End State Change ---

                            snap_details = {
                                "snapper": acting_player,
                                "action_type": action_type_str,
                                "target_rank": target_rank,
                                "success": snap_success,
                                "penalty": False,
                                "removed_opponent_index": target_opp_hand_idx,
                                "snapped_card": serialize_card(card_to_remove),
                            }
                            log_snap_result(snap_details)
                            # Return immediately: next action is the MOVE decision by the snapper
                            return True  # Action processed, waiting for next sub-action
                        else:  # Card rank doesn't match
                            snap_penalty = True
                    else:  # Invalid index
                        snap_penalty = True

                    if snap_penalty:
                        logger.warning(
                            f"Player {acting_player} attempted invalid Snap Opponent: {action} (Target Rank: {target_rank}, Attempted Card: {attempted_card}). Applying penalty."
                        )
                        penalty_deltas = self._apply_penalty(
                            acting_player, self.house_rules.penaltyDrawCount, undo_stack
                        )
                        delta_list.extend(penalty_deltas)

            # Log failed/penalized SnapOpponent attempts
            if not snap_success:
                snap_details = {
                    "snapper": acting_player,
                    "action_type": action_type_str,
                    "target_rank": target_rank,
                    "success": False,
                    "penalty": snap_penalty,
                    "snapped_card": None,
                    "attempted_card": (
                        serialize_card(attempted_card)
                        if snap_penalty and attempted_card
                        else None
                    ),
                }
                log_snap_result(snap_details)

        else:
            # This case should ideally be prevented by get_legal_actions
            logger.error(
                f"Invalid action type {type(action)} received during snap phase processing."
            )
            snap_details = {
                "snapper": acting_player,
                "action_type": "InvalidAction",
                "target_rank": target_rank,
                "success": False,
                "penalty": False,
                "details": f"Received {type(action).__name__}",
                "snapped_card": None,
            }
            log_snap_result(snap_details)
            # Should we apply a penalty here? Maybe just advance snap turn.

        # --- Advance Snap Turn or End Phase ---
        # Do not advance if a pending move action was just set
        if not (isinstance(action, ActionSnapOpponent) and snap_success):
            # --- State Change: Advance snap index ---
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
            # --- End State Change ---

            # Check if snap phase ends
            if self.snap_current_snapper_idx >= len(self.snap_potential_snappers):
                self._end_snap_phase(
                    undo_stack, delta_list
                )  # Also advances main game turn
            # else: The next snapper will act. State remains in snap phase.

        # Game end check should happen after the phase potentially ends/turn advances
        # self._check_game_end(undo_stack, delta_list) # Called by _end_snap_phase if it runs

        return True  # Action was processed

    # --- Snap Phase Initiation and Termination ---

    def _initiate_snap_phase(
        self, discarded_card: Card, undo_stack: Deque, delta_list: StateDelta
    ) -> bool:
        """
        Checks if a discard triggers a snap phase, sets up snap state if so.
        Modifies state via _add_change. Returns True if snap phase started, False otherwise.
        """
        if self.cambia_caller_id is not None:
            # Cannot snap if Cambia has been called (simplification, adjust if rules differ)
            # Or, more specifically, the Cambia caller cannot snap. Rules vary if others can.
            # Let's assume NO snapping after Cambia call for simplicity here.
            # logger.debug("Snap check skipped: Cambia called.")
            # return False
            pass  # Let's allow non-callers to snap for now.

        potential_indices = []
        target_rank = discarded_card.rank

        for p_idx in range(self.num_players):
            if p_idx == self.cambia_caller_id:
                continue  # Cambia caller definitely cannot snap

            # Basic player validation
            if not (
                0 <= p_idx < len(self.players) and hasattr(self.players[p_idx], "hand")
            ):
                logger.warning(
                    f"Initiate Snap Check: Player {p_idx} invalid or missing hand. Skipping."
                )
                continue

            hand = self.players[p_idx].hand
            # Basic hand validation
            if not all(isinstance(card, Card) for card in hand):
                logger.error(
                    f"Initiate Snap Check: Player {p_idx}'s hand contains non-Card objects: {hand}. Skipping snap check."
                )
                continue

            # Check own hand
            can_snap_own = any(card.rank == target_rank for card in hand)

            # Check opponent hand if allowed
            can_snap_opponent = False
            if self.house_rules.allowOpponentSnapping:
                opp_idx = self.get_opponent_index(p_idx)
                # Opponent also cannot be the Cambia caller
                if opp_idx != self.cambia_caller_id:
                    if 0 <= opp_idx < len(self.players) and hasattr(
                        self.players[opp_idx], "hand"
                    ):
                        opp_hand = self.players[opp_idx].hand
                        if not all(isinstance(card, Card) for card in opp_hand):
                            logger.error(
                                "Initiate Snap Check: Opponent %d's hand contains non-Card objects. Skipping snap-opp check for P%d.",
                                opp_idx,
                                p_idx,
                            )
                        else:
                            can_snap_opponent = any(
                                card.rank == target_rank for card in opp_hand
                            )
                    else:
                        logger.warning(
                            "Initiate Snap Check: Opponent %d invalid/missing hand for P%d checking SnapOpponent.",
                            opp_idx,
                            p_idx,
                        )

            if can_snap_own or can_snap_opponent:
                potential_indices.append(p_idx)

        started_snap = False
        if potential_indices:
            logger.debug(
                "Discard of %s triggers potential snap phase for players %s.",
                discarded_card,
                potential_indices,
            )

            # Determine who discarded the card. This is tricky.
            # If called immediately after resolving a pending action, self.pending_action_player might be the discarder.
            # If called after a standard discard/replace, it's the player whose turn just ended conceptually.
            # Heuristic: Use pending_action_player if set and matches the action type, else use previous player index.
            discarder_player = None
            if (
                self.pending_action_player is not None
            ):  # Check if pending action was just resolved
                # We don't store the *resolved* action, just the *next* pending one.
                # So, this isn't reliable. Fallback to previous player.
                pass

            if discarder_player is None:
                # Assume it was the player whose turn it *was* before the current state.
                # If apply_action calls this *before* advancing turn, it's self.current_player_index.
                # If apply_action calls this *after* resolving but *before* advancing, it's self.current_player_index.
                # Let's assume it's the player whose turn it nominally is right now.
                # The snap order starts *after* them.
                # current_nominal_player = self.current_player_index
                # However, if a pending action was resolved, the 'discarder' was pending_action_player.
                # Let's refine: use pending_action_player if it was just cleared, else current_player_index.
                # This requires more state tracking in apply_action.
                # Simplification: Assume the player whose turn it *is* (current_player_index)
                # effectively caused the discard (even if via replace), and snap order starts after them.
                discarder_player = self.current_player_index  # Tentative definition
                logger.debug(
                    "Snap Phase: Assuming discard caused by P%d's action.",
                    discarder_player,
                )

            # Determine snap order: Start from player *after* the discarder.
            ordered_snappers = []
            for i in range(
                1, self.num_players
            ):  # Check players *after* the discarder first
                check_p_idx = (discarder_player + i) % self.num_players
                if check_p_idx in potential_indices:
                    ordered_snappers.append(check_p_idx)
            # Does the discarder get to snap? Rules vary. Assuming YES, but they go last.
            if discarder_player in potential_indices:
                ordered_snappers.append(discarder_player)

            if ordered_snappers:
                # --- State Change: Start Snap Phase ---
                original_snap_phase = self.snap_phase_active
                original_snap_card = self.snap_discarded_card
                original_snap_potentials = list(self.snap_potential_snappers)
                original_snap_idx = self.snap_current_snapper_idx
                original_snap_log = list(self.snap_results_log)  # Preserve old log

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
                    self.snap_results_log = original_snap_log  # Restore previous log

                delta_snap_start = (
                    "start_snap_phase",
                    serialize_card(discarded_card),
                    ordered_snappers,
                )
                self._add_change(
                    change_snap_start,
                    undo_snap_start,
                    delta_snap_start,
                    undo_stack,
                    delta_list,
                )
                # --- End State Change ---
                started_snap = True
                logger.info(
                    "Snap phase started. Discard: %s. Potential snappers (ordered): %s. P%d acts first.",
                    discarded_card,
                    ordered_snappers,
                    self.get_acting_player(),
                )

            else:
                logger.debug(
                    "Potential snappers list empty after ordering/filtering. No snap phase."
                )
                started_snap = False
        else:
            logger.debug(
                "No potential snappers found for discard of %s (Rank %s).",
                discarded_card,
                target_rank,
            )
            started_snap = False

        return started_snap

    def _end_snap_phase(self, undo_stack: Deque, delta_list: StateDelta):
        """Cleans up snap phase state and advances the main game turn."""
        if not self.snap_phase_active:
            return  # Nothing to end

        logger.debug("Ending snap phase.")
        # --- State Change: End Snap Phase ---
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
        # --- End State Change ---

        # Crucially, the turn advances *after* the snap phase concludes.
        # _advance_turn should be available via self
        self._advance_turn(undo_stack, delta_list)

    def _get_snap_target_rank_str(self) -> str:
        """Helper for __str__."""
        if self.snap_phase_active and self.snap_discarded_card:
            return str(self.snap_discarded_card.rank)
        return "N/A"
