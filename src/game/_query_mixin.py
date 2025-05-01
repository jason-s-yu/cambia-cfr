"""
src/game/_query_mixin.py

This module implements card ability mixins for the game engine
"""

import logging
from typing import List, Optional, Set

from ..card import Card
from ..constants import (
    GameAction,
    ActionCallCambia,
    ActionDrawStockpile,
    ActionDrawDiscard,
)

logger = logging.getLogger(__name__)


class QueryMixin:
    """Mixin containing query methods and legal action calculation for CambiaGameState."""

    # --- Direct State Queries ---

    def get_player_hand(self, player_index: int) -> List[Card]:
        """Get the cards in a player's hand."""
        if 0 <= player_index < len(self.players) and hasattr(
            self.players[player_index], "hand"
        ):
            return self.players[player_index].hand
        logger.error(
            "Invalid player index %s or player object missing hand in get_player_hand.",
            player_index,
        )
        return []

    def get_opponent_index(self, player_index: int) -> int:
        """Returns the index of the opponent player, assuming 1v1."""
        return 1 - player_index

    def get_player_card_count(self, player_index: int) -> int:
        """Returns the number of cards in the player's hand."""
        if 0 <= player_index < len(self.players) and hasattr(
            self.players[player_index], "hand"
        ):
            return len(self.players[player_index].hand)
        logger.warning(
            "Invalid player index %s or missing hand in get_player_card_count. Returning 0.",
            player_index,
        )
        return 0

    def get_stockpile_size(self) -> int:
        """Get the size of the stockpile."""
        return len(self.stockpile)

    def get_discard_top(self) -> Optional[Card]:
        """Get the top card in the discard pile. Returns None if empty."""
        return self.discard_pile[-1] if self.discard_pile else None

    def get_turn_number(self) -> int:
        """Returns the current turn number (0-indexed)."""
        return self._turn_number

    def is_terminal(self) -> bool:
        """Returns True if the game is over."""
        return self._game_over

    def get_utility(self, player_id: int) -> float:
        """Returns the final utility for the specified player."""
        if not self.is_terminal():
            logger.warning("get_utility called on non-terminal state!")
            return 0.0
        # Ensure scores are calculated if game ended but calculation hasn't run/set attributes
        # Check specifically against initial state to avoid recalculating ties
        initial_utilities = [0.0] * self.num_players
        if (
            self._game_over
            and self._winner is None  # Winner is None for ties or uncalculated
            and self._utilities
            == initial_utilities  # Only recalc if utilities are default
        ):
            logger.debug(
                "get_utility called on terminal state, but winner/utilities not set. "
                "Calculating scores now."
            )
            # _calculate_final_scores should be available via self (from base or another mixin)
            self._calculate_final_scores(set_attributes=True)
        if 0 <= player_id < self.num_players:
            return self._utilities[player_id]
        logger.error("Invalid player index %d requested for utility.", player_id)
        return 0.0

    def get_player_turn(self) -> int:
        """In Cambia (as implemented), the player whose turn it is might not be the one acting
        (e.g., during snap or pending actions). This returns the nominal turn owner."""
        return self.current_player_index

    def get_acting_player(self) -> int:
        """Returns the index of the player who needs to act *now*."""
        if self.snap_phase_active:
            if self.snap_current_snapper_idx < len(self.snap_potential_snappers):
                potential_snapper = self.snap_potential_snappers[
                    self.snap_current_snapper_idx
                ]
                # Check validity (should always be true if state is consistent)
                if 0 <= potential_snapper < len(self.players) and hasattr(
                    self.players[potential_snapper], "hand"
                ):
                    return potential_snapper
                else:
                    logger.error(
                        "Snap phase acting player index %d invalid or missing hand.",
                        potential_snapper,
                    )
                    return -1  # Indicates an error state
            else:
                # This case implies the snap phase should have ended but didn't.
                logger.error("Snap phase active but snapper index out of bounds.")
                return -1  # Indicates an error state
        elif self.pending_action and self.pending_action_player is not None:
            pending_player = self.pending_action_player
            # Check validity
            if 0 <= pending_player < len(self.players) and hasattr(
                self.players[pending_player], "hand"
            ):
                return pending_player
            else:
                logger.error(
                    "Pending action player index %d invalid or missing hand.",
                    pending_player,
                )
                return -1  # Indicates an error state
        elif not self._game_over:
            current_player = self.current_player_index
            # Check validity
            if 0 <= current_player < len(self.players) and hasattr(
                self.players[current_player], "hand"
            ):
                return current_player
            else:
                logger.error(
                    "Current player index %d invalid or missing hand in active game.",
                    current_player,
                )
                return -1  # Indicates an error state
        else:
            # Game is over, no one is acting
            return -1

    # --- Legal Actions ---

    def get_legal_actions(self) -> Set[GameAction]:
        """Returns the set of valid actions for the current acting player."""
        legal_actions: Set[GameAction] = set()

        if self._game_over:
            return legal_actions

        acting_player = self.get_acting_player()
        if acting_player == -1:
            # Error condition already logged by get_acting_player
            return legal_actions  # No actions possible in error state

        # Delegate based on game phase
        if self.snap_phase_active:
            # _get_legal_snap_actions should be defined in SnapLogicMixin
            return self._get_legal_snap_actions(acting_player)
        elif self.pending_action:
            # _get_legal_pending_actions should be defined in AbilityMixin
            return self._get_legal_pending_actions(acting_player)
        else:
            # Standard start-of-turn actions
            return self._get_legal_start_turn_actions(acting_player)

    def _get_legal_start_turn_actions(self, player: int) -> Set[GameAction]:
        """Calculates legal actions at the start of a regular turn."""
        legal_actions: Set[GameAction] = set()

        # Basic turn actions
        can_draw_stockpile = bool(self.stockpile) or (len(self.discard_pile) > 1)
        can_draw_discard = self.house_rules.allowDrawFromDiscardPile and self.discard_pile

        if can_draw_stockpile:
            legal_actions.add(ActionDrawStockpile())
        if can_draw_discard:
            legal_actions.add(ActionDrawDiscard())

        # Cambia action
        cambia_allowed_round = self.house_rules.cambia_allowed_round
        # Calculate current round number (integer division)
        current_round = self._turn_number // self.num_players
        if self.cambia_caller_id is None and (current_round >= cambia_allowed_round):
            legal_actions.add(ActionCallCambia())

        # Check for stalemate-like conditions only if no actions found yet
        if not legal_actions and not (can_draw_stockpile or can_draw_discard):
            logger.warning(
                "No legal start-of-turn actions and cannot draw/reshuffle for P%s. State: %s.",
                player,
                self,  # Relies on __str__ being available
            )
            # Game end should be triggered by _check_game_end after turn advance fails or here.
            # Returning empty set signals a potential issue.
        elif not legal_actions:
            logger.warning(
                "No legal start-of-turn actions found for player %s in state: %s.",
                player,
                self,  # Relies on __str__
            )
            # This might indicate a state where only Cambia is possible but not allowed yet, etc.

        return legal_actions

    # --- String Representation ---

    def __str__(self) -> str:
        state_desc = ""
        actor = self.get_acting_player()
        actor_str = f"P{actor}" if actor != -1 else "N/A"

        if self.snap_phase_active:
            # Use helper method from SnapLogicMixin if available, otherwise basic info
            snap_target_rank = getattr(self, "_get_snap_target_rank_str", lambda: "N/A")()
            state_desc = f"SnapPhase(Actor: {actor_str}, Target: {snap_target_rank})"
        elif self.pending_action:
            # Use helper method from AbilityMixin if available
            pending_action_name = getattr(
                self, "_get_pending_action_name", lambda: "N/A"
            )()
            state_desc = f"Pending(Actor: {actor_str}, Action: {pending_action_name})"
        elif self._game_over:
            winner_str = f"W:{self._winner}" if self._winner is not None else "W:Tie"
            utils_str = f"U:[{', '.join(f'{u:.1f}' for u in self._utilities)}]"
            state_desc = f"GameOver({winner_str}, {utils_str})"
        else:
            state_desc = f"Turn: {actor_str}"

        discard_top_str = str(self.get_discard_top()) if self.discard_pile else "[]"

        hand_lens = []
        for _, p in enumerate(self.players):
            if hasattr(p, "hand") and isinstance(p.hand, list):
                hand_lens.append(str(len(p.hand)))
            else:
                hand_lens.append("ERR")  # Error state for player hand

        return (
            f"GameState(T#{self._turn_number}, {state_desc}, "
            f"Stock:{len(self.stockpile)}, Disc:{discard_top_str}, "
            f"Hands:[{','.join(hand_lens)}], "
            f"Cambia:{self.cambia_caller_id if self.cambia_caller_id is not None else 'N'}"
            f"({self.turns_after_cambia}))"
        )
