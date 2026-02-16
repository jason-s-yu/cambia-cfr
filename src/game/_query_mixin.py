"""
src/game/_query_mixin.py

Query methods and legal action calculation mixins for the game engine.
"""

import logging
from typing import List, Optional, Set

from src.game.helpers import serialize_card

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
            # Return a copy to prevent external modification? For now, return direct reference.
            return self.players[player_index].hand
        logger.error(
            "QueryMixin: Invalid P%d or missing hand in get_player_hand.", player_index
        )
        return []

    def get_opponent_index(self, player_index: int) -> int:
        """Returns the index of the opponent player, assuming 1v1."""
        if not (0 <= player_index < self.num_players):
            logger.error(
                "QueryMixin: Invalid player index %d for get_opponent_index (Num players: %d)",
                player_index,
                self.num_players,
            )
            return -1  # Indicate error
        return 1 - player_index

    def get_player_card_count(self, player_index: int) -> int:
        """Returns the number of cards in the player's hand."""
        if 0 <= player_index < len(self.players) and hasattr(
            self.players[player_index], "hand"
        ):
            return len(self.players[player_index].hand)
        logger.warning(
            "QueryMixin: Invalid P%d or missing hand in get_player_card_count. Returning 0.",
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
        # Ensure _game_over attribute exists
        return getattr(self, "_game_over", False)

    def get_utility(self, player_id: int) -> float:
        """Returns the final utility for the specified player."""
        if not self.is_terminal():
            logger.warning("QueryMixin: get_utility called on non-terminal state!")
            return 0.0
        # Ensure _calculate_final_scores method exists and scores are set
        if hasattr(self, "_utilities") and isinstance(self._utilities, list):
            initial_utilities = [0.0] * self.num_players
            if (
                self._utilities == initial_utilities
                and hasattr(self, "_calculate_final_scores")
                and callable(self._calculate_final_scores)
            ):
                logger.debug(
                    "get_utility called on terminal state, but scores not set. Calculating."
                )
                try:
                    self._calculate_final_scores(set_attributes=True)
                except Exception as e_score:
                    # JUSTIFIED: Final score calculation should not crash utility getter
                    logger.error(
                        "Error calculating final scores in get_utility: %s",
                        e_score,
                        exc_info=True,
                    )
                    # Return default utility on error
                    return 0.0

            if (
                0 <= player_id < self.num_players
                and len(self._utilities) == self.num_players
            ):
                return self._utilities[player_id]
            else:
                logger.error(
                    "QueryMixin: Invalid player ID %d or utilities array mismatch (%s) for get_utility.",
                    player_id,
                    self._utilities,
                )
                return 0.0
        else:
            logger.error(
                "QueryMixin: Utilities attribute missing or invalid type for get_utility."
            )
            return 0.0

    def get_player_turn(self) -> int:
        """Returns the index of the player whose turn it nominally is."""
        # Ensure current_player_index attribute exists
        return getattr(self, "current_player_index", -1)

    def get_acting_player(self) -> int:
        """Returns the index of the player who needs to act *now*."""
        # Check attributes exist before accessing
        snap_active = getattr(self, "snap_phase_active", False)
        pending_action = getattr(self, "pending_action", None)
        pending_player = getattr(self, "pending_action_player", None)
        snap_idx = getattr(self, "snap_current_snapper_idx", 0)
        potential_snappers = getattr(self, "snap_potential_snappers", [])

        if snap_active:
            if snap_idx < len(potential_snappers):
                potential_snapper = potential_snappers[snap_idx]
                if 0 <= potential_snapper < len(self.players) and hasattr(
                    self.players[potential_snapper], "hand"
                ):
                    return potential_snapper
                else:
                    logger.error(
                        "QueryMixin: Snap phase acting player index %d invalid or missing hand.",
                        potential_snapper,
                    )
                    return -1  # Error state
            else:
                logger.error(
                    "QueryMixin: Snap phase active but snapper index %d OOB (len %d).",
                    snap_idx,
                    len(potential_snappers),
                )
                return -1  # Error state
        elif pending_action and pending_player is not None:
            if 0 <= pending_player < len(self.players) and hasattr(
                self.players[pending_player], "hand"
            ):
                return pending_player
            else:
                logger.error(
                    "QueryMixin: Pending action player index %d invalid or missing hand.",
                    pending_player,
                )
                return -1  # Error state
        elif not self.is_terminal():
            current_player = getattr(self, "current_player_index", -1)
            if 0 <= current_player < len(self.players) and hasattr(
                self.players[current_player], "hand"
            ):
                return current_player
            else:
                logger.error(
                    "QueryMixin: Current player index %d invalid or missing hand in active game.",
                    current_player,
                )
                return -1  # Error state
        else:
            # Game is over
            return -1

    # --- Legal Actions ---

    def get_legal_actions(self) -> Set[GameAction]:
        """Returns the set of valid actions for the current acting player."""
        legal_actions: Set[GameAction] = set()

        if self.is_terminal():
            return legal_actions

        acting_player = self.get_acting_player()
        if acting_player == -1:
            logger.error("QueryMixin: Cannot get legal actions, invalid acting player.")
            return legal_actions

        # Delegate based on game phase
        try:
            if getattr(self, "snap_phase_active", False):
                if hasattr(self, "_get_legal_snap_actions") and callable(
                    self._get_legal_snap_actions
                ):
                    legal_actions = self._get_legal_snap_actions(acting_player)
                else:
                    logger.error(
                        "QueryMixin: Snap active but _get_legal_snap_actions missing."
                    )
            elif getattr(self, "pending_action", None):
                if hasattr(self, "_get_legal_pending_actions") and callable(
                    self._get_legal_pending_actions
                ):
                    legal_actions = self._get_legal_pending_actions(acting_player)
                else:
                    logger.error(
                        "QueryMixin: Pending action but _get_legal_pending_actions missing."
                    )
            else:
                # Standard start-of-turn actions
                if hasattr(self, "_get_legal_start_turn_actions") and callable(
                    self._get_legal_start_turn_actions
                ):
                    legal_actions = self._get_legal_start_turn_actions(acting_player)
                else:
                    logger.error(
                        "QueryMixin: Start of turn but _get_legal_start_turn_actions missing."
                    )

            if not legal_actions and not self.is_terminal():
                logger.warning(
                    "QueryMixin: No legal actions found for P%d in non-terminal state. State: %s, Snap: %s, Pending: %s",
                    acting_player,
                    self,
                    getattr(self, "snap_phase_active", "N/A"),
                    getattr(self, "pending_action", "N/A"),
                )

        except Exception as e_get_legal:
            # JUSTIFIED: Legal action calculation should not crash, return empty set for safety
            logger.exception(
                "QueryMixin: Error calculating legal actions for P%d: %s. State: %s",
                acting_player,
                e_get_legal,
                self,
            )
            legal_actions = set()  # Return empty set on error

        return legal_actions

    def _get_legal_start_turn_actions(self, player: int) -> Set[GameAction]:
        """Calculates legal actions at the start of a regular turn."""
        legal_actions: Set[GameAction] = set()

        # Basic validation
        if not (0 <= player < len(self.players)):
            return legal_actions

        can_draw_stockpile = bool(self.stockpile) or (len(self.discard_pile) > 1)
        can_draw_discard = self.house_rules.allowDrawFromDiscardPile and bool(
            self.discard_pile
        )

        if can_draw_stockpile:
            legal_actions.add(ActionDrawStockpile())
        if can_draw_discard:
            legal_actions.add(ActionDrawDiscard())

        # Cambia action
        cambia_allowed_round = self.house_rules.cambia_allowed_round
        current_round = self._turn_number // self.num_players
        if self.cambia_caller_id is None and (current_round >= cambia_allowed_round):
            legal_actions.add(ActionCallCambia())

        # Note: The stall warning from previous implementation is moved to get_legal_actions general check

        return legal_actions

    # --- String Representation ---

    def __str__(self) -> str:
        """Provides a concise string representation of the game state."""
        parts = []
        try:
            parts.append(f"T#{getattr(self, '_turn_number', -1)}")
            actor = self.get_acting_player()
            actor_str = f"Actor:P{actor}" if actor != -1 else "Actor:N/A"

            state_desc = "State:Unknown"
            if getattr(self, "snap_phase_active", False):
                snap_target_rank = "N/A"
                if hasattr(self, "_get_snap_target_rank_str") and callable(
                    self._get_snap_target_rank_str
                ):
                    snap_target_rank = self._get_snap_target_rank_str()
                state_desc = f"SnapPhase(Target:{snap_target_rank})"
            elif getattr(self, "pending_action", None):
                pending_name = "N/A"
                if hasattr(self, "_get_pending_action_name") and callable(
                    self._get_pending_action_name
                ):
                    pending_name = self._get_pending_action_name()
                state_desc = f"Pending({pending_name})"
            elif getattr(self, "_game_over", False):
                winner_str = f"W:{getattr(self, '_winner', 'N/A')}"
                utils = getattr(self, "_utilities", [])
                utils_str = (
                    f"U:[{','.join(f'{u:.1f}' for u in utils)}]" if utils else "U:[]"
                )
                state_desc = f"GameOver({winner_str},{utils_str})"
            else:
                state_desc = "Turn"

            parts.append(actor_str)
            parts.append(state_desc)

            discard_top = self.get_discard_top()
            parts.append(f"Disc:{serialize_card(discard_top) if discard_top else '[]'}")
            parts.append(f"Stock:{self.get_stockpile_size()}")

            hand_lens = []
            if hasattr(self, "players") and isinstance(self.players, list):
                for i in range(len(self.players)):
                    hand_lens.append(str(self.get_player_card_count(i)))
            else:
                hand_lens.append("ERR")
            parts.append(f"Hands:[{','.join(hand_lens)}]")

            cambia_id = getattr(self, "cambia_caller_id", None)
            cambia_turns = getattr(self, "turns_after_cambia", 0)
            parts.append(
                f"Cambia:{cambia_id if cambia_id is not None else 'N'}({cambia_turns})"
            )

        except Exception as e_str:
            # JUSTIFIED: String representation should not crash
            logger.error("Error generating GameState string representation: %s", e_str)
            return "GameState(Error)"

        return f"GameState({', '.join(parts)})"
