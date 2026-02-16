# src/cfr/recursion_mixin.py
"""
Mixin class originally intended for CFR+ recursive traversal logic.
The core recursion is now handled by src/cfr/worker.py for parallelization.
This mixin retains helper methods related to observation creation/filtering,
delegating to the canonical implementations in worker.py.
"""

import logging
from typing import Dict, List, Optional

from ..agent_state import AgentObservation
from ..constants import (
    GameAction,
    CardObject,
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

    # --- Observation Helper Methods (delegated to worker.py implementations) ---

    def _filter_observation(
        self, obs: AgentObservation, observer_id: int
    ) -> AgentObservation:
        """Creates a player-specific view of the observation, masking private info."""
        from .worker import _filter_observation as _worker_filter_observation

        return _worker_filter_observation(obs, observer_id)

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
        king_swap_indices: Optional[tuple] = None,  # (own_idx, opp_idx) for king swap
    ) -> AgentObservation:
        """Creates the AgentObservation object based on the state *after* the action.

        Delegates to the canonical implementation in worker.py.
        """
        from .worker import _create_observation as _worker_create_observation

        obs = _worker_create_observation(
            prev_state,
            action,
            next_state,
            acting_player,
            snap_results,
            king_swap_indices=king_swap_indices,
        )
        if obs is None:
            # Fallback: return a minimal observation on error
            logger.error(
                "Delegated _create_observation returned None. Creating minimal fallback."
            )
            from ..constants import NUM_PLAYERS

            obs = AgentObservation(
                acting_player=acting_player,
                action=action,
                discard_top_card=next_state.get_discard_top(),
                player_hand_sizes=[
                    next_state.get_player_card_count(i) for i in range(NUM_PLAYERS)
                ],
                stockpile_size=next_state.get_stockpile_size(),
                is_game_over=next_state.is_terminal(),
                current_turn=next_state.get_turn_number(),
            )
        return obs
