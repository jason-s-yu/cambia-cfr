"""src/game/player_state.py"""

from dataclasses import dataclass, field
from typing import List, Tuple, TYPE_CHECKING

# Use TYPE_CHECKING guard for Card import
if TYPE_CHECKING:
    from ..card import Card


@dataclass
class PlayerState:
    # Use string forward reference for Card type hint
    hand: List["Card"] = field(default_factory=list)
    initial_peek_indices: Tuple[int, ...] = (0, 1)
