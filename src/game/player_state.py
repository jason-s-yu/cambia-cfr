# src/game/player_state.py
from dataclasses import dataclass, field
from typing import List, Tuple
from ..card import Card # Use relative import

@dataclass
class PlayerState:
     hand: List[Card] = field(default_factory=list)
     initial_peek_indices: Tuple[int, ...] = (0, 1)