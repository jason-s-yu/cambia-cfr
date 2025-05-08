"""src/card.py"""

from dataclasses import dataclass, field
from typing import Optional, ClassVar, Dict
import uuid
import logging

from .constants import (
    ACE,
    JACK,
    QUEEN,
    KING,
    TEN,
    ALL_RANKS_STR,
    ALL_SUITS,
    RED_SUITS,
    JOKER_RANK_STR,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class Card:
    """Represents a game card with rank, suit, and calculated value."""

    rank: str = field(compare=True)  # e.g., 'A', 'T', 'K', JOKER_RANK_STR
    suit: Optional[str] = field(
        default=None, compare=False
    )  # e.g., 'H', 'S', None for Joker
    id: uuid.UUID = field(
        default_factory=uuid.uuid4, compare=False, repr=False
    )  # Unique ID if needed

    # Define value mapping directly using imported constants
    # Corrected bug: Handle non-numeric ranks explicitly
    _value_map: ClassVar[Dict[str, int]] = {
        JOKER_RANK_STR: 0,  # Use JOKER_RANK_STR
        ACE: 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        TEN: 10,  # Use TEN constant for 'T'
        JACK: 11,
        QUEEN: 12,
        KING: 13,  # Base value for Black King
        # Red King value handled dynamically in property
    }

    @property
    def value(self) -> int:
        """Calculates the score value of the card, considering King color."""
        if self.rank == KING:
            return -1 if self.suit in RED_SUITS else 13
        try:
            return self._value_map[self.rank]
        except KeyError as exc:
            # This should not happen if constants and validation are correct
            logger.error(
                "FATAL: Could not find value for rank '%s' in _value_map.", self.rank
            )
            raise ValueError(f"Invalid rank '{self.rank}' encountered.") from exc

    def __post_init__(self):
        # Basic validation using constants
        if self.rank not in ALL_RANKS_STR:
            raise ValueError(f"Invalid card rank: '{self.rank}'")
        if (
            self.rank != JOKER_RANK_STR and self.suit not in ALL_SUITS
        ):  # Use JOKER_RANK_STR
            raise ValueError(f"Invalid suit '{self.suit}' for rank '{self.rank}'")
        if self.rank == JOKER_RANK_STR and self.suit is not None:  # Use JOKER_RANK_STR
            raise ValueError(
                f"Joker ({JOKER_RANK_STR}) cannot have a suit ('{self.suit}')"
            )

    def __str__(self) -> str:
        # Use '10' for display if rank is TEN ('T')
        display_rank = "10" if self.rank == TEN else self.rank
        return f"{display_rank}{self.suit or ''}"

    def __repr__(self) -> str:
        # Use standard repr format
        return f"Card(rank='{self.rank}', suit='{self.suit}')"


# --- Standard Deck Creation ---
def create_standard_deck(include_jokers: int = 2) -> list["Card"]:
    """Creates a standard 52-card deck plus optional jokers."""
    # Use ALL_RANKS_STR from constants
    deck = [
        Card(rank, suit)
        for rank in ALL_RANKS_STR
        if rank != JOKER_RANK_STR
        for suit in ALL_SUITS
    ]  # Use JOKER_RANK_STR
    deck.extend(
        [Card(rank=JOKER_RANK_STR) for _ in range(include_jokers)]
    )  # Use JOKER_RANK_STR
    return deck
