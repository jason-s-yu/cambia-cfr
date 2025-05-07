"""src/game/helpers.py"""

import logging
from typing import Optional
from ..card import Card
from ..constants import (
    KING,
    QUEEN,
    JACK,
    NINE,
    TEN,
    SEVEN,
    EIGHT,
)

logger = logging.getLogger(__name__)


def card_has_discard_ability(card: Optional[Card]) -> bool:
    """Checks if a card has an ability when discarded from draw."""
    if not card:
        return False
    return card.rank in [SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING]


def serialize_card(card: Optional[Card]) -> Optional[str]:
    """Serializes a card to string or None."""
    return str(card) if card else None
