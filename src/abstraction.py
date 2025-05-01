# src/abstraction.py
from typing import Optional
from .card import Card
from .constants import (
    CardBucket,
    DecayCategory,
    JOKER_RANK_STR,
    KING,
    ACE,
    TWO,
    THREE,
    FOUR,  # Use JOKER_RANK_STR
    FIVE,
    SIX,
    SEVEN,
    EIGHT,
    NINE,
    TEN,
    JACK,
    QUEEN,
    RED_SUITS,
)


def get_card_bucket(card: Optional[Card]) -> CardBucket:
    """Maps a Card object to its corresponding CardBucket."""
    if card is None:
        return CardBucket.UNKNOWN  # Or handle appropriately

    rank = card.rank
    suit = card.suit

    if rank == JOKER_RANK_STR:  # Use JOKER_RANK_STR
        return CardBucket.ZERO
    if rank == KING:
        return CardBucket.NEG_KING if suit in RED_SUITS else CardBucket.HIGH_KING
    if rank == ACE:
        return CardBucket.ACE
    if rank in [TWO, THREE, FOUR]:
        return CardBucket.LOW_NUM
    if rank in [FIVE, SIX]:
        return CardBucket.MID_NUM
    if rank in [SEVEN, EIGHT]:
        return CardBucket.PEEK_SELF
    if rank in [NINE, TEN]:
        return CardBucket.PEEK_OTHER
    if rank in [JACK, QUEEN]:
        return CardBucket.SWAP_BLIND

    # Should be unreachable with valid cards
    raise ValueError(f"Could not determine bucket for card: {card}")


def decay_bucket(bucket: CardBucket) -> DecayCategory:
    """Maps a specific CardBucket to a broader DecayCategory."""
    if bucket in [
        CardBucket.ZERO,
        CardBucket.NEG_KING,
        CardBucket.ACE,
        CardBucket.LOW_NUM,
    ]:
        return DecayCategory.LIKELY_LOW
    if bucket in [CardBucket.MID_NUM, CardBucket.PEEK_SELF]:
        return DecayCategory.LIKELY_MID
    if bucket in [CardBucket.PEEK_OTHER, CardBucket.SWAP_BLIND, CardBucket.HIGH_KING]:
        return DecayCategory.LIKELY_HIGH
    # If already unknown or not applicable, keep it unknown
    return DecayCategory.UNKNOWN
