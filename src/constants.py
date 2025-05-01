# src/constants.py
import enum
from typing import NamedTuple, Any, Union

# Card Ranks (String representation)
ACE = "A"
TWO = "2"
THREE = "3"
FOUR = "4"
FIVE = "5"
SIX = "6"
SEVEN = "7"
EIGHT = "8"
NINE = "9"
TEN = "T"  # Represents 10
JACK = "J"
QUEEN = "Q"
KING = "K"
JOKER_RANK_STR = "R"  # Use 'R' for Joker consistently internally

NUMERIC_RANKS_STR = [TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN]
FACE_RANKS_STR = [JACK, QUEEN, KING]
ALL_RANKS_STR = [ACE] + NUMERIC_RANKS_STR + FACE_RANKS_STR + [JOKER_RANK_STR]

# Card Suits (String representation)
SPADES = "S"
HEARTS = "H"
DIAMONDS = "D"
CLUBS = "C"
RED_SUITS = [HEARTS, DIAMONDS]
BLACK_SUITS = [SPADES, CLUBS]
ALL_SUITS = [SPADES, HEARTS, DIAMONDS, CLUBS]


# --- Card Buckets Enum (Abstraction) ---
class CardBucket(enum.Enum):
    ZERO = 0  # Joker (Value: 0)
    NEG_KING = 1  # Red King (Value: -1, Ability: Look & Swap)
    ACE = 2  # Ace (Value: 1)
    LOW_NUM = 3  # 2, 3, 4 (Value: 2-4)
    MID_NUM = 4  # 5, 6 (Value: 5-6)
    PEEK_SELF = 5  # 7, 8 (Value: 7-8, Ability: Look Own)
    PEEK_OTHER = 6  # 9, T (Value: 9-10, Ability: Look Other's)
    SWAP_BLIND = 7  # Jack, Queen (Value: 11-12, Ability: Blind Swap)
    HIGH_KING = 8  # Black King (Value: 13, Ability: Look & Swap)
    UNKNOWN = 99  # Placeholder for unknown card belief


# --- Memory Decay Categories ---
class DecayCategory(enum.Enum):
    LIKELY_LOW = 100  # ZERO, NEG_KING, ACE, LOW_NUM
    LIKELY_MID = 101  # MID_NUM, PEEK_SELF
    LIKELY_HIGH = 102  # PEEK_OTHER, SWAP_BLIND, HIGH_KING
    UNKNOWN = CardBucket.UNKNOWN  # Re-use UNKNOWN state


class DecisionContext(enum.Enum):
    START_TURN = 0  # Choosing Draw Stockpile / Draw Discard / Call Cambia
    POST_DRAW = 1  # Choosing Discard (Ability/No) / Replace
    SNAP_DECISION = 2  # Choosing Pass Snap / Snap Own / Snap Opponent
    ABILITY_SELECT = 3  # Choosing target/decision for 7/8/9/T/J/Q/K ability
    SNAP_MOVE = 4  # Choosing which card to move after successful Snap Opponent
    TERMINAL = 5  # Although CFR stops before this, useful for completeness


# --- Game Phases / State Estimates ---
class GamePhase(enum.Enum):
    # Simplified phases based on stockpile for now
    START = 0  # Technically pre-first turn, maybe unused in infoset
    EARLY = 1  # Stockpile HIGH
    MID = 2  # Stockpile MEDIUM
    LATE = 3  # Stockpile LOW/EMPTY
    CAMBIA_CALLED = 4  # Cambia is active
    TERMINAL = 5  # Game is over


class StockpileEstimate(enum.Enum):
    HIGH = 0
    MEDIUM = 1
    LOW = 2
    EMPTY = 3


# --- Game Constants ---
INITIAL_HAND_SIZE = 4
NUM_PLAYERS = 2  # Fixed for 1v1

# Avoid circular import with card.py
# Forward declare Card type hint if necessary or import lazily
CardObject = Any  # Use Any temporarily, replace with 'card.Card' if possible

# --- Structured Action Definitions ---
# Using NamedTuple for clarity, hashability, and type checking


class ActionDrawStockpile(NamedTuple):
    pass  # No extra data needed


class ActionDrawDiscard(NamedTuple):
    pass  # No extra data needed (if implemented)


class ActionCallCambia(NamedTuple):
    pass  # No extra data needed


class ActionDiscard(NamedTuple):
    # Card discarded is implicit (the one held from Draw)
    use_ability: bool  # Does the player *intend* to use the ability?


class ActionReplace(NamedTuple):
    # Card used to replace is implicit (the one held from Draw)
    target_hand_index: int


# --- Ability-related Actions (Sub-steps) ---
class ActionAbilityPeekOwnSelect(NamedTuple):
    target_hand_index: int


class ActionAbilityPeekOtherSelect(NamedTuple):
    target_opponent_hand_index: int


class ActionAbilityBlindSwapSelect(NamedTuple):
    own_hand_index: int
    opponent_hand_index: int


class ActionAbilityKingLookSelect(NamedTuple):
    # Choose one card from own hand and one from opponent's hand
    own_hand_index: int
    opponent_hand_index: int


class ActionAbilityKingSwapDecision(NamedTuple):
    # Decision after looking
    perform_swap: bool


# --- Snap Actions ---
# Represents the *attempt* to snap. Engine verifies and applies penalty/success.
class ActionPassSnap(NamedTuple):
    """Action to explicitly pass the opportunity to snap."""


class ActionSnapOwn(NamedTuple):
    # Player attempting snap is implicit (current player in snap phase)
    own_card_hand_index: int  # Which card in hand matches the discard


class ActionSnapOpponent(NamedTuple):
    # Player attempting snap is implicit
    # This requires the snapper to know/guess the opponent's matching card index
    opponent_target_hand_index: (
        int  # Which of *opponent's* cards to snap (must match rank) ### CLARIFIED ###
    )
    # Decision on which card to move comes *after* successful snap
    # own_card_to_move_hand_index: int # Which of *own* cards to move into opponent's slot ### REMOVED - Now a separate step ###


class ActionSnapOpponentMove(NamedTuple):  #
    """Action taken after a successful opponent snap to move own card."""

    own_card_to_move_hand_index: int  # Index of own card to move
    target_empty_slot_index: int  # Index where opponent's card was removed


# Union type for all possible actions
GameAction = Union[
    ActionDrawStockpile,
    ActionDrawDiscard,
    ActionCallCambia,
    ActionDiscard,
    ActionReplace,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionPassSnap,
    ActionSnapOwn,
    ActionSnapOpponent,
    ActionSnapOpponentMove,  # Updated Union
]
