# src/utils.py
"""Utility functions and type aliases for CFR."""

from typing import TypeAlias, Dict, Tuple
from dataclasses import dataclass
import numpy as np


# Type alias for Regret/Strategy Dictionaries
# Using the new dataclass for the key type hint
@dataclass(frozen=True)
class InfosetKey:
    """Represents the key for accessing policy/regret data. Must be hashable."""

    own_hand_tuple: Tuple[int, ...]  # Tuple of CardBucket.value
    opp_belief_tuple: Tuple[int, ...]  # Tuple of CardBucket/DecayCategory.value
    opp_card_count: int
    discard_top_bucket_value: int  # CardBucket.value
    stockpile_size_cat_value: int  # StockpileEstimate.value
    game_phase_value: int  # GamePhase.value
    decision_context_value: int  # DecisionContext.value

    # Helper method for easy conversion to tuple, useful for legacy or simple dict keys
    def astuple(self) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int, int, int, int]:
        return (
            self.own_hand_tuple,
            self.opp_belief_tuple,
            self.opp_card_count,
            self.discard_top_bucket_value,
            self.stockpile_size_cat_value,
            self.game_phase_value,
            self.decision_context_value,
        )


PolicyDict: TypeAlias = Dict[InfosetKey, np.ndarray]
ReachProbDict: TypeAlias = Dict[InfosetKey, float]  # Reach prob sum dictionary

# Type aliases for worker results (local updates)
LocalRegretUpdateDict: TypeAlias = Dict[
    InfosetKey, np.ndarray
]  # Accumulates opponent_reach * instantaneous_regret
LocalStrategyUpdateDict: TypeAlias = Dict[
    InfosetKey, np.ndarray
]  # Accumulates weight * player_reach * strategy
LocalReachProbUpdateDict: TypeAlias = Dict[
    InfosetKey, float
]  # Accumulates weight * player_reach

# Type alias for the combined results returned by a single worker
WorkerResult: TypeAlias = Tuple[
    LocalRegretUpdateDict, LocalStrategyUpdateDict, LocalReachProbUpdateDict
]


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    """Normalizes a numpy array to sum to 1, handling the zero-sum case."""
    if probs is None:
        raise ValueError("Input array cannot be None")

    prob_sum = np.sum(probs)
    if prob_sum > 1e-9:  # Use tolerance for floating point
        return probs / prob_sum

    # If all probabilities are zero (or negative somehow), return uniform distribution
    num_actions = len(probs)
    if num_actions > 0:
        # logger.debug("Normalizing zero/negative probabilities %s to uniform.", probs)
        return np.ones(num_actions) / num_actions

    return np.array([])  # No actions possible


def get_rm_plus_strategy(regret_sum: np.ndarray) -> np.ndarray:
    """Calculates the current strategy profile based on Regret Matching+."""
    # RM+ uses only positive regrets
    positive_regrets = np.maximum(0.0, regret_sum)
    return normalize_probabilities(positive_regrets)
