# src/utils.py
import numpy as np
from typing import TypeAlias, Dict, Tuple, NamedTuple, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .constants import CardBucket, StockpileEstimate, GamePhase, DecisionContext

# Type alias for Regret/Strategy Dictionaries
# Using the new dataclass for the key type hint
@dataclass(frozen=True)
class InfosetKey:
    """Represents the key for accessing policy/regret data. Must be hashable."""
    own_hand_tuple: Tuple[int, ...] # Tuple of CardBucket.value
    opp_belief_tuple: Tuple[int, ...] # Tuple of CardBucket/DecayCategory.value
    opp_card_count: int
    discard_top_bucket_value: int # CardBucket.value
    stockpile_size_cat_value: int # StockpileEstimate.value
    game_phase_value: int # GamePhase.value
    decision_context_value: int # DecisionContext.value


PolicyDict: TypeAlias = Dict[InfosetKey, np.ndarray]
ReachProbDict: TypeAlias = Dict[InfosetKey, float] # Reach prob sum dictionary

def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    """Normalizes a numpy array to sum to 1, handling the zero-sum case."""
    if probs is None:
        raise ValueError("Input array cannot be None")

    prob_sum = np.sum(probs)
    if prob_sum > 1e-9: # Use tolerance for floating point
        return probs / prob_sum
    else:
        # If all probabilities are zero (or negative somehow), return uniform distribution
        num_actions = len(probs)
        if num_actions > 0:
            # logger.debug(f"Normalizing zero/negative probabilities {probs} to uniform.")
            return np.ones(num_actions) / num_actions
        else:
            return np.array([]) # No actions possible

def get_rm_plus_strategy(regret_sum: np.ndarray) -> np.ndarray:
    """Calculates the current strategy profile based on Regret Matching+."""
    # RM+ uses only positive regrets
    positive_regrets = np.maximum(0.0, regret_sum)
    return normalize_probabilities(positive_regrets)