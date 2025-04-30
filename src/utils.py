import numpy as np
from typing import TypeAlias, Dict, Tuple

# Type alias for Infoset Key (must be hashable)
# Structure: (own_hand_tuple, opp_belief_tuple, opp_card_count,
#             discard_top_bucket_enum, stockpile_size_cat_enum, game_phase_enum,
#             # Potentially add minimal history features here if needed
#            )
InfosetKey: TypeAlias = Tuple

# Type alias for Regret/Strategy Dictionaries
PolicyDict: TypeAlias = Dict[InfosetKey, np.ndarray]

def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    """Normalizes a numpy array to sum to 1, handling the zero-sum case."""
    if probs is None:
        raise ValueError("Input array cannot be None")
    
    prob_sum = np.sum(probs)
    if prob_sum > 0:
        return probs / prob_sum
    else:
        # If all probabilities are zero (or negative somehow), return uniform distribution
        num_actions = len(probs)
        if num_actions > 0:
            return np.ones(num_actions) / num_actions
        else:
            return np.array([]) # No actions possible

def get_rm_plus_strategy(regret_sum: np.ndarray) -> np.ndarray:
    """Calculates the current strategy profile based on Regret Matching+."""
    # RM+ uses only positive regrets
    positive_regrets = np.maximum(0.0, regret_sum)
    return normalize_probabilities(positive_regrets)