"""src/utils.py"""

import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, Tuple, TypeAlias, List, Any, Optional, TypedDict

import numpy as np


# Type alias for Regret/Strategy Dictionaries
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
        """Converts InfosetKey to a plain tuple."""
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


@dataclass
class WorkerStats:
    """Statistics reported by a worker for a single simulation."""

    max_depth: int = 0
    nodes_visited: int = 0
    warning_count: int = 0
    error_count: int = 0
    min_depth_after_bottom_out: float = float("inf")


# Type aliases for worker results (local updates)
LocalRegretUpdateDict: TypeAlias = Dict[InfosetKey, np.ndarray]
LocalStrategyUpdateDict: TypeAlias = Dict[InfosetKey, np.ndarray]
LocalReachProbUpdateDict: TypeAlias = Dict[InfosetKey, float]


# --- Simulation Trace Data Structures ---
class SimulationNodeData(TypedDict):
    """Data captured at each decision node during a worker simulation."""

    depth: int
    player: int
    infoset_key: Tuple[Any, ...]  # Serialized InfosetKey (using astuple representation)
    context: str  # DecisionContext name
    strategy: List[float]  # Strategy used at this node (list of probabilities)
    chosen_action: Any  # Serialized chosen action
    state_delta: List[Tuple[str, ...]]  # List of StateDeltaChange tuples from engine


class SimulationTrace(TypedDict):
    """Structure for logging a complete simulation trace."""

    metadata: Dict[str, Any]  # Iteration, worker_id, final_utility, etc.
    history: List[SimulationNodeData]


# Type alias for the combined results returned by a single worker
@dataclass
class WorkerResult:
    """Combined results and stats from a single worker simulation."""

    regret_updates: LocalRegretUpdateDict = field(default_factory=dict)
    strategy_updates: LocalStrategyUpdateDict = field(default_factory=dict)
    reach_prob_updates: LocalReachProbUpdateDict = field(default_factory=dict)
    stats: WorkerStats = field(default_factory=WorkerStats)
    simulation_nodes: List[SimulationNodeData] = field(default_factory=list)
    final_utility: Optional[List[float]] = None


# Type alias for the logging queue
LogQueue: TypeAlias = multiprocessing.Queue


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


def format_large_number(num: int | float) -> str:
    """Formats a large number with metric prefixes (k, M, B, T)."""
    if not isinstance(num, (int, float)):
        return "N/A"

    num_float = float(num)

    if abs(num_float) < 1_000:
        return str(int(num_float))
    if abs(num_float) < 1_000_000:
        return f"{num_float / 1_000:.0f}k"
    if abs(num_float) < 1_000_000_000:
        return f"{num_float / 1_000_000:.2f}M"
    if abs(num_float) < 1_000_000_000_000:
        return f"{num_float / 1_000_000_000:.2f}B"
    return f"{num_float / 1_000_000_000_000:.2f}T"


def format_infoset_count(count: int) -> str:
    """Formats a large number with k/M suffixes (no decimals)."""
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count // 1000}k"
    else:
        return f"{count / 1_000_000:.1f}M"
