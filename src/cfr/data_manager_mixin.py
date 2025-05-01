# src/cfr/data_manager_mixin.py
"""Mixin class for managing CFR data persistence and average strategy computation."""

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from ..constants import DecisionContext
from ..persistence import load_agent_data, save_agent_data
from ..utils import InfosetKey, PolicyDict, normalize_probabilities

logger = logging.getLogger(__name__)

class CFRDataManagerMixin:
    """Handles loading, saving, and computation of CFR policy data."""

    # Attributes expected to be initialized in the main class's __init__
    # self.config: Config
    # self.regret_sum: PolicyDict
    # self.strategy_sum: PolicyDict
    # self.reach_prob_sum: ReachProbDict
    # self.current_iteration: int
    # self.exploitability_results: List[Tuple[int, float]]
    # self.average_strategy: Optional[PolicyDict]
    # self._last_exploit_str: str
    # self._total_infosets_str: str

    def load_data(self, filepath: Optional[str] = None):
        """Loads trainer state (policy dicts, iteration count) from a file."""
        path = filepath or self.config.persistence.agent_data_save_path
        loaded = load_agent_data(path)
        if loaded:
            # Convert tuple keys back to InfosetKey if they exist from older saves
            self.regret_sum = defaultdict(
                lambda: np.array([], dtype=np.float64),
                {
                    InfosetKey(*k) if isinstance(k, tuple) else k: v
                    for k, v in loaded.get("regret_sum", {}).items()
                },
            )
            self.strategy_sum = defaultdict(
                lambda: np.array([], dtype=np.float64),
                {
                    InfosetKey(*k) if isinstance(k, tuple) else k: v
                    for k, v in loaded.get("strategy_sum", {}).items()
                },
            )
            self.reach_prob_sum = defaultdict(
                float,
                {
                    InfosetKey(*k) if isinstance(k, tuple) else k: v
                    for k, v in loaded.get("reach_prob_sum", {}).items()
                },
            )
            # self.current_iteration now stores the last *completed* iteration
            self.current_iteration = loaded.get("iteration", 0)
            exploit_history = loaded.get("exploitability_results", [])
            if isinstance(exploit_history, list) and all(
                isinstance(item, (tuple, list)) and len(item) == 2
                for item in exploit_history
            ):
                self.exploitability_results = [
                    (int(it), float(expl)) for it, expl in exploit_history
                ]
                if self.exploitability_results:
                    last_exploit_val = self.exploitability_results[-1][1]
                    self._last_exploit_str = (
                        f"{last_exploit_val:.3f}"
                        if last_exploit_val != float("inf")
                        else "N/A"
                    )
            else:
                logger.warning(
                    "Invalid exploitability history format found in loaded data. Resetting."
                )
                self.exploitability_results = []
                self._last_exploit_str = "N/A"
            self._total_infosets_str = f"{len(self.regret_sum):,}"
            # Adjust log message: We will re-run the iteration *after* the last completed one
            logger.info(
                "Resuming training. Will start execution from iteration %d.",
                self.current_iteration + 1,
            )
        else:
            logger.info("No saved data found or error loading. Starting fresh.")
            self.regret_sum = defaultdict(lambda: np.array([], dtype=np.float64))
            self.strategy_sum = defaultdict(lambda: np.array([], dtype=np.float64))
            self.reach_prob_sum = defaultdict(float)
            self.current_iteration = 0  # Start fresh means 0 completed iterations
            self.exploitability_results = []
            self._last_exploit_str = "N/A"
            self._total_infosets_str = "0"

    def save_data(self, filepath: Optional[str] = None):
        """Saves the current trainer state to a file."""
        path = filepath or self.config.persistence.agent_data_save_path
        # Save the state corresponding to the *completion* of self.current_iteration
        data_to_save = {
            "regret_sum": dict(self.regret_sum),
            "strategy_sum": dict(self.strategy_sum),
            "reach_prob_sum": dict(self.reach_prob_sum),
            "iteration": self.current_iteration,  # Save the number of the iteration just finished
            "exploitability_results": self.exploitability_results,
        }
        save_agent_data(data_to_save, path)

    def compute_average_strategy(self) -> PolicyDict:
        """
        Computes the average strategy from the accumulated strategy sums.

        The average strategy at infoset I is given by:
            avg_sigma(I, a) = strategy_sum(I, a) / reach_prob_sum(I)
        Where:
            strategy_sum(I, a) = sum_{t=1 to T} [ weight(t) * pi_i(I) * sigma_t(I, a) ]
            reach_prob_sum(I) = sum_{t=1 to T} [ weight(t) * pi_i(I) ]
            weight(t) = max(0, t - delay) if weighted, else 1.0
            pi_i(I) is reach probability of player i at infoset I.
            sigma_t(I, a) is the strategy at iteration t.
        """
        avg_strategy: PolicyDict = {}
        logger.info(
            "Computing average strategy from %d infosets...", len(self.strategy_sum)
        )
        if not self.strategy_sum:
            logger.warning("Strategy sum is empty.")
            return avg_strategy

        zero_reach_count, nan_count, norm_issue_count, mismatched_dim_count = 0, 0, 0, 0
        for infoset_key, s_sum in self.strategy_sum.items():
            # Ensure key is InfosetKey instance
            if isinstance(infoset_key, tuple):
                try:
                    infoset_key = InfosetKey(*infoset_key)
                except TypeError:
                    logger.error(
                        "Failed to convert tuple %s to InfosetKey",
                        infoset_key,
                        exc_info=True,
                    )
                    continue  # Skip this invalid key

            r_sum = self.reach_prob_sum.get(infoset_key, 0.0)
            num_actions_in_sum = len(s_sum)
            normalized_strategy = np.array([])

            if r_sum > 1e-9:
                normalized_strategy = s_sum / r_sum

                # --- Sanity Checks ---
                if np.isnan(normalized_strategy).any():
                    nan_count += 1
                    logger.warning(
                        "NaN found in avg strategy for %s. Sum: %s, Reach: %s. Using uniform.",
                        infoset_key,
                        s_sum,
                        r_sum,
                    )
                    normalized_strategy = (
                        np.ones(num_actions_in_sum) / num_actions_in_sum
                        if num_actions_in_sum > 0
                        else np.array([])
                    )
                current_sum = np.sum(normalized_strategy)
                if (
                    not np.isclose(current_sum, 1.0, atol=1e-6)
                    and len(normalized_strategy) > 0
                ):
                    normalized_strategy_reanorm = normalize_probabilities(
                        normalized_strategy
                    )
                    if not np.isclose(
                        np.sum(normalized_strategy_reanorm), 1.0, atol=1e-6
                    ):
                        norm_issue_count += 1
                        logger.warning(
                            "Avg strategy re-norm failed for %s (Sum: %s -> %s). Using uniform.",
                            infoset_key,
                            current_sum,
                            np.sum(normalized_strategy_reanorm),
                        )
                        normalized_strategy = (
                            np.ones(num_actions_in_sum) / num_actions_in_sum
                            if num_actions_in_sum > 0
                            else np.array([])
                        )
                    else:
                        normalized_strategy = normalized_strategy_reanorm
            else:  # Zero reach sum
                if np.any(s_sum != 0):
                    zero_reach_count += 1
                    logger.warning(
                        "Infoset %s has zero reach sum but non-zero strategy sum %s. Using uniform.",
                        infoset_key,
                        s_sum,
                    )
                normalized_strategy = (
                    np.ones(num_actions_in_sum) / num_actions_in_sum
                    if num_actions_in_sum > 0
                    else np.array([])
                )

            # --- Dimension Check against Regrets ---
            regret_array = self.regret_sum.get(infoset_key)
            if regret_array is not None and len(regret_array) != len(normalized_strategy):
                mismatched_dim_count += 1
                num_actions_regret = len(regret_array)
                context_value = getattr(infoset_key, "decision_context_value", None)
                context_name = (
                    DecisionContext(context_value).name
                    if context_value is not None
                    else "N/A"
                )
                logger.warning(
                    "Final avg strategy dim (%d) mismatch with regret (%d) for %s. Context: %s. Defaulting avg strategy to uniform based on *regret* dim.",
                    len(normalized_strategy),
                    num_actions_regret,
                    infoset_key,
                    context_name,
                )
                normalized_strategy = (
                    np.ones(num_actions_regret) / num_actions_regret
                    if num_actions_regret > 0
                    else np.array([])
                )
            elif regret_array is None and len(normalized_strategy) > 0:
                logger.warning(
                    "Infoset %s has strategy sum but no regret sum entry. Using calculated strategy.",
                    infoset_key,
                )

            avg_strategy[infoset_key] = normalized_strategy

        self.average_strategy = avg_strategy
        logger.info(
            "Average strategy computation complete (%d infosets).",
            len(self.average_strategy),
        )
        if zero_reach_count > 0:
            logger.warning(
                "%d infosets with zero reach sum but non-zero strategy sum.",
                zero_reach_count,
            )
        if nan_count > 0:
            logger.warning("%d infosets with NaN strategy.", nan_count)
        if norm_issue_count > 0:
            logger.warning("%d infosets with norm issues.", norm_issue_count)
        if mismatched_dim_count > 0:
            logger.warning(
                "%d infosets with final dimension mismatch (avg vs regret).",
                mismatched_dim_count,
            )
        return avg_strategy

    def get_average_strategy(self) -> Optional[PolicyDict]:
        """Returns the computed average strategy, computing it if necessary."""
        if self.average_strategy is None:
            logger.warning(
                "Average strategy requested but not computed yet. Computing now..."
            )
            # Need to ensure self has compute_average_strategy method
            if hasattr(self, "compute_average_strategy") and callable(
                self.compute_average_strategy
            ):
                return self.compute_average_strategy()
            else:
                logger.error(
                    "Cannot compute average strategy - method not found on self."
                )
                return None
        return self.average_strategy
