# src/cfr/data_manager_mixin.py
"""Mixin class for managing CFR data persistence and average strategy computation."""

import logging
from collections import defaultdict
from typing import Optional, List, Tuple

import numpy as np

from ..persistence import load_agent_data, save_agent_data
from ..utils import (
    InfosetKey,
    PolicyDict,
    ReachProbDict,
    WorkerResult,  # WorkerResult is a dataclass
    normalize_probabilities,
)

logger = logging.getLogger(__name__)


class CFRDataManagerMixin:
    """Handles loading, saving, and computation of CFR policy data."""

    # Attributes expected to be initialized in the main class's __init__
    config: "Config"  # Forward reference Config
    regret_sum: PolicyDict
    strategy_sum: PolicyDict
    reach_prob_sum: ReachProbDict
    current_iteration: int
    exploitability_results: List[Tuple[int, float]]
    average_strategy: Optional[PolicyDict]
    # Internal display state attributes
    _last_exploit_str: str
    _total_infosets_str: str

    def load_data(self, filepath: Optional[str] = None):
        """Loads trainer state (policy dicts, iteration count) from a file."""
        path = filepath or self.config.persistence.agent_data_save_path
        loaded = load_agent_data(path)
        if loaded:
            # Ensure defaultdict behavior after loading
            self.regret_sum = defaultdict(
                lambda: np.array([], dtype=np.float64),
                {
                    InfosetKey(*k) if isinstance(k, tuple) else k: np.asarray(
                        v, dtype=np.float64
                    )
                    for k, v in loaded.get("regret_sum", {}).items()
                },
            )
            self.strategy_sum = defaultdict(
                lambda: np.array([], dtype=np.float64),
                {
                    InfosetKey(*k) if isinstance(k, tuple) else k: np.asarray(
                        v, dtype=np.float64
                    )
                    for k, v in loaded.get("strategy_sum", {}).items()
                },
            )
            self.reach_prob_sum = defaultdict(
                float,
                {
                    InfosetKey(*k) if isinstance(k, tuple) else k: float(v)
                    for k, v in loaded.get("reach_prob_sum", {}).items()
                },
            )
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
            logger.info(
                "Resuming training. Will start execution from iteration %d.",
                self.current_iteration + 1,
            )
        else:
            logger.info("No saved data found or error loading. Starting fresh.")
            self.regret_sum = defaultdict(lambda: np.array([], dtype=np.float64))
            self.strategy_sum = defaultdict(lambda: np.array([], dtype=np.float64))
            self.reach_prob_sum = defaultdict(float)
            self.current_iteration = 0
            self.exploitability_results = []
            self._last_exploit_str = "N/A"
            self._total_infosets_str = "0"

    def save_data(self, filepath: Optional[str] = None):
        """Saves the current trainer state to a file."""
        path = filepath or self.config.persistence.agent_data_save_path
        data_to_save = {
            "regret_sum": dict(self.regret_sum),
            "strategy_sum": dict(self.strategy_sum),
            "reach_prob_sum": dict(self.reach_prob_sum),
            "iteration": self.current_iteration,
            "exploitability_results": self.exploitability_results,
        }
        save_agent_data(data_to_save, path)

    def _merge_local_updates(self, results: List[Optional[WorkerResult]]):
        """
        Merges the local updates collected from worker processes into the main
        trainer's state dictionaries (regret_sum, strategy_sum, reach_prob_sum).

        Args:
            results: A list where each element is either a WorkerResult object
                     or None if the worker failed.
        """
        if not results:
            logger.warning("Received empty results list for merging.")
            return

        updated_infoset_keys = set()

        for worker_idx, result_obj in enumerate(results):
            if result_obj is None:
                logger.warning("Skipping merge for failed worker %d.", worker_idx)
                continue

            # Access attributes from the WorkerResult object
            local_regret_updates = result_obj.regret_updates
            local_strategy_sum_updates = result_obj.strategy_updates
            local_reach_prob_updates = result_obj.reach_prob_updates
            # worker_stats = result_obj.stats # Available if needed

            for infoset_key, local_regrets in local_regret_updates.items():
                num_actions_local = len(local_regrets)
                if num_actions_local == 0:
                    continue

                updated_infoset_keys.add(infoset_key)
                current_regrets = self.regret_sum.get(infoset_key)

                if (
                    current_regrets is None or len(current_regrets) == 0
                ):  # Also handle if current is empty array
                    self.regret_sum[infoset_key] = local_regrets
                elif len(current_regrets) != num_actions_local:
                    logger.warning(
                        "Merge Warn: Regret dimension mismatch for key %s. Global: %d, Worker %d: %d. Re-initializing global and adding.",
                        infoset_key,
                        len(current_regrets),
                        worker_idx,
                        num_actions_local,
                    )
                    self.regret_sum[infoset_key] = local_regrets
                else:
                    self.regret_sum[infoset_key] += local_regrets

            for infoset_key, local_strategy_sum in local_strategy_sum_updates.items():
                num_actions_local = len(local_strategy_sum)
                if num_actions_local == 0:
                    continue

                updated_infoset_keys.add(infoset_key)
                current_strategy_sum = self.strategy_sum.get(infoset_key)

                if (
                    current_strategy_sum is None or len(current_strategy_sum) == 0
                ):  # Also handle if current is empty array
                    self.strategy_sum[infoset_key] = local_strategy_sum
                elif len(current_strategy_sum) != num_actions_local:
                    logger.warning(
                        "Merge Warn: Strategy Sum dimension mismatch for key %s. Global: %d, Worker %d: %d. Re-initializing global and adding.",
                        infoset_key,
                        len(current_strategy_sum),
                        worker_idx,
                        num_actions_local,
                    )
                    self.strategy_sum[infoset_key] = local_strategy_sum
                else:
                    self.strategy_sum[infoset_key] += local_strategy_sum

            for infoset_key, local_reach_prob in local_reach_prob_updates.items():
                updated_infoset_keys.add(infoset_key)
                self.reach_prob_sum[infoset_key] += local_reach_prob

        for key in updated_infoset_keys:
            if (
                key in self.regret_sum and len(self.regret_sum[key]) > 0
            ):  # Ensure array is not empty before max
                self.regret_sum[key] = np.maximum(0.0, self.regret_sum[key])

        logger.debug(
            "Merge complete. %d unique infosets affected in this merge.",
            len(updated_infoset_keys),
        )

    def compute_average_strategy(self) -> Optional[PolicyDict]:
        """
        Computes the average strategy from the accumulated strategy sums.
        """
        avg_strategy: PolicyDict = {}
        if not self.strategy_sum:
            logger.warning("Cannot compute average strategy: Strategy sum is empty.")
            return avg_strategy  # Return empty dict, not None

        logger.info(
            "Computing average strategy from %d infosets...", len(self.strategy_sum)
        )
        zero_reach_count, nan_count, norm_issue_count, mismatched_dim_count = 0, 0, 0, 0

        for infoset_key_tuple, s_sum in self.strategy_sum.items():
            # Ensure key is InfosetKey instance
            # This should already be handled if workers return InfosetKey objects
            # and load_data correctly reconstructs them.
            if not isinstance(infoset_key_tuple, InfosetKey):
                if isinstance(
                    infoset_key_tuple, tuple
                ):  # Common case from direct dict load
                    try:
                        infoset_key = InfosetKey(*infoset_key_tuple)
                    except Exception:
                        logger.error(
                            "Failed conversion to InfosetKey: %s",
                            infoset_key_tuple,
                            exc_info=True,
                        )
                        mismatched_dim_count += 1  # Count as an issue
                        continue
                else:  # Should not happen if data is consistent
                    logger.error(
                        "Invalid key type in strategy_sum: %s", type(infoset_key_tuple)
                    )
                    mismatched_dim_count += 1
                    continue
            else:
                infoset_key = infoset_key_tuple

            r_sum = self.reach_prob_sum.get(infoset_key, 0.0)
            num_actions_in_sum = len(s_sum)
            normalized_strategy = np.array([])

            if r_sum > 1e-9:
                normalized_strategy = s_sum / r_sum

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
                    if (
                        not np.isclose(
                            np.sum(normalized_strategy_reanorm), 1.0, atol=1e-6
                        )
                        and len(normalized_strategy_reanorm) > 0
                    ):  # Check if reanorm resulted in non-empty
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
            else:
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

            regret_array = self.regret_sum.get(infoset_key)
            if regret_array is not None and len(regret_array) != len(normalized_strategy):
                mismatched_dim_count += 1
                logger.warning(
                    "Final avg strategy dim (%d) mismatch with regret (%d) for %s. Defaulting avg strategy to uniform based on *regret* dim.",
                    len(normalized_strategy),
                    len(regret_array),
                    infoset_key,
                )
                normalized_strategy = (
                    np.ones(len(regret_array)) / len(regret_array)
                    if len(regret_array) > 0
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
                "%d infosets with final dimension mismatch (avg vs regret or key error).",
                mismatched_dim_count,
            )
        return avg_strategy

    def get_average_strategy(self) -> Optional[PolicyDict]:
        """Returns the computed average strategy, computing it if necessary."""
        if self.average_strategy is None:
            logger.info(
                "Average strategy requested but not computed yet. Computing now..."
            )
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
