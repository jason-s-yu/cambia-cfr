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
    WorkerResult,
    normalize_probabilities,
    format_infoset_count,
)
from ..config import Config

logger = logging.getLogger(__name__)


class CFRDataManagerMixin:
    """Handles loading, saving, merging, and computation of CFR policy data."""

    # Attributes expected from main class
    config: Config
    regret_sum: PolicyDict
    strategy_sum: PolicyDict
    reach_prob_sum: ReachProbDict
    current_iteration: int
    exploitability_results: List[Tuple[int, float]]
    average_strategy: Optional[PolicyDict]
    _last_exploit_str: str
    _total_infosets_str: str

    def load_data(self, filepath: Optional[str] = None):
        """Loads trainer state from a file."""
        path = filepath or self.config.persistence.agent_data_save_path
        loaded = load_agent_data(path)
        if loaded:
            try:
                # Convert keys back to InfosetKey during loading
                self.regret_sum = defaultdict(
                    lambda: np.array([], dtype=np.float64),
                    {
                        InfosetKey(*k) if isinstance(k, tuple) else k: np.asarray(
                            v, dtype=np.float64
                        )
                        for k, v in loaded.get("regret_sum", {}).items()
                        if isinstance(k, (InfosetKey, tuple))
                    },
                )
                self.strategy_sum = defaultdict(
                    lambda: np.array([], dtype=np.float64),
                    {
                        InfosetKey(*k) if isinstance(k, tuple) else k: np.asarray(
                            v, dtype=np.float64
                        )
                        for k, v in loaded.get("strategy_sum", {}).items()
                        if isinstance(k, (InfosetKey, tuple))
                    },
                )
                self.reach_prob_sum = defaultdict(
                    float,
                    {
                        InfosetKey(*k) if isinstance(k, tuple) else k: float(v)
                        for k, v in loaded.get("reach_prob_sum", {}).items()
                        if isinstance(k, (InfosetKey, tuple))
                    },
                )
                # Log if any keys were skipped due to invalid type
                for key_type in ["regret_sum", "strategy_sum", "reach_prob_sum"]:
                    original_keys = loaded.get(key_type, {}).keys()
                    loaded_keys = getattr(self, key_type).keys()
                    skipped_count = len(original_keys) - len(loaded_keys)
                    if skipped_count > 0:
                        logger.warning(
                            "Skipped %d invalid keys during loading of %s.",
                            skipped_count,
                            key_type,
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
                        self._last_exploit_str = "N/A"
                else:
                    logger.warning(
                        "Invalid exploitability history format found in loaded data. Resetting."
                    )
                    self.exploitability_results = []
                    self._last_exploit_str = "N/A"

                self._total_infosets_str = format_infoset_count(
                    len(self.regret_sum)
                )  # Use helper
                logger.info(
                    "Resuming training. Will start execution from iteration %d.",
                    self.current_iteration + 1,
                )

            except (TypeError, ValueError) as e_load:
                logger.exception(
                    "Error processing loaded data from %s: %s. Starting fresh.",
                    path,
                    e_load,
                )
                self._reset_data()  # Reset to default state on processing error
        else:
            logger.info("No valid saved data found at %s. Starting fresh.", path)
            self._reset_data()

    def _reset_data(self):
        """Resets data structures to their initial empty state."""
        self.regret_sum = defaultdict(lambda: np.array([], dtype=np.float64))
        self.strategy_sum = defaultdict(lambda: np.array([], dtype=np.float64))
        self.reach_prob_sum = defaultdict(float)
        self.current_iteration = 0
        self.exploitability_results = []
        self.average_strategy = None
        self._last_exploit_str = "N/A"
        self._total_infosets_str = "0"
        logger.info("CFR data structures reset to initial state.")

    def save_data(self, filepath: Optional[str] = None):
        """Saves the current trainer state to a file."""
        path = filepath or self.config.persistence.agent_data_save_path
        try:
            # Convert InfosetKey back to tuple for saving if needed by joblib/pickle
            # (joblib might handle dataclasses fine, but tuple is safer)
            data_to_save = {
                "regret_sum": {
                    k.astuple() if isinstance(k, InfosetKey) else k: v
                    for k, v in self.regret_sum.items()
                },
                "strategy_sum": {
                    k.astuple() if isinstance(k, InfosetKey) else k: v
                    for k, v in self.strategy_sum.items()
                },
                "reach_prob_sum": {
                    k.astuple() if isinstance(k, InfosetKey) else k: v
                    for k, v in self.reach_prob_sum.items()
                },
                "iteration": self.current_iteration,
                "exploitability_results": self.exploitability_results,
            }
            save_agent_data(data_to_save, path)
        except Exception as e_save:  # JUSTIFIED: Broad catch to ensure all save errors are logged; persistence module handles specifics
            logger.exception("Error saving agent data to %s: %s", path, e_save)

    def _merge_local_updates(self, results: List[Optional[WorkerResult]]):
        """Merges local updates from workers into the main trainer's state."""
        if not results:
            logger.warning("Received empty results list for merging.")
            return

        updated_infoset_keys = set()
        merge_warnings = 0
        merge_errors = 0

        for worker_idx, result_obj in enumerate(results):
            if result_obj is None or not isinstance(result_obj, WorkerResult):
                logger.warning(
                    "Skipping merge for invalid/failed worker %d result.", worker_idx
                )
                continue

            local_regret_updates = result_obj.regret_updates
            local_strategy_sum_updates = result_obj.strategy_updates
            local_reach_prob_updates = result_obj.reach_prob_updates

            # Merge Regrets
            for infoset_key, local_regrets in local_regret_updates.items():
                if not isinstance(infoset_key, InfosetKey):
                    logger.error(
                        "Merge Error: Invalid key type '%s' in regret updates from worker %d. Skipping key.",
                        type(infoset_key).__name__,
                        worker_idx,
                    )
                    merge_errors += 1
                    continue
                if not isinstance(local_regrets, np.ndarray):
                    logger.error(
                        "Merge Error: Invalid regret value type '%s' for key %s from worker %d. Skipping key.",
                        type(local_regrets).__name__,
                        infoset_key,
                        worker_idx,
                    )
                    merge_errors += 1
                    continue

                num_actions_local = len(local_regrets)
                if num_actions_local == 0:
                    continue  # Skip empty updates

                updated_infoset_keys.add(infoset_key)
                current_regrets = self.regret_sum.get(infoset_key)

                if current_regrets is None or len(current_regrets) == 0:
                    # First time seeing this infoset or it was empty, initialize directly
                    self.regret_sum[infoset_key] = local_regrets.copy()  # Use copy
                elif len(current_regrets) != num_actions_local:
                    logger.warning(
                        "Merge Warn Regret: Dim mismatch key %s. Global:%d, Worker %d:%d. Re-initializing global + adding.",
                        infoset_key,
                        len(current_regrets),
                        worker_idx,
                        num_actions_local,
                    )
                    merge_warnings += 1
                    self.regret_sum[infoset_key] = (
                        local_regrets.copy()
                    )  # Overwrite with worker's value as base
                else:
                    self.regret_sum[infoset_key] += local_regrets

            # Merge Strategy Sums
            for infoset_key, local_strategy_sum in local_strategy_sum_updates.items():
                if not isinstance(infoset_key, InfosetKey):
                    logger.error(
                        "Merge Error: Invalid key type '%s' in strategy updates from worker %d. Skipping key.",
                        type(infoset_key).__name__,
                        worker_idx,
                    )
                    merge_errors += 1
                    continue
                if not isinstance(local_strategy_sum, np.ndarray):
                    logger.error(
                        "Merge Error: Invalid strategy value type '%s' for key %s from worker %d. Skipping key.",
                        type(local_strategy_sum).__name__,
                        infoset_key,
                        worker_idx,
                    )
                    merge_errors += 1
                    continue

                num_actions_local = len(local_strategy_sum)
                if num_actions_local == 0:
                    continue

                updated_infoset_keys.add(infoset_key)
                current_strategy_sum = self.strategy_sum.get(infoset_key)

                if current_strategy_sum is None or len(current_strategy_sum) == 0:
                    self.strategy_sum[infoset_key] = local_strategy_sum.copy()
                elif len(current_strategy_sum) != num_actions_local:
                    logger.warning(
                        "Merge Warn Strategy: Dim mismatch key %s. Global:%d, Worker %d:%d. Re-initializing global + adding.",
                        infoset_key,
                        len(current_strategy_sum),
                        worker_idx,
                        num_actions_local,
                    )
                    merge_warnings += 1
                    self.strategy_sum[infoset_key] = local_strategy_sum.copy()
                else:
                    self.strategy_sum[infoset_key] += local_strategy_sum

            # Merge Reach Probs
            for infoset_key, local_reach_prob in local_reach_prob_updates.items():
                if not isinstance(infoset_key, InfosetKey):
                    logger.error(
                        "Merge Error: Invalid key type '%s' in reach prob updates from worker %d. Skipping key.",
                        type(infoset_key).__name__,
                        worker_idx,
                    )
                    merge_errors += 1
                    continue
                if not isinstance(local_reach_prob, (float, int)):  # Allow int promotion
                    logger.error(
                        "Merge Error: Invalid reach prob value type '%s' for key %s from worker %d. Skipping key.",
                        type(local_reach_prob).__name__,
                        infoset_key,
                        worker_idx,
                    )
                    merge_errors += 1
                    continue

                updated_infoset_keys.add(infoset_key)
                self.reach_prob_sum[infoset_key] += float(local_reach_prob)

        # Apply RM+ (ensure non-negative regrets)
        for key in updated_infoset_keys:
            if key in self.regret_sum and len(self.regret_sum[key]) > 0:
                self.regret_sum[key] = np.maximum(0.0, self.regret_sum[key])

        logger.debug(
            "Merge complete for iter %d. Affected %d infosets. Warnings: %d, Errors: %d.",
            self.current_iteration,
            len(updated_infoset_keys),
            merge_warnings,
            merge_errors,
        )

    def compute_average_strategy(self) -> Optional[PolicyDict]:
        """Computes the average strategy from the accumulated strategy sums."""
        avg_strategy: PolicyDict = {}
        if not self.strategy_sum:
            logger.warning("Cannot compute average strategy: Strategy sum is empty.")
            return avg_strategy

        logger.info(
            "Computing average strategy from %d infosets...", len(self.strategy_sum)
        )
        (
            zero_reach_count,
            nan_count,
            norm_issue_count,
            key_err_count,
            dim_mismatch_count,
        ) = (0, 0, 0, 0, 0)

        for infoset_key_maybe_tuple, s_sum in self.strategy_sum.items():
            # Ensure key is InfosetKey instance
            if isinstance(infoset_key_maybe_tuple, InfosetKey):
                infoset_key = infoset_key_maybe_tuple
            elif isinstance(infoset_key_maybe_tuple, tuple):
                try:
                    infoset_key = InfosetKey(*infoset_key_maybe_tuple)
                except (TypeError, ValueError):
                    logger.error(
                        "Avg Strat: Invalid tuple key %s. Skipping.",
                        infoset_key_maybe_tuple,
                    )
                    key_err_count += 1
                    continue
            else:
                logger.error(
                    "Avg Strat: Invalid key type %s. Skipping.",
                    type(infoset_key_maybe_tuple).__name__,
                )
                key_err_count += 1
                continue

            r_sum = self.reach_prob_sum.get(infoset_key, 0.0)
            num_actions_strat = len(s_sum)
            normalized_strategy = np.array([])  # Initialize as empty

            if num_actions_strat == 0:  # Skip if strategy sum is empty array
                continue

            if r_sum > 1e-9:
                try:
                    normalized_strategy = s_sum / r_sum
                    if np.isnan(normalized_strategy).any():
                        nan_count += 1
                        logger.warning(
                            "Avg Strat: NaN found for %s. Sum: %s, Reach: %s. Using uniform.",
                            infoset_key,
                            s_sum,
                            r_sum,
                        )
                        normalized_strategy = (
                            np.ones(num_actions_strat) / num_actions_strat
                        )
                except (
                    FloatingPointError,
                    ZeroDivisionError,
                ):  # Catch potential division issues explicitly
                    logger.warning(
                        "Avg Strat: Division error for %s. Sum: %s, Reach: %s. Using uniform.",
                        infoset_key,
                        s_sum,
                        r_sum,
                    )
                    normalized_strategy = np.ones(num_actions_strat) / num_actions_strat
            else:  # Zero reach probability sum
                if np.any(s_sum != 0):
                    zero_reach_count += 1
                    logger.warning(
                        "Avg Strat: Zero reach sum but non-zero strategy sum %s for %s. Using uniform.",
                        s_sum,
                        infoset_key,
                    )
                normalized_strategy = np.ones(num_actions_strat) / num_actions_strat

            # Final check for normalization and dimension consistency with regrets
            if len(normalized_strategy) > 0:
                current_sum = np.sum(normalized_strategy)
                if not np.isclose(current_sum, 1.0, atol=1e-6):
                    logger.warning(
                        "Avg Strat: Strategy for %s doesn't sum to 1 (Sum: %s). Re-normalizing.",
                        infoset_key,
                        current_sum,
                    )
                    normalized_strategy_reanorm = normalize_probabilities(
                        normalized_strategy
                    )
                    # Check if re-normalization worked
                    if len(normalized_strategy_reanorm) == 0 or not np.isclose(
                        np.sum(normalized_strategy_reanorm), 1.0, atol=1e-6
                    ):
                        norm_issue_count += 1
                        logger.error(
                            "Avg Strat: Re-normalization failed for %s. Sum: %s -> %s. Using uniform.",
                            infoset_key,
                            current_sum,
                            np.sum(normalized_strategy_reanorm),
                        )
                        normalized_strategy = (
                            np.ones(num_actions_strat) / num_actions_strat
                        )
                    else:
                        normalized_strategy = normalized_strategy_reanorm

                regret_array = self.regret_sum.get(infoset_key)
                if regret_array is not None and len(regret_array) != len(
                    normalized_strategy
                ):
                    dim_mismatch_count += 1
                    logger.warning(
                        "Avg Strat: Final dim mismatch: Avg[%d] vs Regret[%d] for %s. Using uniform based on regret dim.",
                        len(normalized_strategy),
                        len(regret_array),
                        infoset_key,
                    )
                    normalized_strategy = (
                        np.ones(len(regret_array)) / len(regret_array)
                        if len(regret_array) > 0
                        else np.array([])
                    )
                elif regret_array is None:
                    # This case is possible if an infoset was reached but never by the player whose regrets are stored (e.g., only opponent actions)
                    logger.debug(
                        "Avg Strat: Infoset %s has strategy sum but no regret sum entry. Using calculated strategy.",
                        infoset_key,
                    )

            # Only add if strategy is valid (non-empty)
            if len(normalized_strategy) > 0:
                avg_strategy[infoset_key] = normalized_strategy

        self.average_strategy = avg_strategy
        total_infosets_computed = len(self.average_strategy)
        logger.info(
            "Average strategy computed (%d infosets). Issues: ZeroReach=%d, NaN=%d, Norm=%d, KeyErr=%d, DimMismatch=%d",
            total_infosets_computed,
            zero_reach_count,
            nan_count,
            norm_issue_count,
            key_err_count,
            dim_mismatch_count,
        )
        return avg_strategy

    def get_average_strategy(self) -> Optional[PolicyDict]:
        """Returns the computed average strategy, computing it if necessary."""
        # Compute if not already available or if explicitly requested? For now, compute only if None.
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
