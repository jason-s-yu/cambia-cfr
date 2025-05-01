# src/persistence.py
import joblib
import os
from typing import Tuple, Optional, Dict, Any, TypeAlias
import logging
import numpy as np

from .utils import PolicyDict, InfosetKey

logger = logging.getLogger(__name__)

# Type alias for the reach probability sum dictionary used in CFR+
ReachProbDict: TypeAlias = Dict[InfosetKey, float]


def save_agent_data(data_to_save: Dict[str, Any], filepath: str):
    """Saves the agent's learned data (regrets, strategy sums, reach sums, iteration count)."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(data_to_save, filepath)
        iteration = data_to_save.get("iteration", "N/A")
        logger.info(f"Agent data saved to {filepath} at iteration {iteration}")
    except Exception as e:
        logger.error(f"Error saving agent data to {filepath}: {e}")


def load_agent_data(filepath: str) -> Optional[Dict[str, Any]]:
    """Loads the agent's learned data from a file."""
    try:
        if os.path.exists(filepath):
            loaded_data = joblib.load(filepath)
            iteration = loaded_data.get("iteration", 0)
            # Basic validation: Check if expected keys exist
            if (
                "regret_sum" not in loaded_data
                or "strategy_sum" not in loaded_data
                or "reach_prob_sum" not in loaded_data
            ):
                logger.warning(
                    f"Loaded data from {filepath} missing expected keys. Check file integrity."
                )
            logger.info(
                f"Agent data loaded from {filepath}. Resuming from iteration {iteration}."
            )
            # Return the whole dictionary
            return loaded_data
        else:
            logger.info(f"Agent data file not found at {filepath}. Starting fresh.")
            return None
    except Exception as e:
        logger.error(f"Error loading agent data from {filepath}: {e}")
        return None
