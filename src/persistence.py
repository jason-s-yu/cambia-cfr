"""src/persistence.py"""

from typing import Optional, Dict, Any, TypeAlias
import os
import logging
import pickle

import joblib

from .utils import InfosetKey

logger = logging.getLogger(__name__)

# Type alias for the reach probability sum dictionary used in CFR+
ReachProbDict: TypeAlias = Dict[InfosetKey, float]


def save_agent_data(data_to_save: Dict[str, Any], filepath: str):
    """Saves the agent's learned data to a file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(data_to_save, filepath)
        iteration = data_to_save.get("iteration", "N/A")
        logger.info("Agent data saved to %s at iteration %s", filepath, iteration)
    except (OSError, pickle.PicklingError) as e:
        logger.error("Error saving agent data to %s: %s", filepath, e)


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
                    "Loaded data from %s missing expected keys. Check file integrity.",
                    filepath,
                )
            logger.info(
                "Agent data loaded from %s. Resuming from iteration %d.",
                filepath,
                iteration,
            )
            # Return the whole dictionary
            return loaded_data

        logger.info("Agent data file not found at %s. Starting fresh.", filepath)
        return None
    except (OSError, pickle.UnpicklingError, EOFError) as e:
        logger.error("Error loading agent data from %s: %s", filepath, e)
        return None
