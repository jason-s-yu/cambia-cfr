# src/persistence.py
from typing import Optional, Dict, Any, TypeAlias
import os
import logging
import pickle

import joblib

# Assuming InfosetKey is properly imported elsewhere or defined if needed directly
# from .utils import InfosetKey

logger = logging.getLogger(__name__)

# Placeholder if InfosetKey isn't imported directly but used in type hints
# InfosetKey = Any

# Type alias for the reach probability sum dictionary used in CFR+
ReachProbDict: TypeAlias = Dict[Any, float]  # Use Any if InfosetKey not imported


def save_agent_data(data_to_save: Dict[str, Any], filepath: str):
    """Saves the agent's learned data to a file."""
    if not filepath or not isinstance(filepath, str):
        logger.error(
            "Cannot save agent data: Invalid filepath provided (received: %s).",
            filepath,
        )
        return

    try:
        # Ensure parent directory exists
        parent_dir = os.path.dirname(filepath)
        # Handle case where filepath is just a filename (dirname is '')
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Save the data
        joblib.dump(data_to_save, filepath)

        iteration = data_to_save.get("iteration", "N/A")
        logger.info("Agent data saved to %s at iteration %s", filepath, iteration)

    except (OSError, pickle.PicklingError, TypeError) as e:
        logger.error("Error saving agent data to %s: %s", filepath, e)


def load_agent_data(filepath: str) -> Optional[Dict[str, Any]]:
    """Loads the agent's learned data from a file."""
    if not filepath or not isinstance(filepath, str):
        logger.error(
            "Cannot load agent data: Invalid filepath provided (received: %s).",
            filepath,
        )
        return None

    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
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
                iteration + 1,  # Log the iteration we are starting
            )
            # Return the whole dictionary
            return loaded_data
        if os.path.exists(filepath):  # File exists but is empty
            logger.warning(
                "Agent data file found at %s but is empty. Starting fresh.", filepath
            )
            return None
        else:  # File does not exist
            logger.info("Agent data file not found at %s. Starting fresh.", filepath)
            return None
    except (
        OSError,
        pickle.UnpicklingError,
        EOFError,
        ValueError,
    ) as e:  # Added ValueError for potential joblib issues
        logger.error("Error loading agent data from %s: %s", filepath, e)
        return None
