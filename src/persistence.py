import joblib
import os
from typing import Tuple, Optional
import logging

from .utils import PolicyDict

logger = logging.getLogger(__name__)

def save_agent_data(regret_sum: PolicyDict,
                    strategy_sum: PolicyDict,
                    iteration: int,
                    filepath: str):
    """Saves the agent's learned data (regrets, strategy sums, iteration count)."""
    try:
        data_to_save = {
            'regret_sum': regret_sum,
            'strategy_sum': strategy_sum,
            'iteration': iteration
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(data_to_save, filepath)
        logger.info(f"Agent data saved to {filepath} at iteration {iteration}")
    except Exception as e:
        logger.error(f"Error saving agent data to {filepath}: {e}")

def load_agent_data(filepath: str) -> Optional[Tuple[PolicyDict, PolicyDict, int]]:
    """Loads the agent's learned data from a file."""
    try:
        if os.path.exists(filepath):
            loaded_data = joblib.load(filepath)
            iteration = loaded_data.get('iteration', 0)
            logger.info(f"Agent data loaded from {filepath}. Resuming from iteration {iteration}.")
            return (
                loaded_data.get('regret_sum', {}),
                loaded_data.get('strategy_sum', {}),
                iteration
            )
        else:
            logger.info(f"Agent data file not found at {filepath}. Starting fresh.")
            return None
    except Exception as e:
        logger.error(f"Error loading agent data from {filepath}: {e}")
        return None