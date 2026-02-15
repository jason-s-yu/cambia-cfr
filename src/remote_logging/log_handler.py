# src/remote_logging/log_handler.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class LogHandler(ABC):
    """
    Abstract base class for all logging handlers.
    Defines the standard interface for logging messages.
    """

    @abstractmethod
    def log(self, level: str, message: str, context: Dict[str, Any] = None):
        """
        Logs a single message.

        Args:
            level: The log level (e.g., "INFO", "ERROR").
            message: The log message string.
            context: Optional dictionary for structured data.
        """
        pass

    @abstractmethod
    def log_batch(self, batch: List[Dict[str, Any]]):
        """
        Logs a batch of messages.

        Args:
            batch: A list of log records, where each record is a dictionary
                   containing keys like 'level', 'message', 'context'.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Closes the handler, flushing any buffered logs and releasing resources.
        """
        pass
