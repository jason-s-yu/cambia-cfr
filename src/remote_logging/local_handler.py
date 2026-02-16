# src/remote_logging/local_handler.py

import logging
from typing import Dict, Any, List
from .log_handler import LogHandler


class LocalLogHandler(LogHandler):
    """
    A log handler that directs output to the existing local file logger.
    This acts as an adapter to ensure the local file logging system
    conforms to the LogHandler interface. It does not re-implement
    rotation or archiving, but rather uses the existing setup.
    """

    def __init__(self, main_logger_name: str = "cambia"):
        """
        Initializes the LocalLogHandler.

        Args:
            main_logger_name: The name of the main application logger configured
                              with file rotation, etc.
        """
        self.logger = logging.getLogger(main_logger_name)

    def log(self, level: str, message: str, context: Dict[str, Any] = None):
        """Logs a single message using the standard logging library."""
        log_level = getattr(logging, level.upper(), logging.INFO)
        extra = {"context": context} if context else None
        self.logger.log(log_level, message, extra=extra)

    def log_batch(self, batch: List[Dict[str, Any]]):
        """Logs a batch of messages by iterating and calling log()."""
        for record in batch:
            self.log(
                level=record.get("level", "INFO"),
                message=record.get("message", ""),
                context=record.get("context"),
            )

    def close(self):
        """
        No-op for the local handler, as the main logging shutdown is
        handled elsewhere in the application's lifecycle.
        """
        pass
