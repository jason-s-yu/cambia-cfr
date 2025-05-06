# src/live_log_handler.py
"""Custom logging handler to integrate with Rich Live display."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Use forward reference to avoid circular import at runtime
    from .live_display import LiveDisplayManager


class LiveLogHandler(logging.Handler):
    """
    A logging handler that forwards records to a LiveDisplayManager instance.
    """

    def __init__(self, display_manager: "LiveDisplayManager", level=logging.NOTSET):
        """
        Initialize the handler.

        Args:
            display_manager: The LiveDisplayManager instance to forward logs to.
            level: The minimum logging level for this handler.
        """
        super().__init__(level=level)
        # Ensure display_manager is correctly typed using the forward reference
        self.display_manager: "LiveDisplayManager" = display_manager

    def emit(self, record: logging.LogRecord):
        """
        Forwards the log record to the display manager.
        """
        try:
            # Check if display_manager is assigned before using
            if hasattr(self, "display_manager") and self.display_manager:
                # Let the display manager handle formatting and adding
                self.display_manager.add_log_record(record)
            else:
                # Fallback or error if manager not set (shouldn't happen with correct init)
                pass  # Or maybe print to stderr?
        except Exception:
            self.handleError(record)  # Default error handling
