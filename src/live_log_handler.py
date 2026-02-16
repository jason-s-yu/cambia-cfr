"""
src/live_log_handler.py

Custom logging handler to integrate with Rich Live display.
"""

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .live_display import LiveDisplayManager


class LiveLogHandler(logging.Handler):
    """
    A logging handler that forwards records to a LiveDisplayManager instance,
    with specific filtering for unwanted messages.
    """

    def __init__(self, display_manager: "LiveDisplayManager", level=logging.NOTSET):
        """
        Initialize the handler.

        Args:
            display_manager: The LiveDisplayManager instance to forward logs to.
            level: The minimum logging level for this handler.
        """
        super().__init__(level=level)
        self.display_manager: "LiveDisplayManager" = display_manager

    def emit(self, record: logging.LogRecord):
        """
        Forwards the log record to the display manager, unless it's a
        filtered message type (e.g., archive queuing messages).
        """
        # --- Filtering Logic ---
        # Skip "Queued ... for archiving" messages from serial_rotating_handler
        # These are useful for file logs but clutter the live display.
        try:
            # Check if the logger name and message content match the filter criteria
            if (
                record.name == "src.serial_rotating_handler"
                and "Queued" in record.getMessage()
                and "for archiving" in record.getMessage()
            ):
                # If it matches, simply return and do not forward to the display manager.
                return
        except Exception as e:  # JUSTIFIED: logging handler resilience
            # Handle potential errors during getMessage() or filtering
            # Use stderr to avoid recursion in logging
            print(f"LiveLogHandler filter error: {e}", file=sys.stderr)
            self.handleError(record)
            # Decide whether to proceed or return based on the error
            # For safety, let's return to avoid potentially crashing the display loop
            return

        # --- Forwarding Logic ---
        # If the record was not filtered out, proceed to forward it.
        try:
            if hasattr(self, "display_manager") and self.display_manager:
                self.display_manager.add_log_record(record)
            else:
                pass  # Fallback or error if manager not set (shouldn't happen)
        except Exception as e:  # JUSTIFIED: logging handler resilience
            print(f"LiveLogHandler emit error: {e}", file=sys.stderr)
            self.handleError(record)
