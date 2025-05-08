"""
src/serial_rotating_handler.py

Custom Log Handler for Serial Rotation
"""

import logging.handlers
import multiprocessing
import os
import sys
import re
import glob
import queue
from typing import Optional, Union, TYPE_CHECKING

# Use TYPE_CHECKING for conditional import
if TYPE_CHECKING:
    from .config import LoggingConfig

# NOTE: This module should NOT use the logger it might be attached to
# for its internal operations, especially within doRollover, to avoid recursion.
# Use print(..., file=sys.stderr) for internal status messages if needed.
# logger = logging.getLogger(__name__) # Avoid using logger internally here


class SerialRotatingFileHandler(logging.handlers.BaseRotatingHandler):
    """
    Handler for logging to a set of files, which switches from one file
    to the next when the current file reaches a certain size. The numbering
    scheme ensures file_001.log is the oldest, file_002.log is the next oldest, etc.,
    with padding determined by backupCount. Writes the current active log file path
    to a '.current_log' file in its directory.
    If log archiving is enabled, when the number of log files reaches the backupCount + 1,
    a signal is sent to the archive_queue to perform a batch archive of all
    current log files for this stream.
    """

    CURRENT_LOG_MARKER_FILENAME = ".current_log"

    def __init__(
        self,
        filename_pattern: str,
        mode: str = "a",
        maxBytes: int = 0,
        backupCount: int = 0,
        encoding: Optional[str] = None,
        delay: bool = False,
        archive_queue: Optional[Union[queue.Queue, "multiprocessing.Queue"]] = None,
        logging_config_snapshot: Optional["LoggingConfig"] = None,
    ):
        """
        Initialize the handler.

        The filename_pattern should include the base path and name format
        *without* the serial number and '.log', e.g., "/path/to/log_base".
        The handler will append "_NNN.log". Padding 'NNN' depends on backupCount.

        Args:
            archive_queue: A queue to send file paths to for archiving.
            logging_config_snapshot: A snapshot of the LoggingConfig from the main config.
        """
        self.base_pattern = filename_pattern
        self.current_serial = 1
        self.maxBytes = maxBytes
        self.backupCount = (
            backupCount  # This now triggers batch archiving when count is met
        )
        self.archive_queue = archive_queue
        self.logging_config = logging_config_snapshot

        if self.backupCount > 0:
            # Determine padding based on the backup count itself, ensuring enough digits
            self._pad_length = len(str(self.backupCount))
        else:
            self._pad_length = 4  # Default padding if no backup count

        self._determine_initial_serial()
        # Dynamically adjust padding if the determined initial serial needs more digits
        self._pad_length = max(self._pad_length, len(str(self.current_serial)))

        initial_filename = (
            f"{self.base_pattern}_{self.current_serial:0{self._pad_length}d}.log"
        )

        # Call parent class init with its expected signature
        super().__init__(initial_filename, mode, encoding, delay)

        if not delay:  # If not delaying, stream is open, so write marker
            self._write_current_log_marker()

    def _write_current_log_marker(self):
        """Writes the path of the current log file to a marker file."""
        if self.baseFilename:
            marker_path = os.path.join(
                os.path.dirname(self.baseFilename), self.CURRENT_LOG_MARKER_FILENAME
            )
            try:
                with open(marker_path, "w", encoding="utf-8") as f:
                    f.write(os.path.abspath(self.baseFilename))
            except OSError as e:
                # Use print for critical log handler errors
                print(
                    f"ERROR: SerialRotatingFileHandler: Could not write current log marker to {marker_path}: {e}",
                    file=sys.stderr,
                )

    def _determine_initial_serial(self):
        """Find the highest existing serial number or start from 1."""
        log_dir = os.path.dirname(self.base_pattern)
        base_name_only = os.path.basename(self.base_pattern)
        glob_pattern_str = f"{base_name_only}_*.log"  # Glob pattern relative to log_dir

        # Ensure log_dir exists, otherwise glob might behave unexpectedly or error
        if not os.path.isdir(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except OSError as e:
                # If directory creation fails, cannot determine initial serial reliably
                print(
                    f"ERROR: SerialRotatingFileHandler: Failed to create log directory {log_dir}: {e}",
                    file=sys.stderr,
                )
                self.current_serial = 1
                return

        log_files = glob.glob(os.path.join(log_dir, glob_pattern_str))

        max_serial = 0
        # Regex to match the serial number part of the filename, e.g., "_001.log"
        # We need to match filenames like 'prefix_001.log', 'prefix_002.log' etc.
        # The base_pattern is 'path/to/prefix'.
        filename_regex_pattern = rf"{re.escape(base_name_only)}_(\d+)\.log$"

        for f_path in log_files:
            f_name = os.path.basename(f_path)
            match = re.search(filename_regex_pattern, f_name)
            if match:
                try:
                    serial = int(match.group(1))
                    if serial > max_serial:
                        max_serial = serial
                except ValueError:
                    # Should not happen if regex is correct
                    pass  # Ignore non-integer serials
        self.current_serial = max_serial + 1
        if max_serial == 0:  # If no files found, start at 1
            self.current_serial = 1

    def doRollover(self):
        """
        Do a rollover, closing the current file and opening the next one.
        If archiving is enabled and the number of existing log files reaches backupCount,
        send a batch archive request to the archive_queue *before* opening the new file.
        Uses print() for internal status to avoid recursive logging.
        """
        # DO NOT use logger inside this method to avoid recursion
        if self.stream:
            self.stream.close()
            self.stream = None

        # --- Batch Archiving Trigger (Check *before* incrementing serial) ---
        archive_enabled = self.logging_config and self.logging_config.log_archive_enabled
        if self.backupCount > 0 and archive_enabled and self.archive_queue:
            log_dir = os.path.dirname(self.base_pattern)
            base_name_only = os.path.basename(self.base_pattern)
            glob_pattern_str = f"{base_name_only}_*.log"

            if os.path.isdir(log_dir):
                # Glob for existing files matching the pattern
                existing_log_files = glob.glob(os.path.join(log_dir, glob_pattern_str))
                num_existing_logs = len(existing_log_files)

                # Trigger archive if the number of existing files is >= backupCount.
                # This means the rollover we are about to perform will create the (backupCount + 1)-th file or more.
                if num_existing_logs >= self.backupCount:
                    # Use print for internal status message
                    print(
                        f"INFO: SerialRotatingFileHandler: Log file count ({num_existing_logs}) meets/exceeds backup count ({self.backupCount}). Queuing BATCH ARCHIVE for pattern {self.base_pattern}",
                        file=sys.stderr,  # Log to stderr to avoid interfering with potential stdout redirection
                    )
                    try:
                        # Queue the request with directory and base pattern
                        self.archive_queue.put_nowait(
                            ("BATCH_ARCHIVE", log_dir, self.base_pattern)
                        )  # type: ignore
                    except queue.Full:
                        print(
                            f"ERROR: SerialRotatingFileHandler: Archive queue full. Could not queue BATCH ARCHIVE for {self.base_pattern}.",
                            file=sys.stderr,
                        )
                    except Exception as q_err:
                        print(
                            f"ERROR: SerialRotatingFileHandler: Failed to queue BATCH_ARCHIVE for {self.base_pattern}: {q_err}",
                            file=sys.stderr,
                        )
                # else:
                # Optionally print debug status if needed, but can be noisy
                # print(
                #     f"DEBUG: SerialRotatingFileHandler: Log file count ({num_existing_logs}) below backup count ({self.backupCount}). No archive triggered for {self.base_pattern}.",
                #     file=sys.stderr
                # )
            else:
                # Log directory doesn't exist, cannot check count
                print(
                    f"WARNING: SerialRotatingFileHandler: Log directory {log_dir} not found during rollover archive check.",
                    file=sys.stderr,
                )
        # --- End Batch Archiving Trigger ---

        # Increment serial number for the new file
        self.current_serial += 1

        # Update pad_length dynamically if current_serial exceeds previous padding capacity
        current_pad_needed = len(str(self.current_serial))
        self._pad_length = max(self._pad_length, current_pad_needed)

        # Set the filename for the new log file
        self.baseFilename = (
            f"{self.base_pattern}_{self.current_serial:0{self._pad_length}d}.log"
        )

        # Open the new log file if not delaying
        if not self.delay:
            try:
                self.stream = self._open()
                self._write_current_log_marker()  # Update marker after opening new file
            except Exception as e_open:
                print(
                    f"ERROR: SerialRotatingFileHandler: Failed to open new stream after rollover: {e_open}",
                    file=sys.stderr,
                )

    def shouldRollover(self, record):
        """
        Determine if rollover should occur based on maxBytes.
        Also writes current log marker if stream was opened due to delay.
        """
        if self.stream is None:  # Stream may not have been opened yet
            try:
                self.stream = self._open()
                self._write_current_log_marker()  # Write marker if stream just opened
            except Exception as e_open:
                # Use basic print as logging itself might be failing
                print(
                    f"ERROR: SerialRotatingFileHandler: Failed to open stream in shouldRollover: {e_open}",
                    file=sys.stderr,
                )
                return 0  # Cannot proceed if stream cannot be opened

        if self.maxBytes > 0:
            # Ensure the stream is in append mode for accurate size check if opened in 'w' initially
            # Although our mode is 'a' by default.
            try:
                self.stream.seek(0, 2)  # Go to end of file
                current_size = self.stream.tell()
            except (OSError, ValueError) as e:
                # Handle cases where seek/tell might fail (e.g., stream closed)
                # Cannot use logger here due to potential recursion
                print(
                    f"ERROR: SerialRotatingFileHandler: Error seeking/telling stream position: {e}",
                    file=sys.stderr,
                )
                return 0  # Cannot determine size, don't rollover

            msg = "%s\n" % self.format(record)
            # Use utf-8 as a fallback if no encoding is set.
            try:
                # Attempt to encode with specified encoding or fallback
                msg_bytes = msg.encode(
                    self.encoding if self.encoding is not None else "utf-8", "replace"
                )
            except Exception:
                # If encoding fails for any reason, try a safe fallback
                msg_bytes = msg.encode("utf-8", "replace")

            if current_size + len(msg_bytes) >= self.maxBytes:
                return 1
        return 0
