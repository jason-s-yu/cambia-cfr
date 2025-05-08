"""
src/serial_rotating_handler.py

Custom Log Handler for Serial Rotation
"""

import logging.handlers
import multiprocessing
import os
import re
import glob
import queue
from typing import Optional, Union, TYPE_CHECKING

# Use TYPE_CHECKING for conditional import
if TYPE_CHECKING:
    from .config import LoggingConfig

logger = logging.getLogger(__name__)


class SerialRotatingFileHandler(logging.handlers.BaseRotatingHandler):
    """
    Handler for logging to a set of files, which switches from one file
    to the next when the current file reaches a certain size. The numbering
    scheme ensures file_001.log is the oldest, file_002.log is the next oldest, etc.,
    with padding determined by backupCount. Writes the current active log file path
    to a '.current_log' file in its directory.
    If log archiving is enabled, when the number of log files exceeds backupCount,
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
            self._pad_length = len(
                str(self.backupCount + 1)
            )  # Pad for one more than backupCount for current
        else:
            self._pad_length = 4  # Default padding if no backup count

        self._determine_initial_serial()
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
                    f"ERROR: SerialRotatingFileHandler: Could not write current log marker to {marker_path}: {e}"
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
                    f"ERROR: SerialRotatingFileHandler: Failed to create log directory {log_dir}: {e}"
                )
                self.current_serial = 1
                return

        log_files = glob.glob(os.path.join(log_dir, glob_pattern_str))

        max_serial = 0
        # Regex to match the serial number part of the filename, e.g., "_001.log"
        # It uses self._pad_length if available, or a general pattern.
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
        If archiving is enabled and the number of log files reaches the backupCount + 1,
        send a batch archive request to the archive_queue.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        self.current_serial += 1
        # Update pad_length dynamically if current_serial exceeds previous padding capacity
        current_pad_needed = len(str(self.current_serial))
        if self.backupCount > 0:
            self._pad_length = max(current_pad_needed, len(str(self.backupCount)))
        else:
            self._pad_length = max(current_pad_needed, 4)

        self.baseFilename = (
            f"{self.base_pattern}_{self.current_serial:0{self._pad_length}d}.log"
        )

        # --- Batch Archiving Trigger ---
        archive_enabled = self.logging_config and self.logging_config.log_archive_enabled
        if self.backupCount > 0 and archive_enabled and self.archive_queue:
            log_dir = os.path.dirname(self.base_pattern)
            base_name_only = os.path.basename(self.base_pattern)
            glob_pattern_str = f"{base_name_only}_*.log"

            if os.path.isdir(log_dir):
                log_files = glob.glob(os.path.join(log_dir, glob_pattern_str))
                # Check count *after* potentially creating the new file (though stream isn't opened yet)
                # The check should happen based on files *before* the new one is opened,
                # triggering when the *next* one would exceed the count.
                # So, if count == backupCount, the *next* file created (self.current_serial)
                # will make the total count backupCount + 1, triggering the archive of
                # the files up to and including the one *just before* the current one.
                # Let's adjust: Trigger when len(log_files) >= self.backupCount
                # This means we have the desired number of backups, and the *next* rotation
                # (the one happening now) will create the (backupCount + 1)-th file.
                # We archive the existing `backupCount` files.

                # Re-think: The request says "when the total number [...] reaches log_backup_count + 1".
                # This implies the check happens *after* the new file exists logically.
                # Let's check the count *before* opening the new stream.
                # The current file (before rollover) is `self.base_pattern` + `self.current_serial - 1`.
                # The new file will be `self.base_pattern` + `self.current_serial`.

                # Glob for existing files *before* opening the new one.
                existing_log_files = glob.glob(os.path.join(log_dir, glob_pattern_str))

                # If the number of existing files equals or exceeds backupCount,
                # the new file we are about to create (self.current_serial) will bring the total to backupCount + 1 or more.
                # Let's trigger when the number of existing files is *exactly* backupCount.
                # This means the new file (self.current_serial) will be the (backupCount + 1)-th file.
                # We then archive *all* backupCount + 1 files.

                # Let's simplify: trigger when the glob count hits the threshold needed.
                # If we have backupCount logs, the next one makes backupCount+1.
                # Check after the new file is created (conceptually).
                # Glob again *after* setting self.baseFilename
                new_glob_pattern = os.path.join(log_dir, glob_pattern_str)
                current_log_files = glob.glob(new_glob_pattern)

                # Add the conceptually new file to the count if it wasn't picked by glob yet
                # (it might not exist physically until the stream is opened).
                # It's safer to check the count based on serial numbers.
                # Trigger if self.current_serial > self.backupCount

                # --- Revised Trigger Logic ---
                # Trigger archive if the serial number of the file *we are about to create*
                # is greater than the backup count. This means we already have 'backupCount'
                # older files (1 to backupCount).
                # However, the request implies archiving *all* existing when the threshold is met.
                # Let's stick to the "count reaches backupCount + 1" interpretation.
                # Glob for all files matching the pattern. If count >= backupCount + 1, trigger.

                final_glob_pattern = os.path.join(log_dir, f"{base_name_only}_*.log")
                all_files = glob.glob(final_glob_pattern)
                # Add the file we are about to create if it wasn't globbed (unlikely but possible)
                current_file_path_abs = os.path.abspath(self.baseFilename)
                if current_file_path_abs not in [os.path.abspath(f) for f in all_files]:
                    # This condition shouldn't really be met if the filename follows the pattern
                    # but acts as a safeguard if the file isn't created before this check.
                    effective_count = len(all_files) + 1
                else:
                    effective_count = len(all_files)

                # Trigger if the total number of log files (including the new one)
                # *would be* backupCount + 1. Since we just incremented current_serial,
                # if current_serial is exactly backupCount + 1, it means we just created
                # the file that reaches the threshold.
                if self.current_serial == self.backupCount + 1:
                    logger.info(
                        "Log count threshold reached (%d >= %d). Queuing BATCH ARCHIVE for pattern %s",
                        self.current_serial,
                        self.backupCount + 1,
                        self.base_pattern,
                    )
                    try:
                        # Queue the request with directory and base pattern
                        self.archive_queue.put_nowait(
                            ("BATCH_ARCHIVE", log_dir, self.base_pattern)
                        )  # type: ignore
                    except queue.Full:
                        logger.error(
                            "Archive queue full. Could not queue BATCH ARCHIVE for %s.",
                            self.base_pattern,
                        )
                    except Exception as q_err:
                        logger.error(
                            "Failed to queue BATCH_ARCHIVE for %s: %s",
                            self.base_pattern,
                            q_err,
                        )
            else:
                # Log directory doesn't exist, cannot check count
                pass

        # --- End Batch Archiving Trigger ---

        if not self.delay:
            self.stream = self._open()
            self._write_current_log_marker()  # Update marker after opening new file

    def shouldRollover(self, record):
        """
        Determine if rollover should occur based on maxBytes.
        Also writes current log marker if stream was opened due to delay.
        """
        if self.stream is None:  # Stream may not have been opened yet
            self.stream = self._open()
            self._write_current_log_marker()  # Write marker if stream just opened

        if self.maxBytes > 0:
            # Ensure the stream is in append mode for accurate size check if opened in 'w' initially
            # Although our mode is 'a' by default.
            self.stream.seek(0, 2)  # Go to end of file
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

            if self.stream.tell() + len(msg_bytes) >= self.maxBytes:
                return 1
        return 0
