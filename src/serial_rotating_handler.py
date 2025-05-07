"""
src/serial_rotating_handler.py

Custom Log Handler for Serial Rotation
"""

import logging.handlers
import os
import re
import glob
import queue
from typing import Optional, Union

try:
    from .config import LoggingConfig
except ImportError:
    LoggingConfig = None  # type: ignore


class SerialRotatingFileHandler(logging.handlers.BaseRotatingHandler):
    """
    Handler for logging to a set of files, which switches from one file
    to the next when the current file reaches a certain size. The numbering
    scheme ensures file_001.log is the oldest, file_002.log is the next oldest, etc.,
    with padding determined by backupCount. Writes the current active log file path
    to a '.current_log' file in its directory.
    If log archiving is enabled, files that would be deleted are instead sent to an
    archive queue.
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
        logging_config_snapshot: Optional[LoggingConfig] = None,
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
        self.backupCount = backupCount  # This now defines the window of uncompressed logs
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
        Handles deletion of the oldest file if backupCount is exceeded,
        or sends it to the archive_queue if archiving is enabled.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        self.current_serial += 1
        # Update pad_length dynamically if current_serial exceeds previous padding capacity
        # This is important if backupCount is small but many rotations happen.
        current_pad_needed = len(str(self.current_serial))
        if self.backupCount > 0:
            # Ensure padding accommodates at least up to backupCount + current number
            # Example: backupCount=5, current_serial=10 -> pad for 10
            # Example: backupCount=99, current_serial=5 -> pad for 99
            self._pad_length = max(current_pad_needed, len(str(self.backupCount)))
        else:
            self._pad_length = max(current_pad_needed, 4)

        self.baseFilename = (
            f"{self.base_pattern}_{self.current_serial:0{self._pad_length}d}.log"
        )

        if self.backupCount > 0:
            log_dir = os.path.dirname(self.base_pattern)
            base_name_only = os.path.basename(self.base_pattern)
            glob_pattern_str = f"{base_name_only}_*.log"

            if not os.path.isdir(
                log_dir
            ):  # Should not happen if _determine_initial_serial ran
                files_with_serials = []
            else:
                log_files = glob.glob(os.path.join(log_dir, glob_pattern_str))
                filename_regex_pattern = rf"{re.escape(base_name_only)}_(\d+)\.log$"
                files_with_serials = []
                for f_path in log_files:
                    f_name = os.path.basename(f_path)
                    match = re.search(filename_regex_pattern, f_name)
                    if match:
                        try:
                            serial = int(match.group(1))
                            # Store absolute path for removal/archiving
                            files_with_serials.append((serial, os.path.abspath(f_path)))
                        except ValueError:
                            pass
                files_with_serials.sort()

            # Rollover means we've just created current_serial.
            # We want to keep backupCount *older* files.
            # So, if number of files (excluding current) > backupCount, delete/archive oldest.
            # Total files existing before this new one is len(files_with_serials)
            # Files that are "backups" are those *not* including the one we just rolled to.
            # The list `files_with_serials` contains all files *before* the current one was opened.
            # If we have more files in `files_with_serials` than `backupCount`,
            # the excess ones at the beginning of the sorted list are candidates.

            num_to_consider_for_removal_or_archive = (
                len(files_with_serials) - self.backupCount
            )

            archive_enabled = (
                self.logging_config and self.logging_config.log_archive_enabled
            )

            if num_to_consider_for_removal_or_archive > 0:
                for i in range(num_to_consider_for_removal_or_archive):
                    try:
                        _, file_to_process = files_with_serials[i]
                        if archive_enabled and self.archive_queue:
                            try:
                                self.archive_queue.put_nowait(file_to_process)  # type: ignore
                                # Use print for direct feedback from handler if logger not configured yet/failsafe
                                print(
                                    f"INFO: SerialRotatingFileHandler: Queued {file_to_process} for archiving."
                                )
                            except queue.Full:
                                print(
                                    f"WARNING: SerialRotatingFileHandler: Archive queue full. Could not queue {file_to_process}. Deleting instead."
                                )
                                os.remove(file_to_process)
                            except Exception as q_err:
                                print(
                                    f"ERROR: SerialRotatingFileHandler: Failed to queue {file_to_process} for archiving: {q_err}. Deleting instead."
                                )
                                os.remove(file_to_process)
                        else:
                            # print(f"DEBUG: SerialRotatingFileHandler: Removing old log file {file_to_process} (Archiving disabled or no queue).")
                            os.remove(file_to_process)
                    except IndexError:  # Should not happen if logic is correct
                        break
                    except OSError as e:
                        print(
                            f"ERROR: SerialRotatingFileHandler: Error processing old log file {file_to_process}: {e}"
                        )

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
            msg_bytes = msg.encode(
                self.encoding if self.encoding is not None else "utf-8", "replace"
            )

            if self.stream.tell() + len(msg_bytes) >= self.maxBytes:
                return 1
        return 0
