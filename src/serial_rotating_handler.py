# src/serial_rotating_handler.py
"""Custom Log Handler for Serial Rotation"""

import logging.handlers
import os
import re
import glob


class SerialRotatingFileHandler(logging.handlers.BaseRotatingHandler):
    """
    Handler for logging to a set of files, which switches from one file
    to the next when the current file reaches a certain size. The numbering
    scheme ensures file_001.log is the oldest, file_002.log is the next oldest, etc.,
    with padding determined by backupCount. Writes the current active log file path
    to a '.current_log' file in its directory.
    """

    CURRENT_LOG_MARKER_FILENAME = ".current_log"

    def __init__(
        self,
        filename_pattern: str,
        mode="a",
        maxBytes=0,
        backupCount=0,
        encoding=None,
        delay=False,
    ):
        """
        Initialize the handler.

        The filename_pattern should include the base path and name format
        *without* the serial number and '.log', e.g., "/path/to/log_base".
        The handler will append "_NNN.log". Padding 'NNN' depends on backupCount.
        """
        self.base_pattern = filename_pattern
        self.current_serial = 1
        self.maxBytes = maxBytes
        self.backupCount = backupCount

        if self.backupCount > 0:
            self._pad_length = len(str(self.backupCount))
        else:
            self._pad_length = 4

        self._determine_initial_serial()
        initial_filename = (
            f"{self.base_pattern}_{self.current_serial:0{self._pad_length}d}.log"
        )

        logging.handlers.BaseRotatingHandler.__init__(
            self, initial_filename, mode, encoding, delay
        )

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
        log_files = glob.glob(f"{self.base_pattern}_*.log")
        max_serial = 0
        pattern = rf"_(\d{{{self._pad_length}}})\.log$"
        for f in log_files:
            match = re.search(pattern, f)
            if match:
                serial = int(match.group(1))
                if serial > max_serial:
                    max_serial = serial
        self.current_serial = max_serial + 1
        if max_serial == 0 and self.backupCount > 0:
            self.current_serial = 1

    def doRollover(self):
        """
        Do a rollover, closing the current file and opening the next one.
        Handles deletion of the oldest file if backupCount is exceeded.
        Uses dynamic padding and updates the .current_log marker.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        self.current_serial += 1
        self.baseFilename = (
            f"{self.base_pattern}_{self.current_serial:0{self._pad_length}d}.log"
        )

        if self.backupCount > 0:
            log_files = glob.glob(f"{self.base_pattern}_*.log")
            pattern = rf"_(\d{{{self._pad_length}}})\.log$"
            files_with_serials = []
            for f in log_files:
                match = re.search(pattern, f)
                if match:
                    serial = int(match.group(1))
                    files_with_serials.append((serial, f))
            files_with_serials.sort()
            num_to_delete = len(files_with_serials) - self.backupCount + 1
            if num_to_delete > 0:
                for i in range(num_to_delete):
                    try:
                        _, file_to_delete = files_with_serials[i]
                        os.remove(file_to_delete)
                    except IndexError:
                        break
                    except OSError as e:
                        print(f"Error removing old log file {file_to_delete}: %s", e)

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
            msg = "%s\n" % self.format(record)
            msg_bytes = msg.encode(self.encoding or "utf-8")
            self.stream.seek(0, 2)
            if self.stream.tell() + len(msg_bytes) >= self.maxBytes:
                return 1
        return 0
