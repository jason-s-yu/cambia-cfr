# src/serial_rotating_handler.py (New file)
"""Custom Log Handler for Serial Rotation"""

import logging.handlers
import os
import re
import glob


class SerialRotatingFileHandler(logging.handlers.BaseRotatingHandler):
    """
    Handler for logging to a set of files, which switches from one file
    to the next when the current file reaches a certain size. The numbering
    scheme ensures file_001.log is the oldest, file_002.log is the next oldest, etc.
    """

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
        *without* the serial number, e.g., "/path/to/log_base". The handler
        will append "_NNN.log".
        """
        self.base_pattern = filename_pattern
        self.current_serial = 1
        self.maxBytes = maxBytes
        self.backupCount = backupCount

        # Determine the initial file to open or the next available serial
        self._determine_initial_serial()

        # Construct the initial filename
        if self.backupCount > 0:
            pad_length = len(str(self.backupCount))
        else:
            pad_length = 4  # Default padding if backupCount is not set
        initial_filename = f"{self.base_pattern}_{self.current_serial:0{pad_length}d}.log"

        # Initialize the parent class
        logging.handlers.BaseRotatingHandler.__init__(
            self, initial_filename, mode, encoding, delay
        )

    def _determine_initial_serial(self):
        """Find the highest existing serial number or start from 1."""
        log_files = glob.glob(f"{self.base_pattern}_*.log")
        max_serial = 0
        for f in log_files:
            match = re.search(r"_(\d{3,})\.log$", f)
            if match:
                serial = int(match.group(1))
                if serial > max_serial:
                    max_serial = serial
        # Start writing to the next serial number
        self.current_serial = max_serial + 1
        # If starting fresh and backupCount is set, limit the initial serial
        if max_serial == 0 and self.backupCount > 0:
            self.current_serial = 1  # Always start at 1 if no files exist

        # Ensure we don't immediately exceed backupCount on initialization
        # (relevant if restarting and highest existing serial is >= backupCount)
        # if self.backupCount > 0 and self.current_serial > self.backupCount:
        # This scenario is tricky - should we overwrite the oldest or stop?
        # For simplicity, let's allow starting above backupCount if files exist,
        # and rely on rollover to prune. If starting fresh, it's 1.
        # pass

    def doRollover(self):
        """
        Do a rollover, closing the current file and opening the next one.
        The numbering scheme means we just open the next serial. We also
        handle deletion of the oldest file if backupCount is exceeded.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        # Increment serial for the new file
        self.current_serial += 1
        self.baseFilename = f"{self.base_pattern}_{self.current_serial:03d}.log"

        # Handle backup deletion *before* opening the new file
        if self.backupCount > 0:
            # Find all log files matching the pattern
            log_files = glob.glob(f"{self.base_pattern}_*.log")
            if len(log_files) >= self.backupCount:
                # Parse serial numbers and find the oldest
                files_to_delete = []
                for f in log_files:
                    match = re.search(r"_(\d{3,})\.log$", f)
                    if match:
                        serial = int(match.group(1))
                        # Calculate the cutoff serial number
                        cutoff_serial = self.current_serial - self.backupCount
                        if serial < cutoff_serial:
                            files_to_delete.append((serial, f))

                # Sort by serial number (ascending) to delete the oldest first
                files_to_delete.sort()

                # Delete files older than the backup count allows
                num_to_delete = (
                    len(log_files) - self.backupCount + 1
                )  # Calculate how many need deletion (+1 because we are about to open a new one)
                for i in range(min(num_to_delete, len(files_to_delete))):
                    serial_to_delete, file_to_delete = files_to_delete[i]
                    try:
                        os.remove(file_to_delete)
                        # print(f"DEBUG: Deleting old log file: {file_to_delete}")
                    except OSError as e:
                        # Log error appropriately - can't use logger here easily
                        print(f"Error removing old log file {file_to_delete}: {e}")

        # Open the new file (will be self.baseFilename)
        if not self.delay:
            self.stream = self._open()

    def shouldRollover(self, record):
        """
        Determine if rollover should occur.
        Based on maxBytes.
        """
        if self.stream is None:  # Stream may not have been opened yet
            self.stream = self._open()
        if self.maxBytes > 0:  # Are we rolling over?
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)  # Due to calculations using tell()
            if self.stream.tell() + len(msg) >= self.maxBytes:
                return 1
        return 0
