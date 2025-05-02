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
    with padding determined by backupCount.
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
        *without* the serial number and '.log', e.g., "/path/to/log_base".
        The handler will append "_NNN.log". Padding 'NNN' depends on backupCount.
        """
        self.base_pattern = filename_pattern
        self.current_serial = 1
        self.maxBytes = maxBytes
        self.backupCount = backupCount

        # Determine padding based on backupCount
        if self.backupCount > 0:
            # Calculate padding needed, e.g., backupCount=99 -> pad 2, backupCount=100 -> pad 3
            self._pad_length = len(str(self.backupCount))
        else:
            self._pad_length = 4  # Default padding if backupCount is unlimited or zero

        # Determine the initial file to open or the next available serial
        self._determine_initial_serial()

        # Construct the initial filename using the determined padding
        initial_filename = (
            f"{self.base_pattern}_{self.current_serial:0{self._pad_length}d}.log"
        )

        # Initialize the parent class
        logging.handlers.BaseRotatingHandler.__init__(
            self, initial_filename, mode, encoding, delay
        )

    def _determine_initial_serial(self):
        """Find the highest existing serial number or start from 1."""
        log_files = glob.glob(f"{self.base_pattern}_*.log")
        max_serial = 0
        # Regex to match the padding dynamically based on _pad_length
        pattern = rf"_(\d{{{self._pad_length}}})\.log$"
        for f in log_files:
            match = re.search(pattern, f)
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
        # if self.backupCount > 0 and self.current_serial > self.backupCount:
        #   pass # Let rollover handle pruning

    def doRollover(self):
        """
        Do a rollover, closing the current file and opening the next one.
        Handles deletion of the oldest file if backupCount is exceeded.
        Uses dynamic padding.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        # Increment serial for the new file
        self.current_serial += 1
        # Use the determined padding length
        self.baseFilename = (
            f"{self.base_pattern}_{self.current_serial:0{self._pad_length}d}.log"
        )

        # Handle backup deletion *before* opening the new file
        if self.backupCount > 0:
            # Find all log files matching the pattern
            log_files = glob.glob(f"{self.base_pattern}_*.log")
            # Regex to match the padding dynamically
            pattern = rf"_(\d{{{self._pad_length}}})\.log$"

            files_with_serials = []
            for f in log_files:
                match = re.search(pattern, f)
                if match:
                    serial = int(match.group(1))
                    files_with_serials.append((serial, f))

            # Sort by serial number (ascending) to identify the oldest
            files_with_serials.sort()

            # Determine how many files need to be deleted
            num_to_delete = len(files_with_serials) - self.backupCount + 1
            if num_to_delete > 0:
                for i in range(num_to_delete):
                    try:
                        serial_to_delete, file_to_delete = files_with_serials[i]
                        os.remove(file_to_delete)
                    except IndexError:
                        break  # Stop if we run out of files in the list somehow
                    except OSError as e:
                        print(f"Error removing old log file {file_to_delete}: %s", e)

        # Open the new file (will be self.baseFilename)
        if not self.delay:
            self.stream = self._open()

    def shouldRollover(self, record):
        """
        Determine if rollover should occur based on maxBytes.
        """
        if self.stream is None:  # Stream may not have been opened yet
            # Associate the handler with the logger stream
            self.stream = self._open()
        if self.maxBytes > 0:  # Are we rolling over?
            # Format the message and calculate its size
            msg = "%s\n" % self.format(record)
            # Use encoded length for accurate size comparison
            msg_bytes = msg.encode(self.encoding or "utf-8")
            self.stream.seek(0, 2)  # Move to the end of the stream
            # Check if adding the new message exceeds maxBytes
            if self.stream.tell() + len(msg_bytes) >= self.maxBytes:
                return 1  # Perform rollover
        return 0  # No rollover needed
