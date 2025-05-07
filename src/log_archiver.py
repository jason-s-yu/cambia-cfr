"""
src/log_archiver.py

Manages archiving of worker log files.
"""

import os
import glob
import tarfile
import logging
import datetime
import re
import queue
import threading
import time
from typing import List, Optional, Union

from .config import Config

logger = logging.getLogger(__name__)

# Define a sentinel value for stopping the queue processing
QUEUE_SENTINEL = object()


class LogArchiver:
    """
    Handles archiving of log files passed via a queue.
    Archives are stored either directly in the file's directory or
    in a specified subdirectory within it.
    Operates asynchronously in a separate thread.
    """

    WORKER_DIR_PATTERN = (
        "w*"  # Pattern to identify worker directories (e.g., w0, w1, ...)
    )
    CURRENT_LOG_MARKER_FILENAME = ".current_log"

    def __init__(
        self,
        config: Config,
        archive_queue: Union[queue.Queue, "multiprocessing.Queue"],
        run_log_dir: str,
    ):
        """
        Initializes the LogArchiver.

        Args:
            config: The main application configuration.
            archive_queue: The queue from which to receive file paths for archiving.
            run_log_dir: The absolute path to the current run's log directory (used for context).
        """
        self.config = config
        self.archive_queue = archive_queue
        self.run_log_dir = run_log_dir
        self._archive_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if self.config.logging.log_archive_enabled:
            logger.info(
                "Log archiver initialized (enabled). Run log dir: %s", self.run_log_dir
            )
            logger.debug(
                "Archiver Config: MaxArchives=%d, ArchiveSubDir='%s'",
                self.config.logging.log_archive_max_archives,
                self.config.logging.log_archive_dir,
            )
        else:
            logger.info("Log archiver initialized (disabled).")

    def start(self):
        """Starts the log archiving thread if archiving is enabled."""
        if not self.config.logging.log_archive_enabled:
            logger.debug("Log archiving is disabled. Archiver thread not started.")
            return

        if self._archive_thread is not None and self._archive_thread.is_alive():
            logger.warning("Archive thread already running.")
            return

        self._stop_event.clear()
        self._archive_thread = threading.Thread(
            target=self._process_archive_queue, name="LogArchiverThread"
        )
        self._archive_thread.daemon = (
            True  # Allow main program to exit even if thread is running
        )
        self._archive_thread.start()
        logger.info("Log Archiver thread started.")

    def stop(self, timeout: Optional[float] = 5.0):
        """Signals the archiving thread to stop and waits for it to join."""
        if not self.config.logging.log_archive_enabled or self._archive_thread is None:
            if self.config.logging.log_archive_enabled:
                logger.debug(
                    "Log Archiver stop called, but thread was not running or not enabled."
                )
            return

        if not self._archive_thread.is_alive():
            logger.info("Archive thread already stopped.")
            return

        logger.info("Stopping Log Archiver thread...")
        self._stop_event.set()
        try:
            # Put sentinel on queue to ensure the thread wakes up if blocked on get()
            self.archive_queue.put(QUEUE_SENTINEL, timeout=1.0)  # type: ignore
        except queue.Full:
            logger.warning("Archive queue full while trying to send sentinel for stop.")
        except Exception as e:
            logger.error("Error sending sentinel to archive queue: %s", e)

        self._archive_thread.join(timeout=timeout)
        if self._archive_thread.is_alive():
            logger.warning(
                "Log Archiver thread did not stop within timeout (%s seconds).", timeout
            )
        else:
            logger.info("Log Archiver thread stopped.")
        self._archive_thread = None

    def _process_archive_queue(self):
        """Continuously processes file paths from the archive queue."""
        logger.info("Log Archiver thread entering processing loop.")
        while not self._stop_event.is_set():
            try:
                # Use a short timeout for get() to allow checking self._stop_event periodically
                file_path_to_archive = self.archive_queue.get(block=True, timeout=0.5)  # type: ignore

                if file_path_to_archive is QUEUE_SENTINEL:
                    logger.info("Sentinel received, Log Archiver thread exiting.")
                    break  # Exit loop

                if isinstance(file_path_to_archive, str) and os.path.exists(
                    file_path_to_archive
                ):
                    logger.debug("Received file to archive: %s", file_path_to_archive)
                    self._archive_single_file(file_path_to_archive)
                elif isinstance(file_path_to_archive, str):  # Path doesn't exist
                    logger.warning(
                        "Received path for archiving, but file does not exist: %s",
                        file_path_to_archive,
                    )
                else:
                    logger.warning(
                        "Received invalid item from archive queue: %s",
                        type(file_path_to_archive),
                    )

            except queue.Empty:
                # Timeout occurred, loop again to check self._stop_event
                continue
            except Exception as e:
                logger.error(
                    "Error in Log Archiver processing loop: %s", e, exc_info=True
                )
                # Optional: add a small delay to prevent rapid error logging in case of persistent issues
                time.sleep(0.1)

        logger.info("Log Archiver thread exited processing loop.")
        # Attempt to process any remaining items if not stopping abruptly
        if self._stop_event.is_set():  # If stopping, don't try to drain aggressively
            logger.info("Log Archiver stopping, not attempting to drain queue further.")
            return

        logger.info("Log Archiver attempting to drain queue before final exit...")
        try:
            while True:
                file_path_to_archive = self.archive_queue.get_nowait()  # type: ignore
                if file_path_to_archive is QUEUE_SENTINEL:
                    continue  # Ignore further sentinels if any
                if isinstance(file_path_to_archive, str) and os.path.exists(
                    file_path_to_archive
                ):
                    logger.debug("Draining queue: Archiving %s", file_path_to_archive)
                    self._archive_single_file(file_path_to_archive)
                else:
                    logger.warning(
                        "Draining queue: Invalid or non-existent path %s",
                        file_path_to_archive,
                    )
        except queue.Empty:
            logger.info("Archive queue drained.")
        except Exception as e:
            logger.error("Error draining archive queue: %s", e, exc_info=True)

    def _archive_single_file(self, file_path: str):
        """Archives a single specified log file."""
        if not os.path.exists(file_path):
            logger.warning("Cannot archive file, it does not exist: %s", file_path)
            return

        try:
            worker_log_dir = os.path.dirname(file_path)
            # Attempt to extract worker_id from the directory structure or filename
            # This is a heuristic and might need adjustment based on actual log paths
            worker_id_str = (
                "unknown_owner"  # Default for files not clearly tied to a worker
            )
            dir_name = os.path.basename(worker_log_dir)
            if dir_name.startswith("w") and dir_name[1:].isdigit():
                worker_id_str = dir_name  # e.g. "w0", "w12"
            else:  # Try to infer from filename if directory name is not specific
                # Regex to find "-w<digits>_" or "-main_" in the filename for grouping
                # Example: cambia_run_xxxx-w0_001.log -> w0
                # Example: cambia_run_xxxx-main_001.log -> main
                filename_only = os.path.basename(file_path)
                worker_match = re.search(r"-(w\d+)_", filename_only)
                main_match = re.search(r"-(main)_", filename_only)

                if worker_match:
                    worker_id_str = worker_match.group(1)
                elif main_match:
                    worker_id_str = main_match.group(1)
                # If neither, worker_id_str remains "unknown_owner"

            logger.debug(
                "ARCHIVER_EVENT: Archiving file %s for effective owner_id %s",
                file_path,
                worker_id_str,
            )

            # Create a more descriptive base name: log_owner_<original_filename_stem>
            original_filename_stem = os.path.splitext(os.path.basename(file_path))[0]
            archive_base_name = f"log_{worker_id_str}_{original_filename_stem}"

            created_archive_path = self._archive_files_in_dir(
                worker_log_dir,  # The directory where the file resides
                worker_id_str,  # The derived owner ID for managing archive limits
                [file_path],  # List containing only the single file to archive
                archive_base_name,  # The specific base name for this archive
            )

            if created_archive_path:
                logger.info(
                    "ARCHIVER_EVENT: File %s archived to %s. Deleting original.",
                    file_path,
                    created_archive_path,
                )
                try:
                    os.remove(file_path)
                    logger.debug("ARCHIVER_EVENT: Deleted original file %s", file_path)
                except OSError as e:
                    logger.error(
                        "ARCHIVER_EVENT: Error deleting original archived file %s: %s",
                        file_path,
                        e,
                    )
                # Manage archive limits based on the owner and the prefix used for their archives
                self._manage_archive_limit(
                    worker_log_dir,
                    worker_id_str,
                    archive_base_name_prefix=f"log_{worker_id_str}_",
                )
            else:
                logger.warning(
                    "ARCHIVER_EVENT: Archival of %s attempted but _archive_files_in_dir returned None.",
                    file_path,
                )
        except Exception as e:
            logger.error(
                "ARCHIVER_EVENT: Unexpected error archiving file %s: %s",
                file_path,
                e,
                exc_info=True,
            )

    def _get_target_archive_dir(self, log_file_parent_dir: str) -> str:
        """Determines the target directory for storing archives relative to the log file's location."""
        archive_subdir_name = self.config.logging.log_archive_dir
        if not archive_subdir_name:
            return log_file_parent_dir  # Archive in the same directory as the log file
        else:
            # Ensure the archive subdirectory is created relative to the log_file_parent_dir
            # e.g. if log_file_parent_dir is logs/run_xyz/w0, and archive_subdir_name is "archives",
            # target is logs/run_xyz/w0/archives
            return os.path.join(log_file_parent_dir, archive_subdir_name)

    def _archive_files_in_dir(
        self,
        log_file_parent_dir: str,  # The directory containing the log file(s) to be archived
        owner_id_str: str,  # An ID for the owner (e.g. "w0", "main") for archive naming
        files_to_archive: List[str],
        archive_base_name_template: str,  # Base name, e.g., "worker_w0_logs" or "log_w0_specificfile"
    ) -> Optional[str]:
        """
        Creates a tar.gz archive of the given files.
        The archive is placed relative to log_file_parent_dir.
        """
        if not files_to_archive:
            logger.debug(
                "No files to archive for owner %s, base_name %s.",
                owner_id_str,
                archive_base_name_template,
            )
            return None

        target_archive_dir = self._get_target_archive_dir(log_file_parent_dir)
        logger.debug(
            "Target archive directory for owner %s: %s",
            owner_id_str,
            target_archive_dir,
        )
        try:
            os.makedirs(target_archive_dir, exist_ok=True)
        except OSError as e:
            logger.error(
                "Failed to create archive directory %s: %s", target_archive_dir, e
            )
            return None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Use the template for the base name, ensuring uniqueness with timestamp
        archive_filename = f"{archive_base_name_template}_{timestamp}.tar.gz"
        archive_path = os.path.join(target_archive_dir, archive_filename)
        logger.debug("Attempting to create archive: %s", archive_path)

        try:
            with tarfile.open(archive_path, "w:gz") as tar:
                for file_path in files_to_archive:
                    if not os.path.exists(file_path):
                        logger.warning(
                            "File %s not found during tar creation, skipping.", file_path
                        )
                        continue
                    logger.debug(
                        "Adding to archive %s: %s (as %s)",
                        archive_filename,
                        file_path,
                        os.path.basename(file_path),
                    )
                    tar.add(file_path, arcname=os.path.basename(file_path))
            # Verify archive was created and has content before logging success
            if os.path.exists(archive_path) and os.path.getsize(archive_path) > 0:
                logger.info(
                    "Successfully created archive %s for owner %s with %d file(s).",
                    archive_path,
                    owner_id_str,
                    len(files_to_archive),
                )
                return archive_path
            else:  # Archive creation might have failed silently or produced an empty file.
                logger.error(
                    "Archive %s for owner %s reported as created, but is missing or empty.",
                    archive_path,
                    owner_id_str,
                )
                if os.path.exists(archive_path):  # remove empty/failed archive
                    try:
                        os.remove(archive_path)
                    except OSError:
                        pass  # Best effort
                return None
        except (OSError, tarfile.TarError) as e:
            logger.error(
                "Error creating archive %s for owner %s: %s",
                archive_path,
                owner_id_str,
                e,
            )
            if os.path.exists(archive_path):
                try:
                    os.remove(archive_path)
                except OSError:
                    pass  # Best effort
            return None

    def _manage_archive_limit(
        self, log_file_parent_dir: str, owner_id_str: str, archive_base_name_prefix: str
    ):
        """
        Manages the maximum number of archives for a given owner/type in their directory.
        The prefix should correspond to how archives for this owner are named,
        e.g., "log_w0_" for worker 0, or "log_main_" for main process logs.
        """
        max_archives = self.config.logging.log_archive_max_archives
        if max_archives <= 0:  # 0 or negative means unlimited
            logger.debug(
                "Max archives set to %d (<=0), skipping archive limit management for owner %s in %s.",
                max_archives,
                owner_id_str,
                log_file_parent_dir,
            )
            return

        target_archive_dir = self._get_target_archive_dir(log_file_parent_dir)
        if not os.path.isdir(target_archive_dir):
            logger.debug(
                "Target archive dir %s does not exist, no archives to manage for owner %s.",
                target_archive_dir,
                owner_id_str,
            )
            return

        # Pattern should match archives created by _archive_files_in_dir for this owner/prefix
        # Example prefix: "log_w0_", "log_main_"
        archive_pattern_specific = os.path.join(
            target_archive_dir, f"{archive_base_name_prefix}*.tar.gz"
        )
        logger.debug(
            "Globbing for existing archives for owner %s with pattern: %s",
            owner_id_str,
            archive_pattern_specific,
        )
        current_archives = glob.glob(archive_pattern_specific)
        logger.debug(
            "Found %d archives for owner %s matching prefix '%s': %s",
            len(current_archives),
            owner_id_str,
            archive_base_name_prefix,
            [os.path.basename(f) for f in current_archives],
        )

        if len(current_archives) > max_archives:
            current_archives.sort(
                key=os.path.getmtime
            )  # Sort by modification time (oldest first)
            num_to_delete = len(current_archives) - max_archives
            logger.info(
                "Owner %s has %d archives (prefix '%s') in %s (limit %d). Deleting %d oldest.",
                owner_id_str,
                len(current_archives),
                archive_base_name_prefix,
                target_archive_dir,
                max_archives,
                num_to_delete,
            )
            for i in range(num_to_delete):
                try:
                    file_to_delete = current_archives[i]
                    os.remove(file_to_delete)
                    logger.debug(
                        "Removed old archive for owner %s: %s",
                        owner_id_str,
                        file_to_delete,
                    )
                except OSError as e:
                    logger.error(
                        "Error removing old archive %s for owner %s: %s",
                        file_to_delete,
                        owner_id_str,
                        e,
                    )

    # Deprecated: The periodic scan is no longer the primary mechanism.
    def scan_and_archive_worker_logs(self):
        """
        This method is DEPRECATED. Archiving is now event-driven via the queue.
        Calling this method will log a warning and do nothing.
        """
        logger.warning(
            "scan_and_archive_worker_logs is deprecated. Archiving is now event-driven via the archive queue. No action taken."
        )
