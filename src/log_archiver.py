"""
src/log_archiver.py

Manages archiving of worker log files.
"""

import multiprocessing
import os
import glob
import tarfile
import logging
import datetime
import re
import queue
import threading
import time
from typing import List, Optional, Union, Tuple, TYPE_CHECKING

# Use TYPE_CHECKING for conditional import
if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)

# Define a sentinel value for stopping the queue processing
QUEUE_SENTINEL = object()


class LogArchiver:
    """
    Handles archiving of log files triggered via a queue.
    Archives are created in batches when signaled by the SerialRotatingFileHandler.
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
        config: "Config",
        archive_queue: Union[queue.Queue, "multiprocessing.Queue"],
        run_log_dir: str,
    ):
        """
        Initializes the LogArchiver.

        Args:
            config: The main application configuration.
            archive_queue: The queue from which to receive archive trigger messages.
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
        """Continuously processes messages from the archive queue."""
        logger.info("Log Archiver thread entering processing loop.")
        while not self._stop_event.is_set():
            try:
                # Use a short timeout for get() to allow checking self._stop_event periodically
                queue_item = self.archive_queue.get(block=True, timeout=0.5)  # type: ignore

                if queue_item is QUEUE_SENTINEL:
                    logger.info("Sentinel received, Log Archiver thread exiting.")
                    break  # Exit loop

                # Handle BATCH_ARCHIVE trigger
                if (
                    isinstance(queue_item, tuple)
                    and len(queue_item) == 3
                    and queue_item[0] == "BATCH_ARCHIVE"
                ):
                    _, log_dir, base_pattern = queue_item
                    if isinstance(log_dir, str) and isinstance(base_pattern, str):
                        logger.debug(
                            "Received BATCH_ARCHIVE trigger for dir: %s, pattern: %s",
                            log_dir,
                            base_pattern,
                        )
                        self._perform_batch_archive(log_dir, base_pattern)
                    else:
                        logger.warning(
                            "Received invalid BATCH_ARCHIVE data types: %s", queue_item
                        )
                elif isinstance(
                    queue_item, str
                ):  # Handle legacy single-file paths (optional)
                    # logger.warning("Received single file path '%s' for archiving (batching is preferred). Archiving individually.", queue_item)
                    # self._archive_single_file(queue_item) # This function is removed
                    logger.warning(
                        "Received deprecated single file path '%s' on archive queue. Ignoring.",
                        queue_item,
                    )
                else:
                    logger.warning(
                        "Received invalid item from archive queue: %s", type(queue_item)
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
        # No draining attempt here - stop means stop

    def _extract_serial_from_filename(self, filename: str) -> Optional[int]:
        """Extracts the serial number from a log filename."""
        # Regex assumes pattern like _NNN.log at the end
        match = re.search(r"_(\d+)\.log$", filename)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def _get_owner_id_from_base_pattern(self, base_pattern: str) -> str:
        """Extracts owner ID like 'w0' or 'main' from the base pattern."""
        # Example base_pattern: /path/to/logs/run_id/w0/cambia_run_ts-w0
        # Example base_pattern: /path/to/logs/run_id/cambia_run_ts-main
        base_name = os.path.basename(base_pattern)
        worker_match = re.search(r"-(w\d+)$", base_name)
        main_match = re.search(r"-(main)$", base_name)
        if worker_match:
            return worker_match.group(1)
        elif main_match:
            return main_match.group(1)
        else:
            # Fallback: Use the directory name if it looks like a worker ID
            parent_dir_name = os.path.basename(os.path.dirname(base_pattern))
            if parent_dir_name.startswith("w") and parent_dir_name[1:].isdigit():
                return parent_dir_name
            return "unknown_owner"  # Fallback

    def _perform_batch_archive(self, log_dir: str, base_pattern: str):
        """
        Finds all log files matching the pattern, archives them into a single tar.gz
        (named with the serial range), and deletes the originals on success.
        """
        try:
            owner_id = self._get_owner_id_from_base_pattern(base_pattern)
            glob_pattern = f"{base_pattern}_*.log"
            files_to_archive = glob.glob(glob_pattern)

            if not files_to_archive:
                logger.warning(
                    "Batch archive triggered for %s, but no matching files found.",
                    glob_pattern,
                )
                return

            # Sort files by serial number to find the range
            files_with_serials = []
            max_serial_num_for_padding = 0
            for f_path in files_to_archive:
                serial = self._extract_serial_from_filename(os.path.basename(f_path))
                if serial is not None:
                    files_with_serials.append((serial, f_path))
                    max_serial_num_for_padding = max(max_serial_num_for_padding, serial)
                else:
                    logger.warning(
                        "Could not extract serial number from file %s, skipping for naming/archiving.",
                        f_path,
                    )

            if not files_with_serials:
                logger.error(
                    "Could not extract serial numbers from any found files for %s. Cannot perform batch archive.",
                    glob_pattern,
                )
                return

            files_with_serials.sort()
            oldest_serial, oldest_file_path = files_with_serials[0]
            latest_serial, _ = files_with_serials[-1]
            actual_files_to_archive_paths = [f[1] for f in files_with_serials]

            # Determine padding based on max serial number OR backupCount, whichever is larger
            backup_count = getattr(self.config.logging, "log_backup_count", 0)
            pad_length = max(len(str(max_serial_num_for_padding)), len(str(backup_count)))

            # Format the range string with padding
            range_string = (
                f"{oldest_serial:0{pad_length}d}-{latest_serial:0{pad_length}d}"
            )

            # Construct archive name based on owner and range
            archive_base_name = f"log_{owner_id}_{range_string}"
            logger.info(
                "Performing batch archive for owner '%s'. Found %d files (Serials: %s). Archive base: %s",
                owner_id,
                len(actual_files_to_archive_paths),
                range_string,
                archive_base_name,
            )

            created_archive_path = self._archive_files_in_dir(
                log_dir,  # Pass the directory where logs reside
                owner_id,
                actual_files_to_archive_paths,
                archive_base_name,  # Pass the new base name including the range
            )

            if created_archive_path:
                logger.info(
                    "Batch archive %s created successfully. Deleting %d original files...",
                    created_archive_path,
                    len(actual_files_to_archive_paths),
                )
                deletion_errors = 0
                for file_path in actual_files_to_archive_paths:
                    try:
                        os.remove(file_path)
                        logger.debug("Deleted original log file: %s", file_path)
                    except OSError as e:
                        deletion_errors += 1
                        logger.error(
                            "Error deleting original archived file %s: %s", file_path, e
                        )
                if deletion_errors > 0:
                    logger.warning(
                        "Finished batch archive, but %d errors occurred during deletion of originals.",
                        deletion_errors,
                    )
                else:
                    logger.info("Successfully deleted all original log files.")

                # Manage overall archive limit for this owner
                self._manage_archive_limit(
                    log_dir, owner_id, archive_base_name_prefix=f"log_{owner_id}_"
                )
            else:
                logger.error(
                    "Batch archive creation failed for owner %s, pattern %s. Original files were NOT deleted.",
                    owner_id,
                    base_pattern,
                )

        except Exception as e:
            logger.error(
                "Unexpected error performing batch archive for pattern %s: %s",
                base_pattern,
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
        archive_base_name_template: str,  # Base name, e.g., "log_w0_001-005"
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
        # Use the template for the base name (now including range), ensuring uniqueness with timestamp
        archive_filename = f"{archive_base_name_template}_{timestamp}.tar.gz"
        archive_path = os.path.join(target_archive_dir, archive_filename)
        logger.debug("Attempting to create archive: %s", archive_path)

        files_added_count = 0
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
                    files_added_count += 1
            # Verify archive was created and has content before logging success
            if (
                os.path.exists(archive_path)
                and os.path.getsize(archive_path) > 0
                and files_added_count > 0
            ):
                logger.info(
                    "Successfully created archive %s for owner %s with %d file(s).",
                    archive_path,
                    owner_id_str,
                    files_added_count,
                )
                return archive_path
            else:  # Archive creation might have failed silently or produced an empty file.
                logger.error(
                    "Archive %s for owner %s reported as created, but is missing, empty, or no files were added.",
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

    def get_total_log_size_info(self) -> Tuple[int, int]:
        """
        Calculates the total size of current (active/backup) log files and archived log files.
        Returns:
            Tuple[int, int]: (total_current_log_size_bytes, total_archived_size_bytes)
        """
        total_current_log_bytes = 0
        total_archived_bytes = 0

        if not self.run_log_dir or not os.path.isdir(self.run_log_dir):
            logger.warning(
                "LogArchiver: run_log_dir not set or invalid, cannot calculate log sizes."
            )
            return 0, 0

        try:
            # Scan for all .log files (current and uncompressed backups)
            # This includes the main log directory and worker subdirectories (w0, w1, etc.)
            for dirpath, dirnames, filenames in os.walk(self.run_log_dir, topdown=True):
                # If an archive subdirectory is configured, skip scanning .log files within it
                archive_subdir_name = self.config.logging.log_archive_dir
                if archive_subdir_name and archive_subdir_name in dirnames:
                    # Remove the archive subdir from dirnames to prevent os.walk from descending into it for this loop pass
                    # This ensures we don't count .log files if they somehow end up in an archive dir.
                    # And also prevents the .tar.gz scan below from running inside this .log scan.
                    dirnames[:] = [d for d in dirnames if d != archive_subdir_name]

                for filename in filenames:
                    if filename.endswith(".log"):
                        try:
                            full_path = os.path.join(dirpath, filename)
                            total_current_log_bytes += os.path.getsize(full_path)
                        except OSError:
                            # This might happen if a file is deleted between os.walk and os.path.getsize
                            logger.debug(
                                "Could not get size for current log file (may have been removed): %s",
                                full_path,
                            )

            # Scan for .tar.gz files in archive directories
            # This needs to specifically look inside designated archive subdirectories if configured,
            # or in the log owner's main directory if archive_subdir_name is empty.

            archive_subdir_name = self.config.logging.log_archive_dir

            # Iterate through potential log owner directories (main run_log_dir and worker dirs)
            potential_log_owner_dirs = [self.run_log_dir]  # For main logs
            for item in os.listdir(self.run_log_dir):
                item_path = os.path.join(self.run_log_dir, item)
                if (
                    os.path.isdir(item_path)
                    and item.startswith("w")
                    and item[1:].isdigit()
                ):
                    potential_log_owner_dirs.append(item_path)

            for owner_dir in potential_log_owner_dirs:
                # Determine the actual directory where archives for this owner would be stored
                actual_archive_scan_dir = owner_dir
                if archive_subdir_name:  # If archives are in a specific subdirectory
                    actual_archive_scan_dir = os.path.join(owner_dir, archive_subdir_name)

                if os.path.isdir(actual_archive_scan_dir):
                    for filename in os.listdir(actual_archive_scan_dir):
                        if filename.endswith(".tar.gz"):
                            try:
                                full_path = os.path.join(
                                    actual_archive_scan_dir, filename
                                )
                                total_archived_bytes += os.path.getsize(full_path)
                            except OSError:
                                logger.debug(
                                    "Could not get size for archive file (may have been removed): %s",
                                    full_path,
                                )
        except Exception as e:
            logger.error("Error calculating total log sizes: %s", e, exc_info=True)

        return total_current_log_bytes, total_archived_bytes
