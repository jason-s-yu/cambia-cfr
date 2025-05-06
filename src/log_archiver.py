# src/log_archiver.py
"""Manages archiving of worker log files."""

import os
import glob
import tarfile
import logging
import datetime
import re
from typing import List, Optional

from .config import Config  # Relative import for Config

logger = logging.getLogger(__name__)


class LogArchiver:
    """
    Handles scanning, compressing, and archiving worker log files.
    Archives are stored either directly in the worker's log directory or
    in a specified subdirectory within it.
    """

    WORKER_DIR_PATTERN = (
        "w*"  # Pattern to identify worker directories (e.g., w0, w1, ...)
    )
    CURRENT_LOG_MARKER_FILENAME = ".current_log"

    def __init__(self, config: Config, run_log_dir: str):
        """
        Initializes the LogArchiver.

        Args:
            config: The main application configuration.
            run_log_dir: The absolute path to the current run's log directory.
        """
        self.config = config
        self.run_log_dir = run_log_dir
        # Archive storage directory is now determined per worker later
        if self.config.logging.log_archive_enabled:
            logger.info("Log archiver initialized (enabled).")

    def _get_target_archive_dir(self, worker_log_dir: str) -> str:
        """Determines the target directory for storing archives for a worker."""
        archive_subdir_name = self.config.logging.log_archive_dir
        if not archive_subdir_name:  # Empty or None means store in worker dir itself
            return worker_log_dir
        else:
            # Store in a subdirectory named by log_archive_dir inside the worker dir
            return os.path.join(worker_log_dir, archive_subdir_name)

    def _get_current_log_file(self, worker_log_dir: str) -> Optional[str]:
        """Reads the .current_log marker file to find the active log file."""
        marker_path = os.path.join(worker_log_dir, self.CURRENT_LOG_MARKER_FILENAME)
        if os.path.exists(marker_path):
            try:
                with open(marker_path, "r", encoding="utf-8") as f:
                    current_log_path = f.read().strip()
                if os.path.isabs(current_log_path) and os.path.exists(current_log_path):
                    return current_log_path
                elif not os.path.isabs(current_log_path):
                    abs_path = os.path.join(worker_log_dir, current_log_path)
                    if os.path.exists(abs_path):
                        return abs_path
                logger.warning(
                    "Current log marker at '%s' points to non-existent file: %s",
                    marker_path,
                    current_log_path,
                )
            except OSError as e:
                logger.error(
                    "Error reading current log marker file %s: %s", marker_path, e
                )
        return None

    def _get_rotated_log_files(
        self, worker_log_dir: str, current_log_file_abs: Optional[str]
    ) -> List[str]:
        """
        Gets all .log files in the worker directory, excluding the current one.
        Returns a list of absolute paths to log files ready for potential archiving.
        """
        rotated_logs: List[str] = []
        all_log_files = glob.glob(os.path.join(worker_log_dir, "*.log"))

        for log_file_abs in all_log_files:
            if current_log_file_abs and os.path.samefile(
                log_file_abs, current_log_file_abs
            ):
                continue
            rotated_logs.append(log_file_abs)

        def get_serial_from_filename(f_path):
            match = re.search(r"_(\d+)\.log$", os.path.basename(f_path))
            return int(match.group(1)) if match else -1

        rotated_logs.sort(key=get_serial_from_filename)
        return rotated_logs

    def _archive_files(
        self,
        worker_log_dir: str,  # Pass worker dir to determine target
        worker_id_str: str,
        files_to_archive: List[str],
        archive_base_name: str,
    ) -> Optional[str]:
        """
        Creates a tar.gz archive of the given files in the correct location.

        Args:
            worker_log_dir: Absolute path to the worker's log directory.
            worker_id_str: The string identifier of the worker (e.g., "w0").
            files_to_archive: List of absolute paths to files to include in the archive.
            archive_base_name: Base name for the archive file (timestamp will be added).

        Returns:
            Path to the created archive file, or None on failure.
        """
        if not files_to_archive:
            return None

        target_archive_dir = self._get_target_archive_dir(worker_log_dir)
        try:
            os.makedirs(target_archive_dir, exist_ok=True)
        except OSError as e:
            logger.error(
                "Failed to create archive directory %s: %s", target_archive_dir, e
            )
            return None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        archive_filename = f"{archive_base_name}_{timestamp}.tar.gz"
        archive_path = os.path.join(target_archive_dir, archive_filename)

        try:
            with tarfile.open(archive_path, "w:gz") as tar:
                for file_path in files_to_archive:
                    tar.add(file_path, arcname=os.path.basename(file_path))
            logger.info(
                "Successfully created archive %s for worker %s with %d files.",
                archive_path,
                worker_id_str,
                len(files_to_archive),
            )
            return archive_path
        except (OSError, tarfile.TarError) as e:
            logger.error(
                "Error creating archive %s for worker %s: %s",
                archive_path,
                worker_id_str,
                e,
            )
            if os.path.exists(archive_path):
                try:
                    os.remove(archive_path)
                except OSError:
                    pass
            return None

    def _manage_archive_limit(self, worker_log_dir: str, worker_id_str: str):
        """
        Manages the maximum number of archives for a given worker in its target dir.
        Deletes the oldest archives if the limit is exceeded.
        Values <= 0 in the config mean unlimited archives.
        """
        max_archives = self.config.logging.log_archive_max_archives
        if max_archives <= 0:
            logger.debug(
                "Max archives set to %d, skipping deletion for worker %s.",
                max_archives,
                worker_id_str,
            )
            return

        target_archive_dir = self._get_target_archive_dir(worker_log_dir)
        if not os.path.isdir(target_archive_dir):
            # If target dir doesn't exist, there are no archives to manage
            return

        # Glob for archives within the target directory
        # The pattern needs to be generic enough as base name varies slightly
        archive_pattern = os.path.join(target_archive_dir, "*.tar.gz")
        worker_archives = glob.glob(archive_pattern)

        # Filter further based on a naming convention if needed, e.g., containing worker_id_str
        # For simplicity, let's assume all tar.gz in the target dir belong to this worker limit for now
        # To be more robust, archives created by _archive_files should follow a strict pattern.
        # Let's use the pattern from _archive_files' base name construction:
        # Simplified: assume filenames include `worker_{id_str}_logs_`
        archive_pattern_specific = os.path.join(
            target_archive_dir, f"*worker_{worker_id_str}_logs_*.tar.gz"
        )
        worker_archives = glob.glob(archive_pattern_specific)

        if len(worker_archives) > max_archives:
            worker_archives.sort(key=os.path.getmtime)
            num_to_delete = len(worker_archives) - max_archives
            logger.info(
                "Worker %s has %d archives in %s (limit %d). Deleting %d oldest.",
                worker_id_str,
                len(worker_archives),
                target_archive_dir,
                max_archives,
                num_to_delete,
            )
            for i in range(num_to_delete):
                try:
                    os.remove(worker_archives[i])
                    logger.debug(
                        "Removed old archive for worker %s: %s",
                        worker_id_str,
                        worker_archives[i],
                    )
                except OSError as e:
                    logger.error(
                        "Error removing old archive %s for worker %s: %s",
                        worker_archives[i],
                        worker_id_str,
                        e,
                    )

    def scan_and_archive_worker_logs(self):
        """
        Scans all worker log directories and archives logs if conditions are met.
        """
        if not self.config.logging.log_archive_enabled:
            return

        logger.debug("Starting scan for worker logs to archive in %s", self.run_log_dir)
        try:
            worker_dirs = glob.glob(
                os.path.join(self.run_log_dir, self.WORKER_DIR_PATTERN)
            )
        except Exception as e:
            logger.error(
                "Error globbing for worker directories in %s: %s", self.run_log_dir, e
            )
            return

        for worker_log_dir_abs in worker_dirs:
            if not os.path.isdir(worker_log_dir_abs):
                continue

            worker_id_str = os.path.basename(worker_log_dir_abs)
            logger.debug("Processing worker log directory: %s", worker_log_dir_abs)

            try:
                current_log_file_abs = self._get_current_log_file(worker_log_dir_abs)
                if not current_log_file_abs:
                    logger.warning(
                        "Could not determine current log file for worker %s in %s. Skipping archiving for this worker.",
                        worker_id_str,
                        worker_log_dir_abs,
                    )
                    continue

                rotated_log_files = self._get_rotated_log_files(
                    worker_log_dir_abs, current_log_file_abs
                )
                if not rotated_log_files:
                    logger.debug(
                        "No rotated log files to consider for worker %s.", worker_id_str
                    )
                    continue

                total_size_rotated = sum(
                    os.path.getsize(f) for f in rotated_log_files if os.path.exists(f)
                )
                logger.debug(
                    "Worker %s: %d rotated log files, total size: %s bytes.",
                    worker_id_str,
                    len(rotated_log_files),
                    total_size_rotated,
                )

                if total_size_rotated >= self.config.logging.log_compress_after_bytes:
                    logger.info(
                        "Worker %s: Rotated logs size (%d bytes) exceeds threshold (%d bytes). Attempting to archive.",
                        worker_id_str,
                        total_size_rotated,
                        self.config.logging.log_compress_after_bytes,
                    )

                    # Simplified base name
                    archive_base_name = f"worker_{worker_id_str}_logs"

                    created_archive_path = self._archive_files(
                        worker_log_dir_abs,  # Pass worker dir
                        worker_id_str,
                        rotated_log_files,
                        archive_base_name,
                    )

                    if created_archive_path:
                        logger.info(
                            "Archive created for worker %s: %s. Deleting original rotated files.",
                            worker_id_str,
                            created_archive_path,
                        )
                        files_deleted_count = 0
                        for f_path in rotated_log_files:
                            try:
                                os.remove(f_path)
                                files_deleted_count += 1
                            except OSError as e:
                                logger.error(
                                    "Error deleting archived log file %s for worker %s: %s",
                                    f_path,
                                    worker_id_str,
                                    e,
                                )
                        logger.debug(
                            "Deleted %d original rotated log files for worker %s.",
                            files_deleted_count,
                            worker_id_str,
                        )

                        # Manage limit within the correct target dir
                        self._manage_archive_limit(worker_log_dir_abs, worker_id_str)
                else:
                    logger.debug(
                        "Worker %s: Rotated logs size (%d bytes) below archive threshold (%d bytes).",
                        worker_id_str,
                        total_size_rotated,
                        self.config.logging.log_compress_after_bytes,
                    )
            except Exception as e:
                logger.error(
                    "Unexpected error processing worker directory %s: %s",
                    worker_log_dir_abs,
                    e,
                    exc_info=True,
                )
                continue  # Continue to the next worker directory
