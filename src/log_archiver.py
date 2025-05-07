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
        if self.config.logging.log_archive_enabled:
            logger.info(
                "Log archiver initialized (enabled). Run log dir: %s", self.run_log_dir
            )
            logger.debug(
                "Archiver Config: MaxArchives=%d, CompressAfterBytes=%d, ArchiveSubDir='%s'",
                self.config.logging.log_archive_max_archives,
                self.config.logging.log_compress_after_bytes,
                self.config.logging.log_archive_dir,
            )

    def _get_target_archive_dir(self, worker_log_dir: str) -> str:
        """Determines the target directory for storing archives for a worker."""
        archive_subdir_name = self.config.logging.log_archive_dir
        if not archive_subdir_name:
            return worker_log_dir
        else:
            return os.path.join(worker_log_dir, archive_subdir_name)

    def _get_current_log_file(self, worker_log_dir: str) -> Optional[str]:
        """Reads the .current_log marker file to find the active log file."""
        marker_path = os.path.join(worker_log_dir, self.CURRENT_LOG_MARKER_FILENAME)
        logger.debug("Attempting to read current log marker: %s", marker_path)
        if os.path.exists(marker_path):
            try:
                with open(marker_path, "r", encoding="utf-8") as f:
                    current_log_path_str = f.read().strip()
                logger.debug(
                    "Read from marker '%s': '%s'", marker_path, current_log_path_str
                )

                # Check if it's an absolute path first
                if os.path.isabs(current_log_path_str):
                    if os.path.exists(current_log_path_str):
                        logger.debug(
                            "Current log (absolute from marker) exists: %s",
                            current_log_path_str,
                        )
                        return current_log_path_str
                    else:
                        logger.warning(
                            "Marker path '%s' is absolute but file does not exist: %s",
                            marker_path,
                            current_log_path_str,
                        )
                        return None  # Path in marker is absolute but doesn't exist

                # If not absolute, assume it's relative to worker_log_dir
                abs_path_from_relative = os.path.join(
                    worker_log_dir, current_log_path_str
                )
                if os.path.exists(abs_path_from_relative):
                    logger.debug(
                        "Current log (relative from marker, resolved to absolute) exists: %s",
                        abs_path_from_relative,
                    )
                    return abs_path_from_relative
                else:
                    logger.warning(
                        "Marker path '%s' (relative) resolved to '%s' which does not exist.",
                        current_log_path_str,
                        abs_path_from_relative,
                    )
                    return None
            except OSError as e:
                logger.error(
                    "Error reading current log marker file %s: %s", marker_path, e
                )
        else:
            logger.debug("Current log marker file not found: %s", marker_path)
        return None

    def _get_rotated_log_files(
        self, worker_log_dir: str, current_log_file_abs: Optional[str]
    ) -> List[str]:
        """
        Gets all .log files in the worker directory, excluding the current one.
        Returns a list of absolute paths to log files ready for potential archiving.
        """
        rotated_logs: List[str] = []
        glob_pattern = os.path.join(worker_log_dir, "*.log")
        logger.debug("Globbing for log files with pattern: %s", glob_pattern)
        all_log_files = glob.glob(glob_pattern)
        logger.debug(
            "Found %d total .log files: %s",
            len(all_log_files),
            [os.path.basename(f) for f in all_log_files],
        )

        if current_log_file_abs:
            logger.debug(
                "Current active log file identified as: %s",
                os.path.basename(current_log_file_abs),
            )
        else:
            logger.debug(
                "No current active log file identified. All .log files will be considered rotated."
            )

        for log_file_abs in all_log_files:
            if current_log_file_abs and os.path.samefile(
                log_file_abs, current_log_file_abs
            ):
                logger.debug(
                    "Excluding active log file from rotated list: %s",
                    os.path.basename(log_file_abs),
                )
                continue
            rotated_logs.append(log_file_abs)

        def get_serial_from_filename(f_path):
            match = re.search(r"_(\d+)\.log$", os.path.basename(f_path))
            return int(match.group(1)) if match else -1

        rotated_logs.sort(key=get_serial_from_filename)
        logger.debug(
            "Identified %d rotated log files (sorted): %s",
            len(rotated_logs),
            [os.path.basename(f) for f in rotated_logs],
        )
        return rotated_logs

    def _archive_files(
        self,
        worker_log_dir: str,
        worker_id_str: str,
        files_to_archive: List[str],
        archive_base_name: str,
    ) -> Optional[str]:
        """
        Creates a tar.gz archive of the given files in the correct location.
        """
        if not files_to_archive:
            logger.debug(
                "No files to archive for worker %s, base_name %s.",
                worker_id_str,
                archive_base_name,
            )
            return None

        target_archive_dir = self._get_target_archive_dir(worker_log_dir)
        logger.debug(
            "Target archive directory for worker %s: %s",
            worker_id_str,
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
        archive_filename = f"{archive_base_name}_{timestamp}.tar.gz"
        archive_path = os.path.join(target_archive_dir, archive_filename)
        logger.debug("Attempting to create archive: %s", archive_path)

        try:
            with tarfile.open(archive_path, "w:gz") as tar:
                for file_path in files_to_archive:
                    logger.debug(
                        "Adding to archive %s: %s (as %s)",
                        archive_filename,
                        file_path,
                        os.path.basename(file_path),
                    )
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
        Manages the maximum number of archives for a given worker.
        """
        max_archives = self.config.logging.log_archive_max_archives
        if max_archives <= 0:
            logger.debug(
                "Max archives set to %d (<=0), skipping archive limit management for worker %s.",
                max_archives,
                worker_id_str,
            )
            return

        target_archive_dir = self._get_target_archive_dir(worker_log_dir)
        if not os.path.isdir(target_archive_dir):
            logger.debug(
                "Target archive dir %s does not exist, no archives to manage for worker %s.",
                target_archive_dir,
                worker_id_str,
            )
            return

        archive_pattern_specific = os.path.join(
            target_archive_dir, f"*worker_{worker_id_str}_logs_*.tar.gz"
        )
        logger.debug(
            "Globbing for existing archives for worker %s with pattern: %s",
            worker_id_str,
            archive_pattern_specific,
        )
        worker_archives = glob.glob(archive_pattern_specific)
        logger.debug(
            "Found %d archives for worker %s: %s",
            len(worker_archives),
            worker_id_str,
            [os.path.basename(f) for f in worker_archives],
        )

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
                    file_to_delete = worker_archives[i]
                    os.remove(file_to_delete)
                    logger.debug(
                        "Removed old archive for worker %s: %s",
                        worker_id_str,
                        file_to_delete,
                    )
                except OSError as e:
                    logger.error(
                        "Error removing old archive %s for worker %s: %s",
                        file_to_delete,
                        worker_id_str,
                        e,
                    )

    def scan_and_archive_worker_logs(self):
        """
        Scans all worker log directories and archives logs if conditions are met.
        """
        if not self.config.logging.log_archive_enabled:
            logger.debug("Log archiving is disabled. Skipping scan.")
            return

        logger.debug(
            "ARCHIVER_SCAN: Starting scan for worker logs to archive in %s",
            self.run_log_dir,
        )
        try:
            worker_dirs_pattern = os.path.join(self.run_log_dir, self.WORKER_DIR_PATTERN)
            logger.debug(
                "ARCHIVER_SCAN: Globbing for worker dirs with pattern: %s",
                worker_dirs_pattern,
            )
            worker_dirs = glob.glob(worker_dirs_pattern)
            logger.debug("ARCHIVER_SCAN: Found worker_dirs: %s", worker_dirs)
        except Exception as e:
            logger.error(
                "ARCHIVER_SCAN: Error globbing for worker directories in %s: %s",
                self.run_log_dir,
                e,
            )
            return

        for worker_log_dir_abs in worker_dirs:
            if not os.path.isdir(worker_log_dir_abs):
                logger.debug(
                    "ARCHIVER_SCAN: %s is not a directory, skipping.", worker_log_dir_abs
                )
                continue

            worker_id_str = os.path.basename(worker_log_dir_abs)
            logger.debug(
                "ARCHIVER_SCAN: Processing worker log directory: %s (Worker ID: %s)",
                worker_log_dir_abs,
                worker_id_str,
            )

            try:
                current_log_file_abs = self._get_current_log_file(worker_log_dir_abs)
                if not current_log_file_abs:
                    logger.warning(
                        "ARCHIVER_SCAN: Worker %s: Could not determine current log file in %s. Skipping archiving for this worker.",
                        worker_id_str,
                        worker_log_dir_abs,
                    )
                    continue
                logger.debug(
                    "ARCHIVER_SCAN: Worker %s: Current log file is %s",
                    worker_id_str,
                    os.path.basename(current_log_file_abs),
                )

                rotated_log_files = self._get_rotated_log_files(
                    worker_log_dir_abs, current_log_file_abs
                )
                if not rotated_log_files:
                    logger.debug(
                        "ARCHIVER_SCAN: Worker %s: No rotated log files to consider.",
                        worker_id_str,
                    )
                    continue
                logger.debug(
                    "ARCHIVER_SCAN: Worker %s: Found rotated files: %s",
                    worker_id_str,
                    [os.path.basename(f) for f in rotated_log_files],
                )

                total_size_rotated = sum(
                    os.path.getsize(f) for f in rotated_log_files if os.path.exists(f)
                )
                compress_threshold = self.config.logging.log_compress_after_bytes
                logger.debug(
                    "ARCHIVER_SCAN: Worker %s: %d rotated log files, total size: %d bytes. Threshold: %d bytes.",
                    worker_id_str,
                    len(rotated_log_files),
                    total_size_rotated,
                    compress_threshold,
                )

                if total_size_rotated >= compress_threshold:
                    logger.info(
                        "ARCHIVER_SCAN: Worker %s: Rotated logs size (%d bytes) meets/exceeds threshold (%d bytes). Attempting to archive.",
                        worker_id_str,
                        total_size_rotated,
                        compress_threshold,
                    )

                    archive_base_name = f"worker_{worker_id_str}_logs"
                    logger.debug(
                        "ARCHIVER_SCAN: Worker %s: Archive base name: %s",
                        worker_id_str,
                        archive_base_name,
                    )

                    created_archive_path = self._archive_files(
                        worker_log_dir_abs,
                        worker_id_str,
                        rotated_log_files,  # Archive all found rotated files if threshold met
                        archive_base_name,
                    )

                    if created_archive_path:
                        logger.info(
                            "ARCHIVER_SCAN: Worker %s: Archive created: %s. Deleting original rotated files.",
                            worker_id_str,
                            created_archive_path,
                        )
                        files_deleted_count = 0
                        for f_path in rotated_log_files:
                            try:
                                os.remove(f_path)
                                files_deleted_count += 1
                                logger.debug(
                                    "ARCHIVER_SCAN: Worker %s: Deleted original file %s",
                                    worker_id_str,
                                    os.path.basename(f_path),
                                )
                            except OSError as e:
                                logger.error(
                                    "ARCHIVER_SCAN: Worker %s: Error deleting archived log file %s: %s",
                                    worker_id_str,
                                    f_path,
                                    e,
                                )
                        logger.debug(
                            "ARCHIVER_SCAN: Worker %s: Deleted %d original rotated log files.",
                            worker_id_str,
                            files_deleted_count,
                        )
                        self._manage_archive_limit(worker_log_dir_abs, worker_id_str)
                    else:
                        logger.warning(
                            "ARCHIVER_SCAN: Worker %s: Archival attempted but _archive_files returned None.",
                            worker_id_str,
                        )
                else:
                    logger.debug(
                        "ARCHIVER_SCAN: Worker %s: Rotated logs size (%d bytes) below archive threshold (%d bytes). No archival action taken.",
                        worker_id_str,
                        total_size_rotated,
                        compress_threshold,
                    )
            except (
                Exception
            ) as e:  # Catch exceptions during processing of a single worker dir
                logger.error(
                    "ARCHIVER_SCAN: Worker %s: Unexpected error processing directory %s: %s",
                    worker_id_str,
                    worker_log_dir_abs,
                    e,
                    exc_info=True,  # Add exc_info for full traceback
                )
                continue  # Continue to the next worker directory
        logger.debug("ARCHIVER_SCAN: Finished scan for worker logs to archive.")
