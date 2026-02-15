"""
src/log_archiver.py

This module defines the LogArchiver, a background process responsible for managing log files.
It periodically scans for completed log files, compresses them into .tar.gz archives,
and cleans up old archives to ensure disk space usage remains within configured limits.
"""

import logging
import multiprocessing
import os
import shutil
import tarfile
import time
import traceback
from pathlib import Path
from typing import List

from .config import Config
from .utils import parse_size_to_bytes

# Use a dedicated logger for the archiver itself
logger = logging.getLogger(__name__)


class LogArchiver:
    """
    Manages log file archiving and cleanup in a separate process.
    """

    def __init__(self, config: Config, shutdown_event: multiprocessing.Event):
        """
        Initializes the LogArchiver.

        Args:
            config: The main configuration object.
            shutdown_event: A multiprocessing event to signal shutdown.
        """
        self.config = config.archiver
        self.log_dir = Path(config.logging.directory)
        self.log_file_basename = config.logging.filename.split(".")[0]
        self.shutdown_event = shutdown_event
        self.archive_interval_seconds = self.config.archive_interval_seconds
        self.archive_delay_seconds = self.config.archive_delay_seconds
        self.max_archive_bytes = parse_size_to_bytes(self.config.max_archive_size)

    def run(self):
        """Main loop for the archiver process."""
        logger.info(
            "LogArchiver process started. PID: %d. Watching directory: %s",
            os.getpid(),
            self.log_dir,
        )
        while not self.shutdown_event.is_set():
            try:
                self._archive_and_clean()
            except Exception as e:
                logger.error(
                    "LogArchiver: Unhandled error in main loop: %s\n%s",
                    e,
                    traceback.format_exc(),
                )

            # Wait for the next interval or until shutdown is triggered
            self.shutdown_event.wait(self.archive_interval_seconds)

        logger.info("LogArchiver process shutting down.")

    def _archive_and_clean(self):
        """Core logic to find, archive, and clean log files."""
        logger.debug("LogArchiver: Starting new archive and clean cycle.")

        # --- Step 1: Find and safely claim files to be archived ---
        files_to_archive = self._claim_files_for_archiving()
        if not files_to_archive:
            logger.debug("LogArchiver: No new log files to archive.")
        else:
            logger.info(
                "LogArchiver: Found %d log file(s) to archive.", len(files_to_archive)
            )

        # --- Step 2: Archive each claimed file ---
        for original_path in files_to_archive:
            try:
                # The file has already been renamed to .archiving, so it's safe
                self._create_archive(original_path)
                # After successful archival, the original (.log.N) is deleted
                # by the _create_archive method.
            except Exception as e:
                logger.error(
                    "LogArchiver: Failed to create archive for %s: %s",
                    original_path.name,
                    e,
                    exc_info=True,
                )
                # If archival fails, rename it back so we can try again later
                try:
                    archiving_path = original_path.with_suffix(
                        original_path.suffix + ".archiving"
                    )
                    if archiving_path.exists():
                        archiving_path.rename(original_path)
                        logger.warning(
                            "LogArchiver: Rolled back failed archive for %s.",
                            original_path.name,
                        )
                except Exception as rename_err:
                    logger.error(
                        "LogArchiver: CRITICAL - Failed to roll back renaming of %s. Manual intervention may be needed. Error: %s",
                        archiving_path.name,
                        rename_err,
                    )

        # --- Step 3: Clean up old archives if total size exceeds the limit ---
        if self.max_archive_bytes > 0:
            self._cleanup_old_archives()

        logger.debug("LogArchiver: Archive and clean cycle finished.")

    def _claim_files_for_archiving(self) -> List[Path]:
        """
        Identifies completed log files and claims them using an atomic rename.
        This is the corrected, race-condition-free method.

        Returns:
            A list of Path objects for the original file paths that were successfully claimed.
        """
        claimed_files = []
        try:
            candidate_files = [
                p
                for p in self.log_dir.iterdir()
                if p.is_file()
                and p.name.startswith(self.log_file_basename)
                and not p.name.endswith((".gz", ".archiving", ".lock"))
            ]
        except FileNotFoundError:
            logger.warning(
                "LogArchiver: Log directory not found: %s. Skipping scan.", self.log_dir
            )
            return []

        for path in sorted(candidate_files):
            # Ignore the primary, active log file
            if path.name == self.log_file_basename:
                continue

            try:
                # Check if the file is old enough to be considered for archiving
                mtime = path.stat().st_mtime
                if (time.time() - mtime) < self.archive_delay_seconds:
                    continue  # Not old enough yet

                # **ATOMIC LOCKING STEP**
                # Try to rename the file. If this succeeds, we have "claimed" it.
                archiving_path = path.with_suffix(path.suffix + ".archiving")
                path.rename(archiving_path)

                logger.info("LogArchiver: Claimed %s for archival.", path.name)
                claimed_files.append(path)  # Return the original path for reference

            except FileNotFoundError:
                # File was removed by another process between listing and renaming, which is fine.
                continue
            except Exception as e:
                logger.error(
                    "LogArchiver: Error while trying to claim file %s: %s", path.name, e
                )

        return claimed_files

    def _create_archive(self, original_path: Path):
        """
        Compresses a claimed log file and deletes the original.
        The input file is expected to have been renamed to .archiving already.
        """
        archiving_path = original_path.with_suffix(original_path.suffix + ".archiving")
        archive_path = original_path.with_suffix(original_path.suffix + ".tar.gz")

        if not archiving_path.exists():
            logger.warning(
                "LogArchiver: File %s disappeared before it could be archived.",
                archiving_path.name,
            )
            return

        logger.debug(
            "LogArchiver: Compressing %s to %s", archiving_path.name, archive_path.name
        )

        try:
            # Create a tar.gz file
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(
                    archiving_path, arcname=original_path.name
                )  # Use original name in archive

            # On success, remove the .archiving file
            archiving_path.unlink()
            logger.info("LogArchiver: Successfully archived %s.", original_path.name)
        except Exception as e:
            logger.error(
                "LogArchiver: Failed during tarfile creation for %s: %s",
                archiving_path.name,
                e,
            )
            # Re-raise the exception to be handled by the calling loop
            raise

    def _cleanup_old_archives(self):
        """Deletes the oldest archives if the total size exceeds the configured limit."""
        try:
            archives = [
                p
                for p in self.log_dir.glob(f"{self.log_file_basename}*.tar.gz")
                if p.is_file()
            ]

            if not archives:
                return

            # Sort archives by modification time, oldest first
            archives.sort(key=lambda p: p.stat().st_mtime)

            total_size = sum(p.stat().st_size for p in archives)

            if total_size > self.max_archive_bytes:
                logger.info(
                    "LogArchiver: Total archive size (%.2f GB) exceeds limit (%.2f GB). Cleaning up...",
                    total_size / (1024**3),
                    self.max_archive_bytes / (1024**3),
                )

            while total_size > self.max_archive_bytes and archives:
                oldest_archive = archives.pop(0)
                try:
                    size_to_free = oldest_archive.stat().st_size
                    logger.warning(
                        "LogArchiver: Deleting old archive %s to free up space.",
                        oldest_archive.name,
                    )
                    oldest_archive.unlink()
                    total_size -= size_to_free
                except FileNotFoundError:
                    # It was already deleted, which is fine.
                    continue
                except Exception as e:
                    logger.error(
                        "LogArchiver: Could not delete old archive %s: %s",
                        oldest_archive.name,
                        e,
                    )
                    # Stop trying to clean this cycle to avoid a loop of failures
                    break

        except Exception as e:
            logger.error("LogArchiver: Error during cleanup of old archives: %s", e)


def start_log_archiver_process(
    config: Config, shutdown_event: multiprocessing.Event
) -> multiprocessing.Process:
    """Creates and starts the LogArchiver process."""
    archiver = LogArchiver(config, shutdown_event)
    process = multiprocessing.Process(
        target=archiver.run, name="LogArchiver", daemon=True
    )
    process.start()
    return process
