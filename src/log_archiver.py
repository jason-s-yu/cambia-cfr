"""
src/log_archiver.py

Background thread that processes log archive requests from SerialRotatingFileHandler.

When a rotating handler reaches its backup count, it sends a
("BATCH_ARCHIVE", log_dir, base_pattern) tuple to the archive queue.
This thread consumes those messages, compresses the rotated log files
into tar.gz archives, and manages old archive cleanup.
"""

import glob
import logging
import os
import queue
import tarfile
import threading
import time
from typing import Tuple, Union

from .config import Config

logger = logging.getLogger(__name__)


class LogArchiver(threading.Thread):
    """
    Background thread that compresses rotated log files into tar.gz archives.

    Expected interface (used by main_train.py):
        archiver = LogArchiver(config, archive_queue, run_log_dir)
        archiver.start()
        archiver.run_log_dir = actual_log_dir   # update after logging setup
        current, archived = archiver.get_total_log_size_info()
        archiver.stop(timeout=10.0)
    """

    def __init__(
        self,
        config: Config,
        archive_queue: Union[queue.Queue, "multiprocessing.Queue"],
        run_log_dir: str = "",
    ):
        super().__init__(daemon=True, name="LogArchiver")
        self.config = config
        self.logging_config = config.logging
        self.archive_queue = archive_queue
        self.run_log_dir = run_log_dir
        self._stop_event = threading.Event()

        # Config-driven settings
        self.max_archives = getattr(self.logging_config, "log_archive_max_archives", 10)
        self.archive_subdir = getattr(self.logging_config, "log_archive_dir", "")

        # Cached size info for display
        self._current_log_bytes: int = 0
        self._archived_bytes: int = 0
        self._size_lock = threading.Lock()

    def stop(self, timeout: float = 10.0):
        """Signal the thread to stop and wait for it to finish."""
        self._stop_event.set()
        self.join(timeout=timeout)
        if self.is_alive():
            logger.warning("LogArchiver thread did not stop within %.1fs.", timeout)

    def get_total_log_size_info(self) -> Tuple[int, int]:
        """
        Returns (current_log_bytes, archived_bytes).

        Scans the run log directory for current .log files and .tar.gz archives.
        """
        self._update_size_info()
        with self._size_lock:
            return self._current_log_bytes, self._archived_bytes

    def run(self):
        """Main thread loop: drain archive queue and process batch archive requests."""
        logger.info("LogArchiver thread started.")
        while not self._stop_event.is_set():
            try:
                item = self.archive_queue.get(timeout=1.0)
            except (queue.Empty, EOFError):
                continue
            except Exception as e:  # JUSTIFIED: archiver thread must not crash from queue errors
                logger.debug("LogArchiver queue error: %s", e)
                continue

            try:
                self._process_item(item)
            except Exception as e:  # JUSTIFIED: archiver thread must not crash; errors are logged and skipped
                logger.error("LogArchiver error processing item: %s", e, exc_info=True)

        # Drain remaining items on shutdown
        self._drain_queue()
        logger.info("LogArchiver thread stopped.")

    def _drain_queue(self):
        """Process any remaining items in the queue during shutdown."""
        drained = 0
        while True:
            try:
                item = self.archive_queue.get_nowait()
                self._process_item(item)
                drained += 1
            except (queue.Empty, EOFError):
                break
            except Exception as e:  # JUSTIFIED: draining on shutdown; best-effort
                logger.debug("LogArchiver drain error: %s", e)
                break
        if drained > 0:
            logger.info("LogArchiver drained %d items on shutdown.", drained)

    def _process_item(self, item):
        """Process a single archive queue item."""
        if not isinstance(item, tuple) or len(item) < 3:
            logger.warning("LogArchiver: unexpected queue item: %s", item)
            return

        command, log_dir, base_pattern = item[0], item[1], item[2]

        if command == "BATCH_ARCHIVE":
            self._batch_archive(log_dir, base_pattern)
        else:
            logger.warning("LogArchiver: unknown command '%s'", command)

    def _batch_archive(self, log_dir: str, base_pattern: str):
        """
        Compress all completed log files matching the base pattern into a single tar.gz.

        Args:
            log_dir: Directory containing log files.
            base_pattern: Base pattern for log files (e.g., "logs/run_dir/cambia_run_2025-main").
        """
        base_name = os.path.basename(base_pattern)
        glob_pattern = os.path.join(log_dir, f"{base_name}_*.log")
        log_files = sorted(glob.glob(glob_pattern))

        if not log_files:
            logger.debug("LogArchiver: no files to archive for pattern %s", base_pattern)
            return

        # Determine archive directory
        if self.archive_subdir:
            archive_dir = os.path.join(log_dir, self.archive_subdir)
        else:
            archive_dir = log_dir

        os.makedirs(archive_dir, exist_ok=True)

        # Create archive filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archive_name = f"{base_name}_archive_{timestamp}.tar.gz"
        archive_path = os.path.join(archive_dir, archive_name)

        try:
            with tarfile.open(archive_path, "w:gz") as tar:
                for log_file in log_files:
                    tar.add(log_file, arcname=os.path.basename(log_file))

            # Remove archived files
            for log_file in log_files:
                try:
                    os.remove(log_file)
                except OSError as e:
                    logger.warning("LogArchiver: could not remove %s: %s", log_file, e)

            logger.info(
                "LogArchiver: archived %d files into %s", len(log_files), archive_name
            )

        except (OSError, tarfile.TarError) as e:
            logger.error("LogArchiver: failed to create archive %s: %s", archive_path, e)
            # Clean up partial archive
            if os.path.exists(archive_path):
                try:
                    os.remove(archive_path)
                except OSError:
                    pass

        # Cleanup old archives if max_archives is set
        if self.max_archives > 0:
            self._cleanup_old_archives(archive_dir, base_name)

    def _cleanup_old_archives(self, archive_dir: str, base_name: str):
        """Remove oldest archives if count exceeds max_archives."""
        pattern = os.path.join(archive_dir, f"{base_name}_archive_*.tar.gz")
        archives = sorted(glob.glob(pattern), key=os.path.getmtime)

        while len(archives) > self.max_archives:
            oldest = archives.pop(0)
            try:
                os.remove(oldest)
                logger.info("LogArchiver: removed old archive %s", os.path.basename(oldest))
            except OSError as e:
                logger.warning("LogArchiver: could not remove old archive %s: %s", oldest, e)
                break

    def _update_size_info(self):
        """Scan run_log_dir for current and archived file sizes."""
        if not self.run_log_dir or not os.path.isdir(self.run_log_dir):
            return

        current_bytes = 0
        archived_bytes = 0

        try:
            for entry in os.scandir(self.run_log_dir):
                if entry.is_file():
                    size = entry.stat().st_size
                    if entry.name.endswith(".tar.gz"):
                        archived_bytes += size
                    elif entry.name.endswith(".log"):
                        current_bytes += size

            # Also check archive subdirectory
            if self.archive_subdir:
                archive_path = os.path.join(self.run_log_dir, self.archive_subdir)
                if os.path.isdir(archive_path):
                    for entry in os.scandir(archive_path):
                        if entry.is_file() and entry.name.endswith(".tar.gz"):
                            archived_bytes += entry.stat().st_size

        except OSError as e:
            logger.debug("LogArchiver: error scanning log dir: %s", e)

        with self._size_lock:
            self._current_log_bytes = current_bytes
            self._archived_bytes = archived_bytes
