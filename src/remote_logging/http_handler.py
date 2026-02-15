# src/remote_logging/http_handler.py

import httpx
import queue
import threading
import time
from typing import Dict, Any, List
from .log_handler import LogHandler

class HTTPLogHandler(LogHandler):
    """
    A non-blocking log handler that sends logs to an HTTP endpoint.
    It uses a background thread and a queue to send logs in batches.
    """
    def __init__(self, endpoint: str, batch_size: int = 100, timeout: int = 5):
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.timeout = timeout
        
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._shutdown_event = threading.Event()
        self._thread.start()

    def _worker(self):
        """The background worker that sends logs."""
        client = httpx.Client(timeout=self.timeout)
        while not self._shutdown_event.is_set():
            try:
                # Wait for a short time to allow batching
                time.sleep(1)
                
                # Collect a batch of logs
                batch = []
                while len(batch) < self.batch_size and not self._queue.empty():
                    batch.append(self._queue.get_nowait())

                if batch:
                    try:
                        client.post(self.endpoint, json=batch)
                    except httpx.RequestError as e:
                        print(f"HTTPLogHandler: Failed to send log batch: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"HTTPLogHandler: Unhandled error in worker: {e}")
        
        # Final flush before shutdown
        self._flush_queue(client)
        client.close()

    def _flush_queue(self, client: httpx.Client):
        """Flushes any remaining logs in the queue."""
        while not self._queue.empty():
            batch = []
            while len(batch) < self.batch_size and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            if batch:
                try:
                    client.post(self.endpoint, json=batch)
                except httpx.RequestError:
                    pass # Suppress errors during final shutdown

    def log(self, level: str, message: str, context: Dict[str, Any] = None):
        """Adds a single log record to the queue."""
        record = {"level": level, "message": message, "context": context or {}}
        self._queue.put(record)

    def log_batch(self, batch: List[Dict[str, Any]]):
        """Adds multiple log records to the queue."""
        for record in batch:
            self._queue.put(record)

    def close(self):
        """Signals the worker thread to shut down and waits for it."""
        self._shutdown_event.set()
        self._thread.join(timeout=self.timeout + 1)