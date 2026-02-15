# src/remote_logging/grpc_handler.py

import grpc
import queue
import threading
from typing import Dict, Any, List
from .log_handler import LogHandler

# Import the generated code
from . import logging_pb2
from . import logging_pb2_grpc

from google.protobuf.struct_pb2 import Struct
from google.protobuf.timestamp_pb2 import Timestamp

class GRPCHandler(LogHandler):
    """
    A non-blocking log handler that streams logs to a gRPC endpoint.
    """
    def __init__(self, endpoint: str, timeout: int = 10):
        self.endpoint = endpoint
        self.timeout = timeout
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._shutdown_event = threading.Event()
        self._thread.start()

    def _log_entry_generator(self):
        """A generator that yields log entries from the queue."""
        while not self._shutdown_event.is_set() or not self._queue.empty():
            try:
                record = self._queue.get(timeout=0.1)
                
                # Convert dict to gRPC message
                entry = logging_pb2.LogEntry(
                    level=record.get("level", "INFO"),
                    message=record.get("message", ""),
                    source=record.get("source", "unknown")
                )
                # Set timestamp
                entry.timestamp.FromGetCurrentTime()
                
                # Convert context dict to protobuf Struct
                if record.get("context"):
                    s = Struct()
                    s.update(record["context"])
                    entry.context.update(s.items())
                    
                yield entry
            except queue.Empty:
                continue

    def _worker(self):
        """The background worker that manages the gRPC stream."""
        while not self._shutdown_event.is_set():
            try:
                with grpc.insecure_channel(self.endpoint) as channel:
                    stub = logging_pb2_grpc.LoggingServiceStub(channel)
                    
                    # This call will block until the stream is broken or closed
                    summary = stub.StreamLog(self._log_entry_generator())
                    print(f"gRPCHandler: Stream closed. Received {summary.entries_received} logs.")
                    
            except grpc.RpcError as e:
                print(f"gRPCHandler: Connection error, will retry in {self.timeout}s: {e.details()}")
                self._shutdown_event.wait(self.timeout)

    def log(self, level: str, message: str, context: Dict[str, Any] = None):
        """Adds a single log record to the queue."""
        record = {"level": level, "message": message, "context": context or {}}
        self._queue.put(record)

    def log_batch(self, batch: List[Dict[str, Any]]):
        """Adds multiple log records to the queue."""
        for record in batch:
            self._queue.put(record)

    def close(self):
        """Signals the worker thread to shut down."""
        self._shutdown_event.set()
        self._thread.join(timeout=self.timeout)