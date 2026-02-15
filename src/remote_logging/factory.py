# src/remote_logging/factory.py

from ..config import LoggingConfig
from .log_handler import LogHandler
from .local_handler import LocalLogHandler
from .http_handler import HTTPLogHandler
from .grpc_handler import GRPCHandler


def create_log_handler(config: LoggingConfig, **kwargs) -> LogHandler:
    """
    Factory function to instantiate and return the correct log handler
    based on the provided configuration.
    """
    if config.type == "http":
        if not config.remote:
            raise ValueError("HTTP logging selected but remote config is missing.")
        return HTTPLogHandler(
            endpoint=config.remote.endpoint,
            batch_size=config.remote.batch_size,
            timeout=config.remote.timeout_seconds,
        )
    elif config.type == "grpc":
        if not config.remote:
            raise ValueError("gRPC logging selected but remote config is missing.")
        return GRPCHandler(
            endpoint=config.remote.endpoint, timeout=config.remote.timeout_seconds
        )
    elif config.type == "local":
        # The LocalLogHandler will wrap the existing file logging setup.
        # It might need access to the existing file handler from the main setup.
        # We can pass it via kwargs.
        main_file_handler = kwargs.get("main_file_handler")
        return LocalLogHandler(main_file_handler)
    else:
        raise ValueError(f"Unknown logging handler type: {config.type}")
