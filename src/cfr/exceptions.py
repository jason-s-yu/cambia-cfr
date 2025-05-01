"""
src/cfr/exceptions.py
Custom exceptions for the CFR module.
"""


class GracefulShutdownException(Exception):
    """Custom exception raised when a graceful shutdown is requested."""
