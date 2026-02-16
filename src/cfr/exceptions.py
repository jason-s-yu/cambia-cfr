"""
src/cfr/exceptions.py
Custom exceptions for the CFR module.

Custom exception hierarchy for game state, agent observation, network,
training, storage, and config errors.
"""


class GracefulShutdownException(Exception):
    """Custom exception raised when a graceful shutdown is requested."""


# ============================================================================
# Game State Errors
# ============================================================================


class GameStateError(Exception):
    """Base class for all game state related errors."""


class InvalidGameStateError(GameStateError):
    """
    Raised when game state setup or initialization fails.

    Examples: invalid initial state, malformed game configuration,
    state preconditions not satisfied.
    """


class ActionApplicationError(GameStateError):
    """
    Raised when applying an action to a game state fails.

    Examples: illegal move attempted, action not valid in current state,
    state transition logic error.
    """


class UndoFailureError(GameStateError):
    """
    Raised when the undo system encounters corruption or inconsistency.

    Examples: undo stack corrupted, cannot restore previous state,
    state history mismatch.
    """


# ============================================================================
# Agent State Errors
# ============================================================================


class AgentStateError(Exception):
    """Base class for all agent state related errors."""


class ObservationUpdateError(AgentStateError):
    """
    Raised when belief update or observation processing fails.

    Examples: invalid observation format, belief state update computation error,
    observation incompatible with current agent state.
    """


# ============================================================================
# Encoding Errors
# ============================================================================


class EncodingError(Exception):
    """Base class for all encoding/representation errors."""


class InfosetEncodingError(EncodingError):
    """
    Raised when encoding information sets to feature vectors fails.

    Examples: invalid infoset state, feature extraction error,
    encoding dimension mismatch.
    """


class ActionEncodingError(EncodingError):
    """
    Raised when action index mapping or encoding fails.

    Examples: action not in valid action space, invalid action index,
    action decoding error.
    """


# ============================================================================
# Network Errors
# ============================================================================


class NetworkError(Exception):
    """Base class for all neural network related errors."""


class InvalidNetworkInputError(NetworkError):
    """
    Raised when network input validation fails.

    Examples: incorrect input shape, NaN values detected,
    input outside expected range.
    """


class NetworkInferenceError(NetworkError):
    """
    Raised when neural network forward pass fails.

    Examples: inference computation error, model in incorrect mode,
    output validation failure.
    """


# ============================================================================
# Training Errors
# ============================================================================


class TrainingError(Exception):
    """Base class for all training process related errors."""


class CheckpointSaveError(TrainingError):
    """
    Raised when saving a training checkpoint fails.

    Examples: disk write error, serialization failure,
    insufficient disk space.
    """


class CheckpointLoadError(TrainingError):
    """
    Raised when loading a checkpoint fails or checkpoint is corrupted.

    Examples: file not found, deserialization error,
    version mismatch, corrupted checkpoint data.
    """


class TraversalError(TrainingError):
    """
    Raised when worker traversal of the game tree fails.

    Examples: recursion depth exceeded, invalid traversal state,
    worker synchronization error.
    """


# ============================================================================
# Storage Errors
# ============================================================================


class StorageError(Exception):
    """Base class for all storage system related errors."""


class ReservoirIOError(StorageError):
    """
    Raised when reservoir buffer save or load operations fail.

    Examples: buffer serialization error, file I/O failure,
    buffer corruption on read.
    """


# ============================================================================
# Configuration Errors
# ============================================================================


class ConfigError(Exception):
    """Base class for all configuration related errors."""


class ConfigParseError(ConfigError):
    """
    Raised when configuration file parsing fails.

    Examples: invalid YAML syntax, file not found,
    malformed configuration structure.
    """


class ConfigValidationError(ConfigError):
    """
    Raised when configuration values fail validation.

    Examples: invalid parameter values, missing required fields,
    incompatible configuration options.
    """
