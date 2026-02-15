"""src/config.py"""

from typing import List, Dict, TypeVar, Optional, Union, Literal
from pydantic import BaseModel, Field
import os
import logging
import re
import yaml

T = TypeVar("T")

# --- New Logging Configuration Models ---


class RemoteLoggingConfig(BaseModel):
    """Configuration for remote log handlers (HTTP/gRPC)."""

    endpoint: str = "http://localhost:8080/logs"
    batch_size: int = 100
    timeout_seconds: int = 5


class LoggingConfig(BaseModel):
    """Extended logging configuration."""

    type: Literal["local", "http", "grpc"] = Field(
        "local",
        description="The logging handler to use: 'local' for files, 'http' or 'grpc' for remote streaming.",
    )
    remote: Optional[RemoteLoggingConfig] = Field(
        default_factory=RemoteLoggingConfig, description="Settings for remote handlers."
    )

    # --- Existing LoggingConfig fields can be nested under a 'file' key for clarity ---
    log_level_file: str = "DEBUG"
    log_level_console: str = "ERROR"
    log_dir: str = "logs"
    log_file_prefix: str = "cambia"
    log_max_bytes: int = 9 * 1024 * 1024
    log_backup_count: int = 999
    # ... (rest of the existing logging fields) ...


# ... (rest of the existing config dataclasses: ApiConfig, SystemConfig, etc.) ...


# --- Main Config Class ---
@dataclass
class Config:
    """Root configuration object."""

    api: ApiConfig = field(default_factory=ApiConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    cfr_training: CfrTrainingConfig = field(default_factory=CfrTrainingConfig)
    cfr_plus_params: CfrPlusParamsConfig = field(default_factory=CfrPlusParamsConfig)
    agent_params: AgentParamsConfig = field(default_factory=AgentParamsConfig)
    cambia_rules: CambiaRulesConfig = field(default_factory=CambiaRulesConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)

    # Replace the old logging config with the new, extended one
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    agents: AgentsConfig = field(default_factory=AgentsConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    _source_path: Optional[str] = None


# ... (The existing load_config function will now automatically parse the new structure) ...
# No changes are needed to load_config as long as it uses a library like Pydantic
# or a recursive dataclass parsing method that handles nested models.
# For the purpose of this implementation, I will assume the existing `load_config`
# function in `src/config.py` is capable of this.
