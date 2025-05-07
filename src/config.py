"""src/config.py"""

from typing import List, Dict, TypeVar, Optional, Union
from dataclasses import dataclass, field
import os
import logging
import yaml
import re  # For parsing human-readable sizes

T = TypeVar("T")


# Helper to get nested dict values safely
def get_nested(data: Dict, keys: List[str], default: T) -> T:
    """Safely retrieve a nested value from a dict."""
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current.get(key)
        else:
            return default
    # Handle case where the final value retrieved is None, but default isn't None
    if current is None and default is not None:
        return default
    return current  # type: ignore


def parse_human_readable_size(size_str: Union[str, int]) -> int:
    """Parses a human-readable size string (e.g., '1GB', '500MB', '1024') into bytes."""
    if isinstance(size_str, int):
        return size_str
    if not isinstance(size_str, str):
        raise ValueError(f"Invalid size format: {size_str}. Must be int or string.")

    size_str = size_str.upper().strip()
    match = re.fullmatch(r"(\d+)\s*(KB|MB|GB|TB)?", size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")

    value = int(match.group(1))
    unit = match.group(2)

    if unit == "KB":
        value *= 1024
    elif unit == "MB":
        value *= 1024**2
    elif unit == "GB":
        value *= 1024**3
    elif unit == "TB":
        value *= 1024**4
    # If unit is None, value is already in bytes
    return value


def parse_num_workers(num_workers: str | int) -> int:
    """Parse the number of workers, ensuring it's a positive integer."""
    if isinstance(num_workers, int):
        if num_workers < 0:
            # Allow 0 for auto, interpreted as os.cpu_count()
            if num_workers == 0:  # 0 means auto
                cpu_count = os.cpu_count()
                return cpu_count - 1 if cpu_count is not None and cpu_count > 1 else 1
            raise ValueError("num_workers must be a positive integer or 0 for auto.")
        return num_workers
    if isinstance(num_workers, str):
        if num_workers.lower() == "auto":
            cpu_count = os.cpu_count()
            return cpu_count - 1 if cpu_count is not None and cpu_count > 1 else 1
        try:
            val = int(num_workers)
            if val < 0:
                # Allow 0 for auto
                if val == 0:  # 0 means auto
                    cpu_count = os.cpu_count()
                    return cpu_count - 1 if cpu_count is not None and cpu_count > 1 else 1
                raise ValueError("num_workers must be a positive integer or 0 for auto.")
            return val
        except ValueError:
            raise ValueError(
                f"num_workers string '{num_workers}' must be 'auto' or an integer."
            ) from None
    raise ValueError(
        f"num_workers must be 'auto' or a positive integer. Got: {num_workers}"
    )


@dataclass
class ApiConfig:
    base_url: str = "http://localhost:8080"  # Example default
    auth: Dict[str, str] = field(
        default_factory=dict
    )  # e.g., {"email": "...", "password": "..."} or {"token_file": "..."}


@dataclass
class SystemConfig:
    recursion_limit: int = 10000  # Limit for recursion depth


@dataclass
class CfrTrainingConfig:
    num_iterations: int = 100000
    save_interval: int = 5000
    pruning_enabled: bool = True  # Enable Regret-Based Pruning
    pruning_threshold: float = (
        1.0e-6  # Regrets below this are considered zero for pruning
    )
    exploitability_interval: int = (
        1000  # How often (in iterations) to calculate exploitability
    )
    num_workers: int = (
        1  # Number of parallel workers for simulations. 1 for serial operation
    )


@dataclass
class CfrPlusParamsConfig:
    weighted_averaging_enabled: bool = True
    averaging_delay: int = (
        100  # Start averaging from iteration d+1 (weight = max(0, t - d))
    )


@dataclass
class AgentParamsConfig:
    memory_level: int = 1  # 0: Perfect Recall, 1: Event Decay, 2: Event+Time Decay
    time_decay_turns: int = 3  # Used only if memory_level == 2


@dataclass
class CambiaRulesConfig:
    allowDrawFromDiscardPile: bool = False  # Default House Rules
    allowReplaceAbilities: bool = False
    snapRace: bool = False
    penaltyDrawCount: int = 2
    use_jokers: int = 2
    cards_per_player: int = 4
    initial_view_count: int = 2
    cambia_allowed_round: int = 0  # 0 means allowed immediately
    allowOpponentSnapping: bool = False  # Default to False
    max_game_turns: int = 300  # Limit game length in simulation


@dataclass
class PersistenceConfig:
    agent_data_save_path: str = "cambia_cfr_agent_level1.joblib"


@dataclass
class WorkerLogOverrideConfig:
    worker_ids: List[int] = field(default_factory=list)
    level: str = "INFO"


@dataclass
class WorkerLoggingConfig:
    default_level: str = "INFO"
    sequential_rules: List[Union[str, Dict[str, int]]] = field(default_factory=list)
    overrides: List[WorkerLogOverrideConfig] = field(default_factory=list)


@dataclass
class LoggingConfig:
    log_level_file: str = "DEBUG"  # Logging level for the main process file
    log_level_console: str = "ERROR"  # Logging level for the console
    log_dir: str = "logs"
    log_file_prefix: str = "cambia"
    log_max_bytes: int = (
        9 * 1024 * 1024
    )  # Default for SerialRotatingFileHandler, can be string like "9MB"
    log_backup_count: int = 999
    worker_config: Optional[WorkerLoggingConfig] = None
    # Archiving settings
    log_archive_enabled: bool = False
    log_archive_max_archives: int = (
        10  # Max number of tar.gz archives to keep per worker type
    )
    log_archive_dir: str = "archives"  # Subdirectory within log_dir for archives

    def get_worker_log_level(self, worker_id: int, num_total_workers: int) -> str:
        """
        Determines the log level for a specific worker based on the configuration.
        """
        if self.worker_config is None:
            return (
                self.log_level_file
            )  # Fallback to global file_level if no worker_config

        # 1. Check Overrides first
        if self.worker_config.overrides:
            for override_rule in self.worker_config.overrides:
                if worker_id in override_rule.worker_ids:
                    return override_rule.level.upper()

        # 2. Check Sequential Rules
        current_worker_idx_in_rules = 0
        if self.worker_config.sequential_rules:
            for rule_entry in self.worker_config.sequential_rules:
                level_to_apply: str
                num_workers_for_rule = 1

                if isinstance(rule_entry, str):
                    level_to_apply = rule_entry.upper()
                elif isinstance(rule_entry, dict):
                    # Expecting { "LEVEL": count }
                    if len(rule_entry) != 1:
                        logging.warning(
                            "Invalid sequential_rule format: %s. Skipping.", rule_entry
                        )
                        continue
                    level_to_apply = list(rule_entry.keys())[0].upper()
                    num_workers_for_rule = list(rule_entry.values())[0]
                    if (
                        not isinstance(num_workers_for_rule, int)
                        or num_workers_for_rule < 1
                    ):
                        logging.warning(
                            "Invalid count in sequential_rule: %s. Using 1.", rule_entry
                        )
                        num_workers_for_rule = 1
                else:
                    logging.warning(
                        "Unknown sequential_rule type: %s. Skipping.", rule_entry
                    )
                    continue

                # Check if the current worker_id falls into this rule's range
                if (
                    current_worker_idx_in_rules
                    <= worker_id
                    < current_worker_idx_in_rules + num_workers_for_rule
                ):
                    return level_to_apply
                current_worker_idx_in_rules += num_workers_for_rule

        # 3. Fallback to worker_config.default_level
        if self.worker_config.default_level:
            return self.worker_config.default_level.upper()

        # 4. Ultimate fallback to global log_level_file
        return self.log_level_file.upper()


@dataclass
class Config:
    api: ApiConfig = field(default_factory=ApiConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    cfr_training: CfrTrainingConfig = field(default_factory=CfrTrainingConfig)
    cfr_plus_params: CfrPlusParamsConfig = field(default_factory=CfrPlusParamsConfig)
    agent_params: AgentParamsConfig = field(default_factory=AgentParamsConfig)
    cambia_rules: CambiaRulesConfig = field(default_factory=CambiaRulesConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    _source_path: Optional[str] = None  # Internal field to store config path


def load_config(
    config_path: str = "config.yaml",
) -> Optional[Config]:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
            if config_dict is None:
                print(
                    f"Warning: Config file '{config_path}' is empty or invalid. "
                    f"Using default configuration."
                )
                config_dict = {}

            api_config_dict = config_dict.get("api", {})
            auth_dict = api_config_dict.get("auth", {})
            api_config = ApiConfig(
                base_url=api_config_dict.get("base_url", ApiConfig.base_url),
                auth=auth_dict,
            )

            # Parse WorkerLoggingConfig if present
            worker_log_cfg_dict = get_nested(
                config_dict, ["logging", "worker_config"], None
            )
            worker_logging_config: Optional[WorkerLoggingConfig] = None
            if worker_log_cfg_dict and isinstance(worker_log_cfg_dict, dict):
                sequential_rules_raw = worker_log_cfg_dict.get("sequential_rules", [])
                overrides_raw = worker_log_cfg_dict.get("overrides", [])
                parsed_overrides = []
                if isinstance(overrides_raw, list):
                    for override_item in overrides_raw:
                        if (
                            isinstance(override_item, dict)
                            and "worker_ids" in override_item
                            and "level" in override_item
                            and isinstance(override_item["worker_ids"], list)
                            and isinstance(override_item["level"], str)
                        ):
                            parsed_overrides.append(
                                WorkerLogOverrideConfig(
                                    worker_ids=override_item["worker_ids"],
                                    level=override_item["level"],
                                )
                            )
                        else:
                            logging.warning(
                                "Skipping invalid worker log override item: %s",
                                override_item,
                            )

                worker_logging_config = WorkerLoggingConfig(
                    default_level=worker_log_cfg_dict.get(
                        "default_level", WorkerLoggingConfig.default_level
                    ),
                    sequential_rules=(
                        sequential_rules_raw
                        if isinstance(sequential_rules_raw, list)
                        else []
                    ),
                    overrides=parsed_overrides,
                )

            logging_config = LoggingConfig(
                log_level_file=get_nested(
                    config_dict,
                    ["logging", "log_level_file"],
                    LoggingConfig.log_level_file,
                ),
                log_level_console=get_nested(
                    config_dict,
                    ["logging", "log_level_console"],
                    LoggingConfig.log_level_console,
                ),
                log_dir=get_nested(
                    config_dict, ["logging", "log_dir"], LoggingConfig.log_dir
                ),
                log_file_prefix=get_nested(
                    config_dict,
                    ["logging", "log_file_prefix"],
                    LoggingConfig.log_file_prefix,
                ),
                log_max_bytes=parse_human_readable_size(
                    get_nested(
                        config_dict,
                        ["logging", "log_max_bytes"],
                        LoggingConfig.log_max_bytes,
                    )
                ),
                log_backup_count=get_nested(
                    config_dict,
                    ["logging", "log_backup_count"],
                    LoggingConfig.log_backup_count,
                ),
                worker_config=worker_logging_config,
                log_archive_enabled=get_nested(
                    config_dict,
                    ["logging", "log_archive_enabled"],
                    LoggingConfig.log_archive_enabled,
                ),
                log_archive_max_archives=get_nested(
                    config_dict,
                    ["logging", "log_archive_max_archives"],
                    LoggingConfig.log_archive_max_archives,
                ),
                log_archive_dir=get_nested(
                    config_dict,
                    ["logging", "log_archive_dir"],
                    LoggingConfig.log_archive_dir,
                ),
            )

            cfg = Config(
                api=api_config,
                system=SystemConfig(
                    recursion_limit=get_nested(
                        config_dict,
                        ["system", "recursion_limit"],
                        SystemConfig.recursion_limit,
                    )
                ),
                cfr_training=CfrTrainingConfig(
                    num_iterations=get_nested(
                        config_dict,
                        ["cfr_training", "num_iterations"],
                        CfrTrainingConfig.num_iterations,
                    ),
                    save_interval=get_nested(
                        config_dict,
                        ["cfr_training", "save_interval"],
                        CfrTrainingConfig.save_interval,
                    ),
                    pruning_enabled=get_nested(
                        config_dict,
                        ["cfr_training", "pruning_enabled"],
                        CfrTrainingConfig.pruning_enabled,
                    ),
                    pruning_threshold=get_nested(
                        config_dict,
                        ["cfr_training", "pruning_threshold"],
                        CfrTrainingConfig.pruning_threshold,
                    ),
                    exploitability_interval=get_nested(
                        config_dict,
                        ["cfr_training", "exploitability_interval"],
                        CfrTrainingConfig.exploitability_interval,
                    ),
                    num_workers=parse_num_workers(
                        get_nested(
                            config_dict,
                            ["cfr_training", "num_workers"],
                            CfrTrainingConfig.num_workers,
                        )
                    ),
                ),
                cfr_plus_params=CfrPlusParamsConfig(
                    weighted_averaging_enabled=get_nested(
                        config_dict,
                        ["cfr_plus_params", "weighted_averaging_enabled"],
                        CfrPlusParamsConfig.weighted_averaging_enabled,
                    ),
                    averaging_delay=get_nested(
                        config_dict,
                        ["cfr_plus_params", "averaging_delay"],
                        CfrPlusParamsConfig.averaging_delay,
                    ),
                ),
                agent_params=AgentParamsConfig(
                    memory_level=get_nested(
                        config_dict,
                        ["agent_params", "memory_level"],
                        AgentParamsConfig.memory_level,
                    ),
                    time_decay_turns=get_nested(
                        config_dict,
                        ["agent_params", "time_decay_turns"],
                        AgentParamsConfig.time_decay_turns,
                    ),
                ),
                cambia_rules=CambiaRulesConfig(
                    allowDrawFromDiscardPile=get_nested(
                        config_dict,
                        ["cambia_rules", "allowDrawFromDiscardPile"],
                        CambiaRulesConfig.allowDrawFromDiscardPile,
                    ),
                    allowReplaceAbilities=get_nested(
                        config_dict,
                        ["cambia_rules", "allowReplaceAbilities"],
                        CambiaRulesConfig.allowReplaceAbilities,
                    ),
                    snapRace=get_nested(
                        config_dict,
                        ["cambia_rules", "snapRace"],
                        CambiaRulesConfig.snapRace,
                    ),
                    penaltyDrawCount=get_nested(
                        config_dict,
                        ["cambia_rules", "penaltyDrawCount"],
                        CambiaRulesConfig.penaltyDrawCount,
                    ),
                    use_jokers=get_nested(
                        config_dict,
                        ["cambia_rules", "use_jokers"],
                        CambiaRulesConfig.use_jokers,
                    ),
                    cards_per_player=get_nested(
                        config_dict,
                        ["cambia_rules", "cards_per_player"],
                        CambiaRulesConfig.cards_per_player,
                    ),
                    initial_view_count=get_nested(
                        config_dict,
                        ["cambia_rules", "initial_view_count"],
                        CambiaRulesConfig.initial_view_count,
                    ),
                    cambia_allowed_round=get_nested(
                        config_dict,
                        ["cambia_rules", "cambia_allowed_round"],
                        CambiaRulesConfig.cambia_allowed_round,
                    ),
                    allowOpponentSnapping=get_nested(
                        config_dict,
                        ["cambia_rules", "allowOpponentSnapping"],
                        CambiaRulesConfig.allowOpponentSnapping,
                    ),
                    max_game_turns=get_nested(
                        config_dict,
                        ["cambia_rules", "max_game_turns"],
                        CambiaRulesConfig.max_game_turns,
                    ),
                ),
                persistence=PersistenceConfig(
                    agent_data_save_path=get_nested(
                        config_dict,
                        ["persistence", "agent_data_save_path"],
                        PersistenceConfig.agent_data_save_path,
                    )
                ),
                logging=logging_config,
                _source_path=os.path.abspath(config_path),
            )
            return cfg

    except FileNotFoundError:
        print(
            f"Warning: Config file '{config_path}' not found. Using default configuration."
        )
        return Config(_source_path=None)
    except (
        TypeError,
        KeyError,
        AttributeError,
        yaml.YAMLError,
        ValueError,  # For parse_human_readable_size or parse_num_workers
    ) as e:
        print(
            f"Error loading or parsing config file '{config_path}': {e}. "
            f"Check config structure/types."
        )
        print("Using default configuration.")
        return Config(_source_path=None)
    except IOError as e:
        print(f"Unexpected error loading config file '{config_path}': {e}")
        print("Using default configuration.")
        return Config(_source_path=None)
