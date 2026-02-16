"""src/config.py"""

from typing import List, Dict, TypeVar, Optional, Union
from dataclasses import dataclass, field
import os
import logging
import re
import yaml

T = TypeVar("T")

# --- Configuration Dataclasses ---


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


def parse_num_workers(num_workers: Union[str, int]) -> int:
    """Parse the number of workers, ensuring it's a positive integer."""
    if isinstance(num_workers, int):
        if num_workers < 0:
            # Allow 0 for auto, interpreted as os.cpu_count()
            if num_workers == 0:  # 0 means auto
                cpu_count = os.cpu_count()
                return max(1, (cpu_count - 1)) if cpu_count else 1
            raise ValueError("num_workers must be a positive integer or 0 for auto.")
        return max(1, num_workers)  # Ensure at least 1 worker
    if isinstance(num_workers, str):
        if num_workers.lower() == "auto":
            cpu_count = os.cpu_count()
            return max(1, (cpu_count - 1)) if cpu_count else 1
        try:
            val = int(num_workers)
            if val < 0:
                # Allow 0 for auto
                if val == 0:  # 0 means auto
                    cpu_count = os.cpu_count()
                    return max(1, (cpu_count - 1)) if cpu_count else 1
                raise ValueError("num_workers must be a positive integer or 0 for auto.")
            return max(1, val)  # Ensure at least 1 worker
        except ValueError:
            raise ValueError(
                f"num_workers string '{num_workers}' must be 'auto' or an integer."
            ) from None
    raise ValueError(
        f"num_workers must be 'auto' or a positive integer. Got: {num_workers}"
    )


@dataclass
class ApiConfig:
    """API Client Settings (for online play)."""

    base_url: str = "http://localhost:8080"  # Example default
    auth: Dict[str, str] = field(
        default_factory=dict
    )  # e.g., {"email": "...", "password": "..."} or {"token_file": "..."}


@dataclass
class SystemConfig:
    """General system-level settings."""

    recursion_limit: int = 10000  # Limit for recursion depth


@dataclass
class AnalysisConfig:
    """Parameters for analysis tools, like exploitability calculation."""

    exploitability_num_workers: int = 1  # Default to 1, parsed later


@dataclass
class CfrTrainingConfig:
    """Parameters controlling the CFR training process."""

    num_iterations: int = 100000
    save_interval: int = 5000
    pruning_enabled: bool = True  # Enable Regret-Based Pruning
    pruning_threshold: float = (
        1.0e-6  # Regrets below this are considered zero for pruning
    )
    exploitability_interval: int = (
        1000  # How often (in iterations) to calculate exploitability
    )
    exploitability_interval_seconds: int = (
        0  # Min seconds between exploitability calcs (0=disabled)
    )
    num_workers: int = (
        1  # Number of parallel workers for simulations. 1 for serial operation
    )


@dataclass
class CfrPlusParamsConfig:
    """Parameters specific to CFR+ algorithm variants."""

    weighted_averaging_enabled: bool = True
    averaging_delay: int = (
        100  # Start averaging from iteration d+1 (weight = max(0, t - d))
    )


@dataclass
class AgentParamsConfig:
    """Settings defining CFR agent behavior (memory, abstraction)."""

    memory_level: int = 1  # 0: Perfect Recall, 1: Event Decay, 2: Event+Time Decay
    time_decay_turns: int = 3  # Used only if memory_level == 2


@dataclass
class CambiaRulesConfig:
    """Defines the specific game rules of Cambia."""

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
    """Configuration for saving and loading trained agent data."""

    agent_data_save_path: str = "cambia_cfr_agent_level1.joblib"


@dataclass
class WorkerLogOverrideConfig:
    """Specifies a logging level override for specific worker IDs."""

    worker_ids: List[int] = field(default_factory=list)
    level: str = "INFO"


@dataclass
class WorkerLoggingConfig:
    """Configuration for worker-specific logging levels."""

    default_level: str = "INFO"
    sequential_rules: List[Union[str, Dict[str, int]]] = field(default_factory=list)
    overrides: List[WorkerLogOverrideConfig] = field(default_factory=list)


@dataclass
class LoggingConfig:
    """Settings for configuring logging behavior."""

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
    log_size_update_interval_sec: int = (
        60  # Interval in seconds to update log size display
    )
    # Simulation Trace Logging
    log_simulation_traces: bool = False
    simulation_trace_filename_prefix: str = "simulation_traces"

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


# --- Deep CFR Configuration ---
@dataclass
class DeepCfrConfig:
    """Configuration for Deep CFR training."""

    hidden_dim: int = 256
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 2048
    train_steps_per_iteration: int = 4000
    alpha: float = 1.5
    traversals_per_step: int = 1000
    advantage_buffer_capacity: int = 2_000_000
    strategy_buffer_capacity: int = 2_000_000
    save_interval: int = 10
    use_gpu: bool = False


# --- Baseline Agent Configuration ---
@dataclass
class GreedyAgentConfig:
    """Configuration specific to the Greedy baseline agent."""

    cambia_call_threshold: int = 5


@dataclass
class AgentsConfig:
    """Configuration for baseline agents (used for evaluation)."""

    greedy_agent: GreedyAgentConfig = field(default_factory=GreedyAgentConfig)


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
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    deep_cfr: DeepCfrConfig = field(default_factory=DeepCfrConfig)
    _source_path: Optional[str] = None  # Internal field to store config path


def load_config(
    config_path: str = "config.yaml",
) -> Optional[Config]:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
            if config_dict is None:
                logging.warning(
                    "Config file '%s' is empty or invalid. Using default configuration.",
                    config_path,
                )
                config_dict = {}

            # --- Parse sections ---
            api_config_dict = config_dict.get("api", {})
            auth_dict = api_config_dict.get("auth", {})
            api_config = ApiConfig(
                base_url=api_config_dict.get("base_url", ApiConfig.base_url),
                auth=auth_dict,
            )

            # Parse WorkerLoggingConfig
            worker_log_cfg_dict = get_nested(
                config_dict, ["logging", "worker_config"], None
            )
            worker_logging_config = None
            if worker_log_cfg_dict and isinstance(worker_log_cfg_dict, dict):
                seq_rules = worker_log_cfg_dict.get("sequential_rules", [])
                over_raw = worker_log_cfg_dict.get("overrides", [])
                parsed_overrides = []
                if isinstance(over_raw, list):
                    for item in over_raw:
                        if (
                            isinstance(item, dict)
                            and isinstance(item.get("worker_ids"), list)
                            and isinstance(item.get("level"), str)
                        ):
                            parsed_overrides.append(
                                WorkerLogOverrideConfig(
                                    worker_ids=item["worker_ids"], level=item["level"]
                                )
                            )
                        else:
                            logging.warning("Skipping invalid override: %s", item)
                worker_logging_config = WorkerLoggingConfig(
                    default_level=worker_log_cfg_dict.get(
                        "default_level", WorkerLoggingConfig.default_level
                    ),
                    sequential_rules=seq_rules if isinstance(seq_rules, list) else [],
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
                log_size_update_interval_sec=get_nested(
                    config_dict,
                    ["logging", "log_size_update_interval_sec"],
                    LoggingConfig.log_size_update_interval_sec,
                ),
                log_simulation_traces=get_nested(
                    config_dict,
                    ["logging", "log_simulation_traces"],
                    LoggingConfig.log_simulation_traces,
                ),
                simulation_trace_filename_prefix=get_nested(
                    config_dict,
                    ["logging", "simulation_trace_filename_prefix"],
                    LoggingConfig.simulation_trace_filename_prefix,
                ),
            )

            # --- Parse Baseline Agents Config ---
            agents_config_dict = config_dict.get("agents", {})
            greedy_agent_config_dict = agents_config_dict.get(
                "greedy_agent", {}
            )  # Use .get for safety
            greedy_agent_config = GreedyAgentConfig(
                cambia_call_threshold=greedy_agent_config_dict.get(
                    "cambia_call_threshold", GreedyAgentConfig.cambia_call_threshold
                )
            )
            agents_config = AgentsConfig(greedy_agent=greedy_agent_config)

            # --- Parse Analysis Config ---
            cfr_training_num_workers_default = parse_num_workers(
                get_nested(
                    config_dict,
                    ["cfr_training", "num_workers"],
                    CfrTrainingConfig.num_workers,
                )
            )
            analysis_config_dict = config_dict.get("analysis", {})
            # Default exploitability workers to reasonable number based on training workers or half CPU
            default_exploit_workers = max(1, cfr_training_num_workers_default // 2)
            analysis_config = AnalysisConfig(
                exploitability_num_workers=parse_num_workers(
                    analysis_config_dict.get(
                        "exploitability_num_workers", default_exploit_workers
                    )
                )
            )

            # --- Parse Deep CFR Config ---
            deep_cfr_dict = config_dict.get("deep_cfr", {})
            deep_cfr_config = DeepCfrConfig(
                hidden_dim=deep_cfr_dict.get("hidden_dim", DeepCfrConfig.hidden_dim),
                dropout=deep_cfr_dict.get("dropout", DeepCfrConfig.dropout),
                learning_rate=deep_cfr_dict.get("learning_rate", DeepCfrConfig.learning_rate),
                batch_size=deep_cfr_dict.get("batch_size", DeepCfrConfig.batch_size),
                train_steps_per_iteration=deep_cfr_dict.get(
                    "train_steps_per_iteration", DeepCfrConfig.train_steps_per_iteration
                ),
                alpha=deep_cfr_dict.get("alpha", DeepCfrConfig.alpha),
                traversals_per_step=deep_cfr_dict.get(
                    "traversals_per_step", DeepCfrConfig.traversals_per_step
                ),
                advantage_buffer_capacity=deep_cfr_dict.get(
                    "advantage_buffer_capacity", DeepCfrConfig.advantage_buffer_capacity
                ),
                strategy_buffer_capacity=deep_cfr_dict.get(
                    "strategy_buffer_capacity", DeepCfrConfig.strategy_buffer_capacity
                ),
                save_interval=deep_cfr_dict.get("save_interval", DeepCfrConfig.save_interval),
                use_gpu=deep_cfr_dict.get("use_gpu", DeepCfrConfig.use_gpu),
            )

            # --- Assemble Main Config ---
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
                    exploitability_interval_seconds=get_nested(
                        config_dict,
                        ["cfr_training", "exploitability_interval_seconds"],
                        CfrTrainingConfig.exploitability_interval_seconds,
                    ),
                    # Parse num_workers here after defaults established
                    num_workers=cfr_training_num_workers_default,
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
                agents=agents_config,
                analysis=analysis_config,
                deep_cfr=deep_cfr_config,
                _source_path=os.path.abspath(config_path),
            )
            return cfg

    except FileNotFoundError:
        logging.error(
            "Config file '%s' not found. Using default configuration.", config_path
        )
        return Config(_source_path=None)
    except (TypeError, KeyError, AttributeError, yaml.YAMLError, ValueError) as e:
        logging.exception(
            "Error loading or parsing config file '%s': %s. Check structure/types. Using default config.",
            config_path,
            e,
        )
        return Config(_source_path=None)
    except IOError as e:
        logging.exception(
            "IOError loading config file '%s': %s. Using default config.", config_path, e
        )
        return Config(_source_path=None)
