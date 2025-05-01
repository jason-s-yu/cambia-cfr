# src/config.py
from dataclasses import dataclass, field
import yaml
from typing import List, Dict, TypeVar, Optional # Added Optional

T = TypeVar('T')

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
     # Type check for safety, though it might be too strict sometimes
     # if default is not None and not isinstance(current, type(default)):
     #     print(f"Warning: Type mismatch for config key {'/'.join(keys)}. Expected {type(default)}, got {type(current)}. Using default.")
     #     return default
     return current # type: ignore


@dataclass
class ApiConfig:
    base_url: str = "http://localhost:8080" # Example default
    auth: Dict[str, str] = field(default_factory=dict) # e.g., {"email": "...", "password": "..."} or {"token_file": "..."}

@dataclass
class SystemConfig:
    recursion_limit: int = 10000 # Limit for recursion depth

@dataclass
class CfrTrainingConfig:
    num_iterations: int = 100000
    save_interval: int = 5000
    pruning_enabled: bool = True # Enable Regret-Based Pruning
    pruning_threshold: float = 1.0e-6 # Regrets below this are considered zero for pruning
    exploitability_interval: int = 1000 # How often (in iterations) to calculate exploitability

@dataclass
class CfrPlusParamsConfig:
    weighted_averaging_enabled: bool = True
    averaging_delay: int = 100 # Start averaging from iteration d+1 (weight = max(0, t - d))

@dataclass
class AgentParamsConfig:
    memory_level: int = 1 # 0: Perfect Recall, 1: Event Decay, 2: Event+Time Decay
    time_decay_turns: int = 3 # Used only if memory_level == 2

@dataclass
class CambiaRulesConfig:
    allowDrawFromDiscardPile: bool = False # Default House Rules
    allowReplaceAbilities: bool = False
    snapRace: bool = False
    penaltyDrawCount: int = 2
    use_jokers: int = 2
    cards_per_player: int = 4
    initial_view_count: int = 2
    cambia_allowed_round: int = 0 # 0 means allowed immediately
    allowOpponentSnapping: bool = False # Default to False
    max_game_turns: int = 300 # Limit game length in simulation

@dataclass
class PersistenceConfig:
    agent_data_save_path: str = "cambia_cfr_agent_level1.joblib"

@dataclass
class LoggingConfig:
    log_level_file: str = "DEBUG" # Logging level for the file
    log_level_console: str = "ERROR" # Logging level for the console
    log_dir: str = "logs" # Directory to store log files
    log_file_prefix: str = "cambia" # Prefix for log files within the run directory
    log_max_bytes: int = 9 * 1024 * 1024 # Max size before rotation (~9MB)
    log_backup_count: int = 999 # Max number of backup log files

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

def load_config(config_path: str = "config.yaml") -> Optional[Config]: # Return Optional[Config]
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            if config_dict is None:
                print(f"Warning: Config file '{config_path}' is empty or invalid. Using default configuration.")
                config_dict = {} # Use empty dict to proceed with defaults

            # Retrieve the 'api' section safely, defaulting to an empty dict if not found
            api_config_dict = config_dict.get('api', {})
            # Retrieve the nested 'auth' dictionary within 'api', defaulting to empty if not found
            auth_dict = api_config_dict.get('auth', {})

            # Construct ApiConfig using values from the loaded dict or defaults
            api_config = ApiConfig(
                base_url=api_config_dict.get('base_url', ApiConfig.base_url),
                auth=auth_dict # Directly assign the retrieved auth dictionary
            )

            # Using helper for safer nested access with defaults from dataclasses for other sections
            return Config(
                api=api_config, # Use the manually constructed ApiConfig
                system=SystemConfig(
                    recursion_limit=get_nested(config_dict, ['system', 'recursion_limit'], SystemConfig.recursion_limit)
                ),
                cfr_training=CfrTrainingConfig(
                    num_iterations=get_nested(config_dict, ['cfr_training', 'num_iterations'], CfrTrainingConfig.num_iterations),
                    save_interval=get_nested(config_dict, ['cfr_training', 'save_interval'], CfrTrainingConfig.save_interval),
                    pruning_enabled=get_nested(config_dict, ['cfr_training', 'pruning_enabled'], CfrTrainingConfig.pruning_enabled),
                    pruning_threshold=get_nested(config_dict, ['cfr_training', 'pruning_threshold'], CfrTrainingConfig.pruning_threshold),
                    exploitability_interval=get_nested(config_dict, ['cfr_training', 'exploitability_interval'], CfrTrainingConfig.exploitability_interval)
                ),
                cfr_plus_params=CfrPlusParamsConfig(
                     weighted_averaging_enabled=get_nested(config_dict, ['cfr_plus_params', 'weighted_averaging_enabled'], CfrPlusParamsConfig.weighted_averaging_enabled),
                     averaging_delay=get_nested(config_dict, ['cfr_plus_params', 'averaging_delay'], CfrPlusParamsConfig.averaging_delay)
                ),
                agent_params=AgentParamsConfig(
                     memory_level=get_nested(config_dict, ['agent_params', 'memory_level'], AgentParamsConfig.memory_level),
                     time_decay_turns=get_nested(config_dict, ['agent_params', 'time_decay_turns'], AgentParamsConfig.time_decay_turns)
                ),
                cambia_rules=CambiaRulesConfig(
                     allowDrawFromDiscardPile=get_nested(config_dict, ['cambia_rules', 'allowDrawFromDiscardPile'], CambiaRulesConfig.allowDrawFromDiscardPile),
                     allowReplaceAbilities=get_nested(config_dict, ['cambia_rules', 'allowReplaceAbilities'], CambiaRulesConfig.allowReplaceAbilities),
                     snapRace=get_nested(config_dict, ['cambia_rules', 'snapRace'], CambiaRulesConfig.snapRace),
                     penaltyDrawCount=get_nested(config_dict, ['cambia_rules', 'penaltyDrawCount'], CambiaRulesConfig.penaltyDrawCount),
                     use_jokers=get_nested(config_dict, ['cambia_rules', 'use_jokers'], CambiaRulesConfig.use_jokers),
                     cards_per_player=get_nested(config_dict, ['cambia_rules', 'cards_per_player'], CambiaRulesConfig.cards_per_player),
                     initial_view_count=get_nested(config_dict, ['cambia_rules', 'initial_view_count'], CambiaRulesConfig.initial_view_count),
                     cambia_allowed_round=get_nested(config_dict, ['cambia_rules', 'cambia_allowed_round'], CambiaRulesConfig.cambia_allowed_round),
                     allowOpponentSnapping=get_nested(config_dict, ['cambia_rules', 'allowOpponentSnapping'], CambiaRulesConfig.allowOpponentSnapping),
                     max_game_turns=get_nested(config_dict, ['cambia_rules', 'max_game_turns'], CambiaRulesConfig.max_game_turns)
                ),
                persistence=PersistenceConfig(
                     agent_data_save_path=get_nested(config_dict, ['persistence', 'agent_data_save_path'], PersistenceConfig.agent_data_save_path)
                ),
                logging=LoggingConfig(
                     log_level_file=get_nested(config_dict, ['logging', 'log_level_file'], LoggingConfig.log_level_file),
                     log_level_console=get_nested(config_dict, ['logging', 'log_level_console'], LoggingConfig.log_level_console),
                     log_dir=get_nested(config_dict, ['logging', 'log_dir'], LoggingConfig.log_dir),
                     log_file_prefix=get_nested(config_dict, ['logging', 'log_file_prefix'], LoggingConfig.log_file_prefix),
                     log_max_bytes=get_nested(config_dict, ['logging', 'log_max_bytes'], LoggingConfig.log_max_bytes),
                     log_backup_count=get_nested(config_dict, ['logging', 'log_backup_count'], LoggingConfig.log_backup_count)
                )
            )

    except FileNotFoundError:
        print(f"Warning: Config file '{config_path}' not found. Using default configuration.")
        return Config() # Return default config object
    except (TypeError, KeyError, AttributeError, yaml.YAMLError) as e: # Catch more specific errors
        print(f"Error loading or parsing config file '{config_path}': {e}. Check config structure/types.")
        print("Using default configuration.")
        return Config() # Return default config object
    except Exception as e:
        print(f"Unexpected error loading config file '{config_path}': {e}")
        # Optionally re-raise or return None/default
        # raise # Re-raise the exception for debugging
        print("Using default configuration.")
        return Config()