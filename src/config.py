# src/config.py
from dataclasses import dataclass, field
import yaml
from typing import List, Dict, TypeVar

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
     return current


@dataclass
class ApiConfig:
    base_url: str = "http://localhost:8080" # Example default
    auth: Dict[str, str] = field(default_factory=dict) # e.g., {"email": "...", "password": "..."} or {"token_file": "..."}

@dataclass
class SystemConfig:
    recursion_limit: int = 1000 # Limit for recursion depth

@dataclass
class CfrTrainingConfig:
    num_iterations: int = 10000
    save_interval: int = 1000
    pruning_enabled: bool = True # Enable Regret-Based Pruning
    pruning_threshold: float = 1e-6 # Regrets below this are considered zero for pruning
    exploitability_interval: int = 1000 # How often (in iterations) to calculate exploitability

@dataclass
class CfrPlusParamsConfig:
    weighted_averaging_enabled: bool = True
    averaging_delay: int = 0 # Start averaging from iteration 0 (weight = max(0, t - d))

@dataclass
class AgentParamsConfig:
    memory_level: int = 0 # 0: Perfect Recall, 1: Event Decay, 2: Event+Time Decay
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
    agent_data_save_path: str = "cfr_agent_data.joblib"

@dataclass
class LoggingConfig:
    log_level_file: str = "DEBUG" # Logging level for the file
    log_level_console: str = "WARNING" # Logging level for the console
    log_dir: str = "logs" # Directory to store log files
    log_file_prefix: str = "cambia" # Prefix for timestamped log files

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

def load_config(config_path: str = "config.yaml") -> Config:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            if config_dict is None: config_dict = {}

            # Using helper for safer nested access with defaults
            return Config(
                api=ApiConfig(
                    base_url=get_nested(config_dict, ['api', 'base_url'], "http://localhost:8080"),
                    auth=get_nested(config_dict, ['api', 'auth'], {})
                ),
                system=SystemConfig(
                    recursion_limit=get_nested(config_dict, ['system', 'recursion_limit'], 1000)
                ),
                cfr_training=CfrTrainingConfig(
                    num_iterations=get_nested(config_dict, ['cfr_training', 'num_iterations'], 10000),
                    save_interval=get_nested(config_dict, ['cfr_training', 'save_interval'], 1000),
                    pruning_enabled=get_nested(config_dict, ['cfr_training', 'pruning_enabled'], True),
                    pruning_threshold=get_nested(config_dict, ['cfr_training', 'pruning_threshold'], 1e-6),
                    exploitability_interval=get_nested(config_dict, ['cfr_training', 'exploitability_interval'], 1000)
                ),
                cfr_plus_params=CfrPlusParamsConfig(
                     weighted_averaging_enabled=get_nested(config_dict, ['cfr_plus_params', 'weighted_averaging_enabled'], True),
                     averaging_delay=get_nested(config_dict, ['cfr_plus_params', 'averaging_delay'], 0)
                ),
                agent_params=AgentParamsConfig(
                     memory_level=get_nested(config_dict, ['agent_params', 'memory_level'], 0),
                     time_decay_turns=get_nested(config_dict, ['agent_params', 'time_decay_turns'], 3)
                ),
                cambia_rules=CambiaRulesConfig(
                     allowDrawFromDiscardPile=get_nested(config_dict, ['cambia_rules', 'allowDrawFromDiscardPile'], False),
                     allowReplaceAbilities=get_nested(config_dict, ['cambia_rules', 'allowReplaceAbilities'], False),
                     snapRace=get_nested(config_dict, ['cambia_rules', 'snapRace'], False),
                     penaltyDrawCount=get_nested(config_dict, ['cambia_rules', 'penaltyDrawCount'], 2),
                     use_jokers=get_nested(config_dict, ['cambia_rules', 'use_jokers'], 2),
                     cards_per_player=get_nested(config_dict, ['cambia_rules', 'cards_per_player'], 4),
                     initial_view_count=get_nested(config_dict, ['cambia_rules', 'initial_view_count'], 2),
                     cambia_allowed_round=get_nested(config_dict, ['cambia_rules', 'cambia_allowed_round'], 0),
                     allowOpponentSnapping=get_nested(config_dict, ['cambia_rules', 'allowOpponentSnapping'], False),
                     max_game_turns=get_nested(config_dict, ['cambia_rules', 'max_game_turns'], 300)
                ),
                persistence=PersistenceConfig(
                     agent_data_save_path=get_nested(config_dict, ['persistence', 'agent_data_save_path'], "cfr_agent_data.joblib")
                ),
                logging=LoggingConfig(
                     log_level_file=get_nested(config_dict, ['logging', 'log_level_file'], 'DEBUG'),
                     log_level_console=get_nested(config_dict, ['logging', 'log_level_console'], 'WARNING'),
                     log_dir=get_nested(config_dict, ['logging', 'log_dir'], 'logs'),
                     log_file_prefix=get_nested(config_dict, ['logging', 'log_file_prefix'], 'cambia')
                )
            )

    except FileNotFoundError:
        print(f"Warning: Config file '{config_path}' not found. Using default configuration.")
        return Config() # Return default config object
    except (TypeError, KeyError, AttributeError) as e: # Catch more specific errors from get_nested/dataclass init
        print(f"Error loading config file '{config_path}': {e}. Check config structure/types.")
        print("Using default configuration.")
        return Config() # Return default config object
    except Exception as e:
        print(f"Unexpected error loading config file '{config_path}': {e}")
        print("Using default configuration.")
        return Config() # Return default config object