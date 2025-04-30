# src/config.py
from dataclasses import dataclass, field
import yaml
from typing import Optional, Dict, Any

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
            if config_dict is None: config_dict = {} # Handle empty config file

            # Reconstruct nested dataclasses, ensuring all keys are handled
            system_cfg = config_dict.get('system', {})
            cfr_training_cfg = config_dict.get('cfr_training', {})
            cambia_rules_cfg = config_dict.get('cambia_rules', {})
            logging_cfg = config_dict.get('logging', {})
            api_cfg = config_dict.get('api', {})
            cfr_plus_cfg = config_dict.get('cfr_plus_params', {})
            agent_params_cfg = config_dict.get('agent_params', {})
            persistence_cfg = config_dict.get('persistence', {})

            return Config(
                api=ApiConfig(**api_cfg),
                system=SystemConfig(
                    recursion_limit=system_cfg.get('recursion_limit', 1000)
                ),
                cfr_training=CfrTrainingConfig(
                    num_iterations=cfr_training_cfg.get('num_iterations', 10000),
                    save_interval=cfr_training_cfg.get('save_interval', 1000),
                    pruning_enabled=cfr_training_cfg.get('pruning_enabled', True),
                    pruning_threshold=cfr_training_cfg.get('pruning_threshold', 1e-6)
                ),
                cfr_plus_params=CfrPlusParamsConfig(**cfr_plus_cfg),
                agent_params=AgentParamsConfig(**agent_params_cfg),
                cambia_rules=CambiaRulesConfig(
                     allowDrawFromDiscardPile=cambia_rules_cfg.get('allowDrawFromDiscardPile', False),
                     allowReplaceAbilities=cambia_rules_cfg.get('allowReplaceAbilities', False),
                     snapRace=cambia_rules_cfg.get('snapRace', False),
                     penaltyDrawCount=cambia_rules_cfg.get('penaltyDrawCount', 2),
                     use_jokers=cambia_rules_cfg.get('use_jokers', 2),
                     cards_per_player=cambia_rules_cfg.get('cards_per_player', 4),
                     initial_view_count=cambia_rules_cfg.get('initial_view_count', 2),
                     cambia_allowed_round=cambia_rules_cfg.get('cambia_allowed_round', 0),
                     allowOpponentSnapping=cambia_rules_cfg.get('allowOpponentSnapping', False),
                     max_game_turns=cambia_rules_cfg.get('max_game_turns', 300)
                ),
                persistence=PersistenceConfig(**persistence_cfg),
                logging=LoggingConfig(
                     log_level_file=logging_cfg.get('log_level_file', 'DEBUG'),
                     log_level_console=logging_cfg.get('log_level_console', 'WARNING'),
                     log_dir=logging_cfg.get('log_dir', 'logs'),
                     log_file_prefix=logging_cfg.get('log_file_prefix', 'cambia')
                )
            )

    except FileNotFoundError:
        print(f"Warning: Config file '{config_path}' not found. Using default configuration.")
        return Config()
    except TypeError as e:
        print(f"Error loading config file '{config_path}': {e}. Check config keys and structure.")
        print("Using default configuration.")
        return Config()
    except Exception as e:
        print(f"Unexpected error loading config file '{config_path}': {e}")
        print("Using default configuration.")
        return Config()