# src/config.py
from dataclasses import dataclass, field
import yaml
from typing import Optional, Dict, Any

@dataclass
class ApiConfig:
    base_url: str = "http://localhost:8080" # Example default
    auth: Dict[str, str] = field(default_factory=dict) # e.g., {"email": "...", "password": "..."} or {"token_file": "..."}

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
    # Add any other house rules relevant to the engine here

@dataclass
class PersistenceConfig:
    agent_data_save_path: str = "cfr_agent_data.joblib"

@dataclass
class LoggingConfig:
    log_level: str = "INFO"
    log_dir: str = "logs" # Directory to store log files
    log_file_prefix: str = "cambia" # Prefix for timestamped log files
    # log_file is now dynamically generated

@dataclass
class Config:
    api: ApiConfig = field(default_factory=ApiConfig)
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

            # Basic manual reconstruction for nested dataclasses
            return Config(
                api=ApiConfig(**config_dict.get('api', {})),
                cfr_training=CfrTrainingConfig(**config_dict.get('cfr_training', {})),
                cfr_plus_params=CfrPlusParamsConfig(**config_dict.get('cfr_plus_params', {})),
                agent_params=AgentParamsConfig(**config_dict.get('agent_params', {})),
                cambia_rules=CambiaRulesConfig(
                     allowDrawFromDiscardPile=config_dict.get('cambia_rules', {}).get('allowDrawFromDiscardPile', False),
                     allowReplaceAbilities=config_dict.get('cambia_rules', {}).get('allowReplaceAbilities', False),
                     snapRace=config_dict.get('cambia_rules', {}).get('snapRace', False),
                     penaltyDrawCount=config_dict.get('cambia_rules', {}).get('penaltyDrawCount', 2),
                     use_jokers=config_dict.get('cambia_rules', {}).get('use_jokers', 2),
                     cards_per_player=config_dict.get('cambia_rules', {}).get('cards_per_player', 4),
                     initial_view_count=config_dict.get('cambia_rules', {}).get('initial_view_count', 2),
                     cambia_allowed_round=config_dict.get('cambia_rules', {}).get('cambia_allowed_round', 0),
                     allowOpponentSnapping=config_dict.get('cambia_rules', {}).get('allowOpponentSnapping', False)
                ),
                persistence=PersistenceConfig(**config_dict.get('persistence', {})),
                # Correctly pass keyword arguments for LoggingConfig
                logging=LoggingConfig(
                     log_level=config_dict.get('logging', {}).get('log_level', 'INFO'),
                     log_dir=config_dict.get('logging', {}).get('log_dir', 'logs'),
                     log_file_prefix=config_dict.get('logging', {}).get('log_file_prefix', 'cambia')
                )
            )

    except FileNotFoundError:
        print(f"Warning: Config file '{config_path}' not found. Using default configuration.")
        return Config()
    except TypeError as e:
        # Catch TypeError specifically and print more helpful message
        print(f"Error loading config file '{config_path}': {e}. Check config keys and structure.")
        print("Using default configuration.")
        return Config()
    except Exception as e:
        print(f"Unexpected error loading config file '{config_path}': {e}")
        print("Using default configuration.")
        return Config()