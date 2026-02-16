"""
tests/conftest.py

Shared fixtures and bootstrap logic for all tests.

The project's config.py is currently incomplete (missing dataclass import
and several Config sub-classes). We inject a minimal stub before any
src.* imports so that modules like agent_state and encoding can be loaded.
"""

import sys
import types
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# --- Stub out src.config ---
# Only inject the stub if config hasn't been loaded yet or is broken.
_config_mod = sys.modules.get("src.config")
if _config_mod is None or not hasattr(_config_mod, "Config"):
    _config_stub = types.ModuleType("src.config")

    class _StubConfig:
        """Minimal placeholder for Config and its sub-classes."""
        pass

    # CambiaRulesConfig needs real defaults because CambiaGameState reads them
    class _CambiaRulesConfig:
        """Stub for CambiaRulesConfig with real game defaults."""
        allowDrawFromDiscardPile: bool = False
        allowReplaceAbilities: bool = False
        snapRace: bool = False
        penaltyDrawCount: int = 2
        use_jokers: int = 2
        cards_per_player: int = 4
        initial_view_count: int = 2
        cambia_allowed_round: int = 0
        allowOpponentSnapping: bool = False
        max_game_turns: int = 300

    _config_stub.Config = _StubConfig
    _config_stub.CambiaRulesConfig = _CambiaRulesConfig
    _config_stub.CfrTrainingConfig = _StubConfig
    _config_stub.AgentParamsConfig = _StubConfig
    _config_stub.ApiConfig = _StubConfig
    _config_stub.SystemConfig = _StubConfig
    _config_stub.CfrPlusParamsConfig = _StubConfig
    _config_stub.PersistenceConfig = _StubConfig
    _config_stub.LoggingConfig = _StubConfig
    _config_stub.AgentsConfig = _StubConfig
    _config_stub.AnalysisConfig = _StubConfig

    sys.modules["src.config"] = _config_stub
