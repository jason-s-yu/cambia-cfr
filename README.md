# CFR Trainer for Cambia Self-Play

Trains a Cambia card game agent via Counterfactual Regret Minimization. Supports two training modes:

- Tabular CFR+ with outcome sampling (original)
- Deep CFR with external sampling and neural networks (new)

## Setup

Requires Python 3.11+. The project uses [pyenv](https://github.com/pyenv/pyenv) with [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) for environment management. A `.python-version` file pins the virtualenv name (`cfr`).

```bash
# Create the virtualenv (once)
pyenv virtualenv 3.13 cfr
pyenv activate cfr

# Install dependencies
pip install numpy pyyaml pydantic rich joblib torch typer[all]

# Install the project in editable mode (registers the `cambia` CLI command)
pip install -e .
```

The editable install (`pip install -e .`) reads `pyproject.toml` and registers the `cambia` entry point. After this, the `cambia` command is available in the virtualenv.

## Quickstart

```bash
# Tabular CFR+ training
cambia train tabular --config parallel.config.yaml

# Deep CFR training
cambia train deep --config parallel.config.yaml --steps 100
```

## Usage

All commands are available via the `cambia` CLI (registered by `pip install -e .`) or directly via `python -m src.cli`.

### Tabular CFR+ Training

```bash
cambia train tabular --config parallel.config.yaml
cambia train tabular --config serial.config.yaml --iterations 500
cambia train tabular --config parallel.config.yaml --load
cambia train tabular --config parallel.config.yaml --save-path strategy/my_run.joblib
```

Options:

- `--config`, `-c` -- path to the YAML config file (default: `config.yaml`)
- `--iterations`, `-n` -- override `cfr_training.num_iterations` from config
- `--load` -- load existing agent data from the configured save path before training
- `--save-path`, `-s` -- override `persistence.agent_data_save_path` from config

The trainer displays a Rich live dashboard showing iteration progress, worker status, infoset count, exploitability, and log sizes. Ctrl+C triggers a graceful shutdown with emergency checkpoint save.

The legacy entry point `python -m src.main_train` is still supported for backward compatibility.

### Deep CFR Training

```bash
cambia train deep --config parallel.config.yaml --steps 100
cambia train deep --config parallel.config.yaml --steps 50 --lr 0.0005 --batch-size 4096
cambia train deep --config parallel.config.yaml --checkpoint strategy/deep_cfr_checkpoint.pt --steps 50
```

Options:

- `--config`, `-c` -- path to the YAML config file (default: `config.yaml`)
- `--steps`, `-n` -- number of training steps to run
- `--checkpoint` -- resume from an existing checkpoint
- `--save-path`, `-s` -- override checkpoint save path

Deep CFR override options (override values from the `deep_cfr` config section):

- `--lr` -- learning rate
- `--batch-size` -- training batch size
- `--train-steps` -- SGD steps per iteration
- `--traversals` -- traversals per training step
- `--alpha` -- iteration weighting exponent
- `--buffer-capacity` -- reservoir buffer capacity (applies to both advantage and strategy)
- `--gpu` / `--no-gpu` -- use GPU if available

Deep CFR parameters are loaded from the `deep_cfr` section of the YAML config. CLI override options take precedence.

### Resume and Inspect

Resume training from any checkpoint (auto-detects type by file extension):

```bash
cambia resume strategy/deep_cfr_checkpoint.pt --config parallel.config.yaml --steps 50
cambia resume strategy/cambia_cfr_strategy.joblib --config parallel.config.yaml --steps 1000
```

Display checkpoint metadata:

```bash
cambia info strategy/deep_cfr_checkpoint.pt
cambia info strategy/cambia_cfr_strategy.joblib
```

### Configuration

Both training modes read from the same YAML config file. Two example configs are provided:

- `parallel.config.yaml` -- 23 workers, log archiving enabled, short max turn count
- `serial.config.yaml` -- single worker, minimal logging

#### Config sections

`cfr_training`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `num_iterations` | int | 10000 | Total training iterations (tabular) or training steps (override via `--iterations`) |
| `num_workers` | int | 23 | Number of parallel worker processes |
| `save_interval` | int | 1 | Save checkpoint every N iterations |
| `pruning_enabled` | bool | true | Enable regret-based pruning (tabular only) |
| `pruning_threshold` | float | 1e-6 | Regrets below this are treated as zero for pruning |
| `exploitability_interval` | int | 100 | Compute exploitability every N iterations |
| `exploitability_interval_seconds` | int | 7200 | Compute exploitability after N seconds elapse (fallback) |

`agent_params`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `memory_level` | int | 1 | 0 = perfect recall, 1 = event decay, 2 = event + time decay |
| `time_decay_turns` | int | 3 | Turns before time-based decay triggers (level 2 only) |

`cambia_rules`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `cards_per_player` | int | 4 | Initial hand size |
| `initial_view_count` | int | 2 | Cards peeked at game start |
| `use_jokers` | int | 2 | Number of jokers in the deck |
| `penaltyDrawCount` | int | 2 | Cards drawn on failed snap |
| `cambia_allowed_round` | int | 0 | First round Cambia can be called |
| `max_game_turns` | int | 46 | Turn limit (0 = no limit) |
| `allowDrawFromDiscardPile` | bool | false | Allow drawing from discard pile |
| `allowReplaceAbilities` | bool | false | Abilities trigger on discard-pile draws |
| `allowOpponentSnapping` | bool | false | Allow snapping opponent's cards |

`persistence`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `agent_data_save_path` | str | `strategy/...joblib` | Save/load path for tabular agent data |

`system`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `recursion_limit` | int | 10000 | Python recursion limit (set via `sys.setrecursionlimit`) |

#### `deep_cfr` (Deep CFR parameters)

These are loaded from the `deep_cfr` section of the YAML config and can be overridden via CLI options when using `cambia train deep`.

| Key | Type | Default | CLI Override | Description |
|-----|------|---------|-------------|-------------|
| `hidden_dim` | int | 256 | -- | Hidden layer width |
| `dropout` | float | 0.1 | -- | Dropout rate |
| `learning_rate` | float | 1e-3 | `--lr` | Adam optimizer learning rate |
| `batch_size` | int | 2048 | `--batch-size` | Mini-batch size for network training |
| `train_steps_per_iteration` | int | 4000 | `--train-steps` | SGD steps per training phase |
| `alpha` | float | 1.5 | `--alpha` | Iteration weighting exponent: loss weight = `(t+1)^alpha` |
| `traversals_per_step` | int | 1000 | `--traversals` | Number of game traversals per training step |
| `advantage_buffer_capacity` | int | 2,000,000 | `--buffer-capacity` | Advantage reservoir buffer max samples |
| `strategy_buffer_capacity` | int | 2,000,000 | `--buffer-capacity` | Strategy reservoir buffer max samples |
| `save_interval` | int | 10 | -- | Save checkpoint every N training steps |
| `use_gpu` | bool | false | `--gpu/--no-gpu` | Use CUDA for network training if available |

### Checkpointing and Resume

Tabular mode saves via joblib to the path in `persistence.agent_data_save_path`. Use `--load` to resume.

Deep CFR saves three files per checkpoint:

- `deep_cfr_checkpoint.pt` -- network weights, optimizer state, training metadata
- `deep_cfr_checkpoint_advantage_buffer.npz` -- advantage reservoir buffer
- `deep_cfr_checkpoint_strategy_buffer.npz` -- strategy reservoir buffer

Call `trainer.load_checkpoint(path)` before `trainer.train()` to resume. Buffer capacity can differ between runs; on load, buffers are truncated via random subsampling if the new capacity is smaller.

### Exploitability

Tabular mode computes exploitability (best-response traversal) at intervals configured by `exploitability_interval` and `exploitability_interval_seconds`. This is a full game tree traversal and can take hours for large state spaces. The `analysis.exploitability_num_workers` setting controls parallelism within each best-response computation.

Deep CFR does not yet have integrated exploitability computation. Monitor convergence via the advantage and strategy network loss values logged each training step.

## Project Structure

Key source files:

```
src/
  cli.py                  -- Typer CLI entry point (cambia command)
  main_train.py           -- Training orchestration and infrastructure
  config.py               -- Config loading from YAML (includes DeepCfrConfig)
  constants.py            -- Game actions, card buckets, enums
  encoding.py             -- InfosetKey -> fixed-size tensor encoding (Deep CFR)
  networks.py             -- AdvantageNetwork, StrategyNetwork (PyTorch)
  reservoir.py            -- ReservoirBuffer for training samples (Deep CFR)
  agent_state.py          -- Agent belief state and observation model
  abstraction.py          -- CardBucket mapping
  utils.py                -- InfosetKey, WorkerResult, helper functions
  persistence.py          -- Tabular save/load (joblib)
  analysis_tools.py       -- Exploitability via best-response traversal
  game/
    engine.py             -- CambiaGameState (apply/undo game actions)
    _ability_mixin.py     -- Card ability resolution
    _snap_mixin.py        -- Snap phase logic
    _query_mixin.py       -- Legal action generation
  cfr/
    trainer.py            -- Tabular CFR+ trainer (orchestrator)
    deep_trainer.py       -- Deep CFR trainer (orchestrator)
    worker.py             -- Tabular worker (outcome sampling traversal)
    deep_worker.py        -- Deep CFR worker (external sampling traversal)
    training_loop_mixin.py -- Tabular training loop with worker pool
    data_manager_mixin.py -- Tabular merge step
    recursion_mixin.py    -- Observation helpers (delegates to worker.py)
```

Documentation in `docs/`:

- `docs/dcfr_implementation_plan.md` -- Full design document and audit
- `docs/architecture.md` -- Deep CFR pipeline overview
- `docs/deep_cfr_modules.md` -- Module reference for new files
- `docs/changelog.md` -- Summary of all changes by phase
- `docs/phase0_fixes.md` -- Details on engine bug fixes
- `docs/implementation_notes.md` -- Gaps between plan and implementation

## Development

The editable install (`pip install -e .`) means Python source changes take effect immediately -- no rebuild step. Edit any `.py` file and re-run `cambia` or `python -m src.cli` to pick up changes.

```bash
# Run the CLI directly without the entry point (equivalent to `cambia`)
python -m src.cli train deep --config parallel.config.yaml --steps 2

# Run the legacy entry point (tabular only)
python -m src.main_train --config parallel.config.yaml --iterations 2
```

Re-run `pip install -e .` only if you change `pyproject.toml` (e.g. adding entry points or metadata).

### Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test module
pytest tests/test_encoding.py -v
```

Test modules:

- `test_encoding.py` -- feature encoding and action space mapping
- `test_networks.py` -- network forward passes, masking, strategy conversion
- `test_reservoir.py` -- reservoir buffer sampling, save/load, resize
- `test_phase0_regression.py` -- regression tests for Phase 0 engine bug fixes

## Dependencies

- `numpy` -- array operations
- `torch` (PyTorch) -- neural networks (Deep CFR only; CPU build sufficient)
- `pyyaml` -- config file parsing
- `pydantic` -- config validation
- `rich` -- terminal live display
- `joblib` -- tabular strategy persistence
- `typer[all]` -- CLI framework with Rich integration

## Known Limitations

- The tabular `InfosetKey` does not encode the drawn card at `POST_DRAW` decision points. The Deep CFR encoding fixes this.
- Max game turn count (`max_game_turns`) must be set to prevent infinite games during training. The parallel config defaults to 46.
- Exploitability computation is only available for tabular mode.
- `MAX_HAND` is fixed at 6 in `encoding.py`. Hands larger than 6 cards are clamped.
