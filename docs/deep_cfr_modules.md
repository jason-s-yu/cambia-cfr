# Deep CFR Module Reference

Reference documentation for the five new modules introduced by the Deep CFR refactor.

## `src/encoding.py`

Converts `AgentState` + legal actions into fixed-size numpy tensors for neural network input.

### Constants

- `MAX_HAND = 6` -- maximum hand slots encoded. Hands exceeding this are clamped.
- `SLOT_ENCODING_DIM = 15` -- one-hot dimension per hand slot (10 `CardBucket` + 3 `DecayCategory` + UNKNOWN + EMPTY).
- `INPUT_DIM = 222` -- total feature vector length.
- `NUM_ACTIONS = 146` -- total fixed action space size.

### Feature Vector Layout

| Offset  | Feature                    | Dimensions  | Encoding                    |
| ------- | -------------------------- | ----------- | --------------------------- |
| 0-89    | Own hand (6 slots)         | 6 x 15 = 90 | One-hot per slot            |
| 90-179  | Opponent beliefs (6 slots) | 6 x 15 = 90 | One-hot per slot            |
| 180     | Own card count             | 1           | Normalized scalar /6        |
| 181     | Opponent card count        | 1           | Normalized scalar /6        |
| 182-192 | Drawn card bucket          | 11          | One-hot (10 buckets + NONE) |
| 193-202 | Discard top bucket         | 10          | One-hot                     |
| 203-206 | Stockpile estimate         | 4           | One-hot                     |
| 207-212 | Game phase                 | 6           | One-hot                     |
| 213-218 | Decision context           | 6           | One-hot                     |
| 219-221 | Cambia caller              | 3           | One-hot (SELF/OPP/NONE)     |

Each hand/belief slot uses a unified 15-dim one-hot encoding:

- Indices 0-8: `CardBucket` values (ZERO through HIGH_KING)
- Index 9: unused (gap between bucket 8 and decay 10)
- Indices 10-12: `DecayCategory` values (LIKELY_LOW, LIKELY_MID, LIKELY_HIGH)
- Index 13: UNKNOWN (shared by both `CardBucket.UNKNOWN` and `DecayCategory.UNKNOWN`)
- Index 14: EMPTY (slot does not exist)

### Action Index Layout

| Range   | Action Type                              | Parameters        |
| ------- | ---------------------------------------- | ----------------- |
| 0       | `ActionDrawStockpile`                    | --                |
| 1       | `ActionDrawDiscard`                      | --                |
| 2       | `ActionCallCambia`                       | --                |
| 3       | `ActionDiscard(use_ability=False)`       | --                |
| 4       | `ActionDiscard(use_ability=True)`        | --                |
| 5-10    | `ActionReplace(idx)`                     | idx 0-5           |
| 11-16   | `ActionAbilityPeekOwnSelect(idx)`        | idx 0-5           |
| 17-22   | `ActionAbilityPeekOtherSelect(idx)`      | idx 0-5           |
| 23-58   | `ActionAbilityBlindSwapSelect(own, opp)` | own*6 + opp       |
| 59-94   | `ActionAbilityKingLookSelect(own, opp)`  | own*6 + opp       |
| 95-96   | `ActionAbilityKingSwapDecision(bool)`    | False=95, True=96 |
| 97      | `ActionPassSnap`                         | --                |
| 98-103  | `ActionSnapOwn(idx)`                     | idx 0-5           |
| 104-109 | `ActionSnapOpponent(idx)`                | idx 0-5           |
| 110-145 | `ActionSnapOpponentMove(own, slot)`      | own*6 + slot      |

### Public Functions

```python
def encode_infoset(
    agent_state: AgentState,
    decision_context: DecisionContext,
    drawn_card_bucket: Optional[CardBucket] = None,
) -> np.ndarray:
```

Encodes an agent's information set into a `(222,)` float32 array. The `drawn_card_bucket` parameter should be provided at `POST_DRAW` decision points to encode the drawn card's identity -- this was missing from the tabular `InfosetKey` and is a key improvement.

```python
def action_to_index(action: GameAction) -> int:
```

Maps a `GameAction` NamedTuple to its fixed index in `[0, 146)`. Raises `ValueError` for unrecognized types or out-of-range hand indices.

```python
def index_to_action(index: int, legal_actions: List[GameAction]) -> GameAction:
```

Reverse mapping: finds the legal action matching a given index by scanning the list and calling `action_to_index` on each. Raises `ValueError` if no match is found.

```python
def encode_action_mask(legal_actions: List[GameAction]) -> np.ndarray:
```

Creates a `(146,)` boolean mask with `True` for each legal action's index. Actions with hand indices beyond `MAX_HAND` are silently skipped.

### Usage

```python
from src.encoding import encode_infoset, encode_action_mask, action_to_index
from src.constants import DecisionContext

features = encode_infoset(agent_state, DecisionContext.POST_DRAW, drawn_card_bucket=bucket)
mask = encode_action_mask(legal_actions)
idx = action_to_index(chosen_action)
```

### Design Decisions

- The drawn card bucket is encoded as a separate 11-dim one-hot rather than folding it into the hand encoding. This makes the feature vector layout static regardless of whether a card has been drawn.
- `index_to_action` uses linear scan over legal actions rather than building a reverse lookup table, since legal action lists are small (typically 5-15 actions).
- Hand slots beyond `MAX_HAND` are silently dropped in `encode_action_mask`. The `own_card_count` scalar retains the true count as a signal to the network.

---

## `src/networks.py`

PyTorch `nn.Module` definitions for the advantage and strategy networks.

### Classes

```python
class AdvantageNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = 222,
        hidden_dim: int = 256,
        output_dim: int = 146,
        dropout: float = 0.1,
    )

    def forward(self, features: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
```

Predicts per-action advantage (regret) values. The forward pass masks illegal actions to `-inf`. Input shapes: `features (batch, 222)`, `action_mask (batch, 146)` bool. Output: `(batch, 146)` float.

```python
class StrategyNetwork(nn.Module):
    def __init__(...)  # Same signature as AdvantageNetwork

    def forward(self, features: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
```

Predicts per-action strategy probabilities. Masks illegal actions to `-inf` before softmax, producing a valid probability distribution. Includes NaN guard for the case where all actions are masked.

Both networks share the same architecture:

```
Linear(input_dim, 256) -> ReLU -> Dropout(0.1)
-> Linear(256, 256) -> ReLU -> Dropout(0.1)
-> Linear(256, 128) -> ReLU -> Linear(128, output_dim)
```

Weight initialization: Kaiming normal for linear layers, zeros for biases.

### Standalone Function

```python
def get_strategy_from_advantages(
    advantages: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
```

Converts advantage network output into a strategy distribution using the regret matching pattern: `ReLU(advantages) -> mask illegal -> normalize`. Falls back to uniform over legal actions when all advantages are non-positive.

This function is used during traversal (not training) to derive the current strategy from the advantage network, replacing the tabular `regret_sum -> RM+` lookup.

### Design Decisions

- The advantage network uses `-inf` masking rather than zeroing because downstream consumers (the worker) need to distinguish "predicted zero regret" from "illegal action."
- The strategy network uses softmax (during training) rather than ReLU+normalize because it's trained to directly predict the average strategy, not instantaneous advantages.
- `get_strategy_from_advantages` uses ReLU+normalize (not softmax) to match the RM+ convergence guarantee: only actions with positive predicted advantage get probability mass.

---

## `src/reservoir.py`

Fixed-capacity reservoir sampling buffers for Deep CFR training data.

### Classes

```python
@dataclass
class ReservoirSample:
    features: np.ndarray      # (222,) float32
    target: np.ndarray         # (146,) float32 -- regrets or strategy
    action_mask: np.ndarray    # (146,) bool
    iteration: int             # CFR iteration number for t^alpha weighting
    infoset_key_raw: Optional[Tuple] = None  # Debugging metadata
```

A single training sample. The `iteration` field is used during training to compute `(t+1)^alpha` weights in the loss function.

```python
class ReservoirBuffer:
    def __init__(self, capacity: int = 2_000_000)
    def __len__(self) -> int
    def add(self, sample: ReservoirSample)
    def sample_batch(self, batch_size: int) -> List[ReservoirSample]
    def save(self, path: str)
    def load(self, path: str)
    def resize(self, new_capacity: int)
    def clear(self)
```

Implements Vitter's Algorithm R: each new sample has `capacity / seen_count` probability of entering the buffer, guaranteeing a uniform random sample of all samples ever added. The `seen_count` is tracked separately from buffer length to maintain this guarantee after the buffer fills.

`save()` and `load()` use `np.savez_compressed`, storing features/targets/masks/iterations as stacked arrays plus a metadata array for `seen_count` and `capacity`. On load, if the saved capacity differs from the current instance's capacity, the buffer is truncated via random subsampling.

`resize()` adjusts capacity at runtime. Shrinking randomly subsamples the buffer. Growing just updates the capacity limit.

### Standalone Function

```python
def samples_to_tensors(
    samples: List[ReservoirSample]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

Batch-converts a list of samples into stacked numpy arrays: `(N, 222)` features, `(N, 146)` targets, `(N, 146)` masks, `(N,)` iterations. Returns empty arrays with correct shapes when given an empty list.

### Memory Estimates

| Capacity     | Approx. memory per buffer |
| ------------ | ------------------------- |
| 100K         | ~60 MB                    |
| 500K         | ~300 MB                   |
| 2M (default) | ~1.2 GB                   |
| 5M           | ~3 GB                     |

### Design Decisions

- Samples are stored as a `List[ReservoirSample]` rather than pre-allocated numpy arrays. This simplifies the reservoir replacement logic at the cost of slightly higher memory overhead from Python object bookkeeping.
- The `infoset_key_raw` field is not persisted during save/load. It exists only for in-session debugging.
- `sample_batch` draws without replacement, which is standard for SGD mini-batches.

---

## `src/cfr/deep_trainer.py`

Orchestrates the Deep CFR training loop: traversal scheduling, sample collection, and network training.

### Classes

```python
@dataclass
class DeepCFRConfig:
    input_dim: int = 222
    hidden_dim: int = 256
    output_dim: int = 146
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
```

All Deep CFR hyperparameters in one place. The `alpha` parameter controls iteration weighting in the loss function (1.0 = linear, 1.5 = default, 2.0 = quadratic).

```python
@classmethod
def DeepCFRConfig.from_yaml_config(cls, config: Config, **overrides) -> DeepCFRConfig:
```

Constructs a `DeepCFRConfig` from a `Config` object's `deep_cfr` field (a `DeepCfrConfig` dataclass loaded from YAML), applying any CLI overrides. Override keys with `None` values are ignored. This bridges the YAML-facing `DeepCfrConfig` (in `config.py`) to the runtime `DeepCFRConfig` (in `deep_trainer.py`).

```python
class DeepCFRTrainer:
    def __init__(
        self, config: Config,
        deep_cfr_config: Optional[DeepCFRConfig] = None,
        run_log_dir: Optional[str] = None,
        run_timestamp: Optional[str] = None,
        shutdown_event: Optional[threading.Event] = None,
        progress_queue: Optional[ProgressQueue] = None,
        live_display_manager: Optional[LiveDisplayManager] = None,
        archive_queue: Optional[Union[queue.Queue, multiprocessing.Queue]] = None,
    )

    def train(self, num_training_steps: Optional[int] = None)
    def save_checkpoint(self, filepath: Optional[str] = None)
    def load_checkpoint(self, filepath: Optional[str] = None)
    def get_strategy_network(self) -> StrategyNetwork
    def get_advantage_network(self) -> AdvantageNetwork
```

### Training Loop

Each training step:

1. Serialize advantage network weights as numpy arrays for pickle-friendly transfer to workers.
2. Dispatch `traversals_per_step` traversals across the worker pool. Workers run `run_deep_cfr_worker` and return `DeepCFRWorkerResult` containing advantage and strategy `ReservoirSample` lists.
3. Add all collected samples to the advantage and strategy `ReservoirBuffer` instances.
4. Train the advantage network on the advantage buffer using weighted MSE loss: `weight = ((t+1)^alpha) / mean_weight`, applied per-sample. Loss is computed only over legal actions (masked).
5. Train the strategy network on the strategy buffer with the same loss formulation.
6. Save checkpoint at configured intervals.

The trainer supports both sequential execution (1 worker) and parallel execution via `multiprocessing.Pool`. It handles graceful shutdown by saving a checkpoint before re-raising the shutdown exception.

### Checkpoint Format

The main checkpoint (`.pt` file via `torch.save`) contains:

- Network state dicts and optimizer state dicts
- Training step, total traversals, iteration count
- `DeepCFRConfig` values
- Loss history for both networks
- Paths to the reservoir buffer `.npz` files (saved alongside)

### Design Decisions

- Network weights are serialized to numpy arrays (`v.cpu().numpy()`) before passing to workers, avoiding the need for workers to import CUDA or handle device placement.
- The `alpha` weighting uses `(t+1)^alpha` (not `t^alpha`) to avoid zero weight for iteration 0.
- Weights are normalized by their mean within each batch to prevent the loss magnitude from growing as iteration numbers increase.
- Gradient clipping (`max_norm=1.0`) is applied to both networks for training stability.

---

## `src/cfr/deep_worker.py`

Implements the Deep CFR worker process using External Sampling MCCFR.

### Classes

```python
@dataclass
class DeepCFRWorkerResult:
    advantage_samples: List[ReservoirSample]
    strategy_samples: List[ReservoirSample]
    stats: WorkerStats
    simulation_nodes: List[SimulationNodeData]
    final_utility: Optional[List[float]]
```

Replaces the tabular `WorkerResult` (which contained regret/strategy/reach-prob dicts). The samples are collected during traversal and returned to the trainer for reservoir buffer insertion.

### Public Functions

```python
def run_deep_cfr_worker(
    worker_args: Tuple[int, Config, Optional[Dict], Dict, Optional[queue.Queue],
                       Optional[Any], int, str, str]
) -> Optional[DeepCFRWorkerResult]:
```

Top-level worker entry point, invoked by the trainer via `multiprocessing.Pool.map_async`. Sets up per-worker logging, initializes game state and agent states, deserializes network weights, and runs one external sampling traversal.

The `worker_args` tuple contains: `(iteration, config, network_weights_serialized, network_config, progress_queue, archive_queue, worker_id, run_log_dir, run_timestamp)`.

### Internal Functions

`_deep_traverse(...)` -- the recursive traversal function implementing External Sampling:

- At traverser nodes: loops over all legal actions, applies each with `game_state.apply_action()`, recurses, undoes via the returned undo closure. Computes exact regrets: `regret(a) = v(a)[player] - (strategy . action_values)[player]`. Stores an advantage `ReservoirSample`.
- At opponent nodes: samples one action from the current strategy, recurses once. Stores a strategy `ReservoirSample` with the current strategy as the target.
- At terminal/depth-limit nodes: returns the utility vector.

`_get_strategy_from_network(...)` -- loads serialized weights into a temporary `AdvantageNetwork`, runs a forward pass, applies `get_strategy_from_advantages` (ReLU + normalize). Returns a `(146,)` numpy strategy array. The full-size strategy is then mapped to local action indices for the traversal.

### How It Connects to Other Modules

```text
deep_worker.py
  ├── encoding.py: encode_infoset(), encode_action_mask(), action_to_index()
  ├── networks.py: AdvantageNetwork, get_strategy_from_advantages()
  ├── reservoir.py: ReservoirSample
  ├── worker.py: _create_observation(), _filter_observation() (reused)
  ├── agent_state.py: AgentState.clone(), AgentState.update()
  └── game/engine.py: CambiaGameState.apply_action(), .get_legal_actions(), .is_terminal()
```

### Design Decisions

- The worker reconstructs an `AdvantageNetwork` instance on every `_get_strategy_from_network` call. This is intentional for process isolation -- workers do not maintain persistent network state, only the serialized weights passed by the trainer.
- The updating player alternates each iteration (`iteration % NUM_PLAYERS`), matching the standard Deep CFR protocol where each iteration updates one player's advantage network.
- Agent state updates reuse the tabular worker's `_create_observation` and `_filter_observation` functions directly. This avoids duplicating the observation construction logic.
- The drawn card bucket is read from `game_state.pending_action_data["drawn_card"]` and encoded via `get_card_bucket()` at `POST_DRAW` decision points. This fixes the tabular implementation's blind spot where the agent couldn't distinguish drawn cards.
