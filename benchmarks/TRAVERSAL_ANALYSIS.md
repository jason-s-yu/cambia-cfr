# Deep CFR Traversal Performance Analysis

Generated: 2026-02-16

## Critical Findings

### 1. Logging Was 65% of Traversal Time (FIXED)

The worker root logger was hardcoded to DEBUG level (`deep_worker.py:795`), causing 1.4M+ LogRecord objects per traversal even when the handler filtered them at WARNING. Each LogRecord involves:
- `findCaller()`: Stack frame inspection (~7s per traversal)
- `LogRecord.__init__`: Object creation (~11s per traversal)
- `flush()`: I/O (~6s per traversal when handler is DEBUG)

**Fix**: Set root logger level to match the handler's configured level. Result: **7.4x speedup**.

### 2. Agent State Deep Copy Was 20% (FIXED)

`agent_state.clone()` used `copy.deepcopy()` on three small dicts (max 6 entries each). deepcopy's recursive type inspection is massive overkill. Manual dict copy + dataclass construction gives the same result.

**Fix**: Replace `copy.deepcopy()` with manual shallow copy. Combined result: **~11x speedup per node**.

### 3. Game Tree Explodes Exponentially

| max_game_turns | Nodes | Time (optimized) | Notes |
|---------------|-------|-------------------|-------|
| 5 | 769 | 0.1s | Tiny tree |
| 8 | 35K | 10s | |
| 10 | 55K | 13s | |
| 12 | 60K | 6s | (favorable seed) |
| **15** | **7.5M** | **958s** | 126x jump from 12 |
| 46 (production) | ~10^9+ (est) | hours | Too large to complete |

External sampling enumerates ALL legal actions at traverser nodes. When ability chains and snap phases open up around turn 12-15, the branching factor spikes dramatically.

### 4. Network Inference Is NOT the Bottleneck

Without network weights (uniform strategy = no inference), traversals are already extremely slow. The Python game engine (apply_action, undo, get_legal_actions, agent state management) dominates.

## Optimizations Applied

1. **Network instantiation fix** (`deep_worker.py`): Create AdvantageNetwork once per worker, not once per node. Eliminates millions of PyTorch module constructions per step.

2. **Logger level fix** (`deep_worker.py`): Root logger level matches handler level. Eliminates 1.4M+ LogRecord creations per traversal.

3. **Clone optimization** (`agent_state.py`): Manual copy instead of deepcopy. Eliminates 8M+ recursive copy calls per traversal.

4. **Worker log config** (`deep_train.yaml`): Changed from `sequential_rules: ["DEBUG"]` to `default_level: "WARNING"`.

## Profiled Breakdown (10-turn game, after fixes)

| Component | Time | % |
|-----------|------|---|
| `_deep_traverse` (recursion logic) | 2.8s | 16% |
| `agent_state.update` | 1.2s | 7% |
| `logger.debug` (check+return) | 1.0s | 6% |
| `agent_state.clone` | 0.6s | 3% |
| `_calculate_final_scores` | 0.5s | 3% |
| `_get_legal_pending_actions` | 0.5s | 3% |
| `encode_infoset` | 0.3s | 2% |
| `apply_action` | 0.3s | 2% |
| Other | 10.3s | 58% |
| **Total** | **17.5s** | **100%** |

## Next-Level Optimizations (Research Summary)

### Priority 1: Reduce max_game_turns / Use Outcome Sampling
- External sampling is infeasible for 46-turn games — the tree is too large
- **Outcome sampling**: Sample ONE action at ALL nodes (traverser included), use importance sampling correction. Per-traversal cost = O(depth) instead of O(branching^depth). 1 week to implement.
- **Lower max_game_turns**: Consider 15-20 turns with score estimation for unfinished games.

### Priority 2: Batched Inference Server (when network weights matter)
- KataGo/AlphaZero pattern: Multiple traversal coroutines → shared GPU batch inference
- Not yet needed (game logic dominates, not inference), but will matter when game engine is faster
- 2-3 weeks to implement

### Priority 3: C++ Game Engine
- Python game engine is the fundamental bottleneck: ~5K lines of Python with OOP, deepcopy, isinstance chains
- C++ rewrite typically gives 20-50x speedup
- 3-6 weeks to implement, proven approach (Libratus, Pluribus, OpenSpiel)

### Priority 4: NNUE-style Incremental Inference
- Cache first-layer activations, delta-update on state changes
- Only ~1.3x speedup for this network architecture (first layer is 25% of compute)
- Low priority, but essentially free if C++ engine is built

## Estimated Impact Stack

| Optimization | Speedup | Cumulative | Status |
|-------------|---------|------------|--------|
| Logging fix | 7.4x | 7.4x | DONE |
| Clone fix | 1.5x | 11x | DONE |
| Network instantiation | ~1.2x (later iterations) | ~13x | DONE |
| Outcome sampling | 5-20x | 65-260x | TODO |
| C++ game engine | 20-50x | 1300-13000x | TODO |
| Batched GPU inference | 3-8x | 3900-100000x | TODO |

With outcome sampling alone, the 6-hour training step could drop to 20-70 minutes.
With outcome sampling + C++ engine, it could drop to 1-3 minutes per step.
