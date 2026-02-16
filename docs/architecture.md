# Deep CFR Architecture

Overview of the Deep CFR training pipeline and how it differs from the original tabular CFR+ system.

## Tabular CFR+ (Original)

The original system stores per-infoset regret and strategy data in Python dictionaries (`defaultdict[InfosetKey, np.ndarray]`). Each training iteration:

1. A snapshot of the regret dict is pickled and sent to 23 worker processes.
2. Workers run outcome sampling traversals, accumulating local dict updates.
3. The main process merges updates into the global dicts (serial loop over all keys).
4. RM+ clamping is applied to all updated keys.

The primary bottleneck is the unbounded growth of the regret/strategy dicts as more information sets are visited. At scale, the merge step and snapshot pickling dominate wall-clock time.

## Deep CFR (New)

Deep CFR replaces the tabular storage with two neural networks and reservoir sampling buffers. The regret dict is replaced by the advantage network; the strategy dict is replaced by the strategy network. Training samples are collected via external sampling traversals and stored in fixed-capacity buffers.

### Pipeline

```
                        ┌───────────────────────────────────────────┐
                        │           DeepCFRTrainer                  │
                        │                                           │
                        │  AdvantageNetwork (Vθ) ─── 125K params    │
                        │  StrategyNetwork (Πφ) ─── 125K params     │
                        │  AdvantageBuf (Mv) ──── 2M samples        │
                        │  StrategyBuf (Mπ) ──── 2M samples         │
                        └────────┬──────────────────────┬───────────┘
                                 │                      │
                    ┌────────────┘                      └────────────┐
                    │ serialize weights                 │ collect    │
                    │ (numpy arrays)                    │ samples    │
                    ▼                                   │            │
        ┌───────────────────────┐                       │            │
        │    Worker Pool        │                       │            │
        │                       │                       │            │
        │  ┌─────────────────┐  │                       │            │
        │  │ deep_worker.py  │  │                       │            │
        │  │                 │  │                       │            │
        │  │  encode_infoset │──┼─ encoding.py          │            │
        │  │       │         │  │                       │            │
        │  │       ▼         │  │                       │            │
        │  │  AdvantageNet   │  │                       │            │
        │  │  (inference)    │  │                       │            │
        │  │       │         │  │                       │            │
        │  │       ▼         │  │                       │            │
        │  │  ReLU+normalize │──┼─ networks.py          │            │
        │  │       │         │  │                       │            │
        │  │       ▼         │  │                       │            │
        │  │  External       │  │                       │            │
        │  │  Sampling       │  │                       │            │
        │  │  Traversal      │──┼─ game/engine.py       │            │
        │  │       │         │  │                       │            │
        │  │       ▼         │  │                       │            │
        │  │  ReservoirSample│──┼───────────────────────┘            │
        │  │  (adv + strat)  │  │                                    │
        │  └─────────────────┘  │                                    │
        └───────────────────────┘                                    │
                                                                     │
        ┌────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────┐
  │  Network Training (main process)    │
  │                                     │
  │  Sample batch from Mv               │
  │       │                             │
  │       ▼                             │
  │  Weighted MSE loss:                 │
  │  ((t+1)^α / mean) * MSE(Vθ, target) │
  │       │                             │
  │       ▼                             │
  │  Adam optimizer + grad clip         │
  │                                     │
  │  (repeat for Πφ with Mπ)            │
  └─────────────────────────────────────┘
```

### Data Flow Step by Step

1. The trainer serializes the current advantage network weights to numpy arrays and distributes them to workers.

2. Each worker initializes a game (`CambiaGameState`) and agent belief states (`AgentState`).

3. The worker runs a single external sampling traversal:
   - At the traverser's decision node: encodes the infoset via `encode_infoset()`, queries the advantage network for the current strategy (`ReLU + normalize`), then enumerates all legal actions. For each action, the worker applies it to the game state, recurses, and undoes the action. This produces exact per-action counterfactual values.
   - At opponent decision nodes: encodes the infoset, computes the strategy from the network, samples one action, recurses.
   - At terminal nodes: returns the utility vector.

4. After traversal, the worker has collected:
   - Advantage samples (one per traverser decision node): `(features, regret_target, action_mask, iteration)`.
   - Strategy samples (one per opponent decision node): `(features, strategy_target, action_mask, iteration)`.

5. The trainer adds all samples to the respective reservoir buffers via Algorithm R.

6. The trainer trains each network by sampling mini-batches from the buffers. The loss is weighted MSE where the weight for each sample is `(t+1)^alpha`, normalized by the batch mean. Loss is computed only over legal actions (masked). Gradients are clipped to max norm 1.0.

7. The loop repeats from step 1 with updated network weights.

### External Sampling vs. Outcome Sampling

The traversal mode changed from outcome sampling to external sampling:

| Aspect                | Outcome Sampling (old)      | External Sampling (new)                |
| --------------------- | --------------------------- | -------------------------------------- |
| Traverser nodes       | Sample 1 action, IS-correct | Enumerate all actions                  |
| Opponent nodes        | Sample 1 action             | Sample 1 action                        |
| Regret quality        | Noisy (importance sampling) | Exact (no IS correction)               |
| Cost per traversal    | O(1) per node               | O(branching factor) at traverser nodes |
| Samples per traversal | 1 per visited node          | 1 per traverser node (higher quality)  |

External sampling is more expensive per traversal but produces exact regret estimates, requiring fewer total iterations for convergence. For Cambia's typical branching factor of 5-15 at decision nodes, this is a favorable tradeoff.

### What Changed vs. What Stayed

Changed:

- Storage: tabular dicts -> neural networks + reservoir buffers
- Traversal: outcome sampling -> external sampling
- Worker output: dict updates -> `ReservoirSample` lists
- Merge step: eliminated (samples append to buffers, no per-key merging)
- Strategy lookup: `regret_sum[key] -> RM+` -> `network(features) -> ReLU+normalize`
- Pickling: entire regret dict -> fixed-size network weights (~500KB)

Unchanged:

- Game engine (`CambiaGameState`) and its apply/undo mechanics
- Agent belief state (`AgentState`) and observation model
- Action types (`GameAction` NamedTuples)
- Card abstraction (`CardBucket`, `DecayCategory`)
- Memory decay model (event and time-based)
- Worker-level logging and progress reporting infrastructure
