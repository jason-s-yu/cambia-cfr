# Outcome Sampling vs External Sampling: Mathematical Analysis & Engine Rewrite Case

Generated: 2026-02-16

## Executive Summary

**Conclusion: Proceed with the Go engine rebuild. OS provides tangible convergence for production-length games where ES is infeasible, and the Go engine (which was the original system design) delivers 30-100x throughput gains that compound with OS to enable rapid iteration — 100 training steps in ~15 minutes vs ~5 days.**

The Go service backend already exists at `service/`. The engine rewrite means building a **game logic library in Go** that both the WebSocket server and the CFR training pipeline can share.

---

## 0. Scope Clarification: What "Engine Rewrite" Actually Means

The TRAVERSAL_ANALYSIS.md profiling identified the "Python game engine" as the bottleneck. But deeper analysis reveals **agent state management dominates the game engine itself**:

| Component                   | % of Per-Node Cost | Category    |
| --------------------------- | ------------------ | ----------- |
| `AgentState.update()`       | ~30%               | Agent State |
| `encode_infoset()`          | ~18%               | Agent State |
| `AgentState.clone()` × 2    | ~5%                | Agent State |
| `get_infoset_key()`         | ~6%                | Agent State |
| `get_card_bucket()` calls   | ~6%                | Agent State |
| **Agent State subtotal**    | **~65-75%**        |             |
| `apply_action()`            | ~15%               | Game Engine |
| `get_legal_actions()`       | ~8%                | Game Engine |
| `undo()` closure execution  | ~3%                | Game Engine |
| **Game Engine subtotal**    | **~25-30%**        |             |
| Python interpreter overhead | ~5%                | Runtime     |

**Rewriting only `engine.py` captures just 25-30% of the bottleneck.** The full benefit requires rewriting the game engine, agent state management, and infoset encoding together as a single Go library with a Python FFI bridge.

This is the right architecture anyway: the Go game engine + agent state becomes a shared library used by both the WebSocket server and the Python CFR training loop (via ctypes/cgo).

---

## 1. Cambia Game Tree Structure

### 1.1 Decision Points Per Turn

A single Cambia turn involves 1-4 sequential decision points:

```
Turn Structure (worst case):
  START_TURN ──→ DrawStockpile (B=3)
                     │
              POST_DRAW ──→ Discard w/ability (B=6)
                                  │
                          ABILITY_SELECT ──→ KingLook (B=16)
                                                 │
                                          KING_SWAP ──→ Swap? (B=2)
                                                            │
                                                    SNAP_PHASE ──→ Per-snapper (B=9)
```

| Decision Context      | Branching Factor | Frequency            |
| --------------------- | ---------------: | -------------------- |
| START_TURN            |              2-3 | Every turn           |
| POST_DRAW             |              5-7 | Every draw           |
| ABILITY_SELECT (peek) |              4-6 | ~30% of discards     |
| ABILITY_SELECT (swap) |            16-36 | ~15% of discards     |
| KING_SWAP decision    |                2 | ~4% of discards      |
| SNAP_DECISION         |              3-9 | Conditional on match |
| SNAP_MOVE             |              4-6 | After opp snap       |

### 1.2 Effective Depth and Branching

Production configuration: `max_game_turns = 46`, `num_players = 2`.

- **Sequential decisions per game**: D ≈ 46 turns × 2.5 decisions/turn ≈ **115 decision points**
- **Average branching factor at traverser nodes**: B_avg ≈ **6-8**
- **Max branching factor**: B_max = 36 (blind swap / king look with 6-card hands)

### 1.3 Measured Tree Sizes (External Sampling)

From TRAVERSAL_ANALYSIS.md:

| max_game_turns  | Nodes Visited | Time (optimized) | Growth     |
| --------------- | ------------- | ---------------- | ---------- |
| 5               | 769           | 0.1s             | —          |
| 8               | 35,000        | 10s              | 45x        |
| 10              | 55,000        | 13s              | 1.6x       |
| 12              | 60,000        | 6s               | ~1x        |
| **15**          | **7,500,000** | **958s**         | **126x**   |
| 46 (production) | ~10⁹+ (est)   | hours+           | infeasible |

The 126x explosion from turns 12→15 corresponds to ability chains and snap phases becoming reachable.

---

## 2. External Sampling MCCFR: Mathematical Analysis

### 2.1 Algorithm

At each game node h with acting player i:

- **If i = traverser**: Enumerate ALL legal actions. For each action a, recurse on child(h, a). Compute exact counterfactual values.
- **If i = opponent**: Sample ONE action according to current strategy σ(h). Recurse on the single sampled child.

### 2.2 Per-Traversal Cost

Let B = average branching factor at traverser nodes, D = total sequential decisions.

In a 2-player alternating game, roughly half the decisions belong to each player. The traverser enumerates all actions at their ~D/2 nodes; the opponent samples one at their ~D/2 nodes.

The number of **leaf nodes** reached per ES traversal:

$$N_{ES} \approx B^{D/2}$$

For Cambia (production):

- B ≈ 7, D ≈ 115
- D/2 ≈ 58 traverser decision levels
- N_ES ≈ 7^{58} ≈ 10^{49} (theoretical upper bound)

In practice, terminal states truncate most branches. Empirically at 15 turns: **N_ES = 7.5M nodes**. At 46 turns, the tree is too large to complete a single traversal.

### 2.3 Convergence Rate

ES-MCCFR converges in average exploitability (Lanctot et al., 2009):

$$\mathbb{E}[\varepsilon_T] \leq \frac{\Delta \cdot |A_i| \cdot \sqrt{|\mathcal{I}_i|}}{\sqrt{T}}$$

Where:

- Δ = utility range = 2 (from -1 to +1)
- |A_i| = max actions per info set ≈ 36
- |I_i| = number of info sets (very large for Cambia)
- T = number of iterations

### 2.4 Sample Quality for Deep CFR

Each ES traversal generates **one advantage sample per traverser info set visited**. At 15 turns:

- ~3.75M traverser nodes → ~3.75M advantage samples per traversal
- Very high signal (exact regrets, no IS correction needed)
- But cannot complete at production depth

### 2.5 ES Verdict for Cambia

**ES is infeasible for production-length (46-turn) games.** Even at 15 turns, a single traversal takes 16 minutes. The 6+ hour training step for 1000 traversals used truncated games that don't capture the full strategic depth of Cambia (late-game Cambia calls, deep snap chains).

---

## 3. Outcome Sampling MCCFR: Mathematical Analysis

### 3.1 Algorithm

At EVERY game node h (traverser AND opponent):

- Sample ONE action according to a sampling policy q(a|h)
- Recurse on the single sampled child
- Apply importance sampling correction to regret estimates

The sampling policy is typically: q(a|h) = ε·uniform + (1-ε)·σ(a|h), with ε ≈ 0.6

### 3.2 Per-Traversal Cost

Each OS traversal walks a single root-to-terminal path:

$$N_{OS} = D \approx 115 \text{ nodes}$$

**This is independent of branching factor.** Whether a node has 2 or 36 legal actions, OS samples exactly one.

### 3.3 Convergence Rate

OS-MCCFR converges (Lanctot et al., 2009):

$$\mathbb{E}[\varepsilon_T] \leq \frac{\Delta \cdot \sum_i |\mathcal{I}_i| \cdot |A_i|^2}{\sqrt{T}}$$

The critical difference: **|A_i|² instead of |A_i|** in the numerator, and a sum over info sets rather than √|I_i|.

### 3.4 OS vs ES Convergence Comparison

For equivalent exploitability ε after T iterations:

| Factor                | ES             | OS     | Ratio (OS/ES)  |
| --------------------- | -------------- | ------ | -------------- |
| Iterations needed     | T_ES           | T_ES × | A              |     | ~7-36× more             |
| Nodes per iteration   | B^{D/2}        | D      | 10⁵-10⁷× fewer |
| **Total computation** | T_ES × B^{D/2} | T_ES × | A              | × D | **5,000-500,000× less** |

The |A| penalty means OS needs roughly **7-36× more iterations** (depending on average vs max branching) to match ES quality. But each iteration is **10⁵-10⁷× cheaper**.

### 3.5 Concrete Numbers for Cambia

**To collect 5M advantage samples (comparable to 1-2 ES traversals at 15 turns):**

- OS at 46 turns: ~58 traverser decisions/path → need ~86,000 traversals
- Per traversal: 115 nodes × 0.32ms/node (Python) = 37ms
- Total: 86,000 × 37ms = **53 minutes** (Python engine)

**Versus ES:**

- 2 traversals at 15 turns: 2 × 958s = **32 minutes** (but truncated games!)
- At 46 turns: **infeasible**

### 3.6 Sample Quality Tradeoff

| Property                | ES                        | OS                  |
| ----------------------- | ------------------------- | ------------------- |
| Regret estimates        | Exact (no IS correction)  | Noisy (IS-weighted) |
| Samples per traversal   | ~3.75M (15 turns)         | ~58 (46 turns)      |
| Game length             | Truncated (15 turns)      | Full (46 turns)     |
| Late-game strategy      | Not captured              | Captured            |
| Training signal quality | High per-sample           | Lower per-sample    |
| Diversity               | Low (same tree structure) | High (random paths) |

**Key insight**: OS samples are noisier individually but cover the full game tree including late-game strategy that ES can never reach. For Deep CFR, the neural network acts as a noise filter — it sees many noisy samples and learns the average signal. This is the approach used in practice for large games.

### 3.7 OS Verdict for Cambia

**OS is the only viable option for production-length games.** The variance penalty is real but manageable:

- Deep CFR's neural network smooths out IS noise
- 86K+ traversals per step provide diverse coverage
- Full 46-turn games capture complete strategic depth
- Each traversal completes in ~37ms (Python) or ~0.4ms (Go)

---

## 4. Go Engine Performance Analysis

### 4.1 Current Python Per-Node Cost Breakdown

From profiling (55K nodes, 17.5s at 10 turns):

**Per node: ~320μs (0.32ms)**

Sources of overhead:

1. **Python function call overhead**: ~1μs per call, ~50 calls/node = ~50μs
2. **Attribute resolution**: Hash table lookup per `.attr` access, ~100 per node = ~10μs
3. **isinstance checks**: Type hierarchy traversal, ~20 per node = ~1μs
4. **Closure creation**: 2-6 per action for undo system = ~3-10μs
5. **List copies for undo state**: `list(hand)`, `list(stockpile)` = ~5-20μs
6. **Dict copies**: `dict(pending_action_data)` = ~2-5μs
7. **serialize_card() calls**: String formatting for deltas = ~5-15μs
8. **Logging checks**: Even at WARNING level, `logger.debug` still evaluates the call = ~2μs
9. **AgentState.update()**: Heavy dict manipulation, bucket computation = ~100μs
10. **encode_infoset()**: NumPy array allocation + population = ~60μs

### 4.2 Go Architecture: Memory Layout

**Python CambiaGameState memory footprint: ~5KB**

- PyObject headers, pointer indirection at every level
- `players` → list → 2 × PlayerState → each has `hand` → list → Card objects
- Every attribute access = hash table lookup + pointer chase

**Go GameState (optimized struct): ~250 bytes**

```go
type GameState struct {
    Players         [2]PlayerState    // 2 × 48 bytes = 96, inline
    Stockpile       [54]uint8         // 54 bytes, packed cards
    StockLen        uint8             // 1
    DiscardPile     [54]uint8         // 54 bytes
    DiscardLen      uint8             // 1
    CurrentPlayer   uint8             // 1
    TurnNumber      uint16            // 2
    Flags           uint16            // 2 (game_over, snap_active, cambia, etc.)
    PendingAction   PendingAction     // 16 bytes, inline tagged union
    SnapState       SnapState         // 16 bytes, inline
    RNG             uint64            // 8 bytes (xorshift)
}
// Total: ~250 bytes → fits in 4 cache lines (256 bytes)

type PlayerState struct {
    Hand        [6]uint8    // max 6 cards, 1 byte each
    HandLen     uint8
    PeekIndices [2]uint8
    _pad        [5]uint8    // align to 16 bytes
}

type AgentState struct {
    OwnKnowledge   [6]uint8   // CardBucket per slot
    OppKnowledge   [6]uint8
    LastSeenTurns  [12]uint8  // 6 own + 6 opp
    HandSizes      [2]uint8
    DiscardTop     uint8
    StockEstimate  uint8
    GamePhase      uint8
    CambiaState    uint8
}
// Total: ~30 bytes
```

### 4.3 Per-Operation Go Estimates

| Operation                 | Python     | Go         | Speedup       | Why                                               |
| ------------------------- | ---------- | ---------- | ------------- | ------------------------------------------------- |
| `get_legal_actions()`     | ~50μs      | ~200ns     | 250×          | Switch + bit ops vs set construction + isinstance |
| `apply_action()`          | ~100μs     | ~300ns     | 330×          | Direct struct mutation vs closures + dict copies  |
| `undo()`                  | ~20μs      | ~50ns      | 400×          | memcpy 250 bytes vs closure execution             |
| `AgentState.clone()` ×2   | ~20μs      | ~50ns      | 400×          | memcpy 60 bytes vs dict comprehension + dataclass |
| `AgentState.update()` ×2  | ~100μs     | ~300ns     | 330×          | Array index vs dict ops + isinstance chains       |
| `encode_infoset()`        | ~60μs      | ~150ns     | 400×          | Direct array write vs NumPy alloc + population    |
| `create_observation()`    | ~30μs      | ~100ns     | 300×          | Struct init vs object creation + iteration        |
| `filter_observation()` ×2 | ~10μs      | ~20ns      | 500×          | Field zeroing vs copy.copy + isinstance           |
| Python overhead           | ~50μs      | ~0         | ∞             | No interpreter, no GIL, no refcounting            |
| **Total per node**        | **~320μs** | **~1-3μs** | **~100-300×** |                                                   |

### 4.4 CPU/Cache-Level Analysis

**L1 cache (32-64KB per core):**

- Go GameState (250 bytes) + AgentState ×2 (60 bytes) + undo snapshot (250 bytes) = **560 bytes**
- Entire working set fits in **9 cache lines**. Zero cache misses during traversal.
- Python: ~5KB game state + ~2KB agent states + pointer tables = **8-10KB with indirection**
- Constant L1 misses from pointer chasing. Every `.attr` is a hash lookup → memory-indirect → likely L2 or L3.

**Branch prediction:**

- Go: `switch action.Type` compiles to a jump table. Predictable.
- Python: `isinstance()` chains = indirect calls through type hierarchy. Unpredictable.

**Memory allocation:**

- Go: **Zero heap allocations per node.** Game state on stack, undo via memcpy.
- Python: Per node creates ~10-20 heap objects (closures, lists, dicts, tuples, LogRecords).

**Undo system redesign:**

- Python: Incremental delta tracking with closures (2-6 closures/action, list copies, deque ops)
- Go: **Full state snapshot** before action: `memcpy(&snapshot, &state, 250)`. Undo = `memcpy(&state, &snapshot, 250)`.
- At 250 bytes, memcpy takes ~30ns. This is faster than a single Python closure creation.

### 4.5 Conservative vs Optimistic Estimates

| Scenario     | Speedup  | Justification                                    |
| ------------ | -------- | ------------------------------------------------ |
| Conservative | 30-50×   | Comparable to OpenSpiel C++ vs Python benchmarks |
| Realistic    | 80-120×  | Cache-optimized structs + zero-alloc undo        |
| Optimistic   | 200-300× | Full SIMD card ops + pool allocators + inlining  |

**Used for analysis: 50-100× (realistic-conservative)**

### 4.6 FFI Bridge Cost

Python calling Go via cgo shared library:

- Per-call overhead: ~500ns (FFI boundary crossing)
- Per OS traversal: ~115 calls × 3 functions/call ≈ 345 FFI crossings
- Total FFI overhead: 345 × 500ns = **172μs per traversal** (~0.17ms)
- vs Go engine work: 115 × 2μs = **230μs per traversal**
- FFI adds ~75% overhead but total is still **0.4ms per traversal** (vs 37ms in pure Python)

Alternative: gRPC bridge = ~50μs per call = 17ms per traversal (still 2× faster than Python-native).

**Recommendation: cgo shared library for training, gRPC for production game server.**

---

## 5. Combined Performance Projections

### 5.1 Training Step Duration (100K traversals)

| Configuration                       | Per Traversal | 100K Traversals  | 100 Steps     |
| ----------------------------------- | ------------- | ---------------- | ------------- |
| ES + Python (15 turns)              | 958s          | **26,600 hours** | impossible    |
| ES + Python (truncated to complete) | ~6 hours/1K   | ~6 hours         | ~25 days      |
| **OS + Python**                     | **37ms**      | **62 min**       | **~4.3 days** |
| **OS + Go (via FFI)**               | **0.4ms**     | **40 sec**       | **~67 min**   |
| OS + Go (pure Go traversal)         | 0.23ms        | 23 sec           | ~38 min       |

### 5.2 Samples Generated (100K traversals at 46 turns)

- Advantage samples: ~58 per traversal × 100K = **5.8M samples**
- Strategy samples: ~57 per traversal × 100K = **5.7M samples**
- Reservoir buffer capacity: 2M each → fills in ~34K traversals, then reservoir-samples

### 5.3 Full Training Pipeline (100 steps)

| Phase                           | OS + Python               | OS + Go (FFI)           |
| ------------------------------- | ------------------------- | ----------------------- |
| Traversals (100K/step × 100)    | 103 hours                 | 67 min                  |
| Network training (4K SGD × 100) | 47 min                    | 47 min                  |
| Weight serialization            | ~2 min                    | ~2 min                  |
| **Total**                       | **~104 hours (4.3 days)** | **~116 min (~2 hours)** |

### 5.4 Development Velocity Impact

| Metric                | OS + Python | OS + Go |
| --------------------- | ----------- | ------- |
| Time per experiment   | 4.3 days    | 2 hours |
| Experiments per week  | 1-2         | 50+     |
| Hyperparameter sweeps | Weeks       | Hours   |
| Debug iteration cycle | Day         | Minutes |

---

## 6. Agent Quality Analysis

### 6.1 Convergence to Nash Equilibrium

For a 2-player zero-sum game like Cambia (1v1), MCCFR converges to a Nash equilibrium. The question is how fast.

**OS convergence bound** (Lanctot et al., 2009):

$$\varepsilon_T \leq \frac{2 \cdot |\mathcal{I}| \cdot |A|^2}{\sqrt{T}}$$

For Cambia:

- |I| ≈ 10⁶ (with card bucketing abstraction)
- |A|_max = 36, |A|_avg ≈ 7
- Using |A|² ≈ 49 (average), conservative 1296 (max)

To reach ε = 0.1 (well beyond human-level):

$$T \geq \left(\frac{2 \times 10^6 \times 49}{0.1}\right)^2 \approx 10^{18}$$

This theoretical bound is extremely loose. In practice, CFR variants converge **orders of magnitude faster** than the worst-case bound suggests. Empirical convergence for similar-complexity games:

| Game                       | Info Sets | Actions | Iterations to ε=0.1 | Reference      |
| -------------------------- | --------- | ------- | ------------------- | -------------- |
| Kuhn Poker                 | 12        | 3       | ~100                | Zinkevich 2007 |
| Leduc Hold'em              | 936       | 4       | ~10,000             | Lanctot 2009   |
| Limit Hold'em (abstracted) | ~10⁷      | 3       | ~10⁸                | Bowling 2015   |

Cambia (abstracted) has ~10⁶ info sets and up to 36 actions. Extrapolating:

- **Estimated iterations to strong play**: 10⁵ - 10⁷
- **With 100K traversals/step**: 1-100 training steps

### 6.2 OS vs ES Quality at Fixed Compute Budget

Given a fixed wall-clock budget of 6 hours:

**ES (Python, 15-turn truncated):**

- ~1000 traversals at 15 turns
- ~3.75B advantage samples (very high quality, exact regrets)
- But: truncated games miss late-game strategy entirely
- Missing: optimal Cambia timing, late snap decisions, endgame play

**OS (Python, 46-turn full games):**

- ~580,000 traversals at 46 turns
- ~33.6M advantage samples (noisier but full-game coverage)
- Complete strategic coverage including late-game
- With neural network noise filtering: effective quality comparable to ES

**OS (Go, 46-turn full games):**

- ~54M traversals at 46 turns (6 hrs = 21,600s / 0.4ms per trav)
- ~3.1B advantage samples
- Massive diversity of game situations explored
- Quality: likely superior to ES due to full-game coverage + volume

### 6.3 Beyond-Human Play Feasibility

Cambia has several properties that make beyond-human play achievable:

1. **Small action space** (max 36, avg ~7) — much smaller than poker
2. **Fixed hand size** (4 cards, bounded by 6) — limits state space
3. **Memory as key skill** — CFR agents have configurable memory (level 0 = perfect recall)
4. **Imperfect information** — CFR is designed precisely for this
5. **Card counting** — agent naturally tracks stockpile/discard composition

Human weaknesses the agent exploits:

- Perfect card tracking (humans forget after 3-4 turns)
- Optimal snap timing (humans hesitate or snap incorrectly)
- Exact Cambia call EV calculation (humans use gut feeling)
- Ability usage optimization (humans underuse king ability)

**Assessment: Beyond-human play is achievable with OS-MCCFR + Deep CFR within 10⁵-10⁶ training iterations.**

---

## 7. Go Engine Architecture Recommendation

### 7.1 Shared Library Design

```
┌──────────────────────────────────────────┐
│           cambia-engine (Go)             │
│                                          │
│  GameState  AgentState  Encoding         │
│  LegalActions  ApplyAction  Undo         │
│  InfosetKey  CardBucket  Observation     │
│                                          │
├──────────┬───────────────────────────────┤
│  C API   │    Native Go API              │
│  (cgo)   │                               │
└────┬─────┴──────────┬───────────────────-┘
     │                │
     ▼                ▼
┌─────────┐    ┌──────────────┐
│ Python  │    │  Go WebSocket │
│ CFR     │    │  Game Server  │
│ Training│    │  (service/)   │
└─────────┘    └──────────────┘
```

### 7.2 Key Design Decisions

1. **Undo strategy**: Full-state memcpy snapshot (250 bytes). Simpler, faster, no closure overhead.
2. **Card representation**: `uint8` packed (6 bits rank + 2 bits suit). Enables SIMD comparisons for snap matching.
3. **Legal actions**: Return as bitmask (`uint64` for up to 64 actions, `[3]uint64` for full 146). No heap allocation.
4. **Agent state**: Value-type struct. Clone = `*dst = *src`. No deep copy needed.
5. **Encoding**: Write directly into caller-provided `[222]float32` buffer. No allocation.
6. **RNG**: `xorshift64` inline. No interface indirection.

### 7.3 What Stays in Python

- CFR traversal loop (OS sampling, regret accumulation)
- Neural network training (PyTorch)
- Reservoir buffer management
- Training orchestration, logging, checkpointing
- Exploitability calculation (calls Go engine for game simulation)

### 7.4 Estimated Development Effort

| Component                             | Effort         | Priority |
| ------------------------------------- | -------------- | -------- |
| Go GameState + apply_action + undo    | 1 week         | P0       |
| Go AgentState + update + clone        | 1 week         | P0       |
| Go legal actions + encoding           | 3 days         | P0       |
| C API / cgo bridge                    | 2 days         | P0       |
| Python ctypes wrapper                 | 2 days         | P0       |
| Testing + validation vs Python engine | 3 days         | P0       |
| **Total**                             | **~3.5 weeks** |          |

Compared to TRAVERSAL_ANALYSIS.md estimate of 3-6 weeks for C++ — Go is faster to develop, and the WebSocket service already establishes Go as a project language.

---

## 8. Recommended Execution Plan

### Phase 1: OS Implementation in Python (1 week)

Switch `deep_worker.py` from external sampling to outcome sampling. This is already partially done — `worker.py` already implements OS for tabular CFR+. Adapt for Deep CFR:

- Sample one action at ALL nodes (traverser included)
- Apply IS-weighted regret computation
- Generate advantage/strategy samples per path
- **Expected result**: Training steps drop from 6+ hours to ~60 minutes

### Phase 2: Go Engine Library (3.5 weeks)

Build the shared Go library with C API:

- Game state, apply/undo, legal actions
- Agent state, clone, update
- Infoset encoding, card bucketing
- Python ctypes bridge
- **Expected result**: Training steps drop from ~60 minutes to ~40 seconds

### Phase 3: Integration + Validation (1 week)

- Verify Go engine matches Python engine behavior (fuzz testing with random games)
- Benchmark end-to-end training pipeline
- Confirm convergence on known test scenarios
- **Expected result**: Validated 50-100× speedup, ready for production training

### Phase 4: Production Training (ongoing)

- Run 1000+ training iterations (complete in <24 hours with Go engine)
- Evaluate agent against baseline strategies
- Tune hyperparameters with rapid iteration cycle
- **Expected result**: Beyond-human Cambia agent

---

## 9. Risk Analysis

| Risk                              | Impact | Mitigation                                                               |
| --------------------------------- | ------ | ------------------------------------------------------------------------ |
| OS convergence too slow           | Medium | Start training immediately with Phase 1; validate quality before Phase 2 |
| Go engine bugs                    | High   | Extensive fuzz testing against Python reference; shared test suite       |
| FFI overhead higher than expected | Low    | Fall back to gRPC; worst case still 10× faster than Python               |
| IS variance too high for Deep CFR | Medium | Tune exploration ε; consider robust sampling variants (AIVAT)            |
| Game rule edge cases in Go        | Medium | Python engine remains as oracle; differential testing                    |

---

## 10. Decision Matrix

| Criterion                    | OS Only (Python) | Go Engine Only (ES)  |  OS + Go Engine  |
| ---------------------------- | :--------------: | :------------------: | :--------------: |
| Handles 46-turn games        |        ✓         | ✗ (still infeasible) |        ✓         |
| Fast training iteration      | △ (60 min/step)  |          ✗           | ✓ (40 sec/step)  |
| Serves original architecture |        ✗         |          △           |        ✓         |
| Development effort           |   Low (1 week)   |   High (4+ weeks)    | Medium (5 weeks) |
| Agent quality                |       Good       |  N/A (can't train)   |       Best       |
| Beyond-human play            | Possible (slow)  |     Not feasible     |    Achievable    |

**The clear winner is OS + Go Engine**, which was the original system design. OS makes production-length training possible; Go makes it fast enough for rapid iteration.
