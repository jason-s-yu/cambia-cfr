# Counterfactual Regret Minimization+ Trainer for Cambia Self-Play

## Known limitations

- Tree depth / max turn count=

## Parallel Training (External Sampling Monte Carlo CFR)

We implement parallel training for the Cambia CFR+ agent using a technique similar to **External Sampling Monte Carlo Counterfactual Regret Minimization (ESMCFR)**.

### Goal

The primary objective of this parallelization is to **reduce the wall-clock time required to complete each training iteration**. Standard CFR involves deep traversals of the game tree, which can be computationally intensive and slow on a single CPU core, especially for games with large state spaces or long potential game lengths like Cambia.

### How

1. **Sampling Instead of Full Traversal:** Instead of attempting a full recursive traversal of the game tree in each iteration (as in vanilla CFR), ESMCFR relies on sampling. In each iteration `t`, multiple independent simulations (or "traversals") of the game are run from the initial state to a terminal state.
2. **Fixed Strategy Profile:** Actions within each simulation are chosen based on a *fixed* strategy profile calculated at the *beginning* of iteration `t` (derived from the current accumulated regrets).
3. **Parallel Simulations:** The core idea is that these independent game simulations can be executed concurrently. The implementation uses Python's `multiprocessing` module:
    - The main training process determines the current strategy profile $\sigma^t$.
    - It launches `N` worker processes (where `N` is `num_workers` from the config).
    - Each worker process executes the `run_cfr_simulation_worker` function (`src/cfr/worker.py`).
    - This function initializes a fresh `CambiaGameState` and `AgentState` objects.
    - It then calls `_traverse_game_for_worker`, which simulates one complete game playthrough, sampling actions according to $\sigma^t$.
4. **Local Update Accumulation:** As a worker traverses its simulation, it calculates the instantaneous regrets and strategy contributions for the nodes it visits. Crucially, these updates are **accumulated locally** within dictionaries specific to that worker process. They do *not* modify the shared, global state directly. Updates are weighted by the appropriate reach probabilities encountered during the simulation.
5. **Merging Results:** After the parallel simulations complete, each worker returns its dictionary of local updates to the main process.
    - The main process then executes the `_merge_local_updates` function (`src/cfr/data_manager_mixin.py`).
    - This function iterates through the results from all workers and sequentially adds the local updates to the main trainer's `regret_sum`, `strategy_sum`, and `reach_prob_sum` dictionaries.
    - Finally, Regret Matching+ (`maximum(0, ...)`) is applied to the updated `regret_sum`.

### Improvements and Caveats

- **Time Per Iteration:** By running `N` simulations concurrently across `N` CPU cores, the time taken to gather the necessary samples and updates for a single iteration is significantly reduced compared to running them sequentially. Ideally, with low overhead, the simulation phase of an iteration could be up to `N` times faster.
- **Overall Training Time:** This reduction in time per iteration leads to a faster overall training process in terms of real-world time (wall-clock time), allowing the agent to reach a target level of convergence more quickly.

**What it Does Not Speed Up (Directly):**

- **Total Computation:** It doesn't necessarily decrease the *total* number of computations (CPU-hours) required. It distributes the work across more resources.
- **Merge Step:** The final merging step in the main process remains serial. While typically much faster than the simulations, it could become a bottleneck if `N` is extremely large or the number of unique infosets updated per iteration is vast.
- **Convergence Rate (per iteration):** The convergence properties are now governed by the sampling nature of ESMCFR, which might differ slightly from vanilla CFR, but the goal is faster convergence in *time*.
-
