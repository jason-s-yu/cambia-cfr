# Monte Carlo CFR Sampling Methods

This document outlines the different sampling strategies considered for accelerating Counterfactual Regret Minimization (CFR) training for the 2-player variant of Cambia. We focus on improving performance with parallel execution.

## The Need for Sampling

Vanilla Counterfactual Regret Minimization (CFR) is guaranteed to converge to a Nash Equilibrium in two-player zero-sum games. However, it requires traversing the *entire* game tree on each iteration. For games with large state spaces or long potential game lengths, like Cambia, this full traversal becomes computationally infeasible or prohibitively slow, even for a single iteration.

Monte Carlo CFR (MCCFR) methods address this by sampling only a portion of the game tree on each iteration. The goal is to obtain unbiased estimates of the regrets while significantly reducing the computational cost per iteration. This allows for faster training in terms of wall-clock time, even if more iterations might be needed compared to vanilla CFR due to the variance introduced by sampling.

Our primary goal for using MCCFR in this project is to leverage multi-core processors via parallelization. Multiple workers can independently run sampled game simulations, and their results can be aggregated to update the agent's strategy.

## Sampling Methods Considered

Two primary MCCFR sampling methods were considered:

### Outcome Sampling (OS)

* **Traversal Method:** Samples ONE action at EVERY decision node (acting player, opponent, chance) according to the current strategy profile ($\sigma^t$) or chance probabilities. This results in traversing a single path from the root to a terminal state in each simulation.
* **Regret Update:** Since only one path is explored, regret updates require careful handling to remain unbiased. Importance sampling is typically used. The update at an information set `I` visited along the sampled path `z` depends on the utility `u_i(z)` obtained at the end of the path, the probability of sampling that path `q(z)`, the opponent's reach probability `π_{-i}(I)`, and the current strategy `σ(a|I)`.
  * **Our Planned Implementation:** Uses a specific OS-derived formula where the regret for action `a` at infoset `I` (if chosen action was `a*` with probability `p(a*)`) is estimated based on the returned utility `v(I->a*)` from the recursive call:
    * `utility_estimate = v(I->a*) / p(a*)`
    * `regret(a) ≈ (1 if a=a* else 0) * utility_estimate - σ(a|I) * utility_estimate`
    * The final update is weighted by `π_{-i}(I)` and the iteration weight (for CFR+).
* **Pros:**
  * Lowest computational cost per sampled path (minimal traversal).
  * Relatively simple traversal logic to implement.
* **Cons:**
  * Generally introduces higher variance into the regret estimates compared to ES, as both players' actions are sampled.
  * May require more iterations to converge due to higher variance.

### External Sampling (ES)

* **Traversal Method:** Samples actions only for the **opponent** and **chance** nodes. At nodes where the **acting player** (`i`) makes a decision, *all* legal actions are explored. Recursion proceeds down each of these branches, but opponent/chance moves within those branches are sampled.
* **Regret Update:** Since all of the acting player's actions are explored locally, the regret update is closer to Vanilla CFR (`regret(a) = v(I->a) - v(I)`). However, the counterfactual values `v(I->a)` and `v(I)` are *estimates* obtained via the sampled opponent/chance traversals. Updates are weighted by `π_{-i}(I)`.
* **Pros:**
  * Lower variance in regret estimates than OS because the acting player's contribution is not sampled.
  * Potentially faster convergence in terms of the number of iterations required.
  * Proven to require only a constant factor more iterations than Vanilla CFR while reducing per-iteration cost.
* **Cons:**
  * Higher computational cost per iteration compared to OS, as it explores all branches at the acting player's nodes.
  * Traversal logic is slightly more complex than OS.

| Feature                 | Outcome Sampling (OS)                                  | External Sampling (ES)                                        | Vanilla CFR (Full Tree)          |
| :---------------------- | :----------------------------------------------------- | :------------------------------------------------------------ | :------------------------------- |
| **Player Action** | Sample 1                                               | Explore All                                                   | Explore All                      |
| **Opponent Action** | Sample 1                                               | Sample 1                                                      | Explore All                      |
| **Chance Action** | Sample 1                                               | Sample 1                                                      | Explore All                      |
| **Cost / Iteration** | Low                                                    | Medium                                                        | High                             |
| **Variance** | High                                                   | Low                                                           | Zero (Deterministic)             |
| **Convergence (Iters)** | Potentially Slower                                     | Potentially Faster                                            | Baseline                         |
| **Convergence (Time)** | Potentially Fast (if low cost/iter >> more iters)     | Potentially Fast (good balance of cost/iter and variance)    | Slow                             |

## Current Implementation

We utilize outcome sampling for this agent due to:

* Performance improvements: OS drastically reduces the computation per simulation compared to the previous Parallel Vanilla CFR+ implementation (which explored all actions for *both* players recursively). This directly addresses the goal of reducing wall-clock time per iteration.
* The single-path traversal logic is somewhat simpler to implement and debug compared to the mixed exploration/sampling logic of ES.