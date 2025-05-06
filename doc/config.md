# Configuration

This document describes the configuration options for the Cambia CFR+ training suite for a YAML-based configuration file (e.g., `config.yaml`, `parallel.config.yaml`).

## Root Level Keys

The configuration file is structured with the following top-level keys:

* `api`: Settings related to the API client for online play (currently future work).
* `system`: General system-level settings.
* `cfr_training`: Parameters controlling the CFR training process.
* `cfr_plus_params`: Parameters specific to CFR+ algorithm variants.
* `agent_params`: Settings defining agent behavior, particularly memory and abstraction.
* `cambia_rules`: Defines the specific game rules of Cambia to be used during training and simulation.
* `persistence`: Configuration for saving and loading trained agent data.
* `logging`: Settings for configuring logging behavior for the main process and workers.

---

## `api`

Settings for the API client used for interacting with a live Cambia game server.

* **`base_url`**:
  * Description: The base URL of the Cambia game server.
  * Type: `string`
  * Example: `"http://localhost:8080"`
* **`auth`**:
  * Description: Authentication credentials for the game server.
  * Type: `object` (dictionary)
  * Fields:
    * **`email`**: (Optional) Email for login.
      * Type: `string`
      * Example: `"player@example.com"`
    * **`password`**: (Optional) Password for login.
      * Type: `string`
      * Example: `"securepassword123"`
    * **`token_file`**: (Optional) Path to a file containing a JWT or API token.
      * Type: `string`
      * Example: `"./.game_token"`
  * Example:

        ```yaml
        auth:
          email: "player@example.com"
          password: "securepassword123"
        ```

---

## `system`

General system-level configurations.

* **`recursion_limit`**:
  * Description: Sets Python's recursion limit. Useful for deep game tree traversals in CFR, though the primary traversal is now iterative or managed by workers.
  * Type: `integer`
  * Default: `10000`
  * Example: `15000`

---

## `cfr_training`

Parameters controlling the main Counterfactual Regret Minimization training process.

* **`num_iterations`**:
  * Description: The total number of training iterations to run.
  * Type: `integer`
  * Default: `100000`
  * Example: `1000000`
* **`save_interval`**:
  * Description: How often (in iterations) to save the agent's progress (regret sums, strategy sums).
  * Type: `integer`
  * Default: `5000`
  * Example: `1000`
* **`pruning_enabled`**:
  * Description: Enables or disables Regret-Based Pruning (RBP). If true, actions with regret below `pruning_threshold` might be temporarily ignored.
  * Type: `boolean` (`true` or `false`)
  * Default: `true`
  * Example: `false`
* **`pruning_threshold`**:
  * Description: The regret value below which an action might be pruned if `pruning_enabled` is true.
  * Type: `float`
  * Default: `1.0e-6`
  * Example: `0.00001`
* **`exploitability_interval`**:
  * Description: How often (in iterations) to calculate and log the exploitability of the current average strategy. Set to `0` to disable.
  * Type: `integer`
  * Default: `1000`
  * Example: `500`
* **`num_workers`**:
  * Description: The number of parallel worker processes to use for game simulations.
    * `1`: Runs simulations sequentially in the main process.
    * `0` or `"auto"`: Automatically sets to `os.cpu_count() - 1` (or `1` if only 1 CPU is available).
    * `>1`: Uses that many parallel worker processes.
  * Type: `integer` or `string`
  * Default: `1`
  * Examples: `1` (sequential), `0` (auto), `"auto"` (auto), `8` (8 workers)

---

## `cfr_plus_params`

Parameters specific to variations and enhancements of the CFR algorithm, such as CFR+.

* **`weighted_averaging_enabled`**:
  * Description: If true, uses weighted averaging for the strategy sum, where iteration `t` is weighted by `max(0, t - averaging_delay)`. This often improves convergence in CFR+.
  * Type: `boolean` (`true` or `false`)
  * Default: `true`
  * Example: `false`
* **`averaging_delay`**:
  * Description: The delay `d` used in weighted averaging. Averaging effectively starts from iteration `d+1`.
  * Type: `integer`
  * Default: `100`
  * Example: `0` (weights by `t`)

---

## `agent_params`

Parameters defining the agent's internal model and memory capabilities.

* **`memory_level`**:
  * Description: Controls the agent's memory and abstraction level for opponent modeling.
    * `0`: Perfect Recall (agent remembers all cards it has seen).
    * `1`: Event Decay (agent's knowledge of a card decays to a category upon certain game events, e.g., opponent swaps/replaces).
    * `2`: Event + Time Decay (combines event decay with time-based decay, where knowledge further degrades if not reinforced after `time_decay_turns`).
  * Type: `integer` (0, 1, or 2)
  * Default: `1`
  * Example: `2`
* **`time_decay_turns`**:
  * Description: The number of game turns after which an agent's specific knowledge of an opponent's card (if `memory_level` is 2) will decay if not reinforced by new observations.
  * Type: `integer`
  * Default: `3`
  * Example: `5`

---

## `cambia_rules`

Defines the specific rules of the Cambia game variant to be used for training and simulation. These should match the target environment if playing against external agents or a server.

* **`allowDrawFromDiscardPile`**:
  * Description: If `true`, players can choose to draw the top visible card from the discard pile instead of the stockpile.
  * Type: `boolean`
  * Default: `false`
  * Example: `true`
* **`allowReplaceAbilities`**:
  * Description: If `true`, special card abilities (like King's peek/swap) trigger when a player replaces one of their face-down cards with the drawn card that has an ability. If `false`, abilities only trigger when the drawn card itself is discarded *after drawing*.
  * Type: `boolean`
  * Default: `false`
* **`snapRace`**:
  * Description: If `true` and multiple players can snap simultaneously, a "race" condition determines who acts (not fully defined/implemented in current base spec, typically implies a specific resolution order). If `false`, snaps are resolved in player turn order starting from the player after the one who caused the discard.
  * Type: `boolean`
  * Default: `false`
* **`penaltyDrawCount`**:
  * Description: The number of cards a player must draw from the stockpile as a penalty for an incorrect snap attempt.
  * Type: `integer`
  * Default: `2`
  * Example: `1`
* **`use_jokers`**:
  * Description: The number of Jokers to include in the deck.
  * Type: `integer` (0, 1, or 2)
  * Default: `2`
  * Example: `0` (no jokers)
* **`cards_per_player`**:
  * Description: The initial number of face-down cards dealt to each player.
  * Type: `integer`
  * Default: `4`
  * Example: `3`
* **`initial_view_count`**:
  * Description: The number of their own closest face-down cards a player is allowed to peek at the start of the game.
  * Type: `integer`
  * Default: `2`
  * Example: `1`
* **`cambia_allowed_round`**:
  * Description: The first game round (0-indexed) in which players are allowed to call "Cambia". A round consists of each player having one turn. `0` means Cambia can be called from the very first turn of the game.
  * Type: `integer`
  * Default: `0`
  * Example: `1` (Cambia allowed starting from the second round)
* **`allowOpponentSnapping`**:
  * Description: If `true`, players can attempt to "Snap Opponent", forcing the opponent to discard a matching card and then moving one of their own cards to the opponent's empty slot.
  * Type: `boolean`
  * Default: `false`
  * Example: `true`
* **`max_game_turns`**:
  * Description: The maximum number of turns a game simulation can last before it's automatically ended (e.g., as a draw or based on current scores). Useful for preventing infinitely long games in simulation. `0` means no limit.
  * Type: `integer`
  * Default: `300`
  * Example: `200`

---

## `persistence`

Settings related to saving and loading the trained agent's data.

* **`agent_data_save_path`**:
  * Description: The file path (relative to the execution directory) where the agent's learned data (regret sums, strategy sums, iteration count, etc.) will be saved and loaded from.
  * Type: `string`
  * Default: `"cambia_cfr_agent_level1.joblib"`
  * Example: `"strategy/my_cambia_agent_v1.joblib"`

---

## `logging`

Configuration for logging messages generated during training.

* **`log_level_file`**:
  * Description: The logging level for messages written to the main process's log file. Also acts as a global fallback for worker log levels if `worker_config` is not specified or doesn't cover a worker.
  * Type: `string` (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  * Default: `"DEBUG"`
  * Example: `"INFO"`
* **`log_level_console`**:
  * Description: The logging level for messages displayed on the console (via the Rich live display handler).
  * Type: `string` (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  * Default: `"ERROR"`
  * Example: `"INFO"`
* **`log_dir`**:
  * Description: The directory where log files and run-specific subdirectories will be stored.
  * Type: `string`
  * Default: `"logs"`
  * Example: `"training_logs"`
* **`log_file_prefix`**:
  * Description: A prefix used for naming log files and the run-specific subdirectory.
  * Type: `string`
  * Default: `"cambia"`
  * Example: `"cambia_experiment_A"`
* **`log_max_bytes`**:
  * Description: The maximum size (in bytes) a log file can reach before it's rotated.
  * Type: `integer`
  * Default: `9437184` (approx. 9MB)
  * Example: `10485760` (10MB)
* **`log_backup_count`**:
  * Description: The maximum number of backup log files to keep after rotation.
  * Type: `integer`
  * Default: `999`
  * Example: `10`
* **`worker_config`**:
  * Description: (Optional) Provides fine-grained logging level control for individual worker processes. If omitted, workers will use `log_level_file`.
  * Type: `object` (dictionary)
  * Fields:
    * **`default_level`**:
      * Description: The base log level for worker files, unless a sequential rule or override matches for that specific worker.
      * Type: `string` (DEBUG, INFO, WARNING, ERROR, CRITICAL)
      * Default: `"INFO"`
      * Example: `"WARNING"`
    * **`sequential_rules`**:
      * Description: A list of rules applied sequentially to workers starting from ID 0. Each rule specifies a log level and an optional count of how many workers it applies to.
      * Type: `array` (list) of `string` or `object`
      * Format:
        * Simple string: `"LEVEL"` (applies to 1 worker)
        * Object: `{"LEVEL": count}` (applies to `count` workers)
      * Example:

                ```yaml
                sequential_rules:
                  - "DEBUG"        # Worker 0: DEBUG
                  - "INFO": 2      # Workers 1-2: INFO
                  - "WARNING"      # Worker 3: WARNING
                # Subsequent workers (4+) would use worker_config.default_level
                ```

    * **`overrides`**:
      * Description: A list of specific overrides that apply to designated worker IDs, taking precedence over sequential rules and the default level.
      * Type: `array` (list) of `object`
      * Each object has:
        * `worker_ids`: A list of worker IDs (integers) this override applies to.
        * `level`: The log level (string) to apply.
      * Example:

                ```yaml
                overrides:
                  - worker_ids: [0, 3] # Workers 0 and 3
                    level: "CRITICAL"
                  - worker_ids: [5]    # Worker 5
                    level: "DEBUG"
                ```

  * Example:

        ```yaml
        worker_config:
          default_level: "INFO"
          sequential_rules:
            - "DEBUG"  # Worker 0 gets DEBUG
            - "INFO": 3 # Workers 1, 2, 3 get INFO
          overrides:
            - worker_ids: [0] # Worker 0 specifically gets WARNING, overriding the sequential DEBUG
              level: "WARNING"
            - worker_ids: [10]
              level: "ERROR"
        ```
