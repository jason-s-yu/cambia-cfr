# src/cfr_trainer.py
import logging
import copy
import sys
import time
import traceback
from collections import defaultdict, deque
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .agent_state import AgentObservation, AgentState
from .analysis_tools import AnalysisTools
from .config import Config
from .constants import (
    NUM_PLAYERS,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionAbilityPeekOtherSelect,
    ActionAbilityPeekOwnSelect,
    ActionDiscard,
    ActionReplace,
    ActionSnapOpponentMove,
    CardObject,
    DecisionContext,
    GameAction,
)
from .game.engine import CambiaGameState
from .game.types import StateDelta, UndoInfo
from .persistence import load_agent_data
from .utils import (
    InfosetKey,
    PolicyDict,
    ReachProbDict,
    get_rm_plus_strategy,
    normalize_probabilities,
)

logger = logging.getLogger(__name__)


class CFRTrainer:
    def __init__(self, config: Config, run_log_dir: Optional[str] = None):
        self.config = config
        self.num_players = NUM_PLAYERS
        self.regret_sum: PolicyDict = defaultdict(lambda: np.array([], dtype=np.float64))
        self.strategy_sum: PolicyDict = defaultdict(
            lambda: np.array([], dtype=np.float64)
        )
        self.reach_prob_sum: ReachProbDict = defaultdict(float)
        self.current_iteration = 0
        self.average_strategy: Optional[PolicyDict] = None
        analysis_log_dir = run_log_dir if run_log_dir else config.logging.log_dir
        analysis_log_prefix = config.logging.log_file_prefix
        self.analysis = AnalysisTools(config, analysis_log_dir, analysis_log_prefix)
        self.exploitability_results: List[Tuple[int, float]] = []
        self.run_log_dir = run_log_dir
        self.max_depth_this_iter = 0
        self._last_exploit_str = "N/A"
        self._total_infosets_str = "0"

    def load_data(self, filepath: Optional[str] = None):
        path = filepath or self.config.persistence.agent_data_save_path
        loaded = load_agent_data(path)
        if loaded:
            self.regret_sum = defaultdict(
                lambda: np.array([], dtype=np.float64),
                {
                    InfosetKey(*k) if isinstance(k, tuple) else k: v
                    for k, v in loaded.get("regret_sum", {}).items()
                },
            )
            self.strategy_sum = defaultdict(
                lambda: np.array([], dtype=np.float64),
                {
                    InfosetKey(*k) if isinstance(k, tuple) else k: v
                    for k, v in loaded.get("strategy_sum", {}).items()
                },
            )
            self.reach_prob_sum = defaultdict(
                float,
                {
                    InfosetKey(*k) if isinstance(k, tuple) else k: v
                    for k, v in loaded.get("reach_prob_sum", {}).items()
                },
            )
            self.current_iteration = loaded.get("iteration", 0)
            exploit_history = loaded.get("exploitability_results", [])
            if isinstance(exploit_history, list) and all(
                isinstance(item, (tuple, list)) and len(item) == 2
                for item in exploit_history
            ):
                self.exploitability_results = [
                    (int(it), float(expl)) for it, expl in exploit_history
                ]
                if self.exploitability_results:
                    last_exploit_val = self.exploitability_results[-1][1]
                    self._last_exploit_str = (
                        f"{last_exploit_val:.3f}"
                        if last_exploit_val != float("inf")
                        else "N/A"
                    )
            else:
                logger.warning(
                    "Invalid exploitability history format found in loaded data. Resetting."
                )
                self.exploitability_results = []
                self._last_exploit_str = "N/A"
            self._total_infosets_str = f"{len(self.regret_sum):,}"
            logger.info("Resuming training from iteration %d", self.current_iteration + 1)
        else:
            logger.info("No saved data found or error loading. Starting fresh.")
            self.regret_sum = defaultdict(lambda: np.array([], dtype=np.float64))
            self.strategy_sum = defaultdict(lambda: np.array([], dtype=np.float64))
            self.reach_prob_sum = defaultdict(float)
            self.current_iteration = 0
            self.exploitability_results = []
            self._last_exploit_str = "N/A"
            self._total_infosets_str = "0"

    def save_data(self, filepath: Optional[str] = None):
        from .persistence import save_agent_data

        path = filepath or self.config.persistence.agent_data_save_path
        data_to_save = {
            "regret_sum": {
                k.astuple() if isinstance(k, InfosetKey) else k: v
                for k, v in self.regret_sum.items()
            },
            "strategy_sum": {
                k.astuple() if isinstance(k, InfosetKey) else k: v
                for k, v in self.strategy_sum.items()
            },
            "reach_prob_sum": {
                k.astuple() if isinstance(k, InfosetKey) else k: v
                for k, v in self.reach_prob_sum.items()
            },
            "iteration": self.current_iteration,
            "exploitability_results": self.exploitability_results,
        }
        save_agent_data(data_to_save, path)

    def train(self, num_iterations: Optional[int] = None):
        total_iterations_to_run = (
            num_iterations or self.config.cfr_training.num_iterations
        )
        start_iteration = self.current_iteration
        end_iteration = start_iteration + total_iterations_to_run
        exploitability_interval = self.config.cfr_training.exploitability_interval
        if total_iterations_to_run <= 0:
            logger.warning("Number of iterations to run must be positive.")
            return
        logger.info(
            "Starting CFR+ training from iteration %d up to %d...",
            start_iteration + 1,
            end_iteration,
        )
        loop_start_time = time.time()

        # Create the status bar (position 0, above the main bar)
        status_bar = tqdm(
            total=0,
            position=0,
            bar_format="{desc}",
            desc="Initializing status...",
        )

        # Create the main progress bar (position 1)
        progress_bar = tqdm(
            range(start_iteration, end_iteration),
            desc="CFR+ Training",
            initial=start_iteration,
            total=end_iteration,
            unit="iter",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            position=1,  # Ensure it's below the status bar
            leave=False,  # Keep the main bar visible after loop finishes
        )

        for t in progress_bar:
            iter_start_time = time.time()
            self.current_iteration = t + 1
            self.max_depth_this_iter = 0

            # Update status bar before starting iteration
            status_bar.set_description_str(
                f"Iter {self.current_iteration}/{end_iteration} | Depth:0 | MaxD:0 | Nodes:{self._total_infosets_str} | Expl:{self._last_exploit_str}"
            )

            game_state = None
            try:
                game_state = CambiaGameState(house_rules=self.config.cambia_rules)
                initial_hands_for_log = [list(p.hand) for p in game_state.players]
                initial_peeks_for_log = [
                    p.initial_peek_indices for p in game_state.players
                ]
            except Exception as e:
                logger.exception(
                    "Error initializing game state on iteration %d: %s",
                    self.current_iteration,
                    e,
                )
                progress_bar.set_postfix_str("Error init game, skipping")
                continue
            reach_probs = np.ones(self.num_players, dtype=np.float64)
            initial_agent_states = []
            action_log_for_game: List[Dict] = []
            if not game_state.is_terminal():
                initial_obs = self._create_observation(
                    None, None, game_state, -1, [], None
                )
                for i in range(self.num_players):
                    try:
                        agent = AgentState(
                            player_id=i,
                            opponent_id=game_state.get_opponent_index(i),
                            memory_level=self.config.agent_params.memory_level,
                            time_decay_turns=self.config.agent_params.time_decay_turns,
                            initial_hand_size=len(initial_hands_for_log[i]),
                            config=self.config,
                        )
                        agent.initialize(
                            initial_obs,
                            initial_hands_for_log[i],
                            initial_peeks_for_log[i],
                        )
                        initial_agent_states.append(agent)
                    except Exception as e:
                        logger.exception(
                            "Error initializing AgentState %d on iteration %d: %s",
                            i,
                            self.current_iteration,
                            e,
                        )
                        initial_agent_states = []
                        break
            else:
                logger.error(
                    "Game is terminal immediately after initialization on iteration %d. Skipping.",
                    self.current_iteration,
                )
                progress_bar.set_postfix_str("Error game terminal init, skipping")
                continue
            if not initial_agent_states:
                progress_bar.set_postfix_str("Error init agents, skipping")
                continue
            final_utilities = None
            game_failed = False
            try:
                final_utilities = self._cfr_recursive(
                    game_state,
                    initial_agent_states,
                    reach_probs,
                    self.current_iteration,
                    action_log_for_game,
                    status_bar,  # Pass the status bar here
                    depth=0,
                )
                game_details = self.analysis.format_game_details_for_log(
                    game_state=game_state,
                    iteration=self.current_iteration,
                    initial_hands=initial_hands_for_log,
                    action_sequence=action_log_for_game,
                )
                self.analysis.log_game_history(game_details)
            except RecursionError:
                logger.error(
                    "Iter %d, MaxDepth %d: Recursion depth exceeded! Saving progress and stopping.",
                    self.current_iteration,
                    self.max_depth_this_iter,
                )
                logger.error("State at RecursionError (approx): %s", game_state)
                if action_log_for_game:
                    logger.error(
                        "Last actions:\n%s",
                        "\n".join([f"  {e}" for e in action_log_for_game[-10:]]),
                    )
                logger.error("Traceback:\n%s", traceback.format_exc())
                progress_bar.set_postfix_str("RecursionError!")
                game_failed = True
                self.save_data()
                raise
            except Exception as e:
                logger.exception(
                    "Error during CFR recursion on iteration %d: %s",
                    self.current_iteration,
                    e,
                )
                logger.error("State at Error: %s", game_state)
                progress_bar.set_postfix_str(f"CFRError! {type(e).__name__}")
                game_failed = True
            if game_failed:
                continue
            iter_time = time.time() - iter_start_time

            # --- Calculate Exploitability Periodically ---
            if (
                exploitability_interval > 0
                and self.current_iteration % exploitability_interval == 0
            ):
                exploit_start_time = time.time()
                logger.info(
                    "Calculating exploitability at iteration %d...",
                    self.current_iteration,
                )
                current_avg_strategy = self.compute_average_strategy()
                if current_avg_strategy:
                    exploit = self.analysis.calculate_exploitability(
                        current_avg_strategy, self.config
                    )
                    self.exploitability_results.append((self.current_iteration, exploit))
                    self._last_exploit_str = (
                        f"{exploit:.3f}" if exploit != float("inf") else "N/A"
                    )
                    exploit_calc_time = time.time() - exploit_start_time
                    logger.info(
                        "Exploitability calculated: %.4f (took %.2fs)",
                        exploit,
                        exploit_calc_time,
                    )
                else:
                    logger.warning(
                        "Could not compute average strategy for exploitability calculation."
                    )
                    self._last_exploit_str = "N/A"

            self._total_infosets_str = f"{len(self.regret_sum):,}"

            # Update the main progress bar's postfix *after* the iteration is done
            postfix_dict = {
                "LastT": f"{iter_time:.2f}s",
                "DepthMax": f"{self.max_depth_this_iter}",
                "Expl": self._last_exploit_str,
                "TotalNodes": self._total_infosets_str,
            }
            progress_bar.set_postfix(postfix_dict, refresh=True)

            # Update the status bar description again after iteration finishes
            status_bar.set_description_str(
                f"Iter {self.current_iteration}/{end_iteration} complete | Depth:N/A | MaxD:{self.max_depth_this_iter} | Nodes:{self._total_infosets_str} | Expl:{self._last_exploit_str}"
            )

            if self.current_iteration % self.config.cfr_training.save_interval == 0:
                self.save_data()

        # Close both bars after the loop
        status_bar.close()
        progress_bar.close()

        end_time = time.time()
        total_completed = self.current_iteration - start_iteration
        logger.info("Training loop finished %d iterations.", total_completed)
        logger.info("Total training time: %.2f seconds.", end_time - loop_start_time)
        logger.info("Computing final average strategy...")
        final_avg_strategy = self.compute_average_strategy()
        if final_avg_strategy:
            logger.info("Calculating final exploitability...")
            final_exploit = self.analysis.calculate_exploitability(
                final_avg_strategy, self.config
            )
            if (
                not self.exploitability_results
                or self.exploitability_results[-1][0] != self.current_iteration
            ):
                self.exploitability_results.append(
                    (self.current_iteration, final_exploit)
                )
            else:
                self.exploitability_results[-1] = (self.current_iteration, final_exploit)
            logger.info("Final exploitability: %.4f", final_exploit)
            self._last_exploit_str = (
                f"{final_exploit:.3f}" if final_exploit != float("inf") else "N/A"
            )
        else:
            logger.warning("Could not compute final average strategy.")
            self._last_exploit_str = "N/A"

        # Final update to status bar (can be useful in some terminals)
        tqdm.write(
            f"Final State | MaxDepth:{self.max_depth_this_iter} | Nodes:{self._total_infosets_str} | Expl:{self._last_exploit_str}",
            file=sys.stderr,
        )

        self.save_data()
        logger.info("Final average strategy and data saved.")

    def _cfr_recursive(
        self,
        game_state: CambiaGameState,
        agent_states: List[AgentState],
        reach_probs: np.ndarray,
        iteration: int,
        action_log: List[Dict],
        status_bar: tqdm,  # Pass the status bar instead of progress bar
        depth: int,
    ) -> np.ndarray:
        self.max_depth_this_iter = max(self.max_depth_this_iter, depth)

        # --- Update Status Bar ---
        total_infosets = len(self.regret_sum)
        # Avoid recomputing exploitability here, use the last known value
        status_desc = f"Iter {iteration} | Depth:{depth} | MaxD:{self.max_depth_this_iter} | Nodes:{total_infosets:,} | Expl:{self._last_exploit_str}"
        status_bar.set_description_str(status_desc)

        # --- Base Case: Terminal Node ---
        current_context = DecisionContext.TERMINAL
        if game_state.is_terminal():
            return np.array(
                [game_state.get_utility(i) for i in range(self.num_players)],
                dtype=np.float64,
            )

        # --- Determine Node Type and Context ---
        elif game_state.snap_phase_active:
            current_context = DecisionContext.SNAP_DECISION
        elif game_state.pending_action:
            pending = game_state.pending_action
            if isinstance(pending, ActionDiscard):
                current_context = DecisionContext.POST_DRAW
            elif isinstance(
                pending,
                (
                    ActionAbilityPeekOwnSelect,
                    ActionAbilityPeekOtherSelect,
                    ActionAbilityBlindSwapSelect,
                    ActionAbilityKingLookSelect,
                    ActionAbilityKingSwapDecision,
                ),
            ):
                current_context = DecisionContext.ABILITY_SELECT
            elif isinstance(pending, ActionSnapOpponentMove):
                current_context = DecisionContext.SNAP_MOVE
            else:
                logger.warning(
                    "Iter %d, Depth %d: Unhandled pending action type %s for context determination.",
                    iteration,
                    depth,
                    type(pending),
                )
                current_context = DecisionContext.START_TURN
        else:
            current_context = DecisionContext.START_TURN

        # --- Get Acting Player and Infoset ---
        player = game_state.get_acting_player()
        if player == -1:
            logger.error(
                "Iter %d, Depth %d: Could not determine acting player. State: %s. Context: %s",
                iteration,
                depth,
                game_state,
                current_context.name,
            )
            return np.zeros(self.num_players, dtype=np.float64)
        current_agent_state = agent_states[player]
        opponent = 1 - player
        try:
            if not hasattr(current_agent_state, "own_hand") or not hasattr(
                current_agent_state, "opponent_belief"
            ):
                logger.error(
                    "Iter %d, P%d, Depth %d: AgentState uninitialized. State: %s",
                    iteration,
                    player,
                    depth,
                    current_agent_state,
                )
                return np.zeros(self.num_players, dtype=np.float64)
            base_infoset_tuple = current_agent_state.get_infoset_key()
            infoset_key = InfosetKey(*base_infoset_tuple, current_context.value)
        except Exception:
            logger.exception(
                "Iter %d, P%d, Depth %d: Error getting infoset key. AgentState: %s. GameState: %s. Context: %s",
                iteration,
                player,
                depth,
                current_agent_state,
                game_state,
                current_context.name,
            )
            return np.zeros(self.num_players, dtype=np.float64)

        # --- Get Legal Actions ---
        try:
            legal_actions_set = game_state.get_legal_actions()
            legal_actions = sorted(list(legal_actions_set), key=repr)
        except Exception as e:
            logger.exception(
                "Iter %d, P%d, Depth %d: Error getting legal actions. State: %s. InfosetKey: %s. Context: %s: %s",
                iteration,
                player,
                depth,
                game_state,
                infoset_key,
                current_context.name,
                e,
            )
            return np.zeros(self.num_players, dtype=np.float64)
        num_actions = len(legal_actions)

        # --- Handle No Legal Actions ---
        if num_actions == 0:
            if not game_state.is_terminal():
                logger.warning(
                    "Iter %d, P%d, Depth %d: No legal actions found at infoset %s in non-terminal state %s (Context: %s). Forcing end check.",
                    iteration,
                    player,
                    depth,
                    infoset_key,
                    game_state,
                    current_context.name,
                )
                local_undo_stack: deque[Callable[[], None]] = deque()
                local_delta_list: StateDelta = []
                game_state._check_game_end(local_undo_stack, local_delta_list)
                while local_undo_stack:
                    local_undo_stack.popleft()()
                if game_state.is_terminal():
                    return np.array(
                        [game_state.get_utility(i) for i in range(self.num_players)],
                        dtype=np.float64,
                    )
                else:
                    logger.error(
                        "Iter %d, P%d, Depth %d: State still non-terminal after no-action check. Logic error.",
                        iteration,
                        player,
                        depth,
                    )
                    return np.zeros(self.num_players, dtype=np.float64)
            else:  # Already terminal, should have been caught earlier
                return np.array(
                    [game_state.get_utility(i) for i in range(self.num_players)],
                    dtype=np.float64,
                )

        # --- Initialize/Retrieve Regret and Strategy Sums ---
        current_regrets = self.regret_sum.get(infoset_key)
        if current_regrets is None or len(current_regrets) != num_actions:
            if current_regrets is not None:
                logger.warning(
                    "## TEMP DEBUG ## Iter %d, P%d, Depth %d, Infoset %s: Regret dimension mismatch! Existing: %d, Expected: %d. Context: %s. Re-initializing.",
                    iteration,
                    player,
                    depth,
                    infoset_key,
                    len(current_regrets),
                    num_actions,
                    current_context.name,
                )
            self.regret_sum[infoset_key] = np.zeros(num_actions, dtype=np.float64)
            current_regrets = self.regret_sum[infoset_key]
        current_strategy_sum = self.strategy_sum.get(infoset_key)
        if current_strategy_sum is None or len(current_strategy_sum) != num_actions:
            if current_strategy_sum is not None:
                logger.warning(
                    "## TEMP DEBUG ## Iter %d, P%d, Depth %d, Infoset %s: Strategy sum dimension mismatch! Existing: %d, Expected: %d. Context: %s. Re-initializing.",
                    iteration,
                    player,
                    depth,
                    infoset_key,
                    len(current_strategy_sum),
                    num_actions,
                    current_context.name,
                )
            self.strategy_sum[infoset_key] = np.zeros(num_actions, dtype=np.float64)
        if infoset_key not in self.reach_prob_sum:
            self.reach_prob_sum[infoset_key] = 0.0

        # --- Calculate Current Strategy & Update Strategy Sum ---
        strategy = get_rm_plus_strategy(current_regrets)
        player_reach = reach_probs[player]
        if player_reach > 1e-9:
            if self.config.cfr_plus_params.weighted_averaging_enabled:
                delay = self.config.cfr_plus_params.averaging_delay
                weight = float(max(0, iteration - delay))
            else:
                weight = 1.0
            if weight > 0:
                self.reach_prob_sum[infoset_key] += weight * player_reach
                self.strategy_sum[infoset_key] += weight * player_reach * strategy

        # --- Iterate Through Actions ---
        action_utilities = np.zeros((num_actions, self.num_players), dtype=np.float64)
        node_value = np.zeros(self.num_players, dtype=np.float64)
        use_pruning = self.config.cfr_training.pruning_enabled
        pruning_threshold = self.config.cfr_training.pruning_threshold

        for i, action in enumerate(legal_actions):
            action_prob = strategy[i]
            # --- Regret Pruning ---
            node_positive_regret_sum = np.sum(np.maximum(0.0, current_regrets))
            should_prune = (
                use_pruning
                and player_reach > 1e-6
                and current_regrets[i] <= pruning_threshold
                and node_positive_regret_sum > pruning_threshold
                and iteration > self.config.cfr_plus_params.averaging_delay + 10
            )
            if should_prune:
                continue
            if action_prob < 1e-9:
                continue

            # --- Apply Action and Recurse ---
            action_log_entry = {
                "player": player,
                "turn": game_state.get_turn_number(),
                "depth": depth,
                "context": current_context.name,
                "state_desc_before": str(game_state),
                "infoset_key": infoset_key,
                "action": action,
                "action_prob": action_prob,
                "reach_prob": player_reach,
            }
            state_before_this_action_str = action_log_entry["state_desc_before"]
            state_delta: Optional[StateDelta] = None
            undo_info: Optional[UndoInfo] = None
            state_after_action_desc = "ERROR"
            drawn_card_this_step: Optional[CardObject] = None
            try:
                if isinstance(
                    action, (ActionReplace, ActionDiscard)
                ) and game_state.pending_action_data.get("drawn_card"):
                    drawn_card_this_step = game_state.pending_action_data["drawn_card"]
                state_delta, undo_info = game_state.apply_action(action)
                state_after_action_desc = str(game_state)
            except Exception as apply_err:
                logger.exception(
                    "Iter %d, P%d, Depth %d: Error applying action %s in state %s at infoset %s: %s",
                    iteration,
                    player,
                    depth,
                    action,
                    state_before_this_action_str,
                    infoset_key,
                    apply_err,
                )
                action_log_entry["outcome"] = f"ERROR applying action: {apply_err}"
                action_log.append(action_log_entry)
                undo_info = None
                continue
            observation = self._create_observation(
                None,
                action,
                game_state,
                player,
                game_state.snap_results_log,
                drawn_card_this_step,
            )
            next_agent_states = []
            agent_update_failed = False
            for agent_idx, agent_state in enumerate(agent_states):
                cloned_agent = agent_state.clone()
                try:
                    player_specific_obs = self._filter_observation(observation, agent_idx)
                    cloned_agent.update(player_specific_obs)
                except Exception as e:
                    logger.exception(
                        "Iter %d, P%d, Depth %d: Error updating AgentState %d for action %s. Infoset: %s. Obs: %s: %s",
                        iteration,
                        player,
                        depth,
                        agent_idx,
                        action,
                        infoset_key,
                        observation,
                        e,
                    )
                    agent_update_failed = True
                    break
                next_agent_states.append(cloned_agent)
            if agent_update_failed:
                action_log_entry["outcome"] = "ERROR updating agent state"
                action_log.append(action_log_entry)
                if undo_info:
                    undo_info()
                continue
            next_reach_probs = reach_probs.copy()
            next_reach_probs[player] *= action_prob
            try:
                recursive_utilities = self._cfr_recursive(
                    game_state,
                    next_agent_states,
                    next_reach_probs,
                    iteration,
                    action_log,
                    status_bar,  # Pass status bar down
                    depth + 1,
                )
                action_utilities[i] = recursive_utilities
            except RecursionError as rec_err:
                logger.error(
                    "Iter %d, P%d, Depth %d: Recursion depth exceeded during action: %s. State: %s",
                    iteration,
                    player,
                    depth,
                    action,
                    game_state,
                )
                action_log_entry["outcome"] = f"ERROR: RecursionError - {rec_err}"
                action_log.append(action_log_entry)
                raise
            except Exception as recursive_err:
                logger.exception(
                    "Iter %d, P%d, Depth %d: Error in recursive call for action %s. Infoset: %s: %s",
                    iteration,
                    player,
                    depth,
                    action,
                    infoset_key,
                    recursive_err,
                )
                action_log_entry["outcome"] = f"ERROR in recursion: {recursive_err}"
                action_log.append(action_log_entry)
                if undo_info:
                    undo_info()
                continue
            action_log_entry["outcome_utilities"] = action_utilities[i].tolist()
            action_log_entry["state_desc_after"] = state_after_action_desc
            action_log.append(action_log_entry)
            if undo_info:
                try:
                    undo_info()
                except Exception as undo_e:
                    logger.exception(
                        "FATAL: Iter %d, P%d, Depth %d: Error undoing action %s. State corrupt: %s",
                        iteration,
                        player,
                        depth,
                        action,
                        undo_e,
                    )
                    return np.zeros(self.num_players, dtype=np.float64)
            else:
                logger.error(
                    "Iter %d, P%d, Depth %d: Missing undo info for action %s. State corrupt.",
                    iteration,
                    player,
                    depth,
                    action,
                )
                return np.zeros(self.num_players, dtype=np.float64)

        # --- Calculate Node Value & Update Regrets ---
        node_value = np.sum(strategy[:, np.newaxis] * action_utilities, axis=0)
        opponent_reach = reach_probs[opponent]
        if player_reach > 1e-9:
            player_action_values = action_utilities[:, player]
            player_node_value = node_value[player]
            instantaneous_regret = player_action_values - player_node_value
            update_weight = opponent_reach
            current_regrets_before_update = np.copy(self.regret_sum[infoset_key])
            try:
                updated_regrets = (
                    current_regrets_before_update + update_weight * instantaneous_regret
                )
                self.regret_sum[infoset_key] = np.maximum(0.0, updated_regrets)
            except ValueError as e_regret:
                logger.exception(
                    "Iter %d, P%d, Depth %d, Infoset %s, Context %s: ValueError during regret update: %s",
                    iteration,
                    player,
                    depth,
                    infoset_key,
                    current_context.name,
                    e_regret,
                )
            except Exception as e_regret:
                logger.exception(
                    "Iter %d, P%d, Depth %d, Infoset %s, Context %s: Unexpected error during regret update: %s",
                    iteration,
                    player,
                    depth,
                    infoset_key,
                    current_context.name,
                    e_regret,
                )
        return node_value

    def _filter_observation(
        self, obs: AgentObservation, observer_id: int
    ) -> AgentObservation:
        filtered_obs = copy.copy(obs)
        if obs.drawn_card and obs.acting_player != observer_id:
            if not isinstance(obs.action, ActionDiscard) and not (
                isinstance(obs.action, ActionReplace) and obs.acting_player == observer_id
            ):
                filtered_obs.drawn_card = None
        if obs.peeked_cards:
            new_peeked = {}
            if (
                obs.action
                and isinstance(obs.action, ActionAbilityKingSwapDecision)
                and obs.acting_player == observer_id
            ):
                new_peeked = obs.peeked_cards
            elif (
                obs.action
                and isinstance(obs.action, ActionAbilityPeekOwnSelect)
                and obs.acting_player == observer_id
            ):
                new_peeked = obs.peeked_cards
            elif (
                obs.action
                and isinstance(obs.action, ActionAbilityPeekOtherSelect)
                and obs.acting_player == observer_id
            ):
                new_peeked = obs.peeked_cards
            filtered_obs.peeked_cards = new_peeked if new_peeked else None
        filtered_obs.snap_results = obs.snap_results
        return filtered_obs

    def _create_observation(
        self,
        prev_state: Optional[CambiaGameState],
        action: Optional[GameAction],
        next_state: CambiaGameState,
        acting_player: int,
        snap_results: List[Dict],
        explicit_drawn_card: Optional[CardObject] = None,
    ) -> AgentObservation:
        discard_top = next_state.get_discard_top()
        hand_sizes = [
            next_state.get_player_card_count(i) for i in range(self.num_players)
        ]
        stock_size = next_state.get_stockpile_size()
        cambia_called = next_state.cambia_caller_id is not None
        who_called = next_state.cambia_caller_id
        game_over = next_state.is_terminal()
        turn_num = next_state.get_turn_number()
        drawn_card = explicit_drawn_card
        peeked_cards_dict = None
        if (
            isinstance(next_state.pending_action, ActionAbilityKingSwapDecision)
            and next_state.pending_action_player == acting_player
        ):
            pending_data = next_state.pending_action_data
            if (
                "own_idx" in pending_data
                and "opp_idx" in pending_data
                and "card1" in pending_data
                and "card2" in pending_data
            ):
                opp_idx = next_state.get_opponent_index(acting_player)
                peeked_cards_dict = {
                    (acting_player, pending_data["own_idx"]): pending_data["card1"],
                    (opp_idx, pending_data["opp_idx"]): pending_data["card2"],
                }
        elif isinstance(action, ActionAbilityPeekOwnSelect):
            hand = next_state.get_player_hand(acting_player)
            if hand and 0 <= action.target_hand_index < len(hand):
                peeked_cards_dict = {
                    (acting_player, action.target_hand_index): hand[
                        action.target_hand_index
                    ]
                }
        elif isinstance(action, ActionAbilityPeekOtherSelect):
            opp_idx = next_state.get_opponent_index(acting_player)
            opp_hand = next_state.get_player_hand(opp_idx)
            if opp_hand and 0 <= action.target_opponent_hand_index < len(opp_hand):
                peeked_cards_dict = {
                    (opp_idx, action.target_opponent_hand_index): opp_hand[
                        action.target_opponent_hand_index
                    ]
                }
        final_snap_results = next_state.snap_results_log
        obs = AgentObservation(
            acting_player=acting_player,
            action=action,
            discard_top_card=discard_top,
            player_hand_sizes=hand_sizes,
            stockpile_size=stock_size,
            drawn_card=drawn_card,
            peeked_cards=peeked_cards_dict,
            snap_results=final_snap_results,
            did_cambia_get_called=cambia_called,
            who_called_cambia=who_called,
            is_game_over=game_over,
            current_turn=turn_num,
        )
        return obs

    def compute_average_strategy(self) -> PolicyDict:
        avg_strategy: PolicyDict = {}
        logger.info(
            "Computing average strategy from %d infosets...", len(self.strategy_sum)
        )
        if not self.strategy_sum:
            logger.warning("Strategy sum is empty.")
            return avg_strategy
        zero_reach_count, nan_count, norm_issue_count, mismatched_dim_count = 0, 0, 0, 0
        for infoset_key, s_sum in self.strategy_sum.items():
            if isinstance(infoset_key, tuple):
                infoset_key = InfosetKey(*infoset_key)
            r_sum = self.reach_prob_sum.get(infoset_key, 0.0)
            num_actions_in_sum = len(s_sum)
            normalized_strategy = np.array([])
            if r_sum > 1e-9:
                normalized_strategy = s_sum / r_sum
                if np.isnan(normalized_strategy).any():
                    nan_count += 1
                    logger.warning(
                        "NaN found in avg strategy for %s. Using uniform.", infoset_key
                    )
                    normalized_strategy = (
                        np.ones(num_actions_in_sum) / num_actions_in_sum
                        if num_actions_in_sum > 0
                        else np.array([])
                    )
                current_sum = np.sum(normalized_strategy)
                if (
                    not np.isclose(current_sum, 1.0, atol=1e-6)
                    and len(normalized_strategy) > 0
                ):
                    normalized_strategy = normalize_probabilities(normalized_strategy)
                    if not np.isclose(np.sum(normalized_strategy), 1.0, atol=1e-6):
                        norm_issue_count += 1
                        logger.warning("Avg strategy re-norm failed for %s", infoset_key)
            else:
                if np.any(s_sum != 0):
                    zero_reach_count += 1
                normalized_strategy = (
                    np.ones(num_actions_in_sum) / num_actions_in_sum
                    if num_actions_in_sum > 0
                    else np.array([])
                )
            regret_array = self.regret_sum.get(infoset_key)
            if regret_array is not None and len(regret_array) != len(normalized_strategy):
                mismatched_dim_count += 1
                num_actions_regret = len(regret_array)
                logger.warning(
                    "Avg strategy dim (%d) mismatch with regret (%d) for %s. Context: %s. Defaulting avg strategy.",
                    len(normalized_strategy),
                    num_actions_regret,
                    infoset_key,
                    DecisionContext(infoset_key.decision_context_value).name,
                )
                normalized_strategy = (
                    np.ones(num_actions_regret) / num_actions_regret
                    if num_actions_regret > 0
                    else np.array([])
                )
            avg_strategy[infoset_key] = normalized_strategy
        self.average_strategy = avg_strategy
        logger.info(
            "Average strategy computation complete (%d infosets).",
            len(self.average_strategy),
        )
        if zero_reach_count > 0:
            logger.warning(
                "%d infosets with zero reach sum but non-zero strategy sum.",
                zero_reach_count,
            )
        if nan_count > 0:
            logger.warning("%d infosets with NaN strategy.", nan_count)
        if norm_issue_count > 0:
            logger.warning("%d infosets with norm issues.", norm_issue_count)
        if mismatched_dim_count > 0:
            logger.warning(
                "%d infosets with final dimension mismatch (avg vs regret).",
                mismatched_dim_count,
            )
        return avg_strategy

    def get_average_strategy(self) -> Optional[PolicyDict]:
        if self.average_strategy is None:
            logger.warning(
                "Average strategy requested but not computed yet. Computing now..."
            )
            return self.compute_average_strategy()
        return self.average_strategy
