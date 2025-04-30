# src/cfr_trainer.py
import numpy as np
import time
import logging
from typing import Callable, Dict, List, Optional, Tuple
from collections import defaultdict, deque
import copy
import traceback
from tqdm import tqdm

from .game_engine import CambiaGameState, StateDelta, UndoInfo
from .constants import (
     ActionAbilityKingSwapDecision, GameAction, NUM_PLAYERS,
     ActionPassSnap, ActionSnapOwn, ActionSnapOpponent, ActionSnapOpponentMove,
     ActionDrawStockpile, ActionDrawDiscard, ActionReplace, ActionDiscard,
     ActionAbilityPeekOwnSelect, ActionAbilityPeekOtherSelect, ActionAbilityKingLookSelect,
     ActionAbilityBlindSwapSelect,
     DecisionContext
)
from .agent_state import AgentState, AgentObservation
from .utils import InfosetKey, PolicyDict, ReachProbDict, get_rm_plus_strategy, normalize_probabilities
from .config import Config
from .analysis_tools import AnalysisTools

logger = logging.getLogger(__name__)

class CFRTrainer:
    """Implements the CFR+ algorithm for training a Cambia agent via self-play."""

    def __init__(self, config: Config, run_log_dir: Optional[str] = None): # Accept run_log_dir
        self.config = config
        self.num_players = NUM_PLAYERS
        self.regret_sum: PolicyDict = defaultdict(lambda: np.array([], dtype=np.float64))
        self.strategy_sum: PolicyDict = defaultdict(lambda: np.array([], dtype=np.float64))
        self.reach_prob_sum: ReachProbDict = defaultdict(float)
        self.current_iteration = 0
        self.average_strategy: Optional[PolicyDict] = None
        # Initialize AnalysisTools with run_log_dir if available
        analysis_log_dir = run_log_dir if run_log_dir else config.logging.log_dir
        analysis_log_prefix = config.logging.log_file_prefix
        self.analysis = AnalysisTools(config, analysis_log_dir, analysis_log_prefix)
        self.exploitability_results: List[Tuple[int, float]] = []
        self.run_log_dir = run_log_dir # Store for potential later use

    def load_data(self, filepath: Optional[str] = None):
        """Loads previously saved training data."""
        from .persistence import load_agent_data # Local import OK here
        path = filepath or self.config.persistence.agent_data_save_path
        loaded = load_agent_data(path)
        if loaded:
             # Ensure keys loaded from joblib (potentially tuples) are converted to InfosetKey dataclass
             self.regret_sum = defaultdict(lambda: np.array([], dtype=np.float64),
                                          {InfosetKey(*k) if isinstance(k, tuple) else k: v for k, v in loaded.get('regret_sum', {}).items()})
             self.strategy_sum = defaultdict(lambda: np.array([], dtype=np.float64),
                                           {InfosetKey(*k) if isinstance(k, tuple) else k: v for k, v in loaded.get('strategy_sum', {}).items()})
             self.reach_prob_sum = defaultdict(float,
                                            {InfosetKey(*k) if isinstance(k, tuple) else k: v for k, v in loaded.get('reach_prob_sum', {}).items()})
             self.current_iteration = loaded.get('iteration', 0)
             # Load exploitability history safely
             exploit_history = loaded.get('exploitability_results', [])
             if isinstance(exploit_history, list) and all(isinstance(item, (tuple, list)) and len(item) == 2 for item in exploit_history):
                 self.exploitability_results = [(int(it), float(expl)) for it, expl in exploit_history]
             else:
                 logger.warning("Invalid exploitability history format found in loaded data. Resetting.")
                 self.exploitability_results = []

             logger.info(f"Resuming training from iteration {self.current_iteration + 1}")
        else:
             logger.info("No saved data found or error loading. Starting fresh.")
             self.regret_sum = defaultdict(lambda: np.array([], dtype=np.float64))
             self.strategy_sum = defaultdict(lambda: np.array([], dtype=np.float64))
             self.reach_prob_sum = defaultdict(float)
             self.current_iteration = 0
             self.exploitability_results = []


    def save_data(self, filepath: Optional[str] = None):
        """Saves the current training data."""
        from .persistence import save_agent_data # Local import OK here
        path = filepath or self.config.persistence.agent_data_save_path
        # Convert InfosetKey dataclasses back to tuples for saving with joblib/pickle
        data_to_save = {
            'regret_sum': {k.astuple() if isinstance(k, InfosetKey) else k: v for k, v in self.regret_sum.items()},
            'strategy_sum': {k.astuple() if isinstance(k, InfosetKey) else k: v for k, v in self.strategy_sum.items()},
            'reach_prob_sum': {k.astuple() if isinstance(k, InfosetKey) else k: v for k, v in self.reach_prob_sum.items()},
            'iteration': self.current_iteration,
            'exploitability_results': self.exploitability_results # Save exploitability history
        }
        save_agent_data(data_to_save, path)


    def train(self, num_iterations: Optional[int] = None):
        """Runs the CFR+ training loop with tqdm progress bar."""
        total_iterations_to_run = num_iterations or self.config.cfr_training.num_iterations
        start_iteration = self.current_iteration # Iteration number starts from 0 internally
        end_iteration = start_iteration + total_iterations_to_run # Target iteration number (exclusive in range)
        exploitability_interval = self.config.cfr_training.exploitability_interval

        if total_iterations_to_run <= 0:
             logger.warning("Number of iterations to run must be positive.")
             return

        logger.info(f"Starting CFR+ training from iteration {start_iteration + 1} up to {end_iteration}...")
        loop_start_time = time.time()

        # Setup tqdm Progress Bar
        # Use initial=start_iteration to show correct starting point if resuming
        # total=end_iteration makes the bar represent the target number
        progress_bar = tqdm(
            range(start_iteration, end_iteration),
            desc="CFR+ Training",
            initial=start_iteration, # Start counting from here
            total=end_iteration,     # Target count
            unit="iter",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )

        # --- Training Loop ---
        for t in progress_bar:
            iter_start_time = time.time()
            self.current_iteration = t + 1 # User-facing iteration number (starts from 1)

            # --- Initialize Game ---
            game_state = None
            try:
                game_state = CambiaGameState(house_rules=self.config.cambia_rules)
                initial_hands_for_log = [list(p.hand) for p in game_state.players]
                initial_peeks_for_log = [p.initial_peek_indices for p in game_state.players]
            except Exception as e:
                logger.exception(f"Error initializing game state on iteration {self.current_iteration}: {e}")
                progress_bar.set_postfix_str("Error init game, skipping")
                continue

            reach_probs = np.ones(self.num_players, dtype=np.float64)
            initial_agent_states = []
            action_log_for_game: List[Dict] = []

            if not game_state.is_terminal():
                 initial_obs = self._create_observation(None, None, game_state, -1, [])
                 for i in range(self.num_players):
                      try:
                           agent = AgentState(
                               player_id=i, opponent_id=game_state.get_opponent_index(i),
                               memory_level=self.config.agent_params.memory_level, time_decay_turns=self.config.agent_params.time_decay_turns,
                               initial_hand_size=len(initial_hands_for_log[i]), config=self.config
                           )
                           agent.initialize(initial_obs, initial_hands_for_log[i], initial_peeks_for_log[i])
                           initial_agent_states.append(agent)
                      except Exception as e:
                           logger.exception(f"Error initializing AgentState {i} on iteration {self.current_iteration}: {e}")
                           initial_agent_states = []
                           break
            else:
                 logger.error(f"Game is terminal immediately after initialization on iteration {self.current_iteration}. Skipping.")
                 progress_bar.set_postfix_str("Error game terminal init, skipping")
                 continue

            if not initial_agent_states:
                 progress_bar.set_postfix_str("Error init agents, skipping")
                 continue

            final_utilities = None
            game_failed = False
            try:
                 # --- Run CFR Recursion ---
                 final_utilities = self._cfr_recursive(game_state, initial_agent_states, reach_probs, self.current_iteration, action_log_for_game, depth=0)

                 # --- Log Completed Game ---
                 game_details = self.analysis.format_game_details_for_log(
                      game_state=game_state, iteration=self.current_iteration,
                      initial_hands=initial_hands_for_log, action_sequence=action_log_for_game
                  )
                 self.analysis.log_game_history(game_details)

            except RecursionError:
                logger.error(f"Recursion depth exceeded on iteration {self.current_iteration}! Saving progress and stopping.")
                logger.error(f"State at RecursionError (approx): {game_state}")
                if action_log_for_game: logger.error("Last actions:\n%s", "\n".join([f"  {e}" for e in action_log_for_game[-10:]]))
                logger.error("Traceback:\n%s", traceback.format_exc())
                progress_bar.set_postfix_str("RecursionError!")
                game_failed = True
                self.save_data()
                raise # Re-raise to stop training
            except Exception as e:
                 logger.exception(f"Error during CFR recursion on iteration {self.current_iteration}: {e}")
                 logger.error(f"State at Error: {game_state}")
                 progress_bar.set_postfix_str(f"CFRError! {type(e).__name__}")
                 game_failed = True
                 # Continue to next iteration unless it's fatal

            if game_failed: continue # Skip exploitability calc etc. if game failed

            iter_time = time.time() - iter_start_time
            last_exploit = self.exploitability_results[-1][1] if self.exploitability_results else float('inf')
            exploit_str = f"Expl: {last_exploit:.3f}" if last_exploit != float('inf') else "Expl: N/A"

            # --- Calculate Exploitability Periodically ---
            if self.current_iteration % exploitability_interval == 0:
                 exploit_start_time = time.time()
                 logger.info(f"Calculating exploitability at iteration {self.current_iteration}...")
                 current_avg_strategy = self.compute_average_strategy() # Ensure it's up-to-date
                 if current_avg_strategy:
                      exploit = self.analysis.calculate_exploitability(current_avg_strategy, self.config)
                      self.exploitability_results.append((self.current_iteration, exploit))
                      exploit_str = f"Expl: {exploit:.3f}" # Update string for progress bar
                      exploit_calc_time = time.time() - exploit_start_time
                      logger.info(f"Exploitability calculated: {exploit:.4f} (took {exploit_calc_time:.2f}s)")
                 else:
                      logger.warning("Could not compute average strategy for exploitability calculation.")
                      exploit_str = "Expl: N/A"

            # --- Update Progress Bar Postfix ---
            infoset_count = len(self.regret_sum)
            postfix_dict = {
                "LastT": f"{iter_time:.2f}s",
                "InfoSets": f"{infoset_count:,}",
                exploit_str.split(':')[0].strip(): exploit_str.split(':')[1].strip() # Add exploitability nicely
            }
            progress_bar.set_postfix(postfix_dict, refresh=True)

            # --- Periodic Saving ---
            if self.current_iteration % self.config.cfr_training.save_interval == 0:
                self.save_data()

        # --- End of Loop ---
        progress_bar.close() # Close the progress bar
        end_time = time.time()
        total_completed = self.current_iteration - start_iteration
        logger.info(f"Training loop finished {total_completed} iterations.")
        logger.info(f"Total training time: {end_time - loop_start_time:.2f} seconds.")

        # --- Final Calculations and Save ---
        logger.info("Computing final average strategy...")
        final_avg_strategy = self.compute_average_strategy()
        if final_avg_strategy:
            logger.info("Calculating final exploitability...")
            final_exploit = self.analysis.calculate_exploitability(final_avg_strategy, self.config)
            # Only add if it's a new iteration not already calculated
            if not self.exploitability_results or self.exploitability_results[-1][0] != self.current_iteration:
                self.exploitability_results.append((self.current_iteration, final_exploit))
            else: # Update last entry if calculated on the very last iteration
                self.exploitability_results[-1] = (self.current_iteration, final_exploit)
            logger.info(f"Final exploitability: {final_exploit:.4f}")
        else:
            logger.warning("Could not compute final average strategy.")

        self.save_data() # Save final data including exploitability
        logger.info("Final average strategy and data saved.")


    def _cfr_recursive(self, game_state: CambiaGameState, agent_states: List[AgentState], reach_probs: np.ndarray, iteration: int, action_log: List[Dict], depth: int) -> np.ndarray:
        """
        Recursive CFR+ function. Operates on the *same* game_state object,
        applying and undoing actions. Clones agent_states. Includes depth tracking.
        """
        # --- Determine Decision Context ---
        current_context = DecisionContext.TERMINAL
        if game_state.is_terminal():
            return np.array([game_state.get_utility(i) for i in range(self.num_players)], dtype=np.float64)
        elif game_state.snap_phase_active:
            current_context = DecisionContext.SNAP_DECISION
        elif game_state.pending_action:
            pending = game_state.pending_action
            if isinstance(pending, ActionDiscard): current_context = DecisionContext.POST_DRAW
            elif isinstance(pending, (ActionAbilityPeekOwnSelect, ActionAbilityPeekOtherSelect, ActionAbilityBlindSwapSelect, ActionAbilityKingLookSelect, ActionAbilityKingSwapDecision)): current_context = DecisionContext.ABILITY_SELECT
            elif isinstance(pending, ActionSnapOpponentMove): current_context = DecisionContext.SNAP_MOVE
            else:
                 logger.warning(f"Iter {iteration}, Depth {depth}: Unhandled pending action type {type(pending)} for context determination."); current_context = DecisionContext.START_TURN
        else: current_context = DecisionContext.START_TURN

        player = game_state.get_acting_player()
        if player == -1:
             logger.error(f"Iter {iteration}, Depth {depth}: Could not determine acting player. State: {game_state}. Context: {current_context.name}"); return np.zeros(self.num_players, dtype=np.float64)

        current_agent_state = agent_states[player]
        opponent = 1 - player

        # 1. Get Infoset Key (using the new Dataclass)
        try:
             if not hasattr(current_agent_state, 'own_hand') or not hasattr(current_agent_state, 'opponent_belief'):
                  logger.error(f"Iter {iteration}, P{player}, Depth {depth}: AgentState uninitialized. State: {current_agent_state}"); return np.zeros(self.num_players, dtype=np.float64)
             base_infoset_tuple = current_agent_state.get_infoset_key()
             infoset_key = InfosetKey(*base_infoset_tuple, current_context.value)
        except Exception as e:
             logger.error(f"Iter {iteration}, P{player}, Depth {depth}: Error getting infoset key. AgentState: {current_agent_state}. GameState: {game_state}. Context: {current_context.name}", exc_info=True); return np.zeros(self.num_players, dtype=np.float64)

        # 2. Get Legal Actions
        try:
            legal_actions_set = game_state.get_legal_actions()
            legal_actions = sorted(list(legal_actions_set), key=repr)
        except Exception as e:
            logger.error(f"Iter {iteration}, P{player}, Depth {depth}: Error getting legal actions. State: {game_state}. InfosetKey: {infoset_key}. Context: {current_context.name}", exc_info=True); return np.zeros(self.num_players, dtype=np.float64)

        num_actions = len(legal_actions)
        if num_actions == 0:
             if not game_state.is_terminal():
                  logger.warning(f"Iter {iteration}, P{player}, Depth {depth}: No legal actions found at infoset {infoset_key} in non-terminal state {game_state} (Context: {current_context.name}). Forcing end check.")
                  local_undo_stack: deque[Callable[[], None]] = deque(); local_delta_list: StateDelta = []
                  game_state._check_game_end(local_undo_stack, local_delta_list)
                  while local_undo_stack: local_undo_stack.popleft()() # Just undo
                  if game_state.is_terminal(): return np.array([game_state.get_utility(i) for i in range(self.num_players)], dtype=np.float64)
                  else: logger.error(f"Iter {iteration}, P{player}, Depth {depth}: State still non-terminal after no-action check. Logic error."); return np.zeros(self.num_players, dtype=np.float64)
             else: return np.array([game_state.get_utility(i) for i in range(self.num_players)], dtype=np.float64)

        # 3. Initialize/Validate Infoset Data Dimensions
        current_regrets = self.regret_sum.get(infoset_key)
        if current_regrets is None or len(current_regrets) != num_actions:
             # Log first occurrence per infoset key to reduce noise
             # if current_regrets is not None and not getattr(self, '_logged_mismatch', set()).intersection({infoset_key}):
             #      logger.warning(f"Iter {iteration}, P{player}, Depth {depth}, Infoset {infoset_key}: Regret dimension mismatch! Existing: {len(current_regrets)}, Expected: {num_actions}. Context: {current_context.name}. Re-initializing.")
             #      if not hasattr(self, '_logged_mismatch'): self._logged_mismatch = set()
             #      self._logged_mismatch.add(infoset_key)
             self.regret_sum[infoset_key] = np.zeros(num_actions, dtype=np.float64)
             current_regrets = self.regret_sum[infoset_key]

        current_strategy_sum = self.strategy_sum.get(infoset_key)
        if current_strategy_sum is None or len(current_strategy_sum) != num_actions:
             # if current_strategy_sum is not None and not getattr(self, '_logged_mismatch', set()).intersection({infoset_key}): # Reuse flag
             #      logger.warning(f"Iter {iteration}, P{player}, Depth {depth}, Infoset {infoset_key}: Strategy sum dimension mismatch! Existing: {len(current_strategy_sum)}, Expected: {num_actions}. Context: {current_context.name}. Re-initializing.")
             #      if not hasattr(self, '_logged_mismatch'): self._logged_mismatch = set()
             #      self._logged_mismatch.add(infoset_key)
             self.strategy_sum[infoset_key] = np.zeros(num_actions, dtype=np.float64)

        if infoset_key not in self.reach_prob_sum: self.reach_prob_sum[infoset_key] = 0.0

        # 4. Compute Current Strategy (sigma^t)
        strategy = get_rm_plus_strategy(current_regrets)

        # 5. Update Average Strategy Numerator and Denominator
        player_reach = reach_probs[player]
        if player_reach > 1e-9: # Use tolerance
             if self.config.cfr_plus_params.weighted_averaging_enabled:
                  delay = self.config.cfr_plus_params.averaging_delay; weight = float(max(0, iteration - delay))
             else: weight = 1.0
             if weight > 0:
                 self.reach_prob_sum[infoset_key] += weight * player_reach
                 self.strategy_sum[infoset_key] += weight * player_reach * strategy

        # 6. Recurse on Actions
        action_utilities = np.zeros((num_actions, self.num_players), dtype=np.float64)
        node_value = np.zeros(self.num_players, dtype=np.float64)
        use_pruning = self.config.cfr_training.pruning_enabled
        pruning_threshold = self.config.cfr_training.pruning_threshold

        for i, action in enumerate(legal_actions):
            action_prob = strategy[i]
            # logger.debug(f"Iter {iteration}, P{player}, Depth {depth}, Context {current_context.name}: Processing action {i+1}/{num_actions}: {action} (Prob: {action_prob:.4f})")

            node_positive_regret_sum = np.sum(np.maximum(0.0, current_regrets))
            # Pruning condition: positive regret for action <= threshold AND sum of positive regrets > threshold
            should_prune = (use_pruning and player_reach > 1e-9 and current_regrets[i] <= pruning_threshold and node_positive_regret_sum > pruning_threshold and iteration > 10)

            if should_prune:
                 # logger.debug(f"Iter {iteration}, P{player}, Depth {depth}: Pruning action {action}")
                 continue # Skip recursion for pruned actions

            # Skip recursion if action probability is negligible (even without pruning)
            # This helps speed up, but ensure threshold is small
            if action_prob < 1e-9:
                 # logger.debug(f"Iter {iteration}, P{player}, Depth {depth}: Skipping action {action} low prob.")
                 continue

            action_log_entry = { "player": player, "turn": game_state.get_turn_number(), "depth": depth, "context": current_context.name, "state_desc_before": str(game_state), "infoset_key": infoset_key, "action": action, "action_prob": action_prob, "reach_prob": player_reach }
            state_before_this_action_str = action_log_entry["state_desc_before"]
            state_delta: Optional[StateDelta] = None; undo_info: Optional[UndoInfo] = None; state_after_action_desc = "ERROR";

            try:
                 state_delta, undo_info = game_state.apply_action(action)
                 state_after_action_desc = str(game_state)
            except Exception as apply_err:
                 logger.error(f"Iter {iteration}, P{player}, Depth {depth}: Error applying action {action} in state {state_before_this_action_str} at infoset {infoset_key}: {apply_err}", exc_info=False)
                 action_log_entry["outcome"] = f"ERROR applying action: {apply_err}"; action_log.append(action_log_entry); undo_info = None; continue

            # Observation uses snap log *after* action potentially modifies it
            observation = self._create_observation(None, action, game_state, player, game_state.snap_results_log)

            next_agent_states = []
            agent_update_failed = False
            for agent_idx, agent_state in enumerate(agent_states):
                  cloned_agent = agent_state.clone()
                  try:
                       player_specific_obs = self._filter_observation(observation, agent_idx)
                       cloned_agent.update(player_specific_obs)
                  except Exception as agent_update_err:
                       logger.error(f"Iter {iteration}, P{player}, Depth {depth}: Error updating AgentState {agent_idx} for action {action}. Infoset: {infoset_key}. Obs: {observation}", exc_info=True); agent_update_failed = True; break
                  next_agent_states.append(cloned_agent)

            if agent_update_failed:
                action_log_entry["outcome"] = "ERROR updating agent state"; action_log.append(action_log_entry)
                if undo_info: undo_info() # Try to undo
                continue

            next_reach_probs = reach_probs.copy()
            next_reach_probs[player] *= action_prob # Update reach for the acting player

            try:
                 # logger.debug(f"Iter {iteration}, P{player}, Depth {depth}, Context {current_context.name}: Recursing ({i+1}/{num_actions}) for action: {action}")
                 recursive_utilities = self._cfr_recursive(game_state, next_agent_states, next_reach_probs, iteration, action_log, depth + 1)
                 action_utilities[i] = recursive_utilities
                 # logger.debug(f"Iter {iteration}, P{player}, Depth {depth}, Context {current_context.name}: Returned from recursion for {action}. Util: {recursive_utilities}")
            except RecursionError:
                  logger.error(f"Iter {iteration}, P{player}, Depth {depth}: Recursion depth exceeded during action: {action}. State: {game_state}", exc_info=False); action_log_entry["outcome"] = "ERROR: RecursionError"; action_log.append(action_log_entry); raise
            except Exception as recursive_err:
                  logger.error(f"Iter {iteration}, P{player}, Depth {depth}: Error in recursive call for action {action}. Infoset: {infoset_key}", exc_info=False)
                  action_log_entry["outcome"] = f"ERROR in recursion: {recursive_err}"; action_log.append(action_log_entry)
                  if undo_info: undo_info() # Try to undo
                  continue # Skip to next action

            action_log_entry["outcome_utilities"] = action_utilities[i].tolist(); action_log_entry["state_desc_after"] = state_after_action_desc; action_log.append(action_log_entry)

            if undo_info:
                 try: undo_info()
                 except Exception as undo_e: logger.exception(f"FATAL: Iter {iteration}, P{player}, Depth {depth}: Error undoing action {action}. State corrupt."); return np.zeros(self.num_players, dtype=np.float64)
            else: logger.error(f"Iter {iteration}, P{player}, Depth {depth}: Missing undo info for action {action}. State corrupt."); return np.zeros(self.num_players, dtype=np.float64)

        # --- Calculate Node Value ---
        # Node value calculation should use the computed strategy, not just average over actions taken
        node_value = np.sum(strategy[:, np.newaxis] * action_utilities, axis=0)


        # --- Update Regrets ---
        opponent_reach = reach_probs[opponent]
        if player_reach > 1e-9: # Use tolerance
            player_action_values = action_utilities[:, player]
            player_node_value = node_value[player]
            instantaneous_regret = player_action_values - player_node_value
            # The weight for regret update is the counterfactual reach probability = reach of others
            update_weight = opponent_reach
            current_regrets_before_update = np.copy(self.regret_sum[infoset_key])

            try:
                 updated_regrets = current_regrets_before_update + update_weight * instantaneous_regret
                 # Apply RM+ rule: Regrets >= 0
                 self.regret_sum[infoset_key] = np.maximum(0.0, updated_regrets)
            except Exception as e_regret:
                 logger.error(f"Iter {iteration}, P{player}, Depth {depth}, Infoset {infoset_key}, Context {current_context.name}: Error during regret update: {e_regret}", exc_info=True)

            # if logger.isEnabledFor(logging.DEBUG):
            #     logger.debug(f"Iter {iteration}, P{player}, Depth {depth}, Infoset {infoset_key}, Context {current_context.name}: NodeVal={player_node_value:.4f}, ActionUtils={player_action_values}, InstRegret={instantaneous_regret}")
            #     logger.debug(f"Iter {iteration}, P{player}, Depth {depth}, Infoset {infoset_key}, Context {current_context.name}: Updating Regrets. OppReach: {opponent_reach:.4f}, Weight: {update_weight:.4f}")
            #     logger.debug(f"Iter {iteration}, P{player}, Depth {depth}, Infoset {infoset_key}, Context {current_context.name}: Regrets Old: {current_regrets_before_update}")
            #     logger.debug(f"Iter {iteration}, P{player}, Depth {depth}, Infoset {infoset_key}, Context {current_context.name}: Regrets New: {self.regret_sum[infoset_key]}")

        # logger.debug(f"Iter {iteration}, P{player}, Depth {depth}: Returning node value: {node_value}")
        return node_value


    # --- Helper Methods (_filter_observation, _create_observation, compute_average_strategy, get_average_strategy) ---
    def _filter_observation(self, obs: AgentObservation, observer_id: int) -> AgentObservation:
         """ Filters sensitive information from observation based on observer."""
         filtered_obs = copy.copy(obs)
         if obs.drawn_card and obs.acting_player != observer_id:
              # Hide drawn card unless it was discarded (public via discard_top) or replaced by observer
              if not isinstance(obs.action, ActionDiscard) and not (isinstance(obs.action, ActionReplace) and obs.acting_player == observer_id):
                   filtered_obs.drawn_card = None
         if obs.peeked_cards:
              filtered_peek = {}
              for (p_idx, h_idx), card in obs.peeked_cards.items():
                   # Reveal peek if observer is the peeker OR observer is the one being peeked
                   if p_idx == observer_id or obs.acting_player == observer_id:
                        filtered_peek[(p_idx, h_idx)] = card
              filtered_obs.peeked_cards = filtered_peek if filtered_peek else None
         return filtered_obs

    def _create_observation(self, prev_state: Optional[CambiaGameState], action: Optional[GameAction], next_state: CambiaGameState, acting_player: int, snap_results: List[Dict]) -> AgentObservation:
         """ Creates the observation object based on state change (uses next_state). """
         discard_top = next_state.get_discard_top()
         hand_sizes = [next_state.get_player_card_count(i) for i in range(self.num_players)]
         stock_size = next_state.get_stockpile_size()
         cambia_called = next_state.cambia_caller_id is not None
         who_called = next_state.cambia_caller_id
         game_over = next_state.is_terminal()
         turn_num = next_state.get_turn_number()
         drawn_card = None
         peeked_cards_dict = None # Dictionary to hold peek results {(player_idx, hand_idx): Card}

         # Check pending state for drawn card info (relevant after Draw actions)
         if next_state.pending_action and next_state.pending_action_player == acting_player:
              pending_data = next_state.pending_action_data
              # If the pending action implies a card was just drawn (e.g., Discard/Replace choice)
              if isinstance(next_state.pending_action, ActionDiscard):
                   drawn_card = pending_data.get("drawn_card")

              # Check for peeked cards stored during King Look selection
              elif isinstance(next_state.pending_action, ActionAbilityKingSwapDecision):
                   if "own_idx" in pending_data and "opp_idx" in pending_data and "card1" in pending_data and "card2" in pending_data:
                        opp_idx = next_state.get_opponent_index(acting_player)
                        peeked_cards_dict = {
                            (acting_player, pending_data["own_idx"]): pending_data["card1"],
                            (opp_idx, pending_data["opp_idx"]): pending_data["card2"]
                        }

         # Check action itself for peek results (relevant after Peek/Look actions)
         # Important: Needs access to the *next* state's hands to get the card objects
         if isinstance(action, ActionAbilityPeekOwnSelect):
              hand = next_state.get_player_hand(acting_player)
              if hand and 0 <= action.target_hand_index < len(hand):
                  peeked_cards_dict = {(acting_player, action.target_hand_index): hand[action.target_hand_index]}
         elif isinstance(action, ActionAbilityPeekOtherSelect):
              opp_idx = next_state.get_opponent_index(acting_player)
              opp_hand = next_state.get_player_hand(opp_idx)
              if opp_hand and 0 <= action.target_opponent_hand_index < len(opp_hand):
                  peeked_cards_dict = {(opp_idx, action.target_opponent_hand_index): opp_hand[action.target_opponent_hand_index]}
         # KingLookSelect action itself doesn't reveal cards; the subsequent SwapDecision state holds them
         # So the peeked_cards_dict from pending_action check above covers King ability reveals.

         # Get the most recent snap results log from the game state
         final_snap_results = next_state.snap_results_log # This should be the log *after* the action

         obs = AgentObservation(
             acting_player=acting_player,
             action=action,
             discard_top_card=discard_top,
             player_hand_sizes=hand_sizes,
             stockpile_size=stock_size,
             drawn_card=drawn_card, # Card drawn by acting player (if applicable)
             peeked_cards=peeked_cards_dict, # Cards revealed by peeks/looks
             snap_results=final_snap_results, # Log of snap events in this turn phase
             did_cambia_get_called=cambia_called,
             who_called_cambia=who_called,
             is_game_over=game_over,
             current_turn=turn_num
         )
         return obs

    def compute_average_strategy(self) -> PolicyDict:
        """ Computes the average strategy using the CFR+ formula. """
        avg_strategy: PolicyDict = {}
        logger.info(f"Computing average strategy from {len(self.strategy_sum)} infosets...")
        if not self.strategy_sum: logger.warning("Strategy sum is empty."); return avg_strategy

        zero_reach_count, nan_count, norm_issue_count, mismatched_dim_count = 0, 0, 0, 0

        for infoset_key, s_sum in self.strategy_sum.items():
             # Ensure infoset_key is the dataclass instance
             if isinstance(infoset_key, tuple): infoset_key = InfosetKey(*infoset_key)

             r_sum = self.reach_prob_sum.get(infoset_key, 0.0)
             num_actions_in_sum = len(s_sum)
             normalized_strategy = np.array([]) # Initialize

             if r_sum > 1e-9:
                  normalized_strategy = s_sum / r_sum
                  if np.isnan(normalized_strategy).any():
                       nan_count += 1; normalized_strategy = np.ones(num_actions_in_sum) / num_actions_in_sum if num_actions_in_sum > 0 else np.array([])
                  current_sum = np.sum(normalized_strategy)
                  if not np.isclose(current_sum, 1.0, atol=1e-6) and len(normalized_strategy) > 0:
                        normalized_strategy = normalize_probabilities(normalized_strategy)
                        if not np.isclose(np.sum(normalized_strategy), 1.0, atol=1e-6): norm_issue_count += 1; logger.warning(f"Avg strategy re-norm failed: {infoset_key}")
             else:
                  # Only count as zero reach if strategy sum was non-zero (implies node was visited)
                  if np.any(s_sum != 0): zero_reach_count += 1;
                  normalized_strategy = np.ones(num_actions_in_sum) / num_actions_in_sum if num_actions_in_sum > 0 else np.array([])

             regret_array = self.regret_sum.get(infoset_key)
             # Compare average strategy dimensions against regret dimensions
             if regret_array is not None and len(regret_array) != len(normalized_strategy):
                 mismatched_dim_count += 1; num_actions_regret = len(regret_array)
                 logger.warning(f"Avg strategy dim ({len(normalized_strategy)}) mismatch with regret ({num_actions_regret}) for {infoset_key}. Defaulting avg strategy.")
                 # Use regret dimension as the source of truth for action count
                 normalized_strategy = np.ones(num_actions_regret) / num_actions_regret if num_actions_regret > 0 else np.array([])

             avg_strategy[infoset_key] = normalized_strategy

        self.average_strategy = avg_strategy
        logger.info(f"Average strategy computation complete ({len(self.average_strategy)} infosets).")
        if zero_reach_count > 0: logger.warning(f"{zero_reach_count} infosets with zero reach sum but non-zero strategy sum.")
        if nan_count > 0: logger.warning(f"{nan_count} infosets with NaN strategy.")
        if norm_issue_count > 0: logger.warning(f"{norm_issue_count} infosets with norm issues.")
        if mismatched_dim_count > 0: logger.warning(f"{mismatched_dim_count} infosets with final dimension mismatch (avg vs regret).")
        return avg_strategy

    def get_average_strategy(self) -> Optional[PolicyDict]:
         """Returns the computed average strategy."""
         if self.average_strategy is None:
              logger.warning("Average strategy requested but not computed yet. Computing now...")
              return self.compute_average_strategy()
         return self.average_strategy