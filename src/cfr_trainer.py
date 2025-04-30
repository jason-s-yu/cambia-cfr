# src/cfr_trainer.py
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Set, TypeAlias, Tuple # Added Tuple
from collections import defaultdict
import copy
import traceback

# Import necessary components from the project
from .game_engine import CambiaGameState, StateDelta, UndoInfo # Import delta types
from .constants import (
     ActionAbilityKingSwapDecision, GameAction, NUM_PLAYERS,
     ActionPassSnap, ActionSnapOwn, ActionSnapOpponent,
     ActionDrawStockpile, ActionDrawDiscard, ActionReplace, ActionDiscard,
     ActionAbilityPeekOwnSelect, ActionAbilityPeekOtherSelect, ActionAbilityKingLookSelect,
)
from .agent_state import AgentState, AgentObservation
from .utils import InfosetKey, PolicyDict, get_rm_plus_strategy, normalize_probabilities
from .config import Config
from .card import Card
from .analysis_tools import AnalysisTools

logger = logging.getLogger(__name__)

ReachProbDict: TypeAlias = Dict[InfosetKey, float]

class CFRTrainer:
    """Implements the CFR+ algorithm for training a Cambia agent via self-play."""

    def __init__(self, config: Config):
        self.config = config
        self.num_players = NUM_PLAYERS
        self.regret_sum: PolicyDict = defaultdict(lambda: np.array([], dtype=np.float64))
        self.strategy_sum: PolicyDict = defaultdict(lambda: np.array([], dtype=np.float64))
        self.reach_prob_sum: ReachProbDict = defaultdict(float)
        self.current_iteration = 0
        self.average_strategy: Optional[PolicyDict] = None
        self.analysis = AnalysisTools(config, config.logging.log_dir, config.logging.log_file_prefix)


    def load_data(self, filepath: Optional[str] = None):
        """Loads previously saved training data."""
        from .persistence import load_agent_data
        path = filepath or self.config.persistence.agent_data_save_path
        loaded = load_agent_data(path)
        if loaded:
            self.regret_sum = defaultdict(lambda: np.array([], dtype=np.float64), loaded.get('regret_sum', {}))
            self.strategy_sum = defaultdict(lambda: np.array([], dtype=np.float64), loaded.get('strategy_sum', {}))
            self.reach_prob_sum = defaultdict(float, loaded.get('reach_prob_sum', {}))
            self.current_iteration = loaded.get('iteration', 0)
            logger.info(f"Resuming training from iteration {self.current_iteration + 1}")
        else:
            logger.info("No saved data found or error loading. Starting fresh.")
            self.regret_sum = defaultdict(lambda: np.array([], dtype=np.float64))
            self.strategy_sum = defaultdict(lambda: np.array([], dtype=np.float64))
            self.reach_prob_sum = defaultdict(float)
            self.current_iteration = 0


    def save_data(self, filepath: Optional[str] = None):
        """Saves the current training data."""
        from .persistence import save_agent_data
        path = filepath or self.config.persistence.agent_data_save_path
        data_to_save = {
            'regret_sum': dict(self.regret_sum),
            'strategy_sum': dict(self.strategy_sum),
            'reach_prob_sum': dict(self.reach_prob_sum),
            'iteration': self.current_iteration
        }
        save_agent_data(data_to_save, path)


    def train(self, num_iterations: Optional[int] = None):
        """Runs the CFR+ training loop for the specified number of iterations."""
        total_iterations_to_run = num_iterations or self.config.cfr_training.num_iterations
        start_iteration = self.current_iteration
        end_iteration = start_iteration + total_iterations_to_run

        if total_iterations_to_run <= 0:
             logger.warning("Number of iterations to run must be positive.")
             return

        logger.info(f"Starting CFR+ training from iteration {start_iteration + 1} up to {end_iteration}...")
        loop_start_time = time.time()

        for t in range(start_iteration, end_iteration):
            iter_start_time = time.time()
            self.current_iteration = t + 1

            initial_game_state_for_log = None
            game_state = None
            try:
                game_state = CambiaGameState(house_rules=self.config.cambia_rules)
                # Log initial state *if possible* - requires temporary clone or separate capture
                try:
                     initial_game_state_for_log = game_state._internal_clone() # Use internal clone for logging only
                except Exception as log_clone_err:
                     logger.warning(f"Could not clone initial state for logging: {log_clone_err}")
            except Exception as e:
                logger.exception(f"Error initializing game state on iteration {self.current_iteration}: {e}")
                continue

            reach_probs = np.ones(self.num_players, dtype=np.float64)
            initial_agent_states = []
            action_log_for_game: List[Dict] = []

            if not game_state.is_terminal():
                 initial_obs = self._create_observation(None, None, game_state, -1, [])
                 for i in range(self.num_players):
                      try:
                           agent = AgentState(
                               player_id=i,
                               opponent_id=game_state.get_opponent_index(i),
                               memory_level=self.config.agent_params.memory_level,
                               time_decay_turns=self.config.agent_params.time_decay_turns,
                               initial_hand_size=game_state.get_player_card_count(i),
                               config=self.config
                           )
                           # Use logged initial state for hands if available
                           initial_hand = initial_game_state_for_log.get_player_hand(i) if initial_game_state_for_log else game_state.get_player_hand(i)
                           peek_indices = initial_game_state_for_log.players[i].initial_peek_indices if initial_game_state_for_log else game_state.players[i].initial_peek_indices
                           agent.initialize(initial_obs, initial_hand, peek_indices)
                           initial_agent_states.append(agent)
                      except Exception as e:
                           logger.exception(f"Error initializing AgentState {i} on iteration {self.current_iteration}: {e}")
                           initial_agent_states = []
                           break
            else:
                 logger.error(f"Game is terminal immediately after initialization on iteration {self.current_iteration}. Skipping.")
                 continue

            if not initial_agent_states:
                 continue

            final_utilities = None
            try:
                 final_utilities = self._cfr_recursive(game_state, initial_agent_states, reach_probs, self.current_iteration, action_log_for_game)

                 # Log game history
                 if initial_game_state_for_log:
                    initial_hands_log = [p.hand for p in initial_game_state_for_log.players]
                    game_details = self.analysis.format_game_details_for_log(
                         game_state=game_state, # Use the final state after recursion
                         iteration=self.current_iteration,
                         initial_hands=initial_hands_log,
                         action_sequence=action_log_for_game
                     )
                    self.analysis.log_game_history(game_details)
                 else:
                    logger.warning("Could not log game history: Initial game state was not available.")

            except RecursionError:
                logger.error(f"Recursion depth exceeded on iteration {self.current_iteration}! Saving progress and stopping.")
                logger.error("Traceback:\n%s", traceback.format_exc())
                self.save_data()
                raise
            except Exception as e:
                 logger.exception(f"Error during CFR recursion on iteration {self.current_iteration}: {e}")
                 continue

            iter_time = time.time() - iter_start_time
            if self.current_iteration % 100 == 0 or self.current_iteration == end_iteration:
                 total_elapsed = time.time() - loop_start_time
                 completed_iters_in_run = self.current_iteration - start_iteration
                 iters_per_sec = completed_iters_in_run / total_elapsed if total_elapsed > 0 else 0
                 utility_str = f"Utils: {final_utilities}" if final_utilities is not None else "Utils: N/A"
                 logger.info(f"Iter {self.current_iteration}/{end_iteration} | Last: {iter_time:.3f}s | "
                             f"Infosets: {len(self.regret_sum):,} | Avg Speed: {iters_per_sec:.2f} it/s | {utility_str}")


            if self.current_iteration % self.config.cfr_training.save_interval == 0:
                self.save_data()

        end_time = time.time()
        total_completed = self.current_iteration - start_iteration
        logger.info(f"Training finished {total_completed} iterations.")
        logger.info(f"Total training time: {end_time - loop_start_time:.2f} seconds.")
        self.save_data()
        self.compute_average_strategy()
        logger.info("Final average strategy computed.")


    def _cfr_recursive(self, game_state: CambiaGameState, agent_states: List[AgentState], reach_probs: np.ndarray, iteration: int, action_log: List[Dict]) -> np.ndarray:
        """
        Recursive CFR+ function. Operates on the *same* game_state object,
        applying and undoing actions. Clones agent_states.
        """

        # --- Base Case: Terminal Node ---
        if game_state.is_terminal():
            return np.array([game_state.get_utility(i) for i in range(self.num_players)], dtype=np.float64)

        # --- Identify Acting Player ---
        player = game_state.get_acting_player()
        if player == -1:
             logger.error(f"Could not determine acting player in non-terminal state. State: {game_state}")
             return np.zeros(self.num_players, dtype=np.float64)

        current_agent_state = agent_states[player]
        opponent = 1 - player

        # 1. Get Infoset Key
        try:
             if not hasattr(current_agent_state, 'own_hand') or not hasattr(current_agent_state, 'opponent_belief'):
                  logger.error(f"AgentState for P{player} appears uninitialized before get_infoset_key(). State: {current_agent_state}")
                  return np.zeros(self.num_players, dtype=np.float64)
             infoset_key = current_agent_state.get_infoset_key()
        except Exception as e:
             logger.error(f"Error getting infoset key for P{player}. AgentState: {current_agent_state}. GameState: {game_state}", exc_info=True)
             return np.zeros(self.num_players, dtype=np.float64)

        # 2. Get Legal Actions
        try:
            legal_actions_set = game_state.get_legal_actions()
            legal_actions = sorted(list(legal_actions_set), key=repr)
        except Exception as e:
            logger.error(f"Error getting/sorting legal actions for P{player} at state {game_state}. InfosetKey: {infoset_key}", exc_info=True)
            return np.zeros(self.num_players, dtype=np.float64)

        num_actions = len(legal_actions)
        if num_actions == 0:
             if not game_state.is_terminal():
                  logger.warning(f"No legal actions found for P{player} at infoset {infoset_key} in *non-terminal* state {game_state}. Forcing game end check.")
                  game_state._check_game_end()
                  if game_state.is_terminal():
                       logger.info(f"State confirmed terminal after no-action check for P{player}.")
                       return np.array([game_state.get_utility(i) for i in range(self.num_players)], dtype=np.float64)
                  else:
                       logger.error(f"State still non-terminal after no-action check for P{player}. This indicates a potential engine bug.")
                       return np.zeros(self.num_players, dtype=np.float64)
             else:
                  return np.array([game_state.get_utility(i) for i in range(self.num_players)], dtype=np.float64)


        # 3. Initialize Infoset Data
        current_regrets = self.regret_sum.get(infoset_key)
        if current_regrets is None or current_regrets.shape[0] != num_actions:
            self.regret_sum[infoset_key] = np.zeros(num_actions, dtype=np.float64)
            current_regrets = self.regret_sum[infoset_key]

        current_strategy_sum = self.strategy_sum.get(infoset_key)
        if current_strategy_sum is None or current_strategy_sum.shape[0] != num_actions:
            self.strategy_sum[infoset_key] = np.zeros(num_actions, dtype=np.float64)

        if infoset_key not in self.reach_prob_sum:
            self.reach_prob_sum[infoset_key] = 0.0


        # 4. Compute Current Strategy (sigma^t)
        strategy = get_rm_plus_strategy(current_regrets)

        # 5. Update Average Strategy Numerator and Denominator
        player_reach = reach_probs[player]
        if player_reach > 0:
             if self.config.cfr_plus_params.weighted_averaging_enabled:
                  delay = self.config.cfr_plus_params.averaging_delay
                  weight = float(max(0, iteration - delay))
             else:
                  weight = 1.0

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

            should_prune = (use_pruning and
                           player_reach > 0 and
                           current_regrets[i] <= pruning_threshold and
                           iteration > 1)

            if should_prune:
                 action_utilities[i] = np.zeros(self.num_players)
                 node_value += action_prob * action_utilities[i]
                 continue

            if action_prob < 1e-9: continue

            action_log_entry = {
                "player": player, "turn": game_state.get_turn_number(),
                "state_desc_before": str(game_state), "infoset_key": infoset_key,
                "action": action, "action_prob": action_prob, "reach_prob": player_reach
            }


            # --- State Transition (Apply & Get Undo) ---
            state_delta: Optional[StateDelta] = None
            undo_info: Optional[UndoInfo] = None
            state_after_action_desc = "ERROR" # Default description
            try:
                 # --- Apply Action (modifies game_state) ---
                 # Store state *before* applying for observation creation
                 state_before_action_str = str(game_state) # Capture before modification
                 snap_log_before_action = list(game_state.snap_results_log) # Copy log before action

                 state_delta, undo_info = game_state.apply_action(action)
                 state_after_action_desc = str(game_state) # Capture after modification

            except Exception as e:
                 logger.error(f"Error applying action {action} in state {state_before_action_str} at infoset {infoset_key} (Iter {iteration}): {e}", exc_info=True)
                 action_utilities[i] = np.zeros(self.num_players)
                 node_value += action_prob * action_utilities[i]
                 action_log_entry["outcome"] = "ERROR applying action"
                 action_log_entry["state_desc_after"] = state_after_action_desc # Log state after error if possible
                 action_log.append(action_log_entry)
                 # Attempt to undo if possible? Risky if state is corrupt. Skip undo here.
                 undo_info = None # Prevent undo attempt
                 continue

            # --- Observation Creation (using state *after* action) ---
            # Need the snap log *generated by* the action, which should be in the modified game_state
            current_snap_log = game_state.snap_results_log
            observation = self._create_observation(
                 prev_state=None, # No longer needed if using deltas directly? Pass state before?
                 action=action,
                 next_state=game_state, # Pass current (modified) state
                 acting_player=player,
                 snap_results=current_snap_log # Log from *after* action application
            )

            # --- Agent Belief Update (operates on clones) ---
            next_agent_states = []
            agent_update_failed = False
            for agent_idx, agent_state in enumerate(agent_states):
                  cloned_agent = agent_state.clone()
                  try:
                       player_specific_obs = self._filter_observation(observation, agent_idx)
                       cloned_agent.update(player_specific_obs)
                  except Exception as e:
                       logger.error(f"Error updating AgentState {agent_idx} for P{player} acting with {action}. Infoset: {infoset_key}. Obs: {observation}", exc_info=True)
                       agent_update_failed = True
                       break
                  next_agent_states.append(cloned_agent)

            if agent_update_failed:
                action_utilities[i] = np.zeros(self.num_players)
                node_value += action_prob * action_utilities[i]
                action_log_entry["outcome"] = "ERROR updating agent state"
                action_log_entry["state_desc_after"] = state_after_action_desc
                action_log.append(action_log_entry)
                # Attempt to undo the game state change
                if undo_info:
                     try: undo_info()
                     except Exception as undo_e: logger.error(f"Error undoing action after agent update failure: {undo_e}")
                continue

            # --- Reach Probability Update ---
            next_reach_probs = reach_probs.copy()
            next_reach_probs[player] *= action_prob

            # --- Recursive Call (with modified game_state) ---
            try:
                 # Pass the modified game_state down
                 action_utilities[i] = self._cfr_recursive(
                     game_state, next_agent_states, next_reach_probs, iteration, action_log
                 )
            except Exception as recursive_err:
                  logger.error(f"Error during recursive call for action {action} from state {state_before_action_str}. Infoset: {infoset_key}", exc_info=True)
                  action_utilities[i] = np.zeros(self.num_players) # Assign zero utility on error
                  # Log the error in the action sequence
                  action_log_entry["outcome"] = f"ERROR in recursion: {recursive_err}"
                  action_log_entry["outcome_utilities"] = action_utilities[i].tolist()
                  action_log_entry["state_desc_after"] = state_after_action_desc
                  action_log.append(action_log_entry)
                  # Attempt to undo state before continuing
                  if undo_info:
                      try: undo_info()
                      except Exception as undo_e: logger.error(f"Error undoing action after recursion error: {undo_e}")
                  continue # Move to the next action

            # --- Update Node Value ---
            node_value += action_prob * action_utilities[i]

            # --- Finalize Action Log Entry ---
            action_log_entry["outcome_utilities"] = action_utilities[i].tolist()
            action_log_entry["state_desc_after"] = state_after_action_desc # Already captured
            action_log.append(action_log_entry)

            # --- Undo Action ---
            # Crucial step: Restore game_state for the next action sibling
            if undo_info:
                 try:
                      undo_info()
                      # Optional: Add verification step here to check if undo was successful
                      # if str(game_state) != state_before_action_str: # Basic check
                      #      logger.error(f"Undo failed to restore state correctly after action {action}. Before: '{state_before_action_str}', After Undo: '{str(game_state)}'")
                 except Exception as undo_e:
                      # If undo fails, the state is corrupt, cannot reliably continue this branch
                      logger.exception(f"FATAL: Error undoing action {action} from state {state_before_action_str}. State may be corrupt. Stopping branch.")
                      # Return a value indicating error or raise? Return 0 utility for now.
                      return np.zeros(self.num_players, dtype=np.float64)
            else:
                 # This should not happen if apply_action always returns undo info
                 logger.error(f"Missing undo information for action {action}. State may be corrupt.")
                 return np.zeros(self.num_players, dtype=np.float64)


        # 7. Calculate and Update Regrets
        opponent_reach = reach_probs[opponent]
        if player_reach > 0:
            player_action_values = action_utilities[:, player]
            player_node_value = node_value[player]
            instantaneous_regret = player_action_values - player_node_value
            update_weight = opponent_reach if opponent_reach > 0 else 1.0

            current_regrets = self.regret_sum[infoset_key] # Fetch again
            self.regret_sum[infoset_key] = np.maximum(0.0, current_regrets + update_weight * instantaneous_regret)


        # 8. Return Node Value
        return node_value


    def _filter_observation(self, obs: AgentObservation, observer_id: int) -> AgentObservation:
         """ Filters sensitive information from observation based on observer."""
         filtered_obs = copy.copy(obs)

         if obs.drawn_card and obs.acting_player != observer_id:
              is_replace = isinstance(obs.action, ActionReplace)
              if is_replace:
                   filtered_obs.drawn_card = None

         if obs.peeked_cards and obs.acting_player != observer_id:
              is_king_look = isinstance(obs.action, ActionAbilityKingLookSelect)
              is_peek_other = isinstance(obs.action, ActionAbilityPeekOtherSelect)

              if is_king_look:
                  filtered_peek = {}
                  for (p_idx, h_idx), card in obs.peeked_cards.items():
                      if p_idx == observer_id:
                           filtered_peek[(p_idx, h_idx)] = card
                           break
                  filtered_obs.peeked_cards = filtered_peek if filtered_peek else None
              elif is_peek_other:
                   filtered_obs.peeked_cards = None

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
         peeked_cards_dict = None

         # Info extracted from the state *after* the action was applied
         if next_state.pending_action and next_state.pending_action_player == acting_player:
              # Check if the pending action is the result of a draw
              if isinstance(next_state.pending_action, ActionDiscard):
                   drawn_card = next_state.pending_action_data.get("drawn_card")
              # Check if the pending action is the result of a King Look
              elif isinstance(next_state.pending_action, ActionAbilityKingSwapDecision):
                  data = next_state.pending_action_data
                  if "own_idx" in data and "opp_idx" in data and "card1" in data and "card2" in data:
                      opp_idx = next_state.get_opponent_index(acting_player)
                      peeked_cards_dict = {
                           (acting_player, data["own_idx"]): data["card1"],
                           (opp_idx, data["opp_idx"]): data["card2"]
                      }

         # Check the action itself for completed peeks (that don't lead to pending state)
         if isinstance(action, ActionAbilityPeekOwnSelect):
              if 0 <= action.target_hand_index < next_state.get_player_card_count(acting_player):
                   peeked_card = next_state.get_player_hand(acting_player)[action.target_hand_index]
                   peeked_cards_dict = {(acting_player, action.target_hand_index): peeked_card}
         elif isinstance(action, ActionAbilityPeekOtherSelect):
              opp_idx = next_state.get_opponent_index(acting_player)
              if 0 <= action.target_opponent_hand_index < next_state.get_player_card_count(opp_idx):
                   peeked_card = next_state.get_player_hand(opp_idx)[action.target_opponent_hand_index]
                   peeked_cards_dict = {(opp_idx, action.target_opponent_hand_index): peeked_card}


         obs = AgentObservation(
             acting_player=acting_player, action=action,
             discard_top_card=discard_top, player_hand_sizes=hand_sizes, stockpile_size=stock_size,
             drawn_card=drawn_card, peeked_cards=peeked_cards_dict, snap_results=snap_results,
             did_cambia_get_called=cambia_called, who_called_cambia=who_called,
             is_game_over=game_over, current_turn=turn_num
         )
         return obs


    def compute_average_strategy(self) -> PolicyDict:
        """ Computes the average strategy using the CFR+ formula. """
        avg_strategy: PolicyDict = {}
        logger.info(f"Computing average strategy from {len(self.strategy_sum)} infosets...")

        if not self.strategy_sum: logger.warning("Strategy sum is empty."); return avg_strategy

        zero_reach_count = 0
        nan_count = 0
        norm_issue_count = 0

        for infoset_key, s_sum in self.strategy_sum.items():
             r_sum = self.reach_prob_sum.get(infoset_key, 0.0)

             if r_sum > 1e-9:
                  normalized_strategy = s_sum / r_sum
                  if np.isnan(normalized_strategy).any():
                       logger.warning(f"NaN detected in avg strategy for infoset {infoset_key}. Num: {s_sum}, Denom: {r_sum}. Defaulting to uniform.")
                       nan_count += 1
                       num_actions = len(s_sum)
                       normalized_strategy = np.ones(num_actions) / num_actions if num_actions > 0 else np.array([])
                  current_sum = np.sum(normalized_strategy)
                  if not np.isclose(current_sum, 1.0, atol=1e-6) and len(normalized_strategy) > 0:
                        # logger.debug(f"Avg strategy requires re-normalization for infoset {infoset_key}. Sum: {current_sum}. Strategy before: {normalized_strategy}")
                        normalized_strategy = normalize_probabilities(normalized_strategy)
                        if not np.isclose(np.sum(normalized_strategy), 1.0, atol=1e-6):
                             logger.warning(f"Avg strategy re-normalization failed for infoset {infoset_key}. Sum: {np.sum(normalized_strategy)}. Num: {s_sum}, Denom: {r_sum}. Final Strategy: {normalized_strategy}")
                             norm_issue_count += 1
             else:
                  zero_reach_count += 1
                  num_actions = len(s_sum)
                  normalized_strategy = np.ones(num_actions) / num_actions if num_actions > 0 else np.array([])

             avg_strategy[infoset_key] = normalized_strategy


        self.average_strategy = avg_strategy
        logger.info("Average strategy computation complete.")
        if zero_reach_count > 0: logger.warning(f"Found {zero_reach_count} infosets with zero/negligible reach sum, defaulted to uniform.")
        if nan_count > 0: logger.warning(f"Found {nan_count} infosets resulting in NaN strategy, defaulted to uniform.")
        if norm_issue_count > 0: logger.warning(f"Found {norm_issue_count} infosets with normalization issues after division.")
        logger.info(f"Computed average strategy for {len(self.average_strategy)} infosets.")
        return avg_strategy

    def get_average_strategy(self) -> Optional[PolicyDict]:
         """Returns the computed average strategy."""
         if self.average_strategy is None:
              logger.warning("Average strategy requested but not computed yet. Computing now...")
              return self.compute_average_strategy()
         return self.average_strategy