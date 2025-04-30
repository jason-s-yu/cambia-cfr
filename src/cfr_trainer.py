# src/cfr_trainer.py
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Set, TypeAlias
from collections import defaultdict
import copy
import traceback # For detailed error logging

# Import necessary components from the project
from .game_engine import CambiaGameState # Need full state for simulation
from .constants import (
     ActionAbilityKingSwapDecision, GameAction, NUM_PLAYERS,
     ActionPassSnap, ActionSnapOwn, ActionSnapOpponent,
     # Import other actions if needed for type checking or observation creation
     ActionDrawStockpile, ActionDrawDiscard, ActionReplace, ActionDiscard,
     ActionAbilityPeekOwnSelect, ActionAbilityPeekOtherSelect, ActionAbilityKingLookSelect,
)
from .agent_state import AgentState, AgentObservation # Agent's belief and observation structure
from .utils import InfosetKey, PolicyDict, get_rm_plus_strategy, normalize_probabilities # Key type, Policy dict type, RM+ util
from .config import Config # Configuration dataclass
from .card import Card # For type hinting if needed
from .analysis_tools import AnalysisTools # Import analysis tools

logger = logging.getLogger(__name__)

# Type alias for the reach probability sum dictionary
ReachProbDict: TypeAlias = Dict[InfosetKey, float]

class CFRTrainer:
    """Implements the CFR+ algorithm for training a Cambia agent via self-play."""

    def __init__(self, config: Config):
        self.config = config
        self.num_players = NUM_PLAYERS # Should match game engine
        # Core data structures: maps InfosetKey -> value
        self.regret_sum: PolicyDict = defaultdict(lambda: np.array([], dtype=np.float64))
        self.strategy_sum: PolicyDict = defaultdict(lambda: np.array([], dtype=np.float64))
        self.reach_prob_sum: ReachProbDict = defaultdict(float) # For CFR+ denominator
        self.current_iteration = 0
        # Store the average strategy computed periodically or at the end
        self.average_strategy: Optional[PolicyDict] = None
        # Analysis tools instance (pass log dir/prefix from config)
        self.analysis = AnalysisTools(config, config.logging.log_dir, config.logging.log_file_prefix)


    def load_data(self, filepath: Optional[str] = None):
        """Loads previously saved training data."""
        # Import lazily or ensure no circular dependency at module level
        from .persistence import load_agent_data
        path = filepath or self.config.persistence.agent_data_save_path
        loaded = load_agent_data(path)
        if loaded:
            # Use loaded data, ensuring defaultdict behavior if keys are missing
            self.regret_sum = defaultdict(lambda: np.array([], dtype=np.float64), loaded.get('regret_sum', {}))
            self.strategy_sum = defaultdict(lambda: np.array([], dtype=np.float64), loaded.get('strategy_sum', {}))
            self.reach_prob_sum = defaultdict(float, loaded.get('reach_prob_sum', {})) # Load reach prob sum
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
        # Convert defaultdicts to regular dicts for saving via joblib/pickle
        data_to_save = {
            'regret_sum': dict(self.regret_sum),
            'strategy_sum': dict(self.strategy_sum),
            'reach_prob_sum': dict(self.reach_prob_sum), # Save reach prob sum
            'iteration': self.current_iteration # Save the *last completed* iteration number
        }
        save_agent_data(data_to_save, path)


    def train(self, num_iterations: Optional[int] = None):
        """Runs the CFR+ training loop for the specified number of iterations."""
        total_iterations_to_run = num_iterations or self.config.cfr_training.num_iterations
        start_iteration = self.current_iteration # The last *completed* iteration
        end_iteration = start_iteration + total_iterations_to_run

        if total_iterations_to_run <= 0:
             logger.warning("Number of iterations to run must be positive.")
             return

        logger.info(f"Starting CFR+ training from iteration {start_iteration + 1} up to {end_iteration}...")
        loop_start_time = time.time()

        for t in range(start_iteration, end_iteration):
            iter_start_time = time.time()
            self.current_iteration = t + 1 # Current iteration number (1-based)

            # Initialize game state for this simulation run
            initial_game_state = None # For logging initial hands
            game_state = None
            try:
                game_state = CambiaGameState(house_rules=self.config.cambia_rules)
                # Clone initial state immediately to preserve initial hands for logging
                initial_game_state = game_state.clone()
            except Exception as e:
                logger.exception(f"Error initializing game state on iteration {self.current_iteration}: {e}")
                continue # Skip iteration if game setup fails

            # Initial reach probabilities are 1 for all players at the root
            reach_probs = np.ones(self.num_players, dtype=np.float64)

            # Initialize agent states (one per player)
            initial_agent_states = []
            action_log_for_game: List[Dict] = [] # List to store action details for this game

            if not game_state.is_terminal():
                 initial_obs = self._create_observation(None, None, game_state, -1, []) # Initial state obs
                 for i in range(self.num_players):
                      try:
                           agent = AgentState(
                               player_id=i,
                               opponent_id=game_state.get_opponent_index(i),
                               memory_level=self.config.agent_params.memory_level,
                               time_decay_turns=self.config.agent_params.time_decay_turns,
                               initial_hand_size=game_state.get_player_card_count(i),
                               config=self.config # Pass config to agent state
                           )
                           initial_hand = game_state.get_player_hand(i) # Use getter
                           peek_indices = game_state.players[i].initial_peek_indices
                           agent.initialize(initial_obs, initial_hand, peek_indices)
                           initial_agent_states.append(agent)
                      except Exception as e:
                           logger.exception(f"Error initializing AgentState {i} on iteration {self.current_iteration}: {e}")
                           # If one agent fails, maybe skip the iteration?
                           initial_agent_states = [] # Clear list to indicate failure
                           break
            else:
                 logger.error(f"Game is terminal immediately after initialization on iteration {self.current_iteration}. Skipping.")
                 continue

            if not initial_agent_states: # Skip if agent setup failed
                 continue

            # Start recursive traversal from the root
            final_utilities = None
            try:
                 # Pass iteration number t+1 for CFR+ weighting calculations
                 # Also pass the action log list to be populated during recursion
                 final_utilities = self._cfr_recursive(game_state, initial_agent_states, reach_probs, self.current_iteration, action_log_for_game)

                 # Log game history after the recursive call completes
                 if initial_game_state: # Check if initial state was captured
                    initial_hands_log = [p.hand for p in initial_game_state.players] # Get hands from initial state
                    game_details = self.analysis.format_game_details_for_log(
                         game_state=game_state,
                         iteration=self.current_iteration,
                         initial_hands=initial_hands_log,
                         action_sequence=action_log_for_game # Pass the populated log
                     )
                    self.analysis.log_game_history(game_details)
                 else:
                    logger.warning("Could not log game history: Initial game state was not available.")


            except RecursionError:
                logger.error(f"Recursion depth exceeded on iteration {self.current_iteration}! Saving progress and stopping.")
                logger.error("Traceback:\n%s", traceback.format_exc())
                self.save_data() # Save before exiting due to recursion error
                raise # Re-raise to stop the training loop
            except Exception as e:
                 logger.exception(f"Error during CFR recursion on iteration {self.current_iteration}: {e}")
                 # Consider stopping or just skipping iteration? Skip for now.
                 continue # Skip to next iteration

            # Log progress and save periodically
            iter_time = time.time() - iter_start_time
            if self.current_iteration % 100 == 0 or self.current_iteration == end_iteration:
                 total_elapsed = time.time() - loop_start_time
                 # Calculate speed based on completed iterations since start of this run
                 completed_iters_in_run = self.current_iteration - start_iteration
                 iters_per_sec = completed_iters_in_run / total_elapsed if total_elapsed > 0 else 0
                 # Add utility info if available
                 utility_str = f"Utils: {final_utilities}" if final_utilities is not None else "Utils: N/A"
                 logger.info(f"Iter {self.current_iteration}/{end_iteration} | Last: {iter_time:.3f}s | "
                             f"Infosets: {len(self.regret_sum):,} | Avg Speed: {iters_per_sec:.2f} it/s | {utility_str}")


            if self.current_iteration % self.config.cfr_training.save_interval == 0:
                self.save_data()

        # --- Training Complete ---
        end_time = time.time()
        total_completed = self.current_iteration - start_iteration
        logger.info(f"Training finished {total_completed} iterations.")
        logger.info(f"Total training time: {end_time - loop_start_time:.2f} seconds.")
        self.save_data()
        self.compute_average_strategy()
        logger.info("Final average strategy computed.")


    def _cfr_recursive(self, game_state: CambiaGameState, agent_states: List[AgentState], reach_probs: np.ndarray, iteration: int, action_log: List[Dict]) -> np.ndarray:
        """
        Recursive CFR+ function. Operates on copies of states.
        Args:
            game_state: The current true state of the game (will be cloned).
            agent_states: List of current subjective agent states (will be cloned).
            reach_probs: Numpy array of reach probabilities for [player0, player1].
            iteration: The current training iteration number (1-based).
            action_log: A list to append action details to for game history logging.

        Returns:
            Numpy array of expected node values (utilities) for [player0, player1].
        """

        # --- Base Case: Terminal Node ---
        if game_state.is_terminal():
            return np.array([game_state.get_utility(i) for i in range(self.num_players)], dtype=np.float64)

        # --- Identify Acting Player ---
        player = game_state.get_acting_player()
        if player == -1:
             logger.error(f"Could not determine acting player in non-terminal state. State: {game_state}")
             # Return 0 utility to avoid crashing, but log the error
             return np.zeros(self.num_players, dtype=np.float64)

        current_agent_state = agent_states[player]
        opponent = 1 - player

        # 1. Get Infoset Key from the current player's perspective
        try:
             # Ensure agent state is properly initialized before getting key
             if not hasattr(current_agent_state, 'own_hand') or not hasattr(current_agent_state, 'opponent_belief'):
                  logger.error(f"AgentState for P{player} appears uninitialized before get_infoset_key(). State: {current_agent_state}")
                  # Attempt recovery or return zero utility
                  return np.zeros(self.num_players, dtype=np.float64)
             infoset_key = current_agent_state.get_infoset_key()
        except Exception as e:
             logger.error(f"Error getting infoset key for P{player}. AgentState: {current_agent_state}. GameState: {game_state}", exc_info=True)
             return np.zeros(self.num_players, dtype=np.float64)

        # 2. Get Legal Actions
        try:
            legal_actions_set = game_state.get_legal_actions()
            # Sort actions by their string representation for consistent ordering
            legal_actions = sorted(list(legal_actions_set), key=repr)
        except Exception as e:
            logger.error(f"Error getting/sorting legal actions for P{player} at state {game_state}. InfosetKey: {infoset_key}", exc_info=True)
            return np.zeros(self.num_players, dtype=np.float64)

        num_actions = len(legal_actions)
        if num_actions == 0:
             # It's possible to have no actions if game ends abruptly (e.g., deck empty on required draw)
             # Check if game *should* be terminal here
             if not game_state.is_terminal():
                  logger.warning(f"No legal actions found for P{player} at infoset {infoset_key} in *non-terminal* state {game_state}. Forcing game end check.")
                  # Force a game end check and recalculate terminal state if needed
                  game_state._check_game_end()
                  if game_state.is_terminal():
                       logger.info(f"State confirmed terminal after no-action check for P{player}.")
                       return np.array([game_state.get_utility(i) for i in range(self.num_players)], dtype=np.float64)
                  else:
                       logger.error(f"State still non-terminal after no-action check for P{player}. This indicates a potential engine bug.")
                       return np.zeros(self.num_players, dtype=np.float64) # Return 0 to avoid crash
             else:
                  # Game is already terminal, return utilities (should have been caught by base case)
                  return np.array([game_state.get_utility(i) for i in range(self.num_players)], dtype=np.float64)


        # 3. Initialize Infoset Data if New or Size Mismatch
        # Access using .get first to handle creation cleanly
        current_regrets = self.regret_sum.get(infoset_key)
        if current_regrets is None or current_regrets.shape[0] != num_actions:
            # Initialize with zeros of the correct size
            self.regret_sum[infoset_key] = np.zeros(num_actions, dtype=np.float64)
            current_regrets = self.regret_sum[infoset_key] # Use the newly created array

        current_strategy_sum = self.strategy_sum.get(infoset_key)
        if current_strategy_sum is None or current_strategy_sum.shape[0] != num_actions:
            self.strategy_sum[infoset_key] = np.zeros(num_actions, dtype=np.float64)
            # We don't need to re-assign current_strategy_sum here, just ensure it exists

        if infoset_key not in self.reach_prob_sum:
            self.reach_prob_sum[infoset_key] = 0.0


        # 4. Compute Current Strategy (sigma^t) using RM+
        strategy = get_rm_plus_strategy(current_regrets)

        # 5. Update Average Strategy Numerator and Denominator (CFR+)
        player_reach = reach_probs[player]
        if player_reach > 0:
             # Calculate iteration weight (w_t) using configured delay
             if self.config.cfr_plus_params.weighted_averaging_enabled:
                  delay = self.config.cfr_plus_params.averaging_delay
                  # Iteration is 1-based
                  weight = float(max(0, iteration - delay))
             else:
                  weight = 1.0 # Uniform weighting if disabled

             if weight > 0: # Only update sums if weight is positive
                 self.reach_prob_sum[infoset_key] += weight * player_reach
                 self.strategy_sum[infoset_key] += weight * player_reach * strategy


        # 6. Recurse on Actions
        action_utilities = np.zeros((num_actions, self.num_players), dtype=np.float64)
        node_value = np.zeros(self.num_players, dtype=np.float64)

        # --- Regret-Based Pruning Check (Optional) ---
        use_pruning = self.config.cfr_training.pruning_enabled
        pruning_threshold = self.config.cfr_training.pruning_threshold

        for i, action in enumerate(legal_actions):
            action_prob = strategy[i]

            # --- Pruning ---
            # Prune if enabled, regret is <= threshold, and iteration > 1 (to avoid pruning initial uniform)
            # Also added check for player_reach > 0, no point pruning if node isn't reachable
            should_prune = (use_pruning and
                           player_reach > 0 and
                           current_regrets[i] <= pruning_threshold and
                           iteration > 1)

            if should_prune:
                 # Skip recursion, but still calculate node value contribution using zero utility proxy.
                 # This simplifies implementation but might slightly underestimate node value temporarily.
                 action_utilities[i] = np.zeros(self.num_players) # Use 0 utility for pruned branch
                 node_value += action_prob * action_utilities[i]
                 # logger.debug(f"Pruning action {i} ({action}) at infoset {infoset_key} (Iter:{iteration}, Regret: {current_regrets[i]:.2e}, Reach:{player_reach:.2e})")
                 continue # Skip the rest of the loop for this pruned action

            # If not pruned, proceed with recursion
            # Skip recursion if probability is negligible (avoids potential float precision issues)
            if action_prob < 1e-9: continue

            # --- Action Logging ---
            # Log action *before* applying it to capture state before action
            action_log_entry = {
                "player": player,
                "turn": game_state.get_turn_number(),
                "state_desc_before": str(game_state), # Capture basic string rep
                "infoset_key": infoset_key, # Capture infoset key
                "action": action, # Store action object (will be serialized later)
                "action_prob": action_prob, # Store action probability from strategy
                "reach_prob": player_reach # Store player reach probability
            }


            # --- State Transition ---
            # Wrap state cloning and action application in try-except
            try:
                 # Use the game_state's clone method which has fallback logic
                 next_game_state_candidate = game_state.clone()
                 next_game_state = next_game_state_candidate.apply_action(action)
            except Exception as e:
                 logger.error(f"Error applying action {action} in state {game_state} at infoset {infoset_key} (Iter {iteration}): {e}", exc_info=True)
                 action_utilities[i] = np.zeros(self.num_players) # Assign 0 utility on error
                 node_value += action_prob * action_utilities[i] # Update node value with 0 utility
                 # Log the failed action attempt
                 action_log_entry["outcome"] = "ERROR applying action"
                 action_log.append(action_log_entry)
                 continue # Skip to next action

            # --- Observation Creation ---
            # Pass the snap log *from the state before this action potentially cleared it*
            # The log is relevant to the *result* of the action just taken.
            # Let's get it from next_state after apply_action which should hold the relevant log.
            current_snap_log = next_game_state.snap_results_log
            observation = self._create_observation(
                 prev_state=game_state, action=action, next_state=next_game_state,
                 acting_player=player, snap_results=current_snap_log
            )

            # --- Agent Belief Update ---
            next_agent_states = []
            agent_update_failed = False
            for agent_idx, agent_state in enumerate(agent_states):
                  cloned_agent = agent_state.clone()
                  try:
                       player_specific_obs = self._filter_observation(observation, agent_idx)
                       cloned_agent.update(player_specific_obs)
                  except Exception as e:
                       logger.error(f"Error updating AgentState {agent_idx} for P{player} acting with {action}. Infoset: {infoset_key}. Obs: {observation}", exc_info=True)
                       agent_update_failed = True # Mark failure
                       break # Stop trying to update agents for this branch
                  next_agent_states.append(cloned_agent)

            # If agent update failed, assign 0 utility and skip recursion
            if agent_update_failed:
                action_utilities[i] = np.zeros(self.num_players)
                node_value += action_prob * action_utilities[i]
                # Log the failed agent update
                action_log_entry["outcome"] = "ERROR updating agent state"
                action_log.append(action_log_entry)
                continue

            # --- Reach Probability Update ---
            next_reach_probs = reach_probs.copy()
            # Update reach for *all* players based on the acting player's action probability
            for p_idx in range(self.num_players):
                 if p_idx == player:
                      next_reach_probs[p_idx] *= action_prob
                 else:
                      # Opponent reach probability remains the same *relative* to parent node,
                      # but absolute reach is passed down. CFR updates use relative reach implicitly.
                      # We pass the full reach probability vector down.
                      pass # next_reach_probs[p_idx] = reach_probs[p_idx] (already copied)

            # --- Recursive Call ---
            # Pass the action_log down the recursion
            action_utilities[i] = self._cfr_recursive(
                next_game_state, next_agent_states, next_reach_probs, iteration, action_log
            )

            # --- Update Node Value ---
            node_value += action_prob * action_utilities[i]

            # --- Finalize Action Log Entry ---
            # Add outcome/utility after recursive call returns
            action_log_entry["outcome_utilities"] = action_utilities[i].tolist() # Convert ndarray to list
            action_log_entry["state_desc_after"] = str(next_game_state)
            action_log.append(action_log_entry)


        # 7. Calculate and Update Regrets (CFR+)
        opponent_reach = reach_probs[opponent]
        # Note: RM+ ensures regrets >= 0. Cumulative stored in self.regret_sum.
        # Only update regrets if the node was reachable by the current player
        if player_reach > 0:
            player_action_values = action_utilities[:, player]
            player_node_value = node_value[player]
            instantaneous_regret = player_action_values - player_node_value

            # Update cumulative regret sum, weighted by opponent reach
            # Opponent reach might be 0 if they took a zero-prob action earlier. Use 1.0 as weight if opponent_reach is 0.
            update_weight = opponent_reach if opponent_reach > 0 else 1.0

            # Fetch the current regrets again in case it was initialized inside the loop
            current_regrets = self.regret_sum[infoset_key]
            self.regret_sum[infoset_key] = np.maximum(0.0, current_regrets + update_weight * instantaneous_regret)


        # 8. Return Node Value (expected utilities for P0, P1)
        return node_value


    def _filter_observation(self, obs: AgentObservation, observer_id: int) -> AgentObservation:
         """ Filters sensitive information from observation based on observer."""
         filtered_obs = copy.copy(obs) # Shallow copy is fine for top level

         # --- Hide opponent's drawn card ---
         # If the observer is not the actor AND the drawn card was not immediately discarded
         # (meaning it was potentially replaced), hide it.
         if obs.drawn_card and obs.acting_player != observer_id:
              is_replace = isinstance(obs.action, ActionReplace)
              if is_replace:
                   filtered_obs.drawn_card = None
              # If discarded, card is public via discard_top_card, no need to hide drawn_card

         # --- Hide peek results if observer wasn't the peeker ---
         if obs.peeked_cards and obs.acting_player != observer_id:
              # Only hide if the peek action wasn't targeting the observer
              is_king_look = isinstance(obs.action, ActionAbilityKingLookSelect)
              is_peek_other = isinstance(obs.action, ActionAbilityPeekOtherSelect)

              if is_king_look:
                  # King look reveals both cards to the actor. Observer only sees if they were targeted.
                  filtered_peek = {}
                  for (p_idx, h_idx), card in obs.peeked_cards.items():
                      if p_idx == observer_id:
                           # Should this be revealed? Assume yes for now, server might hide it.
                           # For CFR, assume peek reveals card info for belief update.
                           filtered_peek[(p_idx, h_idx)] = card
                           # If observer was peeked, they don't see the *other* peeked card.
                           break # Only include the observer's card if they were peeked.
                  filtered_obs.peeked_cards = filtered_peek if filtered_peek else None
              elif is_peek_other:
                   # Peek other only reveals opponent card to actor. Observer sees nothing.
                   filtered_obs.peeked_cards = None
              # ActionAbilityPeekOwnSelect is only seen by the actor, handled by acting_player check.

         # --- Snap Results: Assumed fully public ---
         # No filtering needed for snap_results based on current spec.

         return filtered_obs

    def _create_observation(self, prev_state: Optional[CambiaGameState], action: Optional[GameAction], next_state: CambiaGameState, acting_player: int, snap_results: List[Dict]) -> AgentObservation:
         """ Creates the observation object based on state change. """
         discard_top = next_state.get_discard_top()
         hand_sizes = [next_state.get_player_card_count(i) for i in range(self.num_players)]
         stock_size = next_state.get_stockpile_size()
         cambia_called = next_state.cambia_caller_id is not None
         who_called = next_state.cambia_caller_id
         game_over = next_state.is_terminal()
         turn_num = next_state.get_turn_number()

         drawn_card = None
         peeked_cards_dict = None

         # Extract drawn card info if applicable (from pending state *after* draw action)
         if isinstance(action, (ActionDrawStockpile, ActionDrawDiscard)):
              if next_state.pending_action and next_state.pending_action_player == acting_player:
                   drawn_card = next_state.pending_action_data.get("drawn_card")

         # Extract peeked cards info if applicable (from pending state *after* look action)
         if acting_player != -1:
              opp_idx_func = lambda p: next_state.get_opponent_index(p) # Use lambda for safety
              # Peek 7/8 (Own)
              if isinstance(action, ActionAbilityPeekOwnSelect):
                  if 0 <= action.target_hand_index < next_state.get_player_card_count(acting_player):
                       peeked_card = next_state.get_player_hand(acting_player)[action.target_hand_index]
                       peeked_cards_dict = {(acting_player, action.target_hand_index): peeked_card}
              # Peek 9/T (Other)
              elif isinstance(action, ActionAbilityPeekOtherSelect):
                  opp_idx = opp_idx_func(acting_player)
                  if 0 <= action.target_opponent_hand_index < next_state.get_player_card_count(opp_idx):
                       peeked_card = next_state.get_player_hand(opp_idx)[action.target_opponent_hand_index]
                       peeked_cards_dict = {(opp_idx, action.target_opponent_hand_index): peeked_card}
              # King Look (triggered by King discard, selection happens in KingLookSelect)
              elif isinstance(action, ActionAbilityKingLookSelect):
                   # The pending_action_data *after* KingLookSelect contains the peeked cards
                   # Check if the *next* state has this pending data (means look was just chosen)
                   if (next_state.pending_action and
                       isinstance(next_state.pending_action, ActionAbilityKingSwapDecision) and
                       next_state.pending_action_player == acting_player):
                        data = next_state.pending_action_data
                        if "own_idx" in data and "opp_idx" in data and "card1" in data and "card2" in data:
                            opp_idx = opp_idx_func(acting_player)
                            peeked_cards_dict = {
                                 (acting_player, data["own_idx"]): data["card1"],
                                 (opp_idx, data["opp_idx"]): data["card2"]
                            }

         # Use the snap_results directly from the argument (should be from next_state)
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
                  # Check normalization *before* potentially re-normalizing to avoid logging correct cases
                  current_sum = np.sum(normalized_strategy)
                  if not np.isclose(current_sum, 1.0, atol=1e-6) and len(normalized_strategy) > 0: # Added tolerance
                        logger.debug(f"Avg strategy requires re-normalization for infoset {infoset_key}. Sum: {current_sum}. Strategy before: {normalized_strategy}")
                        normalized_strategy = normalize_probabilities(normalized_strategy) # Re-normalize
                        # Check again after re-normalization
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