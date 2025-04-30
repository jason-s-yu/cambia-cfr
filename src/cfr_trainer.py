# src/cfr_trainer.py
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Set, TypeAlias
from collections import defaultdict
import copy

# Import necessary components from the project
from .game_engine import CambiaGameState # Need full state for simulation
from .constants import (
     GameAction, NUM_PLAYERS,
     ActionPassSnap, ActionSnapOwn, ActionSnapOpponent,
     # Import other actions if needed for type checking or observation creation
     ActionDrawStockpile, ActionDrawDiscard, ActionReplace, ActionDiscard,
     ActionAbilityPeekOwnSelect, ActionAbilityPeekOtherSelect, ActionAbilityKingLookSelect,
)
from .agent_state import AgentState, AgentObservation # Agent's belief and observation structure
from .utils import InfosetKey, PolicyDict, get_rm_plus_strategy, normalize_probabilities # Key type, Policy dict type, RM+ util
from .config import Config # Configuration dataclass
from .card import Card # For type hinting if needed

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
            logger.info(f"Resuming training from iteration {self.current_iteration}")
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
            self.current_iteration = t + 1 # Iterations often 1-based

            # Initialize game state for this simulation run
            game_state = CambiaGameState(house_rules=self.config.cambia_rules)

            # Initial reach probabilities are 1 for all players at the root
            reach_probs = np.ones(self.num_players, dtype=np.float64)

            # Initialize agent states (one per player)
            initial_agent_states = []
            if not game_state.is_terminal():
                 initial_obs = self._create_observation(None, None, game_state, -1, []) # Initial state obs
                 for i in range(self.num_players):
                      agent = AgentState(
                          player_id=i,
                          opponent_id=game_state.get_opponent_index(i),
                          memory_level=self.config.agent_params.memory_level,
                          time_decay_turns=self.config.agent_params.time_decay_turns,
                          initial_hand_size=game_state.get_player_card_count(i),
                          config=self.config # Pass config to agent state
                      )
                      initial_hand = game_state.players[i].hand
                      peek_indices = game_state.players[i].initial_peek_indices
                      agent.initialize(initial_obs, initial_hand, peek_indices)
                      initial_agent_states.append(agent)
            else:
                 logger.error("Game seems to be terminal immediately after initialization. Aborting iteration.")
                 continue

            # Start recursive traversal from the root
            try:
                 # Pass iteration number t+1 for CFR+ weighting calculations
                 self._cfr_recursive(game_state, initial_agent_states, reach_probs, self.current_iteration)
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
                 logger.info(f"Iter {self.current_iteration}/{end_iteration} | Last: {iter_time:.3f}s | "
                             f"Infosets: {len(self.regret_sum):,} | Avg Speed: {iters_per_sec:.2f} it/s")


            if self.current_iteration % self.config.cfr_training.save_interval == 0:
                self.save_data()

        # --- Training Complete ---
        end_time = time.time()
        logger.info(f"Training finished {total_iterations_to_run} iterations.")
        logger.info(f"Total training time: {end_time - loop_start_time:.2f} seconds.")
        self.save_data()
        self.compute_average_strategy()
        logger.info("Final average strategy computed.")


    def _cfr_recursive(self, game_state: CambiaGameState, agent_states: List[AgentState], reach_probs: np.ndarray, iteration: int) -> np.ndarray:
        """
        Recursive CFR+ function. Operates on copies of states.
        Args:
            game_state: The current true state of the game (will be cloned).
            agent_states: List of current subjective agent states (will be cloned).
            reach_probs: Numpy array of reach probabilities for [player0, player1].
            iteration: The current training iteration number (1-based).

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
             # Treat as terminal with 0 utility? Risky.
             return np.zeros(self.num_players, dtype=np.float64)

        current_agent_state = agent_states[player]
        opponent = 1 - player

        # 1. Get Infoset Key from the current player's perspective
        try:
             infoset_key = current_agent_state.get_infoset_key()
        except Exception as e:
             logger.error(f"Error getting infoset key for P{player}. AgentState: {current_agent_state}. GameState: {game_state}", exc_info=True)
             return np.zeros(self.num_players, dtype=np.float64) # Treat as error state

        # 2. Get Legal Actions for the current context
        try:
            legal_actions_set = game_state.get_legal_actions()
            # Sort actions for deterministic iteration order, crucial for CFR arrays
            legal_actions = sorted(list(legal_actions_set), key=repr)
        except Exception as e:
            logger.error(f"Error getting/sorting legal actions for P{player} at state {game_state}. InfosetKey: {infoset_key}", exc_info=True)
            return np.zeros(self.num_players, dtype=np.float64)

        num_actions = len(legal_actions)

        if num_actions == 0:
             # This should typically only happen if game is terminal, but we check is_terminal() first.
             # If it happens in non-terminal, it's likely an engine state bug.
             logger.warning(f"No legal actions found for P{player} at infoset {infoset_key} in *non-terminal* state {game_state}. Returning 0 utility.")
             return np.zeros(self.num_players, dtype=np.float64)


        # 3. Initialize Infoset Regret/Strategy Sums/Reach Sums if New or Size Mismatch
        current_regrets = self.regret_sum.get(infoset_key)
        if current_regrets is None or current_regrets.shape[0] != num_actions:
            self.regret_sum[infoset_key] = np.zeros(num_actions, dtype=np.float64)
            current_regrets = self.regret_sum[infoset_key]

        current_strategy_sum = self.strategy_sum.get(infoset_key)
        if current_strategy_sum is None or current_strategy_sum.shape[0] != num_actions:
            self.strategy_sum[infoset_key] = np.zeros(num_actions, dtype=np.float64)

        # Ensure reach_prob_sum entry exists (no size check needed, it's a float)
        if infoset_key not in self.reach_prob_sum:
            self.reach_prob_sum[infoset_key] = 0.0


        # 4. Compute Current Strategy (sigma^t) using RM+
        strategy = get_rm_plus_strategy(current_regrets) # Uses RM+, returns normalized probabilities

        # 5. Update Average Strategy Numerator and Denominator (CFR+)
        player_reach = reach_probs[player]
        if player_reach > 0:
             # Calculate iteration weight (w_t)
             if self.config.cfr_plus_params.weighted_averaging_enabled:
                  delay = self.config.cfr_plus_params.averaging_delay
                  # Iteration is 1-based, so weight is max(0, iteration - delay)
                  weight = float(max(0, iteration - delay))
             else:
                  weight = 1.0 # Uniform weighting if disabled

             # Update reach probability sum (denominator)
             self.reach_prob_sum[infoset_key] += weight * player_reach

             # Update strategy sum (numerator)
             self.strategy_sum[infoset_key] += weight * player_reach * strategy


        # 6. Recurse on Actions
        action_utilities = np.zeros((num_actions, self.num_players), dtype=np.float64)
        node_value = np.zeros(self.num_players, dtype=np.float64)

        for i, action in enumerate(legal_actions):
            action_prob = strategy[i]
            # Skip recursion if probability is negligible (can save compute)
            # Add pruning check here if implementing RBP
            if action_prob < 1e-9: # Threshold to avoid floating point issues
                 continue

            # --- State Transition ---
            next_game_state = game_state.clone() # Clone before applying action
            try:
                 # Apply the action to the cloned state
                 next_game_state = next_game_state.apply_action(action)
            except Exception as e:
                 logger.error(f"Error applying action {action} in state {game_state} at infoset {infoset_key}: {e}", exc_info=True)
                 action_utilities[i] = np.zeros(self.num_players) # Assign 0 utility on error?
                 continue # Skip this action

            # --- Observation Creation ---
            # Get snap log from the state *after* the action was applied
            current_snap_log = next_game_state.snap_results_log
            observation = self._create_observation(
                 prev_state=game_state, # State *before* action
                 action=action,
                 next_state=next_game_state, # State *after* action
                 acting_player=player,
                 snap_results=current_snap_log # Pass the log relevant to the transition
            )

            # --- Agent Belief Update ---
            next_agent_states = []
            for agent_idx, agent_state in enumerate(agent_states):
                  cloned_agent = agent_state.clone()
                  try:
                       player_specific_obs = self._filter_observation(observation, agent_idx)
                       cloned_agent.update(player_specific_obs)
                  except Exception as e:
                       logger.error(f"Error updating AgentState {agent_idx} for P{player} acting with {action}. Infoset: {infoset_key}. Obs: {observation}", exc_info=True)
                       # How to handle agent state update error? Maybe use previous state? Risky.
                       # For now, add the potentially corrupted state.
                  next_agent_states.append(cloned_agent)

            # --- Reach Probability Update ---
            next_reach_probs = reach_probs.copy()
            # Update reach probability for the acting player based on action probability
            next_reach_probs[player] *= action_prob
            # Ensure opponent reach probability is not touched here

            # --- Recursive Call ---
            action_utilities[i] = self._cfr_recursive(
                next_game_state,
                next_agent_states,
                next_reach_probs,
                iteration
            )

            # --- Update Node Value ---
            node_value += action_prob * action_utilities[i]


        # 7. Calculate and Update Regrets (CFR+)
        opponent_reach = reach_probs[opponent]
        # Only update regrets if the node is reachable by the player AND opponent
        if player_reach > 0 and opponent_reach > 0:
            player_action_values = action_utilities[:, player]
            player_node_value = node_value[player]
            instantaneous_regret = player_action_values - player_node_value

            # Update cumulative regret sum using CFR+ rule (RM+)
            # Weight regret update by opponent's reach probability
            # Note: RM+ uses max(0, cumulative_regret + instantaneous), we store cumulative.
            # Ensure we use the previously fetched current_regrets
            self.regret_sum[infoset_key] = np.maximum(0.0, current_regrets + opponent_reach * instantaneous_regret)


        # 8. Return Node Value (expected utilities for P0, P1)
        return node_value

    def _filter_observation(self, obs: AgentObservation, observer_id: int) -> AgentObservation:
         """ Filters sensitive information from observation based on observer."""
         filtered_obs = copy.copy(obs) # Shallow copy is fine for top level

         # --- Hide opponent's drawn card ---
         if obs.drawn_card and obs.acting_player != observer_id:
              # Hide unless it was the card just discarded
              # Check if discard pile top matches drawn card *before* action (tricky)
              # Safer: Just hide opponent's draw unless it's the new discard top
              if obs.discard_top_card != obs.drawn_card:
                   filtered_obs.drawn_card = None

         # --- Hide peek results if observer wasn't the peeker ---
         if obs.peeked_cards and obs.acting_player != observer_id:
              # Exception: King ability reveals cards to both implicitly if swap happens?
              # Rule spec implies peeker peeks, then decides swap. Observer only sees swap.
              # Keep it simple: Only the actor gets peek results.
              filtered_obs.peeked_cards = None # Clear peek dict

         # --- Filter Snap Results? ---
         # Current log includes success, penalty, indices, and potentially the actual card.
         # Should the non-snapping player see the exact card snapped?
         # Argument for: Yes, it's now public knowledge (in discard or was target).
         # Argument against: Makes infoset larger, maybe abstraction is enough.
         # Let's assume snap results are fully public for now (no filtering here).
         # filtered_obs.snap_results = [...] # Filter if needed

         return filtered_obs

    def _create_observation(self, prev_state: Optional[CambiaGameState], action: Optional[GameAction], next_state: CambiaGameState, acting_player: int, snap_results: List[Dict]) -> AgentObservation:
         """ Creates the observation object based on state change. """
         # --- Visible state components from NEXT state ---
         discard_top = next_state.get_discard_top()
         hand_sizes = [next_state.get_player_card_count(i) for i in range(self.num_players)]
         stock_size = next_state.get_stockpile_size()
         cambia_called = next_state.cambia_caller_id is not None
         who_called = next_state.cambia_caller_id
         game_over = next_state.is_terminal()
         turn_num = next_state.get_turn_number() # Use turn number from next state

         # --- Potentially private info for the acting player (if applicable) ---
         drawn_card = None
         peeked_cards_dict = None # Format: {(player_idx, hand_idx): Card}

         # Drawn Card: Extract from next state's pending data if action was Draw*
         if isinstance(action, (ActionDrawStockpile, ActionDrawDiscard)):
              if next_state.pending_action and next_state.pending_action_player == acting_player:
                   drawn_card = next_state.pending_action_data.get("drawn_card")

         # Peeked Cards: Extract from state based on action type
         if acting_player != -1: # Only relevant if there was an acting player
              if isinstance(action, ActionAbilityPeekOwnSelect):
                  # Action completes, info is in next_state's hand
                  if 0 <= action.target_hand_index < next_state.get_player_card_count(acting_player):
                       peeked_card = next_state.get_player_hand(acting_player)[action.target_hand_index]
                       peeked_cards_dict = {(acting_player, action.target_hand_index): peeked_card}
              elif isinstance(action, ActionAbilityPeekOtherSelect):
                  # Action completes, info is in next_state's hand
                  opp_idx = next_state.get_opponent_index(acting_player)
                  if 0 <= action.target_opponent_hand_index < next_state.get_player_card_count(opp_idx):
                       peeked_card = next_state.get_player_hand(opp_idx)[action.target_opponent_hand_index]
                       peeked_cards_dict = {(opp_idx, action.target_opponent_hand_index): peeked_card}
              elif isinstance(action, ActionAbilityKingLookSelect):
                  # Action leads to pending decision, info stored in pending_action_data
                  if next_state.pending_action_data:
                       data = next_state.pending_action_data
                       if "own_idx" in data and "opp_idx" in data and "card1" in data and "card2" in data:
                            peeked_cards_dict = {
                                 (acting_player, data["own_idx"]): data["card1"],
                                 (next_state.get_opponent_index(acting_player), data["opp_idx"]): data["card2"]
                            }

         obs = AgentObservation(
             acting_player=acting_player, # Player who took the action leading to next_state
             action=action,               # The action taken
             discard_top_card=discard_top,
             player_hand_sizes=hand_sizes,
             stockpile_size=stock_size,
             drawn_card=drawn_card,        # Will be filtered for opponent
             peeked_cards=peeked_cards_dict,# Will be filtered for opponent
             snap_results=snap_results,    # Pass the log collected during the state transition
             did_cambia_get_called=cambia_called,
             who_called_cambia=who_called,
             is_game_over=game_over,
             current_turn=turn_num          # Turn number associated with the *next* state
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

             if r_sum > 1e-9: # Use a threshold to avoid division by near-zero
                  normalized_strategy = s_sum / r_sum
                  # Check for normalization issues post-division
                  if np.isnan(normalized_strategy).any():
                       logger.warning(f"NaN detected in avg strategy for infoset {infoset_key}. Num: {s_sum}, Denom: {r_sum}. Defaulting to uniform.")
                       nan_count += 1
                       num_actions = len(s_sum)
                       normalized_strategy = np.ones(num_actions) / num_actions if num_actions > 0 else np.array([])
                  elif not np.isclose(np.sum(normalized_strategy), 1.0) and len(normalized_strategy) > 0:
                        # Re-normalize if slightly off due to float precision
                        normalized_strategy = normalize_probabilities(normalized_strategy)
                        if not np.isclose(np.sum(normalized_strategy), 1.0):
                             logger.warning(f"Avg strategy normalization failed for infoset {infoset_key}. Sum: {np.sum(normalized_strategy)}. Num: {s_sum}, Denom: {r_sum}. Final Strategy: {normalized_strategy}")
                             norm_issue_count += 1
             else:
                  # If total weighted reach is zero or negligible, use uniform strategy
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