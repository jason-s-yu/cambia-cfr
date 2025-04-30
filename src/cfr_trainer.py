# src/cfr_trainer.py
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Set
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

class CFRTrainer:
    """Implements the CFR+ algorithm for training a Cambia agent via self-play."""

    def __init__(self, config: Config):
        self.config = config
        self.num_players = NUM_PLAYERS # Should match game engine
        # Core data structures: maps InfosetKey -> np.ndarray
        self.regret_sum: PolicyDict = defaultdict(lambda: np.array([], dtype=np.float64))
        self.strategy_sum: PolicyDict = defaultdict(lambda: np.array([], dtype=np.float64))
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
            self.regret_sum = defaultdict(lambda: np.array([], dtype=np.float64), loaded[0])
            self.strategy_sum = defaultdict(lambda: np.array([], dtype=np.float64), loaded[1])
            self.current_iteration = loaded[2]
            logger.info(f"Resuming training from iteration {self.current_iteration}")
        else:
            logger.info("No saved data found or error loading. Starting fresh.")
            self.regret_sum = defaultdict(lambda: np.array([], dtype=np.float64))
            self.strategy_sum = defaultdict(lambda: np.array([], dtype=np.float64))
            self.current_iteration = 0


    def save_data(self, filepath: Optional[str] = None):
        """Saves the current training data."""
        from .persistence import save_agent_data
        path = filepath or self.config.persistence.agent_data_save_path
        # Convert defaultdicts to regular dicts for saving via joblib/pickle
        save_agent_data(dict(self.regret_sum), dict(self.strategy_sum), self.current_iteration, path)


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
                 initial_obs = self._create_initial_observation(game_state)
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
                 # Pass iteration number t+1 for potential weighted averaging calculations
                 self._cfr_recursive(game_state, initial_agent_states, reach_probs, self.current_iteration)
            except Exception as e:
                 logger.exception(f"Error during CFR recursion on iteration {self.current_iteration}: {e}")
                 continue

            # Log progress and save periodically
            iter_time = time.time() - iter_start_time
            if self.current_iteration % 100 == 0 or self.current_iteration == end_iteration:
                 total_elapsed = time.time() - loop_start_time
                 iters_per_sec = (self.current_iteration - start_iteration) / total_elapsed if total_elapsed > 0 else 0
                 logger.info(f"Iter {self.current_iteration}/{end_iteration} | Time: {iter_time:.3f}s | "
                             f"Total Infosets: {len(self.regret_sum)} | Speed: {iters_per_sec:.2f} it/s")

            if self.current_iteration % self.config.cfr_training.save_interval == 0:
                self.save_data()

        # --- Training Complete ---
        end_time = time.time()
        logger.info(f"Training finished {total_iterations_to_run} iterations.")
        logger.info(f"Total training time: {end_time - loop_start_time:.2f} seconds.")
        self.save_data()
        self.compute_average_strategy()
        logger.info("Final average strategy computed.")


    def _create_initial_observation(self, game_state: CambiaGameState) -> AgentObservation:
         """ Creates the initial observation before the first turn. """
         return AgentObservation(
             acting_player=-1, # No action yet
             action=None,
             discard_top_card=game_state.get_discard_top(),
             player_hand_sizes=[game_state.get_player_card_count(i) for i in range(self.num_players)],
             stockpile_size=game_state.get_stockpile_size(),
             snap_results=[], # No snaps initially
             did_cambia_get_called=False,
             who_called_cambia=None,
             is_game_over=False,
             current_turn=0 # Before first turn
         )

    def _cfr_recursive(self, game_state: CambiaGameState, agent_states: List[AgentState], reach_probs: np.ndarray, iteration: int) -> np.ndarray:
        """
        Recursive CFR+ function. Operates on copies of states.
        Args:
            game_state: The current true state of the game (will be cloned).
            agent_states: List of current subjective agent states (will be cloned).
            reach_probs: Numpy array of reach probabilities for [player0, player1].
            iteration: The current training iteration number (for potential weighting).

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
             return np.zeros(self.num_players, dtype=np.float64)

        current_agent_state = agent_states[player]
        opponent = 1 - player

        # 1. Get Infoset Key from the current player's perspective
        infoset_key = current_agent_state.get_infoset_key()

        # 2. Get Legal Actions for the current context
        legal_actions_set = game_state.get_legal_actions()
        try:
            legal_actions = sorted(list(legal_actions_set), key=lambda x: str(x))
        except TypeError as e:
             logger.error(f"Could not sort legal actions: {legal_actions_set}. Error: {e}")
             legal_actions = list(legal_actions_set)

        num_actions = len(legal_actions)

        if num_actions == 0:
             logger.warning(f"No legal actions found for player {player} at infoset {infoset_key} in state {game_state}. Game terminal? {game_state.is_terminal()}")
             if not game_state.is_terminal():
                  # This might indicate a bug or an unresolved game state loop
                  return np.zeros(self.num_players, dtype=np.float64)
             else:
                  return np.array([game_state.get_utility(i) for i in range(self.num_players)], dtype=np.float64)


        # 3. Initialize Infoset Regret/Strategy Sums if New
        if infoset_key not in self.regret_sum or self.regret_sum[infoset_key].shape[0] != num_actions:
            self.regret_sum[infoset_key] = np.zeros(num_actions, dtype=np.float64)
        if infoset_key not in self.strategy_sum or self.strategy_sum[infoset_key].shape[0] != num_actions:
            self.strategy_sum[infoset_key] = np.zeros(num_actions, dtype=np.float64)


        # 4. Compute Current Strategy (sigma^t) using RM+
        current_regrets = self.regret_sum[infoset_key]
        strategy = get_rm_plus_strategy(current_regrets) # Uses RM+, returns normalized probabilities

        # The strategy calculated above based on regrets will now be used for snap decisions too.

        # 5. Update Average Strategy Sum (sigma_sum accumulator)
        player_reach = reach_probs[player]
        if player_reach > 0:
             # Apply CFR+ weighting: sum iterates player_reach * strategy
             # Weighting factor (like iteration number) applied during final average calculation
             self.strategy_sum[infoset_key] += player_reach * strategy


        # 6. Recurse on Actions
        action_utilities = np.zeros((num_actions, self.num_players), dtype=np.float64)
        node_value = np.zeros(self.num_players, dtype=np.float64)

        for i, action in enumerate(legal_actions):
            action_prob = strategy[i]
            if action_prob <= 1e-9:
                 continue

            next_game_state = game_state.clone() # Clone before applying action
            try:
                 # Store snap results if action leads out of snap phase
                 snap_results_for_obs = []
                 if game_state.snap_phase_active and not next_game_state.snap_phase_active:
                      # The action taken resolved the current player's snap turn. Record outcome.
                      # Need more sophisticated tracking in game state to know success/penalty.
                      # Placeholder: Assume success if snap action, failure if pass? Too simple.
                      # Requires CambiaGameState.apply_action to return success/penalty info.
                      # Let's assume for now AgentObservation gets this info later.
                      pass # TODO: Pass snap outcome info to observation

                 # Apply the action to the cloned state
                 next_game_state = next_game_state.apply_action(action)

            except Exception as e:
                 logger.error(f"Error applying action {action} in state {game_state} at infoset {infoset_key}: {e}", exc_info=True)
                 action_utilities[i] = np.zeros(self.num_players)
                 continue

            # Create the observation AFTER applying the action
            # Pass snap results if available (requires modification to apply_action/state)
            observation = self._create_observation(game_state, action, next_game_state, player, snap_results=snap_results_for_obs)

            # Clone and update agent states based on the observation
            next_agent_states = []
            for agent_idx, agent_state in enumerate(agent_states):
                  cloned_agent = agent_state.clone()
                  try:
                       player_specific_obs = self._filter_observation(observation, agent_idx)
                       cloned_agent.update(player_specific_obs)
                  except Exception as e:
                       logger.error(f"Error updating AgentState {agent_idx} with obs {observation} from infoset {infoset_key}: {e}")
                  next_agent_states.append(cloned_agent)


            # Calculate next reach probabilities
            next_reach_probs = reach_probs.copy()
            # Correct reach probability update for the acting player
            next_reach_probs[player] *= action_prob

            # Recursively call CFR for the state resulting from the action
            action_utilities[i] = self._cfr_recursive(
                next_game_state,
                next_agent_states,
                next_reach_probs,
                iteration
            )

            # Update node value (expected value using current strategy)
            node_value += action_prob * action_utilities[i]


        # 7. Calculate and Update Regrets (CFR+)
        opponent_reach = reach_probs[opponent]
        if player_reach > 0:
            player_action_values = action_utilities[:, player]
            player_node_value = node_value[player]
            instantaneous_regret = player_action_values - player_node_value

            # Update cumulative regret sum using CFR+ rule (RM+)
            new_regrets = current_regrets + opponent_reach * instantaneous_regret
            self.regret_sum[infoset_key] = np.maximum(0.0, new_regrets)

        # 8. Return Node Value (expected utilities for P0, P1)
        return node_value

    def _filter_observation(self, obs: AgentObservation, observer_id: int) -> AgentObservation:
         """ Filters sensitive information from observation based on observer."""
         # Observer doesn't need filtering for their own actions or public info
         if obs.acting_player == observer_id:
              return obs

         # Create a copy to modify
         filtered_obs = copy.copy(obs) # Shallow copy is fine for top level

         # --- Hide information not visible to the observer ---
         # Hide opponent's drawn card unless it was immediately discarded publicly
         if obs.drawn_card and obs.discard_top_card != obs.drawn_card:
             filtered_obs.drawn_card = None

         # Hide peek results
         filtered_obs.peeked_cards = None

         # Snap results might be public? Depends on game rules / online implementation.
         # Assume for now snap success/failure/penalty is public, but exact card indices might not be.
         # Current AgentObservation doesn't have card indices in snap_results, so no filtering needed here yet.

         return filtered_obs

    def _create_observation(self, prev_state: CambiaGameState, action: GameAction, next_state: CambiaGameState, acting_player: int, snap_results: List[Dict] = []) -> AgentObservation:
         """ Creates the observation object based on state change. """
         # --- Visible state components ---
         discard_top = next_state.get_discard_top()
         hand_sizes = [next_state.get_player_card_count(i) for i in range(self.num_players)]
         stock_size = next_state.get_stockpile_size()
         cambia_called = next_state.cambia_caller_id is not None
         who_called = next_state.cambia_caller_id
         game_over = next_state.is_terminal()
         # Turn number needs careful definition. Use next acting player?
         turn_num = next_state.get_acting_player() if not game_over else -1

         # --- Potentially private info for the acting player ---
         drawn_card = None
         peeked_cards = None

         # Extract drawn card from next state's pending data if applicable
         if isinstance(action, (ActionDrawStockpile, ActionDrawDiscard)):
              if next_state.pending_action and next_state.pending_action_player == acting_player:
                  drawn_card = next_state.pending_action_data.get("drawn_card")

         # Extract peek results from next state's pending data
         # This assumes the engine stores results accessible after the *selection* action
         if isinstance(action, (ActionAbilityPeekOwnSelect, ActionAbilityPeekOtherSelect, ActionAbilityKingLookSelect)):
              # Look in pending_action_data *before* it might be cleared by the next step
              # Example: KingLookSelect sets data, KingSwapDecision uses and clears it.
              # The observation needs data available *after* the Select action.
              # Let's refine this: Peek results should be part of the direct outcome of the peek action.
              # Requires engine modification or different observation structure.
              # Placeholder: Try accessing data from the *next* state if it's still relevant
              if next_state.pending_action_player == acting_player and 'peek_results' in next_state.pending_action_data:
                   peeked_cards = next_state.pending_action_data['peek_results']
              elif prev_state.pending_action_player == acting_player and 'peek_results' in prev_state.pending_action_data and not next_state.pending_action: # Looked then decided (no swap?)
                   peeked_cards = prev_state.pending_action_data['peek_results']


         # --- Snap results ---
         # TODO: Populate `snap_results` accurately from engine state changes if needed.
         # Requires engine `apply_action` for snaps to return detailed outcomes.

         obs = AgentObservation(
             acting_player=acting_player,
             action=action,
             discard_top_card=discard_top,
             player_hand_sizes=hand_sizes,
             stockpile_size=stock_size,
             drawn_card=drawn_card, # Will be filtered for opponent
             peeked_cards=peeked_cards, # Will be filtered for opponent
             snap_results=snap_results, # Assumed public for now
             did_cambia_get_called=cambia_called,
             who_called_cambia=who_called,
             is_game_over=game_over,
             current_turn=turn_num # Approximate turn indicator
         )
         return obs


    def compute_average_strategy(self) -> PolicyDict:
        """
        Computes the average strategy from the accumulated strategy_sum.
        """
        avg_strategy: PolicyDict = {}
        logger.info(f"Computing average strategy from {len(self.strategy_sum)} infosets...")

        if not self.strategy_sum:
             logger.warning("Strategy sum is empty. Cannot compute average strategy.")
             return avg_strategy

        # Simple average: Normalize the accumulated reach-weighted strategies
        for infoset_key, s_sum in self.strategy_sum.items():
            avg_strategy[infoset_key] = normalize_probabilities(s_sum)
            # Check for NaN
            if np.isnan(avg_strategy[infoset_key]).any():
                 logger.warning(f"NaN detected in average strategy for infoset {infoset_key}. Sum was: {s_sum}. Defaulting to uniform.")
                 num_actions = len(s_sum)
                 if num_actions > 0:
                     avg_strategy[infoset_key] = np.ones(num_actions) / num_actions
                 else:
                     avg_strategy[infoset_key] = np.array([])

        self.average_strategy = avg_strategy
        logger.info("Average strategy computation complete.")
        # Log size of average strategy map
        logger.info(f"Computed average strategy for {len(self.average_strategy)} infosets.")
        return avg_strategy

    def get_average_strategy(self) -> Optional[PolicyDict]:
         """Returns the computed average strategy."""
         if self.average_strategy is None:
              logger.warning("Average strategy requested but not computed yet. Computing now...")
              return self.compute_average_strategy() # Compute if not available
         return self.average_strategy