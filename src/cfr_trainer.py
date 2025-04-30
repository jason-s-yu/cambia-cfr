# src/cfr_trainer.py
import numpy as np
import time
import logging
from typing import Callable, Dict, List, Optional, TypeAlias
from collections import defaultdict, deque
import copy
import traceback

from .game_engine import CambiaGameState, StateDelta, UndoInfo
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
                # --- Store initial state BEFORE any modifications ---
                game_state = CambiaGameState(house_rules=self.config.cambia_rules)
                initial_state_str_for_log = str(game_state) # String representation
                initial_hands_for_log = [list(p.hand) for p in game_state.players] # Copy hands
                initial_peeks_for_log = [p.initial_peek_indices for p in game_state.players]
            except Exception as e:
                logger.exception(f"Error initializing game state on iteration {self.current_iteration}: {e}")
                continue

            reach_probs = np.ones(self.num_players, dtype=np.float64)
            initial_agent_states = []
            action_log_for_game: List[Dict] = [] # Log actions for this specific game simulation

            if not game_state.is_terminal():
                 # Pass depth 0 for root call
                 initial_obs = self._create_observation(None, None, game_state, -1, []) # Create initial obs from game state
                 for i in range(self.num_players):
                      try:
                           agent = AgentState(
                               player_id=i,
                               opponent_id=game_state.get_opponent_index(i),
                               memory_level=self.config.agent_params.memory_level,
                               time_decay_turns=self.config.agent_params.time_decay_turns,
                               initial_hand_size=len(initial_hands_for_log[i]), # Use logged initial hand size
                               config=self.config
                           )
                           # Use logged initial state for hands/peeks
                           agent.initialize(initial_obs, initial_hands_for_log[i], initial_peeks_for_log[i])
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
                 # --- Run CFR Recursion ---
                 # Pass the *mutable* game_state object down the recursion
                 # Start depth tracking at 0
                 final_utilities = self._cfr_recursive(game_state, initial_agent_states, reach_probs, self.current_iteration, action_log_for_game, depth=0)

                 # --- Log Completed Game ---
                 # Use the final state of game_state after recursion completes
                 game_details = self.analysis.format_game_details_for_log(
                      game_state=game_state, # Use the final state
                      iteration=self.current_iteration,
                      initial_hands=initial_hands_for_log, # Use the captured initial hands
                      action_sequence=action_log_for_game
                  )
                 self.analysis.log_game_history(game_details)

            except RecursionError:
                logger.error(f"Recursion depth exceeded on iteration {self.current_iteration}! Saving progress and stopping.")
                # Log the game state *at the point of the error* if possible (difficult)
                logger.error(f"State at RecursionError (approx): {game_state}")
                # Log the last few actions if available
                if action_log_for_game:
                    logger.error("Last actions before RecursionError:")
                    for entry in action_log_for_game[-10:]: # Log last 10 actions
                        logger.error(f"  Turn {entry.get('turn', '?')} P{entry.get('player','?')} -> {entry.get('action', '?')}")

                logger.error("Traceback:\n%s", traceback.format_exc())
                self.save_data()
                raise # Re-raise to stop execution
            except Exception as e:
                 logger.exception(f"Error during CFR recursion on iteration {self.current_iteration}: {e}")
                 # Log the game state at the point of the error
                 logger.error(f"State at Error: {game_state}")
                 continue # Continue to the next iteration

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


    def _cfr_recursive(self, game_state: CambiaGameState, agent_states: List[AgentState], reach_probs: np.ndarray, iteration: int, action_log: List[Dict], depth: int) -> np.ndarray:
        """
        Recursive CFR+ function. Operates on the *same* game_state object,
        applying and undoing actions. Clones agent_states. Includes depth tracking.
        """

        # --- Base Case: Terminal Node ---
        if game_state.is_terminal():
            # logger.debug(f"Iter {iteration}, Depth {depth}: Reached terminal state. Utility: {[game_state.get_utility(i) for i in range(self.num_players)]}")
            return np.array([game_state.get_utility(i) for i in range(self.num_players)], dtype=np.float64)

        # --- Identify Acting Player ---
        player = game_state.get_acting_player()
        if player == -1:
             logger.error(f"Iter {iteration}, Depth {depth}: Could not determine acting player in non-terminal state. State: {game_state}")
             return np.zeros(self.num_players, dtype=np.float64)

        current_agent_state = agent_states[player]
        opponent = 1 - player

        # 1. Get Infoset Key
        try:
             if not hasattr(current_agent_state, 'own_hand') or not hasattr(current_agent_state, 'opponent_belief'):
                  logger.error(f"Iter {iteration}, P{player}, Depth {depth}: AgentState appears uninitialized before get_infoset_key(). State: {current_agent_state}")
                  return np.zeros(self.num_players, dtype=np.float64)
             infoset_key = current_agent_state.get_infoset_key()
        except Exception as e:
             logger.error(f"Iter {iteration}, P{player}, Depth {depth}: Error getting infoset key. AgentState: {current_agent_state}. GameState: {game_state}", exc_info=True)
             return np.zeros(self.num_players, dtype=np.float64)

        # 2. Get Legal Actions
        try:
            legal_actions_set = game_state.get_legal_actions()
            legal_actions = sorted(list(legal_actions_set), key=repr) # Keep alphabetical sort for now
        except Exception as e:
            logger.error(f"Iter {iteration}, P{player}, Depth {depth}: Error getting/sorting legal actions at state {game_state}. InfosetKey: {infoset_key}", exc_info=True)
            return np.zeros(self.num_players, dtype=np.float64)

        num_actions = len(legal_actions)
        if num_actions == 0:
             # If no actions but not terminal, it's an engine error or requires forcing game end
             if not game_state.is_terminal():
                  logger.warning(f"Iter {iteration}, P{player}, Depth {depth}: No legal actions found at infoset {infoset_key} in *non-terminal* state {game_state}. Forcing game end check.")
                  # Create an undo stack just for this check
                  local_undo_stack: deque[Callable[[], None]] = deque()
                  local_delta_list: StateDelta = [] # Need dummy delta list
                  # Apply the check (which modifies state and adds to local_undo_stack)
                  game_state._check_game_end(local_undo_stack, local_delta_list)
                  # Execute the undo stack for the check to revert any state changes made by it
                  while local_undo_stack:
                       try: local_undo_stack.popleft()()
                       except Exception as undo_e: logger.error(f"Iter {iteration}, Depth {depth}: Error undoing _check_game_end: {undo_e}")

                  # Now re-check if the state *became* terminal after the check
                  if game_state.is_terminal():
                       logger.info(f"Iter {iteration}, P{player}, Depth {depth}: State confirmed terminal after no-action check.")
                       return np.array([game_state.get_utility(i) for i in range(self.num_players)], dtype=np.float64)
                  else:
                       logger.error(f"Iter {iteration}, P{player}, Depth {depth}: State still non-terminal after no-action check. Engine/logic error. Returning 0 utility.")
                       return np.zeros(self.num_players, dtype=np.float64)
             else: # Already terminal, return utility
                  # logger.debug(f"Iter {iteration}, Depth {depth}: Reached terminal state (no actions). Utility: {[game_state.get_utility(i) for i in range(self.num_players)]}")
                  return np.array([game_state.get_utility(i) for i in range(self.num_players)], dtype=np.float64)


        # 3. Initialize Infoset Data
        current_regrets = self.regret_sum.get(infoset_key)
        if current_regrets is None or current_regrets.shape[0] != num_actions:
            # logger.debug(f"Iter {iteration}, P{player}, Depth {depth}: Initializing regrets/strategy for infoset {infoset_key} with {num_actions} actions.")
            self.regret_sum[infoset_key] = np.zeros(num_actions, dtype=np.float64)
            current_regrets = self.regret_sum[infoset_key] # Update reference

        current_strategy_sum = self.strategy_sum.get(infoset_key)
        if current_strategy_sum is None or current_strategy_sum.shape[0] != num_actions:
            self.strategy_sum[infoset_key] = np.zeros(num_actions, dtype=np.float64)

        if infoset_key not in self.reach_prob_sum:
            self.reach_prob_sum[infoset_key] = 0.0


        # 4. Compute Current Strategy (sigma^t)
        strategy = get_rm_plus_strategy(current_regrets)
        # if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Iter {iteration}, P{player}, Depth {depth}, Infoset {infoset_key}: Regrets={current_regrets}, Strategy={strategy}")


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

        state_before_actions_str = str(game_state) # For logging errors across actions

        for i, action in enumerate(legal_actions):
            action_prob = strategy[i]

            # --- Log Action Attempt ---
            logger.debug(f"Iter {iteration}, P{player}, Depth {depth}: Processing action {i+1}/{num_actions}: {action} (Prob: {action_prob:.4f})")

            # --- Regret-Based Pruning (Corrected Logic) ---
            node_positive_regret_sum = np.sum(np.maximum(0.0, current_regrets))
            should_prune = (use_pruning and
                           player_reach > 0 and
                           current_regrets[i] <= pruning_threshold and # Action's regret is non-positive
                           node_positive_regret_sum > pruning_threshold and # But *other* actions have positive regret
                           iteration > 10) # Warmup period


            if should_prune:
                 logger.debug(f"Iter {iteration}, P{player}, Depth {depth}: Pruning action {action}")
                 action_utilities[i] = np.zeros(self.num_players) # Store 0 util for pruned branch
                 continue # Skip simulation for this action

            # Skip actions with negligible probability unless pruning is active (as pruning might reopen them)
            if action_prob < 1e-9 and not use_pruning:
                 logger.debug(f"Iter {iteration}, P{player}, Depth {depth}: Skipping action {action} due to low probability.")
                 continue

            action_log_entry = {
                "player": player, "turn": game_state.get_turn_number(), "depth": depth, # Add depth to log
                "state_desc_before": str(game_state), # State *before* this specific action
                "infoset_key": infoset_key,
                "action": action, "action_prob": action_prob, "reach_prob": player_reach
            }
            state_before_this_action_str = action_log_entry["state_desc_before"] # Copy for error logging


            # --- State Transition (Apply & Get Undo) ---
            state_delta: Optional[StateDelta] = None
            undo_info: Optional[UndoInfo] = None
            state_after_action_desc = "ERROR" # Default description
            snap_log_before_action = list(game_state.snap_results_log) # Copy log before action

            try:
                 state_delta, undo_info = game_state.apply_action(action)
                 state_after_action_desc = str(game_state) # Capture after modification

            except Exception as apply_err:
                 logger.error(f"Iter {iteration}, P{player}, Depth {depth}: Error applying action {action} in state {state_before_this_action_str} at infoset {infoset_key}: {apply_err}", exc_info=False) # Log only exception message initially
                 # Add more detailed traceback if verbose needed
                 if logger.isEnabledFor(logging.DEBUG): logger.exception("Full traceback for apply_action error:")
                 action_utilities[i] = np.zeros(self.num_players)
                 action_log_entry["outcome"] = f"ERROR applying action: {apply_err}"
                 action_log_entry["state_desc_after"] = state_after_action_desc # Log state after error if possible
                 action_log.append(action_log_entry)
                 # Don't attempt undo if apply failed, state might be corrupt or unchanged
                 undo_info = None # Prevent undo attempt later
                 # If one action fails, maybe skip others in this infoset? For now, continue.
                 continue # Skip recursion for this failed action

            # --- Observation Creation (using state *after* action) ---
            current_snap_log = game_state.snap_results_log # Log from *after* action application
            observation = self._create_observation(
                 prev_state=None, # No longer tracking prev_state explicitly
                 action=action,
                 next_state=game_state, # Pass current (modified) state
                 acting_player=player,
                 snap_results=current_snap_log # Log from *after* action application
            )

            # --- Agent Belief Update (operates on clones) ---
            next_agent_states = []
            agent_update_failed = False
            for agent_idx, agent_state in enumerate(agent_states):
                  # Clone agent state for this branch
                  cloned_agent = agent_state.clone()
                  try:
                       # Filter observation for the specific agent
                       player_specific_obs = self._filter_observation(observation, agent_idx)
                       cloned_agent.update(player_specific_obs)
                  except Exception as agent_update_err:
                       logger.error(f"Iter {iteration}, P{player}, Depth {depth}: Error updating AgentState {agent_idx} for action {action}. Infoset: {infoset_key}. Obs: {observation}", exc_info=True)
                       agent_update_failed = True
                       break # Stop processing agents for this action branch
                  next_agent_states.append(cloned_agent)

            if agent_update_failed:
                action_utilities[i] = np.zeros(self.num_players) # Assign zero utility
                action_log_entry["outcome"] = "ERROR updating agent state"
                action_log_entry["state_desc_after"] = state_after_action_desc
                action_log.append(action_log_entry)
                # Attempt to undo the game state change before continuing
                if undo_info:
                     try: undo_info()
                     except Exception as undo_e: logger.error(f"Iter {iteration}, Depth {depth}: Error undoing action after agent update failure: {undo_e}")
                continue # Move to the next action

            # --- Reach Probability Update ---
            next_reach_probs = reach_probs.copy()
            # Update reach prob using the probability of the action *taken*
            # Note: We scale reach by the action probability *regardless* of who is updating.
            # The regrets are only updated for the current player, but the traversal
            # always follows the computed strategy profile.
            next_reach_probs[player] *= action_prob # This seems correct for player p's contribution
            # Should we scale opponent's reach prob? Typically, pi_{-i}(h) for regret updates.
            # Let's rethink this: reach_probs[p] = pi_p(h), reach_probs[opp] = pi_opp(h)
            # Next reach probs for recursion: pi_p(ha) = pi_p(h)*sigma(a), pi_opp(ha) = pi_opp(h)
            # So only the acting player's reach prob needs scaling by sigma(a).
            # The regret update uses pi_{-i}(h) = reach_probs[opponent]. This seems correct.


            # --- Recursive Call (with modified game_state) ---
            try:
                 logger.debug(f"Iter {iteration}, P{player}, Depth {depth}: Recursing ({i+1}/{num_actions}) for action: {action}")
                 # Pass the modified game_state down, increment depth
                 recursive_utilities = self._cfr_recursive(
                     game_state, next_agent_states, next_reach_probs, iteration, action_log, depth + 1
                 )
                 action_utilities[i] = recursive_utilities
                 logger.debug(f"Iter {iteration}, P{player}, Depth {depth}: Returned from recursion for {action}. Util: {recursive_utilities}")

            except RecursionError: # Catch RecursionError specifically here
                  logger.error(f"Iter {iteration}, P{player}, Depth {depth}: Recursion depth exceeded during action: {action}. State: {game_state}", exc_info=False)
                  action_log_entry["outcome"] = "ERROR: RecursionError"
                  action_log.append(action_log_entry)
                  # Re-raise the error to be caught by the main loop
                  raise
            except Exception as recursive_err:
                  # Simplify error message to avoid recursion in logging
                  logger.error(f"Iter {iteration}, P{player}, Depth {depth}: Error in recursive call for action {action}. Infoset: {infoset_key}", exc_info=False)
                  if logger.isEnabledFor(logging.DEBUG): logger.exception("Full traceback for recursion error:")
                  action_utilities[i] = np.zeros(self.num_players) # Assign zero utility on error
                  # Log the error in the action sequence
                  action_log_entry["outcome"] = f"ERROR in recursion: {recursive_err}"
                  action_log_entry["outcome_utilities"] = action_utilities[i].tolist()
                  action_log_entry["state_desc_after"] = state_after_action_desc
                  action_log.append(action_log_entry)
                  # Attempt to undo state before continuing
                  if undo_info:
                      try: undo_info()
                      except Exception as undo_e: logger.error(f"Iter {iteration}, Depth {depth}: Error undoing action after recursion error: {undo_e}")
                  continue # Move to the next action

            # --- Finalize Action Log Entry (Success Case) ---
            action_log_entry["outcome_utilities"] = action_utilities[i].tolist()
            action_log_entry["state_desc_after"] = state_after_action_desc # Already captured
            action_log.append(action_log_entry)

            # --- Undo Action ---
            if undo_info:
                 try:
                      undo_info()
                 except Exception as undo_e:
                      logger.exception(f"FATAL: Iter {iteration}, P{player}, Depth {depth}: Error undoing action {action} from state {state_before_this_action_str}. State may be corrupt. Stopping branch.")
                      # Return error utility or raise? Raising might stop training. Return 0 for now.
                      return np.zeros(self.num_players, dtype=np.float64)
            else:
                 # This case should ideally not be reached if apply_action worked
                 logger.error(f"Iter {iteration}, P{player}, Depth {depth}: Missing undo information for successful action {action}. State corrupt.")
                 return np.zeros(self.num_players, dtype=np.float64)
        # --- End Action Loop ---


        # --- Calculate Node Value (after exploring all actions/pruning) ---
        node_value = np.sum(strategy[:, np.newaxis] * action_utilities, axis=0)


        # --- Update Regrets ---
        opponent_reach = reach_probs[opponent]
        # Only update regrets if the player acting could reach this node
        if player_reach > 0:
            # Calculate instantaneous regret for the acting player
            player_action_values = action_utilities[:, player]
            player_node_value = node_value[player]
            instantaneous_regret = player_action_values - player_node_value

            # Weight regret update by opponent's reach probability (counterfactual value)
            update_weight = opponent_reach # pi_{-i}(h)

            current_regrets_before_update = np.copy(self.regret_sum[infoset_key]) # Copy for logging
            # Apply Regret Matching+ update: R += weighted_instantaneous_regret, then take max(0, R)
            self.regret_sum[infoset_key] = np.maximum(0.0, current_regrets_before_update + update_weight * instantaneous_regret)

            # --- Log Regret Update ---
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Iter {iteration}, P{player}, Depth {depth}, Infoset {infoset_key}: NodeVal={player_node_value:.4f}, ActionUtils={player_action_values}, InstRegret={instantaneous_regret}")
                logger.debug(f"Iter {iteration}, P{player}, Depth {depth}, Infoset {infoset_key}: Updating Regrets. OppReach: {opponent_reach:.4f}, Weight: {update_weight:.4f}")
                logger.debug(f"Iter {iteration}, P{player}, Depth {depth}, Infoset {infoset_key}: Regrets Old: {current_regrets_before_update}")
                logger.debug(f"Iter {iteration}, P{player}, Depth {depth}, Infoset {infoset_key}: Regrets New: {self.regret_sum[infoset_key]}")


        # --- Return Node Value ---
        logger.debug(f"Iter {iteration}, P{player}, Depth {depth}: Returning node value: {node_value}")
        return node_value


    def _filter_observation(self, obs: AgentObservation, observer_id: int) -> AgentObservation:
         """ Filters sensitive information from observation based on observer."""
         filtered_obs = copy.copy(obs) # Shallow copy is fine for dataclass

         # Mask drawn card unless observer is the actor OR it was discarded (public)
         if obs.drawn_card and obs.acting_player == observer_id:
              pass # Observer drew it, they see it
         elif obs.drawn_card and isinstance(obs.action, ActionDiscard):
              pass # Drawn card was discarded, now public (via discard_top_card)
         elif obs.drawn_card and isinstance(obs.action, ActionReplace):
              # If observer didn't act, they don't see the card used for replacing
              if obs.acting_player != observer_id:
                  filtered_obs.drawn_card = None
         elif obs.drawn_card:
             # Default: if observer didn't act and card wasn't discarded/replaced, hide it
             if obs.acting_player != observer_id:
                  filtered_obs.drawn_card = None


         # Mask peeked cards based on observer
         if obs.peeked_cards:
              filtered_peek = {}
              for (p_idx, h_idx), card in obs.peeked_cards.items():
                   # Observer sees peeks involving themselves or if they were the actor
                   if p_idx == observer_id or obs.acting_player == observer_id:
                        filtered_peek[(p_idx, h_idx)] = card
              filtered_obs.peeked_cards = filtered_peek if filtered_peek else None

         # Snap results log is public knowledge
         # Hand sizes, discard top, stockpile size, game state flags are public

         return filtered_obs

    def _create_observation(self, prev_state: Optional[CambiaGameState], action: Optional[GameAction], next_state: CambiaGameState, acting_player: int, snap_results: List[Dict]) -> AgentObservation:
         """ Creates the observation object based on state change (uses next_state). """
         # Information directly from the state *after* the action
         discard_top = next_state.get_discard_top()
         hand_sizes = [next_state.get_player_card_count(i) for i in range(self.num_players)]
         stock_size = next_state.get_stockpile_size()
         cambia_called = next_state.cambia_caller_id is not None
         who_called = next_state.cambia_caller_id
         game_over = next_state.is_terminal()
         turn_num = next_state.get_turn_number()

         # Information revealed *by* the action or resulting pending state
         drawn_card = None
         peeked_cards_dict = None

         # Check pending state *after* action for revealed info
         if next_state.pending_action and next_state.pending_action_player == acting_player:
              pending_data = next_state.pending_action_data
              # If pending state resulted from a draw
              if isinstance(next_state.pending_action, ActionDiscard):
                   drawn_card = pending_data.get("drawn_card")
              # If pending state resulted from King Look (reveals cards before swap decision)
              elif isinstance(next_state.pending_action, ActionAbilityKingSwapDecision):
                  if "own_idx" in pending_data and "opp_idx" in pending_data and "card1" in pending_data and "card2" in pending_data:
                      opp_idx = next_state.get_opponent_index(acting_player)
                      peeked_cards_dict = {
                           (acting_player, pending_data["own_idx"]): pending_data["card1"],
                           (opp_idx, pending_data["opp_idx"]): pending_data["card2"]
                      }

         # Check the action itself for reveals (e.g., immediate peeks)
         # Need to check the hand state *after* the action but *before* potential state changes from recursion
         # This logic is tricky with mutable state. Assume `next_state` reflects the state immediately after the action.
         if isinstance(action, ActionAbilityPeekOwnSelect):
              if 0 <= action.target_hand_index < next_state.get_player_card_count(acting_player):
                   # Ensure hand access is safe
                   hand = next_state.get_player_hand(acting_player)
                   if hand and 0 <= action.target_hand_index < len(hand):
                        peeked_card = hand[action.target_hand_index]
                        peeked_cards_dict = {(acting_player, action.target_hand_index): peeked_card}
         elif isinstance(action, ActionAbilityPeekOtherSelect):
              opp_idx = next_state.get_opponent_index(acting_player)
              if 0 <= action.target_opponent_hand_index < next_state.get_player_card_count(opp_idx):
                  opp_hand = next_state.get_player_hand(opp_idx)
                  if opp_hand and 0 <= action.target_opponent_hand_index < len(opp_hand):
                       peeked_card = opp_hand[action.target_opponent_hand_index]
                       peeked_cards_dict = {(opp_idx, action.target_opponent_hand_index): peeked_card}

         # Get snap results from the state *after* the action (apply_action updates this)
         # If apply_action failed, snap_results might be stale, but pass what we have.
         final_snap_results = next_state.snap_results_log

         obs = AgentObservation(
             acting_player=acting_player, action=action,
             discard_top_card=discard_top, player_hand_sizes=hand_sizes, stockpile_size=stock_size,
             drawn_card=drawn_card, peeked_cards=peeked_cards_dict, snap_results=final_snap_results,
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