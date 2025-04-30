# src/analysis_tools.py
import logging
import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import asdict

# Import necessary types
from .game_engine import CambiaGameState, PlayerState
from .agent_state import AgentState
from .constants import GameAction
from .card import Card
from .config import Config

logger = logging.getLogger(__name__)

def _serialize_card(card: Card) -> str:
    """Helper to serialize a Card object to a string."""
    return str(card)

def _serialize_action(action: GameAction) -> Dict[str, Any]:
    """Helper to serialize a GameAction (NamedTuple)."""
    if hasattr(action, '_asdict'): # Check if it's a NamedTuple
        return action._asdict()
    else:
        # Handle non-NamedTuple actions if any, or just return representation
        return {"type": type(action).__name__, "details": repr(action)}

class AnalysisTools:
    """Provides tools for analyzing CFR training progress and game history."""

    def __init__(self, config: Config, log_dir: Optional[str] = None, log_file_prefix: Optional[str] = None):
        self.config = config
        self.log_file_path = None
        if log_dir and log_file_prefix:
            os.makedirs(log_dir, exist_ok=True)
            # For analysis, usually one consistent file is better than timestamped
            self.log_file_path = os.path.join(log_dir, f"{log_file_prefix}_history.jsonl")
            logger.info(f"AnalysisTools initialized. Game history will be logged to: {self.log_file_path}")
        else:
            logger.warning("AnalysisTools initialized without log directory/prefix. Game history logging disabled.")
        # Initialize exploitability calculation components if needed
        # self.best_response_calculator = ...

    def calculate_exploitability(self, average_strategy: Dict) -> float:
        """
        Calculates the exploitability of the agent's average strategy.
        Requires implementing a Best Response algorithm.
        """
        logger.warning("Exploitability calculation not yet implemented.")
        # Placeholder implementation
        # 1. Initialize Best Response calculation from the root.
        # 2. Traverse the game tree.
        # 3. At each opponent node, calculate the value of the best response action.
        # 4. At each agent node, calculate the expected value based on the agent's average strategy.
        # 5. Return the value of the best response at the root.
        return -1.0 # Indicate not implemented

    def log_game_history(self, game_details: Dict[str, Any]):
        """
        Logs the details of a completed game simulation to a JSON Lines file.
        """
        if not self.log_file_path:
             logger.debug("Game history logging disabled (no file path).")
             return

        try:
            with open(self.log_file_path, 'a') as f:
                # Ensure details are JSON serializable (convert complex objects if needed)
                json_record = json.dumps(game_details, default=str) # Use default=str for unknown types
                f.write(json_record + '\n')
        except IOError as e:
            logger.error(f"Error writing game history to {self.log_file_path}: {e}")
        except TypeError as e:
            # Log the specific details that failed serialization if possible
            problematic_part = repr(game_details) # Default to repr
            logger.error(f"Error serializing game details to JSON: {e}. Details (repr): {problematic_part}")


    def format_game_details_for_log(self, game_state: CambiaGameState, iteration: int, initial_hands: Optional[List[List[Card]]] = None, action_sequence: Optional[List[Dict]] = None) -> Dict[str, Any]:
         """
         Formats the necessary details from a finished game state for logging.
         Includes placeholders for initial hands and action sequence.
         """
         final_hands_serializable = []
         final_scores = []
         for i, player_state in enumerate(game_state.players):
              if hasattr(player_state, 'hand') and isinstance(player_state.hand, list):
                   hand_str = [_serialize_card(c) for c in player_state.hand]
                   final_hands_serializable.append(hand_str)
                   final_scores.append(sum(c.value for c in player_state.hand))
              else:
                   final_hands_serializable.append(["ERROR - Invalid Player State"])
                   final_scores.append(999) # Error score

         initial_hands_serializable = None
         if initial_hands:
              initial_hands_serializable = [[_serialize_card(c) for c in hand] for hand in initial_hands]

         # Ensure action sequence is serializable
         action_sequence_serializable = None
         if action_sequence:
              action_sequence_serializable = []
              for entry in action_sequence:
                   serializable_entry = entry.copy() # Start with a copy
                   if 'action' in entry and entry['action'] is not None:
                        serializable_entry['action'] = _serialize_action(entry['action'])
                   # Infoset keys (tuples) should be serializable by default json
                   action_sequence_serializable.append(serializable_entry)


         return {
             "game_id": f"sim_{iteration}_{game_state.get_turn_number()}", # Example ID
             "iteration": iteration,
             "player_ids": list(range(game_state.num_players)),
             "initial_hands": initial_hands_serializable, # Needs to be captured at start
             "action_sequence": action_sequence_serializable, # Needs logging during CFR recursion
             "final_hands": final_hands_serializable,
             "final_scores": final_scores,
             "winner": game_state._winner,
             "final_utilities": game_state._utilities,
             "num_turns": game_state.get_turn_number(),
             "cambia_caller": game_state.cambia_caller_id,
             "house_rules": asdict(game_state.house_rules) # Serialize rules dataclass
         }