# src/analysis_tools.py
import logging
import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import numpy as np

# Import necessary types
from .game.engine import CambiaGameState
from .agent_state import AgentState  # Needed for Best Response Agent state
from .constants import GameAction, DecisionContext, NUM_PLAYERS, CardObject
from .config import Config
from .utils import (
    InfosetKey,
    PolicyDict,
    normalize_probabilities,
)  # Need policy dict type

logger = logging.getLogger(__name__)


def _serialize_card(card: CardObject) -> Optional[str]:
    """Helper to serialize a Card object to a string."""
    return str(card) if card else None


def _serialize_action(action: GameAction) -> Dict[str, Any]:
    """Helper to serialize a GameAction (NamedTuple)."""
    if hasattr(action, "_asdict"):  # Check if it's a NamedTuple
        return action._asdict()
    else:
        return {"type": type(action).__name__, "details": repr(action)}


class AnalysisTools:
    """Provides tools for analyzing CFR training progress and game history."""

    def __init__(
        self,
        config: Config,
        log_dir: Optional[str] = None,
        log_file_prefix: Optional[str] = None,
    ):
        self.config = config
        self.log_file_path = None
        if log_dir and log_file_prefix:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file_path = os.path.join(log_dir, f"{log_file_prefix}_history.jsonl")
            logger.info(
                "AnalysisTools initialized. Game history log: %s", self.log_file_path
            )
        else:
            logger.warning(
                "AnalysisTools initialized without log directory/prefix. History logging disabled."
            )

    def calculate_exploitability(
        self, average_strategy: PolicyDict, config: Config
    ) -> float:
        """
        Calculates the exploitability of the agent's average strategy.
        Runs Best Response for each player against the average strategy and averages the results.
        """
        if not average_strategy:
            logger.warning("Cannot calculate exploitability: Average strategy is empty.")
            return float("inf")

        total_value = 0.0
        try:
            # Calculate value for player 0 playing Best Response against player 1's average strategy
            br_value_p0 = self._compute_best_response_value(
                average_strategy, config, br_player=0
            )
            logger.info("Best Response value for Player 0: %.6f", br_value_p0)

            # Calculate value for player 1 playing Best Response against player 0's average strategy
            br_value_p1 = self._compute_best_response_value(
                average_strategy, config, br_player=1
            )
            logger.info("Best Response value for Player 1: %.6f", br_value_p1)

            # Exploitability = (Value of P0's BR + Value of P1's BR) / 2 (for two-player zero-sum)
            exploitability = (br_value_p0 + br_value_p1) / 2.0
            logger.info("Calculated Exploitability: %.6f", exploitability)
            return exploitability

        except Exception:
            logger.exception("Error during exploitability calculation:")
            return float("inf")  # Indicate error

    def _compute_best_response_value(
        self, opponent_avg_strategy: PolicyDict, config: Config, br_player: int
    ) -> float:
        """Computes the value of the best response strategy for 'br_player' against the opponent's fixed average strategy."""
        opponent_player = 1 - br_player
        game_state = CambiaGameState(house_rules=config.cambia_rules)

        # Initialize AgentState for the BR player (needed for infoset key generation)
        initial_obs = self._create_initial_observation(game_state)
        br_agent_state = AgentState(
            player_id=br_player,
            opponent_id=opponent_player,
            memory_level=config.agent_params.memory_level,
            time_decay_turns=config.agent_params.time_decay_turns,
            initial_hand_size=len(game_state.players[br_player].hand),
            config=config,
        )
        br_agent_state.initialize(
            initial_obs,
            game_state.players[br_player].hand,
            game_state.players[br_player].initial_peek_indices,
        )

        # Start recursive calculation from the root
        return self._best_response_recursive(
            game_state, opponent_avg_strategy, br_player, br_agent_state, depth=0
        )

    def _best_response_recursive(
        self,
        game_state: CambiaGameState,
        opponent_avg_strategy: PolicyDict,
        br_player: int,
        br_agent_state: AgentState,
        depth: int,
    ) -> float:
        """Recursive function for Best Response calculation."""
        if game_state.is_terminal():
            return game_state.get_utility(br_player)

        acting_player = game_state.get_acting_player()
        if acting_player == -1:
            logger.error(
                "BR Calc: Could not determine acting player at depth %d. State: %s",
                depth,
                game_state,
            )
            return 0.0  # Error case

        opponent_player = 1 - br_player
        legal_actions_set = game_state.get_legal_actions()
        legal_actions = sorted(list(legal_actions_set), key=repr)
        num_actions = len(legal_actions)

        if num_actions == 0:
            logger.warning(
                "BR Calc: No legal actions at depth %d, non-terminal state %s. Returning 0.",
                depth,
                game_state,
            )
            return 0.0

        # --- Determine Decision Context (same logic as CFR) ---
        current_context = DecisionContext.TERMINAL
        if game_state.snap_phase_active:
            current_context = DecisionContext.SNAP_DECISION
        elif game_state.pending_action:
            pending = game_state.pending_action
            if hasattr(pending, "use_ability"):
                current_context = DecisionContext.POST_DRAW
            elif "Select" in type(pending).__name__:
                current_context = DecisionContext.ABILITY_SELECT
            elif "Decision" in type(pending).__name__:
                current_context = DecisionContext.ABILITY_SELECT
            elif "Move" in type(pending).__name__:
                current_context = DecisionContext.SNAP_MOVE
            else:
                current_context = DecisionContext.START_TURN
        else:
            current_context = DecisionContext.START_TURN

        # --- Node Logic ---
        if acting_player == br_player:
            max_value = -float("inf")
            for action in legal_actions:
                state_delta, undo_info = game_state.apply_action(action)
                next_br_agent_state = br_agent_state.clone()
                obs_after_action = self._create_initial_observation(game_state)
                next_br_agent_state.update(obs_after_action)

                action_value = self._best_response_recursive(
                    game_state,
                    opponent_avg_strategy,
                    br_player,
                    next_br_agent_state,
                    depth + 1,
                )
                if undo_info:
                    undo_info()
                else:
                    logger.error(
                        "BR Calc: Missing undo info for BR player action %s at depth %d.",
                        action,
                        depth,
                    )
                    return -float("inf")
                max_value = max(max_value, action_value)
            return max_value
        else:
            try:
                base_infoset_tuple = br_agent_state.get_infoset_key()
                infoset_key = InfosetKey(*base_infoset_tuple, current_context.value)
            except Exception:
                logger.error(
                    "BR Calc: Error getting infoset key for opponent P%d at depth %d. BR State: %s",
                    acting_player,
                    depth,
                    br_agent_state,
                    exc_info=True,
                )
                return 0.0

            opponent_strategy = opponent_avg_strategy.get(infoset_key)
            if opponent_strategy is None or len(opponent_strategy) != num_actions:
                if opponent_strategy is not None:
                    logger.warning(
                        "BR Calc: Dim mismatch for Opp P%d avg strategy at %s. Have %d, need %d. Using uniform.",
                        acting_player,
                        infoset_key,
                        len(opponent_strategy),
                        num_actions,
                    )
                else:
                    logger.debug(
                        "BR Calc: Opp P%d avg strategy not found for infoset %s. Using uniform.",
                        acting_player,
                        infoset_key,
                    )
                opponent_strategy = (
                    np.ones(num_actions) / num_actions
                    if num_actions > 0
                    else np.array([])
                )

            expected_value = 0.0
            if num_actions > 0 and opponent_strategy.sum() > 1e-9:
                if not np.isclose(opponent_strategy.sum(), 1.0):
                    opponent_strategy = normalize_probabilities(opponent_strategy)

                for i, action in enumerate(legal_actions):
                    action_prob = opponent_strategy[i]
                    if action_prob < 1e-9:
                        continue

                    state_delta, undo_info = game_state.apply_action(action)
                    next_br_agent_state = br_agent_state.clone()
                    obs_after_action = self._create_initial_observation(game_state)
                    next_br_agent_state.update(obs_after_action)

                    recursive_value = self._best_response_recursive(
                        game_state,
                        opponent_avg_strategy,
                        br_player,
                        next_br_agent_state,
                        depth + 1,
                    )
                    if undo_info:
                        undo_info()
                    else:
                        logger.error(
                            "BR Calc: Missing undo info for Opponent action %s at depth %d.",
                            action,
                            depth,
                        )
                        return 0.0
                    expected_value += action_prob * recursive_value
            else:
                logger.warning(
                    "BR Calc: Opponent P%d has zero actions or zero strategy sum at infoset %s. Depth %d.",
                    acting_player,
                    infoset_key,
                    depth,
                )
                return 0.0

            return expected_value

    def _create_initial_observation(self, game_state: CambiaGameState) -> Any:
        """Helper to create a basic observation, e.g., after an action for BR agent update."""
        from .agent_state import AgentObservation  # Local import

        obs = AgentObservation(
            acting_player=game_state.get_acting_player(),
            action=None,
            discard_top_card=game_state.get_discard_top(),
            player_hand_sizes=[
                game_state.get_player_card_count(i) for i in range(NUM_PLAYERS)
            ],
            stockpile_size=game_state.get_stockpile_size(),
            drawn_card=None,
            peeked_cards=None,
            snap_results=game_state.snap_results_log,
            did_cambia_get_called=game_state.cambia_caller_id is not None,
            who_called_cambia=game_state.cambia_caller_id,
            is_game_over=game_state.is_terminal(),
            current_turn=game_state.get_turn_number(),
        )
        return obs

    def log_game_history(self, game_details: Dict[str, Any]):
        """Logs the details of a completed game simulation to a JSON Lines file."""
        if not self.log_file_path:
            return
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:

                def default_serializer(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(
                        obj,
                        (
                            np.int_,
                            np.intc,
                            np.intp,
                            np.int8,
                            np.int16,
                            np.int32,
                            np.int64,
                            np.uint8,
                            np.uint16,
                            np.uint32,
                            np.uint64,
                        ),
                    ):
                        return int(obj)
                    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
                        return {"real": obj.real, "imag": obj.imag}
                    elif isinstance(obj, (np.bool_)):
                        return bool(obj)
                    elif isinstance(obj, (np.void)):
                        return None
                    elif isinstance(obj, InfosetKey):
                        return obj.astuple()
                    try:
                        return str(obj)
                    except TypeError:
                        return repr(obj)

                json_record = json.dumps(game_details, default=default_serializer)
                f.write(json_record + "\n")
        except IOError as e:
            logger.error("Error writing game history to %s: %s", self.log_file_path, e)
        except TypeError as e:
            logger.error(
                "Error serializing game details to JSON: %s. Details (repr): %s",
                e,
                repr(game_details),
            )

    def format_game_details_for_log(
        self,
        game_state: CambiaGameState,
        iteration: int,
        initial_hands: Optional[List[List[CardObject]]] = None,
        action_sequence: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Formats the necessary details from a finished game state for logging."""
        final_hands_serializable, final_scores = [], []
        for i, player_state in enumerate(game_state.players):
            if hasattr(player_state, "hand") and isinstance(player_state.hand, list):
                hand_str = [_serialize_card(c) for c in player_state.hand]
                final_hands_serializable.append(hand_str)
                final_scores.append(sum(c.value for c in player_state.hand if c))
            else:
                final_hands_serializable.append(["ERROR"])
                final_scores.append(999)

        initial_hands_serializable = (
            [[_serialize_card(c) for c in hand] for hand in initial_hands]
            if initial_hands
            else None
        )

        action_sequence_serializable = []
        if action_sequence:
            for entry in action_sequence:
                serializable_entry = {}
                for key, value in entry.items():
                    if key == "action":
                        serializable_entry[key] = (
                            _serialize_action(value) if value else None
                        )
                    elif key == "infoset_key":
                        serializable_entry[key] = (
                            value.astuple() if isinstance(value, InfosetKey) else value
                        )
                    else:
                        serializable_entry[key] = value
                action_sequence_serializable.append(serializable_entry)

        return {
            "game_id": f"sim_{iteration}_{game_state.get_turn_number()}",
            "iteration": iteration,
            "player_ids": list(range(game_state.num_players)),
            "initial_hands": initial_hands_serializable,
            "action_sequence": action_sequence_serializable,
            "final_hands": final_hands_serializable,
            "final_scores": final_scores,
            "winner": game_state._winner,
            "final_utilities": game_state._utilities,
            "num_turns": game_state.get_turn_number(),
            "cambia_caller": game_state.cambia_caller_id,
            "house_rules": asdict(game_state.house_rules),
        }
