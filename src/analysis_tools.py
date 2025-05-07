"""src/analysis_tools.py"""

import logging
import json
import os
import copy
import time
from typing import Dict, Any, Optional
from dataclasses import asdict
import numpy as np

from .game.engine import CambiaGameState
from .card import Card
from .agent_state import AgentState, AgentObservation
from .constants import (
    GameAction,
    DecisionContext,
    NUM_PLAYERS,
    ActionDiscard,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingSwapDecision,
    ActionSnapOpponentMove,
)
from .config import Config
from .utils import InfosetKey, PolicyDict, normalize_probabilities
from .game.helpers import serialize_card

logger = logging.getLogger(__name__)


# Helper function for default serialization in JSON dump
def default_serializer(obj):
    """Default JSON serializer for objects not directly serializable."""
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
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.void)):
        return None
    if isinstance(obj, InfosetKey):
        return obj.astuple()
    if isinstance(obj, Card):
        return serialize_card(obj)
    # Handle GameAction NamedTuples
    if hasattr(obj, "_asdict") and callable(obj._asdict):
        # Serialize card objects within the action dict
        action_dict = obj._asdict()
        serialized_dict = {}
        for k, v in action_dict.items():
            serialized_dict[k] = default_serializer(
                v
            )  # Recursive call for nested objects/cards
        return {type(obj).__name__: serialized_dict}  # Include type name
    # Fallback for other types (like simple action classes)
    try:
        # If it's a class instance without _asdict, just use its name
        if hasattr(obj, "__class__") and not isinstance(obj, type):
            return obj.__class__.__name__
        return str(obj)
    except TypeError:
        return repr(obj)


class AnalysisTools:
    """Provides tools for analyzing CFR training progress and game history."""

    def __init__(
        self,
        config: Config,
        log_dir: Optional[str] = None,
        log_file_prefix: Optional[str] = None,
    ):
        self.config = config
        self.delta_log_file_path = None  # (Backlog 8) Path for detailed delta logs

        if log_dir and log_file_prefix:
            try:
                os.makedirs(log_dir, exist_ok=True)
                # Setup path for the new delta log file
                self.delta_log_file_path = os.path.join(
                    log_dir, f"{log_file_prefix}_game_deltas.jsonl"
                )
                logger.info(
                    "AnalysisTools: Detailed game delta log path: %s",
                    self.delta_log_file_path,
                )
            except OSError as e_mkdir:
                logger.error(
                    "AnalysisTools: Failed to create log directory '%s': %s",
                    log_dir,
                    e_mkdir,
                )
        else:
            logger.warning(
                "AnalysisTools: Log directory/prefix not provided. Detailed delta logging disabled."
            )

    def calculate_exploitability(
        self, average_strategy: PolicyDict, config: Config
    ) -> float:
        """Calculates the exploitability of the agent's average strategy."""
        if not average_strategy:
            logger.warning("Cannot calculate exploitability: Average strategy is empty.")
            return float("inf")

        exploitability = float("inf")  # Default to infinity
        try:
            logger.info("Calculating Best Response for Player 0...")
            br_value_p0 = self._compute_best_response_value(
                average_strategy, config, br_player=0
            )
            logger.info("Best Response value for Player 0: %.6f", br_value_p0)
            # Allow logs to flush - small delay helpful in parallel runs
            if (
                hasattr(config.cfr_training, "num_workers")
                and config.cfr_training.num_workers > 1
            ):
                time.sleep(0.1)

            logger.info("Calculating Best Response for Player 1...")
            br_value_p1 = self._compute_best_response_value(
                average_strategy, config, br_player=1
            )
            logger.info("Best Response value for Player 1: %.6f", br_value_p1)

            if br_value_p0 == float("inf") or br_value_p1 == float("inf"):
                logger.warning(
                    "Exploitability calculation resulted in infinity (BR failed?)."
                )
                exploitability = float("inf")
            else:
                exploitability = (br_value_p0 + br_value_p1) / 2.0
                logger.info("Calculated Exploitability: %.6f", exploitability)

        except Exception as e_exploit:
            logger.exception("Error during exploitability calculation: %s", e_exploit)
            exploitability = float("inf")  # Indicate error

        return exploitability

    def _compute_best_response_value(
        self, opponent_avg_strategy: PolicyDict, config: Config, br_player: int
    ) -> float:
        """Computes the value of the best response strategy against the opponent's fixed average strategy."""
        try:
            game_state = CambiaGameState(house_rules=config.cambia_rules)
            opponent_player = 1 - br_player

            # Create initial observation (needs method defined in this class now)
            initial_obs = self._create_observation_for_br(game_state, None, -1)
            if initial_obs is None:
                logger.error(
                    "BR Setup P%d: Failed to create initial observation.", br_player
                )
                return float("inf")  # Indicate failure

            # Initialize AgentStates for BR player and Opponent view
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

            opp_view_agent_state = AgentState(
                player_id=opponent_player,
                opponent_id=br_player,
                memory_level=config.agent_params.memory_level,
                time_decay_turns=config.agent_params.time_decay_turns,
                initial_hand_size=len(game_state.players[opponent_player].hand),
                config=config,
            )
            opp_view_agent_state.initialize(
                initial_obs,
                game_state.players[opponent_player].hand,
                game_state.players[opponent_player].initial_peek_indices,
            )

            logger.debug("Starting BR calculation for P%d...", br_player)
            value = self._best_response_recursive(
                game_state,
                opponent_avg_strategy,
                br_player,
                br_agent_state,
                opp_view_agent_state,
                depth=0,
            )
            return value
        except Exception as e_br_setup:
            logger.exception(
                "Error setting up Best Response calculation for P%d: %s",
                br_player,
                e_br_setup,
            )
            return float("inf")  # Indicate failure

    def _best_response_recursive(
        self,
        game_state: CambiaGameState,
        opponent_avg_strategy: PolicyDict,
        br_player: int,
        br_agent_state: AgentState,
        opp_view_agent_state: AgentState,  # Pass both
        depth: int,
    ) -> float:
        """Recursive function for Best Response calculation."""
        try:
            if game_state.is_terminal():
                return game_state.get_utility(br_player)

            acting_player = game_state.get_acting_player()
            if acting_player == -1:
                logger.error(
                    "BR Calc(D%d): Invalid acting player. State: %s", depth, game_state
                )
                return 0.0

            opponent_player = 1 - br_player
            legal_actions_set = game_state.get_legal_actions()
            # Sort for deterministic behavior, helpful for debugging
            legal_actions = sorted(list(legal_actions_set), key=repr)
            num_actions = len(legal_actions)

            if num_actions == 0:
                if not game_state.is_terminal():
                    logger.error(
                        "BR Calc(D%d): No legal actions but non-terminal! State: %s",
                        depth,
                        game_state,
                    )
                return game_state.get_utility(br_player)  # Return current utility

            current_context = self._get_decision_context(game_state)
            if current_context is None:
                logger.error(
                    "BR Calc(D%d): Could not determine decision context. State: %s",
                    depth,
                    game_state,
                )
                return 0.0  # Error case

            # --- Node Logic ---
            if acting_player == br_player:
                # Maximize value for BR player
                max_value = -float("inf")
                action_values = []  # Store values for debugging
                for action in legal_actions:
                    state_delta, undo_info = game_state.apply_action(action)
                    if not callable(undo_info):
                        logger.error(
                            "BR Calc(D%d): BR Action %s returned invalid undo. State:%s",
                            depth,
                            action,
                            game_state,
                        )
                        continue  # Skip this action path

                    obs_after_action = self._create_observation_for_br(
                        game_state, action, acting_player
                    )
                    if obs_after_action is None:
                        undo_info()
                        continue

                    next_br_agent_state = br_agent_state.clone()
                    next_opp_view_agent_state = opp_view_agent_state.clone()
                    try:
                        br_obs_filtered = self._filter_observation_for_br(
                            obs_after_action, br_player
                        )
                        next_br_agent_state.update(br_obs_filtered)
                        opp_obs_filtered = self._filter_observation_for_br(
                            obs_after_action, opponent_player
                        )
                        next_opp_view_agent_state.update(opp_obs_filtered)
                    except Exception as e_update:
                        logger.error(
                            "BR Calc(D%d): Error updating agent states after BR action %s: %s",
                            depth,
                            action,
                            e_update,
                            exc_info=True,
                        )
                        undo_info()
                        continue  # Skip this action path

                    action_value = self._best_response_recursive(
                        game_state,
                        opponent_avg_strategy,
                        br_player,
                        next_br_agent_state,
                        next_opp_view_agent_state,
                        depth + 1,
                    )
                    action_values.append(action_value)  # Store for debugging
                    undo_info()  # Restore state
                    max_value = max(max_value, action_value)

                # If no actions were successfully explored, return 0?
                if max_value == -float("inf"):
                    logger.warning(
                        "BR Calc(D%d): BR player P%d had no successful action paths. Returning 0.",
                        depth,
                        br_player,
                    )
                    return 0.0
                return max_value

            else:  # Opponent's turn
                # Use Opponent's view state to get the key (Backlog 10)
                try:
                    base_infoset_tuple = opp_view_agent_state.get_infoset_key()
                    if not isinstance(base_infoset_tuple, tuple):
                        raise TypeError("Infoset key not tuple")
                    infoset_key = InfosetKey(*base_infoset_tuple, current_context.value)
                except Exception as e_key:
                    logger.error(
                        "BR Calc(D%d): Error getting Opponent P%d infoset key: %s. OppView State: %s",
                        depth,
                        acting_player,
                        e_key,
                        opp_view_agent_state,
                        exc_info=True,
                    )
                    return 0.0  # Error case

                opponent_strategy = opponent_avg_strategy.get(infoset_key)
                strategy_was_missing = opponent_strategy is None
                dim_mismatch = False

                if opponent_strategy is None:
                    opponent_strategy = (
                        np.ones(num_actions) / num_actions
                        if num_actions > 0
                        else np.array([])
                    )
                elif len(opponent_strategy) != num_actions:
                    logger.warning(
                        "BR Calc(D%d): Dim mismatch Opp P%d strategy at OppView key %s. Have %d, need %d. Using uniform.",
                        depth,
                        acting_player,
                        infoset_key,
                        len(opponent_strategy),
                        num_actions,
                    )
                    opponent_strategy = (
                        np.ones(num_actions) / num_actions
                        if num_actions > 0
                        else np.array([])
                    )
                    dim_mismatch = True

                expected_value = 0.0
                strategy_sum = (
                    opponent_strategy.sum() if opponent_strategy is not None else 0.0
                )

                if num_actions > 0 and strategy_sum > 1e-9:
                    if not np.isclose(strategy_sum, 1.0):
                        if (
                            not strategy_was_missing and not dim_mismatch
                        ):  # Only warn if strategy was present but unnormalized
                            logger.warning(
                                "BR Calc(D%d): Normalizing opponent strategy key %s (Sum: %f)",
                                depth,
                                infoset_key,
                                strategy_sum,
                            )
                        opponent_strategy = normalize_probabilities(opponent_strategy)
                        if len(opponent_strategy) == 0 or not np.isclose(
                            opponent_strategy.sum(), 1.0
                        ):
                            logger.error(
                                "BR Calc(D%d): Failed to normalize opp strategy for %s. Using uniform.",
                                depth,
                                infoset_key,
                            )
                            opponent_strategy = (
                                np.ones(num_actions) / num_actions
                                if num_actions > 0
                                else np.array([])
                            )

                    for i, action in enumerate(legal_actions):
                        action_prob = opponent_strategy[i]
                        if action_prob < 1e-9:
                            continue

                        state_delta, undo_info = game_state.apply_action(action)
                        if not callable(undo_info):
                            logger.error(
                                "BR Calc(D%d): Opponent Action %s returned invalid undo. State:%s",
                                depth,
                                action,
                                game_state,
                            )
                            continue  # Assume 0 value for this path

                        obs_after_action = self._create_observation_for_br(
                            game_state, action, acting_player
                        )
                        if obs_after_action is None:
                            undo_info()
                            continue

                        next_br_agent_state = br_agent_state.clone()
                        next_opp_view_agent_state = opp_view_agent_state.clone()
                        try:
                            br_obs_filtered = self._filter_observation_for_br(
                                obs_after_action, br_player
                            )
                            next_br_agent_state.update(br_obs_filtered)
                            opp_obs_filtered = self._filter_observation_for_br(
                                obs_after_action, opponent_player
                            )
                            next_opp_view_agent_state.update(opp_obs_filtered)
                        except Exception as e_update:
                            logger.error(
                                "BR Calc(D%d): Error updating agent states after Opp action %s: %s",
                                depth,
                                action,
                                e_update,
                                exc_info=True,
                            )
                            undo_info()
                            continue  # Assume 0 value for this path

                        recursive_value = self._best_response_recursive(
                            game_state,
                            opponent_avg_strategy,
                            br_player,
                            next_br_agent_state,
                            next_opp_view_agent_state,
                            depth + 1,
                        )
                        undo_info()
                        expected_value += action_prob * recursive_value
                else:
                    # This path should only be reached if num_actions > 0 but strategy sum is ~0.
                    if num_actions > 0:
                        logger.warning(
                            "BR Calc(D%d): Opponent P%d zero strategy sum at OppView infoset %s.",
                            depth,
                            acting_player,
                            infoset_key,
                        )
                    # If opponent effectively cannot act according to strategy, BR player gets utility of current state?
                    # This interpretation seems reasonable for BR.
                    return game_state.get_utility(br_player)

                return expected_value

        except Exception as e_br_rec:
            logger.exception(
                "BR Calc(D%d): Unhandled error in recursion: %s. State: %s",
                depth,
                e_br_rec,
                game_state,
            )
            return 0.0  # Return neutral value on unhandled error

    def _get_decision_context(
        self, game_state: CambiaGameState
    ) -> Optional[DecisionContext]:
        """Helper to determine DecisionContext robustly."""
        try:
            if game_state.snap_phase_active:
                return DecisionContext.SNAP_DECISION
            pending = game_state.pending_action
            if pending:
                # Use isinstance for robust type checking
                if isinstance(pending, ActionDiscard):
                    return DecisionContext.POST_DRAW
                if isinstance(
                    pending,
                    (
                        ActionAbilityPeekOwnSelect,
                        ActionAbilityPeekOtherSelect,
                        ActionAbilityBlindSwapSelect,
                        ActionAbilityKingLookSelect,
                        ActionAbilityKingSwapDecision,
                    ),
                ):
                    return DecisionContext.ABILITY_SELECT
                if isinstance(pending, ActionSnapOpponentMove):
                    return DecisionContext.SNAP_MOVE
                logger.warning(
                    "BR Context: Unknown pending action type: %s", type(pending).__name__
                )
                return DecisionContext.START_TURN  # Fallback
            if game_state.is_terminal():
                return DecisionContext.TERMINAL
            return DecisionContext.START_TURN
        except AttributeError as e_attr:
            logger.error(
                "Error determining decision context due to missing attribute: %s", e_attr
            )
            return None
        except Exception as e_ctx:
            logger.error("Error determining decision context: %s", e_ctx, exc_info=True)
            return None

    # Fix: Define _create_observation_for_br within AnalysisTools
    def _create_observation_for_br(
        self,
        game_state: CambiaGameState,
        action: Optional[GameAction],
        acting_player: int,
    ) -> Optional[AgentObservation]:
        """Helper to create a basic observation for BR agent updates."""
        try:
            # This version doesn't need complex drawn_card/peeked_card logic specific to CFR worker
            obs = AgentObservation(
                acting_player=acting_player,
                action=action,
                discard_top_card=game_state.get_discard_top(),
                player_hand_sizes=[
                    game_state.get_player_card_count(i) for i in range(NUM_PLAYERS)
                ],
                stockpile_size=game_state.get_stockpile_size(),
                drawn_card=None,  # Not needed for BR state update based on public info
                peeked_cards=None,  # Not needed for BR state update based on public info
                snap_results=copy.deepcopy(
                    game_state.snap_results_log
                ),  # Send public snap results
                did_cambia_get_called=game_state.cambia_caller_id is not None,
                who_called_cambia=game_state.cambia_caller_id,
                is_game_over=game_state.is_terminal(),
                current_turn=game_state.get_turn_number(),
            )
            return obs
        except Exception as e_obs:
            logger.error("Error creating observation for BR: %s", e_obs, exc_info=True)
            return None

    # Fix: Define _filter_observation_for_br within AnalysisTools
    def _filter_observation_for_br(
        self, obs: AgentObservation, observer_id: int
    ) -> AgentObservation:
        """Filters observation for BR agent state updates (simpler than CFR worker)."""
        filtered_obs = copy.copy(obs)
        # BR state updates don't need masking of draws/peeks as it uses public info + own hand
        filtered_obs.drawn_card = None
        filtered_obs.peeked_cards = None
        return filtered_obs

    # --- Game History Logging (Backlog 8) ---

    def log_game_history(self, game_log_data: Dict[str, Any]):
        """Logs the detailed delta history of a completed game simulation."""
        if not self.delta_log_file_path:
            return

        try:
            # Basic validation of the input dictionary
            if not isinstance(game_log_data, dict) or "history" not in game_log_data:
                logger.warning("Attempted to log invalid game history data structure.")
                return
            if not game_log_data["history"]:  # Log if history is empty, but allow it
                logger.debug(
                    "Logging game history with empty delta list for iter %d, worker %d.",
                    game_log_data.get("iteration", -1),
                    game_log_data.get("worker_id", -1),
                )

            with open(self.delta_log_file_path, "a", encoding="utf-8") as f:
                # Use the defined default_serializer helper function
                json_record = json.dumps(game_log_data, default=default_serializer)
                f.write(json_record + "\n")
        except IOError as e_io:
            logger.error(
                "Error writing game delta history to %s: %s",
                self.delta_log_file_path,
                e_io,
            )
        except TypeError as e_type:
            logger.error("Error serializing game delta details to JSON: %s.", e_type)
            # Attempt to log problematic parts safely
            try:
                problematic_part = {
                    k: repr(v)[:200] for k, v in game_log_data.items() if k != "history"
                }  # Avoid large history list
                problematic_part["history_len"] = len(game_log_data.get("history", []))
                logger.debug("Problematic data (repr): %s", problematic_part)
            except Exception:
                pass  # Avoid errors during error logging

    def format_game_details_for_log(
        self, game_state: CambiaGameState, iteration: int, worker_id: int
    ) -> Dict[str, Any]:
        """Formats the metadata for a finished game state for delta logging."""
        # This function now only prepares metadata. The delta history list is added separately.
        final_hands_serializable, final_scores = [], []
        if hasattr(game_state, "players") and isinstance(game_state.players, list):
            for i, player_state in enumerate(game_state.players):
                if hasattr(player_state, "hand") and isinstance(player_state.hand, list):
                    hand = player_state.hand
                    hand_str = [serialize_card(c) for c in hand]
                    final_hands_serializable.append(hand_str)
                    try:
                        final_scores.append(
                            sum(c.value for c in hand if isinstance(c, Card))
                        )
                    except Exception:
                        final_scores.append(999)
                else:
                    final_hands_serializable.append(["ERROR"])
                    final_scores.append(999)
        else:
            final_hands_serializable = [["ERROR"]] * NUM_PLAYERS
            final_scores = [999] * NUM_PLAYERS

        return {
            "game_log_version": 1,
            "game_id": f"sim_{iteration}_{worker_id}",
            "iteration": iteration,
            "worker_id": worker_id,
            "final_state": {
                "player_ids": list(range(NUM_PLAYERS)),  # Use constant
                "final_hands": final_hands_serializable,
                "final_scores": final_scores,
                "winner": getattr(game_state, "_winner", None),
                "final_utilities": getattr(game_state, "_utilities", []),
                "num_turns": game_state.get_turn_number(),
                "cambia_caller": game_state.cambia_caller_id,
                "house_rules": (
                    asdict(game_state.house_rules)
                    if hasattr(game_state, "house_rules")
                    else {}
                ),
            },
            # "history" field (list of action/delta tuples) is added externally
        }
