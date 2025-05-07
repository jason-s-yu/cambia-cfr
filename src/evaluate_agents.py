"""Script to evaluate different Cambia agents against each other."""

import argparse
import logging
import sys
import random
import time
import copy
from typing import Optional, Dict, Type, Set
from collections import Counter
import numpy as np
from tqdm import tqdm

from src.config import load_config, Config
from src.game.engine import CambiaGameState
from src.agents.baseline_agents import BaseAgent, RandomAgent, GreedyAgent
from src.agent_state import AgentState, AgentObservation
from src.cfr.trainer import CFRTrainer
from src.utils import (
    InfosetKey,
    normalize_probabilities,
)

from src.constants import (
    NUM_PLAYERS,
    GameAction,
    DecisionContext,
    ActionDiscard,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionSnapOpponentMove,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- CFR Agent Wrapper ---


class CFRAgentWrapper(BaseAgent):
    """Wraps a computed average strategy for use in evaluation."""

    def __init__(
        self,
        player_id: int,
        config: Config,
        average_strategy: Dict[InfosetKey, np.ndarray],
    ):
        super().__init__(player_id, config)
        if not isinstance(average_strategy, dict):
            raise TypeError(
                "CFRAgentWrapper requires average_strategy to be a dictionary."
            )
        self.average_strategy = average_strategy
        self.agent_state: Optional[AgentState] = None  # Internal state

    def initialize_state(self, initial_game_state: CambiaGameState):
        """Initialize the internal AgentState."""
        # FIX: Call internal observation creation method
        initial_obs = self._create_observation(initial_game_state, None, -1)
        if initial_obs is None:
            raise RuntimeError(
                f"CFRAgent P{self.player_id} failed to create initial observation."
            )

        self.agent_state = AgentState(
            player_id=self.player_id,
            opponent_id=self.opponent_id,
            memory_level=self.config.agent_params.memory_level,
            time_decay_turns=self.config.agent_params.time_decay_turns,
            initial_hand_size=len(initial_game_state.players[self.player_id].hand),
            config=self.config,
        )
        # Ensure initial hands/peeks are valid before passing
        initial_hand = initial_game_state.players[self.player_id].hand
        initial_peeks = initial_game_state.players[self.player_id].initial_peek_indices
        if not isinstance(initial_hand, list) or not isinstance(initial_peeks, tuple):
            raise TypeError(
                f"CFRAgent P{self.player_id}: Invalid initial hand/peek data."
            )

        self.agent_state.initialize(initial_obs, initial_hand, initial_peeks)
        logger.debug("CFRAgent P%d initialized state.", self.player_id)

    def update_state(self, observation: AgentObservation):
        """Update internal state based on observation."""
        if self.agent_state:
            # FIX: Call internal filtering method
            filtered_obs = self._filter_observation(observation, self.player_id)
            try:
                self.agent_state.update(filtered_obs)
            except Exception as e_update:
                logger.error(
                    "CFRAgent P%d failed to update state: %s. Obs: %s",
                    self.player_id,
                    e_update,
                    filtered_obs,
                    exc_info=True,
                )
                # Decide how to handle state update failure - continue with old state? Raise?
                # For now, log and continue.
        else:
            logger.error(
                "CFRAgent P%d cannot update state, not initialized.", self.player_id
            )

    def choose_action(
        self, game_state: CambiaGameState, legal_actions: Set[GameAction]
    ) -> GameAction:
        """Chooses an action based on the learned average strategy."""
        if not self.agent_state:
            raise RuntimeError(
                f"CFRAgent P{self.player_id} state not initialized before choose_action."
            )
        if not legal_actions:
            raise ValueError(
                f"CFRAgent P{self.player_id} cannot choose from empty legal actions."
            )

        # Determine context (copied from analysis_tools/worker)
        if game_state.snap_phase_active:
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
                current_context = DecisionContext.START_TURN  # Fallback
        else:
            current_context = DecisionContext.START_TURN

        try:
            base_infoset_tuple = self.agent_state.get_infoset_key()
            # Ensure base_infoset_tuple is actually a tuple before unpacking
            if not isinstance(base_infoset_tuple, tuple):
                raise TypeError(
                    f"get_infoset_key returned {type(base_infoset_tuple).__name__}, expected tuple"
                )
            infoset_key = InfosetKey(*base_infoset_tuple, current_context.value)
        except Exception as e_key:
            logger.error(
                "CFRAgent P%d Error getting infoset key: %s. State: %s",
                self.player_id,
                e_key,
                self.agent_state,
                exc_info=True,
            )
            return random.choice(list(legal_actions))

        strategy = self.average_strategy.get(infoset_key)
        action_list = sorted(list(legal_actions), key=repr)
        num_actions = len(action_list)

        # Handle missing strategy or dimension mismatch
        if strategy is None or len(strategy) != num_actions:
            if strategy is not None:
                logger.warning(
                    "CFRAgent P%d strategy dim mismatch key %s (Have %d, Need %d). Using uniform.",
                    self.player_id,
                    infoset_key,
                    len(strategy),
                    num_actions,
                )
            # else: logger.debug("CFRAgent P%d strategy not found for key %s. Using uniform.", self.player_id, infoset_key) # Reduce noise
            strategy = (
                np.ones(num_actions) / num_actions if num_actions > 0 else np.array([])
            )

        # Handle zero-length strategy (should only happen if num_actions is 0, caught earlier)
        if len(strategy) == 0:
            logger.error(
                "CFRAgent P%d: Zero strategy length despite %d legal actions exist.",
                self.player_id,
                num_actions,
            )
            return random.choice(action_list)  # Fallback

        # Normalize strategy if needed (defensive programming)
        strategy_sum = np.sum(strategy)
        if not np.isclose(strategy_sum, 1.0):
            logger.warning(
                "CFRAgent P%d: Normalizing strategy for key %s (Sum: %f)",
                self.player_id,
                infoset_key,
                strategy_sum,
            )
            strategy = normalize_probabilities(strategy)
            if len(strategy) == 0 or not np.isclose(
                np.sum(strategy), 1.0
            ):  # Check normalization result
                logger.error(
                    "CFRAgent P%d: Failed to normalize strategy for %s. Using uniform.",
                    self.player_id,
                    infoset_key,
                )
                strategy = (
                    np.ones(num_actions) / num_actions
                    if num_actions > 0
                    else np.array([])
                )

        # Sample action
        try:
            chosen_index = np.random.choice(num_actions, p=strategy)
            chosen_action = action_list[chosen_index]
        except (
            ValueError
        ) as e_choice:  # Catch errors from np.random.choice (e.g., probabilities don't sum to 1)
            logger.error(
                "CFRAgent P%d: Error choosing action for key %s (strategy %s): %s. Choosing random.",
                self.player_id,
                infoset_key,
                strategy,
                e_choice,
            )
            chosen_action = random.choice(action_list)  # Fallback

        # logger.debug("CFRAgent P%d chose action: %s (Prob: %.3f, Key: %s)", self.player_id, chosen_action, strategy[chosen_index], infoset_key)
        return chosen_action

    # --- Observation helpers moved into CFRAgentWrapper ---
    def _create_observation(
        self,
        game_state: CambiaGameState,
        action: Optional[GameAction],
        acting_player: int,
    ) -> Optional[AgentObservation]:
        """Creates observation needed *by this agent* after an action."""
        try:
            # Simplified for evaluation: Assume agent state uses public info + own known cards
            obs = AgentObservation(
                acting_player=acting_player,
                action=action,
                discard_top_card=game_state.get_discard_top(),
                player_hand_sizes=[
                    game_state.get_player_card_count(i) for i in range(NUM_PLAYERS)
                ],
                stockpile_size=game_state.get_stockpile_size(),
                drawn_card=None,  # Don't pass private draw info during evaluation obs
                peeked_cards=None,  # Don't pass private peek info during evaluation obs
                snap_results=copy.deepcopy(game_state.snap_results_log),  # Public
                did_cambia_get_called=game_state.cambia_caller_id is not None,
                who_called_cambia=game_state.cambia_caller_id,
                is_game_over=game_state.is_terminal(),
                current_turn=game_state.get_turn_number(),
            )
            return obs
        except Exception as e_obs:
            logger.error(
                "CFRAgent P%d: Error creating observation: %s",
                self.player_id,
                e_obs,
                exc_info=True,
            )
            return None

    def _filter_observation(
        self, obs: AgentObservation, observer_id: int
    ) -> AgentObservation:
        """Filters observation for the agent's own perspective (minimal filtering needed here)."""
        # Since _create_observation doesn't include sensitive info, filtering is simpler
        filtered_obs = copy.copy(obs)
        # Ensure fields intended to be private for updates are None
        filtered_obs.drawn_card = None
        filtered_obs.peeked_cards = None
        return filtered_obs


# --- Agent Factory ---

AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "cfr": CFRAgentWrapper,
}


def get_agent(agent_type: str, player_id: int, config: Config, **kwargs) -> BaseAgent:
    """Instantiates an agent based on its type."""
    agent_class = AGENT_REGISTRY.get(agent_type.lower())
    if not agent_class:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Available: {list(AGENT_REGISTRY.keys())}"
        )

    if agent_type.lower() == "cfr":
        avg_strategy = kwargs.get("average_strategy")
        if not avg_strategy or not isinstance(avg_strategy, dict):  # Check type
            raise ValueError("CFRAgent requires 'average_strategy' dictionary.")
        return CFRAgentWrapper(player_id, config, avg_strategy)
    else:
        # Pass config to baseline agents as well
        return agent_class(player_id, config)


# --- Evaluation Loop ---


def run_evaluation(
    config_path: str,
    agent1_type: str,
    agent2_type: str,
    num_games: int,
    strategy_path: Optional[str],
):
    """Runs head-to-head evaluation between two agents."""
    logger.info("--- Starting Agent Evaluation ---")
    logger.info("Config: %s", config_path)
    logger.info("Agent 1 (P0): %s", agent1_type.upper())
    logger.info("Agent 2 (P1): %s", agent2_type.upper())
    logger.info("Number of Games: %d", num_games)
    if agent1_type.lower() == "cfr" or agent2_type.lower() == "cfr":
        if not strategy_path:
            logger.error("Strategy file path (--strategy) required for CFR agent.")
            sys.exit(1)
        logger.info("CFR Strategy File: %s", strategy_path)

    config = load_config(config_path)
    if not config:
        logger.error("Failed to load configuration from %s", config_path)
        sys.exit(1)

    average_strategy = None
    if agent1_type.lower() == "cfr" or agent2_type.lower() == "cfr":
        logger.info("Loading CFR agent data from %s...", strategy_path)
        # Use the CFRTrainer temporarily just for loading/computing strategy
        try:
            # Minimal init, avoids needing full trainer setup dependencies if possible
            temp_trainer = CFRTrainer(config=config)
            temp_trainer.load_data(strategy_path)  # Use load_data method
            logger.info("Computing average strategy...")
            average_strategy = temp_trainer.compute_average_strategy()
            if average_strategy is None:
                logger.error("Failed to compute average strategy from loaded data.")
                sys.exit(1)
            logger.info("Average strategy computed (%d infosets).", len(average_strategy))
        except Exception as e_load:
            logger.exception("Failed to load or process CFR strategy: %s", e_load)
            sys.exit(1)

    try:
        agent1_kwargs = (
            {"average_strategy": average_strategy} if agent1_type.lower() == "cfr" else {}
        )
        agent2_kwargs = (
            {"average_strategy": average_strategy} if agent2_type.lower() == "cfr" else {}
        )
        agent1 = get_agent(agent1_type, player_id=0, config=config, **agent1_kwargs)
        agent2 = get_agent(agent2_type, player_id=1, config=config, **agent2_kwargs)
        agents = [agent1, agent2]
        logger.info("Agents instantiated.")
    except ValueError as e:
        logger.error("Error creating agents: %s", e)
        sys.exit(1)

    results = Counter()
    start_time = time.time()

    for game_num in tqdm(range(1, num_games + 1), desc="Simulating Games", unit="game"):
        try:
            game_state = CambiaGameState(house_rules=config.cambia_rules)
            # Initialize CFR agent states
            for agent in agents:
                if isinstance(agent, CFRAgentWrapper):
                    agent.initialize_state(game_state)

            turn = 0
            max_turns = (
                config.cambia_rules.max_game_turns
                if config.cambia_rules.max_game_turns > 0
                else 500
            )  # Add a default max turn if 0

            while not game_state.is_terminal() and turn < max_turns:
                turn += 1
                acting_player_id = game_state.get_acting_player()
                if acting_player_id == -1:
                    logger.error(
                        "Game %d Turn %d: Invalid acting player (-1). State: %s",
                        game_num,
                        turn,
                        game_state,
                    )
                    results["Errors"] += 1
                    break

                current_agent = agents[acting_player_id]
                try:
                    legal_actions = game_state.get_legal_actions()
                    if not legal_actions:
                        # Check again if terminal, might have become terminal after last action
                        if game_state.is_terminal():
                            break
                        logger.error(
                            "Game %d Turn %d: No legal actions but non-terminal? State: %s",
                            game_num,
                            turn,
                            game_state,
                        )
                        results["Errors"] += 1
                        break

                    # Choose action
                    chosen_action = current_agent.choose_action(game_state, legal_actions)

                    # Apply action
                    state_delta, undo_info = game_state.apply_action(chosen_action)
                    if not callable(
                        undo_info
                    ):  # Should not happen if apply_action succeeds
                        logger.error(
                            "Game %d Turn %d: Action %s applied but returned invalid undo info.",
                            game_num,
                            turn,
                            chosen_action,
                        )
                        results["Errors"] += 1
                        break

                    # Create observation AFTER action
                    observation = None
                    if isinstance(agent1, CFRAgentWrapper) or isinstance(
                        agent2, CFRAgentWrapper
                    ):
                        if hasattr(
                            current_agent, "_create_observation"
                        ):  # Check if method exists
                            observation = current_agent._create_observation(
                                game_state, chosen_action, acting_player_id
                            )

                    # Update agent states (only CFR agents need it)
                    if observation:
                        for agent_idx, agent in enumerate(agents):
                            if isinstance(agent, CFRAgentWrapper):
                                agent.update_state(observation)  # Uses internal filtering

                except Exception as e_turn:
                    logger.exception(
                        "Error during game %d turn %d for P%d: %s. State: %s",
                        game_num,
                        turn,
                        acting_player_id,
                        e_turn,
                        game_state,
                    )
                    results["Errors"] += 1
                    break  # End game on error

            # Game End
            if game_state.is_terminal():
                winner = game_state._winner
                if winner == 0:
                    results["P0 Wins"] += 1
                elif winner == 1:
                    results["P1 Wins"] += 1
                else:
                    results["Ties"] += 1
            elif turn >= max_turns:
                logger.debug(
                    "Game %d reached max turns (%d). Scoring as tie.", game_num, max_turns
                )
                results["MaxTurnTies"] += 1

        except Exception as e_game_loop:
            logger.exception(
                "Critical error during game simulation %d setup or loop: %s",
                game_num,
                e_game_loop,
            )
            results["Errors"] += 1

    end_time = time.time()
    total_time = end_time - start_time
    games_played = sum(results.values())

    # Report Results
    logger.info("--- Evaluation Results ---")
    logger.info("Agents: P0 = %s, P1 = %s", agent1_type.upper(), agent2_type.upper())
    logger.info("Games Simulated: %d", num_games)
    logger.info("Valid Games Completed: %d", games_played)
    logger.info("Total Time: %.2f seconds", total_time)
    if games_played > 0:
        logger.info("Time per Game: %.3f seconds", total_time / games_played)

    p0_wins = results.get("P0 Wins", 0)
    p1_wins = results.get("P1 Wins", 0)
    ties = results.get("Ties", 0)
    max_turn_ties = results.get("MaxTurnTies", 0)
    errors = results.get("Errors", 0)
    total_scored = p0_wins + p1_wins + ties + max_turn_ties  # Denominator for percentages

    logger.info(
        "Score: P0 Wins=%d (%.2f%%), P1 Wins=%d (%.2f%%), Ties=%d (%.2f%%), MaxTurnTies=%d, Errors=%d",
        p0_wins,
        (p0_wins / total_scored * 100) if total_scored else 0,
        p1_wins,
        (p1_wins / total_scored * 100) if total_scored else 0,
        ties,
        (ties / total_scored * 100) if total_scored else 0,
        max_turn_ties,
        errors,
    )
    logger.info("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cambia Agents Head-to-Head")
    parser.add_argument(
        "agent1", help="Type of Agent 1 (P0)", choices=list(AGENT_REGISTRY.keys())
    )
    parser.add_argument(
        "agent2", help="Type of Agent 2 (P1)", choices=list(AGENT_REGISTRY.keys())
    )
    parser.add_argument(
        "-n", "--num_games", type=int, default=100, help="Number of games to simulate"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default=None,
        help="Path to saved CFR agent strategy file (required if agent type is 'cfr')",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging for evaluation script",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logging.getLogger("src.game.engine").setLevel(logging.INFO)
        logging.getLogger("src.agents.baseline_agents").setLevel(logging.DEBUG)
        logging.getLogger("src.agent_state").setLevel(
            logging.INFO
        )  # Keep agent state less verbose unless debugging it
    else:
        # Silence logs below INFO from libraries if not verbose
        logging.getLogger("src.game.engine").setLevel(logging.WARNING)
        logging.getLogger("src.agents.baseline_agents").setLevel(logging.INFO)
        logging.getLogger("src.agent_state").setLevel(logging.WARNING)

    run_evaluation(
        config_path=args.config,
        agent1_type=args.agent1,
        agent2_type=args.agent2,
        num_games=args.num_games,
        strategy_path=args.strategy,
    )
