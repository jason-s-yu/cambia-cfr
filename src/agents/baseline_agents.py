"""Implements simple baseline agents for evaluation purposes."""

import random
import logging
from abc import ABC, abstractmethod
from typing import Set, Optional, List

from ..game.engine import CambiaGameState
from ..constants import (
    ActionDrawStockpile,
    ActionSnapOpponentMove,
    GameAction,
    CardObject,
    ActionCallCambia,
    ActionDiscard,
    ActionReplace,
    ActionSnapOwn,
    ActionSnapOpponent,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    KING,
    SEVEN,
    EIGHT,
    NINE,
    TEN,
)
from ..config import Config
from ..card import Card

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for Cambia agents."""

    player_id: int
    opponent_id: int
    config: Config

    def __init__(self, player_id: int, config: Config):
        self.player_id = player_id
        self.opponent_id = 1 - player_id  # Assuming 2 players
        self.config = config

    @abstractmethod
    def choose_action(
        self, game_state: CambiaGameState, legal_actions: Set[GameAction]
    ) -> GameAction:
        """Selects an action based on the current game state and legal actions."""
        pass


class RandomAgent(BaseAgent):
    """An agent that chooses actions randomly from the legal set."""

    def choose_action(
        self, game_state: CambiaGameState, legal_actions: Set[GameAction]
    ) -> GameAction:
        """Chooses a random action."""
        if not legal_actions:
            # This should ideally not happen if called correctly, game engine handles terminal/no-action states
            logger.error(
                "RandomAgent P%d received empty legal actions set in non-terminal state?",
                self.player_id,
            )
            # What action to return? Raise error? For now, let's try returning a dummy/invalid action
            # Or perhaps return a default safe action if possible, like PassSnap? Difficult to generalize.
            # Let's raise an error to highlight the issue upstream.
            raise ValueError(
                f"RandomAgent P{self.player_id} cannot choose from empty legal actions."
            )

        action_list = list(legal_actions)
        chosen_action = random.choice(action_list)
        logger.debug("RandomAgent P%d chose action: %s", self.player_id, chosen_action)
        return chosen_action


class GreedyAgent(BaseAgent):
    """
    A simple rule-based greedy agent.
    Assumes perfect information (direct access to game_state) for decision making.
    Uses configurable parameters for some decisions.
    """

    def __init__(self, player_id: int, config: Config):
        super().__init__(player_id, config)
        # Get greedy agent specific config
        self.cambia_threshold = config.agents.greedy_agent.cambia_call_threshold
        logger.info(
            "GreedyAgent P%d initialized (Cambia Threshold: %d)",
            self.player_id,
            self.cambia_threshold,
        )
        # Note: This agent currently uses memory_level 0 logic implicitly by accessing true state.

    def _get_hand_value(self, hand: List[CardObject]) -> int:
        """Calculates the point value of a hand (perfect info)."""
        value = 0
        for card in hand:
            if isinstance(card, Card):
                value += card.value
            else:
                value += 99  # Penalize non-card items heavily
        return value

    def choose_action(
        self, game_state: CambiaGameState, legal_actions: Set[GameAction]
    ) -> GameAction:
        """Chooses an action based on greedy heuristics."""
        if not legal_actions:
            raise ValueError(
                f"GreedyAgent P{self.player_id} cannot choose from empty legal actions."
            )

        my_hand = game_state.get_player_hand(self.player_id)
        opp_hand = game_state.get_player_hand(
            self.opponent_id
        )  # Needs opponent's hand for some decisions

        # --- Rule Priorities ---

        # 1. Call Cambia if possible and hand value is low enough
        if ActionCallCambia() in legal_actions:
            current_value = self._get_hand_value(my_hand)
            if current_value <= self.cambia_threshold:
                logger.debug(
                    "GreedyAgent P%d calling Cambia (Value: %d <= Threshold: %d)",
                    self.player_id,
                    current_value,
                    self.cambia_threshold,
                )
                return ActionCallCambia()

        # 2. Handle Snapping: Always snap if possible (prefer own)
        snap_own_actions = {a for a in legal_actions if isinstance(a, ActionSnapOwn)}
        if snap_own_actions:
            chosen_action = min(
                snap_own_actions, key=lambda a: a.own_card_hand_index
            )  # Snap lowest index match
            logger.debug(
                "GreedyAgent P%d chose action (SnapOwn): %s",
                self.player_id,
                chosen_action,
            )
            return chosen_action
        snap_opp_actions = {a for a in legal_actions if isinstance(a, ActionSnapOpponent)}
        if snap_opp_actions:
            # Greedy needs perfect info to know *which* opponent card to snap.
            # Find the first valid snap opponent action based on true opponent hand.
            snap_card = game_state.snap_discarded_card
            if snap_card:
                target_rank = snap_card.rank
                for action in sorted(
                    list(snap_opp_actions), key=lambda a: a.opponent_target_hand_index
                ):
                    opp_idx_target = action.opponent_target_hand_index
                    if (
                        0 <= opp_idx_target < len(opp_hand)
                        and isinstance(opp_hand[opp_idx_target], Card)
                        and opp_hand[opp_idx_target].rank == target_rank
                    ):
                        logger.debug(
                            "GreedyAgent P%d chose action (SnapOpponent): %s",
                            self.player_id,
                            action,
                        )
                        return action
                logger.warning(
                    "GreedyAgent P%d found legal SnapOpponent actions but none matched opponent hand?",
                    self.player_id,
                )
            # Fall through if cannot confirm opponent card

        # 3. Handle Post-Draw Choice (Discard/Replace)
        if any(isinstance(a, (ActionDiscard, ActionReplace)) for a in legal_actions):
            drawn_card = game_state.pending_action_data.get("drawn_card")
            if not drawn_card or not isinstance(drawn_card, Card):
                logger.error(
                    "GreedyAgent P%d in PostDraw state but drawn_card invalid: %s",
                    self.player_id,
                    drawn_card,
                )
                # Fallback: Just discard without ability
                return (
                    ActionDiscard(use_ability=False)
                    if ActionDiscard(use_ability=False) in legal_actions
                    else list(legal_actions)[0]
                )

            best_replace_action: Optional[ActionReplace] = None
            max_value_reduction = -1  # Aim for lowest value hand

            # Evaluate potential replacements
            for i, current_card in enumerate(my_hand):
                if isinstance(current_card, Card):
                    value_if_replaced = self._get_hand_value(
                        my_hand[:i] + [drawn_card] + my_hand[i + 1 :]
                    )
                    current_hand_value = self._get_hand_value(my_hand)
                    reduction = current_hand_value - value_if_replaced
                    # Strictly better replacement based on known value
                    if (
                        drawn_card.value < current_card.value
                        and reduction > max_value_reduction
                    ):
                        max_value_reduction = reduction
                        best_replace_action = ActionReplace(target_hand_index=i)

            # Rule: Replace unknown if drawn card is low enough (at or below threshold)
            if best_replace_action is None and drawn_card.value <= self.cambia_threshold:
                # Find first 'unknown' card - greedy assumes perfect memory, so this rule isn't applicable directly
                # Modify: Replace highest value card if drawn card <= threshold
                highest_value = -float("inf")
                replace_idx = -1
                for i, card in enumerate(my_hand):
                    if isinstance(card, Card) and card.value > highest_value:
                        highest_value = card.value
                        replace_idx = i
                if (
                    replace_idx != -1 and drawn_card.value <= highest_value
                ):  # Only replace if drawn is <= highest
                    best_replace_action = ActionReplace(target_hand_index=replace_idx)
                    logger.debug(
                        "GreedyAgent P%d replacing highest value card %d (%d) with low drawn card %s (%d)",
                        self.player_id,
                        replace_idx,
                        highest_value,
                        drawn_card,
                        drawn_card.value,
                    )

            # If a good replacement exists, take it
            if best_replace_action and best_replace_action in legal_actions:
                logger.debug(
                    "GreedyAgent P%d chose action (Replace): %s",
                    self.player_id,
                    best_replace_action,
                )
                return best_replace_action

            # Otherwise, consider discarding with ability if useful
            can_discard_ability = ActionDiscard(use_ability=True) in legal_actions
            is_utility_card = drawn_card.rank in [
                SEVEN,
                EIGHT,
                NINE,
                TEN,
                KING,
            ]  # 7,8,9,T,K give knowledge/control
            if can_discard_ability and is_utility_card:
                logger.debug(
                    "GreedyAgent P%d chose action (Discard Utility): %s",
                    self.player_id,
                    ActionDiscard(use_ability=True),
                )
                return ActionDiscard(use_ability=True)

            # Default: Simple discard
            logger.debug(
                "GreedyAgent P%d chose action (Default Discard): %s",
                self.player_id,
                ActionDiscard(use_ability=False),
            )
            return ActionDiscard(use_ability=False)

        # 4. Handle Ability Choices
        if isinstance(next(iter(legal_actions), None), ActionAbilityPeekOwnSelect):  # 7/8
            # Peek first unknown card (not implemented as agent has perfect info)
            # Simple: Peek lowest index card
            action = ActionAbilityPeekOwnSelect(target_hand_index=0)
            logger.debug(
                "GreedyAgent P%d chose action (PeekOwn): %s", self.player_id, action
            )
            return action

        if isinstance(
            next(iter(legal_actions), None), ActionAbilityPeekOtherSelect
        ):  # 9/T
            # Peek opponent lowest index
            action = ActionAbilityPeekOtherSelect(target_opponent_hand_index=0)
            logger.debug(
                "GreedyAgent P%d chose action (PeekOther): %s", self.player_id, action
            )
            return action

        if isinstance(
            next(iter(legal_actions), None), ActionAbilityBlindSwapSelect
        ):  # J/Q
            # Rule: Ignore J/Q. How to implement "ignore"? Choose a default non-swap action if possible.
            # This state should only be reachable if discard+ability was forced.
            # Choose the lowest index swap (0,0) as a default if forced.
            action = ActionAbilityBlindSwapSelect(own_hand_index=0, opponent_hand_index=0)
            if action in legal_actions:
                logger.debug(
                    "GreedyAgent P%d chose action (Forced BlindSwap): %s",
                    self.player_id,
                    action,
                )
                return action
            else:  # Should not happen if legal_actions contained only BlindSwap options
                logger.error(
                    "GreedyAgent P%d forced into BlindSwap, but (0,0) invalid?",
                    self.player_id,
                )
                return list(legal_actions)[0]  # Failsafe

        if isinstance(
            next(iter(legal_actions), None), ActionAbilityKingLookSelect
        ):  # King Look
            # Rule: Prioritize own hand knowledge? Not applicable with perfect info.
            # Simple: Look at lowest indices
            action = ActionAbilityKingLookSelect(own_hand_index=0, opponent_hand_index=0)
            logger.debug(
                "GreedyAgent P%d chose action (KingLook): %s", self.player_id, action
            )
            return action

        if isinstance(
            next(iter(legal_actions), None), ActionAbilityKingSwapDecision
        ):  # King Swap
            # Rule: Swap only if it reduces own hand value
            look_data = game_state.pending_action_data
            card1 = look_data.get("card1")  # Own card peeked
            card2 = look_data.get("card2")  # Opp card peeked
            if isinstance(card1, Card) and isinstance(card2, Card):
                if (
                    card2.value < card1.value
                ):  # Swap if opponent card is better (lower value)
                    action = ActionAbilityKingSwapDecision(perform_swap=True)
                    logger.debug(
                        "GreedyAgent P%d chose action (KingSwap=True): %s",
                        self.player_id,
                        action,
                    )
                    return action
            # Default: Don't swap
            action = ActionAbilityKingSwapDecision(perform_swap=False)
            logger.debug(
                "GreedyAgent P%d chose action (KingSwap=False): %s",
                self.player_id,
                action,
            )
            return action

        if isinstance(next(iter(legal_actions), None), ActionSnapOpponentMove):
            # Move lowest value card from own hand to opponent's empty slot
            best_card_idx = -1
            lowest_value = float("inf")
            for i, card in enumerate(my_hand):
                if isinstance(card, Card) and card.value < lowest_value:
                    lowest_value = card.value
                    best_card_idx = i

            if best_card_idx != -1:
                # Get target slot from the first legal action (it's the same for all)
                example_action = next(iter(legal_actions))
                target_slot = example_action.target_empty_slot_index
                chosen_action = ActionSnapOpponentMove(
                    own_card_to_move_hand_index=best_card_idx,
                    target_empty_slot_index=target_slot,
                )
                if chosen_action in legal_actions:
                    logger.debug(
                        "GreedyAgent P%d chose action (SnapMove): %s",
                        self.player_id,
                        chosen_action,
                    )
                    return chosen_action
                else:
                    logger.error(
                        "GreedyAgent P%d SnapMove calculation error?", self.player_id
                    )

            # Fallback if error or no cards
            logger.warning("GreedyAgent P%d fallback SnapMove.", self.player_id)
            return list(legal_actions)[0]

        # 5. Default/Fallback: Choose Draw Stockpile if available, else first legal action
        if ActionDrawStockpile() in legal_actions:
            action = ActionDrawStockpile()
            logger.debug(
                "GreedyAgent P%d chose action (Default DrawStock): %s",
                self.player_id,
                action,
            )
            return action

        # Should only reach here if DrawStockpile not legal (e.g., empty + no reshuffle)
        # Or if some state wasn't handled above.
        chosen_action = list(legal_actions)[0]  # Failsafe: choose first available
        logger.warning(
            "GreedyAgent P%d falling back to first legal action: %s",
            self.player_id,
            chosen_action,
        )
        return chosen_action
