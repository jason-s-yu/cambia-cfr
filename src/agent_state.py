"""
src/agent_state.py

Represents an agent's subjective belief state in the game Cambia.
Handles information abstraction, memory limitations, and state updates based on observations.
"""

from typing import List, Tuple, Optional, Dict, Any, Set, Union, Type
from dataclasses import dataclass, field
import logging
import copy

from .card import Card
from .constants import (
    ActionSnapOpponentMove,
    CardBucket,
    DecayCategory,
    GamePhase,
    StockpileEstimate,
    GameAction,
    ActionReplace,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionSnapOwn,
    ActionSnapOpponent,
)
from .config import Config
from .abstraction import get_card_bucket, decay_bucket

logger = logging.getLogger(__name__)


@dataclass
class KnownCardInfo:
    """Stores info about a card location where the agent knows the card."""

    bucket: CardBucket
    last_seen_turn: int = 0
    card: Optional[Card] = None  # Store actual card for perfect recall/validation

    def __post_init__(self):
        if not isinstance(self.bucket, CardBucket):
            logger.warning(
                "KnownCardInfo initialized with non-Enum bucket: %s. Setting UNKNOWN.",
                self.bucket,
            )
            self.bucket = CardBucket.UNKNOWN


@dataclass
class AgentObservation:
    """Minimum information passed down CFR about state changes."""

    acting_player: int
    action: Optional[GameAction]
    discard_top_card: Optional[Card]
    player_hand_sizes: List[int]
    stockpile_size: int
    drawn_card: Optional[Card] = None
    peeked_cards: Optional[Dict[Tuple[int, int], Card]] = None
    snap_results: List[Dict[str, Any]] = field(default_factory=list)
    did_cambia_get_called: bool = False
    who_called_cambia: Optional[int] = None
    is_game_over: bool = False
    current_turn: int = 0


@dataclass
class AgentState:
    """
    Represents the agent's subjective belief about the game state,
    incorporating abstractions and memory limitations.
    """

    player_id: int
    opponent_id: int
    memory_level: int
    time_decay_turns: int
    initial_hand_size: int
    config: Config

    # Core Beliefs
    own_hand: Dict[int, KnownCardInfo] = field(default_factory=dict)
    opponent_belief: Dict[int, Union[CardBucket, DecayCategory]] = field(
        default_factory=dict
    )
    opponent_last_seen_turn: Dict[int, int] = field(default_factory=dict)

    # Public knowledge
    known_discard_top_bucket: CardBucket = CardBucket.UNKNOWN
    opponent_card_count: int = 0
    stockpile_estimate: StockpileEstimate = StockpileEstimate.HIGH
    game_phase: GamePhase = GamePhase.START
    cambia_caller: Optional[int] = None

    # Internal tracking
    _current_game_turn: int = 0

    def initialize(
        self,
        initial_observation: AgentObservation,
        initial_hand: List[Card],
        initial_peek_indices: Tuple[int, ...],
    ):
        """Initialize belief state at the very start of the game."""
        self.opponent_card_count = initial_observation.player_hand_sizes[self.opponent_id]
        self.stockpile_estimate = self._estimate_stockpile(
            initial_observation.stockpile_size
        )
        self.game_phase = self._estimate_game_phase(
            initial_observation.stockpile_size, None, 0
        )
        self.known_discard_top_bucket = get_card_bucket(
            initial_observation.discard_top_card
        )
        self.cambia_caller = None
        self._current_game_turn = 0

        self.own_hand = {}
        if len(initial_hand) != self.initial_hand_size:
            logger.warning(
                "Agent %d: Initial hand size mismatch! Expected %d, got %d.",
                self.player_id,
                self.initial_hand_size,
                len(initial_hand),
            )
        for i in range(self.initial_hand_size):
            card = initial_hand[i] if i < len(initial_hand) else None
            known = i in initial_peek_indices
            bucket = get_card_bucket(card) if known and card else CardBucket.UNKNOWN
            if known and card:
                logger.debug(
                    "Agent %d initial peek: Index %d is %s (%s)",
                    self.player_id,
                    i,
                    bucket.name,
                    card,
                )
            self.own_hand[i] = KnownCardInfo(
                bucket=bucket, last_seen_turn=0, card=card if known else None
            )

        self.opponent_belief = {
            i: CardBucket.UNKNOWN for i in range(self.opponent_card_count)
        }
        self.opponent_last_seen_turn = {}

        logger.debug(
            "Agent %d initialized (T%d). OH(%d): %s. OB(%d): %s",
            self.player_id,
            self._current_game_turn,
            len(self.own_hand),
            {k: v.bucket.name for k, v in self.own_hand.items()},
            self.opponent_card_count,
            {k: v.name for k, v in self.opponent_belief.items()},
        )

    def update(self, observation: AgentObservation):
        """Updates belief state based on an observation tuple."""
        if observation.current_turn < self._current_game_turn:
            logger.warning(
                "Agent %d received stale observation for T%d (current: T%d). Skipping.",
                self.player_id,
                observation.current_turn,
                self._current_game_turn,
            )
            return
        if observation.is_game_over:
            logger.debug(
                "Agent %d received game over observation. No state updates.",
                self.player_id,
            )
            return

        prev_turn = self._current_game_turn
        self._current_game_turn = observation.current_turn
        logger.debug(
            "===== Agent %d Update: T%d -> T%d =====",
            self.player_id,
            prev_turn,
            self._current_game_turn,
        )
        logger.debug(
            "Observation Action: %s by P%d", observation.action, observation.acting_player
        )
        logger.debug("Snap Results: %s", observation.snap_results)
        logger.debug(
            "Observed Counts: Own=%d, Opp=%d",
            observation.player_hand_sizes[self.player_id],
            observation.player_hand_sizes[self.opponent_id],
        )

        # --- 1. Update Public Knowledge & Counts ---
        self.known_discard_top_bucket = get_card_bucket(observation.discard_top_card)
        observed_opp_count = observation.player_hand_sizes[self.opponent_id]
        observed_own_count = observation.player_hand_sizes[self.player_id]
        self.stockpile_estimate = self._estimate_stockpile(observation.stockpile_size)
        if observation.did_cambia_get_called and self.cambia_caller is None:
            self.cambia_caller = observation.who_called_cambia
        self.game_phase = self._estimate_game_phase(
            observation.stockpile_size, self.cambia_caller, self._current_game_turn
        )
        logger.debug(
            " Public State Updated: Disc:%s, Stock:%s, Phase:%s, Cambia:%s",
            self.known_discard_top_bucket.name,
            self.stockpile_estimate.name,
            self.game_phase.name,
            self.cambia_caller,
        )

        # --- 2. Process Snap Results & Determine Removals/Adds ---
        action = observation.action
        actor = observation.acting_player
        snap_results = observation.snap_results

        own_indices_removed: Set[int] = set()
        opponent_indices_removed: Set[int] = set()
        own_cards_added_count = 0
        opponent_cards_added_count = 0
        num_penalty_cards = self.config.cambia_rules.penaltyDrawCount

        original_own_hand = copy.deepcopy(self.own_hand)
        original_opponent_belief = copy.deepcopy(self.opponent_belief)
        original_opponent_last_seen = copy.deepcopy(self.opponent_last_seen_turn)

        for snap_info in snap_results:
            try:
                snapper = snap_info.get("snapper")
                success = snap_info.get("success", False)
                penalty = snap_info.get("penalty", False)
                action_type = snap_info.get("action_type")

                if penalty:
                    if snapper == self.player_id:
                        own_cards_added_count += num_penalty_cards
                        logger.debug(
                            " -> P%d got penalty (%d cards)",
                            self.player_id,
                            num_penalty_cards,
                        )
                    elif snapper == self.opponent_id:
                        opponent_cards_added_count += num_penalty_cards
                        logger.debug(
                            " -> P%d got penalty (%d cards)",
                            self.opponent_id,
                            num_penalty_cards,
                        )

                elif success:
                    if action_type == "ActionSnapOwn":
                        removed_idx = snap_info.get("removed_own_index")
                        if removed_idx is not None:
                            if snapper == self.player_id:
                                own_indices_removed.add(removed_idx)
                                logger.debug(
                                    " -> P%d snapped own idx %d",
                                    self.player_id,
                                    removed_idx,
                                )
                            elif snapper == self.opponent_id:
                                opponent_indices_removed.add(removed_idx)
                                logger.debug(
                                    " -> P%d snapped own idx %d",
                                    self.opponent_id,
                                    removed_idx,
                                )
                    elif action_type == "ActionSnapOpponent":
                        removed_idx = snap_info.get("removed_opponent_index")
                        if removed_idx is not None:
                            if snapper == self.player_id:  # We snapped opponent
                                opponent_indices_removed.add(removed_idx)
                                logger.debug(
                                    " -> P%d snapped opp idx %d",
                                    self.player_id,
                                    removed_idx,
                                )
                            elif snapper == self.opponent_id:  # Opponent snapped us
                                own_indices_removed.add(removed_idx)
                                logger.debug(
                                    " -> P%d snapped our idx %d",
                                    self.opponent_id,
                                    removed_idx,
                                )

            except Exception as e_snap_proc:
                logger.error(
                    "Agent %d: Error processing snap_info %s: %s",
                    self.player_id,
                    snap_info,
                    e_snap_proc,
                )

        logger.debug(
            " Indices Removed: Own=%s, Opp=%s",
            own_indices_removed,
            opponent_indices_removed,
        )
        logger.debug(
            " Cards Added (Penalty): Own=%d, Opp=%d",
            own_cards_added_count,
            opponent_cards_added_count,
        )

        # --- 3. Reconcile Hand States ---
        try:
            logger.debug(
                " Rebuilding Own Hand (Current Size: %d)... Expecting %d.",
                len(self.own_hand),
                observed_own_count,
            )
            self.own_hand = self._rebuild_hand_state(
                original_dict=original_own_hand,
                removed_indices=own_indices_removed,
                expected_count=observed_own_count,
                added_count=own_cards_added_count,  # Pass added count
                is_own_hand=True,
            )
            logger.debug(
                " Rebuilding Opp Belief (Current Size: %d)... Expecting %d.",
                self.opponent_card_count,
                observed_opp_count,
            )
            new_opponent_belief, self.opponent_last_seen_turn = (
                self._rebuild_belief_state(
                    original_belief_dict=original_opponent_belief,
                    original_last_seen_dict=original_opponent_last_seen,
                    removed_indices=opponent_indices_removed,
                    expected_count=observed_opp_count,
                    added_count=opponent_cards_added_count,  # Pass added count
                )
            )
            # Only update if rebuild didn't return None (indicating fatal error)
            if new_opponent_belief is not None:
                self.opponent_belief = new_opponent_belief
                self.opponent_card_count = (
                    observed_opp_count  # Update count *after* successful reconciliation
                )
                logger.debug(" Belief Reconciliation Successful.")
            else:
                logger.error(
                    "Opponent belief rebuild failed, state potentially corrupted."
                )
                # Maintain previous count but log inconsistency
                self.opponent_card_count = len(self.opponent_belief)

        except Exception as e_reconcile:
            logger.exception(
                "Agent %d: Error during hand/belief reconciliation T%d: %s. State may be inconsistent.",
                self.player_id,
                self._current_game_turn,
                e_reconcile,
            )
            self.opponent_card_count = observed_opp_count

        logger.debug(
            " Reconciled State: OH(%d): %s. OB(%d): %s",
            len(self.own_hand),
            {k: v.bucket.name for k, v in self.own_hand.items()},
            self.opponent_card_count,
            {k: v.name for k, v in self.opponent_belief.items()},
        )

        # --- 4. Updates based on Main Action (operating on reconciled state) ---
        if action and actor != -1:
            try:
                if actor == self.player_id:
                    if isinstance(action, ActionReplace):
                        drawn_card_from_obs = observation.drawn_card
                        target_idx = action.target_hand_index
                        if (
                            drawn_card_from_obs
                        ):  # Card *must* be known if we are replacing
                            drawn_bucket = get_card_bucket(drawn_card_from_obs)
                            if target_idx in self.own_hand:
                                self.own_hand[target_idx] = KnownCardInfo(
                                    bucket=drawn_bucket,
                                    last_seen_turn=self._current_game_turn,
                                    card=drawn_card_from_obs,
                                )
                                logger.debug(
                                    " Agent %d updated own hand idx %d to %s (%s) after replace.",
                                    self.player_id,
                                    target_idx,
                                    drawn_bucket.name,
                                    drawn_card_from_obs,
                                )
                            else:
                                logger.warning(
                                    " Agent %d Replace target index %d not found after reconciliation (Current: %s). State inconsistent.",
                                    self.player_id,
                                    target_idx,
                                    list(self.own_hand.keys()),
                                )
                        else:
                            # This error means the observation creation logic failed upstream
                            logger.error(
                                " Agent %d Replace action observed, but no drawn card in observation! Cannot update belief.",
                                self.player_id,
                            )

                    elif isinstance(
                        action, (ActionAbilityPeekOwnSelect, ActionAbilityKingLookSelect)
                    ):
                        if observation.peeked_cards:
                            for (p_idx, h_idx), card in observation.peeked_cards.items():
                                if not isinstance(card, Card):
                                    continue
                                peeked_bucket = get_card_bucket(card)
                                if p_idx == self.player_id and h_idx in self.own_hand:
                                    self.own_hand[h_idx] = KnownCardInfo(
                                        bucket=peeked_bucket,
                                        last_seen_turn=self._current_game_turn,
                                        card=card,
                                    )
                                    logger.debug(
                                        " Agent %d (%s) updated own idx %d knowledge to %s (%s).",
                                        self.player_id,
                                        type(action).__name__,
                                        h_idx,
                                        peeked_bucket.name,
                                        card,
                                    )
                                elif (
                                    p_idx == self.opponent_id
                                    and h_idx in self.opponent_belief
                                ):
                                    self.opponent_belief[h_idx] = peeked_bucket
                                    self.opponent_last_seen_turn[h_idx] = (
                                        self._current_game_turn
                                    )
                                    logger.debug(
                                        " Agent %d (%s) updated opp idx %d belief to %s (%s).",
                                        self.player_id,
                                        type(action).__name__,
                                        h_idx,
                                        peeked_bucket.name,
                                        card,
                                    )

                    elif isinstance(action, ActionAbilityPeekOtherSelect):
                        if observation.peeked_cards:
                            for (p_idx, h_idx), card in observation.peeked_cards.items():
                                if not isinstance(card, Card):
                                    continue
                                if (
                                    p_idx == self.opponent_id
                                    and h_idx in self.opponent_belief
                                ):
                                    peeked_bucket = get_card_bucket(card)
                                    self.opponent_belief[h_idx] = peeked_bucket
                                    self.opponent_last_seen_turn[h_idx] = (
                                        self._current_game_turn
                                    )
                                    logger.debug(
                                        " Agent %d (PeekOther) updated opp idx %d belief to %s (%s).",
                                        self.player_id,
                                        h_idx,
                                        peeked_bucket.name,
                                        card,
                                    )

                    elif isinstance(action, ActionAbilityBlindSwapSelect):
                        own_idx, opp_idx_target = (
                            action.own_hand_index,
                            action.opponent_hand_index,
                        )
                        if own_idx in self.own_hand:
                            self.own_hand[own_idx] = KnownCardInfo(
                                bucket=CardBucket.UNKNOWN,
                                last_seen_turn=self._current_game_turn,
                                card=None,
                            )
                            logger.debug(
                                " Agent %d (BlindSwap) updated own idx %d to UNKNOWN.",
                                self.player_id,
                                own_idx,
                            )
                        # Decay opponent belief at target index
                        self._trigger_event_decay(
                            target_index=opp_idx_target,
                            trigger_event="swap (blind)",
                            current_turn=self._current_game_turn,
                        )

                    elif (
                        isinstance(action, ActionAbilityKingSwapDecision)
                        and action.perform_swap
                    ):
                        # We performed the swap. We don't know what we received unless we remember the peek.
                        # The card we gave away is now gone.
                        # We need the indices involved, which aren't in this action type. Assume they were stored/passed correctly?
                        # Best effort: Set both potentially involved indices (if remembered) to UNKNOWN.
                        # This relies on state consistency, which might be flawed.
                        # For now, just log. Precise update is complex without more context.
                        logger.debug(
                            " Agent %d observed self perform King Swap. Setting involved own card to UNKNOWN if index available.",
                            self.player_id,
                        )
                        # If we stored the peeked indices in pending_action_data for the Decision step, we could use them here.
                        # Example: own_involved_idx = self.pending_action_data.get("own_idx")
                        #          opp_involved_idx = self.pending_action_data.get("opp_idx")
                        #          if own_involved_idx in self.own_hand: self.own_hand[own_involved_idx] = KnownCardInfo(...) # Set to Unknown
                        #          self._trigger_event_decay(opp_involved_idx, ...)

                    elif isinstance(action, ActionSnapOwn):
                        # Handled by reconciliation
                        pass
                    elif isinstance(action, ActionSnapOpponent):
                        # Handled by reconciliation (opponent card removal) and subsequent move action
                        pass
                    elif isinstance(action, ActionSnapOpponentMove):
                        # Handled by reconciliation (own card removal)
                        pass

                elif actor == self.opponent_id:
                    # Observe opponent actions
                    if isinstance(action, ActionReplace):
                        target_idx = (
                            action.target_hand_index
                        )  # Opponent's index that got replaced
                        self._trigger_event_decay(
                            target_index=target_idx,
                            trigger_event="replace (opponent)",
                            current_turn=self._current_game_turn,
                        )

                    elif isinstance(action, ActionAbilityBlindSwapSelect):
                        opp_own_idx = action.own_hand_index  # Index in opponent's hand
                        our_idx = action.opponent_hand_index  # Index in our hand
                        # Decay opponent belief at their index
                        self._trigger_event_decay(
                            target_index=opp_own_idx,
                            trigger_event="swap (blind, opponent)",
                            current_turn=self._current_game_turn,
                        )
                        # Update our own card to unknown
                        if our_idx in self.own_hand:
                            self.own_hand[our_idx] = KnownCardInfo(
                                bucket=CardBucket.UNKNOWN,
                                last_seen_turn=self._current_game_turn,
                                card=None,
                            )
                            logger.debug(
                                " Agent %d updated own idx %d to UNKNOWN due to opponent BlindSwap.",
                                self.player_id,
                                our_idx,
                            )

                    elif (
                        isinstance(action, ActionAbilityKingSwapDecision)
                        and action.perform_swap
                    ):
                        # Opponent performed swap. We don't know which of their cards unless we peeked.
                        # We only know if *our* card was involved if it was part of the peek observation *for them*.
                        # Without that info, we can only assume an opponent swap occurred.
                        logger.debug(
                            " Agent %d observed opponent King Swap.", self.player_id
                        )
                        # Decay triggered if our card was involved? Cannot know from this obs alone.

                    elif isinstance(action, ActionSnapOwn):
                        # Handled by reconciliation (opponent count change)
                        pass
                    elif isinstance(action, ActionSnapOpponent):
                        # Handled by reconciliation (our count change) and move action
                        pass
                    elif isinstance(action, ActionSnapOpponentMove):
                        # Handled by reconciliation (opponent count change)
                        pass

                    # Opponent discard/peek doesn't directly reveal info unless card known

            except Exception as e_action_proc:
                logger.error(
                    "Agent %d: Error processing action %s in update T%d: %s",
                    self.player_id,
                    action,
                    self._current_game_turn,
                    e_action_proc,
                    exc_info=True,
                )

        # --- 5. Apply Time Decay (Level 2) ---
        if self.memory_level == 2:
            self._apply_time_decay(self._current_game_turn)

        logger.debug(" Update Complete. State: %s", self)

    def _rebuild_hand_state(
        self,
        original_dict: Dict[int, KnownCardInfo],
        removed_indices: Set[int],
        expected_count: int,
        added_count: int,  # Number of new unknown cards added (e.g., penalty)
        is_own_hand: bool,
    ) -> Optional[Dict[int, KnownCardInfo]]:
        """Rebuilds hand state ensuring contiguous indices 0..N-1, handling adds/removals."""
        new_dict: Dict[int, KnownCardInfo] = {}
        current_items = []
        player_desc = "Own" if is_own_hand else "Opponent"  # Should always be Own

        # Collect valid items from original dict, excluding removed ones
        original_indices_sorted = sorted(original_dict.keys())
        logger.debug(
            "  Rebuild %s Hand - Original Indices: %s, Removing: %s",
            player_desc,
            original_indices_sorted,
            removed_indices,
        )
        for idx in original_indices_sorted:
            if idx not in removed_indices:
                current_items.append(original_dict[idx])

        effective_expected_count = (
            expected_count  # This is the final count after adds/removes
        )

        # Check consistency before adding penalty cards
        expected_count_before_adds = len(original_dict) - len(removed_indices)
        if len(current_items) != expected_count_before_adds:
            logger.error(
                "Rebuild %s Hand T%d: Inconsistency! Items after removal (%d) != expected before adds (%d). Removed: %s, Original Indices: %s",
                player_desc,
                self._current_game_turn,
                len(current_items),
                expected_count_before_adds,
                removed_indices,
                original_indices_sorted,
            )
            # Attempt to proceed, but state might be wrong

        # Add placeholders for new cards (penalty) - append to the end conceptually
        for _ in range(added_count):
            current_items.append(
                KnownCardInfo(
                    bucket=CardBucket.UNKNOWN,
                    last_seen_turn=self._current_game_turn,
                    card=None,
                )
            )
            logger.debug(" -> Added placeholder for penalty/new card.")

        # Assign items to new contiguous indices up to the final expected count
        final_item_count = len(current_items)
        for i, item in enumerate(current_items):
            if i < effective_expected_count:
                new_dict[i] = item
            else:
                # This error means adds+removes didn't match final expected count
                logger.error(
                    "Rebuild %s Hand T%d: More items (%d) than expected (%d) after adds/removes. Discarding extra. Removed: %s, Added: %d, Original Indices: %s",
                    player_desc,
                    self._current_game_turn,
                    final_item_count,
                    effective_expected_count,
                    removed_indices,
                    added_count,
                    original_indices_sorted,
                )
                break  # Stop adding extra items

        # Final check - if dict size doesn't match expected count, something went wrong
        if len(new_dict) != effective_expected_count:
            logger.critical(
                "FATAL: %s hand final rebuild T%d failed. Expected %d, got %d. Indices: %s. State likely corrupt!",
                player_desc,
                self._current_game_turn,
                effective_expected_count,
                len(new_dict),
                sorted(new_dict.keys()),
            )
            # Return a default state to prevent downstream crashes? Difficult choice. Return None?
            # For now, log critical error and return the potentially incorrect dict.
            # Consider returning None or raising an exception in future if this proves too problematic.
            # return None
            # Fallback: create default dict of expected size
            # return {i: KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn) for i in range(effective_expected_count)}

        logger.debug(
            "  Rebuild %s Hand - Final Indices: %s (Count: %d)",
            player_desc,
            sorted(new_dict.keys()),
            len(new_dict),
        )
        return new_dict

    def _rebuild_belief_state(
        self,
        original_belief_dict: Dict[int, Union[CardBucket, DecayCategory]],
        original_last_seen_dict: Dict[int, int],
        removed_indices: Set[int],
        expected_count: int,
        added_count: int,
    ) -> Optional[Tuple[Dict[int, Union[CardBucket, DecayCategory]], Dict[int, int]]]:
        """Rebuilds opponent_belief and opponent_last_seen_turn ensuring contiguous indices."""
        new_belief_dict: Dict[int, Union[CardBucket, DecayCategory]] = {}
        new_last_seen_dict: Dict[int, int] = {}
        current_belief_items = []
        current_last_seen_mapping = {}  # Maps OLD index -> last_seen

        original_indices_sorted = sorted(original_belief_dict.keys())
        logger.debug(
            "  Rebuild Opp Belief - Original Indices: %s, Removing: %s",
            original_indices_sorted,
            removed_indices,
        )

        # Collect valid items and their original last_seen timestamps
        for idx in original_indices_sorted:
            if idx not in removed_indices:
                current_belief_items.append(original_belief_dict[idx])
                if idx in original_last_seen_dict:
                    current_last_seen_mapping[idx] = original_last_seen_dict[idx]

        # Add placeholders for new (penalty) cards
        for _ in range(added_count):
            current_belief_items.append(CardBucket.UNKNOWN)
            logger.debug(" -> Added UNKNOWN placeholder for opponent penalty/new card.")

        # Assign items to new contiguous indices
        final_item_count = len(current_belief_items)
        effective_expected_count = expected_count

        # Map old indices that were kept to their corresponding item in current_belief_items
        old_indices_kept = [
            idx for idx in original_indices_sorted if idx not in removed_indices
        ]

        for i, belief in enumerate(current_belief_items):
            if i < effective_expected_count:
                new_belief_dict[i] = belief
                # Restore last_seen timestamp ONLY if this belief came from the original dict
                # New penalty cards (UNKNOWN) should not have a timestamp yet.
                if i < len(
                    old_indices_kept
                ):  # Check if this index corresponds to an original kept item
                    original_idx = old_indices_kept[i]
                    if original_idx in current_last_seen_mapping:
                        new_last_seen_dict[i] = current_last_seen_mapping[original_idx]
            else:
                logger.error(
                    "Rebuild Opponent Belief T%d: More items (%d) than expected (%d) after adds/removes. Discarding extra. Removed: %s, Added: %d, Original Indices: %s",
                    self._current_game_turn,
                    final_item_count,
                    effective_expected_count,
                    removed_indices,
                    added_count,
                    original_indices_sorted,
                )
                break

        # Final check
        if len(new_belief_dict) != effective_expected_count:
            logger.critical(
                "FATAL: Opponent belief final rebuild T%d failed. Expected %d, got %d. Indices: %s. State likely corrupt!",
                self._current_game_turn,
                effective_expected_count,
                len(new_belief_dict),
                sorted(new_belief_dict.keys()),
            )
            # return None, None # Indicate failure
            # Fallback:
            new_belief_dict = {
                i: CardBucket.UNKNOWN for i in range(effective_expected_count)
            }
            new_last_seen_dict = {}

        logger.debug(
            "  Rebuild Opp Belief - Final Indices: %s (Count: %d)",
            sorted(new_belief_dict.keys()),
            len(new_belief_dict),
        )
        # Ensure last_seen only contains keys present in the final belief dict
        valid_last_seen = {
            k: v for k, v in new_last_seen_dict.items() if k in new_belief_dict
        }

        return new_belief_dict, valid_last_seen

    def _estimate_stockpile(self, stock_size: int) -> StockpileEstimate:
        """Estimates stockpile category based on size."""
        try:
            total_cards = 52 + self.config.cambia_rules.use_jokers
            low_threshold = max(1, total_cards // 5)  # e.g., 10 for 54 cards
            med_threshold = max(
                low_threshold + 1, total_cards * 2 // 4
            )  # e.g., 27 for 54 cards
        except AttributeError:
            logger.warning("Config missing during stockpile estimate, using defaults.")
            low_threshold, med_threshold = 10, 27

        if stock_size <= 0:
            return StockpileEstimate.EMPTY
        if stock_size < low_threshold:
            return StockpileEstimate.LOW
        if stock_size < med_threshold:
            return StockpileEstimate.MEDIUM
        return StockpileEstimate.HIGH

    def _estimate_game_phase(
        self, stock_size: int, cambia_caller: Optional[int], current_turn: int
    ) -> GamePhase:
        """Estimates the game phase."""
        if cambia_caller is not None:
            return GamePhase.CAMBIA_CALLED
        stock_cat = self._estimate_stockpile(stock_size)
        # Simple phase based only on stockpile estimate for now
        if stock_cat == StockpileEstimate.EMPTY:
            return GamePhase.LATE
        if stock_cat == StockpileEstimate.LOW:
            return GamePhase.LATE
        if stock_cat == StockpileEstimate.MEDIUM:
            return GamePhase.MID
        # Could add START phase based on turn number if desired
        # if current_turn < 2: return GamePhase.START
        return GamePhase.EARLY

    def _trigger_event_decay(
        self, target_index: int, trigger_event: str, current_turn: int
    ):
        """Applies event-based decay (Levels 1 and 2) to a specific opponent index."""
        if self.memory_level == 0:
            return

        if target_index in self.opponent_belief:
            current_belief = self.opponent_belief[target_index]
            # Only decay if current belief is specific (CardBucket, not UNKNOWN or DecayCategory)
            if (
                isinstance(current_belief, CardBucket)
                and current_belief != CardBucket.UNKNOWN
            ):
                decayed_category = decay_bucket(current_belief)
                self.opponent_belief[target_index] = decayed_category
                self.opponent_last_seen_turn.pop(
                    target_index, None
                )  # Clear timestamp on decay
                logger.debug(
                    " Agent %d decaying Opponent[%d] belief from %s to %s due to %s.",
                    self.player_id,
                    target_index,
                    current_belief.name,
                    decayed_category.name,
                    trigger_event,
                )
            # else: logger.debug(" Decay trigger skipped for Opponent[%d]: Already %s.", target_index, current_belief.name) # Too noisy
        # else: logger.debug(" Decay trigger skipped: Index %d not in opponent belief %s.", target_index, list(self.opponent_belief.keys()))

    def _apply_time_decay(self, current_turn: int):
        """Applies time-based decay (Level 2 only)."""
        if self.memory_level != 2:
            return

        indices_to_decay = [
            idx
            for idx, last_seen in self.opponent_last_seen_turn.items()
            if current_turn - last_seen >= self.time_decay_turns
            and idx in self.opponent_belief  # Check if index still exists
            and isinstance(
                self.opponent_belief[idx], CardBucket
            )  # Check not already decayed
            and self.opponent_belief[idx] != CardBucket.UNKNOWN
        ]

        for idx in indices_to_decay:
            current_belief = self.opponent_belief[idx]
            decayed_category = decay_bucket(current_belief)
            self.opponent_belief[idx] = decayed_category
            self.opponent_last_seen_turn.pop(idx)  # Remove timestamp after decay
            logger.debug(
                " Agent %d applying time decay for Opponent[%d] from %s to %s.",
                self.player_id,
                idx,
                current_belief.name,
                decayed_category.name,
            )

    def get_infoset_key(self) -> Tuple:  # Return plain tuple for direct use
        """Constructs the canonical, hashable infoset key tuple from the current belief state."""
        try:
            # 1. Own Hand Buckets (Tuple sorted by index)
            own_hand_items = sorted(self.own_hand.items())
            own_hand_buckets = tuple(info.bucket.value for _, info in own_hand_items)

            # 2. Opponent Beliefs (Tuple indexed 0..N-1)
            opp_belief_values = []
            for i in range(self.opponent_card_count):  # Use reconciled count
                belief = self.opponent_belief.get(i, CardBucket.UNKNOWN)
                opp_belief_values.append(
                    belief.value if hasattr(belief, "value") else CardBucket.UNKNOWN.value
                )
            opp_belief_tuple = tuple(opp_belief_values)

            # 3. Other components
            opp_count = self.opponent_card_count
            discard_top_val = self.known_discard_top_bucket.value
            stockpile_est_val = self.stockpile_estimate.value
            game_phase_val = self.game_phase.value
            # DecisionContext is added externally by the calling function

            key_tuple = (
                own_hand_buckets,
                opp_belief_tuple,
                opp_count,
                discard_top_val,
                stockpile_est_val,
                game_phase_val,
            )
            return key_tuple
        except Exception as e_key:
            logger.exception(
                "Agent %d: Error generating infoset key tuple T%d: %s",
                self.player_id,
                self._current_game_turn,
                e_key,
            )
            # Return a default/error tuple
            return ((-99,), (-99,), -1, -1, -1, -1)

    def get_potential_opponent_snap_indices(self, target_rank: str) -> List[int]:
        """Returns opponent hand indices the agent *believes* could match the target rank."""
        matching_indices = []
        try:
            # Get target buckets for the rank (could be different for Red/Black King)
            target_buckets = set()
            for suit in [
                "S",
                "H",
            ]:  # Use one black and one red suit to get both King buckets
                try:
                    card = Card(
                        rank=target_rank, suit=suit if target_rank != "R" else None
                    )  # Handle Joker suit
                    target_buckets.add(get_card_bucket(card))
                except ValueError:
                    pass  # Ignore invalid card combos if rank doesn't take suit

            if not target_buckets:
                logger.error(
                    "Agent %d: Could not determine target buckets for rank '%s'.",
                    self.player_id,
                    target_rank,
                )
                return []

        except ValueError:
            logger.error(
                "Agent %d: Invalid target_rank '%s' for get_potential_opponent_snap_indices.",
                self.player_id,
                target_rank,
            )
            return []

        for idx, belief in self.opponent_belief.items():
            if belief == CardBucket.UNKNOWN or isinstance(belief, DecayCategory):
                matching_indices.append(idx)  # Assume unknown/decayed *could* match
            elif isinstance(belief, CardBucket) and belief in target_buckets:
                matching_indices.append(idx)

        return matching_indices

    def clone(self) -> "AgentState":
        """Creates a deep copy of the agent state."""
        try:
            # Use copy.deepcopy for mutable fields
            new_state = AgentState(
                player_id=self.player_id,
                opponent_id=self.opponent_id,
                memory_level=self.memory_level,
                time_decay_turns=self.time_decay_turns,
                initial_hand_size=self.initial_hand_size,
                config=self.config,  # Config can be shallow copied
                own_hand=copy.deepcopy(self.own_hand),
                opponent_belief=copy.deepcopy(self.opponent_belief),
                opponent_last_seen_turn=copy.deepcopy(self.opponent_last_seen_turn),
                known_discard_top_bucket=self.known_discard_top_bucket,
                opponent_card_count=self.opponent_card_count,
                stockpile_estimate=self.stockpile_estimate,
                game_phase=self.game_phase,
                cambia_caller=self.cambia_caller,
                _current_game_turn=self._current_game_turn,
            )
            return new_state
        except Exception as e_clone:
            logger.exception(
                "Agent %d: Error cloning agent state: %s", self.player_id, e_clone
            )
            raise  # Re-raise exception

    def __str__(self) -> str:
        """Provides a concise string representation of the agent's belief state."""
        try:
            own_hand_str = {k: v.bucket.name for k, v in sorted(self.own_hand.items())}
            opp_belief_str = {
                i: self.opponent_belief.get(i, CardBucket.UNKNOWN).name
                for i in range(self.opponent_card_count)
            }
            return (
                f"AgentState(P{self.player_id}, T:{self._current_game_turn}, Phase:{self.game_phase.name}, "
                f"OH({len(self.own_hand)}):{own_hand_str}, OB({self.opponent_card_count}):{opp_belief_str}, "
                f"Disc:{self.known_discard_top_bucket.name}, Stock:{self.stockpile_estimate.name}, Cambia:{self.cambia_caller})"
            )
        except Exception as e_str:
            logger.error(
                "Agent %d: Error generating string representation: %s",
                self.player_id,
                e_str,
            )
            return f"AgentState(P{self.player_id}, Error)"
