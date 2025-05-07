"""src/agent_state.py"""

from typing import List, Tuple, Optional, Dict, Any, Set, Union, TYPE_CHECKING
from dataclasses import dataclass, field
import logging
import copy

from .constants import (
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
)

# Use TYPE_CHECKING guard if Card is imported directly
if TYPE_CHECKING:
    from .card import Card
# Import Card only if needed for isinstance checks etc. within methods
from .card import Card
from .config import Config
from .abstraction import get_card_bucket, decay_bucket

logger = logging.getLogger(__name__)


@dataclass
class KnownCardInfo:
    """Stores info about a card location where the agent knows the card."""

    bucket: CardBucket
    last_seen_turn: int = 0
    # Use the 'Card' type hint directly here since Card is imported
    card: Optional[Card] = None

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
    # Use Card type hint here
    discard_top_card: Optional[Card]
    player_hand_sizes: List[int]
    stockpile_size: int
    # Use Card type hint here
    drawn_card: Optional[Card] = None
    # Use Card type hint here
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
    time_decay_turns: int  # Threshold for Level 2 decay
    initial_hand_size: int  # From config/rules
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
        # Use Card type hint here
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
        for i, card in enumerate(initial_hand):
            known = i in initial_peek_indices
            bucket = get_card_bucket(card) if known else CardBucket.UNKNOWN
            if known:
                logger.debug(
                    "Agent %d initial peek: Index %d is %s",
                    self.player_id,
                    i,
                    bucket.name,
                )
            # Store actual card if known (useful for memory_level 0)
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
            # Optionally update final state if needed for analysis, but typically not for CFR infoset keys
            return

        self._current_game_turn = observation.current_turn

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

        # --- 2. Process Snap Results & Determine Removals/Adds ---
        action = observation.action
        actor = observation.acting_player
        snap_results = observation.snap_results

        own_indices_removed: Set[int] = set()
        opponent_indices_removed: Set[int] = set()
        own_cards_added_count = 0  # Tracks penalty cards added *before* reconciliation
        opponent_cards_added_count = 0

        original_own_hand = dict(self.own_hand)  # Keep original state for reconciliation
        original_opponent_belief = dict(self.opponent_belief)
        original_opponent_last_seen = dict(self.opponent_last_seen_turn)

        for snap_info in snap_results:
            try:  # Add try-except around processing each snap result
                snapper = snap_info.get("snapper")
                success = snap_info.get("success", False)
                penalty = snap_info.get("penalty", False)
                # snapped_card = snap_info.get("snapped_card") # Card object might not be needed here
                removed_own_idx = snap_info.get("removed_own_index")
                removed_opp_idx = snap_info.get("removed_opponent_index")
                num_penalty_cards = self.config.cambia_rules.penaltyDrawCount

                if snapper == self.player_id:  # Our snap
                    if success:
                        if removed_own_idx is not None:
                            own_indices_removed.add(removed_own_idx)
                        elif removed_opp_idx is not None:
                            opponent_indices_removed.add(removed_opp_idx)
                    elif penalty:
                        own_cards_added_count += num_penalty_cards
                elif snapper == self.opponent_id:  # Opponent snap
                    if success:
                        if removed_own_idx is not None:
                            opponent_indices_removed.add(
                                removed_own_idx
                            )  # Opp snapped their own
                        elif removed_opp_idx is not None:
                            own_indices_removed.add(removed_opp_idx)  # Opp snapped ours
                    elif penalty:
                        opponent_cards_added_count += num_penalty_cards
            except Exception as e_snap_proc:
                logger.error(
                    "Agent %d: Error processing snap_info %s: %s",
                    self.player_id,
                    snap_info,
                    e_snap_proc,
                )
                # Continue processing other snap results if possible

        # --- 3. Reconcile Hand States ---
        try:
            self.own_hand = self._rebuild_hand_state(
                original_dict=original_own_hand,
                removed_indices=own_indices_removed,
                expected_count=observed_own_count,
                is_own_hand=True,
            )
            new_opponent_belief, self.opponent_last_seen_turn = (
                self._rebuild_belief_state(
                    original_belief_dict=original_opponent_belief,
                    original_last_seen_dict=original_opponent_last_seen,
                    removed_indices=opponent_indices_removed,
                    expected_count=observed_opp_count,
                )
            )
            self.opponent_belief = new_opponent_belief
            self.opponent_card_count = (
                observed_opp_count  # Update count *after* successful reconciliation
            )
        except Exception as e_reconcile:
            logger.exception(
                "Agent %d: Error during hand/belief reconciliation T%d: %s. State may be inconsistent.",
                self.player_id,
                self._current_game_turn,
                e_reconcile,
            )
            # If reconciliation fails, state is likely bad. Avoid further updates?
            # Fallback: Use observed counts but keep potentially wrong beliefs?
            self.opponent_card_count = observed_opp_count
            # Maybe reset beliefs to UNKNOWN? Risky. Log error and proceed cautiously.
            # self.own_hand = {i: KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn) for i in range(observed_own_count)}
            # self.opponent_belief = {i: CardBucket.UNKNOWN for i in range(observed_opp_count)}
            # self.opponent_last_seen_turn = {}

        # --- 4. Updates based on Main Action (operating on reconciled state) ---
        if action and actor != -1:  # Check actor validity
            try:  # Wrap action processing
                if actor == self.player_id:
                    if isinstance(action, ActionReplace):
                        drawn_card_from_obs = observation.drawn_card
                        target_idx = action.target_hand_index
                        # Backlog 13: Verify drawn_card is present
                        if drawn_card_from_obs:
                            drawn_bucket = get_card_bucket(drawn_card_from_obs)
                            if target_idx in self.own_hand:
                                self.own_hand[target_idx] = KnownCardInfo(
                                    bucket=drawn_bucket,
                                    last_seen_turn=self._current_game_turn,
                                    card=drawn_card_from_obs,
                                )
                                logger.debug(
                                    "Agent %d updated own hand idx %d to %s after replace with %s.",
                                    self.player_id,
                                    target_idx,
                                    drawn_bucket.name,
                                    drawn_card_from_obs,
                                )
                            else:
                                logger.warning(
                                    "Agent %d Replace target index %d not found after reconciliation (Current: %s). State inconsistent.",
                                    self.player_id,
                                    target_idx,
                                    list(self.own_hand.keys()),
                                )
                        else:
                            logger.error(
                                "Agent %d Replace action observed, but no drawn card in observation! Belief update failed.",
                                self.player_id,
                            )

                    elif isinstance(
                        action, (ActionAbilityPeekOwnSelect, ActionAbilityKingLookSelect)
                    ):
                        if observation.peeked_cards:
                            for (p_idx, h_idx), card in observation.peeked_cards.items():
                                if not isinstance(card, Card):
                                    continue  # Skip if not valid card
                                peeked_bucket = get_card_bucket(card)
                                if p_idx == self.player_id and h_idx in self.own_hand:
                                    self.own_hand[h_idx] = KnownCardInfo(
                                        bucket=peeked_bucket,
                                        last_seen_turn=self._current_game_turn,
                                        card=card,
                                    )
                                    logger.debug(
                                        "Agent %d (%s) updated own idx %d knowledge to %s.",
                                        self.player_id,
                                        type(action).__name__,
                                        h_idx,
                                        peeked_bucket.name,
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
                                        "Agent %d (%s) updated opp idx %d belief to %s.",
                                        self.player_id,
                                        type(action).__name__,
                                        h_idx,
                                        peeked_bucket.name,
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
                                        "Agent %d (PeekOther) updated opp idx %d belief to %s.",
                                        self.player_id,
                                        h_idx,
                                        peeked_bucket.name,
                                    )

                    elif isinstance(action, ActionAbilityBlindSwapSelect):
                        own_idx, opp_idx = (
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
                                "Agent %d (BlindSwap) updated own idx %d to UNKNOWN.",
                                self.player_id,
                                own_idx,
                            )
                        self._trigger_event_decay(
                            target_index=opp_idx,
                            trigger_event="swap (blind)",
                            current_turn=self._current_game_turn,
                        )

                    elif (
                        isinstance(action, ActionAbilityKingSwapDecision)
                        and action.perform_swap
                    ):
                        own_involved_idx, opp_involved_idx = -1, -1
                        # King Look info is not directly in *this* observation, agent relies on memory/previous obs.
                        # The swap itself makes both involved cards unknown from the swapping player's perspective.
                        # We need the indices from the action to know which cards were involved.
                        # PROBLEM: ActionAbilityKingSwapDecision doesn't hold the indices.
                        # The indices *should* have been used by the game engine to perform the swap.
                        # The agent state update here relies on the *effect* of the swap, not repeating the logic.
                        # If we peeked our own card (card1) and opp card (card2) and swapped:
                        # Our original card1's index now holds card2 (Unknown).
                        # The opponent's original card2's index now holds card1 (Unknown to us unless we peeked it before).
                        # We need to identify the indices involved based on prior knowledge/observation (difficult) or make assumptions.
                        # Assumption: Blindly set both involved cards to UNKNOWN. This requires getting indices.
                        # Let's rely on the `_rebuild_hand_state` to handle count changes and assume specific knowledge is lost.
                        # We can still trigger decay for the opponent's *potential* involved index if we stored it.
                        # This highlights a limitation if `peeked_cards` isn't available during the SwapDecision update.
                        # For now, just log the event occurred. Specific index updates are hard without more info passed.
                        logger.debug(
                            "Agent %d observed self perform King Swap. Specific belief updates depend on prior peeks.",
                            self.player_id,
                        )

                    # Discard action doesn't change belief unless ability is used (handled by peek/swap cases)
                    # SnapOpponentMove is handled by reconciliation counts.

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
                        self._trigger_event_decay(
                            target_index=opp_own_idx,
                            trigger_event="swap (blind, opponent)",
                            current_turn=self._current_game_turn,
                        )
                        if our_idx in self.own_hand:
                            self.own_hand[our_idx] = KnownCardInfo(
                                bucket=CardBucket.UNKNOWN,
                                last_seen_turn=self._current_game_turn,
                                card=None,
                            )
                            logger.debug(
                                "Agent %d updated own idx %d to UNKNOWN due to opponent BlindSwap.",
                                self.player_id,
                                our_idx,
                            )

                    elif (
                        isinstance(action, ActionAbilityKingSwapDecision)
                        and action.perform_swap
                    ):
                        # Opponent performed King Swap. We don't know which cards unless we were peeked.
                        # Check if the observation contains peek info (it shouldn't for opponent's swap decision step).
                        # We only know a swap happened involving two unknown (to us) indices in opponent hand,
                        # and potentially one known (to us) index in our hand.
                        # Safest is to trigger decay if our card was involved.
                        # How to know if our card was involved? Requires info not present here.
                        # If the game engine provided which indices were swapped publicly, we could use it.
                        # For now, assume we cannot update specific beliefs based on opponent's swap action alone.
                        logger.debug(
                            "Agent %d observed opponent King Swap. Cannot update specific beliefs without more info.",
                            self.player_id,
                        )

                    # Opponent discard/peek/SnapMove don't directly reveal info to us unless card itself is known
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

    def _rebuild_hand_state(
        self,
        original_dict: Dict[int, KnownCardInfo],
        removed_indices: Set[int],
        expected_count: int,
        is_own_hand: bool,
    ) -> Dict[int, KnownCardInfo]:
        """Rebuilds hand state (own_hand) ensuring contiguous indices 0..N-1."""
        new_dict: Dict[int, KnownCardInfo] = {}
        current_items = []

        # Collect valid items from original dict, excluding removed ones, maintaining order
        for idx in sorted(original_dict.keys()):
            if idx not in removed_indices:
                current_items.append(original_dict[idx])

        # Assign items to new contiguous indices
        for i, item in enumerate(current_items):
            if i < expected_count:
                new_dict[i] = item
            else:
                player_desc = (
                    "Own" if is_own_hand else "Opponent"
                )  # Should always be Own here
                logger.error(
                    "Rebuild %s Hand T%d: More items (%d) than expected (%d) after removal. Discarding extra. Removed: %s, Original Indices: %s",
                    player_desc,
                    self._current_game_turn,
                    len(current_items),
                    expected_count,
                    removed_indices,
                    sorted(original_dict.keys()),
                )
                break

        # Add UNKNOWNs if needed to reach expected count (e.g., due to penalty)
        while len(new_dict) < expected_count:
            new_index = len(new_dict)
            if is_own_hand:
                new_dict[new_index] = KnownCardInfo(
                    bucket=CardBucket.UNKNOWN,
                    last_seen_turn=self._current_game_turn,
                    card=None,
                )
                logger.debug(
                    "Agent %d rebuilding own hand added UNKNOWN at index %d (Expected: %d, Current: %d)",
                    self.player_id,
                    new_index,
                    expected_count,
                    len(new_dict) - 1,
                )
            else:  # Should not happen for own hand
                logger.error("_rebuild_hand_state called for opponent belief.")
                break

        if len(new_dict) != expected_count:
            player_desc = "Own" if is_own_hand else "Opponent"
            logger.error(
                "FATAL: %s hand final rebuild T%d failed. Expected %d, got %d. Indices: %s. Removed: %s, Original Indices: %s",
                player_desc,
                self._current_game_turn,
                expected_count,
                len(new_dict),
                sorted(new_dict.keys()),
                removed_indices,
                sorted(original_dict.keys()),
            )
            # Attempt to return a default state to prevent crashes downstream?
            # return {i: KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn) for i in range(expected_count)}

        return new_dict

    def _rebuild_belief_state(
        self,
        original_belief_dict: Dict[int, Union[CardBucket, DecayCategory]],
        original_last_seen_dict: Dict[int, int],
        removed_indices: Set[int],
        expected_count: int,
    ) -> Tuple[Dict[int, Union[CardBucket, DecayCategory]], Dict[int, int]]:
        """Rebuilds opponent_belief and opponent_last_seen_turn ensuring contiguous indices."""
        new_belief_dict: Dict[int, Union[CardBucket, DecayCategory]] = {}
        new_last_seen_dict: Dict[int, int] = {}
        current_belief_items = []
        current_last_seen_mapping = {}  # Store mapping from old index -> last_seen

        for idx in sorted(original_belief_dict.keys()):
            if idx not in removed_indices:
                current_belief_items.append(original_belief_dict[idx])
                if idx in original_last_seen_dict:
                    # Store last seen associated with the original index
                    current_last_seen_mapping[idx] = original_last_seen_dict[idx]

        # Map old indices to new indices
        old_indices_kept = sorted(
            [idx for idx in original_belief_dict if idx not in removed_indices]
        )
        new_idx_map = {
            old_idx: new_idx for new_idx, old_idx in enumerate(old_indices_kept)
        }

        # Assign items and last_seen times to new contiguous indices
        for i, belief in enumerate(current_belief_items):
            if i < expected_count:
                new_belief_dict[i] = belief
                # Find original index corresponding to this item's position
                original_idx = old_indices_kept[i]
                if original_idx in current_last_seen_mapping:
                    new_last_seen_dict[i] = current_last_seen_mapping[original_idx]
            else:
                logger.error(
                    "Rebuild Opponent Belief T%d: More items (%d) than expected (%d) after removal. Discarding extra. Removed: %s, Original Indices: %s",
                    self._current_game_turn,
                    len(current_belief_items),
                    expected_count,
                    removed_indices,
                    sorted(original_belief_dict.keys()),
                )
                break

        # Add UNKNOWNs if needed (e.g., opponent penalty)
        while len(new_belief_dict) < expected_count:
            new_index = len(new_belief_dict)
            new_belief_dict[new_index] = CardBucket.UNKNOWN
            new_last_seen_dict.pop(new_index, None)  # Ensure no stale timestamp
            logger.debug(
                "Agent %d rebuilding opponent belief added UNKNOWN at index %d (Expected: %d, Current: %d)",
                self.player_id,
                new_index,
                expected_count,
                len(new_belief_dict) - 1,
            )

        if len(new_belief_dict) != expected_count:
            logger.error(
                "FATAL: Opponent belief final rebuild T%d failed. Expected %d, got %d. Indices: %s. Removed: %s, Original Indices: %s",
                self._current_game_turn,
                expected_count,
                len(new_belief_dict),
                sorted(new_belief_dict.keys()),
                removed_indices,
                sorted(original_belief_dict.keys()),
            )
            # Fallback?
            # new_belief_dict = {i: CardBucket.UNKNOWN for i in range(expected_count)}
            # new_last_seen_dict = {}

        # Ensure last_seen only contains keys present in the final belief dict (redundant if logic above is correct)
        valid_last_seen = {
            k: v for k, v in new_last_seen_dict.items() if k in new_belief_dict
        }

        return new_belief_dict, valid_last_seen

    def _estimate_stockpile(self, stock_size: int) -> StockpileEstimate:
        """Estimates stockpile category based on size."""
        try:
            total_cards = 52 + self.config.cambia_rules.use_jokers
            low_threshold = max(1, total_cards // 5)
            med_threshold = max(low_threshold + 1, total_cards * 2 // 4)
        except AttributeError:  # Handle case where config might be missing during init?
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
        return GamePhase.EARLY

    def _trigger_event_decay(
        self, target_index: int, trigger_event: str, current_turn: int
    ):
        """Applies event-based decay (Levels 1 and 2) to a specific opponent index."""
        if self.memory_level == 0:
            return

        if target_index in self.opponent_belief:
            current_belief = self.opponent_belief[target_index]
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
                    "Agent %d decaying Opponent[%d] belief from %s to %s due to %s.",
                    self.player_id,
                    target_index,
                    current_belief.name,
                    decayed_category.name,
                    trigger_event,
                )
            # else: logger.debug("Agent %d decay trigger skipped for Opponent[%d]: Already %s.", self.player_id, target_index, current_belief.name)
        # else: logger.debug("Agent %d decay trigger skipped: Index %d not in opponent belief %s.", self.player_id, target_index, list(self.opponent_belief.keys()))

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
                "Agent %d applying time decay for Opponent[%d] from %s to %s.",
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

            # Return as plain tuple for direct use as dict key if needed
            # DecisionContext is added externally by the calling function (CFR worker/BR)
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
            # Return a default/error tuple? Needs careful consideration how CFR handles this.
            # Returning a consistent default might group unrelated states. Raising might be better?
            # For now, let's return a tuple indicating error.
            return ((-99,), (-99,), -1, -1, -1, -1)

    def get_potential_opponent_snap_indices(self, target_rank: str) -> List[int]:
        """Returns opponent hand indices the agent *believes* could match the target rank."""
        matching_indices = []
        try:
            # Need dummy cards to get target bucket(s)
            target_card_black = Card(rank=target_rank, suit="S")  # Use valid suit
            target_card_red = Card(rank=target_rank, suit="H")  # Use valid suit
            target_bucket_black = get_card_bucket(target_card_black)
            target_bucket_red = get_card_bucket(target_card_red)
            target_buckets = {
                target_bucket_black,
                target_bucket_red,
            }  # Use set for efficient check
        except ValueError:  # Handle invalid target_rank
            logger.error(
                "Agent %d: Invalid target_rank '%s' for get_potential_opponent_snap_indices.",
                self.player_id,
                target_rank,
            )
            return []

        for idx, belief in self.opponent_belief.items():
            if belief == CardBucket.UNKNOWN or isinstance(belief, DecayCategory):
                matching_indices.append(idx)  # Assume unknown/decayed could match
            elif isinstance(belief, CardBucket) and belief in target_buckets:
                matching_indices.append(idx)

        return matching_indices

    def clone(self) -> "AgentState":
        """Creates a deep copy of the agent state."""
        try:
            new_state = AgentState(
                player_id=self.player_id,
                opponent_id=self.opponent_id,
                memory_level=self.memory_level,
                time_decay_turns=self.time_decay_turns,
                initial_hand_size=self.initial_hand_size,
                config=self.config,  # Config can be shallow copied
            )
            # Deepcopy mutable dicts
            new_state.own_hand = copy.deepcopy(self.own_hand)
            new_state.opponent_belief = copy.deepcopy(self.opponent_belief)
            new_state.opponent_last_seen_turn = copy.deepcopy(
                self.opponent_last_seen_turn
            )
            # Copy immutable/primitive attributes
            new_state.known_discard_top_bucket = self.known_discard_top_bucket
            new_state.opponent_card_count = self.opponent_card_count
            new_state.stockpile_estimate = self.stockpile_estimate
            new_state.game_phase = self.game_phase
            new_state.cambia_caller = self.cambia_caller
            new_state._current_game_turn = self._current_game_turn
            return new_state
        except Exception as e_clone:
            logger.exception(
                "Agent %d: Error cloning agent state: %s", self.player_id, e_clone
            )
            # Re-raise or return self? Re-raising seems safer.
            raise

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
