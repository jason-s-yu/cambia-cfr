"""
src/agent_state.py

Represents an agent's subjective belief state in the game Cambia.
Handles information abstraction, memory limitations, and state updates based on observations.
"""

from typing import List, Tuple, Optional, Dict, Any, Set, Union
from dataclasses import dataclass, field
import logging
import copy
import traceback

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
)
from .config import Config
from .abstraction import get_card_bucket, decay_bucket
from src.cfr.exceptions import ObservationUpdateError

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
    king_swap_indices: Optional[Tuple[int, int]] = (
        None  # (own_idx, opp_idx) for king swap
    )


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
            # Handle case where initial_hand might be shorter than expected
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
        """
        Updates belief state based on an observation tuple.

        Raises:
            ObservationUpdateError: If belief update or reconciliation fails critically.
        """
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

        observed_own_count = observation.player_hand_sizes[self.player_id]
        observed_opp_count = observation.player_hand_sizes[self.opponent_id]
        logger.debug(
            " Observed Counts: Own=%d, Opp=%d. Current Belief Counts: Own=%d, Opp=%d",
            observed_own_count,
            observed_opp_count,
            len(self.own_hand),
            self.opponent_card_count,
        )

        # --- 1. Update Public Knowledge & Counts ---
        self.known_discard_top_bucket = get_card_bucket(observation.discard_top_card)
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

        # Store pre-snap state for potential rollback or debugging
        # Shallow copies are safe: KnownCardInfo/enums are not mutated between copy and use
        original_own_hand = dict(self.own_hand)
        original_opponent_belief = dict(self.opponent_belief)
        original_opponent_last_seen = dict(self.opponent_last_seen_turn)

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
                            " -> P%d got penalty (+%d cards)",
                            self.player_id,
                            num_penalty_cards,
                        )
                    elif snapper == self.opponent_id:
                        opponent_cards_added_count += num_penalty_cards
                        logger.debug(
                            " -> P%d got penalty (+%d cards)",
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
                            # Check who removed whose card
                            # If we are the snapper, we removed opponent's card
                            if snapper == self.player_id:
                                opponent_indices_removed.add(removed_idx)
                                logger.debug(
                                    " -> P%d snapped opp idx %d (will move)",
                                    self.player_id,
                                    removed_idx,
                                )
                            # If opponent is snapper, they removed our card
                            elif snapper == self.opponent_id:
                                own_indices_removed.add(removed_idx)
                                logger.debug(
                                    " -> P%d snapped our idx %d (will receive)",
                                    self.opponent_id,
                                    removed_idx,
                                )
            except Exception as e_snap_proc:
                # JUSTIFIED: Individual snap result processing error should not crash entire update
                logger.error(
                    "Agent %d: Error processing snap_info %s: %s",
                    self.player_id,
                    snap_info,
                    e_snap_proc,
                )

        # 2a. Handle card moves from SnapOpponentMove (if occurred *this turn*)
        # This needs to happen *before* final reconciliation, as it affects counts and indices.
        if isinstance(action, ActionSnapOpponentMove) and actor != -1:
            # The action contains the indices involved in the *move* step
            snapper_idx = actor  # The player who moved their card
            own_card_idx_moved = action.own_card_to_move_hand_index
            target_slot_idx = action.target_empty_slot_index

            if snapper_idx == self.player_id:  # We moved our card to opponent
                # Mark our card as removed *before* reconciliation
                if (
                    own_card_idx_moved in original_own_hand
                ):  # Check against pre-snap state
                    own_indices_removed.add(own_card_idx_moved)
                    logger.debug(
                        " -> P%d moved card from own idx %d (marked removed)",
                        self.player_id,
                        own_card_idx_moved,
                    )
                # Opponent count increases due to receiving card (handled by observed_opp_count later)
                # Opponent belief at target_slot_idx becomes UNKNOWN (handled later)
            elif snapper_idx == self.opponent_id:  # Opponent moved their card to us
                # Mark opponent card as removed *before* reconciliation
                if (
                    own_card_idx_moved in original_opponent_belief
                ):  # Check against pre-snap state
                    opponent_indices_removed.add(own_card_idx_moved)
                    logger.debug(
                        " -> P%d moved card from their idx %d (marked removed)",
                        self.opponent_id,
                        own_card_idx_moved,
                    )
                # Our count increases due to receiving card (handled by observed_own_count later)
                # Our hand at target_slot_idx becomes UNKNOWN (handled later)

        logger.debug(
            " Indices Removed (Post-Move Check): Own=%s, Opp=%s",
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
                " Rebuilding Own Hand (Start Size: %d, Removed: %s, Added: %d)... Expecting %d.",
                len(original_own_hand),
                own_indices_removed,
                own_cards_added_count,
                observed_own_count,
            )
            rebuilt_own_hand = self._rebuild_hand_state_new(
                original_dict=original_own_hand,
                removed_indices=own_indices_removed,
                expected_final_count=observed_own_count,
                added_placeholder_count=own_cards_added_count,
                placeholder_value=KnownCardInfo(
                    bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn
                ),
                is_own_hand=True,  # Specify this is for own hand
            )
            self.own_hand = rebuilt_own_hand  # Update directly if successful

            logger.debug(
                " Rebuilding Opp Belief (Start Size: %d, Removed: %s, Added: %d)... Expecting %d.",
                len(original_opponent_belief),
                opponent_indices_removed,
                opponent_cards_added_count,
                observed_opp_count,
            )
            rebuilt_opponent_belief, rebuilt_opponent_last_seen = (
                self._rebuild_belief_state_new(
                    original_belief_dict=original_opponent_belief,
                    original_last_seen_dict=original_opponent_last_seen,
                    removed_indices=opponent_indices_removed,
                    expected_final_count=observed_opp_count,
                    added_placeholder_count=opponent_cards_added_count,
                    placeholder_value=CardBucket.UNKNOWN,
                )
            )
            self.opponent_belief = rebuilt_opponent_belief
            self.opponent_last_seen_turn = rebuilt_opponent_last_seen
            self.opponent_card_count = (
                observed_opp_count  # Update count *after* successful reconciliation
            )

            logger.debug(" Belief Reconciliation Successful.")

        except AssertionError as e_reconcile:
            logger.critical(
                "FATAL: Agent %d: Reconciliation failed T%d: %s. State CORRUPTED. Stack:\n%s",
                self.player_id,
                self._current_game_turn,
                e_reconcile,
                traceback.format_exc(),  # Log stack trace
            )
            # Attempt to force counts to match observation to prevent index errors later.
            self.own_hand = {
                i: KnownCardInfo(
                    bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn
                )
                for i in range(observed_own_count)
            }
            self.opponent_belief = {
                i: CardBucket.UNKNOWN for i in range(observed_opp_count)
            }
            self.opponent_last_seen_turn = {}
            self.opponent_card_count = observed_opp_count
            raise ObservationUpdateError("Belief reconciliation failed") from e_reconcile

        except Exception as e_reconcile_other:
            logger.exception(
                "Agent %d: Unexpected error during hand/belief reconciliation T%d: %s. State may be inconsistent.",
                self.player_id,
                self._current_game_turn,
                e_reconcile_other,
            )
            # Attempt to force counts
            self.own_hand = {
                i: KnownCardInfo(
                    bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn
                )
                for i in range(observed_own_count)
            }
            self.opponent_belief = {
                i: CardBucket.UNKNOWN for i in range(observed_opp_count)
            }
            self.opponent_last_seen_turn = {}
            self.opponent_card_count = observed_opp_count
            raise ObservationUpdateError(
                "Unexpected reconciliation error"
            ) from e_reconcile_other

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
                        if target_idx in self.own_hand:
                            if drawn_card_from_obs and isinstance(
                                drawn_card_from_obs, Card
                            ):  # Card known from observation
                                drawn_bucket = get_card_bucket(drawn_card_from_obs)
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
                                # Drawn card is None or not Card - Expected in BR calc
                                logger.debug(
                                    " Agent %d Replace action at idx %d observed without drawn card (expected in BR). Setting belief to UNKNOWN.",
                                    self.player_id,
                                    target_idx,
                                )
                                self.own_hand[target_idx] = KnownCardInfo(
                                    bucket=CardBucket.UNKNOWN,
                                    last_seen_turn=self._current_game_turn,
                                    card=None,
                                )
                        else:
                            logger.warning(
                                " Agent %d Replace target index %d not found after reconciliation (Current: %s). State inconsistent?",
                                self.player_id,
                                target_idx,
                                list(self.own_hand.keys()),
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
                                    # Check if info is new
                                    if (
                                        self.own_hand[h_idx].bucket == CardBucket.UNKNOWN
                                        or self.own_hand[h_idx].card is None
                                    ):
                                        logger.debug(
                                            " Agent %d (%s) updated own idx %d knowledge to %s (%s).",
                                            self.player_id,
                                            type(action).__name__,
                                            h_idx,
                                            peeked_bucket.name,
                                            card,
                                        )
                                        self.own_hand[h_idx] = KnownCardInfo(
                                            bucket=peeked_bucket,
                                            last_seen_turn=self._current_game_turn,
                                            card=card,
                                        )
                                    else:  # Already knew card, just update timestamp if needed
                                        self.own_hand[h_idx].last_seen_turn = (
                                            self._current_game_turn
                                        )
                                elif (
                                    p_idx == self.opponent_id
                                    and h_idx in self.opponent_belief
                                ):
                                    # Check if info is new
                                    if self.opponent_belief[h_idx] != peeked_bucket:
                                        logger.debug(
                                            " Agent %d (%s) updated opp idx %d belief from %s to %s (%s).",
                                            self.player_id,
                                            type(action).__name__,
                                            h_idx,
                                            (
                                                self.opponent_belief[h_idx].name
                                                if isinstance(
                                                    self.opponent_belief[h_idx],
                                                    (CardBucket, DecayCategory),
                                                )
                                                else self.opponent_belief[h_idx]
                                            ),
                                            peeked_bucket.name,
                                            card,
                                        )
                                        self.opponent_belief[h_idx] = peeked_bucket
                                        self.opponent_last_seen_turn[h_idx] = (
                                            self._current_game_turn
                                        )
                                    else:  # No change in belief, just update timestamp
                                        self.opponent_last_seen_turn[h_idx] = (
                                            self._current_game_turn
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
                                    if self.opponent_belief[h_idx] != peeked_bucket:
                                        logger.debug(
                                            " Agent %d (PeekOther) updated opp idx %d belief from %s to %s (%s).",
                                            self.player_id,
                                            h_idx,
                                            (
                                                self.opponent_belief[h_idx].name
                                                if isinstance(
                                                    self.opponent_belief[h_idx],
                                                    (CardBucket, DecayCategory),
                                                )
                                                else self.opponent_belief[h_idx]
                                            ),
                                            peeked_bucket.name,
                                            card,
                                        )
                                        self.opponent_belief[h_idx] = peeked_bucket
                                        self.opponent_last_seen_turn[h_idx] = (
                                            self._current_game_turn
                                        )
                                    else:  # No change in belief, just update timestamp
                                        self.opponent_last_seen_turn[h_idx] = (
                                            self._current_game_turn
                                        )

                    elif isinstance(action, ActionAbilityBlindSwapSelect):
                        own_idx, opp_idx_target = (
                            action.own_hand_index,
                            action.opponent_hand_index,
                        )
                        # Our own card becomes unknown
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
                        # Opponent's card belief decays
                        self._trigger_event_decay(
                            target_index=opp_idx_target,
                            trigger_event="swap (blind, self initiated)",
                            current_turn=self._current_game_turn,
                        )

                    elif (
                        isinstance(action, ActionAbilityKingSwapDecision)
                        and action.perform_swap
                    ):
                        # We initiated the swap. Our card becomes unknown. Opponent card decays.
                        if observation.king_swap_indices is not None:
                            own_involved_idx, opp_involved_idx = (
                                observation.king_swap_indices
                            )
                            if own_involved_idx in self.own_hand:
                                self.own_hand[own_involved_idx] = KnownCardInfo(
                                    bucket=CardBucket.UNKNOWN,
                                    last_seen_turn=self._current_game_turn,
                                    card=None,
                                )
                                logger.debug(
                                    " Agent %d (KingSwap self) updated own idx %d to UNKNOWN.",
                                    self.player_id,
                                    own_involved_idx,
                                )
                            self._trigger_event_decay(
                                target_index=opp_involved_idx,
                                trigger_event="swap (king, self initiated)",
                                current_turn=self._current_game_turn,
                            )
                        else:
                            logger.warning(
                                " Agent %d observed self perform King Swap but king_swap_indices missing from observation.",
                                self.player_id,
                            )

                    # Snap actions affecting self (SnapOwn, SnapOpponentMove) handled by reconciliation

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
                            trigger_event="swap (blind, opponent initiated)",
                            current_turn=self._current_game_turn,
                        )
                        # Update our own card to unknown
                        if our_idx in self.own_hand:
                            # Only log if it was previously known
                            if self.own_hand[our_idx].bucket != CardBucket.UNKNOWN:
                                logger.debug(
                                    " Agent %d updating own idx %d to UNKNOWN due to opponent BlindSwap.",
                                    self.player_id,
                                    our_idx,
                                )
                            self.own_hand[our_idx] = KnownCardInfo(
                                bucket=CardBucket.UNKNOWN,
                                last_seen_turn=self._current_game_turn,
                                card=None,
                            )

                    elif (
                        isinstance(action, ActionAbilityKingSwapDecision)
                        and action.perform_swap
                    ):
                        # Opponent performed swap. Decay their involved card belief.
                        # Update our card to UNKNOWN if it was involved.
                        if observation.king_swap_indices is not None:
                            # Indices are from the actor's perspective (opponent):
                            # king_swap_indices = (actor's own_idx, actor's opp_idx)
                            # actor's opp_idx is OUR index
                            opp_involved_idx, our_involved_idx = (
                                observation.king_swap_indices
                            )
                            self._trigger_event_decay(
                                target_index=opp_involved_idx,
                                trigger_event="swap (king, opponent initiated)",
                                current_turn=self._current_game_turn,
                            )
                            if our_involved_idx in self.own_hand:
                                if (
                                    self.own_hand[our_involved_idx].bucket
                                    != CardBucket.UNKNOWN
                                ):
                                    logger.debug(
                                        " Agent %d updating own idx %d to UNKNOWN due to opponent KingSwap.",
                                        self.player_id,
                                        our_involved_idx,
                                    )
                                self.own_hand[our_involved_idx] = KnownCardInfo(
                                    bucket=CardBucket.UNKNOWN,
                                    last_seen_turn=self._current_game_turn,
                                    card=None,
                                )
                        else:
                            logger.warning(
                                " Agent %d observed opponent King Swap but king_swap_indices missing from observation.",
                                self.player_id,
                            )

                    # Snap actions affecting opponent (SnapOwn, SnapOpponentMove) handled by reconciliation

                    # Opponent discard/peek doesn't reveal info unless the card was already known to us.

            except Exception as e_action_proc:
                # JUSTIFIED: Action processing error should not crash entire update
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

        # --- 6. Final Sanity Check ---
        # Ensure internal counts match observed counts after all updates
        final_own_count = len(self.own_hand)
        final_opp_count = len(self.opponent_belief)
        if final_own_count != observed_own_count or final_opp_count != observed_opp_count:
            logger.error(
                "Agent %d T%d: FINAL STATE COUNT MISMATCH! Own: Expected %d, Got %d. Opp: Expected %d, Got %d. State likely corrupted.",
                self.player_id,
                self._current_game_turn,
                observed_own_count,
                final_own_count,
                observed_opp_count,
                final_opp_count,
            )
            # Attempt recovery? Reset counts? For now, log error.

        logger.debug(" Update Complete. State: %s", self)

    def _rebuild_hand_state_new(
        self,
        original_dict: Dict[int, Any],  # Can be KnownCardInfo or belief
        removed_indices: Set[int],
        expected_final_count: int,
        added_placeholder_count: int,
        placeholder_value: Any,
        is_own_hand: bool = False,
    ) -> Dict[int, Any]:
        """
        Rebuilds a state dictionary (own_hand or opponent_belief) ensuring
        contiguous indices [0..N-1], handling removals and additions.
        Handles count mismatches by prioritizing existing items or adding placeholders.
        """
        new_dict: Dict[int, Any] = {}
        items_to_keep = []
        original_indices_sorted = sorted(original_dict.keys())

        # 1. Collect items that are kept
        for idx in original_indices_sorted:
            if idx not in removed_indices:
                items_to_keep.append(original_dict[idx])

        # 2. Add new placeholder items
        items_to_keep.extend(
            [copy.copy(placeholder_value) for _ in range(added_placeholder_count)]
        )

        # 3. Assign to new dictionary with contiguous indices, handling count mismatches
        current_total_items = len(items_to_keep)
        hand_type_str = "Own Hand" if is_own_hand else "Opp Belief"

        if current_total_items == expected_final_count:
            # Ideal case: counts match
            for new_idx, item in enumerate(items_to_keep):
                new_dict[new_idx] = item
        elif current_total_items > expected_final_count:
            # More items calculated than expected: keep the first 'expected' count
            logger.info(
                "Rebuild %s T%d: More items (%d) than expected (%d) after adds/removes. Keeping first %d. Removed: %s, Added: %d, Original Indices: %s",
                hand_type_str,
                self._current_game_turn,
                current_total_items,
                expected_final_count,
                expected_final_count,
                removed_indices,
                added_placeholder_count,
                original_indices_sorted,
            )
            for new_idx in range(expected_final_count):
                new_dict[new_idx] = items_to_keep[new_idx]
        else:  # current_total_items < expected_final_count
            # Fewer items calculated than expected: add placeholders to reach expected count
            logger.info(
                "Rebuild %s T%d: Fewer items (%d) than expected (%d) after adds/removes. Adding %d placeholders. Removed: %s, Added: %d, Original Indices: %s",
                hand_type_str,
                self._current_game_turn,
                current_total_items,
                expected_final_count,
                expected_final_count - current_total_items,
                removed_indices,
                added_placeholder_count,
                original_indices_sorted,
            )
            for new_idx, item in enumerate(items_to_keep):
                new_dict[new_idx] = item
            for new_idx in range(current_total_items, expected_final_count):
                new_dict[new_idx] = copy.copy(placeholder_value)

        # 4. Final assertion for internal consistency (should always pass if logic above is sound)
        final_count = len(new_dict)
        assert final_count == expected_final_count, (
            f"Rebuild {hand_type_str} T{self._current_game_turn} FINAL COUNT CHECK FAILED! Expected {expected_final_count}, got {final_count}. "
            f"Original indices: {original_indices_sorted}, Removed: {removed_indices}, Added: {added_placeholder_count}. Final keys: {sorted(new_dict.keys())}"
        )

        return new_dict

    def _rebuild_belief_state_new(
        self,
        original_belief_dict: Dict[int, Union[CardBucket, DecayCategory]],
        original_last_seen_dict: Dict[int, int],
        removed_indices: Set[int],
        expected_final_count: int,
        added_placeholder_count: int,
        placeholder_value: Union[CardBucket, DecayCategory],
    ) -> Tuple[Dict[int, Union[CardBucket, DecayCategory]], Dict[int, int]]:
        """Rebuilds opponent_belief and preserves/transfers last_seen timestamps, handling count mismatches."""
        # Use the generic _rebuild_hand_state_new logic by wrapping/unwrapping
        # We need to preserve the original indices to map timestamps correctly

        # Create a temporary dict mapping original index to (belief, timestamp)
        temp_original_dict_with_ts = {}
        for idx, belief in original_belief_dict.items():
            ts = original_last_seen_dict.get(idx)
            temp_original_dict_with_ts[idx] = (belief, ts)

        # Use the generic rebuild logic
        rebuilt_temp_dict = self._rebuild_hand_state_new(
            original_dict=temp_original_dict_with_ts,
            removed_indices=removed_indices,
            expected_final_count=expected_final_count,
            added_placeholder_count=added_placeholder_count,
            placeholder_value=(
                placeholder_value,
                None,
            ),  # Placeholder includes None timestamp
            is_own_hand=False,  # Specify this is for opponent belief
        )

        # Unpack the results back into separate belief and timestamp dictionaries
        new_belief: Dict[int, Union[CardBucket, DecayCategory]] = {}
        new_last_seen: Dict[int, int] = {}
        for new_idx, (belief, ts) in rebuilt_temp_dict.items():
            new_belief[new_idx] = belief
            if ts is not None:
                new_last_seen[new_idx] = ts

        return new_belief, new_last_seen

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

        # Ensure index exists before attempting decay
        if target_index not in self.opponent_belief:
            logger.debug(
                " Agent %d Decay trigger skipped: Index %d not in opponent belief %s.",
                self.player_id,
                target_index,
                list(self.opponent_belief.keys()),
            )
            return

        current_belief = self.opponent_belief[target_index]
        # Only decay if current belief is specific (CardBucket, not UNKNOWN or DecayCategory)
        if (
            isinstance(current_belief, CardBucket)
            and current_belief != CardBucket.UNKNOWN
        ):
            decayed_category = decay_bucket(current_belief)
            # Only update if decay actually changes the category
            if decayed_category != current_belief:
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
            )  # Check not already decayed category
            and self.opponent_belief[idx] != CardBucket.UNKNOWN
        ]

        for idx in indices_to_decay:
            current_belief = self.opponent_belief[idx]
            decayed_category = decay_bucket(current_belief)
            if decayed_category != current_belief:  # Only decay if it changes category
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
            # Ensure own_hand indices are contiguous 0..N-1
            own_hand_buckets = [CardBucket.UNKNOWN.value] * len(self.own_hand)
            for i, info in self.own_hand.items():
                if 0 <= i < len(own_hand_buckets):
                    own_hand_buckets[i] = info.bucket.value
                else:
                    logger.error(
                        "Infoset Key: Own hand index %d out of bounds (size %d)",
                        i,
                        len(self.own_hand),
                    )
            own_hand_tuple = tuple(own_hand_buckets)

            # 2. Opponent Beliefs (Tuple indexed 0..N-1)
            opp_belief_values = [
                CardBucket.UNKNOWN.value
            ] * self.opponent_card_count  # Use reconciled count
            for i, belief in self.opponent_belief.items():
                if 0 <= i < self.opponent_card_count:
                    opp_belief_values[i] = (
                        belief.value
                        if hasattr(belief, "value")
                        else CardBucket.UNKNOWN.value
                    )
                else:
                    logger.error(
                        "Infoset Key: Opponent belief index %d out of bounds (count %d)",
                        i,
                        self.opponent_card_count,
                    )
            opp_belief_tuple = tuple(opp_belief_values)

            # 3. Other components
            opp_count = self.opponent_card_count
            discard_top_val = self.known_discard_top_bucket.value
            stockpile_est_val = self.stockpile_estimate.value
            game_phase_val = self.game_phase.value
            # DecisionContext is added externally by the calling function

            key_tuple = (
                own_hand_tuple,
                opp_belief_tuple,
                opp_count,
                discard_top_val,
                stockpile_est_val,
                game_phase_val,
            )
            return key_tuple
        except Exception as e_key:
            logger.exception(
                "Agent %d: Error generating infoset key tuple T%d: %s. Current State: %s",
                self.player_id,
                self._current_game_turn,
                e_key,
                self,
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

        # Use the reconciled opponent_card_count
        for idx in range(self.opponent_card_count):
            belief = self.opponent_belief.get(idx)  # Use get() for safety
            if belief is None:  # Should not happen if rebuild is correct
                logger.warning(
                    "Agent %d: Missing belief for opponent index %d.", self.player_id, idx
                )
                matching_indices.append(
                    idx
                )  # Assume could match if missing? Or error? Assume match for now.
            elif belief == CardBucket.UNKNOWN or isinstance(belief, DecayCategory):
                matching_indices.append(idx)  # Assume unknown/decayed *could* match
            elif isinstance(belief, CardBucket) and belief in target_buckets:
                matching_indices.append(idx)

        return matching_indices

    def clone(self) -> "AgentState":
        """Creates a copy of the agent state. Uses manual copy instead of deepcopy
        for performance (called millions of times during traversals)."""
        # own_hand: Dict[int, KnownCardInfo] — copy KnownCardInfo objects (mutable dataclass)
        own_hand_copy = {
            k: KnownCardInfo(bucket=v.bucket, last_seen_turn=v.last_seen_turn, card=v.card)
            for k, v in self.own_hand.items()
        }
        # opponent_belief/opponent_last_seen_turn: values are enums/ints (immutable)
        new_state = AgentState(
            player_id=self.player_id,
            opponent_id=self.opponent_id,
            memory_level=self.memory_level,
            time_decay_turns=self.time_decay_turns,
            initial_hand_size=self.initial_hand_size,
            config=self.config,
            own_hand=own_hand_copy,
            opponent_belief=dict(self.opponent_belief),
            opponent_last_seen_turn=dict(self.opponent_last_seen_turn),
            known_discard_top_bucket=self.known_discard_top_bucket,
            opponent_card_count=self.opponent_card_count,
            stockpile_estimate=self.stockpile_estimate,
            game_phase=self.game_phase,
            cambia_caller=self.cambia_caller,
            _current_game_turn=self._current_game_turn,
        )
        return new_state

    def __str__(self) -> str:
        """Provides a concise string representation of the agent's belief state."""
        try:
            own_hand_repr = {}
            for i in range(len(self.own_hand)):  # Iterate based on size
                info = self.own_hand.get(i)
                own_hand_repr[i] = info.bucket.name if info else "MISSING"

            opp_belief_repr = {}
            for i in range(self.opponent_card_count):  # Iterate based on count
                belief = self.opponent_belief.get(i, CardBucket.UNKNOWN)
                opp_belief_repr[i] = (
                    belief.name if hasattr(belief, "name") else repr(belief)
                )

            return (
                f"AgentState(P{self.player_id}, T:{self._current_game_turn}, Phase:{self.game_phase.name}, "
                f"OH({len(self.own_hand)}):{own_hand_repr}, OB({self.opponent_card_count}):{opp_belief_repr}, "
                f"Disc:{self.known_discard_top_bucket.name}, Stock:{self.stockpile_estimate.name}, Cambia:{self.cambia_caller})"
            )
        except Exception as e_str:
            # JUSTIFIED: String representation should not crash
            logger.error(
                "Agent %d: Error generating string representation: %s",
                self.player_id,
                e_str,
            )
            return f"AgentState(P{self.player_id}, Error)"
