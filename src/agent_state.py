# src/agent_state.py
from typing import List, Tuple, Optional, Dict, Any, Set, Union
from dataclasses import dataclass, field
import time # For potential time-based decay, though turn-based is preferred
import logging
import copy # For cloning

from .constants import (
    CardBucket, DecayCategory, GamePhase, StockpileEstimate,
    GameAction, ActionReplace, ActionDiscard, ActionCallCambia,
    ActionAbilityPeekOwnSelect, ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect, ActionAbilityKingLookSelect, ActionAbilityKingSwapDecision,
    ActionSnapOwn, ActionSnapOpponent, ActionSnapOpponentMove, NUM_PLAYERS
)
from .card import Card
from .config import Config
from .abstraction import get_card_bucket, decay_bucket
# Avoid importing GameState directly if possible to enforce information limits
# from .game_engine import CambiaGameState, PlayerState
from .utils import InfosetKey

logger = logging.getLogger(__name__)

@dataclass
class KnownCardInfo:
    """Stores info about a card location where the agent knows the card."""
    bucket: CardBucket
    # Track turn for decay levels 1 & 2
    last_seen_turn: int = 0
    card: Optional[Card] = None # Store actual card for perfect recall/debugging

    def __post_init__(self):
         # Ensure bucket is always a CardBucket enum member
         if not isinstance(self.bucket, CardBucket):
              logger.warning(f"KnownCardInfo initialized with non-Enum bucket: {self.bucket}. Setting UNKNOWN.")
              self.bucket = CardBucket.UNKNOWN

@dataclass
class AgentObservation:
     """ Minimum information passed down CFR about state changes. """
     acting_player: int
     action: Optional[GameAction] # The actual action object taken
     # Publicly visible results:
     discard_top_card: Optional[Card]
     player_hand_sizes: List[int]
     stockpile_size: int
     # Information specific to the observing player:
     drawn_card: Optional[Card] = None # If observing player drew
     peeked_cards: Optional[Dict[Tuple[int, int], Card]] = None # {(player_idx, hand_idx): Card}
     # Snap related info (Now detailed)
     snap_results: List[Dict[str, Any]] = field(default_factory=list) # List of dicts from GameState.snap_results_log
     # General state info
     did_cambia_get_called: bool = False
     who_called_cambia: Optional[int] = None
     is_game_over: bool = False
     current_turn: int = 0 # Turn number *after* action

@dataclass
class AgentState:
    """
    Represents the agent's subjective belief about the game state,
    incorporating abstractions and memory limitations.
    This object is updated based on AgentObservation, NOT direct GameState access.
    """
    player_id: int
    opponent_id: int
    memory_level: int
    time_decay_turns: int # Threshold for Level 2 decay
    initial_hand_size: int # From config/rules
    config: Config # Store config for access to rules etc.

    # --- Core Beliefs ---
    # Use dict where key is logical hand index [0, N-1]
    own_hand: Dict[int, KnownCardInfo] = field(default_factory=dict)
    # Opponent belief: Bucket, DecayCategory, or UNKNOWN
    opponent_belief: Dict[int, Union[CardBucket, DecayCategory]] = field(default_factory=dict)
    opponent_last_seen_turn: Dict[int, int] = field(default_factory=dict) # For decay

    # --- Public knowledge ---
    known_discard_top_bucket: CardBucket = CardBucket.UNKNOWN
    opponent_card_count: int = 0
    stockpile_estimate: StockpileEstimate = StockpileEstimate.HIGH
    game_phase: GamePhase = GamePhase.START
    cambia_caller: Optional[int] = None

    # --- Internal tracking ---
    _current_game_turn: int = 0 # Track overall game turn number from observations


    def initialize(self, initial_observation: AgentObservation, initial_hand: List[Card], initial_peek_indices: Tuple[int, ...]):
        """Initialize belief state at the very start of the game."""
        self.opponent_card_count = initial_observation.player_hand_sizes[self.opponent_id]
        self.stockpile_estimate = self._estimate_stockpile(initial_observation.stockpile_size)
        self.game_phase = self._estimate_game_phase(initial_observation.stockpile_size, None, 0)
        self.known_discard_top_bucket = get_card_bucket(initial_observation.discard_top_card)
        self.cambia_caller = None
        self._current_game_turn = 0

        # Initialize own hand knowledge
        self.own_hand = {}
        for i, card in enumerate(initial_hand):
            known = i in initial_peek_indices
            bucket = get_card_bucket(card) if known else CardBucket.UNKNOWN
            if known: logger.debug(f"Agent {self.player_id} initial peek: Index {i} is {bucket.name}")
            self.own_hand[i] = KnownCardInfo(bucket=bucket, last_seen_turn=0, card=card if known else None)

        # Initialize opponent belief as UNKNOWN
        self.opponent_belief = {i: CardBucket.UNKNOWN for i in range(self.opponent_card_count)}
        self.opponent_last_seen_turn = {}

        logger.debug(f"Agent {self.player_id} initialized (Turn {self._current_game_turn}). Own Hand({len(self.own_hand)}): { {k: v.bucket.name for k,v in self.own_hand.items()} }. Opponent({self.opponent_card_count}): { {k: v.name for k,v in self.opponent_belief.items()} }")


    def update(self, observation: AgentObservation):
        """Updates belief state based on an observation tuple."""
        if observation.current_turn < self._current_game_turn:
            logger.warning(f"Agent {self.player_id} received observation for past turn {observation.current_turn} (current: {self._current_game_turn}). Skipping.")
            return
        self._current_game_turn = observation.current_turn

        # --- 1. Update Public Knowledge & Counts ---
        last_discard_card = observation.discard_top_card
        self.known_discard_top_bucket = get_card_bucket(last_discard_card)
        observed_opp_count = observation.player_hand_sizes[self.opponent_id]
        observed_own_count = observation.player_hand_sizes[self.player_id]
        self.stockpile_estimate = self._estimate_stockpile(observation.stockpile_size)
        if observation.did_cambia_get_called and self.cambia_caller is None:
            self.cambia_caller = observation.who_called_cambia
        self.game_phase = self._estimate_game_phase(observation.stockpile_size, self.cambia_caller, self._current_game_turn)

        # --- 2. Process Snap Results & Belief Updates ---
        action = observation.action
        actor = observation.acting_player
        snap_results = observation.snap_results

        own_indices_removed: Set[int] = set()
        opponent_indices_removed: Set[int] = set()

        for snap_info in snap_results:
            snapper = snap_info.get('snapper')
            success = snap_info.get('success', False)
            penalty = snap_info.get('penalty', False)
            snapped_card = snap_info.get('snapped_card') # Actual Card object or None

            if snapper == self.player_id: # Our snap attempt
                if success:
                    if "removed_own_index" in snap_info and snap_info["removed_own_index"] is not None:
                        removed_idx = snap_info["removed_own_index"]
                        own_indices_removed.add(removed_idx)
                        logger.debug(f"Agent {self.player_id} removing own index {removed_idx} from successful SnapOwn.")
                    elif "removed_opponent_index" in snap_info and snap_info["removed_opponent_index"] is not None:
                        # We successfully snapped opponent's card.
                        removed_opp_idx = snap_info["removed_opponent_index"]
                        opponent_indices_removed.add(removed_opp_idx)
                        # Our belief about this slot becomes UNKNOWN *after* we move a card (handled later by reconciliation + move action)
                        # Mark for potential UNKNOWN update during reconciliation? Or just let reconciliation handle it.
                        logger.debug(f"Agent {self.player_id} removing opponent index {removed_opp_idx} from successful SnapOpponent.")
                        if removed_opp_idx in self.opponent_belief:
                             self.opponent_belief[removed_opp_idx] = CardBucket.UNKNOWN
                             self.opponent_last_seen_turn.pop(removed_opp_idx, None)
                             logger.debug(f"Agent {self.player_id} updating belief at opponent index {removed_opp_idx} to UNKNOWN after successful SnapOpponent.")

                elif penalty:
                    logger.debug(f"Agent {self.player_id} received penalty from snap.")
                    # Hand size increase handled by reconciliation

            elif snapper == self.opponent_id: # Opponent's snap attempt
                if success:
                    if "removed_own_index" in snap_info and snap_info["removed_own_index"] is not None:
                        # Opponent successfully snapped THEIR OWN card
                        removed_opp_idx = snap_info["removed_own_index"] # This index is relative to opponent's hand
                        opponent_indices_removed.add(removed_opp_idx)
                        # We now know this card is gone. Belief about this *location* becomes UNKNOWN.
                        if removed_opp_idx in self.opponent_belief:
                            self.opponent_belief[removed_opp_idx] = CardBucket.UNKNOWN
                            self.opponent_last_seen_turn.pop(removed_opp_idx, None)
                            logger.debug(f"Agent {self.player_id} updating opponent belief at {removed_opp_idx} to UNKNOWN due to successful opponent SnapOwn.")
                        logger.debug(f"Agent {self.player_id} observing opponent removing index {removed_opp_idx} from successful SnapOwn.")
                    elif "removed_opponent_index" in snap_info and snap_info["removed_opponent_index"] is not None:
                        # Opponent successfully snapped OUR card
                        removed_own_idx = snap_info["removed_opponent_index"] # This index is relative to OUR hand
                        own_indices_removed.add(removed_own_idx)
                        # Opponent moves one of their cards here. Our belief about this *slot* becomes UNKNOWN.
                        # This is handled by reconciliation (removing card) and the ActionSnapOpponentMove adding an UNKNOWN slot.
                        logger.debug(f"Agent {self.player_id} observing own index {removed_own_idx} removed by opponent SnapOpponent.")

                elif penalty:
                    logger.debug(f"Agent {self.player_id} observed opponent penalty from snap.")
                    # Opponent hand size increases (reconciliation)

        # Apply removals *before* reconciliation (adjusts indices for reconciliation)
        if own_indices_removed or opponent_indices_removed:
            self._apply_snap_removals(own_indices_removed, opponent_indices_removed)

        # --- 3. Reconcile Hand Sizes/Indices ---
        # Must happen *after* applying known removals but *before* processing main action effects
        self._reconcile_own_hand_indices(observed_own_count)
        self._reconcile_opponent_belief_indices(observed_opp_count)
        # Update opponent card count state *after* reconciliation
        self.opponent_card_count = observed_opp_count

        # --- 4. Updates based on Main Action ---
        if action:
            if actor == self.player_id:
                if isinstance(action, ActionReplace):
                    if observation.drawn_card:
                        target_idx = action.target_hand_index
                        drawn_bucket = get_card_bucket(observation.drawn_card)
                        if target_idx in self.own_hand:
                            self.own_hand[target_idx] = KnownCardInfo(bucket=drawn_bucket, last_seen_turn=self._current_game_turn, card=observation.drawn_card)
                            logger.debug(f"Agent {self.player_id} updated own hand index {target_idx} to {drawn_bucket.name} after replace.")
                        else: logger.warning(f"Agent {self.player_id} Replace target index {target_idx} invalid (Current indices: {list(self.own_hand.keys())}).")
                    else: logger.warning("Replace action observed for self, but no drawn card in observation.")

                elif isinstance(action, ActionDiscard):
                     # Discarding drawn card, no belief update needed unless ability triggers look/swap
                     pass

                elif isinstance(action, ActionAbilityPeekOwnSelect):
                    if observation.peeked_cards:
                        for (p_idx, h_idx), card in observation.peeked_cards.items():
                            if p_idx == self.player_id and h_idx in self.own_hand:
                                peeked_bucket = get_card_bucket(card)
                                self.own_hand[h_idx] = KnownCardInfo(bucket=peeked_bucket, last_seen_turn=self._current_game_turn, card=card)
                                logger.debug(f"Agent {self.player_id} peeked own index {h_idx}, saw {peeked_bucket.name}.")

                elif isinstance(action, ActionAbilityPeekOtherSelect):
                    if observation.peeked_cards:
                        for (p_idx, h_idx), card in observation.peeked_cards.items():
                            if p_idx == self.opponent_id and h_idx in self.opponent_belief:
                                peeked_bucket = get_card_bucket(card)
                                self.opponent_belief[h_idx] = peeked_bucket
                                self.opponent_last_seen_turn[h_idx] = self._current_game_turn
                                logger.debug(f"Agent {self.player_id} peeked opponent index {h_idx}, saw {peeked_bucket.name}.")
                            # else: logger.warning(f"Peeked opponent index {h_idx} not in current belief keys: {list(self.opponent_belief.keys())}")

                elif isinstance(action, ActionAbilityBlindSwapSelect):
                    own_idx, opp_idx = action.own_hand_index, action.opponent_hand_index
                    if own_idx in self.own_hand: self.own_hand[own_idx] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn)
                    self._trigger_event_decay(target_index=opp_idx, trigger_event="swap (blind)", current_turn=self._current_game_turn)

                elif isinstance(action, ActionAbilityKingLookSelect):
                    if observation.peeked_cards:
                        for (p_idx, h_idx), card in observation.peeked_cards.items():
                            if p_idx == self.opponent_id and h_idx in self.opponent_belief:
                                peeked_bucket = get_card_bucket(card)
                                self.opponent_belief[h_idx] = peeked_bucket
                                self.opponent_last_seen_turn[h_idx] = self._current_game_turn
                                logger.debug(f"Agent {self.player_id} looked (King) at opponent index {h_idx}, saw {peeked_bucket.name}.")
                            elif p_idx == self.player_id and h_idx in self.own_hand:
                                peeked_bucket = get_card_bucket(card)
                                self.own_hand[h_idx] = KnownCardInfo(bucket=peeked_bucket, last_seen_turn=self._current_game_turn, card=card)
                                logger.debug(f"Agent {self.player_id} looked (King) at own index {h_idx}, saw {peeked_bucket.name}.")

                elif isinstance(action, ActionAbilityKingSwapDecision) and action.perform_swap:
                    # Need involved indices from observation's peeked_cards (if available from Look step)
                    if observation.peeked_cards:
                        involved_indices = list(observation.peeked_cards.keys())
                        if len(involved_indices) == 2:
                            (p1, idx1), (p2, idx2) = involved_indices
                            own_idx, opp_idx = (-1, -1)
                            if p1 == self.player_id and p2 == self.opponent_id: own_idx, opp_idx = idx1, idx2
                            elif p2 == self.player_id and p1 == self.opponent_id: own_idx, opp_idx = idx2, idx1

                            if own_idx != -1 and opp_idx != -1:
                                if own_idx in self.own_hand: self.own_hand[own_idx] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn)
                                # Belief about opponent card becomes UNKNOWN after swap
                                self._trigger_event_decay(target_index=opp_idx, trigger_event="swap (king)", current_turn=self._current_game_turn)
                            else: logger.warning("Could not determine indices for King Swap decay.")
                        else: logger.warning("Incorrect peeked_cards format for King Swap decay.")
                    else: logger.warning("Missing peeked_cards info for King Swap decay.")

                elif isinstance(action, ActionSnapOpponentMove):
                    # We successfully snapped opponent, then moved one of our cards.
                    # Reconciliation handled count decrease. Moving card removal is implicit in reconciliation.
                    # The opponent gained a card (our moved card), belief is UNKNOWN. Reconciliation handles this.
                    pass

            elif actor == self.opponent_id:
                if isinstance(action, ActionReplace):
                    target_idx = action.target_hand_index # Opponent's index
                    self._trigger_event_decay(target_index=target_idx, trigger_event="replace", current_turn=self._current_game_turn)
                elif isinstance(action, ActionDiscard):
                     pass # No direct belief update unless ability triggers look/swap affecting us
                elif isinstance(action, ActionAbilityBlindSwapSelect):
                    opp_own_idx = action.own_hand_index # Index in opponent's hand
                    our_idx = action.opponent_hand_index # Index in our hand
                    self._trigger_event_decay(target_index=opp_own_idx, trigger_event="swap (blind)", current_turn=self._current_game_turn)
                    if our_idx in self.own_hand: self.own_hand[our_idx] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn)
                elif isinstance(action, ActionAbilityKingSwapDecision) and action.perform_swap:
                    # Opponent performed swap. We don't know which cards unless we were peeked.
                    # If we weren't peeked, we can only apply decay if we guess the indices? Risky.
                    # Safest is to assume no index-specific decay unless we observed the peek.
                    if observation.peeked_cards:
                         involved_indices = list(observation.peeked_cards.keys())
                         our_involved_idx = -1
                         opp_involved_idx = -1
                         for p_idx, h_idx in involved_indices:
                             if p_idx == self.player_id: our_involved_idx = h_idx
                             elif p_idx == self.opponent_id: opp_involved_idx = h_idx

                         if our_involved_idx != -1 and our_involved_idx in self.own_hand:
                              self.own_hand[our_involved_idx] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn)
                         if opp_involved_idx != -1:
                              self._trigger_event_decay(target_index=opp_involved_idx, trigger_event="swap (king)", current_turn=self._current_game_turn)
                    else:
                         logger.debug("Opponent King Swap observed without peek info, cannot decay specific indices.")

                elif isinstance(action, ActionSnapOpponentMove):
                    # Opponent successfully snapped one of THEIR cards, then moved one to US.
                    # Reconciliation handled our count increase (added UNKNOWN).
                    # Opponent count remains same, but the moved card's original slot becomes UNKNOWN.
                    # Need original index from snap results log? Move action doesn't contain it.
                    # Handled by the opponent SnapOwn success case earlier.
                    pass

        # --- 5. Apply Time Decay (Level 2) ---
        if self.memory_level == 2:
            self._apply_time_decay(self._current_game_turn)


    def _reconcile_own_hand_indices(self, expected_count: int):
        """ Adjusts own_hand dict to match expected count, keeping indices 0 to N-1. """
        current_indices = set(self.own_hand.keys())
        current_count = len(current_indices)
        needs_rebuild = False

        if current_count == expected_count:
            # Quick check for contiguous indices
            if not all(i in current_indices for i in range(expected_count)):
                needs_rebuild = True
                logger.warning(f"Own hand has correct count ({expected_count}) but non-contiguous indices: {sorted(list(current_indices))}. Rebuilding.")
        elif current_count < expected_count:
            # Add new UNKNOWN entries (penalty/opponent move)
            added = 0
            new_indices = set(current_indices)
            for i in range(expected_count): # Ensure indices 0..expected-1 exist
                 if i not in current_indices:
                      self.own_hand[i] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn)
                      new_indices.add(i)
                      added += 1
            if added != (expected_count - current_count):
                 logger.error(f"Logic error in reconcile own hand (add). Added {added}, needed {expected_count - current_count}. Current Indices: {current_indices}, New: {new_indices}")
                 needs_rebuild = True # Force rebuild on error
            else:
                logger.debug(f"Agent {self.player_id} reconciled own hand: Added {added} UNKNOWN slots to reach {expected_count}.")
            current_count = len(self.own_hand) # Update count

        elif current_count > expected_count:
            # Remove highest indices first (most likely removals)
            removed_count = 0
            sorted_indices = sorted(list(current_indices), reverse=True)
            for idx in sorted_indices:
                if len(self.own_hand) > expected_count:
                    self.own_hand.pop(idx, None)
                    removed_count += 1
                else: break
            if removed_count > 0:
                logger.debug(f"Agent {self.player_id} reconciled own hand: Removed {removed_count} highest index slots to reach {expected_count}.")
            needs_rebuild = True # Always rebuild after removing higher indices to ensure contiguity
            current_count = len(self.own_hand)

        # Ensure indices are contiguous after potential changes
        if needs_rebuild or current_count != expected_count:
             if current_count != expected_count: logger.warning(f"Own hand reconciliation needed rebuild. Expected {expected_count}, got {current_count}. State before: {sorted(list(current_indices))}")
             current_items = sorted(self.own_hand.items())
             self.own_hand = {i: item[1] for i, item in enumerate(current_items)}
             # Final adjustment if size is still wrong (should not happen ideally)
             while len(self.own_hand) < expected_count: self.own_hand[len(self.own_hand)] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn); logger.error("Added missing UNKNOWN in final own hand rebuild.")
             while len(self.own_hand) > expected_count: self.own_hand.pop(len(self.own_hand) - 1); logger.error("Removed excess card in final own hand rebuild.")

        # Final check
        if len(self.own_hand) != expected_count or not all(i in self.own_hand for i in range(expected_count)):
             logger.error(f"FATAL: Own hand reconciliation failed. Expected {expected_count}, got {len(self.own_hand)}. Indices: {sorted(list(self.own_hand.keys()))}")


    def _reconcile_opponent_belief_indices(self, expected_count: int):
        """ Adjust opponent_belief dict to match expected count, keeping indices 0 to N-1. """
        current_indices = set(self.opponent_belief.keys())
        current_count = len(current_indices)
        needs_rebuild = False

        if current_count == expected_count:
             if not all(i in current_indices for i in range(expected_count)):
                 needs_rebuild = True
                 logger.warning(f"Opponent belief has correct count ({expected_count}) but non-contiguous indices: {sorted(list(current_indices))}. Rebuilding.")
        elif current_count < expected_count:
            # Add new UNKNOWN entries
            added = 0
            new_indices = set(current_indices)
            for i in range(expected_count):
                 if i not in current_indices:
                      self.opponent_belief[i] = CardBucket.UNKNOWN
                      self.opponent_last_seen_turn.pop(i, None) # Clear timestamp for new slot
                      new_indices.add(i)
                      added += 1
            if added != (expected_count - current_count):
                 logger.error(f"Logic error in reconcile opponent belief (add). Added {added}, needed {expected_count - current_count}. Current Indices: {current_indices}, New: {new_indices}")
                 needs_rebuild = True
            else:
                 logger.debug(f"Agent {self.player_id} reconciled opponent belief: Added {added} UNKNOWN slots to reach {expected_count}.")
            current_count = len(self.opponent_belief)

        elif current_count > expected_count:
            # Remove highest indices
            removed_count = 0
            sorted_indices = sorted(list(current_indices), reverse=True)
            for idx in sorted_indices:
                if len(self.opponent_belief) > expected_count:
                    self.opponent_belief.pop(idx, None)
                    self.opponent_last_seen_turn.pop(idx, None)
                    removed_count += 1
                else: break
            if removed_count > 0:
                 logger.debug(f"Agent {self.player_id} reconciled opponent belief: Removed {removed_count} highest index slots to reach {expected_count}.")
            needs_rebuild = True
            current_count = len(self.opponent_belief)

        # Ensure indices are contiguous
        if needs_rebuild or current_count != expected_count:
             if current_count != expected_count: logger.warning(f"Opponent belief reconciliation needed rebuild. Expected {expected_count}, got {current_count}. State before: {sorted(list(current_indices))}")
             current_beliefs = sorted(self.opponent_belief.items())
             current_last_seen = dict(self.opponent_last_seen_turn) # Copy
             self.opponent_belief = {i: item[1] for i, item in enumerate(current_beliefs)}
             self.opponent_last_seen_turn = {i: current_last_seen[item[0]] for i, item in enumerate(current_beliefs) if item[0] in current_last_seen}
             # Add/remove UNKNOWNs if needed
             while len(self.opponent_belief) < expected_count: self.opponent_belief[len(self.opponent_belief)] = CardBucket.UNKNOWN; logger.error("Added missing UNKNOWN in final opp belief rebuild.")
             while len(self.opponent_belief) > expected_count: self.opponent_belief.pop(len(self.opponent_belief) - 1); self.opponent_last_seen_turn.pop(len(self.opponent_belief), None); logger.error("Removed excess belief in final opp belief rebuild.")


        # Final check
        if len(self.opponent_belief) != expected_count or not all(i in self.opponent_belief for i in range(expected_count)):
             logger.error(f"FATAL: Opponent belief reconciliation failed. Expected {expected_count}, got {len(self.opponent_belief)}. Indices: {sorted(list(self.opponent_belief.keys()))}")


    def _apply_snap_removals(self, own_indices_removed: Set[int], opponent_indices_removed: Set[int]):
         """ Removes cards from belief dictionaries due to successful snaps. Does NOT rebuild index map (reconciliation does that). """
         if not own_indices_removed and not opponent_indices_removed: return

         # Process own hand removals
         if own_indices_removed:
              logger.debug(f"Agent {self.player_id} applying own snap removals for indices: {own_indices_removed}")
              for idx in own_indices_removed:
                   self.own_hand.pop(idx, None) # Remove directly

         # Process opponent hand removals
         if opponent_indices_removed:
              logger.debug(f"Agent {self.player_id} applying opponent snap removals for indices: {opponent_indices_removed}")
              for idx in opponent_indices_removed:
                   self.opponent_belief.pop(idx, None) # Remove directly
                   self.opponent_last_seen_turn.pop(idx, None)


    def _estimate_stockpile(self, stock_size: int) -> StockpileEstimate:
        """Estimates stockpile category based on size."""
        # Use rule-based estimate
        total_cards = 52 + (self.config.cambia_rules.use_jokers if hasattr(self.config, 'cambia_rules') else 2)
        low_threshold = max(1, total_cards // 5) # e.g., < ~10 cards
        med_threshold = max(low_threshold + 1, total_cards * 2 // 4) # e.g., < ~27 cards

        if stock_size <= 0: return StockpileEstimate.EMPTY
        if stock_size < low_threshold: return StockpileEstimate.LOW
        if stock_size < med_threshold: return StockpileEstimate.MEDIUM
        return StockpileEstimate.HIGH

    def _estimate_game_phase(self, stock_size: int, cambia_caller: Optional[int], current_turn: int) -> GamePhase:
         """Estimates the game phase."""
         if cambia_caller is not None: return GamePhase.CAMBIA_CALLED
         # Combine stockpile and turn number?
         stock_cat = self._estimate_stockpile(stock_size)
         turn_threshold_mid = 6 * NUM_PLAYERS # Heuristic, ~6 rounds
         turn_threshold_late = 12 * NUM_PLAYERS # Heuristic, ~12 rounds

         if stock_cat == StockpileEstimate.EMPTY: return GamePhase.LATE # Empty is always late
         if stock_cat == StockpileEstimate.LOW: return GamePhase.LATE   # Low stockpile implies late
         if stock_cat == StockpileEstimate.MEDIUM or current_turn >= turn_threshold_late: return GamePhase.MID # Medium stock or many turns
         return GamePhase.EARLY # High stock and few turns


    def _trigger_event_decay(self, target_index: int, trigger_event: str, current_turn: int):
         """Applies event-based decay (Levels 1 and 2) to a specific opponent index."""
         if self.memory_level == 0: return

         if target_index in self.opponent_belief:
              current_belief = self.opponent_belief[target_index]
              if isinstance(current_belief, CardBucket) and current_belief != CardBucket.UNKNOWN:
                   decayed_category = decay_bucket(current_belief)
                   self.opponent_belief[target_index] = decayed_category
                   self.opponent_last_seen_turn.pop(target_index, None) # Clear timestamp on decay
                   logger.debug(f"Agent {self.player_id} decaying Opponent[{target_index}] belief to {decayed_category.name} due to {trigger_event}.")
         # else: logger.warning(f"Agent {self.player_id} decay trigger skipped: Index {target_index} not in opponent belief.")

    def _apply_time_decay(self, current_turn: int):
        """Applies time-based decay (Level 2 only)."""
        if self.memory_level != 2: return

        indices_to_decay = []
        for idx, last_seen in self.opponent_last_seen_turn.items():
            if current_turn - last_seen >= self.time_decay_turns:
                 if idx in self.opponent_belief and isinstance(self.opponent_belief[idx], CardBucket) and self.opponent_belief[idx] != CardBucket.UNKNOWN:
                      indices_to_decay.append(idx)

        for idx in indices_to_decay:
            current_belief = self.opponent_belief[idx]
            decayed_category = decay_bucket(current_belief)
            self.opponent_belief[idx] = decayed_category
            self.opponent_last_seen_turn.pop(idx) # Remove timestamp after decay
            logger.debug(f"Agent {self.player_id} applying time decay for Opponent[{idx}] to {decayed_category.name}.")


    def get_infoset_key(self) -> InfosetKey:
        """Constructs the canonical, hashable infoset key from the current belief state."""
        # 1. Own Hand Buckets (Sorted Tuple)
        own_hand_items = sorted(self.own_hand.items()) # Sort by index
        own_hand_buckets = tuple(info.bucket.value for _, info in own_hand_items)

        # 2. Opponent Beliefs (Tuple indexed 0..N-1)
        opp_belief_values = []
        for i in range(self.opponent_card_count):
             belief = self.opponent_belief.get(i, CardBucket.UNKNOWN)
             opp_belief_values.append(belief.value if hasattr(belief, 'value') else CardBucket.UNKNOWN.value)
        opp_belief_tuple = tuple(opp_belief_values)

        # 3. Opponent Card Count
        opp_count = self.opponent_card_count

        # 4. Discard Pile Top Card Bucket
        discard_top_val = self.known_discard_top_bucket.value

        # 5. Stockpile Size Estimate (Enum value)
        stockpile_est_val = self.stockpile_estimate.value

        # 6. Game Phase (Enum value)
        game_phase_val = self.game_phase.value

        key = (
            own_hand_buckets, opp_belief_tuple, opp_count,
            discard_top_val, stockpile_est_val, game_phase_val,
        )
        return key # type: ignore

    def get_potential_opponent_snap_indices(self, target_rank: str) -> List[int]:
         """ Returns opponent hand indices the agent *believes* could match the target rank. """
         matching_indices = []
         target_bucket = get_card_bucket(Card(rank=target_rank, suit='S')) # Suit needed for King check

         for idx, belief in self.opponent_belief.items():
              if belief == CardBucket.UNKNOWN or isinstance(belief, DecayCategory):
                   # Conservatively assume match if unknown or decayed
                   matching_indices.append(idx)
              elif isinstance(belief, CardBucket):
                   # Check if the bucket could possibly contain the target rank
                   # This requires a reverse mapping or specific checks
                   if belief == target_bucket:
                        matching_indices.append(idx)
                   elif target_rank == 'K' and belief in [CardBucket.NEG_KING, CardBucket.HIGH_KING]:
                        matching_indices.append(idx)
                   # Add more complex checks if needed (e.g., LOW_NUM bucket vs rank '3')
         return matching_indices


    def clone(self) -> 'AgentState':
        """Creates a deep copy of the agent state."""
        new_state = AgentState(
             player_id=self.player_id, opponent_id=self.opponent_id,
             memory_level=self.memory_level, time_decay_turns=self.time_decay_turns,
             initial_hand_size=self.initial_hand_size, config=self.config
        )
        new_state.own_hand = copy.deepcopy(self.own_hand)
        new_state.opponent_belief = copy.deepcopy(self.opponent_belief)
        new_state.opponent_last_seen_turn = copy.deepcopy(self.opponent_last_seen_turn)
        new_state.known_discard_top_bucket = self.known_discard_top_bucket
        new_state.opponent_card_count = self.opponent_card_count
        new_state.stockpile_estimate = self.stockpile_estimate
        new_state.game_phase = self.game_phase
        new_state.cambia_caller = self.cambia_caller
        new_state._current_game_turn = self._current_game_turn
        return new_state


    def __str__(self) -> str:
        own_hand_str = {k: v.bucket.name for k, v in sorted(self.own_hand.items())}
        opp_belief_str = {i: self.opponent_belief.get(i, CardBucket.UNKNOWN).name
                          for i in range(self.opponent_card_count)}

        return (f"AgentState(P{self.player_id}, GT:{self._current_game_turn}, Phase:{self.game_phase.name}, "
                f"OH({len(self.own_hand)}):{own_hand_str}, "
                f"OB({self.opponent_card_count}):{opp_belief_str}, "
                f"Disc:{self.known_discard_top_bucket.name}, Stock:{self.stockpile_estimate.name}, "
                f"Cambia:{self.cambia_caller})")