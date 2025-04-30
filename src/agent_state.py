# src/agent_state.py
from typing import List, Tuple, Optional, Dict, Any, Set, Union
from dataclasses import dataclass, field
import time # For potential time-based decay, though turn-based is preferred
import logging
import copy # For cloning

from .constants import (
    CardBucket, DecayCategory, GamePhase, StockpileEstimate,
    GameAction, ActionReplace, 
    ActionAbilityPeekOwnSelect, ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect, ActionAbilityKingLookSelect, ActionAbilityKingSwapDecision,
    ActionSnapOpponentMove
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
        # Prevent processing same turn multiple times? Risky if multiple obs per turn.
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

        # --- 2. Process Snap Results ---
        action = observation.action
        actor = observation.acting_player
        snap_results = observation.snap_results # Now detailed list of dicts

        # Identify removals from successful snaps
        own_indices_removed: Set[int] = set()
        opponent_indices_removed: Set[int] = set()

        for snap_info in snap_results:
             snapper = snap_info.get('snapper')
             success = snap_info.get('success', False)
             penalty = snap_info.get('penalty', False)

             if snapper == self.player_id: # Our snap attempt
                  if success:
                       if "removed_own_index" in snap_info and snap_info["removed_own_index"] is not None:
                            own_indices_removed.add(snap_info["removed_own_index"])
                       elif "removed_opponent_index" in snap_info and snap_info["removed_opponent_index"] is not None:
                            opponent_indices_removed.add(snap_info["removed_opponent_index"])
                  elif penalty:
                       # Our hand size increases, handled by reconciliation later
                       logger.debug(f"Agent {self.player_id} received penalty from snap.")
             else: # Opponent's snap attempt
                  if success:
                       # If opponent snapped own, their count decreases (reconciliation)
                       # If opponent snapped us, our count decreases (reconciliation)
                       # Opponent move increases our count (reconciliation)
                       pass # Handled by reconciliation based on final counts
                  elif penalty:
                       # Opponent hand size increases (reconciliation)
                       logger.debug(f"Agent {self.player_id} observed opponent penalty from snap.")

        # Apply removals *before* reconciliation
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
             # Updates based on Own Actions
             if actor == self.player_id:
                  if isinstance(action, ActionReplace):
                      if observation.drawn_card:
                           target_idx = action.target_hand_index
                           drawn_bucket = get_card_bucket(observation.drawn_card)
                           # Check if index exists before assignment
                           if target_idx in self.own_hand:
                               self.own_hand[target_idx] = KnownCardInfo(bucket=drawn_bucket, last_seen_turn=self._current_game_turn, card=observation.drawn_card)
                               logger.debug(f"Agent {self.player_id} updated own hand index {target_idx} to {drawn_bucket.name} after replace.")
                           else: logger.warning(f"Agent {self.player_id} Replace target index {target_idx} invalid.")
                      else: logger.warning("Replace action observed for self, but no drawn card in observation.")

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
                                if p_idx == self.opponent_id:
                                     peeked_bucket = get_card_bucket(card)
                                     if h_idx in self.opponent_belief:
                                          self.opponent_belief[h_idx] = peeked_bucket
                                          self.opponent_last_seen_turn[h_idx] = self._current_game_turn
                                          logger.debug(f"Agent {self.player_id} peeked opponent index {h_idx}, saw {peeked_bucket.name}.")
                                     else: logger.warning(f"Peeked opponent index {h_idx} not in current belief keys: {list(self.opponent_belief.keys())}")

                  elif isinstance(action, ActionAbilityBlindSwapSelect):
                       own_idx, opp_idx = action.own_hand_index, action.opponent_hand_index
                       if own_idx in self.own_hand: self.own_hand[own_idx] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn)
                       self._trigger_event_decay(target_index=opp_idx, trigger_event="swap (blind)", current_turn=self._current_game_turn)

                  elif isinstance(action, ActionAbilityKingLookSelect):
                       if observation.peeked_cards:
                            for (p_idx, h_idx), card in observation.peeked_cards.items():
                                 if p_idx == self.opponent_id and h_idx in self.opponent_belief:
                                      peeked_bucket = get_card_bucket(card); self.opponent_belief[h_idx] = peeked_bucket; self.opponent_last_seen_turn[h_idx] = self._current_game_turn
                                      logger.debug(f"Agent {self.player_id} looked (King) at opponent index {h_idx}, saw {peeked_bucket.name}.")
                                 elif p_idx == self.player_id and h_idx in self.own_hand:
                                      peeked_bucket = get_card_bucket(card); self.own_hand[h_idx] = KnownCardInfo(bucket=peeked_bucket, last_seen_turn=self._current_game_turn, card=card)
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
                                   self._trigger_event_decay(target_index=opp_idx, trigger_event="swap (king)", current_turn=self._current_game_turn)
                               else: logger.warning("Could not determine indices for King Swap decay.")
                           else: logger.warning("Incorrect peeked_cards format for King Swap decay.")
                       else: logger.warning("Missing peeked_cards info for King Swap decay.")

                  elif isinstance(action, ActionSnapOpponentMove):
                       # We successfully snapped opponent, then moved one of our cards.
                       # Our hand count decreased - handled by reconciliation.
                       # Opponent hand count increased - handled by reconciliation.
                       own_moved_idx = action.own_card_to_move_hand_index
                       # Need to ensure our own hand index map reflects this removal implicitly
                       # This happens during reconciliation as observed_own_count is smaller.


             # Updates based on Opponent Actions
             elif actor == self.opponent_id:
                  if isinstance(action, ActionReplace):
                      target_idx = action.target_hand_index # Opponent's index
                      self._trigger_event_decay(target_index=target_idx, trigger_event="replace", current_turn=self._current_game_turn)
                  elif isinstance(action, ActionAbilityBlindSwapSelect):
                      opp_own_idx = action.own_hand_index # Index in opponent's hand
                      our_idx = action.opponent_hand_index # Index in our hand
                      self._trigger_event_decay(target_index=opp_own_idx, trigger_event="swap (blind)", current_turn=self._current_game_turn)
                      if our_idx in self.own_hand: self.own_hand[our_idx] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn)
                  elif isinstance(action, ActionAbilityKingSwapDecision) and action.perform_swap:
                      # Opponent performed swap. Need involved indices. Similar logic to self-swap.
                       if observation.peeked_cards: # Check if WE observed the peek (unlikely unless we were part of it)
                            logger.warning("Opponent King Swap observed, decay logic might be incomplete without peek info.")
                       else: # Assume decay based on action type if indices unknown
                           logger.debug("Opponent King Swap: Triggering decay for potentially involved indices if memory allows.")
                           # This requires a more complex memory model to guess involved indices.
                           pass # For now, no automatic decay without index info.
                  elif isinstance(action, ActionSnapOpponentMove):
                       # Opponent successfully snapped one of their cards, then moved one to us.
                       # Our hand count increases - handled by reconciliation.
                       # Opponent hand count stays same (lose one, gain one) - handled by reconciliation.
                       pass

        # --- 5. Apply Time Decay (Level 2) ---
        if self.memory_level == 2:
            self._apply_time_decay(self._current_game_turn)

        # --- Final Check (optional): Log current state ---
        # logger.debug(f"Agent {self.player_id} state after update (Turn {self._current_game_turn}): {self}")


    def _reconcile_own_hand_indices(self, expected_count: int):
        """ Adjusts own_hand dict to match expected count, keeping indices 0 to N-1. """
        current_indices = set(self.own_hand.keys())
        current_count = len(current_indices)
        needs_update = False

        if current_count < expected_count:
            # Add new UNKNOWN entries (penalty/opponent move)
            added = 0
            for i in range(expected_count): # Ensure indices 0..expected-1 exist
                 if i not in current_indices:
                      self.own_hand[i] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn)
                      added += 1
                      needs_update = True
            if added != (expected_count - current_count):
                 logger.error(f"Logic error in reconcile own hand (add). Added {added}, needed {expected_count - current_count}")
            if needs_update: logger.debug(f"Agent {self.player_id} reconciled own hand: Added {added} UNKNOWN slots to reach {expected_count}.")

        elif current_count > expected_count:
            # Remove highest indices (our snap/opponent snap)
            removed_count = 0
            sorted_indices = sorted(list(current_indices), reverse=True)
            for idx in sorted_indices:
                if len(self.own_hand) > expected_count:
                    self.own_hand.pop(idx, None)
                    removed_count += 1
                    needs_update = True
                else: break
            if removed_count > 0:
                logger.debug(f"Agent {self.player_id} reconciled own hand: Removed {removed_count} highest index slots to reach {expected_count}.")

        # Ensure indices are contiguous after potential removals
        if needs_update and len(self.own_hand) != expected_count:
             logger.warning(f"Own hand reconciliation resulted in {len(self.own_hand)} cards, expected {expected_count}. Rebuilding index map.")
             current_items = sorted(self.own_hand.items())
             self.own_hand = {i: item[1] for i, item in enumerate(current_items)}
             # If still wrong size, add/remove UNKNOWNs
             while len(self.own_hand) < expected_count: self.own_hand[len(self.own_hand)] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn)
             while len(self.own_hand) > expected_count: self.own_hand.pop(len(self.own_hand) - 1)

        # Final check
        if len(self.own_hand) != expected_count:
             logger.error(f"FATAL: Own hand reconciliation failed. Expected {expected_count}, got {len(self.own_hand)}. State: {self.own_hand}")


    def _reconcile_opponent_belief_indices(self, expected_count: int):
        """ Adjust opponent_belief dict to match expected count, keeping indices 0 to N-1. """
        current_indices = set(self.opponent_belief.keys())
        current_count = len(current_indices)
        needs_update = False

        if current_count < expected_count:
            # Add new UNKNOWN entries
            added = 0
            for i in range(expected_count):
                 if i not in current_indices:
                      self.opponent_belief[i] = CardBucket.UNKNOWN
                      self.opponent_last_seen_turn.pop(i, None) # Clear timestamp for new slot
                      added += 1
                      needs_update = True
            if added != (expected_count - current_count):
                 logger.error(f"Logic error in reconcile opponent belief (add). Added {added}, needed {expected_count - current_count}")
            if needs_update: logger.debug(f"Agent {self.player_id} reconciled opponent belief: Added {added} UNKNOWN slots to reach {expected_count}.")

        elif current_count > expected_count:
            # Remove highest indices
            removed_count = 0
            sorted_indices = sorted(list(current_indices), reverse=True)
            for idx in sorted_indices:
                if len(self.opponent_belief) > expected_count:
                    self.opponent_belief.pop(idx, None)
                    self.opponent_last_seen_turn.pop(idx, None)
                    removed_count += 1
                    needs_update = True
                else: break
            if removed_count > 0:
                 logger.debug(f"Agent {self.player_id} reconciled opponent belief: Removed {removed_count} highest index slots to reach {expected_count}.")

        # Ensure indices are contiguous
        if needs_update and len(self.opponent_belief) != expected_count:
             logger.warning(f"Opponent belief reconciliation resulted in {len(self.opponent_belief)} slots, expected {expected_count}. Rebuilding index map.")
             current_beliefs = sorted(self.opponent_belief.items())
             current_last_seen = dict(self.opponent_last_seen_turn) # Copy
             self.opponent_belief = {i: item[1] for i, item in enumerate(current_beliefs)}
             self.opponent_last_seen_turn = {i: current_last_seen[item[0]] for i, item in enumerate(current_beliefs) if item[0] in current_last_seen}
             # Add/remove UNKNOWNs if needed
             while len(self.opponent_belief) < expected_count: self.opponent_belief[len(self.opponent_belief)] = CardBucket.UNKNOWN
             while len(self.opponent_belief) > expected_count: self.opponent_belief.pop(len(self.opponent_belief) - 1); self.opponent_last_seen_turn.pop(len(self.opponent_belief), None) # Pop from belief and timestamp dict

        # Final check
        if len(self.opponent_belief) != expected_count:
             logger.error(f"FATAL: Opponent belief reconciliation failed. Expected {expected_count}, got {len(self.opponent_belief)}. State: {self.opponent_belief}")


    def _apply_snap_removals(self, own_indices_removed: Set[int], opponent_indices_removed: Set[int]):
         """ Removes cards from belief dictionaries due to successful snaps. Rebuilds index map. """
         if not own_indices_removed and not opponent_indices_removed: return

         # Process own hand removals
         if own_indices_removed:
              logger.debug(f"Agent {self.player_id} applying own snap removals for indices: {own_indices_removed}")
              new_own_hand = {}
              current_sorted_items = sorted(self.own_hand.items())
              target_idx = 0
              for old_idx, info in current_sorted_items:
                  if old_idx not in own_indices_removed:
                       new_own_hand[target_idx] = info
                       target_idx += 1
              self.own_hand = new_own_hand

         # Process opponent hand removals
         if opponent_indices_removed:
              logger.debug(f"Agent {self.player_id} applying opponent snap removals for indices: {opponent_indices_removed}")
              new_opp_belief = {}
              new_opp_last_seen = {}
              current_sorted_beliefs = sorted(self.opponent_belief.items())
              target_idx = 0
              for old_idx, belief in current_sorted_beliefs:
                  if old_idx not in opponent_indices_removed:
                       new_opp_belief[target_idx] = belief
                       if old_idx in self.opponent_last_seen_turn:
                            new_opp_last_seen[target_idx] = self.opponent_last_seen_turn[old_idx]
                       target_idx += 1
              self.opponent_belief = new_opp_belief
              self.opponent_last_seen_turn = new_opp_last_seen


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
         turn_threshold_mid = 6 # Heuristic, approx 3 rounds
         turn_threshold_late = 12 # Heuristic, approx 6 rounds

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
                   self.opponent_last_seen_turn.pop(target_index, None)
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
        # 1. Own Hand Buckets (Tuple sorted by value)
        own_hand_buckets = sorted([info.bucket.value for info in self.own_hand.values()])
        own_hand_tuple = tuple(own_hand_buckets)

        # 2. Opponent Beliefs (Tuple sorted by value)
        opp_belief_values = []
        for i in range(self.opponent_card_count): # Use count to ensure fixed size if indices missing
             belief = self.opponent_belief.get(i, CardBucket.UNKNOWN)
             if isinstance(belief, (CardBucket, DecayCategory)): opp_belief_values.append(belief.value)
             else: opp_belief_values.append(CardBucket.UNKNOWN.value); logger.warning(f"Invalid belief type at index {i}: {type(belief)}")
        opp_belief_tuple = tuple(sorted(opp_belief_values))

        # 3. Opponent Card Count
        opp_count = self.opponent_card_count

        # 4. Discard Pile Top Card Bucket
        discard_top_val = self.known_discard_top_bucket.value

        # 5. Stockpile Size Estimate (Enum value)
        stockpile_est_val = self.stockpile_estimate.value

        # 6. Game Phase (Enum value)
        game_phase_val = self.game_phase.value

        # Simply return the tuple directly
        key = (
            own_hand_tuple, opp_belief_tuple, opp_count,
            discard_top_val, stockpile_est_val, game_phase_val,
        )
        # Ensure the returned type matches the alias hint for clarity, but construction is standard tuple
        return key # type: ignore

    def get_potential_opponent_snap_indices(self, target_rank: str) -> List[int]:
         """ Returns opponent hand indices the agent *believes* could match the target rank. """
         matching_indices = []
         target_bucket = get_card_bucket(Card(rank=target_rank, suit='S')) # Suit needed for King check

         for idx, belief in self.opponent_belief.items():
              if belief == CardBucket.UNKNOWN or isinstance(belief, DecayCategory):
                   # If unknown or decayed, could it match? Conservatively, yes.
                   # More advanced: use probability distribution if available.
                   matching_indices.append(idx)
              elif isinstance(belief, CardBucket):
                   # If we have a specific bucket belief, check if it *could* match the rank.
                   # This requires mapping bucket back to possible ranks. Complex.
                   # Simple check: if the target rank maps to this bucket, assume match.
                   if belief == target_bucket:
                        matching_indices.append(idx)
                   # Handle edge case: King ability bucket covers both ranks
                   elif target_rank == 'K' and belief in [CardBucket.NEG_KING, CardBucket.HIGH_KING]:
                        matching_indices.append(idx)
                   # Add other checks if buckets span multiple ranks that need specific checks
         return matching_indices


    def clone(self) -> 'AgentState':
        """Creates a deep copy of the agent state."""
        # Use deepcopy for safety, especially with dictionaries
        new_state = AgentState(
             player_id=self.player_id, opponent_id=self.opponent_id,
             memory_level=self.memory_level, time_decay_turns=self.time_decay_turns,
             initial_hand_size=self.initial_hand_size, config=self.config # Pass config copy? Assume shallow ok
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
        opp_belief_str = {}
        for i in range(self.opponent_card_count): # Iterate using count
             belief = self.opponent_belief.get(i, CardBucket.UNKNOWN)
             opp_belief_str[i] = belief.name if hasattr(belief, 'name') else str(belief)

        return (f"AgentState(P{self.player_id}, GT:{self._current_game_turn}, Phase:{self.game_phase.name}, "
                f"OH({len(self.own_hand)}):{own_hand_str}, "
                f"OB({self.opponent_card_count}):{opp_belief_str}, "
                f"Disc:{self.known_discard_top_bucket.name}, Stock:{self.stockpile_estimate.name}, "
                f"Cambia:{self.cambia_caller})")