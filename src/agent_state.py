# src/agent_state.py
from typing import List, Tuple, Optional, Dict, Any, Set, Union
from dataclasses import dataclass, field
import time # For potential time-based decay, though turn-based is preferred
import logging
import copy # For cloning

from .constants import (
    CardBucket, DecayCategory, GamePhase, StockpileEstimate, INITIAL_HAND_SIZE,
    NUM_PLAYERS, JOKER_RANK_STR,
    # Import action types needed for observation parsing
    GameAction, ActionReplace, ActionDiscard, ActionCallCambia,
    ActionAbilityPeekOwnSelect, ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect, ActionAbilityKingLookSelect, ActionAbilityKingSwapDecision,
    ActionSnapOwn, ActionSnapOpponent, ActionSnapOpponentMove, ActionPassSnap # Added snap actions
)
from .card import Card
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
     # Snap related info
     snap_results: Optional[List[Dict[str, Any]]] = None # List of dicts like {'snapper': int, 'action': ActionSnap*, 'success': bool, 'penalty': bool}
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

    # --- Core Beliefs ---
    # Own hand: Use list where index corresponds to position. Fixed size? More robust.
    # Let's assume indices can shift and use a Dict, but manage keys carefully.
    own_hand: Dict[int, KnownCardInfo] = field(default_factory=dict) # index -> KnownCardInfo

    # Opponent hand: Beliefs (Bucket, DecayCategory, or UNKNOWN)
    opponent_belief: Dict[int, Union[CardBucket, DecayCategory]] = field(default_factory=dict) # index -> belief state
    opponent_last_seen_turn: Dict[int, int] = field(default_factory=dict) # index -> turn for decay tracking

    # Public knowledge derived/updated from observations
    known_discard_top_bucket: CardBucket = CardBucket.UNKNOWN # Use Enum member
    opponent_card_count: int = 0
    stockpile_estimate: StockpileEstimate = StockpileEstimate.HIGH
    game_phase: GamePhase = GamePhase.START # Use GamePhase enum from constants
    cambia_caller: Optional[int] = None

    # Internal tracking
    current_agent_turn: int = 0 # Track agent's own turns for Level 2 decay trigger
    _last_observation_turn: int = -1 # Prevent double processing if update called multiple times

    def initialize(self, initial_observation: AgentObservation, initial_hand: List[Card], initial_peek_indices: Tuple[int, ...]):
        """Initialize belief state at the very start of the game."""
        self.opponent_card_count = initial_observation.player_hand_sizes[self.opponent_id]
        self.stockpile_estimate = self._estimate_stockpile(initial_observation.stockpile_size)
        self.game_phase = self._estimate_game_phase(initial_observation.stockpile_size, None) # Start phase
        self.known_discard_top_bucket = get_card_bucket(initial_observation.discard_top_card)
        self.cambia_caller = None
        self.current_agent_turn = 0 # Starts at 0
        self._last_observation_turn = -1

        # Initialize own hand knowledge
        self.own_hand = {}
        for i, card in enumerate(initial_hand):
            # bucket = CardBucket.UNKNOWN # Use Enum member # Default to UNKNOWN
            known = i in initial_peek_indices
            bucket = get_card_bucket(card) if known else CardBucket.UNKNOWN
            if known:
                logger.debug(f"Agent {self.player_id} initial peek: Index {i} is {bucket.name}")
            self.own_hand[i] = KnownCardInfo(bucket=bucket, last_seen_turn=0, card=card if known else None) # Store card only if known

        # Initialize opponent belief as UNKNOWN
        self.opponent_belief = {i: CardBucket.UNKNOWN for i in range(self.opponent_card_count)} # Use Enum member
        self.opponent_last_seen_turn = {}

        logger.debug(f"Agent {self.player_id} initialized. Own Hand: { {k: v.bucket.name for k,v in self.own_hand.items()} }. Opponent has {self.opponent_card_count} cards.")


    def update(self, observation: AgentObservation):
        """Updates belief state based on an observation tuple."""
        # Prevent processing the same observation multiple times if possible
        # if observation.current_turn <= self._last_observation_turn:
        #      return # Already processed this or an earlier state
        # self._last_observation_turn = observation.current_turn

        # Update public knowledge
        last_discard_card = observation.discard_top_card
        self.known_discard_top_bucket = get_card_bucket(last_discard_card)
        # Update counts based on observation (handles snaps/penalties implicitly)
        self.opponent_card_count = observation.player_hand_sizes[self.opponent_id]
        own_card_count = observation.player_hand_sizes[self.player_id]
        # Adjust own hand dict size if needed (e.g. due to penalty)
        # This is complex - requires careful index management or fixed slots.
        # For now, assume indices remain stable unless explicitly removed/shifted by known actions.
        self._reconcile_own_hand_indices(own_card_count)


        self.stockpile_estimate = self._estimate_stockpile(observation.stockpile_size)
        if observation.did_cambia_get_called and self.cambia_caller is None:
            self.cambia_caller = observation.who_called_cambia
        self.game_phase = self._estimate_game_phase(observation.stockpile_size, self.cambia_caller)

        current_turn = observation.current_turn # Turn number *after* action

        # Increment own turn counter if it was our turn to act (or initiate snap)
        if observation.acting_player == self.player_id:
            self.current_agent_turn += 1

        # --- Parse Action and Snap Results ---
        action = observation.action
        actor = observation.acting_player
        snap_results = observation.snap_results or []

        # --- 1. Process Snap Results First ---
        # Snaps change hand states *before* the main action's effect might resolve
        opponent_indices_removed_by_snap: Set[int] = set()
        own_indices_removed_by_snap: Set[int] = set()
        own_card_moved_by_snap: Optional[Tuple[int, int]] = None # (from_idx, to_opp_idx)

        for snap_info in snap_results:
             snapper = snap_info['snapper']
             snap_action = snap_info['action']
             success = snap_info['success']
             penalty = snap_info['penalty']

             if snapper == self.player_id: # Our snap attempt
                  if isinstance(snap_action, ActionSnapOwn) and success:
                       own_indices_removed_by_snap.add(snap_action.own_card_hand_index)
                  elif isinstance(snap_action, ActionSnapOpponent) and success:
                       # We snapped opponent, record which opponent index was removed
                       opponent_indices_removed_by_snap.add(snap_action.opponent_target_hand_index)
                       # We also need to know which card *we* moved later
                       # Assume move action is part of the snap_info if successful? Or separate observation?
                       # Let's assume the move details are implicitly handled by player_hand_sizes update for now.
                       pass
                  elif penalty:
                       # We got a penalty - hand size increases (handled by player_hand_sizes)
                       # Existing cards remain, new cards are UNKNOWN
                       pass
             else: # Opponent's snap attempt
                  if isinstance(snap_action, ActionSnapOwn) and success:
                       # Opponent snapped their own card. Hand size decreases.
                       # Trigger decay for the snapped index.
                       # We don't know *which* index unless observable. Assume we don't.
                       # Handled by opponent_card_count update. Decay relies on other actions.
                       pass
                  elif isinstance(snap_action, ActionSnapOpponent) and success:
                       # Opponent snapped one of our cards.
                       # Which card? Observation needs this info. Assume it's not provided directly.
                       # Handled by player_hand_sizes update. Decay our own knowledge?
                       pass
                       # Opponent also moved one of their cards to our hand.
                       # Handled by player_hand_sizes. New card is UNKNOWN.
                  elif penalty:
                       # Opponent got penalty - hand size increases. New cards UNKNOWN.
                       # Handled by opponent_card_count update.
                       pass

        # Apply removals and index shifts due to successful snaps *before* processing main action
        self._apply_snap_removals(own_indices_removed_by_snap, opponent_indices_removed_by_snap)


        # --- 2. Updates based on Main Action ---
        if action: # If there was a main action besides snaps
             # Updates based on Own Actions
             if actor == self.player_id:
                  # Own hand updates based on draw/replace/peek/ability results
                  if isinstance(action, ActionReplace):
                      if observation.drawn_card:
                           target_idx = action.target_hand_index
                           drawn_bucket = get_card_bucket(observation.drawn_card)
                           self.own_hand[target_idx] = KnownCardInfo(bucket=drawn_bucket, last_seen_turn=current_turn, card=observation.drawn_card)
                           logger.debug(f"Agent {self.player_id} updated own hand index {target_idx} to {drawn_bucket.name} after replace.")
                      else: logger.warning("Replace action observed for self, but no drawn card in observation.")

                  elif isinstance(action, ActionAbilityPeekOwnSelect):
                      if observation.peeked_cards:
                           for (p_idx, h_idx), card in observation.peeked_cards.items():
                               if p_idx == self.player_id and h_idx in self.own_hand:
                                    peeked_bucket = get_card_bucket(card)
                                    # Don't overwrite if already known unless UNKNOWN? Or always update? Always update.
                                    self.own_hand[h_idx] = KnownCardInfo(bucket=peeked_bucket, last_seen_turn=current_turn, card=card)
                                    logger.debug(f"Agent {self.player_id} peeked own index {h_idx}, saw {peeked_bucket.name}.")

                  elif isinstance(action, ActionAbilityPeekOtherSelect):
                      if observation.peeked_cards:
                           for (p_idx, h_idx), card in observation.peeked_cards.items():
                                if p_idx == self.opponent_id:
                                     peeked_bucket = get_card_bucket(card)
                                     # Update opponent belief only if index exists
                                     if h_idx in self.opponent_belief:
                                          self.opponent_belief[h_idx] = peeked_bucket
                                          self.opponent_last_seen_turn[h_idx] = current_turn
                                          logger.debug(f"Agent {self.player_id} peeked opponent index {h_idx}, saw {peeked_bucket.name}.")
                                     else:
                                          logger.warning(f"Peeked opponent index {h_idx} which is not in current belief keys: {list(self.opponent_belief.keys())}")


                  elif isinstance(action, ActionAbilityBlindSwapSelect):
                       own_idx, opp_idx = action.own_hand_index, action.opponent_hand_index
                       # We know our card at own_idx is gone (swapped for unknown)
                       if own_idx in self.own_hand:
                           self.own_hand[own_idx] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=current_turn) # Became unknown
                       # Trigger decay for the opponent index involved
                       self._trigger_event_decay(target_index=opp_idx, trigger_event="swap (blind)", current_turn=current_turn)

                  elif isinstance(action, ActionAbilityKingLookSelect):
                       # Look itself doesn't change state, but gives info for next step
                       # Update belief based on peek results if provided in this obs step
                       if observation.peeked_cards:
                            for (p_idx, h_idx), card in observation.peeked_cards.items():
                                 if p_idx == self.opponent_id and h_idx in self.opponent_belief:
                                      peeked_bucket = get_card_bucket(card)
                                      self.opponent_belief[h_idx] = peeked_bucket
                                      self.opponent_last_seen_turn[h_idx] = current_turn
                                      logger.debug(f"Agent {self.player_id} looked (King) at opponent index {h_idx}, saw {peeked_bucket.name}.")
                                 elif p_idx == self.player_id and h_idx in self.own_hand:
                                      peeked_bucket = get_card_bucket(card)
                                      self.own_hand[h_idx] = KnownCardInfo(bucket=peeked_bucket, last_seen_turn=current_turn, card=card)
                                      logger.debug(f"Agent {self.player_id} looked (King) at own index {h_idx}, saw {peeked_bucket.name}.")

                  elif isinstance(action, ActionAbilityKingSwapDecision) and action.perform_swap:
                       # Swap happened. Need info about which cards were involved from observation or prior state.
                       # Simplification: Invalidate knowledge of involved indices.
                       # Assume peek_results from the LOOK step are available? Risky.
                       # Let's just trigger decay based on action params if possible (needs look info).
                       # If we knew look_data = {'own_idx': i, 'opp_idx': j}:
                       # own_idx = look_data['own_idx']
                       # opp_idx = look_data['opp_idx']
                       # self.own_hand[own_idx] = KnownCardInfo(bucket=CardBucket.UNKNOWN, ...)
                       # self._trigger_event_decay(target_index=opp_idx, ...)
                       logger.warning("King Swap belief update requires info from Look step - current implementation assumes decay only.")


             # Updates based on Opponent Actions
             elif actor == self.opponent_id:
                  # Trigger decay if opponent action affects our knowledge of their hand
                  if isinstance(action, ActionReplace):
                      target_idx = action.target_hand_index # Opponent's index
                      self._trigger_event_decay(target_index=target_idx, trigger_event="replace", current_turn=current_turn)

                  elif isinstance(action, ActionAbilityBlindSwapSelect):
                      # Opponent swapped their card (opp_own_idx) with our card (opp_opp_idx)
                      opp_own_idx = action.own_hand_index
                      opp_opp_idx = action.opponent_hand_index # This is an index in *our* hand
                      # Decay opponent belief at their index
                      self._trigger_event_decay(target_index=opp_own_idx, trigger_event="swap (blind)", current_turn=current_turn)
                      # Update our own hand knowledge at the affected index
                      if opp_opp_idx in self.own_hand:
                           self.own_hand[opp_opp_idx] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=current_turn)

                  elif isinstance(action, ActionAbilityKingSwapDecision) and action.perform_swap:
                      # Opponent performed King swap. Decay involved indices if known.
                      logger.warning("Opponent King Swap belief update requires info from Look step - decay not implemented.")


        # --- 3. Apply Time Decay (Level 2) ---
        if self.memory_level == 2:
            self._apply_time_decay(current_turn)

        # --- 4. Reconcile opponent belief indices ---
        # Ensure belief dict keys match opponent card count
        self._reconcile_opponent_belief_indices(self.opponent_card_count)


    def _reconcile_own_hand_indices(self, expected_count: int):
        """ Adjusts own_hand dict if size mismatches expected count (e.g. penalty draw). """
        current_indices = set(self.own_hand.keys())
        current_count = len(current_indices)

        if current_count < expected_count:
            # Add new UNKNOWN entries for penalty cards
            next_idx = 0
            added = 0
            while added < (expected_count - current_count):
                 if next_idx not in current_indices:
                      self.own_hand[next_idx] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self.current_agent_turn) # Turn? Use current.
                      added += 1
                      logger.debug(f"Agent {self.player_id} added unknown card slot at index {next_idx} due to hand size increase.")
                 next_idx += 1
                 if next_idx > expected_count + 5: # Safety break
                      logger.error("Error reconciling own hand indices - potential infinite loop.")
                      break
        elif current_count > expected_count:
             # This implies cards were removed without explicit action tracking (e.g. opponent snap)
             # We don't know *which* card was removed. Cannot fix indices easily.
             logger.warning(f"Own hand size ({current_count}) > expected ({expected_count}). Indices may be incorrect.")


    def _reconcile_opponent_belief_indices(self, expected_count: int):
        """ Adjust opponent_belief dict to match expected count. """
        current_indices = set(self.opponent_belief.keys())
        current_count = len(current_indices)

        if current_count < expected_count:
             # Add new UNKNOWN entries
             next_idx = 0
             added = 0
             while added < (expected_count - current_count):
                  if next_idx not in current_indices:
                       self.opponent_belief[next_idx] = CardBucket.UNKNOWN
                       added += 1
                  next_idx += 1
                  if next_idx > expected_count + 5: break # Safety
        elif current_count > expected_count:
             # Remove highest indices until count matches
             removed_count = 0
             sorted_indices = sorted(list(current_indices), reverse=True)
             for idx in sorted_indices:
                  if len(self.opponent_belief) > expected_count:
                       self.opponent_belief.pop(idx, None)
                       self.opponent_last_seen_turn.pop(idx, None)
                       removed_count += 1
                  else:
                       break
             if removed_count > 0:
                   logger.debug(f"Removed {removed_count} highest opponent belief indices to match expected count {expected_count}.")


    def _apply_snap_removals(self, own_indices_removed: Set[int], opponent_indices_removed: Set[int]):
         """ Removes cards from belief dictionaries due to successful snaps. Handles index shifts. """
         if not own_indices_removed and not opponent_indices_removed:
              return

         # Process own hand removals
         if own_indices_removed:
              logger.debug(f"Agent {self.player_id} applying own snap removals for indices: {own_indices_removed}")
              new_own_hand = {}
              current_sorted_indices = sorted(self.own_hand.keys())
              target_idx = 0
              for old_idx in current_sorted_indices:
                  if old_idx not in own_indices_removed:
                       new_own_hand[target_idx] = self.own_hand[old_idx]
                       target_idx += 1
              self.own_hand = new_own_hand

         # Process opponent hand removals
         if opponent_indices_removed:
              logger.debug(f"Agent {self.player_id} applying opponent snap removals for indices: {opponent_indices_removed}")
              new_opp_belief = {}
              new_opp_last_seen = {}
              current_sorted_indices = sorted(self.opponent_belief.keys())
              target_idx = 0
              for old_idx in current_sorted_indices:
                  if old_idx not in opponent_indices_removed:
                       new_opp_belief[target_idx] = self.opponent_belief[old_idx]
                       if old_idx in self.opponent_last_seen_turn:
                            new_opp_last_seen[target_idx] = self.opponent_last_seen_turn[old_idx]
                       target_idx += 1
              self.opponent_belief = new_opp_belief
              self.opponent_last_seen_turn = new_opp_last_seen


    def _estimate_stockpile(self, stock_size: int) -> StockpileEstimate:
        """Estimates stockpile category based on size."""
        total_cards = 52 + self.config.cambia_rules.use_jokers if hasattr(self.config, 'cambia_rules') else 54 # Estimate total
        low_threshold = total_cards // 4 # Less than 1/4 left
        med_threshold = total_cards * 2 // 3 # Less than 2/3 left

        if stock_size == 0: return StockpileEstimate.EMPTY
        if stock_size < low_threshold: return StockpileEstimate.LOW
        if stock_size < med_threshold: return StockpileEstimate.MEDIUM
        return StockpileEstimate.HIGH

    def _estimate_game_phase(self, stock_size: int, cambia_caller: Optional[int]) -> GamePhase:
         """Estimates the game phase."""
         if cambia_caller is not None: return GamePhase.CAMBIA_CALLED
         # Base phase on stockpile estimate for simplicity
         stock_cat = self._estimate_stockpile(stock_size)
         if stock_cat == StockpileEstimate.HIGH: return GamePhase.EARLY
         if stock_cat == StockpileEstimate.MEDIUM: return GamePhase.MID
         return GamePhase.LATE # Includes LOW and EMPTY

    def _trigger_event_decay(self, target_index: int, trigger_event: str, current_turn: int):
         """Applies event-based decay (Levels 1 and 2) to a specific opponent index."""
         if self.memory_level == 0: return # No decay for perfect recall

         # Check if the index exists in the current belief map
         if target_index in self.opponent_belief:
              current_belief = self.opponent_belief[target_index]
              # Only decay if we had specific knowledge (a CardBucket, not already decayed or UNKNOWN)
              if isinstance(current_belief, CardBucket) and current_belief != CardBucket.UNKNOWN:
                   decayed_category = decay_bucket(current_belief) # Map specific bucket to broader category
                   self.opponent_belief[target_index] = decayed_category
                   # Clear last seen time when event decay happens
                   self.opponent_last_seen_turn.pop(target_index, None)
                   logger.debug(f"Agent {self.player_id} decaying opponent belief at index {target_index} to {decayed_category.name} due to {trigger_event}.")
         # else: logger.warning(f"Attempted to decay opponent index {target_index} which does not exist.")


    def _apply_time_decay(self, current_turn: int):
        """Applies time-based decay (Level 2 only)."""
        if self.memory_level != 2: return

        indices_to_decay = []
        # Iterate through indices we have timestamps for
        for idx, last_seen in self.opponent_last_seen_turn.items():
            if current_turn - last_seen >= self.time_decay_turns:
                 # Check if belief is currently specific (not already decayed or UNKNOWN)
                 if idx in self.opponent_belief and isinstance(self.opponent_belief[idx], CardBucket) and self.opponent_belief[idx] != CardBucket.UNKNOWN:
                      indices_to_decay.append(idx)

        for idx in indices_to_decay:
            current_belief = self.opponent_belief[idx] # Already checked it's a CardBucket
            decayed_category = decay_bucket(current_belief)
            self.opponent_belief[idx] = decayed_category
            self.opponent_last_seen_turn.pop(idx, None) # Remove timestamp after decay
            logger.debug(f"Agent {self.player_id} applying time decay for opponent index {idx} to {decayed_category.name}.")


    def get_infoset_key(self) -> InfosetKey:
        """Constructs the canonical, hashable infoset key from the current belief state."""
        # 1. Own Hand Buckets (Sorted by index, use UNKNOWN if index missing)
        own_max_idx = max(self.own_hand.keys()) if self.own_hand else -1
        own_current_size = len(self.own_hand) # Actual number of cards we hold
        own_hand_buckets = []
        # Use actual size, assuming indices are compact [0, 1, ..., size-1] after reconciliation
        for i in range(own_current_size):
             info = self.own_hand.get(i)
             # If info is missing for an index < size, it's an error state, but use UNKNOWN
             own_hand_buckets.append(info.bucket.value if info else CardBucket.UNKNOWN.value)
        own_hand_tuple = tuple(own_hand_buckets)

        # 2. Opponent Beliefs (Sorted by index, use value)
        opp_max_idx = max(self.opponent_belief.keys()) if self.opponent_belief else -1
        opp_current_size = self.opponent_card_count # Use tracked count
        opp_belief_list = []
        # Use actual size, assuming indices are compact [0, 1, ..., size-1] after reconciliation
        for i in range(opp_current_size):
            belief = self.opponent_belief.get(i, CardBucket.UNKNOWN) # Default to UNKNOWN
            # Ensure belief is an enum before accessing .value
            if isinstance(belief, (CardBucket, DecayCategory)):
                 opp_belief_list.append(belief.value)
            else:
                 logger.warning(f"Invalid type in opponent_belief at index {i}: {type(belief)}. Using UNKNOWN.")
                 opp_belief_list.append(CardBucket.UNKNOWN.value)
        opp_belief_tuple = tuple(opp_belief_list)

        # 3. Opponent Card Count
        opp_count = self.opponent_card_count

        # 4. Discard Pile Top Card Bucket
        discard_top_val = self.known_discard_top_bucket.value

        # 5. Stockpile Size Estimate (Enum value)
        stockpile_est_val = self.stockpile_estimate.value

        # 6. Game Phase (Enum value)
        game_phase_val = self.game_phase.value

        # Combine into the final tuple
        key = InfosetKey(( # Explicitly cast outer container to InfosetKey type if defined
            own_hand_tuple,
            opp_belief_tuple,
            opp_count,
            discard_top_val,
            stockpile_est_val,
            game_phase_val,
        ))
        return key

    def clone(self) -> 'AgentState':
        """Creates a deep copy of the agent state."""
        # Use deepcopy for safety with nested dictionaries/dataclasses
        # Ensure KnownCardInfo is handled correctly if it becomes more complex
        return copy.deepcopy(self)

    def __str__(self) -> str:
        own_hand_str = {k: v.bucket.name for k, v in sorted(self.own_hand.items())}
        opp_belief_str = {k: v.name for k, v in sorted(self.opponent_belief.items())}
        return (f"AgentState(P{self.player_id}, Turn: {self.current_agent_turn}, "
                f"OwnHand({len(self.own_hand)}): {own_hand_str}, "
                f"OppBelief({self.opponent_card_count}): {opp_belief_str}, "
                f"Discard: {self.known_discard_top_bucket.name}, Stock: {self.stockpile_estimate.name}, "
                f"Phase: {self.game_phase.name}, Cambia: {self.cambia_caller})")