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
    ActionSnapOwn, ActionSnapOpponent, ActionSnapOpponentMove, NUM_PLAYERS,
    CardObject # Assuming CardObject is defined or Any
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
    card: Optional[CardObject] = None # Store actual card for perfect recall/debugging

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
     discard_top_card: Optional[CardObject]
     player_hand_sizes: List[int]
     stockpile_size: int
     # Information specific to the observing player:
     drawn_card: Optional[CardObject] = None # If observing player drew
     peeked_cards: Optional[Dict[Tuple[int, int], CardObject]] = None # {(player_idx, hand_idx): Card}
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


    def initialize(self, initial_observation: AgentObservation, initial_hand: List[CardObject], initial_peek_indices: Tuple[int, ...]):
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

        # --- 2. Process Snap Results & Determine Removals/Adds ---
        action = observation.action
        actor = observation.acting_player
        snap_results = observation.snap_results

        # Sets to track indices REMOVED from each player's perspective
        # These indices are relative to the state *before* the snap phase actions
        own_indices_removed: Set[int] = set()
        opponent_indices_removed: Set[int] = set()

        # Track how many cards were ADDED to each hand (penalty, SnapOpponentMove)
        own_cards_added_count = 0
        opponent_cards_added_count = 0

        # Keep track of original states for reconciliation
        original_own_hand = dict(self.own_hand)
        original_opponent_belief = dict(self.opponent_belief)
        original_opponent_last_seen = dict(self.opponent_last_seen_turn)

        for snap_info in snap_results:
            snapper = snap_info.get('snapper')
            success = snap_info.get('success', False)
            penalty = snap_info.get('penalty', False)
            snapped_card = snap_info.get('snapped_card') # Actual Card object or None
            removed_own_idx = snap_info.get('removed_own_index')
            removed_opp_idx = snap_info.get('removed_opponent_index')
            num_penalty_cards = self.config.cambia_rules.penaltyDrawCount

            if snapper == self.player_id: # Our snap attempt
                if success:
                    if removed_own_idx is not None: # SnapOwn success
                        own_indices_removed.add(removed_own_idx)
                        logger.debug(f"Agent {self.player_id} accounting for own removal at index {removed_own_idx} (SnapOwn).")
                    elif removed_opp_idx is not None: # SnapOpponent success (Opponent loses card)
                        opponent_indices_removed.add(removed_opp_idx)
                        logger.debug(f"Agent {self.player_id} accounting for opponent removal at index {removed_opp_idx} (SnapOpponent).")
                        # Opponent gains a card LATER when we move one (handled by expected count)
                elif penalty:
                    logger.debug(f"Agent {self.player_id} accounting for own penalty draw ({num_penalty_cards} cards).")
                    own_cards_added_count += num_penalty_cards

            elif snapper == self.opponent_id: # Opponent's snap attempt
                if success:
                    if removed_own_idx is not None: # Opponent SnapOwn success
                        opponent_indices_removed.add(removed_own_idx)
                        logger.debug(f"Agent {self.player_id} accounting for opponent removal at index {removed_own_idx} (Opponent SnapOwn).")
                    elif removed_opp_idx is not None: # Opponent SnapOpponent success (We lose card)
                        own_indices_removed.add(removed_opp_idx)
                        logger.debug(f"Agent {self.player_id} accounting for own removal at index {removed_opp_idx} (Opponent SnapOpponent).")
                        # We gain a card when opponent moves one (handled by expected count)
                elif penalty:
                    logger.debug(f"Agent {self.player_id} accounting for opponent penalty draw ({num_penalty_cards} cards).")
                    opponent_cards_added_count += num_penalty_cards

        # --- 3. Reconcile Hand States (using new robust logic) ---
        self.own_hand = self._rebuild_hand_state(
            original_dict=original_own_hand,
            removed_indices=own_indices_removed,
            expected_count=observed_own_count,
            is_own_hand=True
        )
        new_opponent_belief, self.opponent_last_seen_turn = self._rebuild_belief_state(
            original_belief_dict=original_opponent_belief,
            original_last_seen_dict=original_opponent_last_seen,
            removed_indices=opponent_indices_removed,
            expected_count=observed_opp_count
        )
        self.opponent_belief = new_opponent_belief

        # Update opponent card count state *after* reconciliation
        self.opponent_card_count = observed_opp_count


        # --- 4. Updates based on Main Action ---
        # This logic should now operate on the correctly reconciled state
        if action:
            if actor == self.player_id:
                if isinstance(action, ActionReplace):
                    if observation.drawn_card:
                        target_idx = action.target_hand_index
                        drawn_bucket = get_card_bucket(observation.drawn_card)
                        if target_idx in self.own_hand:
                            # Update the known card info directly
                            self.own_hand[target_idx] = KnownCardInfo(bucket=drawn_bucket, last_seen_turn=self._current_game_turn, card=observation.drawn_card)
                            logger.debug(f"Agent {self.player_id} updated own hand index {target_idx} to {drawn_bucket.name} after replace.")
                        else:
                            # This might happen if reconciliation failed, but should be less likely now
                            logger.warning(f"Agent {self.player_id} Replace target index {target_idx} not found after reconciliation (Current indices: {list(self.own_hand.keys())}). State likely inconsistent.")
                    else:
                        logger.warning("Replace action observed for self, but no drawn card in observation.")

                elif isinstance(action, ActionDiscard):
                     # Discarding drawn card, no direct belief update needed here
                     # Ability effects might trigger peeks/swaps handled below
                     pass

                # --- Handle Peek/Look Actions ---
                elif isinstance(action, (ActionAbilityPeekOwnSelect, ActionAbilityKingLookSelect)):
                    if observation.peeked_cards:
                        for (p_idx, h_idx), card in observation.peeked_cards.items():
                            if p_idx == self.player_id and h_idx in self.own_hand:
                                peeked_bucket = get_card_bucket(card)
                                self.own_hand[h_idx] = KnownCardInfo(bucket=peeked_bucket, last_seen_turn=self._current_game_turn, card=card)
                                action_type = "PeekOwn" if isinstance(action, ActionAbilityPeekOwnSelect) else "KingLook"
                                logger.debug(f"Agent {self.player_id} ({action_type}) updated own index {h_idx} knowledge to {peeked_bucket.name}.")
                            elif p_idx == self.opponent_id and h_idx in self.opponent_belief:
                                peeked_bucket = get_card_bucket(card)
                                self.opponent_belief[h_idx] = peeked_bucket
                                self.opponent_last_seen_turn[h_idx] = self._current_game_turn
                                action_type = "KingLook" # Only KingLook involves peeking opponent
                                logger.debug(f"Agent {self.player_id} ({action_type}) updated opponent index {h_idx} belief to {peeked_bucket.name}.")

                elif isinstance(action, ActionAbilityPeekOtherSelect):
                    if observation.peeked_cards:
                        for (p_idx, h_idx), card in observation.peeked_cards.items():
                            if p_idx == self.opponent_id and h_idx in self.opponent_belief:
                                peeked_bucket = get_card_bucket(card)
                                self.opponent_belief[h_idx] = peeked_bucket
                                self.opponent_last_seen_turn[h_idx] = self._current_game_turn
                                logger.debug(f"Agent {self.player_id} (PeekOther) updated opponent index {h_idx} belief to {peeked_bucket.name}.")

                # --- Handle Swap Actions ---
                elif isinstance(action, ActionAbilityBlindSwapSelect):
                    own_idx, opp_idx = action.own_hand_index, action.opponent_hand_index
                    if own_idx in self.own_hand:
                        # Our card at own_idx is now unknown
                        self.own_hand[own_idx] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn, card=None)
                        logger.debug(f"Agent {self.player_id} (BlindSwap) updated own index {own_idx} to UNKNOWN.")
                    # Trigger decay/unknown for opponent's involved index
                    self._trigger_event_decay(target_index=opp_idx, trigger_event="swap (blind)", current_turn=self._current_game_turn)

                elif isinstance(action, ActionAbilityKingSwapDecision) and action.perform_swap:
                    # Find involved indices from the peeked_cards (should be available from LookSelect step)
                    if observation.peeked_cards:
                        involved_indices = list(observation.peeked_cards.keys())
                        if len(involved_indices) == 2:
                            own_idx, opp_idx = (-1, -1)
                            # Determine which index belongs to whom
                            for p_idx, h_idx in involved_indices:
                                if p_idx == self.player_id: own_idx = h_idx
                                elif p_idx == self.opponent_id: opp_idx = h_idx

                            if own_idx != -1 and opp_idx != -1:
                                if own_idx in self.own_hand:
                                    self.own_hand[own_idx] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn, card=None)
                                    logger.debug(f"Agent {self.player_id} (KingSwap) updated own index {own_idx} to UNKNOWN.")
                                # Trigger decay/unknown for opponent's involved index
                                self._trigger_event_decay(target_index=opp_idx, trigger_event="swap (king)", current_turn=self._current_game_turn)
                            else:
                                logger.warning("Could not determine indices for King Swap update from peeked_cards.")
                        else:
                            logger.warning("Incorrect peeked_cards format for King Swap update.")
                    else:
                        logger.warning("Missing peeked_cards info for King Swap update.")

                elif isinstance(action, ActionSnapOpponentMove):
                    # Own hand count reduced by 1 (handled by reconciliation).
                    # Opponent hand count increased by 1 (handled by reconciliation adding UNKNOWN).
                    # No further belief update needed here based on the move action itself.
                    pass

            elif actor == self.opponent_id:
                # Observe opponent actions and update beliefs accordingly
                if isinstance(action, ActionReplace):
                    target_idx = action.target_hand_index # Opponent's index that got replaced
                    # Trigger decay/unknown for the replaced index
                    self._trigger_event_decay(target_index=target_idx, trigger_event="replace (opponent)", current_turn=self._current_game_turn)

                elif isinstance(action, ActionDiscard):
                     # If opponent discards and uses ability that reveals info to us (not possible in current rules)
                     pass

                elif isinstance(action, ActionAbilityBlindSwapSelect):
                    opp_own_idx = action.own_hand_index # Index in opponent's hand involved in swap
                    our_idx = action.opponent_hand_index # Index in our hand involved in swap
                    # Trigger decay for opponent's index
                    self._trigger_event_decay(target_index=opp_own_idx, trigger_event="swap (blind, opponent)", current_turn=self._current_game_turn)
                    # Update our own hand to UNKNOWN
                    if our_idx in self.own_hand:
                        self.own_hand[our_idx] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn, card=None)
                        logger.debug(f"Agent {self.player_id} updated own index {our_idx} to UNKNOWN due to opponent BlindSwap.")

                elif isinstance(action, ActionAbilityKingSwapDecision) and action.perform_swap:
                    # Opponent performed a King swap.
                    if observation.peeked_cards:
                         our_involved_idx = -1
                         opp_involved_idx = -1
                         # Check if *our* card was involved (meaning opponent peeked us)
                         for p_idx, h_idx in observation.peeked_cards.keys():
                             if p_idx == self.player_id: our_involved_idx = h_idx
                             elif p_idx == self.opponent_id: opp_involved_idx = h_idx

                         # Update our hand if involved
                         if our_involved_idx != -1 and our_involved_idx in self.own_hand:
                              self.own_hand[our_involved_idx] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn, card=None)
                              logger.debug(f"Agent {self.player_id} updated own index {our_involved_idx} to UNKNOWN due to opponent KingSwap.")
                         # Trigger decay for opponent's involved index
                         if opp_involved_idx != -1:
                              self._trigger_event_decay(target_index=opp_involved_idx, trigger_event="swap (king, opponent)", current_turn=self._current_game_turn)
                         elif our_involved_idx == -1 and opp_involved_idx == -1:
                              logger.debug("Opponent King Swap observed, but couldn't identify involved indices from peeked_cards.")
                    else:
                         # If we have no peek info, we cannot know which indices were swapped.
                         # Decay isn't safe to apply broadly. Might consider full belief reset if desired.
                         logger.debug("Opponent King Swap observed without peek info, cannot update specific beliefs.")

                elif isinstance(action, ActionSnapOpponentMove):
                    # Opponent successfully snapped one of *their* cards, then moved one to *us*.
                    # Our hand count increased by 1 (handled by reconciliation adding UNKNOWN).
                    # Opponent count stayed the same (loss + gain). Reconciliation ensures correct count.
                    # We don't know which card they moved, so no specific belief update.
                    pass

        # --- 5. Apply Time Decay (Level 2) ---
        if self.memory_level == 2:
            self._apply_time_decay(self._current_game_turn)


    def _rebuild_hand_state(self, original_dict: Dict[int, KnownCardInfo], removed_indices: Set[int], expected_count: int, is_own_hand: bool) -> Dict[int, KnownCardInfo]:
        """Rebuilds own_hand ensuring contiguous indices 0..N-1."""
        new_dict: Dict[int, KnownCardInfo] = {}
        current_items = []

        # Collect valid items from original dict, excluding removed ones
        for idx, item in sorted(original_dict.items()):
             if idx not in removed_indices:
                  current_items.append(item)

        # Assign items to new contiguous indices
        for i, item in enumerate(current_items):
             if i < expected_count:
                 new_dict[i] = item
             else:
                  # This should only happen if expected_count < actual items after removal (logic error elsewhere)
                  player_desc = "Own" if is_own_hand else "Opponent"
                  logger.error(f"Rebuild {player_desc} Hand: More items ({len(current_items)}) than expected ({expected_count}) after removal. Discarding extra.")
                  break # Stop adding extra items

        # Add UNKNOWNs if needed to reach expected count
        while len(new_dict) < expected_count:
             new_index = len(new_dict)
             if is_own_hand:
                 new_dict[new_index] = KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=self._current_game_turn, card=None)
             else: # Opponent belief doesn't store KnownCardInfo
                 # This function shouldn't be called for opponent belief, use _rebuild_belief_state
                 logger.error("_rebuild_hand_state called for opponent belief. Use _rebuild_belief_state.")
                 break # Avoid infinite loop

        # Final check (optional but good for debugging)
        if len(new_dict) != expected_count:
             player_desc = "Own" if is_own_hand else "Opponent"
             logger.error(f"FATAL: {player_desc} hand final rebuild failed. Expected {expected_count}, got {len(new_dict)}. Indices: {sorted(list(new_dict.keys()))}")

        return new_dict

    def _rebuild_belief_state(self, original_belief_dict: Dict[int, Union[CardBucket, DecayCategory]], original_last_seen_dict: Dict[int, int], removed_indices: Set[int], expected_count: int) -> Tuple[Dict[int, Union[CardBucket, DecayCategory]], Dict[int, int]]:
        """Rebuilds opponent_belief and opponent_last_seen_turn ensuring contiguous indices."""
        new_belief_dict: Dict[int, Union[CardBucket, DecayCategory]] = {}
        new_last_seen_dict: Dict[int, int] = {}
        current_belief_items = []
        current_last_seen_mapping = {} # Map old index to last seen turn

        # Collect valid items from original dicts, excluding removed ones
        for idx, belief in sorted(original_belief_dict.items()):
             if idx not in removed_indices:
                  current_belief_items.append(belief)
                  if idx in original_last_seen_dict:
                       # Store last seen time temporarily, associating with the belief *value*
                       current_last_seen_mapping[belief] = original_last_seen_dict[idx] # Note: This assumes beliefs are unique enough or last seen applies to the bucket

        # Re-index beliefs and associated last_seen times
        for i, belief in enumerate(current_belief_items):
             if i < expected_count:
                 new_belief_dict[i] = belief
                 # Try to restore last_seen time if it was associated
                 if belief in current_last_seen_mapping:
                     new_last_seen_dict[i] = current_last_seen_mapping[belief]
                     # Optional: Remove from mapping if consumed? Depends if buckets can repeat.
                     # It's safer to allow multiple indices to share the last_seen if buckets repeat.
             else:
                  logger.error(f"Rebuild Opponent Belief: More items ({len(current_belief_items)}) than expected ({expected_count}) after removal. Discarding extra.")
                  break

        # Add UNKNOWNs if needed
        while len(new_belief_dict) < expected_count:
             new_index = len(new_belief_dict)
             new_belief_dict[new_index] = CardBucket.UNKNOWN
             # Ensure no stale last_seen entry exists for the new UNKNOWN index
             new_last_seen_dict.pop(new_index, None)

        # Final check
        if len(new_belief_dict) != expected_count:
             logger.error(f"FATAL: Opponent belief final rebuild failed. Expected {expected_count}, got {len(new_belief_dict)}. Indices: {sorted(list(new_belief_dict.keys()))}")
        # Ensure last_seen only contains keys present in the final belief dict
        valid_last_seen = {k: v for k, v in new_last_seen_dict.items() if k in new_belief_dict}

        return new_belief_dict, valid_last_seen


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
              # Only decay if it's specific knowledge (CardBucket), not already decayed (DecayCategory) or UNKNOWN
              if isinstance(current_belief, CardBucket) and current_belief != CardBucket.UNKNOWN:
                   decayed_category = decay_bucket(current_belief)
                   self.opponent_belief[target_index] = decayed_category
                   # Clear the last seen timestamp when decay happens
                   self.opponent_last_seen_turn.pop(target_index, None)
                   logger.debug(f"Agent {self.player_id} decaying Opponent[{target_index}] belief from {current_belief.name} to {decayed_category.name} due to {trigger_event}.")
              elif isinstance(current_belief, DecayCategory):
                  logger.debug(f"Agent {self.player_id} decay trigger skipped for Opponent[{target_index}]: Already decayed to {current_belief.name}.")
              else: # Already UNKNOWN
                  logger.debug(f"Agent {self.player_id} decay trigger skipped for Opponent[{target_index}]: Already UNKNOWN.")
         # else: logger.warning(f"Agent {self.player_id} decay trigger skipped: Index {target_index} not in opponent belief {list(self.opponent_belief.keys())}.") # Reduced noise

    def _apply_time_decay(self, current_turn: int):
        """Applies time-based decay (Level 2 only)."""
        if self.memory_level != 2: return

        indices_to_decay = []
        for idx, last_seen in self.opponent_last_seen_turn.items():
            # Check if enough turns have passed since last *specific* observation
            if current_turn - last_seen >= self.time_decay_turns:
                 # Ensure the belief hasn't already decayed or become unknown
                 if idx in self.opponent_belief and isinstance(self.opponent_belief[idx], CardBucket) and self.opponent_belief[idx] != CardBucket.UNKNOWN:
                      indices_to_decay.append(idx)

        for idx in indices_to_decay:
            current_belief = self.opponent_belief[idx] # Should be CardBucket here
            decayed_category = decay_bucket(current_belief)
            self.opponent_belief[idx] = decayed_category
            # Remove timestamp after decay
            self.opponent_last_seen_turn.pop(idx)
            logger.debug(f"Agent {self.player_id} applying time decay for Opponent[{idx}] from {current_belief.name} to {decayed_category.name}.")


    def get_infoset_key(self) -> InfosetKey:
        """Constructs the canonical, hashable infoset key from the current belief state."""
        # 1. Own Hand Buckets (Tuple sorted by index)
        own_hand_items = sorted(self.own_hand.items()) # Sort by index 0..N-1
        own_hand_buckets = tuple(info.bucket.value for _, info in own_hand_items)

        # 2. Opponent Beliefs (Tuple indexed 0..N-1)
        opp_belief_values = []
        # Use self.opponent_card_count which is reconciled
        for i in range(self.opponent_card_count):
             belief = self.opponent_belief.get(i, CardBucket.UNKNOWN) # Get belief for contiguous index i
             # Ensure we get the .value attribute correctly for both Enums
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
        # The DecisionContext value is added externally by the CFR trainer
        return key # type: ignore

    def get_potential_opponent_snap_indices(self, target_rank: str) -> List[int]:
         """ Returns opponent hand indices the agent *believes* could match the target rank. """
         matching_indices = []
         # Need a dummy card to get the target bucket(s)
         target_card_black = Card(rank=target_rank, suit='S') # Suit matters for King
         target_card_red = Card(rank=target_rank, suit='H')
         target_bucket_black = get_card_bucket(target_card_black)
         target_bucket_red = get_card_bucket(target_card_red)
         target_buckets = {target_bucket_black}
         if target_bucket_black != target_bucket_red: # Add red king bucket if different
              target_buckets.add(target_bucket_red)

         for idx, belief in self.opponent_belief.items():
              # If belief is unknown or a general category, assume it *could* match
              if belief == CardBucket.UNKNOWN or isinstance(belief, DecayCategory):
                   matching_indices.append(idx)
              # If belief is a specific bucket, check if it's one of the target buckets
              elif isinstance(belief, CardBucket) and belief in target_buckets:
                   matching_indices.append(idx)
              # Add specific logic if decay categories imply rank possibility (e.g., LIKELY_LOW vs Ace)
              # For now, only specific buckets or unknown/decayed match

         return matching_indices


    def clone(self) -> 'AgentState':
        """Creates a deep copy of the agent state."""
        # Use copy.deepcopy for nested mutable structures like dictionaries
        new_state = AgentState(
             player_id=self.player_id, opponent_id=self.opponent_id,
             memory_level=self.memory_level, time_decay_turns=self.time_decay_turns,
             initial_hand_size=self.initial_hand_size, config=self.config # Config can be shallow copied
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
        # Opponent belief is now guaranteed contiguous 0..N-1
        opp_belief_str = {i: self.opponent_belief.get(i, CardBucket.UNKNOWN).name
                          for i in range(self.opponent_card_count)}

        return (f"AgentState(P{self.player_id}, GT:{self._current_game_turn}, Phase:{self.game_phase.name}, "
                f"OH({len(self.own_hand)}):{own_hand_str}, "
                f"OB({self.opponent_card_count}):{opp_belief_str}, "
                f"Disc:{self.known_discard_top_bucket.name}, Stock:{self.stockpile_estimate.name}, "
                f"Cambia:{self.cambia_caller})")