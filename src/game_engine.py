# src/game_engine.py
import random
from typing import List, Tuple, Optional, Set, Any, Dict, TypeAlias, Callable, Deque
from dataclasses import dataclass, field
import logging
import copy
import numpy as np
import os
import sys
from collections import deque

from .card import Card, create_standard_deck
from .constants import (
    KING, QUEEN, JACK, NINE, TEN, SEVEN, EIGHT,
    GameAction, ActionPassSnap, ActionSnapOwn, ActionSnapOpponent, ActionSnapOpponentMove,
    ActionDiscard, ActionReplace, ActionCallCambia, ActionDrawStockpile, ActionDrawDiscard,
    ActionAbilityPeekOwnSelect, ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect, ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    NUM_PLAYERS
)
from .config import CambiaRulesConfig

logger = logging.getLogger(__name__)

# --- Delta Update Types ---
StateDeltaChange: TypeAlias = Tuple[str, ...]
StateDelta: TypeAlias = List[StateDeltaChange]
UndoInfo: TypeAlias = Callable[[], None]

@dataclass
class PlayerState:
     hand: List[Card] = field(default_factory=list)
     initial_peek_indices: Tuple[int, ...] = (0, 1)

@dataclass
class CambiaGameState:
    """Represents the true, objective state of a 1v1 Cambia game. Now uses delta-based updates."""
    players: List[PlayerState] = field(default_factory=list)
    stockpile: List[Card] = field(default_factory=list)
    discard_pile: List[Card] = field(default_factory=list)
    current_player_index: int = 0
    num_players: int = NUM_PLAYERS
    cambia_caller_id: Optional[int] = None
    turns_after_cambia: int = 0
    house_rules: CambiaRulesConfig = field(default_factory=CambiaRulesConfig)
    _game_over: bool = False
    _winner: Optional[int] = None
    _utilities: List[float] = field(default_factory=lambda: [0.0] * NUM_PLAYERS)
    _turn_number: int = 0

    pending_action: Optional[GameAction] = None
    pending_action_player: Optional[int] = None
    pending_action_data: Dict[str, Any] = field(default_factory=dict)

    snap_phase_active: bool = False
    snap_discarded_card: Optional[Card] = None
    snap_potential_snappers: List[int] = field(default_factory=list)
    snap_current_snapper_idx: int = 0
    snap_results_log: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.players:
            self._setup_game()

    def _setup_game(self):
        """Initializes the deck, shuffles, and deals cards."""
        self.stockpile = create_standard_deck(include_jokers=self.house_rules.use_jokers)
        random.shuffle(self.stockpile)
        initial_peek_count = self.house_rules.initial_view_count
        cards_per_player = self.house_rules.cards_per_player
        self.players = [PlayerState(initial_peek_indices=tuple(range(initial_peek_count))) for _ in range(self.num_players)]

        for _ in range(cards_per_player):
            for i in range(self.num_players):
                if self.stockpile:
                    if i < len(self.players) and hasattr(self.players[i], 'hand') and isinstance(self.players[i].hand, list):
                         self.players[i].hand.append(self.stockpile.pop())
                    else:
                         logger.error(f"Error during setup: Player object {i} invalid or missing hand list.")
                         raise RuntimeError(f"Game setup failed due to invalid PlayerState {i}")
                else:
                    logger.warning("Stockpile empty during initial deal!")
                    break
        self.discard_pile = []
        self.current_player_index = random.randint(0, self.num_players - 1)
        self.cambia_caller_id = None
        self.turns_after_cambia = 0
        self._game_over = False
        self._winner = None
        self._utilities = [0.0] * self.num_players
        self._turn_number = 0
        self.pending_action = None
        self.pending_action_player = None
        self.pending_action_data = {}
        self.snap_phase_active = False
        self.snap_results_log = []
        logger.debug(f"Game setup complete. Player {self.current_player_index} starts (Turn {self._turn_number}).")
        logger.debug(f"Initial hands (hidden): {[len(p.hand) if hasattr(p, 'hand') else 'ERROR' for p in self.players]}")
        logger.debug(f"House Rules: {self.house_rules}")

    # --- Helper for the delta-based undo mechanism ---
    def _add_change(self, change_func: Callable[[], Any], undo_func: Callable[[], None], delta: StateDeltaChange, undo_stack: Deque, delta_list: StateDelta):
         """Applies change, adds undo, records delta."""
         change_func()
         delta_list.append(delta)
         undo_stack.appendleft(undo_func) # Add undo op to the *front* (LIFO)

    def get_player_hand(self, player_index: int) -> List[Card]:
        if 0 <= player_index < len(self.players) and hasattr(self.players[player_index], 'hand'):
            return self.players[player_index].hand
        logger.error(f"Invalid player index {player_index} or player object missing hand in get_player_hand.")
        return []

    def get_opponent_index(self, player_index: int) -> int:
        return 1 - player_index

    def get_player_card_count(self, player_index: int) -> int:
         if 0 <= player_index < len(self.players) and hasattr(self.players[player_index], 'hand'):
            return len(self.players[player_index].hand)
         logger.warning(f"Invalid player index {player_index} or missing hand in get_player_card_count. Returning 0.")
         return 0

    def get_stockpile_size(self) -> int:
        return len(self.stockpile)

    def get_discard_top(self) -> Optional[Card]:
        return self.discard_pile[-1] if self.discard_pile else None

    def get_turn_number(self) -> int:
        return self._turn_number

    def get_legal_actions(self) -> Set[GameAction]:
        """Returns the set of valid actions for the current acting player."""
        legal_actions: Set[GameAction] = set()

        if self._game_over:
            return legal_actions

        acting_player = self.get_acting_player()
        if acting_player == -1:
             logger.error("Cannot get legal actions: Invalid acting player (-1).")
             return legal_actions

        # --- Snap Phase Actions ---
        if self.snap_phase_active:
            if not (0 <= acting_player < len(self.players) and hasattr(self.players[acting_player], 'hand')):
                 logger.error(f"Snap phase: Acting player {acting_player} object invalid or missing hand.")
                 return legal_actions
            snapper_hand = self.players[acting_player].hand
            if self.snap_discarded_card is None:
                 logger.error("Snap phase active but snap_discarded_card is None.")
                 return legal_actions
            target_rank = self.snap_discarded_card.rank

            if not all(isinstance(card, Card) for card in snapper_hand):
                 logger.error(f"Snap phase: Player {acting_player}'s hand contains non-Card objects: {snapper_hand}")
                 return legal_actions

            legal_actions.add(ActionPassSnap())
            for i, card in enumerate(snapper_hand):
                if card.rank == target_rank:
                    legal_actions.add(ActionSnapOwn(own_card_hand_index=i))
            if self.house_rules.allowOpponentSnapping:
                 opponent_idx = self.get_opponent_index(acting_player)
                 if 0 <= opponent_idx < len(self.players) and hasattr(self.players[opponent_idx], 'hand'):
                      opponent_hand = self.players[opponent_idx].hand
                      if not all(isinstance(card, Card) for card in opponent_hand):
                          logger.error(f"Snap phase: Opponent {opponent_idx}'s hand contains non-Card objects: {opponent_hand}")
                      else:
                          for i, card in enumerate(opponent_hand):
                               if card.rank == target_rank:
                                    legal_actions.add(ActionSnapOpponent(opponent_target_hand_index=i))
                 else:
                     logger.warning(f"Snap phase: Opponent object {opponent_idx} invalid or missing hand, cannot check for SnapOpponent.")
            return legal_actions

        # --- Pending Action Resolution ---
        elif self.pending_action:
            if acting_player != self.pending_action_player:
                 logger.error(f"Legal actions requested for P{acting_player} but pending action is for P{self.pending_action_player}")
                 return legal_actions
            if not (0 <= acting_player < len(self.players) and hasattr(self.players[acting_player], 'hand')):
                 logger.error(f"Pending action: Acting player {acting_player} object invalid or missing hand.")
                 return legal_actions

            action_type = self.pending_action
            player = self.pending_action_player

            if isinstance(action_type, ActionDiscard): # Post-Draw Choice
                 legal_actions.add(ActionDiscard(use_ability=False))
                 drawn_card = self.pending_action_data.get("drawn_card")
                 if drawn_card and self._card_has_discard_ability(drawn_card):
                      legal_actions.add(ActionDiscard(use_ability=True))
                 for i in range(self.get_player_card_count(player)):
                      legal_actions.add(ActionReplace(target_hand_index=i))
            elif isinstance(action_type, ActionAbilityPeekOwnSelect):
                 for i in range(self.get_player_card_count(player)):
                      legal_actions.add(ActionAbilityPeekOwnSelect(target_hand_index=i))
            elif isinstance(action_type, ActionAbilityPeekOtherSelect):
                 opp_idx = self.get_opponent_index(player)
                 for i in range(self.get_player_card_count(opp_idx)):
                      legal_actions.add(ActionAbilityPeekOtherSelect(target_opponent_hand_index=i))
            elif isinstance(action_type, ActionAbilityBlindSwapSelect):
                 own_count = self.get_player_card_count(player)
                 opp_idx = self.get_opponent_index(player)
                 opp_count = self.get_player_card_count(opp_idx)
                 for i in range(own_count):
                      for j in range(opp_count):
                           legal_actions.add(ActionAbilityBlindSwapSelect(own_hand_index=i, opponent_hand_index=j))
            elif isinstance(action_type, ActionAbilityKingLookSelect):
                 own_count = self.get_player_card_count(player)
                 opp_idx = self.get_opponent_index(player)
                 opp_count = self.get_player_card_count(opp_idx)
                 for i in range(own_count):
                      for j in range(opp_count):
                           legal_actions.add(ActionAbilityKingLookSelect(own_hand_index=i, opponent_hand_index=j))
            elif isinstance(action_type, ActionAbilityKingSwapDecision):
                 legal_actions.add(ActionAbilityKingSwapDecision(perform_swap=True))
                 legal_actions.add(ActionAbilityKingSwapDecision(perform_swap=False))
            elif isinstance(action_type, ActionSnapOpponentMove):
                 snapper_idx = player
                 target_slot = self.pending_action_data.get("target_empty_slot_index")
                 if target_slot is None: logger.error("Missing target_empty_slot_index for SnapOpponentMove")
                 else:
                     for i in range(self.get_player_card_count(snapper_idx)):
                          legal_actions.add(ActionSnapOpponentMove(own_card_to_move_hand_index=i, target_empty_slot_index=target_slot))
            else:
                 logger.error(f"Unknown pending action type for legal actions: {action_type}")
            return legal_actions

        # --- Standard Start-of-Turn Actions ---
        player = self.current_player_index
        if not (0 <= player < len(self.players) and hasattr(self.players[player], 'hand')):
            logger.error(f"Start of turn: Player {player} object invalid or missing hand.")
            return legal_actions

        can_draw_stockpile = bool(self.stockpile) or (len(self.discard_pile) > 1)
        can_draw_discard = self.house_rules.allowDrawFromDiscardPile and self.discard_pile

        if can_draw_stockpile:
            legal_actions.add(ActionDrawStockpile())
        if can_draw_discard:
             legal_actions.add(ActionDrawDiscard())

        cambia_allowed_round = self.house_rules.cambia_allowed_round
        if self.cambia_caller_id is None and (self._turn_number // self.num_players >= cambia_allowed_round):
             legal_actions.add(ActionCallCambia())

        if not legal_actions and not (can_draw_stockpile or can_draw_discard) and not self._game_over:
             logger.warning(f"No legal actions possible and cannot draw/reshuffle for P{player}. State: {self}. Ending game.")
             pass
        elif not legal_actions and not self._game_over:
             logger.warning(f"No legal actions found for player {player} at start of turn in state: {self}. Ending game.")

        return legal_actions

    def _card_has_discard_ability(self, card: Card) -> bool:
        """Checks if a card has an ability when discarded from draw."""
        return card.rank in [SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING]

    def _serialize_card(self, card: Optional[Card]) -> Optional[str]:
        """Serializes a card to string or None."""
        return str(card) if card else None

    def apply_action(self, action: GameAction) -> Tuple[StateDelta, UndoInfo]:
        """
        Applies the given action by modifying 'self' and returns a StateDelta list
        and a callable UndoInfo.
        """
        if self._game_over:
            logger.warning("Attempted action on a finished game.")
            return [], lambda: None

        acting_player = self.get_acting_player()
        if not (0 <= acting_player < len(self.players) and hasattr(self.players[acting_player], 'hand')):
             logger.error(f"Apply Action: Invalid acting player {acting_player} or missing hand. State: {self}")
             self._game_over = True
             self._calculate_final_scores()
             return [], lambda: None

        # --- Undo Stack & Delta List for this action ---
        undo_stack: Deque[Callable[[], None]] = deque()
        delta_list: StateDelta = []

        # --- Master Undo Function ---
        def undo_action():
            # Execute undo operations in reverse order of changes
            while undo_stack:
                try:
                     undo_func = undo_stack.popleft() # Pop from front (LIFO)
                     undo_func()
                except Exception as e:
                     logger.exception(f"Error during undo operation {undo_func}: {e}. State might be inconsistent.")

        # --- Action Application Logic (Modifies self, uses self._add_change) ---
        try:
            # --- Snap Phase Action Handling ---
            if self.snap_phase_active:
                 if acting_player != self.snap_potential_snappers[self.snap_current_snapper_idx]:
                     logger.error(f"Action {action} received, but expected snap action from P{self.snap_potential_snappers[self.snap_current_snapper_idx]}. Ignoring.")
                     return [], lambda: None

                 logger.debug(f"Snap Phase (Turn {self._turn_number}): Player {acting_player} choosing action: {action}")
                 if self.snap_discarded_card is None:
                     logger.error("Apply action in snap phase, but snap_discarded_card is None. Cannot proceed.")
                     self._end_snap_phase(undo_stack, delta_list)
                     return delta_list, undo_action

                 target_rank = self.snap_discarded_card.rank
                 snap_success = False
                 snap_penalty = False
                 removed_card_info: Optional[Tuple[int, int, Card]] = None
                 snapped_opponent_card_info: Optional[Tuple[int, int, Card]] = None
                 attempted_card: Optional[Card] = None
                 action_type_str = type(action).__name__
                 snap_details: Dict[str, Any] = {}

                 # Helper for logging snap results (uses _add_change with explicit args)
                 def log_snap_result(details_dict):
                      original_log = list(self.snap_results_log)
                      def change(): self.snap_results_log.append(details_dict)
                      def undo(): self.snap_results_log = original_log
                      # Explicitly pass undo_stack and delta_list
                      self._add_change(change, undo, ('snap_log_append', details_dict), undo_stack, delta_list)

                 if isinstance(action, ActionPassSnap):
                      logger.debug(f"Player {acting_player} passes snap.")
                      snap_details = { "snapper": acting_player, "action_type": action_type_str, "target_rank": target_rank, "success": False, "penalty": False, "details": "Passed", "snapped_card": None }
                      log_snap_result(snap_details)

                 elif isinstance(action, ActionSnapOwn):
                      snap_idx = action.own_card_hand_index
                      hand = self.players[acting_player].hand
                      if 0 <= snap_idx < len(hand):
                           attempted_card = hand[snap_idx]
                           if not isinstance(attempted_card, Card):
                                logger.error(f"SnapOwn: Card at index {snap_idx} is not a Card object: {attempted_card}. Applying penalty.")
                                snap_penalty = True
                           elif attempted_card.rank == target_rank:
                                card_to_remove = hand[snap_idx]
                                original_discard_len = len(self.discard_pile)
                                def change_snap_own():
                                     removed = self.players[acting_player].hand.pop(snap_idx)
                                     self.discard_pile.append(removed)
                                def undo_snap_own():
                                     popped_discard = self.discard_pile.pop()
                                     self.players[acting_player].hand.insert(snap_idx, popped_discard)
                                delta_snap_own = ('snap_own_success', acting_player, snap_idx, self._serialize_card(card_to_remove))
                                self._add_change(change_snap_own, undo_snap_own, delta_snap_own, undo_stack, delta_list) # Pass args
                                snap_success = True
                                removed_card_info = (acting_player, snap_idx, card_to_remove)
                                logger.info(f"Player {acting_player} snaps own {card_to_remove} (matching {target_rank}) from index {snap_idx}. Hand size: {len(self.players[acting_player].hand)}")
                           else: snap_penalty = True
                      else: snap_penalty = True

                      if snap_penalty:
                           logger.warning(f"Player {acting_player} attempted invalid Snap Own: {action} (Attempted: {attempted_card}). Applying penalty.")
                           penalty_deltas = self._apply_penalty(acting_player, self.house_rules.penaltyDrawCount, undo_stack)
                           delta_list.extend(penalty_deltas)
                      snap_details = {
                          "snapper": acting_player, "action_type": action_type_str, "target_rank": target_rank,
                          "success": snap_success, "penalty": snap_penalty,
                          "removed_own_index": removed_card_info[1] if removed_card_info else None,
                          "snapped_card": self._serialize_card(removed_card_info[2]) if removed_card_info else None,
                          "attempted_card": self._serialize_card(attempted_card) if snap_penalty and attempted_card else None
                      }
                      log_snap_result(snap_details)

                 elif isinstance(action, ActionSnapOpponent):
                      if not self.house_rules.allowOpponentSnapping:
                           logger.error("Invalid Action: SnapOpponent attempted but rule disallowed.")
                           snap_penalty = True
                           penalty_deltas = self._apply_penalty(acting_player, self.house_rules.penaltyDrawCount, undo_stack)
                           delta_list.extend(penalty_deltas)
                           attempted_card = None
                      else:
                           opp_idx = self.get_opponent_index(acting_player)
                           if not (0 <= opp_idx < len(self.players) and hasattr(self.players[opp_idx], 'hand')):
                                logger.error(f"SnapOpponent: Opponent object {opp_idx} invalid or missing hand.")
                                snap_penalty = True
                                penalty_deltas = self._apply_penalty(acting_player, self.house_rules.penaltyDrawCount, undo_stack)
                                delta_list.extend(penalty_deltas)
                                attempted_card = None
                           else:
                                opp_hand = self.players[opp_idx].hand
                                target_opp_hand_idx = action.opponent_target_hand_index
                                if 0 <= target_opp_hand_idx < len(opp_hand):
                                     attempted_card = opp_hand[target_opp_hand_idx]
                                     if not isinstance(attempted_card, Card):
                                          logger.error(f"SnapOpponent: Card at opponent index {target_opp_hand_idx} is not a Card object: {attempted_card}. Applying penalty.")
                                          snap_penalty = True
                                     elif attempted_card.rank == target_rank:
                                          card_to_remove = opp_hand[target_opp_hand_idx]
                                          def change_snap_opp_remove(): self.players[opp_idx].hand.pop(target_opp_hand_idx)
                                          def undo_snap_opp_remove(): self.players[opp_idx].hand.insert(target_opp_hand_idx, card_to_remove)
                                          delta_snap_opp_remove = ('snap_opponent_remove', opp_idx, target_opp_hand_idx, self._serialize_card(card_to_remove))
                                          self._add_change(change_snap_opp_remove, undo_snap_opp_remove, delta_snap_opp_remove, undo_stack, delta_list) # Pass args
                                          snap_success = True
                                          snapped_opponent_card_info = (opp_idx, target_opp_hand_idx, card_to_remove)
                                          logger.info(f"Player {acting_player} snaps opponent's {card_to_remove} at index {target_opp_hand_idx}. Requires move.")
                                          # --- State Change: Set pending action for move ---
                                          original_pending = (self.pending_action, self.pending_action_player, copy.deepcopy(self.pending_action_data))
                                          original_snap_active = self.snap_phase_active
                                          def change_pending_move():
                                               self.pending_action = ActionSnapOpponentMove(own_card_to_move_hand_index=-1, target_empty_slot_index=-1)
                                               self.pending_action_player = acting_player
                                               self.pending_action_data = {"target_empty_slot_index": target_opp_hand_idx}
                                               self.snap_phase_active = False
                                          def undo_pending_move():
                                               self.pending_action, self.pending_action_player, self.pending_action_data = original_pending
                                               self.snap_phase_active = original_snap_active
                                          delta_pending = ('set_pending_action',
                                                           'ActionSnapOpponentMove', acting_player, {"target_empty_slot_index": target_opp_hand_idx},
                                                           type(original_pending[0]).__name__ if original_pending[0] else None, original_pending[1], original_pending[2])
                                          delta_snap_active = ('set_attr', 'snap_phase_active', False, original_snap_active)
                                          self._add_change(change_pending_move, undo_pending_move, delta_pending, undo_stack, delta_list) # Pass args
                                          delta_list.append(delta_snap_active) # Log snap active change separately
                                          # --- End State Change ---
                                          snap_details = {
                                             "snapper": acting_player, "action_type": action_type_str, "target_rank": target_rank,
                                             "success": snap_success, "penalty": False,
                                             "removed_opponent_index": target_opp_hand_idx,
                                             "snapped_card": self._serialize_card(card_to_remove)
                                          }
                                          log_snap_result(snap_details)
                                          return delta_list, undo_action # Return early
                                     else: snap_penalty = True
                                else: snap_penalty = True

                                if snap_penalty:
                                     logger.warning(f"Player {acting_player} attempted invalid Snap Opponent: {action} (Attempted: {attempted_card}). Applying penalty.")
                                     penalty_deltas = self._apply_penalty(acting_player, self.house_rules.penaltyDrawCount, undo_stack)
                                     delta_list.extend(penalty_deltas)
                      if not snap_success:
                           snap_details = {
                               "snapper": acting_player, "action_type": action_type_str, "target_rank": target_rank,
                               "success": False, "penalty": snap_penalty,
                               "snapped_card": None,
                               "attempted_card": self._serialize_card(attempted_card) if snap_penalty and attempted_card else None
                           }
                           log_snap_result(snap_details)

                 else:
                      logger.error(f"Invalid action type {type(action)} received during snap phase.")
                      snap_details = {
                          "snapper": acting_player, "action_type": "InvalidAction", "target_rank": target_rank,
                          "success": False, "penalty": False, "details": f"Received {type(action).__name__}", "snapped_card": None
                      }
                      log_snap_result(snap_details)

                 # --- State Change: Advance snap index ---
                 original_snap_idx_local = self.snap_current_snapper_idx
                 next_snap_idx = self.snap_current_snapper_idx + 1
                 def change_snap_idx(): self.snap_current_snapper_idx = next_snap_idx
                 def undo_snap_idx(): self.snap_current_snapper_idx = original_snap_idx_local
                 self._add_change(change_snap_idx, undo_snap_idx, ('set_attr', 'snap_current_snapper_idx', next_snap_idx, original_snap_idx_local), undo_stack, delta_list) # Pass args
                 # --- End State Change ---

                 if self.snap_current_snapper_idx >= len(self.snap_potential_snappers):
                      self._end_snap_phase(undo_stack, delta_list)
                 self._check_game_end(undo_stack, delta_list)
                 return delta_list, undo_action

            # --- Pending Action Resolution Handling ---
            elif self.pending_action:
                 if acting_player != self.pending_action_player:
                     logger.error(f"Action {action} received from P{acting_player} but pending action is for P{self.pending_action_player}")
                     return [], lambda: None

                 pending_type = self.pending_action
                 player = self.pending_action_player
                 discard_for_snap_check = None

                 if isinstance(pending_type, ActionDiscard) and isinstance(action, (ActionDiscard, ActionReplace)):
                      drawn_card = self.pending_action_data.get("drawn_card")
                      if not drawn_card:
                           logger.error("Pending post-draw choice but no drawn_card!");
                           self._clear_pending_action(undo_stack, delta_list)
                           return delta_list, undo_action
                      if isinstance(action, ActionDiscard):
                           logger.debug(f"Player {player} discards drawn {drawn_card}. Use ability: {action.use_ability}")
                           original_discard_len = len(self.discard_pile)
                           def change_discard(): self.discard_pile.append(drawn_card)
                           def undo_discard(): self.discard_pile.pop()
                           self._add_change(change_discard, undo_discard, ('list_append', 'discard_pile', original_discard_len, self._serialize_card(drawn_card)), undo_stack, delta_list) # Pass args
                           discard_for_snap_check = drawn_card
                           use_ability = action.use_ability and self._card_has_discard_ability(drawn_card)
                           self._clear_pending_action(undo_stack, delta_list) # Pass args
                           if use_ability:
                                self._trigger_discard_ability(player, drawn_card, undo_stack, delta_list) # Pass args
                                if self.pending_action: return delta_list, undo_action # Return if ability sets new pending state
                      elif isinstance(action, ActionReplace):
                           target_idx = action.target_hand_index
                           hand = self.players[player].hand
                           if 0 <= target_idx < len(hand):
                                replaced_card = hand[target_idx]
                                logger.debug(f"Player {player} replaces card at index {target_idx} ({replaced_card}) with drawn {drawn_card}.")
                                original_card_in_hand = hand[target_idx]
                                original_discard_len = len(self.discard_pile)
                                def change_replace():
                                     self.players[player].hand[target_idx] = drawn_card
                                     self.discard_pile.append(replaced_card)
                                def undo_replace():
                                     popped_discard = self.discard_pile.pop()
                                     self.players[player].hand[target_idx] = original_card_in_hand
                                     if popped_discard is not replaced_card: logger.error(f"Undo Replace Mismatch: Popped {popped_discard}, expected {replaced_card}")
                                delta_replace = ('replace_discard', player, target_idx, self._serialize_card(drawn_card), self._serialize_card(replaced_card))
                                self._add_change(change_replace, undo_replace, delta_replace, undo_stack, delta_list) # Pass args
                                discard_for_snap_check = replaced_card
                                self._clear_pending_action(undo_stack, delta_list) # Pass args
                           else: logger.error(f"Invalid REPLACE action index: {target_idx}"); self._clear_pending_action(undo_stack, delta_list) # Pass args

                 elif isinstance(pending_type, ActionAbilityPeekOwnSelect) and isinstance(action, ActionAbilityPeekOwnSelect):
                       target_idx = action.target_hand_index; hand = self.players[player].hand
                       peeked_card_str = "ERROR"
                       if 0 <= target_idx < len(hand): peeked_card_str = self._serialize_card(hand[target_idx]); logger.info(f"P{player} uses 7/8, peeks own {target_idx}: {peeked_card_str}")
                       else: logger.error(f"Invalid PEEK_OWN index {target_idx}");
                       discard_for_snap_check = self.get_discard_top()
                       self._clear_pending_action(undo_stack, delta_list) # Pass args
                       delta_list.append(('peek_own', player, target_idx, peeked_card_str))
                 elif isinstance(pending_type, ActionAbilityPeekOtherSelect) and isinstance(action, ActionAbilityPeekOtherSelect):
                       opp_idx = self.get_opponent_index(player); target_opp_idx = action.target_opponent_hand_index;
                       peeked_card_str = "ERROR"
                       if 0 <= opp_idx < len(self.players) and hasattr(self.players[opp_idx], 'hand'):
                          opp_hand = self.players[opp_idx].hand
                          if 0 <= target_opp_idx < len(opp_hand): peeked_card_str = self._serialize_card(opp_hand[target_opp_idx]); logger.info(f"P{player} uses 9/T, peeks opp {target_opp_idx}: {peeked_card_str}")
                          else: logger.error(f"Invalid PEEK_OTHER index {target_opp_idx}")
                       else: logger.error(f"Peek Other: Opponent {opp_idx} invalid or missing hand.")
                       discard_for_snap_check = self.get_discard_top()
                       self._clear_pending_action(undo_stack, delta_list) # Pass args
                       delta_list.append(('peek_other', player, opp_idx, target_opp_idx, peeked_card_str))
                 elif isinstance(pending_type, ActionAbilityBlindSwapSelect) and isinstance(action, ActionAbilityBlindSwapSelect):
                       own_h_idx, opp_h_idx = action.own_hand_index, action.opponent_hand_index; opp_idx = self.get_opponent_index(player); hand = self.players[player].hand
                       if 0 <= opp_idx < len(self.players) and hasattr(self.players[opp_idx], 'hand'):
                           opp_hand = self.players[opp_idx].hand
                           if 0 <= own_h_idx < len(hand) and 0 <= opp_h_idx < len(opp_hand):
                                original_own_card = hand[own_h_idx]
                                original_opp_card = opp_hand[opp_h_idx]
                                def change_blind_swap(): hand[own_h_idx], opp_hand[opp_h_idx] = opp_hand[opp_h_idx], hand[own_h_idx]
                                def undo_blind_swap(): hand[own_h_idx], opp_hand[opp_h_idx] = original_own_card, original_opp_card
                                delta_blind_swap = ('swap_blind', player, own_h_idx, opp_idx, opp_h_idx, self._serialize_card(original_own_card), self._serialize_card(original_opp_card))
                                self._add_change(change_blind_swap, undo_blind_swap, delta_blind_swap, undo_stack, delta_list) # Pass args
                                logger.info(f"P{player} uses J/Q, blind swaps own {own_h_idx} with opp {opp_h_idx}.")
                           else: logger.error(f"Invalid BLIND_SWAP indices: own {own_h_idx}, opp {opp_h_idx}")
                       else: logger.error(f"Blind Swap: Opponent {opp_idx} invalid or missing hand.")
                       discard_for_snap_check = self.get_discard_top()
                       self._clear_pending_action(undo_stack, delta_list) # Pass args
                 elif isinstance(pending_type, ActionAbilityKingLookSelect) and isinstance(action, ActionAbilityKingLookSelect):
                       own_h_idx, opp_h_idx = action.own_hand_index, action.opponent_hand_index; opp_idx = self.get_opponent_index(player); hand = self.players[player].hand
                       card1_str, card2_str = "ERROR", "ERROR"
                       card1, card2 = None, None
                       if 0 <= opp_idx < len(self.players) and hasattr(self.players[opp_idx], 'hand'):
                            opp_hand = self.players[opp_idx].hand
                            if 0 <= own_h_idx < len(hand) and 0 <= opp_h_idx < len(opp_hand):
                                card1, card2 = hand[own_h_idx], opp_hand[opp_h_idx]; card1_str, card2_str = self._serialize_card(card1), self._serialize_card(card2)
                                logger.info(f"P{player} uses K, looks at own {own_h_idx} ({card1_str}) and opp {opp_h_idx} ({card2_str}).")
                                original_pending = (self.pending_action, self.pending_action_player, copy.deepcopy(self.pending_action_data))
                                new_pending_data = {"own_idx": own_h_idx, "opp_idx": opp_h_idx, "card1": card1, "card2": card2}
                                def change_king_pending():
                                     self.pending_action = ActionAbilityKingSwapDecision(perform_swap=False)
                                     self.pending_action_player = player
                                     self.pending_action_data = new_pending_data
                                def undo_king_pending():
                                     self.pending_action, self.pending_action_player, self.pending_action_data = original_pending
                                delta_king_pending = ('set_pending_action',
                                         'ActionAbilityKingSwapDecision', player, {'own_idx':own_h_idx, 'opp_idx':opp_h_idx, 'card1':card1_str, 'card2':card2_str},
                                         type(original_pending[0]).__name__ if original_pending[0] else None, original_pending[1], original_pending[2])
                                self._add_change(change_king_pending, undo_king_pending, delta_king_pending, undo_stack, delta_list) # Pass args
                                delta_list.append(('king_look', player, own_h_idx, opp_h_idx, card1_str, card2_str))
                                return delta_list, undo_action # Waiting for swap decision
                            else: logger.error(f"Invalid KING_LOOK indices: own {own_h_idx}, opp {opp_h_idx}. Ability fizzles."); discard_for_snap_check = self.get_discard_top(); self._clear_pending_action(undo_stack, delta_list) # Pass args
                       else: logger.error(f"King Look: Opponent {opp_idx} invalid or missing hand. Ability fizzles."); discard_for_snap_check = self.get_discard_top(); self._clear_pending_action(undo_stack, delta_list) # Pass args
                 elif isinstance(pending_type, ActionAbilityKingSwapDecision) and isinstance(action, ActionAbilityKingSwapDecision):
                       perform_swap = action.perform_swap; look_data = self.pending_action_data
                       own_h_idx, opp_h_idx = look_data.get("own_idx"), look_data.get("opp_idx")
                       card1, card2 = look_data.get("card1"), look_data.get("card2")
                       if own_h_idx is None or opp_h_idx is None or card1 is None or card2 is None: logger.error("Missing data for King Swap decision. Fizzling.")
                       elif perform_swap:
                           opp_idx = self.get_opponent_index(player); hand = self.players[player].hand
                           if 0 <= opp_idx < len(self.players) and hasattr(self.players[opp_idx], 'hand'):
                                opp_hand = self.players[opp_idx].hand
                                if 0 <= own_h_idx < len(hand) and 0 <= opp_h_idx < len(opp_hand):
                                     if hand[own_h_idx] is card1 and opp_hand[opp_h_idx] is card2:
                                          def change_king_swap(): hand[own_h_idx], opp_hand[opp_h_idx] = card2, card1
                                          def undo_king_swap(): hand[own_h_idx], opp_hand[opp_h_idx] = card1, card2
                                          delta_king_swap = ('swap_king', player, own_h_idx, opp_idx, opp_h_idx, self._serialize_card(card1), self._serialize_card(card2))
                                          self._add_change(change_king_swap, undo_king_swap, delta_king_swap, undo_stack, delta_list) # Pass args
                                          logger.info(f"P{player} King ability: Swapped own {own_h_idx} ({card1}) with opp {opp_h_idx} ({card2}).")
                                     else: logger.error(f"Cards changed between King look and swap decision! Swap cancelled.")
                                else: logger.error(f"Indices invalid at King Swap decision ({own_h_idx}, {opp_h_idx}).")
                           else: logger.error(f"King Swap: Opponent {opp_idx} invalid or missing hand.")
                       else: logger.info(f"P{player} King ability: Chose not to swap.")
                       discard_for_snap_check = self.get_discard_top()
                       self._clear_pending_action(undo_stack, delta_list) # Pass args
                 elif isinstance(pending_type, ActionSnapOpponentMove) and isinstance(action, ActionSnapOpponentMove):
                      snapper_idx = player; own_card_idx, target_slot_idx = action.own_card_to_move_hand_index, action.target_empty_slot_index
                      hand = self.players[snapper_idx].hand; opp_idx = self.get_opponent_index(snapper_idx);
                      if 0 <= opp_idx < len(self.players) and hasattr(self.players[opp_idx], 'hand'):
                           opp_hand = self.players[opp_idx].hand
                           if 0 <= own_card_idx < len(hand):
                                moved_card = hand[own_card_idx]
                                if 0 <= target_slot_idx <= len(opp_hand): # Allow insert at end
                                     def change_move():
                                          card = self.players[snapper_idx].hand.pop(own_card_idx)
                                          self.players[opp_idx].hand.insert(target_slot_idx, card)
                                     def undo_move():
                                          card = self.players[opp_idx].hand.pop(target_slot_idx)
                                          self.players[snapper_idx].hand.insert(own_card_idx, card)
                                     delta_move = ('snap_opponent_move', snapper_idx, own_card_idx, opp_idx, target_slot_idx, self._serialize_card(moved_card))
                                     self._add_change(change_move, undo_move, delta_move, undo_stack, delta_list) # Pass args
                                     logger.info(f"P{snapper_idx} completes Snap Opponent: Moves {moved_card} (from own idx {own_card_idx}) to opp idx {target_slot_idx}.")
                                     self._clear_pending_action(undo_stack, delta_list) # Pass args
                                     self._advance_turn(undo_stack, delta_list) # Pass args
                                else: logger.error(f"Invalid target slot index {target_slot_idx} for SnapOpponentMove."); self._clear_pending_action(undo_stack, delta_list); self._advance_turn(undo_stack, delta_list) # Pass args
                           else: logger.error(f"Invalid own card index {own_card_idx} for SnapOpponentMove."); self._clear_pending_action(undo_stack, delta_list); self._advance_turn(undo_stack, delta_list) # Pass args
                      else: logger.error(f"Snap Opponent Move: Opponent {opp_idx} invalid."); self._clear_pending_action(undo_stack, delta_list); self._advance_turn(undo_stack, delta_list) # Pass args
                 else: logger.warning(f"Unhandled pending action ({pending_type}) vs received action ({action})"); self._clear_pending_action(undo_stack, delta_list) # Pass args

                 # After resolving pending action, check for snap phase initiation or advance turn
                 snap_started_here = False
                 if not self.pending_action and not self.snap_phase_active:
                      if discard_for_snap_check and self._initiate_snap_phase(discarded_card=discard_for_snap_check, undo_stack=undo_stack, delta_list=delta_list): # Pass args
                           snap_started_here = True
                 if not snap_started_here and not self.snap_phase_active and not self.pending_action:
                     self._advance_turn(undo_stack, delta_list) # Pass args

            # --- Handle Standard Start-of-Turn Actions ---
            elif isinstance(action, ActionDrawStockpile):
                 player = self.current_player_index
                 drawn_card: Optional[Card] = None
                 reshuffled = False
                 if not self.stockpile:
                      # Pass undo_stack explicitly
                      reshuffle_deltas = self._attempt_reshuffle(undo_stack)
                      if reshuffle_deltas:
                          reshuffled = True
                          delta_list.extend(reshuffle_deltas)
                 if self.stockpile:
                      original_stockpile_len = len(self.stockpile)
                      original_pending = (self.pending_action, self.pending_action_player, copy.deepcopy(self.pending_action_data))
                      drawn_card_for_change = self.stockpile[-1]

                      def change_draw():
                           nonlocal drawn_card
                           card = self.stockpile.pop()
                           drawn_card = card
                           logger.debug(f"P{player} drew {drawn_card} from stockpile.")
                           self.pending_action = ActionDiscard(use_ability=False)
                           self.pending_action_player = player
                           self.pending_action_data = {"drawn_card": drawn_card}
                      def undo_draw():
                           drawn_card_in_pending = self.pending_action_data.get("drawn_card")
                           self.pending_action, self.pending_action_player, self.pending_action_data = original_pending
                           if drawn_card_in_pending: self.stockpile.append(drawn_card_in_pending)
                      delta_draw = ('draw_stockpile', player, self._serialize_card(drawn_card_for_change))
                      delta_pending = ('set_pending_action',
                                        'ActionDiscard', player, {'drawn_card': self._serialize_card(drawn_card_for_change)},
                                        type(original_pending[0]).__name__ if original_pending[0] else None, original_pending[1], original_pending[2])
                      self._add_change(change_draw, undo_draw, delta_draw, undo_stack, delta_list) # Pass args
                      delta_list.append(delta_pending)
                 else:
                      logger.warning(f"P{player} tried DRAW_STOCKPILE, but stockpile/discard empty. Game should end.")
                      original_game_over = self._game_over
                      def change_game_over_draw(): self._game_over = True; self._calculate_final_scores()
                      def undo_game_over_draw(): self._game_over = original_game_over # Note: Does not undo score calculation easily
                      self._add_change(change_game_over_draw, undo_game_over_draw, ('set_attr', '_game_over', True, original_game_over), undo_stack, delta_list) # Pass args

            elif isinstance(action, ActionDrawDiscard):
                  player = self.current_player_index
                  if self.house_rules.allowDrawFromDiscardPile and self.discard_pile:
                       original_discard_len = len(self.discard_pile)
                       original_pending = (self.pending_action, self.pending_action_player, copy.deepcopy(self.pending_action_data))
                       drawn_card_for_change = self.discard_pile[-1]

                       def change_draw_discard():
                            nonlocal drawn_card
                            card = self.discard_pile.pop()
                            drawn_card = card
                            logger.debug(f"P{player} drew {drawn_card} from discard pile.")
                            self.pending_action = ActionDiscard(use_ability=False)
                            self.pending_action_player = player
                            self.pending_action_data = {"drawn_card": drawn_card}
                       def undo_draw_discard():
                            drawn_card_in_pending = self.pending_action_data.get("drawn_card")
                            self.pending_action, self.pending_action_player, self.pending_action_data = original_pending
                            if drawn_card_in_pending: self.discard_pile.append(drawn_card_in_pending)
                       delta_draw = ('draw_discard', player, self._serialize_card(drawn_card_for_change))
                       delta_pending = ('set_pending_action',
                                         'ActionDiscard', player, {'drawn_card': self._serialize_card(drawn_card_for_change)},
                                         type(original_pending[0]).__name__ if original_pending[0] else None, original_pending[1], original_pending[2])
                       self._add_change(change_draw_discard, undo_draw_discard, delta_draw, undo_stack, delta_list) # Pass args
                       delta_list.append(delta_pending)
                  else: logger.error("Invalid Action: DRAW_DISCARD attempted.")

            elif isinstance(action, ActionCallCambia):
                  player = self.current_player_index
                  if self.cambia_caller_id is None:
                      logger.info(f"P{player} calls Cambia!")
                      original_cambia_caller = self.cambia_caller_id
                      original_turns_after = self.turns_after_cambia
                      def change_cambia():
                           self.cambia_caller_id = player
                           self.turns_after_cambia = 0
                      def undo_cambia():
                           self.cambia_caller_id = original_cambia_caller
                           self.turns_after_cambia = original_turns_after
                      delta_caller = ('set_attr', 'cambia_caller_id', player, original_cambia_caller)
                      delta_turns = ('set_attr', 'turns_after_cambia', 0, original_turns_after)
                      self._add_change(change_cambia, undo_cambia, delta_caller, undo_stack, delta_list) # Pass args
                      delta_list.append(delta_turns)
                      self._advance_turn(undo_stack, delta_list) # Pass args
                  else: logger.warning(f"P{player} tried invalid CALL_CAMBIA.")
            else:
                  logger.warning(f"Unhandled action type at start of turn: {type(action)}")

            # Check game end after handling standard turn actions IF NOT already handled
            if not self.snap_phase_active and not self.pending_action:
                self._check_game_end(undo_stack, delta_list) # Pass args

        except Exception as e:
            logger.exception(f"Critical error during apply_action logic for action {action}. State: {self}")
            try:
                 undo_action()
                 logger.info("Successfully executed master undo after apply_action exception.")
            except Exception as undo_e:
                 logger.error(f"Exception during master undo after apply_action error: {undo_e}")
            return [], lambda: None

        # --- Return the collected deltas and the master undo function ---
        return delta_list, undo_action


    def _clear_pending_action(self, undo_stack: Deque, delta_list: StateDelta):
        """Resets the pending action state and adds undo operation and delta."""
        if self.pending_action is None and self.pending_action_player is None and not self.pending_action_data:
            return # Nothing to clear

        original_pending_action = self.pending_action
        original_pending_player = self.pending_action_player
        original_pending_data = copy.deepcopy(self.pending_action_data)

        def change():
            self.pending_action = None
            self.pending_action_player = None
            self.pending_action_data = {}
        def undo():
            self.pending_action = original_pending_action
            self.pending_action_player = original_pending_player
            self.pending_action_data = original_pending_data

        delta = ('clear_pending_action',
                 type(original_pending_action).__name__ if original_pending_action else None,
                 original_pending_player,
                 {k: self._serialize_card(v) if isinstance(v, Card) else v for k, v in original_pending_data.items()}
                 )
        # Explicitly pass undo_stack and delta_list
        self._add_change(change, undo, delta, undo_stack, delta_list)


    def _trigger_discard_ability(self, player_index: int, discarded_card: Card, undo_stack: Deque, delta_list: StateDelta):
        """Sets up the pending state for executing a special ability, adding undo and delta."""
        rank = discarded_card.rank
        logger.debug(f"Player {player_index} triggering ability of discarded {discarded_card}")
        ability_triggered = False
        next_pending_action: Optional[GameAction] = None

        if rank in [SEVEN, EIGHT]: next_pending_action = ActionAbilityPeekOwnSelect(target_hand_index=-1); ability_triggered = True
        elif rank in [NINE, TEN]: next_pending_action = ActionAbilityPeekOtherSelect(target_opponent_hand_index=-1); ability_triggered = True
        elif rank in [JACK, QUEEN]: next_pending_action = ActionAbilityBlindSwapSelect(own_hand_index=-1, opponent_hand_index=-1); ability_triggered = True
        elif rank == KING: next_pending_action = ActionAbilityKingLookSelect(own_hand_index=-1, opponent_hand_index=-1); ability_triggered = True

        if ability_triggered and next_pending_action is not None:
             original_pending_action = self.pending_action
             original_pending_player = self.pending_action_player
             original_pending_data = copy.deepcopy(self.pending_action_data)
             new_pending_data = {"ability_card": discarded_card}

             def change_ability_pending():
                  self.pending_action = next_pending_action
                  self.pending_action_player = player_index
                  self.pending_action_data = new_pending_data
             def undo_ability_pending():
                  self.pending_action = original_pending_action
                  self.pending_action_player = original_pending_player
                  self.pending_action_data = original_pending_data

             delta_ability = ('set_pending_action',
                      type(next_pending_action).__name__, player_index, {'ability_card': self._serialize_card(discarded_card)},
                      type(original_pending_action).__name__ if original_pending_action else None,
                      original_pending_player,
                      {k: self._serialize_card(v) if isinstance(v, Card) else v for k, v in original_pending_data.items()}
                      )
             # Explicitly pass undo_stack and delta_list
             self._add_change(change_ability_pending, undo_ability_pending, delta_ability, undo_stack, delta_list)
        else: logger.debug(f"Card {discarded_card} has no relevant discard ability.")


    def _initiate_snap_phase(self, discarded_card: Card, undo_stack: Deque, delta_list: StateDelta) -> bool:
        """ Checks potential snappers and starts the snap phase, adding undos/deltas. Returns True if started. """
        original_snap_phase = self.snap_phase_active
        original_snap_card = self.snap_discarded_card
        original_snap_potentials = list(self.snap_potential_snappers)
        original_snap_idx = self.snap_current_snapper_idx

        potential_indices = []
        target_rank = discarded_card.rank

        for p_idx in range(self.num_players):
            if not (0 <= p_idx < len(self.players) and hasattr(self.players[p_idx], 'hand')):
                 logger.warning(f"Initiate Snap: Player {p_idx} invalid or missing hand. Skipping.")
                 continue
            if p_idx == self.cambia_caller_id: continue

            hand = self.players[p_idx].hand
            if not all(isinstance(card, Card) for card in hand):
                 logger.error(f"Initiate Snap: Player {p_idx}'s hand contains non-Card objects: {hand}. Skipping snap check.")
                 continue
            can_snap_own = any(card.rank == target_rank for card in hand)
            can_snap_opponent = False
            if self.house_rules.allowOpponentSnapping:
                 opp_idx = self.get_opponent_index(p_idx)
                 if opp_idx != self.cambia_caller_id:
                      if 0 <= opp_idx < len(self.players) and hasattr(self.players[opp_idx], 'hand'):
                           opp_hand = self.players[opp_idx].hand
                           if not all(isinstance(card, Card) for card in opp_hand):
                                logger.error(f"Initiate Snap: Opponent {opp_idx}'s hand contains non-Card objects. Skipping snap-opp check.")
                           else:
                                can_snap_opponent = any(card.rank == target_rank for card in opp_hand)
                      else: logger.warning(f"Initiate Snap: Opponent {opp_idx} invalid/missing hand for P{p_idx} checking SnapOpponent.")

            if can_snap_own or can_snap_opponent:
                 potential_indices.append(p_idx)

        started_snap = False
        if potential_indices:
             logger.debug(f"Discard of {discarded_card} triggers potential snap phase.")
             discarder_player = self.pending_action_player if self.pending_action_player is not None else (self.current_player_index - 1 + self.num_players) % self.num_players

             ordered_snappers = []
             for i in range(1, self.num_players + 1):
                  check_p_idx = (discarder_player + i) % self.num_players
                  if check_p_idx in potential_indices:
                       ordered_snappers.append(check_p_idx)

             if ordered_snappers:
                  started_snap = True
                  def change_snap_start():
                       self.snap_phase_active = True
                       self.snap_discarded_card = discarded_card
                       self.snap_potential_snappers = ordered_snappers
                       self.snap_current_snapper_idx = 0
                  def undo_snap_start():
                       self.snap_phase_active = original_snap_phase
                       self.snap_discarded_card = original_snap_card
                       self.snap_potential_snappers = original_snap_potentials
                       self.snap_current_snapper_idx = original_snap_idx
                  delta_snap_start = ('start_snap_phase', self._serialize_card(discarded_card), ordered_snappers)
                  # Explicitly pass undo_stack and delta_list
                  self._add_change(change_snap_start, undo_snap_start, delta_snap_start, undo_stack, delta_list)
                  logger.debug(f"Snap phase started. Potential snappers (ordered): {ordered_snappers}. P{self.get_acting_player()} acts first.")
             else:
                  logger.debug("Discarder was only potential snapper. No snap phase.")
                  started_snap = False
        else:
             logger.debug(f"No potential snappers found for rank {target_rank}.")
             started_snap = False

        return started_snap


    def _end_snap_phase(self, undo_stack: Deque, delta_list: StateDelta):
         """Cleans up snap phase state and advances the main game turn, adding undos/deltas."""
         if not self.snap_phase_active: return

         logger.debug("Ending snap phase.")
         original_snap_phase = self.snap_phase_active
         original_snap_card = self.snap_discarded_card
         original_snap_potentials = list(self.snap_potential_snappers)
         original_snap_idx = self.snap_current_snapper_idx

         def change_snap_end():
             self.snap_phase_active = False
             self.snap_discarded_card = None
             self.snap_potential_snappers = []
             self.snap_current_snapper_idx = 0
         def undo_snap_end():
             self.snap_phase_active = original_snap_phase
             self.snap_discarded_card = original_snap_card
             self.snap_potential_snappers = original_snap_potentials
             self.snap_current_snapper_idx = original_snap_idx

         delta_snap_end = ('end_snap_phase',)
         # Explicitly pass undo_stack and delta_list
         self._add_change(change_snap_end, undo_snap_end, delta_snap_end, undo_stack, delta_list)
         self._advance_turn(undo_stack, delta_list) # Pass args


    def _attempt_reshuffle(self, undo_stack_outer: Deque) -> Optional[StateDelta]:
        """Reshuffles discard->stockpile if possible, adds undos/deltas. Returns deltas or None."""
        # This function might be called outside apply_action (e.g., during penalty).
        # It needs its OWN delta list and needs to add its undo op to the provided outer stack.
        if not self.stockpile and len(self.discard_pile) > 1:
             logger.info("Stockpile empty. Reshuffling discard pile.")
             original_stockpile = list(self.stockpile)
             original_discard = list(self.discard_pile)
             top_card = self.discard_pile[-1]
             cards_to_shuffle = self.discard_pile[:-1]
             reshuffle_deltas: StateDelta = [] # Local list to return
             shuffled_order_for_delta = list(cards_to_shuffle)
             shuffled_card_strs = [self._serialize_card(c) for c in shuffled_order_for_delta]
             delta_reshuffle = ('reshuffle', shuffled_card_strs, self._serialize_card(top_card))

             def change_reshuffle():
                  self.discard_pile = [top_card]
                  shuffled_cards = list(cards_to_shuffle)
                  random.shuffle(shuffled_cards)
                  self.stockpile = shuffled_cards
                  logger.info(f"Reshuffled {len(self.stockpile)} cards into stockpile.")
             def undo_reshuffle():
                  self.stockpile = original_stockpile
                  self.discard_pile = original_discard
                  logger.debug("Undo reshuffle.")

             # Manually apply change, add undo to outer stack, add delta to local list
             change_reshuffle()
             reshuffle_deltas.append(delta_reshuffle)
             undo_stack_outer.appendleft(undo_reshuffle)
             return reshuffle_deltas
        elif not self.stockpile:
             logger.info("Stockpile empty, cannot reshuffle discard pile (size <= 1).")
             return None
        return None


    def _apply_penalty(self, player_index: int, num_cards: int, undo_stack_main: Deque) -> StateDelta:
        """Adds penalty cards, adds undos and returns deltas."""
        # This is a complex multi-step operation. It generates its own list of deltas.
        # It needs ONE master undo operation added to the main undo stack.
        logger.warning(f"Applying penalty: Player {player_index} draws {num_cards} cards.")
        penalty_deltas: StateDelta = []
        if not (0 <= player_index < len(self.players) and hasattr(self.players[player_index], 'hand')):
             logger.error(f"Cannot apply penalty: Player {player_index} invalid or missing hand.")
             return penalty_deltas

        original_hand_state = list(self.players[player_index].hand)
        original_stockpile_state = list(self.stockpile) # Needed if reshuffle happens
        original_discard_state = list(self.discard_pile) # Needed if reshuffle happens
        drawn_cards_in_penalty: List[Card] = [] # Track cards drawn

        reshuffled_during_penalty = False

        for i in range(num_cards):
             if not self.stockpile:
                  # Attempt reshuffle. It adds its own undo op to OUR undo_stack_main if successful.
                  reshuffle_outcome_deltas = self._attempt_reshuffle(undo_stack_main)
                  if reshuffle_outcome_deltas:
                       reshuffled_during_penalty = True
                       penalty_deltas.extend(reshuffle_outcome_deltas)
                       logger.debug(f"Reshuffled during penalty draw {i+1}/{num_cards}")
                  else:
                       logger.warning(f"Stockpile/discard empty during penalty draw {i+1}/{num_cards}! Cannot draw more.")
                       break # Stop drawing

             if self.stockpile:
                  drawn_card = self.stockpile.pop()
                  drawn_cards_in_penalty.append(drawn_card) # Track for master undo
                  self.players[player_index].hand.append(drawn_card)
                  delta = ('penalty_draw', player_index, self._serialize_card(drawn_card))
                  penalty_deltas.append(delta)
                  logger.debug(f"Player {player_index} penalty draw {i+1}: {drawn_card}")
             else:
                 logger.error("Stockpile empty immediately after attempting reshuffle in penalty draw.")
                 break

        # Create the single master undo function for the entire penalty operation
        def undo_penalty_sequence():
            # Restore the state *before* the penalty loop started
            self.players[player_index].hand = original_hand_state
            # Only restore stock/discard if reshuffle happened, otherwise they weren't touched
            # Note: This assumes _attempt_reshuffle's undo correctly restores pre-reshuffle stock/discard
            if not reshuffled_during_penalty:
                 self.stockpile = original_stockpile_state
                 self.discard_pile = original_discard_state
            # If reshuffle DID happen, its specific undo is already on the main stack
            # and will be called AFTER this master undo. This seems complex.

            # Alternative Undo: Revert specific changes
            # 1. Remove drawn cards from hand (in reverse order added)
            # for _ in range(len(drawn_cards_in_penalty)):
            #      self.players[player_index].hand.pop()
            # 2. Add cards back to stockpile (reverse order drawn)
            # for card in reversed(drawn_cards_in_penalty):
            #      self.stockpile.append(card)
            # 3. Rely on reshuffle's undo being on the main stack separately.

            # Let's stick with the simpler state restoration for now, assuming reshuffle undo works independently.
            logger.debug(f"Undo penalty applied for P{player_index}. State restored to pre-penalty.")


        # Add the master undo function to the main stack passed from apply_action
        undo_stack_main.appendleft(undo_penalty_sequence)
        return penalty_deltas


    def _advance_turn(self, undo_stack: Deque, delta_list: StateDelta):
        """Moves to next player, handles Cambia turns, adds undos/deltas."""
        if self._game_over: return

        original_turn = self._turn_number
        original_player = self.current_player_index
        original_cambia_turns = self.turns_after_cambia
        logger.debug(f"Advancing turn from T#{original_turn} P{original_player}")

        next_player = (self.current_player_index + 1) % self.num_players
        next_turn = self._turn_number + 1
        next_cambia_turns = self.turns_after_cambia
        if self.cambia_caller_id is not None:
            next_cambia_turns += 1

        def change_advance():
            self.current_player_index = next_player
            self._turn_number = next_turn
            self.turns_after_cambia = next_cambia_turns
        def undo_advance():
            self._turn_number = original_turn
            self.current_player_index = original_player
            self.turns_after_cambia = original_cambia_turns
            logger.debug(f"Undo advance turn. Back to T#{original_turn}, P{original_player}")

        delta_player = ('set_attr', 'current_player_index', next_player, original_player)
        delta_turn = ('set_attr', '_turn_number', next_turn, original_turn)
        delta_cambia = ('set_attr', 'turns_after_cambia', next_cambia_turns, original_cambia_turns)

        # Explicitly pass undo_stack and delta_list
        self._add_change(change_advance, undo_advance, delta_player, undo_stack, delta_list)
        delta_list.append(delta_turn)
        if self.cambia_caller_id is not None:
            delta_list.append(delta_cambia)

        self._check_game_end(undo_stack, delta_list) # Pass args


    def _check_game_end(self, undo_stack: Deque, delta_list: StateDelta):
        """Checks and sets game end conditions, adding undos/deltas."""
        if self._game_over: return

        end_condition_met = False
        reason = ""

        max_turns = self.house_rules.max_game_turns
        if max_turns > 0 and self._turn_number >= max_turns:
             end_condition_met = True
             reason = f"Max game turns ({max_turns}) reached"

        if not end_condition_met and self.cambia_caller_id is not None and self.turns_after_cambia >= self.num_players :
            end_condition_met = True
            reason = "Cambia final turns completed"

        if not end_condition_met and not self.pending_action and not self.snap_phase_active:
            player = self.current_player_index
            if 0 <= player < len(self.players) and hasattr(self.players[player], 'hand'):
                 legal_actions = self.get_legal_actions()
                 is_draw_required = not any(a for a in legal_actions if not isinstance(a, (ActionDrawStockpile, ActionDrawDiscard, ActionCallCambia)))
                 can_draw_or_reshuffle = bool(self.stockpile) or (len(self.discard_pile) > 1)
                 if is_draw_required and not can_draw_or_reshuffle:
                      end_condition_met = True
                      reason = f"Player {player} requires draw, but cannot"
            else:
                end_condition_met = True
                reason = f"Game end check (Draw required): Player {player} invalid/missing hand."

        if not end_condition_met and not self.pending_action and not self.snap_phase_active:
             player = self.current_player_index
             if 0 <= player < len(self.players) and hasattr(self.players[player], 'hand'):
                 if not self.get_legal_actions():
                       end_condition_met = True
                       reason = "No legal actions available"
             else:
                end_condition_met = True
                reason = f"Game end check (no actions): Player {player} invalid/missing hand."

        if end_condition_met and not self._game_over:
             logger.info(f"Game ends: {reason}.")
             original_game_over = self._game_over
             original_winner = self._winner
             original_utilities = list(self._utilities)
             temp_winner, temp_utilities = self._calculate_final_scores(set_attributes=False)
             delta_game_end = ('game_end', reason, temp_winner, temp_utilities)

             def change_game_end():
                  self._game_over = True
                  self._calculate_final_scores(set_attributes=True)
             def undo_game_end():
                  self._game_over = original_game_over
                  self._winner = original_winner
                  self._utilities = original_utilities
                  logger.debug("Undo game end.")

             # Explicitly pass undo_stack and delta_list
             self._add_change(change_game_end, undo_game_end, delta_game_end, undo_stack, delta_list)


    def _calculate_final_scores(self, set_attributes=True) -> Tuple[Optional[int], List[float]]:
        """Calculates final scores and winner. Optionally sets instance attributes."""
        if set_attributes and self._winner is not None:
             # Scores already calculated and set previously
             return self._winner, self._utilities

        scores = []
        final_hands_str = []
        for i in range(self.num_players):
            if 0 <= i < len(self.players) and hasattr(self.players[i], 'hand'):
                 current_hand = self.players[i].hand
                 hand_str = [self._serialize_card(c) for c in current_hand]
                 final_hands_str.append(hand_str)
                 if not all(isinstance(card, Card) for card in current_hand):
                      logger.error(f"Calculate score: Player {i}'s hand contains non-Card objects: {current_hand}. Assigning high score.")
                      scores.append(999)
                 else:
                      hand_value = sum(card.value for card in current_hand)
                      scores.append(hand_value)
            else:
                 logger.error(f"Calculate score: Player {i} invalid or missing hand. Assigning high score.")
                 scores.append(999)
                 final_hands_str.append(["ERROR"])

        winner_calculated = None
        utilities_calculated = [0.0] * self.num_players

        if scores:
             min_score = min(scores)
             winners = [i for i, score in enumerate(scores) if score == min_score]

             if len(winners) == 1:
                 winner_calculated = winners[0]
             elif self.cambia_caller_id is not None and self.cambia_caller_id in winners:
                 winner_calculated = self.cambia_caller_id

             if winner_calculated is not None:
                  utilities_calculated = [-1.0] * self.num_players
                  utilities_calculated[winner_calculated] = 1.0
                  # Log only when setting final attributes
                  if set_attributes: logger.info(f"Game Score Calc: Player {winner_calculated} wins with score {min_score}. Final Hands: {final_hands_str}")
             else:
                  if set_attributes: logger.info(f"Game Score Calc: Tie between players {winners} with score {min_score}. Utilities: {utilities_calculated}. Final Hands: {final_hands_str}")
        else:
             logger.error("Cannot calculate final scores: No player scores available.")

        if set_attributes:
            self._winner = winner_calculated
            self._utilities = utilities_calculated
            logger.debug(f"Final Utilities set: {self._utilities}")

        return winner_calculated, utilities_calculated


    def is_terminal(self) -> bool: return self._game_over

    def get_utility(self, player_id: int) -> float:
        """Returns the final utility for the specified player."""
        if not self.is_terminal(): logger.warning("get_utility called on non-terminal state!"); return 0.0
        if self._game_over and self._winner is None and np.all(np.array(self._utilities) == 0):
             logger.debug("get_utility called on terminal state, but winner/utilities not set. Calculating scores now.")
             self._calculate_final_scores(set_attributes=True)
        if 0 <= player_id < self.num_players: return self._utilities[player_id]
        logger.error(f"Invalid player index {player_id} requested for utility.")
        return 0.0


    def get_player_turn(self) -> int: return self.current_player_index

    def get_acting_player(self) -> int:
         """Returns the index of the player who needs to act *now*."""
         if self.snap_phase_active:
             if self.snap_current_snapper_idx < len(self.snap_potential_snappers):
                  potential_snapper = self.snap_potential_snappers[self.snap_current_snapper_idx]
                  if 0 <= potential_snapper < len(self.players) and hasattr(self.players[potential_snapper], 'hand'): return potential_snapper
                  else: logger.error(f"Snap phase acting player index {potential_snapper} invalid or missing hand."); return -1
             else: logger.error("Snap phase active but index out of bounds."); return -1
         elif self.pending_action and self.pending_action_player is not None:
             pending_player = self.pending_action_player
             if 0 <= pending_player < len(self.players) and hasattr(self.players[pending_player], 'hand'): return pending_player
             else: logger.error(f"Pending action player index {pending_player} invalid or missing hand."); return -1
         elif not self._game_over:
             current_player = self.current_player_index
             if 0 <= current_player < len(self.players) and hasattr(self.players[current_player], 'hand'): return current_player
             else: logger.error(f"Current player index {current_player} invalid or missing hand in active game."); return -1
         else: return -1


    def __str__(self) -> str:
        state_desc = ""
        actor = self.get_acting_player()
        actor_str = f"P{actor}" if actor != -1 else "N/A"
        if self.snap_phase_active: state_desc = f"SnapPhase(Actor: {actor_str}, Target: {self.snap_discarded_card.rank if self.snap_discarded_card else 'N/A'})"
        elif self.pending_action: state_desc = f"Pending(Actor: {actor_str}, Action: {type(self.pending_action).__name__})"
        elif self._game_over: state_desc = f"GameOver(W:{self._winner})"
        else: state_desc = f"Turn: {actor_str}"

        discard_top_str = str(self.get_discard_top()) if self.discard_pile else "[]"
        hand_lens = []
        for i, p in enumerate(self.players):
             if hasattr(p, 'hand') and isinstance(p.hand, list): hand_lens.append(len(p.hand))
             else: hand_lens.append("ERR")

        return (f"GameState(T#{self._turn_number}, {state_desc}, "
                f"Stock:{len(self.stockpile)}, Disc:{discard_top_str}, "
                f"Hands:{hand_lens}, "
                f"Cambia:{self.cambia_caller_id})")