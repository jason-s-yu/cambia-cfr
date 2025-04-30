# src/game_engine.py
import random
from typing import List, Tuple, Optional, Set, Any, Dict
from dataclasses import dataclass, field
import logging
import copy
import numpy as np
import os # Added for path validation

from .card import Card, create_standard_deck
from .constants import (
    KING, QUEEN, JACK, NINE, TEN, SEVEN, EIGHT,
    GameAction, ActionPassSnap, ActionSnapOwn, ActionSnapOpponent, ActionSnapOpponentMove,
    ActionDiscard, ActionReplace, ActionCallCambia, ActionDrawStockpile, ActionDrawDiscard,
    ActionAbilityPeekOwnSelect, ActionAbilityPeekOtherSelect, ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect, ActionAbilityKingSwapDecision,
    NUM_PLAYERS
)
from .config import CambiaRulesConfig

logger = logging.getLogger(__name__)

# Define PlayerState if needed (or keep hands directly in GameState for 1v1)
@dataclass
class PlayerState:
     hand: List[Card] = field(default_factory=list)
     initial_peek_indices: Tuple[int, ...] = (0, 1) # Default peek indices

     # Custom deepcopy to ensure hand is copied correctly
     # Rely on the top-level deepcopy's memoization
     def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        # Ensure hand attribute exists before trying to copy
        if hasattr(self, 'hand'):
             # Let deepcopy handle cards recursively, using the provided memo
             setattr(result, 'hand', copy.deepcopy(self.hand, memo))
        else:
             logger.warning("PlayerState.__deepcopy__ called on object without 'hand' attribute.")
             setattr(result, 'hand', [])

        # Copy other immutable attributes like tuple directly
        if hasattr(self, 'initial_peek_indices'):
             setattr(result, 'initial_peek_indices', self.initial_peek_indices)
        else:
             setattr(result, 'initial_peek_indices', (0, 1))

        return result

@dataclass
class CambiaGameState:
    """Represents the true, objective state of a 1v1 Cambia game."""
    players: List[PlayerState] = field(default_factory=list)
    stockpile: List[Card] = field(default_factory=list)
    discard_pile: List[Card] = field(default_factory=list)
    current_player_index: int = 0
    num_players: int = NUM_PLAYERS # Use constant
    cambia_caller_id: Optional[int] = None
    turns_after_cambia: int = 0
    house_rules: CambiaRulesConfig = field(default_factory=CambiaRulesConfig)
    _game_over: bool = False
    _winner: Optional[int] = None
    _utilities: List[float] = field(default_factory=lambda: [0.0] * NUM_PLAYERS)
    _turn_number: int = 0 # Track game progression

    # Intermediate state for multi-step actions (e.g., King ability, Post-Draw Choice, Snap Move)
    pending_action: Optional[GameAction] = None # Stores the type of action needed next
    pending_action_player: Optional[int] = None # Player who needs to act
    pending_action_data: Dict[str, Any] = field(default_factory=dict) # Extra info needed

    # Snap Phase State
    snap_phase_active: bool = False
    snap_discarded_card: Optional[Card] = None
    snap_potential_snappers: List[int] = field(default_factory=list) # Players who *could* snap (ordered)
    snap_current_snapper_idx: int = 0 # Index into snap_potential_snappers
    snap_results_log: List[Dict[str, Any]] = field(default_factory=list) # Log of snap outcomes this phase

    def __post_init__(self):
        if not self.players: # Initialize deck and deal if not already provided
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
                    # Ensure player object exists and has a hand list before appending
                    if i < len(self.players) and hasattr(self.players[i], 'hand') and isinstance(self.players[i].hand, list):
                         self.players[i].hand.append(self.stockpile.pop())
                    else:
                         logger.error(f"Error during setup: Player object {i} invalid or missing hand list.")
                         return # Stop setup on error
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


    def get_player_hand(self, player_index: int) -> List[Card]:
        if 0 <= player_index < len(self.players) and hasattr(self.players[player_index], 'hand'):
            return self.players[player_index].hand
        logger.error(f"Invalid player index {player_index} or player object missing hand.")
        return [] # Return empty list on error

    def get_opponent_index(self, player_index: int) -> int:
        return 1 - player_index # Specific to 1v1

    def get_player_card_count(self, player_index: int) -> int:
         if 0 <= player_index < len(self.players) and hasattr(self.players[player_index], 'hand'):
            return len(self.players[player_index].hand)
         return 0 # Return 0 if invalid index or missing hand

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
            # Check if the acting player object is valid
            if not (0 <= acting_player < len(self.players) and hasattr(self.players[acting_player], 'hand')):
                 logger.error(f"Snap phase: Acting player {acting_player} object invalid or missing hand.")
                 return legal_actions # Cannot determine actions

            snapper_hand = self.players[acting_player].hand
            # Ensure snap_discarded_card is not None before accessing rank
            if self.snap_discarded_card is None:
                 logger.error("Snap phase active but snap_discarded_card is None.")
                 return legal_actions # Cannot determine snap actions
            target_rank = self.snap_discarded_card.rank

            # Always allow passing the snap opportunity
            legal_actions.add(ActionPassSnap())

            # Check for Snap Own
            for i, card in enumerate(snapper_hand):
                if card.rank == target_rank:
                    legal_actions.add(ActionSnapOwn(own_card_hand_index=i))

            # Check for Snap Opponent (if allowed by rules)
            if self.house_rules.allowOpponentSnapping:
                 opponent_idx = self.get_opponent_index(acting_player)
                 # Check if opponent object is valid
                 if 0 <= opponent_idx < len(self.players) and hasattr(self.players[opponent_idx], 'hand'):
                      opponent_hand = self.players[opponent_idx].hand
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
                 return legal_actions # Return empty set for wrong player

            # Check if acting player object is valid
            if not (0 <= acting_player < len(self.players) and hasattr(self.players[acting_player], 'hand')):
                 logger.error(f"Pending action: Acting player {acting_player} object invalid or missing hand.")
                 return legal_actions

            action_type = self.pending_action
            player = self.pending_action_player

            if isinstance(action_type, ActionDiscard): # Waiting for Post-Draw Choice
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
                 for i in range(self.get_player_card_count(snapper_idx)):
                      legal_actions.add(ActionSnapOpponentMove(own_card_to_move_hand_index=i, target_empty_slot_index=target_slot))
            else:
                 logger.error(f"Unknown pending action type for legal actions: {action_type}")

            return legal_actions

        # --- Standard Start-of-Turn Actions ---
        player = self.current_player_index
        # Check if player object is valid before proceeding
        if not (0 <= player < len(self.players) and hasattr(self.players[player], 'hand')):
            logger.error(f"Start of turn: Player {player} object invalid or missing hand.")
            return legal_actions

        # 1. Draw from Stockpile (Handle reshuffle if needed)
        can_draw = True
        if not self.stockpile:
            if not (self.discard_pile and len(self.discard_pile) > 1): # Check if reshuffle possible
                 can_draw = False
        if can_draw:
            legal_actions.add(ActionDrawStockpile())

        # 2. Draw from Discard Pile (if house rule allows)
        if self.house_rules.allowDrawFromDiscardPile and self.discard_pile:
             legal_actions.add(ActionDrawDiscard())

        # 3. Call Cambia
        cambia_allowed_round = self.house_rules.cambia_allowed_round
        if self.cambia_caller_id is None and (self._turn_number // self.num_players >= cambia_allowed_round):
             legal_actions.add(ActionCallCambia())

        # Game End Check (if no actions possible)
        if not legal_actions and not self._game_over and not can_draw and not (self.house_rules.allowDrawFromDiscardPile and self.discard_pile):
            # Only possible state is game end due to deck exhaustion and no Cambia Call allowed/possible
            pass # apply_action will handle this by ending game

        elif not legal_actions and not self._game_over:
             logger.error(f"No legal actions found for player {player} in state: {self}")

        return legal_actions

    def _card_has_discard_ability(self, card: Card) -> bool:
        """Checks if a card has an ability when discarded from draw."""
        return card.rank in [SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING]

    def apply_action(self, action: GameAction) -> 'CambiaGameState':
        """
        Applies the given action and returns the *next* game state.
        Handles card movement, ability triggers, turn changes, penalties, game end checks.
        """
        if self._game_over:
            logger.warning("Attempted action on a finished game.")
            return self

        new_state = self.clone() # Work on a copy
        acting_player = new_state.get_acting_player()

        # Check validity of acting_player and associated PlayerState
        if not (0 <= acting_player < len(new_state.players) and hasattr(new_state.players[acting_player], 'hand')):
             logger.error(f"Apply Action: Invalid acting player {acting_player} or missing hand. State: {new_state}")
             new_state._game_over = True # Force game over if state seems corrupted
             new_state._calculate_final_scores() # Calculate scores for terminal state
             return new_state

        current_player = new_state.current_player_index # Player whose turn it *was* or *is*

        # --- Snap Phase Action Handling ---
        if new_state.snap_phase_active:
            # Verify action came from the correct snapper
            if acting_player != new_state.snap_potential_snappers[new_state.snap_current_snapper_idx]:
                logger.error(f"Action {action} received, but expected snap action from P{self.snap_potential_snappers[self.snap_current_snapper_idx]}. Ignoring.")
                return new_state # Return unchanged state if wrong player acts

            logger.debug(f"Snap Phase (Turn {new_state._turn_number}): Player {acting_player} choosing action: {action}")
            if new_state.snap_discarded_card is None:
                logger.error("Apply action in snap phase, but snap_discarded_card is None. Cannot proceed.")
                new_state._end_snap_phase() # Attempt to recover
                return new_state
            target_rank = new_state.snap_discarded_card.rank

            snap_success = False
            snap_penalty = False
            removed_index = None
            snapped_card_object = None # Store the actual card
            attempted_card = None # Store attempted card for penalty log
            action_type_str = type(action).__name__


            if isinstance(action, ActionPassSnap):
                 logger.debug(f"Player {acting_player} passes snap.")
                 # Record pass in results log
                 new_state.snap_results_log.append({
                      "snapper": acting_player, "action_type": action_type_str, "target_rank": target_rank,
                      "success": False, "penalty": False, "details": "Passed",
                      "snapped_card": None # Explicitly None
                 })

            elif isinstance(action, ActionSnapOwn):
                 snap_idx = action.own_card_hand_index
                 hand = new_state.players[acting_player].hand # Already verified player object is valid
                 attempted_card = hand[snap_idx] if 0 <= snap_idx < len(hand) else None

                 if attempted_card and attempted_card.rank == target_rank:
                      snapped_card = hand.pop(snap_idx)
                      snapped_card_object = snapped_card # Store for logging
                      new_state.discard_pile.append(snapped_card)
                      snap_success = True
                      removed_index = snap_idx # We know which index was removed
                      logger.info(f"Player {acting_player} snaps own {snapped_card} (matching {target_rank}) from index {snap_idx}. Hand size: {len(hand)}")
                 else:
                      snap_penalty = True
                      logger.warning(f"Player {acting_player} attempted invalid Snap Own: {action} (Card: {attempted_card}). Applying penalty.")
                      new_state._apply_penalty(acting_player, new_state.house_rules.penaltyDrawCount)
                 # Record result
                 new_state.snap_results_log.append({
                     "snapper": acting_player, "action_type": action_type_str, "target_rank": target_rank,
                     "success": snap_success, "penalty": snap_penalty,
                     "removed_own_index": removed_index if snap_success else None,
                     "snapped_card": snapped_card_object if snap_success else None, # Add card if success
                     "attempted_card": attempted_card if snap_penalty else None # Log attempted card on penalty
                 })

            elif isinstance(action, ActionSnapOpponent):
                 if not new_state.house_rules.allowOpponentSnapping:
                      logger.error("Invalid Action: SnapOpponent attempted but rule disallowed.")
                      snap_penalty = True
                      new_state._apply_penalty(acting_player, new_state.house_rules.penaltyDrawCount)
                 else:
                      opp_idx = new_state.get_opponent_index(acting_player)
                      # Verify opponent object before accessing hand
                      if not (0 <= opp_idx < len(new_state.players) and hasattr(new_state.players[opp_idx], 'hand')):
                           logger.error(f"SnapOpponent: Opponent object {opp_idx} invalid or missing hand.")
                           snap_penalty = True # Apply penalty as action cannot be validated
                           new_state._apply_penalty(acting_player, new_state.house_rules.penaltyDrawCount)
                           attempted_card = None # Cannot determine attempted card
                      else:
                           opp_hand = new_state.players[opp_idx].hand
                           target_opp_hand_idx = action.opponent_target_hand_index
                           attempted_card = opp_hand[target_opp_hand_idx] if 0 <= target_opp_hand_idx < len(opp_hand) else None

                           if attempted_card and attempted_card.rank == target_rank:
                                removed_card = opp_hand.pop(target_opp_hand_idx)
                                snapped_card_object = removed_card # Store for logging
                                snap_success = True
                                removed_index = target_opp_hand_idx # Index removed from opponent
                                logger.info(f"Player {acting_player} snaps opponent's {removed_card} at index {target_opp_hand_idx}. Requires move.")
                                # --- Set up for move action ---
                                new_state.pending_action = ActionSnapOpponentMove(own_card_to_move_hand_index=-1, target_empty_slot_index=-1) # Placeholder type
                                new_state.pending_action_player = acting_player
                                new_state.pending_action_data = {"target_empty_slot_index": target_opp_hand_idx}
                                new_state.snap_phase_active = False # Exit snap phase to resolve move
                                # Record partial result (move pending) - include snapped card
                                new_state.snap_results_log.append({
                                   "snapper": acting_player, "action_type": action_type_str, "target_rank": target_rank,
                                   "success": snap_success, "penalty": False,
                                   "removed_opponent_index": removed_index,
                                   "snapped_card": snapped_card_object # Add card here
                                })
                                return new_state # Intermediate state waiting for move
                           else:
                                snap_penalty = True
                                logger.warning(f"Player {acting_player} attempted invalid Snap Opponent: {action} (Card: {attempted_card}). Applying penalty.")
                                new_state._apply_penalty(acting_player, new_state.house_rules.penaltyDrawCount)
                 # Record failed/penalty result
                 if not snap_success:
                      new_state.snap_results_log.append({
                          "snapper": acting_player, "action_type": action_type_str, "target_rank": target_rank,
                          "success": False, "penalty": snap_penalty,
                          "snapped_card": None, # No card snapped
                          "attempted_card": attempted_card if snap_penalty else None # Log attempted card on penalty
                      })

            else:
                logger.error(f"Invalid action type {type(action)} received during snap phase.")
                # Record error in log?
                new_state.snap_results_log.append({
                     "snapper": acting_player, "action_type": "InvalidAction", "target_rank": target_rank,
                     "success": False, "penalty": False, "details": f"Received {type(action).__name__}",
                     "snapped_card": None
                })

            # --- Advance Snap Phase ---
            new_state.snap_current_snapper_idx += 1
            if new_state.snap_current_snapper_idx >= len(new_state.snap_potential_snappers):
                 new_state._end_snap_phase() # This advances the main turn
            # Check game end after potential penalties or hand changes
            new_state._check_game_end()
            return new_state


        # --- Pending Action Resolution Handling ---
        elif new_state.pending_action:
            if acting_player != new_state.pending_action_player:
                 logger.error(f"Action {action} received from P{acting_player} but pending action is for P{new_state.pending_action_player}")
                 return new_state # Ignore action from wrong player

            pending_type = new_state.pending_action
            player = new_state.pending_action_player
            discard_for_snap_check = None # Card that might trigger snaps

            # Resolve Post-Draw Choice
            if isinstance(pending_type, ActionDiscard) and isinstance(action, (ActionDiscard, ActionReplace)):
                 drawn_card = new_state.pending_action_data.get("drawn_card")
                 if not drawn_card: logger.error("Pending post-draw choice but no drawn_card!"); new_state._clear_pending_action(); return new_state

                 if isinstance(action, ActionDiscard):
                      logger.debug(f"Player {player} discards drawn {drawn_card}. Use ability: {action.use_ability}")
                      new_state.discard_pile.append(drawn_card)
                      discard_for_snap_check = drawn_card
                      # Clear pending *before* triggering ability to avoid loop if ability also needs choice
                      new_state._clear_pending_action()
                      if action.use_ability and new_state._card_has_discard_ability(drawn_card):
                           new_state._trigger_discard_ability(player, drawn_card)
                           # If ability requires further choice, return state here
                           if new_state.pending_action: return new_state
                      # else: (No ability or ability doesn't require choice) Proceed to snap check/turn end

                 elif isinstance(action, ActionReplace):
                      target_idx = action.target_hand_index
                      hand = new_state.players[player].hand # player object already validated
                      if 0 <= target_idx < len(hand):
                           replaced_card = hand[target_idx]
                           logger.debug(f"Player {player} replaces card at index {target_idx} ({replaced_card}) with drawn {drawn_card}.")
                           hand[target_idx] = drawn_card
                           new_state.discard_pile.append(replaced_card)
                           discard_for_snap_check = replaced_card
                           new_state._clear_pending_action()
                      else:
                           logger.error(f"Invalid REPLACE action index: {target_idx}")
                           new_state._clear_pending_action() # Clear pending even on error

            # Resolve Ability Selections
            elif isinstance(pending_type, ActionAbilityPeekOwnSelect) and isinstance(action, ActionAbilityPeekOwnSelect):
                  target_idx = action.target_hand_index; hand = new_state.players[player].hand
                  if 0 <= target_idx < len(hand): logger.info(f"P{player} uses 7/8, peeks own {target_idx}: {hand[target_idx]}")
                  else: logger.error(f"Invalid PEEK_OWN index {target_idx}");
                  discard_for_snap_check = new_state.discard_pile[-1] if new_state.discard_pile else None
                  new_state._clear_pending_action()
            elif isinstance(pending_type, ActionAbilityPeekOtherSelect) and isinstance(action, ActionAbilityPeekOtherSelect):
                  opp_idx = new_state.get_opponent_index(player); target_opp_idx = action.target_opponent_hand_index;
                  # Validate opponent before access
                  if 0 <= opp_idx < len(new_state.players) and hasattr(new_state.players[opp_idx], 'hand'):
                     opp_hand = new_state.players[opp_idx].hand
                     if 0 <= target_opp_idx < len(opp_hand): logger.info(f"P{player} uses 9/T, peeks opp {target_opp_idx}: {opp_hand[target_opp_idx]}")
                     else: logger.error(f"Invalid PEEK_OTHER index {target_opp_idx}")
                  else: logger.error(f"Peek Other: Opponent {opp_idx} invalid or missing hand.")
                  discard_for_snap_check = new_state.discard_pile[-1] if new_state.discard_pile else None
                  new_state._clear_pending_action()
            elif isinstance(pending_type, ActionAbilityBlindSwapSelect) and isinstance(action, ActionAbilityBlindSwapSelect):
                  own_h_idx, opp_h_idx = action.own_hand_index, action.opponent_hand_index; opp_idx = new_state.get_opponent_index(player); hand = new_state.players[player].hand
                  # Validate opponent before access
                  if 0 <= opp_idx < len(new_state.players) and hasattr(new_state.players[opp_idx], 'hand'):
                      opp_hand = new_state.players[opp_idx].hand
                      if 0 <= own_h_idx < len(hand) and 0 <= opp_h_idx < len(opp_hand):
                          hand[own_h_idx], opp_hand[opp_h_idx] = opp_hand[opp_h_idx], hand[own_h_idx]; logger.info(f"P{player} uses J/Q, blind swaps own {own_h_idx} with opp {opp_h_idx}.")
                      else: logger.error(f"Invalid BLIND_SWAP indices: own {own_h_idx}, opp {opp_h_idx}")
                  else: logger.error(f"Blind Swap: Opponent {opp_idx} invalid or missing hand.")
                  discard_for_snap_check = new_state.discard_pile[-1] if new_state.discard_pile else None
                  new_state._clear_pending_action()
            elif isinstance(pending_type, ActionAbilityKingLookSelect) and isinstance(action, ActionAbilityKingLookSelect):
                  own_h_idx, opp_h_idx = action.own_hand_index, action.opponent_hand_index; opp_idx = new_state.get_opponent_index(player); hand = new_state.players[player].hand
                  # Validate opponent before access
                  if 0 <= opp_idx < len(new_state.players) and hasattr(new_state.players[opp_idx], 'hand'):
                       opp_hand = new_state.players[opp_idx].hand
                       if 0 <= own_h_idx < len(hand) and 0 <= opp_h_idx < len(opp_hand):
                           card1, card2 = hand[own_h_idx], opp_hand[opp_h_idx]; logger.info(f"P{player} uses K, looks at own {own_h_idx} ({card1}) and opp {opp_h_idx} ({card2}).")
                           # Set up next pending step: KingSwapDecision
                           new_state.pending_action = ActionAbilityKingSwapDecision(perform_swap=False); new_state.pending_action_player = player
                           new_state.pending_action_data = {"own_idx": own_h_idx, "opp_idx": opp_h_idx, "card1": card1, "card2": card2} # Store context for swap
                           return new_state # Return intermediate state waiting for swap decision
                       else:
                           logger.error(f"Invalid KING_LOOK indices: own {own_h_idx}, opp {opp_h_idx}. Ability fizzles.");
                           discard_for_snap_check = new_state.discard_pile[-1] if new_state.discard_pile else None
                           new_state._clear_pending_action()
                  else:
                       logger.error(f"King Look: Opponent {opp_idx} invalid or missing hand. Ability fizzles.");
                       discard_for_snap_check = new_state.discard_pile[-1] if new_state.discard_pile else None
                       new_state._clear_pending_action()
            elif isinstance(pending_type, ActionAbilityKingSwapDecision) and isinstance(action, ActionAbilityKingSwapDecision):
                  perform_swap = action.perform_swap; look_data = new_state.pending_action_data
                  own_h_idx, opp_h_idx = look_data.get("own_idx"), look_data.get("opp_idx")
                  card1, card2 = look_data.get("card1"), look_data.get("card2")
                  if own_h_idx is None or opp_h_idx is None or card1 is None or card2 is None:
                      logger.error("Missing data for King Swap decision. Fizzling.")
                  elif perform_swap:
                      opp_idx = new_state.get_opponent_index(player); hand = new_state.players[player].hand
                      # Validate opponent before access
                      if 0 <= opp_idx < len(new_state.players) and hasattr(new_state.players[opp_idx], 'hand'):
                           opp_hand = new_state.players[opp_idx].hand
                           if 0 <= own_h_idx < len(hand) and 0 <= opp_h_idx < len(opp_hand):
                                hand[own_h_idx], opp_hand[opp_h_idx] = card2, card1 # Perform the swap
                                logger.info(f"P{player} King ability: Swapped own {own_h_idx} ({card1}) with opp {opp_h_idx} ({card2}).")
                           else: logger.error(f"Indices invalid at King Swap decision ({own_h_idx}, {opp_h_idx}).")
                      else: logger.error(f"King Swap: Opponent {opp_idx} invalid or missing hand.")
                  else: logger.info(f"P{player} King ability: Chose not to swap.")
                  discard_for_snap_check = new_state.discard_pile[-1] if new_state.discard_pile else None
                  new_state._clear_pending_action()
            elif isinstance(pending_type, ActionSnapOpponentMove) and isinstance(action, ActionSnapOpponentMove):
                 snapper_idx = player; own_card_idx, target_slot_idx = action.own_card_to_move_hand_index, action.target_empty_slot_index
                 hand = new_state.players[snapper_idx].hand; opp_idx = new_state.get_opponent_index(snapper_idx);
                 # Validate opponent before access
                 if 0 <= opp_idx < len(new_state.players) and hasattr(new_state.players[opp_idx], 'hand'):
                      opp_hand = new_state.players[opp_idx].hand
                      if 0 <= own_card_idx < len(hand):
                           moved_card = hand.pop(own_card_idx)
                           if 0 <= target_slot_idx <= len(opp_hand): # Allow inserting at the end
                               opp_hand.insert(target_slot_idx, moved_card)
                               logger.info(f"P{snapper_idx} completes Snap Opponent: Moves {moved_card} (from own idx {own_card_idx}) to opp idx {target_slot_idx}.")
                               # Note: Snap log was already written during the snap attempt.
                               new_state._clear_pending_action()
                               new_state._advance_turn() # Turn advances after move completes
                               return new_state # Return after advancing turn
                           else:
                                logger.error(f"Invalid target slot index {target_slot_idx} (Opp hand size {len(opp_hand)}) for SnapOpponentMove.");
                                hand.insert(own_card_idx, moved_card) # Put card back on error? Risky.
                                new_state._clear_pending_action(); new_state._advance_turn()
                                return new_state # Return after advancing turn
                      else:
                           logger.error(f"Invalid own card index {own_card_idx} (Hand size {len(hand)}) for SnapOpponentMove.");
                           new_state._clear_pending_action(); new_state._advance_turn()
                           return new_state # Return after advancing turn
                 else:
                      logger.error(f"Snap Opponent Move: Opponent {opp_idx} invalid or missing hand.")
                      new_state._clear_pending_action(); new_state._advance_turn()
                      return new_state # Return after advancing turn
            else:
                logger.warning(f"Unhandled pending action ({pending_type}) vs received action ({action})")
                new_state._clear_pending_action() # Clear pending to avoid getting stuck

            # After resolving pending action (except ongoing King/Snap Move), check for snaps or advance turn
            if not new_state.pending_action: # If ability/choice fully resolved
                if discard_for_snap_check and new_state._initiate_snap_phase(discarded_card=discard_for_snap_check):
                    # Snap phase started, next action will be snap action
                    return new_state # Return state waiting for snap action
                elif not new_state.snap_phase_active: # Ensure snap phase didn't start AND not pending
                    new_state._advance_turn() # Advance if no snaps and no further pending steps


        # --- Handle Standard Start-of-Turn Actions ---
        elif isinstance(action, ActionDrawStockpile):
            player = new_state.current_player_index
            if not new_state.stockpile: new_state._attempt_reshuffle()
            if new_state.stockpile:
                drawn_card = new_state.stockpile.pop(); logger.debug(f"P{player} drew {drawn_card} from stockpile.")
                new_state.pending_action = ActionDiscard(use_ability=False); new_state.pending_action_player = player
                new_state.pending_action_data = {"drawn_card": drawn_card}
            else: logger.warning(f"P{player} tried DRAW_STOCKPILE, but stockpile/discard empty. Game should end."); new_state._game_over = True

        elif isinstance(action, ActionDrawDiscard):
             player = new_state.current_player_index
             if new_state.house_rules.allowDrawFromDiscardPile and new_state.discard_pile:
                  drawn_card = new_state.discard_pile.pop(); logger.debug(f"P{player} drew {drawn_card} from discard pile.")
                  new_state.pending_action = ActionDiscard(use_ability=False); new_state.pending_action_player = player
                  new_state.pending_action_data = {"drawn_card": drawn_card} # Allow discard/replace from discard draw
             else: logger.error("Invalid Action: DRAW_DISCARD attempted.")

        elif isinstance(action, ActionCallCambia):
             player = new_state.current_player_index
             if new_state.cambia_caller_id is None: logger.info(f"P{player} calls Cambia!"); new_state.cambia_caller_id = player; new_state.turns_after_cambia = 0; new_state._advance_turn()
             else: logger.warning(f"P{player} tried invalid CALL_CAMBIA.")

        else:
             logger.warning(f"Unhandled action type at start of turn: {type(action)}")

        # Final checks (only if turn didn't already advance and return)
        new_state._check_game_end()
        return new_state

    def _clear_pending_action(self):
        """Resets the pending action state."""
        self.pending_action = None
        self.pending_action_player = None
        self.pending_action_data = {}

    def _trigger_discard_ability(self, player_index: int, discarded_card: Card):
        """Sets up the pending state for executing a special ability."""
        rank = discarded_card.rank
        logger.debug(f"Player {player_index} triggering ability of discarded {discarded_card}")
        ability_triggered = False
        if rank in [SEVEN, EIGHT]: self.pending_action = ActionAbilityPeekOwnSelect(target_hand_index=-1); ability_triggered = True
        elif rank in [NINE, TEN]: self.pending_action = ActionAbilityPeekOtherSelect(target_opponent_hand_index=-1); ability_triggered = True
        elif rank in [JACK, QUEEN]: self.pending_action = ActionAbilityBlindSwapSelect(own_hand_index=-1, opponent_hand_index=-1); ability_triggered = True
        elif rank == KING: self.pending_action = ActionAbilityKingLookSelect(own_hand_index=-1, opponent_hand_index=-1); ability_triggered = True

        if ability_triggered:
             self.pending_action_player = player_index
             self.pending_action_data = {"ability_card": discarded_card}
        else: logger.debug(f"Card {discarded_card} has no relevant discard ability.")


    def _initiate_snap_phase(self, discarded_card: Card) -> bool:
        """ Checks potential snappers and starts the snap phase. Returns True if started. """
        self.snap_phase_active = False
        self.snap_discarded_card = None
        self.snap_potential_snappers = []
        self.snap_current_snapper_idx = 0
        self.snap_results_log = [] # Clear log for new phase

        potential_indices = []
        target_rank = discarded_card.rank

        for p_idx in range(self.num_players):
            # Check if player object is valid before accessing hand
            if not (0 <= p_idx < len(self.players) and hasattr(self.players[p_idx], 'hand')):
                 logger.warning(f"Initiate Snap: Player {p_idx} invalid or missing hand. Skipping.")
                 continue

            if p_idx == self.cambia_caller_id: continue # Cambia caller cannot snap

            hand = self.players[p_idx].hand
            can_snap_own = any(card.rank == target_rank for card in hand)

            can_snap_opponent = False
            if self.house_rules.allowOpponentSnapping:
                 opp_idx = self.get_opponent_index(p_idx)
                 # Check opponent exists and is not the cambia caller
                 if opp_idx != self.cambia_caller_id:
                      # Check if opponent object is valid
                      if 0 <= opp_idx < len(self.players) and hasattr(self.players[opp_idx], 'hand'):
                           opp_hand = self.players[opp_idx].hand
                           can_snap_opponent = any(card.rank == target_rank for card in opp_hand)
                      else:
                           logger.warning(f"Initiate Snap: Opponent {opp_idx} invalid/missing hand for P{p_idx} checking SnapOpponent.")


            if can_snap_own or can_snap_opponent:
                 potential_indices.append(p_idx)

        if potential_indices:
             logger.debug(f"Discard of {discarded_card} triggers snap phase.")
             self.snap_phase_active = True
             self.snap_discarded_card = discarded_card
             # Order is turn order starting from next player after discarder
             discarder_player = self.current_player_index # Player who just finished their action
             ordered_snappers = []
             for i in range(1, self.num_players + 1): # Check self.num_players players
                  check_p_idx = (discarder_player + i) % self.num_players
                  if check_p_idx in potential_indices:
                       ordered_snappers.append(check_p_idx)

             if not ordered_snappers: # Should not happen if potential_indices is not empty
                  logger.error("Potential snappers found, but ordering failed.")
                  self.snap_phase_active = False
                  return False

             self.snap_potential_snappers = ordered_snappers
             self.snap_current_snapper_idx = 0
             logger.debug(f"Potential snappers (ordered): {ordered_snappers}. P{self.get_acting_player()} acts first.")
             return True
        else:
             logger.debug(f"No potential snappers found for rank {target_rank}.")
             return False # No snaps possible

    def _end_snap_phase(self):
         """Cleans up snap phase state and advances the main game turn."""
         logger.debug("Ending snap phase.")
         self.snap_phase_active = False
         # Keep snap_discarded_card and snap_results_log for observation creation?
         # Clear them if they are not needed after snap phase. Let's clear for now.
         self.snap_discarded_card = None
         self.snap_potential_snappers = []
         self.snap_current_snapper_idx = 0
         # Do NOT clear snap_results_log here, needed for observation in CFR
         self._advance_turn() # Advance turn AFTER snap phase concludes

    def _attempt_reshuffle(self):
        """Reshuffles the discard pile into the stockpile if possible."""
        if not self.stockpile and len(self.discard_pile) > 1:
             logger.info("Stockpile empty. Reshuffling discard pile.")
             top_card = self.discard_pile.pop()
             cards_to_shuffle = self.discard_pile
             self.discard_pile = [top_card]
             random.shuffle(cards_to_shuffle)
             self.stockpile = cards_to_shuffle
             logger.info(f"Reshuffled {len(self.stockpile)} cards into stockpile.")
        elif not self.stockpile:
             logger.info("Stockpile empty, cannot reshuffle discard pile (size <= 1).")

    def _apply_penalty(self, player_index: int, num_cards: int):
        """Adds penalty cards to the player's hand, reshuffling if needed."""
        logger.warning(f"Applying penalty: Player {player_index} draws {num_cards} cards.")
        # Check if player object is valid before applying penalty
        if not (0 <= player_index < len(self.players) and hasattr(self.players[player_index], 'hand')):
             logger.error(f"Cannot apply penalty: Player {player_index} invalid or missing hand.")
             return

        drawn_penalty_cards = []
        for _ in range(num_cards):
            if not self.stockpile: self._attempt_reshuffle()
            if self.stockpile:
                drawn_card = self.stockpile.pop()
                self.players[player_index].hand.append(drawn_card)
                drawn_penalty_cards.append(drawn_card)
                logger.debug(f"Player {player_index} penalty draw: {drawn_card}")
            else: logger.warning("Stockpile/discard empty during penalty draw!"); break
        # Note: We don't explicitly return drawn_penalty_cards here, agent only sees count change.

    def _advance_turn(self):
        """Moves to the next player, increments turn counter, handles Cambia turns."""
        if self._game_over: return

        if self.cambia_caller_id is not None:
            self.turns_after_cambia += 1

        self.current_player_index = (self.current_player_index + 1) % self.num_players
        self._turn_number += 1 # Increment turn counter when player changes
        logger.debug(f"Turn advances to {self._turn_number}. Current player: {self.current_player_index}")
        self._check_game_end() # Check if advancing turn ended the game


    def _check_game_end(self):
        """Checks and sets game end conditions."""
        if self._game_over: return

        # 1. Cambia final turns completed
        if self.cambia_caller_id is not None and self.turns_after_cambia >= self.num_players :
            logger.info("Game ends: Cambia called and final turns completed.")
            self._game_over = True

        # 2. Draw required but impossible
        # Check if it's start of turn, no pending action, no snap phase, and DRAW action is required but impossible
        if not self._game_over and not self.pending_action and not self.snap_phase_active:
            player = self.current_player_index
            # Check if player object is valid before getting legal actions
            if 0 <= player < len(self.players) and hasattr(self.players[player], 'hand'):
                 legal_actions = self.get_legal_actions()
                 is_draw_required = not any(isinstance(act, ActionCallCambia) for act in legal_actions) # Draw required if Cambia not legal/chosen
                 if is_draw_required and not self.stockpile and not (self.discard_pile and len(self.discard_pile) > 1):
                      logger.info(f"Game ends: Player {player} requires draw, but stockpile and discard cannot be replenished.")
                      self._game_over = True
            else:
                 logger.error(f"Game end check: Player {player} invalid/missing hand.")
                 self._game_over = True # Force end if state invalid


        # 3. Explicit check if start of turn and no actions possible (covers non-draw end conditions)
        if not self._game_over and not self.pending_action and not self.snap_phase_active:
             player = self.current_player_index
             # Check player object validity
             if 0 <= player < len(self.players) and hasattr(self.players[player], 'hand'):
                 if not self.get_legal_actions():
                       logger.info("Game ends: No legal actions available at start of turn.")
                       self._game_over = True
             else:
                  logger.error(f"Game end check (no actions): Player {player} invalid/missing hand.")
                  self._game_over = True

        if self._game_over:
             self._calculate_final_scores()

    def _calculate_final_scores(self):
        """Calculates final scores and determines winner/utilities."""
        if self._utilities != [0.0] * self.num_players and not np.all(np.array(self._utilities) == 0): return # Avoid recalculating

        scores = []
        for i in range(self.num_players):
            # Check player object validity
            if 0 <= i < len(self.players) and hasattr(self.players[i], 'hand'):
                 hand_value = sum(card.value for card in self.players[i].hand)
                 scores.append(hand_value)
                 logger.info(f"Player {i} final hand: {[str(c) for c in self.players[i].hand]} Score: {hand_value}")
            else:
                 logger.error(f"Calculate score: Player {i} invalid or missing hand. Assigning high score.")
                 scores.append(999) # Assign penalty score

        min_score = min(scores) if scores else 0
        winners = [i for i, score in enumerate(scores) if score == min_score]

        if len(winners) == 1: self._winner = winners[0]
        elif self.cambia_caller_id is not None and self.cambia_caller_id in winners: self._winner = self.cambia_caller_id
        else: self._winner = None # Draw

        if self._winner is not None:
             self._utilities[self._winner] = 1.0
             loser = self.get_opponent_index(self._winner)
             if 0 <= loser < self.num_players: # Check loser index valid
                  self._utilities[loser] = -1.0
             logger.info(f"Player {self._winner} wins with score {min_score}.")
        else: self._utilities = [0.0] * self.num_players; logger.info(f"Tie between players {winners} with score {min_score}.")
        logger.debug(f"Final Utilities: {self._utilities}")

    def is_terminal(self) -> bool: return self._game_over

    def get_utility(self, player_id: int) -> float:
        """Returns the final utility for the specified player."""
        if not self.is_terminal(): logger.warning("get_utility called on non-terminal state!"); return 0.0
        if 0 <= player_id < self.num_players: self._calculate_final_scores(); return self._utilities[player_id] # Ensure calculated
        raise IndexError("Invalid player index for utility")

    def get_player_turn(self) -> int: return self.current_player_index # Whose turn it nominally is

    def get_acting_player(self) -> int:
         """Returns the index of the player who needs to act *now*."""
         if self.snap_phase_active:
             if self.snap_current_snapper_idx < len(self.snap_potential_snappers):
                  # Check if the potential snapper index is valid
                  potential_snapper = self.snap_potential_snappers[self.snap_current_snapper_idx]
                  if 0 <= potential_snapper < len(self.players) and hasattr(self.players[potential_snapper], 'hand'):
                       return potential_snapper
                  else:
                       logger.error(f"Snap phase acting player index {potential_snapper} invalid or missing hand.")
                       return -1
             else: logger.error("Snap phase active but index out of bounds."); return -1
         elif self.pending_action and self.pending_action_player is not None:
             # Check if the pending action player index is valid
             pending_player = self.pending_action_player
             if 0 <= pending_player < len(self.players) and hasattr(self.players[pending_player], 'hand'):
                 return pending_player
             else:
                 logger.error(f"Pending action player index {pending_player} invalid or missing hand.")
                 return -1
         elif not self._game_over:
             # Check if the current player index is valid
             current_player = self.current_player_index
             if 0 <= current_player < len(self.players) and hasattr(self.players[current_player], 'hand'):
                  return current_player
             else:
                  logger.error(f"Current player index {current_player} invalid or missing hand in active game.")
                  return -1
         else: return -1 # Game over


    def clone(self) -> 'CambiaGameState':
        """Creates a deep copy of the game state, handling potential recursion issues."""
        memo = {} # Memoization dictionary for deepcopy

        # Attempt deepcopy first, fall back to manual if RecursionError occurs
        try:
            # The standard deepcopy should now use the custom __deepcopy__ for PlayerState
            cloned_state = copy.deepcopy(self, memo)
            # Post-clone validation
            if len(cloned_state.players) != len(self.players):
                 logger.error(f"Post-clone validation failed: Incorrect player count ({len(cloned_state.players)} vs {len(self.players)}).")
                 raise ValueError("Cloned state has incorrect number of players.")
            for i, p_cloned in enumerate(cloned_state.players):
                 if not isinstance(p_cloned, PlayerState):
                      logger.error(f"Post-clone validation failed: Player {i} is not a PlayerState object ({type(p_cloned)}).")
                      raise TypeError(f"Cloned player {i} is not a PlayerState object.")
                 if not hasattr(p_cloned, 'hand') or not isinstance(p_cloned.hand, list):
                      logger.error(f"Post-clone validation failed: Player {i} missing hand list or invalid type.")
                      raise TypeError(f"Cloned player {i} is invalid or missing hand list.")
                 if not all(isinstance(card, Card) for card in p_cloned.hand):
                      logger.error(f"Post-clone validation failed: Player {i} hand contains non-Card objects.")
                      raise TypeError(f"Cloned player {i} hand contains non-Card objects.")

            return cloned_state
        except RecursionError:
            logger.warning("RecursionError during deepcopy, attempting manual clone.", exc_info=False)
            # --- Manual Cloning (Fallback) ---
            new_state = CambiaGameState.__new__(CambiaGameState)
            memo[id(self)] = new_state # Add self to memo

            try:
                # Copy players using deepcopy (which should use PlayerState.__deepcopy__)
                new_state.players = [copy.deepcopy(p, memo) for p in self.players]
                # Basic validation after manual player copy
                if len(new_state.players) != len(self.players): raise ValueError("Manual clone player count mismatch")
                for i, p_manual in enumerate(new_state.players):
                    if not isinstance(p_manual, PlayerState): raise TypeError(f"Manual clone player {i} invalid type")
                    if not hasattr(p_manual, 'hand'): raise AttributeError(f"Manual clone player {i} missing hand")

                new_state.stockpile = [copy.deepcopy(c, memo) for c in self.stockpile]
                new_state.discard_pile = [copy.deepcopy(c, memo) for c in self.discard_pile]
                new_state.current_player_index = self.current_player_index
                new_state.num_players = self.num_players
                new_state.cambia_caller_id = self.cambia_caller_id
                new_state.turns_after_cambia = self.turns_after_cambia
                new_state.house_rules = copy.deepcopy(self.house_rules, memo)
                new_state._game_over = self._game_over
                new_state._winner = self._winner
                new_state._utilities = list(self._utilities)
                new_state._turn_number = self._turn_number
                new_state.pending_action = self.pending_action
                new_state.pending_action_player = self.pending_action_player
                new_state.pending_action_data = copy.deepcopy(self.pending_action_data, memo)
                new_state.snap_phase_active = self.snap_phase_active
                new_state.snap_discarded_card = copy.deepcopy(self.snap_discarded_card, memo)
                new_state.snap_potential_snappers = list(self.snap_potential_snappers)
                new_state.snap_current_snapper_idx = self.snap_current_snapper_idx
                new_state.snap_results_log = copy.deepcopy(self.snap_results_log, memo)
                return new_state
            except Exception as manual_clone_err:
                logger.exception(f"Error during manual clone attempt: {manual_clone_err}")
                # Re-raise the original RecursionError or a new error indicating manual clone failure
                raise RuntimeError("Manual clone failed after RecursionError.") from manual_clone_err

        except Exception as e:
            logger.exception(f"Unexpected error during game state cloning: {e}")
            raise # Re-raise other exceptions


    def __str__(self) -> str:
        state_desc = ""
        actor = self.get_acting_player()
        actor_str = f"P{actor}" if actor != -1 else "N/A"
        if self.snap_phase_active: state_desc = f"SnapPhase(Actor: {actor_str}, Target: {self.snap_discarded_card.rank if self.snap_discarded_card else 'N/A'})"
        elif self.pending_action: state_desc = f"Pending(Actor: {actor_str}, Action: {type(self.pending_action).__name__})"
        elif self._game_over: state_desc = "GameOver"
        else: state_desc = f"Turn: {actor_str}"

        discard_top_str = str(self.get_discard_top()) if self.discard_pile else "[]"
        # Add check for player hand validity before string formatting
        hand_lens = []
        for i, p in enumerate(self.players):
             if hasattr(p, 'hand') and isinstance(p.hand, list):
                  hand_lens.append(len(p.hand))
             else:
                  hand_lens.append("ERR")


        return (f"GameState(T#{self._turn_number}, {state_desc}, "
                f"Stock:{len(self.stockpile)}, Discard:{discard_top_str}, "
                f"Hands:{hand_lens}, "
                f"Cambia:{self.cambia_caller_id}, Over:{self._game_over})")