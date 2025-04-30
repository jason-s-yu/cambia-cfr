# src/game_engine.py
import random
from typing import List, Tuple, Optional, Set, Any, Dict
from dataclasses import dataclass, field
import logging
import copy

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

    # Intermediate state for multi-step actions (e.g., King ability, Post-Draw Choice, Snap Move)
    pending_action: Optional[GameAction] = None # Stores the type of action needed next
    pending_action_player: Optional[int] = None # Player who needs to act
    pending_action_data: Dict[str, Any] = field(default_factory=dict) # Extra info needed

    # Snap Phase State
    snap_phase_active: bool = False
    snap_discarded_card: Optional[Card] = None
    snap_potential_snappers: List[int] = field(default_factory=list) # Players who *could* snap
    snap_current_snapper_idx: int = 0 # Index into snap_potential_snappers


    def __post_init__(self):
        if not self.players: # Initialize deck and deal if not already provided
            self._setup_game()

    def _setup_game(self):
        """Initializes the deck, shuffles, and deals cards."""
        self.stockpile = create_standard_deck(include_jokers=self.house_rules.use_jokers)
        random.shuffle(self.stockpile)
        # Determine initial peek count based on rules
        initial_peek_count = self.house_rules.initial_view_count
        cards_per_player = self.house_rules.cards_per_player
        self.players = [PlayerState(initial_peek_indices=tuple(range(initial_peek_count))) for _ in range(self.num_players)]

        for _ in range(cards_per_player):
            for i in range(self.num_players):
                if self.stockpile:
                    self.players[i].hand.append(self.stockpile.pop())
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
        self.pending_action = None
        self.pending_action_player = None
        self.pending_action_data = {}
        self.snap_phase_active = False # Start snap phase inactive
        logger.debug(f"Game setup complete. Player {self.current_player_index} starts.")
        logger.debug(f"Initial hands (hidden): {[len(p.hand) for p in self.players]}")
        logger.debug(f"House Rules: {self.house_rules}")


    def get_player_hand(self, player_index: int) -> List[Card]:
        if 0 <= player_index < self.num_players:
            return self.players[player_index].hand
        raise IndexError("Invalid player index")

    def get_opponent_index(self, player_index: int) -> int:
        return 1 - player_index # Specific to 1v1

    def get_player_card_count(self, player_index: int) -> int:
         return len(self.players[player_index].hand)

    def get_stockpile_size(self) -> int:
        return len(self.stockpile)

    def get_discard_top(self) -> Optional[Card]:
        return self.discard_pile[-1] if self.discard_pile else None

    def get_legal_actions(self) -> Set[GameAction]:
        """Returns the set of valid actions for the current acting player."""
        legal_actions: Set[GameAction] = set()

        if self._game_over:
            return legal_actions

        # --- Snap Phase Actions ---
        if self.snap_phase_active:
            if self.snap_current_snapper_idx >= len(self.snap_potential_snappers):
                 # This case should ideally be handled by apply_action ending the phase
                 logger.warning("In snap phase but no more potential snappers. Attempting to end phase.")
                 # Create a temporary state to end the phase and advance
                 temp_state = self.clone()
                 temp_state._end_snap_phase()
                 return temp_state.get_legal_actions() # Return actions for the *next* state

            acting_snapper = self.snap_potential_snappers[self.snap_current_snapper_idx]
            snapper_hand = self.players[acting_snapper].hand
            target_rank = self.snap_discarded_card.rank

            # Always allow passing the snap opportunity
            legal_actions.add(ActionPassSnap())

            # Check for Snap Own
            for i, card in enumerate(snapper_hand):
                if card.rank == target_rank:
                    legal_actions.add(ActionSnapOwn(own_card_hand_index=i))
                    # In simple model, maybe only allow snapping one card? Add break if needed.

            # Check for Snap Opponent (if allowed by rules)
            if self.house_rules.allowOpponentSnapping:
                 opponent_idx = self.get_opponent_index(acting_snapper)
                 opponent_hand = self.players[opponent_idx].hand
                 # Note: Player normally doesn't know opponent's hand.
                 # For engine/CFR simulation, we check the true hand.
                 # The *agent* needs to decide based on its belief.
                 for i, card in enumerate(opponent_hand):
                      if card.rank == target_rank:
                           # Agent needs belief/memory to choose target_opponent_hand_index
                           # For now, engine identifies possibility based on true state.
                           legal_actions.add(ActionSnapOpponent(opponent_target_hand_index=i))

            return legal_actions # Return only snap-related actions

        # --- Pending Action Resolution ---
        elif self.pending_action:
            action_type = self.pending_action
            player = self.pending_action_player
            # Must resolve the pending action
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
                           # Swap own i with opponent j
                           legal_actions.add(ActionAbilityBlindSwapSelect(own_hand_index=i, opponent_hand_index=j))
                 # Add swaps between own cards? Or opponent cards? Check rules. Assume own<->opp only.
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
            elif isinstance(action_type, ActionSnapOpponentMove): #
                 snapper_idx = player
                 target_slot = self.pending_action_data.get("target_empty_slot_index")
                 for i in range(self.get_player_card_count(snapper_idx)):
                      legal_actions.add(ActionSnapOpponentMove(own_card_to_move_hand_index=i, target_empty_slot_index=target_slot))

            else:
                 logger.error(f"Unknown pending action type: {action_type}")
                 # Fallback or error handling

            return legal_actions # Only allow resolving the pending action


        # --- Standard Start-of-Turn Actions ---
        player = self.current_player_index

        # 1. Draw from Stockpile (Handle reshuffle if needed)
        if self.stockpile:
            legal_actions.add(ActionDrawStockpile())
        elif self.discard_pile and len(self.discard_pile) > 1:
            # Can draw after reshuffle
            legal_actions.add(ActionDrawStockpile()) # Represent drawing from newly shuffled stockpile
        else:
             # Cannot draw, game might end or force Cambia
             pass

        # 2. Draw from Discard Pile (if house rule allows)
        if self.house_rules.allowDrawFromDiscardPile and self.discard_pile:
             legal_actions.add(ActionDrawDiscard())

        # 3. Call Cambia
        if self.cambia_caller_id is None:
            # Add conditions based on house rules (e.g., minimum turns passed)
             legal_actions.add(ActionCallCambia())

        # Ensure some action is always possible if game not over
        if not legal_actions and not self._game_over and not self.stockpile and len(self.discard_pile) <= 1:
             # Only possible state is game end due to deck exhaustion
             pass # Apply action will handle this by ending game
        elif not legal_actions and not self._game_over:
             logger.error(f"No legal actions found for player {player} in state: {self}")
             # This might indicate a state error or incomplete legal action logic

        return legal_actions

    def _card_has_discard_ability(self, card: Card) -> bool:
        """Checks if a card has an ability when discarded from draw."""
        # Based on Rule 1. Abilities only trigger on discard *from draw*
        # and house_rules.allowReplaceAbilities == False (assumed default)
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

        # --- Snap Phase Action Handling ---
        if new_state.snap_phase_active:
            acting_snapper = new_state.snap_potential_snappers[new_state.snap_current_snapper_idx]
            logger.debug(f"Snap Phase: Player {acting_snapper} choosing action: {action}")

            if isinstance(action, ActionPassSnap):
                 logger.debug(f"Player {acting_snapper} passes snap.")
                 new_state.snap_current_snapper_idx += 1 # Move to next potential snapper

            elif isinstance(action, ActionSnapOwn):
                 target_rank = new_state.snap_discarded_card.rank
                 snap_idx = action.own_card_hand_index
                 hand = new_state.players[acting_snapper].hand
                 if 0 <= snap_idx < len(hand) and hand[snap_idx].rank == target_rank:
                      snapped_card = hand.pop(snap_idx)
                      new_state.discard_pile.append(snapped_card)
                      logger.info(f"Player {acting_snapper} snaps own {snapped_card} (matching {target_rank}). Hand size: {len(hand)}")
                      # Handle snapping last card if needed (e.g., force Cambia later?)
                 else:
                      logger.warning(f"Player {acting_snapper} attempted invalid Snap Own: {action}. Applying penalty.")
                      new_state._apply_penalty(acting_snapper, new_state.house_rules.penaltyDrawCount)
                 new_state.snap_current_snapper_idx += 1 # Move to next potential snapper

            elif isinstance(action, ActionSnapOpponent):
                 if not new_state.house_rules.allowOpponentSnapping:
                      logger.error("Invalid Action: SnapOpponent attempted but rule disallowed.")
                      # Apply penalty? Or should not be legal? Assume penalty for trying.
                      new_state._apply_penalty(acting_snapper, new_state.house_rules.penaltyDrawCount)
                 else:
                      target_rank = new_state.snap_discarded_card.rank
                      opp_idx = new_state.get_opponent_index(acting_snapper)
                      opp_hand = new_state.players[opp_idx].hand
                      target_opp_hand_idx = action.opponent_target_hand_index

                      if 0 <= target_opp_hand_idx < len(opp_hand) and opp_hand[target_opp_hand_idx].rank == target_rank:
                           removed_card = opp_hand.pop(target_opp_hand_idx)
                           logger.info(f"Player {acting_snapper} snaps opponent's {removed_card} at index {target_opp_hand_idx}.")
                           # --- Must now move own card ---
                           new_state.pending_action = ActionSnapOpponentMove(own_card_to_move_hand_index=-1, target_empty_slot_index=-1) # Placeholder type
                           new_state.pending_action_player = acting_snapper
                           new_state.pending_action_data = {"target_empty_slot_index": target_opp_hand_idx} # Store where card was removed
                           new_state.snap_phase_active = False # Exit snap phase to resolve the move
                           return new_state # Return intermediate state waiting for move choice
                      else:
                           logger.warning(f"Player {acting_snapper} attempted invalid Snap Opponent: {action}. Applying penalty.")
                           new_state._apply_penalty(acting_snapper, new_state.house_rules.penaltyDrawCount)
                 new_state.snap_current_snapper_idx += 1 # Move to next potential snapper

            else:
                logger.error(f"Invalid action type {type(action)} received during snap phase.")
                new_state.snap_current_snapper_idx += 1 # Skip player on error

            # Check if snap phase ends
            if new_state.snap_current_snapper_idx >= len(new_state.snap_potential_snappers):
                 new_state._end_snap_phase()
            # Check game end after potential penalties
            new_state._check_game_end()
            return new_state # Return state after snap action resolved

        # --- Pending Action Resolution Handling ---
        elif new_state.pending_action:
            if new_state.pending_action_player != new_state.current_player_index:
                 logger.error(f"Action received from P{new_state.current_player_index} but pending action is for P{new_state.pending_action_player}")
                 # Handle error? Ignore action? For now, proceed assuming it's the right player.

            pending_type = new_state.pending_action
            player = new_state.pending_action_player
            current_player = new_state.current_player_index # Player whose turn it is *overall*

            # Resolve Post-Draw Choice
            if isinstance(pending_type, ActionDiscard) and isinstance(action, (ActionDiscard, ActionReplace)):
                 drawn_card = new_state.pending_action_data.get("drawn_card")
                 if not drawn_card:
                      logger.error("Pending post-draw choice but no drawn_card in data!")
                      # Clear pending state and hope for recovery?
                      new_state._clear_pending_action()
                      return new_state # Risky

                 if isinstance(action, ActionDiscard):
                      logger.debug(f"Player {player} discards drawn {drawn_card}. Use ability: {action.use_ability}")
                      new_state.discard_pile.append(drawn_card)
                      new_state._clear_pending_action() # Clear before ability trigger
                      if action.use_ability and new_state._card_has_discard_ability(drawn_card):
                           new_state._trigger_discard_ability(player, drawn_card)
                      # Initiate snaps or advance turn if no ability pending
                      if not new_state.pending_action:
                           if new_state._initiate_snap_phase(discarded_card=drawn_card):
                                return new_state # Snap phase started
                           else:
                                new_state._advance_turn() # No snaps possible or phase ended immediately
                 elif isinstance(action, ActionReplace):
                      target_idx = action.target_hand_index
                      hand = new_state.players[player].hand
                      if 0 <= target_idx < len(hand):
                           replaced_card = hand[target_idx]
                           logger.debug(f"Player {player} replaces card at index {target_idx} ({replaced_card}) with drawn {drawn_card}.")
                           hand[target_idx] = drawn_card
                           new_state.discard_pile.append(replaced_card)
                           new_state._clear_pending_action()
                           # Handle replace abilities rule if needed
                           # Initiate snaps for the replaced card or advance turn
                           if new_state._initiate_snap_phase(discarded_card=replaced_card):
                                return new_state # Snap phase started
                           else:
                                new_state._advance_turn()
                      else:
                           logger.error(f"Invalid REPLACE action index: {target_idx}")
                           # Apply penalty? Clear pending state?
                           new_state._clear_pending_action() # Clear pending state on error

            # Resolve Ability Selections
            elif isinstance(pending_type, ActionAbilityPeekOwnSelect) and isinstance(action, ActionAbilityPeekOwnSelect):
                 target_idx = action.target_hand_index
                 hand = new_state.players[player].hand
                 if 0 <= target_idx < len(hand):
                      peeked_card = hand[target_idx]
                      logger.info(f"Player {player} uses 7/8 ability, peeks own index {target_idx}: {peeked_card}")
                      # Store peek result if needed for observation? Usually just info for the player.
                      new_state._clear_pending_action()
                      discarded_card = new_state.discard_pile[-1] # Ability card
                      if new_state._initiate_snap_phase(discarded_card=discarded_card): return new_state
                      else: new_state._advance_turn()
                 else: logger.error(f"Invalid PEEK_OWN index {target_idx}")
            elif isinstance(pending_type, ActionAbilityPeekOtherSelect) and isinstance(action, ActionAbilityPeekOtherSelect):
                 opp_idx = new_state.get_opponent_index(player)
                 target_opp_idx = action.target_opponent_hand_index
                 opp_hand = new_state.players[opp_idx].hand
                 if 0 <= target_opp_idx < len(opp_hand):
                      peeked_card = opp_hand[target_opp_idx]
                      logger.info(f"Player {player} uses 9/T ability, peeks opponent index {target_opp_idx}: {peeked_card}")
                      new_state._clear_pending_action()
                      discarded_card = new_state.discard_pile[-1]
                      if new_state._initiate_snap_phase(discarded_card=discarded_card): return new_state
                      else: new_state._advance_turn()
                 else: logger.error(f"Invalid PEEK_OTHER index {target_opp_idx}")
            elif isinstance(pending_type, ActionAbilityBlindSwapSelect) and isinstance(action, ActionAbilityBlindSwapSelect):
                 own_h_idx = action.own_hand_index
                 opp_h_idx = action.opponent_hand_index
                 opp_idx = new_state.get_opponent_index(player)
                 hand = new_state.players[player].hand
                 opp_hand = new_state.players[opp_idx].hand
                 if 0 <= own_h_idx < len(hand) and 0 <= opp_h_idx < len(opp_hand):
                      card1 = hand[own_h_idx]
                      card2 = opp_hand[opp_h_idx]
                      hand[own_h_idx], opp_hand[opp_h_idx] = card2, card1
                      logger.info(f"Player {player} uses J/Q ability, blind swaps own {own_h_idx} ({card1}) with opp {opp_h_idx} ({card2}).")
                      new_state._clear_pending_action()
                      discarded_card = new_state.discard_pile[-1]
                      if new_state._initiate_snap_phase(discarded_card=discarded_card): return new_state
                      else: new_state._advance_turn()
                 else: logger.error(f"Invalid BLIND_SWAP indices: own {own_h_idx}, opp {opp_h_idx}")
            elif isinstance(pending_type, ActionAbilityKingLookSelect) and isinstance(action, ActionAbilityKingLookSelect):
                 own_h_idx = action.own_hand_index
                 opp_h_idx = action.opponent_hand_index
                 opp_idx = new_state.get_opponent_index(player)
                 hand = new_state.players[player].hand
                 opp_hand = new_state.players[opp_idx].hand
                 if 0 <= own_h_idx < len(hand) and 0 <= opp_h_idx < len(opp_hand):
                      card1 = hand[own_h_idx]
                      card2 = opp_hand[opp_h_idx]
                      logger.info(f"Player {player} uses K ability, looks at own {own_h_idx} ({card1}) and opp {opp_h_idx} ({card2}).")
                      # Now needs swap decision
                      new_state.pending_action = ActionAbilityKingSwapDecision(perform_swap=False) # Type placeholder
                      new_state.pending_action_player = player
                      new_state.pending_action_data = {"own_idx": own_h_idx, "opp_idx": opp_h_idx, "card1": card1, "card2": card2}
                 else:
                      logger.error(f"Invalid KING_LOOK indices: own {own_h_idx}, opp {opp_h_idx}. Ability fizzles.")
                      new_state._clear_pending_action()
                      discarded_card = new_state.discard_pile[-1]
                      if new_state._initiate_snap_phase(discarded_card=discarded_card): return new_state
                      else: new_state._advance_turn()
            elif isinstance(pending_type, ActionAbilityKingSwapDecision) and isinstance(action, ActionAbilityKingSwapDecision):
                 perform_swap = action.perform_swap
                 look_data = new_state.pending_action_data
                 if perform_swap:
                      own_h_idx = look_data["own_idx"]
                      opp_h_idx = look_data["opp_idx"]
                      opp_idx = new_state.get_opponent_index(player)
                      hand = new_state.players[player].hand
                      opp_hand = new_state.players[opp_idx].hand
                      # Re-verify indices just in case state changed? Unlikely mid-ability.
                      if 0 <= own_h_idx < len(hand) and 0 <= opp_h_idx < len(opp_hand):
                           card1 = look_data["card1"]
                           card2 = look_data["card2"]
                           hand[own_h_idx], opp_hand[opp_h_idx] = card2, card1
                           logger.info(f"Player {player} King ability: Swapped own {own_h_idx} ({card1}) with opp {opp_h_idx} ({card2}).")
                      else:
                           logger.error("Indices invalid at King Swap decision time.")
                 else:
                      logger.info(f"Player {player} King ability: Chose not to swap.")
                 new_state._clear_pending_action()
                 discarded_card = new_state.discard_pile[-1]
                 if new_state._initiate_snap_phase(discarded_card=discarded_card): return new_state
                 else: new_state._advance_turn()
            elif isinstance(pending_type, ActionSnapOpponentMove) and isinstance(action, ActionSnapOpponentMove): #
                 snapper_idx = player # Player who needs to make the move choice
                 own_card_idx = action.own_card_to_move_hand_index
                 target_slot_idx = action.target_empty_slot_index
                 hand = new_state.players[snapper_idx].hand
                 opp_idx = new_state.get_opponent_index(snapper_idx)
                 opp_hand = new_state.players[opp_idx].hand

                 if 0 <= own_card_idx < len(hand):
                      moved_card = hand.pop(own_card_idx)
                      # Insert card into opponent's hand at the correct index
                      if 0 <= target_slot_idx <= len(opp_hand): # Check insertion index validity
                           opp_hand.insert(target_slot_idx, moved_card)
                           logger.info(f"Player {snapper_idx} completes Snap Opponent: Moves {moved_card} (from own idx {own_card_idx}) to opponent idx {target_slot_idx}.")
                           new_state._clear_pending_action()
                           # The snap phase was already exited. Now advance turn from original player.
                           new_state._advance_turn()
                      else:
                          logger.error(f"Invalid target slot index {target_slot_idx} for SnapOpponentMove. Opponent hand size: {len(opp_hand)}")
                          new_state._clear_pending_action() # Clear pending state on error
                          # Potentially revert opponent snap? Or just advance? Advance for now.
                          new_state._advance_turn()
                 else:
                      logger.error(f"Invalid own card index {own_card_idx} for SnapOpponentMove.")
                      new_state._clear_pending_action() # Clear pending state on error
                      new_state._advance_turn()

            else:
                 logger.warning(f"Unhandled pending action ({pending_type}) vs received action ({action})")
                 # Attempt recovery: clear pending state and advance turn?
                 new_state._clear_pending_action()
                 new_state._advance_turn()


        # --- Handle Standard Start-of-Turn Actions ---
        elif isinstance(action, ActionDrawStockpile):
            player = new_state.current_player_index
            if not new_state.stockpile: # Check if reshuffle needed
                 new_state._attempt_reshuffle()

            if new_state.stockpile:
                drawn_card = new_state.stockpile.pop()
                logger.debug(f"Player {player} drew {drawn_card} from stockpile.")
                # Set pending state for Discard/Replace choice
                new_state.pending_action = ActionDiscard(use_ability=False) # Placeholder type
                new_state.pending_action_player = player
                new_state.pending_action_data = {"drawn_card": drawn_card}
            else:
                 logger.warning(f"Player {player} tried DRAW_STOCKPILE, but stockpile and discard empty. Game ends.")
                 new_state._game_over = True # End game if no cards available

        elif isinstance(action, ActionDrawDiscard):
             player = new_state.current_player_index
             if new_state.house_rules.allowDrawFromDiscardPile and new_state.discard_pile:
                  drawn_card = new_state.discard_pile.pop()
                  logger.debug(f"Player {player} drew {drawn_card} from discard pile.")
                  # Set pending state for Discard/Replace choice (cannot discard same card)
                  new_state.pending_action = ActionReplace(target_hand_index=-1) # Placeholder type
                  new_state.pending_action_player = player
                  new_state.pending_action_data = {"drawn_card": drawn_card} # Need different logic for replace only?
             else:
                  logger.error("Invalid Action: DRAW_DISCARD attempted.")

        elif isinstance(action, ActionCallCambia):
             player = new_state.current_player_index
             if new_state.cambia_caller_id is None:
                 logger.info(f"Player {player} calls Cambia!")
                 new_state.cambia_caller_id = player
                 new_state.turns_after_cambia = 0
                 new_state._advance_turn() # Next player gets their turn
             else:
                 logger.warning(f"Player {player} tried to call Cambia after it was already called.")

        else:
             logger.warning(f"Unhandled action type at start of turn: {type(action)}")


        # Check game end condition after action resolution and turn advance (if applicable)
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
        if rank in [SEVEN, EIGHT]:
             self.pending_action = ActionAbilityPeekOwnSelect(target_hand_index=-1)
             ability_triggered = True
        elif rank in [NINE, TEN]:
             self.pending_action = ActionAbilityPeekOtherSelect(target_opponent_hand_index=-1)
             ability_triggered = True
        elif rank in [JACK, QUEEN]:
            self.pending_action = ActionAbilityBlindSwapSelect(own_hand_index=-1, opponent_hand_index=-1)
            ability_triggered = True
        elif rank == KING:
             self.pending_action = ActionAbilityKingLookSelect(own_hand_index=-1, opponent_hand_index=-1)
             ability_triggered = True

        if ability_triggered:
             self.pending_action_player = player_index
             self.pending_action_data = {"ability_card": discarded_card} # Store which card triggered it
        else:
             logger.debug(f"Card {discarded_card} has no relevant discard ability or ability use was declined.")


    def _initiate_snap_phase(self, discarded_card: Card) -> bool:
        """
        Checks for potential snappers and starts the snap phase if any exist.
        Returns True if snap phase started, False otherwise.
        """
        self.snap_phase_active = False # Ensure reset
        self.snap_discarded_card = None
        self.snap_potential_snappers = []
        self.snap_current_snapper_idx = 0

        potential_snappers = []
        target_rank = discarded_card.rank

        # Check all players (including discarder, excluding Cambia caller)
        for p_idx in range(self.num_players):
            if p_idx == self.cambia_caller_id: continue

            # Check for Snap Own
            can_snap_own = any(card.rank == target_rank for card in self.players[p_idx].hand)

            # Check for Snap Opponent (if rule allows)
            can_snap_opponent = False
            if self.house_rules.allowOpponentSnapping:
                 opp_idx = self.get_opponent_index(p_idx)
                 can_snap_opponent = any(card.rank == target_rank for card in self.players[opp_idx].hand)

            if can_snap_own or can_snap_opponent:
                 potential_snappers.append(p_idx)

        if potential_snappers:
             # --- Start Snap Phase ---
             logger.debug(f"Discard of {discarded_card} triggers snap phase. Potential snappers: {potential_snappers}")
             self.snap_phase_active = True
             self.snap_discarded_card = discarded_card
             # Order matters if snapRace=True. Use turn order starting from next player.
             current_p = self.current_player_index # Player whose turn it *was*
             ordered_snappers = []
             for i in range(self.num_players):
                  check_p_idx = (current_p + 1 + i) % self.num_players
                  if check_p_idx in potential_snappers:
                       ordered_snappers.append(check_p_idx)

             self.snap_potential_snappers = ordered_snappers
             self.snap_current_snapper_idx = 0
             return True
        else:
             # No potential snappers, proceed normally
             return False

    def _end_snap_phase(self):
         """Cleans up snap phase state and advances the main game turn."""
         logger.debug("Ending snap phase.")
         self.snap_phase_active = False
         self.snap_discarded_card = None
         self.snap_potential_snappers = []
         self.snap_current_snapper_idx = 0
         self._advance_turn()

    def _attempt_reshuffle(self):
         """Reshuffles the discard pile into the stockpile if possible."""
         if not self.stockpile and len(self.discard_pile) > 1:
              logger.info("Stockpile empty. Reshuffling discard pile.")
              top_card = self.discard_pile.pop()
              cards_to_shuffle = self.discard_pile
              self.discard_pile = [top_card] # Keep only the top card
              random.shuffle(cards_to_shuffle)
              self.stockpile = cards_to_shuffle
              logger.info(f"Reshuffled {len(self.stockpile)} cards into stockpile.")
         elif not self.stockpile:
              logger.info("Stockpile empty, cannot reshuffle discard pile (size <= 1).")


    def _apply_penalty(self, player_index: int, num_cards: int):
        """Applies penalty by adding cards to the player's hand, reshuffling if needed."""
        logger.warning(f"Applying penalty: Player {player_index} draws {num_cards} cards.")
        for _ in range(num_cards):
            if not self.stockpile:
                self._attempt_reshuffle()
            if self.stockpile:
                drawn_card = self.stockpile.pop()
                self.players[player_index].hand.append(drawn_card)
                logger.debug(f"Player {player_index} penalty draw: {drawn_card}")
            else:
                logger.warning("Stockpile and discard empty during penalty draw! Cannot draw more.")
                self._check_game_end() # Check if game should end now
                break

    def _advance_turn(self):
        """Moves to the next player, handling Cambia turn counting."""
        if self._game_over: return

        if self.cambia_caller_id is not None:
            self.turns_after_cambia += 1

        self.current_player_index = (self.current_player_index + 1) % self.num_players
        logger.debug(f"Turn advances. Current player: {self.current_player_index}")


    def _check_game_end(self):
        """Checks if the game has ended based on Cambia calls or stockpile/discard exhaustion."""
        if self._game_over: return

        # 1. Cambia called and all players had their final turn
        if self.cambia_caller_id is not None:
            if self.turns_after_cambia >= self.num_players :
                logger.info("Game ends: Cambia called and final turns completed.")
                self._game_over = True

        # 2. Stockpile AND Discard pile are exhausted when a draw is required
        # This check is tricky. Should happen if apply_action tries DRAW but fails.
        # If _game_over was set because draw failed, calculation happens below.

        # 3. Any player has zero cards? (Check rules - usually doesn't end game directly)

        if self._game_over:
             self._calculate_final_scores()


    def _calculate_final_scores(self):
        """Calculates final scores and determines the winner."""
        if self._utilities != [0.0] * self.num_players: # Avoid recalculating if already done
             return

        scores = []
        for i in range(self.num_players):
            hand_value = sum(card.value for card in self.players[i].hand)
            scores.append(hand_value)
            logger.info(f"Player {i} final hand: {[str(c) for c in self.players[i].hand]} Score: {hand_value}")

        min_score = min(scores)
        winners = [i for i, score in enumerate(scores) if score == min_score]

        if len(winners) == 1:
            self._winner = winners[0]
            logger.info(f"Player {self._winner} wins with score {min_score}.")
        else: # Tie
            if self.cambia_caller_id is not None and self.cambia_caller_id in winners:
                self._winner = self.cambia_caller_id
                logger.info(f"Player {self._winner} (Cambia caller) wins tie with score {min_score}.")
            else:
                 self._winner = None # Representing a draw
                 logger.info(f"Tie between players {winners} with score {min_score}. No single winner.")

        # Calculate utilities (+1 for win, -1 for loss, 0 for draw/tie)
        if self._winner is not None:
             self._utilities[self._winner] = 1.0
             loser = self.get_opponent_index(self._winner)
             self._utilities[loser] = -1.0
        else: # Draw
             self._utilities = [0.0] * self.num_players

        logger.debug(f"Final Utilities: {self._utilities}")


    def is_terminal(self) -> bool:
        """Returns True if the game has ended."""
        # Check end condition explicitly if needed, e.g., if _check_game_end missed something
        if not self._game_over and not self.stockpile and len(self.discard_pile) <= 1:
             # If we are in a state where no draw is possible and no other action, game must end
             if not self.get_legal_actions(): # Check if any non-draw action exists
                  logger.info("Game ends: No cards left to draw and no other actions possible.")
                  self._game_over = True
                  self._calculate_final_scores() # Calculate scores at this point

        return self._game_over

    def get_utility(self, player_id: int) -> float:
        """Returns the final utility for the specified player."""
        if not self.is_terminal():
             logger.warning("get_utility called on non-terminal state!")
             return 0.0
        if 0 <= player_id < self.num_players:
             # Ensure scores are calculated if somehow missed
             if self._utilities == [0.0] * self.num_players and self._winner is None and self.cambia_caller_id is None:
                  self._calculate_final_scores()
             return self._utilities[player_id]
        raise IndexError("Invalid player index")

    def get_player_turn(self) -> int:
         """Returns the index of the player whose turn it normally is (ignores snap phase)."""
         return self.current_player_index

    def get_acting_player(self) -> int:
         """Returns the index of the player who needs to act *now* (could be snapper)."""
         if self.snap_phase_active:
             if self.snap_current_snapper_idx < len(self.snap_potential_snappers):
                  return self.snap_potential_snappers[self.snap_current_snapper_idx]
             else:
                  return -1 # Error state or phase should have ended
         elif self.pending_action:
             return self.pending_action_player
         else:
             return self.current_player_index


    def clone(self) -> 'CambiaGameState':
        """Creates a deep copy of the game state."""
        # Using deepcopy on the player list which contains dataclasses
        cloned_players = [copy.deepcopy(p) for p in self.players]
        cloned_state = CambiaGameState(
            players=cloned_players,
            stockpile=list(self.stockpile),
            discard_pile=list(self.discard_pile),
            current_player_index=self.current_player_index,
            num_players=self.num_players,
            cambia_caller_id=self.cambia_caller_id,
            turns_after_cambia=self.turns_after_cambia,
            house_rules=copy.deepcopy(self.house_rules), # Deep copy rules dataclass
             _game_over=self._game_over,
             _winner=self._winner,
             _utilities=list(self._utilities),
             pending_action=self.pending_action, # Actions are tuples/dataclasses, shallow copy ok
             pending_action_player=self.pending_action_player,
             pending_action_data=copy.deepcopy(self.pending_action_data), # Deep copy data dict
             snap_phase_active=self.snap_phase_active,
             snap_discarded_card=self.snap_discarded_card, # Card is immutable, shallow ok
             snap_potential_snappers=list(self.snap_potential_snappers),
             snap_current_snapper_idx=self.snap_current_snapper_idx,
        )
        return cloned_state

    def __str__(self) -> str:
        state_desc = ""
        if self.snap_phase_active:
             acting_snapper = self.snap_potential_snappers[self.snap_current_snapper_idx] if self.snap_current_snapper_idx < len(self.snap_potential_snappers) else 'None'
             state_desc = f"SnapPhase(Actor: P{acting_snapper}, Target: {self.snap_discarded_card.rank})"
        elif self.pending_action:
             state_desc = f"Pending(Actor: P{self.pending_action_player}, Action: {type(self.pending_action).__name__})"
        else:
             state_desc = f"Turn: P{self.current_player_index}"

        discard_top_str = str(self.get_discard_top()) if self.discard_pile else "[]"
        return (f"GameState({state_desc}, "
                f"Stock: {len(self.stockpile)}, Discard: {discard_top_str}, "
                f"Hands: {[len(p.hand) for p in self.players]}, "
                f"Cambia: {self.cambia_caller_id}, Over: {self._game_over})")