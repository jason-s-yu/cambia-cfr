# src/game/engine.py
import random
from collections import deque
from typing import Callable, List, Tuple, Optional, Any, Dict, Deque
from dataclasses import dataclass, field
import logging
import copy

# Use relative imports for modules within the 'game' package
from .types import StateDelta, StateDeltaChange, UndoInfo
from .player_state import PlayerState
from .helpers import serialize_card

# --- Import Mixins ---
from ._query_mixin import QueryMixin
from ._snap_mixin import SnapLogicMixin
from ._ability_mixin import AbilityMixin

# Add imports for other mixins if created (e.g., SetupMixin, TurnLogicMixin)

# Use relative imports for modules outside the 'game' package but within 'src'
from ..card import Card, create_standard_deck
from ..constants import (
    # Keep only constants actually used in *this* file after refactoring
    NUM_PLAYERS,
    GameAction,
    ActionDrawStockpile,
    ActionDrawDiscard,
    ActionCallCambia,
    ActionDiscard
)
from ..config import CambiaRulesConfig

logger = logging.getLogger(__name__)


@dataclass
class CambiaGameState(
    QueryMixin, SnapLogicMixin, AbilityMixin
):  # Add mixins to inheritance
    """
    Represents the true, objective state of a 1v1 Cambia game.
    Uses delta-based updates and mixins for modularity.
    """

    # --- Core State Attributes ---
    players: List[PlayerState] = field(default_factory=list)
    stockpile: List[Card] = field(default_factory=list)
    discard_pile: List[Card] = field(default_factory=list)
    current_player_index: int = 0  # Player whose turn it nominally is
    num_players: int = NUM_PLAYERS
    cambia_caller_id: Optional[int] = None
    turns_after_cambia: int = 0
    house_rules: CambiaRulesConfig = field(default_factory=CambiaRulesConfig)
    _game_over: bool = False
    _winner: Optional[int] = None
    _utilities: List[float] = field(default_factory=lambda: [0.0] * NUM_PLAYERS)
    _turn_number: int = 0

    # --- Pending Action State (Managed primarily by AbilityMixin) ---
    pending_action: Optional[GameAction] = None
    pending_action_player: Optional[int] = None
    pending_action_data: Dict[str, Any] = field(default_factory=dict)

    # --- Snap Phase State (Managed primarily by SnapLogicMixin) ---
    snap_phase_active: bool = False
    snap_discarded_card: Optional[Card] = None
    snap_potential_snappers: List[int] = field(default_factory=list)
    snap_current_snapper_idx: int = 0
    snap_results_log: List[Dict[str, Any]] = field(default_factory=list)

    # --- Initialization ---
    def __post_init__(self):
        # If players list is empty, it indicates a new game needs setup
        if not self.players:
            self._setup_game()
        # Initialize utilities based on num_players if needed (though default_factory handles it)
        if len(self._utilities) != self.num_players:
            self._utilities = [0.0] * self.num_players

    def _setup_game(self):
        """Initializes the deck, shuffles, and deals cards. (Could be moved to SetupMixin)."""
        self.stockpile = create_standard_deck(include_jokers=self.house_rules.use_jokers)
        random.shuffle(self.stockpile)
        initial_peek_count = self.house_rules.initial_view_count
        cards_per_player = self.house_rules.cards_per_player
        self.players = [
            PlayerState(initial_peek_indices=tuple(range(initial_peek_count)))
            for _ in range(self.num_players)
        ]

        # Deal cards
        for _ in range(cards_per_player):
            for i in range(self.num_players):
                if self.stockpile:
                    # Basic validation during setup
                    if (
                        i < len(self.players)
                        and hasattr(self.players[i], "hand")
                        and isinstance(self.players[i].hand, list)
                    ):
                        self.players[i].hand.append(self.stockpile.pop())
                    else:
                        logger.error(
                            "Error during setup: Player object %s invalid or missing hand list.",
                            i,
                        )
                        raise RuntimeError(
                            f"Game setup failed due to invalid PlayerState {i}"
                        )
                else:
                    logger.warning("Stockpile empty during initial deal!")
                    break  # Stop dealing if stockpile runs out

        self.discard_pile = []  # Start with empty discard
        self.current_player_index = random.randint(
            0, self.num_players - 1
        )  # Random start player
        # Reset all other state variables explicitly
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
        self.snap_discarded_card = None
        self.snap_potential_snappers = []
        self.snap_current_snapper_idx = 0
        self.snap_results_log = []

        logger.debug(
            "Game setup complete. Player %s starts (Turn %s).",
            self.current_player_index,
            self._turn_number,
        )
        logger.debug("House Rules: %s", self.house_rules)

    # --- Core Action Application Logic ---

    def apply_action(self, action: GameAction) -> Tuple[StateDelta, UndoInfo]:
        """
        Applies the given action by modifying 'self' and returns a StateDelta list
        and a callable UndoInfo. Delegates logic to mixins based on game phase.
        """
        if self._game_over:
            logger.warning("Attempted action on a finished game.")
            return [], lambda: None  # No change, no undo

        acting_player = self.get_acting_player()  # Method from QueryMixin

        # Basic validation: Ensure the acting player is valid before proceeding
        if acting_player == -1:
            logger.error(
                "Apply Action: Cannot determine valid acting player. State: %s", self
            )
            # This might indicate an inconsistent state. Avoid further changes.
            # Consider setting game_over=True here? For now, just return no-op.
            return [], lambda: None
        if not (
            0 <= acting_player < len(self.players)
            and hasattr(self.players[acting_player], "hand")
        ):
            logger.error(
                "Apply Action: Invalid acting player P%s or missing hand. State: %s",
                acting_player,
                self,
            )
            # This is a critical error state, potentially end the game.
            # self._game_over = True # Needs undo handling if done here
            # self._calculate_final_scores()
            return [], lambda: None

        # --- Undo Stack & Delta List for this action ---
        undo_stack: Deque[Callable[[], None]] = deque()
        delta_list: StateDelta = []

        # --- Master Undo Function ---
        def undo_action():
            # Execute undo operations in reverse order of changes (LIFO)
            while undo_stack:
                try:
                    undo_func = (
                        undo_stack.popleft()
                    )  # Pop from front (where _add_change prepends)
                    undo_func()
                except Exception as e:
                    logger.exception(
                        "Error during undo operation %s: %s. State might be inconsistent.", undo_func, e
                    )
                    # Potentially raise or log more severely

        # --- Action Application Logic (Delegates to Mixins) ---
        try:
            card_discarded_this_step: Optional[Card] = None
            action_processed = False
            turn_should_advance_after_action = False  # Flag to control turn advancement

            # 1. Handle Snap Phase Actions (Uses SnapLogicMixin)
            if self.snap_phase_active:
                # _handle_snap_action performs state changes via _add_change
                action_processed = self._handle_snap_action(
                    action, acting_player, undo_stack, delta_list
                )
                # Snap phase handles its own advancement or termination. Turn advances when phase ends.
                # If _handle_snap_action returns True, it means the action was valid for the phase.
                # If it returns False, it means the action was ignored (e.g., wrong player).

            # 2. Handle Pending Action Resolution (Uses AbilityMixin)
            elif self.pending_action:
                if acting_player == self.pending_action_player:
                    # _handle_pending_action performs state changes via _add_change
                    # It returns the card discarded during this step, if any (for snap check).
                    card_discarded_this_step = self._handle_pending_action(
                        action, acting_player, undo_stack, delta_list
                    )
                    if card_discarded_this_step is not None:
                        # Action successfully resolved the pending state (e.g., discard/replace, ability choice)
                        action_processed = True
                        # If the pending action is now clear AND snap isn't active, the turn might advance.
                        if not self.pending_action and not self.snap_phase_active:
                            # Check for snap initiation *before* advancing turn
                            snap_started = self._initiate_snap_phase(
                                card_discarded_this_step, undo_stack, delta_list
                            )
                            if not snap_started:
                                # If snap didn't start, the main turn can advance
                                turn_should_advance_after_action = True
                        # If a new pending action was set (e.g., King Look -> King Swap), turn doesn't advance yet.
                        # If snap phase started, turn doesn't advance yet.
                    elif action_processed is False and self.pending_action:
                        # _handle_pending_action returning None AND pending_action still set
                        # implies the submitted `action` was invalid for the *current* pending state.
                        # Do nothing, wait for a valid action.
                        logger.warning(
                            "Invalid action %s for pending state %s. Waiting.", action, self.pending_action
                        )
                        action_processed = False  # Mark as not successfully processed
                    else:
                        # _handle_pending_action returned None, but pending state *was* cleared (e.g. error case)
                        action_processed = True  # Action led to clearing pending state
                        turn_should_advance_after_action = (
                            True  # Turn potentially advances if error cleared state
                        )

                else:  # Wrong player for pending action (should be caught earlier by get_acting_player ideally)
                    logger.error(
                        "Action %s from P%d received, but pending action requires P%d", action, acting_player, self.pending_action_player
                    )
                    action_processed = False  # Action ignored

            # 3. Handle Standard Start-of-Turn Actions
            else:
                if acting_player != self.current_player_index:
                    # Should not happen if logic is correct
                    logger.error(
                        "Standard action %s from P%d, but expected P%d", action, acting_player, self.current_player_index
                    )
                    action_processed = False
                else:
                    player = self.current_player_index
                    if isinstance(action, ActionDrawStockpile):
                        drawn_card: Optional[Card] = None
                        # reshuffled = False
                        if not self.stockpile:
                            # _attempt_reshuffle adds its own undo ops
                            reshuffle_deltas = self._attempt_reshuffle(undo_stack)
                            if reshuffle_deltas:
                                reshuffled = True
                                delta_list.extend(reshuffle_deltas)
                                # Reshuffle itself doesn't discard a card for snap check
                        if self.stockpile:
                            # --- State Change: Draw + Set Pending Discard/Replace ---
                            original_pending = (
                                self.pending_action,
                                self.pending_action_player,
                                copy.deepcopy(self.pending_action_data),
                            )
                            drawn_card_for_change = self.stockpile[
                                -1
                            ]  # Card that will be drawn

                            def change_draw_stock():
                                nonlocal drawn_card
                                card = self.stockpile.pop()
                                drawn_card = card
                                logger.debug(
                                    "P%d drew %s from stockpile.", player, drawn_card
                                )
                                # Set pending action for Discard/Replace choice
                                self.pending_action = ActionDiscard(
                                    use_ability=False
                                )  # Placeholder type
                                self.pending_action_player = player
                                self.pending_action_data = {"drawn_card": drawn_card}

                            def undo_draw_stock():
                                # Restore pending state first
                                drawn_card_in_pending = (
                                    self.pending_action_data.get("drawn_card")
                                    if self.pending_action_player == player
                                    else None
                                )
                                (
                                    self.pending_action,
                                    self.pending_action_player,
                                    self.pending_action_data,
                                ) = original_pending
                                # Put card back *if* it was the one recorded in pending state
                                if drawn_card_in_pending:
                                    self.stockpile.append(drawn_card_in_pending)

                            # Log draw itself and the pending state change
                            delta_draw = (
                                "draw_stockpile",
                                player,
                                serialize_card(drawn_card_for_change),
                            )
                            delta_pending = (
                                "set_pending_action",
                                "ActionDiscard",
                                player,
                                {"drawn_card": serialize_card(drawn_card_for_change)},
                                (
                                    type(original_pending[0]).__name__
                                    if original_pending[0]
                                    else None
                                ),
                                original_pending[1],
                                original_pending[2],
                            )
                            self._add_change(
                                change_draw_stock,
                                undo_draw_stock,
                                delta_draw,
                                undo_stack,
                                delta_list,
                            )
                            delta_list.append(
                                delta_pending
                            )  # Log pending state change separately
                            # --- End State Change ---
                            action_processed = True
                            # DO NOT ADVANCE TURN YET - Wait for Discard/Replace decision
                        else:  # Stockpile empty even after reshuffle attempt
                            logger.warning(
                                "P%d tried DRAW_STOCKPILE, but stockpile/discard empty. Game should end.", player
                            )
                            # Game end check will handle this after turn fails to advance
                            action_processed = False  # Action failed

                    elif isinstance(action, ActionDrawDiscard):
                        if (
                            self.house_rules.allowDrawFromDiscardPile
                            and self.discard_pile
                        ):
                            # --- State Change: Draw Disc + Set Pending Discard/Replace ---
                            original_pending = (
                                self.pending_action,
                                self.pending_action_player,
                                copy.deepcopy(self.pending_action_data),
                            )
                            drawn_card_for_change = self.discard_pile[-1]

                            def change_draw_discard():
                                nonlocal drawn_card
                                card = self.discard_pile.pop()
                                drawn_card = card
                                logger.debug(
                                    "P%d drew %s from discard pile.", player, drawn_card
                                )
                                self.pending_action = ActionDiscard(use_ability=False)
                                self.pending_action_player = player
                                self.pending_action_data = {"drawn_card": drawn_card}

                            def undo_draw_discard():
                                drawn_card_in_pending = (
                                    self.pending_action_data.get("drawn_card")
                                    if self.pending_action_player == player
                                    else None
                                )
                                (
                                    self.pending_action,
                                    self.pending_action_player,
                                    self.pending_action_data,
                                ) = original_pending
                                if drawn_card_in_pending:
                                    self.discard_pile.append(drawn_card_in_pending)

                            delta_draw = (
                                "draw_discard",
                                player,
                                serialize_card(drawn_card_for_change),
                            )
                            delta_pending = (
                                "set_pending_action",
                                "ActionDiscard",
                                player,
                                {"drawn_card": serialize_card(drawn_card_for_change)},
                                (
                                    type(original_pending[0]).__name__
                                    if original_pending[0]
                                    else None
                                ),
                                original_pending[1],
                                original_pending[2],
                            )
                            self._add_change(
                                change_draw_discard,
                                undo_draw_discard,
                                delta_draw,
                                undo_stack,
                                delta_list,
                            )
                            delta_list.append(delta_pending)
                            # --- End State Change ---
                            action_processed = True
                            # DO NOT ADVANCE TURN YET
                        else:
                            logger.error(
                                "Invalid Action: DRAW_DISCARD attempted when not allowed or pile empty."
                            )
                            action_processed = False  # Action invalid

                    elif isinstance(action, ActionCallCambia):
                        cambia_allowed_round = self.house_rules.cambia_allowed_round
                        current_round = self._turn_number // self.num_players
                        if (
                            self.cambia_caller_id is None
                            and current_round >= cambia_allowed_round
                        ):
                            logger.info("P%d calls Cambia!", player)
                            # --- State Change: Set Cambia Caller ---
                            original_cambia_caller = self.cambia_caller_id
                            original_turns_after = self.turns_after_cambia

                            def change_cambia():
                                self.cambia_caller_id = player
                                self.turns_after_cambia = 0

                            def undo_cambia():
                                self.cambia_caller_id = original_cambia_caller
                                self.turns_after_cambia = original_turns_after

                            delta_caller = (
                                "set_attr",
                                "cambia_caller_id",
                                player,
                                original_cambia_caller,
                            )
                            delta_turns = (
                                "set_attr",
                                "turns_after_cambia",
                                0,
                                original_turns_after,
                            )
                            self._add_change(
                                change_cambia,
                                undo_cambia,
                                delta_caller,
                                undo_stack,
                                delta_list,
                            )
                            delta_list.append(delta_turns)
                            # --- End State Change ---
                            action_processed = True
                            turn_should_advance_after_action = (
                                True  # Turn advances immediately after calling Cambia
                            )
                        else:
                            logger.warning(
                                "P%d tried invalid CALL_CAMBIA (Already called or too early).", player
                            )
                            action_processed = False  # Action invalid

                    else:  # An action type not handled at the start of a turn
                        logger.warning(
                            "Unhandled action type %s received at start of turn for P%d.", type(action), player
                        )
                        action_processed = False

            # --- Post-Action Processing ---
            if action_processed and turn_should_advance_after_action:
                if (
                    not self.snap_phase_active and not self.pending_action
                ):  # Double check state allows turn advance
                    self._advance_turn(undo_stack, delta_list)
                else:
                    logger.debug(
                        "Turn advancement skipped due to active snap/pending action."
                    )

            # Final game end check after all processing for this action is done
            # (Note: _advance_turn also calls this, but check here too for cases where turn doesn't advance)
            if not self._game_over:  # Avoid redundant checks/calculations
                self._check_game_end(undo_stack, delta_list)

        except Exception:
            logger.exception(
                "Critical error during apply_action logic for action %s. State: %s. Attempting rollback.", action, self
            )
            # Attempt to execute the master undo function
            try:
                undo_action()
                logger.info(
                    "Successfully executed master undo after apply_action exception."
                )
            except Exception as undo_e:
                # If undo fails, the state is likely corrupted.
                logger.error(
                    "Exception during master undo after apply_action error: %s. Game state may be inconsistent!", undo_e
                )
                # Consider forcing game over?
                # self._game_over = True
            # Return empty delta and no-op undo as the action failed critically
            return [], lambda: None

        # Return the collected deltas and the master undo function for this action
        return delta_list, undo_action

    # --- Undo/Delta Helper (Could be in a separate mixin) ---
    def _add_change(
        self,
        change_func: Callable[[], Any],
        undo_func: Callable[[], None],
        delta: StateDeltaChange,
        undo_stack: Deque,
        delta_list: StateDelta,
    ):
        """Applies a change, adds its undo operation to the stack, and records the delta."""
        try:
            change_func()  # Apply the state modification
            delta_list.append(delta)  # Record the change event
            undo_stack.appendleft(
                undo_func
            )  # Add undo op to the *front* (for LIFO execution)
        except Exception as e:
            logger.exception(
                "Error applying change function %s for delta %s: %s", change_func.__name__, delta, e
            )
            # Depending on severity, might want to raise or handle differently

    # --- Reshuffle Logic (Could be in DeckMixin or TurnLogicMixin) ---
    def _attempt_reshuffle(self, undo_stack_outer: Deque) -> Optional[StateDelta]:
        """
        Reshuffles discard pile (except top card) into stockpile if stockpile is empty.
        Adds its *own* undo operation to the provided undo_stack_outer.
        Returns the StateDelta for the reshuffle, or None if no reshuffle occurred.
        """
        if not self.stockpile and len(self.discard_pile) > 1:
            logger.info("Stockpile empty. Reshuffling discard pile.")
            # --- Capture state BEFORE changes for undo ---
            original_stockpile = list(self.stockpile)  # Should be empty
            original_discard = list(self.discard_pile)  # Keep full original order
            top_card = self.discard_pile[-1]
            cards_to_shuffle = self.discard_pile[:-1]
            # Make a copy to shuffle for the new state, preserve original order for undo
            new_stockpile = list(cards_to_shuffle)
            random.shuffle(new_stockpile)

            # --- Define change and undo ---
            def change_reshuffle():
                self.discard_pile = [top_card]
                self.stockpile = new_stockpile
                logger.info("Reshuffled %d cards into stockpile.", len(self.stockpile))

            def undo_reshuffle():
                self.stockpile = original_stockpile  # Restore empty stockpile
                self.discard_pile = original_discard  # Restore original discard pile
                logger.debug("Undo reshuffle.")

            # --- Create delta and add undo to outer stack ---
            shuffled_card_strs = [serialize_card(c) for c in new_stockpile]
            delta_reshuffle: StateDelta = [
                ("reshuffle", shuffled_card_strs, serialize_card(top_card))
            ]
            # Manually apply change *now*
            change_reshuffle()
            # Add the specific undo for *this reshuffle* to the caller's undo stack
            undo_stack_outer.appendleft(undo_reshuffle)
            return delta_reshuffle  # Return the delta event

        elif not self.stockpile:
            logger.info("Stockpile empty, cannot reshuffle discard pile (size <= 1).")
            return None  # No reshuffle happened
        else:
            # Stockpile not empty, no reshuffle needed
            return None

    # --- Penalty Logic (Could be in TurnLogicMixin) ---
    def _apply_penalty(
        self, player_index: int, num_cards: int, undo_stack_main: Deque
    ) -> StateDelta:
        """
        Applies penalty draw(s) to a player, handling reshuffles if needed.
        Adds ONE combined undo operation for the whole penalty sequence to undo_stack_main.
        Returns a list of StateDeltaChanges for the penalty draws/reshuffles.
        """
        logger.warning(
            "Applying penalty: Player %d attempts to draw %d cards.", player_index, num_cards
        )
        penalty_deltas: StateDelta = []
        if not (
            0 <= player_index < len(self.players)
            and hasattr(self.players[player_index], "hand")
        ):
            logger.error(
                "Cannot apply penalty: Player %d invalid or missing hand.", player_index
            )
            return penalty_deltas  # Return empty delta list

        # --- Capture state BEFORE any changes for the single master undo ---
        original_hand_state = list(self.players[player_index].hand)
        original_stockpile_state = list(self.stockpile)
        original_discard_state = list(self.discard_pile)
        cards_actually_drawn_this_penalty: List[Card] = []  # Track for logging/debugging

        # --- Perform draws and potential reshuffles ---
        for i in range(num_cards):
            if not self.stockpile:
                # Try to reshuffle. _attempt_reshuffle adds its own undo op to undo_stack_main.
                reshuffle_outcome_deltas = self._attempt_reshuffle(undo_stack_main)
                if reshuffle_outcome_deltas:
                    penalty_deltas.extend(
                        reshuffle_outcome_deltas
                    )  # Add reshuffle delta(s)
                    logger.debug("Reshuffled during penalty draw %d/%d", i+1, num_cards)
                else:  # Cannot reshuffle
                    logger.warning(
                        "Stockpile/discard empty during penalty draw %d/%d for P%d. Cannot draw more.", i+1, num_cards, player_index
                    )
                    break  # Stop drawing penalty cards

            if self.stockpile:
                # --- Apply state change directly ---
                drawn_card = self.stockpile.pop()
                self.players[player_index].hand.append(drawn_card)
                cards_actually_drawn_this_penalty.append(drawn_card)
                # --- Record delta ---
                delta = ("penalty_draw", player_index, serialize_card(drawn_card))
                penalty_deltas.append(delta)
                # logger.debug(f"Player {player_index} penalty draw {i+1}: {drawn_card}") # Potentially verbose
            else:
                # This case should theoretically not be reached if reshuffle logic is correct
                logger.error(
                    "Stockpile empty immediately after attempting reshuffle in penalty draw. Stopping."
                )
                break

        logger.info(
            "Player %d drew %d cards as penalty.", player_index, len(cards_actually_drawn_this_penalty)
        )

        # --- Create the single master undo function for the *entire penalty draw sequence* ---
        def undo_penalty_sequence():
            # Restore the state to exactly how it was before this function was called.
            # This undo runs *before* any reshuffle undo ops added by _attempt_reshuffle within this call.
            self.players[player_index].hand = original_hand_state
            self.stockpile = original_stockpile_state
            self.discard_pile = original_discard_state
            logger.debug(
                "Undo penalty sequence applied for P%d. State restored to pre-penalty.", player_index
            )

        # Add the master undo function to the *front* of the main stack
        undo_stack_main.appendleft(undo_penalty_sequence)

        return penalty_deltas

    # --- Turn Advancement & Game End Logic (Could be in TurnLogicMixin) ---
    def _advance_turn(self, undo_stack: Deque, delta_list: StateDelta):
        """Advances to the next player, updates turn counts, checks for game end."""
        if self._game_over:
            logger.debug("Attempted to advance turn, but game is already over.")
            return

        # --- Calculate next state ---
        original_turn = self._turn_number
        original_player = self.current_player_index
        original_cambia_turns = self.turns_after_cambia
        logger.debug("Advancing turn from T#%d P%d", original_turn, original_player)

        next_player = (self.current_player_index + 1) % self.num_players
        next_turn = self._turn_number + 1
        next_cambia_turns = self.turns_after_cambia
        # Increment Cambia turn counter *after* each player completes their turn during the final round
        if self.cambia_caller_id is not None:
            next_cambia_turns += 1  # Increment after the current player's turn finishes

        # --- State Change: Update turn, player, cambia count ---
        def change_advance():
            self.current_player_index = next_player
            self._turn_number = next_turn
            self.turns_after_cambia = (
                next_cambia_turns  # Update based on calculation above
            )

        def undo_advance():
            self._turn_number = original_turn
            self.current_player_index = original_player
            self.turns_after_cambia = original_cambia_turns
            logger.debug(
                "Undo advance turn. Back to T#%d, P%d", original_turn, original_player
            )

        # Log changes
        delta_player = ("set_attr", "current_player_index", next_player, original_player)
        delta_turn = ("set_attr", "_turn_number", next_turn, original_turn)
        delta_cambia = (
            "set_attr",
            "turns_after_cambia",
            next_cambia_turns,
            original_cambia_turns,
        )
        self._add_change(
            change_advance, undo_advance, delta_player, undo_stack, delta_list
        )
        delta_list.append(delta_turn)
        if self.cambia_caller_id is not None:  # Only log cambia turn change if relevant
            delta_list.append(delta_cambia)
        # --- End State Change ---

        logger.debug(
            "Advanced turn to T#%d P%d. Cambia turns: %d", self._turn_number, self.current_player_index, self.turns_after_cambia
        )

        # Check game end conditions *after* advancing the turn
        self._check_game_end(undo_stack, delta_list)

    def _check_game_end(self, undo_stack: Deque, delta_list: StateDelta):
        """Checks game end conditions and updates state if game has ended."""
        if self._game_over:
            return  # Already ended

        end_condition_met = False
        reason = ""

        # 1. Max Turns Reached
        max_turns = self.house_rules.max_game_turns
        if max_turns > 0 and self._turn_number >= max_turns:
            end_condition_met = True
            reason = f"Max game turns ({max_turns}) reached"

        # 2. Cambia Round Completed
        # Game ends when turns_after_cambia *reaches* num_players (meaning all players had their final turn)
        if (
            not end_condition_met
            and self.cambia_caller_id is not None
            and self.turns_after_cambia >= self.num_players
        ):
            end_condition_met = True
            reason = f"Cambia final turns ({self.turns_after_cambia}/{self.num_players}) completed"

        # 3. Stalemate: Current player has no actions AND cannot draw/reshuffle
        # Check only if it's a standard turn start (no pending/snap)
        if (
            not end_condition_met
            and not self.pending_action
            and not self.snap_phase_active
        ):
            player = self.current_player_index
            # Validate player state before checking actions
            if 0 <= player < len(self.players) and hasattr(self.players[player], "hand"):
                legal_actions = self.get_legal_actions()  # Use the QueryMixin method
                can_draw_stockpile = bool(self.stockpile)
                can_reshuffle = len(self.discard_pile) > 1
                # If no actions are possible *at all* and they can't draw/reshuffle, it's an end state.
                if not legal_actions and not can_draw_stockpile and not can_reshuffle:
                    end_condition_met = True
                    reason = f"Stalemate: P{player} has no actions and cannot draw/reshuffle (Stock: {len(self.stockpile)}, Disc: {len(self.discard_pile)})"
            else:
                # This indicates a potential inconsistency if reached
                end_condition_met = True
                reason = (
                    f"Game end check (Stalemate): Player {player} invalid/missing hand."
                )

        # If an end condition is met, finalize the game state
        if end_condition_met and not self._game_over:  # Ensure we only trigger this once
            logger.info("Game ends: %s.", reason)
            # Calculate scores but don't set attributes directly yet for undo
            temp_winner, temp_utilities = self._calculate_final_scores(
                set_attributes=False
            )

            # --- State Change: Mark Game Over & Set Scores ---
            original_game_over = self._game_over  # Should be False
            original_winner = self._winner
            original_utilities = list(self._utilities)

            def change_game_end():
                self._game_over = True
                # Re-calculate and set attributes *now* within the change function
                # This ensures the final state reflects the calculation at the time of ending.
                self._calculate_final_scores(set_attributes=True)

            def undo_game_end():
                self._game_over = original_game_over
                self._winner = original_winner
                self._utilities = original_utilities  # Restore previous scores/utilities
                logger.debug("Undo game end.")

            # Log the end event with the calculated outcome
            delta_game_end = ("game_end", reason, temp_winner, temp_utilities)
            self._add_change(
                change_game_end, undo_game_end, delta_game_end, undo_stack, delta_list
            )
            # --- End State Change ---

    def _calculate_final_scores(
        self, set_attributes=True
    ) -> Tuple[Optional[int], List[float]]:
        """
        Calculates final scores based on hand values and determines the winner/utilities.
        Optionally sets the _winner and _utilities attributes on self.
        (Could be moved to TurnLogicMixin or EndgameMixin)
        """
        # If scores were already calculated and set (e.g., by a previous call in get_utility)
        initial_utilities = [0.0] * self.num_players
        if set_attributes and self._utilities != initial_utilities:
            # If setting attributes and utilities are not default, assume already calculated.
            # This check prevents overwriting tie results (where winner is None but utilities are set).
            logger.debug("Scores seem already calculated, returning stored values.")
            return self._winner, self._utilities

        scores = []
        final_hands_str = []  # For logging
        for i in range(self.num_players):
            if 0 <= i < len(self.players) and hasattr(self.players[i], "hand"):
                current_hand = self.players[i].hand
                # Basic validation of hand contents before scoring
                if not all(isinstance(card, Card) for card in current_hand):
                    logger.error(
                        "Calculate score: Player %d's hand contains non-Card objects: %s. Assigning max score.", i, current_hand
                    )
                    scores.append(float("inf"))  # Assign effectively infinite score
                    final_hands_str.append(
                        [str(c) for c in current_hand]
                    )  # Log representation
                else:
                    hand_value = sum(card.value for card in current_hand)
                    scores.append(hand_value)
                    final_hands_str.append([serialize_card(c) for c in current_hand])
            else:
                logger.error(
                    "Calculate score: Player %d invalid or missing hand. Assigning max score.", i
                )
                scores.append(float("inf"))
                final_hands_str.append(["ERROR"])

        winner_calculated: Optional[int] = None
        utilities_calculated: List[float] = [0.0] * self.num_players  # Default to 0

        if not scores:
            logger.error("Cannot calculate final scores: No player scores available.")
            # Keep default utilities [0.0, 0.0] and winner None
        else:
            min_score = min(scores)
            # Check if anyone actually achieved the minimum score (handles all inf case)
            if min_score == float("inf"):
                logger.warning(
                    "All players have invalid hands or max score. Declaring a tie with 0 utility."
                )
                winner_calculated = None
                utilities_calculated = [0.0] * self.num_players  # All tie with 0
            else:
                winners = [i for i, score in enumerate(scores) if score == min_score]

                if len(winners) == 1:
                    winner_calculated = winners[0]
                    utilities_calculated = [-1.0] * self.num_players
                    utilities_calculated[winner_calculated] = 1.0
                elif len(winners) > 1:  # Tie situation
                    # Check if Cambia caller breaks the tie among those tied
                    if (
                        self.cambia_caller_id is not None
                        and self.cambia_caller_id in winners
                    ):
                        winner_calculated = self.cambia_caller_id
                        utilities_calculated = [-1.0] * self.num_players
                        utilities_calculated[winner_calculated] = 1.0
                        logger.info(
                            "Tie score (%d) broken by Cambia caller P%d.", min_score, winner_calculated
                        )
                    else:
                        # True tie among winners list
                        logger.info(
                            "True tie between players %s with score %d.", winners, min_score
                        )
                        winner_calculated = None  # No single winner
                        # Assign utilities: 0 for tied players, -1 for losers
                        utilities_calculated = [
                            -1.0
                        ] * self.num_players  # Start all as losers
                        for p_idx in winners:
                            utilities_calculated[p_idx] = (
                                0.0  # Tied players get 0 utility
                            )
                else:  # Should not happen if scores exist and min_score isn't inf
                    logger.error(
                        "Score calculation error: No minimum score winners found, but min score wasn't infinity."
                    )
                    winner_calculated = None
                    utilities_calculated = [
                        0.0
                    ] * self.num_players  # Default to 0 utility tie

        # Log the final result only when setting attributes
        if set_attributes:
            log_msg = "Game Score Calc: "
            if winner_calculated is not None:
                log_msg += f"Player {winner_calculated} wins with score {scores[winner_calculated]}. "
            else:  # Tie
                tied_players = [
                    i for i, util in enumerate(utilities_calculated) if util == 0.0
                ]
                log_msg += f"Tie between players {tied_players} (Score: {min_score}). "
            log_msg += (
                f"Utilities: {utilities_calculated}. Final Hands: {final_hands_str}"
            )
            logger.info(log_msg)

            # Set the instance attributes
            self._winner = winner_calculated
            self._utilities = utilities_calculated
            logger.debug(
                "Final winner/utilities set: W=%d, U=%s", self._winner, self._utilities
            )

        return winner_calculated, utilities_calculated
