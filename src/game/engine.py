"""src/game/engine.py"""

import random
from collections import deque
from typing import Callable, List, Tuple, Optional, Any, Dict, Deque
from dataclasses import dataclass, field
import logging
import copy

from .types import StateDelta, StateDeltaChange, UndoInfo
from .player_state import PlayerState
from .helpers import serialize_card
from ._query_mixin import QueryMixin
from ._snap_mixin import SnapLogicMixin
from ._ability_mixin import AbilityMixin
from ..card import Card, create_standard_deck
from ..constants import (
    NUM_PLAYERS,
    GameAction,
    ActionDrawStockpile,
    ActionDrawDiscard,
    ActionCallCambia,
    ActionDiscard,
)
from ..config import CambiaRulesConfig

logger = logging.getLogger(__name__)


@dataclass
class CambiaGameState(QueryMixin, SnapLogicMixin, AbilityMixin):
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
        if not self.players:
            self._setup_game()
        if len(self._utilities) != self.num_players:
            self._utilities = [0.0] * self.num_players

    def _setup_game(self):
        """Initializes the deck, shuffles, and deals cards."""
        logger.debug("Setting up new game...")
        try:
            self.stockpile = create_standard_deck(
                include_jokers=self.house_rules.use_jokers
            )
            random.shuffle(self.stockpile)
        except Exception as e_deck:
            logger.critical("Failed to create or shuffle deck: %s", e_deck, exc_info=True)
            raise RuntimeError("Deck creation/shuffle failed") from e_deck

        initial_peek_count = self.house_rules.initial_view_count
        cards_per_player = self.house_rules.cards_per_player
        self.players = []  # Reset players list
        try:
            for i in range(self.num_players):
                self.players.append(
                    PlayerState(initial_peek_indices=tuple(range(initial_peek_count)))
                )
        except Exception as e_player_state:
            logger.critical(
                "Failed to initialize PlayerState: %s", e_player_state, exc_info=True
            )
            raise RuntimeError("PlayerState initialization failed") from e_player_state

        # Deal cards
        try:
            for card_num in range(cards_per_player):
                for i in range(self.num_players):
                    if not self.stockpile:
                        logger.warning(
                            "Stockpile empty during initial deal (Card %d, Player %d)!",
                            card_num + 1,
                            i,
                        )
                        raise RuntimeError(
                            "Stockpile ran out during initial deal"
                        )  # Fail fast if deal incomplete
                    # Basic validation during setup
                    if not (
                        0 <= i < len(self.players)
                        and hasattr(self.players[i], "hand")
                        and isinstance(self.players[i].hand, list)
                    ):
                        logger.error(
                            "Invalid PlayerState object %d detected during deal.", i
                        )
                        raise RuntimeError(f"Invalid PlayerState {i} during deal")
                    self.players[i].hand.append(self.stockpile.pop())
        except Exception as e_deal:
            logger.critical("Error during card dealing: %s", e_deal, exc_info=True)
            raise RuntimeError("Card dealing failed") from e_deal

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
        self.snap_discarded_card = None
        self.snap_potential_snappers = []
        self.snap_current_snapper_idx = 0
        self.snap_results_log = []

        logger.debug(
            "Game setup complete. Player %d starts (Turn %d). House Rules: %s",
            self.current_player_index,
            self._turn_number,
            self.house_rules,
        )

    # --- Core Action Application Logic ---

    def apply_action(self, action: GameAction) -> Tuple[StateDelta, UndoInfo]:
        """
        Applies the given action, modifying 'self' and returning deltas and undo info.
        Delegates logic to mixins based on game phase.
        """
        if self._game_over:
            logger.warning("Attempted action %s on a finished game.", action)
            return [], lambda: None

        acting_player = self.get_acting_player()  # Method from QueryMixin
        if acting_player == -1:
            logger.error(
                "Apply Action: Cannot determine valid acting player for action %s. State: %s",
                action,
                self,
            )
            return [], lambda: None
        if not (
            0 <= acting_player < len(self.players)
            and hasattr(self.players[acting_player], "hand")
        ):
            logger.error(
                "Apply Action: Invalid acting player P%d or missing hand for action %s. State: %s",
                acting_player,
                action,
                self,
            )
            return [], lambda: None

        undo_stack: Deque[Callable[[], None]] = deque()
        delta_list: StateDelta = []

        # Master Undo Function
        def undo_action():
            while undo_stack:
                try:
                    undo_func = undo_stack.popleft()
                    undo_func()
                except Exception as e_undo:
                    logger.exception(
                        "Error during undo operation '%s': %s. State might be inconsistent.",
                        getattr(undo_func, "__name__", repr(undo_func)),
                        e_undo,
                    )
                    # Stop further undos on error? For now, let it continue trying.

        # Action Application Logic
        try:
            card_discarded_this_step: Optional[Card] = None
            action_processed = False
            turn_should_advance_after_action = False  # Flag to control turn advancement

            # 1. Handle Snap Phase Actions
            if self.snap_phase_active:
                if not hasattr(self, "_handle_snap_action"):  # Safety check for mixin
                    logger.error("Snap phase active, but _handle_snap_action not found.")
                    raise AttributeError("_handle_snap_action missing from class")
                action_processed = self._handle_snap_action(
                    action, acting_player, undo_stack, delta_list
                )
                # Snap logic handles advancement or phase end internally. Turn advances when phase ends.

            # 2. Handle Pending Action Resolution
            elif self.pending_action:
                if acting_player == self.pending_action_player:
                    if not hasattr(
                        self, "_handle_pending_action"
                    ):  # Safety check for mixin
                        logger.error(
                            "Pending action exists, but _handle_pending_action not found."
                        )
                        raise AttributeError("_handle_pending_action missing from class")

                    card_discarded_this_step = self._handle_pending_action(
                        action, acting_player, undo_stack, delta_list
                    )

                    if (
                        card_discarded_this_step is not None
                    ):  # Successfully resolved pending state (even if ability fizzled)
                        action_processed = True
                        # Check if snap phase started *as a result* of this action resolving
                        # The check is typically done *after* clearing the pending action
                        snap_started_after_resolve = False
                        if (
                            not self.pending_action and not self.snap_phase_active
                        ):  # If pending action cleared and snap not active
                            if hasattr(self, "_initiate_snap_phase"):
                                snap_started_after_resolve = self._initiate_snap_phase(
                                    card_discarded_this_step, undo_stack, delta_list
                                )
                            else:
                                logger.error("_initiate_snap_phase method missing.")

                        # Turn advances only if pending action is resolved AND snap did not start
                        if not self.pending_action and not self.snap_phase_active:
                            turn_should_advance_after_action = True
                        # If a new pending action was set (e.g., King Look -> King Swap), turn doesn't advance.
                        # If snap phase started, turn doesn't advance.

                    elif (
                        self.pending_action
                    ):  # _handle_pending_action returned None, but pending action still exists
                        logger.warning(
                            "Invalid action '%s' for pending state '%s'. Waiting.",
                            action,
                            self.pending_action,
                        )
                        action_processed = False  # Mark as not successfully processed, wait for valid action
                    else:  # _handle_pending_action returned None AND cleared pending state (e.g., error within handler)
                        logger.warning(
                            "Pending action handler for '%s' returned None but cleared state. Treating as processed.",
                            action,
                        )
                        action_processed = True  # State changed (pending cleared)
                        turn_should_advance_after_action = (
                            True  # Potentially advance turn after error clear
                        )

                else:
                    logger.error(
                        "Action '%s' from P%d received, but pending action requires P%d",
                        action,
                        acting_player,
                        self.pending_action_player,
                    )
                    action_processed = False

            # 3. Handle Standard Start-of-Turn Actions
            else:
                if acting_player != self.current_player_index:
                    logger.error(
                        "Standard action '%s' from P%d, but expected P%d",
                        action,
                        acting_player,
                        self.current_player_index,
                    )
                    action_processed = False
                else:
                    player = self.current_player_index
                    if isinstance(action, ActionDrawStockpile):
                        drawn_card: Optional[Card] = None
                        if not self.stockpile:
                            reshuffle_deltas = self._attempt_reshuffle(undo_stack)
                            if reshuffle_deltas:
                                delta_list.extend(reshuffle_deltas)

                        if self.stockpile:
                            original_pending = (
                                self.pending_action,
                                self.pending_action_player,
                                copy.deepcopy(self.pending_action_data),
                            )
                            drawn_card_for_change = self.stockpile[-1]

                            def change_draw_stock():
                                nonlocal drawn_card
                                card = self.stockpile.pop()
                                drawn_card = card
                                logger.debug(
                                    "P%d drew %s from stockpile.", player, drawn_card
                                )
                                self.pending_action = ActionDiscard(
                                    use_ability=False
                                )  # Placeholder type
                                self.pending_action_player = player
                                self.pending_action_data = {"drawn_card": drawn_card}

                            def undo_draw_stock():
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
                                    self.stockpile.append(drawn_card_in_pending)

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
                            action_processed = True
                            # Turn advances after discard/replace decision

                        else:  # Stockpile empty even after reshuffle attempt
                            logger.warning(
                                "P%d tried DRAW_STOCKPILE, but stockpile/discard empty. Game should end.",
                                player,
                            )
                            action_processed = False  # Action failed

                    elif isinstance(action, ActionDrawDiscard):
                        if (
                            self.house_rules.allowDrawFromDiscardPile
                            and self.discard_pile
                        ):
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
                            action_processed = True
                            # Turn advances after discard/replace decision
                        else:
                            logger.warning(
                                "Invalid Action: DRAW_DISCARD attempted (Allowed:%s, Empty:%s).",
                                self.house_rules.allowDrawFromDiscardPile,
                                not self.discard_pile,
                            )
                            action_processed = False

                    elif isinstance(action, ActionCallCambia):
                        cambia_allowed_round = self.house_rules.cambia_allowed_round
                        current_round = self._turn_number // self.num_players
                        if (
                            self.cambia_caller_id is None
                            and current_round >= cambia_allowed_round
                        ):
                            logger.info("P%d calls Cambia!", player)
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
                            action_processed = True
                            turn_should_advance_after_action = True
                        else:
                            logger.warning(
                                "P%d tried invalid CALL_CAMBIA (Caller:%s, Round:%d, Allowed:%d).",
                                player,
                                self.cambia_caller_id,
                                current_round,
                                cambia_allowed_round,
                            )
                            action_processed = False

                    else:  # An action type not handled at the start of a turn
                        logger.warning(
                            "Unhandled action type %s received at start of turn for P%d.",
                            type(action).__name__,
                            player,
                        )
                        action_processed = False

            # --- Post-Action Processing ---
            if action_processed and turn_should_advance_after_action:
                if not self.snap_phase_active and not self.pending_action:
                    self._advance_turn(undo_stack, delta_list)
                else:
                    logger.debug(
                        "Turn advancement skipped: SnapActive=%s, PendingAction=%s",
                        self.snap_phase_active,
                        (
                            type(self.pending_action).__name__
                            if self.pending_action
                            else None
                        ),
                    )

            # Final game end check (handles cases where action resolves but turn doesn't advance, or after turn advances)
            if not self._game_over:
                self._check_game_end(undo_stack, delta_list)

            # --- Sanity Check ---
            # If no action was processed, but game isn't over and requires action, log potential stall
            if (
                not action_processed
                and not self.pending_action
                and not self.snap_phase_active
                and not self._game_over
            ):
                if not self.get_legal_actions():  # Check if there really are no actions
                    logger.error(
                        "Apply Action: Action %s by P%d was not processed, resulting state has no legal actions but is not terminal! State: %s",
                        action,
                        acting_player,
                        self,
                    )
                    # Consider forcing game over here? Or rely on check_game_end's stalemate logic?
                    # For now, rely on _check_game_end to catch true stalemates.
                else:
                    logger.warning(
                        "Apply Action: Action %s by P%d was not processed, but legal actions still exist. State: %s",
                        action,
                        acting_player,
                        self,
                    )

        except Exception as e_apply:
            logger.exception(
                "Critical error during apply_action for action '%s' by P%d: %s. State: %s. Attempting rollback.",
                action,
                acting_player,
                e_apply,
                self,
            )
            try:
                undo_action()
                logger.info(
                    "Successfully executed master undo after apply_action exception."
                )
            except Exception as e_undo_master:
                logger.error(
                    "Exception during master undo after apply_action error: %s. Game state may be inconsistent!",
                    e_undo_master,
                )
            return [], lambda: None

        return delta_list, undo_action

    # --- Undo/Delta Helper ---
    def _add_change(
        self,
        change_func: Callable[[], Any],
        undo_func: Callable[[], None],
        delta: StateDeltaChange,
        undo_stack: Deque[Callable[[], None]],  # Corrected type hint
        delta_list: StateDelta,
    ):
        """Applies a change, adds its undo operation to the stack, and records the delta."""
        try:
            change_func()
            delta_list.append(delta)
            undo_stack.appendleft(undo_func)
        except Exception as e_change:
            func_name = getattr(change_func, "__name__", repr(change_func))
            logger.exception(
                "Error applying change function '%s' for delta %s: %s",
                func_name,
                delta,
                e_change,
            )
            # Re-raise to be caught by the main apply_action handler for rollback
            raise

    # --- Reshuffle Logic ---
    def _attempt_reshuffle(self, undo_stack_outer: Deque) -> Optional[StateDelta]:
        """
        Reshuffles discard pile (except top card) into stockpile if needed.
        Adds its *own* undo operation to the provided outer undo_stack.
        Returns StateDelta for the reshuffle, or None.
        """
        if self.stockpile:
            return None  # No reshuffle needed

        if len(self.discard_pile) <= 1:
            logger.info("Stockpile empty, cannot reshuffle discard pile (size <= 1).")
            return None  # Cannot reshuffle

        logger.info("Stockpile empty. Reshuffling discard pile.")
        original_stockpile = list(self.stockpile)  # Should be empty
        original_discard = list(self.discard_pile)
        top_card = original_discard[-1]
        cards_to_shuffle = original_discard[:-1]
        new_stockpile = list(cards_to_shuffle)
        random.shuffle(new_stockpile)

        def change_reshuffle():
            self.discard_pile = [top_card]
            self.stockpile = new_stockpile
            logger.info("Reshuffled %d cards into stockpile.", len(self.stockpile))

        def undo_reshuffle():
            self.stockpile = original_stockpile
            self.discard_pile = original_discard
            logger.debug("Undo reshuffle.")

        shuffled_card_strs = [serialize_card(c) for c in new_stockpile]
        delta_reshuffle: StateDelta = [
            ("reshuffle", shuffled_card_strs, serialize_card(top_card))
        ]
        # Apply change *now* and add undo to the *caller's* stack
        try:
            change_reshuffle()
            undo_stack_outer.appendleft(undo_reshuffle)
            return delta_reshuffle
        except Exception as e_reshuffle:
            logger.exception("Error during reshuffle state change: %s", e_reshuffle)
            # Attempt to revert partial changes if possible (undo not on stack yet)
            self.stockpile = original_stockpile
            self.discard_pile = original_discard
            return None  # Indicate reshuffle failed

    # --- Penalty Logic ---
    def _apply_penalty(
        self, player_index: int, num_cards: int, undo_stack_main: Deque
    ) -> StateDelta:
        """
        Applies penalty draw(s), handling reshuffles. Adds ONE combined undo operation.
        Returns a list of StateDeltaChanges for the penalty draws/reshuffles.
        """
        logger.warning(
            "Applying penalty: P%d attempts to draw %d cards.", player_index, num_cards
        )
        penalty_deltas: StateDelta = []
        if not (
            0 <= player_index < len(self.players)
            and hasattr(self.players[player_index], "hand")
        ):
            logger.error(
                "Cannot apply penalty: Player %d invalid or missing hand.", player_index
            )
            return penalty_deltas

        original_hand_state = list(self.players[player_index].hand)
        original_stockpile_state = list(self.stockpile)
        original_discard_state = list(self.discard_pile)
        cards_actually_drawn_this_penalty: List[Card] = []

        try:
            for i in range(num_cards):
                if not self.stockpile:
                    reshuffle_outcome_deltas = self._attempt_reshuffle(undo_stack_main)
                    if reshuffle_outcome_deltas:
                        penalty_deltas.extend(reshuffle_outcome_deltas)
                        logger.debug(
                            "Reshuffled during penalty draw %d/%d", i + 1, num_cards
                        )
                    else:  # Cannot reshuffle
                        logger.warning(
                            "Stockpile/discard empty during penalty draw %d/%d for P%d. Cannot draw more.",
                            i + 1,
                            num_cards,
                            player_index,
                        )
                        break  # Stop drawing

                if self.stockpile:
                    drawn_card = self.stockpile.pop()
                    self.players[player_index].hand.append(drawn_card)
                    cards_actually_drawn_this_penalty.append(drawn_card)
                    delta = ("penalty_draw", player_index, serialize_card(drawn_card))
                    penalty_deltas.append(delta)
                else:  # Should not be reached if reshuffle logic is correct
                    logger.error(
                        "Stockpile empty immediately after attempted reshuffle in penalty draw. Stopping."
                    )
                    break
        except Exception as e_penalty_draw:
            logger.exception(
                "Error during penalty draw loop for P%d: %s", player_index, e_penalty_draw
            )
            # State might be partially changed, undo needs to handle it

        logger.info(
            "Player %d drew %d cards as penalty.",
            player_index,
            len(cards_actually_drawn_this_penalty),
        )

        # Define the single master undo for the *entire penalty sequence*
        def undo_penalty_sequence():
            self.players[player_index].hand = original_hand_state
            self.stockpile = original_stockpile_state
            self.discard_pile = original_discard_state
            logger.debug(
                "Undo penalty sequence applied for P%d. State restored to pre-penalty.",
                player_index,
            )

        undo_stack_main.appendleft(undo_penalty_sequence)
        return penalty_deltas

    # --- Turn Advancement & Game End Logic ---
    def _advance_turn(self, undo_stack: Deque, delta_list: StateDelta):
        """Advances to the next player, updates turn counts, checks for game end."""
        if self._game_over:
            logger.debug("Attempted to advance turn, but game is already over.")
            return

        original_turn = self._turn_number
        original_player = self.current_player_index
        original_cambia_turns = self.turns_after_cambia
        logger.debug("Advancing turn from T#%d P%d", original_turn, original_player)

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
            self.current_player_index = original_player
            self._turn_number = original_turn
            self.turns_after_cambia = original_cambia_turns
            logger.debug(
                "Undo advance turn. Back to T#%d, P%d", original_turn, original_player
            )

        delta_player = ("set_attr", "current_player_index", next_player, original_player)
        delta_turn = ("set_attr", "_turn_number", next_turn, original_turn)
        delta_cambia = (
            "set_attr",
            "turns_after_cambia",
            next_cambia_turns,
            original_cambia_turns,
        )
        try:
            self._add_change(
                change_advance, undo_advance, delta_player, undo_stack, delta_list
            )
            delta_list.append(delta_turn)
            if self.cambia_caller_id is not None:
                delta_list.append(delta_cambia)
        except Exception as e_advance:
            logger.exception("Error applying advance turn state change: %s", e_advance)
            # Undo stack might be inconsistent, re-raise
            raise

        logger.debug(
            "Advanced turn to T#%d P%d. Cambia turns: %d",
            self._turn_number,
            self.current_player_index,
            self.turns_after_cambia,
        )
        self._check_game_end(undo_stack, delta_list)  # Check game end *after* advancing

    def _check_game_end(self, undo_stack: Deque, delta_list: StateDelta):
        """Checks game end conditions and updates state if game has ended."""
        if self._game_over:
            return

        end_condition_met = False
        reason = ""
        max_turns = self.house_rules.max_game_turns

        # 1. Max Turns
        if max_turns > 0 and self._turn_number >= max_turns:
            end_condition_met = True
            reason = f"Max game turns ({max_turns}) reached"

        # 2. Cambia Round Completed
        elif (
            self.cambia_caller_id is not None
            and self.turns_after_cambia >= self.num_players
        ):
            end_condition_met = True
            reason = f"Cambia final turns ({self.turns_after_cambia}/{self.num_players}) completed"

        # 3. Stalemate (Only check if not already ended by other conditions)
        elif not self.pending_action and not self.snap_phase_active:
            player = self.current_player_index
            try:
                # Use helper to check if player is valid before get_legal_actions
                if not (
                    0 <= player < len(self.players)
                    and hasattr(self.players[player], "hand")
                ):
                    logger.error(
                        "Stalemate check: Player %d invalid or missing hand.", player
                    )
                    end_condition_met = True  # Treat as error end state
                    reason = f"Game end check (Stalemate): Invalid player P{player}"
                else:
                    legal_actions = self.get_legal_actions()  # Method from QueryMixin
                    if not legal_actions:
                        # Only a stalemate if cannot draw/reshuffle
                        can_draw_stockpile = bool(self.stockpile)
                        can_reshuffle = len(self.discard_pile) > 1
                        if not can_draw_stockpile and not can_reshuffle:
                            end_condition_met = True
                            reason = f"Stalemate: P{player} has no actions and cannot draw/reshuffle (Stock: {len(self.stockpile)}, Disc: {len(self.discard_pile)})"
                        # else: Has no actions, but can draw/reshuffle, not a true stalemate yet
            except Exception as e_stalemate_check:
                logger.error(
                    "Error during stalemate check for P%d: %s",
                    player,
                    e_stalemate_check,
                    exc_info=True,
                )
                # If checking legal actions fails, maybe end game? Or just log error? Log for now.

        # Finalize Game End
        if end_condition_met and not self._game_over:  # Prevent multiple triggers
            logger.info("Game ends: %s.", reason)
            try:
                temp_winner, temp_utilities = self._calculate_final_scores(
                    set_attributes=False
                )
            except Exception as e_score_calc:
                logger.error(
                    "Error calculating final scores for game end: %s",
                    e_score_calc,
                    exc_info=True,
                )
                temp_winner, temp_utilities = (
                    None,
                    [0.0] * self.num_players,
                )  # Default on error

            original_game_over = self._game_over
            original_winner = self._winner
            original_utilities = list(self._utilities)

            def change_game_end():
                self._game_over = True
                try:
                    # Recalculate within change to ensure final state is set
                    self._calculate_final_scores(set_attributes=True)
                except Exception as e_set_score:
                    logger.error(
                        "Error setting final scores in change_game_end: %s",
                        e_set_score,
                        exc_info=True,
                    )
                    # Set default values to avoid inconsistent state
                    self._winner = None
                    self._utilities = [0.0] * self.num_players

            def undo_game_end():
                self._game_over = original_game_over
                self._winner = original_winner
                self._utilities = original_utilities
                logger.debug("Undo game end.")

            delta_game_end = ("game_end", reason, temp_winner, temp_utilities)
            try:
                self._add_change(
                    change_game_end, undo_game_end, delta_game_end, undo_stack, delta_list
                )
            except Exception as e_add_end:
                logger.error(
                    "Failed to add game end change/undo: %s", e_add_end, exc_info=True
                )
                # Game might not be marked as over, subsequent actions might occur

    def _calculate_final_scores(
        self, set_attributes: bool = True
    ) -> Tuple[Optional[int], List[float]]:
        """
        Calculates final scores and determines winner/utilities.
        Optionally sets _winner and _utilities attributes.
        """
        initial_utilities_check = [0.0] * self.num_players
        if set_attributes and self._utilities != initial_utilities_check:
            logger.debug(
                "Scores already calculated, returning stored: W=%s, U=%s",
                self._winner,
                self._utilities,
            )
            return self._winner, list(self._utilities)

        scores = []
        final_hands_str = []
        valid_scores_exist = False
        for i, player_state in enumerate(self.players):
            if (
                0 <= i < len(self.players)
                and hasattr(player_state, "hand")
                and isinstance(player_state.hand, list)
            ):
                current_hand = player_state.hand
                if not all(isinstance(card, Card) for card in current_hand):
                    logger.error(
                        "Calc Score: P%d hand contains non-Card objects: %s. Assigning max score.",
                        i,
                        current_hand,
                    )
                    scores.append(float("inf"))
                    final_hands_str.append([str(c) for c in current_hand])
                else:
                    hand_value = sum(card.value for card in current_hand)
                    scores.append(hand_value)
                    final_hands_str.append([serialize_card(c) for c in current_hand])
                    if hand_value != float("inf"):
                        valid_scores_exist = True
            else:
                logger.error(
                    "Calc Score: P%d invalid or missing hand. Assigning max score.", i
                )
                scores.append(float("inf"))
                final_hands_str.append(["ERROR"])

        winner_calculated: Optional[int] = None
        utilities_calculated: List[float] = [0.0] * self.num_players

        if not scores:
            logger.error("Cannot calculate final scores: No player scores available.")
        elif not valid_scores_exist:
            logger.warning(
                "All players have invalid hands or max score. Declaring tie with 0 utility."
            )
            winner_calculated = None
            utilities_calculated = [0.0] * self.num_players
        else:
            min_score = min(scores)
            winners = [i for i, score in enumerate(scores) if score == min_score]

            if len(winners) == 1:
                winner_calculated = winners[0]
                utilities_calculated = [-1.0] * self.num_players
                utilities_calculated[winner_calculated] = 1.0
            else:  # Tie situation
                if self.cambia_caller_id is not None and self.cambia_caller_id in winners:
                    winner_calculated = self.cambia_caller_id
                    utilities_calculated = [-1.0] * self.num_players
                    utilities_calculated[winner_calculated] = 1.0
                    logger.info(
                        "Tie score (%s) broken by Cambia caller P%d.",
                        min_score,
                        winner_calculated,
                    )
                else:
                    logger.info(
                        "True tie between players %s with score %s.", winners, min_score
                    )
                    winner_calculated = None
                    utilities_calculated = [-1.0] * self.num_players
                    for p_idx in winners:
                        utilities_calculated[p_idx] = 0.0

        if set_attributes:
            log_msg = "Game Score Calc: "
            if winner_calculated is not None:
                log_msg += (
                    f"P{winner_calculated} wins (Score: {scores[winner_calculated]}). "
                )
            else:
                log_msg += f"Tie (Score: {min_score if valid_scores_exist else 'N/A'}). "
            log_msg += (
                f"Utilities: {utilities_calculated}. Final Hands: {final_hands_str}"
            )
            logger.info(log_msg)

            self._winner = winner_calculated
            self._utilities = list(utilities_calculated)  # Ensure list copy
            logger.debug(
                "Final winner/utilities set: W=%s, U=%s", self._winner, self._utilities
            )

        return winner_calculated, list(utilities_calculated)  # Return list copy
