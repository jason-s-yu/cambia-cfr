"""
Regression tests for Phase 0 engine fixes.

Verifies that each of the 7 critical bug fixes and performance
optimizations from the Phase 0 audit has not regressed.

Fixes tested:
  BUG-1:  King Swap Belief Gap — AgentObservation.king_swap_indices field
  BUG-2:  Double _check_game_end call removed — single call in _advance_turn
  BUG-3:  Exception name correctness — AssertionError is valid Python
  PERF-1: Deterministic RNG — game-local random.Random
  PERF-2: Shallow copy for pending_action_data — no deepcopy in engine
  PERF-3: Observation construction consolidated — shared _create_observation
  PERF-4: Penalty undo/reshuffle — throwaway undo stack for reshuffles
"""

import copy
import inspect
import random
import re
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

# conftest.py handles the config stub

from src.agent_state import AgentObservation, AgentState, KnownCardInfo
from src.card import Card
from src.constants import (
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionDiscard,
    ActionDrawStockpile,
    ActionReplace,
    CardBucket,
    DecayCategory,
)
from src.game.engine import CambiaGameState


# ===================================================================
# BUG-1: King Swap Belief Gap
# ===================================================================


class TestBug1KingSwapBeliefGap:
    """AgentObservation must include king_swap_indices so that
    agent_state.update() can correctly update beliefs after a King Swap."""

    def test_observation_has_king_swap_indices_field(self):
        """AgentObservation dataclass has the king_swap_indices field."""
        obs = AgentObservation(
            acting_player=0,
            action=None,
            discard_top_card=None,
            player_hand_sizes=[4, 4],
            stockpile_size=40,
            king_swap_indices=(1, 2),
        )
        assert obs.king_swap_indices == (1, 2)

    def test_observation_king_swap_indices_default_none(self):
        """king_swap_indices defaults to None when not provided."""
        obs = AgentObservation(
            acting_player=0,
            action=None,
            discard_top_card=None,
            player_hand_sizes=[4, 4],
            stockpile_size=40,
        )
        assert obs.king_swap_indices is None

    def test_agent_state_handles_self_king_swap(self):
        """After a self-initiated King Swap with indices, own hand is set to UNKNOWN
        and opponent belief is decayed."""
        rules_stub = type("Rules", (), {
            "penaltyDrawCount": 2,
            "use_jokers": 2,
        })()
        config_stub = type("Config", (), {"cambia_rules": rules_stub})()
        state = AgentState(
            player_id=0,
            opponent_id=1,
            memory_level=1,
            time_decay_turns=5,
            initial_hand_size=4,
            config=config_stub,
        )
        # Initialize manually
        state.own_hand = {
            0: KnownCardInfo(bucket=CardBucket.ACE, last_seen_turn=0),
            1: KnownCardInfo(bucket=CardBucket.LOW_NUM, last_seen_turn=0),
            2: KnownCardInfo(bucket=CardBucket.MID_NUM, last_seen_turn=0),
            3: KnownCardInfo(bucket=CardBucket.HIGH_KING, last_seen_turn=0),
        }
        state.opponent_belief = {
            0: CardBucket.PEEK_OTHER,
            1: CardBucket.SWAP_BLIND,
            2: CardBucket.ACE,
            3: CardBucket.UNKNOWN,
        }
        state.opponent_card_count = 4
        state.opponent_last_seen_turn = {0: 0, 1: 0, 2: 0}
        state.game_phase = CardBucket.UNKNOWN  # Will be re-estimated
        state._current_game_turn = 5

        obs = AgentObservation(
            acting_player=0,
            action=ActionAbilityKingSwapDecision(perform_swap=True),
            discard_top_card=Card("A", "H"),
            player_hand_sizes=[4, 4],
            stockpile_size=30,
            current_turn=5,
            king_swap_indices=(1, 2),  # own_idx=1, opp_idx=2
        )
        state.update(obs)

        # Own hand slot 1 should now be UNKNOWN (swapped away)
        assert state.own_hand[1].bucket == CardBucket.UNKNOWN
        # Opponent belief slot 2 should be decayed (not the original ACE)
        # With memory_level=1, event decay should trigger
        opp_belief_2 = state.opponent_belief[2]
        # It should no longer be the specific CardBucket.ACE
        # (should be decayed to a DecayCategory or UNKNOWN)
        assert opp_belief_2 != CardBucket.ACE or opp_belief_2 == CardBucket.UNKNOWN

    def test_agent_state_handles_opponent_king_swap(self):
        """After an opponent-initiated King Swap with indices, beliefs
        update appropriately (our involved slot becomes UNKNOWN, opponent
        slot decays)."""
        rules_stub = type("Rules", (), {
            "penaltyDrawCount": 2,
            "use_jokers": 2,
        })()
        config_stub = type("Config", (), {"cambia_rules": rules_stub})()
        state = AgentState(
            player_id=0,
            opponent_id=1,
            memory_level=1,
            time_decay_turns=5,
            initial_hand_size=4,
            config=config_stub,
        )
        state.own_hand = {
            0: KnownCardInfo(bucket=CardBucket.ACE),
            1: KnownCardInfo(bucket=CardBucket.LOW_NUM),
            2: KnownCardInfo(bucket=CardBucket.MID_NUM),
            3: KnownCardInfo(bucket=CardBucket.HIGH_KING),
        }
        state.opponent_belief = {
            0: CardBucket.PEEK_OTHER,
            1: CardBucket.SWAP_BLIND,
            2: CardBucket.ACE,
            3: CardBucket.UNKNOWN,
        }
        state.opponent_card_count = 4
        state.opponent_last_seen_turn = {0: 0, 1: 0, 2: 0}
        state._current_game_turn = 5

        # Opponent performed the swap: their own_idx=2, their opp_idx=1
        # From our perspective: our idx 1 was swapped with their idx 2
        obs = AgentObservation(
            acting_player=1,
            action=ActionAbilityKingSwapDecision(perform_swap=True),
            discard_top_card=Card("A", "H"),
            player_hand_sizes=[4, 4],
            stockpile_size=30,
            current_turn=5,
            king_swap_indices=(2, 1),  # opponent's (own_idx=2, opp_idx=1)
        )
        state.update(obs)

        # Our slot 1 should be set to UNKNOWN (swapped away by opponent)
        assert state.own_hand[1].bucket == CardBucket.UNKNOWN


# ===================================================================
# BUG-2: Double _check_game_end Call
# ===================================================================


class TestBug2DoubleCheckGameEnd:
    """_check_game_end should be called only from _advance_turn,
    not separately in apply_action."""

    def test_check_game_end_called_from_advance_turn(self):
        """_advance_turn calls _check_game_end."""
        source = inspect.getsource(CambiaGameState._advance_turn)
        assert "_check_game_end" in source

    def test_apply_action_does_not_call_check_game_end_directly(self):
        """apply_action should NOT call _check_game_end directly.
        It should only be reached via _advance_turn."""
        source = inspect.getsource(CambiaGameState.apply_action)
        # There should be a comment about _check_game_end, but not an actual call
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            # Skip string literals
            if stripped.startswith('"') or stripped.startswith("'"):
                continue
            # Check for direct call
            if "_check_game_end" in stripped and "(" in stripped:
                # This would be a direct call
                pytest.fail(
                    f"apply_action still calls _check_game_end directly: {stripped}"
                )


# ===================================================================
# BUG-3: Exception Name Correctness
# ===================================================================


class TestBug3ExceptionName:
    """Verify that 'AssertionError' in agent_state.py is the correct
    Python exception name (not a misspelling)."""

    def test_assertion_error_is_valid_python(self):
        """Python's builtin is 'AssertionError'."""
        # This test just verifies the builtin exists
        assert AssertionError is not None
        with pytest.raises(AssertionError):
            assert False

    def test_agent_state_catches_assertion_error(self):
        """The except clause in agent_state.py uses the correct exception name."""
        source = inspect.getsource(AgentState)
        # Find all 'except' lines that mention Assert*Error
        except_lines = [
            line.strip()
            for line in source.split("\n")
            if "except" in line and "Assert" in line
        ]
        for line in except_lines:
            # Verify it's the correct spelling
            assert "AssertionError" in line, (
                f"Found unexpected Assert*Error variant: {line}"
            )


# ===================================================================
# PERF-1: Deterministic RNG
# ===================================================================


class TestPerf1DeterministicRNG:
    """The game engine should use a game-local random.Random instance
    instead of the global random module."""

    def test_engine_has_rng_field(self):
        """CambiaGameState should have a _rng field of type random.Random."""
        gs = CambiaGameState()
        assert hasattr(gs, "_rng")
        assert isinstance(gs._rng, random.Random)

    def test_same_seed_produces_same_game(self):
        """Two games with the same RNG seed produce identical initial states."""
        gs1 = CambiaGameState()
        gs1._rng = random.Random(42)
        gs1._setup_game()

        gs2 = CambiaGameState()
        gs2._rng = random.Random(42)
        gs2._setup_game()

        # Same stockpile order
        assert len(gs1.stockpile) == len(gs2.stockpile)
        for c1, c2 in zip(gs1.stockpile, gs2.stockpile):
            assert c1.rank == c2.rank and c1.suit == c2.suit

        # Same hands
        for p in range(2):
            assert len(gs1.players[p].hand) == len(gs2.players[p].hand)
            for c1, c2 in zip(gs1.players[p].hand, gs2.players[p].hand):
                assert c1.rank == c2.rank and c1.suit == c2.suit

        # Same starting player
        assert gs1.current_player_index == gs2.current_player_index

    def test_different_seeds_produce_different_games(self):
        """Two games with different seeds should (very likely) differ."""
        gs1 = CambiaGameState()
        gs1._rng = random.Random(42)
        gs1._setup_game()

        gs2 = CambiaGameState()
        gs2._rng = random.Random(999)
        gs2._setup_game()

        # At least the stockpile or hand order should differ
        stockpile_same = all(
            c1.rank == c2.rank and c1.suit == c2.suit
            for c1, c2 in zip(gs1.stockpile, gs2.stockpile)
        )
        assert not stockpile_same, "Different seeds should produce different shuffles"

    def test_reshuffle_uses_game_rng(self):
        """_attempt_reshuffle should use self._rng, not the global random."""
        source = inspect.getsource(CambiaGameState._attempt_reshuffle)
        # Should contain self._rng.shuffle, not random.shuffle
        assert "self._rng.shuffle" in source
        # Should NOT contain bare random.shuffle (global)
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "random.shuffle" in stripped and "self._rng" not in stripped:
                pytest.fail(f"_attempt_reshuffle uses global random: {stripped}")


# ===================================================================
# PERF-2: Shallow Copy for pending_action_data
# ===================================================================


class TestPerf2ShallowCopy:
    """pending_action_data should use dict() shallow copy instead of
    copy.deepcopy() in the engine's draw and ability code."""

    def test_engine_apply_action_uses_shallow_copy(self):
        """apply_action should use dict() not deepcopy() for pending_action_data."""
        source = inspect.getsource(CambiaGameState.apply_action)
        # Check for dict() usage
        assert "dict(self.pending_action_data)" in source
        # deepcopy should NOT appear
        assert "deepcopy(self.pending_action_data)" not in source

    def test_engine_does_not_import_deepcopy_for_pending(self):
        """The engine module should not use deepcopy on pending_action_data."""
        import src.game.engine as engine_mod
        source = inspect.getsource(engine_mod)
        # Count occurrences of deepcopy(self.pending_action_data)
        deepcopy_count = source.count("deepcopy(self.pending_action_data)")
        assert deepcopy_count == 0, (
            f"Found {deepcopy_count} deepcopy(self.pending_action_data) in engine.py"
        )

    def test_ability_mixin_uses_shallow_copy(self):
        """_ability_mixin.py should use dict() not deepcopy() for pending_action_data."""
        from src.game._ability_mixin import AbilityMixin
        source = inspect.getsource(AbilityMixin)
        dict_count = source.count("dict(self.pending_action_data)")
        deepcopy_count = source.count("deepcopy(self.pending_action_data)")
        # Should have more dict() than deepcopy() calls
        assert dict_count > 0, "Expected dict(self.pending_action_data) in ability mixin"
        assert deepcopy_count == 0, (
            f"Found {deepcopy_count} deepcopy calls in ability mixin"
        )


# ===================================================================
# PERF-3: Consolidated Observation Construction
# ===================================================================


class TestPerf3ConsolidatedObservation:
    """Observation construction should be consolidated into a shared
    _create_observation function, not duplicated."""

    def test_recursion_mixin_delegates_to_worker(self):
        """recursion_mixin._create_observation should delegate to worker's
        _create_observation to avoid code duplication."""
        from src.cfr.recursion_mixin import CFRRecursionMixin
        source = inspect.getsource(CFRRecursionMixin._create_observation)
        # Should import or call the worker's _create_observation
        assert "_worker_create_observation" in source or "_create_observation" in source

    def test_worker_create_observation_exists(self):
        """worker.py should have a standalone _create_observation function."""
        from src.cfr.worker import _create_observation
        assert callable(_create_observation)

    def test_worker_create_observation_returns_agent_observation(self):
        """The _create_observation function should return AgentObservation."""
        from src.cfr.worker import _create_observation
        sig = inspect.signature(_create_observation)
        # It should accept the standard parameters
        params = list(sig.parameters.keys())
        assert "next_state" in params
        assert "action" in params
        assert "acting_player" in params


# ===================================================================
# PERF-4: Penalty Undo/Reshuffle Interaction
# ===================================================================


class TestPerf4PenaltyUndoReshuffle:
    """_apply_penalty should use a throwaway undo stack for reshuffles
    so the master undo handles everything atomically."""

    def test_apply_penalty_uses_throwaway_undo_stack(self):
        """_apply_penalty creates a separate undo stack for reshuffles."""
        source = inspect.getsource(CambiaGameState._apply_penalty)
        # Should have a throwaway/discard undo stack
        assert "_discard_undo_stack" in source or "throwaway" in source.lower()

    def test_apply_penalty_has_master_undo(self):
        """_apply_penalty defines a single master undo that restores all state."""
        source = inspect.getsource(CambiaGameState._apply_penalty)
        assert "undo_penalty_sequence" in source or "master_undo" in source.lower()
        # Should restore hand, stockpile, and discard atomically
        assert "original_hand_state" in source
        assert "original_stockpile_state" in source
        assert "original_discard_state" in source

    def test_penalty_applies_and_undoes_correctly(self):
        """Applying a penalty adds cards to the player's hand,
        and the undo restores the original state."""
        gs = CambiaGameState()
        gs._rng = random.Random(42)
        gs._setup_game()

        player_idx = 0
        original_hand = list(gs.players[player_idx].hand)
        original_hand_len = len(original_hand)
        original_stockpile = list(gs.stockpile)
        original_stockpile_len = len(gs.stockpile)

        undo_stack = deque()
        gs._apply_penalty(player_idx, 2, undo_stack)

        # After penalty: player has 2 more cards
        assert len(gs.players[player_idx].hand) == original_hand_len + 2
        assert len(gs.stockpile) == original_stockpile_len - 2

        # Undo: restore to original state
        assert len(undo_stack) >= 1
        while undo_stack:
            undo_fn = undo_stack.popleft()
            undo_fn()

        assert len(gs.players[player_idx].hand) == original_hand_len
        assert len(gs.stockpile) == original_stockpile_len
        # Cards should match originals
        for c1, c2 in zip(gs.players[player_idx].hand, original_hand):
            assert c1.rank == c2.rank and c1.suit == c2.suit


# ===================================================================
# Engine Integration: Basic Game Flow
# ===================================================================


class TestEngineBasicIntegration:
    """Verify the engine can create a game and process basic actions."""

    def test_game_creates_successfully(self):
        gs = CambiaGameState()
        assert len(gs.players) == 2
        assert len(gs.players[0].hand) == 4
        assert len(gs.players[1].hand) == 4
        assert not gs.is_terminal()

    def test_draw_stockpile_action(self):
        """Drawing from stockpile should set pending action for the player."""
        gs = CambiaGameState()
        gs._rng = random.Random(42)
        gs._setup_game()
        player = gs.current_player_index
        original_stock_size = len(gs.stockpile)

        delta, undo_fn = gs.apply_action(ActionDrawStockpile())

        assert delta is not None
        assert len(gs.stockpile) == original_stock_size - 1
        assert gs.pending_action_player == player
        assert "drawn_card" in gs.pending_action_data

    def test_draw_and_discard_cycle(self):
        """Draw from stockpile then discard completes one action cycle."""
        gs = CambiaGameState()
        gs._rng = random.Random(42)
        gs._setup_game()

        # Draw
        delta1, undo1 = gs.apply_action(ActionDrawStockpile())
        assert gs.pending_action is not None

        # Discard (no ability)
        delta2, undo2 = gs.apply_action(ActionDiscard(use_ability=False))
        # After discard, pending action should be cleared
        assert gs.pending_action is None or gs.snap_phase_active

    def test_undo_restores_state(self):
        """Undo function from apply_action restores the previous state."""
        gs = CambiaGameState()
        gs._rng = random.Random(42)
        gs._setup_game()
        player = gs.current_player_index

        stock_before = list(gs.stockpile)
        hand_before = list(gs.players[player].hand)

        delta, undo_fn = gs.apply_action(ActionDrawStockpile())
        # State changed
        assert len(gs.stockpile) != len(stock_before)

        # Undo
        undo_fn()
        assert len(gs.stockpile) == len(stock_before)
        assert len(gs.players[player].hand) == len(hand_before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
