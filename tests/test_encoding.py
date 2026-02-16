"""
Tests for src/encoding.py

Covers:
- Tensor feature encoding (encode_infoset)
- Action index mapping (action_to_index / index_to_action)
- Action mask encoding (encode_action_mask)
- Edge cases (empty hands, max hands, all UNKNOWN, etc.)
"""

from types import SimpleNamespace

import numpy as np
import pytest

from src.cfr.exceptions import ActionEncodingError

# conftest.py handles the config stub automatically

from src.agent_state import KnownCardInfo
from src.card import Card
from src.constants import (
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionAbilityPeekOtherSelect,
    ActionAbilityPeekOwnSelect,
    ActionCallCambia,
    ActionDiscard,
    ActionDrawDiscard,
    ActionDrawStockpile,
    ActionPassSnap,
    ActionReplace,
    ActionSnapOpponent,
    ActionSnapOpponentMove,
    ActionSnapOwn,
    CardBucket,
    DecayCategory,
    DecisionContext,
    GamePhase,
    StockpileEstimate,
)
from src.encoding import (
    EMPTY_SLOT_IDX,
    INPUT_DIM,
    MAX_HAND,
    NUM_ACTIONS,
    SLOT_ENCODING_DIM,
    _BUCKET_TO_SLOT_IDX,
    _DECAY_TO_SLOT_IDX,
    _encode_slot,
    action_to_index,
    encode_action_mask,
    encode_infoset,
    index_to_action,
)


# --- Helper to build a fake AgentState ---

def _make_agent_state(
    player_id=0,
    own_hand=None,
    opponent_belief=None,
    opponent_card_count=4,
    known_discard_top_bucket=CardBucket.UNKNOWN,
    stockpile_estimate=StockpileEstimate.HIGH,
    game_phase=GamePhase.EARLY,
    cambia_caller=None,
):
    """Build a SimpleNamespace that matches the AgentState interface for encode_infoset."""
    if own_hand is None:
        own_hand = {
            i: KnownCardInfo(bucket=CardBucket.UNKNOWN) for i in range(4)
        }
    if opponent_belief is None:
        opponent_belief = {
            i: CardBucket.UNKNOWN for i in range(opponent_card_count)
        }
    return SimpleNamespace(
        player_id=player_id,
        own_hand=own_hand,
        opponent_belief=opponent_belief,
        opponent_card_count=opponent_card_count,
        known_discard_top_bucket=known_discard_top_bucket,
        stockpile_estimate=stockpile_estimate,
        game_phase=game_phase,
        cambia_caller=cambia_caller,
    )


# ===== Constants =====


class TestConstants:
    def test_input_dim(self):
        assert INPUT_DIM == 222

    def test_num_actions(self):
        assert NUM_ACTIONS == 146

    def test_max_hand(self):
        assert MAX_HAND == 6

    def test_slot_encoding_dim(self):
        assert SLOT_ENCODING_DIM == 15


# ===== _encode_slot =====


class TestEncodeSlot:
    def test_card_buckets(self):
        """Each CardBucket maps to its expected slot index."""
        expected = {
            CardBucket.ZERO: 0,
            CardBucket.NEG_KING: 1,
            CardBucket.ACE: 2,
            CardBucket.LOW_NUM: 3,
            CardBucket.MID_NUM: 4,
            CardBucket.PEEK_SELF: 5,
            CardBucket.PEEK_OTHER: 6,
            CardBucket.SWAP_BLIND: 7,
            CardBucket.HIGH_KING: 8,
            CardBucket.UNKNOWN: 13,
        }
        for bucket, expected_idx in expected.items():
            assert _encode_slot(bucket) == expected_idx, f"Failed for {bucket}"

    def test_decay_categories(self):
        """Each DecayCategory maps to its expected slot index."""
        expected = {
            DecayCategory.LIKELY_LOW: 10,
            DecayCategory.LIKELY_MID: 11,
            DecayCategory.LIKELY_HIGH: 12,
            DecayCategory.UNKNOWN: 13,
        }
        for cat, expected_idx in expected.items():
            assert _encode_slot(cat) == expected_idx, f"Failed for {cat}"

    def test_none_is_empty(self):
        assert _encode_slot(None) == EMPTY_SLOT_IDX

    def test_raw_int_card_bucket(self):
        """Raw integer values matching CardBucket.value map correctly."""
        assert _encode_slot(0) == 0  # ZERO
        assert _encode_slot(8) == 8  # HIGH_KING
        assert _encode_slot(99) == 13  # UNKNOWN

    def test_raw_int_decay_category(self):
        """Raw integer values matching DecayCategory.value map correctly."""
        assert _encode_slot(100) == 10  # LIKELY_LOW
        assert _encode_slot(101) == 11  # LIKELY_MID
        assert _encode_slot(102) == 12  # LIKELY_HIGH

    def test_unknown_fallback(self):
        """Unrecognized values fall back to UNKNOWN (index 13)."""
        assert _encode_slot(999) == 13
        assert _encode_slot(-1) == 13


# ===== encode_infoset =====


class TestEncodeInfoset:
    def test_output_shape_and_dtype(self):
        state = _make_agent_state()
        features = encode_infoset(state, DecisionContext.START_TURN)
        assert features.shape == (INPUT_DIM,)
        assert features.dtype == np.float32

    def test_all_unknown_hand(self):
        """A hand of all UNKNOWN cards encodes to index 13 in each slot block."""
        state = _make_agent_state()
        features = encode_infoset(state, DecisionContext.START_TURN)
        for slot in range(4):
            block = features[slot * SLOT_ENCODING_DIM : (slot + 1) * SLOT_ENCODING_DIM]
            assert block[13] == 1.0, f"Slot {slot} UNKNOWN not set"
            assert block.sum() == 1.0, f"Slot {slot} has non-one-hot encoding"
        # Slots 4 and 5 should be EMPTY
        for slot in range(4, MAX_HAND):
            block = features[slot * SLOT_ENCODING_DIM : (slot + 1) * SLOT_ENCODING_DIM]
            assert block[EMPTY_SLOT_IDX] == 1.0, f"Slot {slot} EMPTY not set"
            assert block.sum() == 1.0

    def test_known_hand_encoding(self):
        """Known buckets in own hand encode to correct positions."""
        own_hand = {
            0: KnownCardInfo(bucket=CardBucket.ZERO),
            1: KnownCardInfo(bucket=CardBucket.ACE),
            2: KnownCardInfo(bucket=CardBucket.HIGH_KING),
            3: KnownCardInfo(bucket=CardBucket.UNKNOWN),
        }
        state = _make_agent_state(own_hand=own_hand)
        features = encode_infoset(state, DecisionContext.START_TURN)
        # Slot 0: ZERO -> index 0
        assert features[0 * SLOT_ENCODING_DIM + 0] == 1.0
        # Slot 1: ACE -> index 2
        assert features[1 * SLOT_ENCODING_DIM + 2] == 1.0
        # Slot 2: HIGH_KING -> index 8
        assert features[2 * SLOT_ENCODING_DIM + 8] == 1.0
        # Slot 3: UNKNOWN -> index 13
        assert features[3 * SLOT_ENCODING_DIM + 13] == 1.0

    def test_opponent_belief_encoding(self):
        """Opponent beliefs encode in the second block of 6 x 15."""
        opp_belief = {
            0: CardBucket.LOW_NUM,
            1: DecayCategory.LIKELY_HIGH,
            2: CardBucket.UNKNOWN,
            3: CardBucket.PEEK_SELF,
        }
        state = _make_agent_state(opponent_belief=opp_belief, opponent_card_count=4)
        features = encode_infoset(state, DecisionContext.START_TURN)
        opp_offset = MAX_HAND * SLOT_ENCODING_DIM  # 90
        # Slot 0: LOW_NUM -> index 3
        assert features[opp_offset + 0 * SLOT_ENCODING_DIM + 3] == 1.0
        # Slot 1: LIKELY_HIGH -> index 12
        assert features[opp_offset + 1 * SLOT_ENCODING_DIM + 12] == 1.0
        # Slot 2: UNKNOWN -> index 13
        assert features[opp_offset + 2 * SLOT_ENCODING_DIM + 13] == 1.0
        # Slot 3: PEEK_SELF -> index 5
        assert features[opp_offset + 3 * SLOT_ENCODING_DIM + 5] == 1.0

    def test_card_counts_normalized(self):
        """Own and opponent card counts are normalized by MAX_HAND."""
        state = _make_agent_state(opponent_card_count=3)
        state.own_hand = {
            i: KnownCardInfo(bucket=CardBucket.UNKNOWN) for i in range(5)
        }
        features = encode_infoset(state, DecisionContext.START_TURN)
        count_offset = 2 * MAX_HAND * SLOT_ENCODING_DIM  # 180
        assert features[count_offset] == pytest.approx(5 / MAX_HAND)  # own
        assert features[count_offset + 1] == pytest.approx(3 / MAX_HAND)  # opp

    def test_card_count_clamped_to_max(self):
        """Card counts exceeding MAX_HAND are clamped."""
        state = _make_agent_state(opponent_card_count=8)
        state.own_hand = {
            i: KnownCardInfo(bucket=CardBucket.UNKNOWN) for i in range(8)
        }
        features = encode_infoset(state, DecisionContext.START_TURN)
        count_offset = 2 * MAX_HAND * SLOT_ENCODING_DIM
        assert features[count_offset] == pytest.approx(1.0)  # clamped to 6/6
        assert features[count_offset + 1] == pytest.approx(1.0)

    def test_drawn_card_encoding(self):
        """Drawn card bucket encodes in 11-dim one-hot at offset 182."""
        state = _make_agent_state()
        drawn_offset = 2 * MAX_HAND * SLOT_ENCODING_DIM + 2  # 182
        # No drawn card -> index 10 (NONE)
        features_none = encode_infoset(state, DecisionContext.START_TURN)
        assert features_none[drawn_offset + 10] == 1.0
        # Drawn ACE -> index 2 (ACE is 3rd in bucket value list)
        features_ace = encode_infoset(
            state, DecisionContext.POST_DRAW, drawn_card_bucket=CardBucket.ACE
        )
        assert features_ace[drawn_offset + 2] == 1.0
        assert features_ace[drawn_offset + 10] == 0.0  # NONE not set

    def test_drawn_card_zero(self):
        """Drawn ZERO (Joker) encodes at index 0."""
        state = _make_agent_state()
        drawn_offset = 182
        features = encode_infoset(
            state, DecisionContext.POST_DRAW, drawn_card_bucket=CardBucket.ZERO
        )
        assert features[drawn_offset + 0] == 1.0

    def test_drawn_card_unknown(self):
        """Drawn UNKNOWN encodes at index 9."""
        state = _make_agent_state()
        drawn_offset = 182
        features = encode_infoset(
            state, DecisionContext.POST_DRAW, drawn_card_bucket=CardBucket.UNKNOWN
        )
        assert features[drawn_offset + 9] == 1.0

    def test_discard_top_encoding(self):
        """Discard top bucket encodes in 10-dim one-hot at offset 193."""
        discard_offset = 193
        for bucket, expected_idx in [
            (CardBucket.ZERO, 0),
            (CardBucket.HIGH_KING, 8),
            (CardBucket.UNKNOWN, 9),
        ]:
            state = _make_agent_state(known_discard_top_bucket=bucket)
            features = encode_infoset(state, DecisionContext.START_TURN)
            assert features[discard_offset + expected_idx] == 1.0
            assert features[discard_offset : discard_offset + 10].sum() == 1.0

    def test_stockpile_estimate_encoding(self):
        """Stockpile estimate encodes in 4-dim one-hot at offset 203."""
        stock_offset = 203
        mapping = {
            StockpileEstimate.HIGH: 0,
            StockpileEstimate.MEDIUM: 1,
            StockpileEstimate.LOW: 2,
            StockpileEstimate.EMPTY: 3,
        }
        for est, expected_idx in mapping.items():
            state = _make_agent_state(stockpile_estimate=est)
            features = encode_infoset(state, DecisionContext.START_TURN)
            assert features[stock_offset + expected_idx] == 1.0
            assert features[stock_offset : stock_offset + 4].sum() == 1.0

    def test_game_phase_encoding(self):
        """Game phase encodes in 6-dim one-hot at offset 207."""
        phase_offset = 207
        mapping = {
            GamePhase.START: 0,
            GamePhase.EARLY: 1,
            GamePhase.MID: 2,
            GamePhase.LATE: 3,
            GamePhase.CAMBIA_CALLED: 4,
            GamePhase.TERMINAL: 5,
        }
        for phase, expected_idx in mapping.items():
            state = _make_agent_state(game_phase=phase)
            features = encode_infoset(state, DecisionContext.START_TURN)
            assert features[phase_offset + expected_idx] == 1.0
            assert features[phase_offset : phase_offset + 6].sum() == 1.0

    def test_decision_context_encoding(self):
        """Decision context encodes in 6-dim one-hot at offset 213."""
        ctx_offset = 213
        mapping = {
            DecisionContext.START_TURN: 0,
            DecisionContext.POST_DRAW: 1,
            DecisionContext.SNAP_DECISION: 2,
            DecisionContext.ABILITY_SELECT: 3,
            DecisionContext.SNAP_MOVE: 4,
            DecisionContext.TERMINAL: 5,
        }
        state = _make_agent_state()
        for ctx, expected_idx in mapping.items():
            features = encode_infoset(state, ctx)
            assert features[ctx_offset + expected_idx] == 1.0
            assert features[ctx_offset : ctx_offset + 6].sum() == 1.0

    def test_cambia_caller_encoding(self):
        """Cambia caller encodes in 3-dim one-hot at offset 219."""
        cambia_offset = 219
        # NONE
        state = _make_agent_state(cambia_caller=None)
        f = encode_infoset(state, DecisionContext.START_TURN)
        assert f[cambia_offset + 2] == 1.0  # NONE
        # SELF
        state = _make_agent_state(player_id=0, cambia_caller=0)
        f = encode_infoset(state, DecisionContext.START_TURN)
        assert f[cambia_offset + 0] == 1.0  # SELF
        # OPPONENT
        state = _make_agent_state(player_id=0, cambia_caller=1)
        f = encode_infoset(state, DecisionContext.START_TURN)
        assert f[cambia_offset + 1] == 1.0  # OPPONENT

    def test_total_offset_is_222(self):
        """The encoding fills exactly 222 dimensions."""
        state = _make_agent_state()
        features = encode_infoset(state, DecisionContext.START_TURN)
        assert features.shape == (222,)
        # All one-hot blocks contribute at least one non-zero entry each.
        # 12 one-hot blocks (6 own + 6 opp + drawn + discard + stock + phase + ctx + cambia)
        # plus 2 normalized scalars
        non_zero = np.count_nonzero(features)
        assert non_zero >= 14

    def test_empty_hand(self):
        """An agent with no cards encodes all own slots as EMPTY."""
        state = _make_agent_state()
        state.own_hand = {}
        features = encode_infoset(state, DecisionContext.START_TURN)
        for slot in range(MAX_HAND):
            block = features[slot * SLOT_ENCODING_DIM : (slot + 1) * SLOT_ENCODING_DIM]
            assert block[EMPTY_SLOT_IDX] == 1.0

    def test_max_hand_six_slots(self):
        """A full 6-card hand fills all own slots."""
        own_hand = {
            i: KnownCardInfo(bucket=CardBucket.ACE) for i in range(6)
        }
        state = _make_agent_state(own_hand=own_hand)
        features = encode_infoset(state, DecisionContext.START_TURN)
        for slot in range(6):
            block = features[slot * SLOT_ENCODING_DIM : (slot + 1) * SLOT_ENCODING_DIM]
            assert block[2] == 1.0  # ACE index

    def test_features_are_zero_initialized(self):
        """Areas not filled by one-hot or scalar remain zero."""
        state = _make_agent_state()
        features = encode_infoset(state, DecisionContext.START_TURN)
        for slot in range(4, MAX_HAND):
            block = features[slot * SLOT_ENCODING_DIM : (slot + 1) * SLOT_ENCODING_DIM]
            for i in range(SLOT_ENCODING_DIM):
                if i == EMPTY_SLOT_IDX:
                    assert block[i] == 1.0
                else:
                    assert block[i] == 0.0

    def test_different_states_produce_different_features(self):
        """Two different game states produce different feature vectors."""
        s1 = _make_agent_state(game_phase=GamePhase.EARLY)
        s2 = _make_agent_state(game_phase=GamePhase.LATE)
        f1 = encode_infoset(s1, DecisionContext.START_TURN)
        f2 = encode_infoset(s2, DecisionContext.START_TURN)
        assert not np.array_equal(f1, f2)

    def test_decay_in_opponent_belief(self):
        """DecayCategory values in opponent belief encode correctly."""
        opp_belief = {
            0: DecayCategory.LIKELY_LOW,
            1: DecayCategory.LIKELY_MID,
            2: DecayCategory.LIKELY_HIGH,
        }
        state = _make_agent_state(opponent_belief=opp_belief, opponent_card_count=3)
        features = encode_infoset(state, DecisionContext.START_TURN)
        opp_offset = MAX_HAND * SLOT_ENCODING_DIM
        assert features[opp_offset + 0 * SLOT_ENCODING_DIM + 10] == 1.0  # LIKELY_LOW
        assert features[opp_offset + 1 * SLOT_ENCODING_DIM + 11] == 1.0  # LIKELY_MID
        assert features[opp_offset + 2 * SLOT_ENCODING_DIM + 12] == 1.0  # LIKELY_HIGH


# ===== action_to_index =====


class TestActionToIndex:
    def test_draw_stockpile(self):
        assert action_to_index(ActionDrawStockpile()) == 0

    def test_draw_discard(self):
        assert action_to_index(ActionDrawDiscard()) == 1

    def test_call_cambia(self):
        assert action_to_index(ActionCallCambia()) == 2

    def test_discard_no_ability(self):
        assert action_to_index(ActionDiscard(use_ability=False)) == 3

    def test_discard_with_ability(self):
        assert action_to_index(ActionDiscard(use_ability=True)) == 4

    def test_replace(self):
        for i in range(MAX_HAND):
            assert action_to_index(ActionReplace(target_hand_index=i)) == 5 + i

    def test_peek_own(self):
        for i in range(MAX_HAND):
            assert action_to_index(ActionAbilityPeekOwnSelect(target_hand_index=i)) == 11 + i

    def test_peek_other(self):
        for i in range(MAX_HAND):
            assert (
                action_to_index(
                    ActionAbilityPeekOtherSelect(target_opponent_hand_index=i)
                )
                == 17 + i
            )

    def test_blind_swap(self):
        for own in range(MAX_HAND):
            for opp in range(MAX_HAND):
                expected = 23 + own * MAX_HAND + opp
                assert (
                    action_to_index(
                        ActionAbilityBlindSwapSelect(
                            own_hand_index=own, opponent_hand_index=opp
                        )
                    )
                    == expected
                )

    def test_blind_swap_boundaries(self):
        """First and last BlindSwap index."""
        assert action_to_index(
            ActionAbilityBlindSwapSelect(own_hand_index=0, opponent_hand_index=0)
        ) == 23
        assert action_to_index(
            ActionAbilityBlindSwapSelect(own_hand_index=5, opponent_hand_index=5)
        ) == 58

    def test_king_look(self):
        for own in range(MAX_HAND):
            for opp in range(MAX_HAND):
                expected = 59 + own * MAX_HAND + opp
                assert (
                    action_to_index(
                        ActionAbilityKingLookSelect(
                            own_hand_index=own, opponent_hand_index=opp
                        )
                    )
                    == expected
                )

    def test_king_look_boundaries(self):
        assert action_to_index(
            ActionAbilityKingLookSelect(own_hand_index=0, opponent_hand_index=0)
        ) == 59
        assert action_to_index(
            ActionAbilityKingLookSelect(own_hand_index=5, opponent_hand_index=5)
        ) == 94

    def test_king_swap_decision(self):
        assert action_to_index(ActionAbilityKingSwapDecision(perform_swap=False)) == 95
        assert action_to_index(ActionAbilityKingSwapDecision(perform_swap=True)) == 96

    def test_pass_snap(self):
        assert action_to_index(ActionPassSnap()) == 97

    def test_snap_own(self):
        for i in range(MAX_HAND):
            assert action_to_index(ActionSnapOwn(own_card_hand_index=i)) == 98 + i

    def test_snap_opponent(self):
        for i in range(MAX_HAND):
            assert (
                action_to_index(ActionSnapOpponent(opponent_target_hand_index=i))
                == 104 + i
            )

    def test_snap_opponent_move(self):
        for own in range(MAX_HAND):
            for slot in range(MAX_HAND):
                expected = 110 + own * MAX_HAND + slot
                assert (
                    action_to_index(
                        ActionSnapOpponentMove(
                            own_card_to_move_hand_index=own,
                            target_empty_slot_index=slot,
                        )
                    )
                    == expected
                )

    def test_snap_opponent_move_boundaries(self):
        assert action_to_index(
            ActionSnapOpponentMove(own_card_to_move_hand_index=0, target_empty_slot_index=0)
        ) == 110
        assert action_to_index(
            ActionSnapOpponentMove(own_card_to_move_hand_index=5, target_empty_slot_index=5)
        ) == 145

    def test_max_index_is_145(self):
        """The highest valid action index should be 145 (NUM_ACTIONS - 1)."""
        max_idx = action_to_index(
            ActionSnapOpponentMove(own_card_to_move_hand_index=5, target_empty_slot_index=5)
        )
        assert max_idx == NUM_ACTIONS - 1

    def test_all_indices_unique(self):
        """Every action type produces a unique index, covering [0, 146)."""
        all_actions = [
            ActionDrawStockpile(),
            ActionDrawDiscard(),
            ActionCallCambia(),
            ActionDiscard(use_ability=False),
            ActionDiscard(use_ability=True),
        ]
        for i in range(MAX_HAND):
            all_actions.append(ActionReplace(target_hand_index=i))
        for i in range(MAX_HAND):
            all_actions.append(ActionAbilityPeekOwnSelect(target_hand_index=i))
        for i in range(MAX_HAND):
            all_actions.append(ActionAbilityPeekOtherSelect(target_opponent_hand_index=i))
        for own in range(MAX_HAND):
            for opp in range(MAX_HAND):
                all_actions.append(
                    ActionAbilityBlindSwapSelect(own_hand_index=own, opponent_hand_index=opp)
                )
        for own in range(MAX_HAND):
            for opp in range(MAX_HAND):
                all_actions.append(
                    ActionAbilityKingLookSelect(own_hand_index=own, opponent_hand_index=opp)
                )
        all_actions.extend([
            ActionAbilityKingSwapDecision(perform_swap=False),
            ActionAbilityKingSwapDecision(perform_swap=True),
            ActionPassSnap(),
        ])
        for i in range(MAX_HAND):
            all_actions.append(ActionSnapOwn(own_card_hand_index=i))
        for i in range(MAX_HAND):
            all_actions.append(ActionSnapOpponent(opponent_target_hand_index=i))
        for own in range(MAX_HAND):
            for slot in range(MAX_HAND):
                all_actions.append(
                    ActionSnapOpponentMove(
                        own_card_to_move_hand_index=own,
                        target_empty_slot_index=slot,
                    )
                )

        indices = [action_to_index(a) for a in all_actions]
        assert len(indices) == NUM_ACTIONS
        assert len(set(indices)) == NUM_ACTIONS
        assert min(indices) == 0
        assert max(indices) == NUM_ACTIONS - 1

    def test_replace_out_of_range(self):
        with pytest.raises(ActionEncodingError, match="out of range"):
            action_to_index(ActionReplace(target_hand_index=MAX_HAND))

    def test_replace_negative_index(self):
        with pytest.raises(ActionEncodingError, match="out of range"):
            action_to_index(ActionReplace(target_hand_index=-1))

    def test_blind_swap_out_of_range(self):
        with pytest.raises(ActionEncodingError, match="out of range"):
            action_to_index(
                ActionAbilityBlindSwapSelect(own_hand_index=6, opponent_hand_index=0)
            )


# ===== index_to_action =====


class TestIndexToAction:
    def test_round_trip_draw_stockpile(self):
        actions = [ActionDrawStockpile(), ActionDrawDiscard(), ActionCallCambia()]
        for a in actions:
            idx = action_to_index(a)
            recovered = index_to_action(idx, actions)
            assert recovered == a

    def test_round_trip_replace(self):
        legal = [ActionReplace(target_hand_index=i) for i in range(4)]
        for a in legal:
            idx = action_to_index(a)
            recovered = index_to_action(idx, legal)
            assert recovered == a

    def test_round_trip_blind_swap(self):
        legal = [
            ActionAbilityBlindSwapSelect(own_hand_index=1, opponent_hand_index=2),
            ActionAbilityBlindSwapSelect(own_hand_index=3, opponent_hand_index=0),
        ]
        for a in legal:
            idx = action_to_index(a)
            recovered = index_to_action(idx, legal)
            assert recovered == a

    def test_no_match_raises(self):
        legal = [ActionDrawStockpile()]
        with pytest.raises(ActionEncodingError, match="No legal action matches index"):
            index_to_action(99, legal)

    def test_empty_legal_actions_raises(self):
        with pytest.raises(ActionEncodingError):
            index_to_action(0, [])


# ===== encode_action_mask =====


class TestEncodeActionMask:
    def test_output_shape_and_dtype(self):
        mask = encode_action_mask([ActionDrawStockpile()])
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.dtype == bool

    def test_single_action(self):
        mask = encode_action_mask([ActionDrawStockpile()])
        assert mask[0] is np.bool_(True)
        assert mask.sum() == 1

    def test_multiple_actions(self):
        actions = [
            ActionDrawStockpile(),
            ActionDrawDiscard(),
            ActionCallCambia(),
        ]
        mask = encode_action_mask(actions)
        assert mask[0] and mask[1] and mask[2]
        assert mask.sum() == 3

    def test_empty_actions(self):
        mask = encode_action_mask([])
        assert mask.sum() == 0

    def test_replace_actions_mask(self):
        actions = [ActionReplace(target_hand_index=i) for i in range(4)]
        mask = encode_action_mask(actions)
        for i in range(4):
            assert mask[5 + i]
        for i in range(4, MAX_HAND):
            assert not mask[5 + i]

    def test_snap_actions_mask(self):
        actions = [
            ActionPassSnap(),
            ActionSnapOwn(own_card_hand_index=2),
            ActionSnapOpponent(opponent_target_hand_index=1),
        ]
        mask = encode_action_mask(actions)
        assert mask[97]   # PassSnap
        assert mask[100]  # SnapOwn(2)
        assert mask[105]  # SnapOpponent(1)
        assert mask.sum() == 3

    def test_out_of_range_action_skipped(self):
        """Actions with hand index >= MAX_HAND are silently skipped."""
        actions = [ActionReplace(target_hand_index=10)]
        mask = encode_action_mask(actions)
        assert mask.sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
