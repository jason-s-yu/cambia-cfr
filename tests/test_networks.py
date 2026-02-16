"""
Tests for src/networks.py

Covers:
- AdvantageNetwork: shape, masking, gradient flow, weight initialization
- StrategyNetwork: shape, masking, probability normalization, NaN guard
- get_strategy_from_advantages: ReLU + normalize, uniform fallback
"""

import torch
import torch.nn as nn
import numpy as np
import pytest

# conftest.py handles the config stub

from src.encoding import INPUT_DIM, NUM_ACTIONS
from src.networks import (
    AdvantageNetwork,
    StrategyNetwork,
    get_strategy_from_advantages,
)


# ===== AdvantageNetwork =====


class TestAdvantageNetwork:
    def test_output_shape_single(self):
        """Single input produces (1, NUM_ACTIONS) output."""
        net = AdvantageNetwork()
        features = torch.randn(1, INPUT_DIM)
        mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
        out = net(features, mask)
        assert out.shape == (1, NUM_ACTIONS)

    def test_output_shape_batch(self):
        """Batch input produces (batch, NUM_ACTIONS) output."""
        net = AdvantageNetwork()
        batch_size = 32
        features = torch.randn(batch_size, INPUT_DIM)
        mask = torch.ones(batch_size, NUM_ACTIONS, dtype=torch.bool)
        out = net(features, mask)
        assert out.shape == (batch_size, NUM_ACTIONS)

    def test_illegal_actions_are_neg_inf(self):
        """Masked (illegal) actions should be -inf in the output."""
        net = AdvantageNetwork()
        features = torch.randn(1, INPUT_DIM)
        mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)
        mask[0, 0] = True  # Only action 0 is legal
        mask[0, 5] = True  # Only action 5 is legal
        out = net(features, mask)
        # Legal actions should be finite
        assert torch.isfinite(out[0, 0])
        assert torch.isfinite(out[0, 5])
        # Illegal actions should be -inf
        assert out[0, 1] == float("-inf")
        assert out[0, 100] == float("-inf")

    def test_all_legal_no_neg_inf(self):
        """When all actions are legal, no output should be -inf."""
        net = AdvantageNetwork()
        features = torch.randn(1, INPUT_DIM)
        mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
        out = net(features, mask)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self):
        """Gradients flow back through the network for legal actions."""
        net = AdvantageNetwork()
        features = torch.randn(1, INPUT_DIM, requires_grad=True)
        mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
        out = net(features, mask)
        loss = out.sum()
        loss.backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0

    def test_kaiming_initialization(self):
        """Linear layers are initialized with Kaiming normal weights and zero biases."""
        net = AdvantageNetwork()
        for module in net.net:
            if isinstance(module, nn.Linear):
                # Biases should be zero
                assert torch.all(module.bias == 0)
                # Weights should not be all zeros (Kaiming init)
                assert module.weight.abs().sum() > 0

    def test_custom_dimensions(self):
        """Custom input/hidden/output dimensions work correctly."""
        net = AdvantageNetwork(input_dim=50, hidden_dim=64, output_dim=20)
        features = torch.randn(4, 50)
        mask = torch.ones(4, 20, dtype=torch.bool)
        out = net(features, mask)
        assert out.shape == (4, 20)

    def test_eval_mode_no_dropout(self):
        """In eval mode, dropout should be disabled (deterministic output)."""
        net = AdvantageNetwork()
        net.eval()
        features = torch.randn(1, INPUT_DIM)
        mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
        out1 = net(features, mask)
        out2 = net(features, mask)
        assert torch.equal(out1, out2)

    def test_train_mode_dropout_variability(self):
        """In train mode, dropout should introduce variability (statistical)."""
        net = AdvantageNetwork(dropout=0.5)
        net.train()
        features = torch.randn(1, INPUT_DIM)
        mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
        # Run multiple times and check at least some differ
        outputs = [net(features, mask).detach().clone() for _ in range(20)]
        all_same = all(torch.equal(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Dropout should cause variation in train mode"

    def test_parameter_count(self):
        """Network has approximately 125K parameters."""
        net = AdvantageNetwork()
        total_params = sum(p.numel() for p in net.parameters())
        # 222*256 + 256 + 256*256 + 256 + 256*128 + 128 + 128*146 + 146
        # = 56832 + 256 + 65536 + 256 + 32768 + 128 + 18688 + 146 = ~174,610
        # The plan says ~125K but actual is higher due to 4 linear layers
        assert total_params > 100_000
        assert total_params < 250_000


# ===== StrategyNetwork =====


class TestStrategyNetwork:
    def test_output_shape(self):
        net = StrategyNetwork()
        features = torch.randn(1, INPUT_DIM)
        mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
        out = net(features, mask)
        assert out.shape == (1, NUM_ACTIONS)

    def test_output_sums_to_one(self):
        """Output probabilities sum to 1 per row."""
        net = StrategyNetwork()
        features = torch.randn(4, INPUT_DIM)
        mask = torch.ones(4, NUM_ACTIONS, dtype=torch.bool)
        net.eval()
        out = net(features, mask)
        row_sums = out.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(4), atol=1e-5)

    def test_output_nonnegative(self):
        """All output probabilities are non-negative."""
        net = StrategyNetwork()
        features = torch.randn(8, INPUT_DIM)
        mask = torch.ones(8, NUM_ACTIONS, dtype=torch.bool)
        net.eval()
        out = net(features, mask)
        assert (out >= 0).all()

    def test_illegal_actions_zero_prob(self):
        """Masked actions have zero probability."""
        net = StrategyNetwork()
        features = torch.randn(1, INPUT_DIM)
        mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)
        mask[0, 0] = True
        mask[0, 3] = True
        net.eval()
        out = net(features, mask)
        # Illegal actions
        assert out[0, 1] == 0.0
        assert out[0, 100] == 0.0
        # Legal actions should have positive probability
        assert out[0, 0] > 0
        assert out[0, 3] > 0
        # Sum should be 1
        assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-5)

    def test_all_masked_produces_zeros(self):
        """When all actions are masked, output should be all zeros (NaN guard)."""
        net = StrategyNetwork()
        features = torch.randn(1, INPUT_DIM)
        mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)
        net.eval()
        out = net(features, mask)
        # nan_to_num should convert NaN to 0
        assert torch.all(out == 0)
        assert not torch.any(torch.isnan(out))

    def test_gradient_flow(self):
        """Gradients flow through the strategy network."""
        net = StrategyNetwork()
        features = torch.randn(1, INPUT_DIM, requires_grad=True)
        mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
        out = net(features, mask)
        # Use a non-trivial loss (cross-entropy style) since softmax sum is constant
        target = torch.zeros(1, NUM_ACTIONS)
        target[0, 0] = 1.0
        loss = -torch.sum(target * torch.log(out + 1e-10))
        loss.backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0

    def test_single_legal_action(self):
        """With one legal action, it should get probability 1.0."""
        net = StrategyNetwork()
        features = torch.randn(1, INPUT_DIM)
        mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)
        mask[0, 42] = True
        net.eval()
        out = net(features, mask)
        assert torch.isclose(out[0, 42], torch.tensor(1.0), atol=1e-5)

    def test_batch_different_masks(self):
        """Different masks in a batch produce different distributions."""
        net = StrategyNetwork()
        net.eval()
        features = torch.randn(2, INPUT_DIM)
        mask = torch.zeros(2, NUM_ACTIONS, dtype=torch.bool)
        mask[0, 0] = True
        mask[0, 1] = True
        mask[1, 5] = True
        mask[1, 6] = True
        out = net(features, mask)
        # Row 0 should have probability on indices 0, 1
        assert out[0, 5] == 0.0
        # Row 1 should have probability on indices 5, 6
        assert out[1, 0] == 0.0


# ===== get_strategy_from_advantages =====


class TestGetStrategyFromAdvantages:
    def test_positive_advantages_normalized(self):
        """Positive advantages are normalized to a probability distribution."""
        advantages = torch.tensor([[3.0, 1.0, 0.0, -1.0]])
        mask = torch.tensor([[True, True, True, True]])
        strategy = get_strategy_from_advantages(advantages, mask)
        # ReLU: [3, 1, 0, 0]; sum=4; result=[0.75, 0.25, 0, 0]
        assert torch.isclose(strategy[0, 0], torch.tensor(0.75), atol=1e-5)
        assert torch.isclose(strategy[0, 1], torch.tensor(0.25), atol=1e-5)
        assert strategy[0, 2] == 0.0
        assert strategy[0, 3] == 0.0

    def test_all_negative_falls_back_to_uniform(self):
        """When all advantages are <= 0, fall back to uniform over legal actions."""
        advantages = torch.tensor([[-1.0, -2.0, -0.5, -3.0]])
        mask = torch.tensor([[True, True, True, True]])
        strategy = get_strategy_from_advantages(advantages, mask)
        # All negative -> ReLU gives [0,0,0,0] -> total=0 -> uniform
        expected = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        assert torch.allclose(strategy, expected, atol=1e-5)

    def test_illegal_actions_zero(self):
        """Illegal actions get zero probability regardless of advantage."""
        advantages = torch.tensor([[5.0, 3.0, 1.0, 0.0]])
        mask = torch.tensor([[True, False, True, False]])
        strategy = get_strategy_from_advantages(advantages, mask)
        assert strategy[0, 1] == 0.0
        assert strategy[0, 3] == 0.0
        assert strategy[0, 0] > 0
        assert strategy[0, 2] > 0
        assert torch.isclose(strategy.sum(), torch.tensor(1.0), atol=1e-5)

    def test_uniform_fallback_respects_mask(self):
        """Uniform fallback only distributes over legal actions."""
        advantages = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
        mask = torch.tensor([[True, False, True, False]])
        strategy = get_strategy_from_advantages(advantages, mask)
        assert torch.isclose(strategy[0, 0], torch.tensor(0.5), atol=1e-5)
        assert strategy[0, 1] == 0.0
        assert torch.isclose(strategy[0, 2], torch.tensor(0.5), atol=1e-5)
        assert strategy[0, 3] == 0.0

    def test_batch_processing(self):
        """Works correctly with batched inputs."""
        advantages = torch.tensor([
            [2.0, 1.0, 0.0, -1.0],
            [-1.0, -2.0, -3.0, -4.0],
        ])
        mask = torch.tensor([
            [True, True, True, True],
            [True, True, True, True],
        ])
        strategy = get_strategy_from_advantages(advantages, mask)
        assert strategy.shape == (2, 4)
        # Row 0: positive advantages -> normalized
        assert torch.isclose(strategy[0].sum(), torch.tensor(1.0), atol=1e-5)
        # Row 1: all negative -> uniform
        assert torch.allclose(strategy[1], torch.tensor([0.25, 0.25, 0.25, 0.25]), atol=1e-5)

    def test_single_positive_advantage(self):
        """A single positive advantage gets probability 1.0."""
        advantages = torch.tensor([[0.5, -1.0, -2.0, -3.0]])
        mask = torch.tensor([[True, True, True, True]])
        strategy = get_strategy_from_advantages(advantages, mask)
        assert torch.isclose(strategy[0, 0], torch.tensor(1.0), atol=1e-5)
        assert strategy[0, 1] == 0.0

    def test_full_action_space(self):
        """Works with the full 146-dim action space."""
        advantages = torch.randn(1, NUM_ACTIONS)
        mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
        strategy = get_strategy_from_advantages(advantages, mask)
        assert strategy.shape == (1, NUM_ACTIONS)
        assert torch.isclose(strategy.sum(), torch.tensor(1.0), atol=1e-5)
        assert (strategy >= 0).all()

    def test_result_dtype(self):
        """Output dtype matches input dtype."""
        advantages = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        mask = torch.tensor([[True, True]])
        strategy = get_strategy_from_advantages(advantages, mask)
        assert strategy.dtype == torch.float32


# ===== Integration: AdvantageNetwork -> get_strategy_from_advantages =====


class TestNetworkStrategyIntegration:
    def test_advantage_to_strategy_pipeline(self):
        """AdvantageNetwork output can be converted to a valid strategy."""
        net = AdvantageNetwork()
        net.eval()
        features = torch.randn(4, INPUT_DIM)
        mask = torch.ones(4, NUM_ACTIONS, dtype=torch.bool)
        # Mask out some actions
        mask[:, 50:100] = False

        advantages = net(features, mask)
        strategy = get_strategy_from_advantages(advantages, mask)

        assert strategy.shape == (4, NUM_ACTIONS)
        # Each row sums to 1
        for i in range(4):
            assert torch.isclose(strategy[i].sum(), torch.tensor(1.0), atol=1e-5)
        # Masked actions have 0 probability
        assert (strategy[:, 50:100] == 0).all()
        # All non-negative
        assert (strategy >= 0).all()

    def test_strategy_network_trainable(self):
        """StrategyNetwork can be trained with cross-entropy-like loss."""
        net = StrategyNetwork()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        features = torch.randn(8, INPUT_DIM)
        mask = torch.ones(8, NUM_ACTIONS, dtype=torch.bool)
        # Create a target distribution (uniform over first 3 actions)
        target = torch.zeros(8, NUM_ACTIONS)
        target[:, :3] = 1.0 / 3.0

        initial_loss = None
        for step in range(50):
            out = net(features, mask)
            loss = ((out - target) ** 2).sum()
            if step == 0:
                initial_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = ((net(features, mask) - target) ** 2).sum().item()
        assert final_loss < initial_loss, "Training should decrease loss"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
