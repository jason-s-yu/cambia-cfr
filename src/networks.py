"""
src/networks.py

PyTorch neural network modules for Deep CFR.

AdvantageNetwork: Predicts per-action advantage/regret values.
StrategyNetwork: Predicts per-action strategy probabilities.

Architecture (shared):
  Input(222) -> Linear(256) -> ReLU -> Dropout(0.1)
             -> Linear(256) -> ReLU -> Dropout(0.1)
             -> Linear(128) -> ReLU -> Linear(146)

Total parameters: ~125K.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import INPUT_DIM, NUM_ACTIONS
from .cfr.exceptions import InvalidNetworkInputError


class AdvantageNetwork(nn.Module):
    """
    Predicts per-action advantage (regret) values for a given information set.

    Forward pass applies action masking: illegal actions are set to -inf.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 256,
        output_dim: int = NUM_ACTIONS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(
        self, features: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: (batch, input_dim) float tensor of encoded infoset features.
            action_mask: (batch, output_dim) bool tensor, True for legal actions.

        Returns:
            (batch, output_dim) float tensor of advantage values, -inf for illegal actions.

        Raises:
            InvalidNetworkInputError: If input shape is incorrect or contains NaN values.
        """
        # Validate input shape
        if features.dim() != 2 or features.shape[1] != self._input_dim:
            raise InvalidNetworkInputError(
                f"Invalid features shape: expected (batch, {self._input_dim}), got {tuple(features.shape)}"
            )
        if action_mask.dim() != 2 or action_mask.shape[1] != self._output_dim:
            raise InvalidNetworkInputError(
                f"Invalid action_mask shape: expected (batch, {self._output_dim}), got {tuple(action_mask.shape)}"
            )
        if features.shape[0] != action_mask.shape[0]:
            raise InvalidNetworkInputError(
                f"Batch size mismatch: features has {features.shape[0]}, action_mask has {action_mask.shape[0]}"
            )

        # Validate NaN
        if torch.isnan(features).any():
            raise InvalidNetworkInputError("Features tensor contains NaN values")

        out = self.net(features)
        out = out.masked_fill(~action_mask, float("-inf"))
        return out


class StrategyNetwork(nn.Module):
    """
    Predicts per-action strategy probabilities for a given information set.

    Forward pass applies action masking and softmax: illegal actions get 0 probability.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 256,
        output_dim: int = NUM_ACTIONS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(
        self, features: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: (batch, input_dim) float tensor of encoded infoset features.
            action_mask: (batch, output_dim) bool tensor, True for legal actions.

        Returns:
            (batch, output_dim) float tensor of strategy probabilities summing to 1
            per row. Illegal actions have probability 0.

        Raises:
            InvalidNetworkInputError: If input shape is incorrect or contains NaN values.
        """
        # Validate input shape
        if features.dim() != 2 or features.shape[1] != self._input_dim:
            raise InvalidNetworkInputError(
                f"Invalid features shape: expected (batch, {self._input_dim}), got {tuple(features.shape)}"
            )
        if action_mask.dim() != 2 or action_mask.shape[1] != self._output_dim:
            raise InvalidNetworkInputError(
                f"Invalid action_mask shape: expected (batch, {self._output_dim}), got {tuple(action_mask.shape)}"
            )
        if features.shape[0] != action_mask.shape[0]:
            raise InvalidNetworkInputError(
                f"Batch size mismatch: features has {features.shape[0]}, action_mask has {action_mask.shape[0]}"
            )

        # Validate NaN
        if torch.isnan(features).any():
            raise InvalidNetworkInputError("Features tensor contains NaN values")

        out = self.net(features)
        # Mask illegal actions to -inf before softmax
        out = out.masked_fill(~action_mask, float("-inf"))
        # Softmax over action dimension
        probs = F.softmax(out, dim=-1)
        # NaN guard: if all actions are masked (shouldn't happen), replace with 0
        probs = torch.nan_to_num(probs, nan=0.0)
        return probs


def get_strategy_from_advantages(
    advantages: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute a strategy from advantage predictions using ReLU + normalize (regret matching).

    This is used during traversal to convert the AdvantageNetwork output into
    a probability distribution, matching the RM+ approach from tabular CFR.

    Args:
        advantages: (batch, 146) or (146,) raw advantage values.
        action_mask: Same shape as advantages, bool tensor of legal actions.

    Returns:
        Probability distribution over actions, same shape as input.
        Illegal actions have probability 0.
        Falls back to uniform over legal actions if all advantages <= 0.
    """
    # ReLU: only positive advantages get probability mass
    positive = F.relu(advantages)
    # Mask illegal actions
    positive = positive * action_mask.float()
    # Normalize
    total = positive.sum(dim=-1, keepdim=True)
    # Check if any positive advantages exist
    has_positive = total > 0
    if has_positive.all():
        return positive / total

    # Fallback: uniform over legal actions where total == 0
    uniform = action_mask.float()
    uniform_total = uniform.sum(dim=-1, keepdim=True).clamp(min=1.0)
    uniform = uniform / uniform_total

    # Use normalized positive where available, uniform otherwise
    result = torch.where(has_positive, positive / total.clamp(min=1e-10), uniform)
    return result
