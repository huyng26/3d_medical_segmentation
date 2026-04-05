"""Combined Dice + cross-entropy loss (Phase 4)."""
from __future__ import annotations

import torch
import torch.nn as nn


class DiceCELoss(nn.Module):
    """50/50 weighted combination of soft Dice loss and cross-entropy.

    Args:
        num_classes:    Number of segmentation classes (including background).
        dice_weight:    Scalar weight for the Dice term (default 0.5).
        ce_weight:      Scalar weight for the CE term (default 0.5).
        softmax:        Apply softmax to logits before Dice computation.
        squared_denom:  Use squared denominator in Dice for numerical stability.
    """

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        softmax: bool = True,
        squared_denom: bool = True,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits:  Raw model output of shape (B, C, D, H, W).
            targets: Ground-truth integer labels of shape (B, D, H, W).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError
