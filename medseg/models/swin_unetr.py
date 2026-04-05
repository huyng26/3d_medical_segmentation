"""Swin UNETR wrapper around ``monai.networks.nets.SwinUNETR`` (Phase 3).

Fine-tunes the pretrained Swin Transformer encoder with a CNN decoder.
Optionally loads publicly available pretrained weights at construction time.
"""
from __future__ import annotations

from pathlib import Path

import torch.nn as nn


class SwinUNETRWrapper(nn.Module):
    """Thin wrapper that constructs MONAI's SwinUNETR and optionally loads weights.

    Args:
        in_channels:      Number of input channels (1 for CT).
        out_channels:     Number of segmentation classes.
        img_size:         Patch spatial size the model expects (must match training patches).
        feature_size:     Swin embedding dimension (MONAI default 48).
        use_checkpoint:   Enable gradient checkpointing to reduce VRAM usage.
        weights_path:     Path or URL to pretrained weights; ``None`` to skip.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 14,
        img_size: tuple[int, int, int] = (128, 128, 64),
        feature_size: int = 48,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = True,
        weights_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def load_pretrained(self, weights_path: str | Path) -> None:
        """Load pretrained Swin encoder weights (partial weight loading)."""
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
