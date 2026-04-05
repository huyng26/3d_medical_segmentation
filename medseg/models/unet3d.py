"""3D U-Net architecture (Phase 3).

Encoder-decoder with 3D convolutions, batch norm, ReLU, and skip connections.
Decoder uses trilinear upsampling or transposed convolutions (configurable).
"""
from __future__ import annotations

import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Two consecutive Conv3d → Norm → ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch") -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class EncoderBlock(nn.Module):
    """Downsampling encoder stage: ConvBlock3D followed by MaxPool3d."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch") -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class DecoderBlock(nn.Module):
    """Upsampling decoder stage with skip-connection concatenation."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        upsample_mode: str = "trilinear",
        norm: str = "batch",
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x, skip):
        raise NotImplementedError


class UNet3D(nn.Module):
    """Full 3D U-Net.

    Args:
        in_channels:    Number of input image channels (typically 1 for CT).
        out_channels:   Number of segmentation classes.
        features:       Feature map counts at each encoder depth.
        norm:           Normalisation layer type — ``"batch"`` or ``"instance"``.
        upsample_mode:  ``"trilinear"`` or ``"transposed_conv"``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 14,
        features: list[int] | None = None,
        norm: str = "batch",
        upsample_mode: str = "trilinear",
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
