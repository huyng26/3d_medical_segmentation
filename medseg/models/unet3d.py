"""3D U-Net architecture (Phase 3).

Encoder-decoder with 3D convolutions, batch norm, ReLU, and skip connections.
Decoder uses trilinear upsampling or transposed convolutions (configurable).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Two consecutive Conv3d → Norm → ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch") -> None:
        super().__init__()
        if norm == "batch":
            norm_layer = nn.BatchNorm3d
        elif norm == "instance":
            norm_layer = nn.InstanceNorm3d
        else:
            raise ValueError(f"Unsupported norm: {norm!r}. Expected 'batch' or 'instance'.")

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """Downsampling encoder stage: ConvBlock3D followed by MaxPool3d."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch") -> None:
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels, norm=norm)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return x, self.pool(x)


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
        if upsample_mode == "transposed_conv":
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
            self.proj = nn.Identity()
        elif upsample_mode == "trilinear":
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            raise ValueError(
                f"Unsupported upsample_mode: {upsample_mode!r}. "
                "Expected 'trilinear' or 'transposed_conv'."
            )

        self.conv = ConvBlock3D(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            norm=norm,
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = self.proj(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


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
        features = features or [32, 64, 128, 256, 320]
        if len(features) < 2:
            raise ValueError("`features` must contain at least 2 values.")

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for out_ch in features[:-1]:
            self.encoders.append(EncoderBlock(prev_channels, out_ch, norm=norm))
            prev_channels = out_ch

        self.bottleneck = ConvBlock3D(features[-2], features[-1], norm=norm)

        self.decoders = nn.ModuleList()
        dec_in = features[-1]
        for skip_ch in reversed(features[:-1]):
            self.decoders.append(
                DecoderBlock(
                    in_channels=dec_in,
                    skip_channels=skip_ch,
                    out_channels=skip_ch,
                    upsample_mode=upsample_mode,
                    norm=norm,
                )
            )
            dec_in = skip_ch

        self.head = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.head(x)
