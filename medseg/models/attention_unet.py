from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class ConvBlock3D(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate3D(nn.Module):
    def __init__(
        self,
        F_x: int,
        F_g: int,
        inter_channels: int | None = None,
    ) -> None:
        super().__init__()
        inter_channels = inter_channels or max(F_x // 2, 1)

        # 1×1×1 projections
        self.theta_x = nn.Conv3d(F_x, inter_channels, kernel_size=1, bias=False)
        self.phi_g   = nn.Conv3d(F_g, inter_channels, kernel_size=1, bias=False)

        # Collapse to scalar attention map
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)
        self.bn   = nn.BatchNorm3d(F_x)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Apply attention gate.

        Args:
            x: Skip-connection features  ``(B, F_x, D,  H,  W)``.
            g: Gating signal from decoder ``(B, F_g, D', H', W')``.

        Returns:
            Attention-weighted features  ``(B, F_x, D,  H,  W)``.
        """
        theta = self.theta_x(x)                             # (B, C_int, D, H, W)

        # Upsample gating signal to match skip-connection spatial dims
        phi = F.interpolate(
            self.phi_g(g),
            size=theta.shape[2:],
            mode="trilinear",
            align_corners=False,
        )                                                   # (B, C_int, D, H, W)

        alpha = self.psi(self.relu(theta + phi))            # (B, 1, D, H, W)
        return self.bn(alpha * x)                           # (B, F_x, D, H, W)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch") -> None:
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels, norm=norm)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        return feat, self.pool(feat)   # (skip, pooled)


class DecoderBlock(nn.Module):
    """Upsampling decoder stage with attention-gated skip connection.

    The skip-connection features are filtered through an ``AttentionGate3D``
    (using the upsampled decoder signal as a gating cue) before being
    concatenated with the upsampled decoder features.

    Args:
        in_channels:    Channels coming from the previous decoder stage.
        skip_channels:  Channels of the encoder skip-connection.
        out_channels:   Channels produced by this block.
        upsample_mode:  ``"trilinear"`` (default) or ``"transposed_conv"``.
        norm:           ``"batch"`` or ``"instance"``.
    """

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
            self.up   = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
            self.proj = nn.Identity()
        elif upsample_mode == "trilinear":
            self.up   = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            raise ValueError(
                f"Unsupported upsample_mode: {upsample_mode!r}. "
                "Expected 'trilinear' or 'transposed_conv'."
            )

        self.attn = AttentionGate3D(F_x=skip_channels, F_g=out_channels)
        self.conv = ConvBlock3D(out_channels + skip_channels, out_channels, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.up(x))

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

        skip = self.attn(skip, g=x)
        x    = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """3-D Attention U-Net (Oktay et al., 2018).

    Standard encoder-decoder where every skip connection is filtered by an
    additive soft-attention gate before being merged with the decoder path.
    All convolutions are 3-D so the model accepts volumetric inputs directly.

    Args:
        in_channels:   Number of input image channels (1 for CT/MRI).
        out_channels:  Number of segmentation classes.
        features:      Channel counts at each encoder depth.
                       Defaults to ``[32, 64, 128, 256, 320]``.
        norm:          Normalisation layer — ``"batch"`` or ``"instance"``.
        upsample_mode: ``"trilinear"`` (default) or ``"transposed_conv"``.

    Example::

        model = AttentionUNet(in_channels=1, out_channels=14)
        x     = torch.randn(1, 1, 128, 128, 64)
        logits = model(x)   # → (1, 14, 128, 128, 64)
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

        self.encoders: nn.ModuleList = nn.ModuleList()
        prev_ch = in_channels
        for out_ch in features[:-1]:
            self.encoders.append(EncoderBlock(prev_ch, out_ch, norm=norm))
            prev_ch = out_ch

        self.bottleneck = ConvBlock3D(features[-2], features[-1], norm=norm)

        self.decoders: nn.ModuleList = nn.ModuleList()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input volume of shape ``(B, C, D, H, W)``.

        Returns:
            Class logits of shape ``(B, out_channels, D, H, W)``.
        """
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.head(x)

    def load_pretrained_weights(self, pretrained_path: str) -> None:
        """Load pretrained weights from a checkpoint file."""
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Checkpoint file not found: {pretrained_path}")
        
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        
        if isinstance(checkpoint, dict) and "module." in next(iter(checkpoint.keys())):
            state_dict = {k.replace("module.", ""):
                       v for k, v in checkpoint.items()}
        else:
            state_dict = checkpoint

        model_dict = self.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items()
                        if k in model_dict and v.shape == model_dict[k].shape}

        missing = set(model_dict.keys()) - set(filtered_dict.keys())
        if missing:
            print(f"Missing weights for: {missing}")

        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict)

        print(f"Loaded weights from: {pretrained_path}")
        print(f"Loaded {len(filtered_dict)} / {len(model_dict)} weights")


if __name__ == "__main__":
    model = AttentionUNet(in_channels=1, out_channels=14)
    x = torch.randn(1, 1, 96, 96, 96)
    out = model(x)
    print(out.shape)

    print("Number of parameters:", sum(p.numel() for p in model.parameters()))