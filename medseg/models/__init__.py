"""Model registry: 3D U-Net, SkipDenseNet3D, Swin UNETR."""
from __future__ import annotations
import argparse


def build_model(model_name: str, cfg: argparse.Namespace):
    """Construct and return the model described by *cfg*.

    Args:
        model_name: One of ``"unet3d"``, ``"attention_unet"``, ``"swin_unetr"``.
        cfg:        The parsed argparse Namespace returned by ``load_args()``.
    """
    num_classes = cfg.num_classes
    in_channels = cfg.in_channels

    if model_name == "unet3d":
        from .unet3d import UNet3D
        return UNet3D(
            in_channels=in_channels,
            out_channels=num_classes,
        )

    if model_name == "attention_unet":
        from .attention_unet import AttentionUNet
        return AttentionUNet(
            in_channels=in_channels,
            out_channels=num_classes,
        )

    if model_name == "swin_unetr":
        from .swin_unetr import SwinUNETRWrapper
        model = SwinUNETRWrapper(
            in_channels=in_channels,
            out_channels=num_classes,
            img_size=tuple(cfg.img_size),   
        )
        if cfg.pretrain:                    # honour --pretrain flag
            model.load_pretrained(cfg.pretrain)
        return model

    raise ValueError(f"Unsupported model name: {model_name!r}")
