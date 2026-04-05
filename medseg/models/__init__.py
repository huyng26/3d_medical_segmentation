"""Model registry: 3D U-Net, SkipDenseNet3D, Swin UNETR."""
from __future__ import annotations


def build_model(model_name: str, num_classes: int, cfg: dict):
    """Instantiate a model by name using its config dict.

    Args:
        model_name:  One of ``unet3d``, ``skip_densenet3d``, ``swin_unetr``.
        num_classes: Number of output segmentation classes.
        cfg:         Model architecture config (loaded from ``configs/model_*.yaml``).

    Returns:
        torch.nn.Module
    """
    raise NotImplementedError
