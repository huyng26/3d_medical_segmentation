"""Full-volume inference via sliding-window stitching (Phase 5).

Uses ``monai.inferers.sliding_window_inference`` with Gaussian importance
weighting at patch borders. Optional connected-component post-processing
to remove small spurious predicted regions.
"""
from __future__ import annotations

from pathlib import Path

import torch


def predict_volume(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: tuple[int, int, int],
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Run sliding-window inference on a single volume.

    Args:
        model:          Trained segmentation model (eval mode expected).
        image:          Input tensor of shape (1, 1, D, H, W).
        patch_size:     Spatial size of each patch (must match training).
        sw_batch_size:  Number of patches processed simultaneously.
        overlap:        Fractional overlap between adjacent windows.
        device:         Device to run inference on.

    Returns:
        Soft probability map of shape (1, C, D, H, W).
    """
    raise NotImplementedError


def postprocess(
    prob_map: torch.Tensor,
    min_connected_component_size: int = 0,
) -> torch.Tensor:
    """Apply argmax and optional connected-component filtering.

    Args:
        prob_map:                        Soft probability map (1, C, D, H, W).
        min_connected_component_size:    Remove components smaller than this
                                         voxel count (0 disables filtering).

    Returns:
        Integer label map of shape (1, D, H, W).
    """
    raise NotImplementedError


def load_checkpoint(checkpoint_path: str | Path, model: torch.nn.Module, device) -> torch.nn.Module:
    """Load a saved checkpoint into ``model`` and return it in eval mode."""
    raise NotImplementedError
