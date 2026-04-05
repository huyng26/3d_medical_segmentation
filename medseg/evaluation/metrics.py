"""DSC, HD95, and IoU computation with per-class and aggregate reporting (Phase 5)."""
from __future__ import annotations

import numpy as np
import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> np.ndarray:
    """Compute per-class Dice Similarity Coefficient.

    Args:
        pred:        Integer label map (D, H, W) or (B, D, H, W).
        target:      Ground-truth integer label map, same shape as ``pred``.
        num_classes: Total number of classes including background.

    Returns:
        1-D NumPy array of length ``num_classes`` with per-class DSC values.
    """
    raise NotImplementedError


def hausdorff_distance_95(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Compute per-class 95th-percentile Hausdorff distance (HD95).

    Args:
        pred, target:   Integer label maps (D, H, W).
        num_classes:    Total number of classes.
        voxel_spacing:  Physical voxel size in mm for correct distance scaling.

    Returns:
        1-D NumPy array of HD95 values per class.
    """
    raise NotImplementedError


def iou_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> np.ndarray:
    """Compute per-class Intersection over Union.

    Returns:
        1-D NumPy array of IoU values per class.
    """
    raise NotImplementedError


def aggregate_metrics(per_volume_metrics: list[dict]) -> dict:
    """Compute mean ± std across all test volumes for each metric and class.

    Args:
        per_volume_metrics: List of dicts, each returned by ``evaluate_volume``.

    Returns:
        Dict with keys ``dsc_mean``, ``dsc_std``, ``hd95_mean``, ``hd95_std``,
        ``iou_mean``, ``iou_std``, each a NumPy array of length ``num_classes``.
    """
    raise NotImplementedError


def evaluate_volume(
    model,
    image: torch.Tensor,
    label: torch.Tensor,
    patch_size: tuple[int, int, int],
    num_classes: int,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: str = "cuda",
) -> dict:
    """Run inference on a single volume and return all metrics.

    Returns:
        Dict with keys ``dsc``, ``hd95``, ``iou`` (each a NumPy array per class).
    """
    raise NotImplementedError
