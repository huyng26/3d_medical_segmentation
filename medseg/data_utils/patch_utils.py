"""Patch extraction utilities for training and sliding-window inference (Phases 2 & 5)."""
from __future__ import annotations

import numpy as np


def random_patch_coords(
    volume_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    """Sample a random top-left corner and return three slices for a patch."""
    raise NotImplementedError


def sliding_window_coords(
    volume_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    overlap: float = 0.5,
) -> list[tuple[slice, slice, slice]]:
    """Enumerate all patch positions for sliding-window inference.

    Args:
        volume_shape: Spatial dimensions of the full volume (D, H, W).
        patch_size:   Patch spatial size.
        overlap:      Fractional overlap between adjacent patches (0–1).

    Returns:
        List of (slice_d, slice_h, slice_w) tuples covering the full volume.
    """
    raise NotImplementedError


def gaussian_importance_map(patch_size: tuple[int, int, int]) -> np.ndarray:
    """Return a Gaussian weight map for soft stitching at patch borders.

    Matches the strategy used by ``monai.inferers.sliding_window_inference``.
    """
    raise NotImplementedError
