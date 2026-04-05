"""Gradio interface for interactive 3-D segmentation demo (Phase 6).

UI flow:
  1. User uploads a NIfTI file.
  2. User selects model (unet3d | skip_densenet3d | swin_unetr).
  3. User selects viewing axis (axial | sagittal | coronal) and slice index.
  4. App runs sliding-window inference and renders the segmentation overlay.

Models are loaded once at startup to minimise per-request latency.
Inference runs on GPU when available, CPU as fallback.
"""
from __future__ import annotations


def load_models(checkpoint_dir: str) -> dict:
    """Pre-load all three model checkpoints into memory.

    Args:
        checkpoint_dir: Directory containing ``unet3d.pth``,
                        ``skip_densenet3d.pth``, and ``swin_unetr.pth``.

    Returns:
        Dict mapping model name → ``torch.nn.Module`` (eval mode).
    """
    raise NotImplementedError


def segment_nifti(
    nifti_path: str,
    model_name: str,
    axis: str,
    slice_idx: int,
    models: dict,
) -> object:
    """Run inference and return a matplotlib figure with the overlay.

    Args:
        nifti_path:  Path to the uploaded NIfTI file.
        model_name:  One of ``unet3d``, ``skip_densenet3d``, ``swin_unetr``.
        axis:        Viewing plane — ``"axial"``, ``"sagittal"``, or ``"coronal"``.
        slice_idx:   Index of the slice to display.
        models:      Pre-loaded model dict from ``load_models``.

    Returns:
        A ``matplotlib.figure.Figure`` for Gradio to display.
    """
    raise NotImplementedError


def build_interface(models: dict):
    """Construct and return the ``gradio.Blocks`` interface object."""
    raise NotImplementedError


def launch(checkpoint_dir: str = "outputs/checkpoints", share: bool = False) -> None:
    """Load models and launch the Gradio server."""
    raise NotImplementedError
