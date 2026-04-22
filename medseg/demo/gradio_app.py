"""Gradio viewer for pre-computed 3-D segmentation results (Phase 6).

This module is a **pure viewer** — it does not load any model weights or run
any inference.  All segmentation NIfTI files must have been produced
beforehand by running ``predict.py``.

Expected output layout written by ``predict.py``:
    out/
      btcv/
        unet3d/        case_001_seg.nii.gz
        attention_unet/
        swin_unetr/
      msd_task2/
        unet3d/
        ...
      msd_task9/
        ...

UI flow:
  1. Upload the original NIfTI volume.
  2. Select dataset / model (/ MSD task when applicable).
  3. Click "Load segmentation" — app finds the matching pre-computed file.
  4. Choose viewing axis and slice index to browse the overlay.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import gradio as gr

OUT_DIR = os.path.abspath("./out")

SUPPORTED_MODELS = ("unet3d", "attention_unet", "swin_unetr")
SUPPORTED_DATASETS = ("btcv", "msd")
SUPPORTED_MSD_TASKS = (2, 9)


# ---------------------------------------------------------------------------
# NIfTI helpers
# ---------------------------------------------------------------------------

def _resolve_uploaded_path(nifti_path: Any) -> str:
    if isinstance(nifti_path, str):
        return nifti_path
    if isinstance(nifti_path, dict) and "name" in nifti_path:
        return str(nifti_path["name"])
    named = getattr(nifti_path, "name", None)
    if named is not None:
        return str(named)
    raise ValueError("Invalid uploaded file. Please upload a .nii or .nii.gz file.")


def _validate_nifti_path(path: str) -> None:
    lower = path.lower()
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        return
    raise ValueError("Unsupported file type. Please upload a .nii or .nii.gz file.")


def _stem(path: str) -> str:
    """Return the filename stem, stripping .nii.gz or .nii."""
    name = Path(path).name
    for ext in (".nii.gz", ".nii"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def _load_nifti_volume(path: str) -> np.ndarray:
    """Load a NIfTI file with nibabel and return a 3-D array (D, H, W)."""
    img = nib.load(path)
    vol = np.asarray(img.dataobj)
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = vol[..., 0]
    return vol


def _load_seg_volume(path: str) -> np.ndarray:
    """Load a segmentation NIfTI and return an integer label map (D, H, W).

    ``predict.py`` saves integer label maps directly, so no argmax is needed.
    The volume is (D, H, W) or occasionally (D, H, W, 1) if saved with a
    trailing dimension.
    """
    img = nib.load(path)
    arr = np.asarray(img.dataobj)
    # Squeeze any trailing singleton dimensions.
    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.astype(np.int16)


# ---------------------------------------------------------------------------
# Output-cache lookup
# ---------------------------------------------------------------------------

def _dataset_key(dataset: str, msd_task: int) -> str:
    return "btcv" if dataset == "btcv" else f"msd_task{msd_task}"


def _seg_path(
    out_dir: str,
    dataset: str,
    model_name: str,
    msd_task: int,
    volume_stem: str,
) -> str | None:
    """Return the path to the pre-computed segmentation NIfTI, or None."""
    ds_key = _dataset_key(dataset, msd_task)
    candidate = os.path.join(out_dir, ds_key, model_name, f"{volume_stem}_seg.nii.gz")
    if os.path.exists(candidate):
        return candidate
    # .nii fallback
    candidate_nii = os.path.join(out_dir, ds_key, model_name, f"{volume_stem}_seg.nii")
    if os.path.exists(candidate_nii):
        return candidate_nii
    return None


def _available_models_for_cache(
    out_dir: str,
    dataset: str,
    msd_task: int,
) -> list[str]:
    """Return model names that have at least one result in the cache."""
    ds_key = _dataset_key(dataset, msd_task)
    ds_dir = os.path.join(out_dir, ds_key)
    if not os.path.isdir(ds_dir):
        return []
    available = []
    for model_name in SUPPORTED_MODELS:
        model_dir = os.path.join(ds_dir, model_name)
        if os.path.isdir(model_dir) and any(
            f.endswith("_seg.nii.gz") or f.endswith("_seg.nii")
            for f in os.listdir(model_dir)
        ):
            available.append(model_name)
    return available


# ---------------------------------------------------------------------------
# Slice helpers
# ---------------------------------------------------------------------------

def _slice_for_axis(
    volume: np.ndarray,
    seg: np.ndarray,
    axis: str,
    slice_idx: int,
):
    if axis == "axial":
        max_idx = volume.shape[0] - 1
        idx = int(np.clip(slice_idx, 0, max_idx))
        return volume[idx, :, :], seg[idx, :, :], idx, max_idx
    if axis == "coronal":
        max_idx = volume.shape[1] - 1
        idx = int(np.clip(slice_idx, 0, max_idx))
        return volume[:, idx, :], seg[:, idx, :], idx, max_idx
    if axis == "sagittal":
        max_idx = volume.shape[2] - 1
        idx = int(np.clip(slice_idx, 0, max_idx))
        return volume[:, :, idx], seg[:, :, idx], idx, max_idx
    raise ValueError("axis must be one of: axial, coronal, sagittal")


def _uploaded_max_slice(nifti_path: Any, axis: str) -> int:
    """Return the maximum valid slice index for *axis* from the raw NIfTI."""
    path = _resolve_uploaded_path(nifti_path)
    _validate_nifti_path(path)
    vol = _load_nifti_volume(path)
    if axis == "axial":
        return int(vol.shape[0] - 1)
    if axis == "coronal":
        return int(vol.shape[1] - 1)
    if axis == "sagittal":
        return int(vol.shape[2] - 1)
    return 0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_seg_slice(
    nifti_path: Any,
    seg_path: str,
    axis: str,
    slice_idx: int,
) -> plt.Figure:
    """Load the original + segmentation NIfTI and render an overlay figure.

    Args:
        nifti_path: Original input NIfTI (background greyscale image).
        seg_path:   Pre-computed segmentation NIfTI from ``predict.py``.
        axis:       ``"axial"``, ``"sagittal"``, or ``"coronal"``.
        slice_idx:  Index of the slice to display.
    """
    orig_path = _resolve_uploaded_path(nifti_path)
    volume = _load_nifti_volume(orig_path)
    seg = _load_seg_volume(seg_path)

    vol_slice, seg_slice, used_idx, max_idx = _slice_for_axis(volume, seg, axis, slice_idx)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(vol_slice, cmap="gray")
    axes[0].set_title(f"Input ({axis}, slice {used_idx}/{max_idx})")
    axes[0].axis("off")

    seg_masked = np.ma.masked_where(seg_slice == 0, seg_slice)
    axes[1].imshow(vol_slice, cmap="gray")
    axes[1].imshow(seg_masked, cmap="turbo", alpha=0.45)
    axes[1].set_title("Segmentation Overlay")
    axes[1].axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def build_interface(out_dir: str = OUT_DIR) -> gr.Blocks:
    """Construct and return the Gradio Blocks interface."""

    # Seed the model dropdown from whatever is already cached.
    initial_dataset = "btcv"
    initial_task = 2
    initial_models = _available_models_for_cache(out_dir, initial_dataset, initial_task)

    def _update_slider(nifti_file: Any, axis_name: str):
        if nifti_file is None:
            return gr.update(maximum=0, value=0)
        try:
            max_idx = _uploaded_max_slice(nifti_file, axis_name)
        except Exception:
            max_idx = 256
        return gr.update(maximum=max_idx, value=max_idx // 2)

    def _update_dataset_controls(dataset_name: str, msd_task_value: int):
        is_msd = dataset_name == "msd"
        task = int(msd_task_value)
        if task not in SUPPORTED_MSD_TASKS:
            task = SUPPORTED_MSD_TASKS[0]
        model_names = _available_models_for_cache(out_dir, dataset_name, task)
        if not model_names:
            model_names = list(SUPPORTED_MODELS)
        return (
            gr.update(visible=is_msd, value=task),
            gr.update(choices=model_names, value=model_names[0], interactive=True),
        )

    def _update_models_for_task(dataset_name: str, msd_task_value: int):
        task = int(msd_task_value)
        if task not in SUPPORTED_MSD_TASKS:
            task = SUPPORTED_MSD_TASKS[0]
        model_names = _available_models_for_cache(out_dir, dataset_name, task)
        if not model_names:
            model_names = list(SUPPORTED_MODELS)
        return gr.update(choices=model_names, value=model_names[0], interactive=True)

    def _load_and_render(
        nifti_file,
        dataset_name,
        msd_task_value,
        model_name,
        axis_name,
        slice_value,
    ):
        """Find the pre-computed segmentation and render the first overlay.

        Returns ``(figure, seg_path)`` so the path is stored in ``gr.State``
        for subsequent axis/slice re-renders without hitting the filesystem again.
        """
        if nifti_file is None:
            raise gr.Error("Please upload a NIfTI volume first.")

        try:
            orig_path = _resolve_uploaded_path(nifti_file)
            _validate_nifti_path(orig_path)
        except ValueError as exc:
            raise gr.Error(str(exc))

        task = int(msd_task_value)
        if dataset_name != "msd":
            task = SUPPORTED_MSD_TASKS[0]

        vol_stem = _stem(orig_path)
        found = _seg_path(out_dir, dataset_name, model_name, task, vol_stem)
        if found is None:
            raise gr.Error(
                f"No pre-computed segmentation found for volume '{vol_stem}' "
                f"(dataset={dataset_name}, model={model_name}). "
                "Run predict.py first."
            )

        fig = render_seg_slice(
            nifti_path=nifti_file,
            seg_path=found,
            axis=axis_name,
            slice_idx=int(slice_value),
        )
        return fig, found

    def _rerender_slice(nifti_file, seg_path, axis_name, slice_value):
        """Re-render from the cached seg path — no filesystem lookup needed."""
        if seg_path is None or nifti_file is None:
            return None
        try:
            return render_seg_slice(
                nifti_path=nifti_file,
                seg_path=seg_path,
                axis=axis_name,
                slice_idx=int(slice_value),
            )
        except Exception as exc:
            raise gr.Error(str(exc))

    with gr.Blocks(title="3D Medical Segmentation Viewer") as demo:
        gr.Markdown("## 3D Medical Segmentation Viewer")
        gr.Markdown(
            "Upload the original NIfTI volume, pick the dataset and model used "
            "to produce the segmentation, then click **Load segmentation**. "
            "Segmentations must be pre-computed with `predict.py`."
        )

        seg_path_state = gr.State(value=None)

        with gr.Row():
            nifti_input = gr.File(
                label="Original NIfTI volume",
                file_types=[".nii", ".gz"],
                type="filepath",
            )
            dataset_input = gr.Dropdown(
                choices=list(SUPPORTED_DATASETS),
                value=initial_dataset,
                label="Dataset",
            )
            msd_task_input = gr.Dropdown(
                choices=list(SUPPORTED_MSD_TASKS),
                value=initial_task,
                label="MSD Task",
                visible=False,
            )
            model_input = gr.Dropdown(
                choices=initial_models if initial_models else list(SUPPORTED_MODELS),
                value=(initial_models[0] if initial_models else SUPPORTED_MODELS[0]),
                label="Model",
            )

        with gr.Row():
            axis_input = gr.Radio(
                choices=["axial", "coronal", "sagittal"],
                value="axial",
                label="Viewing axis",
            )
            slice_input = gr.Slider(
                minimum=0, maximum=256, value=128, step=1, label="Slice index"
            )

        load_button = gr.Button("Load segmentation", variant="primary")
        out_plot = gr.Plot(label="Segmentation overlay")

        # --- event wiring ---
        dataset_input.change(
            _update_dataset_controls,
            inputs=[dataset_input, msd_task_input],
            outputs=[msd_task_input, model_input],
        )
        msd_task_input.change(
            _update_models_for_task,
            inputs=[dataset_input, msd_task_input],
            outputs=[model_input],
        )
        nifti_input.change(
            _update_slider, inputs=[nifti_input, axis_input], outputs=[slice_input]
        )
        axis_input.change(
            _update_slider, inputs=[nifti_input, axis_input], outputs=[slice_input]
        )

        # Load once, store seg path in State.
        load_button.click(
            _load_and_render,
            inputs=[
                nifti_input,
                dataset_input,
                msd_task_input,
                model_input,
                axis_input,
                slice_input,
            ],
            outputs=[out_plot, seg_path_state],
        )

        # Re-render from the stored path — no file-system lookup.
        axis_input.change(
            _rerender_slice,
            inputs=[nifti_input, seg_path_state, axis_input, slice_input],
            outputs=[out_plot],
        )
        slice_input.change(
            _rerender_slice,
            inputs=[nifti_input, seg_path_state, axis_input, slice_input],
            outputs=[out_plot],
        )

    return demo


def launch(out_dir: str = OUT_DIR, share: bool = False) -> None:
    """Build the interface and launch the Gradio server."""
    demo = build_interface(out_dir=out_dir)
    demo.launch(share=share)


if __name__ == "__main__":
    launch()
