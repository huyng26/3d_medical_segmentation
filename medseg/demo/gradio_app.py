"""Gradio viewer for 3-D medical segmentation labels.

This module is a **pure viewer** - it does not load any model weights or run
any inference.  It overlays the dataset label NIfTI that matches the uploaded
image filename.

Expected label layout:
    data/
      BTCV/
        imagesTs/img0003.nii.gz
        labelsTs/img0003.nii.gz
      MSD/
        Task02_Heart/
          imagesTs/...
          labelsTs/...
        Task09_Spleen/
          imagesTs/...
          labelsTs/...

UI flow:
  1. Upload the original NIfTI volume.
  2. Select dataset (/ MSD task when applicable).
  3. Click "Load label" - app finds the matching label by filename.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = str(PROJECT_ROOT / "out")

SUPPORTED_DATASETS = ("btcv", "msd")
SUPPORTED_MSD_TASKS = (2, 9)
LABEL_ROOT_BTCV_DIR = str(PROJECT_ROOT / "data" / "BTCV")
LABEL_ROOT_MSD_DIR = str(PROJECT_ROOT / "data" / "MSD")
LABEL_SPLITS = ("labelsTs", "labelsTr")

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


def _load_label_nifti_volume(path: str) -> np.ndarray:
    """Load a segmentation NIfTI and return an integer label map (D, H, W).

    This is for loading the original label NIfTIs that come with the datasets,
    which may be one-hot encoded (D, H, W, C) and need argmax to get class labels.
    """
    img = nib.load(path)
    arr = np.asarray(img.dataobj)
    if arr.ndim == 4:
        # Assume one-hot encoding in the last dimension; convert to label map.
        arr = np.argmax(arr, axis=-1)
    return arr.astype(np.int16)

# ---------------------------------------------------------------------------
# Dataset label lookup
# ---------------------------------------------------------------------------


def _msd_task_dir(msd_task: int) -> str:
    if msd_task == 2:
        return "Task02_Heart"
    if msd_task == 9:
        return "Task09_Spleen"
    raise ValueError(f"Unsupported MSD task: {msd_task}")


def _dataset_label_root(
    dataset: str,
    msd_task: int,
) -> str:
    if dataset == "btcv":
        return LABEL_ROOT_BTCV_DIR
    if dataset == "msd":
        return os.path.join(LABEL_ROOT_MSD_DIR, _msd_task_dir(msd_task))
    raise ValueError(f"Unsupported dataset: {dataset}")


def _label_path(
    dataset: str,
    msd_task: int,
    uploaded_path: str,
) -> str | None:
    """Return the matching dataset label path for the uploaded image, or None."""
    root = _dataset_label_root(dataset, msd_task)
    uploaded_name = Path(uploaded_path).name
    uploaded_stem = _stem(uploaded_path)

    candidates: list[str] = []
    for split in LABEL_SPLITS:
        label_dir = os.path.join(root, split)
        candidates.append(os.path.join(label_dir, uploaded_name))
        candidates.append(os.path.join(label_dir, f"{uploaded_stem}.nii.gz"))
        candidates.append(os.path.join(label_dir, f"{uploaded_stem}.nii"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


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
    """Load the original + label NIfTI and render an overlay figure.

    Args:
        nifti_path: Original input NIfTI (background greyscale image).
        seg_path:   Matching dataset label NIfTI.
        axis:       ``"axial"``, ``"sagittal"``, or ``"coronal"``.
        slice_idx:  Index of the slice to display.
    """
    orig_path = _resolve_uploaded_path(nifti_path)
    volume = _load_nifti_volume(orig_path)
    seg = _load_label_nifti_volume(seg_path)

    vol_slice, seg_slice, used_idx, max_idx = _slice_for_axis(volume, seg, axis, slice_idx)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(vol_slice, cmap="gray")
    axes[0].set_title(f"Input ({axis}, slice {used_idx}/{max_idx})")
    axes[0].axis("off")

    seg_masked = np.ma.masked_where(seg_slice == 0, seg_slice)
    axes[1].imshow(vol_slice, cmap="gray")
    axes[1].imshow(seg_masked, cmap="turbo", alpha=0.45)
    axes[1].set_title("Label Overlay")
    axes[1].axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def build_interface(out_dir: str = OUT_DIR) -> gr.Blocks:
    """Construct and return the Gradio Blocks interface."""

    initial_dataset = "btcv"
    initial_task = 2

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
        return gr.update(visible=is_msd, value=task)

    def _load_and_render(
        nifti_file,
        dataset_name,
        msd_task_value,
        axis_name,
        slice_value,
    ):
        """Find the matching label and render the first overlay.

        Returns ``(figure, label_path)`` so the path is stored in ``gr.State``
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
        found = _label_path(dataset_name, task, orig_path)
        if found is None:
            raise gr.Error(
                f"No label found for volume '{vol_stem}' "
                f"(dataset={dataset_name}, expected filename={Path(orig_path).name})."
            )

        fig = render_seg_slice(
            nifti_path=nifti_file,
            seg_path=found,
            axis=axis_name,
            slice_idx=int(slice_value),
        )
        return fig, found

    def _rerender_slice(nifti_file, seg_path, axis_name, slice_value):
        """Re-render from the cached label path without a filesystem lookup."""
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
            "Upload the original NIfTI volume, pick the dataset, then click "
            "**Load label**. The label is loaded from the matching dataset "
            "label folder using the uploaded filename."
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

        with gr.Row():
            axis_input = gr.Radio(
                choices=["axial", "coronal", "sagittal"],
                value="axial",
                label="Viewing axis",
            )
            slice_input = gr.Slider(
                minimum=0, maximum=256, value=128, step=1, label="Slice index"
            )

        load_button = gr.Button("Load label", variant="primary")
        out_plot = gr.Plot(label="Label overlay")

        # --- event wiring ---
        dataset_input.change(
            _update_dataset_controls,
            inputs=[dataset_input, msd_task_input],
            outputs=[msd_task_input],
        )
        nifti_input.change(
            _update_slider, inputs=[nifti_input, axis_input], outputs=[slice_input]
        )
        axis_input.change(
            _update_slider, inputs=[nifti_input, axis_input], outputs=[slice_input]
        )

        # Load once, store the label path in State.
        load_button.click(
            _load_and_render,
            inputs=[
                nifti_input,
                dataset_input,
                msd_task_input,
                axis_input,
                slice_input,
            ],
            outputs=[out_plot, seg_path_state],
        )

        # Re-render from the stored path without another filesystem lookup.
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
