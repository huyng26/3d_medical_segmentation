"""Gradio interface for interactive 3-D segmentation demo (Phase 6).

UI flow:
  1. User uploads a NIfTI file.
  2. User selects model (unet3d | attention_unet | swin_unetr), dataset, and axis/slice.
  3. Click "Run Segmentation" — sliding-window inference runs once; the result
     NIfTI is written to OUT_DIR by SaveImaged (post-transform).
  4. The saved segmentation NIfTI is loaded and the overlay is rendered.
  5. Changing axis or slice re-renders from the saved file instantly — no
     further inference is needed.

Models are loaded once at startup to minimise per-request latency.
Inference runs on GPU when available, CPU as fallback.
"""
from __future__ import annotations
import os
import glob
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Any
import nibabel as nib
from monai.inferers.utils import sliding_window_inference
from medseg.models import build_model
from monai.data import Dataset, DataLoader, decollate_batch
from medseg.data_utils.transforms import build_msd_inference_transforms, build_btcv_inference_transforms
import matplotlib.pyplot as plt
import gradio as gr

OUT_DIR = os.path.abspath("./out")

SUPPORTED_MODELS = ("unet3d", "attention_unet", "swin_unetr")
SUPPORTED_DATASETS = ("btcv", "msd")
SUPPORTED_MSD_TASKS = (2, 9)

def _checkpoint_path(args: argparse.Namespace, model_name: str) -> str:
    if args.dataset == "msd":
        return os.path.join(args.save_dir, f"{model_name}_msd_task{args.msd_task}_best_model.pth")
    return os.path.join(args.save_dir, f"{model_name}_{args.dataset}_best_model.pth")


def _default_demo_args() -> argparse.Namespace:
    # Keep demo startup simple for `python app.py` while allowing overrides later.
    return argparse.Namespace(
        dataset="btcv",
        msd_task=2,
        model_name="unet3d",
        num_classes=14,
        in_channels=1,
        img_size=[96, 96, 96],
        pretrain="",
        save_dir="checkpoints",
        amp=False,
    )


def _num_classes_for_dataset(dataset: str) -> int:
    return 2 if dataset == "msd" else 14


def _build_runtime_args(
    base_args: argparse.Namespace,
    dataset: str,
    msd_task: int,
    model_name: str,
) -> argparse.Namespace:
    runtime_args = argparse.Namespace(**vars(base_args))
    runtime_args.dataset = dataset
    runtime_args.msd_task = msd_task
    runtime_args.model_name = model_name
    runtime_args.num_classes = _num_classes_for_dataset(dataset)
    return runtime_args


def _available_models(base_args: argparse.Namespace, dataset: str, msd_task: int) -> list[str]:
    available: list[str] = []
    for model_name in SUPPORTED_MODELS:
        runtime_args = _build_runtime_args(base_args, dataset, msd_task, model_name)
        if os.path.exists(_checkpoint_path(runtime_args, model_name)):
            available.append(model_name)
    return available


def _get_or_load_model(
    base_args: argparse.Namespace,
    dataset: str,
    msd_task: int,
    model_name: str,
    cache: dict[tuple[str, int, str], torch.nn.Module],
) -> tuple[torch.nn.Module, argparse.Namespace]:
    key = (dataset, msd_task, model_name)
    runtime_args = _build_runtime_args(base_args, dataset, msd_task, model_name)
    if key in cache:
        return cache[key], runtime_args

    ckpt_path = _checkpoint_path(runtime_args, model_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, runtime_args)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    cache[key] = model
    return model, runtime_args


def _extract_logits(outputs):
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, tuple):
        return outputs[0]
    if isinstance(outputs, dict):
        return next(iter(outputs.values()))
    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")


def _resolve_uploaded_path(nifti_path: Any) -> str:
    if isinstance(nifti_path, str):
        return nifti_path
    if isinstance(nifti_path, dict) and "name" in nifti_path:
        return str(nifti_path["name"])
    named_path = getattr(nifti_path, "name", None)
    if named_path is not None:
        return str(named_path)
    raise ValueError("Invalid uploaded file. Please upload a .nii or .nii.gz file.")


def _validate_nifti_path(path: str) -> None:
    lower_path = path.lower()
    if lower_path.endswith(".nii") or lower_path.endswith(".nii.gz"):
        return
    raise ValueError("Unsupported file type. Please upload a .nii or .nii.gz file.")


def _slice_for_axis(volume: np.ndarray, seg: np.ndarray, axis: str, slice_idx: int):
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


def _load_nifti_volume(path: str) -> np.ndarray:
    """Load a NIfTI file with nibabel and return a 3-D numpy array (D, H, W)."""
    img = nib.load(path)
    vol = np.asarray(img.dataobj)
    # Squeeze a trailing size-1 time/component dimension if present.
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = vol[..., 0]
    return vol


def _uploaded_max_slice(nifti_path: Any, axis: str) -> int:
    """Return the maximum valid slice index for *axis* by loading the raw NIfTI."""
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


def _find_seg_output(input_path: str, out_dir: str) -> str | None:
    """Locate the NIfTI segmentation file written by ``SaveImaged``.

    MONAI's ``SaveImaged`` (with ``separate_folder=True``, the default) saves
    to ``{out_dir}/{stem}/{stem}_seg.nii.gz``.  We also check the flat layout
    ``{out_dir}/{stem}_seg.nii.gz`` as a fallback, and use glob for any
    compression variant.
    """
    stem = Path(input_path).name
    for ext in (".nii.gz", ".nii"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break

    candidates = [
        os.path.join(out_dir, stem, f"{stem}_seg.nii.gz"),
        os.path.join(out_dir, stem, f"{stem}_seg.nii"),
        os.path.join(out_dir, f"{stem}_seg.nii.gz"),
        os.path.join(out_dir, f"{stem}_seg.nii"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Broader glob in case MONAI appended extra suffixes
    pattern = os.path.join(out_dir, "**", f"{stem}_seg*")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return sorted(matches)[-1]

    return None


def load_models(args: argparse.Namespace) -> dict[str, torch.nn.Module]:
    """Pre-load all three model checkpoints into memory.

    Args:
        args: Command-line arguments.
        checkpoint_dir: Directory containing ``unet3d.pth``,
                        ``skip_densenet3d.pth``, and ``swin_unetr.pth``.

    Returns:
        Dict mapping model name → ``torch.nn.Module`` (eval mode).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {}
    for model_name in SUPPORTED_MODELS:
        runtime_args = _build_runtime_args(args, args.dataset, args.msd_task, model_name)
        ckpt_path = _checkpoint_path(runtime_args, model_name)
        if not os.path.exists(ckpt_path):
            continue
        model = build_model(model_name, runtime_args)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        loaded[model_name] = model

    if not loaded:
        raise FileNotFoundError(
            f"No checkpoints found in {args.save_dir} for dataset={args.dataset}. "
            "Expected files like unet3d_<dataset>_best_model.pth"
        )
    return loaded

def run_inference_to_file(
    nifti_path: Any,
    args: argparse.Namespace,
    model: torch.nn.Module,
    out_dir: str = OUT_DIR,
) -> str:
    """Run sliding-window inference and save the segmentation NIfTI via ``SaveImaged``.

    Args:
        nifti_path: Path to the uploaded NIfTI file.
        args:       Runtime arguments (dataset, img_size, …).
        model:      Pre-loaded model in eval mode.
        out_dir:    Root directory passed to ``SaveImaged``.

    Returns:
        Absolute path to the saved segmentation NIfTI file.
    """
    path = _resolve_uploaded_path(nifti_path)
    _validate_nifti_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    infer_transforms, post_process = (
        build_msd_inference_transforms(output_dir=out_dir)
        if args.dataset == "msd"
        else build_btcv_inference_transforms(output_dir=out_dir)
    )
    test_loader = DataLoader(
        Dataset(data=[{"image": path}], transform=infer_transforms),
        batch_size=1,
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch in test_loader:
            batch["pred"] = sliding_window_inference(
                inputs=batch["image"].to(device),
                roi_size=tuple(args.img_size),
                sw_batch_size=4,
                predictor=model,
            )
            for item in decollate_batch(batch):
                post_process(item)

    seg_path = _find_seg_output(path, out_dir)
    if seg_path is None:
        raise RuntimeError(
            f"Inference completed but no segmentation file was found under {out_dir!r}. "
            "Check that SaveImaged wrote successfully."
        )
    return seg_path


def render_seg_slice(
    nifti_path: Any,
    seg_path: str,
    axis: str,
    slice_idx: int,
) -> "plt.Figure":
    """Load a saved segmentation NIfTI and render an overlay figure.

    This function does **not** run inference — it only reads the files that
    ``run_inference_to_file`` already wrote to disk.

    Args:
        nifti_path: Original input NIfTI (for the background grey-scale image).
        seg_path:   Path to the saved segmentation NIfTI produced by inference.
        axis:       Viewing plane — ``"axial"``, ``"sagittal"``, or ``"coronal"``.
        slice_idx:  Index of the slice to display.

    Returns:
        A ``matplotlib.figure.Figure`` for Gradio to display.
    """
    orig_path = _resolve_uploaded_path(nifti_path)
    volume = _load_nifti_volume(orig_path)

    # nibabel loads NIfTI as (D, H, W) or (D, H, W, C) — channels last.
    # MONAI's SaveImaged writes the one-hot tensor in that same layout, so
    # argmax along the last axis recovers the integer label map.
    seg_img = nib.load(seg_path)
    seg_arr = np.asarray(seg_img.dataobj)
    if seg_arr.ndim == 4:
        seg = np.argmax(seg_arr, axis=-1)
    else:
        seg = seg_arr.squeeze()

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


def build_interface(args: argparse.Namespace):
    """Construct and return the ``gradio.Blocks`` interface object."""
    model_cache: dict[tuple[str, int, str], torch.nn.Module] = {}
    initial_models = _available_models(args, args.dataset, args.msd_task)

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

        model_names = _available_models(args, dataset_name, task)
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
        model_names = _available_models(args, dataset_name, task)
        if not model_names:
            model_names = list(SUPPORTED_MODELS)
        return gr.update(choices=model_names, value=model_names[0], interactive=True)

    def _run_inference(
        nifti_file,
        dataset_name,
        msd_task_value,
        model_name,
        axis_name,
        slice_value,
    ):
        """Run inference once, save NIfTI, render the first overlay.

        Returns (figure, seg_path) so the seg_path can be stored in State
        for subsequent slice/axis re-renders.
        """
        if nifti_file is None:
            raise gr.Error("Please upload a NIfTI volume first.")

        task = int(msd_task_value)
        if dataset_name != "msd":
            task = SUPPORTED_MSD_TASKS[0]

        try:
            model, runtime_args = _get_or_load_model(
                base_args=args,
                dataset=dataset_name,
                msd_task=task,
                model_name=model_name,
                cache=model_cache,
            )
        except FileNotFoundError as exc:
            raise gr.Error(str(exc))

        try:
            seg_path = run_inference_to_file(
                nifti_path=nifti_file,
                args=runtime_args,
                model=model,
                out_dir=OUT_DIR,
            )
        except Exception as exc:
            raise gr.Error(str(exc))

        fig = render_seg_slice(
            nifti_path=nifti_file,
            seg_path=seg_path,
            axis=axis_name,
            slice_idx=int(slice_value),
        )
        return fig, seg_path

    def _rerender_slice(nifti_file, seg_path, axis_name, slice_value):
        """Re-render from the already-saved segmentation NIfTI — no inference."""
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

    with gr.Blocks(title="3D Medical Segmentation Demo") as demo:
        gr.Markdown("## 3D Medical Segmentation Demo")
        gr.Markdown(
            "Upload a NIfTI file, pick a model and viewing axis, then click "
            "**Run Segmentation**. Afterwards, adjust the axis or slice to "
            "re-render instantly from the saved segmentation — no re-inference needed."
        )

        # Stores the path to the segmentation NIfTI saved by SaveImaged.
        seg_path_state = gr.State(value=None)

        with gr.Row():
            # Windows browsers may not match multi-part extensions like ".nii.gz"
            # reliably. Accept ".gz" in picker; validate strict NIfTI suffix server-side.
            nifti_input = gr.File(label="NIfTI volume", file_types=[".nii", ".gz"], type="filepath")
            dataset_input = gr.Dropdown(
                choices=list(SUPPORTED_DATASETS),
                value=args.dataset,
                label="Dataset",
            )
            msd_task_input = gr.Dropdown(
                choices=list(SUPPORTED_MSD_TASKS),
                value=args.msd_task,
                label="MSD Task",
                visible=args.dataset == "msd",
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
            slice_input = gr.Slider(minimum=0, maximum=256, value=128, step=1, label="Slice index")

        run_button = gr.Button("Run segmentation", variant="primary")
        out_plot = gr.Plot(label="Segmentation overlay")

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
        nifti_input.change(_update_slider, inputs=[nifti_input, axis_input], outputs=[slice_input])
        axis_input.change(_update_slider, inputs=[nifti_input, axis_input], outputs=[slice_input])

        # Run inference once; store the saved seg path in State.
        run_button.click(
            _run_inference,
            inputs=[nifti_input, dataset_input, msd_task_input, model_input, axis_input, slice_input],
            outputs=[out_plot, seg_path_state],
        )

        # Re-render from the saved NIfTI without re-running inference.
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


def launch(checkpoint_dir: str = "checkpoints", share: bool = False) -> None:
    """Load models and launch the Gradio server."""
    args = _default_demo_args()
    args.save_dir = checkpoint_dir
    demo = build_interface(args)
    demo.launch(share=share)


if __name__ == "__main__":
    launch()