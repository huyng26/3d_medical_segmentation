"""Batch inference CLI — Phase 5.

Runs sliding-window inference for a given model + dataset combination and
writes integer label-map NIfTI files to a structured output directory:

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

Usage (from the project root):
    python -m medseg.inference.predict \\
        --dataset btcv \\
        --data_path ./data/BTCV \\
        --model_name unet3d \\
        --num_classes 14 \\
        --checkpoint checkpoints/unet3d_btcv_best_model.pth \\
        --out_dir ./out
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference

from medseg.cfg import load_args
from medseg.models import build_model
from medseg.data_utils.transforms import (
    build_btcv_inference_transforms,
    build_msd_inference_transforms,
)


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

def load_checkpoint(
    ckpt_path: str | Path,
    model: torch.nn.Module,
    device: torch.device | str,
) -> torch.nn.Module:
    """Load a saved state-dict into *model* and return it in eval mode.

    Args:
        ckpt_path: Path to the ``.pth`` checkpoint file.
        model:     Uninitialised model with the correct architecture.
        device:    Target device.

    Returns:
        The model in eval mode with loaded weights.
    """
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_volume(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: tuple[int, int, int],
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Run sliding-window inference on a single pre-processed volume.

    Args:
        model:          Trained segmentation model (eval mode expected).
        image:          Input tensor of shape ``(1, 1, D, H, W)`` on *device*.
        patch_size:     Spatial size of each inference patch.
        sw_batch_size:  Patches processed simultaneously.
        overlap:        Fractional overlap between adjacent windows.
        device:         Device to run inference on.

    Returns:
        Soft probability map of shape ``(1, C, D, H, W)``.
    """
    with torch.no_grad():
        return sliding_window_inference(
            inputs=image,
            roi_size=patch_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )


def postprocess(prob_map: torch.Tensor) -> np.ndarray:
    """Convert a soft probability map to an integer label map.

    Args:
        prob_map: Tensor of shape ``(1, C, D, H, W)``.

    Returns:
        Integer numpy array of shape ``(D, H, W)``.
    """
    # argmax over the channel dimension, squeeze the batch dim
    label_map = torch.argmax(prob_map, dim=1)  # (1, D, H, W)
    return label_map.squeeze(0).cpu().numpy().astype(np.int16)


def save_segmentation(
    label_map: np.ndarray,
    reference_nifti_path: str | Path,
    out_path: str | Path,
) -> None:
    """Save *label_map* as a NIfTI file reusing the original image's geometry.

    Args:
        label_map:            Integer array of shape ``(D, H, W)``.
        reference_nifti_path: Path to the original input NIfTI (for affine/header).
        out_path:             Destination path (parent directories are created).
    """
    ref = nib.load(str(reference_nifti_path))
    seg_img = nib.Nifti1Image(label_map, affine=ref.affine, header=ref.header)
    seg_img.set_data_dtype(np.int16)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(seg_img, str(out_path))


# ---------------------------------------------------------------------------
# Dataset-level orchestration
# ---------------------------------------------------------------------------

def _dataset_key(dataset: str, msd_task: int) -> str:
    """Return the subfolder name used under ``out_dir``."""
    return "btcv" if dataset == "btcv" else f"msd_task{msd_task}"


def _build_image_list(args: argparse.Namespace) -> list[dict[str, str]]:
    """Return a list of ``{"image": path}`` dicts for the test split."""
    import glob as _glob

    data_path = args.data_path
    if args.dataset == "msd":
        task_dir = "Task02_Heart" if args.msd_task == 2 else "Task09_Spleen"
        data_path = os.path.join(data_path, task_dir)

    # Prefer imagesTs; fall back to imagesTr when no dedicated test set exists.
    test_images = sorted(_glob.glob(os.path.join(data_path, "imagesTs", "*.nii.gz")))
    if not test_images:
        test_images = sorted(_glob.glob(os.path.join(data_path, "imagesTr", "*.nii.gz")))

    if not test_images:
        raise FileNotFoundError(
            f"No NIfTI images found under {data_path}/imagesTs or imagesTr"
        )
    return [{"image": p} for p in test_images]


def run_dataset_inference(
    args: argparse.Namespace,
    model: torch.nn.Module,
    data_list: list[dict[str, str]],
    out_dir: str | Path,
) -> None:
    """Run inference on every volume in *data_list* and write results to disk.

    Output path per volume:
        ``{out_dir}/{dataset_key}/{model_name}/{stem}_seg.nii.gz``

    Args:
        args:      Runtime arguments (dataset, msd_task, model_name, img_size).
        model:     Loaded model in eval mode.
        data_list: List of ``{"image": path}`` dicts.
        out_dir:   Root output directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds_key = _dataset_key(args.dataset, args.msd_task)
    pre_transforms, post_transforms = (
        build_msd_inference_transforms()
        if args.dataset == "msd"
        else build_btcv_inference_transforms()
    )

    loader = DataLoader(
        Dataset(data=data_list, transform=pre_transforms),
        batch_size=1,
        shuffle=False,
        num_workers=getattr(args, "num_workers", 0),
    )

    total = len(data_list)
    for idx, batch in enumerate(loader, 1):
        orig_path = data_list[idx - 1]["image"]
        stem = Path(orig_path).name
        for ext in (".nii.gz", ".nii"):
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)]
                break

        print(f"[{idx}/{total}] {stem} ...", end=" ", flush=True)

        batch["pred"] = predict_volume(
            model=model,
            image=batch["image"].to(device),
            patch_size=tuple(args.img_size),
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap,
            device=device,
        )

        # Invert preprocessing transforms back to original image space.
        processed = [post_transforms(item) for item in decollate_batch(batch)]

        # processed[0]["pred"] is a (1, D, H, W) integer tensor after AsDiscreted.
        pred_tensor = processed[0]["pred"]
        label_map = pred_tensor.squeeze(0).cpu().numpy().astype(np.int16)

        out_path = Path(out_dir) / ds_key / args.model_name / f"{stem}_seg.nii.gz"
        save_segmentation(label_map, orig_path, out_path)
        print(f"saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = load_args()

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required for inference.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(args.model_name, args)
    model = load_checkpoint(args.checkpoint, model, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    data_list = _build_image_list(args)
    print(f"Found {len(data_list)} volumes to process.")

    run_dataset_inference(args, model, data_list, args.out_dir)
    print("Done.")
