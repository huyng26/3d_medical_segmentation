"""Preprocess raw BTCV NIfTI files into training-ready volumes (Phase 1).

Pipeline per volume:
  1. Resample to target voxel spacing (1.5 × 1.5 × 2.0 mm by default)
  2. Clip HU to [-175, 250]
  3. Z-score normalise
  4. Save as .npy or .nii.gz under data/processed/btcv/

Usage:
    python scripts/preprocess_btcv.py --config configs/dataset_btcv.yaml
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess BTCV dataset.")
    parser.add_argument(
        "--config", default="configs/dataset_btcv.yaml", help="Dataset config YAML."
    )
    parser.add_argument(
        "--output-format",
        choices=["npy", "nii.gz"],
        default="npy",
        help="Format for preprocessed volume files.",
    )
    return parser.parse_args()


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
