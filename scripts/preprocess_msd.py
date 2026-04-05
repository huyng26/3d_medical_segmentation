"""Preprocess raw MSD NIfTI files into training-ready volumes (Phase 1).

Applies the same pipeline as preprocess_btcv.py to Task03 and Task09.

Usage:
    python scripts/preprocess_msd.py --config configs/dataset_msd_liver.yaml
    python scripts/preprocess_msd.py --config configs/dataset_msd_spleen.yaml
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess an MSD task dataset.")
    parser.add_argument(
        "--config",
        required=True,
        help="Dataset config YAML (dataset_msd_liver.yaml or dataset_msd_spleen.yaml).",
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
