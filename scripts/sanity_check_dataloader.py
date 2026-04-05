"""Visualise a batch of image/mask patches from the DataLoader (Phase 2 deliverable).

Renders three orthogonal slice views for a random patch with the segmentation mask
overlaid in colour. Saves figures to outputs/figures/sanity_check/.

Usage:
    python scripts/sanity_check_dataloader.py \
        --config configs/dataset_btcv.yaml \
        --train-config configs/train_default.yaml \
        --num-batches 4
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise DataLoader batches.")
    parser.add_argument("--config", required=True, help="Dataset config YAML.")
    parser.add_argument(
        "--train-config", default="configs/train_default.yaml", help="Training config YAML."
    )
    parser.add_argument(
        "--num-batches", type=int, default=4, help="Number of batches to visualise."
    )
    parser.add_argument(
        "--out-dir", default="outputs/figures/sanity_check", help="Output directory for figures."
    )
    return parser.parse_args()


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
