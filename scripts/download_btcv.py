"""Download BTCV dataset from the Synapse platform (Phase 1).

Usage:
    python scripts/download_btcv.py --dest data/raw/btcv

Requires a free Synapse account and the ``synapseclient`` Python package.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BTCV from Synapse.")
    parser.add_argument("--dest", default="data/raw/btcv", help="Download destination directory.")
    parser.add_argument(
        "--synapse-id",
        default="syn3193805",
        help="Synapse entity ID for the BTCV dataset.",
    )
    return parser.parse_args()


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
