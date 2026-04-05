"""Download MSD Task03 (Liver) and Task09 (Spleen) from medicaldecathlon.com (Phase 1).

Usage:
    python scripts/download_msd.py --dest data/raw/msd

Downloads and unpacks the two task archives (~6 GB total).
"""
from __future__ import annotations

import argparse

MSD_BASE_URL = "https://medicaldecathlon.com/files"
TASKS = {
    "Task03_Liver": f"{MSD_BASE_URL}/Task03_Liver.tar",
    "Task09_Spleen": f"{MSD_BASE_URL}/Task09_Spleen.tar",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MSD task archives.")
    parser.add_argument("--dest", default="data/raw/msd", help="Download destination directory.")
    return parser.parse_args()


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
