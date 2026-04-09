"""Unified PyTorch Dataset for BTCV and MSD volumes (Phase 2)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from torchvision.transforms import v2
import nibabel as nib 
import os

class MedSegDataset(Dataset):
    """Load preprocessed NIfTI / NumPy volumes and sample 3-D patches.

    Supports both BTCV and MSD layouts via a shared config dict.
    At training time patches are drawn randomly; at inference time the
    caller should use ``patch_utils.sliding_window_coords`` instead.
    """
 
    def __init__(
        self,
        data_dir: str | Path,
        file_list: list[dict[str, str]],
        num_classes: int,
        patch_size: tuple[int, int, int] = (128, 128, 64),
        transforms=None,
        mode: str = "train",
    ) -> None:
        self.data_dir = data_dir 
        if mode == "train":
            split = "imagesTr"
        else:
            split = "imagesTs"
        self.data_dir = os.path.join(self.data_dir, split)
        self.file_list = []

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError

if __name__ == "__main__":