"""Data loading, augmentation, and patch utilities."""
import argparse
from .btcv import build_btcv_dataloader
from .msd import build_msd_dataloader
from typing import Optional
from medseg.utils.config import load_config

def build_dataloader(task:Optional[int] , type: str = 'btcv', mode: str = "train"):
    if type == "btcv":
        cfg = load_config("../configs/dataset_btcv.yaml")
        args = argparse.Namespace(**cfg)
        return build_btcv_dataloader(args, mode)
    elif type == "msd":
        cfg = load_config("../configs/dataset_msd.yaml")
        args = argparse.Namespace(**cfg)
        return build_msd_dataloader(args, mode, task = task)
    else:
        raise ValueError(f"Unsupported dataloader type: {type}")
