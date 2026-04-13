"""Phase 2 smoke test: verify the DataLoader yields correctly shaped tensors."""
from __future__ import annotations

import pytest
import yaml
import os
import argparse

configs_dir = os.path.join(os.path.dirname(__file__), "../configs")
with open(os.path.join(configs_dir, "dataset_btcv.yaml"), "r") as f:
    cfg = yaml.safe_load(f)
    args = argparse.Namespace(**cfg)

print(args)

@pytest.mark.skipif(
    not os.path.exists("data/BTCV/imagesTr"),
    reason="Requires preprocessed data — run after Phase 1."
)

def test_dataloader_yields_patches():
    from medseg.data_utils.msd import build_msd_dataloader
    from medseg.data_utils.btcv import build_btcv_dataloader
    dataloader = build_btcv_dataloader(args, mode="train")
    for batch in dataloader:
        print(batch["image"].shape)
        print(batch["label"].shape)
        break

if __name__ == "__main__":
    test_dataloader_yields_patches()
    pass 