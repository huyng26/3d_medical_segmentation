"""Phase 2 smoke test: verify the DataLoader yields correctly shaped tensors."""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Requires preprocessed data — run after Phase 1.")
def test_dataloader_yields_patches():
    from medseg.data.dataset import MedSegDataset
    from torch.utils.data import DataLoader

    # TODO: point at a small fixture volume once preprocessing is implemented
    raise NotImplementedError
