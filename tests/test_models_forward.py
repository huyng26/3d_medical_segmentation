"""Phase 3 deliverable: verify all three models pass a forward pass without errors.

Each model is constructed with default settings and tested against a
(1, 1, 128, 128, 64) dummy input tensor.  Parameter counts are printed.
"""
from __future__ import annotations

import pytest
import torch

from medseg.models.unet3d import UNet3D
from medseg.models.skip_densenet_3d import SkipDenseNet3D
from medseg.models.swin_unetr import SwinUNETRWrapper

NUM_CLASSES = 14
PATCH_SIZE = (128, 128, 64)


@pytest.mark.parametrize(
    "model_cls,kwargs",
    [
        (UNet3D, {"in_channels": 1, "out_channels": NUM_CLASSES}),
        (SkipDenseNet3D, {"in_channels": 1, "out_channels": NUM_CLASSES}),
        (
            SwinUNETRWrapper,
            {
                "in_channels": 1,
                "out_channels": NUM_CLASSES,
                "img_size": PATCH_SIZE,
            },
        ),
    ],
)
def test_forward_pass(dummy_patch, model_cls, kwargs):
    model = model_cls(**kwargs)
    model.eval()
    with torch.no_grad():
        out = model(dummy_patch)
    assert out.shape == (1, NUM_CLASSES, *PATCH_SIZE), (
        f"{model_cls.__name__} output shape {out.shape} != expected (1, {NUM_CLASSES}, {PATCH_SIZE})"
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{model_cls.__name__}: {n_params:,} parameters")
