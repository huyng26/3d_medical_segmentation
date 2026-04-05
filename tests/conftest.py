"""Shared pytest fixtures."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def dummy_patch() -> torch.Tensor:
    """Return a (1, 1, 128, 128, 64) random float tensor mimicking a CT patch."""
    return torch.randn(1, 1, 128, 128, 64)


@pytest.fixture
def dummy_label() -> torch.Tensor:
    """Return a (1, 128, 128, 64) integer label tensor with 14 classes."""
    return torch.randint(0, 14, (1, 128, 128, 64))
