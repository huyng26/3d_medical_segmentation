"""Experiment logger initialisation for Weights & Biases and/or TensorBoard (Phase 4)."""
from __future__ import annotations


def init_logger(cfg: dict):
    """Initialise the experiment logger(s) specified in ``cfg['logging']['backend']``.

    Supports ``"wandb"``, ``"tensorboard"``, and ``"both"``.

    Args:
        cfg: Top-level training config dict (must contain ``logging`` sub-dict).

    Returns:
        Logger object (wandb Run, SummaryWriter, or a thin wrapper for both).
    """
    raise NotImplementedError


def log_metrics(logger, metrics: dict, step: int) -> None:
    """Log a metrics dict to the active logger at the given step."""
    raise NotImplementedError


def finish_logger(logger) -> None:
    """Flush and close the logger at the end of training."""
    raise NotImplementedError
