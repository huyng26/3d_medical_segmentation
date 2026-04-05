"""YAML config loading and merging utilities."""
from __future__ import annotations

from pathlib import Path


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return it as a plain dict."""
    raise NotImplementedError


def merge_configs(*cfgs: dict) -> dict:
    """Deep-merge an ordered sequence of config dicts.

    Later dicts override earlier ones. Nested dicts are merged recursively
    rather than replaced wholesale.
    """
    raise NotImplementedError


def resolve_dataset_config(dataset_name: str, config_dir: str | Path = "configs") -> dict:
    """Return the merged config for a named dataset.

    Looks for ``configs/dataset_{dataset_name}.yaml``.
    """
    raise NotImplementedError
