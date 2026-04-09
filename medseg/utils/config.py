"""YAML config loading and merging utilities."""
from __future__ import annotations

from pathlib import Path
import yaml

def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return it as a plain dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

