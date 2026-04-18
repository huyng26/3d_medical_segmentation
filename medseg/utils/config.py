"""YAML config loading and merging utilities."""
from __future__ import annotations

from pathlib import Path
import yaml
from typing import Any

def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return it as a plain dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def _deep_merge(base: dict[str, Any], update: dict[str, Any]):
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out