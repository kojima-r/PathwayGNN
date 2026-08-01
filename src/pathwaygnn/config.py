from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML, recursively resolving an optional ``defaults`` list."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    defaults = current.pop("defaults", [])
    merged: dict[str, Any] = {}
    for relative in defaults:
        merged = _merge(merged, load_config(path.parent / relative))
    return _merge(merged, current)

