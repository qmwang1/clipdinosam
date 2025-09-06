import os
import sys
from typing import Any, Dict

import yaml


def _parse_overrides(pairs):
    overrides = {}
    for item in pairs or []:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        # Try to parse YAML scalars (bool, int, float, lists)
        try:
            v_parsed = yaml.safe_load(v)
        except Exception:
            v_parsed = v
        _assign_nested(overrides, k, v_parsed)
    return overrides


def _assign_nested(d: Dict[str, Any], dotted_key: str, value: Any):
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v
    return a


def load_config_with_overrides(path: str, override_pairs=None) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    overrides = _parse_overrides(override_pairs)
    return _deep_update(cfg, overrides)

