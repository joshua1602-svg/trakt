"""Lightweight config resolver scaffold (Pass 1).

This module is intentionally additive and not wired into runtime entrypoints yet.
It provides deterministic merge + provenance capture to support gradual adoption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import yaml


LayerName = str


@dataclass
class ResolvedConfig:
    context: Dict[str, str]
    resolved: Dict[str, Any]
    provenance: Dict[str, LayerName] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _load_yaml_optional(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _merge_dicts(
    base: Dict[str, Any],
    incoming: Mapping[str, Any],
    layer: LayerName,
    provenance: Dict[str, LayerName],
    prefix: str = "",
) -> Dict[str, Any]:
    out = dict(base)
    for key, value in incoming.items():
        path = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dicts(out[key], value, layer, provenance, prefix=path)
            continue

        # Pass 1 list policy: replace list rather than deep list merge.
        out[key] = value
        provenance[path] = layer

    return out


def resolve_layers(
    *,
    context: Mapping[str, str],
    layers: Iterable[Tuple[LayerName, Optional[Path]]],
    runtime_overrides: Optional[Mapping[str, Any]] = None,
) -> ResolvedConfig:
    """Resolve config from ordered layers and optional runtime overrides.

    Args:
        context: e.g. {"platform": "default", "asset": "equity_release", ...}
        layers: ordered list of (layer_name, yaml_path)
        runtime_overrides: optional flat or nested override mapping applied last
    """
    resolved: Dict[str, Any] = {}
    provenance: Dict[str, LayerName] = {}
    warnings: list[str] = []

    for layer_name, path in layers:
        data = _load_yaml_optional(path)
        if not data:
            if path is not None and not path.exists():
                warnings.append(f"Layer '{layer_name}' missing file: {path}")
            continue
        resolved = _merge_dicts(resolved, data, layer_name, provenance)

    if runtime_overrides:
        resolved = _merge_dicts(
            resolved,
            runtime_overrides,
            "runtime_override",
            provenance,
        )

    return ResolvedConfig(
        context=dict(context),
        resolved=resolved,
        provenance=provenance,
        warnings=warnings,
    )


def dump_resolved_yaml(result: ResolvedConfig, output_path: Path) -> None:
    payload = {
        "context": result.context,
        "resolved": result.resolved,
        "provenance": result.provenance,
        "warnings": result.warnings,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

