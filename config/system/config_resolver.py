"""Lightweight config resolver scaffold (Pass 1).

This module is intentionally additive and not wired into runtime entrypoints yet.
It provides deterministic merge + provenance capture to support gradual adoption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml


LayerName = str

# Pass 1 precedence model (general behavior)
DEFAULT_PRECEDENCE: Tuple[LayerName, ...] = (
    "platform",
    "asset",
    "regime",
    "client",
)

# Pass 1 precedence model (charts)
CHART_PRECEDENCE: Tuple[LayerName, ...] = (
    "standard_chart_pack",
    "asset_chart_pack",
    "client_chart_overrides",
)


@dataclass(frozen=True)
class LayerSpec:
    name: LayerName
    path: Optional[Path]


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


def _blocked(path: str, blocked_prefixes: Sequence[str]) -> bool:
    return any(path == p or path.startswith(f"{p}.") for p in blocked_prefixes)


def _merge_dicts(
    base: Dict[str, Any],
    incoming: Mapping[str, Any],
    layer: LayerName,
    provenance: Dict[str, LayerName],
    warnings: list[str],
    *,
    blocked_prefixes: Sequence[str] = (),
    prefix: str = "",
) -> Dict[str, Any]:
    out = dict(base)
    for key, value in incoming.items():
        path = f"{prefix}.{key}" if prefix else str(key)

        if _blocked(path, blocked_prefixes):
            warnings.append(
                f"Skipped blocked override at '{path}' from layer '{layer}'"
            )
            continue

        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dicts(
                out[key],
                value,
                layer,
                provenance,
                warnings,
                blocked_prefixes=blocked_prefixes,
                prefix=path,
            )
            continue

        if path in provenance and provenance[path] != layer:
            warnings.append(
                f"Override at '{path}': '{provenance[path]}' -> '{layer}'"
            )

        # Pass 1 list policy: replace list rather than deep list merge.
        out[key] = value
        provenance[path] = layer

    return out


def resolve_layers(
    *,
    context: Mapping[str, str],
    layers: Iterable[LayerSpec | Tuple[LayerName, Optional[Path]]],
    runtime_overrides: Optional[Mapping[str, Any]] = None,
    blocked_prefixes: Sequence[str] = (),
) -> ResolvedConfig:
    """Resolve config from ordered layers and optional runtime overrides.

    Args:
        context: e.g. {"platform": "default", "asset": "equity_release", ...}
        layers: ordered list of LayerSpec or (layer_name, yaml_path)
        runtime_overrides: optional flat or nested override mapping applied last
        blocked_prefixes: key path prefixes that cannot be overridden
    """
    resolved: Dict[str, Any] = {}
    provenance: Dict[str, LayerName] = {}
    warnings: list[str] = []

    normalized_layers: list[LayerSpec] = []
    for entry in layers:
        if isinstance(entry, LayerSpec):
            normalized_layers.append(entry)
        else:
            lname, lpath = entry
            normalized_layers.append(LayerSpec(name=lname, path=lpath))

    for spec in normalized_layers:
        data = _load_yaml_optional(spec.path)
        if not data:
            if spec.path is not None and not spec.path.exists():
                warnings.append(f"Layer '{spec.name}' missing file: {spec.path}")
            continue
        resolved = _merge_dicts(
            resolved,
            data,
            spec.name,
            provenance,
            warnings,
            blocked_prefixes=blocked_prefixes,
        )

    if runtime_overrides:
        resolved = _merge_dicts(
            resolved,
            runtime_overrides,
            "runtime_override",
            provenance,
            warnings,
            blocked_prefixes=blocked_prefixes,
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

