#!/usr/bin/env python3
"""Pass 1 helper: build a resolved config artifact.

This script is additive scaffolding and is NOT wired into existing runtime entrypoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict
import sys

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.system.config_resolver import (
    CHART_PRECEDENCE,
    DEFAULT_PRECEDENCE,
    LayerSpec,
    resolve_layers,
)


def _load_yaml_optional(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _sha256_dict(data: Dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build resolved config artifact (Pass 1 scaffold)")

    # context labels (metadata only for Pass 1)
    ap.add_argument("--platform", default="default")
    ap.add_argument("--asset", default="equity_release")
    ap.add_argument("--regime", default="ESMA_Annex2")
    ap.add_argument("--client", default="ere")

    # general layers
    ap.add_argument("--platform-config", default=None)
    ap.add_argument("--asset-config", default="config/asset/product_defaults_ERM.yaml")
    ap.add_argument("--regime-config", default="config/regime/annex12_rules.yaml")
    ap.add_argument("--client-config", default="config/client/config_client_ERM_UK.yaml")

    # canonical registry (kept separate/canonical)
    ap.add_argument("--field-registry", default="config/system/fields_registry.yaml")

    # chart layers
    ap.add_argument("--standard-chart-pack", default=None)
    ap.add_argument("--asset-chart-pack", default="config/asset/static_pools_config_erm.yaml")
    ap.add_argument("--client-chart-overrides", default=None)

    # runtime override file (optional yaml)
    ap.add_argument("--runtime-overrides", default=None)

    # output
    ap.add_argument("--out", default="out/resolved_config.yaml")
    return ap.parse_args()


def _to_path(p: str | None) -> Path | None:
    return Path(p) if p else None


def main() -> None:
    args = parse_args()

    context = {
        "platform": args.platform,
        "asset": args.asset,
        "regime": args.regime,
        "client": args.client,
    }

    runtime_overrides = _load_yaml_optional(_to_path(args.runtime_overrides)) if args.runtime_overrides else {}

    general_layers = [
        LayerSpec("platform", _to_path(args.platform_config)),
        LayerSpec("asset", _to_path(args.asset_config)),
        LayerSpec("regime", _to_path(args.regime_config)),
        LayerSpec("client", _to_path(args.client_config)),
    ]

    general = resolve_layers(
        context=context,
        layers=general_layers,
        runtime_overrides=runtime_overrides,
        blocked_prefixes=("fields_registry", "field_registry", "fields"),
    )

    chart_layers = [
        LayerSpec("standard_chart_pack", _to_path(args.standard_chart_pack)),
        LayerSpec("asset_chart_pack", _to_path(args.asset_chart_pack)),
        LayerSpec("client_chart_overrides", _to_path(args.client_chart_overrides)),
    ]
    charts = resolve_layers(context=context, layers=chart_layers)

    field_registry = _load_yaml_optional(_to_path(args.field_registry))

    merged = {
        "metadata": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "context": context,
            "merge_order": list(DEFAULT_PRECEDENCE),
            "chart_merge_order": list(CHART_PRECEDENCE),
            "source_paths": {
                "platform_config": args.platform_config,
                "asset_config": args.asset_config,
                "regime_config": args.regime_config,
                "client_config": args.client_config,
                "field_registry": args.field_registry,
                "standard_chart_pack": args.standard_chart_pack,
                "asset_chart_pack": args.asset_chart_pack,
                "client_chart_overrides": args.client_chart_overrides,
                "runtime_overrides": args.runtime_overrides,
            },
        },
        "resolved": {
            "general": general.resolved,
            "charts": charts.resolved,
            "field_registry": field_registry,
        },
        "provenance": {
            "general": general.provenance,
            "charts": charts.provenance,
        },
        "warnings": {
            "general": general.warnings,
            "charts": charts.warnings,
        },
    }
    merged["metadata"]["resolved_sha256"] = _sha256_dict(merged["resolved"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")

    print(f"Resolved config written: {args.out}")


if __name__ == "__main__":
    main()

