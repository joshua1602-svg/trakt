"""
workflow.py
===========

CLI entry point for the Trakt Validation Agent.

The Validation Agent is the control gate after Transformation. It consumes the
Transformation Agent output package, validates the transformed canonical values +
the transformation issue classifications, and produces a governed validation
readiness package for the Projection Agent.

Run it with::

    python -m engine.validation_agent.workflow \\
      --transformation-manifest onboarding_output/client_001/run_tva_exit_check/output/transformation/30_transformation_manifest.json

It never re-runs raw Gate 1, never source-matches, never mutates upstream
artefacts, and never projects to Annex 2 XML.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.validation_agent.validation_agent import (  # noqa: E402
    build_validation_package,
    TransformationHandoffError,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trakt Validation Agent — consume the transformation package "
        "and produce a validation readiness package for the Projection Agent.")
    p.add_argument(
        "--transformation-manifest", required=True,
        help="Path to the transformation manifest "
        "(output/transformation/30_transformation_manifest.json).")
    p.add_argument("--registry", default="",
                   help="Canonical field registry YAML (defaults to the manifest value).")
    p.add_argument("--regime-config", default="",
                   help="Regime delivery rules YAML (defaults to the manifest value).")
    p.add_argument("--asset-config", default="",
                   help="Asset-class defaults YAML (defaults to the manifest value).")
    p.add_argument("--enum-config-dir", default="",
                   help="Directory holding enum synonym libraries (config/system).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_validation_package(
            args.transformation_manifest,
            registry_path=args.registry,
            regime_config_path=args.regime_config,
            asset_config_path=args.asset_config,
            enum_config_dir=args.enum_config_dir,
        )
    except TransformationHandoffError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    m = result["manifest"]
    print(f"Validation package written to: {result['validation_dir']}")
    print(f"  ready_for_validation_complete = {m['ready_for_validation_complete']}")
    print(f"  ready_for_projection          = {m['ready_for_projection']}")
    print(f"  ready_for_xml_delivery        = {m['ready_for_xml_delivery']}")
    print(f"  validation failures (blocking)= {m['blocking_for_validation_count']}")
    print(f"  projection blockers           = {m['blocking_for_projection_count']}")
    print(f"  next_agent                    = {m['next_agent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
