"""
workflow.py
===========

CLI entry point for the Trakt Transformation Agent.

The Transformation Agent is the deterministic bridge between the Onboarding
Agent handoff package and the Validation Agent. It consumes the governed
canonical onboarding handoff package and central tape and produces a normalized,
validation-ready transformed canonical package.

Run it with::

    python -m engine.transformation_agent.workflow \\
      --handoff-manifest onboarding_output/client_001/run_annex2_onboarding_exit_check/output/handoff/24_onboarding_handoff_manifest.json

It never re-runs raw Gate 1, never fuzzy-matches sources, never mutates
onboarding artefacts, and never projects to Annex 2 XML.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.transformation_agent.transformation_agent import (  # noqa: E402
    build_transformation_package,
    HandoffValidationError,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trakt Transformation Agent — consume the onboarding handoff "
        "package and produce a validation-ready transformed canonical package.")
    p.add_argument(
        "--handoff-manifest", required=True,
        help="Path to the onboarding handoff manifest "
        "(output/handoff/24_onboarding_handoff_manifest.json).")
    p.add_argument("--asset-config", default="",
                   help="Asset-class defaults YAML (defaults to the handoff value).")
    p.add_argument("--regime-config", default="",
                   help="Regime delivery rules YAML (defaults to the handoff value).")
    p.add_argument("--registry", default="",
                   help="Canonical field registry YAML (defaults to the handoff value).")
    p.add_argument("--enum-mapping", default="",
                   help="Optional enum mapping YAML.")
    p.add_argument("--no-dayfirst", action="store_true",
                   help="Parse ambiguous dates month-first instead of day-first.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_transformation_package(
            args.handoff_manifest,
            asset_config_path=args.asset_config,
            regime_config_path=args.regime_config,
            registry_path=args.registry,
            enum_mapping_path=args.enum_mapping,
            dayfirst=not args.no_dayfirst,
        )
    except HandoffValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    manifest = result["manifest"]
    print(f"Transformation package written to: {result['transformation_dir']}")
    print(f"  ready_for_validation    = {manifest['ready_for_validation']}")
    print(f"  ready_for_projection    = {manifest['ready_for_projection']}")
    print(f"  ready_for_xml_delivery  = {manifest['ready_for_xml_delivery']}")
    print(f"  issues                  = {manifest['issue_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
