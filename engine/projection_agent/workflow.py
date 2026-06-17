"""
workflow.py
===========

CLI entry point for the Trakt Projection Agent.

The Projection Agent is the stage after Validation. It consumes the Validation
Agent output package + the transformed canonical tape and produces a governed
**projection package** (an Annex 2 target frame, not XML) for the downstream
Delivery/XML Agent.

Run it with::

    python -m engine.projection_agent.workflow \\
      --validation-manifest onboarding_output/client_001/run_projection_blocker_diagnostic_fix/output/validation/40_validation_manifest.json

It never re-runs Gate 1 / Transformation / Validation, never mutates upstream
artefacts, never invokes the Gate 5 XML builder, and never claims XML readiness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.projection_agent.projection_agent import (  # noqa: E402
    build_projection_package,
    ValidationHandoffError,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trakt Projection Agent — consume the validation package and "
        "produce a projected Annex 2 target frame (NOT XML) for delivery normalisation.")
    p.add_argument(
        "--validation-manifest", required=True,
        help="Path to the validation manifest "
        "(output/validation/40_validation_manifest.json).")
    p.add_argument("--regime-config", default="",
                   help="Annex 2 delivery rules YAML (defaults to the manifest value).")
    p.add_argument("--asset-config", default="",
                   help="Asset-class defaults YAML (defaults to the manifest value).")
    p.add_argument("--registry", default="",
                   help="Canonical field registry YAML (defaults to the manifest value).")
    p.add_argument("--esma-code-order", default="",
                   help="ESMA code order YAML (defaults to config/system/esma_code_order.yaml).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_projection_package(
            args.validation_manifest,
            registry_path=args.registry,
            regime_config_path=args.regime_config,
            asset_config_path=args.asset_config,
            esma_code_order_path=args.esma_code_order,
        )
    except ValidationHandoffError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    m = result["manifest"]
    print(f"Projection package written to: {result['projection_dir']}")
    print(f"  projection_ran                    = {m['projection_ran']}")
    print(f"  projection_complete               = {m['projection_complete']}")
    print(f"  ready_for_delivery_normalisation  = {m['ready_for_delivery_normalisation']}")
    print(f"  ready_for_xml_delivery            = {m['ready_for_xml_delivery']}")
    print(f"  target frame rows                 = {m['frame_row_count']}")
    print(f"  blockers resolved / carried fwd   = "
          f"{m['blockers_resolved_count']} / {m['remaining_blocker_count']}")
    print(f"  delivery-blocking issues          = {m['blocking_for_delivery_issue_count']}")
    print(f"  next_agent                        = {m['next_agent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
