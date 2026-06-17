"""
workflow.py
===========

CLI entry point for the Trakt Delivery/XML Agent v1.

The Delivery/XML Agent is the stage after Projection. It consumes the Projection
package (the projected Annex 2 target frame + controlled blockers) and produces a
governed **delivery package** under ``output/delivery_xml/`` (artefacts 60..64):
a delivery-facing view of the target frame, a delivery-readiness report and
delivery issues. It **refuses** XML generation unless every readiness gate passes.

Run it with::

    python -m engine.delivery_xml_agent.workflow \\
      --projection-manifest onboarding_output/client_001/run_pre_xml_final_check_3/output/projection/50_projection_manifest.json

It never re-runs Projection / Validation / Transformation / Onboarding, never
mutates upstream artefacts, never invokes the frozen Gate 5 XML builder or the
Gate 4b mutator, never silently fills a blocked value, and never produces
production XML in v1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.delivery_xml_agent.delivery_xml_agent import (  # noqa: E402
    build_delivery_package,
    ProjectionHandoffError,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trakt Delivery/XML Agent v1 — consume the Projection package and "
        "produce a delivery package (NO production XML). Refuses XML unless all "
        "delivery-readiness gates pass.")
    p.add_argument(
        "--projection-manifest", required=True,
        help="Path to the projection manifest "
        "(output/projection/50_projection_manifest.json).")
    p.add_argument("--regime-config", default="",
                   help="Annex 2 delivery rules YAML (defaults to the manifest value).")
    p.add_argument("--esma-code-order", default="",
                   help="ESMA code order YAML (defaults to config/system/esma_code_order.yaml).")
    p.add_argument("--registry", default="",
                   help="Canonical field registry YAML (defaults to the manifest value).")
    p.add_argument("--field-universe", default="",
                   help="Annex 2 field universe YAML "
                   "(defaults to config/regime/annex2_field_universe.yaml).")
    p.add_argument("--allow-xml-preview", action="store_true",
                   help="Permit a guarded XML PREVIEW (65/66) ONLY if every readiness "
                   "gate passes. Never bypasses the gates; never produces production XML.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_delivery_package(
            args.projection_manifest,
            regime_config_path=args.regime_config,
            esma_code_order_path=args.esma_code_order,
            registry_path=args.registry,
            field_universe_path=args.field_universe,
            allow_xml_preview=args.allow_xml_preview,
        )
    except ProjectionHandoffError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    m = result["manifest"]
    print(f"Delivery package written to: {result['delivery_dir']}")
    print(f"  delivery_xml_ran                = {m['delivery_xml_ran']}")
    print(f"  delivery_normalisation_complete = {m['delivery_normalisation_complete']}")
    print(f"  xml_generation_allowed          = {m['xml_generation_allowed']}")
    print(f"  xml_generated                   = {m['xml_generated']}")
    print(f"  ready_for_xml_delivery          = {m['ready_for_xml_delivery']}")
    print(f"  delivery frame rows             = {m['frame_row_count']}")
    print(f"  deliverable / blocked           = "
          f"{m['deliverable_row_count']} / {m['blocked_row_count']}")
    print(f"  delivery issues                 = {m['issue_count']}")
    print(f"  next_agent                      = {m['next_agent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
