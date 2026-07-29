#!/usr/bin/env python3
"""
run_annex2_golden.py
====================

Run the full-scale Annex 2 golden reproduction and print the metrics next to the
historically recorded run.

Separated from the normal test suite because it produces roughly 190 MB of XML
and takes a couple of minutes — too heavy for every CI invocation, and the
reduced CI fixture must not be allowed to imply it was run.

    python scripts/run_annex2_golden.py
    python scripts/run_annex2_golden.py --records 11035 --keep

What it proves: the wrapped route produces a schema-valid artefact at the
recorded scale, with every no-data value attributed either to the source data or
to a named builder rule.

What it does not prove: anything about the *original client dataset*. That tape
is not in this repository. The input is generated deterministically from the
committed synthetic projected data — real scale and real components, different
data. See ``docs/annex_delivery_agent.md`` §13.
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core.context import ExecutionContext  # noqa: E402

from engine.annex_delivery_agent.agent import AnnexDeliveryAgent  # noqa: E402
from engine.annex_delivery_agent.contracts import (  # noqa: E402
    DeliveryRequest, DeliveryResult,
)
from engine.annex_delivery_agent.persistence import DeliveryRunStore  # noqa: E402
from scripts.generate_annex2_scale_fixture import build_scale_fixture  # noqa: E402

DEFAULT_SOURCE = (_REPO_ROOT / "synthetic_demo" / "output"
                  / "SYNTHETIC_ERE_Portfolio_012026_ESMA_Annex2_projected.csv")

#: The run recorded before this agent existed, reproduced here for comparison.
HISTORICAL = {
    "records": 11_035,
    "fields": 105,
    "artefact_mb": 208.8,
    "seconds": 171.5,
    "peak_rss_mb": 772.7,
    "xsd": "PASSED",
}


def run(source: Path, work_root: Path, records: int) -> dict:
    projected = build_scale_fixture(source, work_root / "scale_projected.csv",
                                    target_records=records)
    agent = AnnexDeliveryAgent(store=DeliveryRunStore(work_root / "store"))
    context = ExecutionContext.for_internal("ERE", actor_id="golden-regression")

    started = time.perf_counter()
    governed = agent.prepare(context, DeliveryRequest(
        annex_id="ESMA_Annex2", reporting_date="2026-01-31",
        projected_input=str(projected), output_root=str(work_root / "store")))
    elapsed = time.perf_counter() - started

    if governed.result is None:
        raise SystemExit(
            f"golden run blocked: {governed.error.message if governed.error else 'unknown'}")

    result = DeliveryResult.from_dict(governed.result)
    evidence = result.transformation_evidence or {}
    return {
        "status": result.status,
        "records": result.record_count,
        "projected_fields": result.projected_field_count,
        "delivered_fields": result.delivered_field_count,
        "artefact_mb": round(result.artefact.size_bytes / (1024 * 1024), 1),
        "artefact": result.artefact.path,
        "xsd": "PASSED" if result.schema_validation.valid else "FAILED",
        "xsd_errors": result.schema_validation.error_count,
        "seconds": round(elapsed, 1),
        "stage_seconds": {t.stage: round(t.seconds, 1) for t in result.timings},
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "automatic_fills": evidence.get("automatic_fill_count"),
        "coercions": evidence.get("coercion_count"),
        "omissions": evidence.get("omission_count"),
        "nd_reconciliation": {
            k: v for k, v in (evidence.get("nd_reconciliation") or {}).items()
            if k != "per_field_in_prepared_data"},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Full-scale Annex 2 golden reproduction.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE),
                        help="Committed projected Annex 2 CSV to scale up.")
    parser.add_argument("--records", type=int, default=HISTORICAL["records"])
    parser.add_argument("--work-dir", default=None,
                        help="Where to write the run (default: a temporary directory).")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the ~190 MB output instead of deleting it.")
    args = parser.parse_args(argv)

    work_root = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(
        prefix="annex2_golden_"))
    work_root.mkdir(parents=True, exist_ok=True)

    try:
        metrics = run(Path(args.source), work_root, args.records)
        print("\nGolden Annex 2 reproduction")
        print(json.dumps(metrics, indent=2))
        print("\nHistorically recorded run (different source data — see docs §13)")
        print(json.dumps(HISTORICAL, indent=2))

        ok = metrics["xsd"] == "PASSED" and metrics["records"] == args.records
        recon = metrics["nd_reconciliation"]
        ok = ok and recon.get("unattributed") == 0
        print(f"\nResult: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    finally:
        if not args.keep and args.work_dir is None:
            shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
