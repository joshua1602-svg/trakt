#!/usr/bin/env python3
"""run_regulatory_watch_demo.py — end-to-end ESMA Annex 2 Regulatory Watch demo.

Runs the whole Stage 1 flow locally and offline::

    authoritative snapshot -> normalized Annex 2 spec -> comparator
                           -> Trakt impact report -> JSON + Markdown

The candidate is built by applying ONE controlled regulatory change to a COPY
of the vendored ESMA schema: the code value ``FXRL`` ("fixed rate loan") is
withdrawn from ``InterestRateType2Code``. That is a change Trakt is genuinely
exposed to — the effective Annex 2 contract maps the ERM
back-book default ``interest_rate_type = Fixed`` onto ``FXRL`` for RREL42 — so
the demo shows a delta with real implementation consequences rather than a
cosmetic one.

Nothing outside ``--out-dir`` is written. The vendored artefacts and every
active Annex 2 config are opened read-only; the mutation is applied to a copy.

Usage::

    python scripts/run_regulatory_watch_demo.py \
        [--out-dir output/regulatory_watch/demo]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from regulatory_watch import PARSER_VERSION                       # noqa: E402
from regulatory_watch.annex2_spec import parse_annex2_spec        # noqa: E402
from regulatory_watch.comparator import compare                   # noqa: E402
from regulatory_watch.contracts import (                          # noqa: E402
    ARTEFACT_DISCLOSURE_WORKBOOK, ARTEFACT_XML_SCHEMA,
    NO_IMPLEMENTATION_CHANGE,
)
from regulatory_watch.impact import TraktImplementationIndex, assess  # noqa: E402
from regulatory_watch.manifest import load_manifest               # noqa: E402
from regulatory_watch.report import build_report, write_reports   # noqa: E402
from regulatory_watch.snapshots import SnapshotStore, snapshot_manifest  # noqa: E402

DEFAULT_OUT = _REPO / "output" / "regulatory_watch" / "demo"

#: The controlled regulatory change applied to the candidate schema copy.
WITHDRAWN_CODE = "FXRL"
WITHDRAWN_FROM = "InterestRateType2Code"
#: Matches the enumeration whether it is self-closing or carries annotations.
_ENUM_RE = re.compile(
    r'[ \t]*<xs:enumeration value="%s"\s*(?:/>|>.*?</xs:enumeration>)\r?\n?'
    % WITHDRAWN_CODE, re.DOTALL)

#: Fixed so the demo output is byte-reproducible run to run.
RETRIEVED_AT = "2026-01-01T00:00:00Z"
GENERATED_AT = "2026-01-01T00:00:00Z"


def _banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _withdraw_enum_value(schema: Path, target: Path) -> None:
    """Copy the schema and remove one enumeration value from one simple type."""
    text = schema.read_text(encoding="utf-8")
    marker = f'<xs:simpleType name="{WITHDRAWN_FROM}">'
    start = text.index(marker)
    end = text.index("</xs:simpleType>", start)
    block = text[start:end]
    stripped, count = _ENUM_RE.subn("", block, count=1)
    if not count:
        raise SystemExit(f"{WITHDRAWN_CODE} not found in {WITHDRAWN_FROM}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text[:start] + stripped + text[end:], encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    workbook = _REPO / manifest.of_type(ARTEFACT_DISCLOSURE_WORKBOOK)[0].local_path
    schema = _REPO / manifest.of_type(ARTEFACT_XML_SCHEMA)[0].local_path
    store = SnapshotStore(root=out_dir / "snapshots", regime=manifest.regime)

    # ------------------------------------------------------------------ #
    _banner("1. BASELINE — snapshot the authoritative ESMA artefacts")
    baseline_snaps = snapshot_manifest(store, manifest.artefacts, _REPO,
                                       RETRIEVED_AT, PARSER_VERSION)
    for snap in baseline_snaps:
        state = snap.sha256[:16] if snap.ok else snap.retrieval_status
        print(f"  {snap.artefact_id:42} {state}"
              + (f"  ({snap.retrieval_detail})" if not snap.ok else ""))
    baseline = parse_annex2_spec(workbook, schema,
                                 spec_version="auth.099.001.04 / workbook 1.3.1",
                                 snapshots=baseline_snaps)
    print(f"  normalized {len(baseline.fields)} Annex 2 fields "
          f"({baseline.parser_version})")

    # ------------------------------------------------------------------ #
    _banner("2. CANDIDATE — one controlled regulatory change on a schema copy")
    candidate_schema = out_dir / "candidate" / schema.name
    _withdraw_enum_value(schema, candidate_schema)
    print(f"  withdrew code value {WITHDRAWN_CODE} from {WITHDRAWN_FROM}")
    print(f"  candidate schema: {candidate_schema}")

    # Snapshot the SAME artefact set on both sides — only the schema differs —
    # so the source comparison is like-for-like.
    schema_artefact_id = manifest.of_type(ARTEFACT_XML_SCHEMA)[0].artefact_id
    candidate_snaps = []
    for artefact in manifest.artefacts:
        if artefact.artefact_id == schema_artefact_id:
            candidate_snaps.append(store.snapshot_local(
                artefact, candidate_schema, RETRIEVED_AT, PARSER_VERSION))
        elif artefact.local_path:
            candidate_snaps.append(store.snapshot_local(
                artefact, _REPO / artefact.local_path, RETRIEVED_AT,
                PARSER_VERSION))
        else:
            candidate_snaps.append(SnapshotStore.failed_snapshot(
                artefact, RETRIEVED_AT, PARSER_VERSION,
                "no local copy vendored and retrieval is disabled"))
    for snap in candidate_snaps:
        state = snap.sha256[:16] if snap.ok else snap.retrieval_status
        print(f"  {snap.artefact_id:42} {state}")
    candidate = parse_annex2_spec(
        workbook, candidate_schema,
        spec_version="auth.099.001.04 / workbook 1.3.1 + candidate schema",
        snapshots=candidate_snaps)

    # ------------------------------------------------------------------ #
    _banner("3. COMPARATOR — deterministic regulatory delta")
    deltas = compare(baseline, candidate)
    if not deltas:
        print("  no regulatory delta detected")
    for delta in deltas:
        print(f"  {delta.code}  {delta.change_type}  "
              f"severity={delta.severity}  confidence={delta.confidence}")
        print(f"    old: {delta.old_value}")
        print(f"    new: {delta.new_value}")
        print(f"    evidence: {delta.evidence[0].artefact_id} "
              f"{delta.evidence[0].locator}")

    # ------------------------------------------------------------------ #
    _banner("4. IMPACT — which Trakt Annex 2 components are likely affected")
    index = TraktImplementationIndex.load(_REPO)
    impacts = assess(deltas, index=index)
    for finding in impacts:
        print(f"  {finding.code}  {finding.component:42} {finding.status}")
        print(f"    {finding.rationale}")
        if finding.locations:
            print(f"    locations: {', '.join(finding.locations)}")

    # ------------------------------------------------------------------ #
    _banner("5. REPORTS")
    report = build_report(
        baseline, candidate, deltas, impacts, generated_at=GENERATED_AT,
        evidence={
            "demonstration": (
                f"candidate schema withdraws {WITHDRAWN_CODE} from "
                f"{WITHDRAWN_FROM}; the workbook is byte-identical on both "
                f"sides"),
            "source_manifest": manifest.path,
            "mutating_writes": "none — Stage 1 is observational",
        })
    paths = write_reports(report, out_dir)
    print(f"  outcome                    {report.outcome}")
    print(f"  specification_current      {report.specification_current}")
    print(f"  gating sources             {report.gating_source_status}")
    print(f"  corroborating sources      {report.corroborating_source_status}")
    print(f"  spec_status                {report.spec_status}")
    print(f"  regulatory_delta_count     {report.regulatory_delta_count}")
    print(f"  implementation_impact_count {report.implementation_impact_count}")
    print(f"  JSON      -> {paths['json']}")
    print(f"  Markdown  -> {paths['markdown']}")

    actionable = [i for i in impacts if i.status != NO_IMPLEMENTATION_CHANGE]
    print()
    print(f"Actionable impact findings: {len(actionable)}; "
          f"no active configuration was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
