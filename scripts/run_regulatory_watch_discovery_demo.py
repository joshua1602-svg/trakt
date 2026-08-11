#!/usr/bin/env python3
"""run_regulatory_watch_discovery_demo.py — Stage 1B weekly discovery demo.

Runs one weekly-style ESMA publication sweep offline, over a fixture week that
deliberately contains all three shapes Stage 1B must tell apart:

  * an IRRELEVANT publication (a markets newsletter) — rejected by the
    deterministic filter, with its reason recorded;
  * RELEVANT NARRATIVE publications (a final report, a consultation, a Q&A) —
    regulatory developments with no technical change;
  * a TECHNICAL Annex 2 publication (the auth.099 schema/templates package) —
    which matches an allowlisted Stage 1A source and therefore asks Stage 1A to
    run.

For the technical item the demo then runs **Stage 1A for real** (via
``scripts/run_regulatory_watch_demo.py``) and attaches its report, showing the
handover: Stage 1B says *what happened upstream*, Stage 1A says *which fields
changed*. Stage 1B never computes a delta.

Nothing outside ``--out-dir`` is written, and no active configuration is
touched by either stage.

Usage::

    python scripts/run_regulatory_watch_discovery_demo.py \
        [--out-dir output/regulatory_watch/discovery_demo]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from regulatory_watch.discovery.contracts import (                 # noqa: E402
    MATCHES_STAGE1A_SOURCE, NO_TECHNICAL_ARTEFACT, TECHNICAL_SPEC_CHANGE,
)
from regulatory_watch.discovery.contracts import (                 # noqa: E402
    SOURCE_FETCH_FAILED, SOURCE_OK, SourceOutcome,
)
from regulatory_watch.discovery.pipeline import SourcePayload, run_discovery  # noqa: E402
from regulatory_watch.discovery.report import write_reports        # noqa: E402
from regulatory_watch.discovery.retrieval import sha256_bytes      # noqa: E402
from regulatory_watch.discovery.sources import load_publication_sources  # noqa: E402
from regulatory_watch.discovery.state import SeenStore             # noqa: E402

DEFAULT_OUT = _REPO / "output" / "regulatory_watch" / "discovery_demo"
FIXTURE_DIR = (_REPO / "tests" / "fixtures" / "regulatory_watch"
               / "publications")

#: Fixed so the demo output is byte-reproducible run to run.
WEEK_ONE = "2026-02-16T00:00:00Z"
WEEK_TWO = "2026-02-23T00:00:00Z"


def _banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _payloads(sources, when: str):
    out = []
    for source in sources.sources:
        matches = sorted(FIXTURE_DIR.glob(f"{source.source_id}.*"))
        if not matches:
            out.append(SourcePayload(source, None, SourceOutcome(
                source_id=source.source_id, criticality=source.criticality,
                outcome=SOURCE_FETCH_FAILED, retrieved_at=when,
                detail=f"no fixture for {source.source_id}")))
            continue
        payload = matches[0].read_bytes()
        out.append(SourcePayload(source, payload, SourceOutcome(
            source_id=source.source_id, criticality=source.criticality,
            outcome=SOURCE_OK, retrieved_at=when,
            content_sha256=sha256_bytes(payload),
            detail=f"fixture: {matches[0].name}")))
    return out


def _run_stage1a(out_dir: Path) -> Path:
    """Run Stage 1A for real and return its report path (or a missing path)."""
    stage1a_dir = out_dir / "stage1a"
    result = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "run_regulatory_watch_demo.py"),
         "--out-dir", str(stage1a_dir)],
        capture_output=True, text=True, cwd=str(_REPO))
    if result.returncode != 0:
        print("  Stage 1A demo failed; the linkage will show it as unrun:")
        print("  " + (result.stderr or result.stdout).strip()[:400])
    return stage1a_dir / "esma_annex2_regulatory_watch.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = load_publication_sources()
    state_path = out_dir / "seen_publications.json"

    # ------------------------------------------------------------------ #
    _banner("1. UPSTREAM SOURCES — the explicit ESMA allowlist")
    for source in sources.sources:
        print(f"  {source.source_id:44} {source.criticality:14} "
              f"{source.parser:13} enabled={source.enabled}")
    print(f"  retrieval_enabled={sources.retrieval_enabled} "
          f"(offline; this run reads committed fixtures)")

    # ------------------------------------------------------------------ #
    _banner("2. WEEK ONE — discover, dedupe, triage, classify")
    store = SeenStore.load(state_path)
    week_one = run_discovery(sources, _payloads(sources, WEEK_ONE), store,
                             WEEK_ONE)
    store.save()
    report = week_one.report
    print(f"  discovery status           {report.discovery_status}")
    print(f"  sources checked / failed   {report.sources_checked} / "
          f"{report.sources_failed}")
    print(f"  publication sightings      {report.publications_seen}")
    print(f"  new publications           {report.new_publications}")
    print(f"  potentially relevant       {report.potentially_relevant}")
    print(f"  review required            {report.review_required_count}")
    print(f"  technical-spec candidates  "
          f"{report.technical_spec_change_candidates}")

    print()
    print("  IRRELEVANT — rejected by the deterministic filter:")
    for item in report.rejected:
        print(f"    · {item['title'][:64]}")
        print(f"      reason: {item['reason'][:90]}")

    print()
    print("  CLASSIFIED developments:")
    for development in week_one.developments:
        linkage = (development.technical_spec_linkage or {}).get("status")
        marker = "TECHNICAL" if linkage != NO_TECHNICAL_ARTEFACT else "narrative"
        print(f"    · [{marker:9}] {development.publication_type:24} "
              f"{development.development_status}")
        print(f"      {development.publication['title'][:70]}")
        print(f"      why: {development.relevance_reason[:88]}")

    # ------------------------------------------------------------------ #
    _banner("3. STAGE 1A HANDOVER — the technical item only")
    technical = [d for d in week_one.developments
                 if d.development_status == TECHNICAL_SPEC_CHANGE
                 and (d.technical_spec_linkage or {}).get("status")
                 == MATCHES_STAGE1A_SOURCE]
    if not technical:
        print("  no technical-spec-change candidate matched a Stage 1A source")
        stage1a_report = None
    else:
        target = technical[0]
        link = target.technical_spec_linkage["artefact_links"][0]
        print(f"  {target.publication['title'][:70]}")
        print(f"    matches Stage 1A artefact(s): {link['stage1a_artefact_id']}")
        print(f"    Stage 1B verdict: {target.development_status} — "
              f"Stage 1B does NOT compute field deltas")
        print()
        print("  running Stage 1A for real …")
        stage1a_report = _run_stage1a(out_dir)

    # ------------------------------------------------------------------ #
    _banner("4. THE SAME WEEK, RE-CLASSIFIED WITH THE STAGE 1A RESULT ATTACHED")
    # A FRESH store: week one has already been committed, so replaying it
    # against the committed store would (correctly) produce nothing. This
    # re-classification is what the operator sees after Stage 1A has run.
    linked = run_discovery(
        sources, _payloads(sources, WEEK_ONE),
        SeenStore(path=out_dir / "replay_state.json"), WEEK_ONE,
        stage1a_reports=({technical[0].development_id: stage1a_report}
                         if technical and stage1a_report else None),
        commit_state=False)
    attached = False
    for development in linked.developments:
        linkage = development.technical_spec_linkage or {}
        if linkage.get("regulatory_delta_count") is not None:
            attached = True
            print(f"  {development.publication['title'][:64]}")
            print(f"    Stage 1A outcome            "
                  f"{linkage['stage1a_outcome']}")
            print(f"    regulatory deltas           "
                  f"{linkage['regulatory_delta_count']}")
            print(f"    Trakt implementation impacts "
                  f"{linkage['implementation_impact_count']}")
            print(f"    baseline snapshots          "
                  f"{linkage['stage1a_baseline_snapshot_ids']}")
            print(f"    candidate snapshots         "
                  f"{linkage['stage1a_candidate_snapshot_ids']}")
    if not attached:
        print("  no Stage 1A report was attached to any development")
    paths = write_reports(linked.report, out_dir)

    # ------------------------------------------------------------------ #
    _banner("5. WEEK TWO — the same feed again (deduplication)")
    store = SeenStore.load(state_path)
    week_two = run_discovery(sources, _payloads(sources, WEEK_TWO), store,
                             WEEK_TWO)
    store.save()
    print(f"  publication sightings      {week_two.report.publications_seen}")
    print(f"  new publications           {week_two.report.new_publications}")
    print(f"  previously seen (skipped)  "
          f"{week_two.report.previously_seen_publications}")
    print(f"  NEW development records    {len(week_two.developments)}")
    assert not week_two.developments, "week two must create no duplicates"

    # ------------------------------------------------------------------ #
    _banner("6. REPORTS")
    print(f"  JSON      -> {paths['json']}")
    print(f"  Markdown  -> {paths['markdown']}")
    print(f"  State     -> {state_path} ({len(store)} publication(s) known)")
    print()
    print("Stage 1B classified upstream publications and asked Stage 1A to run.")
    print("No active Annex 2 configuration was read for change or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
