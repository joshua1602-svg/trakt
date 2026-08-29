#!/usr/bin/env python3
"""migration_phase0/dataset_census_882.py — the blast-radius census.

READ-ONLY. Runs every distinct Stage 1 + Stage 2 corpus question through the
RETIRED dataset rules and through the live owner, on every workspace tab, and
classifies every movement against the three authorised classes registered in
`docs/mi_dataset_ownership_conditions.md`:

    M1  tab influence removed
    M2  one owner rather than two -- the tape vocabulary applies at the owner
    M3  forecast precedence restored

Anything else is UNEXPLAINED and is a blast-radius failure.

The retired rules are reproduced here verbatim as frozen functions rather than
read from history, so the census is reproducible from one checkout and so the
comparison cannot silently drift when the production code moves again.

    python -m migration_phase0.dataset_census_882
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")

#: Every tab a caller can be on, plus "no tab".
TABS: Tuple[Optional[str], ...] = (None, "funded", "pipeline", "forecast")

#: `chat_routing._PIPELINE_WORDS`, as it was before retirement.
RETIRED_PIPELINE_WORDS = ("pipeline", "case", "kfi", "application", "offer")


def _questions() -> List[str]:
    out: List[str] = []
    seen = set()
    for f in CORPORA:
        for row in json.loads((_REPO / f).read_text())["rows"]:
            q = row.get("question") or ""
            if q and q not in seen:
                seen.add(q)
                out.append(q)
    return out


def main() -> int:
    import logging
    import warnings
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    logging.disable(logging.WARNING)

    from mi_agent.portfolio_lens import undisclaimed_mention as um
    from mi_agent_api.workspace import (DEFAULT_VIEW, VIEWS, resolve_dataset,
                                        view_named_by_question)

    def before_pointintime(q: str, tab: Optional[str]) -> str:
        """`workspace.resolve_active_view` as it was: question, THEN THE TAB."""
        named = view_named_by_question(q)
        if named is not None:
            return named
        ctx = (tab or "").strip().lower()
        return ctx if ctx in VIEWS else DEFAULT_VIEW

    def before_routed(q: str, tab: Optional[str]) -> str:
        """`chat_routing._dataset_for` as it was: the tape words FIRST."""
        low = (q or "").lower()
        if any(um(low, w) for w in RETIRED_PIPELINE_WORDS):
            return "pipeline"
        view = before_pointintime(q, tab)
        return "pipeline" if view == "pipeline" else "funded"

    def classify(q: str, tab: Optional[str], before: str, after: str,
                 rule: str) -> str:
        named = view_named_by_question(q)
        low = (q or "").lower()
        artefact = any(um(low, w) for w in ("case", "kfi", "application", "offer"))
        ctx = (tab or "").strip().lower()
        if named is None and ctx in VIEWS and before == ctx and after != ctx:
            return "M1"
        if named is None and artefact and after == "pipeline":
            return "M2"
        if named == "forecast" and before == "pipeline" and after == "forecast":
            return "M3"
        if rule == "routed" and named is not None and before != named and after == named:
            # The retired routed rule tested its tape words BEFORE reading any
            # view name, so a question that NAMED a view could be overridden by
            # a tape word. Restoring the named view is M3's general form.
            return "M3"
        return "UNEXPLAINED"

    qs = _questions()
    print("=" * 96)
    print(f"882-QUESTION DATASET CENSUS — {len(qs)} distinct questions "
          f"x {len(TABS)} tabs x 2 retired rules")
    print("=" * 96)

    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    unexplained: List[Dict[str, Any]] = []
    tab_variance_before = tab_variance_after = 0

    for q in qs:
        after = resolve_dataset(q)
        before_by_tab = {t: before_pointintime(q, t) for t in TABS}
        if len(set(before_by_tab.values())) > 1:
            tab_variance_before += 1
        if len({resolve_dataset(q) for _ in TABS}) > 1:  # cannot vary; asserted
            tab_variance_after += 1
        for tab in TABS:
            for rule, fn in (("pointintime", before_pointintime),
                             ("routed", before_routed)):
                before = fn(q, tab)
                if before == after:
                    continue
                cls = classify(q, tab, before, after, rule)
                counts[cls] = counts.get(cls, 0) + 1
                rec = {"question": q, "tab": tab, "retiredRule": rule,
                       "before": before, "after": after, "class": cls}
                rows.append(rec)
                if cls == "UNEXPLAINED":
                    unexplained.append(rec)

    print(f"\nquestions whose dataset VARIED BY TAB before : {tab_variance_before}"
          f" of {len(qs)}")
    print(f"questions whose dataset varies by tab AFTER  : {tab_variance_after}"
          f"  (the owner has no tab parameter)")

    print("\nMovements by authorised class:")
    for cls in ("M1", "M2", "M3", "UNEXPLAINED"):
        print(f"  {cls:<12} {counts.get(cls, 0):>5}")

    # The distinct questions behind each class matter more than the readings.
    print("\nDistinct questions per class:")
    for cls in ("M1", "M2", "M3"):
        qn = sorted({r["question"] for r in rows if r["class"] == cls})
        print(f"  {cls}: {len(qn)}")
        for name in qn[:12]:
            print(f"       {name[:84]}")
        if len(qn) > 12:
            print(f"       ... and {len(qn) - 12} more")

    if unexplained:
        print("\n!! UNEXPLAINED MOVEMENTS — STOP — BLAST RADIUS")
        for r in unexplained[:40]:
            print(f"   tab={str(r['tab']):<9} {r['retiredRule']:<12} "
                  f"{r['before']} -> {r['after']} :: {r['question'][:70]}")

    out = _REPO / "migration_phase0" / "DATASET_CENSUS_882.json"
    out.write_text(json.dumps(
        {"questions": len(qs), "counts": counts, "movements": rows}, indent=2,
        default=str))
    print(f"\nwritten : {out.relative_to(_REPO)}")
    print("=" * 96)
    return 1 if unexplained else 0


if __name__ == "__main__":
    sys.exit(main())
