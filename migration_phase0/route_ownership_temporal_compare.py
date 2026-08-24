#!/usr/bin/env python3
"""migration_phase0/route_ownership_temporal_compare.py — what the route owns.

READ-ONLY. Enumerates candidate questions x caller datasets (workspace tabs)
and records, from EXECUTED routing rather than from wording similarity, which
questions the shipped ``temporal_compare`` recogniser actually claims.

The tab axis is the one that matters here and did not matter for the previous
four conversions. ``temporal_compare`` is the first converted route whose
dataset decision reads the CALLER'S WORKSPACE TAB as well as the question, so a
surface enumerated at one tab would hide half of the route's behaviour.

Cases prefixed ``X`` are deliberately listed and expected NOT to be claimed.
They are the discipline: a surface that quietly grows to include questions the
route does not own manufactures an equivalence that means nothing.

    python -m migration_phase0.route_ownership_temporal_compare
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: (case, question, expected_owner_or_None). ``None`` means "claimed by
#: temporal_compare"; a string names why it should NOT be claimed.
CASES: Tuple[Tuple[str, str, Any], ...] = (
    # --- DELIVERED comparisons, on periods this book actually holds --------
    #
    # The cases below them ask for October/November/September, which this
    # fixture does not carry, so every one of them REFUSES. An equivalence
    # measured on refusals alone is vacuous — both sides decline for the same
    # reason and nothing about the calculation is exercised — and Conversion 1
    # already reported one vacuous pass of exactly that shape. These seven run
    # the delivered path with real numbers, one per governed measure family.
    ("A1", "Compare April and May funded balance.", None),
    ("A2", "Compare May and June funded balance.", None),
    ("A3", "Compare April and June funded balance.", None),
    ("A4", "Compare May and June loan count.", None),
    ("A5", "Compare May and June WA current LTV.", None),
    ("A6", "Compare May and June average interest rate.", None),
    ("A7", "Compare May and June borrower age.", None),
    # A bare `case` is a funded loan, end to end and on real numbers — the
    # house rule settled during prerequisite closure, pinned here on the
    # delivered path rather than only at the resolver.
    ("A8", "Compare May and June case count.", None),
    # --- THE RELATIVE-PERIOD PATH, delivered -------------------------------
    #
    # `temporal_compare._match_period` resolves `latest`/`current` to the last
    # governed period and the `_RELATIVE_PRIOR` vocabulary to the one before it.
    # The independent C5 audit found that branch exercised by NO delivered case:
    # every relative-period case on the surface refused for an unrelated reason,
    # so a whole resolution path sat behind a green equivalence. These two run
    # it with real numbers, on the periods this fixture actually holds.
    ("V2", "Compare this month and last month funded balance.", None),
    # --- OMITTED BY THE ORIGINAL DECLARATION, found by the audit ------------
    #
    # Both parse to `temporal_mode == "compare"` and are owned at runtime, and
    # neither was declared. They are refusals here — the periods and the tape
    # are unavailable — but a surface that silently under-covers its own corpus
    # is how a denominator stops meaning anything.
    ("O1", "Compare October and November WA LTV.", None),
    ("O2", "Compare latest pipeline with prior pipeline.", None),
    # --- the plain two-period comparison -----------------------------------
    ("T1", "Compare October and November funded balance.", None),
    ("T2", "Compare September and October funded balance", None),
    ("T3", "Compare the funded balance in October and November.", None),
    ("T4", "Compare October and November loan count.", None),
    # --- measure variants ---------------------------------------------------
    ("M1", "Compare October and November WA current LTV.", None),
    ("M2", "Compare October and November average interest rate.", None),
    # --- the pipeline vocabulary -------------------------------------------
    ("P1", "Compare October and November pipeline amount.", None),
    ("P2", "How did the pipeline amount change from last week?", None),
    ("P3", "Compare October and November case count.", None),
    ("P4", "Compare October and November KFI count.", None),
    ("P5", "Compare October and November application count.", None),
    # --- the disclaiming clause (B21) --------------------------------------
    ("D1", "Compare October and November balance, excluding pipeline cases.", None),
    # --- refusal surface ----------------------------------------------------
    ("R1", "Compare October and December funded balance.", None),
    # --- NOT claimed, and named --------------------------------------------
    ("X1", "What is the funded balance?", "a point-in-time question, no periods"),
    ("X2", "How has the funded balance moved?", "period movement — the governed "
                                                "composite, which sits earlier"),
    ("X3", "Show the funded balance trend over the last six months",
     "evolution — a series, not a pair"),
    ("X4", "What is the geographic exposure?", "geo_exposure"),
    ("X5", "Funded balance bridge from October to November",
     "the funded bridge route"),
    # --- MEASURED CORRECTIONS. Both were first declared OWNED and both were
    #     wrong; the instrument said so and the declaration was fixed to what
    #     executed routing shows, not the other way round.
    ("X6", "Compare October and November forecast balance.",
     "analytical_composition claims it first — the third view never reaches "
     "this route by this phrasing, so `forecast` is NOT on the owned surface"),
    # MEASURED CORRECTION. Declared owned; `period_change_analysis` sits at
    # priority 85 and claims it first. The instrument caught the declaration,
    # which is what it is for.
    ("X8", "Compare the latest funded balance with the prior period.",
     "period_change_analysis claims it first, at priority 85"),
    ("X7", "Compare Narnia and November funded balance.",
     "no route at all: an unresolvable period token leaves `compare_periods` "
     "short of two, so the recogniser never fires"),
)

#: The workspace tab the caller is on. This is the second owner's other input.
DATASETS: Tuple[str, ...] = ("funded", "pipeline")


def _env() -> str:
    import logging
    import warnings
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


def main() -> int:
    client_id = _env()
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    ctx = ExecutionContext.for_internal(client_id)
    print("=" * 78)
    print("temporal_compare — ROUTE OWNERSHIP, from executed routing")
    print("=" * 78)

    records: List[Dict[str, Any]] = []
    claimed = misclaimed = 0
    for case, question, expected_other in CASES:
        by_ds: Dict[str, Any] = {}
        for ds in DATASETS:
            env = execute_governed_mi_query(
                MiQueryRequest(question=question, dataset_context=ds), ctx).result or {}
            md = env.get("metadata") or {}
            recon = env.get("reconciliation") or {}
            by_ds[ds] = {
                "route": md.get("route"),
                "ok": env.get("ok"),
                "controlledRefusal": env.get("controlledRefusal"),
                "datasetInReconciliation": recon.get("dataset"),
                "answer": (env.get("answer") or "")[:220],
                "artifactKinds": [a.get("type") for a in (env.get("artifacts") or [])],
            }
        routes = {v["route"] for v in by_ds.values()}
        owns = routes == {"temporal_compare"}
        want_owned = expected_other is None
        agree = owns == want_owned
        if owns:
            claimed += 1
        if not agree:
            misclaimed += 1
        flag = "   " if agree else "!! "
        print(f"\n{flag}{case}  {question!r}")
        print(f"      routes across tabs   : {sorted(r or '-' for r in routes)}")
        if want_owned:
            print("      expected             : temporal_compare")
        else:
            print(f"      expected NOT claimed : {expected_other}")
        for d, v in by_ds.items():
            print(f"      [tab={d:<8}] ok={str(v['ok']):<5} ds={str(v['datasetInReconciliation']):<9}"
                  f" {str(v['answer'])[:100]}")
        records.append({"case": case, "question": question,
                        "expectedOther": expected_other,
                        "ownedByTemporalCompare": owns, "asExpected": agree,
                        "byDataset": by_ds})

    out = _REPO / "migration_phase0" / "ROUTE_OWNERSHIP_TEMPORAL_COMPARE.json"
    out.write_text(json.dumps({"cases": records}, indent=2, default=str))

    print("\n" + "=" * 78)
    print(f"cases enumerated          : {len(CASES)}  x {len(DATASETS)} tabs "
          f"= {len(CASES) * len(DATASETS)} renders")
    print(f"claimed by temporal_compare: {claimed}")
    print(f"deliberately NOT claimed  : {len(CASES) - claimed}")
    print(f"DISAGREEING WITH EXPECTED : {misclaimed}")
    print(f"written                   : {out.relative_to(_REPO)}")
    print("=" * 78)
    if misclaimed:
        print("\nThe surface is NOT what this instrument declared. Every "
              "disagreement is marked !! above and must be explained before "
              "any equivalence measured on this surface means anything.")
    return 1 if misclaimed else 0


if __name__ == "__main__":
    sys.exit(main())
