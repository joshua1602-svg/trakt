#!/usr/bin/env python3
"""migration_phase0/ranking_recognition_census.py — every ranking-language question.

READ-ONLY. Phase 5. The denominator is ALL 97 corpus questions carrying ranking
language, not the eight `period_change` happens to own, because the question
under test is what the PRODUCT should do, not what the legacy route does.

For each question the table records, side by side:

    contract       what the authoritative interpretation says
    legacy route   what the shipped `period_change` resolver says
    execution      which route actually answered, and whether it delivered

and then classifies the disagreement. THE LEGACY ROUTE IS NOT THE AUTHORITY:
where the two differ, the row records which one matches business intent, so
"make the contract agree with the route" is never the default resolution.

Intent classification is DERIVED, not hand-labelled, from three facts the
question itself carries — a superlative or rank instruction, a change verb or a
named span, and a named dimension — so the census can be re-run and audited
rather than trusted.

    python -m migration_phase0.ranking_recognition_census [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from migration_phase0.route_ownership_period_change import (  # noqa: E402
    PERIOD_CHANGE_ROUTES, _grade, _questions, funded_runs)

#: A question is about a CHANGE when it uses a change verb or names a span.
#: Deliberately wider than any single recogniser, because the point of the
#: census is to find questions a recogniser misses.
_CHANGE_RE = re.compile(
    r"\b(?:grew|grow|grown|growth|increas|ris(?:e|en|ing)|gain|expand|added|add|"
    r"declin|f[ae]ll|drop|decreas|shr(?:ank|unk|ink)|reduc|lost|los(?:e|ing)|"
    r"chang|mov(?:e|ed|ement)|since|versus|compared|between|trend|over the last|"
    r"month[- ]on[- ]month|year[- ]on[- ]year)\b", re.I)
#: A LEVEL superlative — "the largest region" is a ranking of a level, and must
#: never silently become a ranking of a movement, nor the reverse.
_LEVEL_ONLY_RE = re.compile(
    r"\b(?:largest|biggest|highest|lowest|smallest|top|greatest)\b", re.I)


class CensusError(RuntimeError):
    """The census could not be measured. Never absorbed into an empty table."""


def _intended(question: str, has_rank: bool, has_dimension: bool) -> str:
    """What the question asks for, derived from the question alone."""
    if not has_rank:
        return "not_ranking"
    change = bool(_CHANGE_RE.search(question))
    if change:
        return "ranked_movement" if has_dimension else "ranked_movement_no_dimension"
    if _LEVEL_ONLY_RE.search(question):
        return "ranked_level" if has_dimension else "ranked_level_no_dimension"
    return "ranking_underspecified"


def run(depth: int = 6) -> Dict[str, Any]:
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

    tmp = Path(tempfile.mkdtemp())
    out_root = tmp / "onboarding_output"
    from migration_phase0.compound_canary import _write_run
    runs = funded_runs(depth)
    for run_id, rdate, n, scale in runs:
        _write_run(out_root, run_id, rdate, n, scale)
    portfolio, as_of = f"client_001/{runs[-1][0]}", runs[-1][1]

    saved = {k: os.environ.get(k) for k in
             ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_AUTH_ENABLED")}
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(out_root)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    try:
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        from mi_agent import execution_receipt as R
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent.parsed_question import ParsedQuestion
        from mi_agent.period_change import rank_request as rank_mod, recognise
        from mi_agent_api.data_source import semantics_path
        from mi_agent_api.period_change_route import resolve_rank_intent
        from question_interpretation import projection as proj

        semantics = load_mi_semantics(semantics_path())
        if not (semantics.get("fields") or {}):
            raise CensusError("CENSUS INVALID — governed MI semantics did not load")

        client = TestClient(app)
        rows: List[Dict[str, Any]] = []
        for idx, question in enumerate(_questions()):
            if not rank_mod.has_rank_language(question):
                continue
            spec = ParsedQuestion.parse(question, semantics).spec
            terms = R.requested_dimension_terms(question, semantics, None)
            facets = R.detect_requested_facets(
                question, semantics, frame=None, requested_dimensions=terms)
            qi = proj.from_parts(question, spec=spec, facets=facets,
                                 dim_terms=terms, semantics=semantics)
            intent = resolve_rank_intent(question, columns=None)
            recog = recognise(question, spec=spec, view="funded",
                              semantics_context=None)
            resp = client.post("/mi/query", json={
                "question": question, "portfolioId": portfolio,
                "asOfDate": as_of}).json()
            meta = resp.get("metadata") or {}
            dims = [(d.candidate_concept, d.role) for d in (qi.dimensions or [])]
            rows.append({
                "id": f"R{idx:03d}", "question": question,
                "intended": _intended(question, True, bool(dims)),
                # contract
                "contract_operation": qi.operation.type,
                "contract_modifiers": list(qi.operation.modifiers or ()),
                "contract_dimensions": dims,
                "contract_measure": qi.subject.candidate_concept,
                "contract_comparison_periods": list(
                    getattr(qi.time, "comparison_periods", ()) or ()),
                "contract_span": getattr(
                    getattr(qi.time, "trend_window", None), "raw_text", None),
                # legacy route
                "route_rank_requested": bool(intent.requested),
                "route_rank_field": intent.field,
                "route_rank_direction": getattr(intent.request, "direction", None),
                "route_rank_basis": getattr(intent.request, "basis", None),
                "route_rank_limit": getattr(intent.request, "top_n", None),
                "route_recognised": bool(recog.matched),
                "route_recognition_reason": getattr(recog, "reason", None),
                # execution
                "route": meta.get("route"),
                "owned": meta.get("route") in PERIOD_CHANGE_ROUTES,
                "grade": _grade(resp),
                "ranked_applied": bool((meta.get("rankedMovement") or {}).get("applied")),
                "answer": str(resp.get("answer") or "")[:150],
            })
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    if not rows:
        raise CensusError("CENSUS INVALID — no ranking-language questions found")
    return {"rows": rows}


def classify_disagreements(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The contract/route disagreements, in defensible families."""
    out = []
    for r in rows:
        contract_says = r["contract_operation"] == "ranking"
        route_says = r["route_rank_requested"]
        if contract_says == route_says:
            continue
        if contract_says and not route_says:
            family = ("route_narrative_guard"
                      if _LEVEL_ONLY_RE.search(r["question"]) is None
                      else "route_no_rank_request")
        else:
            family = "contract_not_ranking"
        out.append({**r, "family": family,
                    "contract_says_ranking": contract_says,
                    "route_says_ranking": route_says})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--depth", type=int, default=6)
    args = ap.parse_args(argv)

    result = run(args.depth)
    rows = result["rows"]
    dis = classify_disagreements(rows)

    print("=" * 100)
    print(f"RANKING RECOGNITION CENSUS — {len(rows)} corpus questions carrying "
          f"ranking language")
    print("=" * 100)

    print("\nINTENDED CLASSIFICATION (derived from the question, not hand-labelled)")
    for k, v in Counter(r["intended"] for r in rows).most_common():
        print(f"   {k:<32} {v}")

    print("\nCONTRACT vs LEGACY ROUTE")
    print(f"   contract says ranking      : "
          f"{sum(1 for r in rows if r['contract_operation'] == 'ranking')}")
    print(f"   route says ranking         : "
          f"{sum(1 for r in rows if r['route_rank_requested'])}")
    print(f"   route RECOGNISED as period : "
          f"{sum(1 for r in rows if r['route_recognised'])}")
    print(f"   disagreements              : {len(dis)}")

    print("\nWHAT THE CONTRACT CARRIES FOR THESE QUESTIONS")
    print(f"   operation.modifiers non-empty      : "
          f"{sum(1 for r in rows if r['contract_modifiers'])}")
    print(f"   comparison_periods non-empty       : "
          f"{sum(1 for r in rows if r['contract_comparison_periods'])}")
    print(f"   a span named                       : "
          f"{sum(1 for r in rows if r['contract_span'])}")
    print(f"   at least one dimension             : "
          f"{sum(1 for r in rows if r['contract_dimensions'])}")

    print("\nEXECUTION")
    for k, v in Counter(str(r["route"]) for r in rows).most_common():
        print(f"   {k:<32} {v}")
    print(f"   DELIVERED                       : "
          f"{sum(1 for r in rows if r['grade'] == 'DELIVERED')}")
    print(f"   ranking actually applied        : "
          f"{sum(1 for r in rows if r['ranked_applied'])}")

    print("\nDISAGREEMENT FAMILIES")
    for k, v in Counter(d["family"] for d in dis).most_common():
        print(f"   {k:<32} {v}")
    for d in dis:
        print(f"   [{d['family']:<24}] contract={d['contract_operation']!r:<10} "
              f"route_rank={d['route_says_ranking']}  {d['question'][:58]}")

    print("\nMOVEMENT ASKED, LEVEL DELIVERED (the D2 class, over the whole 97)")
    n = 0
    for r in rows:
        if r["intended"].startswith("ranked_movement") and r["grade"] == "DELIVERED" \
                and not r["ranked_applied"] and "as at" in r["answer"].lower():
            n += 1
            print(f"   {r['question'][:62]}")
            print(f"       -> {r['answer'][:96]}")
    print(f"   total: {n}")

    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "disagreements": dis},
                                        indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
