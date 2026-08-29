#!/usr/bin/env python3
"""migration_phase0/contract_sufficiency_portfolio_summary.py — can the plan be built?

READ-ONLY. Phase 1F §5 and §12.

The conversion gate requires, for every route-owned case:

    route-owned cases construct from contract   100%
    external lens injection                       0
    downstream raw-question semantic reads        0

So the question this answers is not "do the economics reconcile" — Phase 0
settled that, 9 cases, 0 differences — but "can `select_population` be planned
from the interpretation contract ALONE, and does the resulting population match
what production selects".

It compares, per (question, caller default):

    production   `chat_routing._resolve_lens(question, source_lens)`
                 — the shipped precedence decision, question vs dropdown
    contract     `build_plan(interpretation).select_population`
                 — everything a compositional plan is allowed to see

`source_portfolio_lens` is a live field on `MiQueryRequest` populated from the
workspace dropdown (`app.py`), so a surface measured only at ``None`` is
measured on one third of its inputs. Phase 1B stopped on precisely the third
that is otherwise unmeasured.

    python -m migration_phase0.contract_sufficiency_portfolio_summary [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

warnings.simplefilter("ignore")
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from migration_phase0.route_ownership_portfolio_summary import (  # noqa: E402
    CANDIDATES, DEFAULTS, _env,
)


def _plan_population(interpretation) -> Dict[str, Any]:  # noqa: D401
    """What a contract-only plan would select. No question, no caller default.

    `build_plan` takes the interpretation and nothing else — the structural
    guarantee `assert_no_question_read` exists to protect — so this is the
    complete set of inputs a compositional plan is permitted.
    """
    from migration_phase0 import shadow_portfolio_summary as shadow

    plan = shadow.build_plan(interpretation, region_column="collateral_geography",
                             has_portfolio_column=True)
    step = next((s for s in plan.steps
                 if s.primitive == shadow.SELECT_POPULATION
                 and s.inputs.get("kind") == "source_portfolio_lens"), None)
    if step is None:
        return {"planned": None, "blocked": "no select_population step"}
    if step.blocked:
        return {"planned": None, "blocked": step.blocked}
    return {"planned": step.inputs.get("scope"),
            "portfolioIds": list(step.inputs.get("portfolio_ids") or []),
            "blocked": None}


def capture(client_id: str) -> Dict[str, Any]:
    from mi_agent_api import chat_routing as routing
    from mi_agent_api import portfolio_context as ctx_mod
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.datasets import semantics_path
    from question_interpretation import projection

    semantics = load_mi_semantics(semantics_path())
    registry = ctx_mod.build_registry()

    rows: List[Dict[str, Any]] = []
    for case, question, provenance in CANDIDATES:
        if not routing._is_portfolio_summary(question):
            continue
        for default in DEFAULTS:
            # PRODUCTION: the shipped precedence decision.
            shipped = routing._resolve_lens(question, default)
            # CONTRACT: what a plan can build. PHASE 1G — the caller context is
            # now an input to the INTERPRETATION, not to the plan: the owner
            # applies precedence once and the claim records both the outcome and
            # its provenance, so `build_plan` still sees only the contract and
            # still cannot reach the question.
            qi = projection.project(question, semantics=semantics,
                                   registry=registry, caller_scope=default)
            contract = _plan_population(qi)
            rows.append({
                "case": case, "question": question, "default": default,
                "shippedScope": shipped.name,
                "shippedFilters": dict(shipped.filters or {}),
                "contractScope": contract.get("planned"),
                "contractIds": contract.get("portfolioIds") or [],
                "contractBlocked": contract.get("blocked"),
                "claim": {"state": qi.source_scope.state,
                          "scope": qi.source_scope.scope,
                          "ids": list(qi.source_scope.portfolio_ids),
                          "provenance": qi.source_scope.provenance,
                          "base": qi.source_scope.base_population,
                          "narrows": qi.source_scope.narrows},
            })
    return {"rows": rows}


def classify(row: Dict[str, Any]) -> str:
    """How a contract-only plan would differ from the shipped route."""
    if row["contractBlocked"]:
        # A blocked plan is a REFUSAL, not an answer with the step omitted, and
        # production refuses these too. Named apart from a match so the record
        # shows which cases construct and which deliberately decline.
        return "BLOCKED"
    shipped, planned = row["shippedScope"], row["contractScope"]
    if shipped == planned:
        return "match"
    # The two failure directions are named apart, because they are different
    # defects: one answers for MORE of the book than the shipped route, one for
    # less. Both are silent; only the first is a widening.
    order = {"total": 3, "direct": 1, "acquired": 1, "cohort": 0}
    if order.get(planned, 0) > order.get(shipped, 0):
        return "WIDENS"
    return "NARROWS"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(
        _REPO / "migration_phase0" / "CONTRACT_SUFFICIENCY_PORTFOLIO_SUMMARY.json"))
    args = ap.parse_args(argv)

    data = capture(_env())
    rows = data["rows"]
    for row in rows:
        row["verdict"] = classify(row)

    print("=" * 118)
    print("portfolio_summary — CAN select_population BE PLANNED FROM THE CONTRACT?")
    print("=" * 118)
    print(f"\n{'case':5s} {'default':9s} {'contract claim':22s} {'plan would':12s} "
          f"{'shipped':12s} verdict")
    print("-" * 118)
    for row in rows:
        claim = f"{row['claim']['provenance']}/{row['claim']['scope']}"
        print(f"{row['case']:5s} {str(row['default']):9s} {claim[:22]:22s} "
              f"{str(row['contractScope'])[:12]:12s} {row['shippedScope'][:12]:12s} "
              f"{row['verdict']}")
    print("-" * 118)

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    total = len(rows)
    ok = counts.get("match", 0)
    print(f"{ok} of {total} (question, caller default) combinations would be "
          f"planned correctly from the contract.")
    for verdict in ("WIDENS", "NARROWS", "BLOCKED"):
        if counts.get(verdict):
            print(f"  {verdict}: {counts[verdict]}")

    # The pairs that prove it: identical claim, opposite required behaviour.
    by_claim: Dict[str, List[Tuple[str, Optional[str], str]]] = {}
    for row in rows:
        key = (f"{row['claim']['state']}/{row['claim']['provenance']}/"
               f"{row['claim']['scope']}/{row['claim']['ids']}")
        by_claim.setdefault(key, []).append(
            (row["case"], row["default"], row["shippedScope"]))
    print("\nIDENTICAL CONTRACT CLAIM, DIFFERENT SHIPPED POPULATION:")
    found = False
    for key, entries in sorted(by_claim.items()):
        scopes = {e[2] for e in entries}
        if len(scopes) > 1:
            found = True
            print(f"  claim {key} -> shipped {sorted(scopes)}")
            for case, default, scope in entries:
                if len(scopes) > 1:
                    print(f"      {case:4s} default={str(default):9s} -> {scope}")
    if not found:
        print("  none — the claim determines the population")

    Path(args.out).write_text(json.dumps(data, indent=2, default=str) + "\n",
                              encoding="utf-8")
    print(f"\nwrote {Path(args.out).relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
