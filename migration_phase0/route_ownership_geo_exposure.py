#!/usr/bin/env python3
"""migration_phase0/route_ownership_geo_exposure.py — what `geo_exposure` owns.

READ-ONLY. Enumerates candidate questions x caller scopes and records, from
EXECUTED routing rather than from wording similarity, which the shipped
`geo_exposure` recogniser actually claims.

Cases prefixed ``X`` are deliberately listed and expected NOT to be claimed.
They are the discipline: a surface that quietly grows to include questions the
route does not own manufactures an equivalence that means nothing, and Phase 1F
measured exactly that mistake. Each X case names the route that should take it.

    python -m migration_phase0.route_ownership_geo_exposure
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
#: geo_exposure"; a string names the route that should claim it instead.
CASES: Tuple[Tuple[str, str, Any], ...] = (
    # --- the concentration phrasings -------------------------------------
    ("G1", "What is the geographic exposure?", None),
    ("G2", "Where is the book concentrated geographically?", None),
    ("G3", "Which region has the largest exposure?", None),
    ("G4", "Funded exposure by ITL3 area", None),
    ("G5", "Where are we most exposed geographically?", None),
    ("G6", "What is the largest geographic concentration?", None),
    ("G7", "Which area has the biggest exposure?", None),
    ("G8", "geographic exposure", None),
    # --- the same, narrowed by a named scope ------------------------------
    ("S1", "Geographic exposure for the acquired book", None),
    ("S2", "Geographic exposure for the direct book", None),
    ("S3", "Where is the acquired book most exposed geographically?", None),
    # --- refusal surface ---------------------------------------------------
    ("R1", "Geographic exposure for the Highgate Mortgages Book", None),
    # --- NOT claimed, and named ------------------------------------------
    ("X1", "Show the balance by region", "an ordinary stratification, not the "
                                        "ITL3 concentration engine"),
    ("X2", "Which region grew the most over the last three months?",
     "period change — a comparison this route would silently drop"),
    ("X3", "Show top 5 regions by balance", "a grouped ranking"),
    ("X4", "Funded balance bridge by region", "the bridge route"),
    ("X5", "Which regions breached the concentration limit?", "risk limits"),
)

SCOPES: Tuple[Any, ...] = (None, "direct", "acquired")


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
    print("geo_exposure — ROUTE OWNERSHIP, from executed routing")
    print("=" * 78)

    records: List[Dict[str, Any]] = []
    claimed = misclaimed = 0
    for case, question, expected_other in CASES:
        by_scope: Dict[str, Any] = {}
        for scope in SCOPES:
            env = execute_governed_mi_query(
                MiQueryRequest(question=question,
                               source_portfolio_lens=scope), ctx).result or {}
            md = env.get("metadata") or {}
            by_scope[str(scope)] = {
                "route": md.get("route"),
                "ok": env.get("ok"),
                "controlledRefusal": env.get("controlledRefusal"),
                "lensApplied": md.get("lensApplied"),
                "answer": (env.get("answer") or "")[:220],
                "reconciliation": env.get("reconciliation"),
                "artifactKinds": [a.get("type") for a in (env.get("artifacts") or [])],
            }
        routes = {v["route"] for v in by_scope.values()}
        owns = routes == {"geo_exposure"}
        want_owned = expected_other is None
        agree = owns == want_owned
        if owns:
            claimed += 1
        if not agree:
            misclaimed += 1
        flag = "   " if agree else "!! "
        print(f"\n{flag}{case}  {question!r}")
        print(f"      routes across scopes : {sorted(r or '-' for r in routes)}")
        if want_owned:
            print(f"      expected             : geo_exposure")
        else:
            print(f"      expected NOT claimed : {expected_other}")
        for s, v in by_scope.items():
            print(f"      [{s:<8}] ok={str(v['ok']):<5} {str(v['answer'])[:120]}")
        records.append({"case": case, "question": question,
                        "expectedOther": expected_other,
                        "ownedByGeo": owns, "asExpected": agree,
                        "byScope": by_scope})

    out = _REPO / "migration_phase0" / "ROUTE_OWNERSHIP_GEO_EXPOSURE.json"
    out.write_text(json.dumps({"cases": records}, indent=2, default=str))

    print("\n" + "=" * 78)
    print(f"cases enumerated          : {len(CASES)}  x {len(SCOPES)} scopes "
          f"= {len(CASES) * len(SCOPES)} renders")
    print(f"claimed by geo_exposure   : {claimed}")
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
