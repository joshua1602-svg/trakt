#!/usr/bin/env python3
"""migration_phase0/route_ownership_portfolio_summary.py — what does the route own?

READ-ONLY. Phase 1F §3.

Phase 0 established that route-surface ownership must be VERIFIED rather than
inferred, and the reason is recorded there: two questions that read like
portfolio summaries ("Summarise the front book", "What is the portfolio position
for the direct book?") are NOT claimed by this recogniser, and comparing on a
question the route does not own manufactures an equivalence that means nothing.

This asks the shipped recogniser itself — `chat_routing._is_portfolio_summary`
— over a candidate bank drawn from every phase that touched this surface, and
then executes the ones it claims so the record carries what the route actually
DOES, not only what it claims.

The candidate bank deliberately includes questions expected NOT to be owned.
A bank containing only owned questions cannot detect ownership drift.

    python -m migration_phase0.route_ownership_portfolio_summary [--out FILE]
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

#: (case, question, provenance). Provenance names the phase that put the case on
#: this surface, so a case cannot be quietly dropped without the record showing
#: which finding it came from.
CANDIDATES: Tuple[Tuple[str, str, str], ...] = (
    # Phase 0 / 1B frozen surface — no source scope named.
    ("A1", "Please provide a portfolio summary", "phase0"),
    ("A2", "Give me a summary of the portfolio", "phase0"),
    ("A3", "Can you summarise the book for me?", "phase0"),
    ("A4", "portfolio summary", "phase0"),
    ("A5", "summarise the portfolio", "phase0"),
    ("A6", "overview of the portfolio", "phase0"),
    # Phase 0 / 1B frozen surface — a source scope named.
    ("L1", "Summarise the acquired book", "phase0"),
    ("L2", "Summarise the direct book", "phase0"),
    ("L3", "portfolio summary for the acquired book", "phase0"),
    # Phase 1C precedence phrasings: these MENTION a portfolio and resolve to
    # total, which is the Phase 1B blocker's whole shape.
    ("P1", "portfolio summary across all portfolios", "phase1c"),
    ("P2", "summarise the portfolio excluding the acquired book", "phase1c"),
    # Phase 1E identity phrasings.
    ("N1", "Summarise the ALP Origination Book", "phase1e"),
    ("N2", "Summarise the alp_acquired book", "phase1e"),
    ("N3", "Summarise the ALP Acquired Back Book", "phase1e"),
    ("Z1", "Summarise the spv1_sponsored portfolio", "phase1e"),
    ("U1", "Summarise the acquired_001 book", "phase1e"),
    ("U2", "Summarise the Highgate Mortgages Book", "phase1e"),
    ("F1", "Summarise the funded book", "phase1e"),
    # Expected NOT owned — kept so ownership drift is detectable.
    ("X1", "Summarise the front book", "phase0-excluded"),
    ("X2", "What is the portfolio position for the direct book?", "phase0-excluded"),
    ("X3", "Summarise the portfolio by region", "stratification"),
    ("X4", "What changed in the portfolio since last month?", "movement"),
    ("X5", "For the London book, give me balance, number of loans, "
           "weighted-average LTV and average borrower age.", "phase1e-control"),
)

#: Caller-supplied defaults to exercise. `source_portfolio_lens` is a live field
#: on `MiQueryRequest`, populated from the workspace dropdown, so a surface
#: measured only at `None` is measured on one third of its inputs.
DEFAULTS: Tuple[Optional[str], ...] = (None, "acquired", "direct")


def _env() -> str:
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


def _interpretation(question: str, semantics, registry) -> Dict[str, Any]:
    from question_interpretation import projection
    qi = projection.project(question, semantics=semantics, registry=registry)
    claim = qi.source_scope
    return {"state": claim.state, "scope": claim.scope,
            "portfolioIds": list(claim.portfolio_ids),
            "portfolioLabel": claim.portfolio_label,
            "rawText": claim.raw_text, "narrows": claim.narrows,
            "reason": claim.reason}


def capture(client_id: str) -> Dict[str, Any]:
    from mi_agent_api import chat_routing as routing
    from mi_agent_api import portfolio_context as ctx_mod
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.datasets import semantics_path
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    semantics = load_mi_semantics(semantics_path())
    registry = ctx_mod.build_registry()
    ctx = ExecutionContext.for_internal(client_id)

    rows: List[Dict[str, Any]] = []
    for case, question, provenance in CANDIDATES:
        # THE RECOGNISER ITSELF decides ownership. Not a list maintained here.
        owned = bool(routing._is_portfolio_summary(question))
        row: Dict[str, Any] = {
            "case": case, "question": question, "provenance": provenance,
            "recogniserClaims": owned,
            "interpretation": _interpretation(question, semantics, registry),
            "byDefault": {},
        }
        for default in DEFAULTS:
            result = execute_governed_mi_query(
                MiQueryRequest(question=question,
                               source_portfolio_lens=default), ctx).result or {}
            metadata = result.get("metadata") or {}
            summary = (result.get("executionSummary")
                       or metadata.get("executionSummary") or {})
            row["byDefault"][str(default)] = {
                "route": metadata.get("route"),
                "ok": bool(result.get("ok")),
                "controlledRefusal": bool(result.get("controlledRefusal")),
                "lensApplied": metadata.get("lensApplied"),
                "portfolioScope": result.get("portfolioScope"),
                "verdict": (result.get("semanticGuard") or {}).get("verdict"),
                "facets": [(f.get("kind"), f.get("label"), f.get("field_key"),
                            f.get("status"))
                           for f in (summary.get("facets") or [])],
                "answer": (result.get("answer") or "").strip().replace("\n", " "),
            }
        rows.append(row)
    return {"cases": rows}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(
        _REPO / "migration_phase0" / "ROUTE_OWNERSHIP_PORTFOLIO_SUMMARY.json"))
    args = ap.parse_args(argv)

    data = capture(_env())
    rows = data["cases"]

    print("=" * 118)
    print("portfolio_summary — ROUTE OWNERSHIP, verified against the recogniser")
    print("=" * 118)
    print(f"\n{'case':5s} {'claims':7s} {'route(None)':19s} {'scope claim':26s} "
          f"{'ok/refuse by default':24s} question")
    print("-" * 118)
    owned_n = 0
    for row in rows:
        by = row["byDefault"]
        interp = row["interpretation"]
        claim = f"{interp['state']}/{interp['scope']}"
        if interp["portfolioIds"]:
            claim += "=" + ",".join(interp["portfolioIds"])
        states = " ".join(
            ("REF" if by[str(d)]["controlledRefusal"]
             else ("ok" if by[str(d)]["ok"] else "no"))
            for d in DEFAULTS)
        owned_n += 1 if row["recogniserClaims"] else 0
        print(f"{row['case']:5s} {str(row['recogniserClaims']):7s} "
              f"{str(by['None']['route']):19s} {claim[:26]:26s} "
              f"{states:24s} {row['question'][:40]}")
    print("-" * 118)
    print(f"{owned_n} of {len(rows)} candidates are claimed by "
          f"`_is_portfolio_summary`.")

    Path(args.out).write_text(json.dumps(data, indent=2, default=str) + "\n",
                              encoding="utf-8")
    print(f"\nwrote {Path(args.out).relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
