#!/usr/bin/env python3
"""migration_phase0/source_scope_truth_table.py — the production source-scope semantics.

READ-ONLY. Executes the real resolution chain for every combination of
(question wording x caller/default scope) and prints what production actually
does. Nothing here is inferred from names.

THE CHAIN, as traced in the source:

    1. DETECT      portfolio_lens.resolve_lens(question)          NL recognition
    2. DEFAULT     lens_from_selection(req.source_portfolio_lens) the dropdown
    3. PRECEDENCE  resolve_lens_with_default(question, default):
                       if mentions_portfolio(question): the QUESTION wins
                       else:                            the DEFAULT wins
    4. GOVERN      portfolio_lens.context_id(lens)
                   -> portfolio_context.resolve_context(...).scope
                                                             semantic -> ids
    5. APPLY       scope.filters -> {source_portfolio_id: [...]}
                   -> evolution._scope_frame_lens(df, filters)

Step 4 is the one the Phase 1A shadow harness skipped: it used the LENS's own
`filters` ({'source_portfolio_type': 'acquired'}) where production uses the
resolved SCOPE's ({'source_portfolio_id': ['alp_acquired']}).

    python -m migration_phase0.source_scope_truth_table
"""
from __future__ import annotations

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

#: (label, question). Chosen to cover every branch of step 3, including the two
#: phrasings Phase 1B found that MENTION a portfolio and resolve to total.
QUESTIONS: Tuple[Tuple[str, str], ...] = (
    ("silent",            "Please provide a portfolio summary"),
    ("silent",            "summarise the portfolio"),
    ("explicit direct",   "Summarise the direct book"),
    ("explicit acquired", "Summarise the acquired book"),
    ("explicit cohort",   "Summarise the acquired_001 book"),
    ("explicit total",    "portfolio summary across all portfolios"),
    ("disclaimed",        "summarise the portfolio excluding the acquired book"),
)

#: Caller/default scope, as `MiQueryRequest.source_portfolio_lens` carries it.
DEFAULTS: Tuple[Optional[Any], ...] = (None, "acquired", "direct", "alp_acquired")


def _env() -> str:
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


def resolve(question: str, default_selection: Optional[Any]) -> Dict[str, Any]:
    """One row of the truth table, through the production chain."""
    from mi_agent import portfolio_lens as lens_mod
    from mi_agent_api import portfolio_context as ctx_mod

    mentions = lens_mod.mentions_portfolio(question)
    default_lens = (lens_mod.lens_from_selection(default_selection)
                    if default_selection is not None else None)
    effective = lens_mod.resolve_lens_with_default(question, default_lens)
    governed = ctx_mod.resolve_context(lens_mod.context_id(effective),
                                       discover_pipeline=False).scope
    return {
        "questionMentionsPortfolio": mentions,
        "questionScope": lens_mod.resolve_lens(question).name,
        "defaultScope": getattr(default_lens, "name", None),
        "effectiveScope": effective.name,
        "wonBy": "question" if mentions else ("default" if default_lens else "neither"),
        "contextId": lens_mod.context_id(effective),
        "governedPortfolioIds": list(governed.portfolio_ids),
        "governedFilters": governed.filters,
        "rawLensFilters": effective.filters,
        "contextKind": governed.context_kind,
        "fellBackToTotal": governed.fell_back_to_total,
    }


def population(client_id: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    """The rows those filters actually select, on the governed book."""
    from mi_agent_api import evolution as evolution_mod

    root = os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"]
    frames = evolution_mod.funded_frames(root, client_id, None)
    df = evolution_mod._scope_frame_lens(frames[-1]["df"], filters or None)
    balance = "current_outstanding_balance"
    return {"rows": int(len(df)),
            "balance": round(float(df[balance].sum()), 2) if balance in df.columns else None}


def main(argv: Optional[Sequence[str]] = None) -> int:
    client_id = _env()
    print("=" * 118)
    print("PRODUCTION SOURCE-SCOPE TRUTH TABLE — executed, not inferred")
    print("=" * 118)
    header = (f"{'question':22s} {'default':12s} {'mentions':9s} {'qScope':9s} "
              f"{'effective':10s} {'wonBy':9s} {'governed portfolio ids':32s} "
              f"{'rows':>7s} {'balance':>18s}")
    print("\n" + header)
    print("-" * 118)

    rows: List[Dict[str, Any]] = []
    divergences: List[str] = []
    for label, question in QUESTIONS:
        for default in DEFAULTS:
            r = resolve(question, default)
            pop = population(client_id, r["governedFilters"])
            r.update({"label": label, "question": question,
                      "defaultSelection": default, **pop})
            rows.append(r)
            print(f"{label:22s} {str(default):12s} {str(r['questionMentionsPortfolio']):9s} "
                  f"{r['questionScope']:9s} {r['effectiveScope']:10s} {r['wonBy']:9s} "
                  f"{str(r['governedPortfolioIds']):32s} {pop['rows']:7d} "
                  f"{pop['balance']:18,.2f}")
            # The Phase 1B blocker, detected rather than asserted: the question
            # SPEAKS to source scope, resolves to total, and therefore beats a
            # default that would otherwise have narrowed.
            if (r["questionMentionsPortfolio"] and r["questionScope"] == "total"
                    and default is not None):
                divergences.append(
                    f"{question!r} + default={default!r}: question wins with "
                    f"total; a contract carrying only the resolved scope would "
                    f"fall back to {default!r}")
        print("-" * 118)

    print("\nBRANCHES OF THE PRECEDENCE RULE, as executed:")
    for won in ("question", "default", "neither"):
        n = sum(1 for r in rows if r["wonBy"] == won)
        print(f"   won by {won:9s} {n:3d} row(s)")

    print("\nRAW LENS FILTERS vs GOVERNED SCOPE FILTERS "
          "(what Phase 1A compared vs what production applies):")
    shown = set()
    for r in rows:
        key = (r["effectiveScope"], json.dumps(r["rawLensFilters"], sort_keys=True))
        if key in shown:
            continue
        shown.add(key)
        same = r["rawLensFilters"] == r["governedFilters"]
        print(f"   {r['effectiveScope']:10s} raw={str(r['rawLensFilters']):46s} "
              f"governed={str(r['governedFilters']):44s} same={same}")

    print(f"\nPRECEDENCE DIVERGENCES that a resolved-scope-only contract cannot "
          f"reproduce: {len(divergences)}")
    for d in divergences:
        print(f"   {d}")

    out = _REPO / "migration_phase0" / "SOURCE_SCOPE_TRUTH_TABLE.json"
    out.write_text(json.dumps({"rows": rows, "divergences": divergences},
                              indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
