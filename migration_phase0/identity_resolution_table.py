#!/usr/bin/env python3
"""migration_phase0/identity_resolution_table.py — current vs target resolution.

READ-ONLY. Runs the same cases against the live governed book AND against a
deterministic multi-portfolio fixture (2 acquired + 1 direct), because the live
book has one portfolio per category and cannot show a category collapsing.

TARGET SEMANTICS (Phase 1E, business-confirmed):

    Funded Book  = the COMPLETE funded population — every governed source
                   category. NOT a synonym for Direct.
    Direct Book  = every governed funded portfolio classified `direct`
    Acquired Book= every governed funded portfolio classified `acquired`
    named book   = exactly that one governed portfolio
    unknown book = must NOT widen to Funded/Total

    python -m migration_phase0.identity_resolution_table
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

#: A registry the live book cannot provide: two acquired portfolios, so
#: "Acquired Book" collapsing to one of them becomes observable.
FIXTURE_RECORDS: Tuple[Dict[str, str], ...] = (
    {"source_portfolio_id": "alp_origination", "source_portfolio_type": "direct",
     "source_portfolio_label": "ALP Origination Book"},
    {"source_portfolio_id": "alp_acquired", "source_portfolio_type": "acquired",
     "source_portfolio_label": "ALP Acquired Back Book"},
    {"source_portfolio_id": "nbs_acquired", "source_portfolio_type": "acquired",
     "source_portfolio_label": "NBS Acquired Book"},
)

#: (case, question, expected governed ids | None = "must not be Total")
CASES: Tuple[Tuple[str, str, Optional[Tuple[str, ...]]], ...] = (
    ("funded category",   "Summarise the funded book",
     ("alp_origination", "alp_acquired", "nbs_acquired")),
    ("direct category",   "Summarise the direct book", ("alp_origination",)),
    ("acquired category", "Summarise the acquired book",
     ("alp_acquired", "nbs_acquired")),
    ("named direct",      "Summarise the ALP Origination Book", ("alp_origination",)),
    ("named acquired A",  "Summarise the ALP Acquired Back Book", ("alp_acquired",)),
    ("named acquired B",  "Summarise the NBS Acquired Book", ("nbs_acquired",)),
    ("governed id",       "Summarise the nbs_acquired book", ("nbs_acquired",)),
    ("unknown label",     "Summarise the Highgate Mortgages Book", None),
    ("storage id",        "Summarise the acquired_001 book", None),
    ("no scope named",    "Please provide a portfolio summary",
     ("alp_origination", "alp_acquired", "nbs_acquired")),
)


def _env() -> str:
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


def resolve_against(registry, question: str) -> Dict[str, Any]:
    """What MI resolves, scored against the governed registry."""
    from mi_agent import portfolio_lens as lens_mod
    from trakt_core import portfolio as portfolio_mod

    try:
        lens = lens_mod.resolve_lens(question, registry=registry)
    except TypeError:                      # pre-1E signature
        lens = lens_mod.resolve_lens(question)
    scope = portfolio_mod.resolve_scope(registry, lens_mod.context_id(lens))
    return {"lensName": lens.name,
            "contextId": lens_mod.context_id(lens),
            "ids": tuple(scope.portfolio_ids),
            "fellBackToTotal": bool(scope.fell_back_to_total)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    _env()
    from trakt_core import portfolio as portfolio_mod

    registry = portfolio_mod.build_registry(FIXTURE_RECORDS, client_id="phase1e")
    total_ids = tuple(registry.ids())

    print("=" * 112)
    print("IDENTITY RESOLUTION — current vs target (2 acquired + 1 direct)")
    print("=" * 112)
    print(f"\nregistry: {list(total_ids)}")
    print(f"  direct   -> {[p.portfolio_id for p in registry.of_type('direct')]}")
    print(f"  acquired -> {[p.portfolio_id for p in registry.of_type('acquired')]}\n")

    header = (f"{'case':18s} {'question':40s} {'MI resolves to':34s} "
              f"{'target':26s} ok")
    print(header)
    print("-" * 112)

    rows: List[Dict[str, Any]] = []
    failures = 0
    for case, question, expected in CASES:
        got = resolve_against(registry, question)
        if expected is None:
            # "must not be answered for the whole book". Scored on the LENS, not
            # on `resolve_scope`, and the distinction is the finding: an
            # UNRESOLVED lens still makes `resolve_scope` fall back to every id
            # with `fell_back_to_total=True`. The contract DISCLOSES the
            # widening; it does not prevent it. What prevents it is the facet
            # layer downstream, proved end-to-end against the live book in
            # docs/mi_phase1e_report.md (both cases refuse). Scoring the raw
            # scope here would have recorded a pass that the governed contract
            # alone does not earn — so the raw scope is printed alongside.
            ok = got["lensName"] == "unresolved"
            target = "lens UNRESOLVED"
        else:
            ok = tuple(got["ids"]) == tuple(expected)
            target = ",".join(expected)
        failures += 0 if ok else 1
        rows.append({"case": case, "question": question, "target": target,
                     "ok": ok, **{k: (list(v) if isinstance(v, tuple) else v)
                                  for k, v in got.items()}})
        shown = got["lensName"] + " " + str(list(got["ids"]))
        if got["fellBackToTotal"]:
            shown = got["lensName"] + " -> fell back to ALL (disclosed)"
        print(f"{case:18s} {question[:40]:40s} {shown[:34]:34s} "
              f"{target[:26]:26s} {'ok' if ok else 'FAIL'}")

    print("-" * 112)
    print(f"{len(CASES) - failures} of {len(CASES)} cases meet the target semantics; "
          f"{failures} do not.")

    out = _REPO / "migration_phase0" / "IDENTITY_RESOLUTION_TABLE.json"
    out.write_text(json.dumps({"registry": list(total_ids), "rows": rows},
                              indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
