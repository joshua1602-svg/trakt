#!/usr/bin/env python3
"""migration_phase0/scope_precedence_matrix.py — question scope × caller context.

READ-ONLY. Phase 1G §4 and §14.

Phase 1F measured 18 owned questions against 3 caller defaults and found 14
silent widenings. This widens the grid in both directions — every question-scope
KIND against every caller-context KIND, including a named governed portfolio on
BOTH sides, which Phase 1F did not exercise at all.

It records, per cell:

    what the QUESTION asked for        the owner's reading of the text alone
    what the CALLER supplied           the workspace selection
    which one WON                      production's precedence decision
    the governed portfolio IDs         what actually gets selected
    the TARGET under §5                the intended product rule

and scores the cell against the target, so a divergence is a named cell rather
than a count.

`--registry fixture` runs against a deterministic 5-portfolio registry (1 direct
+ 2 acquired + 2 SPVs), because the live book holds one portfolio per category
and cannot show a category collapsing onto a portfolio, nor an SPV at all.

    python -m migration_phase0.scope_precedence_matrix [--registry live|fixture]
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

#: 1 direct + 2 acquired + 2 SPVs. The SPVs are typed so that "the acquired
#: book" and "SPV1" are DIFFERENT answers — a category that collapsed onto one
#: portfolio, or a portfolio read as its whole category, is then numerically
#: observable rather than a matter of opinion.
FIXTURE_RECORDS: Tuple[Dict[str, str], ...] = (
    {"source_portfolio_id": "alp_origination", "source_portfolio_type": "direct",
     "source_portfolio_label": "ALP Origination Book"},
    {"source_portfolio_id": "alp_acquired", "source_portfolio_type": "acquired",
     "source_portfolio_label": "ALP Acquired Back Book"},
    {"source_portfolio_id": "nbs_acquired", "source_portfolio_type": "acquired",
     "source_portfolio_label": "NBS Acquired Book"},
    {"source_portfolio_id": "spv1", "source_portfolio_type": "acquired",
     "source_portfolio_label": "SPV1"},
    {"source_portfolio_id": "spv2", "source_portfolio_type": "direct",
     "source_portfolio_label": "SPV2"},
)

#: (id, question, what the QUESTION means under §1). `None` = names no scope.
QUESTIONS: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("q_none",     "Please provide a portfolio summary",   None),
    ("q_funded",   "Summarise the funded book",            "funded"),
    ("q_total",    "portfolio summary across all portfolios", "funded"),
    ("q_direct",   "Summarise the direct book",            "direct"),
    ("q_acquired", "Summarise the acquired book",          "acquired"),
    ("q_named",    "Show SPV2",                            "spv2"),
    ("q_named2",   "Summarise the NBS Acquired Book",      "nbs_acquired"),
    ("q_unknown",  "Summarise SPV9",                       "unresolved"),
    ("q_storage",  "Summarise the acquired_001 book",      "unresolved"),
)

#: The workspace selection, as `MiQueryRequest.source_portfolio_lens` carries it.
CALLERS: Tuple[Optional[str], ...] = (None, "total", "direct", "acquired",
                                      "spv1", "alp_acquired")


def _env() -> str:
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


def _registry(kind: str):
    from trakt_core import portfolio as portfolio_mod
    if kind == "fixture":
        return portfolio_mod.build_registry(FIXTURE_RECORDS, client_id="phase1g")
    from mi_agent_api import portfolio_context as ctx_mod
    return ctx_mod.build_registry()


def target_for(question_means: Optional[str], caller: Optional[str],
               registry) -> str:
    """The §5 product rule, stated once, as an expectation the run is scored on.

    * the question names a scope  -> the question wins, whatever the caller said
    * the question is silent      -> the caller's selection, when one was supplied
    * neither                     -> Funded, the complete funded population
    * the question names something unresolvable -> refuse; NEVER Funded
    """
    if question_means == "unresolved":
        return "unresolved"
    if question_means is not None:
        return question_means
    if caller in (None, "total"):
        return "funded"
    return caller


def observed(question: str, caller: Optional[str], registry,
             semantics) -> Dict[str, Any]:
    """What the CONTRACT resolves for this cell, and what it selects.

    Scored on the interpretation contract rather than on `resolve_scope`,
    because the contract is what the compositional path consumes and what §14
    governs. The legacy scope object is recorded alongside, unscored: it still
    falls back to every id for a scope it cannot resolve
    (`fell_back_to_total`), and Phase 1F recorded that as a stated residual of
    `trakt_core`. What matters here is that the CLAIM does not take that list.
    """
    from mi_agent import portfolio_lens as lens_mod
    from question_interpretation import projection
    from trakt_core import portfolio as portfolio_mod

    q_lens = lens_mod.resolve_lens(question, registry=registry)
    claim = projection.project(question, semantics=semantics, registry=registry,
                               caller_scope=caller).source_scope

    # The legacy path, for the record only.
    default = (lens_mod.lens_from_selection(caller, registry=registry)
               if caller is not None else None)
    effective = lens_mod.resolve_lens_with_default(question, default,
                                                   registry=registry)
    legacy = portfolio_mod.resolve_scope(registry, lens_mod.context_id(effective))

    return {
        "questionScope": q_lens.name,
        "questionLabel": q_lens.label,
        "effectiveScope": claim.scope or "unresolved",
        "effectiveLabel": claim.portfolio_label or claim.raw_text,
        "basePopulation": claim.base_population,
        "provenance": claim.provenance,
        "ids": tuple(claim.portfolio_ids),
        "state": claim.state,
        "legacyIds": tuple(legacy.portfolio_ids),
        "fellBackToTotal": bool(legacy.fell_back_to_total),
        "wonBy": {"explicit_user": "question", "caller_context": "caller",
                  "default": "neither", "unresolved": "question"}.get(
                      claim.provenance, "?"),
    }


def _expected_ids(target: str, registry) -> Optional[Tuple[str, ...]]:
    """The governed portfolio IDs the target names. `None` = must not be scored
    on IDs (an unresolved scope selects nothing legitimate)."""
    if target == "unresolved":
        return None
    if target == "funded":
        return tuple(registry.ids())
    if target in ("direct", "acquired"):
        return tuple(p.portfolio_id for p in registry.of_type(target))
    return (target,)


def score(row: Dict[str, Any], registry) -> str:
    target = row["target"]
    got = tuple(row["ids"])
    if target == "unresolved":
        # The claim must say UNRESOLVABLE and select nothing. Selecting the
        # whole book under a name the registry does not hold is the exact
        # widening §5 forbids.
        if row["state"] != "unresolvable":
            return "NOT-REFUSED"
        return "ok" if not got else "WIDENS-TO-FUNDED"
    if target == "funded":
        # The complete funded population is UNRESTRICTED, not an enumeration —
        # `base_population` carries it and `portfolio_ids` is empty, so a newly
        # onboarded portfolio is inside it without anything changing here.
        return "ok" if (row["basePopulation"] == "funded" and not got) else "WIDENS"
    want = _expected_ids(target, registry)
    if got == tuple(want or ()):
        return "ok"
    if set(want or ()) < set(got):
        return "WIDENS"
    return "NARROWS"


def capture(kind: str) -> Dict[str, Any]:
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.datasets import semantics_path

    registry = _registry(kind)
    semantics = load_mi_semantics(semantics_path())
    rows: List[Dict[str, Any]] = []
    for qid, question, means in QUESTIONS:
        for caller in CALLERS:
            row: Dict[str, Any] = {"question_id": qid, "question": question,
                                   "questionMeans": means, "caller": caller}
            row.update(observed(question, caller, registry, semantics))
            row["target"] = target_for(means, caller, registry)
            row["targetIds"] = list(_expected_ids(row["target"], registry) or [])
            row["verdict"] = score(row, registry)
            rows.append(row)
    return {"registry": kind, "portfolios": list(registry.ids()), "rows": rows}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", choices=("live", "fixture"), default="fixture")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    _env()

    data = capture(args.registry)
    rows = data["rows"]

    print("=" * 122)
    print(f"SOURCE-SCOPE PRECEDENCE — question × caller context "
          f"({args.registry} registry)")
    print("=" * 122)
    print(f"\nportfolios: {data['portfolios']}\n")
    print(f"{'question':11s} {'caller':13s} {'provenance':15s} {'base':9s} "
          f"{'selected ids':38s} {'target':13s} verdict")
    print("-" * 122)
    for row in rows:
        print(f"{row['question_id']:11s} {str(row['caller']):13s} "
              f"{str(row['provenance']):15s} {str(row['basePopulation']):9s} "
              f"{(str(list(row['ids'])) or '[]')[:38]:38s} {row['target']:13s} "
              f"{row['verdict']}")
    print("-" * 122)

    bad = [r for r in rows if r["verdict"] != "ok"]
    print(f"{len(rows) - len(bad)} of {len(rows)} cells meet the §5 target rule.")
    for row in bad:
        print(f"  {row['verdict']:18s} {row['question_id']:11s} "
              f"caller={str(row['caller']):13s} got={list(row['ids'])} "
              f"want={row['targetIds'] or 'NOT the whole book'}")

    out = Path(args.out or (_REPO / "migration_phase0" /
                            f"SCOPE_PRECEDENCE_{args.registry.upper()}.json"))
    out.write_text(json.dumps(data, indent=2, default=str) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
