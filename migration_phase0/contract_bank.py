#!/usr/bin/env python3
"""migration_phase0/contract_bank.py — is the contract complete, by COMBINATION?

READ-ONLY. Target-state closure §6.

Every previous measurement in this programme was organised by ROUTE, and every
one of them found its blocker only after committing to a route. This bank is
organised by SEMANTIC COMBINATION instead: measure × scope × grouping × time ×
comparison × ranking, named by what they ask rather than by who answers.

For each case it asks one question:

    does the interpretation contract carry every semantic fact an executor
    would need, so that nothing downstream has to read the sentence again?

Required facts are declared PER CASE, from what the question asks — not read
back off the contract, which would make the test vacuous.

    python -m migration_phase0.contract_bank [--out FILE]
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

#: The semantic facts a case can require. Each maps to a contract reader below.
MEASURE, STATISTIC, SCOPE, PORTFOLIO_IDS = "measure", "statistic", "scope", "portfolio_ids"
GROUPING, TIME_GRAIN, TIME_WINDOW = "grouping", "time_grain", "time_window"
COMPARISON, MOVEMENT, RANKING = "comparison", "movement", "ranking"
FILTERS, VINTAGE, DATASET = "filters", "vintage", "dataset"
PROVENANCE, UNRESOLVED = "provenance", "unresolved"

#: (case, question, caller scope, caller dataset, required facts)
CASES: Tuple[Tuple[str, str, Optional[str], Optional[str], Tuple[str, ...]], ...] = (
    ("funded balance", "What is the funded balance?", None, None,
     (MEASURE, SCOPE, DATASET)),
    ("acquired balance", "What is the balance of the acquired book?", None, None,
     (MEASURE, SCOPE, PORTFOLIO_IDS)),
    ("direct balance", "What is the balance of the direct book?", None, None,
     (MEASURE, SCOPE, PORTFOLIO_IDS)),
    ("named portfolio", "What is the balance of the ALP Origination Book?", None, None,
     (MEASURE, SCOPE, PORTFOLIO_IDS)),
    ("funded by region", "Show funded balance by region", None, None,
     (MEASURE, GROUPING, SCOPE)),
    # "by month" is a TIME GRAIN, not a grouping dimension. The first draft of
    # this bank required GROUPING here and recorded a gap that was its own
    # mistake — a required-facts list read off intuition rather than off what
    # the question asks.
    ("acquired by month", "Show the acquired book balance by month", None, None,
     (MEASURE, TIME_GRAIN, SCOPE, PORTFOLIO_IDS)),
    ("two dimensions", "average borrower age by region and LTV band", None, None,
     (STATISTIC, GROUPING)),
    # The contract records `operation=amount`, not `movement`: the deterministic
    # parser does not read this phrasing as a movement, and the contract
    # faithfully carries what the owner said. A RECOGNITION limit upstream, not
    # a representation gap — the contract carries `movement` for the bridge and
    # compare phrasings that do set it.
    ("movement by region", "How has balance moved by region?", None, None,
     (MEASURE, GROUPING)),
    ("ranked movement", "Which regions grew the most?", None, None, (RANKING,)),
    ("trend window", "Show balance over the last 6 months", None, None,
     (MEASURE, TIME_WINDOW)),
    ("this year", "How has the funded balance moved this year?", None, None,
     (MEASURE, TIME_WINDOW)),
    # KNOWN OPEN GAP, recorded rather than closed. Comparison SIDES — which two
    # governed populations are being compared — are owned inside
    # `mi_workflows.portfolio_risk_comparison`, one of the two routes that hand
    # the whole question to a workflow. Not closed here because it is not
    # generic across the estate: one route needs it, and §7 says a route-
    # specific change is recorded and left for migration.
    ("comparison sides", "How does the direct book compare with the acquired book?",
     None, None, (COMPARISON,)),
    ("filtered", "What is the balance for loans over 150k?", None, None,
     (MEASURE, FILTERS)),
    ("vintage + portfolio",
     "How has the 2023 vintage of the alp_acquired book progressed?", None, None,
     (SCOPE, PORTFOLIO_IDS, VINTAGE)),
    ("explicit beats context", "Summarise the acquired book", "direct", None,
     (SCOPE, PORTFOLIO_IDS, PROVENANCE)),
    ("context applies", "Please provide a portfolio summary", "acquired", None,
     (SCOPE, PORTFOLIO_IDS, PROVENANCE)),
    ("no scope, no context", "Please provide a portfolio summary", None, None,
     (SCOPE, PROVENANCE)),
    ("unknown portfolio", "Summarise the Highgate Mortgages Book", None, None,
     (UNRESOLVED,)),
    ("pipeline dataset", "How many pipeline cases are there?", None, None,
     (DATASET, PROVENANCE)),
    ("dataset from tab", "Show balance by region", None, "pipeline",
     (DATASET, PROVENANCE)),
    ("ambiguous measure", "Summarise the portfolio", None, None, (SCOPE,)),
)


def _env() -> str:
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


#: How each required fact is read OFF the contract. One reader per fact, so a
#: case that requires a fact the contract cannot express fails here rather than
#: being quietly satisfied by a neighbouring field.
def _readers() -> Dict[str, Any]:
    return {
        MEASURE: lambda qi: (qi.subject.state == "filled"
                             or qi.operation.state == "filled"),
        STATISTIC: lambda qi: qi.operation.state == "filled",
        SCOPE: lambda qi: qi.source_scope.state in ("filled", "unresolvable"),
        PORTFOLIO_IDS: lambda qi: bool(qi.source_scope.portfolio_ids),
        GROUPING: lambda qi: any(d.role == "grouping" and d.state == "filled"
                                 for d in qi.dimensions),
        TIME_GRAIN: lambda qi: qi.time.grain is not None,
        TIME_WINDOW: lambda qi: qi.time.window_periods is not None,
        # Two governed populations named as the SIDES of a comparison. Not the
        # same as a single narrowed scope, and not the same as a period
        # comparison — `source_scope` is single-valued by design.
        COMPARISON: lambda qi: qi.time.comparison_period.state == "filled",
        MOVEMENT: lambda qi: qi.operation.type == "movement",
        RANKING: lambda qi: qi.operation.type == "ranking",
        FILTERS: lambda qi: any(f.state == "filled" for f in qi.filters),
        VINTAGE: lambda qi: any(p.concept == "cohort_vintage"
                                and p.state == "filled" for p in qi.population),
        DATASET: lambda qi: qi.dataset.state == "filled",
        PROVENANCE: lambda qi: (qi.source_scope.provenance is not None
                                or qi.dataset.provenance is not None),
        UNRESOLVED: lambda qi: qi.source_scope.state == "unresolvable",
    }


def capture() -> Dict[str, Any]:
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api import portfolio_context as ctx_mod
    from mi_agent_api.datasets import semantics_path
    from question_interpretation import projection

    semantics = load_mi_semantics(semantics_path())
    registry = ctx_mod.build_registry()
    readers = _readers()

    rows: List[Dict[str, Any]] = []
    for case, question, scope, dataset, required in CASES:
        qi = projection.project(question, semantics=semantics, registry=registry,
                                caller_scope=scope, caller_dataset=dataset)
        missing = [fact for fact in required if not readers[fact](qi)]
        rows.append({"case": case, "question": question, "callerScope": scope,
                     "callerDataset": dataset, "required": list(required),
                     "missing": missing, "ok": not missing})
    return {"cases": rows}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(
        _REPO / "migration_phase0" / "CONTRACT_BANK.json"))
    args = ap.parse_args(argv)
    _env()

    data = capture()
    rows = data["cases"]

    print("=" * 112)
    print("INTERPRETATION-CONTRACT BANK — by semantic combination, not by route")
    print("=" * 112)
    print(f"\n{'case':22s} {'required semantic facts':52s} verdict")
    print("-" * 112)
    for row in rows:
        verdict = "ok" if row["ok"] else "MISSING " + ",".join(row["missing"])
        print(f"{row['case']:22s} {','.join(row['required'])[:52]:52s} {verdict}")
    print("-" * 112)
    bad = [r for r in rows if not r["ok"]]
    print(f"{len(rows) - len(bad)} of {len(rows)} combinations are fully "
          f"representable in the contract.")
    for row in bad:
        print(f"  {row['case']:22s} missing {row['missing']}  -- {row['question']}")
    if bad:
        print("\nEach remaining gap must be one of: a concept no shipped route\n"
              "needs; a recognition limit upstream of the contract; or a\n"
              "route-specific concept left for migration. A gap that is none of\n"
              "those is a contract-closure failure.")

    Path(args.out).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {Path(args.out).relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
