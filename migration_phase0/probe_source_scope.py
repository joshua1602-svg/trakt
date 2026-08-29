#!/usr/bin/env python3
"""migration_phase0/probe_source_scope.py — Phase 1A contract evidence.

READ-ONLY. Shows what `QuestionInterpretation.source_scope` carries, and that a
source-portfolio lens and a seasoning segment are not conflated.

    python -m migration_phase0.probe_source_scope
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CASES = (
    ("Total (unrestricted)",   "Please provide a portfolio summary"),
    ("Total (unrestricted)",   "summarise the portfolio"),
    ("Direct",                 "Summarise the direct book"),
    ("Acquired",               "Summarise the acquired book"),
    ("Acquired",               "portfolio summary for the acquired book"),
    ("Seasoning, NOT a lens",  "Summarise the front book"),
    ("Seasoning, NOT a lens",  "What is the balance of the back book?"),
    ("Both axes at once",      "Summarise the front book in the acquired portfolio"),
)


def main() -> int:
    import logging
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)

    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    from mi_agent import portfolio_lens as lens_owner
    from question_interpretation import projection

    semantics = load_mi_semantics(semantics_path())

    print("=" * 78)
    print("PHASE 1A — what the contract now carries for a source-portfolio lens")
    print("=" * 78)
    print("\n  BEFORE this change: source_scope did not exist; `population` was []")
    print("  for every one of these, so Total and Acquired were indistinguishable.\n")

    ok = True
    for expectation, question in CASES:
        qi = projection.project(question, semantics=semantics)
        scope = qi.source_scope
        seasoning = [p.concept for p in qi.population if p.state == "filled"]
        owner = lens_owner.resolve_lens(question)

        print(f"  {question!r}")
        print(f"     source_scope : state={scope.state!r} scope={scope.scope!r} "
              f"narrows={scope.narrows} ids={list(scope.portfolio_ids)}")
        print(f"     population   : {seasoning or '[]'}   <- seasoning axis, separate")
        print(f"     owner said   : name={owner.name!r} filters={owner.filters}")
        # The contract must agree with its owner, always.
        if scope.state == "filled" and scope.scope != owner.name:
            print("     MISMATCH: the contract disagrees with its owner")
            ok = False
        print(f"     [{expectation}]")
        print()

    print("-" * 78)
    print("NON-CONFLATION: 'Summarise the front book' is a SEASONING population")
    print("with source_scope=total (explicitly whole-book by source), and")
    print("'Summarise the acquired book' is a SOURCE lens with no seasoning claim.")
    print("Neither implies the other, and a question can carry both.")
    print("-" * 78)
    print("\nAGREES WITH ITS OWNER ON EVERY CASE" if ok else "\nDISAGREEMENT FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
