#!/usr/bin/env python3
"""migration_phase0/funded_book_precedence_change.py — the one authorised
client-visible change in Phase 1G, measured end to end.

READ-ONLY. Phase 1G §1 and §5.

Phase 1F found that two explicit whole-book phrasings had opposite precedence:
"across all portfolios" overrode the workspace selection and "the funded book"
deferred to it, answering 3,909 of 11,035 loans for a phrase that names the
COMPLETE funded population. Phase 1G §1 makes the business meaning
authoritative — *Funded Book = all funded assets across Direct and Acquired* —
and §5 makes the rule explicit: an explicitly named scope wins.

That is a user-visible change, so it is measured by name rather than asserted.
Every question here is one the `portfolio_summary` recogniser owns.

    python -m migration_phase0.funded_book_precedence_change
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

warnings.simplefilter("ignore")
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: The phrases §1 makes explicit, and the controls that must NOT move: a silent
#: question (still defers), a phrase already explicit before 1G (unchanged), and
#: a MEASURE that contains the word "funded" (must stay a measure, not a scope).
CASES: Tuple[Tuple[str, str, str], ...] = (
    ("changed", "Summarise the funded book", "now explicit — §1"),
    ("changed", "Summarise the funded portfolio", "now explicit — §1"),
    ("control", "Please provide a portfolio summary", "silent — must still defer"),
    ("control", "portfolio summary across all portfolios",
     "already explicit — must not move"),
    ("control", "Summarise the acquired book", "explicit narrowing — must not move"),
    ("control", "What is the funded balance?", "a MEASURE, not a scope"),
    ("control", "Show funded balance by region", "a MEASURE, not a scope"),
)

DEFAULTS: Tuple[Optional[str], ...] = (None, "acquired", "direct")
_FIGURE = re.compile(r"([\d,]+) loans")


def _env() -> str:
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(
        _REPO / "migration_phase0" / "FUNDED_BOOK_PRECEDENCE.json"))
    args = ap.parse_args(argv)
    client_id = _env()

    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    ctx = ExecutionContext.for_internal(client_id)
    rows: List[Dict[str, Any]] = []
    print("=" * 108)
    print("FUNDED BOOK PRECEDENCE — the one authorised client-visible change")
    print("=" * 108)
    print(f"\n{'kind':8s} {'question':44s} {'no selection':14s} "
          f"{'UI=Acquired':14s} {'UI=Direct':14s}")
    print("-" * 108)
    for kind, question, why in CASES:
        cells = []
        for default in DEFAULTS:
            result = execute_governed_mi_query(
                MiQueryRequest(question=question,
                               source_portfolio_lens=default), ctx).result or {}
            answer = result.get("answer") or ""
            match = _FIGURE.search(answer)
            cells.append(match.group(1) if match else
                         ("refused" if result.get("controlledRefusal") else "-"))
            rows.append({"kind": kind, "question": question, "why": why,
                         "default": default, "loans": cells[-1],
                         "ok": bool(result.get("ok"))})
        print(f"{kind:8s} {question[:44]:44s} {cells[0]:14s} {cells[1]:14s} "
              f"{cells[2]:14s}")
    print("-" * 108)
    print("A 'changed' row answering the same figure across all three columns is\n"
          "the intended outcome: the question names the whole funded book, so the\n"
          "workspace selection no longer narrows it.")

    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {Path(args.out).relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
