#!/usr/bin/env python3
"""migration_phase0/name_resolution_answers.py — what MI ANSWERS, by name.

READ-ONLY. The Phase 1E companion to `identity_resolution_table`, and the
distinction between the two is the finding they exist to keep separate:

  * `identity_resolution_table` scores the LENS and the governed SCOPE — what
    the resolution layer decides. It runs against a deterministic fixture,
    because the live book holds one portfolio per category and cannot show a
    category collapsing onto a portfolio.
  * this instrument runs the questions END TO END against the live governed
    book and records the FIGURE that came back. A lens that resolves correctly
    and a route that then drops it produce a correct table and a wrong answer,
    which is exactly what happened to `spv1_sponsored`.

Run it before and after a change to the resolution layer; the numbers are the
evidence, not the lens names.

    python -m migration_phase0.name_resolution_answers [--out FILE]
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

#: (case, question). Chosen to cover every row of the 1E target semantics that
#: the LIVE book can express: two governed portfolios with funded rows, one
#: governed portfolio with none, both categories, the whole book, a storage
#: folder name, and a book this platform has never onboarded.
CASES: Tuple[Tuple[str, str], ...] = (
    ("named direct",      "Summarise the ALP Origination Book"),
    ("named acquired",    "Summarise the ALP Acquired Back Book"),
    ("governed id",       "Summarise the alp_acquired book"),
    ("zero-row governed", "Summarise the spv1_sponsored portfolio"),
    ("direct category",   "Summarise the direct book"),
    ("acquired category", "Summarise the acquired book"),
    ("funded category",   "Summarise the funded book"),
    ("storage id",        "Summarise the acquired_001 book"),
    ("unknown label",     "Summarise the Highgate Mortgages Book"),
    ("no scope named",    "Please provide a portfolio summary"),
)

#: The headline figure in a governed summary answer, for a before/after diff.
_FIGURE_RE = re.compile(r"holds ([\d,]+) loans with a funded balance of "
                        r"(\u00a3[\d,]+(?:\.\d+)?[a-z]*)")
_COUNT_RE = re.compile(r"([\d,]+) loans")


def _env() -> str:
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


def _figure(answer: str) -> Optional[str]:
    m = _FIGURE_RE.search(answer or "")
    if m:
        return "%s loans / %s" % (m.group(1), m.group(2))
    m = _COUNT_RE.search(answer or "")
    return ("%s loans" % m.group(1)) if m else None


def capture(client_id: str) -> List[Dict[str, Any]]:
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    ctx = ExecutionContext.for_internal(client_id)
    rows: List[Dict[str, Any]] = []
    for case, question in CASES:
        result = execute_governed_mi_query(
            MiQueryRequest(question=question), ctx).result or {}
        metadata = result.get("metadata") or {}
        answer = result.get("answer") or ""
        rows.append({
            "case": case,
            "question": question,
            "ok": bool(result.get("ok")),
            "route": metadata.get("route"),
            "controlledRefusal": bool(result.get("controlledRefusal")),
            "lensApplied": metadata.get("lensApplied"),
            "figure": _figure(answer),
            "answer": answer.strip().replace("\n", " ")[:300],
        })
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None,
                    help="write the rows as JSON to this path")
    args = ap.parse_args(argv)

    rows = capture(_env())

    print("=" * 118)
    print("NAME RESOLUTION — what MI ANSWERS, end to end, against the live book")
    print("=" * 118)
    print(f"\n{'case':18s} {'route':18s} {'ok':6s} {'refused':8s} "
          f"{'figure':28s}")
    print("-" * 118)
    for row in rows:
        print(f"{row['case']:18s} {str(row['route']):18s} "
              f"{str(row['ok']):6s} {str(row['controlledRefusal']):8s} "
              f"{str(row['figure'])[:28]:28s}")
    print("-" * 118)
    for row in rows:
        print(f"\n{row['question']}\n  {row['answer'][:200]}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n",
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
