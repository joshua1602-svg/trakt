#!/usr/bin/env python3
"""migration_phase0/temporal_aspect_census.py — who decides LEVEL vs MOVEMENT.

READ-ONLY. Before one owner can exist, every component that currently infers the
distinction independently has to be found and cross-tabulated, because the owner
must be a SUPERSET of all of them or the change silently removes a signal some
route relies on.

Six independent readers were found by inspection of the estate:

  A  period_change.recognition.has_change_language   CHANGE_MARKERS + period pairs
  B  llm_query_parser._COMPARE_TRIGGER_RE            "how did .. change", "compare"
  C  spec.temporal_mode == "compare"                 B's downstream outcome
  D  interpreter.deterministic                       "compare" | "since last month"
                                                     | ("changed" and "month")
  E  concentration_query._concentration_intent       "moved from" | "what changed"
                                                     | "movement" | "since last" | ...
  F  period_change.recognition.TREND_MARKERS         a SERIES, deliberately not a
                                                     two-point movement

and they disagree. The headline disagreement, found while writing this: reader A
misses "How did the balance change since last month?" — the single most
canonical movement question in the estate — because CHANGE_MARKERS carries
"changed", "change in" and "has changed" but not the bare verb "change".

    python -m migration_phase0.temporal_aspect_census [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")


class CensusError(RuntimeError):
    """The census could not be measured. Never absorbed into an empty table."""


def _questions() -> List[str]:
    out, seen = [], set()
    for f in CORPORA:
        for row in json.loads((_REPO / f).read_text())["rows"]:
            q = row.get("question") or ""
            if q and q not in seen:
                seen.add(q)
                out.append(q)
    return out


def _reader_d(q: str) -> bool:
    """`mi_agent.interpreter.deterministic`'s rule, transcribed exactly."""
    low = f" {q.lower()} "
    return ("compare" in low or "since last month" in low
            or ("changed" in low and "month" in low))


_READER_E_RE = re.compile(
    r"\bmoved from\b|\bwhat( has|'s| has been)? changed\b|\bworsen"
    r"|\bdeteriorat|\bmovement\b|\bthis month\b|\bsince last\b", re.I)


def run() -> Dict[str, Any]:
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

    from mi_agent import llm_query_parser as P
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.parsed_question import ParsedQuestion
    from mi_agent.period_change import recognition as REC
    from mi_agent_api.data_source import semantics_path

    semantics = load_mi_semantics(semantics_path())
    if not (semantics.get("fields") or {}):
        raise CensusError("CENSUS INVALID — governed MI semantics did not load")

    rows: List[Dict[str, Any]] = []
    for q in _questions():
        spec = ParsedQuestion.parse(q, semantics).spec
        text = f" {q.strip().lower()} "
        rows.append({
            "question": q,
            "A_has_change_language": bool(REC.has_change_language(q)),
            "B_compare_trigger": bool(P._COMPARE_TRIGGER_RE.search(q.lower())),
            "C_temporal_mode_compare":
                getattr(spec, "temporal_mode", None) == "compare",
            "D_deterministic": _reader_d(q),
            "E_concentration": bool(_READER_E_RE.search(q)),
            "F_trend_markers": any(m in text for m in REC.TREND_MARKERS),
            "temporal_mode": getattr(spec, "temporal_mode", None),
        })
    if len(rows) != len(_questions()) or not rows:
        raise CensusError("CENSUS INVALID — reading count does not match")
    return {"rows": rows}


#: The readers that claim to answer "is this a two-point CHANGE question".
#: F is excluded on purpose: a trend is a SERIES of levels, which the estate
#: already treats as a separate decline reason, and folding it in here would
#: make the owner claim a movement wherever a time axis appears.
MOVEMENT_READERS = ("A_has_change_language", "B_compare_trigger",
                    "C_temporal_mode_compare", "D_deterministic",
                    "E_concentration")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    result = run()
    rows = result["rows"]
    n = len(rows)

    print("=" * 92)
    print(f"TEMPORAL-ASPECT CENSUS — {n} corpus questions, six independent readers")
    print("=" * 92)
    print(f"\n{'reader':<28}{'says MOVEMENT':>14}")
    print("-" * 44)
    for r in MOVEMENT_READERS + ("F_trend_markers",):
        print(f"{r:<28}{sum(1 for x in rows if x[r]):>14}")

    print("\nAGREEMENT between the five movement readers")
    agree_all = sum(1 for x in rows if len({x[r] for r in MOVEMENT_READERS}) == 1)
    print(f"   all five agree            : {agree_all} of {n}")
    print(f"   at least one disagrees    : {n - agree_all} of {n}")

    print("\nPAIRWISE DISAGREEMENT")
    for i, a in enumerate(MOVEMENT_READERS):
        for b in MOVEMENT_READERS[i + 1:]:
            d = sum(1 for x in rows if x[a] != x[b])
            print(f"   {a:<26} vs {b:<26} {d:>4}")

    print("\nUNION vs each reader — how much each one MISSES")
    for r in MOVEMENT_READERS:
        misses = sum(1 for x in rows
                     if any(x[o] for o in MOVEMENT_READERS) and not x[r])
        print(f"   {r:<28} misses {misses:>4} of the union")
    union = sum(1 for x in rows if any(x[r] for r in MOVEMENT_READERS))
    print(f"   UNION says MOVEMENT on {union} of {n}")

    print("\nQUESTIONS WHERE THE READERS SPLIT (first 20)")
    shown = 0
    for x in rows:
        vals = {r: x[r] for r in MOVEMENT_READERS}
        if len(set(vals.values())) > 1 and shown < 20:
            shown += 1
            on = ",".join(k[0] for k, v in vals.items() if v)
            off = ",".join(k[0] for k, v in vals.items() if not v)
            print(f"   [+{on:<9}] [-{off:<9}] {x['question'][:52]}")

    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
