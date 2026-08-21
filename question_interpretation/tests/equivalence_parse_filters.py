#!/usr/bin/env python3
"""Exhaustive equivalence proof for removing the _parse_filters rewrite.

    python -m question_interpretation.tests.equivalence_parse_filters

This conversion is on the answer path, and the corpus cannot prove it: exactly
ONE of the 690 real-surface questions contains "between", which is the only
construct the rewrite existed for. So the old algorithm is reproduced here
verbatim, against the same helpers, and compared to the live one across a
generated set built to hit every shape the rewrite could have affected —
"between" at the start, the middle and the end of a question, with and without a
resolvable field, either side of every clause connective, alongside postfix
bounds, and more than once in one question.

Comparing the returned filters dict, not a rendering of it: that is what the
caller consumes.
"""
from __future__ import annotations

import itertools
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.simplefilter("ignore")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE = _REPO_ROOT / "due_diligence" / "evidence" / "analytical_intent_v1"
for p in (str(_REPO_ROOT), str(_EVIDENCE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from mi_agent.llm_query_parser import (  # noqa: E402
    _FILTER_COMPARATORS, _POSTFIX_COMPARATORS, _age_equality_value, _age_metric,
    _amount_from_match, _filter_field_of, _parse_categorical_filter,
    _parse_filters,
)

#: The clause splitter exactly as it was before the conversion.
_OLD_CLAUSE_SPLIT_RE = re.compile(
    r"\band\b(?!\s+(?:over|above|older|under|below|younger|more|less)\b)"
    r"|\bwith\b|\bwhere\b|\bwhose\b|\bhaving\b")


def old_parse_filters(q: str, semantics: dict, available_columns=None,
                      unresolved: Optional[List[str]] = None) -> Dict[str, Any]:
    """The original, including the destructive rewrite. Reproduced verbatim."""
    filters: Dict[str, Any] = {}
    work_q = q
    bm = re.search(_FILTER_COMPARATORS[0][0], work_q)
    if bm:
        field = _filter_field_of(work_q[max(0, bm.start() - 40):bm.end()], semantics)
        if field:
            filters[field] = {"op": "between", "value": _amount_from_match(bm, "between")}
        work_q = work_q[:bm.start()] + " " + work_q[bm.end():]

    for clause in _OLD_CLAUSE_SPLIT_RE.split(work_q):
        clause = clause.strip()
        if not clause:
            continue
        field = _filter_field_of(clause, semantics)
        matched = False
        for pattern, op in _POSTFIX_COMPARATORS:
            m = re.search(pattern, clause)
            if m and field:
                filters[field] = {"op": op, "value": float(m.group(1))}
                matched = True
                break
        if matched:
            continue
        for pattern, op in _FILTER_COMPARATORS[1:]:
            m = re.search(pattern, clause)
            if not m:
                continue
            field = _filter_field_of(clause, semantics, available_columns,
                                     anchor=m.start()) or field
            if field:
                filters[field] = {"op": op, "value": _amount_from_match(m, op)}
                matched = True
            elif unresolved is not None:
                unresolved.append(
                    f"'{clause.strip()}' — no governed field matches this "
                    "condition, so the filter was not applied")
            break
        if matched:
            continue
        age_field = _age_metric(semantics, available_columns)
        if field == age_field and age_field:
            age_val = _age_equality_value(clause)
            if age_val is not None:
                filters[age_field] = {"op": "eq", "value": age_val}
                continue
        cat = _parse_categorical_filter(clause, semantics, available_columns)
        if cat:
            filters[cat[0]] = cat[1]
    return filters


_LEADS = ("", "balance ", "how many loans have ", "what is the balance of loans ",
          "balance by region ")
_SUBJECTS = ("LTV", "balance", "borrower age", "interest rate", "valuation",
             "unicorn score")
_BETWEEN = ("between 40 and 60", "between £100k and £250k", "between 20 and 40",
            "between 1.5 and 2.5")
_OTHER = ("above 50", "over 70", "below 40%", "at least 250000", "85+",
          "70 or above", "of more than 40%")
_CONNECTIVES = (" and ", " with ", " where ", " whose ", " having ")
_TRAILS = ("", " by region", " in London", "?")


def generated_questions():
    """Every shape the rewrite could have affected."""
    seen = set()
    for lead, subj, btw, trail in itertools.product(_LEADS, _SUBJECTS, _BETWEEN, _TRAILS):
        for q in ("%s%s %s%s" % (lead, subj, btw, trail),
                  "%s%s%s" % (lead, btw, trail),
                  "%s%s %s" % (btw, lead, subj)):
            if q not in seen:
                seen.add(q)
                yield q
    for subj, btw, conn, other, subj2 in itertools.product(
            _SUBJECTS, _BETWEEN, _CONNECTIVES, _OTHER, _SUBJECTS):
        for q in ("balance where %s %s%s%s %s" % (subj, btw, conn, subj2, other),
                  "balance where %s %s%s%s %s" % (subj2, other, conn, subj, btw)):
            if q not in seen:
                seen.add(q)
                yield q
    # More than one between, and a between with no bound after it.
    for a, b in itertools.product(_BETWEEN, _BETWEEN):
        for q in ("balance where LTV %s and age %s" % (a, b),
                  "balance where LTV %s with balance %s" % (a, b),
                  "balance between and LTV %s" % a,
                  "between %s" % a):
            if q not in seen:
                seen.add(q)
                yield q


def main() -> int:
    from mi_agent.mi_calibration import default_bank_frame
    from mi_agent.mi_query_validator import load_mi_semantics
    from question_interpretation.lexical_decisions import corpus

    semantics = load_mi_semantics(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")
    tape = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
            / "platform" / "alderbridge" / "2026-06-30"
            / "platform_canonical_typed.csv")
    columns = set(default_bank_frame(tape).columns) if tape.exists() else None

    questions = [c["question"] for c in corpus()]
    questions += list(generated_questions())
    print("questions: %d (corpus %d + generated %d)"
          % (len(questions), len(corpus()), len(questions) - len(corpus())))

    mismatches = []
    checked = 0
    for q in questions:
        for cols in (None, columns):
            old_notes, new_notes = [], []
            old = old_parse_filters(q, semantics, cols, unresolved=old_notes)
            new = _parse_filters(q, semantics, cols, unresolved=new_notes)
            checked += 1
            if old != new or old_notes != new_notes:
                mismatches.append((q, cols is not None, old, new,
                                   old_notes, new_notes))

    print("comparisons: %d" % checked)
    print("mismatches:  %d" % len(mismatches))
    for q, with_cols, old, new, on, nn in mismatches[:15]:
        print("\n  %r  (available_columns=%s)" % (q[:90], with_cols))
        print("    old filters: %r" % (old,))
        print("    new filters: %r" % (new,))
        if on != nn:
            print("    old notes: %r\n    new notes: %r" % (on, nn))
    print("\n%s" % ("-" * 70))
    print("EQUIVALENT" if not mismatches else "NOT EQUIVALENT — do not proceed")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
