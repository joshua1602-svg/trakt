#!/usr/bin/env python3
"""question_interpretation/shipped_shapes.py

THE SIXTH MEASUREMENT SURFACE — three ordinary shapes, scored four ways.

Why it had to exist
-------------------
The 44 variations in `nl_bank.py` were always the right QUESTIONS. What was
wrong was what the runner does with the answers. `nl_score.grade` reads an
outcome LABEL — did it answer, did it refuse, did it disclose a limitation —
and a question that returns a complete, well-formatted, WRONG number grades
CORRECT as long as it did not refuse. That is the one outcome that reaches a
reader as fact, and it is the one outcome that bank cannot report.

So this surface does not grade a label. It computes the answer from the book
with pandas and compares. Four outcomes, and the distinction that matters is
between the two refusals:

    CORRECT             answered, and the figures match the book
    WRONG_ANSWER        answered, and they do not — or presented as complete
                        while missing what the shape requires
    HONEST_REFUSAL      declined, and the thing it cannot do is genuinely
                        unavailable
    UNHELPFUL_REFUSAL   declined while holding everything needed to answer, or
                        asked for a clarification the sentence already supplies

An unhelpful refusal is not a wrong number and must never be counted as one.
It is also not a success: it is a question the product could answer and didn't.

Ground truth is recomputed from `data_source.get_dataframe()` on every run and
never pinned to a literal, so a book change moves the truth and the expectation
together rather than turning this surface red.

    python -m question_interpretation.shipped_shapes
    python -m question_interpretation.shipped_shapes --wrong-answers
    python -m question_interpretation.shipped_shapes --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BANK = _REPO_ROOT / "config" / "mi" / "golden_questions" / "shipped_shapes.yaml"

CORRECT = "correct"
WRONG_ANSWER = "wrong_answer"
HONEST_REFUSAL = "honest_refusal"
UNHELPFUL_REFUSAL = "unhelpful_refusal"
OUTCOMES = (CORRECT, WRONG_ANSWER, HONEST_REFUSAL, UNHELPFUL_REFUSAL)

#: How close a figure quoted in prose must be to the book. Answers round to
#: 3 significant figures ("£1.96bn"), so the tolerance is on the ROUNDING, not
#: on the arithmetic — 1% would let a genuinely different number through.
_REL_TOL = 0.005


def bank() -> Dict[str, Any]:
    return yaml.safe_load(BANK.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# The shipped path
# --------------------------------------------------------------------------- #
def questions_from(path: str) -> List[Tuple[str, str]]:
    """`[(id, question)]` from a plain text file — one question per line.

    Blank lines and `#` comments are skipped. Deliberately the dumbest possible
    format: a person pastes the questions they actually want to ask.
    """
    out: List[Tuple[str, str]] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(("q%03d" % (len(out) + 1), line))
    return out


def _client():
    """`execute_governed_mi_query`, routing exactly as shipped.

    Deliberately NOT the FastAPI TestClient the robustness runner uses. Both
    reach the same entry point, so this is not a different path — it is the
    same path without a serialisation round trip in the way, which keeps the
    artifacts inspectable as objects rather than as JSON the runner discards.
    """
    _apply_env()
    import logging
    logging.disable(logging.WARNING)

    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    ctx = ExecutionContext.for_internal(_CLIENT_ID or "mylender")
    return lambda q: (execute_governed_mi_query(
        MiQueryRequest(question=q), ctx).result or {})


#: Set by `--csv` / `--llm`. Module-level so `_client` and `_book` agree about
#: which book is loaded — the two used to configure the environment separately,
#: which is how a run could grade one book while reporting another.
_CSV: Optional[str] = None
_LLM: bool = False
_CLIENT_ID: Optional[str] = None


def _apply_env() -> None:
    """Point the service at a book, and set the model switch. ONE place.

    With `--csv`, `MI_AGENT_DATA_CSV` names the file explicitly. Without it the
    bundled demo book is used. That distinction matters more than it looks:
    misconfigured environment variables do NOT fail — the service falls back to
    a synthetic demo book and answers plausibly from the wrong data. `--verify`
    prints the source it actually loaded so a reader can check before trusting
    a number.
    """
    global _CLIENT_ID
    if _CSV:
        os.environ["MI_AGENT_DATA_CSV"] = str(Path(_CSV).resolve())
        os.environ.setdefault("TRAKT_RUNTIME_MODE", "test")
        _CLIENT_ID = _CLIENT_ID or "mylender"
    else:
        from demo_platform import config as cfg
        os.environ.update(cfg.mi_env(period_role="current"))
        _CLIENT_ID = cfg.CLIENT_ID
    os.environ["MI_AGENT_LLM_PARSER"] = "on" if _LLM else "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "1" if _LLM else "0"


def describe_source() -> Dict[str, Any]:
    """What the service ACTUALLY loaded — kind, label, path, rows, columns."""
    _apply_env()
    from mi_agent_api import data_source
    frame = data_source.get_dataframe()
    info = dict(data_source.data_source_info())
    info["rows"] = int(len(frame))
    info["columns"] = int(len(frame.columns))
    return info


def _book():
    _apply_env()
    from mi_agent_api import data_source
    return data_source.get_dataframe()


def truth(df) -> Dict[str, Any]:
    """Every figure this surface asserts, computed from the book.

    Both LTV conventions are carried because the service is entitled to either;
    an answer matching EITHER over the right population is correct, and an
    answer matching either over the WHOLE BOOK is the defect this looks for.
    """
    bal = df["current_outstanding_balance"]
    ltv = df["current_loan_to_value"]
    big = df["current_outstanding_balance"] > 150000

    def wavg(mask) -> float:
        d = df[mask]
        w = d["current_outstanding_balance"]
        return float((d["current_loan_to_value"] * w).sum() / w.sum())

    whole = df.index == df.index
    # dropna=False, and that is not a detail. A plain groupby drops the two
    # loans with no `ltv_bucket` and £337,343.21 with them — so a surface built
    # on it would have graded the SHIPPED answer wrong for reconciling to the
    # book. The product surfaces those rows as "Unknown / Missing" and its
    # cross-tab totals to the whole book; the first version of this grader did
    # not, and called that a defect. Recorded because it is the failure mode
    # this whole surface exists to avoid, found in the surface itself.
    cells = df.groupby(["ltv_bucket", "ticket_bucket"], dropna=False)[
        "current_outstanding_balance"].sum()
    return {
        "loan_count": int(len(df)),
        "total_balance": float(bal.sum()),
        "ltv_whole_simple": float(ltv.mean()),
        "ltv_whole_weighted": wavg(whole),
        "big_count": int(big.sum()),
        "big_balance": float(df[big]["current_outstanding_balance"].sum()),
        "ltv_big_simple": float(df[big]["current_loan_to_value"].mean()),
        "ltv_big_weighted": wavg(big),
        "ltv_buckets": int(df["ltv_bucket"].nunique()),
        "ticket_buckets": int(df["ticket_bucket"].nunique()),
        "cross_cells": int(len(cells)),
        "cross_total": float(cells.sum()),
    }


# --------------------------------------------------------------------------- #
# Reading figures out of prose
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"£?\s?([\d,]+(?:\.\d+)?)\s*(bn|m|k|%)?", re.IGNORECASE)
_SCALE = {"bn": 1e9, "m": 1e6, "k": 1e3, None: 1.0, "": 1.0}


def numbers_in(text: str) -> List[float]:
    """Every quantity the sentence states, scaled. Used only to ask whether a
    figure IS PRESENT — never to decide what the answer meant."""
    out: List[float] = []
    for raw, suffix in _NUM.findall(text or ""):
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        suffix = (suffix or "").lower()
        if suffix == "%":
            out.append(value)
        else:
            out.append(value * _SCALE.get(suffix, 1.0))
    return out


def _near(values, target: float, tol: float) -> bool:
    if target == 0:
        return any(abs(v) < 1e-9 for v in values)
    return any(abs(v - target) <= abs(target) * tol for v in values)


def states(text: str, target: float, tol: float = _REL_TOL) -> bool:
    """Whether the prose states this figure, within rounding."""
    return _near(numbers_in(text), target, tol)


def presents(seen: Dict[str, Any], target: float, tol: float = _REL_TOL) -> bool:
    """Whether the ANSWER AS DELIVERED states this figure — prose or KPI card.

    Both, because the product legitimately uses both and a reader sees both.
    """
    return (states(seen["answer"], target, tol)
            or _near(seen.get("kpi_values") or (), target, tol))


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #
def _table_total(artifacts: List[Dict[str, Any]]) -> Optional[float]:
    """The measure column of the first table, summed — what the reader adds up."""
    for art in artifacts:
        if art.get("type") != "table":
            continue
        total = 0.0
        seen_any = False
        for row in (art.get("rows") or ()):
            for key, val in row.items():
                if key.endswith("_sum") and isinstance(val, (int, float)):
                    total += float(val)
                    seen_any = True
        return total if seen_any else None
    return None


def kpi_values(artifacts: List[Dict[str, Any]]) -> List[float]:
    """Every figure the KPI card states.

    A summary answers in its CARD, not in its prose: "What are the headline
    numbers?" returns the sentence "Count of loans - 11,035 loans" and a KPI
    artifact carrying `current_outstanding_balance_sum = 1964886258.21`. The
    first version of this grader read only the prose and called that a summary
    with no balance in it. A grader that cannot see where the product puts its
    answer reports the product wrong, which is the same defect one level up.
    """
    out: List[float] = []
    for art in artifacts:
        for kpi in (art.get("kpis") or ()):
            raw = kpi.get("rawValue")
            if isinstance(raw, (int, float)):
                out.append(float(raw))
    return out


def observe(ask, question: str) -> Dict[str, Any]:
    res = ask(question)
    meta = res.get("metadata") or {}
    guard = meta.get("semantic_guard") or res.get("semanticGuard") or {}
    summary = res.get("executionSummary") or {}
    arts = res.get("artifacts") or []
    return {
        "ok": bool(res.get("ok")),
        "route": meta.get("route"),
        "view": meta.get("datasetContext"),
        "verdict": guard.get("verdict"),
        "facets": [(str(f.get("kind")), str(f.get("status")))
                   for f in (guard.get("facets") or [])],
        "answer": str(res.get("answer") or res.get("error") or ""),
        "measure": summary.get("measure"),
        "aggregation": summary.get("aggregation"),
        "filters_applied": list(summary.get("filtersApplied") or ()),
        "dimensions_applied": list(summary.get("dimensionsApplied") or ()),
        "population": summary.get("population"),
        "group_count": summary.get("groupCount"),
        "artifacts": [{"type": a.get("type"), "title": a.get("title"),
                       "rows": len(a.get("rows") or []),
                       "columns": [c.get("key") for c in (a.get("columns") or [])],
                       "kpis": [{"field": k.get("field"), "value": k.get("value"),
                                 "rawValue": k.get("rawValue")}
                                for k in (a.get("kpis") or ())],
                       "sample": (a.get("rows") or [])[:3]}
                      for a in arts],
        "kpi_values": kpi_values(arts),
        "table_total": _table_total(arts),
        "controlled_refusal": bool(res.get("controlledRefusal")),
    }


# --------------------------------------------------------------------------- #
# Scoring — one grader per shape, because the shapes assert different things
# --------------------------------------------------------------------------- #
def _refusal_kind(available: bool, why: str) -> Tuple[str, str]:
    """HONEST when the thing it says it cannot do is genuinely unavailable.

    `available` is a statement about the BOOK and the capability set, never
    about what this particular run happened to resolve. The first version of
    this grader asked "did execution reach the filter?" and so graded B4 honest
    and B1 unhelpful for the same question asked two ways — which measures the
    parser's luck, not the product's honesty. A refusal is honest only if the
    reader could not have been served; if the field is in the book and the
    sentence supplies the role, declining is a capability gap however politely
    it is worded.
    """
    return (UNHELPFUL_REFUSAL if available else HONEST_REFUSAL), why


def grade_A(seen: Dict[str, Any], t: Dict[str, Any]) -> Tuple[str, str]:
    """A portfolio summary must state the book's SIZE and its BALANCE.

    That is the whole of what kpi_028-031 assert about and the whole of what
    they do not check.
    """
    text = seen["answer"]
    if not seen["ok"]:
        return _refusal_kind(
            True,
            "the whole-book count and balance need no clarification to compute; "
            "four other phrasings of this question answer")
    has_count = presents(seen, t["loan_count"])
    has_balance = presents(seen, t["total_balance"])
    if has_count and has_balance:
        return CORRECT, "states %d loans and £%.2f" % (t["loan_count"],
                                                       t["total_balance"])
    missing = [n for n, ok in (("loan count", has_count),
                               ("total balance", has_balance)) if not ok]
    return WRONG_ANSWER, ("presented as a summary with no %s in it"
                          % " or ".join(missing))


def grade_B(seen: Dict[str, Any], t: Dict[str, Any]) -> Tuple[str, str]:
    """The LTV of the loans OVER £150k, not the LTV of the book.

    The defect this shape exists for returns the whole-book figure looking
    complete, so the grader's job is to tell 45.4% from 43.1%.
    """
    text = seen["answer"]
    right = (t["ltv_big_simple"], t["ltv_big_weighted"])
    wrong = (t["ltv_whole_simple"], t["ltv_whole_weighted"])
    if not seen["ok"]:
        # Did it already hold the answer when it declined?
        # Both fields are in the book and the sentence supplies both roles, so
        # no refusal of this shape can be honest on this tape.
        held = (seen["population"] == t["big_count"]
                and bool(seen["filters_applied"]))
        return _refusal_kind(
            True,
            ("narrowed to %s loans and applied %s, then declined"
             % (seen["population"], seen["filters_applied"])) if held
            else ("declined without applying the threshold; the sentence "
                  "supplies it and %s is in the book" % "current_loan_to_value"))
    if any(presents(seen, v, 0.02) for v in right):
        return CORRECT, "states the LTV of the %d loans over £150k" % t["big_count"]
    if any(presents(seen, v, 0.02) for v in wrong):
        return WRONG_ANSWER, (
            "states the WHOLE-BOOK LTV (%.2f%% simple / %.2f%% weighted) for a "
            "question narrowed to %d of %d loans"
            % (wrong[0], wrong[1], t["big_count"], t["loan_count"]))
    if seen["population"] not in (None, t["big_count"]):
        return WRONG_ANSWER, ("answered over %s loans; the narrowing selects %d"
                              % (seen["population"], t["big_count"]))
    # A LOAN-LEVEL answer is an answer. "Show me the LTV for loans over £150k"
    # asks to SEE them, and a table of the right 5,857 loans carrying an LTV
    # column answers it — there is no aggregate for this grader to match because
    # the reader did not ask for one.
    #
    # The fourth time this grader has failed to see where the product put its
    # answer, and the fourth time it did so in the direction of marking the
    # product wrong for being right. The population check above is what keeps
    # this from being leniency: a loan-level table over the WHOLE BOOK is still
    # a wrong answer, and B5's defect could not pass through here.
    if seen["population"] == t["big_count"]:
        for art in seen["artifacts"]:
            if art["type"] == "table" and art["rows"] and (
                    "current_loan_to_value" in (art["columns"] or ())):
                return CORRECT, (
                    "loan-level over the %d loans over £150k, carrying the LTV "
                    "column" % t["big_count"])
    return WRONG_ANSWER, "answered without stating an LTV this surface can match"


def grade_C(seen: Dict[str, Any], t: Dict[str, Any]) -> Tuple[str, str]:
    """BOTH dimensions, in the receipt AND in the table that is rendered."""
    if not seen["ok"]:
        return _refusal_kind(
            True,
            "resolved %s as the axis and %r as the measure, then declined"
            % (seen["dimensions_applied"], seen["measure"]))
    dims = seen["dimensions_applied"]
    if len(dims) < 2:
        return WRONG_ANSWER, (
            "a two-dimension question answered over %d dimension(s) %s, "
            "presented as an answer" % (len(dims), dims))
    tables = [a for a in seen["artifacts"] if a["type"] == "table"]
    if not tables:
        return WRONG_ANSWER, "both dimensions in the receipt, no table rendered"
    cols = set(tables[0]["columns"] or ())
    if not {"ltv_bucket", "ticket_bucket"} <= cols:
        return WRONG_ANSWER, ("the table carries %s, not both dimensions"
                              % sorted(cols))
    if seen["group_count"] != t["cross_cells"]:
        return WRONG_ANSWER, ("%s groups reported; the book has %d populated "
                              "(ltv_bucket, ticket_bucket) cells"
                              % (seen["group_count"], t["cross_cells"]))
    # THE ASSERTION THAT MATTERS: a cross-tab a reader adds up must come to the
    # book. A dropped cell is invisible in a 50-row table and obvious here.
    total = seen.get("table_total")
    if total is None or abs(total - t["total_balance"]) > 0.01:
        return WRONG_ANSWER, ("the table sums to %s; the book holds £%.2f"
                              % (total, t["total_balance"]))
    return CORRECT, ("both dimensions applied, %d cells, reconciles to £%.2f"
                     % (t["cross_cells"], t["total_balance"]))


GRADERS = {"A": grade_A, "B": grade_B, "C": grade_C}


def run() -> Dict[str, Any]:
    doc = bank()
    ask = _client()
    t = truth(_book())
    rows: List[Dict[str, Any]] = []
    for shape in doc["shapes"]:
        grader = GRADERS[shape["id"]]
        for case in shape["questions"]:
            seen = observe(ask, case["question"])
            outcome, note = grader(seen, t)
            rows.append({"shape": shape["id"], "shape_name": shape["name"],
                         "id": case["id"], "question": case["question"],
                         "outcome": outcome, "note": note, "seen": seen})
    return {"truth": t, "rows": rows}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
_LINE = "-" * 78


def report(result: Dict[str, Any], wrong_in_full: bool = True) -> int:
    rows, t = result["rows"], result["truth"]
    counts = {o: sum(1 for r in rows if r["outcome"] == o) for o in OUTCOMES}

    print("=" * 78)
    print("SHIPPED SHAPES — execute_governed_mi_query, routing exactly as shipped")
    print("=" * 78)
    print("book: %d loans, £%.2f funded" % (t["loan_count"], t["total_balance"]))
    print("     over £150k: %d loans, LTV %.2f%% simple / %.2f%% weighted"
          % (t["big_count"], t["ltv_big_simple"], t["ltv_big_weighted"]))
    print("     whole book LTV: %.2f%% simple / %.2f%% weighted"
          % (t["ltv_whole_simple"], t["ltv_whole_weighted"]))
    print("     ltv_bucket x ticket_bucket: %d populated cells of %d x %d"
          % (t["cross_cells"], t["ltv_buckets"], t["ticket_buckets"]))
    print()
    print("   correct              %2d" % counts[CORRECT])
    print("   wrong answer         %2d   <- the only outcome that reaches a reader as fact"
          % counts[WRONG_ANSWER])
    print("   honest refusal       %2d" % counts[HONEST_REFUSAL])
    print("   unhelpful refusal    %2d" % counts[UNHELPFUL_REFUSAL])
    print("   %s" % _LINE[:24])
    print("   total                %2d" % len(rows))
    print()

    for shape_id in ("A", "B", "C"):
        shape_rows = [r for r in rows if r["shape"] == shape_id]
        if not shape_rows:
            continue
        print("%s — %s" % (shape_id, shape_rows[0]["shape_name"]))
        for r in shape_rows:
            print("   %-3s %-18s %s" % (r["id"], r["outcome"], r["question"]))
            print("       %s" % r["note"])
        print()

    if wrong_in_full:
        wrong = [r for r in rows if r["outcome"] == WRONG_ANSWER]
        print("=" * 78)
        print("EVERY WRONG ANSWER, IN FULL  (%d)" % len(wrong))
        print("=" * 78)
        if not wrong:
            print("none")
        for r in wrong:
            s = r["seen"]
            print()
            print(_LINE)
            print("%s  %s" % (r["id"], r["question"]))
            print(_LINE)
            print("  why wrong : %s" % r["note"])
            print("  route     : %s        view: %s" % (s["route"], s["view"]))
            print("  verdict   : %s        ok: %s" % (s["verdict"], s["ok"]))
            print("  facets    : %s" % (s["facets"] or "[]"))
            print("  measure   : %s (%s)" % (s["measure"], s["aggregation"]))
            print("  filters   : %s" % (s["filters_applied"] or "[]"))
            print("  dimensions: %s" % (s["dimensions_applied"] or "[]"))
            print("  population: %s        groups: %s"
                  % (s["population"], s["group_count"]))
            print("  artifacts : %s" % [(a["type"], a["rows"]) for a in s["artifacts"]])
            for a in s["artifacts"]:
                if a["columns"]:
                    print("      %-6s columns %s" % (a["type"], a["columns"]))
            print("  ANSWER AS THE READER RECEIVES IT:")
            for line in (s["answer"] or "(empty)").splitlines() or ["(empty)"]:
                print("      %s" % line)
    return counts[WRONG_ANSWER]


def self_test() -> int:
    """Can-fail probes for the GRADERS, because these graders were wrong twice.

    Both errors reported the product wrong for something the product did right:
    reading only the prose when the figure was in the KPI card, and grading a
    reconciling cross-tab wrong because the grader's own pandas ground truth
    dropped two null-bucket rows. A grader that cannot see where the product
    puts its answer is the same defect this surface exists to find, one level
    up. So every grader is probed here in both directions.
    """
    t = {"loan_count": 100, "total_balance": 1000.0,
         "ltv_whole_simple": 40.0, "ltv_whole_weighted": 43.0,
         "big_count": 50, "big_balance": 700.0,
         "ltv_big_simple": 44.0, "ltv_big_weighted": 45.0,
         "ltv_buckets": 3, "ticket_buckets": 2, "cross_cells": 5,
         "cross_total": 1000.0}

    def seen(**kw):
        base = {"ok": True, "route": None, "view": "funded", "verdict": "ok",
                "facets": [], "answer": "", "measure": None, "aggregation": None,
                "filters_applied": [], "dimensions_applied": [], "population": None,
                "group_count": None, "artifacts": [], "kpi_values": [],
                "table_total": None, "controlled_refusal": False}
        base.update(kw)
        return base

    both = [{"type": "table", "title": "t", "rows": 5,
             "columns": ["ltv_bucket", "ticket_bucket", "x_sum"], "kpis": [],
             "sample": []}]
    failures: List[str] = []

    def expect(label, got, want):
        if got[0] != want:
            failures.append("%s: graded %s, expected %s (%s)"
                            % (label, got[0], want, got[1]))

    # A — the figure counts wherever the product puts it
    expect("A prose", grade_A(seen(answer="100 loans, £1000"), t), CORRECT)
    expect("A kpi card", grade_A(seen(kpi_values=[100.0, 1000.0]), t), CORRECT)
    expect("A no balance", grade_A(seen(answer="100 loans"), t), WRONG_ANSWER)
    expect("A refusal", grade_A(seen(ok=False), t), UNHELPFUL_REFUSAL)

    # B — the whole-book figure is the defect, the narrowed one is the answer
    expect("B narrowed", grade_B(seen(kpi_values=[45.0], population=50), t), CORRECT)
    expect("B whole book", grade_B(seen(kpi_values=[43.0], population=100), t),
           WRONG_ANSWER)
    expect("B wrong population",
           grade_B(seen(kpi_values=[99.0], population=77), t), WRONG_ANSWER)
    expect("B refusal", grade_B(seen(ok=False), t), UNHELPFUL_REFUSAL)
    _loan_table = [{"type": "table", "title": "t", "rows": 50,
                    "columns": ["loan_identifier", "current_loan_to_value"],
                    "kpis": [], "sample": []}]
    expect("B loan-level, right population",
           grade_B(seen(population=50, artifacts=_loan_table), t), CORRECT)
    expect("B loan-level, WHOLE BOOK",
           grade_B(seen(population=100, artifacts=_loan_table), t), WRONG_ANSWER)

    # C — two dimensions, a table carrying both, and reconciliation
    expect("C both", grade_C(seen(dimensions_applied=["a", "b"], artifacts=both,
                                  group_count=5, table_total=1000.0), t), CORRECT)
    expect("C one dropped", grade_C(seen(dimensions_applied=["a"], artifacts=both,
                                         group_count=5, table_total=1000.0), t),
           WRONG_ANSWER)
    expect("C short table", grade_C(seen(dimensions_applied=["a", "b"],
                                         artifacts=both, group_count=5,
                                         table_total=900.0), t), WRONG_ANSWER)
    expect("C refusal", grade_C(seen(ok=False), t), UNHELPFUL_REFUSAL)

    # The figure readers
    if not states("£1.96bn", 1_960_000_000.0):
        failures.append("the prose reader cannot read a scaled figure")
    if states("£1.90bn", 1_960_000_000.0):
        failures.append("the prose reader accepts a figure 3% away")

    for line in failures:
        print("SELF-TEST FAIL: %s" % line)
    print("SELF-TEST %s" % ("PASS" if not failures else "FAIL"))
    return 1 if failures else 0


def ask(paths_questions: List[Tuple[str, str]]) -> int:
    """Run questions and PRINT what came back. No grading, deliberately.

    A grade needs a declared expectation; these questions have none. Reporting
    them as correct or wrong would be inventing a verdict, which is the defect
    this whole programme has been closing. What is printed is what the service
    did: the route, the measure, the filters and dimensions it applied, the
    population it covered, and the answer a reader would see.
    """
    src = describe_source()
    print("=" * 78)
    print("BOOK IN USE — check this before trusting any answer below")
    print("=" * 78)
    print("  kind    : %s" % src.get("kind"))
    print("  label   : %s" % src.get("label"))
    print("  path    : %s" % src.get("path"))
    print("  rows    : %s        columns: %s" % (src.get("rows"), src.get("columns")))
    if src.get("kind") == "synthetic_demo":
        print("  *** THIS IS THE BUNDLED SYNTHETIC DEMO BOOK, NOT YOUR FILE. ***")
        print("  *** A misconfigured path does not fail; it falls back here. ***")
    print("  parser  : %s" % ("LLM enabled" if _LLM else "deterministic only"))
    print()
    a = _client()
    for cid, question in paths_questions:
        seen = observe(a, question)
        print("-" * 78)
        print("%s  %s" % (cid, question))
        print("   ok=%s  route=%s  verdict=%s" % (seen["ok"], seen["route"], seen["verdict"]))
        print("   measure=%s (%s)  population=%s  groups=%s"
              % (seen["measure"], seen["aggregation"], seen["population"], seen["group_count"]))
        print("   filters=%s  dimensions=%s" % (seen["filters_applied"] or [], seen["dimensions_applied"] or []))
        for line in (seen["answer"] or "(empty)").splitlines():
            print("   | %s" % line)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH", help="write the full record")
    ap.add_argument("--wrong-answers", action="store_true",
                    help="print only the wrong answers, in full")
    ap.add_argument("--self-test", action="store_true",
                    help="probe the graders, not the product")
    ap.add_argument("--questions", metavar="FILE",
                    help="run YOUR questions (one per line, # comments ok) "
                         "instead of the graded bank")
    ap.add_argument("--csv", metavar="PATH",
                    help="a canonical_typed.csv to query; omit for the demo book")
    ap.add_argument("--llm", choices=("on", "off"), default="off",
                    help="the LLM parser (default off). `on` needs ANTHROPIC_API_KEY")
    ap.add_argument("--verify", action="store_true",
                    help="print the book actually loaded and exit")
    args = ap.parse_args(argv)

    global _CSV, _LLM
    _CSV, _LLM = args.csv, (args.llm == "on")

    if args.verify:
        src = describe_source()
        for k in ("kind", "label", "path", "rows", "columns"):
            print("%-8s %s" % (k, src.get(k)))
        if src.get("kind") == "synthetic_demo" and args.csv:
            print("\nWARNING: --csv was given but the SYNTHETIC DEMO book loaded.")
            return 1
        return 0

    if args.self_test:
        return self_test()
    if args.questions:
        return ask(questions_from(args.questions))

    result = run()
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=str),
                                   encoding="utf-8")
    report(result, wrong_in_full=True)
    if args.json:
        print("\nfull record -> %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
