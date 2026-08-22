#!/usr/bin/env python3
"""question_interpretation/mi_gate_llm_arm.py

THE GATE — the LLM arm over the 29 time-series phrasings.

A MEASUREMENT, NOT A STAGE. This instrument changes no product code. It runs the
eight time-series shapes declared in ``time_series_surface.SHAPES`` with the LLM
parser ON, against the deterministic arm as the baseline, and reports what
diverged and what each divergence MEANS.

Three things this instrument refuses to do, each because an earlier run did it
and produced a number that had to be thrown away:

1. **It never reads ``parser_mode`` (snake_case).** The metadata key is
   ``parserMode``. An earlier comparison read the snake form, got ``None`` on
   both arms, and reported "15 of 15 identical" with no evidence the model had
   run at all. That run was void.

2. **It never infers that the model ran from a field value.** It COUNTS CALLS,
   by wrapping ``mi_agent.llm_query_parser._invoke``, and reports the count.
   A zero count is a real result, not a bug: ``parse_with_repair`` runs the
   deterministic parser FIRST and ``zero_cost_first`` hands off to the model
   only on validation failure, non-``high`` confidence, or a layered question.

3. **It never rates from the receipt.** Ratings come from the RENDERED ARTIFACT
   via ``time_series_surface.inspect_artifacts`` / ``rate``. ``dimensionsApplied``
   proves nothing — a truthful receipt silent about time is exactly the case P0
   exists for.

One book per process: the two books resolve different datasets through
module-level caches, so mixing them in one process risks serving one book's
frame for the other's question.

    python -m question_interpretation.mi_gate_llm_arm --book alderbridge --repeats 5
    python -m question_interpretation.mi_gate_llm_arm --book kestrelmoor --repeats 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from question_interpretation.time_series_surface import (  # noqa: E402
    ABSENT, PARTIAL, PROVEN, SHAPES, inspect_artifacts, rate, shape_rating)

#: The two ends of a movement. THE BRIEF'S RATING RULE, IN FULL: a time axis is
#: proven by a time column with more than one distinct value **or by a column
#: pair naming the two ends of a movement**. `time_series_surface._TIME_HINTS`
#: implements only the first form, so the movement-shaped artifacts (T7's
#: rank/start_value/end_value table, T8's prior/current table) are recorded by
#: it as carrying no time axis. That under-credits them on BOTH arms equally, so
#: it cannot manufacture a divergence — but it does understate the absolute
#: rating, and the brief names these pairs explicitly.
_MOVEMENT_ENDS: Tuple[Tuple[str, str], ...] = (
    ("prior", "current"), ("start", "end"),
    ("opening", "closing"), ("previous", "latest"),
)


def movement_pair(columns: List[str]) -> Optional[Tuple[str, str]]:
    """The column pair naming the two ends of a movement, if the artifact has one."""
    low = [(c, c.lower()) for c in columns]
    for a, b in _MOVEMENT_ENDS:
        ca = next((c for c, l in low if a in l), None)
        cb = next((c for c, l in low if b in l), None)
        if ca and cb and ca != cb:
            return (ca, cb)
    return None


def column_census(artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Distinct count and sample values for EVERY column of the richest artifact.

    Recorded so the report can state what the artifact actually carries, rather
    than only what a fragment match happened to find. A breakdown delivered under
    a generic column name (`category`, `population`) is invisible to fragment
    matching and must not be silently scored as absent OR silently credited.
    """
    best: Dict[str, Any] = {"artifact": None, "rows": 0, "columns": {},
                            "movement_pair": None}
    for art in artifacts:
        rows = art.get("rows") or []
        if not rows or len(rows) < best["rows"]:
            continue
        cols = list(rows[0].keys())
        census = {}
        for c in cols:
            vals = [r.get(c) for r in rows if r.get(c) is not None]
            distinct = sorted({str(v) for v in vals})
            census[c] = {"distinct": len(distinct), "sample": distinct[:6]}
        best = {"artifact": art.get("type"), "rows": len(rows),
                "columns": census, "movement_pair": movement_pair(cols)}
    return best


def rate_brief(ok: bool, want_dims: List[str], census: Dict[str, Any],
               surface_seen: Dict[str, Any]) -> Tuple[str, str]:
    """The brief's rule applied in full, reported ALONGSIDE the surface's rating.

    Time axis: a time column with >1 distinct value OR a movement column pair.
    Breakdown: a requested-dimension column with >1 distinct value. A generically
    named column is NOT auto-credited as the breakdown — it is reported with its
    values so a reader can see what is there.
    """
    if not ok:
        return ABSENT, "refused"
    has_time = surface_seen["time_points"] > 1 or census["movement_pair"] is not None
    via = ("%d time points in %s" % (surface_seen["time_points"], surface_seen["time_col"])
           if surface_seen["time_points"] > 1
           else "movement pair %s" % (census["movement_pair"],))
    real_dims = {k: v for k, v in surface_seen["dims"].items() if v > 1}
    if not want_dims:
        return (PROVEN, via) if has_time else (ABSENT, "no time axis in the artifact")
    if has_time and real_dims:
        return PROVEN, "%s x %s" % (via, real_dims)
    if has_time and not real_dims:
        generic = {c: m for c, m in census["columns"].items()
                   if m["distinct"] > 1 and c not in (surface_seen["time_col"],)}
        return ABSENT, ("time axis present (%s) but no REQUESTED dimension column; "
                        "columns carrying >1 value: %s" % (via, generic or "none"))
    return ABSENT, "no time axis in the artifact"


#: What a divergence MEANS. The brief asks for exactly this attribution, and for
#: anything that fits neither box to be NAMED rather than forced into one.
SAME = "same outcome"
PROPOSAL_NOT_REACHED = "the model proposed something the route does not reach"
GUARD_REFUSED_PROPOSAL = "the guard refused a proposal the route would have handled"
UNCLASSIFIED = "NEITHER CATEGORY FITS"

#: The three P0 refusals the brief names, with the limb each one protects.
P0_REFUSALS: Tuple[Tuple[str, str], ...] = (
    ("Show me balance by month by region and LTV band", "time axis"),
    ("balance by month broken down by LTV band and region", "time axis"),
    ("How have direct and acquired balances moved over the periods?", "segments"),
)


def phrasings() -> List[Tuple[str, str, str, List[str]]]:
    """(shape id, shape name, question, wanted dimension fragments) — from SHAPES.

    Read from the declaration, never retyped.
    """
    return [(sid, name, q, want)
            for sid, name, qs, want in SHAPES for q in qs]


# --------------------------------------------------------------------------- #
# Did the model actually run? COUNTED, never inferred.
# --------------------------------------------------------------------------- #
class InvocationCounter:
    """Counts real calls into ``llm_query_parser._invoke``.

    A field value is not evidence a call happened; this is. ``install`` is
    idempotent so repeated arm switches cannot stack wrappers and double-count.
    """

    def __init__(self) -> None:
        self.n = 0
        self._original: Optional[Callable] = None

    def install(self) -> None:
        from mi_agent import llm_query_parser as P
        if self._original is not None:
            return
        self._original = P._invoke
        counter = self

        def counting(prompt, model, llm_callable, use_cache=True):
            counter.n += 1
            return counter._original(prompt, model, llm_callable,
                                     use_cache=use_cache)

        P._invoke = counting

    def uninstall(self) -> None:
        if self._original is None:
            return
        from mi_agent import llm_query_parser as P
        P._invoke = self._original
        self._original = None

    def reset(self) -> None:
        self.n = 0


# --------------------------------------------------------------------------- #
# The two arms
# --------------------------------------------------------------------------- #
def _book_env(book: str) -> str:
    """Point the process at one book. Returns its client id."""
    if book == "alderbridge":
        from demo_platform import config as cfg
        os.environ.update(cfg.mi_env(period_role="current"))
        return cfg.CLIENT_ID
    if book == "kestrelmoor":
        from question_interpretation.run_robustness_deterministic import (
            _KESTRELMOOR_ROOT)
        from tests.analytical import second_book as bk
        if not _KESTRELMOOR_ROOT.exists():
            bk.build(_KESTRELMOOR_ROOT)
        os.environ.update(bk.mi_env(_KESTRELMOOR_ROOT))
        return os.environ["MI_AGENT_CLIENT_ID"]
    raise ValueError("unknown book %r" % (book,))


def _set_arm(llm: bool) -> None:
    """Select the arm. ``_mi_llm_config`` reads these per request, so an arm
    switch inside one process takes effect on the next question."""
    os.environ["MI_AGENT_LLM_PARSER"] = "on" if llm else "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "1" if llm else "0"


def _asker(client_id: str):
    import logging
    logging.disable(logging.WARNING)
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext
    ctx = ExecutionContext.for_internal(client_id)
    return lambda q: (execute_governed_mi_query(
        MiQueryRequest(question=q), ctx).result or {})


def llm_arm_available() -> Tuple[bool, str]:
    """Whether the LLM arm can run at all. Reported, never assumed."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, ("ANTHROPIC_API_KEY is not set in the environment; the LLM "
                       "arm cannot run and no result is reported for it")
    try:
        import anthropic  # noqa: F401
    except Exception:  # noqa: BLE001
        return False, "the anthropic package is not installed"
    return True, "available"


def observe(ask, question: str, want: List[str],
            counter: InvocationCounter) -> Dict[str, Any]:
    """One run of one question. Rated from the ARTIFACT; model calls COUNTED."""
    # Delta, not reset: the counter stays cumulative so `total_model_calls`
    # is the run total. Resetting here made that field report the LAST
    # question's count (1) while the true total was 130.
    before = counter.n
    res = ask(question)
    calls = counter.n - before
    meta = res.get("metadata") or {}
    summary = res.get("executionSummary") or {}
    guard = meta.get("semantic_guard") or {}
    arts = res.get("artifacts") or []
    seen = inspect_artifacts(arts, want)
    census = column_census(arts)
    claimed = list(summary.get("dimensionsApplied") or ())
    verdict, why = rate(bool(res.get("ok")), want, seen, claimed)
    facets = guard.get("facets") or []
    disclosed = bool(facets or summary.get("notApplied")
                     or guard.get("verdict") in ("partial", "clarify", "refuse"))
    silent = (verdict == ABSENT and bool(res.get("ok")) and not disclosed
              and ("TIME AXIS DROPPED" in why or "WHOLE-BOOK SERIES" in why))
    if silent:
        why += "  ** SILENTLY — ok=True, no verdict, no facet, no note **"
    return {
        "question": question, "ok": bool(res.get("ok")),
        "route": meta.get("route"), "guard_verdict": guard.get("verdict"),
        # camelCase. The snake form is why an earlier run was void.
        "parser_mode": meta.get("parserMode"),
        "parser_mode_detail": meta.get("parserModeDetail"),
        "model_calls": calls,
        "claimed_dimensions": claimed,
        "claimed_filters": list(summary.get("filtersApplied") or ()),
        "measure": summary.get("measure"), "population": summary.get("population"),
        "artifact": seen, "rating": verdict, "why": why,
        "census": census,
        "rating_brief": rate_brief(bool(res.get("ok")), want, census, seen)[0],
        "why_brief": rate_brief(bool(res.get("ok")), want, census, seen)[1],
        "silent_drop": silent, "disclosed": disclosed,
        "answer": str(res.get("answer") or res.get("error") or "")[:240],
    }


def outcome_key(o: Dict[str, Any]) -> str:
    """What makes two runs THE SAME OUTCOME.

    Artifact-derived, because that is what the rating is derived from. Two runs
    that differ only in prose are the same outcome; two that differ in whether a
    time axis or a real breakdown reached the artifact are not.
    """
    art = o["artifact"]
    return json.dumps({
        "ok": o["ok"], "rating": o["rating"], "route": o["route"],
        "guard_verdict": o["guard_verdict"],
        "time_points": art["time_points"],
        "real_dims": {k: v for k, v in art["dims"].items() if v > 1},
        "silent_drop": o["silent_drop"],
    }, sort_keys=True, default=str)


def attribute(det: Dict[str, Any], llm: Dict[str, Any]) -> str:
    """Which KIND of divergence this is — or that neither kind fits."""
    if outcome_key(det) == outcome_key(llm):
        return SAME
    if det["ok"] and not llm["ok"]:
        return GUARD_REFUSED_PROPOSAL
    if llm["ok"] and not det["ok"]:
        # The LLM arm answered where the deterministic arm refused. That is
        # neither of the brief's two boxes, and is said so rather than forced.
        return UNCLASSIFIED
    if not det["ok"] and not llm["ok"]:
        # Both refused, by different routes or guards. Also neither box.
        return UNCLASSIFIED
    return PROPOSAL_NOT_REACHED


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run(book: str, repeats: int) -> Dict[str, Any]:
    client_id = _book_env(book)
    counter = InvocationCounter()
    counter.install()

    available, why_unavailable = llm_arm_available()

    _set_arm(False)
    ask_det = _asker(client_id)
    det: Dict[str, Dict[str, Any]] = {}
    for _sid, _name, q, want in phrasings():
        det[q] = observe(ask_det, q, want, counter)

    if not available:
        return {"book": book, "repeats": repeats, "llm_available": False,
                "reason": why_unavailable, "rows": [],
                "deterministic": det}

    _set_arm(True)
    ask_llm = _asker(client_id)
    rows: List[Dict[str, Any]] = []
    for sid, name, q, want in phrasings():
        runs = [observe(ask_llm, q, want, counter) for _ in range(repeats)]
        counts = Counter(outcome_key(r) for r in runs)
        modal_key, modal_n = counts.most_common(1)[0]
        modal = next(r for r in runs if outcome_key(r) == modal_key)
        rows.append({
            "shape": sid, "shape_name": name, "question": q,
            "deterministic": det[q], "llm_modal": modal, "llm_runs": runs,
            "distinct_outcomes": len(counts),
            "modal_share": "%d/%d" % (modal_n, repeats),
            "stable": len(counts) == 1,
            "model_calls": [r["model_calls"] for r in runs],
            "parser_modes": [r["parser_mode"] for r in runs],
            "attribution": attribute(det[q], modal),
        })
    return {"book": book, "repeats": repeats, "llm_available": True,
            "rows": rows, "deterministic": det,
            "total_model_calls": counter.n}


def p0_status(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Do the three named P0 refusals hold under the LLM arm?

    A refusal HOLDS when no repeat silently dropped the limb it protects. A run
    that refuses holds it; a run that discloses the loss holds it; a run with
    ok=True and nothing disclosed does not.
    """
    out = []
    for question, limb in P0_REFUSALS:
        row = next((r for r in result["rows"] if r["question"] == question), None)
        if row is None:
            out.append({"question": question, "limb": limb,
                        "holds": None, "note": "not present in this run"})
            continue
        runs = row["llm_runs"]
        silent = [r for r in runs if r["silent_drop"]]
        refused = [r for r in runs if not r["ok"]]
        out.append({
            "question": question, "limb": limb,
            "holds": not silent,
            "refused": "%d/%d" % (len(refused), len(runs)),
            "silent_drops": len(silent),
            "distinct_outcomes": row["distinct_outcomes"],
            "note": ("held — no silent drop in any repeat" if not silent
                     else "BREACHED — %d of %d repeats dropped the %s silently"
                          % (len(silent), len(runs), limb)),
        })
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report(result: Dict[str, Any]) -> int:
    book = result["book"]
    print("=" * 94)
    print("THE GATE — LLM arm over the 29 time-series phrasings — book: %s" % book)
    print("=" * 94)

    if not result["llm_available"]:
        print()
        print("LLM ARM: NOT RUN")
        print("  %s." % result["reason"])
        print("  No LLM outcome is reported and none is estimated.")
        return 2

    rows = result["rows"]

    # DID THE ARM OBSERVE ANYTHING? Asserted before any comparison is quoted.
    total_calls = sum(sum(r["model_calls"]) for r in rows)
    reached = [r for r in rows if sum(r["model_calls"]) > 0]
    print()
    print("DID THE MODEL RUN?  (counted at `_invoke`, not inferred)")
    print("-" * 94)
    print("  total counted model calls        : %d" % total_calls)
    print("  questions that reached the model : %d of %d" % (len(reached), len(rows)))
    modes = Counter(m for r in rows for m in r["parser_modes"])
    print("  parserMode across all LLM runs   : %s" % dict(modes))
    if total_calls == 0:
        print()
        print("  *** THE LLM ARM MADE NO CALLS — every comparison below is void.")
        return 3
    print()
    print("  A zero count on a question is a REAL RESULT: zero_cost_first hands")
    print("  off only on validation failure, non-high confidence, or a layered")
    print("  question. Questions with zero calls are listed per shape below.")

    print()
    print("PER SHAPE  (repeats: %d)" % result["repeats"])
    print("=" * 94)
    for sid, name, qs, _want in SHAPES:
        mine = [r for r in rows if r["shape"] == sid]
        det_rating = shape_rating([r["deterministic"]["rating"] for r in mine])
        llm_rating = shape_rating([r["llm_modal"]["rating"] for r in mine])
        print()
        print("%-3s %-44s  det=%-8s llm=%-8s" % (sid, name, det_rating, llm_rating))
        print("-" * 94)
        for r in mine:
            m = r["llm_modal"]
            print("  %-60s" % r["question"][:60])
            print("    det: %-8s ok=%-5s route=%s" % (r["deterministic"]["rating"],
                  r["deterministic"]["ok"], r["deterministic"]["route"]))
            print("    llm: %-8s ok=%-5s route=%s  modal=%s distinct=%d calls=%s mode=%s"
                  % (m["rating"], m["ok"], m["route"], r["modal_share"],
                     r["distinct_outcomes"], r["model_calls"], r["parser_modes"][0]))
            print("    why: %s" % m["why"])
            if r["attribution"] != SAME:
                print("    DIVERGENCE: %s" % r["attribution"])
                print("      det: measure=%s pop=%s dims=%s time=%s"
                      % (r["deterministic"]["measure"], r["deterministic"]["population"],
                         r["deterministic"]["claimed_dimensions"],
                         r["deterministic"]["artifact"]["time_points"]))
                print("      llm: measure=%s pop=%s dims=%s time=%s"
                      % (m["measure"], m["population"], m["claimed_dimensions"],
                         m["artifact"]["time_points"]))

    print()
    print("=" * 94)
    print("SUMMARY — %s" % book)
    print("=" * 94)
    for sid, name, _qs, _w in SHAPES:
        mine = [r for r in rows if r["shape"] == sid]
        print("  %-3s %-44s det=%-8s llm=%-8s" % (
            sid, name,
            shape_rating([r["deterministic"]["rating"] for r in mine]),
            shape_rating([r["llm_modal"]["rating"] for r in mine])))

    tally = Counter(r["attribution"] for r in rows)
    print()
    print("  ATTRIBUTION")
    for k in (SAME, PROPOSAL_NOT_REACHED, GUARD_REFUSED_PROPOSAL, UNCLASSIFIED):
        print("    %-58s %d" % (k, tally.get(k, 0)))

    unstable = [r for r in rows if not r["stable"]]
    print()
    print("  SELF-DISAGREEMENT across %d repeats: %d of %d questions"
          % (result["repeats"], len(unstable), len(rows)))
    for r in unstable:
        print("     %-4s %-56s %d outcomes (modal %s)"
              % (r["shape"], r["question"][:56], r["distinct_outcomes"],
                 r["modal_share"]))

    silent = [r for r in rows if any(x["silent_drop"] for x in r["llm_runs"])]
    print()
    print("  SILENT DROPS under the LLM arm (any repeat): %d" % len(silent))
    for r in silent:
        n = sum(1 for x in r["llm_runs"] if x["silent_drop"])
        print("     %-4s %-56s %d/%d repeats" % (r["shape"], r["question"][:56],
                                                 n, result["repeats"]))

    refused_runs = sum(1 for r in rows for x in r["llm_runs"] if not x["ok"])
    total_runs = sum(len(r["llm_runs"]) for r in rows)
    print()
    print("  HONEST REFUSALS under the LLM arm: %d of %d runs"
          % (refused_runs, total_runs))

    print()
    print("  THE THREE P0 REFUSALS")
    print("  " + "-" * 90)
    for p in p0_status(result):
        print("    %-58s [%s]" % (p["question"][:58], p["limb"]))
        print("      %s  (refused %s, distinct outcomes %s)"
              % (p["note"], p.get("refused"), p.get("distinct_outcomes")))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", default="alderbridge",
                    choices=("alderbridge", "kestrelmoor"))
    ap.add_argument("--repeats", type=int, default=5,
                    help="LLM runs per question; the arm disagrees with itself")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args(argv)
    result = run(args.book, args.repeats)
    if args.json:
        Path(args.json).write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8")
    return report(result)


if __name__ == "__main__":
    raise SystemExit(main())
