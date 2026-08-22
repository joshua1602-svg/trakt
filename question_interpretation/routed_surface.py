#!/usr/bin/env python3
"""question_interpretation/routed_surface.py

THE THIRD MEASUREMENT SURFACE — the governed service path, routing as shipped.

Why it had to exist
-------------------
Three stages ended with "both standing surfaces are unmoved", and each time the
reason was the same: neither bank exercises the path the change was on.

* The CALIBRATION BANK calls `run_mi_agent_query` directly, bypassing routing
  entirely (backlog B7). It is always point-in-time. Four of the six Stage 5
  trace questions behave differently there from production.
* The ROBUSTNESS BANK does go through `/mi/query` and therefore the real
  routing, but its runner imports `nl_bank` as a frozen artefact, and it grades
  an outcome LABEL — not which route answered, and not what the receipt claimed.

So a defect could sit on the routed path, in front of ordinary questions, and
both surfaces would read clean. One did, for a full commit (`e35a01b`), and a
worse one did for longer: `evolution` returned the whole-book series for
"balance by month by region" while the receipt stamped the breakdown APPLIED.

This drives `execute_governed_mi_query` and asserts STRUCTURE — which route
claimed the question, what verdict the guard reached, and which facet kinds and
statuses it produced. Not answer text: `answer_diff` owns that byte for byte,
and duplicating it here would mean a reworded refusal failed two surfaces.

What it can and cannot see is declared in the bank itself, under
`meta.sees` and `meta.cannot_see`, and `--limits` prints it. That is not
decoration. The failure of the other two banks was never that they were wrong;
it was that their blindness went unwritten until a defect walked through it.

    python -m question_interpretation.routed_surface
    python -m question_interpretation.routed_surface --limits
    python -m question_interpretation.routed_surface --self-test
"""
from __future__ import annotations

import argparse
import json
import os
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

BANK = (_REPO_ROOT / "config" / "mi" / "golden_questions" / "routed_surface.yaml")


def bank() -> Dict[str, Any]:
    return yaml.safe_load(BANK.read_text(encoding="utf-8"))


def _client():
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    import logging
    logging.disable(logging.WARNING)

    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    return lambda q, filters=None: (execute_governed_mi_query(
        MiQueryRequest(question=q, filters=filters or None), ctx).result or {})


def observe(ask, question: str,
            filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Drive one case. ``filters`` is the DRILL-THROUGH API — a UI drill from a
    breakdown into one of its groups — and a case that supplies them is
    exercising a path no question text can reach."""
    res = ask(question, filters)
    meta = res.get("metadata") or {}
    guard = meta.get("semantic_guard") or res.get("semanticGuard") or {}
    # THE POPULATION, added for B22 because this surface could not see it.
    #
    # rt_023, rt_025 and rt_027 were declared expected-to-fail against a live
    # defect — the lens silently narrowing to 3,909 of 11,035 loans — and this
    # surface reported all three FIXED, because it asserted the route, the
    # verdict and the facet kinds and none of those move when a population is
    # silently narrowed. A surface reporting a live defect as closed is worse
    # than one that is silent about it.
    summary = res.get("executionSummary") or {}
    return {
        "route": meta.get("route"),
        # THE VIEW, added for B21 because nothing in the estate asserted it.
        #
        # `resolve_active_view` decides which FRAME the question is answered
        # against, before parsing. Every `datasetContext` anywhere else — in the
        # differ, the robustness runner, the perf harness — is a request
        # PARAMETER being sent, never an assertion on what came back. So a wrong
        # view was unfalsifiable: on this book a forecast question answered on
        # the funded frame returns the identical number, and a pipeline question
        # answered on the wrong frame returns the identical refusal.
        "view": meta.get("datasetContext"),
        "ok": bool(res.get("ok")),
        "verdict": guard.get("verdict"),
        "facets": sorted((str(f.get("kind")), str(f.get("status")))
                         for f in (guard.get("facets") or [])),
        "answer": str(res.get("answer") or res.get("error") or ""),
        "population": summary.get("population"),
        "filters": sorted(str(f) for f in (summary.get("filtersApplied") or ())),
    }


def check(case: Dict[str, Any], seen: Dict[str, Any]) -> List[str]:
    """Every expectation the case states, and only those it states.

    A case that omits `expect_facets` asserts nothing about facets — silence is
    not an assertion of emptiness. Pinning more than a case means to pin is how
    a surface becomes brittle and then gets loosened until it means nothing.
    """
    fails: List[str] = []
    if "expect_route" in case and case["expect_route"] != seen["route"]:
        fails.append("route %r != expected %r" % (seen["route"], case["expect_route"]))
    if "expect_ok" in case and bool(case["expect_ok"]) != seen["ok"]:
        fails.append("ok %r != expected %r" % (seen["ok"], case["expect_ok"]))
    if "expect_verdict" in case and case["expect_verdict"] != seen["verdict"]:
        fails.append("verdict %r != expected %r"
                     % (seen["verdict"], case["expect_verdict"]))
    if case.get("expect_facets") is not None:
        want = sorted((str(k), str(v)) for k, v in case["expect_facets"])
        if want != seen["facets"]:
            fails.append("facets %s != expected %s" % (seen["facets"], want))
    if "expect_view" in case and case["expect_view"] != seen["view"]:
        fails.append("view %r != expected %r" % (seen["view"], case["expect_view"]))
    if "expect_population" in case and case["expect_population"] != seen["population"]:
        fails.append("population %r != expected %r"
                     % (seen["population"], case["expect_population"]))
    if case.get("expect_filters") is not None:
        want = sorted(str(f) for f in case["expect_filters"])
        if want != seen["filters"]:
            fails.append("filtersApplied %s != expected %s" % (seen["filters"], want))
    needle = case.get("expect_answer_contains")
    if needle and str(needle).lower() not in seen["answer"].lower():
        fails.append("answer does not contain %r" % needle)
    return fails


def run() -> Dict[str, Any]:
    doc = bank()
    ask = _client()
    rows: List[Dict[str, Any]] = []
    for case in doc["questions"]:
        seen = observe(ask, case["question"], case.get("filters"))
        fails = check(case, seen)
        rows.append({
            "id": case["id"], "question": case["question"],
            "expected_to_fail": bool(case.get("expected_to_fail")),
            "will_change_when": case.get("will_change_when"),
            "failures": fails, "observed": seen,
        })
    return {"meta": doc["meta"], "rows": rows}


def verdict(rows: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """``(passed, failed, expected_failures_that_now_pass)``.

    A case declared expected-to-fail that PASSES is reported separately and is
    not silently counted as a win: it means the named defect was closed, and the
    case must be re-declared rather than left claiming a defect that is gone.
    """
    passed = failed = fixed = 0
    for row in rows:
        if row["expected_to_fail"]:
            if row["failures"]:
                passed += 1          # failing as declared
            else:
                fixed += 1
        elif row["failures"]:
            failed += 1
        else:
            passed += 1
    return passed, failed, fixed


def self_test() -> int:
    """The surface must be able to fail.

    Runs every case against a deliberately wrong expectation and requires each
    to be caught. A bank whose checker silently accepts anything reports a clean
    sweep forever, which is the shape of half the instrument defects in this
    programme.
    """
    doc = bank()
    failures: List[str] = []
    seen = {"route": "evolution", "ok": True, "verdict": "ok",
            "facets": [("granularity", "applied")], "answer": "a sentence",
            "population": 11035, "filters": [], "view": "funded"}
    probes = [
        ({"expect_route": "not_a_route"}, "route"),
        ({"expect_ok": False}, "ok"),
        ({"expect_verdict": "refuse"}, "verdict"),
        ({"expect_facets": [["grouping_dimension", "lost"]]}, "facets"),
        ({"expect_answer_contains": "a phrase that is absent"}, "answer"),
        ({"expect_view": "a view that was not used"}, "view"),
        ({"expect_population": 999}, "population"),
        ({"expect_filters": ["a filter that was not applied"]}, "filters"),
    ]
    for case, what in probes:
        if not check(case, seen):
            failures.append("the checker did not catch a wrong %s" % what)
    # ...and must PASS a correct expectation, or it would "catch" everything.
    correct = {"expect_route": "evolution", "expect_ok": True,
               "expect_verdict": "ok",
               "expect_facets": [["granularity", "applied"]],
               "expect_answer_contains": "sentence",
               "expect_population": 11035, "expect_filters": [],
               "expect_view": "funded"}
    if check(correct, seen):
        failures.append("the checker rejected a correct expectation")
    # Silence is not an assertion.
    if check({}, seen):
        failures.append("a case stating no expectations was not silent")
    # The bank must declare its own blindness.
    for key in ("sees", "cannot_see"):
        if not doc["meta"].get(key):
            failures.append("the bank does not declare meta.%s" % key)
    if doc["meta"].get("count") != len(doc["questions"]):
        failures.append("meta.count disagrees with the question list")

    for f in failures:
        print("SELF-TEST FAIL: %s" % f)
    print("SELF-TEST %s" % ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


def print_limits(meta: Dict[str, Any]) -> None:
    print("=" * 78)
    print("ROUTED SURFACE — what it can and cannot see")
    print("=" * 78)
    print("entry point: %s\n" % meta.get("entry_point"))
    print("SEES")
    for item in meta.get("sees") or []:
        print("  * %s" % item)
    print("\nCANNOT SEE")
    for item in meta.get("cannot_see") or []:
        print("  * %s" % item)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--limits", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.limits:
        print_limits(bank()["meta"])
        return 0

    result = run()
    rows = result["rows"]
    print("=" * 78)
    print("ROUTED SURFACE — %s" % result["meta"]["name"])
    print("=" * 78)
    for row in rows:
        mark = ("xfail" if row["expected_to_fail"] and row["failures"]
                else "FIXED" if row["expected_to_fail"]
                else "ok" if not row["failures"] else "FAIL")
        print("%-5s %-8s %s" % (row["id"], mark, row["question"][:52]))
        print("        route=%-24s verdict=%-8s facets=%s"
              % (row["observed"]["route"], row["observed"]["verdict"],
                 row["observed"]["facets"]))
        for f in row["failures"]:
            print("        %s %s" % ("(as declared)" if row["expected_to_fail"]
                                     else "->", f))
    passed, failed, fixed = verdict(rows)
    print("\n" + "-" * 78)
    print("passed %d   failed %d   declared-defect now fixed %d"
          % (passed, failed, fixed))
    if fixed:
        print("\nA case declared expected-to-fail now passes. Re-declare it —")
        print("leaving it claiming a defect that is closed makes the surface lie:")
        for row in rows:
            if row["expected_to_fail"] and not row["failures"]:
                print("   %s  %s" % (row["id"], row["question"][:56]))
    print("\nRun with --limits before quoting this surface as coverage.")

    if args.json:
        args.json.write_text(json.dumps(result, indent=2, default=str),
                             encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
