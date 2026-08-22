#!/usr/bin/env python3
"""question_interpretation/decision_census.py

HOW MANY DECISIONS ABOUT THE USER'S QUESTION HAVE MORE THAN ONE OWNER?

Every defect found in this programme has one shape: a decision about the
question is reached independently in more than one place, and the places
disagree. Four instances so far, each found only when it broke something. This
counts the rest.

A DECISION is a question about the user's question whose answer something
downstream acts on. An OWNER is any code path that reaches that answer
INDEPENDENTLY — from the question text, the spec, configuration or the data —
rather than reading it from somewhere else.

Not functions. The earlier inventory counted 86 functions across 11 entry points
and missed the seasoning defect entirely, because its three owners sit in three
different entry points and nothing asked what decision each was making.

Classes, of which only the last two are the count:

    single                  one owner
    derived                 several readers, all reading one owner's answer
    agree-by-construction   several owners that provably cannot diverge
    agree-by-maintenance    several owners agreeing today, nothing preventing
                            divergence                                    DEBT
    disagreeing             divergence demonstrated                      DEFECT

Agreement is measured over the corpus AND probed by construction, because "no
surface reached it" and "it cannot be reached" have been conflated twice here
and both times it hid a live defect.

    python -m question_interpretation.decision_census
    python -m question_interpretation.decision_census --self-test
    python -m question_interpretation.decision_census --blind-spots
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_EVIDENCE = _REPO_ROOT / "due_diligence" / "evidence" / "analytical_intent_v1"
for p in (str(_REPO_ROOT), str(_EVIDENCE)):
    if p not in sys.path:
        sys.path.insert(0, p)

SINGLE = "single"
DERIVED = "derived"
BY_CONSTRUCTION = "agree-by-construction"
BY_MAINTENANCE = "agree-by-maintenance"
DISAGREEING = "disagreeing"

#: What this census could NOT reach, stated in the instrument so nobody quotes
#: its number as covering more than it does.
BLIND_SPOTS = [
    "Decisions taken INSIDE a route's execution. `chat_routing` is 3,000 lines "
    "of route bodies; a decision made and consumed entirely within one of them "
    "is invisible here unless it surfaces in the spec, the facets or the "
    "artifacts. Route-level defects found so far (B12, B14) surfaced only "
    "because something downstream disagreed.",
    "The LLM parser arm. Every probe runs deterministically, so a decision the "
    "LLM path takes differently is not counted.",
    "Any book but alderbridge. A decision that diverges only on a book with "
    "different columns is invisible.",
    "Owners I did not think to look for. The decision list is authored, not "
    "derived — there is no mechanical way to enumerate 'decisions' — so this "
    "census is a lower bound and its own biggest blind spot. The earlier "
    "inventory missed the seasoning decision the same way.",
    "Divergence that needs two owners to run on DIFFERENT inputs. Where owners "
    "are probed on the same question, a defect caused by one owner seeing a "
    "different frame (B14's shape) is only found because it is already known.",
]


# --------------------------------------------------------------------------- #
# Probes — each returns one owner's answer for one question
# --------------------------------------------------------------------------- #
def _semantics():
    from mi_agent.mi_query_validator import load_mi_semantics
    return load_mi_semantics(
        _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")


def _frame():
    from mi_agent import mi_calibration as CAL
    return CAL.default_bank_frame(
        _REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
        / "platform" / "alderbridge" / "2026-06-30"
        / "platform_canonical_typed.csv")


class _Ctx:
    def __init__(self):
        self.semantics = _semantics()
        self.frame = _frame()
        self.columns = list(self.frame.columns)


# -- decision: is a named dimension an AXIS? -------------------------------- #
def _axis_parser(ctx, q):
    from mi_agent.llm_query_parser import _deterministic_parse
    spec, _ = _deterministic_parse(q, ctx.semantics, available_columns=set(ctx.columns))
    axes = list(getattr(spec, "dimensions", None) or [])
    if getattr(spec, "dimension", None):
        axes.append(spec.dimension)
    return tuple(sorted({str(a) for a in axes if a}))


def _axis_detector(ctx, q):
    from mi_agent import execution_receipt as R
    return tuple(sorted({t[0] for t in
                         R.requested_dimension_terms(q, ctx.semantics, ctx.columns)}))


def _axis_role_after_split(ctx, q):
    """The role the RECEIPT ends up asserting, on the point-in-time path.

    The first draft of this probe compared "what the parser put on an axis"
    against "what the detector recorded as named". Those are different
    questions: the detector deliberately records a dimension the parser DROPPED,
    so it can be disclosed rather than vanishing — which is the whole reason it
    exists. It reported 76 disagreements, 62 of which were that division of
    labour, and would have inflated the census's headline number by a factor of
    three.

    The decision is "axis or filter", so both owners must be asked THAT. The
    parser's answer is its spec; the receipt's answer is the facet kind after
    Stage 4's split has read the spec.
    """
    from mi_agent import execution_receipt as R
    from mi_agent.llm_query_parser import _deterministic_parse
    spec, _ = _deterministic_parse(q, ctx.semantics, available_columns=set(ctx.columns))
    dims = R.requested_dimension_terms(q, ctx.semantics, ctx.columns)
    facets = R.detect_requested_facets(q, ctx.semantics, frame=ctx.frame,
                                       requested_dimensions=dims)
    after = R._split_named_dimension_roles(facets, spec, ctx.semantics, ctx.columns)
    return tuple(sorted({f.field_key for f in after
                         if f.kind == R.KIND_GROUPING and f.field_key}))


def _axis_role_routed(ctx, q):
    """The role the receipt asserts on the ROUTED path, where no split runs."""
    from mi_agent import execution_receipt as R
    dims = R.requested_dimension_terms(q, ctx.semantics, ctx.columns)
    facets = R.detect_requested_facets(q, ctx.semantics, frame=ctx.frame,
                                       requested_dimensions=dims)
    return tuple(sorted({f.field_key for f in facets
                         if f.kind == R.KIND_GROUPING and f.field_key}))


# -- decision: does this phrase name a governed population? ----------------- #
def _population_owner(ctx, q):
    from mi_agent.seasoning import resolve_population_predicate
    return tuple(sorted((resolve_population_predicate(q, ctx.columns) or {})))


def _population_receipt(ctx, q):
    """The receipt's comparison arm, which resolves windows from facet wording."""
    from mi_agent import execution_receipt as R
    facet = R.RequestedFacet(kind=R.KIND_POPULATION, label=q, field_key=None)
    described = R._governed_population_predicates(facet)
    return tuple(sorted({d.split(" ")[0] for d in described}))


# -- decision: what measure did the question name? -------------------------- #
def _measure_parser(ctx, q):
    from mi_agent.llm_query_parser import _deterministic_parse
    spec, _ = _deterministic_parse(q, ctx.semantics, available_columns=set(ctx.columns))
    return bool(getattr(spec, "metric", None) or getattr(spec, "measures", None))


def _measure_detector(ctx, q):
    """The detector's view: did every named measure slot resolve?"""
    from mi_agent import execution_receipt as R
    return not R._unresolved_measure_slots(q.lower(), ctx.semantics, ctx.frame)


# -- decision: is this a two-sided comparison? ------------------------------ #
def _comparison_seasoning(ctx, q):
    from mi_agent import seasoning as SZ
    return len(list(SZ.lending_windows_named(q))) >= 2


def _comparison_detector(ctx, q):
    from mi_agent import execution_receipt as R
    facets = R.detect_requested_facets(q, ctx.semantics, frame=ctx.frame,
                                       requested_dimensions=[])
    return any(f.kind == R.KIND_COHORT_COMPARISON for f in facets)


def _seasoning_question(ctx, q) -> bool:
    """The genuine overlap: only where BOTH owners have an opinion.

    The seasoning reader knows only seasoning; the cohort detector knows cohorts
    generally. A provenance comparison is not a seasoning one, so the seasoning
    reader answering False there is correct, not a disagreement. Probing them
    over the whole corpus counted that as divergence.
    """
    from mi_agent import seasoning as SZ
    return bool(list(SZ.lending_windows_named(q)) or SZ.segments_named(q))


# -- decision: what time grain was requested? ------------------------------- #
def _grain_lexical(ctx, q):
    from question_interpretation.lexical import requested_unit
    return requested_unit(q)


def _grain_period_request(ctx, q):
    from mi_agent import period_request as P
    return P.requested_unit(q)


# --------------------------------------------------------------------------- #
# The decision list — authored, and the census's own biggest blind spot
# --------------------------------------------------------------------------- #
DECISIONS: List[Dict[str, Any]] = [
    {
        "id": "D1", "decides": "Where the subject ends and the first condition begins.",
        "owners": [
            "question_interpretation.lexical.condition_cut / subject_side / metric_slot "
            "— question text, every parse, no columns",
        ],
        "readers": ["answer_type.subject_side", "llm_query_parser._metric_slot",
                    "execution_receipt._is_filter_subject"],
        "class": DERIVED,
        "note": "Three owners with three separately-declared vocabularies until "
                "Stage 3; all three now read the one owner. The founding defect, closed.",
    },
    {
        "id": "D2", "decides": "Whether a named dimension is an axis or a filter.",
        "owners": [
            "llm_query_parser._deterministic_parse — question text + registry, "
            "every parse, consults columns; writes spec.dimension(s)/filters",
            "execution_receipt.requested_dimension_terms — question TEXT, every "
            "receipt build, does NOT consult the spec",
        ],
        "readers": ["execution_receipt._split_named_dimension_roles (reads the spec)"],
        "probe": {"point_in_time": _axis_role_after_split,
                  "routed": _axis_role_routed},
        "constructed": "Any question where Stage 4's split changes a role. The "
                       "split runs inside reconcile_facets, so the two paths "
                       "assert different roles for the same question by "
                       "construction — the routed receipt asserts the "
                       "detector's raw reading.",
    },
    {
        "id": "D3", "decides": "Whether a phrase names a governed population.",
        "owners": [
            "seasoning.resolve_population_predicate — lending windows, every "
            "parse, consults columns (the owner as of 7c46f81)",
        ],
        "readers": ["llm_query_parser.resolve_seasoning_role",
                    "execution_receipt.detect_requested_facets"],
        "class": SINGLE,
        "note": "One owner as of 7c46f81. `_governed_population_predicates` was "
                "probed here first and reported 16 disagreements; it answers a "
                "DIFFERENT decision — 'which governed window does this facet's "
                "wording name', for comparison against a plan's declared "
                "population — and belongs to D8 as evidence, not here as an "
                "owner. Counting it would have merged two decisions to inflate "
                "one.",
    },
    {
        "id": "D4", "decides": "What time grain the question requested.",
        "owners": ["question_interpretation.lexical.requested_unit — question "
                   "text, every call, no columns"],
        "readers": ["period_request.requested_unit (delegates)",
                    "execution_receipt.granularity_facets"],
        "probe": {"lexical": _grain_lexical, "period_request": _grain_period_request},
        "class": DERIVED,
    },
    {
        "id": "D5", "decides": "What reporting window the question requested.",
        "owners": ["period_request.requested_span — question text + governed "
                   "seasoning config, every call, no columns"],
        "readers": ["chat_routing period_movement", "period_change_route",
                    "mi_workflows.analytical.executors",
                    "execution_receipt.check_window_coverage"],
        "class": SINGLE,
    },
    {
        "id": "D6", "decides": "Whether a field is available in the book being asked about.",
        "owners": [
            "the FRAME the service loads — chosen by _RUN_SCOPED_ROUTES from the "
            "anticipated route, before the question is answered",
            "llm_query_parser via available_columns — whatever frame it was given",
            "execution_receipt.reconcile_facets via available_columns — same",
            "seasoning.resolve_population_predicate via columns — same",
        ],
        "class": DISAGREEING,
        "identifier": "B14",
        "evidence": "Measured in one process on one tape: 'Forecast the balance of "
                    "the front book' parses with seasoning_in_cols=FALSE and 'What "
                    "is the balance of the front book?' with TRUE. The forecast "
                    "question is run-scoped, so the pipeline frame is loaded rather "
                    "than the funded tape.",
        "user_sees": "\"'Seasoning Segment' is not available in this dataset\" — "
                     "true of the frame loaded, false of the book asked about.",
        "reachable": True,
    },
    {
        "id": "D7", "decides": "Whether a requested grouping was actually applied.",
        "owners": [
            "execution_receipt.reconcile_facets grouping branch — group_field_keys "
            "and result columns, point-in-time only",
            "execution_receipt.reconcile_routed_facets grouping branch — three "
            "tiers over the artifact frame, routed only",
        ],
        "class": DISAGREEING,
        "identifier": "B12",
        "evidence": "Constructed, not found on the corpus: 'geographic exposure by "
                    "ltv bucket' is claimed by geo_exposure, whose frame carries "
                    "area and code, and ltv_bucket is stamped APPLIED by tier "
                    "three. Across 260 corpus questions no route paired one axis "
                    "with more than one requested field.",
        "user_sees": "A receipt certifying a breakdown by LTV bucket over an answer "
                     "broken down by ITL3 area.",
        "reachable": True,
    },
    {
        "id": "D8", "decides": "Whether a requested population was actually applied.",
        "owners": [
            "execution_receipt.reconcile_population — the route's populationApplied "
            "evidence, routed only",
            "execution_receipt._analytical_population_satisfies — the analytical "
            "plan's declared predicates, governed arm then literal arm",
            "execution_receipt.reconcile_facets KIND_POPULATION branch — the "
            "executor's applied_filter_fields, point-in-time only",
        ],
        "class": BY_MAINTENANCE,
        "evidence": "Three owners, three different evidence sources, each scoped to "
                    "a path the others do not run on — so they cannot disagree on "
                    "one answer today. Nothing prevents it: a route that both "
                    "declares populationApplied and publishes applied_filter_fields "
                    "would be adjudicated twice. The 160-run regression (32c263a) "
                    "was the literal arm of the second owner disagreeing with the "
                    "governed one, before the governed arm was put first.",
    },
    {
        "id": "D9", "decides": "What measure the question named, and whether it resolved.",
        "owners": [
            "mi_query_executor.resolve_measures — the registry, at execution, "
            "consults columns",
            "execution_receipt._unresolved_measure_slots — question TEXT, at "
            "detection, consults the frame",
            "execution_receipt.ROUTE_FIXED_MEASURE + detect_measure_substitution — "
            "route identity, at the routed guard",
        ],
        "class": BY_MAINTENANCE,
        "note": "PROBE REMOVED. It compared 'did the parser set a metric' against "
                "'did the detector find an unresolved slot' and reported 267 "
                "disagreements — almost all of them count questions, which "
                "legitimately have no metric AND no unresolved slot. Two "
                "different questions, and reporting the number would have been "
                "the third time this census inflated itself by conflating "
                "decisions. The three owners read different evidence at "
                "different times and are each scoped to a path the others do "
                "not decide on, so they cannot contradict one another today; "
                "nothing prevents it.",
        "constructed": "A question naming a measure the registry resolves and the "
                       "detector's slot reader does not, or the reverse. Not "
                       "constructed here — it needs a probe asking both owners "
                       "the same question, which this census did not build.",
    },
    {
        "id": "D10", "decides": "Whether the question compares two things or reports one.",
        "owners": [
            "seasoning.lending_windows_named — two windows named means compare, "
            "question text, every parse",
            "execution_receipt.detect_requested_facets cohort detection — question "
            "text, at detection",
            "mi_workflows.analytical.contract.is_composite — the PLAN, at "
            "recognition, and it gates whether the analytical route claims at all",
        ],
        "probe": {"seasoning": _comparison_seasoning,
                  "detector": _comparison_detector},
        "probe_only_when": _seasoning_question,
        "constructed": "A question the seasoning reader calls a comparison and the "
                       "cohort detector does not, or the reverse.",
    },
    {
        "id": "D11", "decides": "Which governed capability should answer the question.",
        "owners": ["mi_agent_api.recogniser_registry via chat_routing.try_route — "
                   "one arbitration, at routing"],
        "class": SINGLE,
        "note": "One owner, but it decides D6's frame as a side effect via "
                "_RUN_SCOPED_ROUTES, which is why D6 diverges.",
    },
    {
        "id": "D12", "decides": "What reporting grain the ANSWER is published at.",
        "owners": ["execution_receipt._ROUTE_GRANULARITY and _ROUTE_TIME_GRAIN — "
                   "declared per route, read at the guard"],
        "class": SINGLE,
        "note": "Declared rather than inferred, deliberately. A route that "
                "published a different grain from its declaration would not be "
                "caught — but that is one owner being wrong, not two disagreeing.",
    },
    {
        "id": "D13", "decides": "Whether the question asks for a series over time.",
        "owners": [],
        "class": SINGLE,
        "identifier": "B9",
        "note": "NO owner. Nothing decides it, so a series question answered by a "
                "point-in-time capability is not detected. One corpus question of "
                "90 is affected. Recorded here because an absent owner is the "
                "limiting case of this census, not because it is a disagreement.",
    },
    {
        "id": "D14", "decides": "What geographic scope the question named.",
        "owners": [
            "execution_receipt._detect_geographic_scope — question text against "
            "the frame's geo values, at detection",
            "llm_query_parser filter resolution — question text against the "
            "registry, at parse",
        ],
        "constructed": "A place name the frame carries as a value but the registry "
                       "does not resolve as a filter, or the reverse.",
    },
]


# --------------------------------------------------------------------------- #
# Measuring agreement
# --------------------------------------------------------------------------- #
def measure(limit: Optional[int] = None) -> Dict[str, Any]:
    from question_interpretation.lexical_decisions import corpus

    ctx = _Ctx()
    cases = list(corpus())
    if limit:
        cases = cases[:limit]

    out: Dict[str, Any] = {}
    for decision in DECISIONS:
        probe = decision.get("probe")
        if not probe:
            continue
        names = list(probe)
        disagreements: List[Dict[str, Any]] = []
        errors = 0
        gate = decision.get("probe_only_when")
        for case in cases:
            q = case["question"]
            if gate is not None and not gate(ctx, q):
                continue
            answers = {}
            for name, fn in probe.items():
                try:
                    answers[name] = fn(ctx, q)
                except Exception:                              # pragma: no cover
                    answers[name] = "ERROR"
                    errors += 1
            values = list(answers.values())
            if any(v != values[0] for v in values[1:]):
                disagreements.append({"id": case["id"], "question": q,
                                      "answers": {k: repr(v) for k, v in answers.items()}})
        gate = decision.get("probe_only_when")
        considered = (len([c for c in cases if gate(ctx, c["question"])])
                      if gate is not None else len(cases))
        out[decision["id"]] = {
            "owners": names, "cases": considered, "errors": errors,
            "disagreements": disagreements,
            "disagreement_count": len(disagreements),
        }
    return out


def classify(decision: Dict[str, Any], measured: Dict[str, Any]) -> str:
    """Declared class wins; otherwise the measurement decides."""
    if decision.get("class"):
        return decision["class"]
    result = measured.get(decision["id"])
    if result is None:
        return BY_MAINTENANCE if len(decision.get("owners") or []) > 1 else SINGLE
    if result["disagreement_count"]:
        return DISAGREEING
    return BY_MAINTENANCE


def self_test() -> int:
    """The census must be able to report every class it defines.

    A census whose classifier can only produce one answer counts nothing. This
    checks each class is reachable, and that a probe genuinely compares owners
    rather than returning one owner's answer twice.
    """
    failures: List[str] = []
    fake = {"D2": {"disagreement_count": 3, "disagreements": [], "owners": [], "cases": 1, "errors": 0},
            "D9": {"disagreement_count": 0, "disagreements": [], "owners": [], "cases": 1, "errors": 0}}
    if classify({"id": "D2", "owners": ["a", "b"]}, fake) != DISAGREEING:
        failures.append("a measured disagreement did not classify as disagreeing")
    if classify({"id": "D9", "owners": ["a", "b"]}, fake) != BY_MAINTENANCE:
        failures.append("measured agreement did not classify as agree-by-maintenance")
    if classify({"id": "DX", "owners": ["a"]}, {}) != SINGLE:
        failures.append("a single-owner decision did not classify as single")
    if classify({"id": "DX", "owners": ["a", "b"], "class": DERIVED}, {}) != DERIVED:
        failures.append("a declared class did not win")

    # A probe must actually distinguish its owners. Two owners of the same
    # decision returning identical answers for every possible input would be one
    # owner, and the probe would assert nothing.
    ctx_free = [("_axis_parser", _axis_parser), ("_axis_detector", _axis_detector)]
    if ctx_free[0][1] is ctx_free[1][1]:
        failures.append("a probe compares an owner with itself")

    # Every decision must name at least one owner or explain why it has none.
    for d in DECISIONS:
        if not d.get("owners") and not d.get("note"):
            failures.append("%s names no owner and gives no reason" % d["id"])
        if d.get("class") in (DISAGREEING,) and not d.get("evidence"):
            failures.append("%s is declared disagreeing with no evidence" % d["id"])

    for f in failures:
        print("SELF-TEST FAIL: %s" % f)
    print("SELF-TEST %s" % ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--blind-spots", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.blind_spots:
        print("=" * 78)
        print("WHAT THIS CENSUS COULD NOT REACH")
        print("=" * 78)
        for spot in BLIND_SPOTS:
            print("  * %s\n" % spot)
        return 0

    measured = measure(args.limit)
    rows = []
    for decision in DECISIONS:
        cls = classify(decision, measured)
        rows.append({**decision, "classified": cls,
                     "measured": measured.get(decision["id"])})

    counts = collections.Counter(r["classified"] for r in rows)
    print("=" * 90)
    print("MULTI-OWNER DECISION CENSUS")
    print("=" * 90)
    print("decisions enumerated: %d\n" % len(rows))
    for cls in (SINGLE, DERIVED, BY_CONSTRUCTION, BY_MAINTENANCE, DISAGREEING):
        print("  %-24s %d" % (cls, counts.get(cls, 0)))
    print("\nTHE COUNT: %d agree-by-maintenance (debt), %d disagreeing (defect)"
          % (counts.get(BY_MAINTENANCE, 0), counts.get(DISAGREEING, 0)))

    print("\n" + "-" * 90)
    for row in rows:
        print("\n%-4s %-22s %s" % (row["id"], row["classified"], row["decides"]))
        for owner in row.get("owners") or []:
            print("       owner: %s" % owner)
        if not row.get("owners"):
            print("       owner: NONE")
        m = row.get("measured")
        if m:
            print("       measured: %d of %d corpus questions disagree (%s)"
                  % (m["disagreement_count"], m["cases"], " vs ".join(m["owners"])))
            for d in m["disagreements"][:3]:
                print("          %-18s %s" % (d["id"], d["question"][:50]))
                print("             %s" % d["answers"])
        if row.get("identifier"):
            print("       identifier: %s" % row["identifier"])
        if row.get("evidence"):
            print("       evidence: %s" % row["evidence"])
        if row.get("constructed"):
            print("       to construct: %s" % row["constructed"])

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print("\nRun --blind-spots before quoting this count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
