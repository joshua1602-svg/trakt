#!/usr/bin/env python3
"""question_interpretation/stamp_coverage.py

WHICH ROUTE CAN CONFIRM WHICH KIND OF REQUEST, AND WHERE THE HOLES ARE.

Twice a CORRECT classification has produced a block, because the route that
answered the question could not confirm it had applied what the question asked
for:

  * 32c263a — governed seasoning populations were correctly identified and
    correctly applied, and blocked because the check compared a field NAME
    against predicate TEXT rather than comparing governed definitions. 160 runs.
  * e35a01b — a grouping facet `reconcile_facets` can stamp becomes a population
    facet nothing on the point-in-time path can stamp. LOST, therefore blocked.
    "What is the balance of newly originated loans?" refuses.

Same shape both times: the classification was right and the confirmation was
missing. This is an INVENTORY of where confirmation exists and where it does
not. It fixes nothing.

Three things are measured, and they are different questions:

  RAISED     can a facet of this kind exist on this route at all? Derived from
             the constructors — which function builds a facet of that kind, and
             whether that function runs on that route.
  DISPATCH   does any branch of the reconciler for that route claim the kind?
             Measured with a sentinel status no reconciler writes, so "left
             alone" and "stamped LOST for a stated reason" stay distinguishable.
  PROOF      given evidence hand-built for that kind, can the reconciler reach
             APPLIED? Each bundle is an explicit claim about what counts as
             proof. Where no bundle is written the cell reads `not constructed`,
             which is an untested cell and NOT a hole.

A generous one-size evidence bundle was tried first and reported eleven false
holes. That is this programme's standing failure mode — an instrument carrying
the defect it was built to find — and is why PROOF is per-kind and why absence
of a bundle is never reported as a hole.

    python -m question_interpretation.stamp_coverage
    python -m question_interpretation.stamp_coverage --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: A status no reconciler writes, so "untouched" is answerable without guessing.
_UNTOUCHED = "__untouched__"
_FIELD = "probe_field"

#: Every route that can answer a question and reach a user, with the reconciler
#: that adjudicates its facets. Routed answers never reach the point-in-time
#: executor, so `mi_service._guard_routed_answer` adjudicates them through
#: `reconcile_routed_facets`; only a question NO route claims falls through to
#: the workflow and `reconcile_facets`.
ROUTES: Dict[str, str] = {
    "(point-in-time)": "point_in_time",
    "analytical_composition": "routed",
    "period_change_analysis": "routed",
    "portfolio_summary": "routed",
    "period_movement": "routed",
    "temporal_compare": "routed",
    "evolution": "routed",
    "evolution_funnel": "routed",
    "evolution_pipeline_stage": "routed",
    "forecast_extrapolation": "routed",
    "scenario": "routed",
    "risk_limits": "routed",
    "funded_bridge": "routed",
    "cohort_progression": "routed",
    "cohort_conversion": "routed",
    "geo_exposure": "routed",
}

#: Which constructor raises each kind, and on which routes that constructor
#: runs. `detect_requested_facets` reads the QUESTION and runs on both paths, so
#: everything it raises is reachable everywhere. The other two are path-bound,
#: and that is the whole of the routing dependency.
RAISED_BY: Dict[str, str] = {
    "KIND_STATISTIC": "detector", "KIND_GROUPING": "detector",
    "KIND_RANKING": "detector", "KIND_MULTI_MEASURE": "detector",
    "KIND_RELATIONSHIP": "detector", "KIND_SHARE": "detector",
    "KIND_PROJECTION": "detector", "KIND_COHORT_COMPARISON": "detector",
    "KIND_UNRESOLVED_MEASURE": "detector", "KIND_GEOGRAPHIC_SCOPE": "detector",
    "KIND_THRESHOLD": "detector", "KIND_STRESS": "detector",
    "KIND_CONTRIBUTION": "detector", "KIND_COMPARISON_PERIOD": "detector",
    # population_facets(spec) is appended by mi_service on the ROUTED path only.
    # _split_named_dimension_roles runs inside reconcile_facets, so it raises a
    # population on the POINT-IN-TIME path only.
    "KIND_POPULATION": "both-paths-different-constructors",
    "KIND_UNRESOLVED_ROLE": "split-only",
}


#: A hole is only a DEFECT when the route might actually have honoured the facet
#: and has no way to say so. Two of the three holes this scan finds are
#: deliberate, and the difference is visible in the code rather than asserted
#: here: a kind constructed with a status and a reason ALREADY SET is one the
#: detector has finished adjudicating, and a reconciler branch would have
#: nothing to add.
DESIGNED_HOLES: Dict[str, str] = {
    "KIND_UNRESOLVED_MEASURE": (
        "raised LOST at construction, with its own reason: the parser never "
        "resolved these words, so no execution could have honoured them and "
        "there is no evidence to weigh. Blocking is the intent — the outcome it "
        "prevents is a silent 3-of-4. A reconciler branch would have nothing to "
        "read."),
    "KIND_UNRESOLVED_ROLE": (
        "does not block. `assess` reads it directly, before the blocking test, "
        "and returns a clarification; the facet's STATUS is never consulted, so "
        "having no reconciler branch costs nothing. Only raised under the "
        "clarify variant."),
}


class _Spec:
    def __init__(self, **kw):
        self.filters = kw.get("filters", {})
        self.dimensions = kw.get("dimensions", [])
        self.dimension = kw.get("dimension")
        self.aggregation = kw.get("aggregation")


class _Result:
    def __init__(self, **kw):
        self.result_type = kw.get("result_type")
        self.data = kw.get("data")
        self.metadata = kw.get("metadata", {})


def _narrowed() -> Dict[str, Any]:
    return {"reconciliation": {"total_records": 100, "records_after_filters": 10}}


def _analytical_envelope() -> Dict[str, Any]:
    """An envelope in the shape the readers actually require.

    `analytical_evidence` accepts only a `metadata.analyticalComposition` block
    naming an intent AND its capabilities; `analytical_populations` reads
    `metadata.analyticalFindings[].population.predicate`. The first draft of
    this bundle used neither shape and the self-test caught it — which is the
    only reason this instrument is not reporting a false hole on the routed
    population cell.
    """
    return {"metadata": {
        "analyticalComposition": {"intent": "probe",
                                  "capabilities": ["probe_capability"]},
        "analyticalFindings": [
            {"population": {"predicate": "%s = probe value" % _FIELD}}],
        "rankedField": _FIELD,
    }}


def _pit_bundles(R) -> Dict[str, Dict[str, Any]]:
    """What proves each kind on the point-in-time path."""
    return {
        R.KIND_STATISTIC: dict(
            facet=dict(concepts=("avg",)),
            result=_Result(metadata={"measures_executed": [{"aggregation": "avg"}]})),
        R.KIND_GEOGRAPHIC_SCOPE: dict(
            facet=dict(label="london"), spec=_Spec(filters={_FIELD: "london"}),
            result=_Result(metadata=_narrowed())),
        R.KIND_THRESHOLD: dict(
            spec=_Spec(filters={_FIELD: {"op": "gt", "value": 80}}),
            result=_Result(metadata=_narrowed())),
        R.KIND_STRESS: dict(scenario_applied=True),
        R.KIND_COMPARISON_PERIOD: dict(),
        R.KIND_COHORT_COMPARISON: dict(
            facet=dict(alt_keys=(_FIELD,)),
            result=_Result(metadata={"group_field_keys": [_FIELD]})),
        R.KIND_MULTI_MEASURE: dict(result=_Result(result_type="loan_level")),
        R.KIND_SHARE: dict(spec=_Spec(aggregation="share")),
        R.KIND_CONTRIBUTION: dict(spec=_Spec(aggregation="contribution")),
        R.KIND_RELATIONSHIP: dict(result=_Result(result_type="loan_level")),
        # The spec must put the field on an AXIS. `reconcile_facets` runs the
        # role split on its first line, and a grouping facet whose field has no
        # role from any source is rewritten to KIND_UNRESOLVED_ROLE before any
        # branch sees it — so probing with an empty spec would measure the
        # split, not the grouping branch, and report a false hole.
        R.KIND_GROUPING: dict(
            spec=_Spec(dimensions=[_FIELD]),
            result=_Result(metadata={"group_field_keys": [_FIELD]})),
        R.KIND_RANKING: dict(
            spec=_Spec(dimensions=[_FIELD]),
            result=_Result(metadata={"group_field_keys": [_FIELD]})),
        R.KIND_PROJECTION: dict(),
        # A bundle exists here only because the branch does. Until the
        # reconciler gained a KIND_POPULATION case there was no evidence to
        # construct, and the absence of this entry WAS the finding.
        #
        # `applied_filter_fields` is what the executor publishes after
        # `_require_column` confirms the book carries the column and the mask
        # has run. Note what is NOT here: `spec.filters`. A filter the executor
        # dropped sits in the spec exactly as one it ran, and accepting spec
        # presence as proof is how the P1K silent errors passed.
        R.KIND_POPULATION: dict(
            result=_Result(metadata={"applied_filter_fields": [_FIELD]})),
        # KIND_UNRESOLVED_MEASURE deliberately absent: it is raised LOST with
        # its own reason at construction and no branch reads evidence for it,
        # by design. See DESIGNED_HOLES.
    }


def _routed_bundles(R) -> Dict[str, Dict[str, Any]]:
    """What proves each kind on a routed answer."""
    return {
        R.KIND_STATISTIC: dict(facet=dict(concepts=("avg",)), envelope={}),
        R.KIND_GROUPING: dict(envelope={}),
        R.KIND_POPULATION: dict(
            facet=dict(label="the population probe_field = probe value"),
            envelope=_analytical_envelope()),
        R.KIND_GEOGRAPHIC_SCOPE: dict(
            facet=dict(label="probe value"), envelope=_analytical_envelope()),
        R.KIND_COMPARISON_PERIOD: dict(envelope={}),
        R.KIND_STRESS: dict(envelope={}),
        R.KIND_PROJECTION: dict(envelope={}),
        R.KIND_COHORT_COMPARISON: dict(envelope={}),
        R.KIND_RANKING: dict(envelope=_analytical_envelope()),
    }


def _facet(R, kind: str, **over):
    kw = dict(kind=kind, label="a probe facet", field_key=_FIELD, alt_keys=(),
              concepts=("probe",), status=_UNTOUCHED, reason="")
    kw.update(over)
    return R.RequestedFacet(**kw)


def _run(R, kind, bundle, reconciler, route) -> str:
    facet = _facet(R, kind, **(bundle.get("facet") or {}))
    try:
        if reconciler == "point_in_time":
            out = R.reconcile_facets(
                [facet], spec=bundle.get("spec") or _Spec(),
                query_result=bundle.get("result") or _Result(),
                semantics={"fields": {}}, available_columns=bundle.get("columns"),
                route=route,
                scenario_applied=bundle.get("scenario_applied", False))
        else:
            out = R.reconcile_routed_facets(
                [facet], route=route,
                semantics={"fields": {}}, available_columns=bundle.get("columns"),
                envelope=bundle.get("envelope") or {})
    except Exception as exc:                                  # pragma: no cover
        return "ERROR %s" % type(exc).__name__
    return (out or [facet])[0].status


def _raisable(R, kind: str, route: str) -> Optional[str]:
    """None if raisable here; otherwise why it is unreachable on this route."""
    if kind == R.KIND_POPULATION:
        if route == "(point-in-time)":
            return None          # _split_named_dimension_roles raises it here
        return None              # population_facets(spec) is appended here
    if kind == R.KIND_UNRESOLVED_ROLE:
        if route == "(point-in-time)":
            return None
        return ("_split_named_dimension_roles runs only inside reconcile_facets, "
                "which no routed answer calls")
    return None                  # detector kinds: the detector reads the question


def scan() -> Dict[str, Any]:
    from mi_agent import execution_receipt as R

    kinds = [getattr(R, n) for n in sorted(dir(R))
             if n.startswith("KIND_") and isinstance(getattr(R, n), str)]
    bundles = {"point_in_time": _pit_bundles(R), "routed": _routed_bundles(R)}

    matrix: Dict[str, Dict[str, Any]] = {}
    for route, reconciler in ROUTES.items():
        row: Dict[str, Any] = {}
        for kind in kinds:
            why = _raisable(R, kind, route)
            if why:
                row[kind] = {"cell": "unreachable", "why": why}
                continue
            bare = _run(R, kind, {}, reconciler, route)
            bundle = bundles[reconciler].get(kind)
            proof = (_run(R, kind, bundle, reconciler, route)
                     if bundle is not None else None)
            # FOUR distinct cells, and the distinction is the whole point.
            #
            # A branch that considered the facet and concluded it was not
            # applied is CORRECT, not a hole: "compare with last quarter" on a
            # risk-limit schedule genuinely was not compared, and blocking is
            # the right answer. The defect this inventory exists for is the
            # other thing — no branch claims the kind at all, so the facet keeps
            # its default status whatever the route actually did, and a route
            # that DID honour it has no way to say so.
            dispatched = bare != _UNTOUCHED or (proof not in (None, _UNTOUCHED))
            if not dispatched:
                cell = "no-branch"
            elif proof is None:
                cell = "not constructed"
            elif proof == R.APPLIED:
                cell = "stamped"
            else:
                cell = "route-bound"
            row[kind] = {"cell": cell,
                         "bare": "-" if bare == _UNTOUCHED else bare,
                         "proof": (None if proof is None
                                   else ("-" if proof == _UNTOUCHED else proof)),
                         "blocks": (kind in R.NUMBER_OR_SUBJECT_FACETS
                                    or kind in (R.KIND_POPULATION, R.KIND_GROUPING,
                                                R.KIND_RANKING))}
        matrix[route] = row

    by_name = {getattr(R, n): n for n in dir(R)
               if n.startswith("KIND_") and isinstance(getattr(R, n), str)}
    for row in matrix.values():
        for kind, c in row.items():
            if c["cell"] == "no-branch":
                c["designed"] = DESIGNED_HOLES.get(by_name.get(kind, ""), None)

    holes = ["%s/%s" % (r, k) for r, row in matrix.items()
             for k, c in row.items() if c["cell"] == "no-branch"]
    live_holes = ["%s/%s" % (r, k) for r, row in matrix.items()
                  for k, c in row.items()
                  if c["cell"] == "no-branch" and not c.get("designed")]
    refusing = ["%s/%s" % (r, k) for r, row in matrix.items()
                for k, c in row.items()
                if c["cell"] == "no-branch" and c.get("blocks")]
    route_bound = sorted({k for row in matrix.values()
                          for k, c in row.items() if c["cell"] == "route-bound"})
    return {"kinds": kinds, "matrix": matrix, "holes": holes,
            "live_holes": live_holes, "holes_that_refuse": refusing,
            "route_bound_kinds": route_bound,
            "designed_holes": DESIGNED_HOLES}


def self_test() -> int:
    """The instrument must be able to produce the result it reports.

    Two properties. A hole must be detectable — the one this inventory exists
    for, KIND_POPULATION on the point-in-time path, must come back `hole`. And a
    stamped cell must be detectable, or "everything is a hole" would pass.
    """
    from mi_agent import execution_receipt as R
    result = scan()
    pit = result["matrix"]["(point-in-time)"]
    failures: List[str] = []

    # Every cell value must be PRODUCIBLE, and none of these probes may depend
    # on a defect being present. The first draft of this self-test asserted
    # KIND_POPULATION on the point-in-time path read as a hole — true at the
    # time, and it started failing the moment that hole was closed, which is
    # the correct outcome but the wrong assertion: an instrument must not be
    # anchored to the bug it was built to find.
    if pit[R.KIND_UNRESOLVED_MEASURE]["cell"] != "no-branch":
        failures.append("KIND_UNRESOLVED_MEASURE on the point-in-time path did "
                        "not read as a hole; the instrument cannot see a hole "
                        "at all")
    if pit[R.KIND_GROUPING]["cell"] != "stamped":
        failures.append("KIND_GROUPING on the point-in-time path did not read as "
                        "stamped; the instrument reports holes indiscriminately")
    if pit[R.KIND_POPULATION]["cell"] != "stamped":
        failures.append("KIND_POPULATION on the point-in-time path did not read "
                        "as stamped; the reconciler branch that closed the "
                        "e35a01b hole is missing or unreachable")
    routed = result["matrix"]["analytical_composition"]
    if routed[R.KIND_POPULATION]["cell"] != "stamped":
        failures.append("KIND_POPULATION on a routed answer did not read as "
                        "stamped; the population ledger path is broken")
    if routed[R.KIND_UNRESOLVED_ROLE]["cell"] != "unreachable":
        failures.append("KIND_UNRESOLVED_ROLE read as raisable on a routed "
                        "answer; the instrument cannot tell the two paths apart")
    # The fourth cell must also be producible, or "route-bound" and "no-branch"
    # would be indistinguishable and every route-specific facet would read as a
    # defect. A comparison period on a risk-limit schedule is the canonical one.
    if result["matrix"]["risk_limits"][R.KIND_COMPARISON_PERIOD]["cell"] != "route-bound":
        failures.append("a comparison period on risk_limits did not read as "
                        "route-bound; the instrument cannot tell a correct "
                        "non-application from a missing branch")
    for f in failures:
        print("SELF-TEST FAIL: %s" % f)
    print("SELF-TEST %s" % ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    result = scan()
    short = {k: k.replace("KIND_", "")[:11] for k in result["kinds"]}
    glyph = {"stamped": ".", "no-branch": "HOLE", "unreachable": "n/a",
             "not constructed": "?", "route-bound": "o"}

    print("=" * 100)
    print("STAMP COVERAGE — routes down, facet kinds across")
    print("=" * 100)
    print("  .  stamped — a branch can confirm it on this route")
    print("  o  route-bound — a branch considered it; this route does not do it")
    print("     (correct behaviour, not a hole)")
    print("  HOLE  no branch claims the kind: it keeps LOST whatever the route did")
    print("  ?  not constructed (untested, NOT a hole)")
    print("  n/a  cannot be raised on this route")
    print()
    hdr = "%-26s" % "route"
    for k in result["kinds"]:
        hdr += "%-6s" % short[k][:5]
    print(hdr)
    print("-" * 100)
    for route, row in result["matrix"].items():
        line = "%-26s" % route
        for k in result["kinds"]:
            line += "%-6s" % glyph[row[k]["cell"]]
        print(line)
    print("-" * 100)
    print("kind key:")
    for k in result["kinds"]:
        print("   %-6s %s" % (short[k][:5], k))
    print()
    print("HOLES (%d), of which DESIGNED (%d) and LIVE (%d):"
          % (len(result["holes"]),
             len(result["holes"]) - len(result["live_holes"]),
             len(result["live_holes"])))
    print()
    print("  DESIGNED — a branch would have nothing to read:")
    for kind, why in result["designed_holes"].items():
        n = len([h for h in result["holes"] if h.endswith("/" + kind.replace("KIND_", "").lower())])
        print("    %-26s %s" % (kind, why[:64]))
    print()
    print("  LIVE — the route may have honoured it and cannot say so:")
    for h in result["live_holes"]:
        print("    %s" % h)

    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
