"""simulation.assertions.risk — the governed limit engine, green / amber / red.

The tests evaluated here are built from the SAME governed library the platform
uses (``config/risk/concentration_test_library.yaml``) and are evaluated by the
SAME service the dashboard, the MI query and Copilot consume
(``mi_agent.concentration_tests.evaluation.evaluate_active_tests``). Nothing is
recomputed and no threshold is hard-coded in procedural code: each asset profile
declares its tests, and each manifest declares which limits its economics are
engineered to stress.

An activated configuration is minted here exactly as the operator approval
workflow mints one — the same ``ActiveConfiguration`` dataclass, with evidence
and an approval record naming the simulation as the source. That keeps the
fail-closed rules in play: an unapproved or unknown test still cannot evaluate.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..models import (
    RISK_AMBER,
    RISK_GREEN,
    RISK_RED,
    STAGE_RISK,
    Assertion,
    FailureClass,
)
from . import check

_DEFECT = FailureClass.PRODUCTION_DEFECT

#: Governed status → the RAG state a manifest declares.
_STATUS_TO_RAG = {"pass": RISK_GREEN, "warning": RISK_AMBER, "breach": RISK_RED}


def build_active_configuration(case, *, effective_date: str):
    """Mint an activated configuration for a case, through the production model.

    The tests come from the asset profile; the thresholds come from the case's
    declared risk intent. Both are configuration — there is no client-specific
    limit anywhere in the framework's code.
    """
    from mi_agent.concentration_tests.library import load_library
    from mi_agent.concentration_tests.models import (
        ActiveConfiguration,
        ActiveTest,
        ApprovalRecord,
        SourceEvidence,
    )

    from ..assets import get_profile

    lib = load_library()
    profile = get_profile(case.asset_class)
    tests: List[ActiveTest] = []
    for spec in profile.risk_tests(case):
        metric = lib.get(spec["metric_id"])
        tests.append(ActiveTest(
            test_id=spec["test_id"], metric_id=spec["metric_id"],
            display_name=spec["display_name"], category=spec["category"],
            operator=spec["operator"], threshold=float(spec["threshold"]),
            unit=(metric.unit if metric else "percent"),
            parameters=dict(spec.get("parameters") or {}),
            effective_date=effective_date,
            evidence=SourceEvidence(
                source_reference=f"simulation manifest {case.case_id}",
                source_text=(f"Limit engineered by the {case.asset_class} "
                             f"hardening case {case.case_id}."),
                locator=f"simulation/assets/{case.asset_class}.py"),
            approval=ApprovalRecord(
                decision="approved", operator="simulation",
                approved_metric_id=spec["metric_id"],
                approved_parameters=dict(spec.get("parameters") or {}),
                effective_date=effective_date,
                comments="Activated by the asset-class hardening framework."),
            definition_notes=spec.get("definition_notes", "")))

    config = ActiveConfiguration(
        client_id=case.client_id, portfolio_id=case.portfolio_id, version=1,
        status="active", tests=tests, activated_by="simulation",
        activated_at=f"{effective_date}T00:00:00+00:00",
        reason=f"Asset-class hardening case {case.case_id}",
        library_version=lib.library_version)
    config.content_hash = config.compute_content_hash()
    return config, lib


def evaluate(store, case, truth: Dict[str, Any]) -> Dict[str, Any]:
    """Run the production evaluation service over the final two periods."""
    from mi_agent.concentration_tests.evaluation import evaluate_active_tests

    dates = truth["reporting_dates"]
    current, prior = dates[-1], (dates[-2] if len(dates) > 1 else "")
    config, lib = build_active_configuration(case, effective_date=dates[0])

    current_frame = _frame(store, case, current)
    prior_frame = _frame(store, case, prior) if prior else None
    return evaluate_active_tests(current_frame, prior_frame, config, lib,
                                 reporting_date=current,
                                 prior_reporting_date=prior)


def _frame(store, case, date: str):
    header = store.resolve_as_of(case.client_id, date, route="mi")
    return store.load_loans(header.snapshot_id)


def assert_risk_state(evaluation: Dict[str, Any], case) -> List[Assertion]:
    """The governed outcome must match the risk state the case engineered."""
    out: List[Assertion] = []
    summary = evaluation.get("summary") or {}
    tests = {t["testId"]: t for t in evaluation.get("tests") or []}
    intended = case.risk_intent.intended_status
    targeted = set(case.risk_intent.targeted_limits)

    out.append(check(STAGE_RISK, "every declared test evaluated",
                     len(tests) == summary.get("activeTests"),
                     expected=summary.get("activeTests"), actual=len(tests),
                     failure_class=_DEFECT))

    # No test may report ``unavailable``: an unresolvable field role means the
    # asset class is not actually wired into the governed library.
    unavailable = {tid: t["missingFields"] for tid, t in tests.items()
                   if t["status"] in ("unavailable", "insufficient_data")}
    out.append(check(STAGE_RISK, "every test resolved its canonical fields",
                     not unavailable, expected={}, actual=unavailable,
                     detail="An unavailable limit means the canonical data does "
                            "not carry the field role the test needs.",
                     failure_class=_DEFECT))

    overall = _STATUS_TO_RAG.get(str(summary.get("overallStatus")), "unknown")
    out.append(check(STAGE_RISK, "overall risk state matches the case intent",
                     overall == intended, expected=intended, actual=overall,
                     detail=f"Governed status: {summary.get('overallStatus')!r}.",
                     failure_class=_DEFECT))

    # The NAMED limits are the ones that must have moved.
    profile_tests = _tests_by_limit(case)
    for limit in sorted(targeted):
        test_ids = profile_tests.get(limit, [])
        if not test_ids:
            out.append(check(STAGE_RISK, f"limit {limit!r} has a governed test",
                             False, failure_class=FailureClass.SIMULATION_DEFECT))
            continue
        states = {tid: tests.get(tid, {}).get("status") for tid in test_ids}
        moved = any(s in ("warning", "breach") for s in states.values())
        out.append(check(STAGE_RISK, f"targeted limit {limit!r} is amber or red",
                         moved, expected="warning or breach", actual=states,
                         failure_class=_DEFECT))

    # A green case must not breach anything.
    if intended == RISK_GREEN:
        breached = {tid: t["status"] for tid, t in tests.items()
                    if t["status"] in ("warning", "breach")}
        out.append(check(STAGE_RISK, "a green case breaches nothing",
                         not breached, expected={}, actual=breached,
                         failure_class=_DEFECT))

    # Utilisation and headroom must be present wherever a value was measured —
    # "which concentration is closest to its limit" depends on them.
    measured = [t for t in tests.values() if t["currentValue"] is not None]
    out.append(check(STAGE_RISK, "measured tests report headroom",
                     all(t["headroom"] is not None for t in measured),
                     actual=[t["testId"] for t in measured
                             if t["headroom"] is None],
                     failure_class=_DEFECT))
    out.append(check(STAGE_RISK, "the summary names the closest limit",
                     bool(summary.get("closestToLimit")) or not measured,
                     actual=summary.get("closestToLimit"),
                     failure_class=_DEFECT))
    return out


def _tests_by_limit(case) -> Dict[str, List[str]]:
    """``{limit name: [test ids]}`` for the limits a case declares.

    Derived from the asset profile by re-reading the thresholds it produces for
    the stressed and unstressed forms of the same case: a test whose threshold
    changes under stress IS the test that limit controls.
    """
    import dataclasses

    from ..assets import get_profile
    from ..models import RiskIntent

    profile = get_profile(case.asset_class)
    relaxed_case = dataclasses.replace(
        case, risk_intent=RiskIntent(RISK_GREEN, ()))
    relaxed = {t["test_id"]: t["threshold"] for t in profile.risk_tests(relaxed_case)}

    out: Dict[str, List[str]] = {}
    for limit in case.risk_intent.targeted_limits:
        single = dataclasses.replace(
            case, risk_intent=RiskIntent(case.risk_intent.intended_status, (limit,)))
        for test in profile.risk_tests(single):
            if test["threshold"] != relaxed.get(test["test_id"]):
                out.setdefault(limit, []).append(test["test_id"])
    return out


def assert_movement_disclosed(evaluation: Dict[str, Any],
                              periods: int = 2) -> List[Assertion]:
    """A limit's movement against the prior period must be reported.

    ``periods`` is how many reporting periods the run actually integrated. A
    single-snapshot run — the large-scale ingestion benchmark is one — has no
    prior period BY CONSTRUCTION, and demanding one there would report a
    property of the run shape as a platform defect. With two or more periods a
    missing prior comparison is still a genuine finding.
    """
    tests = evaluation.get("tests") or []
    with_prior = [t for t in tests if t.get("priorAvailable")]
    if not with_prior:
        if periods < 2:
            return [check(
                STAGE_RISK, "no prior-period comparison is claimed",
                not with_prior, expected="no direction of travel",
                detail="This run integrated a single reporting period, so "
                       "there is no prior cut to compare against and the "
                       "evaluation correctly claims no movement.",
                failure_class=_DEFECT)]
        return [check(STAGE_RISK, "a prior period was available for comparison",
                      False, detail="Without a prior period the risk answer "
                                    "cannot state a direction of travel.",
                      failure_class=_DEFECT)]
    return [check(
        STAGE_RISK, "every measured limit reports its prior value and change",
        all(t["priorValue"] is not None or t["currentValue"] is None
            for t in with_prior),
        actual=[t["testId"] for t in with_prior
                if t["priorValue"] is None and t["currentValue"] is not None],
        failure_class=_DEFECT)]


__all__ = ["assert_movement_disclosed", "assert_risk_state",
           "build_active_configuration", "evaluate"]
