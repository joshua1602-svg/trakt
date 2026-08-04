"""simulation.assertions.mi — deterministic MI against independent expected truth.

Every figure here comes from the PRODUCTION MI capability:

* ``mi_agent.states.assembler.total_funded`` — the funded state at one cut;
* ``mi_agent.states.temporal.compare`` / ``trend`` — period movement and series;
* ``mi_agent.mi_runtime.run_mi_query`` — the governed runtime boundary itself;
* ``mi_agent.states.assembler.cohort_by_date`` — vintage / cohort analysis;
* ``analytics_lib.concentration`` — the shared concentration primitive.

Nothing is recalculated here. The expected side comes from
:mod:`simulation.reference_truth`, which imports none of the above.

An HTTP 200 is never treated as a result: every assertion compares a FIGURE.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..models import STAGE_MI, Assertion, FailureClass
from . import MONEY_ATOL, PERCENT_ATOL, check, compare

_FAIL = FailureClass.MI_ASSERTION_FAILURE
_BALANCE = "current_outstanding_balance"


def _sum(frame: pd.DataFrame, column: str = _BALANCE) -> Optional[float]:
    if frame is None or column not in getattr(frame, "columns", []):
        return None
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


# --------------------------------------------------------------------------- #
# Point-in-time funded state
# --------------------------------------------------------------------------- #
def assert_funded_state(store, case, truth: Dict[str, Any]) -> List[Assertion]:
    """The production funded state at every reporting date."""
    from mi_agent.states import assemble_state
    from mi_agent.states.selectors import SnapshotSelector

    out: List[Assertion] = []
    for period in truth["periods"]:
        date = period["reporting_date"]
        selector = SnapshotSelector.as_of(case.client_id, date, route="mi")
        state = assemble_state("total_funded", store, selector=selector, route=None)
        frame = state.frame

        out.append(check(STAGE_MI, f"{date}: funded loan count",
                         len(frame) == period["loan_count"],
                         expected=period["loan_count"], actual=len(frame),
                         failure_class=_FAIL))
        out.append(compare(STAGE_MI, f"{date}: funded current balance",
                           _sum(frame), period["closing_balance"],
                           failure_class=_FAIL))

        # Status distribution, in the tape's reported vocabulary.
        if "account_status" in frame.columns:
            actual = {str(k): round(float(v), 2) for k, v in
                      pd.to_numeric(frame[_BALANCE], errors="coerce").fillna(0.0)
                      .groupby(frame["account_status"].astype(str)).sum().items()}
            expected = _expected_reported_status_balances(period)
            same = (set(actual) == set(expected)
                    and all(abs(actual[k] - expected[k]) <= MONEY_ATOL
                            for k in expected))
            out.append(check(STAGE_MI, f"{date}: status distribution",
                             same, expected=expected, actual=actual,
                             failure_class=_FAIL))
    return out


def _expected_reported_status_balances(period: Dict[str, Any]) -> Dict[str, float]:
    """Expected balance per REPORTED status token.

    The internal ladder has an ``enforcement`` rung; the tape reports it as
    ``DEFAULT`` with the litigation flag set, so the expected truth is folded the
    same way. The fold is stated here, in the expected truth, rather than being
    absorbed by loosening the assertion.
    """
    from ..assets._common import REPORTED_STATUS

    out: Dict[str, float] = {}
    for status, balance in (period["status_balances"] or {}).items():
        token = REPORTED_STATUS.get(status, status.upper())
        out[token] = round(out.get(token, 0.0) + float(balance), 2)
    return out


# --------------------------------------------------------------------------- #
# Period movement
# --------------------------------------------------------------------------- #
def assert_period_movement(store, case, truth: Dict[str, Any]) -> List[Assertion]:
    """Month-on-month movement through the production temporal engine."""
    from mi_agent.states.temporal import compare as temporal_compare

    out: List[Assertion] = []
    periods = truth["periods"]
    for prior, current in zip(periods, periods[1:]):
        result = temporal_compare(
            store, "total_funded", case.client_id, route=None,
            baseline_date=prior["reporting_date"],
            current_date=current["reporting_date"], balance_col=_BALANCE)
        date = current["reporting_date"]
        frame = result.frame
        if frame is None or frame.empty:
            out.append(check(STAGE_MI, f"{date}: comparison produced a result",
                             False, actual=result.issues, failure_class=_FAIL))
            continue
        row = frame.iloc[0]

        expected_move = round(current["closing_balance"] - prior["closing_balance"], 2)
        actual_move = _as_float(row.get("balance_change"))
        out.append(compare(STAGE_MI, f"{date}: period movement", actual_move,
                           expected_move, failure_class=_FAIL))

        expected_direction = _direction(expected_move)
        actual_direction = _direction(actual_move)
        out.append(check(STAGE_MI, f"{date}: month-on-month direction",
                         actual_direction == expected_direction,
                         expected=expected_direction, actual=actual_direction,
                         failure_class=_FAIL))

        out.append(compare(STAGE_MI, f"{date}: baseline balance",
                           _as_float(row.get("baseline_balance")),
                           prior["closing_balance"], failure_class=_FAIL))
        out.append(compare(STAGE_MI, f"{date}: current balance",
                           _as_float(row.get("current_balance")),
                           current["closing_balance"], failure_class=_FAIL))
        out.append(check(STAGE_MI, f"{date}: current loan count",
                         int(row.get("current_count") or 0) == current["loan_count"],
                         expected=current["loan_count"],
                         actual=int(row.get("current_count") or 0),
                         failure_class=_FAIL))
        # Exits are recovered by comparing the two tapes, which is exactly how
        # production reports a redemption when the tape carries only open loans.
        out.append(check(STAGE_MI, f"{date}: exits recovered from the comparison",
                         int(row.get("exited_count") or 0)
                         == current["exited_loan_count"],
                         expected=current["exited_loan_count"],
                         actual=int(row.get("exited_count") or 0),
                         failure_class=_FAIL))
    return out


def _as_float(value) -> Optional[float]:
    try:
        return None if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None


def _direction(value: Optional[float], tolerance: float = 0.5) -> str:
    if value is None:
        return "unknown"
    if value > tolerance:
        return "increase"
    if value < -tolerance:
        return "decrease"
    return "flat"


def assert_trend(store, case, truth: Dict[str, Any]) -> List[Assertion]:
    """The whole series through the production trend engine."""
    from mi_agent.states.temporal import trend as temporal_trend

    dates = truth["reporting_dates"]
    result = temporal_trend(store, "total_funded", case.client_id, route=None,
                            start_date=dates[0], end_date=dates[-1],
                            balance_col=_BALANCE)
    frame = result.frame
    out: List[Assertion] = [check(
        STAGE_MI, "trend covers every reporting date",
        sorted(set(frame.get("reporting_date", pd.Series()).astype(str))) == dates,
        expected=dates,
        actual=sorted(set(frame.get("reporting_date", pd.Series()).astype(str))),
        failure_class=_FAIL)]
    if frame.empty:
        return out
    by_date = (pd.to_numeric(frame["balance"], errors="coerce")
               .groupby(frame["reporting_date"].astype(str)).sum().to_dict())
    for period in truth["periods"]:
        out.append(compare(
            STAGE_MI, f"{period['reporting_date']}: trend balance",
            by_date.get(period["reporting_date"]), period["closing_balance"],
            failure_class=_FAIL))
    return out


# --------------------------------------------------------------------------- #
# Concentration, cohorts and the asset-specific measure
# --------------------------------------------------------------------------- #
def assert_concentration(store, case, truth: Dict[str, Any]) -> List[Assertion]:
    """Largest concentration through the production risk-monitor primitive."""
    from mi_agent.risk_monitor.monitor import run_concentration

    out: List[Assertion] = []
    final = truth["periods"][-1]
    date = final["reporting_date"]
    dimension = _region_dimension(store, case, date)
    if dimension is None:
        return [check(STAGE_MI, f"{date}: a region dimension is available",
                      False, failure_class=_FAIL)]

    result = run_concentration(store, case.client_id, dimension, route="mi",
                               state="total_funded", reporting_date=date)
    frame = result.frame
    if "largest_concentrations" not in final:
        # A combined SPV truth deliberately does not carry a largest-group
        # figure: the largest region across two deliveries is not derivable
        # from each delivery's own largest region, and inventing it here would
        # make the expected truth depend on the platform it is checking.
        return [check(STAGE_MI,
                      f"{date}: largest-region concentration (not combinable)",
                      True, detail="Skipped for a multi-source SPV: the "
                                   "expected figure is not derivable from the "
                                   "per-source truths.")]
    expected = final["largest_concentrations"]["region"]
    if frame.empty:
        return [check(STAGE_MI, f"{date}: concentration produced rows", False,
                      failure_class=_FAIL)]

    top = frame.iloc[0]
    out.append(check(STAGE_MI, f"{date}: largest region",
                     str(top.get(dimension)) == str(expected["group"]),
                     expected=expected["group"], actual=str(top.get(dimension)),
                     failure_class=_FAIL))
    out.append(compare(STAGE_MI, f"{date}: largest region balance",
                       _as_float(top.get("balance_sum")), expected["balance"],
                       failure_class=_FAIL))
    # The production primitive reports a 0-1 share; expected truth is points.
    out.append(compare(STAGE_MI, f"{date}: largest region share",
                       (_as_float(top.get("balance_share")) or 0.0) * 100.0,
                       expected["share_percent"], atol=PERCENT_ATOL,
                       failure_class=_FAIL))
    out.append(compare(STAGE_MI, f"{date}: concentration totals the book",
                       float(pd.to_numeric(frame["balance_sum"],
                                           errors="coerce").fillna(0.0).sum()),
                       expected["total"], failure_class=_FAIL))
    out.append(check(STAGE_MI, f"{date}: concentration is ordered by exposure",
                     list(pd.to_numeric(frame["balance_sum"], errors="coerce"))
                     == sorted(pd.to_numeric(frame["balance_sum"],
                                             errors="coerce"), reverse=True),
                     failure_class=_FAIL))
    return out


def _region_dimension(store, case, date: str) -> Optional[str]:
    header = store.resolve_as_of(case.client_id, date, route="mi")
    columns = set(store.load_loans(header.snapshot_id).columns)
    for candidate in ("collateral_geography", "geographic_region_collateral",
                      "geographic_region_obligor"):
        if candidate in columns:
            return candidate
    return None


def assert_cohorts(store, case, truth: Dict[str, Any]) -> List[Assertion]:
    """Vintage analysis through the production cohort state."""
    from mi_agent.states import assemble_state
    from mi_agent.states.selectors import SnapshotSelector

    final = truth["periods"][-1]
    date = final["reporting_date"]
    selector = SnapshotSelector.as_of(case.client_id, date, route="mi")
    state = assemble_state("cohort_by_origination_date", store, selector=selector,
                           route=None, period="Y")
    frame = state.frame
    column = (state.metadata or {}).get("cohort_column")
    if frame is None or frame.empty or not column or column not in frame.columns:
        return [check(STAGE_MI, f"{date}: origination cohorts assembled", False,
                      actual=(state.metadata or {}).get("cohort_column"),
                      failure_class=_FAIL)]

    actual = {str(k)[:4]: round(float(v), 2) for k, v in
              pd.to_numeric(frame[_BALANCE], errors="coerce").fillna(0.0)
              .groupby(frame[column].astype(str)).sum().items()}
    expected = {k: round(float(v), 2) for k, v in final["vintage_balances"].items()}
    same = (set(actual) == set(expected)
            and all(abs(actual[k] - expected[k]) <= MONEY_ATOL for k in expected))
    return [check(STAGE_MI, f"{date}: origination-vintage balances", same,
                  expected=expected, actual=actual, failure_class=_FAIL)]


def assert_asset_measure(store, case, truth: Dict[str, Any]) -> List[Assertion]:
    """One asset-specific maturity or collateral measure, from the stored frame.

    Computed with the production governed limit library's evaluators rather than
    a bespoke calculation, so this checks the platform's own measure.
    """
    from mi_agent.concentration_tests.library import load_library
    from mi_agent.concentration_tests.metrics import evaluate_metric

    final = truth["periods"][-1]
    date = final["reporting_date"]
    header = store.resolve_as_of(case.client_id, date, route="mi")
    frame = store.load_loans(header.snapshot_id)
    lib = load_library()
    measures = final.get("asset_measures")
    if measures is None:
        return [check(STAGE_MI, f"{date}: asset-specific measure (not combinable)",
                      True, detail="Skipped for a multi-source SPV: shares and "
                                   "weighted measures do not add across "
                                   "separately delivered populations.")]

    if case.asset_class == "bridge":
        metric = lib.get("maturity_within_horizon_share")
        comp = evaluate_metric(frame, lib, metric,
                               {"horizon_days": 90, "denominator": "current_balance",
                                "_reporting_date": date})
        return [compare(STAGE_MI, f"{date}: balance maturing within 90 days",
                        comp.numerator_value,
                        measures["balance_maturing_within_90_days"],
                        failure_class=_FAIL),
                compare(STAGE_MI, f"{date}: share maturing within 90 days",
                        comp.value, measures["share_maturing_within_90_days"],
                        atol=PERCENT_ATOL, failure_class=_FAIL)]

    if case.asset_class == "asset_finance":
        metric = lib.get("residual_value_exposure_share")
        comp = evaluate_metric(frame, lib, metric,
                               {"residual_basis": "balloon", "minimum_amount": 0,
                                "denominator": "current_balance"})
        return [compare(STAGE_MI, f"{date}: balance carrying a balloon",
                        comp.numerator_value, measures["balance_with_balloon"],
                        failure_class=_FAIL)]

    metric = lib.get("ltv_weighted_average")
    comp = evaluate_metric(frame, lib, metric, {"ltv_basis": "current"})
    return [compare(STAGE_MI, f"{date}: weighted-average current LTV",
                    comp.value, final["weighted_average_current_ltv"],
                    atol=PERCENT_ATOL, failure_class=_FAIL)]


# --------------------------------------------------------------------------- #
# The governed runtime boundary
# --------------------------------------------------------------------------- #
def assert_runtime_boundary(store, case, truth: Dict[str, Any]) -> List[Assertion]:
    """The same figures, through ``mi_agent.mi_runtime.run_mi_query``.

    The dashboard and the Copilot action both reach MI through this boundary, so
    a figure that is right in the state assembler but wrong here would still be
    wrong for every user.
    """
    from mi_agent.mi_query_spec import MIQuerySpec
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.mi_runtime import run_mi_query

    from mi_agent_api.data_source import semantics_path

    semantics = load_mi_semantics(str(semantics_path()))
    final = truth["periods"][-1]
    spec = MIQuerySpec(
        state="total_funded", execution_mode="state", route_id="mi",
        temporal_mode="as_of", as_of_date=final["reporting_date"],
        snapshot_client_id=case.client_id)
    result = run_mi_query(spec, semantics=semantics, store=store)
    return [
        check(STAGE_MI, "runtime boundary answered without error", result.ok,
              actual=result.issue_codes(), failure_class=_FAIL),
        check(STAGE_MI, "runtime boundary loan count",
              result.row_count == final["loan_count"],
              expected=final["loan_count"], actual=result.row_count,
              failure_class=_FAIL),
        compare(STAGE_MI, "runtime boundary current balance",
                _sum(result.data), final["closing_balance"], failure_class=_FAIL),
    ]


# --------------------------------------------------------------------------- #
# Channel parity
# --------------------------------------------------------------------------- #
def assert_channel_parity(governed_result) -> List[Assertion]:
    """React and Copilot must receive materially identical governed results.

    Both channels are adapters over ONE ``GovernedResult``. The React presenter
    returns the analytical envelope verbatim; the Copilot action reads
    ``result.result`` from the same object. Parity therefore means: the figures
    a Copilot answer is built from are the figures React renders — asserted here
    on the object both adapters consume.
    """
    from mi_agent_api import presenters

    react = presenters.to_react_payload(governed_result)
    copilot_envelope = governed_result.result
    out: List[Assertion] = []
    for key in ("answer", "ok", "interpreted"):
        out.append(check(
            STAGE_MI, f"React and Copilot see the same '{key}'",
            react.get(key) == (copilot_envelope or {}).get(key),
            expected=(copilot_envelope or {}).get(key), actual=react.get(key),
            failure_class=_FAIL))
    react_artifacts = react.get("artifacts") or []
    copilot_artifacts = (copilot_envelope or {}).get("artifacts") or []
    out.append(check(
        STAGE_MI, "React and Copilot see the same supporting artefacts",
        len(react_artifacts) == len(copilot_artifacts),
        expected=len(copilot_artifacts), actual=len(react_artifacts),
        failure_class=_FAIL))
    out.append(check(
        STAGE_MI, "the governed envelope names one capability",
        (react.get("governance") or {}).get("capability") == "mi.question.answer",
        actual=(react.get("governance") or {}).get("capability"),
        failure_class=_FAIL))
    return out


__all__ = ["assert_asset_measure", "assert_channel_parity", "assert_cohorts",
           "assert_concentration", "assert_funded_state", "assert_period_movement",
           "assert_runtime_boundary", "assert_trend"]
