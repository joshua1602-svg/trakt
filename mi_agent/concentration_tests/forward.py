"""mi_agent.concentration_tests.forward — three-state concentration evaluation.

Evaluates every approved concentration test across three clearly distinguished
portfolio states, reusing the SAME deterministic evaluators and the SAME
governed pipeline forecast outputs — never a second probability model:

* **funded** — the contractual position: governed funded portfolio only.
  Untouched by anything in this module.
* **expected_forecast** — funded exposure at 100% plus each active pipeline
  case weighted by the completion probability the existing forecast engine
  assigned it (empirical stage rates over the weekly-snapshot observation
  window, config fallback below the sufficiency floor). Numerator and
  denominator move together via the probability column the evaluators honour.
* **full_pipeline** — funded plus 100% of all ACTIVE in-scope pipeline
  exposure. A deliberately unrealistic maximum-exposure stress, never a
  prediction, and always labelled as such.

Metric families keep their approved contractual semantics per a governed
treatment table: exposure-weighting where the maths is valid, scenario
inclusion where only whole loans make sense, indicative-only where a
probability-weighted number would mislead, and honest unavailability
otherwise. Never a fractional loan presented as a maximum.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from analytics_lib.numeric import coerce_numeric

from .evaluation import (
    _breach_amount,
    _headroom,
    _status_for,
    _utilization,
)
from .library import ConcentrationLibrary
from .metrics import (
    PROBABILITY_COL,
    evaluate_metric,
    resolve_role_column,
)
from .models import (
    ActiveConfiguration,
    ActiveTest,
    DATA_MISSING,
    DATA_OK,
    STATUS_BREACH,
    STATUS_PASS,
    STATUS_UNAVAILABLE,
    STATUS_WARNING,
)

# --------------------------------------------------------------------------- #
# States and treatments (closed vocabularies)
# --------------------------------------------------------------------------- #
STATE_FUNDED = "funded"
STATE_EXPECTED = "expected_forecast"
STATE_FULL = "full_pipeline"
STATES = (STATE_FUNDED, STATE_EXPECTED, STATE_FULL)

TREATMENT_EXPOSURE = "supported_with_exposure_weighting"
TREATMENT_SCENARIO = "supported_with_scenario_inclusion"
TREATMENT_INDICATIVE = "indicative_only"
TREATMENT_STATE_INDEPENDENT = "state_independent"
TREATMENT_UNSUPPORTED = "unsupported"

#: Governed forecast treatment per evaluator family. The EXPECTED state uses
#: this table; the FULL state uses scenario inclusion (probability 1.0)
#: wherever expected is exposure-weighted or scenario, and the same
#: indicative/unsupported/state-independent classification otherwise.
FORECAST_TREATMENT: Dict[str, str] = {
    # Additive balance-share metrics: expected exposure weights are valid on
    # both sides of the share.
    "share_of_balance": TREATMENT_EXPOSURE,
    "share_of_balance_numeric": TREATMENT_EXPOSURE,
    "postcode_area_share": TREATMENT_EXPOSURE,
    "dimension_share": TREATMENT_EXPOSURE,
    "vintage_share": TREATMENT_EXPOSURE,
    "arrears_share": TREATMENT_EXPOSURE,       # data-permitting (pipeline rarely has arrears)
    "filtered_share": TREATMENT_EXPOSURE,
    "joint_borrower_share": TREATMENT_EXPOSURE,
    # Weighted averages: expected balances as weights.
    "weighted_average": TREATMENT_EXPOSURE,
    "field_average": TREATMENT_EXPOSURE,       # Σ(v×p)/Σ(p): expected balance per expected loan
    # Largest single group by EXPECTED exposure is deterministic and meaningful.
    "largest_group_share": TREATMENT_EXPOSURE,
    # Whole-loan metrics: a fractional loan is never a maximum / a member of a
    # top-N. Expected is indicative only; the stress state includes whole
    # pipeline loans.
    "top_n_share": TREATMENT_SCENARIO,
    "field_maximum": TREATMENT_SCENARIO,
    "field_minimum": TREATMENT_SCENARIO,
    "largest_by_field_share": TREATMENT_SCENARIO,
    "distinct_group_count": TREATMENT_SCENARIO,
    # Absolute per-borrower count covenants have no meaningful
    # probability-weighted compliance status, and pipeline applicants carry no
    # governed borrower identity linkage to the funded book.
    "max_count_per_group": TREATMENT_UNSUPPORTED,
    "multi_loan_borrower_share": TREATMENT_UNSUPPORTED,
    # External references do not depend on the portfolio state.
    "external_index_ratio": TREATMENT_STATE_INDEPENDENT,
}

TREATMENT_NOTES = {
    TREATMENT_EXPOSURE: (
        "Expected state weights each pipeline case's exposure by its "
        "completion probability; numerator and denominator move together."),
    TREATMENT_SCENARIO: (
        "Whole-loan metric: the expected state is indicative only (a "
        "fractional loan is never a maximum or a count); the Full Pipeline "
        "state includes every active in-scope case at 100%."),
    TREATMENT_INDICATIVE: (
        "The expected value is indicative only for this metric family."),
    TREATMENT_STATE_INDEPENDENT: (
        "This test does not depend on the portfolio composition; its value "
        "is the same in every state."),
    TREATMENT_UNSUPPORTED: (
        "Forward-looking evaluation is not supported for this metric family; "
        "the funded value remains the only governed result."),
}

#: Pipeline metadata carried through the state frames for drivers/drill.
_PIPELINE_META_COLS = ("pipeline_stage", "completion_probability",
                      "completion_probability_source",
                      "expected_completion_month", "pipeline_case_identifier",
                      "application_identifier")

#: Pre-completion approximations for basis columns pipeline extracts lack.
#: An unfunded case's advance amount IS its initial balance, and its
#: application valuation IS its original valuation — disclosed, not silent.
_PIPELINE_BASIS_ALIASES = {
    "original_principal_balance": "current_outstanding_balance",
    "original_valuation_amount": "current_valuation_amount",
    # The pipeline extract's region field feeds the same readable-label
    # precedence the funded evaluators use.
    "collateral_geography": "geographic_region_obligor",
}

STATE_COMPONENT_COL = "__state_component__"


# --------------------------------------------------------------------------- #
# State frame construction
# --------------------------------------------------------------------------- #
def active_pipeline(pipeline_df: Optional[pd.DataFrame]
                    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """The ACTIVE in-scope pipeline population plus excluded counts by reason.

    In scope: rows the governed pipeline prep classifies ``pipeline_status ==
    "pipeline"`` (open KFI/APPLICATION/OFFER). Excluded and counted:
    withdrawn/cancelled (``withdrawn``), already funded (``funded`` — those
    loans live in the funded book), and unknown/out-of-scope stages.
    """
    if pipeline_df is None or pipeline_df.empty:
        return pd.DataFrame(), {}
    status = (pipeline_df["pipeline_status"].astype(str)
              if "pipeline_status" in pipeline_df.columns
              else pd.Series("unknown", index=pipeline_df.index))
    excluded: Dict[str, int] = {}
    for reason in ("withdrawn", "funded", "unknown"):
        n = int((status == reason).sum())
        if n:
            key = {"withdrawn": "withdrawn_or_cancelled",
                   "funded": "already_completed",
                   "unknown": "unknown_stage"}[reason]
            excluded[key] = n
    return pipeline_df[status == "pipeline"].copy(), excluded


def build_state_frame(funded_df: Optional[pd.DataFrame],
                      pipeline_df: Optional[pd.DataFrame],
                      state: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Combined evaluation frame for one forward-looking state.

    Funded rows carry probability 1.0; active pipeline rows carry the forecast
    engine's completion probability (expected state) or 1.0 (full-pipeline
    stress). Basis columns the pipeline lacks are aliased from their
    pre-completion equivalents, disclosed in the returned metadata.
    """
    if state not in (STATE_EXPECTED, STATE_FULL):
        raise ValueError(f"not a forward-looking state: {state!r}")
    parts: List[pd.DataFrame] = []
    meta: Dict[str, Any] = {"state": state, "aliasedBasisColumns": [],
                            "pipelineRowsIncluded": 0,
                            "pipelineRowsWithoutProbability": 0,
                            "excludedPipeline": {}}

    if funded_df is not None and not funded_df.empty:
        f = funded_df.copy()
        f[PROBABILITY_COL] = 1.0
        f[STATE_COMPONENT_COL] = "funded"
        parts.append(f)

    active, excluded = active_pipeline(pipeline_df)
    meta["excludedPipeline"] = excluded
    if not active.empty:
        p = active.copy()
        if state == STATE_EXPECTED:
            prob = (coerce_numeric(p["completion_probability"])
                    if "completion_probability" in p.columns
                    else pd.Series(float("nan"), index=p.index))
            meta["pipelineRowsWithoutProbability"] = int(prob.isna().sum())
            # A case with no governed probability contributes nothing to the
            # EXPECTED state (disclosed) — it is never guessed to 0.5 or 1.0.
            p = p[prob.notna() & (prob > 0)]
            p[PROBABILITY_COL] = coerce_numeric(p["completion_probability"])
        else:
            p[PROBABILITY_COL] = 1.0
        for target, source in _PIPELINE_BASIS_ALIASES.items():
            if source in p.columns and (
                    target not in p.columns or p[target].isna().all()):
                p[target] = p[source]
                meta["aliasedBasisColumns"].append(
                    f"{target} := {source} (pre-completion approximation)")
        p[STATE_COMPONENT_COL] = "pipeline"
        meta["pipelineRowsIncluded"] = int(len(p))
        parts.append(p)

    if not parts:
        return pd.DataFrame(), meta
    frame = pd.concat(parts, ignore_index=True, sort=False)
    return frame, meta


# --------------------------------------------------------------------------- #
# Per-test forward evaluation
# --------------------------------------------------------------------------- #
def treatment_for(lib: ConcentrationLibrary, test: ActiveTest) -> str:
    metric = lib.get(test.metric_id)
    if metric is None or metric.evaluator is None:
        return TREATMENT_UNSUPPORTED
    return FORECAST_TREATMENT.get(metric.evaluator, TREATMENT_UNSUPPORTED)


def _state_block(value: Optional[float], test: ActiveTest,
                 comp_status: str) -> Dict[str, Any]:
    return {
        "value": value,
        "status": comp_status,
        "headroom": _headroom(value, test.threshold, test.operator),
        "utilization": _utilization(value, test.threshold, test.operator),
        "breachAmount": _breach_amount(value, test.threshold, test.operator,
                                       comp_status),
    }


def evaluate_forward_states(
    funded_test_row: Dict[str, Any],
    test: ActiveTest,
    lib: ConcentrationLibrary,
    expected_frame: Optional[pd.DataFrame],
    full_frame: Optional[pd.DataFrame],
    *,
    external=None,
) -> Dict[str, Any]:
    """The forward-looking extension block for one already-evaluated test.

    ``funded_test_row`` is the untouched output of the funded evaluation —
    this function only ADDS the expected/full blocks, movement figures and
    treatment disclosure.
    """
    metric = lib.get(test.metric_id)
    treatment = treatment_for(lib, test)
    funded_value = funded_test_row.get("currentValue")
    out: Dict[str, Any] = {
        "forecastTreatment": treatment,
        "forecastTreatmentNote": TREATMENT_NOTES[treatment],
        "expected": None,
        "fullPipeline": None,
        "changeFundedToExpected": None,
        "changeFundedToFullPipeline": None,
        "expectedBreach": False,
        "fullPipelineBreach": False,
        "expectedNumerator": None,
        "expectedDenominator": None,
        "fullPipelineNumerator": None,
        "fullPipelineDenominator": None,
    }

    if metric is None or treatment == TREATMENT_UNSUPPORTED:
        return out
    if treatment == TREATMENT_STATE_INDEPENDENT:
        status = funded_test_row.get("status")
        for key in ("expected", "fullPipeline"):
            out[key] = _state_block(funded_value, test, status)
        return out
    if funded_test_row.get("status") in (STATUS_UNAVAILABLE,):
        # A test that cannot be evaluated on funded data cannot be evaluated
        # forward either (same missing fields) — stay honest, add nothing.
        return out

    def _run(frame: Optional[pd.DataFrame]) -> Optional[Any]:
        if frame is None or frame.empty:
            return None
        return evaluate_metric(frame, lib, metric, dict(test.parameters or {}),
                               external=external)

    # EXPECTED --------------------------------------------------------------- #
    if treatment == TREATMENT_EXPOSURE:
        comp = _run(expected_frame)
        if comp is not None and comp.data_status == DATA_OK:
            status = _status_for(comp.value, test.threshold, test.operator,
                                 test.warning_fraction)
            out["expected"] = _state_block(comp.value, test, status)
            out["expectedNumerator"] = comp.numerator_value
            out["expectedDenominator"] = comp.denominator_value
            out["expectedBreach"] = status == STATUS_BREACH
            if comp.value is not None and funded_value is not None:
                out["changeFundedToExpected"] = round(comp.value - funded_value, 4)
    else:  # scenario / indicative families: expected is indicative only
        out["expected"] = {"value": None, "status": "indicative_only",
                          "headroom": None, "utilization": None,
                          "breachAmount": None}

    # FULL PIPELINE (stress) ------------------------------------------------- #
    comp_full = _run(full_frame)
    if comp_full is not None and comp_full.data_status == DATA_OK:
        status = _status_for(comp_full.value, test.threshold, test.operator,
                             test.warning_fraction)
        out["fullPipeline"] = _state_block(comp_full.value, test, status)
        out["fullPipelineNumerator"] = comp_full.numerator_value
        out["fullPipelineDenominator"] = comp_full.denominator_value
        out["fullPipelineBreach"] = status == STATUS_BREACH
        if comp_full.value is not None and funded_value is not None:
            out["changeFundedToFullPipeline"] = round(
                comp_full.value - funded_value, 4)
    return out


# --------------------------------------------------------------------------- #
# Pipeline drivers
# --------------------------------------------------------------------------- #
def pipeline_drivers(test: ActiveTest, lib: ConcentrationLibrary,
                     expected_frame: Optional[pd.DataFrame],
                     funded_test_row: Dict[str, Any],
                     *, max_rows: int = 25,
                     external=None) -> Dict[str, Any]:
    """The pipeline cases contributing most to a test's expected movement.

    Deterministic and reconciling: the sum of the listed expected
    contributions equals expected numerator − funded numerator (both from the
    same evaluator run). The impact marker walks cases in contribution order
    against the FIXED expected denominator and marks the first case whose
    cumulative addition crosses the warning/breach line.
    """
    metric = lib.get(test.metric_id)
    if metric is None or treatment_for(lib, test) != TREATMENT_EXPOSURE:
        return {"available": False,
                "reason": ("Pipeline drivers are available for "
                           "exposure-weighted metric families only."),
                "drivers": []}
    if expected_frame is None or expected_frame.empty:
        return {"available": False,
                "reason": "No expected pipeline population is available.",
                "drivers": []}
    comp = evaluate_metric(expected_frame, lib, metric,
                           dict(test.parameters or {}), external=external)
    if comp.data_status != DATA_OK or comp.mask is None:
        return {"available": False,
                "reason": comp.notes or "The expected state could not be "
                                        "evaluated.",
                "drivers": []}

    frame = expected_frame
    is_pipeline = frame[STATE_COMPONENT_COL] == "pipeline"
    in_numerator = comp.mask.reindex(frame.index, fill_value=False) & is_pipeline
    rows = frame[in_numerator]

    bal_col = resolve_role_column(frame, lib, "balance_current")
    id_col = next((c for c in ("pipeline_case_identifier",
                               "application_identifier", "loan_id")
                   if c in frame.columns), None)
    dim_col = next(iter(comp.resolved_columns.values()), None)

    prob = coerce_numeric(rows[PROBABILITY_COL])
    bal = coerce_numeric(rows[bal_col]) if bal_col else pd.Series(0.0, index=rows.index)
    expected_contrib = bal * prob
    order = expected_contrib.sort_values(ascending=False, kind="mergesort").index

    funded_num = funded_test_row.get("numeratorValue") or 0.0
    expected_den = comp.denominator_value or 0.0
    threshold = test.threshold
    warning_line = (threshold * test.warning_fraction
                    if (threshold and test.operator == "max") else None)

    drivers: List[Dict[str, Any]] = []
    cumulative = float(funded_num)
    total_movement = float(expected_contrib.sum())
    tipped_warning = tipped_breach = False
    for idx in order:
        cumulative += float(expected_contrib.loc[idx])
        value_at = (round(cumulative / expected_den * 100.0, 2)
                    if expected_den else None)
        impact = ""
        if (test.operator == "max" and value_at is not None
                and threshold is not None):
            if not tipped_breach and value_at > threshold:
                impact, tipped_breach = "tips_breach", True
            elif not tipped_warning and warning_line is not None \
                    and value_at >= warning_line:
                impact, tipped_warning = "tips_warning", True
        drivers.append({
            "caseId": (str(rows.at[idx, id_col]) if id_col else f"case_{idx}"),
            "balance": round(float(bal.loc[idx]), 2),
            "stage": (str(rows.at[idx, "pipeline_stage"])
                      if "pipeline_stage" in rows.columns else None),
            "dimensionValue": (str(rows.at[idx, dim_col])
                               if dim_col and dim_col in rows.columns else None),
            "completionProbability": round(float(prob.loc[idx]), 4),
            "probabilitySource": (str(rows.at[idx, "completion_probability_source"])
                                  if "completion_probability_source" in rows.columns
                                  else None),
            "expectedContribution": round(float(expected_contrib.loc[idx]), 2),
            "fullContribution": round(float(bal.loc[idx]), 2),
            "expectedCompletionMonth": (
                str(rows.at[idx, "expected_completion_month"])
                if "expected_completion_month" in rows.columns
                and pd.notna(rows.at[idx, "expected_completion_month"]) else None),
            "impact": impact or None,
        })

    listed = drivers[:max_rows]
    listed_sum = round(sum(d["expectedContribution"] for d in listed), 2)
    return {
        "available": True,
        "testId": test.test_id,
        "displayName": test.display_name,
        "dimensionColumn": dim_col,
        "drivers": listed,
        "driverCount": len(drivers),
        "truncated": len(drivers) > max_rows,
        "expectedNumeratorMovement": round(total_movement, 2),
        "listedContribution": listed_sum,
        "topShareOfMovement": (round(listed_sum / total_movement, 4)
                               if total_movement else None),
        "reconciles": bool(abs(
            (comp.numerator_value or 0.0) - funded_num - total_movement) < 0.01
            or comp.numerator_value is None),
    }


# --------------------------------------------------------------------------- #
# Expected breach horizon
# --------------------------------------------------------------------------- #
def expected_breach_horizon(test: ActiveTest, lib: ConcentrationLibrary,
                            funded_df: Optional[pd.DataFrame],
                            expected_frame: Optional[pd.DataFrame],
                            funded_test_row: Dict[str, Any],
                            *, external=None) -> Optional[Dict[str, Any]]:
    """Earliest expected-completion period by which the cumulative expected
    contributions push the test past its threshold. Deterministic replay of
    the expected state month by month; requires completion-timing data. Only
    meaningful for exposure-weighted max tests; returns None otherwise."""
    metric = lib.get(test.metric_id)
    if (metric is None or treatment_for(lib, test) != TREATMENT_EXPOSURE
            or test.operator != "max" or test.threshold is None
            or expected_frame is None or expected_frame.empty):
        return None
    if "expected_completion_month" not in expected_frame.columns:
        return {"available": False,
                "reason": "No completion-timing data in the pipeline."}
    is_pipeline = expected_frame[STATE_COMPONENT_COL] == "pipeline"
    months = (expected_frame.loc[is_pipeline, "expected_completion_month"]
              .dropna().astype(str))
    if months.empty:
        return {"available": False,
                "reason": "No completion-timing data in the pipeline."}
    for month in sorted(months.unique()):
        upto = expected_frame[
            (~is_pipeline)
            | (expected_frame["expected_completion_month"].astype(str) <= month)]
        comp = evaluate_metric(upto, lib, metric, dict(test.parameters or {}),
                               external=external)
        if comp.data_status == DATA_OK and comp.value is not None \
                and comp.value > test.threshold:
            return {"available": True, "period": month}
    return {"available": True, "period": None}  # never crosses in the horizon


# --------------------------------------------------------------------------- #
# Emerging-risk intelligence (deterministic, rule-ranked)
# --------------------------------------------------------------------------- #
RISK_CURRENT_BREACH = "current_breach"
RISK_EXPECTED_BREACH = "expected_breach"
RISK_LOW_EXPECTED_HEADROOM = "expected_warning_low_headroom"
RISK_DETERIORATION = "material_deterioration"
RISK_STRESS_ONLY = "full_pipeline_only_breach"
RISK_LIMITATION = "data_or_methodology_limitation"

_RISK_RANK = {
    RISK_CURRENT_BREACH: 1,
    RISK_EXPECTED_BREACH: 2,
    RISK_LOW_EXPECTED_HEADROOM: 3,
    RISK_DETERIORATION: 4,
    RISK_STRESS_ONLY: 5,
    RISK_LIMITATION: 6,
}

#: Default policy knobs (percentage points for percent-unit tests).
DEFAULT_HEADROOM_BUFFER_PP = 1.0
DEFAULT_DETERIORATION_PP = 0.5


def _fmt(value: Optional[float], unit: Optional[str]) -> str:
    if value is None:
        return "unavailable"
    if unit == "percent":
        return f"{value:.1f}%"
    if unit == "count":
        return f"{value:.0f}"
    return f"{value:,.0f}"


def identify_emerging_risks(tests: List[Dict[str, Any]], *,
                            headroom_buffer_pp: float = DEFAULT_HEADROOM_BUFFER_PP,
                            deterioration_pp: float = DEFAULT_DETERIORATION_PP,
                            forecast_weak: bool = False
                            ) -> List[Dict[str, Any]]:
    """Rank issues over the evaluated three-state results.

    Pure rules, one entry per (test, category), ordered by the fixed rank then
    ascending expected headroom then name. The language model never chooses
    or reorders this list.
    """
    risks: List[Dict[str, Any]] = []

    def add(t: Dict[str, Any], category: str, statement: str,
            headroom: Optional[float] = None) -> None:
        risks.append({
            "category": category,
            "rank": _RISK_RANK[category],
            "testId": t.get("testId"),
            "displayName": t.get("displayName"),
            "statement": statement,
            "expectedHeadroom": headroom,
        })

    for t in tests:
        unit = t.get("unit")
        name = t.get("displayName")
        threshold = _fmt(t.get("threshold"), unit)
        expected = t.get("expected") or {}
        full = t.get("fullPipeline") or {}

        if t.get("status") == STATUS_BREACH:
            add(t, RISK_CURRENT_BREACH,
                f"{name} is in breach on the funded book: "
                f"{_fmt(t.get('currentValue'), unit)} against {threshold}.")
            continue
        if t.get("expectedBreach"):
            over = expected.get("breachAmount")
            horizon = (t.get("expectedBreachHorizon") or {}).get("period")
            statement = (
                f"{name} is expected to breach: funded "
                f"{_fmt(t.get('currentValue'), unit)} vs {threshold}; expected "
                f"{_fmt(expected.get('value'), unit)}"
                + (f" (+{over:.1f}pp over)" if over is not None and unit == "percent"
                   else "")
                + (f", crossing around {horizon}" if horizon else "") + ".")
            add(t, RISK_EXPECTED_BREACH, statement,
                headroom=expected.get("headroom"))
            continue
        exp_headroom = expected.get("headroom")
        if expected.get("status") == STATUS_WARNING and exp_headroom is not None \
                and unit == "percent" and exp_headroom <= headroom_buffer_pp:
            add(t, RISK_LOW_EXPECTED_HEADROOM,
                f"{name} has {exp_headroom:.1f}pp of expected headroom "
                f"({_fmt(t.get('currentValue'), unit)} → "
                f"{_fmt(expected.get('value'), unit)} vs {threshold}).",
                headroom=exp_headroom)
            continue
        change = t.get("percentagePointChange")
        adverse = (change is not None and unit == "percent"
                   and ((t.get("operator") == "max" and change >= deterioration_pp)
                        or (t.get("operator") == "min" and change <= -deterioration_pp)))
        if adverse and t.get("status") in (STATUS_PASS, STATUS_WARNING):
            add(t, RISK_DETERIORATION,
                f"{name} deteriorated {abs(change):.2f}pp since "
                f"{t.get('priorReportingDate') or 'the prior period'} "
                f"({_fmt(t.get('priorValue'), unit)} → "
                f"{_fmt(t.get('currentValue'), unit)}).")
            continue
        if t.get("fullPipelineBreach") and not t.get("expectedBreach") \
                and t.get("status") != STATUS_BREACH:
            add(t, RISK_STRESS_ONLY,
                f"{name} breaches only if ALL pipeline converts "
                f"({_fmt(t.get('currentValue'), unit)} → "
                f"{_fmt(full.get('value'), unit)} vs {threshold}) — a "
                "maximum-exposure scenario, not an expected outcome.")
            continue
        if t.get("status") in (STATUS_UNAVAILABLE, "insufficient_data"):
            add(t, RISK_LIMITATION,
                f"{name} cannot currently be calculated "
                f"({t.get('notes') or t.get('dataStatus')}).")

    if forecast_weak:
        risks.append({
            "category": RISK_LIMITATION, "rank": _RISK_RANK[RISK_LIMITATION],
            "testId": None, "displayName": None,
            "statement": ("The completion-trend model is running on "
                          "configured fallback rates for one or more stages — "
                          "the observed sample is below the sufficiency "
                          "floor, so expected results carry less weight."),
            "expectedHeadroom": None,
        })
    risks.sort(key=lambda r: (r["rank"],
                              r["expectedHeadroom"] if r["expectedHeadroom"]
                              is not None else float("inf"),
                              r["displayName"] or "~"))
    return risks


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def extend_with_forward_states(
    evaluated: Dict[str, Any],
    config: ActiveConfiguration,
    lib: ConcentrationLibrary,
    funded_df: Optional[pd.DataFrame],
    pipeline_df: Optional[pd.DataFrame],
    *,
    forecast_meta: Optional[Dict[str, Any]] = None,
    external=None,
    headroom_buffer_pp: float = DEFAULT_HEADROOM_BUFFER_PP,
) -> Dict[str, Any]:
    """Extend a funded evaluation (from ``evaluate_active_tests``) with the
    expected_forecast and full_pipeline states, drivers metadata, horizon and
    emerging risks. The funded fields are never modified."""
    active, _excluded = active_pipeline(pipeline_df)
    if active.empty:
        # No active in-scope pipeline: forward states would just restate the
        # funded numbers — say so explicitly instead of dressing funded up as
        # a forecast.
        evaluated["states"] = {
            "available": False,
            "reason": ("No active in-scope pipeline cases are available, so "
                       "the Expected Forecast and Full Pipeline states are "
                       "not evaluated. Funded results are unaffected."),
        }
        evaluated["emergingRisks"] = identify_emerging_risks(
            evaluated.get("tests", []),
            headroom_buffer_pp=headroom_buffer_pp)
        return evaluated

    expected_frame, expected_meta = build_state_frame(funded_df, pipeline_df,
                                                      STATE_EXPECTED)
    full_frame, full_meta = build_state_frame(funded_df, pipeline_df,
                                              STATE_FULL)
    tests_by_id = {t.test_id: t for t in config.tests}

    for row in evaluated.get("tests", []):
        test = tests_by_id.get(row.get("testId"))
        if test is None:
            continue
        forward = evaluate_forward_states(row, test, lib, expected_frame,
                                          full_frame, external=external)
        if forward.get("expectedBreach"):
            forward["expectedBreachHorizon"] = expected_breach_horizon(
                test, lib, funded_df, expected_frame, row, external=external)
        row.update(forward)

    forecast_weak = bool((forecast_meta or {}).get("stagesUsingConfigFallback"))
    risks = identify_emerging_risks(evaluated.get("tests", []),
                                    headroom_buffer_pp=headroom_buffer_pp,
                                    forecast_weak=forecast_weak)

    summary = evaluated.get("summary") or {}
    tests = evaluated.get("tests", [])
    summary["expectedBreaches"] = sum(1 for t in tests if t.get("expectedBreach"))
    summary["fullPipelineBreaches"] = sum(1 for t in tests
                                          if t.get("fullPipelineBreach"))
    summary["expectedWarnings"] = sum(
        1 for t in tests
        if (t.get("expected") or {}).get("status") == STATUS_WARNING)

    evaluated["emergingRisks"] = risks
    evaluated["states"] = {
        "available": not expected_frame.empty or not full_frame.empty,
        "expected": expected_meta,
        "fullPipeline": full_meta,
    }
    return evaluated


def compute_drivers_for_test(config: ActiveConfiguration,
                             lib: ConcentrationLibrary, test_id: str,
                             funded_df: Optional[pd.DataFrame],
                             pipeline_df: Optional[pd.DataFrame],
                             funded_test_row: Dict[str, Any],
                             *, external=None) -> Dict[str, Any]:
    test = next((t for t in config.tests if t.test_id == test_id), None)
    if test is None:
        return {"available": False,
                "reason": "That test is not in the active configuration.",
                "drivers": []}
    expected_frame, _meta = build_state_frame(funded_df, pipeline_df,
                                              STATE_EXPECTED)
    return pipeline_drivers(test, lib, expected_frame, funded_test_row,
                            external=external)
