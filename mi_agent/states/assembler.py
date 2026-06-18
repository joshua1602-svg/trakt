"""mi_agent.states.assembler — deterministic MI state assembly.

Phase 3 MI state assembler. Pure functions that consume Phase 2 ``SnapshotStore``
outputs (or already-loaded DataFrames) and the Phase 0B ``state_library.yaml`` /
route contracts, and produce analytical DataFrames (``StateResult``) for later MI
queries. They reuse the Phase 1 ``analytics_lib`` for cohorts and bucketing.

Deliberate constraints (Phase 3 scope):
  * no orchestration, no MI Agent runtime wiring, no LLM, no Azure, no charts;
  * no migration / temporal-trend runtime (Phase 4) and no risk monitor (Phase 5);
  * optional-field gaps yield structured issues, never crashes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from analytics_lib import cohort as _cohort
from analytics_lib import materialise_buckets as _materialise_buckets
from snapshot.model import SnapshotNotFoundError
from snapshot.store import SnapshotStore

from . import models as M
from .models import StateResult, make_issue
from .route_contracts import canonical_state, validate_state_for_route
from .selectors import SnapshotSelector

# Default reserved column names (match the Phase 2 reserved loan columns and the
# Phase 0B virtual semantic fields).
BALANCE_COL = "current_outstanding_balance"
FUNDED_STATUS_COL = "funded_status"
PIPELINE_STAGE_COL = "pipeline_stage"
FORECAST_BALANCE_COL = "forecast_funded_balance"
FORECAST_PROB_COL = "forecast_funding_probability"

Source = Union[pd.DataFrame, SnapshotStore]


# --------------------------------------------------------------------------- #
# Frame resolution
# --------------------------------------------------------------------------- #


def _resolve_frame(source: Source, selector: Optional[SnapshotSelector]
                   ) -> Tuple[pd.DataFrame, Dict[str, Any], List[dict]]:
    """Return ``(frame, provenance, issues)`` from a DataFrame or a store.

    A DataFrame source is used directly (copied). A store source requires a
    single-snapshot *selector*; a missing snapshot yields a ``missing_snapshot``
    issue and an empty frame rather than an exception.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy(), {"source": "dataframe"}, []

    if not isinstance(source, SnapshotStore):
        raise TypeError("source must be a pandas DataFrame or a SnapshotStore")
    if selector is None:
        raise ValueError("a SnapshotSelector is required for a SnapshotStore source")

    try:
        header = selector.resolve_single(store=source)
    except SnapshotNotFoundError as exc:
        return (pd.DataFrame(), {"source": "snapshot_store"},
                [make_issue(M.MISSING_SNAPSHOT, M.ERROR, str(exc))])
    frame = source.load_loans(header.snapshot_id)
    provenance = {"source": "snapshot_store", "snapshot_id": header.snapshot_id,
                  "reporting_date": header.reporting_date,
                  "client_id": header.client_id, "route": header.route}
    return frame, provenance, []


def _route_block(state_name: str, route: Optional[str],
                 routes_dir: Optional[Path]) -> Optional[dict]:
    if not route:
        return None
    return validate_state_for_route(state_name, route, routes_dir=routes_dir)


# --------------------------------------------------------------------------- #
# Pure funded / pipeline selection (no resolution, no route check)
# --------------------------------------------------------------------------- #


def _select_funded(frame: pd.DataFrame, funded_status_col: str,
                   pipeline_stage_col: str
                   ) -> Tuple[pd.Series, List[dict], str]:
    """Return ``(funded_mask, issues, method)``."""
    issues: List[dict] = []
    if funded_status_col in frame.columns:
        classified = frame[funded_status_col].map(M.classify_funded_value)
        return classified.fillna(False).astype(bool), issues, "funded_status"

    if pipeline_stage_col in frame.columns:
        issues.append(make_issue(
            M.MISSING_FUNDED_STATUS, M.INFO,
            f"{funded_status_col!r} absent; deriving funded set from "
            f"{pipeline_stage_col!r}", field=funded_status_col))
        stage = frame[pipeline_stage_col].astype(str).str.strip().str.lower()
        return stage.isin(M.FUNDED_STAGE_VALUES), issues, "pipeline_stage_derived"

    # Documented safe fallback: with neither field, the loaded frame *is* the
    # funded book (matches state_library: "Funded book ... v1 default").
    issues.append(make_issue(
        M.MISSING_FUNDED_STATUS, M.WARNING,
        f"neither {funded_status_col!r} nor {pipeline_stage_col!r} present; "
        f"treating all rows as funded (v1 default)", field=funded_status_col))
    return pd.Series(True, index=frame.index), issues, "fallback_all_funded"


def _select_pipeline(frame: pd.DataFrame, funded_status_col: str,
                     pipeline_stage_col: str
                     ) -> Tuple[pd.Series, List[dict], Optional[str]]:
    """Return ``(pipeline_mask, issues, method)``. Never falls back to
    'all rows are pipeline' — that is unsafe."""
    issues: List[dict] = []
    if funded_status_col in frame.columns:
        classified = frame[funded_status_col].map(M.classify_funded_value)
        return classified.eq(False).fillna(False), issues, "funded_status"

    if pipeline_stage_col in frame.columns:
        issues.append(make_issue(
            M.MISSING_FUNDED_STATUS, M.INFO,
            f"{funded_status_col!r} absent; deriving pipeline set from "
            f"{pipeline_stage_col!r}", field=funded_status_col))
        stage = frame[pipeline_stage_col].astype(str).str.strip().str.lower()
        present = frame[pipeline_stage_col].notna()
        return (~stage.isin(M.FUNDED_STAGE_VALUES)) & present, issues, \
            "pipeline_stage_derived"

    issues.append(make_issue(
        M.MISSING_PIPELINE_STAGE, M.WARNING,
        f"neither {pipeline_stage_col!r} nor {funded_status_col!r} present; "
        f"cannot identify pipeline records", field=pipeline_stage_col))
    return pd.Series(False, index=frame.index), issues, None


# --------------------------------------------------------------------------- #
# Core states
# --------------------------------------------------------------------------- #


def total_funded(source: Source, *, selector: Optional[SnapshotSelector] = None,
                 route: Optional[str] = None, balance_col: str = BALANCE_COL,
                 funded_status_col: str = FUNDED_STATUS_COL,
                 pipeline_stage_col: str = PIPELINE_STAGE_COL,
                 routes_dir: Optional[Path] = None) -> StateResult:
    """Funded book at a single selected snapshot."""
    block = _route_block("total_funded", route, routes_dir)
    if block:
        return StateResult("total_funded", pd.DataFrame(), [block],
                           {"assembled": False})

    frame, provenance, issues = _resolve_frame(source, selector)
    meta: Dict[str, Any] = {"assembled": True, **provenance}
    if frame.empty:
        issues.append(make_issue(M.EMPTY_STATE_FRAME, M.WARNING,
                                 "resolved snapshot frame is empty"))
        return StateResult("total_funded", frame, issues, meta)

    if balance_col not in frame.columns:
        issues.append(make_issue(M.MISSING_BALANCE_FIELD, M.WARNING,
                                 f"balance field {balance_col!r} not present",
                                 field=balance_col))

    mask, fissues, method = _select_funded(frame, funded_status_col,
                                           pipeline_stage_col)
    issues.extend(fissues)
    funded = frame[mask].copy()
    meta.update({"selection_method": method, "row_count": int(len(funded))})
    if funded.empty:
        issues.append(make_issue(M.EMPTY_STATE_FRAME, M.WARNING,
                                 "no funded rows selected"))
    return StateResult("total_funded", funded, issues, meta)


def total_pipeline(source: Source, *, selector: Optional[SnapshotSelector] = None,
                   route: Optional[str] = None, balance_col: str = BALANCE_COL,
                   funded_status_col: str = FUNDED_STATUS_COL,
                   pipeline_stage_col: str = PIPELINE_STAGE_COL,
                   routes_dir: Optional[Path] = None) -> StateResult:
    """In-pipeline (unfunded) records at a single selected snapshot."""
    block = _route_block("total_pipeline", route, routes_dir)
    if block:
        return StateResult("total_pipeline", pd.DataFrame(), [block],
                           {"assembled": False})

    frame, provenance, issues = _resolve_frame(source, selector)
    meta: Dict[str, Any] = {"assembled": True, **provenance}
    if frame.empty:
        issues.append(make_issue(M.EMPTY_STATE_FRAME, M.WARNING,
                                 "resolved snapshot frame is empty"))
        return StateResult("total_pipeline", frame, issues, meta)

    if balance_col not in frame.columns:
        issues.append(make_issue(M.MISSING_BALANCE_FIELD, M.WARNING,
                                 f"balance field {balance_col!r} not present",
                                 field=balance_col))

    mask, pissues, method = _select_pipeline(frame, funded_status_col,
                                             pipeline_stage_col)
    issues.extend(pissues)
    pipeline = frame[mask].copy()
    meta.update({"selection_method": method, "row_count": int(len(pipeline))})
    if pipeline.empty:
        issues.append(make_issue(M.EMPTY_STATE_FRAME, M.WARNING,
                                 "no pipeline rows selected"))
    return StateResult("total_pipeline", pipeline, issues, meta)


def total_forecast_funded(
    source: Source, *, selector: Optional[SnapshotSelector] = None,
    route: Optional[str] = None, balance_col: str = BALANCE_COL,
    funded_status_col: str = FUNDED_STATUS_COL,
    pipeline_stage_col: str = PIPELINE_STAGE_COL,
    forecast_balance_col: str = FORECAST_BALANCE_COL,
    forecast_prob_col: str = FORECAST_PROB_COL,
    include_unforecastable: bool = True,
    routes_dir: Optional[Path] = None,
) -> StateResult:
    """Funded book + expected-converted pipeline.

    The expected funded balance for a pipeline row is, in priority order:
      1. ``forecast_funded_balance`` where populated;
      2. ``current_outstanding_balance`` x ``forecast_funding_probability``
         where both are present.
    Probabilities are never invented. Pipeline rows with no usable forecast are
    flagged (``missing_forecast_probability``) and, by default, retained with a
    null ``forecast_contribution`` (set ``include_unforecastable=False`` to drop
    them from the forecast component).

    The output frame tags each row with ``state_component`` (``funded`` /
    ``forecast_pipeline``) and a numeric ``forecast_contribution``.
    """
    block = _route_block("total_forecast_funded", route, routes_dir)
    if block:
        return StateResult("total_forecast_funded", pd.DataFrame(), [block],
                           {"assembled": False})

    frame, provenance, issues = _resolve_frame(source, selector)
    meta: Dict[str, Any] = {"assembled": True, **provenance}
    if frame.empty:
        issues.append(make_issue(M.EMPTY_STATE_FRAME, M.WARNING,
                                 "resolved snapshot frame is empty"))
        return StateResult("total_forecast_funded", frame, issues, meta)

    has_balance = balance_col in frame.columns
    if not has_balance:
        issues.append(make_issue(M.MISSING_BALANCE_FIELD, M.WARNING,
                                 f"balance field {balance_col!r} not present",
                                 field=balance_col))

    funded_mask, fissues, fmethod = _select_funded(frame, funded_status_col,
                                                   pipeline_stage_col)
    pipe_mask, pissues, pmethod = _select_pipeline(frame, funded_status_col,
                                                   pipeline_stage_col)
    issues.extend(fissues)
    # Avoid duplicate funded-status issues already raised by _select_funded.
    issues.extend(i for i in pissues if i["code"] != M.MISSING_FUNDED_STATUS)

    funded = frame[funded_mask].copy()
    funded["state_component"] = "funded"
    funded["forecast_contribution"] = (
        pd.to_numeric(funded[balance_col], errors="coerce")
        if has_balance else pd.NA)

    pipeline = frame[pipe_mask].copy()
    pipeline["state_component"] = "forecast_pipeline"

    # Expected funded balance for the pipeline component.
    contribution = pd.Series(pd.NA, index=pipeline.index, dtype="object")
    have_fc_balance = (forecast_balance_col in pipeline.columns)
    fc_balance = (pd.to_numeric(pipeline[forecast_balance_col], errors="coerce")
                  if have_fc_balance else None)
    have_prob = forecast_prob_col in pipeline.columns
    prob = (pd.to_numeric(pipeline[forecast_prob_col], errors="coerce")
            if have_prob else None)
    bal = (pd.to_numeric(pipeline[balance_col], errors="coerce")
           if has_balance else None)

    n_unforecastable = 0
    for idx in pipeline.index:
        if have_fc_balance and pd.notna(fc_balance.loc[idx]):
            contribution.loc[idx] = float(fc_balance.loc[idx])
        elif (have_prob and bal is not None and pd.notna(prob.loc[idx])
              and pd.notna(bal.loc[idx])):
            contribution.loc[idx] = float(bal.loc[idx]) * float(prob.loc[idx])
        else:
            n_unforecastable += 1  # left as NA
    pipeline["forecast_contribution"] = contribution

    if n_unforecastable:
        issues.append(make_issue(
            M.MISSING_FORECAST_PROBABILITY, M.WARNING,
            f"{n_unforecastable} pipeline row(s) lack both "
            f"{forecast_balance_col!r} and a usable "
            f"{forecast_prob_col!r}+{balance_col!r}; "
            + ("retained with null forecast contribution"
               if include_unforecastable else "excluded from forecast"),
            field=forecast_prob_col, count=n_unforecastable))

    if not include_unforecastable:
        pipeline = pipeline[pipeline["forecast_contribution"].notna()].copy()

    out = pd.concat([funded, pipeline], ignore_index=True)
    forecast_total = pd.to_numeric(out["forecast_contribution"],
                                   errors="coerce").sum()
    meta.update({
        "funded_method": fmethod,
        "pipeline_method": pmethod,
        "funded_count": int(len(funded)),
        "pipeline_count": int(len(pipeline)),
        "forecastable_pipeline_count": int(len(pipeline) - 0
                                           if include_unforecastable
                                           else len(pipeline)),
        "unforecastable_pipeline_count": int(n_unforecastable),
        "forecast_funded_total": float(forecast_total),
        "row_count": int(len(out)),
    })
    if out.empty:
        issues.append(make_issue(M.EMPTY_STATE_FRAME, M.WARNING,
                                 "forecast-funded frame is empty"))
    return StateResult("total_forecast_funded", out, issues, meta)


# --------------------------------------------------------------------------- #
# Cohort states (point-in-time, frame-level; trend runtime is Phase 4)
# --------------------------------------------------------------------------- #


def _translate_cohort_issue(issue: dict) -> dict:
    """Map an analytics_lib.cohort issue code onto the state issue vocabulary."""
    code = issue.get("code")
    if code == "unavailable_field":
        return make_issue(M.MISSING_REQUIRED_STATE_FIELD, M.ERROR,
                          issue.get("message", ""), field=issue.get("field"))
    if code == "invalid_date":
        return make_issue(M.INVALID_DATE, M.WARNING, issue.get("message", ""),
                          field=issue.get("field"), count=issue.get("count"))
    return issue


def cohort_by_date(
    source: Source, *, selector: Optional[SnapshotSelector] = None,
    route: Optional[str] = None, date_field: str = "origination_date",
    period: str = "Y", balance_col: str = BALANCE_COL,
    as_of: Any = None, mob_start_field: Optional[str] = None,
    segment_field: Optional[str] = None, buckets: Optional[List[str]] = None,
    state_name: str = "cohort_by_date", routes_dir: Optional[Path] = None,
) -> StateResult:
    """Augment the (funded) loan frame with a cohort-period column for
    *date_field*, optional months-on-book, optional segmentation column check,
    and optional bucket materialisation. Frame-level output for later
    stratification — no aggregation/charting here.
    """
    block = _route_block(state_name, route, routes_dir)
    if block:
        return StateResult(state_name, pd.DataFrame(), [block],
                           {"assembled": False})

    frame, provenance, issues = _resolve_frame(source, selector)
    meta: Dict[str, Any] = {"assembled": True, "date_field": date_field,
                            "period": period, **provenance}
    if frame.empty:
        issues.append(make_issue(M.EMPTY_STATE_FRAME, M.WARNING,
                                 "resolved snapshot frame is empty"))
        return StateResult(state_name, frame, issues, meta)

    if date_field not in frame.columns:
        issues.append(make_issue(
            M.MISSING_REQUIRED_STATE_FIELD, M.ERROR,
            f"cohort date field {date_field!r} not present; cannot build cohort",
            field=date_field))
        return StateResult(state_name, frame, issues, meta)

    cohort_col = f"{date_field}_cohort"
    out, citems = _cohort.add_cohort_period(frame, date_field, period=period,
                                            out_col=cohort_col)
    issues.extend(_translate_cohort_issue(i) for i in citems)
    meta["cohort_column"] = cohort_col

    # Optional months-on-book against a reporting/as-of date.
    if as_of is not None:
        start = mob_start_field or ("funding_date" if "funding_date" in out.columns
                                    else date_field)
        if start in out.columns:
            out, missues = _cohort.months_on_book(out, start, as_of,
                                                  out_col="months_on_book")
            issues.extend(_translate_cohort_issue(i) for i in missues)
            meta["months_on_book_start"] = start
        else:
            issues.append(make_issue(
                M.MISSING_OPTIONAL_STATE_FIELD, M.WARNING,
                f"months_on_book requested but start field {start!r} absent",
                field=start))

    # Optional segmentation column presence check (graceful).
    if segment_field is not None and segment_field not in out.columns:
        issues.append(make_issue(
            M.MISSING_OPTIONAL_STATE_FIELD, M.WARNING,
            f"segmentation field {segment_field!r} not present; segment view "
            f"unavailable", field=segment_field))
    elif segment_field is not None:
        meta["segment_field"] = segment_field

    # Optional bucket materialisation via analytics_lib.
    if buckets:
        out, bissues, applied = _materialise_buckets(out, buckets=buckets)
        meta["buckets_applied"] = applied
        for bi in bissues:
            sev = bi.get("severity", M.WARNING)
            issues.append(make_issue(
                M.UNAVAILABLE_DIMENSION if bi.get("code") == "unavailable_field"
                else bi.get("code", "bucket_issue"),
                sev, bi.get("message", ""), field=bi.get("field"),
                count=bi.get("count")))

    meta["row_count"] = int(len(out))
    return StateResult(state_name, out, issues, meta)


def _cohort_segment(source: Source, *, segment_field: str, state_name: str,
                    date_field: str = "origination_date", **kwargs) -> StateResult:
    return cohort_by_date(source, date_field=date_field,
                          segment_field=segment_field, state_name=state_name,
                          **kwargs)


def cohort_by_portfolio(source: Source, **kwargs) -> StateResult:
    return _cohort_segment(source, segment_field="portfolio_id",
                           state_name="cohort_by_portfolio", **kwargs)


def cohort_by_spv(source: Source, **kwargs) -> StateResult:
    return _cohort_segment(source, segment_field="spv_id",
                           state_name="cohort_by_spv", **kwargs)


def cohort_by_acquired_portfolio(source: Source, **kwargs) -> StateResult:
    return _cohort_segment(source, segment_field="acquired_portfolio_id",
                           state_name="cohort_by_acquired_portfolio", **kwargs)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

_DATE_FIELD_ALIASES = {
    "cohort_by_origination_date": "origination_date",
    "cohort_by_funding_date": "funding_date",
    "cohort_by_acquisition_date": "acquisition_date",
}

_DISPATCH = {
    "total_funded": total_funded,
    "total_pipeline": total_pipeline,
    "total_forecast_funded": total_forecast_funded,
    "cohort_by_date": cohort_by_date,
    "cohort_by_portfolio": cohort_by_portfolio,
    "cohort_by_spv": cohort_by_spv,
    "cohort_by_acquired_portfolio": cohort_by_acquired_portfolio,
}


def assemble_state(state_name: str, source: Source, **kwargs) -> StateResult:
    """Assemble a state by name, resolving descriptive cohort aliases.

    ``cohort_by_origination_date`` / ``cohort_by_funding_date`` /
    ``cohort_by_acquisition_date`` map to ``cohort_by_date`` with the matching
    ``date_field`` (and keep their descriptive ``state_name`` in the result).
    """
    if state_name in _DATE_FIELD_ALIASES:
        kwargs.setdefault("date_field", _DATE_FIELD_ALIASES[state_name])
        kwargs.setdefault("state_name", state_name)
        return cohort_by_date(source, **kwargs)

    fn = _DISPATCH.get(state_name)
    if fn is None:
        return StateResult(state_name, pd.DataFrame(),
                           [make_issue(M.UNAVAILABLE_STATE, M.ERROR,
                                       f"unknown state {state_name!r}")],
                           {"assembled": False})
    return fn(source, **kwargs)
