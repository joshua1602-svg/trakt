"""mi_agent_api/evolution.py

Funded / pipeline / forecast EVOLUTION (time series) across the governed monthly
funded runs and weekly pipeline extracts already produced by onboarding.

This module REUSES the existing loaders — ``snapshots`` for funded central tapes
and ``pipeline_contract`` for the governed weekly pipeline extracts — and never
re-implements raw onboarding discovery. Each period carries its own reconciliation
(records / balance / coverage) and lineage (run id, reporting date, source file),
matching the point-in-time MI standard.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from analytics_lib.dates import coerce_dates
from analytics_lib.numeric import coerce_numeric
from mi_agent.mi_dataset_profile import PERCENT_POINTS, percent_storage_scale

from . import snapshots as snap
from . import pipeline_contract as pipeline_mod

_BALANCE = "current_outstanding_balance"
# Funded breakdown dimensions exposed over time (kept small + governed).
_FUNDED_BREAKDOWN_DIMS = {
    "broker": "broker_channel",
    "region": "geographic_region_obligor",
    "ltv_bucket": "ltv_bucket",
}
MISSING_BUCKET = "Unknown / Missing"


# --------------------------------------------------------------------------- #
# Small metric helpers (balance-weighted, missing-aware) — mirror snapshots.
# --------------------------------------------------------------------------- #
def _bal_sum(df: pd.DataFrame, col: str = _BALANCE) -> Optional[float]:
    if col not in df.columns:
        return None
    return float(coerce_numeric(df[col]).sum())


def _weighted_avg(df: pd.DataFrame, value_col: str, weight_col: str = _BALANCE) -> Optional[float]:
    if value_col not in df.columns or weight_col not in df.columns:
        return None
    v = coerce_numeric(df[value_col])
    w = coerce_numeric(df[weight_col])
    mask = v.notna() & w.notna()
    denom = float(w[mask].sum())
    if denom == 0:
        return None
    return round(float((v[mask] * w[mask]).sum() / denom), 4)


def _simple_avg(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df.columns:
        return None
    v = coerce_numeric(df[col])
    return round(float(v.mean()), 4) if v.notna().any() else None


def _reconciliation(df: pd.DataFrame, dataset: str, run_id: str,
                    required: List[str]) -> Dict[str, Any]:
    total_n = int(len(df))
    total_bal = _bal_sum(df)
    missing_measure = [c for c in required if c not in df.columns]
    return {
        "dataset": dataset,
        "run_id": run_id,
        "total_records": total_n,
        "total_balance": (round(total_bal, 2) if total_bal is not None else None),
        "records_included": total_n,
        "balance_included": (round(total_bal, 2) if total_bal is not None else None),
        "records_excluded_missing": 0,
        "balance_excluded_missing": 0.0 if total_bal is not None else None,
        "coverage_by_balance_pct": 100.0 if total_bal else None,
        "missing_dimension_fields": [],
        "missing_measure_fields": missing_measure,
        "filters": {},
    }


def _breakdown(df: pd.DataFrame, dim_col: str, value_col: str = _BALANCE
               ) -> List[Dict[str, Any]]:
    """``[{key, value}]`` summing ``value_col`` by ``dim_col``; missing keys go to
    an explicit Unknown / Missing bucket so the breakdown reconciles to the total."""
    if dim_col not in df.columns or value_col not in df.columns:
        return []
    keys = df[dim_col].astype(object)
    blank = ~keys.notna() | keys.astype(str).str.strip().isin(["", "nan", "None", "NaT"])
    keys = keys.where(~blank, MISSING_BUCKET).astype(str)
    grp = coerce_numeric(df[value_col]).groupby(keys).sum()
    return [{"key": str(k), "value": round(float(v), 2)} for k, v in grp.items()]


# --------------------------------------------------------------------------- #
# Funded evolution
# --------------------------------------------------------------------------- #
def _runs_up_to(output_root: str | os.PathLike, client_id: str,
                to_run_id: Optional[str]) -> List[Dict[str, Any]]:
    disc = snap.discover_snapshots(output_root)
    pf = next((p for p in disc.get("portfolios", []) if p.get("client_id") == client_id), None)
    runs = list(pf.get("runs", [])) if pf else []
    if to_run_id:
        cut = next((i for i, r in enumerate(runs) if r["run_id"] == to_run_id), None)
        if cut is not None:
            runs = runs[: cut + 1]
    return runs


def assemble_funded_evolution(frames: List[Dict[str, Any]], client_id: str,
                              to_run_id: Optional[str] = None,
                              breakdowns: Optional[List[str]] = None,
                              *, lineage: Optional[Dict[str, Any]] = None
                              ) -> Dict[str, Any]:
    """Build the funded evolution series from an ordered list of prepared run
    frames — ``[{run_id, reporting_date, df, source}]`` (oldest → newest).

    Shared by the on-disk tape path and the blob platform-canonical path so the
    metric/reconciliation/breakdown shape is IDENTICAL regardless of source."""
    required = [_BALANCE, "current_loan_to_value", "current_interest_rate",
                "youngest_borrower_age"]
    want_breakdowns = breakdowns or ["broker", "region", "ltv_bucket"]

    periods: List[Dict[str, Any]] = []
    run_ids: List[str] = []
    dates: List[Optional[str]] = []
    sources: List[Optional[str]] = []
    bd_series: Dict[str, List[Dict[str, Any]]] = {b: [] for b in want_breakdowns}

    for fr in frames:
        run_id = fr["run_id"]
        df = fr["df"]
        if df is None:
            continue
        rdate = fr.get("reporting_date") or snap.infer_reporting_date(run_id, df)
        source = fr.get("source")
        run_ids.append(run_id)
        dates.append(rdate)
        sources.append(source)
        periods.append({
            "run_id": run_id,
            "reporting_date": rdate,
            "period": (rdate or run_id)[:7],
            "metrics": {
                "funded_balance": _bal_sum(df),
                "loan_count": int(len(df)),
                "wa_ltv": _weighted_avg(df, "current_loan_to_value"),
                "wa_interest_rate": _weighted_avg(df, "current_interest_rate"),
                "avg_borrower_age": _simple_avg(df, "youngest_borrower_age"),
            },
            "reconciliation": _reconciliation(df, "funded", run_id, required),
            "source_file": source,
        })
        for b in want_breakdowns:
            dim_col = _FUNDED_BREAKDOWN_DIMS.get(b)
            if dim_col:
                for row in _breakdown(df, dim_col):
                    bd_series[b].append({"period": (rdate or run_id)[:7], **row})

    return {
        "dataset": "funded",
        "portfolioId": client_id,
        "toRunId": to_run_id,
        "availableRunIds": run_ids,
        "reportingDates": dates,
        "sourceFiles": sources,
        "periods": periods,
        "breakdowns": bd_series,
        "lineage": lineage or {
            "source": "governed monthly central lender tapes (18_central_lender_tape.csv)",
            "metric": "funded book actuals per reporting month",
            "note": "Each period is an independent funded run; no cross-run merge.",
        },
        "singlePeriod": len(periods) <= 1,
    }


def funded_frames(output_root: str | os.PathLike, client_id: str,
                  to_run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ordered prepared funded run frames ``[{run_id, reporting_date, df, source}]``
    (oldest → newest), up to ``to_run_id`` inclusive.

    Blob-aware: the on-disk tape walk (``snap.discover_snapshots``) is
    filesystem-only and cannot enumerate a ``blob://`` platform root, so on such
    a root it returns ZERO periods (the cause of the "no reporting periods" / "£0"
    failures). On a blob root, build from the dated platform canonicals (the same
    source that powers ``/mi/evolution/funded``); fall back to the tape walk on any
    error. Shared by funded_evolution and funded_bridge so both see identical
    periods regardless of source."""
    from . import platform_snapshots_blob as _blob
    if _blob.is_blob_root(output_root):
        try:
            from apps.blob_trigger_app.storage import open_storage
            from .funded_prep import prepare_funded_mi_dataset
            return _blob.build_funded_evolution_frames(
                str(output_root), open_storage(), client_id, to_run_id,
                prepare_funded_mi_dataset)
        except Exception:  # noqa: BLE001 - never break the series on a blob error
            pass
    frames: List[Dict[str, Any]] = []
    for run in _runs_up_to(output_root, client_id, to_run_id):
        run_id = run["run_id"]
        tape = snap.resolve_tape_path(output_root, client_id, run_id)
        if tape is None:
            continue
        try:
            df, _rep = snap.load_prepared_run(tape)
        except Exception:  # noqa: BLE001 - a bad tape never breaks the series
            continue
        frames.append({
            "run_id": run_id,
            "reporting_date": run.get("reporting_date") or snap.infer_reporting_date(run_id, df),
            "df": df,
            "source": str(tape),
        })
    return frames


def funded_evolution(output_root: str | os.PathLike, client_id: str,
                     to_run_id: Optional[str] = None,
                     breakdowns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Funded time series across monthly runs up to ``to_run_id`` (inclusive)."""
    return assemble_funded_evolution(
        funded_frames(output_root, client_id, to_run_id),
        client_id, to_run_id, breakdowns)


# --------------------------------------------------------------------------- #
# Funded balance BRIDGE (attribution waterfall between two periods)
# --------------------------------------------------------------------------- #
def _period_label(fr: Dict[str, Any]) -> str:
    rd = fr.get("reporting_date") or fr.get("run_id")
    return str(rd)[:7] if rd else str(fr.get("run_id"))


def _scope_frame_lens(df, lens_filters: Optional[Dict[str, str]]):
    """Narrow a funded frame to a source-portfolio lens (consolidated = no
    filter; a cohort/type lens filters ``source_portfolio_id``/``_type``).
    Case/whitespace-insensitive; a filter on an absent column is a no-op."""
    if not lens_filters or df is None:
        return df
    work = df
    for col, val in lens_filters.items():
        if col in work.columns:
            work = work[work[col].astype(str).str.strip().str.casefold()
                        == str(val).strip().casefold()]
    return work


_MISSING_TOKENS = {"", "nan", "none", "nat", "<na>"}


def _group_balance(df, col: str) -> Dict[str, float]:
    """Funded balance summed by a dimension column; blank/NaN → 'Unknown / Missing'."""
    if df is None or col not in df.columns:
        return {}
    s = df[col].astype(str).str.strip()
    s = s.mask(s.str.casefold().isin(_MISSING_TOKENS), "Unknown / Missing")
    bal = coerce_numeric(df[_BALANCE])
    grp = bal.groupby(s).sum()
    return {str(k): float(v) for k, v in grp.items()}


def funded_bridge(output_root: str | os.PathLike, client_id: str,
                  dimension_col, *, start_period: Optional[str] = None,
                  to_run_id: Optional[str] = None,
                  lens_filters: Optional[Dict[str, str]] = None,
                  lens_label: str = "Total", top_n: int = 8) -> Dict[str, Any]:
    """Attribution bridge: opening funded balance (start period) → per-category
    change over ``dimension_col`` → closing funded balance (the LATEST period, or
    ``to_run_id``). The per-category deltas sum EXACTLY to (close − open), so the
    waterfall reconciles to the book. ``lens_filters`` scopes the frames for a
    consolidated (None) vs cohort/type view.

    ``dimension_col`` may be a single column or an ordered list of candidate
    columns (e.g. the region family) — the first one actually present in the data
    is used, so attribution works regardless of which column the tape carries."""
    scoped: List[Dict[str, Any]] = []
    for fr in funded_frames(output_root, client_id, to_run_id):
        d = _scope_frame_lens(fr.get("df"), lens_filters)
        if d is not None and len(d):
            scoped.append({**fr, "df": d})
    if len(scoped) < 2:
        return {"available": False, "lens": lens_label,
                "reason": "at least two funded reporting periods are needed for a bridge"}

    # Resolve the dimension column data-aware from the candidate(s).
    candidates = [dimension_col] if isinstance(dimension_col, str) else list(dimension_col or [])
    present_cols = set().union(*[set(f["df"].columns) for f in scoped]) if scoped else set()
    col = next((c for c in candidates if c in present_cols), candidates[0] if candidates else None)
    if not col:
        return {"available": False, "lens": lens_label,
                "reason": "no attribution dimension is available in the funded data"}

    end = scoped[-1]                       # the latest period is always the close
    start = None
    if start_period:
        sp = str(start_period)[:7]
        start = next((f for f in scoped if _period_label(f) == sp), None)
    if start is None or _period_label(start) == _period_label(end):
        start = scoped[0]                  # default: earliest available period
    if _period_label(start) == _period_label(end):
        return {"available": False, "lens": lens_label,
                "reason": "the start and latest period resolve to the same period"}

    a = _group_balance(start["df"], col)
    b = _group_balance(end["df"], col)
    cats = set(a) | set(b)
    contribs = [{"category": c, "start": round(a.get(c, 0.0), 2),
                 "end": round(b.get(c, 0.0), 2),
                 "delta": round(b.get(c, 0.0) - a.get(c, 0.0), 2)} for c in cats]
    contribs.sort(key=lambda r: abs(r["delta"]), reverse=True)
    open_total = round(sum(a.values()), 2)
    close_total = round(sum(b.values()), 2)

    # Top-N contributors by absolute movement + an aggregated "Other" so a
    # many-category bridge stays legible AND still reconciles (Other carries the
    # residual delta).
    if top_n and len(contribs) > top_n:
        head, tail = contribs[:top_n], contribs[top_n:]
        head.append({"category": "Other", "isOther": True, "count": len(tail),
                     "start": round(sum(r["start"] for r in tail), 2),
                     "end": round(sum(r["end"] for r in tail), 2),
                     "delta": round(sum(r["delta"] for r in tail), 2)})
        contribs = head

    return {
        "available": True,
        "dimensionCol": col,
        "lens": lens_label,
        "start": {"period": _period_label(start),
                  "reporting_date": start.get("reporting_date"), "total": open_total},
        "end": {"period": _period_label(end),
                "reporting_date": end.get("reporting_date"), "total": close_total},
        "netChange": round(close_total - open_total, 2),
        "contributions": contribs,
    }


# --------------------------------------------------------------------------- #
# Funded cohort PROGRESSION (static-pool seasoning across reporting periods)
# --------------------------------------------------------------------------- #
_VALUATION_COLS = ("indexed_valuation_amount", "current_valuation_amount",
                   "indexed_value", "original_valuation_amount")
_ORIG_DATE = "origination_date"
_VINTAGE = "vintage_year"


def _pct_fraction(df, col: str) -> Optional[float]:
    """Balance-weighted average of a percent column, normalised to a FRACTION so
    the UI's ×100 formatter renders it correctly (the tape stores LTV as a
    fraction but the interest rate in points)."""
    wavg = _weighted_avg(df, col)
    if wavg is None:
        return None
    if col in df.columns and percent_storage_scale(df[col]) == PERCENT_POINTS:
        return round(wavg / 100.0, 6)
    return wavg


def _nneg_metrics(df) -> Dict[str, Any]:
    """NNEG (no-negative-equity-guarantee) exposure/headroom for a lifetime book:
    exposure = Σ max(0, balance − property value); headroom% = 1 − balance/value
    (balance-weighted). Empty when no valuation column is present."""
    val_col = next((c for c in _VALUATION_COLS if c in df.columns), None)
    if val_col is None:
        return {}
    bal = coerce_numeric(df[_BALANCE])
    val = coerce_numeric(df[val_col])
    mask = bal.notna() & val.notna() & (val > 0)
    if not bool(mask.any()):
        return {}
    b, v = bal[mask], val[mask]
    exposure = float((b - v).clip(lower=0).sum())
    vsum = float(v.sum())
    return {
        "nneg_exposure": round(exposure, 2),
        "nneg_headroom": round(float((v - b).sum()), 2),
        "nneg_headroom_pct": (round(1.0 - float(b.sum()) / vsum, 6) if vsum else None),
    }


def _origination_labels(df, grain: str = "Y"):
    """Per-row origination-cohort label at the requested grain (Y / Q / M), from
    ``origination_date`` (else ``vintage_year`` for year grain). None if neither."""
    if _ORIG_DATE in df.columns:
        od = coerce_dates(df[_ORIG_DATE])
        if od.notna().any():
            g = (grain or "Y").upper()
            if g == "Q":
                return (od.dt.year.astype("Int64").astype(str) + "-Q"
                        + od.dt.quarter.astype("Int64").astype(str))
            if g == "M":
                return od.dt.strftime("%Y-%m")
            return od.dt.year.astype("Int64").astype(str)
    if _VINTAGE in df.columns and df[_VINTAGE].notna().any():
        return df[_VINTAGE].astype("Int64").astype(str)
    return None


def funded_cohort_progression(output_root: str | os.PathLike, client_id: str, *,
                              lens_filters: Optional[Dict[str, str]] = None,
                              lens_label: str = "Total",
                              vintage: Optional[str] = None, grain: str = "Y",
                              to_run_id: Optional[str] = None) -> Dict[str, Any]:
    """Static-pool progression: how a cohort's funded metrics (balance, loan
    count, WA LTV, WA rate, NNEG exposure/headroom) evolve ACROSS reporting
    periods. The cohort is defined by a source-portfolio lens (Total / direct /
    acquired / a cohort id like ``acquired_001``) AND, optionally, an origination
    ``vintage`` at the chosen ``grain`` (Y/Q/M) — so "acquired_001 loans
    originated in 2023" is a first-class cohort even within the consolidated book."""
    vintage_filterable = True
    periods: List[Dict[str, Any]] = []
    for fr in funded_frames(output_root, client_id, to_run_id):
        d = _scope_frame_lens(fr.get("df"), lens_filters)
        if d is None:
            continue
        if vintage:
            labels = _origination_labels(d, grain)
            if labels is None:
                vintage_filterable = False
                d = d.iloc[0:0]
            else:
                d = d[labels.astype(str) == str(vintage)]
        metrics: Dict[str, Any] = {
            "funded_balance": _bal_sum(d),
            "loan_count": int(len(d)),
            "wa_ltv": _pct_fraction(d, "current_loan_to_value"),
            "wa_interest_rate": _pct_fraction(d, "current_interest_rate"),
            "avg_borrower_age": _simple_avg(d, "youngest_borrower_age"),
        }
        metrics.update(_nneg_metrics(d))
        periods.append({
            "period": _period_label(fr),
            "reporting_date": fr.get("reporting_date"),
            "loanCount": int(len(d)),
            "metrics": metrics,
        })

    available = any(p["loanCount"] for p in periods)
    reason = None
    if not available:
        reason = ("no loans match this cohort in any reporting period"
                  if vintage_filterable else
                  "origination vintage is not available on the funded tape")
    metric_keys = ["funded_balance", "loan_count", "wa_ltv", "wa_interest_rate",
                   "avg_borrower_age"]
    if any("nneg_exposure" in p["metrics"] for p in periods):
        metric_keys += ["nneg_exposure", "nneg_headroom", "nneg_headroom_pct"]
    return {
        "dataset": "cohort_progression",
        "portfolioId": client_id,
        "available": available,
        "reason": reason,
        "lens": lens_label,
        "vintage": vintage,
        "grain": grain,
        "metricsAvailable": metric_keys,
        "periods": periods,
        "singlePeriod": len([p for p in periods if p["loanCount"]]) <= 1,
        "lineage": {
            "source": "governed funded reporting periods (static pool)",
            "metric": "cohort funded metrics per reporting period",
            "note": ("Static-pool seasoning: the SAME cohort (source portfolio "
                     "± origination vintage) tracked across reporting periods."),
        },
    }


# --------------------------------------------------------------------------- #
# Pipeline evolution (governed weekly extracts)
# --------------------------------------------------------------------------- #
def pipeline_evolution(pipeline_root: str | os.PathLike, client_id: str,
                       to_run_id: Optional[str] = None, *,
                       historical_model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pipeline time series across the governed UNIQUE weekly extracts.

    When a governed ``historical_model`` is supplied the weighted-expected-funded
    amount is weighted by the SAME empirical stage completion rates used by the
    forecast bridge (falling back to configured rates only where history is thin),
    so the scale-up 'weighted expected pipeline' matches the Forecast tab rather
    than silently using the config-only fallback."""
    inv = pipeline_mod.weekly_extract_inventory(pipeline_root, client_id)
    extracts = inv.get("extracts", [])
    cut_ym = pipeline_mod._year_month(str(to_run_id)) if to_run_id else None

    periods: List[Dict[str, Any]] = []
    by_stage: List[Dict[str, Any]] = []
    sources: List[str] = []
    dates: List[Optional[str]] = []

    for ext in extracts:
        edate = ext.get("pipeline_extract_date")
        if cut_ym and edate and edate[:7] > cut_ym:
            continue
        try:
            df, report = pipeline_mod.load_prepared_pipeline(
                ext, historical_model=historical_model)
        except Exception:  # noqa: BLE001
            continue
        amount = report.get("total_pipeline_amount")
        weighted = report.get("weighted_expected_funded_amount")
        sources.append(ext.get("source_file", ""))
        dates.append(edate)
        periods.append({
            "extract_date": edate,
            "period": (edate or "")[:7],
            "week": edate,
            "metrics": {
                "pipeline_amount": (round(float(amount), 2) if amount is not None else None),
                "pipeline_case_count": int(report.get("row_count", len(df))),
                "weighted_expected_funded_amount": (round(float(weighted), 2)
                                                    if weighted is not None else None),
            },
            "reconciliation": {
                "dataset": "pipeline",
                "extract_date": edate,
                "total_records": int(report.get("row_count", len(df))),
                "total_balance": (round(float(amount), 2) if amount is not None else None),
                "coverage_by_balance_pct": 100.0,
                "missing_measure_fields": [],
                "filters": {},
            },
            "source_file": ext.get("source_file", ""),
        })
        # Pipeline amount AND case count by stage for this extract (multi-line over
        # time, day-level dates). Both metrics are emitted so the UI can chart
        # amount or count, and derive Application/Offer/Completion conversion.
        if "pipeline_stage" in df.columns:
            stage_str = df["pipeline_stage"].astype(str)
            amt = (coerce_numeric(df[_BALANCE]).groupby(stage_str).sum()
                   if _BALANCE in df.columns else None)
            cnt = stage_str.groupby(stage_str).size()
            for stage, n in cnt.items():
                if str(stage).strip() and str(stage) not in ("nan", "None"):
                    val = float(amt.get(stage, 0.0)) if amt is not None else None
                    by_stage.append({
                        "period": (edate or ""), "week": edate, "stage": str(stage),
                        "value": (round(val, 2) if val is not None else None),
                        "count": int(n)})

    return {
        "dataset": "pipeline",
        "portfolioId": client_id,
        "toRunId": to_run_id,
        "availableExtractDates": dates,
        "sourceFiles": sources,
        "sourceFilesScanned": inv.get("sourceFilesScanned"),
        "uniqueWeeklyExtractsUsed": inv.get("uniqueWeeklyExtractsUsed"),
        "periods": periods,
        "byStage": by_stage,
        "lineage": {
            "source": "governed weekly pipeline extracts (deduplicated)",
            "metric": "origination pipeline amount / weighted expected funded per extract",
            "primarySourcePreference": inv.get("primarySourcePreference"),
        },
        "singlePeriod": len(periods) <= 1,
    }


# --------------------------------------------------------------------------- #
# Weekly origination funnel trends (KFI / Application / Offer / Completion)
# --------------------------------------------------------------------------- #
_FUNNEL_STAGES = ("KFI", "APPLICATION", "OFFER", "COMPLETED")
_FUNNEL_LABELS = {"KFI": "KFIs", "APPLICATION": "Applications",
                  "OFFER": "Offers", "COMPLETED": "Completions"}

# The recent conversion rate averages weekly flow over a 5-week window. Require
# at least this many observed weeks in that window before the rate is treated as
# reliable — a 1-2 week rate is too volatile to publish or forecast off.
_CONVERSION_WINDOW = 5
_MIN_CONVERSION_WEEKS = 3


def _window_count(values: List[Optional[float]], window: int) -> int:
    """How many non-null values fall in the trailing ``window`` (i.e. how many
    weeks actually contributed to a trailing average)."""
    tail = [v for v in values[-window:] if v is not None]
    return len(tail)


def _trailing_avg(values: List[Optional[float]], window: int = 5) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    use = vals[-window:]
    return round(sum(use) / len(use), 2)


def _trend(values: List[Optional[float]]) -> str:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return "flat"
    delta = vals[-1] - vals[-2]
    return "up" if delta > 0 else ("down" if delta < 0 else "flat")


def weekly_flow(levels: List[Optional[float]]) -> List[Optional[float]]:
    """Convert a per-week STOCK level series into a per-week FLOW series.

    ``flow[t] = level[t] − level[t-1]`` — the new origination that arrived in
    week ``t`` (net of cases that left the stage). The first week has no prior
    extract, so its flow is ``None`` (never fabricated as the level itself). A
    week whose level is missing, or that follows a missing level, is ``None``.
    This is the semantic the origination funnel charts on by default; the raw
    stock level is retained separately for the optional cumulative line.
    """
    out: List[Optional[float]] = []
    for i, v in enumerate(levels):
        prev = levels[i - 1] if i > 0 else None
        if i == 0 or v is None or prev is None:
            out.append(None)
        else:
            out.append(round(float(v) - float(prev), 2))
    return out


def _conversion_pct(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Conversion share (%) of a stage relative to KFI, divide-by-zero safe."""
    if not denominator or numerator is None:
        return None
    return round(numerator / denominator * 100.0, 2)


def _lagged_value(series: List[Optional[float]], lag: int) -> Tuple[Optional[float], Optional[int]]:
    """The value ``lag`` steps before the latest, with the index it came from.

    Used to shift the KFI denominator back by the KFI->completion timeline so a
    growing pipeline is not compared against itself. ``lag`` is clamped into the
    available history; a missing (``None``) value at the target index returns
    ``(None, idx)`` rather than fabricating a neighbour.
    """
    n = len(series)
    if n == 0:
        return None, None
    idx = n - 1 - max(0, int(lag or 0))
    if idx < 0:
        idx = 0
    return series[idx], idx


def pipeline_funnel_evolution(pipeline_root: str | os.PathLike, client_id: str,
                              to_run_id: Optional[str] = None,
                              lag_weeks: Optional[int] = None) -> Dict[str, Any]:
    """Weekly origination funnel: KFI / Application / Offer / Completion per
    governed weekly extract, FLOW-FIRST.

    For each stage we track two things per week:
      * the STOCK level — total balance / case count sitting at the stage on the
        extract date (``series[stage][*].value|count``); and
      * the weekly FLOW — the week-on-week change in that level
        (``flowSeries[stage][*].flowValue|flowCount``), i.e. the new origination
        that arrived that week.

    The summary therefore reports BOTH bases, clearly separated, so the 5-week
    average and the Δ-vs-prior-week reconcile with one another:
      * ``fiveWeekAvgFlow*`` is the trailing mean of the weekly FLOW (NOT the
        average stock level — that historical bug made a ~£280MM stock average
        sit next to a ~£33MM weekly Δ);
      * ``deltaFlow*`` is the latest weekly flow minus the prior weekly flow;
      * ``*Stock*`` fields carry the level for the optional cumulative line.

    Conversion vs KFI is a *forward* conversion rate: the average weekly FLOW
    into a stage over the last 5 weeks divided by the KFI STOCK as it stood
    ``lag_weeks`` earlier — i.e. the KFI book at the time today's completions
    entered the pipeline. Shifting the denominator back by the KFI->completion
    timeline stops a growing pipeline being compared against itself (the old
    metric summed per-week stock and could exceed 100%). ``lag_weeks`` is the
    median KFI->completion lag in weeks (from the historical completion model);
    when unknown the rate is computed unlagged and flagged as such. Reuses the
    governed weekly pipeline extracts (same source as ``pipeline_evolution``).
    """
    inv = pipeline_mod.weekly_extract_inventory(pipeline_root, client_id)
    extracts = inv.get("extracts", [])
    cut_ym = pipeline_mod._year_month(str(to_run_id)) if to_run_id else None

    weeks: List[Optional[str]] = []
    sources: List[str] = []
    # series[stage] = [{week, value, count}] (STOCK level per week)
    series: Dict[str, List[Dict[str, Any]]] = {s: [] for s in _FUNNEL_STAGES}

    for ext in extracts:
        edate = ext.get("pipeline_extract_date")
        if cut_ym and edate and edate[:7] > cut_ym:
            continue
        try:
            df, _report = pipeline_mod.load_prepared_pipeline(ext)
        except Exception:  # noqa: BLE001
            continue
        weeks.append(edate)
        sources.append(ext.get("source_file", ""))
        stage_col = df["pipeline_stage"].astype(str) if "pipeline_stage" in df.columns else None
        bal = coerce_numeric(df[_BALANCE]) if _BALANCE in df.columns else None
        for stage in _FUNNEL_STAGES:
            if stage_col is None:
                series[stage].append({"week": edate, "value": None, "count": 0})
                continue
            mask = stage_col.str.upper() == stage
            value = round(float(bal[mask].sum()), 2) if bal is not None else None
            series[stage].append({"week": edate, "value": value, "count": int(mask.sum())})

    # Per-week weekly-flow series derived from the stock levels (bars chart this).
    flow_series: Dict[str, List[Dict[str, Any]]] = {}
    for stage in _FUNNEL_STAGES:
        pts = series[stage]
        vflow = weekly_flow([p["value"] for p in pts])
        cflow = weekly_flow([float(p["count"]) for p in pts])
        flow_series[stage] = [
            {"week": pts[i]["week"],
             "flowValue": vflow[i],
             "flowCount": (int(cflow[i]) if cflow[i] is not None else None)}
            for i in range(len(pts))
        ]

    kfi_counts = [float(p["count"]) for p in series["KFI"]]
    kfi_values = [p["value"] for p in series["KFI"]]

    # KFI denominator, shifted back by the KFI->completion lag so the numerator
    # (recent completions) is measured against the KFI book those completions
    # actually came from — not today's larger book.
    lagged = int(lag_weeks) if lag_weeks not in (None, "") else None
    lag_applied = lagged if lagged is not None else 0
    kfi_denom_count, kfi_denom_idx = _lagged_value(kfi_counts, lag_applied)
    kfi_denom_value, _ = _lagged_value(kfi_values, lag_applied)
    denom_week = weeks[kfi_denom_idx] if kfi_denom_idx is not None and kfi_denom_idx < len(weeks) else None

    summary: Dict[str, Any] = {}
    for stage in _FUNNEL_STAGES:
        pts = series[stage]
        values = [p["value"] for p in pts]
        counts = [float(p["count"]) for p in pts]
        value_flows = [f["flowValue"] for f in flow_series[stage]]
        count_flows = [(float(f["flowCount"]) if f["flowCount"] is not None else None)
                       for f in flow_series[stage]]

        latest_flow_value = value_flows[-1] if value_flows else None
        latest_flow_count = count_flows[-1] if count_flows else None
        prior_flow_value = value_flows[-2] if len(value_flows) >= 2 else None
        prior_flow_count = count_flows[-2] if len(count_flows) >= 2 else None

        avg_flow_value = _trailing_avg(value_flows, _CONVERSION_WINDOW)
        avg_flow_count = _trailing_avg(count_flows, _CONVERSION_WINDOW)

        # Forward conversion vs KFI (never for KFI itself, the denominator):
        # average weekly flow into this stage (last 5 weeks) over the lagged KFI
        # stock. A weekly rate; transparent about the lag and the denominator
        # week so it can't be misread as a same-period share. Flagged
        # insufficient (not to be forecast off) until a few weeks are observed.
        conversion: Optional[Dict[str, Any]] = None
        if stage != "KFI":
            weeks_in_window = _window_count(value_flows, _CONVERSION_WINDOW)
            sufficient = weeks_in_window >= _MIN_CONVERSION_WEEKS
            conversion = {
                "basis": "avg_weekly_flow_over_lagged_kfi_stock",
                "lagWeeks": lagged,
                "lagApplied": bool(lagged),
                "denominatorWeek": denom_week,
                "avgWeeklyFlowCount": avg_flow_count,
                "avgWeeklyFlowValue": avg_flow_value,
                "kfiStockCount": (int(kfi_denom_count)
                                  if kfi_denom_count is not None else None),
                "kfiStockValue": kfi_denom_value,
                "weeklyRateCount": _conversion_pct(avg_flow_count, kfi_denom_count),
                "weeklyRateValue": _conversion_pct(avg_flow_value, kfi_denom_value),
                "weeksInWindow": weeks_in_window,
                "minWeeks": _MIN_CONVERSION_WEEKS,
                "sufficient": sufficient,
            }

        summary[stage] = {
            "label": _FUNNEL_LABELS[stage],
            # Weekly FLOW (default basis for the origination funnel).
            "latestFlowValue": latest_flow_value,
            "latestFlowCount": (int(latest_flow_count)
                                if latest_flow_count is not None else None),
            "priorFlowValue": prior_flow_value,
            "priorFlowCount": (int(prior_flow_count)
                               if prior_flow_count is not None else None),
            "fiveWeekAvgFlowValue": _trailing_avg(value_flows, 5),
            "fiveWeekAvgFlowCount": _trailing_avg(count_flows, 5),
            "deltaFlowValue": (round(latest_flow_value - prior_flow_value, 2)
                               if latest_flow_value is not None
                               and prior_flow_value is not None else None),
            "deltaFlowCount": (int(latest_flow_count - prior_flow_count)
                               if latest_flow_count is not None
                               and prior_flow_count is not None else None),
            # STOCK level (drives the optional cumulative line).
            "latestStockValue": values[-1] if values else None,
            "latestStockCount": pts[-1]["count"] if pts else 0,
            "fiveWeekAvgStockValue": _trailing_avg(values, 5),
            "fiveWeekAvgStockCount": _trailing_avg(counts, 5),
            "trend": _trend(value_flows),
            "weeksObserved": len([v for v in values if v is not None]),
            "conversion": conversion,
        }

    return {
        "dataset": "pipeline_funnel",
        "portfolioId": client_id,
        "toRunId": to_run_id,
        "stages": list(_FUNNEL_STAGES),
        "stageLabels": _FUNNEL_LABELS,
        "weeks": weeks,
        "sourceFiles": sources,
        "uniqueWeeklyExtractsUsed": inv.get("uniqueWeeklyExtractsUsed"),
        "series": series,
        "flowSeries": flow_series,
        "summary": summary,
        "conversionLagWeeks": lagged,
        "lineage": {
            "source": "governed weekly pipeline extracts (deduplicated)",
            "metric": "weekly KFI / Application / Offer / Completion — weekly flow (default) and stock level",
            "fiveWeekAverage": "trailing mean of the last 5 weeks of WEEKLY FLOW (level week-on-week change), not the average stock level",
            "conversion": ("forward conversion rate: average weekly flow into a stage (last 5 weeks) "
                           "over the KFI stock lagWeeks earlier (the KFI->completion timeline); "
                           "unlagged when the lag is unknown"),
        },
        "singlePeriod": len(weeks) <= 1,
    }


# --------------------------------------------------------------------------- #
# Forecast bridge evolution (funded balance + weighted pipeline, per funded run)
# --------------------------------------------------------------------------- #
def forecast_evolution(output_root: str | os.PathLike,
                       pipeline_root: str | os.PathLike, client_id: str,
                       to_run_id: Optional[str] = None, *,
                       historical_model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Forecast bridge over time: funded balance per run + the latest weighted
    pipeline contribution available at/under that run's month. A governed
    ``historical_model`` weights the pipeline by the same empirical stage rates as
    the point-in-time bridge (one consistent 'weighted expected pipeline')."""
    funded = funded_evolution(output_root, client_id, to_run_id)
    pipe = pipeline_evolution(pipeline_root, client_id, to_run_id,
                              historical_model=historical_model)
    # Index pipeline weighted-expected by year-month (latest extract per month).
    weighted_by_month: Dict[str, float] = {}
    for p in pipe["periods"]:
        ym = (p.get("period") or "")
        w = p["metrics"].get("weighted_expected_funded_amount")
        if ym and w is not None:
            weighted_by_month[ym] = float(w)  # later extract overwrites -> latest wins

    periods: List[Dict[str, Any]] = []
    for fp in funded["periods"]:
        ym = fp.get("period") or ""
        funded_bal = fp["metrics"].get("funded_balance") or 0.0
        wpipe = weighted_by_month.get(ym)
        periods.append({
            "period": ym,
            "run_id": fp.get("run_id"),
            "reporting_date": fp.get("reporting_date"),
            "metrics": {
                "funded_balance": round(float(funded_bal), 2),
                "weighted_expected_pipeline": (round(wpipe, 2) if wpipe is not None else None),
                "forecast_funded_balance": round(float(funded_bal) + float(wpipe or 0.0), 2),
            },
            "reconciliation": fp.get("reconciliation"),
            "source_file": fp.get("source_file"),
        })
    return {
        "dataset": "forecast",
        "portfolioId": client_id,
        "toRunId": to_run_id,
        "periods": periods,
        "lineage": {
            "source": "funded central tapes + governed weighted pipeline",
            "formula": "forecast = funded balance + Σ(weighted expected pipeline)",
        },
        "singlePeriod": len(periods) <= 1,
    }
