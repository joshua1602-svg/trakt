#!/usr/bin/env python3
"""
mi_query_executor.py

MI Query Executor v1 — execute a validated :class:`MIQuerySpec` against
canonical portfolio data and return a clean, serialisable result object that is
ready for later chart rendering / Streamlit display / HTML / PPTX export.

This module is DETERMINISTIC and ISOLATED:
  * no LLM calls
  * no chart rendering
  * no Streamlit
  * no Azure Blob integration (local CSV path or pandas DataFrame only)
  * no mutation of the input dataframe
  * never executes arbitrary code

------------------------------------------------------------------------------
REPO INSPECTION FINDINGS (v1 assumptions — see mi_agent/README.md for detail)
------------------------------------------------------------------------------
Canonical data
  * Gate 2 (engine/gate_2_transform/canonical_transform.py) writes the active,
    dashboard-ready output ``<stem>_canonical_typed.csv`` locally with
    ``df.to_csv(..., index=False)``; the pipeline (function_app.py) also uploads
    it to Azure Blob.  This executor reads the LOCAL CSV (or a DataFrame) only.
  * Canonical columns are the canonical field-registry names
    (current_outstanding_balance, current_loan_to_value, origination_date, …).
  * Bucket columns (age_bucket, ltv_bucket, ticket_bucket, vintage_year,
    arrears_bucket, term_bucket) are NOT part of canonical truth — they are
    derived at the analytics layer (analytics/mi_prep.py::add_buckets).  This
    executor REUSES bucket columns if already present in the dataframe and
    otherwise falls back to the raw canonical field with a warning.  It does NOT
    build a bucketing engine and does NOT import analytics/ code.

Percentage scale
  * The repo is INCONSISTENT: canonical_transform.py computes LTV as
    ``(balance/valuation)*100`` (whole-number percent) and business rules
    validate LTV in 0–500, but some sample CSVs store LTV as fractions (0.36).
  * The executor therefore DOES NOT rescale percentages.  It heuristically
    detects the apparent scale and records it (with a warning) in result
    metadata so downstream renderers can decide on formatting.

Balance / exposure convention
  * Preferred balance hierarchy: current_outstanding_balance, then
    current_principal_balance.  Used for balance_sum, default weighted-average
    weight, top-N ranking and concentration share.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from analytics_lib.numeric import coerce_numeric

from .mi_query_spec import MIQuerySpec
from .mi_query_validator import load_mi_semantics, validate_mi_query

# Preferred balance/exposure fields, highest priority first.
_BALANCE_HIERARCHY = ("current_outstanding_balance", "current_principal_balance")
# Aggregations whose grouped result is additive (so a simple share is valid).
_ADDITIVE_AGGS = {"sum", "balance_sum", "count", "count_distinct"}
DEFAULT_MAX_LOAN_LEVEL_ROWS = 5000


# --------------------------------------------------------------------------- #
# Errors / result schema
# --------------------------------------------------------------------------- #


class MIQueryExecutionError(Exception):
    """Raised when an MI query cannot be executed (bad spec, missing column…)."""


class MIDuplicateColumnError(MIQueryExecutionError):
    """Raised when the dataset has duplicate column names so a single-name
    selection returns a DataFrame instead of a Series. Carries the duplicate
    names and the query fields they affect so the API can return controlled
    validation output (never a raw 500)."""

    def __init__(self, message: str, duplicate_columns: List[str],
                 affected_fields: List[str]):
        super().__init__(message)
        self.duplicate_columns = duplicate_columns
        self.affected_fields = affected_fields


def _guard_duplicate_columns(spec: "MIQuerySpec", df: pd.DataFrame, semantics: dict) -> None:
    """Fail fast (and controlled) when the dataset has duplicate column names."""
    dup_mask = df.columns.duplicated(keep=False)
    if not dup_mask.any():
        return
    dups = sorted({str(c) for c in df.columns[dup_mask]})
    affected: List[str] = []
    for slot in ("x", "y", "size", "dimension", "metric", "color", "weight_field"):
        key = getattr(spec, slot, None)
        if not key:
            continue
        canon = _canonical_or_self(key, semantics)
        if canon in dups:
            affected.append(f"{slot}={key}->{canon}")
    raise MIDuplicateColumnError(
        "duplicate_column_names: the dataset has duplicate column names "
        f"{dups}; query fields affected: {affected or 'none directly referenced'}. "
        "The prepared dataset must be de-duplicated (data-preparation defect).",
        duplicate_columns=dups, affected_fields=affected)


@dataclass
class MIQueryResult:
    spec: MIQuerySpec
    result_type: str                       # "table" | "summary" | "loan_level"
    data: pd.DataFrame
    resolved_fields: Dict[str, dict] = field(default_factory=dict)
    row_count: int = 0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- serialisation ----------------------------------------------------- #
    def _records(self) -> List[dict]:
        # round-trip through pandas to_json so dates -> ISO strings and
        # NaN -> null produce valid, portable JSON.
        if self.data is None or self.data.empty:
            return []
        return json.loads(self.data.to_json(orient="records", date_format="iso"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "result_type": self.result_type,
            "data": self._records(),
            "resolved_fields": self.resolved_fields,
            "row_count": self.row_count,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_csv(self, path) -> str:
        self.data.to_csv(path, index=False)
        return str(path)

    def preview(self, n: int = 10) -> pd.DataFrame:
        return self.data.head(n)


# --------------------------------------------------------------------------- #
# Field resolution
# --------------------------------------------------------------------------- #


def resolve_semantic_field(field_key: str, semantics: dict) -> dict:
    """Return the semantic-registry entry for ``field_key`` or raise."""
    entry = semantics.get("fields", {}).get(field_key)
    if entry is None:
        raise MIQueryExecutionError(
            f"Semantic field {field_key!r} not found in MI semantic registry"
        )
    return entry


def get_canonical_field(field_key: str, semantics: dict) -> str:
    """Map a semantic field key to its canonical column name."""
    return resolve_semantic_field(field_key, semantics).get("canonical_field", field_key)


def _canonical_or_self(name: str, semantics: dict) -> str:
    """A candidate may already be a canonical name or a semantic key."""
    entry = semantics.get("fields", {}).get(name)
    if entry is not None:
        return entry.get("canonical_field", name)
    return name


def resolve_default_balance_field(semantics: dict, df_columns) -> Optional[str]:
    """First balance field from the preferred hierarchy that exists in the data."""
    cols = set(df_columns)
    for key in _BALANCE_HIERARCHY:
        canonical = _canonical_or_self(key, semantics)
        if canonical in cols:
            return canonical
    return None


def resolve_weight_field(spec: MIQuerySpec, metric_entry: Optional[dict],
                         semantics: dict, df_columns) -> Optional[str]:
    """Resolve the weighted-average weight column, in priority order:

    1. spec.weight_field
    2. metric entry's weight_field
    3. semantics metadata default_weight_field
    4. current_outstanding_balance
    5. current_principal_balance
    """
    cols = set(df_columns)
    candidates: List[Optional[str]] = []
    if spec.weight_field:
        candidates.append(spec.weight_field)
    if metric_entry:
        candidates.append(metric_entry.get("weight_field"))
    candidates.append(semantics.get("metadata", {}).get("default_weight_field"))
    candidates.extend(_BALANCE_HIERARCHY)

    for cand in candidates:
        if not cand:
            continue
        canonical = _canonical_or_self(cand, semantics)
        if canonical in cols:
            return canonical
    return None


def _require_column(df: pd.DataFrame, canonical: str, field_key: str) -> None:
    if canonical not in df.columns:
        raise MIQueryExecutionError(
            f"Canonical column {canonical!r} (for semantic field {field_key!r}) "
            f"is not present in the data"
        )


def _resolve_group_column(field_key: str, semantics: dict, df: pd.DataFrame,
                          warnings: List[str], *, use_bucket: bool = False) -> str:
    """Resolve a grouping column, optionally preferring an existing bucket column."""
    entry = resolve_semantic_field(field_key, semantics)
    canonical = entry.get("canonical_field", field_key)
    bucket = entry.get("bucket_field")
    if use_bucket and bucket:
        if bucket in df.columns:
            return bucket
        warnings.append(
            f"bucket field {bucket!r} for {field_key!r} not present in data; "
            f"using raw field {canonical!r}"
        )
    _require_column(df, canonical, field_key)
    return canonical


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def aggregate_series(df: pd.DataFrame, value_col: Optional[str], aggregation: str,
                     weight_col: Optional[str] = None,
                     balance_col: Optional[str] = None) -> float:
    """Compute a single scalar aggregation over ``df``.

    Supports: sum, avg, median, count, count_distinct, weighted_avg, balance_sum.
    ``distribution`` and ``loan_level`` are structural (not scalar) and are
    handled by the per-chart logic, not here.
    """
    if aggregation == "count":
        return int(len(df))
    if aggregation == "count_distinct":
        if not value_col:
            raise MIQueryExecutionError("count_distinct requires a field")
        return int(df[value_col].nunique(dropna=True))
    if aggregation == "balance_sum":
        if not balance_col:
            raise MIQueryExecutionError("balance_sum requires a balance field")
        return float(coerce_numeric(df[balance_col]).sum())

    if aggregation in ("distribution", "loan_level"):
        raise MIQueryExecutionError(
            f"{aggregation!r} is not a scalar aggregation"
        )

    if not value_col:
        raise MIQueryExecutionError(
            f"aggregation {aggregation!r} requires a metric column"
        )
    vals = coerce_numeric(df[value_col])
    if aggregation == "sum":
        return float(vals.sum())
    if aggregation == "avg":
        return float(vals.mean())
    if aggregation == "median":
        return float(vals.median())
    if aggregation == "weighted_avg":
        if not weight_col:
            raise MIQueryExecutionError("weighted_avg requires a weight field")
        w = coerce_numeric(df[weight_col])
        mask = vals.notna() & w.notna()
        denom = float(w[mask].sum())
        if denom == 0:
            return float("nan")
        return float((vals[mask] * w[mask]).sum() / denom)
    raise MIQueryExecutionError(f"Unsupported aggregation: {aggregation!r}")


def _metric_col_name(value_col: Optional[str], aggregation: str,
                     balance_col: Optional[str] = None) -> str:
    if aggregation == "count":
        return "count"
    if aggregation == "count_distinct":
        return f"{value_col}_count_distinct"
    if aggregation == "balance_sum":
        return f"{balance_col}_sum"
    return f"{value_col}_{aggregation}"


def _grouped_aggregate(work: pd.DataFrame, group_cols: List[str],
                       value_col: Optional[str], aggregation: str,
                       weight_col: Optional[str],
                       balance_col: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """Group by ``group_cols`` and compute one aggregation per group."""
    metric_col = _metric_col_name(value_col, aggregation, balance_col)
    rows: List[dict] = []
    for keys, g in work.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row[metric_col] = aggregate_series(g, value_col, aggregation,
                                           weight_col, balance_col)
        rows.append(row)
    out = pd.DataFrame(rows, columns=list(group_cols) + [metric_col])
    return out, metric_col


# --------------------------------------------------------------------------- #
# Cleaning / filtering helpers
# --------------------------------------------------------------------------- #


def _exclude_missing(work: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, int]:
    """Drop rows whose value in any of ``cols`` is null/blank."""
    mask = pd.Series(True, index=work.index)
    for c in cols:
        col = work[c]
        m = col.notna()
        if col.dtype == object or pd.api.types.is_string_dtype(col):
            s = col.astype(str).str.strip().str.lower()
            m = m & (s != "") & (s != "nan") & (s != "none")
        mask = mask & m
    excluded = int((~mask).sum())
    return work.loc[mask].copy(), excluded


def _stringify(work: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Cast grouping columns to plain strings for robust, deterministic grouping."""
    for c in cols:
        work[c] = work[c].astype(str)
    return work


_OP_ALIASES = {
    ">": "gt", "gt": "gt", "more_than": "gt", "greater_than": "gt", "above": "gt", "over": "gt",
    ">=": "ge", "ge": "ge", "at_least": "ge", "min": "ge",
    "<": "lt", "lt": "lt", "less_than": "lt", "below": "lt", "under": "lt",
    "<=": "le", "le": "le", "at_most": "le", "max": "le",
    "==": "eq", "eq": "eq", "=": "eq", "is": "eq",
    "!=": "ne", "ne": "ne",
    "between": "between",
}


def _apply_numeric_op(col: pd.Series, op: str, value: Any) -> pd.Series:
    """Boolean mask for a numeric comparison operator against a coerced column."""
    s = coerce_numeric(col)
    if op == "between":
        lo, hi = (value if isinstance(value, (list, tuple)) and len(value) == 2
                  else (None, None))
        return (s >= float(lo)) & (s <= float(hi))
    v = float(value)
    return {"gt": s > v, "ge": s >= v, "lt": s < v, "le": s <= v,
            "eq": s == v, "ne": s != v}[op]


def _apply_filters(work: pd.DataFrame, spec: MIQuerySpec, semantics: dict,
                   warnings: List[str]) -> pd.DataFrame:
    if not spec.filters:
        return work
    from .mi_dataset_profile import PERCENT_FRACTION, percent_storage_scale
    for field_key, value in spec.filters.items():
        entry = resolve_semantic_field(field_key, semantics)
        canonical = entry.get("canonical_field", field_key)
        _require_column(work, canonical, field_key)
        before = len(work)
        col = work[canonical]
        if isinstance(value, dict) and ("op" in value or "value" in value
                                        or "min" in value or "max" in value):
            # Structured numeric comparison filter: {"op": ">", "value": 70}.
            op = _OP_ALIASES.get(str(value.get("op", "eq")).strip().lower(), "eq")
            raw = value.get("value", value.get("min", value.get("max")))
            # Scale-aware: a percent threshold (e.g. "ltv over 80") is compared in
            # the column's own storage scale. If the column stores fractions
            # (0.51) but the threshold reads as points (80), convert once here —
            # the single percent-scale source of truth, never re-guessed downstream.
            if entry.get("format") == "percent" and isinstance(raw, (int, float)):
                if percent_storage_scale(col) == PERCENT_FRACTION and abs(raw) > 1.5:
                    raw = raw / 100.0
                    if isinstance(value.get("between"), (list, tuple)):
                        raw = [x / 100.0 for x in value["between"]]
            mask = _apply_numeric_op(col, op, raw)
            work = work[mask.fillna(False)]
            warnings.append(f"filter {field_key} {op} {raw!r} kept {len(work)}/{before} rows")
        elif isinstance(value, (list, tuple, set)):
            work = work[col.isin(list(value))]
            warnings.append(f"filter {field_key} in {list(value)!r} kept {len(work)}/{before} rows")
        else:
            work = work[col == value]
            warnings.append(f"filter {field_key}={value!r} kept {len(work)}/{before} rows")
    return work.copy()


def _group_sum(work: pd.DataFrame, group_cols: List[str], col: str) -> pd.Series:
    tmp = work.copy()
    tmp[col] = coerce_numeric(tmp[col])
    return tmp.groupby(group_cols, sort=False)[col].sum()


def _align(out: pd.DataFrame, group_cols: List[str], series: pd.Series) -> pd.Series:
    if len(group_cols) == 1:
        idx = pd.Index(out[group_cols[0]])
    else:
        idx = pd.MultiIndex.from_frame(out[group_cols])
    return pd.Series(series.reindex(idx).to_numpy(), index=out.index)


def _maybe_concentration(out: pd.DataFrame, metric_col: str, work: pd.DataFrame,
                         group_cols: List[str], aggregation: str,
                         balance_col: Optional[str],
                         warnings: List[str]) -> pd.DataFrame:
    """Add ``concentration_pct`` (share of total) where it is meaningful."""
    try:
        if aggregation in _ADDITIVE_AGGS:
            basis = coerce_numeric(out[metric_col])
        elif balance_col and balance_col in work.columns:
            basis = _align(out, group_cols, _group_sum(work, group_cols, balance_col))
        else:
            warnings.append(
                "concentration_pct not added (non-additive aggregation and no "
                "balance field available)"
            )
            return out
        total = float(coerce_numeric(basis).sum())
        if total and total == total and total != 0:
            out = out.copy()
            out["concentration_pct"] = coerce_numeric(basis) / total * 100.0
        return out
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"concentration_pct not computed: {exc}")
        return out


def _apply_top_n(out: pd.DataFrame, metric_col: str, work: pd.DataFrame,
                 group_cols: List[str], aggregation: str,
                 balance_col: Optional[str], top_n: Optional[int],
                 rank_priority: Tuple[str, ...], warnings: List[str]) -> pd.DataFrame:
    if top_n is None:
        return out
    rank = None
    basis_name = None
    for basis in rank_priority:
        if basis == "balance" and balance_col and balance_col in work.columns:
            rank = _align(out, group_cols, _group_sum(work, group_cols, balance_col))
            basis_name = "balance"
            break
        if basis == "count":
            rank = _align(out, group_cols, work.groupby(group_cols, sort=False).size())
            basis_name = "count"
            break
        if basis == "concentration" and "concentration_pct" in out.columns:
            rank = out["concentration_pct"]
            basis_name = "concentration"
            break
    if rank is None:
        rank = coerce_numeric(out[metric_col])
        basis_name = "metric"
    order = rank.sort_values(ascending=False, kind="mergesort").index
    out2 = out.loc[order].head(int(top_n)).reset_index(drop=True)
    warnings.append(f"top_n={top_n} applied, ranked by {basis_name}")
    return out2


# --------------------------------------------------------------------------- #
# Percent-scale detection
# --------------------------------------------------------------------------- #


def _detect_percent_scale(df: pd.DataFrame, semantics: dict) -> Tuple[str, Optional[float]]:
    medians: List[float] = []
    for entry in semantics.get("fields", {}).values():
        if entry.get("format") != "percent":
            continue
        col = entry.get("canonical_field")
        if col in df.columns:
            s = coerce_numeric(df[col]).dropna()
            if len(s):
                medians.append(float(s.median()))
    if not medians:
        return "unknown", None
    overall = float(pd.Series(medians).median())
    scale = "fraction" if overall <= 1.5 else "whole_number_percent"
    return scale, overall


# --------------------------------------------------------------------------- #
# Per-query executors
# --------------------------------------------------------------------------- #


def _metric_aggregation(spec: MIQuerySpec) -> str:
    """The aggregation to use for a metric, defaulting sanely when unset."""
    agg = spec.aggregation
    if agg in ("loan_level",):  # not meaningful for grouped/summary metric
        return "sum"
    return agg


def _execute_summary(spec, work, semantics, warnings, balance_col):
    if spec.metric:
        entry = resolve_semantic_field(spec.metric, semantics)
        canonical = entry.get("canonical_field")
        _require_column(work, canonical, spec.metric)
        agg = _metric_aggregation(spec)
        weight_col = (resolve_weight_field(spec, entry, semantics, work.columns)
                      if agg == "weighted_avg" else None)
        bcol = balance_col if agg == "balance_sum" else None
        value = aggregate_series(work, canonical, agg, weight_col, bcol)
        col = _metric_col_name(canonical, agg, bcol)
        data = pd.DataFrame([{"loan_count": int(len(work)), col: value}])
    else:
        row = {"loan_count": int(len(work))}
        if balance_col:
            row[f"{balance_col}_sum"] = float(
                coerce_numeric(work[balance_col]).sum()
            )
        data = pd.DataFrame([row])
    return data, "summary"


def _execute_grouped(spec, work, semantics, warnings, balance_col,
                     group_field_keys, *, use_bucket, top_n_allowed,
                     rank_priority, sort_for_date=False):
    """Shared bar / table / treemap / heatmap grouped execution."""
    group_cols: List[str] = []
    date_group = False
    for key in group_field_keys:
        entry = resolve_semantic_field(key, semantics)
        if entry.get("role") == "date":
            date_group = True
        group_cols.append(
            _resolve_group_column(key, semantics, work, warnings, use_bucket=use_bucket)
        )

    # metric / aggregation
    if spec.metric:
        m_entry = resolve_semantic_field(spec.metric, semantics)
        value_col = m_entry.get("canonical_field")
        _require_column(work, value_col, spec.metric)
        agg = _metric_aggregation(spec)
        weight_col = (resolve_weight_field(spec, m_entry, semantics, work.columns)
                      if agg == "weighted_avg" else None)
    else:
        value_col = None
        agg = "count"
        weight_col = None
    bcol = balance_col if agg == "balance_sum" else balance_col

    # exclude missing grouping values, then stringify keys for stable grouping
    work, excluded = _exclude_missing(work, group_cols)
    if excluded:
        warnings.append(
            f"excluded {excluded} row(s) with missing/blank grouping value(s) "
            f"in {group_cols}"
        )
    work = _stringify(work, group_cols)

    out, metric_col = _grouped_aggregate(
        work, group_cols, value_col, agg, weight_col,
        balance_col if agg == "balance_sum" else None,
    )

    # ordering
    if sort_for_date and date_group:
        out = out.sort_values(group_cols, kind="mergesort").reset_index(drop=True)
    else:
        out = out.sort_values(metric_col, ascending=False,
                              kind="mergesort").reset_index(drop=True)

    out = _maybe_concentration(out, metric_col, work, group_cols, agg,
                               balance_col, warnings)

    if top_n_allowed and spec.top_n is not None:
        out = _apply_top_n(out, metric_col, work, group_cols, agg, balance_col,
                           spec.top_n, rank_priority, warnings)
    elif spec.top_n is not None:
        warnings.append("top_n ignored for this output type")

    return out, "table"


def _execute_line(spec, work, semantics, warnings, balance_col):
    date_key = spec.x or spec.dimension
    if not date_key:
        raise MIQueryExecutionError("line chart requires a date x/dimension")
    entry = resolve_semantic_field(date_key, semantics)
    canonical = entry.get("canonical_field")
    bucket = entry.get("bucket_field")

    # Prefer an existing vintage bucket column for yearly cohorts.
    if bucket and "year" in str(bucket).lower() and bucket in work.columns:
        period_col = bucket
        work = work.copy()
        work[period_col] = work[period_col].astype(str)
    else:
        _require_column(work, canonical, date_key)
        period_col = (canonical[:-5] if canonical.endswith("_date") else canonical) + "_month"
        work = work.copy()
        dts = pd.to_datetime(work[canonical], errors="coerce")
        work[period_col] = dts.dt.to_period("M").astype(str)

    work, excluded = _exclude_missing(work, [period_col])
    # 'NaT' periods become the string 'NaT' -> drop them explicitly
    work = work[work[period_col].astype(str).str.lower() != "nat"].copy()
    if excluded:
        warnings.append(f"excluded {excluded} row(s) with missing date for line chart")

    if spec.metric:
        m_entry = resolve_semantic_field(spec.metric, semantics)
        value_col = m_entry.get("canonical_field")
        _require_column(work, value_col, spec.metric)
        agg = _metric_aggregation(spec)
        weight_col = (resolve_weight_field(spec, m_entry, semantics, work.columns)
                      if agg == "weighted_avg" else None)
    else:
        value_col, agg, weight_col = None, "count", None

    out, metric_col = _grouped_aggregate(work, [period_col], value_col, agg,
                                         weight_col,
                                         balance_col if agg == "balance_sum" else None)
    out = out.sort_values(period_col, kind="mergesort").reset_index(drop=True)
    return out, "table"


def _execute_loan_level(spec, work, semantics, warnings, *, need_size,
                        max_loan_level_rows, sample_seed, metadata):
    cols: List[Tuple[str, str]] = []  # (canonical, role-label)
    for slot in ("x", "y") + (("size",) if need_size else ()):
        key = getattr(spec, slot)
        if not key:
            raise MIQueryExecutionError(f"{spec.chart_type} requires {slot}")
        canonical = get_canonical_field(key, semantics)
        _require_column(work, canonical, key)
        cols.append((canonical, slot))
    color_col = None
    if spec.color:
        color_col = get_canonical_field(spec.color, semantics)
        _require_column(work, color_col, spec.color)

    selected = [c for c, _ in cols]
    if color_col and color_col not in selected:
        selected.append(color_col)
    # IMPORTANT: only the requested analytical columns are selected — loan
    # identifiers and other canonical columns are NOT exposed by default.
    out = work[selected].copy()

    # numeric coercion for x / y / size; color preserved as-is. Defensive: a
    # duplicated column name would make out[c] a DataFrame and crash coercion —
    # never pass a DataFrame into coerce_numeric.
    numeric_cols = [c for c, _ in cols]
    dups = sorted({c for c in numeric_cols if (out.columns == c).sum() > 1})
    if dups:
        raise MIDuplicateColumnError(
            f"duplicate_column_names: loan-level x/y/size columns {dups} are "
            "duplicated in the dataset; de-duplicate the prepared dataset.",
            duplicate_columns=dups,
            affected_fields=[f"{role}={c}" for c, role in cols if c in dups])
    for c in numeric_cols:
        out[c] = coerce_numeric(out[c])
    before = len(out)
    out = out.dropna(subset=numeric_cols)
    if len(out) < before:
        warnings.append(
            f"dropped {before - len(out)} loan-level row(s) with non-numeric/null "
            f"x/y/size values"
        )

    original_rows = len(out)
    sampled = False
    if original_rows > max_loan_level_rows:
        out = out.sample(n=max_loan_level_rows, random_state=sample_seed)
        out = out.reset_index(drop=True)
        sampled = True
        warnings.append(
            f"loan-level output capped: sampled {max_loan_level_rows} of "
            f"{original_rows} rows (deterministic seed={sample_seed})"
        )
    else:
        out = out.reset_index(drop=True)

    metadata.update({
        "loan_level_original_rows": int(original_rows),
        "loan_level_returned_rows": int(len(out)),
        "loan_level_sampled": sampled,
        "max_loan_level_rows": int(max_loan_level_rows),
        "sample_seed": int(sample_seed),
        "identifiers_included": False,
    })
    return out, "loan_level"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def _read_data(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, (str, Path)):
        return pd.read_csv(data)
    raise MIQueryExecutionError(
        "data must be a pandas DataFrame or a path to a CSV file"
    )


def execute_mi_query(
    spec: MIQuerySpec,
    data,
    semantics,
    *,
    validate: bool = True,
    max_loan_level_rows: int = DEFAULT_MAX_LOAN_LEVEL_ROWS,
    missing_dimension_policy: str = "exclude",
    top_n_rank_priority: Tuple[str, ...] = ("balance", "count", "concentration"),
    sample_seed: int = 42,
) -> MIQueryResult:
    """Execute a validated :class:`MIQuerySpec` against canonical data.

    See module docstring for the v1 behaviour / assumptions.
    """
    if missing_dimension_policy != "exclude":
        raise MIQueryExecutionError(
            "v1 only supports missing_dimension_policy='exclude'"
        )

    if isinstance(semantics, (str, Path)):
        semantics = load_mi_semantics(semantics)
    df = _read_data(data)

    if validate:
        vr = validate_mi_query(spec, semantics, available_columns=set(df.columns))
        if not vr.ok:
            raise MIQueryExecutionError(
                "spec failed validation: " + "; ".join(vr.errors)
            )

    warnings: List[str] = []
    work = df.copy()  # never mutate the caller's dataframe
    # Duplicate column names make a single-name selection return a DataFrame
    # (crashing numeric coercion). Fail fast with a controlled, explained error.
    _guard_duplicate_columns(spec, work, semantics)
    work = _apply_filters(work, spec, semantics, warnings)

    balance_col = resolve_default_balance_field(semantics, work.columns)
    scale, scale_median = _detect_percent_scale(df, semantics)
    warnings.append(
        f"percent-scale heuristically detected as {scale!r}"
        + (f" (median {scale_median:.4g})" if scale_median is not None else "")
        + "; the executor does NOT rescale percentages"
    )

    metadata: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "semantics_version": semantics.get("metadata", {}).get("version"),
        "intent": spec.intent,
        "chart_type": spec.chart_type,
        "aggregation": spec.aggregation,
        "balance_field_used": balance_col,
        "percent_scale_detected": scale,
        "percent_scale_median": scale_median,
        "missing_dimension_policy": missing_dimension_policy,
        "top_n_rank_priority": list(top_n_rank_priority),
        "input_row_count": int(len(df)),
        "filtered_row_count": int(len(work)),
    }

    # ---- dispatch -------------------------------------------------------- #
    if spec.intent == "summary" or (spec.intent == "chart" and spec.chart_type == "none"):
        data_out, result_type = _execute_summary(spec, work, semantics, warnings, balance_col)

    elif spec.intent == "table":
        keys = []
        if spec.dimension:
            keys = [spec.dimension]
        if keys:
            data_out, result_type = _execute_grouped(
                spec, work, semantics, warnings, balance_col, keys,
                use_bucket=False, top_n_allowed=True, rank_priority=top_n_rank_priority,
            )
        else:
            data_out, result_type = _execute_summary(spec, work, semantics, warnings, balance_col)

    elif spec.chart_type == "bar":
        keys = [spec.dimension or spec.x]
        if not keys[0]:
            raise MIQueryExecutionError("bar chart requires a dimension or x")
        data_out, result_type = _execute_grouped(
            spec, work, semantics, warnings, balance_col, keys,
            use_bucket=False, top_n_allowed=True, rank_priority=top_n_rank_priority,
            sort_for_date=True,
        )

    elif spec.chart_type == "line":
        data_out, result_type = _execute_line(spec, work, semantics, warnings, balance_col)

    elif spec.chart_type in ("scatter", "bubble"):
        data_out, result_type = _execute_loan_level(
            spec, work, semantics, warnings,
            need_size=(spec.chart_type == "bubble"),
            max_loan_level_rows=max_loan_level_rows,
            sample_seed=sample_seed, metadata=metadata,
        )

    elif spec.chart_type == "heatmap":
        keys = _two_dimension_keys(spec)
        data_out, result_type = _execute_grouped(
            spec, work, semantics, warnings, balance_col, keys,
            use_bucket=True, top_n_allowed=False, rank_priority=top_n_rank_priority,
        )

    elif spec.chart_type == "treemap":
        keys = list(spec.hierarchy) or list(spec.dimensions)
        if spec.dimension:
            keys = keys + [spec.dimension]
        keys = [k for k in keys if k]
        if not keys:
            raise MIQueryExecutionError("treemap requires hierarchy/dimensions")
        data_out, result_type = _execute_grouped(
            spec, work, semantics, warnings, balance_col, keys,
            use_bucket=True, top_n_allowed=True, rank_priority=top_n_rank_priority,
        )

    else:
        raise MIQueryExecutionError(
            f"Unsupported intent/chart_type combination: "
            f"{spec.intent!r}/{spec.chart_type!r}"
        )

    resolved = {
        key: {
            "canonical_field": resolve_semantic_field(key, semantics).get("canonical_field"),
            "role": resolve_semantic_field(key, semantics).get("role"),
            "format": resolve_semantic_field(key, semantics).get("format"),
        }
        for key in spec.referenced_fields()
        if key in semantics.get("fields", {})
    }

    return MIQueryResult(
        spec=spec,
        result_type=result_type,
        data=data_out,
        resolved_fields=resolved,
        row_count=int(len(data_out)),
        warnings=warnings,
        metadata=metadata,
    )


def _two_dimension_keys(spec: MIQuerySpec) -> List[str]:
    """Pick the two grouping dimensions for a heatmap from the spec."""
    if spec.dimensions and len(spec.dimensions) >= 2:
        return list(spec.dimensions[:2])
    candidates = [k for k in (spec.x, spec.y, spec.dimension, spec.color) if k]
    if len(candidates) >= 2:
        return candidates[:2]
    raise MIQueryExecutionError(
        "heatmap requires two dimensions (via dimensions=[...], x & y, or "
        "dimension & color)"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Execute an MIQuerySpec against canonical data.")
    parser.add_argument("--semantics", type=Path, required=True,
                        help="Path to mi_semantics_field_registry.yaml")
    parser.add_argument("--spec", type=Path, required=True,
                        help="Path to a JSON MIQuerySpec")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to a canonical CSV")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional path to write the result CSV")
    parser.add_argument("--max-loan-level-rows", type=int,
                        default=DEFAULT_MAX_LOAN_LEVEL_ROWS)
    args = parser.parse_args(argv)

    spec = MIQuerySpec.from_json(Path(args.spec).read_text(encoding="utf-8"))
    try:
        result = execute_mi_query(
            spec, str(args.data), str(args.semantics),
            validate=True, max_loan_level_rows=args.max_loan_level_rows,
        )
    except MIQueryExecutionError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"result_type : {result.result_type}")
    print(f"row_count   : {result.row_count}")
    if result.warnings:
        print("warnings:")
        for w in result.warnings:
            print(f"  - {w}")

    if args.out:
        result.to_csv(args.out)
        print(f"output      : {args.out}")
    else:
        print("\npreview:")
        print(result.preview().to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
