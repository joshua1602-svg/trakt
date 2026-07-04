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

# Explicit bucket for rows whose grouping dimension is missing/blank. Such rows
# are NOT silently dropped (which understates totals) — they are grouped here so
# the result still reconciles to the funded book.
MISSING_BUCKET_LABEL = "Unknown / Missing"
# A group with fewer loans than this is statistically thin; an average over it is
# flagged so the operator does not over-read a high "average" on 1–2 loans.
LOW_GROUP_COUNT = 5


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


def _missing_mask(work: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Boolean mask of rows whose value in ANY of ``cols`` is null/blank."""
    missing = pd.Series(False, index=work.index)
    for c in cols:
        col = work[c]
        m = ~col.notna()
        if col.dtype == object or pd.api.types.is_string_dtype(col):
            s = col.astype(str).str.strip().str.lower()
            m = m | (s == "") | (s == "nan") | (s == "none") | (s == "nat")
        missing = missing | m
    return missing


def _exclude_missing(work: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, int]:
    """Drop rows whose value in any of ``cols`` is null/blank."""
    missing = _missing_mask(work, cols)
    excluded = int(missing.sum())
    return work.loc[~missing].copy(), excluded


def _bucket_missing(work: pd.DataFrame, cols: List[str],
                    label: str = MISSING_BUCKET_LABEL) -> Tuple[pd.DataFrame, int]:
    """Replace null/blank grouping values with an explicit ``Unknown / Missing``
    bucket so the rows are KEPT (and the result reconciles), returning the count
    of rows that were bucketed."""
    work = work.copy()
    bucketed = 0
    for c in cols:
        col = work[c]
        m = ~col.notna()
        if col.dtype == object or pd.api.types.is_string_dtype(col):
            s = col.astype(str).str.strip().str.lower()
            m = m | (s == "") | (s == "nan") | (s == "none") | (s == "nat")
        work[c] = col.astype(object).where(~m, label)
        bucketed = max(bucketed, int(m.sum()))
    return work, bucketed


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
    # Categorical membership (used by the "Other" bar-bucket drill-through).
    "in": "in", "not_in": "not_in", "nin": "not_in", "not in": "not_in",
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
        # Resolve a semantic field key to its canonical column. A drill-through
        # filter may instead arrive keyed by the artifact's own data column
        # (already canonical); tolerate that when the column exists, so the UI can
        # pass either the semantic key or the column name. An unknown key that is
        # neither re-raises -> controlled validation failure (never a 500).
        try:
            entry = resolve_semantic_field(field_key, semantics)
            canonical = entry.get("canonical_field", field_key)
        except MIQueryExecutionError:
            if field_key in work.columns:
                entry, canonical = {}, field_key
            else:
                raise
        _require_column(work, canonical, field_key)
        before = len(work)
        col = work[canonical]
        if isinstance(value, dict) and ("op" in value or "value" in value
                                        or "min" in value or "max" in value):
            # Structured numeric comparison filter: {"op": ">", "value": 70}.
            op = _OP_ALIASES.get(str(value.get("op", "eq")).strip().lower(), "eq")
            raw = value.get("value", value.get("min", value.get("max")))
            # Categorical membership (the "Other" bar bucket drills as NOT IN the
            # shown top-N categories, so the underlying rows are recovered).
            if op in ("in", "not_in"):
                members = raw if isinstance(raw, (list, tuple, set)) else [raw]
                vals = [str(v).strip().casefold() for v in members]
                member_mask = col.astype(str).str.strip().str.casefold().isin(vals)
                work = work[member_mask if op == "in" else ~member_mask]
                warnings.append(f"filter {field_key} {op} {list(members)!r} kept {len(work)}/{before} rows")
                continue
            # Scale-aware: a percent threshold (e.g. "ltv over 80") is compared in
            # the column's own storage scale. If the column stores fractions
            # (0.51) but the threshold reads as points (80), convert once here —
            # the single percent-scale source of truth, never re-guessed downstream.
            # Handles both a scalar threshold and a two-element ``between`` range
            # (e.g. "ltv between 60 and 80" -> [0.6, 0.8]); the range bounds live
            # in ``raw`` (the "value" key), so they must be rescaled here too.
            if entry.get("format") == "percent" \
                    and percent_storage_scale(col) == PERCENT_FRACTION:
                if isinstance(raw, (int, float)) and abs(raw) > 1.5:
                    raw = raw / 100.0
                elif (isinstance(raw, (list, tuple))
                      and all(isinstance(x, (int, float)) for x in raw)
                      and any(abs(float(x)) > 1.5 for x in raw)):
                    raw = [float(x) / 100.0 for x in raw]
            mask = _apply_numeric_op(col, op, raw)
            work = work[mask.fillna(False)]
            warnings.append(f"filter {field_key} {op} {raw!r} kept {len(work)}/{before} rows")
        elif isinstance(value, (list, tuple, set)):
            vals = [str(v).strip().casefold() for v in value]
            work = work[col.astype(str).str.strip().str.casefold().isin(vals)]
            warnings.append(f"filter {field_key} in {list(value)!r} kept {len(work)}/{before} rows")
        elif isinstance(value, str):
            # Case-/whitespace-insensitive categorical match so a normalised
            # value ("South West") matches the prepared dimension value robustly.
            target = value.strip().casefold()
            work = work[col.astype(str).str.strip().str.casefold() == target]
            warnings.append(f"filter {field_key}={value!r} kept {len(work)}/{before} rows")
        else:
            work = work[col == value]
            warnings.append(f"filter {field_key}={value!r} kept {len(work)}/{before} rows")
    return work.copy()


def _balance_sum(df: pd.DataFrame, balance_col: Optional[str]) -> Optional[float]:
    if not balance_col or balance_col not in df.columns:
        return None
    return float(coerce_numeric(df[balance_col]).sum())


def _coverage_block(before_n: int, before_bal: Optional[float],
                    included_df: pd.DataFrame, balance_col: Optional[str],
                    group_cols: List[str], policy: str) -> Dict[str, Any]:
    """Included vs excluded records/balance for a grouped result."""
    inc_n = int(len(included_df))
    inc_bal = _balance_sum(included_df, balance_col)
    return {
        "records_eligible": int(before_n),
        "balance_eligible": (round(before_bal, 2) if before_bal is not None else None),
        "records_included": inc_n,
        "balance_included": (round(inc_bal, 2) if inc_bal is not None else None),
        "records_excluded_missing": int(before_n - inc_n),
        "balance_excluded_missing": (None if before_bal is None or inc_bal is None
                                     else round(before_bal - inc_bal, 2)),
        "missing_dimension_policy": policy,
        "missing_dimension_fields": list(group_cols),
    }


def _build_reconciliation(df: pd.DataFrame, work: pd.DataFrame,
                          balance_col: Optional[str], spec: MIQuerySpec,
                          coverage: Dict[str, Any], result_type: str,
                          metadata: Dict[str, Any],
                          measure_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Assemble the reconciliation / coverage footer for any result.

    Reports the dataset universe, the effect of any filters, and how much of the
    funded book is actually covered by the result vs excluded due to missing
    dimensions/measures. Every MI artifact carries this so a downloaded total can
    always be reconciled against the funded-book snapshot.
    """
    total_n = int(len(df))
    total_bal = _balance_sum(df, balance_col)
    filtered_n = int(len(work))
    filtered_bal = _balance_sum(work, balance_col)
    filters_applied = bool(spec.filters)

    # Missing-MEASURE disclosure: rows in the filtered universe whose metric value
    # is null do not contribute to a sum/avg even when their dimension is known.
    missing_measure_fields: List[str] = []
    records_missing_measure = 0
    balance_missing_measure: Optional[float] = None
    for mcol in (measure_cols or []):
        if mcol and mcol in work.columns:
            miss = ~coerce_numeric(work[mcol]).notna()
            n_miss = int(miss.sum())
            if n_miss:
                missing_measure_fields.append(mcol)
                records_missing_measure = max(records_missing_measure, n_miss)
                if balance_col and balance_col in work.columns:
                    bal = float(coerce_numeric(work.loc[miss, balance_col]).sum())
                    balance_missing_measure = (bal if balance_missing_measure is None
                                               else balance_missing_measure + bal)

    if coverage:  # grouped path supplied precise included/excluded figures
        included_n = coverage["records_included"]
        included_bal = coverage["balance_included"]
        excluded_n = coverage["records_excluded_missing"]
        excluded_bal = coverage["balance_excluded_missing"]
        missing_fields = coverage.get("missing_dimension_fields", [])
        policy = coverage.get("missing_dimension_policy", "bucket")
    else:
        # Summary / line / loan-level: included = the filtered universe (loan-level
        # axis drops are disclosed separately in metadata).
        included_n = filtered_n
        included_bal = filtered_bal
        excluded_n = 0
        excluded_bal = 0.0 if filtered_bal is not None else None
        missing_fields = []
        policy = "n/a"

    cov_pct = None
    if total_bal not in (None, 0) and included_bal is not None:
        cov_pct = round(included_bal / total_bal * 100.0, 1)

    return {
        "dataset": metadata.get("dataset"),
        "run_id": metadata.get("run_id"),
        "total_records": total_n,
        "total_balance": (round(total_bal, 2) if total_bal is not None else None),
        "filters_applied": filters_applied,
        "filters": dict(spec.filters or {}),
        "records_after_filters": filtered_n,
        "balance_after_filters": (round(filtered_bal, 2) if filtered_bal is not None else None),
        "records_included": included_n,
        "balance_included": included_bal,
        "records_excluded_missing": excluded_n,
        "balance_excluded_missing": excluded_bal,
        "missing_dimension_policy": policy,
        "missing_dimension_fields": missing_fields,
        "missing_measure_fields": missing_measure_fields,
        "records_missing_measure": records_missing_measure,
        "balance_missing_measure": (round(balance_missing_measure, 2)
                                    if balance_missing_measure is not None else None),
        "coverage_by_balance_pct": cov_pct,
        "balance_field": balance_col,
    }


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
                 rank_priority: Tuple[str, ...], warnings: List[str],
                 sort_direction: str = "desc") -> pd.DataFrame:
    if top_n is None:
        return out
    rank = None
    basis_name = None
    # A non-additive aggregation (avg / weighted_avg / median / distribution)
    # asked for by the user — "top 10 brokers by AVERAGE ltv" — must be ranked by
    # that metric, not by balance. Balance-first ranking only makes sense for an
    # additive measure (the metric IS a sum/count), so restrict the balance/count
    # priority to those; otherwise rank by the requested metric column directly.
    if aggregation in _ADDITIVE_AGGS:
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
    # Honour the requested direction: "bottom/smallest/lowest N" ranks ascending.
    ascending = str(sort_direction or "desc").strip().lower() == "asc"
    order = rank.sort_values(ascending=ascending, kind="mergesort").index
    out2 = out.loc[order].head(int(top_n)).reset_index(drop=True)
    warnings.append(
        f"top_n={top_n} applied, ranked by {basis_name} "
        f"({'ascending' if ascending else 'descending'})")
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
                     rank_priority, sort_for_date=False,
                     missing_policy="bucket", coverage=None):
    """Shared bar / table / treemap / heatmap grouped execution.

    ``missing_policy``: ``bucket`` (default) keeps rows with a missing grouping
    value under an explicit ``Unknown / Missing`` group so the result reconciles
    to the funded book; ``exclude`` drops them (and discloses the excluded total).
    ``coverage`` (a dict) is filled with included/excluded record + balance totals.
    """
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

    # --- missing grouping values: bucket (default) or exclude --------------
    before_n = int(len(work))
    before_bal = _balance_sum(work, balance_col)
    if missing_policy == "exclude":
        work, excluded = _exclude_missing(work, group_cols)
        if excluded:
            warnings.append(
                f"excluded {excluded} row(s) with missing/blank grouping value(s) "
                f"in {group_cols} (missing data excluded as requested)")
    else:
        work, bucketed = _bucket_missing(work, group_cols)
        if bucketed:
            warnings.append(
                f"{bucketed} row(s) with a missing/blank grouping value were grouped "
                f"under {MISSING_BUCKET_LABEL!r} (not dropped) so totals reconcile")
    work = _stringify(work, group_cols)
    if coverage is not None:
        coverage.update(_coverage_block(before_n, before_bal, work, balance_col,
                                        group_cols, missing_policy))

    out, metric_col = _grouped_aggregate(
        work, group_cols, value_col, agg, weight_col,
        balance_col if agg == "balance_sum" else None,
    )

    # --- supporting columns for a single-dimension grouped table ----------
    # An average is easy to misread without its denominator, so a single-dim
    # grouped result carries loan_count (always) and, for an average, the total.
    if len(group_cols) == 1:
        sizes = work.groupby(group_cols[0], sort=False).size()
        out["loan_count"] = _align(out, group_cols, sizes).astype(int).to_numpy()
        if agg == "avg" and value_col:
            totals = _group_sum(work, group_cols, value_col)
            out[f"{value_col}_total"] = _align(out, group_cols, totals).to_numpy()
        thin = out[out["loan_count"] < LOW_GROUP_COUNT]
        if agg in ("avg", "weighted_avg") and not thin.empty:
            warnings.append(
                f"{len(thin)} group(s) have fewer than {LOW_GROUP_COUNT} loans; "
                f"their {agg} is based on a thin sample and may be unstable")

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
                           spec.top_n, rank_priority, warnings,
                           sort_direction=(spec.sort_direction or "desc"))
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


# Loan-level context columns surfaced (when present) on a ranked "top loans"
# table, so the analyst sees the loan + its key attributes alongside the rank.
_RANKED_CONTEXT_FIELDS = (
    "loan_identifier", "current_outstanding_balance", "current_principal_balance",
    "current_loan_to_value", "original_loan_to_value", "youngest_borrower_age",
    "current_interest_rate", "geographic_region_obligor", "origination_channel",
    "broker_channel",
)


def _execute_ranked_loans(spec, work, semantics, warnings):
    """A loan-level 'top loans' ranking table: sort individual loans by a measure
    and return the head, including the loan identifier and key attributes.

    Deterministic and controlled: a missing rank column raises
    ``MIQueryExecutionError`` (converted to a validation failure upstream), never
    a raw 500.
    """
    sort_key = spec.sort_by or spec.metric
    if not sort_key:
        raise MIQueryExecutionError("ranked loans require a sort_by or metric field")
    entry = resolve_semantic_field(sort_key, semantics)
    sort_col = entry.get("canonical_field", sort_key)
    _require_column(work, sort_col, sort_key)

    out_cols: List[str] = []
    for col in _RANKED_CONTEXT_FIELDS:
        if col in work.columns and col not in out_cols:
            out_cols.append(col)
    if sort_col not in out_cols:
        out_cols.insert(0, sort_col)

    out = work[out_cols].copy()
    out[sort_col] = coerce_numeric(out[sort_col])
    before = len(out)
    out = out.dropna(subset=[sort_col])
    if len(out) < before:
        warnings.append(f"dropped {before - len(out)} loan(s) with non-numeric "
                        f"{sort_col!r} before ranking")

    ascending = str(spec.sort_direction or "desc").lower() == "asc"
    out = out.sort_values(sort_col, ascending=ascending, kind="mergesort")
    limit = spec.limit or spec.top_n or 10
    out = out.head(int(limit)).reset_index(drop=True)
    warnings.append(f"ranked top {len(out)} loan(s) by {sort_col} "
                    f"({'asc' if ascending else 'desc'})")
    return out, "table"


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
    dataset: Optional[str] = None,
    run_id: Optional[str] = None,
) -> MIQueryResult:
    """Execute a validated :class:`MIQuerySpec` against canonical data.

    ``missing_dimension_policy``: ``bucket`` (default) keeps rows with a missing
    grouping value under an explicit ``Unknown / Missing`` group so totals
    reconcile to the funded book; ``exclude`` drops them and discloses the excluded
    balance. Every result carries a ``reconciliation`` block in metadata.
    """
    if missing_dimension_policy not in ("exclude", "bucket"):
        raise MIQueryExecutionError(
            "missing_dimension_policy must be 'bucket' or 'exclude'"
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
        "dataset": dataset,
        "run_id": run_id,
    }
    # Filled by grouped paths with precise included/excluded record + balance totals.
    coverage: Dict[str, Any] = {}

    # ---- dispatch -------------------------------------------------------- #
    if spec.intent == "summary" or (spec.intent == "chart" and spec.chart_type == "none"):
        data_out, result_type = _execute_summary(spec, work, semantics, warnings, balance_col)

    elif spec.intent == "table" and spec.ranking_mode == "loan_level":
        data_out, result_type = _execute_ranked_loans(spec, work, semantics, warnings)

    elif spec.intent == "table":
        keys = []
        if spec.dimension:
            keys = [spec.dimension]
        if keys:
            data_out, result_type = _execute_grouped(
                spec, work, semantics, warnings, balance_col, keys,
                use_bucket=False, top_n_allowed=True, rank_priority=top_n_rank_priority,
                missing_policy=missing_dimension_policy, coverage=coverage,
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
            missing_policy=missing_dimension_policy, coverage=coverage,
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
            missing_policy=missing_dimension_policy, coverage=coverage,
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
            missing_policy=missing_dimension_policy, coverage=coverage,
        )

    else:
        raise MIQueryExecutionError(
            f"Unsupported intent/chart_type combination: "
            f"{spec.intent!r}/{spec.chart_type!r}"
        )

    resolved = {}
    for key in spec.referenced_fields():
        entry = semantics.get("fields", {}).get(key)
        if entry is None:
            continue
        meta = {
            "canonical_field": entry.get("canonical_field"),
            "role": entry.get("role"),
            "format": entry.get("format"),
            "business_name": entry.get("business_name") or entry.get("display_name"),
        }
        # Optional governed provenance note ("sourced from pipeline/KFI; confirm
        # authoritative for funded-book MI") — surfaced in the query lineage.
        if entry.get("source_note"):
            meta["source_note"] = entry.get("source_note")
        resolved[key] = meta

    # Reconciliation / coverage footer — every artifact can be tied back to the
    # funded-book total and discloses any excluded balance.
    measure_cols = []
    for key in (spec.metric, spec.size, spec.x, spec.y):
        if key and key in semantics.get("fields", {}):
            ent = semantics["fields"][key]
            if ent.get("role") == "metric" or ent.get("format") in (
                    "currency", "percent", "integer", "decimal"):
                col = ent.get("canonical_field", key)
                if col not in measure_cols:
                    measure_cols.append(col)
    metadata["reconciliation"] = _build_reconciliation(
        df, work, balance_col, spec, coverage, result_type, metadata,
        measure_cols=measure_cols)

    # Surface the governed derived-metric definition (e.g. "average loan balance"
    # = sum(current_outstanding_balance)/count(loans)) so the computed figure is
    # auditable against the registry — never an unexplained number.
    for name, mdef in (semantics.get("metadata", {}).get("metric_definitions", {}) or {}).items():
        if mdef.get("metric") == spec.metric and mdef.get("aggregation") == spec.aggregation:
            metadata["metric_definition"] = {"name": name, **mdef}
            break

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
