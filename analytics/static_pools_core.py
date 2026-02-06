# static_pools_core.py
"""
Static Pools Core (product-agnostic)

Purpose
-------
Provide reusable mechanics for Static Pools-style analyses:
- Cohorting by origination (year/month)
- Aggregations by segment (region/product/risk bucket)
- Simple measure composition (e.g., outstanding = principal + accrued interest)
- Status transition (migration) flows for Sankey

This module is intentionally product-agnostic:
- It does not define what "status" categories mean.
- It does not define how "risk_bucket" is constructed (LTV vs risk grade vs FICO etc.)
- It does not define how prepayment flows are computed (adapters can supply prepayment_amount).

Adapters (e.g., static_pool_ere.py) should:
- Map/rename their canonical columns to the names specified in StaticPoolsSpec
- Optionally apply product-specific transforms:
  - status taxonomy normalization
  - risk bucket construction (e.g., Original LTV buckets)
  - prepayment_amount calculation (if available/derivable)

Outputs are tidy pandas DataFrames suitable for plotting in Streamlit/Plotly.

Author: SME Loan Data Project
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd


# ----------------------------
# Spec / Contract
# ----------------------------

@dataclass(frozen=True)
class StaticPoolsSpec:
    """
    Column bindings for the input dataframe.

    Two modes are supported:
    - Provide origination_year directly, OR provide origination_date (core will derive year/month).
    - Provide either explicit transitions (status_from/status_to) OR multiple snapshots and
      let core derive transitions between two as_of_date values.
    """
    # Identifiers & dates
    account_id: str = "account_id"
    as_of_date: str = "as_of_date"

    # Origination cohorting (either provide origination_year or origination_date)
    origination_year: str = "origination_year"      # optional if origination_date present
    origination_month: str = "origination_month"    # optional derived output if origination_date present
    origination_date: str = "origination_date"      # optional input; if present, core derives year/month

    # Dimensions
    geo_region: str = "geo_region"
    product_type: str = "product_type"
    risk_bucket: str = "risk_bucket"                # optional segmentation bucket

    # Status
    account_status: str = "account_status"

    # Measures
    principal_outstanding: str = "principal_outstanding"
    interest_accrued: str = "interest_accrued"      # optional
    prepayment_amount: str = "prepayment_amount"    # optional (period flow aligned to as_of_date)

    # Optional explicit transitions
    status_from: str = "status_from"
    status_to: str = "status_to"
    transition_weight: str = "transition_weight"    # optional weight for Sankey


STATIC_POOLS_REQUIRED_COLS = [
    "account_id",
    "as_of_date",
    # origination_year is required unless origination_date is present
    "geo_region",
    "product_type",
    "account_status",
    "principal_outstanding",
]

STATIC_POOLS_OPTIONAL_COLS = [
    "origination_year",
    "origination_month",
    "origination_date",
    "interest_accrued",
    "prepayment_amount",
    "risk_bucket",
    "status_from",
    "status_to",
    "transition_weight",
]


# ----------------------------
# Utilities / Validation
# ----------------------------

def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        out = df.copy()
        out[col] = pd.to_datetime(out[col], errors="coerce")
        return out
    return df


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def validate_static_pools_inputs(df: pd.DataFrame, spec: StaticPoolsSpec) -> None:
    """
    Validates that the input has the minimum required columns to build a panel.
    """
    # Validate required columns using spec bindings (not hard-coded default names)
    required = [
        spec.account_id,
        spec.as_of_date,
        spec.geo_region,
        spec.product_type,
        spec.account_status,
        spec.principal_outstanding,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Static Pools input missing required columns: {missing}. "
            f"Required (spec-bound): {required}. "
            f"Optional: {STATIC_POOLS_OPTIONAL_COLS}."
        )

    # as_of_date parseable
    tmp = pd.to_datetime(df[spec.as_of_date], errors="coerce")
    if tmp.isna().any():
        bad = df.loc[tmp.isna(), spec.as_of_date].head(10).tolist()
        raise ValueError(f"as_of_date has non-parseable values (sample): {bad}")

    # account_id present and non-null
    if df[spec.account_id].isna().any():
        raise ValueError("account_id contains nulls.")

    # Cohorting requirement: either origination_year OR origination_date must be present
    has_year = spec.origination_year in df.columns
    has_date = spec.origination_date in df.columns
    if not (has_year or has_date):
        raise ValueError("Static Pools requires either origination_year or origination_date to be present.")

def add_missing_optional_columns(df: pd.DataFrame, spec: StaticPoolsSpec) -> pd.DataFrame:
    """
    Ensures optional fields exist to simplify downstream aggregation logic.
    Missing optional measures default to 0, missing optional dimensions default to 'ALL'.
    """
    out = df.copy()

    if spec.interest_accrued not in out.columns:
        out[spec.interest_accrued] = 0.0
    if spec.prepayment_amount not in out.columns:
        out[spec.prepayment_amount] = 0.0
    if spec.risk_bucket not in out.columns:
        out[spec.risk_bucket] = "ALL"

    return out


def derive_origination_cohorts(df: pd.DataFrame, spec: StaticPoolsSpec) -> pd.DataFrame:
    """
    If origination_date is provided, derive origination_year and origination_month.
    If origination_year already exists, leaves as-is.
    """
    out = df.copy()

    # If origination_date exists, derive year/month as needed
    if spec.origination_date in out.columns:
        out = _ensure_datetime(out, spec.origination_date)

        if spec.origination_year not in out.columns:
            out[spec.origination_year] = out[spec.origination_date].dt.year

        if spec.origination_month not in out.columns:
            # Represent month as YYYY-MM (string) to be chart-friendly and stable
            out[spec.origination_month] = (
                out[spec.origination_date]
                .dt.to_period("M")
                .astype(str)
            )

    # Validate origination_year numeric/int-like
    if spec.origination_year in out.columns:
        coerced = pd.to_numeric(out[spec.origination_year], errors="coerce")
        if coerced.isna().any():
            raise ValueError("origination_year must be numeric/int-like (or derivable from origination_date).")
        out[spec.origination_year] = coerced.astype(int)

    return out


def _normalize_segment_dims(
    base_dims: Sequence[str],
    spec: StaticPoolsSpec,
    include_geo_region: bool,
    include_product_type: bool,
    include_risk_bucket: bool,
) -> List[str]:
    dims = list(base_dims)
    if include_geo_region and spec.geo_region not in dims:
        dims.append(spec.geo_region)
    if include_product_type and spec.product_type not in dims:
        dims.append(spec.product_type)
    if include_risk_bucket and spec.risk_bucket not in dims:
        dims.append(spec.risk_bucket)
    return dims


# ----------------------------
# Panel Builder
# ----------------------------

TransformFn = Callable[[pd.DataFrame], pd.DataFrame]


def build_static_pools_panel(
    df: pd.DataFrame,
    spec: StaticPoolsSpec = StaticPoolsSpec(),
    *,
    transforms: Optional[Sequence[TransformFn]] = None,
) -> pd.DataFrame:
    """
    Builds a clean, consistent "Static Pools panel" dataframe:
    - Validates minimum input contract
    - Parses dates
    - Derives origination cohorts (year/month) from origination_date if provided
    - Adds missing optional columns with sensible defaults
    - Coerces numeric measure columns
    - Applies optional product-specific transforms (status normalization, risk bucketing, etc.)

    Returns:
        DataFrame: panel ready for chart builders.
    """
    validate_static_pools_inputs(df, spec)

    out = df.copy()
    out = _ensure_datetime(out, spec.as_of_date)

    # Cohorts
    out = derive_origination_cohorts(out, spec)

    # Optional columns (interest/prepay/risk bucket)
    out = add_missing_optional_columns(out, spec)

    # Coerce measures to numeric
    out = _coerce_numeric(out, [spec.principal_outstanding, spec.interest_accrued, spec.prepayment_amount])

    # Apply product-specific transforms last (so they can rely on derived fields existing)
    if transforms:
        for fn in transforms:
            out = fn(out)

    return out

def build_vintage_metric_series(
    panel: pd.DataFrame,
    spec: StaticPoolsSpec = StaticPoolsSpec(),
    *,
    metric_col: str,
    agg_func: str = "mean",  # 'mean', 'sum', 'weighted_mean'
    weight_col: Optional[str] = None,
    include_geo_region: bool = True,
    include_product_type: bool = True,
    include_risk_bucket: bool = False,
) -> pd.DataFrame:
    """
    Universal Engine for Vintage Curves. 
    Calculates metrics based on 'Months on Book' (Age) rather than Calendar Time.
    """
    df = panel.copy()
    df = _ensure_datetime(df, spec.as_of_date)
    df = _ensure_datetime(df, spec.origination_date)

    # 1. Calculate Months on Book (MOB)
    # This is the 'Age' of the loan at the time of the snapshot
    df['mob'] = (
        (df[spec.as_of_date].dt.year - df[spec.origination_date].dt.year) * 12 + 
        (df[spec.as_of_date].dt.month - df[spec.origination_date].dt.month)
    )

    # 2. Define Dimensions
    dims = _normalize_segment_dims(
        base_dims=[spec.origination_year, 'mob'],
        spec=spec,
        include_geo_region=include_geo_region,
        include_product_type=include_product_type,
        include_risk_bucket=include_risk_bucket,
    )

    # 3. Aggregation
    if agg_func == "weighted_mean" and weight_col:
        # Avoid division by zero
        def w_avg(g):
            weights = g[weight_col]
            if weights.sum() == 0: return 0
            return (g[metric_col] * weights).sum() / weights.sum()
        
        agg = df.groupby(dims, dropna=False).apply(w_avg).reset_index(name=metric_col)
    else:
        agg = df.groupby(dims, dropna=False, as_index=False)[metric_col].agg(agg_func)

    return agg

# ----------------------------
# (3) Sankey: Risk Migration by Account Status
# ----------------------------

def _derive_status_transitions_from_snapshots(
    panel: pd.DataFrame,
    spec: StaticPoolsSpec,
    *,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
    dims: Sequence[str],
    weight_col: str,
) -> pd.DataFrame:
    df = panel.copy()
    df = _ensure_datetime(df, spec.as_of_date)

    left = df[df[spec.as_of_date] == date_from].copy()
    right = df[df[spec.as_of_date] == date_to].copy()

    keep_left = [spec.account_id, spec.account_status, weight_col] + list(dims)
    keep_right = [spec.account_id, spec.account_status] + list(dims)

    left = left[keep_left].rename(columns={spec.account_status: "status_from"})
    right = right[keep_right].rename(columns={spec.account_status: "status_to"})

    merged = pd.merge(left, right, on=[spec.account_id] + list(dims), how="inner")
    return merged


def build_status_migration_sankey(
    panel: pd.DataFrame,
    spec: StaticPoolsSpec = StaticPoolsSpec(),
    *,
    include_geo_region: bool = True,
    include_product_type: bool = True,
    include_risk_bucket: bool = True,
    date_from: Optional[Union[str, pd.Timestamp]] = None,
    date_to: Optional[Union[str, pd.Timestamp]] = None,
    status_order: Optional[List[str]] = None,
    weight: str = "principal",  # "principal" or "outstanding"
) -> Dict[str, pd.DataFrame]:
    """
    Produces Sankey inputs:
      nodes: node_label, node_index
      links: [segment dims...], source, target, value, status_from, status_to

    Two ways to get transitions:
    A) Explicit transitions in panel via spec.status_from/spec.status_to columns
    B) Derived transitions between two as_of_date snapshots
    """
    df = panel.copy()
    df = _ensure_datetime(df, spec.as_of_date)

    dims = _normalize_segment_dims(
        base_dims=[],
        spec=spec,
        include_geo_region=include_geo_region,
        include_product_type=include_product_type,
        include_risk_bucket=include_risk_bucket,
    )

    # Weight column
    if weight == "outstanding":
        df["_w_base"] = df[spec.principal_outstanding] + df[spec.interest_accrued]
    else:
        df["_w_base"] = df[spec.principal_outstanding]

    # Case A: explicit transitions
    if spec.status_from in df.columns and spec.status_to in df.columns:
        trans = df.copy()
        trans["status_from"] = trans[spec.status_from]
        trans["status_to"] = trans[spec.status_to]

        if spec.transition_weight in trans.columns:
            trans = _coerce_numeric(trans, [spec.transition_weight])
            trans["_w"] = trans[spec.transition_weight]
        else:
            trans["_w"] = trans["_w_base"]

    # Case B: derive transitions from snapshots
    else:
        available = sorted(pd.to_datetime(df[spec.as_of_date].unique()))
        if len(available) < 2:
            raise ValueError(
                "Sankey requires either explicit status_from/status_to columns or at least "
                "two as_of_date snapshots."
            )

        d0 = pd.to_datetime(date_from) if date_from is not None else available[-2]
        d1 = pd.to_datetime(date_to) if date_to is not None else available[-1]

        trans = _derive_status_transitions_from_snapshots(
            panel=df,
            spec=spec,
            date_from=d0,
            date_to=d1,
            dims=dims,
            weight_col="_w_base",
        )
        trans["_w"] = trans["_w_base"]

    # Aggregate flows
    group_cols = list(dims) + ["status_from", "status_to"]
    links = (
        trans.groupby(group_cols, dropna=False, as_index=False)
             .agg(value=("_w", "sum"))
    )

    # Nodes
    statuses = pd.Index(
        pd.concat([links["status_from"], links["status_to"]], ignore_index=True).unique()
    )

    if status_order:
        ordered = [s for s in status_order if s in set(statuses)]
        remaining = [s for s in statuses.tolist() if s not in set(ordered)]
        node_labels = ordered + remaining
    else:
        node_labels = sorted(statuses.tolist())

    nodes = pd.DataFrame({"node_label": node_labels})
    nodes["node_index"] = range(len(nodes))
    node_map = dict(zip(nodes["node_label"], nodes["node_index"]))

    links["source"] = links["status_from"].map(node_map)
    links["target"] = links["status_to"].map(node_map)

    links_out = links[list(dims) + ["source", "target", "value", "status_from", "status_to"]].copy()
    return {"nodes": nodes, "links": links_out}


# ----------------------------
# (4) Area Chart: Portfolio Run-off (stocks + flows)
# ----------------------------

def build_portfolio_runoff_timeseries(
    panel: pd.DataFrame,
    spec: StaticPoolsSpec = StaticPoolsSpec(),
    *,
    include_geo_region: bool = True,
    include_product_type: bool = True,
    include_risk_bucket: bool = True,
) -> pd.DataFrame:
    """
    Tidy output:
      as_of_date | [segment dims...] | component | amount

    component in {"principal_outstanding", "interest_accrued", "prepayment_amount"}

    Notes:
    - principal_outstanding & interest_accrued are "stock" measures at each as_of_date.
    - prepayment_amount is expected to be a "flow" aligned to as_of_date.
      If not available, adapter can leave it at 0 for v1.
    """
    df = panel.copy()
    df = _ensure_datetime(df, spec.as_of_date)

    dims = _normalize_segment_dims(
        base_dims=[spec.as_of_date],
        spec=spec,
        include_geo_region=include_geo_region,
        include_product_type=include_product_type,
        include_risk_bucket=include_risk_bucket,
    )

    agg = (
        df.groupby(dims, dropna=False, as_index=False)
          .agg(
              principal_outstanding=(spec.principal_outstanding, "sum"),
              interest_accrued=(spec.interest_accrued, "sum"),
              prepayment_amount=(spec.prepayment_amount, "sum"),
          )
    )

    tidy = agg.melt(
        id_vars=dims,
        value_vars=["principal_outstanding", "interest_accrued", "prepayment_amount"],
        var_name="component",
        value_name="amount",
    )
    return tidy


# ----------------------------
# Convenience: segment label
# ----------------------------

def add_segment_label(
    df: pd.DataFrame,
    spec: StaticPoolsSpec = StaticPoolsSpec(),
    *,
    dims: Sequence[str] = ("geo_region", "product_type", "risk_bucket"),
    label_col: str = "segment",
    sep: str = " | ",
) -> pd.DataFrame:
    """
    Adds a human-readable segment label by concatenating dimension values.
    Useful for Plotly color grouping where you prefer a single series identifier.
    """
    out = df.copy()

    cols: List[str] = []
    for d in dims:
        if hasattr(spec, d):
            col = getattr(spec, d)
        else:
            col = d
        if col in out.columns:
            cols.append(col)

    if not cols:
        out[label_col] = "ALL"
        return out

    out[label_col] = out[cols].astype(str).agg(sep.join, axis=1)
    return out
