"""mi_agent.quantile_buckets — asset-agnostic quantile bucketing (Phase 6 Step 0).

Trakt is asset-agnostic, so balance / interest-rate / time-on-book buckets must
NOT default to product-specific fixed bands. The default strategy for these
dimensions is quantile (quartile) bucketing over the *selected population*,
unless a client/asset-specific config overrides it. Asset-specific bands that are
already defined (LTV, borrower age) are preserved by the existing
``analytics_lib.buckets`` engine and are not touched here.

Returns ``(series_or_None, issues)``: on insufficient data the engine returns a
structured ``quantile_bucket_insufficient_data`` issue and does NOT silently fall
back to arbitrary fixed bands.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from mi_agent.states.models import INFO, WARNING, make_issue

QUANTILE_BUCKET_INSUFFICIENT_DATA = "quantile_bucket_insufficient_data"

# Dimensions that default to quantile bucketing, mapped to their source field.
QUANTILE_DIMENSIONS: Dict[str, str] = {
    "balance_band": "current_outstanding_balance",
    "interest_rate_bucket": "current_interest_rate",
    "time_on_book_bucket": "months_on_book",
}


def quantile_bucket_series(series: pd.Series, q: int = 4,
                           labels: Optional[List[str]] = None
                           ) -> Tuple[Optional[pd.Series], List[Dict[str, Any]]]:
    """Quantile-bucket a numeric Series into *q* bands (default quartiles).

    Returns ``(banded_series, issues)``. The banded series is index-aligned to
    the input (NaN where the source is null). On insufficient distinct data the
    series is ``None`` and an issue is recorded.
    """
    issues: List[Dict[str, Any]] = []
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.nunique() < 2:
        issues.append(make_issue(
            QUANTILE_BUCKET_INSUFFICIENT_DATA, WARNING,
            f"need >=2 distinct values for quantile bucketing; got "
            f"{valid.nunique()}", field=series.name))
        return None, issues

    effective_q = min(q, valid.nunique())
    try:
        banded = pd.qcut(numeric, effective_q, duplicates="drop")
    except (ValueError, IndexError) as exc:  # pragma: no cover - defensive
        issues.append(make_issue(
            QUANTILE_BUCKET_INSUFFICIENT_DATA, WARNING,
            f"quantile bucketing failed: {exc}", field=series.name))
        return None, issues

    cats = list(banded.cat.categories)
    n = len(cats)
    if n < 1:
        issues.append(make_issue(
            QUANTILE_BUCKET_INSUFFICIENT_DATA, WARNING,
            "quantile bucketing produced no bins", field=series.name))
        return None, issues

    if labels and len(labels) >= n:
        use_labels = labels[:n]
    else:
        use_labels = [f"Q{i + 1}" for i in range(n)]
    mapping = {cat: use_labels[i] for i, cat in enumerate(cats)}
    out = banded.map(mapping).astype("object")
    out.name = series.name
    if effective_q < q:
        issues.append(make_issue(
            "quantile_bucket_reduced", INFO,
            f"requested {q} quantiles; produced {n} (duplicate edges/low "
            f"cardinality)", field=series.name, count=n))
    return out, issues


def materialise_quantile_bucket(df: pd.DataFrame, dimension: str, *,
                                source_field: Optional[str] = None,
                                q: int = 4, out_col: Optional[str] = None
                                ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Add a quantile-bucket column for a recognised quantile *dimension*.

    Returns ``(df_out, issues)``. If the source column is unavailable, records a
    structured issue and returns the frame unchanged.
    """
    source_field = source_field or QUANTILE_DIMENSIONS.get(dimension, dimension)
    out_col = out_col or dimension
    if source_field not in df.columns:
        return df, [make_issue(
            QUANTILE_BUCKET_INSUFFICIENT_DATA, WARNING,
            f"source field {source_field!r} for quantile dimension "
            f"{dimension!r} not present", field=source_field)]
    banded, issues = quantile_bucket_series(df[source_field], q=q)
    if banded is None:
        return df, issues
    out = df.copy()
    out[out_col] = banded.values
    return out, issues
