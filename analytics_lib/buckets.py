"""analytics_lib.buckets — config-driven bucket materialisation engine.

Phase 1 shared analytics library. Pure functions that read the Phase 0B bucket
edge definitions (``config/mi/buckets.yaml``) and materialise bucket columns on
a pandas DataFrame, returning the transformed frame plus a *structured issue
list* so callers can distinguish:

  * an unavailable source field (column missing entirely);
  * invalid numeric values (non-coercible / NaN where data was expected);
  * out-of-range values (numeric but outside every band);
  * scale normalisation that was applied (decimal <-> percent);
  * a malformed bucket config entry.

No I/O writes, no UI, no Streamlit/Plotly, no legacy ``analytics/`` imports.

Edge / label conventions supported per bucket (both appear in buckets.yaml):

  * ``len(labels) == len(edges) - 1`` — standard capped bands; the final edge
    is an upper cap (e.g. age ``[..., 85, 200]`` -> ``"85+"`` == ``[85, 200)``).
  * ``len(labels) == len(edges)``     — the final label is an *overflow* band
    for values ``>= edges[-1]`` (e.g. LTV ``">=100%"`` == ``[1.00, +inf)``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config_loader import load_bucket_config
from .numeric import coerce_numeric

# --------------------------------------------------------------------------- #
# Issue records
# --------------------------------------------------------------------------- #

# Issue codes (stable strings so callers can branch on them deterministically).
UNAVAILABLE_FIELD = "unavailable_field"
INVALID_NUMERIC = "invalid_numeric"
OUT_OF_RANGE = "out_of_range"
SCALE_NORMALISED = "scale_normalised"
CONFIG_ERROR = "config_error"

# Severities.
ERROR = "error"
WARNING = "warning"
INFO = "info"

# Scales that carry a decimal/percent interpretation worth normalising.
_NORMALISABLE_SCALES = {"decimal_fraction", "percent"}

# Above this column-median a "decimal fraction" column is treated as percent
# (and vice versa). LTV/PD/LGD fractions sit well below; percents sit above.
_PERCENT_DETECT_MEDIAN = 1.5


def _issue(bucket: str, field: Optional[str], code: str, severity: str,
           count: int, message: str) -> Dict[str, Any]:
    return {
        "bucket": bucket,
        "field": field,
        "code": code,
        "severity": severity,
        "count": int(count),
        "message": message,
    }


# --------------------------------------------------------------------------- #
# Scale normalisation
# --------------------------------------------------------------------------- #


def normalise_scale(values: pd.Series, scale: Optional[str]
                    ) -> Tuple[pd.Series, Optional[str]]:
    """Return ``(values, note)`` where *values* is normalised to the canonical
    representation for *scale* and *note* describes any conversion applied.

      * ``decimal_fraction`` -> canonical 0..1 fraction (percent inputs /100).
      * ``percent``          -> canonical 0..100 percent (fraction inputs *100).

    The decision is made at the *column* level using the median of the valid
    values, so a single oddball value never flips the whole column.
    """
    if scale not in _NORMALISABLE_SCALES:
        return values, None

    valid = values.dropna()
    if valid.empty:
        return values, None

    median = float(valid.median())
    if scale == "decimal_fraction" and median > _PERCENT_DETECT_MEDIAN:
        return values / 100.0, "percent inputs divided by 100 -> fraction"
    if scale == "percent" and median <= _PERCENT_DETECT_MEDIAN:
        return values * 100.0, "fraction inputs multiplied by 100 -> percent"
    return values, None


# --------------------------------------------------------------------------- #
# Edge / label resolution
# --------------------------------------------------------------------------- #


def _resolve_bins(edges: List[float], labels: List[str]
                  ) -> Tuple[List[float], List[str]]:
    """Resolve cut edges + labels for the two supported conventions.

    Raises ``ValueError`` if the label count matches neither convention.
    """
    if len(edges) < 2:
        raise ValueError("a bucket needs at least 2 edges")
    if len(labels) == len(edges) - 1:
        return list(edges), list(labels)
    if len(labels) == len(edges):
        # Final label is an overflow band [edges[-1], +inf).
        return list(edges) + [math.inf], list(labels)
    raise ValueError(
        f"labels ({len(labels)}) must equal edges-1 ({len(edges) - 1}) "
        f"or edges ({len(edges)})"
    )


# --------------------------------------------------------------------------- #
# Single-bucket application
# --------------------------------------------------------------------------- #


def apply_bucket(df: pd.DataFrame, bucket_key: str, spec: Dict[str, Any],
                 target: str = "key") -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """Compute a single bucket Series from *spec*.

    Returns ``(series_or_None, issues)``. ``series`` is ``None`` only when the
    source field is unavailable or the config is malformed (an ``error`` issue
    is recorded in that case).
    """
    issues: List[Dict[str, Any]] = []
    source_field = spec.get("source_field")
    edges = spec.get("edges")
    labels = spec.get("labels")
    scale = spec.get("scale")
    right_closed = bool(spec.get("right_closed", False))

    if not source_field:
        issues.append(_issue(bucket_key, None, CONFIG_ERROR, ERROR, 0,
                             "bucket spec is missing 'source_field'"))
        return None, issues

    if source_field not in df.columns:
        issues.append(_issue(
            bucket_key, source_field, UNAVAILABLE_FIELD, ERROR, len(df),
            f"source field {source_field!r} not present in DataFrame"))
        return None, issues

    if not isinstance(edges, list) or not isinstance(labels, list):
        issues.append(_issue(bucket_key, source_field, CONFIG_ERROR, ERROR, 0,
                             "bucket spec needs list 'edges' and 'labels'"))
        return None, issues

    raw = df[source_field]
    # Deterministic shared parser: tolerates comma / currency / accounting
    # formatting so amounts like "111,757.38" are not silently coerced to NaN.
    numeric = coerce_numeric(raw)

    # Invalid numeric = was populated but could not be coerced to a number.
    invalid_mask = numeric.isna() & raw.notna()
    n_invalid = int(invalid_mask.sum())
    if n_invalid:
        issues.append(_issue(
            bucket_key, source_field, INVALID_NUMERIC, WARNING, n_invalid,
            f"{n_invalid} value(s) in {source_field!r} are not numeric"))

    numeric, note = normalise_scale(numeric, scale)
    if note:
        issues.append(_issue(bucket_key, source_field, SCALE_NORMALISED, INFO,
                             int(numeric.notna().sum()),
                             f"scale={scale}: {note}"))

    try:
        cut_edges, cut_labels = _resolve_bins(edges, labels)
    except ValueError as exc:
        issues.append(_issue(bucket_key, source_field, CONFIG_ERROR, ERROR, 0,
                             str(exc)))
        return None, issues

    series = pd.cut(
        numeric, bins=cut_edges, labels=cut_labels,
        right=right_closed, include_lowest=True, ordered=False,
    ).astype("object")

    # Out of range = a valid number that fell outside every band.
    out_mask = series.isna() & numeric.notna()
    n_out = int(out_mask.sum())
    if n_out:
        issues.append(_issue(
            bucket_key, source_field, OUT_OF_RANGE, WARNING, n_out,
            f"{n_out} numeric value(s) in {source_field!r} fell outside all "
            f"bands"))

    series.name = spec.get("semantic_field") if target == "semantic_field" \
        else bucket_key
    return series, issues


# --------------------------------------------------------------------------- #
# Whole-config materialisation
# --------------------------------------------------------------------------- #


def materialise_buckets(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    buckets: Optional[List[str]] = None,
    target: str = "key",
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Optional[str]]]:
    """Materialise bucket columns on a copy of *df*.

    Parameters
    ----------
    df:
        Input loan-level frame (not mutated).
    config:
        ``{bucket_key: spec}`` mapping. Defaults to ``config/mi/buckets.yaml``.
    buckets:
        Optional subset of bucket keys to apply (default: all in *config*).
    target:
        ``"key"`` (default) writes columns under the bucket key; ``"semantic_field"``
        writes under the registry semantic-field name.

    Returns
    -------
    (df_out, issues, applied)
        * ``df_out``  — copy of *df* with materialised bucket columns added;
        * ``issues``  — structured issue records (see module constants);
        * ``applied`` — ``{bucket_key: output_column_or_None}`` so callers know
          exactly which buckets produced a column.
    """
    if config is None:
        config = load_bucket_config()

    keys = list(buckets) if buckets is not None else list(config.keys())
    out = df.copy()
    all_issues: List[Dict[str, Any]] = []
    applied: Dict[str, Optional[str]] = {}

    for key in keys:
        spec = config.get(key)
        if spec is None:
            all_issues.append(_issue(key, None, CONFIG_ERROR, ERROR, 0,
                                    f"bucket {key!r} not found in config"))
            applied[key] = None
            continue
        series, issues = apply_bucket(out, key, spec, target=target)
        all_issues.extend(issues)
        if series is None:
            applied[key] = None
            continue
        col = str(series.name)
        out[col] = series.values
        applied[key] = col

    return out, all_issues, applied
