#!/usr/bin/env python3
"""
mi_dataset_profile.py

ONE per-field metadata profile for a prepared MI dataset, plus the data-aware
query validation built on it.

Every consumer reads THIS profile rather than independently inferring field
meaning, numeric parsing, bucket availability, or display scale:

  * the query workflow validates a spec against real data values here
    (a metric must have numeric values, a dimension must have non-blank values,
    a loan-level x/y/size must have at least one usable row);
  * the API attaches per-field display hints (format + storage scale) from here,
    so React never guesses whether a percent is a fraction (0.51) or points (51);
  * ``mi_agent_api/mi_dataset_contract.py`` and the static review generator build
    their /health + review metadata from here.

It is deterministic, side-effect free, and never mutates the dataframe.
"""

from __future__ import annotations

import os
import threading
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import pandas as pd

from analytics_lib.numeric import coerce_numeric

# Semantic types surfaced to consumers.
TYPE_CURRENCY = "currency"
TYPE_PERCENT = "percent"
TYPE_INTEGER = "integer"
TYPE_STRING = "string"
TYPE_DATE = "date"
TYPE_BUCKET = "bucket"

# Storage scale for a percent field: a fraction (0.51 == 51%) or points (51).
PERCENT_FRACTION = "percent_fraction"
PERCENT_POINTS = "percent_points"
# A percent column whose median exceeds this is stored as points, else a fraction.
_PERCENT_MEDIAN = 1.5

_NUMERIC_TYPES = {TYPE_CURRENCY, TYPE_PERCENT, TYPE_INTEGER}

# Registry format -> React/display format token (mirrors adapters._FORMAT_MAP).
_DISPLAY_FORMAT = {
    TYPE_CURRENCY: "gbp",
    TYPE_PERCENT: "pct",
    TYPE_INTEGER: "number",
    "decimal": "decimal",
    TYPE_DATE: "date",
    TYPE_STRING: "text",
    TYPE_BUCKET: "text",
}


def _non_blank_count(series: pd.Series) -> int:
    """Rows that are neither NaN nor blank/whitespace strings."""
    if series is None:
        return 0
    s = series
    notna = s.notna()
    text = s.astype(str).str.strip()
    blank = text.eq("") | text.str.lower().isin(["nan", "nat", "none", "<na>"])
    return int((notna & ~blank).sum())


def percent_storage_scale(series: pd.Series) -> str:
    """The storage scale of a percent column: ``percent_fraction`` | ``percent_points``.

    The single place this is decided. A column whose non-null median is <= 1.5 is
    a fraction (0.51 == 51%); otherwise it is already in points (51 == 51%).
    """
    s = coerce_numeric(series).dropna()
    if s.empty:
        return PERCENT_FRACTION
    return PERCENT_FRACTION if float(s.median()) <= _PERCENT_MEDIAN else PERCENT_POINTS


def _canonical_index(semantics: dict) -> Dict[str, dict]:
    """Map ``canonical_field`` -> semantic entry (first-seen wins)."""
    out: Dict[str, dict] = {}
    for entry in (semantics.get("fields", {}) or {}).values():
        canon = entry.get("canonical_field")
        if canon and canon not in out:
            out[canon] = entry
    return out


def _semantic_type(entry: Optional[dict], column: str) -> str:
    """Semantic type for a column from its registry entry, falling back to the
    column name (``*_bucket`` -> bucket; otherwise string)."""
    if column.endswith("_bucket"):
        return TYPE_BUCKET
    fmt = (entry or {}).get("format")
    if fmt == "currency":
        return TYPE_CURRENCY
    if fmt == "percent":
        return TYPE_PERCENT
    if fmt == "integer":
        return TYPE_INTEGER
    if fmt == "decimal":
        return "decimal"
    if fmt == "date":
        return TYPE_DATE
    if fmt == "string":
        return TYPE_STRING
    role = (entry or {}).get("role")
    if role == "date":
        return TYPE_DATE
    if role == "metric":
        return "decimal"
    return TYPE_STRING


def _is_groupable(entry: Optional[dict], semantic_type: str) -> bool:
    if semantic_type == TYPE_BUCKET:
        return True
    role = (entry or {}).get("role")
    return role in ("dimension", "date", "flag")


def profile_field(name: str, series: pd.Series, entry: Optional[dict]) -> Dict[str, Any]:
    """The metadata profile for one column."""
    semantic_type = _semantic_type(entry, name)
    numeric_like = semantic_type in _NUMERIC_TYPES or semantic_type == "decimal"
    non_null = _non_blank_count(series)
    numeric_parse = int(coerce_numeric(series).notna().sum()) if numeric_like else 0

    storage_scale: Optional[str] = None
    if semantic_type == TYPE_PERCENT:
        storage_scale = percent_storage_scale(series)

    groupable = _is_groupable(entry, semantic_type)
    return {
        "field": name,
        "semantic_type": semantic_type,
        "storage_scale": storage_scale,
        "display_format": _DISPLAY_FORMAT.get(semantic_type, "text"),
        "non_null": non_null,
        "numeric_parse": numeric_parse,
        # A dimension is usable only if groupable AND it has at least one value.
        "dimension_available": bool(groupable and non_null > 0),
        # A metric is usable only if numeric-typed AND at least one value parses.
        "metric_available": bool(numeric_like and numeric_parse > 0),
        "canonical_field": (entry or {}).get("canonical_field", name),
        "business_name": (entry or {}).get("business_name")
        or (entry or {}).get("display_name") or name,
        "role": (entry or {}).get("role"),
    }


#: Kill switch, matching ``TRAKT_SERVING_CACHE`` / ``TRAKT_CONFIG_CACHE``.
_PROFILE_CACHE_ENV = "TRAKT_PROFILE_CACHE"
_OFF = {"0", "off", "false", "no"}

#: ``id(df) -> (weakref(df), semantics, profile)``. Bounded LRU.
_PROFILE_CACHE: "OrderedDict[int, tuple]" = OrderedDict()
_PROFILE_CACHE_MAX = 16
_PROFILE_LOCK = threading.Lock()
_PROFILE_STATS = {"hits": 0, "misses": 0}


def _profile_cache_enabled() -> bool:
    return str(os.environ.get(_PROFILE_CACHE_ENV, "on")).strip().lower() not in _OFF


def reset_profile_cache() -> None:
    """Drop every memoised profile. For tests and for an operator forcing a rebuild."""
    with _PROFILE_LOCK:
        _PROFILE_CACHE.clear()
        _PROFILE_STATS["hits"] = 0
        _PROFILE_STATS["misses"] = 0


def profile_cache_stats() -> Dict[str, int]:
    with _PROFILE_LOCK:
        return {**_PROFILE_STATS, "entries": len(_PROFILE_CACHE)}


def profile_dataset(df: pd.DataFrame, semantics: dict) -> Dict[str, Any]:
    """Profile every column of a prepared MI dataset against the semantic registry.

    Returns ``{"fields": {col: profile}, "display_hints": {col: {format, scale}}}``.
    ``display_hints`` is the compact map consumers attach to artifacts so the
    frontend formats values without guessing the scale.

    Memoised on the identity of the frame object
    -------------------------------------------
    Profiling counts non-blank values in every column, so it costs a full pass
    over the frame: measured at 1,701 ms of a 1,898 ms warm MI query over 100k
    rows — 80% of the request, spent re-deriving the structure of a dataset that
    had not changed.

    The key is ``id(df)`` guarded by a weak reference, which is **exact** rather
    than approximate. The frames themselves are already cached upstream by
    content signature (``data_source._ACTIVE_CACHE``), so the same unchanged
    dataset is the same object; a republished snapshot is a NEW object and
    therefore a miss. If an id is ever reused by a different object the weakref
    no longer resolves to it and the entry is rejected, so a stale profile cannot
    be served. Tenant and snapshot isolation follow for free: a different
    tenant's frame is a different object.

    That is also why there is no signature parameter to thread through: every
    caller — MI Query, Copilot, ``mi_dataset_contract`` — benefits without
    changing a signature, and none of them has to be trusted to pass the right
    identity.

    ``mi_agent_api.serving_cache`` is deliberately not reused: ``mi_agent`` is
    domain code and importing the API layer here would invert the dependency
    direction. Same reasoning as ``trakt_core.config_cache``.
    """
    if not _profile_cache_enabled():
        return _profile_dataset_uncached(df, semantics)

    key = id(df)
    with _PROFILE_LOCK:
        entry = _PROFILE_CACHE.get(key)
        if entry is not None:
            ref, cached_semantics, profile = entry
            # BOTH identities must still hold: the same live frame, profiled
            # against the same semantics registry object.
            if ref() is df and cached_semantics is semantics:
                _PROFILE_CACHE.move_to_end(key)
                _PROFILE_STATS["hits"] += 1
                return profile
            del _PROFILE_CACHE[key]

    # Built outside the lock: this is a full pass over the frame, and holding a
    # global lock across it would serialise every worker thread on the first
    # request after a snapshot change.
    profile = _profile_dataset_uncached(df, semantics)

    try:
        ref = weakref.ref(df)
    except TypeError:  # pragma: no cover - a frame that cannot be weak-referenced
        return profile

    with _PROFILE_LOCK:
        _PROFILE_STATS["misses"] += 1
        _PROFILE_CACHE[key] = (ref, semantics, profile)
        _PROFILE_CACHE.move_to_end(key)
        while len(_PROFILE_CACHE) > _PROFILE_CACHE_MAX:
            _PROFILE_CACHE.popitem(last=False)
    return profile


def _profile_dataset_uncached(df: pd.DataFrame, semantics: dict) -> Dict[str, Any]:
    """The profile itself. Unchanged from before the memo existed."""
    canon_index = _canonical_index(semantics)
    fields: Dict[str, Dict[str, Any]] = {}
    display_hints: Dict[str, Dict[str, Any]] = {}
    # Positional access so a duplicate column name yields a Series (not a
    # DataFrame); the executor's duplicate-column guard still produces the
    # controlled error during execution.
    for i, col in enumerate(df.columns):
        entry = canon_index.get(col)
        prof = profile_field(col, df.iloc[:, i], entry)
        fields[col] = prof
        display_hints[col] = {
            "format": prof["display_format"],
            "scale": prof["storage_scale"],
        }
    return {"fields": fields, "display_hints": display_hints}


def display_hint_for(profile: Dict[str, Any], column: str) -> Dict[str, Any]:
    """The {format, scale} hint for an emitted result column, stripping the common
    aggregation suffixes so ``current_loan_to_value_weighted_avg`` resolves to the
    base field's hint. Concentration share columns are always points percentages."""
    hints = (profile or {}).get("display_hints", {})
    if column in hints:
        return hints[column]
    if column.endswith("_pct") or "concentration" in column:
        return {"format": "pct", "scale": PERCENT_POINTS}
    base = column
    for suffix in ("_weighted_avg", "_count_distinct", "_sum", "_avg",
                   "_median", "_count"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base in hints:
        return hints[base]
    return {"format": "number", "scale": None}


# --------------------------------------------------------------------------- #
# Data-aware query validation (used by the workflow before "Validation: Passed")
# --------------------------------------------------------------------------- #

def _field_profile(profile: Dict[str, Any], semantics: dict, key: str) -> Optional[Dict[str, Any]]:
    """The column profile for a semantic field key (resolving via canonical_field)."""
    fields = profile.get("fields", {})
    entry = (semantics.get("fields", {}) or {}).get(key)
    canon = (entry or {}).get("canonical_field", key)
    return fields.get(canon) or fields.get(key)


def validate_query_data(spec, df: pd.DataFrame, semantics: dict,
                        profile: Dict[str, Any]) -> List[Dict[str, str]]:
    """Validate a spec against ACTUAL data values, returning a list of errors.

    Each error is ``{"field", "reason", "detail"}``. Empty list == data is usable.
    Complements ``mi_query_validator`` (which checks names/roles only). Covers:
    metric has numeric values; grouping dimension(s) have non-blank values; filter
    fields exist and have values; loan-level x/y/size have usable numeric rows.
    """
    errors: List[Dict[str, str]] = []

    def _prof(key: str) -> Optional[Dict[str, Any]]:
        return _field_profile(profile, semantics, key)

    # Metric must have numeric values (skip pure count aggregations).
    if getattr(spec, "metric", None) and spec.aggregation not in ("count", "count_distinct"):
        p = _prof(spec.metric)
        if p is None:
            errors.append({"field": spec.metric, "reason": "field_missing",
                           "detail": f"metric {spec.metric!r} not present in the prepared dataset"})
        elif not p["metric_available"]:
            errors.append({"field": spec.metric, "reason": "metric_no_numeric_values",
                           "detail": f"metric {spec.metric!r} ({p['canonical_field']}) has no "
                                     f"numeric values after preparation (non-null {p['non_null']}, "
                                     f"numeric {p['numeric_parse']})"})

    # Grouping dimensions must have non-blank values.
    chart_type = getattr(spec, "chart_type", "none")
    grouping_keys: List[str] = []
    if chart_type in ("bar", "line"):
        grouping_keys = [k for k in (getattr(spec, "dimension", None), getattr(spec, "x", None)) if k][:1]
    elif chart_type in ("heatmap", "treemap"):
        grouping_keys = [k for k in ((getattr(spec, "dimensions", None) or [])
                                     + (getattr(spec, "hierarchy", None) or [])
                                     + [getattr(spec, "dimension", None)]) if k]
    elif getattr(spec, "intent", "") == "table" and getattr(spec, "dimension", None):
        grouping_keys = [spec.dimension]
    for key in grouping_keys:
        p = _prof(key)
        if p is None:
            errors.append({"field": key, "reason": "field_missing",
                           "detail": f"dimension {key!r} not present in the prepared dataset"})
        elif not p["dimension_available"]:
            errors.append({"field": key, "reason": "dimension_no_values",
                           "detail": f"dimension {key!r} ({p['canonical_field']}) has no non-blank "
                                     f"values after preparation (e.g. LTV/bucket not derivable)"})

    # Loan-level x/y/size must each have usable numeric values.
    if chart_type in ("scatter", "bubble"):
        slots = ["x", "y"] + (["size"] if chart_type == "bubble" else [])
        for slot in slots:
            key = getattr(spec, slot, None)
            if not key:
                continue
            p = _prof(key)
            if p is None:
                errors.append({"field": key, "reason": "field_missing",
                               "detail": f"{slot} field {key!r} not present in the prepared dataset"})
            elif p["numeric_parse"] == 0:
                errors.append({"field": key, "reason": "loan_level_no_usable_rows",
                               "detail": f"{slot} field {key!r} ({p['canonical_field']}) has no usable "
                                         f"numeric rows after preparation"})

    # Filter fields must exist and carry values.
    for key in (getattr(spec, "filters", None) or {}):
        p = _prof(key)
        if p is None:
            errors.append({"field": key, "reason": "filter_field_missing",
                           "detail": f"filter field {key!r} not present in the prepared dataset"})
        elif p["non_null"] == 0:
            errors.append({"field": key, "reason": "filter_field_no_values",
                           "detail": f"filter field {key!r} ({p['canonical_field']}) has no values"})

    return errors
