#!/usr/bin/env python3
"""
mi_query_validator.py

Validate an :class:`MIQuerySpec` against:
    1. the curated MI semantic registry (mi_semantics_field_registry.yaml), and
    2. (optionally) the set of columns actually available in a dataset.

The validator is pure / side-effect free.  It NEVER touches data values — at
most it is handed the *set of column names* of a dataset.

CLI:
    python -m mi_agent.mi_query_validator \
        --semantics mi_agent/mi_semantics_field_registry.yaml \
        --spec path/to/spec.json \
        --data optional/path/to/canonical.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from .mi_query_spec import (
    AGGREGATIONS,
    CHART_TYPES,
    INTENTS,
    OUTPUT_FORMATS,
    MIQuerySpec,
)

NUMERIC_FORMATS = {"currency", "percent", "integer", "decimal"}
GROUPED_CHART_TYPES = {"bar", "treemap"}


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass
class ValidationResult:
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    resolved_fields: Dict[str, dict] = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "resolved_fields": dict(self.resolved_fields),
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_mi_semantics(path) -> dict:
    """Load the MI semantic registry YAML into a dict with a 'fields' mapping."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "fields" not in data or not isinstance(data["fields"], dict):
        raise ValueError(f"MI semantics file missing 'fields' mapping: {path}")
    return data


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _entry(semantics: dict, key: str) -> Optional[dict]:
    return semantics.get("fields", {}).get(key)


def _is_numeric_entry(entry: dict) -> bool:
    return entry.get("format") in NUMERIC_FORMATS or entry.get("role") == "metric"


def _is_dimension_entry(entry: dict) -> bool:
    return entry.get("role") in ("dimension", "date", "flag")


def _chart_roles(entry: dict) -> Set[str]:
    return set(entry.get("allowed_chart_roles") or [])


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_mi_query(
    spec: MIQuerySpec,
    semantics: dict,
    available_columns: Optional[Set[str]] = None,
) -> ValidationResult:
    result = ValidationResult()
    fields = semantics.get("fields", {})

    # --- 0. enum sanity (check 9) -----------------------------------------
    if spec.intent not in INTENTS:
        result.error(f"Unknown intent: {spec.intent!r} (allowed: {sorted(INTENTS)})")
    if spec.chart_type not in CHART_TYPES:
        result.error(f"Unknown chart_type: {spec.chart_type!r} (allowed: {sorted(CHART_TYPES)})")
    if spec.aggregation not in AGGREGATIONS:
        result.error(f"Unknown aggregation: {spec.aggregation!r}")
    if spec.output_format not in OUTPUT_FORMATS:
        result.error(f"Unknown output_format: {spec.output_format!r}")

    # --- 1. referenced semantic fields exist (check 1) --------------------
    referenced = spec.referenced_fields()
    missing = [f for f in referenced if f not in fields]
    for f in missing:
        result.error(f"Unknown semantic field: {f!r} (not in MI semantic registry)")

    # Build resolved map for fields that DO exist.
    for key in referenced:
        entry = _entry(semantics, key)
        if entry is not None:
            result.resolved_fields[key] = {
                "canonical_field": entry.get("canonical_field", key),
                "role": entry.get("role"),
                "format": entry.get("format"),
            }

    # --- 2. canonical columns exist in dataset (check 2) ------------------
    if available_columns is not None:
        for key, meta in result.resolved_fields.items():
            canonical = meta["canonical_field"]
            if canonical not in available_columns:
                result.error(
                    f"Canonical column {canonical!r} (for semantic field {key!r}) "
                    f"not present in dataset columns"
                )

    # If core references are broken there's no point in deeper structural checks.
    chart_intent = spec.intent == "chart" or (
        spec.chart_type not in (None, "none")
    )

    # --- 3 & 4. chartability + chart-role compatibility -------------------
    # The allowed chart-role tokens for a slot depend on the chart type:
    # scatter/bubble axes are numeric measures, whereas bar/line x is a
    # dimension/date.
    if chart_intent:
        if spec.chart_type in ("scatter", "bubble"):
            x_roles = {"x", "y", "size", "color"}
            y_roles = {"y", "size", "color"}
        else:
            x_roles = {"x", "group", "cohort"}
            y_roles = {"y"}
        _check_slot(result, spec, semantics, "x", x_roles)
        _check_slot(result, spec, semantics, "y", y_roles)
        _check_slot(result, spec, semantics, "size", {"size"})
        _check_slot(result, spec, semantics, "color", {"color", "group"})
        _check_slot(result, spec, semantics, "dimension", {"x", "group", "filter"})
        for dim in (spec.dimensions or []):
            _check_named_slot(result, semantics, dim, {"x", "group", "filter", "color"})
        for dim in (spec.hierarchy or []):
            _check_named_slot(result, semantics, dim, {"x", "group", "filter", "color"})

    # --- 5. aggregation allowed for the metric (check 5) ------------------
    if spec.metric and spec.metric in fields:
        metric_entry = fields[spec.metric]
        allowed = set(metric_entry.get("allowed_aggregations") or [])
        # count / count_distinct are universally permissible counters.
        if spec.aggregation not in allowed and spec.aggregation not in (
            "count", "count_distinct", "loan_level",
        ):
            result.error(
                f"Aggregation {spec.aggregation!r} not allowed for metric "
                f"{spec.metric!r} (allowed: {sorted(allowed)})"
            )

    # --- 7. weighted_avg requires a weight field --------------------------
    if spec.aggregation == "weighted_avg":
        has_weight = bool(spec.weight_field)
        if not has_weight and spec.metric and spec.metric in fields:
            has_weight = bool(fields[spec.metric].get("weight_field"))
        if not has_weight:
            result.error(
                "weighted_avg requires a weight_field (set spec.weight_field or "
                "define weight_field on the metric in the semantic registry)"
            )

    # --- 8. top_n only for grouped outputs --------------------------------
    if spec.top_n is not None:
        grouped = (
            spec.chart_type in GROUPED_CHART_TYPES
            or spec.intent == "table"
            or spec.output_format in ("table", "chart_and_table")
        )
        if not grouped:
            result.error(
                "top_n is only valid for grouped outputs (bar / table / treemap)"
            )

    # --- 6 & 10. chart-type structural requirements -----------------------
    _check_chart_structure(result, spec, semantics)

    return result


def recover_chart_spec(spec: MIQuerySpec, semantics: dict,
                       available_columns: Optional[Set[str]] = None,
                       ) -> Optional["MIQuerySpec"]:
    """Use the validation rules as a RECOVERY/control layer.

    When a spec fails ONLY because the chosen chart type is wrong for the plan, but
    a safe alternative plan validates, return the corrected spec; otherwise None.
    Currently recovers two cases (the rules in the task):

      * KPI: a metric-only query (a metric but no grouping dimension) that was
        proposed as a bar/line/treemap/heatmap -> a KPI/summary (no chart).
      * Table: a grouped plan (a real dimension + metric) whose chart type is
        structurally invalid -> a table.

    It never invents fields and never relaxes the "field must exist / column must
    be present" checks — a genuinely missing dimension still fails cleanly.
    """
    import dataclasses as _dc

    fields = semantics.get("fields", {})

    def _dimension(key: Optional[str]) -> bool:
        return bool(key) and key in fields and _is_dimension_entry(fields[key])

    grouping = [k for k in (spec.dimension, spec.x) if k] + \
        list(spec.dimensions or []) + list(spec.hierarchy or [])
    has_dimension = any(_dimension(k) for k in grouping)
    metric_only = (
        spec.chart_type in ("bar", "line", "treemap", "heatmap")
        and (spec.metric or spec.aggregation in ("count", "count_distinct"))
        and not has_dimension
    )
    if metric_only:
        return _dc.replace(spec, intent="summary", chart_type="none",
                           output_format="table",
                           explanation=(spec.explanation or "") +
                           " [auto-corrected: metric-only query -> KPI/table]")

    # A valid grouped plan whose chart type is structurally wrong -> a table.
    if spec.intent == "chart" and has_dimension and (spec.metric or
                                                     spec.aggregation in ("count", "count_distinct")):
        candidate = _dc.replace(spec, intent="table", chart_type="none",
                                output_format="table")
        if validate_mi_query(candidate, semantics, available_columns).ok:
            return candidate
    return None


def _check_slot(result: ValidationResult, spec: MIQuerySpec, semantics: dict,
                slot: str, allowed_roles: Set[str]) -> None:
    key = getattr(spec, slot)
    if not key:
        return
    _check_named_slot(result, semantics, key, allowed_roles, slot=slot)


def _check_named_slot(result: ValidationResult, semantics: dict, key: str,
                      allowed_roles: Set[str], slot: Optional[str] = None) -> None:
    entry = _entry(semantics, key)
    if entry is None:
        return  # already reported as missing
    label = f"{slot}={key!r}" if slot else f"{key!r}"
    if not entry.get("chartable", False):
        result.error(f"Field {label} is not chartable but is used in a chart role")
    roles = _chart_roles(entry)
    if roles and not (roles & allowed_roles):
        result.error(
            f"Field {label} (chart roles {sorted(roles)}) cannot be used as "
            f"{slot or 'dimension'} (needs one of {sorted(allowed_roles)})"
        )


def _check_chart_structure(result: ValidationResult, spec: MIQuerySpec,
                            semantics: dict) -> None:
    ct = spec.chart_type
    fields = semantics.get("fields", {})

    def numeric(key: Optional[str]) -> bool:
        return bool(key) and key in fields and _is_numeric_entry(fields[key])

    def dimension(key: Optional[str]) -> bool:
        return bool(key) and key in fields and _is_dimension_entry(fields[key])

    if ct == "none":
        if spec.intent == "chart":
            result.error("chart_type 'none' is not valid for intent 'chart'")
        return  # check 10: summary/table with no chart is fine

    if ct == "bar":
        if not (spec.dimension or spec.x):
            result.error("bar chart requires a dimension (or x)")
        if not spec.metric and spec.aggregation not in ("count", "count_distinct"):
            result.error("bar chart requires a metric or a count aggregation")

    elif ct == "line":
        x_entry = fields.get(spec.x) if spec.x else None
        cohort_ok = bool(x_entry) and (
            x_entry.get("role") == "date" or "cohort" in _chart_roles(x_entry)
        )
        if not cohort_ok:
            result.error("line chart requires x to be a date / cohort / trend dimension")
        if not spec.metric:
            result.error("line chart requires a metric")

    elif ct == "scatter":
        if not numeric(spec.x) or not numeric(spec.y):
            result.error("scatter chart requires numeric x and numeric y")

    elif ct == "bubble":
        if not numeric(spec.x) or not numeric(spec.y) or not numeric(spec.size):
            result.error("bubble chart requires numeric x, numeric y and numeric size")

    elif ct == "heatmap":
        dims = [k for k in (spec.x, spec.y, spec.dimension) if k] + list(spec.dimensions or [])
        dim_count = sum(1 for k in dims if dimension(k))
        if dim_count < 2:
            result.error("heatmap requires at least two dimensions")
        has_metric = numeric(spec.metric) or numeric(spec.color) or numeric(spec.size)
        if not has_metric and spec.aggregation not in ("count", "count_distinct"):
            result.error("heatmap requires a metric (intensity) or a count aggregation")

    elif ct == "treemap":
        hier = list(spec.hierarchy or []) + list(spec.dimensions or [])
        if spec.dimension:
            hier.append(spec.dimension)
        if not any(dimension(k) for k in hier):
            result.error("treemap requires at least one hierarchy/dimension field")
        if not (numeric(spec.metric) or numeric(spec.size)) and \
                spec.aggregation not in ("count", "count_distinct"):
            result.error("treemap requires a metric (size) or a count aggregation")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _read_csv_columns(path: Path) -> Set[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
    return {h.strip() for h in header if h.strip()}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an MIQuerySpec.")
    parser.add_argument("--semantics", type=Path, required=True,
                        help="Path to mi_semantics_field_registry.yaml")
    parser.add_argument("--spec", type=Path, required=True,
                        help="Path to a JSON MIQuerySpec")
    parser.add_argument("--data", type=Path, default=None,
                        help="Optional path to a canonical CSV (header used for column check)")
    args = parser.parse_args(argv)

    semantics = load_mi_semantics(args.semantics)
    spec = MIQuerySpec.from_json(Path(args.spec).read_text(encoding="utf-8"))

    columns: Optional[Set[str]] = None
    if args.data is not None:
        columns = _read_csv_columns(args.data)

    result = validate_mi_query(spec, semantics, available_columns=columns)

    print("OK" if result.ok else "FAILED")
    if result.errors:
        print("\nErrors:")
        for e in result.errors:
            print(f"  - {e}")
    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    canonical = sorted({m["canonical_field"] for m in result.resolved_fields.values()})
    if canonical:
        print("\nReferenced canonical fields:")
        for c in canonical:
            print(f"  - {c}")

    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
