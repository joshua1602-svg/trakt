#!/usr/bin/env python3
"""
build_mi_semantics_registry.py

Generate the curated MI semantic field registry from the canonical field
registry.

This is an ADDITIVE tool.  It only READS the canonical registry
(config/system/fields_registry.yaml) and WRITES a new, separate file:

    mi_agent/mi_semantics_field_registry.yaml

The generated file does NOT duplicate the canonical registry.  For every
selected field it stores a thin "semantic" record that REFERENCES the
canonical field by name and adds analytics metadata (role, format, allowed
aggregations, chart roles, weighting / bucketing hints).

Selection rules (OR — any one is sufficient):
    1. core_canonical == true
    2. layer in {performance, product}
    3. category == analytics

The generated metadata is heuristic and is intended as a *starting point* for
human review before production use.

Usage:
    python -m mi_agent.build_mi_semantics_registry
    python -m mi_agent.build_mi_semantics_registry \
        --source config/system/fields_registry.yaml \
        --output mi_agent/mi_semantics_field_registry.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "config" / "system" / "fields_registry.yaml"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "mi_semantics_field_registry.yaml"

VERSION = "0.1.0"

# Acronyms that should be upper-cased in display names.
_ACRONYMS = {
    "ltv": "LTV",
    "lei": "LEI",
    "id": "ID",
    "lgd": "LGD",
    "pd": "PD",
    "isin": "ISIN",
    "nuts": "NUTS",
    "abcp": "ABCP",
    "esma": "ESMA",
    "boe": "BoE",
    "erm": "ERM",
    "iban": "IBAN",
    "yn": "Y/N",
}

# --------------------------------------------------------------------------- #
# Defensive accessors (the canonical registry casing/nesting may vary)
# --------------------------------------------------------------------------- #


def _get(d: dict, *keys, default=None):
    """Case-insensitive first-match lookup across candidate keys."""
    if not isinstance(d, dict):
        return default
    lowered = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if k in d:
            return d[k]
        if str(k).lower() in lowered:
            return lowered[str(k).lower()]
    return default


def load_canonical_registry(path: Path) -> Dict[str, dict]:
    """Return the {field_name: meta} mapping from the canonical registry."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    fields = _get(data, "fields", default=None)
    if not isinstance(fields, dict):
        # Some registries may place fields at the top level.
        fields = {
            k: v
            for k, v in data.items()
            if isinstance(v, dict) and ("layer" in v or "category" in v)
        }
    if not fields:
        raise ValueError(f"No 'fields' mapping found in registry: {path}")
    return fields


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

_PERF_PRODUCT_LAYERS = {"performance", "product"}


def selection_criteria(meta: dict) -> List[str]:
    """Return the list of OR-criteria that caused this field to be selected."""
    crit: List[str] = []
    if _get(meta, "core_canonical") is True:
        crit.append("core_canonical")
    layer = str(_get(meta, "layer", default="") or "").lower()
    if layer in _PERF_PRODUCT_LAYERS:
        crit.append(f"layer:{layer}")
    category = str(_get(meta, "category", default="") or "").lower()
    if category == "analytics":
        crit.append("category:analytics")
    return crit


def select_fields(fields: Dict[str, dict]) -> Dict[str, List[str]]:
    """Return {field_name: source_criteria} for every selected field."""
    selected: Dict[str, List[str]] = {}
    for name, meta in fields.items():
        if not isinstance(meta, dict):
            continue
        crit = selection_criteria(meta)
        if crit:
            selected[name] = crit
    return selected


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #


def _tokens(name: str) -> List[str]:
    return [t for t in str(name).lower().split("_") if t]


def display_name(name: str) -> str:
    parts = []
    for tok in _tokens(name):
        parts.append(_ACRONYMS.get(tok, tok.capitalize()))
    return " ".join(parts) if parts else name


def _contains(name: str, needles) -> bool:
    low = name.lower()
    return any(n in low for n in needles)


def infer_format(name: str, reg_format: Optional[str], allowed_values) -> str:
    """Map to one of: currency|percent|integer|decimal|date|string|boolean."""
    nf = str(reg_format).strip().lower() if reg_format else ""
    av = str(allowed_values).strip().lower() if allowed_values else ""

    # 1. Date
    if nf == "date" or "date" in name.lower():
        return "date"
    # 2. Boolean / flags
    if nf in ("y/n", "boolean", "bool") or av in ("yes_no", "y/n"):
        return "boolean"
    # 3. Categorical (registry already says so) -> string
    if nf in ("list", "string", "currency_code", "enum", "category"):
        return "string"

    # Numeric-ish (decimal / integer / unknown).  Use name tokens.
    if _contains(name, ("ltv", "loan_to_value", "percentage", "pct", "margin", "spread")) or \
            "rate" in _tokens(name):
        return "percent"
    if _contains(name, ("balance", "amount", "valuation", "value", "income",
                         "loss", "recovery", "arrears", "price", "proceeds")):
        return "currency"
    if _contains(name, ("count", "number_of", "term_months", "months")) or \
            "age" in _tokens(name):
        return "integer"
    if nf == "integer":
        return "integer"
    if nf == "decimal":
        return "decimal"
    return "string"


_ID_TOKENS = {"id", "identifier", "lei", "code", "reference", "ref"}
_ID_PHRASES = ("account_number", "loan_number", "_identifier")


def _is_identifier(name: str) -> bool:
    toks = set(_tokens(name))
    if toks & _ID_TOKENS:
        return True
    return _contains(name, _ID_PHRASES)


def infer_role(name: str, fmt: str, reg_format: Optional[str]) -> str:
    """metric | dimension | date | identifier | flag | unknown."""
    if _is_identifier(name):
        return "identifier"
    if fmt == "date":
        return "date"
    if fmt == "boolean":
        return "flag"
    if fmt in ("currency", "percent", "integer", "decimal"):
        return "metric"
    # fmt == "string"
    nf = str(reg_format).strip().lower() if reg_format else ""
    if nf in ("list", "string", "currency_code", "enum", "category"):
        return "dimension"
    # No registry format and nothing else matched -> not safely inferable.
    return "unknown"


def _is_count_metric(name: str) -> bool:
    return _contains(name, ("count", "number_of")) and "age" not in _tokens(name)


def infer_aggregations(role: str, fmt: str, name: str) -> Tuple[List[str], str]:
    """Return (allowed_aggregations, default_aggregation)."""
    if role == "identifier":
        return ["count_distinct"], "count_distinct"
    if role == "date":
        return ["count"], "count"
    if role == "flag":
        return ["count", "balance_sum"], "count"
    if role == "dimension":
        return ["count", "balance_sum"], "count"
    if role == "metric":
        if fmt == "currency":
            return ["sum", "avg", "median"], "sum"
        if fmt == "percent":
            return ["avg", "weighted_avg", "distribution"], "avg"
        if fmt == "integer":
            if _is_count_metric(name):
                return ["sum", "count"], "sum"
            # ages / terms / counts of periods
            return ["avg", "median", "distribution"], "avg"
        # decimal (generic numeric)
        return ["sum", "avg", "median"], "avg"
    # unknown
    return [], ""


def infer_chart_roles(role: str) -> Tuple[List[str], Optional[str]]:
    """Return (allowed_chart_roles, default_chart_role)."""
    if role == "identifier":
        return ["filter"], None
    if role == "date":
        return ["x", "filter", "cohort"], "x"
    if role == "flag":
        return ["filter", "color", "group"], "filter"
    if role == "dimension":
        return ["x", "group", "filter", "color"], "x"
    if role == "metric":
        return ["y", "size", "color"], "y"
    return [], None


def infer_chartable(role: str, name: str) -> bool:
    if role in ("metric", "dimension", "flag"):
        return True
    if role == "date":
        # Useful for trend / cohort only for origination / vintage style dates.
        return "origination" in name.lower()
    # identifier / unknown
    return False


def pick_weight_field(selected_names: set) -> Optional[str]:
    """First available balance field, in preference order."""
    for candidate in ("total_balance", "current_outstanding_balance",
                       "current_principal_balance", "original_principal_balance"):
        if candidate in selected_names:
            return candidate
    return None


def infer_weight_field(role: str, fmt: str, weight_target: Optional[str]) -> Optional[str]:
    if role == "metric" and fmt == "percent":
        return weight_target
    return None


def infer_bucket_field(name: str, role: str, fmt: str) -> Optional[str]:
    low = name.lower()
    if fmt == "date" and "origination" in low:
        return "vintage_year"
    if "ltv" in low or "loan_to_value" in low:
        return "original_ltv_bucket" if "original" in low else "ltv_bucket"
    if "age" in _tokens(name):
        return "age_bucket"
    if "balance" in low and role == "metric":
        return "ticket_bucket"
    return None


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build_entry(name: str, meta: dict, source_criteria: List[str],
                weight_target: Optional[str]) -> dict:
    reg_format = _get(meta, "format")
    allowed_values = _get(meta, "allowed_values")

    fmt = infer_format(name, reg_format, allowed_values)
    role = infer_role(name, fmt, reg_format)
    aggs, default_agg = infer_aggregations(role, fmt, name)
    chart_roles, default_chart_role = infer_chart_roles(role)
    chartable = infer_chartable(role, name)
    weight_field = infer_weight_field(role, fmt, weight_target)
    bucket_field = infer_bucket_field(name, role, fmt)

    notes = ""
    if role == "unknown":
        notes = "requires manual analytics classification"

    return {
        "canonical_field": name,
        "display_name": display_name(name),
        "description": "",
        "source_criteria": source_criteria,
        "role": role,
        "format": fmt,
        "chartable": chartable,
        "allowed_aggregations": aggs,
        "default_aggregation": default_agg,
        "allowed_chart_roles": chart_roles,
        "default_chart_role": default_chart_role,
        "weight_field": weight_field,
        "bucket_field": bucket_field,
        "notes": notes,
    }


def build_registry(source: Path) -> dict:
    fields = load_canonical_registry(source)
    selected = select_fields(fields)
    selected_names = set(selected.keys())
    weight_target = pick_weight_field(selected_names)

    out_fields: Dict[str, dict] = {}
    for name in sorted(selected):
        out_fields[name] = build_entry(
            name, fields[name], selected[name], weight_target
        )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_registry": str(source),
        "selection_rules": [
            "core_canonical true",
            "layer performance/product",
            "category analytics",
        ],
        "field_count": len(out_fields),
        "version": VERSION,
        "default_weight_field": weight_target,
    }

    return {"metadata": metadata, "fields": out_fields}


def write_registry(registry: dict, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# AUTO-GENERATED by mi_agent/build_mi_semantics_registry.py\n"
        "# Curated MI semantic layer over the canonical field registry.\n"
        "# Heuristic metadata — REVIEW BY A HUMAN BEFORE PRODUCTION USE.\n"
        "# Do NOT edit by hand without also updating the build script, or your\n"
        "# changes will be overwritten on the next regeneration.\n"
    )
    with output.open("w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(registry, fh, sort_keys=False, allow_unicode=True, width=100)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the MI semantic field registry.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Path to canonical fields_registry.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Path to write the MI semantic registry YAML")
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"ERROR: source registry not found: {args.source}", file=sys.stderr)
        return 2

    registry = build_registry(args.source)
    write_registry(registry, args.output)

    fields = registry["fields"]
    role_counts: Dict[str, int] = {}
    for entry in fields.values():
        role_counts[entry["role"]] = role_counts.get(entry["role"], 0) + 1

    print(f"Wrote {len(fields)} fields -> {args.output}")
    print("Roles inferred:")
    for role, count in sorted(role_counts.items()):
        print(f"  {role:11s} {count}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
