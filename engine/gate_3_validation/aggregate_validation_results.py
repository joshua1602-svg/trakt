#!/usr/bin/env python3
"""
aggregate_validation_results.py

Aggregates row-level violations from:
- validate_canonical.py  (columns: rule_id,severity,field,row,message)
- validate_business_rules.py (columns: rule_id,severity,description,message,row_index)

into a field-level summary with materiality policy.

Drop-in replacement with:
- Robust input schema handling (field vs field_name; row vs row_index)
- Correct use of loaded YAML dicts (no Path/dict confusion)
- Optional rule_registry.yaml support for rule->field expansion (best effort)

Level-1 integration fix:
- Emits BOTH `canonical_field` and `field_name` columns (backward compatible)
- Dashboard JSON uses `canonical_field`
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import json

import pandas as pd

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

OUT_DIR = Path("out_validation")
OUT_DIR.mkdir(exist_ok=True)

# ----------------------------
# YAML loaders
# ----------------------------

def load_yaml_required(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_yaml_optional(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# ----------------------------
# Normalise violations schemas
# ----------------------------

def normalise_canonical_violations(df: pd.DataFrame) -> pd.DataFrame:
    """
    validate_canonical schema: rule_id,severity,field,row,message
    Normalise to: field_name, issue_type, severity, row_index, message
    """
    if df.empty:
        return df

    out = df.copy()

    if "field" in out.columns and "field_name" not in out.columns:
        out = out.rename(columns={"field": "field_name"})
    if "row" in out.columns and "row_index" not in out.columns:
        out = out.rename(columns={"row": "row_index"})

    # canonical validator uses rule_id as type; keep it as issue_type
    if "issue_type" not in out.columns:
        out["issue_type"] = out["rule_id"] if "rule_id" in out.columns else "CANONICAL"

    # ensure expected columns exist
    for c in ["field_name", "issue_type", "severity", "row_index", "message"]:
        if c not in out.columns:
            out[c] = pd.NA

    return out[["field_name", "issue_type", "severity", "row_index", "message"]]


def normalise_business_violations(df: pd.DataFrame) -> pd.DataFrame:
    """
    validate_business_rules schema: rule_id,severity,description,message,row_index
    Normalise to: field_name, issue_type, severity, row_index, message
    field_name requires rule_registry expansion; default to PORTFOLIO.
    """
    if df.empty:
        return df

    out = df.copy()

    if "row" in out.columns and "row_index" not in out.columns:
        out = out.rename(columns={"row": "row_index"})

    out["issue_type"] = out["rule_id"] if "rule_id" in out.columns else "BUSINESS_RULE"

    for c in ["severity", "row_index", "message"]:
        if c not in out.columns:
            out[c] = pd.NA

    # placeholder field until expansion
    out["field_name"] = "PORTFOLIO"

    return out[["field_name", "issue_type", "severity", "row_index", "message"]]


def expand_rules_to_fields(business_df: pd.DataFrame, rule_registry: Dict[str, Any]) -> pd.DataFrame:
    """
    If rule_registry defines rules.<rule_id>.fields, explode business rule violations
    so each rule contributes to the relevant fields.
    """
    if business_df.empty or not rule_registry:
        return business_df

    rules = (rule_registry.get("rules") or {})
    rows = []

    for _, r in business_df.iterrows():
        rid = r.get("issue_type") or r.get("rule_id")
        meta = rules.get(rid, {}) if rid else {}
        fields = meta.get("fields") or []
        if not fields:
            rows.append(r.to_dict())
        else:
            for f in fields:
                d = r.to_dict()
                d["field_name"] = f
                rows.append(d)

    return pd.DataFrame(rows) if rows else business_df


# ----------------------------
# Policy logic
# ----------------------------

def classify_issue(issue_type: str, issue_policy: Dict[str, Any]) -> str:
    rule_classifications = issue_policy.get("rule_classifications", {}) or {}
    if issue_type in rule_classifications:
        return rule_classifications[issue_type]

    fallback = {
        "CORE001": "mandatory_null",
        "CORE002": "mandatory_null",
        "CORE003": "nd_not_permitted",
        "FMT_DATE": "value_format_error",
        "FMT_CCY_CODE": "value_format_error",
        "FMT_DEC": "value_format_error",
        "FMT_INT": "value_format_error",
        "FMT_BOOL": "value_format_error",
        "ENUM_INVALID": "enum_violation",
        "REG001": "schema_missing",
        "REG002": "mandatory_null",
    }
    return fallback.get(issue_type, "business_logic_violation")


def determine_materiality(error_rate_pct: float, severity: str, classification: str, issue_policy: Dict[str, Any]) -> str:
    thresholds = issue_policy.get("field_aggregation_thresholds", {}) or {}
    high = float((thresholds.get("error_rate_high", {}) or {}).get("threshold", 0.25)) * 100.0
    medium = float((thresholds.get("error_rate_medium", {}) or {}).get("threshold", 0.10)) * 100.0

    classifications = issue_policy.get("classifications", {}) or {}
    default_mat = (classifications.get(classification, {}) or {}).get("default_materiality", "REVIEW")

    sev = str(severity or "").lower()
    if error_rate_pct >= high:
        return "BLOCKING"
    if error_rate_pct >= medium:
        return "REVIEW"
    if sev in ("error",) and default_mat == "BLOCKING":
        return "BLOCKING"
    return default_mat


def allowed_actions(materiality: str, issue_policy: Dict[str, Any]) -> List[str]:
    acts = (issue_policy.get("allowed_actions", {}) or {}).get(materiality, []) or []
    return [a.get("action") for a in acts if isinstance(a, dict) and a.get("action")]


# ----------------------------
# Aggregation
# ----------------------------

def aggregate(df: pd.DataFrame, total_rows: int, issue_policy: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "canonical_field",
            "field_name",
            "issue_type",
            "affected_rows",
            "error_count",
            "error_rate",
            "severity",
            "classification",
            "materiality",
            "suggested_actions",
            "sample_violations",
        ])

    # row_index may contain blanks for column-level issues; treat those as -1 for uniqueness calcs
    tmp = df.copy()
    tmp["row_index"] = pd.to_numeric(tmp["row_index"], errors="coerce").fillna(-1).astype(int)

    out_rows = []
    for (field_name, issue_type), g in tmp.groupby(["field_name", "issue_type"]):
        affected_rows = int(g.loc[g["row_index"] >= 0, "row_index"].nunique())
        error_count = int(len(g))
        error_rate = (affected_rows / total_rows * 100.0) if total_rows else 0.0
        sev = str(g["severity"].iloc[0] if "severity" in g.columns else "error")

        classification = classify_issue(str(issue_type), issue_policy)
        materiality = determine_materiality(error_rate, sev, classification, issue_policy)
        actions = allowed_actions(materiality, issue_policy)

        sample = g.head(5)[["row_index", "message"]].to_dict("records")

        out_rows.append({
            # Level-1 integration: provide canonical_field for lineage/delta tooling
            "canonical_field": field_name,
            # Backward compatibility
            "field_name": field_name,
            "issue_type": issue_type,
            "affected_rows": affected_rows,
            "error_count": error_count,
            "error_rate": round(error_rate, 2),
            "severity": sev,
            "classification": classification,
            "materiality": materiality,
            "suggested_actions": ", ".join([a for a in actions if a]),
            "sample_violations": json.dumps(sample, ensure_ascii=False),
        })

    res = pd.DataFrame(out_rows)

    # Sort: BLOCKING first, then REVIEW, then INFO; within that by error_rate desc
    mat_rank = {"BLOCKING": 0, "REVIEW": 1, "INFO": 2}
    res["_rank"] = res["materiality"].map(mat_rank).fillna(9).astype(int)
    res = res.sort_values(["_rank", "error_rate"], ascending=[True, False]).drop(columns=["_rank"])
    return res


def write_dashboard_json(field_summary: pd.DataFrame, issue_policy: Dict[str, Any], out_path: Path) -> None:
    payload = {
        "summary": {
            "total_fields_affected": int(field_summary["canonical_field"].nunique()) if not field_summary.empty else 0,
            "blocking_count": int((field_summary["materiality"] == "BLOCKING").sum()) if not field_summary.empty else 0,
            "review_count": int((field_summary["materiality"] == "REVIEW").sum()) if not field_summary.empty else 0,
            "info_count": int((field_summary["materiality"] == "INFO").sum()) if not field_summary.empty else 0,
        },
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "policy_version": (issue_policy.get("metadata", {}) or {}).get("version", "unknown"),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Aggregate validation results to field-level summary")
    ap.add_argument("--canonical-violations", default="", help="Path to canonical validation violations CSV")
    ap.add_argument("--business-violations", default="", help="Path to business rules violations CSV")
    ap.add_argument("--input-csv", required=True, help="Path to typed canonical CSV (to get row count)")
    ap.add_argument("--output", default="", help="Output CSV path")
    ap.add_argument("--dashboard-json", default="", help="Output dashboard JSON path (optional)")
    ap.add_argument("--regime", required=True, help="Regime name (metadata only for now)")
    ap.add_argument("--issue-policy", default="issue_policy.yaml", help="Path to issue_policy.yaml (required)")
    ap.add_argument("--rule-registry", default="rule_registry.yaml", help="Path to rule_registry.yaml (optional)")
    args = ap.parse_args()

    issue_policy = load_yaml_required(Path(args.issue_policy))
    rule_registry = load_yaml_optional(Path(args.rule_registry))

    in_path = Path(args.input_csv)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # Keep your existing behaviour (pandas row count). If you want faster later, change here.
    total_rows = int(len(pd.read_csv(in_path, usecols=[0])))
    print(f"[aggregate] Total rows: {total_rows}")

    parts = []
    if args.canonical_violations:
        p = Path(args.canonical_violations)
        if p.exists():
            parts.append(normalise_canonical_violations(pd.read_csv(p)))

    if args.business_violations:
        p = Path(args.business_violations)
        if p.exists():
            b = normalise_business_violations(pd.read_csv(p))
            b = expand_rules_to_fields(b, rule_registry)
            parts.append(b)

    if not parts:
        print("[aggregate] No violations provided/found; writing empty outputs.")
        field_summary = aggregate(pd.DataFrame(), total_rows, issue_policy)
    else:
        combined = pd.concat(parts, ignore_index=True)
        field_summary = aggregate(combined, total_rows, issue_policy)

    out_csv = Path(args.output) if args.output else OUT_DIR / f"{in_path.stem}_field_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    field_summary.to_csv(out_csv, index=False)
    print(f"[aggregate] Wrote: {out_csv}")

    if args.dashboard_json:
        write_dashboard_json(field_summary, issue_policy, Path(args.dashboard_json))
        print(f"[aggregate] Wrote: {args.dashboard_json}")


if __name__ == "__main__":
    main()

