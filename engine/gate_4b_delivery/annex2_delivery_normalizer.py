#!/usr/bin/env python3
"""
annex2_delivery_normalizer.py

Gate 4b delivery normalization for ESMA Annex 2 projected outputs.

Contract:
- input:  *_ESMA_Annex2_projected.csv
- output: *_ESMA_Annex2_delivery_ready.csv
- report: *_ESMA_Annex2_delivery_report.json
- issues: *_ESMA_Annex2_delivery_issues.csv

Design:
- canonical truth remains untouched
- projected CSV is normalized into schema-ready delivery values
- preflight is hard-gate: unresolved errors fail fast with non-zero exit
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

LEI_PATTERN = re.compile(r"^[A-Z0-9]{18}[0-9]{2}$")
ND_PATTERN = re.compile(r"^ND[1-5]$")


def issue_category(issue_type: str) -> str:
    t = str(issue_type).strip().lower()
    if t == "pattern":
        return "pattern / identifier"
    if t == "enum":
        return "enum / code mapping"
    if t == "nd_not_allowed":
        return "ND restriction"
    if t == "precision":
        return "numeric precision"
    if t in {"mandatory_missing", "missing_field"}:
        return "missing mandatory delivery value"
    if t == "choice_branch":
        return "XML choice-branch issues"
    return "other"


@dataclass
class Issue:
    severity: str
    issue_type: str
    field: str
    row_index: int
    message: str
    input_value: str = ""
    output_value: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "delivery_category": issue_category(self.issue_type),
            "issue_type": self.issue_type,
            "field": self.field,
            "row_index": self.row_index,
            "message": self.message,
            "input_value": self.input_value,
            "output_value": self.output_value,
        }


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    s = str(value).strip()
    if s.lower() == "nan":
        return ""
    return s


def normalize_boolean(value: str) -> Optional[str]:
    v = to_str(value).lower()
    if v in {"true", "t", "yes", "y", "1"}:
        return "true"
    if v in {"false", "f", "no", "n", "0"}:
        return "false"
    return None


def validate_lei(value: str) -> bool:
    return bool(LEI_PATTERN.fullmatch(to_str(value).upper()))


def apply_precision(value: str, total_digits: Optional[int], fraction_digits: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
    raw = to_str(value)
    if raw == "":
        return "", None

    try:
        dec = Decimal(raw)
    except InvalidOperation:
        return None, f"Value '{raw}' is not numeric"

    if fraction_digits is not None:
        quant = Decimal("1").scaleb(-fraction_digits)
        dec = dec.quantize(quant, rounding=ROUND_HALF_UP)

    rendered = format(dec, "f")
    if "." in rendered:
        int_part, frac_part = rendered.split(".", 1)
    else:
        int_part, frac_part = rendered, ""

    int_digits = len(int_part.lstrip("-").replace("+", ""))
    frac_digits_count = len(frac_part.rstrip("0"))

    if total_digits is not None and (int_digits + frac_digits_count) > int(total_digits):
        return None, (
            f"Value '{rendered}' exceeds totalDigits={total_digits} "
            f"({int_digits + frac_digits_count} digits)"
        )

    if fraction_digits is not None and frac_digits_count > int(fraction_digits):
        return None, f"Value '{rendered}' exceeds fractionDigits={fraction_digits}"

    return rendered, None


def generate_securitisation_id(lei: str, year: str, seq: int, seq_width: int = 2) -> Optional[str]:
    lei_norm = to_str(lei).upper()
    year_norm = to_str(year)
    if not validate_lei(lei_norm):
        return None
    if not re.fullmatch(r"\d{4}", year_norm):
        return None
    return f"{lei_norm}N{year_norm}{seq:0{seq_width}d}"


def derive_value(
    df: pd.DataFrame,
    row_idx: int,
    derive_rule: Dict[str, Any],
) -> str:
    dtype = str(derive_rule.get("type", "")).strip().lower()
    if dtype == "first_non_blank_from_fields":
        for field in derive_rule.get("fields") or []:
            if field in df.columns:
                candidate = to_str(df.at[row_idx, field])
                if candidate != "":
                    return candidate
        return ""
    if dtype == "months_between_dates":
        start_field = str(derive_rule.get("start_field", "")).strip()
        end_field = str(derive_rule.get("end_field", "")).strip()
        if not start_field or not end_field:
            return ""
        if start_field not in df.columns or end_field not in df.columns:
            return ""
        start_raw = to_str(df.at[row_idx, start_field])
        end_raw = to_str(df.at[row_idx, end_field])
        try:
            start_dt = datetime.strptime(start_raw, "%Y-%m-%d")
            end_dt = datetime.strptime(end_raw, "%Y-%m-%d")
        except Exception:
            return ""
        months = max((end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month), 0)
        return str(months)
    return ""


def _build_outputs(input_csv: Path, output_dir: Path) -> Dict[str, Path]:
    stem = input_csv.stem
    if stem.endswith("_projected"):
        base = stem[: -len("_projected")]
    else:
        base = stem
    return {
        "delivery_ready": output_dir / f"{base}_delivery_ready.csv",
        "report": output_dir / f"{base}_delivery_report.json",
        "issues": output_dir / f"{base}_delivery_issues.csv",
    }


def _normalize_field(
    df: pd.DataFrame,
    out_df: pd.DataFrame,
    field: str,
    rule: Dict[str, Any],
    row_idx: int,
    seq_counter: Dict[str, int],
    default_year: str,
) -> Optional[Issue]:
    mandatory = bool(rule.get("mandatory", False))
    nd_allowed = [str(x).upper() for x in (rule.get("nd_allowed") or [])]
    enforce_presence = bool(rule.get("enforce_presence", mandatory))

    if field not in out_df.columns:
        if enforce_presence:
            return Issue("error", "missing_field", field, row_idx, "Field not present in projected CSV")
        return None

    raw = to_str(df.at[row_idx, field])
    current = raw

    derive_rule = rule.get("derive") if isinstance(rule.get("derive"), dict) else None
    if current == "" and derive_rule:
        current = derive_value(df, row_idx, derive_rule)

    if current == "" and rule.get("default_allowed") and "default_value" in rule:
        current = to_str(rule.get("default_value"))

    generator = rule.get("generator") if isinstance(rule.get("generator"), dict) else None
    if current == "" and generator and generator.get("type") == "securitisation_id":
        lei_field = generator.get("lei_field", "RREL1")
        year_field = generator.get("year_field", "reporting_year")
        year = to_str(df.at[row_idx, year_field]) if year_field in df.columns else default_year
        lei = to_str(df.at[row_idx, lei_field]) if lei_field in df.columns else ""
        seq_key = f"{lei}:{year}"
        seq_counter[seq_key] += 1
        seq = seq_counter[seq_key]
        generated = generate_securitisation_id(lei, year, seq, int(generator.get("sequence_width", 2)))
        if generated is None:
            return Issue(
                "error",
                "pattern",
                field,
                row_idx,
                "Unable to generate ScrtstnIdr from LEI/year",
                input_value=raw,
            )
        current = generated

    if current == "":
        if mandatory:
            return Issue("error", "mandatory_missing", field, row_idx, "Mandatory delivery field missing", raw, current)
        out_df.at[row_idx, field] = current
        return None

    upper = current.upper()
    if ND_PATTERN.fullmatch(upper):
        if upper not in nd_allowed:
            return Issue(
                "error",
                "nd_not_allowed",
                field,
                row_idx,
                f"ND value '{upper}' not allowed for field",
                raw,
                current,
            )
        out_df.at[row_idx, field] = upper
        return None

    transforms = rule.get("transform") if isinstance(rule.get("transform"), dict) else {}

    if transforms.get("boolean") == "xsd_lowercase_true_false":
        b = normalize_boolean(current)
        if b is None:
            return Issue("error", "boolean", field, row_idx, "Boolean must be true/false", raw, current)
        current = b

    for table_name in ("enum_map", "geography_map"):
        mapping = transforms.get(table_name)
        if isinstance(mapping, dict):
            direct = mapping.get(current)
            if direct is None:
                lower_map = {str(k).lower(): str(v) for k, v in mapping.items()}
                direct = lower_map.get(current.lower())
            if direct is None:
                return Issue(
                    "error",
                    "enum",
                    field,
                    row_idx,
                    f"Value '{current}' not found in {table_name}",
                    raw,
                    current,
                )
            current = str(direct)

    validators = rule.get("validators") if isinstance(rule.get("validators"), dict) else {}
    if validators.get("lei") and not validate_lei(current):
        return Issue("error", "pattern", field, row_idx, "Invalid LEI format", raw, current)

    pattern = validators.get("regex")
    if pattern and not re.fullmatch(str(pattern), current):
        return Issue("error", "pattern", field, row_idx, f"Value '{current}' does not match regex", raw, current)

    precision = rule.get("precision") if isinstance(rule.get("precision"), dict) else {}
    if precision:
        cur_num, err = apply_precision(current, precision.get("total_digits"), precision.get("fraction_digits"))
        if err:
            return Issue("error", "precision", field, row_idx, err, raw, current)
        current = cur_num or ""

    out_df.at[row_idx, field] = current
    return None


def normalize_delivery(df: pd.DataFrame, rules: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Issue], Dict[str, Any]]:
    fields_cfg = rules.get("field_rules") if isinstance(rules.get("field_rules"), dict) else {}
    default_year = str((rules.get("defaults") or {}).get("reporting_year", "1900"))

    out_df = df.copy()
    issues: List[Issue] = []
    seq_counter: Dict[str, int] = defaultdict(int)

    for row_idx in range(len(df)):
        for field, rule in fields_cfg.items():
            if not isinstance(rule, dict):
                continue
            issue = _normalize_field(df, out_df, field, rule, row_idx, seq_counter, default_year)
            if issue:
                issues.append(issue)

    counts = Counter([i.issue_type for i in issues])
    category_counts = Counter([issue_category(i.issue_type) for i in issues])
    errors = [i for i in issues if i.severity == "error"]

    summary = {
        "rows_in": int(len(df)),
        "rows_out": int(len(out_df)),
        "issues_total": int(len(issues)),
        "errors_total": int(len(errors)),
        "issue_breakdown": dict(counts),
        "issue_category_breakdown": dict(category_counts),
        "preflight": {
            "status": "PASS" if not errors else "FAIL",
            "blocking_errors": int(len(errors)),
        },
    }
    return out_df, issues, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Annex2 delivery normalizer (Gate 4b)")
    ap.add_argument("--input", required=True, help="Projected Annex2 CSV")
    ap.add_argument("--rules", required=True, help="annex2_delivery_rules.yaml")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    args = ap.parse_args()

    input_csv = Path(args.input)
    rules_path = Path(args.rules)
    output_dir = Path(args.output_dir)

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = _build_outputs(input_csv, output_dir)

    rules = load_yaml(rules_path)
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    logging.info("[Gate 4b] Delivery normalization started: %s", input_csv)
    out_df, issues, summary = normalize_delivery(df, rules)

    out_df.to_csv(outputs["delivery_ready"], index=False)
    pd.DataFrame([i.as_dict() for i in issues]).to_csv(outputs["issues"], index=False)

    report = {
        "input": str(input_csv),
        "rules": str(rules_path),
        "outputs": {k: str(v) for k, v in outputs.items()},
        **summary,
    }
    outputs["report"].write_text(json.dumps(report, indent=2), encoding="utf-8")

    logging.info("[Gate 4b] Delivery-ready CSV............. %s", outputs["delivery_ready"].name)
    logging.info("[Gate 4b] Delivery issues................. %s", outputs["issues"].name)
    logging.info("[Gate 4b] Delivery preflight.............. %s", summary["preflight"]["status"])

    if summary["preflight"]["status"] != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
