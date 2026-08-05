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


# --------------------------------------------------------------------------- #
# Phase 1 delivery instrumentation — OBSERVE ONLY.
#
# Separates the two kinds of ND this stage sees, which the report previously
# conflated into a single "delivery ready" outcome:
#
#   * present_in_input  — the projected CSV already carried an ND code. That is
#     upstream truth (registry/projector), not a delivery decision.
#   * applied_by_rules  — this normaliser substituted an ND because a DECLARED
#     rule in annex2_delivery_rules.yaml said to (``default_allowed`` +
#     ``default_value``). Governed and auditable, but still a value the client
#     did not supply.
#
# Coercions are recorded whenever a declared transform CHANGES a value, with
# the rule that caused it. Counts are exact; individual records are capped.
#
# Nothing here alters a value: removing every ``record_*`` call would leave the
# delivery-ready CSV byte-identical.
# --------------------------------------------------------------------------- #
INSTRUMENTATION_SCHEMA_VERSION = 1

#: Hard cap on individually recorded coercions. Counts remain exact above it.
COERCION_RECORD_CAP = 5000


class _DeliveryInstrumentation:
    """ND provenance and coercion records for one normalisation run."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.in_input_by_code: Dict[str, int] = defaultdict(int)
        self.in_input_by_field: Dict[str, int] = defaultdict(int)
        self.by_rule_by_code: Dict[str, int] = defaultdict(int)
        self.by_rule_by_field: Dict[str, int] = defaultdict(int)
        self.coercions: List[Dict[str, Any]] = []
        self.coercion_count = 0
        self.coercions_truncated = False

    def scan_input_nd(self, df: "pd.DataFrame") -> None:
        """Count every ND code already present in the projected input."""
        for column in df.columns:
            values = df[column].astype(str).str.strip().str.upper()
            matched = values[values.str.fullmatch(r"ND[1-5]", na=False)]
            if matched.empty:
                continue
            self.in_input_by_field[str(column)] += int(len(matched))
            for code, count in matched.value_counts().items():
                self.in_input_by_code[str(code)] += int(count)

    def record_nd_from_rule(self, *, nd_code: str, field: str) -> None:
        self.by_rule_by_code[nd_code] += 1
        self.by_rule_by_field[field] += 1

    def record_coercion(self, *, field_code: str, original_value: str,
                        resulting_value: str, reason: str,
                        row_identifier: Optional[str] = None) -> None:
        self.coercion_count += 1
        if len(self.coercions) < COERCION_RECORD_CAP:
            self.coercions.append({
                "field_code": field_code,
                "row_identifier": row_identifier,
                "original_value": original_value,
                "resulting_value": resulting_value,
                "reason": reason,
            })
        else:
            self.coercions_truncated = True

    def to_dict(self) -> Dict[str, Any]:
        """The report. Zero is stated explicitly — an absent entry is not zero."""
        return {
            "schema_version": INSTRUMENTATION_SCHEMA_VERSION,
            "stage": "gate_4b_delivery_normalisation",
            "nd_present_in_input": {
                "total": int(sum(self.in_input_by_code.values())),
                "by_code": {k: int(v) for k, v in sorted(self.in_input_by_code.items())},
                "by_field": {k: int(v) for k, v in sorted(self.in_input_by_field.items())},
            },
            "nd_applied_by_rules": {
                "total": int(sum(self.by_rule_by_code.values())),
                "by_code": {k: int(v) for k, v in sorted(self.by_rule_by_code.items())},
                "by_field": {k: int(v) for k, v in sorted(self.by_rule_by_field.items())},
            },
            "coercions": {
                "count": int(self.coercion_count),
                "records": list(self.coercions),
                "truncated": bool(self.coercions_truncated),
                "record_cap": COERCION_RECORD_CAP,
            },
        }


_INSTR = _DeliveryInstrumentation()


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
        if current and ND_PATTERN.fullmatch(current.upper()):
            # A DECLARED rule supplied this ND — governed, but still a value
            # the client did not provide.
            _INSTR.record_nd_from_rule(nd_code=current.upper(), field=field)

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
                # enum_map is a strict controlled vocabulary. geography_map is a
                # best-effort LEGACY label->code/year translation: a value that
                # is not a known legacy label is already a valid geography value
                # (e.g. a NUTS3 code "TLG31" or classification year "2021") and
                # passes through unchanged. Shape is enforced downstream (XSD).
                if table_name == "geography_map":
                    continue
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

    # A declared transform CHANGED a supplied value. Recorded so the delivery
    # report can show what the rules did to the client's data, distinct from
    # the builder's own coercion (see xml_builder_annex2._INSTR).
    if raw and current != raw:
        _INSTR.record_coercion(
            field_code=field, original_value=raw, resulting_value=current,
            row_identifier=_row_identifier(df, row_idx),
            reason=_transform_reason(rule))
    out_df.at[row_idx, field] = current
    return None


#: Columns that identify an exposure, best first. RREL3 is the new underlying
#: exposure identifier; RREL2 the original.
_ROW_ID_FIELDS = ("RREL3", "RREL2")


def _row_identifier(df: pd.DataFrame, row_idx: int) -> Optional[str]:
    """The exposure a record belongs to, or ``None`` — never invented."""
    for field in _ROW_ID_FIELDS:
        if field in df.columns:
            value = to_str(df.at[row_idx, field])
            if value:
                return value
    return None


def _transform_reason(rule: Dict[str, Any]) -> str:
    """Which declared rule caused a value to change."""
    transforms = rule.get("transform") if isinstance(rule.get("transform"), dict) else {}
    applied = sorted(str(k) for k in transforms)
    if applied:
        return ("annex2_delivery_rules.yaml field_rules transform: "
                + ", ".join(applied))
    if "precision" in rule:
        return "annex2_delivery_rules.yaml field_rules precision"
    return "annex2_delivery_rules.yaml field_rules normalisation"


def normalize_delivery(df: pd.DataFrame, rules: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Issue], Dict[str, Any]]:
    fields_cfg = rules.get("field_rules") if isinstance(rules.get("field_rules"), dict) else {}
    default_year = str((rules.get("defaults") or {}).get("reporting_year", "1900"))

    _INSTR.reset()
    # ND already present in the projected input, across EVERY column — not just
    # the 68 rule-governed ones. The normaliser passes unlisted columns through
    # verbatim, so counting only rule-governed fields would under-report
    # upstream ND and silently attribute the difference to nothing.
    _INSTR.scan_input_nd(df)
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
        # Phase 1: what this stage did to values, stated explicitly. A run with
        # no coercions records a count of zero rather than omitting the key.
        "delivery_instrumentation": _INSTR.to_dict(),
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
