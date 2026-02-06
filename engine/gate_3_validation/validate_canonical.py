#!/usr/bin/env python3
"""
validate_canonical.py

Validates either:
  A) Full Canonical (truth set) produced by semantic_alignment (active schema), or
  B) Regime projection (schema-complete, ND-padded) produced by a downstream projector

Source of truth: field registry YAML (core_canonical flags, formats, regime mappings, applicability overrides).

Key rules:
- Canonical scope:
    - core_canonical fields must be present and populated
    - ND* codes are NOT allowed in canonical truth set *except* where registry applicability permits
      (e.g., equity_release maturity_date allowed ND2)
- Regime scope:
    - ND codes are allowed
    - stricter ND-per-field permissions can be enforced if registry provides allowed_nd_codes under regime_mapping
"""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from collections import defaultdict

ND_PATTERN = re.compile(r"^ND[1-9]\d*$", re.IGNORECASE)
CCY_CODE_PATTERN = re.compile(r"^[A-Z]{3}$")


# -----------------------------
# Registry loading / selection
# -----------------------------

def load_registry(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "fields" not in data or not isinstance(data["fields"], dict):
        raise ValueError(f"Registry YAML missing top-level 'fields' mapping: {path}")
    return data


def select_fields_for_portfolio(registry: Dict[str, Any], portfolio_type: str) -> Dict[str, Dict[str, Any]]:
    pt = (portfolio_type or "").strip().lower()
    out: Dict[str, Dict[str, Any]] = {}
    for fname, meta in registry["fields"].items():
        fpt = str((meta or {}).get("portfolio_type", "")).strip().lower()
        if fpt == "common" or fpt == pt:
            out[fname] = meta or {}
    return out


def core_true_fields(reg_fields: Dict[str, Dict[str, Any]]) -> List[str]:
    return [f for f, m in reg_fields.items() if bool((m or {}).get("core_canonical", False))]


def get_core_required_fields(reg_fields: Dict[str, Dict[str, Any]]) -> List[str]:
    """Return the list of core canonical fields (core_canonical:true)."""
    return [f for f, m in reg_fields.items() if bool((m or {}).get("core_canonical", False))]


def get_applicability(meta: Dict[str, Any], portfolio_type: str) -> Dict[str, Any]:
    """
    Return applicability override for the given portfolio_type (if any).

    Expected YAML shape (example):
      maturity_date:
        applicability:
          equity_release:
            allowed_missing: true
            severity_if_missing: warning
            nd_default: ND2
            reason: "Lifetime mortgage product has no fixed maturity"
    """
    pt = (portfolio_type or "").strip().lower()
    app = (meta or {}).get("applicability") or {}
    for k, v in app.items():
        if str(k).strip().lower() == pt:
            return v or {}
    return {}


def regime_fields(reg_fields: Dict[str, Dict[str, Any]], regime: str) -> List[Tuple[str, Dict[str, Any]]]:
    rk = str(regime).strip()
    out: List[Tuple[str, Dict[str, Any]]] = []
    for f, m in reg_fields.items():
        rm = (m.get("regime_mapping") or {})
        if rk in rm:
            out.append((f, rm[rk] or {}))
    return out


# -----------------------------
# Validation helpers
# -----------------------------

def is_blank(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() == "nan"


def is_nd(x: Any) -> bool:
    """Return True if value looks like an ND code (e.g., ND1, ND2)."""
    if is_blank(x):
        return False
    return bool(ND_PATTERN.match(str(x).strip()))


def nd_number(val: Any) -> Optional[int]:
    """Return ND code number if val looks like 'ND1'..'ND9', else None."""
    if is_blank(val):
        return None
    m = re.search(r"ND\s*([1-9]\d*)", str(val).strip().upper())
    return int(m.group(1)) if m else None


def coerce_decimal(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"nan": ""})
    s = s.str.replace(r"[^\d\-\.,]", "", regex=True)
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def coerce_int(series: pd.Series) -> pd.Series:
    d = coerce_decimal(series)
    return pd.to_numeric(d, errors="coerce").astype("Int64")


def coerce_date_iso(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=False, utc=False)


def coerce_bool_yn(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "y": "Y", "yes": "Y", "true": "Y", "1": "Y", "t": "Y",
        "n": "N", "no": "N", "false": "N", "0": "N", "f": "N",
        "": pd.NA, "nan": pd.NA
    }
    return s.map(mapping).astype("string")


def load_enum_library(config_dir: Path = None) -> dict:
    """Load enum synonym libraries if present (best-effort).
    Searches config_dir (typically config/system/) for enum_synonyms*.yaml.
    """
    if config_dir is None:
        config_dir = Path(__file__).resolve().parent.parent.parent / "config" / "system"
    libs = []
    for name in ["enum_synonyms.yaml", "enum_synonyms_learned.yaml"]:
        p = config_dir / name
        if p.exists():
            try:
                libs.append(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
            except Exception:
                continue
    merged: Dict[str, Any] = {}
    for lib in libs:
        for enum_name, mapping in (lib or {}).items():
            merged.setdefault(enum_name, {})
            if isinstance(mapping, dict):
                merged[enum_name].update(mapping)
    return merged


def validate_enums(df: pd.DataFrame, fields_meta: dict, enum_lib: dict, allow_nd: bool = True) -> List["Violation"]:
    vs: List[Violation] = []
    for col in df.columns:
        meta = fields_meta.get(col) if fields_meta else None
        if not meta:
            continue
        enum_name = meta.get("allowed_values")
        if not enum_name:
            continue
        enum_map = enum_lib.get(enum_name)
        if not enum_map:
            continue

        allowed = set()
        for canon, syns in enum_map.items():
            allowed.add(str(canon).strip())
            if isinstance(syns, list):
                for s in syns:
                    allowed.add(str(s).strip())
            elif isinstance(syns, str):
                allowed.add(str(syns).strip())

        s = df[col]
        for idx, val in s.items():
            if pd.isna(val):
                continue
            if allow_nd and is_nd(val):
                continue
            v = str(val).strip()
            if v == "":
                continue
            if v not in allowed:
                add_violation(
                    vs, "ENUM_INVALID", "warn", col, int(idx),
                    f"Value '{v}' is not in allowed enum set '{enum_name}'."
                )
    return vs


@dataclass
class Violation:
    rule_id: str
    severity: str  # error | warn | info
    field: str
    row: Optional[int]
    message: str


def add_violation(vs: List[Violation], rule_id: str, severity: str, field: str, row: Optional[int], message: str):
    vs.append(Violation(rule_id, severity, field, row, message))


def normalise_severity(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in ("warn", "warning"):
        return "warn"
    if s in ("info",):
        return "info"
    if s in ("error",):
        return "error"
    # default
    return "warn"


# -----------------------------
# Core validation (with applicability)
# -----------------------------

def validate_core_presence(
    df: pd.DataFrame,
    required: List[str],
    reg_fields: Dict[str, Dict[str, Any]],
    portfolio_type: str
) -> List[Violation]:
    """
    Validate that core_canonical fields are present and populated.

    Applicability overrides (per field) can relax requirements for specific portfolio_type:
      applicability.<portfolio_type>.allowed_missing = true
      applicability.<portfolio_type>.severity_if_missing = warn|info|error (default warn)
      applicability.<portfolio_type>.nd_default = ND2 (optional; allows ND2 in canonical truth set)
      applicability.<portfolio_type>.reason = "..." (optional)
    """
    vs: List[Violation] = []

    # 1) Missing columns
    missing_cols = [c for c in required if c not in df.columns]
    for c in missing_cols:
        meta = reg_fields.get(c, {}) or {}
        app = get_applicability(meta, portfolio_type)

        if bool(app.get("allowed_missing", False)):
            sev = normalise_severity(app.get("severity_if_missing", "warning"))
            reason = str(app.get("reason", "")).strip()
            msg = "Core field is not present but is allowed missing for this portfolio_type"
            if reason:
                msg += f" ({reason})"
            add_violation(vs, "CORE001", sev, c, None, msg + ".")
        else:
            add_violation(vs, "CORE001", "error", c, None, "Missing required core_canonical field (column not present).")

    if missing_cols:
        # If columns are missing, we cannot safely do row-level checks for those columns.
        return vs

    # 2) Blank values / ND values
    for c in required:
        meta = reg_fields.get(c, {}) or {}
        app = get_applicability(meta, portfolio_type)
        allowed_missing = bool(app.get("allowed_missing", False))
        nd_default = str(app.get("nd_default", "")).strip().upper()  # e.g., ND2
        sev = normalise_severity(app.get("severity_if_missing", "warning"))
        reason = str(app.get("reason", "")).strip()

        blank_mask = df[c].apply(is_blank)
        if blank_mask.any():
            if allowed_missing and bool(blank_mask.all()):
                msg = f"Core field is blank for all rows but is allowed missing for portfolio_type='{(portfolio_type or '').strip()}'."
                if reason:
                    msg = msg[:-1] + f" ({reason})."
                add_violation(vs, "CORE002", sev, c, None, msg)
            else:
                for idx in df.index[blank_mask].tolist():
                    add_violation(vs, "CORE002", "error", c, int(idx), "Missing required core_canonical value (blank/null).")

        nd_mask = df[c].apply(is_nd)
        if nd_mask.any():
            # Canonical truth set normally forbids ND codes; allow only if applicability explicitly permits them.
            if allowed_missing and nd_default and bool(nd_mask.all()):
                # Ensure the ND values match the declared default (e.g., all ND2)
                vals = df.loc[nd_mask, c].astype(str).str.strip().str.upper().unique().tolist()
                if len(vals) == 1 and vals[0] == nd_default:
                    msg = (
                        f"Core field contains {nd_default} for all rows and is permitted by applicability "
                        f"for portfolio_type='{(portfolio_type or '').strip()}'."
                    )
                    if reason:
                        msg += f" ({reason})"
                    add_violation(vs, "CORE003", sev, c, None, msg + ".")
                else:
                    # Mixed ND or wrong ND -> treat as error
                    for idx in df.index[nd_mask].tolist():
                        add_violation(vs, "CORE003", "error", c, int(idx), "ND codes are not permitted in full canonical (truth set).")
            else:
                for idx in df.index[nd_mask].tolist():
                    add_violation(vs, "CORE003", "error", c, int(idx), "ND codes are not permitted in full canonical (truth set).")

    return vs


# -----------------------------
# Format validation
# -----------------------------

def validate_formats(df: pd.DataFrame, reg_fields: Dict[str, Dict[str, Any]], scope: str) -> List[Violation]:
    vs: List[Violation] = []
    for col in df.columns:
        if col not in reg_fields:
            continue
        fmt = str(reg_fields[col].get("format", "")).strip().lower()
        if not fmt:
            continue

        series = df[col]

        if scope == "regime":
            mask = ~series.apply(is_nd) & ~series.apply(is_blank)
        else:
            mask = ~series.apply(is_blank)

        if not mask.any():
            continue

        if fmt == "date":
            parsed = coerce_date_iso(series[mask])
            bad = parsed.isna()
            for idx in parsed.index[bad].tolist():
                original_value = df.loc[idx, col]
                add_violation(
                    vs,
                    "FMT_DATE",
                    "error",
                    col,
                    int(idx),
                    f"Invalid date format: '{original_value}' could not parse to ISO 8601."
                )

        elif fmt == "currency_code":
            bad = ~series[mask].astype(str).str.strip().str.upper().str.match(CCY_CODE_PATTERN)
            for idx in series[mask].index[bad].tolist():
                original_value = df.loc[idx, col]
                add_violation(
                    vs, "FMT_CCY_CODE", "error", col, int(idx),
                    f"Invalid ISO currency code: '{original_value}' (expected 3-letter ISO 4217 code)."
                )

        elif fmt in ("decimal", "number", "numeric"):
            parsed = coerce_decimal(series[mask])
            bad = parsed.isna()
            for idx in parsed.index[bad].tolist():
                add_violation(vs, "FMT_DEC", "error", col, int(idx), "Invalid decimal; could not parse.")

        elif fmt in ("integer", "int"):
            parsed = coerce_int(series[mask])
            bad = parsed.isna()
            for idx in parsed.index[bad].tolist():
                add_violation(vs, "FMT_INT", "error", col, int(idx), "Invalid integer; could not parse.")

        elif fmt in ("y/n", "yn", "boolean", "bool"):
            parsed = coerce_bool_yn(series[mask])
            bad = parsed.isna()
            for idx in parsed.index[bad].tolist():
                add_violation(vs, "FMT_BOOL", "error", col, int(idx), "Invalid boolean; expected Y/N (or yes/no/true/false/1/0).")

        elif fmt == "list":
            # enum validation is handled separately
            continue

        else:
            continue

    return vs


# -----------------------------
# Regime validation (basic)
# -----------------------------

def validate_regime_schema_and_mandatory(df: pd.DataFrame, reg_fields: Dict[str, Dict[str, Any]], regime: str) -> List[Violation]:
    vs: List[Violation] = []
    regime_list = regime_fields(reg_fields, regime)

    expected_cols = [f for f, _ in regime_list]
    missing = [c for c in expected_cols if c not in df.columns]
    for c in missing:
        add_violation(vs, "REG001", "error", c, None, f"Missing regime field for {regime} (column not present).")

    if missing:
        return vs

    for f, rm in regime_list:
        allowed_nds = rm.get("allowed_nd_codes", None)
        if allowed_nds is not None and f in df.columns:
            allowed_set = set(int(x) for x in (allowed_nds or []))
            for idx, val in df[f].items():
                ndn = nd_number(val)
                if ndn is None:
                    continue
                if ndn not in allowed_set:
                    add_violation(
                        vs, "ND_PERM", "error", f, int(idx),
                        f"ND{ndn} is not permitted for this field in {regime}."
                    )

        priority = str(rm.get("priority", "")).strip().lower()
        if priority != "mandatory":
            continue
        blank_mask = df[f].apply(is_blank)
        ok_mask = ~blank_mask
        if (~ok_mask).any():
            for idx in df.index[~ok_mask].tolist():
                add_violation(vs, "REG002", "error", f, int(idx), f"Mandatory field for {regime} is blank/null (ND allowed but empty not).")

    return vs

def print_summary(violations: List[Violation], total_rows: int, columns: List[str]):
    """Aggregates and prints a field-centric summary of violations."""
    if not violations:
        print("\n✓ NO VIOLATIONS FOUND")
        return

    # Aggregate by field -> severity -> rule
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    field_error_set = set()
    field_warn_set = set()

    # We also want a sample message for the rule
    rule_messages = {}

    for v in violations:
        stats[v.field][v.severity][v.rule_id] += 1
        rule_messages[(v.field, v.rule_id)] = v.message
        if v.severity == "error":
            field_error_set.add(v.field)
        elif v.severity == "warn":
            field_warn_set.add(v.field)

    print("\n" + "="*60)
    print(f" VALIDATION SUMMARY (Aggregated by Field)")
    print("="*60)

    # Sort fields: Errors first, then Warnings, then alphabetically
    sorted_fields = sorted(stats.keys(), key=lambda k: (
        0 if "error" in stats[k] else 1,
        k
    ))

    for field in sorted_fields:
        sevs = stats[field]
        # Calculate total issues for this field
        total_err = sum(sevs.get("error", {}).values())
        total_warn = sum(sevs.get("warn", {}).values())
        
        status_str = []
        if total_err > 0:
            status_str.append(f"{total_err} Errors")
        if total_warn > 0:
            status_str.append(f"{total_warn} Warnings")
            
        print(f"\nField: {field} [{', '.join(status_str)}]")
        
        # Print breakdown by rule
        for sev in ["error", "warn", "info"]:
            if sev not in sevs:
                continue
            for rule_id, count in sevs[sev].items():
                msg = rule_messages.get((field, rule_id), "")
                # Truncate long messages for summary
                if len(msg) > 80:
                    msg = msg[:77] + "..."
                prefix = "  [ERROR]" if sev == "error" else f"  [{sev.upper()}]"
                pct = (count / total_rows) * 100 if total_rows > 0 else 0
                print(f"{prefix} Rule {rule_id}: {count} rows ({pct:.1f}%) -> {msg}")

    print("\n" + "-"*60)
    print(f" STATISTICS")
    print("-"*60)
    print(f" Total Rows Validated: {total_rows}")
    print(f" Total Columns:        {len(columns)}")
    print(f" Fields with ERRORS:   {len(field_error_set)}")
    print(f" Fields with WARNINGS: {len(field_warn_set)}")
    
    total_violation_count = len(violations)
    print(f" Total Logged Issues:  {total_violation_count} (Detailed in CSV)")
    print("-"*60)

    if field_error_set:
        print(f"✗ VALIDATION FAILED: {len(field_error_set)} fields have data quality errors.")
    else:
        print("✓ VALIDATION PASSED")
        
# -----------------------------
# Main
# -----------------------------

def write_violations(out_path: Path, violations: List[Violation]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rule_id", "severity", "field", "row", "message"])
        for v in violations:
            w.writerow([v.rule_id, v.severity, v.field, "" if v.row is None else v.row, v.message])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Canonical CSV (full canonical or regime projection)")
    p.add_argument("--registry", required=True, help="Field registry YAML")
    p.add_argument("--portfolio-type", default="equity_release", help="Portfolio type (controls registry field selection)")
    p.add_argument("--scope", choices=["canonical", "regime"], default="canonical", help="Validation scope")
    p.add_argument("--regime", default=None, help="Regime name (required if scope=regime)")
    p.add_argument("--out-dir", default="out_validation", help="Output directory for violations CSV")
    p.add_argument("--config", default=None, help="Client config YAML")
    args = p.parse_args()
    
    # -------------------------------
    # Load client config (if provided)
    # -------------------------------
    import yaml

    config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    
    # ---------------------------------
    # Override CLI args from config
    # ---------------------------------
    if "portfolio" in config:
        args.portfolio_type = config["portfolio"].get(
            "asset_class", args.portfolio_type
        )

    if args.scope == "regime" and not args.regime:
        args.regime = config.get("regime")

    if args.scope == "regime" and not args.regime:
        raise ValueError("scope=regime requires a regime (CLI or config)")

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(inp)

    reg_path = Path(args.registry)
    if not reg_path.is_absolute():
        reg_path = Path(__file__).resolve().parent / reg_path

    registry = load_registry(reg_path)
    reg_fields = select_fields_for_portfolio(registry, args.portfolio_type)

    df = pd.read_csv(inp, low_memory=False)

    violations: List[Violation] = []

    if args.scope == "canonical":
        required = get_core_required_fields(reg_fields)
        violations += validate_core_presence(df, required, reg_fields, args.portfolio_type)
        violations += validate_formats(df, reg_fields, scope="canonical")
        enum_lib = load_enum_library(config_dir=reg_path.parent)
        violations += validate_enums(df, reg_fields, enum_lib, allow_nd=False)
        out_name = f"{inp.stem}_canonical_violations.csv"

    else:
        if not args.regime:
            raise ValueError("--regime is required when --scope=regime")
        violations += validate_regime_schema_and_mandatory(df, reg_fields, args.regime)
        violations += validate_formats(df, reg_fields, scope="regime")
        enum_lib = load_enum_library(config_dir=reg_path.parent)
        violations += validate_enums(df, reg_fields, enum_lib, allow_nd=True)
        out_name = f"{inp.stem}_{args.regime}_violations.csv"

    out_path = Path(args.out_dir) / out_name
    write_violations(out_path, violations)

    print(f"Detailed violations log written to: {out_path}")

    # Aggregated Console Reporting
    print_summary(violations, len(df), df.columns.tolist())

    if args.scope == "canonical":
        required = get_core_required_fields(reg_fields)
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"\nMissing Core Fields: {', '.join(missing)}")
            
if __name__ == "__main__":
    main()