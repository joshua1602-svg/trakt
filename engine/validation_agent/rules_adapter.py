"""
rules_adapter.py
================

Clean adapter that runs deterministic canonical value-level and cross-field
validation over the **Transformation Agent transformed canonical tape**, reusing
the existing ``engine.gate_3_validation`` primitives instead of duplicating them.

It reuses:
  * :func:`engine.gate_3_validation.validate_canonical.is_blank` / ``is_nd`` /
    ``nd_number`` — blank / ND detection;
  * ``coerce_decimal`` / ``coerce_date_iso`` / ``coerce_bool_yn`` — type coercion;
  * ``load_registry`` / ``load_enum_library`` — registry + enum library loaders.

The legacy gate-3 validators assume legacy gate outputs and emit row-level
``Violation`` objects keyed to a regime. This adapter instead produces
**field/check-level results** and **value-level issues** scoped to the canonical
transformed tape + the transformation field contract, so the Validation Agent
can validate canonical data without re-running raw Gate 1 or projecting to a
regime.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from engine.gate_3_validation import validate_canonical as vc

# Reused primitives (single source of truth).
is_blank = vc.is_blank
is_nd = vc.is_nd
nd_number = vc.nd_number
coerce_decimal = vc.coerce_decimal
coerce_date_iso = vc.coerce_date_iso
coerce_bool_yn = vc.coerce_bool_yn

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LEI_RE = re.compile(r"^[A-Z0-9]{20}$")
COUNTRY_RE = re.compile(r"^[A-Z]{2}$")
BOOL_OK = {"Y", "N", "TRUE", "FALSE", "T", "F", "YES", "NO", "0", "1"}

# Plausible configured rate bounds (percent). Conservative + configurable.
RATE_MIN = 0.0
RATE_MAX = 100.0


def load_registry_fields(registry_path: str) -> Dict[str, Any]:
    try:
        registry = vc.load_registry(Path(registry_path))
    except Exception:
        return {}
    return registry.get("fields", {}) or {}


def load_enum_lib(config_dir: str = "") -> Dict[str, Any]:
    try:
        return vc.load_enum_library(Path(config_dir) if config_dir else None)
    except Exception:
        return {}


def build_regime_index(regime_cfg: Optional[dict]) -> Dict[str, Dict[str, Any]]:
    """Index regime field rules by canonical field name (projected_source_field)."""
    out: Dict[str, Dict[str, Any]] = {}
    for code, rule in ((regime_cfg or {}).get("field_rules", {}) or {}).items():
        rule = rule or {}
        canonical = rule.get("projected_source_field", "")
        if canonical:
            out[canonical] = {
                "esma_code": code,
                "mandatory": bool(rule.get("mandatory", False)),
                "enforce_presence": bool(rule.get("enforce_presence", False)),
                "nd_allowed": rule.get("nd_allowed", []) or [],
                "default_allowed": bool(rule.get("default_allowed", False)),
                "regex": (rule.get("validators", {}) or {}).get("regex", ""),
            }
    return out


# --------------------------------------------------------------------------- #
# Result + issue records
# --------------------------------------------------------------------------- #

def _result(rule_id, field, canonical, esma, check_type, status, severity,
            checked, failures, warnings, samples, b_val, b_proj, notes="") -> Dict[str, Any]:
    return {
        "validation_rule_id": rule_id,
        "field": field,
        "canonical_field": canonical,
        "esma_code": esma,
        "check_type": check_type,
        "status": status,            # pass | warning | fail | not_checked
        "severity": severity,        # info | warn | error
        "row_count_checked": int(checked),
        "failure_count": int(failures),
        "warning_count": int(warnings),
        "sample_failures": "; ".join(str(s) for s in (samples or [])[:5]),
        "blocking_for_validation": bool(b_val),
        "blocking_for_projection": bool(b_proj),
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# Value-level checks (one result row per field/check; issues for fails/warns)
# --------------------------------------------------------------------------- #

def _nonblank_nonnd(series: pd.Series) -> pd.Series:
    mask = series.map(lambda v: not is_blank(v) and not is_nd(v))
    return series[mask]


def validate_field(
    df: pd.DataFrame,
    canonical: str,
    fmt: str,
    *,
    esma_code: str = "",
    regime_rule: Optional[Dict[str, Any]] = None,
    enum_name: str = "",
    enum_lib: Optional[Dict[str, Any]] = None,
    mandatory: bool = False,
) -> List[Dict[str, Any]]:
    """Run deterministic value-level checks for one canonical column.

    Returns a list of field/check result rows. ND codes are treated as valid
    presence markers (never type/enum failures) — they are a controlled signal,
    not bad data.
    """
    results: List[Dict[str, Any]] = []
    if canonical not in df.columns:
        return results
    col = df[canonical]
    checked = int(len(col))
    rid = f"VR-{canonical}"

    # 1) presence / completeness (for mandatory validation-owned fields)
    rule = regime_rule or {}
    if mandatory:
        blanks = int(col.map(is_blank).sum())
        # A blank in a mandatory field that PERMITS an ND/default is a downstream
        # config gap (blocking for projection, not for validation). A blank in a
        # field with no ND/default permitted (e.g. an identifier) is a true,
        # validation-blocking failure.
        defaultable = bool(rule.get("default_allowed")) or bool(rule.get("nd_allowed"))
        if blanks == 0:
            status, severity, b_val, b_proj = "pass", "info", False, False
        elif defaultable:
            status, severity, b_val, b_proj = "warning", "warn", False, True
        else:
            status, severity, b_val, b_proj = "fail", "error", True, True
        results.append(_result(
            f"{rid}-presence", canonical, canonical, esma_code, "presence",
            status, severity, checked, blanks if not defaultable else 0,
            blanks if defaultable else 0, [] if not blanks else ["<blank>"],
            b_val, b_proj,
            notes=("mandatory field completeness"
                   + ("; ND/default permitted -> config gap" if defaultable else ""))))

    vals = _nonblank_nonnd(col)

    # 2) type / format checks
    if fmt == "date":
        bad = [str(v) for v in vals if not ISO_DATE_RE.match(str(v).strip())]
        results.append(_result(
            f"{rid}-date", canonical, canonical, esma_code, "type_date",
            "pass" if not bad else "fail", "error" if bad else "info",
            len(vals), len(bad), 0, bad, bool(bad), bool(bad),
            notes="ISO YYYY-MM-DD"))
    elif fmt in {"decimal", "number", "float", "integer", "int"}:
        coerced = coerce_decimal(vals.astype(str)) if len(vals) else pd.Series([], dtype=float)
        bad = [str(v) for v, c in zip(vals.tolist(), coerced.tolist()) if pd.isna(c)]
        results.append(_result(
            f"{rid}-numeric", canonical, canonical, esma_code, "type_numeric",
            "pass" if not bad else "fail", "error" if bad else "info",
            len(vals), len(bad), 0, bad, bool(bad), bool(bad)))
    elif fmt in {"boolean", "bool", "y/n"}:
        bad = [str(v) for v in vals if str(v).strip().upper() not in BOOL_OK]
        results.append(_result(
            f"{rid}-boolean", canonical, canonical, esma_code, "type_boolean",
            "pass" if not bad else "fail", "error" if bad else "info",
            len(vals), len(bad), 0, bad, bool(bad), bool(bad)))

    # 3) enum membership (canonical enum library; warnings only — projection
    #    performs the authoritative regime enum mapping)
    if enum_name and enum_lib and enum_name in enum_lib:
        allowed = _enum_allowed(enum_lib[enum_name])
        bad = [str(v) for v in vals
               if str(v).strip() not in allowed and str(v).strip().lower()
               not in {a.lower() for a in allowed}]
        if allowed:
            results.append(_result(
                f"{rid}-enum", canonical, canonical, esma_code, "enum",
                "pass" if not bad else "warning", "warn" if bad else "info",
                len(vals), 0, len(bad), bad, False, bool(bad),
                notes=f"canonical enum set '{enum_name}'"))

    # 4) country code
    if enum_name == "country":
        bad = [str(v) for v in vals if not COUNTRY_RE.match(str(v).strip().upper())]
        results.append(_result(
            f"{rid}-country", canonical, canonical, esma_code, "country_code",
            "pass" if not bad else "fail", "error" if bad else "info",
            len(vals), len(bad), 0, bad, bool(bad), bool(bad)))

    # 5) LEI format
    if canonical.endswith("legal_entity_identifier"):
        bad = [str(v) for v in vals if not LEI_RE.match(str(v).strip().upper())]
        results.append(_result(
            f"{rid}-lei", canonical, canonical, esma_code, "lei",
            "pass" if not bad else "fail", "error" if bad else "info",
            len(vals), len(bad), 0, bad, bool(bad), bool(bad)))

    # 6) rate bounds (percentage / rate fields)
    if ("interest_rate" in canonical and fmt in {"decimal", "number", "float"}
            and "index" not in canonical and "type" not in canonical):
        nums = coerce_decimal(vals.astype(str)) if len(vals) else pd.Series([], dtype=float)
        bad = [str(v) for v, n in zip(vals.tolist(), nums.tolist())
               if pd.notna(n) and (n < RATE_MIN or n > RATE_MAX)]
        results.append(_result(
            f"{rid}-rate", canonical, canonical, esma_code, "rate_bounds",
            "pass" if not bad else "warning", "warn" if bad else "info",
            len(vals), 0, len(bad), bad, False, bool(bad),
            notes=f"rate within [{RATE_MIN},{RATE_MAX}]"))

    # 7) regime regex validator (warnings — authoritative check is at projection)
    rule = regime_rule or {}
    rgx = rule.get("regex")
    if rgx:
        try:
            pat = re.compile(rgx)
            bad = [str(v) for v in vals if not pat.match(str(v).strip())]
            results.append(_result(
                f"{rid}-regex", canonical, canonical, esma_code, "regex",
                "pass" if not bad else "warning", "warn" if bad else "info",
                len(vals), 0, len(bad), bad, False, bool(bad),
                notes="regime validator regex (authoritative at projection)"))
        except re.error:
            pass

    return results


def _enum_allowed(enum_map: Any) -> set:
    allowed: set = set()
    if isinstance(enum_map, dict):
        for canon, syns in enum_map.items():
            allowed.add(str(canon).strip())
            if isinstance(syns, list):
                allowed.update(str(s).strip() for s in syns)
            elif isinstance(syns, str):
                allowed.add(str(syns).strip())
    elif isinstance(enum_map, list):
        allowed.update(str(s).strip() for s in enum_map)
    return {a for a in allowed if a}


# --------------------------------------------------------------------------- #
# Identifier uniqueness
# --------------------------------------------------------------------------- #

def validate_uniqueness(df: pd.DataFrame, id_fields: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for f in id_fields:
        if f not in df.columns:
            continue
        col = df[f]
        present = col[col.map(lambda v: not is_blank(v))]
        dupes = present[present.duplicated(keep=False)]
        n = int(dupes.nunique())
        samples = list(dict.fromkeys(str(v) for v in dupes.tolist()))[:5]
        results.append(_result(
            f"VR-{f}-unique", f, f, "", "identifier_uniqueness",
            "pass" if n == 0 else "fail", "error" if n else "info",
            int(len(present)), int(len(dupes)), 0, samples, bool(n), bool(n),
            notes="duplicate identifier"))
    return results


# --------------------------------------------------------------------------- #
# Cross-field business rules
# --------------------------------------------------------------------------- #

def validate_business_rules(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Canonical, configurable cross-field business rules checkable at this stage.

    Deliberately NOT overfit to Annex 2 XML delivery.
    """
    results: List[Dict[str, Any]] = []

    def num(field: str) -> Optional[pd.Series]:
        if field not in df.columns:
            return None
        return coerce_decimal(df[field].astype(str))

    def nonneg(field: str, rid: str) -> None:
        s = num(field)
        if s is None:
            return
        bad_mask = s.notna() & (s < 0)
        bad = [str(v) for v in df.loc[bad_mask, field].tolist()][:5]
        results.append(_result(
            rid, field, field, "", "cross_field_rule",
            "pass" if not bad else "fail", "error" if bad else "info",
            int(s.notna().sum()), len(bad), 0, bad, bool(bad), bool(bad),
            notes=f"{field} >= 0"))

    nonneg("current_valuation_amount", "BR-CURR-VAL-NONNEG")
    nonneg("original_valuation_amount", "BR-ORIG-VAL-NONNEG")
    nonneg("current_loan_to_value", "BR-CLTV-NONNEG")

    # data_cut_off_date present + parseable
    if "data_cut_off_date" in df.columns:
        col = df["data_cut_off_date"]
        bad = [str(v) for v in col
               if is_blank(v) or (not is_nd(v) and not ISO_DATE_RE.match(str(v).strip()))]
        results.append(_result(
            "BR-CUTOFF-DATE", "data_cut_off_date", "data_cut_off_date", "RREL6",
            "cross_field_rule", "pass" if not bad else "fail",
            "error" if bad else "info", int(len(col)), len(bad), 0, bad,
            bool(bad), bool(bad), notes="data_cut_off_date present + ISO parseable"))

    # loan/unique identifier not null
    for f in ("loan_identifier", "unique_identifier"):
        if f in df.columns:
            blanks = int(df[f].map(is_blank).sum())
            results.append(_result(
                f"BR-{f.upper()}-NOTNULL", f, f, "", "cross_field_rule",
                "pass" if not blanks else "fail", "error" if blanks else "info",
                int(len(df)), blanks, 0, ["<blank>"] if blanks else [],
                bool(blanks), bool(blanks), notes=f"{f} not null"))

    # redemption_date >= origination_date where both present
    if {"redemption_date", "origination_date"}.issubset(df.columns):
        rd = coerce_date_iso(df["redemption_date"])
        od = coerce_date_iso(df["origination_date"])
        both = rd.notna() & od.notna()
        bad_mask = both & (rd < od)
        bad = [f"{r.date()}<{o.date()}" for r, o in
               zip(df.loc[bad_mask, "redemption_date"].pipe(coerce_date_iso).tolist(),
                   df.loc[bad_mask, "origination_date"].pipe(coerce_date_iso).tolist())][:5]
        results.append(_result(
            "BR-REDEEM-AFTER-ORIG", "redemption_date", "redemption_date", "",
            "cross_field_rule", "pass" if not bad_mask.any() else "fail",
            "error" if bad_mask.any() else "info", int(both.sum()),
            int(bad_mask.sum()), 0, bad, bool(bad_mask.any()), bool(bad_mask.any()),
            notes="redemption_date >= origination_date"))

    return results
