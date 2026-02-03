#!/usr/bin/env python3
"""
canonical_transform_frozen_v1_9.py

Purpose (locked contract):
- Read a canonical dataset produced by messy_to_canonical (truth set; no ND padding)
- Standardise formats according to the field registry.
- Enrich Geography (NUTS/ITL) via config-driven strategy.
- Apply deterministic derivations (classification, LTV, reporting date).
- Apply last-mile defaults.
- Emit:
    * <stem>_canonical_typed.csv
    * <stem>_transform_report.json

Target State Updates (v1.9):
- ARCHITECTURE: Config-Driven Reporting Date (Priority 1).
- FIX: "Ghost Rows" purged immediately.
- FIX: Equity Release Principal Balance derivation.
"""

import argparse
import json
import re
import calendar
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import pandas as pd
import yaml


ND_PATTERN = re.compile(r"^ND\d+$", re.IGNORECASE)


def load_registry(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "fields" not in data or not isinstance(data["fields"], dict):
        raise ValueError(f"Registry missing 'fields' mapping: {path}")
    return data


def load_yaml_optional(path_str: str) -> Optional[dict]:
    """Load optional YAML file, returning None if missing or empty."""
    if not path_str or str(path_str).strip() == "":
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        content = yaml.safe_load(p.read_text(encoding="utf-8"))
        return content if content else None
    except Exception as e:
        print(f"Warning: Failed to load YAML from {path_str}: {e}")
        return None

# --- HELPER: Smart Date Parsing ---
def smart_parse_cutoff_date(val, default_year=2025):
    """
    Handles standard dates (2025-11-30) AND Month names (November).
    Returns ISO YYYY-MM-DD string or None.
    """
    if pd.isna(val) or str(val).strip() == "":
        return None
        
    s_val = str(val).strip()
    
    # 1. Try mapping full/short month name (November, Nov)
    month_map = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
    month_map.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
    
    if s_val.lower() in month_map:
        month_idx = month_map[s_val.lower()]
        try:
            last_day = calendar.monthrange(default_year, month_idx)[1]
            return f"{default_year}-{month_idx:02d}-{last_day}"
        except:
            return None
        
    # 2. Fallback to standard ISO parsing
    try:
        dt = pd.to_datetime(s_val, dayfirst=True)
        return dt.strftime("%Y-%m-%d")
    except:
        return None
# ---------------------------

def select_fields_for_portfolio(registry: dict, portfolio_type: str) -> dict:
    """Return registry['fields'] subset relevant to a given portfolio type."""
    pt = (portfolio_type or "").strip().lower()
    out = {}
    for fname, meta in (registry.get("fields") or {}).items():
        fpt = str((meta or {}).get("portfolio_type", "")).strip().lower()
        if fpt == "common" or fpt == pt:
            out[fname] = meta
    return out

def _strip_nd(series: pd.Series) -> pd.Series:
    """Treat ND codes as missing in transform step."""
    if series.dtype == object:
        return series.where(~series.astype(str).str.match(ND_PATTERN), other=pd.NA)
    return series


def to_iso_date(series: pd.Series, dayfirst: bool = True) -> pd.Series:
    """Robust Date Parser (v1.8)"""
    s = _strip_nd(series)
    s_num = pd.to_numeric(s, errors="coerce")
    is_serial = s_num.notna() & (s_num > 25000)
    
    dt_out = pd.Series(pd.NaT, index=s.index)

    if is_serial.any():
        dt_out.loc[is_serial] = pd.to_datetime(
            s_num[is_serial], unit="D", origin="1899-12-30"
        )

    is_str = s.notna() & ~is_serial
    if is_str.any():
        dt_out.loc[is_str] = pd.to_datetime(
            s[is_str], dayfirst=dayfirst, errors="coerce", utc=False
        )

    return dt_out.dt.strftime("%Y-%m-%d")


def to_decimal(series: pd.Series) -> pd.Series:
    s = _strip_nd(series)
    if s.dtype == object:
        cleaned = (
            s.astype(str)
            .str.replace(r"[^\d\-\.,]", "", regex=True)
            .str.replace(",", "", regex=False)
        )
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def to_integer(series: pd.Series) -> pd.Series:
    num = to_decimal(series)
    return num.round(0).astype("Int64")


def to_bool_yn(series: pd.Series) -> pd.Series:
    s = _strip_nd(series)
    if s.dtype != object:
        return s.map(lambda v: "Y" if v == 1 else ("N" if v == 0 else pd.NA))
    t = s.astype(str).str.strip().str.lower()
    truthy = {"y", "yes", "true", "t", "1"}
    falsy = {"n", "no", "false", "f", "0"}
    def _map(v: str):
        if v in truthy: return "Y"
        if v in falsy: return "N"
        return pd.NA
    return t.map(_map)


def to_currency(series: pd.Series, synonym_map: dict | None = None) -> pd.Series:
    s = _strip_nd(series)
    if s.dtype != object:
        return s.astype("string")
    t = s.astype(str).str.strip().str.upper()
    if synonym_map:
        t = t.replace(synonym_map)
    else:
        t = t.replace({"UKP": "GBP", "UKL": "GBP", "EURO": "EUR", "EUROS": "EUR"})
    t = t.replace({"": pd.NA})
    return t.astype("string")


def apply_types(df: pd.DataFrame, fields_meta: dict, currency_synonyms: dict | None = None, dayfirst: bool = True) -> Dict[str, Any]:
    report: Dict[str, Any] = {"fields": {}, "rows": int(len(df))}
    
    for col in list(df.columns):
        meta = fields_meta.get(col)
        if not meta: continue
            
        fmt = str(meta.get("format", "")).strip().lower()
        before_null = int(df[col].isna().sum())
        original = df[col].copy()
        
        if fmt == "date":
            out = to_iso_date(df[col], dayfirst=dayfirst)
        elif fmt in {"decimal", "number", "float"}:
            out = to_decimal(df[col])
        elif fmt in {"integer", "int"}:
            out = to_integer(df[col])
        elif fmt in {"boolean", "bool", "y/n"}:
            out = to_bool_yn(df[col])
        elif fmt in {"currency_code", "ccy_code", "iso_currency_code", "currency", "ccy"} or col.endswith("_currency"):
            out = to_currency(df[col], synonym_map=currency_synonyms)
        else:
            out = _strip_nd(df[col])
            if out.dtype == object:
                out = out.astype("string").str.strip()

        df[col] = out
        
        # Metrics
        nd_mask = original.astype(str).str.match(ND_PATTERN, na=False)
        failures_mask = original.notna() & out.isna() & ~nd_mask
        
        sample = []
        if failures_mask.sum() > 0:
            try:
                sample = original[failures_mask].astype('string').dropna().drop_duplicates().head(5).tolist()
            except: pass
                
        report["fields"][col] = {
            "format": fmt or "string",
            "nulls_before": before_null,
            "nulls_after": int(df[col].isna().sum()),
            "nd_stripped": int(nd_mask.sum()),
            "parse_failures": int(failures_mask.sum()),
            "sample_failures": sample,
        }
    return report

def derive_reporting_date(df, filename, dayfirst, infer_year, derive_month, default_year):
    # This legacy function handles the column-based parsing (95% Case)
    # It is called inside derive_fields IF config override is not present.
    report = {"derived": {}, "skipped": {}}
    col = "data_cut_off_date"
    
    if col not in df.columns:
        report["skipped"][col] = "Column not present"
        return report

    s = df[col]
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)

    # 1. Infer Year Context
    inferred_year = default_year
    if infer_year and filename:
        m = re.search(r"(19\d{2}|20\d{2})", filename)
        if m: inferred_year = int(m.group(1))

    # 2. Derive Month Ends
    if derive_month:
        month_map = {
            "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
            "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
            "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
            "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
        }
        derived = parsed.copy()
        needs = parsed.isna() & s.notna()
        
        for idx in df.index[needs]:
            raw = str(df.at[idx, col]).strip().lower()
            # Try Regex (MM/YYYY or YYYY/MM)
            m1 = re.match(r"^(\d{1,2})\s*[/\-]\s*(19\d{2}|20\d{2})$", raw)
            if m1:
                end = pd.Period(f"{m1.group(2)}-{int(m1.group(1)):02d}", freq="M").end_time.normalize()
                derived.at[idx] = end
                continue
                
            # Try Month Name (needs context year)
            if inferred_year and raw in month_map:
                end = pd.Period(f"{inferred_year}-{month_map[raw]:02d}", freq="M").end_time.normalize()
                derived.at[idx] = end
                continue
        
        df[col] = pd.to_datetime(derived, errors="coerce").dt.strftime("%Y-%m-%d")

    return report

# --- UPDATED: Accepts Config Object ---
def derive_fields(df: pd.DataFrame, portfolio_type: str, filename: str, 
                 dayfirst: bool, infer_year: bool, derive_month: bool, 
                 default_year: Optional[int], config: dict) -> Dict[str, Any]:
    
    deriv_report: Dict[str, Any] = {"derived": {}, "skipped": {}}
    pt = (portfolio_type or "").strip().lower()
    is_erm = pt in {"equity_release", "erm", "rre"}

    # 1. ERM Balance Coherence
    if is_erm:
        for col in ["current_outstanding_balance", "current_principal_balance", "accrued_interest"]:
            if col not in df.columns: df[col] = pd.NA

        o = pd.to_numeric(df["current_outstanding_balance"], errors="coerce")
        p = pd.to_numeric(df["current_principal_balance"], errors="coerce")
        i = pd.to_numeric(df["accrued_interest"], errors="coerce").fillna(0.0)

        # Outstanding = Principal + Accrued
        mask_out = o.isna() & p.notna()
        if mask_out.any():
            df.loc[mask_out, "current_outstanding_balance"] = p[mask_out] + i[mask_out]

        # Principal = Outstanding - Accrued
        o = pd.to_numeric(df["current_outstanding_balance"], errors="coerce")
        mask_prin = p.isna() & o.notna()
        if mask_prin.any():
            df.loc[mask_prin, "current_principal_balance"] = o[mask_prin] - i[mask_prin]

    # 2. LTV Calculations
    for ltv_col, bal_col, val_col in [
        ("current_loan_to_value", "current_principal_balance", "current_valuation_amount"),
        ("original_loan_to_value", "original_principal_balance", "original_valuation_amount")
    ]:
        if {bal_col, val_col}.issubset(df.columns):
            # Ensure column exists
            if ltv_col not in df.columns: 
                df[ltv_col] = pd.NA

            b = pd.to_numeric(df[bal_col], errors="coerce")
            v = pd.to_numeric(df[val_col], errors="coerce")
            
            if ltv_col == "current_loan_to_value":
                mask = b.notna() & v.notna() & (v != 0) 
            else:
                mask = df[ltv_col].isna() & b.notna() & v.notna() & (v != 0)

            if mask.any():
                df.loc[mask, ltv_col] = (b[mask] / v[mask]) * 100.0
                
                # Optional: Add to derivation report so you know it happened
                deriv_report.setdefault("derived", {})[ltv_col] = {
                    "rule_id": f"DERIVE_{ltv_col.upper()}_FORCED", 
                    "filled_rows": int(mask.sum()),
                    "logic": "Forced calc: (bal/val)*100 to fix scaling"
                }

    # 3. Geographic Classification
    if "geographic_region_classification" in df.columns:
        if "geographic_region_classification_source" not in df.columns:
            df["geographic_region_classification_source"] = pd.NA

        # Mark provided
        tgt = df["geographic_region_classification"]
        missing = tgt.isna() | (tgt.astype(str).str.strip() == "")
        df.loc[~missing & df["geographic_region_classification_source"].isna(), "geographic_region_classification_source"] = "provided"

        secured_pts = {"equity_release", "erm", "rre", "cre", "commercial_real_estate", "residential_mortgage"}
        if pt in secured_pts:
            precedence = [("collateral", "geographic_region_collateral"), ("obligor", "geographic_region_obligor")]
        else:
            precedence = [("obligor", "geographic_region_obligor"), ("collateral", "geographic_region_collateral")]

        filled_total = 0
        
        for src_label, src_col in precedence:
            if src_col not in df.columns: continue
            src = df[src_col]
            can_fill = (
                (df["geographic_region_classification"].isna() | (df["geographic_region_classification"].astype(str).str.strip() == ""))
                & src.notna() & (src.astype(str).str.strip() != "")
            )
            if can_fill.any():
                df.loc[can_fill, "geographic_region_classification"] = src[can_fill]
                df.loc[can_fill, "geographic_region_classification_source"] = src_label
                n = int(can_fill.sum())
                filled_total += n

        if filled_total > 0:
            deriv_report.setdefault("derived", {})["geographic_region_classification"] = {
                "rule_id": "DERIVE_GEO_CLASSIFICATION_V1", 
                "filled_rows": filled_total, 
                "logic": f"Precedence: {precedence}"
            }
    
    # 4. REPORTING DATE (Config-Driven Priority)
    # PRIORITY 1: Config Override (The 5% Case)
    static_date = (config.get("portfolio") or {}).get("static_reporting_date")
    
    if static_date:
        print(f"  [CONFIG OVERRIDE] Enforcing reporting date: {static_date}")
        if "data_cut_off_date" not in df.columns:
            df["data_cut_off_date"] = pd.NA
        df["data_cut_off_date"] = static_date
        deriv_report["derived"]["data_cut_off_date"] = {
            "rule_id": "CONFIG_OVERRIDE_DATE",
            "filled_rows": len(df),
            "logic": f"Static value from config: {static_date}"
        }
        
    # PRIORITY 2: Data Derived (The 95% Case)
    elif "data_cut_off_date" in df.columns:
        # Use existing derivation logic
        derive_reporting_date(df, filename, dayfirst, infer_year, derive_month, default_year)
        
        # Apply Smart Parse (fixes "November")
        context_year = default_year or 2025
        if infer_year and filename:
            m = re.search(r"(19\d{2}|20\d{2})", filename)
            if m: context_year = int(m.group(1))

        df["data_cut_off_date"] = df["data_cut_off_date"].apply(
            lambda x: smart_parse_cutoff_date(x, default_year=context_year)
        )

    return deriv_report

# -------------------------------------------------------------------------
# GEOGRAPHIC RESOLUTION ENGINE
# -------------------------------------------------------------------------

def _extract_geo_key(postcode: Any, strategy: str = "uk_outcode") -> str:
    if pd.isna(postcode): return ""
    s = str(postcode).strip().upper().replace(" ", "")
    
    if strategy == "uk_outcode":
        # Robust Regex: Handles Full (SW1A1AA) -> SW1A, and Outcode-only (SW1A) -> SW1A
        full_match = re.match(r"^([A-Z]{1,2}[0-9][A-Z0-9]?)([0-9][A-Z]{2})$", s)
        if full_match: return full_match.group(1)
        
        outcode_match = re.match(r"^[A-Z]{1,2}[0-9][A-Z0-9]?$", s)
        if outcode_match: return s
        return "" 
        
    elif strategy == "eu_prefix_2": return s[:2] if len(s) >= 2 else ""
    elif strategy == "eu_prefix_3": return s[:3] if len(s) >= 3 else ""
    elif strategy == "exact": return s
        
    return ""

def load_region_mapping(csv_path: Path) -> Dict[str, str]:
    df = pd.read_csv(csv_path, dtype=str)
    
    # Auto-detect headers (Safe)
    if 'postcode_key' in df.columns and 'region_code' in df.columns:
        key_col, val_col = 'postcode_key', 'region_code'
    elif 'postcode_prefix' in df.columns and 'itl3_code' in df.columns:
        key_col, val_col = 'postcode_prefix', 'itl3_code'
    elif 'Post Code' in df.columns and 'NUTS318CD' in df.columns:
        key_col, val_col = 'Post Code', 'NUTS318CD'
    else:
        # Fallback with Warning
        key_col, val_col = df.columns[0], df.columns[1]

    m = {}
    for _, r in df[[key_col, val_col]].dropna().iterrows():
        k = str(r[key_col]).strip().upper()
        v = str(r[val_col]).strip().upper()
        if k and v: m[k] = v
    return m

def apply_region_lookup(df: pd.DataFrame, mapping: Dict[str, str], target_col: str, postcode_cols: List[str], strategy: str) -> Dict[str, Any]:
    report = {'target': target_col, 'strategy': strategy, 'derived_rows': 0}
    
    if target_col not in df.columns:
        report['skipped'] = "Target column missing"
        return report

    src_col = next((c for c in postcode_cols if c in df.columns), None)
    if not src_col:
        report['skipped'] = "No source column found"
        return report

    tgt = df[target_col].astype('string')
    missing_mask = tgt.isna() | (tgt.str.strip() == '')
    
    if not missing_mask.any(): return report

    # Extract & Map
    keys = df.loc[missing_mask, src_col].apply(lambda x: _extract_geo_key(x, strategy=strategy))
    derived = keys.map(mapping)
    
    fill_mask = derived.notna() & (derived != "")
    rows_to_update = fill_mask[fill_mask].index
    
    df.loc[rows_to_update, target_col] = derived[rows_to_update]
    
    report['source'] = src_col
    report['derived_rows'] = int(len(rows_to_update))
    return report

def apply_config_defaults(df: pd.DataFrame, config: dict) -> dict:
    defaults = (config.get("defaults") or {})
    report = {"applied_defaults": {}}
    for field, default_value in defaults.items():
        if field == "nd_defaults": continue
        if field not in df.columns: df[field] = pd.NA
        mask = df[field].isna() | (df[field].astype(str).str.strip() == "")
        if mask.any():
            df.loc[mask, field] = default_value
    return report

def main() -> None:
    ap = argparse.ArgumentParser(description="Canonical transform (frozen v1.9)")
    ap.add_argument("canonical_csv")
    ap.add_argument("--registry", required=True)
    ap.add_argument("--portfolio-type", default="equity_release")
    ap.add_argument("--currency-synonyms", default="")
    ap.add_argument("--nuts-uk-csv", default="")
    ap.add_argument("--nuts-target-col", default="")
    ap.add_argument("--nuts-postcode-cols", default="")
    ap.add_argument("--output-dir", default="out")
    ap.add_argument("--output-prefix", default=None)
    ap.add_argument("--no-derivations", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    policy = ((config.get("portfolio") or {}).get("reporting_date_policy") or {})
    DAYFIRST = bool(policy.get("dayfirst_dates", True))
    INFER_YEAR = bool(policy.get("infer_year_from_filename", True))
    DERIVE_MONTH = bool(policy.get("derive_month_end_if_missing", True))
    DEFAULT_YEAR = int(policy.get("default_year", 2025))

    in_path = Path(args.canonical_csv)
    reg_path = Path(args.registry)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, low_memory=False)
    
    # 1. GHOST ROW PURGE (Circuit Breaker)
    print(f"Rows before purge: {len(df)}")
    valid_mask = pd.Series(False, index=df.index)
    if "loan_identifier" in df.columns:
        valid_mask |= df["loan_identifier"].notna() & (df["loan_identifier"].astype(str).str.strip() != "")
    if "unique_identifier" in df.columns:
        valid_mask |= df["unique_identifier"].notna() & (df["unique_identifier"].astype(str).str.strip() != "")
    
    if valid_mask.any():
        df = df[valid_mask].copy()
        print(f"Rows after purge: {len(df)}")
    else:
        print("Warning: No valid identifiers. Skipping purge.")

    registry = load_registry(reg_path)
    fields_meta = select_fields_for_portfolio(registry, args.portfolio_type)
    currency_synonyms = load_yaml_optional(args.currency_synonyms)

    # 2. Typing
    type_report = apply_types(df, fields_meta, currency_synonyms, dayfirst=DAYFIRST)

    # 3. Region Lookup
    nuts_report = {}
    geo_config = config.get("nuts_lookup", {})
    geo_path_str = args.nuts_uk_csv or geo_config.get("source_file")
    if geo_path_str:
        geo_path = Path(geo_path_str)
        if not geo_path.exists() and (Path("reference_data") / geo_path).exists():
            geo_path = Path("reference_data") / geo_path
        if geo_path.exists():
            print(f"Loading regions from {geo_path.name}...")
            region_map = load_region_mapping(geo_path)
            target = args.nuts_target_col or geo_config.get("target_field", "geographic_region_collateral")
            srcs = (args.nuts_postcode_cols or geo_config.get("postcode_columns", "")).split(",")
            nuts_report = {"region_lookup": apply_region_lookup(df, region_map, target, srcs, "uk_outcode")}

    # 4. Derivations (UPDATED: Now passes `config`)
    deriv_report = {}
    if not args.no_derivations:
        deriv_report = derive_fields(df, args.portfolio_type, in_path.name, DAYFIRST, 
                                   INFER_YEAR, DERIVE_MONTH, DEFAULT_YEAR, config)

    # 5. Defaults
    defaults_report = apply_config_defaults(df, config)

    # Output
    stem = args.output_prefix or in_path.stem.replace("_canonical_full", "")
    out_csv = out_dir / f"{stem}_canonical_typed.csv"
    out_json = out_dir / f"{stem}_transform_report.json"

    df.to_csv(out_csv, index=False)
    
    report = {
        "input": str(in_path.name), 
        "output": str(out_csv.name), 
        **type_report, 
        **nuts_report, 
        **deriv_report, 
        **defaults_report
    }
    
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")

if __name__ == "__main__":
    main()
