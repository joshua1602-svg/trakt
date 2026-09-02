#!/usr/bin/env python3
"""
canonical_transform.py

Purpose (locked contract):
- Read a canonical dataset produced by semantic_alignment (truth set; no ND padding)
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
import os
import re
import calendar
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

try:
    from engine import provenance as _provenance
except ModuleNotFoundError:  # pragma: no cover - path bootstrap for subprocess use
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from engine import provenance as _provenance


ND_PATTERN = re.compile(r"^ND\d+$", re.IGNORECASE)
MULTISPACE = re.compile(r"\s+")

DEFAULT_CANONICAL_ENUM_NORMALIZATION: Dict[str, Dict[str, str]] = {
    "property_type": {
        "detached house": "RHOS",
        "detached": "RHOS",
        "house": "RHOS",
        "semi detached": "RHOS",
        "semi-detached": "RHOS",
        "semi detached house": "RHOS",
        "semi-detached house": "RHOS",
        "flat": "RFLT",
        "flat / apartment": "RFLT",
        "apartment": "RFLT",
        "bungalow": "RBGL",
        "rhos": "RHOS",
        "rflt": "RFLT",
        "rbgl": "RBGL",
    },
    "purpose": {
        "home improvements": "RENV",
        "home improvement": "RENV",
        "refinance": "RMRT",
        "refi": "RMRT",
        "debt consolidation": "RMRT",
        "purchase main residence": "PURC",
        "purchase of main residence": "PURC",
        "equity release": "EQRE",
        "renv": "RENV",
        "rmrt": "RMRT",
        "purc": "PURC",
        "eqre": "EQRE",
    },
    "interest_rate_type": {
        "fixed": "FXRL",
        "fixed rate": "FXRL",
        "fxrl": "FXRL",
        "variable": "FLIF",
        "variable rate": "FLIF",
        "floating": "FLIF",
        "flif": "FLIF",
    },
    "customer_type": {
        "individual": "CNEO",
        "joint": "CNEO",
        "cneo": "CNEO",
    },
}


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
        except (ValueError, KeyError):
            return None

    # 2. Fallback to standard ISO parsing
    try:
        dt = pd.to_datetime(s_val, dayfirst=True)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
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


#: Textual renderings of "no value". A CSV round-trip turns a real null into an
#: empty string, and a pandas NA into the literal ``nan``/``NaT``/``<NA>``, so
#: these are ABSENT values, not values that failed to parse. Counting them as
#: parse failures is what made an empty second-borrower DOB look like a broken
#: date.
#:
#: Deliberately limited to unambiguous null RENDERINGS. Client null markers such
#: as ``N/A`` or ``-`` are NOT here: they are real content a source chose to
#: write, and silently voiding them would be a semantic decision this layer has
#: no business making. They keep surfacing as parse failures, which is the point.
_BLANK_TOKENS = {"", "nan", "nat", "none", "null", "<na>"}


def is_blank_token(value: Any) -> bool:
    """True when a cell carries no value (blank / whitespace / null rendering).

    The single definition of "absent" shared by typing and by anything that needs
    to agree with typing about which cells hold a real value — e.g. source-value
    normalisation, which must not offer an operator a decision about a blank.
    """
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in _BLANK_TOKENS


def _blank_token_mask(series: pd.Series) -> pd.Series:
    """Boolean mask of cells that carry no value (see :func:`is_blank_token`)."""
    if series.dtype == object or str(series.dtype) == "string":
        s = series.astype("string").str.strip().str.lower()
        return s.isna() | s.isin(_BLANK_TOKENS)
    return series.isna()


def _void_blanks(series: pd.Series) -> pd.Series:
    """Replace blank-token cells with real NA so parsers never see them."""
    mask = _blank_token_mask(series)
    if not bool(mask.any()):
        return series
    return series.mask(mask, other=pd.NA)


def _normalise_key(value: Any) -> str:
    s = str(value or "").strip().lower()
    return MULTISPACE.sub(" ", s)


def resolve_canonical_enum_normalization(config: dict) -> Dict[str, Dict[str, str]]:
    """
    Resolve canonical enum normalization map.

    Priority:
    1) transformations.canonical_enum_normalization in client config
    2) defaults defined in DEFAULT_CANONICAL_ENUM_NORMALIZATION
    """
    cfg_map = (
        (config.get("transformations") or {}).get("canonical_enum_normalization")
        if isinstance(config, dict) else None
    ) or {}

    merged: Dict[str, Dict[str, str]] = {}
    for field, mapping in DEFAULT_CANONICAL_ENUM_NORMALIZATION.items():
        merged[field] = {_normalise_key(k): str(v).strip() for k, v in mapping.items()}

    for field, mapping in (cfg_map or {}).items():
        if not isinstance(mapping, dict):
            continue
        f = str(field).strip()
        merged.setdefault(f, {})
        for raw_val, target in mapping.items():
            merged[f][_normalise_key(raw_val)] = str(target).strip()

    return merged


def apply_canonical_enum_normalization(
    df: pd.DataFrame,
    normalization_map: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    """
    Deterministically normalize canonical enum strings for internal consistency.
    This is NOT ESMA code mapping.
    """
    report: Dict[str, Any] = {"canonical_enum_normalization": {"fields": {}}}
    if not normalization_map:
        return report

    for field, mapping in normalization_map.items():
        if field not in df.columns or not isinstance(mapping, dict) or not mapping:
            continue

        s = df[field].astype("string")
        non_blank = s.notna() & (s.str.strip() != "")
        changed = 0
        unmapped_samples = []

        for idx in df.index[non_blank]:
            raw = s.at[idx]
            key = _normalise_key(raw)
            if key in mapping:
                target = mapping[key]
                if str(raw).strip() != target:
                    df.at[idx, field] = target
                    changed += 1
            else:
                if len(unmapped_samples) < 5:
                    unmapped_samples.append(str(raw).strip())

        report["canonical_enum_normalization"]["fields"][field] = {
            "rows_considered": int(non_blank.sum()),
            "rows_changed": int(changed),
            "unmapped_examples": sorted(list(dict.fromkeys(unmapped_samples))),
        }

    return report


#: Explicit, ordered date formats tried BEFORE any locale/format inference.
#:
#: Precedence is unambiguous and identical for every column and every row, so a
#: value never depends on what happened to sit in row 0 (pandas infers ONE format
#: from the first non-null element and coerces everything else to NaT — that is
#: format inference, not a parsing rule, and it is why a normalised UK tape could
#: report ``date_parse_failed`` on perfectly valid dates).
#:
#: ISO comes first because ``2011-03-04`` is unambiguous. Day-first UK forms come
#: next because that is the UK source convention: ``01/11/1935`` is 1 November
#: 1935. Month-first forms are NOT in the ladder — adding them would make
#: ``01/11/1935`` ambiguous again.
_ISO_DATE_FORMATS: tuple = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%d",
)
_UK_DATE_FORMATS: tuple = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d %b %Y",
    "%d %B %Y",
)
#: Month-first ladder, used ONLY when a caller explicitly asks for day-last
#: parsing (``dayfirst=False``). Never mixed with the UK ladder.
_US_DATE_FORMATS: tuple = (
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%y",
)


def date_format_ladder(dayfirst: bool = True) -> tuple:
    """The ordered, explicit date formats applied before any inference."""
    return _ISO_DATE_FORMATS + (_UK_DATE_FORMATS if dayfirst else _US_DATE_FORMATS)


def to_iso_date(series: pd.Series, dayfirst: bool = True) -> pd.Series:
    """Deterministically parse a column to ISO ``YYYY-MM-DD``.

    Order of resolution (fixed, never data-dependent):

      1. blank / null renderings become NA (absent, not a parse failure);
      2. Excel serial numbers;
      3. the explicit format ladder (:func:`date_format_ladder`) — ISO first,
         then the UK day-first forms — each applied with an exact ``format=`` so
         ``01/11/1935`` is always 1 November 1935 and the supplied day ``01`` is
         preserved rather than reinterpreted;
      4. only whatever is still unparsed falls back to pandas' ``dayfirst``
         inference, so no previously-parsing format regresses.

    A value that survives all four is genuinely unparseable and stays NaT.
    """
    s = _void_blanks(_strip_nd(series))
    s_num = pd.to_numeric(s, errors="coerce")
    is_serial = s_num.notna() & (s_num > 25000)

    dt_out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    if bool(is_serial.any()):
        dt_out.loc[is_serial] = pd.to_datetime(
            s_num[is_serial], unit="D", origin="1899-12-30"
        )

    is_str = s.notna() & ~is_serial
    if bool(is_str.any()):
        s_str = s[is_str].astype(str).str.strip()
        parsed = pd.Series(pd.NaT, index=s_str.index, dtype="datetime64[ns]")
        for fmt in date_format_ladder(dayfirst):
            remaining = parsed.isna()
            if not bool(remaining.any()):
                break
            attempt = pd.to_datetime(
                s_str[remaining], format=fmt, errors="coerce", utc=False)
            parsed.loc[remaining] = attempt
        # Backstop only for what the explicit ladder could not place, so formats
        # that parsed before this change still parse.
        remaining = parsed.isna()
        if bool(remaining.any()):
            # Per-element dateutil fallback is intentional here (the ladder has
            # already had its say), so pandas' "could not infer format" warning is
            # expected noise rather than a signal.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed.loc[remaining] = pd.to_datetime(
                    s_str[remaining], dayfirst=dayfirst, errors="coerce", utc=False)
        dt_out.loc[is_str] = parsed

    return dt_out.dt.strftime("%Y-%m-%d")


def date_precision_report(original: pd.Series, parsed: pd.Series) -> Dict[str, Any]:
    """Describe the precision the SOURCE actually carried for a date column.

    Some providers know only month and year and set the day to ``01`` by
    convention. That is a real property of the source, not an error and not
    something to re-derive: when every parsed value lands on day 01 the column is
    recorded as month-precision with a source-supplied day. Purely observational —
    no value is changed and nothing is inferred beyond counting.
    """
    iso = pd.Series(parsed).dropna().astype(str)
    iso = iso[iso.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
    total = int(len(iso))
    if total == 0:
        return {"parsed_count": 0, "precision": "unknown", "day_01_count": 0}
    day_01 = int((iso.str.slice(8, 10) == "01").sum())
    return {
        "parsed_count": total,
        "day_01_count": day_01,
        "precision": "month" if day_01 == total else "day",
        "day_convention": ("source_supplied_01" if day_01 == total else ""),
    }


def to_percentage(series: pd.Series) -> pd.Series:
    """Parse a percentage column to the canonical PERCENTAGE-POINT scale.

    The repository's canonical convention for percentage-valued fields is
    percentage points, not a 0-1 fraction: Gate 2 derives
    ``current_loan_to_value = (balance / valuation) * 100``, and the MI/demo
    layers describe LTV and rate fields as "percentage points" throughout. So
    ``20.00%`` is 20.0 here — the ``%`` suffix is stripped and the magnitude is
    kept exactly as written. No rescaling is inferred from the magnitude of the
    value, because guessing that ``0.2`` "means" 20% is exactly the kind of
    silent reinterpretation this pipeline must not do.

    A value that is not a number after stripping ``%`` / thousands separators
    stays NA and is reported as a controlled numeric parse failure.
    """
    s = _void_blanks(_strip_nd(series))
    if s.dtype == object or str(s.dtype) == "string":
        cleaned = (
            s.astype("string")
            .str.strip()
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def to_decimal(series: pd.Series) -> pd.Series:
    s = _void_blanks(_strip_nd(series))
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
    s = _void_blanks(_strip_nd(series))
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
        elif fmt in {"percentage", "percent", "pct"}:
            out = to_percentage(df[col])
        elif fmt in {"decimal", "number", "float"}:
            out = to_decimal(df[col])
        elif fmt in {"integer", "int"}:
            out = to_integer(df[col])
        elif fmt in {"boolean", "bool", "y/n"}:
            out = to_bool_yn(df[col])
        elif fmt in {"currency_code", "ccy_code", "iso_currency_code", "currency", "ccy"} or col.endswith("_currency"):
            out = to_currency(df[col], synonym_map=currency_synonyms)
        else:
            out = _void_blanks(_strip_nd(df[col]))
            if out.dtype == object:
                out = out.astype("string").str.strip()

        df[col] = out

        # Metrics. An ABSENT value is not a parse failure: a CSV round-trip
        # renders a real null as "" (or the literal "nan"/"<NA>"), and counting
        # those as failures is what turned an empty second-borrower DOB into a
        # spurious date_parse_failed. Only a cell that carried an actual value the
        # deterministic rules could not place is a failure.
        nd_mask = original.astype(str).str.match(ND_PATTERN, na=False)
        blank_mask = _blank_token_mask(original)
        failures_mask = original.notna() & out.isna() & ~nd_mask & ~blank_mask

        sample = []
        if failures_mask.sum() > 0:
            try:
                sample = original[failures_mask].astype('string').dropna().drop_duplicates().head(5).tolist()
            except Exception:
                pass

        field_report = {
            "format": fmt or "string",
            "nulls_before": before_null,
            "nulls_after": int(df[col].isna().sum()),
            "nd_stripped": int(nd_mask.sum()),
            "blank_values": int(blank_mask.sum()),
            "parse_failures": int(failures_mask.sum()),
            "sample_failures": sample,
        }
        if fmt == "date":
            # Observational provenance: records that a source carried month-level
            # precision with a conventional day of 01 (see date_precision_report).
            field_report["date_precision"] = date_precision_report(original, out)
        report["fields"][col] = field_report
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


# NUTS / ITL region code shape: e.g. UKI32, UKJ14 (NUTS) or TLG31, TLM50 (UK ITL).
# Uppercase letters/digits, no spaces, 3-6 chars. Readable labels ("West
# Midlands", "Scotland") have spaces and/or are title/lower case and so do NOT
# match, which is exactly how we tell a code from a label.
_NUTS_CODE_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{1,4}$")
_GEO_NODATA = "ND1"
_DEFAULT_NUTS_YEAR = "2021"
_DEFAULT_NUTS_CSV = "uk_itl_master_lookup_v2.csv"


def _is_nuts_code(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    s = str(value).strip()
    if not s or s.upper().startswith("ND"):
        return False
    return bool(_NUTS_CODE_RE.match(s.upper())) and s.upper() == s.replace(" ", "")


def _blank_mask(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    return s.isna() | (s.str.strip() == "") | (s.str.strip() == "<NA>")


def normalize_geography(df: pd.DataFrame, pt: str, config: dict) -> dict:
    """Route geography to the correct fields:

    * readable region labels  -> collateral_geography (analytics display)
    * NUTS3 codes (from postcode lookup, or already-coded inputs)
        -> geographic_region_collateral, and geographic_region_obligor
           (ERM documented assumption: obligor residence == secured property)
    * classification YEAR (config-driven, default 2021) -> geographic_region_classification
    * regulatory NUTS fields left with no derivable code -> ND1 (never a label)

    Never writes a readable label or a region code into the classification year,
    and never fabricates a NUTS code from a label (only postcode lookup derives codes).
    """
    report: dict = {"rule_id": "NORMALIZE_GEOGRAPHY_V2", "actions": {}}
    geo_cfg = (config or {}).get("nuts_lookup", {}) or {}
    enr_cfg = ((config or {}).get("enrichment", {}) or {}).get("uk_nuts3", {}) or {}

    obligor_c, collat_c = "geographic_region_obligor", "geographic_region_collateral"
    class_c, disp_c = "geographic_region_classification", "collateral_geography"
    for c in (obligor_c, collat_c, class_c, disp_c):
        if c not in df.columns:
            df[c] = pd.NA

    # 1. Relocate readable labels out of regulatory NUTS fields into the display field.
    relocated = 0
    for reg_col in (collat_c, obligor_c):
        vals = df[reg_col].astype("string")
        is_label = vals.notna() & (vals.str.strip() != "") & ~vals.map(_is_nuts_code)
        if is_label.any():
            need_disp = is_label & _blank_mask(df[disp_c])
            df.loc[need_disp, disp_c] = df.loc[need_disp, reg_col]
            df.loc[is_label, reg_col] = pd.NA
            relocated += int(is_label.sum())
    if relocated:
        report["actions"]["labels_relocated_to_collateral_geography"] = relocated

    # 2. Derive NUTS3 codes from postcode(s) into collateral (property-based).
    postcode_cols = (geo_cfg.get("postcode_columns") or enr_cfg.get("postcode_columns")
                     or ["postcode", "property_postcode", "property_post_code"])
    if isinstance(postcode_cols, str):
        postcode_cols = [c.strip() for c in postcode_cols.split(",") if c.strip()]
    csv_name = geo_cfg.get("source_file") or enr_cfg.get("nuts_csv_path") or _DEFAULT_NUTS_CSV
    region_map = {}
    for cand in (Path(csv_name), Path("reference_data") / csv_name):
        if cand.exists():
            try:
                region_map = load_region_mapping(cand)
            except Exception:
                region_map = {}
            break
    src_col = next((c for c in postcode_cols if c in df.columns), None)
    derived = 0
    if region_map and src_col:
        keys = df[src_col].apply(lambda x: _extract_geo_key(x, strategy="uk_outcode"))
        codes = keys.map(region_map)
        fill = codes.notna() & (codes.astype("string").str.strip() != "") & _blank_mask(df[collat_c])
        if fill.any():
            df.loc[fill, collat_c] = codes[fill]
            derived += int(fill.sum())
    if derived:
        report["actions"]["collateral_codes_derived_from_postcode"] = derived

    # 3. ERM assumption: obligor region == collateral (secured property) region.
    is_erm = pt in {"equity_release", "erm", "rre"}
    if is_erm:
        copy = _blank_mask(df[obligor_c]) & ~_blank_mask(df[collat_c]) & df[collat_c].map(_is_nuts_code)
        if copy.any():
            df.loc[copy, obligor_c] = df.loc[copy, collat_c]
            report["actions"]["obligor_copied_from_collateral_erm"] = int(copy.sum())

    # 4. Regulatory NUTS fields still blank -> no-data ND1 (never a label).
    for reg_col in (obligor_c, collat_c):
        nd = _blank_mask(df[reg_col])
        if nd.any():
            df.loc[nd, reg_col] = _GEO_NODATA
            report["actions"][f"{reg_col}_set_nodata"] = int(nd.sum())

    # 4b. Preserve GRANULAR ITL3 in explicit canonical fields. These always
    #     retain the granular UK ITL3 code (for FCA/UK reporting + MI drilldown)
    #     even when a regime projection (e.g. ESMA Annex 2) later delivers GBZZZ.
    #     The regulatory geographic_region_* fields keep ITL3 too (backward
    #     compatible); regime-specific GBZZZ is applied at projection on a copy.
    obligor_itl3 = obligor_c + "_itl3"
    collat_itl3 = collat_c + "_itl3"
    df[obligor_itl3] = df[obligor_c]
    df[collat_itl3] = df[collat_c]
    report["actions"]["itl3_fields_populated"] = [obligor_itl3, collat_itl3]

    # 5. Classification = NUTS classification YEAR (config-driven), never a label/code.
    year = str(geo_cfg.get("classification_year")
               or enr_cfg.get("classification_year")
               or (config or {}).get("nuts_classification_year")
               or _DEFAULT_NUTS_YEAR).strip()
    df[class_c] = year if year else _GEO_NODATA
    if "geographic_region_classification_source" not in df.columns:
        df["geographic_region_classification_source"] = pd.NA
    df["geographic_region_classification_source"] = "configured_nuts_classification_year"
    report["actions"]["classification_year"] = year

    return report


# --- UPDATED: Accepts Config Object ---
#: A supplied percentage agrees with its derived reference within this relative
#: tolerance. Wide enough for rounding and for a valuation restated between
#: extracts; far narrower than the 100x it must never confuse.
_PCT_RECONCILE_RTOL = 0.02

#: Deriving LTV from an outstanding balance where no principal balance exists
#: populates RREC12/RREC16 for books that previously reported No Data — an
#: acquired tape carries no principal balance, so this is the whole acquired
#: book. That is a NEW REGULATORY DISCLOSURE, not a bug fix, so the capability
#: ships dark: built and tested, off until the disclosure is approved. With the
#: flag off the fallback is not consulted and those books behave exactly as
#: they do today.
_ACQUIRED_LTV_DISCLOSURE_FLAG = "GATE2_LTV_ACQUIRED_DISCLOSURE"


def acquired_ltv_disclosure_enabled(config: Optional[dict] = None) -> bool:
    """True when LTV may be derived from an outstanding balance. Default False."""
    declared = (config or {}).get(_ACQUIRED_LTV_DISCLOSURE_FLAG.lower())
    if declared is None:
        declared = os.environ.get(_ACQUIRED_LTV_DISCLOSURE_FLAG, "")
    return str(declared).strip().lower() in ("1", "true", "yes", "on", "enabled")


def percentage_source_unit(field: str, config: dict) -> Optional[str]:
    """The declared source unit for a percentage field, or None.

    Read from the alias declaration (``source_unit: percentage_points |
    fraction``). When declared it is authoritative; when absent the caller
    reconciles rather than guessing.
    """
    for key in ("aliases", "alias_map", "field_aliases"):
        block = (config or {}).get(key)
        if isinstance(block, dict):
            entry = block.get(field)
            if isinstance(entry, dict) and entry.get("source_unit"):
                return str(entry["source_unit"]).strip().lower()
    return None


def _first_present(df: pd.DataFrame, columns: tuple) -> Optional[str]:
    """First column that exists and carries at least one value."""
    for col in columns:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            return col
    return None


def _resolve_ltv(df: pd.DataFrame, ltv_col: str, bal_cols: tuple, val_col: str,
                 config: dict) -> Optional[Dict[str, Any]]:
    """Bring one LTV column onto the canonical percentage-point scale.

    In precedence order, and never by magnitude alone:

    1. a DECLARED source unit normalises the supplied value;
    2. otherwise the supplied value is RECONCILED against ``(balance /
       valuation) * 100`` and the scale that agrees is adopted;
    3. rows with no usable supplied value are DERIVED from balance and
       valuation;
    4. anything still unresolved is left exactly as supplied and reported, so
       validation sees it rather than the transform hiding it.

    A supplied value that reconciles is never overwritten — the derivation is a
    reference for agreeing the scale, not a replacement for the client's number.
    """
    # The first entry is the field's own balance; any later entry is a fallback
    # that creates a new disclosure, so it is consulted only when approved.
    usable = bal_cols if acquired_ltv_disclosure_enabled(config) else bal_cols[:1]
    bal_col = _first_present(df, usable)
    val_present = val_col in df.columns and pd.to_numeric(
        df[val_col], errors="coerce").notna().any()
    has_supplied = ltv_col in df.columns and pd.to_numeric(
        df.get(ltv_col), errors="coerce").notna().any()
    if bal_col is None and not has_supplied:
        return None
    if ltv_col not in df.columns:
        df[ltv_col] = pd.NA

    supplied = pd.to_numeric(df[ltv_col], errors="coerce")
    reference = None
    if bal_col is not None and val_present:
        b = pd.to_numeric(df[bal_col], errors="coerce")
        v = pd.to_numeric(df[val_col], errors="coerce")
        reference = (b / v.where(v > 0)) * 100.0

    report: Dict[str, Any] = {
        "rule_id": f"RESOLVE_{ltv_col.upper()}",
        "numerator": bal_col, "denominator": val_col if val_present else None,
        "normalised_rows": 0, "derived_rows": 0, "unresolved_rows": 0,
        "source_unit": None, "basis": None,
    }

    # 1. Declared source unit wins outright.
    declared = percentage_source_unit(ltv_col, config)
    if declared in ("fraction", "percent_fraction"):
        mask = supplied.notna()
        df.loc[mask, ltv_col] = supplied[mask] * 100.0
        report.update(source_unit=declared, basis="declared",
                      normalised_rows=int(mask.sum()))
    elif declared in ("percentage_points", "percent_points", "points"):
        report.update(source_unit=declared, basis="declared",
                      normalised_rows=int(supplied.notna().sum()))
    elif supplied.notna().any() and reference is not None:
        # 2. Reconcile: which reading of the supplied value matches the book?
        comparable = supplied.notna() & reference.notna()
        as_points = np.isclose(supplied[comparable], reference[comparable],
                               rtol=_PCT_RECONCILE_RTOL, atol=0.5)
        as_fraction = np.isclose(supplied[comparable] * 100.0, reference[comparable],
                                 rtol=_PCT_RECONCILE_RTOL, atol=0.5)
        points_hits, fraction_hits = int(as_points.sum()), int(as_fraction.sum())
        if fraction_hits > points_hits:
            mask = supplied.notna()
            df.loc[mask, ltv_col] = supplied[mask] * 100.0
            report.update(source_unit="fraction", basis="reconciled",
                          normalised_rows=int(mask.sum()))
        elif points_hits:
            report.update(source_unit="percentage_points", basis="reconciled",
                          normalised_rows=points_hits)
        else:
            # Neither scale agrees: the value is not a rescaling of this book.
            report.update(basis="unreconciled",
                          unresolved_rows=int(comparable.sum()))
    elif supplied.notna().any():
        # 3. No reference and no declaration — leave it, and say so.
        report.update(basis="undeclared_no_reference",
                      unresolved_rows=int(supplied.notna().sum()))

    # 4. Fill the gaps from balance and valuation.
    if reference is not None:
        gaps = pd.to_numeric(df[ltv_col], errors="coerce").isna() & reference.notna()
        if gaps.any():
            df.loc[gaps, ltv_col] = reference[gaps]
            report["derived_rows"] = int(gaps.sum())
            report["basis"] = report["basis"] or "derived"
    return report if (report["normalised_rows"] or report["derived_rows"]
                      or report["unresolved_rows"]) else None


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

    # 2. LTV — normalise the source, reconcile it, derive only what is absent.
    #
    # Current and original declare their OWN inputs. They used to share a loop
    # whose only asymmetry was the overwrite rule, which is how current LTV came
    # to be force-derived into percentage points while original LTV kept whatever
    # scale the source supplied — two conventions on one canonical field name,
    # visible in the Annex fixtures as RREC12 in points beside RREC16 as a
    # fraction.
    #
    # Numerator precedence is explicit per field. For current LTV a principal
    # balance is preferred and an outstanding balance is the fallback: an
    # acquired tape carries only the latter, so requiring principal silently
    # skipped LTV for the whole acquired book — and with it both LTV validators.
    for ltv_col, bal_cols, val_col in [
        ("current_loan_to_value",
         ("current_principal_balance", "current_outstanding_balance"),
         "current_valuation_amount"),
        ("original_loan_to_value",
         ("original_principal_balance",),
         "original_valuation_amount"),
    ]:
        outcome = _resolve_ltv(df, ltv_col, bal_cols, val_col, config)
        if outcome:
            deriv_report.setdefault("derived", {})[ltv_col] = outcome

    # 3. Geography normalisation (regulatory NUTS fields vs classification year)
    #    Correct semantics (ESMA Annex 2):
    #      RREL11 geographic_region_obligor     = NUTS3 region code (obligor)
    #      RREC6  geographic_region_collateral  = NUTS3 region code (collateral)
    #      RREL12 geographic_region_classification = NUTS classification YEAR
    #    Readable region labels ("West Midlands") belong in the analytics-only
    #    field collateral_geography, NEVER in the regulatory NUTS fields and
    #    NEVER in the classification year.
    geo_report = normalize_geography(df, pt, config)
    if geo_report:
        deriv_report.setdefault("derived", {})["geography"] = geo_report

    
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

    # 5. Governed canonical derivations this module does not own.
    #    Gate 2 keeps its own balance / LTV / geography rules; anything else the
    #    governed library declares is applied here BY NAME, so one canonical field
    #    never ends up with two conventions. Runs last because a rule computed
    #    "as at the cut-off date" needs the cut-off date to have been resolved —
    #    running it earlier would age every borrower against an unparsed value.
    #    Additive and null-preserving: a rule whose inputs are absent reports why
    #    and changes nothing.
    try:
        from engine.transformation_agent import canonical_derivations as _cd
        lib = _cd.apply_selected_derivations(df, _GATE2_LIBRARY_DERIVATIONS)
    except Exception as exc:  # noqa: BLE001 — additive layer, never fatal
        deriv_report.setdefault("skipped", {})["canonical_derivations"] = str(exc)
    else:
        for target, outcome in lib.items():
            bucket = "derived" if outcome.get("applied") else "skipped"
            deriv_report.setdefault(bucket, {})[target] = outcome

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

#: Governed derivations Gate 2 takes from the shared library rather than
#: reimplementing. Gate 2 owns LTV and the balance identities itself, so they are
#: deliberately not in this list.
_GATE2_LIBRARY_DERIVATIONS = ("youngest_borrower_age",)


def resolve_portfolio_defaults(config: dict, source_portfolio_id: str) -> Dict[str, Any]:
    """The fill-if-missing defaults configured for ONE source portfolio.

    A managed-service client is several books, and a fact that is true of one is
    not a fact about the client. The acquired book the seller delivered with no
    origination-channel column has one channel — itself — and saying so is a
    portfolio statement, not a client one, so it is configured per portfolio::

        portfolio_defaults:
          acquired_001:
            broker_channel: Acquired_001

    Adding ``acquired_002`` is a configuration line. Nothing here knows the name
    of any portfolio.
    """
    if not isinstance(config, dict) or not source_portfolio_id:
        return {}
    block = config.get("portfolio_defaults") or {}
    if not isinstance(block, dict):
        return {}
    wanted = str(source_portfolio_id).strip().lower()
    for pid, defaults in block.items():
        if str(pid).strip().lower() == wanted and isinstance(defaults, dict):
            return dict(defaults)
    return {}


def apply_portfolio_defaults(df: pd.DataFrame, config: dict,
                             source_portfolio_id: str) -> dict:
    """Fill blanks in this portfolio's configured default fields.

    Precedence is: a mapped source value, then the configured portfolio default,
    then nothing. A real value is never overwritten, and the report distinguishes
    a defaulted value from a sourced one so provenance survives — a broker channel
    that came from configuration must never read as one the lender stated.
    """
    defaults = resolve_portfolio_defaults(config, source_portfolio_id)
    applied: Dict[str, Any] = {}
    for field, value in defaults.items():
        if field not in df.columns:
            df[field] = pd.NA
        mask = df[field].isna() | (df[field].astype(str).str.strip().isin(("", "nan", "None")))
        filled = int(mask.sum())
        if filled:
            df.loc[mask, field] = value
        applied[field] = {
            "value": value,
            "rows_filled": filled,
            "rows_kept_source_value": int(len(df) - filled),
            "value_origin": "portfolio_default",
            "source_portfolio_id": source_portfolio_id,
        }
    return {"portfolio_defaults": applied} if applied else {}


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
    _provenance.add_cli_arguments(ap)
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

    # 5. Canonical enum normalization (internal standardization; not regime mapping)
    enum_norm_report = apply_canonical_enum_normalization(
        df,
        resolve_canonical_enum_normalization(config),
    )

    # 6. Defaults
    defaults_report = apply_config_defaults(df, config)

    # 6b. Portfolio-scoped defaults (fill-if-missing). Applied after the client
    #     defaults so the more specific statement wins, and keyed on the run's
    #     own source_portfolio_id so no other book can be touched by them.
    _prov_pid = getattr(args, "source_portfolio_id", "") or ""
    portfolio_defaults_report = apply_portfolio_defaults(df, config, _prov_pid)

    # 7. Source-portfolio provenance — stamp every row from run-level metadata.
    # Authoritative: overwrites any provenance columns so the canonical truth set
    # always carries a clean source-cohort tag. Optional here for back-compat;
    # the validation gate (PROV*) fails closed when provenance is absent.
    provenance_report: Dict[str, Any] = {}
    prov = _provenance.provenance_from_args(args, required=False)
    if prov is not None:
        _provenance.stamp_dataframe(df, prov)
        provenance_report = {
            "provenance": prov.to_dict(),
            "provenance_lineage": _provenance.lineage_entries(prov),
        }

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
        **enum_norm_report,
        **defaults_report,
        **portfolio_defaults_report,
        **provenance_report,
    }

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")

if __name__ == "__main__":
    main()
