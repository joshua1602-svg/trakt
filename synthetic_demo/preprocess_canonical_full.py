#!/usr/bin/env python3
"""
Pre-process canonical_full.csv between Gate 1 and Gate 2 to work around
pandas 3.0 StringDtype incompatibilities in canonical_transform.py.

Gate 2's to_decimal() and to_bool_yn() use `series.dtype == object` guards
that evaluate False for pandas 3.0 StringDtype columns, causing messy values
to fall through unprocessed. This script normalises the intermediate CSV so
Gate 2's type functions receive values in formats they can handle.
"""

import csv
import re
import yaml
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CANONICAL_FULL = ROOT / "synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_canonical_full.csv"
CFG = ROOT / "synthetic_demo/config/config_client_SYNTHETIC_ERM.yaml"
REGISTRY = ROOT / "config/system/fields_registry.yaml"

# ── Load config defaults ──────────────────────────────────────────────────────
cfg = yaml.safe_load(CFG.read_text())
defaults = cfg.get("defaults", {})

# ── Determine Y/N fields from registry ───────────────────────────────────────
registry = yaml.safe_load(REGISTRY.read_text())
fields_meta = registry.get("fields", {})
yn_fields = [f for f, v in fields_meta.items()
             if str(v.get("format", "")).upper() in ("Y/N", "YES/NO", "BOOL", "BOOLEAN")]

# ── Determine decimal/numeric fields from registry ───────────────────────────
decimal_formats = {"decimal", "float", "number", "numeric", "integer", "int"}
decimal_fields = [f for f, v in fields_meta.items()
                  if str(v.get("format", "")).lower() in decimal_formats]

# Add common numeric fields not always tagged in registry
extra_numeric = [
    "current_principal_balance", "original_principal_balance",
    "current_interest_rate", "current_loan_to_value", "original_loan_to_value",
    "current_valuation_amount", "youngest_borrower_age",
    "arrears_balance", "default_amount", "allocated_losses",
    "cumulative_recoveries", "deposit_amount", "purchase_price",
    "total_credit_limit", "number_of_days_in_arrears",
    "current_interest_rate_margin", "interest_rate_cap", "interest_rate_floor",
    "payment_due", "current_outstanding_balance", "accrued_interest",
]
decimal_fields = list(set(decimal_fields + extra_numeric))

# String default fields that create float64 columns when absent
string_default_fields = [f for f, v in defaults.items()
                         if isinstance(v, str)
                         and v not in ("Y", "N", "y", "n")
                         and not str(v).replace(".", "").isdigit()]


def clean_numeric(val):
    """Strip currency symbols, percent signs, spaces, commas; return plain decimal string."""
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return ""
    cleaned = re.sub(r"[^\d\-\.]", "", s)
    return cleaned if cleaned else ""


def yn_to_int(val):
    """Map Y/N/True/False → 1/0 integer string for to_bool_yn numeric branch."""
    s = str(val).strip().upper()
    if s in ("Y", "YES", "TRUE", "1"):
        return "1"
    if s in ("N", "NO", "FALSE", "0"):
        return "0"
    return ""


# ── Read + transform ─────────────────────────────────────────────────────────
df = pd.read_csv(CANONICAL_FULL, dtype=str, keep_default_na=False)

# 1. Clean numeric columns
for col in decimal_fields:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# 2. Convert Y/N columns to 1/0 integers
for col in yn_fields:
    if col in df.columns:
        df[col] = df[col].apply(yn_to_int)

# 3. Pre-populate string default columns so Gate 2 doesn't create float64 columns
for field in string_default_fields:
    val = str(defaults[field])
    if field not in df.columns:
        df[field] = val
    else:
        mask = df[field].isin(["", "nan", "None"])
        df.loc[mask, field] = val

# ── Write back ────────────────────────────────────────────────────────────────
df.to_csv(CANONICAL_FULL, index=False, quoting=csv.QUOTE_MINIMAL)
print(f"Pre-processed {CANONICAL_FULL.name}: {len(df)} rows, {len(df.columns)} columns")
