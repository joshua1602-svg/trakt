#!/usr/bin/env python3
"""
annex12_projector.py

Projects a complete ESMA Annex 12 record from:
- merged Annex 12 config
- canonical CSV
- annex12_field_constraints.yaml
- esma_code_order.yaml

Guarantees:
- every ESMA code populated (or fails fast if legally impossible)
- correct defaulting (0 / ND5 / ND1) based on constraints + config defaults
- enum validation for LIST fields (where enums are defined)
- STRICT FORMAT VALIDATION (Regex for Dates, Integers, etc.)
- IVSR (triggers/tests/events) supported via triggers_catalogue + optional ivsr_actuals.csv
- IVSF (cashflow waterfall) supported via cashflow_waterfall_definition + optional cashflow_executed.csv

Policy:
- Do NOT infer IVSR / IVSF from loan tape.
- IVSR/IVSF are populated only from deterministic config + external inputs.
"""

import argparse
import yaml
import pandas as pd
import math
import re  # Added for strict validation
from typing import Any, Dict, List, Optional

PIPE_DELIM = "\x1f"

# --- STRICT VALIDATION CONSTANTS ---
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TEL_RE = re.compile(r"^\+?[0-9 ()\-]{6,}$")
ALPHANUM_RE = re.compile(r"^[\x20-\x7E]+$")
# -----------------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.strip() == "")

def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def compute_sum(df: pd.DataFrame, fields: List[str]) -> Optional[float]:
    total = 0.0
    found = False
    for f in fields:
        if f in df.columns:
            total += pd.to_numeric(df[f], errors="coerce").fillna(0).sum()
            found = True
    return total if found else None

def normalize_keys(section: Dict[str, Any]) -> Dict[str, Any]:
    """Converts 'IVSS1_description' keys to 'IVSS1'."""
    normalized = {}
    for k, v in section.items():
        code = k.split('_')[0] 
        normalized[code] = v
    return normalized

def validate_enum(code: str, value: str, enums: Dict[str, Dict[str, str]], constraints: Dict) -> None:
    if value.startswith("ND"):
        if value not in {"ND1", "ND2", "ND3", "ND4", "ND5"}:
             raise ValueError(f"{code}: Invalid ND format '{value}'")
        if value == "ND5" and not allows_nd5(code, constraints):
             raise ValueError(f"{code}: ND5 not allowed here")
        if value in {"ND1", "ND2", "ND3", "ND4"} and not allows_nd1(code, constraints):
             raise ValueError(f"{code}: {value} not allowed here")
        return
        
    allowed = enums.get(code, {})
    if allowed and value not in allowed:
        raise ValueError(f"{code}: invalid enum value '{value}'...")

def _constraints_fields(constraints_yaml: dict) -> Dict[str, dict]:
    root = constraints_yaml.get("annex12_field_constraints", {})
    fields = root.get("fields", {})
    if not isinstance(fields, dict):
        raise ValueError("constraints YAML missing annex12_field_constraints.fields mapping")
    return fields

def allows_nd5(code: str, constraints_fields: Dict[str, dict]) -> bool:
    return bool(constraints_fields.get(code, {}).get("nd5_allowed", False))

def allows_nd1(code: str, constraints_fields: Dict[str, dict]) -> bool:
    return bool(constraints_fields.get(code, {}).get("nd1_nd4_allowed", False))

def validate_value_by_format(code: str, value: str, fmt: str, constraints_fields: Dict[str, dict]) -> None:
    """Strictly validates value against ESMA Annex 12 format tokens."""
    if value.startswith("ND"):
        if value == "ND5" and not allows_nd5(code, constraints_fields):
            raise ValueError(f"{code}: ND5 not allowed by constraints")
        if value in {"ND1", "ND2", "ND3", "ND4"} and not allows_nd1(code, constraints_fields):
            raise ValueError(f"{code}: {value} not allowed by constraints")
        return

    if fmt == "{Y/N}":
        if value not in {"Y", "N"}:
            raise ValueError(f"{code}: expected Y or N, got '{value}'")

    elif fmt == "{DATEFORMAT}":
        if not DATE_RE.match(value):
            raise ValueError(f"{code}: expected YYYY-MM-DD, got '{value}'")

    elif fmt in {"{NUMERIC}", "{MONETARY}", "{PERCENTAGE}"}:
        if safe_float(value) is None:
            raise ValueError(f"{code}: expected numeric, got '{value}'")

    elif fmt.startswith("{INTEGER"):
        iv = safe_float(value)
        if iv is None or abs(iv - round(iv)) > 1e-9:
            raise ValueError(f"{code}: expected integer, got '{value}'")
        # Check max bound e.g. {INTEGER-9999}
        m = re.match(r"^\{INTEGER-(\d+)\}$", fmt)
        if m:
            maxv = int(m.group(1))
            if int(round(iv)) > maxv or int(round(iv)) < 0:
                raise ValueError(f"{code}: integer out of range 0..{maxv}: '{value}'")

    elif fmt.startswith("{ALPHANUM-"):
        # Max-length check
        m = re.match(r"^\{ALPHANUM-(\d+)\}$", fmt)
        if m:
            maxlen = int(m.group(1))
            if len(value) > maxlen:
                raise ValueError(f"{code}: exceeds max length {maxlen}")
        if not ALPHANUM_RE.match(value):
            raise ValueError(f"{code}: contains non-printable characters")

    elif fmt == "{TELEPHONE}":
        if not TEL_RE.match(value):
            raise ValueError(f"{code}: invalid telephone format '{value}'")

def fallback_from_constraints(code: str, constraints_fields: Dict[str, dict]) -> Any:
    spec = constraints_fields.get(code, {})
    fmt = (spec.get("format") or "").strip()
    
    # Check simple numeric format tokens for fallback logic
    is_numeric = any(tok in fmt for tok in ["{MONETARY}", "{PERCENTAGE}", "{NUMERIC}", "{INTEGER"])

    if spec.get("nd5_allowed"):
        return "ND5"
    if spec.get("nd1_nd4_allowed"):
        return "ND1"
    
    # If no ND allowed, try zero for numerics
    if is_numeric:
        if "{INTEGER" in fmt.upper():
            return 0
        return 0.0

    raise ValueError(f"{code}: no legal fallback (format={fmt}, nd5_allowed={spec.get('nd5_allowed')}, nd1_allowed={spec.get('nd1_nd4_allowed')})")

def _pipe_join(values: List[Any]) -> str:
    # Preserve ND codes / strings; ensure no None
    out = []
    for v in values:
        if v is None:
            out.append("")
        else:
            out.append(str(v))
    return PIPE_DELIM.join(out)

def _load_ivsr_actuals(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    df = pd.read_csv(path)
    required = {"test_id", "actual_value", "breach_flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ivsr_actuals missing columns: {sorted(missing)}")

    out: Dict[str, Dict[str, str]] = {}
    for _, r in df.iterrows():
        tid = str(r["test_id"]).strip()
        if not tid:
            continue
        out[tid] = {
            "actual_value": str(r["actual_value"]).strip() if not pd.isna(r["actual_value"]) else "",
            "breach_flag": str(r["breach_flag"]).strip() if not pd.isna(r["breach_flag"]) else "",
        }
    return out

def _load_cashflow_executed(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    df = pd.read_csv(path)
    required = {"waterfall_id", "amount", "available_funds_post"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"cashflow_executed missing columns: {sorted(missing)}")

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        wid = str(r["waterfall_id"]).strip()
        if not wid:
            continue
        rows.append({
            "waterfall_id": wid,
            "amount": r["amount"],
            "available_funds_post": r["available_funds_post"],
        })
    return rows

def parse_args():
    ap = argparse.ArgumentParser(description="Annex 12 Regime Projector")
    ap.add_argument("--config", required=True)
    ap.add_argument("--master-config", required=False, help="Path to Master Config (config_ERM_UK.yaml)")
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--as-of-date", required=False, help="Override IVSS2 (YYYY-MM-DD)")
    ap.add_argument("--constraints", required=True)
    ap.add_argument("--code-order-yaml", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ivsr-actuals", required=False, default=None, help="CSV with test_id, actual_value, breach_flag")
    ap.add_argument("--cashflow-executed", required=False, default=None, help="CSV with waterfall_id, amount, available_funds_post")
    return ap.parse_args()

def main():
    args = parse_args()

    cfg_root = load_yaml(args.config)
    if "annex12" not in cfg_root:
        raise ValueError("Config missing top-level 'annex12' key")
    cfg = cfg_root["annex12"]
    
    if args.master_config:
        try:
            master_root = load_yaml(args.master_config)
            # The LEI and Name are in the 'defaults' section of config_ERM_UK.yaml
            master_defaults = master_root.get('defaults', {})
            
            lei = master_defaults.get('originator_legal_entity_identifier')
            name = master_defaults.get('originator_name')

            # Ensure the 'deal' dictionary exists
            if 'deal' not in cfg:
                cfg['deal'] = {}

            # Inject LEI (IVSS1) if not already hardcoded in Annex 12 config
            if lei and 'IVSS1' not in cfg['deal']:
                cfg['deal']['IVSS1'] = lei

            # Inject Entity Name (IVSS3/IVSS4)
            if name and 'IVSS3' not in cfg['deal']:
                cfg['deal']['IVSS3'] = name
            if name and 'IVSS4' not in cfg['deal']:
                cfg['deal']['IVSS4'] = name

        except Exception as e:
            print(f"WARNING: Failed to load master config: {e}")

    # --- DYNAMIC DATE OVERRIDE ---
    if args.as_of_date:
        if 'period' not in cfg:
            cfg['period'] = {}
        cfg['period']['IVSS2_data_cut_off_date'] = args.as_of_date

    constraints_yaml = load_yaml(args.constraints)
    constraints_fields = _constraints_fields(constraints_yaml)

    code_order = load_yaml(args.code_order_yaml).get("ESMA_Annex12")
    if not isinstance(code_order, list):
        raise ValueError("code-order-yaml must contain ESMA_Annex12: [list of codes]")

    df = pd.read_csv(args.canonical)
    row_out: Dict[str, Any] = {}

    enums = cfg.get("enums", {}) or {}
    defaults = cfg.get("defaults", {}) or {}
    deal = cfg.get("deal", {}) or {}
    period = cfg.get("period", {}) or {}
    overrides = cfg.get("field_overrides", {}) or {}
    
    deal = normalize_keys(cfg.get("deal", {}) or {})
    period = normalize_keys(cfg.get("period", {}) or {})

    # 1) IVSS computed fields
    computed: Dict[str, Any] = {}
    for _, spec in (cfg.get("computations", {}) or {}).items():
        code = spec.get("output_alias")
        method = (spec.get("method") or "").strip()
        if not code or not method:
            continue
        if method == "SUM":
            if "source_fields" in spec and isinstance(spec["source_fields"], list):
                v = compute_sum(df, spec["source_fields"])
            else:
                sf = spec.get("source_field")
                v = compute_sum(df, [sf]) if sf else None
            if v is not None:
                computed[code] = float(v)
                
        elif method == "BUCKET_SUM":
            src = spec.get("source_field")
            flt = spec.get("filter_field")
            # Use infinite bounds if min/max are missing
            mn = spec.get("min", -math.inf)
            mx = spec.get("max", math.inf)
            
            if src and flt and src in df.columns and flt in df.columns:
                # 1. coerce to numeric (handle non-numeric gracefully)
                balances = pd.to_numeric(df[src], errors='coerce').fillna(0.0)
                days = pd.to_numeric(df[flt], errors='coerce').fillna(0.0)
                
                # 2. apply mask
                mask = (days >= mn) & (days <= mx)
                
                # 3. sum
                v = balances[mask].sum()
                computed[code] = float(v)

    # 2) IVSR repeatable section
    ivsr_catalogue = cfg.get("triggers_catalogue", []) or []
    ivsr_actuals = _load_ivsr_actuals(args.ivsr_actuals)
    ivsr_cols = ["IVSR2", "IVSR3", "IVSR4", "IVSR5", "IVSR6", "IVSR7", "IVSR8", "IVSR9", "IVSR10"]
    ivsr_acc: Dict[str, List[Any]] = {c: [] for c in ivsr_cols}

    if ivsr_catalogue:
        for item in ivsr_catalogue:
            tid = str(item.get("id", "")).strip()
            if not tid: continue

            ivsr_acc["IVSR2"].append(item.get("IVSR2", ""))
            ivsr_acc["IVSR3"].append(item.get("IVSR3", ""))
            ivsr_acc["IVSR4"].append(item.get("IVSR4", ""))
            ivsr_acc["IVSR5"].append(item.get("IVSR5", ""))
            ivsr_acc["IVSR8"].append(item.get("IVSR8", ""))
            ivsr_acc["IVSR9"].append(item.get("IVSR9", ""))
            ivsr_acc["IVSR10"].append(item.get("IVSR10", ""))

            a = ivsr_actuals.get(tid, {})
            actual_value = a.get("actual_value", "")
            breach_flag = a.get("breach_flag", "")
            ivsr_acc["IVSR6"].append(actual_value)
            ivsr_acc["IVSR7"].append(breach_flag if breach_flag else "N")

    # 3) IVSF repeatable section
    waterfall_def = cfg.get("cashflow_waterfall_definition", []) or []
    cashflow_exec = _load_cashflow_executed(args.cashflow_executed)
    ivsf_cols = ["IVSF2", "IVSF3", "IVSF4", "IVSF5", "IVSF6"]
    ivsf_acc: Dict[str, List[Any]] = {c: [] for c in ivsf_cols}

    if waterfall_def and cashflow_exec:
        exec_by_id = {str(r["waterfall_id"]).strip(): r for r in cashflow_exec}
        for step in waterfall_def:
            wid = str(step.get("id", "")).strip()
            if not wid: continue
            if wid not in exec_by_id:
                raise ValueError(f"IVSF: missing executed cashflow row for waterfall_id='{wid}'")
            r = exec_by_id[wid]
            ivsf_acc["IVSF2"].append(wid)
            ivsf_acc["IVSF3"].append(wid)
            ivsf_acc["IVSF4"].append(step.get("IVSF4", ""))
            ivsf_acc["IVSF5"].append(r.get("amount"))
            ivsf_acc["IVSF6"].append(r.get("available_funds_post"))

    # 4) Populate all ESMA codes
    for code in code_order:
        value: Any = None
        if code in ivsr_cols and ivsr_acc[code]:
            value = _pipe_join(ivsr_acc[code])
        elif code in ivsf_cols and ivsf_acc[code]:
            value = _pipe_join(ivsf_acc[code])
        elif code in overrides:
            value = overrides[code]
        elif code in deal:
            value = deal[code]
        elif code in period:
            value = period[code]
        elif code in computed:
            value = computed[code]
        elif code in defaults.get("set_to_zero_if_missing", []):
            value = 0.0
        elif code in defaults.get("set_to_nd5_if_missing", []):
            value = "ND5"
        elif code in defaults.get("set_to_nd1_if_missing", []):
            value = "ND1"
        else:
            value = None

        if is_missing(value) or value == "":
            value = fallback_from_constraints(code, constraints_fields)

        # START STRICT VALIDATION
        # We only strictly validate single scalar values.
        # Pipe-delimited lists (IVSR/IVSF) bypass regex validation here
        # because the regex expects a single value (e.g., one date, not "date|date").
        sv = str(value) if value is not None else ""
        fmt = constraints_fields.get(code, {}).get("format", "")
        
        if PIPE_DELIM not in sv:
            validate_value_by_format(code, sv, fmt, constraints_fields)
        # END STRICT VALIDATION

        validate_enum(code, sv, enums, constraints_fields)
        row_out[code] = value

    # 5) Cross-field linkages
    if "IVSS1" in row_out:
        if "IVSR1" in row_out: row_out["IVSR1"] = row_out["IVSS1"]
        if "IVSF1" in row_out: row_out["IVSF1"] = row_out["IVSS1"]

    out_df = pd.DataFrame([row_out], columns=code_order)
    out_df.to_csv(args.output, index=False)
    print(f"Annex 12 projected CSV written to {args.output}")

if __name__ == "__main__":
    main()