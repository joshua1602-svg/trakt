#!/usr/bin/env python3
"""
lineage_json.py

Generates:
- field_lineage.json (always)
- value_lineage.json (optional; only when tracing is requested)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_read_json(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_yaml(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def safe_read_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path or not path.exists():
        return None
    return pd.read_csv(path)


def infer_id_column(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
    if explicit and explicit in df.columns:
        return explicit
    for c in ["loan_identifier", "loan_id", "LOAN_ID", "LoanID", "LI"]:
        if c in df.columns:
            return c
    return None


def extract_registry_field_specs(registry: dict) -> Dict[str, dict]:
    """
    Normalize different registry shapes into:
      { field_name: meta_dict }

    Supported shapes:
      1) {"fields": {"loan_identifier": {...}, ...}}
      2) {"loan_identifier": {...}, "current_principal_balance": {...}, ...}  (flat)
    """
    if not isinstance(registry, dict) or not registry:
        return {}

    # shape 1
    fields = registry.get("fields")
    if isinstance(fields, dict):
        # keep insertion order as provided in YAML
        return fields

    # shape 2 (flat dict)
    if all(isinstance(v, dict) for v in registry.values()):
        return registry

    return {}


def select_contract_fields(field_specs: Dict[str, dict], portfolio_type: str) -> Dict[str, Any]:
    """
    Compute contract schema fields from registry metadata.

    Rules:
    - Include fields where portfolio_type == 'common' OR matches requested portfolio_type
    - If a field has no portfolio_type metadata, include it (conservative)
    - Core fields are those with meta.core_canonical == True OR meta.core == True
    """
    pt = (portfolio_type or "").strip().lower()
    contract: List[str] = []
    core: List[str] = []

    for fname, meta in (field_specs or {}).items():
        if not isinstance(meta, dict):
            continue

        fpt = str(meta.get("portfolio_type", "")).strip().lower()
        include = False

        if fpt in ("", "none", "null"):
            include = True
        elif fpt == "common" or fpt == pt:
            include = True

        if not include:
            continue

        contract.append(str(fname))

        is_core = bool(meta.get("core_canonical") is True or meta.get("core") is True)
        if is_core:
            core.append(str(fname))

    return {"contract_fields": contract, "core_fields": core}


def extract_header_map(header_map: dict) -> Dict[str, dict]:
    """
    Normalize header mapping report to:
      { canonical_field: { "raw_columns": [...], "methods": [...] } }
    """
    if not isinstance(header_map, dict):
        return {}

    # shape A: {"mappings": [{"raw_header": "...", "canonical_field": "...", "mapping_method": "...", "confidence": ...}, ...]}
    mappings = header_map.get("mappings")
    if isinstance(mappings, list):
        out: Dict[str, dict] = {}
        for m in mappings:
            if not isinstance(m, dict):
                continue

            # Your messy_to_canonical emits these keys:
            raw = m.get("raw_header")
            can = m.get("canonical_field")
            method = m.get("mapping_method")
            conf = m.get("confidence")

            if not raw or not can:
                continue

            can = str(can)
            out.setdefault(can, {"raw_columns": [], "methods": []})
            out[can]["raw_columns"].append(str(raw))
            out[can]["methods"].append(
                {
                    "raw": str(raw),
                    "method": method,
                    "confidence": conf,
                }
            )
        return out

    # shape B: {"raw_to_canonical": {"A":"loan_identifier"}}
    rtc = header_map.get("raw_to_canonical")
    if isinstance(rtc, dict):
        out: Dict[str, dict] = {}
        for raw, can in rtc.items():
            if not raw or not can:
                continue
            can = str(can)
            out.setdefault(can, {"raw_columns": [], "methods": []})
            out[can]["raw_columns"].append(str(raw))
        return out

    return {}


def build_quality_lookup(agg_validation: Optional[pd.DataFrame]) -> Dict[str, dict]:
    if agg_validation is None or agg_validation.empty:
        return {}

    cols = {c.lower(): c for c in agg_validation.columns}
    field_col = cols.get("field") or cols.get("canonical_field") or agg_validation.columns[0]
    err_col = cols.get("error_count") or cols.get("n_errors") or cols.get("errors")
    rate_col = cols.get("error_rate") or cols.get("rate")
    mat_col = cols.get("materiality") or cols.get("severity")

    out: Dict[str, dict] = {}
    for _, r in agg_validation.iterrows():
        f = str(r[field_col])
        rec: Dict[str, Any] = {}

        if err_col and err_col in agg_validation.columns:
            v = pd.to_numeric(r[err_col], errors="coerce")
            rec["error_count"] = int(0 if pd.isna(v) else v)

        if rate_col and rate_col in agg_validation.columns:
            v = pd.to_numeric(r[rate_col], errors="coerce")
            rec["error_rate"] = None if pd.isna(v) else float(v)

        if mat_col and mat_col in agg_validation.columns:
            rec["materiality"] = None if pd.isna(r[mat_col]) else str(r[mat_col])

        out[f] = rec

    return out


def extract_enum_unmapped(enum_report: dict) -> Dict[str, List[str]]:
    uv = enum_report.get("unmapped_values")
    if isinstance(uv, dict):
        out = {}
        for k, v in uv.items():
            if isinstance(v, list):
                out[str(k)] = sorted(list({str(x) for x in v}))
        return out
    return {}


def extract_nd_counts(nd_report: dict) -> Dict[str, int]:
    nda = nd_report.get("nd_defaults_applied") or nd_report.get("applied") or nd_report.get("fields")
    if isinstance(nda, dict):
        out = {}
        for k, v in nda.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
        return out
    return {}


def extract_transform_context(transform_report: Optional[dict]) -> Dict[str, dict]:
    if not transform_report or not isinstance(transform_report, dict):
        return {}
    fields = transform_report.get("fields")
    if not isinstance(fields, dict):
        return {}

    out: Dict[str, dict] = {}
    for f, meta in fields.items():
        if not isinstance(meta, dict):
            continue
        out[str(f)] = {
            "nulls_before": meta.get("nulls_before"),
            "nulls_after": meta.get("nulls_after"),
            "nd_stripped": meta.get("nd_stripped"),
            "parse_failures": meta.get("parse_failures"),
            "sample_failures": meta.get("sample_failures"),
        }
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True, help="Canonical typed CSV")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--registry", default=None, help="Field registry YAML (recommended)")
    ap.add_argument("--portfolio-type", default="equity_release")
    ap.add_argument("--id-col", default=None)

    ap.add_argument("--header-map", default=None)
    ap.add_argument("--enum-report", default=None)
    ap.add_argument("--nd-report", default=None)
    ap.add_argument("--transform-report", default=None)
    ap.add_argument("--agg-validation", default=None)

    ap.add_argument("--trace-loan-id", action="append", default=[])
    ap.add_argument("--trace-field", action="append", default=[])
    ap.add_argument("--max-trace-rows", type=int, default=2000)

    # Optional: print a short banner (useful in orchestrator logs)
    ap.add_argument("--quiet", action="store_true", default=False)

    return ap.parse_args()


def main() -> int:
    ns = parse_args()
    outdir = Path(ns.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ns.canonical)
    id_col = infer_id_column(df, ns.id_col)

    registry = safe_read_yaml(Path(ns.registry)) if ns.registry else {}
    field_specs = extract_registry_field_specs(registry)

    header_map = extract_header_map(safe_read_json(Path(ns.header_map)) if ns.header_map else {})
    enum_report = safe_read_json(Path(ns.enum_report)) if ns.enum_report else {}
    nd_report = safe_read_json(Path(ns.nd_report)) if ns.nd_report else {}

    agg_df = safe_read_csv(Path(ns.agg_validation)) if ns.agg_validation else None
    quality_lookup = build_quality_lookup(agg_df)

    enum_unmapped = extract_enum_unmapped(enum_report)
    nd_counts = extract_nd_counts(nd_report)
    transform_ctx = extract_transform_context(safe_read_json(Path(ns.transform_report)) if ns.transform_report else None)

    present_fields = list(df.columns)

    contract_fields: List[str] = []
    core_fields: List[str] = []
    if field_specs:
        contract_info = select_contract_fields(field_specs, ns.portfolio_type)
        contract_fields = contract_info.get("contract_fields") or []
        core_fields = contract_info.get("core_fields") or []

    fields = contract_fields if contract_fields else sorted(present_fields)

    present_set = set(present_fields)
    contract_set = set(contract_fields) if contract_fields else set()
    contract_present = sorted(list(present_set & contract_set)) if contract_fields else sorted(present_fields)
    missing_contract = sorted(list(contract_set - present_set)) if contract_fields else []
    unexpected_fields = sorted(list(present_set - contract_set)) if contract_fields else []

    field_lineage: Dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "inputs": {
            "canonical_csv": str(ns.canonical),
            "portfolio_type": ns.portfolio_type,
            "registry_yaml": str(ns.registry) if ns.registry else None,
            "header_map_json": str(ns.header_map) if ns.header_map else None,
            "enum_report_json": str(ns.enum_report) if ns.enum_report else None,
            "nd_report_json": str(ns.nd_report) if ns.nd_report else None,
            "agg_validation_csv": str(ns.agg_validation) if ns.agg_validation else None,
            "transform_report_json": str(ns.transform_report) if ns.transform_report else None,
        },
        "id_column": id_col,
        "run_summary": {
            "observed_fields_count": int(len(present_fields)),
            "contract_fields_count": int(len(contract_fields)) if contract_fields else None,
            "core_fields_count": int(len(core_fields)) if core_fields else None,
            "contract_fields_present_count": int(len(contract_present)) if contract_fields else None,
            "contract_fields_present_pct": (
                float(len(contract_present)) / float(len(contract_fields)) if contract_fields else None
            ),
            "core_fields_present_count": (int(len(set(core_fields) & present_set)) if core_fields else None),
            "core_fields_present_pct": (
                float(len(set(core_fields) & present_set)) / float(len(core_fields)) if core_fields else None
            ),
            "missing_contract_fields": missing_contract,
            "unexpected_fields": unexpected_fields,
        },
        "fields": {},
    }

    for f in fields:
        spec = field_specs.get(f, {}) if isinstance(field_specs, dict) else {}
        present_in_run = f in df.columns

        lineage: Dict[str, Any] = {
            "field": f,
            "declared_spec": {
                "core": spec.get("core_canonical", spec.get("core")),
                "format": spec.get("format") or spec.get("type") or spec.get("dtype"),
                "portfolio_type": spec.get("portfolio_type"),
                "layer": spec.get("layer"),
            }
            if spec
            else None,
            "field_status": {
                "present_in_run": bool(present_in_run),
                "expected_by_contract": bool(f in contract_set) if contract_fields else None,
                "reason": (None if present_in_run else "active_schema_field_not_provided"),
            },
            "source": header_map.get(f) if header_map else None,
            "quality": (quality_lookup.get(f) if present_in_run else None),
            "transform_context": {
                "enum_unmapped_values": enum_unmapped.get(f),
                "nd_defaults_applied_count": nd_counts.get(f),
                "canonical_transform": (transform_ctx.get(f) if (transform_ctx and present_in_run) else None),
            },
        }

        field_lineage["fields"][f] = lineage

    field_path = outdir / "field_lineage.json"
    field_path.write_text(json.dumps(field_lineage, indent=2, sort_keys=True), encoding="utf-8")

    # Optional value-level trace
    trace_ids: List[str] = [str(x) for x in (ns.trace_loan_id or [])]
    trace_fields: List[str] = [str(x) for x in (ns.trace_field or []) if str(x) in df.columns]

    value_path: Optional[Path] = None
    if trace_ids and id_col:
        traced = df[df[id_col].astype(str).isin(set(trace_ids))].copy().head(ns.max_trace_rows)
    else:
        traced = pd.DataFrame()

    if not traced.empty and trace_fields:
        records = []
        for _, row in traced.iterrows():
            rec = {"loan_id": str(row[id_col])}
            for f in trace_fields:
                rec[f] = None if pd.isna(row[f]) else row[f]
            records.append(rec)

        value_lineage = {
            "generated_at_utc": utc_now_iso(),
            "note": "Value lineage is targeted tracing. Full cell-by-cell lineage is intentionally not persisted.",
            "id_column": id_col,
            "trace_ids": trace_ids,
            "trace_fields": trace_fields,
            "rows": records,
        }

        value_path = outdir / "value_lineage.json"
        value_path.write_text(json.dumps(value_lineage, indent=2, sort_keys=True), encoding="utf-8")

    if not ns.quiet:
        # ASCII-only, Windows cp1252 safe
        msg = (
            f"[Lineage] Wrote: {field_path}"
            + (f" | {value_path}" if value_path else "")
            + f" | observed_fields={len(present_fields)}"
            + (f" | contract_fields={len(contract_fields)} | core_fields={len(core_fields)}" if contract_fields else "")
        )
        print(msg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


