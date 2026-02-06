#!/usr/bin/env python3
"""
delta_manifest.py

Builds a run_manifest.json for a run and (optionally) a run_delta.json vs a prior run.

Works with:
- canonical CSV (required)
- aggregated validation CSV (optional)
- enum mapping report JSON (optional)
- ND defaults report JSON (optional)
- header mapping report JSON (optional)

Typical usage:
  python delta_manifest.py \
    --current-canonical out/current/canonical_typed.csv \
    --prior-manifest out/prior/run_manifest.json \
    --current-agg-validation out/current/validation_aggregated.csv \
    --outdir out/current

Or build prior manifest from prior canonical:
  python delta_manifest.py \
    --current-canonical out/current/canonical_typed.csv \
    --prior-canonical out/prior/canonical_typed.csv \
    --outdir out/current
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ----------------- helpers -----------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def safe_read_json(path: Optional[Path]) -> Optional[dict]:
    if not path:
        return None
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def safe_read_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if not path.exists():
        return None
    return pd.read_csv(path)


def safe_read_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        if not path.exists():
            return {}
        import yaml  # local import (optional dependency)
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def select_contract_fields_from_registry(registry: Dict[str, Any], portfolio_type: str) -> Dict[str, Any]:
    """Return contract_fields/core_fields using registry metadata (common + portfolio_type)."""
    if not isinstance(registry, dict):
        return {"contract_fields": [], "core_fields": []}
    fields_meta = registry.get("fields") if isinstance(registry.get("fields"), dict) else registry
    if not isinstance(fields_meta, dict):
        return {"contract_fields": [], "core_fields": []}

    pt = (portfolio_type or "").strip().lower()
    contract: List[str] = []
    core: List[str] = []
    for fname, meta in fields_meta.items():
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
        if meta.get("core_canonical") is True or meta.get("core") is True:
            core.append(str(fname))
    return {"contract_fields": contract, "core_fields": core}

def pct(a: float, b: float) -> Optional[float]:
    # percent change from b -> a (prior -> current)
    if b == 0:
        return None
    return (a - b) / b

def top_n_abs_changes(changes: Dict[str, float], n: int = 10) -> List[Tuple[str, float]]:
    return sorted(changes.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]


# ----------------- manifest builders -----------------

def infer_id_column(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
    if explicit and explicit in df.columns:
        return explicit
    candidates = ["loan_identifier", "loan_id", "LOAN_ID", "LoanID", "LI"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def infer_balance_column(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
    if explicit and explicit in df.columns:
        return explicit
    candidates = [
        "current_principal_balance",
        "current_balance",
        "balance",
        "outstanding_balance",
        "CPB",
        "CB",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def numeric_sum_series(s: pd.Series) -> float:
    # coerce numeric; ignore non-numeric
    x = pd.to_numeric(s, errors="coerce")
    return float(x.fillna(0).sum())

def completeness_by_column(df: pd.DataFrame) -> Dict[str, float]:
    # non-null percentage per column
    out: Dict[str, float] = {}
    n = len(df)
    if n == 0:
        return {c: 0.0 for c in df.columns}
    for c in df.columns:
        out[c] = float(df[c].notna().mean())
    return out

def build_quality_stats_from_agg_validation(agg: pd.DataFrame) -> Dict[str, Any]:
    """
    Expected (flexible) columns:
      - field (or canonical_field / Field)
      - error_count (or n_errors)
      - error_rate (optional)
      - materiality (optional)
    """
    if agg is None or agg.empty:
        return {}

    cols = {c.lower(): c for c in agg.columns}
    field_col = cols.get("field") or cols.get("canonical_field") or cols.get("fld") or cols.get("name")
    if not field_col:
        # fallback: first column
        field_col = agg.columns[0]

    err_col = cols.get("error_count") or cols.get("n_errors") or cols.get("errors") or None
    rate_col = cols.get("error_rate") or cols.get("rate") or None
    mat_col = cols.get("materiality") or cols.get("severity") or None

    records = []
    for _, r in agg.iterrows():
        rec = {"field": str(r[field_col])}
        if err_col and err_col in agg.columns:
            rec["error_count"] = int(pd.to_numeric(r[err_col], errors="coerce") or 0)
        if rate_col and rate_col in agg.columns:
            v = pd.to_numeric(r[rate_col], errors="coerce")
            rec["error_rate"] = None if pd.isna(v) else float(v)
        if mat_col and mat_col in agg.columns:
            rec["materiality"] = None if pd.isna(r[mat_col]) else str(r[mat_col])
        records.append(rec)

    # field keyed lookup for delta
    by_field = {rec["field"]: rec for rec in records}

    # overall summary
    total_errors = sum(rec.get("error_count", 0) for rec in records)
    materiality_counts: Dict[str, int] = {}
    for rec in records:
        m = rec.get("materiality")
        if m:
            materiality_counts[m] = materiality_counts.get(m, 0) + 1

    return {
        "total_errors": total_errors,
        "materiality_counts": materiality_counts,
        "fields": by_field,
    }

def build_manifest(
    canonical_csv: Path,
    run_id: str,
    id_col: Optional[str],
    balance_col: Optional[str],
    agg_validation_csv: Optional[Path],
    enum_report_json: Optional[Path],
    nd_report_json: Optional[Path],
    header_map_json: Optional[Path],
    config_paths: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    df = pd.read_csv(canonical_csv)
    inferred_id = infer_id_column(df, id_col)
    inferred_bal = infer_balance_column(df, balance_col)

    registry = safe_read_yaml(registry_yaml) if registry_yaml else {}
    contract_info = select_contract_fields_from_registry(registry, portfolio_type) if registry else {"contract_fields": [], "core_fields": []}
    contract_fields = contract_info.get("contract_fields") or []
    core_fields = contract_info.get("core_fields") or []


    loan_ids: Optional[List[str]] = None
    null_id_rate: Optional[float] = None
    if inferred_id:
        null_id_rate = float(df[inferred_id].isna().mean())
        # do not dump full IDs (can be huge); keep a hash + counts
        ids = df[inferred_id].dropna().astype(str).unique().tolist()
        # stable signature
        h = hashlib.sha256()
        for x in sorted(ids):
            h.update(x.encode("utf-8"))
            h.update(b"\n")
        loan_ids_signature = h.hexdigest()
    else:
        loan_ids_signature = None

    if inferred_bal:
        total_balance = numeric_sum_series(df[inferred_bal])
    else:
        total_balance = None

    comp = completeness_by_column(df)

    agg_df = safe_read_csv(agg_validation_csv)
    quality_stats = build_quality_stats_from_agg_validation(agg_df) if agg_df is not None else {}

    enum_report = safe_read_json(enum_report_json) or {}
    nd_report = safe_read_json(nd_report_json) or {}
    header_map = safe_read_json(header_map_json) or {}

    input_hash = sha256_file(canonical_csv)

    config_hashes = []
    if config_paths:
        for p in config_paths:
            if p.exists() and p.is_file():
                config_hashes.append({"path": str(p), "sha256": sha256_file(p)})

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": utc_now_iso(),
        "inputs": {
            "canonical_csv": str(canonical_csv),
            "canonical_sha256": input_hash,
        },
        "config_hashes": config_hashes,
        "portfolio_stats": {
            "loan_count": int(len(df)),
            "id_column": inferred_id,
            "null_id_rate": null_id_rate,
            "loan_ids_signature": loan_ids_signature,
            "balance_column": inferred_bal,
            "total_balance": total_balance,
        },
        "contract_schema": {
            "registry_yaml": str(registry_yaml) if registry_yaml else None,
            "portfolio_type": portfolio_type,
            "contract_fields": contract_fields,
            "core_fields": core_fields,
        },
        "schema_stats": {
            "column_count": int(len(df.columns)),
            "columns": list(df.columns),
            "completeness": comp,  # 0-1
        },
        "quality_stats": quality_stats,
        "transform_stats": {
            "header_mapping_report": header_map if header_map else None,
            "enum_mapping_report": enum_report if enum_report else None,
            "nd_defaults_report": nd_report if nd_report else None,
        },
    }
    return manifest


# ----------------- delta computation -----------------

def compute_schema_delta(curr: Dict[str, Any], prior: Dict[str, Any]) -> Dict[str, Any]:
    """
    Works across both canonical output modes:
    - full: observed columns are stable; differences typically reflect true contract/config changes
    - active: observed columns vary by tape coverage; differences are treated as run footprint variance
    """
    curr_cols = set((curr.get("schema_stats") or {}).get("columns") or [])
    prior_cols = set((prior.get("schema_stats") or {}).get("columns") or [])
    run_added = sorted(list(curr_cols - prior_cols))
    run_missing = sorted(list(prior_cols - curr_cols))

    # Contract schema (optional; only if --registry provided)
    c_contract = set((((curr.get("contract_schema") or {}).get("contract_fields")) or []))
    p_contract = set((((prior.get("contract_schema") or {}).get("contract_fields")) or []))

    contract_added = sorted(list(c_contract - p_contract)) if (c_contract or p_contract) else []
    contract_removed = sorted(list(p_contract - c_contract)) if (c_contract or p_contract) else []

    def _coverage(present: set, contract: set):
        if not contract:
            return None
        return float(len(present & contract)) / float(len(contract)) if len(contract) else None

    curr_cov = _coverage(curr_cols, c_contract)
    prior_cov = _coverage(prior_cols, p_contract)

    missing_contract_fields = sorted(list(c_contract - curr_cols)) if c_contract else []
    unexpected_fields = sorted(list(curr_cols - c_contract)) if c_contract else []

    ccomp = (curr.get("schema_stats") or {}).get("completeness", {}) or {}
    pcomp = (prior.get("schema_stats") or {}).get("completeness", {}) or {}
    comp_changes: Dict[str, float] = {}
    for col in sorted(list(curr_cols & prior_cols)):
        comp_changes[col] = float(ccomp.get(col, 0.0)) - float(pcomp.get(col, 0.0))
    top_moves = top_n_abs_changes(comp_changes, n=10)

    return {
        "contract_schema_delta": {
            "new_contract_fields": contract_added if (c_contract or p_contract) else None,
            "removed_contract_fields": contract_removed if (c_contract or p_contract) else None,
        },
        "run_footprint_delta": {
            "fields_added_in_run": run_added,
            "fields_missing_in_run": run_missing,
            "missing_contract_fields_in_run": missing_contract_fields if c_contract else None,
            "unexpected_fields_in_run": unexpected_fields if c_contract else None,
            "contract_fields_present_pct": {
                "current": curr_cov,
                "prior": prior_cov,
                "delta": (curr_cov - prior_cov) if (curr_cov is not None and prior_cov is not None) else None,
            },
        },
        "completeness_top_moves": [{"column": c, "delta": d} for c, d in top_moves],
    }

def compute_portfolio_delta(curr: Dict[str, Any], prior: Dict[str, Any]) -> Dict[str, Any]:
    cstats = curr["portfolio_stats"]
    pstats = prior["portfolio_stats"]

    lc = float(cstats.get("loan_count") or 0)
    lp = float(pstats.get("loan_count") or 0)

    bc = cstats.get("total_balance")
    bp = pstats.get("total_balance")
    bal_delta = None
    bal_pct = None
    if bc is not None and bp is not None:
        bal_delta = float(bc) - float(bp)
        bal_pct = pct(float(bc), float(bp))

    return {
        "loan_count": {"current": int(lc), "prior": int(lp), "delta": int(lc - lp), "pct": pct(lc, lp)},
        "total_balance": {"current": bc, "prior": bp, "delta": bal_delta, "pct": bal_pct},
    }

def compute_quality_delta(curr: Dict[str, Any], prior: Dict[str, Any]) -> Dict[str, Any]:
    cq = curr.get("quality_stats") or {}
    pq = prior.get("quality_stats") or {}

    cfields = (cq.get("fields") or {})
    pfields = (pq.get("fields") or {})

    # error_count delta per field
    err_changes: Dict[str, float] = {}
    mat_changes: List[Dict[str, Any]] = []
    new_blocking: List[str] = []

    def norm_mat(x: Any) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip()
        return s if s else None

    for f in set(cfields.keys()) | set(pfields.keys()):
        ce = float(cfields.get(f, {}).get("error_count") or 0)
        pe = float(pfields.get(f, {}).get("error_count") or 0)
        if ce != pe:
            err_changes[f] = ce - pe

        cm = norm_mat(cfields.get(f, {}).get("materiality"))
        pm = norm_mat(pfields.get(f, {}).get("materiality"))
        if cm != pm and (cm is not None or pm is not None):
            mat_changes.append({"field": f, "prior": pm, "current": cm})

        # heuristic: treat "BLOCK" substring as blocking
        if cm and "BLOCK" in cm.upper() and (not pm or "BLOCK" not in pm.upper()):
            new_blocking.append(f)

    top_err_moves = top_n_abs_changes(err_changes, n=10)

    return {
        "total_errors": {
            "current": int(cq.get("total_errors") or 0),
            "prior": int(pq.get("total_errors") or 0),
            "delta": int((cq.get("total_errors") or 0) - (pq.get("total_errors") or 0)),
        },
        "error_count_top_moves": [{"field": f, "delta": d} for f, d in top_err_moves],
        "materiality_changes": mat_changes[:50],
        "new_blocking_fields": sorted(list(set(new_blocking))),
    }

def compute_transform_delta(curr: Dict[str, Any], prior: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic diff of optional reports (enum unmapped values / nd applied counts / header mapping).
    We keep it conservative: detect presence changes + basic keyed diffs if obvious.
    """
    out: Dict[str, Any] = {}

    cts = curr.get("transform_stats") or {}
    pts = prior.get("transform_stats") or {}

    # Enum: look for unmapped_values dict if present
    ce = cts.get("enum_mapping_report") or {}
    pe = pts.get("enum_mapping_report") or {}

    def extract_unmapped(r: dict) -> Dict[str, List[str]]:
        # common patterns: {"unmapped_values": {"FIELD": ["x","y"]}}
        if isinstance(r, dict):
            uv = r.get("unmapped_values")
            if isinstance(uv, dict):
                return {k: sorted(list(set(map(str, v)))) for k, v in uv.items() if isinstance(v, (list, tuple))}
        return {}

    cu = extract_unmapped(ce)
    pu = extract_unmapped(pe)
    if cu or pu:
        new_unmapped_fields = sorted(list(set(cu.keys()) - set(pu.keys())))
        resolved_unmapped_fields = sorted(list(set(pu.keys()) - set(cu.keys())))
        delta_counts = {}
        for f in set(cu.keys()) | set(pu.keys()):
            delta_counts[f] = len(cu.get(f, [])) - len(pu.get(f, []))
        out["enum_unmapped"] = {
            "new_fields": new_unmapped_fields,
            "resolved_fields": resolved_unmapped_fields,
            "top_count_moves": [{"field": f, "delta": d} for f, d in top_n_abs_changes(delta_counts, n=10)],
        }

    # ND defaults: try to find per-field counts
    cnd = cts.get("nd_defaults_report") or {}
    pnd = pts.get("nd_defaults_report") or {}

    def extract_nd_counts(r: dict) -> Dict[str, int]:
        # common patterns: {"nd_defaults_applied": {"FIELD": 123}}
        if isinstance(r, dict):
            nda = r.get("nd_defaults_applied") or r.get("applied") or r.get("fields")
            if isinstance(nda, dict):
                out2 = {}
                for k, v in nda.items():
                    try:
                        out2[str(k)] = int(v)
                    except Exception:
                        continue
                return out2
        return {}

    cndc = extract_nd_counts(cnd)
    pndc = extract_nd_counts(pnd)
    if cndc or pndc:
        deltas = {f: cndc.get(f, 0) - pndc.get(f, 0) for f in set(cndc.keys()) | set(pndc.keys())}
        out["nd_defaults"] = {"top_count_moves": [{"field": f, "delta": d} for f, d in top_n_abs_changes(deltas, 10)]}

    # Header mapping: if it exposes raw->canonical mapping, detect changes
    ch = cts.get("header_mapping_report") or {}
    ph = pts.get("header_mapping_report") or {}

    def extract_header_map(r: dict) -> Dict[str, str]:
        # accept {"mappings": [{"raw":"A","canonical":"loan_identifier"}]} or {"raw_to_canonical": {...}}
        if not isinstance(r, dict):
            return {}
        if isinstance(r.get("raw_to_canonical"), dict):
            return {str(k): str(v) for k, v in r["raw_to_canonical"].items()}
        m = r.get("mappings")
        if isinstance(m, list):
            outm = {}
            for it in m:
                if isinstance(it, dict) and "raw" in it and "canonical" in it:
                    outm[str(it["raw"])] = str(it["canonical"])
            return outm
        return {}

    cm = extract_header_map(ch)
    pm = extract_header_map(ph)
    if cm or pm:
        changed = []
        for raw in set(cm.keys()) | set(pm.keys()):
            if cm.get(raw) != pm.get(raw):
                changed.append({"raw_column": raw, "prior": pm.get(raw), "current": cm.get(raw)})
        out["header_mapping_changes"] = changed[:50]

    return out

def compute_delta(curr_manifest: Dict[str, Any], prior_manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp_utc": utc_now_iso(),
        "current_run_id": curr_manifest.get("run_id"),
        "prior_run_id": prior_manifest.get("run_id"),
        "portfolio_delta": compute_portfolio_delta(curr_manifest, prior_manifest),
        "schema_delta": compute_schema_delta(curr_manifest, prior_manifest),
        "quality_delta": compute_quality_delta(curr_manifest, prior_manifest),
        "transform_delta": compute_transform_delta(curr_manifest, prior_manifest),
    }


# ----------------- CLI -----------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None, help="Run ID; default is timestamp-based.")
    ap.add_argument("--outdir", required=True, help="Directory to write run_manifest.json and run_delta.json")

    ap.add_argument("--registry", default=None, help="Field registry YAML (optional; enables contract schema vs run footprint deltas)")
    ap.add_argument("--portfolio-type", default="equity_release", help="Used with --registry to compute contract schema (common + portfolio_type)")

    ap.add_argument("--current-canonical", required=True, help="Path to current canonical CSV")
    ap.add_argument("--prior-manifest", default=None, help="Path to prior run_manifest.json")
    ap.add_argument("--prior-canonical", default=None, help="Path to prior canonical CSV (if no prior manifest)")

    ap.add_argument("--id-col", default=None, help="Loan ID column name (optional)")
    ap.add_argument("--balance-col", default=None, help="Balance column name (optional)")

    ap.add_argument("--current-agg-validation", default=None, help="Path to current aggregated validation CSV (optional)")
    ap.add_argument("--prior-agg-validation", default=None, help="Path to prior aggregated validation CSV (optional; only used if building prior manifest from prior canonical)")

    ap.add_argument("--current-enum-report", default=None, help="Path to enum mapping report JSON (optional)")
    ap.add_argument("--prior-enum-report", default=None, help="Path to prior enum mapping report JSON (optional)")
    ap.add_argument("--current-nd-report", default=None, help="Path to ND defaults report JSON (optional)")
    ap.add_argument("--prior-nd-report", default=None, help="Path to prior ND defaults report JSON (optional)")
    ap.add_argument("--current-header-map", default=None, help="Path to header mapping report JSON (optional)")
    ap.add_argument("--prior-header-map", default=None, help="Path to prior header mapping report JSON (optional)")

    ap.add_argument("--config", action="append", default=[], help="Config files to hash (repeatable)")

    return ap.parse_args()

def main() -> int:
    ns = parse_args()
    outdir = Path(ns.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_id = ns.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    curr_manifest = build_manifest(
        canonical_csv=Path(ns.current_canonical),
        run_id=run_id,
        id_col=ns.id_col,
        balance_col=ns.balance_col,
        agg_validation_csv=Path(ns.current_agg_validation) if ns.current_agg_validation else None,
        enum_report_json=Path(ns.current_enum_report) if ns.current_enum_report else None,
        nd_report_json=Path(ns.current_nd_report) if ns.current_nd_report else None,
        header_map_json=Path(ns.current_header_map) if ns.current_header_map else None,
        config_paths=[Path(p) for p in (ns.config or [])],
    )

    manifest_path = outdir / "run_manifest.json"
    manifest_path.write_text(json.dumps(curr_manifest, indent=2, sort_keys=True), encoding="utf-8")

    # build/load prior manifest if possible
    prior_manifest = None
    if ns.prior_manifest:
        prior_manifest = safe_read_json(Path(ns.prior_manifest))
    elif ns.prior_canonical:
        prior_run_id = "prior"
        prior_manifest = build_manifest(
            canonical_csv=Path(ns.prior_canonical),
            run_id=prior_run_id,
            id_col=ns.id_col,
            balance_col=ns.balance_col,
            agg_validation_csv=Path(ns.prior_agg_validation) if ns.prior_agg_validation else None,
            enum_report_json=Path(ns.prior_enum_report) if ns.prior_enum_report else None,
            nd_report_json=Path(ns.prior_nd_report) if ns.prior_nd_report else None,
            header_map_json=Path(ns.prior_header_map) if ns.prior_header_map else None,
            config_paths=[Path(p) for p in (ns.config or [])],
        )

    if prior_manifest:
        delta = compute_delta(curr_manifest, prior_manifest)
        (outdir / "run_delta.json").write_text(json.dumps(delta, indent=2, sort_keys=True), encoding="utf-8")
    else:
        # still succeed: only manifest written
        pass

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
