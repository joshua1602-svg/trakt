#!/usr/bin/env python3
"""
trakt_orchestrator.py - Deterministic pipeline orchestrator + investor-friendly run manifest.

Windows-safe:
- No shell multiline strings
- Uses sys.executable consistently (works with py launcher)
- Best-effort metrics + clear gate prints
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd


# -------------------------
# helpers
# -------------------------

def _run(args: List[str], allow_fail: bool = False) -> int:
    """
    Run a command, streaming output to console.
    Forces UTF-8 env for child python processes (Windows-safe).
    Returns exit code; raises only when allow_fail=False.
    """
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(args, env=env)
    if result.returncode != 0 and not allow_fail:
        raise subprocess.CalledProcessError(result.returncode, args)
    return result.returncode


def _count_rows_quick(csv_path: Path) -> int:
    # fast-ish row count without loading everything into memory
    # (still reads the file, but does not parse full df)
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        # subtract header
        return max(sum(1 for _ in f) - 1, 0)


def _safe_read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_matching(dir_path: Path, patterns: List[str]) -> Optional[Path]:
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(dir_path.glob(pat))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _best_effort_validation_paths(val_dir: Path, stem: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Your filenames are not always perfectly stable across versions.
    Find the most likely canonical violations + business violations files.
    """
    canon = _latest_matching(
        val_dir,
        [
            f"{stem}_canonical_typed_canonical_violations.csv",
            f"{stem}*canonical*violations*.csv",
            "*canonical*violations*.csv",
        ],
    )
    biz = _latest_matching(
        val_dir,
        [
            f"{stem}_canonical_business_rules_violations.csv",
            f"{stem}*business*violations*.csv",
            "*business*violations*.csv",
        ],
    )
    return canon, biz


def _summarise_violations(path: Optional[Path]) -> Dict[str, int]:
    if not path or not path.exists():
        return {"errors": 0, "warnings": 0, "rows": 0}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {"errors": 0, "warnings": 0, "rows": 0}

    rows = int(len(df))
    if "severity" in df.columns:
        sev = df["severity"].astype(str).str.lower()
        warnings = int((sev == "warning").sum())
        errors = int((sev == "error").sum())
        return {"errors": errors, "warnings": warnings, "rows": rows}
    # fallback if no severity col
    return {"errors": rows, "warnings": 0, "rows": rows}

def _count_hq_recommendations(header: Optional[dict], min_conf: float = 0.88) -> int:
    """
    Count HQ recommendations from header_mapping_report.json.

    Supports both JSON shapes:
      - top-level: {"hq_recommendations":[...], "hq_recommendations_count": N}
      - legacy:   {"thresholds":{"hq_recommendations":[...], "hq_recommendations_count": N}}

    min_conf is a fraction (0.88 = 88%).
    """
    if not header or not isinstance(header, dict):
        return 0

    # Prefer explicit count if present
    cnt = header.get("hq_recommendations_count")
    if isinstance(cnt, int):
        return cnt

    thr = header.get("thresholds") if isinstance(header.get("thresholds"), dict) else {}

    cnt = thr.get("hq_recommendations_count")
    if isinstance(cnt, int):
        return cnt

    # Fall back to counting list entries
    recs = header.get("hq_recommendations")
    if not isinstance(recs, list):
        recs = thr.get("hq_recommendations")
    if not isinstance(recs, list):
        return 0

    # If items have confidence as 0-1 or 0-100, handle both
    n = 0
    for r in recs:
        if not isinstance(r, dict):
            continue
        c = r.get("confidence")
        if c is None:
            c = r.get("confidence_pct")
            if c is not None:
                try:
                    c = float(c) / 100.0
                except Exception:
                    c = None
        try:
            c = float(c)
        except Exception:
            c = None
        if c is not None and c >= float(min_conf):
            n += 1
    return n

    # Case 2: infer from mappings if candidates are embedded (best-effort)
    mappings = header_json.get("mappings")
    if isinstance(mappings, list):
        n = 0
        for r in mappings:
            if not isinstance(r, dict):
                continue
            # Only consider truly unmapped rows
            if r.get("canonical_field"):
                continue
            # Look for a candidate confidence field if you store it
            cand_conf = r.get("best_candidate_confidence") or r.get("confidence") or r.get("score")
            try:
                if cand_conf is not None and float(cand_conf) >= float(min_conf):
                    n += 1
            except Exception:
                pass
        return n

    return 0

def _field_counts_from_violations(path: Optional[Path]) -> Dict[str, int]:
    """
    Return unique field counts for errors/warnings (best-effort).
    Expects a column named 'field' or 'field_name' and 'severity'.
    """
    if not path or not path.exists():
        return {"fields_with_errors": 0, "fields_with_warnings": 0}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {"fields_with_errors": 0, "fields_with_warnings": 0}

    # find field column
    field_col = "field_name" if "field_name" in df.columns else ("field" if "field" in df.columns else None)
    if field_col is None or "severity" not in df.columns:
        return {"fields_with_errors": 0, "fields_with_warnings": 0}

    sev = df["severity"].astype(str).str.lower()
    fields = df[field_col].astype(str)

    fields_with_errors = int(fields[sev == "error"].nunique())
    fields_with_warnings = int(fields[sev == "warning"].nunique())

    return {"fields_with_errors": fields_with_errors, "fields_with_warnings": fields_with_warnings}

# -------------------------
# main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="trakt run (orchestrator + run manifest)")
    ap.add_argument("--input", required=True, help="Input loan tape CSV/XLSX (e.g. loan_portfolio_112025.csv)")
    ap.add_argument("--config", required=True, help="Annex12 config YAML (e.g. client_config_annex_12.yaml)")

    # optional knobs
    ap.add_argument("--portfolio-type", default="equity_release")
    ap.add_argument("--output-schema", choices=["active", "full"], default="active")
    ap.add_argument("--registry", default="data_standard_definition.yaml")
    ap.add_argument("--master-config", default="asset_policy_uk.yaml")

    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--validation-out-dir", default="out_validation")

    ap.add_argument("--constraints", default="esma_12_integrity_rules.yaml")
    ap.add_argument("--code-order-yaml", default="submission_schema_layout.yaml")
    ap.add_argument("--rules", default="esma_12_disclosure_logic.yaml")
    ap.add_argument("--mapping", default="annex12_mapping.csv")
    ap.add_argument("--currency", default="GBP")
    ap.add_argument("--xsd", default="DRAFT1auth.098.001.04_1.3.0.xsd")

    args = ap.parse_args()

    run_start = time.time()
    py = sys.executable  # ensures we run with same interpreter as `py`

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = Path(args.out_dir)
    val_dir = Path(args.validation_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem

    canonical_full = out_dir / f"{stem}_canonical_full.csv"
    canonical_typed = out_dir / f"{stem}_canonical_typed.csv"
    header_json = out_dir / f"{stem}_header_mapping_report.json"
    transform_json = out_dir / f"{stem}_transform_report.json"
    field_lineage_json = out_dir / "field_lineage.json"
    value_lineage_json = out_dir / "value_lineage.json"
    manifest_path = out_dir / "run_manifest.json"

    annex_projected = Path("annex12_projected.csv")
    annex_xml = Path("annex12_final.xml")

    print("")
    print(f"$ trakt run --config {args.config} --input {args.input}")
    print("")

    # -------------------------
    # Gate 1: Semantic alignment (messy -> canonical)
    # -------------------------
    _run([
        py, "alignment_engine.py",
        "--input", str(input_path),
        "--portfolio-type", args.portfolio_type,
        "--output-schema", args.output_schema,
        "--registry", args.registry,
        "--output-dir", str(out_dir),
    ])

    if not canonical_full.exists():
        raise RuntimeError(f"[Gate 1] Failed: did not produce {canonical_full}")

    fields_mapped = None
    header = _safe_read_json(header_json)
    if header and isinstance(header.get("raw_to_canonical"), dict):
        fields_mapped = len(header["raw_to_canonical"])

    hq_recs = _count_hq_recommendations(header, min_conf=0.88)

    if fields_mapped is not None:
        print(f"[Gate 1] Semantic alignment.............. OK {fields_mapped} fields mapped | {hq_recs} HQ recommendations")
    else:
        print(f"[Gate 1] Semantic alignment.............. OK PASS | {hq_recs} HQ recommendations")

    # -------------------------
    # Transform (typing/derivations)
    # -------------------------
    _run([
        py, "portfolio_synthesizer.py",
        str(canonical_full),
        "--registry", args.registry,
        "--portfolio-type", args.portfolio_type,
        "--config", args.master_config,
        "--output-dir", str(out_dir),
    ])

    if not canonical_typed.exists():
        raise RuntimeError(f"Transform failed: did not produce {canonical_typed}")

    # -------------------------
    # Gate 2: Canonical validation
    # -------------------------
    canon_rc = _run([
        py, "gatekeeper.py",
        str(canonical_typed),
        "--registry", args.registry,
        "--portfolio-type", args.portfolio_type,
        "--scope", "canonical",
        "--out-dir", str(val_dir),
    ], allow_fail=True)


    loan_count = _count_rows_quick(canonical_typed)

    canonical_status = "pass" if canon_rc == 0 else "fail"
    canon_viol_path, biz_viol_path = _best_effort_validation_paths(val_dir, stem)
    canon_stats = _summarise_violations(canon_viol_path)
    canon_field_counts = _field_counts_from_violations(canon_viol_path)


    fields_err = canon_field_counts["fields_with_errors"]
    fields_warn = canon_field_counts["fields_with_warnings"]

    # status: FAIL if any errors, else OK
    gate2_status = "FAIL" if canon_stats["errors"] > 0 else "OK"

    print(
        f"[Gate 2] Canonical validation............ {gate2_status} "
        f"{loan_count:,} loans | {canon_stats['warnings']} warnings ({fields_warn} fields) | "
        f"{canon_stats['errors']} errors ({fields_err} fields)"
    )

    # -------------------------
    # Gate 2.5: Lineage
    # -------------------------

    _run([
        py, "lineage_JSON.py",
        "--canonical", str(canonical_typed),
        "--registry", args.registry,
        "--portfolio-type", args.portfolio_type,
        "--outdir", str(out_dir),
        "--header-map", str(header_json),
        "--transform-report", str(transform_json), 
    ])
    
    lineage_path = out_dir / "field_lineage.json"
    if not lineage_path.exists():
        raise RuntimeError("[Gate 2.5] Lineage failed: field_lineage.json not produced")
    
    if not field_lineage_json.exists():
        raise RuntimeError("[Gate 2.5] Lineage failed: did not produce field_lineage.json")

    # -------------------------
    # Gate 3: Business rules
    # -------------------------
    biz_rc = _run([
        py, "validate_business_rules_aligned_v1_2.py",
        str(canonical_typed),
        "--config", args.master_config,
    ], allow_fail=True)

    biz_stats = _summarise_violations(biz_viol_path)
    # We also try to infer unique rule count if present
    rules_executed = None
    if biz_viol_path and biz_viol_path.exists():
        try:
            bdf = pd.read_csv(biz_viol_path)
            if "rule_id" in bdf.columns:
                rules_executed = int(bdf["rule_id"].nunique())
        except Exception:
            pass

    if biz_stats["rows"] == 0:
        # if no failures file or empty file
        if rules_executed is not None:
            print(f"[Gate 3] Business rules.................. OK {rules_executed} rules | 0 failures")
        else:
            print(f"[Gate 3] Business rules.................. OK PASS")
    else:
        if rules_executed is not None:
            print(f"[Gate 3] Business rules.................. Warn {rules_executed} rules | {biz_stats['rows']} failures")
        else:
            print(f"[Gate 3] Business rules.................. Warn {biz_stats['rows']} failures")

    # -------------------------
    # Gate 4: Regime projection
    # -------------------------
    _run([
        py, "esma_investor_regime_adapter.py",
        "--config", args.config,
        "--master-config", args.master_config,
        "--canonical", str(canonical_typed),
        "--constraints", args.constraints,
        "--code-order-yaml", args.code_order_yaml,
        "--output", str(annex_projected),
    ])

    if not annex_projected.exists():
        raise RuntimeError(f"[Gate 4] Failed: did not produce {annex_projected}")

    print("[Gate 4] Regime projection............... OK ESMA Annex 12")

    # -------------------------
    # Gate 5: XML build + XSD validation
    # -------------------------
    _run([
        py, "esma_investor_disclosure_generator.py",
        "--input", str(annex_projected),
        "--mapping", args.mapping,
        "--rules", args.rules,
        "--code-order-yaml", args.code_order_yaml,
        "--output", str(annex_xml),
        "--currency", args.currency,
        "--xsd", args.xsd,
    ])

    if not annex_xml.exists():
        raise RuntimeError(f"[Gate 5] Failed: did not produce {annex_xml}")

    print("[Gate 5] XSD validation.................. OK PASS")

    # -------------------------
    # Manifest
    # -------------------------
    transform = _safe_read_json(transform_json) or {}
    loan_count_reported = transform.get("rows", loan_count)

    manifest: Dict[str, Any] = {
        "run_id": time.strftime("run_%Y%m%d_%H%M%S"),
        "config": args.config,
        "input_file": str(input_path),
        "portfolio_type": args.portfolio_type,
        "output_schema": args.output_schema,
        "loan_count": int(loan_count_reported) if str(loan_count_reported).isdigit() else loan_count,
        "gates": {
            "semantic_alignment": {
                "status": "pass",
                "fields_mapped": fields_mapped,
                "artifact": str(header_json) if header_json.exists() else None,
            },
            "canonical_transform": {
                "status": "pass",
                "artifact": str(transform_json) if transform_json.exists() else None,
            },
            "canonical_validation": {
                "status": canonical_status,
                "warnings": canon_stats["warnings"],
                "errors": canon_stats["errors"],
                "blocking": canon_stats["errors"] > 0,
                "artifact": str(canon_viol_path) if canon_viol_path else None,
            },
            "business_rules": {
                "status": "pass" if biz_stats["rows"] == 0 else "review",
                "failures": biz_stats["rows"],
                "rules_executed": rules_executed,
                "artifact": str(biz_viol_path) if biz_viol_path else None,
            },
            "regime_projection": {
                "status": "pass",
                "regime": "ESMA_Annex12",
                "artifact": str(annex_projected),
            },
            "xsd_validation": {
                "status": "pass",
                "artifact": str(annex_xml),
            },
        },
        "outputs": [
            str(canonical_full),
            str(canonical_typed),
            str(transform_json) if transform_json.exists() else None,
            str(field_lineage_json) if field_lineage_json.exists() else None,
            str(value_lineage_json) if value_lineage_json.exists() else None,
            str(annex_projected),
            str(annex_xml),
        ],
        "execution_time_seconds": round(time.time() - run_start, 2),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest["outputs"] = [x for x in manifest["outputs"] if x]

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # -------------------------
    # Final output list (investor proof)
    # -------------------------
    print("")
    print("Output artefacts:")
    print(f"  -> {canonical_full}")
    print(f"  -> {canonical_typed}")
    if transform_json.exists():
        print(f"  -> {transform_json}")
    if canon_viol_path:
        print(f"  -> {canon_viol_path}")
    if biz_viol_path:
        print(f"  -> {biz_viol_path}")
    if field_lineage_json.exists():
        print(f"  -> {field_lineage_json}")
    if value_lineage_json.exists():
        print(f"  -> {value_lineage_json}")
    print(f"  -> {annex_projected}")
    print(f"  -> {annex_xml}")
    print(f"  -> {manifest_path}")
    print("")
    print(f"Completed in {manifest['execution_time_seconds']}s")

if __name__ == "__main__":
    main()

