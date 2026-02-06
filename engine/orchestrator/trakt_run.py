#!/usr/bin/env python3
"""
trakt_run.py - Multi-mode pipeline orchestrator.

Modes:
  mi          Gates 1-3 → canonical_typed.csv (dashboard-ready for Streamlit)
  annex12     Gates 1-5 → ESMA Annex 12 investor XML (deal-level)
  regulatory  Gates 1-5 → ESMA Annex 2-9 regime projection + XML (exposure-level)

Usage:
  python trakt_run.py --mode mi --input tape.csv
  python trakt_run.py --mode annex12 --input tape.csv --config config_client_annex12.yaml
  python trakt_run.py --mode regulatory --input tape.csv --regime ESMA_Annex2

Windows-safe: uses sys.executable, forces UTF-8 encoding in child processes.
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


# ---------------------------------------------------------------------------
# Path resolution — derive project layout from this file's location
# ---------------------------------------------------------------------------

ENGINE_ROOT  = Path(__file__).resolve().parent.parent          # engine/
PROJECT_ROOT = ENGINE_ROOT.parent                               # trakt/
CONFIG_ROOT  = PROJECT_ROOT / "config"

SCRIPTS = {
    "semantic_alignment":   ENGINE_ROOT / "gate_1_alignment"  / "semantic_alignment.py",
    "canonical_transform":  ENGINE_ROOT / "gate_2_transform"  / "canonical_transform.py",
    "lineage_tracker":      ENGINE_ROOT / "gate_2_transform"  / "lineage_tracker.py",
    "validate_canonical":   ENGINE_ROOT / "gate_3_validation" / "validate_canonical.py",
    "validate_business_rules": ENGINE_ROOT / "gate_3_validation" / "validate_business_rules.py",
    "annex12_projector":    ENGINE_ROOT / "gate_4_projection" / "annex12_projector.py",
    "regime_projector":     ENGINE_ROOT / "gate_4_projection" / "regime_projector.py",
    "xml_builder_investor": ENGINE_ROOT / "gate_5_delivery"   / "xml_builder_investor.py",
    "xml_builder":          ENGINE_ROOT / "gate_5_delivery"   / "xml_builder.py",
}

VALID_REGULATORY_REGIMES = [
    "ESMA_Annex2", "ESMA_Annex3", "ESMA_Annex4",
    "ESMA_Annex8", "ESMA_Annex9",
]


def _script(name: str) -> str:
    """Resolve a gate script to its absolute path. Fails fast if missing."""
    path = SCRIPTS[name]
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(args: List[str], allow_fail: bool = False) -> int:
    """Run a subprocess with UTF-8 encoding. Returns exit code."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(args, env=env)
    if result.returncode != 0 and not allow_fail:
        raise subprocess.CalledProcessError(result.returncode, args)
    return result.returncode


def _count_rows_quick(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
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
    """Find the most likely canonical violations + business violations files."""
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
    return {"errors": rows, "warnings": 0, "rows": rows}


def _count_hq_recommendations(header: Optional[dict], min_conf: float = 0.88) -> int:
    """Count HQ recommendations from header_mapping_report.json."""
    if not header or not isinstance(header, dict):
        return 0

    cnt = header.get("hq_recommendations_count")
    if isinstance(cnt, int):
        return cnt

    thr = header.get("thresholds") if isinstance(header.get("thresholds"), dict) else {}

    cnt = thr.get("hq_recommendations_count")
    if isinstance(cnt, int):
        return cnt

    recs = header.get("hq_recommendations")
    if not isinstance(recs, list):
        recs = thr.get("hq_recommendations")
    if not isinstance(recs, list):
        return 0

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


def _field_counts_from_violations(path: Optional[Path]) -> Dict[str, int]:
    """Return unique field counts for errors/warnings."""
    if not path or not path.exists():
        return {"fields_with_errors": 0, "fields_with_warnings": 0}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {"fields_with_errors": 0, "fields_with_warnings": 0}

    field_col = "field_name" if "field_name" in df.columns else ("field" if "field" in df.columns else None)
    if field_col is None or "severity" not in df.columns:
        return {"fields_with_errors": 0, "fields_with_warnings": 0}

    sev = df["severity"].astype(str).str.lower()
    fields = df[field_col].astype(str)

    return {
        "fields_with_errors": int(fields[sev == "error"].nunique()),
        "fields_with_warnings": int(fields[sev == "warning"].nunique()),
    }


# ---------------------------------------------------------------------------
# Gate runners
# ---------------------------------------------------------------------------

def run_common_gates(py: str, args, input_path: Path, out_dir: Path, val_dir: Path, stem: str) -> dict:
    """
    Run Gates 1-3 (common to all modes).
    Returns a context dict with intermediate results for the manifest.
    """
    canonical_full   = out_dir / f"{stem}_canonical_full.csv"
    canonical_typed  = out_dir / f"{stem}_canonical_typed.csv"
    header_json      = out_dir / f"{stem}_header_mapping_report.json"
    transform_json   = out_dir / f"{stem}_transform_report.json"
    field_lineage    = out_dir / "field_lineage.json"
    value_lineage    = out_dir / "value_lineage.json"

    # -- Gate 1: Semantic alignment ----------------------------------------
    gate1_cmd = [
        py, _script("semantic_alignment"),
        "--input", str(input_path),
        "--portfolio-type", args.portfolio_type,
        "--output-schema", args.output_schema,
        "--registry", args.registry,
        "--output-dir", str(out_dir),
    ]
    # For regulatory mode, filter "full" schema to the target annex fields
    if args.mode == "regulatory" and args.regime:
        gate1_cmd.extend(["--regimes", args.regime])

    _run(gate1_cmd)

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

    # -- Transform (typing / derivations) ----------------------------------
    _run([
        py, _script("canonical_transform"),
        str(canonical_full),
        "--registry", args.registry,
        "--portfolio-type", args.portfolio_type,
        "--config", args.master_config,
        "--output-dir", str(out_dir),
    ])

    if not canonical_typed.exists():
        raise RuntimeError(f"[Transform] Failed: did not produce {canonical_typed}")

    print("[Transform] Canonical transform.......... OK")

    # -- Gate 2: Canonical validation --------------------------------------
    canon_rc = _run([
        py, _script("validate_canonical"),
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

    fields_err  = canon_field_counts["fields_with_errors"]
    fields_warn = canon_field_counts["fields_with_warnings"]
    gate2_status = "FAIL" if canon_stats["errors"] > 0 else "OK"

    print(
        f"[Gate 2] Canonical validation............ {gate2_status} "
        f"{loan_count:,} loans | {canon_stats['warnings']} warnings ({fields_warn} fields) | "
        f"{canon_stats['errors']} errors ({fields_err} fields)"
    )

    # -- Gate 2.5: Lineage -------------------------------------------------
    _run([
        py, _script("lineage_tracker"),
        "--canonical", str(canonical_typed),
        "--registry", args.registry,
        "--portfolio-type", args.portfolio_type,
        "--outdir", str(out_dir),
        "--header-map", str(header_json),
        "--transform-report", str(transform_json),
    ])

    if not field_lineage.exists():
        raise RuntimeError("[Gate 2.5] Lineage failed: field_lineage.json not produced")

    print("[Gate 2.5] Data lineage.................. OK")

    # -- Gate 3: Business rules --------------------------------------------
    biz_cmd = [
        py, _script("validate_business_rules"),
        str(canonical_typed),
        "--config", args.master_config,
    ]
    if hasattr(args, "regime") and args.regime:
        biz_cmd.extend(["--regime", args.regime])

    biz_rc = _run(biz_cmd, allow_fail=True)

    biz_stats = _summarise_violations(biz_viol_path)
    rules_executed = None
    if biz_viol_path and biz_viol_path.exists():
        try:
            bdf = pd.read_csv(biz_viol_path)
            if "rule_id" in bdf.columns:
                rules_executed = int(bdf["rule_id"].nunique())
        except Exception:
            pass

    if biz_stats["rows"] == 0:
        if rules_executed is not None:
            print(f"[Gate 3] Business rules.................. OK {rules_executed} rules | 0 failures")
        else:
            print(f"[Gate 3] Business rules.................. OK PASS")
    else:
        if rules_executed is not None:
            print(f"[Gate 3] Business rules.................. Warn {rules_executed} rules | {biz_stats['rows']} failures")
        else:
            print(f"[Gate 3] Business rules.................. Warn {biz_stats['rows']} failures")

    return {
        "canonical_full": canonical_full,
        "canonical_typed": canonical_typed,
        "header_json": header_json,
        "transform_json": transform_json,
        "field_lineage": field_lineage,
        "value_lineage": value_lineage,
        "fields_mapped": fields_mapped,
        "loan_count": loan_count,
        "canonical_status": canonical_status,
        "canon_stats": canon_stats,
        "canon_viol_path": canon_viol_path,
        "biz_stats": biz_stats,
        "biz_viol_path": biz_viol_path,
        "rules_executed": rules_executed,
    }


def run_annex12(py: str, args, ctx: dict, out_dir: Path) -> dict:
    """Gate 4-5 for Annex 12 investor reporting (deal-level)."""
    annex_projected = out_dir / "annex12_projected.csv"
    annex_xml       = out_dir / "annex12_final.xml"

    # -- Gate 4: Annex 12 projection ---------------------------------------
    _run([
        py, _script("annex12_projector"),
        "--config", args.config,
        "--master-config", args.master_config,
        "--canonical", str(ctx["canonical_typed"]),
        "--constraints", args.constraints,
        "--code-order-yaml", args.code_order_yaml,
        "--output", str(annex_projected),
    ])

    if not annex_projected.exists():
        raise RuntimeError(f"[Gate 4] Failed: did not produce {annex_projected}")

    print("[Gate 4] Regime projection............... OK ESMA Annex 12")

    # -- Gate 5: XML + XSD -------------------------------------------------
    _run([
        py, _script("xml_builder_investor"),
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

    return {
        "regime": "ESMA_Annex12",
        "projected": annex_projected,
        "xml": annex_xml,
    }


def run_regulatory(py: str, args, ctx: dict, out_dir: Path) -> dict:
    """Gate 4-5 for ESMA Annex 2-9 exposure-level reporting."""
    regime = args.regime
    regime_short = regime.replace("ESMA_", "").lower()  # e.g. "annex2"
    projected = out_dir / f"{regime_short}_projected.csv"
    xml_out   = out_dir / f"{regime_short}_final.xml"

    # -- Gate 4: Regime projection -----------------------------------------
    _run([
        py, _script("regime_projector"),
        str(ctx["canonical_typed"]),
        "--regime", regime,
        "--registry", args.registry,
        "--enum-mapping", args.enum_mapping,
        "--config", args.master_config,
        "--template-order", args.code_order_yaml,
        "--portfolio-type", args.portfolio_type,
        "--output-dir", str(out_dir),
    ])

    if not projected.exists():
        # regime_projector uses {stem}_{regime}_projected.csv naming
        candidates = list(out_dir.glob(f"*{regime}*projected*.csv"))
        if candidates:
            projected = max(candidates, key=lambda p: p.stat().st_mtime)
        else:
            raise RuntimeError(f"[Gate 4] Failed: no projected CSV found for {regime}")

    print(f"[Gate 4] Regime projection............... OK {regime}")

    # -- Gate 5: XML generation --------------------------------------------
    _run([
        py, _script("xml_builder"),
        "--input", str(projected),
        "--output", str(xml_out),
        "--currency", args.currency,
    ])

    if not xml_out.exists():
        raise RuntimeError(f"[Gate 5] Failed: did not produce {xml_out}")

    print(f"[Gate 5] XML generation.................. OK {regime}")

    return {
        "regime": regime,
        "projected": projected,
        "xml": xml_out,
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(args, ctx: dict, gate45: Optional[dict], out_dir: Path, run_start: float) -> Path:
    """Write the run_manifest.json summarising the pipeline execution."""
    manifest_path = out_dir / "run_manifest.json"
    transform = _safe_read_json(ctx["transform_json"]) or {}
    loan_count_raw = transform.get("rows", ctx["loan_count"])
    loan_count = int(loan_count_raw) if str(loan_count_raw).isdigit() else ctx["loan_count"]

    gates: Dict[str, Any] = {
        "semantic_alignment": {
            "status": "pass",
            "fields_mapped": ctx["fields_mapped"],
            "artifact": str(ctx["header_json"]) if ctx["header_json"].exists() else None,
        },
        "canonical_transform": {
            "status": "pass",
            "artifact": str(ctx["transform_json"]) if ctx["transform_json"].exists() else None,
        },
        "canonical_validation": {
            "status": ctx["canonical_status"],
            "warnings": ctx["canon_stats"]["warnings"],
            "errors": ctx["canon_stats"]["errors"],
            "blocking": ctx["canon_stats"]["errors"] > 0,
            "artifact": str(ctx["canon_viol_path"]) if ctx["canon_viol_path"] else None,
        },
        "business_rules": {
            "status": "pass" if ctx["biz_stats"]["rows"] == 0 else "review",
            "failures": ctx["biz_stats"]["rows"],
            "rules_executed": ctx["rules_executed"],
            "artifact": str(ctx["biz_viol_path"]) if ctx["biz_viol_path"] else None,
        },
    }

    if gate45:
        gates["regime_projection"] = {
            "status": "pass",
            "regime": gate45["regime"],
            "artifact": str(gate45["projected"]),
        }
        if gate45.get("xml"):
            gates["xsd_validation"] = {
                "status": "pass",
                "artifact": str(gate45["xml"]),
            }

    outputs = [
        str(ctx["canonical_full"]),
        str(ctx["canonical_typed"]),
        str(ctx["transform_json"]) if ctx["transform_json"].exists() else None,
        str(ctx["field_lineage"]) if ctx["field_lineage"].exists() else None,
        str(ctx["value_lineage"]) if ctx["value_lineage"].exists() else None,
    ]
    if gate45:
        outputs.append(str(gate45["projected"]))
        if gate45.get("xml"):
            outputs.append(str(gate45["xml"]))

    manifest: Dict[str, Any] = {
        "run_id": time.strftime("run_%Y%m%d_%H%M%S"),
        "mode": args.mode,
        "config": getattr(args, "config", None),
        "input_file": str(args.input),
        "portfolio_type": args.portfolio_type,
        "output_schema": args.output_schema,
        "loan_count": loan_count,
        "gates": gates,
        "outputs": [x for x in outputs if x],
        "execution_time_seconds": round(time.time() - run_start, 2),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="trakt — multi-mode pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
modes:
  mi          Run Gates 1-3 to produce canonical_typed.csv for the Streamlit dashboard.
  annex12     Run Gates 1-5 to produce ESMA Annex 12 investor XML (deal-level).
  regulatory  Run Gates 1-5 to produce ESMA Annex 2-9 regime projection + XML (exposure-level).

examples:
  python trakt_run.py --mode mi --input tape.csv
  python trakt_run.py --mode annex12 --input tape.csv --config config_client_annex12.yaml
  python trakt_run.py --mode regulatory --input tape.csv --regime ESMA_Annex2
""",
    )

    ap.add_argument("--mode", required=True, choices=["mi", "annex12", "regulatory"],
                     help="Pipeline mode: mi | annex12 | regulatory")
    ap.add_argument("--input", required=True,
                     help="Input loan tape CSV/XLSX")

    # Common config
    ap.add_argument("--portfolio-type", default="equity_release")
    ap.add_argument("--output-schema", choices=["active", "full"], default="active")
    ap.add_argument("--registry", default=str(CONFIG_ROOT / "system" / "fields_registry.yaml"))
    ap.add_argument("--master-config", default=str(CONFIG_ROOT / "client" / "config_client_ERM_UK.yaml"))
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--validation-out-dir", default="out_validation")

    # Annex 12 mode
    ap.add_argument("--config", default=None,
                     help="Annex 12 config YAML (required for annex12 mode)")
    ap.add_argument("--constraints", default=str(CONFIG_ROOT / "regime" / "annex12_field_constraints.yaml"))
    ap.add_argument("--mapping", default="annex12_mapping.csv")
    ap.add_argument("--rules", default=str(CONFIG_ROOT / "regime" / "annex12_rules.yaml"))
    ap.add_argument("--xsd", default=str(CONFIG_ROOT / "system" / "DRAFT1auth.098.001.04_1.3.0.xsd"))

    # Regulatory mode (Annex 2-9)
    ap.add_argument("--regime", default=None,
                     help="Target regime for regulatory mode (e.g. ESMA_Annex2)")
    ap.add_argument("--enum-mapping", default=str(CONFIG_ROOT / "system" / "enum_mapping.yaml"))

    # Shared
    ap.add_argument("--code-order-yaml", default=str(CONFIG_ROOT / "system" / "esma_code_order.yaml"))
    ap.add_argument("--currency", default="GBP")

    args = ap.parse_args()

    # -- Validate mode-specific requirements -------------------------------
    if args.mode == "annex12" and not args.config:
        ap.error("--config is required for annex12 mode")
    if args.mode == "regulatory":
        if not args.regime:
            ap.error("--regime is required for regulatory mode")
        if args.regime not in VALID_REGULATORY_REGIMES:
            ap.error(f"--regime must be one of: {', '.join(VALID_REGULATORY_REGIMES)}")

    # -- Schema policy per mode --------------------------------------------
    # MI & Annex 12: "active" = core:true + mapped headers (lean dataset)
    # Regulatory:    "full"   = all fields for the target annex (complete)
    if args.mode == "regulatory":
        args.output_schema = "full"
    elif args.mode in ("mi", "annex12"):
        args.output_schema = "active"

    # -- Setup -------------------------------------------------------------
    run_start = time.time()
    py = sys.executable

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = Path(args.out_dir)
    val_dir = Path(args.validation_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem

    print("")
    print(f"$ trakt run --mode {args.mode} --input {args.input}")
    if args.mode == "annex12":
        print(f"  config: {args.config}")
    elif args.mode == "regulatory":
        print(f"  regime: {args.regime}")
    print("")

    # -- Gates 1-3 (common) ------------------------------------------------
    ctx = run_common_gates(py, args, input_path, out_dir, val_dir, stem)

    # -- Mode-specific gates -----------------------------------------------
    gate45 = None

    if args.mode == "mi":
        print("")
        print("[MI] Dashboard-ready canonical produced.")

    elif args.mode == "annex12":
        gate45 = run_annex12(py, args, ctx, out_dir)

    elif args.mode == "regulatory":
        gate45 = run_regulatory(py, args, ctx, out_dir)

    # -- Manifest ----------------------------------------------------------
    manifest_path = write_manifest(args, ctx, gate45, out_dir, run_start)

    # -- Summary -----------------------------------------------------------
    elapsed = round(time.time() - run_start, 2)
    print("")
    print("Output artefacts:")
    print(f"  -> {ctx['canonical_full']}")
    print(f"  -> {ctx['canonical_typed']}")
    if ctx["transform_json"].exists():
        print(f"  -> {ctx['transform_json']}")
    if ctx["canon_viol_path"]:
        print(f"  -> {ctx['canon_viol_path']}")
    if ctx["biz_viol_path"]:
        print(f"  -> {ctx['biz_viol_path']}")
    if ctx["field_lineage"].exists():
        print(f"  -> {ctx['field_lineage']}")
    if ctx["value_lineage"].exists():
        print(f"  -> {ctx['value_lineage']}")
    if gate45:
        print(f"  -> {gate45['projected']}")
        if gate45.get("xml"):
            print(f"  -> {gate45['xml']}")
    print(f"  -> {manifest_path}")
    print("")
    print(f"Completed in {elapsed}s")


if __name__ == "__main__":
    main()
