"""demo_platform.artefacts — the governed outputs the closing scenes show.

SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.

Produces, from the assembled platform canonical the recurring run just published:

  * the consolidated canonical tape (already produced by the assembler — recorded
    here with its row count, balance and content hash);
  * a consolidated validation report, aggregated from the per-portfolio Gate 2 /
    Gate 3 / Gate 3b outputs;
  * the investor PowerPoint pack, via ``mi_agent_pptx.cli`` — the real deck
    builder, which reads the same ``/mi/*`` computations the dashboard uses;
  * the risk-monitoring output, via the ``/mi/risk-limits`` service;
  * an audit manifest: every input file, every gate status, every output, with
    provenance and content hashes.

Each artefact records whether it was actually produced. Anything that could not
be produced is reported as unavailable with the reason — the film only shows what
exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from . import config as cfg

warnings.simplefilter("ignore")

DECK_NAME = "investor_pack.pptx"
POINTER_NAME = "latest_investor_pack.json"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dated_platform_dir(prd: cfg.PeriodSpec) -> Path:
    return (cfg.local_blob_root() / "processed" / "platform"
            / cfg.CLIENT_ID / prd.reporting_date)


# --------------------------------------------------------------------------- #
# 1. Consolidated canonical tape
# --------------------------------------------------------------------------- #
def canonical_tape() -> Dict[str, Any]:
    """Record the assembled consolidated tape for the current period."""
    path = _dated_platform_dir(cfg.CURRENT_PERIOD) / "platform_canonical_typed.csv"
    manifest_path = path.parent / "platform_canonical_manifest.json"
    if not path.exists():
        return {"available": False, "reason": "the platform canonical has not been assembled"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    df = pd.read_csv(path, low_memory=False)
    return {
        "available": True,
        "kind": "consolidated_canonical_tape",
        "title": "Consolidated canonical tape",
        "fileName": path.name,
        "format": "CSV",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "sizeBytes": path.stat().st_size,
        "totalBalance": manifest.get("output_total_balance"),
        "portfolioCount": manifest.get("portfolio_count"),
        "compositeKey": manifest.get("composite_key"),
        "contentSha256": manifest.get("content_sha256"),
        "reportingDate": cfg.CURRENT_PERIOD.reporting_date,
        "producedBy": "engine/platform_assembler.py",
    }


# --------------------------------------------------------------------------- #
# 2. Consolidated validation report
# --------------------------------------------------------------------------- #
def validation_report() -> Dict[str, Any]:
    """Aggregate the current period's per-portfolio validation into one report."""
    per_portfolio: List[Dict[str, Any]] = []
    rule_totals: Dict[str, int] = {}
    total_rows = 0
    total_exceptions = 0
    gate2_ok = True

    for portfolio in cfg.PORTFOLIOS:
        val_dir = cfg.run_validation_dir(portfolio.source_portfolio_id,
                                         cfg.CURRENT_PERIOD.run_id)
        dashboard_path = next(iter(sorted(val_dir.glob("*_dashboard.json"))), None)
        canonical_v = next(iter(sorted(val_dir.glob("*_canonical_typed_canonical_violations.csv"))), None)
        business_v = next(iter(sorted(val_dir.glob("*_business_rules_violations.csv"))), None)

        dashboard = json.loads(dashboard_path.read_text(encoding="utf-8")) if dashboard_path else {}
        canonical_errors = 0
        canonical_warnings = 0
        if canonical_v is not None and canonical_v.exists():
            cv = pd.read_csv(canonical_v)
            if "severity" in cv.columns:
                canonical_errors = int((cv["severity"] == "error").sum())
                canonical_warnings = int((cv["severity"] == "warn").sum())
        business_rows: List[Dict[str, Any]] = []
        if business_v is not None and business_v.exists():
            bv = pd.read_csv(business_v)
            if "rule_id" in bv.columns:
                for rule, group in bv.groupby("rule_id"):
                    count = int(len(group))
                    rule_totals[str(rule)] = rule_totals.get(str(rule), 0) + count
                    business_rows.append({
                        "rule": str(rule),
                        "severity": str(group["severity"].iloc[0]) if "severity" in group else "",
                        "description": str(group["description"].iloc[0]) if "description" in group else "",
                        "rows": count,
                    })
                business_rows.sort(key=lambda r: -r["rows"])
                total_exceptions += int(len(bv))
        # The Gate 3b dashboard carries the exception summary, not a row count, so
        # take the row count from the portfolio's own canonical output — the tape
        # that was actually validated.
        canonical = (cfg.run_output_dir(portfolio.source_portfolio_id,
                                        cfg.CURRENT_PERIOD.run_id)
                     / f"{cfg.source_file(portfolio, cfg.CURRENT_PERIOD).stem}"
                       f"_canonical_typed.csv")
        rows = int(len(pd.read_csv(canonical, low_memory=False))) if canonical.exists() else 0
        total_rows += rows
        if canonical_errors:
            gate2_ok = False
        per_portfolio.append({
            "portfolio": portfolio.display_id,
            "label": portfolio.label,
            "rows": rows,
            "gate3bSummary": dashboard.get("summary") or {},
            "canonicalErrors": canonical_errors,
            "canonicalWarnings": canonical_warnings,
            "businessRuleExceptions": sum(r["rows"] for r in business_rows),
            "businessRules": business_rows,
        })

    return {
        "available": bool(per_portfolio),
        "kind": "validation_report",
        "title": "Validation report",
        "format": "CSV + JSON",
        "reportingDate": cfg.CURRENT_PERIOD.reporting_date,
        "rowsValidated": total_rows,
        "canonicalValidation": "Passed" if gate2_ok else "Exceptions raised",
        "businessRuleExceptions": total_exceptions,
        "exceptionRatePct": (round(total_exceptions / total_rows * 100, 3)
                            if total_rows else None),
        "rulesTriggered": sorted(
            ({"rule": r, "rows": n} for r, n in rule_totals.items()),
            key=lambda r: -r["rows"]),
        "perPortfolio": per_portfolio,
        "producedBy": "engine/gate_3_validation/* (Gates 2, 3, 3b)",
    }


# --------------------------------------------------------------------------- #
# 3. Investor deck
# --------------------------------------------------------------------------- #
def investor_deck(*, verbose: bool = True) -> Dict[str, Any]:
    """Generate the investor PPTX pack with the production deck builder.

    Published into the ``MI_AGENT_DECK_ROOT`` layout the MI API discovers
    (``{client}/latest|{period}/investor_pack.pptx`` plus the latest pointer), so
    the Copilot ``getLatestInvestorDeck`` action resolves it exactly as it would
    in production.
    """
    run_dir = _dated_platform_dir(cfg.CURRENT_PERIOD)
    prior_dir = _dated_platform_dir(cfg.PRIOR_PERIOD)
    if not (run_dir / "platform_canonical_typed.csv").exists():
        return {"available": False, "reason": "no assembled platform canonical for the current period"}

    work = cfg.artefact_dir() / "investor_deck"
    work.mkdir(parents=True, exist_ok=True)
    output = work / DECK_NAME

    env = dict(os.environ)
    env.update(cfg.mi_env())
    env["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = cfg.platform_prefix()

    cmd = [
        sys.executable, "-m", "mi_agent_pptx.cli",
        "--run-dir", str(run_dir),
        "--prior-run-dir", str(prior_dir),
        "--deck-config", str(cfg.REPO_ROOT / "configs" / "pptx" / "investor_pack.yaml"),
        "--client-name", cfg.CLIENT_DISPLAY_NAME,
        "--client-id", cfg.CLIENT_ID,
        "--run-id", cfg.CURRENT_PERIOD.reporting_date,
        "--as-of-date", cfg.CURRENT_PERIOD.reporting_date,
        "--output", str(output),
        "--consolidated",
        "--work-dir", str(work / "scratch"),
        "--repo-root", str(cfg.REPO_ROOT),
    ]
    proc = subprocess.run(cmd, cwd=str(cfg.REPO_ROOT), capture_output=True,
                          text=True, env=env)
    if proc.returncode != 0 or not output.exists():
        tail = "\n".join((proc.stdout or "").splitlines()[-12:])
        err = "\n".join((proc.stderr or "").splitlines()[-12:])
        return {
            "available": False,
            "reason": f"the deck builder exited {proc.returncode}",
            "stdoutTail": tail, "stderrTail": err,
        }

    # Publish into the deck root the MI API + Copilot action discover.
    deck_base = cfg.deck_root() / cfg.CLIENT_ID
    for folder in ("latest", cfg.CURRENT_PERIOD.period):
        target = deck_base / folder
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output, target / DECK_NAME)
    pointer = {
        "deck_name": DECK_NAME,
        "client_id": cfg.CLIENT_ID,
        "reporting_period": cfg.CURRENT_PERIOD.period,
        "generated_at": f"{cfg.CURRENT_PERIOD.reporting_date}T23:59:00Z",
        "synthetic_notice": cfg.SYNTHETIC_BANNER,
    }
    (deck_base / "latest" / POINTER_NAME).write_text(
        json.dumps(pointer, indent=2), encoding="utf-8")

    slides = _count_slides(output)
    if verbose:
        print(f"  [artefacts] investor deck: {slides} slides, "
              f"{output.stat().st_size / 1024:,.0f} KB")
    return {
        "available": True,
        "kind": "investor_deck",
        "title": "Investor pack",
        "fileName": DECK_NAME,
        "format": "PowerPoint (PPTX)",
        "slides": slides,
        "sizeBytes": output.stat().st_size,
        "contentSha256": _sha256(output),
        "reportingPeriod": cfg.CURRENT_PERIOD.period,
        "path": str(output),
        "publishedTo": str(deck_base / "latest" / DECK_NAME),
        "producedBy": "mi_agent_pptx/cli.py",
        "slideTitles": _slide_titles(output),
    }


def _count_slides(path: Path) -> Optional[int]:
    try:
        from pptx import Presentation
        return len(Presentation(str(path)).slides)
    except Exception:  # noqa: BLE001 - a missing count never blocks the artefact
        return None


def _slide_titles(path: Path) -> List[str]:
    try:
        from pptx import Presentation
        titles: List[str] = []
        for slide in Presentation(str(path)).slides:
            title = ""
            for shape in slide.shapes:
                if shape.has_text_frame and shape.text_frame.text.strip():
                    title = shape.text_frame.text.strip().splitlines()[0]
                    break
            titles.append(title[:80])
        return titles
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------- #
# 4. Risk monitoring
# --------------------------------------------------------------------------- #
def risk_monitor() -> Dict[str, Any]:
    """Capture the risk-limit monitoring output from its production service."""
    os.environ.update(cfg.mi_env())
    try:
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        resp = TestClient(app).get("/mi/risk-limits", params={
            "portfolioId": f"{cfg.CLIENT_ID}/{cfg.CURRENT_PERIOD.reporting_date}"})
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"risk service unavailable: {exc}"}
    if resp.status_code != 200:
        return {"available": False, "reason": f"/mi/risk-limits returned HTTP {resp.status_code}"}
    payload = resp.json()
    # The monitor returns one entry per limit TEST (`tests`), each with a RAG
    # status and the observed concentration; the earlier `limits` key does not
    # exist on this envelope.
    tests = payload.get("tests") or []
    summary = payload.get("summary") or {}
    statuses: Dict[str, int] = {}
    for row in tests if isinstance(tests, list) else []:
        status = str((row or {}).get("status") or "").lower()
        if status:
            statuses[status] = statuses.get(status, 0) + 1
    available = bool(payload.get("available")) and bool(tests)
    reason = None
    if not available:
        reason = (payload.get("limitsReason")
                  or "no concentration limits are configured for this client")
    elif not payload.get("fundedDataAvailable"):
        # Limits extracted but no funded data to test them against: report the
        # limits, and say plainly that the observed concentrations are absent.
        reason = "limits extracted; observed concentrations unavailable"
    return {
        "available": available,
        "kind": "risk_monitoring",
        "title": "Concentration risk monitor",
        "format": "JSON",
        "limitCount": len(tests) if isinstance(tests, list) else None,
        "testsPassed": summary.get("testsPassed"),
        "breaches": summary.get("breaches"),
        "unavailableTests": summary.get("unavailable"),
        "fundedDataAvailable": bool(payload.get("fundedDataAvailable")),
        "limitsSource": payload.get("limitsSource"),
        "largestConcentration": summary.get("largestConcentration"),
        "closestHeadroom": summary.get("closestHeadroom"),
        "statusCounts": statuses,
        "reportingDate": cfg.CURRENT_PERIOD.reporting_date,
        "producedBy": "mi_agent_api/risk_limits.py (mi_agent/risk_monitor)",
        "payload": payload,
        "reason": reason,
    }


# --------------------------------------------------------------------------- #
# 5. Regulatory output (attempted honestly; reported unavailable if it cannot run)
# --------------------------------------------------------------------------- #
def regulatory_output(*, verbose: bool = True) -> Dict[str, Any]:
    """Attempt the ESMA Annex 2 regime projection for one portfolio.

    The demonstration's recurring run is MI-mode, because ESMA delivery is a
    separately governed capability that requires a confirmed enum-review pass
    (``engine/enum_agent``) before a projection is allowed to proceed. This runs
    the projector so the outcome is a measured fact rather than an assumption; if
    it cannot complete, the artefact is reported as unavailable with the reason and
    the film does not show it.
    """
    portfolio = cfg.PORTFOLIO_A
    canonical = (cfg.run_output_dir(portfolio.source_portfolio_id,
                                    cfg.CURRENT_PERIOD.run_id)
                 / f"{cfg.source_file(portfolio, cfg.CURRENT_PERIOD).stem}_canonical_typed.csv")
    if not canonical.exists():
        return {"available": False, "reason": "no canonical output to project"}
    out_dir = cfg.artefact_dir() / "regulatory"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(cfg.REPO_ROOT / "engine" / "gate_4_projection" / "regime_projector.py"),
        str(canonical), "--regime", "ESMA_Annex2",
        "--registry", str(cfg.REPO_ROOT / "config" / "system" / "fields_registry.yaml"),
        "--enum-mapping", str(cfg.REPO_ROOT / "config" / "system" / "enum_mapping.yaml"),
        "--config", str(cfg.demo_client_config()),
        "--template-order", str(cfg.REPO_ROOT / "config" / "system" / "esma_code_order.yaml"),
        "--portfolio-type", "equity_release",
        "--output-dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(cfg.REPO_ROOT), capture_output=True, text=True)
    projected = next(iter(sorted(out_dir.glob("*_projected.csv"))), None)
    if proc.returncode != 0 or projected is None:
        reason = next((ln.strip() for ln in reversed((proc.stderr or "").splitlines())
                       if ln.strip()), f"projector exited {proc.returncode}")
        if verbose:
            print(f"  [artefacts] regulatory projection unavailable: {reason[:160]}")
        return {
            "available": False,
            "kind": "regulatory_output",
            "title": "ESMA Annex 2 exposure report",
            "reason": reason[:400],
            "note": "Excluded from the film: the regime projection requires a "
                    "confirmed enum-review pass that this demonstration does not "
                    "run.",
        }
    df = pd.read_csv(projected, low_memory=False)
    if verbose:
        print(f"  [artefacts] regulatory projection: {len(df):,} rows, "
              f"{len(df.columns)} fields")
    return {
        "available": True,
        "kind": "regulatory_output",
        "title": "ESMA Annex 2 exposure report",
        "fileName": projected.name,
        "format": "CSV (projection)",
        "rows": int(len(df)),
        "fields": int(len(df.columns)),
        "sizeBytes": projected.stat().st_size,
        "reportingDate": cfg.CURRENT_PERIOD.reporting_date,
        "producedBy": "engine/gate_4_projection/regime_projector.py",
    }


# --------------------------------------------------------------------------- #
# 6. Audit manifest
# --------------------------------------------------------------------------- #
def orchestration_from_disk() -> Dict[str, Any]:
    """Reconstruct the orchestration result from the manifests already written.

    A selective run (``--artefacts`` without ``--orchestrate``) has no in-process
    orchestration result, and an audit manifest that silently recorded zero
    assembled outputs would be worse than one that failed. The assembler and run
    manifests are on disk, so read them.
    """
    result: Dict[str, Any] = {"runs": [], "assemblies": [], "reconciliation": {}}
    for prd in cfg.PERIODS:
        mpath = _dated_platform_dir(prd) / "platform_canonical_manifest.json"
        if not mpath.exists():
            continue
        m = json.loads(mpath.read_text(encoding="utf-8"))
        result["assemblies"].append({
            "period": prd.period,
            "reporting_date": prd.reporting_date,
            "role": prd.role,
            "total_rows": m.get("total_rows"),
            "total_balance": m.get("output_total_balance"),
            "content_sha256": m.get("content_sha256"),
            "dated_uri": f"{cfg.platform_prefix()}/{prd.reporting_date}",
            "per_portfolio_balance": {
                p["source_portfolio_id"]: p["total_balance"]
                for p in m.get("portfolios", [])
            },
        })
    for portfolio in cfg.PORTFOLIOS:
        for prd in cfg.PERIODS:
            run_manifest = (cfg.run_output_dir(portfolio.source_portfolio_id, prd.run_id)
                            / "run_manifest.json")
            canonical = (cfg.run_output_dir(portfolio.source_portfolio_id, prd.run_id)
                         / f"{cfg.source_file(portfolio, prd).stem}_canonical_typed.csv")
            if not canonical.exists():
                continue
            df = pd.read_csv(canonical, low_memory=False)
            result["runs"].append({
                "portfolio": portfolio.display_id,
                "period": prd.period,
                "rows": int(len(df)),
                "total_outstanding_balance": round(float(pd.to_numeric(
                    df.get("current_outstanding_balance", pd.Series(dtype=float)),
                    errors="coerce").fillna(0.0).sum()), 2),
                "gate_summary": (
                    json.loads(run_manifest.read_text(encoding="utf-8")).get("gates", [])
                    if run_manifest.exists() else []),
            })
    by_period = {a["period"]: a for a in result["assemblies"]}
    cur = by_period.get(cfg.CURRENT_PERIOD.period)
    pri = by_period.get(cfg.PRIOR_PERIOD.period)
    if cur and pri:
        result["reconciliation"] = {
            "current_period": cur["period"],
            "prior_period": pri["period"],
            "current_total_balance": cur["total_balance"],
            "prior_total_balance": pri["total_balance"],
            "consolidated_movement": round(
                (cur["total_balance"] or 0.0) - (pri["total_balance"] or 0.0), 2),
            "per_portfolio_movement": {
                pid: round(cur["per_portfolio_balance"].get(pid, 0.0)
                           - pri["per_portfolio_balance"].get(pid, 0.0), 2)
                for pid in sorted(cur["per_portfolio_balance"])
            },
        }
    return result


def audit_manifest(orchestration_result: Dict[str, Any]) -> Dict[str, Any]:
    """A single provenance record covering the whole reporting cycle."""
    # A caller that has no in-process orchestration result still gets a complete
    # manifest, rather than one that quietly reports nothing was assembled.
    if not orchestration_result.get("assemblies"):
        orchestration_result = orchestration_from_disk()
    inputs: List[Dict[str, Any]] = []
    for portfolio in cfg.PORTFOLIOS:
        for prd in cfg.PERIODS:
            path = cfg.source_file(portfolio, prd)
            if path.exists():
                inputs.append({
                    "portfolio": portfolio.display_id,
                    "period": prd.period,
                    "fileName": path.name,
                    "sizeBytes": path.stat().st_size,
                    "contentSha256": _sha256(path),
                })
    gates: List[Dict[str, Any]] = []
    for run in orchestration_result.get("runs", []):
        gates.append({
            "portfolio": run["portfolio"],
            "period": run["period"],
            "rows": run["rows"],
            "balance": run["total_outstanding_balance"],
            "gates": run.get("gate_summary", []),
            "elapsedSeconds": run.get("elapsed_seconds"),
        })
    outputs = [
        {
            "period": a["period"],
            "reportingDate": a["reporting_date"],
            "rows": a["total_rows"],
            "balance": a["total_balance"],
            "contentSha256": a.get("content_sha256"),
            "uri": a["dated_uri"],
        }
        for a in orchestration_result.get("assemblies", [])
    ]
    return {
        "available": True,
        "kind": "audit_manifest",
        "title": "Audit manifest",
        "format": "JSON",
        "synthetic_notice": cfg.SYNTHETIC_BANNER,
        "clientId": cfg.CLIENT_ID,
        "clientDisplayName": cfg.CLIENT_DISPLAY_NAME,
        "reportingDate": cfg.CURRENT_PERIOD.reporting_date,
        "sourceFiles": inputs,
        "gateExecution": gates,
        "assembledOutputs": outputs,
        "reconciliation": orchestration_result.get("reconciliation", {}),
        "provenanceFields": [
            "source_portfolio_id", "source_portfolio_type", "source_portfolio_label",
            "acquisition_date", "seller_name", "portfolio_cohort",
        ],
        "producedBy": "demo_platform/artefacts.py from the run and assembler manifests",
    }


# --------------------------------------------------------------------------- #
# Build all
# --------------------------------------------------------------------------- #
def build(orchestration_result: Dict[str, Any], *, verbose: bool = True) -> Dict[str, Any]:
    """Produce every governed output and return the artefact catalogue."""
    cfg.artefact_dir().mkdir(parents=True, exist_ok=True)
    catalogue = {
        "synthetic_notice": cfg.SYNTHETIC_BANNER,
        "reportingDate": cfg.CURRENT_PERIOD.reporting_date,
        "canonicalTape": canonical_tape(),
        "validationReport": validation_report(),
        "investorDeck": investor_deck(verbose=verbose),
        "riskMonitor": risk_monitor(),
        "regulatoryOutput": regulatory_output(verbose=verbose),
        "auditManifest": audit_manifest(orchestration_result),
    }
    out = cfg.artefact_dir() / "artefact_catalogue.json"
    out.write_text(json.dumps(catalogue, indent=2, default=str), encoding="utf-8")
    if verbose:
        for key, art in catalogue.items():
            if isinstance(art, dict) and "available" in art:
                mark = "OK  " if art["available"] else "n/a "
                print(f"  [artefacts] {mark}{key}"
                      + ("" if art["available"] else f" — {str(art.get('reason'))[:100]}"))
    return catalogue


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse
    ap = argparse.ArgumentParser(description="Build the demo governed outputs.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    # Re-derive the orchestration result from the on-disk manifests.
    stub = orchestration_from_disk()
    build(stub, verbose=not args.quiet)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
