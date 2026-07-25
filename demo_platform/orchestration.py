"""demo_platform.orchestration — Stage 2: the recurring reporting cycle.

SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.

This is the "run continuously" stage, and it is the real thing: for every
portfolio and every reporting period it invokes

    engine/orchestrator/trakt_run.py --mode mi

exactly as the managed service does — Gate 1 semantic alignment (with the
portfolio's approved onboarding-contract alias overlay), the canonical transform,
Gate 2 canonical validation, Gate 2.5 lineage, Gate 3 business rules and Gate 3b
aggregation — then consolidates the period's portfolio canonicals with

    engine/platform_assembler.py

and publishes the assembled platform canonical into the SAME dated layout
production uses::

    processed/platform/{client}/{YYYY-MM-DD}/platform_canonical_typed.csv   (dated cut)
    processed/platform/{client}/latest/platform_canonical_typed.csv         (current)

Because ``apps.blob_trigger_app.storage`` maps ``blob://`` URIs onto the
filesystem when the backend is ``file``, the MI Agent API then resolves this demo
platform through its normal production code path with no modification and no
Azure dependency (see :func:`demo_platform.config.mi_env`).

Nothing here re-implements pipeline logic. It is a driver.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from . import config as cfg

_TRAKT_RUN = cfg.REPO_ROOT / "engine" / "orchestrator" / "trakt_run.py"
_ASSEMBLER = cfg.REPO_ROOT / "engine" / "platform_assembler.py"

PLATFORM_CANONICAL_NAME = "platform_canonical_typed.csv"
PLATFORM_MANIFEST_NAME = "platform_canonical_manifest.json"


class OrchestrationError(RuntimeError):
    """Raised when a stage of the recurring run fails."""


def _run(cmd: Sequence[str], *, label: str, verbose: bool) -> str:
    """Run a subprocess from the repository root, surfacing failures loudly."""
    if verbose:
        print(f"    $ {label}")
    proc = subprocess.run(
        [str(c) for c in cmd], cwd=str(cfg.REPO_ROOT),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-25:])
        err = "\n".join((proc.stderr or "").splitlines()[-25:])
        raise OrchestrationError(
            f"{label} failed (exit {proc.returncode}).\n--- stdout ---\n{tail}\n"
            f"--- stderr ---\n{err}"
        )
    return proc.stdout or ""


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #
def reset(*, keep_source: bool = False, verbose: bool = True) -> None:
    """Remove all generated demo state so a run starts from a clean environment."""
    targets = [cfg.runs_dir(), cfg.local_blob_root(), cfg.deck_root(),
               cfg.artefact_dir(), cfg.export_dir(), cfg.onboarding_dir(),
               cfg.demo_root() / "scratch"]
    if not keep_source:
        targets.append(cfg.source_dir())
    for t in targets:
        if t.exists():
            shutil.rmtree(t)
            if verbose:
                print(f"  [reset] removed {t.relative_to(cfg.REPO_ROOT) if str(t).startswith(str(cfg.REPO_ROOT)) else t}")


# --------------------------------------------------------------------------- #
# Per-portfolio, per-period pipeline run
# --------------------------------------------------------------------------- #
def _canonical_path(portfolio: cfg.PortfolioSpec, prd: cfg.PeriodSpec) -> Path:
    out_dir = cfg.run_output_dir(portfolio.source_portfolio_id, prd.run_id)
    stem = cfg.source_file(portfolio, prd).stem
    return out_dir / f"{stem}_canonical_typed.csv"


def run_portfolio_period(portfolio: cfg.PortfolioSpec, prd: cfg.PeriodSpec, *,
                         verbose: bool = True) -> Dict[str, Any]:
    """Run the production MI pipeline for one portfolio / one reporting period."""
    source = cfg.source_file(portfolio, prd)
    if not source.exists():
        raise OrchestrationError(
            f"{portfolio.display_id} {prd.period}: source extract {source} not found."
        )
    out_dir = cfg.run_output_dir(portfolio.source_portfolio_id, prd.run_id)
    val_dir = cfg.run_validation_dir(portfolio.source_portfolio_id, prd.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[Any] = [
        sys.executable, _TRAKT_RUN,
        "--mode", "mi",
        "--input", source,
        "--master-config", cfg.demo_client_config(),
        "--extra-aliases-dir", cfg.demo_aliases_dir() / portfolio.source_portfolio_id,
        "--portfolio-type", "equity_release",
        "--source-portfolio-id", portfolio.source_portfolio_id,
        "--source-portfolio-type", portfolio.source_portfolio_type,
        "--source-portfolio-label", portfolio.label,
        "--out-dir", out_dir,
        "--validation-out-dir", val_dir,
    ]
    if portfolio.acquisition_date:
        cmd += ["--acquisition-date", portfolio.acquisition_date]
    if portfolio.seller_name:
        cmd += ["--seller-name", portfolio.seller_name]

    started = time.time()
    stdout = _run(cmd, label=f"trakt_run --mode mi  {portfolio.display_id} {prd.period}",
                  verbose=verbose)
    elapsed = round(time.time() - started, 2)

    canonical = _canonical_path(portfolio, prd)
    if not canonical.exists():
        raise OrchestrationError(
            f"{portfolio.display_id} {prd.period}: pipeline produced no canonical "
            f"output at {canonical}."
        )
    manifest_path = out_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    dashboard = next(iter(sorted(val_dir.glob("*_dashboard.json"))), None)
    validation = json.loads(dashboard.read_text(encoding="utf-8")) if dashboard else {}

    df = pd.read_csv(canonical, low_memory=False)
    balance = float(pd.to_numeric(
        df.get("current_outstanding_balance", pd.Series(dtype=float)),
        errors="coerce").fillna(0.0).sum())

    gate_lines = [ln.strip() for ln in stdout.splitlines()
                  if ln.strip().startswith("[Gate") or ln.strip().startswith("[MI]")]

    record = {
        "portfolio": portfolio.display_id,
        "source_portfolio_id": portfolio.source_portfolio_id,
        "source_portfolio_type": portfolio.source_portfolio_type,
        "period": prd.period,
        "reporting_date": prd.reporting_date,
        "run_id": prd.run_id,
        "source_file": str(source),
        "canonical_output": str(canonical),
        "rows": int(len(df)),
        "total_outstanding_balance": round(balance, 2),
        "elapsed_seconds": elapsed,
        "gate_summary": gate_lines,
        "run_manifest": manifest,
        "validation_dashboard": validation,
    }
    if verbose:
        print(f"    -> {record['rows']:,} loans, "
              f"£{balance / 1e6:,.2f}m, {elapsed}s")
    return record


# --------------------------------------------------------------------------- #
# Assembly + publication
# --------------------------------------------------------------------------- #
def assemble_period(prd: cfg.PeriodSpec, *, verbose: bool = True) -> Dict[str, Any]:
    """Assemble both portfolio canonicals for one period into the platform view.

    Publishes to the dated production layout under the demo's local blob store.
    """
    inputs = [_canonical_path(p, prd) for p in cfg.PORTFOLIOS]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise OrchestrationError(f"{prd.period}: missing portfolio canonicals {missing}.")

    dated_dir = (cfg.local_blob_root() / "processed" / "platform"
                 / cfg.CLIENT_ID / prd.reporting_date)
    dated_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[Any] = [sys.executable, _ASSEMBLER, "--inputs", *inputs,
                      "--out-dir", dated_dir]
    stdout = _run(cmd, label=f"platform_assembler  {prd.period}", verbose=verbose)

    out_csv = dated_dir / PLATFORM_CANONICAL_NAME
    out_manifest = dated_dir / PLATFORM_MANIFEST_NAME
    if not out_csv.exists():
        raise OrchestrationError(f"{prd.period}: assembler produced no {out_csv}.")
    manifest = json.loads(out_manifest.read_text(encoding="utf-8"))

    # Cross-check: the platform total must equal the sum of the portfolio totals.
    # The assembler concatenates without transformation, so any divergence here is
    # a real defect and must stop the run.
    per_portfolio = {p["source_portfolio_id"]: p["total_balance"]
                     for p in manifest.get("portfolios", [])}
    expected = round(sum(per_portfolio.values()), 2)
    actual = round(float(manifest.get("output_total_balance") or 0.0), 2)
    if abs(expected - actual) > 0.05:
        raise OrchestrationError(
            f"{prd.period}: platform assembly does not reconcile — portfolios sum "
            f"to {expected} but the platform canonical totals {actual}."
        )

    if verbose:
        print(f"    -> assembled {manifest['total_rows']:,} loans across "
              f"{manifest['portfolio_count']} portfolios, £{actual / 1e6:,.2f}m")

    return {
        "period": prd.period,
        "reporting_date": prd.reporting_date,
        "role": prd.role,
        "dated_uri": f"{cfg.platform_prefix()}/{prd.reporting_date}",
        "output_csv": str(out_csv),
        "output_manifest": str(out_manifest),
        "total_rows": int(manifest["total_rows"]),
        "portfolio_count": int(manifest["portfolio_count"]),
        "total_balance": actual,
        "per_portfolio_balance": per_portfolio,
        "composite_key": manifest.get("composite_key"),
        "content_sha256": manifest.get("content_sha256"),
        "assembler_stdout": [ln for ln in stdout.splitlines() if ln.strip()],
    }


def publish_latest(*, verbose: bool = True) -> Dict[str, Any]:
    """Point ``latest/`` at the CURRENT period's assembled platform canonical.

    This is the step that refreshes the MI Agent data source: the API resolves
    ``MI_AGENT_PLATFORM_URI`` → ``…/latest/platform_canonical_typed.csv``, so
    publishing here is what makes the new period the active reporting dataset.
    """
    prd = cfg.CURRENT_PERIOD
    base = cfg.local_blob_root() / "processed" / "platform" / cfg.CLIENT_ID
    src = base / prd.reporting_date
    dst = base / "latest"
    dst.mkdir(parents=True, exist_ok=True)
    for name in (PLATFORM_CANONICAL_NAME, PLATFORM_MANIFEST_NAME):
        if (src / name).exists():
            shutil.copyfile(src / name, dst / name)
    pointer = {
        "synthetic_notice": cfg.SYNTHETIC_BANNER,
        "client_id": cfg.CLIENT_ID,
        "reporting_date": prd.reporting_date,
        "period": prd.period,
        "run_id": prd.run_id,
        "platform_canonical": PLATFORM_CANONICAL_NAME,
        "source_dated_cut": f"{cfg.platform_prefix()}/{prd.reporting_date}",
    }
    (dst / "latest_platform_canonical.json").write_text(
        json.dumps(pointer, indent=2), encoding="utf-8")
    if verbose:
        print(f"  [publish] latest -> {prd.reporting_date} "
              f"({cfg.platform_prefix()}/latest)")
    return pointer


# --------------------------------------------------------------------------- #
# Full recurring cycle
# --------------------------------------------------------------------------- #
def run(*, verbose: bool = True) -> Dict[str, Any]:
    """Execute the whole recurring cycle for every portfolio and period."""
    runs: List[Dict[str, Any]] = []
    for prd in cfg.PERIODS:
        if verbose:
            print(f"  [orchestrate] {prd.label} ({prd.reporting_date})")
        for portfolio in cfg.PORTFOLIOS:
            runs.append(run_portfolio_period(portfolio, prd, verbose=verbose))

    assemblies = [assemble_period(prd, verbose=verbose) for prd in cfg.PERIODS]
    pointer = publish_latest(verbose=verbose)

    # Period-over-period reconciliation of the assembled platform totals, computed
    # from the assembler's own manifests (not recomputed from the tapes).
    by_period = {a["period"]: a for a in assemblies}
    cur, pri = by_period[cfg.CURRENT_PERIOD.period], by_period[cfg.PRIOR_PERIOD.period]
    movement = round(cur["total_balance"] - pri["total_balance"], 2)
    per_portfolio_movement = {
        pid: round(cur["per_portfolio_balance"].get(pid, 0.0)
                   - pri["per_portfolio_balance"].get(pid, 0.0), 2)
        for pid in sorted(set(cur["per_portfolio_balance"]) | set(pri["per_portfolio_balance"]))
    }
    if abs(sum(per_portfolio_movement.values()) - movement) > 0.05:
        raise OrchestrationError(
            "Per-portfolio movements do not sum to the consolidated movement: "
            f"{per_portfolio_movement} vs {movement}."
        )

    return {
        "synthetic_notice": cfg.SYNTHETIC_BANNER,
        "runs": runs,
        "assemblies": assemblies,
        "latest_pointer": pointer,
        "reconciliation": {
            "current_period": cur["period"],
            "prior_period": pri["period"],
            "current_total_balance": cur["total_balance"],
            "prior_total_balance": pri["total_balance"],
            "consolidated_movement": movement,
            "per_portfolio_movement": per_portfolio_movement,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse
    ap = argparse.ArgumentParser(description="Run the recurring demo orchestration.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run(verbose=not args.quiet)
    print(json.dumps(result["reconciliation"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
