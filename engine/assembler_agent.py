"""engine.assembler_agent — Assembler Agent.

The Assembler Agent sits between the Onboarding Agent and the downstream
MI / Regime (Projection) Agents:

    Onboarding Agent  →  validated per-portfolio canonical files
                      →  Assembler Agent
                      →  central consolidated canonical
                      →  MI Agent / Regime Projection Agent / future consumers

Its role is deliberately narrow: **consolidate validated per-portfolio canonical
files into one central consolidated canonical, then route that central canonical
to the selected downstream pipeline.** It does not re-run onboarding, re-transform
raw data, change canonical derivation, MI calculations or ESMA Annex 2 logic.

The heavy lifting (discover → select latest per portfolio → composite-key dedup →
combine → write) is reused from :mod:`engine.platform_assembler`; this module adds
pipeline-scope awareness, run-level lineage metadata and downstream routing.

The central canonical is written as ``platform_canonical_typed.csv`` (the name the
MI data-source already resolves) and exposed here as the Assembler Agent's central
consolidated canonical.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    from engine import platform_assembler as _assembler
except ModuleNotFoundError:  # pragma: no cover - path bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from engine import platform_assembler as _assembler

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = _PROJECT_ROOT / "config"
_REGIME_PROJECTOR = _PROJECT_ROOT / "engine" / "gate_4_projection" / "regime_projector.py"

# Pipeline scopes the agent understands.
PIPELINE_MI = "mi"
PIPELINE_REGIME = "regime"
PIPELINE_SUBMISSION_PACK = "submission_pack"
PIPELINE_ELIGIBILITY = "eligibility"
PIPELINE_ALL = "all"

#: Accepted pipeline values (CLI choices).
VALID_PIPELINES = (
    PIPELINE_MI, PIPELINE_REGIME, PIPELINE_SUBMISSION_PACK,
    PIPELINE_ELIGIBILITY, PIPELINE_ALL,
)
#: Pipelines that are actually wired to a downstream consumer today.
IMPLEMENTED_PIPELINES = (PIPELINE_MI, PIPELINE_REGIME, PIPELINE_ALL)
#: Accepted-but-not-yet-routed scopes (the central canonical is still produced).
FUTURE_PIPELINES = (PIPELINE_SUBMISSION_PACK, PIPELINE_ELIGIBILITY)


class AssemblerAgentError(ValueError):
    """Raised for invalid Assembler Agent invocations."""


@dataclass
class AssemblerAgentResult:
    """Outcome of an Assembler Agent run."""

    central_canonical_path: Optional[Path]
    manifest: Dict[str, Any]
    routing: Dict[str, Any]
    pipeline: str
    regime: Optional[str] = None
    dataframe: Any = None
    regime_run: Optional[Dict[str, Any]] = None
    routes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Routing helpers
# --------------------------------------------------------------------------- #

def build_regime_command(
    central_canonical: Union[str, Path],
    out_dir: Union[str, Path],
    regime: str,
    *,
    registry: Optional[str] = None,
    enum_mapping: Optional[str] = None,
    config: Optional[str] = None,
    template_order: Optional[str] = None,
    portfolio_type: str = "equity_release",
    output_prefix: Optional[str] = None,
    allow_unreviewed: bool = False,
    python: Optional[str] = None,
) -> List[str]:
    """Build the command to run the EXISTING regime projector over the central
    canonical. Orchestration only — the projector logic is unchanged."""
    cmd = [
        python or sys.executable, str(_REGIME_PROJECTOR), str(central_canonical),
        "--regime", regime,
        "--registry", registry or str(_CONFIG_ROOT / "system" / "fields_registry.yaml"),
        "--enum-mapping", enum_mapping or str(_CONFIG_ROOT / "system" / "enum_mapping.yaml"),
        "--config", config or str(_CONFIG_ROOT / "client" / "config_client_ERM_UK.yaml"),
        "--template-order", template_order or str(_CONFIG_ROOT / "system" / "esma_code_order.yaml"),
        "--portfolio-type", portfolio_type,
        "--output-dir", str(out_dir),
    ]
    if output_prefix:
        cmd += ["--output-prefix", output_prefix]
    if allow_unreviewed:
        cmd += ["--allow-unreviewed"]
    return cmd


def _mi_routing(central_canonical: Path) -> Dict[str, Any]:
    return {
        "consumer": "mi_agent",
        "central_canonical": str(central_canonical),
        # The MI data-source already prefers a platform canonical when present.
        "data_source_env": {"MI_AGENT_PLATFORM_CANONICAL": str(central_canonical)},
        "note": "Point the MI Agent at this central canonical (or place it where "
                "MI auto-resolves it: MI_AGENT_PLATFORM_DIR / out_platform/).",
    }


def _regime_routing(central_canonical: Path, out_dir: Path, regime: str) -> Dict[str, Any]:
    return {
        "consumer": "regime_projection_agent",
        "central_canonical": str(central_canonical),
        "regime": regime,
        "input_canonical": str(central_canonical),
        "command": build_regime_command(central_canonical, out_dir, regime),
        "note": "Run the existing regime projector with this central canonical as "
                "input. ESMA output stays template-clean; the projector writes the "
                "provenance companion linking each row to source_portfolio_id / "
                "portfolio_cohort.",
    }


# --------------------------------------------------------------------------- #
# Agent API
# --------------------------------------------------------------------------- #

def run_assembler_agent(
    inputs: Union[str, Path, Sequence[Union[str, Path]]],
    out_dir: Union[str, Path],
    *,
    client_id: str,
    pipeline: str = PIPELINE_MI,
    regime: Optional[str] = None,
    assembler_run_id: Optional[str] = None,
    created_at: Optional[str] = None,
    run_regime: bool = False,
    regime_allow_unreviewed: bool = False,
) -> AssemblerAgentResult:
    """Consolidate per-portfolio canonicals and route to ``pipeline``.

    Returns an :class:`AssemblerAgentResult` carrying the central canonical path,
    the lineage manifest and downstream routing info.
    """
    pipeline = (pipeline or PIPELINE_MI).strip().lower()
    if pipeline not in VALID_PIPELINES:
        raise AssemblerAgentError(
            f"pipeline {pipeline!r} invalid; choose one of {VALID_PIPELINES}."
        )
    if pipeline == PIPELINE_REGIME and not regime:
        raise AssemblerAgentError("--regime is required for pipeline=regime.")

    created_at = created_at or datetime.now(timezone.utc).isoformat()
    if assembler_run_id is None:
        stamp = created_at.replace(":", "").replace("-", "").replace(".", "")[:15]
        assembler_run_id = f"asm_{client_id}_{stamp}"

    routes = [PIPELINE_MI, PIPELINE_REGIME] if pipeline == PIPELINE_ALL else [pipeline]

    manifest_extra = {
        "assembler_run_id": assembler_run_id,
        "client_id": client_id,
        "pipeline": pipeline,
        "regime": regime,
        "created_at": created_at,
        "lineage": "Onboarding Agent canonical outputs -> Assembler Agent central "
                   "canonical -> MI Agent / Regime Projection Agent",
        "downstream_routes": routes,
    }

    result = _assembler.assemble_platform_canonical(
        inputs, out_dir, write=True, manifest_extra=manifest_extra,
    )
    central = result.output_csv
    out_dir = Path(out_dir)

    routing: Dict[str, Any] = {}
    if PIPELINE_MI in routes:
        routing[PIPELINE_MI] = _mi_routing(central)
    if PIPELINE_REGIME in routes:
        routing[PIPELINE_REGIME] = _regime_routing(central, out_dir, regime)
    for fut in FUTURE_PIPELINES:
        if pipeline == fut:
            routing[fut] = {
                "consumer": fut,
                "central_canonical": str(central),
                "note": f"{fut!r} is an accepted future pipeline scope; the central "
                        f"canonical is produced but routing is not yet wired.",
            }

    regime_run: Optional[Dict[str, Any]] = None
    if run_regime and PIPELINE_REGIME in routes:
        cmd = build_regime_command(central, out_dir, regime,
                                   allow_unreviewed=regime_allow_unreviewed)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        regime_run = {
            "command": cmd,
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "stderr_tail": (proc.stderr or "")[-2000:],
        }

    return AssemblerAgentResult(
        central_canonical_path=central,
        manifest=result.manifest,
        routing=routing,
        pipeline=pipeline,
        regime=regime,
        dataframe=result.dataframe,
        regime_run=regime_run,
        routes=routes,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m engine.assembler_agent",
        description="Assembler Agent — consolidate validated per-portfolio "
                    "canonical files into one central consolidated canonical and "
                    "route it to a downstream pipeline (mi | regime). Reads "
                    "canonical outputs only; never re-runs onboarding.",
    )
    ap.add_argument("--client-id", required=True, help="Client id recorded in the manifest.")
    ap.add_argument("--pipeline", default=PIPELINE_MI, choices=list(VALID_PIPELINES),
                    help="Downstream pipeline scope. Wired: mi | regime | all. "
                         "Accepted future scopes: submission_pack | eligibility.")
    ap.add_argument("--regime", default=None,
                    help="Target regime for pipeline=regime (e.g. ESMA_Annex2).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", help="Directory scanned for *_canonical_typed.csv.")
    src.add_argument("--inputs", nargs="+", help="Explicit canonical output paths.")
    ap.add_argument("--out-dir", required=True, help="Assembler output directory.")
    ap.add_argument("--run-regime", action="store_true",
                    help="For pipeline=regime: also run the existing regime "
                         "projector over the central canonical (orchestration only).")
    ap.add_argument("--regime-allow-unreviewed", action="store_true",
                    help="Pass --allow-unreviewed to the regime projector.")
    args = ap.parse_args(argv)

    try:
        res = run_assembler_agent(
            args.root if args.root else args.inputs, args.out_dir,
            client_id=args.client_id, pipeline=args.pipeline, regime=args.regime,
            run_regime=args.run_regime,
            regime_allow_unreviewed=args.regime_allow_unreviewed,
        )
    except (_assembler.PlatformAssemblyError, AssemblerAgentError) as exc:
        print(f"[assembler-agent] ERROR: {exc}")
        return 2

    m = res.manifest
    print("=" * 64)
    print(f"Assembler Agent — run {m['assembler_run_id']}")
    print(f"  client: {m['client_id']}  pipeline: {res.pipeline}"
          + (f"  regime: {res.regime}" if res.regime else ""))
    print(f"  central canonical: {res.central_canonical_path}")
    print(f"  portfolios: {m['portfolio_count']}  rows: {m['total_rows']}"
          + (f"  balance: {m['output_total_balance']:,.2f}"
             if m.get("output_total_balance") is not None else ""))
    for p in m["included_portfolios"]:
        print(f"   - {p['source_portfolio_id']:14} {p['snapshot_date']}  "
              f"({p['row_count']} loans)  <- {Path(p['selected_canonical_path']).name}")
    if m["excluded_candidates"]:
        print("  excluded (older snapshots):")
        for e in m["excluded_candidates"]:
            print(f"   - {Path(e['path']).name}: {e['reason']}")
    print(f"  routes: {', '.join(res.routes)}")
    if res.regime_run is not None:
        print(f"  regime projector: {'OK' if res.regime_run['ok'] else 'FAILED'}")
    print("=" * 64)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
