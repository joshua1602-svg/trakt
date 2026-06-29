"""engine.orchestrator_agent.adapters — the agent boundary.

Each orchestration stage is a thin adapter that calls an EXISTING agent's
callable and reports a uniform :class:`StepResult` (ok / blocking / output /
manifest / readiness / blockers). The conductor depends only on the
:class:`AgentAdapters` interface, so:

  * production uses :class:`RealAgentAdapters` (wires the real onboarding /
    transformation / validation agents);
  * the provenance-stamp, Assembler and MI-routing stages are REAL on the base
    class (they are light and own no agent internals);
  * tests subclass :class:`AgentAdapters` and stub only onboard/transform/
    validate, inheriting the real stamp/assemble/route — so the conductor logic
    AND the real consolidation/routing are exercised deterministically.

No agent internals, canonical/MI calculations or Regime logic are modified here.
"""

from __future__ import annotations

import glob
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from engine import provenance as _provenance
from engine import assembler_agent as _assembler_agent


@dataclass
class PortfolioSpec:
    """Operator-supplied identity + input for one source portfolio."""

    source_portfolio_id: str
    input: str
    source_portfolio_type: Optional[str] = None
    source_portfolio_label: Optional[str] = None
    acquisition_date: Optional[str] = None
    seller_name: Optional[str] = None
    allow_unknown_acquisition_date: bool = False


@dataclass
class StepResult:
    ok: bool
    blocking: bool = False
    output_path: Optional[str] = None
    manifest_path: Optional[str] = None
    readiness: Dict[str, Any] = field(default_factory=dict)
    blockers: List[str] = field(default_factory=list)
    message: str = ""


class AgentAdapters:
    """Stage boundary. Real onboarding/transform/validate are abstract; the
    provenance / assemble / route stages are concrete (and reused by tests)."""

    # -- agent stages (overridden by RealAgentAdapters / test stubs) -------- #
    def onboard(self, spec: PortfolioSpec, work_dir: Path) -> StepResult:
        raise NotImplementedError

    def transform(self, spec: PortfolioSpec, handoff_manifest: str, work_dir: Path) -> StepResult:
        raise NotImplementedError

    def validate(self, spec: PortfolioSpec, transformation_manifest: str, work_dir: Path) -> StepResult:
        raise NotImplementedError

    # -- orchestration-owned stages (REAL) ---------------------------------- #
    def stamp_provenance(self, spec: PortfolioSpec, validated_csv: str,
                         out_dir: Path) -> StepResult:
        """Assign + stamp this portfolio's provenance onto its validated canonical.

        Provenance is run-level metadata the orchestrator owns (it knows each
        portfolio's id), so stamping here keeps the agents untouched and
        guarantees the Assembler receives a fully-provenanced canonical. Output
        is ``<source_portfolio_id>_canonical_typed.csv`` (the name the Assembler
        discovers)."""
        try:
            prov = _provenance.build_provenance(
                source_portfolio_id=spec.source_portfolio_id,
                source_portfolio_type=spec.source_portfolio_type,
                source_portfolio_label=spec.source_portfolio_label,
                acquisition_date=spec.acquisition_date,
                seller_name=spec.seller_name,
                allow_unknown_acquisition_date=spec.allow_unknown_acquisition_date,
            )
        except _provenance.ProvenanceError as exc:
            # A provenance config error is a blocking gate (operator must fix the
            # portfolio spec), not a silent default.
            return StepResult(ok=False, blocking=True,
                              blockers=[f"provenance: {exc}"],
                              message=str(exc))
        df = pd.read_csv(validated_csv, low_memory=False)
        _provenance.stamp_dataframe(df, prov)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{spec.source_portfolio_id}_canonical_typed.csv"
        df.to_csv(out_csv, index=False)
        return StepResult(ok=True, output_path=str(out_csv),
                          readiness={"provenance": prov.to_dict()},
                          message=f"stamped {len(df)} rows for {spec.source_portfolio_id}")

    def assemble(self, stamped_paths: Sequence[str], out_dir: Path,
                 client_id: str, target: str, *, regime: Optional[str] = None) -> StepResult:
        """Consolidate the per-portfolio validated canonicals into one central
        canonical via the Assembler Agent."""
        try:
            res = _assembler_agent.run_assembler_agent(
                [str(p) for p in stamped_paths], str(out_dir),
                client_id=client_id,
                pipeline=("regime" if target == "regime" else
                          "all" if target == "all" else "mi"),
                regime=regime,
            )
        except Exception as exc:  # surfaced as a hard failure for this run
            return StepResult(ok=False, blocking=True,
                              blockers=[f"assembler: {exc}"], message=str(exc))
        return StepResult(
            ok=True, output_path=str(res.central_canonical_path),
            readiness={"portfolio_count": res.manifest.get("portfolio_count"),
                       "total_rows": res.manifest.get("total_rows"),
                       "routing": list(res.routing.keys())},
            message=f"central canonical: {res.central_canonical_path}",
        )

    def route_mi(self, central_canonical: str) -> StepResult:
        """Point the MI Agent at the central canonical (data-source resolution
        already prefers a platform canonical when present)."""
        return StepResult(
            ok=True, output_path=central_canonical,
            readiness={"MI_AGENT_PLATFORM_CANONICAL": central_canonical},
            message="MI Agent will resolve the central canonical "
                    "(set MI_AGENT_PLATFORM_CANONICAL or place under out_platform/).",
        )

    def project(self, central_canonical: str, out_dir: Path,
                regime: str) -> StepResult:
        """Project the central canonical to the regime (ESMA Annex 2) via the
        EXISTING regime projector — orchestration only, projector unchanged. The
        ESMA output stays template-clean; the projector emits the provenance
        companion linking each row to source_portfolio_id / portfolio_cohort."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = _assembler_agent.build_regime_command(
            central_canonical, out_dir, regime,
            output_prefix="central", allow_unreviewed=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        projected = glob.glob(str(out_dir / f"*{regime}_projected.csv"))
        companion = glob.glob(str(out_dir / f"*{regime}_provenance.csv"))
        ok = proc.returncode == 0 and bool(projected)
        if not ok:
            return StepResult(
                ok=False, blocking=True,
                blockers=[f"regime projection failed (rc={proc.returncode})"],
                message=(proc.stderr or "")[-1500:])
        return StepResult(
            ok=True, output_path=projected[0],
            readiness={"projected_csv": projected[0],
                       "provenance_companion": (companion[0] if companion else None),
                       "regime": regime},
            message=f"projected → {projected[0]}")


class RealAgentAdapters(AgentAdapters):
    """Wires the real Onboarding / Transformation / Validation agents."""

    def __init__(self, *, registry: Optional[str] = None,
                 client_name: Optional[str] = None,
                 onboarding_mode: str = "mi_only",
                 aliases_dir: str = "config/system",
                 processing_mode: str = "source_onboarding",
                 mapping_config_path: Optional[str] = None):
        self.registry = registry
        self.client_name = client_name
        self.onboarding_mode = onboarding_mode
        self.aliases_dir = aliases_dir
        # processing_mode is the discovery lever the blob trigger sets:
        #   "source_onboarding" — run source discovery/mapping (new/changed source);
        #   "deterministic"     — skip discovery, apply the saved approved mapping.
        self.processing_mode = processing_mode
        self.mapping_config_path = mapping_config_path

    def onboard(self, spec: PortfolioSpec, work_dir: Path) -> StepResult:
        """Run onboarding for one portfolio. The ``mode`` is the MI-vs-regime
        field-requirement lever and selects the output:

          * ``mi_only`` → builds the lean central lender tape (18_…), returned as
            the MI canonical (``output_path``);
          * ``regulatory_mi`` → builds the governed handoff package (24_…) with
            the full ESMA Annex 2 target contract, returned for the Transformation
            Agent (``manifest_path``), gated on ready_for_transformation_validation.
        """
        import json as _json
        from engine.onboarding_agent import workflow as _wf, storage_paths, central_tape_builder

        project_dir = Path(work_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        # Deterministic processing (known source) skips source discovery / mapping
        # review and applies the saved approved decisions; source onboarding (new /
        # changed source) runs discovery. Both stay deterministic-first (LLM off).
        deterministic = self.processing_mode == "deterministic"
        _wf.run_operator_workflow(
            input_dir=spec.input,
            client_name=self.client_name or spec.source_portfolio_id,
            client_id=spec.source_portfolio_id, run_id="run",
            project_dir=str(project_dir), mode=self.onboarding_mode,
            registry=self.registry or "config/system/fields_registry.yaml",
            aliases_dir=self.aliases_dir,
            enable_mapping_review=not deterministic,
            target_first_decisions=(self.mapping_config_path if deterministic else ""))

        if self.onboarding_mode == "mi_only":
            # MI path: build + return the central lender tape (the MI canonical).
            run_paths = storage_paths.resolve_run_paths(
                project_dir=str(project_dir), input_dir=spec.input, output_root=None,
                client_id=spec.source_portfolio_id, run_id="run",
                storage_backend="local", input_uri="", output_uri="")
            res = central_tape_builder.build_central_tapes(
                str(project_dir), run_paths,
                self.registry or "config/system/fields_registry.yaml", mode="mi_only")
            tape = res.get("central_lender_tape_path")
            ok = bool(res.get("central_lender_tape_created")) and bool(tape)
            return StepResult(
                ok=ok, blocking=not ok, output_path=tape,
                readiness={"central_lender_tape": tape, "loan_count": res.get("loan_count")},
                blockers=[] if ok else ["onboarding did not produce a central lender tape"],
                message=f"central tape: {tape}")

        # Regulatory path: the governed handoff package.
        handoff = project_dir / "output" / "handoff" / "24_onboarding_handoff_manifest.json"
        if not handoff.exists():
            return StepResult(ok=False, blocking=True,
                              blockers=["onboarding did not produce a handoff package "
                                        "(24_onboarding_handoff_manifest.json)"],
                              message="no handoff manifest")
        manifest = _json.loads(handoff.read_text(encoding="utf-8"))
        ready = bool(manifest.get("ready_for_transformation_validation"))
        return StepResult(
            ok=ready, blocking=not ready, manifest_path=str(handoff),
            readiness={k: manifest.get(k) for k in
                       ("ready_for_transformation_validation", "ready_for_projection",
                        "target_contract_id")},
            blockers=[] if ready else ["onboarding not ready_for_transformation_validation "
                                       "(mapping review / blocking decision pending)"],
            message=f"handoff: {handoff}")

    def transform(self, spec: PortfolioSpec, handoff_manifest: str, work_dir: Path) -> StepResult:
        from engine.transformation_agent.transformation_agent import build_transformation_package
        res = build_transformation_package(handoff_manifest,
                                           registry_path=self.registry)
        m = res["manifest"]
        ready = bool(m.get("ready_for_validation"))
        return StepResult(
            ok=ready, blocking=not ready,
            output_path=str(Path(res["transformation_dir"]) / "31_transformed_canonical_tape.csv"),
            manifest_path=str(Path(res["transformation_dir"]) / "30_transformation_manifest.json"),
            readiness={k: m.get(k) for k in ("ready_for_validation", "ready_for_projection",
                                             "ready_for_xml_delivery", "issue_count")},
            blockers=[] if ready else ["transformation not ready_for_validation"],
            message=f"transformation: {res['transformation_dir']}")

    def validate(self, spec: PortfolioSpec, transformation_manifest: str, work_dir: Path) -> StepResult:
        from engine.validation_agent.validation_agent import build_validation_package
        res = build_validation_package(transformation_manifest, registry_path=self.registry)
        m = res["manifest"]
        ready = bool(m.get("ready_for_validation_complete"))
        # The validated canonical is the transformed tape gated by validation.
        tx_dir = Path(transformation_manifest).parent
        validated_csv = tx_dir / "31_transformed_canonical_tape.csv"
        return StepResult(
            ok=ready, blocking=not ready, output_path=str(validated_csv),
            manifest_path=str(res.get("validation_manifest_path")
                              or (Path(res.get("validation_dir", tx_dir)) / "40_validation_manifest.json")),
            readiness={k: m.get(k) for k in ("ready_for_validation_complete",
                                             "ready_for_projection", "ready_for_xml_delivery")},
            blockers=[] if ready else ["validation has blocking exceptions "
                                       "(not ready_for_validation_complete)"],
            message="validation complete" if ready else "validation blocked")
