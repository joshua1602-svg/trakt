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


class RealAgentAdapters(AgentAdapters):
    """Wires the real Onboarding / Transformation / Validation agents."""

    def __init__(self, *, registry: Optional[str] = None,
                 client_name: Optional[str] = None,
                 onboarding_mode: str = "mi_only",
                 aliases_dir: str = "config/system"):
        self.registry = registry
        self.client_name = client_name
        self.onboarding_mode = onboarding_mode
        self.aliases_dir = aliases_dir

    def onboard(self, spec: PortfolioSpec, work_dir: Path) -> StepResult:
        """Run onboarding (pack → promote → handoff) for one portfolio and
        return the 24_onboarding_handoff_manifest, gated on
        ``ready_for_transformation_validation``."""
        from engine.onboarding_agent.onboarding_orchestrator import run_onboarding
        from engine.onboarding_agent import storage_paths, central_tape_builder, onboarding_handoff
        import json as _json

        project_dir = work_dir / spec.source_portfolio_id
        project_dir.mkdir(parents=True, exist_ok=True)
        run_onboarding(
            input_dir=spec.input,
            client_name=self.client_name or spec.source_portfolio_id,
            output_dir=str(project_dir),
            registry_path=self.registry or "config/system/fields_registry.yaml",
            aliases_dir=self.aliases_dir,
            mode=self.onboarding_mode,
        )
        run_paths = storage_paths.resolve_run_paths(
            project_dir=str(project_dir), input_dir=spec.input, output_root=None,
            client_id=spec.source_portfolio_id, run_id="run", storage_backend="local",
            input_uri="", output_uri="")
        central_tape_builder.build_central_tapes(
            str(project_dir), run_paths, self.registry or "config/system/fields_registry.yaml",
            mode=self.onboarding_mode)
        handoff = onboarding_handoff.build_handoff_package(
            str(project_dir), Path(run_paths.output_root),
            client_id=spec.source_portfolio_id, client_name=self.client_name or spec.source_portfolio_id,
            run_id="run", mode=self.onboarding_mode,
            registry=self.registry or "config/system/fields_registry.yaml")
        manifest_path = handoff["manifest_json_path"]
        ready = bool(handoff["manifest"].get("ready_for_transformation_validation"))
        return StepResult(
            ok=ready, blocking=not ready, manifest_path=manifest_path,
            readiness={k: handoff["manifest"].get(k) for k in
                       ("ready_for_transformation_validation", "ready_for_projection")},
            blockers=[] if ready else ["onboarding not ready_for_transformation_validation "
                                       "(mapping review / blocking decision pending)"],
            message=f"handoff: {manifest_path}")

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
