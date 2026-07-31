"""operations_control.occ_agent.execution — the synthetic execution adapter.

The synthetic run is driven by the **real** orchestration conductor
(:func:`engine.orchestrator_agent.orchestrator.run_orchestration`) over an
adapter that subclasses the **real** agent seam
(:class:`engine.orchestrator_agent.adapters.AgentAdapters`). The conductor's gate
sequence, halt semantics and resumable state are therefore the production ones —
this module supplies the stage bodies, not the sequencing.

What each stage actually does:

``onboard``
    Real components: :func:`engine.onboarding_agent.file_profiler.profile_file`
    for source inspection, and :class:`engine.gate_1_alignment.
    semantic_alignment.HeaderMapper` — the platform's own tiered alias/fuzzy
    mapper, loaded from the real field registry and alias directory — for header
    mapping. Columns whose mapping is ambiguous or below the configured
    confidence produce a pending-decision artefact in the **existing**
    ``34_target_first_decisions.yaml`` shape, so
    :func:`operations_control.adapters.extract_mapping_decisions` reads them
    unchanged and the halt behaves like any other governed halt.

``transform``
    Real canonical typing: :func:`engine.gate_2_transform.canonical_transform.
    apply_types` against the field registry.

``validate``
    Real validation: canonical core-required checks from
    :mod:`engine.gate_3_validation.validate_canonical`, business rules from
    :func:`engine.gate_3_validation.validate_business_rules.run_rules`, and the
    platform's own materiality assessment
    (:func:`engine.gate_3_validation.aggregate_validation_results.aggregate`,
    driven by ``config/asset/issue_policy.yaml``). A BLOCKING materiality halts
    the run; nothing here can downgrade it.

``stamp_provenance`` / ``assemble`` / ``route_mi``
    **Not overridden.** The base class implementations are the real ones — real
    provenance stamping and the real Assembler Agent
    (:func:`engine.assembler_agent.run_assembler_agent`) — and they are local and
    filesystem-only, so they run for real against the case sandbox.

``project``
    Overridden. The regime projector is invoked live as a subprocess by the base
    class; here the intended command is built with the real
    :func:`engine.assembler_agent.build_regime_command` (contract validation),
    recorded, and returned as a clearly-labelled **simulated** result. It never
    claims a projection ran.

Every stage records which of the five §12 outcomes it reached, so the case and
the readiness package can distinguish "deterministic execution completed" from
"execution simulated" instead of presenting them as the same thing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import yaml

from engine.orchestrator_agent.adapters import (
    AgentAdapters,
    PortfolioSpec,
    StepResult,
)

from ..engine import OpsError
from .run import (
    STAGE_CONTRACT_VALIDATED,
    STAGE_DETERMINISTIC_COMPLETED,
    STAGE_HARD_BLOCKED,
    STAGE_HUMAN_INPUT_REQUIRED,
    STAGE_SIMULATED,
)
from .policy import CAP_LIVE_PIPELINE_TRIGGER, SyntheticPolicy

REPO = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO / "config/system/fields_registry.yaml"
ALIASES_DIR = REPO / "config/system"
ISSUE_POLICY_PATH = REPO / "config/asset/issue_policy.yaml"

#: The existing pending-decision artefact the OCC already reads. Reused verbatim
#: so a synthetic halt and a live halt present identically.
DECISIONS_FILE = "34_target_first_decisions.yaml"

#: Confidence at or above which a deterministic header match stands without a
#: human. Mirrors ``demo_platform.onboarding.LOW_CONFIDENCE`` and the mapper's
#: own reporting threshold, so synthetic and demonstration agree with the engine.
LOW_CONFIDENCE = 0.90

#: Mapper tiers that are exact or contract-backed. A match at one of these is
#: not "inferred" whatever its numeric confidence.
_TRUSTED_TIERS = frozenset({"exact", "normalized", "alias"})


class SyntheticExecutionError(OpsError):
    """The synthetic run could not be prepared or completed."""

    def __init__(self, detail: str):
        super().__init__("OCC_AGENT_SYNTHETIC_RUN_FAILED",
                         f"The synthetic onboarding could not run. ({detail})",
                         http_status=500)


@dataclass
class StageRecord:
    """What one stage did, and how honestly it did it."""

    stage: str
    outcome: str                            # one of the §12 STAGE_* outcomes
    summary: str = ""
    component: str = ""                     # the real component that ran
    metrics: Dict[str, Any] = field(default_factory=dict)
    blockers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "outcome": self.outcome,
                "summary": self.summary, "component": self.component,
                "metrics": self.metrics, "blockers": self.blockers}


class SyntheticOnboardingAdapters(AgentAdapters):
    """Stage bodies for a synthetic run. Sequencing stays with the conductor."""

    def __init__(self, *, artefact_paths: Sequence[Path],
                 policy: SyntheticPolicy, sandbox: Path,
                 asset_type: str = "equity_release",
                 regime: str = "",
                 registry_path: Path = REGISTRY_PATH,
                 aliases_dir: Path = ALIASES_DIR,
                 issue_policy_path: Path = ISSUE_POLICY_PATH,
                 approved_mappings: Optional[Dict[str, str]] = None,
                 case_id: str = "", tenant: str = ""):
        self.artefact_paths = [Path(p) for p in artefact_paths]
        self.policy = policy
        self.sandbox = Path(sandbox).resolve()
        self.asset_type = asset_type
        self.regime = regime
        self.registry_path = Path(registry_path)
        self.aliases_dir = Path(aliases_dir)
        self.issue_policy_path = Path(issue_policy_path)
        #: source column -> canonical field, from human-approved decisions.
        self.approved_mappings = dict(approved_mappings or {})
        self.case_id = case_id
        self.tenant = tenant
        self.records: List[StageRecord] = []
        self.mapping_report: List[Dict[str, Any]] = []
        self.validation_report: List[Dict[str, Any]] = []
        self._assert_inside_sandbox()

    def _assert_inside_sandbox(self) -> None:
        """Every input must already live inside the case sandbox.

        The adapter is handed paths rather than reading a directory, so this is
        the point at which "the agent has no unrestricted filesystem access"
        stops being a claim and becomes a check.
        """
        for path in self.artefact_paths:
            resolved = Path(path).resolve()
            if self.sandbox not in resolved.parents:
                raise SyntheticExecutionError(
                    "an input file is outside the synthetic case sandbox")

    def _record(self, record: StageRecord) -> StageRecord:
        self.records.append(record)
        return record

    # ------------------------------------------------------------------ #
    # onboard — real profiling + real header mapping
    # ------------------------------------------------------------------ #
    def onboard(self, spec: PortfolioSpec, work_dir: Path) -> StepResult:
        from engine.gate_1_alignment.semantic_alignment import (
            HeaderMapper,
            load_aliases_from_dir,
            load_field_registry,
            select_registry_fields,
        )
        from engine.onboarding_agent.file_profiler import profile_file

        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        if not self.artefact_paths:
            self._record(StageRecord(
                stage="onboard", outcome=STAGE_HARD_BLOCKED,
                summary="No source files were provided.",
                blockers=["No source files were provided."]))
            return StepResult(ok=False, blocking=True,
                              blockers=["No source files were provided."],
                              message="no inputs")

        registry = load_field_registry(self.registry_path)
        canonical_fields = select_registry_fields(registry, self.asset_type)
        if not canonical_fields:
            raise SyntheticExecutionError(
                f"no canonical fields are registered for '{self.asset_type}'")
        alias_map = load_aliases_from_dir(self.aliases_dir)
        mapper = HeaderMapper(canonical_fields, alias_map)

        # 1. Real source profiling, per file.
        profiles: Dict[str, List[Dict[str, Any]]] = {}
        for path in self.artefact_paths:
            try:
                profiles[path.name] = [p.__dict__ if hasattr(p, "__dict__")
                                       else dict(p) for p in profile_file(path)]
            except Exception as exc:  # noqa: BLE001 — a finding, not a crash
                profiles[path.name] = []
                self.mapping_report.append({
                    "source_file": path.name, "source_column": "",
                    "canonical_field": "", "tier": "unreadable",
                    "confidence": 0.0,
                    "note": f"could not be profiled ({type(exc).__name__})"})

        # 2. Real header mapping, per column.
        primary = self._primary_tape()
        frame = _read_table(primary)
        decisions: List[Dict[str, Any]] = []
        resolved: Dict[str, str] = {}
        for column in [str(c) for c in frame.columns]:
            approved = self.approved_mappings.get(column)
            if approved is not None:
                if approved and approved != "__ignore__":
                    resolved[column] = approved
                self.mapping_report.append({
                    "source_file": primary.name, "source_column": column,
                    "canonical_field": approved, "tier": "operator_approved",
                    "confidence": 1.0, "note": "confirmed by an operator"})
                continue
            canonical, tier, confidence = mapper.map_one(column)
            trusted = tier in _TRUSTED_TIERS or float(confidence) >= LOW_CONFIDENCE
            self.mapping_report.append({
                "source_file": primary.name, "source_column": column,
                "canonical_field": canonical or "", "tier": tier,
                "confidence": round(float(confidence), 4),
                "note": "" if trusted else "below the confidence threshold"})
            if canonical and trusted:
                resolved[column] = canonical
            elif canonical:
                decisions.append(_mapping_decision(
                    column, canonical, tier, float(confidence),
                    frame[column], primary.name))

        # 3. A canonical field claimed by two columns is an ambiguity a human
        #    must settle — the engine has no basis to prefer one.
        for canonical, columns in _duplicates(resolved).items():
            for column in columns:
                resolved.pop(column, None)
            decisions.append(_ambiguity_decision(canonical, columns, frame,
                                                 primary.name))

        if decisions:
            _write_decisions(work_dir, decisions)
            self._record(StageRecord(
                stage="onboard", outcome=STAGE_HUMAN_INPUT_REQUIRED,
                component="engine.gate_1_alignment.semantic_alignment."
                          "HeaderMapper",
                summary=f"{len(decisions)} column"
                        f"{'s' if len(decisions) != 1 else ''} need your "
                        "confirmation before the data can be used.",
                metrics={"columns": len(frame.columns),
                         "mapped": len(resolved),
                         "needing_confirmation": len(decisions)}))
            return StepResult(
                ok=False, blocking=True,
                blockers=["onboarding not ready_for_transformation_validation "
                          "(mapping review / blocking decision pending)"],
                message="mapping review pending")

        # 4. Build the mapped tape + the handoff manifest.
        mapped = frame.rename(columns=resolved)
        mapped = mapped[[c for c in mapped.columns if c in set(resolved.values())]]
        tape = work_dir / "18_central_lender_tape.csv"
        mapped.to_csv(tape, index=False)
        handoff = work_dir / "24_onboarding_handoff_manifest.json"
        handoff.write_text(json.dumps({
            "ready_for_transformation_validation": True,
            "ready_for_projection": bool(self.regime),
            "target_contract_id": ("regulatory_mi" if self.regime
                                   else "mi_semantics"),
            "source_files": [p.name for p in self.artefact_paths],
            "mapped_columns": len(resolved),
            "canonical_tape": str(tape),
            "runtime_mode": "synthetic",
        }, indent=2), encoding="utf-8")
        (work_dir / "source_profiles.json").write_text(
            json.dumps(profiles, indent=2, default=str), encoding="utf-8")

        self._record(StageRecord(
            stage="onboard", outcome=STAGE_DETERMINISTIC_COMPLETED,
            component="engine.onboarding_agent.file_profiler + "
                      "engine.gate_1_alignment.semantic_alignment.HeaderMapper",
            summary=f"Read {len(self.artefact_paths)} file"
                    f"{'s' if len(self.artefact_paths) != 1 else ''} and "
                    f"matched {len(resolved)} columns.",
            metrics={"rows": int(len(mapped)), "mapped_columns": len(resolved),
                     "source_columns": int(len(frame.columns))}))
        return StepResult(ok=True, output_path=str(tape),
                          manifest_path=str(handoff),
                          readiness={"loan_count": int(len(mapped)),
                                     "target_contract": "mi_semantics"},
                          message=f"central tape: {tape}")

    # ------------------------------------------------------------------ #
    # transform — real canonical typing
    # ------------------------------------------------------------------ #
    def transform(self, spec: PortfolioSpec, handoff_manifest: str,
                  work_dir: Path) -> StepResult:
        from engine.gate_2_transform.canonical_transform import (
            apply_types,
            load_registry,
            select_fields_for_portfolio,
        )
        work_dir = Path(work_dir)
        manifest = json.loads(Path(handoff_manifest).read_text(encoding="utf-8"))
        tape = Path(manifest["canonical_tape"])
        frame = pd.read_csv(tape, low_memory=False)

        registry = load_registry(self.registry_path)
        fields_meta = select_fields_for_portfolio(
            registry, spec.source_portfolio_type or "direct")
        try:
            # apply_types types the frame IN PLACE and returns a per-field
            # report; the typed data is `frame` itself.
            report = apply_types(frame, fields_meta)
        except Exception as exc:  # noqa: BLE001 — surfaced as a hard failure
            self._record(StageRecord(
                stage="transform", outcome=STAGE_HARD_BLOCKED,
                summary="The data could not be typed.",
                blockers=[f"canonical typing failed ({type(exc).__name__})"]))
            return StepResult(ok=False, blocking=True,
                              blockers=[f"canonical typing failed "
                                        f"({type(exc).__name__})"],
                              message="transform failed")
        typed = frame
        typed_fields = report.get("fields") or {}
        parse_failures = {name: spec.get("parse_failures", 0)
                          for name, spec in typed_fields.items()
                          if spec.get("parse_failures")}

        out_dir = work_dir / "transformation"
        out_dir.mkdir(parents=True, exist_ok=True)
        typed_csv = out_dir / "31_transformed_canonical_tape.csv"
        typed.to_csv(typed_csv, index=False)
        manifest_path = out_dir / "30_transformation_manifest.json"
        manifest_path.write_text(json.dumps({
            "ready_for_validation": True,
            "typed_columns": sorted(typed_fields),
            "parse_failures": parse_failures,
            "row_count": int(len(typed)),
            "runtime_mode": "synthetic",
        }, indent=2, default=str), encoding="utf-8")

        self._record(StageRecord(
            stage="transform", outcome=STAGE_DETERMINISTIC_COMPLETED,
            component="engine.gate_2_transform.canonical_transform.apply_types",
            summary=f"Typed {len(typed_fields)} canonical column"
                    f"{'s' if len(typed_fields) != 1 else ''} across "
                    f"{len(typed)} records.",
            metrics={"rows": int(len(typed)),
                     "typed_columns": len(typed_fields),
                     "parse_failures": parse_failures}))
        return StepResult(ok=True, output_path=str(typed_csv),
                          manifest_path=str(manifest_path),
                          readiness={"ready_for_validation": True},
                          message="transformed")

    # ------------------------------------------------------------------ #
    # validate — real canonical + business rules + real materiality
    # ------------------------------------------------------------------ #
    def validate(self, spec: PortfolioSpec, transformation_manifest: str,
                 work_dir: Path) -> StepResult:
        from engine.gate_3_validation import aggregate_validation_results as agg
        from engine.gate_3_validation.validate_business_rules import run_rules
        from engine.gate_3_validation.validate_canonical import (
            get_core_required_fields,
            load_registry,
            select_fields_for_portfolio,
        )
        tx_dir = Path(transformation_manifest).parent
        typed_csv = tx_dir / "31_transformed_canonical_tape.csv"
        frame = pd.read_csv(typed_csv, low_memory=False)

        issue_policy = _load_yaml(self.issue_policy_path)

        # Canonical: core-required fields must be present and populated. Emitted
        # in the canonical validator's own schema (rule_id/severity/field/row/
        # message) so the platform's normaliser handles it unchanged.
        registry = load_registry(self.registry_path)
        fields_meta = select_fields_for_portfolio(
            registry, spec.source_portfolio_type or "direct")
        canonical_rows: List[Dict[str, Any]] = []
        for field_name in get_core_required_fields(fields_meta):
            if field_name not in frame.columns:
                canonical_rows.append({
                    "rule_id": "CORE001", "severity": "error",
                    "field": field_name, "row": -1,
                    "message": f"{field_name} is not present"})
                continue
            blank = frame[field_name].isna() | (
                frame[field_name].astype(str).str.strip() == "")
            for idx in frame.index[blank]:
                canonical_rows.append({
                    "rule_id": "CORE002", "severity": "error",
                    "field": field_name, "row": int(idx),
                    "message": f"{field_name} is empty"})

        # Business rules: the real rule engine.
        try:
            business = run_rules(frame, self.regime or "")
        except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
            self._record(StageRecord(
                stage="validate", outcome=STAGE_HARD_BLOCKED,
                summary="The business rules could not be run.",
                blockers=[f"business rules failed ({type(exc).__name__})"]))
            return StepResult(ok=False, blocking=True,
                              blockers=[f"business rules failed "
                                        f"({type(exc).__name__})"],
                              message="validation failed")

        # Normalise both sources with the platform's own normalisers, then let
        # the platform's aggregator assign materiality from the issue policy.
        frames = []
        if canonical_rows:
            frames.append(agg.normalise_canonical_violations(
                pd.DataFrame(canonical_rows)))
        if business is not None and len(business):
            frames.append(agg.normalise_business_violations(business))
        combined = pd.concat(frames, ignore_index=True) if frames \
            else pd.DataFrame()
        summary = (agg.aggregate(combined, int(len(frame)), issue_policy)
                   if len(combined) else pd.DataFrame())
        self.validation_report = (summary.to_dict("records")
                                  if len(summary) else [])
        blocking = [r for r in self.validation_report
                    if str(r.get("materiality")).upper() == "BLOCKING"]
        review = [r for r in self.validation_report
                  if str(r.get("materiality")).upper() == "REVIEW"]

        val_dir = tx_dir / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)
        val_manifest = val_dir / "40_validation_manifest.json"
        val_manifest.write_text(json.dumps({
            "ready_for_validation_complete": not blocking,
            "findings": self.validation_report,
            "runtime_mode": "synthetic",
        }, indent=2, default=str), encoding="utf-8")

        if blocking:
            blockers = [
                f"{r.get('field_name')}: {r.get('issue_type')} affects "
                f"{r.get('affected_rows')} record(s) "
                f"({r.get('error_rate')}%) — materiality BLOCKING"
                for r in blocking]
            self._record(StageRecord(
                stage="validate", outcome=STAGE_HARD_BLOCKED,
                component="engine.gate_3_validation.validate_business_rules + "
                          "aggregate_validation_results (issue_policy "
                          "materiality)",
                summary=f"{len(blocking)} check(s) failed at BLOCKING "
                        "materiality.",
                metrics={"blocking": len(blocking), "review": len(review),
                         "rows": int(len(frame))},
                blockers=blockers))
            return StepResult(ok=False, blocking=True, blockers=blockers,
                              output_path=str(typed_csv),
                              manifest_path=str(val_manifest),
                              message="validation blocked")

        self._record(StageRecord(
            stage="validate", outcome=STAGE_DETERMINISTIC_COMPLETED,
            component="engine.gate_3_validation.validate_business_rules + "
                      "aggregate_validation_results (issue_policy materiality)",
            summary=("All checks passed." if not review else
                     f"{len(review)} finding(s) need review but do not block."),
            metrics={"blocking": 0, "review": len(review),
                     "rows": int(len(frame))}))
        return StepResult(ok=True, output_path=str(typed_csv),
                          manifest_path=str(val_manifest),
                          readiness={"ready_for_validation_complete": True},
                          message="validation complete")

    # ------------------------------------------------------------------ #
    # stamp / assemble / route: NOT overridden — the base class runs the real
    # provenance stamping and the real Assembler Agent, locally.
    # ------------------------------------------------------------------ #
    def stamp_provenance(self, spec: PortfolioSpec, validated_csv: str,
                         out_dir: Path) -> StepResult:
        result = super().stamp_provenance(spec, validated_csv, out_dir)
        self._record(StageRecord(
            stage="stamp", component="engine.provenance",
            outcome=(STAGE_DETERMINISTIC_COMPLETED if result.ok
                     else STAGE_HARD_BLOCKED),
            summary=result.message, blockers=list(result.blockers)))
        return result

    def assemble(self, stamped_paths: Sequence[str], out_dir: Path,
                 client_id: str, target: str, *,
                 regime: Optional[str] = None) -> StepResult:
        result = super().assemble(stamped_paths, out_dir, client_id, target,
                                  regime=regime)
        self._record(StageRecord(
            stage="assemble", component="engine.assembler_agent",
            outcome=(STAGE_DETERMINISTIC_COMPLETED if result.ok
                     else STAGE_HARD_BLOCKED),
            summary=result.message, metrics=dict(result.readiness or {}),
            blockers=list(result.blockers)))
        return result

    def route_mi(self, central_canonical: str) -> StepResult:
        result = super().route_mi(central_canonical)
        # Routing to the live MI Agent is a downstream handoff, so the synthetic
        # result records the intended destination without pointing anything at
        # it. The base class only computes the reference — nothing is published.
        self._record(StageRecord(
            stage="route", component="engine.orchestrator_agent.adapters."
                                     "AgentAdapters.route_mi",
            outcome=STAGE_CONTRACT_VALIDATED,
            summary="The central canonical satisfies the MI route contract. "
                    "Nothing was published.",
            metrics={"central_canonical": central_canonical}))
        return result

    # ------------------------------------------------------------------ #
    # project — contract-validated, then simulated
    # ------------------------------------------------------------------ #
    def project(self, central_canonical: str, out_dir: Path,
                regime: str) -> StepResult:
        """Validate the projector call and record it. Never runs it.

        The base class spawns the regime projector. In synthetic mode the
        intended command is built with the real
        :func:`engine.assembler_agent.build_regime_command` — so an invalid
        call still fails here — and the result is labelled ``execution
        simulated``. It never reports that a projection happened.
        """
        from engine import assembler_agent as _assembler_agent
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            command = _assembler_agent.build_regime_command(
                central_canonical, out_dir, regime, output_prefix="central",
                allow_unreviewed=True)
        except Exception as exc:  # noqa: BLE001 — an invalid call is a blocker
            self._record(StageRecord(
                stage="project", outcome=STAGE_HARD_BLOCKED,
                summary="The regulatory projection could not be prepared.",
                blockers=[f"projection contract invalid "
                          f"({type(exc).__name__})"]))
            return StepResult(ok=False, blocking=True,
                              blockers=[f"projection contract invalid "
                                        f"({type(exc).__name__})"],
                              message="projection contract invalid")
        plan = out_dir / "intended_projection.json"
        plan.write_text(json.dumps({
            "would_run": [str(c) for c in command],
            "regime": regime,
            "input": central_canonical,
            "execution_status": "simulated_only",
            "runtime_mode": "synthetic",
        }, indent=2), encoding="utf-8")
        self._record(StageRecord(
            stage="project", component="engine.assembler_agent."
                                       "build_regime_command",
            outcome=STAGE_SIMULATED,
            summary="The regulatory projection was prepared and validated but "
                    "not run.",
            metrics={"regime": regime, "plan": str(plan)}))
        return StepResult(ok=True, output_path=str(plan),
                          readiness={"regime": regime,
                                     "execution_status": "simulated_only"},
                          message="projection simulated (not executed)")

    # ------------------------------------------------------------------ #
    def trigger_live_pipeline(self, *, actor: str = "") -> None:
        """The live-handoff seam. Always refused in synthetic mode."""
        self.policy.require(CAP_LIVE_PIPELINE_TRIGGER,
                            detail="orchestration run",
                            case_id=self.case_id, tenant=self.tenant,
                            actor=actor)

    # ------------------------------------------------------------------ #
    def _primary_tape(self) -> Path:
        """The file the canonical tape is built from.

        The loan extract when one is present (it carries loan identity, which
        the Assembler requires); otherwise the first file, so a pack without a
        recognised loan tape still reaches the control that will reject it,
        rather than failing here with a less useful message.
        """
        for path in self.artefact_paths:
            name = path.name.lower()
            if "loan" in name:
                return path
        return self.artefact_paths[0]


# --------------------------------------------------------------------------- #
# Running the conductor
# --------------------------------------------------------------------------- #

def run_synthetic_orchestration(adapters: SyntheticOnboardingAdapters, *,
                                client_id: str, portfolio_id: str,
                                out_root: Path, created_at: str,
                                target: str = "mi",
                                regime: Optional[str] = None,
                                run_id: Optional[str] = None):
    """Drive the REAL conductor over the synthetic adapter.

    ``full_pipeline=True`` so the run takes the production
    onboard → transform → validate → stamp path rather than the lean MI
    shortcut: the point of the exercise is to exercise the gates.
    """
    from engine.orchestrator_agent.orchestrator import run_orchestration

    spec = PortfolioSpec(
        source_portfolio_id=portfolio_id,
        input=str(adapters.sandbox),
        source_portfolio_type=None)
    return run_orchestration(
        client_id, [spec], target=target, out_root=str(out_root),
        adapters=adapters, created_at=created_at, run_id=run_id,
        regime=regime, full_pipeline=True, force_publish=False,
        dataset="funded")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_excel(path)


def _duplicates(resolved: Dict[str, str]) -> Dict[str, List[str]]:
    """canonical field -> the source columns competing for it (2+ only)."""
    out: Dict[str, List[str]] = {}
    for column, canonical in resolved.items():
        out.setdefault(canonical, []).append(column)
    return {k: sorted(v) for k, v in out.items() if len(v) > 1}


def _populated(series: "pd.Series") -> int:
    return int(series.notna().sum() - (series.astype(str).str.strip() == "").sum())


def _mapping_decision(column: str, canonical: str, tier: str,
                      confidence: float, series: "pd.Series",
                      source_file: str) -> Dict[str, Any]:
    """A low-confidence match, in the existing pending-decision shape."""
    return {
        "decision_id": f"map_{_slug(column)}",
        "decision_type": "mapping_confirmation",
        "target_field": canonical,
        "source_column": column,
        "source_file": source_file,
        "status": "pending",
        "blocking": True,
        "recommended_action": "accept_mapping",
        "available_actions": ["accept_mapping", "choose_alternative",
                              "mark_unavailable"],
        "confidence": round(confidence, 4),
        "issue": f"'{column}' looks like {canonical.replace('_', ' ')}, but "
                 "not clearly enough to use without confirmation.",
        "evidence_summary": f"matched at tier '{tier}' with confidence "
                            f"{confidence:.2f}; "
                            f"{_populated(series)} of {len(series)} records "
                            "carry a value.",
    }


def _ambiguity_decision(canonical: str, columns: List[str],
                        frame: "pd.DataFrame",
                        source_file: str) -> Dict[str, Any]:
    """Two columns claiming one canonical field. Always a human decision."""
    counts = {c: _populated(frame[c]) for c in columns}
    detail = "; ".join(f"'{c}' carries values for {n} of {len(frame)} records"
                       for c, n in counts.items())
    preferred = max(counts, key=lambda c: (counts[c], c))
    return {
        "decision_id": f"amb_{_slug(canonical)}",
        "decision_type": "mapping_ambiguity",
        "target_field": canonical,
        "source_column": preferred,
        "source_columns": columns,
        "source_file": source_file,
        "status": "pending",
        "blocking": True,
        "recommended_action": "accept_mapping",
        "available_actions": ["accept_mapping", "choose_alternative",
                              "mark_unavailable"],
        "confidence": 0.5,
        "issue": f"{len(columns)} source fields could represent "
                 f"{canonical.replace('_', ' ')}.",
        "evidence_summary": detail,
        "proposed_mapping": f"{preferred} → {canonical}",
    }


def _write_decisions(work_dir: Path, decisions: List[Dict[str, Any]]) -> Path:
    path = Path(work_dir) / DECISIONS_FILE
    path.write_text(yaml.safe_dump({"decisions": decisions}, sort_keys=False),
                    encoding="utf-8")
    return path


def _slug(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")[:48]
