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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from . import base_mi_gate as _base_mi_gate
from . import cross_file as _cross_file
from . import llm_mapping as _llm
from . import workbook as _workbook
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
                 confirmed_product_profile: str = "",
                 regime: str = "",
                 registry_path: Path = REGISTRY_PATH,
                 aliases_dir: Path = ALIASES_DIR,
                 issue_policy_path: Path = ISSUE_POLICY_PATH,
                 approved_mappings: Optional[Dict[str, str]] = None,
                 llm_policy: Optional["_llm.Policy"] = None,
                 confirm_every_mapping: bool = False,
                 client_defaults: Optional[Dict[str, Any]] = None,
                 source_units: Optional[Dict[str, str]] = None,
                 case_id: str = "", tenant: str = ""):
        self.artefact_paths = [Path(p) for p in artefact_paths]
        self.policy = policy
        self.sandbox = Path(sandbox).resolve()
        self.asset_type = asset_type
        self.regime = regime
        self.registry_path = Path(registry_path)
        self.aliases_dir = Path(aliases_dir)
        self.issue_policy_path = Path(issue_policy_path)
        #: canonical field -> the scale the LENDER writes it on, where the
        #: operator has said. Canonical is percentage POINTS; a lender sending
        #: 0.35 for 35% is on another scale, not wrong. See
        #: :func:`percentage_scaled_fields`.
        self.source_units = {str(k): str(v) for k, v in
                             (source_units or {}).items() if v}
        #: source column -> canonical field, from human-approved decisions.
        self.approved_mappings = dict(approved_mappings or {})
        #: THE FIRST TIME A LENDER'S TAPE IS READ, NOTHING MATCHES ITSELF.
        #:
        #: A governed alias says the NAME is one the platform has seen before.
        #: It does not say this client means the same thing by it, and nobody
        #: has ever said they do — so on a first onboarding a trusted match is
        #: a PROPOSAL, and the human's reading of this lender's columns is the
        #: artefact the onboarding exists to produce. Promotion then carries it
        #: into governed rules at activation and production applies it every
        #: month after, without asking again.
        #:
        #: Off for an amendment: those mappings are already governed and
        #: already answered, and re-asking would make a change to one report a
        #: re-approval of the whole tape.
        self.confirm_every_mapping = bool(confirm_every_mapping)
        #: Columns matched confidently that are waiting on that approval,
        #: keyed ``(source file, source column)``. The pack routinely carries
        #: the same column name in more than one file — a loan identifier is
        #: in every extract — and keying on the name alone made one file's
        #: proposal answer for all of them.
        self.proposed_mappings: Dict[Tuple[str, str], str] = {}
        self.case_id = case_id
        self.tenant = tenant
        self.records: List[StageRecord] = []
        self.mapping_report: List[Dict[str, Any]] = []
        #: Fields more than one file in the pack carries, and whether those
        #: files agree. Reported, never blocking — see :mod:`.cross_file`.
        self.cross_file: List[Dict[str, Any]] = []
        #: Findings the product profile excuses for base MI. Reported, never
        #: hidden: "not applicable to this product" is an answer, and an
        #: operator who cannot see it cannot question it.
        self.excused_findings: List[Dict[str, Any]] = []
        #: The product the operator has confirmed this book to be, when they
        #: have. Empty until then, and nothing is excused without it.
        self.confirmed_product_profile: str = str(confirmed_product_profile
                                                  or "")
        #: The confirmation to put to them, when one is outstanding.
        self.product_profile_decision: Optional[Dict[str, Any]] = None
        #: What the REGULATORY return still wants once the asset pack and the
        #: client configuration have been asked. Never blocks the MI: the two
        #: are separate verdicts and the pipeline has always carried them as
        #: two flags. See :func:`base_mi_gate.regime_outstanding`.
        self.regime_pending: List[Dict[str, Any]] = []
        #: The client's own standing answers — the originator's name, LEI and
        #: country of establishment are captured once at onboarding and written
        #: here, not restated per loan on a monthly extract.
        self.client_defaults: Dict[str, Any] = dict(client_defaults or {})
        #: Which file each consolidated column came from, and which files could
        #: not be joined. A tape built from four files rather than one has to
        #: say so: "where did this balance come from?" is the first question an
        #: approver asks, and the answer must not be inferable only from the
        #: absence of a column. See :func:`consolidate_pack`.
        self.consolidation: Dict[str, Any] = {}
        #: file -> {column: canonical field}, kept past the onboard stage so a
        #: later refusal can say which file mapped the field it is refusing.
        #: See :func:`why_absent`.
        self.resolved_by_file: Dict[str, Dict[str, str]] = {}
        #: What the asset pack's closed-account rule filled, and what it found
        #: contradicting it. A value the platform wrote rather than read has to
        #: be visible as such. See :func:`_closed_account_zeroing`.
        self.closed_accounts: Dict[str, Any] = {}
        #: What the platform's own derivation step did to the tape — LTV scale
        #: normalisation, balance coherence, geography, reporting date. Kept so
        #: the stage can say it rather than leaving a changed value unexplained.
        self.derivations: Dict[str, Any] = {}
        #: The governed budget for asking a model about a column the
        #: deterministic tiers could not settle — see :mod:`.llm_mapping`.
        self.llm_policy: _llm.Policy = llm_policy or _llm.Policy.load()
        #: What the model was asked, what it proposed, and why not where not.
        #: A model that quietly did not run and one that had nothing to say
        #: look identical from the outside; they are not the same thing.
        self.llm: Dict[str, Any] = _llm.Outcome().to_dict()
        #: Reporting-period labels turned into the cut-off dates they stand
        #: for (``August`` -> ``2026-08-31``), per column. Recorded so the
        #: transform is visible rather than silent — see
        #: :func:`_canonicalise_period_cutoffs`.
        self.period_cutoffs: Dict[str, Any] = {}
        self.validation_report: List[Dict[str, Any]] = []
        self._assert_inside_sandbox()

    def _approved_for(self, source_file: str, column: str) -> Optional[str]:
        """What an operator already said about this column OF THIS FILE.

        ``None`` means nobody has answered; ``""`` means they answered "do not
        use it", which is not the same thing and must not collapse into it.

        Keyed on the pair because a pack carries the same column name in
        several files and an answer given about one of them is an answer about
        one of them. The bare column name is still accepted as a fallback, so
        answers recorded before decisions were file-scoped keep working — a
        case mid-onboarding must not lose what a person already confirmed.
        """
        qualified = mapping_key(source_file, column)
        if qualified in self.approved_mappings:
            return self.approved_mappings[qualified]
        return self.approved_mappings.get(column)

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

        # The file the canonical tape is built from. Named before profiling so
        # every mapping-report row can say whether it came from that file.
        primary = self._primary_tape()

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
                    "note": f"could not be profiled ({type(exc).__name__})",
                    "primary": path == primary, "source_sheet": ""})

        # 2. Real header mapping, per column, for EVERY file in the pack.
        #
        #    The canonical tape is still built from the primary tape alone —
        #    that is this adapter's contract and step 4 below is unchanged. But
        #    the mapping REPORT used to cover the primary tape only, so a pack
        #    of three files produced one file's worth of rows and the other two
        #    appeared nowhere. That is a reporting gap, not a modelling one: the
        #    real onboarding orchestrator loads every structured file in the
        #    inventory (``_load_structured_dataframes``) and the central tape
        #    builder consolidates a loan-domain field "even when its
        #    authoritative source is the cashflow extract, because domain
        #    membership follows the canonical field, not the file". An operator
        #    checking what Trakt made of a delivery has to see all of it.
        frame = _read_table(primary)
        decisions: List[Dict[str, Any]] = []
        resolved: Dict[str, str] = {}
        #: The same readings, kept PER FILE, because the tape is now built from
        #: the whole pack rather than from the primary alone. ``resolved`` stays
        #: the primary's own map: the ambiguity and coverage steps below ask
        #: "what does the tape file claim?", which is a different question from
        #: "what feeds the tape?" and is still answered per file.
        resolved_by_file: Dict[str, Dict[str, str]] = {}
        #: Every file's frame, kept so the pack can be compared against itself
        #: at step 3b rather than re-read.
        frames: Dict[str, Any] = {}
        #: file -> {column: canonical field} for every claim this run makes or
        #: proposes. Two columns IN ONE FILE claiming a field is an ambiguity;
        #: two files carrying the same field is ordinary and is reconciled at
        #: step 3b, so the clash is looked for within a file and not across the
        #: pack.
        claims: Dict[str, Dict[str, str]] = {}
        for path in self.artefact_paths:
            is_primary = path == primary
            table = _workbook.read_table(path)
            if table.frame is None:
                continue          # already reported as unreadable at step 1
            file_frame = frame if is_primary else table.frame
            frames[path.name] = file_frame
            claims.setdefault(path.name, {})
            for column in [str(c) for c in file_frame.columns]:
                approved = self._approved_for(path.name, column)
                if approved is not None:
                    if approved and approved != "__ignore__":
                        claims[path.name][column] = approved
                        resolved_by_file.setdefault(path.name, {})[column] = \
                            approved
                        if is_primary:
                            resolved[column] = approved
                    self.mapping_report.append({
                        "source_file": path.name, "source_column": column,
                        "canonical_field": approved, "tier": "operator_approved",
                        "confidence": 1.0, "note": "confirmed by an operator",
                        "primary": is_primary, "source_sheet": table.sheet})
                    continue
                canonical, tier, confidence = mapper.map_one(column)
                trusted = (tier in _TRUSTED_TIERS
                           or float(confidence) >= LOW_CONFIDENCE)
                self.mapping_report.append({
                    "source_file": path.name, "source_column": column,
                    "canonical_field": canonical or "", "tier": tier,
                    "confidence": round(float(confidence), 4),
                    "note": ("" if trusted
                             else "below the confidence threshold"),
                    "primary": is_primary, "source_sheet": table.sheet})
                # EVERY FILE'S COLUMNS ARE PUT TO A PERSON, NOT ONLY THE TAPE'S.
                #
                # This adapter builds its canonical tape from the primary file,
                # and for a long time that was also the limit of what it ASKED
                # about: a column in the cashflow or property extract was
                # matched, reported as "matched automatically", and never
                # shown to anyone. That made one delivery read two ways —
                # forty-five columns proposed and thirty-six settled by the
                # platform on its own — and the thirty-six were settled by
                # exactly the alias registry that the proposal exists to stop
                # trusting unread.
                #
                # It also under-read the delivery. Production does not work
                # from the primary tape alone: the central tape builder
                # consolidates a loan-domain field "even when its
                # authoritative source is the cashflow extract, because domain
                # membership follows the canonical field, not the file". A
                # mapping approved here is promoted to a governed rule scoped
                # to the PORTFOLIO, not to this adapter's tape — so an
                # operator's reading of the property extract's columns is
                # worth exactly as much as their reading of the tape's, and
                # both are wanted before anything is applied every month.
                if canonical and trusted and self.confirm_every_mapping:
                    # A first onboarding. The match is firm and still nobody
                    # has said it is right FOR THIS CLIENT, so it waits — and
                    # the waiting is the point: what a person confirms here is
                    # what gets promoted and applied every month after.
                    self.proposed_mappings[(path.name, column)] = canonical
                    claims[path.name][column] = canonical
                    decisions.append(_mapping_proposal(
                        column, canonical, tier, float(confidence),
                        file_frame[column], path.name, primary=is_primary))
                elif canonical and trusted:
                    claims[path.name][column] = canonical
                    resolved_by_file.setdefault(path.name, {})[column] = \
                        canonical
                    if is_primary:
                        resolved[column] = canonical
                elif canonical:
                    decisions.append(_mapping_decision(
                        column, canonical, tier, float(confidence),
                        file_frame[column], path.name, primary=is_primary))

        # 2b. THE MODEL'S SECOND OPINION, ON WHAT DETERMINISTIC MATCHING COULD
        #     NOT SETTLE.
        #
        #     Zero-cost first, which the loop above already is: the tiered
        #     mapper runs to exhaustion and everything it settles is settled.
        #     What reaches the model is only what it could not place — on a
        #     real hundred-column tape, the thirty columns that matched nothing
        #     and were previously reported as unreadable with no proposal
        #     against any of them, leaving an operator to name each canonical
        #     field from memory.
        #
        #     A suggestion is never a mapping. It does not enter `resolved`, it
        #     cannot reach the tape, and it arrives as a decision that says in
        #     its own words that a model proposed it — so confirming one is a
        #     person's act, and they can see what they are confirming.
        self.llm = _llm.Outcome().to_dict()
        unsettled = _llm.unresolved(self.mapping_report, self.llm_policy)
        if unsettled:
            outcome = _llm.suggest(
                unsettled, frame, policy=self.llm_policy,
                registry_path=self.registry_path, aliases_dir=self.aliases_dir,
                asset_type=self.asset_type)
            self.llm = outcome.to_dict()
            # Scoped to the primary tape: the model is only ever asked about
            # that file, and a same-named column in another file has its own
            # question, which is not this one.
            already = {str(d.get("source_column") or "") for d in decisions
                       if str(d.get("source_file") or "") == primary.name}
            for column, suggestion in outcome.by_column.items():
                for row in self.mapping_report:
                    if (row.get("primary")
                            and row.get("source_column") == column):
                        row["llm_field"] = suggestion.field_name
                        row["llm_confidence"] = round(
                            float(suggestion.confidence), 4)
                        row["llm_reasoning"] = suggestion.reasoning
                if column in already or column not in frame.columns:
                    # A column the deterministic pass already raised a question
                    # about keeps that question; the suggestion is recorded
                    # beside it rather than asked twice.
                    continue
                decisions.append(_llm.decision(
                    suggestion, source_file=primary.name,
                    populated=_populated(frame[column]),
                    rows=int(len(frame))))
            # Narrated when the model ran, and when it was switched ON and
            # still did not — an operator expecting proposals is owed the
            # reason none arrived. Not narrated where it is simply switched
            # off, which is a standing fact about the environment rather than
            # something that happened during this run; `run.llm` carries it
            # either way. The outcome stays DETERMINISTIC: nothing here was
            # simulated, and a model declining to answer is not a simulation.
            if outcome.asked or (self.llm_policy.enabled
                                 and outcome.skipped_because):
                self._record(StageRecord(
                    stage="onboard",
                    outcome=STAGE_DETERMINISTIC_COMPLETED,
                    component="engine.gate_1_alignment.llm_mapper_agent."
                              "LLMFieldMapper",
                    summary=(f"Asked a model about {outcome.asked} column"
                             f"{'s' if outcome.asked != 1 else ''} Trakt could "
                             f"not match on its own; it proposed "
                             f"{len(outcome.by_column)}. Nothing it suggested "
                             "is used until you confirm it."
                             if outcome.asked else
                             f"No model was asked: {outcome.skipped_because}."),
                    metrics={"unsettled_columns": len(unsettled),
                             "asked": outcome.asked,
                             "proposed": len(outcome.by_column)}))

        # 3. A canonical field claimed by two columns is an ambiguity a human
        #    must settle — the engine has no basis to prefer one.
        #
        #    PROPOSALS COUNT HERE TOO. On a first onboarding a confident match
        #    is proposed rather than resolved, so a clash between two of them
        #    would otherwise be invisible until after the set was approved —
        #    and an operator who approves fifteen mappings only to be told two
        #    of them collide has been asked to approve something that was never
        #    coherent. The clash is a real question and it is asked FIRST; the
        #    two columns stop being proposals, because "is this right?" has no
        #    answer while two columns claim the same field.
        #    AND IT IS ASKED PER FILE. The same fact arriving in two files
        #    under two names is not an ambiguity — it is the ordinary shape of
        #    a delivery, and step 3b reconciles it. Only two columns of ONE
        #    file claiming one field is a question the engine cannot answer.
        for file_name, claimed in claims.items():
            file_frame = frames.get(file_name)
            if file_frame is None:
                continue
            for canonical, columns in _duplicates(claimed).items():
                for column in columns:
                    if file_name == primary.name:
                        resolved.pop(column, None)
                    self.proposed_mappings.pop((file_name, column), None)
                decisions = [d for d in decisions
                             if not (d.get("source_file") == file_name
                                     and d.get("source_column") in columns)]
                decisions.append(_ambiguity_decision(
                    canonical, columns, file_frame, file_name,
                    primary=(file_name == primary.name)))

        # 3b. WHERE THE PACK DISAGREES WITH ITSELF.
        #
        #     The same fact usually arrives in more than one file under
        #     different names, and when the files agree that is free
        #     reconciliation. When they disagree the central tape builder
        #     raises a blocking conflict — after activation, which is the one
        #     place a rehearsal exists to have looked first. Reported here,
        #     never blocking: the remedy is a source-precedence rule, which is
        #     not an artefact this surface can write, so presenting it as an
        #     answerable question would be asking for something an operator
        #     cannot give from this screen.
        comparisons = _cross_file.compare(self.mapping_report, frames)
        self.cross_file = [c.to_dict() for c in comparisons]
        conflicts = _cross_file.findings(comparisons)
        if conflicts:
            self._record(StageRecord(
                stage="onboard", outcome=STAGE_HUMAN_INPUT_REQUIRED,
                component="operations_control.occ_agent.cross_file",
                summary=(f"{len(conflicts)} field"
                         f"{'s' if len(conflicts) != 1 else ''} disagree "
                         "between the files in this delivery. Name which "
                         "file to believe before it is built."),
                metrics={"fields_in_several_files": len(comparisons),
                         "fields_disagreeing": len(conflicts)}))

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
        #
        #    FROM THE WHOLE PACK, not from the primary file alone. Every file's
        #    columns are mapped, confirmed and promoted into governed rules, and
        #    until now every file but one was then left out of the delivery it
        #    was mapped for — so a lender shipping its balances in a separate
        #    principal-and-interest extract got a tape with no balance and a
        #    CORE001 refusal for a column the client had in fact supplied.
        #    See :func:`consolidate_pack` for the join and its rules.
        frames.setdefault(primary.name, frame)
        mapped, self.consolidation = consolidate_pack(
            frames, resolved_by_file, primary.name)
        self.resolved_by_file = resolved_by_file
        self.period_cutoffs = _canonicalise_period_cutoffs(
            mapped, self.artefact_paths)
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
            # Lineage for the consolidation: which file each column that is NOT
            # the primary tape's came from, and which files could not be
            # attached. The tape itself carries values, not provenance.
            "consolidation": self.consolidation,
            "runtime_mode": "synthetic",
        }, indent=2), encoding="utf-8")
        (work_dir / "source_profiles.json").write_text(
            json.dumps(profiles, indent=2, default=str), encoding="utf-8")

        read_it_as = ""
        for column, detail in self.period_cutoffs.items():
            pairs = ", ".join(f"{label} as {iso}"
                              for label, iso in sorted(
                                  detail.get("labels", {}).items())[:3])
            if pairs:
                read_it_as = (f" Read the reporting month in "
                              f"{column.replace('_', ' ')} as a cut-off date: "
                              f"{pairs}.")
        # WHERE THE TAPE'S COLUMNS CAME FROM, on the record rather than
        # inferable. A tape consolidated from four files is a different object
        # from one read out of a single extract, and an approver signing the
        # delivery is entitled to see which file each field was taken from and
        # which files could not be attached at all.
        brought_in = self.consolidation.get("added") or {}
        joined_from = sorted({str(f) for f in brought_in.values()})
        consolidated = ""
        if brought_in:
            consolidated = (
                f" {len(brought_in)} column"
                f"{'s' if len(brought_in) != 1 else ''} came from "
                f"{len(joined_from)} other file"
                f"{'s' if len(joined_from) != 1 else ''} in the pack, joined on "
                "the loan identifier.")
        # A file that could NOT be joined is a blocker's worth of news even
        # though the stage completes: its columns are mapped and confirmed, and
        # they are not in the delivery.
        unjoined = [str(f.get("note") or "")
                    for f in (self.consolidation.get("files") or [])
                    if not f.get("joined")]
        self._record(StageRecord(
            stage="onboard", outcome=STAGE_DETERMINISTIC_COMPLETED,
            component="engine.onboarding_agent.file_profiler + "
                      "engine.gate_1_alignment.semantic_alignment.HeaderMapper",
            summary=f"Read {len(self.artefact_paths)} file"
                    f"{'s' if len(self.artefact_paths) != 1 else ''} and "
                    f"matched {len(resolved)} columns on the loan tape."
                    f"{consolidated}{read_it_as}",
            blockers=[n for n in unjoined if n],
            metrics={"rows": int(len(mapped)), "mapped_columns": len(resolved),
                     "consolidated_columns": len(brought_in),
                     "tape_columns": int(len(mapped.columns)),
                     "source_columns": int(len(frame.columns)),
                     "period_labels_dated": sum(
                         len(d.get("labels", {}))
                         for d in self.period_cutoffs.values())}))
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
        # WHAT THE ASSET CLASS ANSWERS ABOUT A CLOSED ACCOUNT.
        #
        # A loan redeemed in the period is absent from the lender's balances
        # extract — there is no balance to report — so the consolidated tape
        # carries a blank and validation refuses it as a missing mandatory
        # value. Nothing is missing: the borrower repaid, and a repaid loan's
        # balance is nought. That is the product's answer, stated in the asset
        # pack, and read here rather than decided here.
        #
        # The platform's own function, called by the platform's own
        # `derive_fields` too, so the rehearsal and production cannot drift
        # into two readings of the same rule.
        self.closed_accounts = _closed_account_zeroing(frame, self.asset_type)
        # THE PLATFORM'S OWN DERIVATIONS, which this stage skipped entirely.
        #
        # It called the platform's `apply_types` and the platform's validators
        # and then built the tape validation would judge WITHOUT the step
        # production runs in between — so the rehearsal judged a thinner tape
        # than the platform builds, and refused deliveries the platform accepts.
        # The same drift as `base_mi_gate`, one gate further on.
        #
        # It is not only a filler of blanks, which is how it was waved away.
        # `_resolve_ltv` brings a SUPPLIED loan-to-value onto the canonical
        # percentage-point scale, reconciling it against balance and valuation:
        # a lender who states LTV as a 0-1 fraction has every row fail LTV002
        # against a rule that expects percentage points, at an error rate the
        # policy then escalates from REVIEW to BLOCKING. Nothing was wrong with
        # the delivery; the scale had simply never been normalised.
        #
        # Reported, never fatal: a derivation that cannot run must leave the
        # tape as the lender sent it and let validation speak, rather than take
        # a delivery down.
        try:
            self.derivations = _derive_fields(
                frame, spec.source_portfolio_type or "", tape.name,
                source_units=self.source_units)
        except Exception as exc:            # noqa: BLE001 — reported, not fatal
            self.derivations = {"error": f"{type(exc).__name__}: {exc}"}
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
                    f"{len(typed)} records.{_closed_sentence(self.closed_accounts)}",
            metrics={"rows": int(len(typed)),
                     "typed_columns": len(typed_fields),
                     "closed_accounts": self.closed_accounts,
                     "derivations": self.derivations,
                     "parse_failures": parse_failures},
            # A closed account STATING a balance is a contradiction: the
            # lender says the loan is repaid and also says money is owed on it.
            # The value is left exactly as they wrote it, and named here
            # because only they can say which of the two is right.
            blockers=[
                f"{n} loan{'s' if n != 1 else ''} with a closed account status "
                f"still state a {f.replace('_', ' ')}. The value the lender "
                "sent has been kept; ask them which is right."
                for f, n in sorted(
                    (self.closed_accounts.get("contradicted") or {}).items())]))
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
            validate_core_presence,
        )
        tx_dir = Path(transformation_manifest).parent
        typed_csv = tx_dir / "31_transformed_canonical_tape.csv"
        frame = pd.read_csv(typed_csv, low_memory=False)

        issue_policy = _load_yaml(self.issue_policy_path)

        # Canonical: core-required fields must be present and populated —
        # through THE PLATFORM'S OWN CHECK, not a second copy of it.
        #
        # This used to re-implement the check inline and hard-code
        # `severity: "error"` on every finding. The real validator asks each
        # field's `applicability` block first, and the registry says, for
        # instance:
        #
        #   maturity_date:
        #     applicability:
        #       equity_release:
        #         allowed_missing: true
        #         severity_if_missing: warning
        #         nd_default: ND2
        #         reason: "Lifetime mortgage / equity release products do not
        #                  have a fixed contractual maturity."
        #
        # So a lifetime mortgage with no maturity date is a WARNING on the
        # platform's own ingestion route and was BLOCKING here — the Agent
        # refusing deliveries the platform accepts, for reasons the
        # configuration had already answered. The same applied to the
        # originator's LEI, which the governed client configuration supplies at
        # projection and the Annex 2 preflight enforces separately.
        #
        # APPLICABILITY IS KEYED ON THE ASSET CLASS ("equity_release"), not on
        # how the book was acquired ("direct" / "acquired"). Passing the
        # portfolio type found nothing and fell through to the strict default,
        # which is the second half of the same defect.
        registry = load_registry(self.registry_path)
        fields_meta = select_fields_for_portfolio(
            registry, spec.source_portfolio_type or "direct")
        canonical_rows: List[Dict[str, Any]] = [
            v.__dict__ if hasattr(v, "__dict__") else dict(v)
            for v in validate_core_presence(
                frame, get_core_required_fields(fields_meta), fields_meta,
                self.asset_type or spec.source_portfolio_type or "direct")]

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
        # WHAT ACTUALLY STOPS THIS RUN, as the product profile decides it.
        #
        # Nothing in the platform's ingestion route gates on materiality — that
        # gate is the Agent's alone, and it used to stop on every BLOCKING
        # finding without asking what the product needs. So a lifetime mortgage
        # was refused for having no maturity date, while the same files loaded
        # through the platform went through.
        #
        # `config/asset/product_profiles.yaml` answers this per field, per
        # product, and has all along. An excused finding is still reported and
        # still on the record; it simply stops being a reason to refuse.
        # Nothing is excused when a regime is being prepared: `base_mi` speaks
        # for management information, and the regulatory return needs more.
        blocking, excused = _base_mi_gate.split(
            self.validation_report, asset_class=self.asset_type,
            regime=self.regime or "",
            confirmed_profile_id=self.confirmed_product_profile)
        # WHAT THE REGULATOR STILL WANTS IS A SEPARATE VERDICT FROM WHETHER THE
        # MI IS SOUND. A field Annex 2 needs and base MI does not has nothing
        # to do with the management information, and holding the MI for it
        # means a lender waiting on one LEI cannot see their own book. Asked of
        # the asset pack and the client configuration first, so `maturity_date`
        # — answered ND5 because a lifetime mortgage has no term — is never put
        # to the operator as something the lender owes us.
        self.regime_pending = _base_mi_gate.regime_outstanding(
            excused, regime=self.regime or "", asset_class=self.asset_type,
            client_defaults=self.client_defaults,
            registry_path=self.registry_path)
        # The question that has to be answered before anything is excused:
        # "is this book a lifetime mortgage?". On the asset class alone the
        # platform PROPOSES a profile rather than applying it, and that guard
        # is not worked around here — until an operator confirms it, every
        # required field keeps blocking. Asked on a REGULATORY run too: the
        # profile decides what base MI needs either way, and suppressing the
        # question on a regime run left the operator with seven blockers, no
        # explanation and nothing to answer.
        pending = _base_mi_gate.needs_confirmation(
            self.asset_type, self.confirmed_product_profile)
        if pending is not None and blocking:
            self.product_profile_decision = _base_mi_gate.confirmation_decision(
                pending, [str(r.get("field_name") or "") for r in blocking])
        self.excused_findings = excused
        if self.regime_pending:
            # REPORTED, NEVER BLOCKING. The stage completes, the MI goes on,
            # and the regulatory return is held instead — see the handoff
            # manifest's `ready_for_projection`, which has always been a flag
            # of its own beside `ready_for_transformation_validation`.
            self._record(StageRecord(
                stage="validate", outcome=STAGE_DETERMINISTIC_COMPLETED,
                component="operations_control.occ_agent.base_mi_gate",
                summary=(f"{len(self.regime_pending)} field"
                         f"{'s' if len(self.regime_pending) != 1 else ''} the "
                         f"{self.regime} return needs "
                         f"{'are' if len(self.regime_pending) != 1 else 'is'} "
                         "outstanding. The management information is "
                         "unaffected and goes on; the regulatory return waits "
                         "for them."),
                metrics={"regime_pending": len(self.regime_pending)},
                blockers=[_base_mi_gate.regime_sentence(r)
                          for r in self.regime_pending]))
        review = [r for r in self.validation_report
                  if str(r.get("materiality")).upper() == "REVIEW"]
        if excused:
            self._record(StageRecord(
                stage="validate", outcome=STAGE_DETERMINISTIC_COMPLETED,
                component="engine.onboarding_agent.product_profile",
                summary=(f"{len(excused)} required field"
                         f"{'s' if len(excused) != 1 else ''} "
                         "not needed for management information on this "
                         "product. Still required for the regulatory return."),
                metrics={"excused": len(excused)},
                blockers=[_base_mi_gate.sentence(r) for r in excused]))

        val_dir = tx_dir / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)
        val_manifest = val_dir / "40_validation_manifest.json"
        val_manifest.write_text(json.dumps({
            "ready_for_validation_complete": not blocking,
            # TWO VERDICTS, WRITTEN AS TWO FLAGS. The MI is ready when nothing
            # base MI needs is missing; the regulatory return is ready only
            # when the regulator's own fields are answered too. Conflating them
            # is what stopped a sound delivery for want of an LEI.
            "ready_for_projection": bool(self.regime) and not self.regime_pending,
            "regime_pending": [
                {"field_name": str(r.get("field_name") or ""),
                 "regime_code": str(r.get("regime_code") or ""),
                 "regime": str(r.get("regime") or "")}
                for r in self.regime_pending],
            "findings": self.validation_report,
            "runtime_mode": "synthetic",
        }, indent=2, default=str), encoding="utf-8")

        if blocking:
            # WHY, NOT ONLY WHAT. A refusal naming a field an operator has
            # mapped and confirmed is unanswerable on its own: their screen
            # says 100%, this line says nought records, and what happened to
            # the file in between is recorded on a different stage. Where the
            # consolidation knows, it says so here. See :func:`why_absent`.
            blockers = []
            for r in blocking:
                field = str(r.get("field_name") or "")
                issue = str(r.get("issue_type") or "")
                # WHAT IT IS ABOUT AND WHAT IT CHECKS, in the operator's words
                # rather than the rule registry's. See :func:`rule_in_words`.
                subject, checks = rule_in_words(issue, field)
                said = (f"{subject}: {issue} affects "
                        f"{r.get('affected_rows')} record(s) "
                        f"({r.get('error_rate')}%) — materiality BLOCKING")
                if checks:
                    said = f"{said}. The check: {checks}"
                because = why_absent(field, self.resolved_by_file,
                                     self.consolidation)
                blockers.append(f"{said}. {because}" if because else said)
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
    """The client's tape, read the way the rest of the platform reads one.

    Delegates to :mod:`operations_control.occ_agent.workbook`, which picks a
    workbook's data sheet and re-detects a header below row one. A plain
    ``pd.read_excel(path)`` took the first sheet and row one, which for a real
    lender extract is the summary tab and a title block.
    """
    table = _workbook.read_table(path)
    if table.frame is None:
        raise SyntheticExecutionError(f"{path.name} could not be read")
    return table.frame


def _run_year(paths: Sequence[Path]) -> Optional[int]:
    """The reporting year the delivery is for, read from its own file names.

    A bare month label cannot be turned into a cut-off date without one, and
    the year is not something to guess at: ``None`` means "no year was stated",
    and the caller then leaves the value exactly as the client wrote it.
    """
    from engine.onboarding_agent import run_context as _rc
    for path in paths:
        for found in _rc.dates_from_filename(path.name) or []:
            try:
                return int(str(found)[:4])
            except (TypeError, ValueError):   # pragma: no cover — guard
                continue
    return None


def _canonicalise_period_cutoffs(mapped: "pd.DataFrame",
                                 paths: Sequence[Path]) -> Dict[str, Any]:
    """Turn a reporting-period LABEL into the cut-off date it stands for.

    THE DEFECT THIS CLOSES. A lender's monthly extract identifies its period in
    a ``Month Run`` column carrying ``August``. ``month run`` is a governed
    alias of ``data_cut_off_date`` (``config/system/aliases_mandatory.yaml``),
    so the column maps correctly — and then canonical typing reads ``August``
    as a date, fails, and writes a blank. Every row of a 568-row book came out
    missing its cut-off date, and the run stopped on a field the client had in
    fact supplied.

    The platform already knows the answer. ``central_tape_builder`` performs
    exactly this step for exactly these fields, turning ``August`` into
    ``2026-08-31`` against the run year and keeping the raw label in lineage.
    The Agent built its tape without it — the same drift, once more, between a
    stage body here and the platform's own build.

    So this calls the builder's own function rather than restating it. What it
    will not do is invent: ``31/08/2026`` is normalised, ``August`` resolves
    only where the delivery's file names state a year, and anything that
    resolves to nothing is left untouched for a human to look at.
    """
    from engine.onboarding_agent.central_tape_builder import (
        _PERIOD_CUTOFF_FIELDS,
        _canonicalise_period_cutoff,
    )
    year = _run_year(paths)
    applied: Dict[str, Any] = {}
    for column in [c for c in mapped.columns if c in _PERIOD_CUTOFF_FIELDS]:
        resolved: Dict[str, str] = {}
        def _canonical(value: Any) -> Any:
            raw = "" if value is None else str(value).strip()
            if not raw:
                return value
            if raw in resolved:
                return resolved[raw] or value
            iso, method, _basis = _canonicalise_period_cutoff(raw, year)
            resolved[raw] = iso
            if iso and method == "period_label_to_month_end":
                # Only the label conversion is reported: re-normalising a date
                # a client already wrote as a date is not news to anyone.
                applied.setdefault(column, {"method": method, "run_year": year,
                                            "labels": {}})
                applied[column]["labels"][raw] = iso
            return iso or value
        mapped[column] = mapped[column].map(_canonical)
    return applied


def _duplicates(resolved: Dict[str, str]) -> Dict[str, List[str]]:
    """canonical field -> the source columns competing for it (2+ only)."""
    out: Dict[str, List[str]] = {}
    for column, canonical in resolved.items():
        out.setdefault(canonical, []).append(column)
    return {k: sorted(v) for k, v in out.items() if len(v) > 1}


def _populated(series: "pd.Series") -> int:
    return int(series.notna().sum() - (series.astype(str).str.strip() == "").sum())


#: The canonical field every file in a pack must carry for its columns to join
#: the loan tape. The platform's own key names are wider than this
#: (``central_tape_builder._LOAN_KEY_NAMES``), but by the time this runs the
#: columns have already been mapped, so the CANONICAL name is the only one that
#: matters — whatever the lender called it.
LOAN_KEY = "loan_identifier"

#: Where a secondary file carries more than one row per loan, this field picks
#: which row speaks for the loan. A per-period extract is the ordinary case:
#: a principal-and-interest file carries a row per loan per reporting period,
#: and the loan tape wants the latest.
PERIOD_FIELD = "data_cut_off_date"

#: Below this, two files' keys are not the same identifier under any rule the
#: platform knows, and joining them would be a guess. Reported, never forced.
KEY_OVERLAP_FLOOR = 0.5


def _key_rule(spine: List[str], other: List[str]) -> Tuple[str, str]:
    """``(rule, rule_for_other)`` — how these two files' loan ids compare.

    A LENDER'S FILES DO NOT SPELL THE LOAN ID THE SAME WAY, and the platform has
    known this all along. ``entity_key_resolver`` names the real cases from real
    packs: ``Loan Policy Number`` of ``760341`` is the same loan as
    ``Account Number`` of ``76034101`` — a stable trailing ``01`` — and
    ``76034101`` read from an Excel numeric column arrives as ``76034101.0``.

    A bare string comparison joins NEITHER, and the failure is silent: every
    lookup misses, the column is added full of blanks or not at all, and the
    tape looks assembled. That is worse than not consolidating, because it
    cannot be seen.

    So the rule is chosen by the platform's own detector rather than restated
    here — ``_detect_global_suffix`` over the two key spaces, then
    ``_canonical_rule_for`` per side, which strips a suffix only when it
    dominates AND does not collapse distinct ids into one. Both are the
    functions ``resolve_entity_keys`` itself uses.
    """
    from engine.onboarding_agent.entity_key_resolver import (
        _canonical_rule_for,
        _detect_global_suffix,
        _numeric_set,
    )
    spine_num, other_num = _numeric_set(spine), _numeric_set(other)
    if not spine_num or not other_num:
        return "exact", "exact"
    suffix = _detect_global_suffix([spine_num, other_num])
    spine_rule, _, _ = _canonical_rule_for(spine_num, suffix)
    other_rule, _, _ = _canonical_rule_for(other_num, suffix)
    return spine_rule, other_rule


def _keys(values: "pd.Series", rule: str) -> "pd.Series":
    """One file's loan ids as comparison keys. Never mutates the source: the
    original value stays on the tape for lineage, as the resolver requires."""
    from engine.onboarding_agent.entity_key_resolver import normalise_key
    return values.map(lambda v: normalise_key(v, rule))


def _mapped_frame(frame: "pd.DataFrame", resolved: Dict[str, str]
                  ) -> "pd.DataFrame":
    """One file's columns, renamed to canonical and narrowed to the mapped set.

    Two source columns mapped to one canonical field would collide on rename,
    which pandas resolves by keeping both under one name — so the first is
    taken and the clash is left to the ambiguity decision that already exists
    for it.
    """
    seen: Dict[str, str] = {}
    for column, canonical in resolved.items():
        if canonical and canonical != "__ignore__" and canonical not in seen:
            seen[canonical] = column
    keep = [c for c in seen.values() if c in frame.columns]
    out = frame[keep].copy()
    out.columns = [next(k for k, v in seen.items() if v == c) for c in keep]
    return out


def _one_row_per_loan(frame: "pd.DataFrame", keys: "pd.Series", file_name: str
                      ) -> Tuple[Optional["pd.DataFrame"], str]:
    """``(frame, note)`` — a secondary file collapsed to one row per loan.

    A LOAN TAPE HAS ONE ROW PER LOAN, and a secondary extract routinely does
    not: a principal-and-interest file carries a row per loan per reporting
    period. Joining it as-is would fan the tape out — every loan repeated once
    per period — and every downstream count, concentration and average would be
    wrong in a way that looks like data rather than like a bug.

    So a file with repeated keys is collapsed, and only in a way that can be
    justified: the latest reporting period wins where the file says what period
    each row is, and where it does not the file contributes NOTHING and says so.
    Picking an arbitrary row would be inventing an answer about which month the
    balance came from.
    """
    if not keys.duplicated().any():
        return frame, ""
    if PERIOD_FIELD not in frame.columns:
        return None, (
            f"{file_name} carries more than one row per loan and no "
            f"{PERIOD_FIELD.replace('_', ' ')}, so Trakt cannot tell which row "
            "speaks for the loan. Map its reporting date, or say which file is "
            "authoritative for these fields.")
    order = pd.to_datetime(frame[PERIOD_FIELD], errors="coerce")
    if order.isna().all():
        return None, (
            f"{file_name} carries more than one row per loan and its "
            f"{PERIOD_FIELD.replace('_', ' ')} could not be read as a date, so "
            "Trakt cannot tell which row speaks for the loan.")
    ranked = frame.assign(_occ_key=keys.to_numpy(), _occ_period=order.to_numpy())
    ranked = ranked.sort_values("_occ_period", na_position="first")
    collapsed = ranked.drop_duplicates("_occ_key", keep="last")
    return collapsed, (
        f"{file_name} carries more than one row per loan; the latest "
        f"{PERIOD_FIELD.replace('_', ' ')} was taken for each.")


def consolidate_pack(frames: Dict[str, Any], resolved_by_file: Dict[str, Dict[str, str]],
                     primary_name: str) -> Tuple["pd.DataFrame", Dict[str, Any]]:
    """EVERY file's mapped columns, as one canonical loan tape.

    WHAT THIS REPLACES. The tape was built from the primary file alone — the
    first file in the pack with "loan" in its name — and every other file's
    mappings were recorded, promoted into governed rules, and then left out of
    the delivery they were mapped for. A lender that ships its balances in a
    separate principal-and-interest extract therefore got a tape with no
    balance, and validation refused it on ``current_principal_balance``:
    CORE001, "column not present", nought records affected, because the column
    was never built rather than because the client never sent it.

    Production does not work that way and never did. ``central_tape_builder``
    consolidates a loan-domain field "even when its authoritative source is the
    cashflow extract, because domain membership follows the canonical field,
    not the file" — so the rehearsal was refusing a delivery the platform would
    have accepted, which is the one thing it must never do.

    THE RULES, AND WHY EACH IS THE SAFE DIRECTION.

    * The PRIMARY file is the spine. It carries loan identity, so it decides
      which loans exist; a secondary file can fill a column but never add a row.
      A loan that appears only in the cashflow extract is a reconciliation
      question, not a loan.
    * The PRIMARY wins a contested field, and a secondary fills only what the
      spine leaves BLANK. That is the loan-domain precedence
      ``central_tape_builder._order_sources`` applies by default, and it means
      consolidation can add facts but never overwrite one.
    * A file with no mapped loan identifier contributes nothing, and says so.
      There is nothing to join on, and joining on row order would silently
      attach one borrower's balance to another's loan.
    * A file with repeated keys is collapsed by the latest reporting period, or
      contributes nothing — see :func:`_one_row_per_loan`.

    Returns the tape and a report naming, per file, what it contributed and
    what it could not.
    """
    spine_resolved = resolved_by_file.get(primary_name) or {}
    spine = _mapped_frame(frames[primary_name], spine_resolved)
    report: Dict[str, Any] = {"primary": primary_name, "files": [], "added": {}}
    if LOAN_KEY not in spine.columns:
        report["files"].append({
            "source_file": primary_name, "joined": False,
            "note": (f"{primary_name} has no mapped loan identifier, so the "
                     "other files in the pack cannot be joined to it.")})
        return spine, report

    for file_name, resolved in sorted(resolved_by_file.items()):
        if file_name == primary_name or file_name not in frames:
            continue
        other = _mapped_frame(frames[file_name], resolved)
        if LOAN_KEY not in other.columns:
            report["files"].append({
                "source_file": file_name, "joined": False,
                "note": (f"{file_name} has no mapped loan identifier, so its "
                         "columns cannot be attached to a loan.")})
            continue
        spare = [c for c in other.columns
                 if c != LOAN_KEY
                 and (c not in spine.columns or _populated(spine[c]) == 0)]
        if not spare:
            report["files"].append({
                "source_file": file_name, "joined": True, "added": [],
                "note": (f"{file_name} carries nothing the loan tape was "
                         "missing.")})
            continue
        # HOW THE TWO FILES SPELL THE SAME LOAN. Chosen by the platform's own
        # detector, not assumed — see :func:`_key_rule`.
        spine_rule, other_rule = _key_rule(
            [str(v) for v in spine[LOAN_KEY].tolist()],
            [str(v) for v in other[LOAN_KEY].tolist()])
        spine_keys = _keys(spine[LOAN_KEY], spine_rule)
        other_keys = _keys(other[LOAN_KEY], other_rule)
        wanted = {k for k in spine_keys if k}
        overlap = (len({k for k in other_keys if k} & wanted) / len(wanted)
                   if wanted else 0.0)
        if overlap < KEY_OVERLAP_FLOOR:
            # NOT THE SAME IDENTIFIER, and forcing it would be a guess. Said
            # out loud: a join that silently matches nothing leaves a column of
            # blanks and a tape that looks assembled, which is worse than not
            # consolidating at all because it cannot be seen.
            report["files"].append({
                "source_file": file_name, "joined": False,
                "overlap": round(overlap, 4), "key_rule": other_rule,
                "primary_key_rule": spine_rule,
                "note": (f"{file_name} and {primary_name} agree on "
                         f"{overlap:.0%} of their loan identifiers, so Trakt "
                         "cannot tell they are the same loans. Check the "
                         "column each file identifies a loan by.")})
            continue
        # THE PERIOD COLUMN TRAVELS WITH THE NARROWING, even when the spine
        # already carries it. `spare` is what this file would ADD, and a
        # cashflow extract's reporting date is precisely what it does not add:
        # the loan extract states the same date, so the column is excluded as
        # redundant. Narrowing to `spare` alone therefore took away the one
        # column `_one_row_per_loan` needs to say which of a loan's monthly rows
        # speaks for it — and the file was dropped with "no data cut off date",
        # against an operator who had mapped it. It is carried here and attached
        # to nothing: only `spare` is written to the spine below.
        for_collapse = [LOAN_KEY, *spare]
        if PERIOD_FIELD in other.columns and PERIOD_FIELD not in for_collapse:
            for_collapse.append(PERIOD_FIELD)
        collapsed, note = _one_row_per_loan(
            other[for_collapse], other_keys, file_name)
        if collapsed is None:
            report["files"].append({"source_file": file_name, "joined": False,
                                    "note": note})
            continue
        lookup = collapsed.set_index("_occ_key") if "_occ_key" in \
            collapsed.columns else collapsed.set_index(other_keys.to_numpy())
        added: List[str] = []
        #: Mapped, joined, and every value that reached the spine was blank.
        #: Silently skipped until now, which is the one failure mode nobody can
        #: see: the file reads as joined, the column is simply absent, and
        #: validation refuses the field as though the client had never sent it.
        blank: List[str] = []
        for column in spare:
            values = spine_keys.map(lookup[column])
            if _populated(values) == 0:
                blank.append(column)
                continue
            spine[column] = values.to_numpy()
            added.append(column)
        # BOTH SIDES OF THE COMPARISON, because a rule named on its own does
        # not say how the match was made. ERE's pack is the case: the cashflow
        # extract shares the loan extract's long key and the property extract
        # carries the short one, so the SPINE is stripped for the second join
        # and left alone for the first. "How were these matched?" is answered
        # by the pair, not by one half of it.
        # THE LOANS THIS FILE DID NOT COVER. A join at 99.8% is a join, and the
        # loans in the remaining 0.2% get a blank rather than a value — which
        # validation then refuses as a missing mandatory value, per row, with
        # no hint that the cause is a loan the lender's own extract does not
        # carry. That is a reconciliation question for the lender, not a data
        # fault, and the two need different actions. A few identifiers are kept
        # so it can be looked up rather than hunted for.
        unmatched = sorted({k for k in spine_keys if k}
                           - {k for k in other_keys if k})
        report["files"].append({"source_file": file_name, "joined": True,
                                "added": sorted(added), "blank": sorted(blank),
                                "note": note,
                                "overlap": round(overlap, 4),
                                "unmatched": len(unmatched),
                                "unmatched_examples": unmatched[:5],
                                "key_rule": other_rule,
                                "primary_key_rule": spine_rule})
        for column in added:
            report["added"][column] = file_name
    return spine, report


#: The scales a lender can write a percentage on. Canonical is POINTS.
SOURCE_UNITS = ("percentage_points", "fraction")


def percentage_scaled_fields(registry_path: Path = REGISTRY_PATH) -> List[str]:
    """Canonical fields where "points or a fraction?" is a real question.

    Read from the registry's own ``unit: percentage_points`` declaration — the
    governed percentage contract — rather than listed here, so a field that
    joins or leaves that contract does not need this module edited too.
    """
    try:
        import yaml as _yaml
        reg = _yaml.safe_load(Path(registry_path).read_text(encoding="utf-8"))
        fields = ((reg or {}).get("fields") or {})
        return sorted(name for name, meta in fields.items()
                      if str((meta or {}).get("unit") or "")
                      == "percentage_points")
    except Exception:                       # pragma: no cover — config guard
        return []


def _derive_fields(frame: "pd.DataFrame", portfolio_type: str,
                   filename: str,
                   source_units: Optional[Dict[str, str]] = None
                   ) -> Dict[str, Any]:
    """The platform's own derivation step, run on the rehearsal's tape.

    ``config`` is empty by design: every key it reads is an OVERRIDE — a
    declared percentage unit, an acquired-LTV disclosure, a static reporting
    date — and a rehearsal that invented one would be rehearsing something
    other than this delivery. With none of them, the derivation reconciles
    against the tape's own balances and valuations, which is what a client
    without those overrides gets in production.

    ``closed_account`` is deliberately absent: it has already run above, before
    balance coherence, which is where a repaid loan's nought has to be written
    for the coherence step to see it.
    """
    from engine.gate_2_transform.canonical_transform import derive_fields
    # A DECLARED UNIT IS AUTHORITATIVE and short-circuits the reconciliation,
    # which is the point of declaring one: reconciliation needs a balance and a
    # valuation to compare against, and a book that has neither — or whose
    # ratios are genuinely small enough to read either way — cannot be settled
    # by arithmetic. The operator's answer can.
    config: Dict[str, Any] = {}
    declared = {f: {"source_unit": u} for f, u in (source_units or {}).items()
                if u}
    if declared:
        config["aliases"] = declared
    return derive_fields(frame, portfolio_type, filename,
                         dayfirst=True, infer_year=True, derive_month=False,
                         default_year=None, config=config) or {}


def _closed_sentence(outcome: Dict[str, Any]) -> str:
    """What the closed-account rule did, for the stage an operator reads.

    A value the platform wrote is not the same as one the lender sent, and the
    tape does not distinguish them. Silence here would make a derived nought
    indistinguishable from a reported one.
    """
    filled = (outcome or {}).get("filled") or {}
    if not filled:
        return ""
    rows = max(int(n) for n in filled.values())
    fields = ", ".join(f.replace("_", " ") for f in sorted(filled))
    return (f" {rows} loan{'s' if rows != 1 else ''} with a closed account "
            f"status had no balance, which the asset class answers as nought: "
            f"{fields} set to 0.")


def _closed_account_zeroing(frame: "pd.DataFrame", asset_class: str
                            ) -> Dict[str, Any]:
    """Apply the asset pack's closed-account answer, and say what it did.

    Never raises: a configuration that cannot be read must leave the tape
    exactly as the lender sent it and refuse as it would have before, rather
    than take a delivery down.
    """
    try:
        import yaml as _yaml
        from engine.gate_2_transform.canonical_transform import (
            zero_balances_on_closed_accounts,
        )
        name = str(asset_class or "").strip().lower().replace(" ", "_")
        if name not in ("equity_release", "erm", "rre"):
            return {}
        pack = REPO / "config" / "asset" / "product_defaults_ERM.yaml"
        if not pack.exists():
            return {}
        cfg = (_yaml.safe_load(pack.read_text(encoding="utf-8")) or {})
        closed = cfg.get("closed_account") or {}
        if not closed.get("statuses"):
            return {}
        return zero_balances_on_closed_accounts(
            frame, statuses=closed.get("statuses") or [],
            zero_fields=closed.get("zero_fields") or [])
    except Exception:                       # pragma: no cover — config guard
        return {}


def rule_in_words(issue_type: str, field_name: str) -> Tuple[str, str]:
    """``(what to call it, what it checks)`` for a business-rule finding.

    A BLOCKER HAS TO SAY WHAT IT IS ABOUT. This is what an operator was handed:

        PORTFOLIO: LTV002 affects 567 record(s) (99.82%) — materiality BLOCKING

        "These error messages are not intuitive enough and also do not allow
         the operator to fix."

    Neither word in front of the colon means anything to them. ``PORTFOLIO`` is
    not a field — ``normalise_business_violations`` sets it as "a placeholder
    until expansion", and the expansion step is never run — and ``LTV002`` is
    an identifier whose meaning lives in a Python list nobody reading a screen
    can open.

    The rule states both, and has all along: ``description`` says what it
    checks, ``required_columns`` says which fields it is about. Read here
    rather than restated, so a rule that changes cannot leave the sentence
    behind.

    Returns the placeholder unchanged when the rule is not one of these, so a
    canonical finding (CORE001, CORE002) reads exactly as it did.
    """
    try:
        from engine.gate_3_validation.validate_business_rules import RULES
    except Exception:                       # pragma: no cover — import guard
        return field_name, ""
    rule = next((r for r in RULES
                 if str(r.get("rule_id") or "") == str(issue_type or "")), None)
    if rule is None:
        return field_name, ""
    fields = [str(c) for c in (rule.get("required_columns") or [])]
    said = str(rule.get("description") or "").strip()
    # "DISABLED: ..." is a note to whoever maintains the rule, not to an
    # operator, and a rule that never runs cannot be the one that stopped them.
    if said.upper().startswith("DISABLED"):
        said = ""
    subject = ", ".join(f.replace("_", " ") for f in fields) if fields else ""
    return (subject or field_name), said


def why_absent(field: str, resolved_by_file: Dict[str, Dict[str, str]],
               consolidation: Dict[str, Any]) -> str:
    """What became of the mapping for ``field``, for an operator reading a
    refusal about it.

    "CORE001 affects 0 record(s)" says a column is not on the tape. It does not
    say why, and the operator's own screen says they mapped it and confirmed it
    at 100%. Both are true at once whenever consolidation drops a file, and the
    reason is recorded where nobody reading the blocker is looking — so the
    refusal was unanswerable and the only way forward was to guess.

    The consolidation knows exactly what happened. This says it, on the line
    that stops the run:

      * nobody mapped it — nothing to add, the field was genuinely not supplied
      * the file it was mapped in could not be joined, and why
      * the file joined and every value that reached the tape was blank, which
        means the loans matched but this column is empty for them

    Returns "" when there is nothing to add, so a finding about a field nobody
    mapped reads exactly as it did before.
    """
    holders = sorted(file_name for file_name, resolved
                     in (resolved_by_file or {}).items()
                     if field in (resolved or {}).values())
    if not holders:
        return ""
    primary = str((consolidation or {}).get("primary") or "")
    by_file = {str(f.get("source_file") or ""): f
               for f in ((consolidation or {}).get("files") or [])}
    said: List[str] = []
    for file_name in holders:
        if file_name == primary:
            # The spine's own column. It is on the tape by construction, so a
            # presence failure here is emptiness rather than absence.
            said.append(f"{file_name} maps a column to it, and it is the loan "
                        "tape itself — so the column is there and the values "
                        "are not.")
            continue
        entry = by_file.get(file_name)
        if entry is None:
            said.append(f"{file_name} maps a column to it but was not read.")
        elif not entry.get("joined"):
            said.append(str(entry.get("note") or
                            f"{file_name} could not be joined to the loan "
                            "tape."))
        elif field in (entry.get("blank") or []):
            said.append(f"{file_name} was joined to the loan tape and every "
                        f"value it carried for this field was blank.")
        elif field in (entry.get("added") or []) and entry.get("unmatched"):
            # PRESENT, AND BLANK FOR THE LOANS THAT FILE DOES NOT CARRY. The
            # symptom is a per-row "missing mandatory value" and the cause is a
            # loan the lender's own extract is missing — which is a question
            # for them, not a fault to fix here. Without this, a delivery is
            # refused over a number with no name attached to it.
            count = int(entry.get("unmatched") or 0)
            examples = [str(v) for v in (entry.get("unmatched_examples") or [])]
            shown = (f" ({', '.join(examples)}"
                     f"{', …' if count > len(examples) else ''})"
                     if examples else "")
            said.append(
                f"{count} loan{'s' if count != 1 else ''} on the loan tape "
                f"{'have' if count != 1 else 'has'} no row in {file_name}"
                f"{shown}, so this field is blank for "
                f"{'them' if count != 1 else 'it'}.")
        elif field not in (entry.get("added") or []):
            said.append(f"{file_name} maps a column to it and the loan tape "
                        "already had the field, so nothing was taken from it.")
    return " ".join(s for s in said if s)


#: How a source file and a source column are written as one key, wherever a
#: mapping is held by the pair rather than by the column name alone. A pack
#: routinely carries "Loan ID" in every extract; keying on the name alone made
#: one answer speak for all of them.
KEY_SEPARATOR = "::"


def mapping_key(source_file: str, column: str) -> str:
    return f"{source_file}{KEY_SEPARATOR}{column}"


def _decision_id(prefix: str, subject: str, source_file: str,
                 primary: bool) -> str:
    """A decision id that is unique across the pack, not just within a file.

    The PRIMARY tape keeps the unqualified id it has always had. That is not
    tidiness: a case part-way through its onboarding has answers recorded
    against those ids, and renaming them would orphan every one. Columns in the
    other files — which were never asked about before this, so have no answers
    to orphan — carry their file in the id, because "Pool" in the property
    extract and "Pool" in the tape are two different questions.
    """
    if primary:
        return f"{prefix}_{_slug(subject)}"
    return f"{prefix}_{_slug(source_file)}__{_slug(subject)}"


def _mapping_decision(column: str, canonical: str, tier: str,
                      confidence: float, series: "pd.Series",
                      source_file: str, *, primary: bool = True
                      ) -> Dict[str, Any]:
    """A low-confidence match, in the existing pending-decision shape."""
    return {
        "decision_id": _decision_id("map", column, source_file, primary),
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


#: A confident match on a first onboarding, waiting for the human reading that
#: makes it this client's mapping rather than the platform's guess at one.
#: Distinct from ``mapping_confirmation`` so the surfaces can tell them apart:
#: a proposal is answered in the TABLE, as part of one approval over the set,
#: and a confirmation is a question about a column nothing settled.
DECISION_MAPPING_PROPOSAL = "mapping_proposal"


def _mapping_proposal(column: str, canonical: str, tier: str,
                      confidence: float, series: "pd.Series",
                      source_file: str, *, primary: bool = True
                      ) -> Dict[str, Any]:
    """A firm match that has not yet been read by a person.

    It carries the same subject as a confirmation — a source column and the
    field it would feed — because that is what promotion turns into a governed
    rule, and one approval over seventy columns has to leave seventy records
    behind or the audit says a person approved "the mappings" and cannot say
    which.
    """
    return {
        "decision_id": _decision_id("map", column, source_file, primary),
        "decision_type": DECISION_MAPPING_PROPOSAL,
        "target_field": canonical,
        "source_column": column,
        "source_file": source_file,
        "status": "pending",
        "blocking": True,
        "recommended_action": "accept_mapping",
        "available_actions": ["accept_mapping", "choose_alternative",
                              "mark_unavailable"],
        "confidence": round(confidence, 4),
        "basis": "deterministic",
        "issue": f"'{column}' reads as {canonical.replace('_', ' ')}.",
        "evidence_summary": (f"matched at tier '{tier}' with confidence "
                             f"{confidence:.2f}; "
                             f"{_populated(series)} of {len(series)} records "
                             "carry a value. This is the first delivery from "
                             "this client, so it is proposed rather than "
                             "applied."),
    }


def _ambiguity_decision(canonical: str, columns: List[str],
                        frame: "pd.DataFrame",
                        source_file: str, *, primary: bool = True
                        ) -> Dict[str, Any]:
    """Two columns claiming one canonical field. Always a human decision."""
    counts = {c: _populated(frame[c]) for c in columns}
    detail = "; ".join(f"'{c}' carries values for {n} of {len(frame)} records"
                       for c, n in counts.items())
    preferred = max(counts, key=lambda c: (counts[c], c))
    return {
        "decision_id": _decision_id("amb", canonical, source_file, primary),
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
