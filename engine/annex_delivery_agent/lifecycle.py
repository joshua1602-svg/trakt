"""
engine.annex_delivery_agent.lifecycle
=====================================

The one delivery lifecycle every annex runs through.

    identify → resolve inputs → preflight → normalise → build → validate
    → governed result → await approval → publish

This module owns the first six. Approval and publication are deliberately *not*
here (see :mod:`engine.annex_delivery_agent.publication`): a stage that can run
automatically and a stage that requires a human decision should not live in the
same loop, or one day the loop will run both.

Nothing in this file mentions an annex, a field code or an XSD. Everything
annex-specific arrives through the provider interface, which is what lets
Annex 3 reuse the state machine, the checkpointing, the timing, the evidence
plumbing and the plain-English failure translation without change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .contracts import (
    ArtefactRef, DeliveryRequest, DeliveryResult, DeliveryStatus, SchemaValidationResult,
    STAGE_BUILD, STAGE_NORMALISE, STAGE_PREFLIGHT, STAGE_VALIDATE, StageTiming,
)
from .evidence import TransformationEvidence
from .persistence import Checkpoint, DeliveryRunStore
from .preflight import PreflightReport, translate_exception
from .providers.base import AnnexProvider, BuildOutcome, NormalisationOutcome, ResolvedInput


def peak_rss_mb() -> Optional[float]:
    """Process peak RSS in MB, or ``None`` where the platform cannot report it."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes; macOS reports bytes.
    import sys
    return usage / 1024.0 if sys.platform != "darwin" else usage / (1024.0 * 1024.0)


@dataclass
class _Timer:
    stages: List[StageTiming] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)

    def record(self, stage: str, seconds: float, *, resumed: bool = False) -> None:
        self.stages.append(StageTiming(stage=stage, seconds=seconds,
                                       peak_rss_mb=peak_rss_mb(),
                                       resumed_from_checkpoint=resumed))

    @property
    def total(self) -> float:
        return time.perf_counter() - self.started


@dataclass
class LifecycleOutcome:
    """What the lifecycle produced, before governance wraps it."""

    status: str
    result: DeliveryResult
    preflight: PreflightReport
    checkpoint: Checkpoint


class DeliveryLifecycle:
    """Drives one provider through the shared stages."""

    def __init__(self, provider: AnnexProvider, *, store: DeliveryRunStore,
                 run_dir: Path, run_id: str, tenant_id: str) -> None:
        self.provider = provider
        self.store = store
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.tenant_id = tenant_id

    # -- public ------------------------------------------------------------- #
    def run(self, request: DeliveryRequest, *, checkpoint: Checkpoint) -> LifecycleOutcome:
        timer = _Timer()
        definition = self.provider.definition
        report = PreflightReport()

        base = DeliveryResult(
            run_id=self.run_id,
            annex_id=definition.annex_id,
            annex_version=definition.report_version,
            status=DeliveryStatus.RECEIVED,
            tenant_id=self.tenant_id,
            portfolio_id=request.portfolio_id,
            reporting_date=request.reporting_date,
            run_directory=str(self.run_dir),
            component_versions=dict(self.provider.component_versions()),
        )

        # -- stage 1: preflight --------------------------------------------- #
        stage_started = time.perf_counter()
        config_versions = self.provider.check_configuration(request, report)
        applicable = self.provider.check_applicability(request, report)
        resolved: Optional[ResolvedInput] = None
        if applicable and report.passed:
            resolved = self.provider.resolve_input(request, report)
            self.provider.preflight(request, resolved, report)
        timer.record(STAGE_PREFLIGHT, time.perf_counter() - stage_started)

        self.store.write_preflight(self.run_dir, report.to_dict())
        base = _replace(base, configuration_versions=dict(config_versions))

        if not report.passed:
            return self._blocked(base, report, timer, checkpoint,
                                 status=DeliveryStatus.PREFLIGHT_BLOCKED)

        assert resolved is not None  # preflight passed, so the input resolved
        checkpoint.mark(STAGE_PREFLIGHT, outputs={"projected_input": str(resolved.path)},
                        detail={"row_count": resolved.row_count,
                                "field_count": resolved.field_count})
        self.store.save_checkpoint(self.run_dir, checkpoint)

        base = _replace(
            base,
            status=DeliveryStatus.READY,
            record_count=resolved.row_count,
            projected_field_count=resolved.field_count,
            source_references=(
                {"kind": "projected_dataset", **resolved.to_dict()},
                {"kind": "schema", "path": definition.schema_reference},
            ),
        )

        # -- stage 2: normalise --------------------------------------------- #
        stage_started = time.perf_counter()
        try:
            normalised, resumed = self._normalise(request, resolved, checkpoint)
        except Exception as exc:  # noqa: BLE001 - translated, never re-raised raw
            return self._component_failure(base, report, timer, checkpoint, exc,
                                           stage="normalise",
                                           status=DeliveryStatus.NORMALISATION_FAILED)
        timer.record(STAGE_NORMALISE, time.perf_counter() - stage_started, resumed=resumed)

        if normalised.blocked:
            return self._normalisation_blocked(base, report, timer, checkpoint, normalised)

        base = _replace(base, intermediate_artefacts=(
            normalised.artefact, *tuple(normalised.side_artefacts)))

        # -- stage 3: build -------------------------------------------------- #
        stage_started = time.perf_counter()
        try:
            built, resumed = self._build(request, normalised, checkpoint)
        except Exception as exc:  # noqa: BLE001
            return self._component_failure(base, report, timer, checkpoint, exc,
                                           stage="build",
                                           status=DeliveryStatus.BUILD_FAILED)
        timer.record(STAGE_BUILD, time.perf_counter() - stage_started, resumed=resumed)

        # -- stage 4: validate ----------------------------------------------- #
        stage_started = time.perf_counter()
        try:
            validation = self.provider.validate(request, built)
        except Exception as exc:  # noqa: BLE001
            return self._component_failure(base, report, timer, checkpoint, exc,
                                           stage="validate",
                                           status=DeliveryStatus.VALIDATION_FAILED)
        timer.record(STAGE_VALIDATE, time.perf_counter() - stage_started)

        self.store.write_validation(self.run_dir, validation.to_dict())
        checkpoint.mark(STAGE_VALIDATE, detail={"valid": validation.valid,
                                                "error_count": validation.error_count})
        self.store.save_checkpoint(self.run_dir, checkpoint)

        # -- evidence -------------------------------------------------------- #
        evidence_failure = ""
        try:
            evidence = self.provider.inspect(request, resolved, normalised, built)
        except Exception as exc:  # noqa: BLE001 - evidence must never fail a run
            # A produced artefact is worth more than its disclosure, so this does
            # not fail the run — but it must not disappear either: the cause is
            # recorded verbatim and the run is forced into review, because an
            # undisclosed automatic value is exactly what review exists to catch.
            evidence = TransformationEvidence()
            evidence_failure = (
                f"The automatic-value disclosure could not be produced for this run "
                f"({type(exc).__name__}: {exc}). The report may contain values the "
                f"system supplied on your behalf that are not listed here — review it "
                f"in full before approving.")
            evidence.notes.append(evidence_failure)
        self.store.write_evidence(self.run_dir, evidence.to_dict())

        blocking, warnings = self.provider.classify(evidence, validation)
        if evidence_failure:
            warnings = [evidence_failure, *warnings]

        status = self._final_status(validation, blocking, warnings)
        result = _replace(
            base,
            status=status,
            record_count=built.record_count or resolved.row_count,
            delivered_field_count=built.field_count,
            artefact=built.artefact,
            schema_validation=validation,
            blocking_errors=tuple(blocking),
            warnings=tuple(warnings),
            transformation_evidence=evidence.to_dict(),
            summary=_summary(status, built, validation, evidence),
            why_it_matters=_why_it_matters(status),
            timings=tuple(timer.stages),
            total_seconds=timer.total,
            peak_rss_mb=peak_rss_mb(),
            publication_eligible=(status in (DeliveryStatus.PREPARED,
                                             DeliveryStatus.PREPARED_WITH_WARNINGS)),
            publication_blockers=tuple(b["message"] for b in blocking),
        )
        return LifecycleOutcome(status=status, result=result, preflight=report,
                                checkpoint=checkpoint)

    # -- stages with checkpoint reuse --------------------------------------- #
    def _normalise(self, request: DeliveryRequest, resolved: ResolvedInput,
                   checkpoint: Checkpoint) -> Tuple[NormalisationOutcome, bool]:
        if checkpoint.reusable(STAGE_NORMALISE):
            existing = checkpoint.output(STAGE_NORMALISE, "delivery_ready")
            if existing is not None:
                detail = (checkpoint.stages[STAGE_NORMALISE].get("detail") or {})
                return NormalisationOutcome(
                    artefact=ArtefactRef.of_file("delivery_ready_csv", existing),
                    row_count=int(detail.get("row_count") or resolved.row_count),
                    field_count=int(detail.get("field_count") or 0),
                    blocking_issue_count=int(detail.get("blocking_issue_count") or 0),
                    summary=dict(detail.get("summary") or {}),
                    frame=None,
                ), True

        outcome = self.provider.normalise(request, resolved, self.run_dir)
        checkpoint.mark(STAGE_NORMALISE,
                        outputs={"delivery_ready": outcome.artefact.path},
                        detail={"row_count": outcome.row_count,
                                "field_count": outcome.field_count,
                                "blocking_issue_count": outcome.blocking_issue_count,
                                "summary": outcome.summary})
        self.store.save_checkpoint(self.run_dir, checkpoint)
        return outcome, False

    def _build(self, request: DeliveryRequest, normalised: NormalisationOutcome,
               checkpoint: Checkpoint) -> Tuple[BuildOutcome, bool]:
        # A completed build is only reusable if the artefact is still byte-intact.
        # Atomic finalisation means a partial file is never present under the
        # artefact name, and the recorded hash catches anything else.
        if checkpoint.reusable(STAGE_BUILD):
            existing = checkpoint.output(STAGE_BUILD, "artefact")
            if existing is not None:
                detail = checkpoint.stages[STAGE_BUILD].get("detail") or {}
                return BuildOutcome(
                    artefact=ArtefactRef.of_file("artefact", existing),
                    record_count=int(detail.get("record_count") or 0),
                    field_count=int(detail.get("field_count") or 0),
                    parsed=None,
                    detail=dict(detail.get("build_detail") or {}),
                ), True

        outcome = self.provider.build(request, normalised, self.run_dir)
        checkpoint.mark(STAGE_BUILD, outputs={"artefact": outcome.artefact.path},
                        detail={"record_count": outcome.record_count,
                                "field_count": outcome.field_count,
                                "build_detail": dict(outcome.detail)})
        self.store.save_checkpoint(self.run_dir, checkpoint)
        return outcome, False

    # -- failure paths ------------------------------------------------------ #
    def _blocked(self, base: DeliveryResult, report: PreflightReport, timer: _Timer,
                 checkpoint: Checkpoint, *, status: str) -> LifecycleOutcome:
        result = _replace(
            base,
            status=status,
            summary=report.operator_message(),
            why_it_matters=(
                "No report file was produced and nothing has been submitted. "
                "The issues below must be resolved before this report can be prepared."),
            blocking_errors=tuple(f.to_dict() for f in report.blockers),
            warnings=tuple(f.message for f in report.warnings),
            timings=tuple(timer.stages),
            total_seconds=timer.total,
            peak_rss_mb=peak_rss_mb(),
            publication_eligible=False,
            publication_blockers=tuple(f.message for f in report.blockers),
        )
        self.store.write_failure(self.run_dir, {"status": status,
                                                "preflight": report.to_dict()})
        return LifecycleOutcome(status=status, result=result, preflight=report,
                                checkpoint=checkpoint)

    def _normalisation_blocked(self, base: DeliveryResult, report: PreflightReport,
                               timer: _Timer, checkpoint: Checkpoint,
                               normalised: NormalisationOutcome) -> LifecycleOutcome:
        count = normalised.blocking_issue_count
        message = (
            f"{self.provider.definition.annex_name} cannot be prepared because "
            f"{count:,} value(s) in the projected data do not meet the delivery rules "
            f"for this report. No report file has been produced.")
        result = _replace(
            base,
            status=DeliveryStatus.NORMALISATION_FAILED,
            summary=message,
            why_it_matters=(
                "These are data problems, not system problems: the values must be "
                "corrected at source or in the client's mapping before the report "
                "can be built."),
            blocking_errors=tuple(
                {"code": "DELIVERY_RULE_VIOLATION", **dict(i)} for i in normalised.issues[:50]),
            intermediate_artefacts=(normalised.artefact, *tuple(normalised.side_artefacts)),
            timings=tuple(timer.stages),
            total_seconds=timer.total,
            peak_rss_mb=peak_rss_mb(),
            publication_eligible=False,
            publication_blockers=(message,),
        )
        self.store.write_failure(self.run_dir, {
            "status": DeliveryStatus.NORMALISATION_FAILED,
            "blocking_issue_count": count,
            "issues": normalised.issues[:200],
            "summary": normalised.summary,
        })
        # The build must not be able to reuse a checkpoint from a blocked run.
        checkpoint.invalidate_from(STAGE_BUILD)
        self.store.save_checkpoint(self.run_dir, checkpoint)
        return LifecycleOutcome(status=DeliveryStatus.NORMALISATION_FAILED, result=result,
                                preflight=report, checkpoint=checkpoint)

    def _component_failure(self, base: DeliveryResult, report: PreflightReport,
                           timer: _Timer, checkpoint: Checkpoint, exc: BaseException,
                           *, stage: str, status: str) -> LifecycleOutcome:
        finding = translate_exception(exc, annex_name=self.provider.definition.annex_name,
                                      stage=stage)
        report.add(finding)
        # Anything downstream of the failed stage is now untrustworthy.
        checkpoint.invalidate_from({"normalise": STAGE_NORMALISE, "build": STAGE_BUILD,
                                    "validate": STAGE_VALIDATE}.get(stage, STAGE_NORMALISE))
        self.store.save_checkpoint(self.run_dir, checkpoint)
        self.store.write_failure(self.run_dir, {
            "status": status, "stage": stage, "finding": finding.to_dict()})

        result = _replace(
            base,
            status=status,
            summary=finding.message,
            why_it_matters=("No report file has been produced and nothing has been "
                            "submitted."),
            blocking_errors=(finding.to_dict(),),
            timings=tuple(timer.stages),
            total_seconds=timer.total,
            peak_rss_mb=peak_rss_mb(),
            publication_eligible=False,
            publication_blockers=(finding.message,),
        )
        return LifecycleOutcome(status=status, result=result, preflight=report,
                                checkpoint=checkpoint)

    # -- status ------------------------------------------------------------- #
    @staticmethod
    def _final_status(validation: SchemaValidationResult, blocking: List[Dict[str, Any]],
                      warnings: List[str]) -> str:
        if blocking or (validation.executed and not validation.valid):
            return DeliveryStatus.VALIDATION_FAILED
        return (DeliveryStatus.PREPARED_WITH_WARNINGS if warnings
                else DeliveryStatus.PREPARED)


# --------------------------------------------------------------------------- #
# Operator-facing wording
# --------------------------------------------------------------------------- #


def _summary(status: str, built: BuildOutcome, validation: SchemaValidationResult,
             evidence: TransformationEvidence) -> str:
    size_mb = built.artefact.size_bytes / (1024 * 1024)
    if status == DeliveryStatus.VALIDATION_FAILED:
        return (f"The report file was created ({built.record_count:,} records, "
                f"{size_mb:.1f} MB) but did not pass schema validation: "
                f"{validation.error_count} error(s). It cannot be submitted.")
    head = (f"Report created and passed schema validation: {built.record_count:,} "
            f"records, {built.field_count} fields, {size_mb:.1f} MB.")
    if status == DeliveryStatus.PREPARED_WITH_WARNINGS:
        return (f"{head} {evidence.automatic_fill_count:,} value(s) were filled "
                f"automatically and {evidence.coercion_count:,} were adjusted — these "
                f"need review before the report is submitted.")
    return head


def _why_it_matters(status: str) -> str:
    if status == DeliveryStatus.VALIDATION_FAILED:
        return ("A report that fails schema validation would be rejected on "
                "submission. Nothing has been published.")
    return ("Passing schema validation means the file is *structurally* acceptable "
            "to the regulator's parser. It does not confirm that the figures are "
            "economically or regulatorily correct, and it does not confirm that the "
            "automatically inserted values are the right ones. The report is prepared, "
            "not submitted: it is published only after explicit approval.")


def _replace(result: DeliveryResult, **changes: Any) -> DeliveryResult:
    from dataclasses import replace as _dc_replace
    return _dc_replace(result, **changes)


__all__ = ["DeliveryLifecycle", "LifecycleOutcome", "peak_rss_mb"]
