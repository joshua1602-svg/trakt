"""operations_control.engine — the Workflow State Engine.

Owns the operational lifecycle: register/classify deliveries, create persistent
workflow runs, execute the existing orchestrator through governed adapters,
park at review gates, persist approvals as scoped rules, rerun affected stages,
gate publication behind explicit approval, and recover after restart.

Everything is persisted through :class:`operations_control.stores.OpsStore`
(the operations-control container) on every transition — API/UI/worker restarts
never lose state. Execution itself delegates to the EXISTING
``engine.orchestrator_agent.run_orchestration`` with ``RealAgentAdapters``;
publication delegates to the EXISTING ``ProductionPersistence.persist_platform``
and ``approvals.write_pending/approve/promote`` promotion path.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from . import language
from .adapters import (
    APPROVED_DECISIONS_FILE,
    DECISIONS_FILE,
    GovernedAdapters,
    _find_artifact,
    translate_run_state,
)
from .classification import Classification, classify_delivery, fingerprint_delivery
from .contracts import (
    BATCH_DATASET_DEFAULT,
    BATCH_DATASETS,
    BATCH_FREQUENCY_DEFAULT,
    DEC_APPROVED,
    DEC_OPEN,
    DEC_REJECTED,
    DEC_SUPERSEDED,
    DecisionRequired,
    Delivery,
    GovernedAgentResult,
    KIND_CLIENT_RULE,
    KIND_PUBLICATION,
    KIND_VALIDATION_EXCEPTION,
    OUTCOME_MI,
    OUTCOME_MI_ANNEX2,
    OUTCOMES,
    PUBLICATION_SCOPE_DEFAULT,
    PUBLICATION_SCOPES,
    REGIME_CAPABLE_DATASETS,
    RUN_AWAITING_PUBLICATION,
    RUN_BLOCKED,
    RUN_CANCELLED,
    RUN_FAILED,
    RUN_HELD,
    RUN_NEEDS_REVIEW,
    RUN_PUBLISHED,
    RUN_RECEIVED,
    RUN_RUNNING,
    ST_APPROVED,
    ST_COMPLETED,
    ST_NEEDS_REVIEW,
    ST_BLOCKED,
    ST_READY,
    ST_RUNNING,
    ST_WAITING,
    STAGE_DELIVERY_PREP,
    STAGE_MAPPING,
    STAGE_PROJECTION,
    STAGE_PUBLICATION,
    STAGE_REG_CONFIG,
    STAGE_VALIDATION,
    STAGE_XML,
    WF_NEW_CLIENT,
    WF_NEW_PORTFOLIO,
    WORKFLOW_TYPES,
    WorkflowRun,
    new_id,
    now_iso,
    stable_hash,
    transition,
)
from .rules import RuleRecord, RuleStore, project_rules_to_client_memory
from .stores import OpsStore

logger = logging.getLogger("trakt.operations_control.engine")

STAGING_ROOT_ENV = "TRAKT_OPS_STAGING_ROOT"
DEFAULT_STAGING_ROOT = ".ops_state/staging"

#: What each outcome is called when an operator has to be told the two intake
#: routes disagree. Plain sentences — never the internal outcome key.
OUTCOME_SENTENCES = {
    OUTCOME_MI: "management information only",
    OUTCOME_MI_ANNEX2: "management information and the ESMA Annex 2 delivery",
}


class OpsError(Exception):
    """Operational error with a machine code + operator-safe message."""

    def __init__(self, code: str, message: str, http_status: int = 400):
        self.code, self.message, self.http_status = code, message, http_status
        super().__init__(f"{code}: {message}")


class OpsEngine:
    """The governed workflow engine. One instance per API process; all state is
    in the OpsStore, so several instances (or restarts) converge on the same
    persisted truth."""

    def __init__(self, store: OpsStore, *, staging_root: Optional[str] = None,
                 real_agents: bool = True,
                 adapter_factory=None,
                 source_registry_factory=None,
                 persistence_factory=None,
                 annex2_stages=None,
                 client_config_path: Optional[str] = None):
        from .rules import memory_root
        self.store = store
        self.rules = RuleStore(store)
        self.staging_root = Path(staging_root or os.environ.get(
            STAGING_ROOT_ENV, DEFAULT_STAGING_ROOT))
        # Captured at construction so background threads never depend on the
        # environment still being set when they finish.
        self.memory_root = memory_root()
        # Governed Annex 2 delivery chain (I1): real runners by default; tests
        # substitute a stub with the same method signatures.
        self._annex2_stages = annex2_stages
        self.client_config_path = Path(client_config_path or os.environ.get(
            "TRAKT_OPS_CLIENT_CONFIG", "config/client/config_client_ERM_UK.yaml"))
        self.real_agents = real_agents
        self._adapter_factory = adapter_factory
        self._source_registry_factory = source_registry_factory
        self._persistence_factory = persistence_factory
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Deliveries + classification
    # ------------------------------------------------------------------ #
    def _source_registry(self):
        if self._source_registry_factory is not None:
            return self._source_registry_factory()
        from .classification import open_source_registry
        return open_source_registry(storage=self.store.storage)

    def register_delivery(self, *, client_id: str, portfolio_id: str,
                          input_path: str, dataset: str = "funded",
                          frequency: str = "monthly",
                          reporting_period: str = "",
                          registered_by: str = "") -> Dict[str, Any]:
        p = Path(input_path)
        if not p.exists():
            raise OpsError("OPS_DELIVERY_NOT_FOUND",
                           "Those files could not be found. Check the location "
                           "and try again.", 400)
        files = ([{"name": p.name}] if p.is_file() else
                 [{"name": f.name} for f in sorted(p.iterdir()) if f.is_file()])
        fingerprint = fingerprint_delivery(input_path)
        delivery = Delivery(
            delivery_id=new_id("del"), client_id=client_id,
            portfolio_id=portfolio_id, input_path=str(p),
            dataset=dataset, frequency=frequency,
            reporting_period=reporting_period, files=files,
            schema_fingerprint=fingerprint, registered_by=registered_by)
        cls = classify_delivery(
            self._source_registry(), client_id=client_id,
            portfolio_id=portfolio_id, dataset=dataset, frequency=frequency,
            fingerprint=fingerprint, reporting_period=reporting_period)
        doc = delivery.to_dict()
        doc["classification"] = cls.to_dict()
        self.store.save_delivery(doc)
        self.store.append_audit(client_id, "delivery_registered",
                                actor=registered_by or "system",
                                detail={"delivery_id": delivery.delivery_id,
                                        "classification": cls.to_dict()})
        return doc

    # ------------------------------------------------------------------ #
    # Workflow creation (idempotent) + start
    # ------------------------------------------------------------------ #
    def create_workflow(self, *, client_id: str, delivery_id: str, outcome: str,
                        workflow_type: Optional[str] = None,
                        created_by: str = "",
                        override_reason: str = "") -> WorkflowRun:
        if outcome not in OUTCOMES:
            raise OpsError("OPS_BAD_OUTCOME", "Choose what Trakt should prepare.", 400)
        delivery = self.store.load_delivery(client_id, delivery_id)
        if delivery is None:
            raise OpsError("OPS_DELIVERY_NOT_FOUND",
                           "That delivery could not be found.", 404)
        suggested = (delivery.get("classification") or {}).get("suggested", WF_NEW_CLIENT)
        final_type = workflow_type or suggested
        if final_type not in WORKFLOW_TYPES:
            raise OpsError("OPS_BAD_TYPE", "Choose a valid workflow type.", 400)

        key = stable_hash(client_id, delivery.get("portfolio_id", ""),
                          delivery.get("reporting_period", ""), outcome,
                          delivery_id)
        existing = self.store.find_by_idempotency_key(client_id, key)
        if existing is not None:
            run = self.store.load_workflow(client_id, existing["workflow_id"])
            if run is not None:
                return run

        run = WorkflowRun(
            workflow_id=new_id("wf"), client_id=client_id,
            portfolio_id=delivery.get("portfolio_id", ""),
            outcome=outcome, workflow_type=final_type, delivery=delivery,
            reporting_period=delivery.get("reporting_period", ""),
            classification_suggested=suggested,
            classification_overridden=bool(workflow_type
                                           and workflow_type != suggested),
            classification_overridden_by=(created_by if workflow_type
                                          and workflow_type != suggested else ""),
            classification_reason=override_reason,
            idempotency_key=key, created_by=created_by)
        run.set_stage("received", ST_COMPLETED)
        self.store.save_workflow(run)
        self.store.append_event(run, "created", actor=created_by or "system",
                                detail={"outcome": outcome,
                                        "workflow_type": final_type})
        self.store.append_audit(
            client_id, "workflow_created", actor=created_by or "system",
            workflow_id=run.workflow_id,
            detail={"outcome": outcome,
                    "classification_suggested": suggested,
                    "classification_final": final_type,
                    "classification_overridden": run.classification_overridden,
                    "override_reason": override_reason})
        return run

    # -- execution ----------------------------------------------------- #
    def start(self, run: WorkflowRun, *, actor: str = "system",
              wait: bool = False) -> WorkflowRun:
        """Start (or resume) execution. Idempotent: a run already executing in
        this process is left alone; a run in a review state must go through
        :meth:`rerun`."""
        with self._lock:
            t = self._threads.get(run.workflow_id)
            if t is not None and t.is_alive():
                return run
            if run.status not in (RUN_RECEIVED, RUN_NEEDS_REVIEW, RUN_BLOCKED,
                                  RUN_FAILED, RUN_HELD):
                raise OpsError("OPS_ALREADY_RUNNING",
                               "This workflow is already in progress.", 409)
            transition(run, RUN_RUNNING)
            run.interrupted = False
            self.store.save_workflow(run)
            self.store.write_lease(run)
            self.store.append_event(run, "started", actor=actor)
            thread = threading.Thread(
                target=self._execute_safely, args=(run.client_id, run.workflow_id),
                name=f"ops-{run.workflow_id}", daemon=True)
            self._threads[run.workflow_id] = thread
            thread.start()
        if wait:
            thread.join()
            return self.store.load_workflow(run.client_id, run.workflow_id) or run
        return run

    def rerun(self, run: WorkflowRun, *, actor: str) -> WorkflowRun:
        """Explicit rerun of the affected stage(s) — resumes the orchestrator
        from persisted state; completed steps are never repeated."""
        if run.status == RUN_RUNNING and self._is_executing(run.workflow_id):
            raise OpsError("OPS_ALREADY_RUNNING",
                           "This workflow is already in progress.", 409)
        run.rerun_count += 1
        self.store.append_audit(run.client_id, "workflow_rerun", actor=actor,
                                workflow_id=run.workflow_id,
                                detail={"rerun_count": run.rerun_count})
        return self.start(run, actor=actor)

    def cancel(self, run: WorkflowRun, *, actor: str, reason: str) -> WorkflowRun:
        transition(run, RUN_CANCELLED)
        self.store.save_workflow(run)
        superseded = self._supersede_open_decisions(run, actor=actor,
                                                    reason=reason)
        self.store.append_event(run, "cancelled", actor=actor,
                                detail={"reason": reason,
                                        "questions_closed": superseded})
        self.store.append_audit(run.client_id, "workflow_cancelled", actor=actor,
                                workflow_id=run.workflow_id,
                                detail={"reason": reason,
                                        "questions_closed": superseded})
        return run

    def _supersede_open_decisions(self, run: WorkflowRun, *, actor: str,
                                  reason: str) -> int:
        """Close the questions a cancelled run was still asking.

        A cancelled delivery has nothing left to decide, so leaving its
        questions open would keep them in the review queue for a workflow
        nobody can act on. They are marked superseded rather than approved or
        rejected: neither answer was given, and recording one would put a
        decision in the audit trail that nobody made.
        """
        closed = 0
        for doc in self.store.list_decisions(run.client_id, status=DEC_OPEN,
                                             workflow_id=run.workflow_id):
            doc.update(status=DEC_SUPERSEDED, resolved_by=actor,
                       resolved_at=now_iso(),
                       resolution_reason=reason or "The delivery was cancelled.")
            self.store.save_decision(run.client_id, doc)
            self.store.append_audit(
                run.client_id, "decision_superseded", actor=actor,
                workflow_id=run.workflow_id,
                decision_id=doc.get("decision_id", ""),
                detail={"cause": "workflow_cancelled"})
            closed += 1
        return closed

    def _is_executing(self, workflow_id: str) -> bool:
        t = self._threads.get(workflow_id)
        return t is not None and t.is_alive()

    # -- the actual execution ------------------------------------------ #
    def _execute_safely(self, client_id: str, workflow_id: str) -> None:
        try:
            self._execute(client_id, workflow_id)
        except Exception:
            logger.exception("workflow execution failed: %s", workflow_id)
            run = self.store.load_workflow(client_id, workflow_id)
            if run is not None:
                run.status = RUN_FAILED
                run.blockers = [language.GENERIC_PROBLEM]
                self.store.save_workflow(run)
                self.store.clear_lease(run)
                self.store.append_event(run, "execution_error")

    def _staging_dir(self, run: WorkflowRun) -> Path:
        return self.staging_root / run.client_id / run.workflow_id

    def _orchestrator_target(self, run: WorkflowRun) -> str:
        # Both outcomes run the conductor on the MI target: the AUTHORITATIVE
        # (and only XSD-proven) Annex 2 route projects from the assembled
        # platform canonical; the regulatory delivery steps are the OCC's own
        # governed chain (see _run_annex2_chain), mirroring the proven route
        # exactly rather than the regulatory-onboarding path that cannot reach
        # XML today.
        return "mi"

    def _annex2_runners(self):
        if self._annex2_stages is not None:
            return self._annex2_stages
        from .annex2.stages import Annex2Stages
        self._annex2_stages = Annex2Stages()
        return self._annex2_stages

    def _config_resolver(self):
        from .configuration.resolver import EffectiveConfigResolver
        return EffectiveConfigResolver(self.store, self.rules,
                                       client_config_path=self.client_config_path)

    @property
    def intake(self):
        from .intake import IntakeService
        if getattr(self, "_intake", None) is None:
            self._intake = IntakeService(self.store,
                                         staging_root=self.staging_root)
        return self._intake

    # ------------------------------------------------------------------ #
    # Input batches (OCC-owned readiness — no sentinel files)
    # ------------------------------------------------------------------ #
    def create_batch(self, *, client_id: str, portfolio_id: str,
                     reporting_date: str, workflow_type: str,
                     created_by: str,
                     auto_start_when_ready: bool = False,
                     dataset: str = "", frequency: str = "") -> Dict[str, Any]:
        """``dataset`` separates a non-funded delivery (pipeline) into its own
        input pack. Omitted or ``"funded"`` reproduces the existing pack identity
        exactly, so nothing already in flight is disturbed."""
        if workflow_type not in OUTCOMES:
            raise OpsError("OPS_BAD_OUTCOME",
                           "Choose what Trakt should prepare.", 400)
        if dataset and dataset not in BATCH_DATASETS:
            raise OpsError("OPS_BAD_DATASET",
                           "Choose which book these files describe.", 400)
        # A pipeline view can never produce a regime delivery. Refusing the
        # combination here means an operator cannot route a delivery into regime
        # reporting by accident — the same rule the blob trigger applies to
        # automated arrivals, enforced at the one other door into intake.
        if (workflow_type == OUTCOME_MI_ANNEX2
                and dataset and dataset not in REGIME_CAPABLE_DATASETS):
            raise OpsError(
                "OPS_DATASET_NOT_REGIME_CAPABLE",
                "Regulatory reporting is prepared from the funded book. "
                "Choose the funded book, or prepare management information "
                "instead.", 400)
        return self.intake.create_batch(
            tenant_id=client_id, client_id=client_id,
            portfolio_id=portfolio_id, reporting_date=reporting_date,
            workflow_type=workflow_type, created_by=created_by,
            auto_start_when_ready=auto_start_when_ready, dataset=dataset,
            frequency=frequency)

    def register_batch_file(self, *, client_id: str, batch_id: str,
                            source_path: str, received_by: str,
                            source_uri: str = "") -> Dict[str, Any]:
        batch = self.intake.load_batch(client_id, batch_id)
        if batch is None:
            raise OpsError("OPS_BATCH_NOT_FOUND",
                           "That input pack could not be found.", 404)
        p = Path(source_path)
        if not p.exists():
            raise OpsError("OPS_DELIVERY_NOT_FOUND",
                           "Those files could not be found. Check the "
                           "location and try again.", 400)
        files = [p] if p.is_file() else sorted(
            f for f in p.iterdir() if f.is_file())
        for f in files:
            batch = self.intake.register_file(
                batch, f, received_by_or_source=received_by,
                source_uri=source_uri)
        return self.assess_batch(client_id=client_id,
                                 batch_id=batch["batch_id"])

    # -- manual delivery: upload through the API, never a browser path ----- #
    def automated_identity(self, prefix: str, container: str) -> Dict[str, Any]:
        """What the AUTOMATED route would call a delivery filed at ``prefix``.

        The derivation is not re-implemented: the production path parser reads
        the location back, and the automated route's own workflow rule
        (:func:`apps.blob_trigger_app.occ_intake.outcome_for_source`) chooses the
        workflow. One rule, replayed — so the answer cannot drift from the
        trigger's answer as either side changes.
        """
        from apps.blob_trigger_app.occ_intake import outcome_for_source
        from apps.blob_trigger_app.path_parser import parse_blob_path

        parsed = parse_blob_path(f"{prefix}/probe.csv", container)
        outcome = outcome_for_source(self, parsed.client_id,
                                     parsed.source_portfolio_id, parsed.dataset)
        return {
            "client_id": parsed.client_id,
            "portfolio_id": parsed.source_portfolio_id,
            "reporting_period": parsed.reporting_period,
            "dataset": parsed.dataset,
            "workflow_type": outcome,
            "batch_id": self.intake.deterministic_batch_id(
                client_id=parsed.client_id,
                portfolio_id=parsed.source_portfolio_id,
                reporting_date=parsed.reporting_period,
                workflow_type=outcome, dataset=parsed.dataset),
        }

    def _require_converged_identity(self, batch: Dict[str, Any], prefix: str,
                                    container: str) -> None:
        """Refuse a manual upload the automated route would identify differently.

        A manual delivery is filed where an automated one would be, so both
        doors will see it. If they disagree about which delivery it is, the same
        files end up split across two input packs — each incomplete, neither
        publishable. That has to be caught before a byte is written, because
        after the write the trigger has already acted on it.
        """
        try:
            automated = self.automated_identity(prefix, container)
        except Exception as exc:  # noqa: BLE001 — unreadable location = refuse
            raise OpsError(
                "OPS_IDENTITY_UNVERIFIABLE",
                "Trakt could not confirm where this delivery belongs. Check "
                "the client, portfolio and reporting period.", 400) from exc
        if automated["batch_id"] == batch["batch_id"]:
            return

        chosen = OUTCOME_SENTENCES.get(batch.get("workflow_type", ""), "")
        expected = OUTCOME_SENTENCES.get(automated["workflow_type"], "")
        if chosen and expected and chosen != expected:
            message = (
                f"Files arriving on their own for this book are prepared as "
                f"{expected}, but you have asked for {chosen}. Sending them "
                "would leave the same delivery split in two. Choose "
                f"{expected}, or ask an administrator to change what this book "
                "reports before sending the files.")
        else:
            message = ("Trakt would treat these files as a different delivery "
                       "from the one you started. Check the client, portfolio "
                       "and reporting period, and start again.")
        self.store.append_audit(
            batch["client_id"], "manual_delivery_refused", actor="system",
            detail={"batch_id": batch["batch_id"],
                    "reason": "identity_divergence",
                    "automated_batch_id": automated["batch_id"],
                    "chosen_workflow_type": batch.get("workflow_type", ""),
                    "automated_workflow_type": automated["workflow_type"]})
        raise OpsError("OPS_IDENTITY_DIVERGENCE", message, 409)

    def upload_batch_files(self, *, client_id: str, batch_id: str,
                           uploads: List[Tuple[str, bytes]],
                           received_by: str) -> Dict[str, Any]:
        """Take files uploaded by an operator into the SAME governed intake the
        blob trigger uses.

        The destination is derived here from the batch's own controlled fields —
        the browser sends file content and a name, never a location. Every
        source file is written to its governed location and registered BEFORE
        readiness is assessed, so the internal run manifest (the governed
        replacement for the ``_READY.json`` sentinel) can only ever be written
        after the whole pack is safely present.
        """
        from .manual_intake import (ManualIntakeError, derive_blob_uri,
                                    derive_raw_prefix, raw_container,
                                    sanitise_filename)

        batch = self.intake.load_batch(client_id, batch_id)
        if batch is None:
            raise OpsError("OPS_BATCH_NOT_FOUND",
                           "That input pack could not be found.", 404)
        if batch.get("workflow_id"):
            raise OpsError(
                "OPS_BATCH_ALREADY_STARTED",
                "Trakt has already started this delivery. Start a new one to "
                "send replacement files.", 409)
        if not uploads:
            raise OpsError("OPS_NO_FILES", "Choose at least one file to send.",
                           400)

        # 1. Derive the governed destination server-side (fail closed).
        container = raw_container()
        try:
            prefix = derive_raw_prefix(
                client_id=batch["client_id"],
                portfolio_id=batch["portfolio_id"],
                reporting_period=batch["reporting_date"],
                dataset=batch.get("dataset") or BATCH_DATASET_DEFAULT,
                frequency=batch.get("frequency") or BATCH_FREQUENCY_DEFAULT,
                container=container)
            named = [(sanitise_filename(name), data) for name, data in uploads]
        except ManualIntakeError as exc:
            raise OpsError("OPS_UPLOAD_REFUSED", str(exc), 400) from exc
        for name, data in named:
            if not data:
                raise OpsError("OPS_UPLOAD_REFUSED",
                               f"'{name}' is empty. Send the file again.", 400)

        # 2. Prove the two doors agree BEFORE anything is written. Files placed
        #    here are also seen by the automated route, so if that route would
        #    identify this location as a different delivery the result is two
        #    half-packs for one delivery. Refuse instead — nothing is written,
        #    so there is nothing to unpick.
        self._require_converged_identity(batch, prefix, container)

        # 3. Place every source file, then register every source file. Both
        #    complete before anything downstream is told the pack is ready.
        with tempfile.TemporaryDirectory(prefix="ops_upload_") as td:
            staged: List[Tuple[Path, str]] = []
            for name, data in named:
                uri = derive_blob_uri(prefix, name)
                self.store.storage.write_bytes(uri, data)
                local = Path(td) / name
                local.write_bytes(data)
                staged.append((local, uri))
                self.store.append_audit(
                    client_id, "manual_delivery_file_placed", actor=received_by,
                    detail={"batch_id": batch["batch_id"], "filename": name,
                            "size": len(data)})
            for local, uri in staged:
                batch = self.intake.register_file(
                    batch, local, received_by_or_source=received_by,
                    source_uri=uri)

        batch["source_prefix"] = prefix
        self.intake.save_batch(batch)

        # 4. Only now assess readiness. The existing intake path decides from
        #    there — including whether to write the run manifest and open the
        #    workflow — exactly as it does for an automated arrival.
        return self.assess_batch(client_id=client_id,
                                 batch_id=batch["batch_id"], actor=received_by)

    def assess_batch(self, *, client_id: str, batch_id: str,
                     actor: str = "system") -> Dict[str, Any]:
        """Classify + reassess readiness; surface ambiguity decisions; start
        automatically when configured. Idempotent — called on every change."""
        batch = self.intake.load_batch(client_id, batch_id)
        if batch is None:
            raise OpsError("OPS_BATCH_NOT_FOUND",
                           "That input pack could not be found.", 404)
        if batch["status"] in ("running", "completed", "failed"):
            return batch

        # Role recognition using approved registry signatures when available.
        role_schemas, aliases = None, None
        try:
            rec = self._source_registry().lookup(
                batch["client_id"], batch["portfolio_id"], "funded", "monthly")
            if rec is not None:
                role_schemas = rec.file_role_schemas or None
                aliases = rec.file_role_aliases or None
        except Exception:  # noqa: BLE001
            pass
        batch = self.intake.classify(batch, role_schemas=role_schemas,
                                     aliases=aliases)

        # Configuration readiness (no persistence — the run pins its own).
        cfg = self._config_resolver().resolve(
            client_id=batch["client_id"], portfolio_id=batch["portfolio_id"],
            outcome=batch["workflow_type"],
            reporting_period=batch["reporting_date"], workflow_id="",
            created_by=actor)
        eff = cfg.effective

        # Ambiguity -> operator decisions (Review Centre), keyed to the batch.
        open_roles = self._sync_file_role_decisions(batch)
        batch = self.intake.assess(
            batch,
            config_status=cfg.status,
            config_id=(eff.effective_config_id if eff else ""),
            config_hash=(eff.content_hash if eff else ""),
            open_blocking_decisions=open_roles)

        if batch["status"] == "ready" and batch.get("auto_start_when_ready") \
                and not batch.get("workflow_id"):
            return self.start_batch(client_id=client_id,
                                    batch_id=batch["batch_id"], actor=actor)
        return batch

    def _sync_file_role_decisions(self, batch: Dict[str, Any]) -> List[str]:
        from .contracts import KIND_FILE_ROLE
        from .intake import load_requirements
        labels = (load_requirements().get("role_labels") or {})
        role_options = [{"value": r, "label": l} for r, l in labels.items()]
        open_ids: List[str] = []
        ambiguous = {f["original_filename"]: f for f in batch["files"]
                     if f.get("recognition_status") == "ambiguous"
                     and f.get("superseded_status") == "current"}
        for name, f in ambiguous.items():
            did = f"{batch['batch_id']}_role_{f['source_file_id']}"
            existing = self.store.load_decision(batch["client_id"], did)
            if existing is not None and existing.get("status") != DEC_OPEN:
                continue
            d = DecisionRequired(
                decision_id=did, kind=KIND_FILE_ROLE,
                title=f"Identify the file '{name}'",
                question=f"Trakt could not identify '{name}' with confidence. "
                         "What is it?",
                blocking=True, category="ambiguous_schema",
                severity="blocking", source_file=name,
                options=role_options, allowed_scopes=["file"],
                default_scope="file",
                subject={"artefact": "input_batch",
                         "batch_id": batch["batch_id"],
                         "source_file_id": f["source_file_id"]})
            doc = d.to_dict()
            doc.update({"status": DEC_OPEN, "workflow_id": batch["batch_id"],
                        "client_id": batch["client_id"], "stage": "received",
                        "created_at": (existing or {}).get("created_at")
                        or now_iso()})
            self.store.save_decision(batch["client_id"], doc)
            open_ids.append(did)
        return open_ids

    def start_batch(self, *, client_id: str, batch_id: str,
                    actor: str) -> Dict[str, Any]:
        """READY -> immutable internal run manifest -> governed execution.
        Idempotent: a batch that already started returns its workflow."""
        batch = self.intake.load_batch(client_id, batch_id)
        if batch is None:
            raise OpsError("OPS_BATCH_NOT_FOUND",
                           "That input pack could not be found.", 404)
        if batch.get("workflow_id"):
            return batch                       # duplicate trigger suppressed
        if batch["status"] != "ready":
            raise OpsError("OPS_BATCH_NOT_READY",
                           batch.get("status_reason") or
                           "This input pack is not ready to process.", 409)
        cfg = self._config_resolver().resolve(
            client_id=batch["client_id"], portfolio_id=batch["portfolio_id"],
            outcome=batch["workflow_type"],
            reporting_period=batch["reporting_date"], workflow_id="",
            created_by=actor)
        eff = cfg.effective
        manifest = self.intake.ensure_manifest(
            batch,
            effective_config={"effective_config_id":
                              (eff.effective_config_id if eff else ""),
                              "content_hash":
                              (eff.content_hash if eff else "")},
            approved_decisions=[d["decision_id"] for d in
                                self.store.list_decisions(
                                    client_id, workflow_id=batch_id)
                                if d.get("status") == DEC_APPROVED],
            expected_outputs=(["platform_canonical", "annex2_submission_xml"]
                              if batch["workflow_type"] == OUTCOME_MI_ANNEX2
                              else ["platform_canonical"]))
        delivery = self.register_delivery(
            client_id=client_id, portfolio_id=batch["portfolio_id"],
            input_path=str(self.intake.batch_dir(batch)),
            reporting_period=batch["reporting_date"], registered_by=actor)
        run = self.create_workflow(
            client_id=client_id, delivery_id=delivery["delivery_id"],
            outcome=batch["workflow_type"], created_by=actor)
        run.batch_id = batch["batch_id"]
        self.store.save_workflow(run)
        batch["workflow_id"] = run.workflow_id
        batch["status"] = "running"
        batch["status_reason"] = "Trakt is processing this input pack."
        batch["started_at"] = now_iso()
        self.intake.save_batch(batch)
        self.store.append_audit(
            client_id, "onboarding_started", actor=actor,
            workflow_id=run.workflow_id,
            detail={"batch_id": batch_id, "run_id": manifest["run_id"],
                    "idempotency_key": manifest["idempotency_key"],
                    "auto_start": batch.get("auto_start_when_ready", False)})
        if run.status == RUN_RECEIVED:
            self.start(run, actor=actor)
        return self.intake.load_batch(client_id, batch_id)

    def _update_batch_from_run(self, run: WorkflowRun) -> None:
        if not run.batch_id:
            return
        batch = self.intake.load_batch(run.client_id, run.batch_id)
        if batch is None:
            return
        mapping = {RUN_AWAITING_PUBLICATION: ("completed",
                                              "Processing finished — awaiting "
                                              "your approval to publish."),
                   RUN_PUBLISHED: ("completed", "Published."),
                   RUN_FAILED: ("failed", "Processing did not finish "
                                          "cleanly.")}
        new = mapping.get(run.status)
        if new and (batch["status"] != new[0]
                    or batch["status_reason"] != new[1]):
            batch["status"], batch["status_reason"] = new
            if new[0] == "completed" and not batch.get("completed_at"):
                batch["completed_at"] = now_iso()
            self.intake.save_batch(batch)

    def _resolve_effective_config(self, run: WorkflowRun):
        """Resolve + persist the immutable effective configuration for this
        run (a NEW version on every execution attempt, so reruns after
        decision approvals pin a fresh decision set while history remains)."""
        resolver = self._config_resolver()
        outcome = resolver.resolve(
            client_id=run.client_id, portfolio_id=run.portfolio_id,
            outcome=run.outcome, reporting_period=run.reporting_period,
            workflow_id=run.workflow_id, created_by=run.created_by or "system",
            version=run.rerun_count + 1)
        eff = outcome.effective
        if eff is not None:
            run.effective_config = {
                "effective_config_id": eff.effective_config_id,
                "version": eff.version,
                "content_hash": eff.content_hash,
                "status": outcome.status,
                "package_versions": {
                    "system": eff.system_config_version,
                    "regime": eff.regime_config_version,
                    "asset": eff.asset_config_version,
                },
                "decision_set_version": eff.decision_set_version,
            }
            self.store.save_workflow(run)
            self.store.append_audit(
                run.client_id, "effective_config_resolved",
                actor="system", workflow_id=run.workflow_id,
                detail={"status": outcome.status,
                        "effective_config_id": eff.effective_config_id,
                        "version": eff.version,
                        "content_hash": eff.content_hash,
                        "decision_set_version": eff.decision_set_version,
                        "conflicts": outcome.conflicts})
        return resolver, outcome

    def _build_adapters(self, run: WorkflowRun, recorder,
                        snapshot: Optional[Dict[str, str]] = None
                        ) -> GovernedAdapters:
        if self._adapter_factory is not None:
            inner = self._adapter_factory(run)
        else:
            from engine.orchestrator_agent.adapters import RealAgentAdapters
            approved = self._approved_decisions_path(run)
            deterministic = approved is not None
            snap = snapshot or {}
            inner = RealAgentAdapters(
                client_name=run.client_id,
                onboarding_mode="mi_only",
                # The agent consumes the immutable effective-configuration
                # snapshot, never live repository YAML.
                registry=snap.get("registry"),
                aliases_dir=snap.get("aliases_dir", "config/system"),
                processing_mode=("deterministic" if deterministic
                                 else "source_onboarding"),
                mapping_config_path=(str(approved) if approved else None),
                dataset=run.delivery.get("dataset", "funded") if
                run.delivery.get("dataset") == "pipeline" else "",
                # Annex 2 runs the typed Gate 2/3 path over the MI contract,
                # matching the proven route's canonical preparation.
                full_pipeline=(run.outcome == OUTCOME_MI_ANNEX2),
                reporting_period=run.reporting_period,
                managed_service=True)
        return GovernedAdapters(inner, recorder)

    def _execute(self, client_id: str, workflow_id: str) -> None:
        from engine.orchestrator_agent.adapters import PortfolioSpec
        from engine.orchestrator_agent.orchestrator import run_orchestration
        from engine.orchestrator_agent.state import RunState

        run = self.store.load_workflow(client_id, workflow_id)
        if run is None:
            return

        staging = self._staging_dir(run)
        staging.mkdir(parents=True, exist_ok=True)

        # Materialise applicable approved rules into client memory (existing
        # MappingMemoryStore artefact) before the agents run.
        applicable = self.rules.applicable(
            client_id=run.client_id, portfolio_id=run.portfolio_id,
            file_ref=run.delivery.get("schema_fingerprint", ""))
        if applicable:
            project_rules_to_client_memory(
                applicable, run.client_id,
                memory_dir=self.memory_root / run.client_id / "client_memory")

        # Resume from persisted orchestrator state when it exists.
        resume_state: Optional[RunState] = None
        orun_id = run.orchestrator_run_id or f"orun_{run.workflow_id}"
        state_path = staging / orun_id / "run_state.json"
        if state_path.exists():
            resume_state = RunState.load(state_path)
            for p in resume_state.portfolios:
                for s in p.steps.values():
                    if s.status in ("halted", "failed", "running"):
                        s.status = "pending"
                if p.status in ("halted", "failed", "running"):
                    p.status = "pending"
            for s in (resume_state.assemble, resume_state.route,
                      resume_state.project):
                if s.status in ("halted", "failed", "running"):
                    s.status = "pending"
            resume_state.force_publish = resume_state.force_publish or \
                self._validation_exception_approved(run)

        # Immutable effective configuration — resolved and persisted BEFORE
        # the agents run; the run pins this version permanently.
        resolver, cfg_outcome = self._resolve_effective_config(run)
        if cfg_outcome.status == "BLOCKED":
            self._park_config_blocked(run, cfg_outcome)
            return
        snapshot: Optional[Dict[str, str]] = None
        if cfg_outcome.effective is not None and self._adapter_factory is None:
            snapshot = resolver.materialise_snapshot(
                cfg_outcome.effective,
                staging / f"config_snapshot_v{cfg_outcome.effective.version:04d}")

        def recorder(step: str, result) -> None:
            self._on_step(client_id, workflow_id, step, result)

        adapters = self._build_adapters(run, recorder, snapshot)
        spec = PortfolioSpec(
            source_portfolio_id=run.portfolio_id,
            input=run.delivery.get("input_path", ""),
            source_portfolio_type=("direct" if run.workflow_type == WF_NEW_CLIENT
                                   else None),
            allow_unknown_acquisition_date=True)

        state = run_orchestration(
            run.client_id, [spec],
            target=self._orchestrator_target(run),
            out_root=str(staging), adapters=adapters,
            created_at=now_iso(), run_id=orun_id,
            resume_state=resume_state,
            full_pipeline=(run.outcome == OUTCOME_MI_ANNEX2),
            force_publish=self._validation_exception_approved(run))

        run = self.store.load_workflow(client_id, workflow_id) or run
        run.orchestrator_run_id = state.run_id
        run.staging_root = str(staging)
        self._apply_run_state(run, state)

    def _apply_run_state(self, run: WorkflowRun, state) -> None:
        """Translate the final orchestrator state into stage results + the
        workflow status, persist everything, and prepare publication if ready."""
        work_dir = (Path(state.out_root) / state.run_id / "portfolios"
                    / run.portfolio_id)
        results = translate_run_state(run, state, work_dir)
        open_blocking = False
        blocked = False
        for stage, gar in results.items():
            self.store.save_result(run.client_id, gar)
            run.set_stage(stage, gar.status, gar.result_id)
            if gar.status == ST_NEEDS_REVIEW:
                open_blocking = True
            if gar.status == ST_BLOCKED:
                blocked = True
            self._sync_decisions(run, gar)

        run.blockers = [b for g in results.values() for b in g.blockers]
        if state.status == "done":
            if run.outcome == OUTCOME_MI_ANNEX2:
                # Canonical preparation done — continue into the governed
                # Annex 2 delivery chain (preflight -> projection ->
                # normalisation -> XML+XSD). It sets the final status itself.
                self.store.save_workflow(run)
                self._run_annex2_chain(run, state)
                return
            new_status = RUN_AWAITING_PUBLICATION
        elif open_blocking:
            new_status = RUN_NEEDS_REVIEW
        elif blocked or state.status == "halted":
            new_status = RUN_BLOCKED
        else:
            new_status = RUN_FAILED
        if run.status != new_status:
            try:
                transition(run, new_status)
            except Exception:
                run.status = new_status
        if new_status == RUN_AWAITING_PUBLICATION:
            self._prepare_publication(run, state)
        self.store.save_workflow(run)
        self.store.clear_lease(run)
        self.store.append_event(run, "execution_finished",
                                detail={"status": run.status})
        self._update_batch_from_run(run)

    # ------------------------------------------------------------------ #
    # Governed Annex 2 delivery chain (I1-I4)
    # ------------------------------------------------------------------ #
    def _client_rule_overrides(self, run: WorkflowRun) -> Dict[str, Any]:
        """Approved client_rule settings from the governed rule store."""
        out: Dict[str, Any] = {}
        for r in self.rules.applicable(client_id=run.client_id,
                                       portfolio_id=run.portfolio_id):
            if r.kind == KIND_CLIENT_RULE:
                p = r.payload or {}
                if p.get("setting"):
                    out[str(p["setting"])] = p.get("value")
        return out

    def _save_stage_gar(self, run: WorkflowRun, stage: str, status: str,
                        summary: str, *, why: str = "",
                        warnings=None, blockers=None, decisions=None,
                        evidence=None, inputs=None, outputs=None) -> None:
        gar = GovernedAgentResult(
            run_id=run.workflow_id, stage=stage, status=status,
            summary=summary, why_it_matters=why,
            decisions_required=decisions or [],
            warnings=warnings or [], blockers=blockers or [],
            evidence=evidence or [],
            input_references=inputs or [], output_references=outputs or [],
            agent_version="trakt-engine", started_at=now_iso(),
            completed_at=now_iso())
        self.store.save_result(run.client_id, gar)
        run.set_stage(stage, status, gar.result_id)
        self._sync_decisions(run, gar)
        self.store.save_workflow(run)

    def _park(self, run: WorkflowRun, status: str) -> None:
        if run.status != status:
            try:
                transition(run, status)
            except Exception:  # noqa: BLE001
                run.status = status
        self.store.save_workflow(run)
        self.store.clear_lease(run)
        self.store.append_event(run, "execution_finished",
                                detail={"status": run.status})
        self._update_batch_from_run(run)

    def _park_config_blocked(self, run: WorkflowRun, cfg_outcome) -> None:
        """Configuration resolution blocked BEFORE any agent ran: park with
        actionable client_rule decisions (the operator supplies the missing
        regulatory details; approval regenerates the effective configuration
        on rerun)."""
        decisions = []
        for c in cfg_outcome.missing:
            decisions.append(DecisionRequired(
                decision_id=f"{run.workflow_id}_cfg_{c['key']}",
                kind=KIND_CLIENT_RULE,
                title=f"Provide the {c['label'].lower()}",
                question=f"{c['problem']} Please provide it.",
                blocking=True, category="missing_required_client_value",
                severity="blocking",
                options=[], allowed_scopes=["client"], default_scope="client",
                evidence=[{"label": "What was found", "kind": "text",
                           "data": {"item": c["label"],
                                    "found": c.get("found_value")
                                    or "nothing configured"}}],
                subject={"artefact": "regulatory_config",
                         "setting": c["key"], "label": c["label"]}))
        summary = (f"{len(decisions)} regulatory detail"
                   f"{'s are' if len(decisions) != 1 else ' is'} needed "
                   "before this run can start.")
        self._save_stage_gar(
            run, STAGE_REG_CONFIG, ST_NEEDS_REVIEW, summary,
            why="These details identify your organisation to the regulator "
                "and appear in every report.",
            decisions=decisions, warnings=cfg_outcome.warnings)
        self._park(run, RUN_NEEDS_REVIEW)

    def _run_annex2_chain(self, run: WorkflowRun, state) -> None:
        """Preflight -> projection -> delivery normalisation -> XML+XSD, each
        a governed stage over the UNMODIFIED authoritative components.
        Idempotent: completed stages with intact artefacts are skipped."""
        runners = self._annex2_runners()
        chain_dir = self._staging_dir(run) / "annex2"
        chain_dir.mkdir(parents=True, exist_ok=True)
        a2 = dict(run.annex2 or {})

        def done(stage: str, artefact_key: Optional[str] = None) -> bool:
            if run.stage_status(stage) != ST_COMPLETED:
                return False
            if artefact_key:
                p = a2.get(artefact_key)
                return bool(p) and Path(str(p)).exists()
            return True

        # ---- Stage: regulatory details preflight (I3) -------------------- #
        from .annex2 import preflight as _pf
        overrides = self._client_rule_overrides(run)
        pf = _pf.run_preflight(client_config_path=self.client_config_path,
                               overrides=overrides,
                               reporting_period=run.reporting_period)
        if pf["status"] == "blocked":
            decisions = []
            for c in pf["blocked"]:
                decisions.append(DecisionRequired(
                    decision_id=f"{run.workflow_id}_cfg_{c['key']}",
                    kind=KIND_CLIENT_RULE,
                    title=f"Provide the {c['label'].lower()}",
                    question=f"{c['problem']} Please provide it.",
                    blocking=True,
                    options=[],
                    allowed_scopes=["client"], default_scope="client",
                    evidence=[{"label": "What was found", "kind": "text",
                               "data": {"item": c["label"],
                                        "found": c.get("found_value") or
                                        "nothing configured"}}],
                    subject={"artefact": "regulatory_config",
                             "setting": c["key"], "label": c["label"]}))
            self._save_stage_gar(
                run, STAGE_REG_CONFIG, ST_NEEDS_REVIEW, pf["summary"],
                why="These details identify your organisation to the "
                    "regulator and appear in every report.",
                decisions=decisions,
                warnings=[c["problem"] for c in pf["needs_review"]])
            for s in (STAGE_PROJECTION, STAGE_DELIVERY_PREP, STAGE_XML):
                if run.stage_status(s) == ST_WAITING:
                    self._save_stage_gar(run, s, ST_WAITING,
                                         "Waiting for the regulatory details.")
            self._save_stage_gar(run, STAGE_PUBLICATION, ST_WAITING,
                                 "Waiting for all steps to finish.")
            self._park(run, RUN_NEEDS_REVIEW)
            return
        self._save_stage_gar(
            run, STAGE_REG_CONFIG, ST_COMPLETED, pf["summary"],
            warnings=[c["problem"] for c in pf["needs_review"]],
            evidence=[{"label": "Details checked", "kind": "table",
                       "data": [{"item": c["label"], "status": c["status"]}
                                for c in pf["checks"]]},
                      {"label": "Configuration versions used", "kind": "table",
                       "data": {**(run.effective_config.get(
                           "package_versions") or {}),
                           "effective_configuration":
                           run.effective_config.get("content_hash", "")[:23]}}])

        # ---- Stage: projection (proven projector via existing seam) ------ #
        if not done(STAGE_PROJECTION, "projected_csv"):
            self._save_stage_gar(run, STAGE_PROJECTION, ST_RUNNING,
                                 "Projecting the regulatory data.")
            effective_cfg = None
            if overrides:
                effective_cfg = _pf.materialise_effective_config(
                    base_config_path=self.client_config_path,
                    overrides=overrides,
                    out_path=chain_dir / "effective_client_config.yaml")
                a2["effective_config"] = str(effective_cfg)
            out = runners.run_projection(
                central_canonical=state.central_canonical_path,
                out_dir=chain_dir / "projection",
                client_config=effective_cfg)
            # Population-path reconciliation (I2) — persisted evidence; only
            # genuinely unsupported required outputs may block.
            from .annex2 import population as _pop
            recon = _pop.reconciliation_document()
            recon_path = chain_dir / "population_reconciliation.json"
            recon_path.write_text(json.dumps(recon, indent=2),
                                  encoding="utf-8")
            a2["population_reconciliation"] = str(recon_path)
            self.store.append_audit(
                run.client_id, "annex2_population_reconciliation",
                actor="system", workflow_id=run.workflow_id,
                detail={"blocking_count": recon["blocking_count"],
                        "blocking_codes": recon["blocking_codes"],
                        "mechanism_counts": recon["mechanism_counts"]})
            if not out.ok:
                self._save_stage_gar(run, STAGE_PROJECTION, ST_BLOCKED,
                                     out.summary, blockers=out.blockers,
                                     warnings=out.warnings)
                run.annex2 = a2
                self._park(run, RUN_BLOCKED)
                return
            recon_warn = ([] if recon["blocking_count"] == 0
                          else [recon["summary_sentence"]])
            a2["projected_csv"] = out.artefacts.get("projected_csv", "")
            a2["projection_metrics"] = out.metrics
            run.annex2 = a2
            self._save_stage_gar(
                run, STAGE_PROJECTION,
                ST_BLOCKED if recon["blocking_count"] else ST_COMPLETED,
                out.summary,
                warnings=out.warnings + recon_warn,
                blockers=([recon["summary_sentence"]]
                          if recon["blocking_count"] else []),
                evidence=[{"label": "How each regulatory field is filled",
                           "kind": "table",
                           "data": recon["mechanism_counts"]}],
                inputs=[state.central_canonical_path or ""],
                outputs=[a2["projected_csv"]])
            if recon["blocking_count"]:
                self._park(run, RUN_BLOCKED)
                return

        # ---- Stage: delivery normalisation (I1 stage 1) ------------------ #
        if not done(STAGE_DELIVERY_PREP, "delivery_ready_csv"):
            self._save_stage_gar(run, STAGE_DELIVERY_PREP, ST_RUNNING,
                                 "Preparing the delivery data.")
            out = runners.run_normalisation(
                projected_csv=a2["projected_csv"],
                out_dir=chain_dir / "delivery")
            audit_detail = {"metrics": out.metrics,
                            "warning_count": len(out.warnings),
                            "blocker_count": len(out.blockers)}
            self.store.append_audit(run.client_id, "annex2_delivery_prep",
                                    actor="system",
                                    workflow_id=run.workflow_id,
                                    detail=audit_detail)
            if not out.ok:
                self._save_stage_gar(
                    run, STAGE_DELIVERY_PREP, ST_BLOCKED, out.summary,
                    why="Publishing figures that fail the delivery checks "
                        "could mislead the regulator.",
                    warnings=out.warnings, blockers=out.blockers,
                    evidence=out.evidence,
                    inputs=[a2.get("projected_csv", "")])
                run.annex2 = a2
                self._park(run, RUN_BLOCKED)
                return
            a2["delivery_ready_csv"] = out.artefacts.get("delivery_ready_csv", "")
            a2["delivery_metrics"] = out.metrics
            run.annex2 = a2
            self._save_stage_gar(
                run, STAGE_DELIVERY_PREP, ST_COMPLETED, out.summary,
                warnings=out.warnings, evidence=out.evidence,
                inputs=[a2.get("projected_csv", "")],
                outputs=[a2["delivery_ready_csv"]])

        # ---- Stage: XML generation + XSD validation (I1 stage 2, I4) ----- #
        if not done(STAGE_XML, "xml"):
            self._save_stage_gar(run, STAGE_XML, ST_RUNNING,
                                 "Creating the XML and checking it against "
                                 "the regulator's format.")
            out = runners.run_xml(delivery_csv=a2["delivery_ready_csv"],
                                  out_dir=chain_dir / "xml")
            self.store.append_audit(
                run.client_id, "annex2_xml_generation", actor="system",
                workflow_id=run.workflow_id,
                detail={"metrics": {k: v for k, v in out.metrics.items()
                                    if k != "xml_sha256"},
                        "xml_sha256": out.metrics.get("xml_sha256", ""),
                        "interventions": out.report.get("interventions", [])})
            if not out.ok:
                self._save_stage_gar(run, STAGE_XML, ST_BLOCKED, out.summary,
                                     blockers=out.blockers,
                                     warnings=out.warnings)
                run.annex2 = a2
                self._park(run, RUN_BLOCKED)
                return
            a2.update({
                "xml": out.artefacts.get("xml", ""),
                "interventions": out.artefacts.get("interventions", ""),
                "xml_metrics": out.metrics,
                "xsd": out.metrics.get("xsd", ""),
                "xsd_result": out.metrics.get("xsd_result", ""),
                "xml_sha256": out.metrics.get("xml_sha256", ""),
            })
            run.annex2 = a2
            self._save_stage_gar(
                run, STAGE_XML, ST_COMPLETED, out.summary,
                why="Nothing is sent to the regulator until you approve it.",
                warnings=out.warnings, evidence=out.evidence,
                inputs=[a2.get("delivery_ready_csv", "")],
                outputs=[a2["xml"]])

        # ---- Ready for approval ------------------------------------------ #
        run.annex2 = a2
        xml_warns = list((self.store.load_result(
            run.client_id, run.workflow_id, STAGE_XML) or
            GovernedAgentResult(run_id="", stage="", status="", summary="")
        ).warnings)
        self._save_stage_gar(
            run, STAGE_PUBLICATION, ST_READY,
            "The Annex 2 delivery is prepared and waiting for your approval "
            "to publish.",
            why="Nothing is published until you approve it.",
            warnings=xml_warns,
            decisions=[DecisionRequired(
                decision_id=f"{run.workflow_id}_publish",
                kind=KIND_PUBLICATION,
                title="Approve publication",
                question="Publish this Annex 2 delivery as the latest "
                         "official version?",
                blocking=True,
                options=[{"value": "publish", "label": "Publish"},
                         {"value": "hold", "label": "Hold — not yet"}],
                default_scope="file",
                subject={"artefact": "publication"})])
        self._prepare_publication(run, state)
        self._park(run, RUN_AWAITING_PUBLICATION)

    def _on_step(self, client_id: str, workflow_id: str, step: str, result) -> None:
        """Live progress: persist a light stage-status update as each agent
        step completes, so polling clients see movement mid-run."""
        run = self.store.load_workflow(client_id, workflow_id)
        if run is None:
            return
        from .adapters import STEP_TO_STAGE
        stage = STEP_TO_STAGE.get(step)
        if stage:
            status = ST_COMPLETED if result.ok else (
                ST_NEEDS_REVIEW if result.blocking else ST_BLOCKED)
            run.set_stage(stage, status)
            self.store.save_workflow(run)

    # ------------------------------------------------------------------ #
    # Decisions
    # ------------------------------------------------------------------ #
    def _sync_decisions(self, run: WorkflowRun, gar: GovernedAgentResult) -> None:
        """Persist newly-surfaced decisions; keep already-resolved ones."""
        for d in gar.decisions_required:
            existing = self.store.load_decision(run.client_id, d.decision_id)
            if existing is not None and existing.get("status") != DEC_OPEN:
                continue
            doc = d.to_dict()
            doc.update({"status": DEC_OPEN, "workflow_id": run.workflow_id,
                        "client_id": run.client_id, "stage": gar.stage,
                        "created_at": existing.get("created_at") if existing
                        else now_iso()})
            self.store.save_decision(run.client_id, doc)

    def resolve_decision(self, *, client_id: str, decision_id: str, action: str,
                         actor: str, value: str = "", scope: str = "portfolio",
                         reason: str = "",
                         actor_is_admin: bool = False) -> Dict[str, Any]:
        from .rules import ADMIN_ONLY_SCOPES
        if scope in ADMIN_ONLY_SCOPES and not actor_is_admin:
            raise OpsError("OPS_ADMIN_SCOPE",
                           "Only an administrator can apply a decision at "
                           "this level.", 403)
        return self._resolve_decision_inner(
            client_id=client_id, decision_id=decision_id, action=action,
            actor=actor, value=value, scope=scope, reason=reason)

    def _resolve_decision_inner(self, *, client_id: str, decision_id: str,
                                action: str, actor: str, value: str = "",
                                scope: str = "portfolio",
                                reason: str = "") -> Dict[str, Any]:
        """Approve / reject / amend one decision. Approvals become governed,
        scoped rules; when no blocking decisions remain open, the affected
        stage is rerun automatically. Duplicate resolutions are rejected."""
        doc = self.store.load_decision(client_id, decision_id)
        if doc is None:
            raise OpsError("OPS_DECISION_NOT_FOUND",
                           "That decision could not be found.", 404)
        if doc.get("status") != DEC_OPEN:
            raise OpsError("OPS_DECISION_ALREADY_RESOLVED",
                           "This decision has already been resolved.", 409)
        if action not in ("approve", "reject", "amend"):
            raise OpsError("OPS_BAD_ACTION", "Choose approve, reject or amend.", 400)

        # Input-pack file identification: applies to the BATCH, not a workflow.
        from .contracts import KIND_FILE_ROLE
        if doc.get("kind") == KIND_FILE_ROLE:
            if action == "reject":
                if not reason:
                    raise OpsError("OPS_REASON_REQUIRED",
                                   "Please say why you are rejecting this.", 400)
                doc.update(status=DEC_REJECTED, resolved_by=actor,
                           resolved_at=now_iso(), resolution_reason=reason)
                self.store.save_decision(client_id, doc)
                return {"decision": doc, "rule": None, "rerun_scheduled": False}
            if not value:
                raise OpsError("OPS_VALUE_REQUIRED",
                               "Choose what this file is.", 400)
            subject = doc.get("subject") or {}
            batch = self.intake.load_batch(client_id,
                                           subject.get("batch_id", ""))
            if batch is None:
                raise OpsError("OPS_BATCH_NOT_FOUND",
                               "That input pack could not be found.", 404)
            self.intake.override_classification(
                batch, subject.get("source_file_id", ""), value, actor=actor)
            doc.update(status=DEC_APPROVED, resolved_by=actor,
                       resolved_at=now_iso(), resolution_value=value,
                       resolution_scope="file")
            self.store.save_decision(client_id, doc)
            batch = self.assess_batch(client_id=client_id,
                                      batch_id=batch["batch_id"], actor=actor)
            return {"decision": doc, "rule": None,
                    "rerun_scheduled": batch.get("status") == "running"}

        run = self.store.load_workflow(client_id, doc.get("workflow_id", ""))
        if run is None:
            raise OpsError("OPS_WORKFLOW_NOT_FOUND",
                           "The workflow for this decision no longer exists.", 404)

        if action == "reject":
            if not reason:
                raise OpsError("OPS_REASON_REQUIRED",
                               "Please say why you are rejecting this.", 400)
            doc.update(status=DEC_REJECTED, resolved_by=actor,
                       resolved_at=now_iso(), resolution_reason=reason)
            self.store.save_decision(client_id, doc)
            self.store.append_audit(client_id, "decision_rejected", actor=actor,
                                    workflow_id=run.workflow_id,
                                    decision_id=decision_id,
                                    detail={"reason": reason})
            return {"decision": doc, "rule": None, "rerun_scheduled": False}

        chosen = value or (doc.get("recommendation") or {}).get("value", "")
        if doc.get("kind") == KIND_PUBLICATION and not chosen:
            chosen = "publish"
        if not chosen:
            raise OpsError("OPS_VALUE_REQUIRED",
                           "Choose an option before approving.", 400)

        # Publication decisions route through the explicit publication gate —
        # they never persist a rule and never bypass the promote path.
        if doc.get("kind") == KIND_PUBLICATION:
            if chosen == "publish":
                pub = self.approve_publication(client_id=client_id,
                                               workflow_id=run.workflow_id,
                                               actor=actor)
                doc.update(status=DEC_APPROVED, resolved_by=actor,
                           resolved_at=now_iso(), resolution_value=chosen)
                self.store.save_decision(client_id, doc)
                return {"decision": doc, "rule": None,
                        "rerun_scheduled": False, "publication": pub}
            pub = self.reject_publication(
                client_id=client_id, workflow_id=run.workflow_id, actor=actor,
                reason=reason or "Held by operator")
            doc.update(status=DEC_REJECTED, resolved_by=actor,
                       resolved_at=now_iso(), resolution_value=chosen,
                       resolution_reason=reason)
            self.store.save_decision(client_id, doc)
            return {"decision": doc, "rule": None,
                    "rerun_scheduled": False, "publication": pub}

        rule = None
        if doc.get("kind") in ("field_mapping", "alias", "enum", "transformation",
                               KIND_VALIDATION_EXCEPTION, KIND_CLIENT_RULE):
            rule = self._persist_rule(run, doc, chosen, scope, actor, reason)

        doc.update(status=DEC_APPROVED, resolved_by=actor, resolved_at=now_iso(),
                   resolution_action=action,
                   resolution_value=chosen, resolution_scope=scope,
                   resolution_reason=reason,
                   rule_id=(rule.rule_id if rule else None),
                   rule_version=(rule.version if rule else None))
        self.store.save_decision(client_id, doc)
        self.store.append_audit(client_id, "decision_approved", actor=actor,
                                workflow_id=run.workflow_id,
                                decision_id=decision_id,
                                rule_id=(rule.rule_id if rule else ""),
                                detail={"action": action, "value": chosen,
                                        "scope": scope})

        self._write_approved_decisions_file(run)

        rerun_scheduled = False
        # Rerun only when the whole review batch is resolved — never mid-batch,
        # so a rerun cannot race decisions the operator is still making.
        still_open = [d for d in self.store.open_decisions(client_id,
                                                           run.workflow_id)
                      if d.get("kind") != KIND_PUBLICATION]
        if not still_open and run.status in (RUN_NEEDS_REVIEW, RUN_BLOCKED):
            stage = doc.get("stage", "")
            run.set_stage(stage or STAGE_MAPPING, ST_APPROVED)
            self.store.save_workflow(run)
            self.rerun(run, actor=actor)
            rerun_scheduled = True
        return {"decision": doc, "rule": (rule.to_dict() if rule else None),
                "rerun_scheduled": rerun_scheduled}

    def _persist_rule(self, run: WorkflowRun, doc: Dict[str, Any], value: str,
                      scope: str, actor: str, reason: str) -> RuleRecord:
        subject = doc.get("subject") or {}
        kind = doc["kind"]
        payload: Dict[str, Any]
        if kind in ("field_mapping", "alias"):
            payload = {"source_column": subject.get("source_column")
                       or subject.get("target_field", ""),
                       "canonical_field": value}
            desc = (f"Treat '{subject.get('source_column') or subject.get('target_field')}' "
                    f"as '{value.replace('_', ' ')}'.")
        elif kind == "enum":
            payload = {"field": subject.get("field", ""),
                       "source_value": subject.get("source_value", ""),
                       "canonical_value": value}
            desc = (f"Read '{subject.get('source_value')}' as '{value}' for "
                    f"{subject.get('field', 'this field')}.")
        elif kind == KIND_VALIDATION_EXCEPTION:
            payload = {"check": subject.get("artefact", "validation"),
                       "disposition": value, "justification": reason}
            desc = "Accepted flagged checks for this delivery."
            scope = "file"      # validation exceptions never generalise silently
        elif kind == KIND_CLIENT_RULE:
            payload = {"setting": subject.get("setting", ""), "value": value}
            desc = (f"Set {subject.get('label') or subject.get('setting')} "
                    f"for this client.")
            scope = "client"    # regulatory identity is client-scoped
        else:
            payload = {"subject": subject.get("decision_id", ""),
                       "selected_action": value}
            desc = f"Approved treatment: {value.replace('_', ' ')}."
        rec = (doc.get("recommendation") or {})
        rule = RuleRecord(
            rule_id="", version=0, kind=kind, scope=scope,
            client_id=(run.client_id if scope not in ("global", "asset")
                       else ""),
            portfolio_id=(run.portfolio_id
                          if scope in ("portfolio", "file", "run") else ""),
            file_ref=(run.delivery.get("schema_fingerprint", "")
                      if scope == "file" else ""),
            run_ref=(run.workflow_id if scope == "run" else ""),
            payload=payload, description=desc,
            suggested_by=rec.get("source", "operator"),
            confidence=rec.get("confidence"),
            decision_id=doc["decision_id"], workflow_id=run.workflow_id,
            approved_by=actor, reason=reason)
        rule = self.rules.approve(rule)
        if rule.kind in ("field_mapping", "alias", "enum") and scope != "file":
            project_rules_to_client_memory(
                [rule], run.client_id,
                memory_dir=self.memory_root / run.client_id / "client_memory")
        self.store.append_audit(run.client_id, "rule_persisted", actor=actor,
                                workflow_id=run.workflow_id, rule_id=rule.rule_id,
                                detail={"kind": rule.kind, "scope": rule.scope,
                                        "version": rule.version})
        return rule

    def _write_approved_decisions_file(self, run: WorkflowRun) -> None:
        """Merge operator approvals into the run's pending 34_ decisions
        template, producing the approved YAML the onboarding agent applies
        deterministically on rerun (the existing decisions-bridge contract)."""
        work_dir = self._staging_dir(run)
        pending = _find_artifact(work_dir, DECISIONS_FILE)
        if pending is None:
            return
        try:
            docy = yaml.safe_load(pending.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            return
        resolved = {d.get("subject", {}).get("decision_id") or
                    d.get("decision_id", ""): d
                    for d in self.store.list_decisions(run.client_id,
                                                       workflow_id=run.workflow_id)
                    if d.get("status") == DEC_APPROVED}
        changed = False
        for entry in docy.get("decisions") or []:
            if not isinstance(entry, dict):
                continue
            match = resolved.get(str(entry.get("decision_id", "")))
            if match is None:
                match = resolved.get(f"{run.workflow_id}_{entry.get('decision_id')}")
            if match is None:
                continue
            entry["status"] = "approved"
            sel = match.get("resolution_value") \
                or entry.get("recommended_action") or entry.get("selected_action")
            if match.get("resolution_action") == "amend":
                # Operator supplied a custom value: record it as a configured
                # static value (an action the apply step already supports).
                entry["selected_action"] = "configure_static_value"
                entry["configured_value"] = match.get("resolution_value")
            else:
                entry["selected_action"] = sel
            # Per-action confirmation fields the deterministic apply requires.
            if sel == "mark_not_applicable":
                entry["not_applicable_confirmed"] = True
            if sel in ("confirm_default_or_nd", "confirm_ND_code"):
                entry["default_confirmed"] = True
            entry["operator_note"] = match.get("resolution_reason") or None
            entry["approved_by"] = match.get("resolved_by", "")
            entry["approved_at"] = match.get("resolved_at", "")
            changed = True
        if not changed:
            return
        out = pending.parent / APPROVED_DECISIONS_FILE
        out.write_text(yaml.safe_dump(docy, sort_keys=False), encoding="utf-8")
        # Durable copy in the operations-control container.
        self.store.storage.write_text(
            f"blob://{self.store.layout.container}/{run.client_id}/workflow-runs/"
            f"{run.workflow_id}/onboarding/{APPROVED_DECISIONS_FILE}",
            out.read_text(encoding="utf-8"))

    def _approved_decisions_path(self, run: WorkflowRun) -> Optional[Path]:
        p = _find_artifact(self._staging_dir(run), APPROVED_DECISIONS_FILE)
        return p

    def _validation_exception_approved(self, run: WorkflowRun) -> bool:
        for d in self.store.list_decisions(run.client_id,
                                           workflow_id=run.workflow_id):
            if (d.get("kind") == KIND_VALIDATION_EXCEPTION
                    and d.get("status") == DEC_APPROVED
                    and d.get("resolution_value") == "proceed"):
                return True
        return False

    # ------------------------------------------------------------------ #
    # Publication
    # ------------------------------------------------------------------ #
    def _prepare_publication(self, run: WorkflowRun, state) -> None:
        period = run.reporting_period or "unspecified"
        existing = self.store.load_publication(run.client_id, period)
        version = 1
        previous_id = None
        if existing is not None:
            if existing.get("workflow_id") == run.workflow_id \
                    and existing.get("status") in ("prepared", "approved", "published"):
                return
            version = int(existing.get("version") or 0) + 1
            previous_id = existing.get("publication_id")
        rule_set = {r.rule_id: r.version for r in self.rules.applicable(
            client_id=run.client_id, portfolio_id=run.portfolio_id,
            file_ref=run.delivery.get("schema_fingerprint", ""))}
        doc = {
            "publication_id": new_id("pub"),
            "client_id": run.client_id,
            "workflow_id": run.workflow_id,
            "workflow_type": run.workflow_type,
            "reporting_period": period,
            "status": "prepared",
            "version": version,
            "previous_publication_id": previous_id,
            "rule_versions": rule_set,
            "source_artefacts": {
                "central_canonical": state.central_canonical_path,
                "delivery_fingerprint": run.delivery.get("schema_fingerprint", ""),
                "orchestrator_run_id": state.run_id,
            },
            # Annex 2 delivery provenance (empty for MI-only outcomes):
            # projected input, normalised delivery data, XML, XSD + result,
            # intervention evidence, effective client configuration, hashes.
            "annex2": dict(run.annex2 or {}),
            "agent_versions": {
                "pipeline": "trakt-engine",
                "projector": "engine/gate_4_projection/regime_projector.py",
                "normaliser": "engine/gate_4b_delivery/"
                              "annex2_delivery_normalizer.py",
                "xml_builder": "engine/gate_5_delivery/xml_builder_annex2.py",
            } if run.outcome == OUTCOME_MI_ANNEX2 else
            {"pipeline": "trakt-engine"},
            "configuration_versions": {
                "outcome": run.outcome,
                "delivery_rules": (run.annex2 or {}).get(
                    "delivery_metrics", {}).get("rules_version", ""),
                "xsd": (run.annex2 or {}).get("xsd", ""),
                "client_config": (run.annex2 or {}).get(
                    "effective_config", "repository default"),
            },
            "prepared_at": now_iso(),
            "approved_by": None, "approved_at": None,
            "published_at": None, "rolled_back_by": None,
        }
        self.store.save_publication(run.client_id, doc)

    def _persistence(self):
        if self._persistence_factory is not None:
            return self._persistence_factory()
        from apps.blob_trigger_app.layout import Layout
        from apps.blob_trigger_app.persistence import ProductionPersistence
        return ProductionPersistence(storage=self.store.storage,
                                     layout=Layout.from_env())

    def approve_publication(self, *, client_id: str, workflow_id: str,
                            actor: str,
                            remember_scope: str = PUBLICATION_SCOPE_DEFAULT
                            ) -> Dict[str, Any]:
        """The ONLY path to production ``latest`` — explicit operator approval,
        then the EXISTING promotion path publishes.

        ``remember_scope`` records how far the operator intended this approval
        to reach. It is recorded and audited; it never widens what this call
        publishes, and the platform-wide scope is not reachable from here — a
        single delivery approval is not the place to set global policy.
        """
        if remember_scope not in PUBLICATION_SCOPES:
            raise OpsError("OPS_BAD_SCOPE",
                           "Choose how far this decision should apply.", 400)
        run = self.store.load_workflow(client_id, workflow_id)
        if run is None:
            raise OpsError("OPS_WORKFLOW_NOT_FOUND",
                           "That workflow could not be found.", 404)
        if run.status != RUN_AWAITING_PUBLICATION:
            raise OpsError("OPS_PUBLICATION_NOT_PREPARED",
                           "This workflow is not ready to publish.", 409)
        period = run.reporting_period or "unspecified"
        pub = self.store.load_publication(client_id, period)
        if pub is None or pub.get("workflow_id") != run.workflow_id:
            raise OpsError("OPS_PUBLICATION_NOT_PREPARED",
                           "This workflow is not ready to publish.", 409)
        if pub.get("status") not in ("prepared",):
            raise OpsError("OPS_ALREADY_PUBLISHED",
                           "This report has already been published.", 409)

        pub.update(status="approved", approved_by=actor, approved_at=now_iso(),
                   approval_scope=remember_scope)
        self.store.save_publication(client_id, pub)

        # Resolve any still-open publication decision for this workflow so it
        # leaves the Review Centre whichever surface the operator used.
        for d in self.store.open_decisions(client_id, workflow_id):
            if d.get("kind") == KIND_PUBLICATION:
                d.update(status=DEC_APPROVED, resolved_by=actor,
                         resolved_at=now_iso(), resolution_value="publish")
                self.store.save_decision(client_id, d)

        # EXISTING promotion paths only: platform canonical -> processed
        # latest+period; Annex 2 delivery artefacts -> the regime prefix.
        persistence = self._persistence()
        central = (pub.get("source_artefacts") or {}).get("central_canonical")
        published: Dict[str, Any] = {}
        if central and Path(str(central)).exists():
            published = persistence.persist_platform(client_id, period, str(central))
        a2 = pub.get("annex2") or {}
        if a2.get("xml") and Path(str(a2["xml"])).exists():
            regime_dir = str(Path(str(a2["xml"])).parent)
            published["regime"] = persistence.persist_regime_dir(
                client_id, period, regime_dir)
        pub.update(status="published", published_at=now_iso(),
                   published_artefacts=published)
        self.store.save_publication(client_id, pub)

        # Onboarding workflows also promote the source record via the EXISTING
        # approvals promote path, so future deliveries route deterministically.
        if run.workflow_type in (WF_NEW_CLIENT, WF_NEW_PORTFOLIO):
            try:
                self._promote_source(run, actor)
            except Exception:  # noqa: BLE001 — promotion failure must not lose publish
                logger.exception("source promotion failed for %s", run.workflow_id)

        try:
            transition(run, RUN_PUBLISHED)
        except Exception:
            run.status = RUN_PUBLISHED
        run.set_stage(STAGE_PUBLICATION, ST_COMPLETED)
        self.store.save_workflow(run)
        self._update_batch_from_run(run)
        self.store.append_event(run, "published", actor=actor,
                                detail={"publication_id": pub["publication_id"]})
        self.store.append_audit(client_id, "publication_approved", actor=actor,
                                workflow_id=run.workflow_id,
                                detail={"publication_id": pub["publication_id"],
                                        "version": pub.get("version"),
                                        "approval_scope": remember_scope})
        return pub

    def _promote_source(self, run: WorkflowRun, actor: str) -> None:
        from apps.blob_trigger_app import approvals as _approvals
        from apps.blob_trigger_app.layout import Layout
        layout = Layout.from_env()
        registry = self._source_registry()
        delivery = run.delivery
        art = _approvals.write_pending(
            self.store.storage, layout,
            kind=_approvals.KIND_NEW_SOURCE,
            client_id=run.client_id, source_book_type=None,
            dataset=delivery.get("dataset", "funded"),
            frequency=delivery.get("frequency", "monthly"),
            source_portfolio_id=run.portfolio_id,
            period=run.reporting_period or "unspecified",
            schema_fingerprint=delivery.get("schema_fingerprint", ""),
            detected_files=[f.get("name", "") for f in delivery.get("files", [])],
            created_at=now_iso())
        approved_path = self._approved_decisions_path(run)
        _approvals.approve(self.store.storage, layout, art["approval_id"],
                           mapping_id=f"ops_{run.workflow_id}",
                           mapping_config_path=(str(approved_path)
                                                if approved_path else None),
                           decided_by=actor, decided_at=now_iso())
        rec = _approvals.promote(self.store.storage, layout, registry,
                                 art["approval_id"])
        rec.last_successful_run_id = run.orchestrator_run_id
        rec.last_successful_reporting_period = run.reporting_period or ""
        registry.upsert(rec)

    def reject_publication(self, *, client_id: str, workflow_id: str,
                           actor: str, reason: str) -> Dict[str, Any]:
        if not reason:
            raise OpsError("OPS_REASON_REQUIRED",
                           "Please say why you are holding this report.", 400)
        run = self.store.load_workflow(client_id, workflow_id)
        if run is None:
            raise OpsError("OPS_WORKFLOW_NOT_FOUND",
                           "That workflow could not be found.", 404)
        if run.status != RUN_AWAITING_PUBLICATION:
            raise OpsError("OPS_PUBLICATION_NOT_PREPARED",
                           "This workflow is not awaiting publication.", 409)
        period = run.reporting_period or "unspecified"
        pub = self.store.load_publication(client_id, period)
        if pub is not None and pub.get("workflow_id") == workflow_id:
            pub.update(status="rejected", approved_by=actor,
                       approved_at=now_iso(), reject_reason=reason)
            self.store.save_publication(client_id, pub)
        transition(run, RUN_HELD)
        self.store.save_workflow(run)
        self.store.append_audit(client_id, "publication_rejected", actor=actor,
                                workflow_id=workflow_id,
                                detail={"reason": reason})
        return pub or {}

    # ------------------------------------------------------------------ #
    # Restart recovery
    # ------------------------------------------------------------------ #
    def recover_on_startup(self) -> List[str]:
        """Reconcile persisted state after a process restart: any workflow left
        in ``running`` with no live executor is marked interrupted and parked
        as blocked (resumable via rerun). Returns affected workflow ids."""
        recovered: List[str] = []
        for client_id in self.store.known_clients():
            for row in self.store.list_workflows(client_id):
                if row.get("status") != RUN_RUNNING:
                    continue
                if self._is_executing(row["workflow_id"]):
                    continue
                run = self.store.load_workflow(client_id, row["workflow_id"])
                if run is None:
                    continue
                run.interrupted = True
                run.blockers = ["This run was interrupted. Choose 'Run again' "
                                "to continue where it left off."]
                try:
                    transition(run, RUN_BLOCKED)
                except Exception:
                    run.status = RUN_BLOCKED
                self.store.save_workflow(run)
                self.store.clear_lease(run)
                self.store.append_event(run, "interrupted_recovered")
                recovered.append(run.workflow_id)
        return recovered
