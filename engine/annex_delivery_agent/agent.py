"""
engine.annex_delivery_agent.agent
=================================

The governed entry point: one agent, any annex.

Its job is everything that must be true regardless of which report is being
prepared — authorise the caller, resolve the report type, derive a deterministic
run identity, decide whether an existing run can be reused, drive the shared
lifecycle, persist the outcome, and return it inside the repository's existing
:class:`trakt_core.envelope.GovernedResult`.

What it deliberately does **not** do:

* know any annex's field codes, schema or rules — those live behind the provider;
* publish anything — that requires an explicit human decision through
  :mod:`engine.annex_delivery_agent.publication`;
* trust a caller-supplied client or portfolio identifier — the tenant always
  comes from the trusted :class:`~trakt_core.context.ExecutionContext`, and the
  portfolio is authorised against it before a single path is resolved.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from trakt_core.context import ExecutionContext
from trakt_core.envelope import (
    STATUS_BLOCKED, STATUS_ERROR, STATUS_PARTIAL_SUCCESS, STATUS_SUCCESS,
    AuditMetadata, GovernedResult, PolicyState, ProvenanceRef, SnapshotRef,
)
from trakt_core.errors import ErrorCode, TraktError
from trakt_core.tenancy import (
    AuthorisedPortfolio, TenantRegistry, authorise_portfolio_access, load_tenant_registry,
)

from .contracts import (
    DeliveryRequest, DeliveryResult, DeliveryStatus, PREPARED_STATUSES,
    STAGE_PREFLIGHT, stable_digest,
)
from .lifecycle import DeliveryLifecycle
from .persistence import DeliveryRunStore
from .publication import PublicationGate, PublicationSink
from .registry import AnnexRegistry, default_registry

CAPABILITY = "annex_delivery.prepare"

#: Default operational root. Deliberately not under any canonical or promoted
#: dataset location: delivery runs are operational state, not published data.
DEFAULT_STORE_ROOT = "out_annex_delivery"


class AnnexDeliveryAgent:
    """Prepare a regulatory annex artefact under governance."""

    def __init__(self, *, registry: Optional[AnnexRegistry] = None,
                 store: Optional[DeliveryRunStore] = None,
                 tenant_registry: Optional[TenantRegistry] = None,
                 publication_sink: Optional[PublicationSink] = None) -> None:
        self.registry = registry or default_registry()
        self.store = store or DeliveryRunStore(DEFAULT_STORE_ROOT)
        self._tenant_registry = tenant_registry
        self.publication_sink = publication_sink

    # -- discovery ---------------------------------------------------------- #
    def supported_annexes(self) -> list[Dict[str, Any]]:
        """What this deployment can prepare. Safe to expose to an operator UI."""
        return [d.to_dict() for d in self.registry.definitions()]

    # -- preparation -------------------------------------------------------- #
    def prepare(self, context: ExecutionContext,
                request: DeliveryRequest) -> GovernedResult[Dict[str, Any]]:
        """Run the delivery lifecycle for one annex and return a governed result."""
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        started = time.perf_counter()

        try:
            authorised = self._authorise(context, request)
            provider = self.registry.resolve(request.annex_id, request.annex_version)
        except TraktError as exc:
            return self._error_envelope(context, request, exc, started_at, started)

        definition = provider.definition
        run_id = self._run_id(context, request, authorised, provider)

        run_dir = self.store.open_run(
            tenant_id=context.tenant_id, annex_id=definition.annex_id,
            portfolio_id=authorised.selector, reporting_date=request.reporting_date,
            run_id=run_id)

        effective = _bind_request(request, authorised)
        self.store.write_request(run_dir, {
            "request": effective.to_dict(),
            "context": context.to_audit_dict(),
            "annex": definition.to_dict(),
            "run_id": run_id,
            "received_at": started_at,
        })

        fingerprint = run_id  # the id *is* the governed-input fingerprint
        checkpoint = self.store.load_checkpoint(run_dir, run_id=run_id,
                                                input_fingerprint=fingerprint)

        # -- idempotency ---------------------------------------------------- #
        if request.force_rerun:
            checkpoint.invalidate_from(STAGE_PREFLIGHT)
            self.store.save_checkpoint(run_dir, checkpoint)
        else:
            existing = self._reusable_result(run_dir)
            if existing is not None:
                return self._envelope(context, effective, existing, authorised,
                                      started_at, started, reused=True)

        checkpoint.attempts += 1
        self.store.save_checkpoint(run_dir, checkpoint)

        lifecycle = DeliveryLifecycle(provider, store=self.store, run_dir=run_dir,
                                      run_id=run_id, tenant_id=context.tenant_id)
        outcome = lifecycle.run(effective, checkpoint=checkpoint)
        self.store.save_checkpoint(run_dir, outcome.checkpoint)
        self.store.write_result(run_dir, outcome.result)

        return self._envelope(context, effective, outcome.result, authorised,
                              started_at, started)

    # -- approval / publication -------------------------------------------- #
    def gate_for(self, context: ExecutionContext, request: DeliveryRequest, *,
                 run_id: str) -> PublicationGate:
        """The approval/publication gate for one run, tenant-checked.

        Deliberately routed through the agent rather than constructed directly by
        a caller: this is where the caller's portfolio claim is authorised again,
        so an approval request cannot reach another tenant's run directory.
        """
        authorised = self._authorise(context, request)
        provider_definition = self.registry.definition(request.annex_id, request.annex_version)
        run_dir = self.store.run_dir(
            tenant_id=context.tenant_id, annex_id=provider_definition.annex_id,
            portfolio_id=authorised.selector, reporting_date=request.reporting_date,
            run_id=run_id)
        if not run_dir.exists():
            raise TraktError(ErrorCode.ARTEFACT_NOT_FOUND,
                             "There is no prepared report for this reference.",
                             request_id=context.request_id)
        return PublicationGate(store=self.store, run_dir=run_dir,
                               sink=self.publication_sink)

    # -- internals ---------------------------------------------------------- #
    def _authorise(self, context: ExecutionContext,
                   request: DeliveryRequest) -> AuthorisedPortfolio:
        """Tenant-bind the request. The caller cannot widen its own access here."""
        registry = self._tenant_registry
        if registry is None:
            registry = load_tenant_registry()
            self._tenant_registry = registry
        # ``client_id`` from the request is a label only. If it names a different
        # tenant, that is an attempt to act for someone else.
        claimed = str(request.client_id or "").strip()
        if claimed and claimed != context.tenant_id:
            raise TraktError(
                ErrorCode.TENANT_MISMATCH,
                "The requested client does not match the authenticated client.",
                request_id=context.request_id,
                details={"requested_client_id": claimed})
        return authorise_portfolio_access(context, request.portfolio_id, registry=registry)

    def _run_id(self, context: ExecutionContext, request: DeliveryRequest,
                authorised: AuthorisedPortfolio, provider: Any) -> str:
        """A deterministic identity for these governed inputs.

        Same tenant, portfolio, annex, version, reporting date, input file *and*
        configuration → same run id → the same directory, which is what makes a
        rerun idempotent instead of a second parallel artefact. Change any of
        them and the id moves, so a stale artefact can never masquerade as
        current.
        """
        definition = provider.definition
        input_path = Path(request.projected_input)
        input_stat = ""
        if input_path.is_file():
            stat = input_path.stat()
            input_stat = f"{stat.st_size}:{int(stat.st_mtime)}"

        config_digest = ""
        try:
            versions = provider.check_configuration(request, _QuietReport())
            config_digest = stable_digest([f"{k}={v}" for k, v in sorted(versions.items())])
        except Exception:  # noqa: BLE001 - preflight will report this properly
            config_digest = "unresolved"

        return "run_" + stable_digest([
            context.tenant_id, authorised.selector, authorised.run_id or "",
            definition.annex_id, definition.report_version,
            request.reporting_date, str(input_path), input_stat, config_digest,
            _options_digest(request.options),
        ])

    def _reusable_result(self, run_dir: Path) -> Optional[DeliveryResult]:
        """A completed run whose artefact is still intact, or ``None``.

        A partial artefact can never satisfy this: the build finalises atomically,
        so the artefact path either holds the complete file or does not exist,
        and the recorded size must still match.
        """
        result = self.store.read_result(run_dir)
        if result is None or result.status not in PREPARED_STATUSES:
            return None
        if result.artefact is None:
            return None
        path = Path(result.artefact.path)
        if not path.is_file() or path.stat().st_size != result.artefact.size_bytes:
            return None
        return result

    # -- envelopes ---------------------------------------------------------- #
    def _envelope(self, context: ExecutionContext, request: DeliveryRequest,
                  result: DeliveryResult, authorised: AuthorisedPortfolio,
                  started_at: str, started: float, *,
                  reused: bool = False) -> GovernedResult[Dict[str, Any]]:
        status = _envelope_status(result.status)
        payload = result.to_dict()
        payload["reused_existing_run"] = reused

        snapshot = SnapshotRef(
            source_kind="projected_dataset",
            dataset_label=Path(request.projected_input).name,
            reporting_date=result.reporting_date,
            snapshot_id=result.run_id,
            content_hash=next((str(s.get("content_hash") or "")
                               for s in result.source_references
                               if s.get("kind") == "projected_dataset"), None),
            row_count=result.record_count,
            approval_state=result.approval.state,
            source_portfolios=(authorised.selector,),
        )
        warnings = tuple(result.warnings)
        if reused:
            warnings = ("This report was prepared earlier from the same data and "
                        "configuration; the existing file was returned rather than "
                        "rebuilt.", *warnings)

        return GovernedResult(
            capability=CAPABILITY,
            status=status,
            request_id=context.request_id,
            tenant_id=context.tenant_id,
            correlation_id=context.correlation_id,
            portfolio_id=authorised.portfolio_id,
            snapshot=snapshot,
            result=payload,
            warnings=warnings,
            policy=PolicyState(
                data_approved=result.approval.approved,
                tenant_authorised=True,
                capability_allowed=True,
                notes=(f"status={result.status}",
                       f"publication_eligible={result.publication_eligible}")),
            provenance=ProvenanceRef(
                source_notes=tuple(dict(s) for s in result.source_references),
                snapshot=snapshot),
            audit=AuditMetadata(
                capability=CAPABILITY,
                request_id=context.request_id,
                tenant_id=context.tenant_id,
                actor_id=context.actor_id,
                actor_type=context.actor_type,
                channel=context.channel,
                outcome=result.status,
                started_at=started_at,
                duration_ms=int((time.perf_counter() - started) * 1000),
                correlation_id=context.correlation_id,
                portfolio_id=authorised.portfolio_id,
                snapshot_id=result.run_id),
            error=(TraktError(ErrorCode.ARTEFACT_INCOMPLETE, result.summary,
                              request_id=context.request_id)
                   if status == STATUS_BLOCKED else None),
        )

    def _error_envelope(self, context: ExecutionContext, request: DeliveryRequest,
                        exc: TraktError, started_at: str,
                        started: float) -> GovernedResult[Dict[str, Any]]:
        return GovernedResult(
            capability=CAPABILITY,
            status=STATUS_BLOCKED,
            request_id=context.request_id,
            tenant_id=context.tenant_id,
            correlation_id=context.correlation_id,
            portfolio_id=request.portfolio_id,
            result=None,
            policy=PolicyState(tenant_authorised=False, capability_allowed=False),
            audit=AuditMetadata(
                capability=CAPABILITY, request_id=context.request_id,
                tenant_id=context.tenant_id, actor_id=context.actor_id,
                actor_type=context.actor_type, channel=context.channel,
                outcome="blocked", started_at=started_at,
                duration_ms=int((time.perf_counter() - started) * 1000),
                correlation_id=context.correlation_id, error_code=exc.code),
            error=exc,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _QuietReport:
    """A preflight report that discards findings.

    Used only while deriving the run id, where the configuration fingerprints are
    wanted but the findings are not — the real preflight runs moments later and
    reports them properly.
    """

    def block(self, *args: Any, **kwargs: Any) -> None:
        return None

    def warn(self, *args: Any, **kwargs: Any) -> None:
        return None

    def add(self, *args: Any, **kwargs: Any) -> None:
        return None


def _bind_request(request: DeliveryRequest, authorised: AuthorisedPortfolio) -> DeliveryRequest:
    """Replace the caller's portfolio claim with the authorised one."""
    from dataclasses import replace
    return replace(request, portfolio_id=authorised.selector,
                   client_id=authorised.tenant_id)


def _options_digest(options: Any) -> str:
    items = sorted((str(k), str(v)) for k, v in dict(options or {}).items())
    return stable_digest([f"{k}={v}" for k, v in items])


def _envelope_status(delivery_status: str) -> str:
    if delivery_status in (DeliveryStatus.PREPARED, DeliveryStatus.APPROVED,
                           DeliveryStatus.PUBLISHED):
        return STATUS_SUCCESS
    if delivery_status in (DeliveryStatus.PREPARED_WITH_WARNINGS,
                           DeliveryStatus.AWAITING_APPROVAL):
        return STATUS_PARTIAL_SUCCESS
    if delivery_status in (DeliveryStatus.PREFLIGHT_BLOCKED,
                           DeliveryStatus.NORMALISATION_FAILED,
                           DeliveryStatus.VALIDATION_FAILED,
                           DeliveryStatus.REJECTED):
        return STATUS_BLOCKED
    if delivery_status == DeliveryStatus.BUILD_FAILED:
        return STATUS_ERROR
    return STATUS_PARTIAL_SUCCESS


__all__ = ["AnnexDeliveryAgent", "CAPABILITY", "DEFAULT_STORE_ROOT"]
