"""mi_agent_api/artefacts.py — the governed artefact-access capability.

    capability id:  artefact.investor_pack.get

One authorisation path for generated artefacts, shared by React
(``GET /mi/decks/download``) and Copilot
(``GET /v1/copilot/artifacts/latest/investor-deck``).

The defect this closes: the React route passed a caller-supplied ``client_id``
straight into ``decks.resolve_deck_local``, which resolves
``blob://{processed}/decks/{client_id}/latest/investor_pack.pptx`` from a
container shared across clients. Any authenticated user could name another
tenant and receive its investor pack. The Copilot tape route already implemented
the equivalent check and refused on mismatch — the rule existed, in one adapter.

Here the tenant comes from :class:`~trakt_core.context.ExecutionContext` and the
requested portfolio is authorised within it before any path is constructed, so
neither channel can be pointed at another tenant's artefact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from trakt_core.audit import emit_audit_event
from trakt_core.context import SCOPE_ARTEFACT_READ, ExecutionContext
from trakt_core.envelope import (
    STATUS_BLOCKED,
    STATUS_ERROR,
    STATUS_SUCCESS,
    AuditMetadata,
    GovernedResult,
    PolicyState,
)
from trakt_core.errors import ErrorCategory, ErrorCode, TraktError
from trakt_core.tenancy import authorise_portfolio_access

from . import decks as decks_mod
from .dependencies import CapabilityDependencies, build_dependencies

logger = logging.getLogger("mi_agent_api.artefacts")

CAPABILITY = "artefact.investor_pack.get"


@dataclass(frozen=True)
class ResolvedArtefact:
    """A located, authorised artefact."""

    artefact_type: str
    local_path: Path
    download_name: str
    content_type: str
    tenant_id: str
    portfolio_id: str
    period: Optional[str] = None
    generated_at: Optional[str] = None
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None

    def to_metadata(self) -> Dict[str, Any]:
        """Client-safe metadata. Deliberately excludes ``local_path`` — the
        server-side location is never part of a response."""
        return {
            "artefactType": self.artefact_type,
            "fileName": self.download_name,
            "contentType": self.content_type,
            "sizeBytes": self.size_bytes,
            "clientId": self.tenant_id,
            "portfolioId": self.portfolio_id,
            "reportingPeriod": self.period,
            "generatedAt": self.generated_at,
            "checksum": self.checksum,
        }


PPTX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation")

_BLOCKING_CATEGORIES = frozenset({
    ErrorCategory.AUTHENTICATION, ErrorCategory.AUTHORISATION, ErrorCategory.POLICY,
})


def _audit(context: ExecutionContext, *, outcome: str, portfolio_id: Optional[str],
           error_code: Optional[str]) -> AuditMetadata:
    return AuditMetadata(
        capability=CAPABILITY, request_id=context.request_id,
        correlation_id=context.correlation_id, tenant_id=context.tenant_id,
        actor_id=context.actor_id, actor_type=context.actor_type,
        channel=context.channel, portfolio_id=portfolio_id, outcome=outcome,
        started_at=datetime.now(timezone.utc).isoformat(), error_code=error_code)


def get_investor_pack(
    context: ExecutionContext,
    *,
    portfolio_id: Optional[str] = None,
    period: Optional[str] = None,
    requested_client_id: Optional[str] = None,
    dependencies: Optional[CapabilityDependencies] = None,
) -> GovernedResult[ResolvedArtefact]:
    """Locate the investor pack for the authenticated tenant.

    ``requested_client_id`` is the DEPRECATED caller-supplied identifier the
    React route still accepts. It is never used to select the artefact; it is
    only compared against the trusted tenant and rejected on conflict, so an
    existing client that still sends its own id keeps working while a caller
    that sends someone else's is refused.
    """
    deps = dependencies or build_dependencies()
    try:
        context.require_scope(SCOPE_ARTEFACT_READ)

        if requested_client_id and requested_client_id != context.tenant_id:
            raise TraktError(
                ErrorCode.TENANT_MISMATCH,
                "The requested artefact belongs to a different tenant.",
                request_id=context.request_id)

        authorised = authorise_portfolio_access(
            context, portfolio_id, registry=deps.tenant_registry)

        if period is not None and not _valid_period(period):
            raise TraktError(ErrorCode.INVALID_INPUT,
                             "The reporting period is not a valid period label.",
                             request_id=context.request_id)

        # The tenant — never the caller's parameter — selects the deck store.
        resolved = decks_mod.resolve_deck_local(context.tenant_id, period)
        if resolved is None:
            raise TraktError(
                ErrorCode.ARTEFACT_NOT_FOUND,
                f"No investor deck is available for {period or 'the latest period'}.",
                request_id=context.request_id)

        path, download_name = resolved
        listing = decks_mod.list_decks(context.tenant_id) or {}
        latest = listing.get("latest") or {}
        try:
            size = path.stat().st_size
        except OSError:
            size = None

        artefact = ResolvedArtefact(
            artefact_type="investor_deck", local_path=Path(path),
            download_name=download_name, content_type=PPTX_MEDIA_TYPE,
            tenant_id=context.tenant_id, portfolio_id=authorised.portfolio_id,
            period=latest.get("period") if period is None else period,
            generated_at=latest.get("generatedAt"), size_bytes=size,
            checksum=latest.get("checksum"))

        result: GovernedResult[ResolvedArtefact] = GovernedResult(
            capability=CAPABILITY, status=STATUS_SUCCESS,
            request_id=context.request_id, correlation_id=context.correlation_id,
            tenant_id=context.tenant_id, portfolio_id=authorised.portfolio_id,
            result=artefact,
            policy=PolicyState(runtime_mode=deps.runtime_mode, tenant_authorised=True,
                               capability_allowed=True, data_approved=True),
            audit=_audit(context, outcome=STATUS_SUCCESS,
                         portfolio_id=authorised.portfolio_id, error_code=None))
        emit_audit_event(result)
        return result

    except TraktError as err:
        status = STATUS_BLOCKED if err.category in _BLOCKING_CATEGORIES else STATUS_ERROR
        result = GovernedResult(
            capability=CAPABILITY, status=status, request_id=context.request_id,
            correlation_id=context.correlation_id, tenant_id=context.tenant_id,
            portfolio_id=portfolio_id, result=None, error=err,
            policy=PolicyState(runtime_mode=deps.runtime_mode),
            audit=_audit(context, outcome=status, portfolio_id=portfolio_id,
                         error_code=err.code))
        emit_audit_event(result)
        return result
    except Exception as exc:  # noqa: BLE001 - a storage fault is not a raw 500
        logger.exception("investor pack resolution failed for tenant=%s",
                         context.tenant_id)
        err = TraktError(ErrorCode.STORAGE_UNAVAILABLE,
                         "The investor-deck store is currently unavailable.",
                         request_id=context.request_id, cause=exc)
        result = GovernedResult(
            capability=CAPABILITY, status=STATUS_ERROR, request_id=context.request_id,
            correlation_id=context.correlation_id, tenant_id=context.tenant_id,
            portfolio_id=portfolio_id, result=None, error=err,
            policy=PolicyState(runtime_mode=deps.runtime_mode),
            audit=_audit(context, outcome=STATUS_ERROR, portfolio_id=portfolio_id,
                         error_code=err.code))
        emit_audit_event(result)
        return result


def _valid_period(period: str) -> bool:
    """A period label is a date or ``latest`` — never a path fragment."""
    import re
    return bool(re.fullmatch(r"latest|\d{4}-\d{2}(-\d{2})?", str(period).strip()))
