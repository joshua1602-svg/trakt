"""trakt_core — the interface-neutral governance core.

Everything in this package is importable **without FastAPI, Starlette, Azure or
any HTTP machinery**, so a governed capability can be invoked from a web route,
a Copilot action, a scheduled job, an MCP tool or plain Python without dragging
a web framework into the process. ``tests/test_governance_dependency_direction.py``
enforces that.

Contents:

  * :mod:`trakt_core.context`   — :class:`ExecutionContext`, the *trusted* caller
    identity (tenant, actor, channel, scopes, request id). Never built from
    request-body fields.
  * :mod:`trakt_core.tenancy`   — the tenant registry and
    :func:`~trakt_core.tenancy.authorise_portfolio_access`: the one place that
    decides whether a context may read a portfolio.
  * :mod:`trakt_core.runtime`   — the runtime mode (production / development /
    test) and its fail-closed startup validation.
  * :mod:`trakt_core.policy`    — the production data-source approval rule.
  * :mod:`trakt_core.errors`    — the machine-readable error taxonomy.
  * :mod:`trakt_core.envelope`  — :class:`GovernedResult`, the stable result
    envelope every governed capability returns.
  * :mod:`trakt_core.audit`     — the compact structured audit event.

The package holds contracts and policy only. Domain calculations stay in
``mi_agent`` / ``analytics_lib`` / ``engine``; dataset resolution stays in
``mi_agent_api.datasets``.
"""

from __future__ import annotations

from .audit import audit_event_from_result, emit_audit_event
from .context import (
    ACTOR_SERVICE,
    ACTOR_SYSTEM,
    ACTOR_USER,
    CHANNEL_AGENT_TO_AGENT,
    CHANNEL_COPILOT,
    CHANNEL_ENTERPRISE_AGENT,
    CHANNEL_INTERNAL,
    CHANNEL_REACT,
    SCOPE_ARTEFACT_READ,
    SCOPE_MI_QUERY,
    SCOPE_PORTFOLIO_READ,
    ExecutionContext,
    new_request_id,
)
from .envelope import (
    STATUS_BLOCKED,
    STATUS_ERROR,
    STATUS_PARTIAL_SUCCESS,
    STATUS_SUCCESS,
    AuditMetadata,
    GovernedResult,
    PolicyState,
    ProvenanceRef,
    SnapshotRef,
)
from .errors import ErrorCategory, ErrorCode, TraktError, http_status_for
from .policy import (
    PRODUCTION_APPROVED_SOURCE_BASES,
    SourceApproval,
    evaluate_source_approval,
)
from .runtime import (
    MODE_DEVELOPMENT,
    MODE_PRODUCTION,
    MODE_TEST,
    is_production,
    runtime_mode,
    validate_runtime_mode,
)
from .tenancy import (
    AuthorisedPortfolio,
    TenantRecord,
    TenantRegistry,
    authorise_portfolio_access,
    load_tenant_registry,
)

__all__ = [
    # context
    "ExecutionContext", "new_request_id",
    "ACTOR_USER", "ACTOR_SERVICE", "ACTOR_SYSTEM",
    "CHANNEL_REACT", "CHANNEL_COPILOT", "CHANNEL_INTERNAL",
    "CHANNEL_ENTERPRISE_AGENT", "CHANNEL_AGENT_TO_AGENT",
    "SCOPE_PORTFOLIO_READ", "SCOPE_MI_QUERY", "SCOPE_ARTEFACT_READ",
    # tenancy
    "TenantRegistry", "TenantRecord", "AuthorisedPortfolio",
    "authorise_portfolio_access", "load_tenant_registry",
    # runtime + policy
    "runtime_mode", "validate_runtime_mode", "is_production",
    "MODE_PRODUCTION", "MODE_DEVELOPMENT", "MODE_TEST",
    "evaluate_source_approval", "SourceApproval",
    "PRODUCTION_APPROVED_SOURCE_BASES",
    # errors
    "TraktError", "ErrorCode", "ErrorCategory", "http_status_for",
    # envelope
    "GovernedResult", "SnapshotRef", "PolicyState", "ProvenanceRef",
    "AuditMetadata",
    "STATUS_SUCCESS", "STATUS_PARTIAL_SUCCESS", "STATUS_BLOCKED", "STATUS_ERROR",
    # audit
    "emit_audit_event", "audit_event_from_result",
]
