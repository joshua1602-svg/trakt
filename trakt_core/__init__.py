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
  * :mod:`trakt_core.organisation` — the *external organisation* registry: which
    Microsoft Entra directory belongs to which organisation. Answers "who is
    asking", which is not the same question as "whose data is this".
  * :mod:`trakt_core.principal` — which individual is asking, bound to an
    organisation by stable Microsoft identity. An independent kill switch;
    entitlements stay at the organisation level.
  * :mod:`trakt_core.resource`  — the permissionable resource model: what a
    portfolio / SPV / facility / population *is*, and which existing data
    population it deterministically means. Answers "what is the thing", and
    deliberately not "who may reach it".
  * :mod:`trakt_core.entitlement` — organisation × resource × capability, and
    :func:`~trakt_core.entitlement.authorise_resource_access`: the one place that
    decides whether an organisation may perform a capability against a resource.
    Server-side configuration only; dormant until configured.
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
    SCOPE_ARTEFACT_GENERATE,
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
from .organisation import (
    KNOWN_ORGANISATION_TYPES,
    ORG_TYPE_INVESTOR,
    ORG_TYPE_OPERATOR,
    ORG_TYPE_ORIGINATOR,
    ORG_TYPE_SERVICER,
    ORG_TYPE_UNKNOWN,
    ORG_TYPE_WAREHOUSE_FUNDER,
    OrganisationRecord,
    OrganisationRegistry,
    load_organisation_registry,
    normalise_directory_id,
)
from .entitlement import (
    KNOWN_GRANT_STATUSES,
    STATUS_ACTIVE,
    STATUS_PROPOSED,
    STATUS_REVOKED,
    AuthorisedResource,
    EntitlementStore,
    Grant,
    ResolvedEntitlements,
    authorise_resource_access,
    load_entitlement_store,
    permitted_resources_for,
    resolve_entitlements,
)
from .principal import (
    KNOWN_PRINCIPAL_STATUSES,
    PRINCIPAL_ACTIVE,
    PRINCIPAL_DISABLED,
    PrincipalBinding,
    PrincipalRegistry,
    load_principal_registry,
    normalise_object_id,
)
from .resource import (
    ATTRIBUTION_PROVENANCE,
    ATTRIBUTION_SPV_FIELD,
    ATTRIBUTION_UNPARTITIONABLE,
    ATTRIBUTION_VIEW,
    ATTRIBUTION_WHOLE_BOOK,
    KIND_FACILITY,
    KIND_POPULATION,
    KIND_PORTFOLIO,
    KIND_SOURCE_PORTFOLIO,
    KIND_SPV,
    POPULATION_FORECAST,
    POPULATION_FUNDED,
    POPULATION_PIPELINE,
    POPULATIONS,
    RESOURCE_KINDS,
    AttributionCheck,
    ResolvedResource,
    ResourceCatalogue,
    ResourceRecord,
    ResourceRef,
    check_attribution,
    load_resource_catalogue,
    resources_from_portfolio_records,
)
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
    "SCOPE_ARTEFACT_GENERATE",
    # tenancy
    "TenantRegistry", "TenantRecord", "AuthorisedPortfolio",
    "authorise_portfolio_access", "load_tenant_registry",
    # organisation (who is asking — not the data-owning tenant)
    "OrganisationRegistry", "OrganisationRecord", "load_organisation_registry",
    "normalise_directory_id", "KNOWN_ORGANISATION_TYPES",
    "ORG_TYPE_ORIGINATOR", "ORG_TYPE_WAREHOUSE_FUNDER", "ORG_TYPE_INVESTOR",
    "ORG_TYPE_SERVICER", "ORG_TYPE_OPERATOR", "ORG_TYPE_UNKNOWN",
    # principals (which individual, and which organisation they belong to)
    "PrincipalBinding", "PrincipalRegistry", "load_principal_registry",
    "normalise_object_id", "PRINCIPAL_ACTIVE", "PRINCIPAL_DISABLED",
    "KNOWN_PRINCIPAL_STATUSES",
    # resource model (what a permissionable thing is — not who may reach it)
    "ResourceRef", "ResourceRecord", "ResolvedResource", "ResourceCatalogue",
    "AttributionCheck", "check_attribution", "load_resource_catalogue",
    "resources_from_portfolio_records",
    "RESOURCE_KINDS", "KIND_PORTFOLIO", "KIND_SOURCE_PORTFOLIO", "KIND_SPV",
    "KIND_FACILITY", "KIND_POPULATION",
    "POPULATIONS", "POPULATION_FUNDED", "POPULATION_PIPELINE", "POPULATION_FORECAST",
    "ATTRIBUTION_PROVENANCE", "ATTRIBUTION_SPV_FIELD", "ATTRIBUTION_VIEW",
    "ATTRIBUTION_WHOLE_BOOK", "ATTRIBUTION_UNPARTITIONABLE",
    # entitlements (organisation × resource × capability)
    "Grant", "EntitlementStore", "ResolvedEntitlements", "AuthorisedResource",
    "authorise_resource_access", "permitted_resources_for",
    "load_entitlement_store", "resolve_entitlements",
    "STATUS_ACTIVE", "STATUS_PROPOSED", "STATUS_REVOKED", "KNOWN_GRANT_STATUSES",
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
