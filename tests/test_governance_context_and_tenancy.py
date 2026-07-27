#!/usr/bin/env python3
"""Execution context, tenant identity and portfolio authorisation.

Covers required cases 6-16: identity mapping into ExecutionContext for both
channels, direct invocation without FastAPI, fail-closed on missing context,
request-id propagation, and the tenant/portfolio separation that stops a
caller-supplied identifier redefining the tenant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core.context import (
    ACTOR_SERVICE,
    ACTOR_SYSTEM,
    ACTOR_USER,
    CHANNEL_COPILOT,
    CHANNEL_ENTERPRISE_AGENT,
    CHANNEL_REACT,
    SCOPE_MI_QUERY,
    SCOPE_PORTFOLIO_READ,
    ExecutionContext,
)
from trakt_core.errors import ErrorCode, TraktError
from trakt_core.tenancy import (
    TenantRecord,
    TenantRegistry,
    authorise_portfolio_access,
    load_tenant_registry,
    split_portfolio_id,
)


# --------------------------------------------------------------------------- #
# ExecutionContext
# --------------------------------------------------------------------------- #
def test_context_requires_a_tenant_and_an_actor():
    with pytest.raises(ValueError):
        ExecutionContext(tenant_id="", actor_id="a")
    with pytest.raises(ValueError):
        ExecutionContext(tenant_id="ERE", actor_id="")


def test_request_id_is_generated_when_not_supplied():
    """Required case 10 — every governed execution is identifiable."""
    ctx = ExecutionContext(tenant_id="ERE", actor_id="user-1")
    assert ctx.request_id.startswith("req_")
    supplied = ExecutionContext(tenant_id="ERE", actor_id="user-1",
                                request_id="req_caller_supplied")
    assert supplied.request_id == "req_caller_supplied"


def test_context_is_immutable():
    ctx = ExecutionContext(tenant_id="ERE", actor_id="user-1")
    with pytest.raises(Exception):
        ctx.tenant_id = "OTHER"  # type: ignore[misc]


def test_missing_scope_is_a_typed_refusal():
    ctx = ExecutionContext(tenant_id="ERE", actor_id="user-1", scopes=frozenset())
    with pytest.raises(TraktError) as exc:
        ctx.require_scope(SCOPE_MI_QUERY)
    assert exc.value.code == ErrorCode.SCOPE_MISSING
    assert exc.value.retryable is False


def test_audit_projection_excludes_the_actor_label():
    """The label may be an email address; actor_id is the minimal identifier."""
    ctx = ExecutionContext(tenant_id="ERE", actor_id="oid-123",
                           actor_label="someone@example.com")
    assert "someone@example.com" not in str(ctx.to_audit_dict())
    assert ctx.to_audit_dict()["actor_id"] == "oid-123"


# --------------------------------------------------------------------------- #
# Identity adapters (required cases 6, 7, 9)
# --------------------------------------------------------------------------- #
def test_easy_auth_principal_maps_into_a_context(monkeypatch):
    monkeypatch.delenv("WEBSITE_SITE_NAME", raising=False)
    monkeypatch.delenv("WEBSITE_INSTANCE_ID", raising=False)
    from mi_agent_api import identity
    from mi_agent_api.auth import Principal

    ctx = identity.context_from_principal(
        Principal(user_id="oid-1", user_details="a@b.com", roles={"client"}),
        tenant_id="ERE", channel=CHANNEL_REACT, request_id="req_x")
    assert (ctx.tenant_id, ctx.actor_id, ctx.channel) == ("ERE", "oid-1", CHANNEL_REACT)
    assert ctx.actor_type == ACTOR_USER
    assert ctx.request_id == "req_x"
    assert ctx.has_scope(SCOPE_MI_QUERY)


def test_copilot_principal_maps_into_a_context():
    from mi_agent_api import identity
    from mi_agent_api.copilot_auth import CopilotPrincipal

    delegated = identity.context_from_copilot_principal(
        CopilotPrincipal(subject="oid-9", name="agent", scopes=["Trakt.Copilot"]),
        tenant_id="ERE")
    assert delegated.channel == CHANNEL_COPILOT
    assert delegated.actor_type == ACTOR_USER

    app_only = identity.context_from_copilot_principal(
        CopilotPrincipal(subject="sp-1", name="svc", roles=["Trakt.Copilot"]),
        tenant_id="ERE")
    assert app_only.actor_type == ACTOR_SERVICE


def test_missing_principal_fails_closed():
    """Required case 9 — no identity means no answer, not a default identity."""
    from mi_agent_api import identity
    with pytest.raises(TraktError) as exc:
        identity.context_from_copilot_principal(None, tenant_id="ERE")
    assert exc.value.code == ErrorCode.AUTHENTICATION_REQUIRED


def test_tenant_is_never_taken_from_a_principal_claim():
    """Deployment-per-tenant: the tenant is configuration, so a principal from
    another directory cannot move itself into another tenant."""
    from mi_agent_api import identity
    from mi_agent_api.copilot_auth import CopilotPrincipal

    ctx = identity.context_from_copilot_principal(
        CopilotPrincipal(subject="oid-9", tenant_id="some-other-entra-tenant"),
        tenant_id="ERE")
    assert ctx.tenant_id == "ERE"


# --------------------------------------------------------------------------- #
# Portfolio authorisation (required cases 11-16)
# --------------------------------------------------------------------------- #
def _registry(**tenants) -> TenantRegistry:
    return TenantRegistry(tenants, configured=True)


ERE_RECORD = TenantRecord(
    tenant_id="ERE", default_portfolio_id="ERE",
    portfolio_ids=frozenset({"ERE", "direct_001", "acquired_001"}))
OTHER_RECORD = TenantRecord(tenant_id="OTHERCO", default_portfolio_id="OTHERCO",
                            portfolio_ids=frozenset({"OTHERCO", "otherco_book"}))


def test_split_portfolio_id_forms():
    assert split_portfolio_id(None) == (None, None)
    assert split_portfolio_id("ERE") == ("ERE", None)
    assert split_portfolio_id("ERE/2026-01-31") == ("ERE", "2026-01-31")
    assert split_portfolio_id("ERE/") == ("ERE", None)


def test_tenant_can_access_its_own_portfolio():
    """Required case 11."""
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u")
    auth = authorise_portfolio_access(ctx, "direct_001",
                                      registry=_registry(ERE=ERE_RECORD))
    assert auth.tenant_id == "ERE"
    assert auth.selector == "direct_001"
    assert auth.portfolio_id == "direct_001"
    assert auth.explicit is True


def test_tenant_cannot_access_another_tenants_portfolio():
    """Required case 12 — the cross-tenant case, refused by name."""
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u")
    with pytest.raises(TraktError) as exc:
        authorise_portfolio_access(ctx, "OTHERCO",
                                   registry=_registry(ERE=ERE_RECORD,
                                                      OTHERCO=OTHER_RECORD))
    assert exc.value.code == ErrorCode.TENANT_MISMATCH


def test_unlisted_portfolio_is_refused_when_a_registry_is_configured():
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u")
    with pytest.raises(TraktError) as exc:
        authorise_portfolio_access(ctx, "someone_elses_book",
                                   registry=_registry(ERE=ERE_RECORD))
    assert exc.value.code == ErrorCode.PORTFOLIO_NOT_AUTHORISED


def test_unknown_tenant_is_refused_when_a_registry_is_configured():
    ctx = ExecutionContext(tenant_id="GHOST", actor_id="u")
    with pytest.raises(TraktError) as exc:
        authorise_portfolio_access(ctx, None, registry=_registry(ERE=ERE_RECORD))
    assert exc.value.code == ErrorCode.PERMISSION_DENIED


def test_default_portfolio_resolves_when_none_is_named():
    """Required case 15 — and the requested id stays None so dataset resolution
    keeps its documented 'active dataset' behaviour."""
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u")
    auth = authorise_portfolio_access(ctx, None, registry=_registry(ERE=ERE_RECORD))
    assert auth.selector == "ERE"
    assert auth.requested_portfolio_id is None
    assert auth.explicit is False


def test_single_tenant_deployment_keeps_its_own_namespace():
    """No tenancy config: the tenant owns any well-formed selector, but the
    tenant itself still comes from the trusted context."""
    registry = load_tenant_registry(config_path="/nonexistent.yaml",
                                    default_tenant_id="ERE")
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u")
    auth = authorise_portfolio_access(ctx, "direct_001", registry=registry)
    assert auth.tenant_id == "ERE" and auth.selector == "direct_001"


def test_portfolio_selector_cannot_traverse_a_storage_path():
    """A selector is a slug. This is what stops '../' reaching another prefix in
    the shared processed container."""
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u")
    registry = load_tenant_registry(config_path="/nonexistent.yaml",
                                    default_tenant_id="ERE")
    for bad in ("../../raw/other", "a/../../b", "..", "a b", "x\ny"):
        with pytest.raises(TraktError) as exc:
            authorise_portfolio_access(ctx, bad, registry=registry)
        assert exc.value.code in (ErrorCode.INVALID_INPUT,
                                  ErrorCode.PORTFOLIO_NOT_AUTHORISED), bad

    # An empty string is "no portfolio named", not an attack: it resolves to the
    # tenant's default exactly as ``None`` does.
    assert authorise_portfolio_access(ctx, "", registry=registry).selector == "ERE"


def test_run_id_is_validated_too():
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u")
    registry = load_tenant_registry(config_path="/nonexistent.yaml",
                                    default_tenant_id="ERE")
    with pytest.raises(TraktError) as exc:
        authorise_portfolio_access(ctx, "ERE/../../etc", registry=registry)
    assert exc.value.code == ErrorCode.INVALID_INPUT


def test_authorisation_requires_the_portfolio_read_scope():
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u", scopes=frozenset())
    with pytest.raises(TraktError) as exc:
        authorise_portfolio_access(ctx, "ERE", registry=_registry(ERE=ERE_RECORD))
    assert exc.value.code == ErrorCode.SCOPE_MISSING


def test_a_broken_tenancy_config_falls_back_rather_than_breaking_the_api(tmp_path):
    bad = tmp_path / "tenancy.yaml"
    bad.write_text("tenants: [this is not a mapping\n")
    registry = load_tenant_registry(config_path=bad, default_tenant_id="ERE")
    assert registry.configured is False
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u")
    assert authorise_portfolio_access(ctx, "ERE", registry=registry).tenant_id == "ERE"


# --------------------------------------------------------------------------- #
# Direct invocation (required case 8) — see also
# tests/test_governance_dependency_direction.py
# --------------------------------------------------------------------------- #
def test_internal_context_can_be_built_without_fastapi():
    ctx = ExecutionContext.for_internal("ERE")
    assert ctx.actor_type == ACTOR_SYSTEM
    assert ctx.has_scope(SCOPE_PORTFOLIO_READ)

    agent = ExecutionContext(
        tenant_id="ERE", actor_id="client-agent-123", actor_type=ACTOR_SERVICE,
        channel=CHANNEL_ENTERPRISE_AGENT,
        scopes=frozenset({SCOPE_PORTFOLIO_READ, SCOPE_MI_QUERY}),
        request_id="req_abc", correlation_id="corr-1")
    assert agent.channel == CHANNEL_ENTERPRISE_AGENT
    assert agent.correlation_id == "corr-1"
