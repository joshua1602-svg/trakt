#!/usr/bin/env python3
"""Sprint 1, Deliverables 3 and 4: machine identity and the agent REST surface.

Three things are being proved:

* a **machine** identity is not a human one — ``actor_type=service``,
  ``channel=enterprise_agent``, scopes DERIVED from grants rather than defaulted,
  and the individual-principal registry deliberately not consulted;
* the development auth mode is a development mode — refused in production,
  and never anonymous even locally;
* the REST surface is a transport adapter — it publishes the registry's schemas,
  invokes through ``execute_governed_tool``, and exposes nothing else.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core.context import (
    ACTOR_SERVICE,
    ACTOR_USER,
    CHANNEL_ENTERPRISE_AGENT,
    DEFAULT_MI_SCOPES,
    SCOPE_ARTEFACT_GENERATE,
    SCOPE_MI_QUERY,
    SCOPE_RISK_READ,
)
from trakt_core.entitlement import EntitlementStore, Grant
from trakt_core.errors import ErrorCode, TraktError
from trakt_core.organisation import (
    ORG_TYPE_INVESTOR,
    OrganisationRecord,
    OrganisationRegistry,
)
from trakt_core.resource import KIND_SOURCE_PORTFOLIO, ResourceRef

from mi_agent_api import agent_auth, identity as identity_mod

TENANT = "ERE"
AGENT_ORG = "a2a_test_agent"
AGENT_DIR = "00000000-0000-4000-8000-00000000a2a7"
OTHER_DIR = "99999999-9999-4999-8999-999999999999"

PORTFOLIO_A = ResourceRef(TENANT, KIND_SOURCE_PORTFOLIO, "direct_001")
PORTFOLIO_B = ResourceRef(TENANT, KIND_SOURCE_PORTFOLIO, "acquired_001")


@dataclass
class _Principal:
    """Shaped exactly as ``agent_auth.AgentPrincipal``."""
    subject: str = "sp-object-id"
    name: str = "a2a-test-agent-app"
    directory_id: Optional[str] = AGENT_DIR
    roles: List[str] = field(default_factory=lambda: ["Trakt.Agent"])
    scopes: List[str] = field(default_factory=list)
    synthetic: bool = False


def _registry() -> OrganisationRegistry:
    return OrganisationRegistry([
        OrganisationRecord(organisation_id=AGENT_ORG,
                           display_name="A2A Test Agent",
                           organisation_type=ORG_TYPE_INVESTOR,
                           microsoft_tenant_ids=(AGENT_DIR,)),
    ], configured=True)


def _store(capabilities=(SCOPE_RISK_READ,), ref=PORTFOLIO_A) -> EntitlementStore:
    return EntitlementStore(
        [Grant(organisation_id=AGENT_ORG, resource_ref=ref,
               capabilities=frozenset(capabilities))],
        configured=True, validate=False)


_UNSET = object()


def _context(**kwargs):
    params = dict(tenant_id=TENANT, organisation_registry=_registry(),
                  entitlement_store=_store())
    params.update(kwargs)
    # A sentinel, not `or`: `principal=None` is a case under test in its own
    # right and must not be replaced by the default.
    principal = params.pop("principal", _UNSET)
    if principal is _UNSET:
        principal = _Principal()
    return identity_mod.context_from_agent_principal(principal, **params)


# --------------------------------------------------------------------------- #
# Scope derivation — a machine does not inherit a human's scope set
# --------------------------------------------------------------------------- #
def test_a_machine_identity_derives_its_scopes_from_its_grants():
    context = _context()
    assert context.scopes == frozenset({SCOPE_RISK_READ})
    # Emphatically NOT the human default set.
    assert context.scopes != DEFAULT_MI_SCOPES
    assert SCOPE_MI_QUERY not in context.scopes
    assert SCOPE_ARTEFACT_GENERATE not in context.scopes


def test_scopes_are_the_union_across_resources_and_narrowing_stays_per_resource():
    """The scope gate answers 'may this caller attempt this verb at all'. Which
    resource it may attempt it against is a separate, per-resource decision — so
    a union here does not widen anything."""
    store = EntitlementStore([
        Grant(organisation_id=AGENT_ORG, resource_ref=PORTFOLIO_A,
              capabilities=frozenset({SCOPE_RISK_READ})),
        Grant(organisation_id=AGENT_ORG, resource_ref=PORTFOLIO_B,
              capabilities=frozenset({SCOPE_MI_QUERY})),
    ], configured=True, validate=False)
    context = _context(entitlement_store=store)

    assert context.scopes == frozenset({SCOPE_RISK_READ, SCOPE_MI_QUERY})
    # The narrowing is still per resource.
    assert context.capabilities_for(PORTFOLIO_A) == frozenset({SCOPE_RISK_READ})
    assert context.capabilities_for(PORTFOLIO_B) == frozenset({SCOPE_MI_QUERY})


def test_an_agent_with_no_grants_holds_no_scopes_at_all():
    """Empty is a real answer, never a fallback to the default set."""
    empty = EntitlementStore([], configured=True, validate=False)
    context = _context(entitlement_store=empty)
    assert context.scopes == frozenset()
    assert context.entitlements is not None
    assert context.entitlements.is_empty()


def test_scopes_from_entitlements_returns_nothing_for_no_entitlements():
    assert identity_mod.scopes_from_entitlements(None) == frozenset()


# --------------------------------------------------------------------------- #
# Machine identity shape
# --------------------------------------------------------------------------- #
def test_a_machine_is_recorded_as_a_service_on_the_agent_channel():
    context = _context()
    assert context.actor_type == ACTOR_SERVICE
    assert context.actor_type != ACTOR_USER
    assert context.channel == CHANNEL_ENTERPRISE_AGENT
    assert context.organisation_id == AGENT_ORG
    assert context.microsoft_tenant_id == AGENT_DIR


def test_the_tenant_is_deployment_configuration_never_the_agents_directory():
    context = _context()
    assert context.tenant_id == TENANT
    assert context.tenant_id != context.microsoft_tenant_id
    assert context.tenant_id != context.organisation_id


def test_an_unregistered_directory_is_refused_rather_than_served_unattributed():
    """Compatibility mode is a legitimate posture for the human channels, which
    authorise through the tenant/portfolio path. It is not one here: every tool
    call authorises through entitlements, and entitlements hang off an
    organisation."""
    with pytest.raises(TraktError) as excinfo:
        _context(principal=_Principal(directory_id=OTHER_DIR))
    assert excinfo.value.code == ErrorCode.ORGANISATION_NOT_REGISTERED

    with pytest.raises(TraktError) as excinfo:
        _context(principal=_Principal(directory_id=None))
    assert excinfo.value.code == ErrorCode.ORGANISATION_NOT_REGISTERED


def test_a_deployment_with_no_organisation_config_still_refuses_an_agent():
    """The human channels fall back to compatibility mode when no organisation
    registry exists. The agent surface must not: an unattributed machine has no
    entitlements, so there would be nothing to check."""
    with pytest.raises(TraktError) as excinfo:
        _context(organisation_registry=OrganisationRegistry([], configured=False))
    assert excinfo.value.code == ErrorCode.ORGANISATION_NOT_REGISTERED


def test_no_principal_is_required_because_a_service_principal_is_not_a_person():
    """``context_from_agent_principal`` must not consult the individual registry:
    a service principal's oid is an APPLICATION. Consulting it would either
    refuse every agent on a principal-gated directory or register machines as
    staff."""
    import inspect
    source = inspect.getsource(identity_mod.context_from_agent_principal)
    assert "resolve_principal" not in source
    assert "principal_registry" not in inspect.signature(
        identity_mod.context_from_agent_principal).parameters


def test_a_missing_principal_is_an_authentication_failure():
    with pytest.raises(TraktError) as excinfo:
        _context(principal=None)
    assert excinfo.value.code == ErrorCode.AUTHENTICATION_REQUIRED


# --------------------------------------------------------------------------- #
# The auth guard
# --------------------------------------------------------------------------- #
def test_agent_auth_reuses_the_copilot_directory_allow_list():
    """The subtle part of accepting several directories is the allow-list and the
    key-selection step, and it must not exist twice. This pins the names so a
    rename breaks loudly instead of silently forking the allow-list."""
    from mi_agent_api import copilot_auth

    for name in ("allowed_directories", "_select_directory", "_jwks_client",
                 "_allowed_issuers", "_jwt", "_PyJWKClient"):
        assert hasattr(copilot_auth, name), name
    assert agent_auth._copilot_auth is copilot_auth


def test_the_default_required_role_is_not_empty():
    """An agent surface should need a deliberate app-role assignment, not admit
    any valid token from an accepted directory."""
    assert agent_auth.DEFAULT_REQUIRED_ROLE
    assert agent_auth._required_role() == agent_auth.DEFAULT_REQUIRED_ROLE


def test_an_operator_can_clear_the_required_role_but_only_explicitly(monkeypatch):
    monkeypatch.setenv("TRAKT_AGENT_REQUIRED_ROLE", "")
    assert agent_auth._required_role() == ""
    monkeypatch.setenv("TRAKT_AGENT_REQUIRED_ROLE", "Custom.Role")
    assert agent_auth._required_role() == "Custom.Role"
    monkeypatch.delenv("TRAKT_AGENT_REQUIRED_ROLE")
    assert agent_auth._required_role() == agent_auth.DEFAULT_REQUIRED_ROLE


def test_disabled_auth_mode_is_refused_in_production(monkeypatch):
    import asyncio

    from fastapi import HTTPException

    monkeypatch.setenv("TRAKT_AGENT_AUTH_MODE", "disabled")
    monkeypatch.setenv("TRAKT_AGENT_DEV_DIRECTORY", AGENT_DIR)
    monkeypatch.setattr(agent_auth, "is_production", lambda: True)

    class _Req:
        headers: Dict[str, str] = {}
        state = type("S", (), {})()

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(agent_auth.agent_auth_guard(_Req()))
    assert excinfo.value.status_code == 503


def test_disabled_auth_mode_is_never_anonymous(monkeypatch):
    """Without a directory no organisation resolves, so no entitlements resolve,
    so every call would be refused anyway. Failing here says why."""
    import asyncio

    from fastapi import HTTPException

    monkeypatch.setenv("TRAKT_AGENT_AUTH_MODE", "disabled")
    monkeypatch.delenv("TRAKT_AGENT_DEV_DIRECTORY", raising=False)
    monkeypatch.setattr(agent_auth, "is_production", lambda: False)

    class _Req:
        headers: Dict[str, str] = {}
        state = type("S", (), {})()

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(agent_auth.agent_auth_guard(_Req()))
    assert excinfo.value.status_code == 503
    assert "TRAKT_AGENT_DEV_DIRECTORY" in excinfo.value.detail


def test_an_unknown_auth_mode_fails_closed(monkeypatch):
    import asyncio

    from fastapi import HTTPException

    monkeypatch.setenv("TRAKT_AGENT_AUTH_MODE", "off")

    class _Req:
        headers: Dict[str, str] = {}
        state = type("S", (), {})()

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(agent_auth.agent_auth_guard(_Req()))
    assert excinfo.value.status_code == 503


def test_the_agent_api_is_off_unless_explicitly_enabled(monkeypatch):
    from mi_agent_api import agent_api

    monkeypatch.delenv("TRAKT_AGENT_API_ENABLED", raising=False)
    assert agent_api.enabled() is False
    monkeypatch.setenv("TRAKT_AGENT_API_ENABLED", "true")
    assert agent_api.enabled() is True


# --------------------------------------------------------------------------- #
# The REST surface
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client(monkeypatch):
    """A minimal app carrying ONLY the agent router.

    Deliberately not ``mi_agent_api.app``: the point of this router is that it is
    a separate, closed surface, and mounting it alone is how that is tested.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mi_agent_api import agent_api

    monkeypatch.setenv("TRAKT_AGENT_AUTH_MODE", "disabled")
    monkeypatch.setenv("TRAKT_AGENT_DEV_DIRECTORY", AGENT_DIR)
    monkeypatch.setattr(agent_auth, "is_production", lambda: False)
    monkeypatch.setattr(agent_api, "default_tenant_id", lambda: TENANT)

    real_context = identity_mod.context_from_agent_principal

    def _patched(principal, **kwargs):
        kwargs.setdefault("organisation_registry", _registry())
        kwargs.setdefault("entitlement_store", _store())
        return real_context(principal, **kwargs)

    monkeypatch.setattr(agent_api.identity_mod, "context_from_agent_principal",
                        _patched)

    # Route every tool execution through the real governed path against the
    # fixture catalogue and a stub evaluator, so the ROUTE is what is under test.
    from tests.test_agent_governed_execution import _deps as _tool_deps

    real_execute = agent_api.trakt_tools.execute_governed_tool

    def _execute(tool_name, arguments, context, **kwargs):
        return real_execute(tool_name, arguments, context,
                            dependencies=_tool_deps())

    monkeypatch.setattr(agent_api.trakt_tools, "execute_governed_tool", _execute)

    app = FastAPI()
    app.include_router(agent_api.router)
    return TestClient(app)


def test_get_tools_publishes_the_schemas_and_the_permitted_resources(client):
    response = client.get("/v1/agent/tools")
    assert response.status_code == 200
    payload = response.json()

    names = {t["name"] for t in payload["tools"]}
    assert "evaluate_covenants" in names
    # Narrowed to this principal's single capability: nothing needing loan:read
    # is offered to a caller that cannot use it.
    assert {t["required_capability"] for t in payload["tools"]} == {SCOPE_RISK_READ}
    assert not (names & {"get_loans", "explain_values", "rank_loans"})

    tool = next(t for t in payload["tools"] if t["name"] == "evaluate_covenants")
    assert tool["required_capability"] == SCOPE_RISK_READ
    assert tool["version"]
    assert tool["input_schema"]["required"] == ["resource"]
    assert "resource" in tool["input_schema"]["properties"]

    assert payload["resources"][SCOPE_RISK_READ] == [PORTFOLIO_A.key]
    assert payload["organisation_id"] == AGENT_ORG
    assert payload["request_id"]


def test_invoking_a_tool_returns_the_governed_envelope(client):
    response = client.post("/v1/agent/tools/evaluate_covenants",
                           json={"resource": PORTFOLIO_A.key})
    assert response.status_code == 200
    payload = response.json()

    assert payload["capability"] == "tool.evaluate_covenants"
    assert payload["status"] == "success"
    assert payload["tenant_id"] == TENANT
    assert payload["schema_version"]
    assert payload["snapshot"]["content_hash"] == "sha256:abc123"
    assert payload["policy"]["data_approved"] is True
    assert payload["audit"]["organisation_id"] == AGENT_ORG
    assert payload["result"]["summary"]["overall_status"] == "breach"


def test_an_unauthorised_resource_is_refused_over_http_with_a_typed_code(client):
    response = client.post("/v1/agent/tools/evaluate_covenants",
                           json={"resource": PORTFOLIO_B.key})
    assert response.status_code == 403
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["error"]["code"] == ErrorCode.RESOURCE_NOT_AUTHORISED
    assert payload["error"]["retryable"] is False
    assert payload["result"] is None


def test_an_unknown_tool_is_a_404_with_a_typed_code(client):
    # A name no tool will plausibly take. This test previously used `get_loan`,
    # which has since been registered — a reminder that a placeholder name is a
    # liability once the surface grows.
    response = client.post("/v1/agent/tools/no_such_tool_exists",
                           json={"resource": PORTFOLIO_A.key})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == ErrorCode.TOOL_NOT_FOUND


def test_invalid_arguments_are_a_400_with_the_failing_paths(client):
    response = client.post("/v1/agent/tools/evaluate_covenants", json={})
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == ErrorCode.TOOL_INPUT_INVALID
    assert any("resource" in v for v in payload["error"]["details"]["violations"])


def test_the_agent_router_exposes_only_the_tool_surface(client):
    """No MI query route, no snapshot route, no deck route — this is a closed
    surface, not a window onto the internal API."""
    from mi_agent_api import agent_api

    paths = {route.path for route in agent_api.router.routes}
    assert paths == {"/v1/agent/tools", "/v1/agent/tools/{tool_name}"}

    assert client.get("/mi/query").status_code == 404
    assert client.get("/v1/agent/mi/query").status_code == 404


def test_the_agent_surface_does_not_route_through_the_language_interpreter():
    """Two probabilistic hops in one call — the agent choosing words and Trakt
    guessing what they meant — is exactly what a typed tool contract removes."""
    from mi_agent_api import agent_api

    source = Path(agent_api.__file__).read_text(encoding="utf-8")
    assert "execute_governed_mi_query" not in source
    assert "mi_service" not in source


def test_the_easy_auth_guard_exempts_the_agent_prefix():
    """A regression test for a real defect this suite initially missed.

    ``auth.auth_guard`` is a GLOBAL dependency on the app: it demands an Easy Auth
    ``X-MS-CLIENT-PRINCIPAL`` header on every path it does not exempt. A machine
    identity carries no such header, so without the exemption every agent call
    was refused 401 by the wrong guard — before ``agent_auth`` ran at all.

    Testing the router in isolation could not catch this, because the global
    dependency lives on the app rather than on the router. Hence this test
    asserts the exemption itself.
    """
    import asyncio

    from mi_agent_api import auth

    assert auth.AGENT_PATH_PREFIX == "/v1/agent"

    class _Req:
        def __init__(self, path):
            self.url = type("U", (), {"path": path})()
            self.method = "POST"
            self.headers: Dict[str, str] = {}
            self.state = type("S", (), {})()

    # Exempt: the agent's own guard authenticates these.
    for path in ("/v1/agent/tools", "/v1/agent/tools/evaluate_covenants"):
        assert asyncio.run(auth.auth_guard(_Req(path))) is None

    # Still guarded: the exemption must not have widened to anything else.
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        asyncio.run(auth.auth_guard(_Req("/mi/query")))


def test_an_inbound_correlation_id_is_carried_into_the_envelope(client):
    response = client.post("/v1/agent/tools/evaluate_covenants",
                           json={"resource": PORTFOLIO_A.key},
                           headers={"X-Correlation-Id": "buyer-run-7"})
    assert response.json()["correlation_id"] == "buyer-run-7"
