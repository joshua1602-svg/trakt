#!/usr/bin/env python3
"""The governed identity→access mapping.

Authorisation used to be Static Web Apps' job: a user was invited, accepted, and
the accepted invitation attached ``client``/``operator`` to the forwarded
identity. In this deployment acceptance never produced a role record —
``az staticwebapp users list`` stayed empty — so every signed-in user arrived
with no MI role and was refused by a 403 the UI could not explain.

These tests pin the replacement. The platform still authenticates; Trakt now
decides what that verified identity may see, from configuration that is
reviewable in the repository.

The single most important test in this file is
``test_a_user_with_no_swa_role_at_all_is_admitted_when_provisioned``: it is the
proof that the invitation mechanism is no longer on the critical path.
"""

from __future__ import annotations

import base64
import json
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
from fastapi.testclient import TestClient

from mi_agent_api import auth as auth_mod
from mi_agent_api.app import app
from trakt_core import access
from trakt_core.errors import ErrorCode, TraktError

client = TestClient(app)

TENANT = "test-tenant"

DIRECTORY = textwrap.dedent("""
    version: 1
    principals:
      - identities: [ops@trakt.io, 11111111-2222-3333-4444-555555555555]
        tenant_id: test-tenant
        role: operator
        display_name: Trakt Operations
      - identities: [analyst@client.example]
        tenant_id: test-tenant
        role: client
      - identities: [narrow@client.example]
        tenant_id: test-tenant
        role: client
        portfolio_contexts: [direct]
      - identities: [revoked@client.example]
        tenant_id: test-tenant
        role: client
        enabled: false
      - identities: [outsider@other.example]
        tenant_id: SOMEONE-ELSE
        role: operator
""")


def _principal(user: Optional[str], roles: Optional[List[str]] = None,
               user_id: str = "oid-default") -> str:
    """A SWA-shaped principal. ``roles`` defaults to what SWA really sends for a
    signed-in user with no invitation: ``["anonymous", "authenticated"]``."""
    return base64.b64encode(json.dumps({
        "identityProvider": "aad",
        "userId": user_id,
        "userDetails": user,
        "userRoles": roles if roles is not None else ["anonymous", "authenticated"],
    }).encode("utf-8")).decode("ascii")


def _hdr(user: Optional[str], roles: Optional[List[str]] = None, **kw) -> dict:
    return {"x-ms-client-principal": _principal(user, roles, **kw)}


@pytest.fixture()
def provisioned(monkeypatch, tmp_path):
    path = tmp_path / "access.yaml"
    path.write_text(DIRECTORY, encoding="utf-8")
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "true")
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", TENANT)
    monkeypatch.setenv("TRAKT_ACCESS_CONFIG", str(path))
    access.reset_cache()
    yield path
    access.reset_cache()


# =========================================================================== #
# 1. The required access matrix
# =========================================================================== #
class TestAccessMatrix:

    def test_a_mapped_operator_is_admitted_as_an_operator(self, provisioned):
        r = client.get("/me", headers=_hdr("ops@trakt.io"))
        assert r.status_code == 200
        body = r.json()
        assert body["role"] == "operator"
        assert body["isOperator"] is True
        assert body["tenantId"] == TENANT

    def test_a_mapped_client_is_admitted_as_a_client(self, provisioned):
        r = client.get("/me", headers=_hdr("analyst@client.example"))
        assert r.status_code == 200
        body = r.json()
        assert body["role"] == "client"
        assert body["isOperator"] is False

    def test_an_unmapped_user_is_refused_with_the_provisioning_message(self, provisioned):
        r = client.get("/mi/snapshots", headers=_hdr("stranger@nowhere.example"))
        assert r.status_code == 403
        body = r.json()
        assert body["errorCode"] == ErrorCode.ACCESS_NOT_PROVISIONED
        assert body["error"] == access.NOT_PROVISIONED_MESSAGE
        assert body["retryable"] is False, (
            "a client agent must not retry a provisioning gap — it will never "
            "clear without an administrator")

    def test_a_disabled_user_is_refused_distinguishably(self, provisioned):
        r = client.get("/mi/snapshots", headers=_hdr("revoked@client.example"))
        assert r.status_code == 403
        assert r.json()["errorCode"] == ErrorCode.ACCESS_DISABLED, (
            "revocation must be visible as revocation, not look like an "
            "account that was never set up")

    def test_a_cross_tenant_identity_cannot_read_this_deployment(self, provisioned):
        """The dataset layer resolves data from deployment configuration, not
        from the context, so a grant naming another tenant would be served THIS
        tenant's book. It has to be refused at the door."""
        r = client.get("/mi/snapshots", headers=_hdr("outsider@other.example"))
        assert r.status_code == 403
        assert r.json()["errorCode"] == ErrorCode.TENANT_MISMATCH

    def test_a_forged_identity_gains_nothing(self, provisioned):
        """Anyone who can reach the App Service directly can present any header
        they like. Forging the OLD authority — the roles — is now inert."""
        for roles in ([ "authenticated", "operator" ],
                      [ "authenticated", "client" ],
                      [ "operator", "client", "admin" ]):
            r = client.get("/mi/snapshots",
                           headers=_hdr("attacker@evil.example", roles))
            assert r.status_code == 403, f"roles {roles} admitted an unknown identity"
            assert r.json()["errorCode"] == ErrorCode.ACCESS_NOT_PROVISIONED

    def test_a_forged_tenant_claim_gains_nothing(self, provisioned):
        """The tenant comes from the directory, never from the principal, so
        adding tenant-looking claims changes nothing."""
        payload = {"identityProvider": "aad", "userId": "oid-x",
                   "userDetails": "attacker@evil.example",
                   "userRoles": ["authenticated", "operator"],
                   "tenant_id": TENANT, "tenantId": TENANT}
        hdr = base64.b64encode(json.dumps(payload).encode()).decode()
        r = client.get("/mi/snapshots", headers={"x-ms-client-principal": hdr})
        assert r.status_code == 403
        assert r.json()["errorCode"] == ErrorCode.ACCESS_NOT_PROVISIONED

    def test_an_anonymous_caller_is_still_401_not_403(self, provisioned):
        """The two conditions stay distinguishable: 401 means the platform
        signed nobody in, 403 means it did and Trakt does not know them."""
        assert client.get("/mi/snapshots").status_code == 401


# =========================================================================== #
# 2. SWA invitations are off the critical path
# =========================================================================== #
class TestInvitationsNoLongerRequired:

    def test_a_user_with_no_swa_role_at_all_is_admitted_when_provisioned(self, provisioned):
        """THE point of this change.

        ``["anonymous", "authenticated"]`` is verbatim what SWA forwards for a
        signed-in user who has never accepted an invitation — the state
        ``az staticwebapp users list`` showed for every real user. Under the old
        rule that was a guaranteed 403. It must now succeed.
        """
        r = client.get("/mi/snapshots",
                       headers=_hdr("analyst@client.example",
                                    ["anonymous", "authenticated"]))
        assert r.status_code == 200

    def test_the_directory_not_the_platform_decides_the_role(self, provisioned):
        """A client the platform calls an operator is still a client."""
        r = client.get("/me", headers=_hdr("analyst@client.example",
                                           ["authenticated", "operator"]))
        assert r.status_code == 200
        assert r.json()["role"] == "client"
        assert r.json()["isOperator"] is False
        assert "operator" in r.json()["platformRoles"], (
            "the platform's view should still be echoed for migration "
            "visibility, just not obeyed")

    def test_an_object_id_resolves_as_well_as_an_email(self, provisioned):
        """A mailbox can be renamed; the object id is the durable key. Both are
        accepted so a deployment can pin either."""
        r = client.get("/me", headers=_hdr(
            None, user_id="11111111-2222-3333-4444-555555555555"))
        assert r.status_code == 200
        assert r.json()["role"] == "operator"

    def test_identity_matching_is_case_insensitive(self, provisioned):
        r = client.get("/me", headers=_hdr("OPS@Trakt.IO"))
        assert r.status_code == 200
        assert r.json()["role"] == "operator"


# =========================================================================== #
# 3. Portfolio contexts
# =========================================================================== #
class TestPortfolioContexts:

    def test_an_unrestricted_grant_may_select_any_context(self, provisioned):
        r = client.get("/mi/snapshots?portfolioContext=acquired",
                       headers=_hdr("analyst@client.example"))
        assert r.status_code == 200

    def test_a_restricted_grant_may_select_its_own_context(self, provisioned):
        r = client.get("/mi/snapshots?portfolioContext=direct",
                       headers=_hdr("narrow@client.example"))
        assert r.status_code == 200

    def test_a_restricted_grant_cannot_select_another_context(self, provisioned):
        r = client.get("/mi/snapshots?portfolioContext=acquired",
                       headers=_hdr("narrow@client.example"))
        assert r.status_code == 403
        assert r.json()["errorCode"] == ErrorCode.PERMISSION_DENIED

    def test_the_lens_alias_is_gated_too(self, provisioned):
        """Several routes accept ``lens`` as the older name for the same thing.
        Gating only the new name would leave the alias as an unchecked way in."""
        r = client.get("/mi/snapshots?lens=acquired",
                       headers=_hdr("narrow@client.example"))
        assert r.status_code == 403

    def test_naming_no_context_is_not_a_denial(self, provisioned):
        """"I asked for nothing" is the route's own default, not a widening."""
        r = client.get("/mi/snapshots", headers=_hdr("narrow@client.example"))
        assert r.status_code == 200

    def test_a_body_borne_lens_is_gated_too(self, provisioned):
        """``/mi/query`` takes the same scope as ``sourcePortfolioLens`` in the
        POST body, where the global guard cannot see it. Missing that would let
        a restricted account simply ask the question instead."""
        r = client.post(
            "/mi/query",
            json={"question": "what is the book worth",
                  "sourcePortfolioLens": "acquired"},
            headers=_hdr("narrow@client.example"))
        assert r.status_code == 403
        assert r.json()["errorCode"] == ErrorCode.PERMISSION_DENIED

    def test_the_body_borne_lens_is_allowed_when_entitled(self, provisioned):
        r = client.post(
            "/mi/query",
            json={"question": "what is the book worth",
                  "sourcePortfolioLens": "direct"},
            headers=_hdr("narrow@client.example"))
        assert r.status_code != 403

    def test_deck_generation_cannot_export_an_unauthorised_context(self, provisioned):
        """An investor pack is the artefact a scope restriction most needs to
        bound: generating one over a context the account cannot view would
        export precisely what it is not allowed to read."""
        r = client.post("/mi/decks/generate",
                        json={"portfolioContext": "acquired"},
                        headers=_hdr("narrow@client.example"))
        assert r.status_code == 403
        assert r.json()["errorCode"] == ErrorCode.PERMISSION_DENIED

    def test_every_context_parameter_the_routes_accept_is_gated(self):
        """The guard checks a fixed list of query parameters. If a route ever
        introduces a third alias, this fails rather than letting the new name
        escape the check.

        Body fields are checked separately because they cannot be gated in the
        guard at all — reading the body there would consume the stream. Any new
        one needs its own call to ``access.authorise_context`` in the handler,
        and this test names the ones that have it.
        """
        import inspect

        declared = set()
        for route in app.routes:
            endpoint = getattr(route, "endpoint", None)
            if endpoint is None:
                continue
            try:
                params = inspect.signature(endpoint).parameters
            except (TypeError, ValueError):  # pragma: no cover - builtins
                continue
            for name in params:
                low = name.lower()
                if "portfoliocontext" in low or low == "lens":
                    declared.add(name)

        missed = declared - set(auth_mod.CONTEXT_QUERY_PARAMS)
        assert not missed, (
            f"context-selecting parameters not gated by auth_guard: {missed}. "
            f"Add them to auth.CONTEXT_QUERY_PARAMS.")

    def test_every_context_field_on_a_request_model_is_gated(self):
        """The body-borne equivalent of the test above."""
        from mi_agent_api import app as app_mod

        #: Body fields that select a governed context, each already passed to
        #: ``access.authorise_context`` by its handler.
        gated_fields = {"sourcePortfolioLens", "portfolioContext"}

        declared = set()
        for name in dir(app_mod):
            model = getattr(app_mod, name)
            fields = getattr(model, "model_fields", None)
            if not isinstance(fields, dict):
                continue
            for field in fields:
                low = field.lower()
                if "portfoliocontext" in low or low.endswith("lens"):
                    declared.add(field)

        missed = declared - gated_fields
        assert not missed, (
            f"context-selecting request-model fields with no authorisation "
            f"check: {missed}. Call access.authorise_context on them in the "
            f"handler and add them here.")


# =========================================================================== #
# 4. The directory itself
# =========================================================================== #
class TestDirectoryLoading:

    def test_a_missing_file_provisions_nobody(self, tmp_path):
        """Fail closed. An authorisation source that degrades to "allow" is not
        an authorisation source — contrast the insight/tenancy configs, which
        fall back to permissive defaults because they govern presentation."""
        directory = access.load_access_directory(tmp_path / "absent.yaml",
                                                 refresh=True)
        assert len(directory) == 0
        assert directory.configured is False
        with pytest.raises(TraktError) as exc:
            access.resolve_grant("anyone@example.com", directory=directory)
        assert exc.value.code == ErrorCode.ACCESS_NOT_PROVISIONED

    def test_an_unparseable_file_provisions_nobody(self, tmp_path):
        path = tmp_path / "access.yaml"
        path.write_text("principals: [ this is not: valid: yaml", encoding="utf-8")
        directory = access.load_access_directory(path, refresh=True)
        assert len(directory) == 0

    def test_an_entry_with_an_unknown_role_is_dropped_not_guessed(self, tmp_path):
        path = tmp_path / "access.yaml"
        path.write_text(textwrap.dedent("""
            principals:
              - identities: [x@y.z]
                tenant_id: T
                role: superuser
        """), encoding="utf-8")
        directory = access.load_access_directory(path, refresh=True)
        assert len(directory) == 0, "an unrecognised role must not be admitted"

    def test_an_entry_without_a_tenant_is_dropped(self, tmp_path):
        path = tmp_path / "access.yaml"
        path.write_text(textwrap.dedent("""
            principals:
              - identities: [x@y.z]
                role: client
        """), encoding="utf-8")
        assert len(access.load_access_directory(path, refresh=True)) == 0

    def test_one_bad_entry_does_not_discard_the_good_ones(self, tmp_path):
        path = tmp_path / "access.yaml"
        path.write_text(textwrap.dedent("""
            principals:
              - identities: [good@y.z]
                tenant_id: T
                role: client
              - identities: [bad@y.z]
                role: client
        """), encoding="utf-8")
        directory = access.load_access_directory(path, refresh=True)
        assert directory.resolve("good@y.z") is not None
        assert directory.resolve("bad@y.z") is None

    def test_an_edit_takes_effect_without_a_restart(self, tmp_path):
        """Cached on mtime, not a TTL: adding a user must not need a redeploy,
        and a steady state must not re-parse YAML on every request."""
        path = tmp_path / "access.yaml"
        path.write_text(textwrap.dedent("""
            principals:
              - identities: [first@y.z]
                tenant_id: T
                role: client
        """), encoding="utf-8")
        assert access.load_access_directory(path).resolve("second@y.z") is None

        import os
        stamp = path.stat().st_mtime_ns
        path.write_text(textwrap.dedent("""
            principals:
              - identities: [first@y.z]
                tenant_id: T
                role: client
              - identities: [second@y.z]
                tenant_id: T
                role: operator
        """), encoding="utf-8")
        # Guarantee a distinct mtime even on a coarse-grained filesystem.
        os.utime(path, ns=(stamp + 1_000_000, stamp + 1_000_000))
        assert access.load_access_directory(path).resolve("second@y.z") is not None

    def test_the_shipped_directory_provisions_the_two_ere_operators(self):
        """The real config/access.yaml, as deployed."""
        directory = access.load_access_directory(
            _REPO_ROOT / "config" / "access.yaml", refresh=True)
        for email in ("admin@ratecheck.uk", "joshuahall@digifinsolutions.co.uk"):
            grant = directory.resolve(email)
            assert grant is not None, f"{email} is not provisioned"
            assert grant.tenant_id == "ERE"
            assert grant.platform_role == access.ROLE_OPERATOR
            assert grant.enabled is True
            assert not grant.restricted_to_contexts, (
                "these operators are deliberately unrestricted; an empty "
                "context list must mean 'all', not 'none'")

    def test_the_directory_holds_no_secrets(self):
        """Identities are not credentials, but a password or key in this file
        would be, and it is committed."""
        text = (_REPO_ROOT / "config" / "access.yaml").read_text(encoding="utf-8")
        # Comments are prose and legitimately discuss secrets; scan the DATA,
        # which is where one would actually sit.
        data = "\n".join(line.split("#", 1)[0]
                         for line in text.splitlines()).lower()
        for token in ("password", "secret", "api_key", "apikey",
                      "-----begin", "bearer "):
            assert token not in data, f"{token!r} appears in the access directory"


# =========================================================================== #
# 5. Health disclosure
# =========================================================================== #
class TestHealthDisclosure:

    def test_health_reports_the_directory_without_authentication(self, provisioned):
        """If the directory fails to deploy, every user is refused — so the one
        signal that diagnoses it must not itself sit behind the directory."""
        r = client.get("/health")
        assert r.status_code == 200
        status = r.json()["governance"]["accessDirectory"]
        assert status["configured"] is True
        assert status["provisioned_identities"] >= 5

    def test_health_does_not_disclose_who_is_provisioned(self, provisioned):
        """A count is diagnostic; a roster is a disclosure."""
        blob = json.dumps(r_json := client.get("/health").json())
        assert "ops@trakt.io" not in blob
        assert "analyst@client.example" not in blob
        assert r_json["governance"]["accessDirectory"]["provisioned_identities"]


# =========================================================================== #
# 6. Cross-channel consistency
# =========================================================================== #
class TestCrossChannel:

    def test_the_execution_context_takes_its_tenant_from_the_grant(self):
        """Every channel — React, PPTX, deck generation, Teams deep links —
        resolves tenancy through ExecutionContext, so putting the grant's tenant
        there is what makes one directory govern all of them."""
        from mi_agent_api import identity as identity_mod

        grant = access.AccessGrant(subject="ops@trakt.io", tenant_id="GRANTED",
                                   platform_role=access.ROLE_OPERATOR)

        class _P:
            user_id = "oid-1"
            user_details = "ops@trakt.io"
            roles = {"authenticated"}

        ctx = identity_mod.context_from_principal(
            _P(), tenant_id="DEPLOYMENT-DEFAULT", grant=grant)
        assert ctx.tenant_id == "GRANTED", (
            "the grant must win over the deployment-wide default, or a "
            "per-identity tenant is unrepresentable")

    def test_copilot_bearer_authentication_is_untouched(self):
        """Copilot validates a real Entra token and does not use the injected
        header, so the directory must not have been wired into its guard."""
        import inspect

        from mi_agent_api import copilot_auth

        source = inspect.getsource(copilot_auth)
        assert "trakt_core.access" not in source
        assert "resolve_grant" not in source

    def test_the_copilot_prefix_still_bypasses_the_header_guard(self, provisioned):
        """The guard must keep deferring to copilot_auth for /v1/copilot."""
        assert auth_mod.COPILOT_PATH_PREFIX == "/v1/copilot"


# =========================================================================== #
# 7. The UI quotes these constants — they must not drift
# =========================================================================== #
class TestFrontendContract:
    """Pinned here rather than in vitest so the browser build does not need node
    type declarations to read a Python file."""

    _TS = (_REPO_ROOT / "frontend" / "mi-agent-ui" / "src" / "domain"
           / "accessErrors.ts")

    def test_the_ui_quotes_the_provisioning_message_verbatim(self):
        """The backend owns the text; the UI quotes it. A drift means the user
        sees a generic failure instead of the explanation — the exact regression
        this change exists to remove."""
        assert self._TS.exists(), "the UI access-error module is missing"
        text = self._TS.read_text(encoding="utf-8")
        assert access.NOT_PROVISIONED_MESSAGE in text, (
            "trakt_core.access.NOT_PROVISIONED_MESSAGE no longer matches "
            f"{self._TS.name}; update both or the panel silently degrades")

    def test_the_ui_uses_the_same_error_codes(self):
        text = self._TS.read_text(encoding="utf-8")
        for code in (ErrorCode.ACCESS_NOT_PROVISIONED, ErrorCode.ACCESS_DISABLED,
                     ErrorCode.TENANT_MISMATCH):
            assert f'"{code}"' in text, f"{code} is not handled by the UI"
