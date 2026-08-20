#!/usr/bin/env python3
"""The dashboard's Entra bearer path: authentication, and the authorisation
that deliberately does not follow from it.

Two things are being proved, and keeping them apart is the point of the file:

* **Microsoft says who you are.** A token is accepted only if it verifies
  against the issuing directory's published keys, names this API's audience,
  has not expired, comes from a directory on the allow-list, carries a
  *delegated* scope, and states both ``tid`` and ``oid``. Every negative case
  below is one of those checks failing.
* **Trakt says whether you may.** A perfectly valid Microsoft token from a
  registered organisation still gets nothing unless the person behind it is a
  registered, active principal. "Signed in" and "entitled" are different
  questions with different answers, and the second one is Trakt's.

The mode switch (``TRAKT_MI_REACT_AUTH_MODE``) is tested as carefully as the
token policy, because the migration's safety rests on it: ``swa`` must behave
exactly as the deployment does today, and ``bearer`` must stop honouring the
Static Web Apps header — while it still does, anything that can reach this API
directly can present one.
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jwt as pyjwt
from cryptography.hazmat.primitives.asymmetric import rsa

from mi_agent_api import copilot_auth, identity as identity_mod, react_auth
from mi_agent_api.app import app
from trakt_core.context import ACTOR_USER, CHANNEL_REACT, DEFAULT_MI_SCOPES
from trakt_core.errors import ErrorCode, TraktError
from trakt_core.organisation import (
    ORG_TYPE_INVESTOR,
    ORG_TYPE_ORIGINATOR,
    OrganisationRecord,
    OrganisationRegistry,
)
from trakt_core.principal import (
    PRINCIPAL_ACTIVE,
    PRINCIPAL_DISABLED,
    PrincipalBinding,
    PrincipalRegistry,
)

TENANT = "ERE"

#: The directory this deployment accepts today (the app setting), and a second
#: external organisation's directory — the case the whole migration exists for.
HOME_DIR = "11111111-1111-4111-8111-111111111111"
EXTERNAL_DIR = "22222222-2222-4222-8222-222222222222"
UNKNOWN_DIR = "33333333-3333-4333-8333-333333333333"

AUDIENCE = "api://mi-api-app-id"
SPA_CLIENT_ID = "spa-client-id"
SCOPE = react_auth.DEFAULT_REQUIRED_SCOPE

JOSH_OID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
COLLEAGUE_OID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"

client = TestClient(app)


# --------------------------------------------------------------------------- #
# Token minting. A real RS256 token against a stubbed JWKS, so the signature,
# audience and expiry checks are genuinely exercised rather than mocked past.
# --------------------------------------------------------------------------- #
_SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class _StubJWK:
    def __init__(self, key):
        self.key = key


class _StubJWKSClient:
    """Serves the one public key this deployment's directories are pretended to
    publish. A token signed by any other key therefore fails verification, which
    is what makes the forged-directory test meaningful."""

    def __init__(self, key):
        self._key = key

    def get_signing_key_from_jwt(self, _token):
        return _StubJWK(self._key)


def _token(*, directory: str = HOME_DIR, audience: Any = AUDIENCE,
           scopes: Optional[str] = SCOPE, roles: Optional[List[str]] = None,
           oid: Optional[str] = JOSH_OID, expires_in: int = 3600,
           issuer: Optional[str] = None, key=None,
           include_tid: bool = True, extra: Optional[Dict[str, Any]] = None) -> str:
    now = int(time.time())
    claims: Dict[str, Any] = {
        "iss": issuer or f"https://login.microsoftonline.com/{directory}/v2.0",
        "aud": audience,
        "iat": now - 10,
        "nbf": now - 10,
        "exp": now + expires_in,
        "name": "Test User",
        "preferred_username": "test.user@example.com",
        "azp": SPA_CLIENT_ID,
    }
    if include_tid:
        claims["tid"] = directory
    if oid is not None:
        claims["oid"] = oid
    if scopes is not None:
        claims["scp"] = scopes
    if roles is not None:
        claims["roles"] = roles
    if extra:
        claims.update(extra)
    return pyjwt.encode(claims, key or _SIGNING_KEY, algorithm="RS256")


def _bearer(**kwargs) -> Dict[str, str]:
    return {"Authorization": f"Bearer {_token(**kwargs)}"}


def _swa_header(roles=("authenticated", "client")) -> Dict[str, str]:
    """The Static Web Apps principal the deployment uses today."""
    blob = base64.b64encode(json.dumps({
        "identityProvider": "aad", "userId": "swa-oid",
        "userDetails": "admin@digifinsolutions.co.uk", "userRoles": list(roles),
    }).encode()).decode()
    return {"x-ms-client-principal": blob}


@pytest.fixture()
def bearer_env(monkeypatch):
    """A deployment configured to accept tokens from HOME_DIR and EXTERNAL_DIR."""
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "true")
    monkeypatch.setenv(react_auth.AUDIENCE_ENV, AUDIENCE)
    monkeypatch.setenv("TRAKT_COPILOT_ENTRA_TENANT_ID", f"{HOME_DIR},{EXTERNAL_DIR}")
    monkeypatch.delenv(react_auth.SCOPE_ENV, raising=False)
    monkeypatch.delenv(identity_mod.REQUIRE_PRINCIPAL_ENV, raising=False)
    monkeypatch.setattr(copilot_auth, "_jwks_client",
                        lambda _tenant: _StubJWKSClient(_SIGNING_KEY.public_key()))
    yield


@pytest.fixture()
def both_mode(bearer_env, monkeypatch):
    monkeypatch.setenv(react_auth.MODE_ENV, react_auth.MODE_BOTH)
    yield


@pytest.fixture()
def bearer_only(bearer_env, monkeypatch):
    monkeypatch.setenv(react_auth.MODE_ENV, react_auth.MODE_BEARER)
    yield


# =========================================================================== #
# 1. The migration switch
# =========================================================================== #
class TestAuthMode:
    """The switch is what makes the cutover reversible, so it is tested as
    carefully as the token policy."""

    def test_the_default_is_the_deployment_we_have_today(self, monkeypatch):
        monkeypatch.delenv(react_auth.MODE_ENV, raising=False)
        assert react_auth.auth_mode() == react_auth.MODE_SWA
        assert react_auth.bearer_enabled() is False

    def test_in_swa_mode_a_bearer_token_is_not_a_credential(self, bearer_env, monkeypatch):
        """Deploying this module with the default must change nothing. A token
        is not merely refused — it is not looked at."""
        monkeypatch.setenv(react_auth.MODE_ENV, react_auth.MODE_SWA)
        assert client.get("/me", headers=_bearer()).status_code == 401

    def test_in_swa_mode_the_existing_header_login_still_works(self, bearer_env, monkeypatch):
        monkeypatch.setenv(react_auth.MODE_ENV, react_auth.MODE_SWA)
        r = client.get("/me", headers=_swa_header())
        assert r.status_code == 200
        assert r.json()["roles"] == ["authenticated", "client"]

    def test_in_both_mode_the_existing_header_login_still_works(self, both_mode):
        """The whole point of `both`: prove the new path without losing the old
        one, so an operator cannot lock themselves out mid-migration."""
        r = client.get("/me", headers=_swa_header())
        assert r.status_code == 200
        assert r.json()["authenticated"] is True

    def test_in_both_mode_a_valid_token_authenticates(self, both_mode):
        r = client.get("/me", headers=_bearer())
        assert r.status_code == 200
        assert r.json()["credential"] == "bearer"

    def test_in_bearer_mode_the_header_stops_being_a_credential(self, bearer_only):
        """The last stage of the migration, and the only one that closes the
        side door: while the header is honoured, anything that can reach this
        API directly can present one."""
        r = client.get("/me", headers=_swa_header(roles=("authenticated", "operator")))
        assert r.status_code == 401

    def test_in_bearer_mode_a_valid_token_authenticates(self, bearer_only):
        assert client.get("/me", headers=_bearer()).status_code == 200

    def test_a_failed_token_never_falls_back_to_the_header(self, both_mode):
        """A rejected token must not become an anonymous-but-header-authenticated
        call. Presenting both is how an attacker would try exactly that."""
        headers = {**_swa_header(roles=("authenticated", "operator")),
                   **_bearer(audience="api://someone-else")}
        assert client.get("/me", headers=headers).status_code == 401

    def test_an_unknown_mode_fails_closed(self, bearer_env, monkeypatch):
        monkeypatch.setenv(react_auth.MODE_ENV, "off")
        assert client.get("/me", headers=_swa_header()).status_code == 503
        assert client.get("/me", headers=_bearer()).status_code == 503

    def test_an_unconfigured_bearer_deployment_says_so(self, both_mode, monkeypatch):
        """503, not 401: a deployment with no audience set refuses every token,
        and that must not look like a bad token to whoever is debugging it."""
        monkeypatch.delenv(react_auth.AUDIENCE_ENV, raising=False)
        assert client.get("/me", headers=_bearer()).status_code == 503

    def test_health_states_the_dashboard_auth_posture(self, both_mode):
        """One request should diagnose the migration state."""
        posture = client.get("/health").json()["governance"]["dashboardAuth"]
        assert posture["mode"] == react_auth.MODE_BOTH
        assert posture["bearerConfigured"] is True
        assert posture["requiresRegisteredPrincipal"] is False


# =========================================================================== #
# 2. Microsoft authentication — the negative cases
# =========================================================================== #
class TestTokenValidation:

    def test_an_authorised_tenant_and_user_reach_the_api(self, both_mode):
        r = client.get("/me", headers=_bearer(directory=EXTERNAL_DIR))
        assert r.status_code == 200
        body = r.json()
        assert body["authenticated"] is True
        assert body["microsoftTenantId"] == EXTERNAL_DIR
        assert body["microsoftObjectId"] == JOSH_OID

    def test_me_echoes_the_identifiers_a_registration_is_written_against(self, both_mode):
        """Step 15 of the rollout — capturing (tid, oid) — should not require
        reading logs or decoding a token by hand."""
        body = client.get("/me", headers=_bearer(directory=EXTERNAL_DIR,
                                                 oid=COLLEAGUE_OID)).json()
        assert (body["microsoftTenantId"], body["microsoftObjectId"]) == \
            (EXTERNAL_DIR, COLLEAGUE_OID)

    def test_an_unknown_tenant_is_refused(self, both_mode):
        assert client.get("/me", headers=_bearer(directory=UNKNOWN_DIR)).status_code == 401

    def test_an_unknown_tenant_is_indistinguishable_from_any_other_bad_token(self, both_mode):
        """The response must not confirm which organisations this deployment
        serves — otherwise the error message is a directory-enumeration oracle."""
        unknown = client.get("/me", headers=_bearer(directory=UNKNOWN_DIR))
        expired = client.get("/me", headers=_bearer(expires_in=-60))
        assert unknown.json() == expired.json()

    def test_a_forged_tenant_claim_cannot_borrow_an_accepted_directory(self, both_mode):
        """`tid` selects a key set from the allow-list; it is not trusted. A
        token claiming an accepted directory but signed by another key must not
        verify."""
        assert client.get("/me", headers=_bearer(key=_OTHER_KEY)).status_code == 401

    def test_a_wrong_audience_is_refused(self, both_mode):
        assert client.get("/me", headers=_bearer(
            audience="api://trakt-copilot-api")).status_code == 401

    def test_the_spa_id_token_is_not_an_api_credential(self, both_mode):
        """An ID token names the SPA's own client id as its audience. Accepting
        one would mean any app the user signs into could call this API."""
        assert client.get("/me", headers=_bearer(
            audience=SPA_CLIENT_ID, scopes=None)).status_code == 401

    def test_an_expired_token_is_refused(self, both_mode):
        assert client.get("/me", headers=_bearer(expires_in=-60)).status_code == 401

    def test_a_token_from_another_issuer_is_refused(self, both_mode):
        assert client.get("/me", headers=_bearer(
            issuer="https://login.microsoftonline.com/other/v2.0")).status_code == 401

    def test_a_token_with_no_tid_is_refused(self, both_mode):
        """A principal is keyed on (tid, oid). An unstated directory cannot be
        authorised, so it is refused rather than inferred."""
        assert client.get("/me", headers=_bearer(include_tid=False)).status_code == 401

    def test_a_token_with_no_oid_is_refused(self, both_mode):
        assert client.get("/me", headers=_bearer(oid=None)).status_code == 401

    def test_an_app_only_token_cannot_act_as_a_dashboard_user(self, both_mode):
        """Client-credentials tokens carry `roles` and no `scp`. A machine has
        its own surface; admitting one here would put no individual behind the
        request for the principal registry to check."""
        r = client.get("/me", headers=_bearer(scopes=None, roles=["Trakt.Agent"]))
        assert r.status_code == 403

    def test_a_token_for_another_trakt_surface_is_refused(self, both_mode):
        """A Copilot scope must not admit a caller to the dashboard."""
        r = client.get("/me", headers=_bearer(scopes="Trakt.Copilot"))
        assert r.status_code == 403

    def test_an_app_role_in_the_token_does_not_confer_operator_status(self, both_mode):
        """Once this API is provisioned in a customer's directory, THEIR admin can
        assign THEIR users any app role it defines. A privilege an external tenant
        can grant itself is not a privilege, so `roles` must never elevate."""
        body = client.get("/me", headers=_bearer(
            roles=["operator"], extra={"scp": SCOPE})).json()
        assert body["isOperator"] is False

    def test_a_malformed_bearer_value_is_refused(self, both_mode):
        for bad in ("Bearer ", "Bearer not-a-token", "Bearer " + "x" * 40):
            r = client.get("/me", headers={"Authorization": bad})
            assert r.status_code == 401, bad

    def test_the_required_scope_can_be_cleared_but_only_explicitly(self, monkeypatch):
        monkeypatch.delenv(react_auth.SCOPE_ENV, raising=False)
        assert react_auth._required_scope() == react_auth.DEFAULT_REQUIRED_SCOPE
        monkeypatch.setenv(react_auth.SCOPE_ENV, "")
        assert react_auth._required_scope() == ""
        monkeypatch.setenv(react_auth.SCOPE_ENV, "Custom.Scope")
        assert react_auth._required_scope() == "Custom.Scope"

    def test_react_auth_reuses_the_copilot_directory_allow_list(self):
        """Registering an organisation must open exactly one door, in one place.
        These names are the seam; a rename should break here rather than fork the
        allow-list silently."""
        for name in ("allowed_directories", "_select_directory", "_jwks_client",
                     "_allowed_issuers"):
            assert hasattr(copilot_auth, name), f"copilot_auth.{name} disappeared"
        source = Path(react_auth.__file__).read_text(encoding="utf-8")
        for name in ("allowed_directories", "_select_directory", "_jwks_client",
                     "_allowed_issuers"):
            assert f"_copilot_auth.{name}" in source, (
                f"react_auth no longer reuses copilot_auth.{name}")


# =========================================================================== #
# 3. Trakt authorisation — signing in is not entitlement
# =========================================================================== #
def _principal(directory: str = EXTERNAL_DIR, subject: str = JOSH_OID,
               status: str = PRINCIPAL_ACTIVE, organisation: str = "becquerel"):
    return PrincipalBinding(organisation_id=organisation,
                            microsoft_tenant_id=directory,
                            microsoft_object_id=subject,
                            display_name="Josh Hall",
                            email="josh.hall@becquerelventures.com",
                            status=status)


def _orgs() -> OrganisationRegistry:
    return OrganisationRegistry([
        OrganisationRecord(organisation_id="ere", display_name="Equity Release Europe",
                           organisation_type=ORG_TYPE_ORIGINATOR,
                           microsoft_tenant_ids=(HOME_DIR,)),
        OrganisationRecord(organisation_id="becquerel", display_name="Becquerel Ventures",
                           organisation_type=ORG_TYPE_INVESTOR,
                           microsoft_tenant_ids=(EXTERNAL_DIR,)),
    ], configured=True)


class _P:
    """Shaped exactly as ``react_auth.ReactPrincipal``."""

    def __init__(self, directory=EXTERNAL_DIR, subject=JOSH_OID,
                 name="josh.hall@becquerelventures.com"):
        self.directory_id = directory
        self.subject = subject
        self.name = name
        self.scopes = [SCOPE]
        self.roles = []


def _context(principal=None, *, orgs=None, principals=None, require=None):
    return identity_mod.context_from_react_principal(
        principal or _P(),
        tenant_id=TENANT,
        organisation_registry=orgs,
        principal_registry=principals,
        require_registered_principal=require,
    )


class TestTraktAuthorisation:

    def test_a_registered_active_user_resolves_their_organisation(self):
        registry = PrincipalRegistry([_principal()], configured=True)
        ctx = _context(orgs=_orgs(), principals=registry)
        assert ctx.organisation_id == "becquerel"
        assert ctx.microsoft_tenant_id == EXTERNAL_DIR
        assert ctx.tenant_id == TENANT          # WHOSE DATA — deployment config
        assert ctx.channel == CHANNEL_REACT
        assert ctx.actor_type == ACTOR_USER
        assert set(ctx.scopes) == set(DEFAULT_MI_SCOPES)

    def test_the_actor_is_the_object_id_never_the_email(self):
        """A display name is user-editable and an email is reassignable, so an
        audit trail keyed on either stops being attributable."""
        ctx = _context(orgs=_orgs(),
                       principals=PrincipalRegistry([_principal()], configured=True))
        assert ctx.actor_id == JOSH_OID
        assert "@" not in ctx.actor_id

    def test_a_valid_microsoft_user_who_is_not_registered_gets_nothing(self):
        """Becquerel's directory is registered and gated (Josh is in it); a
        colleague who is not registered is refused. This is the test that proves
        a valid Microsoft account is not Trakt access."""
        registry = PrincipalRegistry([_principal()], configured=True)
        with pytest.raises(TraktError) as exc:
            _context(_P(subject=COLLEAGUE_OID), orgs=_orgs(), principals=registry)
        assert exc.value.code == ErrorCode.PRINCIPAL_NOT_REGISTERED

    def test_a_disabled_user_is_refused_while_their_organisation_stays_entitled(self):
        registry = PrincipalRegistry([_principal(status=PRINCIPAL_DISABLED)],
                                     configured=True)
        with pytest.raises(TraktError):
            _context(orgs=_orgs(), principals=registry)
        # The organisation itself is untouched: a colleague still resolves.
        ok = PrincipalRegistry([_principal(), _principal(subject=COLLEAGUE_OID)],
                               configured=True)
        assert _context(_P(subject=COLLEAGUE_OID), orgs=_orgs(),
                        principals=ok).organisation_id == "becquerel"

    def test_an_unregistered_directory_is_refused_in_organisation_mode(self):
        with pytest.raises(TraktError) as exc:
            _context(_P(directory=UNKNOWN_DIR), orgs=_orgs(),
                     principals=PrincipalRegistry([], configured=True))
        assert exc.value.code == ErrorCode.ORGANISATION_NOT_REGISTERED

    def test_two_organisations_reach_the_same_client_dataset_distinguishably(self):
        """The point of the whole exercise: ERE's own staff and Becquerel's NED
        both read the ERE book, and every audit line says which of them asked."""
        registry = PrincipalRegistry([
            _principal(),
            _principal(directory=HOME_DIR, subject=COLLEAGUE_OID, organisation="ere"),
        ], configured=True)
        josh = _context(orgs=_orgs(), principals=registry)
        ere_user = _context(_P(directory=HOME_DIR, subject=COLLEAGUE_OID),
                            orgs=_orgs(), principals=registry)
        assert josh.tenant_id == ere_user.tenant_id == TENANT
        assert josh.organisation_id == "becquerel"
        assert ere_user.organisation_id == "ere"

    def test_a_binding_contradicting_its_directory_is_refused(self):
        """Contradictory server-side config is refused rather than resolved in
        either direction."""
        registry = PrincipalRegistry([_principal(organisation="ere")], configured=True)
        with pytest.raises(TraktError):
            _context(orgs=_orgs(), principals=registry)

    def test_compatibility_mode_still_works_before_the_registries_exist(self):
        """Phases 2-4 run before config/organisations.yaml is written. If this
        failed, the bearer path could not be proven before the posture change."""
        ctx = _context(orgs=OrganisationRegistry([], configured=False),
                       principals=PrincipalRegistry([], configured=False))
        assert ctx.organisation_id is None
        assert ctx.actor_id == JOSH_OID

    def test_named_user_licensing_can_be_made_fail_closed(self):
        """With TRAKT_MI_REACT_REQUIRE_PRINCIPAL on, a directory nobody has
        enumerated no longer admits everyone in it."""
        empty = PrincipalRegistry([], configured=True)
        with pytest.raises(TraktError) as exc:
            _context(orgs=_orgs(), principals=empty, require=True)
        assert exc.value.code == ErrorCode.PRINCIPAL_NOT_REGISTERED

    def test_the_fail_closed_switch_is_off_by_default(self, monkeypatch):
        """Because turning it on before the registry is populated refuses
        everyone — including the operator doing the migration."""
        monkeypatch.delenv(identity_mod.REQUIRE_PRINCIPAL_ENV, raising=False)
        assert identity_mod.requires_registered_principal() is False
        ctx = _context(orgs=_orgs(), principals=PrincipalRegistry([], configured=True))
        assert ctx.organisation_id == "becquerel"
        monkeypatch.setenv(identity_mod.REQUIRE_PRINCIPAL_ENV, "true")
        assert identity_mod.requires_registered_principal() is True

    def test_no_principal_means_no_context_rather_than_an_anonymous_one(self):
        with pytest.raises(TraktError) as exc:
            identity_mod.context_from_react_principal(None, tenant_id=TENANT)
        assert exc.value.code == ErrorCode.AUTHENTICATION_REQUIRED


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
