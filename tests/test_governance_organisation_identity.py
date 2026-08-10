#!/usr/bin/env python3
"""Stage 1: external organisation identity.

The property under test throughout is the separation the organisation model
exists to create:

    a validated Microsoft directory says WHO IS ASKING
    it never says WHOSE DATA is served

Everything else here — duplicate rejection, disabled organisations, the two
operating modes — exists to make that separation enforceable rather than
conventional.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core.context import CHANNEL_COPILOT, ExecutionContext
from trakt_core.errors import ErrorCode, TraktError
from trakt_core.organisation import (
    ORG_TYPE_ORIGINATOR,
    ORG_TYPE_WAREHOUSE_FUNDER,
    OrganisationRecord,
    OrganisationRegistry,
    load_organisation_registry,
    normalise_directory_id,
)

ERE_DIR = "11111111-1111-1111-1111-111111111111"
FUNDER_DIR = "22222222-2222-2222-2222-222222222222"
STRANGER_DIR = "99999999-9999-9999-9999-999999999999"

ERE_ORG = OrganisationRecord(
    organisation_id="ere", display_name="Equity Release Europe",
    organisation_type=ORG_TYPE_ORIGINATOR,
    microsoft_tenant_ids=frozenset({ERE_DIR}))
FUNDER_ORG = OrganisationRecord(
    organisation_id="funder_a", display_name="Warehouse Funder A",
    organisation_type=ORG_TYPE_WAREHOUSE_FUNDER,
    microsoft_tenant_ids=frozenset({FUNDER_DIR}))


def _strict(*records) -> OrganisationRegistry:
    return OrganisationRegistry(records or (ERE_ORG, FUNDER_ORG), configured=True)


def _compat() -> OrganisationRegistry:
    return OrganisationRegistry((), configured=False)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def test_record_requires_an_organisation_id():
    with pytest.raises(ValueError):
        OrganisationRecord(organisation_id="")
    with pytest.raises(ValueError):
        OrganisationRecord(organisation_id="   ")


def test_record_is_immutable():
    with pytest.raises(Exception):
        ERE_ORG.organisation_id = "someone_else"  # type: ignore[misc]


def test_directory_ids_compare_case_insensitively():
    """Entra GUIDs are case-insensitive; a registration typed in upper case must
    still match a token that presents it in lower case."""
    record = OrganisationRecord(organisation_id="ERE",
                                microsoft_tenant_ids=frozenset({ERE_DIR.upper()}))
    assert record.organisation_id == "ere"
    assert record.owns_directory(ERE_DIR)
    assert record.owns_directory(ERE_DIR.upper())
    assert _strict(record).for_microsoft_tenant(ERE_DIR.upper()) is record


def test_normalise_directory_id_treats_blank_as_absent():
    """"No directory" must be one value, so it can never match a record that was
    registered with an empty string."""
    assert normalise_directory_id(None) is None
    assert normalise_directory_id("   ") is None
    assert normalise_directory_id(" ABC ") == "abc"


# --------------------------------------------------------------------------- #
# Registry: lookup
# --------------------------------------------------------------------------- #
def test_known_directory_resolves_to_its_organisation():
    registry = _strict()
    assert registry.resolve_microsoft_tenant(ERE_DIR).organisation_id == "ere"
    assert registry.resolve_microsoft_tenant(FUNDER_DIR).organisation_id == "funder_a"


def test_lookup_by_organisation_id():
    registry = _strict()
    assert registry.get("funder_a") is FUNDER_ORG
    assert registry.get("FUNDER_A") is FUNDER_ORG
    assert registry.get("nobody") is None


def test_unknown_directory_is_refused_in_organisation_mode():
    with pytest.raises(TraktError) as exc:
        _strict().resolve_microsoft_tenant(STRANGER_DIR)
    assert exc.value.code == ErrorCode.ORGANISATION_NOT_REGISTERED
    assert exc.value.http_status == 403


def test_absent_directory_is_refused_in_organisation_mode():
    """A request that carries no directory identity at all cannot be attributed
    to an organisation, so it is refused rather than defaulted."""
    with pytest.raises(TraktError) as exc:
        _strict().resolve_microsoft_tenant(None)
    assert exc.value.code == ErrorCode.ORGANISATION_NOT_REGISTERED


def test_disabled_organisation_is_refused():
    disabled = OrganisationRecord(organisation_id="funder_a",
                                  microsoft_tenant_ids=frozenset({FUNDER_DIR}),
                                  enabled=False)
    with pytest.raises(TraktError) as exc:
        _strict(ERE_ORG, disabled).resolve_microsoft_tenant(FUNDER_DIR)
    assert exc.value.code == ErrorCode.ORGANISATION_DISABLED
    assert exc.value.http_status == 403


def test_disabled_organisation_is_excluded_from_the_token_allow_list():
    disabled = OrganisationRecord(organisation_id="funder_a",
                                  microsoft_tenant_ids=frozenset({FUNDER_DIR}),
                                  enabled=False)
    registry = _strict(ERE_ORG, disabled)
    assert registry.microsoft_tenant_ids() == frozenset({ERE_DIR})
    assert registry.microsoft_tenant_ids(enabled_only=False) == frozenset(
        {ERE_DIR, FUNDER_DIR})


def test_compatibility_mode_claims_no_organisation_and_refuses_nothing():
    """The current ERE deployment: no config, so no organisation identity is
    asserted and no caller is newly refused."""
    registry = _compat()
    assert registry.configured is False
    assert registry.resolve_microsoft_tenant(ERE_DIR) is None
    assert registry.resolve_microsoft_tenant(STRANGER_DIR) is None
    assert registry.resolve_microsoft_tenant(None) is None


# --------------------------------------------------------------------------- #
# Registry: construction invariants
# --------------------------------------------------------------------------- #
def test_duplicate_directory_across_organisations_is_rejected():
    """Two organisations claiming one directory means "who is asking" has two
    answers. Silently picking one becomes an authorisation bug the moment
    entitlements exist, so the registry refuses to be built."""
    clash = OrganisationRecord(organisation_id="funder_a",
                               microsoft_tenant_ids=frozenset({ERE_DIR}))
    with pytest.raises(TraktError) as exc:
        OrganisationRegistry((ERE_ORG, clash), configured=True)
    assert exc.value.code == ErrorCode.INVALID_INPUT
    assert exc.value.details["microsoft_tenant_id"] == ERE_DIR
    assert exc.value.details["organisation_ids"] == ["ere", "funder_a"]


def test_duplicate_detection_is_deterministic():
    """Same records, either input order, same error — so a broken config always
    names the same conflicting pair and cannot be diagnosed differently on a
    different worker."""
    clash = OrganisationRecord(organisation_id="funder_a",
                               microsoft_tenant_ids=frozenset({ERE_DIR}))
    messages = []
    for records in ((ERE_ORG, clash), (clash, ERE_ORG)):
        with pytest.raises(TraktError) as exc:
            OrganisationRegistry(records, configured=True)
        messages.append((exc.value.code, exc.value.details))
    assert messages[0] == messages[1]


def test_duplicate_organisation_id_is_rejected():
    with pytest.raises(TraktError) as exc:
        OrganisationRegistry(
            (ERE_ORG, OrganisationRecord(organisation_id="ere")), configured=True)
    assert exc.value.code == ErrorCode.INVALID_INPUT


def test_records_are_returned_in_a_stable_order():
    assert [r.organisation_id for r in _strict().records()] == ["ere", "funder_a"]
    reversed_input = OrganisationRegistry((FUNDER_ORG, ERE_ORG), configured=True)
    assert [r.organisation_id for r in reversed_input.records()] == ["ere", "funder_a"]


# --------------------------------------------------------------------------- #
# Registry: loading
# --------------------------------------------------------------------------- #
_GOOD_CONFIG = f"""
organisations:
  - organisation_id: ere
    display_name: Equity Release Europe
    organisation_type: originator
    microsoft_tenant_ids: ["{ERE_DIR}"]
    enabled: true
  - organisation_id: funder_a
    display_name: Warehouse Funder A
    organisation_type: warehouse_funder
    microsoft_tenant_id: "{FUNDER_DIR}"
"""


def test_absent_config_is_compatibility_mode(tmp_path):
    registry = load_organisation_registry(tmp_path / "nope.yaml")
    assert registry.configured is False
    assert len(registry) == 0


def test_config_loads_both_the_plural_and_singular_directory_spelling(tmp_path):
    path = tmp_path / "organisations.yaml"
    path.write_text(_GOOD_CONFIG, encoding="utf-8")
    registry = load_organisation_registry(path)
    assert registry.configured is True
    assert registry.resolve_microsoft_tenant(ERE_DIR).organisation_id == "ere"
    assert registry.resolve_microsoft_tenant(FUNDER_DIR).organisation_id == "funder_a"
    assert registry.resolve_microsoft_tenant(FUNDER_DIR).enabled is True


def test_loading_is_deterministic(tmp_path):
    path = tmp_path / "organisations.yaml"
    path.write_text(_GOOD_CONFIG, encoding="utf-8")
    first = load_organisation_registry(path)
    second = load_organisation_registry(path)
    assert [r.to_dict() for r in first.records()] == [r.to_dict() for r in second.records()]
    assert first.microsoft_tenant_ids() == second.microsoft_tenant_ids()


def test_mapping_form_of_the_config_is_accepted(tmp_path):
    path = tmp_path / "organisations.yaml"
    path.write_text(
        f'organisations:\n  ere:\n    microsoft_tenant_ids: ["{ERE_DIR}"]\n',
        encoding="utf-8")
    assert load_organisation_registry(path).resolve_microsoft_tenant(
        ERE_DIR).organisation_id == "ere"


def test_status_disabled_is_honoured(tmp_path):
    path = tmp_path / "organisations.yaml"
    path.write_text(
        f'organisations:\n  - organisation_id: ere\n'
        f'    microsoft_tenant_ids: ["{ERE_DIR}"]\n    status: disabled\n',
        encoding="utf-8")
    with pytest.raises(TraktError) as exc:
        load_organisation_registry(path).resolve_microsoft_tenant(ERE_DIR)
    assert exc.value.code == ErrorCode.ORGANISATION_DISABLED


def test_a_broken_config_fails_closed_rather_than_reopening_the_deployment(tmp_path):
    """The deliberate divergence from load_tenant_registry.

    Tenancy degrades to single-tenant on a bad file because the dataset resolver
    still scopes reads. Doing that here would silently take a deployment that had
    declared "these directories only" back to accepting anything — so a bad file
    stays in organisation mode with nothing registered, refusing everything, and
    still does not raise (the process must start so the React path keeps working).
    """
    path = tmp_path / "organisations.yaml"
    path.write_text("organisations: [ this is not: valid: yaml", encoding="utf-8")
    registry = load_organisation_registry(path)
    assert registry.configured is True
    assert len(registry) == 0
    with pytest.raises(TraktError) as exc:
        registry.resolve_microsoft_tenant(ERE_DIR)
    assert exc.value.code == ErrorCode.ORGANISATION_NOT_REGISTERED


def test_an_ambiguous_config_fails_closed_rather_than_picking_one(tmp_path):
    path = tmp_path / "organisations.yaml"
    path.write_text(
        f'organisations:\n'
        f'  - organisation_id: ere\n    microsoft_tenant_ids: ["{ERE_DIR}"]\n'
        f'  - organisation_id: funder_a\n    microsoft_tenant_ids: ["{ERE_DIR}"]\n',
        encoding="utf-8")
    registry = load_organisation_registry(path)
    assert registry.configured is True and len(registry) == 0
    with pytest.raises(TraktError):
        registry.resolve_microsoft_tenant(ERE_DIR)


def test_the_shipped_example_config_is_loadable(tmp_path):
    """The example is documentation, but a broken example teaches a broken shape."""
    source = (_REPO_ROOT / "config" / "organisations.example.yaml").read_text(
        encoding="utf-8")
    path = tmp_path / "organisations.yaml"
    path.write_text(source, encoding="utf-8")
    registry = load_organisation_registry(path)
    assert registry.configured is True
    assert registry.get("ere") is not None


def test_the_repository_ships_no_live_organisation_config():
    """Compatibility is the default: committing config/organisations.yaml would
    put every deployment (and this test suite) into organisation mode."""
    assert not (_REPO_ROOT / "config" / "organisations.yaml").exists()


# --------------------------------------------------------------------------- #
# trakt_core stays interface-neutral
# --------------------------------------------------------------------------- #
def test_organisation_module_imports_without_a_web_framework():
    import subprocess
    import os
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, trakt_core.organisation as o;"
         "assert 'fastapi' not in sys.modules, 'organisation pulled in fastapi';"
         "assert 'starlette' not in sys.modules, 'organisation pulled in starlette';"
         "assert 'pandas' not in sys.modules, 'organisation pulled in pandas';"
         "o.load_organisation_registry();"
         "print('ok')"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
        env={**os.environ, "TRAKT_RUNTIME_MODE": "test", "PYTHONPATH": str(_REPO_ROOT)})
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


# --------------------------------------------------------------------------- #
# ExecutionContext
# --------------------------------------------------------------------------- #
def test_organisation_fields_default_to_absent():
    """Every pre-existing construction site keeps working unchanged."""
    ctx = ExecutionContext(tenant_id="ERE", actor_id="user-1")
    assert ctx.organisation_id is None
    assert ctx.microsoft_tenant_id is None


def test_organisation_fields_are_immutable():
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u", organisation_id="funder_a")
    with pytest.raises(Exception):
        ctx.organisation_id = "ere"  # type: ignore[misc]
    with pytest.raises(Exception):
        ctx.microsoft_tenant_id = ERE_DIR  # type: ignore[misc]


def test_organisation_fields_are_normalised():
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u",
                           organisation_id="  Funder_A ",
                           microsoft_tenant_id=f"  {FUNDER_DIR.upper()} ")
    assert ctx.organisation_id == "funder_a"
    assert ctx.microsoft_tenant_id == FUNDER_DIR
    blank = ExecutionContext(tenant_id="ERE", actor_id="u",
                             organisation_id="  ", microsoft_tenant_id="")
    assert blank.organisation_id is None and blank.microsoft_tenant_id is None


def test_organisation_identity_reaches_the_audit_projection():
    ctx = ExecutionContext(tenant_id="ERE", actor_id="oid-1",
                           organisation_id="funder_a",
                           microsoft_tenant_id=FUNDER_DIR)
    event = ctx.to_audit_dict()
    assert event["organisation_id"] == "funder_a"
    assert event["microsoft_tenant_id"] == FUNDER_DIR
    # WHO ASKED and WHOSE DATA are both present, and distinct.
    assert event["tenant_id"] == "ERE"
    assert event["organisation_id"] != event["tenant_id"]


def test_audit_projection_still_withholds_the_actor_label():
    ctx = ExecutionContext(tenant_id="ERE", actor_id="oid-1",
                           actor_label="someone@example.com",
                           organisation_id="funder_a")
    assert "someone@example.com" not in str(ctx.to_audit_dict())


# --------------------------------------------------------------------------- #
# Copilot identity: the directory resolves the organisation, never the tenant
# --------------------------------------------------------------------------- #
def _principal(tid=None, subject="oid-9", scopes=("Trakt.Copilot",)):
    from mi_agent_api.copilot_auth import CopilotPrincipal
    return CopilotPrincipal(subject=subject, name="agent", tenant_id=tid,
                            scopes=list(scopes))


def test_compatibility_mode_copilot_context_is_unchanged():
    """The current ERE Copilot path: no organisation config, so the context is
    exactly what it was before this stage, plus an unset organisation."""
    from mi_agent_api import identity

    ctx = identity.context_from_copilot_principal(
        _principal(tid=ERE_DIR), tenant_id="ERE",
        organisation_registry=_compat())
    assert ctx.tenant_id == "ERE"
    assert ctx.channel == CHANNEL_COPILOT
    assert ctx.organisation_id is None
    # The directory is still recorded — identity metadata, no authority.
    assert ctx.microsoft_tenant_id == ERE_DIR


def test_a_registered_second_directory_authenticates_and_is_attributed():
    """The point of the stage: a funder signing in from its OWN directory is
    recognised as a distinct organisation."""
    from mi_agent_api import identity

    ctx = identity.context_from_copilot_principal(
        _principal(tid=FUNDER_DIR), tenant_id="ERE",
        organisation_registry=_strict())
    assert ctx.organisation_id == "funder_a"
    assert ctx.microsoft_tenant_id == FUNDER_DIR


def test_an_unregistered_directory_cannot_obtain_a_context():
    from mi_agent_api import identity

    with pytest.raises(TraktError) as exc:
        identity.context_from_copilot_principal(
            _principal(tid=STRANGER_DIR), tenant_id="ERE",
            organisation_registry=_strict())
    assert exc.value.code == ErrorCode.ORGANISATION_NOT_REGISTERED


def test_a_disabled_organisation_cannot_obtain_a_context():
    from mi_agent_api import identity

    disabled = OrganisationRecord(organisation_id="funder_a",
                                  microsoft_tenant_ids=frozenset({FUNDER_DIR}),
                                  enabled=False)
    with pytest.raises(TraktError) as exc:
        identity.context_from_copilot_principal(
            _principal(tid=FUNDER_DIR), tenant_id="ERE",
            organisation_registry=_strict(ERE_ORG, disabled))
    assert exc.value.code == ErrorCode.ORGANISATION_DISABLED


def test_the_directory_never_becomes_the_trakt_tenant():
    """THE Stage 1 invariant. A funder's directory resolves a funder
    organisation; the data-owning tenant is still the deployment's, and the
    organisation id has not leaked into it."""
    from mi_agent_api import identity

    ctx = identity.context_from_copilot_principal(
        _principal(tid=FUNDER_DIR), tenant_id="ERE",
        organisation_registry=_strict())
    assert ctx.tenant_id == "ERE"
    assert ctx.tenant_id != ctx.organisation_id
    assert ctx.tenant_id != ctx.microsoft_tenant_id

    # And the same directory against a different deployment tenant yields the
    # same organisation with that deployment's tenant — the two axes are
    # genuinely independent, not two names for one value.
    other = identity.context_from_copilot_principal(
        _principal(tid=FUNDER_DIR), tenant_id="OTHERCO",
        organisation_registry=_strict())
    assert other.organisation_id == "funder_a"
    assert other.tenant_id == "OTHERCO"


def test_missing_principal_still_fails_closed_before_organisation_resolution():
    from mi_agent_api import identity

    with pytest.raises(TraktError) as exc:
        identity.context_from_copilot_principal(
            None, tenant_id="ERE", organisation_registry=_strict())
    assert exc.value.code == ErrorCode.AUTHENTICATION_REQUIRED


def test_react_path_is_untouched_by_organisation_mode(monkeypatch, tmp_path):
    """Easy Auth hands over no verified directory, so organisation mode must not
    start refusing the signed-in ERE React user."""
    monkeypatch.delenv("WEBSITE_SITE_NAME", raising=False)
    monkeypatch.delenv("WEBSITE_INSTANCE_ID", raising=False)
    config = tmp_path / "organisations.yaml"
    config.write_text(_GOOD_CONFIG, encoding="utf-8")
    monkeypatch.setenv("TRAKT_ORGANISATIONS_CONFIG", str(config))

    from mi_agent_api import identity
    from mi_agent_api.auth import Principal

    ctx = identity.context_from_principal(
        Principal(user_id="oid-1", user_details="a@b.com", roles={"client"}),
        tenant_id="ERE")
    assert ctx.tenant_id == "ERE"
    assert ctx.organisation_id is None


def test_resolution_reads_the_configured_registry_when_none_is_injected(monkeypatch,
                                                                        tmp_path):
    config = tmp_path / "organisations.yaml"
    config.write_text(_GOOD_CONFIG, encoding="utf-8")
    monkeypatch.setenv("TRAKT_ORGANISATIONS_CONFIG", str(config))

    from mi_agent_api import identity

    ctx = identity.context_from_copilot_principal(
        _principal(tid=FUNDER_DIR), tenant_id="ERE")
    assert ctx.organisation_id == "funder_a"

    with pytest.raises(TraktError):
        identity.context_from_copilot_principal(
            _principal(tid=STRANGER_DIR), tenant_id="ERE")


# --------------------------------------------------------------------------- #
# Token layer: which directories are accepted
# --------------------------------------------------------------------------- #
def test_the_single_directory_app_setting_still_works(monkeypatch):
    """Backward compatibility for the deployed ERE app setting."""
    from mi_agent_api import copilot_auth

    monkeypatch.setenv("TRAKT_COPILOT_ENTRA_TENANT_ID", ERE_DIR)
    assert copilot_auth.allowed_directories(_compat()) == [ERE_DIR]


def test_the_app_setting_accepts_a_list(monkeypatch):
    from mi_agent_api import copilot_auth

    monkeypatch.setenv("TRAKT_COPILOT_ENTRA_TENANT_ID",
                       f" {ERE_DIR.upper()} , {FUNDER_DIR} ")
    assert copilot_auth.allowed_directories(_compat()) == sorted([ERE_DIR, FUNDER_DIR])


def test_registered_organisations_extend_the_allow_list(monkeypatch):
    """Registering an organisation is enough; its GUID need not be duplicated
    into an app setting."""
    from mi_agent_api import copilot_auth

    monkeypatch.delenv("TRAKT_COPILOT_ENTRA_TENANT_ID", raising=False)
    assert copilot_auth.allowed_directories(_strict()) == sorted([ERE_DIR, FUNDER_DIR])


def test_an_unconfigured_deployment_accepts_nothing(monkeypatch):
    from mi_agent_api import copilot_auth

    monkeypatch.delenv("TRAKT_COPILOT_ENTRA_TENANT_ID", raising=False)
    assert copilot_auth.allowed_directories(_compat()) == []


def test_directory_selection_prefers_the_token_claim():
    from mi_agent_api import copilot_auth

    token = _unsigned_token({"tid": FUNDER_DIR})
    assert copilot_auth._select_directory(token, [ERE_DIR, FUNDER_DIR]) == FUNDER_DIR


def test_directory_selection_refuses_an_unaccepted_claim():
    """The token layer's half of fail-closed: an unlisted directory is rejected
    before a single signing key is fetched."""
    from mi_agent_api import copilot_auth

    token = _unsigned_token({"tid": STRANGER_DIR})
    assert copilot_auth._select_directory(token, [ERE_DIR, FUNDER_DIR]) is None


def test_a_token_without_a_tid_resolves_only_on_a_single_directory_deployment():
    """Preserves the historical single-tenant behaviour, and refuses to guess
    once the deployment accepts more than one directory."""
    from mi_agent_api import copilot_auth

    token = _unsigned_token({"sub": "oid-1"})
    assert copilot_auth._select_directory(token, [ERE_DIR]) == ERE_DIR
    assert copilot_auth._select_directory(token, [ERE_DIR, FUNDER_DIR]) is None
    assert copilot_auth._select_directory(token, []) is None


def test_garbage_is_not_a_directory():
    from mi_agent_api import copilot_auth
    assert copilot_auth._select_directory("not-a-jwt", [ERE_DIR]) is None


def _unsigned_token(claims: dict) -> str:
    """A syntactically valid JWT with no usable signature.

    Enough to exercise directory *selection*, which happens before verification.
    Nothing here can pass ``_validate_bearer`` — that needs a real JWKS key —
    which is exactly the separation being tested.
    """
    import jwt
    return jwt.encode(claims, "not-the-real-signing-key-and-never-verified",
                      algorithm="HS256")
