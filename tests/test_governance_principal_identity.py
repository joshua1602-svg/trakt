#!/usr/bin/env python3
"""Stage 3.5: individual identity, bound to an organisation.

Stage 1 answered "which company is asking" from the Microsoft directory alone.
That is right for the company and insufficient on its own, because a warehouse
funder is not one person: when an analyst leaves, the directory is still the
funder's directory.

The two properties under test:

    a registered individual resolves to their organisation
    a disabled individual is refused WITHOUT disturbing that organisation

and, just as important, the compatibility property — a directory nobody has
enumerated behaves exactly as it did in Stage 1, so adding Funder A's first user
cannot lock ERE out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core.errors import ErrorCode, TraktError
from trakt_core.organisation import (
    ORG_TYPE_ORIGINATOR,
    ORG_TYPE_WAREHOUSE_FUNDER,
    OrganisationRecord,
    OrganisationRegistry,
)
from trakt_core.principal import (
    PRINCIPAL_ACTIVE,
    PRINCIPAL_DISABLED,
    PrincipalBinding,
    PrincipalRegistry,
    load_principal_registry,
    normalise_object_id,
)

ERE_DIR = "11111111-1111-1111-1111-111111111111"
FUNDER_DIR = "22222222-2222-2222-2222-222222222222"
JANE = "aaaaaaaa-0000-0000-0000-00000000000a"
COLLEAGUE = "bbbbbbbb-0000-0000-0000-00000000000b"
STRANGER = "cccccccc-0000-0000-0000-00000000000c"


def _jane(status=PRINCIPAL_ACTIVE) -> PrincipalBinding:
    return PrincipalBinding(organisation_id="funder_a",
                            microsoft_tenant_id=FUNDER_DIR,
                            microsoft_object_id=JANE,
                            display_name="Jane Smith",
                            email="jane.smith@example.com", status=status)


def _colleague() -> PrincipalBinding:
    return PrincipalBinding(organisation_id="funder_a",
                            microsoft_tenant_id=FUNDER_DIR,
                            microsoft_object_id=COLLEAGUE,
                            display_name="A. Colleague")


def _registry(*bindings, configured=True) -> PrincipalRegistry:
    return PrincipalRegistry(bindings or (_jane(),), configured=configured)


def _organisations() -> OrganisationRegistry:
    return OrganisationRegistry((
        OrganisationRecord(organisation_id="ere", organisation_type=ORG_TYPE_ORIGINATOR,
                           microsoft_tenant_ids=frozenset({ERE_DIR})),
        OrganisationRecord(organisation_id="funder_a",
                           organisation_type=ORG_TYPE_WAREHOUSE_FUNDER,
                           microsoft_tenant_ids=frozenset({FUNDER_DIR})),
    ), configured=True)


# --------------------------------------------------------------------------- #
# The binding
# --------------------------------------------------------------------------- #
def test_a_binding_requires_stable_microsoft_identity():
    for kwargs in ({"organisation_id": ""}, {"microsoft_tenant_id": ""},
                   {"microsoft_object_id": ""}, {"microsoft_object_id": "   "}):
        base = {"organisation_id": "funder_a", "microsoft_tenant_id": FUNDER_DIR,
                "microsoft_object_id": JANE, **kwargs}
        with pytest.raises(ValueError):
            PrincipalBinding(**base)


def test_a_binding_is_immutable_and_case_insensitive():
    binding = PrincipalBinding(organisation_id="Funder_A",
                               microsoft_tenant_id=FUNDER_DIR.upper(),
                               microsoft_object_id=JANE.upper())
    assert binding.organisation_id == "funder_a"
    assert binding.key == (FUNDER_DIR, JANE)
    with pytest.raises(Exception):
        binding.status = PRINCIPAL_DISABLED  # type: ignore[misc]


def test_an_unknown_status_is_refused():
    with pytest.raises(ValueError):
        PrincipalBinding(organisation_id="funder_a", microsoft_tenant_id=FUNDER_DIR,
                         microsoft_object_id=JANE, status="probationary")


def test_identity_is_the_object_id_not_the_email():
    """A display name is user-editable in most directories and an email is
    reassignable; keying on either would be an authorisation bug."""
    registry = _registry()
    assert registry.get(FUNDER_DIR, JANE) is not None
    # Nothing looks anybody up by name or address — there is no such method, and
    # the same person under a different display name is still the same binding.
    renamed = PrincipalBinding(organisation_id="funder_a",
                               microsoft_tenant_id=FUNDER_DIR,
                               microsoft_object_id=JANE,
                               display_name="Jane Smith-Jones",
                               email="new.address@example.com")
    assert renamed.key == _jane().key
    assert not hasattr(registry, "for_email")


def test_duplicate_principals_are_rejected():
    other_org = PrincipalBinding(organisation_id="ere",
                                 microsoft_tenant_id=FUNDER_DIR,
                                 microsoft_object_id=JANE)
    with pytest.raises(TraktError) as exc:
        PrincipalRegistry((_jane(), other_org), configured=True)
    assert exc.value.code == ErrorCode.INVALID_INPUT
    assert exc.value.details["organisation_ids"] == ["ere", "funder_a"]


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def test_a_registered_individual_resolves_to_their_organisation():
    registry = _registry(_jane(), _colleague())
    assert registry.resolve(FUNDER_DIR, JANE).organisation_id == "funder_a"
    assert registry.organisation_for(FUNDER_DIR, COLLEAGUE) == "funder_a"


def test_an_unregistered_individual_in_a_gated_directory_is_refused():
    with pytest.raises(TraktError) as exc:
        _registry().resolve(FUNDER_DIR, STRANGER)
    assert exc.value.code == ErrorCode.PRINCIPAL_NOT_REGISTERED
    assert exc.value.http_status == 403


def test_a_disabled_individual_is_refused():
    with pytest.raises(TraktError) as exc:
        _registry(_jane(status=PRINCIPAL_DISABLED)).resolve(FUNDER_DIR, JANE)
    assert exc.value.code == ErrorCode.PRINCIPAL_DISABLED


def test_disabling_one_person_does_not_disturb_their_colleagues():
    registry = _registry(_jane(status=PRINCIPAL_DISABLED), _colleague())
    with pytest.raises(TraktError):
        registry.resolve(FUNDER_DIR, JANE)
    assert registry.resolve(FUNDER_DIR, COLLEAGUE).active
    # The disabled binding is KEPT, so past activity stays attributable.
    assert registry.get(FUNDER_DIR, JANE) is not None
    assert len(registry.for_organisation("funder_a")) == 2
    assert len(registry.for_organisation("funder_a", active_only=True)) == 1


def test_a_missing_object_id_is_refused_in_a_gated_directory():
    for oid in (None, "", "   "):
        with pytest.raises(TraktError) as exc:
            _registry().resolve(FUNDER_DIR, oid)
        assert exc.value.code == ErrorCode.PRINCIPAL_NOT_REGISTERED


# --------------------------------------------------------------------------- #
# Per-directory enforcement — the compatibility property
# --------------------------------------------------------------------------- #
def test_a_directory_with_no_registered_people_is_not_gated():
    """ERE keeps working when Funder A's first user is added. Enforcement is
    opted into by registering somebody, not by a flag."""
    registry = _registry(_jane())
    assert registry.enforced_for(FUNDER_DIR) is True
    assert registry.enforced_for(ERE_DIR) is False
    # An unknown individual in the ungated directory is simply not checked.
    assert registry.resolve(ERE_DIR, STRANGER) is None


def test_an_absent_config_gates_nothing():
    dormant = PrincipalRegistry((), configured=False)
    assert dormant.enforced_directories() == frozenset()
    assert dormant.resolve(FUNDER_DIR, JANE) is None
    assert dormant.resolve(ERE_DIR, STRANGER) is None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
_CONFIG = f"""
principals:
  - organisation_id: funder_a
    microsoft_tenant_id: "{FUNDER_DIR}"
    microsoft_object_id: "{JANE}"
    display_name: Jane Smith
  - organisation_id: funder_a
    microsoft_tenant_id: "{FUNDER_DIR}"
    microsoft_object_id: "{COLLEAGUE}"
    status: disabled
"""


def test_absent_config_is_dormant(tmp_path):
    registry = load_principal_registry(tmp_path / "nope.yaml")
    assert registry.configured is False and len(registry) == 0


def test_config_loads_and_is_deterministic(tmp_path):
    path = tmp_path / "principals.yaml"
    path.write_text(_CONFIG, encoding="utf-8")
    first, second = load_principal_registry(path), load_principal_registry(path)
    assert [b.to_dict() for b in first.bindings()] == [
        b.to_dict() for b in second.bindings()]
    assert first.resolve(FUNDER_DIR, JANE).display_name == "Jane Smith"
    with pytest.raises(TraktError):
        first.resolve(FUNDER_DIR, COLLEAGUE)


def test_a_broken_config_gates_nothing_rather_than_refusing_everything(tmp_path):
    """The deliberate asymmetry with the organisation registry.

    An unreadable ORGANISATION file means Trakt cannot tell who is asking and
    must refuse. An unreadable PRINCIPAL file means Trakt has lost a narrowing
    control it did not have at all before Stage 3.5 — failing every request
    closed there would be an outage caused by an additive feature.
    """
    path = tmp_path / "principals.yaml"
    path.write_text("principals: [ not: valid: yaml", encoding="utf-8")
    registry = load_principal_registry(path)
    assert registry.configured is True and len(registry) == 0
    assert registry.enforced_for(FUNDER_DIR) is False
    assert registry.resolve(FUNDER_DIR, JANE) is None


def test_the_shipped_example_config_is_loadable(tmp_path):
    source = (_REPO_ROOT / "config" / "principals.example.yaml").read_text(
        encoding="utf-8")
    path = tmp_path / "principals.yaml"
    path.write_text(source, encoding="utf-8")
    assert len(load_principal_registry(path)) == 2


def test_the_repository_ships_no_live_principal_config():
    assert not (_REPO_ROOT / "config" / "principals.yaml").exists()


# --------------------------------------------------------------------------- #
# The identity path
# --------------------------------------------------------------------------- #
def _principal(tid, oid):
    from mi_agent_api.copilot_auth import CopilotPrincipal
    return CopilotPrincipal(subject=oid, name="user", tenant_id=tid,
                            scopes=["Trakt.Copilot"])


def test_the_context_path_is_unchanged_when_no_principals_are_registered():
    """Stage 1 behaviour, preserved exactly."""
    from mi_agent_api import identity

    ctx = identity.context_from_copilot_principal(
        _principal(ERE_DIR, "oid-1"), tenant_id="ERE",
        organisation_registry=_organisations(),
        principal_registry=PrincipalRegistry((), configured=False))
    assert ctx.organisation_id == "ere"
    assert ctx.tenant_id == "ERE"


def test_a_registered_individual_gets_a_context():
    from mi_agent_api import identity

    ctx = identity.context_from_copilot_principal(
        _principal(FUNDER_DIR, JANE), tenant_id="ERE",
        organisation_registry=_organisations(),
        principal_registry=_registry())
    assert ctx.organisation_id == "funder_a"
    assert ctx.microsoft_tenant_id == FUNDER_DIR
    # WHOSE DATA is still the deployment's, untouched by any of this.
    assert ctx.tenant_id == "ERE"


def test_a_disabled_individual_cannot_obtain_a_context_at_all():
    """Refused before an organisation is resolved, so a departed employee never
    reaches their still-entitled employer's grants."""
    from mi_agent_api import identity

    with pytest.raises(TraktError) as exc:
        identity.context_from_copilot_principal(
            _principal(FUNDER_DIR, JANE), tenant_id="ERE",
            organisation_registry=_organisations(),
            principal_registry=_registry(_jane(status=PRINCIPAL_DISABLED)))
    assert exc.value.code == ErrorCode.PRINCIPAL_DISABLED


def test_an_unregistered_individual_in_a_gated_directory_gets_no_context():
    from mi_agent_api import identity

    with pytest.raises(TraktError) as exc:
        identity.context_from_copilot_principal(
            _principal(FUNDER_DIR, STRANGER), tenant_id="ERE",
            organisation_registry=_organisations(),
            principal_registry=_registry())
    assert exc.value.code == ErrorCode.PRINCIPAL_NOT_REGISTERED


def test_a_gated_directory_does_not_gate_the_others():
    """Funder A's users being registered must not lock ERE out."""
    from mi_agent_api import identity

    ctx = identity.context_from_copilot_principal(
        _principal(ERE_DIR, "some-ere-user"), tenant_id="ERE",
        organisation_registry=_organisations(),
        principal_registry=_registry())
    assert ctx.organisation_id == "ere"


def test_a_contradictory_binding_is_refused_in_either_direction():
    """A principal whose binding disagrees with their directory's organisation
    is broken server-side config, and is resolved in neither direction."""
    from mi_agent_api import identity

    mismatched = PrincipalRegistry((
        PrincipalBinding(organisation_id="ere", microsoft_tenant_id=FUNDER_DIR,
                         microsoft_object_id=JANE),), configured=True)
    with pytest.raises(TraktError) as exc:
        identity.context_from_copilot_principal(
            _principal(FUNDER_DIR, JANE), tenant_id="ERE",
            organisation_registry=_organisations(),
            principal_registry=mismatched)
    assert exc.value.code == ErrorCode.PRINCIPAL_NOT_REGISTERED


def test_principal_module_imports_without_a_web_framework_or_pandas():
    import os
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, trakt_core.principal as p;"
         "assert 'fastapi' not in sys.modules;"
         "assert 'starlette' not in sys.modules;"
         "assert 'pandas' not in sys.modules;"
         "p.load_principal_registry();"
         "print('ok')"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
        env={**os.environ, "TRAKT_RUNTIME_MODE": "test", "PYTHONPATH": str(_REPO_ROOT)})
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
