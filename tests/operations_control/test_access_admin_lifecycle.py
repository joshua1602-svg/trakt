"""Stage 3.5: the OCC access lifecycle, from an empty deployment to a live funder.

The scenario these tests follow is the one that actually happens:

    ERE goes live with no funder at all
        → two months later a warehouse lender joins the structure
            → six months later that lender hires somebody
                → later still, somebody leaves

Each of those is a separate, small administrative act. None of them reruns
client onboarding, and none of them is allowed to happen without a named human
confirming it.

The property tested throughout is that the OCC only ever *administers* access.
Every runtime decision still runs ExecutionContext → EntitlementStore →
ResourceCatalogue → authorise_resource_access against published configuration,
and ``test_the_occ_is_not_in_the_runtime_authorisation_path`` holds that line.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from operations_control.access_admin import (
    ADD_GRANT,
    ADD_MICROSOFT_TENANT,
    ADD_ORGANISATION,
    ADD_PRINCIPAL,
    ADD_RESOURCE,
    DISABLE_PRINCIPAL,
    ENABLE_PRINCIPAL,
    REVOKE_GRANT,
    SET_GRANT_CAPABILITIES,
    SOURCE_OCC_AGENT,
    UPDATE_RESOURCE,
    AccessAdminService,
    AccessChange,
    AccessChangeSet,
    ChangeSetError,
    ConfirmationError,
)
from operations_control.api import app as app_module
from operations_control.configuration.packages import (
    ADMIN_LAYERS,
    LAYER_ACCESS,
    ConfigPackageStore,
)
from trakt_core.context import (
    SCOPE_FORECAST_READ,
    SCOPE_MI_QUERY,
    SCOPE_PORTFOLIO_READ,
    SCOPE_RISK_READ,
    ExecutionContext,
)
from trakt_core.entitlement import authorise_resource_access, load_entitlement_store
from trakt_core.errors import ErrorCode, TraktError
from trakt_core.organisation import load_organisation_registry
from trakt_core.principal import load_principal_registry
from trakt_core.resource import ResourceRef, load_resource_catalogue

from .conftest import OP_A, OP_ADMIN, make_engine

ADMIN = {"X-Operator-Token": OP_ADMIN}
OPERATOR = {"X-Operator-Token": OP_A}

ERE_DIR = "11111111-1111-1111-1111-111111111111"
FUNDER_DIR = "22222222-2222-2222-2222-222222222222"
JANE_OID = "aaaaaaaa-0000-0000-0000-00000000000a"
COLLEAGUE_OID = "bbbbbbbb-0000-0000-0000-00000000000b"

FUNDER_CAPS = [SCOPE_PORTFOLIO_READ, SCOPE_MI_QUERY, SCOPE_RISK_READ,
               SCOPE_FORECAST_READ]

OPERATOR_NAME = "Administrator"


@pytest.fixture()
def service(store) -> AccessAdminService:
    return AccessAdminService(store, ConfigPackageStore(store))


@pytest.fixture()
def api(store, source_registry):
    engine = make_engine(store, source_registry)
    app_module.set_engine(engine)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    try:
        yield client, engine
    finally:
        app_module.set_engine(None)


def _change(action, **payload) -> AccessChange:
    return AccessChange(action=action, payload=payload)


def _set(*changes, by=OPERATOR_NAME, source="operator", title="") -> AccessChangeSet:
    return AccessChangeSet(changes=tuple(changes), proposed_by=by, source=source,
                           title=title)


# --------------------------------------------------------------------------- #
# The ERE starting point: live, and nothing configured.
# --------------------------------------------------------------------------- #
def test_an_unconfigured_deployment_starts_empty(service):
    """Seeded from a repository that ships no access files at all, which is the
    current single-client shape."""
    document = service.active_document()
    assert document.organisations == []
    assert document.principals == []
    assert document.resources == []
    assert document.entitlements == []


def test_the_access_layer_is_not_reachable_from_the_generic_admin_routes(api):
    """Access changes must arrive as structured actions, so the raw-YAML admin
    surface deliberately does not serve this layer."""
    assert LAYER_ACCESS not in ADMIN_LAYERS
    client, _ = api
    r = client.get("/ops/admin/config", headers=ADMIN)
    assert r.status_code == 200
    assert LAYER_ACCESS not in (r.json().get("layers") or {})


# --------------------------------------------------------------------------- #
# D. Month +2 — a warehouse funder joins a live client
# --------------------------------------------------------------------------- #
def _spv_i_resource() -> AccessChange:
    return _change(ADD_RESOURCE, tenant_id="ERE", kind="spv", resource_id="spv_i",
                   display_label="SPV I", spv_id="SPV_I",
                   source_portfolio_ids=["direct_001"], population="funded")


def _onboard_funder(service: AccessAdminService, *, confirm=True):
    change_set = _set(
        _spv_i_resource(),
        _change(ADD_ORGANISATION, organisation_id="funder_a",
                display_name="Warehouse Funder A",
                organisation_type="warehouse_funder",
                microsoft_tenant_ids=[FUNDER_DIR]),
        _change(ADD_GRANT, organisation_id="funder_a", resource="ERE/spv/spv_i",
                capabilities=FUNDER_CAPS),
        title="Warehouse Funder A joins the ERE structure")
    proposal = service.propose(change_set)
    if confirm:
        service.confirm(proposal.version, actor=OPERATOR_NAME)
    return proposal


def test_a_funder_can_be_added_to_a_live_client_in_one_change_set(service):
    proposal = _onboard_funder(service, confirm=False)
    assert proposal.valid
    assert proposal.status == "draft"
    assert proposal.change_set.requires_confirmation

    # Nothing is live until a human says so.
    assert service.active_document().organisations == []

    service.confirm(proposal.version, actor=OPERATOR_NAME)
    document = service.active_document()
    assert [o["organisation_id"] for o in document.organisations] == ["funder_a"]
    assert document.organisation("funder_a")["microsoft_tenant_ids"] == [FUNDER_DIR]
    grant = document.grant("funder_a", "ERE/spv/spv_i")
    assert sorted(grant["capabilities"]) == sorted(FUNDER_CAPS)


def test_the_activated_configuration_matches_the_stage_1_to_3_schemas(service, tmp_path):
    """The real proof: publish, then load with the RUNTIME loaders and make the
    authorisation decision the funder would trigger."""
    _onboard_funder(service)
    dest = tmp_path / "published"
    service.materialise(dest, actor=OPERATOR_NAME)

    organisations = load_organisation_registry(dest / "config/organisations.yaml")
    resources = load_resource_catalogue(dest / "config/resources.yaml")
    entitlements = load_entitlement_store(
        dest / "config/entitlements.yaml",
        organisations=organisations, resources=resources)

    assert organisations.resolve_microsoft_tenant(FUNDER_DIR).organisation_id \
        == "funder_a"

    spv_i = ResourceRef.parse("ERE/spv/spv_i")
    context = ExecutionContext(
        tenant_id="ERE", actor_id="pm@funder-a", organisation_id="funder_a",
        microsoft_tenant_id=FUNDER_DIR,
        entitlements=entitlements.resolve("funder_a"))

    auth = authorise_resource_access(context, spv_i, SCOPE_RISK_READ,
                                     catalogue=resources)
    assert auth.predicate() == {"view": "funded",
                                "source_portfolio_id": ["direct_001"],
                                "spv_id": "SPV_I"}
    # …and the data tenant is still ERE's, not the funder's.
    assert auth.tenant_id == "ERE" and auth.organisation_id == "funder_a"


def test_adding_a_funder_leaves_an_audit_trail(service, store):
    _onboard_funder(service)
    events = [r for r in store.list_audit("_admin")
              if r["event"].startswith("access_")]
    assert [e["event"] for e in events] == [
        "access_change_proposed", "access_change_confirmed"]
    confirmed = events[-1]
    assert confirmed["actor"] == OPERATOR_NAME
    assert any("ADD_GRANT funder_a -> ERE/spv/spv_i" in a
               for a in confirmed["detail"]["actions"])
    # The existing chain is hash-linked; adding to it must not break it.
    assert store.verify_audit_chain("_admin")


def test_no_token_or_payload_reaches_the_audit(service, store):
    _onboard_funder(service)
    blob = json.dumps(store.list_audit("_admin"))
    for forbidden in ("Bearer ", "authorization", "X-Operator-Token", OP_ADMIN):
        assert forbidden not in blob


# --------------------------------------------------------------------------- #
# E. Month +8 — add a user, without touching the funder's grants
# --------------------------------------------------------------------------- #
def test_a_user_can_be_added_later_without_rerunning_onboarding(service):
    _onboard_funder(service)
    before = service.active_document()
    grants_before = before.sorted_entitlements()
    orgs_before = before.sorted_organisations()

    proposal = service.propose(_set(
        _change(ADD_PRINCIPAL, organisation_id="funder_a",
                microsoft_tenant_id=FUNDER_DIR, microsoft_object_id=JANE_OID,
                display_name="Jane Smith", email="jane.smith@example.com"),
        title="Add Jane Smith at Warehouse Funder A"))
    assert proposal.valid
    service.confirm(proposal.version, actor=OPERATOR_NAME)

    after = service.active_document()
    # Jane exists…
    jane = after.principal(FUNDER_DIR, JANE_OID)
    assert jane["organisation_id"] == "funder_a" and jane["status"] == "active"
    # …and NOTHING else moved.
    assert after.sorted_entitlements() == grants_before
    assert after.sorted_organisations() == orgs_before
    assert after.sorted_resources() == before.sorted_resources()


def test_a_registered_user_resolves_to_their_organisation(service, tmp_path):
    _onboard_funder(service)
    service.confirm(service.propose(_set(
        _change(ADD_PRINCIPAL, organisation_id="funder_a",
                microsoft_tenant_id=FUNDER_DIR, microsoft_object_id=JANE_OID,
                display_name="Jane Smith"))).version, actor=OPERATOR_NAME)
    dest = tmp_path / "published"
    service.materialise(dest, actor=OPERATOR_NAME)

    principals = load_principal_registry(dest / "config/principals.yaml")
    assert principals.resolve(FUNDER_DIR, JANE_OID).organisation_id == "funder_a"
    assert principals.organisation_for(FUNDER_DIR, JANE_OID) == "funder_a"
    assert [b.microsoft_object_id
            for b in principals.for_organisation("funder_a")] == [JANE_OID]


def test_a_second_user_at_the_same_funder_is_a_one_line_change(service):
    _onboard_funder(service)
    for oid, name in ((JANE_OID, "Jane Smith"), (COLLEAGUE_OID, "A. Colleague")):
        service.confirm(service.propose(_set(
            _change(ADD_PRINCIPAL, organisation_id="funder_a",
                    microsoft_tenant_id=FUNDER_DIR, microsoft_object_id=oid,
                    display_name=name))).version, actor=OPERATOR_NAME)
    document = service.active_document()
    assert len(document.principals) == 2
    # Still ONE grant between them: entitlements live at the organisation.
    assert len(document.entitlements) == 1


# --------------------------------------------------------------------------- #
# F. Revocation
# --------------------------------------------------------------------------- #
def test_disabling_a_user_refuses_them_without_touching_the_funder(service, tmp_path):
    _onboard_funder(service)
    service.confirm(service.propose(_set(
        _change(ADD_PRINCIPAL, organisation_id="funder_a",
                microsoft_tenant_id=FUNDER_DIR, microsoft_object_id=JANE_OID,
                display_name="Jane Smith"),
        _change(ADD_PRINCIPAL, organisation_id="funder_a",
                microsoft_tenant_id=FUNDER_DIR, microsoft_object_id=COLLEAGUE_OID,
                display_name="A. Colleague"))).version, actor=OPERATOR_NAME)

    grants_before = service.active_document().sorted_entitlements()

    service.confirm(service.propose(_set(
        _change(DISABLE_PRINCIPAL, microsoft_tenant_id=FUNDER_DIR,
                microsoft_object_id=JANE_OID),
        title="Jane Smith has left Warehouse Funder A")).version,
        actor=OPERATOR_NAME)

    document = service.active_document()
    assert document.principal(FUNDER_DIR, JANE_OID)["status"] == "disabled"
    # The organisation and its grants are untouched — one person left, not a
    # counterparty.
    assert document.sorted_entitlements() == grants_before
    assert document.organisation("funder_a")["enabled"] is True

    dest = tmp_path / "published"
    service.materialise(dest, actor=OPERATOR_NAME)
    principals = load_principal_registry(dest / "config/principals.yaml")

    with pytest.raises(TraktError) as exc:
        principals.resolve(FUNDER_DIR, JANE_OID)
    assert exc.value.code == ErrorCode.PRINCIPAL_DISABLED
    # Her colleague is unaffected.
    assert principals.resolve(FUNDER_DIR, COLLEAGUE_OID).active
    # And the binding is KEPT, so her past activity stays attributable.
    assert principals.get(FUNDER_DIR, JANE_OID) is not None


def test_a_disabled_user_can_be_restored(service):
    _onboard_funder(service)
    add = _change(ADD_PRINCIPAL, organisation_id="funder_a",
                  microsoft_tenant_id=FUNDER_DIR, microsoft_object_id=JANE_OID)
    service.confirm(service.propose(_set(add)).version, actor=OPERATOR_NAME)
    for action in (DISABLE_PRINCIPAL, ENABLE_PRINCIPAL):
        service.confirm(service.propose(_set(
            _change(action, microsoft_tenant_id=FUNDER_DIR,
                    microsoft_object_id=JANE_OID))).version, actor=OPERATOR_NAME)
    assert service.active_document().principal(
        FUNDER_DIR, JANE_OID)["status"] == "active"


def test_revoking_a_grant_leaves_the_organisation_and_its_people(service):
    _onboard_funder(service)
    service.confirm(service.propose(_set(
        _change(ADD_PRINCIPAL, organisation_id="funder_a",
                microsoft_tenant_id=FUNDER_DIR,
                microsoft_object_id=JANE_OID))).version, actor=OPERATOR_NAME)

    service.confirm(service.propose(_set(
        _change(REVOKE_GRANT, organisation_id="funder_a",
                resource="ERE/spv/spv_i"),
        title="Facility repaid")).version, actor=OPERATOR_NAME)

    document = service.active_document()
    assert document.entitlements == []
    assert document.organisation("funder_a") is not None
    assert document.principal(FUNDER_DIR, JANE_OID) is not None


# --------------------------------------------------------------------------- #
# Adding a resource later — and refusing the unsafe one
# --------------------------------------------------------------------------- #
def test_an_unpartitionable_spv_cannot_be_granted(service):
    """SPV II: registered so it can be named and refused BY NAME. Granting it
    would hand over whatever the enclosing book turns out to be."""
    _onboard_funder(service)
    service.confirm(service.propose(_set(
        _change(ADD_RESOURCE, tenant_id="ERE", kind="spv", resource_id="spv_ii",
                display_label="SPV II", unpartitionable=True,
                unpartitionable_reason="The tape carries no SPV attribution."),
        title="Register SPV II")).version, actor=OPERATOR_NAME)

    proposal = service.propose(_set(
        _change(ADD_GRANT, organisation_id="funder_a", resource="ERE/spv/spv_ii",
                capabilities=[SCOPE_RISK_READ]),
        title="Give Funder A access to SPV II"))
    assert proposal.blocked
    assert any("cannot be separated" in p["message"] for p in proposal.problems)

    with pytest.raises(ConfirmationError):
        service.confirm(proposal.version, actor=OPERATOR_NAME)
    assert service.active_document().grant("funder_a", "ERE/spv/spv_ii") is None


def test_the_same_spv_can_be_granted_once_attribution_exists(service):
    """The fix is in the DATA, not in the grant: supply spv_id, then the same
    request succeeds. Still requires confirmation."""
    _onboard_funder(service)
    service.confirm(service.propose(_set(
        _change(ADD_RESOURCE, tenant_id="ERE", kind="spv", resource_id="spv_ii",
                unpartitionable=True,
                unpartitionable_reason="no attribution"))).version,
        actor=OPERATOR_NAME)

    proposal = service.propose(_set(
        _change(UPDATE_RESOURCE, tenant_id="ERE", kind="spv",
                resource_id="spv_ii", spv_id="SPV_II", population="funded",
                unpartitionable=False),
        _change(ADD_GRANT, organisation_id="funder_a", resource="ERE/spv/spv_ii",
                capabilities=[SCOPE_RISK_READ]),
        title="SPV II attribution supplied; grant Funder A risk visibility"))
    assert proposal.valid
    # Confirmation is still required — a valid proposal is not an approved one.
    assert service.active_document().grant("funder_a", "ERE/spv/spv_ii") is None

    service.confirm(proposal.version, actor=OPERATOR_NAME)
    grant = service.active_document().grant("funder_a", "ERE/spv/spv_ii")
    assert grant["capabilities"] == [SCOPE_RISK_READ]


# --------------------------------------------------------------------------- #
# Capability change
# --------------------------------------------------------------------------- #
def test_a_capability_change_is_explicit_and_versioned(service):
    _onboard_funder(service)
    service.confirm(service.propose(_set(
        _change(SET_GRANT_CAPABILITIES, organisation_id="funder_a",
                resource="ERE/spv/spv_i",
                capabilities=[SCOPE_RISK_READ]))).version, actor=OPERATOR_NAME)
    assert service.active_document().grant(
        "funder_a", "ERE/spv/spv_i")["capabilities"] == [SCOPE_RISK_READ]

    widened = service.propose(_set(
        _change(SET_GRANT_CAPABILITIES, organisation_id="funder_a",
                resource="ERE/spv/spv_i",
                capabilities=[SCOPE_RISK_READ, SCOPE_FORECAST_READ]),
        title="Funder A may see forward-looking risk"))
    service.confirm(widened.version, actor=OPERATOR_NAME)

    assert service.active_document().grant(
        "funder_a", "ERE/spv/spv_i")["capabilities"] == sorted(
            [SCOPE_FORECAST_READ, SCOPE_RISK_READ])

    # Versioned: every step kept its own immutable version, and the prior ones
    # are superseded rather than overwritten.
    versions = ConfigPackageStore(service.store).list_versions(LAYER_ACCESS)
    assert [v["status"] for v in versions][-1] == "active"
    assert versions[-2]["status"] == "superseded"
    assert len(versions) >= 4


# --------------------------------------------------------------------------- #
# C. Confirmation is the gate
# --------------------------------------------------------------------------- #
def test_the_occ_agent_cannot_approve_its_own_proposal(service):
    """The agent may recommend. It must never be both proposer and approver."""
    _onboard_funder(service)
    proposal = service.propose(_set(
        _change(SET_GRANT_CAPABILITIES, organisation_id="funder_a",
                resource="ERE/spv/spv_i",
                capabilities=[SCOPE_RISK_READ, SCOPE_FORECAST_READ]),
        by=SOURCE_OCC_AGENT, source=SOURCE_OCC_AGENT))
    assert proposal.valid            # a good proposal — still not self-approvable

    with pytest.raises(ConfirmationError) as exc:
        service.confirm(proposal.version, actor=SOURCE_OCC_AGENT)
    assert "cannot approve" in str(exc.value)

    # A named operator can confirm the very same proposal.
    service.confirm(proposal.version, actor=OPERATOR_NAME)
    assert service.active_document().grant(
        "funder_a", "ERE/spv/spv_i")["capabilities"] == sorted(
            [SCOPE_FORECAST_READ, SCOPE_RISK_READ])


def test_confirmation_requires_a_named_actor(service):
    proposal = _onboard_funder(service, confirm=False)
    for actor in ("", "   ", "system"):
        with pytest.raises(ConfirmationError):
            service.confirm(proposal.version, actor=actor)


def test_confirming_a_changed_proposal_is_refused(service):
    """The fingerprint is quoted back, so approving one proposal can never
    activate a different one."""
    proposal = _onboard_funder(service, confirm=False)
    with pytest.raises(ConfirmationError) as exc:
        service.confirm(proposal.version, actor=OPERATOR_NAME,
                        fingerprint="not-the-one-i-read")
    assert "changed since it was reviewed" in str(exc.value)
    service.confirm(proposal.version, actor=OPERATOR_NAME,
                    fingerprint=proposal.change_set.fingerprint)


def test_a_proposal_can_be_rejected_and_then_not_confirmed(service):
    proposal = _onboard_funder(service, confirm=False)
    service.reject(proposal.version, actor=OPERATOR_NAME, reason="Not yet signed")
    assert service.active_document().organisations == []
    with pytest.raises(ConfirmationError):
        service.confirm(proposal.version, actor=OPERATOR_NAME)


def test_a_confirmed_proposal_cannot_be_confirmed_twice(service):
    proposal = _onboard_funder(service, confirm=False)
    service.confirm(proposal.version, actor=OPERATOR_NAME)
    with pytest.raises(ConfirmationError):
        service.confirm(proposal.version, actor=OPERATOR_NAME)


def test_a_proposal_shows_before_and_after(service):
    proposal = _onboard_funder(service, confirm=False)
    assert proposal.before["organisations"] == []
    assert [o["organisation_id"] for o in proposal.after["organisations"]] \
        == ["funder_a"]
    summary = proposal.summary()
    assert summary["requires_confirmation"] is True
    assert any(a.startswith("ADD_ORGANISATION funder_a") for a in summary["actions"])


# --------------------------------------------------------------------------- #
# Rejection paths — all fail closed
# --------------------------------------------------------------------------- #
def test_a_grant_for_an_unknown_organisation_is_refused(service):
    _onboard_funder(service)
    with pytest.raises(ChangeSetError) as exc:
        service.propose(_set(_change(ADD_GRANT, organisation_id="ghost_co",
                                     resource="ERE/spv/spv_i",
                                     capabilities=[SCOPE_RISK_READ])))
    assert "no such organisation" in str(exc.value)


def test_a_grant_for_an_unknown_resource_is_refused(service):
    _onboard_funder(service)
    with pytest.raises(ChangeSetError) as exc:
        service.propose(_set(_change(ADD_GRANT, organisation_id="funder_a",
                                     resource="ERE/spv/spv_invented",
                                     capabilities=[SCOPE_RISK_READ])))
    assert "no such resource" in str(exc.value)


def test_a_duplicate_grant_is_refused_up_front(service):
    """Widening an existing grant is SET_GRANT_CAPABILITIES, not a second
    ADD_GRANT — so two entries for one pair can never reach the store."""
    _onboard_funder(service)
    with pytest.raises(ChangeSetError) as exc:
        service.propose(_set(
            _change(ADD_GRANT, organisation_id="funder_a",
                    resource="ERE/spv/spv_i", capabilities=[SCOPE_RISK_READ])))
    assert "already has a grant" in str(exc.value)


def test_an_unrecognised_capability_is_caught_by_validation(service):
    _onboard_funder(service)
    service.confirm(service.propose(_set(
        _change(ADD_RESOURCE, tenant_id="ERE", kind="population",
                resource_id="pipeline", population="pipeline"))).version,
        actor=OPERATOR_NAME)
    proposal = service.propose(_set(
        _change(ADD_GRANT, organisation_id="funder_a",
                resource="ERE/population/pipeline",
                capabilities=["database:drop"])))
    assert proposal.blocked
    assert any("capability" in p["message"].lower() for p in proposal.problems)


def test_a_grant_on_a_disabled_organisation_is_blocked(service):
    _onboard_funder(service)
    service.confirm(service.propose(_set(
        _change("DISABLE_ORGANISATION", organisation_id="funder_a"))).version,
        actor=OPERATOR_NAME)
    service.confirm(service.propose(_set(
        _change(ADD_RESOURCE, tenant_id="ERE", kind="population",
                resource_id="funded", population="funded"))).version,
        actor=OPERATOR_NAME)
    proposal = service.propose(_set(
        _change(ADD_GRANT, organisation_id="funder_a",
                resource="ERE/population/funded", capabilities=[SCOPE_RISK_READ])))
    assert proposal.blocked
    assert any("disabled organisation" in p["message"] for p in proposal.problems)


def test_a_principal_in_an_unowned_directory_is_blocked(service):
    """Caught at proposal time rather than as an inexplicable refusal weeks later."""
    _onboard_funder(service)
    proposal = service.propose(_set(
        _change(ADD_PRINCIPAL, organisation_id="funder_a",
                microsoft_tenant_id=ERE_DIR, microsoft_object_id=JANE_OID,
                display_name="Jane Smith")))
    assert proposal.blocked
    assert any("not registered to" in p["message"] for p in proposal.problems)


def test_a_principal_for_an_unknown_organisation_is_refused(service):
    with pytest.raises(ChangeSetError):
        service.propose(_set(
            _change(ADD_PRINCIPAL, organisation_id="nobody",
                    microsoft_tenant_id=FUNDER_DIR,
                    microsoft_object_id=JANE_OID)))


def test_a_malformed_microsoft_id_is_refused(service):
    _onboard_funder(service)
    for bad in ("", "   "):
        with pytest.raises(ChangeSetError):
            service.propose(_set(
                _change(ADD_PRINCIPAL, organisation_id="funder_a",
                        microsoft_tenant_id=FUNDER_DIR, microsoft_object_id=bad)))


def test_an_unknown_action_is_refused(service):
    with pytest.raises(ChangeSetError):
        AccessChange(action="DELETE_EVERYTHING", payload={})


def test_a_duplicate_directory_across_organisations_is_blocked(service):
    _onboard_funder(service)
    proposal = service.propose(_set(
        _change(ADD_ORGANISATION, organisation_id="funder_b",
                organisation_type="warehouse_funder",
                microsoft_tenant_ids=[FUNDER_DIR])))
    assert proposal.blocked
    assert any("more than one organisation" in p["message"]
               for p in proposal.problems)


def test_a_partially_applicable_change_set_applies_nothing(service):
    """Atomicity: the organisation in change 1 must not survive change 2 failing."""
    _onboard_funder(service)
    with pytest.raises(ChangeSetError):
        service.propose(_set(
            _change(ADD_ORGANISATION, organisation_id="funder_b",
                    organisation_type="warehouse_funder",
                    microsoft_tenant_ids=["33333333-3333-3333-3333-333333333333"]),
            _change(ADD_GRANT, organisation_id="funder_b",
                    resource="ERE/spv/does_not_exist",
                    capabilities=[SCOPE_RISK_READ])))
    assert service.active_document().organisation("funder_b") is None


def test_an_empty_change_set_is_refused():
    with pytest.raises(ChangeSetError):
        AccessChangeSet(changes=(), proposed_by=OPERATOR_NAME)


def test_a_grant_with_no_capabilities_is_refused(service):
    _onboard_funder(service)
    service.confirm(service.propose(_set(
        _change(ADD_RESOURCE, tenant_id="ERE", kind="population",
                resource_id="pipeline", population="pipeline"))).version,
        actor=OPERATOR_NAME)
    with pytest.raises(ChangeSetError):
        service.propose(_set(
            _change(ADD_GRANT, organisation_id="funder_a",
                    resource="ERE/population/pipeline", capabilities=[])))


# --------------------------------------------------------------------------- #
# The API surface
# --------------------------------------------------------------------------- #
def test_the_access_routes_are_admin_only(api):
    client, _ = api
    for method, path in (("get", "/ops/admin/access"),
                         ("get", "/ops/admin/access/proposals"),
                         ("post", "/ops/admin/access/proposals")):
        r = getattr(client, method)(path, headers=OPERATOR,
                                    **({"json": {"changes": []}}
                                       if method == "post" else {}))
        assert r.status_code == 403, path


def test_propose_review_confirm_over_the_api(api):
    client, engine = api
    body = {"title": "Warehouse Funder A joins",
            "changes": [
                {"action": ADD_RESOURCE,
                 "payload": {"tenant_id": "ERE", "kind": "spv",
                             "resource_id": "spv_i", "spv_id": "SPV_I",
                             "population": "funded"}},
                {"action": ADD_ORGANISATION,
                 "payload": {"organisation_id": "funder_a",
                             "organisation_type": "warehouse_funder",
                             "microsoft_tenant_ids": [FUNDER_DIR]}},
                {"action": ADD_GRANT,
                 "payload": {"organisation_id": "funder_a",
                             "resource": "ERE/spv/spv_i",
                             "capabilities": FUNDER_CAPS}},
            ]}
    created = client.post("/ops/admin/access/proposals", json=body, headers=ADMIN)
    assert created.status_code == 201, created.text
    proposal = created.json()["proposal"]
    assert proposal["valid"] is True
    version = proposal["version"]

    listed = client.get("/ops/admin/access/proposals", headers=ADMIN).json()
    assert [p["version"] for p in listed["proposals"]] == [version]

    detail = client.get(f"/ops/admin/access/proposals/{version}",
                        headers=ADMIN).json()["proposal"]
    assert detail["before"]["organisations"] == []

    confirmed = client.post(f"/ops/admin/access/proposals/{version}/confirm",
                            json={"fingerprint": proposal["fingerprint"]},
                            headers=ADMIN)
    assert confirmed.status_code == 200, confirmed.text

    overview = client.get("/ops/admin/access", headers=ADMIN).json()
    assert [o["organisation_id"] for o in overview["organisations"]] == ["funder_a"]
    assert len(overview["organisations"][0]["grants"]) == 1
    assert overview["drafts"] == []


def test_a_blocked_proposal_cannot_be_confirmed_over_the_api(api):
    client, _ = api
    body = {"changes": [
        {"action": ADD_ORGANISATION,
         "payload": {"organisation_id": "funder_a",
                     "microsoft_tenant_ids": [FUNDER_DIR]}},
        {"action": ADD_ORGANISATION,
         "payload": {"organisation_id": "funder_b",
                     "microsoft_tenant_ids": [FUNDER_DIR]}},
    ]}
    created = client.post("/ops/admin/access/proposals", json=body, headers=ADMIN)
    version = created.json()["proposal"]["version"]
    assert created.json()["proposal"]["valid"] is False

    refused = client.post(f"/ops/admin/access/proposals/{version}/confirm",
                          json={}, headers=ADMIN)
    assert refused.status_code == 409
    assert refused.json()["detail"]["errorCode"] == "OPS_ACCESS_NOT_CONFIRMABLE"


# --------------------------------------------------------------------------- #
# The architectural boundary
# --------------------------------------------------------------------------- #
def test_the_occ_is_not_in_the_runtime_authorisation_path():
    """Runtime authorisation must never consult the OCC. Checked structurally:
    nothing on the request path imports the admin package, and the admin package
    is not reachable from trakt_core."""
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    hits = subprocess.run(
        ["git", "grep", "-l", "access_admin", "--",
         "mi_agent_api/", "mi_agent/", "trakt_core/", "engine/"],
        capture_output=True, text=True, cwd=str(repo))
    assert hits.stdout.strip() == "", hits.stdout

    proc = subprocess.run(
        ["python", "-c",
         "import sys, trakt_core.entitlement, trakt_core.principal;"
         "assert not [m for m in sys.modules if 'operations_control' in m];"
         "print('ok')"],
        capture_output=True, text=True, cwd=str(repo))
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_stage35_started_no_route_conversion_or_mcp():
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    hits = subprocess.run(
        # A CALL, not a mention: operations_control/api/app.py legitimately
        # names the runtime path in a comment explaining why access
        # administration is separate from it.
        ["git", "grep", "-l", "authorise_resource_access(", "--",
         "mi_agent_api/", "mi_agent/", "engine/", "operations_control/"],
        capture_output=True, text=True, cwd=str(repo))
    assert hits.stdout.strip() == "", hits.stdout
    assert not (repo / "mi_agent_api" / "mcp_server.py").exists()
