#!/usr/bin/env python3
"""Stage 3: organisation × resource × capability.

The three identity axes now compose, and the test that matters most is the one
that proves they stay separate:

    organisation  who is asking      (Stage 1)
    resource      what is the thing  (Stage 2)
    capability    what may they do   (this stage)
    tenant        whose data is it   (unchanged — never derived from the above)

Everything here is dormant unless configured. ``test_the_legacy_path_is_dormant``
and the compatibility block at the bottom hold that line: a deployment with no
entitlement config behaves exactly as it did before, and "no entitlements" is
never read as "no restrictions".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core.context import (
    SCOPE_ARTEFACT_GENERATE,
    SCOPE_FORECAST_READ,
    SCOPE_MI_QUERY,
    SCOPE_PORTFOLIO_READ,
    SCOPE_RISK_READ,
    ExecutionContext,
)
from trakt_core.entitlement import (
    STATUS_ACTIVE,
    STATUS_PROPOSED,
    EntitlementStore,
    Grant,
    authorise_resource_access,
    load_entitlement_store,
    permitted_resources_for,
    resolve_entitlements,
)
from trakt_core.errors import ErrorCode, TraktError
from trakt_core.organisation import (
    ORG_TYPE_ORIGINATOR,
    ORG_TYPE_WAREHOUSE_FUNDER,
    OrganisationRecord,
    OrganisationRegistry,
)
from trakt_core.resource import (
    FIELD_SPV_ID,
    KIND_POPULATION,
    KIND_PORTFOLIO,
    KIND_SPV,
    POPULATION_FUNDED,
    ResourceCatalogue,
    ResourceRecord,
    ResourceRef,
)
from trakt_core.portfolio import FIELD_PORTFOLIO_ID

# --------------------------------------------------------------------------- #
# The warehouse-funder fixture
#
#   ERE
#   ├── SPV I  / funded      (attributable: the tape carries spv_id)
#   ├── SPV II / funded      (NOT attributable — declared unpartitionable)
#   └── pipeline
#
#   funder_a  ->  SPV I only, four read capabilities.
# --------------------------------------------------------------------------- #
ERE_DIR = "11111111-1111-1111-1111-111111111111"
FUNDER_DIR = "22222222-2222-2222-2222-222222222222"

SPV_I = ResourceRef(tenant_id="ERE", kind=KIND_SPV, resource_id="spv_i")
SPV_II = ResourceRef(tenant_id="ERE", kind=KIND_SPV, resource_id="spv_ii")
PIPELINE = ResourceRef(tenant_id="ERE", kind=KIND_POPULATION, resource_id="pipeline")
ERE_TOTAL = ResourceRef(tenant_id="ERE", kind=KIND_PORTFOLIO, resource_id="ere_total")

FUNDER_CAPS = frozenset({SCOPE_PORTFOLIO_READ, SCOPE_MI_QUERY,
                         SCOPE_RISK_READ, SCOPE_FORECAST_READ})


def _organisations() -> OrganisationRegistry:
    return OrganisationRegistry((
        OrganisationRecord(organisation_id="ere", display_name="ERE",
                           organisation_type=ORG_TYPE_ORIGINATOR,
                           microsoft_tenant_ids=frozenset({ERE_DIR})),
        OrganisationRecord(organisation_id="funder_a",
                           display_name="Warehouse Funder A",
                           organisation_type=ORG_TYPE_WAREHOUSE_FUNDER,
                           microsoft_tenant_ids=frozenset({FUNDER_DIR})),
    ), configured=True)


def _catalogue() -> ResourceCatalogue:
    return ResourceCatalogue((
        ResourceRecord(ref=SPV_I, display_label="SPV I", spv_id="SPV_I",
                       source_portfolio_ids=("direct_001",),
                       population=POPULATION_FUNDED),
        ResourceRecord(ref=SPV_II, display_label="SPV II", unpartitionable=True,
                       unpartitionable_reason="The tape carries no SPV attribution."),
        ResourceRecord(ref=PIPELINE, display_label="ERE Pipeline",
                       population="pipeline"),
        ResourceRecord(ref=ERE_TOTAL, display_label="ERE — all books",
                       whole_tenant_book=True),
    ), configured=True)


def _store(*grants, configured: bool = True) -> EntitlementStore:
    if not grants:
        grants = (Grant(organisation_id="funder_a", resource_ref=SPV_I,
                        capabilities=FUNDER_CAPS),)
    return EntitlementStore(grants, configured=configured,
                            organisations=_organisations(), resources=_catalogue())


def _funder_context(store: EntitlementStore = None, **kw) -> ExecutionContext:
    store = store or _store()
    return ExecutionContext(
        tenant_id=kw.pop("tenant_id", "ERE"),
        actor_id="pm@funder-a.example",
        organisation_id="funder_a",
        microsoft_tenant_id=FUNDER_DIR,
        entitlements=store.resolve("funder_a"),
        **kw)


# =========================================================================== #
# D. The warehouse-funder case
# =========================================================================== #
def test_funder_a_may_read_risk_on_spv_i():
    auth = authorise_resource_access(_funder_context(), SPV_I, SCOPE_RISK_READ,
                                     catalogue=_catalogue())
    assert auth.ref == SPV_I
    assert auth.capability == SCOPE_RISK_READ
    assert auth.organisation_id == "funder_a"
    # WHOSE DATA is still ERE's, and it did not come from the grant.
    assert auth.tenant_id == "ERE"
    # …and the authorisation carries the Stage 2 predicate, funded-pinned.
    assert auth.predicate() == {"view": POPULATION_FUNDED,
                                FIELD_PORTFOLIO_ID: ["direct_001"],
                                FIELD_SPV_ID: "SPV_I"}


def test_funder_a_is_refused_spv_ii():
    """No grant names SPV II, so it is refused — and the refusal is the same one
    a nonexistent resource gets, so nothing is learned by asking."""
    with pytest.raises(TraktError) as exc:
        authorise_resource_access(_funder_context(), SPV_II, SCOPE_RISK_READ,
                                  catalogue=_catalogue())
    assert exc.value.code == ErrorCode.RESOURCE_NOT_AUTHORISED
    assert exc.value.http_status == 403


def test_funder_a_is_refused_the_pipeline():
    with pytest.raises(TraktError) as exc:
        authorise_resource_access(_funder_context(), PIPELINE, SCOPE_MI_QUERY,
                                  catalogue=_catalogue())
    assert exc.value.code == ErrorCode.RESOURCE_NOT_AUTHORISED


def test_funder_a_cannot_generate_artefacts_on_a_resource_it_can_read():
    """The capability axis, on its own. SPV I is reachable; producing an investor
    artefact from it is a different permission and was not granted."""
    with pytest.raises(TraktError) as exc:
        authorise_resource_access(_funder_context(), SPV_I, SCOPE_ARTEFACT_GENERATE,
                                  catalogue=_catalogue())
    assert exc.value.code == ErrorCode.CAPABILITY_NOT_GRANTED
    assert exc.value.details["capability"] == SCOPE_ARTEFACT_GENERATE


def test_granting_artefact_generate_explicitly_is_what_permits_it():
    store = _store(Grant(organisation_id="funder_a", resource_ref=SPV_I,
                         capabilities=FUNDER_CAPS | {SCOPE_ARTEFACT_GENERATE}))
    auth = authorise_resource_access(_funder_context(store), SPV_I,
                                     SCOPE_ARTEFACT_GENERATE, catalogue=_catalogue())
    assert auth.capability == SCOPE_ARTEFACT_GENERATE


def test_the_funders_whole_permitted_surface_is_one_resource():
    ctx = _funder_context()
    assert ctx.permitted_resources == frozenset({SPV_I})
    assert permitted_resources_for(ctx) == (SPV_I,)
    assert permitted_resources_for(ctx, capability=SCOPE_RISK_READ) == (SPV_I,)
    assert permitted_resources_for(ctx, capability=SCOPE_ARTEFACT_GENERATE) == ()


# =========================================================================== #
# Entitlement store
# =========================================================================== #
def test_known_organisation_resolves_its_grants():
    resolved = _store().resolve("funder_a")
    assert resolved.organisation_id == "funder_a"
    assert resolved.permitted_resources == frozenset({SPV_I})
    assert resolved.capabilities_for(SPV_I) == FUNDER_CAPS
    assert resolved.allows(SPV_I, SCOPE_RISK_READ)
    assert not resolved.allows(SPV_I, SCOPE_ARTEFACT_GENERATE)


def test_an_organisation_with_no_grants_resolves_to_an_empty_set():
    """A real answer, not a missing one: registered but entitled to nothing."""
    resolved = _store().resolve("ere")
    assert resolved is not None
    assert resolved.is_empty()
    assert resolved.permitted_resources == frozenset()


def test_an_unknown_organisation_resolves_to_nothing_and_is_refused():
    resolved = _store().resolve("nobody_at_all")
    assert resolved.is_empty()
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u",
                           organisation_id="nobody_at_all", entitlements=resolved)
    with pytest.raises(TraktError) as exc:
        authorise_resource_access(ctx, SPV_I, SCOPE_RISK_READ, catalogue=_catalogue())
    assert exc.value.code == ErrorCode.RESOURCE_NOT_AUTHORISED


def test_a_disabled_grant_confers_nothing():
    store = _store(Grant(organisation_id="funder_a", resource_ref=SPV_I,
                         capabilities=FUNDER_CAPS, enabled=False))
    assert store.resolve("funder_a").is_empty()


def test_a_proposed_grant_confers_nothing():
    """The OCC seam. A proposal is not an approval; only `status: active` grants."""
    store = _store(Grant(organisation_id="funder_a", resource_ref=SPV_I,
                         capabilities=FUNDER_CAPS, status=STATUS_PROPOSED,
                         source="occ_agent"))
    assert store.resolve("funder_a").is_empty()
    # …and it is still recorded, so a human can find and approve it.
    assert len(store.grants_for("funder_a", active_only=False)) == 1


def test_a_grant_naming_an_unregistered_organisation_is_rejected():
    with pytest.raises(TraktError) as exc:
        _store(Grant(organisation_id="ghost_co", resource_ref=SPV_I,
                     capabilities=FUNDER_CAPS))
    assert exc.value.code == ErrorCode.INVALID_INPUT
    assert exc.value.details["organisation_id"] == "ghost_co"


def test_a_grant_naming_a_disabled_organisation_is_rejected():
    orgs = OrganisationRegistry((
        OrganisationRecord(organisation_id="funder_a",
                           microsoft_tenant_ids=frozenset({FUNDER_DIR}),
                           enabled=False),), configured=True)
    with pytest.raises(TraktError) as exc:
        EntitlementStore((Grant(organisation_id="funder_a", resource_ref=SPV_I,
                                capabilities=FUNDER_CAPS),),
                         configured=True, organisations=orgs, resources=_catalogue())
    assert exc.value.code == ErrorCode.INVALID_INPUT


def test_a_grant_naming_an_unknown_resource_is_rejected_at_load():
    ghost = ResourceRef(tenant_id="ERE", kind=KIND_SPV, resource_id="spv_ninety")
    with pytest.raises(TraktError) as exc:
        _store(Grant(organisation_id="funder_a", resource_ref=ghost,
                     capabilities=FUNDER_CAPS))
    assert exc.value.code == ErrorCode.INVALID_INPUT
    assert exc.value.details["resource"] == ghost.key


def test_an_unpartitionable_resource_cannot_be_granted_to_anyone():
    """SPV II. Granting a boundary the data cannot enforce would hand over
    whatever the enclosing book turns out to be, so it is refused at LOAD —
    not left to fail confusingly at request time."""
    with pytest.raises(TraktError) as exc:
        _store(Grant(organisation_id="funder_a", resource_ref=SPV_II,
                     capabilities=FUNDER_CAPS))
    assert exc.value.code == ErrorCode.INVALID_INPUT
    assert "cannot be separated" in exc.value.message


def test_a_grant_naming_a_disabled_resource_is_rejected():
    cat = ResourceCatalogue((
        ResourceRecord(ref=SPV_I, spv_id="SPV_I", population=POPULATION_FUNDED,
                       enabled=False),), configured=True)
    with pytest.raises(TraktError) as exc:
        EntitlementStore((Grant(organisation_id="funder_a", resource_ref=SPV_I,
                                capabilities=FUNDER_CAPS),),
                         configured=True, organisations=_organisations(), resources=cat)
    assert exc.value.code == ErrorCode.INVALID_INPUT


def test_an_unrecognised_capability_is_rejected():
    with pytest.raises(TraktError) as exc:
        _store(Grant(organisation_id="funder_a", resource_ref=SPV_I,
                     capabilities=frozenset({"mi:query", "database:drop"})))
    assert exc.value.code == ErrorCode.INVALID_INPUT
    assert exc.value.details["unknown_capabilities"] == ["database:drop"]


def test_an_empty_capability_list_is_rejected():
    with pytest.raises(TraktError) as exc:
        _store(Grant(organisation_id="funder_a", resource_ref=SPV_I,
                     capabilities=frozenset()))
    assert exc.value.code == ErrorCode.INVALID_INPUT


def test_duplicate_grants_are_rejected_rather_than_merged():
    """Unioning two entries for the same pair would produce access neither one
    states. Contradictory config must not widen."""
    with pytest.raises(TraktError) as exc:
        _store(Grant(organisation_id="funder_a", resource_ref=SPV_I,
                     capabilities=frozenset({SCOPE_RISK_READ})),
               Grant(organisation_id="funder_a", resource_ref=SPV_I,
                     capabilities=frozenset({SCOPE_ARTEFACT_GENERATE})))
    assert exc.value.code == ErrorCode.INVALID_INPUT
    assert exc.value.details["resource"] == SPV_I.key


def test_duplicate_detection_is_deterministic():
    a = Grant(organisation_id="funder_a", resource_ref=SPV_I,
              capabilities=frozenset({SCOPE_RISK_READ}))
    b = Grant(organisation_id="funder_a", resource_ref=SPV_I,
              capabilities=frozenset({SCOPE_MI_QUERY}))
    seen = []
    for order in ((a, b), (b, a)):
        with pytest.raises(TraktError) as exc:
            _store(*order)
        seen.append((exc.value.code, exc.value.details))
    assert seen[0] == seen[1]


def test_the_same_resource_may_be_granted_to_two_organisations():
    store = _store(Grant(organisation_id="funder_a", resource_ref=SPV_I,
                         capabilities=frozenset({SCOPE_RISK_READ})),
                   Grant(organisation_id="ere", resource_ref=SPV_I,
                         capabilities=FUNDER_CAPS))
    assert store.resolve("funder_a").capabilities_for(SPV_I) == {SCOPE_RISK_READ}
    assert store.resolve("ere").capabilities_for(SPV_I) == FUNDER_CAPS


# =========================================================================== #
# ExecutionContext
# =========================================================================== #
def test_resolved_entitlements_are_immutable():
    ctx = _funder_context()
    with pytest.raises(Exception):
        ctx.entitlements = None  # type: ignore[misc]
    with pytest.raises(Exception):
        ctx.entitlements.entries = ()  # type: ignore[misc]
    # The permitted set is a frozenset, so a holder cannot add to it either.
    with pytest.raises(AttributeError):
        ctx.permitted_resources.add(SPV_II)  # type: ignore[attr-defined]


def test_capability_sets_are_frozen():
    ctx = _funder_context()
    with pytest.raises(AttributeError):
        ctx.capabilities_for(SPV_I).add(SCOPE_ARTEFACT_GENERATE)  # type: ignore


def test_entitlement_version_is_deterministic_and_content_addressed():
    first = _store().resolve("funder_a").version
    second = _store().resolve("funder_a").version
    assert first == second and first.startswith("ent_")

    narrower = _store(Grant(organisation_id="funder_a", resource_ref=SPV_I,
                            capabilities=frozenset({SCOPE_RISK_READ})))
    assert narrower.resolve("funder_a").version != first


def test_entitlement_version_is_per_organisation():
    """Another organisation's grants changing must not invalidate this one's —
    the version is what a later cache key will hang off."""
    store_one = _store(Grant(organisation_id="funder_a", resource_ref=SPV_I,
                             capabilities=FUNDER_CAPS))
    store_two = _store(Grant(organisation_id="funder_a", resource_ref=SPV_I,
                             capabilities=FUNDER_CAPS),
                       Grant(organisation_id="ere", resource_ref=ERE_TOTAL,
                             capabilities=frozenset({SCOPE_MI_QUERY})))
    assert store_one.resolve("funder_a").version == store_two.resolve("funder_a").version


def test_existing_contexts_are_backward_compatible():
    ctx = ExecutionContext(tenant_id="ERE", actor_id="user-1")
    assert ctx.entitlements is None
    assert ctx.entitlement_mode is False
    assert ctx.permitted_resources == frozenset()
    assert ctx.entitlement_version is None
    assert ctx.capabilities_for(SPV_I) == frozenset()


def test_the_data_tenant_is_independent_of_organisation_and_grants():
    """THE Stage 3 invariant. A funder's grant over ERE resources does not make
    the funder the tenant, and does not move the tenant."""
    ctx = _funder_context()
    assert ctx.tenant_id == "ERE"
    assert ctx.organisation_id == "funder_a"
    assert ctx.tenant_id != ctx.organisation_id

    # The same grants against a deployment serving a different tenant do not
    # move tenant_id either — and the resource is then refused as cross-tenant.
    other = _funder_context(tenant_id="OTHERCO")
    assert other.tenant_id == "OTHERCO"
    with pytest.raises(TraktError) as exc:
        authorise_resource_access(other, SPV_I, SCOPE_RISK_READ, catalogue=_catalogue())
    assert exc.value.code == ErrorCode.TENANT_MISMATCH


def test_entitlement_version_reaches_the_audit_projection():
    event = _funder_context().to_audit_dict()
    assert event["entitlement_version"].startswith("ent_")
    assert event["organisation_id"] == "funder_a"
    assert event["tenant_id"] == "ERE"
    # A fingerprint, not the grants themselves.
    assert SPV_I.key not in str(event)


# =========================================================================== #
# E. Fail-closed authorisation
# =========================================================================== #
def test_a_context_with_no_entitlements_is_refused_not_waved_through():
    """Absence of configuration is not absence of restriction."""
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u")
    with pytest.raises(TraktError) as exc:
        authorise_resource_access(ctx, SPV_I, SCOPE_RISK_READ, catalogue=_catalogue())
    assert exc.value.code == ErrorCode.ENTITLEMENTS_NOT_RESOLVED


def test_a_missing_scope_refuses_before_the_entitlement_lookup():
    """Two axes, both enforced: the channel/role gate runs first, so a caller who
    never held the capability cannot even probe the grant table."""
    ctx = _funder_context(scopes=frozenset({SCOPE_PORTFOLIO_READ}))
    with pytest.raises(TraktError) as exc:
        authorise_resource_access(ctx, SPV_I, SCOPE_RISK_READ, catalogue=_catalogue())
    assert exc.value.code == ErrorCode.SCOPE_MISSING


def test_an_ungranted_resource_and_a_nonexistent_one_are_indistinguishable():
    """Membership is checked BEFORE the catalogue, so error codes cannot be used
    to enumerate another organisation's resources."""
    ctx = _funder_context()
    real_but_ungranted = PIPELINE
    pure_fiction = ResourceRef(tenant_id="ERE", kind=KIND_SPV,
                               resource_id="spv_does_not_exist")
    codes = set()
    for ref in (real_but_ungranted, pure_fiction):
        with pytest.raises(TraktError) as exc:
            authorise_resource_access(ctx, ref, SCOPE_MI_QUERY, catalogue=_catalogue())
        codes.add(exc.value.code)
    assert codes == {ErrorCode.RESOURCE_NOT_AUTHORISED}


def test_a_malformed_resource_string_is_refused_not_coerced():
    ctx = _funder_context()
    for bad in ("spv_i", "ERE/spv", "", "ERE/spv/spv_i/extra", "../../etc"):
        with pytest.raises(TraktError) as exc:
            authorise_resource_access(ctx, bad, SCOPE_RISK_READ, catalogue=_catalogue())
        assert exc.value.code == ErrorCode.INVALID_INPUT, bad


def test_a_caller_supplied_string_is_checked_against_the_closed_set():
    """The LLM-cannot-widen property. A model may pass any string it likes; it
    is parsed into a ref and tested for membership, never used as a lookup."""
    ctx = _funder_context()
    auth = authorise_resource_access(ctx, "ERE/spv/spv_i", SCOPE_RISK_READ,
                                     catalogue=_catalogue())
    assert auth.ref == SPV_I
    for attempt in ("ERE/spv/spv_ii", "ERE/population/pipeline",
                    "ERE/portfolio/ere_total", "OTHERCO/spv/spv_i"):
        with pytest.raises(TraktError) as exc:
            authorise_resource_access(ctx, attempt, SCOPE_MI_QUERY,
                                      catalogue=_catalogue())
        assert exc.value.code == ErrorCode.RESOURCE_NOT_AUTHORISED, attempt


def test_no_authorisation_path_ever_widens_to_total():
    """Every refusal raises. There is no input to this helper that yields a
    permissive default or a whole-book scope."""
    ctx = _funder_context()
    for ref in (SPV_II, PIPELINE, ERE_TOTAL):
        with pytest.raises(TraktError):
            authorise_resource_access(ctx, ref, SCOPE_MI_QUERY, catalogue=_catalogue())
    # The one resource that IS granted still narrows.
    auth = authorise_resource_access(ctx, SPV_I, SCOPE_MI_QUERY, catalogue=_catalogue())
    assert auth.predicate() != {}


# =========================================================================== #
# Loading
# =========================================================================== #
_CONFIG = """
entitlements:
  - organisation_id: funder_a
    resource: ERE/spv/spv_i
    capabilities: [portfolio:read, mi:query, risk:read, forecast:read]
    status: active
    approved_by: a.person
"""


def _write(tmp_path, text) -> Path:
    path = tmp_path / "entitlements.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_absent_config_leaves_the_model_dormant(tmp_path):
    store = load_entitlement_store(tmp_path / "nope.yaml")
    assert store.configured is False
    assert store.resolve("funder_a") is None


def test_a_config_loads_and_resolves(tmp_path):
    store = load_entitlement_store(_write(tmp_path, _CONFIG),
                                   organisations=_organisations(),
                                   resources=_catalogue())
    assert store.configured is True and len(store) == 1
    assert store.resolve("funder_a").capabilities_for(SPV_I) == FUNDER_CAPS


def test_loading_is_deterministic(tmp_path):
    path = _write(tmp_path, _CONFIG)
    kw = {"organisations": _organisations(), "resources": _catalogue()}
    first = load_entitlement_store(path, **kw)
    second = load_entitlement_store(path, **kw)
    assert [g.to_dict() for g in first.grants()] == [g.to_dict() for g in second.grants()]
    assert first.version == second.version


def test_a_corrupt_config_fails_closed(tmp_path):
    store = load_entitlement_store(_write(tmp_path, "entitlements: [ not: valid: ["),
                                   organisations=_organisations(),
                                   resources=_catalogue())
    assert store.configured is True and len(store) == 0
    assert store.resolve("funder_a").is_empty()


def test_one_invalid_grant_rejects_the_whole_file(tmp_path):
    """Partial application would mean a typo silently changes who can see what."""
    text = _CONFIG + """
  - organisation_id: funder_a
    resource: ERE/spv/spv_ii
    capabilities: [risk:read]
"""
    store = load_entitlement_store(_write(tmp_path, text),
                                   organisations=_organisations(),
                                   resources=_catalogue())
    assert len(store) == 0
    assert store.resolve("funder_a").is_empty()


def test_the_shipped_example_config_is_loadable(tmp_path):
    source = (_REPO_ROOT / "config" / "entitlements.example.yaml").read_text(
        encoding="utf-8")
    store = load_entitlement_store(_write(tmp_path, source),
                                   organisations=_organisations(),
                                   resources=_catalogue())
    assert store.configured is True
    resolved = store.resolve("funder_a")
    assert resolved.permitted_resources == frozenset({SPV_I})
    assert resolved.capabilities_for(SPV_I) == FUNDER_CAPS
    assert SCOPE_ARTEFACT_GENERATE not in resolved.capabilities_for(SPV_I)


def test_the_repository_ships_no_live_entitlement_config():
    assert not (_REPO_ROOT / "config" / "entitlements.yaml").exists()


def test_resolve_entitlements_returns_nothing_without_an_organisation():
    assert resolve_entitlements(None, store=_store()) is None
    assert resolve_entitlements("  ", store=_store()) is None


# =========================================================================== #
# F. Compatibility
# =========================================================================== #
def test_the_capability_vocabulary_is_the_existing_scope_vocabulary():
    """No second vocabulary: a grant uses exactly the name the capability gates
    on, so there is nothing to translate and nothing to drift."""
    from trakt_core.context import DEFAULT_MI_SCOPES, KNOWN_CAPABILITIES

    assert KNOWN_CAPABILITIES == DEFAULT_MI_SCOPES
    for scope in (SCOPE_PORTFOLIO_READ, SCOPE_MI_QUERY, SCOPE_ARTEFACT_GENERATE,
                  SCOPE_RISK_READ, SCOPE_FORECAST_READ):
        assert scope in KNOWN_CAPABILITIES


def test_the_new_scopes_are_granted_to_existing_users():
    """risk:read / forecast:read are inert today (no route checks them) and are
    in the default set so those users do not lose access when routes start to."""
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u")
    assert ctx.has_scope(SCOPE_RISK_READ)
    assert ctx.has_scope(SCOPE_FORECAST_READ)


def test_authorise_portfolio_access_is_untouched():
    from trakt_core.tenancy import (
        TenantRecord, TenantRegistry, authorise_portfolio_access)

    registry = TenantRegistry(
        {"ERE": TenantRecord(tenant_id="ERE", default_portfolio_id="ERE",
                             portfolio_ids=frozenset({"ERE", "direct_001"}))},
        configured=True)
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u")
    assert authorise_portfolio_access(ctx, "direct_001", registry=registry).selector \
        == "direct_001"
    with pytest.raises(TraktError) as exc:
        authorise_portfolio_access(ctx, "acquired_777", registry=registry)
    assert exc.value.code == ErrorCode.PORTFOLIO_NOT_AUTHORISED


def test_the_legacy_path_is_dormant(monkeypatch, tmp_path):
    """A deployment with no entitlement config: the Copilot identity path builds
    exactly the Stage 1 context, entitlements unresolved."""
    monkeypatch.setenv("TRAKT_ENTITLEMENTS_CONFIG", str(tmp_path / "absent.yaml"))
    from mi_agent_api import identity
    from mi_agent_api.copilot_auth import CopilotPrincipal

    ctx = identity.context_from_copilot_principal(
        CopilotPrincipal(subject="oid-1", tenant_id=ERE_DIR, scopes=["Trakt.Copilot"]),
        tenant_id="ERE",
        organisation_registry=OrganisationRegistry((), configured=False))
    assert ctx.tenant_id == "ERE"
    assert ctx.organisation_id is None
    assert ctx.entitlements is None
    assert ctx.entitlement_mode is False


def test_the_identity_path_resolves_and_freezes_grants_when_configured():
    """validated directory -> organisation -> entitlements -> immutable context."""
    from mi_agent_api import identity
    from mi_agent_api.copilot_auth import CopilotPrincipal

    ctx = identity.context_from_copilot_principal(
        CopilotPrincipal(subject="pm-1", tenant_id=FUNDER_DIR,
                         scopes=["Trakt.Copilot"]),
        tenant_id="ERE",
        organisation_registry=_organisations(),
        entitlement_store=_store())
    assert ctx.organisation_id == "funder_a"
    assert ctx.microsoft_tenant_id == FUNDER_DIR
    assert ctx.tenant_id == "ERE"                       # still deployment config
    assert ctx.permitted_resources == frozenset({SPV_I})
    assert ctx.entitlement_mode is True

    auth = authorise_resource_access(ctx, SPV_I, SCOPE_RISK_READ,
                                     catalogue=_catalogue())
    assert auth.entitlement_version == ctx.entitlement_version


def test_entitlement_module_imports_without_a_web_framework_or_pandas():
    import os
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, trakt_core.entitlement as e;"
         "assert 'fastapi' not in sys.modules, 'entitlement pulled in fastapi';"
         "assert 'starlette' not in sys.modules, 'entitlement pulled in starlette';"
         "assert 'pandas' not in sys.modules, 'entitlement pulled in pandas';"
         "s = e.load_entitlement_store();"
         "assert s.resolve('anyone') is None;"
         "print('ok')"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
        env={**os.environ, "TRAKT_RUNTIME_MODE": "test", "PYTHONPATH": str(_REPO_ROOT)})
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_stage3_did_not_start_route_enforcement_or_mcp():
    """Stage 3 constructs the context and offers a helper. Nothing consumes it."""
    import subprocess

    hits = subprocess.run(
        ["git", "grep", "-l", "authorise_resource_access", "--",
         "mi_agent_api/", "mi_agent/", "engine/", "operations_control/"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT))
    assert hits.stdout.strip() == "", (
        "a route or engine already calls authorise_resource_access: "
        f"{hits.stdout}")
    assert not (_REPO_ROOT / "mi_agent_api" / "mcp_server.py").exists()
