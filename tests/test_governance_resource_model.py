#!/usr/bin/env python3
"""Stage 2: the permissionable resource model.

Two properties are under test throughout, and they are the reason the module
exists rather than reusing the presentation scope that was already there:

    a resource expands to a DETERMINISTIC predicate over existing fields
    a resource that cannot be partitioned REFUSES — it never widens

The second is the one that matters. ``mi_agent.portfolio_scope.apply_scope``
returns every row when the provenance column is missing, and
``trakt_core.portfolio.resolve_scope`` widens an unrecognised context to Total.
Both are fine for a display lens. Either one, reused as an entitlement filter,
is a cross-funder disclosure — so the tests below pin the opposite behaviour at
every equivalent point.

Stage 2 grants nobody anything; ``test_stage2_grants_nobody_anything`` asserts
that no entitlement machinery arrived with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core.errors import ErrorCode, TraktError
from trakt_core.portfolio import (
    CONTEXT_KIND_PORTFOLIO,
    FIELD_PORTFOLIO_ID,
    PortfolioRecord,
)
from trakt_core.resource import (
    ATTRIBUTION_PROVENANCE,
    ATTRIBUTION_SPV_FIELD,
    ATTRIBUTION_UNPARTITIONABLE,
    ATTRIBUTION_VIEW,
    ATTRIBUTION_WHOLE_BOOK,
    FIELD_SPV_ID,
    KIND_FACILITY,
    KIND_POPULATION,
    KIND_PORTFOLIO,
    KIND_SOURCE_PORTFOLIO,
    KIND_SPV,
    POPULATION_FUNDED,
    POPULATION_PIPELINE,
    POPULATIONS,
    ResourceCatalogue,
    ResourceRecord,
    ResourceRef,
    check_attribution,
    load_resource_catalogue,
    resources_from_portfolio_records,
)

FULL_FIELDS = {FIELD_PORTFOLIO_ID, FIELD_SPV_ID, "current_outstanding_balance"}
PROVENANCE_ONLY = {FIELD_PORTFOLIO_ID, "current_outstanding_balance"}
NO_PROVENANCE = {"current_outstanding_balance"}


def _ref(kind=KIND_SPV, resource_id="spv_i", tenant="ERE") -> ResourceRef:
    return ResourceRef(tenant_id=tenant, kind=kind, resource_id=resource_id)


SPV_I = ResourceRecord(
    ref=_ref(KIND_SPV, "spv_i"), display_label="SPV I", spv_id="SPV_I",
    source_portfolio_ids=("direct_001",), population=POPULATION_FUNDED)
SPV_II = ResourceRecord(
    ref=_ref(KIND_SPV, "spv_ii"), display_label="SPV II", unpartitionable=True,
    unpartitionable_reason="The tape carries no SPV attribution.")
PIPELINE = ResourceRecord(
    ref=_ref(KIND_POPULATION, "pipeline"), display_label="ERE Pipeline",
    population=POPULATION_PIPELINE)
DIRECT_001 = ResourceRecord(
    ref=_ref(KIND_SOURCE_PORTFOLIO, "direct_001"),
    display_label="ERE Direct", source_portfolio_ids=("direct_001",))
ERE_TOTAL = ResourceRecord(
    ref=_ref(KIND_PORTFOLIO, "ere_total"), display_label="ERE — all books",
    whole_tenant_book=True)


def _catalogue(*records) -> ResourceCatalogue:
    return ResourceCatalogue(
        records or (SPV_I, SPV_II, PIPELINE, DIRECT_001, ERE_TOTAL), configured=True)


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
def test_refs_are_deterministic_and_hashable():
    """A later grant holds a SET of refs, so equality and hashing have to be
    normalisation-proof rather than something every call site remembers."""
    a = ResourceRef(tenant_id="ERE", kind="spv", resource_id="spv_i")
    b = ResourceRef(tenant_id="ERE", kind="SPV", resource_id="  Spv_I ")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    assert a.key == "ERE/spv/spv_i"


def test_the_tenant_is_part_of_the_identity():
    """Two tenants may each own an `spv_i`; nothing may confuse them."""
    mine = ResourceRef(tenant_id="ERE", kind=KIND_SPV, resource_id="spv_i")
    theirs = ResourceRef(tenant_id="OTHERCO", kind=KIND_SPV, resource_id="spv_i")
    assert mine != theirs
    assert len({mine, theirs}) == 2
    with pytest.raises(TraktError) as exc:
        _catalogue().resolve(theirs)
    assert exc.value.code == ErrorCode.RESOURCE_NOT_REGISTERED


def test_ref_round_trips_through_its_canonical_string():
    ref = _ref(KIND_FACILITY, "warehouse_a")
    assert ResourceRef.parse(ref.key) == ref


def test_ref_is_immutable():
    ref = _ref()
    with pytest.raises(Exception):
        ref.resource_id = "spv_ii"  # type: ignore[misc]


def test_resource_kinds_are_validated():
    with pytest.raises(ValueError):
        ResourceRef(tenant_id="ERE", kind="spaceship", resource_id="x")
    for kind in (KIND_PORTFOLIO, KIND_SOURCE_PORTFOLIO, KIND_SPV, KIND_FACILITY,
                 KIND_POPULATION):
        assert ResourceRef(tenant_id="ERE", kind=kind, resource_id="x").kind == kind


def test_identifiers_cannot_traverse_a_storage_path():
    """Same rule tenancy applies to a portfolio selector: a resource id is a
    plain slug, so a catalogue key can never become a path segment."""
    for bad in ("../../etc/passwd", "a/b", "", "   "):
        with pytest.raises(ValueError):
            ResourceRef(tenant_id="ERE", kind=KIND_SPV, resource_id=bad)
    with pytest.raises(ValueError):
        ResourceRecord(ref=_ref(), source_portfolio_ids=("../secrets",))


def test_population_values_are_validated():
    with pytest.raises(ValueError):
        ResourceRecord(ref=_ref(), spv_id="X", population="everything")


def test_populations_match_the_workspace_views():
    """Drift guard: the populations here are duplicated as literals so
    trakt_core stays free of the API package. If workspace.VIEWS changes, this
    fails rather than the two silently disagreeing."""
    from mi_agent_api.workspace import VIEWS
    assert set(POPULATIONS) == set(VIEWS)


# --------------------------------------------------------------------------- #
# Catalogue construction
# --------------------------------------------------------------------------- #
def test_duplicate_resource_ids_are_rejected():
    clash = ResourceRecord(ref=_ref(KIND_SPV, "spv_i"), spv_id="SOMETHING_ELSE")
    with pytest.raises(TraktError) as exc:
        ResourceCatalogue((SPV_I, clash), configured=True)
    assert exc.value.code == ErrorCode.INVALID_INPUT
    assert exc.value.details["resource"] == "ERE/spv/spv_i"


def test_the_same_id_under_a_different_kind_is_not_a_duplicate():
    both = _catalogue(
        ResourceRecord(ref=_ref(KIND_SPV, "alpha"), spv_id="ALPHA"),
        ResourceRecord(ref=_ref(KIND_FACILITY, "alpha"), spv_id="ALPHA"))
    assert len(both) == 2


def test_a_resource_with_no_constraint_is_rejected():
    """THE accidental-widening guard. A record that constrains nothing would
    silently mean the whole tenant, so it cannot be loaded at all."""
    with pytest.raises(TraktError) as exc:
        ResourceCatalogue((ResourceRecord(ref=_ref(KIND_SPV, "empty")),),
                          configured=True)
    assert exc.value.code == ErrorCode.INVALID_INPUT
    assert "whole tenant book" in exc.value.message


def test_whole_book_access_must_be_written_down():
    """The same record becomes legal the moment someone declares the intent."""
    ok = _catalogue(ResourceRecord(ref=_ref(KIND_PORTFOLIO, "everything"),
                                   whole_tenant_book=True))
    assert ok.resolve(_ref(KIND_PORTFOLIO, "everything")).predicate() == {}


def test_catalogue_ordering_is_stable():
    forward = ResourceCatalogue((SPV_I, PIPELINE, DIRECT_001), configured=True)
    reverse = ResourceCatalogue((DIRECT_001, PIPELINE, SPV_I), configured=True)
    assert [r.ref.key for r in forward.records()] == [r.ref.key for r in reverse.records()]


def test_attribution_is_derived_not_declared():
    """An operator cannot mis-declare a resource into looking narrower than the
    constraints it actually carries."""
    assert SPV_I.attribution == ATTRIBUTION_SPV_FIELD
    assert DIRECT_001.attribution == ATTRIBUTION_PROVENANCE
    assert PIPELINE.attribution == ATTRIBUTION_VIEW
    assert ERE_TOTAL.attribution == ATTRIBUTION_WHOLE_BOOK
    assert SPV_II.attribution == ATTRIBUTION_UNPARTITIONABLE


# --------------------------------------------------------------------------- #
# Lookup fails closed
# --------------------------------------------------------------------------- #
def test_unknown_resource_fails_closed():
    with pytest.raises(TraktError) as exc:
        _catalogue().resolve(_ref(KIND_SPV, "spv_xii"))
    assert exc.value.code == ErrorCode.RESOURCE_NOT_REGISTERED
    assert exc.value.http_status == 403


def test_disabled_resource_is_rejected():
    off = ResourceRecord(ref=_ref(KIND_SPV, "spv_i"), spv_id="SPV_I", enabled=False)
    with pytest.raises(TraktError) as exc:
        _catalogue(off).resolve(_ref(KIND_SPV, "spv_i"))
    assert exc.value.code == ErrorCode.RESOURCE_DISABLED


def test_disabled_resources_are_excluded_from_enumeration():
    off = ResourceRecord(ref=_ref(KIND_SPV, "spv_i"), spv_id="SPV_I", enabled=False)
    cat = _catalogue(off, PIPELINE)
    assert [r.key for r in cat.refs()] == ["ERE/population/pipeline"]
    assert len(cat.refs(enabled_only=False)) == 2


def test_an_empty_catalogue_resolves_nothing():
    """"No catalogue" must mean "no resources", never "all of them"."""
    empty = ResourceCatalogue((), configured=False)
    with pytest.raises(TraktError) as exc:
        empty.resolve(_ref(KIND_SPV, "spv_i"))
    assert exc.value.code == ErrorCode.RESOURCE_NOT_REGISTERED


# --------------------------------------------------------------------------- #
# Expansion
# --------------------------------------------------------------------------- #
def test_source_portfolio_resource_expands_to_its_provenance_filter():
    resolved = _catalogue().resolve(_ref(KIND_SOURCE_PORTFOLIO, "direct_001"))
    assert resolved.predicate() == {FIELD_PORTFOLIO_ID: ["direct_001"]}
    assert resolved.owning_tenant_id == "ERE"
    assert resolved.required_fields == {FIELD_PORTFOLIO_ID}
    assert check_attribution(resolved, PROVENANCE_ONLY).ok


def test_spv_resource_expands_when_explicit_attribution_exists():
    resolved = _catalogue().resolve(_ref(KIND_SPV, "spv_i"))
    assert resolved.predicate() == {
        "view": POPULATION_FUNDED,
        FIELD_PORTFOLIO_ID: ["direct_001"],
        FIELD_SPV_ID: "SPV_I",
    }
    assert resolved.attribution == ATTRIBUTION_SPV_FIELD
    assert check_attribution(resolved, FULL_FIELDS).ok


def test_funded_only_resource_retains_the_funded_constraint():
    """The constraint that keeps a future funder out of the unfunded pipeline."""
    resolved = _catalogue().resolve(_ref(KIND_SPV, "spv_i"))
    assert resolved.population == POPULATION_FUNDED
    assert resolved.predicate()["view"] == POPULATION_FUNDED


def test_pipeline_resource_retains_the_pipeline_constraint():
    resolved = _catalogue().resolve(_ref(KIND_POPULATION, "pipeline"))
    assert resolved.predicate() == {"view": POPULATION_PIPELINE}
    # A view-only resource needs no row-level column to be enforceable: the
    # population picks which frame is loaded.
    assert resolved.required_fields == frozenset()
    assert check_attribution(resolved, NO_PROVENANCE).ok


def test_the_population_constraint_cannot_be_changed_after_resolution():
    resolved = _catalogue().resolve(_ref(KIND_SPV, "spv_i"))
    with pytest.raises(Exception):
        resolved.population = POPULATION_PIPELINE  # type: ignore[misc]


def test_expansion_is_deterministic():
    first = _catalogue().resolve(_ref(KIND_SPV, "spv_i")).predicate()
    second = _catalogue().resolve(_ref(KIND_SPV, "spv_i")).predicate()
    assert first == second


# --------------------------------------------------------------------------- #
# Fail-closed: the behaviours that must NOT match the presentation lens
# --------------------------------------------------------------------------- #
def test_an_unknown_resource_never_widens():
    """resolve_scope() answers an unrecognised context with Total. This refuses.
    Same input shape, opposite — and correct — outcome for a security boundary."""
    from trakt_core.portfolio import PortfolioRegistry, resolve_scope

    registry = PortfolioRegistry([PortfolioRecord(portfolio_id="direct_001",
                                                  portfolio_type="direct")])
    lens = resolve_scope(registry, "no_such_thing")
    assert lens.is_total and lens.fell_back_to_total          # the display lens

    with pytest.raises(TraktError) as exc:                     # the resource model
        _catalogue().resolve(_ref(KIND_SPV, "no_such_thing"))
    assert exc.value.code == ErrorCode.RESOURCE_NOT_REGISTERED


def test_missing_provenance_is_a_refusal_not_whole_book_access():
    """apply_scope() returns the WHOLE frame when source_portfolio_id is absent.
    check_attribution refuses instead — a dataset that cannot express the
    boundary is a reason to serve nothing, never a reason to serve everything."""
    resolved = _catalogue().resolve(_ref(KIND_SOURCE_PORTFOLIO, "direct_001"))
    with pytest.raises(TraktError) as exc:
        check_attribution(resolved, NO_PROVENANCE)
    assert exc.value.code == ErrorCode.RESOURCE_NOT_PARTITIONABLE
    assert exc.value.details["missing_fields"] == [FIELD_PORTFOLIO_ID]


def test_missing_spv_attribution_is_a_refusal():
    resolved = _catalogue().resolve(_ref(KIND_SPV, "spv_i"))
    with pytest.raises(TraktError) as exc:
        check_attribution(resolved, PROVENANCE_ONLY)   # has provenance, no spv_id
    assert exc.value.code == ErrorCode.RESOURCE_NOT_PARTITIONABLE
    assert exc.value.details["missing_fields"] == [FIELD_SPV_ID]


def test_an_unpartitionable_spv_refuses_rather_than_mapping_to_its_portfolio():
    """SPV II: registered so it can be named and refused BY NAME, and expanding
    it never falls back to the enclosing book — which would hand a future SPV II
    funder the whole portfolio."""
    resolved = _catalogue().resolve(_ref(KIND_SPV, "spv_ii"))
    assert resolved.partitionable is False
    assert resolved.source_portfolio_ids == ()
    for attempt in (lambda: resolved.predicate(),
                    lambda: check_attribution(resolved, FULL_FIELDS)):
        with pytest.raises(TraktError) as exc:
            attempt()
        assert exc.value.code == ErrorCode.RESOURCE_NOT_PARTITIONABLE
    assert "no SPV attribution" in (resolved.unpartitionable_reason or "")


def test_a_resource_naming_a_nonexistent_source_portfolio_still_narrows():
    """A stale id must produce an EMPTY population, not an unfiltered one. The
    predicate names the missing book; it never degrades to no filter."""
    stale = ResourceRecord(ref=_ref(KIND_SOURCE_PORTFOLIO, "direct_999"),
                           source_portfolio_ids=("direct_999",))
    resolved = _catalogue(stale).resolve(_ref(KIND_SOURCE_PORTFOLIO, "direct_999"))
    assert resolved.predicate() == {FIELD_PORTFOLIO_ID: ["direct_999"]}
    assert resolved.predicate() != {}


def test_a_hand_built_unconstrained_resource_still_refuses_to_expand():
    """Defence in depth. The catalogue rejects this shape at load, so reaching
    predicate() with it means someone bypassed the catalogue."""
    from trakt_core.resource import ResolvedResource

    rogue = ResolvedResource(ref=_ref(KIND_SPV, "rogue"), owning_tenant_id="ERE",
                             attribution=ATTRIBUTION_PROVENANCE)
    with pytest.raises(TraktError) as exc:
        rogue.predicate()
    assert exc.value.code == ErrorCode.RESOURCE_NOT_PARTITIONABLE


def test_no_predicate_is_ever_empty_except_an_explicit_whole_book():
    for ref in _catalogue().refs():
        resolved = _catalogue().resolve(ref)
        if not resolved.partitionable:
            continue
        if resolved.whole_tenant_book:
            assert resolved.predicate() == {}
        else:
            assert resolved.predicate(), f"{ref.key} expanded to an empty filter"


# --------------------------------------------------------------------------- #
# The bridge to the existing portfolio model
# --------------------------------------------------------------------------- #
def test_portfolio_scope_carries_only_the_book_narrowing():
    resolved = _catalogue().resolve(_ref(KIND_SPV, "spv_i"))
    scope = resolved.to_portfolio_scope()
    assert scope is not None
    assert scope.context_kind == CONTEXT_KIND_PORTFOLIO
    assert scope.portfolio_ids == ("direct_001",)
    assert scope.is_total is False
    assert scope.filters == {FIELD_PORTFOLIO_ID: ["direct_001"]}


def test_a_resource_with_no_book_narrowing_yields_no_scope_not_a_total_one():
    """Returning Total here would be exactly the fail-open bug: `None` means
    "no book narrowing available", and the caller must still apply the rest of
    the predicate."""
    resolved = _catalogue().resolve(_ref(KIND_POPULATION, "pipeline"))
    assert resolved.to_portfolio_scope() is None


def test_source_portfolio_resources_can_be_derived_from_the_portfolio_registry():
    """Every source_portfolio_id is already a mandatory-on-every-row boundary, so
    it is a resource without anyone writing it into a config file."""
    records = [PortfolioRecord(portfolio_id="direct_001", portfolio_type="direct",
                               label="ERE Direct"),
               PortfolioRecord(portfolio_id="acquired_001", portfolio_type="acquired")]
    derived = resources_from_portfolio_records("ERE", records,
                                               population=POPULATION_FUNDED)
    cat = ResourceCatalogue(derived, configured=True)
    assert [r.key for r in cat.refs()] == ["ERE/source_portfolio/acquired_001",
                                           "ERE/source_portfolio/direct_001"]
    resolved = cat.resolve(_ref(KIND_SOURCE_PORTFOLIO, "direct_001"))
    assert resolved.predicate() == {"view": POPULATION_FUNDED,
                                    FIELD_PORTFOLIO_ID: ["direct_001"]}
    assert resolved.display_label == "ERE Direct"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def test_absent_config_yields_an_empty_unconfigured_catalogue(tmp_path):
    cat = load_resource_catalogue(tmp_path / "nope.yaml")
    assert cat.configured is False and len(cat) == 0


def test_the_shipped_example_config_is_loadable_and_expands(tmp_path):
    source = (_REPO_ROOT / "config" / "resources.example.yaml").read_text(
        encoding="utf-8")
    path = tmp_path / "resources.yaml"
    path.write_text(source, encoding="utf-8")
    cat = load_resource_catalogue(path)
    assert cat.configured is True

    spv_i = cat.resolve(_ref(KIND_SPV, "spv_i"))
    assert spv_i.predicate()[FIELD_SPV_ID] == "SPV_I"
    assert spv_i.population == POPULATION_FUNDED

    with pytest.raises(TraktError) as exc:
        cat.resolve(_ref(KIND_SPV, "spv_ii")).predicate()
    assert exc.value.code == ErrorCode.RESOURCE_NOT_PARTITIONABLE

    assert cat.resolve(_ref(KIND_POPULATION, "pipeline")).predicate() == {
        "view": POPULATION_PIPELINE}
    assert cat.resolve(_ref(KIND_PORTFOLIO, "ere_total")).predicate() == {}


def test_a_broken_config_leaves_the_catalogue_empty_rather_than_permissive(tmp_path):
    path = tmp_path / "resources.yaml"
    path.write_text("resources: [ not: valid: yaml", encoding="utf-8")
    cat = load_resource_catalogue(path)
    assert cat.configured is True and len(cat) == 0
    with pytest.raises(TraktError):
        cat.resolve(_ref(KIND_SPV, "spv_i"))


def test_a_config_declaring_an_unconstrained_resource_is_rejected_wholesale(tmp_path):
    path = tmp_path / "resources.yaml"
    path.write_text(
        "resources:\n"
        "  - tenant_id: ERE\n    kind: spv\n    resource_id: spv_i\n"
        "    spv_id: SPV_I\n"
        "  - tenant_id: ERE\n    kind: spv\n    resource_id: oops\n",
        encoding="utf-8")
    cat = load_resource_catalogue(path)
    assert len(cat) == 0          # the whole file is refused, not partially applied
    with pytest.raises(TraktError):
        cat.resolve(_ref(KIND_SPV, "spv_i"))


def test_loading_is_deterministic(tmp_path):
    source = (_REPO_ROOT / "config" / "resources.example.yaml").read_text(
        encoding="utf-8")
    path = tmp_path / "resources.yaml"
    path.write_text(source, encoding="utf-8")
    first, second = load_resource_catalogue(path), load_resource_catalogue(path)
    assert [r.to_dict() for r in first.records()] == [
        r.to_dict() for r in second.records()]


def test_the_repository_ships_no_live_resource_config():
    assert not (_REPO_ROOT / "config" / "resources.yaml").exists()


# --------------------------------------------------------------------------- #
# The warehouse-funder shape is representable — and granted to nobody
# --------------------------------------------------------------------------- #
def test_the_warehouse_funder_scope_is_now_expressible(tmp_path):
    """ERE / SPV I / funded — the scope Stage 1 could not write down."""
    source = (_REPO_ROOT / "config" / "resources.example.yaml").read_text(
        encoding="utf-8")
    path = tmp_path / "resources.yaml"
    path.write_text(source, encoding="utf-8")
    cat = load_resource_catalogue(path)

    resolved = cat.resolve(ResourceRef.parse("ERE/spv/spv_i"))
    check_attribution(resolved, FULL_FIELDS)
    assert resolved.owning_tenant_id == "ERE"
    assert resolved.predicate() == {"view": "funded",
                                    FIELD_PORTFOLIO_ID: ["direct_001"],
                                    FIELD_SPV_ID: "SPV_I"}
    # …and the pipeline it must NOT reach is a separate resource entirely.
    assert cat.resolve(ResourceRef.parse("ERE/population/pipeline")).predicate() == {
        "view": "pipeline"}


def test_stage2_grants_nobody_anything():
    """Stage 2 is identity of the THING, not access to it. Nothing here reads an
    organisation, and no entitlement machinery arrived with it."""
    import ast
    import inspect

    import trakt_core.resource as resource_mod
    from trakt_core.context import ExecutionContext

    # The resource model must not consult WHO is asking. Checked structurally
    # (imports and names actually used) rather than by scanning the text, which
    # legitimately cites load_organisation_registry as a precedent in prose.
    tree = ast.parse(inspect.getsource(resource_mod))
    imported = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert not any("organisation" in str(name).lower() for name in imported)
    assert not any("entitle" in str(name).lower() for name in imported)
    assert not hasattr(resource_mod, "Grant")
    assert not hasattr(resource_mod, "EntitlementStore")
    assert not hasattr(resource_mod, "AuthorisedResourceSet")

    # Stage 3 later gave ExecutionContext an entitlement axis, so the original
    # form of this check — "the context has no permitted_resources attribute" —
    # became a statement about which stage had shipped rather than about the
    # resource model. The property that actually matters is unchanged and is
    # asserted directly: naming a resource confers nothing, and a context that
    # was granted nothing permits nothing.
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u")
    assert ctx.entitlements is None
    assert ctx.permitted_resources == frozenset()
    assert ctx.capabilities_for(_ref(KIND_SPV, "spv_i")) == frozenset()


# --------------------------------------------------------------------------- #
# Compatibility
# --------------------------------------------------------------------------- #
def test_authorise_portfolio_access_is_unchanged():
    """Stage 2 adds a parallel vocabulary; it does not touch the one in use."""
    from trakt_core.context import ExecutionContext
    from trakt_core.tenancy import (
        TenantRecord, TenantRegistry, authorise_portfolio_access)

    registry = TenantRegistry(
        {"ERE": TenantRecord(tenant_id="ERE", default_portfolio_id="ERE",
                             portfolio_ids=frozenset({"ERE", "direct_001"}))},
        configured=True)
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u")

    assert authorise_portfolio_access(ctx, "direct_001", registry=registry).selector \
        == "direct_001"
    assert authorise_portfolio_access(ctx, None, registry=registry).selector == "ERE"
    with pytest.raises(TraktError) as exc:
        authorise_portfolio_access(ctx, "acquired_777", registry=registry)
    assert exc.value.code == ErrorCode.PORTFOLIO_NOT_AUTHORISED


def test_the_presentation_lens_still_fails_open_and_was_not_touched():
    """Deliberate: resolve_scope's Total fallback is correct for a stale UI
    selection. Stage 2's job was to build a separate, fail-closed path — not to
    change display behaviour and break the React workspace."""
    from trakt_core.portfolio import PortfolioRegistry, resolve_scope

    registry = PortfolioRegistry([PortfolioRecord(portfolio_id="direct_001",
                                                  portfolio_type="direct")])
    scope = resolve_scope(registry, "stale_selection")
    assert scope.is_total is True
    assert scope.fell_back_to_total is True
    assert scope.requested_context_id == "stale_selection"


def test_resource_module_imports_without_a_web_framework_or_pandas():
    import os
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, trakt_core.resource as r;"
         "assert 'fastapi' not in sys.modules, 'resource pulled in fastapi';"
         "assert 'starlette' not in sys.modules, 'resource pulled in starlette';"
         "assert 'pandas' not in sys.modules, 'resource pulled in pandas';"
         "c = r.load_resource_catalogue();"
         "print('ok')"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
        env={**os.environ, "TRAKT_RUNTIME_MODE": "test", "PYTHONPATH": str(_REPO_ROOT)})
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
