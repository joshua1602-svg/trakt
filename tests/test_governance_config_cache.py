#!/usr/bin/env python3
"""The governance registries are parsed once per content version, not per request.

Before this cache every governed call re-read and re-validated YAML, at a cost
linear in organisations, resources and grants — 7 ms at pilot scale, 5,253 ms at
A2A scale, paid before any data was touched, on every call.

What this suite has to prove is that making it fast did not make it wrong:

* an unchanged file is parsed once, however many callers ask;
* a **changed** file is picked up on the very next call — no TTL, no restart;
* tenant, organisation and resource separation is untouched;
* every permission outcome is byte-for-byte what it was before;
* the cache memoises *configuration*, never an authorisation *decision*.

The last point is the one that matters most. Caching a decision would mean a
revoked grant kept working; caching the config it is decided from does not.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core import config_cache
from trakt_core.context import SCOPE_MI_QUERY, SCOPE_RISK_READ, ExecutionContext
from trakt_core.entitlement import authorise_resource_access, load_entitlement_store
from trakt_core.errors import ErrorCode, TraktError
from trakt_core.organisation import load_organisation_registry
from trakt_core.principal import load_principal_registry
from trakt_core.resource import load_resource_catalogue
from trakt_core.tenancy import authorise_portfolio_access, load_tenant_registry

DIR_A = "aaaaaaaa-0000-4000-8000-000000000000"
DIR_B = "bbbbbbbb-0000-4000-8000-000000000000"


@pytest.fixture(autouse=True)
def _clean_cache():
    config_cache.reset()
    yield
    config_cache.reset()


def _write(tmp_path: Path, monkeypatch, *, grants_for_a=("risk:read",),
           org_b_enabled=True) -> Path:
    """A two-organisation, two-tenant deployment wired through the env vars."""
    orgs = {"organisations": [
        {"organisation_id": "org_a", "organisation_type": "investor",
         "microsoft_tenant_ids": [DIR_A], "enabled": True},
        {"organisation_id": "org_b", "organisation_type": "investor",
         "microsoft_tenant_ids": [DIR_B], "enabled": org_b_enabled},
    ]}
    res = {"resources": [
        {"tenant_id": "ERE", "kind": "source_portfolio", "resource_id": "direct_001",
         "source_portfolio_ids": ["direct_001"]},
        {"tenant_id": "ERE", "kind": "source_portfolio", "resource_id": "acquired_001",
         "source_portfolio_ids": ["acquired_001"]},
    ]}
    ent = {"entitlements": []}
    if grants_for_a:
        ent["entitlements"].append(
            {"organisation_id": "org_a", "resource": "ERE/source_portfolio/direct_001",
             "capabilities": list(grants_for_a)})
    ent["entitlements"].append(
        {"organisation_id": "org_b", "resource": "ERE/source_portfolio/acquired_001",
         "capabilities": ["risk:read"]})
    ten = {"tenants": {"ERE": {"default_portfolio": "ERE",
                               "portfolios": ["ERE", "direct_001", "acquired_001"]},
                       "OTHER": {"default_portfolio": "OTHER",
                                 "portfolios": ["OTHER"]}}}
    for name, doc in (("organisations", orgs), ("resources", res),
                      ("entitlements", ent), ("tenancy", ten),
                      ("principals", {"principals": []})):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(doc))
    monkeypatch.setenv("TRAKT_ORGANISATIONS_CONFIG", str(tmp_path / "organisations.yaml"))
    monkeypatch.setenv("TRAKT_RESOURCES_CONFIG", str(tmp_path / "resources.yaml"))
    monkeypatch.setenv("TRAKT_ENTITLEMENTS_CONFIG", str(tmp_path / "entitlements.yaml"))
    monkeypatch.setenv("TRAKT_TENANCY_CONFIG", str(tmp_path / "tenancy.yaml"))
    monkeypatch.setenv("TRAKT_PRINCIPALS_CONFIG", str(tmp_path / "principals.yaml"))
    return tmp_path


def _context(organisation="org_a", scopes=(SCOPE_RISK_READ,)):
    from trakt_core.entitlement import resolve_entitlements
    return ExecutionContext(
        tenant_id="ERE", actor_id="a", actor_type="service",
        channel="enterprise_agent", scopes=frozenset(scopes),
        organisation_id=organisation,
        entitlements=resolve_entitlements(organisation))


# --------------------------------------------------------------------------- #
# It caches
# --------------------------------------------------------------------------- #
def _count_yaml_reads(fn, n=10):
    """How many .yaml files ``fn`` actually reads from disk over ``n`` calls."""
    reads = []
    real = Path.read_text

    def counting(self, *a, **k):
        if self.suffix in (".yaml", ".yml"):
            reads.append(self.name)
        return real(self, *a, **k)

    Path.read_text = counting
    try:
        for _ in range(n):
            fn()
    finally:
        Path.read_text = real
    return reads


def test_an_unchanged_registry_is_parsed_once_however_many_callers_ask(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch)
    for loader in (load_organisation_registry, load_resource_catalogue,
                   load_entitlement_store, load_tenant_registry,
                   load_principal_registry):
        config_cache.reset()
        loader()                                  # first call parses
        assert _count_yaml_reads(loader, n=20) == [], (
            f"{loader.__name__} re-read its file after the first call")


def test_the_whole_governed_chain_reads_nothing_in_the_steady_state(tmp_path, monkeypatch):
    """The measurement that matters: a warm governed call touches no YAML."""
    _write(tmp_path, monkeypatch)

    def one_governed_call():
        ctx = _context()
        authorise_resource_access(ctx, "ERE/source_portfolio/direct_001", SCOPE_RISK_READ)

    one_governed_call()                            # warm
    assert _count_yaml_reads(one_governed_call, n=10) == []


def test_the_mi_query_governance_step_reads_nothing_in_the_steady_state(tmp_path, monkeypatch):
    """MI Query loads only the tenancy registry — but it loaded it every request."""
    _write(tmp_path, monkeypatch)
    load_tenant_registry(default_tenant_id="ERE")  # warm
    assert _count_yaml_reads(
        lambda: load_tenant_registry(default_tenant_id="ERE"), n=20) == []


def test_the_cache_reports_hits(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch)
    config_cache.reset()
    for _ in range(5):
        load_organisation_registry()
    stats = config_cache.stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 4


# --------------------------------------------------------------------------- #
# …and it refreshes. This is the half that keeps it safe.
# --------------------------------------------------------------------------- #
def test_a_changed_file_is_picked_up_on_the_very_next_call(tmp_path, monkeypatch):
    """No TTL, no restart. An operator edits the file; the next call sees it."""
    _write(tmp_path, monkeypatch)
    assert load_organisation_registry().get("org_a") is not None

    # Disable org_a. `_write` regenerates every file; the mtime moves.
    doc = yaml.safe_load((tmp_path / "organisations.yaml").read_text())
    doc["organisations"][0]["enabled"] = False
    time.sleep(0.01)
    (tmp_path / "organisations.yaml").write_text(yaml.safe_dump(doc))

    record = load_organisation_registry().get("org_a")
    assert record is not None and record.enabled is False


def test_a_revoked_grant_stops_working_on_the_next_request(tmp_path, monkeypatch):
    """The load-bearing safety test.

    Caching *configuration* is safe. Caching a *decision* would mean a revoked
    grant kept working. This proves the first and not the second.
    """
    _write(tmp_path, monkeypatch)
    ctx = _context()
    authorise_resource_access(ctx, "ERE/source_portfolio/direct_001", SCOPE_RISK_READ)

    # Revoke it.
    doc = yaml.safe_load((tmp_path / "entitlements.yaml").read_text())
    doc["entitlements"] = [g for g in doc["entitlements"]
                           if g["organisation_id"] != "org_a"]
    time.sleep(0.01)
    (tmp_path / "entitlements.yaml").write_text(yaml.safe_dump(doc))

    # A context built now resolves the new (empty) entitlements…
    revoked = _context()
    with pytest.raises(TraktError) as excinfo:
        authorise_resource_access(revoked, "ERE/source_portfolio/direct_001",
                                  SCOPE_RISK_READ)
    assert excinfo.value.code == ErrorCode.RESOURCE_NOT_AUTHORISED


def test_an_entitlement_store_refreshes_when_its_RESOURCE_catalogue_changes(tmp_path, monkeypatch):
    """A grant is validated against the other two registries, so its identity is
    theirs as well as its own. Disabling a resource must invalidate the store
    even though entitlements.yaml did not change."""
    _write(tmp_path, monkeypatch)
    assert len(load_entitlement_store()) == 2

    doc = yaml.safe_load((tmp_path / "resources.yaml").read_text())
    doc["resources"][0]["enabled"] = False        # direct_001 off
    time.sleep(0.01)
    (tmp_path / "resources.yaml").write_text(yaml.safe_dump(doc))

    # The whole file is rejected when any grant is invalid — the pre-existing,
    # deliberate fail-closed behaviour. The point here is that it is RE-EVALUATED.
    assert len(load_entitlement_store()) == 0


def test_a_deleted_file_is_noticed(tmp_path, monkeypatch):
    """Absence is a cached state too, and it must be able to change."""
    _write(tmp_path, monkeypatch)
    assert load_organisation_registry().configured is True
    time.sleep(0.01)
    (tmp_path / "organisations.yaml").unlink()
    assert load_organisation_registry().configured is False


def test_the_kill_switch_restores_the_previous_behaviour(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch)
    monkeypatch.setenv("TRAKT_CONFIG_CACHE", "off")
    reads = _count_yaml_reads(load_organisation_registry, n=5)
    assert len(reads) == 5


# --------------------------------------------------------------------------- #
# Isolation is unchanged
# --------------------------------------------------------------------------- #
def test_two_organisations_do_not_share_a_cached_entitlement_result(tmp_path, monkeypatch):
    """The registry is shared; the resolved entitlements are not."""
    _write(tmp_path, monkeypatch)
    a, b = _context("org_a"), _context("org_b")

    authorise_resource_access(a, "ERE/source_portfolio/direct_001", SCOPE_RISK_READ)
    authorise_resource_access(b, "ERE/source_portfolio/acquired_001", SCOPE_RISK_READ)

    # Each is refused the other's resource, identically to before caching.
    for ctx, other in ((a, "ERE/source_portfolio/acquired_001"),
                       (b, "ERE/source_portfolio/direct_001")):
        with pytest.raises(TraktError) as excinfo:
            authorise_resource_access(ctx, other, SCOPE_RISK_READ)
        assert excinfo.value.code == ErrorCode.RESOURCE_NOT_AUTHORISED


def test_tenant_separation_survives_caching(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch)
    registry = load_tenant_registry(default_tenant_id="ERE")
    ctx = ExecutionContext(tenant_id="ERE", actor_id="u",
                           scopes=frozenset({"portfolio:read"}))
    assert authorise_portfolio_access(ctx, "direct_001", registry=registry).selector \
        == "direct_001"
    with pytest.raises(TraktError) as excinfo:
        authorise_portfolio_access(ctx, "OTHER", registry=registry)
    assert excinfo.value.code == ErrorCode.TENANT_MISMATCH


def test_the_default_tenant_is_part_of_the_key(tmp_path, monkeypatch):
    """Two deployments sharing a process must not serve each other's default."""
    (tmp_path / "empty.yaml").write_text("tenants: {}\n")
    monkeypatch.setenv("TRAKT_TENANCY_CONFIG", str(tmp_path / "missing.yaml"))
    one = load_tenant_registry(default_tenant_id="ERE")
    two = load_tenant_registry(default_tenant_id="OTHER")
    assert one.tenant_ids() == frozenset({"ERE"})
    assert two.tenant_ids() == frozenset({"OTHER"})


def test_injected_registries_bypass_the_cache_entirely(tmp_path, monkeypatch):
    """A caller-supplied registry cannot be fingerprinted, so the result must not
    be memoised under the file's identity — or a test fixture would leak into the
    next caller."""
    from trakt_core.organisation import OrganisationRegistry
    from trakt_core.resource import ResourceCatalogue

    _write(tmp_path, monkeypatch)
    config_cache.reset()
    injected = load_entitlement_store(
        organisations=OrganisationRegistry([], configured=True),
        resources=ResourceCatalogue([], configured=True))
    assert config_cache.stats()["entries"] == 0
    assert len(injected) == 0                       # every grant rejected

    # …and the normal path is unaffected by that call.
    assert len(load_entitlement_store()) == 2


# --------------------------------------------------------------------------- #
# Outcomes are unchanged
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("resource,capability,expected", [
    ("ERE/source_portfolio/direct_001", SCOPE_RISK_READ, None),
    ("ERE/source_portfolio/acquired_001", SCOPE_RISK_READ, ErrorCode.RESOURCE_NOT_AUTHORISED),
    ("ERE/source_portfolio/nonexistent", SCOPE_RISK_READ, ErrorCode.RESOURCE_NOT_AUTHORISED),
    ("OTHER/source_portfolio/direct_001", SCOPE_RISK_READ, ErrorCode.RESOURCE_NOT_AUTHORISED),
])
def test_permission_outcomes_are_unchanged(tmp_path, monkeypatch, resource,
                                           capability, expected):
    _write(tmp_path, monkeypatch)
    ctx = _context()
    if expected is None:
        assert authorise_resource_access(ctx, resource, capability) is not None
    else:
        with pytest.raises(TraktError) as excinfo:
            authorise_resource_access(ctx, resource, capability)
        assert excinfo.value.code == expected


def test_a_capability_the_grant_does_not_carry_is_still_refused(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch)
    ctx = _context(scopes=(SCOPE_RISK_READ, SCOPE_MI_QUERY))
    with pytest.raises(TraktError) as excinfo:
        authorise_resource_access(ctx, "ERE/source_portfolio/direct_001", SCOPE_MI_QUERY)
    assert excinfo.value.code == ErrorCode.CAPABILITY_NOT_GRANTED


def test_config_cache_holds_no_execution_context_or_decision(tmp_path, monkeypatch):
    """Structural: nothing in the cache is request-scoped."""
    from trakt_core.entitlement import AuthorisedResource, ResolvedEntitlements

    _write(tmp_path, monkeypatch)
    _context()
    authorise_resource_access(_context(), "ERE/source_portfolio/direct_001",
                              SCOPE_RISK_READ)
    for value in list(config_cache._CACHE.values()):
        assert not isinstance(value, (ExecutionContext, AuthorisedResource,
                                      ResolvedEntitlements))
