"""The governed field-selection policy: roles, applicability, caps and audit.

§6 and §10. The point of these tests is that the policy is real: the registry's
242 entries do NOT all become an overview, a derived input never appears as a
standalone metric, an asset-specific field never leaks onto the wrong book, and
every exclusion is recorded with a reason.
"""

from __future__ import annotations

import pytest

from mi_agent.business_semantics import (
    WORKFLOW_PERIOD_CHANGE,
    load_business_semantics,
)
from mi_agent.period_change import models as m
from mi_agent.period_change import selection as sel

from .test_period_change_calculations import entry as make_entry


@pytest.fixture(scope="module")
def registry():
    return load_business_semantics()


@pytest.fixture(scope="module")
def policy():
    return sel.load_policy()


def all_fields(registry):
    """Every registry field, as though every one were populated at both dates."""
    return set(registry.fields())


def inputs(registry, mode, *, start=None, end=None, assets=("cross_asset",),
           fields=(), concepts=()):
    available = all_fields(registry)
    return sel.SelectionInputs(
        mode=mode, registry=registry,
        start_fields=set(start if start is not None else available),
        end_fields=set(end if end is not None else available),
        asset_classes=tuple(assets), requested_fields=tuple(fields),
        requested_concepts=tuple(concepts))


# --------------------------------------------------------------------------- #
# The policy is configuration, not code
# --------------------------------------------------------------------------- #
def test_the_policy_is_loaded_from_the_governed_config(policy):
    assert sel.POLICY_PATH.exists()
    assert policy.name == "period_change_overview"
    assert policy.workflow_tag == WORKFLOW_PERIOD_CHANGE
    assert policy.overview_concepts
    assert "derived_input" in policy.excluded_overview_roles
    assert "supporting_attribute" in policy.excluded_overview_roles
    assert "static_baseline" in policy.excluded_from_ranking


# --------------------------------------------------------------------------- #
# Portfolio-overview mode — §10.C
# --------------------------------------------------------------------------- #
class TestOverviewMode:
    def test_an_overview_is_a_governed_subset_not_the_whole_registry(self, registry, policy):
        out = sel.select_fields(inputs(registry, m.MODE_PORTFOLIO_OVERVIEW),
                                policy=policy)
        tagged = len(registry.for_workflow(WORKFLOW_PERIOD_CHANGE))
        assert tagged == 106
        assert 0 < len(out.measures) <= policy.max_measures_total
        assert len(out.dimensions) <= policy.max_dimensions
        assert len(out.measures) + len(out.dimensions) < tagged

    def test_the_overview_spreads_across_concepts(self, registry, policy):
        out = sel.select_fields(inputs(registry, m.MODE_PORTFOLIO_OVERVIEW),
                                policy=policy)
        concepts = {registry.get(f).analytical_concept for f in out.measures}
        assert len(concepts) >= 4
        assert concepts <= set(policy.overview_concepts)
        assert set(out.concepts_covered) == concepts

    def test_selection_is_deterministic(self, registry, policy):
        first = sel.select_fields(inputs(registry, m.MODE_PORTFOLIO_OVERVIEW),
                                  policy=policy)
        second = sel.select_fields(inputs(registry, m.MODE_PORTFOLIO_OVERVIEW),
                                   policy=policy)
        assert first.measures == second.measures
        assert first.dimensions == second.dimensions

    def test_exposure_is_represented_by_the_governed_balance(self, registry, policy):
        out = sel.select_fields(inputs(registry, m.MODE_PORTFOLIO_OVERVIEW),
                                policy=policy)
        assert "current_outstanding_balance" in out.measures

    def test_only_entries_tagged_for_period_change_are_candidates(self, registry, policy):
        out = sel.select_fields(inputs(registry, m.MODE_PORTFOLIO_OVERVIEW),
                                policy=policy)
        for field in out.measures + out.dimensions:
            assert WORKFLOW_PERIOD_CHANGE in registry.get(field).workflow_tags

    def test_one_idea_is_not_reported_five_ways(self, registry, policy):
        """Non-duplication: a cap on entries sharing a
        (concept, aggregation, temporality) signature."""
        out = sel.select_fields(inputs(registry, m.MODE_PORTFOLIO_OVERVIEW),
                                policy=policy)
        seen = {}
        for field in out.measures:
            e = registry.get(field)
            key = (e.analytical_concept, e.default_aggregation, e.temporality)
            seen[key] = seen.get(key, 0) + 1
        assert max(seen.values()) <= policy.max_measures_per_signature


# --------------------------------------------------------------------------- #
# Roles — §6
# --------------------------------------------------------------------------- #
class TestRoles:
    def test_measures_become_metrics_and_dimensions_become_distributions(
            self, registry, policy):
        out = sel.select_fields(inputs(registry, m.MODE_PORTFOLIO_OVERVIEW),
                                policy=policy)
        assert all(registry.get(f).analytical_role == "measure" for f in out.measures)
        assert all(registry.get(f).analytical_role == "dimension"
                   for f in out.dimensions)

    def test_derived_inputs_never_appear_as_standalone_overview_metrics(
            self, registry, policy):
        """pd_previous / lgd_current are inputs to a governed migration
        calculation, not metrics. §6 forbids presenting them as such."""
        out = sel.select_fields(inputs(registry, m.MODE_PORTFOLIO_OVERVIEW),
                                policy=policy)
        derived = {e.field for e in registry.for_workflow(WORKFLOW_PERIOD_CHANGE)
                   if e.analytical_role == "derived_input"}
        assert derived
        assert not derived & set(out.measures)
        assert not derived & set(out.dimensions)
        excluded = {e.field for e in out.excluded
                    if e.reason == sel.EXCLUDED_ROLE_NOT_ELIGIBLE}
        assert derived <= excluded

    def test_prior_and_current_risk_pairs_are_excluded_by_name_of_role_not_field(
            self, registry, policy):
        out = sel.select_fields(inputs(registry, m.MODE_PORTFOLIO_OVERVIEW),
                                policy=policy)
        for field in ("pd_previous", "pd_current", "lgd_previous", "lgd_current"):
            assert field not in out.measures

    def test_supporting_attributes_are_not_ordinary_portfolio_metrics(
            self, registry, policy):
        out = sel.select_fields(inputs(registry, m.MODE_PORTFOLIO_OVERVIEW),
                                policy=policy)
        supporting = {e.field for e in registry.entries.values()
                      if e.analytical_role == "supporting_attribute"}
        assert supporting
        assert not supporting & set(out.measures)


# --------------------------------------------------------------------------- #
# Temporality — §7
# --------------------------------------------------------------------------- #
def test_static_baselines_are_excluded_from_ordinary_ranking(registry, policy):
    static = {f for f, e in registry.entries.items()
              if e.temporality == "static_baseline"}
    assert static
    out = sel.select_fields(inputs(registry, m.MODE_PORTFOLIO_OVERVIEW), policy=policy)
    assert not static & set(out.measures)


def test_a_period_change_tagged_static_baseline_is_excluded_with_its_reason(policy):
    """The committed registry tags no static baseline for period change; the rule
    is still proved directly so it cannot regress silently."""
    from mi_agent.business_semantics import BusinessSemanticsRegistry

    baseline = make_entry("original_loan_to_value", temporality="static_baseline",
                          concept="leverage")
    registry = BusinessSemanticsRegistry(
        version="test", schema_version=2, source_registry="test",
        entries={baseline.field: baseline}, taxonomy={})
    out = sel.select_fields(sel.SelectionInputs(
        mode=m.MODE_PORTFOLIO_OVERVIEW, registry=registry,
        start_fields={baseline.field}, end_fields={baseline.field},
        asset_classes=("cross_asset",)), policy=policy)
    assert out.measures == ()
    assert out.excluded[0].reason == sel.EXCLUDED_STATIC_BASELINE


# --------------------------------------------------------------------------- #
# Asset applicability — §9
# --------------------------------------------------------------------------- #
class TestAssetApplicability:
    def test_an_equity_release_field_is_not_reported_for_an_sme_book(
            self, registry, policy):
        out = sel.select_fields(
            inputs(registry, m.MODE_PORTFOLIO_OVERVIEW, assets=("sme",)),
            policy=policy)
        for field in out.measures + out.dimensions:
            applicability = registry.get(field).asset_applicability
            assert "cross_asset" in applicability or "sme" in applicability

    def test_an_inapplicable_entry_records_why(self, registry, policy):
        out = sel.select_fields(
            inputs(registry, m.MODE_PORTFOLIO_OVERVIEW, assets=("sme",)),
            policy=policy)
        reasons = {e.reason for e in out.excluded}
        assert sel.EXCLUDED_ASSET_NOT_APPLICABLE in reasons

    def test_an_unidentified_book_admits_only_cross_asset_entries(
            self, registry, policy):
        out = sel.select_fields(
            inputs(registry, m.MODE_PORTFOLIO_OVERVIEW, assets=()), policy=policy)
        for field in out.measures + out.dimensions:
            assert "cross_asset" in registry.get(field).asset_applicability


# --------------------------------------------------------------------------- #
# Availability — §9
# --------------------------------------------------------------------------- #
class TestAvailability:
    def test_a_field_absent_from_both_snapshots_is_omitted(self, registry, policy):
        present = {"current_outstanding_balance", "account_status"}
        out = sel.select_fields(
            inputs(registry, m.MODE_PORTFOLIO_OVERVIEW, start=present, end=present),
            policy=policy)
        assert set(out.measures) <= present
        assert set(out.dimensions) <= present
        assert any(e.reason == sel.EXCLUDED_ABSENT_FROM_BOTH for e in out.excluded)

    def test_a_field_present_at_only_one_date_is_still_a_candidate(
            self, registry, policy):
        out = sel.select_fields(
            inputs(registry, m.MODE_PORTFOLIO_OVERVIEW,
                   start={"account_status"},
                   end={"current_outstanding_balance", "account_status"}),
            policy=policy)
        assert "current_outstanding_balance" in out.measures

    def test_a_column_of_nothing_but_nulls_is_not_an_available_field(self):
        import pandas as pd
        frame = pd.DataFrame({"a": [1.0, 2.0], "b": [None, None]})
        assert sel.populated_fields(frame) == {"a"}

    def test_availability_states(self):
        assert sel.availability("x", {"x"}, {"x"}) == sel.AVAILABILITY_BOTH
        assert sel.availability("x", {"x"}, set()) == sel.AVAILABILITY_START_ONLY
        assert sel.availability("x", set(), {"x"}) == sel.AVAILABILITY_END_ONLY
        assert sel.availability("x", set(), set()) == sel.AVAILABILITY_NEITHER


# --------------------------------------------------------------------------- #
# Concept mode — §10.B
# --------------------------------------------------------------------------- #
class TestConceptMode:
    def test_a_concept_selects_only_that_concept(self, registry, policy):
        out = sel.select_fields(
            inputs(registry, m.MODE_CONCEPT, concepts=("credit_quality",)),
            policy=policy)
        assert out.measures
        assert all(registry.get(f).analytical_concept == "credit_quality"
                   for f in out.measures + out.dimensions)
        assert out.concepts_covered == ("credit_quality",)

    def test_derived_inputs_are_excluded_from_concept_mode_too(self, registry, policy):
        out = sel.select_fields(
            inputs(registry, m.MODE_CONCEPT, concepts=("credit_quality",)),
            policy=policy)
        assert all(registry.get(f).analytical_role != "derived_input"
                   for f in out.measures)

    def test_out_of_scope_concepts_are_recorded_as_excluded(self, registry, policy):
        out = sel.select_fields(
            inputs(registry, m.MODE_CONCEPT, concepts=("credit_quality",)),
            policy=policy)
        assert any(e.reason == sel.EXCLUDED_CONCEPT_NOT_IN_SCOPE
                   for e in out.excluded)


# --------------------------------------------------------------------------- #
# Requested-metric mode — §10.A
# --------------------------------------------------------------------------- #
class TestRequestedMetricMode:
    def test_only_the_requested_field_is_analysed(self, registry, policy):
        out = sel.select_fields(
            inputs(registry, m.MODE_REQUESTED_METRIC,
                   fields=("current_loan_to_value",)), policy=policy)
        assert out.measures == ("current_loan_to_value",)
        assert out.dimensions == ()

    def test_a_requested_dimension_becomes_a_distribution(self, registry, policy):
        out = sel.select_fields(
            inputs(registry, m.MODE_REQUESTED_METRIC, fields=("account_status",)),
            policy=policy)
        assert out.dimensions == ("account_status",)
        assert out.measures == ()

    def test_a_field_outside_the_registry_is_refused_not_substituted(
            self, registry, policy):
        """§2: never silently substitute another metric for an unavailable one."""
        out = sel.select_fields(
            inputs(registry, m.MODE_REQUESTED_METRIC, fields=("not_a_field",)),
            policy=policy)
        assert out.measures == () and out.dimensions == ()
        assert out.excluded[0].reason == sel.EXCLUDED_NOT_IN_REGISTRY

    def test_a_requested_field_absent_from_both_snapshots_is_refused(
            self, registry, policy):
        out = sel.select_fields(
            inputs(registry, m.MODE_REQUESTED_METRIC,
                   start={"account_status"}, end={"account_status"},
                   fields=("current_loan_to_value",)), policy=policy)
        assert out.measures == ()
        assert out.excluded[0].reason == sel.EXCLUDED_ABSENT_FROM_BOTH


# --------------------------------------------------------------------------- #
# Audit — §16
# --------------------------------------------------------------------------- #
def test_every_exclusion_carries_a_reason(registry, policy):
    out = sel.select_fields(inputs(registry, m.MODE_PORTFOLIO_OVERVIEW), policy=policy)
    assert out.excluded
    assert all(e.reason for e in out.excluded)
    data = out.to_dict()
    assert data["policy_name"] == policy.name
    assert data["policy_version"] == policy.version
    assert len(data["excluded_candidates"]) == len(out.excluded)
