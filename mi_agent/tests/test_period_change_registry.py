"""The Business Semantics Registry loader, and the governed source-override seam.

The registry itself is validated by ``tests/test_business_semantics_registry.py``
(schema, taxonomy, exclusions). These tests cover the LOADER: that it reads v2
faithfully, refuses a schema it does not understand, exposes the metadata the
period-change workflow depends on, and applies a governed source override without
any workflow code changing.
"""

from __future__ import annotations

import pytest
import yaml

from mi_agent import business_semantics as bs


@pytest.fixture(scope="module")
def registry():
    return bs.load_business_semantics()


@pytest.fixture(scope="module")
def raw():
    with bs.REGISTRY_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def test_loads_the_committed_v2_registry(registry):
    assert registry.schema_version == bs.SUPPORTED_SCHEMA_VERSION == 2
    assert registry.version == "0.3.0"
    assert len(registry) == 243


def test_every_yaml_field_becomes_a_typed_entry(registry, raw):
    assert set(registry.fields()) == set(raw["fields"])
    entry = registry.get("current_loan_to_value")
    assert entry.analytical_role == bs.ROLE_MEASURE
    assert entry.temporality == bs.TEMPORALITY_POINT_IN_TIME
    assert entry.default_aggregation == bs.AGG_WEIGHTED_AVERAGE
    assert entry.weight_field == "current_outstanding_balance"
    assert entry.rationale


def test_an_unsupported_schema_version_is_refused_not_misread():
    """A future schema must fail loudly: silently reading v3 as v2 would publish
    numbers derived from metadata that no longer means what the loader thinks."""
    with pytest.raises(bs.BusinessSemanticsError):
        bs.parse_registry({"metadata": {"schema_version": 3}, "fields": {"a": {}}})


def test_an_empty_registry_is_refused():
    with pytest.raises(bs.BusinessSemanticsError):
        bs.parse_registry({"metadata": {"schema_version": 2}, "fields": {}})


def test_the_loader_constants_match_the_registry_taxonomy(registry):
    """The named constants are the registry's controlled vocabulary, not a copy
    that can drift."""
    taxonomy = registry.taxonomy
    assert set(taxonomy["analytical_roles"]) == {
        bs.ROLE_MEASURE, bs.ROLE_DIMENSION, bs.ROLE_DERIVED_INPUT,
        bs.ROLE_SUPPORTING_ATTRIBUTE}
    assert set(taxonomy["temporality"]) == {
        bs.TEMPORALITY_POINT_IN_TIME, bs.TEMPORALITY_PERIOD_FLOW,
        bs.TEMPORALITY_CUMULATIVE, bs.TEMPORALITY_STATIC_BASELINE}
    assert set(taxonomy["default_aggregations"]) == {
        bs.AGG_SUM, bs.AGG_AVERAGE, bs.AGG_WEIGHTED_AVERAGE, bs.AGG_SHARE,
        bs.AGG_DISTRIBUTION, bs.AGG_COUNT}
    assert set(taxonomy["directionality"]) == {
        bs.DIRECTION_HIGHER_IS_BETTER, bs.DIRECTION_HIGHER_IS_WORSE,
        bs.DIRECTION_LOWER_IS_BETTER, bs.DIRECTION_LOWER_IS_WORSE,
        bs.DIRECTION_NEUTRAL, bs.DIRECTION_CONTEXT_DEPENDENT}
    assert bs.WORKFLOW_PERIOD_CHANGE in taxonomy["workflow_tags"]


# --------------------------------------------------------------------------- #
# Workflow-tag projection
# --------------------------------------------------------------------------- #
def test_for_workflow_returns_only_tagged_entries(registry):
    tagged = registry.for_workflow(bs.WORKFLOW_PERIOD_CHANGE)
    assert tagged
    assert all(bs.WORKFLOW_PERIOD_CHANGE in e.workflow_tags for e in tagged)
    assert len(tagged) < len(registry)


def test_for_workflow_is_deterministically_ordered(registry):
    names = [e.field for e in registry.for_workflow(bs.WORKFLOW_PERIOD_CHANGE)]
    assert names == sorted(names)


def test_known_v2_caveats_are_carried_verbatim(registry):
    """§18: the workflow must respect the registry's own classifications rather
    than reinterpreting them from field names."""
    assert registry.get("allocated_losses").temporality == bs.TEMPORALITY_CUMULATIVE
    # Named like cumulative amounts; governed as point-in-time. Not reinterpreted.
    for field in ("further_advance_amount", "negative_amortisation"):
        entry = registry.get(field)
        if entry is not None:
            assert entry.temporality == bs.TEMPORALITY_POINT_IN_TIME


def test_previous_period_risk_inputs_are_derived_inputs(registry):
    """§18: prior/current snapshot-pair fields are derived inputs, never metrics."""
    for field in ("pd_previous", "pd_current", "lgd_previous", "lgd_current",
                  "ifrs9_stage_previous", "ifrs9_stage_current"):
        assert registry.get(field).analytical_role == bs.ROLE_DERIVED_INPUT


# --------------------------------------------------------------------------- #
# Asset applicability
# --------------------------------------------------------------------------- #
def test_cross_asset_entries_apply_to_any_book(registry):
    entry = registry.get("current_outstanding_balance")
    assert entry.applies_to_asset(("sme",))
    assert entry.applies_to_asset(())


def test_asset_specific_entries_apply_only_to_a_matching_book(registry):
    specific = next(e for e in registry.for_workflow(bs.WORKFLOW_PERIOD_CHANGE)
                    if bs.ASSET_CROSS_ASSET not in e.asset_applicability)
    asset = specific.asset_applicability[0]
    assert specific.applies_to_asset((asset,))
    assert not specific.applies_to_asset(("sme" if asset != "sme" else "corporate",))


def test_an_unidentified_book_admits_only_cross_asset_entries(registry):
    """The conservative reading: an equity-release field is never reported for a
    portfolio whose asset class could not be established."""
    specific = next(e for e in registry.for_workflow(bs.WORKFLOW_PERIOD_CHANGE)
                    if bs.ASSET_CROSS_ASSET not in e.asset_applicability)
    assert not specific.applies_to_asset(())


# --------------------------------------------------------------------------- #
# Governed source overrides — the extension point for §18
# --------------------------------------------------------------------------- #
def test_a_source_override_can_restate_temporality_without_code_changes(registry):
    overrides = {"client_x": {"allocated_losses": {"temporality": "period_flow"}}}
    view = registry.for_source("client_x", overrides)

    assert view.get("allocated_losses").temporality == "period_flow"
    assert view.get("allocated_losses").is_overridden
    # The base registry is untouched: an override is a VIEW, not a mutation.
    assert registry.get("allocated_losses").temporality == bs.TEMPORALITY_CUMULATIVE


def test_an_override_records_what_it_replaced(registry):
    overrides = {"c": {"allocated_losses": {"temporality": "period_flow"}}}
    entry = registry.for_source("c", overrides).get("allocated_losses")
    assert entry.overridden_from["source_id"] == "c"
    assert entry.overridden_from["previous"]["temporality"] == bs.TEMPORALITY_CUMULATIVE


def test_an_override_cannot_re_curate_what_a_field_means(registry):
    """An override corrects a SOURCE CONVENTION. Re-pointing a field at another
    concept or role is curation and belongs in the registry build, not here."""
    overrides = {"c": {"allocated_losses": {"analytical_concept": "exposure",
                                            "analytical_role": "dimension"}}}
    entry = registry.for_source("c", overrides).get("allocated_losses")
    assert entry.analytical_concept == "loss"
    assert entry.analytical_role == bs.ROLE_MEASURE
    assert not entry.is_overridden


def test_an_override_for_an_unknown_field_is_ignored(registry):
    view = registry.for_source("c", {"c": {"not_a_field": {"temporality": "x"}}})
    assert "not_a_field" not in view


def test_no_source_id_returns_the_same_registry(registry):
    assert registry.for_source(None) is registry
    assert registry.for_source("unknown_source", {}) is registry


def test_provenance_identifies_the_registry_that_produced_a_result(registry):
    provenance = registry.to_provenance()
    assert provenance["registry_version"] == "0.3.0"
    assert provenance["schema_version"] == 2
    assert provenance["entry_count"] == 243
    assert provenance["overridden_fields"] == []
