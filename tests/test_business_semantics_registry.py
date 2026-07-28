"""Validation tests for the governed Business Semantics Registry.

Covers the quality controls required when the registry was created:

* every Business Semantics Registry field exists in the source registry;
* no duplicate field keys in the YAML;
* controlled taxonomy values only (analytical concept, categories, workflow
  tags, aggregation type, directionality, confidence, asset applicability);
* every included field has a rationale and at least one analytical
  concept/category;
* representative excluded fields (identifiers, names, addresses, free text,
  provenance, technical fields) do not appear accidentally;
* the YAML loads successfully;
* registry generation is deterministic and matches the committed file;
* the generator refuses to overwrite the reviewed output without --force.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "config" / "business_semantics_registry.yaml"
SOURCE_PATH = REPO_ROOT / "config" / "system" / "fields_registry.yaml"
BUILDER_PATH = REPO_ROOT / "scripts" / "build_business_semantics_registry.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_business_semantics_registry", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


@pytest.fixture(scope="module")
def registry():
    with REGISTRY_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def source_fields():
    with SOURCE_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["fields"]


# --------------------------------------------------------------------------- #
# Loading / structure
# --------------------------------------------------------------------------- #


def test_yaml_loads_successfully(registry):
    assert isinstance(registry, dict)
    assert isinstance(registry.get("metadata"), dict)
    assert isinstance(registry.get("fields"), dict)
    assert registry["fields"], "registry must contain at least one field"


def test_no_duplicate_field_keys():
    """yaml.safe_load silently keeps the last duplicate — detect explicitly."""

    class StrictLoader(yaml.SafeLoader):
        pass

    def _construct_mapping(loader, node, deep=False):
        keys = [loader.construct_object(k, deep=deep) for k, _ in node.value]
        dupes = {k for k in keys if keys.count(k) > 1}
        assert not dupes, f"duplicate YAML keys: {sorted(dupes)}"
        return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

    StrictLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)
    with REGISTRY_PATH.open(encoding="utf-8") as fh:
        yaml.load(fh, Loader=StrictLoader)


def test_metadata_counts_consistent(registry, source_fields):
    meta = registry["metadata"]
    assert meta["included_field_count"] == len(registry["fields"])
    assert meta["source_field_count"] == len(source_fields)


# --------------------------------------------------------------------------- #
# Source-registry integrity
# --------------------------------------------------------------------------- #


def test_every_field_exists_in_source_registry(registry, source_fields):
    missing = sorted(set(registry["fields"]) - set(source_fields))
    assert not missing, f"registry fields not in source registry: {missing}"


def test_source_field_matches_key(registry):
    for name, entry in registry["fields"].items():
        assert entry["source_field"] == name


# --------------------------------------------------------------------------- #
# Controlled taxonomy
# --------------------------------------------------------------------------- #

REQUIRED_ENTRY_KEYS = {
    "source_field", "display_name", "analytical_concept", "analytical_role",
    "temporality", "categories", "workflow_tags", "directionality",
    "default_aggregation", "weight_field", "share_basis",
    "portfolio_comparability", "supports_materiality_assessment",
    "asset_applicability", "confidence", "rationale",
}

ALLOWED_DIRECTIONALITY = {
    "higher_is_worse", "lower_is_worse", "higher_is_better",
    "lower_is_better", "neutral", "context_dependent",
}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
ALLOWED_WORKFLOW_TAGS = {
    "period_change", "portfolio_comparison", "ranking", "monitoring",
}
ALLOWED_ANALYTICAL_ROLES = {
    "measure", "dimension", "derived_input", "supporting_attribute",
}
ALLOWED_TEMPORALITY = {
    "point_in_time", "period_flow", "cumulative", "static_baseline",
}
ALLOWED_PORTFOLIO_COMPARABILITY = {
    "comparable", "requires_scale_alignment", "within_asset_class_only",
    "not_comparable",
}


def test_entries_have_required_keys(registry):
    for name, entry in registry["fields"].items():
        missing = REQUIRED_ENTRY_KEYS - set(entry)
        assert not missing, f"{name}: missing keys {sorted(missing)}"


def test_taxonomy_lists_match_builder(registry, builder):
    """The taxonomy embedded in the YAML is the builder's controlled taxonomy."""
    embedded = registry["metadata"]["taxonomy"]
    expected = {k: sorted(v) for k, v in builder.TAXONOMY.items()}
    assert embedded == expected


def test_controlled_taxonomy_values_only(registry):
    tx = registry["metadata"]["taxonomy"]
    for name, entry in registry["fields"].items():
        assert entry["analytical_concept"] in tx["analytical_concepts"], name
        assert entry["categories"], f"{name}: at least one category required"
        for c in entry["categories"]:
            assert c in tx["categories"], f"{name}: bad category {c!r}"
        for t in entry["workflow_tags"]:
            assert t in tx["workflow_tags"], f"{name}: bad tag {t!r}"
        assert entry["default_aggregation"] in tx["default_aggregations"], name
        assert entry["analytical_role"] in tx["analytical_roles"], name
        assert entry["temporality"] in tx["temporality"], name
        assert entry["portfolio_comparability"] in tx["portfolio_comparability"], name
        for a in entry["asset_applicability"]:
            assert a in tx["asset_applicability"], f"{name}: bad asset {a!r}"


def test_valid_directionality_values(registry):
    for name, entry in registry["fields"].items():
        assert entry["directionality"] in ALLOWED_DIRECTIONALITY, (
            f"{name}: invalid directionality {entry['directionality']!r}")


def test_valid_confidence_values(registry):
    for name, entry in registry["fields"].items():
        assert entry["confidence"] in ALLOWED_CONFIDENCE, (
            f"{name}: invalid confidence {entry['confidence']!r}")


def test_workflow_tags_use_only_approved_values(registry):
    for name, entry in registry["fields"].items():
        tags = entry["workflow_tags"]
        assert tags, f"{name}: at least one workflow tag expected"
        bad = set(tags) - ALLOWED_WORKFLOW_TAGS
        assert not bad, f"{name}: unapproved workflow tags {sorted(bad)}"
        assert len(tags) == len(set(tags)), f"{name}: duplicate workflow tags"


def test_every_field_has_rationale(registry):
    for name, entry in registry["fields"].items():
        assert isinstance(entry["rationale"], str), name
        assert entry["rationale"].strip(), f"{name}: empty rationale"


def test_every_field_has_concept_and_category(registry):
    for name, entry in registry["fields"].items():
        assert entry["analytical_concept"], f"{name}: concept required"
        assert len(entry["categories"]) >= 1, f"{name}: category required"


# --------------------------------------------------------------------------- #
# Exclusions
# --------------------------------------------------------------------------- #

# Representative fields that must NEVER appear in the Business Semantics
# Registry: identifiers, names, addresses, free text, provenance/audit,
# ingestion metadata, technical control fields, non-analytical dates.
KNOWN_EXCLUSIONS = [
    # identifiers / references
    "loan_identifier", "unique_identifier", "borrower_identifier",
    "underlying_exposure_identifier", "pool_identifier",
    "originator_legal_entity_identifier",
    "international_securities_identification_number",
    "source_portfolio_id",
    # names / addresses / free text
    "borrower_legal_name", "property_name", "property_address",
    "description", "sponsor", "tenant_name", "seller_name",
    "prepayment_terms_description",
    # provenance / audit / ingestion metadata
    "risk_model_version", "risk_model_source", "source_portfolio_label",
    "data_cut_off_date",
    # technical control / structural fields
    "day_count_convention", "rounding_increment", "waterfall_type",
    "currency_swap_notional", "noteholder_consent",
    # non-analytical dates and static descriptive facts
    "payment_date", "borrower_1_DOB", "net_square_metres",
    "geographic_region_classification",
]


def test_excluded_fields_do_not_appear(registry):
    present = [f for f in KNOWN_EXCLUSIONS if f in registry["fields"]]
    assert not present, f"excluded fields present in registry: {present}"


def test_uncertain_pending_review_fields_not_in_registry(registry, builder):
    for name, info in builder.UNCERTAIN.items():
        if info["status"] == "excluded_pending_review":
            assert name not in registry["fields"], (
                f"{name} is excluded pending review but present in registry")
        else:
            assert name in registry["fields"], (
                f"{name} is included_ambiguous but missing from registry")


# --------------------------------------------------------------------------- #
# Schema v2: versioning, analytical_role, temporality
# --------------------------------------------------------------------------- #


def test_schema_version_and_content_version_are_separate(registry):
    meta = registry["metadata"]
    assert meta["schema_version"] == 2
    assert isinstance(meta["version"], str) and meta["version"], (
        "content version must be retained alongside schema_version")


def test_every_entry_has_valid_analytical_role(registry):
    for name, entry in registry["fields"].items():
        assert entry["analytical_role"] in ALLOWED_ANALYTICAL_ROLES, (
            f"{name}: invalid analytical_role {entry['analytical_role']!r}")


def test_every_entry_has_valid_temporality(registry):
    for name, entry in registry["fields"].items():
        assert entry["temporality"] in ALLOWED_TEMPORALITY, (
            f"{name}: invalid temporality {entry['temporality']!r}")


DERIVED_INPUT_FIELDS = [
    "pd_previous", "pd_current", "lgd_previous", "lgd_current",
    "risk_grade_previous", "risk_grade_current",
    "ifrs9_stage_previous", "ifrs9_stage_current",
    "days_in_arrears_prior", "prior_principal_balances",
]


def test_snapshot_pair_fields_are_derived_inputs(registry):
    for name in DERIVED_INPUT_FIELDS:
        entry = registry["fields"][name]
        assert entry["analytical_role"] == "derived_input", (
            f"{name}: migration/previous-period input must be derived_input, "
            f"not a standalone {entry['analytical_role']}")


def test_cumulative_fields_identified_as_cumulative(registry):
    named_cumulative = [n for n in registry["fields"]
                        if n.startswith("cumulative_")]
    assert named_cumulative, "expected cumulative_* fields in the registry"
    for name in named_cumulative + ["allocated_losses"]:
        entry = registry["fields"][name]
        assert entry["temporality"] == "cumulative", (
            f"{name}: cumulative series must be marked cumulative so "
            f"period-change workflows difference it first")


def test_period_flows_are_not_marked_cumulative_or_static(registry):
    for name, entry in registry["fields"].items():
        if "_in_period" in name or "_in_current_period" in name:
            assert entry["temporality"] == "period_flow", (
                f"{name}: in-period flow must be period_flow")


def test_original_and_securitisation_baselines_are_static(registry):
    baselines = [n for n in registry["fields"]
                 if n.startswith("original_") or "securitisation" in n]
    assert baselines, "expected original_*/securitisation baseline fields"
    for name in baselines:
        entry = registry["fields"][name]
        assert entry["temporality"] == "static_baseline", (
            f"{name}: origination/securitisation-anchored value must be "
            f"static_baseline")
        assert "period_change" not in entry["workflow_tags"], (
            f"{name}: a static baseline cannot support period-change")


# --------------------------------------------------------------------------- #
# Schema v2: weight_field / share_basis
# --------------------------------------------------------------------------- #


def test_weighted_averages_have_canonical_weight_field(registry, source_fields):
    for name, entry in registry["fields"].items():
        if entry["default_aggregation"] == "weighted_average":
            weight = entry["weight_field"]
            assert weight, f"{name}: weighted_average requires weight_field"
            assert weight in source_fields, (
                f"{name}: weight_field {weight!r} is not a canonical field")
        else:
            assert entry["weight_field"] is None, (
                f"{name}: weight_field must be null for "
                f"{entry['default_aggregation']}")


def test_share_metrics_have_explicit_share_basis(registry, source_fields):
    for name, entry in registry["fields"].items():
        if entry["default_aggregation"] == "share":
            basis = entry["share_basis"]
            assert basis == "count" or basis in source_fields, (
                f"{name}: share_basis must be 'count' or a canonical weight "
                f"field, got {basis!r}")
        else:
            assert entry["share_basis"] is None, (
                f"{name}: share_basis must be null for non-share entries")


# --------------------------------------------------------------------------- #
# Schema v2: portfolio comparability
# --------------------------------------------------------------------------- #

INTERNAL_SCALE_FIELDS = [
    "internal_risk_grade", "internal_risk_score", "internal_risk_stage",
    "bank_internal_rating",
    "bank_internal_loss_given_default_lgd_estimate",
    "bank_internal_loss_given_default_lgd_estimate_down_turn",
    "corporate_guarantor_bank_internal_1_year_probability_default",
    "servicer_watchlist_code",
]


def test_valid_portfolio_comparability_values(registry):
    for name, entry in registry["fields"].items():
        assert entry["portfolio_comparability"] in \
            ALLOWED_PORTFOLIO_COMPARABILITY, (
                f"{name}: invalid portfolio_comparability "
                f"{entry['portfolio_comparability']!r}")
        assert "comparable_across_portfolios" not in entry, (
            f"{name}: legacy boolean must be replaced by "
            f"portfolio_comparability")


def test_internal_scales_require_alignment(registry):
    for name in INTERNAL_SCALE_FIELDS:
        entry = registry["fields"][name]
        assert entry["portfolio_comparability"] == "requires_scale_alignment", (
            f"{name}: lender-specific internal scale must not be marked "
            f"unconditionally comparable")


def test_derived_inputs_are_not_comparable(registry):
    for name in DERIVED_INPUT_FIELDS:
        entry = registry["fields"][name]
        assert entry["portfolio_comparability"] == "not_comparable", (
            f"{name}: derived inputs are not standalone comparables")


# --------------------------------------------------------------------------- #
# Schema v2: concentration re-homing
# --------------------------------------------------------------------------- #

FORMER_CONCENTRATION_DIMENSIONS = [
    "geographic_region_obligor", "geographic_region_collateral",
    "collateral_geography", "postcode", "borrower_jurisdiction",
    "broker_channel", "origination_channel", "originator_name",
    "servicer_name", "product_type", "erm_product_type",
    "erm_sub_product_type", "debt_type", "asset_type", "purpose",
    "exposure_currency_denomination", "source_portfolio_type",
    "enterprise_size", "obligor_basel_iii_segment",
    "borrower_basel_iii_segment", "nace_industry_code", "customer_type",
]


def test_concentration_is_not_a_primary_concept(registry):
    assert "concentration" not in \
        registry["metadata"]["taxonomy"]["analytical_concepts"]
    offenders = [n for n, e in registry["fields"].items()
                 if e["analytical_concept"] == "concentration"]
    assert not offenders, (
        f"concentration must not be a primary concept: {offenders}")


def test_former_concentration_dimensions_rehomed(registry):
    for name in FORMER_CONCENTRATION_DIMENSIONS:
        entry = registry["fields"][name]
        assert entry["analytical_concept"] != "concentration", name
        assert entry["analytical_role"] == "dimension", (
            f"{name}: concentration suitability must follow from "
            f"analytical_role: dimension")


# --------------------------------------------------------------------------- #
# Determinism / governance
# --------------------------------------------------------------------------- #


def test_registry_generation_is_deterministic(builder):
    first = builder.dump_registry(builder.build_registry(SOURCE_PATH))
    second = builder.dump_registry(builder.build_registry(SOURCE_PATH))
    assert first == second


def test_committed_registry_matches_regeneration(builder):
    regenerated = builder.dump_registry(builder.build_registry(SOURCE_PATH))
    committed = REGISTRY_PATH.read_text(encoding="utf-8")
    assert committed == regenerated, (
        "config/business_semantics_registry.yaml is stale — regenerate with "
        "scripts/build_business_semantics_registry.py --force")


def test_refuses_to_overwrite_reviewed_output_without_force(builder, tmp_path):
    out = tmp_path / "business_semantics_registry.yaml"
    text = builder.dump_registry(builder.build_registry(SOURCE_PATH))
    out.write_text("reviewed: human decision\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        builder.write_registry(text, out, force=False)
    assert out.read_text(encoding="utf-8") == "reviewed: human decision\n"
    builder.write_registry(text, out, force=True)
    assert out.read_text(encoding="utf-8") == text


def test_identical_content_write_is_allowed_without_force(builder, tmp_path):
    out = tmp_path / "business_semantics_registry.yaml"
    text = builder.dump_registry(builder.build_registry(SOURCE_PATH))
    out.write_text(text, encoding="utf-8")
    builder.write_registry(text, out, force=False)  # no-op, must not raise
    assert out.read_text(encoding="utf-8") == text
