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
    "source_field", "display_name", "analytical_concept", "categories",
    "workflow_tags", "directionality", "aggregation_type",
    "comparable_across_portfolios", "supports_materiality_assessment",
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
        assert entry["aggregation_type"] in tx["aggregation_types"], name
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
