"""The governed metric registry: schema-valid, unique, closed, and incapable
of accepting an arbitrary formula."""

from __future__ import annotations

import pytest
import yaml

from mi_agent.concentration_tests.library import (
    CATEGORIES,
    DEFAULT_LIBRARY_PATH,
    KNOWN_EVALUATORS,
    LibraryError,
    load_library,
    validate_registry,
)
from mi_agent.concentration_tests.metrics import EVALUATORS
from mi_agent.concentration_tests.models import OPERATORS


class TestRegistrySchema:
    def test_the_shipped_registry_is_valid(self, lib):
        assert lib.metrics, "the registry must not be empty"
        assert lib.library_version

    def test_metric_ids_are_unique(self):
        doc = yaml.safe_load(DEFAULT_LIBRARY_PATH.read_text(encoding="utf-8"))
        ids = [m["metric_id"] for m in doc["metrics"]]
        assert len(ids) == len(set(ids))

    def test_every_declared_evaluator_is_registered_in_code(self, lib):
        for m in lib.all():
            if m.implementation_status in ("implemented", "interface_only"):
                assert m.evaluator in EVALUATORS, m.metric_id

    def test_known_evaluators_constant_matches_the_code(self):
        assert set(KNOWN_EVALUATORS) == set(EVALUATORS)

    def test_categories_and_operators_are_closed(self, lib):
        for m in lib.all():
            assert m.category in CATEGORIES
            assert m.supported_operators
            for op in m.supported_operators:
                assert op in OPERATORS

    def test_required_roles_are_declared(self, lib):
        for m in lib.all():
            for role in list(m.required_roles) + list(m.optional_roles):
                assert lib.role(role) is not None, (m.metric_id, role)

    def test_meaningful_initial_coverage(self, lib):
        by_cat = {c: lib.by_category(c) for c in CATEGORIES}
        for category in ("geography", "property_value", "loan_balance",
                        "borrower", "rate_product", "ltv", "performance",
                        "composition", "external_index"):
            assert by_cat[category], f"no metrics in {category}"
        assert any(m.implementation_status == "interface_only"
                   for m in by_cat["external_index"]), \
            "index tests must be a governed interface, not a simulation"

    def test_validate_registry_reports_defects(self):
        problems = validate_registry({
            "field_roles": {},
            "metrics": [
                {"metric_id": "a", "category": "nope",
                 "implementation_status": "implemented", "evaluator": "magic",
                 "supported_operators": ["max"], "parameters": {}},
                {"metric_id": "a", "category": "geography",
                 "implementation_status": "implemented",
                 "evaluator": "share_of_balance",
                 "supported_operators": ["max"], "parameters": {}},
            ]})
        text = " ".join(problems)
        assert "unknown category" in text
        assert "Duplicate metric_id" in text
        assert "not a registered deterministic evaluator" in text

    def test_a_broken_registry_refuses_to_load(self, tmp_path):
        bad = tmp_path / "lib.yaml"
        bad.write_text("metrics:\n  - metric_id: x\n    category: geography\n"
                       "    implementation_status: implemented\n"
                       "    evaluator: not_a_thing\n"
                       "    supported_operators: [max]\n    parameters: {}\n")
        with pytest.raises(LibraryError):
            load_library(bad, use_cache=False)


class TestParameterValidation:
    def test_valid_parameters_pass(self, lib):
        assert lib.validate_parameters(
            "geo_region_share", {"regions": ["East Anglia"]}) == []

    def test_missing_required_parameter_is_named(self, lib):
        problems = lib.validate_parameters("geo_region_share", {})
        assert any("regions" in p and "missing" in p for p in problems)

    def test_undeclared_parameter_is_rejected(self, lib):
        problems = lib.validate_parameters(
            "geo_region_share", {"regions": ["x"], "formula": "a/b"})
        assert any("not declared" in p for p in problems)

    def test_enum_values_are_enforced(self, lib):
        problems = lib.validate_parameters(
            "property_value_above_share",
            {"amount": 1.0, "value_basis": "made_up"})
        assert any("value_basis" in p for p in problems)

    def test_types_are_enforced(self, lib):
        problems = lib.validate_parameters(
            "balance_above_share", {"amount": "lots"})
        assert any("must be a number" in p for p in problems)
        problems = lib.validate_parameters("top_n_loans_share", {"n": 0})
        assert any("at least 1" in p for p in problems)

    def test_filter_list_rejects_expression_shapes(self, lib):
        problems = lib.validate_parameters(
            "primitive_filtered_share",
            {"filters": [{"role": "region", "values": ["London"],
                          "formula": "balance * 2"}]})
        assert any("formulas and expressions are not accepted" in p
                   for p in problems)

    def test_filter_list_requires_governed_roles(self, lib):
        problems = lib.validate_parameters(
            "primitive_filtered_share",
            {"filters": [{"role": "hacked_column", "values": ["x"]}]})
        assert any("governed field role" in p for p in problems)


class TestAliases:
    def test_aliases_resolve_longest_first(self, lib):
        hits = lib.find_by_alias(
            "the net weighted average coupon must be at least 3.75%")
        assert hits[0][0].metric_id == "rate_net_wac"

    def test_composability_is_declared(self, lib):
        assert lib.get("geo_region_share").composable is True
        assert lib.get("rate_net_wac").composable is False
