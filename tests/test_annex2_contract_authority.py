#!/usr/bin/env python3
"""tests/test_annex2_contract_authority.py — one Annex 2 truth, derived.

Annex 2 requirements used to be written down twice: once in the authoritative
sources, and once in ``config/regime/annex2_delivery_rules.yaml``, which five
pipeline stages read in preference to them. Where the two disagreed the
hand-written copy won — so a field could be optional at Gate 4b and mandatory at
Gate 5, and a canonical field could report as one ESMA code in validation and
another in delivery.

These tests pin the properties that make a single derived contract trustworthy:
each fact comes from the layer that owns it, values come from the layers that own
values, and the two gates that decide completeness cannot disagree.

Run: python -m pytest tests/test_annex2_contract_authority.py
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine.regime_contract import workbook_index as wb          # noqa: E402
from engine.regime_contract.annex2_contract import (              # noqa: E402
    ASSET, OPERATOR, REGISTRY, UNIVERSE, WORKBOOK, XSD,
    as_delivery_rules, build_contract, contract)

_UNIVERSE = _REPO / "config" / "regime" / "annex2_field_universe.yaml"
_REGISTRY = _REPO / "config" / "system" / "fields_registry.yaml"
_ASSET_PACK = _REPO / "config" / "asset" / "product_defaults_ERM.yaml"
_LEGACY = _REPO / "config" / "regime" / "annex2_delivery_rules.yaml"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


class TestTheContractIsComplete(unittest.TestCase):
    """1. Every Annex 2 field the regulator defines is in the contract."""

    def test_all_107_fields_are_present(self):
        universe = _yaml(_UNIVERSE)["fields"]
        self.assertEqual(len(universe), 107)
        self.assertEqual(sorted(contract().fields), sorted(universe))

    def test_no_code_is_invented(self):
        universe = set(_yaml(_UNIVERSE)["fields"])
        self.assertTrue(set(contract().fields) <= universe,
                        "the contract must not carry a code the regulator's "
                        "own field universe does not define")


class TestEachFactComesFromItsOwner(unittest.TestCase):
    """2-8. Provenance, per property."""

    @classmethod
    def setUpClass(cls):
        cls.c = contract()

    def test_canonical_binding_comes_from_the_registry(self):
        registry = _yaml(_REGISTRY)["fields"]
        expected = {}
        for name, spec in registry.items():
            code = (((spec or {}).get("regime_mapping") or {})
                    .get("ESMA_Annex2") or {}).get("code")
            if code:
                expected.setdefault(code, name)
        for code, fc in self.c.fields.items():
            self.assertEqual(fc.canonical_field, expected.get(code, ""), code)
            self.assertEqual(fc.provenance.get("canonical_field"), REGISTRY, code)

    def test_the_two_known_misbindings_are_gone(self):
        """The defects the hand-written rules carried, named explicitly."""
        self.assertEqual(self.c["RREL65"].canonical_field, "date_of_restructuring")
        self.assertEqual(self.c["RREC7"].canonical_field, "occupancy_type")

    def test_mandatory_comes_from_workbook_multiplicity(self):
        for code, fc in self.c.fields.items():
            self.assertEqual(fc.mandatory, wb.is_mandatory(code), code)
            self.assertEqual(fc.provenance.get("mandatory"), WORKBOOK, code)

    def test_no_data_permissions_come_from_the_field_universe(self):
        universe = _yaml(_UNIVERSE)["fields"]
        for code, fc in self.c.fields.items():
            meta = universe[code] or {}
            expected = []
            if meta.get("nd1_4_allowed"):
                expected += ["ND1", "ND2", "ND3", "ND4"]
            if meta.get("nd5_allowed"):
                expected += ["ND5"]
            self.assertEqual(list(fc.nd_allowed), expected, code)
            self.assertEqual(fc.provenance.get("nd_allowed"), UNIVERSE, code)

    def test_enum_identities_come_from_the_xsd(self):
        graded = [c for c, fc in self.c.fields.items() if fc.enum_values]
        self.assertGreater(len(graded), 20)
        for code in graded:
            fc = self.c[code]
            self.assertEqual(fc.provenance.get("enum_values"), XSD, code)
            for value in fc.enum_values:
                self.assertEqual(fc.enum_map.get(value), value,
                                 f"{code}: the regulator's own code {value} must "
                                 f"be accepted verbatim")

    def test_a_valid_esma_code_is_never_rejected(self):
        """The hand-written maps refused 67 codes the schema defines."""
        for code, fc in self.c.fields.items():
            missing = [v for v in fc.enum_values if v not in fc.enum_map]
            self.assertEqual(missing, [], f"{code} would reject {missing}")

    def test_source_word_mappings_come_from_enum_configuration(self):
        """A lender's own word for a code is configuration, not the schema."""
        syn = _yaml(_REPO / "config" / "system" / "enum_synonyms.yaml")
        pairs = (syn.get("purpose") or {}).get("manual") or {}
        self.assertTrue(pairs, "the synonym layer must carry purpose mappings")
        emap = self.c["RREL27"].enum_map
        for lender_word, code in pairs.items():
            if code in self.c["RREL27"].enum_values:
                self.assertEqual(emap.get(lender_word), code, lender_word)

    def test_precision_is_never_narrower_than_the_schema(self):
        from engine.regime_contract.annex2_contract import _facet_dict
        for code, fc in self.c.fields.items():
            if not fc.precision:
                continue
            _t, facets = _facet_dict(code)
            self.assertEqual(fc.precision["total_digits"],
                             int(facets["totalDigits"]), code)
            self.assertEqual(fc.precision["fraction_digits"],
                             int(facets.get("fractionDigits") or 0), code)
            self.assertEqual(fc.provenance.get("precision"), XSD, code)

    def test_patterns_come_from_the_schema_or_not_at_all(self):
        for code, fc in self.c.fields.items():
            if fc.pattern:
                self.assertEqual(fc.provenance.get("pattern"), XSD, code)


class TestValuesBelongToTheirLayers(unittest.TestCase):
    """9-11. The contract states what is permitted; layers state what is true."""

    def test_the_contract_carries_no_values(self):
        for code, rule in as_delivery_rules()["field_rules"].items():
            self.assertNotIn("default_value", rule, code)
            self.assertNotIn("default_allowed", rule, code)

    def test_product_defaults_are_asset_scoped(self):
        pack = _yaml(_ASSET_PACK)
        self.assertEqual(pack.get("asset_class"), "equity_release")
        self.assertIn("lien", pack["defaults"])
        self.assertIn("revision_margin_1", pack["nd_defaults"])

    def test_client_facts_are_client_scoped(self):
        from operations_control.annex2.nd_treatments import client_regulatory_values
        cfg = _yaml(_REPO / "config" / "client" / "config_client_ERM_UK.yaml")
        values = client_regulatory_values(cfg.get("defaults") or {})
        self.assertIn("originator_name", values)
        self.assertIn("originator_legal_entity_identifier", values)
        # ...and nowhere else. A lender's identity is not a product fact.
        pack = _yaml(_ASSET_PACK)
        for field in values:
            self.assertNotIn(field, pack.get("defaults", {}))
            self.assertNotIn(field, pack.get("nd_defaults", {}))

    def test_an_operator_translation_cannot_widen_the_regulator_vocabulary(self):
        widened = build_contract(operator_translations={
            "account_status": {"Live": "PERF", "Anything": "NOT_A_CODE"}})
        emap = widened["RREL69"].enum_map
        self.assertEqual(emap.get("Live"), "PERF",
                         "an approved translation of the lender's word applies")
        self.assertNotIn("Anything", emap,
                         "a target the schema does not define is dropped")
        self.assertEqual(widened["RREL69"].provenance.get("enum_map"), OPERATOR)


class TestGate4bAndGate5CannotDisagree(unittest.TestCase):
    """12. The invariant this whole migration exists to establish.

    Gate 5 refuses a mandatory element that is blank. Gate 4b must already have
    said so — otherwise a delivery passes preparation and is refused at the last
    step, which is exactly what happened before: Gate 4b reported one issue on a
    frame Gate 5 refused twenty-one times.
    """

    def test_mandatory_agrees_with_the_builder_on_both_branches(self):
        """Computed exactly as the builder computes it, branch by branch.

        Gate 5 selects an XML branch for the value in hand and then reads THAT
        branch's multiplicity — so a code with an optional branch somewhere can
        still be refused when the branch chosen for an empty value is [1..1].
        Taking the minimum across all branches, as an earlier reading did, makes
        Gate 4b call such a field optional and hands Gate 5 a frame it refuses.
        """
        from engine.gate_5_delivery.xml_builder_annex2 import (
            load_mapping_specs, select_specs_for_value, _parse_multiplicity,
            RECORD_ANCHOR)
        workbook = str(wb.WORKBOOK)
        for mode in ("PRF", "NPRF"):
            c = contract(mode)
            specs_by_code = load_mapping_specs(workbook, wb.SHEET, mode)
            for code, specs in specs_by_code.items():
                if not code.startswith(("RREL", "RREC")):
                    continue
                record = [s for s in specs if RECORD_ANCHOR in s.path]
                chosen = select_specs_for_value(record or specs, "")
                builder_mandatory = bool(chosen) and _parse_multiplicity(
                    chosen[0].multiplicity)[0] >= 1
                self.assertEqual(
                    c[code].mandatory, builder_mandatory,
                    f"{code} in {mode}: Gate 4b says "
                    f"mandatory={c[code].mandatory}, the builder would "
                    f"{'refuse' if builder_mandatory else 'accept'} a blank")

    def test_gate_4b_requires_nothing_the_schema_permits_to_be_absent(self):
        for mode in ("PRF", "NPRF"):
            c = contract(mode)
            for code in c.mandatory_codes():
                self.assertTrue(wb.is_mandatory(code, mode),
                                f"{code} is required by Gate 4b in {mode} but the "
                                f"workbook permits it to be absent")

    def test_a_blank_mandatory_field_is_an_error_before_xml_generation(self):
        import importlib.util
        import pandas as pd
        spec = importlib.util.spec_from_file_location(
            "_norm_inv",
            _REPO / "engine" / "gate_4b_delivery" / "annex2_delivery_normalizer.py")
        norm = importlib.util.module_from_spec(spec)
        sys.modules["_norm_inv"] = norm
        spec.loader.exec_module(norm)
        rules = as_delivery_rules()
        # RREL2 is mandatory and admits no no-data code: blank is fatal.
        self.assertTrue(rules["field_rules"]["RREL2"]["mandatory"])
        frame = pd.DataFrame({"RREL2": ["", ""]})
        _out, issues, summary = norm.normalize_delivery(
            frame, {"field_rules": {"RREL2": rules["field_rules"]["RREL2"]}})
        self.assertEqual(summary["preflight"]["status"], "FAIL")
        self.assertTrue(any(i.issue_type == "mandatory_missing" for i in issues))


class TestTheLegacyFileIsRetired(unittest.TestCase):
    """14. No production consumer reads the retired rules file."""

    #: Directories that RUN — the production route, plus the analysis scripts,
    #: the simulator, the demo platform and the deployment manifest, each of
    #: which used to read the retired file and would now fail on a missing path.
    PRODUCTION = ("engine", "operations_control", "apps", "mi_agent_api",
                  "mi_agent_operator", "trakt_core", "regulatory_watch",
                  "scripts", "simulation", "demo_platform", "deploy",
                  "analytics_lib", "snapshot")

    def test_the_file_is_gone(self):
        self.assertFalse(_LEGACY.exists(),
                         "annex2_delivery_rules.yaml has been retired")

    def test_no_production_module_names_it(self):
        hits = []
        for pkg in self.PRODUCTION:
            root = _REPO / pkg
            if not root.exists():
                continue
            out = subprocess.run(
                ["grep", "-rl", "annex2_delivery_rules", str(root),
                 "--include=*.py", "--include=*.yaml", "--include=*.json"],
                capture_output=True, text=True)
            hits += [ln for ln in out.stdout.splitlines()
                     if ln.strip() and "__pycache__" not in ln]
        self.assertEqual(hits, [],
                         "a production module still references the retired file")


if __name__ == "__main__":
    unittest.main(verbosity=2)
