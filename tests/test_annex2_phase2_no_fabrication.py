#!/usr/bin/env python3
"""tests/test_annex2_phase2_no_fabrication.py — Phase 2: nothing is invented.

Phase 2 removed the one fabrication path in the Annex 2 delivery tail and moved
the RREL20/RREL21 no-data answer out of builder code into governed delivery
rules. Two properties matter and both are pinned here:

1. **The builder invents nothing.** A value it cannot place in a typed branch is
   routed to NoData where the mapping permits one, and otherwise the run fails
   visibly. It is never replaced with a plausible-looking substitute.
2. **The regulatory answer is configuration.** "There is no secondary obligor,
   so this is not applicable" is now a readable, reviewable rule in
   ``config/regime/annex2_delivery_rules.yaml`` — not a hardcoded line in an
   XML builder.

Output neutrality is asserted end-to-end by the 105/107 benchmark
(``docs/annex2_delivery_migration.md``), whose XML SHA-256 is unchanged.

Run: python -m pytest tests/test_annex2_phase2_no_fabrication.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_NORMALIZER = _REPO / "engine" / "gate_4b_delivery" / "annex2_delivery_normalizer.py"
_BUILDER = _REPO / "engine" / "gate_5_delivery" / "xml_builder_annex2.py"
_RULES = _REPO / "config" / "regime" / "annex2_delivery_rules.yaml"
_XSD = _REPO / "config" / "system" / "DRAFT1auth.099.001.04_1.3.0.xsd"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


NORM = _load(_NORMALIZER, "_phase2_normalizer")
BUILD = _load(_BUILDER, "_phase2_builder")


# --------------------------------------------------------------------------- #
# 1. RREL12 — the fabrication is gone
# --------------------------------------------------------------------------- #
class TestRrel12IsNeverFabricated(unittest.TestCase):
    """`_coerce_record_value_for_branch` must not invent a year."""

    def setUp(self):
        BUILD._INSTR.reset()

    def tearDown(self):
        BUILD._INSTR.reset()

    def test_a_valid_iso_year_passes_through_unchanged(self):
        self.assertEqual(BUILD._coerce_record_value_for_branch("RREL12", "2021"),
                         "2021")
        report = BUILD._INSTR.to_dict()
        self.assertEqual(report["routed_to_nodata"]["count"], 0)
        self.assertEqual(report["coercions"]["count"], 0)

    def test_an_invalid_value_never_becomes_2026(self):
        """The exact defect, pinned."""
        for bad in ("not-a-year", "20211", "ND5", "21", "two thousand"):
            BUILD._INSTR.reset()
            result = BUILD._coerce_record_value_for_branch("RREL12", bad)
            self.assertNotEqual(result, "2026",
                                f"{bad!r} was fabricated into a year")
            self.assertEqual(result, "",
                             "an unplaceable value must be routed, not altered")

    def test_the_hardcoded_year_is_gone_from_the_source(self):
        source = _BUILDER.read_text(encoding="utf-8")
        self.assertNotIn('return "2026"', source)

    def test_routing_is_recorded_with_its_reason(self):
        BUILD._coerce_record_value_for_branch("RREL12", "not-a-year")
        report = BUILD._INSTR.to_dict()
        self.assertEqual(report["routed_to_nodata"]["count"], 1)
        record = report["routed_to_nodata"]["records"][0]
        self.assertEqual(record["field_code"], "RREL12")
        self.assertEqual(record["original_value"], "not-a-year")
        self.assertEqual(record["resulting_value"], "")
        self.assertIn("NoData", record["routed_to"])
        self.assertIn("never replaced with a fabricated year", record["reason"])

    def test_zero_routing_is_stated_explicitly(self):
        report = BUILD._INSTR.to_dict()
        self.assertEqual(report["routed_to_nodata"]["count"], 0)
        self.assertEqual(report["routed_to_nodata"]["records"], [])

    def test_the_report_asserts_zero_fabricated_values(self):
        self.assertEqual(BUILD._INSTR.to_dict()["fabricated_values"]["count"], 0)

    def test_the_zero_claim_is_scoped_to_the_builder(self):
        """The note must not claim more than Gate 5 can know.

        Gate 4b applies its own declared transforms. A Gate 5 report asserting
        "no value anywhere was fabricated" would be a claim it has no standing
        to make — see the RREL12 `geography_map` recorded in
        docs/annex2_delivery_migration.md.
        """
        note = BUILD._INSTR.to_dict()["fabricated_values"]["note"]
        self.assertIn("BUILDER", note)
        self.assertIn("Gate 5 only", note)

    def test_other_codes_are_untouched(self):
        self.assertEqual(BUILD._coerce_record_value_for_branch("RREL9", "text"),
                         "text")
        self.assertEqual(BUILD._INSTR.to_dict()["routed_to_nodata"]["count"], 0)

    def test_an_empty_value_is_not_treated_as_a_routing_event(self):
        self.assertEqual(BUILD._coerce_record_value_for_branch("RREL12", ""), "")
        self.assertEqual(BUILD._INSTR.to_dict()["routed_to_nodata"]["count"], 0)


class TestUnplaceableValuesFailVisibly(unittest.TestCase):
    """Where no branch accepts a value, the run must stop, not improvise."""

    def test_a_mandatory_field_with_no_valid_branch_raises(self):
        source = _BUILDER.read_text(encoding="utf-8")
        self.assertIn(
            "Mandatory record field '{code}' has no valid mapping branch",
            source.replace('f"Mandatory record field', '"Mandatory record field'))

    def test_the_builder_refuses_a_missing_secondary_income_branch(self):
        """Behavioural: feed the builder the pre-Phase-2 input and it stops.

        Before Phase 2 this exact input produced an ND5 the builder chose for
        itself. It now raises, and the message names the file that fixes it —
        a delivery-rule gap is reported as a delivery-rule gap.
        """
        from lxml import etree

        ns = "urn:test"
        order_index = BUILD.build_order_index({})
        root = etree.Element(f"{{{ns}}}Document")
        record = BUILD.create_new_record_node(
            root, "/Document/Root/UndrlygXpsrRcrd", ns, order_index)
        etree.SubElement(
            etree.SubElement(record, f"{{{ns}}}UndrlygXpsrCmonData"),
            f"{{{ns}}}FinDtls")

        with self.assertRaises(ValueError) as ctx:
            BUILD._ensure_scndry_oblgr_incm_defaults(record, ns, order_index)
        message = str(ctx.exception)
        self.assertIn("RREL20", message)
        self.assertIn("annex2_delivery_rules.yaml", message,
                      "the refusal must point at the governed rule")
        # ...and it must not have written the node it refused to decide.
        self.assertEqual(
            root.xpath("//*[local-name()='NoData' and text()='ND5']"), [])

    def test_the_injection_code_is_gone_not_merely_bypassed(self):
        source = _BUILDER.read_text(encoding="utf-8")
        block = source[source.index("def _ensure_scndry_oblgr_incm_defaults"):]
        block = block[:block.index("\ndef ")]
        self.assertNotIn('nd_leaf.text = "ND5"', block,
                         "the builder still injects an ND code")
        self.assertNotIn("_INSTR.record_nd(", block)


# --------------------------------------------------------------------------- #
# 2. RREL20 / RREL21 — governed rules, correct ND semantics
# --------------------------------------------------------------------------- #
class TestSecondaryIncomeRulesAreDeclared(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = yaml.safe_load(_RULES.read_text(encoding="utf-8"))["field_rules"]

    def test_both_codes_are_declared(self):
        for code in ("RREL20", "RREL21"):
            self.assertIn(code, self.rules, f"{code} is not a governed rule")

    def test_they_default_to_nd5_not_applicable(self):
        """ND5 = 'Not applicable'. There is no secondary obligor."""
        for code in ("RREL20", "RREL21"):
            rule = self.rules[code]
            self.assertTrue(rule["default_allowed"])
            self.assertEqual(rule["default_value"], "ND5")
            self.assertIn("ND5", rule["nd_allowed"])

    def test_the_nd1_versus_nd5_distinction_is_preserved(self):
        """RREL19 is the structural sibling and must stay on ND1.

        Primary income IS applicable and merely unavailable (ND1). Secondary
        income is NOT applicable where there is no secondary obligor (ND5).
        Collapsing the two would misreport why the field is empty.
        """
        self.assertEqual(self.rules["RREL19"]["default_value"], "ND1")
        self.assertEqual(self.rules["RREL20"]["default_value"], "ND5")
        self.assertEqual(self.rules["RREL21"]["default_value"], "ND5")

    def test_the_workbook_semantic_paths_are_correct(self):
        self.assertEqual(self.rules["RREL20"]["workbook_semantic"],
                         "ScndryOblgrIncm/IncmVal")
        self.assertEqual(self.rules["RREL21"]["workbook_semantic"],
                         "ScndryOblgrIncm/Vrfctn")

    def test_the_projected_source_fields_are_named(self):
        self.assertEqual(self.rules["RREL20"]["projected_source_field"],
                         "secondary_income")
        self.assertEqual(self.rules["RREL21"]["projected_source_field"],
                         "secondary_income_verification")

    def test_the_list_field_is_constrained_to_the_workbook_vocabulary(self):
        """RREL21 is a {LIST} field; an unconstrained rule is a latent defect.

        No projection carries `secondary_income_verification` today, so every
        record takes the ND5 default — but a rule that would pass an arbitrary
        value into an enumerated XSD element is one source column away from the
        class of problem Phase 2 removed. The map is the IDENTITY over the
        workbook's own six codes, matching the sibling RREL19: it constrains
        without inventing a translation.
        """
        allowed = {"SCRT", "SCNF", "VRFD", "NVRF", "SCRG", "OTHR"}
        enum_map = (self.rules["RREL21"].get("transform") or {}).get("enum_map") or {}
        self.assertTrue(enum_map, "RREL21 must not be left unconstrained")
        self.assertEqual(set(enum_map), allowed)
        self.assertEqual(set(enum_map.values()), allowed)
        # Identity, not a translation — every key maps to itself.
        self.assertEqual(enum_map, {c: c for c in allowed})
        # The same vocabulary the primary-income sibling already uses.
        self.assertEqual(set(self.rules["RREL19"]["transform"]["enum_map"]), allowed)

    def test_rrel20_is_an_amount_and_needs_no_enum_map(self):
        """RREL20 is IncmVal, a monetary amount — not a {LIST} field."""
        self.assertNotIn("enum_map", self.rules["RREL20"].get("transform") or {})

    def test_they_do_not_enforce_a_projected_source(self):
        """No canonical source exists; requiring one would fail every run."""
        for code in ("RREL20", "RREL21"):
            self.assertFalse(self.rules[code].get("enforce_presence", False))
            self.assertFalse(self.rules[code].get("mandatory", False))

    def test_the_xsd_actually_mandates_the_element(self):
        """The basis for defaulting rather than omitting, read from the schema."""
        text = _XSD.read_text(encoding="utf-8", errors="replace")
        index = text.find('name="ScndryOblgrIncm"')
        self.assertGreater(index, 0, "ScndryOblgrIncm not found in the XSD")
        declaration = text[index - 20:index + 60]
        self.assertNotIn("minOccurs", declaration,
                         "if minOccurs were declared, omission might be lawful")
        block = re.search(
            r'<xs:complexType name="SecondaryIncome2">(.*?)</xs:complexType>',
            text, re.S).group(1)
        for child in ("IncmVal", "Vrfctn"):
            self.assertIn(f'name="{child}"', block)

    def test_no_nd_contamination_of_the_canonical_layer(self):
        """ND codes belong to the regime/delivery layer only."""
        registry = yaml.safe_load(
            (_REPO / "config" / "system" / "fields_registry.yaml")
            .read_text(encoding="utf-8"))["fields"]
        for name in ("secondary_income", "secondary_income_verification"):
            meta = registry.get(name) or {}
            self.assertNotIn("ND5", str(meta.get("default", "")))
            self.assertNotIn("ND5", str(meta.get("allowed_values", "")))


# --------------------------------------------------------------------------- #
# 3. Absent-column defaulting is generic, not field-specific
# --------------------------------------------------------------------------- #
class TestDeclaredDefaultsCreateAbsentColumns(unittest.TestCase):
    """Driven entirely by the rules; no code names RREL20/RREL21."""

    def _rules(self, **extra):
        base = {
            "defaults": {"reporting_year": "2026"},
            "field_rules": {
                "RREL3": {"mandatory": True},
                "ZZZ01": {"mandatory": False, "enforce_presence": False,
                          "default_allowed": True, "default_value": "ND5",
                          "nd_allowed": ["ND5"]},
                "ZZZ02": {"mandatory": False, "enforce_presence": False},
                "ZZZ03": {"mandatory": True, "enforce_presence": True},
            },
        }
        base["field_rules"].update(extra)
        return base

    def _frame(self):
        return pd.DataFrame({"RREL3": ["EXP-1", "EXP-2", "EXP-3"]})

    def test_an_absent_field_with_a_declared_default_is_created(self):
        out, _issues, summary = NORM.normalize_delivery(self._frame(), self._rules())
        self.assertIn("ZZZ01", out.columns)
        self.assertEqual(list(out["ZZZ01"]), ["ND5", "ND5", "ND5"])
        self.assertEqual(summary["columns_created_from_declared_defaults"], ["ZZZ01"])

    def test_an_absent_field_without_a_default_stays_absent(self):
        out, _issues, _summary = NORM.normalize_delivery(self._frame(), self._rules())
        self.assertNotIn("ZZZ02", out.columns,
                         "a field with no authorised default must not appear")

    def test_a_field_enforcing_presence_is_not_silently_defaulted(self):
        out, issues, _summary = NORM.normalize_delivery(self._frame(), self._rules())
        self.assertNotIn("ZZZ03", out.columns)
        self.assertTrue(any(i.issue_type == "missing_field" for i in issues),
                        "a required projected source must still be reported missing")

    def test_the_created_values_are_recorded_as_rule_applied_nd(self):
        _out, _issues, summary = NORM.normalize_delivery(self._frame(), self._rules())
        instr = summary["delivery_instrumentation"]
        self.assertEqual(instr["nd_applied_by_rules"]["by_field"]["ZZZ01"], 3)
        self.assertEqual(instr["nd_applied_by_rules"]["by_code"]["ND5"], 3)
        # ...and NOT counted as input ND, which would misattribute the source.
        self.assertEqual(instr["nd_present_in_input"]["total"], 0)

    def test_no_field_code_is_hard_coded_in_the_normaliser(self):
        source = _NORMALIZER.read_text(encoding="utf-8")
        body = source[source.index("def _add_defaulted_columns"):]
        body = body[:body.index("\ndef ")]
        for code in ("RREL20", "RREL21"):
            self.assertNotIn(f'"{code}"', body,
                             "column creation must be rule-driven, not per-field")

    def test_a_non_nd_default_is_not_miscounted_as_nd(self):
        rules = self._rules(ZZZ04={"mandatory": False, "enforce_presence": False,
                                   "default_allowed": True, "default_value": "OTHR"})
        out, _issues, summary = NORM.normalize_delivery(self._frame(), rules)
        self.assertEqual(list(out["ZZZ04"]), ["OTHR"] * 3)
        self.assertNotIn("ZZZ04",
                         summary["delivery_instrumentation"]["nd_applied_by_rules"]["by_field"])

    def test_the_callers_frame_is_not_mutated(self):
        """Creating a column must not reach back into the caller's data."""
        frame = self._frame()
        before = list(frame.columns)
        NORM.normalize_delivery(frame, self._rules())
        self.assertEqual(list(frame.columns), before,
                         "normalize_delivery must not add columns to its input")

    def test_column_creation_is_deterministic(self):
        first, _i, s1 = NORM.normalize_delivery(self._frame(), self._rules())
        second, _j, s2 = NORM.normalize_delivery(self._frame(), self._rules())
        self.assertEqual(list(first.columns), list(second.columns))
        self.assertEqual(s1["columns_created_from_declared_defaults"],
                         s2["columns_created_from_declared_defaults"])

    def test_the_accounting_still_reconciles(self):
        out, _issues, summary = NORM.normalize_delivery(self._frame(), self._rules())
        instr = summary["delivery_instrumentation"]
        expected = (instr["nd_present_in_input"]["total"]
                    + instr["nd_applied_by_rules"]["total"])
        actual = int(out.astype(str).apply(
            lambda s: s.str.strip().str.upper().str.fullmatch(r"ND[1-5]", na=False)
        ).sum().sum())
        self.assertEqual(actual, expected)


# --------------------------------------------------------------------------- #
# 4. A headline field count must not overstate coverage
# --------------------------------------------------------------------------- #
class TestFieldProvenanceIsReported(unittest.TestCase):
    """105 populated from source + 2 populated by rule = 107 represented.

    Before Phase 2 the two secondary-income codes were absent from the
    delivery-ready CSV and injected by the builder, so a column count of 105
    happened to exclude them. They are now real columns, and a bare count of
    107 would read as *more* client coverage than Phase 1 delivered. It is not:
    the same two fields still carry ND5. The split says so.
    """

    def _rules(self):
        return {
            "defaults": {"reporting_year": "2026"},
            "field_rules": {
                "RREL3": {"mandatory": True},
                "ZZZ01": {"mandatory": False, "enforce_presence": False,
                          "default_allowed": True, "default_value": "ND5",
                          "nd_allowed": ["ND5"]},
                "ZZZ02": {"mandatory": False, "enforce_presence": False,
                          "default_allowed": True, "default_value": "ND5",
                          "nd_allowed": ["ND5"]},
            },
        }

    def _frame(self):
        return pd.DataFrame({"RREL3": ["EXP-1", "EXP-2"], "RREL4": ["A", "B"],
                             "RREL5": ["X", "Y"]})

    def test_the_split_is_reported_and_adds_up(self):
        out, _issues, summary = NORM.normalize_delivery(self._frame(), self._rules())
        prov = summary["field_provenance"]
        self.assertEqual(prov["populated_from_projected_source"], 3)
        self.assertEqual(prov["populated_by_declared_delivery_rule"], 2)
        self.assertEqual(prov["represented_in_submission"], 5)
        self.assertEqual(prov["populated_from_projected_source"]
                         + prov["populated_by_declared_delivery_rule"],
                         prov["represented_in_submission"])
        self.assertEqual(prov["represented_in_submission"], len(out.columns))

    def test_the_rule_populated_fields_are_named_not_just_counted(self):
        _out, _issues, summary = NORM.normalize_delivery(self._frame(), self._rules())
        self.assertEqual(summary["field_provenance"]["declared_rule_fields"],
                         ["ZZZ01", "ZZZ02"])

    def test_a_run_with_no_defaults_reports_zero_not_nothing(self):
        rules = {"defaults": {}, "field_rules": {"RREL3": {"mandatory": True}}}
        _out, _issues, summary = NORM.normalize_delivery(self._frame(), rules)
        prov = summary["field_provenance"]
        self.assertEqual(prov["populated_by_declared_delivery_rule"], 0)
        self.assertEqual(prov["declared_rule_fields"], [])
        self.assertEqual(prov["populated_from_projected_source"], 3)

    def test_the_demo_manifest_carries_the_split_not_only_the_total(self):
        source = (_REPO / "demo_platform" / "artefacts.py").read_text(encoding="utf-8")
        self.assertIn("fieldsFromProjectedSource", source)
        self.assertIn("fieldsFromDeliveryRule", source)

    def test_the_rrel12_geography_map_is_recorded_as_outstanding(self):
        """A known latent issue must stay visible until it is decided.

        The RREL12 rule maps region NAMES to the classification year "2026".
        A classification year is not derivable from a region name, so this is
        the same shape as the builder fabrication Phase 2 removed — but it is
        declared configuration, it is inert on the benchmark (every projected
        RREL12 value is already 2021), and choosing the right behaviour is a
        regulatory decision Phase 2 was scoped out of.

        This test does not require the map to exist or to be removed. It
        requires that, as long as it exists, the migration document still says
        so — so the next phase inherits the finding rather than rediscovering it.
        """
        rules = yaml.safe_load(_RULES.read_text(encoding="utf-8"))
        geo = ((rules["field_rules"]["RREL12"].get("transform") or {})
               .get("geography_map") or {})
        doc = (_REPO / "docs" / "annex2_delivery_migration.md").read_text(
            encoding="utf-8")
        if geo:
            self.assertIn("geography_map", doc,
                          "the RREL12 geography_map is still declared but the "
                          "migration document no longer records it")
            self.assertIn("RREL12", doc)

    def test_the_production_rules_declare_exactly_two_defaulted_codes(self):
        """Pins the benchmark's 105 + 2. A third would change the headline."""
        rules = yaml.safe_load(_RULES.read_text(encoding="utf-8"))
        defaulted = sorted(
            code for code, rule in (rules.get("field_rules") or {}).items()
            if isinstance(rule, dict)
            and rule.get("default_allowed")
            and not rule.get("enforce_presence", rule.get("mandatory", False))
            and str(rule.get("default_value") or "").strip())
        self.assertEqual(defaulted, ["RREL20", "RREL21"],
                         "the set of rule-populated Annex 2 fields changed; "
                         "re-state the benchmark split in "
                         "docs/annex2_delivery_migration.md")


# --------------------------------------------------------------------------- #
# 5. The same input must produce the same delivery-ready CSV and the same report
# --------------------------------------------------------------------------- #
class TestPhase2IsDeterministicAndOutputNeutral(unittest.TestCase):
    """Instrumentation observes; it must never influence.

    End-to-end neutrality is proven by the benchmark SHA-256
    (``a21f8a4c…d685d``, unchanged across Phase 1 and Phase 2). What is checked
    here is the property that makes that reproducible: identical input yields a
    byte-identical CSV and identical counts, run after run.
    """

    def _rules(self):
        return {
            "defaults": {"reporting_year": "2026"},
            "field_rules": {
                "RREL3": {"mandatory": True},
                "RREL20": {"mandatory": False, "enforce_presence": False,
                           "default_allowed": True, "default_value": "ND5",
                           "nd_allowed": ["ND1", "ND5"]},
            },
        }

    def _frame(self):
        return pd.DataFrame({"RREL3": [f"EXP-{i}" for i in range(25)],
                             "RREL4": ["ND1"] * 25})

    def test_two_runs_produce_a_byte_identical_csv(self):
        first, _i, _s = NORM.normalize_delivery(self._frame(), self._rules())
        second, _j, _t = NORM.normalize_delivery(self._frame(), self._rules())
        self.assertEqual(first.to_csv(index=False), second.to_csv(index=False))

    def test_two_runs_produce_identical_instrumentation(self):
        _a, _i, s1 = NORM.normalize_delivery(self._frame(), self._rules())
        _b, _j, s2 = NORM.normalize_delivery(self._frame(), self._rules())
        self.assertEqual(s1["delivery_instrumentation"],
                         s2["delivery_instrumentation"])
        self.assertEqual(s1["field_provenance"], s2["field_provenance"])

    def test_the_nd_accounting_reconciles_against_the_csv_itself(self):
        out, _issues, summary = NORM.normalize_delivery(self._frame(), self._rules())
        instr = summary["delivery_instrumentation"]
        counted = int(out.astype(str).apply(
            lambda s: s.str.strip().str.upper().str.fullmatch(r"ND[1-5]", na=False)
        ).sum().sum())
        self.assertEqual(instr["nd_present_in_input"]["total"], 25)   # RREL4
        self.assertEqual(instr["nd_applied_by_rules"]["total"], 25)   # RREL20
        self.assertEqual(counted, 50,
                         "the split must account for every ND cell, no more")

    def test_the_builder_no_longer_contributes_to_that_total(self):
        """Phase 1 had to add builder ND to reconcile against the XML.

        Phase 2 removed the injection, so the CSV total and the XML total are
        now the same number: the builder adds nothing of its own.
        """
        source = _BUILDER.read_text(encoding="utf-8")
        self.assertEqual(source.count("_INSTR.record_nd("), 1,
                         "only the non-performing historical block may still "
                         "inject ND; RREL20/RREL21 moved to declared rules")
        self.assertIn("_ensure_hstrcl_colltn_nd_defaults", source)

    def test_the_builder_report_states_zero_fabrications_explicitly(self):
        BUILD._INSTR.reset()
        try:
            report = BUILD._INSTR.to_dict()
            self.assertEqual(report["fabricated_values"]["count"], 0)
            self.assertEqual(report["routed_to_nodata"]["count"], 0)
            self.assertEqual(report["routed_to_nodata"]["records"], [])
        finally:
            BUILD._INSTR.reset()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
