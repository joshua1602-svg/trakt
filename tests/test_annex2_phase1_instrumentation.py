#!/usr/bin/env python3
"""tests/test_annex2_phase1_instrumentation.py — Phase 1 delivery visibility.

Phase 1 makes the Annex 2 delivery tail state what it does to values, without
changing a single one. Three things must be separable in the evidence:

    governed rule-declared ND   the delivery rules said to substitute it
    builder-injected ND         the Gate 5 builder inserted it itself
    value coercion              a supplied value was replaced

and a run with none of a thing must SAY zero — an absent key is not evidence
of absence.

The load-bearing property is output-neutrality: instrumentation observes, it
never alters. That is asserted directly here by normalising the same input
twice and comparing bytes, and end-to-end by the committed 105/107 benchmark
(``docs/annex2_delivery_migration.md``) whose XML SHA-256 is unchanged.

Run: python -m pytest tests/test_annex2_phase1_instrumentation.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_NORMALIZER = _REPO / "engine" / "gate_4b_delivery" / "annex2_delivery_normalizer.py"
_BUILDER = _REPO / "engine" / "gate_5_delivery" / "xml_builder_annex2.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


NORM = _load(_NORMALIZER, "_phase1_normalizer")
BUILD = _load(_BUILDER, "_phase1_builder")


class TestInstrumentationSchema(unittest.TestCase):
    """The report shape both stages promise."""

    def test_normaliser_reports_all_three_categories_explicitly(self):
        report = NORM._DeliveryInstrumentation().to_dict()
        self.assertEqual(report["stage"], "gate_4b_delivery_normalisation")
        self.assertEqual(report["schema_version"], 1)
        for key in ("nd_present_in_input", "nd_applied_by_rules", "coercions"):
            self.assertIn(key, report)
        for key in ("nd_present_in_input", "nd_applied_by_rules"):
            self.assertEqual(report[key]["total"], 0)
            self.assertEqual(report[key]["by_code"], {})
            self.assertEqual(report[key]["by_field"], {})

    def test_builder_reports_injection_by_code_field_and_branch(self):
        report = BUILD._DeliveryInstrumentation().to_dict()
        self.assertEqual(report["stage"], "gate_5_xml_builder")
        nd = report["nd_injected_by_builder"]
        for key in ("total", "by_code", "by_field", "by_branch"):
            self.assertIn(key, nd)

    def test_zero_coercions_is_recorded_explicitly_not_omitted(self):
        for module in (NORM, BUILD):
            report = module._DeliveryInstrumentation().to_dict()
            self.assertIn("coercions", report)
            self.assertEqual(report["coercions"]["count"], 0)
            self.assertEqual(report["coercions"]["records"], [])
            self.assertFalse(report["coercions"]["truncated"])

    def test_a_coercion_is_recorded_with_full_provenance(self):
        instr = BUILD._DeliveryInstrumentation()
        instr.record_coercion(field_code="RREL12", original_value="not-a-year",
                              resulting_value="2026", row_identifier="EXP-1",
                              reason="unit test")
        report = instr.to_dict()
        self.assertEqual(report["coercions"]["count"], 1)
        record = report["coercions"]["records"][0]
        for key in ("field_code", "row_identifier", "original_value",
                    "resulting_value", "reason"):
            self.assertIn(key, record)
        self.assertEqual(record["original_value"], "not-a-year")
        self.assertEqual(record["resulting_value"], "2026")

    def test_counts_stay_exact_when_records_are_capped(self):
        instr = BUILD._DeliveryInstrumentation()
        for i in range(BUILD.COERCION_RECORD_CAP + 25):
            instr.record_coercion(field_code="RREL12", original_value=str(i),
                                  resulting_value="2026", reason="unit test")
        report = instr.to_dict()
        self.assertEqual(report["coercions"]["count"],
                         BUILD.COERCION_RECORD_CAP + 25)
        self.assertEqual(len(report["coercions"]["records"]),
                         BUILD.COERCION_RECORD_CAP)
        self.assertTrue(report["coercions"]["truncated"])

    def test_nd_categories_do_not_share_a_counter(self):
        instr = NORM._DeliveryInstrumentation()
        instr.record_nd_from_rule(nd_code="ND5", field="RREL30")
        report = instr.to_dict()
        self.assertEqual(report["nd_applied_by_rules"]["total"], 1)
        self.assertEqual(report["nd_present_in_input"]["total"], 0,
                         "rule-applied ND must not be counted as input ND")


class TestBuilderCoercionIsObservedNotHidden(unittest.TestCase):
    """The RREL12 substitution is still applied — and now visible.

    Phase 1 deliberately does NOT remove the coercion; that is Phase 2. What it
    must not do is leave it silent.
    """

    def test_a_non_year_rrel12_value_is_recorded(self):
        BUILD._INSTR.reset()
        result = BUILD._coerce_record_value_for_branch("RREL12", "not-a-year")
        self.assertEqual(result, "2026", "Phase 1 must not change behaviour")
        report = BUILD._INSTR.to_dict()
        self.assertEqual(report["coercions"]["count"], 1)
        record = report["coercions"]["records"][0]
        self.assertEqual(record["field_code"], "RREL12")
        self.assertEqual(record["original_value"], "not-a-year")
        self.assertEqual(record["resulting_value"], "2026")
        self.assertIn("RREL12", record["reason"])
        BUILD._INSTR.reset()

    def test_a_valid_year_is_not_coerced_and_records_nothing(self):
        BUILD._INSTR.reset()
        self.assertEqual(BUILD._coerce_record_value_for_branch("RREL12", "2021"),
                         "2021")
        self.assertEqual(BUILD._INSTR.to_dict()["coercions"]["count"], 0)

    def test_other_codes_are_untouched(self):
        BUILD._INSTR.reset()
        self.assertEqual(BUILD._coerce_record_value_for_branch("RREL9", "text"),
                         "text")
        self.assertEqual(BUILD._INSTR.to_dict()["coercions"]["count"], 0)


class TestNormaliserOutputNeutrality(unittest.TestCase):
    """Instrumentation must not change the delivery-ready CSV."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="phase1_neutral_"))
        rules = {
            "defaults": {"reporting_year": "2026"},
            "field_rules": {
                "RREL3": {"mandatory": True},
                "RREL30": {"mandatory": False, "default_allowed": True,
                           "default_value": "ND5", "nd_allowed": ["ND5"]},
                "RREL31": {"mandatory": False, "nd_allowed": ["ND1", "ND5"]},
            },
        }
        cls.rules_path = cls.tmp / "rules.yaml"
        cls.rules_path.write_text(yaml.safe_dump(rules), encoding="utf-8")
        cls.frame = pd.DataFrame({
            "RREL3": ["EXP-1", "EXP-2", "EXP-3"],
            "RREL30": ["", "", "ND5"],
            "RREL31": ["ND1", "", "ND5"],
            # A pass-through column with no rule at all.
            "RREC99": ["ND1", "value", "ND5"],
        })
        cls.rules = rules

    def _normalise(self):
        return NORM.normalize_delivery(self.frame.copy(), self.rules)

    def test_repeated_runs_are_deterministic(self):
        first_df, _, first_summary = self._normalise()
        second_df, _, second_summary = self._normalise()
        pd.testing.assert_frame_equal(first_df, second_df)
        self.assertEqual(
            json.dumps(first_summary["delivery_instrumentation"], sort_keys=True),
            json.dumps(second_summary["delivery_instrumentation"], sort_keys=True))

    def test_input_nd_counts_every_column_including_pass_through(self):
        _, _, summary = self._normalise()
        nd_in = summary["delivery_instrumentation"]["nd_present_in_input"]
        # RREL31 ND1+ND5, RREC99 ND1+ND5, RREL30 ND5 = 5 input ND cells.
        self.assertEqual(nd_in["total"], 5)
        self.assertEqual(nd_in["by_code"], {"ND1": 2, "ND5": 3})
        self.assertIn("RREC99", nd_in["by_field"],
                      "a pass-through column's ND must still be counted")

    def test_rule_applied_nd_is_separated_from_input_nd(self):
        _, _, summary = self._normalise()
        instr = summary["delivery_instrumentation"]
        # RREL30 is blank on two rows and defaults to ND5 by declared rule.
        self.assertEqual(instr["nd_applied_by_rules"]["total"], 2)
        self.assertEqual(instr["nd_applied_by_rules"]["by_code"], {"ND5": 2})
        self.assertEqual(instr["nd_applied_by_rules"]["by_field"], {"RREL30": 2})
        # ...and the ND5 already present on row 3 is NOT double counted.
        self.assertEqual(instr["nd_present_in_input"]["by_field"]["RREL30"], 1)

    def test_the_accounting_reconciles_to_the_output(self):
        out_df, _, summary = self._normalise()
        instr = summary["delivery_instrumentation"]
        expected = (instr["nd_present_in_input"]["total"]
                    + instr["nd_applied_by_rules"]["total"])
        actual = int(out_df.astype(str).apply(
            lambda s: s.str.strip().str.upper().str.fullmatch(r"ND[1-5]", na=False)
        ).sum().sum())
        self.assertEqual(actual, expected,
                         "input ND + rule-applied ND must equal the ND in the "
                         "delivery-ready output")

    def test_zero_coercions_reported_when_nothing_was_transformed(self):
        _, _, summary = self._normalise()
        self.assertEqual(
            summary["delivery_instrumentation"]["coercions"]["count"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
