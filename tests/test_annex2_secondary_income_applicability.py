#!/usr/bin/env python3
"""tests/test_annex2_secondary_income_applicability.py — RREL20/RREL21 policy.

Equity-release loans **commonly have joint borrowers**, so a secondary obligor
frequently exists. What is unusual about the product is different: borrower
income — primary and secondary alike — is generally not part of equity-release
underwriting, which runs on age, property value and LTV.

That distinction decides which no-data code is correct, and these tests pin the
properties that must hold whichever code wins:

* a real supplied value is **never** overwritten by a default;
* borrower count alone never determines the code;
* the decision lives in configuration, never in Python;
* canonical truth never carries a regime ND code;
* the two configuration files that express this decision cannot disagree
  silently — if they diverge, this file says so out loud.

The ND1-vs-ND5 determination itself is recorded as OPEN. See the comment block
above RREL20 in ``config/regime/annex2_delivery_rules.yaml`` and
``docs/annex2_delivery_migration.md``.

Run: python -m pytest tests/test_annex2_secondary_income_applicability.py
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_NORMALIZER = _REPO / "engine" / "gate_4b_delivery" / "annex2_delivery_normalizer.py"
_RULES = _REPO / "config" / "regime" / "annex2_delivery_rules.yaml"
_ERM = _REPO / "config" / "asset" / "product_defaults_ERM.yaml"
_STANDARDS = _REPO / "config" / "system" / "standards_library.yaml"
_UNIVERSE = _REPO / "config" / "regime" / "annex2_field_universe.yaml"

_spec = importlib.util.spec_from_file_location("_sec_income_norm", _NORMALIZER)
NORM = importlib.util.module_from_spec(_spec)
sys.modules["_sec_income_norm"] = NORM
_spec.loader.exec_module(NORM)

#: The two codes under review, and the canonical fields behind them.
_CODES = ("RREL20", "RREL21")
_SOURCES = {"RREL20": "secondary_income", "RREL21": "secondary_income_verification"}


def _rules():
    return yaml.safe_load(_RULES.read_text(encoding="utf-8"))


def _erm_nd_defaults():
    return (yaml.safe_load(_ERM.read_text(encoding="utf-8")) or {}).get("nd_defaults") or {}


def _only(*codes, **overrides):
    """A rules dict carrying just the codes under test.

    RREL3 stays a plain frame column (the exposure identifier the normaliser
    uses for row attribution); it needs no rule of its own.
    """
    full = _rules()
    fields = {}
    for code in codes:
        rule = dict(full["field_rules"][code])
        rule.update(overrides.get(code, {}))
        fields[code] = rule
    return {"defaults": full.get("defaults", {}), "field_rules": fields}


# --------------------------------------------------------------------------- #
# 1. The product rationale, stated correctly
# --------------------------------------------------------------------------- #
class TestTheRationaleIsAboutUnderwritingNotBorrowerCount(unittest.TestCase):
    """Joint borrowers exist. The reason must never be "there is no obligor"."""

    def test_no_configuration_claims_equity_release_has_no_secondary_obligor(self):
        for path in (_RULES, _ERM):
            text = path.read_text(encoding="utf-8").lower()
            for phrase in ("there is no secondary obligor",
                           "no secondary obligor exists",
                           "single obligor there is no secondary"):
                self.assertNotIn(phrase, text,
                                 f"{path.name} justifies the code on borrower "
                                 f"absence; equity release has joint borrowers")

    def test_the_rules_state_the_underwriting_rationale(self):
        text = _RULES.read_text(encoding="utf-8").lower()
        self.assertIn("joint borrower", text)
        self.assertIn("underwriting", text)

    def test_nd1_means_not_required_by_underwriting_criteria(self):
        """The definition that makes this decision finely balanced."""
        nd = yaml.safe_load(_STANDARDS.read_text(encoding="utf-8"))["ND_OPTIONS"]
        self.assertIn("not required by the lending or underwriting criteria",
                      nd["ND1"]["description"])
        self.assertEqual(nd["ND5"]["description"], "Not applicable")

    def test_the_standard_setter_forbids_nd5_on_the_primary_income_twin(self):
        """RREL16 carries identical wording and cannot use ND5.

        This is the evidence that "income was not used to underwrite" does not
        by itself make an income field not-applicable.
        """
        u = yaml.safe_load(_UNIVERSE.read_text(encoding="utf-8"))
        fields = u.get("fields") or u
        self.assertIs(fields["RREL16"]["nd5_allowed"], False)
        self.assertIs(fields["RREL19"]["nd5_allowed"], False)
        for code in _CODES:
            self.assertIs(fields[code]["nd5_allowed"], True)
        self.assertIn("used to underwrite", str(fields["RREL16"]["content"]))
        self.assertIn("used to underwrite", str(fields["RREL20"]["content"]))


# --------------------------------------------------------------------------- #
# 2. The two configuration layers must not disagree silently
# --------------------------------------------------------------------------- #
class TestProductDefaultAndDeliveryRuleAgree(unittest.TestCase):
    """A known, recorded divergence — asserted so it cannot go quiet.

    ``product_defaults_ERM.yaml`` applies ND1 across the whole income family;
    the delivery rule currently declares ND5, inherited from the pre-Phase-2
    builder. The delivery path uses the rule, so ND5 is what ships today.

    This test does **not** decide the question. It fails the moment the two
    files agree — at which point the divergence is resolved and this test
    should be replaced by a plain equality assertion.
    """

    def test_the_divergence_is_exactly_where_it_is_documented(self):
        rules = _rules()["field_rules"]
        erm = _erm_nd_defaults()
        observed = {
            code: (rules[code].get("default_value"), erm.get(_SOURCES[code]))
            for code in _CODES
        }
        self.assertEqual(
            observed,
            {"RREL20": ("ND5", "ND1"), "RREL21": ("ND5", "ND1")},
            "the RREL20/RREL21 divergence changed. If it was RESOLVED, replace "
            "this test with an equality assertion and update the comment above "
            "RREL20 in annex2_delivery_rules.yaml plus "
            "docs/annex2_delivery_migration.md. If it MOVED, it must not have "
            "moved silently.")

    def test_the_open_question_is_recorded_next_to_the_rule(self):
        text = _RULES.read_text(encoding="utf-8")
        self.assertIn("UNDER REVIEW", text,
                      "an unresolved regulatory decision must be visible in "
                      "the configuration that carries it")

    def test_the_whole_income_family_is_treated_consistently_by_the_product(self):
        """The ERM product position is uniform ND1 across all income fields."""
        erm = _erm_nd_defaults()
        for field in ("primary_income", "primary_income_type",
                      "primary_income_currency", "primary_income_verification",
                      "secondary_income", "secondary_income_verification"):
            self.assertEqual(erm.get(field), "ND1",
                             f"{field} breaks the product's uniform income "
                             f"treatment; that is a regulatory decision")

    def test_the_product_reserves_nd5_for_features_it_genuinely_lacks(self):
        erm = _erm_nd_defaults()
        for field in ("maturity_date", "scheduled_principal_payment_frequency"):
            self.assertEqual(erm.get(field), "ND5",
                             "ND5 marks a product feature that does not exist")


# --------------------------------------------------------------------------- #
# 3. Behaviour — the properties that hold whichever code wins
# --------------------------------------------------------------------------- #
class TestSuppliedValuesAreNeverOverwritten(unittest.TestCase):
    """A real value always beats a default. This is the load-bearing property."""

    def test_single_borrower_no_income_supplied_takes_the_declared_default(self):
        rules = _only(*_CODES)
        out, _issues, summary = NORM.normalize_delivery(
            pd.DataFrame({"RREL3": ["EXP-SINGLE-1"]}), rules)
        declared = _rules()["field_rules"]["RREL20"]["default_value"]
        self.assertEqual(out.at[0, "RREL20"], declared)
        self.assertEqual(out.at[0, "RREL21"],
                         _rules()["field_rules"]["RREL21"]["default_value"])
        self.assertEqual(summary["field_provenance"]
                         ["populated_by_declared_delivery_rule"], 2)

    def test_joint_borrower_no_income_supplied_takes_the_same_default(self):
        """Borrower count must not change the code by itself.

        Income is outside equity-release underwriting whether there are one or
        two borrowers, so a joint-borrower loan with no income data reports
        exactly what a single-borrower loan reports.
        """
        rules = _only(*_CODES)
        single, _i, _s = NORM.normalize_delivery(
            pd.DataFrame({"RREL3": ["EXP-SINGLE-1"]}), rules)
        joint, _j, _t = NORM.normalize_delivery(
            pd.DataFrame({"RREL3": ["EXP-JOINT-1"]}), rules)
        self.assertEqual(single.at[0, "RREL20"], joint.at[0, "RREL20"])
        self.assertEqual(single.at[0, "RREL21"], joint.at[0, "RREL21"])

    def test_a_real_secondary_income_value_is_preserved(self):
        rules = _only("RREL20")
        frame = pd.DataFrame({"RREL3": ["EXP-JOINT-1", "EXP-JOINT-2"],
                              "RREL20": ["42500.00", ""]})
        out, _issues, _summary = NORM.normalize_delivery(frame, rules)
        self.assertEqual(out.at[0, "RREL20"], "42500.00",
                         "a supplied secondary income was overwritten")
        # The blank row still receives the governed default.
        self.assertEqual(out.at[1, "RREL20"],
                         _rules()["field_rules"]["RREL20"]["default_value"])

    def test_a_real_secondary_income_verification_value_is_preserved(self):
        rules = _only("RREL21")
        frame = pd.DataFrame({"RREL3": ["EXP-JOINT-1", "EXP-JOINT-2"],
                              "RREL21": ["VRFD", ""]})
        out, issues, _summary = NORM.normalize_delivery(frame, rules)
        self.assertEqual(out.at[0, "RREL21"], "VRFD",
                         "a supplied verification code was overwritten")
        self.assertEqual([i.field for i in issues if i.severity == "error"], [])

    def test_a_supplied_value_is_not_counted_as_a_rule_applied_nd(self):
        rules = _only("RREL20")
        frame = pd.DataFrame({"RREL3": ["EXP-1", "EXP-2"],
                              "RREL20": ["42500.00", "51000.00"]})
        _out, _issues, summary = NORM.normalize_delivery(frame, rules)
        instr = summary["delivery_instrumentation"]
        self.assertNotIn("RREL20", instr["nd_applied_by_rules"]["by_field"])
        self.assertEqual(summary["field_provenance"]
                         ["populated_by_declared_delivery_rule"], 0,
                         "a column the projection supplied is not rule-created")

    def test_an_applicable_but_unavailable_field_can_be_configured_as_nd1(self):
        """ND1 stays reachable for a book where income IS underwritten."""
        rules = _only("RREL20", RREL20={"default_value": "ND1"})
        out, issues, summary = NORM.normalize_delivery(
            pd.DataFrame({"RREL3": ["EXP-1"]}), rules)
        self.assertEqual(out.at[0, "RREL20"], "ND1")
        self.assertEqual([i for i in issues if i.severity == "error"], [])
        self.assertEqual(summary["delivery_instrumentation"]
                         ["nd_applied_by_rules"]["by_field"]["RREL20"], 1)

    def test_both_nd1_and_nd5_are_permitted_so_the_choice_is_configuration(self):
        rules = _rules()["field_rules"]
        for code in _CODES:
            allowed = [str(x).upper() for x in rules[code]["nd_allowed"]]
            self.assertIn("ND1", allowed)
            self.assertIn("ND5", allowed)

    def test_no_python_module_hard_codes_the_secondary_income_answer(self):
        """The decision must live in configuration, not in code."""
        for path in (_NORMALIZER,
                     _REPO / "engine" / "gate_5_delivery" / "xml_builder_annex2.py"):
            source = path.read_text(encoding="utf-8")
            for code in _CODES:
                for literal in (f'"{code}": "ND5"', f"'{code}': 'ND5'",
                                f'"{code}": "ND1"', f"'{code}': 'ND1'"):
                    self.assertNotIn(literal, source,
                                     f"{path.name} hard-codes {code}")


# --------------------------------------------------------------------------- #
# 4. Canonical truth stays free of regime ND codes
# --------------------------------------------------------------------------- #
class TestNoNdContaminationOfCanonicalTruth(unittest.TestCase):
    def test_the_canonical_registry_declares_no_nd_default(self):
        registry = (_REPO / "config" / "system" / "fields_registry.yaml").read_text(
            encoding="utf-8")
        block = []
        for line in registry.splitlines():
            stripped = line.strip()
            if stripped.startswith(("secondary_income:", "secondary_income_verification:")):
                block.append(stripped)
        for entry in block:
            self.assertNotIn("ND1", entry)
            self.assertNotIn("ND5", entry)

    def test_nd_defaults_live_in_regime_and_asset_config_only(self):
        """ND for these fields is a DELIVERY decision, not canonical data."""
        erm = _erm_nd_defaults()
        self.assertIn("secondary_income", erm,
                      "the product-level ND decision must stay declared")
        rules = _rules()["field_rules"]
        for code in _CODES:
            self.assertTrue(str(rules[code].get("default_value") or "").startswith("ND"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
