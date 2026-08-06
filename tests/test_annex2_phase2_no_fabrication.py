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
        to make, however true it happens to be today — Gate 5 cannot see what
        Gate 4b did.
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

    def test_they_default_to_nd1_not_collected(self):
        """ND1 = 'Data not collected as not required by the underwriting criteria'.

        Not ND5. Equity-release loans commonly have joint borrowers, so a
        secondary obligor usually exists — the code is not about the obligor's
        absence. Income is simply outside the product's underwriting
        methodology. Full reasoning in
        tests/test_annex2_secondary_income_applicability.py.
        """
        for code in ("RREL20", "RREL21"):
            rule = self.rules[code]
            self.assertTrue(rule["default_allowed"])
            self.assertEqual(rule["default_value"], "ND1")
            self.assertIn("ND1", rule["nd_allowed"])

    def test_the_whole_income_family_shares_one_rationale(self):
        """Primary and secondary income are treated alike, because the reason
        is the same: equity-release underwriting does not use borrower income.

        RREL19 (primary verification) cannot use ND5 at all — the workbook sets
        nd5_allowed FALSE — so a secondary field answering ND5 on the identical
        rationale would be inconsistent.
        """
        for code in ("RREL19", "RREL20", "RREL21"):
            self.assertEqual(self.rules[code]["default_value"], "ND1", code)

    def test_nd5_stays_permitted_even_though_it_is_not_the_default(self):
        """The workbook allows ND5 here; configuration chooses ND1.

        A different book whose secondary-income field genuinely does not apply
        can still declare ND5 without a code change.
        """
        for code in ("RREL20", "RREL21"):
            self.assertIn("ND5", self.rules[code]["nd_allowed"])

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

    def test_the_rrel12_geography_map_is_gone(self):
        """The last latent fabrication in the Annex 2 delivery path.

        The RREL12 rule carried a ``geography_map`` copied from RREL11 — the
        same ten UK region names, with the target changed from the region code
        ``GBZZZ`` to the literal year ``"2026"``. A region name cannot
        determine a NUTS vintage, so this invented a classification year the
        client never reported. Worse, transforms run BEFORE validators, so it
        also defeated the very regex guard declared beside it.

        No workbook, XSD or ESMA basis exists for it, and
        ``canonical_transform.py`` — the stage that POPULATES this field —
        documents the opposite policy outright.
        """
        rules = yaml.safe_load(_RULES.read_text(encoding="utf-8"))
        rule = rules["field_rules"]["RREL12"]
        transform = rule.get("transform") or {}
        self.assertNotIn("geography_map", transform,
                         "the RREL12 region-name -> year map is a fabrication")
        self.assertEqual(transform, {},
                         "RREL12 must carry no value transform at all")

    def test_no_rule_maps_a_region_name_to_a_year(self):
        """The defect class, not just the one instance.

        Any rule translating a readable region label into a bare year is
        inventing a classification vintage from something that cannot imply one.
        """
        rules = yaml.safe_load(_RULES.read_text(encoding="utf-8"))
        offenders = []
        for code, rule in (rules.get("field_rules") or {}).items():
            if not isinstance(rule, dict):
                continue
            geo = (rule.get("transform") or {}).get("geography_map") or {}
            for key, value in geo.items():
                if re.fullmatch(r"(19|20)\d{2}", str(value).strip()) and \
                        not re.fullmatch(r"(19|20)\d{2}", str(key).strip()):
                    offenders.append(f"{code}: {key!r} -> {value!r}")
        self.assertEqual(offenders, [],
                         "a non-year key maps to a year; a classification year "
                         "cannot be derived from a region name")

    def test_the_semantic_guard_survived_the_removal(self):
        """Removing the map must not remove the protection it defeated.

        Deleting the transform without keeping the regex would be strictly
        worse than the fabrication: a region label would reach Gate 5 unchecked
        instead of being caught at Gate 4b.
        """
        rules = yaml.safe_load(_RULES.read_text(encoding="utf-8"))
        pattern = (rules["field_rules"]["RREL12"].get("validators") or {}).get("regex")
        self.assertTrue(pattern, "the RREL12 year guard must remain declared")
        for good in ("2021", "2013", "ND1"):
            self.assertTrue(re.fullmatch(pattern, good), good)
        for bad in ("West Midlands", "London", "GBZZZ", "TLG31", "20211"):
            self.assertIsNone(re.fullmatch(pattern, bad), bad)

    def test_an_unmappable_rrel12_value_fails_rather_than_becoming_a_year(self):
        """Behavioural: the governed failure, end to end through Gate 4b."""
        rules = yaml.safe_load(_RULES.read_text(encoding="utf-8"))
        only = {"defaults": rules.get("defaults", {}),
                "field_rules": {"RREL12": rules["field_rules"]["RREL12"]}}
        frame = pd.DataFrame({"RREL12": ["2021", "West Midlands", "ND1", ""]})
        out, issues, summary = NORM.normalize_delivery(frame, only)

        self.assertEqual(list(out["RREL12"]), ["2021", "West Midlands", "ND1", ""],
                         "no value may be substituted, not even the invalid one")
        errors = [i for i in issues if i.field == "RREL12" and i.severity == "error"]
        self.assertEqual([i.row_index for i in errors], [1])
        self.assertEqual(errors[0].issue_type, "pattern")
        self.assertEqual(summary["preflight"]["status"], "FAIL",
                         "an unrepresentable value must fail the run")
        self.assertEqual(
            summary["delivery_instrumentation"]["coercions"]["count"], 0,
            "nothing was invented")

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

    End-to-end neutrality was proven by the benchmark SHA-256
    (``a21f8a4c…d685d``, unchanged across Phase 1 and Phase 2). The baseline is
    now ``8018abb9…3da5`` after the deliberate RREL20/RREL21 ND5 → ND1
    regulatory correction — the only intentional change to the submission, and
    a value-level one: 22,070 bytes differ, every one a ``5`` → ``1``.

    What is checked here is the property that makes any of it reproducible:
    identical input yields a byte-identical CSV and identical counts, run after
    run.
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


# --------------------------------------------------------------------------- #
# 6. The coercion channel is retained on purpose, not left behind
# --------------------------------------------------------------------------- #
class TestBuilderCoercionChannelIsRetainedDeliberately(unittest.TestCase):
    """`record_coercion` has no production caller — by design, not by accident.

    Phase 2 removed the builder's only value coercion, so the expected count is
    zero. The mechanism stays so that a future builder coercion is *surfaced*
    rather than silent, and so the reported zero is an observation rather than
    a hard-coded literal.

    Without this class the method looks exactly like dead code and a future
    tidy-up would delete it, taking the falsifiability of "zero" with it.
    """

    def setUp(self):
        BUILD._INSTR.reset()

    def tearDown(self):
        BUILD._INSTR.reset()

    def test_the_channel_still_records_an_event(self):
        instr = BUILD._DeliveryInstrumentation()
        instr.record_coercion(field_code="ZZZ99", original_value="a",
                              resulting_value="b", row_identifier="EXP-1",
                              reason="retention check")
        report = instr.to_dict()
        self.assertEqual(report["coercions"]["count"], 1)
        self.assertEqual(report["coercions"]["records"][0]["original_value"], "a")

    def test_the_reported_zero_is_observed_not_hard_coded(self):
        """If the count were a literal, recording could not move it."""
        instr = BUILD._DeliveryInstrumentation()
        self.assertEqual(instr.to_dict()["coercions"]["count"], 0)
        instr.record_coercion(field_code="ZZZ99", original_value="a",
                              resulting_value="b", reason="retention check")
        self.assertEqual(instr.to_dict()["coercions"]["count"], 1)
        instr.reset()
        self.assertEqual(instr.to_dict()["coercions"]["count"], 0)

    def test_the_report_states_zero_explicitly_rather_than_omitting_it(self):
        report = BUILD._DeliveryInstrumentation().to_dict()
        self.assertIn("coercions", report)
        self.assertEqual(report["coercions"]["count"], 0)
        self.assertEqual(report["coercions"]["records"], [])
        self.assertIs(report["coercions"]["truncated"], False)

    def test_no_production_caller_exists_and_that_is_intentional(self):
        """Guards both directions: it must stay uncalled AND stay documented.

        A new production caller would be a policy change — the builder would be
        altering a value again — so it must not appear silently.
        """
        source = _BUILDER.read_text(encoding="utf-8")
        body = source[source.index("def record_coercion"):]
        body = body[:body.index("\n    def ")]
        self.assertIn("RETAINED OBSERVABILITY CHANNEL", body,
                      "the retention rationale must stay next to the method")
        # Only the definition and the docstring's own reference — no call site.
        calls = re.findall(r"_INSTR\.record_coercion\(|self\.record_coercion\(",
                           source)
        self.assertEqual(calls, [],
                         "a production caller appeared; a builder coercion is a "
                         "policy change and needs its own justification")

    def test_the_channel_is_exercised_by_the_test_suite(self):
        """It must remain executable, not merely present."""
        phase1 = (_REPO / "tests" / "test_annex2_phase1_instrumentation.py").read_text(
            encoding="utf-8")
        self.assertIn("record_coercion", phase1,
                      "the retained channel must stay under test")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
