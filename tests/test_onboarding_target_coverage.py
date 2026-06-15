#!/usr/bin/env python3
"""tests/test_onboarding_target_coverage.py — target-contract-first coverage (28a/28b/28c).

Covers the eight required behaviours of the target-contract-first onboarding
coverage model:

  1. MI mode loads target fields from mi_semantics_field_registry.
  2. The target coverage matrix is target-field-led, not source-column-led.
  3. Residual source columns are suppressed from the compact human decision queue.
  4. Duplicate / overlapping source fields are represented as alternatives for one
     target field.
  5. Regulatory / Annex 2 mode loads Annex 2 target fields.
  6. Annex 2 fields with known ND/default/config rules are NOT marked as missing
     source data.
  7. Annex 12 is not included in the Annex 2 target coverage implementation.
  8. The compact human decision queue contains only target-coverage decisions /
     blocking residuals, not all source-column review rows.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.onboarding_agent import target_coverage as tcov
from engine.onboarding_agent.llm_assisted_mapping import run_llm_assisted_mapping

ERE = str(_REPO_ROOT / "synthetic_demo" / "input" / "SYNTHETIC_ERE_Portfolio_012026.csv")
ANNEX2 = str(_REPO_ROOT / "tests" / "fixtures" / "annex2_delivery_ready_no_npe.csv")


def _ev(col, registry_field="", *, null_rate=0.0, distinct=10,
        dtype="string", value_profile=""):
    """Build a minimal column-evidence row for unit tests."""
    return {
        "source_file": "src.csv", "source_sheet": "", "source_column": col,
        "normalized_column": col.lower().replace(" ", "_"),
        "domain_guess": "unknown", "file_domain_guess": "unknown",
        "null_rate": null_rate, "distinct_count": distinct, "data_type_guess": dtype,
        "candidate_existing_registry_fields": registry_field,
        "candidate_existing_pipeline_contract_fields": "",
        "candidate_alias_matches": "", "candidate_semantic_alignment_matches": "",
        "known_client_memory_matches": "", "candidate_value_profile_matches": value_profile,
    }


def _run(input_file, mode):
    warnings.simplefilter("ignore")
    out = Path(tempfile.mkdtemp())
    res = run_llm_assisted_mapping(input_file=input_file, output_dir=str(out),
                                   mode=mode, client_id="t", run_id="r1")
    return res, out


# --------------------------------------------------------------------------- #
# 1 + 5 + 7 — target contract loading
# --------------------------------------------------------------------------- #
class TestTargetContractLoading(unittest.TestCase):
    def test_mi_mode_loads_mi_semantics_registry(self):
        cid, csrc, fields = tcov.load_target_contract("mi_only", {})
        self.assertEqual(cid, "mi_semantics_field_registry")
        self.assertTrue(csrc.endswith("mi_semantics_field_registry.yaml"))
        self.assertEqual(len(fields), 72)  # not pruned at this stage
        names = {f["target_field"] for f in fields}
        self.assertIn("account_status", names)
        self.assertIn("current_interest_rate", names)

    def test_regulatory_mode_loads_annex2(self):
        cid, csrc, fields = tcov.load_target_contract("regulatory_mi", {})
        self.assertEqual(cid, "esma_annex_2")
        self.assertTrue(csrc.endswith("annex2_delivery_rules.yaml"))
        names = {f["target_field"] for f in fields}
        self.assertIn("RREL1", names)
        self.assertIn("RREC9", names)

    def test_annex12_not_included_in_annex2_contract(self):
        _cid, csrc, fields = tcov.load_target_contract("regulatory_mi", {})
        names = [f["target_field"] for f in fields]
        # No Annex 12 (IVSS / IVSR) deal-level codes leak into Annex 2 coverage.
        self.assertFalse(any(n.startswith("IVSS") or n.startswith("IVSR") for n in names))
        self.assertNotIn("annex12", csrc.lower())
        self.assertIn("annex2", csrc.lower())


# --------------------------------------------------------------------------- #
# 2 — target-field-led, not source-column-led
# --------------------------------------------------------------------------- #
class TestTargetFieldLed(unittest.TestCase):
    def test_one_row_per_target_field_independent_of_source_count(self):
        res, _ = _run(ERE, "mi_only")
        tf = res["target_first_coverage"]
        _cid, _csrc, fields = tcov.load_target_contract("mi_only", {})
        # Coverage matrix has exactly one row per target field …
        self.assertEqual(len(tf["coverage"]), len(fields))
        # … and the count is the contract size, NOT the source-column count.
        source_cols = len(res["evidence"])
        self.assertNotEqual(len(tf["coverage"]), source_cols)
        self.assertEqual(tf["coverage_summary"]["target_fields_total"], 72)

    def test_coverage_status_vocabulary(self):
        res, _ = _run(ERE, "mi_only")
        allowed = {tcov.SOURCE_MAPPED, tcov.SOURCE_MAPPED_ALT, tcov.DERIVED,
                   tcov.CONFIGURED_STATIC, tcov.DEFAULTED, tcov.DEFAULTED_ND,
                   tcov.NOT_APPLICABLE, tcov.MISSING_REQUIRED, tcov.NEEDS_CONFIRMATION}
        for r in res["target_first_coverage"]["coverage"]:
            self.assertIn(r["coverage_status"], allowed)


# --------------------------------------------------------------------------- #
# 3 + 8 — residuals suppressed; compact decision queue is target-led
# --------------------------------------------------------------------------- #
class TestCompactDecisionQueue(unittest.TestCase):
    def test_residual_columns_not_in_decision_queue(self):
        res, _ = _run(ERE, "mi_only")
        tf = res["target_first_coverage"]
        residual_cols = {(r["source_file"], r["source_column"])
                         for r in tf["residual"]}
        decision_cols = {(d["source_file"], d["source_column"])
                         for d in tf["decision_queue"] if d["source_column"]}
        # No residual source column leaks into the compact decision queue.
        self.assertEqual(residual_cols & decision_cols, set())
        # All residuals are suppressed from the main queue.
        self.assertTrue(all(r["suppressed_from_main_queue"] for r in tf["residual"]))

    def test_queue_is_compact_and_target_led(self):
        res, _ = _run(ERE, "mi_only")
        tf = res["target_first_coverage"]
        old_33_total = res["review_queue"]["summary"]["total_columns_reviewed"]
        dq = tf["decision_queue"]
        # Compact: materially smaller than the source-column 33 review queue.
        self.assertLess(len(dq), old_33_total)
        # Only target-coverage decision types / blocking residuals.
        allowed = {tcov.D_MISSING, tcov.D_CONFLICT, tcov.D_PRIORITY, tcov.D_VALUE,
                   tcov.D_CONFIG, tcov.D_ND, tcov.D_EXTENSION, tcov.D_PARSE}
        for d in dq:
            self.assertIn(d["decision_type"], allowed)
        # Every decision references a target field OR is a blocking residual.
        for d in dq:
            self.assertTrue(d["target_field"] or d["blocking"], d)


# --------------------------------------------------------------------------- #
# 4 — duplicate / overlapping source fields as alternatives for one target
# --------------------------------------------------------------------------- #
class TestAlternatives(unittest.TestCase):
    def test_two_sources_one_target_become_alternatives(self):
        _cid, _csrc, fields = tcov.load_target_contract("mi_only", {})
        # Two source columns both pointing at the SAME canonical target.
        evidence = [
            _ev("Current Interest Rate", registry_field="current_interest_rate"),
            _ev("Product Rate", registry_field="current_interest_rate"),
            _ev("Some Unrelated Note", value_profile=""),
        ]
        cov_rows, matched = tcov.build_target_coverage(
            "mi_only", {}, "mi_semantics_field_registry", "src", fields,
            evidence, resolved_rows=[])
        by_field = {r["target_field"]: r for r in cov_rows}
        rate = by_field["current_interest_rate"]
        self.assertEqual(rate["coverage_status"], tcov.SOURCE_MAPPED_ALT)
        self.assertTrue(rate["selected_source_column"])
        self.assertTrue(rate["alternative_source_candidates"])
        # The non-selected duplicate is captured in the residual register as an
        # alternative source for that target field.
        residual = tcov.build_source_residual_register("mi_only", evidence, matched)
        dup = [r for r in residual if r["residual_class"] == tcov.R_DUP_ALT]
        self.assertEqual(len(dup), 1)
        self.assertEqual(dup[0]["duplicate_of_target_field"], "current_interest_rate")


# --------------------------------------------------------------------------- #
# 6 — Annex 2 ND/default/config fields are not "missing source data"
# --------------------------------------------------------------------------- #
class TestAnnex2DefaultsNotMissing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # ERE pack has readable headers, so ND/default/config Annex 2 fields with
        # no matching source column must fall back to their rule, not "missing".
        cls.res, _ = _run(ERE, "regulatory_mi")
        cls.cov = {r["target_field"]: r
                   for r in cls.res["target_first_coverage"]["coverage"]}

    def test_nd_default_field_is_defaulted_nd(self):
        # RREL16 (primary_income) is mandatory with default_value ND1.
        r = self.cov["RREL16"]
        self.assertEqual(r["coverage_status"], tcov.DEFAULTED_ND)
        self.assertNotEqual(r["coverage_status"], tcov.MISSING_REQUIRED)
        self.assertTrue(r["nd_rule_applied"])

    def test_static_default_is_configured_static(self):
        # RREC8 (lien) has a fixed default value of "1".
        self.assertEqual(self.cov["RREC8"]["coverage_status"], tcov.CONFIGURED_STATIC)

    def test_derive_rule_is_derived(self):
        # RREL25 (original_term) has a months_between_dates derivation.
        self.assertEqual(self.cov["RREL25"]["coverage_status"], tcov.DERIVED)

    def test_known_rule_fields_not_counted_missing(self):
        summary = self.res["target_first_coverage"]["coverage_summary"]
        # The well-known ND/default/config fields must not inflate "missing".
        for code in ("RREL16", "RREL22", "RREC8", "RREL25", "RREC6"):
            self.assertNotEqual(self.cov[code]["coverage_status"], tcov.MISSING_REQUIRED)
        self.assertGreater(summary["derived_config_defaulted_fields"], 0)


# --------------------------------------------------------------------------- #
# Artefacts exist on disk
# --------------------------------------------------------------------------- #
class TestArtefactsWritten(unittest.TestCase):
    def test_28abc_artefacts_written(self):
        _res, out = _run(ERE, "mi_only")
        for name in (
            "28a_target_coverage_matrix.csv", "28a_target_coverage_matrix.json",
            "28a_target_coverage_summary.md",
            "28b_source_residual_register.csv", "28b_source_residual_register.json",
            "28b_source_residual_summary.md",
            "28c_human_decision_queue.csv", "28c_human_decision_queue.json",
            "28c_human_decision_summary.md",
        ):
            self.assertTrue((out / name).exists(), name)
        # The source-column 33 queue remains as audit detail.
        self.assertTrue((out / "33_mapping_review_queue.csv").exists())


if __name__ == "__main__":
    unittest.main()
