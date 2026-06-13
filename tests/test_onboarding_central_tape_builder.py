#!/usr/bin/env python3
"""tests/test_onboarding_central_tape_builder.py — PART 15 (8–17)."""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from onboarding_domain_fixtures import REGISTRY, SCENARIO_A, build_run
from engine.onboarding_agent import central_tape_builder


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _build(scenario=SCENARIO_A, mode="regulatory_mi", ingest=True,
           drop_precedence=False, reg=False):
    project, pdir, rp = build_run(
        scenario, mode=mode, ingest=ingest, drop_precedence=drop_precedence,
        regulatory_reporting_enabled=reg,
    )
    res = central_tape_builder.build_central_tapes(
        pdir, rp, str(REGISTRY), mode=mode, regulatory_reporting_enabled=reg
    )
    return project, pdir, rp, res


class TestLenderTape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project, cls.pdir, cls.rp, cls.res = _build()
        cls.tape = _read_csv(cls.res["central_lender_tape_path"])
        cls.lineage = _read_csv(cls.res["central_tape_lineage_path"])

    # 8. Central lender tape is created from combined master tape.
    def test_tape_created(self):
        self.assertTrue(self.res["central_lender_tape_created"])
        self.assertTrue(Path(self.res["central_lender_tape_path"]).exists())
        self.assertTrue(self.tape)

    # 9. One row per funded loan.
    def test_one_row_per_loan(self):
        ids = [r["loan_identifier"] for r in self.tape]
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(set(ids)), 8)

    # 10. Loan and collateral fields populated from the same source file.
    def test_loan_and_collateral_from_same_file(self):
        cols = self.res["lender_summary"]["columns"]
        self.assertIn("current_principal_balance", cols)   # loan domain
        self.assertIn("property_post_code", cols)          # collateral domain
        # property_post_code lineage points at the combined master tape.
        pc = [r for r in self.lineage if r["canonical_field"] == "property_post_code"]
        self.assertTrue(pc)
        self.assertTrue(all(r["source_file"] == "master_loan_collateral_tape.csv" for r in pc))

    # 11. Lineage records source file / column for each populated field.
    def test_lineage_has_source_file_and_column(self):
        self.assertTrue(self.lineage)
        for r in self.lineage:
            self.assertTrue(r["source_file"], r)
            self.assertTrue(r["source_column"], r)
            self.assertTrue(r["domain"])

    # 12. Matching duplicate values are treated as validation sources.
    def test_matching_values_become_validation_sources(self):
        # L0001 balance agrees between master tape and cashflow extract.
        bal = [r for r in self.lineage
               if r["canonical_field"] == "current_principal_balance"
               and r["loan_identifier"] == "L0001"]
        self.assertTrue(bal)
        self.assertTrue(bal[0]["validation_sources"], "expected a validation source")
        self.assertIn(bal[0]["conflict_status"], ("validated", "resolved_by_precedence"))


class TestConflicts(unittest.TestCase):
    # 13. Conflicting values produce conflict gaps unless precedence is approved.
    def test_conflict_without_precedence(self):
        _, _, _, res = _build(drop_precedence=True)
        gaps = _read_csv(res["central_tape_gaps_path"])
        conflicts = [g for g in gaps if g["issue_type"] == "value_conflict"]
        self.assertTrue(conflicts, "expected conflict gaps when no precedence approved")
        self.assertTrue(any(g["canonical_field"] == "current_principal_balance"
                            for g in conflicts))

    def test_conflict_resolved_with_precedence(self):
        _, _, _, res = _build(ingest=True, drop_precedence=False)
        gaps = _read_csv(res["central_tape_gaps_path"])
        conflicts = [g for g in gaps if g["issue_type"] == "value_conflict"]
        self.assertFalse(conflicts, "approved precedence should resolve conflicts")


class TestPipelineTape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project, cls.pdir, cls.rp, cls.res = _build()
        cls.pipeline = _read_csv(cls.res["central_pipeline_tape_path"])
        cls.tape = _read_csv(cls.res["central_lender_tape_path"])

    # 14. Pipeline rows without linked_loan_id go to the pipeline tape only.
    def test_application_only_rows_not_in_lender_tape(self):
        lender_ids = {r["loan_identifier"] for r in self.tape}
        app_ids = {r["application_id"] for r in self.pipeline}
        self.assertIn("A1002", app_ids)
        self.assertNotIn("A1002", lender_ids)
        application_only = [r for r in self.pipeline
                            if not str(r["linked_loan_identifier"]).strip()]
        self.assertTrue(application_only)

    # 15. Pipeline rows with linked_loan_id record the relationship.
    def test_linked_rows_record_relationship(self):
        linked = {r["application_id"]: r for r in self.pipeline
                  if str(r["linked_loan_identifier"]).strip()}
        self.assertIn("A1001", linked)
        self.assertEqual(linked["A1001"]["linked_loan_identifier"], "L0001")
        self.assertEqual(str(linked["A1001"]["linked_to_central_lender_tape"]), "True")


class TestModeScope(unittest.TestCase):
    # 16. MI-only excludes regulatory non-core fields from the central lender tape.
    def test_mi_only_excludes_regulatory_noncore(self):
        _, _, _, res = _build(mode="mi_only")
        cols = res["lender_summary"]["columns"]
        self.assertIn("current_principal_balance", cols)  # core stays
        self.assertNotIn("property_post_code", cols)      # regulatory non-core dropped
        self.assertNotIn("current_loan_to_value", cols)

    # 17. Regulatory+MI includes regulatory fields where in scope.
    def test_regulatory_mi_includes_regulatory(self):
        _, _, _, res = _build(mode="regulatory_mi")
        cols = res["lender_summary"]["columns"]
        self.assertIn("property_post_code", cols)
        self.assertIn("current_valuation_amount", cols)


if __name__ == "__main__":
    unittest.main()
