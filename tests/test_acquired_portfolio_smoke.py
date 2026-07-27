#!/usr/bin/env python3
"""tests/test_acquired_portfolio_smoke.py

End-to-end smoke test for the ERE ``acquired_001`` MI pack.

Drives the real pipeline — scoped backfill → router → orchestrator → onboarding →
transformation → validation → stamp → assembler — over the acquired source pack
(``AcquiredLoanTape.csv``, 885 rows, two legacy rump populations in one file), and
asserts the outcomes the acquired path previously could not reach:

  * onboarding is ready with zero blockers despite two row-level cut-off dates;
  * transformation clears the ``date_parse_failed`` / ``boolean_parse_failed``
    blockers;
  * both ``protected_equity_percentage`` and ``protected_equity_flag`` survive
    into the canonical, the flag derived from the parsed percentage;
  * every source row keeps its own cut-off date — none is replaced by the pack's
    folder period;
  * the acquired canonical is published and assembled alongside the direct book.

The file is NOT split, no portfolio-specific values are configured, and no manual
pre-processing happens: the pack is processed exactly as delivered.

This exercises real agents over a real (synthetic) tape, so it is slower than the
unit regressions. Run: python -m pytest tests/test_acquired_portfolio_smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apps.blob_trigger_app import approvals as APPROVALS
from apps.blob_trigger_app import backfill as BF
from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.persistence import ProductionPersistence
from apps.blob_trigger_app.source_registry import SourceRegistry
from apps.blob_trigger_app.storage import Storage
from tests.helpers import acquired_pack as AP

DIRECT_COLUMNS = [
    "Loan Policy Number", "Cut Off Date", "Policy Completion Date",
    "Original Loan Amount", "Current Outstanding Balance", "Loan Interest Rate",
    "Post Code", "Borrower 1 DOB", "Borrower 1 Gender", "Borrower 2 DOB",
    "Borrower 2 Gender", "Protected Equity", "Original Property Value",
    "Latest Property Value", "Valuation Date", "Policy Status", "Product",
]
DIRECT_ROWS = 40
DIRECT_CUT_OFF = "2026-06-30"


def _write_direct_pack(root: Path) -> Path:
    """A small DIRECT funded pack, so the assembler has two portfolios to combine
    and backwards compatibility for the existing direct path is observable."""
    folder = (root / "raw-v2" / "ERE" / "direct" / "funded" / "monthly"
              / "direct_001" / DIRECT_CUT_OFF)
    folder.mkdir(parents=True, exist_ok=True)
    lines = [",".join(DIRECT_COLUMNS)]
    for i in range(DIRECT_ROWS):
        lines.append(",".join([
            str(90000000 + i), DIRECT_CUT_OFF, "14/03/2020",
            f"{120000 + i * 100:.2f}", f"{150000 + i * 100:.2f}", "6.10",
            "NP4 8AB", f"01/{(i % 12) + 1:02d}/1948", "M",
            "" if i % 3 == 0 else f"01/{(i % 12) + 1:02d}/1950",
            "" if i % 3 == 0 else "F",
            "0.00%" if i % 2 else "25.00%",
            f"{300000 + i * 500:.2f}", f"{310000 + i * 500:.2f}",
            "2026/06/30", "Inforce", "Direct Lifetime 6.10%"]))
    (folder / "LoanExtract.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "_READY.json").write_text("{}", encoding="utf-8")
    return folder


class TestAcquiredPortfolioSmoke(unittest.TestCase):
    """One run of the acquired pack, asserted from every artefact it produced."""

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory(prefix="acquired_smoke_")
        cls.root = Path(cls._td.name)
        AP.write_blob_tree(cls.root)
        _write_direct_pack(cls.root)

        cls.storage = Storage(cls.root)
        cls.layout = Layout()
        cls.persistence = ProductionPersistence(cls.storage, cls.layout)
        cls.registry = SourceRegistry(
            "blob://trakt-state/registry/source_registry.yaml", storage=cls.storage)
        cls.out_dir = cls.root / "out"

        # Each pack is a NEW source on first sight, so it halts at one-click
        # pending_review; the operator approves + promotes it, and the scoped
        # rerun then processes it deterministically. The direct book goes first
        # so the platform canonical has both portfolios to consolidate.
        cls.direct_first_pass = cls._backfill(
            selector=BF.PackSelector(source_portfolio_id="direct_001"))
        cls._approve_and_promote("direct_001")
        cls.direct_results = cls._backfill(
            selector=BF.PackSelector(source_portfolio_id="direct_001"), force=True)

        cls.first_pass = cls._backfill(
            selector=BF.PackSelector(pack_key=AP.PACK_KEY))
        cls._approve_and_promote(AP.SOURCE_PORTFOLIO_ID)
        cls.results = cls._backfill(
            selector=BF.PackSelector(pack_key=AP.PACK_KEY), force=True)
        cls.run_dir = cls._portfolio_dir(cls.results[0]["run_id"])

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    # -- harness ---------------------------------------------------------- #

    @classmethod
    def _backfill(cls, **kw):
        return BF.run_backfill(
            cls.storage, cls.persistence, cls.registry, container="raw-v2",
            out_dir=str(cls.out_dir), **kw)

    @classmethod
    def _approve_and_promote(cls, source_portfolio_id: str):
        """The operator's one-click approval of a new source, then promotion."""
        pending = APPROVALS.list_pending(cls.storage, cls.layout)
        match = [a for a in pending
                 if source_portfolio_id in a.get("approval_id", "")]
        assert match, f"no pending approval for {source_portfolio_id}"
        approval_id = match[0]["approval_id"]
        artefact = APPROVALS.show(cls.storage, cls.layout, approval_id)
        # A real mapping id — never the literal placeholder, which would promote
        # an unusable mapping.
        mapping_id = (artefact.get("suggested_mapping_id")
                      or f"{source_portfolio_id}_accepted_v1")
        assert "<" not in mapping_id, f"placeholder mapping id: {mapping_id}"
        APPROVALS.approve(
            cls.storage, cls.layout, approval_id, mapping_id=mapping_id,
            mapping_config_path=artefact.get("suggested_mapping_config_path"),
            decided_by="smoke")
        APPROVALS.promote(cls.storage, cls.layout, cls.registry, approval_id)
        # Re-read the registry so the promoted record is visible to the rerun.
        cls.registry = SourceRegistry(
            "blob://trakt-state/registry/source_registry.yaml", storage=cls.storage)

    @classmethod
    def _portfolio_dir(cls, run_id):
        return cls.out_dir / run_id / "portfolios" / AP.SOURCE_PORTFOLIO_ID

    def _json(self, relative):
        return json.loads((self.run_dir / relative).read_text(encoding="utf-8"))

    def _tape(self):
        return pd.read_csv(
            self.run_dir / "output" / "transformation" / "31_transformed_canonical_tape.csv",
            dtype=str)

    # -- gate: onboarding -------------------------------------------------- #

    def test_onboarding_gate_passes_with_zero_blockers(self):
        readiness = self._json("output/handoff/25_onboarding_handoff_readiness.json")
        self.assertTrue(readiness["ready_for_transformation_validation"])
        self.assertEqual(readiness["blocking_decision_count"], 0)
        self.assertEqual(readiness["operator_decision_pending_count"], 0)
        self.assertEqual(readiness["blocking_contract_rows"], [])

    def test_onboarding_resolves_the_cut_off_date_as_row_level(self):
        manifest = self._json("output/handoff/24_onboarding_handoff_manifest.json")
        self.assertTrue(manifest["data_cut_off_date_row_level"])
        self.assertEqual(sorted(manifest["data_cut_off_date_row_level_values"]),
                         [AP.RUMP_A_CUT_OFF, AP.RUMP_B_CUT_OFF])
        # No single pack-level date is claimed for a field that varies per loan.
        self.assertEqual(manifest["data_cut_off_date"], "")

    def test_folder_period_is_carried_as_pack_context(self):
        manifest = self._json("output/handoff/24_onboarding_handoff_manifest.json")
        self.assertEqual(manifest["reporting_period_context"], AP.PACK_PERIOD)

    def test_central_tape_carries_every_source_row(self):
        manifest = self._json("output/handoff/24_onboarding_handoff_manifest.json")
        self.assertEqual(manifest["central_tape_row_count"], AP.ROW_COUNT)

    # -- gate: transformation ---------------------------------------------- #

    def test_transformation_gate_passes(self):
        readiness = self._json("output/transformation/33_transformation_readiness.json")
        self.assertTrue(readiness["ready_for_validation"])
        self.assertEqual(readiness["blocking_for_validation_count"], 0)
        self.assertEqual(readiness["central_tape_row_count"], AP.ROW_COUNT)

    def test_the_two_reported_parse_blockers_are_gone(self):
        issues = self._json("output/transformation/35_transformation_issues.json")
        counts = issues["issue_type_counts"]
        self.assertEqual(counts.get("date_parse_failed", 0), 0)
        self.assertEqual(counts.get("boolean_parse_failed", 0), 0)
        self.assertEqual(counts.get("derivation_failed", 0), 0)

    def test_remaining_issues_are_non_blocking_mi_completeness_warnings(self):
        # MI completeness warnings are preserved as warnings, not promoted to
        # blockers: base MI must not require all 105+ optional fields.
        issues = self._json("output/transformation/35_transformation_issues.json")
        self.assertEqual(issues["blocking_for_validation"], 0)
        self.assertGreater(issues["issue_count"], 0)
        self.assertTrue(set(issues["issue_type_counts"]) <=
                        {"source_absent", "semantic_derivation_required",
                         "pending_projection_rule", "enum_unmapped",
                         "operator_decision_pending"})

    def test_transformed_tape_has_885_rows(self):
        self.assertEqual(len(self._tape()), AP.ROW_COUNT)

    # -- data: DOB ---------------------------------------------------------- #

    def test_dob_values_are_canonical_iso_dates(self):
        tape = self._tape()
        dob = tape["borrower_1_DOB"].dropna()
        self.assertEqual(len(dob), AP.ROW_COUNT)
        self.assertTrue(dob.str.match(r"^\d{4}-\d{2}-\d{2}$").all())

    def test_supplied_day_01_is_preserved_across_the_whole_tape(self):
        tape = self._tape()
        self.assertTrue((tape["borrower_1_DOB"].dropna().str[-2:] == "01").all())

    def test_missing_second_borrower_dob_stays_null(self):
        tape = self._tape()
        expected_blank = sum(1 for i in range(AP.ROW_COUNT) if i % 7 == 0)
        self.assertEqual(int(tape["borrower_2_DOB"].isna().sum()), expected_blank)

    def test_source_month_precision_is_recorded_in_lineage(self):
        # The provider knows month + year and supplies day 01. That convention is
        # recorded as provenance rather than inferred away.
        lineage = self._json("output/transformation/34_transformation_lineage.json")
        precision = lineage["source_date_precision"]
        for column in ("borrower_1_DOB", "borrower_2_DOB"):
            self.assertEqual(precision[column]["precision"], "month", column)
            self.assertEqual(precision[column]["day_convention"],
                             "source_supplied_01", column)
        # A column whose days genuinely vary is NOT labelled month-precision.
        self.assertEqual(precision["data_cut_off_date"]["precision"], "day")

    # -- data: protected equity -------------------------------------------- #

    def test_both_protected_equity_fields_are_present(self):
        tape = self._tape()
        self.assertIn("protected_equity_percentage", tape.columns)
        self.assertIn("protected_equity_flag", tape.columns)

    def test_percentage_uses_the_canonical_percentage_point_scale(self):
        tape = self._tape()
        values = sorted(set(tape["protected_equity_percentage"].dropna().astype(float)))
        self.assertEqual(values, [0.0, 20.0, 30.0, 50.0])

    def test_flag_matches_the_parsed_percentage_row_by_row(self):
        tape = self._tape()
        zero, positive, blank = AP.expected_protected_equity()
        flags = tape["protected_equity_flag"]
        self.assertEqual(int((flags == "N").sum()), zero)
        self.assertEqual(int((flags == "Y").sum()), positive)
        self.assertEqual(int(flags.isna().sum()), blank)
        # And the pairing holds per row, not just in aggregate.
        pct = pd.to_numeric(tape["protected_equity_percentage"], errors="coerce")
        self.assertTrue(((pct > 0) == (flags == "Y")).all())
        self.assertTrue(((pct == 0) == (flags == "N")).all())
        self.assertTrue((pct.isna() == flags.isna()).all())

    def test_flag_holds_only_the_canonical_boolean_or_null(self):
        tape = self._tape()
        self.assertTrue(set(tape["protected_equity_flag"].dropna()) <= {"Y", "N"})

    def test_lineage_shows_percentage_sourced_and_flag_derived(self):
        # 22 — the flag must be attributable to the percentage, and the
        # percentage to the source column.
        lineage = self._json("output/transformation/34_transformation_lineage.json")
        flag = next(r for r in lineage["transformation_lineage"]
                    if r.get("source_canonical_field") == "protected_equity_flag")
        self.assertEqual(flag["value_origin"], "derived")
        self.assertEqual(flag["derived_from"], "protected_equity_percentage")
        self.assertEqual(flag["derivation_rule"], "positive_number_to_flag")

        onboarding = lineage["onboarding_lineage"]
        pct = next(r for r in onboarding
                   if r.get("canonical_field") == "protected_equity_percentage")
        self.assertEqual(pct["source_column"], "Protected Equity")
        self.assertEqual(pct["source_file"], AP.FILE_NAME)

    def test_protected_equity_is_not_copied_into_the_flag(self):
        # 21 — the defect: the source column mapped into BOTH fields.
        lineage = self._json("output/transformation/34_transformation_lineage.json")
        flag_sources = [r.get("source_column") for r in lineage["onboarding_lineage"]
                        if r.get("canonical_field") == "protected_equity_flag"]
        self.assertNotIn("Protected Equity", flag_sources)

    # -- data: cut-off dates ------------------------------------------------ #

    def test_mixed_source_cut_off_dates_survive_transformation(self):
        tape = self._tape()
        counts = tape["data_cut_off_date"].value_counts().to_dict()
        self.assertEqual(counts, {AP.RUMP_A_CUT_OFF: AP.RUMP_A_ROWS,
                                  AP.RUMP_B_CUT_OFF: AP.RUMP_B_ROWS})

    def test_no_source_row_was_replaced_by_the_folder_period(self):
        # The pack folder period is 2026-06-30; rump A's 512 rows must NOT have
        # been rewritten to it.
        tape = self._tape()
        self.assertEqual(int((tape["data_cut_off_date"] == AP.PACK_PERIOD).sum()),
                         AP.RUMP_B_ROWS)

    def test_the_file_was_not_split_into_sub_portfolios(self):
        # One source portfolio, one canonical output, differences kept as data.
        self.assertEqual(len(self.results), 1)
        self.assertEqual(self.results[0]["source_portfolio_id"],
                         AP.SOURCE_PORTFOLIO_ID)

    # -- gates: stamp + assembler ------------------------------------------ #

    def test_stamp_gate_produces_the_acquired_canonical(self):
        stamped = self.run_dir / "stamped" / "acquired_001_canonical_typed.csv"
        self.assertTrue(stamped.exists(), stamped)
        self.assertEqual(len(pd.read_csv(stamped, dtype=str)), AP.ROW_COUNT)

    def test_accepted_artefact_is_published_for_the_acquired_portfolio(self):
        accepted = self.out_dir / "_accepted" / "ERE" / "acquired_001_canonical_typed.csv"
        self.assertTrue(accepted.exists(), accepted)
        self.assertEqual(len(pd.read_csv(accepted, dtype=str)), AP.ROW_COUNT)

    def _platform(self):
        # The CROSS-portfolio canonical the assembler rebuilds from the accepted
        # store (out/_platform), not a single run's own out_platform.
        path = self.out_dir / "_platform" / "platform_canonical_typed.csv"
        self.assertTrue(path.exists(), path)
        return pd.read_csv(path, dtype=str)

    def test_assembled_platform_canonical_holds_both_portfolios(self):
        df = self._platform()
        self.assertEqual(len(df), AP.ROW_COUNT + DIRECT_ROWS)
        self.assertEqual(sorted(df["source_portfolio_id"].unique()),
                         ["acquired_001", "direct_001"])

    def test_acquired_row_level_dates_survive_assembly(self):
        df = self._platform()
        acquired = df[df["source_portfolio_id"] == "acquired_001"]
        self.assertEqual(acquired["data_cut_off_date"].value_counts().to_dict(),
                         {AP.RUMP_A_CUT_OFF: AP.RUMP_A_ROWS,
                          AP.RUMP_B_CUT_OFF: AP.RUMP_B_ROWS})

    def test_direct_portfolio_behaviour_is_unchanged(self):
        # Backwards compatibility: the direct book has ONE cut-off date and is
        # resolved exactly as before.
        df = self._platform()
        direct = df[df["source_portfolio_id"] == "direct_001"]
        self.assertEqual(len(direct), DIRECT_ROWS)
        self.assertEqual(set(direct["data_cut_off_date"]), {DIRECT_CUT_OFF})

    # -- governance --------------------------------------------------------- #

    def test_new_source_halted_for_one_click_review_before_approval(self):
        # The approval gate is not bypassed: pass 1 stopped at pending_review.
        self.assertEqual(self.first_pass[0]["status"], "pending_review")

    def test_approved_rerun_processes_deterministically(self):
        self.assertEqual(self.results[0]["status"], "processed")
        self.assertEqual(self.results[0]["decision"], "deterministic")

    def test_source_file_is_never_mutated(self):
        source = (self.root / "raw-v2" / AP.BLOB_PREFIX / AP.FILE_NAME)
        raw = pd.read_csv(source, dtype=str)
        self.assertEqual(len(raw), AP.ROW_COUNT)
        self.assertEqual(sorted(raw["Cut Off Date"].unique()),
                         [AP.RUMP_A_CUT_OFF, AP.RUMP_B_CUT_OFF])
        self.assertEqual(raw["Borrower 1 DOB"].iloc[0], "01/01/1930")
        self.assertEqual(raw["Protected Equity"].iloc[1], "20.00%")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
