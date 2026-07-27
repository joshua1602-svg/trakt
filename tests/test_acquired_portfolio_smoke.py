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
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apps.blob_trigger_app import approvals as APPROVALS
from apps.blob_trigger_app import backfill as BF
from apps.blob_trigger_app import ops as OPS
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


class _AcquiredRunBase(unittest.TestCase):
    """Shared harness: onboard, approve, promote and rerun the acquired pack.

    Each subclass gets its OWN container tree, state store and registry, so a
    test that goes on to process further packs (and therefore re-pins the
    registry) cannot disturb another class's assertions. That independence
    matters here: the outcome of a pack depends on the registry state the
    previous pack left behind, so shared state would make results depend on test
    order.
    """

    maxDiff = None
    prefix = "acquired_smoke_"

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory(prefix=cls.prefix)
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


class TestAcquiredPortfolioSmoke(_AcquiredRunBase):
    """One run of the acquired pack, asserted from every artefact it produced."""

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


class TestAcquiredPackRecurrence(_AcquiredRunBase):
    """Later deliveries of an already-approved acquired source.

    Runs in its own container tree and state store: processing further packs
    re-pins the registry, so sharing that state with the artefact assertions
    above would make both sets of results depend on test order.
    """

    prefix = "acquired_recurrence_"

    def test_recurring_packs_process_without_re_approval_but_drift_still_stops(self):
        """One-click review is a FIRST-SIGHT gate, not a per-delivery tax.

        Two states of the same promoted source, asserted in order because each
        one leaves the registry as the next one finds it:

          1. a later delivery of the SAME shape must process straight through.
             The fingerprint is structural (column names, order, sheets, file
             type) — never the filename, the period folder or any cell value — so
             the next acquired pack matches the promoted signature. Re-asking
             here is what turns "approve once" into "approve every month".
          2. a MATERIAL change — a mandatory field disappearing — must still stop
             for review rather than run on a mapping approved for another shape.

        (An ADDITIVE optional column is deliberately not material: the approval
        policy documents reorder / header-rename / additive-optional as
        auto-approvable, precisely so routine supplier tweaks do not queue.)
        """
        # 1. Same schema, later period → no operator intervention.
        later = "2026-09-30"
        AP.write_blob_tree(self.root, period=later)
        recurring = self._backfill(
            selector=BF.PackSelector(pack_key=AP.pack_key_for_period(later)))
        self.assertEqual(len(recurring), 1)
        self.assertEqual(recurring[0]["status"], "processed")
        self.assertEqual(recurring[0]["decision"], "deterministic")
        pending = " ".join(a["approval_id"] for a
                           in APPROVALS.list_pending(self.storage, self.layout))
        self.assertNotIn(later, pending)

        # 2. Mandatory loan identifier gone → material → held for review.
        drifted = "2026-12-31"
        AP.write_blob_tree(self.root, period=drifted,
                           drop_columns=("Loan Policy Number",))
        held = self._backfill(
            selector=BF.PackSelector(pack_key=AP.pack_key_for_period(drifted)))
        self.assertEqual(len(held), 1)
        self.assertNotEqual(held[0]["status"], "processed")


class TestSourceValuePlaceholderDecision(unittest.TestCase):
    """The full operator loop for a placeholder value in the acquired tape.

    The delivery carries ``TBC`` in ``Borrower 1 DOB``. End to end:

      1. it fails deterministic parsing and holds the transformation gate — an
         undecided placeholder is a data question, not something to guess at;
      2. onboarding surfaces it as a NON-BLOCKING operator decision naming the
         exact value, column and row count;
      3. the operator approves ``treat_source_value_as_null``;
      4. the rerun nulls it before parsing, the gate passes, and the nulling is
         reported and counted rather than silent;
      5. the NEXT pack from the same source reuses the decision — no second ask.
    """

    PLACEHOLDER_ROWS = 6

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory(prefix="acquired_tbc_")
        cls.root = Path(cls._td.name)
        AP.write_blob_tree(cls.root, dob_placeholder_rows=cls.PLACEHOLDER_ROWS)
        cls.storage = Storage(cls.root)
        cls.layout = Layout()
        cls.persistence = ProductionPersistence(cls.storage, cls.layout)
        cls.registry = SourceRegistry(
            "blob://trakt-state/registry/source_registry.yaml", storage=cls.storage)
        cls.out_dir = cls.root / "out"

        # First sight of the source: one-click review, then promote, so later
        # packs route deterministically and the placeholder decision is the only
        # thing left to settle.
        cls.first_pass = cls._backfill(
            selector=BF.PackSelector(pack_key=AP.PACK_KEY))
        cls.first_run_dir = cls._latest_portfolio_dir()
        _AcquiredRunBase._approve_and_promote.__func__(cls, AP.SOURCE_PORTFOLIO_ID)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    @classmethod
    def _backfill(cls, **kw):
        return BF.run_backfill(
            cls.storage, cls.persistence, cls.registry, container="raw-v2",
            out_dir=str(cls.out_dir), **kw)

    @classmethod
    def _latest_portfolio_dir(cls):
        runs = sorted(cls.out_dir.glob("orun_*"), key=lambda p: p.name)
        return runs[-1] / "portfolios" / AP.SOURCE_PORTFOLIO_ID

    def _queue(self, run_dir=None):
        path = (run_dir or self.first_run_dir) / "28c_human_decision_queue.json"
        return json.loads(path.read_text(encoding="utf-8")).get("rows", [])

    def _placeholder_decisions(self, run_dir=None):
        return [d for d in self._queue(run_dir)
                if d.get("decision_type") == "source_value_normalisation"]

    # -- 1 + 2: surfaced, non-blocking, undecided value still fails ---------- #

    def test_placeholder_is_surfaced_as_an_operator_decision(self):
        found = self._placeholder_decisions()
        self.assertTrue(found, "TBC was not surfaced as a decision")
        decision = next(d for d in found
                        if d.get("canonical_field") == "borrower_1_DOB")
        self.assertEqual(decision["source_value"], AP.DOB_PLACEHOLDER)
        self.assertEqual(decision["source_column"], "Borrower 1 DOB")
        self.assertEqual(decision["affected_row_count"], self.PLACEHOLDER_ROWS)

    def test_the_decision_is_not_blocking(self):
        decision = self._placeholder_decisions()[0]
        self.assertFalse(decision["blocking"])
        self.assertIn("treat_source_value_as_null", decision["options"])

    def test_the_decision_carries_its_full_scope(self):
        decision = self._placeholder_decisions()[0]
        self.assertEqual(decision["client_id"], AP.CLIENT_ID)
        self.assertEqual(decision["source_portfolio_id"], AP.SOURCE_PORTFOLIO_ID)
        self.assertEqual(decision["target_contract_id"], "mi_semantics_field_registry")

    def test_an_undecided_placeholder_still_fails_parsing(self):
        issues = json.loads(
            (self.first_run_dir / "output" / "transformation"
             / "35_transformation_issues.json").read_text(encoding="utf-8"))
        date_failures = [r for r in issues["rows"]
                         if r["issue_type"] == "date_parse_failed"
                         and r["canonical_field"] == "borrower_1_DOB"]
        self.assertTrue(date_failures, "undecided TBC should still fail")
        self.assertIn(AP.DOB_PLACEHOLDER, date_failures[0]["source_value_sample"])
        self.assertTrue(date_failures[0]["blocking_for_validation"])

    # -- 3 + 4: approve, rerun, gate passes, nulling is reported ------------- #

    def test_approved_decision_clears_the_gate_and_is_reported(self):
        run_dir = _approve_placeholder_and_rerun(self)

        readiness = json.loads(
            (run_dir / "output" / "transformation"
             / "33_transformation_readiness.json").read_text(encoding="utf-8"))
        self.assertTrue(readiness["ready_for_validation"])
        self.assertEqual(readiness["blocking_for_validation_count"], 0)
        # Non-blocking, but counted — not silent.
        self.assertEqual(readiness["source_value_normalised_row_count"],
                         self.PLACEHOLDER_ROWS)

        issues = json.loads(
            (run_dir / "output" / "transformation"
             / "35_transformation_issues.json").read_text(encoding="utf-8"))
        self.assertEqual(issues["issue_type_counts"].get("date_parse_failed", 0), 0)
        warned = [r for r in issues["rows"]
                  if r["issue_type"] == "source_value_normalised"]
        self.assertEqual(len(warned), 1)
        self.assertFalse(warned[0]["blocking_for_validation"])
        self.assertEqual(warned[0]["source_value_sample"], AP.DOB_PLACEHOLDER)

        tape = pd.read_csv(
            run_dir / "output" / "transformation" / "31_transformed_canonical_tape.csv",
            dtype=str)
        self.assertEqual(len(tape), AP.ROW_COUNT)
        self.assertEqual(int(tape["borrower_1_DOB"].isna().sum()),
                         self.PLACEHOLDER_ROWS)
        # Every surviving value is still a canonical ISO date.
        self.assertTrue(tape["borrower_1_DOB"].dropna()
                        .str.match(r"^\d{4}-\d{2}-\d{2}$").all())

    def test_lineage_records_the_original_value_and_treatment(self):
        run_dir = _approve_placeholder_and_rerun(self)
        lineage = json.loads(
            (run_dir / "output" / "transformation"
             / "34_transformation_lineage.json").read_text(encoding="utf-8"))
        applied = lineage["source_value_normalisations"]
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0]["source_value"], AP.DOB_PLACEHOLDER)
        self.assertEqual(applied[0]["treatment"], "null")
        self.assertEqual(applied[0]["canonical_field"], "borrower_1_DOB")
        self.assertEqual(applied[0]["source_column"], "Borrower 1 DOB")
        self.assertEqual(applied[0]["normalised_row_count"], self.PLACEHOLDER_ROWS)
        self.assertEqual(applied[0]["target_contract_id"],
                         "mi_semantics_field_registry")

        onboarding = json.loads(
            (run_dir / "output" / "handoff"
             / "27_onboarding_handoff_lineage.json").read_text(encoding="utf-8"))
        notes = [r["lineage_note"] for r in onboarding["rows"]]
        self.assertTrue(any("source_value_normalisation" in n and
                            AP.DOB_PLACEHOLDER in n for n in notes))

    # -- 5: reapplied to the next pack, no second ask ------------------------ #

    def test_the_decision_is_reapplied_to_the_next_pack(self):
        _approve_placeholder_and_rerun(self)
        later = "2027-03-31"
        AP.write_blob_tree(self.root, period=later,
                           dob_placeholder_rows=self.PLACEHOLDER_ROWS)
        results = self._backfill(
            selector=BF.PackSelector(pack_key=AP.pack_key_for_period(later)))
        self.assertEqual(results[0]["status"], "processed")

        run_dir = self._latest_portfolio_dir()
        readiness = json.loads(
            (run_dir / "output" / "transformation"
             / "33_transformation_readiness.json").read_text(encoding="utf-8"))
        self.assertTrue(readiness["ready_for_validation"])
        self.assertEqual(readiness["source_value_normalised_row_count"],
                         self.PLACEHOLDER_ROWS)
        # The settled value is not queued again.
        self.assertEqual(
            [d for d in self._placeholder_decisions(run_dir)
             if d.get("source_value") == AP.DOB_PLACEHOLDER], [])


def _approve_placeholder_and_rerun(case):
    """Approve the TBC decision as an operator would, then rerun the pack.

    Cached per class so the several assertions about the approved run share one
    execution rather than repeating the whole pipeline.
    """
    cached = getattr(case.__class__, "_approved_run_dir", None)
    if cached is not None:
        return cached

    # 1. The operator edits the exported template: status approved + the action.
    template = case.first_run_dir / "34_target_first_decisions.yaml"
    doc = yaml.safe_load(template.read_text(encoding="utf-8"))
    approved_any = False
    for entry in doc.get("decisions", []) or []:
        if (entry.get("decision_type") == "source_value_normalisation"
                and entry.get("source_value") == AP.DOB_PLACEHOLDER):
            entry["status"] = "approved"
            entry["selected_action"] = "treat_source_value_as_null"
            entry["approved_by"] = "ops"
            approved_any = True
    assert approved_any, "no TBC decision found in the exported template"

    # 2. Persisted where a scoped rerun looks for approved decisions.
    case.storage.write_text(
        case.layout.run_onboarding_uri(
            AP.PACK_KEY, "34_target_first_decisions_approved.yaml"),
        yaml.safe_dump(doc, sort_keys=False))

    # 3. Rerun the pack; backfill localises and applies the approved decisions.
    case._backfill(selector=BF.PackSelector(pack_key=AP.PACK_KEY), force=True,
                   require_approved_decisions=True)
    case.__class__._approved_run_dir = case._latest_portfolio_dir()

    # 4. PROMOTE the approved decisions onto the SOURCE registry entry. Approval
    #    alone is pack-scoped; promotion is what makes the decision the source's
    #    standing mapping, so every later pack applies it without being asked
    #    again. This is the documented approve → rerun → promote loop.
    OPS.promote_pack(case.persistence, case.registry, AP.PACK_KEY)
    case.registry = SourceRegistry(
        "blob://trakt-state/registry/source_registry.yaml", storage=case.storage)
    return case.__class__._approved_run_dir



if __name__ == "__main__":  # pragma: no cover
    unittest.main()
