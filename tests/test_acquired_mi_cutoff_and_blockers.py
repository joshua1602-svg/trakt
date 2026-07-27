#!/usr/bin/env python3
"""tests/test_acquired_mi_cutoff_and_blockers.py

Two onboarding regressions raised by the ERE ``acquired_001`` MI pack.

**Mixed row-level cut-off dates.** The acquired portfolio is one commercial book
assembled from two legacy rump populations, and the rumps were cut on different
dates. The run-context resolver treated two distinct ``data_cut_off_date`` values
in the source column as a conflict and raised a blocking operator decision. For
the MI contract that is legitimate portfolio data: the dates must be preserved
per loan, the blob folder period stays pack context, and nothing blocks. For a
regulatory contract — one submission, one reporting period — it must still block.

**Stale blockers.** ``blocking_decision_count`` was read from the 28c decision
queue summary, a snapshot taken BEFORE approved decisions were applied. An
approved decision that resolved the last blocking row therefore left the pack
reporting a blocker with an empty queue and no blocking contract row.

Run: python -m pytest tests/test_acquired_mi_cutoff_and_blockers.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine.onboarding_agent import onboarding_handoff as oh
from engine.onboarding_agent import run_context as rc
from engine.transformation_agent import transformation_agent as ta

MI_CONTRACT = "mi_semantics_field_registry"
A2_CONTRACT = "esma_annex_2"

RUMP_A = "2026-05-31"
RUMP_B = "2026-06-30"
FOLDER_PERIOD = "2026-06-30"


def _central_tape(root: Path, cut_offs) -> Path:
    """A central lender tape carrying one ``data_cut_off_date`` value per row."""
    path = root / "output" / "central" / "18_central_lender_tape.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["loan_identifier,data_cut_off_date,current_outstanding_balance"]
    for i, cut in enumerate(cut_offs):
        lines.append(f"L{i:04d},{cut},{100000 + i}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _coverage(root: Path, contract_id: str, rows) -> None:
    (root / "28a_target_coverage_matrix.json").write_text(
        json.dumps({"target_contract_id": contract_id, "rows": rows}), encoding="utf-8")


def _cutoff_row(**over):
    row = {"target_field": "data_cut_off_date", "canonical_field": "data_cut_off_date",
           "esma_code": "", "coverage_status": "source_mapped",
           "selected_source_file": "AcquiredLoanTape.csv",
           "selected_source_column": "Cut Off Date",
           "requires_user_decision": False, "blocking": False}
    row.update(over)
    return row


class _Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.addCleanup(self.td.cleanup)

    def _handoff(self, *, contract_id=MI_CONTRACT, cut_offs=None, rows=None,
                 reporting_period=FOLDER_PERIOD, **kw):
        cut_offs = cut_offs if cut_offs is not None else ([RUMP_A] * 5 + [RUMP_B] * 3)
        _central_tape(self.root, cut_offs)
        _coverage(self.root, contract_id, rows if rows is not None else [_cutoff_row()])
        return oh.build_handoff_package(
            self.root, self.root / "output", client_id="ERE", run_id="orun_test",
            mode="mi_only" if contract_id == MI_CONTRACT else "regulatory_mi",
            reporting_period=reporting_period, managed_service=True, **kw)


# --------------------------------------------------------------------------- #
# Resolver-level policy
# --------------------------------------------------------------------------- #
class TestRunContextPolicy(_Base):

    def _extract(self, contract_id, cut_offs):
        tape = _central_tape(self.root, cut_offs)
        return rc.extract_data_cut_off_date(
            self.root, tape, folder_period=FOLDER_PERIOD,
            managed_service=True, target_contract_id=contract_id)

    def test_mi_accepts_mixed_row_level_cut_off_dates(self):
        # 1 — the reported blocker: two rump books, two cut-offs, one MI tape.
        res = self._extract(MI_CONTRACT, [RUMP_A] * 5 + [RUMP_B] * 3)
        self.assertFalse(res["conflict"])
        self.assertTrue(res["row_level"])
        self.assertEqual(res["row_level_distinct_values"], [RUMP_A, RUMP_B])
        self.assertEqual(res["source"], rc.SRC_SOURCE_ROW_LEVEL)

    def test_regulatory_contract_still_blocks_on_mixed_cut_off_dates(self):
        # 5 — one submission, one reporting period. Unchanged, deliberately.
        res = self._extract(A2_CONTRACT, [RUMP_A] * 5 + [RUMP_B] * 3)
        self.assertTrue(res["conflict"])
        self.assertFalse(res["row_level"])
        self.assertIn(RUMP_A, res["conflict_detail"])

    def test_unknown_contract_keeps_the_strict_default(self):
        # Permissiveness is opt-in per contract; an unrecognised contract does
        # not silently inherit MI behaviour.
        res = self._extract("some_future_contract", [RUMP_A, RUMP_B])
        self.assertTrue(res["conflict"])

    def test_single_cut_off_date_resolves_normally_for_mi(self):
        # Backwards compatibility for existing direct portfolios.
        res = self._extract(MI_CONTRACT, [RUMP_B] * 4)
        self.assertFalse(res["conflict"])
        self.assertFalse(res["row_level"])
        self.assertEqual(res["value"], RUMP_B)
        self.assertEqual(res["source"], rc.SRC_SOURCE_COLUMN)

    def test_folder_period_is_recorded_as_pack_context(self):
        # 3 — the period folder stays available even when it is not the value.
        res = self._extract(MI_CONTRACT, [RUMP_A] * 3 + [RUMP_B] * 2)
        self.assertEqual(res["folder_period"], FOLDER_PERIOD)
        self.assertEqual(res["folder_period_date"], FOLDER_PERIOD)

    def test_folder_period_still_resolves_when_the_source_has_none(self):
        # A pack whose source carries no cut-off column keeps working.
        tape = self.root / "output" / "central" / "18_central_lender_tape.csv"
        tape.parent.mkdir(parents=True, exist_ok=True)
        tape.write_text("loan_identifier\nL1\n", encoding="utf-8")
        res = rc.extract_data_cut_off_date(
            self.root, tape, folder_period=FOLDER_PERIOD, managed_service=True,
            target_contract_id=MI_CONTRACT)
        self.assertEqual(res["value"], FOLDER_PERIOD)
        self.assertEqual(res["source"], rc.SRC_FOLDER_PERIOD)

    def test_policy_is_contract_scoped_configuration(self):
        self.assertEqual(
            rc.load_run_context_policy("data_cut_off_date", MI_CONTRACT),
            {"row_level_variation": rc.VARIATION_ALLOWED})
        self.assertEqual(
            rc.load_run_context_policy("data_cut_off_date", A2_CONTRACT),
            {"row_level_variation": rc.VARIATION_CONFLICT})
        # Anything unlisted keeps the strict default.
        self.assertEqual(
            rc.load_run_context_policy("some_other_field", MI_CONTRACT),
            {"row_level_variation": rc.VARIATION_CONFLICT})


# --------------------------------------------------------------------------- #
# Handoff package
# --------------------------------------------------------------------------- #
class TestHandoffCutOffHandling(_Base):

    def test_mixed_cut_off_dates_do_not_block_mi_onboarding(self):
        # 1 — end state: ready for transformation, zero blockers.
        res = self._handoff()
        manifest, readiness = res["manifest"], res["readiness"]
        self.assertTrue(manifest["data_cut_off_date_row_level"])
        self.assertEqual(readiness["blocking_decision_count"], 0)
        self.assertEqual(readiness["operator_decision_pending_count"], 0)
        self.assertTrue(readiness["ready_for_transformation_validation"])

    def test_row_level_values_are_recorded_in_the_manifest(self):
        manifest = self._handoff()["manifest"]
        self.assertEqual(manifest["data_cut_off_date_row_level_values"],
                         [RUMP_A, RUMP_B])
        self.assertEqual(manifest["data_cut_off_date_source"],
                         rc.SRC_SOURCE_ROW_LEVEL)

    def test_folder_period_recorded_as_pack_context_not_as_the_value(self):
        # 3 + 4 — available as context; never presented as the portfolio's date.
        manifest = self._handoff()["manifest"]
        self.assertEqual(manifest["reporting_period_context"], FOLDER_PERIOD)
        self.assertEqual(manifest["data_cut_off_date"], "")

    def test_contract_row_is_resolved_and_preserves_row_values(self):
        res = self._handoff()
        rows = json.loads(Path(res["field_contract_json_path"]).read_text())["rows"]
        row = next(r for r in rows if r["target_field"] == "data_cut_off_date")
        self.assertEqual(row["handoff_status"], "resolved")
        self.assertEqual(row["next_agent_action"], "preserve_row_level_source_values")
        self.assertFalse(row["blocking_decision"])

    def test_lineage_records_the_row_level_resolution(self):
        res = self._handoff()
        lineage = json.loads(Path(res["lineage_path"]).read_text())["rows"]
        notes = [r["lineage_note"] for r in lineage
                 if r["canonical_field"] == "data_cut_off_date"]
        self.assertTrue(any("run_context_row_level" in n for n in notes))
        self.assertTrue(any("source row values preserved" in n for n in notes))

    def test_regulatory_contract_still_raises_a_blocking_decision(self):
        # 5 — the regulatory control is intact end-to-end, not just in the resolver.
        res = self._handoff(contract_id=A2_CONTRACT)
        readiness = res["readiness"]
        self.assertEqual(readiness["blocking_decision_count"], 1)
        self.assertFalse(readiness["ready_for_transformation_validation"])
        rows = json.loads(Path(res["field_contract_json_path"]).read_text())["rows"]
        row = next(r for r in rows if r["target_field"] == "data_cut_off_date")
        self.assertEqual(row["handoff_status"], "blocking")


# --------------------------------------------------------------------------- #
# Transformation: row values survive
# --------------------------------------------------------------------------- #
class TestRowLevelDatesSurviveTransformation(_Base):

    def _df(self, cut_offs):
        import pandas as pd
        return pd.DataFrame({"loan_identifier": [f"L{i}" for i in range(len(cut_offs))],
                             "data_cut_off_date": list(cut_offs)})

    def _contract_rows(self, resolved=RUMP_A):
        # The contract row carries whatever value onboarding RESOLVED for the
        # field; Transformation materialises that, it does not re-resolve.
        return [{"target_field": "data_cut_off_date",
                 "canonical_field": "data_cut_off_date", "esma_code": "",
                 "handoff_classification": oh.HC_SOURCE_CONTEXT_MAPPED,
                 "selected_value_sample": resolved}]

    def test_source_row_dates_are_unchanged(self):
        # 2 — each rump keeps its own cut-off.
        df = self._df([RUMP_A, RUMP_A, RUMP_B, RUMP_B])
        ta._materialise_run_context_fields(
            df, self._contract_rows(), ta._IssueLog(),
            {"data_cut_off_date_row_level": True,
             "run_context_fields": ["data_cut_off_date"]})
        self.assertEqual(df["data_cut_off_date"].tolist(),
                         [RUMP_A, RUMP_A, RUMP_B, RUMP_B])

    def test_folder_period_does_not_overwrite_row_level_dates(self):
        # 4 — the pack period is 2026-06-30; rump A rows must NOT become it.
        df = self._df([RUMP_A, RUMP_A, RUMP_B])
        ta._materialise_run_context_fields(
            df, self._contract_rows(), ta._IssueLog(),
            {"data_cut_off_date_row_level": True,
             "data_cut_off_date": FOLDER_PERIOD,
             "data_cut_off_date_source": rc.SRC_FOLDER_PERIOD,
             "run_context_fields": ["data_cut_off_date"]})
        self.assertEqual(df["data_cut_off_date"].tolist(), [RUMP_A, RUMP_A, RUMP_B])
        self.assertEqual(df["data_cut_off_date"].tolist().count(FOLDER_PERIOD), 1)

    def test_resolved_pack_date_fills_only_rows_without_a_date(self):
        # 4 — even in the single-value case, a real source date is never replaced.
        df = self._df([RUMP_A, "", RUMP_A])
        ta._materialise_run_context_fields(
            df, self._contract_rows(FOLDER_PERIOD), ta._IssueLog(),
            {"data_cut_off_date": FOLDER_PERIOD,
             "data_cut_off_date_source": rc.SRC_FOLDER_PERIOD,
             "run_context_fields": ["data_cut_off_date"]})
        self.assertEqual(df["data_cut_off_date"].tolist(),
                         [RUMP_A, FOLDER_PERIOD, RUMP_A])

    def test_explicit_operator_override_still_replaces_the_column(self):
        # The one case where replacing source values IS the instruction.
        df = self._df([RUMP_A, RUMP_B])
        ta._materialise_run_context_fields(
            df, self._contract_rows("2026-07-31"), ta._IssueLog(),
            {"data_cut_off_date": "2026-07-31",
             "data_cut_off_date_source": rc.SRC_CLI_OVERRIDE,
             "run_context_fields": ["data_cut_off_date"]})
        self.assertEqual(set(df["data_cut_off_date"]), {"2026-07-31"})

    def test_mixed_row_dates_do_not_produce_a_date_parse_failure(self):
        df = self._df([RUMP_A, RUMP_B, ""])
        ta._materialise_run_context_fields(
            df, self._contract_rows(), ta._IssueLog(),
            {"data_cut_off_date_row_level": True,
             "run_context_fields": ["data_cut_off_date"]})
        rf = ta.g2.load_registry_fields("config/system/fields_registry.yaml")
        report = ta.g2.normalize_types(df, rf)
        self.assertEqual(report["fields"]["data_cut_off_date"]["parse_failures"], 0)


# --------------------------------------------------------------------------- #
# Final-state blocker counting
# --------------------------------------------------------------------------- #
class TestFinalStateBlockerCount(_Base):

    def _blocking_row(self):
        return {"target_field": "current_principal_balance",
                "canonical_field": "current_principal_balance", "esma_code": "",
                "coverage_status": "missing_required",
                "requires_user_decision": True, "blocking": True,
                "decision_reason": "required target field has no source"}

    def _queue(self, blocking_decisions: int, rows):
        (self.root / "28c_human_decision_queue.json").write_text(
            json.dumps({"summary": {"blocking_decisions": blocking_decisions},
                        "rows": rows}), encoding="utf-8")

    def _applied(self, fields):
        (self.root / "35_target_first_decision_application_log.json").write_text(
            json.dumps({"rows": [{"target_field": f, "result": "applied"}
                                 for f in fields]}), encoding="utf-8")

    def test_unresolved_blocking_row_is_counted(self):
        # Control: a genuine blocker is still a blocker.
        self._queue(1, [{"target_field": "current_principal_balance", "blocking": True,
                         "decision_id": "D-1"}])
        res = self._handoff(rows=[_cutoff_row(), self._blocking_row()])
        self.assertEqual(res["readiness"]["blocking_decision_count"], 1)
        self.assertFalse(res["readiness"]["ready_for_transformation_validation"])

    def test_approved_decision_resolving_the_last_row_gives_zero(self):
        # 6 — the approved decision is applied, so nothing blocks.
        self._queue(1, [{"target_field": "current_principal_balance", "blocking": True,
                         "decision_id": "D-1"}])
        self._applied(["current_principal_balance"])
        res = self._handoff(rows=[_cutoff_row(), self._blocking_row()])
        self.assertEqual(res["readiness"]["blocking_decision_count"], 0)
        self.assertTrue(res["readiness"]["ready_for_transformation_validation"])

    def test_no_stale_pre_application_blocker_survives(self):
        # 7 — the 28c summary still SAYS 1; the final contract says 0, and the
        # final contract wins.
        self._queue(1, [{"target_field": "current_principal_balance", "blocking": True,
                         "decision_id": "D-1"}])
        self._applied(["current_principal_balance"])
        res = self._handoff(rows=[_cutoff_row(), self._blocking_row()])
        readiness = res["readiness"]
        self.assertEqual(readiness["blocking_decision_count"], 0)
        self.assertEqual(readiness["blocking_decision_source"], "final_field_contract")
        self.assertEqual(readiness["blocking_contract_rows"], [])

    def test_empty_queue_with_no_blocking_rows_cannot_block(self):
        # 6 — the reported end state: empty queue, no blocking rows, zero.
        self._queue(0, [])
        res = self._handoff()
        self.assertEqual(res["readiness"]["blocking_decision_count"], 0)
        self.assertEqual(res["readiness"]["operator_decision_pending_count"], 0)
        self.assertTrue(res["readiness"]["ready_for_transformation_validation"])

    def test_blocking_rows_are_named_in_the_manifest(self):
        # A blocker must be identifiable, not just a number.
        self._queue(1, [{"target_field": "current_principal_balance", "blocking": True,
                         "decision_id": "D-1"}])
        res = self._handoff(rows=[_cutoff_row(), self._blocking_row()])
        named = [r["target_field"] for r in res["readiness"]["blocking_contract_rows"]]
        self.assertEqual(named, ["current_principal_balance"])

    def test_count_is_derived_from_the_contract_not_a_side_counter(self):
        contract = [{"handoff_status": "blocking"}, {"handoff_status": "resolved"},
                    {"handoff_status": "blocking"}]
        count, unrepresented = oh.count_final_blockers(contract)
        self.assertEqual((count, unrepresented), (2, False))

    def test_run_context_conflict_with_no_contract_row_is_still_visible(self):
        # A conflict the contract cannot express is surfaced explicitly rather
        # than added invisibly to a counter.
        count, unrepresented = oh.count_final_blockers(
            [{"handoff_status": "resolved"}],
            context_conflict=True, context_rows_present=0)
        self.assertEqual((count, unrepresented), (1, True))

    def test_run_context_conflict_carried_by_a_contract_row_is_not_double_counted(self):
        count, unrepresented = oh.count_final_blockers(
            [{"handoff_status": "blocking"}],
            context_conflict=True, context_rows_present=1)
        self.assertEqual((count, unrepresented), (1, False))


class TestClassifyFieldRespectsAppliedDecisions(unittest.TestCase):

    def test_pending_blocking_decision_blocks(self):
        row = {"coverage_status": "missing_required", "requires_user_decision": True,
               "blocking": True}
        self.assertEqual(oh.classify_field(row)["handoff_status"], "blocking")

    def test_applied_decision_clears_the_block(self):
        row = {"coverage_status": "source_mapped", "requires_user_decision": True,
               "blocking": True}
        out = oh.classify_field(row, decision_applied=True)
        self.assertNotEqual(out["handoff_status"], "blocking")
        self.assertEqual(out["handoff_classification"],
                         oh.HC_APPROVED_DECISION_APPLIED)

    def test_applied_decision_clears_a_blocking_missing_required(self):
        row = {"coverage_status": "missing_required", "requires_user_decision": False,
               "blocking": True}
        out = oh.classify_field(row, decision_applied=True)
        self.assertNotEqual(out["handoff_status"], "blocking")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
