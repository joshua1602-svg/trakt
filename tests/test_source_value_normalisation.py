#!/usr/bin/env python3
"""tests/test_source_value_normalisation.py

Operator-approved source-value normalisation.

A source file may carry placeholder text where a value is not yet known — the
acquired tape's ``Borrower 1 DOB`` containing ``TBC``. Deterministic parsing must
reject that, because a parser that quietly swallows unknown text will one day
swallow a real data error. But once an operator has decided what that specific
value means, for that specific column and source, the decision should be recorded
once and reapplied to every later pack.

These cover the whole loop: detect → propose → approve → apply → reapply, and the
boundaries that keep it safe — nothing global, nothing cross-contract, nothing
unapproved.

Run: python -m pytest tests/test_source_value_normalisation.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine.gate_2_transform import canonical_transform as ct
from engine.onboarding_agent import source_value_rules as svr
from engine.onboarding_agent import target_coverage as tcov
from engine.onboarding_agent import target_first_decisions as tfd
from engine.transformation_agent import transformation_agent as ta

MI = "mi_semantics_field_registry"
A2 = "esma_annex_2"
SOURCE_FILE = "AcquiredLoanTape.csv"
SOURCE_COLUMN = "Borrower 1 DOB"
CANONICAL = "borrower_1_DOB"


def _ctx(contract=MI, client="ERE", portfolio="acquired_001"):
    return svr.NormalisationContext(
        client_id=client, source_portfolio_id=portfolio,
        target_contract_id=contract, source_schema_fingerprint="sha256:abc")


def _decision(**over):
    d = {
        "decision_id": "DQ-mi_only-001",
        "decision_type": svr.DECISION_TYPE,
        "selected_action": svr.ACTION_TREAT_AS_NULL,
        "status": "approved",
        "target_field": CANONICAL,
        "source_value": "TBC",
        "canonical_field": CANONICAL,
        "source_column": SOURCE_COLUMN,
        "source_file": SOURCE_FILE,
        "target_contract_id": MI,
        "client_id": "ERE",
        "source_portfolio_id": "acquired_001",
        "approved_by": "ops",
    }
    d.update(over)
    return d


def _contract_row(**over):
    row = {"target_field": CANONICAL, "canonical_field": CANONICAL,
           "selected_source_file": SOURCE_FILE,
           "selected_source_column": SOURCE_COLUMN,
           "target_contract_id": MI}
    row.update(over)
    return row


def _tape(values):
    return pd.DataFrame({CANONICAL: list(values)})


def _contract_rows_or_default(rows):
    """``None`` means "the usual contract row"; ``[]`` means "no contract row"."""
    return _CONTRACT_DEFAULT if rows is None else rows


_CONTRACT_DEFAULT = [_contract_row()]


# --------------------------------------------------------------------------- #
# Detection — propose, never decide
# --------------------------------------------------------------------------- #
class TestDetection(unittest.TestCase):

    def _detect(self, values, fmt="date", existing=None, contract=MI,
                canonical=CANONICAL):
        df = pd.DataFrame({SOURCE_COLUMN: list(values)})
        mapping = svr.mapping_rows_from(
            [{"source_file": SOURCE_FILE, "source_sheet": "",
              "source_column": SOURCE_COLUMN, "resolved_target_field": canonical}])
        return svr.detect_candidates(
            [(SOURCE_FILE, "", df)], mapping, {canonical: {"format": fmt}},
            _ctx(contract), existing_rules=existing)

    def test_mapping_rows_cover_columns_outside_the_target_contract(self):
        # borrower_1_DOB is mapped into the canonical tape but is NOT a field of
        # the MI semantics contract. Driving detection off the coverage matrix
        # alone would never see it — and it is exactly the reported case.
        rows = svr.mapping_rows_from(
            [{"source_file": SOURCE_FILE, "source_column": SOURCE_COLUMN,
              "resolved_target_field": CANONICAL}],
            coverage_rows=[{"target_field": "current_interest_rate",
                            "canonical_field": "current_interest_rate"}])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["canonical_field"], CANONICAL)
        # With no coverage row to name it, the canonical field is its own label.
        self.assertEqual(rows[0]["target_field"], CANONICAL)

    def test_mapping_rows_borrow_the_target_field_label_when_there_is_one(self):
        rows = svr.mapping_rows_from(
            [{"source_file": SOURCE_FILE, "source_column": "Protected Equity",
              "resolved_target_field": "protected_equity_percentage"}],
            coverage_rows=[{"target_field": "protected_equity_percentage",
                            "canonical_field": "protected_equity_percentage"}])
        self.assertEqual(rows[0]["target_field"], "protected_equity_percentage")

    def test_unmapped_columns_produce_no_mapping_rows(self):
        self.assertEqual(svr.mapping_rows_from(
            [{"source_file": SOURCE_FILE, "source_column": "Junk",
              "resolved_target_field": ""}]), [])

    def test_unparseable_value_is_surfaced(self):
        found = self._detect(["01/11/1935", "TBC", "01/01/1944", "TBC"])
        self.assertEqual([c["source_value"] for c in found], ["TBC"])
        self.assertEqual(found[0]["affected_row_count"], 2)
        self.assertEqual(found[0]["canonical_field"], CANONICAL)
        self.assertEqual(found[0]["source_column"], SOURCE_COLUMN)

    def test_candidate_carries_its_full_scope(self):
        c = self._detect(["TBC"])[0]
        self.assertEqual(c["client_id"], "ERE")
        self.assertEqual(c["source_portfolio_id"], "acquired_001")
        self.assertEqual(c["target_contract_id"], MI)
        self.assertEqual(c["source_file"], SOURCE_FILE)
        self.assertEqual(c["source_schema_fingerprint"], "sha256:abc")

    def test_valid_values_produce_no_candidates(self):
        self.assertEqual(self._detect(["01/11/1935", "13/05/1940", ""]), [])

    def test_blanks_and_nd_codes_are_not_candidates(self):
        # Already handled as absent; they are not values needing a decision.
        self.assertEqual(self._detect(["01/11/1935", "", "  ", "nan", "ND5"]), [])

    def test_an_already_approved_value_is_not_re_asked(self):
        rules = svr.rules_from_decisions([_decision()], _ctx())
        found = self._detect(["TBC", "garbage"], existing=rules)
        self.assertEqual([c["source_value"] for c in found], ["garbage"])

    def test_detection_covers_non_date_formats_too(self):
        # Generic by construction — nothing here is date-specific.
        found = self._detect(["20.00%", "n/k"], fmt="percentage")
        self.assertEqual([c["source_value"] for c in found], ["n/k"])

    def test_string_fields_are_not_probed(self):
        # A string column has no deterministic parse to fail.
        self.assertEqual(self._detect(["anything at all"], fmt="string"), [])

    def test_candidates_are_capped_per_column(self):
        many = [f"junk-{i}" for i in range(20)]
        self.assertLessEqual(len(self._detect(many)), svr.MAX_CANDIDATES_PER_COLUMN)


# --------------------------------------------------------------------------- #
# The decision itself
# --------------------------------------------------------------------------- #
class TestDecisionSurface(unittest.TestCase):

    def test_action_is_generic_and_registered(self):
        # Named for the TREATMENT, not for DOB, dates or any client.
        self.assertEqual(svr.ACTION_TREAT_AS_NULL, "treat_source_value_as_null")
        self.assertIn(svr.ACTION_TREAT_AS_NULL, tfd.SUPPORTED_ACTIONS)

    def test_queue_entry_is_non_blocking_and_offers_the_action(self):
        candidate = {
            "target_field": CANONICAL, "canonical_field": CANONICAL,
            "source_file": SOURCE_FILE, "source_column": SOURCE_COLUMN,
            "source_value": "TBC", "affected_row_count": 3,
            "canonical_format": "date", "target_contract_id": MI,
            "client_id": "ERE", "source_portfolio_id": "acquired_001",
        }
        rows = tcov.build_human_decision_queue("mi_only", [], [], [candidate])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["decision_type"], svr.DECISION_TYPE)
        self.assertFalse(row["blocking"])
        self.assertIn(svr.ACTION_TREAT_AS_NULL, row["options"])
        self.assertIn("TBC", row["issue"])

    def test_template_carries_the_scope_for_reapplication(self):
        candidate = {"target_field": CANONICAL, "canonical_field": CANONICAL,
                     "source_file": SOURCE_FILE, "source_column": SOURCE_COLUMN,
                     "source_value": "TBC", "affected_row_count": 3,
                     "canonical_format": "date", "target_contract_id": MI,
                     "client_id": "ERE", "source_portfolio_id": "acquired_001"}
        rows = tcov.build_human_decision_queue("mi_only", [], [], [candidate])
        template = tfd.build_decision_template(rows, "mi_only", client_id="ERE")
        entry = template["decisions"][0]
        for key in ("source_value", "canonical_field", "source_column",
                    "client_id", "source_portfolio_id", "target_contract_id"):
            self.assertTrue(entry.get(key), f"{key} missing from template entry")
        self.assertEqual(entry["status"], "pending")
        self.assertIn(svr.ACTION_TREAT_AS_NULL, template["supported_actions"])

    def test_pending_decision_yields_no_rule(self):
        pending = _decision(status="pending", selected_action=None)
        self.assertEqual(svr.rules_from_decisions([pending], _ctx()), [])

    def test_decision_without_a_source_value_is_rejected(self):
        # Nulling an unspecified value would be catastrophic; never inferred.
        self.assertIsNone(svr.rule_from_decision(_decision(source_value=""), _ctx()))
        self.assertIsNone(svr.rule_from_decision(_decision(source_value=None), _ctx()))

    def test_application_records_the_rule_on_the_coverage_row(self):
        cov = [{"target_field": CANONICAL, "canonical_field": CANONICAL,
                "selected_source_column": SOURCE_COLUMN, "target_contract_id": MI,
                "requires_user_decision": True, "blocking": False}]
        queue = [{"decision_id": "DQ-mi_only-001", "target_field": CANONICAL}]
        cov, remaining, log = tfd.apply_decisions(cov, queue, [_decision()])

        self.assertEqual(log[0]["application_status"], tfd.APPLIED)
        self.assertEqual(remaining, [])
        rules = cov[0]["source_value_rules"]
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["source_value"], "TBC")
        self.assertEqual(rules[0]["treatment"], svr.TREATMENT_NULL)
        self.assertEqual(rules[0]["target_contract_id"], MI)
        # The field is still source-mapped; only some of its values are absent.
        self.assertNotIn("coverage_status", [k for k in cov[0] if k == "configured_static"])

    def test_application_is_idempotent(self):
        cov = [{"target_field": CANONICAL, "canonical_field": CANONICAL,
                "selected_source_column": SOURCE_COLUMN, "target_contract_id": MI}]
        queue = [{"decision_id": "DQ-mi_only-001", "target_field": CANONICAL}]
        cov, _r, _l = tfd.apply_decisions(cov, queue, [_decision()])
        cov, _r, _l = tfd.apply_decisions(cov, queue, [_decision()])
        self.assertEqual(len(cov[0]["source_value_rules"]), 1)


# --------------------------------------------------------------------------- #
# Scope
# --------------------------------------------------------------------------- #
class TestScope(unittest.TestCase):

    def setUp(self):
        self.rules = svr.rules_from_decisions([_decision()], _ctx())

    def _applies(self, ctx, contract_rows=None, column=CANONICAL):
        """Whether the approved rule actually nulls anything on a real tape.

        Exercised through ``apply_rules`` — the path production uses — rather
        than a predicate, so the tests cannot pass while the real path differs.
        """
        df = pd.DataFrame({column: ["TBC", "01/11/1935"]})
        reports = svr.apply_rules(
            df, self.rules,
            _contract_rows_or_default(contract_rows), ctx)
        return bool(reports)

    def test_applies_within_its_scope(self):
        self.assertTrue(self._applies(_ctx()))

    def test_does_not_apply_to_another_target_contract(self):
        # A regulatory contract maintains its own treatment, or none.
        self.assertFalse(self._applies(_ctx(contract=A2)))

    def test_does_not_apply_to_another_client(self):
        self.assertFalse(self._applies(_ctx(client="OTHER")))

    def test_does_not_apply_to_another_source_portfolio(self):
        self.assertFalse(self._applies(_ctx(portfolio="direct_001")))

    def test_does_not_apply_when_the_field_is_now_mapped_from_another_column(self):
        # The mapping changed, so the approved decision is about something else
        # and must be re-taken rather than blindly reused.
        self.assertFalse(self._applies(
            _ctx(), contract_rows=[_contract_row(
                selected_source_column="Some Other Column")]))

    def test_does_not_touch_another_canonical_field(self):
        df = pd.DataFrame({"borrower_2_DOB": ["TBC"]})
        svr.apply_rules(df, self.rules, [_contract_row()], _ctx())
        self.assertEqual(df["borrower_2_DOB"].tolist(), ["TBC"])

    def test_applies_when_the_field_is_not_a_target_contract_row_at_all(self):
        # borrower_1_DOB is mapped into the tape but is not an MI target field:
        # with no contract row to check against, the rule still applies.
        self.assertTrue(self._applies(_ctx(), contract_rows=[]))

    def test_value_matching_ignores_case_and_padding(self):
        rule = self.rules[0]
        self.assertEqual(rule.value_key, "tbc")
        df = _tape(["  tbc  ", "TBC", "Tbc"])
        report = svr.apply_rules_to_column(df, CANONICAL, self.rules)
        self.assertEqual(report["normalised_row_count"], 3)


# --------------------------------------------------------------------------- #
# Nothing global
# --------------------------------------------------------------------------- #
class TestNoGlobalLeakage(unittest.TestCase):

    def test_tbc_is_not_a_global_null_marker(self):
        self.assertNotIn("tbc", ct._BLANK_TOKENS)

    def test_an_unscoped_column_still_fails_on_tbc(self):
        # Same value, same format, no approved rule → still a strict failure.
        df = pd.DataFrame({"borrower_2_DOB": ["01/11/1935", "TBC"]})
        report = ct.apply_types(df, {"borrower_2_DOB": {"format": "date"}})
        self.assertEqual(report["fields"]["borrower_2_DOB"]["parse_failures"], 1)
        self.assertEqual(report["fields"]["borrower_2_DOB"]["sample_failures"], ["TBC"])


# --------------------------------------------------------------------------- #
# Application, before parsing
# --------------------------------------------------------------------------- #
class TestApplication(unittest.TestCase):

    def setUp(self):
        self.rules = svr.rules_from_decisions([_decision()], _ctx())

    def _apply_then_type(self, values):
        df = _tape(values)
        reports = svr.apply_rules(df, self.rules, [_contract_row()], _ctx())
        type_report = ct.apply_types(df, {CANONICAL: {"format": "date"}})
        return df, reports, type_report

    def test_approved_value_becomes_null_not_a_parse_failure(self):
        df, reports, type_report = self._apply_then_type(
            ["01/11/1935", "TBC", "01/01/1944"])
        self.assertEqual(df[CANONICAL].tolist()[0], "1935-11-01")
        self.assertTrue(pd.isna(df[CANONICAL].tolist()[1]))
        self.assertEqual(type_report["fields"][CANONICAL]["parse_failures"], 0)
        self.assertEqual(reports[CANONICAL]["normalised_row_count"], 1)

    def test_unapproved_invalid_value_still_fails_strictly(self):
        _df, _reports, type_report = self._apply_then_type(
            ["01/11/1935", "TBC", "garbage", "31/02/1990"])
        self.assertEqual(type_report["fields"][CANONICAL]["parse_failures"], 2)
        self.assertEqual(sorted(type_report["fields"][CANONICAL]["sample_failures"]),
                         ["31/02/1990", "garbage"])

    def test_valid_values_are_never_touched(self):
        df, _r, _t = self._apply_then_type(["01/11/1935", "TBC", "13/05/1940"])
        self.assertEqual(df[CANONICAL].tolist()[0], "1935-11-01")
        self.assertEqual(df[CANONICAL].tolist()[2], "1940-05-13")

    def test_report_records_the_original_value_and_treatment(self):
        _df, reports, _t = self._apply_then_type(["TBC", "TBC"])
        rule = reports[CANONICAL]["rules"][0]
        self.assertEqual(rule["source_value"], "TBC")
        self.assertEqual(rule["treatment"], svr.TREATMENT_NULL)
        self.assertEqual(rule["normalised_row_count"], 2)
        self.assertEqual(rule["decision_id"], "DQ-mi_only-001")

    def test_a_field_with_no_rules_is_untouched(self):
        df = pd.DataFrame({"borrower_2_DOB": ["TBC"]})
        reports = svr.apply_rules(
            df, self.rules,
            [_contract_row(canonical_field="borrower_2_DOB",
                           selected_source_column="Borrower 2 DOB")], _ctx())
        self.assertEqual(reports, {})
        self.assertEqual(df["borrower_2_DOB"].tolist(), ["TBC"])


# --------------------------------------------------------------------------- #
# Transformation wiring: warning, count, lineage
# --------------------------------------------------------------------------- #
class TestTransformationWiring(unittest.TestCase):

    def _run(self, values, handoff_extra=None):
        df = _tape(values)
        issues = ta._IssueLog()
        handoff = {
            # The manifest's own client_id is the SOURCE PORTFOLIO at the
            # onboarding layer — deliberately set to the wrong-looking value here
            # so the test proves the explicit scope block is what is used.
            "client_id": "acquired_001",
            "target_contract_id": MI,
            "source_value_scope": {
                "client_id": "ERE", "source_portfolio_id": "acquired_001",
                "target_contract_id": MI},
            "source_value_rules": [
                svr.rules_from_decisions([_decision()], _ctx())[0].as_row()],
        }
        handoff.update(handoff_extra or {})
        reports = ta._apply_source_value_rules(df, [_contract_row()], issues, handoff)
        return df, issues, reports

    def test_normalisation_is_a_non_blocking_warning(self):
        _df, issues, _r = self._run(["01/11/1935", "TBC", "TBC"])
        rows = [i for i in issues.rows
                if i["issue_type"] == ta.IT_SOURCE_VALUE_NORMALISED]
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["blocking_for_validation"])
        self.assertFalse(rows[0]["blocking_for_projection"])
        self.assertEqual(rows[0]["severity"], "info")

    def test_warning_counts_the_rows_and_names_the_value(self):
        _df, issues, _r = self._run(["TBC", "TBC", "01/11/1935"])
        row = next(i for i in issues.rows
                   if i["issue_type"] == ta.IT_SOURCE_VALUE_NORMALISED)
        self.assertIn("2 row(s)", row["description"])
        self.assertEqual(row["source_value_sample"], "TBC")
        self.assertIn("DQ-mi_only-001", row["description"])

    def test_no_warning_when_nothing_matched(self):
        _df, issues, _r = self._run(["01/11/1935", "13/05/1940"])
        self.assertEqual([i for i in issues.rows
                          if i["issue_type"] == ta.IT_SOURCE_VALUE_NORMALISED], [])

    def test_rules_are_ignored_for_a_different_target_contract(self):
        # The handoff says this run is regulatory; the MI-approved rule must not
        # fire, and TBC stays put to fail parsing.
        df, issues, reports = self._run(
            ["TBC"], handoff_extra={"source_value_scope": {
                "client_id": "ERE", "source_portfolio_id": "acquired_001",
                "target_contract_id": A2}})
        self.assertEqual(reports, {})
        self.assertEqual(df[CANONICAL].tolist(), ["TBC"])
        self.assertEqual(issues.rows, [])

    def test_rules_can_arrive_on_the_contract_rows_alone(self):
        # Backwards compatibility with a handoff that has no manifest roll-up.
        df = _tape(["TBC"])
        issues = ta._IssueLog()
        rule_row = svr.rules_from_decisions([_decision()], _ctx())[0].as_row()
        reports = ta._apply_source_value_rules(
            df, [_contract_row(source_value_rules=[rule_row])], issues,
            {"source_value_scope": {"client_id": "ERE",
                                    "source_portfolio_id": "acquired_001",
                                    "target_contract_id": MI}})
        self.assertEqual(reports[CANONICAL]["normalised_row_count"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
