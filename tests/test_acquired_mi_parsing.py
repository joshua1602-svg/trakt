#!/usr/bin/env python3
"""tests/test_acquired_mi_parsing.py

Deterministic parsing + derivation regressions raised by the ERE ``acquired_001``
MI pack (``AcquiredLoanTape.csv``).

Two transformation blockers motivated these:

  * ``borrower_1_DOB`` / ``borrower_2_DOB`` reported ``date_parse_failed`` on a
    normalised UK tape, because the parser relied on pandas inferring ONE format
    from the first non-null element, and because a blank second-borrower DOB was
    counted as a failed parse rather than an absent value;

  * ``protected_equity_flag`` reported ``boolean_parse_failed``, because the
    source column ``Protected Equity`` (percentage TEXT such as ``20.00%``) was
    mapped straight into the flag and handed to a Y/N parser.

Run: python -m pytest tests/test_acquired_mi_parsing.py
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
from engine.transformation_agent import canonical_derivations as cd
from engine.transformation_agent import transformation_agent as ta

_DATE_META = {"borrower_1_DOB": {"format": "date"},
              "borrower_2_DOB": {"format": "date"}}


def _dob_frame(b1, b2=None):
    return pd.DataFrame({"borrower_1_DOB": list(b1),
                         "borrower_2_DOB": list(b2 if b2 is not None else b1)})


# --------------------------------------------------------------------------- #
# DOB parsing
# --------------------------------------------------------------------------- #
class TestDobParsing(unittest.TestCase):
    """UK ``DD/MM/YYYY`` dates of birth, with day 01 supplied by source convention."""

    def test_uk_dob_1935_parses_day_first(self):
        # 11 — 01/11/1935 is 1 November 1935, not 11 January 1935.
        out = ct.to_iso_date(pd.Series(["01/11/1935"], dtype=object))
        self.assertEqual(out.tolist(), ["1935-11-01"])

    def test_uk_dob_1944_parses_day_first(self):
        # 12 — an all-ones date must still resolve deterministically.
        out = ct.to_iso_date(pd.Series(["01/01/1944"], dtype=object))
        self.assertEqual(out.tolist(), ["1944-01-01"])

    def test_supplied_day_01_is_preserved_not_reinterpreted(self):
        # The provider knows month + year and sets day 01. That day is DATA.
        out = ct.to_iso_date(pd.Series(["01/11/1935", "01/01/1944", "01/03/1950"],
                                       dtype=object))
        self.assertEqual(out.tolist(), ["1935-11-01", "1944-01-01", "1950-03-01"])

    def test_unambiguous_day_first_value_confirms_the_convention(self):
        # 13/05/1940 can only be day-first, and must agree with the ladder.
        out = ct.to_iso_date(pd.Series(["13/05/1940"], dtype=object))
        self.assertEqual(out.tolist(), ["1940-05-13"])

    def test_first_row_does_not_decide_the_column_format(self):
        # The defect: pandas infers ONE format from element 0 and coerces the
        # rest. An explicit ladder must parse every row on its own merits.
        values = ["01/11/1935", "1944-01-01", "13/05/1940", "1/3/1950"]
        out = ct.to_iso_date(pd.Series(values, dtype=object))
        self.assertEqual(out.tolist(),
                         ["1935-11-01", "1944-01-01", "1940-05-13", "1950-03-01"])

    def test_blank_second_borrower_dob_stays_null(self):
        # 13 — a loan with one borrower has no second DOB. Null, not a failure.
        df = _dob_frame(["01/11/1935", "01/01/1944"], ["01/11/1938", ""])
        ct.apply_types(df, _DATE_META)
        self.assertEqual(df["borrower_2_DOB"].tolist()[0], "1938-11-01")
        self.assertTrue(pd.isna(df["borrower_2_DOB"].tolist()[1]))

    def test_valid_uk_dob_produces_no_date_parse_failed(self):
        # 14 — the reported blocker, in both columns, including blanks.
        df = _dob_frame(["01/11/1935", "01/01/1944", "13/05/1940"],
                        ["01/11/1938", "", "  "])
        report = ct.apply_types(df, _DATE_META)
        for col in ("borrower_1_DOB", "borrower_2_DOB"):
            self.assertEqual(report["fields"][col]["parse_failures"], 0, col)
        self.assertEqual(report["fields"]["borrower_2_DOB"]["blank_values"], 2)

        issues = ta._IssueLog()
        ta._record_parse_issues(report, {}, [], issues)
        self.assertEqual([i for i in issues.rows
                          if i["issue_type"] == ta.IT_DATE_PARSE], [])

    def test_invalid_dates_still_fail_deterministically(self):
        # 15 — validation is NOT weakened: nonsense and impossible dates fail.
        df = pd.DataFrame({"borrower_1_DOB": ["01/11/1935", "not-a-date",
                                              "31/02/1990", "99/99/9999"]})
        report = ct.apply_types(df, {"borrower_1_DOB": {"format": "date"}})
        self.assertEqual(report["fields"]["borrower_1_DOB"]["parse_failures"], 3)
        self.assertEqual(df["borrower_1_DOB"].tolist()[0], "1935-11-01")

        issues = ta._IssueLog()
        ta._record_parse_issues(report, {}, [], issues)
        self.assertEqual(len([i for i in issues.rows
                              if i["issue_type"] == ta.IT_DATE_PARSE]), 1)

    def test_month_precision_is_recorded_from_the_source(self):
        # Provenance, not inference: every parsed value landing on day 01 is
        # recorded as month precision with a source-supplied day.
        df = _dob_frame(["01/11/1935", "01/01/1944", "01/03/1950"])
        report = ct.apply_types(df, _DATE_META)
        precision = report["fields"]["borrower_1_DOB"]["date_precision"]
        self.assertEqual(precision["precision"], "month")
        self.assertEqual(precision["day_convention"], "source_supplied_01")
        self.assertEqual(precision["day_01_count"], 3)

    def test_day_precision_recorded_when_days_genuinely_vary(self):
        df = _dob_frame(["01/11/1935", "13/05/1940", "22/07/1951"])
        report = ct.apply_types(df, _DATE_META)
        precision = report["fields"]["borrower_1_DOB"]["date_precision"]
        self.assertEqual(precision["precision"], "day")
        self.assertEqual(precision["day_convention"], "")


# --------------------------------------------------------------------------- #
# Protected equity — percentage parsing
# --------------------------------------------------------------------------- #
class TestProtectedEquityPercentage(unittest.TestCase):

    def _typed(self, values):
        df = pd.DataFrame({"protected_equity_percentage": list(values)})
        report = ct.apply_types(
            df, {"protected_equity_percentage": {"format": "percentage"}})
        return df, report

    def test_canonical_scale_is_percentage_points(self):
        # 17 / 18 — the repository convention (see current_loan_to_value =
        # (balance/valuation)*100): 20.00% is 20.0, NOT 0.20.
        df, _ = self._typed(["20.00%", "50.00%", "30.00%"])
        self.assertEqual(df["protected_equity_percentage"].tolist(),
                         [20.0, 50.0, 30.0])

    def test_zero_percent_is_numeric_zero_not_null(self):
        # 16 — zero is a real, known value and must survive as zero.
        df, report = self._typed(["0.00%"])
        self.assertEqual(df["protected_equity_percentage"].tolist(), [0.0])
        self.assertEqual(report["fields"]["protected_equity_percentage"]["parse_failures"], 0)

    def test_blank_percentage_is_null(self):
        # 19 (percentage half)
        df, report = self._typed(["20.00%", "", "  "])
        values = df["protected_equity_percentage"].tolist()
        self.assertEqual(values[0], 20.0)
        self.assertTrue(pd.isna(values[1]) and pd.isna(values[2]))
        self.assertEqual(report["fields"]["protected_equity_percentage"]["parse_failures"], 0)

    def test_malformed_percentage_is_a_controlled_parse_issue(self):
        # 20 — surfaced as numeric_parse_failed, never silently zeroed. A client
        # null marker like "N/A" is real content, so it fails here too rather
        # than being silently voided.
        df, report = self._typed(["20.00%", "N/A", "twenty percent"])
        self.assertEqual(report["fields"]["protected_equity_percentage"]["parse_failures"], 2)

        issues = ta._IssueLog()
        ta._record_parse_issues(report, {}, [], issues)
        numeric = [i for i in issues.rows if i["issue_type"] == ta.IT_NUMERIC_PARSE]
        self.assertEqual(len(numeric), 1)
        self.assertEqual(numeric[0]["severity"], "error")

    def test_percentage_text_never_reaches_the_boolean_parser(self):
        # 23 — the reported blocker. The percentage column is typed as a
        # percentage; nothing sends "20.00%" through a Y/N parser.
        df = pd.DataFrame({"protected_equity_percentage": ["0.00%", "20.00%", "50.00%"]})
        report = ct.apply_types(
            df, {"protected_equity_percentage": {"format": "percentage"}})
        issues = ta._IssueLog()
        ta._record_parse_issues(report, {}, [], issues)
        self.assertEqual([i for i in issues.rows
                          if i["issue_type"] == ta.IT_BOOLEAN_PARSE], [])


# --------------------------------------------------------------------------- #
# Protected equity — derived flag
# --------------------------------------------------------------------------- #
class TestProtectedEquityFlagDerivation(unittest.TestCase):

    def _derived(self, percentages):
        df = pd.DataFrame({"protected_equity_percentage": list(percentages)})
        results = cd.apply_derivations(df)
        return df, results["protected_equity_flag"]

    def test_zero_percent_gives_flag_N_and_keeps_the_number(self):
        # 16
        df, res = self._derived([0.0])
        self.assertEqual(df["protected_equity_flag"].tolist(), ["N"])
        self.assertEqual(df["protected_equity_percentage"].tolist(), [0.0])
        self.assertTrue(res["applied"])

    def test_positive_percent_gives_flag_Y_and_keeps_the_number(self):
        # 17 / 18
        df, _ = self._derived([20.0, 50.0, 30.0])
        self.assertEqual(df["protected_equity_flag"].tolist(), ["Y", "Y", "Y"])
        self.assertEqual(df["protected_equity_percentage"].tolist(), [20.0, 50.0, 30.0])

    def test_blank_percentage_gives_null_flag(self):
        # 19 — absent stays absent in BOTH fields; never defaulted to N.
        df, _ = self._derived([None])
        self.assertIsNone(df["protected_equity_flag"].tolist()[0])

    def test_full_distribution(self):
        df, res = self._derived([0.0, 20.0, 50.0, None, 30.0])
        self.assertEqual(df["protected_equity_flag"].tolist(),
                         ["N", "Y", "Y", None, "Y"])
        self.assertEqual(res["value_counts"], {"Y": 3, "N": 1, "null": 1})

    def test_flag_only_ever_holds_the_canonical_boolean_or_null(self):
        df, _ = self._derived([0.0, 20.0, None, 100.0])
        self.assertTrue(set(df["protected_equity_flag"].dropna()) <= {"Y", "N"})

    def test_derivation_declares_its_rule_and_parent_field(self):
        # 22 (derivation half) — the flag is attributable, never anonymous.
        _, res = self._derived([20.0])
        self.assertEqual(res["derived_from"], "protected_equity_percentage")
        self.assertEqual(res["rule"], "positive_number_to_flag")

    def test_source_percentage_is_never_discarded_by_the_derivation(self):
        df, _ = self._derived([0.0, 20.0, 50.0])
        self.assertEqual(df["protected_equity_percentage"].tolist(), [0.0, 20.0, 50.0])

    def test_uninterpretable_source_value_is_surfaced_not_defaulted(self):
        df = pd.DataFrame({"protected_equity_percentage": [20.0, "n/k"]})
        issues = ta._IssueLog()
        results = ta._apply_canonical_derivations(df, issues)
        res = results["protected_equity_flag"]
        self.assertEqual(res["failure_count"], 1)
        self.assertIsNone(df["protected_equity_flag"].tolist()[1])
        self.assertEqual([i["issue_type"] for i in issues.rows],
                         [ta.IT_DERIVATION_FAILED])

    def test_derivation_is_skipped_when_the_source_field_is_absent(self):
        df = pd.DataFrame({"loan_identifier": ["L1"]})
        results = cd.apply_derivations(df)
        self.assertFalse(results["protected_equity_flag"]["applied"])
        self.assertNotIn("protected_equity_flag", df.columns)

    def test_null_never_becomes_a_flag_value(self):
        self.assertIsNone(cd.RULES["positive_number_to_flag"](None))
        self.assertIsNone(cd.RULES["positive_number_to_flag"](""))
        self.assertIsNone(cd.RULES["positive_number_to_flag"](float("nan")))


# --------------------------------------------------------------------------- #
# Target-contract wiring: percentage is sourced, flag is derived
# --------------------------------------------------------------------------- #
class TestProtectedEquityContractWiring(unittest.TestCase):
    """The MI target contract must not point both fields at ``Protected Equity``."""

    def setUp(self):
        from engine.onboarding_agent import target_coverage as tcov
        self.tcov = tcov
        _cid, _src, fields = tcov.load_mi_target_contract()
        self.by_field = {f["target_field"]: f for f in fields}

    def test_percentage_is_a_first_class_mi_field(self):
        self.assertIn("protected_equity_percentage", self.by_field)
        pct = self.by_field["protected_equity_percentage"]
        self.assertFalse(pct["derived"])
        self.assertIn("protected equity", [s.lower() for s in pct["synonyms"]])

    def test_flag_is_declared_derived_from_the_percentage(self):
        flag = self.by_field["protected_equity_flag"]
        self.assertTrue(flag["derived"])
        self.assertEqual(flag["derived_from"], "protected_equity_percentage")

    def test_flag_does_not_claim_the_bare_protected_equity_name(self):
        # The root cause: "Protected Equity" as the FLAG's business name/synonym
        # made the flag match the percentage source column by name.
        flag = self.by_field["protected_equity_flag"]
        self.assertNotIn("protected equity", [s.lower() for s in flag["synonyms"]])

    def _candidates(self, column="Protected Equity"):
        return [{"source_file": "AcquiredLoanTape.csv", "source_sheet": "",
                 "source_column": column, "confidence": 0.85,
                 "basis": "name_synonym_match"}]

    _PARENTS = {"protected_equity_percentage": {
        ("AcquiredLoanTape.csv", "", "Protected Equity")}}

    def test_derived_field_does_not_take_its_parents_source_column(self):
        # 21 — even if the flag DID match the same column, the guard stops it
        # being selected.
        flag = dict(self.by_field["protected_equity_flag"])
        kept, note, demoted = self.tcov._apply_derived_source_guard(
            flag, self._candidates(), self._PARENTS)
        self.assertEqual(kept, [])
        self.assertIn("protected_equity_percentage", note)
        # …and the evidence is still recorded, not deleted.
        self.assertIn("Protected Equity", demoted)

    def test_derived_field_keeps_a_genuinely_different_source_column(self):
        # Backwards compatibility: a source that really does carry both fields
        # still maps the derived one directly.
        flag = dict(self.by_field["protected_equity_flag"])
        cands = self._candidates("Protected Equity Flag")
        kept, note, demoted = self.tcov._apply_derived_source_guard(
            flag, cands, self._PARENTS)
        self.assertEqual(kept, cands)
        self.assertEqual((note, demoted), ("", ""))

    def test_a_shared_column_is_demoted_below_a_distinct_one(self):
        flag = dict(self.by_field["protected_equity_flag"])
        distinct = self._candidates("Protected Equity Flag")[0]
        shared = self._candidates()[0]
        kept, note, _ = self.tcov._apply_derived_source_guard(
            flag, [shared, distinct], self._PARENTS)
        self.assertEqual([c["source_column"] for c in kept],
                         ["Protected Equity Flag", "Protected Equity"])
        self.assertIn("demoted", note)

    def test_guard_is_inert_for_fields_with_no_declared_parent(self):
        plain = {"target_field": "current_interest_rate", "derived_from": ""}
        cands = self._candidates("Loan Interest Rate")
        kept, note, demoted = self.tcov._apply_derived_source_guard(plain, cands, {})
        self.assertEqual(kept, cands)
        self.assertEqual((note, demoted), ("", ""))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
