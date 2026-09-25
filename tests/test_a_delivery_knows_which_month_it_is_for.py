"""The run said August. The operator was asked for the reporting date.

Two questions reached the operator on ERE's August delivery that the delivery
had already answered:

* ``reporting_date`` — "missing", blocking. The Operations Control Centre
  recorded the delivery as ``2026-08`` and handed that to onboarding as its
  context. Two things then threw it away: ``normalize_to_iso`` reads only whole
  dates, so ``2026-08`` was read as nothing; and the period fill sat inside the
  product-profile derivations, behind an "is equity release detected?" gate
  that has nothing to do with what month it is.

* ``data_cut_off_date`` — "which of these is authoritative?". Every one of
  ERE's extracts carries ``Month Run = August``. Three columns claimed the
  field, so the operator was asked to choose between three spellings of the
  month their own delivery was for.

The rule now: the period is read from the run, and a period column that only
restates it on every row is not a question. A column that names ANY other
month keeps its question — a delivery disagreeing with itself is exactly what
an operator must see.
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.onboarding_agent import target_coverage as tcov

LOAN = "LoanExtract One - OMNI 2026_09_01.xlsx"
PROP = "PropertyExtract - Omni 2026_09_01.xlsx"


def _missing(field="reporting_date"):
    return {"target_field": field, "coverage_status": tcov.MISSING_REQUIRED,
            "blocking": True, "requires_user_decision": True,
            "selected_value": ""}


class TestThePeriodIsReadHoweverTheRunSpelledIt:

    @pytest.mark.parametrize("given,expected", [
        ("2026-08", "2026-08-31"),          # what the OCC records
        ("2026-08-31", "2026-08-31"),
        ("August 2026", "2026-08-31"),
        ("2026_08", "2026-08-31"),
        ("", ""),
        ("not a period", ""),
    ])
    def test_context_period(self, given, expected):
        assert tcov._context_period({"reporting_date": given}) == expected

    def test_the_cut_off_is_read_when_the_reporting_date_is_absent(self):
        assert tcov._context_period(
            {"data_cut_off_date": "2026-08"}) == "2026-08-31"


class TestThePeriodIsNotAProductQuestion:

    def test_a_missing_reporting_date_is_filled_from_the_context(self):
        rows = [_missing()]
        tcov.apply_period_inference(rows, run_id="run",
                                    context={"reporting_date": "2026-08"})
        assert rows[0]["coverage_status"] == tcov.DEFAULTED_VALUE
        assert rows[0]["selected_value"] == "2026-08-31"
        assert rows[0]["blocking"] is False

    def test_it_needs_no_product_profile(self):
        """The coverage run calls it on its own, whatever the product."""
        import inspect
        src = inspect.getsource(tcov.run_target_first_coverage)
        assert "apply_period_inference(" in src

    def test_a_real_source_column_still_wins(self):
        rows = [{"target_field": "reporting_date",
                 "coverage_status": tcov.SOURCE_MAPPED,
                 "selected_source_column": "Report Date"}]
        tcov.apply_period_inference(rows, context={"reporting_date": "2026-08"})
        assert rows[0]["coverage_status"] == tcov.SOURCE_MAPPED

    def test_no_period_fills_nothing(self):
        rows = [_missing()]
        tcov.apply_period_inference(rows, run_id="run", context={})
        assert rows[0]["coverage_status"] == tcov.MISSING_REQUIRED


def _tables(loan_month="August", prop_month="August"):
    n = 5
    return [(LOAN, "", pd.DataFrame({"Month Run": [loan_month] * n})),
            (PROP, "", pd.DataFrame({"Month Run": [prop_month] * n}))]


def _claimed(field="data_cut_off_date"):
    return {"target_field": field,
            "coverage_status": tcov.SOURCE_MAPPED_ALT,
            "selected_source_file": LOAN, "selected_source_sheet": "",
            "selected_source_column": "Month Run",
            "alternative_source_candidates": f"{PROP}::::Month Run (0.95)",
            "requires_user_decision": True, "blocking": False,
            "decision_reason": "multiple candidates",
            "operator_question": "Which source column is the authoritative "
                                 f"source for '{field}'?"}


class TestAMonthColumnThatAgreesIsNotAQuestion:

    def test_two_files_saying_august_settle_to_the_period_end(self):
        rows = [_claimed()]
        left, settled = tcov.settle_period_fields(rows, [], _tables(),
                                                  "2026-08-31")
        assert rows[0]["coverage_status"] == tcov.DEFAULTED_VALUE
        assert rows[0]["selected_value"] == "2026-08-31"
        assert rows[0]["requires_user_decision"] is False
        assert rows[0]["operator_question"] == ""
        assert settled and settled[0]["target_field"] == "data_cut_off_date"

    def test_the_reason_names_the_columns_it_read(self):
        rows = [_claimed()]
        tcov.settle_period_fields(rows, [], _tables(), "2026-08-31")
        assert "Month Run" in rows[0]["default_reason"]
        assert LOAN in rows[0]["default_reason"]

    def test_the_regulatory_name_for_the_same_date_is_settled_too(self):
        rows = [_claimed("cut_off_date")]
        tcov.settle_period_fields(rows, [], _tables(), "2026-08-31")
        assert rows[0]["coverage_status"] == tcov.DEFAULTED_VALUE

    def test_an_unreadable_value_question_is_withdrawn_when_it_agrees(self):
        rows = [dict(_claimed(), alternative_source_candidates="",
                     requires_user_decision=False,
                     coverage_status=tcov.SOURCE_MAPPED)]
        asked = [{"canonical_field": "data_cut_off_date",
                  "target_field": "data_cut_off_date",
                  "source_file": LOAN, "source_column": "Month Run",
                  "source_value": "August"}]
        left, _ = tcov.settle_period_fields(rows, asked, _tables(),
                                            "2026-08-31")
        assert left == []
        assert rows[0]["coverage_status"] == tcov.DEFAULTED_VALUE

    def test_a_whole_date_every_column_agrees_on_is_kept(self):
        n = 3
        tables = [(LOAN, "", pd.DataFrame({"Month Run": ["2026-08-28"] * n})),
                  (PROP, "", pd.DataFrame({"Month Run": ["28/08/2026"] * n}))]
        rows = [_claimed()]
        tcov.settle_period_fields(rows, [], tables, "2026-08-31")
        assert rows[0]["selected_value"] == "2026-08-28"


class TestAMonthColumnThatDisagreesIsStillAsked:

    @pytest.mark.parametrize("loan,prop", [("July", "August"),
                                           ("August", "September")])
    def test_any_other_month_keeps_the_question(self, loan, prop):
        rows = [_claimed()]
        left, settled = tcov.settle_period_fields(
            rows, [], _tables(loan, prop), "2026-08-31")
        assert rows[0]["coverage_status"] == tcov.SOURCE_MAPPED_ALT
        assert rows[0]["requires_user_decision"] is True
        assert settled == []

    def test_one_stray_row_keeps_the_question(self):
        """Never by majority."""
        tables = [(LOAN, "", pd.DataFrame(
            {"Month Run": ["August"] * 9 + ["July"]})),
            (PROP, "", pd.DataFrame({"Month Run": ["August"] * 10}))]
        rows = [_claimed()]
        tcov.settle_period_fields(rows, [], tables, "2026-08-31")
        assert rows[0]["requires_user_decision"] is True

    def test_a_value_it_cannot_read_keeps_its_question(self):
        rows = [dict(_claimed(), alternative_source_candidates="")]
        asked = [{"canonical_field": "data_cut_off_date",
                  "source_value": "TBC"}]
        tables = [(LOAN, "", pd.DataFrame({"Month Run": ["TBC"] * 3}))]
        left, _ = tcov.settle_period_fields(rows, asked, tables, "2026-08-31")
        assert left == asked

    def test_without_a_period_nothing_is_settled(self):
        rows = [_claimed()]
        left, settled = tcov.settle_period_fields(rows, [], _tables(), "")
        assert settled == [] and rows[0]["requires_user_decision"] is True

    def test_a_field_that_is_not_a_period_is_never_touched(self):
        rows = [dict(_claimed(), target_field="current_valuation_amount")]
        tcov.settle_period_fields(rows, [], _tables(), "2026-08-31")
        assert rows[0]["coverage_status"] == tcov.SOURCE_MAPPED_ALT


class TestAQuestionWithNoCoverageRowIsSettledToo:
    """ERE's older layouts: `Month Run` is a confirmed mapping to the cut-off,
    the target contract has no row for it, and the value check asked "Is
    'October' in 'Month Run' a placeholder?" on a delivery for October."""

    def _asked(self, value="October"):
        return [{"canonical_field": "data_cut_off_date",
                 "target_field": "data_cut_off_date",
                 "source_file": LOAN, "source_column": "Month Run",
                 "source_value": value}]

    def test_the_delivery_s_own_month_is_not_a_question(self):
        tables = [(LOAN, "", pd.DataFrame({"Month Run": ["October"] * 4}))]
        left, settled = tcov.settle_period_fields([], self._asked(), tables,
                                                  "2025-10-31")
        assert left == []
        assert settled and settled[0]["value"] == "2025-10-31"

    def test_another_month_is_still_asked(self):
        tables = [(LOAN, "", pd.DataFrame({"Month Run": ["October"] * 3
                                           + ["September"]}))]
        asked = self._asked()
        left, settled = tcov.settle_period_fields([], asked, tables,
                                                  "2025-10-31")
        assert left == asked and settled == []

    def test_a_field_that_is_not_a_period_is_left_alone(self):
        asked = [dict(self._asked()[0], canonical_field="current_valuation_date",
                      target_field="current_valuation_date")]
        tables = [(LOAN, "", pd.DataFrame({"Month Run": ["October"] * 2}))]
        left, _ = tcov.settle_period_fields([], asked, tables, "2025-10-31")
        assert left == asked
