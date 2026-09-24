"""August's answers, applied to July's files.

ERE's extracts are named with the date they were cut:

    PropertyExtract - Omni 2026_09_01.xlsx     (the August delivery)
    PropertyExtract - Omni 2026_08_01.xlsx     (July, in the backfill)

Every answer an operator gave about a FILE was keyed on that exact name — 16
of the 19 set-asides, every "this file is authoritative", the file each
confirmed source is read from. A backfill of historic tapes matched none of
them and would have asked the whole onboarding again, month by month.

A file is now recognised by its FAMILY, the name without its date. The exact
name still wins; the family is the fallback, and a family two files share is
never guessed between.
"""

from __future__ import annotations

import pytest

from engine.onboarding_agent import target_coverage as tcov
from engine.onboarding_agent.file_identity import (file_family, resolve,
                                                    same_source)

AUG_PROP = "PropertyExtract - Omni 2026_09_01.xlsx"
JUL_PROP = "PropertyExtract - Omni 2026_08_01.xlsx"
JUL_LOAN = "LoanExtract One - OMNI 2026_08_01.xlsx"
JUL_PANDI = "Principal And Interest - OMNI 2026_08_01.xlsx"


class TestTheFamily:

    @pytest.mark.parametrize("a,b", [
        (AUG_PROP, JUL_PROP),
        ("Tape 2026-09-01.csv", "Tape 2025-12-01.csv"),
        ("tape_202609.csv", "tape_202512.csv"),
        ("Book Sep 2026.xlsx", "Book March 2025.xlsx"),
        ("Loans.xlsx", "loans.xlsx"),
    ])
    def test_same_source_across_months(self, a, b):
        assert same_source(a, b)

    @pytest.mark.parametrize("a,b", [
        (AUG_PROP, JUL_LOAN),
        ("Tape 2026-09-01.csv", "Tape 2026-09-01.xlsx"),
    ])
    def test_different_sources_stay_different(self, a, b):
        assert not same_source(a, b)

    def test_a_number_that_is_not_a_date_is_kept(self):
        assert file_family("Pool 2026 report.xlsx") == "pool 2026 report.xlsx"


class TestResolvingAnotherMonthsName:

    def test_the_exact_name_wins(self):
        assert resolve(AUG_PROP, [AUG_PROP, JUL_PROP]) == AUG_PROP

    def test_the_family_finds_this_month_s_file(self):
        assert resolve(AUG_PROP, [JUL_LOAN, JUL_PROP, JUL_PANDI]) == JUL_PROP

    def test_two_of_a_family_are_not_guessed_between(self):
        assert resolve(AUG_PROP, [JUL_PROP,
                                  "PropertyExtract - Omni 2026_07_01.xlsx"]) == ""

    def test_nothing_of_the_family_resolves_to_nothing(self):
        assert resolve(AUG_PROP, [JUL_LOAN]) == ""


class TestAugustsSetAsidesApplyToJuly:

    def test_a_set_aside_holds_in_another_month(self):
        rows = [{"source_file": JUL_PANDI, "source_column": "Current Interest Rate"},
                {"source_file": JUL_LOAN, "source_column": "Current Interest Rate"}]
        kept = tcov.without_set_asides(
            rows, [("Principal And Interest - OMNI 2026_09_01.xlsx",
                    "Current Interest Rate")])
        assert [r["source_file"] for r in kept] == [JUL_LOAN]


class TestTheTapeFindsJulysFile:

    def test_the_coverage_selection_resolves_to_this_month(self):
        from engine.onboarding_agent import central_tape_builder as ctb
        inventory = {JUL_PROP: {"file_path": "/x/jul_prop.xlsx"},
                     JUL_LOAN: {"file_path": "/x/jul_loan.xlsx"}}
        forced, _c, _d = ctb._coverage_selections(
            [{"target_field": "current_valuation_amount",
              "coverage_status": "source_mapped",
              "selected_source_file": AUG_PROP,
              "selected_source_column": "Latest Valuation"}],
            inventory, {"current_valuation_amount": {}})
        assert forced["current_valuation_amount"].file_name == JUL_PROP
        assert forced["current_valuation_amount"].file_path == "/x/jul_prop.xlsx"

    def test_an_authoritative_file_holds_in_another_month(self):
        from engine.onboarding_agent import central_tape_builder as ctb
        srcs = [ctb._Source(file_name=JUL_LOAN, file_path="", column="Val",
                            sheet="", method="", confidence=1.0,
                            classification=""),
                ctb._Source(file_name=JUL_PROP, file_path="", column="Val",
                            sheet="", method="", confidence=1.0,
                            classification="")]
        ordered = ctb._order_sources(
            "current_valuation_amount", srcs,
            {"current_valuation_amount": {"primary_source_file": AUG_PROP}},
            {}, {})
        assert ordered[0].file_name == JUL_PROP
