"""A real warehouse-facility Schedule 8, end to end through extraction.

This is the document shape the OCC actually receives: negotiated figures still
in square brackets, a limits table whose comparison sits in the lead-in
sentence, and a defined denominator with a contractual floor. Every one of its
seventeen limits must come back with a threshold and an operator — a schedule
that reads as "—" everywhere is the failure this suite exists to catch.

The one test left open (Net WAC) is open for a real reason: the schedule
defines Portfolio Net WAC as the weighted average rate minus the "Series Fixed
Rate", a figure that lives elsewhere in the facility. That is a question for
the client, not something to infer.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from mi_agent.concentration_tests.matching import (
    build_proposals,
    extract_definitions,
    normalise_source_text,
)
from mi_agent.concentration_tests.models import (
    ConcernCode,
    MATCH_AMBIGUOUS,
    MATCH_UNSUPPORTED,
)

SCHEDULE = (Path(__file__).parent / "fixtures"
            / "warehouse_facility_schedule_8.txt").read_text(encoding="utf-8")

#: Every limit the schedule states: ten region rows, two property-value tests,
#: two principal-balance tests, one age test, Net WAC, and joint borrowers.
EXPECTED_LIMIT_COUNT = 17


@pytest.fixture(scope="module")
def proposals(lib):
    return build_proposals(lib=lib, client_id="client_a",
                           source_reference="Warehouse Facility Schedule 8",
                           text=SCHEDULE)


class TestBracketedFigures:
    """Bracketed numbers are a drafting convention, not arithmetic."""

    @pytest.mark.parametrize("raw, expected", [
        ("[50]%", "50%"),
        ("£[1,500,000]", "£1,500,000"),
        ("[3.75]% per annum", "3.75% per annum"),
        ("[0]%", "0%"),
    ])
    def test_placeholders_are_unwrapped(self, raw, expected):
        assert normalise_source_text(raw) == expected

    @pytest.mark.parametrize("raw", ["[Reserved]", "[sic]", "[Schedule 8]"])
    def test_non_numeric_brackets_are_left_alone(self, raw):
        assert normalise_source_text(raw) == raw


class TestDefinedDenominator:
    def test_the_floor_is_read_from_the_definition(self):
        definitions = extract_definitions(SCHEDULE)
        [defined] = definitions.values()
        assert defined.term == "Concentration Limit Denominator"
        assert defined.floor_amount == 33_000_000.0

    def test_a_definition_is_not_proposed_as_a_test(self, proposals):
        assert not [p for p in proposals
                    if "means an amount equal to" in p.evidence.source_text]

    def test_the_floor_reaches_every_clause_that_cites_the_term(self, proposals):
        citing = [p for p in proposals
                  if "Concentration Limit Denominator" in p.evidence.source_text]
        # Everything except the average-balance test and Net WAC, neither of
        # which is expressed as a share of the portfolio.
        assert len(citing) == EXPECTED_LIMIT_COUNT - 2
        assert all(p.parameters.get("denominator_floor") == 33_000_000.0
                   for p in citing)

    def test_a_clause_that_does_not_cite_it_is_left_alone(self, proposals):
        average = _one(proposals, "balance_average")
        assert "denominator_floor" not in average.parameters


class TestLimitsTable:
    def test_every_region_row_carries_its_own_threshold(self, proposals):
        rows = {tuple(p.parameters["regions"]): p.extracted_threshold
                for p in proposals if p.proposed_metric_id == "geo_region_share"}
        assert rows == {
            ("London", "South East"): 50.0,
            ("North East",): 10.0,
            ("North West",): 10.0,
            ("Yorkshire And The Humber",): 15.0,
            ("East Midlands",): 15.0,
            ("West Midlands",): 15.0,
            ("East Of England",): 25.0,
            ("South West",): 20.0,
            ("Scotland",): 10.0,
            ("Wales",): 10.0,
        }

    def test_rows_inherit_the_comparison_from_the_lead_in(self, proposals):
        regional = [p for p in proposals
                    if p.proposed_metric_id == "geo_region_share"]
        assert {p.extracted_operator for p in regional} == {"max"}

    def test_the_row_is_never_shown_without_the_sentence_governing_it(
            self, proposals):
        scotland = next(p for p in proposals
                        if p.parameters.get("regions") == ["Scotland"])
        assert "does not exceed the limits set out below" \
            in scotland.evidence.source_text
        assert "UKM Scotland 10%" in scotland.evidence.source_text

    def test_the_lead_in_does_not_also_become_a_thresholdless_proposal(
            self, proposals):
        assert not [p for p in proposals
                    if p.extracted_threshold is None]


class TestBorrowerClauses:
    def test_two_borrowers_maps_to_the_joint_metric_by_count(self, proposals):
        joint = _one(proposals, "borrower_joint_share")
        assert joint.parameters["joint_basis"] == "borrower_count"
        assert joint.extracted_threshold == 90.0
        assert joint.match_outcome != MATCH_UNSUPPORTED
        assert not joint.concern_codes

    def test_aggregate_borrower_lending_has_a_governed_metric(self, proposals):
        agg = _one(proposals, "borrower_aggregate_balance_share")
        assert agg.parameters["amount"] == 1_000_000.0
        assert agg.parameters["aggregate_basis"] == "original"
        assert agg.extracted_threshold == 10.0

    def test_the_age_test_reads_the_youngest_borrower(self, proposals):
        age = _one(proposals, "borrower_age_share")
        assert age.parameters == {
            "age": 55.0, "comparison": "below", "age_basis": "youngest",
            "denominator_floor": 33_000_000.0, "denominator": "current_balance"}
        assert age.extracted_threshold == 0.0


class TestWholeSchedule:
    def test_every_stated_limit_becomes_exactly_one_proposal(self, proposals):
        assert len(proposals) == EXPECTED_LIMIT_COUNT

    def test_nothing_is_unsupported(self, proposals):
        assert not [p for p in proposals
                    if p.match_outcome == MATCH_UNSUPPORTED]

    def test_every_proposal_has_a_threshold_and_a_comparison(self, proposals):
        assert all(p.extracted_threshold is not None for p in proposals)
        assert all(p.extracted_operator in ("max", "min") for p in proposals)

    def test_only_net_wac_still_needs_the_client(self, proposals):
        unresolved = [p for p in proposals if p.concern_codes]
        assert [p.proposed_metric_id for p in unresolved] == ["rate_net_wac"]
        assert unresolved[0].concern_codes == [
            ConcernCode.NET_WAC_DEFINITION_UNCERTAIN]
        assert unresolved[0].match_outcome == MATCH_AMBIGUOUS
        # The question asked names the missing figure, not the whole clause.
        assert unresolved[0].confirmation_questions

    def test_the_schedule_resolves_overwhelmingly_without_a_question(
            self, proposals):
        outcomes = Counter(p.match_outcome for p in proposals)
        assert outcomes[MATCH_AMBIGUOUS] == 1


def _one(proposals, metric_id):
    [found] = [p for p in proposals if p.proposed_metric_id == metric_id]
    return found
