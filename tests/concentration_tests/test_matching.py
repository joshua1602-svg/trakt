"""Extraction and library mapping: deterministic, total, and honest about what
it could not resolve. The model seam may only propose; it cannot decide."""

from __future__ import annotations

from mi_agent.concentration_tests.matching import (
    build_proposals,
    extract_from_rows,
    extract_from_text,
)
from mi_agent.concentration_tests.models import (
    ConcernCode,
    MATCH_AMBIGUOUS,
    MATCH_COMPOSED,
    MATCH_MATCHED,
    MATCH_UNSUPPORTED,
    PROPOSAL_PENDING_APPROVAL,
    PROPOSAL_PENDING_CONFIRMATION,
    PROPOSAL_UNSUPPORTED,
)


def propose(lib, text="", rows=None, **kw):
    return build_proposals(lib=lib, client_id="client_a",
                           source_reference="covenant.txt", text=text,
                           rows=rows, **kw)


class TestExtraction:
    def test_threshold_and_operator_extraction(self):
        [e] = extract_from_text(
            "1.1 Loans located in Wales must not exceed 10% of the Portfolio.")
        assert e.operator == "max"
        assert e.threshold_percent == 10.0
        assert e.locator == "clause 1.1"

    def test_minimum_operator(self):
        [e] = extract_from_text("The Net WAC must be at least 3.75%.")
        assert e.operator == "min"
        assert e.threshold_percent == 3.75

    def test_amount_units_parse_millions_and_thousands(self):
        [a] = extract_from_text("Loans above £1.5 million must not exceed 10%.")
        assert a.threshold_amount == 1_500_000
        [b] = extract_from_text("Loans below £100k must not exceed 10%.")
        assert b.threshold_amount == 100_000

    def test_prose_without_numbers_is_left_alone(self):
        assert extract_from_text(
            "The Borrower shall provide monthly reporting.") == []

    def test_structured_rows(self):
        rows = extract_from_rows([
            {"Test": "East Anglia exposure", "Threshold": "25%",
             "Direction": "max",
             "Definition": "Balance in East Anglia over portfolio balance"},
        ])
        assert rows[0].threshold_percent == 25.0
        assert rows[0].operator == "max"
        assert rows[0].locator == "row 1"

    def test_malformed_row_survives_as_low_confidence(self):
        rows = extract_from_rows(["not even a dict"])
        assert rows[0].extraction_confidence <= 0.2


class TestMapping:
    def test_direct_match(self, lib):
        [p] = propose(lib, "1.1 The aggregate Current Balance of Loans secured "
                           "on Properties located in East Anglia must not "
                           "exceed 25% of the Portfolio.")
        assert p.match_outcome == MATCH_MATCHED
        assert p.proposed_metric_id == "geo_region_share"
        assert p.parameters == {"regions": ["East Anglia"]}
        assert p.extracted_threshold == 25.0
        assert p.status == PROPOSAL_PENDING_APPROVAL
        assert p.evidence.source_text.startswith("1.1")

    def test_alias_match(self, lib):
        [p] = propose(lib, "The gross weighted average coupon must be at "
                           "least 4.0%.")
        assert p.proposed_metric_id == "rate_gross_wac"
        assert p.match_outcome == MATCH_MATCHED

    def test_combined_region_is_composed(self, lib):
        [p] = propose(lib, "Loans located in London plus the South East must "
                           "not exceed 50% of the Portfolio.")
        assert p.match_outcome == MATCH_COMPOSED
        assert p.parameters["regions"] == ["London", "South East"]

    def test_ambiguous_valuation_basis_raises_a_concern(self, lib):
        [p] = propose(lib, "Loans with a valuation above £1,500,000 must not "
                           "exceed 10% of the Portfolio.")
        assert p.match_outcome == MATCH_AMBIGUOUS
        assert ConcernCode.BALANCE_BASIS_UNDEFINED in p.concern_codes
        assert p.status == PROPOSAL_PENDING_CONFIRMATION

    def test_ambiguous_net_wac_asks_the_right_question(self, lib):
        [p] = propose(lib, "The Net WAC of the Portfolio must be at least "
                           "3.75%.")
        assert p.match_outcome == MATCH_AMBIGUOUS
        assert ConcernCode.NET_WAC_DEFINITION_UNCERTAIN in p.concern_codes
        assert any("servicing" in q for q in p.confirmation_questions)

    def test_joint_borrowers_ask_for_the_basis(self, lib):
        [p] = propose(lib, "The aggregate Current Balance attributable to "
                           "joint Borrowers must not exceed 90% of the "
                           "Portfolio.")
        assert ConcernCode.JOINT_DEFINITION_UNCERTAIN in p.concern_codes

    def test_hpi_maps_to_the_governed_interface_pending_configuration(self, lib):
        [p] = propose(lib, "The current HPI index divided by the reference "
                           "HPI index must be at least 90%.")
        assert p.proposed_metric_id == "index_hpi_ratio"
        assert ConcernCode.INDEX_SERIES_MISSING in p.concern_codes
        assert ConcernCode.INDEX_BASE_DATE_MISSING in p.concern_codes

    def test_duplicate_tests_are_flagged(self, lib):
        text = ("1.1 Loans located in Wales must not exceed 10% of the "
                "Portfolio.\n"
                "9.9 Loans located in Wales must not exceed 10% of the "
                "Portfolio.")
        props = propose(lib, text)
        assert ConcernCode.DUPLICATED_TEST in props[1].concern_codes
        assert ConcernCode.DUPLICATED_TEST not in props[0].concern_codes

    def test_conflicting_thresholds_flag_both(self, lib):
        text = ("1.1 Loans located in Wales must not exceed 10% of the "
                "Portfolio.\n"
                "9.9 Loans located in Wales must not exceed 12% of the "
                "Portfolio.")
        props = propose(lib, text)
        for p in props:
            assert ConcernCode.CONFLICTING_THRESHOLDS in p.concern_codes

    def test_unsupported_test_raises_a_structured_request(self, lib):
        [p] = propose(lib, "The portfolio duration must not exceed 7 years.")
        assert p.match_outcome == MATCH_UNSUPPORTED
        assert p.status == PROPOSAL_UNSUPPORTED
        assert p.implementation_request["source_text"]
        assert ConcernCode.UNSUPPORTED_METRIC in p.concern_codes

    def test_catalogued_but_unimplemented_metric_is_unsupported(self, lib):
        [p] = propose(lib, "The cumulative loss ratio must not exceed 1%.")
        assert p.proposed_metric_id == "perf_cumulative_loss_share"
        assert p.match_outcome == MATCH_UNSUPPORTED

    def test_percentage_vs_amount_units(self, lib):
        [p] = propose(lib, "The average initial principal balance must not "
                           "exceed £300,000.")
        assert p.proposed_metric_id == "balance_average"
        assert p.extracted_threshold == 300_000
        assert p.extracted_unit == "currency"


class TestModelSeam:
    def test_assist_candidates_go_through_the_same_mapping(self, lib):
        class Assist:
            def propose(self, text):
                return [{"source_text": "Loans located in Wales must not "
                                        "exceed 10% of the Portfolio.",
                         "name": "Wales limit", "threshold_percent": 10.0,
                         "operator": "max", "confidence": 0.8}]
        props = propose(lib, text="The parties agree as follows.",
                        assist=Assist())
        assert len(props) == 1
        assert props[0].proposed_metric_id == "geo_region_share"
        assert props[0].evidence.locator == "model_proposed"

    def test_assist_cannot_smuggle_an_invalid_operator(self, lib):
        class Assist:
            def propose(self, text):
                return [{"source_text": "Wales exposure capped somehow",
                         "operator": "EXECUTE(rm -rf)", "confidence": 2.0}]
        props = propose(lib, text="Nothing extractable here at all.",
                        assist=Assist())
        assert props[0].extracted_operator in ("", "max", "min")
        assert 0.0 <= props[0].extraction_confidence <= 1.0

    def test_missing_evidence_is_flagged(self, lib):
        class Assist:
            def propose(self, text):
                return [{"source_text": " ", "name": "ghost"}]
        props = propose(lib, text="Prose without numbers.", assist=Assist())
        assert props == []  # blank source text is not even a candidate
