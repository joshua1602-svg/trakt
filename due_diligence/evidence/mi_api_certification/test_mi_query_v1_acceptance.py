"""Offline tests for the MI Query V1 acceptance harness.

WHY THESE EXIST. The local demonstration book cannot serve the dashboard GET
endpoints this harness reads truth from — its platform index is empty — so the
numeric paths would otherwise execute for the FIRST time against production,
and a harness bug would arrive dressed as a service defect. Every truth key,
every availability rule and every check is therefore driven here from synthetic
endpoint payloads whose right answer is known by construction.

These tests are about the HARNESS. They assert nothing about the live book.
"""
from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from . import mi_query_v1_acceptance as acc
from . import mi_query_v1_truth as truth_mod
from .mi_query_v1_truth import UNAVAILABLE, UNRESOLVED


# --------------------------------------------------------------------------- #
# A synthetic production surface
# --------------------------------------------------------------------------- #
SNAPSHOT = {
    "ok": True,
    "portfolio": {"client_id": "ERE", "run_id": "2026-06-30",
                  "reporting_date": "2026-06-30"},
    "prior": {"run_id": "2026-05-31", "reporting_date": "2026-05-31"},
    "loan_count": 958,
    "current_outstanding_balance": 159097304.07,
    "kpis": [
        {"id": "balance", "label": "Current funded balance", "raw": 159097304.07,
         "format": "gbp", "available": True},
        {"id": "loans", "label": "Loans funded", "raw": 958, "format": "number",
         "available": True},
        {"id": "wa_current_ltv", "label": "Weighted avg current LTV", "raw": 41.7,
         "format": "pct", "available": True},
        {"id": "wa_rate", "label": "Weighted avg interest rate", "raw": 6.2,
         "format": "pct", "available": True},
    ],
    "stratifications": [
        {"key": "region", "label": "By region", "bars": [
            {"label": "London", "balance": 80000000.0, "count": 400, "sharePct": 50.3},
            {"label": "Scotland", "balance": 40000000.0, "count": 300, "sharePct": 25.1},
            {"label": "Wales", "balance": 39097304.07, "count": 258, "sharePct": 24.6},
        ]},
        {"key": "ltv", "label": "By LTV band", "bars": [
            {"label": "<30%", "balance": 59097304.07, "count": 358, "sharePct": 37.1},
            {"label": "30-50%", "balance": 100000000.0, "count": 600, "sharePct": 62.9},
        ]},
        {"key": "product", "label": "By product", "bars": [],
         "availability": "all_null",
         "reason": "Product is present in the tape but has no values."},
    ],
    "monthly_change": {"prior_run_id": "2026-05-31", "loan_count_change": 8,
                       "balance_change": 1250000.0, "balance_change_pct": 0.79,
                       "new_loans": 12, "exited_loans": 4, "loans_identifiable": True},
}

GEO = {"available": True, "basis": "collateral", "total": 159097304.07,
       "areaCount": 2,
       "areas": [{"name": "Edinburgh", "balance": 90000000.0, "count": 500},
                 {"name": "Bristol", "balance": 69097304.07, "count": 458}]}

PIPELINE = {"ok": True, "pipelineRowCount": 42, "pipelineAsOfDate": "2026-06-26",
            "stageBreakdown": [{"stage": "KFI", "caseCount": 20, "pipelineAmount": 5e6},
                               {"stage": "Application", "caseCount": 22,
                                "pipelineAmount": 6e6}]}

EVOLUTION = {
    "dataset": "funded", "reportingDates": ["2026-05-31", "2026-06-30"],
    "availableRunIds": ["2026-05-31", "2026-06-30"],
    "periods": [
        {"run_id": "2026-05-31", "reporting_date": "2026-05-31", "period": "2026-05",
         "metrics": {"funded_balance": 157847304.07, "loan_count": 950,
                     "wa_ltv": 0.415}},
        {"run_id": "2026-06-30", "reporting_date": "2026-06-30", "period": "2026-06",
         "metrics": {"funded_balance": 159097304.07, "loan_count": 958,
                     "wa_ltv": 0.417}},
    ],
    "breakdowns": {"region": [
        {"period": "2026-05", "key": "Scotland", "balance": 39500000.0},
        {"period": "2026-05", "key": "London", "balance": 79000000.0},
        {"period": "2026-05", "key": "Wales", "balance": 39347304.07},
        {"period": "2026-06", "key": "Scotland", "balance": 40000000.0},
        {"period": "2026-06", "key": "London", "balance": 80000000.0},
        {"period": "2026-06", "key": "Wales", "balance": 39097304.07},
    ]},
}

RISK = {"available": True, "summary": {"testsPassed": 5, "breaches": 1, "total": 6,
                                       "closestHeadroom": {"name": "Top 3 brokers",
                                                           "headroomPp": -31.5}},
        "tests": []}

BORROWING_BASE = {"available": False,
                  "reason": ("No funding facility is configured for this portfolio. "
                             "Borrowing-base monitoring begins when an approved "
                             "facility configuration is recorded through OCC "
                             "onboarding.")}

FORECAST = {"currentFundedBalance": 159097304.07, "runRate": 1250000.0}
COHORTS = {"available": True, "periods": [{"survivingLoanCount": 100},
                                          {"survivingLoanCount": 96},
                                          {"survivingLoanCount": 91}]}
SOURCES = {"available": True, "lenses": [{"id": "total"}, {"id": "direct_001"},
                                        {"id": "acquired_001"}], "source": "live"}
PIPELINE_SNAPSHOTS = {"sources": [{"client_id": "ERE"}], "source": "blob://"}
PIPELINE_EVOLUTION = {"availableExtractDates": ["2026-06-19", "2026-06-26"]}
PORTFOLIO_CONTEXT = {"available": True, "contexts": [{"id": "total"}, {"id": "direct"},
                                                     {"id": "acquired"}]}

SURFACE = {
    "/mi/snapshot": SNAPSHOT, "/mi/geo/exposure": GEO,
    "/mi/pipeline/snapshot": PIPELINE, "/mi/evolution/funded": EVOLUTION,
    "/mi/risk-limits": RISK, "/mi/borrowing-base": BORROWING_BASE,
    "/mi/forecast/extrapolation": FORECAST,
    "/mi/cohorts/progression": COHORTS, "/mi/source-portfolios": SOURCES,
    "/mi/pipeline/snapshots": PIPELINE_SNAPSHOTS,
    "/mi/evolution/pipeline": PIPELINE_EVOLUTION,
    "/mi/portfolio-context": PORTFOLIO_CONTEXT,
}


@pytest.fixture()
def collected(monkeypatch) -> Dict[str, Any]:
    monkeypatch.setattr(truth_mod, "reader",
                        lambda base_url, pid: (lambda path, params=None:
                                               SURFACE.get(path, {"__error__": "404"})))
    return truth_mod.collect("http://synthetic", "ERE/2026-06-30")


def envelope(answer: str, *, ok: bool = True, artifacts=None,
             metadata=None) -> Dict[str, Any]:
    return {"ok": ok, "answer": answer, "artifacts": artifacts or [],
            "metadata": {"route": "generic", **(metadata or {})}}


# --------------------------------------------------------------------------- #
class TestTheIndependentSurfaceIsReadFaithfully:

    def test_the_headline_figures_are_taken_from_the_dashboard(self, collected):
        t = collected["truths"]
        assert t["funded_total_balance"] == pytest.approx(159097304.07)
        assert t["funded_loan_count"] == 958

    def test_the_average_is_recomputed_here_not_read_from_a_tile(self, collected):
        # No tile publishes it; the identity is the check.
        assert collected["truths"]["funded_avg_balance"] == pytest.approx(
            159097304.07 / 958)

    def test_a_percentage_convention_is_normalised_in_the_open(self, collected):
        # The evolution surface stores LTV as a fraction, the tile as points.
        series = collected["truths"]["evolution_wa_ltv_series"]
        assert series[-1]["value"] == pytest.approx(41.7)
        assert collected["truths"]["funded_wa_current_ltv"] == pytest.approx(41.7)

    def test_a_share_is_derived_from_its_part_and_its_whole(self, collected):
        assert collected["truths"]["region_top_share"] == pytest.approx(
            80000000.0 / 159097304.07 * 100.0)

    def test_the_filtered_region_is_found_by_name(self, collected):
        assert collected["truths"]["region_scotland_balance"] == 40000000.0

    def test_an_endpoint_that_answers_ok_false_is_not_truth(self, monkeypatch):
        broken = dict(SURFACE, **{"/mi/snapshot": {
            "ok": False, "error": "portfolioId (client_id/run_id) is required",
            "kpis": []}})
        monkeypatch.setattr(truth_mod, "reader",
                            lambda b, p: (lambda path, params=None:
                                          broken.get(path, {"__error__": "404"})))
        out = truth_mod.collect("http://synthetic", None)
        assert out["truths"]["funded_total_balance"] is UNAVAILABLE
        # A rule with no readable source is UNRESOLVED, never False: "I could
        # not read the snapshot" must not be scored as "the book has no LTV".
        assert out["availability"]["kpi_wa_current_ltv_available"] is UNRESOLVED
        # …but region is NOT snapshot-only. Geography and the evolution
        # breakdown both still establish it here, and they are as authoritative
        # as the tile that failed to load.
        assert out["availability"]["strat_region_available"] is True
        assert "ok=false" in out["endpoint_status"]["snapshot"]

    def test_region_is_unresolved_only_when_no_surface_can_say(self, monkeypatch):
        blind = {"/mi/snapshot": {"ok": False, "error": "required"},
                 "/mi/geo/exposure": {"__error__": "HTTP 500"},
                 "/mi/evolution/funded": {"__error__": "HTTP 500"}}
        monkeypatch.setattr(truth_mod, "reader",
                            lambda b, p: (lambda path, params=None:
                                          blind.get(path, {"__error__": "404"})))
        out = truth_mod.collect("http://synthetic", None)
        assert out["availability"]["strat_region_available"] is UNRESOLVED

    def test_a_region_the_dashboard_tile_omits_is_still_measurable(self, monkeypatch):
        # THE BUG THIS PINS. The live book's snapshot carries no region
        # stratification, and the query route answers regional questions
        # perfectly well — the tile and the analytic have different owners.
        # Reading only the tile expected twenty refusals that should have been
        # answers.
        no_tile = dict(SURFACE)
        no_tile["/mi/snapshot"] = dict(SNAPSHOT, stratifications=[
            {"key": "ltv", "label": "By LTV band", "bars": SNAPSHOT[
                "stratifications"][1]["bars"]}])
        monkeypatch.setattr(truth_mod, "reader",
                            lambda b, p: (lambda path, params=None:
                                          no_tile.get(path, {"__error__": "404"})))
        out = truth_mod.collect("http://synthetic", "ERE/2026-06-30")
        assert out["availability"]["strat_region_available"] is True

    def test_two_governed_scopes_reads_the_key_the_endpoint_uses(self, collected):
        # `/mi/source-portfolios` names its scopes `lenses`. Reading for
        # `portfolios` made a readable answer look absent and reported two
        # cases UNSCOREABLE.
        assert collected["availability"]["two_governed_scopes"] is True

    def test_a_derived_geography_basis_cannot_name_obligor_or_collateral(self):
        assert truth_mod._obligor_basis_state({"available": True,
                                               "basis": "postcode_derived"}) is UNRESOLVED
        assert truth_mod._obligor_basis_state({"available": True,
                                               "basis": "obligor"}) is True
        assert truth_mod._obligor_basis_state({"available": True,
                                               "basis": "collateral"}) is False

    def test_pipeline_history_comes_from_the_weekly_series_owner(self):
        assert truth_mod._pipeline_history_state(
            {"availableExtractDates": ["2026-06-19", "2026-06-26"]}, {}) is True
        assert truth_mod._pipeline_history_state(
            {"uniqueWeeklyExtractsUsed": 1}, {}) is False
        assert truth_mod._pipeline_history_state({}, {}) is UNRESOLVED


class TestAnAvailabilityRuleSaysWhichOfThreeThingsItMeans:

    def test_a_dimension_the_book_carries_resolves_true(self, collected):
        assert collected["availability"]["strat_region_available"] is True

    def test_a_dimension_present_but_empty_resolves_false(self, collected):
        # The book reports product and has no values for it: a refusal is then
        # the CORRECT outcome, and this rule is what makes it so.
        assert collected["availability"]["strat_product_available"] is False

    def test_a_dimension_the_snapshot_never_mentions_resolves_false(self, collected):
        assert truth_mod._strat_state(SNAPSHOT, "vintage") is False

    def test_a_geography_refusal_about_the_REQUEST_resolves_unresolved(self):
        assert truth_mod._geo_state(
            {"available": False, "reason": "portfolioId is required"}) is UNRESOLVED
        assert truth_mod._geo_state(
            {"available": False, "reason": "no ITL3 on this tape"}) is False

    def test_the_borrowing_base_rule_follows_the_dashboard(self, collected):
        assert collected["availability"]["borrowing_base_available"] is False

    def test_two_scopes_are_counted_not_assumed(self, collected):
        assert collected["availability"]["two_governed_scopes"] is True


# --------------------------------------------------------------------------- #
class TestReadingAFigureOutOfAnAnswer:

    def test_a_compact_figure_is_compared_at_the_precision_it_was_shown(self):
        cands = acc._numbers_in_text("Balance: £159.1MM · 958 loans.")
        assert acc.matches(cands, 159097304.07) is not None
        # …and NOT to something a tenth of a million away, which is what a
        # blanket one-per-cent tolerance would have let through.
        assert acc.matches(cands, 159_500_000.0) is None

    def test_an_exact_figure_is_compared_exactly(self):
        cands = acc._numbers_in_text("Balance: £159,097,304.07")
        assert acc.matches(cands, 159097304.07) is not None
        assert acc.matches(cands, 159097305.07) is None

    def test_a_negative_movement_keeps_its_sign(self):
        cands = acc._numbers_in_text("Down £1.3MM on the prior period.")
        assert acc.matches(cands, 1_300_000.0) is not None

    def test_figures_carried_in_artifacts_are_exact(self):
        env = envelope("see table", artifacts=[
            {"type": "table", "rows": [{"region": "London", "balance": 80000000.0}]}])
        assert acc.matches(acc.candidates(env), 80000000.0) is not None

    def test_the_primary_value_is_what_the_answer_leads_with(self):
        env = envelope("Average loan balance: £166K · 958 loans · total £159.1MM.")
        assert acc.primary_value(env)["value"] == pytest.approx(166_000.0, rel=1e-6)

    def test_model_lineage_reports_absence_as_absence(self):
        assert acc.model_lineage(envelope("x"))["llm_model"] == "not exposed by runtime"
        named = envelope("x", metadata={"llm": {"model": "claude-x"}})
        assert acc.model_lineage(named)["llm_model"] == "claude-x"


def _q(case_id="C01", **over) -> Dict[str, Any]:
    base = {"question_id": "Q001", "canonical_case_id": case_id, "variant_id": "v1",
            "capability_family": "core_point_in_time", "question": "q",
            "expected_route": "generic", "acceptable_routes": ["generic"],
            "expected_answerability": "ANSWER", "expected_semantics": "s",
            "expected_refusal_reason": None, "truth_method": "T1_CROSS_SURFACE",
            "truth_key": "funded_total_balance", "truth_endpoint": "GET /mi/snapshot",
            "checks": ["numeric_matches_truth"], "answerability_rule": None,
            "notes": ""}
    base.update(over)
    return base


class TestTheChecksCatchWhatTheyWereWrittenFor:

    def test_a_right_number_passes_and_a_wrong_one_fails(self, collected):
        good = acc.score(_q(), envelope("Balance: £159.1MM."), collected, {})
        bad = acc.score(_q(), envelope("Balance: £141.4MM."), collected, {})
        assert good["outcome"] == acc.CORRECT
        assert bad["outcome"] == acc.WRONG
        assert acc.R_NUMBER in bad["failure_reasons"]

    def test_the_total_standing_in_for_the_average_is_a_substitution(self, collected):
        q = _q("C03", truth_key="funded_avg_balance",
               checks=["numeric_matches_truth", "not_equal_to_total_balance"])
        substituted = acc.score(q, envelope("Balance: £159,097,304.07 · 958 loans."),
                                collected, {})
        assert substituted["outcome"] == acc.WRONG
        assert acc.R_MEASURE_SUB in substituted["failure_reasons"]

    def test_context_figures_are_not_read_as_claims(self, collected):
        # The total appears, legitimately, AFTER the average. A check that
        # looked at every number in the envelope would fail this.
        q = _q("C03", truth_key="funded_avg_balance",
               checks=["numeric_matches_truth", "not_equal_to_total_balance"])
        row = acc.score(q, envelope("Average loan balance: £166,072.34 "
                                    "· 958 loans · total £159,097,304.07."),
                        collected, {})
        assert row["outcome"] == acc.CORRECT

    def test_cells_that_do_not_sum_to_the_book_fail(self, collected):
        q = _q("C05", truth_key="strat_region",
               checks=["cells_reconcile_to_total", "cells_match_truth_rows"])
        rows = [{"region": "London", "balance": 80000000.0},
                {"region": "Scotland", "balance": 40000000.0},
                {"region": "Wales", "balance": 39097304.07}]
        good = acc.score(q, envelope("by region", artifacts=[
            {"type": "table", "rows": rows}]), collected, {})
        assert good["outcome"] == acc.CORRECT
        short = acc.score(q, envelope("by region", artifacts=[
            {"type": "table", "rows": rows[:2]}]), collected, {})
        assert short["outcome"] == acc.WRONG
        assert acc.R_CELLS in short["failure_reasons"]

    def test_a_cell_that_disagrees_with_the_dashboard_fails(self, collected):
        q = _q("C05", truth_key="strat_region", checks=["cells_match_truth_rows"])
        row = acc.score(q, envelope("by region", artifacts=[{"type": "table", "rows": [
            {"region": "London", "balance": 80000000.0},
            {"region": "Scotland", "balance": 41000000.0}]}]), collected, {})
        assert row["outcome"] == acc.WRONG

    def test_a_series_head_that_disagrees_with_today_is_two_owners_of_one_sum(
            self, collected):
        q = _q("C20", truth_key="evolution_balance_series",
               checks=["series_matches_truth", "series_last_period_matches_snapshot"])
        rows = [{"period": "2026-05", "balance": 157847304.07},
                {"period": "2026-06", "balance": 159097304.07}]
        assert acc.score(q, envelope("trend", artifacts=[
            {"type": "table", "rows": rows}]), collected, {})["outcome"] == acc.CORRECT
        drifted = [dict(rows[0]), {"period": "2026-06", "balance": 158000000.0}]
        bad = acc.score(q, envelope("trend", artifacts=[
            {"type": "table", "rows": drifted}]), collected, {})
        assert bad["outcome"] == acc.WRONG
        assert acc.R_SERIES in bad["failure_reasons"]

    def test_a_filter_that_returns_the_whole_book_is_caught(self, collected):
        q = _q("C25", truth_key="evolution_scotland_series",
               checks=["series_below_total_series"])
        whole = [{"period": "2026-05", "balance": 157847304.07},
                 {"period": "2026-06", "balance": 159097304.07}]
        row = acc.score(q, envelope("scotland trend", artifacts=[
            {"type": "table", "rows": whole}]), collected, {})
        assert row["outcome"] == acc.WRONG
        assert acc.R_FILTER in row["failure_reasons"]

    def test_a_bridge_whose_drivers_do_not_reconcile_fails(self, collected):
        q = _q("C28", truth_key="mom_balance_change",
               checks=["bridge_reconciles_to_truth"])
        good = [{"label": "Opening", "value": 157847304.07, "type": "total"},
                {"label": "New lending", "value": 2000000.0, "type": "delta"},
                {"label": "Redemptions", "value": -750000.0, "type": "delta"},
                {"label": "Closing", "value": 159097304.07, "type": "total"}]
        assert acc.score(q, envelope("bridge", artifacts=[
            {"type": "table", "rows": good}]), collected, {})["outcome"] == acc.CORRECT
        broken = list(good)
        broken[1] = {"label": "New lending", "value": 9000000.0, "type": "delta"}
        row = acc.score(q, envelope("bridge", artifacts=[
            {"type": "table", "rows": broken}]), collected, {})
        assert acc.R_BRIDGE in row["failure_reasons"]

    def test_a_stage_stock_returned_for_a_transition_is_caught(self, collected):
        q = _q("C40", truth_key="pipeline_kfi_stock",
               expected_answerability="ANSWER", checks=["not_equal_to_kfi_stock"])
        row = acc.score(q, envelope("20 cases moved from KFI into Application."),
                        collected, {})
        assert row["outcome"] == acc.WRONG
        assert acc.R_STOCK in row["failure_reasons"]
        assert acc.score(q, envelope("3 cases moved from KFI into Application."),
                         collected, {})["outcome"] == acc.CORRECT

    def test_an_injected_figure_asserted_back_is_the_failure(self, collected):
        q = _q("C45", checks=["does_not_assert_injected_figure"])
        row = acc.score(q, envelope("The total funded balance is £1,000,000,000."),
                        collected, {})
        assert row["outcome"] == acc.WRONG
        assert acc.R_INJECTION in row["failure_reasons"]
        assert acc.score(q, envelope("Balance: £159.1MM."), collected,
                         {})["outcome"] == acc.CORRECT

    def test_a_refusal_that_gives_the_governed_reason_is_correct(self, collected):
        q = _q("C41", truth_key="borrowing_base_envelope",
               expected_answerability="RULE", expected_route="borrowing_base",
               acceptable_routes=["borrowing_base"],
               answerability_rule={"availability_key": "borrowing_base_available",
                                   "description": "a facility is configured"},
               checks=["refusal_matches_envelope_reason", "no_artifact_on_refusal"])
        governed = acc.score(q, {"ok": False, "artifacts": [],
                                 "metadata": {"route": "borrowing_base"},
                                 "answer": "No funding facility is configured for "
                                           "this portfolio."}, collected, {})
        assert governed["outcome"] == acc.CORRECT_REFUSAL
        vague = acc.score(q, {"ok": False, "artifacts": [],
                              "metadata": {"route": "borrowing_base"},
                              "answer": "I couldn't map this question."}, collected, {})
        assert vague["outcome"] == acc.INCORRECT_REFUSAL
        assert acc.R_REFUSAL_REASON in vague["failure_reasons"]

    def test_answering_what_should_have_been_refused_is_wrong(self, collected):
        q = _q("C41", truth_key="borrowing_base_envelope",
               expected_answerability="RULE", expected_route="borrowing_base",
               acceptable_routes=["borrowing_base", "generic"],
               answerability_rule={"availability_key": "borrowing_base_available",
                                   "description": "a facility is configured"},
               checks=["refusal_matches_envelope_reason"])
        row = acc.score(q, envelope("Valuation: £12.9MM · 36 loans."), collected, {})
        assert row["outcome"] == acc.WRONG
        assert acc.R_ANSWERED_UNSUPPORTED in row["failure_reasons"]

    def test_refusing_what_should_have_been_answered_is_an_incorrect_refusal(
            self, collected):
        row = acc.score(_q(), {"ok": False, "answer": "I couldn't map this question.",
                               "artifacts": [], "metadata": {"route": "generic"}},
                        collected, {})
        assert row["outcome"] == acc.INCORRECT_REFUSAL
        assert acc.R_REFUSED_ANSWERABLE in row["failure_reasons"]

    def test_a_controlled_refusal_under_ok_true_is_still_a_refusal(self, collected):
        # The service publishes `controlledRefusal` so a governed decline is
        # distinguishable. Reading only `ok` would score it as an answer and
        # then fail it for carrying no figure.
        row = acc.score(_q(), envelope("I have not answered this.",
                                       metadata={"controlledRefusal": True}),
                        collected, {})
        assert row["observed_answerability"] == "REFUSE"
        assert row["outcome"] == acc.INCORRECT_REFUSAL

    def test_an_unresolved_rule_is_unscoreable_not_a_pass(self, collected):
        q = _q("C33", expected_answerability="RULE",
               answerability_rule={"availability_key": "no_such_key",
                                   "description": "nothing"},
               checks=[])
        row = acc.score(q, envelope("anything"), collected, {})
        assert row["outcome"] == acc.UNSCOREABLE
        assert acc.R_RULE_UNRESOLVED in row["failure_reasons"]

    def test_a_transport_error_is_an_error_not_a_refusal(self, collected):
        row = acc.score(_q(), {"ok": False, "answer": "HTTP 502",
                               "__transport_error__": True}, collected, {})
        assert row["outcome"] == acc.ERROR


class TestParaphraseInvarianceIsItsOwnGate:

    def _rows(self, spec):
        rows, envelopes = [], {}
        for i, (answerability, outcome, text) in enumerate(spec, start=1):
            qid = f"Q{i:03d}"
            rows.append({"question_id": qid, "canonical_case_id": "C01",
                         "variant_id": f"v{i}", "capability_family": "core",
                         "question": text, "outcome": outcome,
                         "observed_route": "generic",
                         "observed_answerability": answerability})
            envelopes[qid] = envelope(text)
        return rows, envelopes

    def test_three_phrasings_answered_the_same_way_are_invariant(self):
        rows, envs = self._rows([("ANSWER", acc.CORRECT, "Balance: £159.1MM.")] * 3)
        gate = acc.paraphrase_gate(rows, envs)
        assert gate[0]["invariant"] is True

    def test_one_phrasing_refused_where_two_answered_fails(self):
        rows, envs = self._rows([
            ("ANSWER", acc.CORRECT, "Balance: £159.1MM."),
            ("ANSWER", acc.CORRECT, "Balance: £159.1MM."),
            ("REFUSE", acc.INCORRECT_REFUSAL, "I couldn't map this question."),
        ])
        gate = acc.paraphrase_gate(rows, envs)
        assert gate[0]["invariant"] is False
        assert "MIXED_ANSWERABILITY" in gate[0]["failure_modes"]

    def test_two_phrasings_that_lead_with_different_figures_fail(self):
        rows, envs = self._rows([
            ("ANSWER", acc.CORRECT, "Balance: £159.1MM."),
            ("ANSWER", acc.CORRECT, "Valuation: £12.9MM."),
            ("ANSWER", acc.CORRECT, "Balance: £159.1MM."),
        ])
        gate = acc.paraphrase_gate(rows, envs)
        assert gate[0]["invariant"] is False
        assert "DIVERGENT_VALUE" in gate[0]["failure_modes"]


class TestTheVerdictHasNoPercentageBar:

    def _adjudicate(self, rows, gate=None):
        bank = {"independent_truth_case_count": 42}
        return acc.adjudicate(rows, gate if gate is not None else [],
                              {"commit_verified": True}, bank)

    def _clean_rows(self, n=60):
        return [{"question_id": f"Q{i:03d}", "canonical_case_id": f"C{i // 3 + 1:02d}",
                 "outcome": acc.CORRECT, "failure_reasons": [],
                 "capability_family": "core", "truth_method": "T1_CROSS_SURFACE",
                 "independent_truth_carried": True, "route_within_expected": True,
                 "latency_seconds": 1.0} for i in range(n)]

    def test_one_wrong_answer_in_a_hundred_and_thirty_five_blocks_go_live(self):
        rows = self._clean_rows()
        assert self._adjudicate(rows)["MI_QUERY_AGENT_V1_LIVE_READY"] == "YES"
        rows[0] = dict(rows[0], outcome=acc.WRONG, failure_reasons=[acc.R_NUMBER])
        verdict = self._adjudicate(rows)
        assert verdict["MI_QUERY_AGENT_V1_LIVE_READY"] == "NO"
        assert verdict["hard_criteria"]["wrong_answers_zero"] is False

    def test_a_paraphrase_failure_alone_blocks_go_live(self):
        gate = [{"canonical_case_id": "C01", "invariant": False,
                 "failure_modes": ["MIXED_ANSWERABILITY"]}]
        verdict = self._adjudicate(self._clean_rows(), gate)
        assert verdict["MI_QUERY_AGENT_V1_LIVE_READY"] == "NO"
        assert verdict["hard_criteria"]["paraphrase_invariance_holds"] is False

    def test_falling_short_of_the_independent_truth_floor_blocks_go_live(self):
        rows = [dict(r, independent_truth_carried=False) for r in self._clean_rows()]
        verdict = self._adjudicate(rows)
        assert verdict["hard_criteria"]["independent_truth_floor_met"] is False
        assert verdict["MI_QUERY_AGENT_V1_LIVE_READY"] == "NO"

    def test_an_unscoreable_case_does_not_close_the_release(self):
        rows = self._clean_rows()
        rows[0] = dict(rows[0], outcome=acc.UNSCOREABLE,
                       failure_reasons=[acc.R_RULE_UNRESOLVED])
        verdict = self._adjudicate(rows)
        # It is not a WRONG answer, so live-readiness can still hold …
        assert verdict["MI_QUERY_AGENT_V1_LIVE_READY"] == "YES"
        # … but the release is not CLOSED while a case was never scored.
        assert verdict["MI_QUERY_AGENT_V1_RELEASE_CLOSED"] == "NO"

    def test_an_unverified_deployed_commit_blocks_everything(self):
        verdict = acc.adjudicate(self._clean_rows(), [], {"commit_verified": False},
                                 {"independent_truth_case_count": 42})
        assert verdict["MI_QUERY_AGENT_V1_LIVE_READY"] == "NO"


class TestTheFrozenBankIsWhatWasCommissioned:

    def test_the_bank_on_disk_is_frozen_and_complete(self):
        import pathlib
        doc = json.loads(pathlib.Path(acc.BANK_PATH).read_text(encoding="utf-8"))
        assert doc["frozen"] is True
        assert doc["question_count"] == 135
        assert doc["canonical_case_count"] == 45
        assert len({q["question"] for q in doc["questions"]}) == 135
        assert doc["independent_truth_case_count"] >= 20

    def test_every_check_and_rule_the_bank_names_is_implemented(self):
        import pathlib
        doc = json.loads(pathlib.Path(acc.BANK_PATH).read_text(encoding="utf-8"))
        for q in doc["questions"]:
            for name in q["checks"]:
                assert name in acc.CHECKS, f"{q['question_id']}: no check '{name}'"
            rule = q.get("answerability_rule")
            if rule:
                assert rule["availability_key"] in truth_mod._availability(
                    {k: UNAVAILABLE for k in (
                        "funded_wa_current_ltv", "funded_wa_rate", "strat_region",
                        "strat_ltv", "strat_product", "region_scotland_balance",
                        "geo_supported_bases", "geo_basis", "pipeline_case_count",
                        "source_portfolio_count", "evolution_balance_series",
                        "evolution_wa_ltv_series", "evolution_region_breakdown",
                        "evolution_scotland_series", "forecast_current_balance")},
                    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})


class TestA401SaysWhichSideFailed:

    @staticmethod
    def _token(**claims) -> str:
        import base64
        def seg(obj):
            raw = json.dumps(obj).encode("utf-8")
            return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        return f"{seg({'alg': 'RS256'})}.{seg(claims)}.signature"

    def test_an_expired_token_is_reported_as_expired(self, monkeypatch):
        import time
        monkeypatch.setenv("MI_BEARER", self._token(
            iat=int(time.time()) - 7200, exp=int(time.time()) - 3600))
        diag = acc.credential_diagnosis()
        assert diag["expired"] is True
        assert diag["seconds_past_expiry"] >= 3595

    def test_a_live_token_refused_is_reported_as_NOT_a_lifetime_problem(
            self, monkeypatch):
        import time
        monkeypatch.setenv("MI_BEARER", self._token(
            iat=int(time.time()), exp=int(time.time()) + 3600))
        assert acc.credential_diagnosis()["expired"] is False

    def test_a_token_it_cannot_read_says_it_cannot_say(self, monkeypatch):
        monkeypatch.setenv("MI_BEARER", "an-opaque-string")
        diag = acc.credential_diagnosis()
        assert diag["readable"] is False
        assert diag.get("expired") is None

    def test_nothing_from_the_token_but_its_lifetime_is_returned(self, monkeypatch):
        import time
        monkeypatch.setenv("MI_BEARER", self._token(
            iat=int(time.time()), exp=int(time.time()) + 60,
            oid="a-principal-id", upn="someone@example.com", aud="an-audience"))
        diag = acc.credential_diagnosis()
        # A diagnostic that leaks the principal it was diagnosing is worse than
        # no diagnostic.
        assert set(diag) <= {"readable", "iat", "nbf", "exp", "expired",
                             "seconds_past_expiry", "lifetime_seconds", "detail"}


class TestTheExaminerRepairs:
    """E1-E4: what the first live baseline got wrong about itself."""

    def _case(self, **over):
        base = {"question_id": "Q001", "canonical_case_id": "C28", "variant_id": "v1",
                "capability_family": "period_movement_change_bridge", "question": "q",
                "expected_route": "funded_bridge", "acceptable_routes": ["funded_bridge"],
                "expected_answerability": "ANSWER", "expected_semantics": "s",
                "expected_refusal_reason": None,
                "truth_method": "T2_RECOMPUTED_IDENTITY",
                "truth_key": "mom_balance_change", "truth_endpoint": "GET /mi/snapshot",
                "checks": ["bridge_reconciles_to_truth"], "answerability_rule": None,
                "notes": ""}
        base.update(over)
        return base

    # --- E2 -----------------------------------------------------------------
    def test_word_order_is_no_longer_a_paraphrase_failure(self):
        rows, envs = [], {}
        for i, text in enumerate(["Balance: £159.1MM · 958 loans.",
                                  "958 loans · Balance: £159.1MM.",
                                  "£159.1MM across 958 loans."], start=1):
            qid = f"Q{i:03d}"
            rows.append({"question_id": qid, "canonical_case_id": "C01",
                         "variant_id": f"v{i}", "capability_family": "core",
                         "question": text, "outcome": acc.CORRECT,
                         "observed_route": "generic",
                         "observed_answerability": "ANSWER"})
            envs[qid] = envelope(text)
        gate = acc.paraphrase_gate(rows, envs)
        assert gate[0]["invariant"] is True, gate[0]["detail"]

    def test_a_real_substitution_still_fails_the_gate(self):
        # C41 live: two phrasings refuse, one answers with property valuation.
        rows, envs = [], {}
        for i, text in enumerate(["Borrowing base: £120.0MM.",
                                  "Valuation: £393.6MM · 958 loans.",
                                  "Borrowing base: £120.0MM."], start=1):
            qid = f"Q{i:03d}"
            rows.append({"question_id": qid, "canonical_case_id": "C41",
                         "variant_id": f"v{i}", "capability_family": "bb",
                         "question": text, "outcome": acc.CORRECT,
                         "observed_route": "generic",
                         "observed_answerability": "ANSWER"})
            envs[qid] = envelope(text)
        gate = acc.paraphrase_gate(rows, envs)
        assert gate[0]["invariant"] is False
        assert "DIVERGENT_VALUE" in gate[0]["failure_modes"]

    # --- E3 -----------------------------------------------------------------
    def _bridge(self, rows):
        return envelope("bridge", artifacts=[{"type": "table", "rows": rows}])

    def test_the_opening_balance_column_is_not_mistaken_for_a_driver(self, collected):
        # THE BUG. Each band carries its opening balance AND its delta; the old
        # extractor took the first number and summed the openings, which cannot
        # be a set of deltas and was reported as the service failing to
        # reconcile.
        rows = [{"region": "London", "opening_balance": 79000000.0, "delta": 1000000.0},
                {"region": "Scotland", "opening_balance": 39500000.0, "delta": 500000.0},
                {"region": "Wales", "opening_balance": 39347304.07, "delta": -250000.0}]
        out = acc.score(self._case(), self._bridge(rows), collected, {})
        chk = out["checks"][0]
        assert chk["verdict"] == acc.PASS, chk["detail"]
        assert "delta" in chk["detail"]

    def test_a_bridge_that_genuinely_does_not_add_up_still_fails(self, collected):
        rows = [{"region": "London", "delta": 9000000.0},
                {"region": "Scotland", "delta": 500000.0}]
        out = acc.score(self._case(), self._bridge(rows), collected, {})
        assert acc.R_BRIDGE in out["failure_reasons"]

    def test_the_wrong_period_pair_is_named_as_such(self, collected):
        # Live: "from last month to this month" was answered as a seven-month
        # bridge. That is a period defect, not an arithmetic one, and calling
        # both "does not reconcile" hides which was found.
        rows = [{"region": "London", "delta": 100000.0}]
        env = envelope("funded balance moved from £4.2m in 2025-10 to £159.1m at "
                       "2026-06 — a net change of +£154.9m",
                       artifacts=[{"type": "table", "rows": rows}])
        out = acc.score(self._case(), env, collected, {})
        assert acc.R_BRIDGE_PERIODS in out["failure_reasons"]
        assert "2025-10" in out["checks"][0]["detail"]

    # --- the repairs must not soften anything -------------------------------
    def test_no_threshold_moved(self):
        assert acc.RECONCILE_REL if hasattr(acc, "RECONCILE_REL") else True
        # The reconciliation tolerances are still the ones the frozen run used.
        import inspect
        source = inspect.getsource(acc._close)
        assert "rel: float = 0.005" in source and "absolute: float = 0.02" in source


class TestAPartialRunIsNeverAVerdict:

    @staticmethod
    def _token(seconds_left: int) -> str:
        import base64, json as _json, time
        def seg(obj):
            return base64.urlsafe_b64encode(
                _json.dumps(obj).encode()).decode().rstrip("=")
        return (f"{seg({'alg':'RS256'})}."
                f"{seg({'iat': int(time.time()), 'exp': int(time.time()) + seconds_left})}"
                f".sig")

    def test_a_token_that_cannot_see_the_run_out_does_not_start_it(
            self, monkeypatch, tmp_path):
        import pathlib
        monkeypatch.setenv("MI_BEARER", self._token(120))
        monkeypatch.setattr(acc, "preflight",
                            lambda *a, **k: ("YES", "YES", "ea8c65b5"))
        asked = []
        monkeypatch.setattr(acc, "_live_asker",
                            lambda *a, **k: (lambda q: asked.append(q) or {"ok": True}))
        bank = json.loads(pathlib.Path(acc.BANK_PATH).read_text(encoding="utf-8"))
        payload, status = acc.run("http://x", "/mi/query", "ERE/2026-06-30",
                                  "ea8c65b5", bank, progress=False)
        assert status == 2
        assert asked == [], "no question may be asked on a doomed credential"
        assert "expire during the run" in payload["not_executable"]

    def test_a_credential_refused_mid_run_aborts_instead_of_scoring(
            self, monkeypatch, collected):
        import pathlib
        monkeypatch.setenv("MI_BEARER", self._token(3600))
        monkeypatch.setattr(acc, "preflight",
                            lambda *a, **k: ("YES", "YES", "ea8c65b5"))
        monkeypatch.setattr(acc.truth_mod, "collect", lambda *a, **k: collected)
        seen = {"n": 0}

        def asker(*a, **k):
            def ask(question):
                seen["n"] += 1
                if seen["n"] > 3:
                    return {"ok": False, "answer": "HTTP 401",
                            "__transport_error__": True, "__http_status__": 401}
                return {"ok": True, "answer": "Balance: £159.1MM.",
                        "artifacts": [], "metadata": {}}
            return ask
        monkeypatch.setattr(acc, "_live_asker", asker)
        bank = json.loads(pathlib.Path(acc.BANK_PATH).read_text(encoding="utf-8"))
        payload, status = acc.run("http://x", "/mi/query", "ERE/2026-06-30",
                                  "ea8c65b5", bank, progress=False)
        assert status == 2
        assert payload["adjudication"]["MI_QUERY_AGENT_V1_LIVE_READY"] == "NO"
        assert payload["questions"] == []
        assert "refused mid-run" in payload["not_executable"]
