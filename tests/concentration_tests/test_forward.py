"""Three-state evaluation: Funded stays contractual and untouched; Expected
Forecast reuses the forecast engine's probabilities on both sides of every
share; Full Pipeline is the labelled maximum-exposure stress; drivers
reconcile; the emerging-risk ranking is rule-fixed."""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent.concentration_tests import forward
from mi_agent.concentration_tests.evaluation import evaluate_active_tests
from mi_agent.concentration_tests.metrics import PROBABILITY_COL, evaluate_metric
from mi_agent.concentration_tests.models import ActiveConfiguration, ActiveTest


def active(metric_id, threshold, operator="max", params=None, **kw):
    return ActiveTest(metric_id=metric_id, threshold=threshold,
                      operator=operator,
                      display_name=kw.pop("display_name", metric_id),
                      parameters=params or {}, **kw)


def config(*tests):
    return ActiveConfiguration(client_id="c", version=2, tests=list(tests))


@pytest.fixture()
def pipeline_frame() -> pd.DataFrame:
    """Five active cases (three South West), one withdrawn, one completed,
    one unknown, one active without a probability."""
    return pd.DataFrame({
        "pipeline_case_identifier": ["P1", "P2", "P3", "P4", "P5",
                                     "PW", "PC", "PU", "PN"],
        "current_outstanding_balance": [640_000, 510_000, 455_000, 390_000,
                                        120_000, 900_000, 800_000, 700_000,
                                        600_000],
        "geographic_region_obligor": ["South West", "South West", "South West",
                                      "London", "London", "South West",
                                      "South West", "South West", "South West"],
        "pipeline_stage": ["OFFER", "OFFER", "APPLICATION", "APPLICATION",
                           "KFI", "WITHDRAWN", "COMPLETED", "UNKNOWN", "KFI"],
        "pipeline_status": ["pipeline"] * 5 + ["withdrawn", "funded",
                                               "unknown", "pipeline"],
        "completion_probability": [0.8, 0.8, 0.5, 0.5, 0.25,
                                   None, None, None, None],
        "completion_probability_source": ["historical_stage_rate"] * 5
                                         + [None, None, None, "missing_stage"],
        "expected_completion_month": ["2026-01", "2026-02", "2026-02",
                                      "2026-03", "2026-03", None, None, None,
                                      None],
    })


@pytest.fixture()
def sw_funded() -> pd.DataFrame:
    """Funded book: 18.4m South West of 100m total (18.4%)."""
    return pd.DataFrame({
        "loan_id": [f"L{i}" for i in range(10)],
        "current_outstanding_balance": [18.4e6 / 2] * 2 + [81.6e6 / 8] * 8,
        "collateral_geography": ["South West"] * 2 + ["London"] * 8,
        "current_interest_rate": [6.0] * 2 + [5.0] * 8,
    })


class TestStateFrames:
    def test_exclusions_are_counted_by_reason(self, sw_funded, pipeline_frame):
        frame, meta = forward.build_state_frame(sw_funded, pipeline_frame,
                                                forward.STATE_FULL)
        assert meta["excludedPipeline"] == {
            "withdrawn_or_cancelled": 1, "already_completed": 1,
            "unknown_stage": 1}
        # 10 funded + 6 active (incl. the no-probability case in FULL).
        assert len(frame) == 16

    def test_expected_drops_cases_without_a_probability(self, sw_funded,
                                                        pipeline_frame):
        frame, meta = forward.build_state_frame(sw_funded, pipeline_frame,
                                                forward.STATE_EXPECTED)
        assert meta["pipelineRowsWithoutProbability"] == 1  # PN, disclosed
        assert meta["pipelineRowsIncluded"] == 5
        pipeline_rows = frame[frame[forward.STATE_COMPONENT_COL] == "pipeline"]
        assert set(pipeline_rows["pipeline_case_identifier"]) == \
            {"P1", "P2", "P3", "P4", "P5"}

    def test_funded_rows_carry_probability_one(self, sw_funded, pipeline_frame):
        frame, _ = forward.build_state_frame(sw_funded, pipeline_frame,
                                             forward.STATE_EXPECTED)
        funded_rows = frame[frame[forward.STATE_COMPONENT_COL] == "funded"]
        assert (funded_rows[PROBABILITY_COL] == 1.0).all()

    def test_basis_aliases_are_disclosed(self, sw_funded, pipeline_frame):
        frame, meta = forward.build_state_frame(sw_funded, pipeline_frame,
                                                forward.STATE_EXPECTED)
        assert any(a.startswith("collateral_geography :=")
                   for a in meta["aliasedBasisColumns"])
        pipeline_rows = frame[frame[forward.STATE_COMPONENT_COL] == "pipeline"]
        assert pipeline_rows["collateral_geography"].notna().all()


class TestExpectedCalculations:
    def test_expected_share_moves_numerator_and_denominator(self, lib,
                                                            sw_funded,
                                                            pipeline_frame):
        frame, _ = forward.build_state_frame(sw_funded, pipeline_frame,
                                             forward.STATE_EXPECTED)
        comp = evaluate_metric(frame, lib, lib.get("geo_region_share"),
                               {"regions": ["South West"]})
        # Expected SW pipeline: 640k×0.8 + 510k×0.8 + 455k×0.5 = 1,147,500.
        # Expected non-SW pipeline: 390k×0.5 + 120k×0.25 = 225,000.
        exp_num = 18.4e6 + 1_147_500
        exp_den = 100e6 + 1_147_500 + 225_000
        assert comp.numerator_value == pytest.approx(exp_num, abs=1)
        assert comp.denominator_value == pytest.approx(exp_den, abs=1)
        assert comp.value == pytest.approx(exp_num / exp_den * 100, abs=0.01)

    def test_expected_weighted_average_uses_expected_weights(self, lib,
                                                             sw_funded,
                                                             pipeline_frame):
        pipe = pipeline_frame.assign(current_interest_rate=7.0)
        frame, _ = forward.build_state_frame(sw_funded, pipe,
                                             forward.STATE_EXPECTED)
        comp = evaluate_metric(frame, lib, lib.get("rate_gross_wac"), {})
        w_funded = sw_funded.current_outstanding_balance
        funded_wsum = float((sw_funded.current_interest_rate * w_funded).sum())
        pipe_w = 1_147_500 + 225_000
        expected = (funded_wsum + 7.0 * pipe_w) / (float(w_funded.sum()) + pipe_w)
        assert comp.value == pytest.approx(round(expected, 2), abs=0.01)

    def test_full_pipeline_uses_full_balances(self, lib, sw_funded,
                                              pipeline_frame):
        frame, _ = forward.build_state_frame(sw_funded, pipeline_frame,
                                             forward.STATE_FULL)
        comp = evaluate_metric(frame, lib, lib.get("geo_region_share"),
                               {"regions": ["South West"]})
        # Full SW pipeline: 640k + 510k + 455k + 600k (PN, no prob needed).
        full_num = 18.4e6 + 2_205_000
        full_den = 100e6 + 2_205_000 + 510_000
        assert comp.value == pytest.approx(full_num / full_den * 100, abs=0.01)


class TestThreeStateOrchestration:
    def _run(self, lib, funded, pipeline, cfg, **kw):
        ev = evaluate_active_tests(funded, None, cfg, lib,
                                   reporting_date="2025-11-30")
        return forward.extend_with_forward_states(
            ev, cfg, lib, funded, pipeline,
            forecast_meta={"stagesUsingConfigFallback": []}, **kw)

    def test_funded_results_are_unchanged_by_the_extension(self, lib,
                                                           sw_funded,
                                                           pipeline_frame):
        cfg = config(active("geo_region_share", 20.0,
                            params={"regions": ["South West"]},
                            display_name="SW"))
        plain = evaluate_active_tests(sw_funded, None, cfg, lib,
                                      reporting_date="2025-11-30")
        extended = self._run(lib, sw_funded, pipeline_frame, cfg)
        funded_keys = ("currentValue", "status", "headroom", "utilization",
                       "numeratorValue", "denominatorValue",
                       "loansInNumerator", "totalLoans")
        for key in funded_keys:
            assert extended["tests"][0][key] == plain["tests"][0][key]

    def test_expected_breach_with_funded_pass(self, lib, sw_funded,
                                              pipeline_frame):
        # Warning band tightened to 97% so funded 18.4% passes while the
        # expected 19.28% crosses the 19.2% limit — the pass→expected-breach
        # archetype.
        cfg = config(active("geo_region_share", 19.2,
                            params={"regions": ["South West"]},
                            display_name="SW", warning_fraction=0.97))
        out = self._run(lib, sw_funded, pipeline_frame, cfg)
        t = out["tests"][0]
        assert t["status"] == "pass"
        assert t["expected"]["value"] == pytest.approx(19.28, abs=0.01)
        assert t["expectedBreach"] is True
        assert t["expected"]["status"] == "breach"
        assert t["changeFundedToExpected"] == pytest.approx(0.88, abs=0.01)
        assert out["summary"]["expectedBreaches"] == 1
        risks = out["emergingRisks"]
        assert risks[0]["category"] == forward.RISK_EXPECTED_BREACH
        assert "expected to breach" in risks[0]["statement"]

    def test_full_pipeline_only_breach_is_classified(self, lib, sw_funded,
                                                     pipeline_frame):
        # Full 20.06% > 20.0 limit; expected 19.28% and funded 18.4% do not
        # breach. Buffer lowered so the low-expected-headroom rule (higher
        # rank) does not claim the test first.
        cfg = config(active("geo_region_share", 20.0,
                            params={"regions": ["South West"]},
                            display_name="SW"))
        out = self._run(lib, sw_funded, pipeline_frame, cfg,
                        headroom_buffer_pp=0.5)
        t = out["tests"][0]
        assert t["expectedBreach"] is False
        assert t["fullPipelineBreach"] is True
        assert t["fullPipeline"]["value"] == pytest.approx(20.06, abs=0.01)
        risks = out["emergingRisks"]
        assert any(r["category"] == forward.RISK_STRESS_ONLY for r in risks)
        assert out["summary"]["fullPipelineBreaches"] == 1
        assert out["summary"]["expectedBreaches"] == 0

    def test_maximum_loan_semantics_never_probability_weighted(self, lib,
                                                               sw_funded,
                                                               pipeline_frame):
        cfg = config(active("balance_maximum", 9_000_000,
                            params={"balance_basis": "current"},
                            display_name="Max loan"))
        out = self._run(lib, sw_funded, pipeline_frame, cfg)
        t = out["tests"][0]
        assert t["forecastTreatment"] == forward.TREATMENT_SCENARIO
        assert t["expected"]["status"] == "indicative_only"
        assert t["expected"]["value"] is None
        # Full pipeline still reports the true maximum (a funded loan here).
        assert t["fullPipeline"]["value"] == pytest.approx(10_200_000, abs=1)

    def test_count_covenants_are_unsupported_forward(self, lib, sw_funded,
                                                     pipeline_frame):
        funded = sw_funded.assign(borrower_identifier=[f"B{i}" for i in range(10)])
        cfg = config(active("borrower_max_loans", 5, params={},
                            display_name="Loans per borrower"))
        out = self._run(lib, funded, pipeline_frame, cfg)
        t = out["tests"][0]
        assert t["forecastTreatment"] == forward.TREATMENT_UNSUPPORTED
        assert t["expected"] is None
        assert t["fullPipeline"] is None
        assert t["currentValue"] == 1.0  # funded result intact

    def test_unavailable_funded_test_stays_unavailable_forward(self, lib,
                                                               sw_funded,
                                                               pipeline_frame):
        cfg = config(active("ltv_weighted_average", 35.0,
                            params={"ltv_basis": "current"},
                            display_name="WA LTV"))
        out = self._run(lib, sw_funded, pipeline_frame, cfg)
        t = out["tests"][0]
        assert t["status"] == "unavailable"
        assert t["expected"] is None and t["fullPipeline"] is None

    def test_missing_pipeline_means_no_forward_states(self, lib, sw_funded):
        cfg = config(active("geo_region_share", 20.0,
                            params={"regions": ["South West"]}))
        out = self._run(lib, sw_funded, None, cfg)
        t = out["tests"][0]
        # Funded numbers are never dressed up as a forecast.
        assert "expected" not in t or t.get("expected") is None
        assert out["states"]["available"] is False
        assert "No active in-scope pipeline" in out["states"]["reason"]

    def test_expected_breach_horizon_is_the_crossing_month(self, lib,
                                                           sw_funded,
                                                           pipeline_frame):
        cfg = config(active("geo_region_share", 19.0,
                            params={"regions": ["South West"]},
                            display_name="SW"))
        out = self._run(lib, sw_funded, pipeline_frame, cfg)
        t = out["tests"][0]
        assert t["expectedBreach"] is True
        horizon = t["expectedBreachHorizon"]
        assert horizon["available"] is True
        assert horizon["period"] == "2026-02"


class TestDrivers:
    def _drivers(self, lib, funded, pipeline, threshold=20.0):
        cfg = config(active("geo_region_share", threshold,
                            params={"regions": ["South West"]},
                            display_name="SW", test_id="ct_sw"))
        ev = evaluate_active_tests(funded, None, cfg, lib)
        return forward.compute_drivers_for_test(
            cfg, lib, "ct_sw", funded, pipeline, ev["tests"][0])

    def test_drivers_reconcile_to_expected_numerator_movement(self, lib,
                                                              sw_funded,
                                                              pipeline_frame):
        out = self._drivers(lib, sw_funded, pipeline_frame)
        assert out["available"] and out["reconciles"] is True
        assert out["expectedNumeratorMovement"] == pytest.approx(1_147_500, abs=1)
        listed = sum(d["expectedContribution"] for d in out["drivers"])
        assert listed == pytest.approx(1_147_500, abs=1)

    def test_ranking_is_deterministic_by_contribution(self, lib, sw_funded,
                                                      pipeline_frame):
        out = self._drivers(lib, sw_funded, pipeline_frame)
        contribs = [d["expectedContribution"] for d in out["drivers"]]
        assert contribs == sorted(contribs, reverse=True)
        assert out["drivers"][0]["caseId"] == "P1"

    def test_excluded_and_funded_loans_never_appear(self, lib, sw_funded,
                                                    pipeline_frame):
        out = self._drivers(lib, sw_funded, pipeline_frame)
        ids = {d["caseId"] for d in out["drivers"]}
        assert ids.isdisjoint({"PW", "PC", "PU", "PN"})
        assert not ids & set(sw_funded.loan_id)

    def test_contribution_reflects_the_forecast_probability(self, lib,
                                                            sw_funded,
                                                            pipeline_frame):
        out = self._drivers(lib, sw_funded, pipeline_frame)
        p1 = next(d for d in out["drivers"] if d["caseId"] == "P1")
        assert p1["completionProbability"] == 0.8
        assert p1["expectedContribution"] == pytest.approx(
            p1["balance"] * 0.8, abs=0.01)
        assert p1["fullContribution"] == p1["balance"]
        assert p1["probabilitySource"] == "historical_stage_rate"

    def test_unsupported_families_refuse_drivers(self, lib, sw_funded,
                                                 pipeline_frame):
        cfg = config(active("balance_maximum", 1.0,
                            params={"balance_basis": "current"},
                            test_id="ct_max"))
        ev = evaluate_active_tests(sw_funded, None, cfg, lib)
        out = forward.compute_drivers_for_test(cfg, lib, "ct_max", sw_funded,
                                               pipeline_frame, ev["tests"][0])
        assert out["available"] is False
        assert "exposure-weighted" in out["reason"]


class TestEmergingRisks:
    def test_rule_order_is_fixed(self):
        tests = [
            {"testId": "a", "displayName": "Stress only", "unit": "percent",
             "operator": "max", "threshold": 10.0, "currentValue": 8.0,
             "status": "pass", "fullPipelineBreach": True,
             "expectedBreach": False, "fullPipeline": {"value": 11.0},
             "expected": {"status": "pass", "headroom": 2.0}},
            {"testId": "b", "displayName": "Current breach", "unit": "percent",
             "operator": "max", "threshold": 10.0, "currentValue": 11.0,
             "status": "breach"},
            {"testId": "c", "displayName": "Expected breach", "unit": "percent",
             "operator": "max", "threshold": 10.0, "currentValue": 9.0,
             "status": "warning", "expectedBreach": True,
             "expected": {"value": 10.5, "breachAmount": 0.5,
                          "status": "breach", "headroom": -0.5}},
            {"testId": "d", "displayName": "Low headroom", "unit": "percent",
             "operator": "max", "threshold": 10.0, "currentValue": 9.0,
             "status": "warning",
             "expected": {"value": 9.4, "status": "warning", "headroom": 0.6}},
        ]
        risks = forward.identify_emerging_risks(tests)
        assert [r["category"] for r in risks] == [
            forward.RISK_CURRENT_BREACH, forward.RISK_EXPECTED_BREACH,
            forward.RISK_LOW_EXPECTED_HEADROOM, forward.RISK_STRESS_ONLY]

    def test_weak_forecast_appends_a_limitation(self):
        risks = forward.identify_emerging_risks([], forecast_weak=True)
        assert len(risks) == 1
        assert risks[0]["category"] == forward.RISK_LIMITATION
        assert "sufficiency floor" in risks[0]["statement"]

    def test_stress_wording_never_reads_as_a_prediction(self):
        tests = [{"testId": "a", "displayName": "SW", "unit": "percent",
                  "operator": "max", "threshold": 10.0, "currentValue": 8.0,
                  "status": "pass", "fullPipelineBreach": True,
                  "expectedBreach": False, "fullPipeline": {"value": 11.0},
                  "expected": {"status": "pass", "headroom": 2.0}}]
        [risk] = forward.identify_emerging_risks(tests)
        assert "maximum-exposure scenario" in risk["statement"]
        assert "not an expected outcome" in risk["statement"]
