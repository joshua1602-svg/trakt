#!/usr/bin/env python3
"""Phase 3A — the Weekly Portfolio Brief must be deterministic and honest.

A brief is only useful if a reader can trust three things: that a number is the
same number the dashboard shows, that silence means "nothing material happened"
rather than "this broke", and that the same week always produces the same brief.

These tests pin all three, plus the measures that are genuinely new here
(mean/median ticket, balance-weighted LTV with coverage, band mix).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from mi_agent_api import insight_config as cfg
from mi_agent_api import insight_generators as gen
from mi_agent_api import insight_metrics as im
from mi_agent_api import insight_engine as eng
from mi_agent_api.insight_contract import (
    CONCENTRATION_PROXIMITY, DATA_QUALITY, LTV_MIX_SHIFT, PIPELINE_MOVEMENT,
    SEVERITY_ATTENTION, SEVERITY_CONCERN, SEVERITY_INFO, TICKET_SIZE,
    WEIGHTED_LTV, Insight,
)

CTX = {"tenant_id": "ERE", "portfolio_id": "ERE", "portfolio_context": "total",
       "as_of_date": "2026-08-07", "comparison_date": "2026-07-31",
       "run_id": "2026-08-07"}


def frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame({
        im.CASE_KEY: [r.get("id", f"C{i}") for i, r in enumerate(rows)],
        im.BALANCE: [r.get("amount", 100_000) for r in rows],
        im.LTV: [r.get("ltv", 0.30) for r in rows],
        im.TICKET_BAND: [r.get("ticket_band", "100-150k") for r in rows],
        im.LTV_BAND: [r.get("ltv_band", "30-40%") for r in rows],
        "broker_channel": [r.get("broker", "Air") for r in rows],
        "geographic_region_obligor": [r.get("region", "London") for r in rows],
    })


def many(n: int, **kw) -> List[Dict[str, Any]]:
    return [{"id": f"C{i}", **kw} for i in range(n)]


# =========================================================================== #
# Metrics
# =========================================================================== #
class TestTicketSize(unittest.TestCase):
    def test_reports_mean_and_median_together(self):
        m = im.ticket_size(frame(many(3, amount=100)), frame(many(3, amount=50)))
        self.assertEqual(m["current_average"], 100.0)
        self.assertEqual(m["current_median"], 100.0)
        self.assertEqual(m["average_change_pct"], 100.0)
        self.assertEqual(m["median_change_pct"], 100.0)

    def test_a_tail_moves_the_mean_but_not_the_median(self):
        """The reason both are reported: they answer different questions."""
        prior = frame([{"id": f"C{i}", "amount": 100} for i in range(9)])
        current = frame([{"id": f"C{i}", "amount": 100} for i in range(9)]
                        + [{"id": "BIG", "amount": 100_000}])
        m = im.ticket_size(current, prior)
        self.assertGreater(m["average_change_pct"], 100)
        self.assertEqual(m["median_change_pct"], 0.0)

    def test_duplicate_identifiers_do_not_inflate_the_case_count(self):
        f = frame([{"id": "A", "amount": 100}, {"id": "A", "amount": 100}])
        m = im.ticket_size(f, None)
        self.assertEqual(m["current_case_count"], 1)
        self.assertEqual(m["current_average"], 200.0)

    def test_empty_frames_do_not_raise(self):
        m = im.ticket_size(None, None)
        self.assertIsNone(m["current_average"])
        self.assertEqual(m["current_case_count"], 0)


class TestWeightedLtv(unittest.TestCase):
    def test_is_balance_weighted_not_a_straight_average(self):
        f = frame([{"id": "A", "amount": 900_000, "ltv": 0.80},
                   {"id": "B", "amount": 100_000, "ltv": 0.20}])
        m = im.weighted_ltv(f, None)
        self.assertAlmostEqual(m["current"], 0.74, places=4)   # not 0.50

    def test_coverage_is_by_balance_not_by_case(self):
        """A few large cases with no LTV distort far more than many small ones."""
        f = frame([{"id": "A", "amount": 900_000, "ltv": None},
                   {"id": "B", "amount": 100_000, "ltv": 0.20}])
        m = im.weighted_ltv(f, None)
        self.assertEqual(m["current_coverage_pct"], 10.0)      # not 50%
        self.assertEqual(m["current_excluded_balance"], 900_000.0)

    def test_change_is_reported_in_percentage_points(self):
        m = im.weighted_ltv(frame(many(3, ltv=0.32)), frame(many(3, ltv=0.30)))
        self.assertAlmostEqual(m["change_pp"], 2.0, places=2)

    def test_states_the_direction_convention(self):
        m = im.weighted_ltv(frame(many(2)), None)
        self.assertIn("higher", m["methodology"]["direction"].lower())
        self.assertIn("deterioration", m["methodology"]["direction"].lower())


class TestBandMix(unittest.TestCase):
    def test_shares_are_of_balance_and_sum_to_100(self):
        f = frame([{"id": "A", "amount": 750, "ticket_band": ">=1m"},
                   {"id": "B", "amount": 250, "ticket_band": "<50k"}])
        m = im.band_mix(f, None, im.TICKET_BAND)
        shares = {r["band"]: r["current_share_pct"] for r in m["bands"]}
        self.assertEqual(shares[">=1m"], 75.0)
        self.assertAlmostEqual(sum(shares.values()), 100.0, places=2)

    def test_largest_shift_is_by_absolute_move(self):
        cur = frame([{"id": "A", "amount": 700, "ticket_band": ">=1m"},
                     {"id": "B", "amount": 300, "ticket_band": "<50k"}])
        pri = frame([{"id": "A", "amount": 200, "ticket_band": ">=1m"},
                     {"id": "B", "amount": 800, "ticket_band": "<50k"}])
        top = im.band_mix(cur, pri, im.TICKET_BAND)["largest_shift"]
        self.assertIn(top["band"], (">=1m", "<50k"))
        self.assertEqual(abs(top["share_change_pp"]), 50.0)

    def test_a_blank_band_becomes_an_explicit_unknown(self):
        f = frame([{"id": "A", "amount": 100, "ticket_band": ""}])
        bands = [r["band"] for r in im.band_mix(f, None, im.TICKET_BAND)["bands"]]
        self.assertIn("Unknown", bands)

    def test_ordering_is_deterministic_on_ties(self):
        cur = frame([{"id": "A", "amount": 500, "ticket_band": "zzz"},
                     {"id": "B", "amount": 500, "ticket_band": "aaa"}])
        a = im.band_mix(cur, None, im.TICKET_BAND)["bands"]
        b = im.band_mix(cur.iloc[::-1], None, im.TICKET_BAND)["bands"]
        self.assertEqual([r["band"] for r in a], [r["band"] for r in b])


# =========================================================================== #
# Generators — materiality
# =========================================================================== #
class TestMateriality(unittest.TestCase):
    def _detail(self, change_pct: float, value: float = 1_000_000):
        change = value * change_pct / 100
        return {"available": True,
                "headline_metric": {"label": "Pipeline balance", "value": value,
                                    "change": change, "change_pct": change_pct},
                "counts": {"current": 100, "comparison": 95, "change": 5},
                "contributors": {"brokers": [{"name": "Air", "amount": change,
                                              "share_of_change_pct": 100.0,
                                              "case_count": 3}], "regions": []},
                "components": {}, "methodology": {}, "source_dates": {}}

    def test_an_immaterial_pipeline_move_is_suppressed_with_a_reason(self):
        ins, omit = gen.pipeline_movement(CTX, self._detail(0.5))
        self.assertEqual(ins, [])
        self.assertEqual(omit[0].category, "immaterial")
        self.assertIn("below", omit[0].reason)

    def test_a_material_pipeline_move_is_reported(self):
        ins, omit = gen.pipeline_movement(CTX, self._detail(4.8))
        self.assertEqual(len(ins), 1)
        self.assertEqual(omit, [])
        self.assertIn("4.8%", ins[0].headline)
        self.assertIn("Air", ins[0].summary)

    def test_a_missing_prior_week_is_unavailable_not_immaterial(self):
        ins, omit = gen.pipeline_movement(
            CTX, {"available": False, "reason": "This is the earliest week."})
        self.assertEqual(ins, [])
        self.assertEqual(omit[0].category, "unavailable")

    def test_materiality_scales_with_the_portfolio(self):
        """The same absolute move qualifies on a small book, not on a large one."""
        small = gen.pipeline_movement(CTX, self._detail(10.0, value=1_000_000))[0]
        large = gen.pipeline_movement(CTX, self._detail(0.2, value=5_000_000_000))[0]
        self.assertEqual(len(small), 1)
        self.assertEqual(large, [])

    def test_ticket_size_needs_enough_cases_to_be_stable(self):
        m = im.ticket_size(frame(many(3, amount=200)), frame(many(3, amount=100)))
        ins, omit = gen.ticket_size(CTX, m)
        self.assertEqual(ins, [])
        self.assertIn("not stable", omit[0].reason)

    def test_a_mean_only_move_is_described_as_a_tail_effect(self):
        prior = frame([{"id": f"C{i}", "amount": 100} for i in range(30)])
        current = frame([{"id": f"C{i}", "amount": 100} for i in range(30)]
                        + [{"id": "BIG", "amount": 1_000_000}])
        ins, _ = gen.ticket_size(CTX, im.ticket_size(current, prior))
        self.assertEqual(len(ins), 1)
        self.assertIn("larger cases", ins[0].summary)

    def test_weighted_ltv_is_suppressed_below_the_coverage_floor(self):
        f = frame([{"id": "A", "amount": 900, "ltv": None},
                   {"id": "B", "amount": 100, "ltv": 0.9}])
        ins, omit = gen.weighted_ltv(CTX, im.weighted_ltv(f, frame(many(2, ltv=0.2))))
        self.assertEqual(ins, [])
        self.assertEqual(omit[0].category, "unavailable")
        self.assertIn("coverage", omit[0].reason.lower() + " coverage")

    def test_a_rising_weighted_ltv_is_attention_not_info(self):
        m = im.weighted_ltv(frame(many(5, ltv=0.35)), frame(many(5, ltv=0.30)))
        ins, _ = gen.weighted_ltv(CTX, m)
        self.assertEqual(ins[0].severity, SEVERITY_ATTENTION)
        self.assertIn("deterioration", ins[0].summary)

    def test_a_falling_weighted_ltv_is_info(self):
        m = im.weighted_ltv(frame(many(5, ltv=0.25)), frame(many(5, ltv=0.30)))
        ins, _ = gen.weighted_ltv(CTX, m)
        self.assertEqual(ins[0].severity, SEVERITY_INFO)
        self.assertIn("improvement", ins[0].summary)

    def test_mix_shift_reports_only_the_largest_band(self):
        cur = frame([{"id": "A", "amount": 700, "ltv_band": "70-80%"},
                     {"id": "B", "amount": 300, "ltv_band": "20-30%"}])
        pri = frame([{"id": "A", "amount": 200, "ltv_band": "70-80%"},
                     {"id": "B", "amount": 800, "ltv_band": "20-30%"}])
        ins, _ = gen.ltv_mix(CTX, im.band_mix(cur, pri, im.LTV_BAND))
        self.assertEqual(len(ins), 1)
        self.assertIn("no cause is inferred", ins[0].summary.lower())

    def test_completions_uses_a_higher_gate_than_pipeline(self):
        d = self._detail(5.0)
        self.assertEqual(gen.completions_movement(CTX, d)[0], [])
        self.assertEqual(len(gen.pipeline_movement(CTX, d)[0]), 1)

    def test_completions_never_claims_funded_growth(self):
        ins, _ = gen.completions_movement(CTX, self._detail(25.0))
        self.assertIn("not funded balance growth", ins[0].summary)
        self.assertIn("completed stage", ins[0].headline.lower())


class TestConversionGenerator(unittest.TestCase):
    BASE = {"basis": "avg_weekly_flow_over_lagged_kfi_stock", "weeklyRateValue": 4.3,
            "weeklyRateCount": 4.1, "avgWeeklyFlowValue": 4_100_000,
            "kfiStockValue": 96_000_000, "kfiStockCount": 512,
            "avgWeeklyFlowCount": 21, "lagWeeks": 6, "lagApplied": True,
            "denominatorWeek": "2026-06-26", "weeksInWindow": 5, "minWeeks": 4,
            "sufficient": True}

    def test_disabling_conversion_omits_it_explicitly(self):
        """Off must be stated, not silent — and it is what lets the route skip
        resolving the funnel, which is most of the brief's cold cost."""
        import os, tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write("insights:\n  conversion:\n    enabled: false\n")
            os.environ[cfg.PATH_ENV] = fh.name
        cfg.reset_cache()
        try:
            ins, omit = gen.conversion(CTX, {**self.BASE, "sufficient": False})
            self.assertEqual(ins, [])
            self.assertIn("disabled by configuration", omit[0].reason)
        finally:
            os.environ.pop(cfg.PATH_ENV, None)
            cfg.reset_cache()

    def test_a_healthy_sufficient_conversion_is_suppressed(self):
        ins, omit = gen.conversion(CTX, self.BASE)
        self.assertEqual(ins, [])
        self.assertIn("no comparison rate", omit[0].reason.lower())

    def test_an_insufficient_window_is_reported(self):
        ins, _ = gen.conversion(CTX, {**self.BASE, "sufficient": False,
                                      "weeksInWindow": 2})
        self.assertEqual(len(ins), 1)
        self.assertEqual(ins[0].severity, SEVERITY_ATTENTION)
        self.assertIn("provisional", ins[0].headline.lower())

    def test_it_never_manufactures_a_prior_period_movement(self):
        ins, _ = gen.conversion(CTX, {**self.BASE, "sufficient": False})
        blob = str(ins[0].to_dict()).lower()
        self.assertNotIn("percentage point", blob.replace("percentage-point", ""))
        self.assertIn("no prior-period conversion rate is published",
                      ins[0].methodology["no_comparison_rate"].lower())

    def test_it_states_that_attribution_is_not_valid(self):
        ins, _ = gen.conversion(CTX, {**self.BASE, "sufficient": False})
        self.assertIn("not valid for conversion",
                      ins[0].methodology["no_attribution"].lower())
        self.assertEqual(ins[0].contributors, {})


class TestConcentrationGenerator(unittest.TestCase):
    """`util` / `expected_util` / `fullPipeline.utilization` are all ALREADY
    percentage-of-limit POINTS — exactly what
    concentration_tests.evaluation._utilization returns (value/threshold*100)
    and exactly what the Risk Limits table displays. 93 means 93%, not 0.93.
    `headroom` is in the metric's OWN unit (percentage points here, since this
    fixture's test is unit="percent") — also points, never a fraction.

    CHANGED DELIBERATELY, together with insight_generators.concentration. The
    fixture used to encode fractions (0.93 for "93%"), which is what the
    defect this pins actually was: the generator multiplied an
    already-percentage value by 100 a second time, printing "93%" as "9300%"
    on the live book at 4340% and 4039%. See
    test_weekly_brief_utilisation_matches_risk_limits.py for the reproduction
    against the real evaluator.
    """

    def _snap(self, util, expected_util=None, expected_breach=False,
              states=True):
        return {
            "available": True, "reportingDate": "2026-07-31",
            "forecast": {"methodology": "completion_trend_model",
                         "currentSnapshot": "2026-08-07"},
            "states": {"available": states},
            "tests": [{
                "testId": "t1", "displayName": "Largest region share",
                "currentValue": 31.0, "threshold": 35.0, "utilization": util,
                "unit": "percent", "headroom": 4.0, "status": "warning",
                "expected": {"value": 36.0, "utilization": expected_util,
                             "headroom": -1.0, "status": "breach"},
                "fullPipeline": {"value": 41.0, "utilization": 117.0,
                                 "status": "breach"},
                "expectedBreach": expected_breach,
                "expectedBreachHorizon": {"available": True, "period": "2026-10"},
            }],
        }

    def test_a_comfortable_test_is_suppressed(self):
        ins, omit = gen.concentration(CTX, self._snap(50.0))
        self.assertEqual(ins, [])
        self.assertEqual(omit[0].category, "immaterial")

    def test_a_test_near_its_limit_is_reported(self):
        ins, _ = gen.concentration(CTX, self._snap(93.0))
        self.assertEqual(len(ins), 1)
        self.assertIn("93%", ins[0].headline)
        self.assertEqual(ins[0].severity, SEVERITY_ATTENTION)

    def test_an_expected_breach_is_a_concern(self):
        ins, _ = gen.concentration(CTX, self._snap(93.0, 103.0, expected_breach=True))
        self.assertEqual(ins[0].severity, SEVERITY_CONCERN)
        self.assertIn("2026-10", ins[0].summary)

    def test_the_forecast_reaching_the_threshold_alone_qualifies(self):
        """A comfortable funded position with a breaching forecast still matters."""
        ins, _ = gen.concentration(CTX, self._snap(40.0, 110.0, expected_breach=True))
        self.assertEqual(len(ins), 1)

    def test_the_stress_scenario_is_labelled_hypothetical(self):
        ins, _ = gen.concentration(CTX, self._snap(93.0, 103.0))
        self.assertIn("hypothetical", ins[0].summary.lower())
        self.assertIn("not a forecast",
                      ins[0].methodology["all_pipeline_basis"].lower())

    def test_the_funded_date_is_separate_from_the_pipeline_date(self):
        ins, _ = gen.concentration(CTX, self._snap(93.0))
        self.assertEqual(ins[0].source_dates["funded_as_of"], "2026-07-31")
        self.assertEqual(ins[0].source_dates["pipeline_as_of"], "2026-08-07")

    def test_it_does_not_recompute_the_test(self):
        ins, _ = gen.concentration(CTX, self._snap(93.0))
        self.assertIn("not recomputed", ins[0].methodology["source"])
        self.assertEqual(ins[0].metrics["current_value"], 31.0)

    def test_the_headroom_is_points_not_a_rescaled_fraction(self):
        """THE OTHER HALF OF THE DEFECT. headroom=4.0 (4.00pp) was printed as
        pct(4.0 * 100, 2) + a literal "pp" — "400.00%pp" — a percent sign AND a
        points suffix on the same number."""
        ins, _ = gen.concentration(CTX, self._snap(93.0))
        self.assertIn("headroom 4.00pp", ins[0].summary)
        self.assertNotIn("%pp", ins[0].summary)

    def test_a_currency_unit_headroom_is_money_not_a_percent(self):
        """The reported defect's second example: Average principal balance is
        a currency-unit test, so its headroom is a balance, not a percentage —
        £178,831 must never be run through pct()."""
        snap = self._snap(40.0)
        snap["tests"][0]["unit"] = "currency"
        snap["tests"][0]["headroom"] = 178_831.0
        snap["tests"][0]["displayName"] = "Average principal balance"
        ins, _ = gen.concentration(CTX, snap)
        self.assertEqual(ins, [])  # 40% utilisation is comfortable, correctly
        # Force it into range to see the headroom text.
        snap["tests"][0]["utilization"] = 93.0
        ins, _ = gen.concentration(CTX, snap)
        self.assertIn("£179k", ins[0].summary)
        self.assertNotIn("%", ins[0].summary.split("headroom")[1].split(",")[0])


class TestDataQualityGenerator(unittest.TestCase):
    CLEAN = {"case_count": 100, "broker_reassignment_pct": 0.0,
             "region_reassignment_pct": 0.0, "unusable_case_id_count": 0,
             "unusable_case_id_balance": 0.0, "unusable_case_id_pct": 0.0,
             "duplicate_case_id_count": 0, "duplicate_case_id_pct": 0.0,
             "unknown_broker_share_pct": 0.0, "unknown_region_share_pct": 0.0,
             "ltv_coverage_pct": 100.0, "methodology": {}}

    def test_a_clean_week_is_silent_by_default(self):
        ins, omit = gen.data_quality(CTX, self.CLEAN)
        self.assertEqual(ins, [])
        self.assertEqual(omit[0].category, "immaterial")

    def test_unstable_dimensions_are_a_concern_and_flag_attribution_risk(self):
        ins, _ = gen.data_quality(CTX, {**self.CLEAN,
                                        "broker_reassignment_pct": 92.3})
        self.assertEqual(ins[0].severity, SEVERITY_CONCERN)
        self.assertIn("attribution", ins[0].summary.lower())
        self.assertTrue(ins[0].data_quality["attribution_risk"])

    def test_an_unknown_share_breach_is_attention_not_concern(self):
        ins, _ = gen.data_quality(CTX, {**self.CLEAN,
                                        "unknown_broker_share_pct": 12.0})
        self.assertEqual(ins[0].severity, SEVERITY_ATTENTION)
        self.assertFalse(ins[0].data_quality["attribution_risk"])

    def test_it_can_be_configured_to_speak_when_clean(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write("insights:\n  data_quality:\n    emit_when_clean: true\n")
            path = fh.name
        import os
        os.environ[cfg.PATH_ENV] = path
        cfg.reset_cache()
        try:
            ins, _ = gen.data_quality(CTX, self.CLEAN)
            self.assertEqual(len(ins), 1)
            self.assertIn("No material data-quality issues", ins[0].headline)
        finally:
            os.environ.pop(cfg.PATH_ENV, None)
            cfg.reset_cache()


# =========================================================================== #
# Selection
# =========================================================================== #
def _ins(t: str, sev: str = SEVERITY_INFO, disc: str = "") -> Insight:
    return Insight(insight_type=t, tenant_id="ERE", portfolio_id="ERE",
                   portfolio_context="total", as_of_date="2026-08-07",
                   comparison_date="2026-07-31", headline="h", summary="s",
                   severity=sev, discriminator=disc)


class TestSelection(unittest.TestCase):
    def test_severity_outranks_type(self):
        out, _ = eng.select([_ins(PIPELINE_MOVEMENT, SEVERITY_INFO),
                             _ins(TICKET_SIZE, SEVERITY_CONCERN)])
        self.assertEqual(out[0].insight_type, TICKET_SIZE)

    def test_concentration_and_data_quality_lead_at_equal_severity(self):
        out, _ = eng.select([_ins(TICKET_SIZE), _ins(PIPELINE_MOVEMENT),
                             _ins(DATA_QUALITY), _ins(CONCENTRATION_PROXIMITY)])
        self.assertEqual([i.insight_type for i in out][:2],
                         [CONCENTRATION_PROXIMITY, DATA_QUALITY])

    def test_the_brief_is_capped_at_eight(self):
        out, omit = eng.select(
            [_ins(CONCENTRATION_PROXIMITY, disc=f"t{i}") for i in range(20)])
        self.assertLessEqual(len(out), 8)
        self.assertTrue(omit)

    def test_per_type_caps_are_applied(self):
        out, omit = eng.select(
            [_ins(CONCENTRATION_PROXIMITY, disc=f"t{i}") for i in range(5)]
            + [_ins(TICKET_SIZE, disc=f"b{i}") for i in range(3)])
        self.assertEqual(sum(1 for i in out
                             if i.insight_type == CONCENTRATION_PROXIMITY), 2)
        self.assertEqual(sum(1 for i in out if i.insight_type == TICKET_SIZE), 1)
        self.assertTrue(any(o.category == "capped" for o in omit))

    def test_anything_dropped_by_a_cap_is_reported_not_silent(self):
        _out, omit = eng.select(
            [_ins(CONCENTRATION_PROXIMITY, disc=f"t{i}") for i in range(5)])
        capped = [o for o in omit if o.category == "capped"]
        self.assertEqual(len(capped), 1)
        self.assertIn("not shown", capped[0].reason)

    def test_ordering_is_stable_across_input_order(self):
        items = [_ins(TICKET_SIZE), _ins(CONCENTRATION_PROXIMITY, disc="a"),
                 _ins(DATA_QUALITY), _ins(LTV_MIX_SHIFT)]
        a, _ = eng.select(list(items))
        b, _ = eng.select(list(reversed(items)))
        self.assertEqual([i.insight_type for i in a], [i.insight_type for i in b])

    def test_priority_descends_with_position(self):
        out, _ = eng.select([_ins(CONCENTRATION_PROXIMITY, disc="a"),
                             _ins(TICKET_SIZE)])
        self.assertGreater(out[0].priority, out[1].priority)


class TestInsightIdentity(unittest.TestCase):
    def test_the_id_is_deterministic(self):
        self.assertEqual(_ins(TICKET_SIZE).insight_id, _ins(TICKET_SIZE).insight_id)

    def test_the_id_separates_tenants_scopes_types_dates_and_discriminators(self):
        base = _ins(TICKET_SIZE)
        other = _ins(TICKET_SIZE); other.tenant_id = "OTHER"
        scope = _ins(TICKET_SIZE); scope.portfolio_context = "acquired_001"
        date = _ins(TICKET_SIZE); date.as_of_date = "2026-08-14"
        for variant in (other, scope, date, _ins(WEIGHTED_LTV),
                        _ins(TICKET_SIZE, disc="x")):
            self.assertNotEqual(base.insight_id, variant.insight_id)

    def test_the_id_ignores_the_values_so_a_correction_updates_not_duplicates(self):
        a, b = _ins(TICKET_SIZE), _ins(TICKET_SIZE)
        a.metrics = {"current_average": 1}
        b.metrics = {"current_average": 999}
        self.assertEqual(a.insight_id, b.insight_id)

    def test_notification_eligibility_follows_severity(self):
        self.assertFalse(_ins(TICKET_SIZE, SEVERITY_INFO).notification_eligible)
        self.assertTrue(_ins(TICKET_SIZE, SEVERITY_ATTENTION).notification_eligible)
        self.assertTrue(_ins(DATA_QUALITY, SEVERITY_CONCERN).notification_eligible)

    def test_no_case_level_identifier_is_carried(self):
        ins = _ins(PIPELINE_MOVEMENT)
        ins.contributors = {"brokers": [{"name": "Air", "amount": 1,
                                         "case_count": 2}]}
        self.assertNotIn("case_id", str(ins.to_dict()))


# =========================================================================== #
# Configuration
# =========================================================================== #
class TestConfiguration(unittest.TestCase):
    def tearDown(self):
        import os
        os.environ.pop(cfg.PATH_ENV, None)
        cfg.reset_cache()

    def test_the_shipped_file_loads(self):
        cfg.reset_cache()
        conf = cfg.load()
        self.assertTrue(conf["source"].endswith("insights.yaml"))
        self.assertEqual(conf["insights"]["pipeline_movement"]["min_change_pct"], 2.0)

    def test_a_missing_file_falls_back_to_defaults_rather_than_failing(self):
        import os
        os.environ[cfg.PATH_ENV] = "/nonexistent/insights.yaml"
        cfg.reset_cache()
        conf = cfg.load()
        self.assertEqual(conf["source"], "defaults")
        self.assertEqual(conf["insights"]["completions"]["min_change_pct"], 10.0)

    def test_a_partial_override_keeps_the_other_defaults(self):
        import os, tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write("insights:\n  completions:\n    min_change_pct: 1.0\n")
            os.environ[cfg.PATH_ENV] = fh.name
        cfg.reset_cache()
        conf = cfg.load()
        self.assertEqual(conf["insights"]["completions"]["min_change_pct"], 1.0)
        self.assertEqual(conf["insights"]["pipeline_movement"]["min_change_pct"], 2.0)

    def test_no_threshold_is_hard_coded_in_the_generators(self):
        """A literal here would be a portfolio assumption baked into code."""
        src = (_REPO_ROOT / "mi_agent_api" / "insight_generators.py").read_text()
        for literal in ("5000000", "5_000_000", "£5m"):
            self.assertNotIn(literal, src)


# =========================================================================== #
# Engine-level behaviour
# =========================================================================== #
class TestEngineOnRealData(unittest.TestCase):
    """End-to-end against the committed governed pack."""

    FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack" / "pipeline"

    @classmethod
    def setUpClass(cls):
        if not cls.FIXTURE.exists():
            raise unittest.SkipTest("fixture pack not present")

    def _build(self, **kw):
        return eng.build(str(self.FIXTURE), "client_001", tenant_id="client_001", **kw)

    def test_a_brief_is_produced_and_every_type_is_accounted_for(self):
        brief = self._build()
        self.assertEqual(brief["status"], "success")
        self.assertLessEqual(brief["insight_count"], 8)
        # Nothing may vanish: every type either produced an insight or said why.
        accounted = ({i["insight_type"] for i in brief["insights"]}
                     | {o["insight_type"] for o in brief["omitted"]})
        for t in ("PIPELINE_MOVEMENT", "TICKET_SIZE", "WEIGHTED_LTV",
                  "TICKET_MIX_SHIFT", "LTV_MIX_SHIFT", "COMPLETIONS_MOVEMENT",
                  "CONVERSION_CONTEXT", "CONCENTRATION_PROXIMITY", "DATA_QUALITY"):
            self.assertIn(t, accounted, f"{t} silently disappeared from the brief")

    def test_the_same_week_produces_the_same_brief(self):
        a, b = self._build(), self._build()
        self.assertEqual([i["insight_id"] for i in a["insights"]],
                         [i["insight_id"] for i in b["insights"]])
        self.assertEqual(a["insights"], b["insights"])

    def test_one_failing_generator_yields_a_partial_brief_not_an_empty_one(self):
        real = gen.ticket_size
        gen.ticket_size = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            brief = self._build()
        finally:
            gen.ticket_size = real
        self.assertEqual(brief["status"], "partial")
        self.assertIn("could not be produced", brief["reason"])
        errored = [o for o in brief["omitted"] if o["category"] == "error"]
        self.assertEqual([o["insight_type"] for o in errored], ["TICKET_SIZE"])
        # Everything else still ran.
        self.assertIn("PIPELINE_MOVEMENT",
                      {o["insight_type"] for o in brief["omitted"]}
                      | {i["insight_type"] for i in brief["insights"]})

    def test_a_missing_source_is_a_controlled_unavailable_brief(self):
        brief = eng.build("/nonexistent/root", "client_001", tenant_id="client_001")
        self.assertEqual(brief["status"], "unavailable")
        self.assertEqual(brief["insights"], [])
        self.assertIn("No governed weekly pipeline extract", brief["reason"])

    def test_the_earliest_week_reports_no_comparison_rather_than_inventing_one(self):
        from mi_agent_api import pipeline_contract as pc
        dates = sorted(e["pipeline_extract_date"] for e
                       in pc.weekly_extract_inventory(str(self.FIXTURE),
                                                      "client_001")["extracts"]
                       if e.get("pipeline_extract_date"))
        brief = self._build(as_of=dates[0])
        self.assertIsNone(brief["comparison_date"])
        reasons = " ".join(o["reason"] for o in brief["omitted"]).lower()
        self.assertIn("prior week", reasons)

    def test_scope_is_carried_into_every_insight_and_its_id(self):
        total = self._build(scope="total")
        acquired = self._build(scope="acquired_001")
        self.assertEqual(acquired["portfolio_context"], "acquired_001")
        for i in acquired["insights"]:
            self.assertEqual(i["portfolio_context"], "acquired_001")
        ids_t = {i["insight_id"] for i in total["insights"]}
        ids_a = {i["insight_id"] for i in acquired["insights"]}
        self.assertFalse(ids_t & ids_a, "scopes must not share insight ids")

    def test_dates_are_explicit_and_never_conflated(self):
        sd = self._build()["source_dates"]
        self.assertIsNotNone(sd["pipeline_as_of"])
        self.assertNotEqual(sd["pipeline_as_of"], sd["pipeline_comparison"])


if __name__ == "__main__":
    unittest.main()
