"""The single evaluation service: statuses, headroom, period movement, honest
unavailability, and drill-through that reconciles exactly."""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent.concentration_tests.evaluation import (
    drillthrough,
    evaluate_active_tests,
)
from mi_agent.concentration_tests.models import (
    ActiveConfiguration,
    ActiveTest,
    ApprovalRecord,
    SourceEvidence,
)


def active(metric_id, threshold, operator="max", params=None, **kw):
    return ActiveTest(
        metric_id=metric_id, threshold=threshold, operator=operator,
        display_name=kw.pop("display_name", metric_id),
        parameters=params or {},
        evidence=SourceEvidence(source_reference="covenant.txt",
                                source_text="clause"),
        approval=ApprovalRecord(decision="approved", operator="Alice",
                                decided_at="2026-01-10T09:00:00+00:00"),
        **kw)


def config(*tests):
    return ActiveConfiguration(client_id="client_a", version=3,
                               tests=list(tests), activated_by="Alice",
                               library_version="1.0.0")


class TestStatuses:
    def test_pass_warning_breach(self, lib, funded_frame):
        cfg = config(
            active("geo_region_share", 50.0,
                   params={"regions": ["East Anglia"]}),       # 35% → pass
            active("geo_region_share", 36.5,
                   params={"regions": ["London", "South East"]}),  # 36% → warning
            active("geo_region_share", 30.0,
                   params={"regions": ["East Anglia"]}),       # 35% → breach
        )
        out = evaluate_active_tests(funded_frame, None, cfg, lib,
                                    reporting_date="2025-11-30")
        statuses = [t["status"] for t in out["tests"]]
        assert statuses == ["pass", "warning", "breach"]
        breach = out["tests"][2]
        assert breach["breachAmount"] == 5.0
        assert breach["headroom"] == -5.0
        assert breach["utilization"] == pytest.approx(116.67, abs=0.01)
        assert out["summary"]["overallStatus"] == "breach"

    def test_exact_threshold_passes_a_max_test(self, lib, funded_frame):
        cfg = config(active("geo_region_share", 35.0,
                            params={"regions": ["East Anglia"]}))
        out = evaluate_active_tests(funded_frame, None, cfg, lib)
        # 35.0 vs limit 35.0 → not a breach, but inside the 90% warning band.
        assert out["tests"][0]["status"] == "warning"

    def test_warning_policy_is_configurable(self, lib, funded_frame):
        strict = config(active("geo_region_share", 40.0,
                               params={"regions": ["East Anglia"]},
                               warning_fraction=0.80))
        out = evaluate_active_tests(funded_frame, None, strict, lib)
        assert out["tests"][0]["status"] == "warning"  # 35 ≥ 40 × 0.8
        relaxed = config(active("geo_region_share", 40.0,
                                params={"regions": ["East Anglia"]},
                                warning_fraction=0.95))
        out = evaluate_active_tests(funded_frame, None, relaxed, lib)
        assert out["tests"][0]["status"] == "pass"

    def test_min_operator(self, lib, funded_frame):
        cfg = config(active("rate_gross_wac", 6.0, operator="min"))
        out = evaluate_active_tests(funded_frame, None, cfg, lib)
        assert out["tests"][0]["status"] == "breach"

    def test_zero_threshold_max(self, lib, funded_frame):
        cfg = config(active("borrower_age_share", 0.0,
                            params={"age": 55, "comparison": "below",
                                    "age_basis": "youngest"}))
        out = evaluate_active_tests(funded_frame, None, cfg, lib)
        assert out["tests"][0]["status"] == "breach"  # 15% of book under 55

    def test_missing_field_is_unavailable_not_pass(self, lib, funded_frame):
        df = funded_frame.drop(columns=["collateral_geography"])
        cfg = config(active("geo_region_share", 25.0,
                            params={"regions": ["East Anglia"]}))
        out = evaluate_active_tests(df, None, cfg, lib)
        t = out["tests"][0]
        assert t["status"] == "unavailable"
        assert t["currentValue"] is None
        assert t["missingFields"]
        assert out["summary"]["passes"] == 0

    def test_empty_frame_is_insufficient_data(self, lib):
        cfg = config(active("geo_region_share", 25.0,
                            params={"regions": ["East Anglia"]}))
        out = evaluate_active_tests(pd.DataFrame(), None, cfg, lib)
        assert out["tests"][0]["status"] == "insufficient_data"

    def test_pending_effective_date_and_expired(self, lib, funded_frame):
        cfg = config(
            active("geo_region_share", 25.0,
                   params={"regions": ["East Anglia"]},
                   effective_date="2026-06-30"),
            active("geo_region_share", 25.0,
                   params={"regions": ["East Anglia"]},
                   expiry_date="2024-01-01"),
        )
        out = evaluate_active_tests(funded_frame, None, cfg, lib,
                                    reporting_date="2025-11-30")
        assert out["tests"][0]["status"] == "pending_effective_date"
        assert out["tests"][1]["status"] == "expired"

    def test_external_index_unavailable_without_a_provider(self, lib,
                                                          funded_frame):
        cfg = config(active("index_hpi_ratio", 90.0, operator="min",
                            params={"index_source": "ONS",
                                    "index_series": "UK-all",
                                    "reference_date": "2024-01-31"}))
        out = evaluate_active_tests(funded_frame, None, cfg, lib)
        t = out["tests"][0]
        assert t["status"] == "unavailable"
        assert t["dataStatus"] == "external_reference_unconfigured"
        assert "not simulated" in t["notes"]


class TestPeriodChange:
    def test_movement_and_status_transition(self, lib, funded_frame):
        prior = funded_frame.copy()
        # Prior period: East Anglia was 250k of 900k (drop L4's 200k → add 100k)
        prior.loc[3, "current_outstanding_balance"] = 100_000
        cfg = config(active("geo_region_share", 30.0,
                            params={"regions": ["East Anglia"]}))
        out = evaluate_active_tests(funded_frame, prior, cfg, lib,
                                    reporting_date="2025-11-30",
                                    prior_reporting_date="2025-10-31")
        t = out["tests"][0]
        assert t["currentValue"] == 35.0
        assert t["priorValue"] == pytest.approx(27.78, abs=0.01)
        assert t["absoluteChange"] == pytest.approx(7.22, abs=0.01)
        assert t["percentagePointChange"] == t["absoluteChange"]
        assert t["priorStatus"] == "warning"  # 27.78 ≥ 30×0.9
        assert t["status"] == "breach"
        assert t["statusTransition"] == "warning -> breach"
        assert t["deteriorated"] is True
        assert out["summary"]["deteriorations"] == 1
        assert out["priorReportingDate"] == "2025-10-31"

    def test_missing_prior_period_is_explicitly_unavailable(self, lib,
                                                           funded_frame):
        cfg = config(active("geo_region_share", 50.0,
                            params={"regions": ["East Anglia"]}))
        out = evaluate_active_tests(funded_frame, None, cfg, lib,
                                    reporting_date="2025-11-30")
        t = out["tests"][0]
        assert t["priorAvailable"] is False
        assert t["priorValue"] is None
        assert t["absoluteChange"] is None
        assert out["priorAvailable"] is False


class TestDrillThrough:
    def test_population_reconciles_exactly(self, lib, funded_frame):
        test = active("geo_region_share", 25.0,
                      params={"regions": ["East Anglia"]})
        out = drillthrough(funded_frame, lib, test)
        assert out["available"] is True
        assert out["reconciles"] is True
        assert out["rowCount"] == 3
        assert out["loansInNumerator"] == 3
        total = sum(r["current_outstanding_balance"] for r in out["rows"])
        assert round(total, 2) == out["numeratorValue"] == 350_000
        assert out["denominatorValue"] == 1_000_000

    def test_denominator_reconciles_to_the_frame(self, lib, funded_frame):
        test = active("balance_above_share", 10.0,
                      params={"amount": 90_000, "balance_basis": "current"})
        out = drillthrough(funded_frame, lib, test)
        assert out["denominatorValue"] == round(
            float(funded_frame.current_outstanding_balance.sum()), 2)

    def test_no_data_is_a_controlled_state(self, lib):
        test = active("geo_region_share", 25.0,
                      params={"regions": ["East Anglia"]})
        out = drillthrough(pd.DataFrame(), lib, test)
        assert out["available"] is False
        assert out["rows"] == []
