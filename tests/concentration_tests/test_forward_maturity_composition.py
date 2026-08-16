"""Case-level composition survives probability assignment, and the maturity
correction propagates through it into forward concentration.

Two things are proved here, and they are different claims:

* **Composition.** Two pipelines with identical total amount, case count, stage
  composition and stage probabilities, differing ONLY in where the exposure
  sits in the constrained dimension, must produce different expected
  concentrations. If the pipeline were aggregated before probability weighting
  they would produce the same number, and forward concentration would be blind
  to exactly the thing it exists to measure.

* **Propagation.** A maturity-corrected stage probability reaches the forward
  concentration calculation through the retained case attributes. The fixture is
  deliberately built so the constrained dimension (London) is early-stage-heavy,
  so the corrected probability RAISES expected London concentration. That
  direction is a property of this fixture, not a general claim about production.
"""

from __future__ import annotations

import datetime as _dt
from typing import List

import pandas as pd
import pytest

from mi_agent.concentration_tests import forward
from mi_agent.concentration_tests.evaluation import evaluate_active_tests
from mi_agent.concentration_tests.metrics import evaluate_metric
from mi_agent.concentration_tests.models import ActiveConfiguration, ActiveTest
from mi_agent_api.pipeline_history import (
    MATURITY_FLAG_ENV, build_historical_completion_model,
)
from mi_agent_api.pipeline_prep import prepare_pipeline_mi_dataset

M = 1_000_000.0
LONDON_LIMIT_PCT = 25.0
START = _dt.date(2025, 1, 6)


def _week(i: int) -> str:
    return (START + _dt.timedelta(days=i * 7)).isoformat()


# --------------------------------------------------------------------------- #
# Funded book: 20m London of 100m — 20%, comfortably inside a 25% limit.
# --------------------------------------------------------------------------- #
@pytest.fixture()
def funded() -> pd.DataFrame:
    return pd.DataFrame({
        "loan_id": [f"L{i}" for i in range(10)],
        "current_outstanding_balance": [10 * M] * 10,
        "collateral_geography": ["London"] * 2 + ["Wales"] * 4 + ["Scotland"] * 4,
    })


def _prepared(rows: List[dict], model) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    prepared, _ = prepare_pipeline_mi_dataset(
        raw, as_of_date="2025-08-04", historical_model=model)
    return prepared


def _case(idx: int, stage: str, region: str) -> dict:
    return {"Account Number": f"P{idx}", "Status": stage,
            "Loan Amount": 5 * M, "Property Region": region}


#: Concentrated: every KFI case is London. 30m total, 6 cases, 4 KFI + 2 OFFER.
CONCENTRATED = [_case(i, "KFI", "London") for i in range(4)] + \
               [_case(i + 4, "Offer", "Wales") for i in range(2)]

#: Diversified: identical totals, count and stage composition; London holds one
#: KFI case instead of four.
DIVERSIFIED = [_case(0, "KFI", "London"), _case(1, "KFI", "Wales"),
               _case(2, "KFI", "Scotland"), _case(3, "KFI", "North East"),
               _case(4, "Offer", "Wales"), _case(5, "Offer", "Scotland")]


def london_test() -> ActiveConfiguration:
    return ActiveConfiguration(client_id="c", version=1, tests=[ActiveTest(
        metric_id="geo_region_share", threshold=LONDON_LIMIT_PCT,
        operator="max", display_name="London concentration",
        parameters={"regions": ["London"]})])


def expected_london_share(lib, funded_df, pipeline_df) -> float:
    frame, _meta = forward.build_state_frame(funded_df, pipeline_df,
                                             forward.STATE_EXPECTED)
    comp = evaluate_metric(frame, lib, lib.get("geo_region_share"),
                           {"regions": ["London"]})
    return comp.value


# --------------------------------------------------------------------------- #
# Composition — the pipelines are identical in every respect but placement
# --------------------------------------------------------------------------- #
class TestCompositionSurvivesProbabilityAssignment:
    def test_the_two_pipelines_are_genuinely_equivalent_apart_from_placement(self):
        a, b = _prepared(CONCENTRATED, None), _prepared(DIVERSIFIED, None)
        assert a["current_outstanding_balance"].sum() == \
            b["current_outstanding_balance"].sum() == 30 * M
        assert len(a) == len(b) == 6
        assert sorted(a["pipeline_stage"]) == sorted(b["pipeline_stage"])
        # Same probability multiset — probability depends on stage alone.
        assert sorted(a["completion_probability"]) == \
            sorted(b["completion_probability"])

    def test_concentrated_and_diversified_pipelines_differ(self, lib, funded):
        conc = expected_london_share(lib, funded, _prepared(CONCENTRATED, None))
        div = expected_london_share(lib, funded, _prepared(DIVERSIFIED, None))
        assert conc != pytest.approx(div)
        # Funded London is 20%. The concentrated pipeline pushes toward the
        # limit; the diversified one dilutes away from it.
        assert conc > 20.0 > div

    def test_only_constrained_dimension_cases_enter_the_numerator(self, lib,
                                                                  funded):
        frame, _ = forward.build_state_frame(funded, _prepared(CONCENTRATED, None),
                                             forward.STATE_EXPECTED)
        comp = evaluate_metric(frame, lib, lib.get("geo_region_share"),
                               {"regions": ["London"]})
        # 20m funded London + 4 x 5m x 0.20 expected London pipeline.
        assert comp.numerator_value == pytest.approx(20 * M + 4 * M)
        # Denominator is funded plus the probability-weighted ELIGIBLE pipeline
        # (London KFI at 0.20 and Wales OFFER at 0.75), not the gross pipeline.
        assert comp.denominator_value == pytest.approx(
            100 * M + 4 * M + 7.5 * M)

    def test_expected_state_stays_distinct_from_full_pipeline(self, lib, funded):
        pipeline = _prepared(CONCENTRATED, None)
        expected = expected_london_share(lib, funded, pipeline)
        full_frame, _ = forward.build_state_frame(funded, pipeline,
                                                  forward.STATE_FULL)
        full = evaluate_metric(full_frame, lib, lib.get("geo_region_share"),
                               {"regions": ["London"]}).value
        assert expected != pytest.approx(full)
        # Full pipeline takes every London case at 100%, so it must be higher.
        assert full > expected


# --------------------------------------------------------------------------- #
# Propagation — a corrected KFI rate reaches the concentration number
# --------------------------------------------------------------------------- #
def _kfi_history(tmp_path) -> list:
    """30 weeks of KFI entrants, true conversion 0.6, four-week completion lag.

    KFI's configured maturity window is 90 days, so entrants in the final ~13
    weeks are immature. The pooled estimator counts them as failures and lands
    near 0.52; the matured estimator recovers 0.6.
    """
    weeks, per_week, completing, lag = 30, 5, 3, 4
    entries = []
    for w in range(weeks):
        rows = []
        for e in range(weeks):
            if e > w:
                continue
            for i in range(per_week):
                done = i < completing and w >= e + lag
                rows.append({
                    "Account Number": f"H{e:03d}_{i}",
                    "Status": "Completed" if done else "KFI",
                    "Date Funds Released": _week(e + lag) if done else "",
                    "Loan Amount": 5 * M,
                    "Property Region": "London",
                })
        path = tmp_path / f"KFI_Pipeline_{_week(w).replace('-', '_')}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        entries.append({"source_file": str(path), "pipeline_extract_date": _week(w)})
    return entries


class TestMaturityCorrectionPropagates:
    def test_corrected_kfi_probability_raises_expected_london_concentration(
            self, lib, funded, tmp_path, monkeypatch):
        entries = _kfi_history(tmp_path)

        monkeypatch.delenv(MATURITY_FLAG_ENV, raising=False)
        model_off = build_historical_completion_model(entries)
        monkeypatch.setenv(MATURITY_FLAG_ENV, "1")
        model_on = build_historical_completion_model(entries)

        kfi_off = model_off["stage_rates"]["KFI"]
        kfi_on = model_on["stage_rates"]["KFI"]
        # The fixture's whole point: the corrected rate is materially higher.
        assert kfi_on > kfi_off
        assert kfi_on == pytest.approx(0.6, abs=1e-4)

        # The probability must actually land on the cases...
        pipeline_off = _prepared(CONCENTRATED, model_off)
        pipeline_on = _prepared(CONCENTRATED, model_on)
        london_on = pipeline_on[pipeline_on["collateral_geography"] == "London"]
        assert london_on["completion_probability"].tolist() == pytest.approx(
            [kfi_on] * len(london_on))

        # ...and carry through to the forward concentration number.
        share_off = expected_london_share(lib, funded, pipeline_off)
        share_on = expected_london_share(lib, funded, pipeline_on)
        assert share_on > share_off

    def test_corrected_mode_moves_the_test_materially_toward_its_limit(
            self, lib, funded, tmp_path, monkeypatch):
        """End-to-end through the governed evaluator, not just the metric."""
        entries = _kfi_history(tmp_path)
        monkeypatch.setenv(MATURITY_FLAG_ENV, "1")
        model_on = build_historical_completion_model(entries)
        pipeline = _prepared(CONCENTRATED, model_on)

        config = london_test()
        evaluated = evaluate_active_tests(funded, None, config, lib,
                                          reporting_date="2025-08-04",
                                          prior_reporting_date="")
        evaluated = forward.extend_with_forward_states(
            evaluated, config, lib, funded, pipeline)

        row = evaluated["tests"][0]
        assert row["currentValue"] == pytest.approx(20.0)      # funded London
        assert row["expected"] is not None
        assert row["expected"]["value"] > row["currentValue"]
        # Still inside the limit here — the assertion is direction of travel,
        # not a manufactured breach.
        assert row["expected"]["utilization"] > row["utilization"]
        assert row["forecastTreatment"] == forward.TREATMENT_EXPOSURE
