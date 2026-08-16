"""Maturity correction in the standard pipeline conversion model.

The pooled estimator counts a case first seen at a stage in the final snapshot
as a failed conversion, because it has not completed. It has not been given the
chance. These tests pin the corrected estimator that admits a case only once it
has been observed for the configured stage-to-funding window, prove the
correction propagates through probability assignment into forward concentration,
and prove the existing forecast does not move while the flag is off.

Synthetic histories are written as weekly CSV extracts in ``tmp_path`` because
``build_historical_completion_model`` reads files — the same seam production
uses. Column headers are the real governed aliases (``Account Number`` ->
``pipeline_case_identifier``, ``Status`` -> ``pipeline_stage``), so the tests
exercise the actual column resolution rather than a hand-built frame.
"""

from __future__ import annotations

import datetime as _dt
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import pytest

from mi_agent_api.pipeline_history import (
    MATURITY_FLAG_ENV, MIN_OBSERVATIONS, RATE_CONFIG, RATE_MATURED,
    RATE_POOLED, build_historical_completion_model, historical_model_evidence,
)
from mi_agent_api.pipeline_prep import (
    _stage_days_to_fund, _stage_probabilities, prepare_pipeline_mi_dataset,
)

WEEK = 7
START = _dt.date(2025, 1, 6)

#: OFFER's configured maturity window is the shortest, so a test needs the
#: fewest weeks to build both a matured and an immature population.
STAGE = "OFFER"
STAGE_LABEL = "Offer"


# --------------------------------------------------------------------------- #
# Synthetic weekly history
# --------------------------------------------------------------------------- #
Case = Tuple[str, int, bool]  # (case_id, entry_week, ever_completes)


def _week_date(index: int) -> str:
    return (START + _dt.timedelta(days=index * WEEK)).isoformat()


def write_history(tmp_path, weeks: int, cases: Sequence[Case], *,
                  lag_weeks: int = 4, stage_label: str = STAGE_LABEL,
                  blank_identity: Sequence[str] = (),
                  region_by_case: Optional[dict] = None,
                  amount: float = 100_000.0) -> List[dict]:
    """One CSV per week; a case appears from its entry week onward.

    A completing case sits at ``stage_label`` until ``entry_week + lag_weeks``,
    from which snapshot it reads COMPLETED. A non-completing case stays at the
    stage for the rest of the window — it has not withdrawn, it simply has not
    converted, which is exactly the population the pooled estimator mishandles.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    entries: List[dict] = []
    for w in range(weeks):
        rows = []
        for case_id, entry, completes in cases:
            if entry > w:
                continue
            done = completes and w >= entry + lag_weeks
            rows.append({
                "Account Number": ("" if case_id in blank_identity else case_id),
                "KFI Number": ("" if case_id in blank_identity
                               else f"K{case_id}"),
                "Status": "Completed" if done else stage_label,
                "Date Funds Released": (_week_date(entry + lag_weeks)
                                        if done else ""),
                "Loan Amount": amount,
                "Property Region": (region_by_case or {}).get(case_id, "London"),
            })
        path = tmp_path / f"KFI_Pipeline_{_week_date(w).replace('-', '_')}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        entries.append({"source_file": str(path),
                        "pipeline_extract_date": _week_date(w)})
    return entries


def uniform_cases(weeks: int, per_week: int, completing: int) -> List[Case]:
    """``per_week`` entrants every week, ``completing`` of which convert."""
    out: List[Case] = []
    for w in range(weeks):
        for i in range(per_week):
            out.append((f"C{w:03d}_{i}", w, i < completing))
    return out


def stage_block(model, stage: str = STAGE) -> dict:
    return (model["historicalCompletionRateByStage"] or {})[stage]


@pytest.fixture()
def corrected(monkeypatch):
    monkeypatch.setenv(MATURITY_FLAG_ENV, "1")


@pytest.fixture()
def flag_off(monkeypatch):
    monkeypatch.delenv(MATURITY_FLAG_ENV, raising=False)


# --------------------------------------------------------------------------- #
# 1 — uniform entry: the corrected estimator recovers the true probability
# --------------------------------------------------------------------------- #
def test_uniform_entry_matured_recovers_true_p_pooled_is_attenuated(tmp_path):
    """True p = 0.6, lag 4 weeks, uniform entry over a 26-week window.

    Entrants in the last four weeks cannot have completed yet, so the pooled
    denominator carries them as failures. The matured estimator excludes them
    and recovers 0.6 exactly.
    """
    weeks, per_week, completing = 26, 5, 3          # p = 3/5
    cases = uniform_cases(weeks, per_week, completing)
    model = build_historical_completion_model(
        write_history(tmp_path, weeks, cases), min_observations=1)

    block = stage_block(model)
    assert block["maturedRate"] == pytest.approx(0.6, abs=1e-4)

    # The pooled rate is attenuated by roughly (W - T) / W and must NOT be what
    # the corrected estimator reports.
    assert block["pooledRate"] < block["maturedRate"]
    assert block["pooledRate"] == pytest.approx(0.6 * (weeks - 4) / weeks,
                                                abs=0.02)
    assert block["maturedRate"] != pytest.approx(block["pooledRate"], abs=1e-4)
    # Matured is a strict subset of pooled: same completions, fewer denominators.
    assert block["maturedObserved"] < block["pooledObserved"]


# --------------------------------------------------------------------------- #
# 2 — growing book: pooled degrades with recency skew, matured does not
# --------------------------------------------------------------------------- #
#: Entry volume rises toward the window end as ``skew`` increases. Blocks of
#: five with three converters keep the TRUE probability at exactly 0.6 for
#: every week and every skew — otherwise integer rounding on small weekly
#: volumes would move p with the skew and the test would measure the fixture
#: rather than the censoring.
def _skewed_cases(weeks: int, skew: int) -> List[Case]:
    out: List[Case] = []
    for w in range(weeks):
        blocks = 1 + skew * w
        for i in range(5 * blocks):
            out.append((f"S{w:03d}_{i}", w, i < 3 * blocks))
    return out


@pytest.mark.parametrize("skew", [0, 1])
def test_growing_book_pooled_degrades_matured_stays_stable(tmp_path, skew):
    weeks = 14
    model = build_historical_completion_model(
        write_history(tmp_path, weeks, _skewed_cases(weeks, skew)),
        min_observations=1)
    block = stage_block(model)
    # The matured estimator recovers the true 0.6 whatever the entry gradient.
    assert block["maturedRate"] == pytest.approx(0.6, abs=1e-4)
    assert block["pooledRate"] < block["maturedRate"]


def test_recency_skew_widens_the_pooled_gap_but_not_the_matured_one(tmp_path):
    """The censoring mechanism itself: more recent-weighted entry means more of
    the pooled denominator has had no chance to convert."""
    weeks = 14
    flat = build_historical_completion_model(
        write_history(tmp_path / "flat", weeks, _skewed_cases(weeks, 0)),
        min_observations=1)
    steep = build_historical_completion_model(
        write_history(tmp_path / "steep", weeks, _skewed_cases(weeks, 1)),
        min_observations=1)

    flat_b, steep_b = stage_block(flat), stage_block(steep)
    # Pooled deteriorates as entry skews toward the window end...
    assert steep_b["pooledRate"] < flat_b["pooledRate"]
    # ...while the matured estimator is identical, because it never counted the
    # immature entrants that the skew added.
    assert steep_b["maturedRate"] == pytest.approx(flat_b["maturedRate"],
                                                   abs=1e-4)


# --------------------------------------------------------------------------- #
# 3 — immature cohort clears the raw floor but must not become load-bearing
# --------------------------------------------------------------------------- #
def test_immature_cohort_clears_raw_floor_but_not_the_matured_floor(tmp_path):
    """15 cases, all entered inside the maturity window, none completed yet.

    Against the CURRENT methodology this publishes a 0.0 empirical rate that
    overrides the configured 0.75. The corrected mode must fall back to config.
    """
    weeks = 3                                   # 14 days < OFFER's 30-day window
    cases = [(f"N{i}", 1, False) for i in range(15)]
    entries = write_history(tmp_path, weeks, cases)

    model = build_historical_completion_model(entries)
    block = stage_block(model)

    # The defect this test protects against: the raw floor is cleared...
    assert block["pooledObserved"] >= MIN_OBSERVATIONS
    assert block["pooledRate"] == 0.0
    # ...but no case has matured, so the corrected floor is not.
    assert block["maturedObserved"] == 0
    assert block["maturedObserved"] < MIN_OBSERVATIONS

    # Flag OFF reproduces the defect exactly (this is today's behaviour).
    assert model["stage_rates"].get(STAGE) == 0.0
    assert block["rateSource"] == RATE_POOLED


def test_immature_cohort_falls_back_to_config_when_corrected(tmp_path,
                                                             corrected):
    weeks = 3
    cases = [(f"N{i}", 1, False) for i in range(15)]
    model = build_historical_completion_model(
        write_history(tmp_path, weeks, cases))
    block = stage_block(model)

    assert block["rateSource"] == RATE_CONFIG
    assert block["rateUsed"] is None
    # The near-zero empirical rate never reaches the forecast...
    assert STAGE not in model["stage_rates"]
    # ...and the fallback is disclosed through the existing evidence path that
    # already drives the Teams forecast-weakness limitation.
    assert STAGE in model["stagesUsingConfigFallback"]


# --------------------------------------------------------------------------- #
# 4 — mixed mature / immature
# --------------------------------------------------------------------------- #
def test_mixed_cohort_admits_only_matured_cases(tmp_path):
    weeks = 12
    mature = [(f"M{i}", 0, i < 6) for i in range(10)]      # entered week 0
    immature = [(f"I{i}", weeks - 1, False) for i in range(8)]  # final snapshot
    model = build_historical_completion_model(
        write_history(tmp_path, weeks, mature + immature), min_observations=1)
    block = stage_block(model)

    assert block["pooledObserved"] == 18
    assert block["maturedObserved"] == 10          # the immature 8 are excluded
    assert block["maturedCompleted"] == 6
    assert block["maturedRate"] == pytest.approx(0.6)
    # Pooled dilutes the same six completions across eighteen cases.
    assert block["pooledRate"] == pytest.approx(6 / 18, abs=1e-4)
    assert block["maturityWindowDays"] == _stage_days_to_fund()[STAGE]


def test_matured_rate_is_deterministic_across_repeated_builds(tmp_path):
    weeks = 12
    cases = uniform_cases(weeks, 4, 2)
    entries = write_history(tmp_path, weeks, cases)
    first = stage_block(build_historical_completion_model(entries,
                                                          min_observations=1))
    second = stage_block(build_historical_completion_model(entries,
                                                           min_observations=1))
    assert first == second


# --------------------------------------------------------------------------- #
# 5 — identifier integrity
# --------------------------------------------------------------------------- #
def test_unidentified_rows_are_excluded_and_counted(tmp_path):
    weeks = 12
    good = [(f"G{i}", 0, i < 5) for i in range(10)]
    unstitchable = [(f"X{i}", 0, True) for i in range(4)]
    entries = write_history(tmp_path, weeks, good + unstitchable,
                            blank_identity={c[0] for c in unstitchable})
    model = build_historical_completion_model(entries, min_observations=1)
    block = stage_block(model)

    # Identified cases contribute normally; unidentified ones contaminate
    # neither the pooled nor the matured population.
    assert block["pooledObserved"] == 10
    assert block["maturedObserved"] == 10
    assert block["maturedRate"] == pytest.approx(0.5)

    # The exclusion is surfaced rather than silent, and the model still works.
    assert model["identityExcludedRowCount"] > 0
    assert "identifier" in (model["identityExcludedReason"] or "")
    assert model["available"] is True


def test_a_frame_with_no_identifier_column_does_not_fail_the_model(tmp_path):
    path = tmp_path / "KFI_Pipeline_2025_01_06.csv"
    pd.DataFrame({"Status": ["Offer"] * 5, "Loan Amount": [1.0] * 5}).to_csv(
        path, index=False)
    model = build_historical_completion_model(
        [{"source_file": str(path), "pipeline_extract_date": "2025-01-06"}])
    assert model["available"] is False
    assert model["stage_rates"] == {}


# --------------------------------------------------------------------------- #
# 6 — flag-off regression
# --------------------------------------------------------------------------- #
def test_flag_off_stage_rates_and_probabilities_are_unchanged(tmp_path,
                                                              flag_off):
    """With the feature disabled the forecast must not move: stage_rates stay
    the pooled empirical rates and completion_probability follows them."""
    weeks = 26
    cases = uniform_cases(weeks, 5, 3)
    model = build_historical_completion_model(write_history(tmp_path, weeks,
                                                            cases))
    block = stage_block(model)

    # stage_rates is the POOLED estimator, exactly as before the correction.
    assert model["maturityCorrectionEnabled"] is False
    assert block["rateSource"] == RATE_POOLED
    assert model["stage_rates"][STAGE] == block["pooledRate"]
    assert model["stage_rates"][STAGE] == block["rate"]        # legacy key
    assert model["stagesUsingMaturedRates"] == []

    # ...and the probability assigned to a live case is that pooled rate.
    raw = pd.DataFrame({"Account Number": ["L1"], "Status": [STAGE_LABEL],
                        "Loan Amount": [250_000.0],
                        "Property Region": ["London"]})
    prepared, report = prepare_pipeline_mi_dataset(
        raw, as_of_date=_week_date(weeks - 1), historical_model=model)
    assert prepared["completion_probability"].iloc[0] == pytest.approx(
        block["pooledRate"])
    assert report["completion_probability_basis"] == "historical_observed"


def test_flag_on_changes_only_the_selected_rate(tmp_path, monkeypatch):
    """The two modes differ in stage_rates and rateSource — and in nothing
    else about the pooled evidence, which is published either way."""
    weeks = 26
    entries = write_history(tmp_path, weeks, uniform_cases(weeks, 5, 3))

    monkeypatch.delenv(MATURITY_FLAG_ENV, raising=False)
    off = build_historical_completion_model(entries)
    monkeypatch.setenv(MATURITY_FLAG_ENV, "1")
    on = build_historical_completion_model(entries)

    off_b, on_b = stage_block(off), stage_block(on)
    # Pooled evidence is identical in both modes.
    for field in ("pooledObserved", "pooledCompleted", "pooledRate",
                  "maturedObserved", "maturedCompleted", "maturedRate"):
        assert off_b[field] == on_b[field]
    # Only the selection differs.
    assert off_b["rateSource"] == RATE_POOLED
    assert on_b["rateSource"] == RATE_MATURED
    assert on["stage_rates"][STAGE] == on_b["maturedRate"]
    assert off["stage_rates"][STAGE] == off_b["pooledRate"]
    assert on["stage_rates"][STAGE] > off["stage_rates"][STAGE]


# --------------------------------------------------------------------------- #
# 8 — matured cohort stability diagnostic
# --------------------------------------------------------------------------- #
def test_earlier_and_later_matured_cohorts_diverge_without_moving_the_rate(
        tmp_path, corrected):
    """Conversion deteriorates in the second half of the matured sample.

    The diagnostic must show it; it must not change the published rate.
    """
    weeks = 24
    cases: List[Case] = []
    for w in range(weeks):
        completing = 4 if w < weeks // 2 else 1        # 80% -> 20%
        for i in range(5):
            cases.append((f"D{w:03d}_{i}", w, i < completing))
    model = build_historical_completion_model(
        write_history(tmp_path, weeks, cases), min_observations=1)
    block = stage_block(model)

    assert block["maturedCohortSplitAvailable"] is True
    assert block["maturedObservedEarlierCohort"] > 0
    assert block["maturedObservedLaterCohort"] > 0
    assert block["maturedRateEarlierCohort"] > block["maturedRateLaterCohort"]
    assert block["maturedSampleStart"] <= block["maturedSampleEnd"]

    # Evidence only: the published rate is the whole matured sample, not a
    # cohort, and no stability threshold suppresses it.
    assert block["rateUsed"] == block["maturedRate"]
    assert model["stage_rates"][STAGE] == block["maturedRate"]


def test_a_thin_matured_sample_reports_no_cohort_split(tmp_path):
    weeks = 12
    model = build_historical_completion_model(
        write_history(tmp_path, weeks, [("ONE", 0, True)]), min_observations=1)
    block = stage_block(model)
    assert block["maturedObserved"] == 1
    assert block["maturedCohortSplitAvailable"] is False
    assert block["maturedRateEarlierCohort"] is None
    assert block["maturedRateLaterCohort"] is None


# --------------------------------------------------------------------------- #
# 10 — the evidence path carries the maturity disclosure
# --------------------------------------------------------------------------- #
def test_evidence_block_carries_maturity_without_dropping_legacy_fields(
        tmp_path, corrected):
    weeks = 26
    model = build_historical_completion_model(
        write_history(tmp_path, weeks, uniform_cases(weeks, 5, 3)))
    evidence = historical_model_evidence(model, "historical_observed")

    # Additive: every field a consumer read before is still there.
    for legacy in ("weeklyFilesUsed", "observationWindowStart",
                   "observationWindowEnd", "stagesUsingHistoricalRates",
                   "stagesUsingConfigFallback", "stableIdentifierUsed"):
        assert legacy in evidence

    assert evidence["maturityCorrectionEnabled"] is True
    assert STAGE in evidence["stagesUsingMaturedRates"]
    assert evidence["maturityWindowDaysByStage"][STAGE] == \
        _stage_days_to_fund()[STAGE]
    stage_evidence = evidence["maturityByStage"][STAGE]
    assert stage_evidence["rateSource"] == RATE_MATURED
    assert stage_evidence["maturedObserved"] > 0
    assert stage_evidence["pooledRate"] < stage_evidence["maturedRate"]


# --------------------------------------------------------------------------- #
# 9 — young client fallback
# --------------------------------------------------------------------------- #
def test_young_client_keeps_a_working_forecast_on_configured_probabilities(
        tmp_path, corrected):
    """Real volume, no matured history: the forecast stays operational and the
    standard configured probability is used."""
    weeks = 3
    cases = [(f"Y{i}", 0, False) for i in range(40)]     # plenty of volume
    model = build_historical_completion_model(
        write_history(tmp_path, weeks, cases))

    assert model["casesTracked"] == 40
    assert stage_block(model)["rateSource"] == RATE_CONFIG
    assert STAGE in model["stagesUsingConfigFallback"]

    raw = pd.DataFrame({"Account Number": ["L1"], "Status": [STAGE_LABEL],
                        "Loan Amount": [250_000.0],
                        "Property Region": ["London"]})
    prepared, report = prepare_pipeline_mi_dataset(
        raw, as_of_date=_week_date(weeks - 1), historical_model=model)

    configured = _stage_probabilities()[STAGE]
    assert prepared["completion_probability"].iloc[0] == pytest.approx(configured)
    assert report["completion_probability_basis"] == "stage_config"
    assert prepared["weighted_expected_funded_amount"].iloc[0] == pytest.approx(
        250_000.0 * configured)
