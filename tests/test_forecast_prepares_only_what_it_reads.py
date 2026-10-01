"""A forecast question prepares only the extracts its figures read (speed,
2026-10-01 runs on 2e9e1cc4).

"What is the base forecast?" and "When does the upside forecast reach £100m?"
each took about 140 s on a cold process: the forecast extrapolation — the
Forecast tab's own owner — prepared all 90 of ERE's weekly extracts twice
over, once for the forecast series and once for the origination funnel, and
read from them the latest funded month's weighted pipeline and the funnel's
headline figures. It now prepares the extracts those figures come from and no
others, and every figure it publishes is the one it published before: these
tests compare it, field for field, with the same owner over every extract.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mi_agent_api import evolution
from mi_agent_api import forecast_extrapolation as fx
from mi_agent_api import pipeline_contract as pc
from tests.interpretation_v2.test_specialist_runtime_forecast import (  # noqa: F401
    _CLIENT, funded_root)
from tests.test_the_pipeline_forecast_runs_off_by_stage import _simulate

#: The funnel figures the extrapolation reads (and the Forecast tab shows as
#: the KFI model's diagnostic inputs).
_HEADLINE = ("latestFlowValue", "latestFlowCount", "priorFlowValue",
             "priorFlowCount", "fiveWeekAvgFlowValue", "fiveWeekAvgFlowCount",
             "deltaFlowValue", "deltaFlowCount", "latestStockValue",
             "latestStockCount", "fiveWeekAvgStockValue", "fiveWeekAvgStockCount",
             "conversion")


@pytest.fixture(scope="module")
def weekly(tmp_path_factory) -> Path:
    """Twenty weekly extracts, 2025-09-08 to 2026-01-19: months that overlap
    the funded book's runs, several extracts in each."""
    root = tmp_path_factory.mktemp("pipeline")
    _simulate(root, n_cases=600, weeks=20)
    (root / "ERE").rename(root / _CLIENT)
    return root


@pytest.mark.parametrize("lag", [None, 0, 3, 12, 19, 40])
def test_the_funnel_headline_is_the_full_series_own(weekly, lag):
    full = evolution.pipeline_funnel_evolution(weekly, _CLIENT, lag_weeks=lag)
    tail = evolution.pipeline_funnel_evolution(
        weekly, _CLIENT, lag_weeks=lag, tail=evolution.funnel_tail_needed(lag))
    assert tail["weeks"][-1] == full["weeks"][-1]
    assert len(tail["weeks"]) == min(len(full["weeks"]),
                                     evolution.funnel_tail_needed(lag))
    for stage, figures in full["summary"].items():
        for key in _HEADLINE:
            assert tail["summary"][stage][key] == figures[key], (stage, key)


def _over_every_extract(monkeypatch):
    """The same owner, preparing every extract as it did before."""
    forecast, funnel = evolution.forecast_evolution, evolution.pipeline_funnel_evolution
    monkeypatch.setattr(evolution, "forecast_evolution",
                        lambda *a, latest_only=False, **k: forecast(*a, **k))
    monkeypatch.setattr(evolution, "pipeline_funnel_evolution",
                        lambda *a, tail=None, **k: funnel(*a, **k))


@pytest.mark.parametrize("run", [None, "mi_2025_11", "mi_2025_09"])
def test_the_extrapolation_is_the_one_over_every_extract(funded_root, weekly,
                                                         monkeypatch, run):
    history = pc.build_pipeline_history(weekly, _CLIENT)
    prepared = []
    load = pc.load_extract_summary
    monkeypatch.setattr(pc, "load_extract_summary",
                        lambda ext, **k: prepared.append(ext) or load(ext, **k))
    fast = fx.build_extrapolation(funded_root, weekly, _CLIENT, run,
                                  history_model=history)
    asked = len(prepared)
    _over_every_extract(monkeypatch)
    slow = fx.build_extrapolation(funded_root, weekly, _CLIENT, run,
                                  history_model=history)
    assert fast == slow
    assert fast["currentWeightedPipelineForecast"]["weightedExpectedPipeline"] is not None
    # Never more extracts read than the series over every one, and fewer
    # whenever the history is longer than the figures need.
    assert asked <= len(prepared) - asked
    if run is None:
        assert asked < len(prepared) - asked
