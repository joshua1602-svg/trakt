"""tests/test_funnel_evolution.py

Weekly origination funnel trends (Part 2): weekly KFI / Application / Offer /
Completion value + count, 5-week average, latest week value/count, delta vs prior
week, and the single-period state. Uses the governed client_001 pipeline fixture.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import evolution as evo

_PIPELINE_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack"


def test_five_week_average_helper():
    assert evo._trailing_avg([10.0, 20.0, 30.0], 5) == 20.0
    assert evo._trailing_avg([1, 2, 3, 4, 5, 6, 7], 5) == 5.0  # last 5: 3..7
    assert evo._trailing_avg([], 5) is None
    assert evo._trailing_avg([None, 10.0], 5) == 10.0


def test_trend_helper():
    assert evo._trend([1.0, 2.0]) == "up"
    assert evo._trend([2.0, 1.0]) == "down"
    assert evo._trend([1.0]) == "flat"


def test_funnel_evolution_builds_stage_series():
    warnings.simplefilter("ignore")
    out = evo.pipeline_funnel_evolution(_PIPELINE_FIXTURE, "client_001", None)
    assert out["dataset"] == "pipeline_funnel"
    assert out["stages"] == ["KFI", "APPLICATION", "OFFER", "COMPLETED"]
    # Every stage has a per-week series with value + count keys.
    for stage in out["stages"]:
        assert stage in out["series"]
        for pt in out["series"][stage]:
            assert "week" in pt and "value" in pt and "count" in pt


def test_funnel_summary_metrics():
    warnings.simplefilter("ignore")
    out = evo.pipeline_funnel_evolution(_PIPELINE_FIXTURE, "client_001", None)
    summ = out["summary"]
    for stage in out["stages"]:
        s = summ[stage]
        assert "latestValue" in s and "latestCount" in s
        assert "fiveWeekAvgValue" in s and "fiveWeekAvgCount" in s
        assert "deltaValue" in s and "deltaCount" in s
        assert s["trend"] in ("up", "down", "flat")
    # Completions stage has at least one completed case in the fixture.
    assert summ["COMPLETED"]["latestCount"] >= 1
    assert summ["KFI"]["label"] == "KFIs"


def test_funnel_has_source_and_weeks():
    warnings.simplefilter("ignore")
    out = evo.pipeline_funnel_evolution(_PIPELINE_FIXTURE, "client_001", None)
    assert out["weeks"]
    assert out["sourceFiles"]
    assert out["lineage"]["source"].startswith("governed weekly pipeline")


def test_funnel_single_period_state(tmp_path):
    # No pipeline data -> empty, single-period.
    out = evo.pipeline_funnel_evolution(tmp_path, "client_001", None)
    assert out["singlePeriod"] is True
    assert out["weeks"] == []


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
