"""tests/test_mi_capability_recontent.py — the re-rating instrument must be able
to fail.

The re-rating exists because the previous instrument credited a limb only when a
COLUMN NAME matched. Content matching trades that error for a worse one if it is
sloppy: crediting a breakdown because some column happens to hold two values.
Every test here pairs a case the instrument must credit with one it must refuse.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from question_interpretation import mi_capability_recontent as R
from question_interpretation.time_series_surface import ABSENT, PROVEN


REGIONS = {"london", "north east", "north west", "scotland", "south east"}
SOURCES = {"direct", "acquired"}


def _census(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return R.full_census([{"type": "table", "rows": rows}])


# --------------------------------------------------------------------------- #
# 1. A DIMENSION IS ITS VALUES, NOT ITS COLUMN NAME
# --------------------------------------------------------------------------- #
def test_a_generic_column_holding_the_dimension_values_is_credited():
    """The exact T7 defect: `category` holds regions. Name matching missed it."""
    rows = [{"category": "London", "start_value": 1, "end_value": 2},
            {"category": "Scotland", "start_value": 3, "end_value": 4}]
    rating, why, ev = R.rate_content(True, ["region"], _census(rows),
                                     {"region": REGIONS})
    assert rating == PROVEN
    assert ev["dimension"]["column"] == "category"
    assert "category" in why


def test_a_decorated_label_still_matches_its_domain_value():
    """The T8 case: the book stores `Front Book`, the artifact renders
    `Front Book (0-12 months)`."""
    rows = [{"population": "Front Book (0-12 months)", "prior": 1, "current": 2},
            {"population": "Back Book (13+ months)", "prior": 3, "current": 4}]
    rating, _why, ev = R.rate_content(True, ["seasoning"], _census(rows),
                                      {"seasoning": {"front book", "back book"}})
    assert rating == PROVEN
    assert ev["dimension"]["column"] == "population"


def test_an_unrelated_column_with_two_values_is_NOT_credited():
    """The failure mode content matching could introduce. `measure` holds two
    values; neither is a region. It must not buy a breakdown."""
    rows = [{"measure": "balance", "period": "2025-01", "prior": 1, "current": 2},
            {"measure": "count", "period": "2025-02", "prior": 3, "current": 4}]
    rating, why, ev = R.rate_content(True, ["region"], _census(rows),
                                     {"region": REGIONS})
    assert rating == ABSENT
    assert ev["dimension"] is None
    assert "NO column carries" in why


def test_one_matching_value_is_not_a_breakdown():
    """A column holding a single region is a filter, not a breakdown."""
    rows = [{"category": "London", "prior": 1, "current": 2},
            {"category": "London", "prior": 3, "current": 4}]
    rating, _why, _ev = R.rate_content(True, ["region"], _census(rows),
                                       {"region": REGIONS})
    assert rating == ABSENT


def test_column_carries_matches_both_directions_but_not_strangers():
    assert R.column_carries(["Direct", "Acquired"], SOURCES) == ["Direct", "Acquired"]
    assert R.column_carries(["Front Book (0-12 months)"], {"front book"}) != []
    assert R.column_carries(["balance", "count"], REGIONS) == []


# --------------------------------------------------------------------------- #
# 2. THE TIME AXIS — THE BRIEF'S RULE, BOTH FORMS
# --------------------------------------------------------------------------- #
def test_movement_pair_is_a_time_axis():
    rows = [{"population": "Direct", "prior": 1, "current": 2}]
    assert "movement pair" in R.time_axis(_census(rows))


def test_a_named_time_column_with_many_values_is_a_time_axis():
    rows = [{"period": "2025-01", "balance": 1}, {"period": "2025-02", "balance": 2}]
    assert "period" in R.time_axis(_census(rows))


def test_a_time_column_with_one_value_is_not_a_time_axis():
    rows = [{"period": "2025-01", "region": "London", "balance": 1},
            {"period": "2025-01", "region": "Scotland", "balance": 2}]
    assert R.time_axis(_census(rows)) is None


def test_no_time_axis_refuses_even_with_a_perfect_breakdown():
    """A breakdown over one period is not a time series, however clean."""
    rows = [{"period": "2025-01", "category": "London", "balance": 1},
            {"period": "2025-01", "category": "Scotland", "balance": 2}]
    rating, why, _ev = R.rate_content(True, ["region"], _census(rows),
                                      {"region": REGIONS})
    assert rating == ABSENT
    assert "no time axis" in why


def test_a_refusal_is_never_credited():
    rating, why, _ev = R.rate_content(False, ["region"], _census([]),
                                      {"region": REGIONS})
    assert rating == ABSENT and why == "refused"


# --------------------------------------------------------------------------- #
# 3. THE DOMAIN MUST BE THE REPORTING DIMENSION, NOT A CODE COLUMN
# --------------------------------------------------------------------------- #
def test_domains_exclude_high_cardinality_code_columns(monkeypatch):
    """`geographic_region_obligor` holds 172 NUTS codes; `collateral_geography`
    holds the 11 reporting regions. A question about regions means the latter."""
    import pandas as pd
    frame = pd.DataFrame({
        "collateral_geography": ["London", "Scotland"] * 40,
        "geographic_region_obligor": ["TLC%d" % i for i in range(80)],
    })

    class _DS:
        @staticmethod
        def get_dataframe():
            return frame

    monkeypatch.setitem(__import__("sys").modules, "mi_agent_api.data_source", _DS)
    domains = R.dimension_domains(["geograph"])
    assert "london" in domains["geograph"]
    assert not any(v.startswith("tlc") for v in domains["geograph"]), \
        "a 80-value code column must not become the region domain"


# --------------------------------------------------------------------------- #
# 4. SAME SHAPE IS NOT SAME REQUEST
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question,target", [
    ("Which region has grown fastest?", "region"),
    ("Rank regions by balance growth over time", "region"),
    ("Which LTV band moved most between periods?", "ltv_band"),
    ("Compare balance over time for direct and acquired", "source_split"),
    ("How have direct and acquired balances moved over the periods?", "source_split"),
    ("How has the front book moved over time compared with the back book?", "seasoning"),
    ("Show me balance by month for loans over £150k", "threshold"),
    ("Show me balance by month", "none"),
])
def test_request_target_separates_requests_within_one_shape(question, target):
    assert R.request_target(question) == target


def test_a_different_parameter_of_the_same_shape_is_not_a_routing_gap():
    """T7's LTV-band phrasing must NOT be called reachable because the REGION
    phrasing works. That would overstate routing and understate P1's scope."""
    rows = [
        {"shape": "T7", "question": "Which region has grown fastest?",
         "status": R.DELIVERS},
        {"shape": "T7", "question": "Which LTV band moved most between periods?",
         "status": R.INCOMPLETE},
    ]
    sib = R.sibling_analysis({"rows": rows})
    strict = {r["question"]: r["gap"] for r in sib["strict"]}
    assert strict["Which LTV band moved most between periods?"] == R.CAPABILITY_GAP
    assert sib["strict_routing_gaps"] == 0
    # the LOOSE view, which groups by shape alone, would have called it routing
    assert sib["routing_gaps"] == 1


def test_the_same_request_in_other_words_is_a_routing_gap():
    rows = [
        {"shape": "T8", "question": "Compare balance over time for direct and acquired",
         "status": R.DELIVERS},
        {"shape": "T8", "question": "How have direct and acquired balances moved over the periods?",
         "status": R.REFUSED},
    ]
    sib = R.sibling_analysis({"rows": rows})
    assert sib["strict_routing_gaps"] == 1


# --------------------------------------------------------------------------- #
# 5. THE PREFLIGHT MUST REFUSE THE TWO TRAPS
# --------------------------------------------------------------------------- #
def test_preflight_refuses_production_mode(monkeypatch):
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "production")
    problems = R.preflight("kestrelmoor")
    assert any("production" in p for p in problems)


def test_preflight_passes_in_development_mode(monkeypatch):
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "development")
    assert R.preflight("kestrelmoor") == [], \
        "the preflight must not block a correctly configured run"


def test_preflight_refuses_a_missing_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    monkeypatch.setattr(cfg, "local_blob_root", lambda: tmp_path / "gone")
    problems = R.preflight("alderbridge")
    assert any("workspace is missing" in p for p in problems)
    assert any("run_demo" in p for p in problems), "must name the rebuild command"


def test_a_breakdown_marker_outranks_a_filter_mention():
    """"balance over time by region for the front book" is a REGION request
    filtered to the front book. Labelling it `seasoning` would group it with the
    wrong siblings."""
    assert R.request_target(
        "balance over time by region for the front book") == "region"
    assert R.request_target(
        "Show me balance by month by region for loans over £150k") == "region"
    assert R.request_target("balance by month split by LTV band") == "ltv_band"
