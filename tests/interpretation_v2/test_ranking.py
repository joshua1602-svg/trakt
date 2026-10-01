"""A ranking states which end and how many (vocabulary 2.22.0; twins run
2026-10-01).

"Which region has the highest average loan balance?" and "Which region has
the largest funded balance?" were read correctly and refused: the funded
runtime served no `rank`, and the pipeline listed none. A ranking also had
nowhere to say WHICH end was meant — "the smallest" read the same as "the
largest". The intent now carries `ranking` (`order`: highest / lowest, and
the `limit` the reader named); the compiler carries it to the plan; the
funded executor ranks by the figure itself in that direction, cut to that
number, and the pipeline orders the Pipeline tab's own breakdown the same
way. Nothing is ranked by a second implementation.
"""
from __future__ import annotations

import pytest

from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import IntentParseError, parse_candidate_intent
from tests.interpretation_v2.test_funded_breadth import (  # noqa: F401
    _intent, _served, book, semantics)
from tests.interpretation_v2.test_specialist_runtime_pipeline import (
    _PIPELINE_INTENT, _semantics)
from tests.interpretation_v2.test_specialist_runtime_pipeline import (
    _served as _served_pipeline)

_COUNT_BY_CHANNEL = dict(operation="rank",
                         measures=[{"concept": "loan", "statistic": "count"}],
                         dimensions=["origination_channel"])


def _compile(payload):
    return DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(payload))


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [{"order": "best"}, {"limit": 0}, {"limit": 51},
                                 {"limit": "5"}, {"order": "highest", "by": "x"}])
def test_a_ranking_is_an_end_and_a_bounded_number_only(bad):
    with pytest.raises(IntentParseError):
        parse_candidate_intent(dict(_intent(**_COUNT_BY_CHANNEL), ranking=bad))


def test_a_rank_naming_no_end_is_read_highest_first_and_says_so():
    plan = _compile(_intent(**_COUNT_BY_CHANNEL)).plan.to_dict()
    assert plan["ranking"] == {"order": "highest", "limit": None, "defaulted": True}
    assert any("highest first" in n for n in plan["provenance"]["notes"])


def test_a_ranking_on_an_operation_with_nothing_to_order_conflicts():
    result = _compile(dict(_intent(operation="point_in_time",
                                   measures=[{"concept": "loan",
                                              "statistic": "count"}]),
                           ranking={"order": "lowest"}))
    assert result.plan is None
    assert "CONFLICTING_CLAIMS" in result.codes()


def test_a_plan_without_a_ranking_keeps_its_identity():
    """Every plan recorded before the slot existed replays to its own id."""
    plan = _compile(_intent(operation="breakdown",
                            measures=[{"concept": "loan", "statistic": "count"}],
                            dimensions=["origination_channel"])).plan
    assert plan.ranking is None
    assert "ranking" not in plan._authorised_content()


# --------------------------------------------------------------------------- #
# the funded book
# --------------------------------------------------------------------------- #

def _channel_counts(book):
    return book["origination_channel"].value_counts()


@pytest.mark.parametrize("ranking, expected_lead", [
    ({"order": "highest"}, "highest"), ({"order": "lowest"}, "lowest")])
def test_the_funded_book_names_the_end_asked_for(monkeypatch, book, semantics,
                                                 ranking, expected_lead):
    payload, record, coverage = _served(
        dict(_intent(**_COUNT_BY_CHANNEL), ranking=ranking), monkeypatch, book,
        semantics)
    assert payload is not None, record.get("execution")
    counts = _channel_counts(book)
    name = counts.idxmax() if expected_lead == "highest" else counts.idxmin()
    value = counts.max() if expected_lead == "highest" else counts.min()
    assert payload["answer"].startswith(
        f"{name} has the {expected_lead} number of loans: {value}")
    assert coverage["unaccounted"] == []


def test_the_funded_book_names_the_number_asked_for(monkeypatch, book, semantics):
    payload, record, coverage = _served(
        dict(_intent(**_COUNT_BY_CHANNEL), ranking={"order": "highest", "limit": 2}),
        monkeypatch, book, semantics)
    assert payload is not None, record.get("execution")
    top = _channel_counts(book).head(2)
    listed = ", ".join(f"{k} {v}" for k, v in top.items())
    assert f"— the 2 highest: {listed}." in payload["answer"]
    assert len(record["execution"]["grouped_cells"]) == 2
    assert coverage["unaccounted"] == []


def test_a_count_ranking_is_ranked_by_the_count_not_the_balance(monkeypatch, book,
                                                               semantics):
    """The executor's default top-N basis for an additive figure is balance
    first; a governed ranking is ranked by its own figure."""
    payload, record, _ = _served(
        dict(_intent(**_COUNT_BY_CHANNEL), ranking={"order": "highest", "limit": 1}),
        monkeypatch, book, semantics)
    counts = _channel_counts(book)
    cells = record["execution"]["grouped_cells"]
    assert [c["origination_channel"] for c in cells] == [counts.idxmax()]


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ranking, word", [
    ({"order": "highest"}, "largest"), ({"order": "lowest"}, "smallest"),
    ({"order": "highest", "limit": 2}, "the 2 largest")])
def test_the_pipeline_orders_its_own_breakdown(monkeypatch, ranking, word):
    payload, record = _served_pipeline(
        dict(_PIPELINE_INTENT, operation="rank", dimensions=["pipeline_stage"],
             ranking=ranking), monkeypatch, semantics=_semantics())
    assert payload is not None, record.get("execution")
    assert f"Live pipeline amount by stage — {word}: " in payload["answer"]
    rows = payload["artifacts"][0]["rows"]
    values = [r["value"] for r in rows]
    assert values == sorted(values, reverse=ranking["order"] == "highest")
    assert len(rows) == (ranking.get("limit") or len(rows))
