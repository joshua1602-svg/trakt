"""Every funded answer says what it measured, how it grouped it, the leading
figures and the date it is as at — in the text itself (P0 D4; owner
instruction 2026-09-29: fix the wording before the proof stage).

A breakdown used to read "Here is the bar for your query, covering 10 groups":
true, and it named neither the measure nor the grouping, so a text channel and
the evidence a bank run is judged on carried no answer. The sentence is now
composed by the one owner every funded answer uses (`adapters._answer`), from
the same rows, labels and formatters the table is built from, and the governed
path attaches the same execution receipt the legacy path does — built from what
ran and the book's own cut-off date, never from the question.
"""
from __future__ import annotations

import pytest

from mi_agent_api import adapters
from tests.interpretation_v2.test_funded_breadth import (  # noqa: F401
    _REGION, _intent, _served, book, harmonised, semantics)


def test_a_breakdown_names_its_measure_axis_and_leaders_in_the_owners_order():
    rows = [{"region": "Wales", "current_outstanding_balance_sum": 1_200_000.0},
            {"region": "London", "current_outstanding_balance_sum": 3_900_000.0},
            {"region": "Scotland", "current_outstanding_balance_sum": 2_100_000.0},
            {"region": "North East", "current_outstanding_balance_sum": 500_000.0}]
    line = adapters._answer(None, {"data": rows, "row_count": 4,
                                   "resolved_fields": {}}, "bar", {},
                            {"metric": "current_outstanding_balance",
                             "aggregation": "sum", "dimension": "region",
                             "dimensions": ["region"]})
    assert line.startswith("Current Outstanding Balance by Region — largest: "
                           "London ")
    assert line.index("London") < line.index("Scotland") < line.index("Wales")
    assert "North East" not in line and "and 1 more (4 groups)" in line
    assert "Here is the" not in line


def test_an_average_is_highest_and_a_count_is_a_number_of_loans():
    rows = [{"region": "A", "x_avg": 2.0, "loan_count": 3},
            {"region": "B", "x_avg": 5.0, "loan_count": 1}]
    qr = {"data": rows, "row_count": 2, "resolved_fields": {}}
    avg = adapters._answer(None, qr, "bar", {},
                           {"metric": "x", "aggregation": "avg",
                            "dimensions": ["region"]})
    assert "— highest: B " in avg
    count = adapters._answer(None, qr, "bar", {},
                             {"aggregation": "count", "dimensions": ["region"]})
    assert count.startswith("Number of loans by Region — largest: A 3")


def test_the_governed_region_answer_reads_as_an_answer(monkeypatch, harmonised,
                                                      semantics):
    payload, record, _ = _served(_intent(geography=_REGION), monkeypatch,
                                 harmonised, semantics)
    assert payload is not None, record.get("execution")
    lead, receipt = payload["answer"].split("\n\n", 1)
    top = max(record["execution"]["grouped_cells"], key=lambda c: c["value"])
    assert lead.startswith(f"Balance by Region — largest: "
                           f"{top['canonical_region_reporting']} £")
    # D4: the measure, the grouping, the loans and the as-at, from what ran.
    assert receipt.startswith("Calculated: Total Balance · grouped by Region")
    assert "as at 30 November 2025" in receipt
    assert payload["executionSummary"]["period"] == "30 November 2025"


def test_a_single_figure_carries_the_receipt_too(monkeypatch, book, semantics):
    payload, record, _ = _served(
        _intent(operation="point_in_time",
                measures=[{"concept": "current_outstanding_balance",
                           "statistic": "median"}]),
        monkeypatch, book, semantics)
    assert payload["answer"].startswith("Median Balance: £")
    assert "as at 30 November 2025" in payload["answer"]


def test_a_series_reads_start_to_end_not_ranked():
    rows = [{"reporting_date": "2025-07-31", "x_sum": 9_400_000.0},
            {"reporting_date": "2025-11-30", "x_sum": 12_100_000.0},
            {"reporting_date": "2025-09-30", "x_sum": 20_000_000.0}]
    line = adapters._answer(None, {"data": rows, "row_count": 3,
                                   "resolved_fields": {}}, "line", {},
                            {"metric": "x", "aggregation": "sum",
                             "dimensions": []})
    assert line.startswith("X over 3 reporting dates, 2025-07-31 to 2025-11-30: "
                           "from 9,400,000 to 12,100,000")
    assert "largest" not in line          # a series is not a ranking


def test_a_grouped_series_names_the_leaders_at_its_latest_date():
    rows = [{"reporting_date": d, "stage": s, "x_sum": v}
            for d in ("2025-10-31", "2025-11-30")
            for s, v in (("OFFER", 3.0), ("KFI", 1.0))]
    line = adapters._answer(None, {"data": rows, "row_count": 4,
                                   "resolved_fields": {}}, "line", {},
                            {"metric": "x", "aggregation": "sum",
                             "dimensions": ["stage"]})
    assert line.startswith("X by Stage — largest: OFFER 3, KFI 1 (2 groups), "
                           "at 2025-11-30 — the latest of 2 reporting dates "
                           "from 2025-10-31.")


def test_a_pipeline_answer_says_which_extract_it_is_as_at(monkeypatch):
    """D4 on the pipeline: the figure and the weekly extract it was read from."""
    from tests.interpretation_v2.test_specialist_runtime_pipeline import (
        _PIPELINE_INTENT, _served as _served_pipeline)
    payload, record = _served_pipeline(dict(_PIPELINE_INTENT), monkeypatch)
    assert payload is not None, record.get("execution")
    extract = payload["metadata"]["governedPlan"]["executed"]["dataset"]["as_of_date"]
    assert extract
    assert payload["answer"].startswith("The live pipeline amount is £")
    assert f", as at the weekly extract of {extract}." in payload["answer"]
