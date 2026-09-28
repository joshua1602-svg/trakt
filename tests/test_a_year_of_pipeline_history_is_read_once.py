"""A year of weekly pipeline history is prepared once, not on every request.

ERE's backfill published 90 weekly pipeline snapshots and the Evolution tab
went blank. Evolution, the origination funnel and the forecast bridge each
walk every weekly extract, and they read the prepared frames through a memo
that holds 64 — so each request evicted the frames it was about to need and
re-prepared the whole year, for every chart, on every visit, until the
request timed out.

The series now read a small per-extract summary memoised on the same
identity (source bytes, as-of date, historical model), sized for years of
weekly history. The numbers are unchanged; only the repeated work is gone.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mi_agent_api import evolution as ev
from mi_agent_api import pipeline_contract as pc
from mi_agent_api import serving_cache


def _write_weeks(root: Path, n: int) -> None:
    start = pd.Timestamp("2025-09-01")
    for i in range(n):
        day = (start + pd.Timedelta(days=4 * i)).strftime("%Y-%m-%d")
        folder = root / "ERE" / day
        folder.mkdir(parents=True)
        pd.DataFrame({
            "Account Number": [f"A{i}-1", f"A{i}-2", f"A{i}-3"],
            "Loan Amount": [100000 + i, 50000, 75000],
            "Status": ["KFI", "Offer", "Completed"],
        }).to_csv(folder / "pipeline_snapshot.csv", index=False)


@pytest.fixture()
def year_of_weeks(tmp_path, monkeypatch):
    monkeypatch.setattr(serving_cache, "resolved_tenant", lambda: "ere")
    for cache in (pc._PIPELINE_PREP_CACHE, pc._EXTRACT_SUMMARY_CACHE):
        cache.clear()
    weeks = pc._PIPELINE_PREP_CACHE.max_entries + 10
    _write_weeks(tmp_path, weeks)
    prepared = []
    real = pc.prepare_pipeline_mi_dataset

    def counting(*a, **kw):
        prepared.append(kw.get("source_file"))
        return real(*a, **kw)
    monkeypatch.setattr(pc, "prepare_pipeline_mi_dataset", counting)
    return tmp_path, weeks, prepared


def test_more_weeks_than_the_frame_memo_holds_are_prepared_once(year_of_weeks):
    root, weeks, prepared = year_of_weeks
    first = ev.pipeline_evolution(root, "ERE")
    assert len(first["periods"]) == weeks
    assert len(prepared) == weeks
    prepared.clear()
    again = ev.pipeline_evolution(root, "ERE")
    ev.pipeline_funnel_evolution(root, "ERE")
    assert prepared == []
    assert again["periods"] == first["periods"]
    assert again["byStage"] == first["byStage"]


def test_the_series_read_the_same_numbers(year_of_weeks):
    root, weeks, _prepared = year_of_weeks
    evo = ev.pipeline_evolution(root, "ERE")
    last = evo["periods"][-1]["metrics"]
    assert last["pipeline_case_count"] == 3
    assert last["pipeline_amount"] == 100000 + (weeks - 1) + 50000 + 75000
    stages = {r["stage"]: r for r in evo["byStage"] if r["week"] == evo["periods"][-1]["week"]}
    assert stages["OFFER"]["value"] == 50000 and stages["OFFER"]["count"] == 1
    funnel = ev.pipeline_funnel_evolution(root, "ERE")
    assert funnel
