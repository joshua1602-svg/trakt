"""Every pipeline figure is the OPEN pipeline: KFI, Application and Offer.

Completed and withdrawn cases stay in ERE's weekly extract with their balance,
so "Total pipeline amount £1.18BN" carried £85MM of completed and £128MM of
withdrawn cases, and the movement drill-down showed four cases "leaving the
active pipeline" for £0. The tiles, the weekly series and the drill-down now
measure open cases only, and say what they leave out.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mi_agent_api import movement_detail as md
from mi_agent_api import pipeline_contract as pc
from mi_agent_api.pipeline_prep import OPEN_STAGES, prepare_pipeline_mi_dataset


def _prepared(status, amounts, ids=None):
    raw = pd.DataFrame({
        "Account Number": ids or [f"A{i}" for i in range(len(status))],
        "Status": status,
        "Loan Amount": amounts,
        "Estimated Value": [400_000] * len(status),
        "Broker": ["B1"] * len(status),
    })
    df, report = prepare_pipeline_mi_dataset(raw, as_of_date="2026-09-24")
    return df, report


def test_open_stages_are_kfi_application_offer():
    assert OPEN_STAGES == ("KFI", "APPLICATION", "OFFER")


def test_the_snapshot_counts_open_cases_and_discloses_the_rest():
    df, report = _prepared(["KFI", "Application", "Offer", "Completed", "Withdrawn"],
                           [10_000, 20_000, 30_000, 40_000, 50_000])
    snap = pc.compute_pipeline_snapshot(df, report, {}, client_id="ERE", run_id="x")
    assert snap["pipelineRowCount"] == 3
    assert snap["pipelineAmount"] == 60_000
    assert {r["stage"] for r in snap["stageBreakdown"]} == {"KFI", "APPLICATION", "OFFER"}
    excluded = {r["stage"]: r for r in snap["excludedFromOpenPipeline"]["stages"]}
    assert excluded["COMPLETED"]["amount"] == 40_000
    assert excluded["WITHDRAWN"]["caseCount"] == 1
    assert snap["extractRowCount"] == 5


def test_a_case_that_completes_takes_its_balance_out_of_the_pipeline():
    prior, _ = _prepared(["KFI", "Offer", "Offer", "Completed"],
                         [10_000, 20_000, 30_000, 40_000], ids=["A", "B", "C", "D"])
    current, _ = _prepared(["KFI", "Completed", "Withdrawn", "Completed", "Offer"],
                           [10_000, 20_000, 30_000, 40_000, 5_000],
                           ids=["A", "B", "C", "D", "E"])
    out = md.build_movement_detail(md.DETAIL_PIPELINE, current, prior,
                                   as_of_date="2026-09-24",
                                   comparison_date="2026-09-17", portfolio_id="ERE")
    comp = out["components"]
    assert comp["progressed_out"] == {"amount": -50_000.0, "cases": 2}
    assert comp["new"] == {"amount": 5_000.0, "cases": 1}
    # D was completed in both weeks: not part of the open pipeline's movement.
    assert sum(c["cases"] for c in comp.values()) == 4
    assert out["headline_metric"]["value"] == 15_000
    assert out["headline_metric"]["change"] == -45_000
    assert out["counts"] == {"current": 2, "comparison": 3, "change": -1}


def test_contributor_cases_are_the_cases_that_moved():
    prior, _ = _prepared(["KFI", "KFI", "KFI"], [10_000, 20_000, 30_000],
                         ids=["A", "B", "C"])
    current, _ = _prepared(["KFI", "KFI", "KFI", "KFI"], [10_000, 20_000, 30_000, 7_000],
                           ids=["A", "B", "C", "D"])
    out = md.build_movement_detail(md.DETAIL_PIPELINE, current, prior,
                                   as_of_date="2026-09-24",
                                   comparison_date="2026-09-17", portfolio_id="ERE")
    broker = out["contributors"]["brokers"][0]
    assert broker["amount"] == 7_000
    assert broker["case_count"] == 1, "one case moved; three sat unchanged"
