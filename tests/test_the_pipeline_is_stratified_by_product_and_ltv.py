"""The pipeline snapshot stratifies by product and by LTV band.

The Pipeline page showed stage, completion month, broker and region. ERE's
pipeline file also carries the product and the property value, so the
snapshot now breaks the pipeline down by product and by LTV band (loan amount
/ estimated value, banded by the same engine the funded book uses) — with the
same amount, count and weighted rows as the other breakdowns.
"""
from __future__ import annotations

import pandas as pd

from mi_agent_api import pipeline_contract as pc
from mi_agent_api.pipeline_prep import prepare_pipeline_mi_dataset


def _snapshot():
    raw = pd.DataFrame({
        "Account Number": ["A1", "A2", "A3", "A4"],
        "Status": ["Offer", "Application", "KFI", "Completed"],
        "Loan Amount": [100_000, 50_000, 30_000, 80_000],
        "Estimated Value": [400_000, 100_000, 300_000, 200_000],
        "Product": ["Lump Sum", "Drawdown", "Drawdown", "Lump Sum"],
    })
    df, report = prepare_pipeline_mi_dataset(raw, as_of_date="2026-09-24")
    return pc.compute_pipeline_snapshot(df, report, {}, client_id="ERE", run_id="x")


def test_the_pipeline_is_broken_down_by_product():
    snap = _snapshot()
    rows = {r["key"]: r for r in snap["productBreakdown"]}
    # The open pipeline: A4 is Completed, so it is not in the breakdown.
    assert rows["Lump Sum"]["pipelineAmount"] == 100_000
    assert rows["Lump Sum"]["caseCount"] == 1
    assert rows["Drawdown"]["pipelineAmount"] == 80_000
    assert snap["productBreakdownFull"] == snap["productBreakdown"]


def test_the_pipeline_is_broken_down_by_ltv_band():
    snap = _snapshot()
    rows = snap["ltvBreakdown"]
    assert rows, "LTV derives from loan amount / estimated value"
    assert sum(r["caseCount"] for r in rows) == 3
    assert sum(r["pipelineAmount"] for r in rows) == 180_000
