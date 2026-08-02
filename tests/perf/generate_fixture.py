#!/usr/bin/env python3
"""tests/perf/generate_fixture.py — a production-REPRESENTATIVE MI fixture tree.

Builds a synthetic ``blob://`` tree that the MI Agent API reads through its
**real** production code path. Nothing here is a stub: the tree is laid out
exactly as ``apps.blob_trigger_app.layout`` publishes it, and the API is pointed
at it with the filesystem storage backend (``TRAKT_STORAGE_BACKEND=file`` +
``TRAKT_LOCAL_BLOB_ROOT``), so ``platform_snapshots_blob``,
``_materialise_pipeline_root``, ``build_index`` and every preparation layer run
unchanged::

    {root}/processed-v2/platform/{client}/{YYYY-MM-DD}/platform_canonical_typed.csv
    {root}/processed-v2/platform/{client}/latest/platform_canonical_typed.csv
    {root}/processed-v2/pipeline/{client}/{YYYY-MM-DD}/pipeline_snapshot.csv
    {root}/processed-v2/pipeline/{client}/latest/pipeline_snapshot.csv

The COLUMNS are the ones the MI serving layer actually consumes (see
``funded_prep._DIM_SPEC`` and ``snapshots._STRAT_SOURCE_COLUMNS``) — the volume
and shape drive the work being measured, so the measurement is representative of
the serving cost even though the values are synthetic.

Deterministic: seeded, so a before/after comparison measures the code change and
nothing else. Values are meaningless; this fixture is for TIMING only and must
never be used to assert a business figure.
"""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path
from typing import List

import pandas as pd

CLIENT = "ERE"
PROCESSED = "processed-v2"
PLATFORM_CANONICAL_NAME = "platform_canonical_typed.csv"
PIPELINE_SNAPSHOT_NAME = "pipeline_snapshot.csv"

BROKERS = ["Air", "Fluent", "Responsible Life", "Key Group", "Age Partnership",
           "Equity Release Supermarket", "HUB Financial", "Bower", "LifeTime",
           "My Community Finance", "Direct", "Other Intermediary"]
REGIONS = ["London", "South East", "South West", "East of England",
           "West Midlands", "East Midlands", "Yorkshire and The Humber",
           "North West", "North East", "Scotland", "Wales"]
PRODUCTS = ["Lifetime Mortgage", "Drawdown Lifetime", "Enhanced Lifetime",
            "Payment Term Lifetime"]
STATUSES = ["Performing", "Performing", "Performing", "Performing", "Redeemed"]
STAGES = ["KFI", "APPLICATION", "OFFER", "COMPLETED", "WITHDRAWN"]
PORTFOLIOS = [("direct_001", "direct"), ("acquired_001", "acquired")]


def _month_ends(n: int, last: date) -> List[date]:
    """``n`` month-end dates ending at ``last`` (oldest first)."""
    out, cur = [], last
    for _ in range(n):
        out.append(cur)
        first = cur.replace(day=1)
        cur = first - timedelta(days=1)
    return list(reversed(out))


def _weeks(n: int, last: date) -> List[date]:
    return list(reversed([last - timedelta(days=7 * i) for i in range(n)]))


def funded_frame(rng: random.Random, rows: int, cut: date) -> pd.DataFrame:
    """One dated platform canonical, with the columns the MI layer consumes."""
    recs = []
    for i in range(rows):
        pid, ptype = PORTFOLIOS[0] if rng.random() < 0.75 else PORTFOLIOS[1]
        valuation = rng.randrange(180_000, 1_400_000)
        balance = round(valuation * rng.uniform(0.12, 0.55), 2)
        orig = cut - timedelta(days=rng.randrange(30, 3650))
        recs.append({
            "loan_identifier": f"L{i:07d}",
            "unique_identifier": f"{CLIENT}-{cut.isoformat()}-{i:07d}",
            "current_outstanding_balance": balance,
            "original_principal_balance": round(balance * rng.uniform(0.7, 1.0), 2),
            "current_valuation_amount": valuation,
            "original_valuation_amount": round(valuation * rng.uniform(0.8, 1.0), 2),
            "current_loan_to_value": round(balance / valuation, 4),
            "current_interest_rate": round(rng.uniform(0.028, 0.084), 4),
            "origination_date": orig.isoformat(),
            "reporting_date": cut.isoformat(),
            "data_cut_off_date": cut.isoformat(),
            "account_status": rng.choice(STATUSES),
            "geographic_region_obligor": rng.choice(REGIONS),
            "geographic_region_collateral": rng.choice(REGIONS),
            "broker_channel": rng.choice(BROKERS),
            "origination_channel": rng.choice(BROKERS),
            "product_type": rng.choice(PRODUCTS),
            "erm_product_type": rng.choice(PRODUCTS),
            "youngest_borrower_age": rng.randrange(55, 92),
            "protected_equity_percentage": round(rng.uniform(0.0, 0.5), 3),
            "source_portfolio_id": pid,
            "source_portfolio_type": ptype,
            "source_portfolio_label": pid.replace("_", " ").title(),
            "client_id": CLIENT,
        })
    return pd.DataFrame.from_records(recs)


def pipeline_frame(rng: random.Random, rows: int, extract: date) -> pd.DataFrame:
    """One weekly pipeline snapshot in the published RAW-header shape.

    Case ids are STABLE across weeks (``CASE{n}``) so the cohort/completion
    model has real week-over-week timelines to track — which is precisely the
    work the historical-model build does.
    """
    recs = []
    for i in range(rows):
        loan = rng.randrange(40_000, 500_000)
        value = round(loan * rng.uniform(2.0, 7.0), 2)
        recs.append({
            "Company": CLIENT,
            "Pool": "Main",
            "Account Number": f"CASE{i:06d}",
            "KFI Number": f"KFI{i:06d}",
            "Broker": rng.choice(BROKERS),
            "KFI Submitted Date": (extract - timedelta(days=rng.randrange(7, 200))).isoformat(),
            "DOB App 1": (extract - timedelta(days=rng.randrange(20000, 32000))).isoformat(),
            "Gender APP 1": rng.choice(["M", "F"]),
            "Loan Amount": loan,
            "Estimated Value": value,
            "Property Value": value,
            "Product": rng.choice(PRODUCTS),
            "Product Rate": round(rng.uniform(0.03, 0.09), 4),
            "Property Region": rng.choice(REGIONS),
            "Status": rng.choices(STAGES, weights=[40, 25, 20, 10, 5])[0],
            "Application Submitted Date": (extract - timedelta(days=rng.randrange(5, 150))).isoformat(),
            "Offer Date": (extract - timedelta(days=rng.randrange(1, 90))).isoformat(),
            "Date Funds Released": "",
            "Loan Purpose": rng.choice(["Home improvements", "Debt consolidation",
                                        "Gifting", "Holiday", "Other"]),
        })
    return pd.DataFrame.from_records(recs)


def build(root: Path, *, months: int, funded_rows: int,
          weeks: int, pipeline_rows: int, seed: int = 20260802) -> dict:
    rng = random.Random(seed)
    base = root / PROCESSED
    last_month = date(2026, 7, 31)
    last_week = date(2026, 8, 7)

    platform_dir = base / "platform" / CLIENT
    for cut in _month_ends(months, last_month):
        frame = funded_frame(rng, funded_rows, cut)
        target = platform_dir / cut.isoformat()
        target.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target / PLATFORM_CANONICAL_NAME, index=False)
    latest = platform_dir / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    frame.to_csv(latest / PLATFORM_CANONICAL_NAME, index=False)

    pipeline_dir = base / "pipeline" / CLIENT
    for extract in _weeks(weeks, last_week):
        pframe = pipeline_frame(rng, pipeline_rows, extract)
        target = pipeline_dir / extract.isoformat()
        target.mkdir(parents=True, exist_ok=True)
        pframe.to_csv(target / PIPELINE_SNAPSHOT_NAME, index=False)
    platest = pipeline_dir / "latest"
    platest.mkdir(parents=True, exist_ok=True)
    pframe.to_csv(platest / PIPELINE_SNAPSHOT_NAME, index=False)

    return {
        "root": str(root),
        "months": months, "funded_rows": funded_rows,
        "weeks": weeks, "pipeline_rows": pipeline_rows,
        "platform_root": f"blob://{PROCESSED}/platform/{CLIENT}",
        "pipeline_root": f"blob://{PROCESSED}/pipeline/{CLIENT}",
        "platform_uri": f"blob://{PROCESSED}/platform/{CLIENT}/latest/{PLATFORM_CANONICAL_NAME}",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--funded-rows", type=int, default=5000)
    ap.add_argument("--weeks", type=int, default=26)
    ap.add_argument("--pipeline-rows", type=int, default=1500)
    args = ap.parse_args()
    info = build(Path(args.root), months=args.months, funded_rows=args.funded_rows,
                 weeks=args.weeks, pipeline_rows=args.pipeline_rows)
    for k, v in info.items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
