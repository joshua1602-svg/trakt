#!/usr/bin/env python3
"""Generate a synthetic borrower base that exercises the borrowing base.

A lifetime-mortgage (equity release) book shaped so every Schedule 8 limit has
something to say about it: one region over its limit, two tests close enough to
warn, and the rest comfortably inside. The point is a DEMONSTRABLE dashboard,
not a flattering one — a book on which every test passes shows nothing about
whether the tests work.

Deterministic: seeded, so the same command produces the same book and the same
figures every time. Nothing here is a real borrower, a real property or a real
loan; the identifiers are synthetic and the geography is a label, not an
address.

    python scripts/generate_synthetic_borrowing_base.py --out synthetic.csv

The composition targets, as a share of the £148.2m book:

    London + South East   46%   limit 50%   -> WARNING (92% utilised)
    East of England       26%   limit 25%   -> BREACH
    South West             8%   limit 20%
    West Midlands          5%   limit 15%
    East Midlands          4%   limit 15%
    Yorkshire & Humber     4%   limit 15%
    North West             3%   limit 10%
    Scotland               2%   limit 10%
    Wales                  1%   limit 10%
    North East             1%   limit 10%

    Original valuation > £1.5m   9.5%  limit 10%  -> WARNING
    Original valuation < £150k   3.0%  limit 10%
    Borrower aggregate initial principal > £1m  ~6%  limit 10%
    Average initial principal    ~£185k  limit £300k
    Youngest borrower under 55    0%     limit 0%   (equity release: all 55+)
    Two borrowers                ~58%    limit 90%
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

SEED = 20251130
CLIENT_ID = "ere_funding_uk"
REPORTING_DATE = "2025-11-30"

#: Region label -> share of the book. Labels are the Schedule 8 wording, which
#: is what the governed geo_region_share test matches on.
REGION_SHARES: Dict[str, float] = {
    "London": 0.26,
    "South East": 0.20,          # London + South East = 46% against a 50% limit
    "East Of England": 0.26,     # over the 25% limit, on purpose
    "South West": 0.08,
    "West Midlands": 0.05,
    "East Midlands": 0.04,
    "Yorkshire And The Humber": 0.04,
    "North West": 0.03,
    "Scotland": 0.02,
    "Wales": 0.01,
    "North East": 0.01,
}

BOOK_BALANCE = 148_200_000.0
LOAN_COUNT = 1_240

#: Share of the book on properties originally valued above £1.5m / below £150k.
HIGH_VALUE_SHARE = 0.095
LOW_VALUE_SHARE = 0.030

#: Borrowers holding several loans, whose AGGREGATE initial principal passes
#: £1,000,000 while no single loan does. Six groups of five of the book's
#: larger advances puts roughly 6% of the balance into that test's population —
#: enough for the test to be measuring something, well inside its 10% limit.
MULTI_LOAN_BORROWERS = 6
LOANS_PER_MULTI_BORROWER = 5

#: Share of loans with two borrowers (joint lives).
JOINT_SHARE = 0.58

#: Loans deliberately made INELIGIBLE (arrears / LTV) and UNDETERMINED (an
#: eligibility input the book does not carry for them) under a GOVERNED
#: facility. Under the prototype assumption these are eligible like everything
#: else — the flags are here so both states can be demonstrated from one book.
ARREARS_SHARE = 0.022
HIGH_LTV_SHARE = 0.018
MISSING_INPUT_SHARE = 0.030


def _region_column(rng: np.random.Generator, balances: np.ndarray) -> List[str]:
    """Assign regions so each one's BALANCE share hits its target.

    Assigning by loan count would miss: the shares are contractual limits on
    balance, and a region of small loans would land well under its target.
    Loans are walked in descending balance and dealt into whichever region is
    furthest below its target, which converges without an optimiser.
    """
    total = float(balances.sum())
    targets = {r: share * total for r, share in REGION_SHARES.items()}
    running = {r: 0.0 for r in REGION_SHARES}
    order = np.argsort(-balances)
    out: List[str] = [""] * len(balances)
    for idx in order:
        region = max(running, key=lambda r: targets[r] - running[r])
        out[idx] = region
        running[region] += float(balances[idx])
    return out


def build(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- balances -------------------------------------------------------- #
    # Lognormal, then scaled so the book totals exactly the target. Equity
    # release skews small with a long tail, which is what makes the
    # high-value-property test interesting.
    raw = rng.lognormal(mean=11.5, sigma=0.55, size=LOAN_COUNT)
    balances = np.round(raw / raw.sum() * BOOK_BALANCE, 2)
    balances[-1] = round(BOOK_BALANCE - balances[:-1].sum(), 2)

    rows: List[Dict[str, Any]] = []
    regions = _region_column(rng, balances)

    # Which loans carry which characteristic, chosen by BALANCE so the shares
    # are balance shares (which is what the limits measure).
    order = np.argsort(-balances)
    cumulative = np.cumsum(balances[order]) / BOOK_BALANCE
    high_value = set(order[cumulative <= HIGH_VALUE_SHARE].tolist())
    ascending = np.argsort(balances)
    low_cumulative = np.cumsum(balances[ascending]) / BOOK_BALANCE
    low_value = set(ascending[low_cumulative <= LOW_VALUE_SHARE].tolist())

    joint = set(rng.choice(LOAN_COUNT, size=int(LOAN_COUNT * JOINT_SHARE),
                           replace=False).tolist())
    arrears = set(rng.choice(LOAN_COUNT, size=int(LOAN_COUNT * ARREARS_SHARE),
                             replace=False).tolist())
    remaining = [i for i in range(LOAN_COUNT) if i not in arrears]
    high_ltv = set(rng.choice(remaining, size=int(LOAN_COUNT * HIGH_LTV_SHARE),
                              replace=False).tolist())
    remaining = [i for i in remaining if i not in high_ltv]
    missing_input = set(rng.choice(remaining,
                                   size=int(LOAN_COUNT * MISSING_INPUT_SHARE),
                                   replace=False).tolist())

    # Borrowers with several loans, so the AGGREGATE initial principal test has
    # a population that no per-loan test would find. Drawn from the LARGEST
    # advances that are not already in the high-value-property population, so
    # each group's aggregate clears £1,000,000 while no single loan does —
    # which is the whole point of a borrower-level aggregation.
    multi_loan: Dict[int, str] = {}
    pool = [int(i) for i in np.argsort(-balances) if int(i) not in high_value]
    picks = pool[: MULTI_LOAN_BORROWERS * LOANS_PER_MULTI_BORROWER]
    for n, loan_index in enumerate(picks):
        multi_loan[loan_index] = f"BG{n // LOANS_PER_MULTI_BORROWER:03d}"

    origin = _dt.date(2018, 1, 31)
    for i in range(LOAN_COUNT):
        balance = float(balances[i])
        # Equity release rolls up: the current balance exceeds the initial
        # advance by the accrued interest, so the initial principal is a
        # fraction of it. That is also what keeps the AVERAGE initial principal
        # (limit £300k) well inside its limit on a book of larger balances.
        rolled_up_years = float(rng.uniform(1.5, 6.5))
        rate = float(np.round(rng.normal(6.4, 0.55), 3))
        original = float(np.round(balance / (1.0 + rate / 100.0) ** rolled_up_years, 2))

        if i in high_value:
            valuation = float(np.round(rng.uniform(1_550_000, 4_200_000), 0))
        elif i in low_value:
            valuation = float(np.round(rng.uniform(88_000, 148_000), 0))
        else:
            valuation = float(np.round(max(balance * rng.uniform(2.4, 4.2),
                                           160_000), 0))

        borrowers = 2 if i in joint else 1
        # An equity-release borrower is 55 at the very youngest, which is what
        # makes the Schedule 8 under-55 test a genuine 0%.
        youngest = int(rng.integers(58, 88))

        rows.append({
            "loan_id": f"ERE{100000 + i}",
            "client_id": CLIENT_ID,
            "borrower_identifier": multi_loan.get(i, f"B{200000 + i}"),
            "reporting_date": REPORTING_DATE,
            "origination_date": (origin + _dt.timedelta(
                days=int(rng.integers(0, 2200)))).isoformat(),
            "current_outstanding_balance": balance,
            "original_principal_balance": original,
            "original_valuation_amount": valuation,
            "current_valuation_amount": float(np.round(valuation * rng.uniform(
                1.02, 1.22), 0)),
            "collateral_geography": regions[i],
            "number_of_borrowers": borrowers,
            "youngest_borrower_age": youngest,
            "oldest_borrower_age": youngest + (int(rng.integers(0, 9))
                                               if borrowers == 2 else 0),
            "current_interest_rate": rate,
            "interest_rate_type": "FIXED",
            # Eligibility inputs. A blank days_past_due is the honest shape of
            # a book that does not report arrears for every loan — and it is
            # what makes those loans UNDETERMINED rather than eligible.
            "days_past_due": ("" if i in missing_input
                              else (int(rng.integers(45, 210)) if i in arrears
                                    else 0)),
            "account_status": "PERFORMING" if i not in arrears else "ARREARS",
        })

    df = pd.DataFrame(rows)
    # Original LTV drives the second eligibility rule. Set the flagged loans
    # above the threshold by RAISING the advance, not by shrinking the
    # property, so the valuation-based tests are untouched.
    for i in high_ltv:
        df.loc[i, "original_principal_balance"] = float(np.round(
            df.loc[i, "original_valuation_amount"] * 0.62, 2))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="synthetic_borrowing_base.csv",
                        type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    df = build(args.seed)
    df.to_csv(args.out, index=False)

    total = df["current_outstanding_balance"].sum()
    print(f"{len(df):,} loans · £{total:,.0f} · {args.out}")
    shares = (df.groupby("collateral_geography")["current_outstanding_balance"]
              .sum() / total * 100).sort_values(ascending=False)
    for region, share in shares.items():
        print(f"  {region:26s} {share:6.2f}%")
    print(f"  average initial principal  £"
          f"{df['original_principal_balance'].mean():,.0f}")
    agg = (df.groupby("borrower_identifier")["original_principal_balance"]
           .sum())
    big = set(agg[agg > 1_000_000].index)
    print(f"  borrower aggregate > £1m   "
          f"{df.loc[df.borrower_identifier.isin(big), 'current_outstanding_balance'].sum() / total * 100:6.2f}%"
          f"  ({len(big)} borrowers)")
    for name, mask in (("orig valuation > £1.5m",
                        df.original_valuation_amount > 1_500_000),
                       ("orig valuation < £150k",
                        df.original_valuation_amount < 150_000)):
        print(f"  {name:26s} "
              f"{df.loc[mask, 'current_outstanding_balance'].sum() / total * 100:6.2f}%")
    print(f"  two borrowers              "
          f"{df.loc[df.number_of_borrowers == 2, 'current_outstanding_balance'].sum() / total * 100:6.2f}%")
    print(f"  youngest borrower under 55 "
          f"{df.loc[df.youngest_borrower_age < 55, 'current_outstanding_balance'].sum() / total * 100:6.2f}%")


if __name__ == "__main__":
    main()
