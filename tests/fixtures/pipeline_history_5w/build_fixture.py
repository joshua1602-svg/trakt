#!/usr/bin/env python3
"""tests/fixtures/pipeline_history_5w/build_fixture.py

Generates a DETERMINISTIC five-week governed pipeline history.

Why it exists: `pipeline_evolution` had **zero** periods against the demo store,
so seven of `evolution`'s thirty-two owned questions and two of its three route
identities could not be exercised at all. An equivalence measured there would be
green and meaningless — refusal compared against refusal.

Written in the CANONICAL M2L/KFI weekly-extract schema, discovered by
`pipeline_contract._PIPELINE_SOURCE_GLOBS` like any other governed extract, and
prepared by the ordinary `prepare_pipeline_mi_dataset`. **No production branch,
alias or validation change exists for it.** If this fixture ever stops being
readable by the shipped prep layer, that is a real finding, not a fixture bug.

THE MOVEMENTS ARE THE POINT. Eight cases across five Fridays:

    case     w1 05-01      w2 05-08      w3 05-15      w4 05-22      w5 05-29
    2001     KFI           APPLICATION   OFFER         COMPLETED     COMPLETED
    2002     KFI           KFI           APPLICATION   APPLICATION   OFFER
    2003     OFFER         OFFER         OFFER         OFFER         OFFER
    2004     APPLICATION   APPLICATION   APPLICATION   WITHDRAWN     WITHDRAWN
    2005     -             KFI           KFI           APPLICATION   APPLICATION
    2006     KFI           KFI           KFI           KFI           KFI
    2007     APPLICATION   OFFER         OFFER         COMPLETED     COMPLETED
    2008     -             -             KFI           KFI           OFFER

    2001, 2007  progress the whole funnel and COMPLETE
    2002, 2008  progress partway
    2003, 2006  stay put (one late, one early)
    2004        leaves the pipeline (WITHDRAWN)
    2005, 2008  ENTER after week one

Stage counts follow arithmetically and are asserted in
`tests/test_pipeline_history_fixture.py` — the fixture is only useful if a
failure is inspectable.

    python tests/fixtures/pipeline_history_5w/build_fixture.py
"""
from __future__ import annotations

import csv
from pathlib import Path

WEEKS = ("2026-05-01", "2026-05-08", "2026-05-15", "2026-05-22", "2026-05-29")

#: `Status` is the governed stage source; `pipeline_prep._STAGE_CANON` maps it.
#: `None` means the case is not in that week's extract at all.
STAGES = {
    "2001": ("KFI", "Application", "Offer", "Completed", "Completed"),
    "2002": ("KFI", "KFI", "Application", "Application", "Offer"),
    "2003": ("Offer", "Offer", "Offer", "Offer", "Offer"),
    "2004": ("Application", "Application", "Application", "Withdrawn", "Withdrawn"),
    "2005": (None, "KFI", "KFI", "Application", "Application"),
    "2006": ("KFI", "KFI", "KFI", "KFI", "KFI"),
    "2007": ("Application", "Offer", "Offer", "Completed", "Completed"),
    "2008": (None, None, "KFI", "KFI", "Offer"),
}

#: Round and distinct, so a wrong subtotal names the case that caused it.
LOAN_AMOUNT = {"2001": 100000, "2002": 200000, "2003": 300000, "2004": 400000,
               "2005": 500000, "2006": 600000, "2007": 700000, "2008": 800000}

REGION = {"2001": "North West", "2002": "London", "2003": "South East",
          "2004": "Wales", "2005": "North West", "2006": "London",
          "2007": "South East", "2008": "Scotland"}

HEADER = ["Company", "Pool", "Account Number", "KFI Number", "Broker",
          "KFI Submitted Date", "DOB App 1", "Gender APP 1", "DOB App 2",
          "Gender APP 2", "Loan Amount", "Estimated Value", "Product",
          "Product Rate", "Loan Plan", "Facility", "Max Facility",
          "Max Entitlement", "Property Region", "PEG Percentage", "Fees Added",
          "Property Value", "Loan Purpose", "Loan Purpose Detail", "Status",
          "DPR Status", "Application Submitted Date", "Offer Date",
          "Date Funds Released", "Rejection Reason A", "Rejection Reason B",
          "KFI Used For App", "Contracted Payment Period",
          "Interest Payment Percentage"]

_DPR = {"Withdrawn": "Declined", "Completed": "Approved"}


def _row(case: str, status: str, week: str) -> list:
    amount = LOAN_AMOUNT[case]
    value = amount * 2                      # a fixed LTV of 50%, so WA LTV is stable
    return ["Synthetic Lender Ltd", "POOL_S", f"ACC{case}", f"KFI{case}",
            "Broker Synthetic", "2026-04-03",
            "1950-01-15", "M", "1952-02-20", "F",
            f"{amount}.00", f"{value}.00", "Lifetime Mortgage Lump Sum", "6.50",
            "Plan S", f"{amount}.00", f"{amount + 50000}.00", f"{value}.00",
            REGION[case], "25.0", "0.00", f"{value}.00",
            "Home Improvements", "Synthetic case", status,
            _DPR.get(status, "Pending"),
            "2026-04-10" if status != "KFI" else "",
            "2026-04-24" if status in ("Offer", "Completed") else "",
            week if status == "Completed" else "",
            "Withdrawn by applicant" if status == "Withdrawn" else "", "",
            "Yes", "12", "0"]


def main() -> int:
    here = Path(__file__).resolve().parent
    written = []
    for i, week in enumerate(WEEKS):
        folder = here / week
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"M2L_KFI_and_Pipeline_{week.replace('-', '_')}.csv"
        rows = [_row(c, s[i], week) for c, s in sorted(STAGES.items())
                if s[i] is not None]
        with target.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(HEADER)
            w.writerows(rows)
        written.append((week, len(rows)))
    for week, n in written:
        print(f"  {week}  {n} cases")
    print(f"{len(written)} weekly extracts written under {here}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
