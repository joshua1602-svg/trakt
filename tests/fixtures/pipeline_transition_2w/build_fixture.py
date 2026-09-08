#!/usr/bin/env python3
"""tests/fixtures/pipeline_transition_2w/build_fixture.py

A DETERMINISTIC two-snapshot governed pipeline pair that exercises every event
class the stage-transition capability publishes.

Why a new fixture rather than an extension of ``pipeline_history_5w``: that pack
is the pinned input for pipeline evolution, funnel and stage-stock assertions,
so adding cases to it would change existing governed outputs. This sprint is
additive, so the new proof gets its own pack and nothing already asserted moves.

Written in the CANONICAL M2L/KFI weekly-extract schema, discovered by
``pipeline_contract._PIPELINE_SOURCE_GLOBS`` like any other governed extract, and
prepared by the ordinary ``prepare_pipeline_mi_dataset``. **No production branch,
alias or validation change exists for it.**

THE TRANSITIONS ARE THE POINT. Fourteen cases across two Fridays:

    case   prior (06-05)        latest (06-12)       classification
    3001   KFI        100k      KFI        100k      stayer, amount unchanged
    3002   KFI        200k      KFI        220k      stayer, amount +20k
    3003   APPLICATION 300k     APPLICATION 280k     stayer, amount -20k
    3004   KFI        400k      APPLICATION 400k     KFI -> APPLICATION
    3005   KFI        500k      APPLICATION 520k     KFI -> APPLICATION, +20k
    3006   APPLICATION 600k     OFFER      600k      APPLICATION -> OFFER
    3007   APPLICATION 700k     OFFER      690k      APPLICATION -> OFFER, -10k
    3008   OFFER      800k      COMPLETED  800k      OFFER -> COMPLETED
    3009   -                    KFI        900k      new arrival into KFI
    3010   -                    APPLICATION 150k     new arrival into APPLICATION
    3011   COMPLETED  1,000k    -                    departure, outcome COMPLETED
    3012   WITHDRAWN  1,100k    -                    departure, outcome WITHDRAWN
    3013   OFFER      1,200k    -                    departure, UNCLASSIFIED
    3014   APPLICATION 1,300k   -                    departure, UNCLASSIFIED

APPLICATION deliberately has movement in BOTH directions in the same window
(two transitions in, one new arrival in, two transitions out, one departure out,
one stayer), which is exactly the case a net figure cannot describe.

Amounts are round and distinct so a wrong subtotal names the case that caused
it. Every expected total in ``test_pipeline_stage_transition.py`` is arithmetic
on this table, not an opinion.

    python tests/fixtures/pipeline_transition_2w/build_fixture.py
"""
from __future__ import annotations

import csv
from pathlib import Path

WEEKS = ("2026-06-05", "2026-06-12")

#: ``(stage, loan amount)`` per week. ``None`` means the case is not in that
#: week's extract at all. ``Status`` is the governed stage source, normalised by
#: ``pipeline_prep._STAGE_CANON``.
CASES = {
    "3001": (("KFI", 100000), ("KFI", 100000)),
    "3002": (("KFI", 200000), ("KFI", 220000)),
    "3003": (("Application", 300000), ("Application", 280000)),
    "3004": (("KFI", 400000), ("Application", 400000)),
    "3005": (("KFI", 500000), ("Application", 520000)),
    "3006": (("Application", 600000), ("Offer", 600000)),
    "3007": (("Application", 700000), ("Offer", 690000)),
    "3008": (("Offer", 800000), ("Completed", 800000)),
    "3009": (None, ("KFI", 900000)),
    "3010": (None, ("Application", 150000)),
    "3011": (("Completed", 1000000), None),
    "3012": (("Withdrawn", 1100000), None),
    "3013": (("Offer", 1200000), None),
    "3014": (("Application", 1300000), None),
}

REGION = {"3001": "North West", "3002": "London", "3003": "South East",
          "3004": "Wales", "3005": "North West", "3006": "London",
          "3007": "South East", "3008": "Scotland", "3009": "North West",
          "3010": "London", "3011": "South East", "3012": "Wales",
          "3013": "Scotland", "3014": "London"}

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


def _row(case: str, status: str, amount: int, week: str) -> list:
    value = amount * 2                      # a fixed LTV of 50%, so WA LTV is stable
    return ["Synthetic Lender Ltd", "POOL_T", f"ACC{case}", f"KFI{case}",
            "Broker Synthetic", "2026-05-08",
            "1950-01-15", "M", "1952-02-20", "F",
            f"{amount}.00", f"{value}.00", "Lifetime Mortgage Lump Sum", "6.50",
            "Plan T", f"{amount}.00", f"{amount + 50000}.00", f"{value}.00",
            REGION[case], "25.0", "0.00", f"{value}.00",
            "Home Improvements", "Synthetic case", status,
            _DPR.get(status, "Pending"),
            "2026-05-15" if status != "KFI" else "",
            "2026-05-29" if status in ("Offer", "Completed") else "",
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
        rows = [_row(c, w[i][0], w[i][1], week)
                for c, w in sorted(CASES.items()) if w[i] is not None]
        with target.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(HEADER)
            writer.writerows(rows)
        written.append((week, len(rows)))
    for week, n in written:
        print(f"  {week}  {n} cases")
    print(f"{len(written)} weekly extracts written under {here}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
