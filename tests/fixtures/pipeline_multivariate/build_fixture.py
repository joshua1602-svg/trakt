#!/usr/bin/env python3
"""tests/fixtures/pipeline_multivariate/build_fixture.py

A DETERMINISTIC governed pipeline pack built for ONE purpose: to make
multivariate pipeline questions DISCRIMINATING.

WHY A NEW PACK. Neither committed pipeline fixture can test the questions the
multivariate audit asks, and the reason is in the data rather than in the agent:

    pipeline_transition_2w   borrower_type  joint 10 of 10   LTV constant 0.5
    pipeline_history_5w      borrower_type  joint  8 of 8    LTV constant 0.5

"What share of Offer pipeline is joint borrowers?" is 100% on both, and "how
much Offer pipeline has LTV above 60%?" is nothing on both. A bank measured
there would grade an agent that silently DROPPED the borrower-type filter as
correct, which is precisely the silent-substitution failure the audit exists to
detect. Neither existing pack is edited: both are pinned inputs for governed
outputs that are already asserted, and this sprint changes no existing
assertion.

WRITTEN IN THE CANONICAL M2L/KFI WEEKLY-EXTRACT SCHEMA, discovered by
``pipeline_contract._PIPELINE_SOURCE_GLOBS`` like any other governed extract and
prepared by the ordinary ``prepare_pipeline_mi_dataset``. **No production
branch, alias or validation change exists for it.** Every governed field the
audit reads is DERIVED by production preparation from these raw columns:

    Status                        -> pipeline_stage        (_normalise_stage)
    DOB App 2 / Gender APP 2      -> borrower_type         (_derive_borrower_type)
    Property Region               -> geographic_region_obligor
    Loan Amount / Property Value  -> current_loan_to_value (_derive_ltv, a RATIO)
    Loan Amount                   -> current_outstanding_balance
    Product Rate                  -> current_interest_rate

THE CROSS-TAB IS THE POINT. Every filter has to change the answer visibly, so a
dropped filter cannot pass as a right answer:

  * borrower type is mixed WITHIN every stage, never uniform;
  * LTV spans 0.32-0.78, so "above 60%" is a strict, non-empty subset of Offer
    that is neither all of it nor none of it;
  * regions are spread so a region filter inside a stage is a real narrowing;
  * balances are distinct round numbers, so a wrong subtotal names the cases
    that caused it.

FOUR EXTRACTS ACROSS A MONTH BOUNDARY (29 May, 5 / 12 / 26 June) so
"compared with the previous month" has a governed May snapshot to mean, and so
the audit can establish what the current Query time semantics actually resolve
it to rather than assuming.

    python tests/fixtures/pipeline_multivariate/build_fixture.py
"""

from __future__ import annotations

import csv
from pathlib import Path

WEEKS = ("2026-05-29", "2026-06-05", "2026-06-12", "2026-06-26")

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

#: ``id: (stage, joint, region, loan, valuation, rate)``
#:
#: LTV is loan/valuation and is what makes the threshold questions real:
#: 400/1250 = 0.32 … 780/1000 = 0.78.
CASES = {
    # ---- OFFER: 10 cases, 6 joint / 4 single, 4 above 60% LTV ------------- #
    "5001": ("Offer", True,  "London",     500_000, 1_000_000, 6.10),  # 0.50
    "5002": ("Offer", True,  "London",     650_000, 1_000_000, 6.25),  # 0.65 >60
    "5003": ("Offer", True,  "South East", 700_000, 1_000_000, 5.95),  # 0.70 >60
    "5004": ("Offer", True,  "North West", 300_000,   750_000, 6.40),  # 0.40
    "5005": ("Offer", True,  "Scotland",   450_000, 1_000_000, 6.55),  # 0.45
    "5006": ("Offer", True,  "Midlands",   360_000,   800_000, 6.05),  # 0.45
    "5007": ("Offer", False, "London",     780_000, 1_000_000, 6.70),  # 0.78 >60
    "5008": ("Offer", False, "South East", 240_000,   750_000, 5.80),  # 0.32
    "5009": ("Offer", False, "Wales",      320_000,   800_000, 6.15),  # 0.40
    "5010": ("Offer", False, "North West", 660_000, 1_000_000, 6.85),  # 0.66 >60

    # ---- APPLICATION: 12 cases, 7 joint / 5 single, 4 in London ---------- #
    "5011": ("Application", True,  "London",     420_000, 1_000_000, 6.20),  # 0.42
    "5012": ("Application", True,  "London",     560_000,   800_000, 6.45),  # 0.70
    "5013": ("Application", True,  "South East", 250_000,   500_000, 5.90),  # 0.50
    "5014": ("Application", True,  "North West", 375_000,   750_000, 6.30),  # 0.50
    "5015": ("Application", True,  "Scotland",   180_000,   500_000, 6.60),  # 0.36
    "5016": ("Application", True,  "Midlands",   455_000,   700_000, 6.05),  # 0.65
    "5017": ("Application", True,  "Wales",      210_000,   600_000, 6.35),  # 0.35
    "5018": ("Application", False, "London",     640_000, 1_000_000, 6.75),  # 0.64
    "5019": ("Application", False, "London",     290_000,   500_000, 5.85),  # 0.58
    "5020": ("Application", False, "South East", 480_000,   750_000, 6.50),  # 0.64
    "5021": ("Application", False, "North West", 155_000,   500_000, 6.15),  # 0.31
    "5022": ("Application", False, "Midlands",   330_000,   600_000, 6.90),  # 0.55

    # ---- KFI: 10 cases, 5 joint / 5 single -------------------------------- #
    "5023": ("KFI", True,  "London",     200_000,   500_000, 6.00),
    "5024": ("KFI", True,  "South East", 340_000,   800_000, 6.25),
    "5025": ("KFI", True,  "North West", 275_000,   500_000, 6.50),
    "5026": ("KFI", True,  "Scotland",   140_000,   400_000, 6.10),
    "5027": ("KFI", True,  "Midlands",   390_000,   600_000, 6.65),
    "5028": ("KFI", False, "London",     520_000,   800_000, 5.95),
    "5029": ("KFI", False, "Wales",      165_000,   500_000, 6.40),
    "5030": ("KFI", False, "South East", 430_000,   700_000, 6.20),
    "5031": ("KFI", False, "North West", 225_000,   600_000, 6.80),
    "5032": ("KFI", False, "Scotland",   310_000,   500_000, 6.05),

    # ---- COMPLETED: 5 cases ---------------------------------------------- #
    "5033": ("Completed", True,  "London",     600_000, 1_000_000, 6.30),
    "5034": ("Completed", True,  "South East", 350_000,   700_000, 6.15),
    "5035": ("Completed", False, "North West", 480_000,   800_000, 6.55),
    "5036": ("Completed", False, "Midlands",   270_000,   600_000, 6.00),
    "5037": ("Completed", True,  "Scotland",   410_000,   750_000, 6.45),

    # ---- WITHDRAWN: 3 cases ---------------------------------------------- #
    "5038": ("Withdrawn", True,  "Wales",      190_000,   500_000, 6.70),
    "5039": ("Withdrawn", False, "London",     540_000,   900_000, 6.20),
    "5040": ("Withdrawn", False, "South East", 260_000,   600_000, 6.35),
}

#: Cases absent from the earlier extracts, so the stage stock GROWS toward the
#: latest week and "compared with the previous month" has something to compare.
#: 29 May carries neither, 5 June carries the first, 12 June both.
LATE_ARRIVALS = {"5011": 2, "5012": 2, "5019": 1, "5002": 1, "5030": 2}

STAGE_ROW = {"KFI": ("KFI", "Pending", "", "", ""),
             "Application": ("Application", "Pending", "2026-05-15", "", ""),
             "Offer": ("Offer", "Pending", "2026-05-15", "2026-05-29", ""),
             "Completed": ("Completed", "Approved", "2026-05-15", "2026-05-29",
                           "2026-06-12"),
             "Withdrawn": ("Withdrawn", "Declined", "2026-05-15", "", "")}


def rows_for(week_index: int):
    out = []
    for case_id, (stage, joint, region, loan, valuation, rate) in CASES.items():
        if LATE_ARRIVALS.get(case_id, 0) > week_index:
            continue
        status, dpr, applied, offered, released = STAGE_ROW[stage]
        rejection = "Withdrawn by applicant" if stage == "Withdrawn" else ""
        out.append([
            "Synthetic Lender Ltd", "POOL_M", "ACC%s" % case_id,
            "KFI%s" % case_id, "Broker Synthetic", "2026-05-01",
            "1950-01-15", "M",
            # THE BORROWER-TYPE SWITCH: a populated second applicant is JOINT.
            "1952-02-20" if joint else "", "F" if joint else "",
            "%.2f" % loan, "%.2f" % valuation, "Lifetime Mortgage Lump Sum",
            "%.2f" % rate, "Plan M", "%.2f" % loan, "%.2f" % (loan + 50_000),
            "%.2f" % valuation, region, "25.0", "0.00", "%.2f" % valuation,
            "Home Improvements", "Synthetic case", status, dpr,
            applied, offered, released, rejection, "", "Yes", "12", "0",
        ])
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    for index, week in enumerate(WEEKS):
        folder = here / week
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / ("M2L_KFI_and_Pipeline_%s.csv" % week.replace("-", "_"))
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(HEADER)
            writer.writerows(rows_for(index))
        print("wrote %s (%d cases)" % (path, len(rows_for(index))))


if __name__ == "__main__":
    main()
