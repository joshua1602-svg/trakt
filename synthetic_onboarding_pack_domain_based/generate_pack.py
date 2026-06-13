"""
generate_pack.py
================

Deterministically (re)generate the domain-based synthetic onboarding packs used
by the Onboarding Agent Phase-1 tests and demos.

Two scenarios prove the engine reasons about DOMAINS, not files:

  scenario_a_combined/  - loan + borrower + collateral in ONE master tape
  scenario_b_split/     - loan / collateral split across files

Both carry the same seeded edge cases:
  * clean overlap        : original_principal / loan_amount / initial_advance agree
  * balance overlap      : current_balance vs cashflow principal_outstanding agree
  * material conflict     : L0003 / L0005 / L0006 balances differ vs cashflow
  * collateral in tape   : property_post_code / collateral_region / valuation_amount
  * pipeline mix         : some applications linked to funded loans, some not
  * missing values       : L0005 valuation, L0006 interest_rate
  * enum issue           : L0004 employment_status = manual
  * reporting date drift : master tape 2026-01-31, cashflow 2026-02-01
  * ambiguous field names: loan amount / initial advance / original principal
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

# loan_id, borrower_name, employment_status, current_balance, original_principal,
# loan_amount, initial_advance, origination_date, maturity_date, interest_rate,
# property_type, property_post_code, collateral_region, valuation_amount,
# current_ltv
LOANS = [
    ("L0001", "Borrower One",   "EMPLOYED",      "148250.55", "150000.00", "2021-03-14", "2046-03-14", "4.85", "Detached house", "SW1A 2AA", "North West",    "295000.00", "0.50"),
    ("L0002", "Borrower Two",   "RETIRED",       "96120.10",  "100000.00", "2020-07-02", "2045-07-02", "4.60", "Flat",           "M1 4WX",   "South East",    "210000.00", "0.46"),
    ("L0003", "Borrower Three", "SELF_EMPLOYED", "210400.00", "215000.00", "2019-11-20", "2044-11-20", "5.10", "Semi-detached",  "B1 1AA",   "West Midlands", "410000.00", "0.51"),
    ("L0004", "Borrower Four",  "manual",        "75300.25",  "80000.00",  "2022-01-10", "2047-01-10", "4.95", "Terraced",       "LS1 2AB",  "Yorkshire",     "165000.00", "0.46"),
    ("L0005", "Borrower Five",  "EMPLOYED",      "305000.00", "300000.00", "2018-05-05", "2043-05-05", "5.25", "Detached house", "EH1 1AA",  "Scotland",      "",          "0.55"),
    ("L0006", "Borrower Six",   "RETIRED",       "52000.00",  "55000.00",  "2023-02-14", "2048-02-14", "",     "Bungalow",       "CF10 1AA", "Wales",         "130000.00", "0.40"),
    ("L0007", "Borrower Seven", "EMPLOYED",      "188000.00", "190000.00", "2020-09-30", "2045-09-30", "4.75", "Flat",           "NE1 1AA",  "North East",    "350000.00", "0.54"),
    ("L0008", "Borrower Eight", "PART_TIME",     "99000.00",  "100000.00", "2021-12-01", "2046-12-01", "4.80", "Detached house", "BS1 1AA",  "South West",    "220000.00", "0.45"),
]

# loan_id -> cashflow principal_outstanding (conflicts on L0003/L0005/L0006).
CASHFLOW_BALANCE = {
    "L0001": "148250.55", "L0002": "96120.10", "L0003": "195000.00",
    "L0004": "75300.25", "L0005": "301500.00", "L0006": "52650.00",
    "L0007": "188000.00", "L0008": "99000.00",
}

MASTER_DATE = "2026-01-31"
CASHFLOW_DATE = "2026-02-01"

# application_id, broker, stage, offer_date, expected_completion_date,
# expected_funding_amount, requested_loan_amount, linked_loan_id, product_name
PIPELINE = [
    ("A1001", "Broker Alpha", "Offer Issued",  "2026-01-10", "2026-02-20", "175000.00", "175000.00", "L0001", "Lifetime Mortgage Lump Sum"),
    ("A1002", "Broker Beta",  "Application",   "2026-01-12", "2026-03-01", "90000.00",  "90000.00",  "",      "Lifetime Mortgage Drawdown"),
    ("A1003", "Broker Gamma", "Underwriting",  "2026-01-15", "2026-03-10", "120000.00", "120000.00", "",      "Lifetime Mortgage Lump Sum"),
    ("A1004", "Broker Alpha", "Offer Issued",  "2026-01-20", "2026-02-28", "200000.00", "205000.00", "L0002", "Lifetime Mortgage Lump Sum"),
]

WAREHOUSE_MD = """# Warehouse Funding Agreement (synthetic)

This is a synthetic warehouse / funding agreement for onboarding tests.

- Warehouse facility present: Yes
- Warehouse lender name: Synthetic Funding Bank plc
- Facility limit: GBP 50,000,000
- Advance rate: 85%
- Margin: 2.25% over SONIA
- Interest index: SONIA
- Availability period: 24 months
- Reporting frequency: Monthly

Eligibility criteria summary: UK lifetime mortgages secured on residential
property, maximum LTV 60% at origination.
"""


def _write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _cashflow_rows():
    rows = []
    for loan in LOANS:
        lid = loan[0]
        bal = CASHFLOW_BALANCE[lid]
        # scheduled interest ~ balance * rate/12 (synthetic, not exact)
        rows.append([lid, CASHFLOW_DATE, bal, "599.01", "420.10", "12.00",
                     "1031.11", "0.00", CASHFLOW_DATE])
    return rows


def _pipeline_rows():
    return [list(p) for p in PIPELINE]


def generate(base: Path = HERE) -> None:
    # ---- Scenario A: combined master loan + collateral tape ----
    a = base / "scenario_a_combined"
    master_header = [
        "loan_id", "borrower_name", "employment_status", "current_balance",
        "original_principal", "loan_amount", "initial_advance", "origination_date",
        "maturity_date", "interest_rate", "property_type", "property_post_code",
        "collateral_region", "valuation_amount", "current_ltv", "reporting_date",
    ]
    master_rows = []
    for (lid, bname, emp, cbal, orig, odate, mdate, rate, ptype, pcode, region,
         val, ltv) in LOANS:
        master_rows.append([
            lid, bname, emp, cbal, orig, orig, orig, odate, mdate, rate, ptype,
            pcode, region, val, ltv, MASTER_DATE,
        ])
    _write_csv(a / "master_loan_collateral_tape.csv", master_header, master_rows)
    _write_csv(
        a / "cashflow_report.csv",
        ["loan_id", "payment_date", "principal_outstanding", "scheduled_interest_payment",
         "scheduled_principal_payment", "fee_amount", "total_cashflow",
         "redemption_amount", "reporting_date"],
        _cashflow_rows(),
    )
    _write_csv(
        a / "pipeline_report.csv",
        ["application_id", "broker_name", "application_stage", "offer_date",
         "expected_completion_date", "expected_funding_amount", "requested_loan_amount",
         "linked_loan_id", "product_name"],
        _pipeline_rows(),
    )
    (a / "warehouse_funding_agreement.md").write_text(WAREHOUSE_MD, encoding="utf-8")

    # ---- Scenario B: split loan and collateral ----
    b = base / "scenario_b_split"
    loan_header = [
        "loan_id", "borrower_name", "employment_status", "current_balance",
        "original_principal", "loan_amount", "initial_advance", "origination_date",
        "maturity_date", "interest_rate", "reporting_date",
    ]
    loan_rows = []
    collateral_rows = []
    for (lid, bname, emp, cbal, orig, odate, mdate, rate, ptype, pcode, region,
         val, ltv) in LOANS:
        loan_rows.append([lid, bname, emp, cbal, orig, orig, orig, odate, mdate,
                          rate, MASTER_DATE])
        collateral_rows.append([f"C{lid[1:]}", lid, ptype, pcode, region, val, ltv,
                                MASTER_DATE])
    _write_csv(b / "loan_report.csv", loan_header, loan_rows)
    _write_csv(
        b / "collateral_report.csv",
        ["collateral_id", "loan_id", "property_type", "property_post_code",
         "collateral_region", "valuation_amount", "current_ltv", "reporting_date"],
        collateral_rows,
    )
    _write_csv(
        b / "cashflow_report.csv",
        ["loan_id", "payment_date", "principal_outstanding", "scheduled_interest_payment",
         "scheduled_principal_payment", "fee_amount", "total_cashflow",
         "redemption_amount", "reporting_date"],
        _cashflow_rows(),
    )
    _write_csv(
        b / "pipeline_report.csv",
        ["application_id", "broker_name", "application_stage", "offer_date",
         "expected_completion_date", "expected_funding_amount", "requested_loan_amount",
         "linked_loan_id", "product_name"],
        _pipeline_rows(),
    )
    (b / "warehouse_funding_agreement.md").write_text(WAREHOUSE_MD, encoding="utf-8")


if __name__ == "__main__":
    generate()
    print(f"Synthetic domain packs written under {HERE}")
