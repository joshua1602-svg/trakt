#!/usr/bin/env python3
"""
Deterministic generator for two *non-equity-release* synthetic funded loan tapes.

Purpose
-------
Trakt is an Equity Release Mortgage (ERM) / ESMA-Annex-2 platform. These two
portfolios exist to stress-test how "hardened" the existing registries and
onboarding infrastructure are when a **new asset class** is presented:

  * ``auto_finance``       — a UK auto-finance lender (HP / PCP), secured on
                             motor vehicles (ESMA Annex 5 territory).
  * ``unsecured_consumer`` — a UK unsecured consumer lender (personal loans /
                             debt consolidation), no collateral (ESMA Annex 6
                             territory).

Both tapes are RAW lender extracts: header names are deliberately the sort of
strings a real lender would ship, chosen so that the *core* loan economics and
the *credit-risk block* resolve through the deterministic aligner, while the
asset-class-specific attributes (vehicle make/model/mileage/VIN; consumer
affordability/DTI/residential-status) surface the registry coverage gaps.

The data is fully synthetic and internally consistent:
  * current balance <= original balance (amortising);
  * arrears loans carry days-past-due > 0 and a worse IFRS 9 stage;
  * defaulted loans carry an allocated loss and a recovery;
  * PD / LGD rise with IFRS 9 stage; auto collateral depreciates with age/mileage.

Run ``python synthetic_portfolios/generate_portfolios.py`` to (re)write the CSVs.
Output is deterministic (fixed seed) so the committed tapes and the tests stay
in lock-step.
"""
from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20260704  # fixed → deterministic tapes
CUTOFF = date(2026, 1, 31)  # reporting / data cut-off date


def _iso(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def _add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    day = min(d.day, 28)
    return date(y, m, day)


# ---------------------------------------------------------------------------
# Credit-risk helpers (shared by both asset classes)
# ---------------------------------------------------------------------------

def _ifrs9_stage(dpd: int, defaulted: bool) -> str:
    if defaulted or dpd >= 90:
        return "Stage 3"
    if dpd >= 30:
        return "Stage 2"
    return "Stage 1"


def _pd_for_stage(stage: str, rng: random.Random) -> float:
    base = {"Stage 1": (0.004, 0.030),
            "Stage 2": (0.060, 0.180),
            "Stage 3": (0.550, 0.950)}[stage]
    return round(rng.uniform(*base), 4)


def _grade_for_stage(stage: str, rng: random.Random) -> str:
    pool = {"Stage 1": ["A1", "A2", "A3", "B1", "B2"],
            "Stage 2": ["B3", "C1", "C2"],
            "Stage 3": ["D1", "D2", "E1"]}[stage]
    return rng.choice(pool)


def _credit_score(stage: str, rng: random.Random) -> int:
    # UK bureau-style score band (0-999); worse stage → lower score.
    band = {"Stage 1": (720, 960),
            "Stage 2": (560, 719),
            "Stage 3": (300, 559)}[stage]
    return rng.randint(*band)


# ---------------------------------------------------------------------------
# AUTO FINANCE — HP / PCP secured on motor vehicles
# ---------------------------------------------------------------------------

AUTO_HEADERS = [
    # --- identity / dates / structure (expected to map) ---
    "Loan Identifier", "Original Obligor Identifier", "Data Cut Off Date",
    "Origination Date", "Maturity Date", "Original Term",
    "Agreement Type",  # HP / PCP / Lease Purchase  (GAP: no canonical field)
    # --- economics (expected to map) ---
    "Original Principal Balance", "Current Principal Balance",
    "Current Interest Rate", "Interest Rate Type",
    "Deposit Amount",
    "Balloon Payment",       # PCP GFV        (GAP)
    "Monthly Instalment",    # (GAP)
    "Payment Frequency", "Purpose", "Origination Channel", "Currency",
    # --- performance / arrears / default (expected to map) ---
    "Account Status", "Days Past Due", "Arrears Balance",
    "Default Date", "Default Amount", "Allocated Losses", "Cumulative Recoveries",
    # --- credit-risk metrics (expected to map to the common risk block) ---
    "IFRS9 Stage", "Internal Risk Grade", "Credit Score",
    "Probability of Default", "Loss Given Default", "Exposure at Default",
    # --- borrower (mixed: income/employment/geo map; age does not) ---
    "Employment Status", "Primary Income", "Borrower Age",
    "Geographic Region - Obligor", "Postcode",
    # --- vehicle collateral (value/LTV/new-used map; make/model/etc. GAP) ---
    "Collateral Type", "New or Used",
    "Original Valuation Amount", "Current Valuation Amount",
    "Original Loan-To-Value", "Current Loan-To-Value",
    "Vehicle Make", "Vehicle Model", "Vehicle Registration Year",
    "Mileage", "Vehicle Identification Number", "Fuel Type",
]

_AUTO_VEHICLES = [
    ("Volkswagen", "Golf", "Petrol", 24000, 12000),
    ("Ford", "Focus", "Petrol", 21000, 10500),
    ("BMW", "3 Series", "Diesel", 38000, 19000),
    ("Tesla", "Model 3", "Electric", 45000, 27000),
    ("Toyota", "Corolla", "Hybrid", 27000, 14000),
    ("Audi", "A4", "Diesel", 39000, 18500),
    ("Nissan", "Qashqai", "Petrol", 26000, 12500),
    ("Kia", "Sportage", "Hybrid", 31000, 16000),
    ("Vauxhall", "Corsa", "Petrol", 18000, 8500),
    ("Mercedes-Benz", "A-Class", "Petrol", 34000, 17000),
]
_AUTO_REGIONS = ["North West", "South East", "London", "West Midlands",
                 "Yorkshire", "Scotland", "Wales", "East of England",
                 "South West", "North East"]
_AUTO_POSTCODES = ["M1 4WX", "GU1 1AA", "EC1A 1BB", "B1 1AA", "LS1 2AB",
                   "EH1 1AA", "CF10 1AA", "CB1 1AA", "BS1 1AA", "NE1 1AA"]
_AUTO_CHANNELS = ["Dealer", "Broker", "Direct", "Online"]


def build_auto(n: int = 26) -> list[dict]:
    rng = random.Random(SEED)
    rows = []
    for i in range(1, n + 1):
        lid = f"AUTO-{i:04d}"
        agreement = rng.choices(["HP", "PCP", "Lease Purchase"], weights=[5, 4, 1])[0]
        term = rng.choice([36, 48, 60])
        age_m = rng.randint(3, term - 3)
        orig = _add_months(CUTOFF, -age_m)
        maturity = _add_months(orig, term)

        make, model, fuel, new_price, used_price = rng.choice(_AUTO_VEHICLES)
        is_new = rng.random() < 0.45
        new_or_used = "New" if is_new else "Used"
        veh_orig_val = float(new_price if is_new else round(used_price * rng.uniform(0.9, 1.1)))
        # Depreciation ~1.4%/month, floored.
        depr = max(0.30, 1.0 - 0.014 * age_m)
        veh_curr_val = round(veh_orig_val * depr, 2)

        deposit = round(veh_orig_val * rng.uniform(0.05, 0.20), 2)
        original_principal = round(veh_orig_val - deposit, 2)
        balloon = round(veh_curr_val * rng.uniform(0.35, 0.45), 2) if agreement == "PCP" else 0.0

        # Amortise (PCP amortises toward the balloon, HP toward zero).
        frac_paid = age_m / term
        target_end = balloon if agreement == "PCP" else 0.0
        current_principal = round(original_principal - (original_principal - target_end) * frac_paid, 2)

        apr = round(rng.uniform(6.9, 15.9), 2)
        orig_ltv = round(original_principal / veh_orig_val, 4)
        curr_ltv = round(current_principal / veh_curr_val, 4) if veh_curr_val else ""

        # Performance
        roll = rng.random()
        if roll < 0.62:
            dpd, defaulted = 0, False
        elif roll < 0.80:
            dpd, defaulted = rng.randint(1, 29), False
        elif roll < 0.92:
            dpd, defaulted = rng.randint(30, 89), False
        else:
            dpd, defaulted = rng.randint(90, 210), True

        instalment = round((current_principal - target_end) / max(1, term - age_m)
                           + current_principal * apr / 100 / 12, 2)
        arrears_balance = round(instalment * (dpd / 30.0), 2) if dpd else 0.0
        account_status = "Defaulted" if defaulted else ("Arrears" if dpd else "Performing")

        if defaulted:
            default_date = _iso(_add_months(CUTOFF, -rng.randint(1, 6)))
            default_amount = current_principal
            lgd = round(rng.uniform(0.35, 0.65), 4)
            allocated_losses = round(current_principal * lgd, 2)
            cum_recoveries = round(current_principal * rng.uniform(0.10, 0.45), 2)
        else:
            default_date, default_amount, allocated_losses, cum_recoveries = "", 0.0, 0.0, 0.0

        stage = _ifrs9_stage(dpd, defaulted)
        pd = _pd_for_stage(stage, rng)
        if not defaulted:
            lgd = round(rng.uniform(0.30, 0.55), 4)
        ead = current_principal
        grade = _grade_for_stage(stage, rng)
        score = _credit_score(stage, rng)

        emp = rng.choices(["EMPLOYED", "SELF_EMPLOYED", "PART_TIME", "RETIRED"],
                          weights=[6, 2, 1, 1])[0]
        income = rng.randint(22000, 78000)
        b_age = rng.randint(21, 68)
        ridx = rng.randrange(len(_AUTO_REGIONS))

        rows.append({
            "Loan Identifier": lid,
            "Original Obligor Identifier": f"OBL-AUTO-{i:04d}",
            "Data Cut Off Date": _iso(CUTOFF),
            "Origination Date": _iso(orig),
            "Maturity Date": _iso(maturity),
            "Original Term": term,
            "Agreement Type": agreement,
            "Original Principal Balance": f"{original_principal:.2f}",
            "Current Principal Balance": f"{current_principal:.2f}",
            "Current Interest Rate": f"{apr:.2f}",
            "Interest Rate Type": "Fixed",
            "Deposit Amount": f"{deposit:.2f}",
            "Balloon Payment": f"{balloon:.2f}",
            "Monthly Instalment": f"{instalment:.2f}",
            "Payment Frequency": "Monthly",
            "Purpose": "Vehicle Purchase",
            "Origination Channel": _AUTO_CHANNELS[rng.randrange(len(_AUTO_CHANNELS))],
            "Currency": "GBP",
            "Account Status": account_status,
            "Days Past Due": dpd,
            "Arrears Balance": f"{arrears_balance:.2f}",
            "Default Date": default_date,
            "Default Amount": f"{default_amount:.2f}",
            "Allocated Losses": f"{allocated_losses:.2f}",
            "Cumulative Recoveries": f"{cum_recoveries:.2f}",
            "IFRS9 Stage": stage,
            "Internal Risk Grade": grade,
            "Credit Score": score,
            "Probability of Default": f"{pd:.4f}",
            "Loss Given Default": f"{lgd:.4f}",
            "Exposure at Default": f"{ead:.2f}",
            "Employment Status": emp,
            "Primary Income": income,
            "Borrower Age": b_age,
            "Geographic Region - Obligor": _AUTO_REGIONS[ridx],
            "Postcode": _AUTO_POSTCODES[ridx],
            "Collateral Type": "Motor Vehicle",
            "New or Used": new_or_used,
            "Original Valuation Amount": f"{veh_orig_val:.2f}",
            "Current Valuation Amount": f"{veh_curr_val:.2f}",
            "Original Loan-To-Value": f"{orig_ltv:.4f}",
            "Current Loan-To-Value": f"{curr_ltv:.4f}" if curr_ltv != "" else "",
            "Vehicle Make": make,
            "Vehicle Model": model,
            "Vehicle Registration Year": orig.year,
            "Mileage": (rng.randint(0, 200) if is_new else rng.randint(8000, 78000)),
            "Vehicle Identification Number": f"WVW{rng.randint(10**11, 10**12 - 1)}",
            "Fuel Type": fuel,
        })
    return rows


# ---------------------------------------------------------------------------
# UNSECURED CONSUMER — personal loans / debt consolidation (no collateral)
# ---------------------------------------------------------------------------

CONSUMER_HEADERS = [
    "Loan Identifier", "Original Obligor Identifier", "Data Cut Off Date",
    "Origination Date", "Maturity Date", "Original Term",
    "Product Type",           # Personal Loan / Debt Consolidation / Retail Finance (GAP)
    "Secured / Unsecured",    # Unsecured  (GAP — documents the unsecured nature)
    "Original Principal Balance", "Current Principal Balance",
    "Current Interest Rate", "Interest Rate Type",
    "Monthly Instalment",     # (GAP)
    "Payment Frequency", "Purpose", "Origination Channel", "Currency",
    "Account Status", "Days Past Due", "Arrears Balance",
    "Default Date", "Default Amount", "Allocated Losses", "Cumulative Recoveries",
    "IFRS9 Stage", "Internal Risk Grade", "Credit Score",
    "Probability of Default", "Loss Given Default", "Exposure at Default",
    "Employment Status", "Primary Income", "Borrower Age",
    "Geographic Region - Obligor", "Postcode",
    "Debt To Income Ratio",           # maps to debt_to_income_ratio
    "Number of Dependents",           # (GAP)
    "Residential Status",             # Homeowner / Tenant / Living with family (GAP)
    "Affordability Assessment Result",  # (GAP)
]

_CONSUMER_PURPOSES = ["Debt Consolidation", "Home Improvement", "Car Purchase",
                      "Wedding", "Holiday", "Medical", "Other"]
_CONSUMER_PRODUCTS = ["Personal Loan", "Debt Consolidation Loan", "Retail Finance"]
_CONSUMER_CHANNELS = ["Online", "Broker", "Branch", "Price Comparison"]
_CONSUMER_REGIONS = _AUTO_REGIONS
_CONSUMER_POSTCODES = _AUTO_POSTCODES
_RESIDENTIAL = ["Homeowner", "Tenant", "Living with family"]


def build_consumer(n: int = 26) -> list[dict]:
    rng = random.Random(SEED + 1)
    rows = []
    for i in range(1, n + 1):
        lid = f"UCL-{i:04d}"
        term = rng.choice([12, 24, 36, 48, 60])
        age_m = rng.randint(2, term - 2)
        orig = _add_months(CUTOFF, -age_m)
        maturity = _add_months(orig, term)

        original_principal = float(rng.choice([1500, 3000, 5000, 7500, 10000,
                                               12500, 15000, 20000, 25000]))
        frac_paid = age_m / term
        current_principal = round(original_principal * (1 - frac_paid), 2)
        apr = round(rng.uniform(8.9, 34.9), 2)

        roll = rng.random()
        if roll < 0.58:
            dpd, defaulted = 0, False
        elif roll < 0.76:
            dpd, defaulted = rng.randint(1, 29), False
        elif roll < 0.90:
            dpd, defaulted = rng.randint(30, 89), False
        else:
            dpd, defaulted = rng.randint(90, 240), True

        instalment = round(current_principal / max(1, term - age_m)
                           + current_principal * apr / 100 / 12, 2)
        arrears_balance = round(instalment * (dpd / 30.0), 2) if dpd else 0.0
        account_status = "Charged Off" if defaulted else ("Delinquent" if dpd else "Current")

        stage = _ifrs9_stage(dpd, defaulted)
        pd = _pd_for_stage(stage, rng)
        # Unsecured → high LGD.
        lgd = round(rng.uniform(0.75, 0.95), 4) if defaulted else round(rng.uniform(0.65, 0.90), 4)
        ead = current_principal

        if defaulted:
            default_date = _iso(_add_months(CUTOFF, -rng.randint(1, 8)))
            default_amount = current_principal
            allocated_losses = round(current_principal * lgd, 2)
            cum_recoveries = round(current_principal * rng.uniform(0.02, 0.20), 2)
        else:
            default_date, default_amount, allocated_losses, cum_recoveries = "", 0.0, 0.0, 0.0

        grade = _grade_for_stage(stage, rng)
        score = _credit_score(stage, rng)
        emp = rng.choices(["EMPLOYED", "SELF_EMPLOYED", "PART_TIME", "UNEMPLOYED", "RETIRED"],
                          weights=[6, 2, 1, 1, 1])[0]
        income = rng.randint(16000, 65000)
        dti = round(rng.uniform(0.15, 0.55), 3)
        b_age = rng.randint(20, 70)
        ridx = rng.randrange(len(_CONSUMER_REGIONS))

        rows.append({
            "Loan Identifier": lid,
            "Original Obligor Identifier": f"OBL-UCL-{i:04d}",
            "Data Cut Off Date": _iso(CUTOFF),
            "Origination Date": _iso(orig),
            "Maturity Date": _iso(maturity),
            "Original Term": term,
            "Product Type": rng.choice(_CONSUMER_PRODUCTS),
            "Secured / Unsecured": "Unsecured",
            "Original Principal Balance": f"{original_principal:.2f}",
            "Current Principal Balance": f"{current_principal:.2f}",
            "Current Interest Rate": f"{apr:.2f}",
            "Interest Rate Type": "Fixed",
            "Monthly Instalment": f"{instalment:.2f}",
            "Payment Frequency": "Monthly",
            "Purpose": rng.choice(_CONSUMER_PURPOSES),
            "Origination Channel": _CONSUMER_CHANNELS[rng.randrange(len(_CONSUMER_CHANNELS))],
            "Currency": "GBP",
            "Account Status": account_status,
            "Days Past Due": dpd,
            "Arrears Balance": f"{arrears_balance:.2f}",
            "Default Date": default_date,
            "Default Amount": f"{default_amount:.2f}",
            "Allocated Losses": f"{allocated_losses:.2f}",
            "Cumulative Recoveries": f"{cum_recoveries:.2f}",
            "IFRS9 Stage": stage,
            "Internal Risk Grade": grade,
            "Credit Score": score,
            "Probability of Default": f"{pd:.4f}",
            "Loss Given Default": f"{lgd:.4f}",
            "Exposure at Default": f"{ead:.2f}",
            "Employment Status": emp,
            "Primary Income": income,
            "Borrower Age": b_age,
            "Geographic Region - Obligor": _CONSUMER_REGIONS[ridx],
            "Postcode": _CONSUMER_POSTCODES[ridx],
            "Debt To Income Ratio": f"{dti:.3f}",
            "Number of Dependents": rng.randint(0, 4),
            "Residential Status": rng.choice(_RESIDENTIAL),
            "Affordability Assessment Result": rng.choice(["Pass", "Pass", "Pass", "Refer"]),
        })
    return rows


# ---------------------------------------------------------------------------
# Cashflow / performance report (exercises the cashflow domain)
# ---------------------------------------------------------------------------

def build_cashflows(loan_rows: list[dict]) -> list[dict]:
    """One current-period cashflow line per loan (scheduled vs actual)."""
    rng = random.Random(SEED + 7)
    out = []
    for r in loan_rows:
        instal = float(r["Monthly Instalment"])
        dpd = int(r["Days Past Due"])
        # A delinquent loan pays less than scheduled.
        paid_ratio = 0.0 if dpd >= 90 else (rng.uniform(0.3, 0.8) if dpd else 1.0)
        cur = float(r["Current Principal Balance"])
        apr = float(r["Current Interest Rate"])
        sched_int = round(cur * apr / 100 / 12, 2)
        sched_prin = round(max(0.0, instal - sched_int), 2)
        out.append({
            "Loan Identifier": r["Loan Identifier"],
            "Payment Date": _iso(CUTOFF),
            "Scheduled Interest Payment": f"{sched_int:.2f}",
            "Scheduled Principal Payment": f"{sched_prin:.2f}",
            "Actual Interest Paid": f"{sched_int * paid_ratio:.2f}",
            "Actual Principal Paid": f"{sched_prin * paid_ratio:.2f}",
            "Arrears Balance": r["Arrears Balance"],
            "Data Cut Off Date": _iso(CUTOFF),
        })
    return out


def _write_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"  wrote {path.relative_to(HERE.parent)}  ({len(rows)} rows, {len(headers)} cols)")


def main() -> None:
    print("Generating non-equity-release synthetic funded loan tapes…")

    auto = build_auto()
    _write_csv(HERE / "auto_finance" / "auto_finance_funded_loan_tape.csv",
               AUTO_HEADERS, auto)
    _write_csv(HERE / "auto_finance" / "auto_finance_cashflow_report.csv",
               list(build_cashflows(auto)[0].keys()), build_cashflows(auto))

    consumer = build_consumer()
    _write_csv(HERE / "unsecured_consumer" / "unsecured_consumer_funded_loan_tape.csv",
               CONSUMER_HEADERS, consumer)
    _write_csv(HERE / "unsecured_consumer" / "unsecured_consumer_cashflow_report.csv",
               list(build_cashflows(consumer)[0].keys()), build_cashflows(consumer))

    print("Done.")


if __name__ == "__main__":
    main()
