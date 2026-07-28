#!/usr/bin/env python3
"""build_multibook_input.py — the multi-book, multi-period synthetic source data.

SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.

Produces raw source extracts for three books across two month-ends, in the same
source schema the single-book demo already uses, so the real pipeline consumes
them unchanged.

Book identities mirror ``demo_platform.config`` exactly — same slugs, same
provenance types, same balance-sheet statuses — so the film and the web
demonstration describe one platform in one vocabulary:

    alp_origination   direct    warehoused   ALP Origination Book
    alp_acquired      acquired  warehoused   ALP Acquired Back Book
    spv1_sponsored    direct    SOLD         SPV1 Sponsored Securitisation

The third book is the point of the exercise. It was originated by the sponsor
and then securitised and sold, so it is off balance sheet — but the sponsor
retains servicing, risk retention and investor reporting on it. It is therefore
governed alongside the platform and aggregates with it only in the *sponsor*
view. Two totals are legitimate at once, and the demonstration says which is
which rather than quietly picking one.

Modelled, not sampled
---------------------
Each loan is generated once — completion date, region, valuation, advance, fixed
roll-up rate, youngest-borrower age — and every reported figure at a month-end is
*derived* from that loan's own economics:

    balance(d)   = advance x (1 + r/12) ^ months_since_origination
    valuation(d) = original valuation x (1 + hpi/12) ^ months_since_origination
    LTV(d)       = balance(d) / valuation(d)
    age(d)       = age at completion + whole years elapsed

so LTV reconciles to balance and valuation by construction, and the movement
between the two periods is the movement the economics imply — never a stored
delta. Everything is seeded, so two runs produce byte-identical files.

Deliberate exceptions
---------------------
Three, in the current period only, each a different failure mode caught at a
different gate — see ``SEEDED_EXCEPTIONS``. The prior period is clean, so
period-on-period movement is not muddied by data-quality noise.

Usage::

    python synthetic_demo/build_multibook_input.py
    python synthetic_demo/build_multibook_input.py --check   # fail if stale
"""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import random
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
OUTPUT_DIR = _HERE.parent / "input" / "multibook"

#: Fixed seed. Two runs of the same code produce byte-identical files.
RANDOM_SEED = 20260630

#: Annual house-price drift applied to valuations, monthly-compounded.
HPI_ANNUAL = 0.024


# --------------------------------------------------------------------------- #
# Books and periods — mirroring demo_platform.config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Book:
    source_portfolio_id: str
    source_portfolio_type: str      # direct | acquired  (origination provenance)
    label: str
    short_label: str
    #: warehoused (held, destined for a future deal) | sold (derecognised).
    #: Distinct from the provenance type: a securitisation the sponsor
    #: originated and then sold is `direct` provenance and `sold` status, and
    #: both are true at once.
    balance_sheet_status: str
    status_detail: str
    loans: int
    channel: str
    acquisition_date: Optional[str] = None
    seller_name: Optional[str] = None
    #: New loans written into the current period (origination books only).
    new_business: int = 0
    #: Loans redeeming during the current period.
    redemptions: int = 0


BOOKS: Tuple[Book, ...] = (
    Book(
        source_portfolio_id="alp_origination",
        source_portfolio_type="direct",
        label="ALP Origination Book",
        short_label="Origination",
        balance_sheet_status="warehoused",
        status_detail="Warehoused, destined for SPV2",
        loans=44,
        channel="Direct",
        new_business=3,
    ),
    Book(
        source_portfolio_id="alp_acquired",
        source_portfolio_type="acquired",
        label="ALP Acquired Back Book",
        short_label="Acquired",
        balance_sheet_status="warehoused",
        status_detail="Warehoused, destined for SPV2",
        loans=38,
        channel="Broker",
        acquisition_date="2024-09-30",
        seller_name="Cranmere Life Assurance (synthetic)",
        redemptions=1,
    ),
    Book(
        source_portfolio_id="spv1_sponsored",
        source_portfolio_type="direct",
        label="SPV1 Sponsored Securitisation",
        short_label="SPV1",
        balance_sheet_status="sold",
        status_detail="Servicing, risk retention and investor reporting retained",
        loans=34,
        channel="IFA",
    ),
)

#: The warehoused platform — what sits on the sponsor's balance sheet.
PLATFORM_BOOK_IDS = ("alp_origination", "alp_acquired")
#: Everything the sponsor still reports on, including the deal it sold.
SPONSOR_BOOK_IDS = tuple(b.source_portfolio_id for b in BOOKS)


@dataclass(frozen=True)
class Period:
    reporting_date: date
    role: str        # prior | current
    label: str


PERIODS: Tuple[Period, ...] = (
    Period(date(2026, 5, 31), "prior", "May 2026"),
    Period(date(2026, 6, 30), "current", "June 2026"),
)
CURRENT = PERIODS[-1]
PRIOR = PERIODS[0]


# --------------------------------------------------------------------------- #
# Source vocabulary — reused from the existing single-book extract so Gate 1's
# alias resolution and Gate 2's enum mapping behave identically.
# --------------------------------------------------------------------------- #
REGIONS: Tuple[Tuple[str, str], ...] = (
    ("B1 1AA", "West Midlands"),
    ("BS1 5AH", "South West"),
    ("CB2 1TN", "East of England"),
    ("CF10 1EP", "Wales"),
    ("EH1 1YZ", "Scotland"),
    ("LS1 4AP", "Yorkshire and The Humber"),
    ("M1 1AE", "North West"),
    ("NG1 5FS", "East Midlands"),
    ("RG1 2DP", "South East"),
    ("SW1A 1AA", "London"),
)

PROPERTY_TYPES = ("Detached House", "Semi Detached", "Bungalow", "Flat / Apartment")
TENURES = ("Freehold", "Freehold", "Freehold", "LEASEHOLD")
PURPOSES = ("Home improvements", "Debt Consolidation", "Equity release",
            "Purchase Main Residence")
#: Replaces the previous 'manual' placeholder, which failed the canonical
#: enumeration on 100% of rows and read as a broken feed rather than a finding.
#:
#: These are the ESMA codes that the canonical enumeration
#: (config/system/enum_synonyms.yaml) and the Annex 2 delivery rule for RREL13
#: BOTH already accept — PNNR pensioner, EMUK employed, SFEM self-employed.
#: Deliberately chosen from that intersection so the demonstration needs no
#: change to config/regime/, which is under a change freeze.
EMPLOYMENT_STATUSES = ("PNNR", "PNNR", "SFEM", "EMUK")
COLLATERAL_TYPE = "residential property"

SOURCE_HEADERS: Tuple[str, ...] = (
    "Underlying Exposure Identifier", "Loan Identifier", "Unique Loan Identifier",
    "Original Underlying Exposure Identifier", "Original Obligor Identifier",
    "New Underlying Exposure Identifier", "New Obligor Identifier",
    "Origination Dt", "Maturity DATE", "Original Principal Balance",
    "Current Principal Balance GBP", "Current Interest Rate",
    "Original Loan To Value", "Current Loan To Value", "Current Valuation Amount",
    "Current Valuation Date", "Current Valuation Method", "Post Code", "Region ",
    "Property Type ", "tenure", "Borrower Age (Youngest)", "Account Status",
    "Redemption Date", "Data Cut-Off Date", "Interest Rate Type", "Occupancy Type",
    "Amortisation Type", "Origination Channel", "Purpose", "Total Credit Limit",
    "Number of Days in Arrears", "Exposure Currency Denomination",
    "Pool Addition Date", "Date of Repurchase", "Purchase Price", "Arrears Balance",
    "Default Amount", "Default Date", "Allocated Losses", "Cumulative Recoveries",
    "Litigation", "Deposit Amount", "Collateral Currency", "Collateral Type",
    "Original Collateral Identifier", "New Collateral Identifier",
    "Scheduled Principal Payment Frequency", "Scheduled Interest Payment Frequency",
    "Payment Due", "Current Interest Rate Index", "Current Interest Rate Index Tenor",
    "Current Interest Rate Margin", "Interest Rate Reset Interval",
    "Interest Rate Cap", "Interest Rate Floor", "Prepayment Fee",
    "Principal Grace Period End Date", "Credit Impaired Obligor", "Resident",
    "Geographic Region Obligor", "Employment Status",
)


# --------------------------------------------------------------------------- #
# Deliberate data-quality exceptions
# --------------------------------------------------------------------------- #
#: Three exceptions, current period only. Each is a different failure mode, and
#: each is caught at a different gate — which is the point: a single seeded
#: failure would show that validation runs, not that it reasons.
#:
#:   valuation_method-> Gate 3 ENUM_INVALID *and* Gate 4b RREC14 enum error.
#:                      A seller's own valuation taxonomy the canonical model
#:                      does not carry. Both gates agree: blocks the submission.
#:   valuation_date  -> Gate 4b RREC15 mandatory-missing ONLY. Gate 3 validates
#:                      the values that are present, so an absent mandatory
#:                      Annex value is invisible to it. This is the case that
#:                      proves the reconciliation has to run in both directions.
#:   maturity_date   -> Gate 3 business rule DAT002. An internal contradiction
#:                      no field-level check would find, and not an Annex
#:                      enumeration issue at all.
SEEDED_EXCEPTIONS: Dict[str, Dict[str, Any]] = {
    "current_valuation_method": {
        "book": "alp_acquired",
        "loan_offsets": (4, 11, 19),
        "value": "AVM (desktop)",
        "demonstrates": (
            "A value outside the enumerated list. The acquired seller's tape "
            "records three valuations by its own desktop method; neither the "
            "canonical enumeration nor ESMA Annex 2 RREC14 carries that code, "
            "so the submission would state a valuation method that does not "
            "exist in the regime."
        ),
    },
    "current_valuation_date": {
        "book": "spv1_sponsored",
        "loan_offsets": (7, 22),
        "value": "",
        "demonstrates": (
            "A missing mandatory field. Two revaluation dates did not survive "
            "the servicing migration. The canonical validator does not object — "
            "it checks the values that are there — but Annex 2 RREC15 is "
            "mandatory, so the submission is blocked at the regime layer only."
        ),
    },
    "maturity_date": {
        "book": "alp_acquired",
        "loan_offsets": (27,),
        "demonstrates": (
            "A date inconsistency. One re-keyed maturity date precedes its own "
            "origination date. No field-level check would find it; the business "
            "rule that compares the two does (DAT002)."
        ),
    },
}


# --------------------------------------------------------------------------- #
# Date helpers — month-end arithmetic only
# --------------------------------------------------------------------------- #
def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _months_between(start: date, end: date) -> int:
    return max(0, (end.year - start.year) * 12 + (end.month - start.month))


def _add_months(d: date, months: int) -> date:
    total = d.month - 1 + months
    year, month = d.year + total // 12, total % 12 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def _uk(d: Optional[date]) -> str:
    """dd/mm/yyyy — the source system's format, not the canonical one."""
    return d.strftime("%d/%m/%Y") if d else ""


# --------------------------------------------------------------------------- #
# The loan model
# --------------------------------------------------------------------------- #
@dataclass
class Loan:
    book: Book
    index: int
    origination: date
    maturity: date
    advance: float
    rate: float               # annual nominal, roll-up
    original_valuation: float
    age_at_completion: int
    postcode: str
    region: str
    property_type: str
    tenure: str
    purpose: str
    employment_status: str
    #: Month-end at which this loan redeems, if it does.
    redeems_on: Optional[date] = None
    #: Month-end from which this loan first appears (new business).
    first_reported: Optional[date] = None

    def balance(self, at: date) -> float:
        months = _months_between(self.origination, at)
        return round(self.advance * (1 + self.rate / 1200) ** months, 2)

    def valuation(self, at: date) -> float:
        months = _months_between(self.origination, at)
        return round(self.original_valuation * (1 + HPI_ANNUAL / 12) ** months, 2)

    def ltv(self, at: date) -> float:
        return round(self.balance(at) / self.valuation(at), 4)

    def age(self, at: date) -> int:
        return self.age_at_completion + (at.year - self.origination.year)

    def reported_at(self, at: date) -> bool:
        if self.first_reported and at < self.first_reported:
            return False
        if self.redeems_on and at > self.redeems_on:
            return False
        return True


def _build_loans(book: Book, rng: random.Random) -> List[Loan]:
    """Generate one book's loans once. Every period derives from these."""
    loans: List[Loan] = []
    total = book.loans + book.new_business

    for i in range(total):
        is_new = i >= book.loans
        # New business completes inside the current period; the rest is seasoned.
        if is_new:
            origination = _month_end(CURRENT.reporting_date.year,
                                     CURRENT.reporting_date.month)
        else:
            seasoning = rng.randint(4, 96)
            origination = _add_months(_month_end(CURRENT.reporting_date.year,
                                                 CURRENT.reporting_date.month),
                                      -seasoning)

        postcode, region = REGIONS[rng.randrange(len(REGIONS))]
        original_valuation = float(rng.randrange(185_000, 1_250_000, 5_000))
        initial_ltv = rng.uniform(0.21, 0.48)
        advance = round(original_valuation * initial_ltv, 2)
        rate = round(rng.uniform(5.85, 8.95), 3)
        age_at_completion = rng.randint(58, 84)
        # A lifetime mortgage runs to the later of redemption or a long stop.
        maturity = _add_months(origination, 12 * rng.randint(14, 26))

        loans.append(Loan(
            book=book,
            index=i,
            origination=origination,
            maturity=maturity,
            advance=advance,
            rate=rate,
            original_valuation=original_valuation,
            age_at_completion=age_at_completion,
            postcode=postcode,
            region=region,
            property_type=PROPERTY_TYPES[rng.randrange(len(PROPERTY_TYPES))],
            tenure=TENURES[rng.randrange(len(TENURES))],
            purpose=PURPOSES[rng.randrange(len(PURPOSES))],
            employment_status=EMPLOYMENT_STATUSES[rng.randrange(len(EMPLOYMENT_STATUSES))],
            first_reported=CURRENT.reporting_date if is_new else None,
        ))

    # Redemptions leave the book after the prior period, so the movement between
    # the two cuts carries a genuine exit rather than only accrual.
    for offset in range(book.redemptions):
        loans[13 + offset * 7].redeems_on = PRIOR.reporting_date

    return loans


# --------------------------------------------------------------------------- #
# Source-row rendering
# --------------------------------------------------------------------------- #
def _row(loan: Loan, period: Period, seq: int) -> Dict[str, str]:
    at = period.reporting_date
    book = loan.book
    ident = f"{book.short_label.upper()}-{seq:04d}"
    balance = loan.balance(at)
    valuation = loan.valuation(at)

    maturity = loan.maturity
    valuation_date = _month_end(at.year, at.month)
    valuation_method = "Full survey"

    # Deliberate exceptions, current period only.
    if period.role == "current":
        vm = SEEDED_EXCEPTIONS["current_valuation_method"]
        if book.source_portfolio_id == vm["book"] and loan.index in vm["loan_offsets"]:
            valuation_method = vm["value"]

        vd = SEEDED_EXCEPTIONS["current_valuation_date"]
        if book.source_portfolio_id == vd["book"] and loan.index in vd["loan_offsets"]:
            valuation_date = None

        md = SEEDED_EXCEPTIONS["maturity_date"]
        if book.source_portfolio_id == md["book"] and loan.index in md["loan_offsets"]:
            # Re-keyed the year: maturity now precedes origination.
            maturity = _add_months(loan.origination, -18)

    return {
        "Underlying Exposure Identifier": ident,
        "Loan Identifier": ident,
        "Unique Loan Identifier": f"SYN-{book.short_label.upper()}-{seq:04d}",
        "Original Underlying Exposure Identifier": ident,
        "Original Obligor Identifier": f"OBO-{book.short_label.upper()}-{seq:04d}",
        "New Underlying Exposure Identifier": f"{ident}-N",
        "New Obligor Identifier": f"OBL-{book.short_label.upper()}-{seq:04d}",
        "Origination Dt": _uk(loan.origination),
        "Maturity DATE": _uk(maturity),
        "Original Principal Balance": f"{loan.advance:,.2f}",
        "Current Principal Balance GBP": f" £{balance:.2f} ",
        "Current Interest Rate": f"{loan.rate}",
        "Original Loan To Value": f"{round(loan.advance / loan.original_valuation, 4)}",
        "Current Loan To Value": f"{loan.ltv(at)}",
        "Current Valuation Amount": f"{valuation:,.0f}",
        "Current Valuation Date": _uk(valuation_date),
        "Current Valuation Method": valuation_method,
        "Post Code": loan.postcode,
        "Region ": loan.region,
        "Property Type ": loan.property_type,
        "tenure": loan.tenure,
        "Borrower Age (Youngest)": str(loan.age(at)),
        "Account Status": "Active",
        "Redemption Date": "",
        "Data Cut-Off Date": _uk(at),
        "Interest Rate Type": "Fixed",
        "Occupancy Type": "Owner Occupied",
        "Amortisation Type": "Interest roll-up",
        "Origination Channel": book.channel,
        "Purpose": loan.purpose,
        "Total Credit Limit": f"{round(loan.advance * 1.05, 2):,.2f}",
        "Number of Days in Arrears": "0",
        "Exposure Currency Denomination": "GBP",
        "Pool Addition Date": _uk(book.acquisition_date and
                                  date.fromisoformat(book.acquisition_date)
                                  or loan.origination),
        "Date of Repurchase": "",
        "Purchase Price": f"{loan.original_valuation:,.0f}",
        "Arrears Balance": "0",
        "Default Amount": "0",
        "Default Date": "",
        "Allocated Losses": "0",
        "Cumulative Recoveries": "0",
        "Litigation": "N",
        "Deposit Amount": "0",
        "Collateral Currency": "GBP",
        "Collateral Type": COLLATERAL_TYPE,
        "Original Collateral Identifier": f"{ident}-COLL-O",
        "New Collateral Identifier": f"{ident}-COLL-N",
        "Scheduled Principal Payment Frequency": "Quarterly",
        "Scheduled Interest Payment Frequency": "Monthly",
        "Payment Due": "0",
        "Current Interest Rate Index": "SONIA",
        "Current Interest Rate Index Tenor": "1M",
        "Current Interest Rate Margin": "0",
        "Interest Rate Reset Interval": "12",
        "Interest Rate Cap": "15",
        "Interest Rate Floor": "0",
        "Prepayment Fee": "0",
        "Principal Grace Period End Date": _uk(_add_months(at, 6)),
        "Credit Impaired Obligor": "N",
        "Resident": "Y",
        "Geographic Region Obligor": loan.region,
        "Employment Status": loan.employment_status,
    }


def build() -> Dict[str, Any]:
    """Write every book/period extract. Returns a manifest of what was written."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"books": [], "periods": [], "files": []}

    for period in PERIODS:
        manifest["periods"].append({
            "role": period.role,
            "reportingDate": period.reporting_date.isoformat(),
            "label": period.label,
        })

    for book in BOOKS:
        rng = random.Random(f"{RANDOM_SEED}:{book.source_portfolio_id}")
        loans = _build_loans(book, rng)
        manifest["books"].append({
            "sourcePortfolioId": book.source_portfolio_id,
            "sourcePortfolioType": book.source_portfolio_type,
            "label": book.label,
            "shortLabel": book.short_label,
            "balanceSheetStatus": book.balance_sheet_status,
            "statusDetail": book.status_detail,
        })

        for period in PERIODS:
            rows = []
            seq = 0
            for loan in loans:
                if not loan.reported_at(period.reporting_date):
                    continue
                seq += 1
                rows.append(_row(loan, period, seq))

            name = f"{book.source_portfolio_id}_{period.reporting_date.isoformat()}.csv"
            path = OUTPUT_DIR / name
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(SOURCE_HEADERS))
                writer.writeheader()
                writer.writerows(rows)

            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest["files"].append({
                "file": name,
                "book": book.source_portfolio_id,
                "period": period.reporting_date.isoformat(),
                "rows": len(rows),
                "sha256": digest,
            })
            print(f"  {name:44} {len(rows):4d} rows  {digest[:12]}")

    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="Fail if regenerating would change any committed file.")
    args = ap.parse_args()

    if args.check:
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(OUTPUT_DIR.glob("*.csv"))} if OUTPUT_DIR.exists() else {}

    print(f"Building multi-book synthetic source extracts into {OUTPUT_DIR}")
    build()

    if args.check:
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(OUTPUT_DIR.glob("*.csv"))}
        if before != after:
            print("STALE: committed extracts differ from a fresh build.", file=sys.stderr)
            sys.exit(1)
        print("OK: committed extracts are reproducible.")


if __name__ == "__main__":
    main()
