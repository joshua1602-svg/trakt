"""Fixtures for the borrowing-base suite.

THE RULE THIS SUITE HOLDS TO: no expected number is produced by calling the
code under test. Every expectation below is either a literal worked out by hand
from the supplied Schedule 8 and the operator's facility terms, or arithmetic
written out in the test itself.

The synthetic portfolio is deliberately tiny and deliberately round. A book of
two loans totalling exactly £100,000,000 makes every concentration share
readable as pounds-per-million, so an assertion of "15.0%" is checkable by
eye against the frame that produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from mi_agent.borrowing_base.models import (
    EligibilityRule,
    FacilityConfiguration,
)

REPO = Path(__file__).resolve().parents[2]
SCHEDULE_8 = (REPO / "tests" / "concentration_tests" / "fixtures"
              / "warehouse_facility_schedule_8.txt")

#: The operator's supplied prototype terms. Repeated here as literals rather
#: than read from configuration: a test that loads the same file the code loads
#: cannot notice the file changing.
COMMITMENT = 250_000_000.0
ADVANCE_RATE = 1.03
DENOMINATOR_FLOOR = 33_000_000.0

#: A round Financing Portfolio. £100m over two loans, so a slice of £15m is
#: 15.00% of the denominator with nothing to work out.
PORTFOLIO_TOTAL = 100_000_000.0

#: A region with no Schedule 8 limit, used to hold the balance that is NOT the
#: subject of a boundary test.
UNLIMITED_REGION = "Northern Ireland"


def facility(**over: Any) -> FacilityConfiguration:
    """The prototype ERE warehouse facility, constructed explicitly.

    Built from literals rather than loaded, so a test states the terms it
    depends on. ``config.py`` is tested separately, against the real file.
    """
    spec: Dict[str, Any] = dict(
        client_id="test_client",
        facility_id="TEST_WAREHOUSE_01",
        facility_type="warehouse",
        currency="GBP",
        commitment=COMMITMENT,
        advance_rate=ADVANCE_RATE,
        concentration_denominator_floor=DENOMINATOR_FLOOR,
        current_drawn_amount=None,
        environment="prototype",
        eligibility_rule_version="test-0",
        prototype_assume_financing_portfolio_eligible=True,
    )
    spec.update(over)
    return FacilityConfiguration(**spec)


def production_facility(rules: Optional[List[EligibilityRule]] = None,
                        **over: Any) -> FacilityConfiguration:
    """A PRODUCTION facility. The prototype assumption is never honoured here."""
    spec: Dict[str, Any] = dict(
        client_id="test_client",
        facility_id="PROD_WAREHOUSE_01",
        commitment=COMMITMENT,
        advance_rate=ADVANCE_RATE,
        concentration_denominator_floor=DENOMINATOR_FLOOR,
        environment="production",
        eligibility_rules=list(rules or []),
        eligibility_rule_version="prod-1" if rules else "",
    )
    spec.update(over)
    return FacilityConfiguration(**spec)


def loan(**over: Any) -> Dict[str, Any]:
    """One synthetic loan with every field the Schedule 8 metrics read.

    The defaults are chosen so a loan is uninteresting to every test except the
    one that overrides a field: an unlimited region, a mid-range valuation, one
    borrower aged 70, and a £250k initial balance.
    """
    row: Dict[str, Any] = {
        "loan_id": over.pop("loan_id", "L0001"),
        "collateral_geography": UNLIMITED_REGION,
        "current_outstanding_balance": 1_000_000.0,
        "original_principal_balance": 250_000.0,
        "original_valuation_amount": 500_000.0,
        "borrower_identifier": "B0001",
        "youngest_borrower_age": 70.0,
        "number_of_borrowers": 1,
        "current_interest_rate": 6.5,
    }
    row.update(over)
    return row


def frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([loan(**r) for r in rows])


def two_loan_book(*, subject_balance: float, **subject: Any) -> pd.DataFrame:
    """A £100,000,000 book: one SUBJECT loan, and the remainder parked in a
    region no Schedule 8 limit covers.

    The remainder is what makes the denominator exactly ``PORTFOLIO_TOTAL``, so
    the subject's share is ``subject_balance / 1,000,000`` percent — readable
    without a calculator.
    """
    rest = PORTFOLIO_TOTAL - subject_balance
    rows = [{"loan_id": "SUBJECT",
             "current_outstanding_balance": subject_balance, **subject}]
    if rest:
        rows.append({"loan_id": "REST", "current_outstanding_balance": rest})
    return frame(rows)


@pytest.fixture(scope="session")
def schedule_8_text() -> str:
    return SCHEDULE_8.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def lib():
    from mi_agent.concentration_tests.library import load_library
    return load_library()
