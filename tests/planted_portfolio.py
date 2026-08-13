#!/usr/bin/env python3
"""A small, production-shaped portfolio with deliberately planted cases.

Twelve loans, chosen so that every branch of the credit path has a case that
exercises it and a **known truth stated here as a literal**. The truth table
below is arithmetic done by hand and written down; nothing in this module calls
``select_valuation``, ``assemble_loan`` or ``explain_values`` to work out what
the answer should be. A fixture that derives its expectations from the code
under test proves only that the code is self-consistent.

The planted cases, and what each is for:

    LN-N-001   normal                    a recent full valuation, everything agrees
    LN-H-002   high LTV                  95%, the figure a buyer screens on
    LN-S-003   stale valuation           the full one is 90 months old and is
                                         REJECTED; the policy falls back
    LN-M-004   multiple valuations       full / indexed / purchase price on one
                                         collateral, and the INDEXED one is the
                                         most recent — preference must beat recency
    LN-I-005   indexed selected          the full one has aged out, so the indexed
                                         observation legitimately backs the LTV
    LN-V-006   missing valuation         no observations at all — refuse, do not
                                         invent a denominator
    LN-F-007   missing canonical field   current_loan_to_value is absent from the
                                         tape although the inputs are present
    LN-X-009   unsupportable LTV         the tape carries a number but EVERY
                                         observation has aged out. The tape's
                                         figure stands and the explanation says
                                         it is unsupported — that is a finding
    LN-C-008A-D  concentration breach    one borrower, one region, 43.75% / 55.625%

Plus a field-level validation exception, which is where a validation failure
actually lives: mapping, transformation and validation are properties of a FIELD
within a snapshot, so ``current_loan_to_value`` carries ``validation_status:
failed`` in the lineage index and every loan's explanation of that field inherits
it. There is no per-cell validation state to plant.

The reporting date is 2026-07-31 and every age in the truth table is measured
against it, so the expectations do not drift as the wall clock moves.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core.lineage import (
    ORIGIN_DERIVED,
    ORIGIN_MAPPED,
    ORIGIN_RUN_METADATA,
    LineageEntry,
    LineageIndex,
)

REPORTING_DATE = "2026-07-31"
BOOK = "direct_001"
SNAPSHOT_ID = "snap_planted_0001"
CONTENT_HASH = "sha256:planted00000000000000000000000000000000000000000000000000000000"

#: Hand arithmetic, written down. 3.2m of principal across twelve loans.
TOTAL_PRINCIPAL = 3_200_000.0

#: region -> (balance, share of TOTAL_PRINCIPAL as a percentage)
REGION_TRUTH: Dict[str, Any] = {
    "London":     (1_780_000.0, 55.625),
    "South East": (  720_000.0, 22.500),
    "North West": (  430_000.0, 13.4375),
    "Wales":      (  270_000.0,  8.4375),
}

#: The one borrower holding more than one loan.
CONCENTRATED_BORROWER = "BR-CONC-01"
CONCENTRATED_BORROWER_BALANCE = 1_400_000.0
CONCENTRATED_BORROWER_SHARE = 43.75


# --------------------------------------------------------------------------- #
# The rows
# --------------------------------------------------------------------------- #
def _loan(loan_id: str, borrower: str, region: str, balance: float, *,
          ltv: Optional[float],
          current_amount: Optional[float] = None,
          current_date: Optional[str] = None,
          current_method: Optional[str] = None,
          original_amount: Optional[float] = None,
          original_date: Optional[str] = None,
          indexed_amount: Optional[float] = None,
          indexed_date: Optional[str] = None,
          indexed_ltv: Optional[float] = None,
          collateral_id: Optional[str] = None,
          status: str = "performing",
          arrears: float = 0.0,
          days_in_arrears: int = 0,
          stage: str = "1") -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "loan_identifier": loan_id,
        "borrower_identifier": borrower,
        "source_portfolio_id": BOOK,
        "source_portfolio_type": "direct",
        "portfolio_cohort": "2026H1",
        "account_status": status,
        "current_principal_balance": balance,
        "current_outstanding_balance": balance,
        "original_principal_balance": balance * 1.1,
        "current_loan_to_value": ltv,
        "indexed_loan_to_value": indexed_ltv,
        "current_interest_rate": 5.25,
        "arrears_balance": arrears,
        "number_of_days_in_arrears": days_in_arrears,
        "ifrs9_stage": stage,
        "origination_date": "2021-03-01",
        "maturity_date": "2046-03-01",
        "current_valuation_amount": current_amount,
        "current_valuation_date": current_date,
        "current_valuation_method": current_method,
        "current_valuation_basis": "market_value" if current_amount else None,
        "original_valuation_amount": original_amount,
        "date_of_valuation_at_securitisation": original_date,
        "indexed_value": indexed_amount,
        "index_determination_date": indexed_date,
        "collateral_type": "residential_property",
        "property_type": "detached_house",
        "occupancy_type": "owner_occupied",
        "geographic_region_obligor": region,
        "geographic_region_collateral": region,
        "borrower_type": "individual",
        "employment_status": "employed",
        "exposure_currency_denomination": "GBP",
        "collateral_currency": "GBP",
        "reporting_date": REPORTING_DATE,
    }
    if collateral_id is not None:
        # Deliberately the servicer-side spelling, with NO collateral_identifier.
        # A tape shaped this way is what proves the collateral id is resolved the
        # same way by the loan object and by the explanation.
        row["collateral_id"] = collateral_id
    return row


ROWS: List[Dict[str, Any]] = [
    # --- normal: recent full valuation, indexed and purchase price also present
    _loan("LN-N-001", "BR-000001", "South East", 200_000.0, ltv=50.00,
          current_amount=400_000.0, current_date="2026-05-31",
          current_method="full valuation",
          original_amount=350_000.0, original_date="2021-03-01",
          indexed_amount=410_000.0, indexed_date="2026-06-30",
          indexed_ltv=48.78),

    # --- high LTV
    _loan("LN-H-002", "BR-000002", "London", 380_000.0, ltv=95.00,
          current_amount=400_000.0, current_date="2026-04-30",
          current_method="full valuation"),

    # --- stale full valuation; falls back to the purchase price
    _loan("LN-S-003", "BR-000003", "Wales", 150_000.0, ltv=50.00,
          current_amount=320_000.0, current_date="2019-01-04",
          current_method="full valuation",
          original_amount=300_000.0, original_date="2018-11-15"),

    # --- multiple observations, the INDEXED one most recent
    _loan("LN-M-004", "BR-000004", "South East", 300_000.0, ltv=60.00,
          current_amount=500_000.0, current_date="2026-06-30",
          current_method="full valuation",
          original_amount=430_000.0, original_date="2020-02-14",
          indexed_amount=520_000.0, indexed_date="2026-07-15",
          indexed_ltv=57.69,
          collateral_id="COL-LN-M-004"),

    # --- the full valuation has aged out; the indexed one legitimately wins
    _loan("LN-I-005", "BR-000005", "North West", 180_000.0, ltv=60.00,
          current_amount=250_000.0, current_date="2021-01-31",
          current_method="full valuation",
          original_amount=240_000.0, original_date="2018-06-01",
          indexed_amount=300_000.0, indexed_date="2026-06-30",
          indexed_ltv=60.00),

    # --- no valuation at all
    _loan("LN-V-006", "BR-000006", "Wales", 120_000.0, ltv=None),

    # --- the canonical field itself is missing, though its inputs are present
    _loan("LN-F-007", "BR-000007", "South East", 220_000.0, ltv=None,
          current_amount=400_000.0, current_date="2026-03-31",
          current_method="full valuation"),

    # --- a number on the tape that NO observation can support
    _loan("LN-X-009", "BR-000009", "North West", 250_000.0, ltv=62.50,
          current_amount=400_000.0, current_date="2022-01-31",
          current_method="full valuation",
          indexed_amount=420_000.0, indexed_date="2024-12-31",
          indexed_ltv=59.52,
          status="arrears", arrears=4_200.0, days_in_arrears=64, stage="2"),

    # --- one borrower, one region: the concentration breach
    _loan("LN-C-008A", CONCENTRATED_BORROWER, "London", 400_000.0, ltv=80.00,
          current_amount=500_000.0, current_date="2026-06-30",
          current_method="full valuation"),
    _loan("LN-C-008B", CONCENTRATED_BORROWER, "London", 350_000.0, ltv=70.00,
          current_amount=500_000.0, current_date="2026-06-30",
          current_method="full valuation"),
    _loan("LN-C-008C", CONCENTRATED_BORROWER, "London", 300_000.0, ltv=60.00,
          current_amount=500_000.0, current_date="2026-06-30",
          current_method="full valuation"),
    _loan("LN-C-008D", CONCENTRATED_BORROWER, "London", 350_000.0, ltv=70.00,
          current_amount=500_000.0, current_date="2026-06-30",
          current_method="full valuation"),
]


def planted_frame() -> pd.DataFrame:
    """The planted portfolio as a canonical-shaped frame."""
    return pd.DataFrame(ROWS)


# --------------------------------------------------------------------------- #
# The truth table — stated, not computed
# --------------------------------------------------------------------------- #
# Ages are months from the observation date to 2026-07-31, by hand:
#   2026-05-31 ->  2      2026-06-30 ->  1      2026-04-30 ->  3
#   2019-01-04 -> 90      2021-01-31 -> 66      2022-01-31 -> 54
#   2024-12-31 ->  7 ... measured on (year, month) only: (2026-2024)*12 + (7-12) = 19
#   2026-07-15 ->  0      2026-03-31 ->  4
TRUTH: Dict[str, Dict[str, Any]] = {

    "LN-N-001": {
        "case": "normal",
        "observations": 3,
        "selected_type": "full",
        "selected_amount": 400_000.0,
        "selected_date": "2026-05-31",
        "current_loan_to_value": 50.00,
        # Both lower-preference observations qualify on age and are rejected
        # only because a full valuation was available.
        "rejected_reasons_contain": ["a qualifying 'full' valuation was available"],
        "resolved": True,
    },

    "LN-H-002": {
        "case": "high LTV",
        "observations": 1,
        "selected_type": "full",
        "selected_amount": 400_000.0,
        "selected_date": "2026-04-30",
        "current_loan_to_value": 95.00,
        "resolved": True,
    },

    "LN-S-003": {
        "case": "stale valuation",
        "observations": 2,
        # The full one is 90 months old against a 24-month limit.
        "selected_type": "purchase_price",
        "selected_amount": 300_000.0,
        "selected_date": "2018-11-15",
        "current_loan_to_value": 50.00,
        "rejected_reasons_contain": ["90 months old at 2026-07-31",
                                     "the policy allows 24"],
        "resolved": True,
    },

    "LN-M-004": {
        "case": "multiple valuations",
        "observations": 3,
        # The indexed observation (2026-07-15) is MORE RECENT than the full one
        # (2026-06-30). Preference order must still win.
        "selected_type": "full",
        "selected_amount": 500_000.0,
        "selected_date": "2026-06-30",
        "current_loan_to_value": 60.00,
        "collateral_id": "COL-LN-M-004",
        "resolved": True,
    },

    "LN-I-005": {
        "case": "indexed valuation selected",
        "observations": 3,
        # full is 66 months old (limit 24) -> rejected; indexed is 1 month old.
        "selected_type": "indexed",
        "selected_amount": 300_000.0,
        "selected_date": "2026-06-30",
        "current_loan_to_value": 60.00,
        "rejected_reasons_contain": ["66 months old at 2026-07-31"],
        "resolved": True,
    },

    "LN-V-006": {
        "case": "missing valuation",
        "observations": 0,
        "selected_type": None,
        "current_loan_to_value": None,
        "resolved": False,
        "reason": "no valuation observations are available for this collateral",
    },

    "LN-F-007": {
        "case": "missing canonical field",
        "observations": 1,
        "selected_type": "full",
        "selected_amount": 400_000.0,
        "selected_date": "2026-03-31",
        # The inputs are present and a denominator IS selectable; the tape simply
        # does not carry the ratio. Trakt must not fill the gap by computing one.
        "current_loan_to_value": None,
        "resolved": True,
    },

    "LN-X-009": {
        "case": "unsupportable LTV",
        "observations": 2,
        # full 54 months (limit 24); indexed 19 months (limit 12). Nothing left.
        "selected_type": None,
        "current_loan_to_value": 62.50,
        "resolved": False,
        "reason": "no observation qualified under this policy",
        "rejected_reasons_contain": ["54 months old at 2026-07-31",
                                     "19 months old at 2026-07-31"],
    },

    "LN-C-008A": {"case": "concentration", "observations": 1,
                  "selected_type": "full", "selected_amount": 500_000.0,
                  "current_loan_to_value": 80.00, "resolved": True},
    "LN-C-008B": {"case": "concentration", "observations": 1,
                  "selected_type": "full", "selected_amount": 500_000.0,
                  "current_loan_to_value": 70.00, "resolved": True},
    "LN-C-008C": {"case": "concentration", "observations": 1,
                  "selected_type": "full", "selected_amount": 500_000.0,
                  "current_loan_to_value": 60.00, "resolved": True},
    "LN-C-008D": {"case": "concentration", "observations": 1,
                  "selected_type": "full", "selected_amount": 500_000.0,
                  "current_loan_to_value": 70.00, "resolved": True},
}

#: Which section of the assembled object each canonical field must land in.
#: Independently stated so a silent re-assignment in the entity model is caught.
SECTION_TRUTH: Dict[str, str] = {
    "loan_identifier": "loan",
    "source_portfolio_id": "loan",
    "portfolio_cohort": "loan",
    "reporting_date": "loan",
    "borrower_identifier": "borrower",
    "borrower_type": "borrower",
    "employment_status": "borrower",
    "geographic_region_obligor": "borrower",
    "collateral_type": "collateral",
    "property_type": "collateral",
    "geographic_region_collateral": "collateral",
    "current_principal_balance": "contract",
    "current_loan_to_value": "contract",
    "indexed_loan_to_value": "contract",
    "origination_date": "contract",
    "maturity_date": "contract",
    "account_status": "performance",
    "arrears_balance": "performance",
    "number_of_days_in_arrears": "performance",
    "ifrs9_stage": "performance",
}

#: Valuation columns must NOT appear as scalars anywhere — they are observations.
VALUATION_COLUMNS = (
    "current_valuation_amount", "current_valuation_date", "current_valuation_method",
    "current_valuation_basis", "original_valuation_amount",
    "date_of_valuation_at_securitisation", "indexed_value", "index_determination_date",
)


# --------------------------------------------------------------------------- #
# The lineage index for this snapshot — including the planted validation failure
# --------------------------------------------------------------------------- #
def planted_lineage_index(tenant_id: str, snapshot_id: str = SNAPSHOT_ID
                          ) -> LineageIndex:
    """One entry per canonical FIELD, which is where validation state lives."""
    return LineageIndex(
        tenant_id=tenant_id, snapshot_id=snapshot_id, content_hash=CONTENT_HASH,
        reporting_date=REPORTING_DATE,
        source_dataset="planted_202607_central_tape.csv",
        entries={
            "current_principal_balance": LineageEntry(
                canonical_field="current_principal_balance", origin=ORIGIN_MAPPED,
                source_field="OUT_PRIN", source_dataset="july_servicer_tape.csv",
                mapping_method="alias", mapping_confidence=1.0,
                mapping_version="aliases_llm_confirmed.yaml@2026-06-14",
                validation_status="passed",
                validation_rules=("BAL001", "BAL101"), validation_error_count=0,
                format="decimal"),
            # The planted VALIDATION EXCEPTION. Field-level, because that is the
            # grain at which validation is actually recorded.
            "current_loan_to_value": LineageEntry(
                canonical_field="current_loan_to_value", origin=ORIGIN_DERIVED,
                derivation_rule="RESOLVE_CURRENT_LOAN_TO_VALUE",
                calculation_method="balance_over_valuation",
                calculation_version="1.2.0",
                validation_status="failed", validation_rules=("LTV001", "LTV004"),
                validation_error_count=2, unit="percentage_points",
                format="decimal"),
            "current_valuation_amount": LineageEntry(
                canonical_field="current_valuation_amount", origin=ORIGIN_MAPPED,
                source_field="VAL_AMT", source_dataset="july_collateral_tape.csv",
                mapping_method="alias", mapping_confidence=1.0,
                validation_status="passed", validation_rules=("VAL001",),
                validation_error_count=0, format="decimal"),
            "geographic_region_obligor": LineageEntry(
                canonical_field="geographic_region_obligor", origin=ORIGIN_MAPPED,
                source_field="REGION", source_dataset="july_servicer_tape.csv",
                mapping_method="alias", mapping_confidence=0.92,
                validation_status="passed", validation_rules=("GEO001",),
                validation_error_count=0),
            "source_portfolio_id": LineageEntry(
                canonical_field="source_portfolio_id", origin=ORIGIN_RUN_METADATA,
                validation_status="passed", validation_rules=("PROV001",),
                validation_error_count=0),
        })


#: The planted field-level validation exception, stated independently.
VALIDATION_EXCEPTION = {
    "canonical_field": "current_loan_to_value",
    "status": "failed",
    "rules_applied": ["LTV001", "LTV004"],
    "error_count": 2,
}
