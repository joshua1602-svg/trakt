"""simulation.assets._common — roll-forward mechanics shared by every asset class.

Status transitions, exits, write-offs and revaluation behave the same way
whatever the collateral is; only the *rates* differ, and those come from
:class:`simulation.knobs.Knobs`. Keeping them here means a bridge loan and an
equipment contract cannot drift into two different definitions of "default".

Every function is pure with respect to the random generator: it takes the
generator explicitly and never reaches for module-level randomness.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..dates import add_month_ends, months_between, parse_iso
from ..models import (
    MOVE_RECOVERY,
    MOVE_REDEMPTION,
    MOVE_WRITE_OFF,
    STATUS_ARREARS,
    STATUS_DEFAULT,
    STATUS_ENFORCEMENT,
    STATUS_PERFORMING,
    STATUS_REDEEMED,
    STATUS_WRITTEN_OFF,
    LoanState,
)

#: How a loan state is REPORTED on a lender tape. The production business rules
#: (BAL102/BAL103, ARR001, DEF001) are written against these upper-case tokens,
#: so the simulator writes what the platform actually validates rather than its
#: own internal vocabulary.
REPORTED_STATUS = {
    # "Active" rather than "PERFORMING": it is the token the governed enum
    # configuration already maps to the ESMA PERF code, so a performing account
    # reaches delivery without a bespoke mapping.
    "performing": "Active",
    # ARR001 accepts ARRE / DEFAULT / RESTRUCTURED for an account carrying
    # arrears, and DEF001 requires DEFAULT wherever a default date exists.
    "arrears": "ARRE",
    "default": "DEFAULT",
    # An account in enforcement IS in default; enforcement is disclosed through
    # the litigation flag, which is how the canonical model expresses it. Adding
    # a fourth status token here would break DEF001 for no analytical gain.
    "enforcement": "DEFAULT",
    "redeemed": "REDEEMED",
    "repossessed": "CLOSED",
    "written_off": "CLOSED",
}

#: Canonical fields every asset class emits. Each is `portfolio_type: common`
#: (or core) in ``config/system/fields_registry.yaml``, so they are selected for
#: every ``--portfolio-type`` the framework uses.
COMMON_CANONICAL_FIELDS = (
    "loan_identifier",
    "unique_identifier",
    "original_underlying_exposure_identifier",
    "new_underlying_exposure_identifier",
    "underlying_exposure_identifier",
    "original_obligor_identifier",
    "new_obligor_identifier",
    "data_cut_off_date",
    "origination_date",
    "pool_addition_date",
    "original_principal_balance",
    "current_principal_balance",
    "current_outstanding_balance",
    "original_valuation_amount",
    "current_valuation_amount",
    "current_valuation_method",
    "current_interest_rate",
    "interest_rate_type",
    "amortisation_type",
    "account_status",
    "arrears_balance",
    "number_of_days_in_arrears",
    "number_of_days_in_principal_arrears",
    "default_amount",
    "default_date",
    "allocated_losses",
    "cumulative_recoveries",
    "redemption_date",
    "maturity_date",
    "original_term",
    "total_credit_limit",
    "payment_due",
    "exposure_currency_denomination",
    "collateral_geography",
    "postcode",
    "purpose",
    "originator_name",
    "seller_name",
    "servicer_name",
    "origination_channel",
    "resident",
    "credit_impaired_obligor",
    "litigation",
    "new_collateral_identifier",
    "original_collateral_identifier",
    "collateral_unique_identifier",
    "collateral_currency",
    "collateral_type",
    "prepayment_fee",
)


def borrower_id(rng, loan_index: int, borrowers_per_hundred: int,
                prefix: str) -> str:
    """Deterministic obligor id honouring the configured borrower density.

    ``borrowers_per_hundred < 100`` deliberately makes several facilities share
    one obligor, which is what a single-borrower concentration limit measures.
    """
    density = max(1, min(100, int(borrowers_per_hundred)))
    bucket = (loan_index * density) // 100
    return f"{prefix}-OBL-{bucket:06d}"


def annual_to_monthly(rate_points: float) -> float:
    """Annual percentage points → a monthly compounding factor increment."""
    return float(rate_points) / 100.0 / 12.0


def revalue(loan: LoanState, monthly_growth: float) -> None:
    """Move the collateral / asset value one month, never below zero."""
    loan.current_value = max(0.0, round(loan.current_value * (1.0 + monthly_growth), 2))


def days_in_arrears(loan: LoanState) -> int:
    return int(loan.attributes.get("arrears_months", 0)) * 30


def advance_status(rng, loan: LoanState, knobs) -> None:
    """One month of performance transitions for an OPEN loan.

    The ladder is deliberately monotonic and explicit:
    performing → arrears → default → enforcement → written off, with a cure
    path back to performing from arrears only. Nothing skips a rung, so the
    migration matrix the production risk monitor computes has a knowable shape.
    """
    if loan.status == STATUS_PERFORMING:
        if float(rng.random()) < knobs.arrears_entry_rate:
            loan.status = STATUS_ARREARS
            loan.attributes["arrears_months"] = 1
        return

    if loan.status == STATUS_ARREARS:
        if float(rng.random()) < knobs.arrears_cure_rate:
            loan.status = STATUS_PERFORMING
            loan.attributes["arrears_months"] = 0
            loan.attributes["arrears_balance"] = 0.0
            return
        loan.attributes["arrears_months"] = int(
            loan.attributes.get("arrears_months", 0)) + 1
        if float(rng.random()) < knobs.default_rate:
            loan.status = STATUS_DEFAULT
            loan.attributes.setdefault("default_date", loan.attributes.get(
                "pending_reporting_date"))
        return

    if loan.status == STATUS_DEFAULT:
        loan.attributes["arrears_months"] = int(
            loan.attributes.get("arrears_months", 0)) + 1
        if float(rng.random()) < knobs.enforcement_rate:
            loan.status = STATUS_ENFORCEMENT
        return

    if loan.status == STATUS_ENFORCEMENT:
        loan.attributes["arrears_months"] = int(
            loan.attributes.get("arrears_months", 0)) + 1


def accrue_arrears(loan: LoanState) -> None:
    """Arrears balance grows by one month's contractual amount while impaired."""
    if loan.status in (STATUS_ARREARS, STATUS_DEFAULT, STATUS_ENFORCEMENT):
        monthly_due = float(loan.attributes.get("monthly_payment_due", 0.0))
        loan.attributes["arrears_balance"] = round(
            float(loan.attributes.get("arrears_balance", 0.0)) + monthly_due, 2)


def redeem(loan: LoanState, reporting_date: str, *, reason: str = "redemption"
           ) -> None:
    """Close a loan at par: the whole closing balance leaves as a redemption."""
    loan.record(MOVE_REDEMPTION, loan.closing_balance)
    loan.attributes["redemption_date"] = reporting_date
    loan.closing_balance = 0.0
    loan.status = STATUS_REDEEMED
    loan.exit_date = reporting_date
    loan.exit_reason = reason


def write_off(loan: LoanState, reporting_date: str, knobs, *,
              proceeds: Optional[float] = None, reason: str = "write_off",
              status: str = STATUS_WRITTEN_OFF,
              proceeds_kind: str = MOVE_RECOVERY) -> None:
    """Close an impaired loan, splitting the balance into proceeds and loss.

    Both parts are recorded SEPARATELY — exactly as a lender's loss table would —
    and together they equal the whole closing balance, so the movement identity
    ``opening + additions - reductions == closing`` still holds exactly.
    ``proceeds_kind`` lets asset finance report the same cash as an *asset
    disposal* rather than a *recovery*, which is what its loss table calls it.
    """
    balance = loan.closing_balance
    recovered = round(min(balance, proceeds if proceeds is not None
                          else balance * knobs.recovery_fraction), 2)
    loss = round(balance - recovered, 2)
    if recovered:
        loan.record(proceeds_kind, recovered)
    loan.record(MOVE_WRITE_OFF, loss)
    loan.attributes["cumulative_recoveries"] = round(
        float(loan.attributes.get("cumulative_recoveries", 0.0)) + recovered, 2)
    loan.attributes["allocated_losses"] = round(
        float(loan.attributes.get("allocated_losses", 0.0)) + loss, 2)
    loan.closing_balance = 0.0
    loan.status = status
    loan.exit_date = reporting_date
    loan.exit_reason = reason


def common_canonical(loan: LoanState, reporting_date: str, case) -> Dict[str, Any]:
    """The canonical fields every asset class fills in the same way."""
    attrs = loan.attributes
    return {
        "loan_identifier": loan.loan_id,
        # Business rule ID001 requires a unique identifier on every row.
        "unique_identifier": loan.loan_id,
        "original_underlying_exposure_identifier": f"EXP-{loan.loan_id}",
        "new_underlying_exposure_identifier": f"EXP-{loan.loan_id}",
        "underlying_exposure_identifier": f"EXP-{loan.loan_id}",
        "original_obligor_identifier": loan.borrower_id,
        "new_obligor_identifier": loan.borrower_id,
        "data_cut_off_date": reporting_date,
        "origination_date": loan.origination_date,
        "pool_addition_date": attrs.get("pool_addition_date", loan.funding_date),
        "original_principal_balance": round(loan.original_principal, 2),
        "current_principal_balance": round(loan.closing_balance, 2),
        "current_outstanding_balance": round(loan.closing_balance, 2),
        "original_valuation_amount": round(loan.original_value, 2),
        "current_valuation_amount": round(loan.current_value, 2),
        "current_valuation_method": attrs.get("valuation_method", "Indexed"),
        "current_interest_rate": round(loan.interest_rate, 4),
        "interest_rate_type": attrs.get("interest_rate_type", "Fixed"),
        "amortisation_type": attrs.get("amortisation_type", "Other"),
        "account_status": REPORTED_STATUS.get(loan.status, loan.status.upper()),
        "arrears_balance": round(float(attrs.get("arrears_balance", 0.0)), 2),
        "number_of_days_in_arrears": days_in_arrears(loan),
        # ARR002 measures PRINCIPAL arrears, a distinct canonical field. A
        # roll-up product has no contractual principal due, so the two agree
        # only where a payment schedule exists.
        "number_of_days_in_principal_arrears": (
            days_in_arrears(loan)
            if float(attrs.get("monthly_payment_due", 0.0)) > 0 else 0),
        "default_amount": (round(loan.closing_balance, 2)
                           if loan.status in (STATUS_DEFAULT, STATUS_ENFORCEMENT)
                           else 0.0),
        "default_date": attrs.get("default_date", ""),
        "allocated_losses": round(float(attrs.get("allocated_losses", 0.0)), 2),
        "cumulative_recoveries": round(float(attrs.get("cumulative_recoveries", 0.0)), 2),
        "redemption_date": attrs.get("redemption_date", ""),
        "maturity_date": attrs.get("maturity_date", ""),
        "original_term": int(attrs.get("original_term_months", 0)),
        "total_credit_limit": round(float(attrs.get("credit_limit",
                                                    loan.original_principal)), 2),
        "payment_due": round(float(attrs.get("monthly_payment_due", 0.0)), 2),
        "exposure_currency_denomination": case.currency,
        "collateral_geography": loan.region,
        "postcode": loan.postcode,
        "purpose": attrs.get("purpose", "Other"),
        "originator_name": attrs.get("originator_name", ""),
        "seller_name": attrs.get("seller_name", ""),
        "servicer_name": attrs.get("servicer_name", ""),
        "origination_channel": attrs.get("origination_channel", "Direct"),
        "resident": "Y",
        "credit_impaired_obligor": "N",
        "litigation": "Y" if loan.status == STATUS_ENFORCEMENT else "N",
        "new_collateral_identifier": f"COL-{loan.loan_id}",
        "original_collateral_identifier": f"COL-{loan.loan_id}",
        "collateral_unique_identifier": f"COL-{loan.loan_id}",
        "collateral_currency": case.currency,
        "collateral_type": attrs.get("collateral_type", "Other"),
        # An early-repayment charge is only levied on a redemption, and the tape
        # carries the book at the cut-off, so it is nil for an open exposure.
        "prepayment_fee": 0.0,
    }


def maturity_from_term(origination_date: str, term_months: int) -> str:
    return add_month_ends(parse_iso(origination_date), int(term_months)).isoformat()


def months_to_maturity(reporting_date: str, maturity_date: str) -> int:
    if not maturity_date:
        return 0
    return months_between(parse_iso(reporting_date), parse_iso(maturity_date))


__all__ = [
    "COMMON_CANONICAL_FIELDS", "accrue_arrears", "advance_status",
    "annual_to_monthly", "borrower_id", "common_canonical", "days_in_arrears",
    "maturity_from_term", "months_to_maturity", "redeem", "revalue", "write_off",
]
