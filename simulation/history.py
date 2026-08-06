"""simulation.history — month-by-month funded-book roll-forward.

Produces ONE :class:`~simulation.models.EconomicHistory` per case: a coherent
sequence of monthly funded snapshots with stable loan identifiers, where every
period's opening balance plus its movements equals its closing balance under
transparent rules.

This is the framework's single economic truth. Everything downstream — the
source dialects, the expected-truth package and every assertion — is derived
from it, which is why the reconciliation below is enforced here rather than
being left as something to check later.

Funded books only. There is no pipeline record, application or origination
funnel anywhere in this module.
"""

from __future__ import annotations

import copy
from typing import Dict, List

from .assets import GenContext, get_profile
from .generator import build_opening_book, make_rng, originate, period_volume
from .knobs import resolve_knobs
from .models import (
    MOVE_RESTATEMENT,
    POSITIVE_MOVEMENTS,
    CaseManifest,
    EconomicHistory,
    LoanState,
    PeriodSnapshot,
)

#: Absolute tolerance, in currency units, for the movement reconciliation.
RECONCILIATION_TOLERANCE = 0.05


class ReconciliationError(AssertionError):
    """A generated period did not reconcile — a simulator defect, never a pass."""


def _totals(loans) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for loan in loans:
        for kind, amount in loan.movements.items():
            out[kind] = round(out.get(kind, 0.0) + amount, 2)
    return out


def _snapshot_balance(loans) -> float:
    return round(sum(l.closing_balance for l in loans), 2)


def generate_history(case: CaseManifest, *, loan_count: int | None = None
                     ) -> EconomicHistory:
    """Generate the complete funded history for *case*.

    ``loan_count`` overrides the manifest's ``opening_loan_count`` so one
    manifest can be run at small, standard and large scale without editing it.
    """
    knobs = resolve_knobs(case.asset_class, case.economic_intent)
    profile = get_profile(case.asset_class)
    rng = make_rng(case.seed)
    opening_count = int(loan_count or case.opening_loan_count)

    book: List[LoanState] = build_opening_book(rng, case, knobs, opening_count)
    next_index = opening_count
    periods: List[PeriodSnapshot] = []

    for period_index, reporting_date in enumerate(case.reporting_dates()):
        ctx = GenContext(case, period_index, reporting_date, knobs)

        # 1. Opening balance: the book as it stood at the end of last period.
        opening_balance = _snapshot_balance(book)
        for loan in book:
            loan.reset_period()

        # 2. Roll every loan on book one month.
        for loan in book:
            profile.roll(rng, loan, ctx)

        # 3. New business and acquisitions written during the period.
        additions: List[LoanState] = []
        funded_n = period_volume(rng, knobs.new_business_rate, len(book))
        if funded_n:
            additions += originate(rng, case, knobs, count=funded_n,
                                   period_index=period_index,
                                   reporting_date=reporting_date,
                                   next_index=next_index, cohort="direct")
            next_index += funded_n
        acquired_n = period_volume(rng, knobs.acquisition_rate, len(book))
        if acquired_n:
            additions += originate(rng, case, knobs, count=acquired_n,
                                   period_index=period_index,
                                   reporting_date=reporting_date,
                                   next_index=next_index, cohort="acquired")
            next_index += acquired_n

        # 4. A declared restatement corrects the PRIOR period's reported
        #    balance in this period. Signed and additive, so the identity holds.
        if (knobs.restatement_period == period_index
                and knobs.restatement_fraction and book):
            _apply_restatement(book, knobs.restatement_fraction)

        all_loans = book + additions
        closing_balance = _snapshot_balance(all_loans)
        movement_totals = _totals(all_loans)

        open_loans = [l for l in all_loans if l.is_open]
        exits = [l for l in all_loans if not l.is_open]

        snapshot = PeriodSnapshot(
            reporting_date=reporting_date, period_index=period_index,
            loans=tuple(copy.deepcopy(l) for l in open_loans),
            exits=tuple(copy.deepcopy(l) for l in exits),
            opening_balance=opening_balance, closing_balance=closing_balance,
            movement_totals=movement_totals)

        if not snapshot.reconciles(RECONCILIATION_TOLERANCE):
            delta = sum(v if k in POSITIVE_MOVEMENTS else -v
                        for k, v in movement_totals.items())
            raise ReconciliationError(
                f"{case.case_id} period {period_index} ({reporting_date}) does "
                f"not reconcile: opening {opening_balance:,.2f} + movements "
                f"{delta:,.2f} = {opening_balance + delta:,.2f}, but closing is "
                f"{closing_balance:,.2f}. Movements: {movement_totals}")
        periods.append(snapshot)

        # 5. Carry only the loans still on book into the next period.
        book = open_loans

    return EconomicHistory(case_id=case.case_id, seed=case.seed,
                           asset_class=case.asset_class, periods=tuple(periods))


def _apply_restatement(book: List[LoanState], fraction: float) -> None:
    """A controlled correction to previously reported balances.

    Applied to the FIRST loans in stable id order so it is reproducible, and
    recorded as a signed ``restatement`` movement so the period still reconciles
    and the correction is visible rather than absorbed.
    """
    affected = book[: max(1, len(book) // 20)]
    for loan in affected:
        adjustment = round(-abs(loan.closing_balance * fraction), 2)
        loan.closing_balance = round(loan.closing_balance + adjustment, 2)
        loan.record(MOVE_RESTATEMENT, adjustment)
        loan.attributes["restated"] = "Y"


def canonical_frame_rows(case: CaseManifest, history: EconomicHistory,
                         period_index: int) -> List[Dict]:
    """One canonical-field-space row per loan ON BOOK at a reporting date.

    A funded monthly tape reports the book as it stands at the cut-off. Loans
    that left during the period are NOT carried at a zero balance: doing so
    would make every exit breach the platform's own BAL103 ("a zero balance must
    carry a closed status") and LTV001 ("current LTV must be greater than zero")
    rules, and a simulator should not manufacture violations to describe a
    perfectly ordinary redemption.

    Exits are therefore recovered exactly as production does it — by comparing
    two consecutive reporting dates. That is a stronger test of the historical
    integration and period-change path than restating them on the tape would be.
    """
    profile = get_profile(case.asset_class)
    knobs = resolve_knobs(case.asset_class, case.economic_intent)
    snapshot = history.period(period_index)
    ctx = GenContext(case, period_index, snapshot.reporting_date, knobs)
    return [profile.canonical_row(loan, ctx) for loan in snapshot.loans]


__all__ = ["RECONCILIATION_TOLERANCE", "ReconciliationError",
           "canonical_frame_rows", "generate_history"]
