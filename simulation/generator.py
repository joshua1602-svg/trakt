"""simulation.generator — the deterministic economic generator.

Creates the opening funded book for a case and the new business it writes in
each reporting month. It knows nothing about source formats, canonical fields or
the pipeline: its only output is loan STATE, which :mod:`simulation.history`
rolls forward and :mod:`simulation.dialects` later renders.

Determinism rules this module holds to, so a seed genuinely reproduces a case:

* one :class:`numpy.random.Generator`, created from the manifest seed and passed
  explicitly — there is no module-level randomness anywhere;
* loans are created in a fixed order, so the n-th draw always feeds the n-th
  loan regardless of how many periods are later requested;
* no wall-clock, no filesystem and no environment reads.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .assets import GenContext, get_profile
from .dates import add_month_ends, parse_iso
from .knobs import Knobs
from .models import CaseManifest, LoanState
from .assets import _common as C


def make_rng(seed: int) -> "np.random.Generator":
    """The ONE generator a case uses. PCG64 is stable across numpy versions."""
    return np.random.Generator(np.random.PCG64(int(seed)))


def _loan_id(case: CaseManifest, index: int) -> str:
    return f"{case.portfolio_id.upper()}-{index:07d}"


def build_opening_book(rng, case: CaseManifest, knobs: Knobs,
                       loan_count: int) -> List[LoanState]:
    """The funded book as it stands at the START of the first reporting period.

    Each loan is originated at a seasoning offset drawn from the case's knobs and
    then aged forward to the opening date by the SAME roll-forward the history
    uses — so a seasoned book's balance, valuation and LTV are the genuine result
    of its own economics rather than a separately sampled "current" figure.
    """
    profile = get_profile(case.asset_class)
    start = parse_iso(case.start_reporting_date)
    # The opening book is the book at the start of period 0, i.e. one month
    # before the first reporting date.
    opening_anchor = add_month_ends(start, -1)

    book: List[LoanState] = []
    for i in range(loan_count):
        seasoning = int(rng.integers(knobs.seasoning_months[0],
                                     knobs.seasoning_months[1] + 1))
        origination = add_month_ends(opening_anchor, -seasoning).isoformat()
        cohort = ("acquired"
                  if float(rng.random()) < knobs.opening_acquired_share
                  else "direct")
        loan = profile.open_loan(rng, GenContext(case, 0,
                                                 opening_anchor.isoformat(), knobs),
                                 loan_id=_loan_id(case, i),
                                 origination_date=origination, cohort=cohort)
        loan.borrower_id = C.borrower_id(rng, i, knobs.borrowers_per_hundred,
                                         case.portfolio_id.upper())
        if cohort == "acquired":
            loan.attributes["seller_name"] = "Simulated Portfolio Vendor Ltd"
            loan.attributes["pool_addition_date"] = opening_anchor.isoformat()
        book.append(loan)

    _apply_size_concentration(book, knobs)
    _age_opening_book(rng, book, case, knobs, opening_anchor)
    return book


def _apply_size_concentration(book: List[LoanState], knobs: Knobs) -> None:
    """Scale up the first N loans to create a deliberate size concentration.

    Applied at ORIGINATION, before ageing, so the enlarged loans amortise, accrue
    and revalue exactly like every other loan — the concentration is economic,
    not painted on at the reporting date.
    """
    if knobs.large_loan_multiplier <= 1.0 or knobs.large_loan_count <= 0:
        return
    for loan in book[:knobs.large_loan_count]:
        factor = float(knobs.large_loan_multiplier)
        loan.original_principal = round(loan.original_principal * factor, 2)
        loan.opening_balance = round(loan.opening_balance * factor, 2)
        loan.closing_balance = round(loan.closing_balance * factor, 2)
        loan.original_value = round(loan.original_value * factor, 2)
        loan.current_value = round(loan.current_value * factor, 2)
        for key in ("credit_limit", "balloon_amount", "monthly_capital",
                    "monthly_payment_due", "original_residual_value",
                    "current_residual_value", "reserve_facility"):
            if key in loan.attributes:
                loan.attributes[key] = round(float(loan.attributes[key]) * factor, 2)


def _age_opening_book(rng, book: List[LoanState], case: CaseManifest,
                      knobs: Knobs, opening_anchor) -> None:
    """Age each opening loan from origination to the opening date.

    Ageing uses a CLEAN knob set — no new business, no exits — because the
    opening book is a stock, not a history: whatever left before the simulation
    started is not on it. Interest, amortisation and revaluation still run, so
    the stock is internally consistent.
    """
    profile = get_profile(case.asset_class)
    # Exits are suppressed (whatever left before the simulation started is not
    # on the opening book) but the impairment ladder is NOT: a seasoned book
    # genuinely carries some arrears at its opening date, and suppressing that
    # would make every history start artificially clean.
    quiet = knobs.with_(redemption_rate=0.0, write_off_rate=0.0,
                        repossession_rate=0.0, extension_rate=0.0)
    for loan in book:
        months = C.months_to_maturity(loan.origination_date,
                                      opening_anchor.isoformat())
        # ``months_to_maturity`` is months_between(a, b) — here, months of
        # seasoning from origination to the opening date.
        for step in range(months):
            date = add_month_ends(parse_iso(loan.origination_date), step + 1)
            ctx = GenContext(case, 0, date.isoformat(), quiet)
            profile.roll(rng, loan, ctx)
            if not loan.is_open:
                # A quiet roll should never close a loan; if the asset profile
                # closes it on contractual maturity, restart the contract so the
                # opening book is genuinely on-book at the opening date.
                _reopen_matured(loan, opening_anchor)
        loan.movements = {}
        loan.opening_balance = loan.closing_balance


def _reopen_matured(loan: LoanState, opening_anchor) -> None:
    """A contract that matured during ageing is re-dated to still be on book.

    Only reachable for finite-term assets whose drawn seasoning exceeded their
    drawn term. Re-dating (rather than discarding) keeps the opening loan count
    exactly equal to the manifest's ``opening_loan_count``, which the reference
    truth relies on.
    """
    from .models import STATUS_PERFORMING
    term = int(loan.attributes.get("original_term_months", 12))
    loan.status = STATUS_PERFORMING
    loan.exit_date = None
    loan.exit_reason = None
    loan.attributes.pop("redemption_date", None)
    # Re-dating restarts the CONTRACT, so its impairment history restarts too.
    # Leaving arrears behind on a performing loan would breach the platform's
    # own ARR001 rule — and would be the simulator manufacturing the violation.
    loan.attributes["arrears_months"] = 0
    loan.attributes["arrears_balance"] = 0.0
    loan.attributes.pop("default_date", None)
    loan.closing_balance = loan.original_principal
    loan.origination_date = add_month_ends(opening_anchor, -max(1, term // 2)).isoformat()
    loan.funding_date = loan.origination_date
    loan.attributes["maturity_date"] = C.maturity_from_term(
        loan.origination_date, term)
    loan.attributes["lease_expiry_date"] = loan.attributes["maturity_date"]


def originate(rng, case: CaseManifest, knobs: Knobs, *, count: int,
              period_index: int, reporting_date: str, next_index: int,
              cohort: str = "direct") -> List[LoanState]:
    """Loans funded (or acquired) DURING a reporting period.

    A new loan enters at its advance and records that advance as its movement,
    so the period reconciles: opening + additions - reductions == closing.
    """
    from .models import MOVE_ACQUIRED, MOVE_FUNDED
    profile = get_profile(case.asset_class)
    ctx = GenContext(case, period_index, reporting_date, knobs)
    kind = MOVE_ACQUIRED if cohort == "acquired" else MOVE_FUNDED

    out: List[LoanState] = []
    for offset in range(count):
        loan = profile.open_loan(rng, ctx, loan_id=_loan_id(case, next_index + offset),
                                 origination_date=reporting_date, cohort=cohort)
        loan.borrower_id = C.borrower_id(rng, next_index + offset,
                                         knobs.borrowers_per_hundred,
                                         case.portfolio_id.upper())
        loan.attributes["pool_addition_date"] = reporting_date
        if cohort == "acquired":
            loan.attributes["seller_name"] = "Simulated Portfolio Vendor Ltd"
        loan.opening_balance = 0.0
        loan.movements = {}
        loan.record(kind, loan.closing_balance)
        out.append(loan)
    return out


def period_volume(rng, rate: float, book_size: int) -> int:
    """How many loans a rate implies for a book of *book_size*.

    The fractional remainder is resolved by a draw rather than by rounding, so a
    2% rate on a 100-loan book genuinely averages two loans a month instead of
    always producing exactly two.
    """
    expected = float(rate) * max(0, int(book_size))
    whole = int(expected)
    return whole + (1 if float(rng.random()) < (expected - whole) else 0)


__all__ = ["build_opening_book", "make_rng", "originate", "period_volume"]
