"""mi_agent.borrowing_base.calculator — the pure deterministic calculation.

No I/O, no clock, no configuration lookup, no dashboard import: inputs in,
figures out. Everything it needs — the trusted canonical frame, the approved
facility terms, the governed eligibility already derived onto the frame, and
the governed concentration results — is passed in.

THE EQUATIONS, exactly as implemented:

    financing_portfolio_balance   = Σ current balance where the loan is in the
                                    facility's Financing Portfolio
    eligible_current_balance      = Σ current balance where status == ELIGIBLE
    concentration_limit_denominator
                                  = MAX(concentration_denominator_floor,
                                        eligible_current_balance)
    gross_borrowing_base          = eligible_current_balance × advance_rate
    available_borrowing_base      = MIN(gross_borrowing_base,
                                        facility_commitment)
    borrowing_base_headroom       = available_borrowing_base
                                    − current_drawn_amount
    borrowing_base_deficiency     = ABS(MIN(borrowing_base_headroom, 0))
    borrowing_base_utilisation_pct= current_drawn_amount
                                    ÷ available_borrowing_base × 100
    facility_utilisation_pct      = current_drawn_amount
                                    ÷ facility_commitment × 100

Nothing else is applied. There is no haircut, no reserve, no
overcollateralisation, and no deduction for a concentration breach — Schedule 8
establishes the concentration tests and is silent on what breaching one does to
the borrowing base, so v1 monitors and deducts nothing. The seam for a future
approved treatment is :func:`concentration_adjustment`, which refuses any
treatment other than ``monitor_only`` rather than inventing one.

The headroom is retained NEGATIVE when the facility is over-drawn. Presenting
it as £0 alongside a deficiency is a UI decision; the governed number is the
real one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from analytics_lib.numeric import coerce_numeric

from .eligibility import (
    eligible_mask,
    in_financing_portfolio_mask,
    status_mask,
)
from .models import (
    ELIGIBLE,
    INELIGIBLE,
    NOT_CALCULABLE,
    PROTOTYPE_ASSUMPTION_NOTE,
    TREATMENT_MONITOR_ONLY,
    UNDETERMINED,
    FacilityConfiguration,
)

#: A measure is either a number or the NOT_CALCULABLE sentinel.
Measure = Union[float, str, None]

#: Money is rounded to the penny at the boundary so two callers cannot disagree
#: in the fifteenth decimal place. The rounding is applied ONCE, to the output.
_MONEY_DP = 2
_PCT_DP = 4

#: Candidate current-balance columns, in the same order the governed field-role
#: vocabulary declares them. Kept as a literal fallback so the calculator stays
#: usable on a frame prepared without the concentration library present.
_BALANCE_CANDIDATES = ("current_outstanding_balance", "current_principal_balance")


class ReconciliationError(ValueError):
    """A reconciliation invariant failed. The calculation is refused."""


def _round(value: Optional[float], dp: int = _MONEY_DP) -> Optional[float]:
    return None if value is None else round(float(value), dp)


def _balance_column(df: pd.DataFrame, lib: Any = None) -> Optional[str]:
    if lib is not None:
        try:
            from mi_agent.concentration_tests.metrics import resolve_role_column
            col = resolve_role_column(df, lib, "balance_current")
            if col:
                return col
        except Exception:  # noqa: BLE001
            pass
    for col in _BALANCE_CANDIDATES:
        if col in getattr(df, "columns", []) and df[col].notna().any():
            return col
    return None


# --------------------------------------------------------------------------- #
# Eligibility reconciliation
# --------------------------------------------------------------------------- #
@dataclass
class EligibilitySummary:
    """The facility's population, partitioned exactly once."""

    financing_portfolio_loan_count: int = 0
    financing_portfolio_balance: float = 0.0
    eligible_loan_count: int = 0
    eligible_current_balance: float = 0.0
    ineligible_loan_count: int = 0
    ineligible_current_balance: float = 0.0
    undetermined_loan_count: int = 0
    undetermined_current_balance: float = 0.0
    balance_column: str = ""
    invariants: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def reconciles(self) -> bool:
        return all(i["holds"] for i in self.invariants)

    def share_of_financing_portfolio(self, balance: float) -> Optional[float]:
        if not self.financing_portfolio_balance:
            return None
        return round(balance / self.financing_portfolio_balance * 100.0, _PCT_DP)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "financingPortfolioLoanCount": self.financing_portfolio_loan_count,
            "financingPortfolioBalance": self.financing_portfolio_balance,
            "eligibleLoanCount": self.eligible_loan_count,
            "eligibleCurrentBalance": self.eligible_current_balance,
            "eligibleShareOfFinancingPortfolioPct":
                self.share_of_financing_portfolio(self.eligible_current_balance),
            "ineligibleLoanCount": self.ineligible_loan_count,
            "ineligibleCurrentBalance": self.ineligible_current_balance,
            "ineligibleShareOfFinancingPortfolioPct":
                self.share_of_financing_portfolio(self.ineligible_current_balance),
            "undeterminedLoanCount": self.undetermined_loan_count,
            "undeterminedCurrentBalance": self.undetermined_current_balance,
            "undeterminedShareOfFinancingPortfolioPct":
                self.share_of_financing_portfolio(
                    self.undetermined_current_balance),
            "balanceColumn": self.balance_column,
            "invariants": self.invariants,
            "reconciles": self.reconciles,
        }


def summarise_eligibility(df: pd.DataFrame, facility: FacilityConfiguration,
                          *, lib: Any = None) -> EligibilitySummary:
    """Partition the Financing Portfolio and check that it partitions.

    The three statuses are read from the SAME governed column with three
    mutually exclusive comparisons, so a loan cannot be in two of them; the
    invariants below then prove that no loan fell out of all three.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        summary = EligibilitySummary()
        summary.invariants = _invariants(summary)
        return summary

    column = _balance_column(df, lib)
    balance = (coerce_numeric(df[column]) if column
               else pd.Series(0.0, index=df.index))
    balance = balance.fillna(0.0)

    scope = in_financing_portfolio_mask(df, facility)
    masks = {
        ELIGIBLE: eligible_mask(df, facility),
        INELIGIBLE: status_mask(df, INELIGIBLE, facility),
        UNDETERMINED: status_mask(df, UNDETERMINED, facility),
    }
    summary = EligibilitySummary(
        financing_portfolio_loan_count=int(scope.sum()),
        financing_portfolio_balance=_round(float(balance[scope].sum())),
        eligible_loan_count=int(masks[ELIGIBLE].sum()),
        eligible_current_balance=_round(float(balance[masks[ELIGIBLE]].sum())),
        ineligible_loan_count=int(masks[INELIGIBLE].sum()),
        ineligible_current_balance=_round(
            float(balance[masks[INELIGIBLE]].sum())),
        undetermined_loan_count=int(masks[UNDETERMINED].sum()),
        undetermined_current_balance=_round(
            float(balance[masks[UNDETERMINED]].sum())),
        balance_column=column or "",
    )
    overlap = int(((masks[ELIGIBLE] & masks[INELIGIBLE])
                   | (masks[ELIGIBLE] & masks[UNDETERMINED])
                   | (masks[INELIGIBLE] & masks[UNDETERMINED])).sum())
    summary.invariants = _invariants(summary, overlap=overlap)
    return summary


def _invariants(summary: EligibilitySummary, *, overlap: int = 0
                ) -> List[Dict[str, Any]]:
    """The reconciliation invariants, each with the numbers that prove it."""
    count_parts = (summary.eligible_loan_count + summary.ineligible_loan_count
                   + summary.undetermined_loan_count)
    balance_parts = round(summary.eligible_current_balance
                          + summary.ineligible_current_balance
                          + summary.undetermined_current_balance, _MONEY_DP)
    # A penny of tolerance, and only a penny: the parts are each rounded to the
    # penny before being summed, so three roundings can differ from one by at
    # most 1.5p. Anything larger is a real gap, not a rounding artefact.
    balance_holds = abs(balance_parts - summary.financing_portfolio_balance) <= 0.02
    return [
        {
            "invariant": "eligibility_balances_partition_financing_portfolio",
            "statement": ("financing_portfolio_balance = "
                          "eligible + ineligible + undetermined"),
            "expected": summary.financing_portfolio_balance,
            "actual": balance_parts,
            "holds": bool(balance_holds),
        },
        {
            "invariant": "eligibility_counts_partition_financing_portfolio",
            "statement": ("eligible + ineligible + undetermined loan count = "
                          "financing_portfolio_loan_count"),
            "expected": summary.financing_portfolio_loan_count,
            "actual": count_parts,
            "holds": bool(count_parts == summary.financing_portfolio_loan_count),
        },
        {
            "invariant": "no_loan_holds_two_eligibility_statuses",
            "statement": ("no loan is simultaneously ELIGIBLE, INELIGIBLE or "
                          "UNDETERMINED"),
            "expected": 0,
            "actual": int(overlap),
            "holds": bool(overlap == 0),
        },
    ]


# --------------------------------------------------------------------------- #
# The concentration-breach seam
# --------------------------------------------------------------------------- #
def concentration_adjustment(facility: FacilityConfiguration,
                             concentration_results: Optional[List[Dict[str, Any]]]
                             ) -> Dict[str, Any]:
    """How much a concentration breach removes from the borrowing base.

    v1: nothing, always, because the supplied Schedule 8 states the tests and
    not the contractual consequence of failing one. The excess exposure is
    still measured and reported — it is simply not deducted.

    The signature is the seam a future approved treatment plugs into. An
    unrecognised or not-yet-implemented treatment REFUSES rather than guessing:
    silently falling back to monitor_only would present an unadjusted borrowing
    base as if the configured adjustment had been applied.
    """
    treatment = facility.borrowing_base_treatment
    if treatment != TREATMENT_MONITOR_ONLY:
        raise ReconciliationError(
            f"borrowing_base_treatment '{treatment}' is configured but no "
            "approved contractual rule for it has been implemented. The "
            "borrowing base is refused rather than calculated on a guess.")
    breached = [t for t in (concentration_results or [])
                if str(t.get("status")) == "breach"]
    excess = [t.get("breachAmount") for t in breached
              if isinstance(t.get("breachAmount"), (int, float))]
    return {
        "treatment": treatment,
        "amount": 0.0,
        "breachedTestCount": len(breached),
        "excessMeasuredPct": round(sum(excess), _PCT_DP) if excess else None,
        "note": ("Concentration breaches are monitored. Schedule 8 as supplied "
                 "establishes the tests but not the contractual consequence of "
                 "a breach for the borrowing base, so nothing is deducted."),
    }


# --------------------------------------------------------------------------- #
# Nearest / binding concentration — deterministic, never modelled
# --------------------------------------------------------------------------- #
def nearest_concentration(concentration_results: Optional[List[Dict[str, Any]]]
                          ) -> Dict[str, Any]:
    """The binding concentration limit, chosen arithmetically.

    Ranking, in order: any breach outranks any non-breach; among breaches the
    largest utilisation binds; among non-breaches the smallest headroom binds.
    Ties break on testId so the answer is stable across runs. No model, no LLM,
    no judgement.
    """
    tests = [t for t in (concentration_results or [])
             if t.get("currentValue") is not None
             and t.get("headroom") is not None]
    breached = [t for t in (concentration_results or [])
                if str(t.get("status")) == "breach"]

    def _utilisation(t: Dict[str, Any]) -> float:
        u = t.get("utilization")
        return float(u) if isinstance(u, (int, float)) else float("-inf")

    binding: Optional[Dict[str, Any]] = None
    if breached:
        binding = sorted(breached,
                         key=lambda t: (-_utilisation(t), str(t.get("testId"))))[0]
    elif tests:
        binding = sorted(tests,
                         key=lambda t: (float(t["headroom"]),
                                        str(t.get("testId"))))[0]

    return {
        "nearest_concentration_limit": (binding or {}).get("displayName")
        if binding else NOT_CALCULABLE,
        "nearest_concentration_limit_test_id": (binding or {}).get("testId")
        if binding else None,
        "nearest_concentration_headroom_pct": (
            binding.get("headroom") if binding
            and str(binding.get("unit") or "percent") == "percent"
            else (NOT_CALCULABLE if binding is None else None)),
        "nearest_concentration_headroom_amount": (
            _headroom_amount(binding) if binding else NOT_CALCULABLE),
        "nearest_concentration_utilisation_pct": (
            binding.get("utilization") if binding else NOT_CALCULABLE),
        "nearest_concentration_status": (
            binding.get("status") if binding else NOT_CALCULABLE),
        "breached_concentration_count": len(breached),
        "breached_concentrations": [
            {"testId": t.get("testId"), "displayName": t.get("displayName"),
             "currentValue": t.get("currentValue"), "threshold": t.get("threshold"),
             "unit": t.get("unit"), "utilization": t.get("utilization"),
             "breachAmount": t.get("breachAmount"),
             "excessAmount": _headroom_amount(t, excess=True),
             "denominatorValue": t.get("denominatorValue")}
            for t in sorted(breached,
                            key=lambda t: (-_utilisation(t),
                                           str(t.get("testId"))))],
    }


def _headroom_amount(test: Optional[Dict[str, Any]], *, excess: bool = False
                     ) -> Optional[float]:
    """Headroom (or excess) in currency, where the maths permits it.

    A percentage test measured against a stated denominator converts exactly:
    headroom_pct × denominator ÷ 100. A test with no denominator (an average, a
    count, a rate) has no currency headroom and returns None rather than a
    number built from a denominator it never used.
    """
    if not test:
        return None
    unit = str(test.get("unit") or "percent")
    denominator = test.get("denominatorValue")
    value = test.get("breachAmount") if excess else test.get("headroom")
    if unit != "percent" or not isinstance(denominator, (int, float)) \
            or not isinstance(value, (int, float)):
        return None
    return _round(value / 100.0 * float(denominator))


# --------------------------------------------------------------------------- #
# The facility calculation
# --------------------------------------------------------------------------- #
@dataclass
class BorrowingBaseResult:
    """Every governed borrowing-base figure, plus what could not be produced."""

    facility_id: str = ""
    currency: str = "GBP"
    eligibility: EligibilitySummary = field(default_factory=EligibilitySummary)

    concentration_limit_denominator: Measure = None
    concentration_limit_denominator_floor: Optional[float] = None
    concentration_denominator_floor_binding: Optional[bool] = None

    advance_rate: Optional[float] = None
    gross_borrowing_base: Measure = None
    facility_commitment: Optional[float] = None
    available_borrowing_base: Measure = None
    facility_cap_binding: Optional[bool] = None

    current_drawn_amount: Measure = None
    borrowing_base_headroom: Measure = None
    borrowing_base_deficiency: Measure = None
    borrowing_base_utilisation_pct: Measure = None
    facility_utilisation_pct: Measure = None

    concentration_adjustment: Dict[str, Any] = field(default_factory=dict)
    nearest: Dict[str, Any] = field(default_factory=dict)

    missing_inputs: List[str] = field(default_factory=list)
    prototype_assumptions_used: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def reconciles(self) -> bool:
        return self.eligibility.reconciles

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facilityId": self.facility_id,
            "currency": self.currency,
            **self.eligibility.to_dict(),
            "concentrationLimitDenominator": self.concentration_limit_denominator,
            "concentrationLimitDenominatorFloor":
                self.concentration_limit_denominator_floor,
            "concentrationDenominatorFloorBinding":
                self.concentration_denominator_floor_binding,
            "advanceRate": self.advance_rate,
            "advanceRatePct": (None if self.advance_rate is None
                               else round(self.advance_rate * 100.0, 6)),
            "grossBorrowingBase": self.gross_borrowing_base,
            "facilityCommitment": self.facility_commitment,
            "availableBorrowingBase": self.available_borrowing_base,
            "facilityCapBinding": self.facility_cap_binding,
            "currentDrawnAmount": self.current_drawn_amount,
            "borrowingBaseHeadroom": self.borrowing_base_headroom,
            "borrowingBaseDeficiency": self.borrowing_base_deficiency,
            "borrowingBaseUtilisationPct": self.borrowing_base_utilisation_pct,
            "facilityUtilisationPct": self.facility_utilisation_pct,
            "concentrationAdjustment": self.concentration_adjustment,
            **{_camel(k): v for k, v in self.nearest.items()},
            "missingInputs": self.missing_inputs,
            "prototypeAssumptionsUsed": self.prototype_assumptions_used,
            "notes": self.notes,
            "reconciles": self.reconciles,
        }


def _camel(snake: str) -> str:
    head, *rest = snake.split("_")
    return head + "".join(p.title() for p in rest)


def calculate(
    df: Optional[pd.DataFrame],
    facility: FacilityConfiguration,
    *,
    concentration_results: Optional[List[Dict[str, Any]]] = None,
    lib: Any = None,
    strict: bool = True,
) -> BorrowingBaseResult:
    """The facility calculation. Pure: same inputs, same figures, every time.

    ``strict`` (the default) raises :class:`ReconciliationError` when the
    eligibility population does not reconcile — fail closed, per the governance
    rules. ``strict=False`` returns the result with the failed invariants
    attached, for a diagnostic surface that needs to SHOW the failure rather
    than be stopped by it.
    """
    summary = summarise_eligibility(df, facility, lib=lib)
    result = BorrowingBaseResult(
        facility_id=facility.facility_id,
        currency=facility.currency,
        eligibility=summary,
        concentration_limit_denominator_floor=(
            facility.concentration_denominator_floor),
        advance_rate=facility.advance_rate,
        facility_commitment=facility.commitment,
    )

    if not summary.reconciles:
        failed = [i["invariant"] for i in summary.invariants if not i["holds"]]
        if strict:
            raise ReconciliationError(
                "The borrowing-base eligibility population does not reconcile "
                f"({', '.join(failed)}). The calculation is refused.")
        result.notes.append(
            "Reconciliation failed: " + ", ".join(failed)
            + ". Figures below are diagnostic only.")

    # Declared from the FACILITY, not from a receipt plumbed in from
    # elsewhere. An assumption that only reaches the envelope when some caller
    # remembers to pass a derivation receipt is an assumption that will
    # eventually go unreported — which is the one thing it must never do.
    if facility.prototype_assumption_active:
        result.prototype_assumptions_used.append(PROTOTYPE_ASSUMPTION_NOTE)

    result.concentration_adjustment = concentration_adjustment(
        facility, concentration_results)
    result.nearest = nearest_concentration(concentration_results)

    eligible = summary.eligible_current_balance

    # --- Concentration Limit Denominator ---------------------------------- #
    floor = facility.concentration_denominator_floor
    if floor is None:
        result.concentration_limit_denominator = NOT_CALCULABLE
        result.missing_inputs.append("concentration_denominator_floor")
        result.notes.append(
            "No Concentration Limit Denominator floor is configured, so the "
            "contractual denominator cannot be formed.")
    else:
        result.concentration_limit_denominator = _round(max(floor, eligible))
        result.concentration_denominator_floor_binding = bool(floor > eligible)

    # --- Gross borrowing base --------------------------------------------- #
    if facility.advance_rate is None:
        result.gross_borrowing_base = NOT_CALCULABLE
        result.missing_inputs.append("advance_rate")
    else:
        result.gross_borrowing_base = _round(eligible * facility.advance_rate)

    # --- Available borrowing base ----------------------------------------- #
    gross = result.gross_borrowing_base
    if not isinstance(gross, (int, float)):
        result.available_borrowing_base = NOT_CALCULABLE
    elif facility.commitment is None:
        # Uncapped is not the same as unknown. With no commitment the facility
        # cap cannot bind, so the available base is the gross base, and the
        # absence is disclosed rather than treated as an infinite commitment.
        result.available_borrowing_base = gross
        result.facility_cap_binding = None
        result.missing_inputs.append("facility_commitment")
        result.notes.append(
            "No facility commitment is configured; the borrowing base is "
            "shown uncapped.")
    else:
        result.available_borrowing_base = _round(min(gross, facility.commitment))
        result.facility_cap_binding = bool(facility.commitment < gross)

    # --- Drawings, headroom, deficiency, utilisation ----------------------- #
    available = result.available_borrowing_base
    if not facility.drawn_available:
        result.current_drawn_amount = NOT_CALCULABLE
        result.borrowing_base_headroom = NOT_CALCULABLE
        result.borrowing_base_deficiency = NOT_CALCULABLE
        result.borrowing_base_utilisation_pct = NOT_CALCULABLE
        result.facility_utilisation_pct = NOT_CALCULABLE
        result.missing_inputs.append("current_drawn_amount")
        result.notes.append(
            "The current drawn amount under the facility has not been "
            "supplied, so headroom, utilisation and any deficiency cannot be "
            "calculated. The borrowing base itself is unaffected.")
    else:
        drawn = _round(float(facility.current_drawn_amount))
        result.current_drawn_amount = drawn
        if isinstance(available, (int, float)):
            # The governed headroom keeps its sign. Flooring it here would
            # erase an over-draw everywhere downstream.
            headroom = _round(available - drawn)
            result.borrowing_base_headroom = headroom
            result.borrowing_base_deficiency = _round(abs(min(headroom, 0.0)))
            result.borrowing_base_utilisation_pct = (
                _round(drawn / available * 100.0, _PCT_DP) if available
                else NOT_CALCULABLE)
            if not available:
                result.notes.append(
                    "Borrowing-base utilisation is not calculable against a "
                    "zero available borrowing base.")
        else:
            result.borrowing_base_headroom = NOT_CALCULABLE
            result.borrowing_base_deficiency = NOT_CALCULABLE
            result.borrowing_base_utilisation_pct = NOT_CALCULABLE
        if facility.commitment:
            result.facility_utilisation_pct = _round(
                drawn / facility.commitment * 100.0, _PCT_DP)
        else:
            result.facility_utilisation_pct = NOT_CALCULABLE

    return result


__all__ = [
    "BorrowingBaseResult", "EligibilitySummary", "ReconciliationError",
    "calculate", "summarise_eligibility", "nearest_concentration",
    "concentration_adjustment",
]
