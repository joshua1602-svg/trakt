"""mi_agent.borrowing_base.analysis — the borrowing-base analysis helper.

Three small, deterministic compositions over what the governed engine has
ALREADY produced, kept in one module so the MI Query Agent has exactly one
analysis owner beside the calculator:

1. **Ineligibility reason summary** — "why are loans ineligible?", read off the
   two governed eligibility columns over the complete Financing Portfolio.
2. **Period validity and change** — which historical periods the facility
   configuration and its drawing are demonstrably valid for, and the change in
   a named measure between two governed envelopes.
3. **Borrowing-base bridge** — opening → three exact drivers → closing, from two
   governed envelopes.

Nothing here reads a loan to decide anything, applies an advance rate to a
balance, or copies an equation from ``calculator.py``. Every figure the module
handles was produced by the calculator for one period; this module partitions,
validates and subtracts.

----------------------------------------------------------------------------
1. Ineligibility reasons
----------------------------------------------------------------------------
Only ``status == INELIGIBLE`` is summarised. UNDETERMINED is a different
governed state — a missing input is not an ineligibility failure — and is never
mixed in.

ONE PRIMARY REASON PER LOAN. The derivation applies the approved rules in their
configured order and stamps the FIRST failing rule's reason code; a loan that
fails several rules carries the first one and is counted once, under it. The
table is therefore a breakdown by PRIMARY governed reason. It does not
enumerate every rule a loan failed, and it must not be read as if it did.

The totals are the calculator's own :func:`summarise_eligibility` partition, so
the table reconciles to the SAME figures the borrowing-base envelope and the
dashboard's eligibility split carry, or it refuses:

    Σ reason loan_count = ineligible loan count           (exact)
    Σ reason balance    = ineligible current balance      (penny tolerance)
    every INELIGIBLE loan carries exactly one reason code

Reason descriptions come from the APPROVED facility eligibility rule whose
``reason_code`` (or ``rule_id``) the loan carries — contractual wording, never
generated text.

----------------------------------------------------------------------------
2. Period validity — what the platform can and cannot demonstrate
----------------------------------------------------------------------------
The platform holds ONE facility configuration: one commitment, one advance
rate, one set of eligibility rules, one operator-supplied drawing with its
``current_drawn_amount_as_of``. Historical frames are stamped with eligibility
under that current configuration. None of that is historical facility state,
and none is invented here:

* **Configuration applicability.** A configuration is demonstrably applicable
  to a historical period only from a GOVERNED approval date —
  ``governance.approved_at`` in the facility register. ``effective_date`` is
  operator-entered metadata that no engine module reads and OCC does not
  govern, so it is NOT used as a cutoff. A configuration with no approval date
  (the prototype's situation) is applicable to the CURRENT position only —
  the position the dashboard presents it against — and every historical
  measure is NOT_CALCULABLE with the limitation named.

* **Drawings.** ``current_drawn_amount`` is time-varying. It is valid for a
  snapshot ONLY when its governed ``current_drawn_amount_as_of`` equals that
  snapshot's reporting date. Being the latest frame is not evidence. For any
  other period the drawing is WITHHELD from the calculator, which then
  reports drawn, headroom, deficiency and both utilisations as NOT_CALCULABLE
  itself — this module never rewrites a figure the calculator produced.

----------------------------------------------------------------------------
3. The bridge — exact by construction
----------------------------------------------------------------------------
From the calculator's own statements ``gross = eligible × advance rate`` and
``available = MIN(gross, commitment)``, define per envelope
``cap effect = available − gross``. Then, in this deterministic order:

    opening borrowing base
    + eligible collateral effect = (E_c − E_o) × a_o        (opening rate)
    + advance-rate effect        = E_c × (a_c − a_o)        (closing collateral)
    + change in cap effect       = (B_c − G_c) − (B_o − G_o)
    = closing borrowing base

The three effects sum to B_c − B_o exactly; the only slack is the calculator's
penny rounding of each gross base (≤ ½p each) and the penny rounding of each
effect here (≤ ½p each), so the disclosed tolerance is five pence. A bridge
outside it is REFUSED. No haircut, reserve, concentration or
overcollateralisation line exists because the v1 calculator applies none.

Headroom, where BOTH periods carry a period-valid drawing:
``opening headroom + Δ borrowing base − Δ drawn = closing headroom``; otherwise
NOT_CALCULABLE with the missing input named.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from analytics_lib.numeric import coerce_numeric

from .calculator import (
    _MONEY_DP,
    _PCT_DP,
    _balance_column,
    summarise_eligibility,
)
from .eligibility import eligibility_available, status_mask
from .models import (
    FIELD_ELIGIBILITY_REASON,
    INELIGIBLE,
    NOT_CALCULABLE,
    FacilityConfiguration,
)

_BLANK_TOKENS = ("", "nan", "none", "nat", "<na>", "null")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _round(value: float, dp: int = _MONEY_DP) -> float:
    return round(float(value), dp)


def _pct(part: float, whole: float) -> Any:
    if not whole:
        return NOT_CALCULABLE
    return round(float(part) / float(whole) * 100.0, _PCT_DP)


# =========================================================================== #
# 1. Ineligibility reason summary
# =========================================================================== #
#: The code a loan carries when the derivation stamped no reason on it. It
#: cannot legitimately occur on an INELIGIBLE loan, so its presence is an
#: invariant failure, reported by name rather than summed into a row.
REASON_MISSING = "reason_missing"


@dataclass
class ReasonRow:
    """One PRIMARY governed ineligibility reason and the loans carrying it."""

    reason_code: str
    reason_description: str
    rule_id: Optional[str]
    loan_count: int
    balance: float
    share_of_ineligible_loans_pct: Any
    share_of_ineligible_balance_pct: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasonCode": self.reason_code,
            "reasonDescription": self.reason_description,
            "ruleId": self.rule_id,
            "loanCount": self.loan_count,
            "balance": self.balance,
            "shareOfIneligibleLoansPct": self.share_of_ineligible_loans_pct,
            "shareOfIneligibleBalancePct": self.share_of_ineligible_balance_pct,
        }


@dataclass
class IneligibilityReasonSummary:
    """The reason table, the totals it reconciles to, and the proof."""

    available: bool
    reason: str = ""
    facility_id: str = ""
    eligibility_rule_version: str = ""
    balance_column: str = ""
    financing_portfolio_loan_count: int = 0
    financing_portfolio_balance: float = 0.0
    ineligible_loan_count: int = 0
    ineligible_balance: float = 0.0
    ineligible_loan_share_pct: Any = NOT_CALCULABLE
    ineligible_balance_share_pct: Any = NOT_CALCULABLE
    rows: List[ReasonRow] = field(default_factory=list)
    invariants: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    #: Stated once, carried on every serialisation, so no consumer can present
    #: the table as an enumeration of every failed rule.
    basis: str = ("primary_reason: one governed reason per INELIGIBLE loan — "
                  "the first failing approved rule in configured order")

    @property
    def reconciles(self) -> bool:
        return bool(self.invariants) and all(i["holds"] for i in self.invariants)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason or None,
            "basis": self.basis,
            "facilityId": self.facility_id,
            "eligibilityRuleVersion": self.eligibility_rule_version or None,
            "balanceColumn": self.balance_column,
            "financingPortfolioLoanCount": self.financing_portfolio_loan_count,
            "financingPortfolioBalance": self.financing_portfolio_balance,
            "ineligibleLoanCount": self.ineligible_loan_count,
            "ineligibleBalance": self.ineligible_balance,
            "ineligibleLoanSharePct": self.ineligible_loan_share_pct,
            "ineligibleBalanceSharePct": self.ineligible_balance_share_pct,
            "rows": [r.to_dict() for r in self.rows],
            "invariants": self.invariants,
            "reconciles": self.reconciles,
            "notes": self.notes,
        }


def rule_descriptions(facility: FacilityConfiguration
                      ) -> Dict[str, Dict[str, str]]:
    """``{reason_code: {rule_id, description}}`` from the APPROVED rules.

    The code a loan carries is ``rule.reason_code or rule.rule_id`` — exactly
    what the derivation writes — so the lookup is by that same expression.
    """
    out: Dict[str, Dict[str, str]] = {}
    for rule in facility.eligibility_rules:
        code = rule.reason_code or rule.rule_id
        if not code or code in out:
            continue
        out[code] = {"rule_id": rule.rule_id,
                     "description": rule.reason or rule.description or code}
    return out


def _blank(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return series.isna() | text.isin(_BLANK_TOKENS)


def summarise_ineligibility_reasons(
    df: Optional[pd.DataFrame],
    facility: FacilityConfiguration,
    *,
    lib: Any = None,
) -> IneligibilityReasonSummary:
    """The primary-reason table over the complete governed Financing Portfolio.

    Fails closed: an eligibility population that does not reconcile, or an
    INELIGIBLE loan with no reason, makes the summary ``available: False``
    with the failed invariant named. The rows are still returned so a
    diagnostic surface can SHOW the failure; a consumer must read
    ``available`` before presenting them as governed.
    """
    out = IneligibilityReasonSummary(
        available=False, facility_id=facility.facility_id,
        eligibility_rule_version=facility.eligibility_rule_version)

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        out.reason = "No funded data for this period."
        return out
    if not eligibility_available(df):
        out.reason = ("No governed eligibility determination exists for this "
                      "book, so ineligibility cannot be broken down by reason.")
        return out

    # THE TOTALS ARE THE CALCULATOR'S: same partition, same balance column,
    # same rounding as the borrowing-base envelope.
    summary = summarise_eligibility(df, facility, lib=lib)
    out.balance_column = summary.balance_column
    out.financing_portfolio_loan_count = summary.financing_portfolio_loan_count
    out.financing_portfolio_balance = summary.financing_portfolio_balance
    out.ineligible_loan_count = summary.ineligible_loan_count
    out.ineligible_balance = summary.ineligible_current_balance
    out.ineligible_loan_share_pct = _pct(summary.ineligible_loan_count,
                                         summary.financing_portfolio_loan_count)
    out.ineligible_balance_share_pct = _pct(
        summary.ineligible_current_balance, summary.financing_portfolio_balance)

    column = _balance_column(df, lib)
    balance = (coerce_numeric(df[column]) if column
               else pd.Series(0.0, index=df.index)).fillna(0.0)
    mask = status_mask(df, INELIGIBLE, facility)
    reasons = (df[FIELD_ELIGIBILITY_REASON]
               if FIELD_ELIGIBILITY_REASON in df.columns
               else pd.Series(pd.NA, index=df.index, dtype="object"))
    sub_reasons, sub_balance = reasons[mask], balance[mask]
    blank = _blank(sub_reasons)
    codes = sub_reasons.astype(str).str.strip().where(~blank, REASON_MISSING)

    descriptions = rule_descriptions(facility)
    rows: List[ReasonRow] = []
    if len(codes):
        grouped = pd.DataFrame({"code": codes.to_numpy(),
                                "balance": sub_balance.to_numpy()})
        agg = grouped.groupby("code", sort=False)["balance"].agg(["count", "sum"])
        for code, row in agg.iterrows():
            code = str(code)
            approved = descriptions.get(code)
            if approved is None and code != REASON_MISSING:
                out.notes.append(
                    f"Reason code '{code}' is carried by INELIGIBLE loans but "
                    "no currently approved eligibility rule declares it; the "
                    "code is shown verbatim.")
            rows.append(ReasonRow(
                reason_code=code,
                reason_description=(
                    approved["description"] if approved
                    else ("No reason recorded" if code == REASON_MISSING
                          else code)),
                rule_id=approved["rule_id"] if approved else None,
                loan_count=int(row["count"]),
                balance=_round(row["sum"]),
                share_of_ineligible_loans_pct=_pct(
                    int(row["count"]), summary.ineligible_loan_count),
                share_of_ineligible_balance_pct=_pct(
                    float(row["sum"]), summary.ineligible_current_balance)))
    rows.sort(key=lambda r: (-r.balance, -r.loan_count, r.reason_code))
    out.rows = rows

    count_sum = sum(r.loan_count for r in rows)
    balance_sum = _round(sum(r.balance for r in rows))
    # A penny per row, and only that: each row's balance is rounded once to
    # the penny before the rows are summed. Anything larger is a real gap.
    tolerance = round(max(0.02, 0.01 * (len(rows) + 1)), 2)
    missing_reason = int(blank.sum())
    out.invariants = [
        {"invariant": "reason_counts_sum_to_ineligible_count",
         "statement": "Σ reason loan_count = ineligible loan count",
         "expected": summary.ineligible_loan_count, "actual": count_sum,
         "holds": bool(count_sum == summary.ineligible_loan_count)},
        {"invariant": "reason_balances_sum_to_ineligible_balance",
         "statement": "Σ reason balance = ineligible current balance",
         "expected": summary.ineligible_current_balance, "actual": balance_sum,
         "tolerance": tolerance,
         "holds": bool(abs(balance_sum - summary.ineligible_current_balance)
                       <= tolerance)},
        {"invariant": "every_ineligible_loan_carries_one_reason",
         "statement": "no INELIGIBLE loan has a blank reason code",
         "expected": 0, "actual": missing_reason,
         "holds": bool(missing_reason == 0)},
        {"invariant": "eligibility_population_reconciles",
         "statement": ("the calculator's eligibility partition reconciles "
                       "(see the borrowing-base envelope invariants)"),
         "expected": True, "actual": bool(summary.reconciles),
         "holds": bool(summary.reconciles)},
    ]
    if not out.reconciles:
        failed = [i["invariant"] for i in out.invariants if not i["holds"]]
        out.reason = ("The ineligibility reason population does not reconcile "
                      f"({', '.join(failed)}). The breakdown is refused rather "
                      "than presented as governed.")
        return out
    out.available = True
    return out


# =========================================================================== #
# 2. Period validity and change
# =========================================================================== #
DRAWN_BASIS_VALID = "operator_supplied_as_of_snapshot_date"
DRAWN_BASIS_NOT_SUPPLIED = "current_drawn_amount_not_supplied"
DRAWN_BASIS_AS_OF_MISMATCH = "current_drawn_amount_as_of_not_this_snapshot"

MISSING_PERIOD_DRAWN = "period_valid_drawn_amount"
MISSING_CONFIGURATION_APPLICABILITY = "governed_configuration_applicability"

#: Measures whose value for a period depends on a period-valid drawing.
DRAWN_DEPENDENT_MEASURES = (
    "facility_drawn", "borrowing_base_headroom", "borrowing_base_deficiency",
    "borrowing_base_utilisation", "facility_utilisation",
)


def _date(value: Any) -> str:
    return str(value or "").strip()[:10]


def configuration_applicable_from(facility: Optional[FacilityConfiguration]
                                  ) -> Optional[str]:
    """The governed date from which the configuration demonstrably applies.

    Read from ``governance.approved_at`` — the operator approval recorded in
    the facility register. ``effective_date`` is deliberately NOT consulted:
    it is optional operator-entered metadata that no engine module reads.
    ``None`` means no historical applicability is demonstrable.
    """
    governance = getattr(facility, "governance", None) or {}
    approved = _date(governance.get("approved_at"))
    return approved or None


def configuration_applies_at(facility: Optional[FacilityConfiguration],
                             reporting_date: Optional[str], *,
                             is_current: bool) -> bool:
    """Is the configuration demonstrably applicable to this period?

    The current position always: it is what the dashboard presents the
    configuration against. A historical period only from the governed
    approval date onward; with no approval date, never.
    """
    if is_current:
        return True
    applicable_from = configuration_applicable_from(facility)
    if not applicable_from or not reporting_date:
        return False
    return _date(reporting_date) >= applicable_from


def drawn_valid_at(facility: Optional[FacilityConfiguration],
                   reporting_date: Optional[str]) -> str:
    """The drawing's validity for one snapshot, as a DRAWN_BASIS constant."""
    if facility is None or not facility.drawn_available:
        return DRAWN_BASIS_NOT_SUPPLIED
    as_of = _date(facility.current_drawn_amount_as_of)
    if as_of and reporting_date and as_of == _date(reporting_date):
        return DRAWN_BASIS_VALID
    return DRAWN_BASIS_AS_OF_MISMATCH


def facility_for_period(facility: FacilityConfiguration,
                        reporting_date: Optional[str]) -> FacilityConfiguration:
    """The approved terms with the drawing withheld unless valid for the date.

    Handed to the calculator per period, so the calculator — the single owner
    of NOT_CALCULABLE — reports the drawn-dependent measures as not calculable
    with ``current_drawn_amount`` named, rather than this layer overwriting
    figures it produced.
    """
    if drawn_valid_at(facility, reporting_date) == DRAWN_BASIS_VALID:
        return facility
    return dataclasses.replace(facility, current_drawn_amount=None,
                               current_drawn_amount_as_of="")


@dataclass
class PeriodPosition:
    """One governed period's borrowing-base envelope and its validity."""

    run_id: str
    reporting_date: Optional[str]
    envelope: Mapping[str, Any]
    is_current: bool = False
    configuration_applicable: bool = True
    drawn_basis: str = DRAWN_BASIS_NOT_SUPPLIED
    notes: List[str] = field(default_factory=list)

    @property
    def period_label(self) -> str:
        return str(self.reporting_date)[:7] if self.reporting_date else self.run_id

    @property
    def available(self) -> bool:
        return bool(self.envelope.get("available")) and self.configuration_applicable

    @property
    def reconciles(self) -> bool:
        return self.envelope.get("reconciles") is not False

    @property
    def drawn_valid(self) -> bool:
        return self.drawn_basis == DRAWN_BASIS_VALID

    def measure(self, measure_id: str) -> Any:
        """The governed value for this period, or NOT_CALCULABLE."""
        if not self.available:
            return NOT_CALCULABLE
        if measure_id in DRAWN_DEPENDENT_MEASURES and not self.drawn_valid:
            return NOT_CALCULABLE
        value = (self.envelope.get("measures") or {}).get(measure_id,
                                                          NOT_CALCULABLE)
        return value if _is_number(value) else NOT_CALCULABLE

    def why_not_calculable(self, measure_id: str) -> Optional[str]:
        """The governed reason a measure is NOT_CALCULABLE here, or None."""
        if _is_number(self.measure(measure_id)):
            return None
        if not self.configuration_applicable:
            return ("the facility configuration carries no governed approval "
                    "date demonstrating it applied at "
                    f"{self.reporting_date or self.run_id} (missing input: "
                    f"{MISSING_CONFIGURATION_APPLICABILITY})")
        if not self.envelope.get("available"):
            return str(self.envelope.get("reason")
                       or "the borrowing base is unavailable for this period")
        if measure_id in DRAWN_DEPENDENT_MEASURES:
            if self.drawn_basis == DRAWN_BASIS_NOT_SUPPLIED:
                return ("the current drawn amount under the facility has not "
                        "been supplied (missing input: current_drawn_amount)")
            return ("the operator-supplied drawing is not stated as at "
                    f"{self.reporting_date or self.run_id} (missing input: "
                    f"{MISSING_PERIOD_DRAWN})")
        missing = list(self.envelope.get("missingInputs") or [])
        return ("missing input: " + ", ".join(missing)) if missing else (
            "the measure is not calculable for this period")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runId": self.run_id,
            "reportingDate": self.reporting_date,
            "period": self.period_label,
            "isCurrent": self.is_current,
            "configurationApplicable": self.configuration_applicable,
            "drawnBasis": self.drawn_basis,
            "available": self.available,
            "reconciles": self.reconciles,
            "notes": list(self.notes),
        }


def position_for(run_id: str, reporting_date: Optional[str],
                 envelope: Mapping[str, Any], *,
                 facility: Optional[FacilityConfiguration],
                 is_current: bool) -> PeriodPosition:
    """Wrap one period's envelope with its governed validity."""
    applicable = configuration_applies_at(facility, reporting_date,
                                          is_current=is_current)
    basis = drawn_valid_at(facility, reporting_date)
    notes: List[str] = []
    if not applicable:
        notes.append(
            "The facility configuration carries no governed approval date "
            f"demonstrating it applied at {reporting_date or run_id}; no "
            "historical borrowing-base measure is calculable for that period.")
    if basis == DRAWN_BASIS_VALID:
        notes.append(f"Drawings are the operator-supplied amount stated as at "
                     f"{facility.current_drawn_amount_as_of}, the snapshot date.")
    elif basis == DRAWN_BASIS_AS_OF_MISMATCH:
        notes.append(
            "The operator-supplied drawing is stated as at "
            f"{facility.current_drawn_amount_as_of or 'an unstated date'}, not "
            f"{reporting_date or run_id}; drawn, headroom, deficiency and "
            "utilisation are not calculable for this period.")
    return PeriodPosition(run_id=str(run_id), reporting_date=reporting_date,
                          envelope=envelope, is_current=is_current,
                          configuration_applicable=applicable,
                          drawn_basis=basis, notes=notes)


@dataclass
class MeasureChange:
    """One measure, two periods, the difference — or why there is none."""

    measure_id: str
    opening_period: str
    closing_period: str
    opening: Any
    closing: Any
    change: Any = NOT_CALCULABLE
    change_pct: Any = NOT_CALCULABLE
    reason: Optional[str] = None

    @property
    def calculable(self) -> bool:
        return _is_number(self.change)

    def to_dict(self) -> Dict[str, Any]:
        return {"measureId": self.measure_id,
                "openingPeriod": self.opening_period,
                "closingPeriod": self.closing_period,
                "opening": self.opening, "closing": self.closing,
                "change": self.change, "changePct": self.change_pct,
                "calculable": self.calculable, "reason": self.reason}


def change(opening: PeriodPosition, closing: PeriodPosition, measure_id: str,
           *, unit: str = "currency") -> MeasureChange:
    """closing − opening for one governed measure. Composition only."""
    o, c = opening.measure(measure_id), closing.measure(measure_id)
    out = MeasureChange(measure_id=measure_id,
                        opening_period=opening.period_label,
                        closing_period=closing.period_label,
                        opening=o, closing=c)
    if not _is_number(o):
        out.reason = (f"opening {opening.period_label}: "
                      f"{opening.why_not_calculable(measure_id)}")
        return out
    if not _is_number(c):
        out.reason = (f"closing {closing.period_label}: "
                      f"{closing.why_not_calculable(measure_id)}")
        return out
    dp = _PCT_DP if unit == "percent" else (0 if unit == "count" else _MONEY_DP)
    delta = round(float(c) - float(o), dp)
    out.change = int(delta) if unit == "count" else delta
    if unit != "percent" and o:
        out.change_pct = round((float(c) - float(o)) / abs(float(o)) * 100.0,
                               _PCT_DP)
    return out


def series(positions: Sequence[PeriodPosition], measure_id: str
           ) -> List[Dict[str, Any]]:
    """The measure across every period, oldest → newest, gaps disclosed."""
    out: List[Dict[str, Any]] = []
    for p in positions:
        value = p.measure(measure_id)
        out.append({"period": p.period_label, "runId": p.run_id,
                    "reportingDate": p.reporting_date,
                    "value": value if _is_number(value) else None,
                    "status": "ok" if _is_number(value) else NOT_CALCULABLE,
                    "reason": p.why_not_calculable(measure_id)})
    return out


# =========================================================================== #
# 3. The bridge
# =========================================================================== #
#: Two gross roundings (≤ ½p each) plus three effect roundings (≤ ½p each),
#: rounded up to a whole number of pence.
RECONCILIATION_TOLERANCE = 0.05
#: Already-rounded envelope figures composed once.
HEADROOM_TOLERANCE = 0.02

#: The order the drivers are composed and presented in. Fixed.
DRIVER_ORDER = ("eligible_collateral_effect", "advance_rate_effect",
                "facility_cap_effect")
DRIVER_LABELS = {
    "eligible_collateral_effect": "Eligible collateral effect",
    "advance_rate_effect": "Advance rate effect",
    "facility_cap_effect": "Facility cap effect",
}

#: Envelope keys each side of the bridge must carry as NUMBERS.
_REQUIRED = {
    "eligibleCurrentBalance": "eligible collateral",
    "advanceRate": "advance rate",
    "grossBorrowingBase": "gross borrowing base",
    "availableBorrowingBase": "borrowing base",
}


@dataclass
class BridgeSide:
    """The opening or closing snapshot, as read off its envelope."""

    label: str
    run_id: Optional[str]
    reporting_date: Optional[str]
    borrowing_base: float
    eligible_balance: float
    advance_rate: float
    gross_borrowing_base: float

    @property
    def cap_effect(self) -> float:
        return _round(self.borrowing_base - self.gross_borrowing_base)

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "runId": self.run_id,
                "reportingDate": self.reporting_date,
                "borrowingBase": self.borrowing_base,
                "eligibleBalance": self.eligible_balance,
                "advanceRate": self.advance_rate,
                "advanceRatePct": round(self.advance_rate * 100.0, 6),
                "grossBorrowingBase": self.gross_borrowing_base,
                "facilityCapEffect": self.cap_effect}


@dataclass
class BridgeDriver:
    key: str
    label: str
    value: float

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label, "value": self.value}


@dataclass
class BorrowingBaseBridge:
    """Opening → three drivers → closing, with the proof it reconciles."""

    available: bool
    reason: str = ""
    missing_inputs: List[str] = field(default_factory=list)
    opening: Optional[BridgeSide] = None
    closing: Optional[BridgeSide] = None
    drivers: List[BridgeDriver] = field(default_factory=list)
    net_change: Any = NOT_CALCULABLE
    reconciliation: Dict[str, Any] = field(default_factory=dict)
    headroom: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def waterfall_rows(self) -> List[Dict[str, Any]]:
        """The Artifact Workspace waterfall shape: total, deltas, total."""
        if not self.available or self.opening is None or self.closing is None:
            return []
        rows: List[Dict[str, Any]] = [{
            "label": f"Opening borrowing base ({self.opening.label})",
            "value": self.opening.borrowing_base, "type": "total"}]
        for d in self.drivers:
            rows.append({"label": d.label, "value": d.value, "type": "delta"})
        rows.append({
            "label": f"Closing borrowing base ({self.closing.label})",
            "value": self.closing.borrowing_base, "type": "total"})
        return rows

    def to_dict(self) -> Dict[str, Any]:
        return {"available": self.available, "reason": self.reason or None,
                "missingInputs": list(self.missing_inputs),
                "opening": self.opening.to_dict() if self.opening else None,
                "closing": self.closing.to_dict() if self.closing else None,
                "drivers": [d.to_dict() for d in self.drivers],
                "driverOrder": list(DRIVER_ORDER),
                "netChange": self.net_change,
                "reconciliation": dict(self.reconciliation),
                "headroom": dict(self.headroom),
                "notes": list(self.notes)}


def _side(envelope: Mapping[str, Any], label: str, run_id: Optional[str],
          reporting_date: Optional[str], missing: List[str]
          ) -> Optional[BridgeSide]:
    if not envelope.get("available"):
        missing.append(f"{label}: {envelope.get('reason') or 'unavailable'}")
        return None
    if envelope.get("reconciles") is False:
        missing.append(f"{label}: the eligibility population does not reconcile")
        return None
    values: Dict[str, float] = {}
    for key, name in _REQUIRED.items():
        value = envelope.get(key)
        if not _is_number(value):
            missing.append(f"{label}: {name} ({key}) is {NOT_CALCULABLE}")
            continue
        values[key] = float(value)
    if len(values) != len(_REQUIRED):
        return None
    return BridgeSide(label=label, run_id=run_id, reporting_date=reporting_date,
                      borrowing_base=values["availableBorrowingBase"],
                      eligible_balance=values["eligibleCurrentBalance"],
                      advance_rate=values["advanceRate"],
                      gross_borrowing_base=values["grossBorrowingBase"])


def bridge(opening_envelope: Mapping[str, Any],
           closing_envelope: Mapping[str, Any], *,
           opening_label: str, closing_label: str,
           opening_run_id: Optional[str] = None,
           closing_run_id: Optional[str] = None,
           opening_reporting_date: Optional[str] = None,
           closing_reporting_date: Optional[str] = None,
           opening_drawn_valid: bool = False,
           closing_drawn_valid: bool = False) -> BorrowingBaseBridge:
    """Compose the bridge from two governed envelopes, or refuse by name."""
    missing: List[str] = []
    opening = _side(opening_envelope, opening_label, opening_run_id,
                    opening_reporting_date, missing)
    closing = _side(closing_envelope, closing_label, closing_run_id,
                    closing_reporting_date, missing)
    if opening is None or closing is None:
        return BorrowingBaseBridge(
            available=False, missing_inputs=missing,
            reason=("The borrowing-base bridge cannot be built because a "
                    "required input is missing or not calculable: "
                    + "; ".join(missing) + "."))

    e_o, e_c = opening.eligible_balance, closing.eligible_balance
    a_o, a_c = opening.advance_rate, closing.advance_rate
    collateral = _round((e_c - e_o) * a_o)
    rate = _round(e_c * (a_c - a_o))
    cap = _round(closing.cap_effect - opening.cap_effect)
    drivers = [BridgeDriver(k, DRIVER_LABELS[k], v) for k, v in zip(
        DRIVER_ORDER, (collateral, rate, cap))]
    composed = _round(opening.borrowing_base + collateral + rate + cap)
    difference = _round(composed - closing.borrowing_base)
    holds = abs(difference) <= RECONCILIATION_TOLERANCE
    out = BorrowingBaseBridge(
        available=bool(holds), opening=opening, closing=closing,
        drivers=drivers,
        net_change=_round(closing.borrowing_base - opening.borrowing_base),
        reconciliation={
            "statement": ("opening borrowing base + eligible collateral effect "
                          "+ advance rate effect + facility cap effect = "
                          "closing borrowing base"),
            "expected": closing.borrowing_base, "actual": composed,
            "difference": difference, "tolerance": RECONCILIATION_TOLERANCE,
            "holds": bool(holds)})
    if not holds:
        out.reason = ("The borrowing-base bridge does not reconcile to the "
                      f"closing borrowing base (difference {difference:+.2f} "
                      f"against a {RECONCILIATION_TOLERANCE:.2f} tolerance). "
                      "It is refused rather than presented.")
        return out
    out.notes.append(
        "Drivers in deterministic order: eligible collateral effect priced at "
        "the opening advance rate, then the advance-rate effect on closing "
        "collateral, then the change in the facility-cap effect. No haircut, "
        "reserve, concentration or overcollateralisation effect exists in the "
        "v1 calculator, so none appears here.")
    out.headroom = _headroom_bridge(opening_envelope, closing_envelope, out,
                                    opening_drawn_valid, closing_drawn_valid)
    return out


def _headroom_bridge(opening_envelope: Mapping[str, Any],
                     closing_envelope: Mapping[str, Any],
                     base: BorrowingBaseBridge,
                     opening_drawn_valid: bool, closing_drawn_valid: bool
                     ) -> Dict[str, Any]:
    """opening headroom + Δ borrowing base − Δ drawn = closing headroom."""
    missing: List[str] = []
    for label, valid in ((base.opening.label, opening_drawn_valid),
                         (base.closing.label, closing_drawn_valid)):
        if not valid:
            missing.append(f"{label}: no period-valid facility drawing "
                           f"(missing input: {MISSING_PERIOD_DRAWN})")
    values: Dict[str, float] = {}
    if not missing:
        for side, env in (("opening", opening_envelope),
                          ("closing", closing_envelope)):
            for key in ("currentDrawnAmount", "borrowingBaseHeadroom"):
                value = env.get(key)
                if _is_number(value):
                    values[f"{side}.{key}"] = float(value)
                else:
                    missing.append(f"{side}: {key} is {NOT_CALCULABLE}")
    if missing:
        return {"available": False, "status": NOT_CALCULABLE,
                "missingInputs": missing,
                "reason": ("The headroom bridge is not calculable: "
                           + "; ".join(missing) + ".")}
    d_o, d_c = values["opening.currentDrawnAmount"], values["closing.currentDrawnAmount"]
    h_o, h_c = values["opening.borrowingBaseHeadroom"], values["closing.borrowingBaseHeadroom"]
    delta_drawn = _round(d_c - d_o)
    composed = _round(h_o + base.net_change - delta_drawn)
    difference = _round(composed - h_c)
    holds = abs(difference) <= HEADROOM_TOLERANCE
    return {"available": bool(holds), "status": "ok" if holds else NOT_CALCULABLE,
            "openingHeadroom": h_o, "changeInBorrowingBase": base.net_change,
            "changeInDrawn": delta_drawn, "closingHeadroom": h_c,
            "reconciliation": {
                "statement": ("opening headroom + Δ borrowing base − Δ drawn "
                              "= closing headroom"),
                "expected": h_c, "actual": composed, "difference": difference,
                "tolerance": HEADROOM_TOLERANCE, "holds": bool(holds)},
            "reason": (None if holds else
                       "The headroom bridge does not reconcile; it is refused.")}


__all__ = [
    # reasons
    "summarise_ineligibility_reasons", "IneligibilityReasonSummary",
    "ReasonRow", "rule_descriptions", "REASON_MISSING",
    # periods
    "PeriodPosition", "MeasureChange", "position_for", "change", "series",
    "configuration_applicable_from", "configuration_applies_at",
    "drawn_valid_at", "facility_for_period", "DRAWN_BASIS_VALID",
    "DRAWN_BASIS_NOT_SUPPLIED", "DRAWN_BASIS_AS_OF_MISMATCH",
    "DRAWN_DEPENDENT_MEASURES", "MISSING_PERIOD_DRAWN",
    "MISSING_CONFIGURATION_APPLICABILITY",
    # bridge
    "bridge", "BorrowingBaseBridge", "BridgeSide", "BridgeDriver",
    "DRIVER_ORDER", "DRIVER_LABELS", "RECONCILIATION_TOLERANCE",
    "HEADROOM_TOLERANCE",
]
