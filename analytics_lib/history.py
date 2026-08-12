"""analytics_lib.history — observed behaviour across snapshots.

Shared, like everything else in this package: React, MI Query and the agent
tools all read these functions. Nothing here is securitisation-specific and
nothing here belongs to an agent.

What is genuinely new, and why
------------------------------
Most of what a historical review needs already exists and is reused rather than
rebuilt — ``mi_agent.risk_monitor.migration.migration_matrix`` for transitions,
``analytics_lib.cohort`` for vintages, ``mi_agent.period_change`` for two-period
movement. Three things did not exist:

* **a multi-period series.** ``period_change`` is strictly pairwise; a
  twelve-month trend is a different query shape, not a loop over pairs.
* **an observed prepayment / redemption rate.** The canonical model already
  separates scheduled from unscheduled principal; nothing computed a rate from
  it.
* **loss and recovery rates.** Same position: the fields exist, the arithmetic
  did not.

Observed, never projected
-------------------------
Every function here reports what the snapshots actually show. There is no
forecast, no scenario, no assumption and no extrapolation — a rate over a window
that is shorter than the window asked for is reported as being over the window
that exists, rather than scaled up to fill it.

The trap this module is written around
--------------------------------------
**Not all balance reduction is prepayment.** A book that amortises on schedule,
redeems at maturity, and writes off a defaulted loan has three different causes
of a falling balance, and only one of them is prepayment. Inferring a rate from
the change in total balance would be wrong in a way that looks entirely
plausible — and would flatter or damn a portfolio depending on which cause
happened to dominate. So the numerator here is an explicit field
(``unscheduled_principal_collections``), never a residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

BALANCE_FIELD = "current_principal_balance"
LOAN_ID_FIELD = "loan_identifier"

#: The canonical field carrying UNSCHEDULED principal, which is what a
#: prepayment rate is a rate of. Declared here rather than inferred.
UNSCHEDULED_PRINCIPAL_FIELD = "unscheduled_principal_collections"
REDEMPTIONS_FIELD = "redemptions_received_in_period"
SCHEDULED_DUE_FIELD = "total_scheduled_principal_interest_due"

LOSSES_FIELD = "allocated_losses"
RECOVERIES_FIELD = "cumulative_recoveries"
RECOVERIES_IN_PERIOD_FIELD = "recoveries_in_period"
ORIGINAL_BALANCE_FIELD = "original_principal_balance"

#: Methodology identifiers, versioned because a change to any of them changes
#: reported rates and an explanation that cited one must keep meaning it.
SMM_METHOD = "OBSERVED_SMM@v1"
CPR_METHOD = "OBSERVED_CPR@v1"
LOSS_METHOD = "OBSERVED_LOSS@v1"
RECOVERY_METHOD = "OBSERVED_RECOVERY@v1"
SERIES_METHOD = "OBSERVED_SERIES@v1"


# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #
def portfolio_series(
    snapshots: Mapping[str, pd.DataFrame],
    measures: Mapping[str, Callable[[pd.DataFrame], Optional[float]]],
    *,
    periods: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Evaluate named measures across ordered snapshots.

    One pass per snapshot for ALL measures, not one pass per measure — a
    twelve-period, eight-measure request costs twelve frame walks rather than
    ninety-six. That is the difference between a trend being cheap enough to ask
    for routinely and being something a reviewer avoids.

    Measures are callables so this stays free of any particular metric library;
    the caller supplies whatever deterministic function it already uses, which is
    how React and the agent end up on the same arithmetic.
    """
    order = list(periods) if periods is not None else sorted(snapshots)
    series: Dict[str, List[Optional[float]]] = {name: [] for name in measures}
    loan_counts: List[int] = []
    balances: List[float] = []

    for period in order:
        frame = snapshots.get(period)
        if frame is None:
            for name in measures:
                series[name].append(None)
            loan_counts.append(0)
            balances.append(0.0)
            continue
        loan_counts.append(int(len(frame)))
        balances.append(float(pd.to_numeric(frame[BALANCE_FIELD], errors="coerce")
                              .sum()) if BALANCE_FIELD in frame.columns else 0.0)
        for name, measure in measures.items():
            try:
                value = measure(frame)
            except Exception:  # noqa: BLE001 - one bad measure must not lose the series
                value = None
            series[name].append(None if value is None else float(value))

    return {
        "method": SERIES_METHOD,
        "periods": order,
        "period_count": len(order),
        "loan_count": loan_counts,
        "total_balance": balances,
        "series": {name: {"values": values,
                          "first": _first(values), "last": _last(values),
                          "change": _change(values),
                          "direction": _direction(values)}
                   for name, values in series.items()},
    }


def _exit_balance(opening: pd.DataFrame,
                  closing: pd.DataFrame) -> Tuple[float, int]:
    """Balance of loans present at the open and absent at the close.

    Returns ``(balance, loan_count)``. Needs a stable identifier in both frames;
    without one it returns zero rather than guessing, because a rate built on a
    guessed population is worse than no rate.
    """
    if any(LOAN_ID_FIELD not in f.columns for f in (opening, closing)):
        return 0.0, 0
    opening_ids = opening[LOAN_ID_FIELD].astype(str)
    closing_ids = set(closing[LOAN_ID_FIELD].astype(str))
    gone = ~opening_ids.isin(closing_ids)
    if not gone.any():
        return 0.0, 0
    balances = pd.to_numeric(opening.loc[gone, BALANCE_FIELD],
                             errors="coerce").fillna(0.0)
    return float(balances.sum()), int(gone.sum())


def _first(values: Sequence[Optional[float]]) -> Optional[float]:
    return next((v for v in values if v is not None), None)


def _last(values: Sequence[Optional[float]]) -> Optional[float]:
    return next((v for v in reversed(values) if v is not None), None)


def _change(values: Sequence[Optional[float]]) -> Optional[float]:
    first, last = _first(values), _last(values)
    if first is None or last is None:
        return None
    return round(last - first, 6)


def _direction(values: Sequence[Optional[float]]) -> str:
    """``rising`` / ``falling`` / ``flat`` / ``unknown``.

    Deliberately describes the endpoints only. Characterising the *shape* of a
    series — accelerating, reverting, seasonal — is a judgement, and encoding one
    here would put an opinion in a measurement.
    """
    change = _change(values)
    if change is None:
        return "unknown"
    if abs(change) < 1e-9:
        return "flat"
    return "rising" if change > 0 else "falling"


# --------------------------------------------------------------------------- #
# Prepayment and redemption
# --------------------------------------------------------------------------- #
@dataclass
class PrepaymentResult:
    """An observed prepayment rate, with everything needed to reproduce it."""

    method: str
    smm_pct: Optional[float] = None
    cpr_pct: Optional[float] = None
    periods_observed: int = 0
    unscheduled_principal: float = 0.0
    redemptions: float = 0.0
    opening_balance_total: float = 0.0
    per_period: List[Dict[str, Any]] = field(default_factory=list)
    available: bool = True
    reason: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "available": self.available,
            "reason": self.reason or None,
            "smm_pct": self.smm_pct,
            "cpr_pct": self.cpr_pct,
            "periods_observed": self.periods_observed,
            "unscheduled_principal": self.unscheduled_principal,
            "redemptions": self.redemptions,
            "opening_balance_total": self.opening_balance_total,
            "per_period": self.per_period,
            "notes": self.notes,
        }


def prepayment_rate(
    snapshots: Mapping[str, pd.DataFrame],
    *,
    periods: Optional[Sequence[str]] = None,
    redemptions_by_period: Optional[Mapping[str, float]] = None,
    include_redemptions: bool = True,
    annualise: bool = True,
) -> PrepaymentResult:
    """Observed single-monthly mortality, and its annualised form.

    **Numerator** — unscheduled principal in the period:
    ``unscheduled_principal_collections`` from the closing snapshot, plus full
    redemptions when ``include_redemptions``. Redemptions must be supplied
    separately because *a redeemed loan is absent from the closing tape*: the
    exit cannot be read from the frame that no longer contains it.

    **Denominator** — the OPENING balance of the period, from the prior snapshot.

    **Excluded**, deliberately and by name: scheduled amortisation, contractual
    maturity, and default-related balance reduction. None of those is
    prepayment, and a rate that swept them in would move for reasons that have
    nothing to do with borrowers repaying early.

    **Annualisation** — ``CPR = 1 - (1 - SMM)^12``, applied to the mean monthly
    SMM over the observed window. Reported alongside the SMM rather than
    instead of it, because the annualised figure is a projection of the observed
    one onto a year and the reader should be able to see both.

    Returns ``available: False`` with a reason when the tape carries no
    unscheduled-principal field. It does **not** fall back to inferring a rate
    from the change in total balance: that would be wrong in a way that looks
    entirely plausible.
    """
    order = list(periods) if periods is not None else sorted(snapshots)
    if len(order) < 2:
        return PrepaymentResult(
            method=SMM_METHOD, available=False, periods_observed=0,
            reason="a prepayment rate needs at least two consecutive snapshots; "
                   f"{len(order)} was supplied.")

    probe = snapshots.get(order[-1])
    if probe is None or UNSCHEDULED_PRINCIPAL_FIELD not in probe.columns:
        return PrepaymentResult(
            method=SMM_METHOD, available=False,
            reason=(f"the tape carries no {UNSCHEDULED_PRINCIPAL_FIELD!r}, so "
                    "unscheduled principal cannot be identified. A rate is NOT "
                    "inferred from the change in total balance: scheduled "
                    "amortisation, maturity and default-related reduction would "
                    "all be counted as prepayment."))

    per_period: List[Dict[str, Any]] = []
    smm_values: List[float] = []
    total_unscheduled = 0.0
    total_redemptions = 0.0
    total_opening = 0.0

    for index in range(1, len(order)):
        opening_period, closing_period = order[index - 1], order[index]
        opening = snapshots.get(opening_period)
        closing = snapshots.get(closing_period)
        if opening is None or closing is None:
            continue

        opening_balance = float(pd.to_numeric(
            opening[BALANCE_FIELD], errors="coerce").sum())
        unscheduled = float(pd.to_numeric(
            closing[UNSCHEDULED_PRINCIPAL_FIELD], errors="coerce").fillna(0).sum())

        redemptions = 0.0
        redemption_source = "not counted"
        if include_redemptions:
            if redemptions_by_period is not None:
                redemptions = float(redemptions_by_period.get(closing_period, 0.0))
                redemption_source = "supplied series"
            elif REDEMPTIONS_FIELD in closing.columns and float(pd.to_numeric(
                    closing[REDEMPTIONS_FIELD], errors="coerce").fillna(0).sum()):
                redemptions = float(pd.to_numeric(
                    closing[REDEMPTIONS_FIELD], errors="coerce").fillna(0).sum())
                redemption_source = f"{REDEMPTIONS_FIELD} field"
            else:
                # A redeemed loan is ABSENT from the closing tape, so the field
                # on that tape cannot carry its exit — the row is gone. Exits are
                # therefore derived by set difference against the opening
                # snapshot, valued at the opening balance.
                #
                # This is what makes the rate work on a real servicer tape, where
                # a per-period redemption column is often simply not populated.
                # It counts every exit, so a loan leaving through maturity or
                # repossession is included too; `exits_are_redemptions` records
                # that caveat rather than hiding it.
                redemptions, exited = _exit_balance(opening, closing)
                redemption_source = ("derived from loans absent in the closing "
                                     f"snapshot ({exited} loan(s))")

        exits_derived = redemption_source.startswith("derived")

        numerator = unscheduled + redemptions
        smm = (numerator / opening_balance * 100.0) if opening_balance else None
        if smm is not None:
            smm_values.append(smm)

        total_unscheduled += unscheduled
        total_redemptions += redemptions
        total_opening += opening_balance
        per_period.append({
            "period": closing_period,
            "opening_period": opening_period,
            "opening_balance": round(opening_balance, 2),
            "unscheduled_principal": round(unscheduled, 2),
            "redemptions": round(redemptions, 2),
            "redemption_source": redemption_source,
            "smm_pct": None if smm is None else round(smm, 6),
        })

    if not smm_values:
        return PrepaymentResult(
            method=SMM_METHOD, available=False,
            reason="no period had a non-zero opening balance to divide by.")

    mean_smm = sum(smm_values) / len(smm_values)
    cpr = None
    if annualise:
        cpr = (1.0 - (1.0 - mean_smm / 100.0) ** 12) * 100.0

    notes = [
        "Numerator is unscheduled principal"
        + (" plus full redemptions." if include_redemptions else ", excluding redemptions."),
        "Denominator is the OPENING balance of each period.",
        "Scheduled amortisation, contractual maturity and default-related "
        "balance reduction are excluded — none of them is prepayment.",
    ]
    if include_redemptions and any(p["redemption_source"].startswith("derived")
                                   for p in per_period):
        notes.append(
            "Full redemptions were derived from loans present in the opening "
            "snapshot and absent from the closing one, valued at their opening "
            "balance — a redeemed loan is gone from the closing tape, so its "
            "exit cannot be read there. This counts EVERY exit, so a loan "
            "leaving through contractual maturity or repossession is included "
            "alongside genuine early redemption.")
    if annualise:
        notes.append("CPR = 1 - (1 - mean SMM)^12, an annualisation of the "
                     "observed monthly rate — not a forecast.")

    return PrepaymentResult(
        method=CPR_METHOD if annualise else SMM_METHOD,
        smm_pct=round(mean_smm, 6),
        cpr_pct=None if cpr is None else round(cpr, 6),
        periods_observed=len(smm_values),
        unscheduled_principal=round(total_unscheduled, 2),
        redemptions=round(total_redemptions, 2),
        opening_balance_total=round(total_opening, 2),
        per_period=per_period, notes=notes)


# --------------------------------------------------------------------------- #
# Loss and recovery
# --------------------------------------------------------------------------- #
def loss_and_recovery(
    snapshots: Mapping[str, pd.DataFrame],
    *,
    periods: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Observed cumulative losses and recoveries, on EVERY defensible denominator.

    Several legitimate definitions exist and they answer different questions, so
    all of them are reported and none is silently chosen:

        cumulative loss / original balance   lifetime experience of the pool
        cumulative loss / opening balance    loss relative to where the window began
        cumulative loss / current balance    loss relative to what remains
        recoveries / losses                  how much of the loss came back

    Picking one and calling it "the loss rate" is how a portfolio gets described
    as three times better or worse than it is, depending on which denominator
    happened to be chosen.

    ``allocated_losses`` and ``cumulative_recoveries`` are *cumulative* fields in
    the canonical model, so the closing snapshot carries the running total. A
    period figure is the difference between two snapshots, not a sum across them
    — summing cumulative fields would count the same loss in every period it
    remained on the tape.
    """
    order = list(periods) if periods is not None else sorted(snapshots)
    if not order:
        return {"available": False, "reason": "no snapshots supplied."}

    closing = snapshots.get(order[-1])
    opening = snapshots.get(order[0])
    if closing is None:
        return {"available": False, "reason": "the closing snapshot is missing."}

    missing = [f for f in (LOSSES_FIELD,) if f not in closing.columns]
    if missing:
        return {"available": False,
                "reason": (f"the tape carries no {', '.join(missing)}, so "
                           "realised losses cannot be measured. No loss figure "
                           "is inferred from balance movement.")}

    def _sum(frame: Optional[pd.DataFrame], column: str) -> float:
        if frame is None or column not in frame.columns:
            return 0.0
        return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())

    cumulative_losses = _sum(closing, LOSSES_FIELD)
    cumulative_recoveries = _sum(closing, RECOVERIES_FIELD)
    opening_losses = _sum(opening, LOSSES_FIELD)
    opening_recoveries = _sum(opening, RECOVERIES_FIELD)

    original_balance = _sum(closing, ORIGINAL_BALANCE_FIELD)
    opening_balance = _sum(opening, BALANCE_FIELD)
    current_balance = _sum(closing, BALANCE_FIELD)

    def _rate(numerator: float, denominator: float) -> Optional[float]:
        return round(numerator / denominator * 100, 6) if denominator else None

    period_losses = cumulative_losses - opening_losses
    period_recoveries = cumulative_recoveries - opening_recoveries

    return {
        "available": True,
        "method": LOSS_METHOD,
        "window": {"from": order[0], "to": order[-1], "periods": len(order)},
        "cumulative_losses": round(cumulative_losses, 2),
        "cumulative_recoveries": round(cumulative_recoveries, 2),
        "period_losses": round(period_losses, 2),
        "period_recoveries": round(period_recoveries, 2),
        "denominators": {
            "original_balance": round(original_balance, 2),
            "opening_balance": round(opening_balance, 2),
            "current_balance": round(current_balance, 2),
        },
        # Every defensible rate, each named for the question it answers.
        "loss_rate_on_original_pct": _rate(cumulative_losses, original_balance),
        "loss_rate_on_opening_pct": _rate(cumulative_losses, opening_balance),
        "loss_rate_on_current_pct": _rate(cumulative_losses, current_balance),
        "period_loss_rate_on_opening_pct": _rate(period_losses, opening_balance),
        "recovery_rate_on_losses_pct": _rate(cumulative_recoveries,
                                             cumulative_losses),
        "net_loss": round(cumulative_losses - cumulative_recoveries, 2),
        "net_loss_rate_on_original_pct": _rate(
            cumulative_losses - cumulative_recoveries, original_balance),
        "notes": [
            "Several legitimate loss denominators exist and each answers a "
            "different question, so all are reported and none is chosen for you.",
            "allocated_losses and cumulative_recoveries are CUMULATIVE fields: "
            "the closing snapshot carries the running total, and a period "
            "figure is the difference between two snapshots rather than a sum "
            "across them.",
            "Losses are read from the governed field, never inferred from "
            "balance movement.",
        ],
    }


# --------------------------------------------------------------------------- #
# Cohort comparison
# --------------------------------------------------------------------------- #
def compare_cohorts(
    frame: pd.DataFrame,
    cohort_mask: pd.Series,
    measures: Mapping[str, Callable[[pd.DataFrame], Optional[float]]],
    *,
    cohort_label: str = "cohort",
    remainder_label: str = "rest of portfolio",
) -> Dict[str, Any]:
    """One cohort against the REST of the book, on the same measures.

    The comparison is the point. "The high-LTV cohort has 4% arrears" means
    nothing until the reader knows the rest of the book is at 1.2% — and an
    agent that reports the first without the second has produced a number
    rather than a finding.

    Bounded by construction: two groups, N measures, one pass each. No per-loan
    output and no request per loan.
    """
    cohort = frame[cohort_mask]
    remainder = frame[~cohort_mask]

    def _balance(subset: pd.DataFrame) -> float:
        if BALANCE_FIELD not in subset.columns:
            return 0.0
        return float(pd.to_numeric(subset[BALANCE_FIELD], errors="coerce").sum())

    total_balance = _balance(frame)
    rows: Dict[str, Dict[str, Any]] = {}
    for name, measure in measures.items():
        try:
            in_cohort = measure(cohort) if len(cohort) else None
        except Exception:  # noqa: BLE001
            in_cohort = None
        try:
            in_remainder = measure(remainder) if len(remainder) else None
        except Exception:  # noqa: BLE001
            in_remainder = None
        difference = (None if in_cohort is None or in_remainder is None
                      else round(float(in_cohort) - float(in_remainder), 6))
        rows[name] = {
            "cohort": None if in_cohort is None else round(float(in_cohort), 6),
            "remainder": (None if in_remainder is None
                          else round(float(in_remainder), 6)),
            "difference": difference,
            # A ratio makes "three times worse" readable, but only where the
            # comparator is non-zero — otherwise it is undefined, not infinite.
            "ratio": (round(float(in_cohort) / float(in_remainder), 4)
                      if in_cohort is not None and in_remainder
                      else None),
        }

    cohort_balance = _balance(cohort)
    return {
        "cohort_label": cohort_label,
        "remainder_label": remainder_label,
        "cohort": {
            "loan_count": int(len(cohort)),
            "balance": round(cohort_balance, 2),
            "balance_pct_of_portfolio": (round(cohort_balance / total_balance * 100, 6)
                                         if total_balance else None),
        },
        "remainder": {
            "loan_count": int(len(remainder)),
            "balance": round(_balance(remainder), 2),
        },
        "measures": rows,
        "note": ("A cohort figure is only interpretable against the rest of the "
                 "book. Both are reported so neither is quoted alone."),
    }
