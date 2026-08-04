"""simulation.reference_truth — the INDEPENDENT expected-truth package.

The single most important property of this module is what it does **not** do: it
never calls the production MI, risk or regime code it is used to check. Every
figure below is computed here, from the generated loan state, with small,
transparent arithmetic a reviewer can verify by hand. If this module imported
``mi_agent`` the whole framework would collapse into the production code
checking itself.

The arithmetic is deliberately naive — plain sums and ratios over Python
dataclasses, no pandas, no bucketing library, no configuration. Where the
production definition is genuinely more nuanced (weighted averages excluding
missing values, for example) the expected truth states the SIMPLE definition and
the assertion applies an explicit tolerance, rather than the simulator quietly
reimplementing the production rule.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .dates import months_between, parse_iso
from .models import (
    MOVE_ACQUIRED,
    MOVE_AMORTISATION,
    MOVE_DISPOSAL,
    MOVE_FUNDED,
    MOVE_FURTHER_ADVANCE,
    MOVE_INTEREST,
    MOVE_RECOVERY,
    MOVE_REDEMPTION,
    MOVE_REPAYMENT,
    MOVE_RESTATEMENT,
    MOVE_WRITE_OFF,
    POSITIVE_MOVEMENTS,
    CaseManifest,
    EconomicHistory,
    LoanState,
    PeriodSnapshot,
)

#: Reference-truth format version. Bumped when a figure's DEFINITION changes, so
#: a stored fixture cannot be silently compared against a different definition.
TRUTH_VERSION = 1


# --------------------------------------------------------------------------- #
# Small, auditable primitives
# --------------------------------------------------------------------------- #
def _total(loans: Iterable[LoanState]) -> float:
    return round(sum(l.closing_balance for l in loans), 2)


def _weighted_average(pairs: Iterable[Tuple[float, float]]) -> Optional[float]:
    """Balance-weighted mean of ``(value, weight)``; ``None`` with no weight."""
    num = 0.0
    den = 0.0
    for value, weight in pairs:
        if value is None or weight is None:
            continue
        num += float(value) * float(weight)
        den += float(weight)
    return round(num / den, 6) if den else None


def _group_balances(loans: Iterable[LoanState], key) -> Dict[str, float]:
    out: Dict[str, float] = defaultdict(float)
    for loan in loans:
        out[str(key(loan))] += loan.closing_balance
    return {k: round(v, 2) for k, v in sorted(out.items())}


def _largest(groups: Dict[str, float]) -> Dict[str, Any]:
    """The biggest group, its balance and its share. Empty ⇒ nulls, never zero."""
    total = round(sum(groups.values()), 2)
    if not groups or total <= 0:
        return {"group": None, "balance": None, "share_percent": None,
                "total": total}
    group = max(groups, key=lambda k: (groups[k], k))
    return {"group": group, "balance": round(groups[group], 2),
            "share_percent": round(groups[group] / total * 100.0, 4),
            "total": total}


def _current_ltv(loan: LoanState) -> Optional[float]:
    return (round(loan.closing_balance / loan.current_value * 100.0, 6)
            if loan.current_value else None)


# --------------------------------------------------------------------------- #
# Asset-specific dimensions
# --------------------------------------------------------------------------- #
def _asset_dimensions(asset_class: str) -> Dict[str, Any]:
    """``{name: key_function}`` for the dimensions this asset class is measured on."""
    if asset_class == "equity_release":
        return {
            "region": lambda l: l.region,
            "product": lambda l: l.attributes.get("erm_product_type", ""),
            "cohort": lambda l: l.cohort,
        }
    if asset_class == "bridge":
        return {
            "region": lambda l: l.region,
            "charge_type": lambda l: l.attributes.get("charge_type", ""),
            "exit_route": lambda l: l.attributes.get("exit_route", ""),
            "collateral_type": lambda l: l.attributes.get("collateral_type", ""),
        }
    return {
        "region": lambda l: l.region,
        "manufacturer": lambda l: l.attributes.get("manufacturer", ""),
        "vendor": lambda l: l.attributes.get("vendor_name", ""),
        "industry": lambda l: l.attributes.get("lessee_industry", ""),
        "collateral_type": lambda l: l.attributes.get("collateral_type", ""),
        "contract_type": lambda l: l.attributes.get("contract_type", ""),
    }


def _asset_measures(asset_class: str, loans: List[LoanState],
                    reporting_date: str) -> Dict[str, Any]:
    """One asset-specific maturity or collateral measure per family."""
    total = _total(loans)
    if asset_class == "equity_release":
        ages = [(int(l.attributes.get("age_at_origination", 0))
                 + months_between(parse_iso(l.origination_date),
                                  parse_iso(reporting_date)) // 12,
                 l.closing_balance) for l in loans]
        return {
            "weighted_average_borrower_age": _weighted_average(ages),
            "balance_over_60_percent_ltv": round(sum(
                l.closing_balance for l in loans
                if (_current_ltv(l) or 0.0) > 60.0), 2),
        }
    if asset_class == "bridge":
        # A maturity wall is everything falling due inside the horizon,
        # INCLUDING facilities already past their contractual maturity: a loan
        # that should have repaid last month is the most acute part of the wall,
        # not something to exclude from it.
        within_90 = round(sum(
            l.closing_balance for l in loans
            if (parse_iso(l.attributes.get("maturity_date", reporting_date))
                - parse_iso(reporting_date)).days <= 90), 2)
        extended = round(sum(l.closing_balance for l in loans
                             if int(l.attributes.get("extensions_granted", 0)) > 0), 2)
        return {
            "balance_maturing_within_90_days": within_90,
            "share_maturing_within_90_days": (round(within_90 / total * 100.0, 4)
                                              if total else None),
            "extended_balance": extended,
            "extended_share_percent": (round(extended / total * 100.0, 4)
                                       if total else None),
            "second_charge_balance": round(sum(
                l.closing_balance for l in loans
                if int(l.attributes.get("lien", 1)) >= 2), 2),
        }
    balloon = round(sum(float(l.attributes.get("balloon_amount", 0.0))
                        for l in loans), 2)
    residual = round(sum(float(l.attributes.get("current_residual_value", 0.0))
                         for l in loans), 2)
    return {
        "total_balloon_amount": balloon,
        "balloon_share_of_balance": (round(balloon / total * 100.0, 4)
                                     if total else None),
        "total_current_residual_value": residual,
        "balance_with_balloon": round(sum(
            l.closing_balance for l in loans
            if float(l.attributes.get("balloon_amount", 0.0)) > 0), 2),
    }


# --------------------------------------------------------------------------- #
# Period truth
# --------------------------------------------------------------------------- #
def period_truth(case: CaseManifest, snapshot: PeriodSnapshot) -> Dict[str, Any]:
    """The independently computed expected figures for one reporting date."""
    on_book = list(snapshot.loans)
    total = _total(on_book)
    movements = dict(snapshot.movement_totals)

    status_balances: Dict[str, float] = defaultdict(float)
    status_counts: Dict[str, int] = defaultdict(int)
    for loan in on_book:
        status_balances[loan.status] += loan.closing_balance
        status_counts[loan.status] += 1
    for loan in snapshot.exits:
        status_counts[loan.status] += 1

    dimensions = _asset_dimensions(case.asset_class)
    concentrations = {name: _largest(_group_balances(on_book, key))
                      for name, key in dimensions.items()}
    largest_borrower = _largest(_group_balances(on_book, lambda l: l.borrower_id))

    vintages = _group_balances(on_book, lambda l: l.origination_date[:4])
    cohorts = _group_balances(on_book, lambda l: l.cohort)

    additions = round(movements.get(MOVE_FUNDED, 0.0)
                      + movements.get(MOVE_ACQUIRED, 0.0), 2)
    reductions = round(
        movements.get(MOVE_REDEMPTION, 0.0) + movements.get(MOVE_REPAYMENT, 0.0)
        + movements.get(MOVE_AMORTISATION, 0.0), 2)

    delta = round(sum(v if k in POSITIVE_MOVEMENTS else -v
                      for k, v in movements.items()), 2)

    return {
        "reporting_date": snapshot.reporting_date,
        "period_index": snapshot.period_index,
        # -- headline ------------------------------------------------------ #
        "loan_count": len(on_book),
        "exited_loan_count": len(snapshot.exits),
        # The tape reports the book at the cut-off; exits are recovered by
        # comparing consecutive reporting dates (see history.canonical_frame_rows).
        "reported_row_count": len(on_book),
        "opening_balance": round(snapshot.opening_balance, 2),
        "closing_balance": total,
        "period_movement": round(total - snapshot.opening_balance, 2),
        # -- movements ----------------------------------------------------- #
        "funded_additions": round(movements.get(MOVE_FUNDED, 0.0), 2),
        "acquired_additions": round(movements.get(MOVE_ACQUIRED, 0.0), 2),
        "total_additions": additions,
        "rolled_up_interest": round(movements.get(MOVE_INTEREST, 0.0), 2),
        "further_advances": round(movements.get(MOVE_FURTHER_ADVANCE, 0.0), 2),
        "redemptions": round(movements.get(MOVE_REDEMPTION, 0.0), 2),
        "repayments": round(movements.get(MOVE_REPAYMENT, 0.0), 2),
        "principal_amortisation": round(movements.get(MOVE_AMORTISATION, 0.0), 2),
        "total_reductions": reductions,
        "write_offs": round(movements.get(MOVE_WRITE_OFF, 0.0), 2),
        "recoveries": round(movements.get(MOVE_RECOVERY, 0.0), 2),
        "asset_disposals": round(movements.get(MOVE_DISPOSAL, 0.0), 2),
        "restatement": round(movements.get(MOVE_RESTATEMENT, 0.0), 2),
        "movement_reconciles": snapshot.reconciles(),
        "movement_delta": delta,
        # -- weighted averages --------------------------------------------- #
        "weighted_average_current_ltv": _weighted_average(
            [(_current_ltv(l), l.closing_balance) for l in on_book]),
        "weighted_average_interest_rate": _weighted_average(
            [(l.interest_rate, l.closing_balance) for l in on_book]),
        # -- distribution --------------------------------------------------- #
        "status_balances": {k: round(v, 2) for k, v in sorted(status_balances.items())},
        "status_counts": dict(sorted(status_counts.items())),
        "largest_concentrations": concentrations,
        "largest_borrower": largest_borrower,
        "vintage_balances": vintages,
        "cohort_balances": cohorts,
        # -- asset-specific -------------------------------------------------- #
        "asset_measures": _asset_measures(case.asset_class, on_book,
                                          snapshot.reporting_date),
    }


def build_expected_truth(case: CaseManifest, history: EconomicHistory
                         ) -> Dict[str, Any]:
    """The full expected-truth package for a case, ready to persist as JSON."""
    periods = [period_truth(case, snapshot) for snapshot in history.periods]
    first, last = periods[0], periods[-1]
    return {
        "truth_version": TRUTH_VERSION,
        "case_id": case.case_id,
        "seed": case.seed,
        "simulation_version": case.simulation_version,
        "asset_class": case.asset_class,
        "asset_subtype": case.asset_subtype,
        "client_id": case.client_id,
        "portfolio_id": case.portfolio_id,
        "spv_id": case.spv_id,
        "currency": case.currency,
        "reporting_dates": [p["reporting_date"] for p in periods],
        "periods": periods,
        "history_summary": {
            "opening_balance": first["opening_balance"],
            "closing_balance": last["closing_balance"],
            "total_movement": round(last["closing_balance"]
                                    - first["opening_balance"], 2),
            "direction": _direction(last["closing_balance"],
                                    first["opening_balance"]),
            "month_on_month_direction": [
                _direction(p["closing_balance"], p["opening_balance"])
                for p in periods],
            "opening_loan_count": first["loan_count"],
            "closing_loan_count": last["loan_count"],
        },
        # -- what the case ASSERTS about the platform ----------------------- #
        "expected_risk_status": case.risk_intent.intended_status,
        "expected_targeted_limits": list(case.risk_intent.targeted_limits),
        "expected_regime": {
            "applicability": case.regime.applicability,
            "regime_id": case.regime.regime_id,
            "artefact_status": case.regime.artefact_status,
            "reason": case.regime.reason,
        },
        "expected_validation_warnings": list(case.expected_validation_warnings),
        "expected_outcome": case.expectation,
        "expected_failure_class": case.expected_failure_class,
    }


def _direction(current: float, prior: float, tolerance: float = 0.5) -> str:
    if current - prior > tolerance:
        return "increase"
    if prior - current > tolerance:
        return "decrease"
    return "flat"


#: Period figures that are ADDITIVE across separately delivered funded
#: populations inside one SPV. A balance or a count sums; a weighted average
#: does not, and a "largest concentration" is not the sum of two largests, so
#: neither is combined here — the SPV-level value for those is measured from
#: the assembled book, never inferred.
_ADDITIVE_PERIOD_FIELDS = (
    "loan_count", "exited_loan_count", "reported_row_count",
    "opening_balance", "closing_balance", "period_movement",
    "funded_additions", "acquired_additions", "total_additions",
    "rolled_up_interest", "further_advances", "redemptions", "repayments",
    "principal_amortisation", "total_reductions", "write_offs", "recoveries",
    "asset_disposals", "restatement", "movement_delta",
)


def combine_truths(case: CaseManifest,
                   by_source: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The SPV-level expected truth for a multi-source case.

    Computed by ADDING the independently computed per-source truths, which is
    the whole point: the platform figure a case asserts against is arrived at
    without reading anything the platform produced, and without re-running the
    generator over a merged book.

    Only genuinely additive figures are combined. Weighted averages and
    largest-concentration figures are deliberately omitted rather than
    approximated — an SPV's weighted-average LTV is not the sum, nor the mean,
    of its parts' weighted averages.
    """
    order = [s.portfolio_id for s in case.funded_sources()]
    firsts = [by_source[pid] for pid in order]
    dates = firsts[0]["reporting_dates"]
    for pid in order:
        if by_source[pid]["reporting_dates"] != dates:
            raise ValueError(
                f"{pid}: reporting dates differ across sources; a multi-source "
                f"SPV must deliver aligned month-ends")

    periods: List[Dict[str, Any]] = []
    for index, date in enumerate(dates):
        parts = [by_source[pid]["periods"][index] for pid in order]
        combined: Dict[str, Any] = {"reporting_date": date, "period_index": index}
        for field in _ADDITIVE_PERIOD_FIELDS:
            values = [p[field] for p in parts]
            total = sum(values)
            combined[field] = (round(total, 2) if isinstance(total, float)
                               else int(total))
        combined["movement_reconciles"] = all(p["movement_reconciles"]
                                              for p in parts)
        # Key-wise sums: a status balance, a vintage balance and a cohort
        # balance all add across separately delivered populations.
        for field in ("status_balances", "status_counts", "vintage_balances",
                      "cohort_balances"):
            merged: Dict[str, float] = defaultdict(float)
            for part in parts:
                for key, value in (part.get(field) or {}).items():
                    merged[key] += value
            combined[field] = {
                k: (round(v, 2) if isinstance(v, float) else int(v))
                for k, v in sorted(merged.items())}
        # A BALANCE-weighted average recombines exactly, because the weight is
        # the balance each part already reports: Σ(avg_i · bal_i) / Σ(bal_i).
        for field in ("weighted_average_current_ltv",
                      "weighted_average_interest_rate"):
            num = sum((p[field] or 0.0) * p["closing_balance"]
                      for p in parts if p.get(field) is not None)
            den = sum(p["closing_balance"] for p in parts
                      if p.get(field) is not None)
            combined[field] = round(num / den, 6) if den else None
        # Deliberately NOT combined: ``largest_concentrations``,
        # ``largest_borrower`` and ``asset_measures``. The largest group in a
        # combined book is not the larger of two largest groups — the true
        # answer needs the full group breakdown, which the per-source truth
        # does not carry. Rather than approximate it, the SPV truth omits the
        # figure and the assertions that need it state that they skipped it.
        combined["by_source"] = {
            pid: {"loan_count": by_source[pid]["periods"][index]["loan_count"],
                  "closing_balance":
                      by_source[pid]["periods"][index]["closing_balance"],
                  "opening_balance":
                      by_source[pid]["periods"][index]["opening_balance"]}
            for pid in order}
        periods.append(combined)

    first, last = periods[0], periods[-1]
    sources = {s.portfolio_id: s.source_portfolio_type
               for s in case.funded_sources()}
    return {
        "truth_version": TRUTH_VERSION,
        "case_id": case.case_id, "seed": case.seed,
        "simulation_version": case.simulation_version,
        "asset_class": case.asset_class, "asset_subtype": case.asset_subtype,
        "client_id": case.client_id, "portfolio_id": case.portfolio_id,
        "spv_id": case.spv_id, "currency": case.currency,
        "multi_source": True,
        "source_portfolio_ids": order,
        "source_portfolio_types": sources,
        "reporting_dates": list(dates),
        "periods": periods,
        "by_source": by_source,
        "history_summary": {
            "opening_balance": first["opening_balance"],
            "closing_balance": last["closing_balance"],
            "total_movement": round(last["closing_balance"]
                                    - first["opening_balance"], 2),
            "direction": _direction(last["closing_balance"],
                                    first["opening_balance"]),
            "month_on_month_direction": [
                _direction(p["closing_balance"], p["opening_balance"])
                for p in periods],
            "opening_loan_count": first["loan_count"],
            "closing_loan_count": last["loan_count"],
        },
        "expected_risk_status": case.risk_intent.intended_status,
        "expected_targeted_limits": list(case.risk_intent.targeted_limits),
        "expected_regime": {
            "applicability": case.regime.applicability,
            "regime_id": case.regime.regime_id,
            "artefact_status": case.regime.artefact_status,
            "reason": case.regime.reason,
        },
        "expected_validation_warnings": list(case.expected_validation_warnings),
    }


def write_expected_truth(truth: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stable serialisation: sorted keys, fixed separators, no wall-clock.
    path.write_text(json.dumps(truth, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")
    return path


__all__ = ["TRUTH_VERSION", "build_expected_truth", "combine_truths",
           "period_truth", "write_expected_truth"]
