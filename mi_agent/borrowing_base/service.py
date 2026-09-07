"""mi_agent.borrowing_base.service — the stable deterministic measure interface.

The seam a later MI Query Agent sprint registers governed measures against.
Nothing here imports Streamlit, FastAPI, React or any dashboard module, and
nothing here parses a question: it takes a trusted canonical frame and a client
identifier and returns named measures with their units, definitions and
provenance.

The MI Query Agent is NOT wired to this in this sprint. Its parsing, semantic
composition and routing are untouched; this module is what a later sprint can
register without redesigning anything.

    from mi_agent.borrowing_base.service import evaluate, measure_definitions

    snapshot = evaluate(df, client_id="ere_funding_uk")
    snapshot.measure("borrowing_base")        # -> 103_000_000.0
    snapshot.measure("borrowing_base_headroom")  # -> "NOT_CALCULABLE"

Every measure is either a number or :data:`NOT_CALCULABLE`. There is no third
possibility, and a measure is never zero because an input was missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .calculator import BorrowingBaseResult, ReconciliationError, calculate
from .config import load_facility
from .models import NOT_CALCULABLE, FacilityConfiguration
from .receipt import build_receipt


@dataclass(frozen=True)
class MeasureDefinition:
    """One governed measure a later MI sprint can register."""

    measure_id: str
    display_name: str
    unit: str                 # currency | percent | count | label
    definition: str
    #: How the value is read off a :class:`BorrowingBaseResult`.
    reader: Callable[[BorrowingBaseResult], Any] = field(repr=False,
                                                         default=lambda r: None)

    def to_dict(self) -> Dict[str, Any]:
        return {"measure_id": self.measure_id,
                "display_name": self.display_name,
                "unit": self.unit,
                "definition": self.definition}


def _nearest(key: str) -> Callable[[BorrowingBaseResult], Any]:
    return lambda r: (r.nearest or {}).get(key, NOT_CALCULABLE)


#: THE REGISTRY. Ordered, stable identifiers — a later MI sprint binds to these
#: names, so renaming one is a breaking change, not a tidy-up.
MEASURES: Dict[str, MeasureDefinition] = {
    m.measure_id: m for m in (
        MeasureDefinition(
            "financing_portfolio_balance", "Financing Portfolio balance",
            "currency",
            "Current balance of every loan in the facility's Financing "
            "Portfolio, whatever its eligibility status.",
            lambda r: r.eligibility.financing_portfolio_balance),
        MeasureDefinition(
            "eligible_balance", "Eligible collateral", "currency",
            "Current balance of loans whose governed eligibility status is "
            "ELIGIBLE.",
            lambda r: r.eligibility.eligible_current_balance),
        MeasureDefinition(
            "ineligible_balance", "Ineligible collateral", "currency",
            "Current balance of loans whose governed eligibility status is "
            "INELIGIBLE.",
            lambda r: r.eligibility.ineligible_current_balance),
        MeasureDefinition(
            "undetermined_balance", "Undetermined collateral", "currency",
            "Current balance of loans whose eligibility cannot yet be "
            "determined — never counted as eligible.",
            lambda r: r.eligibility.undetermined_current_balance),
        MeasureDefinition(
            "eligible_loan_count", "Eligible loans", "count",
            "Number of loans whose governed eligibility status is ELIGIBLE.",
            lambda r: r.eligibility.eligible_loan_count),
        MeasureDefinition(
            "concentration_limit_denominator",
            "Concentration Limit Denominator", "currency",
            "MAX(contractual floor, eligible current balance).",
            lambda r: r.concentration_limit_denominator),
        MeasureDefinition(
            "advance_rate", "Advance rate", "percent",
            "The facility's contractual advance rate, as a percentage.",
            lambda r: (NOT_CALCULABLE if r.advance_rate is None
                       else round(r.advance_rate * 100.0, 6))),
        MeasureDefinition(
            "gross_borrowing_base", "Gross borrowing base", "currency",
            "Eligible current balance x advance rate, before the facility cap.",
            lambda r: r.gross_borrowing_base),
        MeasureDefinition(
            "borrowing_base", "Borrowing base", "currency",
            "MIN(gross borrowing base, facility commitment) — the available "
            "borrowing base.",
            lambda r: r.available_borrowing_base),
        MeasureDefinition(
            "facility_commitment", "Facility commitment", "currency",
            "The facility's committed amount.",
            lambda r: (NOT_CALCULABLE if r.facility_commitment is None
                       else r.facility_commitment)),
        MeasureDefinition(
            "facility_drawn", "Facility drawn", "currency",
            "Current drawings under the facility, as supplied by the operator.",
            lambda r: r.current_drawn_amount),
        MeasureDefinition(
            "borrowing_base_headroom", "Borrowing-base headroom", "currency",
            "Available borrowing base minus current drawings. Retained "
            "negative when the facility is over-drawn.",
            lambda r: r.borrowing_base_headroom),
        MeasureDefinition(
            "borrowing_base_deficiency", "Borrowing-base deficiency",
            "currency",
            "ABS(MIN(headroom, 0)) — the amount by which drawings exceed the "
            "available borrowing base.",
            lambda r: r.borrowing_base_deficiency),
        MeasureDefinition(
            "borrowing_base_utilisation", "Borrowing-base utilisation",
            "percent",
            "Current drawings as a percentage of the available borrowing base.",
            lambda r: r.borrowing_base_utilisation_pct),
        MeasureDefinition(
            "facility_utilisation", "Facility utilisation", "percent",
            "Current drawings as a percentage of the facility commitment.",
            lambda r: r.facility_utilisation_pct),
        MeasureDefinition(
            "nearest_concentration_limit", "Nearest concentration limit",
            "label",
            "The binding Schedule 8 concentration limit, chosen "
            "deterministically: any breach outranks any non-breach; among "
            "breaches the largest utilisation binds; otherwise the smallest "
            "headroom binds.",
            _nearest("nearest_concentration_limit")),
        MeasureDefinition(
            "nearest_concentration_headroom",
            "Nearest concentration headroom", "percent",
            "Percentage-point headroom on the binding concentration limit.",
            _nearest("nearest_concentration_headroom_pct")),
        MeasureDefinition(
            "nearest_concentration_headroom_amount",
            "Nearest concentration headroom (amount)", "currency",
            "Headroom on the binding concentration limit converted to "
            "currency at that test's own denominator.",
            _nearest("nearest_concentration_headroom_amount")),
        MeasureDefinition(
            "breached_concentration_count", "Breached concentrations", "count",
            "How many Schedule 8 concentration limits are currently breached.",
            _nearest("breached_concentration_count")),
    )
}

#: Stable ordering for any caller that enumerates.
MEASURE_IDS = tuple(MEASURES)


def measure_definitions() -> List[Dict[str, Any]]:
    """The registry, as plain dicts. Safe to serialise into a catalogue."""
    return [MEASURES[m].to_dict() for m in MEASURE_IDS]


@dataclass
class BorrowingBaseSnapshot:
    """One evaluation: the facility, the figures, the measures, the receipt."""

    available: bool
    reason: str = ""
    facility: Optional[FacilityConfiguration] = None
    result: Optional[BorrowingBaseResult] = None
    receipt: Dict[str, Any] = field(default_factory=dict)
    configuration_problems: List[str] = field(default_factory=list)

    def measure(self, measure_id: str) -> Any:
        """One named measure, or NOT_CALCULABLE when it cannot be produced."""
        definition = MEASURES.get(measure_id)
        if definition is None:
            raise KeyError(f"Unknown borrowing-base measure {measure_id!r}.")
        if self.result is None:
            return NOT_CALCULABLE
        value = definition.reader(self.result)
        return NOT_CALCULABLE if value is None else value

    def measures(self) -> Dict[str, Any]:
        return {m: self.measure(m) for m in MEASURE_IDS}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason or None,
            "facility": self.facility.summary() if self.facility else None,
            "configurationProblems": self.configuration_problems,
            **(self.result.to_dict() if self.result else {}),
            "measures": self.measures() if self.result else {},
            "receipt": self.receipt,
        }


def evaluate(
    df: Optional[pd.DataFrame],
    *,
    client_id: str = "",
    facility: Optional[FacilityConfiguration] = None,
    concentration_results: Optional[List[Dict[str, Any]]] = None,
    concentration_rule_version: str = "",
    concentration_configuration_hash: str = "",
    concentration_library_version: str = "",
    as_of_date: str = "",
    portfolio_scope: Optional[Dict[str, Any]] = None,
    eligibility_derivation: Optional[Dict[str, Any]] = None,
    run_id: str = "",
    lib: Any = None,
    strict: bool = True,
) -> BorrowingBaseSnapshot:
    """Evaluate the borrowing base for one frame. The whole public entry point.

    Pass ``facility`` to calculate against explicit approved terms; otherwise
    the governed configuration for ``client_id`` is loaded. With neither, the
    snapshot is unavailable with the reason stated — never a default facility.
    """
    facility = facility or (load_facility(client_id) if client_id else None)
    if facility is None:
        return BorrowingBaseSnapshot(
            available=False,
            reason=("No funding facility is configured for this portfolio. "
                    "Borrowing-base monitoring begins when OCC records an "
                    "approved facility configuration."))
    problems = facility.validate()
    if any(not p.endswith("It will NOT be honoured.") for p in problems):
        return BorrowingBaseSnapshot(
            available=False,
            reason="The facility configuration is not usable: "
                   + "; ".join(problems),
            facility=facility, configuration_problems=problems)

    try:
        result = calculate(df, facility,
                           concentration_results=concentration_results,
                           lib=lib, strict=strict)
    except ReconciliationError as exc:
        return BorrowingBaseSnapshot(
            available=False, reason=str(exc), facility=facility,
            configuration_problems=problems)

    receipt = build_receipt(
        result, facility,
        as_of_date=as_of_date,
        portfolio_scope=portfolio_scope,
        eligibility_derivation=eligibility_derivation,
        concentration_results=concentration_results,
        concentration_rule_version=concentration_rule_version,
        concentration_configuration_hash=concentration_configuration_hash,
        concentration_library_version=concentration_library_version,
        run_id=run_id)
    return BorrowingBaseSnapshot(available=True, facility=facility,
                                 result=result, receipt=receipt,
                                 configuration_problems=problems)


__all__ = ["evaluate", "measure_definitions", "MEASURES", "MEASURE_IDS",
           "BorrowingBaseSnapshot", "MeasureDefinition"]
