"""mi_agent.borrowing_base.receipt — the reproducible calculation receipt.

One document that answers "why did Trakt report this borrowing base?" without
the reader needing access to the run: the facility terms and their version, the
eligibility rule version and every prototype assumption that was in play, the
population the figures were measured over, each equation's inputs and output,
and the concentration results as evaluated at the same moment.

The receipt is a plain dict so it serialises to JSON or YAML unchanged, and it
is snake_case throughout — it is an audit artefact, not a wire payload for the
React dashboard, which reads the camelCase envelope instead.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

from .calculator import BorrowingBaseResult
from .models import NOT_CALCULABLE, canonical_json, stable_hash

RECEIPT_SCHEMA_VERSION = "1.0.0"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _concentration_row(test: Dict[str, Any]) -> Dict[str, Any]:
    """One concentration result, reduced to what reproduces the conclusion."""
    return {
        "test_id": test.get("testId"),
        "metric_id": test.get("metricId"),
        "display_name": test.get("displayName"),
        "category": test.get("category"),
        "operator": test.get("operator"),
        "threshold": test.get("threshold"),
        "unit": test.get("unit"),
        "current_value": test.get("currentValue"),
        "numerator_value": test.get("numeratorValue"),
        "denominator_value": test.get("denominatorValue"),
        "denominator_basis": test.get("denominatorBasis"),
        "loans_in_numerator": test.get("loansInNumerator"),
        "headroom": test.get("headroom"),
        "utilisation_pct": test.get("utilization"),
        "breach_amount": test.get("breachAmount"),
        "status": test.get("status"),
        "data_status": test.get("dataStatus"),
        "missing_fields": test.get("missingFields") or [],
        "population": test.get("population"),
        "source_text": (test.get("provenance") or {}).get("sourceText"),
    }


def build_receipt(
    result: BorrowingBaseResult,
    facility,
    *,
    as_of_date: str = "",
    portfolio_scope: Optional[Dict[str, Any]] = None,
    eligibility_derivation: Optional[Dict[str, Any]] = None,
    concentration_results: Optional[List[Dict[str, Any]]] = None,
    concentration_rule_version: str = "",
    concentration_configuration_hash: str = "",
    concentration_library_version: str = "",
    run_id: str = "",
    timestamp: str = "",
) -> Dict[str, Any]:
    """The full audit receipt for one borrowing-base calculation."""
    eligibility = result.eligibility
    assumptions = list(result.prototype_assumptions_used)
    for note in ((eligibility_derivation or {}).get(
            "prototype_assumptions_used") or []):
        if note not in assumptions:
            assumptions.append(note)

    receipt: Dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_type": "borrowing_base_calculation",
        "timestamp": timestamp or _now_iso(),

        # --- what was calculated, on whose terms -------------------------- #
        "facility_id": result.facility_id,
        "facility_type": facility.facility_type,
        "facility_label": facility.facility_label or facility.facility_id,
        "client_id": facility.client_id,
        "currency": result.currency,
        "facility_config_version": facility.config_version,
        "facility_config_source": facility.config_source,
        "facility_config_hash": facility.content_hash(),
        "facility_environment": facility.environment,
        "facility_governance": dict(facility.governance or {}),
        "as_of_date": as_of_date or None,
        "run_id": run_id or None,

        # --- the population ------------------------------------------------ #
        "portfolio_scope": portfolio_scope or {},
        "financing_portfolio_loan_count":
            eligibility.financing_portfolio_loan_count,
        "financing_portfolio_balance": eligibility.financing_portfolio_balance,

        # --- eligibility --------------------------------------------------- #
        "eligibility_rule_version": facility.eligibility_rule_version or None,
        "eligibility_governed": facility.eligibility_governed,
        "eligibility_derivation": eligibility_derivation or {},
        "eligible_loan_count": eligibility.eligible_loan_count,
        "eligible_balance": eligibility.eligible_current_balance,
        "ineligible_loan_count": eligibility.ineligible_loan_count,
        "ineligible_balance": eligibility.ineligible_current_balance,
        "undetermined_loan_count": eligibility.undetermined_loan_count,
        "undetermined_balance": eligibility.undetermined_current_balance,

        # --- the equations, each with its inputs --------------------------- #
        "concentration_denominator": result.concentration_limit_denominator,
        "concentration_denominator_floor":
            result.concentration_limit_denominator_floor,
        "concentration_denominator_floor_binding":
            result.concentration_denominator_floor_binding,
        "advance_rate": result.advance_rate,
        "advance_rate_pct": (None if result.advance_rate is None
                             else round(result.advance_rate * 100.0, 6)),
        "facility_commitment": result.facility_commitment,
        "gross_borrowing_base": result.gross_borrowing_base,
        "available_borrowing_base": result.available_borrowing_base,
        "facility_cap_binding": result.facility_cap_binding,
        "current_drawn_amount": result.current_drawn_amount,
        "headroom": result.borrowing_base_headroom,
        "deficiency": result.borrowing_base_deficiency,
        "borrowing_base_utilisation_pct": result.borrowing_base_utilisation_pct,
        "facility_utilisation_pct": result.facility_utilisation_pct,
        "equations": [
            "concentration_denominator = MAX(floor, eligible_balance)",
            "gross_borrowing_base = eligible_balance x advance_rate",
            "available_borrowing_base = MIN(gross_borrowing_base, commitment)",
            "headroom = available_borrowing_base - current_drawn_amount",
            "deficiency = ABS(MIN(headroom, 0))",
            "borrowing_base_utilisation_pct = "
            "current_drawn_amount / available_borrowing_base x 100",
            "facility_utilisation_pct = "
            "current_drawn_amount / facility_commitment x 100",
        ],

        # --- concentrations ------------------------------------------------ #
        "concentration_rule_version": concentration_rule_version or None,
        "concentration_configuration_hash": concentration_configuration_hash
        or None,
        "concentration_library_version": concentration_library_version or None,
        "concentration_population": facility.concentration_population,
        "concentration_results": [_concentration_row(t)
                                  for t in (concentration_results or [])],
        "concentration_adjustment": result.concentration_adjustment,
        "nearest_concentration": dict(result.nearest),

        # --- honesty ------------------------------------------------------- #
        "invariants": eligibility.invariants,
        "reconciles": result.reconciles,
        "missing_inputs": result.missing_inputs,
        "not_calculable_measures": [
            name for name, value in (
                ("concentration_denominator",
                 result.concentration_limit_denominator),
                ("gross_borrowing_base", result.gross_borrowing_base),
                ("available_borrowing_base", result.available_borrowing_base),
                ("current_drawn_amount", result.current_drawn_amount),
                ("headroom", result.borrowing_base_headroom),
                ("deficiency", result.borrowing_base_deficiency),
                ("borrowing_base_utilisation_pct",
                 result.borrowing_base_utilisation_pct),
                ("facility_utilisation_pct", result.facility_utilisation_pct),
            ) if value == NOT_CALCULABLE],
        "prototype_assumptions_used": assumptions,
        "notes": list(result.notes),
    }
    # A content hash over everything except the wall-clock stamp, so two
    # receipts for the same inputs are provably the same calculation.
    stamped = {k: v for k, v in receipt.items() if k != "timestamp"}
    receipt["content_hash"] = stable_hash(canonical_json(stamped))
    return receipt


__all__ = ["build_receipt", "RECEIPT_SCHEMA_VERSION"]
