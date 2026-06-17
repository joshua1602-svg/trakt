"""
delivery_readiness.py
======================

Delivery-readiness gates for the Delivery/XML Agent v1.

Each gate is a named, auditable check with a boolean ``passed`` and a human
reason. ``xml_generation_allowed`` is true **only** when every gate passes — the
agent refuses XML otherwise. The gates intentionally mirror the review questions
in ``docs/delivery_xml_agent_v1_review.md`` (Q7).

This module is pure: it computes the gates from already-derived facts and never
performs I/O, never mutates the frame, and never raises.
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__ = [
    "compute_delivery_readiness",
    "GATE_NAMES",
]

GATE_NAMES = [
    "projection_complete",
    "ready_for_delivery_normalisation",
    "ready_for_xml_delivery",
    "no_delivery_blocking_projection_issues",
    "no_blocked_target_frame_rows",
    "no_mandatory_blank_without_nd",
    "no_delivery_format_violations",
    "required_header_metadata_present",
    "record_grouping_determinable",
    "template_code_order_complete",
]


def _gate(name: str, passed: bool, reason: str) -> Dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "reason": reason}


def compute_delivery_readiness(
    *,
    projection_complete: bool,
    ready_for_delivery_normalisation: bool,
    ready_for_xml_delivery: bool,
    delivery_blocking_projection_issue_count: int,
    blocked_frame_row_count: int,
    mandatory_blank_without_nd_count: int,
    format_violation_count: int,
    missing_header_metadata: List[str],
    rows_without_record_group: int,
    missing_required_order_codes: List[str],
) -> Dict[str, Any]:
    """Compute the named delivery-readiness gates and the overall verdict.

    Returns a dict with ``gates`` (list), ``xml_generation_allowed`` and
    ``delivery_normalisation_complete``.
    """
    gates: List[Dict[str, Any]] = [
        _gate("projection_complete", projection_complete,
              "projection package reports projection_complete"),
        _gate("ready_for_delivery_normalisation", ready_for_delivery_normalisation,
              "projection package reports ready_for_delivery_normalisation"),
        _gate("ready_for_xml_delivery", ready_for_xml_delivery,
              "projection package reports ready_for_xml_delivery (always false pre-delivery)"),
        _gate("no_delivery_blocking_projection_issues",
              delivery_blocking_projection_issue_count == 0,
              f"{delivery_blocking_projection_issue_count} delivery-blocking projection issue(s) remain"),
        _gate("no_blocked_target_frame_rows", blocked_frame_row_count == 0,
              f"{blocked_frame_row_count} target-frame row(s) carry a blocked_* status"),
        _gate("no_mandatory_blank_without_nd", mandatory_blank_without_nd_count == 0,
              f"{mandatory_blank_without_nd_count} mandatory field(s) blank without an allowed/selected ND"),
        _gate("no_delivery_format_violations", format_violation_count == 0,
              f"{format_violation_count} value(s) violate delivery format/enum rules"),
        _gate("required_header_metadata_present", not missing_header_metadata,
              ("all required XML header/report metadata present"
               if not missing_header_metadata
               else f"missing header/report metadata: {', '.join(missing_header_metadata)}")),
        _gate("record_grouping_determinable", rows_without_record_group == 0,
              f"{rows_without_record_group} row(s) have an indeterminate record group"),
        _gate("template_code_order_complete", not missing_required_order_codes,
              ("template/code order complete for required XML fields"
               if not missing_required_order_codes
               else f"{len(missing_required_order_codes)} required code(s) missing from esma_code_order")),
    ]

    xml_generation_allowed = all(g["passed"] for g in gates)
    # Delivery normalisation can complete even if some non-mandatory structural
    # gates fail, but for v1 we keep it strict: normalisation is complete only
    # when there are no blocked rows, no blocking issues and no format violations.
    delivery_normalisation_complete = bool(
        blocked_frame_row_count == 0
        and delivery_blocking_projection_issue_count == 0
        and mandatory_blank_without_nd_count == 0
        and format_violation_count == 0
        and ready_for_delivery_normalisation
    )
    return {
        "gates": gates,
        "xml_generation_allowed": xml_generation_allowed,
        "delivery_normalisation_complete": delivery_normalisation_complete,
    }
