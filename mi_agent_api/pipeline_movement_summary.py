"""The whole pipeline's movement for one governed interval, as one structure.

WHAT THIS IS FOR. `stage_movement_query` answers about ONE stage or ONE
transition: it needs a named stage or a source/destination pair and returns
nothing without them. "Give me the stage movement summary" names neither, and
routing it at that capability would have forced it to invent a stage — the
substitution the estate forbids. What was missing is not a route but an
ALL-STAGES summary, and this is it.

IT COMPOSES, IT DOES NOT CALCULATE. Every figure is read from the governed
payload `movement_detail.resolve_stage_transition_detail` already publishes —
the same payload the stage route, the React movement-detail endpoint and the
PPTX deck consume. There is no second preparation path, no second snapshot
pairing, and no metric defined here.

WHERE THE DATA DOES NOT EVIDENCE AN ELEMENT, IT IS OMITTED AND NAMED.
`governed_outcome` is a canonical terminal stage only where the prior extract
recorded one; everything else stays `unclassified_departure`. Reporting those as
withdrawals would invent an economic fact, so the withdrawal split is omitted
with its reason rather than filled in. The same rule governs every element in
`omitted`.

ONE RESULT, TWO CONSUMERS. The MI query route renders this; the Teams proactive
briefing consumes the same structure. So it is DATA — nothing here formats a
sentence, ranks for readability or writes prose. A model may summarise these
facts downstream; it may not change them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

#: Bumped when the published shape changes in a way a consumer must notice.
SUMMARY_VERSION = 1

#: The governed stage a completion lands in. Read from the payload's own
#: transitions; never inferred from a departure.
COMPLETED_STAGE = "COMPLETED"

#: A departure the prior extract gave no terminal stage for. Not a withdrawal.
UNCLASSIFIED_DEPARTURE = "unclassified_departure"


def _num(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _unavailable(reason: str, code: Optional[str] = None) -> Dict[str, Any]:
    return {"available": False, "version": SUMMARY_VERSION,
            "reason": reason, "reason_code": code}


def build(payload: Any) -> Dict[str, Any]:
    """The movement summary for one governed snapshot pair.

    Returns ``{"available": False, "reason": ...}`` for a payload that carries
    no comparison, and never raises on a shape it does not recognise: a
    briefing that cannot be built must say so, not fail its caller.
    """
    if not isinstance(payload, dict):
        return _unavailable("No governed pipeline movement payload was supplied.")
    if not payload.get("available"):
        return _unavailable(
            str(payload.get("reason")
                or "No governed pipeline movement is available for this period."),
            payload.get("reason_code"))

    reconciliation = payload.get("reconciliation") or {}
    by_stage_in = reconciliation.get("by_stage") or []
    if not by_stage_in:
        return _unavailable(
            "The governed movement payload carries no per-stage reconciliation.",
            payload.get("reason_code"))

    omitted: List[Dict[str, str]] = []

    # ---- per stage, straight from the governed reconciliation ------------- #
    by_stage = [{
        "stage": row.get("stage"),
        "opening_cases": _int(row.get("opening_case_count")),
        "arrivals": _int(row.get("new_arrivals")) + _int(row.get("transitions_in")),
        "new_arrivals": _int(row.get("new_arrivals")),
        "transitions_in": _int(row.get("transitions_in")),
        "transitions_out": _int(row.get("transitions_out")),
        "departures": _int(row.get("departures")),
        "stayers": _int(row.get("stayers")),
        "closing_cases": _int(row.get("closing_case_count")),
        "opening_balance": _num(row.get("opening_amount")),
        "closing_balance": _num(row.get("closing_amount")),
        "stayer_balance_change": _num(row.get("stayer_amount_change")),
        "count_residual": _int(row.get("count_reconciliation_residual")),
        "balance_residual": _num(row.get("amount_reconciliation_residual")),
    } for row in by_stage_in]

    counts = payload.get("counts") or {}
    opening_cases = _int(counts.get("comparison"))
    closing_cases = _int(counts.get("current"))
    opening_balance = sum(r["opening_balance"] for r in by_stage)
    closing_balance = sum(r["closing_balance"] for r in by_stage)

    totals = payload.get("event_totals") or {}
    entrants = totals.get("new_arrival") or {}
    stayers = totals.get("stayer") or {}
    departed = totals.get("departure") or {}

    # ---- progressions, as the payload states them ------------------------- #
    progressions = [{
        "source": t.get("source_stage"),
        "destination": t.get("destination_stage"),
        "cases": _int(t.get("case_count")),
        "balance": _num(t.get("latest_amount")),
        "balance_change": _num(t.get("amount_change")),
    } for t in (payload.get("transitions") or [])]

    # ---- completions: a governed transition INTO the completed stage ------ #
    completed = [p for p in progressions
                 if str(p["destination"] or "").upper() == COMPLETED_STAGE]
    if completed:
        completions = {"cases": sum(p["cases"] for p in completed),
                       "balance": sum(p["balance"] for p in completed)}
    else:
        completions = None
        omitted.append({
            "element": "completions",
            "reason": "no governed transition into %s in this interval"
                      % COMPLETED_STAGE})

    # ---- departures, split ONLY where the extract evidenced an outcome ---- #
    departure_rows = payload.get("departures") or []
    by_outcome: Dict[str, Dict[str, Any]] = {}
    for row in departure_rows:
        outcome = str(row.get("governed_outcome") or UNCLASSIFIED_DEPARTURE)
        bucket = by_outcome.setdefault(outcome, {"outcome": outcome, "cases": 0,
                                                 "balance": 0.0})
        bucket["cases"] += _int(row.get("case_count"))
        bucket["balance"] += _num(row.get("prior_amount"))
    # A COMPLETION IS NOT A WITHDRAWAL. Both are departures with an evidenced
    # terminal stage, and merging them would report cases that finished as cases
    # that fell out. Only a non-completed terminal outcome is attrition.
    attrition = {k: v for k, v in by_outcome.items()
                 if k != UNCLASSIFIED_DEPARTURE and k.upper() != COMPLETED_STAGE}
    if attrition:
        withdrawals = {"cases": sum(v["cases"] for v in attrition.values()),
                       "balance": sum(v["balance"] for v in attrition.values()),
                       "by_outcome": sorted(attrition.values(),
                                            key=lambda r: r["outcome"])}
    else:
        withdrawals = None
        omitted.append({
            "element": "withdrawals",
            "reason": "the governed extracts evidence no terminal outcome other "
                      "than completion for any departure in this interval; the "
                      "rest are reported as departures rather than resolved "
                      "into withdrawals"})
    if any(k == UNCLASSIFIED_DEPARTURE for k in by_outcome):
        omitted.append({
            "element": "departure_outcome_split",
            "reason": "%d departure(s) carry no terminal stage in the prior "
                      "extract and are reported unattributed"
                      % by_outcome[UNCLASSIFIED_DEPARTURE]["cases"]})

    # ---- ranking is ORDER, never a new fact ------------------------------- #
    largest = {
        "progressions": sorted(progressions, key=lambda r: -abs(r["balance"])),
        "attrition": sorted(
            (v for k, v in by_outcome.items() if k.upper() != COMPLETED_STAGE),
            key=lambda r: -abs(r["balance"])),
    }

    count_residual = sum(r["count_residual"] for r in by_stage)
    balance_residual = sum(r["balance_residual"] for r in by_stage)
    tolerance = _num(reconciliation.get("amount_tolerance")) or 0.01

    return {
        "available": True,
        "version": SUMMARY_VERSION,
        "measure": payload.get("measure"),
        "stage_field": payload.get("stage_field"),
        "portfolio_id": payload.get("portfolio_id"),
        "window": {"opening_date": payload.get("comparison_date"),
                   "closing_date": payload.get("as_of_date")},
        "opening": {"cases": opening_cases, "balance": opening_balance},
        "closing": {"cases": closing_cases, "balance": closing_balance},
        "net": {"cases": closing_cases - opening_cases,
                "balance": closing_balance - opening_balance},
        "entrants": {"cases": _int(entrants.get("case_count")),
                     "balance": _num(entrants.get("latest_amount"))},
        "progressions": progressions,
        "completions": completions,
        "departures": {"cases": _int(departed.get("case_count")),
                       "balance": _num(departed.get("prior_amount")),
                       "by_outcome": sorted(by_outcome.values(),
                                            key=lambda r: r["outcome"])},
        "withdrawals": withdrawals,
        "persistent": {"cases": _int(stayers.get("case_count")),
                       "balance_change": (_num(stayers.get("latest_amount"))
                                          - _num(stayers.get("prior_amount")))},
        "by_stage": by_stage,
        "largest": largest,
        "reconciliation": {
            "ok": count_residual == 0 and abs(balance_residual) <= tolerance,
            "count_residual": count_residual,
            "balance_residual": balance_residual,
            "balance_tolerance": tolerance,
        },
        "omitted": omitted,
        "sources": payload.get("sources"),
        "source_dates": payload.get("source_dates"),
    }
