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


# --------------------------------------------------------------------------- #
# Reading the sentence
# --------------------------------------------------------------------------- #
import re                                                       # noqa: E402
from dataclasses import dataclass                               # noqa: E402

ROUTE_NAME = "pipeline_movement_summary"
RECOGNITION_KEY = "pipeline_movement_summary"

#: The SUBJECT: the pipeline's movement as a whole. Both words are required —
#: "movement" alone is a funded-bridge word ("Show movement by region") and
#: claiming it here would take a question this route cannot answer.
_MOVEMENT = r"(?:stage movement|pipeline movement|movement (?:through|across) "\
            r"the (?:pipeline|funnel)|funnel movement)"
#: The SHAPE: a summary of it, or a comparison of it.
_SUMMARY = r"(?:summar(?:y|ise|ize)|overview|breakdown|reconcil)"
_COMPARATIVE = r"(?:compare|versus|vs\b|changed?|prior period|last period|"\
               r"previous period|since last|month on month|period on period)"

_MOVEMENT_RE = re.compile(_MOVEMENT, re.I)
_SUMMARY_RE = re.compile(_SUMMARY, re.I)
_COMPARATIVE_RE = re.compile(_COMPARATIVE, re.I)


@dataclass(frozen=True)
class MovementSummaryRequest:
    """What the sentence asked for. Carries no data and decides no figure."""

    comparative: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"subtype": "comparison" if self.comparative else "summary"}


def read(question: Any) -> Optional[MovementSummaryRequest]:
    """The reading, or ``None`` where this route may not claim the sentence.

    TWO GATES, and the second is what keeps this route from stealing work.
    The sentence must name the pipeline's movement as a subject, AND ask for it
    summarised or compared. "Show movement by region" names neither and stays
    with the funded bridge; "How many cases moved from KFI to Application?"
    names a transition and stays with `pipeline_stage_movement`, which reads it
    first because it registers first.
    """
    text = str(question or "")
    if not _MOVEMENT_RE.search(text):
        return None
    if not (_SUMMARY_RE.search(text) or _COMPARATIVE_RE.search(text)):
        return None
    # A SENTENCE THAT NAMES ONE STAGE IN MOTION IS NOT THIS. Asked of the
    # existing owner rather than re-read, so the two can never disagree about
    # what a single stage movement is — and so this route never has to invent
    # a stage to honour a question.
    from . import stage_movement_query as _stage

    if _stage.names_a_stage_movement(text):
        return None
    return MovementSummaryRequest(comparative=bool(_COMPARATIVE_RE.search(text)))


def compare(current: Dict[str, Any], prior: Dict[str, Any]) -> Dict[str, Any]:
    """Two governed movement summaries, and the difference between them.

    BOTH SIDES ARE MOVEMENTS. A comparison that put a point-in-time stock on
    one side would answer a different question from the one asked — which is
    exactly what happened when `temporal_compare` claimed "Compare stage
    movement with the prior period".
    """
    if not (current or {}).get("available"):
        return {"available": False, "version": SUMMARY_VERSION,
                "reason": (current or {}).get("reason")
                          or "No governed movement summary for this period."}
    if not (prior or {}).get("available"):
        return {"available": False, "version": SUMMARY_VERSION,
                "reason": (prior or {}).get("reason")
                          or "There is no prior governed movement interval to "
                             "compare this one against."}
    def _delta(block: str, key: str) -> float:
        return (_num(current[block][key]) - _num(prior[block][key]))

    return {
        "available": True,
        "version": SUMMARY_VERSION,
        "current": current,
        "prior": prior,
        "delta": {
            "net_cases": _delta("net", "cases"),
            "net_balance": _delta("net", "balance"),
            "entrant_cases": _delta("entrants", "cases"),
            "entrant_balance": _delta("entrants", "balance"),
            "departure_cases": _delta("departures", "cases"),
            "departure_balance": _delta("departures", "balance"),
        },
    }


# --------------------------------------------------------------------------- #
# The route
# --------------------------------------------------------------------------- #
def resolve(root: str, client_id: str, *, history=None,
            as_of: Optional[str] = None) -> Dict[str, Any]:
    """The summary for one governed interval, from the governed resolver.

    A thin composition seam, published so the Teams briefing can ask for the
    same structure without going through the chat route.
    """
    from . import movement_detail as detail

    return build(detail.resolve_stage_transition_detail(
        root, client_id, as_of=as_of, historical_model=history))


def resolve_comparison(root: str, client_id: str, *, history=None
                       ) -> Dict[str, Any]:
    """The latest governed interval against the one before it.

    The prior interval is the SAME resolver asked for an earlier point — the
    neighbour rule already owns which extracts pair, so there is no second
    pairing engine and no second idea of what a period is.
    """
    current = resolve(root, client_id, history=history)
    if not current.get("available"):
        return compare(current, {})
    opening = (current.get("window") or {}).get("opening_date")
    prior = resolve(root, client_id, history=history, as_of=opening)
    return compare(current, prior)


def recognise(request: Any) -> Any:
    from .recogniser_registry import Recognition

    reading = read(getattr(request, "question", ""))
    if reading is None:
        return Recognition.no("no governed pipeline movement summary request")
    request.remember_recognition(RECOGNITION_KEY, reading)
    return Recognition.yes(reason=reading.to_dict()["subtype"])


def handle(request: Any) -> Optional[Dict[str, Any]]:
    """Compose the governed summary and publish it. Renders; computes nothing."""
    from . import currency as currency_mod
    from .chat_routing import _envelope, _source, _table_artifact, _undeliverable

    reading = request.recalled_recognition(RECOGNITION_KEY) or read(request.question)
    if reading is None:
        return None
    root = request.pipeline_root
    if not root:
        return None

    spec_dict = dict(request.spec_dict or {})
    history = request.resolve_history_model()
    money = lambda v: currency_mod.format_money(v, suffixes=("bn", "m", "k"))

    if reading.comparative:
        result = resolve_comparison(root, request.client_id, history=history)
        if not result.get("available"):
            return _undeliverable(
                question=request.question, spec=spec_dict, artifacts=[],
                answer=result.get("reason"), error=result.get("reason"),
                route=ROUTE_NAME, warnings=[result.get("reason")])
        summary = result["current"]
        delta = result["delta"]
        answer = (
            "Pipeline movement %s to %s, against the interval ending %s: "
            "net %+d cases (%s), %+d entrants, %+d departures."
            % (summary["window"]["opening_date"], summary["window"]["closing_date"],
               result["prior"]["window"]["closing_date"],
               summary["net"]["cases"], money(summary["net"]["balance"]),
               delta["entrant_cases"], delta["departure_cases"]))
        meta_key, meta_value = "pipelineMovementComparison", result
    else:
        summary = resolve(root, request.client_id, history=history)
        if not summary.get("available"):
            return _undeliverable(
                question=request.question, spec=spec_dict, artifacts=[],
                answer=summary.get("reason"), error=summary.get("reason"),
                route=ROUTE_NAME, warnings=[summary.get("reason")])
        answer = (
            "Pipeline movement %s to %s: opened %d cases (%s), closed %d (%s), "
            "net %+d (%s). %d new entrants, %d progressions, %d departures."
            % (summary["window"]["opening_date"], summary["window"]["closing_date"],
               summary["opening"]["cases"], money(summary["opening"]["balance"]),
               summary["closing"]["cases"], money(summary["closing"]["balance"]),
               summary["net"]["cases"], money(summary["net"]["balance"]),
               summary["entrants"]["cases"],
               sum(p["cases"] for p in summary["progressions"]),
               summary["departures"]["cases"]))
        meta_key, meta_value = "pipelineMovementSummary", summary

    if summary.get("omitted"):
        answer += " Not evidenced: " + "; ".join(
            "%s (%s)" % (o["element"], o["reason"]) for o in summary["omitted"]) + "."

    rows = [{"stage": r["stage"], "opening_cases": r["opening_cases"],
             "arrivals": r["arrivals"], "departures": r["departures"],
             "closing_cases": r["closing_cases"],
             "opening_balance": r["opening_balance"],
             "closing_balance": r["closing_balance"]}
            for r in summary["by_stage"]]
    window = summary["window"]
    columns = [{"key": k, "label": k.replace("_", " ").title()}
               for k in ("stage", "opening_cases", "arrivals", "departures",
                         "closing_cases", "opening_balance", "closing_balance")]
    artifact = _table_artifact(
        "Pipeline movement by stage", columns=columns, rows=rows,
        spec=spec_dict, portfolio_id=request.client_id,
        as_of=window["closing_date"],
        description="Opening, arrivals, departures and closing per governed stage.")
    # WHAT THIS ANSWER READ, in the estate's own record. The coverage ledger
    # accounts for a `dataset` concept against `reconciliation.dataset` and
    # nothing else — the DECISION (`datasetContext`) is not evidence that the
    # tape was read. This summary reads the governed weekly pipeline extracts,
    # so it says so, and the case counts are the ones it reported.
    reconciliation = {
        "dataset": "pipeline",
        "total_records": summary["closing"]["cases"],
        "records_included": summary["closing"]["cases"],
        "opening_records": summary["opening"]["cases"],
        "reporting_date": window["closing_date"],
        "comparison_date": window["opening_date"],
    }
    envelope = _envelope(
        ok=True, question=request.question, spec=spec_dict, answer=answer,
        reconciliation=reconciliation,
        artifacts=[artifact] if artifact else [], route=ROUTE_NAME,
        source_notes=[_source("Governed pipeline movement summary", spec_dict,
                              request.client_id, window["closing_date"],
                              engine="mi_agent_api.pipeline_movement_summary")])
    meta = envelope.setdefault("metadata", {})
    meta[meta_key] = meta_value
    # THE AXIS THIS ANSWER IS CUT BY, declared rather than inferred. The summary
    # reports every governed stage, so a question naming "stage" is answered on
    # that axis and the receipt can see it — the same `groupedBy` declaration
    # other routes make about themselves.
    from question_interpretation.lexical import PIPELINE_STAGE_FIELD

    meta["groupedBy"] = [PIPELINE_STAGE_FIELD]
    meta["movementWindow"] = dict(summary["window"])
    return envelope


def recogniser():
    """This capability's registry entry.

    Priority 79, ahead of `pipeline_summary` at 81 — which claimed "Give me the
    stage movement summary" on the word "summary" alone and then refused,
    because a point-in-time pipeline summary can honour neither a comparison
    period nor a stage. That is the substitution this route removes.

    79 rather than 80 because `portfolio_summary` (80) and `pipeline_summary`
    (81) are ONE capability split by dataset and a test asserts they stay
    adjacent. Reading before both is safe: `read` requires the pipeline's
    movement as a subject, so "Summarise the funded portfolio" is not a
    sentence this route can claim.

    It cannot take work from `pipeline_stage_movement` (120) despite reading
    first: `read` asks `names_a_stage_movement` directly and returns None for
    any sentence that owner claims, so deference here is EXPLICIT rather than
    an artefact of registration order — and the two can never disagree about
    what a single stage movement is, because only one of them defines it.
    """
    from .recogniser_registry import Recogniser

    return Recogniser(
        name=ROUTE_NAME, priority=79, lens_aware=False,
        description=("Whole-pipeline movement across every governed stage for "
                     "one reporting interval, and against the prior one."),
        metadata={
            "governed_capability":
                "mi_agent_api.movement_detail.resolve_stage_transition_detail",
            "computes_nothing": True,
            "structured_result": "pipelineMovementSummary",
        },
        recognise=recognise, handle=handle)
