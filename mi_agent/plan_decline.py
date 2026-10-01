"""What the governed path says when it does not answer (owner decision D18).

Owner, 2026-09-30: "Do not use the old system." Until then a question the
governed path understood but declined fell through to the legacy path, and the
15:53 check showed what that costs: "Is any of the pipeline overdue to complete,
and how much?" was read correctly — the count and the amount of the overdue
cases — declined for asking two figures at once, and the legacy path answered
with the WHOLE pipeline, dropping "overdue". A wrong answer is worse than no
answer.

So a decline is the answer. It says, in words a reader uses, what the question
was understood as and why it is not answered, and it never carries a figure.
Nothing here computes, substitutes or re-reads the question: it reads the
evidence record the canary already wrote — the model's reading, the compiled
plan, the reason the plan was not served.

THE REASON IS READ BY FAMILY, not looked up code by code. Every runtime names
its reasons in the same grammar (`MEASURE_NOT_SUPPORTED`, `PERIOD_NOT_AVAILABLE`,
`HISTORY_UNAVAILABLE`, …), so a reason a runtime adds tomorrow still gets a
true sentence rather than a generic one or none.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

#: The envelope's parser mode: a governed decline. Never mistaken for a
#: governed answer by the coverage gate, which reads `governed_plan`.
DECLINED_MODE = "governed_plan_declined"

# What kind of decline this is — the operator's classification, which
# `mi_service` maps onto its stable error codes. A decline is not a breakage,
# and the record has to be able to tell the two apart.
MODEL_UNAVAILABLE = "model_unavailable"   # the language step did not complete
CLARIFY = "clarify"                       # one more detail is needed
UNSUPPORTED = "unsupported"               # understood; not answerable yet
UNAVAILABLE = "unavailable"               # the data it needs could not be read
FAILED = "failed"                         # something broke producing it

_NOTHING_GUESSED = ("Nothing was guessed, and no other figure was put in its "
                    "place.")

#: Reasons that mean the answer could not be produced reliably. Checked
#: before the families: "RECONCILIATION_FAILED" is a breakage, not a period.
_FAILED_CODES = frozenset({
    "EXECUTION_FAILED", "RENDER_FAILED", "UNEXPECTED_ERROR",
    "PLAN_RECEIPT_RECONCILIATION_FAILED", "RECONCILIATION_FAILED",
    # The parts of a composed answer declared different data
    # (`plan_composition`): withheld, never shown side by side.
    "COMPOSED_FIGURES_NOT_ALIGNED"})

#: Reasons that mean the data the answer needs could not be read for this
#: request — true of the request, not of the question.
_UNAVAILABLE_CODES = frozenset({
    "TEMPORAL_STORE_UNAVAILABLE", "CHANGE_OWNER_INPUTS_UNAVAILABLE",
    "SOURCE_UNAVAILABLE", "HISTORY_UNAVAILABLE", "INPUTS_UNAVAILABLE",
    "FORECAST_UNAVAILABLE", "MOVEMENT_UNAVAILABLE", "SNAPSHOT_LOAD_FAILED",
    "NO_SNAPSHOTS", "BRIDGE_NOT_AVAILABLE"})

#: Reasons whose sentence is more specific than their family's.
_SPECIFIC: Mapping[str, str] = {
    "SCALE_NOT_CONFIGURED": "no securitisation stage is recorded for this "
                            "client, so 'scale' has no threshold to measure "
                            "against",
    "FIELD_NOT_IN_BOOK": "this book does not record the information it needs",
    "FILTER_VALUE_NOT_IN_BOOK": "this book does not record the value it "
                                "narrows to",
    # D21: not a failure to read — the figure is not stated until the history
    # it depends on is enough, so "try again" would be wrong.
    "RATE_NOT_MEASURED": "the client's history is not yet enough to measure "
                         "the stage rates it depends on, and no configured "
                         "rate is used in their place",
    "FIGURE_WITHHELD": "the client's history is not yet enough to measure "
                       "the stage rates it depends on, and no configured "
                       "rate is used in their place",
    "FIELD_UNAVAILABLE": "the figure it needs is not one this analysis "
                         "publishes",
    "PERIOD_NOT_A_PAIR": "a change needs two points in time, and the question "
                         "names one",
    "PERIOD_LABEL_UNRESOLVED": "I could not tell which period was meant",
    "PERIOD_LABEL_AMBIGUOUS": "the period named could mean more than one date",
    "PERIOD_RANGE_REVERSED": "the period asked for ends before it starts",
    "PERIOD_NOT_AVAILABLE": "the data for that period is not available",
    "INSUFFICIENT_SNAPSHOTS": "there are not enough snapshots to cover that "
                              "period",
    "EMPTY_ACROSS_EVERY_SNAPSHOT": "nothing matched it in any snapshot",
    "UNSUPPORTED_CADENCE": "the data is not kept at that frequency",
    "AMBIGUOUS_READING": "the question could mean more than one figure here",
    "POPULATION_VINTAGE_SKEW": "the data it combines is from dates too far "
                               "apart to combine",
    "NOT_SINGLE_OUTPUT": "it asks for more than one answer at once",
    "TOO_MANY_DIMENSIONS": "it asks for more breakdowns at once than I can "
                           "give",
    "EXPLICIT_LENS": "it names a portfolio view I cannot apply yet",
    "FIGURE_UNAVAILABLE": "the data does not carry a value for that figure "
                          "at this date",
    "EXECUTED_POPULATION_UNPROVEN": "I could not prove the figure would be "
                                    "measured on the part of the book asked "
                                    "about",
    "NO_MEASURE": "it does not name a figure I can produce",
}

#: The reason families, by the leading word of the code: what kind of thing
#: the question asked for that is not available.
_FAMILIES: Tuple[Tuple[str, str], ...] = (
    ("MEASURE", "that figure, or that combination of figures, is not one I "
                "can produce in a single answer yet"),
    ("DIMENSION", "that breakdown is not one I can provide for this figure "
                  "yet"),
    ("FILTER", "it narrows the figure in a way I cannot apply yet"),
    ("GEOGRAPHY", "that geography is not one this figure is published by"),
    ("OPERATION", "that kind of analysis is not one I can do for this figure "
                  "yet"),
    ("STATISTIC", "that statistic does not apply to this figure"),
    ("PERIOD", "the time period asked for is not one I can answer for yet"),
    ("POPULATION", "it asks about a part of the book this analysis does not "
                   "cover"),
    ("COMPARISON", "that comparison is not one I can make yet"),
    ("TARGET", "that comparison is not one I can make yet"),
    ("SCOPE", "it asks about a part of the book this analysis does not "
              "cover"),
    ("CAPABILITY", "that analysis is not available for this book yet"),
    ("FORM", "that kind of change question is not one I can answer yet"),
)

#: Why a question compiled to no plan, as a reader would be told.
_COMPILE_REASON: Mapping[str, str] = {
    "UNREGISTERED_CONCEPT": "it names something the book has no governed "
                            "definition for",
    "UNSUPPORTED_COMPOSITION": "that combination is not one I can answer",
    "UNSUPPORTED_OPERATION": "that kind of analysis is not one I can do for "
                             "this figure",
    "UNSUPPORTED_STATISTIC": "that statistic does not apply to this figure",
    "UNSUPPORTED_FILTER": "that restriction does not apply to this figure",
    "CAPABILITY_UNAVAILABLE": "that analysis is not available for this book",
    "CHANGE_FORM_NOT_CONNECTED": "that kind of change question is not one I "
                                 "can answer yet",
}

_COMPARATOR: Mapping[str, str] = {
    "eq": "is", "ne": "is not", "gt": "is above", "gte": "is at least",
    "lt": "is below", "lte": "is at most", "between": "is between",
    "in": "is one of", "not_in": "is not one of"}


def runtime_code(reason: str) -> str:
    """The runtime's own code inside the canary's reason string:
    `INELIGIBLE:MEASURE_NOT_SUPPORTED` -> `MEASURE_NOT_SUPPORTED`,
    `PLAN_RECEIPT_RECONCILIATION_FAILED:detail` -> the code before the detail."""
    code = str(reason or "")
    for prefix in ("INELIGIBLE:", "TEMPORAL_NOT_RESOLVED:"):
        if code.startswith(prefix):
            return code[len(prefix):]
    return code.split(":", 1)[0]


def kind(reason: str) -> str:
    """What kind of decline `reason` is, for the operator's record."""
    code = str(reason or "")
    if code.startswith("INTERPRETER"):
        return MODEL_UNAVAILABLE
    if code.startswith("CLARIFY"):
        return CLARIFY
    inner = runtime_code(code)
    if inner in _FAILED_CODES:
        return FAILED
    if inner in _UNAVAILABLE_CODES:
        return UNAVAILABLE
    return UNSUPPORTED


def plain_reason(code: str) -> str:
    """A runtime reason code in a reader's words."""
    if code in _SPECIFIC:
        return _SPECIFIC[code]
    head = code.split("_", 1)[0]
    for family, text in _FAMILIES:
        if head in (family, family + "S"):       # FILTER_… and FILTERS_…
            return text
    return "it is outside what I can answer yet"


def _labels() -> Mapping[str, str]:
    try:
        from mi_agent.interpretation_v2.vocabulary import load_governed_vocabulary
        return {cid: c.label for cid, c in
                load_governed_vocabulary().concepts.items() if c.label}
    except Exception:                                                # noqa: BLE001
        return {}


def _named(concept: Any, labels: Mapping[str, str]) -> str:
    text = str(concept or "")
    return (labels.get(text) or text.replace("_", " ")).lower()


def _value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(_value(v) for v in value)
    return str(value).replace("_", " ")


def understood(plan: Optional[Mapping[str, Any]]) -> str:
    """The compiled plan, in a reader's words: the figures, the population,
    any restriction and any breakdown. Empty when there is no plan."""
    if not plan:
        return ""
    labels = _labels()
    output = (plan.get("outputs") or [{}])[0] or {}
    measures = [_named(m.get("concept"), labels)
                for m in (output.get("measures") or ()) if m.get("concept")]
    base = str((plan.get("population") or {}).get("base") or "")
    parts: List[str] = []
    if measures:
        parts.append(" and ".join(measures))
    elif plan.get("operation"):
        parts.append(f"a {str(plan['operation']).replace('_', ' ')}")
    if base:
        parts.append({"funded": "for the funded book",
                      "pipeline": "for the pipeline",
                      "forecast": "for the forecast"}.get(base, f"for the {base}"))
    for f in list(plan.get("filters") or ()) + list(output.get("filters") or ()):
        comparator = _COMPARATOR.get(str(f.get("comparator") or "eq"), "is")
        parts.append(f"where {_named(f.get('concept'), labels)} {comparator} "
                     f"{_value(f.get('value'))}")
    for d in (output.get("dimensions") or ()):
        parts.append(f"by {_named(d.get('concept'), labels)}")
    return ", ".join(parts)


def _first_blocking(body: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    for a in ((body.get("interpretation") or {}).get("ambiguities") or ()):
        if isinstance(a, Mapping) and a.get("blocking"):
            return a
    return None


def message(body: Mapping[str, Any], reason: str) -> str:
    """The decline, as the reader is told it. Never carries a figure."""
    compiler = body.get("compiler") or {}
    reading = understood(compiler.get("plan"))
    code = str(reason or "")
    what = kind(code)

    if what == MODEL_UNAVAILABLE:
        return ("I could not complete the language-understanding step for this "
                "question, so I have not answered it. " + _NOTHING_GUESSED
                + " Please try again.")
    if what == CLARIFY:
        note = str((_first_blocking(body) or {}).get("note") or "").strip()
        return ("I need one more detail before I can answer"
                + (f": {note}" if note else ".") + " " + _NOTHING_GUESSED)
    if code.startswith("REFUSE"):
        why = [_COMPILE_REASON.get(str(r.get("code") or ""))
               for r in (compiler.get("reasons") or ()) if isinstance(r, Mapping)]
        why = [w for w in why if w]
        return ("I have not answered this: "
                + (why[0] if why else "it is outside what I can answer")
                + ". " + _NOTHING_GUESSED)
    absent = ((body.get("execution") or {}).get("filter_values_not_in_book")
              if runtime_code(code) == "FILTER_VALUE_NOT_IN_BOOK" else None)
    if absent:
        plain = "; ".join(_not_recorded(a) for a in absent)
    elif what == FAILED:
        plain = ("the figure could not be produced reliably for this request, "
                 "so it was withheld")
    elif what == UNAVAILABLE:
        plain = ("the data it needs could not be read for this request; please "
                 "try again")
    else:
        plain = plain_reason(runtime_code(code))
    # A QUESTION ASKING FOR SEVERAL FIGURES is answered for all of them or for
    # none (`plan_composition`), so the decline names the figure that could
    # not be produced.
    failed = str((body.get("composition") or {}).get("failed_figure") or "")
    if failed:
        plain = f"for the {_named(failed, _labels())}, {plain}"
    if reading:
        return (f"I understood this as {reading}, but I have not answered it: "
                f"{plain}. {_NOTHING_GUESSED}")
    return f"I have not answered this: {plain}. {_NOTHING_GUESSED}"


def _not_recorded(absent: Mapping[str, Any]) -> str:
    """A filter value the book does not record, said as a fact about the book,
    with the values it does record where they are a short category list."""
    label = str(absent.get("label") or absent.get("field") or "that field")
    asked = " or ".join(f"'{v}'" for v in absent.get("values") or ())
    book = absent.get("book_values")
    if book:
        return (f"this book records {label} as {', '.join(book)} — {asked} "
                f"is not one of them")
    return f"no loan in this book has {label} {asked}"


def envelope(*, question: str, body: Mapping[str, Any], reason: str,
             view: Optional[str]) -> Dict[str, Any]:
    """The declined answer, in the shape every channel already renders a
    refusal in (`mi_service._error_envelope`), with what was understood, the
    reason and its kind on the metadata for the audit."""
    text = message(body, reason)
    compiler = body.get("compiler") or {}
    return {
        "ok": False, "error": text, "question": question, "answer": text,
        "interpreted": "", "spec": {},
        "validation": {"ok": False, "errors": [text], "warnings": [],
                       "resolved_fields": {}},
        "artifacts": [], "warnings": [], "assumptions": [], "diagnostics": [],
        "sourceNotes": [],
        "metadata": {"engine": "mi_agent", "source": "python", "mock": False,
                     "datasetContext": view, "parserMode": DECLINED_MODE,
                     "governedDecline": {
                         "reason": str(reason or ""), "kind": kind(reason),
                         "understood": understood(compiler.get("plan")),
                         "plan_id": compiler.get("plan_id")}},
    }


def fallback_envelope(*, question: str, reason: str,
                      view: Optional[str]) -> Dict[str, Any]:
    """The decline when the evidence record itself cannot be read. Still a
    decline and still no figure: D18 holds even when the wording cannot be
    built from the record."""
    return envelope(question=question, body={}, reason=reason, view=view)
