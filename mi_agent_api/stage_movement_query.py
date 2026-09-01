"""mi_agent_api/stage_movement_query — MI Query as a CONSUMER of the governed
pipeline stage-transition capability.

WHAT THIS IS
------------
One recogniser and one handler, registered in the existing
``recogniser_registry`` like every other governed route. It reads a question for
a pipeline stage MOVEMENT, asks
:func:`mi_agent_api.movement_detail.resolve_stage_transition_detail` for the
governed answer, and renders the field it was asked for.

WHAT IT IS NOT
--------------
It is not a second interpreter, a second stage vocabulary, a second analytical
engine or a second plan. It computes NOTHING:

* it never loads a pipeline snapshot;
* it never joins two snapshots or matches a case;
* it never compares stages, counts arrivals, departures or stayers;
* it never derives an amount change or a reconciliation.

Every figure it publishes is a value selected out of the governed payload by
key. The only arithmetic performed here is *rendering* — turning a governed
float into a currency string — which is the same thing every other route does.

THE DEFECT THIS CLOSES
----------------------
Measured on ``tests/fixtures/pipeline_transition_2w`` at the starting SHA:

    "How many cases went from KFI into Application?"
        -> "3 loans · Current Outstanding Balance: £1.2MM.
            Calculated: Count of loans · Pipeline Stage = KFI · 3 loans."

Three is the CURRENT STOCK at KFI. The question asked how many cases MOVED, and
the governed answer is two. A stock figure standing in for a transition is the
silent substitution this capability exists to make unrepresentable, and it is
why recognition here claims the question rather than leaving it to a route whose
only pipeline figure is a stock.

RECOGNITION IS DELIBERATELY NARROW
----------------------------------
A question is a stage movement only when it names GOVERNED STAGES — through
``question_interpretation.lexical.pipeline_stage_vocabulary``, the estate's one
question-side stage vocabulary, derived from ``pipeline_prep._STAGE_CANON`` —
AND puts them in an explicit movement construction. The bare word "movement" is
never the discriminator: it is already owned by the funded bridge, period change
and period movement, and every one of those keeps it.

The route registers LAST, at the highest priority number and on
``DEFAULT_CONFIDENCE``, so any existing recogniser that also matches wins by
construction rather than by a rule written here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

#: The pre-claim reading, handed from ``recognise`` to ``handle`` through the
#: registry's existing memo rather than by reading the sentence twice.
RECOGNITION_KEY = "stage_movement"

#: Stable route id, published as ``metadata.route``.
ROUTE_NAME = "pipeline_stage_movement"

# --------------------------------------------------------------------------- #
# Movement subtypes — one per governed payload section, and nothing else.
# --------------------------------------------------------------------------- #
TRANSITION = "transition"
NEW_ARRIVAL = "new_arrival"
STAYER = "stayer"
DEPARTURE = "departure"
RECONCILIATION = "reconciliation"

#: What was asked FOR: a case count, a monetary amount, or the amount amendment
#: on cases that did not move.
MEASURE_COUNT = "count"
MEASURE_AMOUNT = "amount"
MEASURE_AMOUNT_CHANGE = "amount_change"


# --------------------------------------------------------------------------- #
# Vocabulary
#
# Every stage spelling comes from the governed vocabulary. Only the MOVEMENT
# words live here, because no existing governed structure carries them and
# inventing a stage table would be the duplicate taxonomy §9 forbids.
# --------------------------------------------------------------------------- #
#: Verbs that assert a case CHANGED STAGE. "Movement" and "change" are absent on
#: purpose — both are owned by other routes.
_TRANSITION_VERBS = ("moved", "move", "moves", "moving", "progressed",
                     "progress", "progresses", "progressing", "transitioned",
                     "transition", "transitions", "transitioning", "went",
                     "go", "goes", "going", "advanced", "advance", "advances",
                     "advancing", "migrated", "migrate", "migrates")

#: Connectors that make a pair of stages DIRECTIONAL. The source is whichever
#: stage the sentence puts before the connector.
_CONNECTORS = ("->", "→", "to", "into", "in to", "onto", "on to", "through to",
               "reached", "reaching", "reach")

_ARRIVAL_WORDS = ("new arrival", "new arrivals", "newly entered", "newly arrived",
                  "entered", "enter", "enters", "entering", "arrived", "arrive",
                  "arrives", "arriving", "new case", "new cases",
                  "new pipeline case", "new pipeline cases")

_STAYER_WORDS = ("stayed", "stay", "stays", "staying", "stayer", "stayers",
                 "remained", "remain", "remains", "remaining", "persisted",
                 "persist", "persists", "persisting", "unchanged stage")

_DEPARTURE_WORDS = ("leaving", "left", "leave", "leaves", "departed", "depart",
                    "departs", "departing", "departure", "departures", "exited",
                    "exit", "exits", "exiting")

_RECONCILIATION_WORDS = ("reconcile", "reconciles", "reconciled",
                         "reconciliation", "opening to closing")

#: Words asking for MONEY rather than a case count.
_AMOUNT_WORDS = ("balance", "amount", "value", "amounts", "values", "£", "sum",
                 "worth", "money")

#: Words asking for a COUNT. "How many" is the strongest and is handled apart.
_COUNT_WORDS = ("how many", "number of", "case count", "count of", "count the")

#: THE DEFERENCES. A sentence carrying any of these belongs to a route that
#: already owns it, and this recogniser declines regardless of what else it
#: sees. Conversion is the cohort route's; forecast, expectation and projection
#: belong to the forecast and analytical-composition routes; a named trend or
#: series is evolution's.
_DEFER_TERMS = ("conversion", "convert", "converts", "converted", "converting",
                "forecast", "forecasts", "forecasting", "projected", "project",
                "projection", "expected", "expect", "expects", "expectation",
                "run rate", "run-rate", "scenario", "what if", "what-if",
                "trend", "over time", "by week", "by month", "weekly",
                "monthly", "evolution", "each week", "per week", "seasonal",
                "cohort", "vintage", "funnel")


def _norm(question: Optional[str]) -> str:
    return " %s " % re.sub(r"\s+", " ", str(question or "").strip().lower())


def _has(text: str, terms: Tuple[str, ...]) -> bool:
    return any(re.search(r"(?<![a-z])%s(?![a-z])" % re.escape(t), text)
               for t in terms)


# --------------------------------------------------------------------------- #
# The reading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StageMovement:
    """What the sentence asked for, in the governed payload's own terms."""

    subtype: str
    measure: str
    source: Optional[str] = None
    destination: Optional[str] = None
    stage: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"subtype": self.subtype, "measure": self.measure,
                "source": self.source, "destination": self.destination,
                "stage": self.stage}


def _stage_mentions(text: str) -> List[Tuple[int, str]]:
    """``(position, canonical stage)`` for every governed stage the sentence names.

    THE GOVERNED VOCABULARY AND NOTHING ELSE.
    ``question_interpretation.lexical.pipeline_stage_vocabulary`` is derived from
    ``mi_agent_api.pipeline_prep._STAGE_CANON`` and already drops the spellings
    that collide with a governed dataset name ("funded"), which is exactly the
    adjustment a question-side reader needs. No second table exists here.

    Longest spelling first at each position, so "offer issued" is not read as
    "offer", and de-duplicated by position so one span yields one stage.
    """
    from question_interpretation.lexical import pipeline_stage_vocabulary

    vocab = pipeline_stage_vocabulary()
    spans: List[Tuple[int, int, str]] = []
    for spelling in sorted(vocab, key=len, reverse=True):
        for match in re.finditer(r"(?<![a-z])%s(?:e?s)?(?![a-z])"
                                 % re.escape(spelling), text):
            start, end = match.span()
            if any(s < end and start < e for s, e, _ in spans):
                continue  # inside a longer spelling already claimed
            spans.append((start, end, vocab[spelling]))
    return [(s, stage) for s, _, stage in sorted(spans)]


def _directional_pair(text: str, mentions: List[Tuple[int, str]]
                      ) -> Optional[Tuple[str, str]]:
    """``(source, destination)`` where the sentence puts two stages in order.

    Two constructions, both explicit:

      A. a CONNECTOR between two distinct stages — "KFI to Application",
         "Application into Offer", "Offer -> Completed", "offers reached
         Completion". The source is the stage before the connector.
      B. a TRANSITION VERB anywhere, with exactly two distinct stages named —
         "cases moved from KFI to Application" already matches (A); this covers
         "which cases progressed, KFI and Application" style orderings where the
         connector sits elsewhere. Funnel order decides, because a case cannot
         move backwards through the governed funnel in this construction.

    Returns ``None`` unless two DISTINCT governed stages are named. One stage is
    a stock question and belongs to its existing owner.
    """
    distinct = []
    for _, stage in mentions:
        if stage not in distinct:
            distinct.append(stage)
    if len(distinct) != 2:
        return None

    first_pos = {}
    for pos, stage in mentions:
        first_pos.setdefault(stage, pos)

    a, b = sorted(distinct, key=lambda s: first_pos[s])
    between = text[first_pos[a]:first_pos[b]]
    if _has(between, _CONNECTORS) or "->" in between or "→" in between:
        return a, b
    if _has(text, _TRANSITION_VERBS):
        from question_interpretation.lexical import canonical_pipeline_stages

        order = {s: i for i, s in enumerate(canonical_pipeline_stages())}
        src, dst = sorted(distinct, key=lambda s: order.get(s, 99))
        return src, dst
    return None


def _measure_for(text: str, subtype: str) -> str:
    """Count, amount, or the governed amount amendment — read from the sentence."""
    if subtype == STAYER and (_has(text, ("amount change", "amendment"))
                              or re.search(r"\bchang\w*\b", text)):
        return MEASURE_AMOUNT_CHANGE
    if any(w in text for w in _COUNT_WORDS):
        return MEASURE_COUNT
    if _has(text, _AMOUNT_WORDS) or " how much " in text:
        return MEASURE_AMOUNT
    return MEASURE_COUNT


def read(question: Optional[str]) -> Optional[StageMovement]:
    """The sentence as a stage-movement request, or ``None``.

    THE ONE READER. ``recognise`` calls it and stores the result; ``handle``
    takes the stored reading and never looks at the sentence again.
    """
    text = _norm(question)
    if not text.strip():
        return None
    if _has(text, _DEFER_TERMS):
        return None

    mentions = _stage_mentions(text)
    if not mentions:
        return None
    stages = []
    for _, stage in mentions:
        if stage not in stages:
            stages.append(stage)

    # 1. A directional pair is a TRANSITION, and outranks everything else: the
    #    sentence named both ends of a movement.
    pair = _directional_pair(text, mentions)
    if pair is not None:
        src, dst = pair
        return StageMovement(subtype=TRANSITION,
                             measure=_measure_for(text, TRANSITION),
                             source=src, destination=dst)

    # Every remaining subtype is about ONE stage. Two stages with no directional
    # construction is not a movement this capability can bind, so it declines
    # rather than guessing which one the reader meant.
    if len(stages) != 1:
        return None
    stage = stages[0]

    if _has(text, _RECONCILIATION_WORDS) or (
            _has(text, ("opening",)) and _has(text, ("closing",))):
        return StageMovement(subtype=RECONCILIATION, measure=MEASURE_COUNT,
                             stage=stage)
    if _has(text, _DEPARTURE_WORDS):
        return StageMovement(subtype=DEPARTURE,
                             measure=_measure_for(text, DEPARTURE), stage=stage)
    if _has(text, _STAYER_WORDS):
        return StageMovement(subtype=STAYER,
                             measure=_measure_for(text, STAYER), stage=stage)
    if _has(text, _ARRIVAL_WORDS) and (" new " in text or _has(
            text, ("entered", "arrived", "arrivals", "arrival", "arriving",
                   "entering"))):
        return StageMovement(subtype=NEW_ARRIVAL,
                             measure=_measure_for(text, NEW_ARRIVAL),
                             destination=stage)
    return None


# --------------------------------------------------------------------------- #
# Delegation — the governed capability answers; this selects and renders.
# --------------------------------------------------------------------------- #
def _label(stage: Optional[str]) -> str:
    """A governed canonical stage in reader's English, not in payload spelling."""
    return {"KFI": "KFI", "APPLICATION": "Application", "OFFER": "Offer",
            "COMPLETED": "Completion", "WITHDRAWN": "Withdrawn",
            }.get(str(stage or ""), str(stage or "").title())


def _window(payload: Dict[str, Any]) -> str:
    """The governed reporting window, exactly as the capability reports it."""
    latest, prior = payload.get("as_of_date"), payload.get("comparison_date")
    if latest and prior:
        return "between %s and %s" % (prior, latest)
    return "in the latest comparison window"


def _cases(n: int) -> str:
    return "%d case%s" % (int(n), "" if int(n) == 1 else "s")


def _transition_row(payload: Dict[str, Any], src: str, dst: str
                    ) -> Optional[Dict[str, Any]]:
    """The governed ``source -> destination`` row, by key lookup. No arithmetic."""
    for row in payload.get("transitions") or []:
        if row.get("source_stage") == src and row.get("destination_stage") == dst:
            return row
    return None


def _stage_is_governed(payload: Dict[str, Any], stage: str) -> bool:
    """Is this stage one the governed reconciliation reports for this window?

    Read from the payload's own per-stage reconciliation, which covers every
    stage present in either snapshot. A stage the window does not carry is
    refused rather than answered with a zero, because "no cases moved" and "that
    stage is not in this pipeline" are different statements.
    """
    rows = ((payload.get("reconciliation") or {}).get("by_stage")) or []
    return any(r.get("stage") == stage for r in rows)


def _reconciliation_row(payload: Dict[str, Any], stage: str
                        ) -> Optional[Dict[str, Any]]:
    for row in ((payload.get("reconciliation") or {}).get("by_stage")) or []:
        if row.get("stage") == stage:
            return row
    return None


def compose(reading: StageMovement, payload: Dict[str, Any], *,
            money: Any) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[str]]:
    """``(answer, rows, refusal)`` for one reading against one governed payload.

    ``money`` renders a governed float in the request's resolved currency; it is
    the caller's formatter, so this module owns no currency policy.

    EVERY FIGURE IS A KEY LOOKUP. Where the governed payload has no row for what
    was asked, this returns a refusal — never a zero, and never a stock.
    """
    if not payload.get("available"):
        return None, [], (payload.get("reason") or
                          "The governed pipeline stage-transition analysis is "
                          "not available for this book.")
    window = _window(payload)

    if reading.subtype == TRANSITION:
        src, dst = reading.source, reading.destination
        for stage in (src, dst):
            if not _stage_is_governed(payload, stage):
                return None, [], (
                    "%s is not a stage the governed pipeline carries %s, so I "
                    "cannot report movement %s it." % (
                        _label(stage), window,
                        "from" if stage == src else "into"))
        row = _transition_row(payload, src, dst)
        if row is None:
            return ("No cases moved from %s to %s %s."
                    % (_label(src), _label(dst), window)), [], None
        if reading.measure == MEASURE_AMOUNT:
            prior, latest = row["prior_amount"], row["latest_amount"]
            # The amendment is stated only where it is VISIBLE at the precision
            # the reader is given. £1,300,000 and £1,290,000 both render as
            # £1.3m, and "carried at £1.3m … and at £1.3m" reads as an error.
            if money(prior) == money(latest):
                answer = ("%s moved from %s to %s %s, across %s."
                          % (money(latest), _label(src), _label(dst), window,
                             _cases(row["case_count"])))
            else:
                answer = (
                    "%s moved from %s to %s %s, across %s. Those cases were "
                    "carried at %s at %s in the prior extract and at %s at %s "
                    "in the latest one."
                    % (money(latest), _label(src), _label(dst), window,
                       _cases(row["case_count"]), money(prior), _label(src),
                       money(latest), _label(dst)))
        else:
            answer = ("%s moved from %s to %s %s."
                      % (_cases(row["case_count"]), _label(src), _label(dst),
                         window))
        return answer, [dict(row)], None

    if reading.subtype == NEW_ARRIVAL:
        stage = reading.destination
        if not _stage_is_governed(payload, stage):
            return None, [], ("%s is not a stage the governed pipeline carries "
                              "%s." % (_label(stage), window))
        row = next((r for r in payload.get("new_arrivals") or []
                    if r.get("destination_stage") == stage), None)
        if row is None:
            return ("No new cases entered %s %s." % (_label(stage), window)), [], None
        if reading.measure == MEASURE_AMOUNT:
            answer = ("%s of new pipeline entered %s %s, across %s."
                      % (money(row["latest_amount"]), _label(stage), window,
                         _cases(row["case_count"])))
        else:
            answer = ("%s newly entered %s %s, carrying %s."
                      % (_cases(row["case_count"]), _label(stage), window,
                         money(row["latest_amount"])))
        return answer, [dict(row)], None

    if reading.subtype == STAYER:
        stage = reading.stage
        if not _stage_is_governed(payload, stage):
            return None, [], ("%s is not a stage the governed pipeline carries "
                              "%s." % (_label(stage), window))
        row = next((r for r in payload.get("stayers") or []
                    if r.get("stage") == stage), None)
        if row is None:
            return ("No cases stayed at %s %s." % (_label(stage), window)), [], None
        if reading.measure == MEASURE_AMOUNT_CHANGE:
            change = float(row["amount_change"])
            direction = "up" if change > 0 else ("down" if change < 0 else "flat")
            answer = ("%s that stayed at %s changed in value by %s (%s), from "
                      "%s to %s %s."
                      % (_cases(row["case_count"]), _label(stage),
                         money(abs(change)), direction, money(row["prior_amount"]),
                         money(row["latest_amount"]), window))
        elif reading.measure == MEASURE_AMOUNT:
            answer = ("%s stayed at %s %s, carrying %s in the latest extract."
                      % (_cases(row["case_count"]), _label(stage), window,
                         money(row["latest_amount"])))
        else:
            answer = ("%s stayed at %s %s."
                      % (_cases(row["case_count"]), _label(stage), window))
        return answer, [dict(row)], None

    if reading.subtype == DEPARTURE:
        stage = reading.stage
        if not _stage_is_governed(payload, stage):
            return None, [], ("%s is not a stage the governed pipeline carries "
                              "%s." % (_label(stage), window))
        rows = [dict(r) for r in payload.get("departures") or []
                if r.get("source_stage") == stage]
        moved = [dict(r) for r in payload.get("transitions") or []
                 if r.get("source_stage") == stage]
        if not rows and not moved:
            return ("No cases left %s %s." % (_label(stage), window)), [], None
        parts = ["%s moved on to %s" % (_cases(r["case_count"]),
                                        _label(r["destination_stage"]))
                 for r in moved]
        for r in rows:
            outcome = r.get("governed_outcome")
            if r.get("outcome_evidence") == "none" or not outcome or \
                    str(outcome).startswith("unclassified"):
                parts.append("%s left the governed extracts with no outcome the "
                             "data evidences" % _cases(r["case_count"]))
            else:
                parts.append("%s left recorded as %s"
                             % (_cases(r["case_count"]), _label(outcome)))
        answer = ("Of the cases that left %s %s: %s."
                  % (_label(stage), window, "; ".join(parts)))
        return answer, moved + rows, None

    if reading.subtype == RECONCILIATION:
        stage = reading.stage
        row = _reconciliation_row(payload, stage)
        if row is None:
            return None, [], ("%s is not a stage the governed pipeline carries "
                              "%s." % (_label(stage), window))
        answer = (
            "%s stage reconciles %s: opening %s, plus %s newly arrived and %s "
            "transferred in, less %s transferred out and %s departed, giving a "
            "closing %s. Opening balance %s, closing balance %s."
            % (_label(stage), window, _cases(row["opening_case_count"]),
               _cases(row["new_arrivals"]), _cases(row["transitions_in"]),
               _cases(row["transitions_out"]), _cases(row["departures"]),
               _cases(row["closing_case_count"]), money(row["opening_amount"]),
               money(row["closing_amount"])))
        return answer, [dict(row)], None

    return None, [], "This stage-movement question could not be bound."


# --------------------------------------------------------------------------- #
# Registry surface — recognise, then handle. Two functions, one reading.
# --------------------------------------------------------------------------- #
def recognise(request: Any) -> Any:
    """Does this request ask about pipeline stage movement?

    Pure and side-effect-free apart from storing its own reading on the request,
    which is the registry's existing pre-claim memo (``remember_recognition``) —
    the same seam ``period_change`` uses so a handler never re-reads the sentence.
    """
    from .recogniser_registry import Recognition

    reading = read(getattr(request, "question", ""))
    if reading is None:
        return Recognition.no("no governed stage movement construction")
    request.remember_recognition(RECOGNITION_KEY, reading)
    return Recognition.yes(reason="%s %s" % (reading.subtype, reading.measure))


def handle(request: Any) -> Optional[Dict[str, Any]]:
    """Ask the governed capability, and render what it returns.

    Returns ``None`` only where this route cannot claim the question at all —
    no reading, or no governed pipeline root — so the request falls through to
    exactly the behaviour it had before, as the registry contract requires.
    """
    from . import currency as currency_mod
    from . import movement_detail as detail_mod
    from .chat_routing import _envelope, _source, _table_artifact, _undeliverable

    reading = request.recalled_recognition(RECOGNITION_KEY) or read(request.question)
    if reading is None:
        return None
    root = request.pipeline_root
    if not root:
        return None

    spec_dict, mangled = account_for_mangled_spans(dict(request.spec_dict or {}))

    # THE DELEGATION. The same governed resolver, the same governed extracts and
    # the same neighbour rule the React movement-detail endpoint and the PPTX
    # deck already consume. Query computes no part of this.
    #
    # NO ``as_of``. The request's as-of is the FUNDED reporting cut-off
    # (30 June 2026); the resolver's is a WEEKLY PIPELINE EXTRACT DATE
    # (12 June 2026). They are different axes, and passing one for the other
    # made ``select_pair`` match no extract and the capability report "no
    # governed weekly pipeline extract matches this point" for every question.
    # Omitting it asks for the capability's own latest governed pair — the same
    # window the movement-detail endpoint uses with no chart point hovered — and
    # every answer below STATES that window, so nothing is implied about a
    # window the reader did not get.
    payload = detail_mod.resolve_stage_transition_detail(
        root, request.client_id,
        historical_model=request.resolve_history_model())

    def money(value: Any) -> str:
        return currency_mod.format_money(value, suffixes=("bn", "m", "k"))

    answer, rows, refusal = compose(reading, payload, money=money)
    notes = [_source("Governed pipeline stage transitions", spec_dict,
                     request.portfolio_id, payload.get("as_of_date"),
                     engine="mi_agent_api.movement_detail."
                            "resolve_stage_transition_detail")]
    if refusal is not None or answer is None:
        return _undeliverable(
            question=request.question, spec=spec_dict,
            answer=refusal or "This stage-movement analysis could not be produced.",
            route=ROUTE_NAME, source_notes=notes)

    artifacts: List[Dict[str, Any]] = []
    if rows:
        artifacts.append(_table_artifact(
            "%s — governed stage movement" % _movement_title(reading),
            columns=_columns(rows), rows=rows, spec=spec_dict,
            portfolio_id=request.portfolio_id,
            as_of=payload.get("as_of_date")))
    envelope = _envelope(
        ok=True, question=request.question, answer=answer, spec=spec_dict,
        artifacts=artifacts, route=ROUTE_NAME, source_notes=notes,
        # THE DATASET THIS ANSWER WAS RECONCILED AGAINST. Every figure came from
        # the governed weekly pipeline extracts, and the coverage ledger reads
        # this field — not the route name — to decide whether a question that
        # said "pipeline" was answered from the pipeline.
        reconciliation={"dataset": "pipeline", "coverage_by_balance_pct": 100.0},
        lens_applied=False)
    # THE NARROWING THIS ANSWER APPLIED, declared through the population ledger
    # every other route declares through. The answer is about named governed
    # stages and nothing else, so a reader — and the execution receipt — must be
    # able to see that the pipeline WAS narrowed to them rather than infer it
    # from the prose. Execution evidence only: the stages named here are the
    # ones the governed payload was selected on.
    from question_interpretation.lexical import PIPELINE_STAGE_FIELD

    stages = [s for s in (reading.source, reading.destination, reading.stage) if s]
    envelope.setdefault("metadata", {})["populationApplied"] = {
        "applied": ["%s (%s %s)" % (PIPELINE_STAGE_FIELD,
                                    reading.subtype.replace("_", " "),
                                    " to ".join(_label(s) for s in stages))],
        "unavailable": [], "rowsBefore": None, "rowsAfter": None,
    }
    envelope["metadata"]["stageMovement"] = {
        **reading.to_dict(),
        "accountedForSpans": mangled,
        "asOfDate": payload.get("as_of_date"),
        "comparisonDate": payload.get("comparison_date"),
        "identifier": payload.get("identifier"),
        "methodologyVersion": (payload.get("methodology") or {}).get("version"),
    }
    return envelope


#: Residuals are the capability's own self-proof, not a reader's column.
_HIDDEN_COLUMNS = ("count_reconciliation_residual",
                   "amount_reconciliation_residual")


def _columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Table columns for whichever governed keys these rows actually carry.

    The label and the format are DERIVED from the key rather than listed, so a
    field the capability adds later appears correctly with no change here — and
    no raw payload key reaches a reader as a column label.
    """
    keys: List[str] = []
    for row in rows:
        keys.extend(k for k in row if k not in keys and k not in _HIDDEN_COLUMNS)
    out = []
    for key in keys:
        money_col = key.endswith("_amount") or key.endswith("amount_change")
        counted = key.endswith("_count") or key in (
            "new_arrivals", "transitions_in", "transitions_out", "departures",
            "stayers")
        fmt = "gbp" if money_col else ("number" if counted else "text")
        out.append({"key": key, "label": key.replace("_", " ").capitalize(),
                    "align": "left" if fmt == "text" else "right",
                    "format": fmt})
    return out


def _construction_words() -> set:
    """Every word this route's own movement construction is built from.

    The governed stage spellings plus this module's movement vocabulary — one
    set, derived from the same tuples recognition uses, so it cannot drift from
    what was actually recognised.
    """
    from question_interpretation.lexical import pipeline_stage_vocabulary

    words = set()
    for spelling in pipeline_stage_vocabulary():
        words.update(re.findall(r"[a-z]+", spelling.lower()))
    for group in (_TRANSITION_VERBS, _CONNECTORS, _ARRIVAL_WORDS, _STAYER_WORDS,
                  _DEPARTURE_WORDS, _RECONCILIATION_WORDS):
        for term in group:
            words.update(re.findall(r"[a-z]+", term.lower()))
    # Grammatical filler that binds the construction and names nothing.
    words.update({"from", "the", "a", "an", "of", "at", "in", "stage", "stages",
                  "case", "cases", "pipeline"})
    return words


def account_for_mangled_spans(spec_dict: Dict[str, Any]) -> Tuple[Dict[str, Any],
                                                                  List[str]]:
    """Drop unresolved-category notes this route's construction fully accounts for.

    THE SPAN WAS NEVER A CATEGORY. Reading "How many cases moved from Offer to
    Completion?", the parser proposes ``offer to completion`` as a categorical
    value, finds no such value in the book, and records
    ``unknown category: 'offer to completion'``. The routed guard then refuses
    with *"No loans in this book match that filter ('offer to completion')"* —
    a statement about the CLIENT'S DATA that is not true, about a filter the
    reader never asked for. ``migration_phase0/data_claim_audit.py`` classifies
    exactly this shape as ``QUOTES_A_MANGLED_PHRASE``.

    This route bound that span as a governed source and destination stage and
    answered from it, so the span is accounted for and the note is stale. A note
    is dropped ONLY when every alphabetic word in it belongs to the construction
    this route recognised — the same governed span-ownership rule
    ``RouteRequest.for_recognition`` applies to recognition. Anything the reader
    said that this route does NOT own (a broker, a region, a product) keeps its
    note and still refuses.

    Returns ``(spec, dropped)``; the dropped notes are published as evidence
    rather than disappearing.
    """
    notes = list(spec_dict.get("unavailable_filters") or ())
    if not notes:
        return spec_dict, []
    from mi_agent.llm_query_parser import UNKNOWN_CATEGORY_PREFIX

    owned = _construction_words()
    kept, dropped = [], []
    for note in notes:
        text = str(note)
        if not text.startswith(UNKNOWN_CATEGORY_PREFIX):
            kept.append(note)
            continue
        name = text[len(UNKNOWN_CATEGORY_PREFIX):].strip().strip("'\"")
        words = re.findall(r"[a-z]+", name.lower())
        if words and all(w in owned for w in words):
            dropped.append(text)
        else:
            kept.append(note)
    if not dropped:
        return spec_dict, []
    out = dict(spec_dict)
    out["unavailable_filters"] = kept
    return out, dropped


def _movement_title(reading: StageMovement) -> str:
    if reading.subtype == TRANSITION:
        return "%s to %s" % (_label(reading.source), _label(reading.destination))
    if reading.subtype == NEW_ARRIVAL:
        return "New arrivals into %s" % _label(reading.destination)
    if reading.subtype == STAYER:
        return "Cases staying at %s" % _label(reading.stage)
    if reading.subtype == DEPARTURE:
        return "Departures from %s" % _label(reading.stage)
    return "%s stage reconciliation" % _label(reading.stage)


def recogniser():
    """This capability's registry entry.

    Priority 120 — AFTER every existing recogniser — and on
    ``DEFAULT_CONFIDENCE``, so route ownership is settled by the registry's own
    deterministic ordering and any existing owner that also matches keeps the
    question. Deference is structural here, not a rule this module asserts.
    """
    from .recogniser_registry import Recogniser

    return Recogniser(
        name=ROUTE_NAME, priority=120, lens_aware=False,
        description=("Gross case-level pipeline stage movement between the two "
                     "latest governed weekly extracts."),
        metadata={
            "governed_capability":
                "mi_agent_api.movement_detail.resolve_stage_transition_detail",
            "stage_vocabulary":
                "question_interpretation.lexical.pipeline_stage_vocabulary",
            "computes_nothing": True,
            "subtypes": (TRANSITION, NEW_ARRIVAL, STAYER, DEPARTURE,
                         RECONCILIATION),
        },
        recognise=recognise, handle=handle)
