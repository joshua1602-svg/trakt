"""mi_agent_api/borrowing_base_query.py — borrowing-base questions on /mi/query.

THE ONE MI INTEGRATION MODULE for the governed borrowing base. It registers a
recogniser with the capability registry and answers from the SAME governed
``borrowingBase`` envelope the React Eligibility & Concentrations tab and the
standalone ``GET /mi/borrowing-base`` route consume:

    /mi/query
        ↓  recognise()   — explicit borrowing-base / facility / ineligibility
        ↓                  vocabulary only; bare "headroom" or "utilisation"
        ↓                  stays with its existing owner
        ↓  handle()
        ↓
    borrowing_base_api.compute_borrowing_base   (current position — the block
                                                 the dashboard renders)
    borrowing_base_api.compute_from_frames      (one call per governed period,
                                                 for change / trend / bridge)
    mi_agent.borrowing_base.analysis            (reason summary, period
                                                 validity, bridge composition)
        ↓
    select the requested governed measure(s) → answer / existing artifacts

Nothing here calculates a borrowing base, derives eligibility, or reads a loan
to decide anything. It selects, formats and discloses.

REFUSALS ARE GOVERNED, NEVER SILENT. Once borrowing-base intent is claimed, any
facet the v1 surface cannot honour — a grouping ("by region"), a narrowing
("for Scotland", a portfolio lens, a drill-through filter), a loan-level
listing, a forecast, a haircut or reserve or concentration deduction — refuses
by name rather than answering the unnarrowed question and dropping the facet.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mi_agent.borrowing_base import analysis
from mi_agent.borrowing_base.models import NOT_CALCULABLE
from mi_agent.borrowing_base.service import MEASURES

from .recogniser_registry import Recognition, Recogniser, RouteRequest

logger = logging.getLogger("mi_agent_api.borrowing_base_query")

ROUTE_NAME = "borrowing_base"
#: Between period_change (85) / temporal_compare (90) and risk_limits (100).
#: Position is documentary: the confidence below decides the arbitration.
ROUTE_PRIORITY = 95
#: Above the analytical layer (0.8) and the workflows (0.7): a question that
#: names the borrowing base explicitly is not a composite of other
#: capabilities and must not be answered in part by one. The vocabulary gate
#: in :func:`recognise` is what makes that confidence safe.
ROUTE_CONFIDENCE = 0.85

#: How many governed periods a trend or bridge may reach back.
MAX_PERIODS = 24

# --------------------------------------------------------------------------- #
# Vocabulary — explicit, closed, and read only against the (span-masked)
# question the registry hands to recognition.
# --------------------------------------------------------------------------- #
_BB_CORE_RE = re.compile(r"\bborrowing base\b")
_FACILITY_RE = re.compile(
    r"\bfacility (?:utili[sz]ation|drawn|drawings?|drawing|commitment|headroom|"
    r"position|drawdown)\b"
    r"|\b(?:drawn|drawings?|drawdown|utili[sz]ation|commitment|headroom) "
    r"(?:under|on|against|of) (?:the |our )?(?:funding |warehouse )?facility\b"
    r"|\b(?:funding|warehouse) facility\b"
    r"|\bfacility (?:is|are) drawn\b")
_ELIGIBLE_RE = re.compile(
    r"\bineligib\w*\b|\beligible collateral\b"
    r"|\b(?<!in)eligible (?:current |loan |mortgage loan )?balance\b"
    r"|\b(?<!in)eligible mortgage loans?\b|\beligibility reasons?\b")
#: Words that mean a limit question is a CONCENTRATION question. Without the
#: core vocabulary above they are never claimed here.
_CONCENTRATION_RE = re.compile(r"\bconcentration\b|\bschedule 8\b|\blimit tests?\b")

_FUNDED_VIEWS = ("funded", "", "mi")

#: Concepts the v1 surface does not own. Claimed (the question IS about the
#: borrowing base) and then refused by name, so no other route substitutes.
_UNSUPPORTED: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bhaircut"), "haircut effects"),
    (re.compile(r"\breserves?\b"), "reserve effects"),
    (re.compile(r"\bover ?collateral"), "overcollateralisation"),
    (re.compile(r"\bcovenant"), "covenant analysis"),
    (re.compile(r"\bforecast|\bproject(?:ed|ion)s?\b|\bnext (?:month|quarter|"
                r"year)\b|\bscenario|\bwhat if\b|\boutlook\b|\bexpected to\b|"
                r"\bwill (?:the|we|it|our)\b|\brun[\s-]?rate\b"), "forecasting"),
    (re.compile(r"\boptimi[sz]"), "facility optimisation"),
    (re.compile(r"\baverage\b|\bmedian\b|\bweighted\b|\bdistribution\b|\bmean\b"),
     "a statistic over the borrowing base (each measure is one governed figure)"),
    (re.compile(r"\ballocat"), "multi-facility allocation"),
    (re.compile(r"\bwhich loans\b|\blist (?:the |of |all )?loans\b|\bshow (?:me )?"
                r"(?:the |all )?loans\b|\bloan[\s-]level\b|\bloan by loan\b"),
     "loan-level listing (the Eligibility & Concentrations tab's drill-down "
     "shows the loans behind each status)"),
    (re.compile(r"\bconcentration\b|\bschedule 8\b"),
     "concentration-limit deductions (the borrowing base deducts nothing for "
     "a concentration breach; ask the concentration-limit question directly)"),
)
#: A grouping the question asked for that v1 cannot honour. "by reason" and
#: the time axis are the two groupings this surface owns.
_BY_RE = re.compile(
    r"\bby (?!reasons?\b|ineligibility\b|month\b|months\b|period\b|periods\b|"
    r"week\b|quarter\b|year\b|reporting\b|the\b|how\b|what\b|balance\b|"
    r"count\b|number\b|loan\b|loans\b)([a-z][a-z_ -]{1,30}?)(?=[\s?.,;]|$)")

_REASON_RE = re.compile(
    r"\breasons?\b|\bwhy (?:are|is|do|does|were|was)\b.*\bineligib"
    r"|\bbreak ?down\b|\bcauses? of\b|\bwhat makes\b.*\bineligib")
_BRIDGE_RE = re.compile(
    r"\bbridge\b|\bwaterfall\b|\bwhat drove\b|\bdrivers?\b|\bdecompos"
    r"|\bwhy (?:did|has|have|is|was)\b.*\b(?:chang|mov|fall|fell|ris|rose|"
    r"increas|decreas|grow|grew|shr)\w*"
    r"|\bexplain (?:the )?(?:change|movement)\b|\bmovement\b|\battribut")
_TREND_RE = re.compile(
    r"\bover time\b|\btrend\b|\bby month\b|\bmonthly\b|\beach month\b"
    r"|\bevery month\b|\btime series\b|\bhistory\b|\bhistorical(?:ly)?\b"
    r"|\bby period\b|\bby reporting\b|\bover the (?:last|past) \d+ months?\b"
    r"|\bover the months\b|\bmonth by month\b")
_TABLE_RE = re.compile(r"\bas a table\b|\bin a table\b|\btabular\b|\btable\b")
_WATERFALL_RE = re.compile(r"\bwaterfall\b")

#: Measure vocabulary, most specific first. Each match is CONSUMED from the
#: text so "borrowing base headroom" cannot also read as "borrowing base".
_MEASURE_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bgross borrowing base\b"), "gross_borrowing_base"),
    (re.compile(r"\bborrowing base utili[sz]ation\b"
                r"|\butili[sz]ation (?:of|under|against) the borrowing base\b"),
     "borrowing_base_utilisation"),
    (re.compile(r"\bfacility utili[sz]ation\b"
                r"|\butili[sz]ation (?:of|under|against) the "
                r"(?:funding |warehouse )?facility\b"
                r"|\butili[sz]ation of the (?:facility )?commitment\b"),
     "facility_utilisation"),
    (re.compile(r"\b(?:borrowing base|facility) headroom\b|\bheadroom\b"),
     "borrowing_base_headroom"),
    (re.compile(r"\bdeficiency\b|\bshortfall\b|\bover[\s-]?drawn\b"),
     "borrowing_base_deficiency"),
    (re.compile(r"\badvance rate\b"), "advance_rate"),
    (re.compile(r"\bfacility commitment\b|\bcommitment\b|\bcommitted amount\b"),
     "facility_commitment"),
    (re.compile(r"\bfacility drawn\b|\bdrawn\b|\bdrawings?\b|\bdrawdown\b"),
     "facility_drawn"),
    (re.compile(r"\beligible collateral\b"
                r"|\b(?<!in)eligible (?:current |loan |mortgage loan )?balance\b"),
     "eligible_balance"),
    (re.compile(r"\bhow many (?:loans )?(?:are |is )?(?<!in)eligible\b"
                r"|\bnumber of (?<!in)eligible loans\b"
                r"|\b(?<!in)eligible loans?\b(?! balance)"
                r"|\b(?<!in)eligible mortgage loans?\b(?! balance)"),
     "eligible_loan_count"),
    (re.compile(r"\bfinancing portfolio balance\b"),
     "financing_portfolio_balance"),
    (re.compile(r"\bborrowing base\b"), "borrowing_base"),
)
_BARE_UTILISATION_RE = re.compile(r"\butili[sz]ation\b")
_BB_CONTEXT_RE = re.compile(
    r"\b(?:under|of|against|on|in|for|within) (?:the |our )?borrowing base\b")

#: The ineligible family is read from three signals over the remaining text.
_INELIGIBLE_RE = re.compile(r"\bineligib\w*\b")
_SHARE_OF_BALANCE_RE = re.compile(
    r"\b(?:share|proportion|percentage|percent|%|fraction) of (?:the |our )?"
    r"(?:financing portfolio |total |book |portfolio )?balance\b"
    r"|\bbalance share\b|\bby balance\b|\bshare of (?:the )?financing "
    r"portfolio balance\b|\bof the financing portfolio balance\b")
_SHARE_OF_LOANS_RE = re.compile(
    r"\b(?:share|proportion|percentage|percent|%|fraction) of (?:the |our |all )?"
    r"(?:ineligible |eligible )?loans\b|\bloan share\b|\bby (?:count|number)\b"
    r"|\bshare of (?:the )?book\b")
_SHARE_RE = re.compile(r"\bshare\b|\bproportion\b|\bpercentage\b|\bpercent\b"
                       r"|\b%|\bfraction\b|\bwhat % ")
_COUNT_RE = re.compile(r"\bhow many\b|\bnumber of\b|\bcount\b")
_BALANCE_RE = re.compile(r"\bbalance\b|\bcollateral\b|\bamount\b|\bhow much\b"
                         r"|\bvalue\b|\bexposure\b")

#: What a NOT_CALCULABLE measure's missing input means, in plain words. The
#: SAME phrases the dashboard's tiles use, so the two surfaces explain a gap
#: identically.
_MISSING_INPUT_PHRASE = {
    "current_drawn_amount": ("the current drawn amount under the facility has "
                             "not been supplied"),
    "facility_commitment": "no facility commitment is configured",
    "advance_rate": "no advance rate is configured",
    "concentration_denominator_floor": ("no Concentration Limit Denominator "
                                        "floor is configured"),
}
_DRAWN_DEPENDENT = set(analysis.DRAWN_DEPENDENT_MEASURES)

_INTENT_CURRENT = "current_position"
_INTENT_REASONS = "ineligibility_reasons"
_INTENT_CHANGE = "period_change"
_INTENT_TREND = "period_trend"
_INTENT_BRIDGE = "bridge"


def _normalise(question: str) -> str:
    """The one normaliser's text, single-spaced, with the route's own spelling
    variant folded. The hyphen rule is NOT restated here — see
    `question_interpretation.normalise`."""
    from question_interpretation.normalise import normalise_question

    text = " ".join(normalise_question(question).split())
    return text.replace("utilization", "utilisation")


# --------------------------------------------------------------------------- #
# Reading the question — once, at recognition, kept for the handler
# --------------------------------------------------------------------------- #
@dataclass
class Reading:
    """Everything the handler needs, read from the sentence exactly once."""

    matched: bool
    reason: str = ""
    intent: str = _INTENT_CURRENT
    measures: List[str] = field(default_factory=list)
    wants_table: bool = False
    wants_waterfall: bool = False
    unsupported: List[str] = field(default_factory=list)
    requested_start: Optional[str] = None
    requested_end: Optional[str] = None
    relative_mode: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"intent": self.intent, "measures": list(self.measures),
                "wantsTable": self.wants_table,
                "wantsWaterfall": self.wants_waterfall,
                "unsupported": list(self.unsupported),
                "requestedStart": self.requested_start,
                "requestedEnd": self.requested_end,
                "relativeMode": self.relative_mode}


RECOGNITION_KEY = "borrowing_base_reading"


def _ineligible_measures(text: str) -> Tuple[List[Tuple[int, str]], str]:
    """``([(position, measure_id)], remainder)`` for the ineligible family.

    count / balance / count share / balance share, each named explicitly and
    each placed at the position of the word that asked for it, so a
    multi-measure answer keeps the question's own order. The phrases read are
    CONSUMED from the returned remainder so no later pattern re-reads them.
    """
    out: List[Tuple[int, str]] = []
    remainder = text

    def _take(pattern: re.Pattern[str]) -> Optional[int]:
        nonlocal remainder
        m = pattern.search(remainder)
        if not m:
            return None
        remainder = remainder[:m.start()] + " " * (m.end() - m.start()) + remainder[m.end():]
        return m.start()

    anchor = _INELIGIBLE_RE.search(text)
    anchor_pos = anchor.start() if anchor else 0
    balance_share = _take(_SHARE_OF_BALANCE_RE)
    loan_share = _take(_SHARE_OF_LOANS_RE)
    bare_share = None
    if balance_share is None and loan_share is None:
        bare_share = _take(_SHARE_RE)
    count = _take(_COUNT_RE)
    balance = _take(_BALANCE_RE)
    if count is not None:
        out.append((count, "ineligible_loan_count"))
    if balance is not None:
        out.append((balance, "ineligible_balance"))
    if loan_share is not None:
        out.append((loan_share, "ineligible_loan_share"))
    if bare_share is not None:
        out.append((bare_share, "ineligible_loan_share"))
        out.append((bare_share, "ineligible_balance_share"))
    if balance_share is not None:
        out.append((balance_share, "ineligible_balance_share"))
    if not out:
        # "Tell me about ineligible loans": the two governed facts, both named.
        out = [(anchor_pos, "ineligible_loan_count"),
               (anchor_pos, "ineligible_balance")]
    remainder = _INELIGIBLE_RE.sub(lambda m: " " * len(m.group(0)), remainder)
    return out, remainder


def read(question: str, spec: Any = None, *, view: str = "funded",
         source_lens: Any = None) -> Reading:
    """Decide whether the sentence is a borrowing-base question, and read it."""
    text = _normalise(question)
    if (view or "funded").strip().lower() not in _FUNDED_VIEWS:
        return Reading(False, "not the funded dataset")
    core = bool(_BB_CORE_RE.search(text))
    facility = bool(_FACILITY_RE.search(text))
    eligible = bool(_ELIGIBLE_RE.search(text))
    # THE CAPABILITY'S OWN CLAIM (I6): "how much collateral value are we able
    # to borrow against" names no borrowing-base word this table knows, and is
    # a borrowing-base question. The claim is the same one the parser masked
    # the measure binder with, so recognition and parsing cannot disagree.
    from mi_agent.borrowing_base import capacity as _capacity

    capacity = list(_capacity.claims(text))
    if capacity:
        core = True
    if not (core or facility or eligible):
        return Reading(False, "no borrowing-base vocabulary")
    if _CONCENTRATION_RE.search(text) and not core:
        # "ineligible under the concentration tests" et al.: a limit question.
        return Reading(False, "concentration-limit vocabulary without the "
                              "borrowing base named")

    reading = Reading(True, "borrowing-base vocabulary")
    reading.wants_table = bool(_TABLE_RE.search(text))
    reading.wants_waterfall = bool(_WATERFALL_RE.search(text))

    # --- facets this surface cannot honour, refused by name ---------------- #
    for pattern, label in _UNSUPPORTED:
        if pattern.search(text):
            reading.unsupported.append(label)
    for m in _BY_RE.finditer(text):
        reading.unsupported.append(f"a breakdown by {m.group(1).strip()}")
    filters = dict(getattr(spec, "filters", None) or {})
    if filters:
        reading.unsupported.append(
            "narrowing to " + ", ".join(f"{k} = {v}" for k, v in filters.items()))
    dims = list(getattr(spec, "dimensions", None) or [])
    single = getattr(spec, "dimension", None)
    if single and single not in dims:
        dims.append(single)
    if dims and not _REASON_RE.search(text) and not _TREND_RE.search(text):
        reading.unsupported.append("a breakdown by " + ", ".join(map(str, dims)))
    if source_lens:
        reading.unsupported.append(
            f"a portfolio lens ({source_lens}); the Eligibility & "
            "Concentrations tab presents the scoped facility position")

    # --- intent ------------------------------------------------------------ #
    from mi_agent.period_change.recognition import (  # governed period owner
        _explicit_periods,
        _relative_mode,
    )
    start, end = _explicit_periods(question, spec)
    relative = _relative_mode(question)
    reading.requested_start, reading.requested_end = start, end
    reading.relative_mode = None if (start or end) else relative

    movement = False
    try:
        from question_interpretation.lexical import is_movement_question
        movement = bool(is_movement_question(question))
    except Exception:  # noqa: BLE001 - the owner missing reads as level
        movement = False
    # THE SAME OWNER THE RECEIPT RAISES ITS COMPARISON FACET FROM. A question
    # that owner reads as a period comparison must reach the change or bridge
    # path here, never the current position — that is what makes membership
    # of `TEMPORAL_ROUTES` safe for this route.
    try:
        from mi_agent import execution_receipt as _receipt
        movement = movement or bool(_receipt._detect_comparison_period(text))
    except Exception:  # noqa: BLE001
        pass

    if eligible and _REASON_RE.search(text) and not (core and _BRIDGE_RE.search(text)):
        reading.intent = _INTENT_REASONS
    elif core and _BRIDGE_RE.search(text):
        reading.intent = _INTENT_BRIDGE
    elif _TREND_RE.search(text):
        reading.intent = _INTENT_TREND
    elif movement or start or end or relative:
        reading.intent = _INTENT_CHANGE
    else:
        reading.intent = _INTENT_CURRENT

    # --- measures ---------------------------------------------------------- #
    # Read in the question's own order: every match records where it was
    # found, and the phrases read are BLANKED (offsets preserved) so a later,
    # broader pattern cannot re-read them.
    found: List[Tuple[int, str]] = []
    remainder = text
    if _INELIGIBLE_RE.search(remainder):
        family, remainder = _ineligible_measures(remainder)
        found.extend(family)

    def _take(pattern: re.Pattern[str], measure_id: str) -> None:
        nonlocal remainder
        m = pattern.search(remainder)
        if not m:
            return
        found.append((m.start(), measure_id))
        remainder = pattern.sub(lambda x: " " * len(x.group(0)), remainder)

    for pattern, measure_id in _MEASURE_PATTERNS[:-1]:
        _take(pattern, measure_id)
    if _BARE_UTILISATION_RE.search(remainder):
        pos = _BARE_UTILISATION_RE.search(remainder).start()
        remainder = _BARE_UTILISATION_RE.sub(lambda x: " " * len(x.group(0)), remainder)
        if facility and not core:
            found.append((pos, "facility_utilisation"))
        elif core and not facility:
            found.append((pos, "borrowing_base_utilisation"))
        else:
            found.append((pos, "borrowing_base_utilisation"))
            found.append((pos + 1, "facility_utilisation"))
    # "headroom UNDER THE BORROWING BASE" names the base as context for the
    # measure already read, not as a second measure.
    remainder = _BB_CONTEXT_RE.sub(lambda x: " " * len(x.group(0)), remainder)
    _take(_MEASURE_PATTERNS[-1][0], _MEASURE_PATTERNS[-1][1])

    measures = [m for _, m in sorted(found, key=lambda pair: pair[0])]
    if reading.intent == _INTENT_BRIDGE:
        measures = ["borrowing_base"]
    elif not measures and capacity:
        measures = [_capacity.measure_for(concept) for _cap, concept, *_ in capacity]
    elif not measures:
        measures = (["borrowing_base", "facility_commitment", "facility_drawn",
                     "borrowing_base_headroom", "facility_utilisation"]
                    if facility and not core else ["borrowing_base"])
    reading.measures = [m for m in dict.fromkeys(measures) if m in MEASURES]
    return reading


# --------------------------------------------------------------------------- #
# Registry contract
# --------------------------------------------------------------------------- #
def recognise(request: RouteRequest) -> Recognition:
    reading = read(request.question, request.spec, view=request.view,
                   source_lens=request.source_lens)
    remember = getattr(request, "remember_recognition", None)
    if remember is not None:
        remember(RECOGNITION_KEY, reading)
    if not reading.matched:
        return Recognition.no(reading.reason)
    return Recognition.yes(ROUTE_CONFIDENCE, f"{reading.reason}:{reading.intent}")


def recogniser() -> Recogniser:
    return Recogniser(
        name=ROUTE_NAME, priority=ROUTE_PRIORITY, lens_aware=False,
        description=("Straightforward facility borrowing-base questions from "
                     "the SAME governed envelope the dashboard renders."),
        metadata={"owner": "mi_agent.borrowing_base",
                  "envelope": "borrowing_base_api.compute_borrowing_base",
                  "analysis": "mi_agent.borrowing_base.analysis",
                  "intents": (_INTENT_CURRENT, _INTENT_REASONS, _INTENT_CHANGE,
                              _INTENT_TREND, _INTENT_BRIDGE)},
        recognise=recognise, handle=handle)


# --------------------------------------------------------------------------- #
# Formatting — the measure's OWN declared unit, never guessed
# --------------------------------------------------------------------------- #
def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fmt(measure_id: str, value: Any) -> str:
    from . import chat_routing as _routing
    if not _is_number(value):
        return "not calculable"
    unit = MEASURES[measure_id].unit
    if unit == "currency":
        return _routing._gbp(float(value))
    if unit == "percent":
        return f"{float(value):.2f}%"
    if unit == "count":
        return f"{int(round(float(value))):,}"
    return str(value)


def _fmt_delta(measure_id: str, value: Any) -> str:
    if not _is_number(value):
        return "not calculable"
    sign = "+" if float(value) >= 0 else "−"
    unit = MEASURES[measure_id].unit
    body = _fmt(measure_id, abs(float(value)))
    if unit == "percent":
        body = f"{abs(float(value)):.2f} pp"
    return f"{sign}{body}"


def _label(measure_id: str) -> str:
    return MEASURES[measure_id].display_name


def _value_format(measure_id: str) -> str:
    unit = MEASURES[measure_id].unit
    return {"currency": "gbp", "percent": "pct", "count": "number"}.get(unit, "text")


def _why_missing(envelope: Dict[str, Any], measure_id: str) -> str:
    missing = list(envelope.get("missingInputs") or [])
    if measure_id in _DRAWN_DEPENDENT and "current_drawn_amount" in missing:
        return _MISSING_INPUT_PHRASE["current_drawn_amount"] + \
            " (missing input: current_drawn_amount)"
    if measure_id in ("borrowing_base", "gross_borrowing_base",
                      "borrowing_base_headroom", "borrowing_base_deficiency",
                      "borrowing_base_utilisation") and "advance_rate" in missing:
        return _MISSING_INPUT_PHRASE["advance_rate"] + " (missing input: advance_rate)"
    if measure_id in ("facility_commitment", "facility_utilisation") \
            and "facility_commitment" in missing:
        return _MISSING_INPUT_PHRASE["facility_commitment"] + \
            " (missing input: facility_commitment)"
    if missing:
        phrases = [_MISSING_INPUT_PHRASE.get(m, m) for m in missing]
        return "; ".join(phrases) + " (missing input: " + ", ".join(missing) + ")"
    return "its inputs are not available for this position"


# --------------------------------------------------------------------------- #
# Envelope helpers
# --------------------------------------------------------------------------- #
def _refuse(request: RouteRequest, answer: str, *, reading: Optional[Reading],
            warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    from . import chat_routing as _routing
    out = _routing._undeliverable(
        question=request.question, answer=answer, spec=request.spec_dict,
        route=ROUTE_NAME, warnings=warnings or [])
    out["metadata"]["borrowingBase"] = {
        "reading": reading.to_dict() if reading else None}
    return out


def _source_note(envelope: Dict[str, Any]) -> Dict[str, Any]:
    facility = envelope.get("facility") or {}
    label = facility.get("facilityLabel") or facility.get("facilityId") or "facility"
    return {
        "label": "Borrowing base",
        "note": (f"{label}: governed borrowing-base envelope (configuration "
                 f"{facility.get('configVersion') or 'unversioned'}, hash "
                 f"{facility.get('configHash') or 'n/a'}), evaluated at "
                 f"{envelope.get('reportingDate') or 'the latest snapshot'} on "
                 f"run {envelope.get('toRunId') or 'latest'} — the same block "
                 "the Eligibility & Concentrations dashboard renders."),
    }


def _disclosures(envelope: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for note in envelope.get("prototypeAssumptionsUsed") or []:
        out.append(f"Prototype assumption in use: {note}")
    for problem in envelope.get("configurationProblems") or []:
        out.append(f"Configuration: {problem}")
    return out


def _current_envelope(request: RouteRequest) -> Dict[str, Any]:
    from . import borrowing_base_api as bb_api
    return bb_api.compute_borrowing_base(request.output_root, request.client_id,
                                         request.run_id)


def _facility(request: RouteRequest):
    from mi_agent.borrowing_base.config import load_facility
    try:
        return load_facility(request.client_id)
    except Exception:  # noqa: BLE001 - unavailable reads as no facility
        return None


def _gate(request: RouteRequest, envelope: Dict[str, Any], reading: Reading
          ) -> Optional[Dict[str, Any]]:
    """The refusals every intent shares: unsupported facets, no facility,
    unreconciled population. ``None`` means proceed."""
    if reading.unsupported:
        facets = "; ".join(dict.fromkeys(reading.unsupported))
        return _refuse(request, (
            "The borrowing-base question was recognised, but it asks for "
            f"something the governed borrowing base does not provide: {facets}. "
            "The facility position, eligibility split, ineligibility reasons, "
            "period change and the borrowing-base bridge are the v1 surface; "
            "nothing was answered in their place."), reading=reading)
    if not envelope.get("available"):
        return _refuse(request, str(envelope.get("reason") or
                                    "The borrowing base is not available for "
                                    "this portfolio."), reading=reading)
    if envelope.get("reconciles") is False:
        failed = [i.get("invariant") for i in (envelope.get("invariants") or [])
                  if not i.get("holds")]
        return _refuse(request, (
            "The borrowing-base eligibility population does not reconcile ("
            + ", ".join(map(str, failed)) + "), so no figure from it is "
            "presented as a governed borrowing base."), reading=reading,
            warnings=_disclosures(envelope))
    return None


# --------------------------------------------------------------------------- #
# What the question carries that this surface cannot honour — read from the
# receipt's OWN facet detector, so the route and the guard agree by construction
# --------------------------------------------------------------------------- #
def _unhonoured_facets(request: RouteRequest) -> List[str]:
    """Labels of every requested facet this surface does not serve."""
    from mi_agent import execution_receipt as _receipt

    honoured = {
        _receipt.KIND_SHARE, _receipt.KIND_MULTI_MEASURE,
        _receipt.KIND_UNRESOLVED_MEASURE, _receipt.KIND_COMPARISON_PERIOD,
        _receipt.KIND_SERIES_AXIS, _receipt.KIND_GRANULARITY,
        _receipt.KIND_TIME_GRAIN, _receipt.KIND_STATISTIC,
    }
    frame = None
    try:
        resolver = (getattr(request, "base_frame_resolver", None)
                    or request.frame_resolver)
        if resolver is not None:
            frame = resolver(request.view, request.portfolio_id)
    except Exception:  # noqa: BLE001 - no frame, facets read from wording alone
        frame = None
    try:
        facets = _receipt.detect_requested_facets(
            request.question, dict(request.semantics or {}), frame=frame)
    except Exception as exc:  # noqa: BLE001 - the guard downstream still runs
        logger.info("facet audit unavailable for %r: %s", request.question, exc)
        return []
    return list(dict.fromkeys(
        str(f.label) for f in facets if f.kind not in honoured))


#: The receipt's measure-concept vocabulary, per governed measure.
_CONCEPT_OF_MEASURE = {
    "ineligible_loan_count": "count", "eligible_loan_count": "count",
    "financing_portfolio_loan_count": "count",
    "ineligible_balance": "balance", "eligible_balance": "balance",
    "financing_portfolio_balance": "balance", "borrowing_base": "balance",
    "gross_borrowing_base": "balance", "facility_commitment": "balance",
    "facility_drawn": "balance", "borrowing_base_headroom": "balance",
    "borrowing_base_deficiency": "balance",
    "ineligible_loan_share": "share", "ineligible_balance_share": "share",
}
_SLOT_RESOLVED_RE = re.compile(
    r"\bshare\b|\bproportion\b|\bpercentage\b|\bineligib|\beligible collateral\b"
    r"|\bheadroom\b|\butilisation\b|\bdrawn\b|\bcommitment\b|\bborrowing base\b"
    r"|\badvance rate\b|\bdeficiency\b")


def _measure_set(request: RouteRequest, reading: Reading,
                 delivered: Sequence[str] = ()) -> Dict[str, Any]:
    """The measure-set evidence this answer publishes for the receipt guard.

    Concepts are named in the receipt's own vocabulary; slots are the
    coordinated-list phrases the PARSER'S own slot reader found, kept only
    where this route's vocabulary resolved them.
    """
    concepts = {_CONCEPT_OF_MEASURE[m] for m in reading.measures
                if m in _CONCEPT_OF_MEASURE} | set(delivered)
    slots: List[str] = []
    try:
        from mi_agent.llm_query_parser import unresolved_measure_slots
        for slot in unresolved_measure_slots(request.question,
                                             dict(request.semantics or {})):
            if _SLOT_RESOLVED_RE.search(_normalise(slot)):
                slots.append(slot)
    except Exception:  # noqa: BLE001 - no slots declared, guard decides
        pass
    return {"concepts": sorted(concepts), "slots": slots}


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def handle(request: RouteRequest) -> Optional[Dict[str, Any]]:
    reading = request.recalled_recognition(RECOGNITION_KEY) or read(
        request.question, request.spec, view=request.view,
        source_lens=request.source_lens)
    if not reading.matched:
        return None
    try:
        envelope = _current_envelope(request)
    except Exception as exc:  # noqa: BLE001 - never 500 the chat
        logger.warning("borrowing-base envelope unavailable: %s", exc)
        return _refuse(request, "The borrowing-base service could not be "
                                f"reached: {exc}", reading=reading)
    gate = _gate(request, envelope, reading)
    if gate is not None:
        return gate
    unhonoured = _unhonoured_facets(request)
    if unhonoured:
        return _refuse(request, (
            "The borrowing-base question was recognised, but it asks for "
            "something the governed borrowing base does not provide: "
            + "; ".join(unhonoured) + ". Nothing was answered in its place."),
            reading=reading)
    if reading.intent == _INTENT_REASONS:
        return _answer_reasons(request, envelope, reading)
    if reading.intent == _INTENT_BRIDGE:
        return _answer_bridge(request, envelope, reading)
    if reading.intent == _INTENT_TREND:
        return _answer_trend(request, envelope, reading)
    if reading.intent == _INTENT_CHANGE:
        return _answer_change(request, envelope, reading)
    return _answer_current(request, envelope, reading)


def _finish(request: RouteRequest, envelope: Dict[str, Any], reading: Reading,
            *, answer: str, artifacts: List[Dict[str, Any]],
            warnings: List[str], extra: Dict[str, Any],
            delivered: Sequence[str] = ()) -> Dict[str, Any]:
    from . import chat_routing as _routing
    out = _routing._envelope(
        ok=True, question=request.question, answer=answer,
        spec=request.spec_dict, artifacts=artifacts, route=ROUTE_NAME,
        source_notes=[_source_note(envelope)], warnings=warnings)
    out["metadata"]["measureSet"] = _measure_set(request, reading, delivered)
    facility = envelope.get("facility") or {}
    out["metadata"]["borrowingBase"] = {
        "reading": reading.to_dict(),
        "reportingDate": envelope.get("reportingDate"),
        "toRunId": envelope.get("toRunId"),
        "facilityId": facility.get("facilityId"),
        "configVersion": facility.get("configVersion"),
        "configHash": facility.get("configHash"),
        "prototypeAssumptionsUsed": list(envelope.get("prototypeAssumptionsUsed") or []),
        "reconciles": envelope.get("reconciles"),
        **extra,
    }
    return out


def _answer_current(request: RouteRequest, envelope: Dict[str, Any],
                    reading: Reading) -> Dict[str, Any]:
    from . import chat_routing as _routing
    measures = envelope.get("measures") or {}
    facility = envelope.get("facility") or {}
    values: Dict[str, Any] = {}
    not_calculable: Dict[str, str] = {}
    parts: List[str] = []
    for measure_id in reading.measures:
        value = measures.get(measure_id, NOT_CALCULABLE)
        if _is_number(value):
            values[measure_id] = value
            parts.append(f"{_label(measure_id)} {_fmt(measure_id, value)}")
        else:
            why = _why_missing(envelope, measure_id)
            not_calculable[measure_id] = why
            parts.append(f"{_label(measure_id)} is not calculable — {why}")

    context = (f"As at {envelope.get('reportingDate') or 'the latest snapshot'}"
               f" ({facility.get('facilityLabel') or facility.get('facilityId')}")
    if _is_number(envelope.get("advanceRatePct")) and "borrowing_base" in values:
        context += (f", advance rate {envelope['advanceRatePct']:.0f}% on eligible "
                    f"collateral {_routing._gbp(envelope.get('eligibleCurrentBalance'))}")
        if envelope.get("facilityCapBinding"):
            context += ", capped at the facility commitment"
    context += ")."
    answer = "; ".join(parts) + ". " + context

    warnings = _disclosures(envelope)
    drawn_as_of = facility.get("currentDrawnAmountAsOf")
    if any(m in _DRAWN_DEPENDENT for m in values) and drawn_as_of \
            and str(drawn_as_of)[:10] != str(envelope.get("reportingDate") or "")[:10]:
        warnings.append(
            f"Drawings are the operator-supplied amount stated as at {drawn_as_of}, "
            f"not at the snapshot date {envelope.get('reportingDate')}.")
    elif any(m in _DRAWN_DEPENDENT for m in values) and not drawn_as_of:
        warnings.append("Drawings are the operator-supplied current amount; "
                        "no as-of date was recorded for it.")

    artifacts: List[Dict[str, Any]] = []
    if len(reading.measures) >= 2:
        rows = [{"measure": _label(m),
                 "value": _fmt(m, measures.get(m, NOT_CALCULABLE)),
                 "status": ("governed" if m in values else
                            f"NOT_CALCULABLE — {not_calculable[m]}")}
                for m in reading.measures]
        artifacts.append(_routing._table_artifact(
            f"Borrowing-base position — {envelope.get('reportingDate') or 'latest'}",
            columns=[{"key": "measure", "label": "Measure", "align": "left", "format": "text"},
                     {"key": "value", "label": "Value", "align": "right", "format": "text"},
                     {"key": "status", "label": "Status", "align": "left", "format": "text"}],
            rows=rows, spec=request.spec_dict, portfolio_id=request.portfolio_id,
            as_of=request.as_of,
            description="Governed borrowing-base measures; NOT_CALCULABLE names "
                        "the missing input rather than showing a zero."))
    return _finish(request, envelope, reading, answer=answer, artifacts=artifacts,
                   warnings=warnings,
                   extra={"intent": _INTENT_CURRENT, "values": values,
                          "notCalculable": not_calculable,
                          "drawnAsOf": drawn_as_of})


def _answer_reasons(request: RouteRequest, envelope: Dict[str, Any],
                    reading: Reading) -> Dict[str, Any]:
    from . import chat_routing as _routing
    from . import concentration_tests_api as conc_mod

    facility = _facility(request)
    if facility is None:
        return _refuse(request, "No funding facility is configured for this "
                                "portfolio.", reading=reading)
    try:
        df, _prior, _rd, _prd, _run = conc_mod._resolve_frames(
            request.output_root, request.client_id, request.run_id)
    except Exception as exc:  # noqa: BLE001
        return _refuse(request, "The governed funded frames could not be "
                                f"resolved: {exc}", reading=reading)
    lib = None
    try:
        from mi_agent.concentration_tests.library import load_library
        lib = load_library()
    except Exception:  # noqa: BLE001 - the literal balance fallback then applies
        lib = None
    summary = analysis.summarise_ineligibility_reasons(df, facility, lib=lib)

    # THE TABLE MUST RECONCILE TO THE ENVELOPE THE DASHBOARD SHOWS. Same
    # partition by construction; asserted anyway, because a silent drift here
    # would be two ineligible figures on two surfaces.
    env_count = envelope.get("ineligibleLoanCount")
    env_balance = envelope.get("ineligibleCurrentBalance")
    drift = (summary.ineligible_loan_count != env_count
             or (_is_number(env_balance)
                 and abs(summary.ineligible_balance - float(env_balance)) > 0.02))
    if not summary.available or drift:
        why = summary.reason if not summary.available else (
            "the reason population does not match the governed borrowing-base "
            f"envelope (envelope {env_count} loans / {env_balance}; reasons "
            f"{summary.ineligible_loan_count} / {summary.ineligible_balance})")
        return _refuse(request, f"The ineligibility reason breakdown is refused: "
                                f"{why}", reading=reading,
                       warnings=_disclosures(envelope))

    total_count = summary.ineligible_loan_count
    if total_count == 0:
        answer = (f"No loans are ineligible as at {envelope.get('reportingDate')}: "
                  f"the Financing Portfolio's {summary.financing_portfolio_loan_count:,} "
                  f"loans are {envelope.get('eligibleLoanCount', 0):,} eligible and "
                  f"{envelope.get('undeterminedLoanCount', 0):,} undetermined "
                  "(undetermined is not ineligible — an input is missing). "
                  "There is no reason breakdown to show.")
        return _finish(request, envelope, reading, answer=answer, artifacts=[],
                       warnings=_disclosures(envelope),
                       extra={"intent": _INTENT_REASONS,
                              "reasons": summary.to_dict()})

    rows = [{"reason": r.reason_description, "code": r.reason_code,
             "loans": r.loan_count, "balance": r.balance,
             "share_loans": r.share_of_ineligible_loans_pct,
             "share_balance": r.share_of_ineligible_balance_pct}
            for r in summary.rows]
    table = _routing._table_artifact(
        f"Ineligible loans by primary reason — {envelope.get('reportingDate')}",
        columns=[
            {"key": "reason", "label": "Reason", "align": "left", "format": "text"},
            {"key": "loans", "label": "Loans", "align": "right", "format": "number"},
            {"key": "balance", "label": "Balance", "align": "right", "format": "gbp"},
            {"key": "share_loans", "label": "% of ineligible loans", "align": "right",
             "format": "pct", "scale": "percent_points"},
            {"key": "share_balance", "label": "% of ineligible balance", "align": "right",
             "format": "pct", "scale": "percent_points"},
        ],
        rows=rows, spec=request.spec_dict, portfolio_id=request.portfolio_id,
        as_of=request.as_of,
        description=("One primary governed reason per ineligible loan — the first "
                     "failing approved eligibility rule in configured order. Rows "
                     f"reconcile to {total_count:,} ineligible loans and "
                     f"{_routing._gbp(summary.ineligible_balance)}."))
    lead = summary.rows[0]
    answer = (
        f"{total_count:,} loans are ineligible as at {envelope.get('reportingDate')} "
        f"({_routing._gbp(summary.ineligible_balance)}; "
        f"{_fmt('ineligible_loan_share', summary.ineligible_loan_share_pct)} of "
        f"Financing Portfolio loans, "
        f"{_fmt('ineligible_balance_share', summary.ineligible_balance_share_pct)} of "
        f"its balance). By primary reason: "
        + "; ".join(
            f"{r.reason_description} — {r.loan_count:,} loans, "
            f"{_routing._gbp(r.balance)} ({_fmt('ineligible_loan_share', r.share_of_ineligible_loans_pct)} "
            f"of ineligible loans, {_fmt('ineligible_balance_share', r.share_of_ineligible_balance_pct)} "
            "of ineligible balance)" for r in summary.rows)
        + f". Largest: {lead.reason_description}. Each loan is counted once under "
          "its primary reason, the first approved rule it fails.")
    warnings = _disclosures(envelope) + list(summary.notes)
    return _finish(request, envelope, reading, answer=answer, artifacts=[table],
                   warnings=warnings, delivered=("count", "balance", "share"),
                   extra={"intent": _INTENT_REASONS, "reasons": summary.to_dict()})


# --------------------------------------------------------------------------- #
# Periods — discovered by the governed loaders, resolved by the TIME owner
# --------------------------------------------------------------------------- #
def _positions(request: RouteRequest, facility) -> List[analysis.PeriodPosition]:
    """One governed envelope per period, from the SAME calculator."""
    from . import borrowing_base_api as bb_api
    from . import evolution as evolution_mod

    frames = evolution_mod.funded_frames(request.output_root, request.client_id,
                                         request.run_id)
    frames = list(frames or [])[-MAX_PERIODS:]
    positions: List[analysis.PeriodPosition] = []
    for index, frame in enumerate(frames):
        is_current = index == len(frames) - 1
        reporting_date = frame.get("reporting_date")
        envelope = bb_api.compute_from_frames(
            frame.get("df"), client_id=request.client_id,
            facility=analysis.facility_for_period(facility, reporting_date),
            reporting_date=reporting_date, run_id=frame.get("run_id"))
        positions.append(analysis.position_for(
            str(frame.get("run_id")), reporting_date, envelope,
            facility=facility, is_current=is_current))
    return positions


def _resolve_pair(request: RouteRequest, reading: Reading,
                  positions: Sequence[analysis.PeriodPosition]
                  ) -> Tuple[Optional[analysis.PeriodPosition],
                             Optional[analysis.PeriodPosition],
                             Optional[str], List[str]]:
    """(opening, closing, refusal, notes) via the governed period resolver."""
    from mi_agent.period_change.models import PeriodChangeFailure, SnapshotFrame
    from mi_agent.period_change.periods import PeriodRequest, resolve_periods

    max_gap = None
    try:
        from mi_agent.period_change.selection import load_policy
        max_gap = load_policy().max_snapshot_gap_days(None)
    except Exception:  # noqa: BLE001 - no ceiling rather than no answer
        max_gap = None
    snapshots = [SnapshotFrame(snapshot_id=p.run_id, reporting_date=p.reporting_date)
                 for p in positions]
    try:
        resolution = resolve_periods(
            snapshots,
            PeriodRequest(requested_start=reading.requested_start,
                          requested_end=reading.requested_end,
                          relative_mode=reading.relative_mode),
            max_gap_days=max_gap)
    except PeriodChangeFailure as failure:
        return None, None, failure.message, []
    by_id = {p.run_id: p for p in positions}
    notes = list(resolution.adjustment_notes)
    return (by_id[resolution.start_snapshot.snapshot_id],
            by_id[resolution.end_snapshot.snapshot_id], None, notes)


def _history(request: RouteRequest, reading: Reading
             ) -> Tuple[Any, List[analysis.PeriodPosition], Optional[Dict[str, Any]]]:
    """``(facility, positions, refusal)`` — the governed periods, or why not."""
    facility = _facility(request)
    if facility is None:
        return None, [], _refuse(request, "No funding facility is configured for "
                                          "this portfolio.", reading=reading)
    try:
        positions = _positions(request, facility)
    except Exception as exc:  # noqa: BLE001
        return facility, [], _refuse(request, "The governed historical periods "
                                              f"could not be resolved: {exc}",
                                     reading=reading)
    if len(positions) < 2:
        return facility, positions, _refuse(request, (
            "A period change needs two governed reporting periods and this book "
            f"has {len(positions)}."), reading=reading)
    return facility, positions, None


def _applicability_note(facility) -> str:
    """The two facts history turns on, as this facility's record states them."""
    ungoverned = analysis.terms_governed_for_history(facility)
    effective, maturity = analysis.contractual_window(facility)
    if ungoverned:
        return (f"Historical borrowing-base measures are not calculable: "
                f"{ungoverned}. An approved Eligible Mortgage Loan definition "
                "(recorded through OCC) is what makes history answerable.")
    if not effective:
        return ("Historical borrowing-base measures are not calculable: the "
                "facility record states no effective date, so the contractual "
                "period the approved terms cover is not recorded.")
    return (f"Historical periods are evaluated under the approved facility terms "
            f"within their recorded contractual window ({effective} to "
            f"{maturity or 'open'}); the register holds one configuration version "
            "and no amendment history, so those terms are taken as constant "
            "across it.")


def _answer_change(request: RouteRequest, envelope: Dict[str, Any],
                   reading: Reading) -> Dict[str, Any]:
    from . import chat_routing as _routing
    facility, positions, refusal = _history(request, reading)
    if refusal is not None:
        return refusal
    opening, closing, failure, notes = _resolve_pair(request, reading, positions)
    if failure:
        return _refuse(request, failure, reading=reading)

    changes = [analysis.change(opening, closing, m, unit=MEASURES[m].unit)
               for m in reading.measures]
    if not any(c.calculable for c in changes):
        why = "; ".join(dict.fromkeys(c.reason for c in changes if c.reason))
        return _refuse(request, (
            f"The change from {opening.period_label} to {closing.period_label} is "
            f"not calculable for {', '.join(_label(c.measure_id) for c in changes)}: "
            f"{why}. " + _applicability_note(facility)), reading=reading,
            warnings=_disclosures(envelope) + notes)

    parts: List[str] = []
    for c in changes:
        if c.calculable:
            pct = (f" ({c.change_pct:+.2f}%)" if _is_number(c.change_pct) else "")
            parts.append(f"{_label(c.measure_id)} moved from "
                         f"{_fmt(c.measure_id, c.opening)} ({c.opening_period}) to "
                         f"{_fmt(c.measure_id, c.closing)} ({c.closing_period}): "
                         f"{_fmt_delta(c.measure_id, c.change)}{pct}")
        else:
            parts.append(f"{_label(c.measure_id)} change is not calculable — {c.reason}")
    answer = "; ".join(parts) + "."
    rows = [{"measure": _label(c.measure_id),
             "opening": _fmt(c.measure_id, c.opening),
             "closing": _fmt(c.measure_id, c.closing),
             "change": _fmt_delta(c.measure_id, c.change),
             "change_pct": (f"{c.change_pct:+.2f}%" if _is_number(c.change_pct) else "—")}
            for c in changes]
    table = _routing._table_artifact(
        f"Borrowing-base change — {opening.period_label} to {closing.period_label}",
        columns=[{"key": "measure", "label": "Measure", "align": "left", "format": "text"},
                 {"key": "opening", "label": opening.period_label, "align": "right", "format": "text"},
                 {"key": "closing", "label": closing.period_label, "align": "right", "format": "text"},
                 {"key": "change", "label": "Change", "align": "right", "format": "text"},
                 {"key": "change_pct", "label": "Change %", "align": "right", "format": "text"}],
        rows=rows, spec=request.spec_dict, portfolio_id=request.portfolio_id,
        as_of=request.as_of,
        description="Each period evaluated by the same governed borrowing-base "
                    "calculator; the change is composed from those envelopes.")
    warnings = _disclosures(envelope) + notes + [_applicability_note(facility)]
    warnings += [n for p in (opening, closing) for n in p.notes]
    return _finish(request, envelope, reading, answer=answer, artifacts=[table],
                   warnings=list(dict.fromkeys(warnings)),
                   extra={"intent": _INTENT_CHANGE,
                          "opening": opening.to_dict(), "closing": closing.to_dict(),
                          "changes": [c.to_dict() for c in changes]})


def _answer_trend(request: RouteRequest, envelope: Dict[str, Any],
                  reading: Reading) -> Dict[str, Any]:
    from . import chat_routing as _routing
    facility, positions, refusal = _history(request, reading)
    if refusal is not None:
        return refusal
    artifacts: List[Dict[str, Any]] = []
    parts: List[str] = []
    all_series: Dict[str, List[Dict[str, Any]]] = {}
    for index, measure_id in enumerate(reading.measures[:3]):
        pts = analysis.series(positions, measure_id)
        all_series[measure_id] = pts
        plotted = [p for p in pts if p["value"] is not None]
        if len(plotted) < 2:
            # One calculable point is a position, not a trend.
            continue
        rows = [{"period": p["period"], "value": p["value"]} for p in pts]
        fmt = _value_format(measure_id)
        hints = {"value": {"format": fmt,
                           "scale": "percent_points" if fmt == "pct" else None}}
        title = f"{_label(measure_id)} by reporting period"
        if reading.wants_table:
            artifacts.append(_routing._table_artifact(
                title, columns=[
                    {"key": "period", "label": "Period", "align": "left", "format": "text"},
                    {"key": "value", "label": _label(measure_id), "align": "right",
                     "format": fmt, **({"scale": "percent_points"} if fmt == "pct" else {})},
                    {"key": "status", "label": "Status", "align": "left", "format": "text"}],
                rows=[{"period": p["period"], "value": p["value"],
                       "status": p["status"] if p["value"] is not None
                       else f"{NOT_CALCULABLE} — {p['reason']}"} for p in pts],
                spec=request.spec_dict, portfolio_id=request.portfolio_id,
                as_of=request.as_of))
        else:
            artifacts.append(_routing._chart_artifact(
                title, chart_type="line", x_key="period", rows=rows,
                series=[{"key": "value", "label": _label(measure_id),
                         "color": _routing._PALETTE[index % len(_routing._PALETTE)]}],
                value_format=fmt, spec=request.spec_dict,
                portfolio_id=request.portfolio_id, as_of=request.as_of,
                display_hints=hints,
                description="One governed borrowing-base evaluation per reporting "
                            "period; periods without a calculable value are gaps."))
        first, last = plotted[0], plotted[-1]
        parts.append(f"{_label(measure_id)}: {_fmt(measure_id, first['value'])} "
                     f"({first['period']}) → {_fmt(measure_id, last['value'])} "
                     f"({last['period']}) over {len(plotted)} of {len(pts)} periods")
    if not artifacts:
        reasons = "; ".join(dict.fromkeys(
            p["reason"] for pts in all_series.values() for p in pts if p["reason"]))
        return _refuse(request, (
            "A trend needs at least two governed periods with a calculable "
            f"value, and {', '.join(_label(m) for m in reading.measures)} has "
            f"fewer: {reasons}. " + _applicability_note(facility)),
            reading=reading, warnings=_disclosures(envelope))
    answer = "; ".join(parts) + "."
    warnings = _disclosures(envelope) + [_applicability_note(facility)]
    return _finish(request, envelope, reading, answer=answer, artifacts=artifacts,
                   warnings=warnings,
                   extra={"intent": _INTENT_TREND,
                          "periods": [p.to_dict() for p in positions],
                          "series": all_series})


def _answer_bridge(request: RouteRequest, envelope: Dict[str, Any],
                   reading: Reading) -> Dict[str, Any]:
    from . import chat_routing as _routing
    facility, positions, refusal = _history(request, reading)
    if refusal is not None:
        return refusal
    opening, closing, failure, notes = _resolve_pair(request, reading, positions)
    if failure:
        return _refuse(request, failure, reading=reading)

    if not opening.configuration_applicable or not closing.configuration_applicable:
        side = opening if not opening.configuration_applicable else closing
        return _refuse(request, (
            f"The borrowing-base bridge from {opening.period_label} to "
            f"{closing.period_label} is not calculable: "
            f"{side.why_not_calculable('borrowing_base')}. "
            + _applicability_note(facility)), reading=reading,
            warnings=_disclosures(envelope) + notes)

    result = analysis.bridge(
        opening.envelope, closing.envelope,
        opening_label=opening.period_label, closing_label=closing.period_label,
        opening_run_id=opening.run_id, closing_run_id=closing.run_id,
        opening_reporting_date=opening.reporting_date,
        closing_reporting_date=closing.reporting_date,
        opening_drawn_valid=opening.drawn_valid,
        closing_drawn_valid=closing.drawn_valid)
    if not result.available:
        return _refuse(request, result.reason, reading=reading,
                       warnings=_disclosures(envelope) + notes)

    rows = result.waterfall_rows()
    title = (f"Borrowing-base bridge — {opening.period_label} to "
             f"{closing.period_label}")
    if reading.wants_table and not reading.wants_waterfall:
        artifact = _routing._table_artifact(
            title, columns=[
                {"key": "label", "label": "Step", "align": "left", "format": "text"},
                {"key": "value", "label": "Amount", "align": "right", "format": "gbp"},
                {"key": "type", "label": "Type", "align": "left", "format": "text"}],
            rows=rows, spec=request.spec_dict, portfolio_id=request.portfolio_id,
            as_of=request.as_of,
            description="Opening borrowing base, three exact drivers, closing "
                        "borrowing base; reconciles within the calculator's "
                        "penny rounding.")
    else:
        artifact = _routing._chart_artifact(
            title, chart_type="waterfall", x_key="label", rows=rows,
            series=[{"key": "value", "label": "Borrowing base",
                     "color": _routing._PALETTE[0]}],
            value_format="gbp", spec=request.spec_dict,
            portfolio_id=request.portfolio_id, as_of=request.as_of,
            display_hints={"value": {"format": "gbp", "scale": None}},
            description=(f"Opening {opening.period_label} → eligible collateral "
                         "effect → advance rate effect → facility cap effect → "
                         f"closing {closing.period_label}."))

    drivers = "; ".join(f"{d.label} {_fmt_delta('borrowing_base', d.value)}"
                        for d in result.drivers)
    answer = (
        f"Borrowing base moved from {_routing._gbp(result.opening.borrowing_base)} "
        f"({opening.period_label}) to {_routing._gbp(result.closing.borrowing_base)} "
        f"({closing.period_label}): {_fmt_delta('borrowing_base', result.net_change)}. "
        f"Drivers, in order: {drivers}. The bridge reconciles to the closing "
        f"borrowing base within {result.reconciliation['tolerance']:.2f} "
        f"(difference {result.reconciliation['difference']:+.2f}).")
    headroom = result.headroom or {}
    if "borrowing_base_headroom" in reading.measures or "headroom" in _normalise(request.question):
        if headroom.get("available"):
            answer += (f" Headroom: {_routing._gbp(headroom['openingHeadroom'])} "
                       f"+ Δ borrowing base {_fmt_delta('borrowing_base', headroom['changeInBorrowingBase'])} "
                       f"− Δ drawn {_fmt_delta('borrowing_base', headroom['changeInDrawn'])} "
                       f"= {_routing._gbp(headroom['closingHeadroom'])}.")
        else:
            answer += f" {headroom.get('reason')}"
    warnings = _disclosures(envelope) + notes + list(result.notes)
    warnings.append(_applicability_note(facility))
    warnings += [n for p in (opening, closing) for n in p.notes]
    return _finish(request, envelope, reading, answer=answer, artifacts=[artifact],
                   warnings=list(dict.fromkeys(warnings)),
                   extra={"intent": _INTENT_BRIDGE, "bridge": result.to_dict(),
                          "opening": opening.to_dict(), "closing": closing.to_dict()})


__all__ = ["ROUTE_NAME", "ROUTE_PRIORITY", "ROUTE_CONFIDENCE", "Reading",
           "read", "recognise", "recogniser", "handle", "RECOGNITION_KEY"]
