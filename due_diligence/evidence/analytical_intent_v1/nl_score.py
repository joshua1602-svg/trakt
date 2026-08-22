"""Score the NL robustness runs against §5's definition of CORRECT.

Objective by construction: the expected analytical intent, the expected route
OWNER and the required capabilities are declared per intent BEFORE the data is
read, so a run is graded against a contract rather than against how the prose
sounds.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# What each of the nine intents must produce. Declared, not inferred.
# --------------------------------------------------------------------------- #
# owner: "analytical" -> the analytical capability layer must claim it
#        "<route>"    -> an existing route OWNS it by the documented deference
#                        rule, and claiming it with the analytical layer would
#                        be the defect, not the other way round.
EXPECTED = {
    "Q1": {"owner": "analytical", "intent": "origination_profile_change",
           "capabilities": {"period_movement", "population_profile"},
           "needs": ("population:front_book", "period:two")},
    "Q2": {"owner": "forecast_extrapolation", "intent": None,
           "capabilities": set(), "needs": ("threshold",)},
    "Q3": {"owner": "analytical", "intent": "pipeline_offer_outlook",
           "capabilities": {"pipeline_stock", "pipeline_completion_forecast"},
           "needs": ("population:offer", "forecast", "timing")},
    "Q4": {"owner": "forecast_extrapolation", "intent": None,
           "capabilities": set(), "needs": ("run_rate",)},
    "Q5": {"owner": "risk_limits", "intent": None,
           "capabilities": set(), "needs": ("limits",)},
    "Q6": {"owner": "risk_limits", "intent": None,
           "capabilities": set(), "needs": ("limits", "ranking")},
    "Q7": {"owner": "analytical", "intent": "vintage_risk_comparison",
           "capabilities": {"portfolio_snapshot", "vintage_analysis"},
           "needs": ("comparison", "cohort")},
    "Q8": {"owner": "analytical", "intent": "population_movement_comparison",
           "capabilities": {"period_movement"},
           "needs": ("comparison", "period:two")},
    "Q9": {"owner": "analytical", "intent": "funded_balance_outlook",
           "capabilities": {"funded_balance_forecast",
                            "pipeline_completion_forecast"},
           "needs": ("forecast",)},
}

ROUTE_ANALYTICAL = "analytical_composition"

# --------------------------------------------------------------------------- #
# Outcome vocabulary (brief §6)
# --------------------------------------------------------------------------- #
CORRECT = "CORRECT"
CORRECT_DISCLOSED = "CORRECT_WITH_DISCLOSED_LIMITATION"
HONEST_PARTIAL = "HONEST_PARTIAL"
SAFE_REFUSAL = "SAFE_REFUSAL"
#: A refusal of a question the product could have answered. NOT a wrong number,
#: and NOT a success: it is a question the reader was entitled to an answer to.
#: Split out of SAFE_REFUSAL because the only test used to be `len(answer) > 40`
#: — "safe" meant the refusal sentence was longer than forty characters, so a
#: refusal that declined while HOLDING the answer graded identically to one that
#: was right to decline.
UNHELPFUL_REFUSAL = "UNHELPFUL_REFUSAL"
#: Answered, and a figure this grader can check against the book does not match.
#: The one outcome that reaches a reader as fact.
WRONG_FIGURE = "WRONG_FIGURE"
INCORRECT_SUCCESSFUL = "INCORRECT_SUCCESSFUL"
SILENT_SEMANTIC_ERROR = "SILENT_SEMANTIC_ERROR"
HARD_FAILURE = "HARD_FAILURE"

# --------------------------------------------------------------------------- #
# Cause vocabulary (operator's list)
# --------------------------------------------------------------------------- #
C_PARSER = "parser / business semantics"
C_PERIOD = "comparison-period recognition"
C_POPULATION = "population recognition"
C_CONTENTION = "route contention"
C_PLANNING = "capability planning"
C_EXEC = "deterministic execution"
C_GUARD = "guard coverage"
C_NARRATIVE = "narrative/presentation"
C_POPULATION_DROPPED = "a stated narrowing did not reach execution"

#: A dimension with this many groups is degenerate — one bar per handful of
#: loans is not a profile, whatever the prose says.
DEGENERATE_GROUPS = 200


#: A sentence stating a row-level narrowing. Deliberately BROADER than either
#: production comparator vocabulary, because this is a grader: it must be able
#: to see a threshold the product missed. If it only knew the words production
#: knows, it could never report the words production does not.
_THRESHOLD_IN_QUESTION = re.compile(
    r"\b(?:over|above|more than|greater than|bigger than|larger than|"
    r"higher than|exceeding|in excess of|at least|no less than|under|below|"
    r"less than|fewer than|smaller than|lower than|at most|no more than|"
    r"up to|between|older than|younger than)\b[^.?!]{0,20}?"
    r"[£$]?\s?\d", re.IGNORECASE)


def question_states_a_narrowing(question: str) -> bool:
    """Whether the SENTENCE asks for fewer rows than the whole book."""
    return bool(_THRESHOLD_IN_QUESTION.search(question or ""))


def check_population(question: str, run: dict) -> "tuple | None":
    """(cause, note) when a narrowing the sentence states did not reach the
    figure, else None.

    THE ONE FIGURE CHECK THIS GRADER CLAIMS. Scoped on purpose. Six of the nine
    declared intents ask for a forward figure, a run rate or a limit headroom,
    and there is no expression for the right answer to "when will we reach
    £100m?" — a grader that claimed to verify those would be the same defect
    this one is being extended to fix. What IS checkable for any question is how
    many loans the answer covers, and that is exactly the shape that returned
    the whole-book LTV for "loans bigger than £150k".
    """
    if not question_states_a_narrowing(question):
        return None
    summary = run.get("executionSummary") or {}
    population = summary.get("population")
    total = summary.get("populationTotal")
    if population is None or total is None:
        return None
    if summary.get("filtersApplied"):
        return None
    if population != total:
        return None
    return (C_POPULATION_DROPPED,
            "the question states a threshold; the answer covers the whole book "
            "(%s of %s loans) with no filter applied" % (population, total))


def _answer_groups(answer: str) -> int:
    m = re.search(r"covering\s+([\d,]+)\s+group", answer or "")
    return int(m.group(1).replace(",", "")) if m else 0



# --------------------------------------------------------------------------- #
# "Materially answers" — added AFTER inspecting cases, and it makes grading
# FAIRER, not more flattering.
#
# A route other than the intended one can still answer the question. Q7.3
# ("how different is the risk profile of recent originations versus the back
# book?") was answered by the generic executor grouping by seasoning_segment
# with six governed measures across both sides — that IS the comparison the
# reader asked for, reached by a different mechanism. Condemning it because the
# analytical layer did not claim it would be scoring the plumbing, not the
# answer.
#
# The test is structural, not impressionistic: it reads the SPEC that executed
# and asks whether the population, the measures and the period structure the
# question needs are actually present.
# --------------------------------------------------------------------------- #
_SEASONING_KEYS = ("seasoning_segment", "seasoning_bucket", "months_on_book",
                   "vintage_year")


def materially_answers(intent: str, run: dict) -> bool:
    spec = run.get("spec") or {}
    answer = (run.get("answer") or "").lower()
    dim = str(spec.get("dimension") or "")
    filters = spec.get("filters") or {}
    measures = {m.get("field") for m in (spec.get("measures") or [])
                if isinstance(m, dict)}
    temporal = bool(spec.get("temporal_mode") or spec.get("cohort_progression"))
    groups = _answer_groups(run.get("answer") or "")

    if intent == "Q1":
        # profile of NEW originations, ACROSS periods. Both halves required.
        population = (dim in _SEASONING_KEYS
                      or any(k in _SEASONING_KEYS for k in filters))
        return population and temporal and not (groups >= DEGENERATE_GROUPS)
    if intent == "Q7":
        # front vs older, on risk measures, at one date. No period needed.
        split = dim in _SEASONING_KEYS or any(k in _SEASONING_KEYS for k in filters)
        risk = bool(measures & {"current_loan_to_value", "current_interest_rate",
                                "youngest_borrower_age", "months_on_book"})
        return split and risk and groups >= 2
    if intent == "Q8":
        # two named populations, moving over time.
        return temporal and groups >= 2
    if intent in ("Q3", "Q9"):
        # a forward expectation is required; the point-in-time executor has none.
        return bool(spec.get("forecast_mode")) or "forecast" in answer
    if intent == "Q4":
        return "run-rate" in answer or "run rate" in answer
    if intent == "Q2":
        return "reach" in answer or "milestone" in answer
    if intent in ("Q5", "Q6"):
        return "limit" in answer and ("headroom" in answer or "breach" in answer)
    return False

def held_the_answer(run: dict) -> bool:
    """Whether execution resolved what it needed and declined anyway.

    A measure plus either a population or a filter means the figure was in hand.
    B1 — "What is the LTV for loan tickets above £150k?" — applied
    `Balance > 150000`, reached 5,857 loans, resolved Current LTV, and then
    asked how the reader meant the word "ticket".
    """
    summary = run.get("executionSummary") or {}
    if not summary.get("measure"):
        return False
    return bool(summary.get("filtersApplied") or summary.get("population"))


def grade_refusal(run: dict, siblings_answered: bool = False) -> tuple:
    """SAFE or UNHELPFUL, as a rule rather than a judgement.

    The old test was `len(answer) > 40` and nothing else, so "safe" meant the
    refusal sentence was longer than forty characters. Two mechanical tests
    replace it, and neither is impressionistic:

      * the record shows execution HELD the answer, or
      * another phrasing of the SAME intent answered — four variations of one
        question answering while the fifth refuses is a phrasing gap, not a
        capability limit.

    Everything else stays SAFE. A refusal is presumed honest; it is the
    EVIDENCE of avoidable refusal that has to be produced, not the reverse.
    """
    answer = run.get("answer") or ""
    if len(answer) <= 40:
        return SILENT_SEMANTIC_ERROR, [C_NARRATIVE], "refused without a reason"
    if held_the_answer(run):
        summary = run.get("executionSummary") or {}
        return (UNHELPFUL_REFUSAL, [C_GUARD],
                "declined while holding the answer: measure %r, %s loans, "
                "filters %s" % (summary.get("measure"), summary.get("population"),
                                summary.get("filtersApplied")))
    if siblings_answered:
        return (UNHELPFUL_REFUSAL, [C_PARSER],
                "another phrasing of this intent answered; this one refused")
    return SAFE_REFUSAL, [], "refused with a stated reason"


def grade(intent: str, run: dict, question: str = "",
          siblings_answered: bool = False) -> tuple:
    """(outcome, causes, note) for one run.

    `question` and `siblings_answered` are optional so every existing caller
    keeps working; without them the two new checks stay silent rather than
    guessing, because a grader that assumes what it was not told is how this
    one came to report a trebled figure as CORRECT.
    """
    expected = EXPECTED[intent]
    route = run.get("route")
    ok = bool(run.get("ok"))
    answer = run.get("answer") or ""
    spec = run.get("spec") or {}
    causes = []

    if run.get("hardFailure"):
        return HARD_FAILURE, [C_EXEC], "the request raised"

    # ---- refusals: honest, or a question the product could have answered -- #
    if not ok:
        return grade_refusal(run, siblings_answered=siblings_answered)

    # ---- a figure the book can check, checked BEFORE the route is judged -- #
    #
    # Before anything else about how the answer was reached: does it cover the
    # rows the sentence asked for? A right route over the wrong population is
    # still a wrong number in front of a reader, and grading the plumbing first
    # is how "entire funded portfolio · 11,035 loans" came back CORRECT.
    dropped = check_population(question, run)
    if dropped is not None:
        cause, why = dropped
        return WRONG_FIGURE, [cause, C_PARSER], why

    # ---- a route-owned intent: the OWNING route is the correct outcome ---- #
    if expected["owner"] != "analytical":
        if route == expected["owner"]:
            return CORRECT, [], f"answered by its owning route {route}"
        if route == ROUTE_ANALYTICAL:
            return (INCORRECT_SUCCESSFUL, [C_CONTENTION],
                    "the analytical layer claimed a route-owned intent")
        # Diverted elsewhere with ok=True. Graded on the SAME rule as an
        # analytical intent: a guard that said ok makes it silent, which is the
        # more severe classification, not the more forgiving one.
        if materially_answers(intent, run):
            return (CORRECT_DISCLOSED, [C_CONTENTION],
                    f"answered by {route or 'the generic executor'} rather than "
                    f"{expected['owner']}, but materially answers the question")
        causes = [C_CONTENTION, C_PARSER]
        if run.get("semanticGuard") == "ok":
            causes.append(C_GUARD)
            return (SILENT_SEMANTIC_ERROR, causes,
                    f"answered by {route or 'the generic executor'} instead of "
                    f"{expected['owner']}; guard passed with "
                    f"{len(run.get('guardFacets') or [])} facet(s)")
        return (INCORRECT_SUCCESSFUL, causes,
                f"answered by {route or 'the generic executor'}, not "
                f"{expected['owner']}")

    # ---- an analytical intent -------------------------------------------- #
    if route == ROUTE_ANALYTICAL:
        got = set(run.get("capabilities") or [])
        if run.get("intent") != expected["intent"]:
            return (INCORRECT_SUCCESSFUL, [C_PLANNING],
                    f"planned {run.get('intent')}, expected {expected['intent']}")
        missing = expected["capabilities"] - got
        if missing:
            return (HONEST_PARTIAL, [C_PLANNING],
                    f"missing capability {sorted(missing)}")
        return CORRECT, [], f"analytical plan {run.get('intent')}"

    # Another route (or the generic executor) answered an ANALYTICAL intent
    # with ok=True. That is the dangerous case: a confident answer to a
    # different analytical question.
    if materially_answers(intent, run):
        return (CORRECT_DISCLOSED, [C_CONTENTION],
                f"answered by {route or 'the generic executor'} rather than the "
                "analytical layer, but materially answers the question")
    groups = _answer_groups(answer)
    degenerate = groups >= DEGENERATE_GROUPS
    causes.append(C_CONTENTION)
    if spec.get("dimension") in ("origination_date", "maturity_date") or degenerate:
        causes.append(C_PARSER)
    if run.get("semanticGuard") == "ok":
        causes.append(C_GUARD)
    facets = [f.get("kind") for f in (run.get("guardFacets") or [])]
    if "comparison_period" not in facets and "period" in str(expected["needs"]):
        causes.append(C_PERIOD)
    if not run.get("populationApplied") and "population" in str(expected["needs"]):
        causes.append(C_POPULATION)
    note = (f"route={route or 'generic executor'}, "
            f"dimension={spec.get('dimension')}, groups={groups}, "
            f"guard={run.get('semanticGuard')}")
    if run.get("semanticGuard") == "ok":
        return SILENT_SEMANTIC_ERROR, causes, note
    return INCORRECT_SUCCESSFUL, causes, note
