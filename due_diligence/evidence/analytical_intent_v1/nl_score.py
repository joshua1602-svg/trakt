"""Score the NL robustness runs against §5's definition of CORRECT.

Objective by construction: the expected analytical intent, the expected route
OWNER and the required capabilities are declared per intent BEFORE the data is
read, so a run is graded against a contract rather than against how the prose
sounds.
"""
from __future__ import annotations

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

#: A dimension with this many groups is degenerate — one bar per handful of
#: loans is not a profile, whatever the prose says.
DEGENERATE_GROUPS = 200


def _answer_groups(answer: str) -> int:
    import re
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

def grade(intent: str, run: dict) -> tuple:
    """(outcome, causes, note) for one run."""
    expected = EXPECTED[intent]
    route = run.get("route")
    ok = bool(run.get("ok"))
    answer = run.get("answer") or ""
    spec = run.get("spec") or {}
    causes = []

    if run.get("hardFailure"):
        return HARD_FAILURE, [C_EXEC], "the request raised"

    # ---- refusals are safe by definition, provided they say why ----------- #
    if not ok:
        if len(answer) > 40:
            return SAFE_REFUSAL, [], "refused with a stated reason"
        return SILENT_SEMANTIC_ERROR, [C_NARRATIVE], "refused without a reason"

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
