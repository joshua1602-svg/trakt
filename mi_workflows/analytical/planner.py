"""mi_workflows.analytical.planner — question → analytical plan.

A plan says WHICH governed capabilities are needed and why. It never says what
any number is. Two producers, one validator:

* :func:`plan_for` — deterministic. Recognises the analytical question shapes
  from the structured parse the platform already produced plus a small
  controlled vocabulary, exactly as every other recogniser does. Always
  available, and the only producer used when no model is configured.
* :func:`plan_from_proposal` — accepts a plan a language model proposed and
  puts it through :func:`validate` before anything executes. An unknown
  capability, a missing required input or an input the capability does not
  declare rejects the whole proposal and the deterministic plan stands. This is
  the same discipline the MI Query parser applies to an LLM-proposed
  ``MIQuerySpec``: the model may propose, the deterministic stack disposes.

The deference rule
------------------
:func:`plan_for` returns ``None`` — meaning "this is not for us" — whenever the
work is a single capability an existing route already owns. The layer engages
only where composition is genuinely required, so every question the product
answers correctly today keeps the answer it has.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mi_agent import portfolio_lens as _portfolio_lens
from mi_agent import seasoning as _seasoning

from .contract import (
    AnalyticalPlan,
    CapabilityCall,
    KIND_COHORT,
    KIND_COMPARISON,
    KIND_COMPOSITION,
    KIND_FORECAST,
    KIND_MEASURE,
    KIND_MOVEMENT,
)
from .executors import (
    PROFILE_DIMENSIONS, PROFILE_MEASURES, PROFILE_MOVEMENT_MEASURES,
    WINDOW_LATEST_PAIR, WINDOW_RETAINED,
)
from .registry import CAPABILITIES, CapabilityRegistry, UnknownCapabilityError
from . import populations as pops

# --------------------------------------------------------------------------- #
# Controlled vocabulary — layer-local, so no shared recogniser's vocabulary
# shifts underneath a route that already works.
# --------------------------------------------------------------------------- #
_PROFILE_TERMS = ("profile", "characteristics", "mix", "composition",
                  "what kind of", "what sort of", "make-up", "makeup")

_CHANGE_TERMS = ("changed", "change", "changing", "moved", "movement",
                 "shifted", "shift", "trend", "evolved", "evolving",
                 "grown", "growth", "grew", "declined", "fallen", "risen")

#: Comparative framing. Deliberately includes "relative to", which the shared
#: portfolio-lens vocabulary does not carry — widening that vocabulary would
#: change how every other route reads a question.
_COMPARATIVE_TERMS = (" relative to ", " compared with ", " compared to ",
                      " versus ", " vs ", " vs. ", " against ", "compare",
                      "comparison", " than ", " how does ", " how do ")

_RISK_PROFILE_TERMS = ("risk profile", "credit profile", "risk characteristics",
                       "how risky", "riskiness")

_VINTAGE_TERMS = ("vintage", "vintages", "seasoned", "seasoning", "older loans",
                  "cohort", "cohorts")

_OFFER_TERMS = ("offer", "offers", "offered")
_PIPELINE_TERMS = ("pipeline", "kfi", "application", "applications",
                   "outstanding offers", "case", "cases")
_COMPLETION_TERMS = ("complete", "completes", "completing", "completion",
                     "completions", "convert", "converting", "conversion",
                     "drawdown", "draw down", "fund", "funded")
_TIMING_TERMS = ("when", "timing", "by when", "how soon", "which month")

_FORECAST_BALANCE_TERMS = ("forecast funded balance", "forecast balance",
                           "projected funded balance", "projected balance",
                           "expected funded balance", "forecast aum",
                           "where will the book", "what will the book be")

_RECENT_TERMS = ("last few months", "recent months", "past few months",
                 "recently", "last several months", "over the last few")

#: A question naming one of these is asking for a raw tape or an export, not an
#: analysis. Declined outright so the layer never intercepts a data request.
_EXPORT_TERMS = ("export", "download", "csv", "loan-level list", "list the loans",
                 "raw tape")


def _norm(question: str) -> str:
    """Lower-cased, single-spaced and space-padded, with punctuation neutralised.

    Padding lets a vocabulary entry be matched as a whole phrase (" vs " never
    matches inside "versus"). Neutralising punctuation matters for the SAME
    reason a category has to be matched as a whole word: "…relative to London?"
    ends in a question mark, and without this "london" would fail to match the
    category the book actually carries.
    """
    text = "".join(c if (c.isalnum() or c in "&'-£%") else " "
                   for c in str(question or "").lower())
    return " " + " ".join(text.split()) + " "


def _any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _is_comparative(text: str) -> bool:
    return _any(text, _COMPARATIVE_TERMS)


# --------------------------------------------------------------------------- #
# Population-pair resolution — governed sides only
# --------------------------------------------------------------------------- #
def resolve_population_pair(question: str, frame=None, semantics=None
                            ) -> Optional[Tuple[pops.PopulationSpec,
                                                pops.PopulationSpec]]:
    """Two governed populations a comparative question names, or ``None``.

    Strategies, in precedence order, each delegating to the resolver that owns
    the concept. A question that names no two governed populations returns
    ``None`` and the plan says so — nothing is approximated.

    ``frame`` may be a dataframe or a zero-argument callable returning one. Only
    the last strategy needs data, and recognition runs on every routed question,
    so a callable is resolved ONLY if the first two strategies both decline.
    """
    text = _norm(question)

    # 1. Provenance — the governed portfolio lens owns direct / acquired.
    lenses = _portfolio_lens.resolve_comparison_lenses(question)
    if len(lenses) >= 2:
        return (pops.provenance_population(lenses[0].name),
                pops.provenance_population(lenses[1].name))
    if _is_comparative(text):
        has_direct = _portfolio_lens._contains_any(text, _portfolio_lens._DIRECT_TERMS)
        has_acquired = _portfolio_lens._contains_any(text,
                                                     _portfolio_lens._ACQUIRED_TERMS)
        if has_direct and has_acquired:
            return (pops.provenance_population(_portfolio_lens.LENS_DIRECT),
                    pops.provenance_population(_portfolio_lens.LENS_ACQUIRED))

    # 2. Seasoning — the governed binary partition.
    named = _seasoning.segments_named(question)
    if len(named) >= 2:
        return (pops.seasoning_population(named[0]),
                pops.seasoning_population(named[1]))
    if len(named) == 1 and _is_comparative(text):
        return (pops.seasoning_population(named[0]),
                pops.seasoning_population(pops.other_seasoning_segment(named[0])))

    # 3. Two values of one governed dimension, matched literally against the
    #    values the book actually carries. A category the book does not hold is
    #    never approximated to a neighbouring one.
    pair = _dimension_value_pair(text, frame, semantics)
    if pair:
        return pair
    return None


def _dimension_value_pair(text: str, frame, semantics
                          ) -> Optional[Tuple[pops.PopulationSpec,
                                              pops.PopulationSpec]]:
    if frame is None or not _is_comparative(text):
        return None
    if callable(frame):
        try:
            frame = frame()
        except Exception:  # noqa: BLE001 - no frame simply means no pair
            return None
        if frame is None:
            return None
    fields = (semantics or {}).get("fields", {}) if isinstance(semantics, dict) else {}
    columns = set(getattr(frame, "columns", []))
    candidates: List[Tuple[int, str, str, List[str]]] = []
    for key in sorted(fields):
        entry = fields[key] or {}
        if entry.get("role") != "dimension":
            continue
        column = entry.get("canonical_field", key)
        if column not in columns:
            continue
        try:
            values = [str(v) for v in frame[column].dropna().unique()]
        except Exception:  # noqa: BLE001 - an unreadable column is simply skipped
            continue
        if len(values) > 200:
            continue     # an identifier-like column, not a comparison dimension
        # Whole-word matches only, longest first, so "South East" wins over any
        # shorter value it contains and a two-letter code cannot match by luck.
        hits = [v for v in sorted(values, key=len, reverse=True)
                if len(v) >= 3 and f" {v.lower()} " in text]
        if len(hits) >= 2:
            label = entry.get("business_name") or key.replace("_", " ")
            candidates.append((sum(len(h) for h in hits[:2]), column, label, hits))
    if not candidates:
        return None
    # The MOST SPECIFIC match wins — the dimension whose two named values are the
    # longest — with the column name breaking a tie, so the same question always
    # resolves to the same pair whatever order the registry happens to be in.
    _length, column, label, hits = max(candidates, key=lambda c: (c[0], c[1]))
    return (pops.dimension_population(column, hits[0], f"{label} {hits[0]}"),
            pops.dimension_population(column, hits[1], f"{label} {hits[1]}"))


# --------------------------------------------------------------------------- #
# Deterministic planning
# --------------------------------------------------------------------------- #
#: Plan intents. One per composite question shape the layer serves.
INTENT_ORIGINATION_PROFILE_CHANGE = "origination_profile_change"
INTENT_PIPELINE_OFFER_OUTLOOK = "pipeline_offer_outlook"
INTENT_VINTAGE_RISK_COMPARISON = "vintage_risk_comparison"
INTENT_POPULATION_MOVEMENT_COMPARISON = "population_movement_comparison"
INTENT_FUNDED_BALANCE_OUTLOOK = "funded_balance_outlook"

ALL_INTENTS: Tuple[str, ...] = (
    INTENT_ORIGINATION_PROFILE_CHANGE, INTENT_PIPELINE_OFFER_OUTLOOK,
    INTENT_VINTAGE_RISK_COMPARISON, INTENT_POPULATION_MOVEMENT_COMPARISON,
    INTENT_FUNDED_BALANCE_OUTLOOK,
)


def plan_for(question: str, *, spec: Any = None, frame=None,
             semantics: Optional[Mapping[str, Any]] = None,
             registry: CapabilityRegistry = CAPABILITIES
             ) -> Optional[AnalyticalPlan]:
    """The deterministic analytical plan for a question, or ``None`` to defer.

    ``None`` means one of two things, and both are correct outcomes: the
    question is not analytical composition at all, or it is work a single
    existing route already owns.
    """
    text = _norm(question)
    if _any(text, _EXPORT_TERMS):
        return None

    for build in (_plan_origination_profile_change, _plan_pipeline_offer_outlook,
                  _plan_funded_balance_outlook, _plan_vintage_risk_comparison,
                  _plan_population_movement_comparison):
        plan = build(question, text, spec, frame, semantics)
        if plan is not None:
            errors = validate(plan, registry)
            if errors:  # a deterministic plan that fails its own contract is a bug
                raise AssertionError("; ".join(errors))
            return plan
    return None


def _plan_origination_profile_change(question, text, spec, frame, semantics
                                     ) -> Optional[AnalyticalPlan]:
    """Q: how has the profile of new originations changed lately?"""
    if not _any(text, _CHANGE_TERMS):
        return None
    if not _any(text, _PROFILE_TERMS):
        return None
    named = _seasoning.segments_named(question)
    if len(named) != 1:
        return None
    if _is_comparative(text):
        return None  # a two-population comparison, not a single-population trend
    population = pops.seasoning_population(named[0])
    pops.assert_not_fabricated([population], question)
    # "the last few months" asks across the retained window, not just the last
    # pair of snapshots. Which snapshots were actually compared is disclosed on
    # the answer either way, so the reader can see the window they were given.
    window = WINDOW_RETAINED if _any(text, _RECENT_TERMS) else WINDOW_LATEST_PAIR
    return AnalyticalPlan(
        intent=INTENT_ORIGINATION_PROFILE_CHANGE,
        required_kinds=(KIND_MOVEMENT,),
        rationale=(f"The question asks how the profile of {population.label} has "
                   "changed, which needs the population's current composition "
                   "and its movement across the retained reporting snapshots."),
        calls=(
            CapabilityCall("period_movement",
                           {"population": population,
                            "measures": list(PROFILE_MOVEMENT_MEASURES),
                            "dimensions": [d for d, _ in PROFILE_DIMENSIONS],
                            "window": window},
                           because="what moved, and how the composition shifted"),
            CapabilityCall("population_profile",
                           {"population": population,
                            "dimensions": [d for d, _ in PROFILE_DIMENSIONS],
                            "top_n": 3},
                           because="the profile as it stands now"),
        ))


def _plan_pipeline_offer_outlook(question, text, spec, frame, semantics
                                 ) -> Optional[AnalyticalPlan]:
    """Q: what is in the pipeline at offer, how much completes, and when?"""
    if not _any(text, _PIPELINE_TERMS):
        return None
    if not _any(text, _OFFER_TERMS):
        return None
    if not (_any(text, _COMPLETION_TERMS) or _any(text, _TIMING_TERMS)):
        return None
    return AnalyticalPlan(
        intent=INTENT_PIPELINE_OFFER_OUTLOOK,
        required_kinds=(KIND_MEASURE, KIND_FORECAST),
        rationale=("The question asks for the offer-stage stock, the share of it "
                   "expected to complete and when — a pipeline reading and a "
                   "completion forecast, composed."),
        calls=(
            CapabilityCall("pipeline_stock", {"stages": ["OFFER"]},
                           because="what is outstanding at offer today"),
            CapabilityCall("pipeline_completion_forecast",
                           {"stages": ["OFFER"]},
                           because="how much of it is expected to complete, and when"),
        ))


def _plan_funded_balance_outlook(question, text, spec, frame, semantics
                                 ) -> Optional[AnalyticalPlan]:
    """Q: given the pipeline, what is the forecast funded balance?"""
    forecast_named = _any(text, _FORECAST_BALANCE_TERMS)
    composed = (_any(text, _PIPELINE_TERMS)
                and _any(text, ("forecast", "project", "expect"))
                and _any(text, ("balance", "book", "aum", "loans")))
    if not (forecast_named or composed):
        return None
    if _any(text, ("run rate", "run-rate")):
        return None  # the run-rate route owns this
    return AnalyticalPlan(
        intent=INTENT_FUNDED_BALANCE_OUTLOOK,
        required_kinds=(KIND_FORECAST,),
        rationale=("The question asks where the book lands given what is in the "
                   "pipeline: the funded position, the expected completions and "
                   "their timing, composed into a projected balance."),
        calls=(
            CapabilityCall("funded_balance_forecast", {},
                           because="funded today, plus probability-weighted completions"),
            CapabilityCall("pipeline_completion_forecast", {},
                           because="when those completions are expected to land"),
        ))


def _plan_vintage_risk_comparison(question, text, spec, frame, semantics
                                  ) -> Optional[AnalyticalPlan]:
    """Q: how does the risk profile of older vintages compare with the front book?"""
    if not _is_comparative(text):
        return None
    if not (_any(text, _RISK_PROFILE_TERMS) or _any(text, _PROFILE_TERMS)):
        return None
    if not _any(text, _VINTAGE_TERMS):
        return None
    pair = resolve_population_pair(question, frame, semantics)
    if pair is None:
        return None
    pops.assert_not_fabricated(list(pair), question)
    measures = list(PROFILE_MEASURES)
    return AnalyticalPlan(
        intent=INTENT_VINTAGE_RISK_COMPARISON,
        required_kinds=(KIND_COMPARISON, KIND_COHORT),
        rationale=(f"The question compares {pair[0].label} with {pair[1].label} on "
                   "the governed risk measures the book carries, and asks for the "
                   "vintage detail behind them."),
        calls=(
            CapabilityCall("portfolio_snapshot",
                           {"measures": measures, "population": pair[0]},
                           because=f"the governed measures for {pair[0].label}"),
            CapabilityCall("portfolio_snapshot",
                           {"measures": measures, "population": pair[1]},
                           because=f"the governed measures for {pair[1].label}"),
            CapabilityCall("vintage_analysis", {},
                           because="the per-vintage detail behind the two sides"),
        ))


def _plan_population_movement_comparison(question, text, spec, frame, semantics
                                         ) -> Optional[AnalyticalPlan]:
    """Q: how has the balance from X changed relative to Y?"""
    if not _is_comparative(text):
        return None
    if not _any(text, _CHANGE_TERMS):
        return None
    pair = resolve_population_pair(question, frame, semantics)
    if pair is None:
        return None
    pops.assert_not_fabricated(list(pair), question)
    measures = ["current_outstanding_balance"]
    if _any(text, _PROFILE_TERMS) or _any(text, _RISK_PROFILE_TERMS):
        measures = list(PROFILE_MOVEMENT_MEASURES)
    return AnalyticalPlan(
        intent=INTENT_POPULATION_MOVEMENT_COMPARISON,
        required_kinds=(KIND_MOVEMENT, KIND_COMPARISON),
        rationale=(f"The question asks how {pair[0].label} moved relative to "
                   f"{pair[1].label}, which is one governed period movement per "
                   "population and a comparison of the two."),
        calls=(
            CapabilityCall("period_movement",
                           {"population": pair[0], "measures": measures,
                            "window": WINDOW_RETAINED},
                           because=f"movement for {pair[0].label}"),
            CapabilityCall("period_movement",
                           {"population": pair[1], "measures": measures,
                            "window": WINDOW_RETAINED},
                           because=f"movement for {pair[1].label}"),
        ))


# --------------------------------------------------------------------------- #
# Validation — every plan, whoever proposed it
# --------------------------------------------------------------------------- #
def validate(plan: AnalyticalPlan,
             registry: CapabilityRegistry = CAPABILITIES) -> List[str]:
    """Everything wrong with a plan, before any data is touched."""
    errors: List[str] = []
    if not plan.calls:
        errors.append("the plan calls no capability")
    for call in plan.calls:
        capability = registry.get(call.capability)
        if capability is None:
            errors.append(f"unknown analytical capability {call.capability!r}")
            continue
        errors.extend(capability.validate_inputs(call.inputs))
    return errors


def plan_from_proposal(proposal: Mapping[str, Any], *, question: str,
                       registry: CapabilityRegistry = CAPABILITIES
                       ) -> Tuple[Optional[AnalyticalPlan], List[str]]:
    """Turn a model's proposed plan into a validated one, or reject it.

    The proposal shape is deliberately small — an intent, a rationale and an
    ordered list of ``{capability, inputs, because}`` — because the model's job
    is to choose governed capabilities, not to describe an analysis. Inputs that
    name a governed population are resolved HERE, through the same governed
    resolvers the deterministic planner uses, so a model can ask for "the front
    book" but cannot invent one.
    """
    errors: List[str] = []
    raw_calls = proposal.get("calls") if isinstance(proposal, Mapping) else None
    if not isinstance(raw_calls, (list, tuple)) or not raw_calls:
        return None, ["the proposal contains no capability calls"]
    calls: List[CapabilityCall] = []
    specs: List[pops.PopulationSpec] = []
    for entry in raw_calls:
        if not isinstance(entry, Mapping):
            errors.append("a capability call is not an object")
            continue
        name = str(entry.get("capability") or "")
        inputs = dict(entry.get("inputs") or {})
        population = inputs.get("population")
        if isinstance(population, str):
            resolved = _population_from_name(population)
            if resolved is None:
                errors.append(f"{name}: {population!r} is not a governed population")
                continue
            inputs["population"] = resolved
            specs.append(resolved)
        elif isinstance(population, pops.PopulationSpec):
            specs.append(population)
        calls.append(CapabilityCall(name, inputs,
                                    because=str(entry.get("because") or "")))
    plan = AnalyticalPlan(
        intent=str(proposal.get("intent") or "llm_proposed"),
        rationale=str(proposal.get("rationale") or ""),
        origin="llm", calls=tuple(calls))
    errors.extend(validate(plan, registry))
    if not errors:
        try:
            pops.assert_not_fabricated(specs, question)
        except pops.FabricatedPopulation as exc:
            errors.append(str(exc))
    return (None if errors else plan), errors


def _population_from_name(name: str) -> Optional[pops.PopulationSpec]:
    """A governed population from a model-supplied name, or ``None``.

    Only names the governed resolvers already own are accepted; there is no
    free-text population.
    """
    key = str(name).strip().lower()
    if key in ("total", "whole book", "the whole book", "entire book"):
        return pops.TOTAL
    segments = _seasoning.segments_named(key)
    if len(segments) == 1:
        return pops.seasoning_population(segments[0])
    if key in ("direct", "the direct book", "acquired", "the acquired book"):
        lens = _portfolio_lens.resolve_lens(key)
        if lens.name in (_portfolio_lens.LENS_DIRECT, _portfolio_lens.LENS_ACQUIRED):
            return pops.provenance_population(lens.name)
    return None
