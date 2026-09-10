"""Semantic equivalence — when do two readings mean the same thing?

Paraphrase invariance is the property this architecture lives or dies on. Three
wordings of one question must compile to the same authorised work, or the
control plane is not deterministic in the way that matters to a reader.

The hard part is what to IGNORE. Two paraphrases legitimately differ in:

    the question text
    the evidence spans quoted from it
    the order dimensions and filters happen to be listed in
    the model id and token counts of the call that read them

and none of that changes what Trakt would compute. A comparison that noticed
them would report a failure every time, and would be measuring wording — the
very thing the interpretation layer exists to absorb.

What must NOT differ is the authorised content: capability, operation,
population, bound fields, statistics, weights, filters, geography, period,
outputs. That is exactly the plan's own content hash, so equivalence here is not
a second opinion about the plan — it IS the plan, minus provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .outcomes import CompileResult, OUTCOME_PLAN
from .plan import GovernedQueryPlan


def plan_fingerprint(plan: GovernedQueryPlan) -> str:
    """The authorised content of a plan, as one comparable identity."""
    return plan.plan_id


def outcome_fingerprint(result: CompileResult) -> Tuple[str, ...]:
    """A comparable identity for ANY compile result, plan or not.

    A refusal is a governed answer, so two paraphrases that both refuse for the
    same reason ARE invariant. Scoring them as a disagreement would penalise the
    system for being consistently correct about what it cannot do.
    """
    if result.outcome == OUTCOME_PLAN and result.plan is not None:
        return (OUTCOME_PLAN, plan_fingerprint(result.plan))
    return (result.outcome,) + tuple(sorted(set(result.codes())))


@dataclass(frozen=True)
class EquivalenceReport:
    """Whether a set of paraphrases agreed, and where they did not."""

    canonical_id: str
    fingerprints: Tuple[Tuple[str, Tuple[str, ...]], ...]
    invariant: bool
    divergent_slots: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"canonical_id": self.canonical_id, "invariant": self.invariant,
                "fingerprints": {qid: list(fp) for qid, fp in self.fingerprints},
                "divergent_slots": list(self.divergent_slots)}


#: The plan slots a divergence is attributed to, in the order a reader would
#: want them: the coarsest disagreement first, because a capability mismatch
#: explains every downstream one and reporting all of them equally would bury it.
_SLOTS: Tuple[str, ...] = (
    "outcome", "capability", "operation", "population", "period", "geography",
    "filters", "outputs",
)


def _slot_values(result: CompileResult) -> Dict[str, Any]:
    if result.outcome != OUTCOME_PLAN or result.plan is None:
        return {"outcome": (result.outcome,) + tuple(sorted(set(result.codes())))}
    plan = result.plan
    return {
        "outcome": (OUTCOME_PLAN,),
        "capability": plan.capability,
        "operation": plan.operation,
        "population": (plan.population.base, plan.population.lens,
                       plan.population.seasoning,
                       tuple(sorted((p.canonical_field or "", p.comparator,
                                     str(p.value))
                                    for p in plan.population.scope_predicates))),
        "period": (plan.period.form, plan.period.grain, plan.period.periods_back,
                   plan.period.contract),
        "geography": ((plan.geography.canonical_field, plan.geography.group_by,
                       tuple(sorted(map(str, plan.geography.values))))
                      if plan.geography else None),
        "filters": tuple(sorted((f.canonical_field or f.concept, f.comparator,
                                 str(f.value)) for f in plan.filters)),
        "outputs": tuple(sorted(
            (tuple(sorted((m.canonical_field or m.concept, m.statistic,
                           m.weight_field or "") for m in o.measures)),
             tuple(sorted(d.canonical_field or d.concept for d in o.dimensions)),
             tuple(sorted((f.canonical_field or f.concept, f.comparator,
                           str(f.value)) for f in o.filters)),
             (o.geography.canonical_field if o.geography else None))
            for o in plan.outputs)),
    }


def compare_results(canonical_id: str,
                    results: Sequence[Tuple[str, CompileResult]]) -> EquivalenceReport:
    """Do these paraphrases mean the same thing?

    ``results`` is ``[(question_id, CompileResult), ...]`` for one canonical.
    """
    fingerprints = tuple((qid, outcome_fingerprint(result))
                         for qid, result in results)
    invariant = len({fp for _, fp in fingerprints}) <= 1
    divergent: List[str] = []
    if not invariant:
        slot_values = [_slot_values(result) for _, result in results]
        outcomes = {values.get("outcome") for values in slot_values}
        if len(outcomes) > 1:
            # One variant planned and another refused. Every other slot then
            # "differs" only because a refusal has none of them, and listing
            # them all would bury the one fact that matters.
            divergent.append("outcome")
        else:
            for slot in _SLOTS:
                if slot == "outcome":
                    continue
                seen = {repr(values.get(slot)) for values in slot_values}
                if len(seen) > 1:
                    divergent.append(slot)
        if not divergent:
            # The plans differ but no named slot explains it. Reporting "they
            # disagree, we cannot say where" is honest; reporting nothing would
            # look like agreement.
            divergent.append("plan_identity_unexplained")
    return EquivalenceReport(canonical_id=canonical_id, fingerprints=fingerprints,
                             invariant=invariant,
                             divergent_slots=tuple(divergent))


# --------------------------------------------------------------------------- #
# Scoring one intent against an expected semantic fixture
# --------------------------------------------------------------------------- #

#: The material semantic dimensions scored independently. Independence matters:
#: a reading that gets the population right and the statistic wrong is a
#: different finding from one that inverted both, and a single pass/fail would
#: report them identically.
SCORED_DIMENSIONS: Tuple[str, ...] = (
    "capability", "operation", "measures", "statistic", "weight", "population",
    "filters", "dimensions", "geography_basis", "geography_level",
    "temporal", "comparison", "output_structure",
)


def _measure_terms(intent) -> Tuple[str, ...]:
    terms = [m.concept for m in intent.measures]
    for output in intent.effective_outputs():
        terms.extend(m.concept for m in output.measures)
    return tuple(sorted(set(terms)))


def _statistics(intent) -> Tuple[str, ...]:
    stats = [m.statistic for m in intent.measures if m.statistic]
    for output in intent.effective_outputs():
        stats.extend(m.statistic for m in output.measures if m.statistic)
    return tuple(sorted(set(stats)))


def _weights(intent) -> Tuple[str, ...]:
    weights = [m.weight for m in intent.measures if m.weight]
    for output in intent.effective_outputs():
        weights.extend(m.weight for m in output.measures if m.weight)
    return tuple(sorted(set(weights)))


def _filter_keys(intent) -> Tuple[Tuple[Any, ...], ...]:
    keys = [f.key() for f in intent.filters]
    for output in intent.effective_outputs():
        keys.extend(f.key() for f in output.filters)
    return tuple(sorted(keys))


def _dimension_terms(intent) -> Tuple[str, ...]:
    terms = list(intent.dimensions)
    for output in intent.effective_outputs():
        terms.extend(output.dimensions)
    return tuple(sorted(set(terms)))


def observed_dimensions(intent) -> Dict[str, Any]:
    """The scored dimensions as this intent actually states them."""
    geography = intent.geography
    if not geography.requested:
        for output in intent.effective_outputs():
            if output.geography.requested:
                geography = output.geography
                break
    return {
        "capability": intent.capability,
        "operation": intent.operation,
        "measures": _measure_terms(intent),
        "statistic": _statistics(intent),
        "weight": _weights(intent),
        "population": (intent.population.base, intent.population.lens,
                       intent.population.seasoning),
        "filters": _filter_keys(intent),
        "dimensions": _dimension_terms(intent),
        "geography_basis": geography.basis if geography.requested else None,
        "geography_level": geography.level if geography.requested else None,
        "temporal": intent.time.form,
        "comparison": intent.comparison.kind,
        "output_structure": len(intent.effective_outputs()),
    }


def observed_dimensions_from_plan(plan: GovernedQueryPlan) -> Dict[str, Any]:
    """The scored dimensions as the COMPILED PLAN states them.

    Preferred over the intent wherever a plan exists. A question that says "the
    balance" without saying "total" leaves ``statistic`` empty in the intent and
    the governed registry supplies ``sum`` — scoring the intent would mark that
    reading wrong for being terse, when what Trakt would compute is exactly
    right. What the system would do is the thing worth measuring.
    """
    measures, statistics, weights = [], [], []
    dimensions, filters = [], [f.key() if hasattr(f, "key") else
                              (f.concept, f.comparator,
                               tuple(sorted(map(str, f.value)))
                               if isinstance(f.value, (list, tuple)) else f.value)
                              for f in plan.filters]
    geography = plan.geography
    for output in plan.outputs:
        for measure in output.measures:
            measures.append(measure.concept)
            statistics.append(measure.statistic)
            if measure.weight_concept:
                weights.append(measure.weight_concept)
        dimensions.extend(d.concept for d in output.dimensions)
        filters.extend((f.concept, f.comparator,
                        tuple(sorted(map(str, f.value)))
                        if isinstance(f.value, (list, tuple)) else f.value)
                       for f in output.filters)
        if geography is None and output.geography is not None:
            geography = output.geography
    return {
        "capability": plan.capability,
        "operation": plan.operation,
        "measures": tuple(sorted(set(measures))),
        # 'capability' is the statistic of a specialist measure — the owner
        # decides — and is not a statistic anybody chose, so it is not scored
        # as one.
        "statistic": tuple(sorted({s for s in statistics if s != "capability"})),
        "weight": tuple(sorted(set(weights))),
        "population": (plan.population.base, plan.population.lens,
                       plan.population.seasoning),
        "filters": tuple(sorted(filters, key=repr)),
        "dimensions": tuple(sorted(set(dimensions))),
        "geography_basis": geography.requested_basis if geography else None,
        "geography_level": geography.requested_level if geography else None,
        "temporal": plan.period.form,
        "comparison": plan.comparison_kind,
        "output_structure": len(plan.outputs),
    }


def _canonical_concept(term: Any, vocabulary) -> str:
    """A fixture's word, resolved to the identifier the registry gives it.

    The expectations are written in business words ("balance", "current_ltv")
    and a plan binds governed identifiers ("current_outstanding_balance"). Both
    are put through the SAME index before comparison, so the score measures
    whether the reading agrees about the CONCEPT — not whether the fixture
    happened to spell it the way the registry does.
    """
    text = str(term).strip().lower()
    if vocabulary is None:
        return text
    concept = vocabulary.resolve(text)
    return concept.concept_id if concept is not None else text


def _normalise_expected(key: str, value: Any, vocabulary=None) -> Any:
    if value is None:
        return None
    if key in ("measures", "dimensions", "weight"):
        return tuple(sorted(_canonical_concept(v, vocabulary) for v in value))
    if key == "statistic":
        return tuple(sorted(str(v).strip().lower() for v in value))
    if key == "population":
        return tuple(value)
    if key == "filters":
        return tuple(sorted(
            (_canonical_concept(f["concept"], vocabulary),
             str(f["comparator"]).lower(),
             tuple(sorted(map(str, f["value"])))
             if isinstance(f.get("value"), (list, tuple)) else f.get("value"))
            for f in value))
    if key == "output_structure":
        return int(value)
    return value


def score_intent(intent, expected: Mapping[str, Any], *,
                 plan: Optional[GovernedQueryPlan] = None,
                 vocabulary=None) -> Dict[str, Optional[bool]]:
    """Per-dimension truth against a human-reviewable fixture.

    A dimension the fixture does not state is scored ``None`` — NOT correct.
    A harness that cannot tell "checked and right" from "did not look" reports
    confidence it has not earned, which is the failure mode this repository has
    already written down once and does not need to learn twice.
    """
    observed = (observed_dimensions_from_plan(plan) if plan is not None
                else observed_dimensions(intent))
    scored: Dict[str, Optional[bool]] = {}
    for key in SCORED_DIMENSIONS:
        if key not in expected:
            scored[key] = None
            continue
        scored[key] = observed.get(key) == _normalise_expected(
            key, expected[key], vocabulary)
    return scored
