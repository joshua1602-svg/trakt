#!/usr/bin/env python3
"""A GovernedQueryPlan, handed to the existing deterministic owner. Shadow only.

WHAT THIS IS FOR. `mi_agent.interpretation_v2` produces a signed-off
`GovernedQueryPlan`; `mi_agent.mi_query_executor.execute_mi_query` is the proven
generic calculation owner. Nothing converts between them — `query_plan_adapter`
lifts an `MIQuerySpec` into `mi_agent.query_plan.QueryPlan`, which is a different
object going the other way. This module is the missing seam, and it is
deliberately the thinnest thing that can be called a seam.

IT IS A DISPATCHER, NOT A READER. Every value it places into an `MIQuerySpec` has
already been bound by the deterministic compiler and re-validated against the
governed registry. The module therefore:

    * never sees the question text — it is not a parameter, and a test asserts
      the module's source contains no `question` identifier;
    * never imports `re`, so it cannot pattern-match anything;
    * never defaults a value the plan left empty. A missing value is an
      ineligibility or a refusal, never a guess;
    * never calculates. `execute_mi_query` owns arithmetic, as it always has.

SLICE 1 IS NARROW ON PURPOSE. The eligibility contract below was derived from
recorded evidence, not chosen: of the 135 plans in
`interpretation_v2/evidence/run8_135_signoff_2b00172.json`, exactly the
`generic_analysis` + `point_in_time`/`breakdown` + `current` + at-most-two-axes
shape is expressible in the generic executor's own vocabulary
(`{avg, count, max, median, min, sum, weighted_avg}`) with no scope dimension the
production path resolves elsewhere. Everything else — a specialist capability, a
stated historical period, an explicit Direct/Acquired lens, a governed geography
axis, a second output, a third axis — is INELIGIBLE and says so. Widening this to
raise coverage would be the defect, not the improvement.

THE OLD PATH OWNS THE ANSWER. This module cannot serve a user. It returns a
`ShadowOutcome` for comparison and nothing else; `mi_service` calls it inside a
suppress-everything guard with the flag off by default.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import (Any, Dict, FrozenSet, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.query_plan import AVERAGE, COUNT, SUM, WEIGHTED_AVERAGE
from mi_agent.query_plan_compiler import (_MANY_DIMENSIONS, _RENDERING,
                                          _executor_value)

#: The shadow flag, following the repository's `MI_AGENT_*` environment
#: convention. Two states only: this slice has no SERVE mode to enable.
SHADOW_ENV_VAR = "MI_AGENT_PLAN_SHADOW"
SHADOW_OFF = "off"
SHADOW_ON = "shadow"


def shadow_mode() -> str:
    """`off` (the default) or `shadow`. Anything unrecognised reads as off."""
    value = str(os.environ.get(SHADOW_ENV_VAR) or "").strip().lower()
    return SHADOW_ON if value == SHADOW_ON else SHADOW_OFF


# --------------------------------------------------------------------------- #
# the eligibility contract
# --------------------------------------------------------------------------- #

#: The one capability this slice dispatches. A specialist capability owns its own
#: input contract and its own arithmetic; handing its plan to the generic
#: executor would compute a simpler question than the reader asked.
ELIGIBLE_CAPABILITY = "generic_analysis"

#: Operations the generic executor can express. `distribution` is excluded
#: deliberately: it is owned by `period_change/distribution.py`, not by a
#: groupby.
ELIGIBLE_OPERATIONS = frozenset({"point_in_time", "breakdown"})

#: The only period form this slice accepts. The production path resolves ONE
#: frame per request and `mi_agent_workflow` refuses capabilities measured across
#: snapshots; a stated historical period belongs to the temporal owner, which
#: this slice does not touch.
ELIGIBLE_PERIOD_FORMS = frozenset({"current"})

#: The governed portfolio ROLES an explicit lens may name. `all` is the default
#: and is not a lens at all — it states the whole funded book, which is the
#: population every question already has. These are the roles
#: `interpretation_v2.vocabulary.POPULATION_LENSES` already admits, minus that
#: default; this adapter does not widen the vocabulary, it stops refusing it.
GOVERNED_LENS_ROLES = frozenset({"direct", "acquired"})

#: The two governed scope columns. A ROLE says what kind of book; a NAME says
#: which one. Different axes, different columns, and both may apply at once.
_ROLE_FIELD = "source_portfolio_type"
_SOURCE_ID_FIELD = "source_portfolio_id"

#: The population lenses the governed path may execute.
#:
#: `""` / `all` / `total` / `none` all state the whole funded book and produce no
#: predicate. `direct` and `acquired` are the governed ROLES, which the compiler
#: has always bound to a `source_portfolio_type` predicate
#: (`compiler._POPULATION_LENS_FIELD`) — and which this adapter used to refuse
#: as `EXPLICIT_LENS` because nothing carried that predicate through to the
#: executor. Now that `plan_predicates` does, the refusal would be the adapter
#: declining a plan it can express.
#:
#: A role is not a client's portfolio id and never becomes one here: the compiler
#: binds the role to the canonical role COLUMN, and a book whose tape carries no
#: such column fails at compile time with CONCEPT_UNAVAILABLE rather than
#: answering over a wider population.
ELIGIBLE_POPULATION_LENS = frozenset({"", "all", "total", "none"}) | GOVERNED_LENS_ROLES

#: Plan statistic -> the executor's aggregation vocabulary. A lookup, not a
#: judgement: the compiler already decided the statistic and the registry already
#: permitted it.
_AGGREGATION: Mapping[str, str] = {
    "sum": SUM,
    "average": AVERAGE,
    "weighted_average": WEIGHTED_AVERAGE,
    "count": COUNT,
}

#: What the executor calls a row count, mirroring `query_plan_compiler`'s own
#: constant rather than inventing a second name for the same thing.
_ROW_COUNT_FIELD = "loan_count"

MAX_DIMENSIONS = 2

# Ineligibility reasons. Stable strings, because the ledger groups by them.
NOT_A_PLAN = "NOT_A_PLAN"
CAPABILITY_NOT_GENERIC = "CAPABILITY_NOT_GENERIC"
OPERATION_NOT_GENERIC = "OPERATION_NOT_GENERIC"
PERIOD_NOT_CURRENT = "PERIOD_NOT_CURRENT"
EXPLICIT_LENS = "EXPLICIT_LENS"
#: A plan naming a governed role whose scope predicate is missing. The compiler
#: cannot produce this — it binds the predicate or refuses with
#: CONCEPT_UNAVAILABLE — but the adapter is a separate owner and the cost of
#: being wrong is the worst shape available: an answer computed over the WHOLE
#: book while the plan, the ledger and the reader all say "acquired".
SCOPE_NOT_BOUND = "SCOPE_NOT_BOUND"
NOT_SINGLE_OUTPUT = "NOT_SINGLE_OUTPUT"
TOO_MANY_DIMENSIONS = "TOO_MANY_DIMENSIONS"
MEASURE_NOT_GENERIC = "MEASURE_NOT_GENERIC"
MEASURE_UNBOUND = "MEASURE_UNBOUND"
DIMENSION_UNBOUND = "DIMENSION_UNBOUND"
FILTER_UNBOUND = "FILTER_UNBOUND"
COMPARISON_REQUESTED = "COMPARISON_REQUESTED"
TARGET_REQUESTED = "TARGET_REQUESTED"
NO_MEASURE = "NO_MEASURE"
GEOGRAPHY_REQUESTED = "GEOGRAPHY_REQUESTED"
FILTER_NOT_EXPRESSIBLE = "FILTER_NOT_EXPRESSIBLE"
#: The runtime could not say WHICH POPULATION it executed over. Fail-closed by
#: construction: a caller that does not declare its frame's identity gets no
#: governed answer, because an unproven population is exactly the state in which
#: a pipeline question is answered with funded numbers.
EXECUTED_POPULATION_UNPROVEN = "EXECUTED_POPULATION_UNPROVEN"
#: The plan asks for a population this runtime does not execute.
POPULATION_NOT_EXECUTABLE = "POPULATION_NOT_EXECUTABLE"
#: The plan asks for one population and the runtime loaded another.
POPULATION_BASE_MISMATCH = "POPULATION_BASE_MISMATCH"

#: WHICH POPULATIONS THIS RUNTIME CAN EXECUTE. Funded, and only funded: the
#: generic executor binds funded fields against a funded frame, and `mi_service`
#: resolves that frame through `datasets._resolve_query_frame("funded", …)`.
#:
#: THIS IS THE REGISTRATION POINT FOR A FUTURE OWNER, and the reason
#: `population.base` does not need redesigning when one arrives: a deterministic
#: pipeline runtime declares `pipeline` here (or declares its own set and calls
#: `check_population_base` with it), and nothing in `CandidateIntent`, the
#: compiler or the plan changes. Until then a pipeline plan is refused rather
#: than approximated, which is the whole point.
EXECUTABLE_POPULATIONS: FrozenSet[str] = frozenset({"funded"})

#: The governed default. `compiler._bind_population` already resolves an unstated
#: base to `funded`, so every plan carries one; this is the belt for the braces,
#: and it keeps a plan dict assembled by a test honest too.
DEFAULT_POPULATION_BASE = "funded"


@dataclass(frozen=True)
class ShadowOutcome:
    """One shadow attempt. Never a user-visible answer.

    `eligible=False` is the ordinary case and carries `reason`; the production
    path proceeds untouched either way.
    """

    eligible: bool
    reason: str = ""
    detail: str = ""
    plan_id: str = ""
    #: The spec this adapter handed to the existing executor, for the ledger.
    spec: Optional[MIQuerySpec] = None
    #: What the plan asked for, read off the plan and never re-derived.
    requested: Mapping[str, Any] = field(default_factory=dict)
    #: The scalar the existing executor produced, when there was one.
    value: Optional[float] = None
    #: The executor's own metadata — the resolved half of the receipt.
    receipt: Mapping[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    #: Set when the existing executor raised. Captured, never propagated.
    error: str = ""

    @property
    def executed(self) -> bool:
        return self.eligible and not self.error and self.spec is not None


def _bound(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _measures_of(output: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    return tuple(output.get("measures") or ())


def _as_mapping(plan: Any) -> Mapping[str, Any]:
    """A plan as a plain mapping, whether it arrived as the object or a dict.

    The evidence files hold plans as dicts and the live path holds them as
    `GovernedQueryPlan`; reading one shape means the replay harness and
    production exercise the same code.
    """
    if isinstance(plan, Mapping):
        return plan
    to_dict = getattr(plan, "to_dict", None)
    return to_dict() if callable(to_dict) else {}


def check_capability(plan: Any) -> Tuple[bool, str, str]:
    """`(ok, reason, detail)` for the capability alone. Slice-agnostic.

    Split out of `check_eligibility` so slice 2 can require the same generic
    capability without restating the test. Nothing about time is decided here.
    """
    body = _as_mapping(plan)
    if not body:
        return False, NOT_A_PLAN, "no plan, or a plan with no readable body"
    if body.get("capability") != ELIGIBLE_CAPABILITY:
        return (False, CAPABILITY_NOT_GENERIC,
                f"capability={body.get('capability')!r} is specialist; it owns its "
                f"own input contract and arithmetic")
    return True, "", ""


def check_eligibility(plan: Any) -> Tuple[bool, str, str]:
    """`(eligible, reason, detail)` for Slice 1. Reads the plan only.

    UNCHANGED BEHAVIOUR. Slice 2 needed the structural half of this test with a
    different temporal perimeter, so the structural half moved to
    `check_structure` and is called from here in the same position it occupied
    before. The reason codes, their precedence and their details are identical,
    and the slice 1 corpus replay is what says so rather than this sentence.
    """
    body = _as_mapping(plan)
    ok, reason, detail = check_capability(plan)
    if not ok:
        return ok, reason, detail

    if body.get("operation") not in ELIGIBLE_OPERATIONS:
        return (False, OPERATION_NOT_GENERIC,
                f"operation={body.get('operation')!r} is not expressible in the "
                f"generic executor's vocabulary")

    period = body.get("period") or {}
    if period.get("form") not in ELIGIBLE_PERIOD_FORMS:
        return (False, PERIOD_NOT_CURRENT,
                f"period.form={period.get('form')!r} needs the temporal owner, "
                f"which this slice does not touch")

    return check_structure(plan)


def requested_population_base(plan: Any) -> str:
    """WHICH POPULATION the plan asks about. The requested side, and its owner.

    Read off `GovernedQueryPlan.population.base` and from nothing else — never
    the question, never the frame, never the rows that came back. The plan is
    the transcription of what was asked, and this is the field that carries it.
    """
    population = (_as_mapping(plan).get("population") or {})
    base = str(population.get("base") or "").strip().lower()
    return base or DEFAULT_POPULATION_BASE


def check_population_base(plan: Any, executed_population: Any,
                          executable: Optional[Iterable[str]] = None
                          ) -> Tuple[bool, str, str]:
    """Does the population the plan ASKED FOR match the one that will RUN?

    THE DEFECT THIS CLOSES. `spec_for_plan` binds measures, filters, dimensions
    and scope predicates — and not `population.base`. Three plans asking for
    funded, pipeline and whole_book bound to three IDENTICAL specs, the
    perimeter admitted a `generic_analysis` plan whatever its base, and
    `mi_service` supplies the funded frame. So a plan stating
    `population.base = pipeline` would have executed over funded rows and been
    served as a governed answer, with the coverage ledger — which reconciles
    filters and dimensions only — seeing nothing wrong. Measured on the
    signed-off corpus: of 135 recorded plans, 83 funded, 34 pipeline, 8
    forecast and 1 whole_book, and the only reason none of the 43 non-funded
    ones reached the executor is that each was already ineligible for an
    unrelated reason. That is luck, not a control.

    THE TWO SIDES, AND WHERE EACH COMES FROM.

        requested   `plan.population.base`, the plan's own transcription
        executed    `executed_population`, DECLARED by the runtime that
                    resolved the frame — `mi_service` passes the same `view` it
                    called `datasets._resolve_query_frame` with

    Neither side is inferred. The question is not re-read, the result rows are
    not counted, and no field is sniffed: this compares two governed records the
    way `reconcile` compares the bound spec with the execution receipt.

    THREE WAYS TO FAIL, ALL CLOSED.

    1. The runtime declared nothing. An unproven population is the state the
       defect lived in, so it refuses rather than assuming funded.
    2. The plan wants a population this runtime does not execute — `pipeline`,
       `forecast`, or `whole_book`, which spans two frames and cannot be proven
       against one. A future owner registers itself in `EXECUTABLE_POPULATIONS`
       rather than being special-cased here.
    3. The plan wants one population and the runtime loaded another.

    `(eligible, reason, detail)`, matching every other perimeter check.
    """
    allowed = frozenset(str(p).strip().lower() for p in
                        (executable if executable is not None
                         else EXECUTABLE_POPULATIONS))
    executed = str(executed_population or "").strip().lower()
    if not executed:
        return (False, EXECUTED_POPULATION_UNPROVEN,
                "the runtime did not declare which population it executes over, "
                "so the plan's population cannot be proven")
    requested = requested_population_base(plan)
    if requested not in allowed:
        return (False, POPULATION_NOT_EXECUTABLE,
                f"population.base={requested!r} is not executed by this runtime "
                f"(it executes {sorted(allowed)}); it is refused rather than "
                f"answered over {executed!r}")
    if requested != executed:
        return (False, POPULATION_BASE_MISMATCH,
                f"the plan asks about the {requested!r} population and the "
                f"runtime loaded the {executed!r} one")
    return True, "", ""


def plan_predicates(body: Mapping[str, Any],
                    output: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    """EVERY governed predicate this plan authorises, in one place.

    The plan states its predicates in three slots — plan-level `filters`,
    output-level `filters`, and `population.scope_predicates` — and the third is
    where an explicit Direct/Acquired lens lands. `compiler._bind_population`
    puts it there rather than in `filters` because the population axis is
    resolved before the output is, but by the time a plan exists it is an
    ordinary `FilterBinding` on `source_portfolio_type` and nothing downstream
    should care which slot carried it.

    ONE ACCESSOR BECAUSE THREE READERS MUST NOT DRIFT. The eligibility check,
    the executor bind and the coverage ledger each used to read the two `filters`
    slots directly, so a scope predicate would have passed the duplicate-field
    check unseen, been dropped on the way to the executor, and been absent from
    the requested side of the ledger — a silent scope drop that reads as a
    correct total. Slice 2 lost a week to exactly that shape of divergence
    between a requested side and an executed one.

    `lens == "all"` produces no scope predicate at the compiler, so the default
    total-funded population is unchanged here by construction rather than by a
    branch.
    """
    population = body.get("population") or {}
    return (tuple(body.get("filters") or ())
            + tuple(output.get("filters") or ())
            + tuple(population.get("scope_predicates") or ()))


def check_structure(plan: Any) -> Tuple[bool, str, str]:
    """Everything the generic executor needs that is NOT about time.

    The population lens, the geography contract, the comparison and target
    slots, the single output, the axis count and their bindings, the single
    measure and its statistic, and one predicate per field. Called by slice 1
    and by slice 2 alike, so a plan that is structurally inexpressible is
    refused identically whichever period it names.
    """
    body = _as_mapping(plan)
    if not body:
        return False, NOT_A_PLAN, "no plan, or a plan with no readable body"

    population = body.get("population") or {}
    lens = str(population.get("lens") or "").strip().lower()
    if lens not in ELIGIBLE_POPULATION_LENS:
        return (False, EXPLICIT_LENS,
                f"population.lens={lens!r} is resolved and applied by "
                f"portfolio_lens on the production path")
    # A STATED ROLE MUST ARRIVE AS A PREDICATE. The compiler binds one or refuses,
    # so a role with no predicate is a plan no compiler wrote — and executing it
    # would answer over the whole book under the word "acquired", which is the
    # one failure a scope axis exists to prevent. Refused rather than repaired:
    # inventing the predicate here would make this module a second owner of what
    # a role means.
    bound_scope = {str(predicate.get("canonical_field") or "")
                   for predicate in (population.get("scope_predicates") or ())
                   if str(predicate.get("canonical_field") or "")}
    if lens in GOVERNED_LENS_ROLES and _ROLE_FIELD not in bound_scope:
        return (False, SCOPE_NOT_BOUND,
                f"population.lens={lens!r} states a governed role and the plan "
                f"carries no scope predicate to apply it")
    # THE SAME RULE FOR A NAMED BOOK. A plan naming "the ALP back book" with no
    # predicate on the identity column would be computed over every book and
    # labelled with the one book the reader asked about.
    if str(population.get("source_reference") or "").strip() \
            and _SOURCE_ID_FIELD not in bound_scope:
        return (False, SCOPE_NOT_BOUND,
                f"population.source_reference="
                f"{population.get('source_reference')!r} names a portfolio and "
                f"the plan carries no scope predicate to apply it")

    # GEOGRAPHY IS ITS OWN OWNER, AND THIS SLICE BINDS NONE OF IT. A plan's
    # geography binding carries a resolved basis and level chosen by
    # `config/asset/mi_geography.yaml`, and it may group, restrict, or both. The
    # adapter binds no geography axis and no geography predicate, so a plan that
    # states one cannot be executed here without computing a DIFFERENT breakdown
    # from the one the reader authorised — and doing that silently is the exact
    # failure this whole control plane exists to stop.
    #
    # Found by the slice 1 corpus replay, not by reasoning: five of the eligible
    # recorded plans (Q14A/B/C, Q16A, Q16C) carry `group_by: True` on
    # `canonical_region_reporting`, and the shadow was grouping by their other
    # axis alone. The plan is the contract; an axis it states and this module
    # cannot carry is an ineligibility, never a narrowing.
    for binding in (body.get("geography"),
                    *(o.get("geography") for o in (body.get("outputs") or ()))):
        if isinstance(binding, Mapping) and binding:
            return (False, GEOGRAPHY_REQUESTED,
                    f"geography level={binding.get('resolved_level')!r} "
                    f"field={binding.get('canonical_field')!r} "
                    f"group_by={bool(binding.get('group_by'))} — owned by the "
                    f"geography basis resolver, which this slice does not touch")

    comparison = str(body.get("comparison_kind") or "none").strip().lower()
    if comparison != "none":
        return False, COMPARISON_REQUESTED, f"comparison_kind={comparison!r}"
    if body.get("target"):
        return False, TARGET_REQUESTED, "a target is a milestone, not an aggregate"

    outputs = tuple(body.get("outputs") or ())
    if len(outputs) != 1:
        return False, NOT_SINGLE_OUTPUT, f"{len(outputs)} outputs"

    output = outputs[0]
    dimensions = tuple(output.get("dimensions") or ())
    if len(dimensions) > MAX_DIMENSIONS:
        return (False, TOO_MANY_DIMENSIONS,
                f"{len(dimensions)} dimensions, limit {MAX_DIMENSIONS}")
    for dim in dimensions:
        if not _bound(dim.get("canonical_field")):
            return (False, DIMENSION_UNBOUND,
                    f"dimension {dim.get('concept')!r} has no canonical field")

    measures = _measures_of(output)
    if not measures:
        return False, NO_MEASURE, "no measure to compute"
    if len(measures) != 1:
        # Several measures over one population is a multi-output question the
        # generic single-metric spec cannot carry without loss.
        return False, NOT_SINGLE_OUTPUT, f"{len(measures)} measures in one output"
    measure = measures[0]
    statistic = str(measure.get("statistic") or "")
    if statistic not in _AGGREGATION:
        return (False, MEASURE_NOT_GENERIC,
                f"statistic={statistic!r} is capability-owned or unsupported here")
    if statistic != "count" and not _bound(measure.get("canonical_field")):
        return (False, MEASURE_UNBOUND,
                f"measure {measure.get('concept')!r} has no canonical field")
    if statistic == "weighted_average" and not _bound(measure.get("weight_field")):
        return (False, MEASURE_UNBOUND,
                "a weighted average names no weight field")

    seen_filter_fields = set()
    for flt in plan_predicates(body, output):
        canonical_field = flt.get("canonical_field")
        if not _bound(canonical_field):
            return (False, FILTER_UNBOUND,
                    f"filter {flt.get('concept')!r} has no canonical field")
        # ONE PREDICATE PER FIELD IS THE EXECUTOR'S WIRE FORMAT, not a choice
        # this module gets to make: `MIQuerySpec.filters` is keyed by field, so a
        # second predicate on the same one would overwrite the first and the
        # population would silently widen or narrow. A band belongs in a single
        # `between`, and a plan that states two bounds separately is refused.
        if canonical_field in seen_filter_fields:
            return (False, FILTER_NOT_EXPRESSIBLE,
                    f"two predicates on {canonical_field!r}; the executor's "
                    f"filter format carries one per field and dropping either "
                    f"would change the population")
        seen_filter_fields.add(canonical_field)

    return True, "", ""


# --------------------------------------------------------------------------- #
# the mechanical bind
# --------------------------------------------------------------------------- #

def _filters_for(body: Mapping[str, Any],
                 output: Mapping[str, Any]) -> Dict[str, Any]:
    """Plan filter bindings → the executor's `{field: value | {op, value}}`.

    The same wire format `query_plan_compiler._filters_for` produces, written
    against the plan's bindings. A categorical equality is a bare value; anything
    with a direction keeps its operator. The value passes through the estate's own
    `_executor_value`, so a multi-valued bound arrives as the LIST the executor's
    filters have always held rather than as a tuple.

    Caller must have established eligibility: this shape holds ONE predicate per
    field, so a plan with two on the same field is refused upstream rather than
    quietly losing a bound here.
    """
    out: Dict[str, Any] = {}
    for flt in plan_predicates(body, output):
        field_name = flt.get("canonical_field")
        comparator = str(flt.get("comparator") or "eq").lower()
        value = _executor_value(flt.get("value"))
        if comparator in ("eq", "", "equals"):
            out[field_name] = value
        else:
            out[field_name] = {"op": comparator, "value": value}
    return out


def spec_for_plan(plan: Any, *,
                  measure_binding: Optional[Tuple[Optional[str], str]] = None
                  ) -> MIQuerySpec:
    """`GovernedQueryPlan` → `MIQuerySpec`. Mechanical; every value pre-bound.

    Caller must have established eligibility. Nothing here decides a semantic:
    the statistic, the field, the axes and the predicates were all bound by the
    deterministic compiler and re-validated against the governed registry before
    this plan existed.

    `measure_binding` is `(canonical_field, aggregation)` for the ONE case the
    plan genuinely cannot state: a CAPABILITY-OWNED measure. Those measures carry
    `statistic="capability"` and `canonical_field=null` BY DESIGN — the governed
    vocabulary declares them owned by a capability precisely so the model is
    never shown how they are built, which means the field and the arithmetic are
    the capability's to supply and not this module's to guess. A specialist
    runtime that holds that knowledge passes it here rather than assembling a
    second `MIQuerySpec` of its own: everything else about the spec — the axes,
    the predicates, and the rendering rules borrowed from the compiler — is
    identical for a specialist measure and a generic one, and a second builder
    would be a second presentation owner with no reason to exist.

    It is an OVERRIDE OF TWO FIELDS, not an escape hatch. Both halves are
    required together, so a caller cannot supply a field and leave the
    arithmetic to a default.
    """
    body = _as_mapping(plan)
    output = (tuple(body.get("outputs") or ()) or ({},))[0]
    measure = (_measures_of(output) or ({},))[0]

    if measure_binding is not None:
        bound_field, aggregation = measure_binding
        if aggregation not in _AGGREGATION.values():
            raise ValueError(f"aggregation {aggregation!r} is not a governed "
                             f"executor aggregation")
        metric = _ROW_COUNT_FIELD if aggregation == COUNT else bound_field
        if aggregation != COUNT and not _bound(metric):
            raise ValueError("a capability measure binding must name the field "
                             "its arithmetic runs over")
    else:
        statistic = str(measure.get("statistic") or "")
        aggregation = _AGGREGATION[statistic]
        metric = (_ROW_COUNT_FIELD if aggregation == COUNT
                  else measure.get("canonical_field"))
    axes = [d.get("canonical_field") for d in (output.get("dimensions") or ())]

    # PRESENTATION, borrowed rather than invented. `MIQuerySpec` defaults to an
    # intent/chart_type pair its own validator rejects, and the estate already
    # owns the mapping from dimensionality to rendering in
    # `query_plan_compiler._RENDERING` — "one axis is a bar; two are a matrix".
    # Re-deriving it would make a second presentation owner, and the plan
    # deliberately has no opinion about rendering.
    #
    # A SECOND AXIS IS NOT GIVEN A CHART ROLE. The executor validates `y` against
    # the field's own declared chart roles, and a governed dimension may legally
    # have none — `collateral_geography` carries ['color','filter','group','x'].
    # Whether a field can be a chart's y axis is a presentation property of the
    # field, and the plan knows nothing about it, so the adapter declines to
    # assert a role it cannot verify and renders two axes as a table instead. The
    # ANALYSIS is unaffected: the executor's `_all_group_dims` reads
    # `spec.dimensions` then `spec.dimension`, so both axes are still grouped.
    # Preserving the analysis and simplifying the picture is the direction
    # `_RENDERING`'s own comment requires; the reverse would not be allowed.
    if len(axes) > 1:
        intent, chart_type, output_format = _MANY_DIMENSIONS
    else:
        intent, chart_type, output_format = _RENDERING.get(len(axes),
                                                          _MANY_DIMENSIONS)

    return MIQuerySpec(
        intent=intent, chart_type=chart_type, output_format=output_format,
        metric=None if aggregation == COUNT else metric,
        aggregation=aggregation,
        weight_field=measure.get("weight_field") or None,
        dimension=axes[0] if axes else None,
        x=axes[0] if len(axes) == 1 else None,
        dimensions=list(axes),
        filters=_filters_for(body, output),
        explanation="Shadow execution of a governed plan (slice 1).",
    )


def requested_semantics(plan: Any) -> Dict[str, Any]:
    """What the plan ASKED for, for the ledger's requested half.

    Read straight off the plan. This is not a second interpretation; it is a
    transcription, and it exists so a divergence can be attributed without
    re-opening the plan object later.
    """
    body = _as_mapping(plan)
    output = (tuple(body.get("outputs") or ()) or ({},))[0]
    measure = (_measures_of(output) or ({},))[0]
    period = body.get("period") or {}
    return {
        "capability": body.get("capability"),
        "operation": body.get("operation"),
        "measure_concept": measure.get("concept"),
        "measure_field": measure.get("canonical_field"),
        "statistic": measure.get("statistic"),
        "weight_field": measure.get("weight_field"),
        "dimensions": [d.get("canonical_field")
                       for d in (output.get("dimensions") or ())],
        "dimension_concepts": [d.get("concept")
                               for d in (output.get("dimensions") or ())],
        # A LIST, AND THE COMPARATOR WITH IT. This was a `{field: value}` dict,
        # which lost two things the resolved half of the receipt keeps: the
        # direction (`applied_predicates` records `op`, so "age > 55" and
        # "age == 55" were indistinguishable on the requested side), and a second
        # predicate on the same field (an LTV band collapsed to one bound). A
        # ledger that cannot state what was asked cannot adjudicate a divergence.
        # THE SCOPE IS PART OF WHAT WAS ASKED. It reaches the executor through
        # `plan_predicates`, so it must reach the ledger the same way or the
        # coverage owner would find a predicate in the receipt that the request
        # never claimed — and on a temporal answer it must hold for every
        # snapshot, which the slice 2 per-receipt rule already enforces.
        "filters": [{"field": f.get("canonical_field"),
                     "comparator": str(f.get("comparator") or "eq"),
                     "value": f.get("value")}
                    for f in plan_predicates(body, output)],
        "population": body.get("population") or {},
        "geography": body.get("geography") or {},
        "period_form": period.get("form"),
        "period_contract": period.get("contract"),
    }


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

def execute_shadow_governed_plan(plan: Any, resolved_frame: Any,
                                 semantics: Any) -> ShadowOutcome:
    """Validate, bind, and hand the plan to the existing deterministic owner.

    `resolved_frame` is the frame the production request already resolved —
    this module never selects one. `semantics` is the governed field registry the
    production path already loaded.

    Returns a `ShadowOutcome` in every circumstance, including failure. It raises
    nothing: a shadow that could take down a served answer would be worse than no
    shadow at all.
    """
    eligible, reason, detail = check_eligibility(plan)
    plan_id = str(_as_mapping(plan).get("plan_id") or "")
    if not eligible:
        return ShadowOutcome(eligible=False, reason=reason, detail=detail,
                             plan_id=plan_id)

    try:
        spec = spec_for_plan(plan)
        requested = requested_semantics(plan)
    except Exception as exc:                                     # noqa: BLE001
        return ShadowOutcome(eligible=True, plan_id=plan_id,
                             error=f"bind failed: {type(exc).__name__}: {exc}"[:300])

    try:
        from mi_agent.mi_query_executor import execute_mi_query
        result = execute_mi_query(spec, resolved_frame, semantics)
    except Exception as exc:                                     # noqa: BLE001
        return ShadowOutcome(eligible=True, plan_id=plan_id, spec=spec,
                             requested=requested,
                             error=f"{type(exc).__name__}: {exc}"[:300])

    metadata = dict(getattr(result, "metadata", None) or {})
    value = _scalar_of(result, spec)
    return ShadowOutcome(
        eligible=True, plan_id=plan_id, spec=spec, requested=requested,
        value=value,
        receipt={
            "aggregation": metadata.get("aggregation"),
            "group_field_keys": list(metadata.get("group_field_keys") or ()),
            "applied_predicates": metadata.get("applied_predicates"),
            "input_row_count": metadata.get("input_row_count"),
            "filtered_row_count": metadata.get("filtered_row_count"),
            "balance_field_used": metadata.get("balance_field_used"),
            "result_type": getattr(result, "result_type", None),
            "row_count": getattr(result, "row_count", None),
        },
        warnings=tuple(str(w)[:200] for w in (getattr(result, "warnings", ()) or ())),
    )


def value_column(spec: MIQuerySpec) -> str:
    """The executor column carrying this spec's figure.

    `execute_mi_query` names an aggregated column `<metric>_<aggregation>` and a
    row count `loan_count`. One owner for that naming, because a caller that
    guessed it would silently read the wrong column of a grouped frame.
    """
    if spec.aggregation == COUNT:
        return _ROW_COUNT_FIELD
    return f"{spec.metric}_{spec.aggregation}"


def reconcile_receipt(spec: Any, result: Any) -> Tuple[bool, str]:
    """Did the executor apply every predicate and group on every axis the spec
    named? `(ok, why_not)`.

    The STRUCTURAL half of `plan_serving_canary.reconcile`, split out so the
    temporal runtime can ask the same question of each snapshot without also
    inheriting the serving decision about an empty population — which is a
    presentation ruling about ONE answer, and not the same ruling for one point
    of a series.

    Reads the bound spec and the executor's own receipt. No question, no plan
    re-interpretation.
    """
    metadata = dict(getattr(result, "metadata", None) or {})
    applied = {str(entry.get("field")) for entry in
               (metadata.get("applied_predicates") or ())
               if isinstance(entry, Mapping)}
    for field_name in (getattr(spec, "filters", None) or {}):
        if str(field_name) not in applied:
            return False, f"predicate on {field_name!r} was not applied"

    grouped = {str(key) for key in (metadata.get("group_field_keys") or ())}
    for axis in (getattr(spec, "dimensions", None) or ()):
        if str(axis) not in grouped:
            return False, f"axis {axis!r} was not grouped"
    return True, ""


def _scalar_of(result: Any, spec: MIQuerySpec) -> Optional[float]:
    """The single figure, when the execution produced one.

    A grouped execution has no scalar and returns None rather than a total the
    plan never asked for — inventing one here would be this module deciding an
    analytical question.
    """
    frame = getattr(result, "data", None)
    if frame is None or getattr(frame, "empty", False):
        return None
    if list(getattr(spec, "dimensions", None) or ()):
        return None
    for column in (f"{spec.metric}_{spec.aggregation}", _ROW_COUNT_FIELD,
                   spec.metric or ""):
        if column and column in frame.columns:
            try:
                return float(frame.iloc[0][column])
            except (TypeError, ValueError):
                return None
    numeric = [c for c in frame.columns
               if str(frame[c].dtype).startswith(("int", "float"))]
    if len(numeric) == 1:
        try:
            return float(frame.iloc[0][numeric[0]])
        except (TypeError, ValueError):
            return None
    return None


# --------------------------------------------------------------------------- #
# the shadow comparison ledger
# --------------------------------------------------------------------------- #

#: Where the ledger is written. Absent means "record nothing", so a deployment
#: can run shadow execution without producing evidence it has nowhere to put.
LEDGER_ENV_VAR = "MI_AGENT_PLAN_SHADOW_LEDGER"

# Parity classifications. A difference is not automatically a defect in the new
# path — the old path is a control, not the truth oracle.
EXACT_SEMANTIC_PARITY = "EXACT_SEMANTIC_PARITY"
PRESENTATION_ONLY = "PRESENTATION_ONLY"
OLD_PATH_SEMANTIC_DIFFERENCE = "OLD_PATH_SEMANTIC_DIFFERENCE"
NEW_PATH_SEMANTIC_DIFFERENCE = "NEW_PATH_SEMANTIC_DIFFERENCE"
NUMERICAL_DIFFERENCE = "NUMERICAL_DIFFERENCE"
DISPOSITION_DIFFERENCE = "DISPOSITION_DIFFERENCE"
SHADOW_EXECUTION_ERROR = "SHADOW_EXECUTION_ERROR"
NOT_ELIGIBLE = "NOT_ELIGIBLE"

#: Test/replay seam. Production leaves this None and the interpreter is built
#: lazily, so an OFF flag makes no model call and imports no interpreter.
_PLAN_PROVIDER: Optional[Any] = None


def set_plan_provider(provider: Optional[Any]) -> None:
    """Inject a plan source. For tests and offline replay only."""
    global _PLAN_PROVIDER
    _PLAN_PROVIDER = provider


def _old_value(result: Any) -> Optional[float]:
    """The legacy path's headline figure, if its envelope carries one.

    Read defensively and without interpretation: an envelope shape this does not
    recognise yields None and the comparison says so, rather than guessing which
    number was the answer.
    """
    if not isinstance(result, Mapping):
        return None
    for key in ("value", "total", "result"):
        candidate = result.get(key)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return float(candidate)
    metrics = result.get("metrics")
    if isinstance(metrics, Mapping):
        for candidate in metrics.values():
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return float(candidate)
    return None


def classify(outcome: "ShadowOutcome", old_result: Any) -> Tuple[str, str]:
    """`(classification, note)` comparing one shadow outcome with the control.

    Numerical and semantic parity are judged separately and answer TEXT is never
    compared — two paths can phrase one correct figure differently, and a prose
    match would hide a population difference behind identical wording.
    """
    if not outcome.eligible:
        return NOT_ELIGIBLE, outcome.reason
    if outcome.error:
        return SHADOW_EXECUTION_ERROR, outcome.error
    old_ok = bool(isinstance(old_result, Mapping) and old_result.get("ok"))
    if not old_ok:
        return (DISPOSITION_DIFFERENCE,
                "the shadow executed where the control did not answer")
    new_value, old_value = outcome.value, _old_value(old_result)
    if new_value is None or old_value is None:
        return (PRESENTATION_ONLY,
                f"no comparable scalar (new={new_value!r}, control={old_value!r})")
    if abs(new_value - old_value) < 0.01:
        return EXACT_SEMANTIC_PARITY, ""
    return (NUMERICAL_DIFFERENCE,
            f"control={old_value!r} shadow={new_value!r} — needs adjudication, "
            f"the control is not the truth oracle")


def _ledger_record(outcome: "ShadowOutcome", classification: str, note: str,
                   *, view: Optional[str], portfolio_id: Optional[str],
                   old_result: Any, case_id: str = "") -> Dict[str, Any]:
    """One comparison row. Aggregates and field NAMES only — never a loan row."""
    return {
        "case_id": case_id,
        "classification": classification,
        "note": note[:300],
        "eligible": outcome.eligible,
        "ineligible_reason": outcome.reason,
        "plan_id": outcome.plan_id,
        "requested": dict(outcome.requested),
        "caller_dataset_view": view,
        "caller_portfolio_id": portfolio_id,
        "new_value": outcome.value,
        "new_receipt": dict(outcome.receipt),
        "new_warnings": list(outcome.warnings),
        "new_error": outcome.error,
        "control_ok": bool(isinstance(old_result, Mapping)
                           and old_result.get("ok")),
        "control_value": _old_value(old_result),
        "control_route": (old_result.get("route")
                          if isinstance(old_result, Mapping) else None),
    }


def _append_ledger(record: Mapping[str, Any]) -> None:
    import json
    path = str(os.environ.get(LEDGER_ENV_VAR) or "").strip()
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def observe(*, result: Any, frame: Any, semantics: Any,
            view: Optional[str] = None, portfolio_id: Optional[str] = None,
            plan: Any = None, case_id: str = "") -> Optional[Dict[str, Any]]:
    """The production shadow hook. Returns a record, or None when inactive.

    Swallows everything. A shadow that could break a served answer would be worse
    than no shadow, so every failure path here ends in a ledger row or silence —
    never an exception reaching the caller.

    `plan` may be supplied directly (replay); otherwise a provider is consulted.
    With the flag off this returns immediately, having imported no interpreter and
    made no model call.
    """
    try:
        if shadow_mode() != SHADOW_ON:
            return None
        governed_plan = plan
        if governed_plan is None and _PLAN_PROVIDER is not None:
            governed_plan = _PLAN_PROVIDER()
        if governed_plan is None:
            return None
        outcome = execute_shadow_governed_plan(governed_plan, frame, semantics)
        classification, note = classify(outcome, result)
        record = _ledger_record(outcome, classification, note, view=view,
                                portfolio_id=portfolio_id, old_result=result,
                                case_id=case_id)
        _append_ledger(record)
        return record
    except Exception:                                            # noqa: BLE001
        return None
