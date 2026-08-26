"""mi_agent_api/analytical_plan.py — the compositional plan layer.

Shipped routes whose execution is a PLAN over derived primitives rather than a
route-specific procedure:

    Conversion 1   portfolio_summary
    Conversion 2   period_movement

Named for what it is. It arrived as `portfolio_summary_plan.py` because one
route needed it; the plan artefact, the population step and the primitive
vocabulary were never route-specific, and Conversion 2 reuses all three
unchanged. Renamed rather than copied — a second plan layer would be the
duplication this programme exists to remove.

    interpretation contract -> plan -> existing primitives -> the same result

What this replaces is exactly one call: `movement_summary.portfolio_summary`.
Everything after it — the prose, the KPI/chart/table artifacts, the envelope,
the receipt — is untouched, and that is deliberate. The economics were proven
equivalent in Phase 0 and re-proven on the governed population path in Phase 1G;
the boundary this conversion has to hold is the one a shadow could not reach,
which is that the RESULT SHAPE feeding all of that is identical.

THE HARD RULE
-------------
`build_plan` receives a :class:`QuestionInterpretation` and a frame context. It
never receives, reads or is passed the raw question. `assert_no_question_read`
enforces it structurally, so an edit that reintroduces a second semantic owner
fails loudly rather than silently.

WHERE EACH SEMANTIC FACT COMES FROM
-----------------------------------
All of it from the contract, none of it from the sentence:

    which portfolios      source_scope.portfolio_ids   governed registry ids
    which population      source_scope.base_population funded / direct / acquired
    was it asked for      source_scope.provenance      decides caller precedence
    could it be resolved  source_scope.state           UNRESOLVABLE blocks

THE PRIMITIVES
--------------
    stack periods      evolution.funded_frames
    select population  evolution._scope_frame_lens, over governed ids
    resolve measure    evolution.assemble_funded_evolution (x5 metrics)
    group              movement_summary._regional_exposure, _cohorts
    rank               the sort + head inside _regional_exposure
    compare            movement_summary._delta, prior period vs current

`portfolio_summary` uses five of the seven; `period_movement` adds `compare`.
Both reuse EXISTING implementations — A2's fourth threshold is a NEW
implementation of a primitive, and Phase 4 is already the consolidation of the
four `group` implementations that exist. A fifth would make its own successor
phase larger. `project` is unused by either.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: Primitive ids, as derived by the scoping study.
STACK_PERIODS = "stack_periods"
SELECT_POPULATION = "select_population"
RESOLVE_MEASURE = "resolve_measure"
GROUP = "group"
RANK = "rank"
COMPARE = "compare"

#: A plan step that cannot be built from the contract carries this. A plan with
#: any blocked step is a REFUSAL, never an answer with the step omitted.
BLOCKED_NO_CONTRACT_FIELD = "no contract field"

#: The GOVERNED MODES of `select_population`. They were string literals in three
#: places before a second mode existed; naming them is what stops a reader from
#: assuming there is only one.
#:
#: The two narrowing modes are deliberately SEPARATE STRUCTURES, not one filter
#: bag. `source_portfolio_lens` narrows by governed portfolio IDENTITY, which the
#: registry decides — Phase 1C measured the two readings diverging at GBP300
#: against GBP1,200 on a book with two portfolios of one type. `row_predicates`
#: narrows by VALUE, on a governed field. Collapsing them into `lens_filters`
#: would put identity back into the value channel, which is the P1I-A ruling in
#: reverse.
KIND_SOURCE_PORTFOLIO_LENS = "source_portfolio_lens"
KIND_ROW_PREDICATES = "row_predicates"
KIND_WHOLE_DATASET = "whole_dataset"

#: The five governed headline measures, and how each is resolved.
HEADLINE_MEASURES: Tuple[Tuple[str, str], ...] = (
    ("funded_balance", "sum"),
    ("loan_count", "count"),
    ("wa_ltv", "weighted_avg"),
    ("wa_interest_rate", "weighted_avg"),
    ("avg_borrower_age", "avg"),
)


@dataclass(frozen=True)
class Step:
    primitive: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    because: str = ""
    blocked: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"primitive": self.primitive,
                               "inputs": dict(self.inputs), "because": self.because}
        if self.blocked:
            out["blocked"] = self.blocked
        return out


@dataclass(frozen=True)
class Plan:
    steps: Tuple[Step, ...]
    declares_grouped_by: Tuple[str, ...] = ()

    @property
    def blocked(self) -> Tuple[Step, ...]:
        return tuple(s for s in self.steps if s.blocked)

    @property
    def executable(self) -> bool:
        return not self.blocked

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps],
                "declaresGroupedBy": list(self.declares_grouped_by),
                "blocked": [s.to_dict() for s in self.blocked]}


def build_plan(interpretation, *, region_column: Optional[str],
               has_portfolio_column: bool) -> Plan:
    """The plan for a portfolio-summary question. The question is NOT a parameter."""
    steps: List[Step] = [
        Step(STACK_PERIODS,
             {"dataset": "funded", "take": "latest", "disclose": "periodCount"},
             because="the headline position is the latest governed snapshot, and "
                     "the count of available periods is disclosed"),
    ]

    steps.append(_population_step(
        getattr(interpretation, "source_scope", None)))

    for metric, aggregation in HEADLINE_MEASURES:
        steps.append(Step(RESOLVE_MEASURE, {"metric": metric,
                                            "aggregation": aggregation},
                          because="a governed headline metric"))

    grouped_by: List[str] = []
    if region_column:
        grouped_by.append(region_column)
        steps.append(Step(GROUP, {"by": [region_column], "measure": "funded_balance",
                                  "aggregation": "sum", "share_of": "scope_total"},
                          because="the summary names the largest regional exposures"))
        steps.append(Step(RANK, {"of": region_column, "basis": "funded_balance",
                                 "direction": "desc", "top_n": "TOP_REGIONS",
                                 "residual": None},
                          because="largest first, truncated for legibility"))
    if has_portfolio_column:
        grouped_by.append("source_portfolio_id")
        steps.append(Step(GROUP, {"by": ["source_portfolio_id"],
                                  "measure": "funded_balance", "aggregation": "sum"},
                          because="the summary splits by source portfolio"))
    return Plan(tuple(steps), tuple(grouped_by))


def _population_step(scope) -> Step:
    """The `select_population` step — the ONE place the scope states are read.

    Shared by every route plan, because the decision is not route-specific
    and a second copy would drift apart from this one.
    """
    state = getattr(scope, "state", "empty")
    if state == "unresolvable":
        # THE QUESTION NAMED A SCOPE AND IT COULD NOT BE RESOLVED, and this does
        # NOT block. The distinction matters and is easy to get backwards:
        #
        #   EMPTY          nobody looked. Nothing can be planned from it, and
        #                  reading it as Total would widen a population the
        #                  question may have narrowed. It blocks.
        #   UNRESOLVABLE   the owner looked and found a name this book does not
        #                  hold. That is a REFUSAL, and the refusal already has
        #                  a single route-independent owner: the facet layer
        #                  raises it as a LOST narrowing and `assess` declines
        #                  the answer (Phase 1E, proved across three routes).
        #
        # Blocking here would put a SECOND refusal owner in the plan, and
        # measured, it also cost route identity: the route deferred, the answer
        # fell through to the point-in-time path, and 23 payload and receipt
        # fields moved on a question that refuses either way.
        #
        # The step is recorded with `unresolved` so the plan still DECLARES what
        # it could not do — the plan is auditable, and the receipt decides.
        return Step(
            SELECT_POPULATION,
            {"kind": "source_portfolio_lens", "base_population": None,
             "portfolio_ids": [], "provenance": scope.provenance,
             "unresolved": True, "label": _label_for(scope)},
            because=("the question named a scope this book does not hold; the "
                     "receipt layer refuses it, and no narrowing is applied"))
    elif state == "filled":
        return Step(
            SELECT_POPULATION,
            {"kind": KIND_SOURCE_PORTFOLIO_LENS,
             "base_population": scope.base_population,
             "portfolio_ids": list(scope.portfolio_ids),
             "provenance": scope.provenance,
             "label": _label_for(scope)},
            because=("the contract carries a resolved source scope "
                     f"({scope.base_population!r}, {scope.provenance!r})"))

    # EMPTY — nobody looked. The ONE state that blocks: reading it as Total
    # would widen a population the question may have narrowed, which is the
    # defect the whole programme exists to remove.
    return Step(
        SELECT_POPULATION, {"kind": "source_portfolio_lens"},
        because="the route narrows by portfolio lens",
        blocked=(BLOCKED_NO_CONTRACT_FIELD + ": source_scope is "
                 f"{state!r}" + (f" ({scope.reason})" if getattr(
                     scope, "reason", None) else "")
                 + ". Absence of a resolved scope is NOT Total."))


def _whole_dataset_step(route: str, dataset: Optional[str] = None) -> Step:
    """The `select_population` step for a route that DOES NOT NARROW.

    The counterpart to :func:`_population_step`, and the reason it exists is an
    independent audit finding: the first version of this rule let a plan builder
    opt out of the governed population decision by writing the literal string
    `"whole_dataset"`. A magic string is not a governed fact — a future route
    that SHOULD narrow could claim it, and nothing would notice.

    So the exemption is now CHECKED, against the one place the platform already
    declares which routes narrow: `Recogniser.lens_aware`, from which
    `chat_routing._lens_aware_routes` is derived and on which the product's own
    "Scope not narrowed" disclosure already depends. A route that declares it
    narrows cannot obtain this step; it gets a BLOCKED one naming the
    contradiction, and the plan refuses rather than quietly widening.

    Passing a route this registry does not know is also blocked. Not being able
    to prove the claim is not the same as the claim being false, and a plan is
    not the place to decide which.
    """
    inputs = {"kind": KIND_WHOLE_DATASET, "dataset": dataset}
    because = ("this route does not narrow by source portfolio; a named scope "
               "is refused by the facet layer as a lost narrowing")
    try:
        from .chat_routing import REGISTRY  # local: chat_routing imports us
        declared = {r.name: r.lens_aware for r in REGISTRY.ordered()}
    except Exception as exc:  # noqa: BLE001 - unprovable is not the same as false
        return Step(SELECT_POPULATION, inputs, because=because,
                    blocked=(BLOCKED_NO_CONTRACT_FIELD + ": the route registry "
                             "is unavailable, so a whole-dataset claim cannot "
                             f"be proved ({exc})"))
    if route not in declared:
        return Step(SELECT_POPULATION, inputs, because=because,
                    blocked=(BLOCKED_NO_CONTRACT_FIELD + f": route {route!r} is "
                             "not in the governed registry, so its whole-dataset "
                             "claim cannot be proved"))
    if declared[route]:
        return Step(SELECT_POPULATION, inputs, because=because,
                    blocked=(BLOCKED_NO_CONTRACT_FIELD + f": route {route!r} is "
                             "declared lens_aware — it NARROWS — so it may not "
                             "plan the whole dataset. Use `_population_step`."))
    return Step(SELECT_POPULATION, inputs, because=because)


def row_predicate_step(interpretation) -> Optional[Step]:
    """The `select_population` step for VALUE predicates — the second mode.

    Built EXCLUSIVELY from `RowPredicateClaim`, which the governed parser
    already resolved: `_filter_field_of` bound the field once, upstream of every
    route, and `population.material_predicates` normalised the result. Nothing
    here reads `spec.filters`, the question text, or a provenance string, so a
    route planning from this cannot re-derive a filter's meaning even by
    accident — there is no English within reach.

    Returns ``None`` when the question carries no row predicate. That is the
    ordinary case and it is NOT a blocked step: a question that narrows nothing
    plans no narrowing.
    """
    claims = [c for c in (getattr(interpretation, "row_predicates", None) or [])
              if getattr(c, "field_key", None)]
    if not claims:
        return None
    predicates = [{"field": c.field_key, "op": c.operator, "value": c.value}
                  for c in claims]
    described = "; ".join(f"{d['field']} {d['op']} {d['value']}" for d in predicates)
    return Step(SELECT_POPULATION,
                {"kind": KIND_ROW_PREDICATES, "predicates": predicates},
                because=f"the contract carries resolved row predicates ({described})")


def governed_stage(interpretation):
    """``(stage, names_axis)`` — read from the CONTRACT, never from the question.

    Both halves already sit on the contract, produced by
    `lexical.pipeline_stage_request`, THE one place a stage is read from a
    sentence:

      stage       a `FilterClaim` carrying the canonical value (OFFER, not
                  "offer issued"), sourced to that reader
      names_axis  a `DimensionClaim` on `pipeline_stage` whose role is GROUPING
                  — the question named the stage DIMENSION and no single stage

    `_route_evolution` used to hold two more readers of the same fact: a
    membership test against a five-substring `_FUNNEL_KEYWORDS` map, and a
    substring test against three hard-coded phrases. Those are the duplicate
    owners this retires.
    """
    from question_interpretation.lexical import PIPELINE_STAGE_FIELD
    from question_interpretation.schema import FILLED, GROUPING

    stage = next((c.categorical_value for c in
                  (getattr(interpretation, "filters", None) or [])
                  if getattr(c, "source", "") == "lexical.pipeline_stage_request"
                  and getattr(c, "categorical_value", None)), None)
    # `candidate_concept`, not `concept`. The first cut of this reader used the
    # wrong attribute name, `getattr` returned None for every claim, and the
    # axis silently read False — which moved both stage-axis questions off
    # `evolution_pipeline_stage` onto the plain evolution path. The 882-question
    # blast caught it; nothing else would have, because both questions refuse in
    # this environment either way and only the ROUTE and the refusal wording
    # differed.
    axis = any(getattr(d, "candidate_concept", None) == PIPELINE_STAGE_FIELD
               and getattr(d, "role", None) == GROUPING
               and d.state == FILLED
               for d in (getattr(interpretation, "dimensions", None) or []))
    return stage, axis


def governed_stage_step(interpretation) -> Optional[Step]:
    """The stage as a row-predicate population step.

    A DIFFERENT step from `row_predicate_step`, deliberately, and the reason is
    measured rather than stylistic. `pipeline_stage` never appears in
    `spec.filters` — 0 of 882 corpus questions — so promoting the governed stage
    claim into the global `row_predicates` channel would attach
    `pipeline_stage eq COMPLETED` to questions the claim fires on but that are
    NOT stage narrowings: 35 of the 39 stage-naming questions route elsewhere,
    10 of them to `forecast_extrapolation`, where "completion" is a forecast
    TIME concept ("forecast by completion month", "show projected completions").

    So the stage is consumed where it plays a narrowing role — in the evolution
    plan — rather than asserted globally. That is a decision about WHERE a
    governed claim is read, not about what it means; the vocabulary stays whole
    and stays owned by `pipeline_stage_request`.
    """
    stage, _axis = governed_stage(interpretation)
    if not stage:
        return None
    from question_interpretation.lexical import PIPELINE_STAGE_FIELD
    return Step(SELECT_POPULATION,
                {"kind": KIND_ROW_PREDICATES,
                 "predicates": [{"field": PIPELINE_STAGE_FIELD, "op": "eq",
                                 "value": stage}]},
                because=f"the contract carries the governed pipeline stage {stage}")


def evolution_dataset(interpretation) -> Optional[str]:
    """Which governed dataset the series is built from, from the ONE owner.

    `workspace.resolve_dataset` decided it once and the projection carried it.
    `_route_evolution` used to call that resolver a second time on the raw
    question — agreement by maintenance, on all 32 owned questions, right up
    until someone changed one of them.
    """
    claim = getattr(interpretation, "dataset", None)
    return getattr(claim, "dataset", None) if claim is not None else None


def row_predicates(plan_or_step) -> List[Any]:
    """The governed `Predicate` objects a plan selects rows by.

    Returns the executor's own `Predicate`, not a dict, because the one thing
    every caller must NOT do is re-interpret these. They go straight to
    `governed_predicate_mask` — the single owner of what a predicate means.
    """
    from mi_agent.population import Predicate

    steps = (plan_or_step.steps if hasattr(plan_or_step, "steps")
             else ([plan_or_step] if plan_or_step is not None else []))
    out: List[Any] = []
    for step in steps:
        if (step is None or step.primitive != SELECT_POPULATION
                or step.inputs.get("kind") != KIND_ROW_PREDICATES or step.blocked):
            continue
        for entry in step.inputs.get("predicates") or []:
            out.append(Predicate(entry["field"], entry["op"], entry["value"]))
    return out


def _label_for(scope) -> str:  # noqa: D401
    """The scope's name as the ANSWER says it.

    `portfolio_label` is the governed display label a named portfolio resolved
    to; `raw_text` is the category name the owner produced ("Direct",
    "Acquired"); Total carries neither, because it is the absence of a
    narrowing. Taken from the contract rather than rebuilt, so the prose says
    what the shipped route says.
    """
    return (getattr(scope, "portfolio_label", None)
            or getattr(scope, "raw_text", None) or "Total")


def lens_filters(plan: Plan) -> Optional[Dict[str, Any]]:
    """The row filters this plan selects, or ``None`` for the whole population.

    Governed portfolio ids, never a raw type column: the registry decides group
    membership, and Phase 1C measured the two paths diverging at GBP300 against
    GBP1,200 on a book with two portfolios of one type.
    """
    step = next((s for s in plan.steps
                 if s.primitive == SELECT_POPULATION
                 and s.inputs.get("kind") == KIND_SOURCE_PORTFOLIO_LENS
                 and not s.blocked), None)
    if step is None:
        return None
    ids = list(step.inputs.get("portfolio_ids") or [])
    return {"source_portfolio_id": ids} if ids else None


def lens_label(plan: Plan) -> str:
    """The SCOPE's label. Kind-aware, and it has to be: once a plan can carry
    two `select_population` steps, "the first one" is no longer the lens."""
    step = next((s for s in plan.steps
                 if s.primitive == SELECT_POPULATION and not s.blocked
                 and s.inputs.get("kind") != KIND_ROW_PREDICATES), None)
    return (step.inputs.get("label") if step else None) or "Total"


def portfolio_summary(output_root, client_id: str, *, interpretation,
                      to_run_id: Optional[str] = None) -> Dict[str, Any]:
    """The current reporting period's headline position, COMPOSED.

    A drop-in for `movement_summary.portfolio_summary`: same arguments except
    that the population comes from the interpretation contract instead of a lens
    the caller resolved, and the same result dict. Everything downstream — prose,
    artifacts, envelope, receipt — is therefore unchanged by construction, which
    is most of the equivalence argument and the reason the switch is one line.
    """
    from . import evolution as evolution_mod
    from . import movement_summary as summary_mod

    frames = evolution_mod.funded_frames(output_root, client_id, to_run_id)
    df0 = frames[0].get("df") if frames else None
    region_column = summary_mod._region_column(df0) if df0 is not None else None
    has_portfolio = (df0 is not None
                     and summary_mod._PORTFOLIO_ID in getattr(df0, "columns", []))

    plan = build_plan(interpretation, region_column=region_column,
                      has_portfolio_column=has_portfolio)
    label = lens_label(plan)
    if plan.blocked:
        # A blocked plan REFUSES. The shape matches the unavailable branch the
        # route already knows how to handle, so the refusal travels the path
        # that exists rather than a new one.
        return {"available": False, "lens": label,
                "reason": plan.blocked[0].blocked,
                "planBlocked": [s.to_dict() for s in plan.blocked]}

    filters = lens_filters(plan)
    scoped = [{**f, "df": evolution_mod._scope_frame_lens(f.get("df"), filters)}
              for f in frames]
    scoped = [f for f in scoped if f["df"] is not None and len(f["df"])]
    if not scoped:
        return {"available": False, "lens": label,
                "reason": "no governed reporting period is available for this scope"}

    evo = evolution_mod.assemble_funded_evolution(scoped, client_id, to_run_id)
    periods = evo.get("periods") or []
    if not periods:
        return {"available": False, "lens": label,
                "reason": "no governed reporting period is available for this scope"}

    current = periods[-1]
    df = scoped[-1]["df"]
    region_col = summary_mod._region_column(df)
    return {
        "available": True,
        "lens": label,
        "period": current.get("period"),
        "reportingDate": current.get("reporting_date"),
        "metrics": summary_mod._metrics(current),
        "regionColumn": region_col,
        "topRegions": (summary_mod._regional_exposure(df, region_col)
                       if region_col else []),
        "cohorts": summary_mod._cohorts(df),
        "cohortBalances": summary_mod.cohort_balances(df),
        "periodCount": len(periods),
        "sourceFiles": [f.get("source") for f in scoped],
        "declaredGroupedBy": list(plan.declares_grouped_by),
    }


# --------------------------------------------------------------------------- #
# CONVERSION 2 — period_movement
# --------------------------------------------------------------------------- #
#: The comparison window when the question names no span: one governed reporting
#: period, i.e. month on month. The route's own long-standing default, carried
#: here rather than restated at the call site.
DEFAULT_SPAN_PERIODS = 1


def build_period_movement_plan(interpretation, *, region_column: Optional[str],
                               has_portfolio_column: bool) -> Plan:
    """The plan for a period-movement question. The question is NOT a parameter.

    The same population step as `build_plan`, plus the two things that make this
    a movement rather than a position: a second period stacked, and a `compare`
    across the pair.

    THE WINDOW COMES FROM THE CONTRACT. `time.window_periods` carries what
    `period_request.requested_span` read — the magnitude, not only the wording —
    which the target-state closure added precisely because this route was asking
    that owner a second time for it.
    """
    time = getattr(interpretation, "time", None)
    span = getattr(time, "window_periods", None) or DEFAULT_SPAN_PERIODS
    steps: List[Step] = [
        Step(STACK_PERIODS,
             {"dataset": "funded", "take": "pair", "span_periods": span,
              "governed_window": bool(getattr(time, "window_governed", False)),
              "disclose": "periodsAvailable"},
             because=(f"a movement compares the current governed snapshot with "
                      f"the one {span} reporting period(s) before it")),
        _population_step(getattr(interpretation, "source_scope", None)),
    ]
    for metric, aggregation in HEADLINE_MEASURES:
        steps.append(Step(RESOLVE_MEASURE, {"metric": metric,
                                            "aggregation": aggregation},
                          because="a governed headline metric, on both sides"))
    steps.append(Step(COMPARE,
                      {"of": [m for m, _a in HEADLINE_MEASURES],
                       "between": "prior period and current period",
                       "as": "absolute delta"},
                      because="the movement IS the comparison"))

    grouped_by: List[str] = []
    if region_column:
        grouped_by.append(region_column)
        steps.append(Step(GROUP, {"by": [region_column], "measure": "funded_balance",
                                  "aggregation": "sum", "of": "the delta"},
                          because="the answer attributes the movement by region"))
    if has_portfolio_column:
        grouped_by.append("source_portfolio_id")
        steps.append(Step(GROUP, {"by": ["source_portfolio_id"],
                                  "measure": "funded_balance",
                                  "aggregation": "sum", "of": "the delta"},
                          because="the answer attributes the movement by source "
                                  "portfolio"))
    return Plan(tuple(steps), tuple(grouped_by))


def span_periods(plan: Plan) -> int:
    """The comparison window this plan stacks, from the plan alone."""
    step = next((s for s in plan.steps if s.primitive == STACK_PERIODS), None)
    return int((step.inputs.get("span_periods") if step else None)
               or DEFAULT_SPAN_PERIODS)


def period_movement(output_root, client_id: str, *, interpretation,
                    to_run_id: Optional[str] = None) -> Dict[str, Any]:
    """Movement across the governed metrics, COMPOSED.

    A drop-in for `movement_summary.period_movement`: the population and the
    window come from the interpretation contract instead of from two separate
    reads of the question, and the same result dict comes back — so the prose,
    the artifacts, the envelope and the receipt are unchanged by construction.
    """
    from . import evolution as evolution_mod
    from . import movement_summary as summary_mod

    frames = evolution_mod.funded_frames(output_root, client_id, to_run_id)
    df0 = frames[0].get("df") if frames else None
    region_column = summary_mod._region_column(df0) if df0 is not None else None
    has_portfolio = (df0 is not None
                     and summary_mod._PORTFOLIO_ID in getattr(df0, "columns", []))

    plan = build_period_movement_plan(interpretation, region_column=region_column,
                                      has_portfolio_column=has_portfolio)
    label = lens_label(plan)
    if plan.blocked:
        return {"available": False, "lens": label,
                "reason": plan.blocked[0].blocked,
                "planBlocked": [s.to_dict() for s in plan.blocked]}

    # EXISTING IMPLEMENTATION, reused. A2's fourth threshold is a NEW
    # implementation of a primitive; the periods, deltas, regional bridge and
    # cohort attribution all already exist there, and re-deriving them here
    # would add a second owner of the same economics for no gain.
    return summary_mod.period_movement(
        output_root, client_id, to_run_id=to_run_id,
        lens_filters=lens_filters(plan), lens_label=label,
        span_periods=span_periods(plan))


# --------------------------------------------------------------------------- #
# CONVERSION 3 — `geo_exposure`, composed.
#
# The measurement conversion. Its route reads exactly ONE semantic fact from the
# question — the source scope — and that fact was already bridged by Conversion
# 1 and generalised by Conversion 2. So this section is the test of whether a
# route whose semantics are already carried migrates through route wiring alone.
#
# It is also the first converted route handed a RESOLVED FRAME rather than an
# output root, which is the one generic gap the re-baseline predicted.
# --------------------------------------------------------------------------- #

#: The ITL3 view truncates for legibility. The route's own long-standing
#: constant, carried here rather than restated at the call site.
TOP_AREAS = 15


def scope_frame(plan: Plan, df: Any) -> Any:
    """Narrow ONE already-resolved frame to the population a plan selects.

    SHARED. `portfolio_summary` and `period_movement` reach `_scope_frame_lens`
    through `funded_frames`, because they stack governed periods; a
    point-in-time route is handed the frame instead. This is the entry point for
    that case, and it is deliberately the SAME narrowing — `lens_filters` reads
    governed portfolio ids off the plan and `evolution._scope_frame_lens`
    applies them.

    That matters more than its size. `chat_routing._apply_lens_filter` is a
    second implementation of this narrowing, reached from a lens object rather
    than from a plan. The two agree today. Routing the converted path through
    the plan's own filters means the compositional layer has ONE narrowing, and
    it is the one the governed population step decided.
    """
    from . import evolution as evolution_mod

    return evolution_mod._scope_frame_lens(df, lens_filters(plan))


def build_geo_exposure_plan(interpretation) -> Plan:
    """The plan for a geographic-concentration question.

    The question is NOT a parameter, and here that is nearly the whole story:
    scope is the ONLY thing this route ever read from it. Everything else is the
    route's identity — ITL3 is the grouping, balance is the measure, largest
    first is the order.

    No `stack_periods`: geographic concentration is a POINT-IN-TIME question,
    answered from the frame the caller is working in. Declaring a period step
    would claim a governance property this answer does not have.
    """
    steps: List[Step] = [
        _population_step(getattr(interpretation, "source_scope", None)),
        Step(RESOLVE_MEASURE, {"metric": "funded_balance", "aggregation": "sum"},
             because="exposure is the funded balance"),
        Step(RESOLVE_MEASURE, {"metric": "loan_count", "aggregation": "count"},
             because="the count of loans behind each area's exposure"),
        Step(GROUP, {"by": ["itl3_code"], "measure": "funded_balance",
                     "aggregation": "sum", "share_of": "scope_total"},
             because="ITL3 area is the governed geographic grain, and each "
                     "area's share is a share of the SCOPE, not of the platform"),
        Step(RANK, {"of": "itl3_code", "basis": "funded_balance",
                    "direction": "desc", "top_n": TOP_AREAS, "residual": None},
             because="largest first, truncated for legibility"),
    ]
    return Plan(tuple(steps), ("itl3_code",))


def geo_exposure(df: Any, *, interpretation) -> Dict[str, Any]:
    """Funded exposure by ITL3 area, COMPOSED.

    Takes the frame the caller already resolved — this route answers at one
    date — and returns the same engine result the shipped path returned, plus
    the scope label and whether a narrowing was applied. Everything downstream
    (the bar, the table, the prose, the envelope, the receipt) is therefore
    unchanged by construction.
    """
    from . import geo as geo_mod

    plan = build_geo_exposure_plan(interpretation)
    label = lens_label(plan)
    narrowed = lens_filters(plan) is not None
    if plan.blocked:
        return {"available": False, "lens": label, "narrowed": narrowed,
                "reason": plan.blocked[0].blocked,
                "planBlocked": [s.to_dict() for s in plan.blocked]}

    scoped = scope_frame(plan, df)
    if scoped is None or not len(scoped):
        return {"available": False, "lens": label, "narrowed": narrowed,
                "empty_scope": True,
                "reason": f"no rows in scope for {label}"}

    # EXISTING IMPLEMENTATION, reused. The ITL3 resolution, the per-area
    # weighted LTV and the coverage arithmetic all already exist there, and a
    # second copy would be a second owner of the same economics.
    result = dict(geo_mod.exposure_by_itl3(scoped))
    result["lens"] = label
    result["narrowed"] = narrowed
    result["declaredGroupedBy"] = list(plan.declares_grouped_by)
    return result


# --------------------------------------------------------------------------- #
# CONVERSION 4 — the `dimensions` axis, bridged.
#
# The first NEW contract axis connected to the plan layer since Conversion 2
# bridged `time`. Conversions 1–3 all drew on the two axes already carried, so
# this is the measurement of what adding one costs.
#
# Both accessors below are SHARED: they read the authoritative contract and are
# reusable by any later route needing a grouping or a named start period. They
# know nothing about `funded_bridge`.
# --------------------------------------------------------------------------- #

#: The contract's own role value for a dimension that is an AXIS rather than a
#: selector. Compared as a literal, the way this module already compares
#: `source_scope.state` — the plan layer reads the contract, it does not import
#: its enums.
ROLE_GROUPING = "grouping"


def grouping_concepts(interpretation) -> Tuple[str, ...]:
    """The governed dimension concepts the contract carries AS A GROUPING.

    THE `dimensions` AXIS BRIDGE. Governed field identity only — the concept key
    the parser resolved, never the user's wording — and the ROLE IS OBEYED
    rather than assumed: a dimension the contract marked `filter` or left
    `unresolved` is not returned here, so a caller cannot silently promote a
    selector into an axis. That is the collapse the role split exists to
    prevent, and reading `candidate_concept` while ignoring `role` would
    reintroduce it one call site at a time.

    Order is the contract's order, so a caller wanting "the" grouping takes the
    first and a caller supporting several takes them all.
    """
    out: List[str] = []
    for dim in (getattr(interpretation, "dimensions", None) or ()):
        if getattr(dim, "role", None) != ROLE_GROUPING:
            continue
        concept = getattr(dim, "candidate_concept", None)
        if concept and concept not in out:
            out.append(str(concept))
    return tuple(out)


def comparison_period(interpretation) -> Optional[str]:
    """The named start period the contract carries, or ``None``.

    NOT A NEW AXIS. `time` was bridged by Conversion 2, but only through
    `window_periods` — the magnitude of a trailing window. A question that names
    a period to compare FROM ("since March 2026") states a different fact, and
    it already has a governed home in `time.comparison_period`. This reads that
    field; it does not create a semantic owner and it does not read the
    question.

    Only a FILLED slot answers. An empty one means the question named no start
    period, which is not the same as naming one that could not be resolved.

    DELIBERATELY UNCHANGED by the structural closure below. This returns the
    slot's WORDING, and for a question naming two periods that wording is the
    display join — "October, November". Making it return the first period
    instead would change what this function MEANS, not how it is represented,
    on all five corpus questions that carry a comparison period. The structural
    read is a separate accessor, and the one caller that wants structure asks
    for it by name.
    """
    time = getattr(interpretation, "time", None)
    slot = getattr(time, "comparison_period", None)
    if slot is None or getattr(slot, "state", "empty") != "filled":
        return None
    return getattr(slot, "raw_text", None)


def comparison_periods(interpretation) -> Tuple[str, ...]:
    """The periods the question named to compare, IN ORDER.

    The structural read. `comparison_period` carries the wording a reader is
    shown — `", ".join(...)` — and splitting that back apart is re-parsing a
    serialisation, which breaks on any period label containing the separator.
    This reads the values the contract states.

    Empty when the question named no comparison, which a consumer must
    distinguish from a comparison it could not resolve: the slot's state says
    which, and this deliberately does not guess.
    """
    time = getattr(interpretation, "time", None)
    periods = getattr(time, "comparison_periods", None) or ()
    return tuple(str(p) for p in periods)


def dataset_of(interpretation) -> str:
    """WHICH GOVERNED TAPE the answer is built from — funded, pipeline, forecast.

    THE `dataset` AXIS BRIDGE. `mi_agent_api.workspace.resolve_dataset` is the
    single semantic owner and the contract carries its answer; this reads that
    answer and decides nothing. It does not see the question and it does not
    see the caller's workspace tab — the tab was retired as a semantic input
    precisely so a route could not reintroduce it here.

    Falls back to the governed default when the claim is not FILLED, which is
    the owner's own fallback rather than a second opinion: `resolve_dataset`
    returns `funded` for a question naming no dataset, so an unfilled claim and
    a defaulted one mean the same thing.
    """
    claim = getattr(interpretation, "dataset", None)
    if claim is None or getattr(claim, "state", "empty") != "filled":
        return "funded"
    return str(getattr(claim, "dataset", None) or "funded")


def measure_request(interpretation) -> Tuple[Optional[str], str]:
    """The measure the question asked for, as `(metric, aggregation)`.

    THE `subject` AXIS BRIDGE, and the shape is dictated by the existing
    measure resolvers, which take the parser's two fields. The contract
    deliberately collapses them onto ONE governed concept — resolution belongs
    to the registry, not to the claim — so this expands that concept back into
    the pair the resolver expects, and does so in one place instead of at every
    call site.

    `loan_count` is the concept the projection records when the parser named no
    metric but asked for a count, so it expands to `(None, "count")`; every
    other concept is a metric summed.

    THE KNOWN LOSSY EDGE, stated rather than hidden: a question whose parser
    output was `metric="loan_count"` with a non-count aggregation projects to
    the same concept and would be read here as a count request. Measured across
    all 42 readings of the owned `temporal_compare` surface, production and this
    expansion agree on every one, so the edge is unreachable there today. It is
    a property of the contract collapsing two fields into one, and closing it
    means the claim carrying the aggregation — a contract change, not a route's
    to make.
    """
    subject = getattr(interpretation, "subject", None)
    if subject is None or getattr(subject, "state", "empty") != "filled":
        return None, "sum"
    concept = getattr(subject, "candidate_concept", None)
    if concept == "loan_count":
        return None, "count"
    return (str(concept) if concept else None), "sum"


# --------------------------------------------------------------------------- #
# `temporal_compare`, composed.
# --------------------------------------------------------------------------- #
#: How many periods a comparison needs. Named because the refusal below turns
#: on it and a bare `2` in a guard says nothing about why.
COMPARE_PERIOD_COUNT = 2


def build_temporal_compare_plan(interpretation) -> Plan:
    """The plan for a governed two-period comparison.

    The question is NOT a parameter. Every semantic fact this route ever read
    from it now arrives on the contract: the dataset (the ownership
    remediation), the measure (`subject`), and the period pair (carried
    structurally by the time contract).

    A plan with fewer than two periods is BLOCKED rather than defaulted. The
    shipped route refused the same case in the same words, and defaulting a
    missing period would answer a different question from the one asked.

    THE POPULATION STEP IS DELIBERATELY NOT `_population_step`. This route does
    not narrow by source portfolio — the shipped path never passed a scope to
    the engine, and `lensApplied` is False on every owned case. Planning a
    narrowing it does not apply would put a false narrowing on the receipt; a
    question that NAMES a scope is already refused by the facet layer as a lost
    one, which is the correct owner and is left alone. So the step declares
    what actually happens: the whole dataset.
    """
    periods = comparison_periods(interpretation)
    dataset = dataset_of(interpretation)
    metric, aggregation = measure_request(interpretation)

    period_step = Step(
        STACK_PERIODS,
        {"dataset": dataset, "take": "named_pair", "periods": list(periods),
         "disclose": "availablePeriods"},
        because="a comparison opens at the first named period and closes at the second")
    if len(periods) < COMPARE_PERIOD_COUNT:
        period_step = Step(
            STACK_PERIODS,
            {"dataset": dataset, "take": "named_pair", "periods": list(periods)},
            because="a comparison needs two governed reporting periods",
            blocked=(BLOCKED_NO_CONTRACT_FIELD + ": time.comparison_periods "
                     f"names {len(periods)} period(s), not {COMPARE_PERIOD_COUNT}"))

    steps: List[Step] = [
        period_step,
        _whole_dataset_step("temporal_compare", dataset),
        Step(RESOLVE_MEASURE, {"metric": metric, "aggregation": aggregation},
             because="the contract's subject concept, expanded for the resolver"),
        Step(COMPARE,
             {"of": ["period_a", "period_b"], "as": "absolute and percentage delta",
              "direction": "b relative to a"},
             because="the comparison reports the movement from the first period to the second"),
    ]
    return Plan(tuple(steps))


def compare_period_pair(plan: Plan) -> Tuple[Optional[str], Optional[str]]:
    """The ordered pair this plan compares, from the plan alone."""
    step = next((s for s in plan.steps if s.primitive == STACK_PERIODS), None)
    periods = list(step.inputs.get("periods") or ()) if step else []
    if len(periods) < COMPARE_PERIOD_COUNT:
        return None, None
    return periods[0], periods[1]


def compare_dataset(plan: Plan) -> Optional[str]:
    """The tape this plan runs against, from the plan alone."""
    step = next((s for s in plan.steps if s.primitive == SELECT_POPULATION), None)
    return (step.inputs.get("dataset") if step else None)


def temporal_compare(output_root, pipeline_root, client_id: str,
                     to_run_id: Optional[str], *, interpretation) -> Dict[str, Any]:
    """A governed two-period comparison, COMPOSED.

    A drop-in for `temporal_compare.run_temporal_compare` as this route called
    it: the dataset, the measure and the period pair all come from the contract,
    and the same result dict comes back — so the prose, the chart, the table,
    the envelope and the receipt are unchanged by construction.
    """
    from . import temporal_compare as compare_mod

    plan = build_temporal_compare_plan(interpretation)
    if plan.blocked:
        return {"available": False, "reason": plan.blocked[0].blocked,
                "planBlocked": [s.to_dict() for s in plan.blocked]}

    period_a, period_b = compare_period_pair(plan)
    metric_step = next(s for s in plan.steps if s.primitive == RESOLVE_MEASURE)
    # EXISTING IMPLEMENTATION, reused. The period matching, the deltas, the
    # direction and the insufficient-data response already live there;
    # re-deriving them would add a second owner of the same economics.
    return compare_mod.run_temporal_compare(
        output_root, pipeline_root, client_id, to_run_id,
        dataset=compare_dataset(plan),
        metric=metric_step.inputs.get("metric"),
        aggregation=metric_step.inputs.get("aggregation"),
        period_a=period_a, period_b=period_b)


# --------------------------------------------------------------------------- #
# `funded_bridge`, composed.
# --------------------------------------------------------------------------- #

#: How many contributors the waterfall keeps before aggregating the residual
#: into "Other". The route's own long-standing constant.
BRIDGE_TOP_N = 8


def build_funded_bridge_plan(interpretation, *, dimension_key: Optional[str],
                             dimension_label: str) -> Plan:
    """The plan for a funded-balance attribution bridge.

    The question is NOT a parameter. Both semantic facts this route ever read
    from it now arrive on the contract: the source scope (Conversion 1) and the
    attribution dimension (bridged above). The start period is read from the
    contract too, and declared so the plan states the window it compares over.

    `dimension_key` is the GOVERNED CONCEPT the registry resolved for the
    contract's grouping; the executor turns it into the column(s) the tape
    carries. A plan with no dimension is BLOCKED rather than defaulted here —
    the route's fallback is its own convention and belongs at its own call site,
    not inside a step that claims the contract asked for it.
    """
    # THE STRUCTURAL READ. `comparison_period` returns the slot's WORDING, which
    # for a two-period question is the display join "October, November" — a
    # start period no tape has. The pair is on the contract now, so the plan
    # takes the first period from it rather than from a rendered string.
    #
    # Proved representation-only before it was switched: across all 12 cases
    # executed routing shows this route owns, the structural first period and
    # the join are IDENTICAL — every bridge question names at most one period.
    # The fallback keeps a contract built by an older projection working.
    _periods = comparison_periods(interpretation)
    _from = _periods[0] if _periods else comparison_period(interpretation)
    # A WINDOW IS A PERIOD STATEMENT TOO.
    #
    # A question can pin the opening period by NAMING it ("from October") or by
    # stating how far back it reaches ("last month", "over the last 3 months").
    # This plan read only the first, so "show the balance bridge for last month"
    # arrived with `comparison_periods=[]`, `window_periods=1`, and opened at
    # the EARLIEST snapshot instead: a bridge labelled for one month that showed
    # five, +£59.2m where the month moved +£22.6m.
    #
    # `window_periods` is the contract's own magnitude — the same field
    # Conversion 2 and C7 read for a span — so the window is declared here and
    # the executor opens that many periods back. No wording is read anywhere.
    _window = getattr(getattr(interpretation, "time", None), "window_periods", None)
    _period_inputs = {"dataset": "funded", "take": "pair", "from": _from,
                      "disclose": "periodsAvailable"}
    if _from is None and _window:
        _period_inputs["window_periods"] = int(_window)
    steps: List[Step] = [
        Step(STACK_PERIODS, _period_inputs,
             because=("a bridge opens at a named start period, else the period "
                      "the stated window reaches back to, else the earliest "
                      "governed period, and closes at the latest")),
        _population_step(getattr(interpretation, "source_scope", None)),
        Step(RESOLVE_MEASURE, {"metric": "funded_balance", "aggregation": "sum"},
             because="the bridge attributes movement in the funded balance"),
    ]
    if not dimension_key:
        steps.append(Step(
            GROUP, {"by": None},
            because="a bridge attributes movement BY a dimension",
            blocked=(BLOCKED_NO_CONTRACT_FIELD + ": no governed dimension carries "
                     "a grouping role, and none is available to fall back to")))
        return Plan(tuple(steps), ())
    steps.append(Step(GROUP, {"by": [dimension_key], "measure": "funded_balance",
                              "aggregation": "sum", "of": "the delta",
                              "label": dimension_label, "top_n": BRIDGE_TOP_N,
                              "residual": "Other"},
                      because="the waterfall attributes the movement by this axis"))
    steps.append(Step(COMPARE,
                      {"of": ["funded_balance"], "between": "start period and latest",
                       "as": "absolute delta", "reconciles_to": "netChange"},
                      because="the per-category deltas sum exactly to the net change"))
    return Plan(tuple(steps), (dimension_key,))


def bridge_start_period(plan: Plan) -> Optional[str]:
    """The start period this plan opens at, from the plan alone."""
    step = next((s for s in plan.steps if s.primitive == STACK_PERIODS), None)
    return (step.inputs.get("from") if step else None)


def bridge_window_periods(plan: Plan) -> Optional[int]:
    """How many governed periods back this plan opens, when it states a window."""
    step = next((s for s in plan.steps if s.primitive == STACK_PERIODS), None)
    return (step.inputs.get("window_periods") if step else None)


def funded_bridge(output_root, client_id: str, *, interpretation,
                  dimension_columns, dimension_key: Optional[str],
                  dimension_label: str,
                  to_run_id: Optional[str] = None) -> Dict[str, Any]:
    """A governed funded-balance attribution bridge, COMPOSED.

    A drop-in for `evolution.funded_bridge` as this route called it: the
    population, the attribution axis and the start period all come from the
    contract, and the same result dict comes back — so the waterfall, the table,
    the prose, the envelope and the receipt are unchanged by construction.

    `dimension_columns` is the registry's resolution of the governed concept
    into the column(s) this tape may carry; the plan owns WHICH concept, the
    registry owns how it is spelled.
    """
    from . import evolution as evolution_mod

    plan = build_funded_bridge_plan(interpretation, dimension_key=dimension_key,
                                    dimension_label=dimension_label)
    label = lens_label(plan)
    if plan.blocked:
        return {"available": False, "lens": label,
                "reason": plan.blocked[0].blocked,
                "planBlocked": [s.to_dict() for s in plan.blocked]}

    # EXISTING IMPLEMENTATION, reused. The period pair, the per-category deltas,
    # the "Other" residual and the missing-dimension refusal all already exist
    # there — re-deriving them would add a second owner of the same economics,
    # and the missing-dimension guard is the one that keeps a bridge on an
    # absent column from reporting GBP0 for a book that moved.
    out = evolution_mod.funded_bridge(
        output_root, client_id, dimension_columns,
        start_period=bridge_start_period(plan), to_run_id=to_run_id,
        window_periods=bridge_window_periods(plan),
        lens_filters=lens_filters(plan), lens_label=label,
        top_n=BRIDGE_TOP_N)
    if out.get("available"):
        out["declaredGroupedBy"] = list(plan.declares_grouped_by)
    return out
