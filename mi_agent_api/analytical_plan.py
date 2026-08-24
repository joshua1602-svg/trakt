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
            {"kind": "source_portfolio_lens",
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
                 and s.inputs.get("kind") == "source_portfolio_lens"
                 and not s.blocked), None)
    if step is None:
        return None
    ids = list(step.inputs.get("portfolio_ids") or [])
    return {"source_portfolio_id": ids} if ids else None


def lens_label(plan: Plan) -> str:
    step = next((s for s in plan.steps
                 if s.primitive == SELECT_POPULATION and not s.blocked), None)
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
    steps: List[Step] = [
        Step(STACK_PERIODS,
             {"dataset": "funded", "take": "pair", "from": _from,
              "disclose": "periodsAvailable"},
             because=("a bridge opens at a named start period, else the earliest "
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
        lens_filters=lens_filters(plan), lens_label=label,
        top_n=BRIDGE_TOP_N)
    if out.get("available"):
        out["declaredGroupedBy"] = list(plan.declares_grouped_by)
    return out
