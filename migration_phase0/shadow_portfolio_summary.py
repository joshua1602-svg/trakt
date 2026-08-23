#!/usr/bin/env python3
"""migration_phase0/shadow_portfolio_summary.py — the first migration slice, shadow only.

SHADOW ONLY. Nothing here is imported by production, wired into a route, or
placed behind a flag. It exists to answer one question with evidence:

    Can `portfolio_summary` be expressed as a composition of the seven derived
    primitives, fed from the INTERPRETATION CONTRACT rather than from the raw
    question, and produce the shipped answer's economics exactly?

THE PRIMITIVES USED (5 of 7; `compare` and `project` are not needed):

    stack periods     mi_agent_api.evolution.funded_frames
    select population (BLOCKED — see below)
    resolve measure   mi_agent_api.evolution.assemble_funded_evolution metrics
    group             balance by region; balance by source portfolio
    rank              largest-first, truncated to TOP_REGIONS

THE HARD RULE THIS MODULE OBEYS
-------------------------------
``build_plan`` receives a :class:`QuestionInterpretation` and a context. It
NEVER receives, reads, inspects or is passed the raw question string. That is
enforced by its signature and asserted by ``assert_no_question_read``.

THE BLOCKER, AND HOW IT WAS CLEARED (Phase 1A)
----------------------------------------------
Phase 0 blocked 9 of 9 cases here. The shipped route narrows by PORTFOLIO LENS
(``direct`` / ``acquired`` / an SPV id) resolved by
``mi_agent.portfolio_lens.resolve_lens(question)`` — from the RAW QUESTION — and
the interpretation contract carried no claim for it, so an empty ``population``
list could mean either "the whole book" or "nobody looked".

Phase 1A added ``QuestionInterpretation.source_scope``, which carries THAT SAME
OWNER's reading. ``mi_agent.portfolio_lens`` is still the only thing that decides
what "the acquired book" means; the contract transports its answer, and this
plan consumes the transported answer.

What has NOT changed: ``state`` still decides. EMPTY and UNRESOLVABLE are not
Total and still block. Only ``FILLED`` plans a selection, and ``scope=total`` is
a positive reading — "the owner looked and found no source narrowing" — not an
absence.

    python -m migration_phase0.shadow_portfolio_summary
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: Primitive ids, as derived by the scoping study.
SELECT_POPULATION = "select_population"
RESOLVE_MEASURE = "resolve_measure"
GROUP = "group"
STACK_PERIODS = "stack_periods"
RANK = "rank"

#: Why a step could not be planned. A plan carrying one of these is a REFUSAL,
#: never an answer with the step quietly omitted.
BLOCKED_NO_CONTRACT_FIELD = "no interpretation-contract field carries this"


@dataclass(frozen=True)
class Step:
    """One primitive invocation in a shadow plan."""

    primitive: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    because: str = ""
    blocked: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {"primitive": self.primitive, "inputs": dict(self.inputs),
               "because": self.because}
        if self.blocked:
            out["blocked"] = self.blocked
        return out


@dataclass(frozen=True)
class ShadowPlan:
    steps: Tuple[Step, ...]
    #: Field keys the plan DECLARES it grouped by — the evidence
    #: ``execution_receipt.grouping_proven`` reads. Declared by the plan, not by
    #: a route name.
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
                "executable": self.executable,
                "blocked": [s.to_dict() for s in self.blocked]}


# --------------------------------------------------------------------------- #
# Planning — interpretation in, plan out. The question NEVER enters.
# --------------------------------------------------------------------------- #
def build_plan(interpretation, *, region_column: Optional[str],
               has_portfolio_column: bool) -> ShadowPlan:
    """The shadow plan for a portfolio-summary question.

    ``interpretation`` is a :class:`question_interpretation.schema.QuestionInterpretation`.
    The raw question is deliberately NOT a parameter.
    """
    steps: List[Step] = [
        Step(STACK_PERIODS,
             {"dataset": "funded", "take": "latest", "disclose": "periodCount"},
             because="the headline position is the latest governed snapshot, and "
                     "the count of available periods is disclosed"),
    ]

    # -- select population, from the contract ONLY --------------------------
    #
    # PHASE 1A. `source_scope` carries `mi_agent.portfolio_lens`'s reading, so
    # this step is now planned from the contract. `state` is what decides:
    # EMPTY and UNRESOLVABLE are NOT Total, and both block. Reading either as
    # Total would widen a population the question may have narrowed.
    scope = getattr(interpretation, "source_scope", None)
    state = getattr(scope, "state", "empty")
    if state == "filled":
        steps.append(Step(
            SELECT_POPULATION,
            {"kind": "source_portfolio_lens", "scope": scope.scope,
             "portfolio_ids": list(scope.portfolio_ids)},
            because=("the contract carries a resolved source scope "
                     f"({scope.scope!r})"
                     + ("; it narrows nothing" if not scope.narrows else "")))) 
    else:
        steps.append(Step(
            SELECT_POPULATION, {"kind": "source_portfolio_lens"},
            because="the shipped route narrows by portfolio lens",
            blocked=(BLOCKED_NO_CONTRACT_FIELD + ": source_scope is "
                     f"{state!r}" + (f" ({scope.reason})" if getattr(
                         scope, "reason", None) else "")
                     + ". Absence of a resolved scope is NOT Total.")))

    # A seasoning population is a DIFFERENT AXIS and is planned separately. A
    # question can carry both, and neither implies the other.
    seasoning = [p for p in interpretation.population
                 if p.concept == "seasoning_segment" and p.state == "filled"]
    if seasoning:
        steps.append(Step(
            SELECT_POPULATION,
            {"kind": "seasoning_segment", "claim": seasoning[0].raw_text},
            because="the contract carries a filled seasoning population claim"))

    # -- the five governed headline measures --------------------------------
    for metric, aggregation in (("funded_balance", "sum"),
                                ("loan_count", "count"),
                                ("wa_ltv", "weighted_avg"),
                                ("wa_interest_rate", "weighted_avg"),
                                ("avg_borrower_age", "avg")):
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
    return ShadowPlan(tuple(steps), tuple(grouped_by))


def assert_no_question_read(interpretation) -> None:
    """`build_plan` must not be reachable from the raw question.

    Cheap structural guard: the plan builder's signature carries no question
    parameter, and the interpretation object it does receive is used only through
    its typed claims. Asserted here so a future edit that adds a question
    parameter fails loudly rather than silently reintroducing a second semantic
    owner.
    """
    import inspect
    params = set(inspect.signature(build_plan).parameters)
    forbidden = params & {"question", "q", "text", "raw", "raw_question"}
    if forbidden:
        raise AssertionError(
            "build_plan must never receive the raw question; found %s" % sorted(forbidden))


# --------------------------------------------------------------------------- #
# Execution — existing primitives, unmodified
# --------------------------------------------------------------------------- #
def lens_for(plan: ShadowPlan):
    """The governed lens this PLAN selects, or ``None`` if it selects none.

    The scope name comes from the plan (which got it from the contract, which
    got it from the owner). Turning that name into row filters is asked of the
    SAME owner, through the path it already exposes for an explicit selection —
    so no question text is read and no second resolver exists.

    ``lens_from_selection`` falls back to Total for anything it does not
    recognise, so the rebuilt lens is CHECKED against the scope the plan
    claimed. A silent fallback to Total here would be the population widening
    this whole exercise exists to prevent.
    """
    from mi_agent import portfolio_lens as lens_owner

    step = next((s for s in plan.steps
                 if s.primitive == SELECT_POPULATION
                 and s.inputs.get("kind") == "source_portfolio_lens"
                 and not s.blocked), None)
    if step is None:
        return None
    scope = step.inputs.get("scope")
    ids = step.inputs.get("portfolio_ids") or []
    lens = lens_owner.lens_from_selection(ids if ids else scope)
    if lens.name != scope:
        raise AssertionError(
            f"the lens owner rebuilt {lens.name!r} from a plan claiming "
            f"{scope!r}; refusing rather than narrowing to the wrong population")
    return lens


def execute_plan(plan: ShadowPlan, *, output_root: str, client_id: str
                 ) -> Dict[str, Any]:
    """Run the plan with the primitives that already ship.

    Every input comes from the PLAN. Nothing is supplied by the caller, and the
    raw question is not reachable from here.
    """
    from mi_agent_api import evolution as evolution_mod
    from mi_agent_api import movement_summary as summary_mod

    lens = lens_for(plan)
    lens_filters = (lens.filters or None) if lens is not None else None

    frames = evolution_mod.funded_frames(output_root, client_id, None)
    frames = [{**f, "df": evolution_mod._scope_frame_lens(f.get("df"), lens_filters)}
              for f in frames]
    frames = [f for f in frames if f["df"] is not None and len(f["df"])]
    if not frames:
        return {"available": False,
                "reason": "no governed reporting period is available for this scope"}

    evo = evolution_mod.assemble_funded_evolution(frames, client_id, None)
    periods = evo.get("periods") or []
    if not periods:
        return {"available": False,
                "reason": "no governed reporting period is available for this scope"}

    current = periods[-1]
    df = frames[-1]["df"]
    region_col = summary_mod._region_column(df)
    return {
        "available": True,
        "period": current.get("period"),
        "reportingDate": current.get("reporting_date"),
        "metrics": summary_mod._metrics(current),
        "regionColumn": region_col,
        "topRegions": summary_mod._regional_exposure(df, region_col) if region_col else [],
        "cohorts": summary_mod._cohorts(df),
        "periodCount": len(periods),
        "lensFromPlan": {"scope": getattr(lens, "name", None),
                         "filters": dict(lens_filters or {})},
        "declaredGroupedBy": list(plan.declares_grouped_by),
    }
