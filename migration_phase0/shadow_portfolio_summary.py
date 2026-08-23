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

WHY IT IS BLOCKED
-----------------
The shipped route narrows by PORTFOLIO LENS (``direct`` / ``acquired`` / an SPV
id), resolved by ``mi_agent.portfolio_lens.resolve_lens(question)`` — from the
RAW QUESTION. The interpretation contract has no claim that carries it:
measured, ``QuestionInterpretation.population`` is EMPTY for "Summarise the
acquired book" while the route resolves ``source_portfolio_type=acquired``.

An empty ``population`` therefore cannot be read as "no narrowing was asked
for": absence-of-claim is not evidence-of-total when the concept is
unrepresentable. So this plan DECLARES the lens unresolvable rather than
assuming Total, and the equivalence run below reports which questions that
blocks. Assuming Total would be the silent population widening the P1L work
exists to prevent.

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
    seasoning = [p for p in interpretation.population
                 if p.concept == "seasoning_segment" and p.state == "filled"]
    if seasoning:
        steps.append(Step(
            SELECT_POPULATION,
            {"kind": "seasoning_segment", "claim": seasoning[0].raw_text},
            because="the contract carries a filled seasoning population claim"))
    else:
        # THE BLOCKER. See the module docstring: an empty population list cannot
        # be read as Total while the portfolio lens is unrepresentable.
        steps.append(Step(
            SELECT_POPULATION, {"kind": "portfolio_lens"},
            because="the shipped route narrows by portfolio lens",
            blocked=BLOCKED_NO_CONTRACT_FIELD + ": QuestionInterpretation has no "
                    "claim for a source-portfolio lens (direct / acquired / SPV "
                    "id); mi_agent.portfolio_lens.resolve_lens reads the raw "
                    "question instead"))

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
def execute_plan(plan: ShadowPlan, *, output_root: str, client_id: str,
                 lens_filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Run the plan with the primitives that already ship.

    ``lens_filters`` is supplied by the CALLER for the equivalence run only, and
    the result records that it was supplied externally — it is exactly the value
    the plan could not obtain from the contract.
    """
    from mi_agent_api import evolution as evolution_mod
    from mi_agent_api import movement_summary as summary_mod

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
        "lensFiltersSuppliedExternally": lens_filters or {},
        "declaredGroupedBy": list(plan.declares_grouped_by),
    }
