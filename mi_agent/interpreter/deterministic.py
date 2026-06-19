"""mi_agent.interpreter.deterministic — rule-based baseline interpreter (Phase 8A).

A small, deterministic, keyword-driven interpreter that maps a controlled set of
business questions onto MIQuerySpec v2 dicts, then normalises + validates them.
It is the **baseline + grading harness**, NOT the final LLM interpreter. It calls
no external services and computes no analytics.

Every produced spec is passed through ``MIQuerySpec.normalized()`` and
``validate_query_spec()`` — an interpretation is only "ok" if it validates.
Ambiguous/under-specified questions return a clarification instead of guessing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.mi_spec_validation import validate_query_spec

from .models import DETERMINISTIC, InterpretationResult, InterpreterContext


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(text).strip().lower())


def _has(q: str, *terms: str) -> bool:
    return all(t in q for t in terms)


def _any(q: str, *terms: str) -> bool:
    return any(t in q for t in terms)


def _state_of(q: str) -> Optional[str]:
    if _any(q, "forecast"):
        return "total_forecast_funded"
    if _any(q, "pipeline"):
        return "total_pipeline"
    if _any(q, "funded", "funding", "book"):
        return "total_funded"
    return None


def _build(question: str, spec_dict: Dict[str, Any], ctx: InterpreterContext,
           confidence: float = 1.0) -> InterpretationResult:
    spec = MIQuerySpec.from_dict(spec_dict).normalized()
    vr = validate_query_spec(spec)
    return InterpretationResult(
        raw_question=question, candidate_spec=spec_dict, normalized_spec=spec,
        validation_result=vr, confidence=confidence, issues=list(vr.issues),
        clarification_required=False, interpretation_method=DETERMINISTIC)


def _clarify(question: str, ask: str) -> InterpretationResult:
    return InterpretationResult(
        raw_question=question, clarification_required=True,
        clarification_question=ask, confidence=None,
        interpretation_method=DETERMINISTIC)


# Dimension keywords → resolved field (governed, never bare ambiguous terms).
def _concentration_dimension(q: str, ctx: InterpreterContext
                             ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (dimension, bucket_strategy, clarification) for a 'by X' phrase."""
    if "portfolio" in q:
        if not ctx.portfolio_config_available:
            return None, None, ("Which Trakt portfolio reference? No portfolio "
                                "reference config is available for this client.")
        return "portfolio_id", None, None
    if "region" in q:
        return "geographic_region_obligor", None, None
    if "broker" in q:
        return "broker_channel", None, None
    if "stage" in q:
        return "pipeline_stage", None, None
    if _any(q, "balance band", "balance bucket"):
        return "balance_band", "quantile", None
    if _any(q, "interest rate", "rate bucket"):
        return "interest_rate_bucket", "quantile", None
    if _any(q, "time on book", "months on book"):
        return "time_on_book_bucket", "quantile", None
    if "ltv" in q:
        return "ltv_bucket", "configured", None
    return None, None, None


def interpret(question: str,
              context: Optional[InterpreterContext] = None) -> InterpretationResult:
    """Interpret a business question into a validated MIQuerySpec v2 (or a
    clarification)."""
    ctx = context or InterpreterContext()
    q = _norm(question)
    base = {"route_id": ctx.route_id, "snapshot_client_id": ctx.snapshot_client_id}

    # ---- Bucket views ("show X buckets") ---------------------------------- #
    if "bucket" in q or _any(q, "time on book", "ltv"):
        dim, strategy, clar = _concentration_dimension(q, ctx)
        if dim:
            spec = {**base, "risk_monitor_mode": "concentration",
                    "state": "total_funded", "concentration_dimension": dim}
            if strategy:
                spec["bucket_strategy"] = strategy
                if strategy == "quantile":
                    spec["bucket_field"] = {
                        "balance_band": "current_outstanding_balance",
                        "interest_rate_bucket": "current_interest_rate",
                        "time_on_book_bucket": "months_on_book",
                    }.get(dim)
            return _build(question, spec, ctx)

    # ---- Risk migration --------------------------------------------------- #
    if "migration" in q or _has(q, "migrate"):
        if "ifrs" in q:
            mdim = "ifrs9_stage"
        elif _any(q, "pd"):
            mdim = "pd_bucket"
        elif _any(q, "grade", "rating"):
            mdim = "internal_risk_grade"
        else:
            return _clarify(question, "Which risk dimension should migrate — "
                            "risk grade, IFRS 9 stage, or PD bucket?")
        return _build(question, {**base, "risk_monitor_mode": "migration",
                                 "migration_dimension": mdim,
                                 "baseline_date": ctx.prev_period,
                                 "current_date": ctx.as_of}, ctx)

    # ---- Risk deterioration flags ----------------------------------------- #
    if _has(q, "risk", "deteriorat"):
        return _build(question, {**base, "risk_monitor_mode": "flags",
                                 "risk_dimension": "internal_risk_grade",
                                 "baseline_date": ctx.prev_period,
                                 "current_date": ctx.as_of}, ctx)

    # ---- Concentration ("concentration by X" / "too concentrated by X") --- #
    if _any(q, "concentrat"):
        dim, strategy, clar = _concentration_dimension(q, ctx)
        if clar:
            return _clarify(question, clar)
        if dim:
            return _build(question, {**base, "risk_monitor_mode": "concentration",
                                     "state": "total_funded",
                                     "concentration_dimension": dim}, ctx)
        return _clarify(question, "Concentration by which dimension — region, "
                        "broker, product, risk grade?")

    # ---- Temporal: trend over last three months --------------------------- #
    if _has(q, "trend") or _has(q, "over the last three months"):
        state = _state_of(q) or "total_funded"
        return _build(question, {**base, "state": state, "temporal_mode": "trend",
                                 "start_date": ctx.range_start,
                                 "end_date": ctx.as_of, "trend_grain": "monthly"},
                      ctx)

    # ---- Temporal: compare to last month / what changed ------------------- #
    if _has(q, "compare") or _has(q, "since last month") \
            or (_has(q, "changed") and "month" in q):
        state = _state_of(q) or "total_funded"
        return _build(question, {**base, "state": state, "temporal_mode": "compare",
                                 "baseline_date": ctx.prev_period,
                                 "current_date": ctx.as_of,
                                 "comparison_basis": "balance"}, ctx)

    # ---- Concentration / breakdown "by X" of a state ---------------------- #
    if " by " in f" {q} ":
        state = _state_of(q) or "total_funded"
        dim, strategy, clar = _concentration_dimension(q, ctx)
        if clar:
            return _clarify(question, clar)
        if dim:
            spec = {**base, "risk_monitor_mode": "concentration", "state": state,
                    "concentration_dimension": dim}
            if strategy:
                spec["bucket_strategy"] = strategy
            return _build(question, spec, ctx)
        return _clarify(question, "Break down by which dimension?")

    # ---- Current state ("show total funded/pipeline/forecast") ------------ #
    state = _state_of(q)
    if state and _any(q, "total", "show", "current") and "stage" not in q:
        return _build(question, {**base, "state": state,
                                 "temporal_mode": "latest"}, ctx)

    # ---- Ambiguous / clarification --------------------------------------- #
    if q.strip() in ("show stage", "stage") or (_has(q, "stage")
                                                and state is None):
        return _clarify(question, "'Stage' is ambiguous. Did you mean pipeline "
                        "stage (in a pipeline view), IFRS 9 stage, or internal "
                        "risk stage?")
    if "portfolio" in q:
        return _clarify(question, "Which portfolio measure? e.g. 'funded balance "
                        "by portfolio'. (A Trakt portfolio reference config is "
                        "required.)")
    if "risk" in q:
        return _clarify(question, "What about risk — migration (grade/IFRS 9/PD), "
                        "deterioration flags, or concentration?")
    if _any(q, "change", "changed"):
        return _clarify(question, "Over what period? e.g. 'compare funded balance "
                        "to last month'.")
    if "rate" in q:
        return _clarify(question, "Do you mean average interest rate, or interest "
                        "rate buckets, and over which population?")

    return _clarify(question, "Could you rephrase? I can answer funded / pipeline "
                    "/ forecast state, trends, comparisons, concentration, and "
                    "risk migration questions.")
