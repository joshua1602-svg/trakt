#!/usr/bin/env python3
"""Phase 3A — assembling the Weekly Portfolio Brief.

Three jobs, kept apart on purpose:

  resolve   gather the governed inputs — the same prepared frames the charts
            use, the same movement payloads the hovers use, the same
            concentration and funnel outputs the workspace uses;
  generate  run each generator, isolating its failure;
  select    order deterministically, apply caps, and cut to the brief limit.

Nothing here calculates a metric. Every number in the brief comes from a
generator, which in turn comes from an existing governed output or from
``insight_metrics``. There is no second source of economic truth.

Partial failure is a first-class outcome: one generator raising must not cost
the reader the other seven. A failed type becomes an ``Omission`` with category
``error`` and the brief status becomes ``partial`` — the reader is told a
section is missing rather than being shown a brief that looks complete.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from trakt_core import perf as _perf

from . import insight_config as cfg
from . import insight_generators as gen
from . import insight_metrics as metrics
from . import movement_detail as md
from .insight_contract import (
    OMITTED_ERROR, SEVERITY_RANK, TYPE_PRIORITY, Insight, Omission, build_brief,
)

logger = logging.getLogger("mi_agent_api.insights")

FLAG_ENV = "TRAKT_MI_WEEKLY_BRIEF"
_ON = ("1", "true", "on", "yes", "enabled")


def weekly_brief_enabled() -> bool:
    """Read per call, so a test (and an operator) can toggle without a reload."""
    import os
    return (os.environ.get(FLAG_ENV) or "").strip().lower() in _ON


# --------------------------------------------------------------------------- #
# Selection — deterministic, no LLM, no randomness
# --------------------------------------------------------------------------- #
def rank_key(insight: Insight) -> Tuple[int, int, str, str]:
    """Total order over insights.

    Severity first (a breach outranks an observation whatever its type), then
    the type's base priority (concentration and data quality above the rest),
    then type name and discriminator so two insights that tie still order
    identically on every run.
    """
    return (
        -SEVERITY_RANK.get(insight.severity, 0),
        -TYPE_PRIORITY.get(insight.insight_type, 0),
        insight.insight_type,
        insight.discriminator or "",
    )


def select(insights: List[Insight], *, limits: Optional[Dict[str, Any]] = None
           ) -> Tuple[List[Insight], List[Omission]]:
    """Order, cap per type, cut to the brief limit.

    Anything dropped by a cap becomes an explicit omission — a brief that
    silently truncated would read as "this is everything", which it would not
    be. ``priority`` is stamped here rather than by the generators, so ordering
    can be tuned without changing what an insight claims about itself.
    """
    lim = limits or cfg.brief_limits()
    max_total = int(lim.get("max_insights", 8))
    per_type = lim.get("max_per_type") or {}

    ordered = sorted(insights, key=rank_key)
    kept: List[Insight] = []
    dropped: Dict[str, int] = {}
    seen: Dict[str, int] = {}

    for ins in ordered:
        cap = int(per_type.get(ins.insight_type, max_total))
        if seen.get(ins.insight_type, 0) >= cap or len(kept) >= max_total:
            dropped[ins.insight_type] = dropped.get(ins.insight_type, 0) + 1
            continue
        seen[ins.insight_type] = seen.get(ins.insight_type, 0) + 1
        kept.append(ins)

    for i, ins in enumerate(kept):
        # Descending, so the first card carries the highest number.
        ins.priority = max_total * 10 - i * 10

    omissions = [
        Omission(t, f"{n} further {t} insight(s) were not shown: the brief is "
                    f"capped at {int(per_type.get(t, max_total))} of this type "
                    f"and {max_total} in total.", "capped")
        for t, n in sorted(dropped.items())
    ]
    return kept, omissions


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def _safe(name: str, fn: Callable[[], Any], default: Any = None) -> Any:
    """Run one resolution step; a failure degrades that input, not the brief."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        logger.warning("weekly brief: %s unavailable (%s)", name, exc)
        return default


@_perf.stage_fn("weekly_brief_resolve")
def resolve_inputs(root: str, client_id: str, *, as_of: Optional[str],
                   historical_model: Optional[Dict[str, Any]],
                   scope: Optional[str],
                   concentration_snapshot: Optional[Dict[str, Any]],
                   funnel: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Gather every governed input the generators need, once.

    The two weekly frames are loaded ONCE and shared by the movement payloads
    and by every new measure, so the brief costs the same two cached frame
    lookups the pipeline hover already costs. ``historical_model`` is passed
    through purely so those lookups hit the frames the charts already prepared
    rather than preparing them again under a different cache key.
    """
    from . import pipeline_contract as pipeline_mod

    inv = pipeline_mod.weekly_extract_inventory(root, client_id)
    cur_e, pri_e = md.select_pair(inv.get("extracts", []), as_of)
    out: Dict[str, Any] = {
        "current_extract": cur_e, "comparison_extract": pri_e,
        "as_of_date": (cur_e or {}).get("pipeline_extract_date"),
        "comparison_date": (pri_e or {}).get("pipeline_extract_date"),
        "run_id": (cur_e or {}).get("run_id"),
        "concentration": concentration_snapshot,
        "funnel": funnel,
    }
    if cur_e is None:
        out["frames"] = (None, None)
        return out

    cur = _safe("current frame", lambda: pipeline_mod.load_prepared_pipeline(
        cur_e, historical_model=historical_model)[0])
    pri = (_safe("comparison frame", lambda: pipeline_mod.load_prepared_pipeline(
        pri_e, historical_model=historical_model)[0]) if pri_e is not None else None)
    out["frames"] = (cur, pri)

    if pri is not None:
        out["pipeline_detail"] = _safe("pipeline movement", lambda: md.build_movement_detail(
            md.DETAIL_PIPELINE, cur, pri,
            as_of_date=out["as_of_date"], comparison_date=out["comparison_date"],
            portfolio_id=client_id, scope=scope, run_id=out["run_id"],
            source_file=md._basename(cur_e.get("source_file")),
            comparison_source_file=md._basename(pri_e.get("source_file"))))
        out["completions_detail"] = _safe("completions movement", lambda: md.build_movement_detail(
            md.DETAIL_COMPLETIONS, cur, pri,
            as_of_date=out["as_of_date"], comparison_date=out["comparison_date"],
            portfolio_id=client_id, scope=scope, run_id=out["run_id"]))
        out["ticket"] = _safe("ticket size", lambda: metrics.ticket_size(cur, pri))
        out["ltv"] = _safe("weighted ltv", lambda: metrics.weighted_ltv(cur, pri))
        out["ticket_mix"] = _safe("ticket mix", lambda: metrics.band_mix(
            cur, pri, metrics.TICKET_BAND))
        out["ltv_mix"] = _safe("ltv mix", lambda: metrics.band_mix(
            cur, pri, metrics.LTV_BAND))
        out["data_quality"] = _safe("data quality", lambda: metrics.data_quality(
            cur, pri, (out.get("pipeline_detail") or {}).get("methodology")))
    return out


def _conversion_inputs(funnel: Optional[Dict[str, Any]]
                       ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """The COMPLETED stage's governed conversion block and cohort percentage."""
    if not funnel:
        return None, None
    summary = (funnel.get("summary") or {}).get("COMPLETED") or {}
    conv = summary.get("conversion")
    cohort = None
    cohorts = funnel.get("cohortConversion") or funnel.get("cohortSeries")
    if isinstance(cohorts, dict):
        series = cohorts.get("COMPLETED")
        if isinstance(series, list) and series:
            last = series[-1]
            cohort = last.get("pct") if isinstance(last, dict) else None
    return conv, cohort


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
@_perf.stage_fn("weekly_brief_build")
def build(root: str, client_id: str, *, tenant_id: str,
          as_of: Optional[str] = None,
          historical_model: Optional[Dict[str, Any]] = None,
          scope: Optional[str] = None,
          concentration_snapshot: Optional[Dict[str, Any]] = None,
          funnel: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The Weekly Portfolio Brief for one portfolio scope and one week."""
    conf = cfg.load()
    ctx_scope = scope or "total"

    data = resolve_inputs(
        root, client_id, as_of=as_of, historical_model=historical_model,
        scope=ctx_scope, concentration_snapshot=concentration_snapshot,
        funnel=funnel)

    ctx = {
        "tenant_id": tenant_id, "portfolio_id": client_id,
        "portfolio_context": ctx_scope, "run_id": data.get("run_id"),
        "as_of_date": data.get("as_of_date"),
        "comparison_date": data.get("comparison_date"),
    }

    if data.get("current_extract") is None:
        return build_brief(
            [], [], tenant_id=tenant_id, portfolio_id=client_id,
            portfolio_context=ctx_scope, as_of_date=as_of, comparison_date=None,
            config_source=conf.get("source"), status="unavailable",
            reason="No governed weekly pipeline extract is available.")

    conv, cohort_pct = _conversion_inputs(data.get("funnel"))

    # (label, callable) — each isolated, so one failure costs one section.
    steps: List[Tuple[str, Callable[[], gen.Result]]] = [
        ("PIPELINE_MOVEMENT", lambda: gen.pipeline_movement(ctx, data.get("pipeline_detail"))),
        ("COMPLETIONS_MOVEMENT", lambda: gen.completions_movement(ctx, data.get("completions_detail"))),
        ("TICKET_SIZE", lambda: gen.ticket_size(ctx, data.get("ticket"))),
        ("WEIGHTED_LTV", lambda: gen.weighted_ltv(ctx, data.get("ltv"))),
        ("TICKET_MIX_SHIFT", lambda: gen.ticket_mix(ctx, data.get("ticket_mix"))),
        ("LTV_MIX_SHIFT", lambda: gen.ltv_mix(ctx, data.get("ltv_mix"))),
        ("CONVERSION_CONTEXT", lambda: gen.conversion(ctx, conv, cohort_pct)),
        ("CONCENTRATION_PROXIMITY", lambda: gen.concentration(ctx, data.get("concentration"))),
        ("DATA_QUALITY", lambda: gen.data_quality(ctx, data.get("data_quality"))),
    ]

    produced: List[Insight] = []
    omissions: List[Omission] = []
    failures = 0
    for label, step in steps:
        try:
            ins, omit = step()
            produced.extend(ins)
            omissions.extend(omit)
        except Exception as exc:  # noqa: BLE001 - one section must not cost the rest
            failures += 1
            logger.warning("weekly brief: %s generator failed: %s", label, exc)
            omissions.append(Omission(
                label, "This insight could not be produced for this week.",
                OMITTED_ERROR))

    kept, capped = select(produced, limits=conf.get("brief"))
    omissions.extend(capped)

    return build_brief(
        kept, omissions, tenant_id=tenant_id, portfolio_id=client_id,
        portfolio_context=ctx_scope, as_of_date=data.get("as_of_date"),
        comparison_date=data.get("comparison_date"), run_id=data.get("run_id"),
        config_source=conf.get("source"),
        status="partial" if failures else "success",
        reason=(f"{failures} insight type(s) could not be produced."
                if failures else None),
        source_dates={
            "pipeline_as_of": data.get("as_of_date"),
            "pipeline_comparison": data.get("comparison_date"),
            "funded_as_of": (data.get("concentration") or {}).get("reportingDate"),
            "forecast_observation_window_end":
                ((data.get("concentration") or {}).get("forecast") or {})
                .get("observationWindowEnd"),
        })
