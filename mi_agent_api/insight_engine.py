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


def _select_with(insights: List[Insight], lim: Dict[str, Any]
                 ) -> Tuple[List[Insight], List[Omission]]:
    """Order, cap per type, cut to the brief limit.

    Anything dropped by a cap becomes an explicit omission — a brief that
    silently truncated would read as "this is everything", which it would not
    be. ``priority`` is stamped here rather than by the generators, so ordering
    can be tuned without changing what an insight claims about itself.
    """
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


def select(insights: List[Insight], *, limits: Optional[Dict[str, Any]] = None
           ) -> Tuple[List[Insight], List[Omission]]:
    """Weekly selection. Unchanged: the weekly brief's caps, by default."""
    return _select_with(insights, limits or cfg.brief_limits())


def select_funded(insights: List[Insight], *,
                  limits: Optional[Dict[str, Any]] = None
                  ) -> Tuple[List[Insight], List[Omission]]:
    """Monthly selection. Same ordering rule, the monthly brief's caps.

    One selector, two limit sets — rather than two selectors — because the
    ORDER is the thing that must not diverge: severity, then type priority, then
    a deterministic tiebreak. Only how many survive it differs.
    """
    return _select_with(insights, limits or cfg.funded_brief_limits())


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


# --------------------------------------------------------------------------- #
# The monthly funded brief
#
# A deliberate sibling of ``build`` rather than a mode of it. The two resolve
# different sources over different periods (weekly extracts against monthly
# runs), and folding them into one function with a flag is how a funded figure
# eventually acquires a pipeline date. What they DO share is the contract, the
# ordering rule, the omission discipline and the failure isolation — all reached
# through the same helpers below.
# --------------------------------------------------------------------------- #
#: Governed funded dimensions the mix generator asks about, with the label a
#: reader sees. Every one is a prepared column ``funded_prep`` already
#: materialises (``CORE_FUNDED_DIMENSIONS``) — nothing here derives a dimension.
FUNDED_MIX_DIMENSIONS: List[Tuple[str, str]] = [
    ("product", "Product"),
    ("geographic_region_obligor", "Region"),
    ("ltv_bucket", "LTV band"),
    ("age_bucket", "Borrower age band"),
    ("borrower_type", "Borrower structure"),
    ("vintage_year", "Origination vintage"),
    ("source_portfolio_id", "Source portfolio"),
]

#: Candidate columns per logical dimension, first present wins. Mirrors the
#: ``group`` kind in ``funded_prep._DIM_SPEC``: a tape may carry region as any of
#: three columns and product under either of two, and a mix answer must not
#: depend on which one arrived.
_MIX_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "product": ("erm_product_type", "product_type", "product"),
    "geographic_region_obligor": ("geographic_region_obligor",
                                  "geographic_region_collateral",
                                  "collateral_geography"),
}


def _mix_shifts(current, prior, *, source_dates: Dict[str, Any],
                population: str = "combined") -> List[Dict[str, Any]]:
    """Share-of-balance movement per governed dimension, largest band each.

    Grouping is ``evolution._group_balance`` — the same function the funded
    bridge groups with, including its ``Unknown / Missing`` handling — so a mix
    share and a bridge contribution can never be computed two ways.

    Only the single largest band move per dimension is returned. A list of every
    band that twitched is noise: the generator's job is to say what changed, and
    on a seven-band LTV split six of the moves are the mechanical complement of
    the seventh.
    """
    from . import evolution as evolution_mod

    out: List[Dict[str, Any]] = []
    if current is None or prior is None:
        return out

    for dimension, label in FUNDED_MIX_DIMENSIONS:
        candidates = _MIX_COLUMNS.get(dimension, (dimension,))
        col = next((c for c in candidates
                    if c in getattr(current, "columns", ())
                    and c in getattr(prior, "columns", ())), None)
        if col is None:
            continue

        cur_groups = evolution_mod._group_balance(current, col)
        pri_groups = evolution_mod._group_balance(prior, col)
        cur_total, pri_total = sum(cur_groups.values()), sum(pri_groups.values())
        if not cur_total or not pri_total:
            continue

        best: Optional[Dict[str, Any]] = None
        # Sorted, and the comparison is strict, so a tie resolves to the first
        # category by name rather than to whatever order a set happened to
        # iterate in. On a two-band dimension the two moves are exact
        # complements and equally true, so which one is reported must at least
        # be the SAME one on every run — the selector below is deterministic and
        # would be undermined by a non-deterministic input. Same tie rule as
        # ``movement_detail.rank_contributors``: magnitude, then name ascending.
        for category in sorted(set(cur_groups) | set(pri_groups), key=str):
            cur_bal = cur_groups.get(category, 0.0)
            pri_bal = pri_groups.get(category, 0.0)
            change_pp = (cur_bal / cur_total - pri_bal / pri_total) * 100.0
            if best is None or abs(change_pp) > abs(best["share_change_pp"]):
                best = {
                    "dimension": dimension, "dimension_label": label,
                    "column": col, "category": str(category),
                    "current_balance": round(cur_bal, 2),
                    "prior_balance": round(pri_bal, 2),
                    "current_share_pct": round(cur_bal / cur_total * 100.0, 2),
                    "prior_share_pct": round(pri_bal / pri_total * 100.0, 2),
                    "share_change_pp": round(change_pp, 2),
                    "population": population,
                    "source_dates": dict(source_dates),
                }
        if best is not None:
            out.append(best)
    return out


@_perf.stage_fn("funded_brief_resolve")
def resolve_funded_inputs(output_root, client_id: str, *,
                          to_run_id: Optional[str] = None,
                          lens_filters: Optional[Dict[str, Any]] = None,
                          concentration: Optional[Dict[str, Any]] = None,
                          scope: Optional[Any] = None) -> Dict[str, Any]:
    """Gather every governed input the funded generators need, once.

    The two period frames are resolved ONCE, through
    ``evolution.funded_frames``, and shared by the decomposition, the underlying
    lens and the mix shifts. ``period_movement`` and the concentration snapshot
    come from their existing services untouched.
    """
    from . import evolution as evolution_mod
    from . import funded_composition as comp
    from . import movement_summary as movement_mod

    out: Dict[str, Any] = {"concentration": concentration}

    out["movement"] = _safe("funded period movement", lambda: movement_mod.period_movement(
        output_root, client_id, to_run_id=to_run_id, lens_filters=lens_filters))

    frames = _safe("funded frames", lambda: evolution_mod.funded_frames(
        output_root, client_id, to_run_id, scope=scope), default=[]) or []
    scoped = []
    for f in frames:
        d = evolution_mod._scope_frame_lens(f.get("df"), lens_filters)
        if d is not None and len(d):
            scoped.append({**f, "df": d})

    if len(scoped) < 2:
        out["decomposition"] = {
            "available": False,
            "reason": ("at least two governed funded reporting periods are "
                       "needed to decompose a movement")}
        out["frames"] = (None, None)
        return out

    cur, pri = scoped[-1], scoped[-2]
    out["frames"] = (cur["df"], pri["df"])
    out["as_of_date"] = cur.get("reporting_date")
    out["comparison_date"] = pri.get("reporting_date")
    out["run_id"] = cur.get("run_id")
    source_dates = {"funded_as_of": out["as_of_date"],
                    "funded_comparison": out["comparison_date"]}

    decomposition = _safe("funded composition",
                          lambda: comp.decompose(cur["df"], pri["df"]))
    if decomposition:
        decomposition.update({
            "currentReportingDate": out["as_of_date"],
            "priorReportingDate": out["comparison_date"],
        })
    out["decomposition"] = decomposition or {
        "available": False, "reason": "the funded movement could not be decomposed"}

    # The underlying book, resolved ONLY when something was added — the lens is
    # the existing one, over the continuing portfolio ids.
    out["underlying"] = None
    underlying_filters = comp.underlying_lens_filters(out["decomposition"] or {})
    out["underlying_filters"] = underlying_filters
    if underlying_filters:
        underlying = _safe("underlying book", lambda: comp.decompose(
            evolution_mod._scope_frame_lens(cur["df"], underlying_filters),
            evolution_mod._scope_frame_lens(pri["df"], underlying_filters)))
        if underlying:
            underlying.update({
                "currentReportingDate": out["as_of_date"],
                "priorReportingDate": out["comparison_date"],
            })
        out["underlying"] = underlying

    # CHARACTERISTIC movement is read on the UNDERLYING book whenever a
    # portfolio was added, and labelled as such.
    #
    # Mix and weighted LTV describe what the book IS, and an arriving book
    # rewrites both by construction. Measured on the combined population, an
    # incumbent book whose LTV rose 30% to 38% alongside a low-LTV acquisition
    # published "LTV moved from 30.0% to 29.0%" — an improvement, on a book that
    # deteriorated by eight points. That is the masking this review exists to
    # prevent, and it is worse than an omission because it is confidently wrong.
    #
    # The combined position is not lost: the balance decomposition above states
    # the addition, and concentration is evaluated on the whole book by its own
    # governed tests. What changes here is only which population the
    # characteristic MOVEMENT describes — the one that was there for both
    # periods, which is the only population a movement can honestly be measured
    # over.
    mix_current, mix_prior = cur["df"], pri["df"]
    out["characteristic_population"] = "combined"
    if underlying_filters:
        mix_current = evolution_mod._scope_frame_lens(cur["df"], underlying_filters)
        mix_prior = evolution_mod._scope_frame_lens(pri["df"], underlying_filters)
        out["characteristic_population"] = "underlying"

    out["mix_shifts"] = _safe("funded mix", lambda: _mix_shifts(
        mix_current, mix_prior, source_dates=source_dates,
        population=out["characteristic_population"]), default=[])
    out["underlying_ltv"] = _safe("underlying ltv", lambda: _weighted_ltv_points(
        mix_current, mix_prior)) if underlying_filters else None
    return out


_LTV_COLUMN = "current_loan_to_value"


def _weighted_ltv_points(current, prior) -> Optional[Dict[str, Any]]:
    """Balance-weighted LTV, in points, for one population across two periods.

    Same weighting and the same scale rule the funded series uses — a stored
    ratio is converted, a stored percentage is not — so an underlying LTV and a
    combined one are the same measure over different populations rather than two
    definitions.
    """
    from analytics_lib.numeric import coerce_numeric

    def _points(df):
        if df is None or not len(df) or _LTV_COLUMN not in df.columns:
            return None
        value = coerce_numeric(df[_LTV_COLUMN])
        weight = coerce_numeric(df[_BALANCE_COLUMN])
        usable = value.notna() & weight.notna()
        if not usable.any() or float(weight[usable].sum()) <= 0:
            return None
        wavg = float((value[usable] * weight[usable]).sum()
                     / weight[usable].sum())
        return round(wavg * 100.0 if abs(wavg) <= 1.5 else wavg, 2)

    cur_pts, pri_pts = _points(current), _points(prior)
    if cur_pts is None or pri_pts is None:
        return None
    return {"current": cur_pts, "prior": pri_pts,
            "change_pp": round(cur_pts - pri_pts, 2)}


_BALANCE_COLUMN = "current_outstanding_balance"


@_perf.stage_fn("funded_brief_build")
def build_funded(output_root, client_id: str, *, tenant_id: str,
                 to_run_id: Optional[str] = None,
                 lens_filters: Optional[Dict[str, Any]] = None,
                 scope: Optional[str] = None,
                 concentration_snapshot: Optional[Dict[str, Any]] = None,
                 resolved: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The Monthly Funded Brief for one portfolio scope and one reporting period.

    ``resolved`` lets a caller that has already gathered the inputs pass them in
    rather than have them resolved twice — the same courtesy ``build`` extends
    through ``concentration_snapshot``.
    """
    from . import insight_generators_funded as fgen

    conf = cfg.load()
    ctx_scope = scope or "total"
    data = resolved or resolve_funded_inputs(
        output_root, client_id, to_run_id=to_run_id, lens_filters=lens_filters,
        concentration=concentration_snapshot)

    ctx = {
        "tenant_id": tenant_id, "portfolio_id": client_id,
        "portfolio_context": ctx_scope, "run_id": data.get("run_id"),
        "as_of_date": data.get("as_of_date"),
        "comparison_date": data.get("comparison_date"),
    }

    movement = data.get("movement")
    if not (movement or {}).get("available") and \
            not (data.get("decomposition") or {}).get("available"):
        return build_brief(
            [], [], tenant_id=tenant_id, portfolio_id=client_id,
            portfolio_context=ctx_scope, as_of_date=data.get("as_of_date"),
            comparison_date=data.get("comparison_date"),
            config_source=conf.get("source"), status="unavailable",
            reason=((movement or {}).get("reason")
                    or "No comparable governed funded reporting period is available."))

    steps: List[Tuple[str, Callable[[], Any]]] = [
        ("FUNDED_MOVEMENT", lambda: fgen.funded_movement(ctx, movement)),
        ("FUNDED_COMPOSITION",
         lambda: fgen.funded_composition(ctx, data.get("decomposition"))),
        ("UNDERLYING_BOOK_MOVEMENT",
         lambda: fgen.underlying_book(ctx, data.get("decomposition"),
                                      data.get("underlying"))),
        ("FUNDED_MIX_SHIFT", lambda: fgen.mix_shift(ctx, data.get("mix_shifts"))),
        ("FUNDED_LTV_MOVEMENT",
         lambda: fgen.ltv_movement(ctx, movement, data.get("underlying_ltv"))),
        ("RISK_LIMIT_TRANSITION",
         lambda: fgen.risk_limit_transitions(ctx, data.get("concentration"))),
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
            logger.warning("funded brief: %s generator failed: %s", label, exc)
            omissions.append(Omission(
                label, "This insight could not be produced for this period.",
                OMITTED_ERROR))

    kept, capped = select_funded(produced, limits=conf.get("funded_brief"))
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
            "funded_as_of": data.get("as_of_date"),
            "funded_comparison": data.get("comparison_date"),
        })
