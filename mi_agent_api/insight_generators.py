#!/usr/bin/env python3
"""Phase 3A — one generator per insight type.

Each generator takes already-resolved governed data and returns either an
``Insight`` or an ``Omission`` explaining why not. Nothing here reads storage,
nothing here defines a metric, and nothing here decides the final ordering —
that is the selector's job.

Two rules run through all of them:

* **Thresholds come from configuration, never from a literal in this file.**
  A number written here would be a portfolio assumption baked into code.

* **Narrative is template-driven and cautious.** "driven by", "largest
  contributors", "observed shift" — never "caused by", because the contributor
  methodology proves association within a decomposition, not causality.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import insight_config as cfg
from . import insight_metrics as metrics
from .insight_contract import (
    COMPLETIONS_MOVEMENT, CONCENTRATION_PROXIMITY, CONVERSION_CONTEXT,
    DATA_QUALITY, LTV_MIX_SHIFT, OMITTED_IMMATERIAL, OMITTED_UNAVAILABLE,
    PIPELINE_MOVEMENT, SEVERITY_ATTENTION, SEVERITY_CONCERN, SEVERITY_INFO,
    TICKET_MIX_SHIFT, TICKET_SIZE, WEIGHTED_LTV, Insight, Omission,
)

Result = Tuple[List[Insight], List[Omission]]


# --------------------------------------------------------------------------- #
# Formatting — one place, so every headline reads the same way
# --------------------------------------------------------------------------- #
def money(v: Optional[float]) -> str:
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1e9:
        return f"£{v / 1e9:.2f}bn"
    if a >= 1e6:
        return f"£{v / 1e6:.1f}m"
    if a >= 1e3:
        return f"£{v / 1e3:.0f}k"
    return f"£{v:,.0f}"


def signed_money(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return ("+" if v >= 0 else "−") + money(abs(v))


def pct(v: Optional[float], dp: int = 1) -> str:
    return "—" if v is None else f"{v:.{dp}f}%"


def signed_pct(v: Optional[float], dp: int = 1) -> str:
    return "—" if v is None else f"{'+' if v >= 0 else ''}{v:.{dp}f}%"


def direction(v: Optional[float]) -> str:
    if v is None:
        return "was unchanged"
    return "increased" if v > 0 else ("decreased" if v < 0 else "was unchanged")


def _names(rows: Optional[List[Dict[str, Any]]], n: int = 2) -> str:
    """The leading contributor names, for a summary sentence."""
    items = [str(r.get("name")) for r in (rows or [])[:n] if r.get("name")]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return " and ".join([", ".join(items[:-1]), items[-1]])


def _base(ctx: Dict[str, Any], **kw) -> Dict[str, Any]:
    """Identity every insight carries, from the resolved execution context."""
    out = {
        "tenant_id": ctx.get("tenant_id") or "-",
        "portfolio_id": ctx.get("portfolio_id") or "-",
        "portfolio_context": ctx.get("portfolio_context") or "total",
        "run_id": ctx.get("run_id"),
        "as_of_date": ctx.get("as_of_date"),
        "comparison_date": ctx.get("comparison_date"),
    }
    out.update(kw)
    return out


# --------------------------------------------------------------------------- #
# 1. Pipeline movement — reuses the Phase 2A attribution verbatim
# --------------------------------------------------------------------------- #
def pipeline_movement(ctx: Dict[str, Any], detail: Optional[Dict[str, Any]]) -> Result:
    if not detail or not detail.get("available"):
        return [], [Omission(PIPELINE_MOVEMENT,
                             (detail or {}).get("reason")
                             or "No comparable prior week is available.",
                             OMITTED_UNAVAILABLE)]
    t = cfg.thresholds("pipeline_movement")
    head = detail.get("headline_metric") or {}
    change, value = head.get("change"), head.get("value")
    change_pct = head.get("change_pct")

    # Two gates, either qualifies: a large relative move on a small book, or a
    # move that is large against the book even when the percentage looks modest.
    share_pct = (abs(change) / abs(value) * 100
                 if change is not None and value else None)
    by_pct = change_pct is not None and abs(change_pct) >= t.get("min_change_pct", 2.0)
    by_share = (share_pct is not None
                and share_pct >= t.get("min_change_share_of_pipeline_pct", 2.0))
    if not (by_pct or by_share):
        return [], [Omission(
            PIPELINE_MOVEMENT,
            f"Pipeline moved {signed_pct(change_pct)}, below the "
            f"{t.get('min_change_pct')}% materiality threshold.",
            OMITTED_IMMATERIAL)]

    brokers = (detail.get("contributors") or {}).get("brokers") or []
    regions = (detail.get("contributors") or {}).get("regions") or []
    counts = detail.get("counts") or {}
    lead = _names(brokers)
    summary = (f"Pipeline {direction(change)} by {money(abs(change or 0))} to "
               f"{money(value)}.")
    if lead:
        summary += f" Largest contributors: {lead}."
    if counts.get("change"):
        summary += (f" Case count {direction(counts['change'])} by "
                    f"{abs(counts['change'])}.")

    return [Insight(
        insight_type=PIPELINE_MOVEMENT,
        headline=f"Pipeline {direction(change)} {signed_pct(change_pct)}",
        summary=summary,
        severity=SEVERITY_INFO,
        metrics={
            "current_amount": value,
            "comparison_amount": (round(value - change, 2)
                                  if value is not None and change is not None else None),
            "change": change, "change_pct": change_pct,
            "current_case_count": counts.get("current"),
            "comparison_case_count": counts.get("comparison"),
            "case_count_change": counts.get("change"),
            "change_share_of_pipeline_pct": (round(share_pct, 2)
                                             if share_pct is not None else None),
        },
        contributors={"brokers": brokers[:3], "regions": regions[:3]},
        components=detail.get("components") or {},
        methodology={**(detail.get("methodology") or {}),
                     "reconciliation": "contributors sum to the headline change"},
        source_dates=detail.get("source_dates") or {},
        deep_link="/mi/pipeline",
        **_base(ctx),
    )], []


# --------------------------------------------------------------------------- #
# 2. Ticket size
# --------------------------------------------------------------------------- #
def ticket_size(ctx: Dict[str, Any], m: Optional[Dict[str, Any]]) -> Result:
    if not m:
        return [], [Omission(TICKET_SIZE, "No comparable prior week is available.",
                             OMITTED_UNAVAILABLE)]
    t = cfg.thresholds("ticket_size")
    n = m.get("current_case_count") or 0
    if n < t.get("min_case_count", 20):
        return [], [Omission(
            TICKET_SIZE,
            f"Only {n} cases in the pipeline; mean and median ticket are not "
            f"stable below {t.get('min_case_count')}.", OMITTED_UNAVAILABLE)]

    avg_c, med_c = m.get("average_change_pct"), m.get("median_change_pct")
    by_avg = avg_c is not None and abs(avg_c) >= t.get("min_average_change_pct", 5.0)
    by_med = med_c is not None and abs(med_c) >= t.get("min_median_change_pct", 5.0)
    if not (by_avg or by_med):
        return [], [Omission(
            TICKET_SIZE,
            f"Average ticket moved {signed_pct(avg_c)} and median "
            f"{signed_pct(med_c)}, both below threshold.", OMITTED_IMMATERIAL)]

    # Which one moved is the story. If only the mean moved, say so explicitly —
    # that is a tail effect, and conflating it with a broad shift would mislead.
    if by_avg and by_med:
        headline = f"Ticket size {direction(med_c)} {signed_pct(med_c)} (median)"
        summary = (f"Average ticket {direction(avg_c)} {signed_pct(avg_c)} to "
                   f"{money(m.get('current_average'))} and median "
                   f"{direction(med_c)} {signed_pct(med_c)} to "
                   f"{money(m.get('current_median'))}.")
    elif by_avg:
        headline = f"Average ticket size {direction(avg_c)} {signed_pct(avg_c)}"
        summary = (f"Average ticket {direction(avg_c)} to "
                   f"{money(m.get('current_average'))} while the median moved "
                   f"{signed_pct(med_c)} — the shift is concentrated in larger "
                   f"cases rather than across the book.")
    else:
        headline = f"Median ticket size {direction(med_c)} {signed_pct(med_c)}"
        summary = (f"Median ticket {direction(med_c)} to "
                   f"{money(m.get('current_median'))} while the average moved "
                   f"{signed_pct(avg_c)} — a broad shift rather than a tail effect.")

    return [Insight(
        insight_type=TICKET_SIZE, headline=headline, summary=summary,
        severity=SEVERITY_INFO,
        metrics={k: v for k, v in m.items() if k != "methodology"},
        methodology=m.get("methodology") or {},
        source_dates={"pipeline_as_of": ctx.get("as_of_date"),
                      "pipeline_comparison": ctx.get("comparison_date")},
        deep_link="/mi/pipeline",
        **_base(ctx),
    )], []


# --------------------------------------------------------------------------- #
# 3. Weighted-average LTV
# --------------------------------------------------------------------------- #
def weighted_ltv(ctx: Dict[str, Any], m: Optional[Dict[str, Any]]) -> Result:
    if not m:
        return [], [Omission(WEIGHTED_LTV, "No comparable prior week is available.",
                             OMITTED_UNAVAILABLE)]
    t = cfg.thresholds("weighted_ltv")
    cov = m.get("current_coverage_pct")
    floor = t.get("min_coverage_pct", 90.0)
    if cov is None or cov < floor:
        return [], [Omission(
            WEIGHTED_LTV,
            f"LTV covers {pct(cov)} of pipeline balance, below the {floor}% "
            f"required for a representative weighted average.",
            OMITTED_UNAVAILABLE)]

    change_pp = m.get("change_pp")
    if change_pp is None or abs(change_pp) < t.get("min_change_pp", 1.0):
        return [], [Omission(
            WEIGHTED_LTV,
            f"Weighted-average LTV moved {change_pp if change_pp is not None else '—'}"
            f"pp, below the {t.get('min_change_pp')}pp threshold.",
            OMITTED_IMMATERIAL)]

    worse = change_pp > 0
    cur_pct = (m.get("current") or 0) * 100
    pri_pct = (m.get("comparison") or 0) * 100
    return [Insight(
        insight_type=WEIGHTED_LTV,
        headline=(f"Weighted-average LTV {direction(change_pp)} by "
                  f"{abs(change_pp):.2f} percentage points"),
        summary=(f"Balance-weighted LTV moved from {pri_pct:.1f}% to "
                 f"{cur_pct:.1f}% — a {'deterioration' if worse else 'improvement'} "
                 f"of {abs(change_pp):.2f}pp, over {pct(cov)} of pipeline balance."),
        severity=SEVERITY_ATTENTION if worse else SEVERITY_INFO,
        metrics={k: v for k, v in m.items() if k != "methodology"},
        methodology=m.get("methodology") or {},
        data_quality={"coverage_pct": cov,
                      "excluded_balance": m.get("current_excluded_balance"),
                      "excluded_cases": m.get("current_excluded_cases")},
        source_dates={"pipeline_as_of": ctx.get("as_of_date"),
                      "pipeline_comparison": ctx.get("comparison_date")},
        deep_link="/mi/pipeline",
        **_base(ctx),
    )], []


# --------------------------------------------------------------------------- #
# 4. Mix shift — one per family, the most material only
# --------------------------------------------------------------------------- #
def mix_shift(ctx: Dict[str, Any], m: Optional[Dict[str, Any]], *,
              insight_type: str, config_key: str, label: str) -> Result:
    if not m or not m.get("largest_shift"):
        return [], [Omission(insight_type,
                             f"No {label} band data is available for both weeks.",
                             OMITTED_UNAVAILABLE)]
    t = cfg.thresholds(config_key)
    top = m["largest_shift"]
    move = top.get("share_change_pp")
    floor = t.get("min_share_change_pp", 5.0)
    if move is None or abs(move) < floor:
        return [], [Omission(
            insight_type,
            f"Largest {label} band shift was {move}pp, below the {floor}pp "
            f"threshold.", OMITTED_IMMATERIAL)]

    band = top.get("band")
    return [Insight(
        insight_type=insight_type,
        discriminator=str(band),
        headline=(f"{band} {direction(move)} from "
                  f"{top['comparison_share_pct']:.0f}% to "
                  f"{top['current_share_pct']:.0f}% of pipeline balance"),
        summary=(f"Observed shift in {label} mix: the {band} band moved "
                 f"{move:+.1f} percentage points of pipeline balance. "
                 f"Describes the observed mix only; no cause is inferred."),
        severity=SEVERITY_INFO,
        metrics={"band": band, "current_share_pct": top["current_share_pct"],
                 "comparison_share_pct": top["comparison_share_pct"],
                 "share_change_pp": move, "basis": m.get("basis"),
                 "band_column": m.get("band_column"),
                 "all_bands": m.get("bands", [])[:10]},
        methodology=m.get("methodology") or {},
        source_dates={"pipeline_as_of": ctx.get("as_of_date"),
                      "pipeline_comparison": ctx.get("comparison_date")},
        deep_link="/mi/pipeline",
        **_base(ctx),
    )], []


def ticket_mix(ctx, m) -> Result:
    return mix_shift(ctx, m, insight_type=TICKET_MIX_SHIFT,
                     config_key="ticket_mix", label="ticket-size")


def ltv_mix(ctx, m) -> Result:
    return mix_shift(ctx, m, insight_type=LTV_MIX_SHIFT,
                     config_key="ltv_mix", label="LTV")


# --------------------------------------------------------------------------- #
# 5. Completed-stage movement — never described as funded growth
# --------------------------------------------------------------------------- #
def completions_movement(ctx: Dict[str, Any],
                         detail: Optional[Dict[str, Any]]) -> Result:
    if not detail or not detail.get("available"):
        return [], [Omission(COMPLETIONS_MOVEMENT,
                             (detail or {}).get("reason")
                             or "No comparable prior week is available.",
                             OMITTED_UNAVAILABLE)]
    t = cfg.thresholds("completions")
    head = detail.get("headline_metric") or {}
    change, change_pct = head.get("change"), head.get("change_pct")
    if change_pct is None or abs(change_pct) < t.get("min_change_pct", 10.0):
        return [], [Omission(
            COMPLETIONS_MOVEMENT,
            f"Completed-stage value moved {signed_pct(change_pct)}, below the "
            f"{t.get('min_change_pct')}% threshold.", OMITTED_IMMATERIAL)]

    brokers = (detail.get("contributors") or {}).get("brokers") or []
    regions = (detail.get("contributors") or {}).get("regions") or []
    counts = detail.get("counts") or {}
    lead = _names(brokers)
    summary = (f"Completed-stage value {direction(change)} by "
               f"{money(abs(change or 0))} to {money(head.get('value'))}. "
               f"This is pipeline completion movement — cases reaching completed "
               f"stage — not funded balance growth.")
    if lead:
        summary += f" Largest contributors: {lead}."

    return [Insight(
        insight_type=COMPLETIONS_MOVEMENT,
        headline=(f"Cases reaching completed stage {direction(change)} "
                  f"{signed_pct(change_pct)}"),
        summary=summary, severity=SEVERITY_INFO,
        metrics={
            "current_completed_value": head.get("value"),
            "change": change, "change_pct": change_pct,
            "current_completed_case_count": counts.get("current"),
            "case_count_change": counts.get("change"),
        },
        contributors={"brokers": brokers[:3], "regions": regions[:3]},
        components=detail.get("components") or {},
        methodology=detail.get("methodology") or {},
        source_dates=detail.get("source_dates") or {},
        deep_link="/mi/pipeline",
        **_base(ctx),
    )], []


# --------------------------------------------------------------------------- #
# 6. Conversion — aggregate only, no attribution, no invented comparison
# --------------------------------------------------------------------------- #
def conversion(ctx: Dict[str, Any], conv: Optional[Dict[str, Any]],
               cohort_pct: Optional[float] = None) -> Result:
    if not cfg.thresholds("conversion").get("enabled", True):
        return [], [Omission(CONVERSION_CONTEXT,
                             "Conversion reporting is disabled by configuration "
                             "for this deployment.", OMITTED_UNAVAILABLE)]
    if not conv:
        return [], [Omission(CONVERSION_CONTEXT,
                             "No governed conversion evidence is available.",
                             OMITTED_UNAVAILABLE)]
    t = cfg.thresholds("conversion")
    rate = conv.get("weeklyRateValue")
    sufficient = bool(conv.get("sufficient"))
    floor = t.get("minimum_acceptable_rate")

    below = floor is not None and rate is not None and rate < float(floor)
    insufficient = t.get("require_sufficient", True) and not sufficient
    if not (below or insufficient):
        return [], [Omission(
            CONVERSION_CONTEXT,
            "Conversion is within its governed window and above any configured "
            "floor; no comparison rate is published, so no movement is reported.",
            OMITTED_IMMATERIAL)]

    if insufficient:
        headline = "Conversion evidence is provisional"
        summary = (f"Only {conv.get('weeksInWindow')} of "
                   f"{conv.get('minWeeks')}+ required weeks are observed, so the "
                   f"conversion rate should not be forecast from yet.")
        severity = SEVERITY_ATTENTION
    else:
        lag_weeks = conv.get("lagWeeks")
        lag_note = (f" lagged {lag_weeks} weeks"
                    if conv.get("lagApplied") and lag_weeks else "")
        headline = f"Weekly conversion is {pct(rate)}, below the configured floor"
        summary = (f"Weekly velocity is {pct(rate)} by value against a floor of "
                   f"{pct(float(floor))}. Numerator is the 5-week average flow; "
                   f"denominator is the KFI stock{lag_note}.")
        severity = SEVERITY_ATTENTION

    return [Insight(
        insight_type=CONVERSION_CONTEXT, headline=headline, summary=summary,
        severity=severity,
        metrics={
            "definition": conv.get("basis"),
            "weekly_rate_value_pct": rate,
            "weekly_rate_count_pct": conv.get("weeklyRateCount"),
            "cumulative_cohort_pct": cohort_pct,
            "numerator_value": conv.get("avgWeeklyFlowValue"),
            "numerator_count": conv.get("avgWeeklyFlowCount"),
            "denominator_value": conv.get("kfiStockValue"),
            "denominator_count": conv.get("kfiStockCount"),
            "lag_weeks": conv.get("lagWeeks"),
            "lag_applied": conv.get("lagApplied"),
            "denominator_week": conv.get("denominatorWeek"),
            "weeks_in_window": conv.get("weeksInWindow"),
            "minimum_weeks": conv.get("minWeeks"),
            "sufficient": sufficient,
        },
        methodology={
            "basis": conv.get("basis"),
            "no_comparison_rate": ("No prior-period conversion rate is published, "
                                   "so no percentage-point movement is reported."),
            "no_attribution": ("Broker and regional attribution is not valid for "
                               "conversion: the numerator is a trailing average "
                               "flow and the denominator a lagged KFI stock."),
            "version": metrics.METHODOLOGY_VERSION,
        },
        source_dates={"pipeline_as_of": ctx.get("as_of_date"),
                      "denominator_week": conv.get("denominatorWeek")},
        deep_link="/mi/pipeline",
        **_base(ctx),
    )], []


# --------------------------------------------------------------------------- #
# 7. Concentration — reuses the governed test output, never recomputes it
# --------------------------------------------------------------------------- #
def concentration(ctx: Dict[str, Any], snapshot: Optional[Dict[str, Any]]) -> Result:
    if not snapshot or not snapshot.get("available"):
        return [], [Omission(CONCENTRATION_PROXIMITY,
                             (snapshot or {}).get("reason")
                             or "No approved concentration tests are available.",
                             OMITTED_UNAVAILABLE)]
    t = cfg.thresholds("concentration")
    amber = t.get("amber_utilisation_pct", 90.0) / 100.0
    red = t.get("red_utilisation_pct", 100.0) / 100.0
    states_available = bool((snapshot.get("states") or {}).get("available"))
    funded_date = snapshot.get("reportingDate")
    forecast = snapshot.get("forecast") or {}

    candidates: List[Tuple[float, Insight]] = []
    for test in snapshot.get("tests") or []:
        util = test.get("utilization")
        expected = test.get("expected") or {}
        exp_util = expected.get("utilization") if states_available else None
        horizon = test.get("expectedBreachHorizon") or {}
        worst = max([u for u in (util, exp_util) if u is not None], default=None)
        if worst is None or worst < amber:
            continue

        breached = util is not None and util >= red
        exp_breach = bool(test.get("expectedBreach"))
        severity = (SEVERITY_CONCERN if (breached or exp_breach)
                    else SEVERITY_ATTENTION)

        name = test.get("displayName") or test.get("testId") or "Concentration test"
        headline = (f"{name} is at {pct((util or 0) * 100, 0)} of its limit"
                    if util is not None else f"{name} utilisation is unavailable")
        summary = (f"Current funded utilisation {pct((util or 0) * 100)} "
                   f"(as of {funded_date or 'unknown date'}), headroom "
                   f"{pct((test.get('headroom') or 0) * 100, 2)}pp, status "
                   f"{test.get('status')}.")
        if states_available and exp_util is not None:
            summary += (f" Expected forecast utilisation {pct(exp_util * 100)}"
                        f"{', forecast to breach' if exp_breach else ''}")
            if horizon.get("available") and horizon.get("period"):
                summary += f" around {horizon['period']}"
            summary += "."
        full = test.get("fullPipeline") or {}
        if states_available and full.get("utilization") is not None:
            summary += (f" If all active pipeline converted — a hypothetical "
                        f"stress, not a forecast — utilisation would be "
                        f"{pct(full['utilization'] * 100)}.")

        candidates.append((worst, Insight(
            insight_type=CONCENTRATION_PROXIMITY,
            discriminator=str(test.get("testId") or name),
            headline=headline, summary=summary, severity=severity,
            metrics={
                "test_id": test.get("testId"), "test_name": name,
                "current_value": test.get("currentValue"),
                "limit": test.get("threshold"),
                "utilisation": util, "headroom": test.get("headroom"),
                "status": test.get("status"),
                "forecast_value": expected.get("value") if states_available else None,
                "forecast_utilisation": exp_util,
                "forecast_headroom": expected.get("headroom") if states_available else None,
                "forecast_status": expected.get("status") if states_available else None,
                "expected_breach": exp_breach,
                "expected_breach_horizon": horizon.get("period"),
                "all_pipeline_value": full.get("value") if states_available else None,
                "all_pipeline_utilisation": full.get("utilization") if states_available else None,
                "all_pipeline_status": full.get("status") if states_available else None,
            },
            methodology={
                "source": "governed concentration tests — not recomputed here",
                "forecast_methodology": forecast.get("methodology"),
                "forecast_observation_window_end": forecast.get("observationWindowEnd"),
                "all_pipeline_basis": ("hypothetical stress: 100% of active "
                                       "pipeline completes. Not a forecast."),
                "version": metrics.METHODOLOGY_VERSION,
            },
            source_dates={
                "funded_as_of": funded_date,
                "pipeline_as_of": forecast.get("currentSnapshot"),
                "forecast_observation_window_end": forecast.get("observationWindowEnd"),
            },
            deep_link="/mi/risk-limits",
            **_base(ctx),
        )))

    if not candidates:
        return [], [Omission(
            CONCENTRATION_PROXIMITY,
            f"No concentration test is at or above {t.get('amber_utilisation_pct')}% "
            f"of its limit on either the funded or forecast state.",
            OMITTED_IMMATERIAL)]

    # Most proximate first; the selector applies the per-type cap.
    candidates.sort(key=lambda c: (-c[0], c[1].discriminator))
    return [i for _u, i in candidates], []


# --------------------------------------------------------------------------- #
# 8. Data quality — governs whether the rest of the brief can be believed
# --------------------------------------------------------------------------- #
def data_quality(ctx: Dict[str, Any], dq: Optional[Dict[str, Any]]) -> Result:
    if not dq:
        return [], [Omission(DATA_QUALITY, "No comparison frames are available.",
                             OMITTED_UNAVAILABLE)]
    t = cfg.thresholds("data_quality")
    breaches: List[str] = []

    def over(value: Optional[float], limit: Optional[float]) -> bool:
        return value is not None and limit is not None and value > float(limit)

    reassign_limit = t.get("max_dimension_reassignment_pct")
    if over(dq.get("broker_reassignment_pct"), reassign_limit):
        breaches.append(
            f"{dq['broker_reassignment_pct']:.1f}% of cases changed broker "
            f"between the two weeks")
    if over(dq.get("region_reassignment_pct"), reassign_limit):
        breaches.append(
            f"{dq['region_reassignment_pct']:.1f}% of cases changed region")
    unknown_limit = t.get("max_unknown_share_pct")
    if over(dq.get("unknown_broker_share_pct"), unknown_limit):
        breaches.append(
            f"{dq['unknown_broker_share_pct']:.1f}% of balance has no broker")
    if over(dq.get("unknown_region_share_pct"), unknown_limit):
        breaches.append(
            f"{dq['unknown_region_share_pct']:.1f}% of balance has no region")
    if over(dq.get("unusable_case_id_pct"), t.get("max_unusable_id_share_pct")):
        breaches.append(
            f"{dq['unusable_case_id_count']} case(s) carry no usable identifier "
            f"({money(dq.get('unusable_case_id_balance'))})")
    if over(dq.get("duplicate_case_id_pct"), t.get("max_duplicate_id_share_pct")):
        breaches.append(f"{dq['duplicate_case_id_count']} duplicate case identifier(s)")

    if not breaches:
        if not t.get("emit_when_clean", False):
            return [], [Omission(DATA_QUALITY,
                                 "No data-quality threshold was breached.",
                                 OMITTED_IMMATERIAL)]
        return [Insight(
            insight_type=DATA_QUALITY,
            headline="No material data-quality issues this week",
            summary=("Dimension stability, identifier quality and LTV coverage "
                     "are all within their configured thresholds."),
            severity=SEVERITY_INFO,
            metrics={k: v for k, v in dq.items() if k != "methodology"},
            methodology=dq.get("methodology") or {},
            source_dates={"pipeline_as_of": ctx.get("as_of_date"),
                          "pipeline_comparison": ctx.get("comparison_date")},
            deep_link="/mi/pipeline", **_base(ctx),
        )], []

    # A high reassignment rate is the serious case: it means the contributor
    # attribution elsewhere in this brief cannot be relied on.
    attribution_risk = (over(dq.get("broker_reassignment_pct"), reassign_limit)
                        or over(dq.get("region_reassignment_pct"), reassign_limit))
    summary = "; ".join(breaches) + "."
    if attribution_risk:
        summary += (" Contributor attribution in this brief should be treated "
                    "with caution while dimensions are unstable.")

    return [Insight(
        insight_type=DATA_QUALITY,
        headline=(f"{len(breaches)} data-quality threshold"
                  f"{'s' if len(breaches) > 1 else ''} breached"),
        summary=summary,
        severity=SEVERITY_CONCERN if attribution_risk else SEVERITY_ATTENTION,
        metrics={k: v for k, v in dq.items() if k != "methodology"},
        methodology=dq.get("methodology") or {},
        data_quality={"breaches": breaches, "attribution_risk": attribution_risk},
        source_dates={"pipeline_as_of": ctx.get("as_of_date"),
                      "pipeline_comparison": ctx.get("comparison_date")},
        deep_link="/mi/pipeline",
        **_base(ctx),
    )], []
