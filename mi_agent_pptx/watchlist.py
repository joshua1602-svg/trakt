"""mi_agent_pptx.watchlist — what requires attention before the next period.

A watch list is only useful if it is short and if everything on it is real. Two
failure modes are worse than having no list at all: turning ordinary movement
into a warning, and burying a genuine concern among twenty observations.

So this module is deliberately conservative:

* every item is derived from a governed payload — nothing is inferred;
* every item carries its evidence, its source date and the portfolio context it
  applies to, so a reader can check it;
* immaterial movements are suppressed rather than listed;
* severity comes from governed thresholds and status vocabularies, never from
  wording;
* **no management actions are proposed.** Recommending what to do about a
  concentration breach is a decision, not an observation, and nothing in the
  governed evidence authorises the deck to make it.

Items reuse the shared :class:`~mi_agent_api.insight_contract.Insight` contract
so a watch item and an executive observation are the same kind of object, carry
the same provenance, and can be ranked together.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mi_agent_api.insight_contract import (
    SEVERITY_ATTENTION, SEVERITY_CONCERN, SEVERITY_INFO, SEVERITY_RANK, Insight,
)

#: Watch item types, in investor priority order.
CONCENTRATION_BREACH = "CONCENTRATION_BREACH"
CONCENTRATION_FORECAST = "CONCENTRATION_FORECAST_BREACH"
CONCENTRATION_STRESS = "CONCENTRATION_STRESS_BREACH"
CONCENTRATION_PROXIMITY = "CONCENTRATION_PROXIMITY"
DATA_COVERAGE = "DATA_COVERAGE"
REPORTING_DATE_SPREAD = "REPORTING_DATE_SPREAD"
ACQUIRED_RUNOFF = "ACQUIRED_RUNOFF"
LTV_MOVEMENT = "WEIGHTED_LTV_MOVEMENT"
CONCENTRATION_SHIFT = "CONCENTRATION_SHIFT"
PIPELINE_MOVEMENT = "PIPELINE_MOVEMENT"

WATCH_PRIORITY: Dict[str, int] = {
    CONCENTRATION_BREACH: 100,
    CONCENTRATION_FORECAST: 95,
    CONCENTRATION_STRESS: 85,
    CONCENTRATION_PROXIMITY: 80,
    REPORTING_DATE_SPREAD: 70,
    DATA_COVERAGE: 65,
    LTV_MOVEMENT: 55,
    CONCENTRATION_SHIFT: 50,
    ACQUIRED_RUNOFF: 45,
    PIPELINE_MOVEMENT: 40,
}

#: Caps. A watch list longer than this stops being a watch list.
MAX_WATCH_ITEMS = 5
MAX_POSITIVES = 3

#: A limit within this many points of its threshold is worth watching even
#: though it currently passes.
PROXIMITY_UTILISATION_PCT = 85.0
#: Share movement below this is composition noise, not a shift.
MATERIAL_SHARE_PP = 2.0
#: Weighted-LTV movement below this is not worth an investor's attention.
MATERIAL_LTV_PP = 0.5


def _item(ctx: Mapping[str, Any], item_type: str, headline: str, summary: str, *,
          severity: str = SEVERITY_ATTENTION, metrics: Optional[Dict[str, Any]] = None,
          discriminator: str = "", scope: Optional[str] = None,
          source_date: Optional[str] = None,
          methodology: Optional[Dict[str, Any]] = None) -> Insight:
    return Insight(
        insight_type=item_type,
        tenant_id=ctx.get("tenant_id") or "-",
        portfolio_id=ctx.get("portfolio_id") or "-",
        portfolio_context=scope or ctx.get("portfolio_context") or "total",
        as_of_date=source_date or ctx.get("as_of_date"),
        comparison_date=ctx.get("comparison_date"),
        headline=headline, summary=summary, severity=severity,
        discriminator=discriminator, metrics=metrics or {},
        methodology=methodology or {},
        source_dates={"as_of": source_date or ctx.get("as_of_date")})


# --------------------------------------------------------------------------- #
# Generators.
# --------------------------------------------------------------------------- #

def _concentration_items(ctx, concentration) -> List[Insight]:
    from . import concentration as C

    rows = C.adapt_tests(concentration)
    if not rows:
        return []
    forward = C.forward_states_available(concentration)
    out: List[Insight] = []
    for r in rows:
        label, unit = r["label"], r["unit"]
        current = C.format_measure(r["value"], unit)
        limit = C.format_measure(r["limit"], unit)
        date = r.get("reporting_date")
        if r["status"] == C.STATUS_BREACH:
            out.append(_item(
                ctx, CONCENTRATION_BREACH,
                f"{label} is in breach at {current} against a {limit} limit.",
                f"The test is measured at {current} against its contractual "
                f"limit of {limit} at the funded reporting date.",
                severity=SEVERITY_CONCERN, discriminator=str(r["test_id"]),
                metrics={"value": r["value"], "limit": r["limit"],
                         "utilisation": r["utilisation"], "headroom": r["headroom"]},
                source_date=date,
                methodology={"source": "governed concentration evaluation"}))
        elif forward and r.get("expected_breach"):
            horizon = r.get("breach_horizon")
            out.append(_item(
                ctx, CONCENTRATION_FORECAST,
                f"{label} is within limit but is forecast to breach"
                + (f" around {horizon}." if horizon else "."),
                f"Current {current} against a {limit} limit; the expected "
                f"forecast state exceeds the limit"
                + (f", crossing around {horizon}" if horizon else "") + ".",
                severity=SEVERITY_CONCERN, discriminator=str(r["test_id"]),
                metrics={"expected_value": r["expected_value"],
                         "expected_utilisation": r["expected_utilisation"],
                         "breach_horizon": horizon},
                source_date=date,
                methodology={"source": "governed expected-forecast state"}))
        elif forward and r.get("stress_breach"):
            out.append(_item(
                ctx, CONCENTRATION_STRESS,
                f"{label} would breach under the all-pipeline-converts stress.",
                f"Current {current} against a {limit} limit remains within "
                f"limit, and the expected forecast does not breach. The "
                f"all-pipeline-converts case would. This is a stress, not the "
                f"expected outcome.",
                severity=SEVERITY_ATTENTION, discriminator=str(r["test_id"]),
                metrics={"stress_value": r["stress_value"],
                         "stress_utilisation": r["stress_utilisation"]},
                source_date=date,
                methodology={"source": "governed full-pipeline stress state"}))
        elif (r["utilisation"] or 0) >= PROXIMITY_UTILISATION_PCT:
            out.append(_item(
                ctx, CONCENTRATION_PROXIMITY,
                f"{label} is at {r['utilisation']:.0f}% of its limit.",
                f"Measured at {current} against a {limit} limit — within "
                f"limit but the closest test to its threshold.",
                severity=SEVERITY_ATTENTION, discriminator=str(r["test_id"]),
                metrics={"utilisation": r["utilisation"], "headroom": r["headroom"]},
                source_date=date,
                methodology={"source": "governed concentration evaluation"}))
    return out


def _portfolio_items(ctx, portfolio) -> List[Insight]:
    if portfolio is None:
        return []
    out: List[Insight] = []
    # NOTE: mixed reporting dates are deliberately NOT raised here. The cover
    # carries the warning and the executive summary raises it as an attention
    # observation; repeating it a third time would crowd the watch list with a
    # standing disclosure rather than something to act on before next period.
    for sl in portfolio.type_slices:
        change = sl.balance_movement
        if change is None or change >= 0 or not sl.balance:
            continue
        pct = change / abs(sl.balance - change) * 100.0 if (sl.balance - change) else None
        if pct is not None and abs(pct) < 1.0:
            continue
        out.append(_item(
            ctx, ACQUIRED_RUNOFF,
            f"The {sl.label.lower()} reduced over the period.",
            f"Balance fell to {sl.balance:,.0f}"
            + (f" ({pct:+.1f}%)" if pct is not None else "")
            + ". The reduction is observed in the balance movement; no "
              "forecast of continued run-off is implied.",
            severity=SEVERITY_INFO, discriminator=sl.portfolio_type,
            scope=sl.portfolio_type,
            metrics={"balance": sl.balance, "movement": change},
            source_date=sl.reporting_date,
            methodology={"source": "governed funded snapshot per portfolio type"}))
    return out


def _movement_items(ctx, movement) -> List[Insight]:
    """A material shift in the shape of the book, not merely its size."""
    out: List[Insight] = []
    for key, bridge in (movement or {}).items():
        if not getattr(bridge, "available", False):
            continue
        # ONE shift per dimension. Within a dimension the shares are
        # complementary — if one category gains 8pp the others lose 8pp between
        # them — so listing every mover states the same fact several times and
        # crowds out genuinely different findings.
        shifts = []
        for contributor in bridge.contributors:
            if contributor.is_other:
                continue
            pp = bridge.share_change_pp(contributor)
            if pp is None or abs(pp) < MATERIAL_SHARE_PP:
                continue
            shifts.append((abs(pp), pp, contributor))
        if not shifts:
            continue
        for _magnitude, pp, contributor in [max(shifts, key=lambda s: s[0])]:
            direction = "increased" if pp > 0 else "reduced"
            out.append(_item(
                ctx, CONCENTRATION_SHIFT,
                f"{contributor.category} share of the book {direction} "
                f"{abs(pp):.1f} percentage points.",
                f"Within {bridge.label.lower()}, {contributor.category} moved "
                f"{contributor.delta:+,.0f} over the period, changing its share "
                f"of funded balance by {pp:+.1f} percentage points.",
                severity=SEVERITY_ATTENTION,
                discriminator=f"{key}:{contributor.category}",
                metrics={"delta": contributor.delta, "share_change_pp": pp,
                         "dimension": bridge.dimension_col},
                source_date=bridge.end_date,
                methodology={"source": "governed funded attribution bridge"}))
    return out


def _ltv_items(ctx, evolution) -> List[Insight]:
    periods = list((evolution or {}).get("periods") or ())
    if len(periods) < 2:
        return []
    from .insights import _as_points
    raw = [(p.get("metrics") or {}).get("wa_ltv") for p in periods[-2:]]
    prior, current = _as_points(raw)
    if prior is None or current is None:
        return []
    delta = current - prior
    if abs(delta) < MATERIAL_LTV_PP:
        return []
    worsening = delta > 0
    return [_item(
        ctx, LTV_MOVEMENT,
        f"Weighted average current LTV {'increased' if worsening else 'improved'} "
        f"{abs(delta):.1f} percentage points to {current:.1f}%.",
        f"Balance-weighted current LTV moved from {prior:.1f}% to {current:.1f}% "
        f"over the period.",
        severity=SEVERITY_ATTENTION if worsening else SEVERITY_INFO,
        metrics={"wa_ltv": current, "wa_ltv_prior": prior, "change_pp": delta},
        methodology={"source": "governed funded evolution series"})]


def _coverage_items(ctx, movement, concentration) -> List[Insight]:
    """Dimensions the book cannot report on — a coverage gap, stated once."""
    missing = [b.label for b in (movement or {}).values()
               if not getattr(b, "available", False)
               and "carries no" in str(getattr(b, "reason", ""))]
    if not missing:
        return []
    return [_item(
        ctx, DATA_COVERAGE,
        f"{len(missing)} reporting dimension(s) are not available on this book.",
        "The funded data carries no " + ", ".join(m.lower() for m in missing[:4])
        + ". Those breakdowns are omitted rather than estimated.",
        severity=SEVERITY_INFO,
        metrics={"missing_dimensions": missing},
        methodology={"source": "governed field coverage"})]


# --------------------------------------------------------------------------- #
# Selection.
# --------------------------------------------------------------------------- #

def _rank(item: Insight) -> Tuple[int, int, str, str]:
    return (-SEVERITY_RANK.get(item.severity, 0),
            -WATCH_PRIORITY.get(item.insight_type, 0),
            item.insight_type, item.discriminator or "")


def build(data: Any) -> Dict[str, Any]:
    """The watch list for one deck: at most five items, plus stable positives."""
    portfolio = getattr(data, "portfolio", None)
    ctx = {
        "tenant_id": portfolio.tenant_id if portfolio else "-",
        "portfolio_id": portfolio.client_id if portfolio else "-",
        "portfolio_context": portfolio.report_scope if portfolio else "total",
        "as_of_date": getattr(data, "reporting_date", None),
        "comparison_date": None,
    }
    produced: List[Insight] = []
    for generator in (
        lambda: _concentration_items(ctx, getattr(data, "concentration", None)),
        lambda: _portfolio_items(ctx, portfolio),
        lambda: _movement_items(ctx, getattr(data, "movement", None)),
        lambda: _ltv_items(ctx, getattr(data, "funded_evolution", None)),
        lambda: _coverage_items(ctx, getattr(data, "movement", None),
                                getattr(data, "concentration", None)),
    ):
        try:
            produced.extend(generator())
        except Exception:  # noqa: BLE001 — one generator, one section
            continue

    ordered = sorted(produced, key=_rank)
    watch = [i for i in ordered
             if i.severity in (SEVERITY_CONCERN, SEVERITY_ATTENTION)][:MAX_WATCH_ITEMS]
    positives = [i for i in ordered if i.severity == SEVERITY_INFO][:MAX_POSITIVES]
    for i, item in enumerate(watch):
        item.priority = MAX_WATCH_ITEMS * 10 - i * 10
    return {
        "watch": watch,
        "observations": positives,
        "count": len(watch),
        # An empty list is a finding in itself, and must be stated rather than
        # leaving the reader to wonder whether the check ran.
        "none_identified": not watch,
    }
