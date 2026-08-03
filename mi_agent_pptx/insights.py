"""mi_agent_pptx.insights — the deterministic investment summary.

Turns governed MI payloads into a small set of investor-facing observations.
There is **no LLM anywhere** — not in generation, not in wording, not in
selection. Every sentence is a template over a figure that a governed compute
function already produced, so the same approved run always yields the same
summary, word for word.

This EXTENDS the existing governed insight contract rather than replacing it:
:class:`~mi_agent_api.insight_contract.Insight`, :class:`Omission`, the severity
vocabulary and the formatting helpers are imported from
``mi_agent_api``. What is added here is a set of generators for the **funded
book** — in particular the direct-versus-acquired attribution that explains why
the total moved — because the Weekly Portfolio Brief's engine resolves from
weekly *pipeline* extracts and returns ``unavailable`` without them.

Nothing here calculates an economic value. Balances, counts, weighted averages
and movements are read from ``compute_funded_snapshot`` and the governed
evolution payloads. A generator that has no governed figure returns an
``Omission`` with the reason, never an estimate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mi_agent_api.insight_contract import (
    SEVERITY_ATTENTION, SEVERITY_CONCERN, SEVERITY_INFO, SEVERITY_RANK,
    Insight, Omission, OMITTED_IMMATERIAL, OMITTED_UNAVAILABLE,
)
from mi_agent_api.insight_generators import money, pct, signed_money, signed_pct

Result = Tuple[List[Insight], List[Omission]]

# -- deck insight types ------------------------------------------------------
FUNDED_MOVEMENT = "FUNDED_MOVEMENT"
MOVEMENT_ATTRIBUTION = "MOVEMENT_ATTRIBUTION"
PORTFOLIO_MIX = "PORTFOLIO_MIX"
PORTFOLIO_RUNOFF = "PORTFOLIO_RUNOFF"
WEIGHTED_LTV_TREND = "WEIGHTED_LTV_TREND"
CONCENTRATION = "CONCENTRATION"
RISK_LIMITS = "RISK_LIMITS"
PIPELINE_OUTLOOK = "PIPELINE_OUTLOOK"
FORECAST_OUTLOOK = "FORECAST_OUTLOOK"
REPORTING_SCOPE = "REPORTING_SCOPE"
MOVEMENT_DRIVERS = "MOVEMENT_DRIVERS"
PORTFOLIO_POSITION = "PORTFOLIO_POSITION"

#: Ordering weight for the executive summary. Attribution outranks the headline
#: movement because "why" is the investor question; scope and risk caveats
#: outrank both because a caveat the reader misses is a caveat that failed.
DECK_TYPE_PRIORITY: Dict[str, int] = {
    REPORTING_SCOPE: 110,
    RISK_LIMITS: 100,
    FUNDED_MOVEMENT: 90,
    PORTFOLIO_POSITION: 88,
    MOVEMENT_ATTRIBUTION: 85,
    MOVEMENT_DRIVERS: 82,
    PORTFOLIO_MIX: 80,
    PORTFOLIO_RUNOFF: 70,
    CONCENTRATION: 60,
    WEIGHTED_LTV_TREND: 50,
    FORECAST_OUTLOOK: 40,
    PIPELINE_OUTLOOK: 30,
}

#: Below this, a balance movement is reported as immaterial rather than narrated.
MATERIAL_BALANCE_PCT = 0.1
#: Below this (percentage points), a weighted-average move is not worth a line.
MATERIAL_POINTS = 0.1

MAX_INSIGHTS = 8
MIN_INSIGHTS = 5

#: The governed insight vocabulary this module emits, for the artefact metadata.
#: Bumped whenever the observation vocabulary or wording changes, so a
#: reader can tell which engine produced a given pack.
INSIGHT_VERSION = "deck-3"


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #

def _pct_change(current: Optional[float], change: Optional[float]) -> Optional[float]:
    """Percentage change implied by a level and its movement."""
    if current is None or change is None:
        return None
    prior = current - change
    if not prior:
        return None
    return change / abs(prior) * 100.0


def _as_points(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Normalise a percentage series to percentage POINTS.

    The decision is made once for the whole series — if every value sits at or
    below 1.5 the series is a fraction — rather than per value, so a genuine
    1.2% rate in a series of percentages is never silently multiplied by 100.
    """
    present = [v for v in values if v is not None]
    if not present:
        return list(values)
    fractional = max(abs(v) for v in present) <= 1.5
    return [None if v is None else (v * 100.0 if fractional else v) for v in values]


def _base(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "tenant_id": ctx.get("tenant_id") or "-",
        "portfolio_id": ctx.get("portfolio_id") or "-",
        "portfolio_context": ctx.get("portfolio_context") or "total",
        "run_id": ctx.get("run_id"),
        "as_of_date": ctx.get("as_of_date"),
        "comparison_date": ctx.get("comparison_date"),
    }


def _insight(ctx: Mapping[str, Any], insight_type: str, headline: str, summary: str,
             *, severity: str = SEVERITY_INFO, metrics: Optional[Dict[str, Any]] = None,
             contributors: Optional[Dict[str, Any]] = None,
             methodology: Optional[Dict[str, Any]] = None,
             discriminator: str = "") -> Insight:
    return Insight(insight_type=insight_type, headline=headline, summary=summary,
                   severity=severity, discriminator=discriminator,
                   metrics=metrics or {}, contributors=contributors or {},
                   methodology=methodology or {},
                   source_dates=dict(ctx.get("source_dates") or {}),
                   **_base(ctx))


# --------------------------------------------------------------------------- #
# Generators — one per observation, each isolated.
# --------------------------------------------------------------------------- #

def movement_drivers(ctx: Mapping[str, Any], movement) -> Result:
    """Where the movement came from, across the governed dimensions.

    This is the observation an investor asks for after "how much" — and it is
    grounded in the attribution bridge, so the named contributor is evidenced
    rather than inferred.
    """
    from . import movement as _mv

    bridges = {k: b for k, b in (movement or {}).items()
               if getattr(b, "available", False)}
    if not bridges:
        return [], [Omission(MOVEMENT_DRIVERS,
                             "no governed attribution is available for this "
                             "period.", OMITTED_UNAVAILABLE)]
    # Prefer the dimensions an investor asks about first.
    for key in ("region", "broker", "ticket", "ltv"):
        bridge = bridges.get(key)
        if bridge is None:
            continue
        ups, downs = bridge.movers(limit=1)
        if not ups and not downs:
            continue
        parts = []
        if ups:
            parts.append(f"{ups[0].category} contributed the largest increase "
                         f"({_mv._signed(ups[0].delta)})")
        if downs:
            parts.append(f"{downs[0].category} the largest reduction "
                         f"({_mv._signed(downs[0].delta)})")
        return [_insight(
            ctx, MOVEMENT_DRIVERS,
            f"Movement was concentrated by {bridge.label.lower()}.",
            "; ".join(parts) + f", measured across {bridge.label.lower()} "
            f"between {bridge.start_period} and {bridge.end_period}.",
            contributors={c.category: c.delta for c in bridge.contributors
                          if not c.is_other},
            metrics={"dimension": bridge.dimension_col,
                     "net_change": bridge.total_delta},
            methodology={"source": "governed funded attribution bridge",
                         "basis": "per-category deltas sum exactly to the "
                                  "headline movement"},
        )], []
    return [], [Omission(MOVEMENT_DRIVERS,
                         "no dimension moved materially over the period.",
                         OMITTED_IMMATERIAL)]


def portfolio_position(ctx: Mapping[str, Any], portfolio) -> Result:
    """The book's position, stated when nothing else cleared its threshold.

    A stable portfolio is a finding, not an absence of one. Without this a book
    that genuinely did not move produced an EMPTY executive summary, which reads
    as a failed run rather than as a quiet period — and blocked publication on a
    perfectly valid pack.
    """
    if portfolio is None or portfolio.total_balance is None:
        return [], [Omission(PORTFOLIO_POSITION, "no governed funded balance.",
                             OMITTED_UNAVAILABLE)]
    from .deck_context import raw_of
    change = raw_of(portfolio.snapshot, "mom_balance")
    loans = int(portfolio.loan_count or 0)
    if change is None:
        detail = ("This is the first reporting period available for the book, so "
                  "no period-on-period comparison is shown.")
        headline = (f"Funded portfolio stands at {money(portfolio.total_balance)} "
                    f"across {loans:,} loans.")
    else:
        pc = _pct_change(portfolio.total_balance, change)
        detail = (f"The book moved {signed_money(change)}"
                  + (f" ({signed_pct(pc)})" if pc is not None else "")
                  + ", below the reporting threshold for a material movement.")
        headline = (f"Funded portfolio was broadly unchanged at "
                    f"{money(portfolio.total_balance)} across {loans:,} loans.")
    return [_insight(
        ctx, PORTFOLIO_POSITION, headline, detail,
        metrics={"funded_balance": portfolio.total_balance,
                 "loan_count": portfolio.loan_count, "balance_change": change},
        methodology={"source": "compute_funded_snapshot"},
    )], []


def funded_movement(ctx: Mapping[str, Any], portfolio) -> Result:
    """How the funded book moved over the period."""
    if portfolio is None:
        return [], [Omission(FUNDED_MOVEMENT, "no portfolio context resolved.",
                             OMITTED_UNAVAILABLE)]
    from .deck_context import raw_of
    balance = portfolio.total_balance
    change = raw_of(portfolio.snapshot, "mom_balance")
    if balance is None:
        return [], [Omission(FUNDED_MOVEMENT, "no governed funded balance.",
                             OMITTED_UNAVAILABLE)]
    if change is None:
        return [], [Omission(FUNDED_MOVEMENT,
                             "no prior reporting period to compare against.",
                             OMITTED_UNAVAILABLE)]
    pc = _pct_change(balance, change)
    if pc is not None and abs(pc) < MATERIAL_BALANCE_PCT:
        return [], [Omission(FUNDED_MOVEMENT,
                             f"funded balance moved {signed_pct(pc)}, below the "
                             f"{MATERIAL_BALANCE_PCT}% reporting threshold.",
                             OMITTED_IMMATERIAL)]
    verb = "increased" if change > 0 else ("decreased" if change < 0 else "was unchanged")
    pct_part = f" ({signed_pct(pc)})" if pc is not None else ""
    return [_insight(
        ctx, FUNDED_MOVEMENT,
        f"Funded portfolio {verb} {money(abs(change))}{pct_part} to {money(balance)}.",
        f"Total funded balance closed the period at {money(balance)} across "
        f"{int(portfolio.loan_count or 0):,} loans, {verb} {money(abs(change))}"
        f"{pct_part} against the prior reporting period.",
        metrics={"funded_balance": balance, "balance_change": change,
                 "balance_change_pct": pc, "loan_count": portfolio.loan_count},
        methodology={"source": "compute_funded_snapshot",
                     "basis": "month-on-month against the prior dated funded cut"},
    )], []


def movement_attribution(ctx: Mapping[str, Any], portfolio) -> Result:
    """WHY the total moved — the direct/acquired decomposition.

    This is the observation the deck exists to make: a blended total of a growing
    origination book and a redeeming acquired book describes neither.
    """
    if portfolio is None or len(portfolio.type_slices) < 2:
        return [], [Omission(MOVEMENT_ATTRIBUTION,
                             "only one portfolio type is in scope, so the total "
                             "needs no attribution.", OMITTED_IMMATERIAL)]
    moves = [(s, s.balance_movement) for s in portfolio.type_slices
             if s.balance_movement is not None]
    if not moves:
        return [], [Omission(MOVEMENT_ATTRIBUTION,
                             "no prior reporting period to attribute movement "
                             "against.", OMITTED_UNAVAILABLE)]
    risers = sorted([m for m in moves if m[1] > 0], key=lambda m: -m[1])
    fallers = sorted([m for m in moves if m[1] < 0], key=lambda m: m[1])
    parts: List[str] = []
    if risers:
        parts.append(" and ".join(f"{s.label.lower()} ({signed_money(v)})"
                                  for s, v in risers))
    detail: List[str] = []
    if risers:
        lead = risers[0][0]
        detail.append(f"Growth was driven by {lead.label.lower()} "
                      f"({signed_money(risers[0][1])})")
    if fallers:
        detail.append(", ".join(f"the {s.label.lower()} reduced {money(abs(v))}"
                                for s, v in fallers))
    headline = ("Growth driven by " + parts[0] + "." if parts else
                "The book reduced across every portfolio type.")
    return [_insight(
        ctx, MOVEMENT_ATTRIBUTION, headline,
        "; ".join(detail) + "." if detail else headline,
        contributors={s.portfolio_type: v for s, v in moves},
        metrics={"attributed_movement": sum(v for _s, v in moves)},
        methodology={"source": "compute_funded_snapshot per portfolio type",
                     "basis": "the same month-on-month basis as the total"},
    )], []


def portfolio_mix(ctx: Mapping[str, Any], portfolio) -> Result:
    """What the book is made of, by type."""
    if portfolio is None or len(portfolio.type_slices) < 2:
        return [], [Omission(PORTFOLIO_MIX, "only one portfolio type is in scope.",
                             OMITTED_IMMATERIAL)]
    total = portfolio.total_balance
    if not total:
        return [], [Omission(PORTFOLIO_MIX, "no governed funded balance.",
                             OMITTED_UNAVAILABLE)]
    shares = [(s, (s.balance or 0.0) / total * 100.0) for s in portfolio.type_slices]
    shares.sort(key=lambda x: -x[1])
    lead, lead_share = shares[0]
    text = "; ".join(f"{s.label.lower()} {pct(share)} of balance across "
                     f"{int(s.loan_count or 0):,} loans" for s, share in shares)
    return [_insight(
        ctx, PORTFOLIO_MIX,
        f"{lead.label} represent {pct(lead_share)} of funded balance.",
        text.capitalize() + ".",
        metrics={s.portfolio_type: {"balance": s.balance, "share_pct": share,
                                    "loan_count": s.loan_count}
                 for s, share in shares},
        methodology={"source": "compute_funded_snapshot per portfolio type"},
    )], []


def portfolio_runoff(ctx: Mapping[str, Any], portfolio) -> Result:
    """A book that is redeeming — reported as run-off, not as underperformance."""
    if portfolio is None:
        return [], [Omission(PORTFOLIO_RUNOFF, "no portfolio context resolved.",
                             OMITTED_UNAVAILABLE)]
    out: List[Insight] = []
    for s in portfolio.type_slices:
        change = s.balance_movement
        if change is None or change >= 0:
            continue
        pc = _pct_change(s.balance, change)
        if pc is not None and abs(pc) < MATERIAL_BALANCE_PCT:
            continue
        loans = s.loan_movement
        n_exited = int(abs(loans)) if loans is not None and loans < 0 else 0
        exited = (f" across {n_exited:,} fewer loan{'s' if n_exited != 1 else ''}"
                  if n_exited else "")
        out.append(_insight(
            ctx, PORTFOLIO_RUNOFF,
            f"{s.label} reduced during the period by {money(abs(change))}"
            f"{'' if pc is None else ' (' + signed_pct(pc) + ')'}.",
            # Deliberately states the OBSERVED movement only. Attributing it to
            # redemption or amortisation would name a cause the balance
            # decomposition does not evidence, and calling it "expected" would
            # imply a forecast that has not been produced.
            f"The {s.label.lower()} closed at {money(s.balance)}{exited}. "
            f"The reduction is observed in the period balance movement; no "
            f"forecast of continued run-off is implied.",
            discriminator=s.portfolio_type,
            metrics={"balance": s.balance, "balance_change": change,
                     "balance_change_pct": pc, "loan_change": loans},
            methodology={"source": "compute_funded_snapshot per portfolio type"},
        ))
    if not out:
        return [], [Omission(PORTFOLIO_RUNOFF, "no portfolio type reduced "
                             "materially over the period.", OMITTED_IMMATERIAL)]
    return out, []


def weighted_ltv_trend(ctx: Mapping[str, Any], evolution: Optional[Mapping[str, Any]]) -> Result:
    """Weighted-average current LTV and its direction, from the governed series."""
    periods = list((evolution or {}).get("periods") or ())
    if len(periods) < 2:
        return [], [Omission(WEIGHTED_LTV_TREND,
                             "fewer than two reporting periods are available.",
                             OMITTED_UNAVAILABLE)]
    raw = [(p.get("metrics") or {}).get("wa_ltv") for p in periods[-2:]]
    prior, current = _as_points(raw)
    if current is None or prior is None:
        return [], [Omission(WEIGHTED_LTV_TREND,
                             "weighted LTV is not available on both periods.",
                             OMITTED_UNAVAILABLE)]
    delta = current - prior
    if abs(delta) < MATERIAL_POINTS:
        return [], [Omission(WEIGHTED_LTV_TREND,
                             f"weighted LTV moved {delta:+.2f}pp, below the "
                             f"{MATERIAL_POINTS}pp reporting threshold.",
                             OMITTED_IMMATERIAL)]
    # A falling LTV is a lower-risk book; say "improved" only in that direction.
    verb = "improved" if delta < 0 else "increased"
    return [_insight(
        ctx, WEIGHTED_LTV_TREND,
        f"Weighted average current LTV {verb} to {pct(current)}.",
        f"Balance-weighted current LTV moved {delta:+.1f} percentage points "
        f"from {pct(prior)} to {pct(current)} over the period.",
        metrics={"wa_ltv": current, "wa_ltv_prior": prior, "wa_ltv_change_pp": delta},
        methodology={"source": "funded evolution series",
                     "basis": "balance-weighted current LTV"},
    )], []


def concentration(ctx: Mapping[str, Any], geo: Optional[Mapping[str, Any]]) -> Result:
    """Geographic concentration — the covenant-relevant exposure view."""
    areas = sorted((geo or {}).get("areas") or (),
                   key=lambda a: a.get("balance", 0), reverse=True)
    total = (geo or {}).get("total") or sum(a.get("balance", 0) for a in areas)
    if not areas or not total:
        return [], [Omission(CONCENTRATION, "no geographic exposure resolved.",
                             OMITTED_UNAVAILABLE)]
    top = areas[0]
    top5 = sum(a.get("balance", 0) for a in areas[:5]) / total * 100.0
    top_share = (top.get("balance", 0) / total) * 100.0
    name = top.get("itl3_name") or top.get("itl3_code") or "the largest area"
    return [_insight(
        ctx, CONCENTRATION,
        f"Top-5 area concentration is {pct(top5)} of funded balance.",
        f"The largest single exposure is {name} at {pct(top_share)}; the top five "
        f"areas account for {pct(top5)} of the book across "
        f"{(geo or {}).get('areaCount', len(areas))} areas.",
        metrics={"top5_pct": top5, "top_area": name, "top_area_pct": top_share,
                 "area_count": (geo or {}).get("areaCount", len(areas))},
        methodology={"source": "exposure_by_itl3"},
    )], []


def risk_limits(ctx: Mapping[str, Any], risk: Optional[Mapping[str, Any]]) -> Result:
    """Whether the book is operating inside its approved limits."""
    tests = (risk or {}).get("tests") or []
    summary = (risk or {}).get("summary") or {}
    if not tests:
        return [], [Omission(RISK_LIMITS, "no governed risk-limit artefact for "
                             "this run.", OMITTED_UNAVAILABLE)]
    breaches = int(summary.get("breaches") or 0)
    warnings = int(summary.get("warnings") or 0)
    passed = int(summary.get("testsPassed") or 0)
    if breaches:
        sev, head = SEVERITY_CONCERN, (
            f"{breaches} concentration limit{'s' if breaches != 1 else ''} breached.")
    elif warnings:
        sev, head = SEVERITY_ATTENTION, (
            f"{warnings} concentration limit{'s' if warnings != 1 else ''} within "
            f"warning range.")
    else:
        sev, head = SEVERITY_INFO, "No concentration limits breached."
    return [_insight(
        ctx, RISK_LIMITS, head,
        f"Of {len(tests)} governed limit tests, {passed} passed, {warnings} are in "
        f"warning range and {breaches} are in breach at the reporting date.",
        severity=sev,
        metrics={"tests": len(tests), "passed": passed, "warnings": warnings,
                 "breaches": breaches},
        methodology={"source": "compute_risk_limits"},
    )], []


def pipeline_outlook(ctx: Mapping[str, Any], pipeline: Optional[Mapping[str, Any]]) -> Result:
    """What is likely to fund next."""
    amount = (pipeline or {}).get("pipelineAmount")
    cases = (pipeline or {}).get("pipelineRowCount")
    if not amount or not cases:
        return [], [Omission(PIPELINE_OUTLOOK, "no governed pipeline source for "
                             "this book.", OMITTED_UNAVAILABLE)]
    weighted = (pipeline or {}).get("weightedExpectedFundedAmount")
    tail = (f", of which {money(weighted)} is probability-weighted expected funding"
            if weighted else "")
    return [_insight(
        ctx, PIPELINE_OUTLOOK,
        f"Pipeline of {money(amount)} across {int(cases):,} cases.",
        f"The origination pipeline stands at {money(amount)} across "
        f"{int(cases):,} cases{tail}.",
        metrics={"pipeline_amount": amount, "case_count": cases,
                 "weighted_expected": weighted},
        methodology={"source": "compute_pipeline_snapshot",
                     "basis": "direct origination only; acquired books do not originate"},
    )], []


def forecast_outlook(ctx: Mapping[str, Any], forecast: Optional[Mapping[str, Any]]) -> Result:
    """Where the portfolio is heading on the current book."""
    bridge = (forecast or {}).get("forecastBridge") or {}
    projected = bridge.get("forecastFundedBalance")
    if projected is None:
        return [], [Omission(FORECAST_OUTLOOK, "no forecast is available for this "
                             "book.", OMITTED_UNAVAILABLE)]
    funded = bridge.get("fundedBalance")
    weighted = bridge.get("weightedExpectedFundedAmount")
    # With no pipeline contribution the "forecast" is the funded balance restated.
    # Reporting that as an outlook would imply a projection that was never made.
    if not weighted:
        return [], [Omission(FORECAST_OUTLOOK,
                             "no pipeline contributes to this book, so the forecast "
                             "would restate the funded balance.", OMITTED_UNAVAILABLE)]
    return [_insight(
        ctx, FORECAST_OUTLOOK,
        f"Forecast funded balance of {money(projected)}.",
        f"Adding {money(weighted)} of probability-weighted expected completions to "
        f"{money(funded)} of funded balance gives a forecast of {money(projected)}. "
        f"This is the current book's expected completions only, not future new "
        f"business.",
        metrics={"forecast_funded_balance": projected, "funded_balance": funded,
                 "weighted_expected": weighted},
        methodology={"source": "compute_forecast_bridge",
                     "basis": "funded + Σ(weighted pipeline); expected case, not a stress"},
    )], []


def reporting_scope(ctx: Mapping[str, Any], portfolio) -> Result:
    """A caveat when the constituent books do not share one reporting date."""
    if portfolio is None or not portfolio.has_mixed_reporting_dates:
        return [], [Omission(REPORTING_SCOPE, "all books in scope share one "
                             "reporting date.", OMITTED_IMMATERIAL)]
    pairs = "; ".join(f"{lbl}: {d}" for lbl, d in
                      sorted(portfolio.type_reporting_dates.items()))
    return [_insight(
        ctx, REPORTING_SCOPE,
        "Constituent books are reported as at different dates.",
        f"Total funded balance combines books at different reporting dates "
        f"({pairs}). Period-on-period totals should be read with that in mind.",
        severity=SEVERITY_ATTENTION,
        metrics={"reporting_dates": portfolio.reporting_dates},
        methodology={"source": "governed portfolio registry"},
    )], []


# --------------------------------------------------------------------------- #
# Selection.
# --------------------------------------------------------------------------- #

def _rank(ins: Insight) -> Tuple[int, int, str, str]:
    """Total order: severity, then investor priority, then a stable tiebreak."""
    return (-SEVERITY_RANK.get(ins.severity, 0),
            -DECK_TYPE_PRIORITY.get(ins.insight_type, 0),
            ins.insight_type, ins.discriminator or "")


def select(insights: List[Insight], *, limit: int = MAX_INSIGHTS
           ) -> Tuple[List[Insight], List[Omission]]:
    """Order deterministically and cut to the executive-summary limit.

    Mirrors the Weekly Brief's selector: anything dropped by the cap becomes an
    explicit omission, because a summary that silently truncated would read as
    "this is everything".
    """
    ordered = sorted(insights, key=_rank)
    kept, dropped = ordered[:limit], ordered[limit:]
    for i, ins in enumerate(kept):
        ins.priority = limit * 10 - i * 10
    omissions = [
        Omission(t, f"{sum(1 for d in dropped if d.insight_type == t)} further "
                    f"{t} observation(s) were not shown: the executive summary is "
                    f"capped at {limit}.", "capped")
        for t in sorted({d.insight_type for d in dropped})
    ]
    return kept, omissions


def build(data: Any, *, limit: int = MAX_INSIGHTS) -> Dict[str, Any]:
    """The executive summary for one deck.

    Each generator is isolated: one failing must not cost the reader the others.
    """
    portfolio = getattr(data, "portfolio", None)
    ctx: Dict[str, Any] = {
        "tenant_id": portfolio.tenant_id if portfolio else "-",
        "portfolio_id": portfolio.client_id if portfolio else "-",
        "portfolio_context": portfolio.report_scope if portfolio else "total",
        "run_id": getattr(data, "run_id", None),
        "as_of_date": getattr(data, "reporting_date", None),
        "comparison_date": None,
        "source_dates": (portfolio.reporting_dates if portfolio else {}),
    }

    steps = [
        (REPORTING_SCOPE, lambda: reporting_scope(ctx, portfolio)),
        (FUNDED_MOVEMENT, lambda: funded_movement(ctx, portfolio)),
        (MOVEMENT_ATTRIBUTION, lambda: movement_attribution(ctx, portfolio)),
        (MOVEMENT_DRIVERS,
         lambda: movement_drivers(ctx, getattr(data, "movement", None))),
        (PORTFOLIO_MIX, lambda: portfolio_mix(ctx, portfolio)),
        (PORTFOLIO_RUNOFF, lambda: portfolio_runoff(ctx, portfolio)),
        (WEIGHTED_LTV_TREND,
         lambda: weighted_ltv_trend(ctx, getattr(data, "funded_evolution", {}))),
        (CONCENTRATION, lambda: concentration(ctx, getattr(data, "geo", {}))),
        (RISK_LIMITS, lambda: risk_limits(ctx, getattr(data, "risk", {}))),
        (FORECAST_OUTLOOK, lambda: forecast_outlook(ctx, getattr(data, "forecast", {}))),
        (PIPELINE_OUTLOOK, lambda: pipeline_outlook(ctx, getattr(data, "pipeline", {}))),
    ]

    produced: List[Insight] = []
    omitted: List[Omission] = []
    failures = 0
    for label, fn in steps:
        try:
            ins, oms = fn()
            produced.extend(ins)
            omitted.extend(oms)
        except Exception as exc:  # noqa: BLE001 — one generator, one section
            failures += 1
            omitted.append(Omission(label, f"generation failed: "
                                           f"{type(exc).__name__}: {exc}", "error"))

    if not produced:
        # Everything was immaterial or unavailable. State the position rather
        # than returning an empty summary.
        try:
            fallback, fallback_omissions = portfolio_position(ctx, portfolio)
            produced.extend(fallback)
            omitted.extend(fallback_omissions)
        except Exception as exc:  # noqa: BLE001
            omitted.append(Omission(PORTFOLIO_POSITION, f"generation failed: "
                                                        f"{type(exc).__name__}: {exc}",
                                    "error"))

    kept, capped = select(produced, limit=limit)
    omitted.extend(capped)
    return {
        "insight_version": INSIGHT_VERSION,
        "status": "partial" if failures else "success",
        "portfolio_context": ctx["portfolio_context"],
        "as_of_date": ctx["as_of_date"],
        "insights": kept,
        "omitted": omitted,
        "count": len(kept),
        "below_minimum": len(kept) < MIN_INSIGHTS,
    }
