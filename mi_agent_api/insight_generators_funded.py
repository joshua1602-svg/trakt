#!/usr/bin/env python3
"""One generator per funded (monthly) insight type.

The monthly review had no materiality layer. The weekly pipeline brief gates
nine observations against configured thresholds and records an explicit omission
for everything it suppressed; the funded card stated its figures unconditionally,
so a £4k movement and a £24m movement produced the same three bullets and a
reader had no way to tell a quiet month from a loud one.

These generators close that, under the same rules the weekly ones follow:

* **Thresholds come from configuration, never from a literal here.** A number
  written in this file would be a portfolio assumption baked into code.
* **Nothing is calculated.** Every figure arrives from ``period_movement``,
  ``funded_composition`` or the governed concentration evaluation, and is
  carried through. The one thing computed here is a share of a total, which is
  the arithmetic of deciding whether to speak.
* **Suppression is stated.** Below threshold produces an ``Omission`` with
  category ``immaterial`` — silence is what "we did not look" sounds like.

WHAT IS DELIBERATELY ABSENT
---------------------------
There is no funded data-quality generator. The governed data-quality signals for
a funded book — field completeness, validation exceptions, regulatory coverage —
exist as agent tools over the lineage index, not as anything the notification
resolver holds, and building a second validation path inside notifications to
reach them is exactly the duplication this package refuses elsewhere. The
Portfolio Review agent reads those tools directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import insight_config as cfg
from .insight_contract import (
    FUNDED_COMPOSITION, FUNDED_LTV_MOVEMENT, FUNDED_MIX_SHIFT, FUNDED_MOVEMENT,
    OMITTED_IMMATERIAL, OMITTED_UNAVAILABLE, RISK_LIMIT_TRANSITION,
    SEVERITY_ATTENTION, SEVERITY_CONCERN, SEVERITY_INFO,
    UNDERLYING_BOOK_MOVEMENT, Insight, Omission,
)
from .insight_generators import _base, direction, money, pct, signed_money, \
    signed_pct

Result = Tuple[List[Insight], List[Omission]]

#: Human wording for each governed decomposition component. Presentation only —
#: the component keys are owned by ``funded_composition``.
COMPONENT_LABEL = {
    "portfolio_additions": "portfolio additions",
    "portfolio_disposals": "portfolio disposals",
    "organic_new_lending": "new lending",
    "exits": "redemptions and exits",
    "existing_book_movement": "existing-book movement",
}

#: Concentration statuses, worst first, for wording a transition.
_WORSENING = "deteriorated"
_IMPROVING = "improved"


def _pp(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:+.1f}pp"


def _ltv_severity(delta: Optional[float], floor: float) -> str:
    """A material RISE in weighted LTV is a risk finding; a fall is an observation.

    Direction is the only judgement here, and it is not a threshold: rising LTV
    is credit deterioration and falling LTV is not, which is true of every
    secured book and needs no portfolio-specific number to say. The size gate is
    the configured materiality floor already applied above — nothing new is
    introduced, and the estate does not acquire a second risk framework.

    It matters because the Risk Review is otherwise entitled to say "no material
    portfolio risks were identified" in a month when the book's weighted LTV
    rose twenty points. That sentence sits beside the movement on the same card.
    Ranked below breaches, limit transitions and data quality, so it surfaces
    only when nothing more contractual is competing for the reader's attention.
    """
    if delta is None or abs(delta) < floor:
        return SEVERITY_INFO
    return SEVERITY_ATTENTION if delta > 0 else SEVERITY_INFO


def _addition_headline(lead: Dict[str, Any]) -> str:
    """The headline for a period an addition dominates.

    A share of the movement is only quotable when the movement is larger than
    the addition and points the same way. Otherwise the addition is stated
    against the book, which is a figure that always means something.
    """
    if lead.get("exceeds_movement"):
        share = lead.get("share_of_closing_balance")
        if share is not None:
            return (f"{lead['label']} added {money(lead['balance'])}, "
                    f"{pct(share * 100)} of the closing book")
        return f"{lead['label']} added {money(lead['balance'])}"
    return (f"{lead['label']} accounts for "
            f"{pct((lead['share_of_movement'] or 0) * 100)} of the movement")


# --------------------------------------------------------------------------- #
# 1. Headline funded movement
# --------------------------------------------------------------------------- #
def funded_movement(ctx: Dict[str, Any],
                    movement: Optional[Dict[str, Any]]) -> Result:
    """Balance and loan-count movement against the prior reporting period."""
    if not movement or not movement.get("available"):
        return [], [Omission(FUNDED_MOVEMENT,
                             (movement or {}).get("reason")
                             or "No comparable prior reporting period is available.",
                             OMITTED_UNAVAILABLE)]

    t = cfg.thresholds("funded_movement")
    current = movement.get("current") or {}
    prior = movement.get("prior") or {}
    delta = movement.get("delta") or {}

    balance = current.get("funded_balance")
    change = delta.get("funded_balance")
    opening = prior.get("funded_balance")
    change_pct = (change / abs(opening) * 100.0
                  if change is not None and opening else None)

    # One gate: movement against the OPENING book. A funded book has no
    # equivalent of the pipeline's second "share of stock" gate, because for a
    # stock measured against itself the two gates are the same number.
    if change_pct is None or abs(change_pct) < t.get("min_change_pct", 1.0):
        return [], [Omission(
            FUNDED_MOVEMENT,
            f"Funded balance moved {signed_pct(change_pct)}, below the "
            f"{t.get('min_change_pct')}% materiality threshold.",
            OMITTED_IMMATERIAL)]

    loan_change = delta.get("loan_count")
    summary = (f"Funded balance {direction(change)} by {money(abs(change or 0))} "
               f"to {money(balance)}.")
    if loan_change:
        summary += (f" Loan count {direction(loan_change)} by "
                    f"{abs(int(loan_change))} to {int(current.get('loan_count') or 0):,}.")

    return [Insight(
        insight_type=FUNDED_MOVEMENT,
        headline=f"Funded balance {direction(change)} {signed_pct(change_pct)}",
        summary=summary,
        severity=SEVERITY_INFO,
        metrics={
            "current_balance": balance,
            "prior_balance": opening,
            "change": change,
            "change_pct": None if change_pct is None else round(change_pct, 2),
            "current_loan_count": current.get("loan_count"),
            "prior_loan_count": prior.get("loan_count"),
            "loan_count_change": loan_change,
            # Carried, not gated: the agent drills on these, the card does not
            # lead with them.
            "wa_interest_rate_change": delta.get("wa_interest_rate"),
            "avg_borrower_age_change": delta.get("avg_borrower_age"),
        },
        contributors={"regions": (movement.get("regionContributions") or [])[:3],
                      "cohorts": (movement.get("cohortMovements") or [])[:3]},
        methodology={"basis": "governed funded period movement",
                     "reconciliation": movement.get("reconciliation") or {}},
        source_dates={"funded_as_of": movement.get("currentReportingDate"),
                      "funded_comparison": movement.get("priorReportingDate")},
        deep_link="/mi/funded",
        **_base(ctx),
    )], []


# --------------------------------------------------------------------------- #
# 2. Composition — why the book moved
# --------------------------------------------------------------------------- #
def funded_composition(ctx: Dict[str, Any],
                       decomposition: Optional[Dict[str, Any]]) -> Result:
    """New lending, exits, accretion and portfolio additions, as components.

    Emitted whenever the decomposition names more than one material component,
    or names a portfolio addition at all. A single-component month is already
    told by the headline: "the book grew and all of it was new lending" is the
    headline plus one word, and a second card saying it is noise.
    """
    if not decomposition or not decomposition.get("available"):
        return [], [Omission(FUNDED_COMPOSITION,
                             (decomposition or {}).get("reason")
                             or "The funded movement could not be decomposed.",
                             OMITTED_UNAVAILABLE)]

    from . import funded_composition as comp

    t = cfg.thresholds("funded_composition")
    floor = float(t.get("min_component_share_of_gross_pct", 5.0)) / 100.0
    named = comp.narrative_components(decomposition)
    named = [r for r in named if r["share_of_gross"] >= floor]
    additions = decomposition.get("portfolio_additions") or []

    if not additions and len(named) < 2:
        return [], [Omission(
            FUNDED_COMPOSITION,
            "The funded movement has a single material component, which the "
            "headline movement already states.",
            OMITTED_IMMATERIAL)]

    lead = comp.dominant_addition(
        decomposition,
        share_floor=float(t.get("dominant_addition_share_pct", 50.0)) / 100.0)

    movement = decomposition.get("movement")
    parts = [f"{money(abs(r['amount']))} {COMPONENT_LABEL.get(r['component'], r['component'])}"
             for r in named]
    summary = (f"The {money(abs(movement or 0))} movement is "
               f"{', '.join(parts)}." if parts else
               f"The book moved {signed_money(movement)}.")

    severity = SEVERITY_INFO
    if lead:
        # An addition that explains the period is worth flagging, because every
        # other figure in the review has to be read against it.
        severity = SEVERITY_ATTENTION
        kind = ("the acquisition of" if lead["portfolio_type"] == comp.TYPE_ACQUIRED
                else "the addition of the source portfolio")
        if lead.get("exceeds_movement"):
            # The addition is larger than the net movement, so it cannot be
            # stated as a share of it: "£50.0m of the £5.0m movement" is not a
            # sentence about the world. Stated against the book instead, with
            # the offsetting components named so the small net is explained
            # rather than left looking like a small event.
            offsets = [r for r in named
                       if r["component"] in ("exits", "portfolio_disposals")]
            summary = (f"{kind.capitalize()} {lead['label']} added "
                       f"{money(lead['balance'])}, against a net movement of "
                       f"{signed_money(movement)}. " + summary)
            if offsets:
                summary += (" The net is smaller than the addition because "
                            + ", ".join(
                                f"{money(abs(r['amount']))} of "
                                f"{COMPONENT_LABEL.get(r['component'], r['component'])}"
                                for r in offsets) + " offset it.")
        else:
            summary = (f"{money(lead['balance'])} of the "
                       f"{money(abs(movement or 0))} movement reflects {kind} "
                       f"{lead['label']}. " + summary)

    unavailable = decomposition.get("unavailable") or {}
    return [Insight(
        insight_type=FUNDED_COMPOSITION,
        headline=(_addition_headline(lead) if lead
                  else "What moved the funded book"),
        summary=summary,
        severity=severity,
        discriminator=(lead["source_portfolio_id"] if lead else ""),
        metrics={
            "opening_balance": decomposition.get("opening_balance"),
            "closing_balance": decomposition.get("closing_balance"),
            "movement": movement,
            **{k: v for k, v in (decomposition.get("components") or {}).items()},
            **(decomposition.get("counts") or {}),
        },
        contributors={
            "portfolio_additions": additions,
            "portfolio_disposals": decomposition.get("portfolio_disposals") or [],
            "components": named,
        },
        methodology={
            "basis": (decomposition.get("reconciliation") or {}).get("basis"),
            "reconciliation": decomposition.get("reconciliation") or {},
            "provenance": ("portfolio additions are resolved from governed "
                           "source-portfolio identity, never from the size of "
                           "the movement"),
        },
        data_quality={"unavailable": unavailable} if unavailable else {},
        source_dates={"funded_as_of": decomposition.get("currentReportingDate"),
                      "funded_comparison": decomposition.get("priorReportingDate")},
        deep_link="/mi/funded",
        **_base(ctx),
    )], []


# --------------------------------------------------------------------------- #
# 3. The underlying book, when an addition dominates
# --------------------------------------------------------------------------- #
def underlying_book(ctx: Dict[str, Any],
                    decomposition: Optional[Dict[str, Any]],
                    underlying: Optional[Dict[str, Any]]) -> Result:
    """The incumbent book's own movement, excluding portfolios added this period.

    Produced only when something was added — otherwise the underlying book IS
    the book, and a second card restating the headline under a different name
    would imply a distinction that does not exist.
    """
    if not decomposition or not (decomposition.get("portfolio_additions") or []):
        return [], [Omission(
            UNDERLYING_BOOK_MOVEMENT,
            "No source portfolio was added this period, so the underlying book "
            "is the whole book.",
            OMITTED_IMMATERIAL)]
    if not underlying or not underlying.get("available"):
        return [], [Omission(
            UNDERLYING_BOOK_MOVEMENT,
            (underlying or {}).get("reason")
            or "The underlying book could not be resolved for this period.",
            OMITTED_UNAVAILABLE)]

    opening = underlying.get("opening_balance")
    movement = underlying.get("movement")
    change_pct = (movement / abs(opening) * 100.0
                  if movement is not None and opening else None)

    components = underlying.get("components") or {}
    added = len(decomposition["portfolio_additions"])
    scope = (f"Excluding the portfolio added this period" if added == 1
             else f"Excluding the {added} portfolios added this period")
    if movement:
        summary = (f"{scope}, the existing book {direction(movement)} by "
                   f"{money(abs(movement))} ({signed_pct(change_pct)}) to "
                   f"{money(underlying.get('closing_balance'))}.")
    else:
        # "was unchanged by £0" is not a sentence. A flat book is stated flatly.
        summary = (f"{scope}, the existing book was unchanged at "
                   f"{money(underlying.get('closing_balance'))}.")
    if components.get("exits"):
        summary += (f" That is net of {money(abs(components['exits']))} of "
                    f"redemptions and exits.")

    return [Insight(
        insight_type=UNDERLYING_BOOK_MOVEMENT,
        headline=f"Underlying book {direction(movement)} {signed_pct(change_pct)}",
        summary=summary,
        severity=SEVERITY_INFO,
        metrics={
            "opening_balance": opening,
            "closing_balance": underlying.get("closing_balance"),
            "movement": movement,
            "movement_pct": None if change_pct is None else round(change_pct, 2),
            **components,
        },
        methodology={
            "population": ("the source portfolios present in BOTH reporting "
                           "periods, narrowed through the existing governed "
                           "portfolio lens"),
            "portfolio_ids": decomposition.get("continuing_portfolio_ids") or [],
        },
        source_dates={"funded_as_of": underlying.get("currentReportingDate"),
                      "funded_comparison": underlying.get("priorReportingDate")},
        deep_link="/mi/funded",
        **_base(ctx),
    )], []


# --------------------------------------------------------------------------- #
# 4. Mix shift over a governed dimension
# --------------------------------------------------------------------------- #
def mix_shift(ctx: Dict[str, Any], shifts: Optional[List[Dict[str, Any]]]) -> Result:
    """The largest share move per governed dimension, where it is material.

    One generator over every dimension rather than one generator per dimension:
    product, geography, LTV band, borrower age band, borrower structure, vintage
    and source portfolio are the same question asked of different columns, and a
    generator each would be six copies of one rule.
    """
    if not shifts:
        return [], [Omission(FUNDED_MIX_SHIFT,
                             "No comparable prior reporting period is available "
                             "for the portfolio mix.",
                             OMITTED_UNAVAILABLE)]

    t = cfg.thresholds("funded_mix")
    floor = float(t.get("min_share_change_pp", 3.0))

    kept: List[Insight] = []
    below: List[str] = []
    for shift in shifts:
        change_pp = shift.get("share_change_pp")
        if change_pp is None or abs(change_pp) < floor:
            below.append(f"{shift.get('dimension_label') or shift.get('dimension')}")
            continue
        label = shift.get("dimension_label") or shift.get("dimension")
        band = shift.get("category")
        kept.append(Insight(
            insight_type=FUNDED_MIX_SHIFT,
            headline=f"{label}: {band} {direction(change_pp)} {_pp(change_pp)}",
            summary=(
                f"{band} moved from {pct(shift.get('prior_share_pct'))} to "
                f"{pct(shift.get('current_share_pct'))} of the "
                + ("funded book" if shift.get("population") != "underlying"
                   else "existing book, excluding portfolios added this period")
                + f" ({_pp(change_pp)})."),
            severity=SEVERITY_INFO,
            discriminator=f"{shift.get('dimension')}:{band}",
            metrics={
                "dimension": shift.get("dimension"),
                "category": band,
                "current_share_pct": shift.get("current_share_pct"),
                "prior_share_pct": shift.get("prior_share_pct"),
                "share_change_pp": change_pp,
                "current_balance": shift.get("current_balance"),
                "prior_balance": shift.get("prior_balance"),
                "population": shift.get("population", "combined"),
            },
            methodology={"basis": "share of funded balance by governed dimension",
                         "grouping": "the same grouping the funded bridge uses"},
            source_dates=shift.get("source_dates") or {},
            deep_link="/mi/funded",
            **_base(ctx),
        ))

    omissions = []
    if below and not kept:
        omissions.append(Omission(
            FUNDED_MIX_SHIFT,
            f"No governed dimension moved by more than {floor}pp "
            f"({len(below)} checked).",
            OMITTED_IMMATERIAL))
    return kept, omissions


# --------------------------------------------------------------------------- #
# 5. Weighted-average LTV
# --------------------------------------------------------------------------- #
def ltv_movement(ctx: Dict[str, Any],
                 movement: Optional[Dict[str, Any]],
                 underlying: Optional[Dict[str, Any]] = None) -> Result:
    """Balance-weighted LTV movement on the funded book.

    ``underlying`` is the same measure over the portfolios present in BOTH
    periods, supplied only when something was added. Where it exists it LEADS,
    and the combined figure is stated beside it.

    That ordering is the whole point. An arriving book rewrites a weighted
    average by construction: an incumbent book whose LTV rose 30% to 38%
    alongside a low-LTV acquisition produces a combined move of −1.0pp, and a
    reader told only that would conclude credit quality improved in a month it
    deteriorated by eight points. Both numbers are true; only one of them is a
    statement about the book that was already there.
    """
    if not movement or not movement.get("available"):
        return [], [Omission(FUNDED_LTV_MOVEMENT,
                             "No comparable prior reporting period is available.",
                             OMITTED_UNAVAILABLE)]

    t = cfg.thresholds("funded_ltv")
    floor = float(t.get("min_change_pp", 0.5))
    delta = (movement.get("delta") or {}).get("wa_ltv_points")
    current = (movement.get("current") or {}).get("wa_ltv_points")
    prior = (movement.get("prior") or {}).get("wa_ltv_points")

    if delta is None or current is None:
        return [], [Omission(FUNDED_LTV_MOVEMENT,
                             "Weighted-average LTV is unavailable for one of the "
                             "two reporting periods.",
                             OMITTED_UNAVAILABLE)]

    u_delta = (underlying or {}).get("change_pp")
    # Material if EITHER population moved materially. Gating on the combined
    # figure alone is how the masking survives: the arriving book can hold the
    # combined move under the threshold while the incumbent book moves a long
    # way, and the month would then be reported as having no LTV movement.
    if abs(delta) < floor and (u_delta is None or abs(u_delta) < floor):
        return [], [Omission(
            FUNDED_LTV_MOVEMENT,
            f"Weighted-average LTV moved {_pp(delta)}, below the "
            f"{t.get('min_change_pp')}pp materiality threshold.",
            OMITTED_IMMATERIAL)]

    metrics: Dict[str, Any] = {
        "current_wa_ltv_pct": current, "prior_wa_ltv_pct": prior,
        "change_pp": delta, "population": "combined",
    }
    if u_delta is not None:
        metrics.update({
            "underlying_current_wa_ltv_pct": underlying["current"],
            "underlying_prior_wa_ltv_pct": underlying["prior"],
            "underlying_change_pp": u_delta,
        })
        headline = (f"Underlying weighted-average LTV {direction(u_delta)} "
                    f"{_pp(u_delta)}")
        summary = (f"Excluding portfolios added this period, balance-weighted "
                   f"current LTV moved from {pct(underlying['prior'])} to "
                   f"{pct(underlying['current'])} ({_pp(u_delta)}). Including "
                   f"them the combined book moved from {pct(prior)} to "
                   f"{pct(current)} ({_pp(delta)}).")
        # Opposite directions is the case a reader is most likely to misread,
        # so it is said outright rather than left to be inferred from two
        # numbers with different signs.
        if u_delta * delta < 0:
            summary += (" The combined movement is in the opposite direction to "
                        "the underlying book's.")
        severity = _ltv_severity(u_delta, floor)
    else:
        headline = f"Weighted-average LTV {direction(delta)} {_pp(delta)}"
        summary = (f"Balance-weighted current LTV moved from {pct(prior)} to "
                   f"{pct(current)} ({_pp(delta)}).")
        severity = _ltv_severity(delta, floor)

    return [Insight(
        insight_type=FUNDED_LTV_MOVEMENT,
        headline=headline,
        summary=summary,
        severity=severity,
        metrics=metrics,
        methodology={
            "basis": "balance-weighted current LTV, governed funded series",
            "population": ("underlying (portfolios present in both periods), "
                           "with the combined book stated beside it"
                           if u_delta is not None else "combined"),
        },
        source_dates={"funded_as_of": movement.get("currentReportingDate"),
                      "funded_comparison": movement.get("priorReportingDate")},
        deep_link="/mi/funded",
        **_base(ctx),
    )], []


# --------------------------------------------------------------------------- #
# 6. Risk-limit status transitions
# --------------------------------------------------------------------------- #
#: Governed status → the severity a TRANSITION INTO it carries. Not a threshold:
#: the statuses and the transitions are produced by the approved concentration
#: configuration, and this only decides how loudly to report one.
_TRANSITION_SEVERITY = {
    "breach": SEVERITY_CONCERN,
    "warning": SEVERITY_ATTENTION,
    "pass": SEVERITY_INFO,
}


def risk_limit_transitions(ctx: Dict[str, Any],
                           concentration: Optional[Dict[str, Any]]) -> Result:
    """Tests that CROSSED a governed status boundary this period.

    Reads ``statusTransition``, ``status``, ``priorStatus`` and ``deteriorated``
    off the governed evaluation. It defines no limit, no threshold and no status:
    those belong to the operator-approved concentration configuration, and a
    second opinion about them here would be a second risk framework.

    A resolved breach is reported as well as a new one. A review that only ever
    delivers bad news teaches its reader that silence means nothing happened,
    which is the state this whole package is built to avoid.
    """
    if not concentration or not concentration.get("available"):
        return [], [Omission(RISK_LIMIT_TRANSITION,
                             (concentration or {}).get("reason")
                             or "No governed concentration evaluation is available.",
                             OMITTED_UNAVAILABLE)]

    t = cfg.thresholds("risk_limit_transition")
    report_improvements = bool(t.get("report_improvements", True))

    kept: List[Insight] = []
    for test in concentration.get("tests") or []:
        transition = test.get("statusTransition")
        if not transition:
            continue
        status = str(test.get("status") or "")
        prior_status = str(test.get("priorStatus") or "")
        worsened = bool(test.get("deteriorated"))
        if not worsened and not report_improvements:
            continue

        movement = _WORSENING if worsened else _IMPROVING
        severity = (_TRANSITION_SEVERITY.get(status, SEVERITY_INFO) if worsened
                    else SEVERITY_INFO)
        name = test.get("displayName") or test.get("testId")
        utilisation = test.get("utilization")

        summary = (f"{name} {movement} from {prior_status} to {status}"
                   f" ({prior_status} → {status}).")
        if utilisation is not None:
            summary += f" Utilisation is {pct(utilisation * 100)} of the limit."
        if test.get("headroom") is not None:
            summary += f" Headroom {signed_money(test.get('headroom'))}." \
                if str(test.get("unit")) != "percent" \
                else f" Headroom {_pp(test.get('headroom'))}."

        kept.append(Insight(
            insight_type=RISK_LIMIT_TRANSITION,
            headline=f"{name}: {prior_status} → {status}",
            summary=summary,
            severity=severity,
            discriminator=str(test.get("testId") or name),
            metrics={
                "test_id": test.get("testId"),
                "status": status, "prior_status": prior_status,
                "status_transition": transition,
                "deteriorated": worsened,
                "current_value": test.get("currentValue"),
                "prior_value": test.get("priorValue"),
                "threshold": test.get("threshold"),
                "utilisation": utilisation,
                "headroom": test.get("headroom"),
                "breach_amount": test.get("breachAmount"),
                "unit": test.get("unit"),
            },
            methodology={
                "owner": ("the operator-approved concentration configuration; "
                          "no limit, status or threshold is defined here"),
                "lineage": concentration.get("lineage") or {},
            },
            source_dates={"funded_as_of": test.get("reportingDate"),
                          "funded_comparison": test.get("priorReportingDate")},
            deep_link="/mi/risk-limits",
            **_base(ctx),
        ))

    if not kept:
        return [], [Omission(
            RISK_LIMIT_TRANSITION,
            "No governed concentration test changed status this period.",
            OMITTED_IMMATERIAL)]
    return kept, []
