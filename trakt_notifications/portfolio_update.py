#!/usr/bin/env python3
"""Message 1 — Portfolio Update. "What changed following this update?"

Operational information only. Every figure is lifted from a governed MI output
resolved by :mod:`.sources`; this module chooses which of them a five-bullet
card can carry, and writes the sentence around them.

Three rules shape the selection:

* **A missing comparison is stated, never manufactured.** Where the governed
  five-week average is absent — a new deployment with two weeks of history, a
  service that did not resolve — the bullet says so and gives the reason. It is
  never replaced by a different baseline that happens to be available, because
  "9% above average" computed against a different denominator is a wrong
  statement, not a weaker one.

* **A combined update keeps its two dates apart.** Weekly pipeline and monthly
  funded data are approved on different cycles and describe different things.
  The card labels each figure with its own date and never implies a shared
  reporting period.

* **Contributors are quoted, not derived.** "Growth was led by Broker A and the
  South East" comes from the governed PIPELINE_MOVEMENT insight's contributor
  decomposition, which reconciles to the headline change. Nothing here inspects
  a frame to work out who led anything.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mi_agent_api.insight_contract import (
    COMPLETIONS_MOVEMENT, CONVERSION_CONTEXT, PIPELINE_MOVEMENT,
)

from . import deep_links, formatting as fmt
from .contract import (
    MESSAGE_PORTFOLIO_UPDATE, SEVERITY_INFO, MessageItem, NotificationMessage,
    UPDATE_COMBINED, UPDATE_FUNDED, UPDATE_PIPELINE,
)
from .sources import (
    CAP_FUNDED_MOVEMENT, CAP_PIPELINE_EVOLUTION, GovernedInputs,
)

#: Metric keys on the governed ``fiveWeekAverage`` block.
_M_AMOUNT = "pipeline_amount"
_M_COUNT = "pipeline_case_count"
_M_WEIGHTED = "weighted_expected_funded_amount"


def build(inputs: GovernedInputs, *, update_type: str,
          max_items: int = 5) -> NotificationMessage:
    """The Portfolio Update for one approved update."""
    pipeline_as_of = inputs.source_dates.get("pipeline_as_of")
    funded_as_of = inputs.source_dates.get("funded_as_of")

    items: List[MessageItem] = []
    #: Caveats about what the card is showing. Held apart from the observations
    #: because they must survive the item cap: a note that comparisons are
    #: based on two weeks of history is exactly the line a reader needs, and
    #: exactly the line a naive cap would drop first.
    disclosures: List[MessageItem] = []
    insight_ids: List[str] = []
    leads: List[str] = []

    if update_type in (UPDATE_PIPELINE, UPDATE_COMBINED):
        lead = _pipeline_lead(inputs, update_type, pipeline_as_of)
        if lead:
            leads.append(lead)
        items.extend(_pipeline_items(inputs, insight_ids))
        reason = _unavailable_comparison_reason(inputs)
        if reason:
            disclosures.append(MessageItem(
                text=reason, metric="five_week_comparison", unavailable=True))
    if update_type in (UPDATE_FUNDED, UPDATE_COMBINED):
        lead = _funded_lead(inputs, update_type, funded_as_of)
        if lead:
            leads.append(lead)
        funded, funded_disclosures = _funded_items(inputs, insight_ids)
        items.extend(funded)
        disclosures.extend(funded_disclosures)

    headline = _headline(inputs, update_type, pipeline_as_of, funded_as_of)

    # The cap governs OBSERVATIONS, not disclosures. Items are already ordered
    # most-informative-first by the sequence they were appended in.
    budget = max(max_items - len(disclosures), 1)
    kept = items[:budget]
    if len(items) > len(kept):
        # Never silently truncate — a card that dropped two observations must
        # not read as though it showed everything.
        kept = kept[:max(budget - 1, 0)] + [MessageItem(
            text=(f"{len(items) - max(budget - 1, 0)} further observations are "
                  f"in Trakt."), metric="truncated")]
    kept = kept + disclosures

    return NotificationMessage(
        message_type=MESSAGE_PORTFOLIO_UPDATE,
        headline=headline,
        summary=" ".join(leads) or None,
        # An operational update is information, never a grading. Severity on
        # this message is fixed at info by design: raising it would make the
        # Risk Review's severity meaningless.
        severity=SEVERITY_INFO,
        items=kept,
        deep_link=deep_links.for_update(
            update_type, portfolio_context=inputs.portfolio_context,
            pipeline_as_of=pipeline_as_of, funded_as_of=funded_as_of),
        insight_ids=insight_ids,
        provenance=_provenance(inputs, update_type, pipeline_as_of, funded_as_of),
    )


# --------------------------------------------------------------------------- #
# Headline
# --------------------------------------------------------------------------- #
def _headline(inputs: GovernedInputs, update_type: str,
              pipeline_as_of: Optional[str],
              funded_as_of: Optional[str]) -> str:
    if update_type == UPDATE_FUNDED:
        return f"Monthly Funded Update — {fmt.human_date(funded_as_of)}"
    if update_type == UPDATE_COMBINED:
        # Both dates in the headline, because the single most likely
        # misreading of a combined card is that one date governs both figures.
        return (f"Portfolio Update — pipeline {fmt.human_date(pipeline_as_of)}, "
                f"funded {fmt.human_date(funded_as_of)}")
    return f"Weekly Pipeline Update — {fmt.human_date(pipeline_as_of)}"


# --------------------------------------------------------------------------- #
# Pipeline content
# --------------------------------------------------------------------------- #
def _five_week(inputs: GovernedInputs) -> Dict[str, Any]:
    return ((inputs.pipeline_evolution or {}).get("fiveWeekAverage") or {})


def _comparison_clause(inputs: GovernedInputs, metric: str) -> Optional[str]:
    """The governed five-week comparison for one metric, or ``None``.

    ``None`` means the caller must omit the clause with a reason; it never
    means "no difference", which :func:`fmt.against_baseline` says explicitly.
    """
    block = _five_week(inputs)
    if not block.get("available"):
        return None
    difference = (block.get("differencePct") or {}).get(metric)
    return fmt.against_baseline(difference)


def _unavailable_comparison_reason(inputs: GovernedInputs) -> Optional[str]:
    """Why the pipeline five-week comparison is missing or weak.

    Names the PIPELINE comparison specifically. The completions comparison
    comes from the origination funnel, which is a separate governed source and
    can still be available when this one is not — a blanket "five-week
    comparisons are unavailable" alongside a completions comparison would
    contradict itself on the same card.
    """
    if inputs.missing(CAP_PIPELINE_EVOLUTION):
        return ("Pipeline five-week comparisons are unavailable: the governed "
                "weekly series could not be resolved for this update.")
    block = _five_week(inputs)
    if not block.get("available"):
        return ("Pipeline five-week comparisons are unavailable: no prior "
                "weekly history has been observed for this portfolio yet.")
    weeks = int(block.get("weeksObserved") or 0)
    if weeks < int(block.get("window") or 5):
        return (f"Pipeline five-week comparisons are based on {weeks} observed "
                f"week{'s' if weeks != 1 else ''} so far.")
    return None


def _pipeline_lead(inputs: GovernedInputs, update_type: str,
                   pipeline_as_of: Optional[str]) -> Optional[str]:
    """``Pipeline contains 812 cases totalling £503.4m.``

    The current position, stated once, so the bullets below it are free to be
    comparisons rather than repeating the levels they compare against.
    """
    current = (_five_week(inputs).get("current") or {})
    cases = current.get(_M_COUNT)
    amount = current.get(_M_AMOUNT)
    if cases is None and amount is None:
        return None
    parts = []
    if cases is not None:
        parts.append(f"{fmt.count(cases)} cases")
    if amount is not None:
        parts.append(f"totalling {fmt.money(amount)}")
    lead = f"Pipeline contains {' '.join(parts)}"
    if update_type == UPDATE_COMBINED:
        # A combined card must never let one date be read as governing both
        # datasets, so the pipeline sentence carries its own.
        lead += f" as at {fmt.human_date(pipeline_as_of)}"
    return lead + "."


def _pipeline_items(inputs: GovernedInputs,
                    insight_ids: List[str]) -> List[MessageItem]:
    """The pipeline bullets, most informative first.

    Ordered so that if the card's item cap bites, what survives is the
    comparisons and the attribution — the things a reader cannot get from the
    lead sentence — rather than a restatement of levels.
    """
    items: List[MessageItem] = []
    block = _five_week(inputs)
    current = block.get("current") or {}

    # 1. Case count against the governed five-week average.
    cases = current.get(_M_COUNT)
    clause = _comparison_clause(inputs, _M_COUNT)
    if cases is not None and clause:
        items.append(MessageItem(text=f"Case count is {clause}.",
                                 metric=_M_COUNT, raw=cases))

    # 2. Completions — value AND count, from the governed funnel's weekly flow.
    completions = _completions_item(inputs, insight_ids)
    if completions is not None:
        items.append(completions)

    # 3. Weighted expected funding — the forecast contribution of the pipeline.
    weighted = current.get(_M_WEIGHTED)
    if weighted is not None:
        items.append(MessageItem(
            text=f"Weighted expected funding is {fmt.money(weighted)}.",
            metric=_M_WEIGHTED, raw=weighted))

    # 4. Material movement and who led it — quoted from the governed insight.
    movement = _movement_item(inputs, insight_ids)
    if movement is not None:
        items.append(movement)

    # 5. Pipeline value against the average, where the movement insight did not
    # already make the point.
    amount = current.get(_M_AMOUNT)
    amount_clause = _comparison_clause(inputs, _M_AMOUNT)
    if amount is not None and amount_clause and movement is None:
        items.append(MessageItem(text=f"Pipeline value is {amount_clause}.",
                                 metric=_M_AMOUNT, raw=amount))

    # 6. Conversion, only where the governed evidence is sufficient.
    conversion = _conversion_item(inputs, insight_ids)
    if conversion is not None:
        items.append(conversion)

    return items


def _completions_item(inputs: GovernedInputs,
                      insight_ids: List[str]) -> Optional[MessageItem]:
    """Completions this week against the governed five-week average FLOW.

    Uses the funnel's ``fiveWeekAvgFlow*`` — the trailing mean of the weekly
    flow — rather than the stock average, because a completions figure is a
    flow. Comparing a weekly flow against an average stock level was a real
    defect in this codebase's history and the funnel's own docstring records it.
    """
    summary = ((inputs.funnel or {}).get("summary") or {}).get("COMPLETED") or {}
    value = summary.get("latestFlowValue")
    count = summary.get("latestFlowCount")
    if value is None and count is None:
        return None

    parts = []
    if count is not None:
        parts.append(f"{fmt.count(count)} cases")
    if value is not None:
        parts.append(fmt.money(value))
    text = f"Completions totalled {' and '.join(parts)}"

    avg_value = summary.get("fiveWeekAvgFlowValue")
    if value is not None and avg_value:
        difference = (value - avg_value) / abs(avg_value) * 100.0
        clause = fmt.against_baseline(round(difference, 1))
        if clause:
            text += f", {clause}"
    text += "."

    insight = inputs.insight(COMPLETIONS_MOVEMENT)
    if insight and insight.get("insight_id"):
        insight_ids.append(str(insight["insight_id"]))
    return MessageItem(text=text, metric="completions_flow", raw=value)


def _movement_item(inputs: GovernedInputs,
                   insight_ids: List[str]) -> Optional[MessageItem]:
    """Material pipeline movement with its largest broker and regional contributor.

    Emitted only when the Insight Engine judged the movement material against
    the configured threshold. Below it there is no movement insight, and this
    module does not get to disagree.
    """
    insight = inputs.insight(PIPELINE_MOVEMENT)
    if not insight:
        return None
    insight_ids.append(str(insight.get("insight_id")))

    contributors = insight.get("contributors") or {}
    brokers = contributors.get("brokers") or []
    regions = contributors.get("regions") or []
    metrics = insight.get("metrics") or {}
    change = metrics.get("change")

    lead_broker = (brokers[0].get("name") if brokers else None)
    lead_region = (regions[0].get("name") if regions else None)

    direction_word = "Growth" if (change or 0) >= 0 else "The reduction"
    named = [n for n in (lead_broker, lead_region) if n]
    if named:
        text = (f"{direction_word} of {fmt.money(abs(change or 0))} was led by "
                f"{' and '.join(named)}.")
    else:
        text = (f"Pipeline moved {fmt.signed_money(change)} "
                f"({fmt.signed_pct(metrics.get('change_pct'))}).")
    return MessageItem(text=text, metric="pipeline_movement", raw=change,
                       insight_id=str(insight.get("insight_id")))


def _conversion_item(inputs: GovernedInputs,
                     insight_ids: List[str]) -> Optional[MessageItem]:
    """Conversion context, only where the governed window is sufficient.

    The funnel flags a rate computed over too few weeks as insufficient. A
    Portfolio Update is not the place to publish a provisional rate — the Risk
    Review carries that as a limitation instead.
    """
    summary = ((inputs.funnel or {}).get("summary") or {}).get("COMPLETED") or {}
    conversion = summary.get("conversion") or {}
    if not conversion.get("sufficient"):
        return None
    rate = conversion.get("weeklyRateValue")
    if rate is None:
        return None
    insight = inputs.insight(CONVERSION_CONTEXT)
    if insight and insight.get("insight_id"):
        insight_ids.append(str(insight["insight_id"]))
    return MessageItem(
        text=(f"Weekly conversion to completion is {fmt.pct(rate)} by value "
              f"over the governed {conversion.get('weeksInWindow')}-week window."),
        metric="weekly_conversion_value_pct", raw=rate)


# --------------------------------------------------------------------------- #
# Funded content
# --------------------------------------------------------------------------- #
def _funded_lead(inputs: GovernedInputs, update_type: str,
                 funded_as_of: Optional[str]) -> Optional[str]:
    """``Funded balance is £1.42bn as at 31 July, +£24.1m on the month.``"""
    movement = inputs.funded_movement or {}
    if not movement.get("available"):
        return None
    balance = (movement.get("current") or {}).get("funded_balance")
    if balance is None:
        return None
    lead = f"Funded balance is {fmt.money(balance)}"
    if update_type == UPDATE_COMBINED:
        lead += f" as at {fmt.human_date(funded_as_of)}"
    delta = (movement.get("delta") or {}).get("funded_balance")
    if delta is not None:
        lead += f", {fmt.signed_money(delta)} on the month"
    return lead + "."


def _funded_items(inputs: GovernedInputs, insight_ids: List[str]
                  ) -> "tuple[List[MessageItem], List[MessageItem]]":
    """``(observations, disclosures)`` — disclosures survive the item cap."""
    movement = inputs.funded_movement or {}
    if not movement.get("available"):
        reason = inputs.unavailable.get(CAP_FUNDED_MOVEMENT) or movement.get("reason")
        return [], [MessageItem(
            text=("Funded movement is unavailable for this update"
                  + (f": {reason}." if reason else ".")),
            metric="funded_movement", unavailable=True)]

    current = movement.get("current") or {}
    delta = movement.get("delta") or {}
    items: List[MessageItem] = []

    loans = current.get("loan_count")
    loans_delta = delta.get("loan_count")
    if loans is not None:
        text = f"Loan count is {fmt.count(loans)}"
        if loans_delta is not None:
            text += f" ({int(loans_delta):+,} on the month)"
        text += "."
        items.append(MessageItem(text=text, metric="loan_count", raw=loans))

    # Direct / acquired contribution — the governed cohort decomposition,
    # which reconciles to the consolidated movement.
    cohorts = [c for c in (movement.get("cohortMovements") or [])
               if c.get("delta")]
    if cohorts:
        lead = cohorts[0]
        items.append(MessageItem(
            text=(f"Largest book contribution: {lead.get('label')} at "
                  f"{fmt.signed_money(lead.get('delta'))}."),
            metric="cohort_contribution", raw=lead.get("delta")))

    ltv_delta = delta.get("wa_ltv_points")
    if ltv_delta is not None:
        wa = current.get("wa_ltv_points")
        items.append(MessageItem(
            text=(f"Weighted-average LTV is {fmt.pct(wa)}, "
                  f"{fmt.pp(ltv_delta)} on the month."),
            metric="wa_ltv_points", raw=wa))

    primary = movement.get("primaryRegion")
    if primary and primary.get("region"):
        items.append(MessageItem(
            text=(f"Largest regional contribution: {primary['region']} at "
                  f"{fmt.signed_money(primary.get('delta'))}."),
            metric="region_contribution", raw=primary.get("delta")))
    return items, []


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _provenance(inputs: GovernedInputs, update_type: str,
                pipeline_as_of: Optional[str],
                funded_as_of: Optional[str]) -> Dict[str, Any]:
    """Compact, non-identifying provenance for the card footer.

    Dates and methodology versions only. Run ids and storage URIs are
    deliberately absent: they are internal identifiers, and a card is a
    user-facing surface.
    """
    out: Dict[str, Any] = {"update_type": update_type}
    if update_type in (UPDATE_PIPELINE, UPDATE_COMBINED) and pipeline_as_of:
        out["pipeline_as_of"] = pipeline_as_of
    if update_type in (UPDATE_FUNDED, UPDATE_COMBINED) and funded_as_of:
        out["funded_as_of"] = funded_as_of
    block = _five_week(inputs)
    if block.get("available"):
        out["five_week_basis"] = block.get("basis")
        out["five_week_weeks_observed"] = block.get("weeksObserved")
    return out
