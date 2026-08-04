#!/usr/bin/env python3
"""Message 2 — Risk Review. "What risks or emerging constraints should the user
consider?"

Produced for every approved update, including the ones with nothing to report:
a silent week is indistinguishable from a broken pipeline, and a reader who
only hears from the system when something is wrong learns to distrust silence.

The ranking is not this module's. ``identify_emerging_risks`` in the governed
concentration package already orders findings by a fixed rank — current breach,
expected breach, low expected headroom, deterioration, stress-only, limitation
— and that order is taken as given. Insight-derived findings (conversion, LTV
mix, data quality) are merged in at the position their governed severity earns
them, using the Insight Engine's own severity vocabulary. Nothing here re-grades
anything.

The one judgement this module does make is the distinction the product depends
on:

    "No material portfolio risks were identified from this update."
    "No material risks were identified from the checks that were available."

The first is a claim about the portfolio. The second is a claim about the
checks. Saying the first when a check did not run would be a false clear state,
so a clear message is only ever issued when nothing was unavailable, and the
unavailable checks are named on the card when they were.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from mi_agent.concentration_tests.forward import (
    RISK_CURRENT_BREACH, RISK_DETERIORATION, RISK_EXPECTED_BREACH,
    RISK_LIMITATION, RISK_LOW_EXPECTED_HEADROOM, RISK_STRESS_ONLY,
)
from mi_agent_api.insight_contract import (
    CONVERSION_CONTEXT, DATA_QUALITY, LTV_MIX_SHIFT, SEVERITY_ATTENTION,
    SEVERITY_CONCERN, TICKET_MIX_SHIFT, WEIGHTED_LTV,
)

from . import deep_links, formatting as fmt, recommendation
from .config import recommendation_level
from .contract import (
    MESSAGE_RISK_REVIEW, SEVERITY_ATTENTION as MSG_ATTENTION,
    SEVERITY_CLEAR, SEVERITY_CONCERN as MSG_CONCERN, MessageItem,
    NotificationMessage, max_severity,
)
from .sources import GovernedInputs, unavailable_summary

#: Governed risk categories a Risk Review may carry, and the severity each
#: maps to. Taken from the governed rank order; never re-derived from values.
_CATEGORY_SEVERITY = {
    RISK_CURRENT_BREACH: MSG_CONCERN,
    RISK_EXPECTED_BREACH: MSG_CONCERN,
    RISK_LOW_EXPECTED_HEADROOM: MSG_ATTENTION,
    RISK_DETERIORATION: MSG_ATTENTION,
    RISK_STRESS_ONLY: MSG_ATTENTION,
    RISK_LIMITATION: MSG_ATTENTION,
}

#: Deep-link category per governed risk category.
_LINK_CATEGORY = {
    RISK_CURRENT_BREACH: "current_breach",
    RISK_EXPECTED_BREACH: "expected_breach",
    RISK_LOW_EXPECTED_HEADROOM: "low_expected_headroom",
    RISK_DETERIORATION: "deterioration",
    RISK_STRESS_ONLY: "stress_only_breach",
    RISK_LIMITATION: "limitation",
}

#: Insight types that constitute a risk finding rather than an observation, and
#: the rank they slot in at. Ranks sit BELOW the concentration categories
#: (which run 1-6) except for data quality, which governs whether the rest of
#: the update can be believed at all and therefore outranks everything but a
#: live breach.
_INSIGHT_RISKS: Dict[str, Tuple[int, str]] = {
    DATA_QUALITY: (2, "data_quality"),
    CONVERSION_CONTEXT: (7, "conversion"),
    WEIGHTED_LTV: (8, "ltv_mix"),
    LTV_MIX_SHIFT: (9, "ltv_mix"),
    TICKET_MIX_SHIFT: (10, "ltv_mix"),
}

CLEAR_HEADLINE = "Risk Review"
CLEAR_STATEMENT = "No material portfolio risks were identified from this update."
PARTIAL_STATEMENT = ("No material risks were identified from the checks that "
                     "were available.")


def build(inputs: GovernedInputs, *, max_items: int = 3,
          action_rules: Optional[Dict[str, Dict[str, Any]]] = None
          ) -> NotificationMessage:
    """The Risk Review for one approved update."""
    findings = _collect(inputs)
    unavailable = unavailable_summary(inputs)

    if not findings:
        return _clear_message(inputs, unavailable)

    kept = findings[:max_items]
    items: List[MessageItem] = []
    insight_ids: List[str] = []
    levels: List[str] = []

    for finding in kept:
        item, level = _render(inputs, finding, action_rules or {})
        items.append(item)
        if finding.get("insight_id"):
            insight_ids.append(str(finding["insight_id"]))
        if level:
            levels.append(level)

    if len(findings) > len(kept):
        items.append(MessageItem(
            text=(f"{len(findings) - len(kept)} further risk item"
                  f"{'s are' if len(findings) - len(kept) > 1 else ' is'} "
                  f"shown in Trakt."),
            metric="truncated"))

    if unavailable:
        items.append(_unavailable_item(unavailable))

    severity = max_severity([f.get("severity") for f in kept])
    lead = kept[0]
    return NotificationMessage(
        message_type=MESSAGE_RISK_REVIEW,
        headline=_headline(lead),
        severity=severity,
        items=items,
        deep_link=deep_links.for_risk(
            lead.get("link_category"),
            portfolio_context=inputs.portfolio_context,
            as_of=(inputs.source_dates.get("funded_as_of")
                   or inputs.source_dates.get("pipeline_as_of")),
            test_id=lead.get("test_id")),
        deep_link_label=(deep_links.label_for(deep_links.TAB_RISK_LIMITS)
                         if lead.get("test_id") else "Open Trakt"),
        insight_ids=insight_ids,
        unavailable_checks=unavailable,
        recommendation_level=(_highest_level(levels) if levels else None),
        provenance=_provenance(inputs),
    )


# --------------------------------------------------------------------------- #
# Collection — governed findings, in the governed order
# --------------------------------------------------------------------------- #
def _collect(inputs: GovernedInputs) -> List[Dict[str, Any]]:
    """Every governed risk finding, ordered by the governed rank.

    Concentration findings arrive already ranked from
    ``identify_emerging_risks``; insight findings are slotted in at their
    declared rank. Ties break on the finding's name so two runs over the same
    week always order identically.
    """
    findings: List[Dict[str, Any]] = []

    for risk in ((inputs.concentration or {}).get("emergingRisks") or []):
        category = str(risk.get("category") or "")
        findings.append({
            "rank": int(risk.get("rank") or 99),
            "category": category,
            "link_category": _LINK_CATEGORY.get(category, "current_breach"),
            "severity": _CATEGORY_SEVERITY.get(category, MSG_ATTENTION),
            "name": risk.get("displayName") or "Concentration test",
            "statement": risk.get("statement") or "",
            "test_id": risk.get("testId"),
        })

    for insight in inputs.insights():
        insight_type = str(insight.get("insight_type") or "")
        if insight_type not in _INSIGHT_RISKS:
            continue
        severity = str(insight.get("severity") or "")
        # Only insights the Insight Engine itself graded above informational
        # are risk findings. An observation stays an observation.
        if severity not in (SEVERITY_ATTENTION, SEVERITY_CONCERN):
            continue
        rank, link_category = _INSIGHT_RISKS[insight_type]
        findings.append({
            "rank": rank,
            "category": insight_type,
            "link_category": link_category,
            "severity": severity,
            "name": insight.get("headline") or insight_type,
            "statement": insight.get("summary") or "",
            "insight_id": insight.get("insight_id"),
            "test_id": None,
        })

    findings.sort(key=lambda f: (f["rank"], str(f.get("name") or "")))
    return findings


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _headline(lead: Dict[str, Any]) -> str:
    """``Risk Review — South West concentration``.

    Names the leading finding rather than counting findings, because a reader
    scanning a notification list needs to know what it is about before opening
    it.
    """
    name = str(lead.get("name") or "").strip()
    if not name:
        return CLEAR_HEADLINE
    # Insight headlines are already sentences; concentration test names are not.
    if lead.get("test_id"):
        return f"{CLEAR_HEADLINE} — {name}"
    return f"{CLEAR_HEADLINE} — {name[0].lower() + name[1:]}"


def _render(inputs: GovernedInputs, finding: Dict[str, Any],
            action_rules: Dict[str, Dict[str, Any]]
            ) -> Tuple[MessageItem, Optional[str]]:
    """One risk item: the governed statement, then evidence, then at most one
    controlled recommendation sentence."""
    sentences = [finding.get("statement") or ""]
    level: Optional[str] = None

    test_id = finding.get("test_id")
    contributors = inputs.contributors.get(str(test_id)) if test_id else None
    if contributors:
        resolved = recommendation.resolve(
            contributors, level=recommendation_level(),
            action_rule=action_rules.get(str(test_id)))
        sentences.extend(recommendation.lines(resolved))
        level = resolved.get("level")

    text = " ".join(s.strip() for s in sentences if s and s.strip())
    return MessageItem(text=text, metric=finding.get("category"),
                       insight_id=finding.get("insight_id")), level


def _unavailable_item(unavailable: List[str]) -> MessageItem:
    return MessageItem(
        text=("Checks unavailable for this update: "
              + ", ".join(unavailable) + "."),
        metric="unavailable_checks", unavailable=True)


def _clear_message(inputs: GovernedInputs,
                   unavailable: List[str]) -> NotificationMessage:
    """The no-material-risk message.

    The statement is weakened when any check was unavailable, and the
    unavailable checks are listed, so the reader can see the difference between
    "we looked and found nothing" and "we could not look everywhere".
    """
    partial = bool(unavailable)
    items = [MessageItem(
        text=(PARTIAL_STATEMENT if partial else CLEAR_STATEMENT),
        metric="clear_state")]

    # The specific reassurances a reader wants, but only for the checks that
    # actually ran. A line claiming "no concentration breaches" when the
    # concentration service was down would be the exact false clear state this
    # message must never produce.
    concentration = inputs.concentration or {}
    if concentration.get("available"):
        items.append(MessageItem(
            text="No current concentration breaches.", metric="concentration"))
        if (concentration.get("states") or {}).get("available"):
            items.append(MessageItem(
                text="No forecast breaches within the governed horizon.",
                metric="forecast_concentration"))
    if inputs.brief and inputs.brief.get("status") in ("success", "partial"):
        items.append(MessageItem(
            text=("No material deterioration in conversion, LTV mix or data "
                  "quality."), metric="insight_checks"))
    if partial:
        items.append(_unavailable_item(unavailable))

    return NotificationMessage(
        message_type=MESSAGE_RISK_REVIEW,
        headline=CLEAR_HEADLINE,
        severity=SEVERITY_CLEAR,
        items=items,
        deep_link=deep_links.for_risk(
            "current_breach", portfolio_context=inputs.portfolio_context,
            as_of=(inputs.source_dates.get("funded_as_of")
                   or inputs.source_dates.get("pipeline_as_of"))),
        unavailable_checks=unavailable,
        provenance=_provenance(inputs),
    )


def _highest_level(levels: List[str]) -> str:
    from .config import LEVEL_RANK
    return max(levels, key=lambda level: LEVEL_RANK.get(level, 0))


def _provenance(inputs: GovernedInputs) -> Dict[str, Any]:
    """Compact provenance. Dates and methodology only — never a run id."""
    concentration = inputs.concentration or {}
    forecast = concentration.get("forecast") or {}
    out: Dict[str, Any] = {}
    if concentration.get("reportingDate"):
        out["funded_as_of"] = concentration["reportingDate"]
    if inputs.source_dates.get("pipeline_as_of"):
        out["pipeline_as_of"] = inputs.source_dates["pipeline_as_of"]
    if forecast.get("methodology"):
        out["forecast_methodology"] = forecast["methodology"]
    out["risk_ranking"] = ("governed emerging-risk rank order; not reordered "
                           "for delivery")
    return out
