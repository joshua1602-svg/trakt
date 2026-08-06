#!/usr/bin/env python3
"""Deep links from a card into the corresponding React MI view.

A deep link carries *context*, never *authority*. It names a portfolio context,
a reporting date, a tab and — where one applies — the governed test or insight
identifier, so the view opens on the thing the card was about instead of blank.
The tenant is NOT in the link: React and the API resolve it from the caller's
authenticated session and revalidate the user on every request. Someone who
obtains a link and is not entitled to that portfolio gets a refusal from the
API, not a view.

Targets mirror the ``deep_link`` values the Insight Engine already stamps on its
insights (``/mi/pipeline``, ``/mi/risk-limits``), so a card and the brief send a
reader to the same place.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from urllib.parse import urlencode

#: Base URL of the React MI Workspace. Reuses the variable the Copilot
#: Workspace promotion already uses, so one deployment setting drives both and
#: they cannot point at different environments.
BASE_URL_ENV = "TRAKT_COPILOT_WORKSPACE_BASE_URL"

# Tabs, matching the existing React view identifiers.
TAB_PIPELINE = "pipeline"
TAB_FUNDED = "funded"
TAB_FORECAST = "forecast"
TAB_RISK_LIMITS = "risk-limits"

#: Governed route per target. Kept in one table so a renamed React route is a
#: one-line change and no card can be left pointing at a route that moved.
_ROUTES = {
    TAB_PIPELINE: "/mi/pipeline",
    TAB_FUNDED: "/mi/funded",
    TAB_FORECAST: "/mi/forecast",
    TAB_RISK_LIMITS: "/mi/risk-limits",
}


def base_url() -> Optional[str]:
    value = (os.environ.get(BASE_URL_ENV) or "").strip()
    return value.rstrip("/") or None


def build(tab: str, *, portfolio_context: Optional[str] = None,
          as_of: Optional[str] = None, test_id: Optional[str] = None,
          insight_id: Optional[str] = None,
          evidence: Optional[str] = None) -> Optional[str]:
    """A deep link, or ``None`` when no workspace base URL is configured.

    ``None`` rather than a fabricated relative URL: a card with no button is
    honest about there being nowhere to send the reader, whereas a link that
    404s reads as a broken product.
    """
    root = base_url()
    if root is None:
        return None
    path = _ROUTES.get(tab, _ROUTES[TAB_PIPELINE])
    params: Dict[str, Any] = {}
    if portfolio_context:
        params["portfolioContext"] = portfolio_context
    if as_of:
        params["asOf"] = as_of
    if test_id:
        params["testId"] = test_id
    if insight_id:
        params["insightId"] = insight_id
    if evidence:
        params["evidence"] = evidence
    # Sorted so the same context always produces a byte-identical link — a
    # notification payload has to be reproducible to be auditable.
    query = urlencode(sorted(params.items()))
    return f"{root}{path}?{query}" if query else f"{root}{path}"


def for_update(update_type: str, *, portfolio_context: Optional[str],
               pipeline_as_of: Optional[str],
               funded_as_of: Optional[str]) -> Optional[str]:
    """Where a Portfolio Update should land.

    A pipeline update opens Origination/Pipeline; a funded update opens Funded;
    a combined update opens Forecast, the one view that legitimately shows both
    datasets side by side without implying they share a reporting date.
    """
    from .contract import UPDATE_COMBINED, UPDATE_FUNDED

    if update_type == UPDATE_FUNDED:
        return build(TAB_FUNDED, portfolio_context=portfolio_context,
                     as_of=funded_as_of)
    if update_type == UPDATE_COMBINED:
        return build(TAB_FORECAST, portfolio_context=portfolio_context,
                     as_of=funded_as_of or pipeline_as_of)
    return build(TAB_PIPELINE, portfolio_context=portfolio_context,
                 as_of=pipeline_as_of)


#: Which view answers which governed risk category. A concentration finding
#: belongs on Risk Limits; a data-quality finding belongs with its evidence.
_RISK_TAB = {
    "current_breach": TAB_RISK_LIMITS,
    "expected_breach": TAB_RISK_LIMITS,
    "low_expected_headroom": TAB_RISK_LIMITS,
    "stress_only_breach": TAB_RISK_LIMITS,
    "deterioration": TAB_RISK_LIMITS,
    "limitation": TAB_RISK_LIMITS,
    "conversion": TAB_PIPELINE,
    "forecast_reduction": TAB_FORECAST,
    "ltv_mix": TAB_PIPELINE,
    "data_quality": TAB_PIPELINE,
}


def for_risk(category: Optional[str], *, portfolio_context: Optional[str],
             as_of: Optional[str], test_id: Optional[str] = None,
             insight_id: Optional[str] = None) -> Optional[str]:
    """Where a Risk Review should land, by governed risk category."""
    tab = _RISK_TAB.get(str(category or ""), TAB_RISK_LIMITS)
    evidence = "data-quality" if category == "data_quality" else None
    return build(tab, portfolio_context=portfolio_context, as_of=as_of,
                 test_id=test_id, insight_id=insight_id, evidence=evidence)


def label_for(tab: str) -> str:
    return {
        TAB_RISK_LIMITS: "Open concentration analysis",
        TAB_PIPELINE: "Open Trakt",
        TAB_FUNDED: "Open Trakt",
        TAB_FORECAST: "Open Trakt",
    }.get(tab, "Open Trakt")
