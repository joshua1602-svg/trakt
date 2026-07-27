"""copilot_workspace.py — intelligent Workspace promotion for Copilot answers.

A PURE, deterministic classifier over the MI envelope the shared service already
produced. It owns NO analytical behaviour: it neither re-parses, re-computes nor
re-ranks anything — it only reads signals the engine already emitted and decides
whether pointing the user at the richer React MI Workspace would genuinely help.

Classes
-------
``fully-served-simple``   The Copilot text answer fully satisfies the request
                          (single lookup). NO Workspace link.
``fully-served-complex``  Copilot answered completely, but the interactive
                          Workspace (chart / drill-through / filtering) offers a
                          materially richer view. Workspace link included.
``unmet``                 Copilot could not fully serve it (failed, a chart was
                          asked for but only a table could be returned, or the
                          result was truncated). Workspace link included as the
                          primary next step.

Only ``fully-served-complex`` and ``unmet`` carry a Workspace link. The link is
built from ``TRAKT_COPILOT_WORKSPACE_BASE_URL`` plus enough state (client, run,
question, filters) for the Workspace to RESTORE context rather than open blank.
When that env var is unset the classification still runs; the link is simply
suppressed (a pure server-side deploy toggle).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

CLASS_SIMPLE = "fully-served-simple"
CLASS_COMPLEX = "fully-served-complex"
CLASS_UNMET = "unmet"

#: A supporting table with at least this many total rows is a "large result" —
#: big enough that the interactive Workspace beats a capped text/table dump.
LARGE_RESULT_ROWS = 15

#: Question wording that asks for a picture Copilot cannot draw (it renders text
#: and tables only). Used ONLY to detect a chart request that went unmet.
_VISUAL_RE = re.compile(
    r"\b(chart|charts|graph|graphs|plot|plots|visuali[sz]e|visuali[sz]ation|"
    r"trend|trend\s?line|over\s+time|distribution|histogram|breakdown|"
    r"heat\s?map|scatter|bar\s+chart|line\s+chart|pie\s+chart)\b",
    re.IGNORECASE)

#: Engine intents that are inherently exploratory rather than a single lookup.
_EXPLORATORY_INTENTS = {
    "comparison", "compare", "evolution", "cohort", "cohorts",
    "forecast", "temporal", "risk_limits",
}


def _chart_produced(artifacts: List[Dict[str, Any]]) -> bool:
    return any((a or {}).get("type") == "chart" for a in artifacts or [])


def _multi_dimension(spec: Dict[str, Any]) -> bool:
    dims = spec.get("dimensions") or []
    hierarchy = spec.get("hierarchy") or []
    return len(dims) >= 2 or len(hierarchy) >= 2


def _multi_metric(spec: Dict[str, Any]) -> bool:
    metrics = spec.get("metrics")
    return isinstance(metrics, list) and len(metrics) >= 2


def _exploratory(spec: Dict[str, Any]) -> bool:
    intent = str(spec.get("intent") or "").strip().lower()
    if intent in _EXPLORATORY_INTENTS:
        return True
    # A ranked top-N is an exploratory drill candidate.
    return bool(spec.get("top_n"))


def _filtered(spec: Dict[str, Any]) -> bool:
    # Copilot has no structured filter/lens input; a filtered result is one the
    # user can only refine further in the Workspace.
    return bool(spec.get("filters"))


def classify(*, ok: bool, question: str, spec: Optional[Dict[str, Any]],
             artifacts: Optional[List[Dict[str, Any]]],
             truncated: bool, max_total_rows: int) -> str:
    """Classify one answer. See module docstring for the (confirmed) heuristic."""
    spec = spec or {}
    artifacts = artifacts or []
    chart_produced = _chart_produced(artifacts)

    # 1. UNMET — Copilot could not fully serve the request.
    if not ok:
        return CLASS_UNMET
    if _VISUAL_RE.search(question or "") and not chart_produced:
        return CLASS_UNMET
    if truncated:
        return CLASS_UNMET

    # 2. FULLY-SERVED-COMPLEX — answered, but the Workspace is materially richer.
    if (chart_produced
            or max_total_rows >= LARGE_RESULT_ROWS
            or _multi_dimension(spec)
            or _multi_metric(spec)
            or _exploratory(spec)
            or _filtered(spec)):
        return CLASS_COMPLEX

    # 3. FULLY-SERVED-SIMPLE — a plain lookup the text answer fully satisfies.
    return CLASS_SIMPLE


def build_workspace_url(*, client: Optional[str], run: Optional[str],
                        question: Optional[str],
                        filters: Optional[Dict[str, Any]]) -> Optional[str]:
    """A deep link that RESTORES Workspace context, or ``None`` when
    ``TRAKT_COPILOT_WORKSPACE_BASE_URL`` is unset."""
    base = (os.environ.get("TRAKT_COPILOT_WORKSPACE_BASE_URL") or "").strip()
    if not base:
        return None
    params: List[Tuple[str, str]] = []
    if client:
        params.append(("client", str(client)))
    if run:
        params.append(("run", str(run)))
    if question:
        params.append(("q", str(question)))
    if filters:
        params.append(("filters", json.dumps(filters, separators=(",", ":"),
                                              sort_keys=True)))
    if not params:
        return base
    sep = "&" if "?" in base else "?"
    return f"{base.rstrip('/') if sep == '?' else base}{sep}{urlencode(params)}"


def promote(*, ok: bool, question: str, spec: Optional[Dict[str, Any]],
            artifacts: Optional[List[Dict[str, Any]]], truncated: bool,
            max_total_rows: int, client: Optional[str], run: Optional[str],
            filters: Optional[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    """``(classification, workspaceUrl_or_None)``.

    The link is emitted ONLY for ``fully-served-complex`` and ``unmet`` (and only
    when a Workspace base URL is configured)."""
    classification = classify(
        ok=ok, question=question, spec=spec, artifacts=artifacts,
        truncated=truncated, max_total_rows=max_total_rows)
    if classification == CLASS_SIMPLE:
        return classification, None
    url = build_workspace_url(client=client, run=run, question=question,
                              filters=filters)
    return classification, url
