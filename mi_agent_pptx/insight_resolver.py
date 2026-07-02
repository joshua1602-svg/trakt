"""mi_agent_pptx.insight_resolver — slide straplines (deterministic or LLM-supplied).

Every slide carries a strapline below its title. Straplines are resolved in this
priority order (per the deck spec):

1. **LLM-supplied** — an optional MI Agent JSON artifact mapping ``slide_id`` to
   a pre-written strapline (``insight_resolver`` never calls an LLM itself; it
   only *consumes* an artifact the pipeline may have produced).
2. **Deterministic fallback** — a concise, institutional one-liner assembled from
   already-resolved metric values. No metric ⇒ no fabricated insight; instead a
   neutral coverage line is used.

Straplines are capped at 18–24 words and never fabricate figures that were not
resolved from artifacts/registries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .metric_resolver import MetricResult

MAX_WORDS = 24
MIN_TARGET_WORDS = 18  # target ceiling for concision; not a hard floor


def _clip_words(text: str, max_words: int = MAX_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:") + "."


@dataclass
class StraplineResolver:
    """Resolve a strapline for each slide."""

    metrics: Dict[str, MetricResult]
    llm_artifact: Optional[Dict[str, Any]] = None

    def _llm_strapline(self, slide_id: str) -> Optional[str]:
        art = self.llm_artifact
        if not isinstance(art, dict):
            return None
        # Accept either {slide_id: "text"} or {"straplines": {slide_id: {...}}}.
        container = art.get("straplines", art)
        if not isinstance(container, dict):
            return None
        entry = container.get(slide_id)
        if entry is None:
            return None
        if isinstance(entry, str):
            return _clip_words(entry.strip())
        if isinstance(entry, dict):
            text = entry.get("text") or entry.get("strapline")
            if text:
                return _clip_words(str(text).strip())
        return None

    def _m(self, key: str) -> Optional[MetricResult]:
        m = self.metrics.get(key)
        return m if (m and m.ok) else None

    def _deterministic(self, slide_id: str, slide_type: str) -> str:
        """Assemble a concise strapline from resolved metrics only."""
        m = self._m

        if slide_id in ("executive_summary", "cover"):
            funded = m("funded_balance")
            count = m("loan_count")
            ltv = m("wa_current_ltv")
            parts: List[str] = []
            if funded:
                parts.append(f"Funded book of {funded.display}")
            if count:
                parts.append(f"across {count.display} loans")
            if ltv:
                parts.append(f"at {ltv.display} weighted current LTV")
            if parts:
                return _clip_words(", ".join(parts) + ".")

        if slide_id == "pipeline_overview":
            pipe = m("total_pipeline")
            if pipe:
                return _clip_words(
                    f"Total pipeline of {pipe.display} tracked by stage, "
                    f"ticket size and distribution channel.")

        if slide_id == "pipeline_conversion":
            return ("Stage-to-stage conversion and cohort progression across the "
                    "active pipeline, where conversion history is available.")

        if slide_id in ("pipeline_forecast", "forecast_snapshot"):
            fc = m("forecast_funded_balance")
            if fc:
                return _clip_words(
                    f"Deterministic baseline forecast projecting {fc.display} "
                    f"expected funded balance.")
            return ("Deterministic baseline run-rate forecast of expected funded "
                    "balance from the current pipeline.")

        if slide_id == "funded_evolution":
            funded = m("funded_balance")
            if funded:
                return _clip_words(
                    f"Funded balance evolution to {funded.display}, tracking loan "
                    f"count and average ticket over time.")

        if slide_id.startswith("stratification"):
            return ("Funded balance stratified across registry-authorised risk "
                    "and exposure dimensions.")

        if slide_id == "risk_analytics":
            return ("Multi-dimensional exposure concentration across LTV, borrower "
                    "age and region by funded balance.")

        if slide_id == "vintage":
            return ("Origination vintages with weighted LTV and borrower-age "
                    "profile by funding cohort.")

        if slide_id == "risk_monitor":
            breached = m("limits_breached")
            if breached:
                return _clip_words(
                    f"Concentration versus limits: {breached.display} breached, "
                    f"monitored on a red/amber/green basis.")
            return ("Portfolio concentration monitored against risk limits on a "
                    "red/amber/green utilisation basis.")

        if slide_id in ("methodology", "notes"):
            return ("Source artifacts, cut-off basis, forecast method and data "
                    "limitations for this pack.")

        # Generic fallback: largest resolved headline metric, else neutral.
        for key in ("funded_balance", "total_pipeline", "loan_count"):
            mm = m(key)
            if mm:
                return _clip_words(
                    f"{mm.label} of {mm.display} for the current reporting period.")
        return "Prepared from the latest MI Agent pipeline run."

    def resolve(self, slide_id: str, slide_type: str = "charts") -> str:
        """Return the strapline for *slide_id* (LLM artifact first)."""
        llm = self._llm_strapline(slide_id)
        if llm:
            return llm
        return self._deterministic(slide_id, slide_type)
