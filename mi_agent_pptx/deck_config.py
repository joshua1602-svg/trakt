"""mi_agent_pptx.deck_config — YAML-driven deck configuration.

The deck is fully config-driven: slides, their titles, strapline sources, chart
specs, metric specs, field requirements, fallback behaviour, lens applicability
and broker-suppression are all declared in a YAML file (see
``configs/pptx/investor_pack.yaml``) rather than hard-coded in the builder. This
module loads and lightly validates that file into typed accessors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SlideSpec:
    """One slide declaration from the deck config."""

    id: str
    type: str                       # cover | kpi | charts | risk_monitor | methodology | appendix
    title: str = ""
    strapline_source: str = "deterministic"  # deterministic | llm
    lens: str = "any"               # any | funded | pipeline | forecast
    kpis: List[str] = field(default_factory=list)
    charts: List[Dict[str, Any]] = field(default_factory=list)
    mandatory: bool = False
    suppress_broker_consolidated: bool = False
    notes: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeckConfig:
    """Parsed deck configuration."""

    name: str
    theme: str
    default_lens: str
    footer: str
    slides: List[SlideSpec]
    metrics: Dict[str, Dict[str, Any]]
    max_strapline_words: int = 24
    suppress_broker_consolidated: bool = True
    logo_path: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def slide_count(self) -> int:
        return len(self.slides)

    def metric_spec(self, key: str) -> Dict[str, Any]:
        spec = dict(self.metrics.get(key, {}))
        spec.setdefault("key", key)
        return spec

    def metric_specs(self, keys: List[str]) -> List[Dict[str, Any]]:
        return [self.metric_spec(k) for k in keys]


def load_deck_config(path: str | Path) -> DeckConfig:
    """Load and validate a deck config YAML into a :class:`DeckConfig`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"deck config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    deck = data.get("deck", {}) or {}
    metrics = data.get("metrics", {}) or {}

    slides: List[SlideSpec] = []
    for entry in data.get("slides", []) or []:
        slides.append(SlideSpec(
            id=entry.get("id", f"slide_{len(slides) + 1}"),
            type=entry.get("type", "charts"),
            title=entry.get("title", ""),
            strapline_source=entry.get("strapline_source", "deterministic"),
            lens=entry.get("lens", "any"),
            kpis=list(entry.get("kpis", []) or []),
            charts=list(entry.get("charts", []) or []),
            mandatory=bool(entry.get("mandatory", False)),
            suppress_broker_consolidated=bool(
                entry.get("suppress_broker_consolidated", False)),
            notes=entry.get("notes", ""),
            raw=entry,
        ))

    return DeckConfig(
        name=deck.get("name", "MI Agent Investor Pack"),
        theme=deck.get("theme", "trakt_mi_agent_dark"),
        default_lens=deck.get("default_lens", "total"),
        footer=deck.get("footer", "trakt MI Agent · Confidential"),
        slides=slides,
        metrics=metrics,
        max_strapline_words=int(deck.get("max_strapline_words", 24)),
        suppress_broker_consolidated=bool(
            deck.get("suppress_broker_consolidated", True)),
        logo_path=deck.get("logo_path"),
        raw=data,
    )
