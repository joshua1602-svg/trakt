"""mi_agent.portfolio_reference — Trakt portfolio reference model (Phase 6 Step 0).

Trakt assigns its own portfolio references per client during onboarding,
independent of how a source tape labels the portfolio. MI reports against the
Trakt reference (``portfolio_id`` / ``portfolio_name``). This module provides the
minimal config shape + helpers the runtime boundary and tests need — it does NOT
implement onboarding orchestration.

Key rules:
  * "portfolio" -> Trakt portfolio_id / portfolio_name (requires config);
  * "acquired portfolio" -> acquired_portfolio_id;
  * "SPV" -> spv_id;
  * never resolve "portfolio" to acquired_portfolio_id;
  * if no portfolio reference is configured -> structured issue, never a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# The canonical (snapshot/registry) column the Trakt portfolio reference maps to.
PORTFOLIO_ID_FIELD = "portfolio_id"
PORTFOLIO_NAME_FIELD = "portfolio_name"
ACQUIRED_PORTFOLIO_FIELD = "acquired_portfolio_id"
SPV_FIELD = "spv_id"


@dataclass
class PortfolioReference:
    portfolio_id: str
    portfolio_name: Optional[str] = None
    source_portfolio_label: Optional[str] = None
    source_portfolio_field: Optional[str] = None
    spv_id: Optional[str] = None
    acquired_portfolio_id: Optional[str] = None


@dataclass
class PortfolioReferenceConfig:
    """Per-client Trakt portfolio reference configuration."""

    client_id: str
    client_name: Optional[str] = None
    client_slug: Optional[str] = None
    portfolio_reference_pattern: str = "{client_slug}_{sequence:03d}"
    portfolios: List[PortfolioReference] = field(default_factory=list)

    # -- introspection ----------------------------------------------------- #

    @property
    def has_portfolio_references(self) -> bool:
        return any(p.portfolio_id for p in self.portfolios)

    def portfolio_ids(self) -> List[str]:
        return [p.portfolio_id for p in self.portfolios if p.portfolio_id]

    def mint_reference(self, sequence: int) -> str:
        """Deterministically mint the Nth Trakt portfolio reference id."""
        slug = self.client_slug or (self.client_id or "client").lower()
        return self.portfolio_reference_pattern.format(
            client_slug=slug, client_id=self.client_id, sequence=sequence)

    # -- serialisation ----------------------------------------------------- #

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioReferenceConfig":
        ports = [PortfolioReference(
            portfolio_id=p.get("portfolio_id"),
            portfolio_name=p.get("portfolio_name"),
            source_portfolio_label=p.get("source_portfolio_label"),
            source_portfolio_field=p.get("source_portfolio_field"),
            spv_id=p.get("spv_id"),
            acquired_portfolio_id=p.get("acquired_portfolio_id"),
        ) for p in (data.get("portfolios") or [])]
        return cls(
            client_id=data.get("client_id"),
            client_name=data.get("client_name"),
            client_slug=data.get("client_slug"),
            portfolio_reference_pattern=data.get(
                "portfolio_reference_pattern", "{client_slug}_{sequence:03d}"),
            portfolios=ports,
        )


def load_portfolio_reference_config(path: Path | str) -> PortfolioReferenceConfig:
    """Load a portfolio reference config from YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"portfolio reference config not found: {path}")
    return PortfolioReferenceConfig.from_dict(
        yaml.safe_load(path.read_text(encoding="utf-8")) or {})
