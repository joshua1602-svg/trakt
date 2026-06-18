"""mi_agent.risk_monitor.models — result, issues and config for the risk monitor.

Phase 5 risk monitor foundations. Pure data structures + the
``config/mi/risk_monitor.yaml`` loader. Reuses the shared issue convention from
``mi_agent.states.models``. No UI, no charts, no LLM, no Azure, no legacy
``analytics/`` imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# Reuse the shared issue helpers / severities / cross-phase codes.
from mi_agent.states.models import (  # noqa: F401
    ERROR,
    INFO,
    WARNING,
    MISSING_BASELINE_SNAPSHOT,
    MISSING_CURRENT_SNAPSHOT,
    make_issue,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RISK_MONITOR_CONFIG = REPO_ROOT / "config" / "mi" / "risk_monitor.yaml"

BALANCE_COL = "current_outstanding_balance"
DEFAULT_KEY = "loan_id"

# --------------------------------------------------------------------------- #
# Movement types
# --------------------------------------------------------------------------- #

UNCHANGED = "unchanged"
IMPROVED = "improved"
DETERIORATED = "deteriorated"
NEW = "new"
EXITED = "exited"
CHANGED = "changed"      # differing values but the dimension is unordered
UNKNOWN = "unknown"      # value missing on one/both sides for an in-both loan

# --------------------------------------------------------------------------- #
# Phase 5 issue codes (cross-phase codes imported above)
# --------------------------------------------------------------------------- #

MISSING_STABLE_KEY_FOR_MIGRATION = "missing_stable_key_for_migration"
MISSING_MIGRATION_DIMENSION = "missing_migration_dimension"
UNORDERED_MIGRATION_DIMENSION = "unordered_migration_dimension"
INSUFFICIENT_SNAPSHOTS_FOR_TRAJECTORY = "insufficient_snapshots_for_trajectory"
MISSING_CONCENTRATION_DIMENSION = "missing_concentration_dimension"
MISSING_LIMIT_CONFIG = "missing_limit_config"
CONCENTRATION_BELOW_MINIMUM_THRESHOLD = "concentration_below_minimum_threshold"
UNSUPPORTED_RISK_MONITOR_ROUTE = "unsupported_risk_monitor_route"
EMPTY_RISK_MONITOR_RESULT = "empty_risk_monitor_result"


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #


@dataclass
class RiskMonitorResult:
    """Outcome of a risk-monitor computation (migration / concentration /
    trajectory / per-loan flags)."""

    kind: str
    frame: pd.DataFrame
    issues: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(i.get("severity") == ERROR for i in self.issues)

    @property
    def row_count(self) -> int:
        return int(len(self.frame))

    def issue_codes(self) -> List[str]:
        return [i["code"] for i in self.issues]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def load_risk_monitor_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load ``config/mi/risk_monitor.yaml`` (``{}`` if absent)."""
    path = Path(path) if path else DEFAULT_RISK_MONITOR_CONFIG
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def get_ordering(config: Optional[Dict[str, Any]],
                 dimension: str) -> Optional[List[str]]:
    """Return the BEST->WORST ordering for *dimension*, or ``None`` (unordered)."""
    if not config:
        return None
    orderings = config.get("deterioration_orderings") or {}
    order = orderings.get(dimension)
    if isinstance(order, list) and order:
        return [str(v).strip() for v in order]
    return None


def get_concentration_thresholds(config: Optional[Dict[str, Any]]
                                 ) -> Dict[str, float]:
    if config and isinstance(config.get("concentration_thresholds"), dict):
        ct = config["concentration_thresholds"]
        return {"amber": float(ct.get("amber", 0.20)),
                "red": float(ct.get("red", 0.30))}
    return {"amber": 0.20, "red": 0.30}


def get_approaching_at(config: Optional[Dict[str, Any]]) -> float:
    if config and isinstance(config.get("trajectory"), dict):
        return float(config["trajectory"].get("approaching_limit_at", 0.90))
    return 0.90


def get_trajectory_window(config: Optional[Dict[str, Any]]) -> int:
    if not config:
        return 3
    if config.get("trajectory_window") is not None:
        return int(config["trajectory_window"])
    if isinstance(config.get("trajectory"), dict):
        return int(config["trajectory"].get("min_snapshots", 3))
    return 3


def get_minimums(config: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not config:
        return {"balance": 0.0, "count": 0}
    return {"balance": float(config.get("minimum_balance_threshold", 0) or 0),
            "count": int(config.get("minimum_count_threshold", 0) or 0)}
