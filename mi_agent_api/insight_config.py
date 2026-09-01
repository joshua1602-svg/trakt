#!/usr/bin/env python3
"""Phase 3A — materiality configuration for the Weekly Portfolio Brief.

Thin loader over ``config/mi/insights.yaml``, following the same convention as
``analytics_lib.config_loader``. There is no new configuration framework here:
one YAML file, read through the mechanism the bucket definitions already use.

The defaults below are the contract. A missing file, a missing section or a
missing key all fall back to them, so a partial override is safe and a
configuration problem degrades to "the documented defaults" rather than to a
failed brief. The engine reads thresholds ONLY through this module — no
threshold is written in the calculation layer.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("mi_agent_api.insights")

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = _REPO_ROOT / "config" / "mi" / "insights.yaml"

#: Override for tests and for a deployment that ships its own thresholds.
PATH_ENV = "TRAKT_MI_INSIGHTS_CONFIG"

#: The documented defaults. Every threshold is relative — never an absolute
#: currency amount, which would be the whole story on a small book and noise on
#: a large one.
DEFAULTS: Dict[str, Any] = {
    "pipeline_movement": {
        "min_change_pct": 2.0,
        "min_change_share_of_pipeline_pct": 2.0,
    },
    "ticket_size": {
        "min_average_change_pct": 5.0,
        "min_median_change_pct": 5.0,
        "min_case_count": 20,
    },
    "weighted_ltv": {
        "min_change_pp": 1.0,
        "min_coverage_pct": 90.0,
    },
    "ticket_mix": {"min_share_change_pp": 5.0},
    "ltv_mix": {"min_share_change_pp": 5.0},
    "completions": {"min_change_pct": 10.0},
    "conversion": {
        # Turning this off skips resolving the origination funnel entirely,
        # which is 63% of the brief's cold cost (3.5s of 5.6s on a 26-week
        # root) and exists solely to supply this one insight. Left ON by
        # default: the saving is a deployment's choice, not a silent default.
        "enabled": True,
        "minimum_acceptable_rate": None,
        "require_sufficient": True,
    },
    "concentration": {
        "amber_utilisation_pct": 90.0,
        "red_utilisation_pct": 100.0,
        "forecast_horizon_weeks": 12,
    },
    "data_quality": {
        "max_dimension_reassignment_pct": 5.0,
        "max_unknown_share_pct": 5.0,
        "max_unusable_id_share_pct": 1.0,
        "max_duplicate_id_share_pct": 1.0,
        "emit_when_clean": False,
    },

    # ---- Funded (monthly) --------------------------------------------------
    # Lower than the weekly gates by design. A funded book is a stock reported
    # monthly, so it moves less than a pipeline does in a week, and a threshold
    # tuned for weekly flow would suppress a month worth reading.
    "funded_movement": {"min_change_pct": 1.0},
    "funded_composition": {
        "min_component_share_of_gross_pct": 5.0,
        "dominant_addition_share_pct": 50.0,
    },
    "funded_mix": {"min_share_change_pp": 3.0},
    "funded_ltv": {"min_change_pp": 0.5},
    # Not a threshold: the statuses and transitions are the approved
    # concentration configuration's. This decides only whether an IMPROVEMENT is
    # worth a card, and defaults to yes.
    "risk_limit_transition": {"report_improvements": True},
}

BRIEF_DEFAULTS: Dict[str, Any] = {
    "max_insights": 8,
    "max_per_type": {
        "CONCENTRATION_PROXIMITY": 2,
        "DATA_QUALITY": 1,
        "TICKET_SIZE": 1,
        "WEIGHTED_LTV": 1,
        "TICKET_MIX_SHIFT": 1,
        "LTV_MIX_SHIFT": 1,
        "PIPELINE_MOVEMENT": 1,
        "COMPLETIONS_MOVEMENT": 1,
        "CONVERSION_CONTEXT": 1,
    },
}

#: The monthly review is richer than the weekly one and its limits say so. More
#: total, more risk transitions (a month can cross several limits and reporting
#: three of four would be the omission a reader most needs not to have), and
#: several mix shifts because the dimensions are independent questions.
FUNDED_BRIEF_DEFAULTS: Dict[str, Any] = {
    "max_insights": 12,
    "max_per_type": {
        "RISK_LIMIT_TRANSITION": 4,
        "CONCENTRATION_PROXIMITY": 2,
        "FUNDED_COMPOSITION": 1,
        "FUNDED_MOVEMENT": 1,
        "UNDERLYING_BOOK_MOVEMENT": 1,
        "FUNDED_LTV_MOVEMENT": 1,
        "FUNDED_MIX_SHIFT": 3,
    },
}

_CACHE: Dict[str, Any] = {}


def _merge(base: Dict[str, Any], over: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """One level of override per section — a partial file overrides only what
    it names, so a deployment can tune one threshold without restating all."""
    out = copy.deepcopy(base)
    for section, values in (over or {}).items():
        if isinstance(values, dict) and isinstance(out.get(section), dict):
            out[section].update(values)
        else:
            out[section] = values
    return out


def _read(path: Path) -> Dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load(path: Optional[Path | str] = None, *, refresh: bool = False
         ) -> Dict[str, Any]:
    """``{"insights": {...}, "brief": {...}}`` with defaults applied.

    Cached on the resolved path: the file is deployment configuration, not
    per-request state. ``refresh=True`` re-reads it (used by tests).
    """
    resolved = Path(path or os.environ.get(PATH_ENV) or DEFAULT_PATH)
    key = str(resolved)
    if not refresh and key in _CACHE:
        return _CACHE[key]

    raw: Dict[str, Any] = {}
    try:
        if resolved.exists():
            raw = _read(resolved)
        else:
            logger.info("insight config %s not found — using defaults", resolved)
    except Exception as exc:  # noqa: BLE001 - configuration must never fail closed
        logger.warning("insight config %s unreadable (%s) — using defaults",
                       resolved, exc)

    cfg = {
        "insights": _merge(DEFAULTS, (raw or {}).get("insights")),
        "brief": _merge(BRIEF_DEFAULTS, (raw or {}).get("brief")),
        "funded_brief": _merge(FUNDED_BRIEF_DEFAULTS,
                               (raw or {}).get("funded_brief")),
        "source": str(resolved) if resolved.exists() else "defaults",
        "version": (raw or {}).get("version", 1),
    }
    _CACHE[key] = cfg
    return cfg


def thresholds(section: str, path: Optional[Path | str] = None) -> Dict[str, Any]:
    """The threshold block for one insight family."""
    return load(path).get("insights", {}).get(section, {}) or {}


def brief_limits(path: Optional[Path | str] = None) -> Dict[str, Any]:
    return load(path).get("brief", {}) or {}


def funded_brief_limits(path: Optional[Path | str] = None) -> Dict[str, Any]:
    return load(path).get("funded_brief", {}) or {}


def reset_cache() -> None:
    _CACHE.clear()
