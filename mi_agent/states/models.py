"""mi_agent.states.models — result + issue model for MI state assembly.

Phase 3 MI state assembler. Pure data structures shared across the assembler,
selectors and route-eligibility helpers. No UI, no charts, no LLM, no Azure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# Issue codes & severities (compatible with analytics_lib / snapshot phases)
# --------------------------------------------------------------------------- #

MISSING_REQUIRED_STATE_FIELD = "missing_required_state_field"
MISSING_OPTIONAL_STATE_FIELD = "missing_optional_state_field"
UNAVAILABLE_STATE = "unavailable_state"
UNAVAILABLE_DIMENSION = "unavailable_dimension"
MISSING_SNAPSHOT = "missing_snapshot"
EMPTY_STATE_FRAME = "empty_state_frame"
MISSING_FORECAST_PROBABILITY = "missing_forecast_probability"
MISSING_BALANCE_FIELD = "missing_balance_field"
MISSING_FUNDED_STATUS = "missing_funded_status"
MISSING_PIPELINE_STAGE = "missing_pipeline_stage"
UNSUPPORTED_STATE_FOR_ROUTE = "unsupported_state_for_route"
INVALID_DATE = "invalid_date"

ERROR = "error"
WARNING = "warning"
INFO = "info"


def make_issue(code: str, severity: str, message: str,
               field: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    issue = {"code": code, "severity": severity, "message": message,
             "field": field}
    issue.update(extra)
    return issue


# --------------------------------------------------------------------------- #
# Funded / pipeline value vocabularies (case-insensitive)
# --------------------------------------------------------------------------- #

FUNDED_STATUS_VALUES = {
    "funded", "funded_book", "f", "true", "1", "yes", "y",
    "completed", "complete", "live", "active", "on_book", "drawn",
}
PIPELINE_STATUS_VALUES = {
    "pipeline", "in_pipeline", "unfunded", "pending", "p", "false", "0",
    "no", "n", "application", "applied", "offer", "kfi", "prospect",
}
# pipeline_stage values that mean the loan has actually completed/funded.
FUNDED_STAGE_VALUES = {
    "funded", "completed", "complete", "funds_released", "drawn",
    "drawdown", "completion", "live",
}


def classify_funded_value(value: Any) -> Optional[bool]:
    """Return True (funded), False (pipeline) or None (unknown) for a
    ``funded_status`` cell."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip().lower()
    if text in FUNDED_STATUS_VALUES:
        return True
    if text in PIPELINE_STATUS_VALUES:
        return False
    return None


# --------------------------------------------------------------------------- #
# StateResult
# --------------------------------------------------------------------------- #


@dataclass
class StateResult:
    """Outcome of assembling one MI state.

    ``frame`` is the analytical loan-level DataFrame (possibly empty);
    ``issues`` is the structured issue list; ``metadata`` carries provenance and
    summary numbers (counts, totals, fallbacks applied, snapshot ids).
    """

    state: str
    frame: pd.DataFrame
    issues: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when no error-severity issue was recorded."""
        return not any(i.get("severity") == ERROR for i in self.issues)

    @property
    def row_count(self) -> int:
        return int(len(self.frame))

    def issue_codes(self) -> List[str]:
        return [i["code"] for i in self.issues]
