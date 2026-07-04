"""Pipeline-vs-funded timing disclosure.

The pipeline is a continuous weekly operational view; funded actuals are a monthly
cut that usually lags the latest pipeline extract (c. 2-3 weeks in steady state,
more while funded artefacts are being caught up). We deliberately do NOT suppress
or truncate the latest pipeline just because funded actuals lag — instead we
DISCLOSE the date difference so the two anchors are never silently conflated.

``timing_disclosure`` returns a small, camelCased payload the API attaches to the
pipeline / forecast responses and the UI renders as a lightweight, NON-BLOCKING
banner. Above a configurable threshold (``TRAKT_PIPELINE_LAG_WARN_DAYS``, default
45) the message is strengthened to prompt a check that funded actuals are pending.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Dict, Optional

DEFAULT_WARN_DAYS = 45


def _parse(d: Optional[str]) -> Optional[date]:
    if not d:
        return None
    try:
        return date.fromisoformat(str(d)[:10])
    except (ValueError, TypeError):
        return None


def _warn_days(explicit: Optional[int]) -> int:
    if explicit is not None:
        return int(explicit)
    try:
        return int(os.environ.get("TRAKT_PIPELINE_LAG_WARN_DAYS", str(DEFAULT_WARN_DAYS)))
    except (ValueError, TypeError):
        return DEFAULT_WARN_DAYS


def timing_disclosure(funded_actuals_as_of: Optional[str],
                      pipeline_extract_as_of: Optional[str],
                      *, warn_days: Optional[int] = None) -> Dict[str, Any]:
    """A non-blocking disclosure of the funded-vs-pipeline date difference.

    ``level`` is ``none`` when the pipeline is not later than funded (or a date is
    missing), ``info`` when it is later within the threshold, and ``warning`` when
    the lag exceeds the threshold. The funded and pipeline anchors are always
    echoed so callers/UI can display both explicitly."""
    warn = _warn_days(warn_days)
    out: Dict[str, Any] = {
        "fundedActualsAsOf": funded_actuals_as_of,
        "pipelineExtractAsOf": pipeline_extract_as_of,
        "lagDays": None,
        "level": "none",
        "message": None,
        "warnThresholdDays": warn,
    }
    fd, pd_ = _parse(funded_actuals_as_of), _parse(pipeline_extract_as_of)
    if fd is None or pd_ is None:
        return out
    lag = (pd_ - fd).days
    out["lagDays"] = lag
    if lag <= 0:
        return out  # pipeline not later than funded — nothing to disclose
    if lag > warn:
        out["level"] = "warning"
        out["message"] = (
            f"Pipeline extract is {lag} days after the selected funded reporting "
            f"date. Confirm funded actuals are pending before relying on forecast "
            f"bridge metrics.")
    else:
        out["level"] = "info"
        out["message"] = (
            f"Funded actuals are as of {funded_actuals_as_of}. Pipeline is shown "
            f"using the latest weekly extract dated {pipeline_extract_as_of}.")
    return out
