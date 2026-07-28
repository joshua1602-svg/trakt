"""mi_agent.period_change — the governed Period Change Analysis workflow.

The first analytical workflow that consumes the Business Semantics Registry.
It answers "what changed between two reporting snapshots?" deterministically:
every number is computed here, from governed metadata, with no LLM anywhere in
the calculation path.

Execution path
--------------
::

    question
      └─► mi_agent.parsed_question.ParsedQuestion        (the EXISTING single parse)
            └─► period_change.recognition.recognise()    (intent + mode + periods)
                  └─► period_change.workflow.run_period_change_analysis()
                        ├─► periods.resolve_periods()            two governed snapshots
                        ├─► business_semantics.load_…()          BSR v2 metadata
                        ├─► selection.select_fields()            governed field policy
                        ├─► calculations.metric_change()         per-field movement
                        ├─► distribution.distribution_change()   composition shift
                        └─► bridge.balance_bridge()              loan-level reconciliation
                              └─► PeriodChangeResult             governed result contract

The package is deliberately free of any interface dependency: it takes
:class:`~mi_agent.period_change.models.SnapshotFrame` objects from a caller and
returns a typed result. ``mi_agent_api/period_change_route.py`` is the adapter
that supplies those snapshots from the existing governed dataset services and
renders the result through the existing presenters.
"""

from __future__ import annotations

from .models import (  # noqa: F401
    WORKFLOW_ID,
    CALCULATION_VERSION,
    DistributionChange,
    MetricChange,
    PeriodChangeResult,
    PeriodResolution,
    SnapshotFrame,
)
from .recognition import PeriodChangeIntent, recognise  # noqa: F401
from .workflow import PeriodChangeRequest, run_period_change_analysis  # noqa: F401

__all__ = [
    "WORKFLOW_ID",
    "CALCULATION_VERSION",
    "DistributionChange",
    "MetricChange",
    "PeriodChangeIntent",
    "PeriodChangeRequest",
    "PeriodChangeResult",
    "PeriodResolution",
    "SnapshotFrame",
    "recognise",
    "run_period_change_analysis",
]
