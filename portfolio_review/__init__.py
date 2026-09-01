"""portfolio_review — the autonomous period review.

Reuses the readiness agent's loop, session and governed tool surface rather than
adding a second agent framework beside them. What is new here is an objective, a
submission schema, and the snapshot pair the review is pinned to.

    objective + snapshot pair
        -> readiness_agent.run_assessment          the loop, unchanged
            -> GovernedSession                     three verbs, no DataFrame
                -> execute_governed_tool           capability + entitlement
                    -> the governed services       the workspace also reads
        -> ranked findings + evidence

The model decides what to investigate. It never calculates a portfolio fact,
and the enforcement is structural rather than instructional: it is handed tool
results, never a frame.
"""

from __future__ import annotations

from .objective import (  # noqa: F401
    MONTHLY_FUNDED_OBJECTIVE,
    SUBMIT_REVIEW,
    SYSTEM_PROMPT,
    WEEKLY_PIPELINE_OBJECTIVE,
    objective_for,
)
from .controller import (  # noqa: F401
    PERIOD_MONTHLY_FUNDED,
    PERIOD_WEEKLY_PIPELINE,
    ReviewOutcome,
    run_review,
)

__all__ = [
    "MONTHLY_FUNDED_OBJECTIVE",
    "PERIOD_MONTHLY_FUNDED",
    "PERIOD_WEEKLY_PIPELINE",
    "ReviewOutcome",
    "SUBMIT_REVIEW",
    "SYSTEM_PROMPT",
    "WEEKLY_PIPELINE_OBJECTIVE",
    "objective_for",
    "run_review",
]
