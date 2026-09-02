"""portfolio_review — the autonomous Portfolio Review Agent.

An MI analyst. It answers one question — *what materially changed in the
operating portfolio since the previous accepted reporting period, why, and what
should management pay attention to* — and it is deliberately incapable of
answering others.

It reuses the readiness agent's loop and session rather than adding a second
agent framework beside them, because duplicating a safety property is how one
copy quietly loses it. What is new here is a mandate, an objective, a
publication gate and a card.

    mandate + objective + snapshot pair
        -> readiness_agent.run_assessment          the loop, unchanged
            -> MIScopedSession                     refuses out-of-mandate tools
                -> GovernedSession                 three verbs, no DataFrame
                    -> execute_governed_tool       capability + entitlement
                        -> the governed services   the workspace also reads
        -> numeric_gate                            no ungoverned figure escapes
            -> brief                               a ranked card, inside a budget

WHY THERE ARE THREE CONTROLS AND NOT ONE PROMPT
-----------------------------------------------
The first version of this agent had one control — a system prompt stating its
rules in the strongest terms available. Run against a real model on real
canonical, it published a figure it had worked out by addition and announced a
breach of a rulebook whose own payload said it was not a real agreement. Every
sentence in that prompt is still here, and none of them is load-bearing any
more:

    mandate      decides what the agent may ask        (tool allow-list)
    numeric_gate decides what it may publish           (figure must be governed)
    brief        decides how much of it a reader sees  (ranked, budgeted)

The prompt's job is to make the model want to do the right thing so the gates
rarely bite. The gates' job is to make it true when it doesn't.
"""

from __future__ import annotations

from .mandate import (  # noqa: F401
    ALLOWED,
    EXCLUDED,
    EXCLUDED_NAMES,
    IN_SCOPE,
    MANDATE,
    PROHIBITED,
    audit_registry,
    is_allowed,
)
from .objective import (  # noqa: F401
    MONTHLY_FUNDED_OBJECTIVE,
    SUBMIT_REVIEW,
    SYSTEM_PROMPT,
    WEEKLY_PIPELINE_OBJECTIVE,
    objective_for,
)
from .session import MIScopedSession  # noqa: F401
from .brief import Card, render  # noqa: F401
from .controller import (  # noqa: F401
    PERIOD_MONTHLY_FUNDED,
    PERIOD_WEEKLY_PIPELINE,
    ReviewOutcome,
    run_review,
)

__all__ = [
    "ALLOWED",
    "Card",
    "EXCLUDED",
    "EXCLUDED_NAMES",
    "IN_SCOPE",
    "MANDATE",
    "MIScopedSession",
    "MONTHLY_FUNDED_OBJECTIVE",
    "PERIOD_MONTHLY_FUNDED",
    "PERIOD_WEEKLY_PIPELINE",
    "PROHIBITED",
    "ReviewOutcome",
    "SUBMIT_REVIEW",
    "SYSTEM_PROMPT",
    "WEEKLY_PIPELINE_OBJECTIVE",
    "audit_registry",
    "is_allowed",
    "objective_for",
    "render",
    "run_review",
]
