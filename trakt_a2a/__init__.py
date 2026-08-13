"""trakt_a2a — the Trakt Securitisation Readiness Agent, addressable over A2A.

One skill, one boundary. A general enterprise agent discovers this specialist
from its Agent Card, delegates a business objective, and receives a structured
evidence-backed assessment. It never learns which metrics Trakt used to produce
it, and it never needs to.

    Enterprise agent  --A2A-->  this package
                                     |
                                     v
                        Securitisation Readiness Agent (Sprint 3)
                                     |
                                    MCP
                                     v
                          governed Trakt tools
                                     |
                                     v
                     shared deterministic analytics

The layers above the analytics are transport and judgement. None of them
calculates anything.
"""

from .card import (A2A_PROTOCOL_VERSION, AGENT_CARD_PATH, ARTIFACT_MEDIA_TYPE,
                   SKILL_ID, agent_card)
from .server import (A2AError, CallerIdentity,
                     SecuritisationReadinessA2AServer, readiness_artifact)
from .tasks import Task, TaskStore

__all__ = [
    "A2A_PROTOCOL_VERSION", "AGENT_CARD_PATH", "ARTIFACT_MEDIA_TYPE",
    "SKILL_ID", "agent_card", "A2AError", "CallerIdentity",
    "SecuritisationReadinessA2AServer", "readiness_artifact",
    "Task", "TaskStore",
]
