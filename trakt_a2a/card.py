"""trakt_a2a.card — the Agent Card, per A2A v1.0.

The Agent Card is how a stranger learns what can be delegated here. It is
published at ``/.well-known/agent-card.json`` and it answers one question:
*what work can I hand this agent, and how do I reach it?*

What the card must NOT say
--------------------------
Not one Trakt metric identifier. Not ``composition_vintage_share``, not
``OBSERVED_CPR@v2``, not ``cohort_comparison``. A caller that had to know those
would not be delegating — it would be remote-controlling, and the moment Trakt
renamed a metric the caller would break. The card describes **business
capability**; how Trakt produces the answer is Trakt's business and is free to
change underneath.

This is enforced, not merely intended: ``tests/test_a2a_card.py`` asserts that no
governed tool name and no methodology identifier appears anywhere in the
published card.

Version pinning
---------------
The A2A version is stated per interface (``protocolVersion`` on each
``supportedInterfaces`` entry), which is where v1.0 moved it — a single
top-level version could not describe an agent that speaks JSON-RPC at one
endpoint and gRPC at another.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

#: The A2A specification revision implemented. Stated as data so that a future
#: upgrade is a visible edit rather than a drift.
A2A_PROTOCOL_VERSION = "1.0"

#: Where a client looks first. Fixed by the specification; not ours to choose.
AGENT_CARD_PATH = "/.well-known/agent-card.json"

#: The single skill Sprint 4 exposes. One good specialist proves the
#: architecture; a catalogue of thin ones would prove less and cost more.
SKILL_ID = "securitisation_readiness_assessment"

SKILL_DESCRIPTION = (
    "Assess a governed loan portfolio for securitisation readiness. Identifies "
    "material strengths, weaknesses, risks and areas requiring further "
    "diligence, investigates significant findings, and returns evidence-backed "
    "conclusions with the source of every figure and the authority behind every "
    "threshold. Distinguishes what was measured from what was assumed, and "
    "reports what could not be assessed as carefully as what could."
)

#: Stated in the card so a caller knows the boundary before it delegates, rather
#: than discovering it in a refusal. Each line is a real limit of the specialist.
SKILL_LIMITS = (
    "Does not issue, predict or imply a credit rating.",
    "Does not produce PD, LGD or expected-loss estimates.",
    "Does not model borrower behaviour, prepayment forecasts or expected life.",
    "Does not run scenarios or stress tests.",
    "Reports only what the governed portfolio evidences; declines rather than "
    "estimates when an input is missing.",
)


def agent_card(*, endpoint_url: str,
               provider_name: str = "Trakt",
               provider_url: str = "https://trakt.example",
               agent_version: str = "1.0.0",
               streaming: bool = False,
               signatures: Optional[List[Dict[str, Any]]] = None
               ) -> Dict[str, Any]:
    """The published Agent Card.

    ``endpoint_url`` is supplied by the deployment rather than hardcoded — the
    card must name the endpoint a caller will actually reach, and that differs
    per environment. A card that advertised a wrong endpoint would be worse than
    no card: the caller would delegate confidently to nothing.
    """
    card: Dict[str, Any] = {
        "name": "Trakt Securitisation Readiness Agent",
        "description": (
            "A specialist credit agent that assesses governed loan portfolios "
            "for securitisation readiness. Delegate an objective; it decides "
            "what evidence it needs, obtains it from governed portfolio "
            "intelligence, and returns a structured, evidence-backed "
            "assessment."),
        "version": agent_version,
        "provider": {"organization": provider_name, "url": provider_url},
        "supportedInterfaces": [{
            "url": endpoint_url,
            "protocolBinding": "JSONRPC",
            "protocolVersion": A2A_PROTOCOL_VERSION,
        }],
        "capabilities": {
            "streaming": streaming,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["application/json", "text/plain"],
        "securitySchemes": {
            "enterprise_agent_oauth": {
                "type": "oauth2",
                "description": (
                    "Microsoft Entra client-credentials token carrying the "
                    "Trakt.Agent application role. The token establishes WHICH "
                    "AGENT is calling. It does not, by itself, grant access to "
                    "any portfolio — portfolio entitlement is resolved "
                    "separately from the calling organisation's grants."),
                "flows": {"clientCredentials": {
                    "tokenUrl": "https://login.microsoftonline.com/"
                                "{tenant}/oauth2/v2.0/token",
                    "scopes": {"Trakt.Agent": "Delegate work to a Trakt "
                                              "specialist agent"},
                }},
            },
        },
        "securityRequirements": [{"enterprise_agent_oauth": ["Trakt.Agent"]}],
        "skills": [{
            "id": SKILL_ID,
            "name": "Securitisation readiness assessment",
            "description": SKILL_DESCRIPTION,
            "tags": ["securitisation", "credit", "portfolio", "risk",
                     "structured-finance", "due-diligence"],
            "examples": [
                "Assess Portfolio II for securitisation readiness. Identify "
                "material strengths, weaknesses, risks and areas requiring "
                "further diligence.",
                "Is this portfolio ready to take to a term securitisation, and "
                "what would need to be remediated first?",
                "What is the main reason this portfolio is not ready?",
            ],
            "inputModes": ["text/plain"],
            "outputModes": ["application/json"],
            "securityRequirements": [{"enterprise_agent_oauth": ["Trakt.Agent"]}],
        }],
    }
    if signatures:
        card["signatures"] = signatures
    return card


#: What the caller is told it will get back, so it can decide whether the
#: specialist is worth delegating to before it spends a token doing so.
ARTIFACT_MEDIA_TYPE = "application/vnd.trakt.readiness-assessment+json"
