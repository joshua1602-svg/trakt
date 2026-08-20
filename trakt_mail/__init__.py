"""trakt_mail — outbound onboarding email over Microsoft Graph.

A transport, and nothing else. The Operations Control Centre decides *whether*
a communication may leave Trakt and *what* it says; this package only carries
it, and only when a human has already approved it.

The seam it plugs into already exists:
:class:`operations_control.occ_agent.communication.CommunicationAdapter`. The
OCC Agent's pack workflow — ``DRAFTED → HUMAN_REVIEW_REQUIRED →
APPROVED_TO_SEND → SENT`` — is unchanged by this package's existence, and
``OccAgentService.send_pack`` still refuses to run in any state but
``APPROVED_TO_SEND``. Registering this adapter changes one thing: the receipt
now says ``sent: True`` and carries the real message identifier, instead of
saying plainly that nothing was sent.

Deliberately modelled on :mod:`trakt_notifications` (the Teams capability that
is already in production): a thin ``urllib`` REST client rather than an SDK,
client-credentials authentication, classified retryable/permanent failures, an
injectable transport so every path is testable without a network, and a
configuration module that fails closed.

Nothing here reads an LLM, and nothing here chooses a recipient, a subject or a
body — all three arrive from the governed pack.

Imports are resolved on first use rather than at module import. ``config`` and
``graph_client`` depend on nothing but the standard library, so a caller that
needs only those — a Function App worker, a diagnostic — does not pull in the
Operations Control engine (and its analytical dependency tree) to get them.
``adapter`` does, because it implements an OCC contract.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "GraphMailAdapter",
    "GraphMailClient",
    "GraphMailError",
    "MailConfig",
    "MailDeliveryRefused",
    "MailNotConfigured",
    "SentMessage",
    "load",
    "outbound_adapter",
    "outbound_enabled",
]

_SOURCES = {
    "GraphMailAdapter": "adapter",
    "MailDeliveryRefused": "adapter",
    "outbound_adapter": "adapter",
    "GraphMailClient": "graph_client",
    "GraphMailError": "graph_client",
    "SentMessage": "graph_client",
    "MailConfig": "config",
    "MailNotConfigured": "config",
    "load": "config",
    "outbound_enabled": "config",
}


def __getattr__(name: str) -> Any:          # PEP 562
    module = _SOURCES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list:
    return sorted(__all__)
