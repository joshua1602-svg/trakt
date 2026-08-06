"""trakt_notifications — proactive Teams delivery for approved portfolio updates.

A **delivery** capability, not an analytics one. Every number that reaches a
Teams card was produced by an existing governed MI or Insight service and is
carried through this package unchanged. Nothing here calculates a portfolio
metric, re-runs a concentration test, produces a forecast, infers a cause,
alters a severity or invents a recommendation.

The package is deliberately layered so that each of those prohibitions is
enforced by structure rather than by discipline:

  ``contract``        the versioned envelope — two messages, deterministic ids;
  ``sources``         the ONLY module that talks to MI, and it only reads;
  ``portfolio_update``/``risk_review``
                      selection and wording over already-governed values;
  ``recipients``      who may receive what, from authenticated identity;
  ``outbox``          durable at-least-once delivery with idempotency;
  ``cards``           presentation;
  ``teams_client``    transport;
  ``trigger``         the approval hook.

Import cost is kept low: nothing at package import time reaches MI, Azure or
the network.
"""

from __future__ import annotations

from .contract import (  # noqa: F401
    CONTRACT_VERSION,
    MESSAGE_PORTFOLIO_UPDATE,
    MESSAGE_RISK_REVIEW,
    NotificationBatch,
    NotificationMessage,
    UPDATE_COMBINED,
    UPDATE_FUNDED,
    UPDATE_PIPELINE,
)

__all__ = [
    "CONTRACT_VERSION",
    "MESSAGE_PORTFOLIO_UPDATE",
    "MESSAGE_RISK_REVIEW",
    "NotificationBatch",
    "NotificationMessage",
    "UPDATE_COMBINED",
    "UPDATE_FUNDED",
    "UPDATE_PIPELINE",
]
