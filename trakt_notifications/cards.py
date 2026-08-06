#!/usr/bin/env python3
"""Adaptive Card rendering — presentation only.

Institutional and plain. A notification card competes with every other card in
someone's Teams feed, and the way it wins is by being readable in two seconds,
not by being decorated.

What a card may contain is deliberately narrow, and the constraints are checks
rather than conventions:

* no loan-level data — the payload it renders has none, because
  :mod:`.risk_review` and :mod:`.portfolio_update` only ever read dimension
  aggregates, and a test asserts the rendered card carries no case identifier;
* no internal identifiers — run ids and blob URIs never reach a card;
* no dashboard tables — up to five observations or three risk items;
* one action, the deep link, and none where no workspace is configured.

Card schema 1.5, which is what Teams renders in a personal chat. Nothing here
uses a feature that degrades badly on an older client: text blocks, a fact set
and one action.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .contract import (
    MESSAGE_RISK_REVIEW, NotificationBatch, NotificationMessage,
    SEVERITY_CARD_STYLE, SEVERITY_CLEAR,
)

CARD_SCHEMA = "http://adaptivecards.io/schemas/adaptive-card.json"
CARD_VERSION = "1.5"
CARD_CONTENT_TYPE = "application/vnd.microsoft.card.adaptive"

BRAND = "Trakt"
#: Matches the Teams manifest accentColor, so a card and the app chrome agree.
BRAND_COLOUR = "#1F3A5F"


def _text(text: str, *, size: Optional[str] = None,
          weight: Optional[str] = None, colour: Optional[str] = None,
          wrap: bool = True, subtle: bool = False,
          spacing: Optional[str] = None) -> Dict[str, Any]:
    block: Dict[str, Any] = {"type": "TextBlock", "text": text, "wrap": wrap}
    if size:
        block["size"] = size
    if weight:
        block["weight"] = weight
    if colour:
        block["color"] = colour
    if subtle:
        block["isSubtle"] = True
    if spacing:
        block["spacing"] = spacing
    return block


def _context_line(batch: NotificationBatch,
                  message: NotificationMessage) -> str:
    """Portfolio and dates, on one subtle line under the title.

    A combined update shows both dates explicitly. The reader must be able to
    see that the pipeline figure and the funded figure are as at different
    days without inferring it from the numbers.
    """
    parts: List[str] = [_portfolio_label(batch)]
    dates = message.provenance or {}
    if dates.get("pipeline_as_of") and dates.get("funded_as_of"):
        parts.append(f"pipeline {dates['pipeline_as_of']}")
        parts.append(f"funded {dates['funded_as_of']}")
    elif dates.get("pipeline_as_of"):
        parts.append(f"pipeline as at {dates['pipeline_as_of']}")
    elif dates.get("funded_as_of"):
        parts.append(f"funded as at {dates['funded_as_of']}")
    return "  ·  ".join(p for p in parts if p)


def _portfolio_label(batch: NotificationBatch) -> str:
    context = str(batch.portfolio_context or "total")
    return "Total portfolio" if context == "total" else context


def _update_type_label(batch: NotificationBatch) -> str:
    return {"PIPELINE": "Weekly pipeline update",
            "FUNDED": "Monthly funded update",
            "COMBINED": "Pipeline and funded update"}.get(
        batch.update_type, "Portfolio update")


def _actions(message: NotificationMessage) -> List[Dict[str, Any]]:
    """The single deep-link action, or none.

    No button at all where no workspace base URL is configured — a card whose
    only action 404s is worse than a card with no action.
    """
    if not message.deep_link:
        return []
    return [{"type": "Action.OpenUrl", "title": message.deep_link_label,
             "url": message.deep_link}]


def _provenance_block(message: NotificationMessage) -> List[Dict[str, Any]]:
    """Compact provenance as card footer text.

    Only what a reader can act on: the basis of a comparison, and the fact that
    risk items are in the governed order. Methodology detail lives behind the
    deep link rather than being reproduced on the card.
    """
    lines: List[str] = []
    provenance = message.provenance or {}
    weeks = provenance.get("five_week_weeks_observed")
    if weeks is not None:
        lines.append(f"Five-week comparison over {weeks} observed weeks.")
    if provenance.get("forecast_methodology"):
        lines.append(str(provenance["forecast_methodology"]))
    if not lines:
        return []
    return [_text(" ".join(lines), size="Small", subtle=True, spacing="Small")]


def render(batch: NotificationBatch,
           message: NotificationMessage) -> Dict[str, Any]:
    """One Adaptive Card for one message."""
    style = SEVERITY_CARD_STYLE.get(message.severity, "default")
    body: List[Dict[str, Any]] = []

    # Header — brand, then what kind of update this is.
    body.append({
        "type": "ColumnSet",
        "columns": [
            {"type": "Column", "width": "auto",
             "items": [_text(BRAND, weight="Bolder", size="Medium")]},
            {"type": "Column", "width": "stretch",
             "items": [_text(_update_type_label(batch), size="Small",
                             subtle=True, wrap=False)]},
        ],
    })

    # The severity-styled container holds the headline, so the state of the
    # message is visible before any of its text is read.
    header_items: List[Dict[str, Any]] = [
        _text(message.headline, size="Large", weight="Bolder"),
        _text(_context_line(batch, message), size="Small", subtle=True),
    ]
    if message.summary:
        header_items.append(_text(message.summary, spacing="Small"))
    body.append({
        "type": "Container",
        "style": style,
        "bleed": True,
        "items": header_items,
    })

    # Observations / risk items, as bullets.
    for item in message.items:
        body.append(_text(
            f"• {item.text}",
            colour=("Attention" if item.unavailable
                    and message.message_type == MESSAGE_RISK_REVIEW else None),
            subtle=bool(item.unavailable),
            spacing="Small"))

    body.extend(_provenance_block(message))

    return {
        "type": "AdaptiveCard",
        "$schema": CARD_SCHEMA,
        "version": CARD_VERSION,
        "body": body,
        "actions": _actions(message),
        "msteams": {"width": "Full"},
    }


def attachment(batch: NotificationBatch,
               message: NotificationMessage) -> Dict[str, Any]:
    """The Bot Framework attachment envelope around a rendered card."""
    return {"contentType": CARD_CONTENT_TYPE,
            "content": render(batch, message)}


def summary_text(batch: NotificationBatch,
                 message: NotificationMessage) -> str:
    """Fallback text for notification previews and accessibility.

    Teams shows this in the activity feed and toast, where the card itself is
    not rendered, so it has to carry the headline rather than a generic label.
    """
    if message.message_type == MESSAGE_RISK_REVIEW \
            and message.severity == SEVERITY_CLEAR:
        return f"{BRAND}: {message.headline} — no material risks identified"
    return f"{BRAND}: {message.headline}"
