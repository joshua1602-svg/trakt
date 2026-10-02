"""Conversation memory (P0 design §34, §38, §39; owner decisions D24, D25).

WHAT IS REMEMBERED (D24). The last question the agent answered, as the
complete question it answered, and an ask-back still open — nothing else.
Phase 1 carried the ask-back alone: when the governed path cannot answer
without one more detail it asks for it ("At NUTS3 the borrower's and the
property's region are different answers"), and the reply ("The property's")
is read with the question it answers. Phase 2 carries the last answered
question too, so a follow-up ("And by broker?") is read against it. The
memory is the QUESTION, never a figure: what a follow-up means is read again
from words, and every gate a first question passes runs again
(`mi_agent.interpretation_v2.conversation_reader`).

WHERE THE MEMORY LIVES. In a SIGNED token handed back with each answer and
ask-back and returned with the next message — not on the server. The API runs
as two workers with no shared store, and a token any worker can verify is the
standard answer to that; it also means nothing about a conversation outlives
it on the server. The token holds the user's own question text and the
agent's own ask, never a figure or a row, and it is signed, so it cannot be
edited into something the user did not ask.

WHAT IT IS BOUND TO. The user it was issued to, the book it was asked about
(`book_scope`: the client, the selected portfolio and lens), and the chat it
belongs to (the client starts a new chat id when the chat is
cleared), for `memory_minutes` from its issue — which is the delivery of the
answer or ask-back (D24: idle is counted from delivery). Anything else is refused and the reply is read on its own, and the
answer says the earlier question has lapsed (D24: expiry is stated, never
silent).

SWITCHED. `MI_AGENT_CONVERSATION=on` and a signing key of at least 32
characters in `MI_AGENT_CONVERSATION_KEY`; without either, no token is issued
and none is read — every request is exactly what it was before this module.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SWITCH_ENV = "MI_AGENT_CONVERSATION"
KEY_ENV = "MI_AGENT_CONVERSATION_KEY"
MIN_KEY_LENGTH = 32
#: 2 since phase 2 (§39): a token carries the last answered question as well
#: as an open ask-back. A version-1 token is refused as malformed — it can only
#: be one issued in the five minutes before the deployment that changed it.
TOKEN_VERSION = 2
TOKEN_KIND = "conversation"
#: What the agent's last message was, as the caller is told: an ask-back
#: waiting for its reply, or an answer a follow-up may build on.
KIND_ASK_BACK = "ask_back"
KIND_FOLLOW_UP = "follow_up"

_CONFIG = Path(__file__).resolve().parents[1] / "config/mi/conversation.yaml"

#: Why a returned token was not used. Stable strings: the record groups by them.
LAPSED_EXPIRED = "expired"
LAPSED_OTHER_USER = "another_user"
LAPSED_OTHER_BOOK = "another_book"
LAPSED_OTHER_CHAT = "another_chat"
LAPSED_TAMPERED = "tampered"
LAPSED_MALFORMED = "malformed"
LAPSED_TOO_MANY_ASKS = "too_many_asks"
#: The message could not be read with the earlier question (the conversation
#: reader failed, or its reading did not pass its guards).
LAPSED_UNREAD = "unread"


def _settings() -> Dict[str, Any]:
    import yaml
    try:
        return yaml.safe_load(_CONFIG.read_text(encoding="utf-8")) or {}
    except Exception:                                                # noqa: BLE001
        return {}


def memory_minutes() -> float:
    """D24: the conversation owner's setting, not a figure in code."""
    return float(_settings().get("memory_minutes") or 0)


def max_asks() -> int:
    return int(_settings().get("max_asks") or 1)


def _key() -> bytes:
    return str(os.environ.get(KEY_ENV) or "").encode("utf-8")


def enabled() -> bool:
    """On only when switched on AND a signing key is set — fail closed."""
    switched = str(os.environ.get(SWITCH_ENV) or "").strip().lower() == "on"
    return switched and len(_key()) >= MIN_KEY_LENGTH and memory_minutes() > 0


@dataclass(frozen=True)
class PendingAsk:
    """The question the agent asked back about, as the reply must read it.

    ``question`` is the user's original question; ``turns`` the asks and
    replies already exchanged about it (oldest first); ``ask`` what the agent
    is waiting for now."""

    question: str
    ask: str
    turns: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Memory:
    """What a conversation holds between two requests (D24): the last question
    the agent answered, as the complete question it answered, and an ask-back
    still open. Either may be absent; a memory with neither is no memory."""

    last: Optional[str] = None
    pending: Optional[PendingAsk] = None

    @property
    def empty(self) -> bool:
        return not self.last and self.pending is None

    @property
    def kind(self) -> str:
        return KIND_ASK_BACK if self.pending is not None else KIND_FOLLOW_UP


@dataclass(frozen=True)
class Returned:
    """What a returned token turned out to be: the memory, or why not."""

    memory: Optional[Memory] = None
    lapsed: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.memory is not None

    @property
    def pending(self) -> Optional[PendingAsk]:
        return self.memory.pending if self.memory is not None else None


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(payload: bytes) -> str:
    return _b64(hmac.new(_key(), payload, hashlib.sha256).digest())


def issue(*, principal: str, book: str, chat: Optional[str], memory: Memory,
          now: Optional[float] = None) -> Optional[str]:
    """A token for what the conversation now holds, or None when it is
    switched off or holds nothing. An ask-back past `max_asks` about one
    question is not held — the caller says so (`LAPSED_TOO_MANY_ASKS`)."""
    if not enabled():
        return None
    pending = memory.pending
    if pending is not None and len(pending.turns) >= max_asks():
        pending = None
    if not memory.last and pending is None:
        return None
    body = {"v": TOKEN_VERSION, "k": TOKEN_KIND,
            "p": str(principal or "").strip().lower(),
            "c": str(book or ""), "h": str(chat or ""),
            "t": int(time.time() if now is None else now),
            "l": str(memory.last or ""),
            "q": pending.question if pending else "",
            "a": pending.ask if pending else "",
            "r": [list(t) for t in pending.turns] if pending else []}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{_b64(payload)}.{_sign(payload)}"


def issue_ask_back(*, principal: str, book: str, chat: Optional[str],
                   pending: PendingAsk, last: Optional[str] = None,
                   now: Optional[float] = None) -> Optional[str]:
    """A token for one ask-back (with the last answered question, if any), or
    None when switched off or asked back about too often already."""
    if len(pending.turns) >= max_asks():
        return None
    return issue(principal=principal, book=book, chat=chat,
                 memory=Memory(last=last, pending=pending), now=now)


def read(token: Optional[str], *, principal: str, book: str,
         chat: Optional[str], now: Optional[float] = None) -> Optional[Returned]:
    """The memory a returned token carries, or why it is not used.

    None when no token was returned or the conversation is switched off —
    the request is then exactly a stand-alone one."""
    if not token or not enabled():
        return None
    try:
        encoded, signature = str(token).split(".", 1)
        payload = _unb64(encoded)
    except Exception:                                                # noqa: BLE001
        return Returned(lapsed=LAPSED_MALFORMED)
    if not hmac.compare_digest(_sign(payload), signature):
        return Returned(lapsed=LAPSED_TAMPERED)
    try:
        body = json.loads(payload.decode("utf-8"))
    except Exception:                                                # noqa: BLE001
        return Returned(lapsed=LAPSED_MALFORMED)
    if body.get("v") != TOKEN_VERSION or body.get("k") != TOKEN_KIND:
        return Returned(lapsed=LAPSED_MALFORMED)
    if body.get("p") != str(principal or "").strip().lower():
        return Returned(lapsed=LAPSED_OTHER_USER)
    if body.get("c") != str(book or ""):
        return Returned(lapsed=LAPSED_OTHER_BOOK)
    if body.get("h") != str(chat or ""):
        return Returned(lapsed=LAPSED_OTHER_CHAT)
    age = (time.time() if now is None else now) - float(body.get("t") or 0)
    if age < 0 or age > memory_minutes() * 60:
        return Returned(lapsed=LAPSED_EXPIRED)
    turns = tuple((str(a), str(r)) for a, r in (body.get("r") or ()))
    pending = (PendingAsk(question=str(body.get("q") or ""),
                          ask=str(body.get("a") or ""), turns=turns)
               if body.get("q") else None)
    memory = Memory(last=str(body.get("l") or "") or None, pending=pending)
    if memory.empty:
        return Returned(lapsed=LAPSED_MALFORMED)
    return Returned(memory=memory)


def book_scope(client_id: Optional[str], portfolio_id: Optional[str],
               lens: Any = None) -> str:
    """The book a question was asked about, as a token binds it: a reply about
    another client, portfolio or lens is not read with it (D24: nothing
    carries across books)."""
    lens_text = (",".join(sorted(str(x) for x in lens)) if isinstance(lens, (list, tuple))
                 else str(lens or ""))
    return f"{client_id or ''}|{portfolio_id or ''}|{lens_text}"


def lapsed_notice(reason: str) -> str:
    """D24: a reply whose earlier question could not be used is answered on
    its own, and says so."""
    if reason == LAPSED_EXPIRED:
        minutes = memory_minutes()
        span = f"{minutes:g} minute{'s' if minutes != 1 else ''}"
        return (f"More than {span} passed since I asked, so the earlier question "
                f"has lapsed and I have read this on its own.")
    if reason == LAPSED_TOO_MANY_ASKS:
        return ("That question needed more detail than I can gather one reply at "
                "a time, so I have read this on its own; please ask it again in "
                "full.")
    if reason == LAPSED_UNREAD:
        return ("I could not read this together with your earlier question, so "
                "I have read it on its own.")
    return ("The earlier question could not be used here, so I have read this on "
            "its own.")
