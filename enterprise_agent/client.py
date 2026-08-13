"""enterprise_agent.client — a general enterprise agent that knows no finance.

This is the *caller* side of the Sprint 4 proof, and its value is entirely in
what it does **not** contain. It has no securitisation methodology, no metric
names, no thresholds, no investigation plan and no idea what makes a portfolio
ready or unready. It knows three things a real enterprise agent would know:

    the objective its user gave it,
    the portfolio identifier its user named,
    the organisation it acts for.

Everything else it learns from the Agent Card at delegation time.

Why this shape is the proof
---------------------------
If the caller had to name the specific measures and thresholds a readiness
review turns on, Trakt would not be a specialist — it would be a remote
procedure with extra steps, and every change to Trakt's methodology would break
its callers. The claim being tested is that a generic agent can delegate an
*outcome* it cannot itself compute, and get back something it can act on without
understanding how it was produced.

``tests/test_a2a_delegation.py`` reads this whole file — code and prose alike —
and fails if a governed tool name, a methodology identifier, a domain term or an
unexpected numeric constant appears anywhere in it. Scanning the prose too is
deliberate: it costs one careful docstring and it closes the loophole where
knowledge creeps in as a comment and is later promoted to code.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional


@dataclass
class DelegationOutcome:
    """What the enterprise agent got back, in terms it can act on."""

    task_id: str
    context_id: str
    state: str
    #: The specialist's structured deliverable, passed through untouched. The
    #: caller does not interpret the finance; it routes it.
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

    @property
    def succeeded(self) -> bool:
        return self.state == "completed"

    @property
    def headline(self) -> Optional[str]:
        """The one field a generic agent can legitimately read.

        It is a category the specialist assigned, not a number the caller
        computed. Routing on it — escalate, notify, file — is generic
        behaviour; interpreting it would not be.
        """
        return (self.result or {}).get("overallReadiness")


class EnterpriseAgent:
    """A generic client-side agent that delegates work it cannot do itself.

    ``transport`` is any callable taking a JSON-RPC request dict and returning a
    JSON-RPC response dict. In this proof it is an in-process function; in a
    deployment it would be an HTTPS POST. The agent cannot tell the difference,
    which is the point of using the protocol rather than a Python import.
    """

    def __init__(self, *, organisation: str,
                 transport: Callable[[Mapping[str, Any]], Mapping[str, Any]],
                 card_fetcher: Callable[[], Mapping[str, Any]]) -> None:
        self.organisation = organisation
        self._transport = transport
        self._fetch_card = card_fetcher
        self._card: Optional[Dict[str, Any]] = None

    # -- discovery ---------------------------------------------------------- #
    def discover(self) -> Dict[str, Any]:
        """Read the remote agent's card. This is the only thing that tells this
        agent what the remote one can do."""
        self._card = dict(self._fetch_card())
        return self._card

    def capabilities(self) -> List[Dict[str, Any]]:
        card = self._card or self.discover()
        return list(card.get("skills") or [])

    def find_skill_for(self, need: str) -> Optional[Dict[str, Any]]:
        """Match a plain-language need against advertised skills.

        Matching is on the card's own ``tags``, ``name`` and ``description`` —
        text the remote agent published about itself. The caller contributes no
        vocabulary of its own, which is why this stays honest: it is not
        recognising securitisation, it is recognising that words in the user's
        request appear in what the remote agent says it does.
        """
        words = {w for w in _words(need) if len(w) > 3}
        best, best_score = None, 0
        for skill in self.capabilities():
            published = set(_words(" ".join([
                skill.get("name", ""), skill.get("description", ""),
                " ".join(skill.get("tags") or []),
            ])))
            score = len(words & published)
            if score > best_score:
                best, best_score = skill, score
        return best

    def endpoint(self) -> Optional[str]:
        """Where to send work, and over which binding — from the card."""
        card = self._card or self.discover()
        for interface in card.get("supportedInterfaces") or []:
            if interface.get("protocolBinding") == "JSONRPC":
                return interface.get("url")
        return None

    def authentication_required(self) -> List[str]:
        card = self._card or self.discover()
        return sorted({scheme for requirement in
                       card.get("securityRequirements") or []
                       for scheme in requirement})

    # -- delegation --------------------------------------------------------- #
    def delegate(self, objective: str, *, portfolio: str,
                 context_id: Optional[str] = None,
                 task_id: Optional[str] = None) -> DelegationOutcome:
        """Hand an objective to the specialist and return what came back.

        The objective is passed through **verbatim**. This agent does not
        decompose it into steps, and it must not: choosing the investigation is
        the specialist's job, and a caller that sent a plan would be running the
        specialist rather than delegating to it.
        """
        request = {
            "jsonrpc": "2.0",
            "id": _new_id("rpc"),
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": _new_id("msg"),
                    "role": "user",
                    "parts": [{"kind": "text", "text": objective}],
                    "metadata": {"resource": portfolio},
                    **({"contextId": context_id} if context_id else {}),
                    **({"taskId": task_id} if task_id else {}),
                },
            },
        }
        return self._send(request)

    def ask_follow_up(self, question: str, *, context_id: str
                      ) -> DelegationOutcome:
        """Ask about work already done, in the same conversation."""
        return self.delegate(question, portfolio="", context_id=context_id)

    def poll(self, task_id: str) -> DelegationOutcome:
        return self._send({"jsonrpc": "2.0", "id": _new_id("rpc"),
                           "method": "tasks/get", "params": {"id": task_id}})

    # -- transport ---------------------------------------------------------- #
    def _send(self, request: Mapping[str, Any]) -> DelegationOutcome:
        response = self._transport(request)
        if "error" in response:
            error = dict(response["error"])
            return DelegationOutcome(task_id="", context_id="", state="failed",
                                     error=error)
        task = response.get("result") or {}
        return DelegationOutcome(
            task_id=task.get("id", ""),
            context_id=task.get("contextId", ""),
            state=(task.get("status") or {}).get("state", "unknown"),
            result=_first_data_part(task),
            error=(task.get("status") or {}).get("message"))


def _first_data_part(task: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    for artifact in task.get("artifacts") or []:
        for part in artifact.get("parts") or []:
            if part.get("kind") == "data":
                return dict(part.get("data") or {})
    return None


def _words(text: str) -> List[str]:
    return [w for w in "".join(
        c.lower() if c.isalnum() else " " for c in text).split() if w]


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
