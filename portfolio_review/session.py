"""The Portfolio Review Agent's door — a narrowed :class:`GovernedSession`.

Two jobs, both of which the red-team showed a prompt cannot do.

**Refuse what is out of mandate.** ``mandate.tool_schemas()`` already withholds
the readiness tools, so an obedient model never learns they exist. That is not
enough on its own: a model that has read a tool name anywhere — a docstring, a
governed warning, its own earlier turn — can still emit the call. This refuses
it here, before ``execute_governed_tool`` is reached, and returns the refusal as
a payload the agent can reason about. Offering fewer tools is what usually
works; refusing the call is what makes it true.

**Keep the evidence the numeric gate needs.** ``GovernedSession.transcript()``
keeps a *digest* of each result, which is right for an audit record and useless
for grounding: to decide whether £1.88m came from Trakt you need the numbers
Trakt returned, at full precision, with the field they came from. This keeps
them, indexed, and hands the index to :mod:`portfolio_review.numeric_gate`.

It adds no capability. Every call still goes through the wrapped session, which
still goes through the same governed execution path an external client agent
would reach over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from . import mandate
from .numeric_gate import GovernedIndex

#: What the agent is told when it names a tool outside the mandate. Phrased as a
#: scope statement rather than an error because it IS one: the call was refused
#: because this agent does not do that, not because anything went wrong. The
#: agent is expected to note it and carry on, and the phrasing that produced
#: that behaviour is the phrasing that names the owning agent.
_REFUSAL = (
    "OUT_OF_MANDATE: `{tool}` is not available to the Portfolio Review Agent. "
    "{reason} This belongs to the {owner}. Do not report on it, do not "
    "estimate what it would have said, and do not list it as a gap in this "
    "review — it is out of scope rather than unavailable."
)

_UNKNOWN = (
    "OUT_OF_MANDATE: `{tool}` is not a tool the Portfolio Review Agent can "
    "call. Use one of the governed MI tools you were given."
)


class MIScopedSession:
    """A governed session narrowed to the Portfolio Review mandate."""

    def __init__(self, session: Any):
        self._session = session
        self.index = GovernedIndex()
        #: Every call attempted, in order, with whether the mandate allowed it.
        #: This is §15's tool-call audit, produced as a by-product of enforcing
        #: the thing it audits rather than reconstructed afterwards.
        self.attempts: List[Dict[str, Any]] = []

    # -- the three verbs, narrowed ------------------------------------------ #
    def call(self, tool: str, arguments: Optional[Mapping[str, Any]] = None
             ) -> Dict[str, Any]:
        args = dict(arguments or {})
        if not mandate.is_allowed(tool):
            payload = {"refused": True, "error_code": "OUT_OF_MANDATE",
                       "message": self._refusal_text(tool)}
            self.attempts.append({"tool": tool, "arguments": args,
                                  "allowed": False, "executed": False,
                                  "result": payload})
            return payload

        payload = self._session.call(tool, args)
        self.index.absorb(tool, payload)
        self.attempts.append({"tool": tool, "arguments": args, "allowed": True,
                              "executed": True, "result": payload})
        return payload

    def capabilities(self) -> Dict[str, Any]:
        return self._session.capabilities()

    def transcript(self) -> List[Dict[str, Any]]:
        return self._session.transcript()

    # -- audit -------------------------------------------------------------- #
    def out_of_mandate_calls(self) -> List[Dict[str, Any]]:
        """Attempts the mandate refused. Empty is the expected result."""
        return [a for a in self.attempts if not a["allowed"]]

    def tool_call_audit(self) -> List[Dict[str, Any]]:
        """§15, one row per attempt."""
        return [{
            "tool": a["tool"],
            "in_allow_list": a["allowed"],
            "executed": a["executed"],
            "refused": bool((a["result"] or {}).get("refused")),
        } for a in self.attempts]

    @staticmethod
    def _refusal_text(tool: str) -> str:
        exclusion = mandate.exclusion_for(tool)
        if exclusion is None:
            return _UNKNOWN.format(tool=tool)
        return _REFUSAL.format(tool=tool, reason=exclusion.reason,
                               owner=exclusion.belongs_to)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)
