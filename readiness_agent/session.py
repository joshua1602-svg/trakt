"""readiness_agent.session — the only door an agent goes through.

This is the machine-facing surface a Securitisation Readiness Agent uses, and
it is deliberately narrow. An agent gets three verbs:

    capabilities()   what does Trakt know about this portfolio, and why not?
    call(tool, args) execute one governed tool
    transcript()     what did I do, and what did Trakt return?

and nothing else. In particular it gets **no** DataFrame, no path to a CSV or
Parquet file, no private analytics function, no storage handle and no way to
widen its own authority. Everything goes through
``trakt_tools.execution.execute_governed_tool``, which is the same entry point
an external client-owned agent would reach over HTTP — so the reference agent
built on this surface has no advantage that a third-party agent would lack.

Why that restraint matters more than it looks
---------------------------------------------
An agent handed a DataFrame will compute. It will take a mean of a rate column,
or filter to a segment and divide two sums, and it will do so fluently and
wrongly — the exact failure Sprint 2.5E found in MI Query, where a *supplied*
LGD estimate answered a question about *realised* severity. Denying the agent
raw data is not a security measure here; it is a correctness measure. If the
only way to get a number is to ask Trakt for it, then every number in the
conclusion carries Trakt's methodology and provenance with it.

What this class does NOT do
---------------------------
It holds no investigation strategy, no metric preferences and no ordering. It
does not decide what to look at, and it must never acquire a default sequence
of calls — the moment it has one, the agent above it is executing a checklist
and the autonomy claim is dead. Choosing what to ask is the agent's job and
this module's non-job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from trakt_core.context import ExecutionContext
from trakt_core.envelope import STATUS_SUCCESS
from trakt_tools.execution import ToolDependencies, execute_governed_tool


@dataclass(frozen=True)
class ToolCallRecord:
    """One governed call, as it will appear in the audit trail.

    Deliberately records *actions and governed results*, never the agent's
    reasoning. A stored chain of thought is a liability — it reads as though
    it were evidence, and it is not: only the tool inputs and Trakt's outputs
    are reconstructible facts about what happened.
    """

    sequence: int
    tool: str
    arguments: Dict[str, Any]
    status: str
    elapsed_ms: float
    error_code: Optional[str] = None
    #: Small, quotable summary of the governed result — enough to reconstruct
    #: the finding without storing an entire loan tape in the audit record.
    result_digest: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence, "tool": self.tool,
            "arguments": self.arguments, "status": self.status,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "error_code": self.error_code,
            "result_digest": self.result_digest,
        }


class GovernedSession:
    """An authorised agent's connection to Trakt."""

    def __init__(self, context: ExecutionContext, resource: str, *,
                 dependencies: Optional[ToolDependencies] = None) -> None:
        self._context = context
        self.resource = resource
        self._resource = resource
        self._dependencies = dependencies
        self._calls: List[ToolCallRecord] = []
        self._capability_cache: Optional[Dict[str, Any]] = None

    # -- discovery ---------------------------------------------------------- #
    def capabilities(self, *, refresh: bool = False) -> Dict[str, Any]:
        """What Trakt knows about this portfolio, and why not where it does not.

        Cached for the life of the session unless explicitly refreshed. An
        agent re-discovering capabilities between every question is a common
        and expensive failure mode, and the cache makes it free rather than
        forbidden — the agent is not prevented from asking again, it simply
        does not pay twice for the same answer.
        """
        if self._capability_cache is None or refresh:
            outcome = self.call("portfolio_capabilities",
                                {"resource": self._resource,
                                 "include_definition": True})
            self._capability_cache = outcome
        return self._capability_cache

    def available_metrics(self) -> List[str]:
        return [m["metric"] for m in self.capabilities().get("metrics", [])
                if m.get("status") == "AVAILABLE"]

    def unavailable_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Everything Trakt will not produce here, with its reason. The agent
        needs this as much as it needs the available set: a readiness review
        that cannot say what it was unable to assess is not a review."""
        return {m["metric"]: m for m in self.capabilities().get("metrics", [])
                if m.get("status") != "AVAILABLE"}

    # -- execution ---------------------------------------------------------- #
    def call(self, tool: str, arguments: Optional[Mapping[str, Any]] = None
             ) -> Dict[str, Any]:
        """Execute one governed tool. Refusals are returned, not raised.

        A refused call is information — "this portfolio has no contractual
        WAL" is a finding — so it comes back as a payload the agent can reason
        about rather than an exception it has to catch and interpret.
        """
        args = dict(arguments or {})
        args.setdefault("resource", self._resource)

        started = time.perf_counter()
        result = execute_governed_tool(tool, args, self._context,
                                       dependencies=self._dependencies)
        elapsed = (time.perf_counter() - started) * 1000.0

        payload: Dict[str, Any]
        if result.status == STATUS_SUCCESS:
            payload = dict(result.result or {})
            error_code = None
        else:
            error = result.error
            error_code = getattr(error, "code", None)
            error_code = getattr(error_code, "value", error_code)
            payload = {
                "refused": True,
                "error_code": str(error_code) if error_code else "REFUSED",
                "message": getattr(error, "message", "") or "",
            }

        self._calls.append(ToolCallRecord(
            sequence=len(self._calls) + 1, tool=tool, arguments=args,
            status=result.status, elapsed_ms=elapsed,
            error_code=str(error_code) if error_code else None,
            result_digest=_digest(payload)))
        return payload

    # -- audit -------------------------------------------------------------- #
    def transcript(self) -> List[Dict[str, Any]]:
        return [call.to_dict() for call in self._calls]

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def efficiency(self) -> Dict[str, Any]:
        """Tool-call economics, for the evaluation harness.

        ``repeated_calls`` counts identical (tool, arguments) pairs asked more
        than once — the cheapest signal that an agent is looping rather than
        investigating.
        """
        seen: Dict[str, int] = {}
        for call in self._calls:
            key = f"{call.tool}:{sorted(call.arguments.items())!r}"
            seen[key] = seen.get(key, 0) + 1
        return {
            "total_calls": len(self._calls),
            "distinct_calls": len(seen),
            "repeated_calls": sum(count - 1 for count in seen.values()
                                  if count > 1),
            "total_elapsed_ms": round(
                sum(c.elapsed_ms for c in self._calls), 2),
            "refused_calls": sum(1 for c in self._calls
                                 if c.status != STATUS_SUCCESS),
            "by_tool": _count_by_tool(self._calls),
        }

    def audit_record(self, objective: str) -> Dict[str, Any]:
        """Everything needed to reconstruct the run, and nothing that isn't.

        Objective, actor, organisation, portfolio, the capabilities discovered,
        every governed call with its inputs and a digest of its result. No
        chain of thought: what the agent *did* is auditable, what it *thought*
        is not evidence.
        """
        return {
            "objective": objective,
            "actor": getattr(self._context, "user_id", None),
            "organisation": getattr(self._context, "tenant_id", None),
            "request_id": getattr(self._context, "request_id", None),
            "resource": self._resource,
            "capabilities_discovered": (
                self._capability_cache.get("summary")
                if self._capability_cache else None),
            "tool_calls": self.transcript(),
            "efficiency": self.efficiency(),
        }


_DIGEST_KEYS = (
    "available", "status", "reason", "reason_code", "method",
    "methodology_version", "authority", "value", "metric", "summary",
    "wal_years", "ytm_pct", "cpr_pct", "smm_pct", "annualised_cdr_pct",
    "mean_cure_rate_pct", "default_stock_pct", "stale_balance_pct",
    "total_balance", "loan_count", "outcome", "explanation",
)


def _digest(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """A quotable summary of a governed result.

    Bounded on purpose: an audit record that embedded whole loan tapes would
    be unreadable and would quietly become a second copy of the data.
    """
    digest: Dict[str, Any] = {}
    for key in _DIGEST_KEYS:
        if key in payload:
            value = payload[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                digest[key] = value
            elif isinstance(value, dict):
                digest[key] = {k: v for k, v in list(value.items())[:8]
                               if isinstance(v, (str, int, float, bool))}
    for collection in ("metrics", "per_period", "groups", "rows"):
        if isinstance(payload.get(collection), list):
            digest[f"{collection}_count"] = len(payload[collection])
    return digest


def _count_by_tool(calls: List[ToolCallRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for call in calls:
        counts[call.tool] = counts.get(call.tool, 0) + 1
    return counts
