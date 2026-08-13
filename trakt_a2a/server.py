"""trakt_a2a.server — the A2A boundary in front of the specialist.

This module is a **delegation boundary**, and the discipline that makes it one is
that it contains no analytics. It receives an objective, decides whether it is
authorised and in scope, hands it to the Sprint 3 Securitisation Readiness Agent
over MCP, and serialises what comes back. Grep it for arithmetic and you will
find none; the tests assert that rather than trusting the grep.

The three things it does own
----------------------------
**Authorisation, in two separate steps.** A valid A2A token establishes *which
agent is calling*. It does not establish *what that agent may read*. Those are
different questions with different answers, and collapsing them is the failure
mode this whole layer exists to avoid — so ``_authorise`` resolves the caller's
identity first and the portfolio entitlement second, and a caller that passes the
first and fails the second gets a refusal that says so.

**Scope.** The card advertises one skill with stated limits. A request for a
credit rating is refused with ``rejected`` and the reason, rather than answered
with something adjacent. An agent that quietly does the nearest thing it can is
worse than one that declines.

**Task state.** An assessment takes minutes, so it is a Task with a lifecycle a
caller can poll, and its result is an Artifact rather than prose.

What it deliberately does not do
--------------------------------
It does not summarise. The structured assessment crosses this boundary intact —
every finding keeps its fact, its rule and *whose* rule it is, its severity and
its evidence. A wrapper that flattened that into a paragraph would be a lossy
layer wearing a protocol's clothes, and ``tests/test_a2a_evidence.py`` fails if
it becomes one.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from trakt_core.context import ExecutionContext

from .card import ARTIFACT_MEDIA_TYPE, SKILL_ID, agent_card
from .tasks import (AUTH_REQUIRED, COMPLETED, FAILED, REJECTED, SUBMITTED,
                    WORKING, Task, TaskStore)

#: JSON-RPC error codes. The -32000 range is the specification's
#: implementation-defined space; each is mapped to exactly one condition so a
#: caller can branch on the code rather than parse the message.
ERR_INVALID_PARAMS = -32602
ERR_TASK_NOT_FOUND = -32001
ERR_UNAUTHENTICATED = -32002
ERR_FORBIDDEN = -32003
ERR_INTERNAL = -32603

#: Phrases that name work this specialist does not do. Deliberately a small,
#: explicit list rather than a classifier: the limits are published in the Agent
#: Card, so a caller can predict a refusal instead of discovering it, and a
#: reviewer can read every case in one screen. It is keyword matching and is
#: labelled as such — a semantic classifier here would be a second reasoning
#: system inside a transport boundary.
OUT_OF_SCOPE = (
    ("credit rating", "Issuing or predicting a credit rating. This agent is "
                      "not a rating agency and the Agent Card says so."),
    ("rate this portfolio", "Issuing a credit rating."),
    ("assign a rating", "Issuing a credit rating."),
    ("probability of default", "Producing PD estimates. This agent reports "
                               "observed default behaviour, not modelled "
                               "probabilities."),
    ("expected loss", "Producing expected-loss or ECL estimates."),
    ("loss given default estimate", "Producing an LGD model estimate."),
    ("stress test", "Running scenarios or stress tests."),
    ("scenario analysis", "Running scenarios or stress tests."),
    ("forecast", "Forecasting future portfolio behaviour."),
    ("expected wal", "Producing an expected (behavioural) weighted-average "
                     "life. Contractual life is available; expected life "
                     "requires a behavioural model this agent does not build."),
)


class A2AError(Exception):
    """An error that is safe to send across the boundary.

    Carries a code and a bounded message and nothing else. Anything that
    escapes as a bare exception is caught at the edge and replaced with a
    generic internal error, because a stack trace crossing an agent boundary
    tells a stranger about our filesystem.
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_rpc(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message}


@dataclass
class CallerIdentity:
    """Who is calling, resolved from the transport — never from the body.

    ``organisation_id`` is the entitlement subject. ``agent_id`` is the machine.
    They are separate fields because they answer separate questions, and a
    deployment that conflated them would let any agent of any organisation read
    any other's book.
    """

    agent_id: str
    organisation_id: str
    context: ExecutionContext

    @property
    def audit_identity(self) -> Dict[str, Any]:
        return {
            "calling_agent": self.agent_id,
            "organisation": self.organisation_id,
            "actor": getattr(self.context, "actor_id", None),
            "actor_type": getattr(self.context, "actor_type", None),
            "channel": getattr(self.context, "channel", None),
        }


class SecuritisationReadinessA2AServer:
    """The Trakt specialist, addressable over A2A.

    ``assessor`` and ``session_factory`` are injected rather than imported so
    that the boundary can be tested without spending money on a model, and so
    that this module has no import path to Trakt's analytics at all.
    """

    def __init__(self, *,
                 endpoint_url: str,
                 session_factory: Callable[..., Any],
                 assessor: Callable[..., Any],
                 authoriser: Callable[[CallerIdentity, str], None],
                 store: Optional[TaskStore] = None,
                 streaming: bool = False) -> None:
        self._endpoint_url = endpoint_url
        self._session_factory = session_factory
        self._assessor = assessor
        self._authoriser = authoriser
        self._store = store or TaskStore()
        self._streaming = streaming
        #: contextId -> the last completed assessment, so a follow-up is
        #: answered from evidence already produced rather than by re-running a
        #: portfolio the caller has already paid to assess.
        self._assessments: Dict[str, Dict[str, Any]] = {}
        #: taskId -> the specialist's governed-call transcript, so a
        #: completed delegation stays auditable after the run object is gone.
        self._transcripts: Dict[str, List[Dict[str, Any]]] = {}

    # -- discovery ---------------------------------------------------------- #
    @property
    def store(self) -> TaskStore:
        return self._store

    def agent_card(self) -> Dict[str, Any]:
        return agent_card(endpoint_url=self._endpoint_url,
                          streaming=self._streaming)

    # -- JSON-RPC ----------------------------------------------------------- #
    def handle(self, request: Mapping[str, Any],
               caller: Optional[CallerIdentity]) -> Dict[str, Any]:
        """One JSON-RPC 2.0 request. Never raises across the boundary."""
        rpc_id = request.get("id")
        try:
            method = request.get("method")
            params = request.get("params") or {}
            if caller is None:
                raise A2AError(ERR_UNAUTHENTICATED,
                               "Authentication required. Present an enterprise "
                               "agent token carrying the Trakt.Agent role.")
            if method == "message/send":
                result = self.message_send(params, caller)
            elif method == "tasks/get":
                result = self.tasks_get(params, caller)
            else:
                raise A2AError(ERR_INVALID_PARAMS,
                               f"Unsupported method: {method!r}")
            return {"jsonrpc": "2.0", "id": rpc_id, "result": result}
        except A2AError as exc:
            return {"jsonrpc": "2.0", "id": rpc_id, "error": exc.to_rpc()}
        except Exception:  # noqa: BLE001 — deliberate boundary catch
            # Bounded on purpose. The detail belongs in our logs, not in a
            # stranger's error handler.
            return {"jsonrpc": "2.0", "id": rpc_id,
                    "error": {"code": ERR_INTERNAL,
                              "message": "The specialist agent failed to "
                                         "complete the request."}}

    # -- message/send ------------------------------------------------------- #
    def message_send(self, params: Mapping[str, Any],
                     caller: CallerIdentity) -> Dict[str, Any]:
        message = params.get("message") or {}
        objective = _text_of(message)
        if not objective:
            raise A2AError(ERR_INVALID_PARAMS,
                           "message.parts must contain a text part stating the "
                           "objective.")

        context_id = (message.get("contextId")
                      or params.get("contextId") or _new_id("ctx"))

        # A follow-up on an existing conversation is answered from the
        # assessment already produced. Re-running would be a second charge for
        # a question the evidence already answers.
        if context_id in self._assessments and not _wants_new_assessment(params):
            return self._follow_up(objective, context_id, caller)

        task_id = message.get("taskId") or params.get("taskId") or _new_id("task")

        existing = self._store.get(task_id)
        if existing is not None:
            # Idempotency: the same taskId is the same instruction. A retried
            # request returns the task, it does not start a second assessment.
            return existing.to_dict()

        resource = _resource_of(params, message)
        if not resource:
            raise A2AError(ERR_INVALID_PARAMS,
                           "A portfolio identifier is required. Supply it as "
                           "metadata.resource on the message.")

        task = self._store.create(task_id, context_id, metadata={
            "skill": SKILL_ID, "resource": resource,
            **caller.audit_identity})
        task.messages.append(_user_message(objective, context_id, task_id))

        reason = _out_of_scope(objective)
        if reason:
            task.transition(REJECTED, message=reason)
            task.error = {"reason": reason,
                          "advertisedSkill": SKILL_ID}
            return task.to_dict()

        try:
            self._authoriser(caller, resource)
        except A2AError as exc:
            state = AUTH_REQUIRED if exc.code == ERR_UNAUTHENTICATED else REJECTED
            task.transition(state, message=exc.message)
            task.error = {"reason": exc.message, "code": exc.code}
            # Raised rather than returned: an authorisation failure is an error
            # for the caller to handle, not a result to read.
            raise

        task.transition(WORKING)
        try:
            # The task id becomes the correlation id for everything below this
            # line — the specialist's run, its MCP calls, and every governed
            # execution's audit event. One identifier, chosen at the boundary
            # where the work was requested, is what makes the run
            # reconstructable end to end without a tracing system.
            session = _make_session(self._session_factory, caller, resource,
                                    correlation_id=task.id)
            run = self._assessor(session)
        except Exception as exc:  # noqa: BLE001
            task.transition(FAILED, message="assessment did not complete")
            task.error = {"reason": _bounded(exc)}
            return task.to_dict()

        assessment = getattr(run, "assessment", None)
        if not assessment:
            task.transition(FAILED, message="the specialist produced no "
                                            "assessment")
            task.error = {"reason": getattr(run, "stopped_reason", "") or
                                    "no assessment produced"}
            return task.to_dict()

        self._assessments[context_id] = assessment
        task.add_artifact(readiness_artifact(assessment, run=run,
                                             resource=resource))
        # Kept beside the task so an auditor can reconstruct the delegation
        # without the specialist's process still being alive.
        task.metadata["governedCalls"] = len(getattr(run, "transcript", []) or [])
        # The specialist's own wall time, so protocol overhead can be
        # computed as (total - this) rather than (total - governed time).
        # Those differ by the whole of the model's thinking, and reporting
        # the second as "protocol overhead" would blame A2A for the LLM.
        task.metadata["specialistElapsedS"] = getattr(run, "elapsed_s", None)
        task.metadata["usage"] = dict(getattr(run, "usage", {}) or {})
        self._transcripts[task.id] = list(getattr(run, "transcript", []) or [])
        task.transition(COMPLETED)
        return task.to_dict()

    def transcript_for(self, task_id: str) -> List[Dict[str, Any]]:
        """Every governed call the specialist made for one task.

        Actions and governed results only — the audit trail deliberately holds
        no model reasoning, because what the agent *did* is a reconstructable
        fact and what it *thought* is not evidence.
        """
        return list(self._transcripts.get(task_id, []))

    # -- tasks/get ---------------------------------------------------------- #
    def tasks_get(self, params: Mapping[str, Any],
                  caller: CallerIdentity) -> Dict[str, Any]:
        task_id = params.get("id") or params.get("taskId")
        task = self._store.get(task_id) if task_id else None
        if task is None:
            raise A2AError(ERR_TASK_NOT_FOUND, f"No such task: {task_id!r}")
        if task.metadata.get("organisation") != caller.organisation_id:
            # Not "not found" — the caller is authenticated and this is a real
            # task; it is simply not theirs. Saying so is honest and leaks
            # nothing, because they already supplied the id.
            raise A2AError(ERR_FORBIDDEN,
                           "This task belongs to another organisation.")
        return task.to_dict()

    # -- follow-up ---------------------------------------------------------- #
    def _follow_up(self, question: str, context_id: str,
                   caller: CallerIdentity) -> Dict[str, Any]:
        """Answer from the assessment already produced in this context.

        Deterministic retrieval, not a second reasoning pass. It returns
        governed evidence the specialist already committed to; it does not
        compose a new conclusion, because a new conclusion reached without new
        evidence is exactly the thing this architecture exists to prevent.
        """
        assessment = self._assessments[context_id]
        task_id = _new_id("task")
        task = self._store.create(task_id, context_id, metadata={
            "skill": SKILL_ID, "followUp": True, **caller.audit_identity})
        task.messages.append(_user_message(question, context_id, task_id))
        task.transition(WORKING)

        answer = _answer_from_assessment(question, assessment)
        task.add_artifact({
            "artifactId": _new_id("artifact"),
            "name": "follow-up-answer",
            "description": "Answered from the completed assessment in this "
                           "context. No portfolio was re-run.",
            "parts": [{"kind": "data", "data": answer,
                       "mediaType": ARTIFACT_MEDIA_TYPE}],
        })
        task.transition(COMPLETED)
        return task.to_dict()


# --------------------------------------------------------------------------- #
# Artifact construction — structure preserved, nothing summarised
# --------------------------------------------------------------------------- #
def readiness_artifact(assessment: Mapping[str, Any], *, run: Any = None,
                       resource: str = "") -> Dict[str, Any]:
    """The Sprint 3 assessment, as an A2A Artifact.

    Every field crosses intact. The temptation at a protocol boundary is to
    render a summary because it looks tidier on the wire; the cost is that the
    consuming agent can no longer tell a measured fact from a supplied estimate,
    or a warehouse limit from a rating-agency requirement. Those distinctions are
    the product.
    """
    data: Dict[str, Any] = {
        "skill": SKILL_ID,
        "resource": resource,
        "overallReadiness": assessment.get("overall_readiness"),
        "summary": assessment.get("summary"),
        "strengths": list(assessment.get("strengths") or []),
        "materialFindings": [_finding(f)
                             for f in assessment.get("material_findings") or []],
        "couldNotAssess": [dict(g)
                           for g in assessment.get("could_not_assess") or []],
        "furtherDiligence": list(assessment.get("further_diligence") or []),
    }
    if run is not None:
        # Provenance, not telemetry-for-its-own-sake: it lets the caller
        # reconstruct which governed calls produced the conclusion.
        data["evidenceTrail"] = {
            "governedCalls": (run.efficiency or {}).get("total_calls"),
            "distinctCalls": (run.efficiency or {}).get("distinct_calls"),
            "toolsUsed": sorted((run.efficiency or {}).get("by_tool", {})),
            "stoppedReason": run.stopped_reason,
        }
    return {
        "artifactId": _new_id("artifact"),
        "name": "securitisation-readiness-assessment",
        "description": "Structured readiness assessment with evidence.",
        "parts": [{"kind": "data", "data": data,
                   "mediaType": ARTIFACT_MEDIA_TYPE}],
    }


def _finding(finding: Mapping[str, Any]) -> Dict[str, Any]:
    """One material finding, with its evidence model intact.

    ``fact`` / ``rule`` / ``judgement`` stay separate across the wire. Merging
    them into a sentence would destroy the one distinction a consuming agent
    most needs: what Trakt measured, whose threshold applies, and what is the
    specialist's opinion.
    """
    return {
        "title": finding.get("title"),
        "fact": finding.get("fact"),
        "rule": finding.get("rule"),
        "judgement": finding.get("judgement"),
        "severity": finding.get("severity"),
        "evidenceTools": list(finding.get("evidence_tools") or []),
    }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_session(factory: Callable[..., Any], caller: "CallerIdentity",
                  resource: str, *, correlation_id: str) -> Any:
    """Build the specialist's session, passing the correlation id when the
    factory accepts one.

    Tolerant of a two-argument factory on purpose: a deployment that has no
    correlation plumbing should still work, and should not be forced to
    accept a parameter it will ignore.
    """
    try:
        return factory(caller, resource, correlation_id=correlation_id)
    except TypeError:
        return factory(caller, resource)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _text_of(message: Mapping[str, Any]) -> str:
    parts = message.get("parts") or []
    texts = [p.get("text", "") for p in parts
             if isinstance(p, Mapping) and p.get("kind", "text") == "text"]
    return "\n".join(t for t in texts if t).strip()


def _resource_of(params: Mapping[str, Any], message: Mapping[str, Any]) -> str:
    for source in (message.get("metadata") or {}, params.get("metadata") or {}):
        value = source.get("resource") or source.get("portfolio")
        if value:
            return str(value)
    return ""


def _wants_new_assessment(params: Mapping[str, Any]) -> bool:
    metadata = params.get("metadata") or {}
    return bool(metadata.get("newAssessment"))


def _user_message(text: str, context_id: str, task_id: str) -> Dict[str, Any]:
    return {"messageId": _new_id("msg"), "role": "user", "taskId": task_id,
            "contextId": context_id, "parts": [{"kind": "text", "text": text}]}


def _out_of_scope(objective: str) -> str:
    lowered = objective.lower()
    for phrase, reason in OUT_OF_SCOPE:
        if phrase in lowered:
            return (f"Outside the advertised skill: {reason} "
                    f"This agent performs securitisation readiness assessment "
                    f"only.")
    return ""


def _bounded(exc: Exception) -> str:
    """A failure description safe to send to a stranger."""
    return f"{type(exc).__name__} while executing the assessment"


def _answer_from_assessment(question: str, assessment: Mapping[str, Any]
                            ) -> Dict[str, Any]:
    """Retrieve, do not reason.

    Two question shapes are recognised because the Agent Card advertises them by
    example. Anything else gets the whole structured assessment and is told
    plainly that the question was not specifically matched — which is more
    useful than a confident answer to a question that was not asked.
    """
    findings = list(assessment.get("material_findings") or [])
    ranked = sorted(findings, key=lambda f: {"high": 0, "medium": 1,
                                             "low": 2}.get(f.get("severity"), 3))
    lowered = question.lower()

    if ranked and any(k in lowered for k in
                      ("main reason", "biggest", "most material", "principal "
                       "reason", "why is", "not ready", "evidence")):
        top = ranked[0]
        return {
            "answeredFrom": "completed assessment in this context",
            "portfolioReRun": False,
            "overallReadiness": assessment.get("overall_readiness"),
            "mostMaterialFinding": _finding(top),
            "supportingFindings": [_finding(f) for f in ranked[1:3]],
        }
    return {
        "answeredFrom": "completed assessment in this context",
        "portfolioReRun": False,
        "note": "The question did not match a supported follow-up shape; the "
                "full structured assessment is returned unchanged.",
        "overallReadiness": assessment.get("overall_readiness"),
        "materialFindings": [_finding(f) for f in ranked],
    }
