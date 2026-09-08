"""readiness_agent.agent — the autonomous Securitisation Readiness Agent.

The agent is a loop, not a workflow. It receives an objective, discovers what
Trakt knows, and then decides for itself what to ask. There is deliberately no
metric list, no ordering and no "now check LTV" anywhere in this file or its
prompt — if there were, the result would be workflow automation wearing an
agent's clothes, and the experiment would prove nothing.

What the model is allowed to do
-------------------------------
Understand the objective, choose which governed capability to call next, decide
when a finding deserves a follow-up, weigh materiality, and write the narrative.

What the model is NOT allowed to do
-----------------------------------
Arithmetic. Every number in the output must have come from a Trakt tool call,
and the system prompt says so in the strongest terms available — but the real
enforcement is structural: the agent has no data to compute over. It sees tool
results, not tapes. That is why :class:`~readiness_agent.session.GovernedSession`
hands out three verbs and no DataFrame.

Tool schemas are not restated here
----------------------------------
They are generated from ``trakt_tools.registry``, so the tools the agent can
call are exactly the tools Trakt publishes, with exactly the published schemas.
A hand-maintained copy would drift, and the first symptom would be an agent
calling something that no longer exists.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .session import GovernedSession

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MAX_STEPS = 40
DEFAULT_MAX_TOKENS = 8000

#: The objective. Sparse on purpose — this is the whole input the agent gets
#: about *what to do*, and it names no metric.
OBJECTIVE = (
    "Assess this portfolio for securitisation readiness. Identify material "
    "strengths, weaknesses, risks and areas requiring further diligence. "
    "Support every conclusion with governed Trakt evidence."
)

SYSTEM_PROMPT = """You are a credit analyst assessing a loan portfolio for \
securitisation readiness. You work through Trakt, a governed portfolio-\
intelligence system.

HOW TRAKT WORKS

Trakt owns the data and every calculation. You own the investigation: what to \
look at, what it means, what matters, and when you have enough evidence.

Call `portfolio_capabilities` first. It tells you what Trakt can and cannot \
produce for THIS portfolio, cheaply, without computing anything. Each capability \
carries a status:

  AVAILABLE                Trakt can calculate it. Ask for it if you want it.
  UNAVAILABLE              applicable, but an input or history is missing.
  NOT_APPLICABLE           economically meaningless for this portfolio.
  ASSUMPTION_REQUIRED      needs an unknown future variable, e.g. a rate path.
  MODEL_REQUIRED           needs behavioural modelling Trakt does not perform.
  METHODOLOGY_NOT_APPROVED Trakt holds the data; the definition is unsettled.

These are not interchangeable and you must not treat them as such. If Trakt \
refuses a calculation, that refusal IS the finding — report it and say what \
would be needed. Never substitute a different number for one Trakt declined to \
produce, and never estimate one yourself.

ABSOLUTE RULES

1. You perform NO arithmetic. Every number you state must have been returned by \
a Trakt tool call in this session. Do not add, average, scale, annualise or \
convert anything. If you need a number, ask for it.
2. Distinguish FACT, RULE and JUDGEMENT in every material conclusion.
   FACT is what Trakt measured. RULE is a threshold, and you must name its \
source — a warehouse agreement, transaction criteria and a Trakt internal \
screening threshold are different authorities and a PASS under one is not \
clearance under another. JUDGEMENT is yours, and must be labelled as yours.
3. Trakt's internal screening thresholds are NOT rating-agency requirements, \
covenants or regulatory limits. Never present them as such.
4. You are not a rating agency. Do not predict or imply a rating.
5. Do not confuse observed with modelled. Realised loss severity is what \
happened; a supplied loss-given-default estimate is someone's model. Observed \
default is not a probability of default. Contractual life is not expected life.
6. Report what you could NOT assess as carefully as what you could.

INVESTIGATING

Follow the evidence. A headline figure may conceal a weaker segment beneath it; \
a comfortable level may have an uncomfortable trajectory; a value may be unusual \
without breaching anything. Drill into what looks interesting, and stop when \
further calls would not change your conclusion. Do not investigate everything \
exhaustively, and do not repeat calls you have already made.

FINISHING

When you have enough evidence, call `submit_assessment` exactly once with your \
findings. Overall readiness is a category, never a score: READY_TO_PROGRESS, \
READY_WITH_ISSUES, MATERIAL_REMEDIATION_REQUIRED or INCOMPLETE_ASSESSMENT."""


#: The agent's only non-Trakt tool: how it declares it has finished. Structured
#: so the evaluation harness scores fields rather than parsing prose.
SUBMIT_TOOL = {
    "name": "submit_assessment",
    "description": ("Submit the completed securitisation readiness assessment. "
                    "Call this exactly once, when you have enough governed "
                    "evidence to support your conclusions."),
    "input_schema": {
        "type": "object",
        "properties": {
            "overall_readiness": {
                "type": "string",
                "enum": ["READY_TO_PROGRESS", "READY_WITH_ISSUES",
                         "MATERIAL_REMEDIATION_REQUIRED",
                         "INCOMPLETE_ASSESSMENT"],
            },
            "summary": {"type": "string",
                        "description": "Two or three sentences."},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "material_findings": {
                "type": "array",
                "description": "Every finding you consider material.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "fact": {"type": "string",
                                 "description": "What Trakt measured, with the "
                                                "number and the metric name."},
                        "rule": {"type": "string",
                                 "description": "The threshold and ITS SOURCE, "
                                                "or 'no rule supplied'."},
                        "judgement": {"type": "string",
                                      "description": "Your assessment, as yours."},
                        "severity": {"type": "string",
                                     "enum": ["high", "medium", "low"]},
                        "evidence_tools": {"type": "array",
                                           "items": {"type": "string"}},
                    },
                    "required": ["title", "fact", "judgement", "severity"],
                },
            },
            "could_not_assess": {
                "type": "array",
                "description": "Capabilities Trakt declined, and why it matters.",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string"},
                        "status": {"type": "string"},
                        "implication": {"type": "string"},
                    },
                    "required": ["metric", "status"],
                },
            },
            "further_diligence": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["overall_readiness", "summary", "material_findings"],
    },
}


def governed_tool_schemas() -> List[Dict[str, Any]]:
    """The agent's tool surface, generated from Trakt's own registry."""
    from trakt_tools.registry import all_tools

    schemas: List[Dict[str, Any]] = []
    for spec in sorted(all_tools(), key=lambda s: s.name):
        description = spec.description
        if spec.agent_guidance:
            description = f"{description}\n\nGuidance: {spec.agent_guidance}"
        schemas.append({
            "name": spec.name,
            "description": description[:1800],
            "input_schema": spec.input_schema,
        })
    return schemas


@dataclass
class AgentRun:
    """One complete assessment, and everything observable about how it went."""

    objective: str
    resource: str
    model: str
    assessment: Optional[Dict[str, Any]] = None
    stopped_reason: str = ""
    steps: int = 0
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    efficiency: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0
    #: Text the model produced alongside its tool calls. Kept because it is
    #: observable output, not private reasoning.
    narration: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective, "resource": self.resource,
            "model": self.model, "assessment": self.assessment,
            "stopped_reason": self.stopped_reason, "steps": self.steps,
            "transcript": self.transcript, "efficiency": self.efficiency,
            "usage": self.usage, "elapsed_s": round(self.elapsed_s, 2),
            "narration": self.narration,
        }


def run_assessment(session: GovernedSession, *,
                   objective: str = OBJECTIVE,
                   context_note: str = "",
                   model: str = DEFAULT_MODEL,
                   max_steps: int = DEFAULT_MAX_STEPS,
                   client: Any = None,
                   on_step: Optional[Callable[[int, str], None]] = None
                   ) -> AgentRun:
    """Run one autonomous assessment.

    The loop is: model decides, Trakt executes, model sees the governed result,
    model decides again. Nothing here inspects what the model asked for or
    steers it toward a metric — the only control is the step ceiling, which
    exists so a confused run costs a bounded amount rather than an unbounded one.
    """
    import anthropic

    client = client or anthropic.Anthropic()
    tools = governed_tool_schemas() + [SUBMIT_TOOL]

    opening = (
        f"{objective}\n\n"
        f"Portfolio resource: {session.resource}\n"
    )
    if context_note:
        opening += f"\nContext: {context_note}\n"

    messages: List[Dict[str, Any]] = [{"role": "user", "content": opening}]
    run = AgentRun(objective=objective, resource=session.resource, model=model)
    usage = {"input_tokens": 0, "output_tokens": 0}
    started = time.perf_counter()

    for step in range(1, max_steps + 1):
        run.steps = step
        response = client.messages.create(
            model=model, max_tokens=DEFAULT_MAX_TOKENS,
            # CACHE THE PREFIX THIS LOOP RE-SENDS. A request renders as
            # tools -> system -> messages, so a breakpoint on the system block
            # covers the governed tool schemas as well: together roughly 1.4k
            # tokens that are byte-identical on every step, re-bought on each
            # one. The TTL is an hour because a readiness run is a session, not
            # a burst, and the five-minute default expires between steps that
            # wait on a tool.
            #
            # The QUESTION-shaped half — `messages`, which grows a turn at a
            # time — is deliberately left outside the breakpoint: it changes on
            # every step, and a breakpoint there would write a new cache entry
            # per turn rather than read one.
            system=[{"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            tools=tools, messages=messages)
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        for block in response.content:
            if block.type == "text" and block.text.strip():
                run.narration.append(block.text.strip())

        if not tool_uses:
            run.stopped_reason = "model stopped without submitting"
            break

        messages.append({"role": "assistant", "content": response.content})
        results: List[Dict[str, Any]] = []
        finished = False

        for block in tool_uses:
            if on_step:
                on_step(step, block.name)
            if block.name == "submit_assessment":
                run.assessment = dict(block.input)
                run.stopped_reason = "submitted"
                finished = True
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": "Assessment recorded."})
                continue
            payload = session.call(block.name, dict(block.input or {}))
            results.append({
                "type": "tool_result", "tool_use_id": block.id,
                "content": _serialise(payload),
            })

        messages.append({"role": "user", "content": results})
        if finished:
            break
    else:
        run.stopped_reason = f"step ceiling reached ({max_steps})"

    run.elapsed_s = time.perf_counter() - started
    run.transcript = session.transcript()
    run.efficiency = session.efficiency()
    run.usage = usage
    return run


#: Governed results can be large. Truncating protects the context window, and
#: the truncation is VISIBLE so the model cannot mistake a cut-off list for a
#: complete one and reason over a partial population without knowing it.
_MAX_RESULT_CHARS = 12000


def _serialise(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, default=str, ensure_ascii=False)
    if len(text) <= _MAX_RESULT_CHARS:
        return text
    return (text[:_MAX_RESULT_CHARS]
            + f'\n\n[TRUNCATED: result was {len(text)} characters. Narrow the '
              'request — e.g. filter, or ask for fewer groups — rather than '
              'reasoning over a partial result.]')
