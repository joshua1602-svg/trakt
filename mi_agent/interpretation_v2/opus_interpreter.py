"""The interpretation owner: one question in, one CandidateIntent out.

This is the ONLY place a language model is consulted in the control plane, and
it is consulted about one thing: what does this sentence mean?

What the model receives
-----------------------
    the user's question
    the governed semantic vocabulary (terms, not columns)
    the closed enumerations it may choose from
    the CandidateIntent JSON schema

What the model never receives
-----------------------------
    loan rows, borrower data, balances
    calculated portfolio values, MI answers
    dataframe contents of any kind
    the canonical field registry

The prompt is assembled here and nowhere else, so
``tests/interpretation_v2/test_model_sees_no_data.py`` can assert the whole
payload — it has one function to inspect, not a call graph.

Structured generation
---------------------
The intent is generated through a tool schema with
``additionalProperties: false`` at every level and ``tool_choice`` pinned to
that tool. That is strict structured generation where the provider supports it.
It is still not trusted: :func:`~mi_agent.interpretation_v2.intent.parse_candidate_intent`
re-validates everything fail-closed, because a schema is a request and a parser
is a guarantee.

Failure is never a plan
-----------------------
No key, no client, a transport error, malformed output, a code marker, a date in
a semantic slot — every one of them produces an
:class:`InterpretationOutcome` with ``intent=None`` and a governed reason. There
is no repair loop and no partial salvage: the previous architecture's repair loop
is exactly how a governed refusal became a negotiated answer.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from .intent import (
    INTENT_SCHEMA_VERSION,
    CandidateIntent,
    IntentParseError,
    IntentProvenance,
    candidate_intent_json_schema,
    parse_candidate_intent,
)
from .outcomes import (
    MODEL_OUTPUT_MALFORMED,
    MODEL_UNAVAILABLE,
    CompileReason,
    REASON_CODES,
)
from .vocabulary import GovernedVocabulary, load_governed_vocabulary

INTERPRETER_VERSION = "interpretation_v2.opus_interpreter/1.0.0"

#: The tool the model fills in. Named for what it does, so the model's own
#: reasoning about the call is about interpretation rather than querying.
INTENT_TOOL_NAME = "emit_candidate_intent"

#: Read from the repository's own configuration so this package does not
#: introduce a second opinion about which model MI language work runs on.
try:  # pragma: no cover - configuration import, exercised indirectly
    from mi_agent.mi_agent_config import DEFAULT_MODEL as CONFIGURED_MODEL
except Exception:  # noqa: BLE001
    CONFIGURED_MODEL = "claude-opus-5"


SYSTEM_PROMPT = """\
You are the interpretation layer of a governed portfolio-MI system. Your ONLY \
job is to read one business question and record what it MEANS, using the \
governed semantic vocabulary you are given.

You are not a query engine. You never calculate anything, you never see any \
data, and you never choose how a figure is produced.

RULES

1. Use ONLY terms from the supplied vocabulary. If the question needs a concept \
   that is not there, do not substitute the nearest one — record it in \
   `ambiguity` and leave the slot empty. A refusal downstream is correct; a \
   near-miss binding is not.
2. Never emit a database column, a table name, a snapshot identifier, a date, \
   SQL, pandas, or any code. Time is stated SEMANTICALLY: `current`, \
   `previous_reporting_period`, `relative_pair`, `explicit_period`, `range`, \
   `series`, `forward_looking`. If the question names a period in words, put \
   those words in `time.labels` — "last month", "April" — never a date.
3. A specialist capability owns its own arithmetic. If the question asks about \
   the borrowing base, headroom, a bridge, pipeline stage movement or a \
   forecast, name the capability and its operation and name the specialist \
   measure. Do NOT decompose it into eligible balances, advance rates, \
   reserves or any other component: you would be guessing at a methodology \
   somebody else owns. A concept marked `owned_by_capability` takes NO \
   `statistic` and NO `weight` — the capability decides both — and it needs no \
   period stated for a movement it defines itself.
3a. A filter value must come from the dimension's `values` list. If a \
   dimension shows no governed value list, do NOT assert a value against it: \
   record it in `ambiguity` instead. A value you cannot check is a guess.
4. If the question asks for more than one thing about the SAME population, use \
   `outputs` — one entry per figure or table, with `filters` on an output that \
   apply only to that figure. Do not split one question into unrelated ones.
5. Geography has two axes and both matter: `basis` is whose geography (the \
   borrower's = obligor, the property's = collateral, or the client's own \
   reporting taxonomy) and `level` is how fine. If the question says "region" \
   without saying whose, leave `basis` empty rather than guessing.
6. Record `evidence`: for each material claim, the words from the question that \
   support it.
7. `statistic` is what the question asks for. If it does not say, leave it \
   empty and the governed registry's default will apply. Do not invent one.
8. `ambiguity` has two uses and the difference matters. Set `blocking: true` \
   ONLY when you could not choose and left the slot empty — that stops the \
   question and asks the user. Set `blocking: false` to disclose a reading you \
   DID take, which is recorded alongside the answer. Most notes are \
   non-blocking; a careful reader flags something on nearly every sentence, and \
   that must not turn every question into a refusal to answer.

Answer only by calling the `emit_candidate_intent` tool.
"""


class InterpreterClient(Protocol):
    """The transport boundary. Implementations own auth, retries and the SDK.

    Kept this thin so tests never touch a network and a replayed run is
    indistinguishable from a live one to everything downstream.
    """

    def emit_intent(self, *, system: Sequence[Mapping[str, Any]], user: str,
                    tool_schema: Mapping[str, Any],
                    tool_name: str) -> "ModelResponse":
        ...


@dataclass(frozen=True)
class ModelResponse:
    """What a client returns: the payload, and what actually served it."""

    payload: Optional[Mapping[str, Any]]
    model_id: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""
    raw_text: str = ""


@dataclass(frozen=True)
class InterpretationOutcome:
    """One interpretation attempt.

    ``intent`` is None whenever anything went wrong, and ``reason`` says what.
    A caller cannot accidentally proceed: there is no partially-valid intent.
    """

    question: str
    intent: Optional[CandidateIntent] = None
    reason: Optional[CompileReason] = None
    model_id: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)
    raw_payload: Optional[Mapping[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.intent is not None


# --------------------------------------------------------------------------- #
# Prompt assembly — the one place the model's input is built
# --------------------------------------------------------------------------- #

def build_system_blocks(vocabulary: GovernedVocabulary) -> List[Dict[str, Any]]:
    """The standing context: the rules, then the governed vocabulary.

    The vocabulary is identical for every question in a run, so it belongs in
    the cached prefix rather than being re-sent 135 times — a breakpoint on the
    last block covers the tools and the system together.

    Nothing in here is data. ``vocabulary.prompt_payload`` is the only source of
    the vocabulary block and it carries no canonical fields, no rows, no totals
    and no answers; ``test_model_sees_no_data`` asserts that over this exact
    function's output.
    """
    return [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text",
         "text": ("GOVERNED VOCABULARY — the only terms you may use:\n"
                  + json.dumps(vocabulary.prompt_payload(), indent=1,
                               sort_keys=True)),
         "cache_control": {"type": "ephemeral"}},
    ]


def build_user_prompt(question: str) -> str:
    """The user message: the question, and nothing else."""
    return ("QUESTION:\n" + question.strip()
            + f"\n\nRecord what this question means by calling {INTENT_TOOL_NAME}.")


def build_tool_schema() -> Dict[str, Any]:
    """The tool definition the model generates against."""
    return {
        "name": INTENT_TOOL_NAME,
        "description": ("Record the meaning of the question as a CandidateIntent. "
                        "Semantic terms only — never a column, a date, a snapshot "
                        "identifier or any code."),
        "input_schema": candidate_intent_json_schema(),
    }


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #

class AnthropicInterpreterClient:
    """A real Anthropic-backed client. The SDK is imported lazily.

    Records the model identifier the API ACTUALLY served, not the one that was
    requested. A run that silently fell back to another model is a run whose
    acceptance claim would be false, and the only way to know is to read it back.
    """

    def __init__(self, *, model: str = CONFIGURED_MODEL,
                 api_key: Optional[str] = None, max_tokens: int = 4096,
                 temperature: Optional[float] = None,
                 timeout: float = 120.0) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def emit_intent(self, *, system: Sequence[Mapping[str, Any]], user: str,
                    tool_schema: Mapping[str, Any],
                    tool_name: str) -> ModelResponse:  # pragma: no cover - networked
        if not self._api_key:
            return ModelResponse(payload=None,
                                 error="no ANTHROPIC_API_KEY in the environment")
        try:
            import anthropic
        except ImportError as exc:
            return ModelResponse(payload=None, error=f"anthropic SDK absent: {exc}")

        client = anthropic.Anthropic(api_key=self._api_key, timeout=self._timeout)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "system": [dict(block) for block in system],
            "tools": [dict(tool_schema)],
            "tool_choice": {"type": "tool", "name": tool_name},
            "messages": [{"role": "user", "content": user}],
        }
        # Not every model exposes a temperature control, and the SDK rejects the
        # argument outright where it does not. Omitted unless asked for, so a
        # model that has no such knob is not unreachable because of one.
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        try:
            message = client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - transport failure is an outcome
            return ModelResponse(payload=None, error=f"{type(exc).__name__}: {exc}")

        usage = {}
        if getattr(message, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(message.usage, "input_tokens", None),
                "output_tokens": getattr(message.usage, "output_tokens", None),
                "cache_creation_input_tokens":
                    getattr(message.usage, "cache_creation_input_tokens", None),
                "cache_read_input_tokens":
                    getattr(message.usage, "cache_read_input_tokens", None),
            }
        payload = None
        text_parts: List[str] = []
        for block in message.content or ():
            if getattr(block, "type", "") == "tool_use" and getattr(block, "name", "") == tool_name:
                payload = getattr(block, "input", None)
            elif getattr(block, "type", "") == "text":
                text_parts.append(getattr(block, "text", ""))
        return ModelResponse(payload=payload,
                             model_id=str(getattr(message, "model", "") or ""),
                             usage=usage, raw_text="".join(text_parts),
                             error="" if payload is not None
                                   else "no tool_use block in the response")


class ReplayClient:
    """Replays recorded model payloads. Used by tests and by offline scoring.

    A replayed payload goes through exactly the same parse and the same guards
    as a live one — which is the point: the compiler's behaviour is provable
    without a network, and a recorded run cannot be accepted on softer terms
    than the run that produced it.
    """

    def __init__(self, payloads: Mapping[str, Mapping[str, Any]], *,
                 model_id: str = "replay") -> None:
        self._payloads = dict(payloads)
        self._model_id = model_id

    def emit_intent(self, *, system: Sequence[Mapping[str, Any]], user: str,
                    tool_schema: Mapping[str, Any],
                    tool_name: str) -> ModelResponse:
        question = user.rsplit("QUESTION:\n", 1)[-1].split("\n\nRecord what")[0].strip()
        if question not in self._payloads:
            return ModelResponse(payload=None,
                                 error=f"no recorded payload for {question!r}")
        return ModelResponse(payload=self._payloads[question],
                             model_id=self._model_id)


class UnavailableClient:
    """A client that cannot serve. Exists so 'no model' is a testable path."""

    def __init__(self, reason: str = "interpreter unavailable") -> None:
        self._reason = reason

    def emit_intent(self, **_: Any) -> ModelResponse:
        return ModelResponse(payload=None, error=self._reason)


# --------------------------------------------------------------------------- #
# The interpreter
# --------------------------------------------------------------------------- #

class OpusInterpreter:
    """Question -> CandidateIntent. The single owner of natural language."""

    version = INTERPRETER_VERSION

    def __init__(self, client: InterpreterClient, *,
                 vocabulary: Optional[GovernedVocabulary] = None) -> None:
        self.client = client
        self.vocabulary = vocabulary or load_governed_vocabulary()

    def interpret(self, question: str) -> InterpretationOutcome:
        system = build_system_blocks(self.vocabulary)
        user = build_user_prompt(question)
        response = self.client.emit_intent(system=system, user=user,
                                           tool_schema=build_tool_schema(),
                                           tool_name=INTENT_TOOL_NAME)

        if response.payload is None:
            return InterpretationOutcome(
                question=question,
                reason=CompileReason(MODEL_UNAVAILABLE, "interpreter",
                                     response.error or "no payload returned"),
                model_id=response.model_id, usage=response.usage)

        provenance = IntentProvenance(
            question=question,
            model_id=response.model_id,
            interpreter_version=self.version,
            vocabulary_version=self.vocabulary.version,
            usage=dict(response.usage or {}),
        )
        try:
            intent = parse_candidate_intent(response.payload, provenance=provenance)
        except IntentParseError as exc:
            code = exc.code if exc.code in REASON_CODES else MODEL_OUTPUT_MALFORMED
            return InterpretationOutcome(
                question=question,
                reason=CompileReason(code, exc.subject, exc.detail),
                model_id=response.model_id, usage=response.usage,
                raw_payload=response.payload)
        except Exception as exc:  # noqa: BLE001 - any parse failure is a refusal
            return InterpretationOutcome(
                question=question,
                reason=CompileReason(MODEL_OUTPUT_MALFORMED, "intent",
                                     f"{type(exc).__name__}: {exc}"),
                model_id=response.model_id, usage=response.usage,
                raw_payload=response.payload)

        return InterpretationOutcome(question=question, intent=intent,
                                     model_id=response.model_id,
                                     usage=response.usage,
                                     raw_payload=response.payload)


def interpret_and_compile(question: str, interpreter: OpusInterpreter,
                          compiler: Any) -> Tuple[InterpretationOutcome, Any]:
    """The whole control plane: question -> intent -> plan | refuse | clarify.

    Returns both halves so a caller can tell an interpretation failure from a
    governed compilation refusal — they are different findings about the system
    and collapsing them would hide which half needs work.
    """
    from .outcomes import refuse

    outcome = interpreter.interpret(question)
    if not outcome.ok:
        return outcome, refuse(outcome.reason,
                               compiler_version=getattr(compiler, "version", ""))
    return outcome, compiler.compile(outcome.intent)
