"""The interpretation owner: one question in, one CandidateIntent out.

This is the ONLY place a language model is consulted in the control plane, and
it is consulted about one thing: what does this sentence mean?

What the model receives
-----------------------
    the user's question
    the closed enumerations it may choose from
    the CandidateIntent JSON schema
    READ-ONLY metadata tools over Trakt's governed registries

The last of those is the change measurement forced. A single flat vocabulary
dumped into every prompt cost 24k tokens a question and still left the
interpreter unable to verify that "drawdown" is a product; it now retrieves what
it needs, the way an analyst inspects a workbook's headers and definitions
before deciding what a question means.

Opus may therefore SEE canonical identifiers and name them. That does not make
it authoritative: the compiler independently re-derives existence, ambiguity,
asset applicability, portfolio availability and physical binding for every
concept before a plan can carry it.

What the model never receives
-----------------------------
    loan rows, borrower data, balances
    calculated portfolio values, MI answers
    dataframe contents of any kind
    facility commitments, advance rates or drawn amounts
    a reporting date or any other snapshot handle

The prompt is assembled here and nowhere else, and every metadata tool result
comes from one dispatcher, so ``test_model_sees_no_data.py`` can assert the
whole surface — two functions and one service to inspect, not a call graph.

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
from .metadata import GovernedMetadataService, metadata_tool_schemas
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
job is to read one business question and record what it MEANS.

You are not a query engine. You never calculate anything, you never see any \
data, and you never choose how a figure is produced. Everything you name is a \
PROPOSAL: a deterministic compiler independently re-validates every concept \
against the same governed registry before anything runs, and will refuse \
rather than approximate.

HOW TO WORK

You have read-only metadata tools over Trakt's governed registries. Use them — \
do not guess a concept identifier from memory. A normal sequence is:

  * `search_concepts` to find the governed identifier for a business word;
  * `get_concept_metadata` to confirm its role, temporality and permitted
    statistics;
  * `get_allowed_values` BEFORE asserting any filter value;
  * `search_capabilities` / `get_capability_metadata` when the question asks
    for a named analysis rather than a figure;
  * `get_asset_metadata` and `get_portfolio_semantic_context` when the question
    depends on what this environment actually is.

Retrieve what you need, then call `emit_candidate_intent` exactly once.

MINIMUM SUFFICIENT GOVERNED INTENT

Record the SMALLEST intent that completely and faithfully answers the question \
asked — no smaller, and no larger.

This is not literalism. Infer an element the question leaves unstated when \
answering it REQUIRES that element and the governed metadata, the asset and \
client context and the capability catalogue leave one materially clear reading. \
"Show balance by region" does not need the user to name a statistic, a \
capability or a geography basis when each is a unique governed consequence of \
what was asked.

It is the other direction that needs discipline. Do NOT add a measure, \
dimension, comparison, analysis or output because it is useful, interesting, \
customarily shown beside the requested figure, analytically related, or simply \
available in Trakt. The test is:

    necessary to answer the question   ->  include
    merely useful to add               ->  leave out

A capability's `owned_measures` list is its INVENTORY, not a menu to fill in. \
Name the measures the question asks to RECEIVE. What the capability needs \
internally to compute them is its own business, and it does not need you to \
enumerate it.

RULES

1. Name governed concept identifiers you have confirmed through the tools. If \
   the question needs a concept the registry does not carry, do not substitute \
   the nearest one — record it in `ambiguity` and leave the slot empty. A \
   refusal downstream is correct; a near-miss binding is not.
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
3a. A filter value must come from `get_allowed_values`. If that reports \
   `has_governed_values: false`, do NOT assert a value against it: record it in \
   `ambiguity` with blocking=true instead. A value you cannot check is a guess.
3b. A figure the question names as a GOAL goes in `target`, never in `filters`. \
   "When will we reach one hundred million?" is a milestone whose target is the \
   governed balance concept and the figure named; it does not narrow the \
   population to loans above that figure. A `forecast_milestone` without a \
   target is incomplete.
3c. A period-on-period movement is stated ONCE, by the operation and \
   `time.form` — there is no `period_pair` comparison. `comparison` is only for \
   two POPULATIONS or two DIMENSION VALUES held against each other, and both \
   sides must be named.
4. If the question asks for more than one thing about the SAME population, use \
   `outputs` — one entry per figure or table, with `filters` on an output that \
   apply only to that figure. Do not split one question into unrelated ones.
5. Geography has two axes and both matter: `basis` is whose geography (the \
   borrower's = obligor, the property's = collateral, or the client's own \
   reporting taxonomy) and `level` is how fine. If the question says "region" \
   without saying whose, leave `basis` empty rather than guessing — at level \
   `reporting` a governed default resolves it, so an empty basis there is safe \
   and is NOT a blocking ambiguity.
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

BEFORE YOU SUBMIT

Check three things. Act on them; do not narrate them.

A. PRESERVATION. Every material semantic element the question states \
   EXPLICITLY — measure, statistic, weight, population or portfolio lens, \
   filter, grouping or dimension, geography basis or level, temporal \
   relationship, comparison, target or threshold, analytical operation — is \
   either represented in the intent, or recorded as a blocking ambiguity \
   because you could not bind it confidently. An element the user named is \
   never simply absent from an intent that otherwise goes ahead. Dropping it \
   quietly and answering something broader is the one outcome that must not \
   happen.

B. NECESSITY. For every measure, dimension, comparison, analysis and output you \
   included: is it necessary to answer THIS question? If it is not, remove it.

C. AMBIGUITY. Do two materially different governed readings remain — readings \
   that would make Trakt execute different work? Then set a blocking ambiguity. \
   If the intended analysis is materially clear, go ahead and record it.

   Terseness is not ambiguity. Nor is a user saying "region" instead of a \
   registry field name, leaving an optional companion measure unstated, not \
   naming an internal Trakt capability, or omitting a presentation preference. \
   Prefer a faithful reading over an unnecessary question, and an honest \
   question over a speculative reading.

Answer only by calling the `emit_candidate_intent` tool.
"""


class InterpreterClient(Protocol):
    """The transport boundary. Implementations own auth, retries and the SDK.

    ``emit_intent`` runs the WHOLE exchange — metadata retrieval included —
    because a client that returned one turn at a time would put the tool loop in
    two places. The dispatcher it is handed is the only thing that can answer a
    metadata call, and it can only answer with metadata.
    """

    def emit_intent(self, *, system: Sequence[Mapping[str, Any]], user: str,
                    tool_schema: Mapping[str, Any], tool_name: str,
                    metadata_tools: Sequence[Mapping[str, Any]] = (),
                    dispatch: Optional[Any] = None) -> "ModelResponse":
        ...


@dataclass(frozen=True)
class ModelResponse:
    """What a client returns: the payload, and what actually served it."""

    payload: Optional[Mapping[str, Any]]
    model_id: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""
    raw_text: str = ""
    #: The metadata retrievals the model made on the way. Provenance only.
    metadata_calls: Tuple[Mapping[str, Any], ...] = ()


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
    #: Which governed metadata the model retrieved before answering.
    metadata_calls: Tuple[Mapping[str, Any], ...] = ()

    @property
    def ok(self) -> bool:
        return self.intent is not None


# --------------------------------------------------------------------------- #
# Prompt assembly — the one place the model's input is built
# --------------------------------------------------------------------------- #

def build_system_blocks(vocabulary: GovernedVocabulary) -> List[Dict[str, Any]]:
    """The standing context: the rules, then a short orientation block.

    Deliberately NOT the registry. Dumping 150 concepts into every prompt cost
    24k tokens a question and still under-described every one of them; the model
    now RETRIEVES what it needs through the metadata tools, the way an analyst
    inspects the workbook's headers and definitions before deciding what a
    question means.

    Identical for every question in a run, so it sits in the cached prefix — a
    breakpoint on the last block covers the tools and the system together.

    Nothing in here is data. ``test_model_sees_no_data`` asserts that over this
    exact function's output and over every metadata tool result.
    """
    return [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text",
         "text": ("GOVERNED CONTEXT — the closed enumerations, and how to find "
                  "everything else:\n"
                  + json.dumps(vocabulary.orientation_payload(), indent=1,
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

    #: How many retrieve-then-think rounds the model gets before the intent tool
    #: is forced. Bounded because an unbounded loop is an unbounded bill, and
    #: because a question needing more than this many lookups is a question the
    #: vocabulary does not describe well enough — which is a finding, not a
    #: reason to keep paying.
    max_rounds = 6

    def __init__(self, *, model: str = CONFIGURED_MODEL,
                 api_key: Optional[str] = None, max_tokens: int = 4096,
                 temperature: Optional[float] = None,
                 timeout: float = 180.0, max_rounds: Optional[int] = None) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        if max_rounds is not None:
            self.max_rounds = max(1, int(max_rounds))

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def emit_intent(self, *, system: Sequence[Mapping[str, Any]], user: str,
                    tool_schema: Mapping[str, Any], tool_name: str,
                    metadata_tools: Sequence[Mapping[str, Any]] = (),
                    dispatch: Optional[Any] = None
                    ) -> ModelResponse:  # pragma: no cover - networked
        if not self._api_key:
            return ModelResponse(payload=None,
                                 error="no ANTHROPIC_API_KEY in the environment")
        try:
            import anthropic
        except ImportError as exc:
            return ModelResponse(payload=None, error=f"anthropic SDK absent: {exc}")

        client = anthropic.Anthropic(api_key=self._api_key, timeout=self._timeout)
        tools = [dict(t) for t in metadata_tools] + [dict(tool_schema)]
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user}]
        usage = {"input_tokens": 0, "output_tokens": 0,
                 "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        calls: List[Dict[str, Any]] = []
        model_id = ""

        for round_index in range(self.max_rounds):
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": self._max_tokens,
                "system": [dict(block) for block in system],
                "tools": tools,
                "messages": messages,
            }
            # The last round FORCES the intent tool. Left to itself a model can
            # keep retrieving; the loop has to end in a verdict, and ending it
            # by giving up would turn a bounded budget into a silent failure.
            kwargs["tool_choice"] = ({"type": "tool", "name": tool_name}
                                     if round_index == self.max_rounds - 1
                                     else {"type": "auto"})
            # Not every model exposes a temperature control, and the SDK rejects
            # the argument outright where it does not.
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature
            try:
                message = client.messages.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - transport failure is an outcome
                return ModelResponse(payload=None, usage=usage, model_id=model_id,
                                     error=f"{type(exc).__name__}: {exc}",
                                     metadata_calls=tuple(calls))

            model_id = str(getattr(message, "model", "") or "") or model_id
            if getattr(message, "usage", None) is not None:
                for key in usage:
                    usage[key] += int(getattr(message.usage, key, 0) or 0)

            payload = None
            text_parts: List[str] = []
            metadata_requests: List[Any] = []
            for block in message.content or ():
                kind = getattr(block, "type", "")
                if kind == "tool_use":
                    if getattr(block, "name", "") == tool_name:
                        payload = getattr(block, "input", None)
                    else:
                        metadata_requests.append(block)
                elif kind == "text":
                    text_parts.append(getattr(block, "text", ""))

            if payload is not None:
                return ModelResponse(payload=payload, model_id=model_id,
                                     usage=usage, raw_text="".join(text_parts),
                                     metadata_calls=tuple(calls))

            if not metadata_requests:
                return ModelResponse(payload=None, model_id=model_id, usage=usage,
                                     raw_text="".join(text_parts),
                                     error="no tool_use block in the response",
                                     metadata_calls=tuple(calls))

            messages.append({"role": "assistant", "content": message.content})
            results = []
            for block in metadata_requests:
                name = getattr(block, "name", "")
                arguments = getattr(block, "input", {}) or {}
                result = (dispatch(name, arguments) if dispatch is not None
                          else {"error": "no metadata service available"})
                calls.append({"tool": name, "arguments": dict(arguments)})
                results.append({"type": "tool_result",
                                "tool_use_id": getattr(block, "id", ""),
                                "content": json.dumps(result, default=str)})
            messages.append({"role": "user", "content": results})

        return ModelResponse(payload=None, model_id=model_id, usage=usage,
                             error=f"no intent after {self.max_rounds} rounds",
                             metadata_calls=tuple(calls))


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
                    tool_schema: Mapping[str, Any], tool_name: str,
                    metadata_tools: Sequence[Mapping[str, Any]] = (),
                    dispatch: Optional[Any] = None) -> ModelResponse:
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
                 vocabulary: Optional[GovernedVocabulary] = None,
                 metadata: Optional[GovernedMetadataService] = None) -> None:
        self.client = client
        self.vocabulary = vocabulary or load_governed_vocabulary()
        #: The metadata service reads from the SAME index the compiler validates
        #: against. A service that drifted from it would advertise concepts that
        #: then refuse, which is worse than showing the model nothing.
        self.metadata = metadata or GovernedMetadataService(self.vocabulary)

    def interpret(self, question: str) -> InterpretationOutcome:
        system = build_system_blocks(self.vocabulary)
        user = build_user_prompt(question)
        service = GovernedMetadataService(self.vocabulary)
        response = self.client.emit_intent(
            system=system, user=user, tool_schema=build_tool_schema(),
            tool_name=INTENT_TOOL_NAME,
            metadata_tools=metadata_tool_schemas(),
            dispatch=service.call)

        if response.payload is None:
            return InterpretationOutcome(
                question=question,
                reason=CompileReason(MODEL_UNAVAILABLE, "interpreter",
                                     response.error or "no payload returned"),
                model_id=response.model_id, usage=response.usage,
                metadata_calls=response.metadata_calls)

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
                raw_payload=response.payload,
                metadata_calls=response.metadata_calls)
        except Exception as exc:  # noqa: BLE001 - any parse failure is a refusal
            return InterpretationOutcome(
                question=question,
                reason=CompileReason(MODEL_OUTPUT_MALFORMED, "intent",
                                     f"{type(exc).__name__}: {exc}"),
                model_id=response.model_id, usage=response.usage,
                raw_payload=response.payload,
                metadata_calls=response.metadata_calls)

        return InterpretationOutcome(question=question, intent=intent,
                                     model_id=response.model_id,
                                     usage=response.usage,
                                     raw_payload=response.payload,
                                     metadata_calls=response.metadata_calls)


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
