"""The conversation reader: a follow-up or a reply, made one complete question
(P0 design §39; owner decisions D24, D25).

WHY A SEPARATE STEP. The interpreter is signed off on complete questions: its
view is pinned to the baseline (§37), and the D13 sign-off measured it. A
follow-up ("And by broker?") or a reply to the agent's own question ("The
property's") is not a complete question. Rather than teach the interpreter a
second kind of message — a second view, measured nowhere — this step turns
the message and what the conversation holds into ONE complete question, and
the signed-off interpreter reads that question exactly as it reads one typed
in full. It is the standard design for conversational analytics (the
follow-up is rewritten into a self-contained question before the single-
question pipeline runs), and it means:

  - a complete question is never re-worded: the reader returns it word for
    word, and it reaches the interpreter exactly as if no conversation
    existed (D24: a complete question never inherits);
  - what carried over is visible: the answer states the complete question it
    answered, in the user's own words;
  - every gate a first question passes — the interpreter, the compiler, the
    perimeter, the runtimes — runs again on the complete question (§34).

WHAT IT SEES. The last question the agent answered (the complete question,
never its figures), an ask-back still open, the user's latest message, and
the governed terms the book uses — no data, no figure, no row. What it
returns is checked here before it is used: a complete question may not hold
a number the conversation did not (it cannot invent a period, an amount or a
threshold), and a malformed or failed reading is not used at all — the
message is then read on its own, and the answer says so (D24).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .vocabulary import GovernedVocabulary, load_governed_vocabulary

READER_VERSION = "interpretation_v2.conversation_reader/1.1.0"
READER_TOOL_NAME = "record_complete_question"

#: What a reading can be.
COMPLETE = "complete"   # the complete question (the message itself, if complete)
ASK = "ask"             # one more detail from the user would settle it
CANNOT = "cannot"       # it refers to something the conversation does not hold
OUTCOMES = (COMPLETE, ASK, CANNOT)

#: Bounds on what the reader may hand back. A complete question is a sentence,
#: not a paragraph; an ask is one short question.
MAX_QUESTION_CHARS = 400
MAX_ASK_CHARS = 300

SYSTEM_PROMPT = """\
You read the latest message in a conversation between a user and a governed \
management-information agent for a lender (its funded loan book, its pipeline \
of applications not yet funded, and its forecast). Your ONLY job is to turn \
that message into ONE complete question.

You do not answer anything. You never see figures. A separate reader, which \
is shown only complete questions, works out what the question means and \
whether it can be answered; what you return is the question it is shown.

You are given the last question the agent answered (in full), any detail the \
agent asked the user for since, and the user's latest message.

RULES

1. A complete question — a full question that would mean the same to someone \
who never saw the conversation — is returned EXACTLY as written, word for \
word, with outcome "complete". Nothing carries over into it, even when it is \
about the same subject as before or names the same figure.

2. Any other message is a follow-up: a short message that names only a \
figure, a grouping, a period or a value without a full question around it, \
or a message that refers back to the conversation (with words such as \
"that", "it", "those", "instead", "also", or opening with "and" or "what \
about"). Write the complete question it makes when read with the \
conversation: the earlier question with the message's change applied, so \
that someone who never saw the conversation would read it the same way. Keep \
everything from the earlier question that the message does not change. Keep \
the user's own words; add only what the message leaves out and the \
conversation supplies.

3. Never add anything the conversation does not contain: no period, date, \
amount, threshold, place, name, grouping or restriction that neither the \
earlier question nor the message states.

4. A new grouping replaces the earlier question's grouping, however the \
message opens — with "and", with "what about", or with the grouping alone. \
The earlier grouping is kept, and the new one added, only when the message \
says so in words: split further, within each, as well as, or two groupings \
named together. A message that narrows to one value of the earlier grouping \
keeps the figure, restricts it to that value, and drops that grouping. A \
message that takes a change back — to the usual, standard or default \
setting, or to everything again — is the earlier question without that \
detail; where the earlier question does not have it, the earlier question \
as it stands.

5. A message that answers a detail the agent asked for supplies that detail: \
write the question the agent asked about, completed with it — and, where that \
question was itself a follow-up, read with the last question answered.

6. Ask only when you cannot tell what in the conversation the message \
refers to — which figure, grouping, period or value of the earlier question \
it changes — and one more detail from the user would settle that: return \
outcome "ask" with that one detail as a short question the user can answer, \
naming the choices when there are only a few. Never ask what a word or \
phrase means, or which period, grain, window, basis or part of the book the \
user intends: keep the user's own words for it in the complete question. The \
reader of complete questions holds the book's definitions and defaults, and \
asks the user itself when a question is unclear.

7. If the message refers to something the conversation does not hold — you \
are shown only the last question the agent answered and any detail asked for \
since — return outcome "cannot".

Use the book's governed terms, listed below, only to recognise what the user \
means and to name choices; never to add a figure, grouping or restriction the \
user did not ask for.

Answer only by calling the `record_complete_question` tool.\
"""


@dataclass(frozen=True)
class Reading:
    """What the reader made of one message."""

    outcome: str = ""
    question: str = ""
    ask: str = ""
    reason: str = ""
    #: Why the reading is not used (empty when it is).
    error: str = ""
    model_id: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error and self.outcome in OUTCOMES


def _norm(text: str) -> str:
    return " ".join(str(text or "").split()).strip().rstrip("?.! ").lower()


def carried(message: str, question: str) -> bool:
    """Whether the complete question differs from what the user typed — i.e.
    whether anything was carried over into it."""
    return _norm(message) != _norm(question)


# --------------------------------------------------------------------------- #
# What the reader is shown
# --------------------------------------------------------------------------- #

def _unique(values: Sequence[Any]) -> List[str]:
    seen: Dict[str, str] = {}
    for value in values:
        text = str(value).replace("_", " ").strip()
        if text and text.lower() not in seen:
            seen[text.lower()] = text
    return list(seen.values())


#: How each owning capability is named to the reader.
_AREAS = (
    (None, "Funded loan book"),
    ("pipeline", "Pipeline (applications not yet funded)"),
    ("pipeline_stage_movement", "Movement of pipeline cases between stages"),
    ("forecast", "Forecast"),
    ("funded_bridge", "Funded balance movement"),
    ("concentration", "Concentration"),
    ("limit_assessment", "Limits"),
    ("borrowing_base", "Borrowing base"),
    ("portfolio_summary", "Portfolio overview"),
)


def governed_terms(vocabulary: GovernedVocabulary) -> Dict[str, Any]:
    """The book's governed terms, by area: the figures, the groupings, and the
    values of the short closed lists (scenarios, stages, bands). Names only —
    nothing here is data."""
    areas: Dict[str, Dict[str, Any]] = {}
    for capability, title in _AREAS:
        concepts = [c for c in vocabulary.concepts.values()
                    if c.owning_capability == capability and c.label]
        if not concepts:
            continue
        figures = _unique(c.label for c in concepts if c.role == "measure")
        groupings = _unique(c.label for c in concepts
                            if c.role in ("dimension", "flag"))
        values = {c.label: _unique(c.values) for c in concepts
                  if c.values and len(_unique(c.values)) <= 8}
        areas[title] = {"figures": figures, "groupings": groupings,
                        **({"values": values} if values else {})}
    return areas


def build_system_blocks(vocabulary: Optional[GovernedVocabulary] = None
                        ) -> List[Dict[str, Any]]:
    """The reader's standing context: its rules and the governed terms.
    Identical for every request, so cached."""
    vocabulary = vocabulary or load_governed_vocabulary()
    return [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text",
         "text": ("THE BOOK'S GOVERNED TERMS, by area:\n"
                  + json.dumps(governed_terms(vocabulary), sort_keys=True,
                               separators=(", ", ": "))),
         "cache_control": {"type": "ephemeral"}},
    ]


def build_user_prompt(message: str, memory: Any) -> str:
    """The conversation as the reader is shown it: what it holds, then the
    user's latest message."""
    last = str(getattr(memory, "last", "") or "").strip()
    pending = getattr(memory, "pending", None)
    lines = ["THE LAST QUESTION THE AGENT ANSWERED:",
             last or "(none in this conversation)", ""]
    if pending is not None:
        lines += ["SINCE THEN THE AGENT WAS ASKED:", str(pending.question).strip(),
                  ""]
        for asked, replied in pending.turns:
            lines += ["THE AGENT ASKED FOR A DETAIL:", str(asked).strip(),
                      "THE USER REPLIED:", str(replied).strip(), ""]
        lines += ["THE AGENT ASKED FOR ONE MORE DETAIL:", str(pending.ask).strip(),
                  ""]
    lines += ["THE USER'S LATEST MESSAGE:", str(message).strip(), "",
              f"Record the complete question by calling {READER_TOOL_NAME}."]
    return "\n".join(lines)


def build_tool_schema() -> Dict[str, Any]:
    return {
        "name": READER_TOOL_NAME,
        "description": ("Record the user's latest message as one complete "
                        "question, or the one detail needed to make it one."),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "outcome": {"type": "string", "enum": list(OUTCOMES)},
                "question": {
                    "type": "string",
                    "description": ("Outcome complete: the complete question. "
                                    "The latest message word for word when it "
                                    "is complete on its own.")},
                "ask": {
                    "type": "string",
                    "description": ("Outcome ask: the one detail needed, as a "
                                    "short question to the user.")},
                "reason": {
                    "type": "string",
                    "description": "Outcome cannot: why, in one sentence."},
            },
            "required": ["outcome"],
        },
    }


# --------------------------------------------------------------------------- #
# The guards on what comes back
# --------------------------------------------------------------------------- #

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers(text: str) -> set:
    return {n.replace(",", "") for n in _NUMBER.findall(str(text or ""))}


def _conversation_text(message: str, memory: Any) -> str:
    pending = getattr(memory, "pending", None)
    parts = [message, str(getattr(memory, "last", "") or "")]
    if pending is not None:
        parts += [pending.question, pending.ask]
        parts += [x for turn in pending.turns for x in turn]
    return "\n".join(str(p) for p in parts)


def check(payload: Any, *, message: str, memory: Any,
          choices: str = "") -> Reading:
    """The reader's payload as a Reading, or a Reading with the error that
    keeps it from being used. Fail-closed: anything unexpected is an error.

    `choices` is the text of the governed terms: an ask may name the values
    of a closed list ("75-80") as choices; a complete question may not add
    one the user did not state."""
    if not isinstance(payload, Mapping):
        return Reading(error="no reading returned")
    unknown = set(payload) - {"outcome", "question", "ask", "reason"}
    if unknown:
        return Reading(error=f"unexpected fields {sorted(unknown)}")
    outcome = str(payload.get("outcome") or "")
    question = " ".join(str(payload.get("question") or "").split())
    ask = " ".join(str(payload.get("ask") or "").split())
    reason = " ".join(str(payload.get("reason") or "").split())
    if outcome not in OUTCOMES:
        return Reading(error=f"outcome {outcome!r} is not one of {OUTCOMES}")
    if outcome == COMPLETE:
        if not question:
            return Reading(error="a complete reading with no question")
        if len(question) > MAX_QUESTION_CHARS:
            return Reading(error="the complete question is too long")
        text = question
    elif outcome == ASK:
        if not ask or len(ask) > MAX_ASK_CHARS:
            return Reading(error="an ask with no detail, or too long a one")
        text = ask
    else:
        text = reason
        if len(reason) > MAX_ASK_CHARS:
            return Reading(error="the reason is too long")
    # NOTHING THE CONVERSATION DID NOT HOLD. A number in what the reader
    # wrote that is in neither the message nor what the conversation holds
    # was invented — a period, an amount, a threshold nobody asked for.
    held = _numbers(_conversation_text(message, memory))
    if outcome == ASK:
        held |= _numbers(choices)
    invented = _numbers(text) - held
    if invented:
        return Reading(error=f"it added {sorted(invented)}, which the "
                             f"conversation does not hold")
    return Reading(outcome=outcome, question=question, ask=ask, reason=reason)


# --------------------------------------------------------------------------- #
# The reader
# --------------------------------------------------------------------------- #

class ConversationReader:
    """Message + memory -> one complete question (or an ask, or a cannot)."""

    version = READER_VERSION

    def __init__(self, client: Any, *,
                 vocabulary: Optional[GovernedVocabulary] = None) -> None:
        self.client = client
        self.vocabulary = vocabulary or load_governed_vocabulary()

    def read(self, message: str, memory: Any) -> Reading:
        try:
            response = self.client.emit_intent(
                system=build_system_blocks(self.vocabulary),
                user=build_user_prompt(message, memory),
                tool_schema=build_tool_schema(), tool_name=READER_TOOL_NAME)
        except Exception as exc:                                     # noqa: BLE001
            return Reading(error=f"{type(exc).__name__}: {exc}"[:300])
        if response.payload is None:
            return Reading(error=response.error or "no reading returned",
                           model_id=response.model_id, usage=response.usage)
        reading = check(response.payload, message=message, memory=memory,
                        choices=json.dumps(governed_terms(self.vocabulary)))
        return Reading(outcome=reading.outcome, question=reading.question,
                       ask=reading.ask, reason=reading.reason,
                       error=reading.error, model_id=response.model_id,
                       usage=dict(response.usage or {}))


def default_reader() -> ConversationReader:
    """The reader over the same model client and settings as the interpreter."""
    from .opus_interpreter import AnthropicInterpreterClient
    return ConversationReader(AnthropicInterpreterClient(max_tokens=4096))


def reader_view(vocabulary: Optional[GovernedVocabulary] = None) -> Dict[str, Any]:
    """EVERYTHING THE READER IS SHOWN, as production builds it — pinned like
    the interpreter's (`test_the_models_view_is_the_baselines`), so a change
    to it is a measured change."""
    from .opus_interpreter import AnthropicInterpreterClient
    from .opus_interpreter import forces_tool, uses_fallbacks
    client = AnthropicInterpreterClient(max_tokens=4096)
    return {"system": build_system_blocks(vocabulary),
            "user": build_user_prompt("<MESSAGE>", None),
            "tool": build_tool_schema(), "tool_name": READER_TOOL_NAME,
            "model": client.model, "max_tokens": client._max_tokens,
            "temperature": client._temperature, "effort": client.effort,
            "forced_tool": forces_tool(client.model),
            "fallbacks": uses_fallbacks(client.model)}


def reader_view_fingerprint(vocabulary: Optional[GovernedVocabulary] = None) -> str:
    blob = json.dumps(reader_view(vocabulary), sort_keys=True,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
