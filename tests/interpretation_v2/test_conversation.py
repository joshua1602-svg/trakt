"""Conversation: the reply to an ask-back (phase 1) and the follow-up (phase 2)
— P0 design §34, §38, §39; owner decisions D24, D25.

A reply ("The property's") or a follow-up ("And by broker?") cannot stand on
its own; read with what the conversation holds it is a complete question.
These tests hold the conversation to its rules:

  - the memory is a SIGNED token bound to the user, the book and the chat,
    holding the last answered question and an ask-back still open, for
    `memory_minutes` from the delivery of the agent's last message — anything
    else is refused and the message is read on its own, and says so;
  - the conversation reader makes the message ONE complete question; the
    signed-off interpreter reads only complete questions, through the
    baseline's view, untouched;
  - a complete question is never re-worded and nothing carries into it;
  - what carried over is stated; an ask-back keeps the last answered
    question; a decline does not replace it (D25);
  - switched off, nothing differs from before.
"""
from __future__ import annotations

import base64
import json

import pytest

from mi_agent import conversation as convo
from mi_agent import plan_serving_canary as canary
from mi_agent.interpretation_v2 import conversation_reader as reader
from mi_agent.interpretation_v2 import opus_interpreter
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.interpretation_v2.opus_interpreter import (
    InterpretationOutcome, ModelResponse, OpusInterpreter, build_user_prompt)
from tests.interpretation_v2.test_specialist_runtime_pipeline import (
    _CLIENT, _HISTORY_ROOT, _PIPELINE_INTENT, _SOURCE)

_KEY = "k" * 40
_NOW = 1_800_000_000.0
_ASK = convo.PendingAsk(question="Compare the pipeline amount.",
                        ask="Compare with what is not stated.")
_LAST = "What is the pipeline amount by stage?"


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(convo.SWITCH_ENV, "on")
    monkeypatch.setenv(convo.KEY_ENV, _KEY)


def _issue(memory=convo.Memory(last=_LAST), **over):
    args = dict(principal="user-a", book="ERE||", chat="chat-1", memory=memory,
                now=_NOW)
    args.update(over)
    return convo.issue(**args)


def _read(token, **over):
    args = dict(principal="user-a", book="ERE||", chat="chat-1", now=_NOW + 60)
    args.update(over)
    return convo.read(token, **args)


# --------------------------------------------------------------------------- #
# the memory
# --------------------------------------------------------------------------- #

def test_switched_off_nothing_is_issued_or_read(monkeypatch):
    monkeypatch.delenv(convo.SWITCH_ENV, raising=False)
    monkeypatch.setenv(convo.KEY_ENV, _KEY)
    assert not convo.enabled()
    assert _issue() is None
    assert _read("anything") is None


def test_a_short_key_keeps_it_off(monkeypatch):
    monkeypatch.setenv(convo.SWITCH_ENV, "on")
    monkeypatch.setenv(convo.KEY_ENV, "short")
    assert not convo.enabled()


def test_the_memory_reads_back_what_it_holds(on):
    """D24: the last answered question, and an ask-back still open."""
    memory = convo.Memory(last=_LAST, pending=_ASK)
    assert _read(_issue(memory)).memory == memory
    assert _read(_issue(convo.Memory(last=_LAST))).memory == convo.Memory(last=_LAST)
    assert _read(_issue(convo.Memory(pending=_ASK))).memory == convo.Memory(pending=_ASK)


def test_a_memory_holding_nothing_is_not_issued(on):
    assert _issue(convo.Memory()) is None


def test_the_memory_lasts_five_minutes_from_delivery(on):
    """D24: the clock runs from the delivery of the agent's message (its issue)."""
    token = _issue()
    assert convo.memory_minutes() == 5
    assert _read(token, now=_NOW + 299).ok
    assert _read(token, now=_NOW + 301).lapsed == convo.LAPSED_EXPIRED


@pytest.mark.parametrize("over, reason", [
    ({"principal": "user-b"}, convo.LAPSED_OTHER_USER),
    ({"book": "ERE|ERE/2026-07-31|"}, convo.LAPSED_OTHER_BOOK),
    ({"chat": "chat-2"}, convo.LAPSED_OTHER_CHAT),
])
def test_a_token_belongs_to_its_user_book_and_chat(on, over, reason):
    """F05 (another user), E04 (another book), a cleared chat (a new chat id)."""
    assert _read(_issue(), **over).lapsed == reason


def _forged(token, **changes):
    encoded, signature = token.split(".", 1)
    body = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    body.update(changes)
    forged = base64.urlsafe_b64encode(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    return f"{forged}.{signature}"


def test_an_edited_token_is_refused(on):
    """F01: a token edited to another book, or its question edited, keeps its
    old signature and is refused."""
    token = _issue()
    assert _read(_forged(token, c="OTHER||"), book="OTHER||").lapsed == convo.LAPSED_TAMPERED
    assert _read(_forged(token, l="Show every borrower.")).lapsed == convo.LAPSED_TAMPERED
    assert _read("not-a-token").lapsed == convo.LAPSED_MALFORMED


def test_a_phase_one_token_is_refused_as_malformed(on):
    """A version-1 token can only be one issued before this deployment."""
    import hashlib
    import hmac
    body = {"v": 1, "k": "ask_back", "p": "user-a", "c": "ERE||", "h": "chat-1",
            "t": int(_NOW), "q": "q", "a": "a", "r": []}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    signature = base64.urlsafe_b64encode(
        hmac.new(_KEY.encode(), payload, hashlib.sha256).digest()).decode().rstrip("=")
    token = base64.urlsafe_b64encode(payload).decode().rstrip("=") + "." + signature
    assert _read(token).lapsed == convo.LAPSED_MALFORMED


def test_a_question_is_asked_back_about_three_times_at_most(on):
    turns = tuple(("ask", "reply") for _ in range(convo.max_asks()))
    pending = convo.PendingAsk(question="q", ask="a", turns=turns)
    assert convo.issue_ask_back(principal="user-a", book="ERE||", chat="chat-1",
                                pending=pending) is None
    # The last answered question is still held; the ask is not.
    held = _read(_issue(convo.Memory(last=_LAST, pending=pending))).memory
    assert held == convo.Memory(last=_LAST)


def test_the_lapse_is_stated():
    assert convo.lapsed_notice(convo.LAPSED_EXPIRED).startswith(
        "More than 5 minutes passed since I asked")
    assert "read this on its own" in convo.lapsed_notice(convo.LAPSED_OTHER_BOOK)
    assert "read it on its own" in convo.lapsed_notice(convo.LAPSED_UNREAD)


# --------------------------------------------------------------------------- #
# the conversation reader
# --------------------------------------------------------------------------- #

_MEMORY = convo.Memory(last="What is the 8-week completion run rate?")


def _check(payload, message="And 12?", memory=_MEMORY, choices=""):
    return reader.check(payload, message=message, memory=memory, choices=choices)


def test_a_complete_reading_is_used():
    got = _check({"outcome": "complete",
                  "question": "What is the 12-week completion run rate?"})
    assert got.ok and got.question == "What is the 12-week completion run rate?"


@pytest.mark.parametrize("payload, why", [
    (None, "no reading"),
    ({"outcome": "answer"}, "outcome"),
    ({"outcome": "complete"}, "no question"),
    ({"outcome": "complete", "question": "q" * 401}, "too long"),
    ({"outcome": "ask"}, "no detail"),
    ({"outcome": "complete", "question": "q", "figure": 1}, "unexpected"),
])
def test_a_malformed_reading_is_not_used(payload, why):
    got = _check(payload)
    assert not got.ok and why in got.error


def test_the_reader_is_told_the_d25_reading_conventions():
    """§39.1: "and" or "what about" before a grouping does not add it — a new
    grouping replaces the earlier one unless the message says in words to add
    it; a change taken back is the earlier question without it; and the
    reader asks only what a message refers to, never what it means (the
    interpreter holds the definitions and defaults)."""
    from mi_agent.interpretation_v2.conversation_reader import SYSTEM_PROMPT
    rules = " ".join(SYSTEM_PROMPT.split())
    assert "A new grouping replaces the earlier question's grouping, however" in rules
    assert "only when the message says so in words" in rules
    assert "takes a change back" in rules
    assert "Never ask what a word or phrase means" in rules
    assert "keep the user's own words for it in the complete question" in rules


def test_a_reading_may_not_add_a_number_the_conversation_does_not_hold():
    """Rule 3, enforced: no period, amount or threshold nobody stated."""
    got = _check({"outcome": "complete",
                  "question": "What is the 12-week completion run rate in 2027?"})
    assert not got.ok and "2027" in got.error
    # Numbers the message or the memory hold are the conversation's own.
    assert _check({"outcome": "complete",
                   "question": "Compare the 8-week and 12-week run rates."}).ok


def test_an_ask_may_name_a_closed_lists_values_and_a_question_may_not():
    choices = json.dumps({"Age Bucket": ["70-75", "75-80"]})
    ask = _check({"outcome": "ask", "question": "",
                  "ask": "Which age band: 70-75 or 75-80?"},
                 message="Just that band", choices=choices)
    assert ask.ok
    invented = _check({"outcome": "complete",
                       "question": "Show balance for the 75-80 age band."},
                      message="Just that band", choices=choices)
    assert not invented.ok


def test_carried_is_the_question_differing_from_the_message():
    assert not reader.carried("What is the pipeline amount?",
                              "what is the pipeline amount")
    assert reader.carried("By broker?", "What is the pipeline amount by broker?")


def test_the_reader_is_shown_the_conversation_in_order():
    pending = convo.PendingAsk(question="Q0", ask="A1", turns=(("A0", "R0"),))
    text = reader.build_user_prompt("R1", convo.Memory(last="L", pending=pending))
    assert (text.index("L") < text.index("Q0") < text.index("A0")
            < text.index("R0") < text.index("A1") < text.index("R1"))
    assert "(none in this conversation)" in reader.build_user_prompt(
        "R1", convo.Memory(pending=pending))


def test_the_reader_sees_no_data():
    """Names and rules only: no figure, no row, no client."""
    view = json.dumps(reader.reader_view())
    for forbidden in ("£", "client_001", "current_outstanding_balance"):
        assert forbidden not in view


class _Answers:
    """A client that returns one reader payload, and records what it was shown."""

    def __init__(self, payload=None, error=""):
        self.payload, self.error, self.seen = payload, error, []

    def emit_intent(self, *, system, user, tool_schema, tool_name, **_):
        self.seen.append(user)
        return ModelResponse(payload=self.payload, error=self.error,
                             model_id="claude-opus-5")


def test_the_reader_reads_through_its_own_tool():
    client = _Answers({"outcome": "complete", "question": "What is X by broker?"})
    got = reader.ConversationReader(client).read("By broker?", _MEMORY)
    assert got.ok and got.question == "What is X by broker?"
    assert "By broker?" in client.seen[0]


# --------------------------------------------------------------------------- #
# the interpreter reads complete questions only
# --------------------------------------------------------------------------- #

def test_the_interpreter_has_one_view():
    """A follow-up never reaches the interpreter as anything but a complete
    question: there is no second user message for it to be shown (§37)."""
    assert not hasattr(opus_interpreter, "build_reply_prompt")

    class _Capturing:
        users = []

        def emit_intent(self, *, user, **_):
            self.users.append(user)
            return ModelResponse(payload=None, error="captured")

    client = _Capturing()
    OpusInterpreter(client).interpret("What is the pipeline amount by broker?")
    assert client.users == [build_user_prompt("What is the pipeline amount by broker?")]


# --------------------------------------------------------------------------- #
# served
# --------------------------------------------------------------------------- #

class _Interpreter:
    def __init__(self, payload, seen):
        self.payload, self.seen = payload, seen

    def interpret(self, question, **_):
        self.seen.append(question)
        return InterpretationOutcome(
            question=question, intent=parse_candidate_intent(dict(self.payload)),
            model_id="claude-opus-5", raw_payload=dict(self.payload))


class _Reader:
    def __init__(self, reading=None, raises=False):
        self.reading, self.raises, self.calls = reading, raises, []

    def read(self, message, memory):
        self.calls.append((message, memory))
        if self.raises:
            raise RuntimeError("reader down")
        return self.reading


class _Principal:
    actor_id = "canary-principal"


def _turn(message, token=None, reading=None, raises=False, monkeypatch=None):
    fake = _Reader(reading, raises)
    canary.set_reader_factory(lambda: fake)
    try:
        return canary.read_turn(message, token=token, principal="canary-principal",
                                book="ERE||", chat="chat-1"), fake
    finally:
        canary.set_reader_factory(None)


def _serve(payload, monkeypatch, turn, question=None):
    from mi_agent import plan_shadow_evidence as evidence
    from mi_agent import plan_shadow_wiring as wiring
    monkeypatch.setenv("MI_AGENT_PLAN_SERVE", "canary")
    monkeypatch.setenv("MI_AGENT_PLAN_SERVE_PRINCIPALS", "canary-principal")
    seen, written = [], []
    wiring.set_interpreter_factory(lambda: _Interpreter(payload, seen))
    monkeypatch.setattr(evidence, "write", lambda body: written.append(body))
    try:
        out = canary.respond(
            question=question or (turn.question if turn else "q"),
            context=_Principal(), client_id=_CLIENT, run_id=None,
            legacy_result={"ok": True}, frame=None, semantics={}, view="funded",
            portfolio_id=f"{_CLIENT}/2025-10-01",
            render_portfolio_id=f"{_CLIENT}/2025-10-01", as_of=None,
            pipeline_source=_SOURCE, pipeline_root=_HISTORY_ROOT,
            pipeline_client_id=_CLIENT, pipeline_history=None,
            conversation=turn)
    finally:
        wiring.set_interpreter_factory(None)
    return out, (written[-1] if written else {}), seen


def _held(out):
    return convo.read(out["conversation"]["continuation"],
                      principal="canary-principal", book="ERE||",
                      chat="chat-1").memory


_ASKS_BACK = dict(_PIPELINE_INTENT, ambiguity=[
    {"slot": "comparison", "note": "Compare with what is not stated.",
     "blocking": True}])
_INELIGIBLE = dict(_PIPELINE_INTENT, measures=[{"concept": "pipeline_amount"},
                                               {"concept": "balance"}])


def test_an_answer_hands_back_the_question_it_answered(on, monkeypatch):
    """A stand-alone question is answered exactly as before, and the memory
    it hands back is that question, for a follow-up to build on."""
    turn, fake = _turn("What is the pipeline amount?")
    out, record, seen = _serve(_PIPELINE_INTENT, monkeypatch, turn)
    assert fake.calls == []                       # nothing to read it with
    assert seen == ["What is the pipeline amount?"]
    assert out["answer"].startswith("The live pipeline amount")
    assert out["conversation"]["kind"] == convo.KIND_FOLLOW_UP
    assert _held(out) == convo.Memory(last="What is the pipeline amount?")
    assert record["conversation"]["continuation_issued"] is True


def test_a_follow_up_is_answered_as_its_complete_question(on, monkeypatch):
    token = _issue(convo.Memory(last="What is the pipeline amount?"),
                   principal="canary-principal", now=None)
    full = "What is the pipeline amount at Offer stage?"
    turn, fake = _turn("Just the Offers", token,
                       reader.Reading(outcome=reader.COMPLETE, question=full))
    assert fake.calls[0][1] == convo.Memory(last="What is the pipeline amount?")
    out, record, seen = _serve(_PIPELINE_INTENT, monkeypatch, turn)
    assert seen == [full]                         # the interpreter's only view
    assert out["answer"].startswith(
        f"Following on from your previous question, I read this as “{full}”.")
    assert out["conversation"]["readAs"] == full
    assert _held(out) == convo.Memory(last=full)
    assert record["conversation"]["read_as"] == full
    assert record["conversation"]["kind"] == "follow_up"


def test_a_complete_question_carries_nothing(on, monkeypatch):
    """D24: returned word for word, read exactly as on its own, and the
    answer claims no carry."""
    token = _issue(convo.Memory(last="Show balance by region."),
                   principal="canary-principal", now=None)
    message = "What is the pipeline amount?"
    turn, _ = _turn(message, token,
                    reader.Reading(outcome=reader.COMPLETE, question=message))
    out, record, seen = _serve(_PIPELINE_INTENT, monkeypatch, turn)
    assert seen == [message]
    assert out["answer"].startswith("The live pipeline amount")
    assert "readAs" not in out["conversation"]
    assert _held(out) == convo.Memory(last=message)


def test_a_reply_is_answered_as_the_question_it_completes(on, monkeypatch):
    token = _issue(convo.Memory(pending=_ASK), principal="canary-principal",
                   now=None)
    full = "Compare the pipeline amount with the previous extract."
    turn, fake = _turn("With the previous extract", token,
                       reader.Reading(outcome=reader.COMPLETE, question=full))
    assert fake.calls[0][1].pending == _ASK
    out, _, seen = _serve(_PIPELINE_INTENT, monkeypatch, turn)
    assert seen == [full]
    assert out["answer"].startswith(f"With your reply, I read your question as “{full}”.")


def test_an_ask_back_keeps_the_last_answer_and_opens_an_ask(on, monkeypatch):
    token = _issue(convo.Memory(last=_LAST), principal="canary-principal", now=None)
    turn, _ = _turn("Compare the pipeline amount.", token,
                    reader.Reading(outcome=reader.COMPLETE,
                                   question="Compare the pipeline amount."))
    out, record, _ = _serve(_ASKS_BACK, monkeypatch, turn)
    assert out["answer"].startswith(
        "I need one more detail before I can answer: Compare with what is not stated.")
    assert out["answer"].endswith("Reply with it and I will answer your question.")
    assert out["conversation"]["kind"] == convo.KIND_ASK_BACK
    assert out["conversation"]["expiresInSeconds"] == 300
    assert _held(out) == convo.Memory(last=_LAST, pending=_ASK)


def test_a_decline_does_not_replace_the_memory(on, monkeypatch):
    """D25: the next follow-up is read against the last ANSWERED question."""
    token = _issue(convo.Memory(last=_LAST), principal="canary-principal", now=None)
    full = "What is the pipeline amount and balance by stage?"
    turn, _ = _turn("And the balance?", token,
                    reader.Reading(outcome=reader.COMPLETE, question=full))
    out, _, _ = _serve(_INELIGIBLE, monkeypatch, turn)
    assert out["ok"] is False
    assert _held(out) == convo.Memory(last=_LAST)


def test_the_reader_asks_for_the_one_detail_itself(on, monkeypatch):
    """An unclear follow-up is asked about, never guessed — and no plan runs."""
    from mi_agent import plan_shadow_evidence as evidence
    monkeypatch.setenv("MI_AGENT_PLAN_SERVE", "canary")
    monkeypatch.setenv("MI_AGENT_PLAN_SERVE_PRINCIPALS", "canary-principal")
    written = []
    monkeypatch.setattr(evidence, "write", lambda body: written.append(body))
    token = _issue(convo.Memory(last="Show balance by region."),
                   principal="canary-principal", now=None)
    turn, _ = _turn("And that one?", token,
                    reader.Reading(outcome=reader.ASK, ask="Which region do you mean?"))
    assert turn.immediate
    out = canary.converse_without_plan(turn, context=_Principal(), view="funded")
    assert out["ok"] is False
    assert out["answer"].startswith(
        "I need one more detail before I can answer: Which region do you mean?")
    assert out["answer"].endswith("Reply with it and I will answer your question.")
    held = _held(out)
    assert held.last == "Show balance by region."
    assert held.pending == convo.PendingAsk(question="And that one?",
                                            ask="Which region do you mean?")
    assert written[-1]["serving"]["reason"] == canary.CLARIFY_CONVERSATION


def test_a_reference_the_memory_does_not_hold_is_declined(on, monkeypatch):
    """Memory holds one answer (D24): the agent says so rather than guessing."""
    from mi_agent import plan_shadow_evidence as evidence
    monkeypatch.setenv("MI_AGENT_PLAN_SERVE", "canary")
    monkeypatch.setenv("MI_AGENT_PLAN_SERVE_PRINCIPALS", "canary-principal")
    monkeypatch.setattr(evidence, "write", lambda body: None)
    token = _issue(convo.Memory(last="Show balance by region."),
                   principal="canary-principal", now=None)
    turn, _ = _turn("Compare it with an earlier answer", token,
                    reader.Reading(outcome=reader.CANNOT, reason="older answer"))
    out = canary.converse_without_plan(turn, context=_Principal(), view="funded")
    assert "I keep only the last question I answered" in out["answer"]
    assert "The last question I answered was “Show balance by region.”" in out["answer"]
    assert _held(out) == convo.Memory(last="Show balance by region.")


@pytest.mark.parametrize("reading, raises", [
    (reader.Reading(error="it added ['2027']"), False),
    (None, True),
])
def test_a_message_the_reader_could_not_read_is_read_on_its_own(
        on, monkeypatch, reading, raises):
    token = _issue(convo.Memory(last=_LAST), principal="canary-principal", now=None)
    turn, _ = _turn("What is the pipeline amount?", token, reading, raises)
    assert turn.lapsed == convo.LAPSED_UNREAD and turn.question == turn.message
    out, _, seen = _serve(_PIPELINE_INTENT, monkeypatch, turn)
    assert seen == ["What is the pipeline amount?"]
    assert out["answer"].startswith(
        "I could not read this together with your earlier question")


def test_a_lapsed_memory_is_not_read_and_says_so(on, monkeypatch):
    token = _issue(convo.Memory(last=_LAST), principal="canary-principal",
                   now=_NOW)                          # long expired
    turn, fake = _turn("What is the pipeline amount?", token)
    assert fake.calls == [] and turn.lapsed == convo.LAPSED_EXPIRED
    out, _, _ = _serve(_PIPELINE_INTENT, monkeypatch, turn)
    assert out["answer"].startswith("More than 5 minutes passed since I asked")


def test_the_clock_runs_from_delivery_not_from_the_question(on, monkeypatch):
    """E03: a slow answer does not use up the five minutes."""
    clock = {"now": _NOW}
    monkeypatch.setattr(convo.time, "time", lambda: clock["now"])

    class _Slow(_Interpreter):
        def interpret(self, question, **kw):
            clock["now"] += 300                       # the answer takes 5 minutes
            return super().interpret(question, **kw)

    from mi_agent import plan_shadow_wiring as wiring
    turn, _ = _turn("What is the pipeline amount by stage?")
    monkeypatch.setattr(wiring, "set_interpreter_factory",
                        lambda f, _real=wiring.set_interpreter_factory:
                        _real(f if f is None else (lambda: _Slow(_PIPELINE_INTENT, []))))
    out, _, _ = _serve(_PIPELINE_INTENT, monkeypatch, turn)
    clock["now"] += 240                               # replied 4 minutes later
    assert convo.read(out["conversation"]["continuation"],
                      principal="canary-principal", book="ERE||",
                      chat="chat-1").ok


def test_switched_off_an_answer_is_what_it_was(monkeypatch):
    monkeypatch.delenv(convo.SWITCH_ENV, raising=False)
    assert canary.read_turn("q", token="t", principal="p", book="b", chat="c") is None
    out, _, _ = _serve(_ASKS_BACK, monkeypatch, None)
    assert "conversation" not in out
    assert "Reply with it" not in out["answer"]
    assert out["answer"].endswith(
        "Nothing was guessed, and no other figure was put in its place.")
