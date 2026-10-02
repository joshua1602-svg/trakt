"""Conversation, phase 1: the reply to an ask-back (P0 design §34, D24, D25).

When the governed path cannot answer without one more detail it asks for it;
the reply cannot stand on its own, and read with the question it answers it is
a complete question. These tests hold the phase to its rules:

  - the memory is a SIGNED token bound to the user, the book and the chat, for
    `memory_minutes` from the delivery of the ask-back — anything else is
    refused and the reply is read on its own, and says so;
  - a reply reaches the model with the question it answers; a stand-alone
    question's view is the signed-off baseline's, untouched;
  - switched off, nothing differs from before.
"""
from __future__ import annotations

import base64
import json

import pytest

from mi_agent import conversation as convo
from mi_agent.interpretation_v2.opus_interpreter import (
    InterpretationOutcome, OpusInterpreter, build_reply_prompt, build_user_prompt)
from tests.interpretation_v2.test_specialist_runtime_pipeline import (
    _PIPELINE_INTENT, _served)

_KEY = "k" * 40
_NOW = 1_800_000_000.0
_ASK = convo.PendingAsk(question="Compare the pipeline amount.",
                        ask="Compare with what is not stated.")


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(convo.SWITCH_ENV, "on")
    monkeypatch.setenv(convo.KEY_ENV, _KEY)


def _issue(pending=_ASK, **over):
    args = dict(principal="user-a", book="ERE||", chat="chat-1", pending=pending,
                now=_NOW)
    args.update(over)
    return convo.issue_ask_back(**args)


def _read(token, **over):
    args = dict(principal="user-a", book="ERE||", chat="chat-1", now=_NOW + 60)
    args.update(over)
    return convo.read(token, **args)


# --------------------------------------------------------------------------- #
# the token
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


def test_the_reply_reads_back_the_question_it_answers(on):
    returned = _read(_issue())
    assert returned.ok and returned.pending == _ASK


def test_the_memory_lasts_five_minutes_from_delivery(on):
    """D24: the clock runs from the delivery of the ask-back (its issue)."""
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
    """F05 (another user), E04 (another book), cleared chat (a new chat id)."""
    assert _read(_issue(), **over).lapsed == reason


def test_an_edited_token_is_refused(on):
    """F01: a token edited to another book keeps its old signature."""
    encoded, signature = _issue().split(".", 1)
    body = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    body["c"] = "OTHER||"
    forged = base64.urlsafe_b64encode(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    assert _read(f"{forged}.{signature}", book="OTHER||").lapsed == convo.LAPSED_TAMPERED
    assert _read("not-a-token").lapsed == convo.LAPSED_MALFORMED


def test_a_question_is_asked_back_about_three_times_at_most(on):
    turns = tuple(("ask", "reply") for _ in range(convo.max_asks()))
    assert _issue(convo.PendingAsk(question="q", ask="a", turns=turns)) is None


def test_the_lapse_is_stated():
    assert convo.lapsed_notice(convo.LAPSED_EXPIRED).startswith(
        "More than 5 minutes passed since I asked")
    assert "read this on its own" in convo.lapsed_notice(convo.LAPSED_OTHER_BOOK)


# --------------------------------------------------------------------------- #
# what the model reads
# --------------------------------------------------------------------------- #

def test_a_reply_is_read_with_the_question_it_answers():
    pending = convo.PendingAsk(question="Q0", ask="A1", turns=(("A0", "R0"),))
    text = build_reply_prompt("R1", pending)
    assert text.index("Q0") < text.index("A0") < text.index("R0") \
        < text.index("A1") < text.index("R1")
    assert "record that question alone" in text   # D24: a complete question never inherits


class _Capturing:
    def __init__(self):
        self.users = []

    def emit_intent(self, *, system, user, tool_schema, tool_name, **_):
        from mi_agent.interpretation_v2.opus_interpreter import ModelResponse
        self.users.append(user)
        return ModelResponse(payload=None, error="captured")


def test_a_stand_alone_question_is_read_exactly_as_before():
    client = _Capturing()
    interpreter = OpusInterpreter(client)
    interpreter.interpret("What is the pipeline amount?")
    interpreter.interpret("With the previous extract", reply_to=_ASK)
    assert client.users[0] == build_user_prompt("What is the pipeline amount?")
    assert client.users[1] == build_reply_prompt("With the previous extract", _ASK)


# --------------------------------------------------------------------------- #
# served
# --------------------------------------------------------------------------- #

_ASKS_BACK = dict(_PIPELINE_INTENT, ambiguity=[
    {"slot": "comparison", "note": "Compare with what is not stated.",
     "blocking": True}])


def test_an_ask_back_hands_back_a_continuation(on, monkeypatch):
    out, record = _served(_ASKS_BACK, monkeypatch, decline=True,
                          question="Compare the pipeline amount.",
                          conversation_book="ERE||", conversation_chat="chat-1")
    assert out["answer"].startswith(
        "I need one more detail before I can answer: Compare with what is not stated.")
    assert out["answer"].endswith("Reply with it and I will answer your question.")
    token = out["conversation"]["continuation"]
    assert out["conversation"]["expiresInSeconds"] == 300
    returned = convo.read(token, principal="canary-principal", book="ERE||",
                          chat="chat-1")
    assert returned.pending == convo.PendingAsk(
        question="Compare the pipeline amount.",
        ask="Compare with what is not stated.")
    assert record["conversation"]["continuation_issued"] is True


def test_the_reply_is_answered_and_says_what_it_read(on, monkeypatch):
    seen = {}

    class _Scripted:
        def interpret(self, question, **kwargs):
            from mi_agent.interpretation_v2.intent import parse_candidate_intent
            seen.update(question=question, **kwargs)
            return InterpretationOutcome(
                question=question,
                intent=parse_candidate_intent(dict(_PIPELINE_INTENT)),
                model_id="claude-opus-5", raw_payload=dict(_PIPELINE_INTENT))

    from mi_agent import plan_shadow_wiring as wiring
    monkeypatch.setattr(wiring, "set_interpreter_factory",
                        lambda f, _real=wiring.set_interpreter_factory:
                        _real(f if f is None else (lambda: _Scripted())))
    out, record = _served(_PIPELINE_INTENT, monkeypatch, decline=True,
                          question="With the previous extract", reply_to=_ASK)
    assert seen["question"] == "With the previous extract"
    assert seen["reply_to"] == _ASK
    assert out["answer"].startswith("With your reply, I read your earlier question as ")
    assert record["conversation"]["reply_to"]["question"] == _ASK.question


def test_a_lapsed_question_is_answered_on_its_own_and_says_so(on, monkeypatch):
    out, record = _served(_PIPELINE_INTENT, monkeypatch, decline=True,
                          conversation_lapsed=convo.LAPSED_EXPIRED)
    assert out["answer"].startswith("More than 5 minutes passed since I asked")
    assert record["conversation"]["lapsed"] == convo.LAPSED_EXPIRED


def test_switched_off_an_ask_back_is_what_it_was(monkeypatch):
    monkeypatch.delenv(convo.SWITCH_ENV, raising=False)
    out, _record = _served(_ASKS_BACK, monkeypatch, decline=True)
    assert "conversation" not in out
    assert "Reply with it" not in out["answer"]
    assert out["answer"].endswith(
        "Nothing was guessed, and no other figure was put in its place.")
