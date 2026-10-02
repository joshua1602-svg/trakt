"""POST /mi/query reads a message in its conversation before anything else
reads the question (§34, §38, §39).

The continuation is untrusted caller input. The service checks it against the
TRUSTED principal and the AUTHORISED book — never against anything the body
says about who is asking — and, when the conversation reader made the message
a complete question, the rest of the request is the request for THAT
question. The checking itself is `tests/interpretation_v2/test_conversation.py`;
this proves the seam.
"""
from __future__ import annotations

from unittest import mock

from fastapi.testclient import TestClient

from mi_agent import conversation as convo
from mi_agent import plan_decline
from mi_agent import plan_serving_canary as canary
from mi_agent_api.app import app


def _declined():
    return plan_decline.envelope(question="q", body={},
                                 reason="INELIGIBLE:PERIOD_NOT_SUPPORTED",
                                 view="funded")


def _ask(body, turn_question=None, immediate=False, enabled=True):
    calls, reads, without = [], [], []

    def _respond(**kwargs):
        calls.append(kwargs)
        return _declined()

    def _read_turn(message, **kwargs):
        reads.append(dict(kwargs, message=message))
        turn = canary.Turn(message=message, principal=kwargs["principal"],
                           book=kwargs["book"], chat=kwargs["chat"],
                           question=turn_question or message)
        if immediate:
            from mi_agent.interpretation_v2 import conversation_reader as reader
            turn.reading = reader.Reading(outcome=reader.ASK, ask="Which one?")
        return turn

    def _without(turn, **kwargs):
        without.append(turn)
        return _declined()

    with mock.patch.object(canary, "handles", return_value=True), \
            mock.patch.object(convo, "enabled", return_value=enabled), \
            mock.patch.object(canary, "respond", side_effect=_respond), \
            mock.patch.object(canary, "read_turn", side_effect=_read_turn), \
            mock.patch.object(canary, "converse_without_plan", side_effect=_without), \
            TestClient(app, raise_server_exceptions=False) as client:
        client.post("/mi/query", json=body)
    return calls, reads, without


def test_the_memory_is_checked_against_the_trusted_principal_and_book():
    calls, reads, _ = _ask({"question": "And by broker?", "continuation": "tok",
                            "conversationId": "chat-1"})
    assert reads and reads[0]["token"] == "tok" and reads[0]["chat"] == "chat-1"
    turn = calls[0]["conversation"]
    # The principal checked is the authenticated one, and the book the
    # authorised one — the body named neither.
    assert reads[0]["principal"] == canary.principal_of(calls[0]["context"])
    assert reads[0]["book"] == turn.book
    assert turn.message == "And by broker?"


def test_the_request_is_the_request_for_the_complete_question():
    full = "What is the pipeline amount by broker?"
    calls, _, _ = _ask({"question": "And by broker?", "continuation": "tok"},
                       turn_question=full)
    assert calls[0]["question"] == full
    assert calls[0]["conversation"].question == full


def test_a_turn_the_reader_settled_runs_no_plan():
    calls, _, without = _ask({"question": "And that one?", "continuation": "tok"},
                             immediate=True)
    assert without and without[0].message == "And that one?"
    assert calls == []


def test_switched_off_the_question_is_a_stand_alone_question():
    calls, reads, _ = _ask({"question": "What is the pipeline amount?",
                            "continuation": "tok"}, enabled=False)
    assert reads == []
    assert calls[0]["conversation"] is None
    assert calls[0]["question"] == "What is the pipeline amount?"
