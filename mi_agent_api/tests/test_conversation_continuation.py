"""POST /mi/query carries a reply's continuation to the governed path (§34
phase 1).

The continuation is untrusted caller input. The service checks it against the
TRUSTED principal and the AUTHORISED book — never against anything the body
says about who is asking — and hands the governed path the question it
answers, or why it lapsed. The checking itself is
`tests/interpretation_v2/test_conversation_phase1.py`; this proves the seam.
"""
from __future__ import annotations

from unittest import mock

from fastapi.testclient import TestClient

from mi_agent import conversation as convo
from mi_agent import plan_decline
from mi_agent import plan_serving_canary as canary
from mi_agent_api.app import app

_PENDING = convo.PendingAsk(question="Compare the pipeline amount.",
                            ask="Compare with what is not stated.")


def _declined():
    return plan_decline.envelope(question="q", body={},
                                 reason="INELIGIBLE:PERIOD_NOT_SUPPORTED",
                                 view="funded")


def _ask(body, returned):
    calls, reads = [], []

    def _respond(**kwargs):
        calls.append(kwargs)
        return _declined()

    def _read(token, **kwargs):
        reads.append(dict(kwargs, token=token))
        return returned

    with mock.patch.object(canary, "handles", return_value=True), \
            mock.patch.object(canary, "respond", side_effect=_respond), \
            mock.patch.object(convo, "read", side_effect=_read), \
            TestClient(app, raise_server_exceptions=False) as client:
        client.post("/mi/query", json=body)
    return calls, reads


def test_a_returned_continuation_reaches_the_governed_path():
    calls, reads = _ask({"question": "With the previous extract",
                         "continuation": "tok", "conversationId": "chat-1"},
                        convo.Returned(pending=_PENDING))
    assert reads and reads[0]["token"] == "tok" and reads[0]["chat"] == "chat-1"
    # The principal checked is the authenticated one, and the book the
    # authorised one — the body named neither.
    principal = canary.principal_of(calls[0]["context"])
    assert reads[0]["principal"] == principal
    assert reads[0]["book"] == calls[0]["conversation_book"]
    assert calls[0]["reply_to"] == _PENDING
    assert calls[0]["conversation_lapsed"] is None
    assert calls[0]["conversation_chat"] == "chat-1"


def test_a_lapsed_continuation_is_passed_as_lapsed():
    calls, _ = _ask({"question": "By broker?", "continuation": "old"},
                    convo.Returned(lapsed=convo.LAPSED_EXPIRED))
    assert calls[0]["reply_to"] is None
    assert calls[0]["conversation_lapsed"] == convo.LAPSED_EXPIRED


def test_a_question_without_one_is_a_stand_alone_question():
    calls, _ = _ask({"question": "What is the pipeline amount?"}, None)
    assert calls[0]["reply_to"] is None
    assert calls[0]["conversation_lapsed"] is None
