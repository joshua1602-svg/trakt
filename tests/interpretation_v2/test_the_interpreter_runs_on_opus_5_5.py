"""The interpreter's client on Claude Opus 5.5 (P0 design §40).

Claude Opus 5.5 rejects a forced `tool_choice` and always thinks. So the
intent tool is asked for in the prompt and not forced; a response without the
call is asked for once more; a safety-classifier decline is not asked again
(it is a decline, recorded with its category); effort is set explicitly; and
a decline on a false positive is re-run on Anthropic's recommended model
(server-side fallback). Tested here against a stand-in SDK — no network, no
model credit.
"""
from __future__ import annotations

import sys
import types

import pytest

from mi_agent.interpretation_v2.opus_interpreter import (
    CONFIGURED_EFFORT, CONFIGURED_MODEL, FALLBACK_BETA,
    AnthropicInterpreterClient, forces_tool)

_SYSTEM = [{"type": "text", "text": "rules"}]
_TOOL = {"name": "emit_candidate_intent", "input_schema": {"type": "object"}}


def _block(kind, **fields):
    return types.SimpleNamespace(type=kind, **fields)


def _message(*blocks, stop="end_turn", category=None):
    return types.SimpleNamespace(
        model="claude-opus-5-5", content=list(blocks), stop_reason=stop,
        stop_details=types.SimpleNamespace(category=category) if category else None,
        usage=types.SimpleNamespace(input_tokens=10, output_tokens=5,
                                    cache_read_input_tokens=100,
                                    cache_creation_input_tokens=0))


@pytest.fixture
def sdk(monkeypatch):
    """A stand-in `anthropic` module that returns scripted messages."""
    sent, script = [], []

    class _Messages:
        def create(self, **kwargs):
            sent.append(kwargs)
            return script.pop(0)

    class _Client:
        def __init__(self, **_):
            self.messages = _Messages()

    monkeypatch.setitem(sys.modules, "anthropic",
                        types.SimpleNamespace(Anthropic=_Client))
    return sent, script


def _emit(client):
    return client.emit_intent(system=_SYSTEM, user="QUESTION:\nq",
                              tool_schema=_TOOL, tool_name=_TOOL["name"])


def test_the_configured_model_is_opus_5_5_at_low_effort():
    assert CONFIGURED_MODEL == "claude-opus-5-5"
    assert CONFIGURED_EFFORT == "low"
    assert not forces_tool("claude-opus-5-5") and forces_tool("claude-opus-5")


def test_what_is_sent_to_opus_5_5(sdk):
    sent, script = sdk
    script.append(_message(_block("thinking", thinking=""),
                           _block("tool_use", name=_TOOL["name"], input={"a": 1})))
    out = _emit(AnthropicInterpreterClient(api_key="k"))
    assert out.payload == {"a": 1} and out.model_id == "claude-opus-5-5"
    kwargs = sent[0]
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["tool_choice"] == {"type": "auto"}          # never forced here
    assert kwargs["extra_body"] == {"output_config": {"effort": "low"},
                                    "fallbacks": "default"}
    assert kwargs["extra_headers"] == {"anthropic-beta": FALLBACK_BETA}
    assert "thinking" not in kwargs and "temperature" not in kwargs
    assert kwargs["max_tokens"] >= 16000                       # room to think


def test_a_response_without_the_call_is_asked_for_once_more(sdk):
    sent, script = sdk
    script += [_message(_block("text", text="Here is my reading")),
               _message(_block("tool_use", name=_TOOL["name"], input={"a": 2}))]
    out = _emit(AnthropicInterpreterClient(api_key="k"))
    assert out.payload == {"a": 2}
    assert len(sent) == 2 and out.usage["model_calls"] == 2


def test_only_once(sdk):
    sent, script = sdk
    script += [_message(_block("text", text="no")), _message(_block("text", text="no"))]
    out = _emit(AnthropicInterpreterClient(api_key="k"))
    assert out.payload is None and "no tool_use" in out.error
    assert len(sent) == 2


def test_a_safety_decline_is_recorded_and_not_asked_again(sdk):
    sent, script = sdk
    script.append(_message(stop="refusal", category="cyber"))
    out = _emit(AnthropicInterpreterClient(api_key="k"))
    assert out.payload is None and "safety classifier (cyber)" in out.error
    assert len(sent) == 1


def test_a_model_that_accepts_a_forced_call_is_still_forced(sdk):
    """Claude Opus 5 (the signed-off build) keeps its forced call and no retry."""
    sent, script = sdk
    script.append(_message(_block("text", text="no")))
    out = _emit(AnthropicInterpreterClient(api_key="k", model="claude-opus-5"))
    assert sent[0]["tool_choice"] == {"type": "tool", "name": _TOOL["name"]}
    assert len(sent) == 1 and out.payload is None
