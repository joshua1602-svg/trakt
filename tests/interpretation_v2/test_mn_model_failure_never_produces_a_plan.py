"""M, N · A failed or malformed model call cannot produce an executable plan.

The dangerous shape is not "the model errored". It is "the model errored and
something downstream produced an answer anyway". Every failure mode below is
therefore asserted to end at a CompileResult that carries no plan.
"""

from __future__ import annotations

import pytest

from mi_agent.interpretation_v2 import (
    OUTCOME_CLARIFY,
    OUTCOME_PLAN,
    OUTCOME_REFUSE,
    DeterministicCompiler,
    OpusInterpreter,
    UnavailableClient,
    interpret_and_compile,
)

from .conftest import ScriptedClient, intent_payload


# --------------------------------------------------------------------------- #
# M · the model could not be invoked
# --------------------------------------------------------------------------- #

def test_an_unavailable_model_produces_a_refusal_and_no_plan(vocabulary, compiler):
    interpreter = OpusInterpreter(UnavailableClient("no API key"),
                                  vocabulary=vocabulary)
    outcome, result = interpret_and_compile("What is the total balance?",
                                            interpreter, compiler)
    assert not outcome.ok
    assert result.outcome == OUTCOME_REFUSE
    assert result.codes() == ["MODEL_UNAVAILABLE"]
    assert result.plan is None


def test_a_transport_error_produces_a_refusal_and_no_plan(vocabulary, compiler):
    interpreter = OpusInterpreter(ScriptedClient(None, error="ConnectionError"),
                                  vocabulary=vocabulary)
    outcome, result = interpret_and_compile("anything", interpreter, compiler)
    assert result.outcome == OUTCOME_REFUSE
    assert result.plan is None
    assert "ConnectionError" in result.reasons[0].detail


def test_no_client_means_no_call_can_be_claimed(vocabulary):
    """A run that could not reach a model must not report a model id."""
    interpreter = OpusInterpreter(UnavailableClient(), vocabulary=vocabulary)
    outcome = interpreter.interpret("What is the total balance?")
    assert outcome.model_id == ""
    assert outcome.intent is None


# --------------------------------------------------------------------------- #
# N · the model returned something unusable
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("payload,expected_code", [
    ("not an object", "MODEL_OUTPUT_MALFORMED"),
    ({}, "INTENT_SCHEMA_VERSION_UNSUPPORTED"),
    ({"schema_version": "candidate_intent/0.9"}, "INTENT_SCHEMA_VERSION_UNSUPPORTED"),
    ({"schema_version": "candidate_intent/1.0", "capability": "nonsense",
      "operation": "point_in_time"}, "INTENT_SCHEMA_INVALID"),
    ({"schema_version": "candidate_intent/1.0", "capability": "generic_analysis",
      "operation": "teleport"}, "INTENT_SCHEMA_INVALID"),
])
def test_malformed_output_produces_a_refusal_and_no_plan(payload, expected_code,
                                                         vocabulary, compiler):
    interpreter = OpusInterpreter(ScriptedClient(payload), vocabulary=vocabulary)
    outcome, result = interpret_and_compile("anything", interpreter, compiler)
    assert not outcome.ok
    assert result.outcome in (OUTCOME_REFUSE, OUTCOME_CLARIFY)
    assert result.plan is None
    assert expected_code in result.codes()


def test_a_code_bearing_payload_produces_a_refusal_and_no_plan(vocabulary, compiler):
    payload = intent_payload(measures=[{"concept": "df['balance'].sum()"}])
    interpreter = OpusInterpreter(ScriptedClient(payload), vocabulary=vocabulary)
    outcome, result = interpret_and_compile("anything", interpreter, compiler)
    assert result.outcome == OUTCOME_REFUSE
    assert result.codes() == ["MODEL_OUTPUT_CONTAINS_CODE"]
    assert result.plan is None


def test_there_is_no_repair_loop(vocabulary, compiler):
    """One call, one verdict.

    The previous architecture re-prompted on a validation failure and accepted a
    spec in which the governed statistic had been changed. A repair loop is how
    a refusal becomes a negotiation, so this one has none — asserted by counting
    the calls.
    """
    client = ScriptedClient(intent_payload(measures=[{"concept": "ebitda"}]))
    interpreter = OpusInterpreter(client, vocabulary=vocabulary)
    outcome, result = interpret_and_compile("anything", interpreter, compiler)
    assert client.calls == 1
    assert result.outcome == OUTCOME_REFUSE
    assert result.plan is None


def test_a_wellformed_payload_still_has_to_survive_the_compiler(vocabulary, compiler):
    """Parsing is not authorisation. A perfectly-shaped intent naming a concept
    Trakt does not govern is still a refusal."""
    client = ScriptedClient(intent_payload(measures=[{"concept": "ebitda"}]))
    interpreter = OpusInterpreter(client, vocabulary=vocabulary)
    outcome, result = interpret_and_compile("anything", interpreter, compiler)
    assert outcome.ok, "the payload is schema-valid"
    assert result.outcome == OUTCOME_REFUSE
    assert result.plan is None


def test_a_valid_payload_does_produce_a_plan(vocabulary, compiler):
    """The negative tests above would pass vacuously if nothing ever planned."""
    client = ScriptedClient(intent_payload(), model_id="claude-opus-5")
    interpreter = OpusInterpreter(client, vocabulary=vocabulary)
    outcome, result = interpret_and_compile("What is the total balance?",
                                            interpreter, compiler)
    assert result.outcome == OUTCOME_PLAN
    assert result.plan.provenance.model_id == "claude-opus-5"
    assert result.plan.provenance.question == "What is the total balance?"
