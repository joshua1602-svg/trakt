"""The model is shown a vocabulary, and never any data.

Asserted over the exact payload the interpreter sends. This is the claim the
acceptance report makes as LOAN_ROWS_SENT_TO_MODEL = 0 and
PORTFOLIO_VALUES_SENT_TO_MODEL = 0, so it is measured rather than asserted in
prose.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from mi_agent.interpretation_v2 import (
    OpusInterpreter,
    build_system_blocks,
    build_tool_schema,
    build_user_prompt,
    canonical_field_names,
)

from .conftest import ScriptedClient, intent_payload


def _full_payload(vocabulary, question: str) -> str:
    return json.dumps({
        "system": build_system_blocks(vocabulary),
        "user": build_user_prompt(question),
        "tools": [build_tool_schema()],
    })


def test_the_prompt_contains_no_loan_rows_and_no_portfolio_values(vocabulary):
    payload = _full_payload(vocabulary, "What is our total funded balance?")

    # No currency figures, no thousands-separated numbers, no loan identifiers.
    assert not re.search(r"[£$€]\s?\d", payload), "a currency figure reached the prompt"
    assert not re.search(r"\b\d{1,3}(,\d{3})+\b", payload), "a formatted total reached the prompt"
    assert not re.search(r"\bLN[-_]?\d{4,}\b", payload, re.IGNORECASE)
    for banned in ("loan_id", "borrower_name", "account_number", "dataframe",
                   "csv", "rows=", "head()", "to_dict("):
        assert banned not in payload.lower(), f"{banned!r} reached the prompt"


def test_the_prompt_exposes_no_canonical_field_that_is_not_a_business_term(vocabulary):
    payload = _full_payload(vocabulary, "anything")
    business_terms = {c.term for c in vocabulary.concepts.values()}
    leaked = [f for f in canonical_field_names()
              if f not in business_terms and f'"{f}"' in payload]
    assert leaked == [], f"canonical fields exposed to the model: {leaked}"


def test_the_vocabulary_payload_carries_no_binding(vocabulary):
    payload = vocabulary.prompt_payload()

    def walk(node, path="root"):
        if isinstance(node, dict):
            for key, value in node.items():
                assert key not in ("canonical_field", "field", "column",
                                   "snapshot", "sql"), f"{path}.{key}"
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(payload)


def test_the_interpreter_sends_exactly_what_the_builders_produce(vocabulary):
    """No other code path may add to the prompt."""
    client = ScriptedClient(intent_payload())
    interpreter = OpusInterpreter(client, vocabulary=vocabulary)
    interpreter.interpret("What is our total funded balance?")

    assert client.last_system == build_system_blocks(vocabulary)
    assert client.last_user == build_user_prompt("What is our total funded balance?")
    assert client.last_tool_schema == build_tool_schema()


def test_a_dataframe_can_never_reach_the_interpreter(vocabulary):
    """There is no parameter to pass one through."""
    import inspect

    signature = inspect.signature(OpusInterpreter.interpret)
    assert list(signature.parameters) == ["self", "question"]

    frame = pd.DataFrame({"current_outstanding_balance": [1_000_000.0]})
    client = ScriptedClient(intent_payload())
    interpreter = OpusInterpreter(client, vocabulary=vocabulary)
    with pytest.raises(TypeError):
        interpreter.interpret("total balance", frame)  # type: ignore[call-arg]


def test_the_user_message_is_the_question_and_nothing_else():
    question = "What is our total funded balance?"
    user = build_user_prompt(question)
    assert question in user
    assert len(user) < len(question) + 200, (
        "the user message grew something other than the question")
