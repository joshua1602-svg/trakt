"""The model is shown METADATA, and never any data.

Opus now reads Trakt's governed registries through retrieval tools, so the
surface to police is wider than one prompt: it is the standing block, the tool
definitions, AND every result any tool can return. All of it is walked here.

This is the claim the acceptance report makes as LOAN_ROWS_SENT_TO_MODEL = 0
and PORTFOLIO_VALUES_SENT_TO_MODEL = 0, so it is measured rather than asserted
in prose.

The sharpest case is the funding facility. ``config/risk/funding_facilities.yaml``
carries a £250m commitment, an advance rate and a concentration floor two keys
away from the facility's name, and the addendum permits only "presence/absence
of a funding facility". The adapter allowlists the identity keys and drops the
rest; ``test_a_facility_is_present_without_its_figures`` is what keeps that true
when somebody adds a key upstream.
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
)
from mi_agent.interpretation_v2.metadata import (
    GovernedMetadataService,
    metadata_tool_schemas,
    portfolio_semantic_context,
)

from .conftest import ScriptedClient, intent_payload


def _every_tool_result(vocabulary) -> list:
    """Everything the metadata surface can hand back, in one list.

    Broad on purpose: a search that returns every concept, metadata for a
    sample across all roles, values for every dimension, all capabilities, and
    the two environment tools. If a portfolio value can reach the model at all,
    it is in here.
    """
    service = GovernedMetadataService(vocabulary)
    results = [service.get_asset_metadata(),
               service.get_portfolio_semantic_context(),
               service.search_capabilities(limit=40),
               service.search_concepts("", limit=40)]
    for query in ("balance", "ltv", "region", "borrower", "stage", "product",
                  "arrears", "facility", "borrowing base", "concentration"):
        results.append(service.search_concepts(query, limit=40))
    for concept in vocabulary.concepts.values():
        results.append(service.get_concept_metadata(concept.concept_id))
        if concept.role == "dimension":
            results.append(service.get_allowed_values(concept.concept_id))
    for capability in sorted(vocabulary.capabilities):
        results.append(service.get_capability_metadata(capability))
    return results


def _whole_surface(vocabulary, question: str = "What is our total balance?") -> str:
    return json.dumps({
        "system": build_system_blocks(vocabulary),
        "user": build_user_prompt(question),
        "tools": metadata_tool_schemas() + [build_tool_schema()],
        "tool_results": _every_tool_result(vocabulary),
    }, default=str)


def test_no_loan_row_or_portfolio_value_can_reach_the_model(vocabulary):
    surface = _whole_surface(vocabulary)

    assert not re.search(r"[£$€]\s?\d", surface), "a currency figure reached the model"
    assert not re.search(r"\b\d{1,3}(,\d{3})+\b", surface), "a formatted total reached the model"
    assert not re.search(r"\bLN[-_]?\d{4,}\b", surface, re.IGNORECASE)
    # Nothing with the magnitude of a balance, a commitment or a valuation.
    big = [n for n in re.findall(r"\b\d{6,}\b", surface)]
    assert big == [], f"large bare numbers reached the model: {big[:5]}"
    for banned in ("loan_id", "borrower_name", "account_number", "csv",
                   "head()", "to_dict(", "iloc", "read_sql"):
        assert banned not in surface.lower(), f"{banned!r} reached the model"


def test_a_facility_is_present_without_its_figures(vocabulary):
    """Presence is governed context. The commitment is a position."""
    context = GovernedMetadataService(vocabulary).get_portfolio_semantic_context()
    assert context["funding_facility_configured"] is True
    assert context["funding_facilities"], "premise: a facility is configured"

    raw = json.dumps(context, default=str)
    for figure in ("commitment", "advance_rate", "current_drawn_amount",
                   "concentration_denominator_floor", "250000000", "33000000"):
        assert figure not in raw, f"{figure!r} leaked with the facility"
    for facility in context["funding_facilities"]:
        assert set(facility) <= {"facility_id", "facility_label", "facility_type",
                                 "currency", "client_id"}


def test_no_reporting_date_or_snapshot_handle_reaches_the_model(vocabulary):
    """Time is stated semantically and resolved downstream.

    The client config carries a static reporting date. Handing it over would let
    the model pin the period an answer covers, from a prompt with no data in it.
    """
    surface = _whole_surface(vocabulary)
    assert "2025-11-30" not in surface
    assert not re.search(r"\b\d{4}-\d{2}-\d{2}\b", surface), \
        "a calendar date reached the model"
    assert "static_reporting_date" not in surface
    assert "reporting_date" not in json.dumps(portfolio_semantic_context())


def test_every_metadata_tool_is_read_only(vocabulary):
    """There is no tool that writes, executes, or fetches a row."""
    names = {tool["name"] for tool in metadata_tool_schemas()}
    assert names == {"search_concepts", "get_concept_metadata",
                     "get_allowed_values", "search_capabilities",
                     "get_capability_metadata", "get_asset_metadata",
                     "get_portfolio_semantic_context"}
    for tool in metadata_tool_schemas():
        blob = json.dumps(tool).lower()
        for verb in ("write", "update", "delete", "execute", "run_", "query_data",
                     "fetch_rows", "sample", "preview"):
            assert verb not in blob, f"{tool['name']} mentions {verb!r}"


def test_an_unknown_tool_is_an_error_not_a_guess(vocabulary):
    service = GovernedMetadataService(vocabulary)
    result = service.call("read_loan_tape", {"portfolio": "x"})
    assert "error" in result
    assert result["available"]


def test_the_interpreter_sends_exactly_what_the_builders_produce(vocabulary):
    """No other code path may add to the prompt."""
    client = ScriptedClient(intent_payload())
    interpreter = OpusInterpreter(client, vocabulary=vocabulary)
    interpreter.interpret("What is our total funded balance?")

    assert client.last_system == build_system_blocks(vocabulary)
    assert client.last_user == build_user_prompt("What is our total funded balance?")
    assert client.last_tool_schema == build_tool_schema()
    assert list(client.last_metadata_tools) == metadata_tool_schemas()


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
