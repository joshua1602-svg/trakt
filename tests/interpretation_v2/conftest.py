"""Shared fixtures for the interpretation/compiler tests."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import pytest

from mi_agent.interpretation_v2 import (
    INTENT_SCHEMA_VERSION,
    CompilerContext,
    DeterministicCompiler,
    ModelResponse,
    OpusInterpreter,
    load_governed_vocabulary,
    parse_candidate_intent,
)


@pytest.fixture(scope="session")
def vocabulary():
    return load_governed_vocabulary()


@pytest.fixture()
def compiler(vocabulary):
    return DeterministicCompiler(CompilerContext(vocabulary))


def intent_payload(**overrides: Any) -> Dict[str, Any]:
    """A minimal valid payload, with slots overridden per test.

    Kept as a function rather than a fixture so a test can build several
    variants in one body — most of these tests are about the difference between
    two payloads.
    """
    payload: Dict[str, Any] = {
        "schema_version": INTENT_SCHEMA_VERSION,
        "capability": "generic_analysis",
        "operation": "point_in_time",
        "measures": [{"concept": "balance", "statistic": "sum"}],
        "population": {"base": "funded", "lens": "all", "seasoning": "any"},
        "time": {"form": "current"},
    }
    payload.update(overrides)
    return payload


def build_intent(**overrides: Any):
    return parse_candidate_intent(intent_payload(**overrides))


class ScriptedClient:
    """An interpreter client that returns a fixed payload or a fixed failure."""

    def __init__(self, payload: Optional[Mapping[str, Any]] = None, *,
                 error: str = "", model_id: str = "scripted") -> None:
        self.payload = payload
        self.error = error
        self.model_id = model_id
        self.calls = 0
        self.last_system = None
        self.last_user = None
        self.last_tool_schema = None
        self.last_metadata_tools = None
        self.last_dispatch = None

    def emit_intent(self, *, system, user, tool_schema, tool_name,
                    metadata_tools=(), dispatch=None):
        self.calls += 1
        self.last_metadata_tools = metadata_tools
        self.last_dispatch = dispatch
        self.last_system = system
        self.last_user = user
        self.last_tool_schema = tool_schema
        return ModelResponse(payload=self.payload, model_id=self.model_id,
                             error=self.error)


@pytest.fixture()
def scripted_interpreter(vocabulary):
    def _make(payload=None, *, error: str = ""):
        client = ScriptedClient(payload, error=error)
        return OpusInterpreter(client, vocabulary=vocabulary), client
    return _make
