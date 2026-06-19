"""Phase 8B — Anthropic-first LLM interpreter adapter tests.

Proves the Anthropic-first adapter turns model output into a governed, validated
MIQuerySpec v2 — and that it is safe by construction: malformed JSON, lists,
code, unknown fields, unsupported chart types, hallucinated fields, invalid
specs and client errors all degrade to a structured clarification instead of
executing anything. The LLM only proposes a spec; all validation is
deterministic. No external Anthropic calls, no API keys, no network — fake
clients only. Reuses the Phase 8A golden dataset + evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mi_agent.interpreter import (
    InterpreterContext,
    evaluate_interpretation,
    interpret,
    interpret_from_llm_output,
    interpret_with_anthropic,
    load_golden,
    parse_spec_json,
)
from mi_agent.interpreter.anthropic import (
    LLM_CLIENT_ERROR,
    LLM_HALLUCINATED_FIELD,
    LLM_MALFORMED_JSON,
    LLM_OUTPUT_CONTAINS_CODE,
    LLM_OUTPUT_NOT_OBJECT,
    LLM_UNKNOWN_FIELD,
    LLM_UNSUPPORTED_CHART_TYPE,
)
from mi_agent.interpreter.models import LLM_STUB
from mi_agent.interpreter.prompt import build_mi_spec_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = load_golden()


# --------------------------------------------------------------------------- #
# Fake (mock) clients — the only client implementations used in tests.
# --------------------------------------------------------------------------- #


class FakeClient:
    """Returns a fixed canned completion and records the prompt it received."""

    def __init__(self, response):
        self.response = response
        self.prompt = None
        self.calls = 0

    def complete_mi_spec_json(self, prompt):
        self.prompt = prompt
        self.calls += 1
        return self.response


class BoomClient:
    """Raises — proves transport errors never leak out of the adapter."""

    def complete_mi_spec_json(self, prompt):
        raise RuntimeError("network down")


def _ctx(example) -> InterpreterContext:
    return InterpreterContext(**(example.get("context_overrides") or {}))


# --------------------------------------------------------------------------- #
# 1. Golden reuse — valid examples flow through the adapter and validate
# --------------------------------------------------------------------------- #

_VALID_GOLDEN = [e for e in GOLDEN
                 if e.get("interpreter_supported") and e["expected_valid"]
                 and not e.get("expected_clarification_required")]


def _good_completion_for(ex) -> str:
    """A realistic 'good' model completion: a complete spec (with context date
    anchors resolved, as the prompt instructs) that should validate. Derived from
    the deterministic baseline so the adapter is graded against the same golden
    expectation on the same governed contract."""
    spec = interpret(ex["question"], _ctx(ex)).normalized_spec
    payload = {k: v for k, v in spec.to_dict().items()
               if v not in (None, [], {}, "none")}
    return json.dumps(payload)


@pytest.mark.parametrize("ex", _VALID_GOLDEN,
                         ids=[e["question"][:45] for e in _VALID_GOLDEN])
def test_golden_valid_via_fake_anthropic(ex):
    # The "model" returns a complete, valid spec as JSON (dates resolved from
    # context, exactly as the constrained prompt requires).
    client = FakeClient(_good_completion_for(ex))
    result = interpret_with_anthropic(ex["question"], _ctx(ex), client)
    assert result.interpretation_method == LLM_STUB
    assert client.calls == 1
    report = evaluate_interpretation(
        result,
        expected_spec=ex.get("expected_spec"),
        expected_valid=ex.get("expected_valid"),
        expected_clarification_required=False,
    )
    assert report.passed, (ex["question"], report.mismatches)
    assert result.ok and not result.clarification_required


# --------------------------------------------------------------------------- #
# 2. parse_spec_json — safe parsing of messy model output
# --------------------------------------------------------------------------- #


def test_parse_dict_passthrough():
    obj, issues = parse_spec_json({"state": "total_funded"})
    assert obj == {"state": "total_funded"} and issues == []


def test_parse_plain_json():
    obj, issues = parse_spec_json('{"state": "total_funded"}')
    assert obj["state"] == "total_funded" and not issues


def test_parse_strips_markdown_fences():
    raw = '```json\n{"state": "total_pipeline"}\n```'
    obj, issues = parse_spec_json(raw)
    assert obj == {"state": "total_pipeline"} and not issues


def test_parse_extracts_object_from_prose():
    raw = 'Sure! Here is the spec:\n{"state": "total_funded"}\nHope that helps.'
    obj, issues = parse_spec_json(raw)
    assert obj == {"state": "total_funded"} and not issues


def test_parse_rejects_json_list():
    obj, issues = parse_spec_json('[{"state": "total_funded"}]')
    assert obj is None
    assert any(i["code"] == LLM_OUTPUT_NOT_OBJECT for i in issues)


def test_parse_rejects_python_list_object():
    obj, issues = parse_spec_json(["a", "b"])
    assert obj is None
    assert any(i["code"] == LLM_OUTPUT_NOT_OBJECT for i in issues)


def test_parse_rejects_scalar():
    obj, issues = parse_spec_json("42")
    assert obj is None
    assert any(i["code"] == LLM_OUTPUT_NOT_OBJECT for i in issues)


def test_parse_malformed_json():
    obj, issues = parse_spec_json('{"state": "total_funded"')
    assert obj is None
    assert any(i["code"] == LLM_MALFORMED_JSON for i in issues)


def test_parse_detects_code_output():
    raw = "```python\nimport pandas as pd\ndf = pd.read_csv('x')\n```"
    obj, issues = parse_spec_json(raw)
    assert obj is None
    assert any(i["code"] == LLM_OUTPUT_CONTAINS_CODE for i in issues)


def test_parse_empty_and_none():
    for raw in ("", "   ", None):
        obj, issues = parse_spec_json(raw)
        assert obj is None and issues


# --------------------------------------------------------------------------- #
# 3. Clarification object from the model is honoured
# --------------------------------------------------------------------------- #


def test_model_clarification_object():
    raw = json.dumps({"clarification_required": True,
                      "clarification_question": "Which risk view?"})
    result = interpret_from_llm_output("show risk", raw)
    assert result.clarification_required
    assert result.clarification_question == "Which risk view?"
    assert not result.ok
    assert result.normalized_spec is None


# --------------------------------------------------------------------------- #
# 4. Governance — unknown fields, chart types, hallucinations
# --------------------------------------------------------------------------- #


def test_unknown_field_is_warned_but_not_fatal():
    raw = json.dumps({"state": "total_funded", "temporal_mode": "latest",
                      "made_up_field": 123})
    result = interpret_from_llm_output("show total funded", raw)
    assert any(c == LLM_UNKNOWN_FIELD for c in result.issue_codes())
    # Unknown fields are dropped by from_dict, so the spec still validates.
    assert result.ok and not result.clarification_required


def test_unsupported_chart_type_forces_clarification():
    raw = json.dumps({"state": "total_funded", "temporal_mode": "latest",
                      "chart_type": "pie"})
    result = interpret_from_llm_output("show total funded", raw)
    assert any(c == LLM_UNSUPPORTED_CHART_TYPE for c in result.issue_codes())
    assert result.clarification_required and not result.ok


def test_hallucinated_field_forces_clarification_when_semantics_given():
    semantics = {"fields": {"portfolio_id": {}, "internal_risk_grade": {}}}
    raw = json.dumps({"execution_mode": "risk", "state": "total_funded",
                      "risk_monitor_mode": "concentration",
                      "dimension": "wibble_field"})
    result = interpret_from_llm_output("show funded by wibble", raw,
                                       semantics=semantics)
    assert any(c == LLM_HALLUCINATED_FIELD for c in result.issue_codes())
    assert result.clarification_required and not result.ok


def test_known_field_passes_hallucination_check():
    semantics = {"fields": {"portfolio_id": {}}}
    raw = json.dumps({"execution_mode": "risk", "state": "total_funded",
                      "risk_monitor_mode": "concentration",
                      "dimension": "portfolio_id"})
    result = interpret_from_llm_output("show funded by portfolio", raw,
                                       semantics=semantics)
    assert LLM_HALLUCINATED_FIELD not in result.issue_codes()
    assert result.ok


# --------------------------------------------------------------------------- #
# 5. Invalid specs never execute — always clarify
# --------------------------------------------------------------------------- #


def test_invalid_spec_compare_without_dates_clarifies():
    raw = json.dumps({"route_id": "mi", "state": "total_funded",
                      "temporal_mode": "compare"})
    result = interpret_from_llm_output("compare funded", raw)
    assert not result.ok
    assert result.clarification_required
    assert "temporal_selector_incomplete" in result.issue_codes()


def test_invalid_route_for_state_clarifies():
    raw = json.dumps({"route_id": "regulatory_annex2", "state": "total_funded",
                      "temporal_mode": "latest"})
    result = interpret_from_llm_output("show funded on regulatory route", raw)
    assert not result.ok and result.clarification_required
    assert "invalid_route_for_state" in result.issue_codes()


def test_malformed_output_clarifies_never_raises():
    result = interpret_from_llm_output("show total funded", "not json at all {")
    assert result.clarification_required and not result.ok
    assert result.normalized_spec is None


# --------------------------------------------------------------------------- #
# 6. Client boundary — transport errors are captured, not raised
# --------------------------------------------------------------------------- #


def test_client_error_becomes_clarification():
    result = interpret_with_anthropic("show total funded", InterpreterContext(),
                                      BoomClient())
    assert result.clarification_required and not result.ok
    assert any(c == LLM_CLIENT_ERROR for c in result.issue_codes())


def test_adapter_passes_prompt_to_client():
    client = FakeClient(json.dumps({"state": "total_funded",
                                    "temporal_mode": "latest"}))
    interpret_with_anthropic("show total funded", InterpreterContext(), client)
    assert client.prompt is not None
    assert "show total funded" in client.prompt


# --------------------------------------------------------------------------- #
# 7. Prompt builder — encodes the hard constraints
# --------------------------------------------------------------------------- #


def test_prompt_contains_constraints_and_context():
    ctx = InterpreterContext()
    prompt = build_mi_spec_prompt("show total funded", ctx)
    low = prompt.lower()
    assert "json only" in low
    assert "do not compute" in low or "do not calculate" in low
    assert "clarification_required" in prompt
    # Context anchors are embedded for date resolution.
    assert ctx.as_of in prompt
    # Allowed enums present.
    assert "total_funded" in prompt
    assert "concentration" in prompt
    # The question is included.
    assert "show total funded" in prompt


def test_prompt_lists_semantic_fields_when_provided():
    semantics = {"fields": {"portfolio_id": {}, "geographic_region_obligor": {}}}
    prompt = build_mi_spec_prompt("show by portfolio", InterpreterContext(),
                                  semantics=semantics)
    assert "portfolio_id" in prompt
    assert "geographic_region_obligor" in prompt


# --------------------------------------------------------------------------- #
# 8. Guards — no network/SDK at import, no forbidden imports
# --------------------------------------------------------------------------- #


def test_no_sdk_import_at_module_load():
    import sys
    # Importing the adapter must not pull in the anthropic SDK.
    import mi_agent.interpreter.anthropic  # noqa: F401
    assert "anthropic" not in [m for m in sys.modules
                               if m == "anthropic"], "anthropic SDK imported eagerly"


def test_no_forbidden_eager_imports_in_adapter():
    text = (REPO_ROOT / "mi_agent" / "interpreter" / "anthropic.py").read_text(
        encoding="utf-8")
    # The literal SDK import token must not appear (lazy importlib only).
    assert "import anthropic" not in text
    for token in ("import openai", "import streamlit", "import azure",
                  "from azure", "from analytics ", "from analytics."):
        assert token not in text


def test_safe_by_construction_invalid_never_ok():
    # A spec that fails validation must never report ok / must clarify.
    raw = json.dumps({"route_id": "mi", "risk_monitor_mode": "concentration",
                      "state": "total_pipeline", "dimension": "stage"})
    result = interpret_from_llm_output("show pipeline by stage-ish", raw)
    assert not result.ok
    assert result.clarification_required
