"""Owner decision D18 (2026-09-30): "Do not use the old system."

A question the governed path does not answer is declined in words, and never
handed to the legacy path. These tests pin what the decline says: what the
question was understood as, why it is not answered, what kind of decline it is
for the operator, and that it never carries a figure.
"""
from __future__ import annotations

import importlib
import re

import pytest

from mi_agent import plan_decline as decline
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent

_GENERIC = "it is outside what I can answer yet"

#: The modules whose reason codes reach a reader through `respond`.
_RUNTIMES = ("mi_agent.plan_runtime_adapter", "mi_agent.plan_pipeline_runtime",
             "mi_agent.plan_forecast_runtime",
             "mi_agent.plan_stage_movement_runtime",
             "mi_agent.plan_temporal_runtime", "mi_agent.plan_material_summary",
             "mi_agent.plan_metric_delta", "mi_agent.plan_attribution",
             "mi_agent.semantic_engine")

#: Codes those modules declare that are NOT decline reasons: the shadow
#: comparison's verdicts, and "no plan at all", which never reaches a runtime
#: through `respond` (a question with no plan is a compiler REFUSE/CLARIFY).
_NOT_DECLINES = frozenset({
    "NOT_A_PLAN", "NOT_ELIGIBLE", "DISPOSITION_DIFFERENCE",
    "EXACT_SEMANTIC_PARITY", "NEW_PATH_SEMANTIC_DIFFERENCE",
    "NUMERICAL_DIFFERENCE", "OLD_PATH_SEMANTIC_DIFFERENCE",
    "PRESENTATION_ONLY", "SHADOW_EXECUTION_ERROR"})


def _declared_codes():
    for name in _RUNTIMES:
        module = importlib.import_module(name)
        for key, value in vars(module).items():
            if (key.isupper() and isinstance(value, str) and value == key
                    and re.fullmatch(r"[A-Z][A-Z_]+", value)):
                yield value


def _overdue_two_figures():
    """hv2_pipeline_013_1's reading: the count AND the amount of the overdue
    pipeline — the question the legacy path answered with the whole pipeline."""
    body = {"schema_version": "candidate_intent/1.0", "capability": "pipeline",
            "operation": "summary", "population": {"base": "pipeline"},
            "measures": [{"concept": "pipeline_case_count"},
                         {"concept": "pipeline_amount"}],
            "filters": [{"concept": "expected_completion_timing",
                         "comparator": "eq", "value": "overdue"}],
            "time": {"form": "current"}}
    plan = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(body)).plan
    assert plan is not None
    return plan.to_dict()


def test_every_reason_a_runtime_declares_has_a_sentence():
    """A reason a runtime adds tomorrow is worded by its family; one that fits
    no family fails here, before a reader is told "outside what I can answer"."""
    unworded = sorted({code for code in _declared_codes()
                       if code not in _NOT_DECLINES
                       and decline.kind(code) == decline.UNSUPPORTED
                       and decline.plain_reason(code) == _GENERIC})
    assert unworded == []


def test_the_decline_says_what_was_understood_and_why():
    plan = _overdue_two_figures()
    text = decline.message({"compiler": {"plan": plan}},
                           "INELIGIBLE:MEASURE_NOT_SUPPORTED")
    assert text.startswith("I understood this as ")
    assert "overdue" in text
    assert "for the pipeline" in text
    assert "combination of figures" in text
    assert text.endswith("Nothing was guessed, and no other figure was put in "
                         "its place.")


def test_the_decline_carries_no_figure():
    plan = _overdue_two_figures()
    for reason in ("INELIGIBLE:MEASURE_NOT_SUPPORTED", "EXECUTION_FAILED",
                   "INELIGIBLE:HISTORY_UNAVAILABLE", "INTERPRETER_FAILURE",
                   "CLARIFY_NOT_SERVED_IN_THIS_SLICE",
                   "REFUSE_NOT_SERVED_IN_THIS_SLICE"):
        text = decline.message({"compiler": {"plan": plan}}, reason)
        assert "£" not in text and "%" not in text, reason
        assert not re.search(r"\d", text), (reason, text)


@pytest.mark.parametrize("reason, kind", [
    ("INTERPRETER_FAILURE", decline.MODEL_UNAVAILABLE),
    ("CLARIFY_NOT_SERVED_IN_THIS_SLICE", decline.CLARIFY),
    ("REFUSE_NOT_SERVED_IN_THIS_SLICE", decline.UNSUPPORTED),
    ("INELIGIBLE:MEASURE_NOT_SUPPORTED", decline.UNSUPPORTED),
    ("INELIGIBLE:SCALE_NOT_CONFIGURED", decline.UNSUPPORTED),
    ("TEMPORAL_NOT_RESOLVED:PERIOD_NOT_AVAILABLE", decline.UNSUPPORTED),
    ("INELIGIBLE:HISTORY_UNAVAILABLE", decline.UNAVAILABLE),
    ("TEMPORAL_STORE_UNAVAILABLE", decline.UNAVAILABLE),
    ("EXECUTION_FAILED", decline.FAILED),
    ("RENDER_FAILED", decline.FAILED),
    ("UNEXPECTED_ERROR", decline.FAILED),
    ("PLAN_RECEIPT_RECONCILIATION_FAILED:predicate lost", decline.FAILED),
])
def test_each_reason_is_the_right_kind_of_decline(reason, kind):
    assert decline.kind(reason) == kind


def test_a_clarification_asks_the_models_own_question():
    body = {"interpretation": {"ambiguities": [
        {"blocking": False, "note": "not this one"},
        {"blocking": True, "note": "which month do you mean?"}]}}
    assert decline.message(body, "CLARIFY_NOT_SERVED_IN_THIS_SLICE") == (
        "I need one more detail before I can answer: which month do you mean? "
        "Nothing was guessed, and no other figure was put in its place.")


def test_a_compiler_refusal_names_its_reason():
    body = {"compiler": {"reasons": [{"code": "UNREGISTERED_CONCEPT"}]}}
    text = decline.message(body, "REFUSE_NOT_SERVED_IN_THIS_SLICE")
    assert "no governed definition" in text


def test_the_envelope_is_the_refusal_shape_every_channel_renders():
    plan = _overdue_two_figures()
    body = {"compiler": {"plan": plan, "plan_id": plan["plan_id"]}}
    env = decline.envelope(question="q", body=body,
                           reason="INELIGIBLE:MEASURE_NOT_SUPPORTED", view="pipeline")
    assert env["ok"] is False
    assert env["answer"] == env["error"] == env["validation"]["errors"][0]
    assert env["artifacts"] == []
    meta = env["metadata"]
    assert meta["parserMode"] == decline.DECLINED_MODE
    assert meta["datasetContext"] == "pipeline"
    assert meta["governedDecline"] == {
        "reason": "INELIGIBLE:MEASURE_NOT_SUPPORTED",
        "kind": decline.UNSUPPORTED,
        "understood": decline.understood(plan), "plan_id": plan["plan_id"]}


def test_a_comparator_is_read_as_words():
    plan = {"population": {"base": "funded"},
            "outputs": [{"measures": [{"concept": "loan"}], "dimensions": [],
                         "filters": [{"concept": "borrower_age",
                                      "comparator": "gt", "value": 55}]}]}
    assert "where borrower age is above 55" in decline.understood(plan)


def test_a_region_is_read_with_its_level_and_basis():
    """The askback run (2026-10-02): "With your reply, I read your earlier
    question as balance, for the funded book" — the region the reply supplied
    was not in the reading. It is part of the plan, and is said."""
    plan = {"population": {"base": "funded"},
            "outputs": [{"measures": [{"concept": "balance"}], "dimensions": [],
                         "filters": []}],
            "geography": {"canonical_field": "geographic_region_collateral",
                          "resolved_level": "nuts3", "group_by": True}}
    assert decline.understood(plan).endswith(
        "by NUTS3 region (the property's location)")
    plan["geography"] = {"canonical_field": "geographic_region_obligor_itl3",
                         "resolved_level": "itl3", "group_by": False,
                         "values": ["Kent"]}
    assert decline.understood(plan).endswith(
        "where the ITL3 sub-region (the borrower's address) is Kent")
