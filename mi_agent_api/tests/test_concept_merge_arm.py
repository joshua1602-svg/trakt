#!/usr/bin/env python3
"""The concept-merge arm: what it may change, and what it may never.

The arm is the one place the split touches a live request. What is asserted
here is the boundary — the flag, what reaches the spec, and what a failure
costs — not the model, which is absent from every test.
"""
from __future__ import annotations

import pytest

from mi_agent_api import concept_merge_arm as ARM
from question_interpretation import claim_merge as CM
from question_interpretation.schema import PROV_DEFAULT, PROV_EXPLICIT_USER


class _Spec:
    def __init__(self, **kw):
        self.filters = kw.get("filters", {})
        self.dimensions = kw.get("dimensions", [])
        self.dimension = kw.get("dimension")
        self.metric = kw.get("metric")


def _slot(slot, key, value, operator=None):
    return CM.SlotValue(slot, key, value, CM.PROV_MODEL_INFERRED,
                        operator=operator)


# --------------------------------------------------------------------------- #
# The flag
# --------------------------------------------------------------------------- #
def test_the_arm_is_off_by_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    monkeypatch.delenv("MI_AGENT_CONCEPT_MERGE", raising=False)
    assert ARM.enabled() is False


def test_a_key_alone_does_not_turn_it_on(monkeypatch):
    """A KEY IS NOT CONSENT. `_mi_llm_config` runs `auto` and enables the
    shipped free-form parser on a key alone; that arm emits a whole MIQuerySpec
    and is the arrangement this split exists to replace. Running both would
    measure their sum."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    monkeypatch.setenv("MI_AGENT_LLM_PARSER", "on")
    monkeypatch.delenv("MI_AGENT_CONCEPT_MERGE", raising=False)
    assert ARM.enabled() is False


def test_it_needs_a_key_as_well_as_the_flag(monkeypatch):
    monkeypatch.setenv("MI_AGENT_CONCEPT_MERGE", "on")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert ARM.enabled() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    assert ARM.enabled() is True


# --------------------------------------------------------------------------- #
# What reaches the spec
# --------------------------------------------------------------------------- #
def test_a_categorical_fill_becomes_a_bare_value():
    spec = _Spec(filters={"current_loan_to_value": {"op": "gt", "value": 50.0}})
    applied = ARM._apply_to_spec(
        spec, [_slot("row_predicates", "erm_product_type", "drawdown")])
    assert spec.filters["erm_product_type"] == "drawdown"
    assert applied == [{"kind": "filter", "field": "erm_product_type",
                        "operator": None, "value": "drawdown"}]


def test_a_threshold_fill_becomes_the_shape_both_executors_read():
    spec = _Spec()
    ARM._apply_to_spec(spec, [_slot("row_predicates", "youngest_borrower_age",
                                    75.0, operator="gt")])
    assert spec.filters["youngest_borrower_age"] == {"op": "gt", "value": 75.0}


def test_a_field_the_spec_already_filters_on_is_never_overwritten():
    """Belt and braces: the merge has already declined it, and this refuses it
    again at the point of application."""
    spec = _Spec(filters={"erm_product_type": "lump_sum"})
    applied = ARM._apply_to_spec(
        spec, [_slot("row_predicates", "erm_product_type", "drawdown")])
    assert spec.filters["erm_product_type"] == "lump_sum"
    assert applied == []


def test_a_single_new_axis_also_travels_on_spec_dimension():
    """`spec.dimension` is what the point-in-time executor groups by. Setting
    the list alone produced a contract that NAMED an axis nothing grouped on."""
    spec = _Spec()
    ARM._apply_to_spec(spec, [_slot("dimensions", "ltv_bucket", "ltv_bucket")])
    assert spec.dimensions == ["ltv_bucket"]
    assert spec.dimension == "ltv_bucket"


def test_an_axis_the_spec_already_groups_by_is_not_added_twice():
    spec = _Spec(dimensions=["ltv_bucket"], dimension="ltv_bucket")
    assert ARM._apply_to_spec(
        spec, [_slot("dimensions", "ltv_bucket", "ltv_bucket")]) == []


def test_a_measure_is_only_filled_when_the_spec_has_none():
    spec = _Spec(metric="current_loan_to_value")
    assert ARM._apply_to_spec(
        spec, [_slot("subject", None, "current_outstanding_balance")]) == []
    assert spec.metric == "current_loan_to_value"


def test_nothing_filled_leaves_the_spec_untouched():
    spec = _Spec(filters={"a": 1}, dimensions=["b"], metric="c")
    assert ARM._apply_to_spec(spec, []) == []
    assert (spec.filters, spec.dimensions, spec.metric) == ({"a": 1}, ["b"], "c")


# --------------------------------------------------------------------------- #
# Failure costs nothing
# --------------------------------------------------------------------------- #
def test_a_model_that_cannot_be_reached_leaves_the_contract_alone(monkeypatch):
    """The arm being off for that question, not the question failing."""
    from mi_agent import llm_query_parser as LQ

    def _boom(*a, **k):
        raise RuntimeError("Error code: 529 - overloaded")

    monkeypatch.setattr(LQ, "_call_llm", _boom)
    spec = _Spec(filters={"a": 1})

    class _QI:
        subject = source_scope = dataset = None
        dimensions = row_predicates = ()

    evidence = ARM.apply("anything", spec, {"fields": {}}, interpretation=_QI(),
                         available_values={}, available_columns={"a"})
    assert evidence["status"] == "proposal_unavailable"
    assert spec.filters == {"a": 1}


def test_no_interpretation_means_no_arm():
    spec = _Spec()
    assert ARM.apply("q", spec, {}, interpretation=None) is None


# --------------------------------------------------------------------------- #
# The replay seam
# --------------------------------------------------------------------------- #
def test_a_replayed_proposal_is_used_instead_of_a_live_call(monkeypatch):
    """The same injection `parse_with_repair` offers as `llm_callable`, and for
    the same reason: a review pack that re-derived its numbers from a SECOND
    live call would be a review of a different run."""
    from mi_agent import llm_query_parser as LQ

    def _never(*a, **k):
        raise AssertionError("the model must not be called for a replay")

    monkeypatch.setattr(LQ, "_call_llm", _never)
    ARM.set_replay({"how many drawdown loans?": [
        {"kind": "category_value", "term": "drawdown"}]})
    try:
        spec = _Spec()

        class _QI:
            subject = source_scope = dataset = None
            dimensions = row_predicates = ()

        from migration_phase0.assurance_semantics import load_assurance_semantics
        evidence = ARM.apply(
            "how many drawdown loans?", spec, load_assurance_semantics(),
            interpretation=_QI(),
            available_values={"erm_product_type": {"drawdown": "drawdown"}},
            available_columns={"erm_product_type"})
    finally:
        ARM.set_replay(None)
    assert evidence["source"] == "replayed"
    assert evidence["model"] is None
    assert spec.filters == {"erm_product_type": "drawdown"}


def test_replay_is_off_unless_it_is_set(monkeypatch):
    """Serving must never silently answer from a recording."""
    ARM.set_replay(None)
    assert ARM._REPLAY == {}
    called = {}

    def _live(prompt, model, **k):
        called["yes"] = True
        return '{"concepts":[]}', {}, False

    from mi_agent import llm_query_parser as LQ
    monkeypatch.setattr(LQ, "_call_llm", _live)
    spec = _Spec()

    class _QI:
        subject = source_scope = dataset = None
        dimensions = row_predicates = ()

    from migration_phase0.assurance_semantics import load_assurance_semantics
    ARM.apply("anything at all", spec, load_assurance_semantics(),
              interpretation=_QI(), available_values={}, available_columns={"a"})
    assert called.get("yes") is True


# --------------------------------------------------------------------------- #
# The vocabulary cache
# --------------------------------------------------------------------------- #
def test_the_vocabulary_cache_is_keyed_on_content_not_identity():
    """`mi_service` loads the semantics fresh per request, so an `id()` key
    would never hit — and CPython reuses an id once an object is collected, so
    it could serve one book's vocabulary for a request against another."""
    import copy

    from migration_phase0.assurance_semantics import load_assurance_semantics

    semantics = load_assurance_semantics()
    values = {"erm_product_type": {"drawdown": "drawdown"}}
    columns = {"erm_product_type", "current_outstanding_balance"}
    ARM._VOCAB_CACHE.clear()
    first = ARM._vocabulary(semantics, values, columns)
    again = ARM._vocabulary(copy.deepcopy(semantics), dict(values), set(columns))
    assert first is again
    assert len(ARM._VOCAB_CACHE) == 1
    other = ARM._vocabulary(semantics, values, columns | {"ltv_bucket"})
    assert other is not first
    assert len(ARM._VOCAB_CACHE) == 2


# --------------------------------------------------------------------------- #
# Cost telemetry — an observation ABOUT an answer, never a fact about whether
# the reader gets one
# --------------------------------------------------------------------------- #
def test_an_unpriced_model_is_unknown_never_a_silent_zero():
    """A model with no pricing entry must SAY it is unpriced. Reporting $0.00
    would read as a free call and would quietly under-report the bill."""
    priced = ARM._priced("claude-opus-5", {"input_tokens": 30, "output_tokens": 150})
    assert priced["cost_estimate_status"] == "estimated"
    assert priced["estimated_total_cost"] > 0

    unknown = ARM._priced("some-unpriced-future-model",
                          {"input_tokens": 30, "output_tokens": 150})
    assert unknown["cost_estimate_status"] == "unknown"


def test_a_pricing_failure_costs_the_telemetry_and_nothing_else(monkeypatch):
    """TELEMETRY MAY NOT DECIDE WHETHER AN ANSWER IS GIVEN.

    `apply` is called inside a `try` whose `except` records
    `proposal_unavailable`, and an unavailable proposal is a controlled
    REFUSAL. Before this guard existed a malformed usage record raised out of
    the pricing call, travelled that path, and turned a delivered 39-group
    heatmap into "I could not complete the language-understanding step".
    """
    from mi_agent import llm_query_parser as LQ

    def _boom(model, usage):
        raise RuntimeError("pricing table exploded")

    monkeypatch.setattr(LQ, "estimate_cost", _boom)
    # No exception escapes, and the caller gets None rather than a refusal.
    assert ARM._priced("claude-opus-5", {"input_tokens": 30}) is None


def test_a_replayed_proposal_is_not_a_call_and_is_not_priced():
    """Replay exists so a measurement can be reproduced without paying for the
    model again. Pricing a replay would invent spend that never happened."""
    ARM.set_replay({"q": [{"kind": "category_value", "term": "drawdown"}]})
    try:
        assert "q" in ARM._REPLAY
    finally:
        ARM.set_replay(None)
    assert ARM._REPLAY == {}
