"""PHASE 4 — MODEL-LAYER PROOF, with REPLAYED proposals (no network).

Five properties of the concept-merge arm, each proved by replaying a recorded
proposal into the real arm against the real contract:

  P1  fill-only: an EMPTY slot is filled, and the fill is disclosed as
      model-inferred with the measure's own aggregation;
  P2  never overwrite: a slot the reader filled is never replaced, and the
      disagreement is reported;
  P3  registered concepts only: an unregistered term is rejected — no nearest
      match, nothing applied;
  P4  no ambiguity resolved by preference: a value two governed fields carry
      is rejected as ambiguous, nothing applied;
  P5  model unavailable is a governed refusal, never a silent downgrade: the
      arm reports PROPOSAL_UNAVAILABLE and the service refuses on it;
  P6  determinism: the same replay yields the same contract, twice.
"""
from __future__ import annotations

import copy

import pytest


@pytest.fixture
def arm(monkeypatch):
    from mi_agent_api import concept_merge_arm as arm

    monkeypatch.setenv("MI_AGENT_CONCEPT_MERGE", "on")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    yield arm
    arm.set_replay(None)


def _apply(arm, interpret, semantics, book_values, columns, question, proposals):
    arm.set_replay({question: proposals})
    parsed, qi = interpret(question)
    evidence = arm.apply(question, parsed.spec, semantics, interpretation=qi,
                         available_values=book_values, available_columns=columns)
    return parsed, evidence


class TestP1FillOnly:
    def test_an_empty_subject_is_filled_and_disclosed(self, arm, interpret, semantics,
                                                      book_values, columns):
        from question_interpretation.schema import PROV_MODEL_INFERRED
        parsed, ev = _apply(arm, interpret, semantics, book_values, columns,
                            "How big is the book by region?",
                            [{"kind": "measure", "term": "balance", "covers": "big"}])
        assert ev["status"] == "applied"
        assert parsed.spec.metric == "current_outstanding_balance"
        assert parsed.spec.aggregation == "sum"
        assert parsed.spec.metric_source == PROV_MODEL_INFERRED
        assert [a["kind"] for a in ev["applied"]] == ["measure"]


class TestP2NeverOverwrite:
    def test_a_reader_filled_subject_stands(self, arm, interpret, semantics,
                                            book_values, columns):
        parsed, ev = _apply(arm, interpret, semantics, book_values, columns,
                            "What is the total funded balance by region?",
                            [{"kind": "measure", "term": "ltv", "covers": "balance"}])
        assert ev["status"] == "no_change" and ev["applied"] == []
        assert parsed.spec.metric == "current_outstanding_balance"
        assert any(f["is_conflict"] for f in ev["findings"])

    def test_an_asked_count_stands(self, arm, interpret, semantics, book_values, columns):
        parsed, ev = _apply(arm, interpret, semantics, book_values, columns,
                            "How many loans do we have by region?",
                            [{"kind": "measure", "term": "balance", "covers": "loans"}])
        assert ev["applied"] == []
        assert parsed.spec.aggregation == "count" and parsed.spec.metric is None


class TestP3RegisteredOnly:
    def test_an_unregistered_term_is_rejected_not_matched(self, arm, interpret, semantics,
                                                          book_values, columns):
        from question_interpretation import concept_proposal as CP
        parsed, ev = _apply(arm, interpret, semantics, book_values, columns,
                            "How big is the book by region?",
                            [{"kind": "measure", "term": "book size", "covers": "big"}])
        assert ev["applied"] == []
        assert ev["rejected"] and ev["rejected"][0]["rejected"] == CP.REJECT_UNREGISTERED
        assert parsed.spec.metric is None


class TestP4NoAmbiguityByPreference:
    def test_a_value_two_fields_carry_is_rejected(self, arm, interpret, semantics,
                                                  book_values, columns):
        """A value the book carries under two governed fields with different
        value domains is ambiguous; the arm must not pick one."""
        from question_interpretation import concept_proposal as CP
        from mi_agent.categorical_spans import preferred_field
        ambiguous = None
        seen = {}
        for field, values in book_values.items():
            for v in (values.keys() if hasattr(values, "keys") else values):
                seen.setdefault(str(v).strip().lower(), set()).add(field)
        for v, fields in seen.items():
            if len(fields) > 1 and preferred_field(sorted(fields), semantics) is None:
                ambiguous = v
                break
        if ambiguous is None:
            pytest.skip("this book carries no ambiguous categorical value")
        parsed, ev = _apply(arm, interpret, semantics, book_values, columns,
                            "What is the funded balance?",
                            [{"kind": "category_value", "term": ambiguous, "covers": ambiguous}])
        assert ev["applied"] == []
        assert ev["rejected"][0]["rejected"] == CP.REJECT_AMBIGUOUS
        assert not parsed.spec.filters


class TestP5UnavailableIsARefusal:
    def test_the_arm_reports_unavailable_and_the_service_refuses(
            self, arm, interpret, semantics, book_values, columns, monkeypatch):
        from mi_agent import llm_query_parser as LQ
        from mi_agent_api import mi_service as S

        def _down(*a, **k):
            raise RuntimeError("model unreachable")
        monkeypatch.setattr(LQ, "_call_llm", _down)
        arm.set_replay(None)
        q = "How big is the book by region?"
        parsed, qi = interpret(q)
        ev = arm.apply(q, parsed.spec, semantics, interpretation=qi,
                       available_values=book_values, available_columns=columns)
        assert ev["status"] == arm.PROPOSAL_UNAVAILABLE
        assert parsed.spec.metric is None, "nothing was substituted"
        envelope = {"ok": True, "answer": "36 loans", "metadata": {"conceptMerge": ev}}
        out = S._enforce_model_availability(envelope)
        assert out["ok"] is False
        assert out["metadata"].get("modelUnavailableRefused") or out.get("error")


class TestP6Determinism:
    def test_the_same_replay_gives_the_same_contract(self, arm, interpret, semantics,
                                                     book_values, columns):
        q = "How big is the book by region?"
        props = [{"kind": "measure", "term": "balance", "covers": "big"}]
        a, ev_a = _apply(arm, interpret, semantics, book_values, columns, q, props)
        b, ev_b = _apply(arm, interpret, semantics, book_values, columns, q, props)
        assert a.spec.to_dict() == b.spec.to_dict()
        strip = lambda e: {k: v for k, v in e.items() if k not in ("usage", "cost")}
        assert strip(ev_a) == strip(ev_b)
