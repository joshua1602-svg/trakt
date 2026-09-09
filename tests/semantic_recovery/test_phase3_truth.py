"""PHASE 3/5 — focused truth tests for the remediations, on the author's own
wording (never a frozen-bank question).

P0-E  the weighting qualifier is the statistic owner's, never a measure;
P1-B  a coordinated measure list executes every slot or refuses by name;
I5    the ledger refuses a pipeline-status or lending-window claim the
      point-in-time path cannot carry, and lets a level stand.
"""
from __future__ import annotations

import pytest


class TestP0EWeightingQualifier:
    @pytest.mark.parametrize("question", [
        "What is the balance-weighted average LTV?",
        "On a balance weighted basis, what is the book's LTV?",
        "What is the LTV weighted by balance?",
    ])
    def test_the_qualifier_names_the_weight_not_a_measure(self, parse, question):
        spec = parse(question).spec
        assert spec.metric == "current_loan_to_value"
        assert spec.aggregation == "weighted_avg"
        assert spec.weight_field == "current_outstanding_balance"
        assert not spec.measures, "no second measure was read out of the qualifier"

    def test_an_exposure_weighting_is_the_balance_weight(self, parse, semantics):
        spec = parse("What is the exposure-weighted borrower age by region?").spec
        assert spec.metric == "youngest_borrower_age"
        assert spec.aggregation == "weighted_avg"
        assert spec.weight_field == "current_outstanding_balance"
        assert spec.dimension == "collateral_geography"

    def test_the_geo_exposure_route_does_not_read_the_qualifier(self):
        from mi_agent_api import chat_routing as CR
        assert CR._is_geo_exposure("What is the exposure-weighted borrower age by region?") is False
        assert CR._is_geo_exposure("Show exposure by region") is True

    def test_the_statistic_owner_reads_any_weighting(self):
        from mi_agent import statistic as S
        assert S.statistic_named("value-weighted ltv") == "weighted_avg"
        assert S.weight_word_named("weighted by exposure") == "exposure"
        assert S.weight_word_named("the un-weighted average") is None


class TestP1BMultiMeasure:
    def test_a_parallel_interrogative_is_a_second_slot(self, semantics, columns):
        from mi_agent import llm_query_parser as P
        assert P.unresolved_measure_slots(
            "what ltv and what rate is the book running at?", semantics, columns) == ("rate",)

    def test_a_genuine_second_clause_is_not_a_slot(self, semantics, columns):
        from mi_agent import llm_query_parser as P
        assert P.unresolved_measure_slots(
            "the largest exposure and what share of the book is it", semantics, columns) == ()

    def test_two_resolved_measures_both_execute(self, parse):
        spec = parse("Weighted average loan-to-value and interest rate, please.").spec
        fields = [m["field"] for m in (spec.measures or [])]
        assert fields == ["current_loan_to_value", "current_interest_rate"]


class TestI5LedgerOnThePointInTimePath:
    def _guard(self, interpret, question, spec_overrides=None):
        from mi_agent import semantic_claims as SC
        from mi_agent_api import mi_service as S
        parsed, qi = interpret(question)
        claims = SC.build(question, interpretation=qi, spec=parsed.spec,
                          parse_meta=parsed.meta)
        result = {"ok": True, "answer": "36 loans", "spec": parsed.spec.to_dict(),
                  "metadata": {}, "artifacts": []}
        return S._fail_closed_analytical(result, question=question, view="funded",
                                         claims=claims), claims

    def test_a_pipeline_status_on_the_funded_book_refuses(self, interpret):
        out, claims = self._guard(interpret, "How many pending loans do we have?")
        assert "dataset:pipeline" in claims.requirements
        assert out["ok"] is False and out["controlledRefusal"] is True

    def test_a_lending_window_nobody_applied_refuses(self, interpret):
        out, claims = self._guard(interpret, "What is the balance of new loans this month?")
        assert any(r.startswith("population:") for r in claims.requirements)
        assert out["ok"] is False and "new lending" in out["answer"]

    def test_a_level_on_the_live_book_stands(self, interpret):
        out, claims = self._guard(interpret, "How many live loans do we have?")
        assert claims.requirements == ()
        assert out["ok"] is True

    def test_an_earlier_period_level_refuses(self, interpret):
        out, claims = self._guard(interpret, "What was the funded balance in the prior period?")
        assert "period_as_at" in claims.requirements
        assert out["ok"] is False and "prior period" in out["answer"]
