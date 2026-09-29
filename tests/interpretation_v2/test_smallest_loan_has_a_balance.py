"""The smallest loan has a balance (owner decision D14, 2026-09-29).

"What is the smallest loan?" answered £0 on the 2026-09-29 full bank: a
redeemed loan's balance is zeroed on purpose (`closed_account.zero_fields`), so
the minimum over the whole funded book is nought — true, and not the smallest
loan. The owner's rule: the minimum balance counts only balances above zero.

The rule is the REGISTRY's (`statistic_scope` on current_outstanding_balance,
from the build script's curation), carried onto the governed concept, and
written onto the plan by the compiler as a predicate on the figure's own
output — so the executor applies it like any filter, the receipt discloses it
and the coverage ledger proves it. Nothing downstream knows the rule exists.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.interpretation_v2.outcomes import UNSUPPORTED_COMPOSITION
from mi_agent.interpretation_v2.vocabulary import load_governed_vocabulary
from tests.interpretation_v2.test_funded_breadth import (  # noqa: F401
    _intent, _served, book, semantics)

_MIN = [{"concept": "current_outstanding_balance", "statistic": "min"}]
_SCOPE = {"concept": "current_outstanding_balance", "comparator": "gt",
          "canonical_field": "current_outstanding_balance", "value": 0,
          "capability_owner": None}


def _compile(**over):
    return DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(_intent(operation="point_in_time", **over)))


def test_the_rule_is_the_registrys():
    concept = load_governed_vocabulary().resolve("current_outstanding_balance")
    assert concept.scope_for("min") == ("gt", 0)
    assert concept.scope_for("max") is None and concept.scope_for("sum") is None
    # The model is told, so it does not add the predicate itself.
    assert "min" in concept.metadata_view()["statistic_scope"]


def test_the_minimum_carries_the_predicate_on_its_own_figure():
    result = _compile(measures=_MIN)
    plan = result.plan.to_dict()
    assert list(plan["outputs"][0]["filters"]) == [_SCOPE]
    assert list(plan["filters"]) == []
    assert any("owner decision D14" in note
               for note in plan["provenance"]["notes"])


@pytest.mark.parametrize("statistic", ["max", "sum", "median", "average"])
def test_no_other_statistic_is_scoped(statistic):
    plan = _compile(measures=[{"concept": "current_outstanding_balance",
                               "statistic": statistic}]).plan.to_dict()
    assert list(plan["outputs"][0]["filters"]) == []


def test_a_question_that_already_restricts_the_balance_stands_on_its_own():
    """"The smallest loan over £50k" already keeps only balances above zero."""
    result = _compile(measures=_MIN, filters=[
        {"concept": "current_outstanding_balance", "comparator": "gt",
         "value": 50000}])
    plan = result.plan.to_dict()
    assert list(plan["outputs"][0]["filters"]) == []
    assert [f["value"] for f in plan["filters"]] == [50000]


def test_the_minimum_beside_another_figure_is_refused_not_answered_wrongly():
    """In one output the predicate would change the loan count too."""
    result = _compile(measures=_MIN + [{"concept": "current_outstanding_balance",
                                        "statistic": "count"}])
    assert result.plan is None
    assert UNSUPPORTED_COMPOSITION in result.codes()


def test_the_smallest_loan_is_the_smallest_balance_above_zero(monkeypatch, book,
                                                              semantics):
    frame = book.copy()
    frame.loc[frame.index[0], "current_outstanding_balance"] = 0.0  # redeemed
    balances = pd.to_numeric(frame["current_outstanding_balance"])
    assert balances.min() == 0.0, "premise: the book holds a zeroed loan"

    payload, record, coverage = _served(_intent(operation="point_in_time",
                                                measures=_MIN),
                                        monkeypatch, frame, semantics)
    assert payload is not None, record.get("execution")
    assert record["execution"]["value"] == pytest.approx(
        float(balances[balances > 0].min()))
    assert record["execution"]["value"] > 0
    assert coverage["unaccounted"] == []
    # The answer says which loans it counted.
    assert "Balance > 0" in payload["answer"], payload["answer"]
    assert f"{int((balances > 0).sum()):,} loans" in payload["answer"]
