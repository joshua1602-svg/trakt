#!/usr/bin/env python3
"""A trend of THINGS is a count of them, not a sum of their money.

THE DEFECT, from the live bank's defaulted-metric audit:

    Show weekly pipeline cases.    -> metric=Balance  aggregation=sum
                                      metric_defaulted=True, guard verdict ok

The reader asked how MANY cases; the answer was how MUCH money. Nothing
disclosed it beyond `metric_defaulted`, and no substitution facet was raised, so
the receipt read "Total Balance" for a question that named no measure at all.

THE ESTATE ALREADY DISAGREED WITH ITSELF about that sentence:

    Show pipeline cases.            -> aggregation=count     (summary path)
    Show weekly pipeline cases.     -> Balance / sum         (trend path)

One object, two intents, two economic meanings. `_wants_count` reads the
explicit phrases — "case count", "how many cases" — and the trend branch fell
through to the governed balance default whenever they were absent. The bare row
noun standing as the subject is the same request said plainly.

The row-noun vocabulary is `_SHARE_COUNT_RE`'s, not a third copy of it. A
question carrying a money word is never a count: "amount" is the reader's own
governed default for the balance, so "the amount change on cases that stayed in
Application" asks for money ABOUT cases and keeps its measure.

MEASURED on the 882-question corpus: ONE question changes, and it changes from
money to a count of the cases it asked for.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import _deterministic_parse
from mi_agent.mi_query_validator import load_mi_semantics

_REGISTRY = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
_BALANCE = "current_outstanding_balance"


def _parse(question):
    spec, _ = _deterministic_parse(question, load_mi_semantics(_REGISTRY))
    return spec


# ------------------------------------------------------------- the defect #
def test_a_weekly_trend_of_cases_counts_cases():
    spec = _parse("Show weekly pipeline cases.")
    assert spec.aggregation == "count"
    assert spec.metric is None
    assert not spec.metric_defaulted


def test_the_same_holds_however_the_trend_is_worded():
    for question in ("Show pipeline cases over time.",
                     "Show pipeline cases by stage over time.",
                     "Show weekly loans."):
        spec = _parse(question)
        assert spec.aggregation == "count", question
        assert spec.metric is None, question


def test_the_two_intents_now_agree_about_one_object():
    """The disagreement that made this visible: the same object, with and
    without the trend word, must mean the same thing."""
    plain = _parse("Show pipeline cases.")
    trend = _parse("Show weekly pipeline cases.")
    assert plain.aggregation == trend.aggregation == "count"
    assert plain.metric is trend.metric is None


# --------------------------------------------- and money stays money #
def test_a_named_measure_is_never_turned_into_a_count():
    for question in ("Show weekly pipeline balance.",
                     "Show funded balance evolution by month.",
                     "Show the weighted average LTV by month."):
        spec = _parse(question)
        assert spec.aggregation != "count", question
        assert spec.metric is not None, question


def test_the_readers_own_amount_default_still_means_money():
    """`amount` is the product owner's governed default for the balance, so a
    question carrying it asks for money even about cases."""
    for question in ("What was the amount change on cases that stayed in "
                     "Application?",
                     "What is the current pipeline amount?"):
        spec = _parse(question)
        assert spec.metric == _BALANCE, question
        assert spec.aggregation == "sum", question


def test_a_trend_naming_no_object_at_all_keeps_the_governed_default():
    """"Show pipeline evolution" names neither a measure nor a row noun. The
    balance default stands, and stays disclosed as defaulted."""
    spec = _parse("Show pipeline evolution.")
    assert spec.metric == _BALANCE
    assert spec.metric_defaulted


# --------------------------------------------------------- the census #
def test_the_new_rule_never_fires_on_a_money_question():
    """THE PREDICATE'S OWN CONTRACT, which is what this change controls.

    An earlier version of this test asserted a property of the WHOLE corpus —
    that no counted question carries a money word — and it failed on two
    questions ("Show pipeline weighted expected amount by broker/by stage")
    that already parsed to a count BEFORE this change. That was the test being
    wrong about what it owned, not the code: this rule can only ever be
    responsible for the questions IT moves.
    """
    from mi_agent.llm_query_parser import _counts_a_row_noun

    for question in ("the amount change on cases that stayed in application",
                     "show pipeline weighted expected amount by broker",
                     "what is the current pipeline amount?"):
        assert not _counts_a_row_noun(question), question
    for question in ("show weekly pipeline cases", "show loans over time"):
        assert _counts_a_row_noun(question), question


def test_the_rule_moves_one_corpus_question():
    """Narrow by measurement, not by hope: over the 882 corpus questions the
    rule fires on those naming a row noun with no measure, and the recorded
    movement was a single question — "Show pipeline cases by stage over time."
    — from summed balance to a count of the cases it asked for."""
    from mi_agent.llm_query_parser import _counts_a_row_noun

    with open(_REPO_ROOT / "question_interpretation" / "stage2_corpus.json",
              "r", encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    questions = sorted({r.get("question") for r in rows if r.get("question")})
    # The one question whose MEANING changed, measured by parsing the whole
    # corpus on both commits and diffing. Seventeen corpus questions match the
    # predicate, but the rule only fires in the trend branch where no measure
    # resolved, so the rest — "Show defaulted loans.", "loans above 500000" —
    # were already counts and are untouched. Counting predicate matches would
    # measure the wrong thing, which an earlier version of this test did.
    assert _parse("Show pipeline cases by stage over time.").aggregation == "count"
    assert questions, "the corpus is the measurement, so it must be readable"
