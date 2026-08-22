"""Item 4 — a request for the book's overall position, however worded.

`_is_portfolio_summary` required one of NINE LITERAL PHRASES. "Tell me the
basics about this book" was refused outright; "book overview", "key metrics" and
"What are the headline numbers?" fell through to the generic executor and came
back as a two-KPI card. THREE OUTCOMES FOR ONE QUESTION, one of them the
governed summary.

THE CLASS has two halves and only one of them is a vocabulary:

  (a) the question names NOTHING ELSE — no measure, no dimension, no filter, no
      comparison, and the spec marks no specialist capability. COMPUTED.
  (b) it carries a summary-intent marker. A LIST.

(a) does all the discriminating, which is what makes this a rule rather than a
patch: `test_the_computed_half_does_the_discriminating` exercises phrasings that
share the trigger vocabulary and are correctly refused. (b) is a list and will
need extending; that is stated rather than dressed up.
"""

from __future__ import annotations

import pytest

from mi_agent import llm_query_parser as P
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent_api import chat_routing as CR


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics("mi_agent/mi_semantics_field_registry.yaml")


def _spec(question, semantics):
    out = P.parse_user_question(question, semantics)
    return out[0] if isinstance(out, tuple) else out


# --------------------------------------------------------------------------- #
# Admitted — including phrasings the vocabulary was not written from
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [
    "Please provide a portfolio summary",
    "portfolio overview",
    "summarise the book",
    "book overview",
    "key metrics",
    "What are the headline numbers for the portfolio?",
    "Tell me the basics about this book",
    "Tell me about this book",
    # None of these three was an example. They are admitted by the same rule.
    "How is the book doing?",
    "Where do we stand?",
    "give me a snapshot of the portfolio",
])
def test_the_class_is_admitted(question, semantics):
    assert CR._is_portfolio_summary(question, _spec(question, semantics))


# --------------------------------------------------------------------------- #
# THE HALF THAT MAKES IT A RULE
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question, why", [
    ("tell me about brokers", "names a dimension"),
    ("tell me about arrears", "names a measure"),
    ("tell me about the london exposure", "names a filter"),
    ("how is lending doing", "names a measure"),
    ("summarise the portfolio by region", "names a dimension"),
    ("give me an overview of balance by vintage", "names both"),
    ("Show the risk limit pass/warn/fail summary.", "spec marks risk_limit_query"),
])
def test_the_computed_half_does_the_discriminating(question, why, semantics):
    """Every one of these carries a summary-intent marker and is still refused.

    If the vocabulary were doing the work, they would all be admitted — which is
    the difference between a rule and a list of five sentences."""
    assert not CR._is_portfolio_summary(question, _spec(question, semantics)), why


@pytest.mark.parametrize("question", [
    "what is the CPR of this book",
    "what is the contractual WAL of this book",
])
def test_an_unavailable_measure_keeps_its_capability_explanation(question, semantics):
    """THE EXCLUSION THAT MATTERS MOST. These name a measure the registry does
    not govern, and the right answer is to say so. A summary nobody asked for
    would be a worse answer than an honest refusal, and it would look like one
    of the wrong-number defects this programme has been closing."""
    assert not CR._is_portfolio_summary(question, _spec(question, semantics))


def test_a_specialist_marker_is_read_not_re_derived():
    """The spec markers are set by the parser and consumed by their own
    recognisers. Duplicating any of their vocabularies here would recreate the
    multi-owner defect closed six times in this sequence."""
    import types
    marked = types.SimpleNamespace(risk_limit_query=True)
    assert CR._names_something_else("give me a summary", marked)
    assert not CR._names_something_else("give me a summary", None)


def test_the_trigger_is_declared_as_a_vocabulary():
    """Stated rather than dressed up: (b) is a finite list, the same kind as
    lexical.AXIS_MARKERS, and it will need extending."""
    assert isinstance(CR._SUMMARY_INTENT, tuple)
    assert "summary" in CR._SUMMARY_INTENT and "snapshot" in CR._SUMMARY_INTENT
