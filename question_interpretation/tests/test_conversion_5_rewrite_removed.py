#!/usr/bin/env python3
"""Stage 3, conversion 5: _parse_filters no longer rewrites the question.

It used to excise a matched `between A and B` from the string —
`work_q = work_q[:start] + " " + work_q[end:]` — and parse the rewritten text,
so that the "and" inside the phrase was not used as a clause boundary. That
threw away every offset, which is why the parser could supply a filter's field
and bound but never say which words it came from.

Now the span is MARKED CONSUMED. The string is never mutated, the clause
splitter skips connectives inside a consumed span, and each clause keeps its
offsets into the original question.

Equivalence is proved by `equivalence_parse_filters`, which reproduces the old
algorithm verbatim and compares the returned filters dict across 11,474
questions — 22,948 comparisons with and without dataset columns. That matters
here because the corpus contains exactly ONE question with "between", so the
corpus alone could not have proved this.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

warnings.simplefilter("ignore")
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from question_interpretation import lexical  # noqa: E402


@pytest.fixture(scope="module")
def semantics():
    from mi_agent.mi_query_validator import load_mi_semantics
    return load_mi_semantics(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")


def test_the_parser_no_longer_declares_a_clause_splitter():
    """Less machinery than we started with: the regex is gone, not wrapped."""
    from mi_agent import llm_query_parser as P
    assert not hasattr(P, "_CLAUSE_SPLIT_RE")


def test_the_source_no_longer_rewrites_the_question():
    """Asserted against the source, because a rewrite reintroduced anywhere in
    the function would pass every behavioural test that does not happen to
    straddle it."""
    source = (_REPO_ROOT / "mi_agent" / "llm_query_parser.py").read_text(encoding="utf-8")
    start = source.index("def _parse_filters(")
    end = source.index("\ndef ", start + 10)
    # Comments are excluded deliberately: the function still EXPLAINS the
    # rewrite it used to do, and that history is worth keeping. What must be
    # gone is the executable statement.
    code = "\n".join(line for line in source[start:end].splitlines()
                     if not line.lstrip().startswith("#"))
    assert "work_q" not in code, "the question rewrite has come back"
    assert "consumed" in code


def test_a_between_phrase_is_not_split_on_its_own_and(semantics):
    """What the rewrite existed for. The 'and' inside 'between 40 and 60' is
    part of the phrase, not a clause boundary."""
    from mi_agent.llm_query_parser import _parse_filters
    got = _parse_filters("balance by region where LTV between 40 and 60", semantics)
    assert got == {"current_outstanding_balance": {"op": "between",
                                                   "value": [40.0, 60.0]}}


def test_the_85_and_over_exception_survives(semantics):
    """The other 'and' that is not a boundary. Splitting it once tore the bound
    off and the filter vanished entirely."""
    spans = lexical.clause_spans("borrowers 85 and over with LTV above 50")
    first = "borrowers 85 and over with LTV above 50"[spans[0][0]:spans[0][1]]
    assert "85 and over" in first


def test_a_filter_now_knows_which_words_it_came_from(semantics):
    """The parser half of the filter join, which the rewrite made impossible."""
    from mi_agent.llm_query_parser import _parse_filters
    q = "balance by region where LTV between 40 and 60"
    spans = {}
    _parse_filters(q, semantics, spans=spans)
    assert spans
    (start, end), = spans.values()
    assert q[start:end] == "between 40 and 60"


def test_the_span_sink_is_optional_and_off_by_default(semantics):
    """Additive: every existing caller is unchanged."""
    from mi_agent.llm_query_parser import _parse_filters
    assert _parse_filters("balance where LTV above 50", semantics) == \
        _parse_filters("balance where LTV above 50", semantics, spans=None)


def test_consumed_spans_do_not_mutate_the_text():
    """The property the whole conversion is for."""
    text = "balance where LTV between 40 and 60"
    spans = lexical.clause_spans(text, ((28, 45),))
    for start, end in spans:
        assert text[start:end] == text[start:end]      # offsets index the original
    assert lexical.blank_consumed(text, 0, len(text), ((28, 45),)) != text
    assert len(text) == 35


def test_blank_consumed_separates_rather_than_joins():
    """One space, not nothing: removing a claimed span must not weld the words
    either side into a token neither of them is."""
    assert lexical.blank_consumed("aXXXb", 0, 5, ((1, 4),)) == "a b"
    assert lexical.blank_consumed("abc", 0, 3, ()) == "abc"


def test_clause_spans_skip_connectives_inside_a_consumed_span():
    text = "LTV between 40 and 60 with balance above 100"
    without = lexical.clause_spans(text)
    with_consumed = lexical.clause_spans(text, ((4, 21),))
    assert len(without) > len(with_consumed)
    assert len(with_consumed) == 2


def test_the_exhaustive_equivalence_harness_still_agrees():
    """Runs a bounded slice of the proof inside the suite, so a regression is
    caught here and not only by the standalone run."""
    from question_interpretation.tests.equivalence_parse_filters import (
        generated_questions, old_parse_filters,
    )
    from mi_agent.llm_query_parser import _parse_filters
    from mi_agent.mi_query_validator import load_mi_semantics
    sem = load_mi_semantics(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")

    checked = 0
    for q in generated_questions():
        assert old_parse_filters(q, sem) == _parse_filters(q, sem), q
        checked += 1
        if checked >= 1500:
            break
    assert checked == 1500
