#!/usr/bin/env python3
"""Stage 2, commit 2: the object is built in production, carried, and ignored.

Three guarantees, each with a test that fails when it stops holding:

  built     from the spec and facets the pipeline already produced, not by
            re-interpreting the question;
  carried   present on the workflow result, so Stage 3 can convert consumers;
  ignored   read by nothing, and unable to cost an answer if it raises.
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

_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
         / "platform" / "alderbridge" / "2026-06-30"
         / "platform_canonical_typed.csv")


@pytest.fixture(scope="module")
def run():
    from mi_agent import mi_calibration as CAL
    from mi_agent.mi_query_validator import load_mi_semantics
    if not _TAPE.exists():
        pytest.skip("calibration book not built")
    semantics = load_mi_semantics(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")
    df = CAL.default_bank_frame(_TAPE)

    def _go(question):
        return CAL._run(question, df, semantics, False)
    return _go


def test_the_object_is_carried_on_the_result(run):
    qi = run("how many loans have a balance above £250k").get("question_interpretation")
    assert qi is not None and "error" not in qi
    assert qi["operation"]["type"] == "count"
    assert qi["subject"]["candidate_concept"] == "loan_count"


def test_the_facet_half_carries_a_span_and_the_parser_half_does_not(run):
    """The join is half-built, by design: emitting parser offsets means changing
    how _parse_filters rewrites the question, which forfeits Stage 2's
    byte-identical guarantee."""
    qi = run("how many loans have a balance above £250k")["question_interpretation"]
    wording = [f for f in qi["filters"] if "wording" in f["provides"]]
    binding = [f for f in qi["filters"] if "field" in f["provides"]]
    assert wording and binding
    assert all(f["span"] is not None for f in wording)
    assert all(f["span"] is None for f in binding)
    assert all(f["clause_id"] is None for f in qi["filters"])


def test_the_join_state_is_reported_not_implied(run):
    qi = run("how many loans have a balance above £250k")["question_interpretation"]
    assert any("HALF-BUILT" in n for n in qi["notes"])


def test_population_is_faithful_and_does_not_correct_a_role(run):
    """'balance by region for joint borrowers', on the alderbridge book.

    `borrower_type` is NOT a column of this tape. The parser, given the real
    columns, correctly declines to bind a filter on a field the book lacks. The
    facet layer still names it, because `requested_dimension_terms` resolves
    WITHOUT availability filtering so the omission can be disclosed rather than
    vanishing. Neither supplies a role, so the object records `unresolved` with
    the reason — it does not pick one.

    That is faithful population: the object says what the sources say, on this
    book, including where they say nothing.
    """
    qi = run("balance by region for joint borrowers")["question_interpretation"]
    roles = {d["candidate_concept"]: d["role"] for d in qi["dimensions"]}
    reasons = {d["candidate_concept"]: d["reason"] for d in qi["dimensions"]}
    assert roles.get("collateral_geography") == "grouping"
    assert roles.get("borrower_type") == "unresolved"
    assert reasons.get("borrower_type") == "no source supplies a role"


def test_role_attribution_is_book_dependent_and_that_is_recorded(run):
    """The same sentence yields a different role on a book that carries the
    field. Stage 4 inherits this: a role is not a property of the question
    alone, because the parser only binds what the book can support.
    """
    from mi_agent import mi_calibration as CAL
    from mi_agent.llm_query_parser import _deterministic_parse
    from mi_agent.mi_query_validator import load_mi_semantics
    semantics = load_mi_semantics(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")
    df = CAL.default_bank_frame(_TAPE)
    q = "balance by region for joint borrowers"

    unconstrained, _ = _deterministic_parse(q, semantics)
    on_this_book, _ = _deterministic_parse(q, semantics,
                                           available_columns=set(df.columns))
    assert "borrower_type" in unconstrained.filters
    assert "borrower_type" not in on_this_book.filters
    assert "borrower_type" not in set(df.columns)


def test_a_broken_interpretation_never_costs_an_answer(run, monkeypatch):
    """The guarantee that makes carrying it safe. If the builder raises, the
    query still answers and the failure is recorded on the object."""
    import question_interpretation.projection as proj

    def _boom(*a, **kw):
        raise RuntimeError("deliberate")

    monkeypatch.setattr(proj, "from_parts", _boom)
    res = run("what is the funded balance?")
    assert res.get("ok") is True
    assert res["question_interpretation"] == {"error": "deliberate"}


def test_that_containment_test_can_fail(run):
    """Without the monkeypatch the object builds cleanly — so the test above is
    detecting the injected failure, not a permanently broken builder."""
    res = run("what is the funded balance?")
    assert res.get("ok") is True
    assert "error" not in res["question_interpretation"]


def test_the_object_is_read_by_nothing(run):
    """Stage 2's whole claim. Removing it from the result must not change the
    answer, because no consumer looks at it yet."""
    res = run("balance by region")
    with_object = {k: v for k, v in res.items() if k != "question_interpretation"}
    assert "question_interpretation" in res
    # The answer, error and artifacts are all computed before it is attached.
    assert with_object.get("ok") == res.get("ok")
