"""Phase 1A — the source-portfolio scope claim, at the CONTRACT boundary.

These test the interpretation contract itself, not a route. The migration
blocker Phase 0 recorded was that `QuestionInterpretation` carried no claim for a
source-portfolio lens, so a downstream plan could not tell "the whole book" from
"nobody looked" — and reading the second as the first is the P1L population
widening.

THE RULE THESE PROTECT: `mi_agent.portfolio_lens` remains the SINGLE OWNER of
what "the acquired book" means. The contract transports its answer. There is no
vocabulary in `question_interpretation` for this, and adding one here would
recreate the second-owner defect the programme spent a month removing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from question_interpretation.schema import (  # noqa: E402
    EMPTY, FILLED, UNRESOLVABLE, SCOPE_ACQUIRED, SCOPE_COHORT, SCOPE_DIRECT,
    SCOPE_TOTAL, QuestionInterpretation, SourceScopeClaim,
)


@pytest.fixture(scope="module")
def semantics():
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    return load_mi_semantics(semantics_path())


def _scope(question, semantics):
    from question_interpretation import projection
    return projection.project(question, semantics=semantics).source_scope


# --------------------------------------------------------------------------- #
# Positive: every scope the owner can resolve arrives on the contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question,expected", [
    ("Please provide a portfolio summary", SCOPE_TOTAL),
    ("summarise the portfolio", SCOPE_TOTAL),
    ("Summarise the direct book", SCOPE_DIRECT),
    ("Summarise the acquired book", SCOPE_ACQUIRED),
    ("portfolio summary for the acquired book", SCOPE_ACQUIRED),
    ("Summarise the acquired_001 book", SCOPE_COHORT),
])
def test_the_owners_scope_reaches_the_contract(question, expected, semantics):
    claim = _scope(question, semantics)
    assert claim.state == FILLED
    assert claim.scope == expected
    assert claim.source == "mi_agent.portfolio_lens"


def test_a_named_book_carries_its_id(semantics):
    claim = _scope("Summarise the acquired_001 book", semantics)
    assert claim.scope == SCOPE_COHORT
    assert claim.portfolio_ids == ("acquired_001",), claim.portfolio_ids


def test_the_contract_never_disagrees_with_its_owner(semantics):
    """The whole design in one assertion: transported, never re-derived."""
    from mi_agent import portfolio_lens as owner
    for question in ("Please provide a portfolio summary", "Summarise the direct book",
                     "Summarise the acquired book", "Summarise the front book",
                     "Summarise the acquired_001 book", "balance by region"):
        claim = _scope(question, semantics)
        assert claim.state == FILLED
        assert claim.scope == owner.resolve_lens(question).name, question


# --------------------------------------------------------------------------- #
# Distinction: absence, Total, and the seasoning axis are three different things
# --------------------------------------------------------------------------- #
def test_total_is_a_positive_reading_not_an_absence(semantics):
    """`total` means the owner LOOKED and found no source narrowing."""
    claim = _scope("Please provide a portfolio summary", semantics)
    assert claim.state == FILLED and claim.scope == SCOPE_TOTAL
    assert claim.narrows is False


def test_an_unconsulted_claim_is_not_total():
    """A default-constructed claim must never read as the whole book."""
    claim = QuestionInterpretation(question="q").source_scope
    assert claim.state == EMPTY
    assert claim.scope is None
    assert claim.narrows is False   # it narrows nothing AND it selects nothing


def test_an_unresolvable_claim_is_not_total():
    claim = SourceScopeClaim(state=UNRESOLVABLE, reason="owner unavailable")
    assert claim.scope is None and claim.narrows is False


def test_a_filled_claim_must_name_a_scope():
    with pytest.raises(ValueError):
        SourceScopeClaim(state=FILLED)


def test_an_unknown_scope_is_refused_not_mapped():
    with pytest.raises(ValueError):
        SourceScopeClaim(state=FILLED, scope="spv_of_some_kind")


def test_seasoning_is_not_a_source_lens(semantics):
    """"the front book" is a SEASONING population and no source narrowing."""
    from question_interpretation import projection
    qi = projection.project("Summarise the front book", semantics=semantics)
    assert qi.source_scope.scope == SCOPE_TOTAL
    assert qi.source_scope.narrows is False
    assert [p.concept for p in qi.population if p.state == FILLED] == \
        ["seasoning_segment"]


def test_a_source_lens_is_not_a_seasoning_population(semantics):
    from question_interpretation import projection
    qi = projection.project("Summarise the acquired book", semantics=semantics)
    assert qi.source_scope.scope == SCOPE_ACQUIRED
    assert [p.concept for p in qi.population if p.state == FILLED] == []


def test_both_axes_can_be_carried_at_once(semantics):
    """Neither implies the other, so a question may name both."""
    from question_interpretation import projection
    qi = projection.project("Summarise the front book in the acquired portfolio",
                            semantics=semantics)
    assert qi.source_scope.scope == SCOPE_ACQUIRED
    assert "seasoning_segment" in [p.concept for p in qi.population
                                   if p.state == FILLED]


def test_a_question_naming_no_scope_does_not_acquire_one(semantics):
    for question in ("balance by region", "how many loans are there?",
                     "average LTV by broker"):
        claim = _scope(question, semantics)
        assert claim.narrows is False, question


# --------------------------------------------------------------------------- #
# Ownership: the plan builder consumes the contract, and cannot read the question
# --------------------------------------------------------------------------- #
def test_the_plan_builder_takes_no_question_and_calls_no_text_resolver():
    """Structural, so an edit that reintroduces a second owner fails loudly."""
    import ast
    import inspect
    from migration_phase0 import shadow_portfolio_summary as shadow

    shadow.assert_no_question_read(None)

    source = inspect.getsource(shadow.build_plan)
    tree = ast.parse(source.lstrip())
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else
        getattr(node.func, "id", "")
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    # The text-reading entry points of the lens owner. build_plan may not call
    # any of them: it consumes the claim the contract already carries.
    forbidden = {"resolve_lens", "resolve_lens_with_default", "lens_from_term",
                 "resolve_comparison_lenses", "segments_named",
                 "resolve_population_predicate"}
    assert not (called & forbidden), sorted(called & forbidden)


def test_the_executor_rebuilds_the_lens_from_the_plan_not_the_question():
    """`lens_for` maps a resolved scope NAME to filters, through the same owner.

    Asserted over the parsed CODE with docstrings stripped — an earlier version
    of this test matched the word "question" in `lens_for`'s own prose, which
    proves nothing about what it executes.
    """
    import ast
    import inspect
    from migration_phase0 import shadow_portfolio_summary as shadow

    tree = ast.parse(inspect.getsource(shadow.lens_for).lstrip())
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else
        getattr(node.func, "id", "")
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    assert "lens_from_selection" in called, sorted(called)
    assert "resolve_lens" not in called, sorted(called)

    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "question" not in (names | attrs)


# --------------------------------------------------------------------------- #
# PHASE 1B PREREQUISITE — declared failing.
#
# The conversion of `portfolio_summary` stopped here. `source_scope` carries WHAT
# scope the owner resolved; it does not carry WHETHER THE QUESTION NAMED ONE, and
# that second fact is what decides precedence over a caller-supplied default:
#
#     portfolio_lens.resolve_lens_with_default(text, default):
#         if mentions_portfolio(text): return resolve_lens(text)   # question wins
#         return default or total_lens()                           # dropdown wins
#
# Two questions the route OWNS mention a portfolio and resolve to `total`:
#
#     "portfolio summary across all portfolios"
#     "summarise the portfolio excluding the acquired book"
#
# For those, the shipped route answers the WHOLE BOOK even when the workspace
# dropdown selects Acquired (verified end to end: £1.96bn / 11,035 loans with
# source_portfolio_lens="acquired"). A plan reading only `source_scope` sees
# `filled/total, narrows=False` — indistinguishable from "no portfolio named" —
# falls back to the dropdown, and narrows to Acquired.
#
# That is a silent population narrowing on an owned question, so the conversion
# stopped rather than working around it.
# --------------------------------------------------------------------------- #
# CLOSED IN PHASE 1G. The `xfail(strict=True)` that stood here from Phase 1B is
# removed rather than relaxed: `SourceScopeClaim.provenance` now records whether
# the question named the scope, so the two readings are distinguishable and this
# asserts the property instead of pre-registering its absence.
@pytest.mark.parametrize("question", [
    "portfolio summary across all portfolios",
    "summarise the portfolio excluding the acquired book",
])
def test_the_contract_says_whether_the_question_named_a_source_scope(question, semantics):
    """A question that SPEAKS to source scope must be distinguishable from one
    that is silent about it, even when both resolve to `total`."""
    from mi_agent import portfolio_lens as owner

    named = _scope(question, semantics)
    silent = _scope("Please provide a portfolio summary", semantics)

    assert owner.mentions_portfolio(question) is True
    assert owner.mentions_portfolio("Please provide a portfolio summary") is False
    assert named.scope == SCOPE_TOTAL and silent.scope == SCOPE_TOTAL

    # The property the conversion needs and the contract does not yet have.
    assert named.as_dict() != silent.as_dict(), (
        "the contract cannot tell 'the question said whole book' from 'the "
        "question said nothing about source scope', so a plan cannot know "
        "whether the question overrides a caller-supplied default")
