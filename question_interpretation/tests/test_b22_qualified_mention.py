"""B22 — a bare substring must not silently narrow the population.

`portfolio_lens.resolve_lens` read `direct`, `acquired`, `purchased`, `organic`
as bare substrings. Those are ordinary English about lending, so:

    "What is the balance for loans purchased at auction?"
      -> Acquired cohort, 3,909 of 11,035 loans

A complete, correctly formatted answer over 35% of the book, for a question
about how a property was bought.

CONSTRUCTED COVERAGE. The lens narrows ZERO of the 697 corpus questions; its only
pre-existing coverage is `tests/test_p1i_scope_resolution.py` and
`tests/test_p1j1_vintage_seasoning.py`. Every case here is constructed, so a
green corpus proves the fix did not reach the corpus — not that it is right.
"""
from __future__ import annotations

import pytest

from mi_agent import portfolio_lens as PL


def _lens(question):
    return PL.resolve_lens(question).name


# --------------------------------------------------------------------------- #
# Ordinary English must not select a book
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [
    "What is the balance for loans purchased at auction?",
    "What is the balance where the borrower acquired the property recently?",
    "Show balance by region for directly held collateral.",
    "How many loans were bought at auction by the borrower?",
    "What is the organic growth in balance?",
])
def test_an_unqualified_mention_does_not_select(question):
    assert _lens(question) == PL.LENS_TOTAL, question


# --------------------------------------------------------------------------- #
# THE CAN-FAIL — the same words, doing the job the decision assumes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("phrase,expected", [
    ("the acquired book", PL.LENS_ACQUIRED),
    ("the purchased book", PL.LENS_ACQUIRED),
    ("the bought book", PL.LENS_ACQUIRED),
    ("the acquisition book", PL.LENS_ACQUIRED),
    ("the purchased back book", PL.LENS_ACQUIRED),
    ("the direct book", PL.LENS_DIRECT),
    ("the directly originated book", PL.LENS_DIRECT),
    ("directly originated loans", PL.LENS_DIRECT),
    ("organic lending", PL.LENS_DIRECT),
])
def test_a_qualified_mention_still_selects(phrase, expected):
    """A fix that stopped the lens narrowing would satisfy every test above and
    destroy the decision.

    THE VOCABULARY IS THE FRAGILE PART OF THIS FIX, and it was wrong twice
    before it was right. Every correction came from coverage written years
    earlier, by someone else:

      `_SCOPE_QUALIFIERS` alone      dropped "the bought book", "the acquisition
                                     book", "organic lending"   (test_p1j1)
      adding "lending" alone         dropped "directly originated loans" and
                                     "the purchased back book"  (test_source_
                                                                 portfolio_
                                                                 provenance)

    Nine phrases, and I would not have found five of them. That is the argument
    for recording a defect with a test rather than only fixing it, arriving from
    the other side: the tests that caught these were the RECORD of earlier
    provenance defects, and they were still working.
    """
    assert _lens(f"what is the balance of {phrase}?") == expected


# --------------------------------------------------------------------------- #
# One helper, two callers — not two implementations
# --------------------------------------------------------------------------- #
def test_the_qualified_mention_test_has_one_implementation():
    """Ruling 1. The scope resolver and the lens resolver ask the same question
    of different vocabularies; duplicating the test would create a second owner
    of the decision B22 exists to consolidate."""
    import inspect

    source = inspect.getsource(PL)
    assert source.count("def _qualified_span_re") == 1
    # Both span functions are built from it, neither hand-rolls a pattern.
    assert "_LENS_PHRASE_RE = _qualified_span_re(" in source
    assert PL.lens_phrase_spans("of the acquired book")
    assert PL.scope_phrase_spans("of the funded book")


def test_the_two_vocabularies_are_genuinely_different():
    """Which is why the helper is parameterised rather than fixed."""
    assert "bought" not in PL._SCOPE_QUALIFIERS
    assert any("bought" in q for q in PL._LENS_QUALIFIERS)
    for noun in ("lending", "loans"):
        assert noun not in PL._SCOPE_NOUNS
        assert noun in PL._LENS_NOUNS


def test_one_adjective_may_sit_between_the_qualifier_and_the_noun():
    """"the purchased BACK book" is governed provenance language.

    Bounded to one short word, and a noun is still required — which is what
    keeps "purchased at auction" (qualifier, two words, no noun) out.
    """
    assert _lens("what is the balance of the purchased back book?") == \
        PL.LENS_ACQUIRED
    assert _lens("What is the balance for loans purchased at auction?") == \
        PL.LENS_TOTAL


# --------------------------------------------------------------------------- #
# A disclaiming construction DECLINES; it does not select the opposite
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [
    "What is the balance excluding the acquired book?",
    "What is the balance ignoring the acquired book?",
    "What is the balance other than the direct book?",
    "Show balance by region net of the acquired book.",
])
def test_a_disclaimed_scope_declines(question):
    """Ruling 2. "Excluding the acquired book" states what is NOT wanted.
    Inferring "therefore the direct book" is a guess about scope, and a guess
    about scope is a substitution."""
    assert _lens(question) == PL.LENS_TOTAL
    assert PL.disclaims_scope(question)


def test_the_disclaimed_phrase_is_reported_not_just_the_fact():
    """So a reader recording the declined narrowing gets both from one owner
    rather than re-deriving the wording."""
    assert PL.disclaimed_scope_phrase(
        "What is the balance excluding the acquired book?") == \
        "excluding the acquired book"
    assert PL.disclaimed_scope_phrase("What is the balance?") is None


def test_a_plain_selection_is_not_disclaimed():
    assert not PL.disclaims_scope("What is the balance of the acquired book?")


# --------------------------------------------------------------------------- #
# A bare TERM is not a question
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("term,expected", [
    ("direct", PL.LENS_DIRECT),
    ("acquired", PL.LENS_ACQUIRED),
    ("the direct book", PL.LENS_DIRECT),
    ("the acquired book", PL.LENS_ACQUIRED),
])
def test_a_term_caller_needs_no_qualification(term, expected):
    """Several call sites hold a book name already — a registry key, a spec
    field, a planner branch that has matched it. Requiring qualification without
    a separate entry point would tell them they hold none.

    `mi_workflows/analytical/populations.py` had already worked around the
    absence of this entry point by building `f"the {lens_term} book"` by hand,
    which is how the distinction was found.
    """
    assert PL.lens_from_term(term).name == expected


def test_the_term_entry_point_is_not_the_question_one():
    """The two are separate on purpose: a bare term selects through one and not
    through the other, and that difference IS the fix."""
    assert PL.lens_from_term("acquired").name == PL.LENS_ACQUIRED
    assert PL.resolve_lens("acquired").name == PL.LENS_TOTAL


# --------------------------------------------------------------------------- #
# The seasoning vocabulary stays out — P1J-1's defect must not return
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("phrase", [
    "new lending", "new originations", "recent originations", "the front book",
    "the back book",
])
def test_a_seasoning_phrase_is_never_provenance(phrase):
    """"organic lending" made `lending` a lens noun, which brings the seasoning
    vocabulary within reach of the same pattern. It must not select."""
    assert _lens(f"what is the balance of {phrase}?") == PL.LENS_TOTAL
