"""Item 1 — one vocabulary for "is this a comparator", two independent detectors.

`llm_query_parser._FILTER_COMPARATORS` builds the predicate that narrows rows.
`execution_receipt._THRESHOLD_PATTERNS` records that the SENTENCE asked for a
narrowing. They kept separate word lists and agreed on 16 of 30 phrases. Where
both were blind — bigger/larger/higher/smaller/lower than — the narrowing
vanished, no facet was raised, the honour-or-clarify guard had nothing to
honour, and the whole book came back as fact.

WHAT IS SHARED IS THE VOCABULARY, NOT THE OWNER. The two must keep detecting
independently: a threshold the parser misses must still be RAISED, so the guard
refuses instead of answering the whole book. `test_the_detectors_stay_separate`
is the guard on that, and it is the one that matters most — the tempting
"simplification" is to derive the facet from the applied filters, which would
delete the protection this whole item exists to strengthen.
"""

from __future__ import annotations

import re

import pytest

from mi_agent import execution_receipt as R
from mi_agent import llm_query_parser as P
from question_interpretation import lexical as L


def _parser_ops(probe: str):
    return [op for pattern, op in P._FILTER_COMPARATORS
            if re.search(pattern, probe, re.I)]


def _receipt_words(probe: str):
    return [w for pattern, w in R._THRESHOLD_PATTERNS
            if re.search(pattern, probe, re.I)]


# --------------------------------------------------------------------------- #
# One vocabulary
# --------------------------------------------------------------------------- #
def _code_strings(module):
    """Every string LITERAL in a module that is not a docstring.

    Read from the AST rather than the source text, because comments are not in
    the AST — and the first version of this test failed on the comment
    EXPLAINING the fix, which is a test measuring prose rather than code.
    """
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(module))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    body[0].value, ast.Constant) and isinstance(
                    body[0].value.value, str):
                docstrings.add(id(body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docstrings]


def test_neither_consumer_keeps_its_own_word_list():
    """The phrases live in ONE place. A consumer that hard-codes its own is how
    the two lists came to disagree on fourteen of thirty phrases."""
    for module, name in ((P, "llm_query_parser"), (R, "execution_receipt")):
        literals = " || ".join(_code_strings(module))
        for phrase in ("bigger than", "larger than", "smaller than",
                       "in excess of", "capped at"):
            assert phrase not in literals, (
                "%s names %r in its own code; the vocabulary is owned by "
                "question_interpretation.lexical" % (name, phrase))


@pytest.mark.parametrize("phrase", [
    "over", "above", "more than", "greater than", "bigger than", "larger than",
    "higher than", "exceeding", "in excess of", "at least", "no less than",
    "minimum of", "greater than or equal to", "under", "below", "less than",
    "fewer than", "smaller than", "lower than", "beneath", "at most",
    "no more than", "up to", "maximum of", "capped at", "older than",
    "younger than",
])
def test_both_consumers_know_every_phrase(phrase):
    """The measured defect: they agreed on 16 of 30 and the gaps were silent."""
    probe = "loans %s 150000" % phrase
    assert _parser_ops(probe), "the parser cannot apply %r" % phrase
    assert _receipt_words(probe), "the receipt cannot raise %r" % phrase


@pytest.mark.parametrize("phrase, op", [
    ("no more than", "le"), ("not more than", "le"), ("no less than", "ge"),
    ("greater than or equal to", "ge"), ("less than or equal to", "le"),
    ("more than", "gt"), ("less than", "lt"), ("equal to", "eq"),
])
def test_a_negated_comparator_does_not_invert(phrase, op):
    """"no more than 150000" CONTAINS "more than 150000". Without the guard the
    `gt` pattern matches inside the `le` phrase and the filter runs backwards —
    a confidently reversed narrowing, which is worse than no narrowing."""
    assert _parser_ops("loans %s 150000" % phrase) == [op]


def test_the_vocabulary_is_ordered_longest_first():
    """Ordering is load-bearing and owned, so a caller cannot reintroduce the
    shadowing bug by reordering its own list."""
    lengths = [len(p) for p, _ in L.COMPARATOR_PHRASES]
    assert lengths == sorted(lengths, reverse=True)


def test_the_receipt_word_comes_from_the_operator():
    """One mapping, not a second list of phrases mapping to words."""
    for _, op in L.COMPARATOR_PHRASES:
        assert op in L.COMPARATOR_WORD


# --------------------------------------------------------------------------- #
# THE ONE THAT MATTERS: two detectors, deliberately
# --------------------------------------------------------------------------- #
def test_the_detectors_stay_separate():
    """The receipt reads the SENTENCE, never the applied filters.

    This is the protection, not an inefficiency. Today `exceeding`, `up to` and
    the rest are known to both — but the moment a phrase is known to the
    receipt and not the parser, the facet is raised, execution cannot honour
    it, and the guard REFUSES rather than answering the whole book. Deriving
    the facet from the filters would turn every such phrase silent, which is
    exactly the failure this item was opened to remove.
    """
    facets = R._detect_thresholds("what is the ltv for loans flurbed than 150000")
    assert facets == [], "sanity: an unknown word raises nothing"

    raised = R._detect_thresholds("what is the ltv for loans over 150000")
    assert raised, "a threshold in the sentence must be raised"
    # Raised from the TEXT ALONE — no spec, no filters, no execution result was
    # passed in, and none can be.
    import inspect
    assert list(inspect.signature(R._detect_thresholds).parameters) == ["q"]


# --------------------------------------------------------------------------- #
# The currency window
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("phrase", [
    "over", "above", "more than", "bigger than", "larger than", "higher than",
    "smaller than", "lower than", "less than",
])
def test_the_currency_window_covers_the_whole_phrase(phrase):
    """A fixed twelve-character probe was wide enough for "over " and "more
    than " and one character too narrow for "bigger than " — so the £ fell
    outside, the currency test failed, and the threshold bound to
    `current_loan_to_value` instead of the balance. No loan has an LTV over
    150,000, so the answer became a refusal: safer than the whole-book figure,
    and still not the answer.

    A fixed span around a variable-length vocabulary is the same hard-coding
    one layer below the lists themselves."""
    clause = "loans %s £150k" % phrase
    m = None
    for pattern, _ in P._FILTER_COMPARATORS[1:]:
        m = re.search(pattern, clause)
        if m:
            break
    assert m is not None, "no comparator matched %r" % clause
    window = clause[max(0, m.start() - 2):m.end()]
    assert re.search(r"£\s*$|£\s*\d", window), (
        "the currency sign falls outside the window for %r" % phrase)
