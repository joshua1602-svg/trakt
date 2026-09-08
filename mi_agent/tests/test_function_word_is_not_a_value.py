#!/usr/bin/env python3
"""A bare English function word is not a category value, whatever the tape carries.

WHY THIS FILE EXISTS. Value matching is against the values the BOOK actually
carries — no vocabulary of our own, so an asset class the tape does not carry
offers nothing to match. That is the right instinct and it is what makes the
value owner portable between clients.

Real tapes carry short codes. Measured on a live book whose `internal_risk_grade`
is A/B/C: the "a" in *"Give me **a** concise overview of the funded portfolio"*
was claimed as a governed value on that field. Execution never applied it — it is
an article, not a filter — so the coverage ledger found a stated concept with no
disposition and refused, exactly as it is designed to:

    "I understood that you asked about a, but I could not confirm it was applied
     to this calculation. I have not answered over a wider population instead."

The guard was right. The concept was fabricated. **12 of the 166 accepted
questions** broke that way on the live book — every one a natural phrasing
("Give me a…", "Show a table of…") — while the synthetic acceptance book, which
carries no single-letter value, showed none of it. An entire class of defect that
the bank could not see, because the bank's data had no short codes in it.

The tests below therefore INJECT what a real tape has. A fixture without a
one-character value cannot exercise this, which is the whole point.

The guard sits in `value_field` because that is the owner both the coverage
ledger and the span mask route through. Only the ledger was actually
poisoned — `mask_value_spans` admits multi-word values only, so a
one-character code was never maskable — and the last class below pins that
asymmetry, because it is the reason the fix belongs where it does.
"""
from __future__ import annotations

import unittest

from mi_agent import categorical_spans as CS

#: A book shaped like a real one: ordinary values, plus a graded field whose
#: codes are single characters.
BOOK = {
    "internal_risk_grade": ["A", "B", "C"],
    "geographic_region_obligor": ["London", "Scotland", "North West"],
    "erm_product_type": ["lump_sum", "drawdown"],
    "account_status": ["performing", "redeemed"],
}


class TestFunctionWordsAreNotValues(unittest.TestCase):

    def test_the_article_that_broke_the_live_book(self):
        self.assertIsNone(
            CS.value_field("a", BOOK),
            "the article 'a' resolved to a governed field; every question "
            "containing it would be refused for a concept nothing can carry")

    def test_the_closed_classes(self):
        """One token, and only words that can never name a category alone."""
        for word in ("a", "an", "the", "i", "it", "is", "of", "in", "to",
                     "and", "no", "not", "for", "by", "with", "what", "how"):
            with self.subTest(word=word):
                self.assertIsNone(CS.value_field(word, BOOK))

    def test_case_and_spacing_do_not_smuggle_one_through(self):
        for variant in ("A", " a ", "  THE  "):
            with self.subTest(variant=variant):
                self.assertIsNone(CS.value_field(variant, BOOK))


class TestRealValuesStillResolve(unittest.TestCase):
    """The guard is narrow. Everything that matched before still matches."""

    def test_business_values_are_untouched(self):
        for value, field in (("London", "geographic_region_obligor"),
                             ("Scotland", "geographic_region_obligor"),
                             ("lump sum", "erm_product_type"),
                             ("drawdown", "erm_product_type"),
                             ("performing", "account_status"),
                             ("redeemed", "account_status")):
            with self.subTest(value=value):
                hit = CS.value_field(value, BOOK)
                self.assertIsNotNone(hit, "%r stopped resolving" % value)
                self.assertEqual(hit[0], field)

    def test_no_content_word_is_on_the_list(self):
        """The list must never grow to include a word a book could mean.

        `direct`, `total` and `offer` are portfolio scopes, lens names and
        pipeline stages elsewhere in the estate. A guard that swallowed them
        would trade a false positive for a silent narrowing, which is the worse
        of the two.
        """
        for word in ("direct", "acquired", "total", "offer", "completed",
                     "london", "drawdown", "performing"):
            with self.subTest(word=word):
                self.assertNotIn(word, CS._FUNCTION_WORDS)

    def test_a_multi_word_span_is_never_blocked(self):
        """Only a span of exactly one function word is refused."""
        BOOK_WITH_PHRASE = dict(BOOK, plan_name=["a la carte"])
        hit = CS.value_field("a la carte", BOOK_WITH_PHRASE)
        self.assertIsNotNone(hit,
                             "a multi-word value beginning with an article was "
                             "blocked; the guard is meant to be one token wide")
        self.assertEqual(hit[0], "plan_name")


class TestTheSpanMaskWasNeverTheProblem(unittest.TestCase):
    """Why the guard belongs in `value_field` and not in the span mask.

    `mask_value_spans` blanks claimed spans out of a question before the scope
    owner reads it — a plausible second victim of the same bug, and it is not
    one: `_claimable` admits only MULTI-WORD values, so a one-token value has
    never been maskable. That asymmetry is precisely why a single-character grade
    could poison the coverage ledger while leaving the mask untouched, and why
    the guard has to sit in `value_field`, which both paths share.
    """

    def test_a_single_token_value_is_not_maskable_by_construction(self):
        self.assertEqual(CS.value_spans("Show balance for London loans", BOOK), ())

    def test_a_multi_word_value_is_still_masked(self):
        """WHITESPACE in the book's own spelling decides, not the normalised
        form: `lump_sum` is one word and unmaskable, `North West` is two and is
        masked. The guard added here changes neither."""
        question = "Show balance for North West loans"
        masked = CS.mask_value_spans(question, BOOK)
        self.assertNotIn("North West", masked)
        self.assertEqual(len(masked), len(question),
                         "masking must preserve offsets for the next reader")

    def test_the_question_that_broke_reaches_the_scope_owner_intact(self):
        question = "Give me a concise overview of the funded portfolio."
        self.assertEqual(CS.mask_value_spans(question, BOOK), question)


if __name__ == "__main__":
    unittest.main()
