"""Which book a delivery belongs to is a declared answer, not a guess at a name.

Storage files every delivery under ``direct`` or ``acquired``:

    raw-v2/{client}/{book}/{dataset}/{frequency}/{portfolio}/{period}/

Onboarding asks which it is — a required question with its own vocabulary,
"Originated by the client" or "Acquired from a third party". Nothing in the
path derivation read that answer. The folder was decided by one test on the
identifier: does it start with the word "acquired"?

So ``purchased_001`` filed under ``direct/`` because it does not start with the
word, and ``alp_acquired`` filed under ``direct/`` because the word is at the
end — both while the case said acquired, and neither raising anything. The
delivery landed in the wrong book and every reading of it downstream inherited
that, silently, which is the part that matters: a misfiled delivery does not
look like a failure, it looks like a smaller book.

The other two derivations in the estate return ``None`` for an id that gives no
clue — ``engine.provenance.derive_portfolio_type`` says so outright, "so the
caller can fail closed asking for an explicit type". This was the only one of
the three that guessed, and the only one that writes.
"""

from __future__ import annotations

import pytest

from operations_control.manual_intake import (
    ManualIntakeError,
    derive_book_type,
    derive_raw_prefix,
)

PERIOD = "2026-06-30"


# --------------------------------------------------------------------------- #
# The declaration decides
# --------------------------------------------------------------------------- #

class TestTheDeclaredAnswerWins:
    @pytest.mark.parametrize("portfolio,declared,expected", [
        # The cases that were WRONG: a name carrying no usable clue, and a
        # declaration that was never consulted.
        ("purchased_001", "acquired", "acquired"),
        ("alp_acquired", "acquired", "acquired"),
        ("pool_a", "acquired", "acquired"),
        ("pool_a", "direct", "direct"),
        # And the conventional names, which must be unaffected.
        ("acquired_001", "acquired", "acquired"),
        ("direct_001", "direct", "direct"),
    ])
    def test_it_is_used_rather_than_the_name(self, portfolio, declared,
                                             expected):
        assert derive_book_type(portfolio, declared) == expected

    def test_a_declaration_reaches_the_path(self):
        prefix = derive_raw_prefix(
            client_id="ERE", portfolio_id="purchased_001",
            reporting_period=PERIOD, dataset="funded", frequency="monthly",
            portfolio_type="acquired")
        assert "/acquired/" in prefix
        assert "/direct/" not in prefix

    def test_an_unrecognised_declaration_is_refused(self):
        with pytest.raises(ManualIntakeError):
            derive_book_type("pool_a", "purchased")


class TestTwoGovernedAnswersMayNotDisagree:
    """A conforming name contradicting the declaration is not a tie to break.

    Picking either silently is how the wrong one reaches storage, and both are
    answers someone gave deliberately.
    """

    @pytest.mark.parametrize("portfolio,declared", [
        ("direct_001", "acquired"),
        ("acquired_001", "direct"),
    ])
    def test_the_pair_is_refused(self, portfolio, declared):
        with pytest.raises(ManualIntakeError) as excinfo:
            derive_book_type(portfolio, declared)
        assert "declared" in str(excinfo.value).lower()


# --------------------------------------------------------------------------- #
# With no declaration, the name must carry it — or nothing is written
# --------------------------------------------------------------------------- #

class TestWithoutADeclaration:
    @pytest.mark.parametrize("portfolio,expected", [
        ("direct_001", "direct"),
        ("acquired_001", "acquired"),
        ("ACQUIRED_002", "acquired"),
    ])
    def test_a_conforming_name_still_decides(self, portfolio, expected):
        assert derive_book_type(portfolio) == expected

    @pytest.mark.parametrize("portfolio", [
        "purchased_001",     # the word, but not the convention
        "alp_acquired",      # the word, at the wrong end
        "pool_a",            # no clue at all
    ])
    def test_an_unrecognisable_name_is_refused_not_guessed(self, portfolio):
        """Each of these filed as `direct` before, whatever the case said."""
        with pytest.raises(ManualIntakeError) as excinfo:
            derive_book_type(portfolio)
        message = str(excinfo.value).lower()
        assert "originated" in message and "acquired" in message

    def test_the_refusal_reaches_the_caller_before_anything_is_written(self):
        """`derive_raw_prefix` is called BEFORE the first byte is placed, so a
        refusal here costs a message rather than a misfiled delivery."""
        with pytest.raises(ManualIntakeError):
            derive_raw_prefix(
                client_id="ERE", portfolio_id="pool_a",
                reporting_period=PERIOD, dataset="funded", frequency="monthly")


# --------------------------------------------------------------------------- #
# The ordinary case is untouched
# --------------------------------------------------------------------------- #

class TestTheConventionalNamesAreUnaffected:
    @pytest.mark.parametrize("portfolio,book", [
        ("direct_001", "direct"),
        ("acquired_001", "acquired"),
    ])
    def test_the_prefix_is_what_it_always_was(self, portfolio, book):
        prefix = derive_raw_prefix(
            client_id="ERE", portfolio_id=portfolio,
            reporting_period=PERIOD, dataset="funded", frequency="monthly")
        assert prefix.endswith(
            f"/ERE/{book}/funded/monthly/{portfolio}/{PERIOD}")

    def test_a_pipeline_delivery_is_filed_apart_from_the_funded_one(self):
        """The other axis, asserted beside this one so the two stay distinct:
        the book segment says WHOSE it is, the dataset segment says WHICH."""
        funded = derive_raw_prefix(
            client_id="ERE", portfolio_id="direct_001",
            reporting_period=PERIOD, dataset="funded", frequency="monthly")
        pipeline = derive_raw_prefix(
            client_id="ERE", portfolio_id="direct_001",
            reporting_period=PERIOD, dataset="pipeline", frequency="weekly")
        assert "/direct/funded/monthly/" in funded
        assert "/direct/pipeline/weekly/" in pipeline
