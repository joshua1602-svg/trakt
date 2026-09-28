"""A repaid loan's balance is nought, not missing.

REPORTED FROM THE LIVE CASE, the last thing between a lender and their own MI:

    current_principal_balance: CORE002 affects 1 record(s) (0.18%)
                               — materiality BLOCKING

    "One of the loans was repaid and hence the current balance is zero."

A lender's balances extract carries the loans that still have a balance, so a
loan redeemed during the period is simply absent from it. The loan extract
still lists it — it was on the book for part of the period and belongs in the
reporting — and the consolidated tape therefore shows a blank, which validation
refuses as a missing mandatory value.

The refusal is wrong, and not by a technicality. Nothing is missing: the
borrower repaid. The outstanding balance of a repaid loan is nought, and that
is a property of the PRODUCT rather than of this client's data quality, which
is why it is stated in the asset pack and only read here. The operator's own
rule, in their words:

    "Any core_canonical: true fields that are not met for MI purposes must
     first consult the asset and client configuration to assess whether there
     are any rules."

AND IT IS NOT A DEFAULT FOR AN ABSENT VALUE, which is the whole safety
argument. Defaulting a missing balance to nought would understate a portfolio
silently the first time a balances extract arrived truncated — the exact class
of failure this work exists to remove. So the rule fires on POSITIVE EVIDENCE
that the account has closed, and the test that matters most here is
``TestATruncatedBalancesExtractStillRefuses``: every live loan a short file
dropped stays blank and stays blocking.

PARTIAL REPAYMENTS NEVER REACH IT. A loan repaying part of its balance stays
open and keeps its status. Confirmed with the first ERM lender through
onboarding — "Redeemed means fully repaid" — and the asset pack carries the
warning for any lender whose status vocabulary says otherwise.
"""

from __future__ import annotations

import pandas as pd

from operations_control.occ_agent.execution import (
    _closed_account_zeroing,
    _closed_sentence,
)

FIELD = "current_principal_balance"
ASSET = "equity_release"


def _book(statuses, balances, field=FIELD):
    return pd.DataFrame({
        "loan_identifier": [f"L{i}" for i in range(len(statuses))],
        "account_status": statuses,
        field: balances,
    })


class TestTheReportedCase:
    def test_the_repaid_loan_gets_a_balance_of_nought(self):
        book = _book(["PERF", "PERF", "RDMD"], [100.0, 200.0, None])
        _closed_account_zeroing(book, ASSET)
        assert list(book[FIELD]) == [100.0, 200.0, 0.0]

    def test_the_live_loans_are_untouched(self):
        """A rule that reaches a loan it was not about is a worse bug than the
        one it fixes."""
        book = _book(["PERF", "PERF", "RDMD"], [100.0, 200.0, None])
        _closed_account_zeroing(book, ASSET)
        assert list(book[FIELD])[:2] == [100.0, 200.0]

    def test_it_reads_the_lender_s_own_label_as_well_as_the_code(self):
        """``account_status`` is a governed enum — Redeemed -> RDMD — and this
        runs either side of that normalisation."""
        book = _book(["Active", "Redeemed"], [100.0, None])
        _closed_account_zeroing(book, ASSET)
        assert list(book[FIELD]) == [100.0, 0.0]

    def test_it_says_what_it_wrote(self):
        """A value the platform wrote is not the same as one the lender sent,
        and the tape does not distinguish them."""
        book = _book(["RDMD"], [None])
        said = _closed_sentence(_closed_account_zeroing(book, ASSET))
        assert "closed account status" in said
        assert "nought" in said
        assert "current principal balance" in said


class TestATruncatedBalancesExtractStillRefuses:
    """THE TEST THAT MATTERS MOST.

    The objection to this rule is that it writes a nought where the lender
    wrote nothing, and a rule that does that indiscriminately would report a
    whole portfolio at nought the day a balances file arrived short — silently,
    and looking exactly like data.

    It does not do that. It reads a status, and a loan that has not closed is
    left blank and goes on blocking.
    """

    def test_live_loans_with_no_balance_stay_blank(self):
        book = _book(["PERF", "PERF", "RDMD"], [None, None, None])
        _closed_account_zeroing(book, ASSET)
        assert book[FIELD].isna().tolist() == [True, True, False]

    def test_a_blank_status_derives_nothing(self):
        """No evidence is not evidence of closure."""
        book = _book(["", None], [None, None])
        _closed_account_zeroing(book, ASSET)
        assert book[FIELD].isna().all()

    def test_an_unrecognised_status_derives_nothing(self):
        book = _book(["IN ARREARS", "DEFAULTED"], [None, None])
        _closed_account_zeroing(book, ASSET)
        assert book[FIELD].isna().all()

    def test_a_book_with_no_status_column_is_left_alone(self):
        """A lender who does not send an account status gets the behaviour they
        had before this rule existed, not a tape of noughts."""
        book = pd.DataFrame({"loan_identifier": ["L0"], FIELD: [None]})
        out = _closed_account_zeroing(book, ASSET)
        assert book[FIELD].isna().all()
        assert not out.get("filled")


class TestAStatedBalanceAlwaysWins:
    def test_a_closed_account_stating_a_balance_keeps_it(self):
        """The lender says the loan is repaid AND that money is owed on it.
        Only they can say which is right, so neither is overwritten."""
        book = _book(["RDMD"], [4250.0])
        _closed_account_zeroing(book, ASSET)
        assert list(book[FIELD]) == [4250.0]

    def test_the_contradiction_is_counted_and_named(self):
        book = _book(["RDMD", "RDMD"], [4250.0, 10.0])
        out = _closed_account_zeroing(book, ASSET)
        assert out["contradicted"][FIELD] == 2
        assert not out.get("filled")

    def test_a_stated_nought_is_not_a_contradiction(self):
        """A lender who reports the repaid loan at nought themselves is
        agreeing with us, not disagreeing."""
        book = _book(["RDMD"], [0.0])
        out = _closed_account_zeroing(book, ASSET)
        assert not out.get("contradicted")
        assert not out.get("filled")


class TestItIsTheAssetClassThatAnswers:
    def test_another_asset_class_gets_nothing(self):
        """The ERM pack speaks for equity release. A buy-to-let book with a
        closed-account status of its own is not governed by it."""
        book = _book(["RDMD"], [None])
        out = _closed_account_zeroing(book, "buy_to_let")
        assert out == {}
        assert book[FIELD].isna().all()

    def test_the_pack_is_what_names_the_statuses(self):
        """Not restated in code. If the pack stops listing a status, the rule
        stops firing on it."""
        book = _book(["RDMD"], [None])
        out = _closed_account_zeroing(book, ASSET)
        assert "RDMD" in out["statuses"]

    def test_both_balances_are_answered(self):
        """They are the same fact at different points of the waterfall, and a
        tape that zeroed one and left the other blank would be incoherent."""
        book = pd.DataFrame({
            "loan_identifier": ["L0"],
            "account_status": ["RDMD"],
            "current_principal_balance": [None],
            "current_outstanding_balance": [None],
        })
        _closed_account_zeroing(book, ASSET)
        assert list(book["current_principal_balance"]) == [0.0]
        assert list(book["current_outstanding_balance"]) == [0.0]


class TestItNeverTakesADeliveryDown:
    def test_an_empty_book_is_not_an_error(self):
        book = _book([], [])
        assert _closed_account_zeroing(book, ASSET) is not None

    def test_a_missing_asset_class_is_not_an_error(self):
        book = _book(["RDMD"], [None])
        assert _closed_account_zeroing(book, "") == {}

    def test_nothing_is_said_when_nothing_was_filled(self):
        """The sentence is for news. A book with no redemptions this month
        must not carry a line about redemptions."""
        book = _book(["PERF"], [100.0])
        assert _closed_sentence(_closed_account_zeroing(book, ASSET)) == ""
