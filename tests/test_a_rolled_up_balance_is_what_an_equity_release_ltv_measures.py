"""On a lifetime mortgage, LTV is measured against the rolled-up balance.

ERE's funded book blocked at readiness on LTV002 — "reported LTV not consistent
with balance and valuation" — across 358 of 568 loans, 63.03%. Nothing was wrong
with the tape. Two separate defects met on it:

1.  THE NUMERATOR. The interest on a lifetime mortgage is never paid. It rolls
    up and is secured on the same property, so the amount lent against the house
    is the OUTSTANDING balance and the principal balance is merely the part of
    it that was advanced. A lender quotes its LTV on the rolled-up figure.
    Dividing by the principal balance instead understates LTV by the whole of
    the accrued interest, which on a seasoned loan is the larger share — so
    every loan past its first few months disagreed with its own stated LTV. The
    disagreement was manufactured by the division, not found in the data. The
    37% that passed were the recently written loans, where little had yet
    accrued.

2.  THE ESCALATION. LTV002 declares itself ``severity: warning``, and the
    production batch path reads exactly that: its gate diagnostics count a
    finding as blocking from the rule's own severity, so a warning ships as a
    warning at any rate. Onboarding instead read the AGGREGATED materiality,
    which promoted anything at or above a 25% error rate to BLOCKING regardless
    of what the rule had declared. The same file, run the same way, was a
    warning in production and a wall in onboarding.

Volume means SYSTEMATIC, not FATAL. A cross-check that disagrees across a whole
book is the strongest case there is for a human reading it, and no case at all
for refusing the book.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

from engine.gate_2_transform.canonical_transform import derive_fields  # noqa: E402
from engine.gate_3_validation.aggregate_validation_results import (  # noqa: E402
    determine_materiality,
)


@pytest.fixture(scope="module")
def rules():
    spec = importlib.util.spec_from_file_location(
        "vbr", REPO / "engine/gate_3_validation/validate_business_rules.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {r["rule_id"]: r for r in mod.RULES}


@pytest.fixture(scope="module")
def issue_policy():
    with open(REPO / "config/asset/issue_policy.yaml") as fh:
        return yaml.safe_load(fh) or {}


#: One of ERE's own loans, seasoned. £91,688 advanced against a £225,000 house,
#: £45,000 of interest rolled up since. The lender states 60.75% — 136,688/225,000
#: — and it is right. Measured against principal alone it reads 40.75%, twenty
#: points adrift of a number that is not wrong.
SEASONED = dict(
    current_principal_balance=[91_688.0],
    current_outstanding_balance=[136_688.0],
    accrued_interest=[45_000.0],
    current_valuation_amount=[225_000.0],
    current_loan_to_value=[60.75],
)

#: The same loan in its first weeks: £1,132 accrued, LTV 41.25%. Before this
#: fix, THIS is the loan that passed — which is why 37% of the book did.
NEW = dict(
    current_principal_balance=[91_688.0],
    current_outstanding_balance=[92_820.0],
    accrued_interest=[1_132.0],
    current_valuation_amount=[225_000.0],
    current_loan_to_value=[41.25],
)


class TestTheCheckAcceptsTheBalanceTheProductUses:

    def test_a_seasoned_lifetime_mortgage_is_not_an_error(self, rules):
        passed = rules["LTV002"]["test"](pd.DataFrame(SEASONED))
        assert bool(passed.iloc[0]), (
            "a lender's own LTV, correctly quoted on the rolled-up balance, "
            "was reported as inconsistent with its own book")

    def test_a_freshly_written_loan_still_passes(self, rules):
        assert bool(rules["LTV002"]["test"](pd.DataFrame(NEW)).iloc[0])

    def test_an_ltv_that_matches_neither_balance_still_fails(self, rules):
        wrong = dict(SEASONED, current_loan_to_value=[22.0])
        assert not bool(rules["LTV002"]["test"](pd.DataFrame(wrong)).iloc[0]), (
            "the check must still catch an LTV that is genuinely incoherent")

    def test_an_amortising_loan_is_tested_exactly_as_before(self, rules):
        """Where the two balances are one number, this is one test, not two."""
        amortising = dict(
            current_principal_balance=[100_000.0, 100_000.0],
            current_outstanding_balance=[100_000.0, 100_000.0],
            current_valuation_amount=[250_000.0, 250_000.0],
            current_loan_to_value=[40.0, 31.0],
        )
        got = rules["LTV002"]["test"](pd.DataFrame(amortising)).tolist()
        assert got == [True, False]

    def test_no_outstanding_balance_column_changes_nothing(self, rules):
        """A tape that carries only a principal balance is judged on it."""
        only_principal = {k: v for k, v in SEASONED.items()
                          if k != "current_outstanding_balance"}
        assert not bool(
            rules["LTV002"]["test"](pd.DataFrame(only_principal)).iloc[0])


class TestTheDerivationDividesByTheRolledUpBalance:

    def _derive(self, portfolio_type):
        df = pd.DataFrame(dict(SEASONED, current_loan_to_value=[None]))
        derive_fields(df, portfolio_type, "tape.csv", dayfirst=True,
                      infer_year=False, derive_month=False,
                      default_year=None, config={})
        return pd.to_numeric(df["current_loan_to_value"], errors="coerce").iloc[0]

    def test_equity_release_derives_against_the_outstanding_balance(self):
        assert self._derive("equity_release") == pytest.approx(60.75, abs=0.01)

    def test_a_derived_ltv_then_satisfies_the_check(self, rules):
        df = pd.DataFrame(dict(SEASONED, current_loan_to_value=[None]))
        derive_fields(df, "equity_release", "tape.csv", dayfirst=True,
                      infer_year=False, derive_month=False,
                      default_year=None, config={})
        assert bool(rules["LTV002"]["test"](df).iloc[0]), (
            "the transform and the validator must agree on the numerator")

    def test_an_amortising_portfolio_keeps_the_principal_balance(self):
        assert self._derive("rmbs") == pytest.approx(40.75, abs=0.01)


class TestTheRolledUpInterestIsReadByANameThatExists:
    """Balance coherence added ``accrued_interest``, which is not a field.

    The registry calls the cumulative figure ``cumulative_accrued_interest``
    and a single period's ``accrued_interest_in_period``. Neither is
    ``accrued_interest``, so the column was created empty on every run and the
    addition was always of nought: the step whose entire purpose is that an
    equity release balance is principal PLUS what rolled up on it was copying
    one balance to the other. A lender who sent only a principal balance got an
    outstanding balance that was not outstanding — and an LTV short by exactly
    the interest this was meant to add.
    """

    def _coherence(self, **cols):
        df = pd.DataFrame(cols)
        derive_fields(df, "equity_release", "tape.csv", dayfirst=True,
                      infer_year=False, derive_month=False,
                      default_year=None, config={})
        return df

    def test_the_outstanding_balance_includes_what_rolled_up(self):
        df = self._coherence(
            current_principal_balance=[91_688.0],
            cumulative_accrued_interest=[45_000.0],
            current_valuation_amount=[225_000.0])
        assert pd.to_numeric(
            df["current_outstanding_balance"]).iloc[0] == 136_688.0

    def test_and_the_ltv_that_follows_is_the_lenders_own(self):
        df = self._coherence(
            current_principal_balance=[91_688.0],
            cumulative_accrued_interest=[45_000.0],
            current_valuation_amount=[225_000.0])
        assert pd.to_numeric(df["current_loan_to_value"]).iloc[0] == \
            pytest.approx(60.75, abs=0.01)

    def test_a_period_accrual_is_not_the_amount_owed(self):
        """``accrued_interest_in_period`` must not be added as if cumulative."""
        df = self._coherence(
            current_principal_balance=[91_688.0],
            accrued_interest_in_period=[935.15],
            current_valuation_amount=[225_000.0])
        assert pd.to_numeric(
            df["current_outstanding_balance"]).iloc[0] == 91_688.0

    def test_a_stated_outstanding_balance_is_never_recomputed(self):
        df = self._coherence(
            current_principal_balance=[91_688.0],
            current_outstanding_balance=[136_688.0],
            cumulative_accrued_interest=[45_000.0],
            current_valuation_amount=[225_000.0])
        assert pd.to_numeric(
            df["current_outstanding_balance"]).iloc[0] == 136_688.0

    def test_no_accrual_column_leaves_the_balances_equal(self):
        """What it did before, for a tape that says nothing about interest."""
        df = self._coherence(
            current_principal_balance=[91_688.0],
            current_valuation_amount=[225_000.0])
        assert pd.to_numeric(
            df["current_outstanding_balance"]).iloc[0] == 91_688.0


class TestVolumeDoesNotPromoteAWarningIntoAWall:

    def test_a_warning_across_the_whole_book_is_a_review(self, issue_policy):
        assert determine_materiality(
            63.03, "warning", "business_logic_violation", issue_policy) == "REVIEW"

    def test_an_error_across_the_whole_book_still_blocks(self, issue_policy):
        assert determine_materiality(
            63.03, "error", "business_logic_violation", issue_policy) == "BLOCKING"

    def test_a_mandatory_null_at_scale_still_blocks(self, issue_policy):
        """CORE002 is an error. Nothing here softens a missing required value."""
        assert determine_materiality(
            30.0, "error", "mandatory_null", issue_policy) == "BLOCKING"

    def test_an_unstated_severity_is_treated_as_it_always_was(self, issue_policy):
        assert determine_materiality(
            30.0, "", "business_logic_violation", issue_policy) == "BLOCKING"

    def test_a_warning_below_the_threshold_is_unchanged(self, issue_policy):
        assert determine_materiality(
            12.0, "warning", "business_logic_violation", issue_policy) == "REVIEW"
