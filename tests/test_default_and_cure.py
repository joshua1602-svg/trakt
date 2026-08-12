#!/usr/bin/env python3
"""Default and cure, on portfolios small enough to check by hand.

Two metrics with very different provenance, and the tests say which is which.

**Default rate is the regulator's.** ESMA's Annex 12 investor-report schema
(``DRAFT1auth.098.001.04``, ``AnlsdCstDfltRate``) defines it verbatim:

    Periodic CDR = (total current balance of underlying exposures classified as
    defaulted DURING the period) / (total current balance of NON-DEFAULTED
    underlying exposures at the BEGINNING of the period)

Every choice a default rate normally forces — balance versus count, flow versus
stock, whether already-defaulted loans sit in the denominator — is answered by
that sentence, so the tests assert against it rather than against a preference.

**Cure rate is not.** ESMA carries no loan-level cure concept at all: its only
"cure" fields are covenant and trigger cure periods. So the cure convention is
Trakt's, selected and versioned, and the tests pin the two decisions that
matter — the denominator is the population *able* to cure, and improvement
within delinquency is **not** a cure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analytics_lib.history import (
    CURE_RATE_METHOD,
    DEFAULT_RATE_METHOD,
    cure_rate,
    default_rate,
)

P1, P2, P3 = "2026-01-31", "2026-02-28", "2026-03-31"


def _frame(rows):
    """rows: (loan_id, balance, status, dpd)."""
    return pd.DataFrame([
        {"loan_identifier": loan, "current_principal_balance": balance,
         "account_status": status, "number_of_days_in_arrears": dpd}
        for loan, balance, status, dpd in rows])


# =========================================================================== #
# Default rate — the brief's worked example
# =========================================================================== #
def _default_book():
    """The brief's case:

        opening   A £100 current   B £100 current   C £100 defaulted
        closing   A £100 current   B £100 DEFAULTED C £100 defaulted

    On ESMA's definition the answer follows without a choice being made:

        numerator   = B's closing balance                = £100
                      (A did not default; C was ALREADY in default, so it did
                      not default *during* the period)
        denominator = non-defaulted opening balance = A + B = £200
                      (C is excluded — it could not default again)
        periodic CDR = 100 / 200 = 50.0%

    Note what the denominator is NOT: £300, the whole book. Using it would give
    33.3% and would understate the rate on any portfolio carrying legacy
    defaults — the more defaults a book already has, the more it would flatter.
    """
    opening = _frame([("A", 100.0, "Performing", 0),
                      ("B", 100.0, "Performing", 0),
                      ("C", 100.0, "Defaulted", 120)])
    closing = _frame([("A", 100.0, "Performing", 0),
                      ("B", 100.0, "Defaulted", 95),
                      ("C", 100.0, "Defaulted", 150)])
    return {P1: opening, P2: closing}


def test_the_default_rate_follows_esmas_numerator_and_denominator():
    result = default_rate(_default_book(), periods=[P1, P2])
    assert result["available"] is True
    assert result["method"] == DEFAULT_RATE_METHOD

    period = result["per_period"][0]
    assert period["newly_defaulted_balance"] == pytest.approx(100.0)
    assert period["newly_defaulted_count"] == 1
    assert period["eligible_opening_balance"] == pytest.approx(200.0)
    assert period["periodic_default_rate_pct"] == pytest.approx(50.0)
    # The wrong denominator, named so a future change cannot drift onto it.
    assert period["periodic_default_rate_pct"] != pytest.approx(33.333333,
                                                                abs=1e-4)


def test_a_loan_already_in_default_is_in_neither_side():
    """C is in default at the opening and at the close. It contributes to the
    stock and to nothing else — counting it again every period would compound
    one default event for the life of the book."""
    result = default_rate(_default_book(), periods=[P1, P2])
    period = result["per_period"][0]
    assert period["eligible_opening_balance"] == pytest.approx(200.0)
    assert period["newly_defaulted_balance"] == pytest.approx(100.0)


def test_default_stock_and_default_rate_are_different_numbers():
    """"How much is in default" and "how much defaulted this period" are
    different questions, and a book can have a large stock with a zero rate."""
    result = default_rate(_default_book(), periods=[P1, P2])
    period = result["per_period"][0]
    assert period["default_stock_balance"] == pytest.approx(200.0)  # B and C
    assert period["default_stock_pct"] == pytest.approx(66.666667, abs=1e-4)
    assert period["periodic_default_rate_pct"] == pytest.approx(50.0)


def test_a_book_with_defaults_but_no_new_ones_has_a_zero_default_rate():
    """The clearest statement that this is a flow. Nothing defaulted, so the
    rate is 0% — while a third of the book sits in default."""
    opening = _frame([("A", 100.0, "Performing", 0),
                      ("C", 100.0, "Defaulted", 120)])
    closing = _frame([("A", 100.0, "Performing", 0),
                      ("C", 100.0, "Defaulted", 150)])
    period = default_rate({P1: opening, P2: closing},
                          periods=[P1, P2])["per_period"][0]
    assert period["periodic_default_rate_pct"] == pytest.approx(0.0)
    assert period["default_stock_pct"] == pytest.approx(50.0)


def test_the_annualisation_is_geometric_like_cpr():
    """1 - (1 - periodic)^12. On a 1% monthly rate that is 11.36%, not 12% —
    worked by hand. A linear 12x would overstate it."""
    opening = _frame([(f"L{i}", 100.0, "Performing", 0) for i in range(100)])
    closing_rows = [(f"L{i}", 100.0, "Performing", 0) for i in range(100)]
    closing_rows[0] = ("L0", 100.0, "Defaulted", 95)
    closing = _frame(closing_rows)

    period = default_rate({P1: opening, P2: closing},
                          periods=[P1, P2])["per_period"][0]
    assert period["periodic_default_rate_pct"] == pytest.approx(1.0)
    expected = (1 - (1 - 0.01) ** 12) * 100
    assert expected == pytest.approx(11.361513, abs=1e-5)
    assert period["annualised_cdr_pct"] == pytest.approx(expected, abs=1e-5)
    assert period["annualised_cdr_pct"] != pytest.approx(12.0, abs=1e-3)


def test_default_is_not_inferred_from_days_past_due():
    """A loan at 120 days with no default evidence is delinquent, not
    defaulted. CRR Article 178 needs materiality and consecutive days that a
    DPD column alone does not carry, and the securitisation documentation may
    define default differently again."""
    opening = _frame([("A", 100.0, "Performing", 0),
                      ("B", 100.0, "Performing", 0)])
    closing = _frame([("A", 100.0, "Performing", 0),
                      ("B", 100.0, "Performing", 120)])
    period = default_rate({P1: opening, P2: closing},
                          periods=[P1, P2])["per_period"][0]
    assert period["newly_defaulted_balance"] == pytest.approx(0.0)


def test_a_disappearing_loan_is_not_a_default():
    """Absence is not evidence. A loan that leaves may have redeemed, been sold
    or been written off, and only the last is a default."""
    opening = _frame([("A", 100.0, "Performing", 0),
                      ("B", 100.0, "Performing", 0)])
    closing = _frame([("A", 100.0, "Performing", 0)])
    period = default_rate({P1: opening, P2: closing},
                          periods=[P1, P2])["per_period"][0]
    assert period["newly_defaulted_balance"] == pytest.approx(0.0)


def test_one_snapshot_gives_no_default_rate_and_says_stock_is_different():
    result = default_rate({P1: _default_book()[P1]}, periods=[P1])
    assert result["available"] is False
    assert "at least two consecutive snapshots" in result["reason"]
    assert "STOCK" in result["reason"]


def test_a_tape_with_no_default_evidence_refuses():
    frame = pd.DataFrame({"loan_identifier": ["A"],
                          "current_principal_balance": [100.0],
                          "number_of_days_in_arrears": [0]})
    result = default_rate({P1: frame, P2: frame}, periods=[P1, P2])
    assert result["available"] is False
    assert "RREL72" in result["reason"] and "RREL69" in result["reason"]
    assert "not inferred from days past due" in result["reason"]


def test_the_result_names_the_authority_it_quotes():
    result = default_rate(_default_book(), periods=[P1, P2])
    assert "AnlsdCstDfltRate" in result["authority"]
    assert any("probability of default" in n for n in result["notes"]), (
        "an observed default rate must say it is not a PD")


# =========================================================================== #
# Cure rate — the brief's worked example
# =========================================================================== #
def _cure_book():
    """The brief's case:

        opening   A £100 current   B £100 90 DPD   C £100 60 DPD
        closing   A £100 current   B £100 CURRENT  C £100 30 DPD

    Eligible to cure = the delinquent opening balance = B + C = £200.
    A is current at the opening and cannot cure.

        cured        = B  = £100   ->  cure rate = 100/200 = 50.0%
        improved     = C  = £100   (60 -> 30 days: better, still delinquent)

    C is the case the methodology turns on. Counting it would give 100%.
    """
    opening = _frame([("A", 100.0, "Performing", 0),
                      ("B", 100.0, "Arrears", 90),
                      ("C", 100.0, "Arrears", 60)])
    closing = _frame([("A", 100.0, "Performing", 0),
                      ("B", 100.0, "Performing", 0),
                      ("C", 100.0, "Arrears", 30)])
    return {P1: opening, P2: closing}


def test_a_cure_is_a_return_to_current_and_improvement_is_not():
    result = cure_rate(_cure_book(), periods=[P1, P2])
    assert result["available"] is True
    assert result["method"] == CURE_RATE_METHOD

    period = result["per_period"][0]
    assert period["eligible_opening_balance"] == pytest.approx(200.0)
    assert period["cured_balance"] == pytest.approx(100.0)   # B only
    assert period["cured_count"] == 1
    assert period["cure_rate_pct"] == pytest.approx(50.0)
    # C improved from 60 to 30 days and is still delinquent.
    assert period["improved_not_cured_balance"] == pytest.approx(100.0)
    # Treating improvement as cure would give 100%. It does not.
    assert period["cure_rate_pct"] != pytest.approx(100.0)


def test_the_denominator_is_the_population_able_to_cure_not_the_book():
    """£200 of £300 was delinquent. Dividing by the whole book would give
    33.3% and would make the cure rate a function of how healthy the portfolio
    is rather than of how its delinquency resolved."""
    period = cure_rate(_cure_book(), periods=[P1, P2])["per_period"][0]
    assert period["eligible_opening_balance"] == pytest.approx(200.0)
    assert period["cure_rate_pct"] == pytest.approx(50.0)
    assert period["cure_rate_pct"] != pytest.approx(33.333333, abs=1e-4)


def test_a_loan_that_worsens_is_neither_cured_nor_improved():
    opening = _frame([("B", 100.0, "Arrears", 30)])
    closing = _frame([("B", 100.0, "Arrears", 90)])
    period = cure_rate({P1: opening, P2: closing},
                       periods=[P1, P2])["per_period"][0]
    assert period["cure_rate_pct"] == pytest.approx(0.0)
    assert period["worsened_balance"] == pytest.approx(100.0)
    assert period["improved_not_cured_balance"] == pytest.approx(0.0)


def test_a_loan_that_cures_then_redefaults_is_both_in_its_own_period():
    """Not one nets the other out. Over three periods B cures in the first and
    defaults in the second, and the series shows each where it happened —
    which is what lets a reviewer see churn rather than a flat zero."""
    opening = _frame([("B", 100.0, "Arrears", 90)])
    middle = _frame([("B", 100.0, "Performing", 0)])
    closing = _frame([("B", 100.0, "Defaulted", 95)])
    snapshots = {P1: opening, P2: middle, P3: closing}

    cures = cure_rate(snapshots, periods=[P1, P2, P3])["per_period"]
    assert cures[0]["cure_rate_pct"] == pytest.approx(100.0)
    # Period two: B is current at the opening, so it is not eligible to cure.
    assert cures[1]["eligible_opening_balance"] == pytest.approx(0.0)

    defaults = default_rate(snapshots, periods=[P1, P2, P3])["per_period"]
    assert defaults[0]["periodic_default_rate_pct"] == pytest.approx(0.0)
    assert defaults[1]["periodic_default_rate_pct"] == pytest.approx(100.0)


def test_a_delinquent_loan_that_disappears_is_excluded_and_disclosed():
    """Reading absence as a cure would flatter every book that sells its worst
    loans."""
    opening = _frame([("B", 100.0, "Arrears", 90),
                      ("C", 100.0, "Arrears", 60)])
    closing = _frame([("C", 100.0, "Performing", 0)])
    period = cure_rate({P1: opening, P2: closing},
                       periods=[P1, P2])["per_period"][0]

    assert period["exited_tape_balance"] == pytest.approx(100.0)
    assert period["cured_balance"] == pytest.approx(100.0)      # C only
    # B stays in the denominator: it was able to cure and its outcome is
    # unknown, so removing it would silently raise the rate.
    assert period["eligible_opening_balance"] == pytest.approx(200.0)
    assert period["cure_rate_pct"] == pytest.approx(50.0)


def test_a_loan_already_in_default_is_not_eligible_to_cure():
    """A write-back is a different event from a delinquency cure, and folding
    it in would let one book's write-back policy move a delinquency figure."""
    opening = _frame([("B", 100.0, "Defaulted", 120),
                      ("C", 100.0, "Arrears", 60)])
    closing = _frame([("B", 100.0, "Performing", 0),
                      ("C", 100.0, "Performing", 0)])
    period = cure_rate({P1: opening, P2: closing},
                       periods=[P1, P2])["per_period"][0]
    assert period["eligible_opening_balance"] == pytest.approx(100.0)  # C only
    assert period["cure_rate_pct"] == pytest.approx(100.0)


def test_a_delinquent_loan_that_defaults_is_reported_separately():
    opening = _frame([("B", 100.0, "Arrears", 60)])
    closing = _frame([("B", 100.0, "Defaulted", 95)])
    period = cure_rate({P1: opening, P2: closing},
                       periods=[P1, P2])["per_period"][0]
    assert period["defaulted_balance"] == pytest.approx(100.0)
    assert period["cure_rate_pct"] == pytest.approx(0.0)


def test_the_cure_result_admits_its_convention_is_not_a_regulators():
    """The asymmetry against the default rate, published rather than implied."""
    result = cure_rate(_cure_book(), periods=[P1, P2])
    assert "ESMA defines NO loan-level cure concept" in result["authority"]
    assert any("NOT quoted from a regulator" in n for n in result["notes"])


def test_cure_needs_days_past_due_and_says_so():
    frame = pd.DataFrame({"loan_identifier": ["A"],
                          "current_principal_balance": [100.0],
                          "account_status": ["Performing"]})
    result = cure_rate({P1: frame, P2: frame}, periods=[P1, P2])
    assert result["available"] is False
    assert "RREL68" in result["reason"]
