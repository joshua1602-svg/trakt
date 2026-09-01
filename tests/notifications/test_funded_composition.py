"""Why the funded book moved — and never calling a balance jump an acquisition.

Five months, each a different shape: ordinary organic growth, a repayment-heavy
month, an acquisition month, a mixed month, and the underlying book read with the
acquisition excluded. Every expected figure below is stated as a literal and
computed by hand from the fixture, never by calling the code under test.

Portfolio ids here are deliberately ``portfolio_alpha`` / ``portfolio_beta`` /
``portfolio_gamma`` — names no production module knows. A rule that only works
for ``acquired_001`` would fail this file, which is the point of it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent_api import funded_composition as fc

BALANCE = "current_outstanding_balance"
LOAN = "loan_identifier"
PORTFOLIO = "source_portfolio_id"


def _frame(rows) -> pd.DataFrame:
    """A prepared funded frame: loan, balance, governed portfolio identity."""
    return pd.DataFrame([
        {LOAN: loan, BALANCE: bal, PORTFOLIO: pid,
         "source_portfolio_label": label, **extra}
        for loan, bal, pid, label, extra in rows])


def _p(loan, bal, pid, label="Book", **extra):
    return (loan, bal, pid, label, extra)


# --------------------------------------------------------------------------- #
# 1. An ordinary organic month
# --------------------------------------------------------------------------- #
def test_an_organic_month_attributes_growth_to_new_lending_and_accretion():
    """No portfolio changed hands. Growth is new loans plus accretion.

        opening   L1 100 + L2 200            = 300
        closing   L1 110 + L2 200 + L3  50   = 360
        organic   L3                         = +50
        existing  L1 110-100                 = +10
    """
    prior = _frame([_p("L1", 100.0, "portfolio_alpha"),
                    _p("L2", 200.0, "portfolio_alpha")])
    current = _frame([_p("L1", 110.0, "portfolio_alpha"),
                      _p("L2", 200.0, "portfolio_alpha"),
                      _p("L3", 50.0, "portfolio_alpha")])

    out = fc.decompose(current, prior)

    assert out["opening_balance"] == 300.0
    assert out["closing_balance"] == 360.0
    assert out["movement"] == 60.0
    assert out["components"] == {
        "portfolio_additions": 0.0, "portfolio_disposals": 0.0,
        "organic_new_lending": 50.0, "exits": 0.0,
        "existing_book_movement": 10.0,
    }
    assert out["portfolio_additions"] == []
    assert out["reconciliation"]["reconciles"] is True
    assert out["counts"] == {"new_loans": 1, "exited_loans": 0, "held_loans": 2}


# --------------------------------------------------------------------------- #
# 2. A repayment-heavy month
# --------------------------------------------------------------------------- #
def test_a_repayment_heavy_month_separates_exits_from_accretion():
    """Loans leaving is a different fact from balances falling.

        opening   L1 100 + L2 200 + L3 300 = 600
        closing   L1  95                   =  95
        exits     L2 + L3                  = -500
        existing  L1 95-100                =  -5
    """
    prior = _frame([_p("L1", 100.0, "portfolio_alpha"),
                    _p("L2", 200.0, "portfolio_alpha"),
                    _p("L3", 300.0, "portfolio_alpha")])
    current = _frame([_p("L1", 95.0, "portfolio_alpha")])

    out = fc.decompose(current, prior)

    assert out["movement"] == -505.0
    assert out["components"]["exits"] == -500.0
    assert out["components"]["existing_book_movement"] == -5.0
    assert out["components"]["organic_new_lending"] == 0.0
    assert out["reconciliation"]["reconciles"] is True
    assert out["counts"]["exited_loans"] == 2


# --------------------------------------------------------------------------- #
# 3. An acquisition month
# --------------------------------------------------------------------------- #
ACQ_PRIOR = _frame([
    _p("A1", 60_000_000.0, "portfolio_alpha", "Direct Book"),
    _p("A2", 52_000_000.0, "portfolio_alpha", "Direct Book"),
])
ACQ_CURRENT = _frame([
    _p("A1", 61_000_000.0, "portfolio_alpha", "Direct Book"),
    _p("A2", 52_000_000.0, "portfolio_alpha", "Direct Book"),
    _p("A3", 3_000_000.0, "portfolio_alpha", "Direct Book"),
    _p("B1", 40_000_000.0, "acquired_portfolio_beta", "Portfolio B",
       source_portfolio_type="acquired", acquisition_date="2026-07-15"),
    _p("B2", 28_000_000.0, "acquired_portfolio_beta", "Portfolio B",
       source_portfolio_type="acquired", acquisition_date="2026-07-15"),
])


def test_an_acquisition_month_is_not_reported_as_organic_growth():
    """The month the brief has to get right.

        opening  60 + 52                    = 112m
        closing  61 + 52 + 3 + 40 + 28      = 184m   (+72m)
        addition portfolio_beta 40 + 28     = +68m
        organic  A3                         =  +3m
        existing A1 61-60                   =  +1m
    """
    out = fc.decompose(ACQ_CURRENT, ACQ_PRIOR)

    assert out["opening_balance"] == 112_000_000.0
    assert out["closing_balance"] == 184_000_000.0
    assert out["movement"] == 72_000_000.0
    assert out["components"]["portfolio_additions"] == 68_000_000.0
    assert out["components"]["organic_new_lending"] == 3_000_000.0
    assert out["components"]["existing_book_movement"] == 1_000_000.0
    assert out["reconciliation"]["reconciles"] is True

    added = out["portfolio_additions"]
    assert len(added) == 1
    assert added[0]["source_portfolio_id"] == "acquired_portfolio_beta"
    assert added[0]["label"] == "Portfolio B"
    assert added[0]["portfolio_type"] == fc.TYPE_ACQUIRED
    assert added[0]["acquisition_date"] == "2026-07-15"
    assert added[0]["loan_count"] == 2


def test_the_dominant_addition_is_identified_by_share_of_movement():
    lead = fc.dominant_addition(fc.decompose(ACQ_CURRENT, ACQ_PRIOR))
    assert lead is not None
    assert lead["source_portfolio_id"] == "acquired_portfolio_beta"
    # 68 of 72.
    assert lead["share_of_movement"] == pytest.approx(0.9444, abs=1e-4)


def test_a_small_addition_does_not_dominate_a_month():
    """The test is share of the movement, not "an addition happened"."""
    prior = _frame([_p("L1", 100.0, "portfolio_alpha")])
    current = _frame([_p("L1", 100.0, "portfolio_alpha"),
                      _p("L2", 900.0, "portfolio_alpha"),
                      _p("N1", 10.0, "portfolio_gamma")])
    assert fc.dominant_addition(fc.decompose(current, prior)) is None


# --------------------------------------------------------------------------- #
# The rule that matters most
# --------------------------------------------------------------------------- #
def test_a_balance_jump_alone_is_never_called_an_acquisition():
    """A book that doubled inside ONE portfolio produces no addition at all.

    This is the inference the whole module exists to refuse. Nothing about the
    size of the movement may create a portfolio addition; only identity does.
    """
    prior = _frame([_p("L1", 100_000_000.0, "portfolio_alpha")])
    current = _frame([_p("L1", 100_000_000.0, "portfolio_alpha"),
                      _p("L2", 100_000_000.0, "portfolio_alpha")])

    out = fc.decompose(current, prior)

    assert out["movement"] == 100_000_000.0
    assert out["portfolio_additions"] == []
    assert out["components"]["portfolio_additions"] == 0.0
    assert out["components"]["organic_new_lending"] == 100_000_000.0
    assert fc.dominant_addition(out) is None


def test_an_addition_identity_cannot_classify_is_not_called_acquired():
    """No explicit type, no governed prefix — so it is a new source portfolio.

    Reported, never guessed: an unclassified addition is still a real fact about
    the book, and the reader is entitled to see it without being told a
    provenance nobody asserted.
    """
    prior = _frame([_p("L1", 100.0, "portfolio_alpha")])
    current = _frame([_p("L1", 100.0, "portfolio_alpha"),
                      _p("N1", 500.0, "portfolio_gamma")])

    added = fc.decompose(current, prior)["portfolio_additions"]
    assert len(added) == 1
    assert added[0]["portfolio_type"] == fc.TYPE_UNCLASSIFIED


def test_an_explicit_type_column_beats_the_id_prefix():
    """Identity the rows assert outranks identity the id encodes.

    A client whose ids carry no Trakt prefix is fully supported, because the
    column is the primary authority and the prefix only the fallback.
    """
    prior = _frame([_p("L1", 100.0, "portfolio_alpha")])
    current = _frame([_p("L1", 100.0, "portfolio_alpha"),
                      _p("N1", 500.0, "lender_book_17",
                         source_portfolio_type="acquired")])

    added = fc.decompose(current, prior)["portfolio_additions"]
    assert added[0]["portfolio_type"] == fc.TYPE_ACQUIRED


def test_a_disposed_portfolio_is_reported_and_reconciles():
    """A book leaving is as much a fact as a book arriving.

    ``period_movement`` iterates the CURRENT frame's cohorts, so a departed
    portfolio lands in its residual. Here it is a named component.
    """
    prior = _frame([_p("L1", 100.0, "portfolio_alpha"),
                    _p("X1", 400.0, "portfolio_gamma")])
    current = _frame([_p("L1", 120.0, "portfolio_alpha")])

    out = fc.decompose(current, prior)

    assert out["components"]["portfolio_disposals"] == -400.0
    assert out["components"]["existing_book_movement"] == 20.0
    assert out["movement"] == -380.0
    assert out["reconciliation"]["reconciles"] is True
    assert out["portfolio_disposals"][0]["source_portfolio_id"] == "portfolio_gamma"


# --------------------------------------------------------------------------- #
# 4. Mixed acquisition + organic, and the underlying lens
# --------------------------------------------------------------------------- #
def test_the_underlying_lens_reuses_the_existing_population_mechanism():
    """"Excluding the acquisition" is the existing lens over continuing ids."""
    out = fc.decompose(ACQ_CURRENT, ACQ_PRIOR)
    filters = fc.underlying_lens_filters(out)

    assert filters == {"source_portfolio_id": ["portfolio_alpha"]}

    # And it narrows through the SAME function every other lens uses.
    from mi_agent_api import evolution as evolution_mod
    narrowed = evolution_mod._scope_frame_lens(ACQ_CURRENT, filters)
    assert set(narrowed[LOAN]) == {"A1", "A2", "A3"}


def test_the_underlying_book_movement_excludes_the_acquisition_entirely():
    """The incumbent book grew £4m; the headline says £72m. Both are true.

    A £68m acquisition must not be able to hide a deterioration, or manufacture
    an improvement, in the book that was already there.
    """
    out = fc.decompose(ACQ_CURRENT, ACQ_PRIOR)
    filters = fc.underlying_lens_filters(out)

    from mi_agent_api import evolution as evolution_mod
    underlying = fc.decompose(
        evolution_mod._scope_frame_lens(ACQ_CURRENT, filters),
        evolution_mod._scope_frame_lens(ACQ_PRIOR, filters))

    assert underlying["movement"] == 4_000_000.0
    assert underlying["components"]["portfolio_additions"] == 0.0
    assert underlying["components"]["organic_new_lending"] == 3_000_000.0
    assert underlying["components"]["existing_book_movement"] == 1_000_000.0
    # 4m on 112m.
    assert underlying["movement"] / underlying["opening_balance"] == \
        pytest.approx(0.0357, abs=1e-4)


def test_no_addition_means_no_underlying_lens():
    """A Total answer may never be presented as an underlying-book answer."""
    prior = _frame([_p("L1", 100.0, "portfolio_alpha")])
    current = _frame([_p("L1", 150.0, "portfolio_alpha")])
    assert fc.underlying_lens_filters(fc.decompose(current, prior)) is None


# --------------------------------------------------------------------------- #
# 5. Scale — adding a portfolio is data, not a code change
# --------------------------------------------------------------------------- #
def test_three_generic_portfolios_decompose_with_no_production_change():
    """Two incumbents and two arrivals, none of them named in any module.

        opening  P1 100 + P2 200                    = 300
        closing  P1 100 + P2 210 + P3 500 + P4 700  = 1510
        additions   500 + 700                       = +1200
        existing    P2 210-200                      = +10
    """
    prior = _frame([_p("L1", 100.0, "portfolio_alpha"),
                    _p("L2", 200.0, "portfolio_beta")])
    current = _frame([
        _p("L1", 100.0, "portfolio_alpha"),
        _p("L2", 210.0, "portfolio_beta"),
        _p("L3", 500.0, "portfolio_gamma", source_portfolio_type="acquired"),
        _p("L4", 700.0, "portfolio_delta", source_portfolio_type="acquired"),
    ])

    out = fc.decompose(current, prior)

    assert out["movement"] == 1210.0
    assert out["components"]["portfolio_additions"] == 1200.0
    assert out["components"]["existing_book_movement"] == 10.0
    assert [p["source_portfolio_id"] for p in out["portfolio_additions"]] == \
        ["portfolio_delta", "portfolio_gamma"]
    assert all(p["portfolio_type"] == fc.TYPE_ACQUIRED
               for p in out["portfolio_additions"])
    assert out["reconciliation"]["reconciles"] is True
    assert fc.underlying_lens_filters(out) == {
        "source_portfolio_id": ["portfolio_alpha", "portfolio_beta"]}


# --------------------------------------------------------------------------- #
# Degradation — the two granularities fail independently
# --------------------------------------------------------------------------- #
def test_without_portfolio_identity_the_acquisition_split_is_refused():
    """No source_portfolio_id: the split is named as unavailable, not guessed."""
    prior = pd.DataFrame([{LOAN: "L1", BALANCE: 100.0}])
    current = pd.DataFrame([{LOAN: "L1", BALANCE: 100.0},
                            {LOAN: "L2", BALANCE: 900.0}])

    out = fc.decompose(current, prior)

    assert PORTFOLIO in out["unavailable"]
    assert out["portfolio_additions"] == []
    # The loan-level split still works and still reconciles.
    assert out["components"]["organic_new_lending"] == 900.0
    assert out["reconciliation"]["reconciles"] is True


def test_without_a_loan_identifier_the_continuing_book_is_reported_whole():
    """No loan key: new lending and exits are named unavailable, not apportioned."""
    prior = pd.DataFrame([{BALANCE: 100.0, PORTFOLIO: "portfolio_alpha"}])
    current = pd.DataFrame([{BALANCE: 100.0, PORTFOLIO: "portfolio_alpha"},
                            {BALANCE: 500.0, PORTFOLIO: "portfolio_gamma"}])

    out = fc.decompose(current, prior)

    assert "loan_identifier" in out["unavailable"]
    assert out["components"]["organic_new_lending"] is None
    assert out["components"]["exits"] is None
    # Portfolio identity still separates the addition from the incumbent book.
    assert out["components"]["portfolio_additions"] == 500.0
    assert out["components"]["existing_book_movement"] == 0.0
    assert out["reconciliation"]["reconciles"] is True


# --------------------------------------------------------------------------- #
# Narrative selection
# --------------------------------------------------------------------------- #
def test_immaterial_components_are_not_named():
    """A £1k exit does not stand beside a £68m addition in a sentence."""
    prior = _frame([_p("L1", 100_000_000.0, "portfolio_alpha"),
                    _p("L9", 1_000.0, "portfolio_alpha")])
    current = _frame([_p("L1", 100_000_000.0, "portfolio_alpha"),
                      _p("N1", 68_000_000.0, "portfolio_gamma",
                         source_portfolio_type="acquired")])

    named = fc.narrative_components(fc.decompose(current, prior))

    assert [r["component"] for r in named] == ["portfolio_additions"]
    # The full decomposition still carries it.
    assert fc.decompose(current, prior)["components"]["exits"] == -1_000.0
