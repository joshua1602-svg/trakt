"""The decomposition, against canonical the pipeline actually produced.

``test_funded_composition`` states each frame outright — a handful of rows whose
arithmetic a reader can do in their head, which is the right way to pin a rule.
It is also the weaker evidence, because I wrote both the rule and the data it
runs on.

This runs the same code over ``synthetic_demo/output/multibook/`` — two
consecutive platform canonicals produced by the real pipeline, 116 and 118 loans
across three source portfolios, committed to the repository and authored by
nobody involved in this change.

Two things it catches that the stated fixtures cannot:

* **The prefix convention does not apply to this data.** The ids are
  ``alp_acquired``, ``alp_origination`` and ``spv1_sponsored``, so
  ``derive_portfolio_type`` returns ``None`` for every one of them and
  classification has to come from the ``source_portfolio_type`` column. That is
  the path a client whose ids follow their own naming will take, and it is
  exercised here on data that was never shaped to exercise it.
* **Reconciliation over a real distribution.** 115 held loans with capitalised
  interest, three arrivals and one exit, at full precision — where a rounding
  or masking error in the partition would show up as a residual and a
  four-row fixture would not.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from engine.provenance import derive_portfolio_type
from mi_agent_api import evolution as evolution_mod
from mi_agent_api import funded_composition as fc

_ROOT = Path(__file__).resolve().parents[2]
_MULTIBOOK = _ROOT / "synthetic_demo" / "output" / "multibook"
PRIOR_CSV = _MULTIBOOK / "platform_2026-05-31_canonical_typed.csv"
CURRENT_CSV = _MULTIBOOK / "platform_2026-06-30_canonical_typed.csv"

pytestmark = pytest.mark.skipif(
    not (PRIOR_CSV.exists() and CURRENT_CSV.exists()),
    reason="the committed multibook canonical pair is not present")

ACQUIRED_BOOK = "alp_acquired"


@pytest.fixture(scope="module")
def prior() -> pd.DataFrame:
    return pd.read_csv(PRIOR_CSV, low_memory=False)


@pytest.fixture(scope="module")
def current() -> pd.DataFrame:
    return pd.read_csv(CURRENT_CSV, low_memory=False)


# --------------------------------------------------------------------------- #
# An ordinary month, at full precision
# --------------------------------------------------------------------------- #
def test_the_partition_reconciles_over_real_canonical(current, prior):
    """Every pound of both frames lands in exactly one component.

    Asserted at full precision rather than to a tolerance: the components sum to
    the movement by construction, so a residual here means the partition stopped
    partitioning, not that the arithmetic drifted.
    """
    out = fc.decompose(current, prior)

    assert out["available"] is True
    assert out["unavailable"] == {}
    stated = [v for v in out["components"].values() if v is not None]
    assert sum(stated) == pytest.approx(out["movement"], abs=0.01)
    assert out["reconciliation"]["reconciles"] is True
    assert out["reconciliation"]["residual"] == 0.0


def test_the_real_month_is_organic_and_is_reported_as_such(current, prior):
    """No portfolio arrived, so no addition is reported however the book moved."""
    out = fc.decompose(current, prior)

    assert out["portfolio_additions"] == []
    assert out["portfolio_disposals"] == []
    assert out["components"]["portfolio_additions"] == 0.0
    assert fc.dominant_addition(out) is None
    # And the loan-level split is real: arrivals, an exit, and a held book.
    assert out["counts"]["new_loans"] == 3
    assert out["counts"]["exited_loans"] == 1
    assert out["counts"]["held_loans"] == 115


def test_capitalised_interest_lands_in_existing_book_movement(current, prior):
    """A roll-up book grows without lending, and that is a distinct component.

    Reported as existing-book movement rather than as new lending — the
    distinction the equity-release simulation case exists to protect.
    """
    out = fc.decompose(current, prior)
    assert out["components"]["existing_book_movement"] > 0
    assert out["components"]["organic_new_lending"] > 0
    assert out["components"]["exits"] < 0


# --------------------------------------------------------------------------- #
# Classification, where the prefix convention does not apply
# --------------------------------------------------------------------------- #
def test_no_real_id_here_carries_the_trakt_prefix(current):
    """The premise of the next test, asserted rather than assumed."""
    ids = sorted(current["source_portfolio_id"].dropna().unique())
    assert ids == ["alp_acquired", "alp_origination", "spv1_sponsored"]
    assert all(derive_portfolio_type(pid) is None for pid in ids)


def test_the_stated_type_column_classifies_what_the_prefix_cannot(current):
    """A client naming its own books is fully supported.

    The column is the primary authority and the id prefix only the fallback, so
    a deployment that never adopted the ``direct_`` / ``acquired_`` convention
    still gets correct provenance.
    """
    assert fc.classify_portfolio(current, "alp_acquired") == fc.TYPE_ACQUIRED
    assert fc.classify_portfolio(current, "alp_origination") == fc.TYPE_DIRECT
    assert fc.classify_portfolio(current, "spv1_sponsored") == fc.TYPE_DIRECT


# --------------------------------------------------------------------------- #
# An acquisition month, built from the real data
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def acquisition(current, prior):
    """The month the acquired book arrived.

    Constructed by REMOVING that portfolio from the prior period rather than by
    inventing one: every loan, balance and date in the arriving book is the
    pipeline's own output.
    """
    return fc.decompose(
        current, prior[prior["source_portfolio_id"] != ACQUIRED_BOOK])


def test_a_real_arriving_book_is_detected_from_identity(acquisition):
    added = acquisition["portfolio_additions"]
    assert len(added) == 1
    assert added[0]["source_portfolio_id"] == ACQUIRED_BOOK
    assert added[0]["portfolio_type"] == fc.TYPE_ACQUIRED
    assert added[0]["loan_count"] == 37
    # Read from the canonical, not supplied by the test.
    assert added[0]["acquisition_date"] == "2024-09-30"


def test_the_acquisition_month_still_reconciles(acquisition):
    stated = [v for v in acquisition["components"].values() if v is not None]
    assert sum(stated) == pytest.approx(acquisition["movement"], abs=0.01)
    assert acquisition["reconciliation"]["reconciles"] is True


def test_the_arriving_book_dominates_the_month(acquisition):
    lead = fc.dominant_addition(acquisition)
    assert lead is not None
    assert lead["source_portfolio_id"] == ACQUIRED_BOOK
    # ~£11.97m of a ~£12.83m movement.
    assert lead["share_of_movement"] == pytest.approx(0.93, abs=0.01)


def test_the_underlying_book_is_the_incumbent_portfolios(acquisition, current,
                                                         prior):
    """+52% at the headline, +3.5% underneath. Both true, and both stated."""
    filters = fc.underlying_lens_filters(acquisition)
    assert filters == {"source_portfolio_id": ["alp_origination",
                                               "spv1_sponsored"]}

    acq_prior = prior[prior["source_portfolio_id"] != ACQUIRED_BOOK]
    underlying = fc.decompose(
        evolution_mod._scope_frame_lens(current, filters),
        evolution_mod._scope_frame_lens(acq_prior, filters))

    headline_pct = acquisition["movement"] / acquisition["opening_balance"]
    underlying_pct = underlying["movement"] / underlying["opening_balance"]

    assert headline_pct == pytest.approx(0.52, abs=0.02)
    assert underlying_pct == pytest.approx(0.035, abs=0.005)
    assert underlying["components"]["portfolio_additions"] == 0.0
    assert underlying["reconciliation"]["reconciles"] is True
