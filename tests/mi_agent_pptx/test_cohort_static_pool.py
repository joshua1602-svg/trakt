"""Does the governed cohort service hold the pool fixed at formation?

The static-pool contract is precise:

  * a loan belongs to one vintage for life;
  * the pool is fixed at formation;
  * the surviving count can hold or fall;
  * the surviving count can NEVER rise.

These tests assert that against actual loan identifiers rather than against
counts alone, because a count can hold while the membership changes underneath
it — which is exactly what a first, weaker version of this test failed to catch.

The answer matters to the deck: every retention and exit figure on the seasoning
slide is computed over the pool, so if the pool moves those figures are computed
over a moving population and mean nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mi_agent_pptx import cohorts as CO

CENTRAL = "18_central_lender_tape.csv"


def _loan(uid, bal, cut, orig, ltv=45.0):
    return {"unique_identifier": uid, "source_portfolio_id": "direct_001",
            "source_portfolio_type": "direct",
            "current_outstanding_balance": bal, "current_principal_balance": bal,
            "current_valuation_amount": bal / 0.45, "current_loan_to_value": ltv,
            "current_interest_rate": 7.0, "youngest_borrower_age": 72,
            "collateral_geography": "London", "geographic_region_obligor": "London",
            "origination_date": orig, "data_cut_off_date": cut}


def _write(root, run_id, cut, rows):
    central = root / "acme" / run_id / "central"
    central.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(central / CENTRAL, index=False)
    dated = root / cut
    dated.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dated / "platform_canonical_typed.csv", index=False)


def _progression(root, vintage="2020"):
    from mi_agent_api import evolution
    return evolution.funded_cohort_progression(str(root), "acme", vintage=vintage,
                                               grain="Y")


# --------------------------------------------------------------------------- #
# The membership rule, proven on identifiers.
# --------------------------------------------------------------------------- #

def test_a_late_boarded_loan_does_not_enter_an_already_formed_pool(tmp_path):
    """THE root-cause test, now asserting the fixed contract.

    Period 2 keeps both formation loans AND boards a third carrying a 2020
    origination date. Under the previous per-period re-filter the service
    reported 3 — the late arrival joined its vintage. Membership is now frozen
    at the formation cut, so it reports 2 in both periods and the newcomer is
    excluded by identifier.
    """
    root = tmp_path / "root"
    _write(root, "mi_2026_05", "2026-05-31",
           [_loan("L1", 100_000.0, "2026-05-31", "2020-03-01"),
            _loan("L2", 100_000.0, "2026-05-31", "2020-04-01")])
    _write(root, "mi_2026_06", "2026-06-30",
           [_loan("L1", 100_000.0, "2026-06-30", "2020-03-01"),
            _loan("L2", 100_000.0, "2026-06-30", "2020-04-01"),
            _loan("L3", 100_000.0, "2026-06-30", "2020-09-01")])   # boarded late

    payload = _progression(root)
    counts = [p["loanCount"] for p in payload["periods"]]
    assert counts == [2, 2], "a late-boarded loan entered an already-formed pool"
    assert payload["membershipRule"] == "fixed_at_formation"
    assert payload["formationLoanIds"] == ["L1", "L2"]
    assert payload["formationLoanCount"] == 2
    # The newcomer is absent from every surviving set, by identifier.
    for period in payload["periods"]:
        assert "L3" not in period["survivingLoanIds"]
        assert set(period["survivingLoanIds"]) <= set(payload["formationLoanIds"])


def test_a_closed_pool_is_reported_as_a_closed_pool(tmp_path):
    """The same service on a book where nothing joins: counts fall, and the
    surviving identifiers are a strict subset of the formation identifiers."""
    root = tmp_path / "root"
    formation = ["A", "B", "C", "D"]
    _write(root, "mi_2026_05", "2026-05-31",
           [_loan(u, 100_000.0, "2026-05-31", "2020-03-01") for u in formation])
    survivors = ["A", "C", "D"]
    _write(root, "mi_2026_06", "2026-06-30",
           [_loan(u, 100_000.0, "2026-06-30", "2020-03-01") for u in survivors])

    series = CO.adapt_progression(_progression(root), "2020")
    assert set(survivors) < set(formation), "the fixture is not a closed pool"
    assert [p.loan_count for p in series.live] == [4, 3]
    assert series.violates_static_pool is False
    assert series.formation_count == 4 and series.surviving_count == 3
    assert series.exits == 1
    assert series.retention("loan_count") == pytest.approx(75.0)
    # And the contract carries the identifiers, not just the counts.
    payload = _progression(root)
    assert payload["formationLoanIds"] == sorted(formation)
    assert payload["periods"][-1]["survivingLoanIds"] == sorted(survivors)
    assert payload["periods"][-1]["exitsCount"] == 1
    assert payload["periods"][-1]["loanRetention"] == pytest.approx(75.0)


# --------------------------------------------------------------------------- #
# What the deck does about it.
# --------------------------------------------------------------------------- #

def test_the_service_can_no_longer_produce_a_rising_cohort(tmp_path):
    """The case the deck used to have to suppress now cannot arise upstream."""
    root = tmp_path / "root"
    _write(root, "mi_2026_05", "2026-05-31",
           [_loan("L1", 100_000.0, "2026-05-31", "2020-03-01")])
    _write(root, "mi_2026_06", "2026-06-30",
           [_loan("L1", 100_000.0, "2026-06-30", "2020-03-01"),
            _loan("L2", 100_000.0, "2026-06-30", "2020-09-01")])

    series = CO.adapt_progression(_progression(root), "2020")
    assert series.violates_static_pool is False
    assert [p.loan_count for p in series.live] == [1, 1]
    assert CO.plottable([series]) == [series]


def test_the_publication_gate_blocks_a_plotted_rising_cohort():
    """The backstop: if a rising cohort ever reaches a slide, the deck must not
    publish, because every retention and exit figure beside it would be computed
    over a moving population."""
    from types import SimpleNamespace

    from mi_agent_pptx.preflight import _gate_cohort_static_pool_integrity

    rising = {"available": True, "lens": "Total", "metricsAvailable": [],
              "periods": [{"period": "2026-05", "loanCount": 1,
                           "metrics": {"funded_balance": 100.0}},
                          {"period": "2026-06", "loanCount": 2,
                           "metrics": {"funded_balance": 200.0}}]}
    data = SimpleNamespace(cohort_series={"available": True,
                                          "series": {"2020": rising}})
    # Excluded by selection, so the gate passes and the slide simply omits it.
    assert _gate_cohort_static_pool_integrity(data).passed is True

    # Force it past selection: the gate is what stands between it and a reader.
    import mi_agent_pptx.cohorts as C
    original = C.plottable
    try:
        C.plottable = lambda series: list(series)
        gate = _gate_cohort_static_pool_integrity(data)
    finally:
        C.plottable = original
    assert gate.passed is False and gate.mandatory is True
    assert "gain loans" in gate.detail


def test_the_deck_makes_no_claim_that_a_cohort_can_gain_loans():
    """The wording that described the service's re-filtering must not survive:
    the deck now excludes those cohorts, so describing them would explain a case
    the reader can never see."""
    import mi_agent_pptx

    root = Path(mi_agent_pptx.__file__).parent
    for name in ("deck.py",):
        body = "\n".join(l for l in (root / name).read_text(encoding="utf-8").splitlines()
                         if not l.strip().startswith("#"))
        assert "can also gain balance" not in body
        assert "boards a loan of that vintage" not in body


# --------------------------------------------------------------------------- #
# Cross-channel parity: React, PPTX and the chat/Copilot path all call ONE
# function, so parity is structural — but assert it, because "they call the same
# thing" has been true of this codebase before while the callers differed.
# --------------------------------------------------------------------------- #

def test_every_channel_reaches_the_same_service(tmp_path):
    import inspect

    from mi_agent_api import app as react_app, chat_routing
    from mi_agent_pptx import mi_api as pptx_api

    for module, label in ((react_app, "React route"),
                          (chat_routing, "chat / Copilot"),
                          (pptx_api, "PPTX")):
        src = inspect.getsource(module)
        assert "funded_cohort_progression" in src, f"{label} no longer calls it"


def test_the_contract_is_versioned_and_identical_across_channels(tmp_path):
    """Same tenant, scope, vintage and reporting date → identical payload."""
    from mi_agent_api import evolution
    from mi_agent_api.evolution import COHORT_METHODOLOGY_VERSION

    root = tmp_path / "root"
    _write(root, "mi_2026_05", "2026-05-31",
           [_loan(u, 100_000.0, "2026-05-31", "2020-03-01") for u in "ABCD"])
    _write(root, "mi_2026_06", "2026-06-30",
           [_loan(u, 101_000.0, "2026-06-30", "2020-03-01") for u in "ACD"])

    calls = [evolution.funded_cohort_progression(str(root), "acme", vintage="2020",
                                                 grain="Y") for _ in range(3)]
    react, copilot, pptx = calls
    for field in ("methodologyVersion", "membershipRule", "cohortId",
                  "formationDate", "formationLoanIds", "formationLoanCount",
                  "formationBalance"):
        assert react[field] == copilot[field] == pptx[field], field
    assert react["methodologyVersion"] == COHORT_METHODOLOGY_VERSION

    for a, b, c in zip(react["periods"], copilot["periods"], pptx["periods"]):
        for field in ("reporting_date", "survivingLoanIds", "survivingLoanCount",
                      "survivingBalance", "exitsCount", "loanRetention",
                      "balanceRetention", "seasoningPeriods"):
            assert a[field] == b[field] == c[field], field
        assert a["metrics"]["wa_ltv"] == b["metrics"]["wa_ltv"] == c["metrics"]["wa_ltv"]

    last = react["periods"][-1]
    assert last["survivingLoanIds"] == ["A", "C", "D"]
    assert last["exitsCount"] == 1
    assert last["loanRetention"] == pytest.approx(75.0)
    # Contract item 7: balance can outrun loan count because balances on the
    # SURVIVING loans grew, never because a new loan entered.
    assert last["balanceRetention"] > last["loanRetention"]
    assert last["seasoningPeriods"] == 1


def test_the_ltv_unit_is_inferred_from_the_period_not_the_cohort(tmp_path):
    """A deeply seasoned roll-up cohort must not be rendered 100x too small."""
    import pandas as pd

    from mi_agent_api.evolution import _pct_fraction

    period = pd.DataFrame({"current_loan_to_value": [0.44, 0.46, 1.60, 1.90],
                           "current_outstanding_balance": [1e5] * 4})
    cohort = period.iloc[2:]                       # only the >150% LTV loans
    assert _pct_fraction(cohort, "current_loan_to_value") == pytest.approx(0.0175)
    assert _pct_fraction(cohort, "current_loan_to_value",
                         period) == pytest.approx(1.75)


# --------------------------------------------------------------------------- #
# The remaining named regression cases.
# --------------------------------------------------------------------------- #

def test_a_corrected_later_upload_replaces_that_period_only(tmp_path):
    """Contract item 9: a correction to a LATER period changes that period's
    observation and leaves the formation rule alone."""
    root = tmp_path / "root"
    _write(root, "mi_2026_05", "2026-05-31",
           [_loan(u, 100_000.0, "2026-05-31", "2020-03-01") for u in "ABC"])
    _write(root, "mi_2026_06", "2026-06-30",
           [_loan(u, 100_000.0, "2026-06-30", "2020-03-01") for u in "AB"])
    before = _progression(root)

    # The June upload is corrected: B was reported redeemed in error.
    _write(root, "mi_2026_06", "2026-06-30",
           [_loan(u, 100_000.0, "2026-06-30", "2020-03-01") for u in "ABC"])
    after = _progression(root)

    assert after["formationDate"] == before["formationDate"]
    assert after["formationLoanIds"] == before["formationLoanIds"] == ["A", "B", "C"]
    assert before["periods"][-1]["survivingLoanIds"] == ["A", "B"]
    assert after["periods"][-1]["survivingLoanIds"] == ["A", "B", "C"]
    assert after["periods"][-1]["exitsCount"] == 0


def test_a_corrected_formation_snapshot_reforms_the_pool(tmp_path):
    """Contract item 9, second half: correcting the FORMATION cut re-forms it,
    because the formation cut is derived from the frames rather than stored."""
    root = tmp_path / "root"
    _write(root, "mi_2026_05", "2026-05-31",
           [_loan(u, 100_000.0, "2026-05-31", "2020-03-01") for u in "AB"])
    _write(root, "mi_2026_06", "2026-06-30",
           [_loan(u, 100_000.0, "2026-06-30", "2020-03-01") for u in "ABC"])
    assert _progression(root)["formationLoanIds"] == ["A", "B"]

    # C was in the book at formation and had been omitted in error.
    _write(root, "mi_2026_05", "2026-05-31",
           [_loan(u, 100_000.0, "2026-05-31", "2020-03-01") for u in "ABC"])
    reformed = _progression(root)
    assert reformed["formationLoanIds"] == ["A", "B", "C"]
    assert reformed["periods"][-1]["survivingLoanCount"] == 3


def test_a_backdated_upload_before_formation_re_forms_at_the_earlier_cut(tmp_path):
    root = tmp_path / "root"
    _write(root, "mi_2026_06", "2026-06-30",
           [_loan(u, 100_000.0, "2026-06-30", "2020-03-01") for u in "AB"])
    assert _progression(root)["formationDate"] == "2026-06-30"

    _write(root, "mi_2026_05", "2026-05-31",
           [_loan(u, 100_000.0, "2026-05-31", "2020-03-01") for u in "ABC"])
    later = _progression(root)
    assert later["formationDate"] == "2026-05-31"
    assert later["formationLoanCount"] == 3
    assert later["periods"][-1]["exitsCount"] == 1          # C is gone by June


def test_duplicate_formation_identifiers_are_flagged_not_hidden(tmp_path):
    root = tmp_path / "root"
    _write(root, "mi_2026_05", "2026-05-31",
           [_loan("A", 100_000.0, "2026-05-31", "2020-03-01"),
            _loan("A", 100_000.0, "2026-05-31", "2020-03-01"),
            _loan("B", 100_000.0, "2026-05-31", "2020-03-01")])
    _write(root, "mi_2026_06", "2026-06-30",
           [_loan("A", 100_000.0, "2026-06-30", "2020-03-01")])
    payload = _progression(root)
    assert payload["dataQuality"].get("duplicate_formation_ids") == 1
    assert payload["formationLoanIds"] == ["A", "B"]


def test_a_tape_without_a_loan_identifier_declines_to_form_a_pool(tmp_path):
    """No identifier means no fixed membership, so the service says so rather
    than returning a series that only looks like a static pool."""
    import pandas as pd

    root = tmp_path / "root"
    for run_id, cut in (("mi_2026_05", "2026-05-31"), ("mi_2026_06", "2026-06-30")):
        rows = [_loan("X", 100_000.0, cut, "2020-03-01")]
        frame = pd.DataFrame(rows).drop(columns=["unique_identifier"])
        central = root / "acme" / run_id / "central"
        central.mkdir(parents=True, exist_ok=True)
        frame.to_csv(central / CENTRAL, index=False)
        dated = root / cut
        dated.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dated / "platform_canonical_typed.csv", index=False)

    payload = _progression(root)
    assert payload["available"] is False
    assert "loan identifier" in (payload["reason"] or "")


def test_no_prior_period_forms_but_does_not_season(tmp_path):
    root = tmp_path / "root"
    _write(root, "mi_2026_06", "2026-06-30",
           [_loan(u, 100_000.0, "2026-06-30", "2020-03-01") for u in "AB"])
    payload = _progression(root)
    assert payload["formationLoanCount"] == 2
    assert payload["singlePeriod"] is True
    assert CO.plottable([CO.adapt_progression(payload, "2020")]) == []
