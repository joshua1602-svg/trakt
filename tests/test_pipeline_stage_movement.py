#!/usr/bin/env python3
"""tests/test_pipeline_stage_movement.py — per-stage pipeline reconciliation.

    opening live + arrivals - departures ± amount change on stayers = closing live

This is a COMPOSITION of two prepared weekly extracts joined on the governed
case identifier. Nothing is modelled and no new primitive is introduced.

The three properties that make it trustworthy, each pinned below:

  * it reconciles to zero residual on every live stage;
  * a case whose LOAN AMOUNT is amended stays the same case — it must appear as
    an amount movement, never as an exit plus an arrival;
  * where case identity cannot be governed it reports UNAVAILABLE with a reason,
    and never falls back to a guess.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pd = pytest.importorskip("pandas")

LADDER = ("KFI", "APPLICATION", "OFFER", "COMPLETED")


def _write_week(root: Path, client: str, week: str, rows: list) -> None:
    folder = root / client / "pipeline" / week
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        folder / f"M2L_KFI_and_Pipeline_{week.replace('-', '_')}.csv", index=False)


def _case(cid: str, stage: str, amount: float) -> dict:
    return {"Account Number": cid, "current_outstanding_balance": amount,
            "current_valuation_amount": amount / 0.45, "current_loan_to_value": 45.0,
            "youngest_borrower_age": 70, "current_interest_rate": 7.0,
            "collateral_geography": "London", "geographic_region_obligor": "London",
            "broker_channel": "Direct", "product_type": "Lifetime Mortgage",
            "pipeline_stage": stage, "pipeline_status": "live",
            "completion_probability": 0.5}


@pytest.fixture()
def book(tmp_path, monkeypatch):
    """Two weeks in which cases progress, arrive, leave and are AMENDED."""
    root = tmp_path / "runs"
    _write_week(root, "mv", "2026-06-12", [
        _case("ACC001", "KFI", 100_000),          # -> APPLICATION next week
        _case("ACC002", "KFI", 200_000),          # stays KFI, amount amended
        _case("ACC003", "APPLICATION", 300_000),  # -> OFFER
        _case("ACC004", "APPLICATION", 400_000),  # withdrawn
        _case("ACC005", "OFFER", 500_000),        # -> COMPLETED
        _case("ACC006", "OFFER", 600_000),        # stays OFFER
    ])
    _write_week(root, "mv", "2026-06-26", [
        _case("ACC001", "APPLICATION", 100_000),
        _case("ACC002", "KFI", 220_000),          # SAME case, amount amended +20k
        _case("ACC003", "OFFER", 300_000),
        _case("ACC004", "WITHDRAWN", 400_000),
        _case("ACC005", "COMPLETED", 500_000),
        _case("ACC006", "OFFER", 600_000),
        _case("ACC007", "KFI", 700_000),          # brand new case
    ])
    monkeypatch.setenv("MI_AGENT_PIPELINE_ROOT", str(root))
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    return root


def _movement(root):
    from mi_agent_api import evolution
    return evolution.pipeline_stage_movement(root, "mv")


def _stage(result, name):
    return next(s for s in result["stages"] if s["stage"] == name)


# --------------------------------------------------------------------------- #
# 1. It reconciles.
# --------------------------------------------------------------------------- #

def test_every_live_stage_reconciles_to_zero_residual(book):
    result = _movement(book)
    assert result["available"], result.get("reason")
    assert result["reconciles"], result["stages"]
    for stage in result["stages"]:
        assert stage["residual"] == 0.0, stage


def test_the_identity_is_the_one_it_claims(book):
    """Recompute the stated identity from the reported legs."""
    result = _movement(book)
    for st in result["stages"]:
        assert (st["openingAmount"] + st["arrivalAmount"] - st["departureAmount"]
                + st["amountChangeOnPersisting"]) == pytest.approx(
                    st["closingAmount"], abs=0.01), st


def test_departures_say_where_the_case_went(book):
    """Catches: a funnel that cannot tell 'left the stage' from 'left the
    pipeline'. ACC001 moved KFI -> APPLICATION; ACC004 withdrew."""
    result = _movement(book)
    kfi = _stage(result, "KFI")
    destinations = {d["stage"]: d for d in kfi["departuresByDestination"]}
    assert "APPLICATION" in destinations, destinations
    assert destinations["APPLICATION"]["caseCount"] == 1

    application = _stage(result, "APPLICATION")
    dest = {d["stage"]: d for d in application["departuresByDestination"]}
    assert set(dest) == {"OFFER", "WITHDRAWN"}, dest


# --------------------------------------------------------------------------- #
# 2. A changing amount cannot change identity.
# --------------------------------------------------------------------------- #

def test_an_amended_amount_is_a_movement_not_an_exit_and_an_arrival(book):
    """THE PROPERTY THE SNAPSHOT REGISTER'S KEY CANNOT HOLD.

    ACC002 is at KFI in both weeks with its amount amended 200k -> 220k. It must
    appear once, as +20,000 of amount change on a persisting case.

    ``snapshot.keys.make_pipeline_opportunity_id`` hashes ``loan_amount``, so
    under that key ACC002 would be a 200,000 departure and a 220,000 arrival,
    with zero amount change — inflating both flow legs and erasing the movement.
    """
    kfi = _stage(_movement(book), "KFI")
    assert kfi["persistingCaseCount"] == 1, kfi
    assert kfi["amountChangeOnPersisting"] == pytest.approx(20_000.0), kfi
    # ACC001 departed to APPLICATION and ACC007 arrived; ACC002 did neither.
    assert kfi["departureCaseCount"] == 1
    assert kfi["arrivalCaseCount"] == 1
    assert kfi["departureAmount"] == pytest.approx(100_000.0)   # ACC001 only
    assert kfi["arrivalAmount"] == pytest.approx(700_000.0)     # ACC007 only


def test_the_identifier_is_the_contract_key_not_a_hash(book):
    result = _movement(book)
    assert result["identifierField"] == "pipeline_case_identifier"


# --------------------------------------------------------------------------- #
# 3. No identity, no answer.
# --------------------------------------------------------------------------- #

def test_a_case_anonymous_extract_suppresses_rather_than_guesses(tmp_path, monkeypatch):
    """Catches: inventing a fallback identity so the page can render."""
    root = tmp_path / "runs"
    for week in ("2026-06-12", "2026-06-26"):
        rows = [_case(f"ACC{i:03d}", "KFI", 100_000 + i) for i in range(4)]
        for r in rows:
            r.pop("Account Number")            # no governed case key at all
        _write_week(root, "anon", week, rows)
    monkeypatch.setenv("MI_AGENT_PIPELINE_ROOT", str(root))
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")

    from mi_agent_api import evolution
    result = evolution.pipeline_stage_movement(root, "anon")
    assert result["available"] is False
    assert "pipeline_case_identifier" in result["reason"]
    assert "stages" not in result


def test_duplicate_identifiers_are_refused_not_averaged(tmp_path, monkeypatch):
    """A duplicated key is not an identity. Refuse rather than pick one."""
    root = tmp_path / "runs"
    for week in ("2026-06-12", "2026-06-26"):
        _write_week(root, "dup", week, [
            _case("ACC001", "KFI", 100_000),
            _case("ACC001", "KFI", 200_000),     # same key twice
            _case("ACC002", "OFFER", 300_000),
        ])
    monkeypatch.setenv("MI_AGENT_PIPELINE_ROOT", str(root))
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")

    from mi_agent_api import evolution
    result = evolution.pipeline_stage_movement(root, "dup")
    assert result["available"] is False, result


def test_one_extract_is_not_a_movement(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    _write_week(root, "one", "2026-06-26", [_case("ACC001", "KFI", 100_000)])
    monkeypatch.setenv("MI_AGENT_PIPELINE_ROOT", str(root))
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    from mi_agent_api import evolution
    result = evolution.pipeline_stage_movement(root, "one")
    assert result["available"] is False
    assert "two governed weekly extracts" in result["reason"]
