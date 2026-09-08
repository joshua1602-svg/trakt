#!/usr/bin/env python3
"""tests/test_borrowing_base_slide.py

The pack's facility page, held to the dashboard's figures.

A borrowing base is a covenant number. If the pack and the dashboard disagree
about eligible collateral, headroom or utilisation, neither can be relied on —
so the slide reads the governed envelope verbatim and computes nothing, and the
adapter is where that is enforced.

The case these mostly exist for is the one a dash would hide: a measure the
facility configuration cannot support comes back as ``NOT_CALCULABLE``, and it
has to be printed as the named missing input. "—" reads as nil, and a 0 on a
headroom tile reads as fully drawn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mi_agent_pptx import borrowing_base as BB     # noqa: E402
from mi_agent_pptx import composition as C         # noqa: E402


def snap(**over):
    """A healthy governed snapshot; tests override the field under test."""
    base = {
        "available": True, "reconciles": True,
        "facility": {"facilityLabel": "Warehouse A",
                     "facilityType": "revolving_warehouse",
                     "currentDrawnAmountAsOf": "2026-06-30"},
        "advanceRatePct": 85.0, "facilityCommitment": 120_000_000.0,
        "financingPortfolioLoanCount": 400, "financingPortfolioBalance": 100_000_000.0,
        "eligibleLoanCount": 360, "eligibleCurrentBalance": 90_000_000.0,
        "eligibleShareOfFinancingPortfolioPct": 90.0,
        "ineligibleLoanCount": 30, "ineligibleCurrentBalance": 7_000_000.0,
        "ineligibleShareOfFinancingPortfolioPct": 7.0,
        "undeterminedLoanCount": 10, "undeterminedCurrentBalance": 3_000_000.0,
        "undeterminedShareOfFinancingPortfolioPct": 3.0,
        "grossBorrowingBase": 76_500_000.0, "availableBorrowingBase": 76_500_000.0,
        "facilityCapBinding": False, "currentDrawnAmount": 60_000_000.0,
        "borrowingBaseHeadroom": 16_500_000.0, "borrowingBaseDeficiency": 0.0,
        "borrowingBaseUtilisationPct": 78.4, "facilityUtilisationPct": 50.0,
        "concentrationLimitDenominator": 90_000_000.0,
        "concentrationDenominatorFloorBinding": False,
        "missingInputs": [], "prototypeAssumptionsUsed": [],
        "concentrationAdjustment": {"note": "Breaches are monitored, not deducted."},
    }
    base.update(over)
    return base


def env(**over):
    return {"borrowingBase": snap(**over)}


# --------------------------------------------------------------------------- #
# Availability.
# --------------------------------------------------------------------------- #

def test_a_book_with_no_facility_has_no_borrowing_base():
    assert BB.available({"borrowingBase": {"available": False,
                                           "reason": "No funding facility."}}) is False


def test_the_engines_own_reason_is_what_the_page_says():
    """Not a generic message. The engine knows why and the reader gets that."""
    e = {"borrowingBase": {"available": False,
                           "reason": "No funding facility is configured."}}
    assert BB.reason(e) == "No funding facility is configured."


def test_a_missing_block_is_not_a_crash():
    for e in ({}, None, {"borrowingBase": None}, {"borrowingBase": []}):
        assert BB.available(e) is False
        assert BB.reason(e)


def test_the_slide_is_omitted_where_there_is_no_facility():
    """Most books have concentration tests and no facility, so the fact is
    separate from has_concentration even though they share an envelope."""
    spec = yaml.safe_load((_ROOT / "configs" / "pptx" / "investor_pack.yaml")
                          .read_text(encoding="utf-8"))
    slide = next(x for x in spec["slides"] if x["id"] == "borrowing_base")
    assert slide["when"] == "has_borrowing_base"
    assert C.evaluate_condition(slide["when"], {"has_borrowing_base": False}) is False
    assert C.evaluate_condition(slide["when"], {"has_borrowing_base": True}) is True


def test_the_omission_carries_a_reason_a_reader_understands():
    """The appendix says why a page is absent, in the reader's words."""
    assert "no funding facility" in C._CONDITION_WORDING["has_borrowing_base"].lower()


# --------------------------------------------------------------------------- #
# NOT_CALCULABLE is an answer, not a gap.
# --------------------------------------------------------------------------- #

def test_a_not_calculable_measure_is_never_a_dash_or_a_zero():
    """THE CASE THIS PAGE IS MOST DANGEROUS ON.

    Drawings not supplied means headroom is unknown. Printed as "—" a reader
    reads nil; printed as 0 they read fully drawn. Both are covenant statements
    the data does not support.
    """
    s = snap(currentDrawnAmount=BB.NOT_CALCULABLE,
             borrowingBaseHeadroom=BB.NOT_CALCULABLE,
             facilityUtilisationPct=BB.NOT_CALCULABLE,
             missingInputs=["current_drawn_amount"])
    for tile in BB.tiles(s):
        if tile["label"] in ("FACILITY DRAWN", "HEADROOM", "FACILITY UTILISATION"):
            assert tile["value"] is None, tile
            assert tile["missing"] == "facility drawings not supplied", tile


def test_the_missing_input_is_named_in_the_readers_words():
    for key, phrase in BB.MISSING_INPUT_LABEL.items():
        s = snap(missingInputs=[key])
        assert BB.missing_phrase(s, key) == phrase
        assert "not supplied" in phrase or "not configured" in phrase


def test_a_missing_input_not_in_the_list_is_not_claimed():
    assert BB.missing_phrase(snap(missingInputs=[]), "advance_rate") is None


def test_not_calculable_never_becomes_a_number():
    assert BB.money(BB.NOT_CALCULABLE) is None
    assert BB.pct(BB.NOT_CALCULABLE) is None
    assert BB.count(BB.NOT_CALCULABLE) == "—"


# --------------------------------------------------------------------------- #
# The figures are the engine's.
# --------------------------------------------------------------------------- #

def test_the_adapter_computes_nothing():
    """No division, no capping, no flooring — the panel's own constraint.

    Walked as a syntax tree rather than grepped, so a division genuinely in the
    code fails and a slash in a docstring does not. String concatenation is
    allowed: joining a sentence is not arithmetic on a covenant figure.
    """
    import ast

    src = (_ROOT / "mi_agent_pptx" / "borrowing_base.py").read_text(encoding="utf-8")
    arithmetic = (ast.Div, ast.Mult, ast.Sub, ast.FloorDiv, ast.Pow, ast.Mod)
    offenders = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, arithmetic):
            offenders.append((type(node.op).__name__, node.lineno))
        # Addition is only allowed on strings and lists.
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            operands = (node.left, node.right)
            if any(isinstance(o, ast.Constant) and isinstance(o.value, (int, float))
                   for o in operands):
                offenders.append(("Add", node.lineno))
    assert not offenders, offenders


def test_headroom_is_the_engines_headroom():
    assert BB.tiles(snap())[3]["value"] == BB.money(16_500_000.0)


def test_an_over_drawn_facility_shows_nil_headroom_and_the_deficiency():
    """Presentation only — the governed headroom stays negative on the
    envelope. A negative headroom on a tile reads as a rebate."""
    s = snap(borrowingBaseHeadroom=-4_000_000.0,
             borrowingBaseDeficiency=4_000_000.0)
    assert BB.over_drawn(s) is True
    tile = next(t for t in BB.tiles(s) if t["label"] == "HEADROOM")
    assert tile["value"] == BB.money(0)
    assert "deficiency" in tile["sub"]
    assert tile["status"] == "breach"
    assert s["borrowingBaseHeadroom"] == -4_000_000.0, "the envelope was mutated"


def test_the_cap_is_reported_where_it_binds():
    tile = next(t for t in BB.tiles(snap(facilityCapBinding=True))
                if t["label"] == "BORROWING BASE")
    assert tile["sub"] == "capped at facility commitment"


def test_the_gross_base_is_shown_where_the_cap_does_not_bind():
    tile = next(t for t in BB.tiles(snap()) if t["label"] == "BORROWING BASE")
    assert tile["sub"].startswith("gross ")


# --------------------------------------------------------------------------- #
# Parity with the dashboard.
# --------------------------------------------------------------------------- #

def test_the_five_measures_are_the_dashboards_five_in_its_order():
    assert [t["label"] for t in BB.tiles(snap())] == [
        "ELIGIBLE COLLATERAL", "BORROWING BASE", "FACILITY DRAWN",
        "HEADROOM", "FACILITY UTILISATION"]


def test_the_split_is_the_dashboards_three_in_its_order():
    assert [r["label"] for r in BB.split(snap())] == [
        "Eligible", "Ineligible", "Undetermined"]


def test_the_utilisation_thresholds_are_the_dashboards():
    """100 rose, 90 amber — the same two numbers the panel tones on, so a
    facility that looks comfortable on screen looks comfortable in the pack."""
    assert BB.utilisation_status(snap(facilityUtilisationPct=100.0)) == "breach"
    assert BB.utilisation_status(snap(facilityUtilisationPct=90.0)) == "warning"
    assert BB.utilisation_status(snap(facilityUtilisationPct=89.9)) == "pass"
    assert BB.utilisation_status(snap(facilityUtilisationPct=BB.NOT_CALCULABLE)) \
        == "unavailable"


def test_the_missing_input_vocabulary_matches_the_panels():
    """One vocabulary across both surfaces, checked against the .tsx itself."""
    tsx = (_ROOT / "frontend" / "mi-agent-ui" / "src" / "components" / "risk"
           / "BorrowingBasePanel.tsx").read_text(encoding="utf-8")
    for key, phrase in BB.MISSING_INPUT_LABEL.items():
        assert key in tsx, key
        assert phrase in tsx, phrase


# --------------------------------------------------------------------------- #
# What must be said before the figures are read.
# --------------------------------------------------------------------------- #

def test_an_unreconciled_population_disqualifies_the_figures_out_loud():
    """Showing a borrowing base off a population that does not reconcile, and
    not saying so, is a covenant claim the data cannot support."""
    alerts = BB.alerts(snap(reconciles=False))
    assert alerts[0]["tone"] == "breach"
    assert "NOT a governed borrowing base" in alerts[0]["text"]


def test_the_reconciliation_alert_outranks_the_others():
    alerts = BB.alerts(snap(reconciles=False, borrowingBaseHeadroom=-1.0,
                            borrowingBaseDeficiency=1.0,
                            prototypeAssumptionsUsed=["assumed X"]))
    assert "does not reconcile" in alerts[0]["text"]


def test_a_prototype_assumption_is_declared():
    alerts = BB.alerts(snap(prototypeAssumptionsUsed=["Financing portfolio assumed."]))
    assert any("Prototype assumption in use." in a["text"] for a in alerts)


def test_a_healthy_facility_says_nothing_it_does_not_need_to():
    assert BB.alerts(snap()) == []


def test_the_breach_treatment_travels_with_the_limit():
    """A reader must not assume the base is already net of a breach — this
    facility monitors rather than deducts, and the note says so."""
    note = BB.concentration_note(snap(
        nearestConcentrationLimit="Region — London", nearestConcentrationHeadroomPct=4.25,
        nearestConcentrationHeadroomAmount=3_800_000.0, breachedConcentrationCount=0))
    assert "Region — London" in note and "4.25% headroom" in note
    assert "monitored, not deducted" in note


def test_the_population_line_names_what_the_split_is_of():
    line = BB.population_line(snap(concentrationDenominatorFloorBinding=True))
    assert "Financing Portfolio" in line and "400 loans" in line
    assert "Concentration Limit Denominator" in line
    assert "contractual floor binding" in line


# --------------------------------------------------------------------------- #
# Why loans are out.
# --------------------------------------------------------------------------- #

def _derivation(**counts):
    return {"eligibility_derivation": {"applied": True, "reason_counts": counts}}


def test_the_reasons_are_the_derivations_own_counts():
    """The dashboard answers this with a drill-down button; a pack carries the
    shape of the answer instead. Nothing is classified or counted here."""
    rows = BB.exclusion_reasons(snap(receipt=_derivation(
        all_approved_eligibility_rules_satisfied=360,
        outside_financing_portfolio=22, eligibility_rule_input_missing=10)))
    assert [(r["code"], r["count"]) for r in rows] == [
        ("outside_financing_portfolio", 22),
        ("eligibility_rule_input_missing", 10)]


def test_the_reason_a_loan_is_eligible_is_not_a_reason_it_is_out():
    rows = BB.exclusion_reasons(snap(receipt=_derivation(
        all_approved_eligibility_rules_satisfied=400)))
    assert rows == []


def test_a_derivation_that_did_not_run_says_nothing_rather_than_nil():
    """Absent is not the same as "no loans are ineligible", so the panel is
    omitted rather than drawn empty."""
    assert BB.exclusion_reasons(snap(receipt={})) == []
    assert BB.exclusion_reasons(snap(
        receipt={"eligibility_derivation": {"applied": False}})) == []
    assert BB.exclusion_reasons(snap()) == []


def test_empty_groups_are_not_listed():
    rows = BB.exclusion_reasons(snap(receipt=_derivation(
        outside_financing_portfolio=0, eligibility_rule_input_missing=4)))
    assert [r["code"] for r in rows] == ["eligibility_rule_input_missing"]


def test_the_largest_group_leads_and_ties_do_not_reorder():
    counts = {f"rule_{i}": 5 for i in range(4)}
    counts["big_one"] = 40
    first = BB.exclusion_reasons(snap(receipt=_derivation(**counts)))
    assert first[0]["code"] == "big_one"
    for _ in range(4):
        assert BB.exclusion_reasons(snap(receipt=_derivation(**counts))) == first


def test_the_page_carries_only_what_it_can_read():
    counts = {f"rule_{i}": 10 - i for i in range(9)}
    assert len(BB.exclusion_reasons(snap(receipt=_derivation(**counts)))) \
        == BB.MAX_REASONS


def test_a_configured_rules_own_code_is_humanised_not_guessed_at():
    """A rule supplies its own code, which cannot be known in advance. It stays
    recognisably the code, so an operator can check the pack against the
    configuration — but it does not print "Ltv" at a funder."""
    assert BB.reason_label("ltv_above_facility_cap") == "LTV above facility cap"
    assert BB.reason_label("dscr_below_floor") == "DSCR below floor"
    assert BB.reason_label("some_bespoke_rule") == "Some bespoke rule"


def test_the_known_vocabulary_is_written_out_in_full():
    for code, label in BB.REASON_LABEL.items():
        assert BB.reason_label(code) == label
        assert label[0].isupper() and "_" not in label


def test_an_absent_code_never_prints_as_none():
    assert BB.reason_label(None) == "Unattributed"
    assert BB.reason_label("") == "Unattributed"
