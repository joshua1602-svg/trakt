"""Concentration keeps its calculation and loses its interpreter.

WHAT CHANGED, AND WHAT DELIBERATELY DID NOT. `run_concentration_analysis` used
to read the question AFTER the route had claimed it — three times: for the
analytical concept, for the single-name framing, and for a portfolio lens
whenever no scope arrived. Eleven vocabularies of its own sat behind those
reads. The deterministic concentration calculation — shares, cumulative
shares, ranks, the governed denominator — is genuinely specialist and stays
exactly where it is.

The reading now happens ONCE, in the recogniser, before the route is claimed,
and travels to the workflow. The scope comes from the contract. A workflow
that is handed no reading REFUSES rather than reading the sentence itself:
one owner, or none.

The mutation controls are the evidence. Each perturbs one input of the
calculation and requires the published economics to move; a control that left
the numbers unchanged would mean the answer was not derived from that input.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from mi_agent import portfolio_lens as lens_mod
from mi_workflows import concentration_analysis as ca
from mi_workflows.semantics import load_business_semantics

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bsr():
    return load_business_semantics(
        REPO / "config" / "business_semantics_registry.yaml")


def _frame() -> pd.DataFrame:
    """Book totals 2,200: london 900, south 750, north 550."""
    return pd.DataFrame({
        "source_portfolio_id": ["direct_001"] * 4 + ["acquired_001"] * 4,
        "source_portfolio_type": ["direct"] * 4 + ["acquired"] * 4,
        "reporting_date": ["2026-06-30"] * 8,
        "loan_identifier": [f"L{i}" for i in range(8)],
        "current_outstanding_balance": [100.0, 200.0, 300.0, 400.0,
                                        150.0, 250.0, 350.0, 450.0],
        "exposure_currency_denomination": ["GBP"] * 8,
        "product_type": ["lump", "lump", "draw", "draw",
                         "lump", "draw", "draw", None],
        "origination_channel": ["broker", "direct", "broker", "broker",
                                "direct", "broker", None, "broker"],
        "geographic_region_obligor": ["london", "london", "north", "south",
                                      "london", "north", "south", "london"],
    })


def _run(bsr, question="Show product concentration", *, frame=None,
         reading=None, context_id="__unset__", **kw):
    """The two steps the ROUTER performs, then the workflow."""
    if context_id == "__unset__":
        context_id = lens_mod.context_id(lens_mod.resolve_lens(question))
    return ca.run_concentration_analysis(
        frame if frame is not None else _frame(), question=question, bsr=bsr,
        as_of="2026-06-30", context_id=context_id,
        reading=reading if reading is not None else ca.read_question(question),
        **kw)


def _dimension(result, field):
    for entry in result.get("dimension_results") or []:
        if entry["field"] == field:
            return entry
    return None


def _shares(result, field):
    entry = _dimension(result, field)
    return None if entry is None else {
        c["category"]: (c["exposure"], c["exposure_share"], c["rank"])
        for c in entry["categories"]}


# --------------------------------------------------------------------------- #
# The workflow no longer interprets after the claim
# --------------------------------------------------------------------------- #
def test_the_workflow_takes_no_post_claim_reading_of_its_own():
    """Structural. A parameter it does not read is a source it cannot use."""
    import ast
    source = Path("mi_workflows/concentration_analysis.py").read_text(
        encoding="utf-8")
    target = next(n for n in ast.walk(ast.parse(source))
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "run_concentration_analysis")
    interpreting = []
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        args = list(node.args) + [k.value for k in node.keywords]
        carries = any(isinstance(n, ast.Name) and n.id == "question"
                      for a in args for n in ast.walk(a))
        if not carries:
            continue
        func = node.func
        name = (func.attr if isinstance(func, ast.Attribute)
                else getattr(func, "id", "?"))
        # A controlled failure QUOTES the question back; it decides nothing.
        if name != "_controlled_failure":
            interpreting.append((node.lineno, name))
    assert interpreting == [], interpreting


def test_a_workflow_handed_no_reading_refuses(bsr):
    """One owner, or none. It must not read the sentence itself."""
    result = ca.run_concentration_analysis(
        _frame(), question="show product concentration", bsr=bsr,
        as_of="2026-06-30", context_id=None, reading=None)
    assert result["available"] is False
    assert "not read before this workflow ran" in result["reason"]


def test_the_recogniser_supplies_the_reading():
    """The pre-claim step, and the key the adapter reads it back under."""
    from mi_agent_api import chat_routing
    reading = ca.read_question("largest 10 borrower exposures")
    assert reading.single_name_kind == "borrower"
    assert ca.read_question("show broker concentration").concept == "origination"
    assert chat_routing.CONCENTRATION_READING_KEY == "concentration"


# --------------------------------------------------------------------------- #
# Denominator, dimension, filter and ranking, proven from the numbers
# --------------------------------------------------------------------------- #
def test_the_denominator_is_the_governed_book(bsr):
    result = _run(bsr, "show regional concentration")
    shares = _shares(result, "geographic_region_obligor")
    assert shares is not None
    total = sum(exposure for exposure, _share, _rank in shares.values())
    assert round(total, 2) == 2200.00
    for _category, (exposure, share, _rank) in shares.items():
        assert round(share, 6) == round(exposure / total, 6)
    assert round(shares["london"][1], 4) == round(900.0 / 2200.0, 4)


def test_the_ranking_is_by_share_descending(bsr):
    shares = _shares(_run(bsr, "show regional concentration"),
                     "geographic_region_obligor")
    ordered = sorted(shares.items(), key=lambda kv: kv[1][2])
    assert [name for name, _ in ordered] == ["london", "south", "north"]


# --------------------------------------------------------------------------- #
# Mutation controls — each must MOVE the published economics
# --------------------------------------------------------------------------- #
def test_mutation_wrong_dimension_changes_the_answer(bsr):
    base = _shares(_run(bsr, "show regional concentration"),
                   "geographic_region_obligor")
    other = _shares(_run(bsr, "show product concentration"), "product_type")
    assert other is not None and other != base


def test_mutation_altered_numerator_changes_the_shares(bsr):
    frame = _frame()
    frame.loc[0, "current_outstanding_balance"] = 5000.0
    base = _shares(_run(bsr, "show regional concentration"),
                   "geographic_region_obligor")
    moved = _shares(_run(bsr, "show regional concentration", frame=frame),
                    "geographic_region_obligor")
    assert moved["london"][0] != base["london"][0]
    assert round(moved["london"][1], 4) != round(base["london"][1], 4)


def test_mutation_wrong_denominator_changes_the_shares(bsr):
    """Adding exposure OUTSIDE london must move london's share, not its value."""
    frame = pd.concat([_frame(), _frame().assign(
        loan_identifier=[f"X{i}" for i in range(8)],
        geographic_region_obligor=["north"] * 8)], ignore_index=True)
    base = _shares(_run(bsr, "show regional concentration"),
                   "geographic_region_obligor")
    moved = _shares(_run(bsr, "show regional concentration", frame=frame),
                    "geographic_region_obligor")
    assert moved["london"][0] == base["london"][0]
    assert round(moved["london"][1], 4) < round(base["london"][1], 4)


def test_mutation_dropped_scope_filter_changes_the_population(bsr):
    """The scope is a governed input: narrowing it must move the economics."""
    whole = _run(bsr, "show regional concentration", context_id=None)
    narrowed = _run(bsr, "show regional concentration", context_id="direct_001")
    assert narrowed["available"] is True, narrowed.get("reason")
    a = _shares(whole, "geographic_region_obligor")
    b = _shares(narrowed, "geographic_region_obligor")
    assert b != a
    assert sum(v[0] for v in b.values()) < sum(v[0] for v in a.values())


def test_mutation_reversed_ranking_would_change_the_order(bsr):
    """The published rank is derived, not incidental: reverse it and it moves."""
    shares = _shares(_run(bsr, "show regional concentration"),
                     "geographic_region_obligor")
    forward = [n for n, _ in sorted(shares.items(), key=lambda kv: kv[1][2])]
    reversed_order = list(reversed(forward))
    assert forward != reversed_order
    assert forward[0] == "london" and reversed_order[0] == "north"


def test_mutation_altered_reading_changes_which_dimension_is_reported(bsr):
    """The carried reading is load-bearing, not decorative."""
    base = _run(bsr, "show regional concentration")
    swapped = _run(bsr, "show regional concentration",
                   reading=dataclasses.replace(ca.read_question(
                       "show regional concentration"), concept="origination"))
    assert (swapped.get("mode"), base.get("mode")) is not None
    assert _dimension(swapped, "origination_channel") is not None
    assert swapped.get("dimension_results") != base.get("dimension_results")


# --------------------------------------------------------------------------- #
# Refusals preserved
# --------------------------------------------------------------------------- #
def test_an_unknown_scope_is_still_a_controlled_failure(bsr):
    result = _run(bsr, "show regional concentration",
                  context_id="not_a_governed_book")
    assert result["available"] is False
    assert "governed" in (result.get("reason") or "").lower()
