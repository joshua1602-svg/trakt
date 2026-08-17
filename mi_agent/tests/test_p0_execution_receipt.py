#!/usr/bin/env python3
"""tests/test_p0_execution_receipt.py — P0 launch-hardening regressions.

The launch invariant under test:

    The MI Agent may answer correctly, answer partially with an explicit
    limitation, or refuse. It must NOT silently answer a materially different
    question.

Two groups:

**A. Execution receipt** — the receipt must describe what EXECUTED, not what the
user typed. A receipt that says "London" while the executor calculated the whole
book would be worse than no receipt at all, so every case here asserts against
the executed frame rather than against the question.

**B. Fail closed** — a material facet that does not survive into execution must
produce a refusal (or an explicitly disclosed partial), never a broader figure
presented as the narrow one.

These run against a small in-memory frame so they are fast and deterministic;
the same behaviour is exercised end to end over the governed synthetic book by
``scripts/run_mi_capability_review.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import execution_receipt as er
from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent.mi_query_validator import load_mi_semantics

_SEMANTICS_PATH = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_SEMANTICS_PATH)


@pytest.fixture(scope="module")
def frame():
    """A deterministic funded tape: 200 loans, three regions, two LTV bands."""
    rng = np.random.default_rng(20260817)
    n = 200
    regions = rng.choice(["London", "South East", "Wales"], n, p=[.4, .35, .25])
    return pd.DataFrame({
        "current_outstanding_balance": rng.uniform(50_000, 400_000, n).round(2),
        "current_loan_to_value": rng.uniform(20, 70, n).round(2),
        "current_interest_rate": rng.uniform(4, 8, n).round(2),
        "current_valuation_amount": rng.uniform(200_000, 900_000, n).round(2),
        "youngest_borrower_age": rng.integers(60, 95, n),
        "collateral_geography": regions,
        "ltv_bucket": rng.choice(["40-50%", "50-60%"], n),
        "occupancy_type": rng.choice(["Owner Occupied", "Second Home"], n),
        "amortisation_type": ["Interest roll-up"] * n,
        "data_cut_off_date": ["2026-06-30"] * n,
    })


def _ask(question, frame, semantics, **kw):
    return run_mi_agent_query(question, frame, semantics, **kw)


def _receipt(result) -> str:
    return ((result.get("execution_receipt") or {}).get("receipt")) or ""


def _summary(result) -> dict:
    return result.get("execution_receipt") or {}


# =========================================================================== #
# A. Execution receipt — describes EXECUTION, not the question
# =========================================================================== #
def test_receipt_names_the_geography_that_was_actually_applied(frame, semantics):
    """A1-shaped: a scoped question that DID apply its scope names it, and the
    population is the scoped population — not the whole book."""
    result = _ask("total balance in london", frame, semantics)
    assert result["ok"], result.get("error")
    summary = _summary(result)
    assert summary["filtersApplied"] == ["London"]
    assert summary["narrowed"] is True
    expected = int((frame["collateral_geography"] == "London").sum())
    assert summary["population"] == expected
    assert summary["population"] < len(frame)
    assert "London" in _receipt(result)
    assert f"{expected:,} loans" in _receipt(result)


def test_receipt_for_a_whole_book_answer_claims_no_geography(frame, semantics):
    """The unfiltered counterpart must say so explicitly, so it can never be
    mistaken for the scoped answer above."""
    result = _ask("what is the total balance?", frame, semantics)
    assert result["ok"], result.get("error")
    summary = _summary(result)
    assert summary["filtersApplied"] == []
    assert summary["narrowed"] is False
    assert summary["population"] == len(frame)
    receipt = _receipt(result)
    assert "entire funded portfolio" in receipt
    for region in ("London", "South East", "Wales"):
        assert region not in receipt


def test_receipt_names_an_applied_age_threshold(frame, semantics):
    result = _ask("how many loans with youngest borrower age over 85", frame, semantics)
    assert result["ok"], result.get("error")
    summary = _summary(result)
    assert summary["narrowed"] is True
    expected = int((frame["youngest_borrower_age"] > 85).sum())
    assert summary["population"] == expected
    assert "85" in _receipt(result)


def test_receipt_population_comes_from_the_executed_frame_not_the_wording(
        frame, semantics):
    """The population must be read from execution. Asking for a scope that
    matches a KNOWN NUMBER of rows proves the receipt is not echoing prose."""
    result = _ask("total balance in wales", frame, semantics)
    assert result["ok"], result.get("error")
    expected = int((frame["collateral_geography"] == "Wales").sum())
    assert _summary(result)["population"] == expected
    # And the same question against a different region gives a different count.
    other = _ask("total balance in london", frame, semantics)
    assert _summary(other)["population"] != expected


def test_receipt_names_both_applied_dimensions(frame, semantics):
    result = _ask("show me balance by region and ltv band", frame, semantics)
    assert result["ok"], result.get("error")
    summary = _summary(result)
    assert len(summary["dimensionsApplied"]) == 2
    receipt = _receipt(result)
    assert "grouped by" in receipt
    assert summary["groupCount"] and summary["groupCount"] > 1


def test_receipt_discloses_a_requested_dimension_the_dataset_lacks(frame, semantics):
    """A4-shaped: region applies, borrower type cannot. The answer stands, and
    the missing facet is NAMED rather than disappearing."""
    result = _ask("show me balance by region by borrower type", frame, semantics)
    assert result["ok"], result.get("error")
    summary = _summary(result)
    assert "borrower type" in " ".join(summary["notApplied"]).lower()
    assert result["semantic_guard"]["verdict"] == er.VERDICT_PARTIAL
    statuses = {f["label"]: f["status"] for f in summary["facets"]}
    assert statuses.get("borrower type") == er.UNAVAILABLE


def test_receipt_never_claims_a_stress_that_did_not_run(frame, semantics):
    """A receipt may only name a scenario when one was genuinely executed. The
    point-in-time path has no stress capability, so this must refuse."""
    result = _ask("what would a 10% house price fall do to my weighted average ltv?",
                  frame, semantics)
    assert result["ok"] is False
    assert _summary(result).get("scenario") in (None, "")
    assert "unstressed" in (result.get("error") or "")


def test_receipt_states_the_reporting_date_from_the_data(frame, semantics):
    result = _ask("what is the total balance?", frame, semantics)
    assert "30 June 2026" in _receipt(result)


# =========================================================================== #
# B. Fail closed — material intent that did not survive
# =========================================================================== #
def test_missing_geographic_scope_cannot_return_the_whole_book(frame, semantics):
    """The headline failure: "average LTV in London" answered from 11,035 loans.
    The average branch does not carry the filter, so this must refuse."""
    result = _ask("what is the average ltv in london?", frame, semantics)
    assert result["ok"] is False
    assert result.get("semantic_refusal") is True
    assert "London" in result["error"]
    assert "not substituted" in result["error"]


def test_missing_age_threshold_cannot_return_whole_book_exposure(frame, semantics):
    result = _ask("what is my exposure to borrowers over 85?", frame, semantics)
    assert result["ok"] is False
    assert "85" in result["error"]
    assert "not substituted" in result["error"]


def test_missing_ltv_threshold_cannot_be_answered_with_the_wa_ltv(frame, semantics):
    result = _ask("what proportion of the book is eligible for a 75% ltv securitisation?",
                  frame, semantics)
    assert result["ok"] is False
    assert "75" in result["error"]


def test_missing_stress_cannot_return_an_unstressed_kpi(frame, semantics):
    result = _ask("what would a 10% house price fall do to my weighted average ltv?",
                  frame, semantics)
    assert result["ok"] is False
    assert result.get("chart_result") is None


def test_an_unavailable_dimension_is_never_replaced_by_another_field(frame, semantics):
    """A3-shaped: neither requested dimension exists, and the parser reaches for
    amortisation_type. Substitution must refuse, and must NAME the substitute."""
    result = _ask("show me balance by ltv by borrower type", frame, semantics)
    assert result["ok"] is False
    error = result["error"].lower()
    assert "amortisation" in error
    assert "did not ask for" in error


def test_an_unavailable_dimension_never_simply_disappears(frame, semantics):
    """Either applied, or named. Never absent from the response."""
    result = _ask("show me balance by region by borrower type", frame, semantics)
    labels = {f["label"] for f in _summary(result)["facets"]}
    assert "borrower type" in labels
    assert all(f["status"] != er.LOST for f in _summary(result)["facets"])


def test_a_forward_projection_cannot_be_answered_point_in_time(frame, semantics):
    result = _ask("if origination continues at the current rate, what will the "
                  "balance be at year end?", frame, semantics)
    assert result["ok"] is False
    assert "projection" in result["error"].lower()


def test_a_measure_the_question_did_not_name_is_refused(frame, semantics):
    """Answering an LTV question with an interest-rate figure is a substitution
    even though both are valid measures."""
    substitution = er.detect_measure_substitution(
        "what is the weighted average ltv?", metric_key="current_interest_rate")
    assert substitution and "ltv" in substitution


def test_multiple_measures_are_refused_rather_than_silently_reduced_to_one(
        frame, semantics):
    result = _ask("what is the average ltv and the average borrower age?",
                  frame, semantics)
    assert result["ok"] is False
    assert "more than one measure" in result["error"]


def test_a_malformed_facet_produces_a_controlled_failure_not_a_crash(frame, semantics):
    """A guard fault must degrade to a warning, never propagate."""
    facets = [er.RequestedFacet(kind="not-a-real-kind", label="x")]
    reconciled = er.reconcile_facets(
        facets, spec=object(), query_result=None, semantics=semantics)
    assert reconciled[0].status == er.LOST  # untouched default, no exception


def test_working_answers_are_not_disabled_by_the_guard(frame, semantics):
    """Scope control: questions with no material facet must be unaffected."""
    for question in ("what is the total balance?",
                     "what is the weighted average interest rate?",
                     "show me balance by region"):
        result = _ask(question, frame, semantics)
        assert result["ok"] is True, f"{question!r} -> {result.get('error')}"
        assert result["semantic_guard"]["verdict"] == er.VERDICT_OK
        assert _receipt(result).startswith("Calculated:")


# =========================================================================== #
# Detector unit tests — the conservative-by-design claims
# =========================================================================== #
@pytest.mark.parametrize("question,expected", [
    ("what is the run rate of new lending?", []),
    ("what is the average ltv?", ["ltv"]),
    ("what is the average ltv and borrower age?", ["ltv", "age"]),
    ("if origination continues at the current rate, what is the balance?", ["balance"]),
])
def test_named_measure_concepts_ignores_velocity_phrases(question, expected):
    assert er.named_measure_concepts(question) == expected


@pytest.mark.parametrize("question", [
    "if origination continues at the current rate, what will the balance be?",
    "what is the funded balance?",
    "show me balance by region",
])
def test_stress_detector_does_not_fire_on_non_stress_questions(question):
    assert er.detect_requested_facets(question, {"fields": {}}) == [] or \
        all(f.kind != er.KIND_STRESS
            for f in er.detect_requested_facets(question, {"fields": {}}))


def test_relationship_detector_ignores_the_temporal_sense_of_relative_to():
    facets = er.detect_requested_facets(
        "which regions are we most exposed to relative to last month?", {"fields": {}})
    assert all(f.kind != er.KIND_RELATIONSHIP for f in facets)


# =========================================================================== #
# Period grain — a temporal route that compared a SHORTER span than asked
# =========================================================================== #
def _period_envelope(opening: str, closing: str) -> dict:
    return {"ok": True, "artifacts": [
        {"type": "kpi", "items": [{"label": "Opening period", "value": opening},
                                  {"label": "Closing period", "value": closing}]}]}


def test_since_inception_answered_over_one_month_is_downgraded():
    """B17-shaped: period_change genuinely compares two periods, but May-to-June
    is not "since inception". The route states its dates honestly; without this
    check nothing states that the REQUESTED span was not honoured."""
    facets = [er.RequestedFacet(kind=er.KIND_COMPARISON_PERIOD,
                                label="comparison period (since inception)",
                                status=er.APPLIED)]
    out = er.check_period_grain(facets, _period_envelope("31 May 2026", "30 June 2026"))
    assert out[0].status == er.UNSUPPORTED
    assert "does not cover" in out[0].reason
    verdict, message = er.assess(er.ExecutionReceipt(facets=out))
    assert verdict == er.VERDICT_REFUSE


def test_last_month_answered_over_one_month_is_left_applied():
    facets = [er.RequestedFacet(kind=er.KIND_COMPARISON_PERIOD,
                                label="comparison period (last month)",
                                status=er.APPLIED)]
    out = er.check_period_grain(facets, _period_envelope("31 May 2026", "30 June 2026"))
    assert out[0].status == er.APPLIED


def test_last_quarter_answered_over_a_quarter_is_left_applied():
    facets = [er.RequestedFacet(kind=er.KIND_COMPARISON_PERIOD,
                                label="comparison period (last quarter)",
                                status=er.APPLIED)]
    out = er.check_period_grain(facets, _period_envelope("31 March 2026", "30 June 2026"))
    assert out[0].status == er.APPLIED


def test_a_route_that_declares_no_periods_is_never_downgraded():
    """Never refuse on an unverifiable basis."""
    facets = [er.RequestedFacet(kind=er.KIND_COMPARISON_PERIOD,
                                label="comparison period (since inception)",
                                status=er.APPLIED)]
    out = er.check_period_grain(facets, {"ok": True, "artifacts": []})
    assert out[0].status == er.APPLIED
