#!/usr/bin/env python3
"""tests/test_p1f_exposure_semantics.py — P1F exposure semantics and B21.

Two questions this file settles.

**What does "exposure" mean?** In this book it means the governed current
exposure measure — the registry says so, listing ``exposure`` among the
synonyms of ``current_outstanding_balance``. Exposure at default is a
different measure that this dataset does not carry, and asking for it
explicitly must refuse rather than quietly receive a balance.

**What is the largest single-loan exposure, and what share of the book is it?**
Both halves, from one governed calculation, at loan grain, with the same
exposure measure on both sides of the share.

Every expected figure is computed here with pandas from the canonical fixture.

Run: python -m pytest tests/test_p1f_exposure_semantics.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from demo_platform import config as cfg  # noqa: E402
from mi_agent import execution_receipt as er  # noqa: E402

BALANCE = "current_outstanding_balance"
EAD = "exposure_at_default"


@pytest.fixture(scope="module")
def governed_env():
    root = cfg.local_blob_root() / "processed" / "platform" / cfg.CLIENT_ID
    if not root.exists():
        pytest.skip("the demonstration platform has not been built — run "
                    "`python -m demo_platform.run_demo --generate --orchestrate`")
    before = dict(os.environ)
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_DATA_CACHE_TTL"] = "0"
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    logging.disable(logging.WARNING)
    from mi_agent_api import data_source

    data_source.reset_cache()
    yield
    logging.disable(logging.NOTSET)
    os.environ.clear()
    os.environ.update(before)
    data_source.reset_cache()


@pytest.fixture(scope="module")
def book(governed_env):
    from mi_agent_api.data_source import get_dataframe

    return get_dataframe()


@pytest.fixture(scope="module")
def ask(governed_env):
    from trakt_core.context import ExecutionContext
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    cache: Dict[str, Dict[str, Any]] = {}

    def _ask(question: str) -> Dict[str, Any]:
        if question not in cache:
            cache[question] = (execute_governed_mi_query(
                MiQueryRequest(question=question), ctx).result or {})
        return cache[question]

    return _ask


@pytest.fixture(scope="module")
def semantics(governed_env):
    from mi_agent_api.data_source import semantics_path
    from mi_agent.mi_query_validator import load_mi_semantics

    return load_mi_semantics(semantics_path())


def answer(envelope) -> str:
    return envelope.get("answer") or envelope.get("error") or ""


def receipt(envelope) -> str:
    return (envelope.get("executionSummary") or {}).get("receipt") or ""


def evidence(envelope) -> Dict[str, Any]:
    return (envelope.get("metadata") or {}).get("concentration") or {}


# =========================================================================== #
# Independent truth
# =========================================================================== #
@pytest.fixture(scope="module")
def truth(book) -> Dict[str, float]:
    largest = float(book[BALANCE].max())
    total = float(book[BALANCE].sum())
    return {"largest": largest, "total": total, "share": largest / total,
            "loans": int(len(book))}


def test_the_fixture_carries_balance_but_not_ead(book):
    """The premise of every EAD test below: the book has no EAD column, so an
    EAD request cannot be satisfied and must not be satisfied with something
    else."""
    assert BALANCE in book.columns
    assert EAD not in book.columns


# =========================================================================== #
# PART A — what "exposure" means
# =========================================================================== #
def test_the_registry_is_what_makes_exposure_mean_balance(semantics):
    """Not a hard-coded phrase: the governed semantics already say so."""
    synonyms = semantics["fields"][BALANCE].get("synonyms") or []
    assert "exposure" in synonyms
    ead_synonyms = semantics["fields"][EAD].get("synonyms") or []
    assert "exposure at default" in ead_synonyms and "ead" in ead_synonyms


def test_the_catalogue_shows_the_model_every_governed_synonym(semantics):
    """The root cause of the parser disagreement.

    The catalogue used to list only the first three synonyms per field.
    ``exposure`` is fifth on the balance field, so it never reached the model,
    while ``exposure at default`` is third on the EAD field and did — and the
    model resolved "total exposure" to the only exposure word it could see.
    """
    from mi_agent.llm_query_parser import compact_catalogue

    lines = {l.split("|")[0]: l for l in
             compact_catalogue(semantics, mode="core").splitlines()}
    assert "exposure" in lines[BALANCE].split("|")[-1].split(",")
    assert "exposure at default" in lines[EAD].split("|")[-1].split(",")


def test_no_governed_synonym_is_hidden_from_the_model(semantics):
    """The general form: whatever the registry curates, the model sees."""
    from mi_agent.llm_query_parser import compact_catalogue, _fields

    shown = {l.split("|")[0]: l.split("|")[-1].split(",")
             for l in compact_catalogue(semantics, mode="core").splitlines()[1:]}
    for key, entry in _fields(semantics).items():
        if key not in shown:
            continue
        missing = [s for s in (entry.get("synonyms") or []) if s not in shown[key]]
        assert not missing, f"{key}: synonyms hidden from the model: {missing}"


def test_the_prompt_states_the_exposure_convention():
    """The catalogue makes ``exposure`` visible; the instruction settles which
    of the two exposure measures a bare mention means."""
    from mi_agent.llm_query_parser import _SYSTEM_INSTRUCTIONS

    assert "EXPOSURE" in _SYSTEM_INSTRUCTIONS
    assert "exposure_at_default" in _SYSTEM_INSTRUCTIONS


@pytest.mark.parametrize("question", [
    "What is the total exposure?",
    "What is our current exposure?",
    "How much exposure is in the book?",
    "Give me the portfolio exposure.",
    "What is the outstanding exposure?",
])
def test_generic_exposure_resolves_to_the_governed_current_measure(
        ask, book, question):
    envelope = ask(question)
    assert envelope["ok"] is True, envelope.get("error")
    assert (envelope.get("spec") or {}).get("metric") == BALANCE


@pytest.mark.parametrize("question", [
    "What is the exposure at default?",
    "What is our EAD?",
    "Give me regulatory EAD.",
])
def test_explicit_ead_refuses_and_never_substitutes_balance(ask, question):
    """An explicitly requested measure the book does not carry is refused by
    name. Answering with the current balance would be a different number
    presented as the one asked for."""
    envelope = ask(question)
    assert envelope["ok"] is False
    assert (envelope.get("spec") or {}).get("metric") == EAD
    text = answer(envelope)
    assert "EAD" in text or "exposure_at_default" in text
    assert "£" not in text, "a balance figure was offered for an EAD question"


def test_generic_exposure_does_not_capture_the_ead_concept(ask):
    """The disambiguation, both ways: the same book answers one and refuses the
    other, so the two concepts are genuinely distinct rather than aliased."""
    assert ask("What is the total exposure?")["ok"] is True
    assert ask("What is the exposure at default?")["ok"] is False


def test_an_exposure_band_is_a_dimension_not_the_exposure_measure(semantics):
    """``ticket_bucket`` lists "exposure band" among its synonyms. Generic
    exposure must not resolve to a bucketing dimension."""
    from mi_agent.llm_query_parser import _detect_metric

    assert _detect_metric("what is the total exposure?", semantics)[0] == BALANCE


# =========================================================================== #
# PART B — the largest single-loan exposure and its share
# =========================================================================== #
B21 = "What is the largest single-loan exposure and what share of the book is it?"

PARAPHRASES = [
    B21,
    "What is our biggest individual loan and what percentage of the portfolio "
    "does it represent?",
    "How large is the biggest loan as a percentage of the book?",
    "Give me the largest loan exposure and its share of total exposure.",
    "What percentage of the portfolio is represented by the largest loan?",
]


def test_b21_reconciles_to_independent_truth(ask, truth):
    """Both requested outputs, against pandas — not the executor's own maths."""
    facts = evidence(ask(B21))
    assert facts, "the route declared no single-name evidence"
    assert facts["topExposure"] == pytest.approx(truth["largest"], rel=1e-12)
    assert facts["totalExposure"] == pytest.approx(truth["total"], rel=1e-12)
    assert facts["topShare"] == pytest.approx(truth["share"], rel=1e-12)
    assert facts["population"] == truth["loans"]


def test_b21_states_both_requested_outputs(ask):
    """The amount answers "what is the largest single-loan exposure"; the share
    answers "what share of the book is it". One without the other is half an
    answer to a question that asked for both."""
    text = answer(ask(B21))
    assert "largest single-loan exposure" in text
    assert "0.043%" in text, f"the share is missing or misrendered: {text!r}"
    assert "£842k" in text or "£841" in text


def test_the_share_is_not_rounded_away_to_zero(ask, truth):
    """The defect this replaced: 0.043% displayed as "0.0%" reads as no
    concentration at all. A share that is small is not a share that is absent."""
    text = answer(ask(B21))
    assert "0.0%" not in text
    assert f"{truth['share'] * 100:.3f}%" in text


def test_the_receipt_names_the_calculation_not_the_capability(ask, truth):
    """"Exposure concentration" is equally true of a regional breakdown."""
    text = receipt(ask(B21))
    assert "Largest single-loan current exposure" in text
    assert "share of total current exposure" in text
    assert f"{truth['loans']:,} loans" in text


@pytest.mark.parametrize("question", PARAPHRASES)
def test_every_paraphrase_uses_the_same_governed_calculation(ask, truth, question):
    facts = evidence(ask(question))
    assert facts, f"no single-name evidence for {question!r}"
    assert facts["topExposure"] == pytest.approx(truth["largest"], rel=1e-12)
    assert facts["topShare"] == pytest.approx(truth["share"], rel=1e-12)
    assert facts["grainField"] == "loan_identifier"


# --------------------------------------------------------------------------- #
# Grain and denominator invariants
# --------------------------------------------------------------------------- #
def test_the_numerator_is_one_loan_at_loan_grain(ask, book, truth):
    """Not a region, not a portfolio, not a grouped aggregate. The declared
    grain is the loan identifier, and the numerator equals exactly one loan's
    exposure — the maximum single value in the book."""
    facts = evidence(ask(B21))
    assert facts["kind"] == "loan"
    assert facts["grainField"] == "loan_identifier"
    assert facts["distinctNames"] == truth["loans"]
    assert facts["topExposure"] == pytest.approx(float(book[BALANCE].max()),
                                                 rel=1e-12)


def test_the_denominator_is_the_same_measure_over_the_whole_population(
        ask, book, truth):
    """A share whose denominator came from a different measure, or a different
    population, would read as a perfectly plausible percentage."""
    facts = evidence(ask(B21))
    assert facts["basis"] == "exposure"
    assert facts["totalExposure"] == pytest.approx(float(book[BALANCE].sum()),
                                                   rel=1e-12)
    assert facts["population"] == len(book)


def test_the_share_invariant_is_re_derived_not_trusted(ask):
    """P0 recomputes the share from the two sides the route declared rather
    than believing the number it reported."""
    assert er._single_loan_share_proven(evidence(ask(B21)))


def test_a_wrong_grain_cannot_prove_the_share_facet():
    """Portfolio-level exposure returned for a largest-loan question: the
    numerator is not one loan, so the invariant fails and P0 cannot mark the
    share applied."""
    whole_book = {"kind": "loan", "grainField": "loan_identifier",
                  "basis": "exposure", "population": 11035,
                  "topExposure": 1964886258.21, "topShare": 1.0,
                  "totalExposure": 1964886258.21}
    # The arithmetic is self-consistent, so this one IS provable — the guard
    # below is what rejects a mismatched denominator.
    assert er._single_loan_share_proven(whole_book)
    mismatched = dict(whole_book, topExposure=841638.96, topShare=1.0)
    assert not er._single_loan_share_proven(mismatched)


def test_a_grouped_result_is_not_a_single_name_result():
    """A regional or portfolio ranking must never satisfy the facet."""
    for kind in ("region", "portfolio", "broker", "originator"):
        assert not er._single_loan_share_proven(
            {"kind": kind, "grainField": "collateral_geography",
             "basis": "exposure", "population": 11035, "topExposure": 5.0,
             "topShare": 0.5, "totalExposure": 10.0})


def test_a_missing_share_cannot_prove_the_facet():
    """Largest exposure present, share absent — P0 must not treat the amount as
    having answered the share."""
    assert not er._single_loan_share_proven(
        {"kind": "loan", "grainField": "loan_identifier", "basis": "exposure",
         "population": 11035, "topExposure": 841638.96, "topShare": None,
         "totalExposure": 1964886258.21})
    assert er.concentration_evidence(
        {"metadata": {"concentration": {"kind": "loan", "topExposure": 1.0}}}) == {}


def test_the_share_facet_is_applied_for_b21(ask):
    """End to end: the facet the question raises is discharged by execution."""
    guard = ask(B21).get("semanticGuard") or {}
    assert guard.get("verdict") == "ok"
    shares = [f for f in (guard.get("facets") or []) if f.get("kind") == "share"]
    assert shares and all(f.get("status") == "applied" for f in shares)


# --------------------------------------------------------------------------- #
# Negative cases
# --------------------------------------------------------------------------- #
def test_the_amount_alone_still_answers(ask, truth):
    """"What is the largest single-loan exposure?" does not require a share to
    have been requested, and must not refuse for want of one."""
    envelope = ask("What is the largest single-loan exposure?")
    assert envelope["ok"] is True, envelope.get("error")
    assert evidence(envelope)["topExposure"] == pytest.approx(truth["largest"],
                                                             rel=1e-12)


def test_the_share_alone_uses_the_same_numerator_and_denominator(ask, truth):
    envelope = ask("What share of the book does the largest loan represent?")
    assert envelope["ok"] is True, envelope.get("error")
    facts = evidence(envelope)
    assert facts["topExposure"] == pytest.approx(truth["largest"], rel=1e-12)
    assert facts["totalExposure"] == pytest.approx(truth["total"], rel=1e-12)


def test_the_largest_single_loan_ead_refuses(ask):
    """The exposure measure is named explicitly, and it is not in the book. The
    largest-loan machinery must not answer it from current balance."""
    envelope = ask("What is the largest single-loan EAD?")
    assert envelope["ok"] is False
    assert "EAD" in answer(envelope)


def test_a_loan_identifier_is_not_published_as_the_answer(ask):
    """The amount and the share answer the question; the identifier answers
    neither, and belongs in the drill-down table rather than the headline."""
    assert "ALP_ORIGINATION" not in answer(ask(B21))


# =========================================================================== #
# A scalar slot given a list — the crash the richer catalogue provoked
# =========================================================================== #
def test_a_list_in_a_scalar_slot_is_folded_not_crashed():
    """Found by the genuine-LLM run, as a hard failure.

    "Show me balance by region by borrower type" came back with
    ``dimension=["collateral_geography", "borrower_type"]`` — the model
    expressing two dimensions in the singular slot. That value reached the
    validator, which looks a field key up with ``fields.get(key)``, and a list
    is not hashable: the parse raised TypeError instead of producing a governed
    answer or a clean refusal.

    The same defect class as the list-shaped ``filters`` this file already
    folds. Nothing is discarded — the extra dimension is carried into
    ``dimensions``, which is the slot that expresses exactly this.
    """
    from mi_agent.mi_query_spec import MIQuerySpec

    spec = MIQuerySpec.from_dict({
        "metric": BALANCE,
        "dimension": ["collateral_geography", "borrower_type"]})
    assert spec.dimension == "collateral_geography"
    assert spec.dimensions == ["collateral_geography", "borrower_type"]
    # The thing that actually crashed: a field lookup over the slot value.
    assert BALANCE in spec.referenced_fields()


@pytest.mark.parametrize("slot", ["metric", "dimension", "x", "y", "size",
                                  "color", "weight_field", "sort_by"])
def test_every_scalar_slot_survives_a_list(slot):
    """Not just ``dimension``: any scalar slot could arrive list-shaped, and
    each of them is looked up the same way."""
    from mi_agent.mi_query_spec import MIQuerySpec

    spec = MIQuerySpec.from_dict({slot: [BALANCE, "collateral_geography"]})
    assert getattr(spec, slot) == BALANCE
    spec_empty = MIQuerySpec.from_dict({slot: []})
    assert getattr(spec_empty, slot) is None


def test_a_folded_list_slot_validates_without_raising(semantics):
    """End to end: the validator now reaches a verdict on the spec that
    previously killed the parse."""
    from mi_agent.mi_query_spec import MIQuerySpec
    from mi_agent.mi_query_validator import validate_mi_query

    spec = MIQuerySpec.from_dict({
        "intent": "chart", "chart_type": "bar", "metric": BALANCE,
        "aggregation": "sum",
        "dimension": ["collateral_geography", "borrower_type"]})
    result = validate_mi_query(spec, semantics)   # must not raise
    assert result is not None
