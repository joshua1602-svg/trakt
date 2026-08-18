#!/usr/bin/env python3
"""tests/test_p1g_measure_identity.py — measure SEMANTIC IDENTITY.

The second half of P1G. Cohort identity proved that the cohorts compared are
the cohorts asked for; this proves the same of the MEASURE:

    every materially requested measure named in the question
      -> the governed resolved measure set
        -> the executed measure set
          -> the receipt

or is explicitly classified as unavailable, unsupported or rejected.

A comparison route executing other perfectly valid measures is not evidence
that the requested one was answered. The failure this exists to stop:

    "How does the direct book compare with the acquired book on BORROWER AGE?"
    -> "Direct has higher observed Current Outstanding Balance than Acquired.
        Direct has higher observed Loan Count than Acquired."

Age is never mentioned. Both figures are correct, and neither is the answer.

The hole was narrow and exact: the substitution check read the spec's SINGULAR
``metric``. A P1E measure-set spec carries ``metric=None``, so there was nothing
to compare against and the check returned None — which is why this defect was
intermittent rather than constant, and why a single sampled run called it safe.

Run: python -m pytest tests/test_p1g_measure_identity.py
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
LTV = "current_loan_to_value"
AGE = "youngest_borrower_age"
ROUTE = "portfolio_risk_comparison"

AGE_QUESTION = ("How does the direct book compare with the acquired book on "
                "borrower age?")
LTV_QUESTION = ("What is the average LTV of the direct book vs the acquired "
                "book?")


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


def answer(envelope) -> str:
    return envelope.get("answer") or envelope.get("error") or ""


def _substitution(question, executed_fields, *, metric=None, route=ROUTE):
    """The guard's verdict for a question answered with ``executed_fields``."""
    compared = {"measuresCompared": list(executed_fields)}
    return er.detect_measure_substitution(
        question, route=route, metric_key=metric,
        executed_concepts=er.comparison_measure_concepts(compared))


# =========================================================================== #
# Adversarial: the requested measure was dropped
# =========================================================================== #
def test_age_requested_balance_and_count_executed_is_rejected():
    """The exact B25 failure, in the shape that made it intermittent: a measure
    set with no singular metric."""
    verdict = _substitution(AGE_QUESTION, [BALANCE, "loan_count"], metric=None)
    assert verdict is not None
    assert "age" in verdict


def test_ltv_requested_balance_executed_is_rejected():
    verdict = _substitution(LTV_QUESTION, [BALANCE], metric=None)
    assert verdict is not None
    assert "ltv" in verdict


def test_a_neighbouring_measure_does_not_satisfy_the_request():
    """Interest rate is a governed measure and a reasonable thing to compare.
    It is not borrower age."""
    verdict = _substitution(AGE_QUESTION, ["current_interest_rate"], metric=None)
    assert verdict is not None


@pytest.mark.parametrize("executed", [
    [BALANCE], [BALANCE, "loan_count"], [LTV], [LTV, BALANCE],
    ["current_interest_rate"], ["arrears_balance"],
])
def test_no_other_measure_set_can_stand_in_for_age(executed):
    """Generic, not a B25 special case: whatever else executed, if age did not,
    the age question is not answered."""
    assert _substitution(AGE_QUESTION, executed, metric=None) is not None


# =========================================================================== #
# The requested measure DID execute
# =========================================================================== #
def test_age_requested_and_age_executed_is_accepted():
    assert _substitution(AGE_QUESTION, [AGE], metric=None) is None


def test_ltv_requested_and_ltv_executed_is_accepted():
    assert _substitution(LTV_QUESTION, [LTV], metric=None) is None


def test_the_requested_measure_among_others_is_accepted():
    """A multi-measure comparison that includes the requested one has answered
    it; the others are additional context, not a substitution."""
    assert _substitution(
        AGE_QUESTION, [BALANCE, "loan_count", LTV, AGE], metric=None) is None


def test_a_question_naming_no_measure_is_never_refused_on_this_basis():
    """Scope control. "Where is my portfolio most concentrated" names no
    measure, so any governed measure answers it."""
    assert _substitution("Where is my portfolio most concentrated?",
                         [BALANCE], metric=None, route=None) is None


def test_the_singular_metric_slot_still_works():
    """Backward compatibility: a pre-P1E spec carries metric and no set."""
    assert _substitution(AGE_QUESTION, [], metric=BALANCE) is not None
    assert _substitution(AGE_QUESTION, [], metric=AGE) is None


# =========================================================================== #
# End to end, governed path
# =========================================================================== #
def test_the_governed_age_comparison_is_correct(ask):
    """The target outcome: correct, not merely safe. The governed comparison
    exists and reconciles, so the question should be ANSWERED."""
    envelope = ask(AGE_QUESTION)
    assert envelope["ok"] is True, envelope.get("error")
    text = answer(envelope)
    assert "Youngest Borrower Age" in text
    declared = (envelope.get("metadata") or {}).get("portfolioComparison") or {}
    assert AGE in (declared.get("measuresCompared") or [])


def test_the_governed_ltv_comparison_is_correct(ask):
    envelope = ask(LTV_QUESTION)
    assert envelope["ok"] is True, envelope.get("error")
    assert "Loan To Value" in answer(envelope)


def test_a_multi_measure_comparison_accounts_for_every_requested_measure(ask):
    """P1E policy, unchanged: every requested measure is compared or named."""
    envelope = ask("Compare the direct and acquired books on balance, loan "
                   "count, weighted-average LTV and average borrower age.")
    assert envelope["ok"] is True, envelope.get("error")
    text = answer(envelope)
    for measure in ("Current Outstanding Balance", "Loan Count",
                    "Current Loan To Value", "Youngest Borrower Age"):
        assert measure in text, f"{measure} unaccounted for"


def test_an_unavailable_requested_measure_refuses_without_substituting(ask):
    """No neighbouring measure is offered in its place.

    "Credit score" resolves to the governed internal risk score, which the
    registry marks originator-specific and therefore not comparable across
    books. The refusal names the measure it could not compare and says plainly
    that it did not compare a different one — which is the property under test,
    not the particular wording the question happened to use.
    """
    envelope = ask("How does the direct book compare with the acquired book "
                   "on credit score?")
    text = answer(envelope).lower()
    assert envelope["ok"] is False, f"answered anyway: {text[:200]}"
    assert "score" in text, "the refusal does not name the measure"
    assert "not compared a different measure" in text
    # And no substitute figure was smuggled in alongside the refusal.
    for neighbour in ("outstanding balance", "loan to value", "borrower age"):
        assert neighbour not in text, f"offered {neighbour} instead"


# =========================================================================== #
# The two halves of P1G compose
# =========================================================================== #
def test_measure_identity_does_not_weaken_cohort_identity(ask):
    """A vintage question stays refused; measure identity must not hand it an
    escape route by accepting a balance comparison."""
    envelope = ask("Is the credit quality of new origination better or worse "
                   "than the back book?")
    assert envelope["ok"] is False
    assert "how long the loans have been on the book" in answer(envelope).lower()


def test_both_identities_are_subject_facets_that_refuse():
    assert er.KIND_COHORT_COMPARISON in er.NUMBER_OR_SUBJECT_FACETS
    assert er.KIND_MULTI_MEASURE in er.NUMBER_OR_SUBJECT_FACETS
