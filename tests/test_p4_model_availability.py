#!/usr/bin/env python3
"""Model availability may decide WHETHER Trakt answers. Never WHAT it answers.

The augmentation arm reads the reader's language; the governed estate keeps
every other decision. When the arm's call does not happen, the deterministic
reading that remains may be narrower than the sentence — and executing it
anyway publishes an answer to a question nobody asked, with nothing in the
envelope to say so.

Measured on this build, before the rule below existed: with the arm switched on
and the credit exhausted, twenty of twenty runs of one product-scoped question
returned a whole-book answer.

What is asserted here is the state machine, at the boundary:

    call raised (provider error, usage limit, timeout, unreachable)  -> UNAVAILABLE
    reply unreadable (not JSON, wrong shape)                         -> UNAVAILABLE
    arm could not be built or run at all                             -> UNAVAILABLE
    call succeeded and proposed nothing                              -> no_change
    call succeeded and proposed something                            -> applied

and that only the first three refuse. The fourth is a real answer about the
reader's sentence and is not a failure; inferring one from the other — reading
`[]` as "the model did not answer" — is the confusion this file exists to
prevent.

No test here consumes credit. Every one injects at the model call itself.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import llm_query_parser as LQ
from mi_agent_api import concept_merge_arm as ARM
from mi_agent_api.app import app

client = TestClient(app)


def _write_run(root: Path, run_id: str, reporting_date: str, n: int) -> None:
    rng = np.random.default_rng(abs(hash(run_id)) % (2**32))
    df = pd.DataFrame({
        "loan_identifier": [f"{run_id}_{i}" for i in range(n)],
        "current_outstanding_balance": rng.uniform(120_000, 280_000, n).round(2),
        "current_loan_to_value": rng.uniform(20, 55, n).round(1),
        "current_interest_rate": rng.uniform(3, 8, n).round(2),
        "youngest_borrower_age": rng.integers(62, 88, n),
        "geographic_region_obligor": rng.choice(
            ["London", "South East", "Scotland", "Wales", "East"], n),
        # THE NARROWING THE SENTENCE ASKS FOR. Half the book is one product and
        # half the other, so an answer that loses the product is not merely
        # imprecise — it reports twice the population.
        "erm_product_type": ["drawdown" if i % 2 else "lump_sum" for i in range(n)],
        "reporting_date": [reporting_date] * n,
    })
    d = root / "client_001" / run_id / "output" / "central"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "18_central_lender_tape.csv", index=False)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    warnings.simplefilter("ignore")
    monkeypatch.chdir(_REPO_ROOT)
    root = tmp_path / "onboarding_output"
    _write_run(root, "mi_2025_11", "2025-11-30", 80)
    monkeypatch.setenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", str(root))
    # The FREE-FORM parser stays off throughout: this is the augmentation arm's
    # state machine, and running both would measure their sum.
    monkeypatch.setenv("MI_AGENT_LLM_PARSER", "off")
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "false")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-used")
    yield


def _arm_on(monkeypatch):
    monkeypatch.setenv("MI_AGENT_CONCEPT_MERGE", "on")
    assert ARM.enabled() is True, "the arm must be ON for these tests to mean anything"


def _arm_off(monkeypatch):
    monkeypatch.setenv("MI_AGENT_CONCEPT_MERGE", "off")
    assert ARM.enabled() is False


#: A question with a governed narrowing the deterministic parser does not carry.
#: Coverage sees the loss and refuses it on its own.
_NARROWED = "Break drawdown balance down by region."
#: A question every owner names and execution carries in full. Coverage has
#: nothing to say about it, so anything that refuses it refused for another
#: reason — which is what makes it the right probe for the availability rule.
_CLEAN = "Show total funded balance by region."


def _ask(question: str = _NARROWED) -> dict:
    return client.post("/mi/query", json={
        "question": question, "portfolioId": "client_001/mi_2025_11",
        "datasetContext": "funded", "asOfDate": "2025-11-30",
    }).json()


def _evidence(resp: dict):
    return (resp.get("metadata") or {}).get("conceptMerge")


def _coverage_clean(resp: dict) -> bool:
    cov = (resp.get("metadata") or {}).get("semanticCoverage") or {}
    return not (cov.get("unaccounted") or [])


def _rows(resp: dict) -> int:
    return max([len(a.get("rows") or []) for a in (resp.get("artifacts") or [])] or [0])


def _raises(exc):
    def _call(prompt, model, use_cache=True):
        raise exc
    return _call


def _returns(text):
    def _call(prompt, model, use_cache=True):
        return text, {"input_tokens": 1, "output_tokens": 1}, False
    return _call


# --------------------------------------------------------------------------- #
# A. The unavailable states. Every one of them refuses.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("exc, label", [
    (RuntimeError("rate_limit_error: usage limit reached"), "usage limit"),
    (RuntimeError("Error code: 529 - overloaded_error"), "provider error"),
    (TimeoutError("Request timed out."), "timeout"),
    (ConnectionError("Failed to establish a new connection"), "unreachable"),
])
def test_a_failed_call_is_an_availability_state_and_refuses(monkeypatch, exc, label):
    """Asked of a question the DETERMINISTIC estate answers perfectly.

    That is the point. Coverage cannot flag this sentence — every concept in it
    is named and carried — so the only thing that can refuse it is the arm's own
    availability status. If the rule were merely coverage under another name it
    would answer here, and the case the whole rule exists for is exactly the one
    coverage cannot see.
    """
    _arm_on(monkeypatch)
    monkeypatch.setattr(LQ, "_call_llm", _raises(exc))
    r = _ask(_CLEAN)
    assert _evidence(r)["status"] == ARM.PROPOSAL_UNAVAILABLE, label
    assert r["ok"] is False, f"{label} answered anyway"
    assert r.get("controlledRefusal") is True
    assert not r.get("artifacts")
    assert _coverage_clean(r), "this question must be coverage-clean to prove the point"


@pytest.mark.parametrize("text, label", [
    ("I think you want the drawdown product.", "prose, not JSON"),
    ("{'concepts': []}", "not JSON"),
    ('{"proposals": []}', "no concepts key"),
    ('{"concepts": {"kind": "population"}}', "concepts is not a list"),
    ('{"concepts": [{"kind": "population"}]}', "a concept has no term"),
])
def test_a_reply_that_cannot_be_read_is_unavailable_not_a_proposal_of_nothing(
        monkeypatch, text, label):
    _arm_on(monkeypatch)
    monkeypatch.setattr(LQ, "_call_llm", _returns(text))
    r = _ask(_CLEAN)
    assert _evidence(r)["status"] == ARM.PROPOSAL_UNAVAILABLE, label
    assert r["ok"] is False, f"{label} answered anyway"


# --------------------------------------------------------------------------- #
# B. The successful states. Neither of them refuses.
# --------------------------------------------------------------------------- #
def test_a_successful_empty_proposal_is_not_an_availability_failure(monkeypatch):
    """THE DISTINCTION THE WHOLE RULE TURNS ON.

    A model that read the sentence and had nothing to add is not a model that
    could not be reached. It reports `no_change`, the answer stands, and no
    refusal is owed — which is why availability is read from the call's own
    status and never from the length of the proposal list.
    """
    _arm_on(monkeypatch)
    monkeypatch.setattr(LQ, "_call_llm", _returns('{"concepts": []}'))
    r = _ask("Show total funded balance by region.")
    ev = _evidence(r)
    assert ev["status"] == "no_change"
    assert ev["proposed"] == []
    assert ev["status"] != ARM.PROPOSAL_UNAVAILABLE
    assert r["ok"] is True, "a successful empty proposal must not refuse"


def test_a_successful_proposal_reaches_the_contract(monkeypatch):
    _arm_on(monkeypatch)
    monkeypatch.setattr(LQ, "_call_llm", _returns(
        '{"concepts": [{"kind": "category_value", "term": "drawdown", '
        '"covers": "drawdown"}]}'))
    r = _ask()
    ev = _evidence(r)
    assert ev["status"] == "applied", ev.get("rejected")
    assert ev["proposed"] and ev["proposed"][0]["term"] == "drawdown"
    assert ev["applied"], "a bound proposal reached no slot"
    # AND IT NARROWED. The concept the deterministic parse lost is now carried,
    # coverage is clean, and the answer covers half the book rather than all of
    # it — which is the reach the arm exists to add.
    assert r["ok"] is True
    assert _coverage_clean(r)
    assert (r.get("spec") or {}).get("filters", {}).get("erm_product_type")


def test_the_arm_being_switched_off_is_not_an_availability_failure(monkeypatch):
    """An arm that was never asked cannot have failed to answer. It publishes no
    evidence, and the deterministic estate answers exactly as it always did."""
    _arm_off(monkeypatch)
    r = _ask("Show total funded balance by region.")
    assert _evidence(r) is None
    assert r["ok"] is True


# --------------------------------------------------------------------------- #
# C. The property the rule exists for
# --------------------------------------------------------------------------- #
def test_an_unavailable_model_never_widens_a_product_scoped_question(monkeypatch):
    """The whole book is half drawdown. Losing the product doubles the answer.

    The two guards overlap here and that is the design: coverage names the lost
    concept, availability names the lost call, and a question carrying both is
    refused by the first one to see it. What is asserted is the property, not
    which guard delivered it — no rows, for either reason.
    """
    _arm_on(monkeypatch)
    monkeypatch.setattr(LQ, "_call_llm", _raises(RuntimeError("usage limit")))
    r = _ask(_NARROWED)
    assert _evidence(r)["status"] == ARM.PROPOSAL_UNAVAILABLE
    assert r["ok"] is False
    assert _rows(r) == 0, "an unavailable model published rows for a lost narrowing"


def test_an_unavailable_model_refuses_the_easy_questions_too(monkeypatch):
    """No question is exempt, and none is exempted by being easy.

    A deterministic reading that happens to be complete cannot be told apart
    from one that is narrower than the sentence, because the only instruments
    that could tell them apart — the coverage ledger and the execution receipt —
    are built from the same owners that produced the reading. Until an
    independent proof exists, every one of these refuses.
    """
    _arm_on(monkeypatch)
    monkeypatch.setattr(LQ, "_call_llm", _raises(RuntimeError("usage limit")))
    for q in (_CLEAN, "Show total funded balance.", "How many loans are there?",
              "Show weighted-average LTV by region.", _NARROWED):
        r = _ask(q)
        assert r["ok"] is False, q
        assert _rows(r) == 0, q


def test_availability_is_read_from_the_call_not_from_an_empty_list(monkeypatch):
    """The two states are distinguishable at the arm's own boundary, with the
    envelope out of the picture: one produced a proposal list of length zero and
    the other produced no proposal at all."""
    _arm_on(monkeypatch)
    from question_interpretation import claim_merge as CM

    class _Spec:
        filters, dimensions, dimension, metric = {}, [], None, None

    def _run(call):
        monkeypatch.setattr(LQ, "_call_llm", call)
        return ARM.apply("Break drawdown balance down by region.", _Spec(), {},
                         interpretation=CM.Interpretation()
                         if hasattr(CM, "Interpretation") else object())

    empty = _run(_returns('{"concepts": []}'))
    failed = _run(_raises(RuntimeError("usage limit")))
    assert failed["status"] == ARM.PROPOSAL_UNAVAILABLE
    assert "proposed" not in failed, "an unavailable call must publish no proposals"
    assert empty["status"] != ARM.PROPOSAL_UNAVAILABLE
    assert empty["proposed"] == []
