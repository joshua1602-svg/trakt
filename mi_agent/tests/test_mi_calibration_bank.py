#!/usr/bin/env python3
"""tests/test_mi_calibration_bank.py

Enforces the curated MI calibration bank
(``config/mi/golden_questions/ere_mi_calibration_250.yaml``): 250+ realistic
business-user questions, each with declared EXPECTED semantic behaviour, run
through the REAL deterministic MI path (parser -> executor -> adapter) with
per-case checks (metric / dimensions applied / filters applied / dimension &
filter invariants / artifact type / reconciliation / required columns; and for
refuse/clarify cases that no narrower/unfiltered answer is returned and a reason
is surfaced).

This is ADDITIONAL to the generated registry harness
(``test_mi_query_invariants.py``) — both run.

A case flagged ``known_gap`` states the IDEAL expectation and is xfailed with the
gap's reason (never loosened). A NON-known-gap case that fails is a hard failure
and breaks the build.

Run:  python -m pytest mi_agent/tests/test_mi_calibration_bank.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent import mi_calibration as CAL
from mi_agent.mi_query_harness import build_fixture

_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
_CASES = CAL.load_bank()


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_SEMANTICS)


@pytest.fixture(scope="module")
def df():
    return build_fixture()


@pytest.fixture(scope="module")
def bank(df, semantics):
    results = [CAL.evaluate_case(c, df, semantics) for c in _CASES]
    return results, CAL.summarise_bank(results)


# --------------------------------------------------------------------------- #
# Bank shape
# --------------------------------------------------------------------------- #
def test_bank_has_at_least_250_curated_questions():
    assert len(_CASES) >= 250, len(_CASES)


def test_bank_covers_all_ten_categories():
    cats = {c["category"] for c in _CASES}
    for expected in ("basic_kpi", "single_dim", "two_dim", "filtered", "ranking",
                     "pipeline", "forecast", "risk", "data_quality",
                     "unsupported", "ambiguous"):
        assert expected in cats, expected


def test_every_case_has_required_metadata():
    for c in _CASES:
        for key in ("id", "category", "question", "expected_status",
                    "expected_artifact_type", "expected_scope"):
            assert key in c, (c.get("id"), key)
        assert c["expected_status"] in ("answer", "refuse", "clarify")


# --------------------------------------------------------------------------- #
# The headline guarantee: NO hard failures (every non-known-gap case holds its
# declared expectation).
# --------------------------------------------------------------------------- #
def test_no_hard_failures(bank):
    results, summary = bank
    hard = [(r.id, r.question, r.failures) for r in results
            if not r.ok and not r.known_gap]
    assert not hard, (
        f"{len(hard)} curated cases failed their declared expectation:\n"
        + "\n".join(f"  {i}: {q} :: {f}" for i, q, f in hard))


def test_every_category_has_zero_hard_failures(bank):
    results, _ = bank
    by_cat: dict = {}
    for r in results:
        if not r.ok and not r.known_gap:
            by_cat.setdefault(r.category, []).append(r.id)
    assert not by_cat, by_cat


# --------------------------------------------------------------------------- #
# Per-case parametrised check (known_gap cases xfail with their reason).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_case(case, df, semantics):
    r = CAL.evaluate_case(case, df, semantics)
    if case.get("known_gap"):
        if not r.ok:
            pytest.xfail(f"known gap: {case['known_gap']}")
        # A known_gap that now passes is fine (the gap may have been fixed) — it
        # simply stops xfailing; no assertion needed.
        return
    assert r.ok, f"{case['id']} ({case['question']}): {r.failures}"


# --------------------------------------------------------------------------- #
# Refuse/clarify cases genuinely fail closed (no data artifact, reason present).
# --------------------------------------------------------------------------- #
def test_refuse_and_clarify_cases_are_fail_closed(bank):
    results, _ = bank
    for r in results:
        if r.status in ("refuse", "clarify") and not r.known_gap:
            assert r.ok, (r.id, r.question, r.failures)
            assert not any(t == "kpi" or t == "table" or t.startswith("chart:")
                           for t in r.observed_artifacts), (r.id, r.observed_artifacts)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
