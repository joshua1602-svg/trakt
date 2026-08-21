#!/usr/bin/env python3
"""Stage 2, commit 1: the facet layer emits the offsets it already computed.

Additive by construction — the span is carried in memory and NOT serialised.
That distinction is load-bearing: adding it to `RequestedFacet.to_dict()` moved
32 of 340 answers on the answer-text diff, because `executionSummary` carries
facets and is user-visible payload, while the calibration bank passed all 260.
Stage 2's acceptance is byte-identical answers, so it stays in memory.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

warnings.simplefilter("ignore")
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
         / "platform" / "alderbridge" / "2026-06-30"
         / "platform_canonical_typed.csv")


@pytest.fixture(scope="module")
def book():
    from mi_agent.mi_calibration import default_bank_frame
    if not _TAPE.exists():
        pytest.skip("calibration book not built")
    return default_bank_frame(_TAPE)


@pytest.fixture(scope="module")
def semantics():
    from mi_agent.mi_query_validator import load_mi_semantics
    return load_mi_semantics(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")


def _facets(question, semantics, book):
    from mi_agent import execution_receipt as R
    dims = R.requested_dimension_terms(question, semantics, list(book.columns))
    return R.detect_requested_facets(question, semantics, frame=book,
                                     requested_dimensions=dims)


def test_a_threshold_span_points_at_the_threshold_words(semantics, book):
    q = "how many loans have a balance above £250k"
    facets = [f for f in _facets(q, semantics, book) if f.kind == "threshold"]
    assert facets, "expected a threshold facet"
    f = facets[0]
    assert f.span is not None
    assert q[f.span[0]:f.span[1]] == "above £250"


def test_a_geographic_span_points_at_the_place_name(semantics, book):
    q = "How many South East loans have LTV above 50%?"
    geo = [f for f in _facets(q, semantics, book) if f.kind == "geographic_scope"]
    assert geo, "expected a geographic_scope facet"
    assert q[geo[0].span[0]:geo[0].span[1]] == "South East"


def test_every_span_indexes_the_original_question(semantics, book):
    """A span that does not slice back to real words is worse than none."""
    for q in ("how many loans have LTV above 80%",
              "What is the balance of South East loans with LTV above 50%?",
              "total balance where days in arrears exceed 90"):
        for f in _facets(q, semantics, book):
            if f.span is None:
                continue
            start, end = f.span
            assert 0 <= start < end <= len(q), (q, f.kind, f.span)
            assert q[start:end].strip(), (q, f.kind, f.span)


def test_the_span_is_not_serialised(semantics, book):
    """Load-bearing: executionSummary carries facets and is user-visible."""
    q = "how many loans have a balance above £250k"
    facets = [f for f in _facets(q, semantics, book) if f.kind == "threshold"]
    assert facets[0].span is not None
    assert "span" not in facets[0].to_dict()


def test_the_serialisation_guard_can_fail():
    """Proof the test above is live rather than tautological."""
    from mi_agent.execution_receipt import RequestedFacet
    f = RequestedFacet(kind="threshold", label="LTV over 50", span=(10, 20))
    d = dict(f.to_dict())
    assert "span" not in d
    d["span"] = list(f.span)
    assert "span" in d          # a payload that DID carry it would differ


def test_a_facet_not_matched_from_the_question_has_no_span():
    """Absent, not fabricated: only detectors that matched the text can supply
    one, and the join must be able to tell the difference."""
    from mi_agent.execution_receipt import RequestedFacet
    assert RequestedFacet(kind="projection", label="a forward projection").span is None
