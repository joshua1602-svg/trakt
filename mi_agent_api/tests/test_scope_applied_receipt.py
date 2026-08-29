#!/usr/bin/env python3
"""A route that narrows to a book must SAY that it narrowed to it.

    "Summarise the month-on-month movement in the Direct book"   £22.6m
    "What changed in the Direct portfolio since last month?"     £12.4m

Both answered the same question about the same book on the same tape. The first
was wrong by £10.2m — it reported the whole portfolio's movement — and the two
envelopes were IDENTICAL in every field a consumer can read: same
``portfolioScope.context_id: "direct"``, same absent population, same absent
narrowing, same absent facet. A wrong figure vouched for by its own receipt.

Two causes, and this module holds both shut.

ONE. `contract_scope.lens_from_contract` returned `portfolio_lens._type_lens`,
whose filters are ``{source_portfolio_type: "direct"}``. Every consumer that
narrows a frame filters on ``source_portfolio_id``. The contract lens therefore
named a book and carried nothing to narrow BY.

TWO. `chat_routing._apply_lens_filter` stated that precondition in its
docstring — "``_resolve_lens`` returns the registry-resolved id list (never a
type string)" — and enforced nothing. Handed a lens with no ids it returned the
frame whole. Five snapshots went in at 520/545/570/600/640 rows and came out at
520/545/570/600/640.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mi_agent import portfolio_lens as PL
from mi_agent_api import chat_routing as CR
from mi_agent_api import evolution as EV


@pytest.fixture()
def two_books():
    return pd.DataFrame({
        "source_portfolio_id": ["direct_001", "direct_001", "acquired_001"],
        "source_portfolio_type": ["direct", "direct", "acquired"],
        "current_outstanding_balance": [100.0, 200.0, 400.0],
    })


# --------------------------------------------------------------------------- #
# The guard
# --------------------------------------------------------------------------- #
def test_a_lens_that_names_a_book_and_carries_no_id_is_refused(two_books):
    """THE DEFECT. A type lens over a multi-portfolio book must not widen."""
    with pytest.raises(CR.LensNotApplied) as exc:
        CR._apply_lens_filter(two_books, PL.lens_from_term("direct"))
    assert "direct" in str(exc.value).lower()


def test_a_scope_the_registry_does_not_hold_is_refused(two_books):
    """`_unresolved_lens` deliberately carries no filters. Returning the frame
    for it answered "concentration for acquired_009" with the whole book."""
    with pytest.raises(CR.LensNotApplied):
        CR._apply_lens_filter(two_books, PL._unresolved_lens("acquired_009"))


def test_total_narrows_nothing_and_is_not_an_error(two_books):
    assert len(CR._apply_lens_filter(two_books, PL.total_lens())) == 3


def test_a_book_without_provenance_is_already_the_scope():
    """One source portfolio, no column to filter on. Unchanged is correct, and
    it is RECORDED as applied because it is."""
    frame = pd.DataFrame({"current_outstanding_balance": [1.0, 2.0]})
    evidence = []
    out = CR._apply_lens_filter(frame, PL.lens_from_term("direct"),
                                evidence_out=evidence)
    assert len(out) == 2
    assert evidence and evidence[0]["rows_before"] == evidence[0]["rows_after"] == 2


# --------------------------------------------------------------------------- #
# The ids, wherever the lens carries them
# --------------------------------------------------------------------------- #
def test_an_explicit_selection_narrows_on_its_cohort_ids(two_books):
    """`_selection_lens` — "exactly those, never their type" — leaves `filters`
    EMPTY and carries its ids in `cohort_ids`. Reading only `filters` treated an
    explicit multi-book selection exactly like the type lens above."""
    evidence = []
    out = CR._apply_lens_filter(two_books, PL._selection_lens(["direct_001"]),
                                evidence_out=evidence)
    assert len(out) == 2
    assert evidence[0]["rows_before"] == 3 and evidence[0]["rows_after"] == 2


def test_a_registry_resolved_lens_narrows_on_its_filters(two_books):
    lens = PL.PortfolioLens("direct", "Direct",
                            {PL.SOURCE_ID_FIELD: ["direct_001"]})
    assert len(CR._apply_lens_filter(two_books, lens)) == 2


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #
def test_the_narrowing_is_recorded_with_both_row_counts(two_books):
    """Without this, a route that narrowed and a route that did not publish the
    same envelope — which is how the £10.2m error survived being measured."""
    evidence = []
    lens = PL.PortfolioLens("direct", "Direct",
                            {PL.SOURCE_ID_FIELD: ["direct_001"]})
    CR._apply_lens_filter(two_books, lens, evidence_out=evidence)
    assert evidence == [{"context": "direct", "label": "Direct",
                         "rows_before": 3, "rows_after": 2,
                         "detail": "direct_001"}]


def test_recording_is_opt_in_and_changes_nothing_when_absent(two_books):
    lens = PL.PortfolioLens("direct", "Direct",
                            {PL.SOURCE_ID_FIELD: ["direct_001"]})
    assert len(CR._apply_lens_filter(two_books, lens)) == 2


def test_the_second_lens_owner_records_the_same_way(two_books):
    """`evolution._scope_frame_lens` is the other place a frame is narrowed to a
    scope — `period_movement` and `portfolio_summary` reach it. It records the
    same fact so a consumer needs one vocabulary, not two."""
    evidence = []
    out = EV._scope_frame_lens(two_books, {PL.SOURCE_ID_FIELD: ["direct_001"]},
                               evidence_out=evidence)
    assert len(out) == 2
    assert evidence[0]["rows_before"] == 3 and evidence[0]["rows_after"] == 2


def test_declare_scope_publishes_nothing_when_nothing_was_narrowed():
    envelope = {}
    CR._declare_scope(envelope, None, label="Total")
    assert "scopeApplied" not in (envelope.get("metadata") or {})


def test_declare_scope_publishes_the_applied_narrowing():
    envelope = {}
    CR._declare_scope(envelope, {"detail": "direct_001", "rowsBefore": 640,
                                 "rowsAfter": 441, "snapshots": 5},
                      context="direct", label="Direct")
    assert envelope["metadata"]["scopeApplied"] == {
        "context": "direct", "label": "Direct", "detail": "direct_001",
        "rowsBefore": 640, "rowsAfter": 441, "snapshots": 5}


# --------------------------------------------------------------------------- #
# The contract lens
# --------------------------------------------------------------------------- #
def test_the_contract_lens_is_resolved_to_ids_not_a_type(monkeypatch):
    """`lens_from_contract` handed a type lens to a filter that reads ids. The
    resolution now happens where the lens is built, so a contract-derived lens
    and a sentence-derived one are the same object by construction."""
    from mi_agent_api import contract_scope as CS

    class _Scope:
        filters = {PL.SOURCE_ID_FIELD: ["direct_001", "direct_002"]}

    class _Resolved:
        scope = _Scope()

    monkeypatch.setattr("mi_agent_api.portfolio_context.resolve_context",
                        lambda *a, **k: _Resolved())
    lens = CS._through_the_registry(PL.lens_from_term("direct"))
    assert lens.filters == {PL.SOURCE_ID_FIELD: ["direct_001", "direct_002"]}
    assert lens.name == "direct"


def test_lens_from_contract_returns_a_lens_a_route_can_actually_apply(monkeypatch, two_books):
    """The end-to-end statement, and the one the defect needed: what the
    CONTRACT hands a route must narrow. Asserting on the helper alone left the
    wiring untested — restoring the bare `lens_from_term` call kept every other
    test in this module green."""
    from mi_agent_api import contract_scope as CS

    class _Scope:
        filters = {PL.SOURCE_ID_FIELD: ["direct_001"]}

    class _Resolved:
        scope = _Scope()

    class _Claim:
        state = "filled"
        scope = "direct"
        portfolio_ids = ()

    class _Interpretation:
        source_scope = _Claim()

    monkeypatch.setattr("mi_agent_api.portfolio_context.resolve_context",
                        lambda *a, **k: _Resolved())
    lens = CS.lens_from_contract(_Interpretation())
    assert len(CR._apply_lens_filter(two_books, lens)) == 2


def test_an_unavailable_registry_leaves_the_lens_unresolved_not_widened(monkeypatch, two_books):
    """Best-effort resolution, exactly as `_resolve_lens` is. What keeps the
    class shut is that the unresolved lens is then REFUSED downstream rather
    than quietly widening."""
    from mi_agent_api import contract_scope as CS

    def _boom(*a, **k):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr("mi_agent_api.portfolio_context.resolve_context", _boom)
    lens = CS._through_the_registry(PL.lens_from_term("direct"))
    assert lens.filters == {PL.SOURCE_TYPE_FIELD: "direct"}
    with pytest.raises(CR.LensNotApplied):
        CR._apply_lens_filter(two_books, lens)
