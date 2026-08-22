"""D6 (B14) — "available" is asked of the BOOK, never of the loaded projection.

`resolve_active_view` is a SUBSTRING TEST: the word "forecast" anywhere in the
question selects the forecast view, before the question is parsed and before any
route claims it. That view is a derived frame of **twelve** columns where the
book carries **seventy-six**, and every availability check read whichever frame
had been loaded. So:

    "What is the forecast run rate for the front book?"
      -> "front book — field is unavailable in this dataset"

true of the projection, false of the book, which carries `seasoning_segment` for
all 11,035 loans. Its sibling asked about REGION — one of the twelve the
projection kept — already said the true thing.

The boundary these tests exist to hold: `book_columns` answers a SCHEMA question
and never a VALUE question. `geographic_values` and `dimension_values` must keep
reading the LOADED frame, because a value can only be recognised in rows that
exist. Confusing the two turns D6 into B19 and takes the wrong-number class with
it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mi_agent import execution_receipt as R
from mi_agent import mi_calibration as CAL
from mi_agent.mi_query_validator import load_mi_semantics


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
         / "platform" / "alderbridge" / "2026-06-30"
         / "platform_canonical_typed.csv")


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(
        _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")


@pytest.fixture(scope="module")
def book():
    if not _TAPE.exists():
        pytest.skip("calibration book not built")
    return CAL.default_bank_frame(_TAPE)


# --------------------------------------------------------------------------- #
# The accessor
# --------------------------------------------------------------------------- #
def test_an_unstamped_frame_is_its_own_book():
    """The funded view. Nothing changes there, which is why 610 of 697 corpus
    questions cannot move."""
    frame = pd.DataFrame({"a": [1], "b": [2]})
    assert R.book_columns(frame) == {"a", "b"}


def test_a_stamped_frame_reports_the_book_it_was_projected_from():
    frame = pd.DataFrame({"a": [1]})
    frame.attrs["book_columns"] = ["a", "b", "seasoning_segment"]
    assert R.book_columns(frame) == {"a", "b", "seasoning_segment"}


def test_an_empty_frame_does_not_raise():
    """A pandas Index has no truth value, and the routed guard's blanket except
    would have swallowed the ValueError along with the entire receipt. Found by
    running it, not by reading it."""
    assert R.book_columns(pd.DataFrame()) == set()
    assert R.book_columns(None) == set()


# --------------------------------------------------------------------------- #
# The projection is stamped where it is built
# --------------------------------------------------------------------------- #
def test_the_forecast_projection_carries_the_books_schema():
    """The schema was already an argument to the function that drops it —
    `build_forecast_view_frame(funded_df, pipeline_df)`. Carried, not re-loaded."""
    from mi_agent_api import datasets as ds
    from mi_agent_api import workspace as ws

    funded = pd.DataFrame({
        "current_outstanding_balance": [1.0, 2.0],
        "collateral_geography": ["London", "Wales"],
        "seasoning_segment": ["Front Book", "Back Book"],
        "vintage_year": [2024, 2019]})
    frame = ws.build_forecast_view_frame(funded, None)
    assert "seasoning_segment" not in frame.columns, (
        "the projection is expected to DROP it; that is the premise")
    ds._stamp_book_columns(frame, funded)
    assert "seasoning_segment" in R.book_columns(frame)
    assert "vintage_year" in R.book_columns(frame)


def test_the_view_resolver_actually_stamps_it():
    """THE WIRING, not the helper.

    Written after mutating the call site away and watching the test above pass:
    it drove `_stamp_book_columns` directly, so it proved the stamp works and not
    that anything applies it. A helper test is not a wiring test.
    """
    import os

    if not _TAPE.exists():
        pytest.skip("calibration book not built")
    from demo_platform import config as cfg

    os.environ.update(cfg.mi_env(period_role="current"))
    from mi_agent_api import datasets as ds

    frame, error = ds._resolve_query_frame("forecast", None)
    if frame is None:
        pytest.skip("no forecast view on this book: %s" % (error,))
    assert "seasoning_segment" not in set(frame.columns)
    assert "seasoning_segment" in R.book_columns(frame), (
        "the forecast view is not carrying the book's schema, so every "
        "availability check downstream is reading the projection again")
    assert len(R.book_columns(frame)) > len(set(frame.columns))


# --------------------------------------------------------------------------- #
# THE BOUNDARY — schema sites read the book, value sites read the frame
# --------------------------------------------------------------------------- #
def test_value_readers_still_read_the_loaded_frame(semantics, book):
    """B19's boundary, asserted so D6 cannot quietly cross it.

    A value can only be recognised in rows that exist. A projection carrying no
    `account_status` column recognises no value of it — which is why "forecast
    run rate for active loans" still answers over the whole book, and why that
    is B19 rather than something this commit closed.
    """
    projection = pd.DataFrame({
        "current_outstanding_balance": [1.0],
        "collateral_geography": ["London"]})
    projection.attrs["book_columns"] = list(book.columns)

    # The SCHEMA answer knows about the book...
    assert "account_status" in R.book_columns(projection)
    # ...and the VALUE readers do not, because the rows are not there.
    assert R.dimension_values(projection, semantics).get("active") is None
    assert R.geographic_values(projection, semantics).get("london") \
        == "collateral_geography"
    # RECORDED, because mutation says so: moving these readers onto
    # `book_columns` is INERT — the column lookup that follows raises and the
    # scan skips it, so the property holds two ways rather than one. Kept as the
    # statement of the boundary, not as its only enforcement.


def test_a_field_absent_from_the_book_is_still_unavailable(semantics):
    """The can-fail. A change that simply stopped saying UNAVAILABLE would
    satisfy every test above and destroy the one true use of it — this tape
    carries no `broker_channel`, and a reader asking for it must be told so."""
    facet = R.RequestedFacet(kind=R.KIND_GROUPING, label="broker",
                             field_key="broker_channel")

    class _Result:
        metadata = {"reconciliation": {}}
        data = None

    out = R.reconcile_facets(
        [facet], spec=type("S", (), {"filters": {}, "dimensions": []})(),
        query_result=_Result(), semantics=semantics,
        available_columns={"current_outstanding_balance", "collateral_geography"})
    assert out[0].status == R.UNAVAILABLE
    assert "unavailable in this dataset" in out[0].reason


def test_a_field_the_book_carries_is_lost_not_unavailable(semantics):
    """The whole of D6 in one assertion. Both refuse; only one is honest."""
    facet = R.RequestedFacet(kind=R.KIND_GROUPING, label="vintage",
                             field_key="vintage_year")

    class _Result:
        metadata = {"reconciliation": {}}
        data = None

    out = R.reconcile_facets(
        [facet], spec=type("S", (), {"filters": {}, "dimensions": []})(),
        query_result=_Result(), semantics=semantics,
        available_columns={"current_outstanding_balance", "vintage_year"})
    assert out[0].status == R.LOST
    assert "unavailable" not in out[0].reason


# --------------------------------------------------------------------------- #
# The two owners must ask the same question
# --------------------------------------------------------------------------- #
def test_the_seasoning_owner_and_the_dimension_reader_agree_on_a_projection(
        semantics, book):
    """Found DURING implementation, not by inspection, and it was a wrong number.

    `requested_dimension_terms` was moved onto the book's schema and the
    seasoning owner inside `detect_requested_facets` was still reading the frame.
    One suppressed the grouping because the owner had taken the phrase; the other
    raised no population because the projection lacks the column. The field left
    the receipt entirely and a narrowed question returned the whole-book run rate
    with NOTHING recorded — worse than the false message D6 set out to fix.
    """
    projection = pd.DataFrame({
        "current_outstanding_balance": [1.0],
        "collateral_geography": ["London"]})
    projection.attrs["book_columns"] = list(book.columns)
    question = "what is the forecast run rate for the front book"

    facets = R.detect_requested_facets(
        question, semantics, frame=projection,
        requested_dimensions=R.requested_dimension_terms(
            question, semantics, sorted(R.book_columns(projection))))
    seasoning = [(f.kind, f.field_key) for f in facets
                 if f.field_key == "seasoning_segment"]
    assert seasoning == [(R.KIND_POPULATION, "seasoning_segment")], (
        "the seasoning field must reach the receipt exactly once, as the "
        "population the owner resolved: %s" % (facets,))
