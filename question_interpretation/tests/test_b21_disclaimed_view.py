"""B21 — a disclaimed view word must not choose the frame.

`resolve_active_view` was a bare substring test, so the clause RULING OUT a view
is what selected it. "What is the balance by vintage, ignoring the forecast?"
loaded the forecast projection — twelve of the funded book's seventy-one columns
— and died at prepared-data validation on `vintage_year`, a column the funded
book carries.

THE EVIDENCE HERE IS PARTLY CONSTRUCTED, AND WHICH PART MATTERS.

The severity argument rests on a coincidence of this tape. `build_forecast_view_frame`
puts the forecast CONTRIBUTION into `current_outstanding_balance` — the same
column name, a different meaning — and with no pipeline the contribution IS the
funded balance, so the two frames agree to the penny. On this book, answering a
funded question on the forecast frame changes no number. On any book carrying a
pipeline it changes every number, silently, under the same field name.

So `test_the_frames_diverge_once_a_pipeline_exists` builds a pipeline rather than
loading one, and says so. Everything else here runs on the real tape.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent import portfolio_lens as lens
from mi_agent_api import chat_routing as routing
from mi_agent_api import workspace as ws


# --------------------------------------------------------------------------- #
# The owner
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question, term, expected", [
    # A disclaimed mention does not select.
    ("What is the balance by vintage, ignoring the forecast?", "forecast", False),
    ("What is the balance by seasoning segment excluding pipeline cases?",
     "pipeline", False),
    ("balance for the whole book, excluding forecast contributions", "forecast", False),
    ("balance net of pipeline", "pipeline", False),
    # One undisclaimed mention is enough. A disclaiming construction DECLINES;
    # it does not select the opposite.
    ("the forecast excluding pipeline", "forecast", True),
    ("the forecast excluding pipeline", "pipeline", False),
    # The word IS the subject — the corpus's own pipeline family.
    ("How much pipeline is overdue?", "pipeline", True),
    ("Which broker has the largest pipeline?", "pipeline", True),
    ("Show forecast balance by region.", "forecast", True),
    ("pipeline amount by stage", "pipeline", True),
    # Substring semantics, unchanged where nothing is disclaimed.
    ("What are the forecasts?", "forecast", True),
    ("How is forecasting done?", "forecast", True),
])
def test_the_undisclaimed_mention_test(question, term, expected):
    assert lens.undisclaimed_mention(question, term) is expected


def test_a_disclaimer_does_not_reach_across_a_sentence_boundary():
    """The gap is bounded and never crosses `.;?!`, so a disclaimer in one
    clause cannot silently rule out a term in the next sentence."""
    assert lens.undisclaimed_mention(
        "Exclude the acquired book. What is the forecast?", "forecast") is True


def test_one_disclaimer_rules_out_every_term_it_reaches():
    """"excluding pipeline cases" disclaims `pipeline` AND `case`, which is why
    the test is applied per term rather than over an alternation."""
    q = "balance by seasoning segment excluding pipeline cases"
    assert lens.undisclaimed_mention(q, "pipeline") is False
    assert lens.undisclaimed_mention(q, "case") is False


# --------------------------------------------------------------------------- #
# Both callers, because the enumeration found two
# --------------------------------------------------------------------------- #
def test_the_view_resolver_reads_it():
    assert ws.resolve_active_view(
        "What is the balance by vintage, ignoring the forecast?", None) == "funded"
    assert ws.resolve_active_view(
        "What is the balance by seasoning segment excluding pipeline cases?",
        None) == "funded"


def test_the_view_resolver_still_selects_a_genuine_view():
    """THE CAN-FAIL. A too-aggressive guard destroys these, and on this book the
    view is the ONLY thing that can show it: the frames agree to the penny, so a
    forecast question wrongly answered on the funded frame returns the identical
    number."""
    assert ws.resolve_active_view("Show forecast balance by region.", None) == "forecast"
    assert ws.resolve_active_view("How much pipeline is overdue?", None) == "pipeline"
    assert ws.resolve_active_view("forecast funded balance", "funded") == "forecast"
    assert ws.resolve_active_view("amount by region", "pipeline") == "pipeline"


def test_the_dataset_resolver_reads_the_same_test():
    """THE SECOND OWNER, found by enumerating where the view decision arrives —
    not by a failing test. `_dataset_for` is on the same governed path, reached
    through the compare and evolution routes, and its word list is WIDER:
    `case`, `kfi`, `application`, `offer`. Fixing only `resolve_active_view`
    would have left the excluding question narrowed to the very dataset it
    excluded."""
    assert routing._dataset_for(
        "What is the balance by seasoning segment excluding pipeline cases?",
        "funded") == "funded"
    # The can-fail for this half.
    assert routing._dataset_for("pipeline amount by stage", "funded") == "pipeline"
    assert routing._dataset_for("How many cases completed?", "funded") == "pipeline"
    assert routing._dataset_for("balance by region", "pipeline") == "pipeline"


def test_the_disclaiming_test_has_one_implementation():
    """One helper, three callers. A second copy would recreate the multi-owner
    defect this closes — and `_dataset_for`'s wider vocabulary is exactly the
    reason the helper is parameterised rather than hard-coded, the same lesson
    `_qualified_span_re` learned in B22 by dropping five governed phrases."""
    import inspect
    for module, name in ((ws, "workspace"), (routing, "chat_routing")):
        src = inspect.getsource(module)
        assert "_DISCLAIMERS" not in src, (
            f"{name} carries its own disclaimer vocabulary — a second owner")


# --------------------------------------------------------------------------- #
# B22's behaviour is untouched
# --------------------------------------------------------------------------- #
def test_the_scope_disclaimer_is_unchanged():
    assert lens.disclaimed_scope_phrase(
        "the balance excluding the acquired book") == "excluding the acquired book"
    assert lens.disclaimed_scope_phrase("balance for the acquired book") is None
    assert lens.disclaimed_scope_phrase("loans purchased at auction") is None


# --------------------------------------------------------------------------- #
# The severity, on constructed data — stated, not implied
# --------------------------------------------------------------------------- #
def test_the_frames_agree_on_this_tape():
    """THE COINCIDENCE that makes B21 change no number here. Asserted rather
    than assumed, because the whole severity argument depends on it being a
    property of this book and not of the code."""
    funded = pd.DataFrame({
        "current_outstanding_balance": [100.0, 250.0, 75.0],
        "collateral_geography": ["London", "Wales", "London"]})
    frame = ws.build_forecast_view_frame(funded, None)
    assert float(frame["current_outstanding_balance"].sum()) == pytest.approx(425.0)
    assert float(funded["current_outstanding_balance"].sum()) == pytest.approx(425.0)


def test_the_frames_diverge_once_a_pipeline_exists():
    """THE EVIDENCE IS CONSTRUCTED. This book carries no pipeline, so the funded
    and forecast frames agree to the penny and the defect moves no number on the
    tape. A pipeline is built here to show what it moves on a book that has one.

    Measured on the real funded book with three constructed pipeline cases:

        no pipeline    funded 1,964,886,258.21   forecast 1,964,886,258.21   0.00
        with pipeline  funded 1,964,886,258.21   forecast 1,965,656,258.21   770,000.00

    £770,000 arriving under `current_outstanding_balance` — the same field name
    the funded balance uses — for a question whose sentence RULED THE FORECAST
    OUT, and disclosed nowhere. The proportions here are the same shape at a
    scale a test can assert.
    """
    funded = pd.DataFrame({
        "current_outstanding_balance": [100.0, 250.0, 75.0],
        "collateral_geography": ["London", "Wales", "London"]})
    pipeline = pd.DataFrame({
        "weighted_expected_funded_amount": [400.0, 300.0, 70.0],
        "collateral_geography": ["London", "London", "Wales"]})

    without = ws.build_forecast_view_frame(funded, None)
    with_pipe = ws.build_forecast_view_frame(funded, pipeline)

    funded_total = float(funded["current_outstanding_balance"].sum())
    assert float(without["current_outstanding_balance"].sum()) == pytest.approx(funded_total)
    diverged = float(with_pipe["current_outstanding_balance"].sum())
    assert diverged == pytest.approx(funded_total + 770.0)
    assert diverged != pytest.approx(funded_total), (
        "the divergence is the severity argument; if this ever passes the "
        "constructed frame has stopped constructing anything")
    assert set(with_pipe["state_component"]) == {"funded", "forecast_pipeline"}


def test_the_disclaimed_question_now_reaches_the_column_it_needed():
    """The failure was not a wrong number — it was no answer at all. The
    forecast projection drops `vintage_year`, so the question died at
    prepared-data validation on a column the funded book carries."""
    funded = pd.DataFrame({
        "current_outstanding_balance": [100.0, 250.0],
        "vintage_year": [2024, 2019],
        "collateral_geography": ["London", "Wales"]})
    projection = ws.build_forecast_view_frame(funded, None)
    assert "vintage_year" not in projection.columns, (
        "the projection is expected to DROP it; that is the premise")
    assert ws.resolve_active_view(
        "What is the balance by vintage, ignoring the forecast?", None) == "funded"
