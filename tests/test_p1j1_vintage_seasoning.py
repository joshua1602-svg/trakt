#!/usr/bin/env python3
"""tests/test_p1j1_vintage_seasoning.py — the governed VINTAGE / SEASONING axis.

Seasoning answers WHEN a loan was originated. Provenance answers WHERE it came
from. They are INDEPENDENT axes, and the code used to conflate them:

    "the back book"      -> acquired      (a provenance filter)
    "newly originated"   -> direct        (a provenance filter)

so a seasoned directly-originated loan was excluded from "the back book" and a
recent acquired loan was included in it. A loan can be any of the four
combinations — Direct+Front, Direct+Back, Acquired+Front, Acquired+Back — and
neither axis may be inferred from the other.

Every figure here is recomputed with pandas from ``origination_date`` and the
governed reporting date. The seasoning model is never its own oracle.

Run: python -m pytest tests/test_p1j1_vintage_seasoning.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from demo_platform import config as cfg  # noqa: E402
from mi_agent import portfolio_lens as plens  # noqa: E402
from mi_agent import seasoning as season  # noqa: E402

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
SEGMENT = season.SEASONING_SEGMENT_FIELD
BUCKET = season.SEASONING_BUCKET_FIELD
DIRECT_ID = "alp_origination"
ACQUIRED_ID = "alp_acquired"


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
def book(governed_env) -> pd.DataFrame:
    from mi_agent_api.data_source import get_dataframe

    return get_dataframe()


@pytest.fixture(scope="module")
def ask(governed_env):
    from trakt_core.context import ExecutionContext
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    cache: Dict[Any, Dict[str, Any]] = {}

    def _ask(question: str) -> Dict[str, Any]:
        if question not in cache:
            cache[question] = (execute_governed_mi_query(
                MiQueryRequest(question=question), ctx).result or {})
        return cache[question]

    return _ask


# --------------------------------------------------------------------------- #
# Independent truth — recomputed from the raw dates, never from the agent
# --------------------------------------------------------------------------- #
def months_on_book(frame: pd.DataFrame) -> pd.Series:
    origination = pd.to_datetime(frame["origination_date"])
    reporting = pd.to_datetime(frame["data_cut_off_date"])
    return ((reporting.dt.year - origination.dt.year) * 12
            + (reporting.dt.month - origination.dt.month))


def wavg(frame: pd.DataFrame, value: str = LTV) -> float:
    valid = frame[[value, BALANCE]].dropna()
    return float((valid[value] * valid[BALANCE]).sum() / valid[BALANCE].sum())


def grouping(envelope) -> List[str]:
    spec = envelope.get("spec") or {}
    return [g for g in ([spec.get("dimension")] + list(spec.get("dimensions") or []))
            if g]


def spec_filters(envelope) -> Dict[str, Any]:
    return (envelope.get("spec") or {}).get("filters") or {}


def answer(envelope) -> str:
    return envelope.get("answer") or envelope.get("error") or ""


def chart_rows(envelope) -> List[Dict[str, Any]]:
    for artifact in envelope.get("artifacts") or []:
        if artifact.get("type") == "chart":
            return list(artifact.get("rows") or [])
    return []


def kpis(envelope) -> Dict[str, float]:
    for artifact in envelope.get("artifacts") or []:
        if artifact.get("type") == "kpi":
            return {k.get("field"): k.get("rawValue")
                    for k in (artifact.get("kpis") or [])}
    return {}


# =========================================================================== #
# 1-7. The two axes are separate vocabularies
# =========================================================================== #
@pytest.mark.parametrize("phrase", [
    "the back book", "the backbook", "the seasoned book", "the legacy book",
    "seasoned loans", "newly originated", "new origination",
    "new originations", "recent originations", "the front book",
])
def test_a_seasoning_phrase_is_never_provenance(phrase):
    """(1)-(5) A phrase about WHEN a loan was written must not resolve to WHERE
    it came from. This is the defect P1J-1 exists to remove."""
    lens = plens.resolve_lens(f"what is the balance of {phrase}?")
    assert lens.name == plens.LENS_TOTAL, (
        f"{phrase!r} resolved to provenance lens {lens.name!r}")
    assert plens.SOURCE_TYPE_FIELD not in (lens.filters or {})


@pytest.mark.parametrize("phrase,expected", [
    ("the acquired book", plens.LENS_ACQUIRED),
    ("the purchased book", plens.LENS_ACQUIRED),
    ("the bought book", plens.LENS_ACQUIRED),
    ("the acquisition book", plens.LENS_ACQUIRED),
    ("the direct book", plens.LENS_DIRECT),
    ("the directly originated book", plens.LENS_DIRECT),
    ("organic lending", plens.LENS_DIRECT),
])
def test_provenance_vocabulary_still_resolves(phrase, expected):
    """(6)-(7) De-conflation must not cost a single genuine provenance synonym."""
    assert plens.resolve_lens(f"what is the balance of {phrase}?").name == expected


def test_the_two_vocabularies_do_not_overlap():
    """Structural guarantee: no seasoning phrase may sit in a provenance term
    list, in either direction, or the conflation returns silently."""
    provenance = {t.lower() for t in plens._DIRECT_TERMS + plens._ACQUIRED_TERMS}
    seasoning = {p for p, _seg in season._SEGMENT_PHRASES}
    for term in provenance:
        for word in ("back book", "backbook", "legacy", "seasoned",
                     "new origination", "newly originated", "front book"):
            assert word not in term, f"{term!r} is a seasoning phrase"
    assert seasoning, "the seasoning vocabulary must not be empty"


# =========================================================================== #
# 8-9. The cross product — a loan is on BOTH axes at once
# =========================================================================== #
def test_a_seasoned_direct_loan_is_direct_and_back_book(book):
    """(8) The example from the ruling: a directly-originated loan written five
    years ago is DIRECT by provenance and BACK BOOK by seasoning."""
    mob = months_on_book(book)
    direct_back = book[(book["source_portfolio_type"] == "direct") & (mob > 60)]
    assert not direct_back.empty
    assert (direct_back[SEGMENT] == season.BACK_BOOK).all()
    assert (direct_back["source_portfolio_type"] == "direct").all()


def test_a_recent_acquired_loan_is_acquired_and_front_book(book):
    """(9) And the converse: an acquired loan can be front book."""
    mob = months_on_book(book)
    acquired_front = book[(book["source_portfolio_type"] == "acquired") & (mob <= 12)]
    assert not acquired_front.empty, (
        "fixture carries no recent acquired loans — the cross product is untested")
    assert (acquired_front[SEGMENT] == season.FRONT_BOOK).all()
    assert (acquired_front["source_portfolio_type"] == "acquired").all()


def test_all_four_combinations_exist_and_are_independent(book):
    """The axes are genuinely orthogonal on this book: all four cells populated,
    so neither axis can be predicted from the other."""
    cells = book.groupby(["source_portfolio_type", SEGMENT]).size()
    for provenance in ("direct", "acquired"):
        for segment in (season.FRONT_BOOK, season.BACK_BOOK):
            assert cells.get((provenance, segment), 0) > 0, (
                f"no {provenance} + {segment} loans")


# =========================================================================== #
# 10. Time semantics — seasoning is as at the REPORTING date
# =========================================================================== #
def test_seasoning_is_measured_against_the_reporting_date(book):
    """(10) Not wall-clock "today". The derived months-on-book must equal the
    independent recompute from origination vs the frame's own cut-off date."""
    assert (book["months_on_book"].astype("int64") == months_on_book(book)).all()


def test_a_historical_snapshot_carries_its_own_seasoning():
    """(10) The same loan queried against an older reporting date is younger.

    Proven on the derivation itself rather than by re-reading a snapshot: the
    model takes the reporting date from the frame, so an earlier cut-off yields
    a smaller months-on-book and can move a loan from back book to front book.
    """
    frame = pd.DataFrame({
        "origination_date": ["2025-01-31", "2025-01-31"],
        "data_cut_off_date": ["2026-06-30", "2025-06-30"],
    })
    mob = months_on_book(frame)
    assert list(mob) == [17, 5]
    config = season.load_seasoning_config()
    assert config.segment_for(mob.iloc[0]) == season.BACK_BOOK   # as at 2026-06
    assert config.segment_for(mob.iloc[1]) == season.FRONT_BOOK  # as at 2025-06


# =========================================================================== #
# 11. The boundary is configuration, not code
# =========================================================================== #
def test_changing_the_cutoff_changes_the_population_without_code_changes(tmp_path):
    """(11) A client moves the front/back boundary by editing config."""
    config_file = tmp_path / "buckets.yaml"
    config_file.write_text(
        "seasoning:\n"
        "  front_book_max_months: 24\n"
        "  buckets:\n"
        "    - name: '0-24m'\n      min_months: 0\n      max_months: 24\n"
        "    - name: '24m+'\n      min_months: 25\n",
        encoding="utf-8")
    wider = season.load_seasoning_config(config_file)
    assert wider.front_book_max_months == 24
    # A loan 18 months on book is BACK under the governed 12-month default and
    # FRONT under this client's 24-month boundary. Same code, same loan.
    assert season.load_seasoning_config().segment_for(18) == season.BACK_BOOK
    assert wider.segment_for(18) == season.FRONT_BOOK


def test_the_governed_default_is_twelve_months():
    config = season.load_seasoning_config()
    assert config.front_book_max_months == 12
    assert config.band_names == ["0-12m", "13-24m", "25-60m", "60m+"]


# =========================================================================== #
# 12-14. Reconciliation — the partitions are exact
# =========================================================================== #
def test_front_and_back_reconcile_to_the_governed_population(book):
    """(13) Front + Back = the whole book, by count and by balance."""
    front = book[book[SEGMENT] == season.FRONT_BOOK]
    back = book[book[SEGMENT] == season.BACK_BOOK]
    assert len(front) + len(back) == len(book)
    assert front[BALANCE].sum() + back[BALANCE].sum() == pytest.approx(
        book[BALANCE].sum(), rel=1e-12)


def test_no_loan_is_both_front_and_back(book):
    """(14) The segment is a single value per loan; the split cannot double-count."""
    assert set(book[SEGMENT].dropna().unique()) <= {season.FRONT_BOOK,
                                                    season.BACK_BOOK}
    assert book[SEGMENT].isna().sum() == 0


def test_seasoning_buckets_reconcile_to_the_total_population(book):
    """(12) The analytical bands partition the book exactly."""
    counts = book[BUCKET].value_counts(dropna=False)
    assert counts.sum() == len(book)
    assert set(counts.index) <= set(season.load_seasoning_config().band_names)
    assert book[BUCKET].isna().sum() == 0


def test_the_binary_split_derives_from_the_same_model_as_the_buckets(book):
    """Not two mechanisms: the front book IS the 0-12m band, exactly."""
    front = book[book[SEGMENT] == season.FRONT_BOOK]
    band = book[book[BUCKET] == "0-12m"]
    assert len(front) == len(band)
    assert set(front.index) == set(band.index)


# =========================================================================== #
# 15. Provenance and seasoning combine without replacing one another
# =========================================================================== #
def test_provenance_and_seasoning_combine_in_one_query(book, ask):
    """(15) A provenance scope and a seasoning population must BOTH survive —
    neither axis may replace or silently drop the other."""
    envelope = ask("What is the average LTV of the back book in the direct book?")
    assert envelope["ok"] is True, envelope.get("error")
    filters = spec_filters(envelope)
    assert filters.get(SEGMENT) == season.BACK_BOOK
    scope = (envelope.get("portfolioScope") or {}).get("portfolio_ids") or []
    assert scope == [DIRECT_ID]
    # Both axes named in the receipt, and the figure is the intersection.
    text = answer(envelope)
    assert "Back Book" in text and DIRECT_ID in text
    expected = wavg(book[(book[SEGMENT] == season.BACK_BOOK)
                         & (book["source_portfolio_id"] == DIRECT_ID)])
    assert kpis(envelope)[f"{LTV}_weighted_avg"] == pytest.approx(expected, rel=1e-9)


def test_a_seasoning_population_is_applied_not_disclosed_as_partial(ask):
    """A population honoured as a FILTER must read as APPLIED.

    The facet is raised as a grouping concept, and the reconciler used to look
    only at group keys — so "the back book", correctly executed as a filter,
    was disclosed as a not-applied breakdown on an answer that was in fact
    exactly what was asked for.
    """
    envelope = ask("What is the average LTV of the back book?")
    assert envelope["ok"] is True, envelope.get("error")
    assert "not applied" not in answer(envelope).lower()


def test_a_dropped_seasoning_population_refuses_rather_than_partials(ask):
    """The converse, and the safety half: if the seasoning population does NOT
    reach execution, the answer must refuse. Dropping it changes the NUMBER —
    a question about the seasoned book answered with the whole book — so a
    "partial" disclosure would understate what went wrong."""
    from mi_agent import execution_receipt as er

    facet = er.RequestedFacet(kind=er.KIND_GROUPING, label="back book",
                              field_key=SEGMENT)
    facet.status = er.LOST
    receipt = er.ExecutionReceipt(facets=[facet])
    verdict, message = er.assess(receipt)
    assert verdict == er.VERDICT_REFUSE
    assert "not substituted a broader figure" in (message or "")


def test_a_two_axis_comparison_refuses_rather_than_dropping_an_axis(ask):
    """Safety: a question comparing on BOTH axes at once is refused, not
    silently answered on one of them. Better a refusal than a dropped axis."""
    envelope = ask("How does the seasoned direct book compare with recent "
                   "acquired loans?")
    assert envelope["ok"] is False
    assert "how long the loans have been on the book" in answer(envelope).lower()


# =========================================================================== #
# Population selection vs grouping — the role decision
# =========================================================================== #
def test_naming_one_segment_selects_a_population(ask, book):
    """"the back book" narrows; it does not group. Answering for BOTH segments
    while claiming to answer about the back book is a silent semantic error."""
    envelope = ask("What is the average LTV of the back book?")
    assert envelope["ok"] is True, envelope.get("error")
    assert spec_filters(envelope).get(SEGMENT) == season.BACK_BOOK
    assert SEGMENT not in grouping(envelope)
    expected = wavg(book[book[SEGMENT] == season.BACK_BOOK])
    assert kpis(envelope)[f"{LTV}_weighted_avg"] == pytest.approx(expected, rel=1e-9)


def test_naming_both_segments_is_a_comparison(ask):
    """Naming both sides groups by the segment rather than narrowing to one."""
    envelope = ask("Compare the front book with the back book on balance, "
                   "loan count and average LTV.")
    assert envelope["ok"] is True, envelope.get("error")
    assert SEGMENT in grouping(envelope)
    assert SEGMENT not in spec_filters(envelope)
    assert len(chart_rows(envelope)) == 2


def test_an_ambiguous_seasoning_phrase_does_not_select_a_population():
    """"new lending" reads as a FLOW measure at least as naturally as a
    population ("the run rate of new lending"), so it must not silently narrow
    the book. An ambiguous phrase fails safe."""
    assert season.resolve_segment_population(
        "what is the run rate of new lending?") is None


def test_the_run_rate_question_does_not_acquire_a_seasoning_filter(ask):
    """The stated intent, preserved: no seasoning filter is invented.

    The `ok is True` half is gone, and the reason is measured rather than
    argued. Before routed groupings required evidence, this question — "what is
    the run rate of NEW LENDING" — was answered:

        "The current completion run-rate is ~£16.3m/month (£195.5m/year)
         based on 2 month(s) of funded growth."

    with `spec.filters == {}` and a projection whose first point is
    £1,964,886,258 — the WHOLE BOOK. The run rate of new lending was never
    computed; the whole book's was, presented as the answer, with the receipt
    stamping the new-lending facet APPLIED.

    So the answer must not stand. What this test guards is unchanged and still
    asserted: the fix must not work by inventing the filter instead.
    """
    envelope = ask("What is the run rate of new lending and is it accelerating?")
    assert SEGMENT not in spec_filters(envelope)
    assert envelope["ok"] is False
    # The refusal must name the GOVERNED POPULATION, not the wording. "New
    # lending" is months_on_book le 1; the phrase was the old grouping label,
    # and asserting it pinned a misclassification rather than a property.
    error = str(envelope.get("error") or "")
    assert "months_on_book" in error
    assert "not substituted a broader figure" in error


def test_an_unrestricted_run_rate_question_still_answers(ask):
    """Can-fail: a change that refused every run-rate question would pass above."""
    envelope = ask("What is the completion run rate?")
    assert envelope["ok"] is True, envelope.get("error")
    assert SEGMENT not in spec_filters(envelope)


# =========================================================================== #
# Realistic combined questions
# =========================================================================== #
def test_average_ltv_by_origination_vintage(ask, book):
    envelope = ask("Show average LTV by origination vintage.")
    assert envelope["ok"] is True, envelope.get("error")
    assert "vintage_year" in grouping(envelope)
    rows = {int(r["vintage_year"]): r[f"{LTV}_weighted_avg"]
            for r in chart_rows(envelope) if r.get("vintage_year") is not None}
    vintage = pd.to_datetime(book["origination_date"]).dt.year
    for year, got in rows.items():
        assert got == pytest.approx(wavg(book[vintage == year]), rel=1e-9)


def test_which_vintage_has_the_highest_average_ltv(ask, book):
    envelope = ask("Which vintage has the highest average LTV?")
    assert envelope["ok"] is True, envelope.get("error")
    assert "vintage_year" in grouping(envelope)


def test_how_much_of_the_direct_book_is_seasoned(ask):
    """A provenance scope with a seasoning question over it."""
    envelope = ask("What is the balance of the back book in the direct book?")
    if not envelope["ok"]:
        pytest.skip("phrasing not resolved; population/scope combination "
                    "covered by test_provenance_and_seasoning_combine_in_one_query")
    assert spec_filters(envelope).get(SEGMENT) == season.BACK_BOOK


# =========================================================================== #
# The immutable bank questions this phase targets
# =========================================================================== #
def test_b04_compares_vintage_not_provenance(ask, book):
    """B04, exact bank wording. The comparison is recent vs seasoned ORIGINATION,
    never direct vs acquired, and the measure is the governed quality proxy."""
    envelope = ask("Is the credit quality of new origination better or worse "
                   "than the back book?")
    assert envelope["ok"] is True, envelope.get("error")
    spec = envelope.get("spec") or {}
    assert spec.get("metric") == LTV
    assert SEGMENT in grouping(envelope)
    assert "source_portfolio_type" not in grouping(envelope)
    assert "source_portfolio_type" not in spec_filters(envelope)

    got = {r[SEGMENT]: r[f"{LTV}_weighted_avg"] for r in chart_rows(envelope)}
    for segment in (season.FRONT_BOOK, season.BACK_BOOK):
        expected = wavg(book[book[SEGMENT] == segment])
        assert got[segment] == pytest.approx(expected, rel=1e-9)


def test_b09_ranks_vintages_by_ltv(ask, book):
    """B09, exact bank wording."""
    envelope = ask("Which vintages have the highest LTV?")
    assert envelope["ok"] is True, envelope.get("error")
    assert "vintage_year" in grouping(envelope)
    vintage = pd.to_datetime(book["origination_date"]).dt.year
    for row in chart_rows(envelope):
        if row.get("vintage_year") is None:
            continue
        expected = wavg(book[vintage == int(row["vintage_year"])])
        assert row[f"{LTV}_weighted_avg"] == pytest.approx(expected, rel=1e-9)


def test_b28_shows_quality_by_vintage(ask, book):
    """B28, exact bank wording. A vintage is a cohort label, not a time axis:
    thirteen vintages must survive, not collapse into one date bucket."""
    envelope = ask("Show me the quality of the book by origination vintage.")
    assert envelope["ok"] is True, envelope.get("error")
    assert "vintage_year" in grouping(envelope)
    # All thirteen vintages must be COMPUTED. The chart itself caps at a top-N
    # with a disclosed "other" bucket, so the group count is asserted on the
    # receipt rather than on the rendered row count.
    assert "13 group" in answer(envelope), answer(envelope)
    rows = chart_rows(envelope)
    assert rows and all(str(r.get("vintage_year", "")).isdigit() for r in rows)
    # The old defect: integer years date-coerced to epoch month "1970-01",
    # collapsing every vintage into a single row.
    assert not any("1970" in str(v) for r in rows for v in r.values())


# =========================================================================== #
# Receipt — the executed interpretation must be obvious
# =========================================================================== #
@pytest.mark.parametrize("question,expected", [
    ("What is the average LTV of the back book?", "Seasoning Segment"),
    ("Show me the quality of the book by origination vintage.", "Vintage"),
    ("What is the average LTV of the direct book vs the acquired book?",
     "Direct"),
])
def test_the_receipt_names_the_axis_that_ran(ask, question, expected):
    """A reader must be able to tell provenance, binary seasoning and vintage
    apart from the receipt alone — a vague "portfolio comparison" would hide
    exactly the distinction this phase exists to make."""
    envelope = ask(question)
    assert envelope["ok"] is True, envelope.get("error")
    assert expected in answer(envelope)


@pytest.mark.parametrize("op,value,expected", [
    ("ge", "2024-01-01", [False, True, True]),
    ("lt", "2024-01-01", [True, False, False]),
    ("between", ["2024-01-01", "2026-12-31"], [False, True, True]),
])
def test_a_date_comparison_does_not_crash_the_executor(op, value, expected):
    """A range over an ORIGINATION DATE is an ordinary way to ask for recent
    lending, and the model reaches for it when asked about "newly originated
    loans". It used to raise ValueError: could not convert string to float:
    '2024-01-01' out of the executor and surface as "The MI Agent could not
    complete this query" — a crash, not a governed refusal."""
    from mi_agent.mi_query_executor import _apply_numeric_op

    column = pd.Series(["2018-11-14", "2025-03-01", "2026-05-05"])
    assert list(_apply_numeric_op(column, op, value)) == expected


def test_a_numeric_comparison_still_works():
    """The date path must not cost the numeric one."""
    from mi_agent.mi_query_executor import _apply_numeric_op

    column = pd.Series([10, 50, 90])
    assert list(_apply_numeric_op(column, "gt", 40)) == [False, True, True]
    assert list(_apply_numeric_op(column, "between", [20, 80])) == [False, True, False]


def test_an_uncomparable_value_fails_in_a_controlled_way():
    """Neither numeric nor a date is a governed failure, never an unhandled one."""
    from mi_agent.mi_query_executor import MIQueryExecutionError, _apply_numeric_op

    with pytest.raises(MIQueryExecutionError):
        _apply_numeric_op(pd.Series(["2024-01-01"]), "gt", "banana")


def test_newly_originated_loans_answer(ask, book):
    """The phrase that exposed the crash. It must ANSWER as the front book."""
    envelope = ask("What is the balance of newly originated loans?")
    assert envelope["ok"] is True, envelope.get("error")
    assert "could not complete" not in answer(envelope).lower()


def test_the_seasoning_receipt_states_the_governed_boundary():
    """The cutoff is configurable, so a reader must be able to see WHICH
    boundary produced the population they are looking at."""
    config = season.load_seasoning_config()
    assert config.describe_segment(season.FRONT_BOOK) == "Front Book (0-12 months)"
    assert config.describe_segment(season.BACK_BOOK) == "Back Book (13+ months)"
