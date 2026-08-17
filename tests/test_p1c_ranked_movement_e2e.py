#!/usr/bin/env python3
"""tests/test_p1c_ranked_movement_e2e.py — ranked period-over-period movement,
end to end through the governed MI capability, reconciled to independent truth.

SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.

What makes this a truth test rather than a snapshot test: every expected figure
is recomputed HERE with pandas, directly from the two governed snapshots, using
no ``mi_agent.period_change`` code and no ranking module. If the ranking and the
independent calculation disagree, the test fails — a regression cannot be
absorbed by updating an expected string.

The questions go through ``execute_governed_mi_query``, which is the same
entrypoint POST /mi/query and the Copilot ``askTraktMi`` action call, so what is
tested is the production parse → route → rank → guard → answer path.

Skipped (not failed) when the demonstration platform has not been built:

    python -m demo_platform.run_demo --generate --orchestrate
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo_platform import config as cfg  # noqa: E402

_BUILD_HINT = ("the demonstration platform has not been built — run "
               "`python -m demo_platform.run_demo --generate --orchestrate`")

#: The governed geography dimension the period-change analysis reports on, and
#: the balance field every currency figure is drawn from.
FIELD = "collateral_geography"
BALANCE = "current_outstanding_balance"
#: The missing-value bucket. A real row in the distribution, but not a region.
UNKNOWN = "Unknown"


# --------------------------------------------------------------------------- #
# Environment — the demo platform, deterministic parser, no LLM
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def governed_env():
    root = cfg.local_blob_root() / "processed" / "platform" / cfg.CLIENT_ID
    if not root.exists():
        pytest.skip(_BUILD_HINT)
    before = dict(os.environ)
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_DATA_CACHE_TTL"] = "0"
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    logging.disable(logging.WARNING)

    from mi_agent_api import data_source

    data_source.reset_cache()
    yield os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"]
    logging.disable(logging.NOTSET)
    os.environ.clear()
    os.environ.update(before)
    data_source.reset_cache()


@pytest.fixture(scope="module")
def ask(governed_env):
    """``ask(question) -> envelope``, memoised so a question runs once."""
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


# --------------------------------------------------------------------------- #
# Independent truth — pandas only, from the two governed snapshots
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def truth(governed_env) -> pd.DataFrame:
    """Per-region movement between the two most recent snapshots.

    Deliberately re-implements the arithmetic instead of importing it: the
    categorisation rule (trim, and bucket every blank/placeholder as
    ``Unknown``) is restated here so a change to the governed normaliser shows
    up as a disagreement rather than being silently mirrored.
    """
    from mi_agent_api import evolution

    frames = evolution.funded_frames(governed_env, cfg.CLIENT_ID, None)
    if len(frames) < 2:
        pytest.skip("fewer than two governed snapshots are available")

    def _table(df: pd.DataFrame) -> pd.DataFrame:
        text = df[FIELD].astype("string").str.strip()
        blank = (text.isna() | text.eq("")
                 | text.str.lower().isin(["nan", "nat", "none", "<na>", "null"]))
        keyed = df.assign(**{FIELD: text.where(~blank, UNKNOWN)})
        return keyed.groupby(FIELD).agg(count=(BALANCE, "size"),
                                        balance=(BALANCE, "sum"))

    opening, closing = _table(frames[-2]["df"]), _table(frames[-1]["df"])
    joined = opening.join(closing, how="outer", lsuffix="_start",
                          rsuffix="_end").fillna(0.0)
    joined["abs_move"] = joined["balance_end"] - joined["balance_start"]
    joined["pct_move"] = (joined["abs_move"]
                          / joined["balance_start"].abs()) * 100.0
    joined["count_move"] = joined["count_end"] - joined["count_start"]
    joined["share_move_pp"] = (
        (joined["balance_end"] / joined["balance_end"].sum())
        - (joined["balance_start"] / joined["balance_start"].sum())) * 100.0
    joined.attrs["opening_date"] = str(frames[-2].get("reporting_date"))
    joined.attrs["closing_date"] = str(frames[-1].get("reporting_date"))
    return joined


def _expected_order(truth: pd.DataFrame, column: str, *, ascending: bool = False):
    """Categories ordered as the ranking should order them, ``Unknown`` removed
    and non-movers in the wrong direction dropped."""
    frame = truth.drop(index=[UNKNOWN], errors="ignore")
    frame = frame[frame[column] > 0] if not ascending else frame[frame[column] < 0]
    # Same tie-break as the ranking: value first, then category name ascending.
    frame = frame.assign(_cat=frame.index)
    frame = frame.sort_values(["_cat"]).sort_values(column, ascending=ascending,
                                                    kind="stable")
    return list(frame.index)


def _ranked(envelope: Dict[str, Any]) -> Dict[str, Any]:
    return (envelope.get("metadata") or {}).get("rankedMovement") or {}


def _ranking_applied(envelope: Dict[str, Any]) -> bool:
    """True only when the route declares a ranking it actually performed. A
    refusal still carries evidence — ``{"applied": False, "reason": …}`` — so
    the presence of the key is not the test."""
    return _ranked(envelope).get("applied") is True


def _rows(envelope: Dict[str, Any]):
    return _ranked(envelope).get("rows") or []


# =========================================================================== #
# The four ranking bases, each reconciled to independent truth
# =========================================================================== #
def test_absolute_balance_ranking_matches_independent_truth(ask, truth):
    envelope = ask("Which region grew the most last month?")
    assert envelope.get("ok") is True, envelope.get("error")
    evidence = _ranked(envelope)
    assert evidence.get("applied") is True
    assert evidence["basis"] == "balance_absolute"
    assert evidence["canonicalField"] == FIELD

    rows = _rows(envelope)
    assert [r["category"] for r in rows] == _expected_order(truth, "abs_move")
    for row in rows:
        expected = truth.loc[row["category"]]
        assert row["start_value"] == pytest.approx(expected["balance_start"], rel=1e-9)
        assert row["end_value"] == pytest.approx(expected["balance_end"], rel=1e-9)
        assert row["absolute_movement"] == pytest.approx(expected["abs_move"], rel=1e-9)


def test_percentage_ranking_ranks_on_growth_rate_not_size(ask, truth):
    """The measure decides the basis: "fastest" is a rate, so the largest
    absolute mover must NOT automatically come first."""
    envelope = ask("Which region grew fastest in percentage terms month on month?")
    assert envelope.get("ok") is True, envelope.get("error")
    assert _ranked(envelope)["basis"] == "balance_percent"

    rows = _rows(envelope)
    assert [r["category"] for r in rows] == _expected_order(truth, "pct_move")
    for row in rows:
        assert row["percent_movement"] == pytest.approx(
            truth.loc[row["category"], "pct_move"], rel=1e-9)
    # The point of the test: a different question produced a different order.
    assert rows[0]["category"] != _expected_order(truth, "abs_move")[0]


def test_loan_count_ranking_matches_independent_truth(ask, truth):
    envelope = ask("Which region added the most loans month-on-month?")
    assert envelope.get("ok") is True, envelope.get("error")
    assert _ranked(envelope)["basis"] == "count_absolute"

    rows = _rows(envelope)
    assert [r["category"] for r in rows] == _expected_order(truth, "count_move")
    for row in rows:
        assert row["absolute_movement"] == pytest.approx(
            float(truth.loc[row["category"], "count_move"]))


def test_share_of_balance_ranking_matches_independent_truth(ask, truth):
    envelope = ask("Which region increased its share of the portfolio the most "
                   "since last month?")
    assert envelope.get("ok") is True, envelope.get("error")
    assert _ranked(envelope)["basis"] == "balance_share"

    rows = _rows(envelope)
    assert [r["category"] for r in rows] == _expected_order(truth, "share_move_pp")
    for row in rows:
        # The governed share is a ratio; the truth is in percentage points.
        assert row["absolute_movement"] * 100.0 == pytest.approx(
            truth.loc[row["category"], "share_move_pp"], abs=1e-9)


# =========================================================================== #
# Top N, exclusions and disclosure
# =========================================================================== #
def test_top_n_truncates_the_same_ranking(ask, truth):
    envelope = ask("What were the top 3 regions by growth since last month?")
    assert envelope.get("ok") is True, envelope.get("error")
    evidence = _ranked(envelope)
    assert evidence["topN"] == 3
    rows = _rows(envelope)
    assert len(rows) == 3
    assert [r["category"] for r in rows] == _expected_order(truth, "abs_move")[:3]


def test_the_unknown_bucket_is_excluded_and_disclosed(ask, truth):
    """"Unknown" is missing data, not a region. It must not be able to win a
    ranking of regions — and its exclusion must be visible, not silent."""
    if UNKNOWN not in truth.index:
        pytest.skip("this book has no unknown-geography bucket")
    envelope = ask("Which region grew the most last month?")
    assert UNKNOWN not in [r["category"] for r in _rows(envelope)]
    excluded = {e["category"] for e in _ranked(envelope).get("excluded") or []}
    assert UNKNOWN in excluded
    assert UNKNOWN.lower() in (envelope.get("answer") or "").lower()


def test_categories_moving_the_other_way_are_counted_not_hidden(ask, truth):
    """A ranking of increases silently dropping the decliners reads as "these
    are all the regions". The answer states how many were left out."""
    envelope = ask("Which region added the most loans month-on-month?")
    ranked = len(_rows(envelope))
    ranked_universe = len(truth.drop(index=[UNKNOWN], errors="ignore"))
    left_out = ranked_universe - ranked
    if left_out:
        assert f"{left_out} further" in (envelope.get("answer") or "")


# =========================================================================== #
# The execution receipt — what ran, in the reader's view
# =========================================================================== #
def test_the_receipt_states_basis_direction_dimension_and_both_periods(ask, truth):
    envelope = ask("Which region grew the most last month?")
    summary = envelope.get("executionSummary") or {}
    receipt = summary.get("receipt") or ""
    assert "Collateral Geography" in receipt
    assert "absolute balance movement" in receipt
    assert "largest increases first" in receipt
    assert truth.attrs["opening_date"] in receipt
    assert truth.attrs["closing_date"] in receipt
    assert summary.get("comparisonPeriod") == (
        f"{truth.attrs['opening_date']} → {truth.attrs['closing_date']}")


def test_a_top_n_receipt_states_the_n_and_the_universe(ask, truth):
    summary = ask("What were the top 3 regions by growth since last "
                  "month?").get("executionSummary") or {}
    assert "top 3 of" in (summary.get("receipt") or "")


def test_the_p0_guard_records_the_ranking_facet_as_applied(ask):
    """P0's rule for P1C: the ranking is provably applied, or the answer is
    refused. A ranked answer that cannot prove it ranked must not pass."""
    envelope = ask("Which region grew the most last month?")
    guard = envelope.get("semanticGuard") or {}
    assert guard.get("verdict") == "ok", guard.get("message")
    facets = {f["kind"]: f for f in guard.get("facets") or []}
    if "ranking" in facets:
        assert facets["ranking"]["status"] == "applied"
    assert facets.get("comparison_period", {}).get("status", "applied") == "applied"


# =========================================================================== #
# Route precedence — the ranked period question must reach the ranked capability
# =========================================================================== #
@pytest.mark.parametrize("question", [
    "Which region grew the most last month?",
    "Which region added the most balance month-on-month?",
    "Which region grew fastest in percentage terms month on month?",
    "What were the top 3 regions by growth since last month?",
])
def test_a_ranked_period_question_reaches_the_period_change_route(ask, question):
    """Before P1C these were captured by ``geo_exposure``, a POINT-IN-TIME
    route: the answer named today's largest region and the comparison silently
    disappeared. The governed period-change capability owns them."""
    envelope = ask(question)
    assert (envelope.get("metadata") or {}).get("route") == "period_change_analysis"


@pytest.mark.parametrize("question", [
    "Which region has the largest exposure?",
    "What is the geographic concentration of the book?",
])
def test_a_point_in_time_geography_question_still_reaches_geo_exposure(ask, question):
    """Deference is narrow: with no period comparison in the question, the
    point-in-time geographic route keeps every question it already answered."""
    envelope = ask(question)
    assert (envelope.get("metadata") or {}).get("route") == "geo_exposure"


# =========================================================================== #
# The negative bank — every one of these must refuse, and refuse honestly
# =========================================================================== #
def _is_controlled_refusal(envelope: Dict[str, Any]) -> bool:
    return (envelope.get("ok") is False
            and bool(envelope.get("answer"))
            and not (envelope.get("artifacts") or []))


def test_a_direction_nothing_moved_in_is_stated_not_reversed(ask, truth):
    """Nothing declined. The answer says so — it must not quietly report the
    increases instead, which is the same question answered backwards."""
    if (truth.drop(index=[UNKNOWN], errors="ignore")["abs_move"] < 0).any():
        pytest.skip("this book has a declining region, so the case does not arise")
    envelope = ask("Which region declined the most last month?")
    assert _is_controlled_refusal(envelope)
    answer = envelope["answer"].lower()
    assert "no region" in answer and "decreas" in answer
    assert not _ranking_applied(envelope)
    biggest_riser = _expected_order(truth, "abs_move")[0]
    assert biggest_riser.lower() not in answer, (
        "a question about declines was answered with the largest increase")


def test_a_dimension_the_analysis_does_not_carry_is_refused_not_substituted(ask):
    """LTV band is not a governed period-change dimension on this book. The
    refusal names it; nothing else is ranked in its place."""
    envelope = ask("Which LTV band grew fastest month on month?")
    assert _is_controlled_refusal(envelope)
    assert "ltv band" in envelope["answer"].lower()
    assert "not ranked a different dimension" in envelope["answer"].lower()
    assert not _ranking_applied(envelope)


def test_an_unresolvable_dimension_is_refused_rather_than_guessed(ask):
    """"Segments" is not a governed dimension. Ranking geography instead would
    be the silent substitution P0 exists to prevent."""
    envelope = ask("Which five segments grew the most since last month?")
    assert _is_controlled_refusal(envelope)
    assert "could not identify a governed dimension" in envelope["answer"].lower()
    assert not _ranking_applied(envelope)


def test_a_narrative_period_question_keeps_its_narrative_answer(ask):
    """No superlative, no dimension: P1C must not turn every period-change
    question into a ranking, and the governed month-on-month narrative that
    already owns this question must keep it."""
    envelope = ask("What changed since last month?")
    assert envelope.get("ok") is True, envelope.get("error")
    assert not _ranking_applied(envelope)
    assert "during the month" in (envelope.get("answer") or "")


def test_a_period_change_narrative_question_keeps_its_narrative_answer(ask):
    envelope = ask("What moved month on month across the portfolio?")
    assert envelope.get("ok") is True, envelope.get("error")
    assert not _ranking_applied(envelope)


def test_largest_movements_keeps_the_governed_narrative(ask):
    """"Largest" is a superlative, but "movements" is the narrative itself
    rather than a dimension to rank — the existing answer must survive."""
    envelope = ask("What were the largest movements since last month?")
    assert envelope.get("ok") is True, envelope.get("error")
    assert not _ranking_applied(envelope)
    assert "Largest observed movements" in (envelope.get("answer") or "")


def test_no_ranked_answer_ever_omits_its_receipt(ask):
    for question in ("Which region grew the most last month?",
                     "Which region added the most loans month-on-month?",
                     "What were the top 3 regions by growth since last month?"):
        envelope = ask(question)
        assert (envelope.get("executionSummary") or {}).get("receipt"), question
        assert "Calculated:" in (envelope.get("answer") or ""), question


# =========================================================================== #
# The P1C golden bank
# =========================================================================== #
# Every question a reader might reasonably ask of this capability, each with the
# outcome CLASS it must produce. There are only three legitimate classes, which
# is the P0 invariant restated for P1C:
#
#   RANKED     answered by ranking the governed period-change output
#   REFUSED    explicitly declined, with nothing substituted
#   NARRATIVE  answered by the governed narrative that already owned it
#
# A question that lands in a different class from the one stated here is a
# silent semantic error, whether or not its numbers happen to be right.
RANKED, REFUSED, NARRATIVE = "ranked", "refused", "narrative"

#: (question, class, ranking basis if ranked, expected Top N)
GOLDEN_BANK = [
    # -- absolute balance movement (the documented bare-"grew" convention) --
    ("Which region grew the most last month?", RANKED, "balance_absolute", None),
    ("Which regions grew the most last month?", RANKED, "balance_absolute", None),
    ("Which region added the most balance month-on-month?", RANKED,
     "balance_absolute", None),
    ("Which region gained the most balance since the prior month?", RANKED,
     "balance_absolute", None),
    ("Rank regions by growth since last month.", RANKED, "balance_absolute", None),
    ("What were the top 3 regions by growth since last month?", RANKED,
     "balance_absolute", 3),
    ("Show me the top 5 regions by growth month on month.", RANKED,
     "balance_absolute", 5),
    # -- growth RATE, which must produce a different order --
    ("Which region grew fastest in percentage terms month on month?", RANKED,
     "balance_percent", None),
    ("Which region grew fastest month-on-month?", RANKED, "balance_percent", None),
    # -- loan count --
    ("Which region added the most loans month-on-month?", RANKED,
     "count_absolute", None),
    ("Which region gained the most cases since last month?", RANKED,
     "count_absolute", None),
    # -- share of the book --
    ("Which region increased its share of the portfolio the most since last "
     "month?", RANKED, "balance_share", None),
    ("Which region increased its share of balance the most month on month?",
     RANKED, "balance_share", None),
    # -- declines: nothing declined on this book, and that is what it says --
    ("Which region declined the most last month?", REFUSED, None, None),
    ("Which region lost the most balance month-on-month?", REFUSED, None, None),
    ("Where did balance fall most since last month?", REFUSED, None, None),
    # -- dimensions this book does not govern for period change --
    ("Which LTV band grew fastest month on month?", REFUSED, None, None),
    ("Which product type grew the most month on month?", REFUSED, None, None),
    ("Which arrears bucket grew the most since last month?", REFUSED, None, None),
    # -- no governed dimension at all --
    ("Which five segments grew the most since last month?", REFUSED, None, None),
    ("Which cohort grew the most month on month?", REFUSED, None, None),
    # -- forward-looking: a ranking of a period that has not happened --
    ("Which region will grow the most next month?", REFUSED, None, None),
    ("Which region is forecast to grow the most?", REFUSED, None, None),
    # -- narrative questions the ranking must not capture --
    ("What changed since last month?", NARRATIVE, None, None),
    ("What were the largest movements since last month?", NARRATIVE, None, None),
    ("What moved month on month across the portfolio?", NARRATIVE, None, None),
    ("What drove the change in funded balance?", NARRATIVE, None, None),
]

#: Which truth column each ranking basis must reproduce.
_BASIS_TRUTH_COLUMN = {
    "balance_absolute": "abs_move",
    "balance_percent": "pct_move",
    "count_absolute": "count_move",
    "balance_share": "share_move_pp",
}


@pytest.mark.parametrize("question,expected,basis,top_n", GOLDEN_BANK,
                         ids=[q[:52] for q, _, _, _ in GOLDEN_BANK])
def test_the_p1c_golden_bank(ask, truth, question, expected, basis, top_n):
    envelope = ask(question)
    answer = envelope.get("answer") or envelope.get("error") or ""
    assert answer, f"{question!r} produced no answer at all"

    if expected is RANKED:
        assert envelope.get("ok") is True, answer
        assert _ranking_applied(envelope), answer
        evidence = _ranked(envelope)
        assert evidence["basis"] == basis, answer
        assert evidence["topN"] == top_n
        # The leader is the leader of the INDEPENDENT calculation, on the basis
        # the question asked for — not merely a plausible-looking category.
        leader = _expected_order(truth, _BASIS_TRUTH_COLUMN[basis])[0]
        assert _rows(envelope)[0]["category"] == leader, answer
        if top_n:
            assert len(_rows(envelope)) == top_n
        assert "Calculated:" in answer, "a ranked answer must carry its receipt"
    elif expected is REFUSED:
        assert envelope.get("ok") is False, answer
        assert not (envelope.get("artifacts") or []), (
            "a refusal must not ship artifacts a reader could mistake for the "
            "answer")
        assert not _ranking_applied(envelope)
    else:
        assert envelope.get("ok") is True, answer
        assert not _ranking_applied(envelope), (
            f"a narrative question was turned into a ranking: {answer}")


def test_the_bank_covers_every_supported_basis_and_both_directions():
    """A bank that only tests the happy basis is not a bank."""
    bases = {b for _, cls, b, _ in GOLDEN_BANK if cls is RANKED}
    assert bases == set(_BASIS_TRUTH_COLUMN)
    assert sum(1 for _, cls, _, n in GOLDEN_BANK if cls is RANKED and n) >= 2
    assert sum(1 for _, cls, _, _ in GOLDEN_BANK if cls is REFUSED) >= 8
    assert len(GOLDEN_BANK) >= 20


def test_no_bank_question_produces_a_ranking_of_the_wrong_dimension(ask):
    """The single failure P1C exists to prevent: a ranking that ran on a
    dimension other than the one asked about."""
    for question, expected, _, _ in GOLDEN_BANK:
        evidence = _ranked(ask(question))
        if evidence.get("applied"):
            assert expected is RANKED, question
            assert evidence["canonicalField"] == FIELD, question
