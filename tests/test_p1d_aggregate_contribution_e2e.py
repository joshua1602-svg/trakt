#!/usr/bin/env python3
"""tests/test_p1d_aggregate_contribution_e2e.py — B11 and its paraphrases,
end to end through the governed MI capability, reconciled to independent truth.

SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.

B11 — "Which region contributes most to the weighted average LTV?" — was the one
silent semantic error left in the 40-question bank after P1C. The agent returned
a chart titled with the question and ordered by each region's own LTV, so West
Midlands appeared first; the region that actually contributes most to the book's
weighted average is South East.

Every expectation here is recomputed with pandas from the loan book, using the
definition of a weighted average rather than any agent code:

    portfolio WA = sum(w*v) / sum(w)
    contribution_g = sum over g of (w*v) / sum over the book of w

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

REGION = "collateral_geography"
LTV = "current_loan_to_value"
BALANCE = "current_outstanding_balance"


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


@pytest.fixture(scope="module")
def truth(governed_env) -> pd.DataFrame:
    """Contribution per region, from the definition, with pandas only.

    The dropna rule is restated here rather than imported: a loan with no LTV
    contributes nothing and must not sit in the denominator either, which is the
    rule the executor applies. Stating it in the test is what makes a change to
    it visible.
    """
    from mi_agent_api.data_source import get_dataframe

    book = get_dataframe()
    frame = book[[REGION, LTV, BALANCE]].dropna(subset=[LTV, BALANCE]).copy()
    # Missing grouping values are BUCKETED, not dropped — the same rule the
    # executor applies, restated so a change to it shows up here.
    region = frame[REGION].astype("string").str.strip()
    blank = (region.isna() | region.eq("")
             | region.str.lower().isin(["nan", "none", "nat"]))
    frame[REGION] = region.where(~blank, "Unknown / Missing")
    total_weight = float(frame[BALANCE].sum())
    grouped = frame.assign(_wv=frame[LTV] * frame[BALANCE]).groupby(REGION)
    out = pd.DataFrame({
        "contribution": grouped["_wv"].sum() / total_weight,
        "weight_share_pct": grouped[BALANCE].sum() / total_weight * 100.0,
        "group_ltv": grouped["_wv"].sum() / grouped[BALANCE].sum(),
    }).sort_values("contribution", ascending=False)
    out.attrs["portfolio_wa_ltv"] = float(frame[LTV].mul(frame[BALANCE]).sum()
                                          / total_weight)
    return out


def _rows(envelope: Dict[str, Any]):
    for artifact in envelope.get("artifacts") or []:
        if artifact.get("type") == "table":
            return artifact.get("rows") or artifact.get("data") or []
    return []


#: Every phrasing must produce the SAME governed calculation.
PARAPHRASES = [
    "Which region contributes most to the weighted average LTV?",
    "Which region is the largest contributor to the weighted average LTV?",
    "Which region is the biggest contributor to portfolio WA LTV?",
    "Which region drives most of the weighted average LTV?",
    "Which region accounts for most of the weighted average LTV?",
    "What is the regional contribution to the weighted average LTV?",
]


# =========================================================================== #
# The answer, reconciled to independent truth
# =========================================================================== #
def test_b11_reports_the_true_largest_contributor(ask, truth):
    envelope = ask(PARAPHRASES[0])
    assert envelope.get("ok") is True, envelope.get("error")
    rows = _rows(envelope)
    assert rows, "the contribution table is missing"

    leader = truth.index[0]
    assert rows[0][REGION] == leader
    # …and it is NOT the region that merely has the highest LTV.
    assert truth["group_ltv"].idxmax() != leader, (
        "fixture assumption: on this book the two rankings must differ")
    assert rows[0][REGION] in (envelope.get("answer") or "")


def test_every_contribution_figure_matches_the_definition(ask, truth):
    # The envelope rounds to 10 decimal places for transport, so an absolute
    # floor sits alongside the relative tolerance; the relative check is what
    # actually pins the arithmetic.
    tolerance = dict(rel=1e-9, abs=1e-9)
    for row in _rows(ask(PARAPHRASES[0])):
        expected = truth.loc[row[REGION]]
        assert row[f"{LTV}_contribution"] == pytest.approx(
            expected["contribution"], **tolerance)
        assert row["weight_share_pct"] == pytest.approx(
            expected["weight_share_pct"], **tolerance)
        assert row[f"{LTV}_weighted_avg"] == pytest.approx(
            expected["group_ltv"], **tolerance)


def test_the_contributions_sum_to_the_portfolio_weighted_average(ask, truth):
    rows = _rows(ask(PARAPHRASES[0]))
    total = sum(r[f"{LTV}_contribution"] for r in rows)
    assert total == pytest.approx(truth.attrs["portfolio_wa_ltv"], rel=1e-9)


def test_the_rows_are_ordered_by_contribution(ask, truth):
    assert [r[REGION] for r in _rows(ask(PARAPHRASES[0]))] == list(truth.index)


@pytest.mark.parametrize("question", PARAPHRASES, ids=[q[:50] for q in PARAPHRASES])
def test_every_paraphrase_produces_the_same_governed_calculation(ask, truth, question):
    envelope = ask(question)
    assert envelope.get("ok") is True, envelope.get("error")
    rows = _rows(envelope)
    assert rows[0][REGION] == truth.index[0], question
    assert rows[0][f"{LTV}_contribution"] == pytest.approx(
        truth.iloc[0]["contribution"], rel=1e-9)


# =========================================================================== #
# The two intents stay distinct
# =========================================================================== #
def test_the_answer_names_both_figures_so_the_difference_is_visible(ask, truth):
    """The leading contributor and the highest-LTV region are different regions
    on this book. An answer that names only one invites the same misreading the
    old behaviour caused."""
    answer = ask(PARAPHRASES[0]).get("answer") or ""
    assert truth.index[0] in answer
    assert truth["group_ltv"].idxmax() in answer


def test_the_chart_is_titled_with_the_calculation_not_the_question(ask):
    titles = [a.get("title") for a in ask(PARAPHRASES[0]).get("artifacts") or []
              if a.get("type") in ("chart", "table")]
    assert titles
    for title in titles:
        assert "Contribution to Portfolio" in title
        assert "contributes most" not in (title or "").lower()


def test_the_receipt_states_the_contribution_calculation(ask):
    receipt = (ask(PARAPHRASES[0]).get("executionSummary") or {}).get("receipt") or ""
    assert "Contribution to portfolio weighted-average" in receipt
    assert "grouped by Region" in receipt


def test_the_p0_guard_records_the_contribution_facet_as_applied(ask):
    guard = ask(PARAPHRASES[0]).get("semanticGuard") or {}
    assert guard.get("verdict") == "ok", guard.get("message")
    facets = {f["kind"]: f for f in guard.get("facets") or []}
    assert facets["aggregate_contribution"]["status"] == "applied"


def test_a_plain_highest_ltv_question_is_not_turned_into_a_contribution(ask):
    """The distinctness requirement, end to end: no contribution facet, and no
    contribution calculation."""
    envelope = ask("Show me weighted average LTV by region")
    guard = envelope.get("semanticGuard") or {}
    kinds = {f["kind"] for f in guard.get("facets") or []}
    assert "aggregate_contribution" not in kinds
    summary = envelope.get("executionSummary") or {}
    assert "Contribution" not in (summary.get("receipt") or "")


def test_a_contribution_question_with_no_dimension_is_refused(ask):
    """"What drives most of the weighted average LTV?" names nothing to
    attribute the contribution across. Choosing a dimension for the reader would
    be answering a question they did not ask, so it refuses."""
    envelope = ask("What drives most of the weighted average LTV?")
    assert envelope.get("ok") is False
    assert "contribution to the portfolio weighted average" in (
        envelope.get("answer") or "")
    assert not (envelope.get("artifacts") or [])


def test_a_genuine_balance_bridge_question_is_untouched(ask):
    """The contribution recogniser runs before the bridge recogniser, so this
    pins that it cannot claim a bridge question. A bridge decomposes a balance
    MOVEMENT between two dates; a balance is a sum, not a weighted aggregate."""
    envelope = ask("What drove the change in funded balance?")
    assert envelope.get("ok") is True, envelope.get("error")
    assert (envelope.get("metadata") or {}).get("route") == "funded_bridge"
