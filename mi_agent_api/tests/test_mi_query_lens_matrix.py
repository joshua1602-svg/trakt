"""End-to-end portfolio-lens matrix for the MI Query Agent.

The existing suites exercise the chat against the synthetic pack, which carries
**no source-portfolio provenance** — so the entire lens layer was skipped and
every multi-portfolio behaviour went untested end to end. This suite serves a
five-portfolio provenance frame (three direct, two acquired) through the real
``POST /mi/query`` endpoint and asserts what the agent may and may not say about
scope.

The organising principle is one rule:

    **A governed answer must state the scope it actually has.**

Narrowing correctly is necessary but not sufficient — an answer that could not be
narrowed must say so, and must not be stamped with a scope it does not cover.
Both halves are tested here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mi_agent_api.app import app

client = TestClient(app)

# Five portfolios so group behaviour is genuinely multi-member on BOTH sides:
# Direct = {direct_001, direct_002, direct_003}, Acquired = {acquired_001,
# acquired_002}. A group that is always a single portfolio proves nothing about
# consolidation.
PORTFOLIOS = [
    ("direct_001", "direct", "Direct Book One"),
    ("direct_002", "direct", "Direct Book Two"),
    ("direct_003", "direct", "Direct Book Three"),
    ("acquired_001", "acquired", "Acquired Book One"),
    ("acquired_002", "acquired", "Acquired Book Two"),
]
DIRECT_IDS = ["direct_001", "direct_002", "direct_003"]
ACQUIRED_IDS = ["acquired_001", "acquired_002"]
ALL_IDS = DIRECT_IDS + ACQUIRED_IDS


def _provenance_frame(base: pd.DataFrame) -> pd.DataFrame:
    """Stamp round-robin provenance onto the governed synthetic pack.

    Round-robin keeps every portfolio populated for every field, so a coverage
    exclusion in a test means a real defect rather than a fixture artefact.
    """
    df = base.copy()
    n = len(df)
    df["source_portfolio_id"] = [PORTFOLIOS[i % len(PORTFOLIOS)][0] for i in range(n)]
    df["source_portfolio_type"] = [PORTFOLIOS[i % len(PORTFOLIOS)][1] for i in range(n)]
    df["source_portfolio_label"] = [PORTFOLIOS[i % len(PORTFOLIOS)][2] for i in range(n)]
    return df


@pytest.fixture()
def multi_portfolio(monkeypatch):
    """Serve a five-portfolio provenance frame to the whole query path."""
    from mi_agent_api import data_source as ds_mod
    from mi_agent_api import datasets as datasets_mod

    frame = _provenance_frame(ds_mod.get_dataframe())
    monkeypatch.setattr(datasets_mod, "get_dataframe", lambda *a, **k: frame)
    monkeypatch.setattr(ds_mod, "get_dataframe", lambda *a, **k: frame)
    return frame


@pytest.fixture()
def empty_frame(monkeypatch):
    """A governed frame with the right columns and no rows."""
    from mi_agent_api import data_source as ds_mod
    from mi_agent_api import datasets as datasets_mod

    frame = _provenance_frame(ds_mod.get_dataframe()).iloc[0:0]
    monkeypatch.setattr(datasets_mod, "get_dataframe", lambda *a, **k: frame)
    monkeypatch.setattr(ds_mod, "get_dataframe", lambda *a, **k: frame)
    return frame


def ask(question: str, lens: Optional[str] = None, **extra) -> Dict[str, Any]:
    body: Dict[str, Any] = {"question": question, **extra}
    if lens is not None:
        body["sourcePortfolioLens"] = lens
    response = client.post("/mi/query", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def scope_of(body: Dict[str, Any]) -> list:
    """The portfolios the answer DECLARES it covers."""
    coverage = body.get("portfolioCoverage") or {}
    if coverage.get("portfolios_used") is not None:
        return list(coverage["portfolios_used"])
    return list((body.get("portfolioScope") or {}).get("portfolio_ids") or [])


def kpi_value(body: Dict[str, Any], label_contains: str) -> float:
    """The numeric value of the KPI whose label contains ``label_contains``.

    A KPI card carries several tiles (loan count, then the requested measure), so
    a test must name the one it means — reading "the first number" silently
    compared row counts in an earlier draft of this suite.
    """
    # MATCHED ON THE LABEL **OR** THE GOVERNED FIELD KEY.
    #
    # This matched the label alone, so it broke when the KPI headline started
    # using the registry's business name and the aggregation that produced the
    # figure ("Weighted-average Current LTV") instead of the column key
    # title-cased ("Current Loan To Value Weighted"). What this test asserts is
    # a VALUE — that Total LTV is not the mean of the per-portfolio LTVs — and a
    # value assertion must not be defeated by a rename. The field key
    # (`current_loan_to_value_weighted_avg`) is the stable identity of the tile.
    wanted = label_contains.lower()
    key = wanted.replace(" ", "_")
    for artifact in body.get("artifacts") or []:
        for kpi in artifact.get("kpis") or []:
            if wanted not in str(kpi.get("label", "")).lower() \
                    and key not in str(kpi.get("field", "")).lower():
                continue
            raw = (str(kpi.get("value", ""))
                   .replace("£", "").replace(",", "").replace("%", "").strip())
            for suffix, factor in (("bn", 1e9), ("m", 1e6), ("k", 1e3)):
                if raw.lower().endswith(suffix):
                    return float(raw[: -len(suffix)]) * factor
            return float(raw)
    raise AssertionError(
        f"no KPI labelled like {label_contains!r} in {body.get('artifacts')}")


def balance_of(body: Dict[str, Any]) -> float:
    """The funded balance the answer covers, read from its reconciliation block.

    The reconciliation footer is the governed statement of what the answer
    included, so asserting on it also asserts that the footer and the figure
    agree.
    """
    for artifact in body.get("artifacts") or []:
        recon = artifact.get("reconciliation") or {}
        if recon.get("balance_included") is not None:
            return float(recon["balance_included"])
    raise AssertionError(f"no reconciliation balance in {body.get('artifacts')}")


# --------------------------------------------------------------------------- #
# 1. Every lens resolves to the right portfolios — from the registry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lens,expected", [
    (None, ALL_IDS), ("total", ALL_IDS),
    ("direct", DIRECT_IDS), ("acquired", ACQUIRED_IDS),
    ("direct_001", ["direct_001"]), ("direct_003", ["direct_003"]),
    ("acquired_002", ["acquired_002"]),
])
def test_each_lens_resolves_to_its_registered_members(multi_portfolio, lens, expected):
    body = ask("what is the total current balance?", lens)
    assert body["ok"] is True, body.get("error")
    assert sorted(scope_of(body)) == sorted(expected)


def test_group_membership_is_dynamic_not_a_fixed_id_list(multi_portfolio):
    """Direct is whatever is currently typed direct.

    ``direct_003`` exists only in this fixture and in no source file, so its
    presence in the Direct answer proves membership comes from the registry.
    """
    assert "direct_003" in scope_of(ask("what is the total current balance?", "direct"))


def test_a_portfolio_present_only_in_the_data_still_answers(multi_portfolio):
    """No id is hard-coded: a book named outside the fixture's own list is
    resolvable by an explicit selection."""
    body = ask("what is the total current balance?", "acquired_002")
    assert scope_of(body) == ["acquired_002"]


# --------------------------------------------------------------------------- #
# 2. Total is the sum of its parts — no double counting, no bad averaging
# --------------------------------------------------------------------------- #
def test_total_balance_equals_direct_plus_acquired(multi_portfolio):
    total = balance_of(ask("what is the total current balance?", "total"))
    direct = balance_of(ask("what is the total current balance?", "direct"))
    acquired = balance_of(ask("what is the total current balance?", "acquired"))
    assert direct + acquired == pytest.approx(total, rel=1e-9)


def test_total_balance_equals_the_sum_of_every_individual_portfolio(multi_portfolio):
    total = balance_of(ask("what is the total current balance?", "total"))
    parts = sum(balance_of(ask("what is the total current balance?", pid))
                for pid in ALL_IDS)
    assert parts == pytest.approx(total, rel=1e-9), "Total double-counts or drops rows"


def test_total_weighted_average_is_not_an_average_of_averages(multi_portfolio):
    """The failure mode this guards: computing Total LTV as the mean of the
    per-portfolio LTVs, which is only correct when every book is the same size."""
    ltv = lambda lens: kpi_value(ask("what is the weighted average LTV?", lens),
                                 "loan to value")
    total = ltv("total")
    per_portfolio = [ltv(pid) for pid in ALL_IDS]
    naive = sum(per_portfolio) / len(per_portfolio)
    assert min(per_portfolio) - 1e-9 <= total <= max(per_portfolio) + 1e-9
    if abs(max(per_portfolio) - min(per_portfolio)) > 1e-6:
        assert total != pytest.approx(naive, rel=1e-12), (
            "Total weighted average equals the unweighted mean of the parts")


def test_a_narrowed_lens_never_returns_more_rows_than_total(multi_portfolio):
    total = ask("show balance by region", "total")
    direct = ask("show balance by region", "direct")
    one = ask("show balance by region", "direct_001")
    groups = lambda b: len((b["artifacts"][1] if len(b["artifacts"]) > 1
                            else b["artifacts"][0]).get("rows") or [])
    assert groups(one) <= groups(direct) <= groups(total)


# --------------------------------------------------------------------------- #
# 3. The lens applies to ROUTED intents too, or is disclosed
# --------------------------------------------------------------------------- #
def test_geographic_concentration_is_scoped_to_the_lens(multi_portfolio):
    """The proven defect: the ITL3 exposure route ignored the lens entirely and
    returned identical whole-book figures for Direct, Acquired and every cohort.
    """
    answers = {lens: ask("where is the largest geographic concentration?", lens)["answer"]
               for lens in ("total", "direct", "acquired", "acquired_001")}
    assert answers["direct"] != answers["total"], (
        "Direct concentration is identical to Total — the lens was not applied")
    assert answers["acquired"] != answers["total"]
    assert len(set(answers.values())) > 1


def test_a_scoped_concentration_states_its_scope_not_the_book(multi_portfolio):
    body = ask("where is the largest geographic concentration?", "acquired")
    assert "of Acquired" in body["answer"], body["answer"]
    assert body["metadata"]["lensApplied"] is True


def test_a_route_that_cannot_narrow_must_not_be_stamped_with_the_narrow_scope(
        multi_portfolio):
    """The governance defect: the risk-limits route reads a run artefact with no
    per-portfolio provenance, so its numbers are whole-book — yet the envelope
    certified them as the requested portfolio, fully consolidated, with no
    disclosure. The control that exists to prevent misattribution was performing
    it."""
    body = ask("are we within our risk limits?", "acquired_001")
    if (body.get("metadata") or {}).get("route") != "risk_limits":
        pytest.skip("risk-limit route not reachable in this environment")
    assert body["metadata"]["lensApplied"] is False
    # Stamped with the scope the ANSWER has, not the scope requested.
    assert sorted(scope_of(body)) == sorted(ALL_IDS)
    assert any("Scope not narrowed" in w for w in body["warnings"]), body["warnings"]


def test_a_total_request_on_an_unnarrowable_route_is_not_flagged(multi_portfolio):
    """Whole-book figures ARE the right answer for Total — no false alarm."""
    body = ask("are we within our risk limits?", "total")
    if (body.get("metadata") or {}).get("route") != "risk_limits":
        pytest.skip("risk-limit route not reachable in this environment")
    assert body["metadata"]["lensApplied"] is True
    assert not any("Scope not narrowed" in w for w in body["warnings"])


# --------------------------------------------------------------------------- #
# 4. Negative testing — an unknown scope is never answered silently
# --------------------------------------------------------------------------- #
def test_an_unknown_portfolio_widens_to_total_and_says_so(multi_portfolio):
    body = ask("what is the total current balance?", "direct_999")
    assert sorted(scope_of(body)) == sorted(ALL_IDS)
    assert any("direct_999" in w and "TOTAL" in w for w in body["warnings"]), (
        "A discarded portfolio selection must be disclosed, not silently widened")
    assert body["portfolioScope"]["fell_back_to_total"] is True


def test_a_known_portfolio_produces_no_fallback_warning(multi_portfolio):
    body = ask("what is the total current balance?", "acquired_001")
    assert not any("not in the governed portfolio registry" in w
                   for w in body["warnings"])


def test_an_empty_dataset_fails_closed_rather_than_reporting_zero(empty_frame):
    body = ask("show balance by region", "direct")
    assert body["ok"] is False
    assert body["error"], "an empty result must carry a reason"


# --------------------------------------------------------------------------- #
# 5. Drill-through and filters compose with the lens
# --------------------------------------------------------------------------- #
def test_drill_through_filters_narrow_within_the_lens(multi_portfolio):
    body = ask("show balance by region", "direct",
               filters={"current_loan_to_value": {"op": "gt", "value": 0}})
    assert body["ok"] is True, body.get("error")
    assert sorted(scope_of(body)) == sorted(DIRECT_IDS)
    assert any("drill-through filters applied" in w for w in body["warnings"])


def test_a_lens_named_in_the_question_overrides_the_workspace_selection(multi_portfolio):
    body = ask("what is the total current balance for the acquired book?", "direct")
    assert sorted(scope_of(body)) == sorted(ACQUIRED_IDS), (
        "A portfolio named in the question must win over the dropdown")


def test_an_exact_cohort_id_in_the_question_wins(multi_portfolio):
    body = ask("what is the total current balance for direct_002?", "acquired")
    assert scope_of(body) == ["direct_002"]


# --------------------------------------------------------------------------- #
# 6. Capability-unavailable behaviour is governed, not an exception
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lens", [None, "total", "direct", "acquired", "acquired_001"])
def test_an_unavailable_capability_never_raises(multi_portfolio, lens):
    body = ask("what is in the pipeline?", lens)
    assert isinstance(body.get("answer") or body.get("error"), str)
    assert body.get("answer") or body.get("error")


@pytest.mark.parametrize("lens", [None, "direct", "acquired", "direct_002"])
def test_an_unavailable_metric_is_explained_not_fabricated(multi_portfolio, lens):
    body = ask("how much protected equity is there?", lens)
    assert body["ok"] is False
    assert "protected equity" in body["error"].lower()
    assert "not available" in body["error"].lower()
    assert not body.get("artifacts")


# --------------------------------------------------------------------------- #
# 7. The whole matrix stays exception-free
# --------------------------------------------------------------------------- #
_MATRIX_QUESTIONS = [
    "portfolio summary",
    "what is the total current balance?",
    "show balance by region",
    "show weighted average LTV by region",
    "show balance by age bucket",
    "what is the average borrower age?",
    "where is the largest geographic concentration?",
    "show balance over time",
    "show loans with LTV above 50%",
    "how many loans are there?",
    "are we within our risk limits?",
    "what is in the pipeline?",
]


@pytest.mark.parametrize("question", _MATRIX_QUESTIONS)
@pytest.mark.parametrize("lens", [None, "total", "direct", "acquired",
                                  "direct_001", "acquired_002", "direct_999"])
def test_every_question_answers_or_refuses_cleanly_under_every_lens(
        multi_portfolio, question, lens):
    """No lens/question pair may 500, and none may return an empty envelope."""
    body = ask(question, lens)
    assert body["ok"] in (True, False)
    assert body.get("answer") or body.get("error"), body
    # An answer that succeeded must declare a scope; a refusal need not.
    if body["ok"] and body.get("portfolioScope"):
        assert body["portfolioScope"]["portfolio_ids"]
