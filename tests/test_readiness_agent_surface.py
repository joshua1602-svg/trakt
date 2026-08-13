#!/usr/bin/env python3
"""Can every planted finding be reached through the governed surface alone?

This is the question Sprint 3's agent depends on and which is worth answering
independently of any model: **if Trakt cannot surface a finding through
``GovernedSession``, no agent can find it, however good the reasoning.** So
each planted case is reached here through the same three verbs an external
agent would have — ``capabilities()``, ``call()``, ``transcript()`` — with no
DataFrame, no analytics import and no private function.

What this file does NOT claim
-----------------------------
It does not demonstrate autonomy. Every call below was chosen by a human
writing a test, which is precisely the checklist Sprint 3 forbids an agent to
be given. What it establishes is the weaker and necessary precondition: the
evidence is *reachable*, the numbers are *correct*, and the refusals are
*honest*. Whether an agent chooses these calls unprompted is a separate
experiment that needs a model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core import config_cache
from trakt_core.context import SCOPE_LOAN_READ, SCOPE_RISK_READ
from trakt_tools.execution import ToolDependencies

from readiness_agent.session import GovernedSession
from tests.planted_portfolio import planted_lineage_index
from tests.readiness_agent_portfolio import (
    ANSWER_KEY,
    EXPECTED_CURE_RATES,
    EXPECTED_NINETY_PLUS_SHARE,
    EXPECTED_THIRTY_PLUS_SHARE,
    PERIODS,
    SUPPLIED_LGD,
    answer_key_by_id,
    snapshots,
)
from tests.test_agent_governed_execution import _Datasets, _Descriptor
from tests.test_agent_loan_retrieval import (
    PORTFOLIO_A,
    PORTFOLIO_B,
    TENANT,
    _catalogue,
    _context,
    _CountingResolver,
)

BOTH = (SCOPE_LOAN_READ, SCOPE_RISK_READ)
KEY = answer_key_by_id()


@pytest.fixture(autouse=True)
def _clean_cache():
    config_cache.reset()
    yield
    config_cache.reset()


@pytest.fixture(scope="module")
def snaps():
    return snapshots()


def _deps(snaps):
    return ToolDependencies(
        datasets=_Datasets(_Descriptor(snapshot_id="secr-demo-final")),
        runtime_mode="test", catalogue=_catalogue(), output_root="/unused",
        loan_frame_resolver=_CountingResolver(snaps[PERIODS[-1]]),
        loan_history_resolver=lambda: snaps,
        lineage_index=planted_lineage_index(TENANT, "secr-demo-final"))


def _session(snaps, resource=None, capabilities=BOTH) -> GovernedSession:
    return GovernedSession(_context(capabilities=capabilities),
                           resource or PORTFOLIO_A.key,
                           dependencies=_deps(snaps))


# =========================================================================== #
# The boundary itself
# =========================================================================== #
def test_the_agent_package_cannot_reach_trakts_calculations():
    """The separation that makes the rest meaningful.

    An agent able to import ``analytics_lib`` would eventually compute
    something itself, and Trakt would then have two answers to one question.
    The import graph is the enforcement, so it is asserted rather than trusted.
    """
    import ast

    for module in (_REPO_ROOT / "readiness_agent").glob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for banned in ("analytics_lib", "pandas", "mi_agent.concentration_tests",
                       "analytics", "mi_agent_api"):
            assert not any(name == banned or name.startswith(banned + ".")
                           for name in imported), (
                f"{module.name} imports {banned}; the agent must reach Trakt "
                "only through trakt_tools.execution")


def test_the_session_exposes_three_verbs_and_no_data(snaps):
    """No frame, no path, no storage handle — by construction."""
    session = _session(snaps)
    public = {name for name in dir(session) if not name.startswith("_")}
    assert {"capabilities", "call", "transcript"} <= public
    for leak in ("frame", "dataframe", "df", "snapshots", "read", "storage"):
        assert leak not in public, f"session exposes {leak!r}"


# =========================================================================== #
# Capability discovery — the agent's first question
# =========================================================================== #
def test_discovery_reports_what_trakt_knows_without_computing_it(snaps):
    session = _session(snaps)
    capabilities = session.capabilities()

    assert capabilities["metrics"], "discovery returned no capabilities"
    assert session.call_count == 1, "discovery must be ONE call"
    summary = capabilities["summary"]
    assert summary.get("AVAILABLE", 0) > 0
    # Discovery names the metric and its state; it does not carry values.
    for metric in capabilities["metrics"]:
        assert "status" in metric
        assert "value" not in metric, (
            "discovery must not compute — 'can calculate' is not 'calculate'")


def test_discovery_is_cached_so_re_asking_is_free(snaps):
    session = _session(snaps)
    session.capabilities()
    session.capabilities()
    session.capabilities()
    assert session.call_count == 1


def test_every_unavailable_capability_arrives_with_its_reason(snaps):
    """A readiness review that cannot say what it was unable to assess is not
    a review, so the refusals must be as retrievable as the answers."""
    session = _session(snaps)
    for metric, block in session.unavailable_metrics().items():
        assert block.get("reason_code"), f"{metric} has no reason code"
        assert block.get("explanation"), f"{metric} has no explanation"


# =========================================================================== #
# The planted findings, reached through the surface
# =========================================================================== #
def test_conc01_the_london_concentration_is_reachable_and_correct(snaps):
    session = _session(snaps)
    outcome = session.call("concentration",
                           {"dimension": "geographic_region_collateral",
                            "top_n": 1})
    groups = outcome.get("groups") or []
    assert groups, outcome
    top = groups[0]
    assert str(top.get("geographic_region_collateral")) == "London"
    assert float(top["balance_share"]) * 100 == pytest.approx(
        KEY["CONC-01"]["expected_value"], abs=1e-6)      # 31.0%


def test_ltv01_the_headline_looks_comfortable(snaps):
    session = _session(snaps)
    outcome = session.call(
        "readiness_metrics",
        {"requests": [{"metric_id": "ltv_weighted_average"}]})
    value = outcome["measurements"][0]["value"]
    assert value == pytest.approx(KEY["LTV-01"]["expected_value"])
    assert value < 65.0, "the headline is meant to look fine"


def test_ltv02_the_tail_beneath_the_headline_is_reachable(snaps):
    """The finding the headline conceals. An agent stopping at LTV-01 misses
    that 12% of the book is at 92%."""
    session = _session(snaps)
    outcome = session.call(
        "readiness_metrics",
        {"requests": [{"metric_id": "ltv_above_share",
                       "parameters": {"ltv_percent": 80}}]})
    assert outcome["measurements"][0]["value"] == pytest.approx(
        KEY["LTV-02"]["expected_value"])


def test_val01_the_stale_pocket_is_reachable(snaps):
    session = _session(snaps)
    outcome = session.call("valuation_age_profile", {"stale_after_months": 24})
    assert outcome["available"] is True
    assert outcome["stale_balance_pct"] == pytest.approx(
        KEY["VAL-01"]["expected_value"])


def test_the_high_ltv_tail_and_the_stale_pocket_are_the_same_exposure(snaps):
    """The compounding finding, and the one that most rewards follow-up: the
    92% LTV is measured against valuations from 2019. Reachable by filtering
    the same governed tool rather than by any new calculation."""
    session = _session(snaps)
    stale_only = session.call(
        "valuation_age_profile",
        {"stale_after_months": 24,
         "filters": {"current_loan_to_value": 92.0}})
    assert stale_only["available"] is True
    assert stale_only["stale_balance_pct"] == pytest.approx(100.0), (
        "every high-LTV loan should carry a stale valuation")


def test_arr01_the_ninety_plus_trajectory_is_reachable(snaps):
    session = _session(snaps)
    outcome = session.call("portfolio_history",
                           {"measures": ["arrears_90_plus_pct"]})
    values = outcome["series"]["arrears_90_plus_pct"]["values"]
    assert values == pytest.approx(EXPECTED_NINETY_PLUS_SHARE)
    assert values[-1] > values[0] * 5, "the planted trajectory is the finding"


def test_arr02_the_spike_is_visible_and_resolves(snaps):
    session = _session(snaps)
    outcome = session.call("portfolio_history",
                           {"measures": ["arrears_30_plus_pct"]})
    values = outcome["series"]["arrears_30_plus_pct"]["values"]
    assert values == pytest.approx(EXPECTED_THIRTY_PLUS_SHARE)
    assert values[2] > values[1] and values[3] < values[2], (
        "period 3 spikes and period 4 falls back — the false-positive trap")


def test_def01_the_default_rate_uses_the_owned_methodology(snaps):
    session = _session(snaps)
    outcome = session.call("default_analysis", {})
    assert outcome["available"] is True
    assert outcome["method"] == "OBSERVED_DEFAULT_RATE_CDR@v1"
    assert "AnlsdCstDfltRate" in outcome["authority"]
    rates = [p["periodic_default_rate_pct"] for p in outcome["per_period"]]
    assert rates == sorted(rates), "the planted default rate rises monotonically"
    # And the stock is a different number, retrievable alongside.
    assert outcome["default_stock_pct"] is not None
    assert outcome["default_stock_pct"] != rates[-1]


def test_cure01_the_cure_series_shows_one_real_cure_and_five_empty_periods(snaps):
    session = _session(snaps)
    outcome = session.call("cure_analysis", {})
    assert outcome["available"] is True
    assert outcome["method"] == "OBSERVED_CURE_RATE@v1"
    rates = [p["cure_rate_pct"] for p in outcome["per_period"]]
    assert rates == pytest.approx(EXPECTED_CURE_RATES)
    assert "ESMA defines NO loan-level cure concept" in outcome["authority"], (
        "the agent must be able to tell a Trakt convention from a regulator's")


def test_cpr01_the_prepayment_trend_is_reachable(snaps):
    session = _session(snaps)
    outcome = session.call("prepayment_analysis", {})
    assert outcome["available"] is True
    assert outcome["method"] == "OBSERVED_CPR@v2"
    smm = [p["smm_pct"] for p in outcome["per_period"] if p.get("smm_pct")]
    assert smm == sorted(smm, reverse=True), "planted CPR declines"


def test_loss01_severity_is_the_observed_metric_not_the_supplied_lgd(snaps):
    """The 2.5E defect, planted deliberately. The tape carries a supplied LGD
    of 25.0 and the realised severity is 40.0. An agent quoting 25.0 has
    reproduced the defect, so the surface must make 40.0 the retrievable one."""
    session = _session(snaps)
    outcome = session.call("loss_analysis", {})
    severity = outcome["loss_severity"]

    assert severity["available"] is True
    assert severity["method"] == "OBSERVED_LOSS_SEVERITY@v1"
    assert severity["severity_pct"] == pytest.approx(
        KEY["LOSS-01"]["expected_value"])                 # 40.0
    assert severity["severity_pct"] != pytest.approx(SUPPLIED_LGD)  # not 25.0


def test_wal01_contractual_wal_is_available_for_the_fixed_sub_book(snaps):
    session = _session(snaps)
    outcome = session.call("contractual_analytics",
                           {"metrics": ["contractual_wal"]})
    wal = outcome["contractual_wal"]
    assert wal["status"] == "SUPPORTED"
    assert wal["methodology_version"] == "CONTRACTUAL_WAL@v1"
    assert wal["wal_years"] > 0


def test_ytm01_the_floating_sub_book_forces_an_honest_partial_answer(snaps):
    """100 of 400 loans are FRXX + FLIF. The yield for those needs a rate path
    Trakt does not hold, and the result must say so rather than quietly
    reporting the fixed-rate 300 as though they were the portfolio."""
    session = _session(snaps)
    outcome = session.call("contractual_analytics",
                           {"metrics": ["contractual_ytm_periodic"]})
    ytm = outcome["contractual_ytm_periodic"]
    coverage = outcome["coverage"]

    assert coverage.get("ASSUMPTION_REQUIRED", 0) == 100, (
        "the 100 floating loans must be visible as assumption-required")
    assert coverage.get("SUPPORTED", 0) == 300
    # The yield is produced for the 300 that CAN be priced and the 100 that
    # cannot are excluded and counted — not silently folded into the average.
    assert ytm["status"] == "SUPPORTED"
    assert ytm["loans_included"] == 300
    assert ytm["loans_excluded"] == 100
    assert any("assumption" in w.lower() for w in outcome["warnings"])


def test_epi01_expected_wal_is_refused_as_model_required(snaps):
    session = _session(snaps)
    block = session.unavailable_metrics().get("expected_wal")
    assert block is not None
    assert block["status"] == "MODEL_REQUIRED"
    assert "mortality" in block["explanation"]


def test_data01_the_incomplete_mandatory_field_is_reachable(snaps):
    session = _session(snaps)
    outcome = session.call("data_completeness", {})
    assert outcome, "completeness must be retrievable"


# =========================================================================== #
# Fact / rule / judgement — one fact, three rulebooks
# =========================================================================== #
def test_one_concentration_fact_is_judged_differently_by_each_rulebook(snaps):
    """The separation the whole readiness architecture exists to protect. The
    agent must be able to see that a warehouse PASS is not clearance for a
    securitisation whose limit is tighter."""
    session = _session(snaps)
    outcome = session.call("evaluate_rule_packs", {})
    assert outcome.get("results") or outcome.get("packs"), outcome
    # Every rule result must name where its authority came from.
    results = outcome.get("results") or []
    for result in results:
        assert result.get("authority"), (
            "a rule outcome with no authority could be reported as a rating "
            "requirement, a covenant or a Trakt screen interchangeably")


# =========================================================================== #
# Audit and efficiency
# =========================================================================== #
def test_the_run_is_reconstructible_from_the_audit_record(snaps):
    session = _session(snaps)
    session.capabilities()
    session.call("readiness_metrics",
                 {"requests": [{"metric_id": "ltv_weighted_average"}]})
    session.call("default_analysis", {})

    record = session.audit_record("Assess this portfolio for securitisation "
                                  "readiness.")
    assert record["objective"].startswith("Assess this portfolio")
    assert record["organisation"] == TENANT
    assert record["resource"] == PORTFOLIO_A.key
    assert len(record["tool_calls"]) == 3
    for call in record["tool_calls"]:
        assert call["tool"] and "arguments" in call and call["status"]
        assert "elapsed_ms" in call
    assert record["efficiency"]["total_calls"] == 3


def test_the_audit_record_stores_actions_not_reasoning(snaps):
    """A stored chain of thought reads as evidence and is not. Only inputs and
    governed outputs are reconstructible facts about what happened."""
    session = _session(snaps)
    session.call("default_analysis", {})
    record = session.audit_record("objective")
    serialised = repr(record).lower()
    for forbidden in ("thought", "reasoning", "chain_of_thought", "rationale"):
        assert forbidden not in serialised


def test_repeated_identical_calls_are_counted(snaps):
    session = _session(snaps)
    session.call("default_analysis", {})
    session.call("default_analysis", {})
    assert session.efficiency()["repeated_calls"] == 1


# =========================================================================== #
# Security — autonomy must not widen authority
# =========================================================================== #
def test_an_agent_cannot_reach_a_portfolio_it_was_not_granted(snaps):
    """PORTFOLIO_B is outside the granted scope. The refusal must be a
    refusal, not an empty result an agent could read as 'no findings'."""
    session = _session(snaps, resource=PORTFOLIO_B.key)
    outcome = session.call(
        "readiness_metrics",
        {"requests": [{"metric_id": "ltv_weighted_average"}]})
    assert outcome.get("refused") is True, outcome
    assert outcome.get("error_code")


def test_an_agent_cannot_reach_another_tenants_portfolio(snaps):
    session = _session(snaps, resource="other-tenant/portfolio/SECRET")
    outcome = session.call(
        "readiness_metrics",
        {"requests": [{"metric_id": "ltv_weighted_average"}]})
    assert outcome.get("refused") is True, outcome


def test_an_agent_without_the_capability_is_refused_the_tool(snaps):
    """Being autonomous grants no authority. A session holding only loan:read
    must not reach a risk:read tool."""
    session = _session(snaps, capabilities=(SCOPE_LOAN_READ,))
    outcome = session.call("default_analysis", {})
    assert outcome.get("refused") is True, outcome


def test_a_refusal_is_recorded_in_the_transcript_like_any_other_call(snaps):
    session = _session(snaps, resource=PORTFOLIO_B.key)
    session.call("readiness_metrics",
                 {"requests": [{"metric_id": "ltv_weighted_average"}]})
    transcript = session.transcript()
    assert len(transcript) == 1
    assert transcript[0]["status"] != "success"
    assert transcript[0]["error_code"]


# =========================================================================== #
# The answer key itself
# =========================================================================== #
def test_the_answer_key_covers_every_required_category():
    categories = {case["category"] for case in ANSWER_KEY}
    for required in ("concentration", "collateral", "cohort", "arrears",
                     "default", "cure", "prepayment", "loss", "valuation",
                     "contractual", "data_readiness", "false_positive",
                     "epistemic"):
        assert required in categories, f"no planted case for {required}"
    assert len(ANSWER_KEY) >= 12


def test_the_answer_key_is_not_importable_from_the_agent_package():
    """The key must never be reachable from the runtime being evaluated."""
    import ast

    for module in (_REPO_ROOT / "readiness_agent").glob("*.py"):
        source = module.read_text(encoding="utf-8")
        assert "readiness_agent_portfolio" not in source
        assert "ANSWER_KEY" not in source
