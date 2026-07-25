"""Tests for the demonstration's exported metrics and the two answering surfaces.

SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.

These run against the EXPORTED fixtures, so they test what the film will actually
show. They are skipped (not failed) when the demonstration has not been built
yet — the build is a ~2-minute pipeline run, so it is a deliberate step:

    python -m demo_platform.run_demo --all

`test_demo_platform_safety.py` covers the safety gate; this module covers the
metric reconciliation, the MI Agent / Copilot agreement, and the assembled
platform view.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo_platform import assertions, config as cfg   # noqa: E402

_BUILD_HINT = (
    "the demonstration has not been built — run "
    "`python -m demo_platform.run_demo --all`"
)


@pytest.fixture(scope="module")
def export():
    path = cfg.export_dir() / "demo_metrics.json"
    if not path.exists():
        pytest.skip(_BUILD_HINT)
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def total(export):
    return export["metrics"]["lenses"]["total"]


@pytest.fixture(scope="module")
def platform_current():
    path = (cfg.local_blob_root() / "processed" / "platform" / cfg.CLIENT_ID
            / cfg.CURRENT_PERIOD.reporting_date / "platform_canonical_typed.csv")
    if not path.exists():
        pytest.skip(_BUILD_HINT)
    return pd.read_csv(path, low_memory=False)


# --------------------------------------------------------------------------- #
# The reconciliation gate as a test
# --------------------------------------------------------------------------- #
def test_the_reconciliation_gate_passes(export):
    """Every narrative claim must be true of the produced data."""
    report = assertions.run(export, verbose=False)
    failed = [c for c in report["checks"] if not c["passed"]]
    assert not failed, "\n".join(f"{c['check']}: {c['detail']}" for c in failed)
    assert report["checksRun"] >= 30


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def test_both_portfolios_retain_their_identity(platform_current):
    ids = set(platform_current["source_portfolio_id"].astype(str).str.strip().unique())
    assert ids == {p.source_portfolio_id for p in cfg.PORTFOLIOS}


def test_provenance_is_present_on_every_row(platform_current):
    for field in ("source_portfolio_id", "source_portfolio_type", "portfolio_cohort"):
        assert field in platform_current.columns
        assert platform_current[field].notna().all(), field


def test_the_composite_platform_key_is_unique(platform_current):
    assert "platform_loan_key" in platform_current.columns
    assert platform_current["platform_loan_key"].duplicated().sum() == 0


def test_consolidated_loan_count_and_balance(platform_current, total):
    lo, hi = cfg.LOAN_COUNT_BAND
    assert lo <= len(platform_current) <= hi
    tape_total = float(
        pd.to_numeric(platform_current["current_outstanding_balance"], errors="coerce")
        .fillna(0.0).sum())
    assert abs(tape_total - total["movement"]["current"]["funded_balance"]) < 0.05


def test_consolidated_balance_equals_the_sum_of_the_portfolios(platform_current):
    ids = platform_current["source_portfolio_id"].astype(str).str.strip()
    balance = pd.to_numeric(
        platform_current["current_outstanding_balance"], errors="coerce").fillna(0.0)
    per_portfolio = sum(
        float(balance[ids == p.source_portfolio_id].sum()) for p in cfg.PORTFOLIOS)
    assert abs(per_portfolio - float(balance.sum())) < 0.05


# --------------------------------------------------------------------------- #
# Movement metrics
# --------------------------------------------------------------------------- #
def test_monthly_funded_balance_change(total):
    movement = total["movement"]["delta"]["funded_balance"]
    assert cfg.TARGET_MOVEMENT_TOTAL.ok(movement), movement


def test_portfolio_contributions_sum_to_the_movement(total):
    mv = total["movement"]
    cohorts = {c["id"]: c["delta"] for c in mv["cohortMovements"]}
    assert cfg.TARGET_MOVEMENT_A.ok(cohorts[cfg.PORTFOLIO_A.source_portfolio_id])
    assert cfg.TARGET_MOVEMENT_B.ok(cohorts[cfg.PORTFOLIO_B.source_portfolio_id])
    assert abs(sum(cohorts.values()) - mv["delta"]["funded_balance"]) < 0.05


def test_regional_contributions_sum_to_the_movement(total):
    mv = total["movement"]
    total_regions = sum(c["delta"] for c in mv["regionContributions"])
    assert abs(total_regions - mv["delta"]["funded_balance"]) < 5_000.0


def test_the_primary_region_is_the_south_east(total):
    mv = total["movement"]
    assert mv["primaryRegion"]["region"] == cfg.PRIMARY_MOVEMENT_REGION
    assert mv["primaryRegionShare"] >= cfg.PRIMARY_REGION_MIN_SHARE
    assert mv["primaryRegionLead"] >= cfg.PRIMARY_REGION_MIN_LEAD


def test_the_completions_attribution_is_evidenced(total):
    """The answer may only say "driven by completions" when that is measurable."""
    mv = total["movement"]
    assert mv["completionsEvidenced"] is True
    assert mv["completionsShareOfPrimaryRegion"] >= 0.5


def test_weighted_average_ltv_and_borrower_age(total):
    cur = total["movement"]["current"]
    assert cfg.TARGET_WA_LTV_CURRENT.ok(cur["wa_ltv_points"]), cur["wa_ltv_points"]
    assert cfg.TARGET_BORROWER_AGE_CURRENT.ok(cur["avg_borrower_age"])


def test_ltv_is_broadly_stable_and_age_rises(total):
    delta = total["movement"]["delta"]
    assert abs(delta["wa_ltv_points"]) < cfg.WA_LTV_STABILITY_MAX_CHANGE_POINTS
    assert delta["avg_borrower_age"] > 0


def test_three_governed_reporting_periods(total):
    periods = [p["period"] for p in total["series"]]
    assert periods == [p.period for p in cfg.PERIODS]


def test_per_portfolio_lenses_reconcile_to_the_total(export):
    lenses = export["metrics"]["lenses"]
    total_balance = lenses["total"]["summary"]["metrics"]["funded_balance"]
    per_lens = sum(
        lenses[p.source_portfolio_id]["summary"]["metrics"]["funded_balance"]
        for p in cfg.PORTFOLIOS)
    assert abs(per_lens - total_balance) < 0.05


# --------------------------------------------------------------------------- #
# The two surfaces
# --------------------------------------------------------------------------- #
def test_the_mi_agent_answers_both_required_questions(export):
    turns = export["miAgent"]["turns"]
    assert [t["question"] for t in turns] == list(cfg.MI_QUESTIONS)
    for turn in turns:
        assert turn["ok"] is True
        assert turn["answer"] and len(turn["answer"]) > 80
        assert turn["artifacts"], turn["question"]


def test_the_copilot_surface_answers_both_required_questions(export):
    answers = export["copilot"]["answers"]
    assert len(answers) == len(cfg.MI_QUESTIONS)
    for answer in answers:
        assert answer["answer"] and len(answer["answer"]) > 80


def test_both_surfaces_report_identical_figures(export):
    """The two surfaces must never diverge — they share one execution path."""
    checks = assertions._surface_agreement_checks(export)
    failed = [c for c in checks if not c.passed]
    assert not failed, "\n".join(f"{c.name}: {c.detail}" for c in failed)


def test_the_mi_agent_answer_quotes_the_governed_figures(export, total):
    """The summary answer must contain the loan count and the funded balance."""
    answer = export["miAgent"]["turns"][0]["answer"]
    metrics = total["summary"]["metrics"]
    assert f"{int(metrics['loan_count']):,}" in answer
    assert "43.2%" in answer or "43.1%" in answer
    assert cfg.PRIMARY_MOVEMENT_REGION in answer


def test_the_movement_answer_states_the_narrative(export):
    answer = export["miAgent"]["turns"][1]["answer"]
    assert "increased" in answer
    assert cfg.PRIMARY_MOVEMENT_REGION in answer
    assert "broadly stable" in answer
    for p in cfg.PORTFOLIOS:
        assert p.label in answer, p.label


def test_no_internal_query_detail_reaches_the_fixture(export):
    """The captured envelopes must not carry implementation detail.

    If a query spec or engine metadata is not in the fixture, it cannot appear on
    screen — which is why the capture step strips it rather than the video layer
    hiding it.
    """
    banned_keys = {"spec", "validation", "resolved_fields", "trace", "prompt",
                   "interpreted", "metadata"}
    for turn in export["miAgent"]["turns"]:
        assert not (banned_keys & set(turn)), sorted(banned_keys & set(turn))
        for artifact in turn["artifacts"]:
            assert "source" not in artifact
            assert "spec" not in artifact


def test_the_copilot_download_url_is_redacted(export):
    action = export["copilot"]["artefactAction"]
    if action is None:
        pytest.skip("no investor deck was published in this build")
    assert "<redacted>" in action["downloadUrl"]
    assert "sig=" not in action["downloadUrl"]


def test_the_copilot_actions_are_the_three_that_exist(export):
    """Only the v1 capabilities in deploy/copilot-agent/ai-plugin.json."""
    assert export["copilot"]["actions"] == [
        "askTraktMi", "getLatestInvestorDeck", "getLatestCanonicalTape",
    ]


# --------------------------------------------------------------------------- #
# Artefacts
# --------------------------------------------------------------------------- #
def test_the_artefact_catalogue_records_availability_honestly():
    path = cfg.artefact_dir() / "artefact_catalogue.json"
    if not path.exists():
        pytest.skip(_BUILD_HINT)
    catalogue = json.loads(path.read_text(encoding="utf-8"))
    for key, artefact in catalogue.items():
        if not isinstance(artefact, dict) or "available" not in artefact:
            continue
        if not artefact["available"]:
            # An unavailable artefact must say why, so the film can omit it.
            assert artefact.get("reason"), f"{key} is unavailable with no reason"


def test_the_core_artefacts_were_produced():
    path = cfg.artefact_dir() / "artefact_catalogue.json"
    if not path.exists():
        pytest.skip(_BUILD_HINT)
    catalogue = json.loads(path.read_text(encoding="utf-8"))
    for key in ("canonicalTape", "validationReport", "investorDeck", "auditManifest"):
        assert catalogue[key]["available"] is True, (key, catalogue[key].get("reason"))


def test_the_validation_report_counts_the_rows_it_validated():
    path = cfg.artefact_dir() / "artefact_catalogue.json"
    if not path.exists():
        pytest.skip(_BUILD_HINT)
    report = json.loads(path.read_text(encoding="utf-8"))["validationReport"]
    lo, hi = cfg.LOAN_COUNT_BAND
    assert lo <= report["rowsValidated"] <= hi, report["rowsValidated"]
    assert report["businessRuleExceptions"] > 0, (
        "a run with no exceptions would be showing a validation step that cannot fail"
    )
    assert report["exceptionRatePct"] < 5.0, report["exceptionRatePct"]
