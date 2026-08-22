#!/usr/bin/env python3
"""tests/test_asset_class_lifecycle.py — the governed asset-class handoff.

The architecture this pins: onboarding decides a portfolio's ASSET CLASS once,
writes it to the governed portfolio registry, and every MI consumer reads it
from there. Before this existed, the decision lived only in
``27_onboarding_context.json`` — which nothing downstream reads — so MI always
ran with an unknown asset class and could admit only ``cross_asset`` semantics.

The chain under test:

    onboarding decision
      -> portfolio_registry.yaml            (engine.onboarding_agent.portfolio_registry_writer)
      -> portfolio metadata overlay          (mi_agent.portfolio_metadata)
      -> PortfolioRecord.asset_class         (trakt_core.portfolio)
      -> PortfolioScope.asset_classes
      -> period-change field selection  AND  portfolio comparison

Both MI consumers are asserted, because the defect was that they read two
different, unconnected sources.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import portfolio_registry_writer as writer
from mi_agent.portfolio_metadata import normalise_asset_class
from trakt_core.portfolio import (
    FIELD_PORTFOLIO_ID, FIELD_PORTFOLIO_TYPE, PortfolioRecord, build_registry,
    resolve_scope)


# =========================================================================== #
# One vocabulary (step 5)
# =========================================================================== #
@pytest.mark.parametrize("declared,expected", [
    ("equity_release_mortgage", "equity_release"),   # onboarding's own dialect
    ("Equity Release Mortgage", "equity_release"),
    ("lifetime_mortgage", "equity_release"),
    ("equity_release", "equity_release"),            # already canonical
    ("sme_loan", "sme"),
    ("bridging", "bridge"),
    ("asset_finance", "equipment_leasing"),
    ("", None),
    (None, None),
])
def test_the_boundary_resolves_one_governed_vocabulary(declared, expected):
    assert normalise_asset_class(declared) == expected


def test_an_unrecognised_class_passes_through_rather_than_being_dropped():
    """The table resolves known synonyms; it does not restrict what a client may
    declare. A new asset class must not silently vanish."""
    assert normalise_asset_class("marine_finance") == "marine_finance"


# =========================================================================== #
# Onboarding writes it (step 3)
# =========================================================================== #
def test_onboarding_publishes_the_asset_class_it_decided(tmp_path):
    target = tmp_path / "portfolio_registry.yaml"
    writer.publish(target,
                   [{"source_portfolio_id": "direct_001"},
                    {"source_portfolio_id": "acquired_001"}],
                   asset_class="equity_release_mortgage")
    document = yaml.safe_load(target.read_text())
    assert document["portfolios"] == [
        {"source_portfolio_id": "direct_001", "asset_class": "equity_release"},
        {"source_portfolio_id": "acquired_001", "asset_class": "equity_release"}]


def test_a_per_portfolio_class_beats_the_client_default(tmp_path):
    """A client with two books of different asset classes needs no second
    mechanism."""
    target = tmp_path / "registry.yaml"
    writer.publish(target,
                   [{"source_portfolio_id": "erm_001"},
                    {"source_portfolio_id": "bridge_001", "asset_class": "bridging"}],
                   asset_class="equity_release")
    got = {e["source_portfolio_id"]: e["asset_class"]
           for e in yaml.safe_load(target.read_text())["portfolios"]}
    assert got == {"erm_001": "equity_release", "bridge_001": "bridge"}


def test_publishing_preserves_keys_the_writer_does_not_own(tmp_path):
    """A supplied runoff curve must survive a re-run of onboarding."""
    target = tmp_path / "registry.yaml"
    target.write_text(yaml.safe_dump({"portfolios": [
        {"source_portfolio_id": "acquired_001", "originates": False,
         "runoff_profile_id": "client_curve_2026",
         "runoff_curve": [1.0, 0.98, 0.96]}]}))
    writer.publish(target, [{"source_portfolio_id": "acquired_001"}],
                   asset_class="equity_release")
    entry = yaml.safe_load(target.read_text())["portfolios"][0]
    assert entry["asset_class"] == "equity_release"
    assert entry["runoff_profile_id"] == "client_curve_2026"
    assert entry["runoff_curve"] == [1.0, 0.98, 0.96]
    assert entry["originates"] is False


def test_a_portfolio_onboarding_could_not_classify_is_written_without_one(tmp_path):
    """No invention. An unclassified book carries no asset class, and MI reads
    the absence conservatively."""
    target = tmp_path / "registry.yaml"
    writer.publish(target, [{"source_portfolio_id": "unknown_001"}])
    assert yaml.safe_load(target.read_text())["portfolios"] == [
        {"source_portfolio_id": "unknown_001"}]


# =========================================================================== #
# It lands on the governed model (steps 1-2)
# =========================================================================== #
def _records():
    return [{FIELD_PORTFOLIO_ID: "direct_001", FIELD_PORTFOLIO_TYPE: "direct"},
            {FIELD_PORTFOLIO_ID: "acquired_001", FIELD_PORTFOLIO_TYPE: "acquired"}]


def test_the_portfolio_record_carries_the_governed_asset_class():
    registry = build_registry(_records(), metadata={
        "direct_001": {"asset_class": "equity_release"},
        "acquired_001": {"asset_class": "equity_release"}})
    assert {p.portfolio_id: p.asset_class for p in registry.portfolios} == {
        "direct_001": "equity_release", "acquired_001": "equity_release"}
    assert registry.portfolios[0].to_dict()["asset_class"] == "equity_release"


def test_asset_class_is_never_inferred_from_the_id_or_the_type():
    """Governed metadata only. A guessed class would admit asset-specific
    metrics to a book that never declared one."""
    registry = build_registry(_records())
    assert all(p.asset_class is None for p in registry.portfolios)
    assert registry.asset_classes() == ()


def test_a_partially_declared_scope_reads_as_unknown():
    """One undeclared book makes the SCOPE unknown — deliberately not the same
    as "they differ", and the conservative reading downstream."""
    registry = build_registry(_records(), metadata={
        "direct_001": {"asset_class": "equity_release"}})
    assert registry.asset_classes() == ()
    assert registry.asset_classes(["direct_001"]) == ("equity_release",)


def test_the_resolved_scope_carries_the_classes_of_its_members():
    registry = build_registry(_records(), metadata={
        "direct_001": {"asset_class": "equity_release"},
        "acquired_001": {"asset_class": "equity_release"}})
    for context in (None, "direct", "acquired", "direct_001"):
        assert resolve_scope(registry, context).asset_classes == ("equity_release",)


def test_a_mixed_asset_client_reports_both_classes():
    registry = build_registry(
        [{FIELD_PORTFOLIO_ID: "erm_001", FIELD_PORTFOLIO_TYPE: "direct"},
         {FIELD_PORTFOLIO_ID: "bridge_001", FIELD_PORTFOLIO_TYPE: "direct"}],
        metadata={"erm_001": {"asset_class": "equity_release"},
                  "bridge_001": {"asset_class": "bridge"}})
    assert resolve_scope(registry, None).asset_classes == ("bridge", "equity_release")
    assert resolve_scope(registry, "bridge_001").asset_classes == ("bridge",)


def test_the_overlay_normalises_the_vocabulary_on_the_way_in(tmp_path, monkeypatch):
    """Onboarding's dialect never reaches the Business Semantics Registry."""
    from mi_agent import portfolio_metadata

    registry_file = tmp_path / "portfolio_registry.yaml"
    registry_file.write_text(yaml.safe_dump({"portfolios": [
        {"source_portfolio_id": "direct_001",
         "asset_class": "equity_release_mortgage"}]}))
    monkeypatch.setenv(portfolio_metadata.ENV_REGISTRY_PATH, str(registry_file))
    overlay = portfolio_metadata.load_portfolio_metadata("any_client")
    assert overlay["direct_001"]["asset_class"] == "equity_release"


# =========================================================================== #
# Both MI consumers read the SAME source (step 4)
# =========================================================================== #
def test_the_comparison_workflow_prefers_governed_metadata_over_the_tape():
    """The defect: this consumer read an asset_class COLUMN, which most tapes do
    not carry, while period-change read a scope argument no caller supplied."""
    import pandas as pd

    from mi_workflows import portfolio_risk_comparison as prc

    frame = pd.DataFrame({"current_outstanding_balance": [1.0, 2.0]})
    scope = resolve_scope(build_registry(_records(), metadata={
        "direct_001": {"asset_class": "equity_release"},
        "acquired_001": {"asset_class": "equity_release"}}), "direct")

    assert prc._asset_classes(frame) == ()               # no column, as before
    assert prc._asset_classes(frame, scope) == ("equity_release",)


def test_the_tape_column_still_works_as_a_fallback():
    import pandas as pd

    from mi_workflows import portfolio_risk_comparison as prc

    frame = pd.DataFrame({prc.ASSET_CLASS_FIELD: ["equity_release"] * 2})
    assert prc._asset_classes(frame) == ("equity_release",)
    # …and an undeclared scope does not erase it.
    assert prc._asset_classes(frame, resolve_scope(
        build_registry(_records()), None)) == ("equity_release",)


def test_period_change_reads_the_asset_class_from_the_registry():
    """``resolve_asset_classes`` previously had no production caller, so every
    analysis ran with an unknown asset class."""
    from mi_agent_api.period_change_route import scope_ref_from_lens

    registry = build_registry(_records(), metadata={
        "direct_001": {"asset_class": "equity_release"},
        "acquired_001": {"asset_class": "equity_release"}})

    class _Lens:
        filters = {"source_portfolio_id": ["direct_001", "acquired_001"]}
        label = "Total"

    assert scope_ref_from_lens(_Lens()).asset_classes == ()
    assert scope_ref_from_lens(_Lens(), registry=registry).asset_classes == (
        "equity_release",)


def test_an_explicit_caller_argument_still_wins():
    from mi_agent_api.period_change_route import scope_ref_from_lens

    registry = build_registry(_records(), metadata={
        "direct_001": {"asset_class": "equity_release"},
        "acquired_001": {"asset_class": "equity_release"}})

    class _Lens:
        filters = {"source_portfolio_id": ["direct_001"]}
        label = "Direct"

    assert scope_ref_from_lens(_Lens(), asset_classes=("sme",),
                               registry=registry).asset_classes == ("sme",)


# =========================================================================== #
# End to end on the demonstration platform
# =========================================================================== #
@pytest.fixture(scope="module")
def governed_env():
    """The demo client, whose onboarding stage publishes the registry."""
    import logging
    import os

    from demo_platform import config as cfg

    root = cfg.local_blob_root() / "processed" / "platform" / cfg.CLIENT_ID
    if not root.exists():
        pytest.skip("the demonstration platform has not been built — run "
                    "`python -m demo_platform.run_demo --generate --orchestrate`")
    if not cfg.portfolio_registry_path().exists():
        pytest.skip("onboarding has not published the governed portfolio "
                    "registry — run `python -m demo_platform.run_demo --all`")
    before = dict(os.environ)
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_DATA_CACHE_TTL"] = "0"
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    logging.disable(logging.WARNING)
    from mi_agent_api import data_source

    data_source.reset_cache()
    yield cfg
    logging.disable(logging.NOTSET)
    os.environ.clear()
    os.environ.update(before)
    data_source.reset_cache()


def test_the_asset_class_reaches_the_mi_agent_end_to_end(governed_env):
    """The whole chain, on the real fixture."""
    from mi_agent.portfolio_metadata import load_portfolio_metadata
    from mi_agent.portfolio_scope import registry_for_frame
    from mi_agent_api.data_source import get_dataframe

    overlay = load_portfolio_metadata(governed_env.CLIENT_ID)
    assert overlay, "onboarding published no governed portfolio metadata"
    assert all(e.get("asset_class") == "equity_release" for e in overlay.values())

    registry = registry_for_frame(get_dataframe(), client_id=governed_env.CLIENT_ID)
    assert registry.asset_classes() == ("equity_release",)
    assert all(p.asset_class == "equity_release" for p in registry.portfolios)


def test_b25_compares_borrower_age_and_reconciles(governed_env):
    """The question this handoff was blocking. Truth computed here with pandas."""
    from trakt_core.context import ExecutionContext
    from mi_agent_api.data_source import get_dataframe
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    envelope = execute_governed_mi_query(
        MiQueryRequest(question="How does the direct book compare with the "
                                "acquired book on borrower age?"),
        ExecutionContext.for_internal(governed_env.CLIENT_ID)).result
    assert envelope["ok"] is True, envelope.get("error")

    comparisons = envelope["workflow"]["metric_comparisons"]
    assert [c["field"] for c in comparisons] == ["youngest_borrower_age"]
    assert comparisons[0]["aggregation"] == "average"

    book = get_dataframe()
    cohort, age = "source_portfolio_type", "youngest_borrower_age"
    expected_direct = float(book[book[cohort] == "direct"][age].mean())
    expected_acquired = float(book[book[cohort] == "acquired"][age].mean())
    assert comparisons[0]["portfolio_a"]["value"] == pytest.approx(
        expected_direct, rel=1e-9)
    assert comparisons[0]["portfolio_b"]["value"] == pytest.approx(
        expected_acquired, rel=1e-9)

    receipt = (envelope.get("executionSummary") or {}).get("receipt") or ""
    assert "Youngest Borrower Age" in receipt
    assert "Direct vs Acquired" in receipt


def test_equity_release_semantics_are_now_admitted_to_period_change(governed_env):
    """"Equity Release MI = the common core PLUS this asset's own semantics."
    With the book's class known, ER-specific governed fields become eligible;
    SME and CRE fields stay excluded, because this book is not those."""
    import os

    from mi_agent_api.period_change_route import analyse_period_change

    result = analyse_period_change(
        client_id=governed_env.CLIENT_ID,
        output_root=os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"],
        question="what changed since last month?").to_dict()
    selection = result["field_selection"]
    assert "youngest_borrower_age" in selection["selected_measures"], (
        "an equity-release measure is still excluded from an equity-release book")

    still_excluded = {c["canonical_field"] for c in selection["excluded_candidates"]
                      if "asset" in str(c.get("reason", "")).lower()}
    assert still_excluded, "no asset-specific field is being excluded at all"
    assert "youngest_borrower_age" not in still_excluded
