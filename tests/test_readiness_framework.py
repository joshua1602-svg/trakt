#!/usr/bin/env python3
"""The readiness framework, and the separation it exists to protect.

Four things must never blur, and this suite is mostly about the boundaries
between them:

    FACT       Trakt measured 28.0% of balance in London.
    EXTERNAL   The warehouse permits 35%. It passes.
    SCREENING  Trakt's internal threshold is 25%. It flags.
    JUDGEMENT  Whether that matters is the reviewer's call.

The sharpest test here is ``test_one_fact_three_rulebooks_three_verdicts``: the
readiness portfolio is built so a single measured number produces a *pass*, a
*breach* and a *flag* at once. A book that failed everything would prove nothing
about whether the rulebooks are genuinely independent of the calculation.

Every expectation is checked against ``tests/readiness_portfolio.TRUTH``, whose
values are hand arithmetic written down rather than derived from the code under
test.
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
from trakt_core.envelope import STATUS_BLOCKED, STATUS_SUCCESS
from trakt_core.errors import ErrorCode
from trakt_core.readiness import (
    EXTERNAL_AUTHORITIES,
    JUDGEMENT_ONLY,
    METRIC_TYPES,
    OUTCOME_BREACH,
    OUTCOME_CLEAR,
    OUTCOME_FLAG,
    OUTCOME_PASS,
    STATUSES,
    load_framework,
    load_rule_packs,
)
from trakt_tools.execution import ToolDependencies, execute_governed_tool

from tests.planted_portfolio import planted_lineage_index
from tests.readiness_portfolio import (
    GEOGRAPHIC_VERDICTS,
    SNAPSHOT_ID,
    TOP_10_VERDICTS,
    TRUTH,
    readiness_frame,
)
from tests.test_agent_governed_execution import _Datasets, _Descriptor
from tests.test_agent_loan_retrieval import (
    PORTFOLIO_A,
    PORTFOLIO_B,
    SPV_I,
    TENANT,
    _catalogue,
    _context,
    _CountingResolver,
)

BOTH = (SCOPE_LOAN_READ, SCOPE_RISK_READ)


@pytest.fixture(autouse=True)
def _clean_cache():
    config_cache.reset()
    yield
    config_cache.reset()


def _deps(resolver=None) -> ToolDependencies:
    return ToolDependencies(
        datasets=_Datasets(_Descriptor(snapshot_id=SNAPSHOT_ID)),
        runtime_mode="test", catalogue=_catalogue(), output_root="/unused",
        loan_frame_resolver=resolver or _CountingResolver(readiness_frame()),
        lineage_index=planted_lineage_index(TENANT, SNAPSHOT_ID))


def _run(tool: str, args: Dict[str, Any], *, deps=None, context=None):
    return execute_governed_tool(tool, args,
                                 context or _context(capabilities=BOTH),
                                 dependencies=deps or _deps())


def _ok(tool: str, args: Dict[str, Any], **kwargs):
    result = _run(tool, args, **kwargs)
    assert result.status == STATUS_SUCCESS, result.error and result.error.to_dict()
    return result.result


# =========================================================================== #
# The fixture describes itself correctly
# =========================================================================== #
def test_the_readiness_portfolio_matches_its_stated_truth():
    df = readiness_frame()
    assert len(df) == TRUTH["loan_count"]
    assert df["current_principal_balance"].sum() == pytest.approx(
        TRUTH["total_balance"])
    for region, share in TRUTH["region_shares"].items():
        balance = df.loc[df["geographic_region_collateral"] == region,
                         "current_principal_balance"].sum()
        assert balance / TRUTH["total_balance"] * 100 == pytest.approx(share)
    largest = max(TRUTH["region_shares"].items(), key=lambda kv: kv[1])
    assert largest[0] == TRUTH["largest_region"]
    assert largest[1] == pytest.approx(TRUTH["largest_region_share"])


def test_the_portfolio_is_built_to_separate_the_rulebooks():
    """A book that failed everything would prove nothing about whether the
    rulebooks are independent of the calculation."""
    for pack, (outcome, threshold) in GEOGRAPHIC_VERDICTS.items():
        share = TRUTH["largest_region_share"]
        if outcome in (OUTCOME_PASS,):
            assert share <= threshold, pack
        else:
            assert share > threshold, pack
    assert {o for o, _t in GEOGRAPHIC_VERDICTS.values()} == {
        OUTCOME_PASS, OUTCOME_BREACH, OUTCOME_FLAG}


# =========================================================================== #
# The framework registry
# =========================================================================== #
def test_the_shipped_framework_loads_and_is_internally_consistent():
    framework = load_framework()
    assert framework.configured
    assert framework.version >= 1
    ids = [m.id for m in framework.metrics]
    assert len(ids) == len(set(ids)), "duplicate metric ids"
    for metric in framework.metrics:
        assert metric.type in METRIC_TYPES, metric.id
        assert metric.status in STATUSES, metric.id
        assert metric.category in framework.categories, metric.id
        assert metric.agent_investigation_guidance, (
            f"{metric.id} has no investigation guidance; an agent given a "
            "metric and no guidance will invent an interpretation")


def test_a_metric_served_by_a_registered_tool_is_not_still_marked_a_gap():
    """Status drift, which this caught in Sprint 2.5: five metrics had their
    tools built, registered and tested while their framework status still said
    SMALL_GAP. The published framework then understated Trakt's own coverage —
    an agent reading it would have skipped capabilities that existed.

    A metric may legitimately name a tool it is not yet READY for (the tool
    exists but the metric needs more), so this asserts the narrower thing: if
    the named tool is REGISTERED, the status must not claim the calculation is
    missing.
    """
    from trakt_tools.registry import get

    framework = load_framework()
    for metric in framework.metrics:
        if metric.fact_tool and get(metric.fact_tool) is not None:
            assert metric.status not in ("SMALL_GAP", "DATA_GAP"), (
                f"{metric.id} is served by registered tool "
                f"{metric.fact_tool!r} but is still marked {metric.status}")


def test_a_metric_claiming_a_screening_rule_has_one_in_the_pack():
    """A dangling screening_rule would publish a threshold that does not exist."""
    framework = load_framework()
    packs = {p.pack_id: p for p in load_rule_packs()}
    screening = packs.get("TRAKT_SCREENING")
    assert screening is not None
    declared = {r.rule_id for r in screening.rules}
    for metric in framework.metrics:
        if metric.screening_rule:
            assert metric.screening_rule in declared, (
                f"{metric.id} names screening rule {metric.screening_rule!r}, "
                "which the screening pack does not define")


def test_a_screening_rule_points_back_at_a_real_framework_metric():
    framework = load_framework()
    for pack in load_rule_packs():
        for rule in pack.rules:
            assert framework.get(rule.framework_metric) is not None, (
                f"{pack.pack_id}/{rule.rule_id} references unknown framework "
                f"metric {rule.framework_metric!r}")


def test_measure_only_metrics_carry_no_screening_threshold():
    """MEASURE_ONLY means Trakt forms no view. A threshold would be a view."""
    framework = load_framework()
    for metric in framework.metrics:
        if metric.type == "MEASURE_ONLY":
            assert metric.screening_rule is None, metric.id


def test_every_screening_rule_states_why_its_threshold_is_what_it_is():
    """An unexplained threshold is indistinguishable from an invented one, and
    an agent quoting it cannot say where it came from."""
    for pack in load_rule_packs():
        if pack.is_external:
            continue
        for rule in pack.rules:
            assert len(rule.rationale) > 40, (
                f"{rule.rule_id} has no substantive rationale")


def test_coverage_is_not_inflated_by_counting_judgement_only_metrics():
    """Scoring JUDGEMENT_ONLY as covered would claim credit for something no
    deterministic calculation should exist for; scoring it as a gap would imply
    work that should never be done. It is excluded from the denominator."""
    framework = load_framework()
    coverage = framework.coverage()
    judgement = [m.id for m in framework.metrics if m.status == JUDGEMENT_ONLY]
    assert coverage["excluded_judgement_only"] == judgement
    assert coverage["scored_metrics"] == len(framework.metrics) - len(judgement)
    assert 0 <= coverage["coverage_pct"] <= 100
    # A metric counts as deterministic only when READY *and* served by a tool.
    for metric in framework.metrics:
        if metric.deterministic:
            assert metric.status == "READY" and metric.fact_tool, metric.id


# =========================================================================== #
# Rule packs: authority, and the vocabularies that keep them apart
# =========================================================================== #
def test_internal_and_external_packs_use_different_outcome_words():
    """A screening flag and a contractual breach read alike in a summary and
    mean entirely different things. Different vocabularies make the confusion
    structurally impossible rather than merely discouraged."""
    packs = {p.pack_id: p for p in load_rule_packs()}
    internal = packs["TRAKT_SCREENING"]
    assert internal.outcome_for(True) == OUTCOME_FLAG
    assert internal.outcome_for(False) == OUTCOME_CLEAR
    assert not internal.is_external

    external = packs["EXAMPLE_WAREHOUSE"]
    assert external.outcome_for(True) == OUTCOME_BREACH
    assert external.outcome_for(False) == OUTCOME_PASS
    assert external.is_external
    assert external.authority in EXTERNAL_AUTHORITIES


def test_the_trakt_pack_labels_itself_as_internal_everywhere():
    packs = {p.pack_id: p for p in load_rule_packs()}
    screening = packs["TRAKT_SCREENING"]
    assert screening.authority == "trakt_internal"
    label = screening.authority_label.lower()
    assert "not a transaction requirement" in label
    assert "not a regulatory requirement" in label


def test_the_example_packs_declare_themselves_synthetic():
    """They exist to demonstrate the architecture. Nothing should ever mistake
    them for a real agreement or a rating methodology."""
    for pack_id in ("EXAMPLE_WAREHOUSE", "EXAMPLE_PROPOSED_SECURITISATION"):
        pack = next(p for p in load_rule_packs() if p.pack_id == pack_id)
        assert "SYNTHETIC" in pack.pack_name.upper()
        assert "synthetic" in pack.authority_label.lower()
        assert "SYNTHETIC" in pack.source_document.upper()


def test_no_pack_claims_to_be_a_rating_agency_methodology():
    """Trakt does not model agency criteria and does not predict ratings."""
    for pack in load_rule_packs():
        blob = f"{pack.pack_name} {pack.description} {pack.authority_label}".lower()
        for forbidden in ("moody", "s&p", "standard & poor", "fitch", "dbrs"):
            assert forbidden not in blob, f"{pack.pack_id} references {forbidden}"


# =========================================================================== #
# One fact, many rulebooks — the architectural claim
# =========================================================================== #
def test_one_fact_three_rulebooks_three_verdicts():
    """The whole point of Sprint 2.5, in one assertion.

    28.0% of balance in one region. The warehouse permits 35% and PASSES. The
    proposed criteria permit 27% and BREACH. Trakt's screen is 25% and FLAGS.
    One measurement, three answers, no recalculation.
    """
    payload = _ok("evaluate_rule_packs", {"resource": PORTFOLIO_A.key})
    geographic = {r["pack"]: r for r in payload["results"]
                  if r["framework_metric"] == "CONC_GEOGRAPHIC"}
    assert set(geographic) == set(GEOGRAPHIC_VERDICTS)

    for pack, (expected_outcome, expected_threshold) in GEOGRAPHIC_VERDICTS.items():
        result = geographic[pack]
        assert result["value"] == pytest.approx(TRUTH["largest_region_share"]), pack
        assert result["threshold"] == pytest.approx(expected_threshold), pack
        assert result["outcome"] == expected_outcome, pack

    # And every one of them read the SAME number.
    assert len({r["value"] for r in geographic.values()}) == 1


def test_the_same_split_holds_for_granularity():
    payload = _ok("evaluate_rule_packs", {"resource": PORTFOLIO_A.key})
    top_n = {r["pack"]: r for r in payload["results"]
             if r["framework_metric"] == "CONC_TOP_N_LOANS"}
    for pack, (expected_outcome, threshold) in TOP_10_VERDICTS.items():
        assert top_n[pack]["value"] == pytest.approx(TRUTH["top_10_share"])
        assert top_n[pack]["threshold"] == pytest.approx(threshold)
        assert top_n[pack]["outcome"] == expected_outcome, pack


def test_each_fact_is_computed_once_however_many_packs_read_it():
    """Not only cheaper. If each pack measured its own London share, a parameter
    drift could let the warehouse pass and the criteria fail for a reason no
    reviewer could explain. Sharing makes that impossible."""
    payload = _ok("evaluate_rule_packs", {"resource": PORTFOLIO_A.key})
    assert payload["facts_computed"] < payload["rules_evaluated"]

    # Three packs reference the geographic metric with identical parameters, so
    # it is computed once and all three read the same number.
    geographic = [r for r in payload["results"]
                  if r["framework_metric"] == "CONC_GEOGRAPHIC"]
    assert len(geographic) == 3
    assert len({r["value"] for r in geographic}) == 1

    # Facts are keyed by (metric, parameters), so the count can never exceed the
    # number of distinct such pairs across every pack.
    distinct = {repr(sorted((r.get("fact", {}).get("detail", {})
                             .get("metric_id") or r["framework_metric"],
                             repr(sorted(r["parameters"].items())))))
                for r in payload["results"]}
    assert payload["facts_computed"] <= len(distinct) + 1


def test_the_whole_evaluation_costs_one_dataset_access():
    resolver = _CountingResolver(readiness_frame())
    _ok("evaluate_rule_packs", {"resource": PORTFOLIO_A.key},
        deps=_deps(resolver))
    assert resolver.calls == 1


def test_every_result_carries_the_authority_it_came_from():
    """The field an agent must read before reporting anything."""
    payload = _ok("evaluate_rule_packs", {"resource": PORTFOLIO_A.key})
    for result in payload["results"]:
        assert result["authority"], result["rule_id"]
        assert "is_external_criterion" in result
        assert result["authority_label"], result["rule_id"]
        if result["is_external_criterion"]:
            assert result["outcome"] in (OUTCOME_PASS, OUTCOME_BREACH,
                                         "unavailable")
        else:
            assert result["outcome"] in (OUTCOME_FLAG, OUTCOME_CLEAR,
                                         "unavailable")


def test_flags_and_breaches_are_counted_separately_and_never_summed():
    payload = _ok("evaluate_rule_packs", {"resource": PORTFOLIO_A.key})
    summary = payload["summary"]
    assert "flags" in summary["trakt_screening"]
    assert "breaches" in summary["external_criteria"]
    assert "flags" not in summary["external_criteria"]
    assert "breaches" not in summary["trakt_screening"]
    assert "must not be added together" in summary["note"]


def test_no_packs_configured_is_reported_as_unevaluated_not_as_a_pass():
    """An absent rulebook has not been satisfied."""
    import tempfile

    with tempfile.TemporaryDirectory() as empty:
        import os
        os.environ["TRAKT_RULE_PACK_DIR"] = empty
        try:
            config_cache.reset()
            payload = _ok("evaluate_rule_packs", {"resource": PORTFOLIO_A.key})
        finally:
            os.environ.pop("TRAKT_RULE_PACK_DIR", None)
            config_cache.reset()
    assert payload["results"] == []
    assert any("not a pass" in w for w in payload["warnings"])


def test_selecting_packs_narrows_the_evaluation():
    payload = _ok("evaluate_rule_packs", {"resource": PORTFOLIO_A.key,
                                          "packs": ["TRAKT_SCREENING"]})
    assert {p["pack"] for p in payload["packs"]} == {"TRAKT_SCREENING@v1"}
    assert payload["summary"]["external_criteria"]["evaluated"] == 0


# =========================================================================== #
# The facts themselves
# =========================================================================== #
def test_the_framework_tool_publishes_coverage_and_guidance():
    payload = _ok("readiness_framework", {"resource": PORTFOLIO_A.key})
    assert payload["framework"].startswith("SECURITISATION_READINESS@v")
    assert payload["authority"] == "trakt_internal"
    assert payload["coverage"]["coverage_pct"] is not None
    assert len(payload["metrics"]) == len(load_framework().metrics)
    for metric in payload["metrics"]:
        assert metric["agent_investigation_guidance"]


def test_readiness_metrics_measures_through_the_governed_library():
    payload = _ok("readiness_metrics", {
        "resource": PORTFOLIO_A.key,
        "requests": [
            {"metric_id": "geo_largest_region_share", "label": "geo"},
            {"metric_id": "top_n_loans_share", "parameters": {"n": 10},
             "label": "top10"},
            {"metric_id": "perf_arrears_share", "parameters": {"min_days": 90},
             "label": "dpd90"},
        ]})
    by_label = {m["label"]: m for m in payload["measurements"]}
    assert by_label["geo"]["value"] == pytest.approx(
        TRUTH["largest_region_share"])
    assert by_label["top10"]["value"] == pytest.approx(TRUTH["top_10_share"])
    assert by_label["dpd90"]["value"] == pytest.approx(
        TRUTH["arrears_90_plus_share"])
    for measurement in payload["measurements"]:
        assert measurement["calculation_source"].startswith(
            "concentration_tests.metrics.")


def test_a_library_metric_with_no_evaluator_is_unavailable_not_estimated():
    """perf_cumulative_loss_share is declared and not implemented. Reporting an
    estimate would be Trakt inventing a number."""
    payload = _ok("readiness_metrics", {
        "resource": PORTFOLIO_A.key,
        "requests": [{"metric_id": "perf_cumulative_loss_share"}]})
    measurement = payload["measurements"][0]
    assert measurement["available"] is False
    assert "no registered evaluator" in measurement["reason"]
    assert payload["unresolved"]


def test_an_unknown_metric_is_reported_rather_than_silently_dropped():
    payload = _ok("readiness_metrics", {
        "resource": PORTFOLIO_A.key,
        "requests": [{"metric_id": "not_a_real_metric"}]})
    assert payload["measurements"][0]["available"] is False
    assert "not in the governed metric library" in (
        payload["measurements"][0]["reason"])


# =========================================================================== #
# Valuation evidence
# =========================================================================== #
def test_the_valuation_profile_finds_the_planted_stale_and_missing_cohorts():
    payload = _ok("valuation_age_profile", {"resource": PORTFOLIO_A.key})
    assert payload["available"] is True
    assert payload["stale_balance_pct"] == pytest.approx(
        TRUTH["stale_valuation_share"])
    assert payload["missing_balance_pct"] == pytest.approx(
        TRUTH["missing_valuation_share"])
    assert payload["as_of"] == "2026-07-31"


def test_the_valuation_profile_is_balance_weighted_not_counted():
    """A stale valuation on a large loan matters more than one on a small
    loan, and a count would hide that."""
    payload = _ok("valuation_age_profile", {"resource": PORTFOLIO_A.key})
    stale_loans = payload["stale_loan_count"]
    count_share = stale_loans / TRUTH["loan_count"] * 100
    assert payload["stale_balance_pct"] != pytest.approx(count_share, abs=0.01)


def test_the_stale_threshold_matches_the_governed_selection_policy():
    """The screening rule and the LTV derivation must agree about what stale
    means, or a loan can be stale in one place and current in the other."""
    from trakt_core.valuation import load_valuation_policies

    policy = load_valuation_policies().policy("CURRENT_LTV_SELECTION")
    payload = _ok("valuation_age_profile", {"resource": PORTFOLIO_A.key})
    assert payload["stale_after_months"] == policy.max_age_months["full"]


def test_the_profile_never_measures_age_against_the_wall_clock():
    """A profile that moved on its own could not be reproduced."""
    payload = _ok("valuation_age_profile", {"resource": PORTFOLIO_A.key})
    assert payload["as_of"] == "2026-07-31"      # the tape's reporting date
    import datetime
    assert payload["as_of"] != datetime.date.today().isoformat()


def test_missing_valuations_are_excluded_from_the_age_distribution_but_reported():
    """They are silently absent from every LTV metric, so the profile has to
    surface them explicitly or the book looks better than the data supports."""
    payload = _ok("valuation_age_profile", {"resource": PORTFOLIO_A.key})
    distributed = sum(b["loan_count"] for b in payload["age_distribution"])
    assert distributed == payload["population"]["loans_with_usable_valuation"]
    assert payload["missing_loan_count"] > 0
    assert distributed + payload["missing_loan_count"] == TRUTH["loan_count"]


# =========================================================================== #
# Regulatory readiness
# =========================================================================== #
def test_regulatory_readiness_separates_blockers_from_degraded_fields():
    """The regime has three tiers and conflating them turns a blocker into a
    footnote or a footnote into a blocker."""
    payload = _ok("regulatory_readiness", {"resource": PORTFOLIO_A.key})
    summary = payload["summary"]
    assert summary["regime_fields"] > 0
    assert summary["blocking_fields"] > 0
    assert (summary["blocking_fields"] + summary["nd1_4_permitted_fields"]
            + summary["nd5_permitted_fields"]) == summary["regime_fields"]
    for gap in payload["blocking_gaps"]:
        assert gap["nd1_4_allowed"] is False and gap["nd5_allowed"] is False


def test_a_blocking_gap_blocks_the_submission():
    payload = _ok("regulatory_readiness", {"resource": PORTFOLIO_A.key})
    if payload["blocking_gaps"]:
        assert payload["summary"]["submission_blocked"] is True
        assert any("blocker, not a quality issue" in w
                   for w in payload["warnings"])


def test_regulatory_readiness_refuses_to_imply_portfolio_quality():
    payload = _ok("regulatory_readiness", {"resource": PORTFOLIO_A.key})
    limitations = " ".join(payload["limitations"]).lower()
    assert "not evidence that the portfolio is sound" in limitations
    assert "different questions" in limitations


def test_field_detail_is_opt_in():
    """107 rows on every call would drown the summary that matters."""
    without = _ok("regulatory_readiness", {"resource": PORTFOLIO_A.key})
    assert without["fields"] == []
    with_detail = _ok("regulatory_readiness", {"resource": PORTFOLIO_A.key,
                                               "include_field_detail": True})
    assert len(with_detail["fields"]) == with_detail["summary"]["regime_fields"]


# =========================================================================== #
# Governance holds across the new surface
# =========================================================================== #
READINESS_TOOLS = (
    ("readiness_framework", {}),
    ("readiness_metrics", {"requests": [{"metric_id": "geo_largest_region_share"}]}),
    ("valuation_age_profile", {}),
    ("regulatory_readiness", {}),
    ("evaluate_rule_packs", {}),
)


@pytest.mark.parametrize("tool,extra", READINESS_TOOLS)
def test_an_ungranted_resource_is_refused(tool, extra):
    result = _run(tool, {"resource": PORTFOLIO_B.key, **extra})
    assert result.status == STATUS_BLOCKED, tool
    assert result.error_code == ErrorCode.RESOURCE_NOT_AUTHORISED, tool


@pytest.mark.parametrize("tool,extra", [t for t in READINESS_TOOLS
                                        if t[0] != "readiness_framework"])
def test_an_spv_scoped_resource_is_refused_rather_than_widened(tool, extra):
    result = _run(tool, {"resource": SPV_I.key, **extra},
                  context=_context(SPV_I, capabilities=BOTH))
    assert result.status == STATUS_BLOCKED, tool
    assert result.error_code == ErrorCode.RESOURCE_NOT_PARTITIONABLE, tool


@pytest.mark.parametrize("tool,extra", READINESS_TOOLS)
def test_every_readiness_tool_publishes_what_it_cost(tool, extra):
    payload = _ok(tool, {"resource": PORTFOLIO_A.key, **extra})
    telemetry = payload["_telemetry"]
    assert telemetry["tool"] == tool
    assert telemetry["duration_ms"] is not None


@pytest.mark.parametrize("tool,extra", READINESS_TOOLS)
def test_every_readiness_tool_states_its_scope(tool, extra):
    payload = _ok(tool, {"resource": PORTFOLIO_A.key, **extra})
    assert payload["scope"]["resource"] == PORTFOLIO_A.key


def test_the_readiness_tools_need_only_the_aggregate_capability():
    """None of them exposes an individual obligation, so none requires
    loan:read — the stronger grant stays reserved for tools that name loans."""
    from trakt_tools.registry import get

    for tool, _extra in READINESS_TOOLS:
        assert get(tool).required_capability == SCOPE_RISK_READ, tool


def test_no_readiness_tool_holds_its_own_calculation():
    """Structural. A securitisation-specific calculation would be the second
    engine this sprint forbids."""
    import inspect

    from trakt_tools.handlers import readiness as module

    source = inspect.getsource(module)
    # Measurement must be delegated: these are the delegation points.
    assert "evaluate_metric" in source
    assert "analytics_lib.valuation_age" in source
    # And no arithmetic on portfolio values beyond assembling the response.
    for forbidden in ("df.groupby", ".sum() /", "np.average"):
        assert forbidden not in source, (
            f"trakt_tools.handlers.readiness computes with {forbidden!r}")
