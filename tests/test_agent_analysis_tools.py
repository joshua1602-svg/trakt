#!/usr/bin/env python3
"""The aggregate tools, and the reason they exist.

An agent given only ``get_loans`` answers "what is the weighted-average LTV of
the London exposure?" by retrieving the London loans and averaging them itself.
The answer is then the *agent's* — no governed definition, no provenance, no
reproducibility — and it costs a row per loan to get there.

So this suite holds two lines.

**The numbers are Trakt's.** Every aggregate is checked against
``tests/planted_portfolio``'s hand-stated truth, not against a re-run of the
same code. London is 55.625% of the book because that is what 1,780,000 over
3,200,000 is, and it is written down as such.

**The aggregates are cheaper, and visibly so.** ``rank_loans`` returns twenty
identifiers out of a scanned population and publishes both counts, so an agent
can see that ranking beat retrieving without being told.
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
from trakt_tools.execution import ToolDependencies, execute_governed_tool
from trakt_tools.handlers.analysis import MAX_GROUPS, MAX_RANK

from tests.planted_portfolio import (
    CONCENTRATED_BORROWER,
    CONCENTRATED_BORROWER_SHARE,
    REGION_TRUTH,
    SNAPSHOT_ID,
    TOTAL_PRINCIPAL,
    TRUTH,
    VALIDATION_EXCEPTION,
    planted_frame,
    planted_lineage_index,
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


def _deps(resolver=None, **extra) -> ToolDependencies:
    return ToolDependencies(
        datasets=_Datasets(_Descriptor(snapshot_id=SNAPSHOT_ID)),
        runtime_mode="test", catalogue=_catalogue(), output_root="/unused",
        loan_frame_resolver=resolver or _CountingResolver(planted_frame()),
        lineage_index=planted_lineage_index(TENANT), **extra)


def _run(tool: str, args: Dict[str, Any], *, deps=None, context=None):
    return execute_governed_tool(tool, args,
                                 context or _context(capabilities=BOTH),
                                 dependencies=deps or _deps())


def _ok(tool: str, args: Dict[str, Any], **kwargs):
    result = _run(tool, args, **kwargs)
    assert result.status == STATUS_SUCCESS, result.error and result.error.to_dict()
    return result.result


# =========================================================================== #
# portfolio_summary
# =========================================================================== #
def test_a_summary_reports_the_size_the_truth_table_states():
    payload = _ok("portfolio_summary", {"resource": PORTFOLIO_A.key})
    population = payload["population"]
    assert population["loan_count"] == len(TRUTH)
    assert population["total_balance"] == pytest.approx(TOTAL_PRINCIPAL)


def test_the_weighted_average_ltv_is_weighted_by_balance_not_by_loan():
    """The distinction that matters in credit: a 95% LTV on 380,000 moves the
    book average more than a 50% LTV on 120,000, and an unweighted mean hides
    exactly the exposure a buyer is pricing.

    Worked by hand from the truth table, over the ten loans that carry an LTV:
        sum(balance * ltv) / sum(balance of those loans)
    """
    payload = _ok("portfolio_summary", {"resource": PORTFOLIO_A.key})

    from tests.planted_portfolio import ROWS
    priced = [r for r in ROWS if r["current_loan_to_value"] is not None]
    numerator = sum(r["current_principal_balance"] * r["current_loan_to_value"]
                    for r in priced)
    denominator = sum(r["current_principal_balance"] for r in priced)
    expected = numerator / denominator

    weighted = payload["weighted_averages"]["current_loan_to_value"]
    simple = payload["weighted_averages"]["current_loan_to_value_simple_average"]
    assert weighted == pytest.approx(expected)
    assert simple != pytest.approx(weighted), (
        "the planted book was built so the two differ; if they match, the "
        "weighting is not being applied")


def test_the_summary_composition_matches_the_stated_region_shares():
    payload = _ok("portfolio_summary", {"resource": PORTFOLIO_A.key})
    regions = {row["geographic_region_collateral"]: row
               for row in payload["composition"]["geographic_region_collateral"]}
    for region, (balance, share) in REGION_TRUTH.items():
        assert regions[region]["balance_sum"] == pytest.approx(balance), region
        assert regions[region]["balance_share"] * 100 == pytest.approx(share), region


def test_the_summary_counts_the_planted_arrears_case():
    payload = _ok("portfolio_summary", {"resource": PORTFOLIO_A.key})
    # Exactly one planted loan is in arrears: LN-X-009, 4,200 at 64 days.
    assert payload["arrears"]["loans_in_arrears"] == 1
    assert payload["arrears"]["arrears_balance"] == pytest.approx(4_200.0)
    assert payload["arrears"]["loans_over_90_days"] == 0


def test_one_summary_costs_one_dataset_access():
    resolver = _CountingResolver(planted_frame())
    _ok("portfolio_summary", {"resource": PORTFOLIO_A.key}, deps=_deps(resolver))
    assert resolver.calls == 1


# =========================================================================== #
# stratify
# =========================================================================== #
def test_stratify_reproduces_the_stated_shares():
    payload = _ok("stratify", {"resource": PORTFOLIO_A.key,
                               "dimension": "geographic_region_collateral"})
    assert payload["group_count"] == len(REGION_TRUTH)
    for row in payload["groups"]:
        balance, share = REGION_TRUTH[row["geographic_region_collateral"]]
        assert row["balance_sum"] == pytest.approx(balance)
        assert row["balance_share"] * 100 == pytest.approx(share)


def test_stratify_orders_by_balance_descending():
    """Deterministic ordering, so an agent can read 'the largest' off the top."""
    payload = _ok("stratify", {"resource": PORTFOLIO_A.key,
                               "dimension": "geographic_region_collateral"})
    balances = [row["balance_sum"] for row in payload["groups"]]
    assert balances == sorted(balances, reverse=True)
    assert payload["groups"][0]["geographic_region_collateral"] == "London"


def test_stratify_reports_balance_weighted_metrics_per_category():
    payload = _ok("stratify", {"resource": PORTFOLIO_A.key,
                               "dimension": "geographic_region_collateral",
                               "weighted_metrics": ["current_loan_to_value"]})
    london = next(r for r in payload["groups"]
                  if r["geographic_region_collateral"] == "London")
    # London by hand: 380k@95 + 400k@80 + 350k@70 + 300k@60 + 350k@70
    numerator = (380_000 * 95 + 400_000 * 80 + 350_000 * 70
                 + 300_000 * 60 + 350_000 * 70)
    assert (london["current_loan_to_value_weighted_avg"]
            == pytest.approx(numerator / 1_780_000))


def test_a_truncated_breakdown_says_so():
    """A silently shortened league table reads as the whole one."""
    payload = _ok("stratify", {"resource": PORTFOLIO_A.key,
                               "dimension": "geographic_region_collateral",
                               "limit": 2})
    assert len(payload["groups"]) == 2
    assert payload["group_count"] == 4
    assert payload["truncated"] is True
    assert any("categories exist" in w for w in payload["warnings"])


def test_a_filter_narrows_the_population_and_the_shares_follow():
    payload = _ok("stratify", {"resource": PORTFOLIO_A.key,
                               "dimension": "borrower_identifier",
                               "filters": {"geographic_region_collateral": "London"}})
    assert payload["population"]["loan_count"] == 5
    assert payload["population"]["total_balance"] == pytest.approx(1_780_000.0)
    shares = sum(row["balance_share"] for row in payload["groups"])
    assert shares == pytest.approx(1.0)


def test_an_unknown_filter_column_is_refused_not_ignored():
    """Silently dropping a filter answers a NARROWER question with a WIDER
    population and labels it correctly — the most dangerous shape of wrong."""
    result = _run("stratify", {"resource": PORTFOLIO_A.key,
                               "dimension": "account_status",
                               "filters": {"not_a_column": "x"}})
    assert result.status != STATUS_SUCCESS
    assert result.error_code == ErrorCode.TOOL_INPUT_INVALID
    assert "not silently ignored" in result.error.message


def test_an_unknown_dimension_is_refused_with_the_available_ones():
    result = _run("stratify", {"resource": PORTFOLIO_A.key,
                               "dimension": "not_a_column"})
    assert result.error_code == ErrorCode.TOOL_INPUT_INVALID
    assert "available" in (result.error.details or {})


def test_the_group_ceiling_is_published_and_enforced():
    result = _run("stratify", {"resource": PORTFOLIO_A.key,
                               "dimension": "account_status",
                               "limit": MAX_GROUPS + 1})
    assert result.status != STATUS_SUCCESS      # the schema bounds it
    assert result.error_code == ErrorCode.TOOL_INPUT_INVALID


# =========================================================================== #
# concentration
# =========================================================================== #
def test_borrower_concentration_matches_the_stated_share():
    payload = _ok("concentration", {"resource": PORTFOLIO_A.key,
                                    "dimension": "borrower_identifier",
                                    "top_n": 1})
    assert payload["balance_concentration_pct"] == pytest.approx(
        CONCENTRATED_BORROWER_SHARE)
    assert payload["groups"][0]["borrower_identifier"] == CONCENTRATED_BORROWER
    assert payload["groups"][0]["loan_count"] == 4


def test_region_concentration_matches_the_stated_share():
    payload = _ok("concentration", {"resource": PORTFOLIO_A.key,
                                    "dimension": "geographic_region_collateral",
                                    "top_n": 1})
    assert payload["balance_concentration_pct"] == pytest.approx(
        REGION_TRUTH["London"][1])


def test_concentration_reports_a_share_and_refuses_to_call_it_a_breach():
    """Whether 55.6% breaches anything depends on an operator-approved
    threshold, and that lives in the covenant configuration with its approver
    and version. Answering it here would be a second, unapproved opinion."""
    payload = _ok("concentration", {"resource": PORTFOLIO_A.key,
                                    "dimension": "geographic_region_collateral"})
    assert payload["limit_evaluation"] is None
    assert any("evaluate_covenants" in w for w in payload["warnings"])
    assert "status" not in payload and "breach" not in payload


def test_concentration_is_reported_in_percentage_points():
    """analytics_lib returns fractions; every percentage Trakt publishes is in
    points, and a silent unit change between the two would be a 100x error."""
    payload = _ok("concentration", {"resource": PORTFOLIO_A.key,
                                    "dimension": "geographic_region_collateral",
                                    "top_n": 4})
    assert payload["balance_concentration_pct"] == pytest.approx(100.0)


# =========================================================================== #
# rank_loans — the bridge from population to investigation
# =========================================================================== #
def test_ranking_finds_the_planted_high_ltv_loan_first():
    payload = _ok("rank_loans", {"resource": PORTFOLIO_A.key,
                                 "metric": "current_loan_to_value",
                                 "direction": "highest", "limit": 3})
    ranked = [loan["loan_identifier"] for loan in payload["loans"]]
    assert ranked[0] == "LN-H-002"            # 95.00, the planted high-LTV case
    assert payload["loans"][0]["current_loan_to_value"] == pytest.approx(95.0)
    values = [loan["current_loan_to_value"] for loan in payload["loans"]]
    assert values == sorted(values, reverse=True)


def test_ranking_the_other_way_returns_the_other_end():
    payload = _ok("rank_loans", {"resource": PORTFOLIO_A.key,
                                 "metric": "current_loan_to_value",
                                 "direction": "lowest", "limit": 3})
    values = [loan["current_loan_to_value"] for loan in payload["loans"]]
    assert values == sorted(values)


def test_ranking_discloses_how_much_of_the_book_carries_the_metric():
    """LN-V-006 and LN-F-007 carry no LTV. Ranking a 10-of-12 populated field
    ranks the 10, and an agent that cannot see that reports 'the highest LTV in
    the book' about a subset."""
    payload = _ok("rank_loans", {"resource": PORTFOLIO_A.key,
                                 "metric": "current_loan_to_value"})
    coverage = payload["metric_coverage"]
    assert coverage["loans_with_metric"] == 10
    assert coverage["loans_without_metric"] == 2
    assert any("carry no" in w for w in payload["warnings"])
    returned = {loan["loan_identifier"] for loan in payload["loans"]}
    assert "LN-V-006" not in returned and "LN-F-007" not in returned


def test_ranking_returns_identifiers_rather_than_rows():
    """It is a shortlist, not a retrieval. The natural next step is ONE bounded
    get_loans call for the loans that matter."""
    payload = _ok("rank_loans", {"resource": PORTFOLIO_A.key,
                                 "metric": "current_loan_to_value", "limit": 3})
    for loan in payload["loans"]:
        assert set(loan) == {"loan_identifier", "current_loan_to_value"}


def test_ranking_can_carry_a_short_list_of_extra_fields():
    payload = _ok("rank_loans", {"resource": PORTFOLIO_A.key,
                                 "metric": "current_loan_to_value", "limit": 2,
                                 "include_fields": ["current_principal_balance",
                                                    "not_on_this_tape"]})
    for loan in payload["loans"]:
        assert "current_principal_balance" in loan
        assert "not_on_this_tape" not in loan


def test_ranking_is_bounded():
    result = _run("rank_loans", {"resource": PORTFOLIO_A.key,
                                 "metric": "current_loan_to_value",
                                 "limit": MAX_RANK + 1})
    assert result.error_code == ErrorCode.TOOL_INPUT_INVALID


def test_ranking_is_deterministic_on_ties():
    """Four planted loans share an LTV of 70.00 or 60.00. A ranking that
    reordered between runs would make a shortlist unreproducible."""
    args = {"resource": PORTFOLIO_A.key, "metric": "current_loan_to_value",
            "limit": 12}
    first = [l["loan_identifier"] for l in _ok("rank_loans", args)["loans"]]
    second = [l["loan_identifier"] for l in _ok("rank_loans", args)["loans"]]
    assert first == second


# =========================================================================== #
# data_completeness
# =========================================================================== #
def test_completeness_finds_the_planted_gaps():
    payload = _ok("data_completeness", {"resource": PORTFOLIO_A.key})
    by_field = {row["canonical_field"]: row for row in payload["fields"]}

    # Two loans carry no LTV.
    assert by_field["current_loan_to_value"]["populated"] == 10
    assert by_field["current_loan_to_value"]["missing"] == 2
    assert by_field["current_loan_to_value"]["completeness_pct"] == pytest.approx(
        10 / 12 * 100, abs=1e-4)

    # One loan carries no valuation at all.
    assert by_field["current_valuation_amount"]["missing"] == 1
    # Everything carries an identifier and a balance.
    assert by_field["loan_identifier"]["completeness_pct"] == pytest.approx(100.0)
    assert by_field["current_principal_balance"]["missing"] == 0


def test_completeness_lists_the_worst_fields_first():
    payload = _ok("data_completeness", {"resource": PORTFOLIO_A.key})
    percentages = [row["completeness_pct"] for row in payload["fields"]]
    assert percentages == sorted(percentages)


def test_completeness_can_report_only_the_gaps():
    payload = _ok("data_completeness", {"resource": PORTFOLIO_A.key,
                                        "max_completeness_pct": 99.9})
    for row in payload["fields"]:
        assert row["completeness_pct"] <= 99.9
    assert payload["fully_populated_count"] > 0     # counted over the whole tape


def test_a_field_absent_from_the_tape_is_distinguished_from_one_that_is_empty():
    payload = _ok("data_completeness", {
        "resource": PORTFOLIO_A.key,
        "fields": ["current_loan_to_value", "not_on_this_tape"]})
    reported = {row["canonical_field"] for row in payload["fields"]}
    assert reported == {"current_loan_to_value"}
    assert any("not_on_this_tape" in w for w in payload["warnings"])


# =========================================================================== #
# list_validation_exceptions
# =========================================================================== #
def test_the_planted_validation_exception_is_listed():
    payload = _ok("list_validation_exceptions", {"resource": PORTFOLIO_A.key})
    assert payload["lineage_available"] is True
    assert payload["grain"] == "canonical_field"
    exceptions = {e["canonical_field"]: e for e in payload["exceptions"]}
    failed = exceptions[VALIDATION_EXCEPTION["canonical_field"]]
    assert failed["validation_status"] == VALIDATION_EXCEPTION["status"]
    assert failed["rules_applied"] == VALIDATION_EXCEPTION["rules_applied"]
    assert failed["error_count"] == VALIDATION_EXCEPTION["error_count"]
    assert payload["total_error_count"] == VALIDATION_EXCEPTION["error_count"]


def test_passing_fields_are_not_listed_as_exceptions_by_default():
    payload = _ok("list_validation_exceptions", {"resource": PORTFOLIO_A.key})
    reported = {e["canonical_field"] for e in payload["exceptions"]}
    assert "current_principal_balance" not in reported
    assert payload["exception_count"] == 1


def test_a_caller_can_ask_for_every_status():
    payload = _ok("list_validation_exceptions", {"resource": PORTFOLIO_A.key,
                                                 "status": ["passed", "failed"]})
    reported = {e["canonical_field"] for e in payload["exceptions"]}
    assert "current_principal_balance" in reported
    assert "current_loan_to_value" in reported


def test_no_lineage_index_reports_unknown_and_not_clean():
    """'I don't know' and 'nothing wrong' are different answers, and a diligence
    agent must not read the first as the second."""
    deps = ToolDependencies(
        datasets=_Datasets(_Descriptor(snapshot_id=SNAPSHOT_ID)),
        runtime_mode="test", catalogue=_catalogue(), output_root="/unused",
        loan_frame_resolver=_CountingResolver(planted_frame()),
        lineage_index=None, lineage_index_path="/nonexistent/lineage_index.json")
    payload = _ok("list_validation_exceptions", {"resource": PORTFOLIO_A.key},
                  deps=deps)
    assert payload["lineage_available"] is False
    assert payload["exceptions"] == []
    assert any("does NOT mean the tape is clean" in w
               for w in payload["warnings"])


# =========================================================================== #
# Governance holds for every one of them
# =========================================================================== #
AGGREGATE_TOOLS = (
    ("portfolio_summary", {}),
    ("stratify", {"dimension": "account_status"}),
    ("concentration", {"dimension": "borrower_identifier"}),
    ("rank_loans", {"metric": "current_loan_to_value"}),
    ("data_completeness", {}),
    ("list_validation_exceptions", {}),
)


@pytest.mark.parametrize("tool,extra", AGGREGATE_TOOLS)
def test_an_ungranted_resource_is_refused_by_every_analysis_tool(tool, extra):
    result = _run(tool, {"resource": PORTFOLIO_B.key, **extra})
    assert result.status == STATUS_BLOCKED, tool
    assert result.error_code == ErrorCode.RESOURCE_NOT_AUTHORISED, tool


@pytest.mark.parametrize("tool,extra", AGGREGATE_TOOLS)
def test_an_spv_scoped_resource_is_refused_rather_than_widened(tool, extra):
    """Answering from the enclosing book under an SPV's label is the fail-open
    every one of these has to refuse independently."""
    result = _run(tool, {"resource": SPV_I.key, **extra},
                  context=_context(SPV_I, capabilities=BOTH))
    assert result.status == STATUS_BLOCKED, tool
    assert result.error_code == ErrorCode.RESOURCE_NOT_PARTITIONABLE, tool


@pytest.mark.parametrize("tool,extra", AGGREGATE_TOOLS)
def test_every_analysis_tool_states_the_scope_it_answered_under(tool, extra):
    payload = _ok(tool, {"resource": PORTFOLIO_A.key, **extra})
    scope = payload["scope"]
    assert scope["resource"] == PORTFOLIO_A.key
    assert scope["source_portfolio_ids"] == ["direct_001"]


def test_aggregates_do_not_require_the_loan_read_capability():
    """Aggregate MI tells you a book's shape; it does not expose an individual
    obligation. rank_loans DOES name loans, so it needs the stronger grant."""
    risk_only = _context(PORTFOLIO_A, capabilities=(SCOPE_RISK_READ,))
    assert _run("portfolio_summary", {"resource": PORTFOLIO_A.key},
                context=risk_only).status == STATUS_SUCCESS

    blocked = _run("rank_loans", {"resource": PORTFOLIO_A.key,
                                  "metric": "current_loan_to_value"},
                   context=risk_only)
    assert blocked.status == STATUS_BLOCKED
    assert blocked.error_code == ErrorCode.SCOPE_MISSING


# =========================================================================== #
# The efficiency claim, measured rather than asserted
# =========================================================================== #
def test_ranking_beats_retrieving_and_the_telemetry_shows_it(capsys):
    """The argument for the aggregate tools, made in the numbers an agent can
    actually see: rank_loans publishes rows_scanned and rows_returned, so a
    caller can tell that a shortlist cost a fraction of a retrieval."""
    ranked = _run("rank_loans", {"resource": PORTFOLIO_A.key,
                                 "metric": "current_loan_to_value", "limit": 3})
    retrieved = _run("get_loans", {"resource": PORTFOLIO_A.key,
                                   "loan_ids": sorted(TRUTH)})

    rank_telemetry = ranked.result["_telemetry"]
    get_telemetry = retrieved.result["_telemetry"]

    with capsys.disabled():
        print(f"\n    rank_loans  scanned={rank_telemetry['rows_scanned']:4d} "
              f"returned={rank_telemetry['rows_returned']:4d} "
              f"selectivity={rank_telemetry['selectivity']}"
              f"\n    get_loans   scanned={get_telemetry['rows_scanned']:4d} "
              f"returned={get_telemetry['rows_returned']:4d} "
              f"selectivity={get_telemetry['selectivity']}")

    assert rank_telemetry["rows_scanned"] == len(TRUTH)
    assert rank_telemetry["rows_returned"] == 3
    assert rank_telemetry["selectivity"] < get_telemetry["selectivity"]
    assert get_telemetry["selectivity"] == pytest.approx(1.0)


@pytest.mark.parametrize("tool,extra", AGGREGATE_TOOLS)
def test_every_tool_publishes_what_it_cost(tool, extra):
    """A client agent cannot read Trakt's logs. An agent that cannot see what its
    last call cost has no way to choose a cheaper next one."""
    payload = _ok(tool, {"resource": PORTFOLIO_A.key, **extra})
    telemetry = payload["_telemetry"]
    assert telemetry["tool"] == tool
    assert telemetry["duration_ms"] is not None
    assert telemetry["rows_returned"] >= 0
    assert isinstance(telemetry["cache"], dict)
