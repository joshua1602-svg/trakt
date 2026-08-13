#!/usr/bin/env python3
"""Sprint 1, Deliverable 2: governed tool execution.

``execute_governed_tool`` is the one path an external agent call takes, and its
ORDER is the contract:

    1 tool exists → 2 arguments valid → 3 capability → 4 entitlement over the
    named resource → 5 dataset approved → 6 handler → 7 envelope → 8 audit

The tests here are the Sprint 1 exit tests for that deliverable:

* a caller without the correct scope is refused;
* a caller without an entitlement to the portfolio is refused;
* an unauthorised portfolio and a nonexistent one are **indistinguishable**;
* a caller with the correct permissions receives the values the existing
  covenant implementation produced, unchanged;
* nothing reads data before step 5 — asserted by counting calls to the dataset
  resolver, not by reading the code;
* every path, including every refusal, emits an audit event.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import trakt_tools
from trakt_core.context import (
    ACTOR_SERVICE,
    CHANNEL_ENTERPRISE_AGENT,
    SCOPE_MI_QUERY,
    SCOPE_RISK_READ,
    ExecutionContext,
)
from trakt_core.entitlement import EntitlementStore, Grant
from trakt_core.envelope import STATUS_BLOCKED, STATUS_ERROR, STATUS_SUCCESS
from trakt_core.errors import ErrorCode
from trakt_core.resource import (
    KIND_PORTFOLIO,
    KIND_SOURCE_PORTFOLIO,
    KIND_SPV,
    POPULATION_PIPELINE,
    ResourceCatalogue,
    ResourceRecord,
    ResourceRef,
)
from trakt_tools.execution import ToolDependencies, execute_governed_tool

# --------------------------------------------------------------------------- #
# The fixture: one tenant, three resources, one entitled agent.
#
#   ERE
#   ├── source_portfolio/direct_001    PORTFOLIO A   granted (risk:read)
#   ├── source_portfolio/acquired_001  PORTFOLIO B   NOT granted
#   ├── population/pipeline                          granted, wrong population
#   └── spv/spv_i                                    granted, SPV narrowing
# --------------------------------------------------------------------------- #
TENANT = "ERE"
AGENT_ORG = "a2a_test_agent"

PORTFOLIO_A = ResourceRef(TENANT, KIND_SOURCE_PORTFOLIO, "direct_001")
PORTFOLIO_B = ResourceRef(TENANT, KIND_SOURCE_PORTFOLIO, "acquired_001")
NONEXISTENT = ResourceRef(TENANT, KIND_SOURCE_PORTFOLIO, "never_registered")
WHOLE_BOOK = ResourceRef(TENANT, KIND_PORTFOLIO, "ere_total")
PIPELINE = ResourceRef(TENANT, "population", "pipeline")
SPV_I = ResourceRef(TENANT, KIND_SPV, "spv_i")


def _catalogue() -> ResourceCatalogue:
    return ResourceCatalogue([
        ResourceRecord(ref=PORTFOLIO_A, display_label="Portfolio A",
                       source_portfolio_ids=("direct_001",)),
        ResourceRecord(ref=PORTFOLIO_B, display_label="Portfolio B",
                       source_portfolio_ids=("acquired_001",)),
        ResourceRecord(ref=WHOLE_BOOK, display_label="ERE total",
                       whole_tenant_book=True),
        ResourceRecord(ref=PIPELINE, display_label="ERE pipeline",
                       population=POPULATION_PIPELINE),
        ResourceRecord(ref=SPV_I, display_label="SPV I", spv_id="SPV_I",
                       source_portfolio_ids=("direct_001",)),
    ], configured=True)


def _context(*, grants: Optional[Dict[ResourceRef, List[str]]] = None,
             scopes: Optional[List[str]] = None,
             organisation: Optional[str] = AGENT_ORG) -> ExecutionContext:
    """A service context shaped exactly as ``agent_auth`` produces one."""
    grants = {PORTFOLIO_A: [SCOPE_RISK_READ]} if grants is None else grants
    entitlements = None
    if organisation is not None:
        store = EntitlementStore(
            [Grant(organisation_id=organisation, resource_ref=ref,
                   capabilities=frozenset(caps))
             for ref, caps in grants.items()],
            configured=True, validate=False)
        entitlements = store.resolve(organisation)
    if scopes is None:
        # The production derivation: a machine's scopes are the union of its
        # grants (see identity.scopes_from_entitlements), never a default set.
        scopes = sorted({c for caps in grants.values() for c in caps})
    return ExecutionContext(
        tenant_id=TENANT, actor_id="sp-agent-1", actor_type=ACTOR_SERVICE,
        channel=CHANNEL_ENTERPRISE_AGENT, scopes=frozenset(scopes),
        organisation_id=organisation, entitlements=entitlements)


@dataclass
class _Descriptor:
    source_base: str = "platform_canonical"
    available: bool = True
    source_kind: str = "platform_canonical"
    label: str = "ere_202607_central_tape.csv"
    reporting_date: str = "2026-07-31"
    snapshot_id: str = "snap_test_0001"
    content_hash: str = "sha256:abc123"
    row_count: int = 4821
    source_portfolios: tuple = ("direct_001", "acquired_001")


class _Datasets:
    """A dataset resolver that COUNTS its calls.

    The count is how "governance runs before data" is proven: a refusal at
    steps 1-4 must leave it at zero. Asserting on behaviour rather than reading
    the source is what makes the ordering contract enforceable.
    """

    def __init__(self, descriptor: Optional[_Descriptor] = None, raises: bool = False):
        self.descriptor = descriptor or _Descriptor()
        self.raises = raises
        self.calls = 0

    def describe_active_dataset(self):
        self.calls += 1
        if self.raises:
            raise RuntimeError("blob storage unreachable")
        return self.descriptor


#: What the shared covenant implementation returns. The tool must deliver these
#: numbers unchanged — that identity is the point of the whole exercise.
EVALUATED_ENVELOPE: Dict[str, Any] = {
    "portfolioId": TENANT,
    "available": True,
    "source": "approved_configuration",
    "approvalStatus": "approved",
    "reportingDate": "2026-07-31",
    "priorReportingDate": "2026-06-30",
    "tests": [
        {"testId": "GEO_SE", "metricId": "share_of_balance",
         "displayName": "South East concentration", "category": "geographic",
         "currentValue": 34.7182, "priorValue": 31.9,
         "threshold": 33.0, "operator": "<=", "unit": "percent",
         "utilization": 1.0520666666666667, "headroom": -1.7182,
         "breachAmount": 1_718_200.0, "status": "breach", "priorStatus": "pass",
         "statusTransition": "pass_to_breach", "deteriorated": True,
         "loansInNumerator": 1_204, "totalLoans": 4_821,
         "missingFields": [], "numeratorValue": 34_718_200.0,
         "denominatorValue": 100_000_000.0, "denominatorBasis": "current_balance",
         "provenance": {"approvedBy": "A. Operator", "sourceReference": "Schedule 8 §4.2"},
         "definition": {"numerator": "balance in region"},
         "configurationVersion": 7},
        {"testId": "BRK_TOP1", "metricId": "largest_group_share",
         "displayName": "Largest broker", "category": "counterparty",
         "currentValue": 12.5, "priorValue": None, "threshold": 20.0,
         "operator": "<=", "unit": "percent", "utilization": 0.625,
         "headroom": 7.5, "breachAmount": None, "status": "pass",
         "loansInNumerator": 601, "totalLoans": 4_821, "missingFields": [],
         "provenance": {}, "definition": {}, "configurationVersion": 7},
    ],
    "summary": {"overallStatus": "breach", "activeTests": 2, "breaches": 1,
                "warnings": 0, "passes": 1, "unavailable": 0,
                "deteriorations": 1, "reportingDate": "2026-07-31",
                "priorAvailable": True, "closestToLimit": "GEO_SE"},
    "lineage": {"source": "approved_configuration", "configurationVersion": 7,
                "configurationHash": "cfg_9f2c", "activatedBy": "A. Operator",
                "activatedAt": "2026-07-02T10:00:00Z", "libraryVersion": "1.0.0",
                "dataSource": "governed funded canonical frames",
                "reportingDate": "2026-07-31",
                "priorReportingDate": "2026-06-30"},
}


def _evaluator(captured: Optional[Dict[str, Any]] = None):
    """Stands in for ``compute_concentration_tests`` with the SAME signature."""
    def _evaluate(output_root, client_id, to_run_id, *, scope=None):
        if captured is not None:
            captured.update({"output_root": output_root, "client_id": client_id,
                             "to_run_id": to_run_id, "scope": scope})
        return dict(EVALUATED_ENVELOPE)
    return _evaluate


def _deps(datasets: Optional[_Datasets] = None, *, evaluator=None,
          mode: str = "test") -> ToolDependencies:
    return ToolDependencies(
        datasets=datasets or _Datasets(),
        runtime_mode=mode,
        catalogue=_catalogue(),
        output_root="/does/not/matter",
        covenant_evaluator=evaluator or _evaluator(),
    )


def _run(resource, *, context=None, deps=None, tool="evaluate_covenants",
         arguments=None):
    args = {"resource": resource.key if hasattr(resource, "key") else resource}
    if arguments is not None:
        args = arguments
    return execute_governed_tool(tool, args, context or _context(),
                                 dependencies=deps or _deps())


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #
def test_an_entitled_agent_receives_the_shared_implementations_values_unchanged():
    """The Deliverable 2 exit test.

    Every number the agent receives is the number the covenant implementation
    produced. Not rounded, not re-derived, not defaulted.
    """
    result = _run(PORTFOLIO_A)
    assert result.status == STATUS_SUCCESS, result.error and result.error.to_dict()

    payload = result.result
    assert payload["available"] is True
    assert payload["source"] == "approved_configuration"

    expected = {t["testId"]: t for t in EVALUATED_ENVELOPE["tests"]}
    assert len(payload["tests"]) == len(expected)
    for got in payload["tests"]:
        src = expected[got["test_id"]]
        assert got["current_value"] == src["currentValue"]
        assert got["prior_value"] == src["priorValue"]
        assert got["threshold"] == src["threshold"]
        assert got["utilisation"] == src["utilization"]
        assert got["headroom"] == src["headroom"]
        assert got["breach_amount"] == src["breachAmount"]
        assert got["status"] == src["status"]
        assert got["total_loans"] == src["totalLoans"]

    assert payload["summary"]["overall_status"] == "breach"
    assert payload["summary"]["breaches"] == 1


def test_the_result_carries_the_configuration_provenance_an_agent_needs():
    """A covenant number without the definition that produced it is an assertion,
    not evidence."""
    payload = _run(PORTFOLIO_A).result
    lineage = payload["lineage"]
    assert lineage["configuration_version"] == 7
    assert lineage["configuration_hash"] == "cfg_9f2c"
    assert lineage["activated_by"] == "A. Operator"
    assert lineage["library_version"] == "1.0.0"
    assert lineage["reporting_date"] == "2026-07-31"

    breach = next(t for t in payload["tests"] if t["test_id"] == "GEO_SE")
    assert breach["provenance"]["approved_by"] == "A. Operator"
    assert breach["provenance"]["source_reference"] == "Schedule 8 §4.2"


def test_the_envelope_carries_the_snapshot_the_answer_came_from():
    result = _run(PORTFOLIO_A)
    assert result.snapshot.snapshot_id == "snap_test_0001"
    assert result.snapshot.content_hash == "sha256:abc123"
    assert result.snapshot.reporting_date == "2026-07-31"
    assert result.policy.data_approved is True
    assert result.policy.tenant_authorised is True
    assert result.provenance.snapshot.snapshot_id == "snap_test_0001"


def test_the_authorised_narrowing_is_passed_to_the_evaluator_and_disclosed():
    """The scope the evaluation ran under is the scope the ENTITLEMENT permits —
    not a lens the caller chose."""
    captured: Dict[str, Any] = {}
    result = _run(PORTFOLIO_A, deps=_deps(evaluator=_evaluator(captured)))

    scope = captured["scope"]
    assert scope is not None
    assert tuple(scope.portfolio_ids) == ("direct_001",)
    # The tenant comes from the trusted context, never from an argument.
    assert captured["client_id"] == TENANT

    assert result.result["scope"]["source_portfolio_ids"] == ["direct_001"]
    assert result.result["scope"]["resource"] == PORTFOLIO_A.key


def test_a_whole_book_resource_runs_unnarrowed_because_it_says_so():
    captured: Dict[str, Any] = {}
    context = _context(grants={WHOLE_BOOK: [SCOPE_RISK_READ]})
    result = execute_governed_tool(
        "evaluate_covenants", {"resource": WHOLE_BOOK.key}, context,
        dependencies=_deps(evaluator=_evaluator(captured)))
    assert result.status == STATUS_SUCCESS
    assert captured["scope"] is None
    assert result.result["scope"]["whole_tenant_book"] is True


# --------------------------------------------------------------------------- #
# Refusals — the Sprint 1 permission exit tests
# --------------------------------------------------------------------------- #
def test_a_caller_without_the_capability_is_refused():
    """Deliverable 2 exit test: no scope, no call."""
    context = _context(grants={PORTFOLIO_A: [SCOPE_MI_QUERY]})
    assert SCOPE_RISK_READ not in context.scopes

    result = _run(PORTFOLIO_A, context=context)
    assert result.status == STATUS_BLOCKED
    assert result.error_code == ErrorCode.SCOPE_MISSING
    assert result.http_status == 403


def test_a_caller_without_an_entitlement_to_the_portfolio_is_refused():
    """Deliverable 3 exit test: Portfolio A yes, Portfolio B no — same agent,
    same capability, same call."""
    context = _context()
    assert _run(PORTFOLIO_A, context=context).status == STATUS_SUCCESS

    refused = _run(PORTFOLIO_B, context=context)
    assert refused.status == STATUS_BLOCKED
    assert refused.error_code == ErrorCode.RESOURCE_NOT_AUTHORISED
    assert refused.http_status == 403


def test_an_unauthorised_portfolio_and_a_nonexistent_one_are_indistinguishable():
    """Deliverable 3 exit test, and the one worth being strict about.

    If a refusal for a real-but-ungranted book differed from a refusal for a book
    that does not exist, a caller could enumerate another organisation's
    portfolios by comparing responses. Code, message and details must match.
    """
    context = _context()
    ungranted = _run(PORTFOLIO_B, context=context)
    absent = _run(NONEXISTENT, context=context)

    assert ungranted.error_code == absent.error_code == ErrorCode.RESOURCE_NOT_AUTHORISED
    assert ungranted.error.message == absent.error.message
    assert ungranted.http_status == absent.http_status
    # `details` names the resource the caller asked for — which it already knows —
    # and nothing about whether it exists.
    assert set(ungranted.error.details) == set(absent.error.details)
    assert ungranted.error.details.get("resource") == PORTFOLIO_B.key
    assert absent.error.details.get("resource") == NONEXISTENT.key


def test_a_context_with_no_entitlements_is_refused_not_waved_through():
    """'No entitlements' is never 'no restrictions'."""
    context = _context(organisation=None, scopes=[SCOPE_RISK_READ])
    assert context.entitlements is None

    result = _run(PORTFOLIO_A, context=context)
    assert result.status == STATUS_BLOCKED
    assert result.error_code == ErrorCode.ENTITLEMENTS_NOT_RESOLVED


def test_a_grant_for_another_tenant_is_refused():
    other = ResourceRef("OTHER_CLIENT", KIND_SOURCE_PORTFOLIO, "direct_001")
    context = _context(grants={other: [SCOPE_RISK_READ]})
    result = _run(other, context=context)
    assert result.status == STATUS_BLOCKED
    assert result.error_code == ErrorCode.TENANT_MISMATCH


# --------------------------------------------------------------------------- #
# Refusals the handler owns — a constraint it cannot honour
# --------------------------------------------------------------------------- #
def test_a_resource_pinned_to_another_population_is_refused_not_answered():
    """Concentration tests are defined on the funded book. Answering a
    pipeline-pinned resource from the funded book would report on data the
    resource does not name."""
    context = _context(grants={PIPELINE: [SCOPE_RISK_READ]})
    result = execute_governed_tool("evaluate_covenants", {"resource": PIPELINE.key},
                                   context, dependencies=_deps())
    assert result.status == STATUS_BLOCKED
    assert result.error_code == ErrorCode.RESOURCE_NOT_PARTITIONABLE
    assert "funded" in result.error.message


def test_an_spv_scoped_resource_is_refused_rather_than_widened():
    """The funded concentration path narrows by BOOK. An SPV grant answered here
    would silently return the enclosing book — the exact fail-open the resource
    model exists to prevent."""
    context = _context(grants={SPV_I: [SCOPE_RISK_READ]})
    result = execute_governed_tool("evaluate_covenants", {"resource": SPV_I.key},
                                   context, dependencies=_deps())
    assert result.status == STATUS_BLOCKED
    assert result.error_code == ErrorCode.RESOURCE_NOT_PARTITIONABLE
    assert "SPV" in result.error.message


def test_a_book_scoped_call_is_refused_when_the_tape_cannot_carry_the_boundary(monkeypatch):
    """``apply_scope`` returns the frame UNCHANGED when it has no
    ``source_portfolio_id`` column. For a UI lens that is a degradation; for an
    entitlement it is the whole book with a narrower label on it."""
    from mi_agent_api import concentration_tests_api as conc_mod

    monkeypatch.setattr(
        conc_mod, "funded_attribution_status",
        lambda root, client_id, run_id=None: {
            "available": True, "attributed": False,
            "reason": "The governed funded tape carries no source_portfolio_id "
                      "column, so a book boundary cannot be enforced on it."})

    result = _run(PORTFOLIO_A)
    assert result.status == STATUS_BLOCKED
    assert result.error_code == ErrorCode.RESOURCE_NOT_PARTITIONABLE
    assert "source_portfolio_id" in result.error.message


# --------------------------------------------------------------------------- #
# Ordering — governance before data
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", ["unknown_tool", "bad_arguments",
                                  "missing_scope", "not_entitled"])
def test_nothing_reads_data_before_authorisation_succeeds(case):
    """Counted, not read off the source.

    ``_Datasets.calls`` must still be zero for every refusal that happens at
    steps 1-4, which is what "no dataframe is touched for a caller that is not
    entitled to it" means operationally.
    """
    datasets = _Datasets()
    deps = _deps(datasets)

    if case == "unknown_tool":
        result = execute_governed_tool("no_such_tool", {"resource": PORTFOLIO_A.key},
                                       _context(), dependencies=deps)
        assert result.error_code == ErrorCode.TOOL_NOT_FOUND
    elif case == "bad_arguments":
        result = execute_governed_tool("evaluate_covenants", {}, _context(),
                                       dependencies=deps)
        assert result.error_code == ErrorCode.TOOL_INPUT_INVALID
    elif case == "missing_scope":
        result = _run(PORTFOLIO_A, context=_context(grants={PORTFOLIO_A: [SCOPE_MI_QUERY]}),
                      deps=deps)
        assert result.error_code == ErrorCode.SCOPE_MISSING
    else:
        result = _run(PORTFOLIO_B, deps=deps)
        assert result.error_code == ErrorCode.RESOURCE_NOT_AUTHORISED

    assert datasets.calls == 0, (
        f"the dataset resolver was called {datasets.calls} time(s) for a "
        f"{case} refusal — governance must run before data")


def test_the_dataset_is_read_exactly_once_on_the_authorised_path():
    datasets = _Datasets()
    result = _run(PORTFOLIO_A, deps=_deps(datasets))
    assert result.status == STATUS_SUCCESS
    assert datasets.calls == 1


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def test_unknown_tool_names_the_available_tools_rather_than_guessing():
    result = execute_governed_tool("evaluate_covenannts", {"resource": PORTFOLIO_A.key},
                                   _context(), dependencies=_deps())
    assert result.error_code == ErrorCode.TOOL_NOT_FOUND
    assert result.http_status == 404
    assert "evaluate_covenants" in result.error.details["available_tools"]


def test_invalid_arguments_report_the_failing_paths_but_not_the_values():
    """An agent needs to know WHICH argument was wrong to correct it. It does not
    need its own value echoed back, and echoing one would put caller data into an
    error body and, through it, into a client-side log."""
    result = execute_governed_tool(
        "evaluate_covenants",
        {"resource": "not-a-resource-reference", "secret_note": "borrower name"},
        _context(), dependencies=_deps())
    assert result.error_code == ErrorCode.TOOL_INPUT_INVALID
    assert result.http_status == 400
    violations = " ".join(result.error.details["violations"])
    assert "resource" in violations and "secret_note" in violations
    assert "borrower name" not in violations
    assert "not-a-resource-reference" not in violations


def test_an_optional_argument_may_be_omitted_and_takes_its_published_default():
    captured: Dict[str, Any] = {}
    result = execute_governed_tool(
        "evaluate_covenants", {"resource": PORTFOLIO_A.key}, _context(),
        dependencies=_deps(evaluator=_evaluator(captured)))
    assert result.status == STATUS_SUCCESS
    assert captured["to_run_id"] is None


def test_a_category_filter_that_matches_nothing_warns_rather_than_returning_everything():
    """Silently returning the full set would make an agent read the whole book as
    the filtered answer."""
    result = execute_governed_tool(
        "evaluate_covenants",
        {"resource": PORTFOLIO_A.key, "category": "liquidity"},
        _context(), dependencies=_deps())
    assert result.status == STATUS_SUCCESS
    assert result.result["tests"] == []
    assert any("liquidity" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# Policy and failure handling
# --------------------------------------------------------------------------- #
def test_an_unapproved_data_source_blocks_after_authorisation():
    datasets = _Datasets(_Descriptor(source_base="synthetic_demo"))
    result = _run(PORTFOLIO_A, deps=_deps(datasets, mode="production"))
    assert result.status == STATUS_BLOCKED
    assert result.error_code in (ErrorCode.DATA_SOURCE_NOT_APPROVED,
                                 ErrorCode.DATA_SOURCE_UNAVAILABLE)
    assert result.policy.data_approved is False


def test_an_unavailable_dataset_is_a_governed_outcome_not_a_crash():
    datasets = _Datasets(_Descriptor(available=False, source_base="unavailable"))
    result = _run(PORTFOLIO_A, deps=_deps(datasets))
    assert result.error_code == ErrorCode.DATA_SOURCE_UNAVAILABLE


def test_a_storage_fault_is_a_typed_error_not_a_raw_500():
    result = _run(PORTFOLIO_A, deps=_deps(_Datasets(raises=True)))
    assert result.status == STATUS_ERROR
    assert result.error_code == ErrorCode.STORAGE_UNAVAILABLE
    assert "blob" not in result.error.message.lower()


def test_a_handler_defect_never_leaks_internal_detail_to_the_agent():
    def _explode(output_root, client_id, to_run_id, *, scope=None):
        raise KeyError("/mnt/blob/ERE/secret/path.csv")

    result = _run(PORTFOLIO_A, deps=_deps(evaluator=_explode))
    assert result.status == STATUS_ERROR
    assert result.error_code == ErrorCode.TOOL_EXECUTION_FAILED
    serialised = str(result.to_dict())
    assert "/mnt/blob" not in serialised
    assert "KeyError" not in serialised


def test_execute_governed_tool_never_raises_across_the_boundary():
    """Whatever goes wrong, an agent gets an envelope it can parse."""
    for args in ({}, {"resource": 42}, {"resource": PORTFOLIO_B.key},
                 {"resource": "a/b"}, None):
        result = execute_governed_tool("evaluate_covenants", args, _context(),
                                       dependencies=_deps())
        assert result.status in (STATUS_BLOCKED, STATUS_ERROR)
        assert result.error is not None
        assert result.to_dict()["error"]["code"]


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def _audit_events(caplog) -> List[str]:
    return [r.getMessage() for r in caplog.records
            if r.name == "trakt.audit"]


@pytest.mark.parametrize("resource,expected", [
    (PORTFOLIO_A, STATUS_SUCCESS),
    (PORTFOLIO_B, STATUS_BLOCKED),
])
def test_every_path_emits_exactly_one_audit_event(caplog, resource, expected):
    with caplog.at_level(logging.INFO, logger="trakt.audit"):
        result = _run(resource)
    assert result.status == expected
    events = _audit_events(caplog)
    assert len(events) == 1
    assert result.request_id in events[0]


def test_the_audit_line_records_the_organisation_the_resource_and_the_outcome():
    result = _run(PORTFOLIO_A)
    audit = result.audit.to_dict()
    assert audit["capability"] == "tool.evaluate_covenants"
    assert audit["organisation_id"] == AGENT_ORG
    assert audit["actor_type"] == ACTOR_SERVICE
    assert audit["channel"] == CHANNEL_ENTERPRISE_AGENT
    assert audit["tenant_id"] == TENANT
    assert audit["portfolio_id"] == PORTFOLIO_A.key
    assert audit["snapshot_id"] == "snap_test_0001"
    assert audit["outcome"] == STATUS_SUCCESS
    assert audit["duration_ms"] is not None


def test_the_audit_event_never_carries_the_answer_body(caplog):
    with caplog.at_level(logging.INFO, logger="trakt.audit"):
        _run(PORTFOLIO_A)
    line = _audit_events(caplog)[0]
    assert "34.7182" not in line          # no computed value
    assert "South East" not in line       # no test content
    assert "tests" not in line


def test_a_correlation_id_threads_the_agents_trace_through_to_the_audit_line():
    """What later makes 'show me every tool call behind this decision' a query
    rather than a reconstruction."""
    from dataclasses import replace
    context = replace(_context(), correlation_id="buyer-agent-run-42")
    result = _run(PORTFOLIO_A, context=context)
    assert result.correlation_id == "buyer-agent-run-42"
    assert result.audit.to_dict()["correlation_id"] == "buyer-agent-run-42"


# --------------------------------------------------------------------------- #
# The serialised contract an agent actually reads
# --------------------------------------------------------------------------- #
def test_the_serialised_result_is_json_safe_and_leaks_no_paths():
    import json

    result = _run(PORTFOLIO_A)
    blob = json.dumps(result.to_dict())          # must not raise
    assert "/does/not/matter" not in blob        # the output root never escapes
    payload = json.loads(blob)
    assert payload["capability"] == "tool.evaluate_covenants"
    assert payload["schema_version"]
    assert payload["status"] == STATUS_SUCCESS
    assert payload["tenant_id"] == TENANT
    assert payload["snapshot"]["content_hash"] == "sha256:abc123"
    assert payload["policy"]["data_approved"] is True
    assert payload["audit"]["organisation_id"] == AGENT_ORG
