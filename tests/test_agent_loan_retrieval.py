#!/usr/bin/env python3
"""``get_loans`` is the primitive; ``get_loan`` is a wrapper over it.

The measurement that fixed this contract shape: retrieving twenty loans one at a
time cost 1,617 ms at one million rows against 76 ms for a single batched call —
a 21x penalty, before HTTP and the governance chain are counted. An agent given
only a single-loan tool calls it once per loan, so the batch form has to be the
primitive rather than an optimisation offered later.

What is asserted here:

* one dataset access and no per-loan lookup, whatever the batch size;
* ``get_loan`` reaches the data through ``get_loans`` and nowhere else;
* deterministic ordering, so a caller can zip results against its own list;
* bounded — the tool refuses a request large enough to be analysis;
* an absent loan is reported, never silently dropped;
* the authorisation boundary narrows the frame, and refuses when it cannot.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import trakt_tools
from trakt_core.context import (
    ACTOR_SERVICE,
    CHANNEL_ENTERPRISE_AGENT,
    SCOPE_LOAN_READ,
    SCOPE_RISK_READ,
    ExecutionContext,
)
from trakt_core.entitlement import EntitlementStore, Grant
from trakt_core.envelope import STATUS_BLOCKED, STATUS_SUCCESS
from trakt_core.errors import ErrorCode
from trakt_core.resource import (
    KIND_PORTFOLIO,
    KIND_SOURCE_PORTFOLIO,
    KIND_SPV,
    ResourceCatalogue,
    ResourceRecord,
    ResourceRef,
)
from trakt_tools.execution import ToolDependencies, execute_governed_tool
from trakt_tools.handlers.loans import MAX_LOAN_IDS

from tests.test_agent_governed_execution import _Datasets

TENANT = "ERE"
ORG = "a2a_test_agent"
PORTFOLIO_A = ResourceRef(TENANT, KIND_SOURCE_PORTFOLIO, "direct_001")
PORTFOLIO_B = ResourceRef(TENANT, KIND_SOURCE_PORTFOLIO, "acquired_001")
WHOLE_BOOK = ResourceRef(TENANT, KIND_PORTFOLIO, "ere_total")
SPV_I = ResourceRef(TENANT, KIND_SPV, "spv_i")
NO_PROVENANCE = ResourceRef(TENANT, KIND_SOURCE_PORTFOLIO, "unattributed")

N_ROWS = 4_000


def _frame(with_provenance: bool = True) -> pd.DataFrame:
    """A canonical-shaped frame: two books, deterministic values."""
    rows = []
    for i in range(N_ROWS):
        book = "direct_001" if i % 2 == 0 else "acquired_001"
        rows.append({
            "loan_identifier": f"LN{i:06d}",
            "borrower_identifier": f"BR{i // 3:06d}",
            "source_portfolio_id": book,
            "account_status": "performing" if i % 5 else "arrears",
            "current_principal_balance": 100_000.0 + i,
            "current_loan_to_value": 40.0 + (i % 50) / 10,
            "current_valuation_amount": 250_000.0 + i,
            "geographic_region_obligor": ["South East", "London", "Wales"][i % 3],
            "reporting_date": "2026-07-31",
        })
    df = pd.DataFrame(rows)
    if not with_provenance:
        df = df.drop(columns=["source_portfolio_id"])
    return df


class _CountingResolver:
    """Counts how many times the governed frame is resolved."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.df


def _catalogue() -> ResourceCatalogue:
    return ResourceCatalogue([
        ResourceRecord(ref=PORTFOLIO_A, display_label="Portfolio A",
                       source_portfolio_ids=("direct_001",)),
        ResourceRecord(ref=PORTFOLIO_B, display_label="Portfolio B",
                       source_portfolio_ids=("acquired_001",)),
        ResourceRecord(ref=WHOLE_BOOK, display_label="ERE total",
                       whole_tenant_book=True),
        ResourceRecord(ref=SPV_I, display_label="SPV I", spv_id="SPV_I",
                       source_portfolio_ids=("direct_001",)),
        ResourceRecord(ref=NO_PROVENANCE, display_label="Unattributed",
                       source_portfolio_ids=("direct_001",)),
    ], configured=True)


def _context(*refs, capabilities=(SCOPE_LOAN_READ,)) -> ExecutionContext:
    refs = refs or (PORTFOLIO_A,)
    store = EntitlementStore(
        [Grant(organisation_id=ORG, resource_ref=r,
               capabilities=frozenset(capabilities)) for r in refs],
        configured=True, validate=False)
    return ExecutionContext(
        tenant_id=TENANT, actor_id="sp-agent", actor_type=ACTOR_SERVICE,
        channel=CHANNEL_ENTERPRISE_AGENT, scopes=frozenset(capabilities),
        organisation_id=ORG, entitlements=store.resolve(ORG))


def _deps(resolver=None, datasets=None) -> ToolDependencies:
    return ToolDependencies(
        datasets=datasets or _Datasets(), runtime_mode="test",
        catalogue=_catalogue(), output_root="/unused",
        loan_frame_resolver=resolver or _CountingResolver(_frame()))


def _run(tool: str, args: Dict[str, Any], *, context=None, deps=None):
    return execute_governed_tool(tool, args, context or _context(),
                                 dependencies=deps or _deps())


# --------------------------------------------------------------------------- #
# The primitive
# --------------------------------------------------------------------------- #
def test_a_batch_resolves_the_dataset_exactly_once():
    """The anti-N+1 assertion, made structurally rather than by timing."""
    resolver = _CountingResolver(_frame())
    ids = [f"LN{i:06d}" for i in range(0, 40, 2)]      # 20 loans, all in book A
    result = _run("get_loans", {"resource": PORTFOLIO_A.key, "loan_ids": ids},
                  deps=_deps(resolver))
    assert result.status == STATUS_SUCCESS, result.error and result.error.to_dict()
    assert result.result["returned_count"] == 20
    assert resolver.calls == 1, (
        f"the frame was resolved {resolver.calls} times for one batch")


def test_results_come_back_in_the_order_requested():
    """So a caller can zip against its own list without matching on a key."""
    ids = ["LN000010", "LN000002", "LN000006", "LN000000"]
    result = _run("get_loans", {"resource": PORTFOLIO_A.key, "loan_ids": ids})
    returned = [loan["loan_identifier"] for loan in result.result["loans"]]
    assert returned == ids


def test_a_missing_loan_is_reported_not_silently_dropped():
    """498 of 500 returned without saying which two are missing would be read as
    a smaller book, not as a gap."""
    ids = ["LN000000", "LN999999", "LN000002"]
    result = _run("get_loans", {"resource": PORTFOLIO_A.key, "loan_ids": ids})
    payload = result.result
    assert [l["loan_identifier"] for l in payload["loans"]] == ["LN000000", "LN000002"]
    assert payload["not_found"] == ["LN999999"]
    assert payload["requested_count"] == 3
    assert payload["returned_count"] == 2


def test_duplicate_identifiers_are_collapsed_and_disclosed():
    result = _run("get_loans", {"resource": PORTFOLIO_A.key,
                                "loan_ids": ["LN000000", "LN000000", "LN000002"]})
    payload = result.result
    assert payload["returned_count"] == 2
    assert any("duplicate" in w for w in payload["warnings"])


def test_the_batch_is_bounded():
    """Above the ceiling the caller is doing analysis, and should be aggregating."""
    ids = [f"LN{i:06d}" for i in range(MAX_LOAN_IDS + 1)]
    result = _run("get_loans", {"resource": PORTFOLIO_A.key, "loan_ids": ids})
    assert result.status in (STATUS_BLOCKED, "error")
    assert result.error_code == ErrorCode.TOOL_INPUT_INVALID


def test_the_default_projection_is_curated_not_every_column():
    result = _run("get_loans", {"resource": PORTFOLIO_A.key, "loan_ids": ["LN000000"]})
    fields = result.result["fields_returned"]
    assert "loan_identifier" in fields
    assert "current_principal_balance" in fields
    # The default is a cross-asset superset intersected with this tape.
    assert set(fields) <= set(_frame().columns)


def test_an_explicit_field_list_is_honoured_and_gaps_are_disclosed():
    result = _run("get_loans", {
        "resource": PORTFOLIO_A.key, "loan_ids": ["LN000000"],
        "fields": ["current_loan_to_value", "not_on_this_tape"]})
    payload = result.result
    assert "current_loan_to_value" in payload["fields_returned"]
    assert "loan_identifier" in payload["fields_returned"]     # always included
    assert payload["fields_unavailable"] == ["not_on_this_tape"]
    assert any("not_on_this_tape" in w for w in payload["warnings"])


# --------------------------------------------------------------------------- #
# The wrapper
# --------------------------------------------------------------------------- #
def test_get_loan_is_a_wrapper_with_no_lookup_of_its_own():
    """Structural: one implementation, not two."""
    import inspect

    from trakt_tools.handlers import loans as loans_mod

    source = inspect.getsource(loans_mod.get_loan)
    assert "get_loans(" in source
    for forbidden in ("resolve_governed_frame", "isin", "to_dict(orient"):
        assert forbidden not in source, (
            f"get_loan performs its own {forbidden} — it must delegate")


def test_get_loan_returns_the_same_record_as_a_one_element_batch():
    single = _run("get_loan", {"resource": PORTFOLIO_A.key, "loan_id": "LN000004"})
    batch = _run("get_loans", {"resource": PORTFOLIO_A.key, "loan_ids": ["LN000004"]})
    assert single.result["found"] is True
    assert single.result["loan"] == batch.result["loans"][0]


def test_get_loan_reports_a_miss_rather_than_erroring():
    result = _run("get_loan", {"resource": PORTFOLIO_A.key, "loan_id": "LN999999"})
    assert result.status == STATUS_SUCCESS
    assert result.result["found"] is False
    assert result.result["loan"] is None


# --------------------------------------------------------------------------- #
# The authorisation boundary
# --------------------------------------------------------------------------- #
def test_the_resource_narrows_the_frame_to_its_own_book():
    """A loan in the other book is 'not found', not 'here it is'."""
    result = _run("get_loans", {"resource": PORTFOLIO_A.key,
                                "loan_ids": ["LN000000", "LN000001"]})
    payload = result.result
    assert [l["loan_identifier"] for l in payload["loans"]] == ["LN000000"]
    assert payload["not_found"] == ["LN000001"]      # odd ids are in book B


def test_an_ungranted_resource_is_refused():
    result = _run("get_loans", {"resource": PORTFOLIO_B.key,
                                "loan_ids": ["LN000001"]})
    assert result.status == STATUS_BLOCKED
    assert result.error_code == ErrorCode.RESOURCE_NOT_AUTHORISED


def test_the_loan_read_capability_is_required_separately_from_mi():
    """Aggregate MI tells you a book's shape; this exposes individual
    obligations, so it is a separate grant."""
    risk_only = _context(PORTFOLIO_A, capabilities=(SCOPE_RISK_READ,))
    result = _run("get_loans", {"resource": PORTFOLIO_A.key, "loan_ids": ["LN000000"]},
                  context=risk_only)
    assert result.status == STATUS_BLOCKED
    assert result.error_code == ErrorCode.SCOPE_MISSING


def test_loan_read_is_not_granted_by_default_to_human_channels():
    from trakt_core.context import DEFAULT_MI_SCOPES
    assert SCOPE_LOAN_READ not in DEFAULT_MI_SCOPES


def test_a_book_scoped_resource_refuses_a_tape_that_cannot_carry_the_boundary():
    """`apply_scope`-style fail-open is the exact bug this refuses: a frame with
    no source_portfolio_id would return the whole book under a narrower label."""
    resolver = _CountingResolver(_frame(with_provenance=False))
    result = _run("get_loans", {"resource": NO_PROVENANCE.key,
                                "loan_ids": ["LN000000"]},
                  context=_context(NO_PROVENANCE), deps=_deps(resolver))
    assert result.status == STATUS_BLOCKED
    assert result.error_code == ErrorCode.RESOURCE_NOT_PARTITIONABLE
    assert "source_portfolio_id" in result.error.message


def test_an_spv_scoped_resource_is_refused_rather_than_widened():
    result = _run("get_loans", {"resource": SPV_I.key, "loan_ids": ["LN000000"]},
                  context=_context(SPV_I))
    assert result.status == STATUS_BLOCKED
    assert result.error_code == ErrorCode.RESOURCE_NOT_PARTITIONABLE


def test_a_whole_book_resource_sees_both_books():
    result = _run("get_loans", {"resource": WHOLE_BOOK.key,
                                "loan_ids": ["LN000000", "LN000001"]},
                  context=_context(WHOLE_BOOK))
    assert result.result["returned_count"] == 2


# --------------------------------------------------------------------------- #
# The measurement that decided the contract
# --------------------------------------------------------------------------- #
def test_one_batch_beats_twenty_wrapper_calls(capsys):
    """Recorded, not merely asserted — the ratio is the argument for the shape.

    The threshold is deliberately loose (2x): this runs on shared CI hardware and
    the point is the ORDER of the difference, not a precise figure.
    """
    df = _frame()
    ids = [f"LN{i:06d}" for i in range(0, 40, 2)]
    context = _context()

    def twenty_singles():
        for loan_id in ids:
            execute_governed_tool("get_loan",
                                  {"resource": PORTFOLIO_A.key, "loan_id": loan_id},
                                  context,
                                  dependencies=_deps(_CountingResolver(df)))

    def one_batch():
        execute_governed_tool("get_loans",
                              {"resource": PORTFOLIO_A.key, "loan_ids": ids},
                              context, dependencies=_deps(_CountingResolver(df)))

    twenty_singles(); one_batch()                      # warm
    t = time.perf_counter(); twenty_singles(); singles_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter(); one_batch(); batch_ms = (time.perf_counter() - t) * 1000

    with capsys.disabled():
        print(f"\n    20 loans via get_loan x20 : {singles_ms:8.1f} ms"
              f"\n    20 loans via get_loans x1 : {batch_ms:8.1f} ms"
              f"\n    improvement               : {singles_ms / max(batch_ms, 1e-6):8.1f}x")

    assert batch_ms < singles_ms / 2, (
        f"batching gained only {singles_ms / max(batch_ms, 1e-6):.1f}x")
