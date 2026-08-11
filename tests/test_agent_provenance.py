#!/usr/bin/env python3
"""``explain_values`` is the primitive; ``explain_value`` is a wrapper.

Provenance is what turns a number into a fact. This suite holds two lines:

**It must be efficient.** Due diligence asks about tens of values at once, so the
batch form is the primitive — one frame access and one index load for the whole
request, not one of each per value.

**It must be exact.** Provenance from the wrong snapshot or the wrong tenant is
worse than no provenance, because it is confidently wrong. The index is bound to
one ``(tenant, snapshot)`` and refuses any other, and a snapshot with no index
reports that plainly rather than inventing a source.

The index itself is compact by construction — one entry per canonical FIELD, not
per cell — because mapping, transformation and validation are properties of a
field within a snapshot.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core import config_cache
from trakt_core.envelope import STATUS_BLOCKED, STATUS_ERROR, STATUS_SUCCESS
from trakt_core.errors import ErrorCode
from trakt_core.lineage import (
    ORIGIN_DERIVED,
    ORIGIN_MAPPED,
    ORIGIN_RUN_METADATA,
    LineageEntry,
    LineageIndex,
    build_index_from_field_lineage,
    load_lineage_index,
)
from trakt_tools.execution import ToolDependencies, execute_governed_tool

from tests.test_agent_governed_execution import _Datasets, _Descriptor
from tests.test_agent_loan_retrieval import (
    PORTFOLIO_A,
    PORTFOLIO_B,
    TENANT,
    _catalogue,
    _context,
    _CountingResolver,
    _frame,
)

SNAPSHOT_ID = "snap_test_0001"


@pytest.fixture(autouse=True)
def _clean_cache():
    config_cache.reset()
    yield
    config_cache.reset()


def _index(tenant=TENANT, snapshot_id=SNAPSHOT_ID) -> LineageIndex:
    return LineageIndex(
        tenant_id=tenant, snapshot_id=snapshot_id,
        content_hash="sha256:abc123", reporting_date="2026-07-31",
        source_dataset="ere_202607_central_tape.csv",
        entries={
            "current_principal_balance": LineageEntry(
                canonical_field="current_principal_balance", origin=ORIGIN_MAPPED,
                source_field="OUT_PRIN", source_dataset="july_servicer_tape.csv",
                mapping_method="alias", mapping_confidence=1.0,
                mapping_version="aliases_llm_confirmed.yaml@2026-06-14",
                validation_status="passed",
                validation_rules=("BAL001", "BAL101"), validation_error_count=0,
                unit=None, format="decimal"),
            "current_loan_to_value": LineageEntry(
                canonical_field="current_loan_to_value", origin=ORIGIN_DERIVED,
                derivation_rule="RESOLVE_CURRENT_LOAN_TO_VALUE",
                calculation_method="balance_over_valuation",
                calculation_version="1.2.0",
                validation_status="failed", validation_rules=("LTV001",),
                validation_error_count=3, unit="percentage_points",
                format="decimal"),
            "source_portfolio_id": LineageEntry(
                canonical_field="source_portfolio_id", origin=ORIGIN_RUN_METADATA,
                validation_status="passed", validation_rules=("PROV001",),
                validation_error_count=0),
        })


def _deps(index=None, snapshot_id=SNAPSHOT_ID, resolver=None) -> ToolDependencies:
    return ToolDependencies(
        datasets=_Datasets(_Descriptor(snapshot_id=snapshot_id)),
        runtime_mode="test", catalogue=_catalogue(), output_root="/unused",
        loan_frame_resolver=resolver or _CountingResolver(_frame()),
        lineage_index=index if index is not None else _index())


def _run(tool: str, args: Dict[str, Any], *, context=None, deps=None):
    return execute_governed_tool(tool, args, context or _context(),
                                 dependencies=deps or _deps())


# --------------------------------------------------------------------------- #
# The evidence itself
# --------------------------------------------------------------------------- #
def test_a_single_value_carries_the_minimum_provenance_envelope():
    result = _run("explain_value", {"resource": PORTFOLIO_A.key,
                                    "loan_id": "LN000000",
                                    "canonical_field": "current_principal_balance"})
    assert result.status == STATUS_SUCCESS
    payload = result.result
    assert payload["resolved"] is True
    e = payload["explanation"]

    assert e["value"] == 100_000.0
    assert e["canonical_field"] == "current_principal_balance"
    assert e["source_field"] == "OUT_PRIN"
    assert e["origin"] == ORIGIN_MAPPED
    assert e["source"]["snapshot_id"] == SNAPSHOT_ID
    assert e["source"]["content_hash"] == "sha256:abc123"
    assert e["source"]["source_dataset"] == "july_servicer_tape.csv"
    assert e["effective_date"] == "2026-07-31"
    assert e["transformation"]["mapping_method"] == "alias"
    assert e["transformation"]["mapping_version"] == "aliases_llm_confirmed.yaml@2026-06-14"
    assert e["validation"]["status"] == "passed"
    assert e["validation"]["rules_applied"] == ["BAL001", "BAL101"]


def test_a_derived_value_states_its_rule_and_calculation_version():
    """'Why is LTV 40.0?' must answer with the rule, not just the number."""
    result = _run("explain_value", {"resource": PORTFOLIO_A.key,
                                    "loan_id": "LN000000",
                                    "canonical_field": "current_loan_to_value"})
    e = result.result["explanation"]
    assert e["origin"] == ORIGIN_DERIVED
    assert e["transformation"]["derivation_rule"] == "RESOLVE_CURRENT_LOAN_TO_VALUE"
    assert e["calculation"]["method"] == "balance_over_valuation"
    assert e["calculation"]["version"] == "1.2.0"
    assert e["unit"] == "percentage_points"
    # A failing validation is disclosed, not smoothed over.
    assert e["validation"]["status"] == "failed"
    assert e["validation"]["error_count"] == 3


def test_a_run_metadata_field_says_so_rather_than_claiming_a_source_column():
    result = _run("explain_value", {"resource": PORTFOLIO_A.key,
                                    "loan_id": "LN000000",
                                    "canonical_field": "source_portfolio_id"})
    e = result.result["explanation"]
    assert e["origin"] == ORIGIN_RUN_METADATA
    assert e["source_field"] is None


# --------------------------------------------------------------------------- #
# Bulk is the primitive
# --------------------------------------------------------------------------- #
def test_a_bulk_request_resolves_the_frame_once():
    resolver = _CountingResolver(_frame())
    requests = [{"loan_id": f"LN{i:06d}", "canonical_field": "current_principal_balance"}
                for i in range(0, 60, 2)]
    result = _run("explain_values", {"resource": PORTFOLIO_A.key, "requests": requests},
                  deps=_deps(resolver=resolver))
    assert result.status == STATUS_SUCCESS
    assert result.result["explained_count"] == 30
    assert resolver.calls == 1


def test_bulk_answers_come_back_in_the_order_requested():
    requests = [{"loan_id": "LN000004", "canonical_field": "current_loan_to_value"},
                {"loan_id": "LN000000", "canonical_field": "current_principal_balance"}]
    result = _run("explain_values", {"resource": PORTFOLIO_A.key, "requests": requests})
    got = [(e["loan_id"], e["canonical_field"]) for e in result.result["explanations"]]
    assert got == [("LN000004", "current_loan_to_value"),
                   ("LN000000", "current_principal_balance")]


def test_explain_value_is_a_wrapper_with_no_lookup_of_its_own():
    import inspect

    from trakt_tools.handlers import provenance as mod

    source = inspect.getsource(mod.explain_value)
    assert "explain_values(" in source
    for forbidden in ("resolve_governed_frame", "_index_for", "isin"):
        assert forbidden not in source


def test_thirty_values_in_bulk_beat_thirty_single_calls(capsys):
    """Recorded, not merely asserted — the ratio is the argument for the shape."""
    df = _frame()
    context = _context()
    pairs = [(f"LN{i:06d}", "current_principal_balance") for i in range(0, 60, 2)]

    def singles():
        for loan_id, field in pairs:
            execute_governed_tool(
                "explain_value",
                {"resource": PORTFOLIO_A.key, "loan_id": loan_id,
                 "canonical_field": field},
                context, dependencies=_deps(resolver=_CountingResolver(df)))

    def bulk():
        execute_governed_tool(
            "explain_values",
            {"resource": PORTFOLIO_A.key,
             "requests": [{"loan_id": l, "canonical_field": f} for l, f in pairs]},
            context, dependencies=_deps(resolver=_CountingResolver(df)))

    singles(); bulk()                                   # warm
    t = time.perf_counter(); singles(); singles_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter(); bulk(); bulk_ms = (time.perf_counter() - t) * 1000

    with capsys.disabled():
        print(f"\n    30 values via explain_value x30  : {singles_ms:8.1f} ms"
              f"\n    30 values via explain_values x1  : {bulk_ms:8.1f} ms"
              f"\n    improvement                      : {singles_ms / max(bulk_ms, 1e-6):8.1f}x")

    assert bulk_ms < singles_ms / 2


def test_the_bulk_request_is_bounded():
    from trakt_tools.handlers.provenance import MAX_REQUESTS

    requests = [{"loan_id": "LN000000", "canonical_field": "current_principal_balance"}
                for _ in range(MAX_REQUESTS + 1)]
    result = _run("explain_values", {"resource": PORTFOLIO_A.key, "requests": requests})
    assert result.error_code == ErrorCode.TOOL_INPUT_INVALID


# --------------------------------------------------------------------------- #
# Exactness: snapshot and tenant binding
# --------------------------------------------------------------------------- #
def test_an_index_from_another_snapshot_is_refused_not_served():
    """Provenance from the wrong snapshot is worse than none."""
    result = _run("explain_value",
                  {"resource": PORTFOLIO_A.key, "loan_id": "LN000000",
                   "canonical_field": "current_principal_balance"},
                  deps=_deps(index=_index(snapshot_id="snap_SOMETHING_ELSE")))
    assert result.status in (STATUS_BLOCKED, STATUS_ERROR)
    assert result.error_code == ErrorCode.SNAPSHOT_STALE


def test_an_index_from_another_tenant_is_refused():
    result = _run("explain_value",
                  {"resource": PORTFOLIO_A.key, "loan_id": "LN000000",
                   "canonical_field": "current_principal_balance"},
                  deps=_deps(index=_index(tenant="OTHER_CLIENT")))
    assert result.status == STATUS_BLOCKED
    assert result.error_code == ErrorCode.TENANT_MISMATCH


def test_a_snapshot_with_no_index_says_so_rather_than_inventing_a_source():
    result = _run("explain_values",
                  {"resource": PORTFOLIO_A.key,
                   "requests": [{"loan_id": "LN000000",
                                 "canonical_field": "current_principal_balance"}]},
                  deps=ToolDependencies(
                      datasets=_Datasets(), runtime_mode="test",
                      catalogue=_catalogue(), output_root=None,
                      loan_frame_resolver=_CountingResolver(_frame()),
                      lineage_index=None, lineage_index_path="/no/such/index.json"))
    assert result.status == STATUS_SUCCESS
    payload = result.result
    assert payload["lineage_available"] is False
    assert any("no lineage index" in w for w in payload["warnings"])
    e = payload["explanations"][0]
    assert e["value"] == 100_000.0            # the value is still governed
    assert e["source_field"] is None          # its origin is unknown, not guessed


# --------------------------------------------------------------------------- #
# Missing / unresolvable requests
# --------------------------------------------------------------------------- #
def test_an_unknown_loan_or_field_is_reported_with_a_reason():
    requests = [
        {"loan_id": "LN999999", "canonical_field": "current_principal_balance"},
        {"loan_id": "LN000000", "canonical_field": "not_on_this_tape"},
        {"loan_id": "", "canonical_field": "current_principal_balance"},
        {"loan_id": "LN000000", "canonical_field": "current_principal_balance"},
    ]
    result = _run("explain_values", {"resource": PORTFOLIO_A.key, "requests": requests})
    payload = result.result
    assert payload["explained_count"] == 1
    reasons = {u["reason"] for u in payload["unresolved"]}
    assert any("no such loan" in r for r in reasons)
    assert any("does not carry" in r for r in reasons)
    assert any("both required" in r for r in reasons)


def test_the_authorisation_boundary_applies_to_provenance_too():
    result = _run("explain_value", {"resource": PORTFOLIO_B.key,
                                    "loan_id": "LN000001",
                                    "canonical_field": "current_principal_balance"})
    assert result.status == STATUS_BLOCKED
    assert result.error_code == ErrorCode.RESOURCE_NOT_AUTHORISED


# --------------------------------------------------------------------------- #
# The index: compact, built at ingestion, cached on content
# --------------------------------------------------------------------------- #
def test_the_index_is_per_field_not_per_cell():
    """The design guarantee: a 130-column tape has a 130-entry index whatever
    the row count."""
    index = _index()
    assert len(index.entries) == 3
    assert set(index.fields()) == {"current_principal_balance",
                                   "current_loan_to_value", "source_portfolio_id"}


def test_the_index_round_trips_through_json(tmp_path):
    path = _index().write(tmp_path / "lineage_index.json")
    loaded = load_lineage_index(path)
    assert loaded is not None
    assert loaded.tenant_id == TENANT
    assert loaded.snapshot_id == SNAPSHOT_ID
    entry = loaded.get("current_principal_balance")
    assert entry.source_field == "OUT_PRIN"
    assert entry.validation_rules == ("BAL001", "BAL101")


def test_the_index_is_read_once_per_content_version(tmp_path):
    path = _index().write(tmp_path / "lineage_index.json")
    load_lineage_index(path)
    reads = []
    real = Path.read_text

    def counting(self, *a, **k):
        if self.name == "lineage_index.json":
            reads.append(1)
        return real(self, *a, **k)

    Path.read_text = counting
    try:
        for _ in range(20):
            load_lineage_index(path)
    finally:
        Path.read_text = real
    assert reads == []


def test_an_absent_index_is_none_not_an_error(tmp_path):
    assert load_lineage_index(tmp_path / "missing.json") is None


def test_the_index_is_built_from_the_lineage_the_pipeline_already_emits():
    """Compacted from field_lineage.json rather than instrumented separately, so
    it cannot drift from the lineage it summarises."""
    field_lineage = {
        "generated_at_utc": "2026-07-31T00:00:00Z",
        "fields": {
            "current_principal_balance": {
                "field": "current_principal_balance",
                "field_status": {"present_in_run": True},
                "source": {"raw_columns": ["OUT_PRIN"],
                           "methods": [{"raw": "OUT_PRIN", "method": "alias",
                                        "confidence": 1.0}]},
                "quality": {"error_count": 0},
                "declared_spec": {"format": "decimal"},
            },
            "current_loan_to_value": {
                "field": "current_loan_to_value",
                "field_status": {"present_in_run": True},
                "source": None,
                "quality": {"error_count": 2},
                "declared_spec": {"format": "decimal"},
            },
            "source_portfolio_id": {
                "field": "source_portfolio_id",
                "field_status": {"present_in_run": True},
                "source": None, "quality": {"error_count": 0},
            },
            "never_supplied": {
                "field": "never_supplied",
                "field_status": {"present_in_run": False},
                "source": None, "quality": None,
            },
        },
    }
    index = build_index_from_field_lineage(
        field_lineage, tenant_id=TENANT, snapshot_id=SNAPSHOT_ID,
        source_dataset="july_tape.csv",
        derivations={"current_loan_to_value": {"rule": "RESOLVE_CURRENT_LOAN_TO_VALUE"}},
        registry_fields={"current_loan_to_value": {"unit": "percentage_points"}},
        run_metadata_fields=["source_portfolio_id"])

    mapped = index.get("current_principal_balance")
    assert mapped.origin == ORIGIN_MAPPED
    assert mapped.source_field == "OUT_PRIN"
    assert mapped.mapping_method == "alias"
    assert mapped.validation_status == "passed"

    derived = index.get("current_loan_to_value")
    assert derived.origin == ORIGIN_DERIVED
    assert derived.derivation_rule == "RESOLVE_CURRENT_LOAN_TO_VALUE"
    assert derived.validation_status == "failed"
    assert derived.unit == "percentage_points"

    assert index.get("source_portfolio_id").origin == ORIGIN_RUN_METADATA

    absent = index.get("never_supplied")
    assert absent.present_in_run is False
    assert absent.validation_status == "not_validated"


def test_the_ingestion_step_writes_the_index(tmp_path):
    """End to end through the real lineage_tracker CLI."""
    import subprocess

    canonical = tmp_path / "canonical_typed.csv"
    pd.DataFrame({"loan_identifier": ["LN1", "LN2"],
                  "current_principal_balance": [1.0, 2.0],
                  "source_portfolio_id": ["direct_001", "direct_001"]}
                 ).to_csv(canonical, index=False)
    header_map = tmp_path / "header_map.json"
    header_map.write_text(json.dumps({"mappings": [
        {"raw_header": "OUT_PRIN", "canonical_field": "current_principal_balance",
         "mapping_method": "alias", "confidence": 1.0}]}))

    out = subprocess.run(
        [sys.executable, "engine/gate_2_transform/lineage_tracker.py",
         "--canonical", str(canonical), "--outdir", str(tmp_path / "out"),
         "--header-map", str(header_map), "--tenant-id", TENANT,
         "--snapshot-id", SNAPSHOT_ID, "--quiet"],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr

    index_path = tmp_path / "out" / "lineage_index.json"
    assert index_path.exists(), "ingestion did not write the lineage index"
    index = load_lineage_index(index_path)
    assert index.tenant_id == TENANT and index.snapshot_id == SNAPSHOT_ID
    entry = index.get("current_principal_balance")
    assert entry is not None and entry.source_field == "OUT_PRIN"
    # The full document is still written, unchanged.
    assert (tmp_path / "out" / "field_lineage.json").exists()
