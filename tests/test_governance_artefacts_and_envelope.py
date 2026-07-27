#!/usr/bin/env python3
"""tests/test_governance_artefacts_and_envelope.py

Three governance properties that the MI-question tests do not reach:

1. **Artefact access is tenant-safe.** A deck is selected by the authenticated
   tenant, never by a caller-supplied ``client_id``. Before this change the
   React route passed ``client_id`` straight into the deck resolver, so any
   authenticated user could fetch another tenant's investor pack out of the
   shared deck container.
2. **The governed envelope is complete and serialisable.** Success and blocked
   results both carry the identifiers, snapshot metadata, policy state and typed
   error a machine caller needs — and the whole thing round-trips through JSON
   without an interface-specific type.
3. **The audit event is emitted and safe.** It carries the identifiers and omits
   the answer body, signed URLs and storage paths.

Plus the enterprise-agent boundary proof: the illustrative adapter at the bottom
is the whole of what a future client-agent endpoint has to write.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core.audit import audit_event_from_result, emit_audit_event
from trakt_core.context import (
    ACTOR_SERVICE,
    CHANNEL_ENTERPRISE_AGENT,
    DEFAULT_MI_SCOPES,
    ExecutionContext,
)
from trakt_core.envelope import STATUS_BLOCKED, STATUS_SUCCESS
from trakt_core.errors import ErrorCode
from trakt_core.tenancy import TenantRecord, TenantRegistry

DECK_BYTES = b"PK\x03\x04 fake pptx"


def _context(tenant="ERE", **kw):
    return ExecutionContext(tenant_id=tenant, actor_id="user-1", **kw)


def _registry(*tenant_ids, configured=True, portfolios=None):
    records = {
        t: TenantRecord(tenant_id=t, default_portfolio_id=t,
                        portfolio_ids=frozenset(portfolios or ()))
        for t in tenant_ids
    }
    return TenantRegistry(records, configured=configured)


@pytest.fixture
def deck_root(tmp_path, monkeypatch):
    """Two tenants' decks in one shared store — the cross-tenant setup."""
    for tenant in ("ERE", "other_tenant"):
        latest = tmp_path / tenant / "latest"
        latest.mkdir(parents=True)
        (latest / "investor_pack.pptx").write_bytes(DECK_BYTES)
        (latest / "latest_investor_pack.json").write_text(json.dumps({
            "reporting_period": "2026-01", "generated_at": "2026-02-01T00:00:00Z",
            "checksum": f"sha256:{tenant}", "size_bytes": len(DECK_BYTES),
        }))
    monkeypatch.setenv("MI_AGENT_DECK_ROOT", str(tmp_path))
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", "ERE")
    return tmp_path


# --------------------------------------------------------------------------- #
# 22-26 — artefact access
# --------------------------------------------------------------------------- #
def test_tenant_can_retrieve_its_own_deck(deck_root):
    from mi_agent_api import artefacts
    from mi_agent_api.dependencies import build_dependencies

    result = artefacts.get_investor_pack(
        _context(), dependencies=build_dependencies(tenant_registry=_registry("ERE")))
    assert result.status == STATUS_SUCCESS
    assert result.result.tenant_id == "ERE"
    assert result.result.local_path.read_bytes() == DECK_BYTES


def test_cross_tenant_deck_access_is_refused(deck_root):
    """The core fix: naming another tenant does not fetch its pack."""
    from mi_agent_api import artefacts
    from mi_agent_api.dependencies import build_dependencies

    deps = build_dependencies(tenant_registry=_registry("ERE", "other_tenant"))
    result = artefacts.get_investor_pack(
        _context(), portfolio_id="other_tenant", dependencies=deps)
    assert result.status == STATUS_BLOCKED
    assert result.error.code == ErrorCode.TENANT_MISMATCH
    assert result.result is None


def test_conflicting_client_id_is_refused_not_honoured(deck_root):
    """The deprecated ``client_id`` parameter cannot select the artefact."""
    from mi_agent_api import artefacts
    from mi_agent_api.dependencies import build_dependencies

    result = artefacts.get_investor_pack(
        _context(), requested_client_id="other_tenant",
        dependencies=build_dependencies(tenant_registry=_registry("ERE")))
    assert result.status == STATUS_BLOCKED
    assert result.error.code == ErrorCode.TENANT_MISMATCH


def test_matching_client_id_still_works_for_compatibility(deck_root):
    from mi_agent_api import artefacts
    from mi_agent_api.dependencies import build_dependencies

    result = artefacts.get_investor_pack(
        _context(), requested_client_id="ERE",
        dependencies=build_dependencies(tenant_registry=_registry("ERE")))
    assert result.status == STATUS_SUCCESS


def test_react_route_cannot_be_pointed_at_another_tenant(deck_root, monkeypatch):
    """End to end through the HTTP adapter, not just the capability."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "false")
    from mi_agent_api.app import app

    client = TestClient(app)
    mine = client.get("/mi/decks/download")
    assert mine.status_code == 200
    assert mine.content == DECK_BYTES

    theirs = client.get("/mi/decks/download", params={"client_id": "other_tenant"})
    assert theirs.status_code == 403
    assert theirs.json()["errorCode"] == ErrorCode.TENANT_MISMATCH


def test_artefact_not_found_is_distinct_from_unauthorised(deck_root):
    from mi_agent_api import artefacts
    from mi_agent_api.dependencies import build_dependencies

    result = artefacts.get_investor_pack(
        _context(), period="1999-01",
        dependencies=build_dependencies(tenant_registry=_registry("ERE")))
    assert result.error.code == ErrorCode.ARTEFACT_NOT_FOUND
    assert result.error.http_status == 404


def test_period_cannot_traverse_the_artefact_store(deck_root):
    from mi_agent_api import artefacts
    from mi_agent_api.dependencies import build_dependencies

    result = artefacts.get_investor_pack(
        _context(), period="../../other_tenant/latest",
        dependencies=build_dependencies(tenant_registry=_registry("ERE")))
    assert result.error.code == ErrorCode.INVALID_INPUT


def test_metadata_never_exposes_the_server_side_path(deck_root):
    from mi_agent_api import artefacts
    from mi_agent_api.dependencies import build_dependencies

    result = artefacts.get_investor_pack(
        _context(), dependencies=build_dependencies(tenant_registry=_registry("ERE")))
    blob = json.dumps(result.result.to_metadata())
    assert str(deck_root) not in blob
    assert "local_path" not in blob


# --------------------------------------------------------------------------- #
# 27-31 — the governed envelope
# --------------------------------------------------------------------------- #
def _mi_result(monkeypatch, tmp_path, *, tenant="ERE", portfolio=None):
    from mi_agent_api import mi_service
    from mi_agent_api.dependencies import build_dependencies

    demo = sorted(_REPO_ROOT.glob("synthetic_demo/**/*canonical_typed.csv"))[0]
    monkeypatch.setenv("MI_AGENT_DATA_CSV", str(demo))
    monkeypatch.setenv("MI_AGENT_DATA_CACHE_TTL", "0")
    monkeypatch.setenv("MI_AGENT_LLM_PARSER", "off")
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    from mi_agent_api import data_source
    data_source.reset_cache()
    return mi_service.execute_governed_mi_query(
        mi_service.MiQueryRequest(question="What is the total current balance?",
                                  portfolio_id=portfolio),
        _context(tenant),
        build_dependencies(tenant_registry=_registry(tenant), mode="test"))


def test_success_envelope_carries_the_required_metadata(monkeypatch, tmp_path):
    result = _mi_result(monkeypatch, tmp_path)
    assert result.capability == "mi.question.answer"
    assert result.schema_version
    assert result.status == STATUS_SUCCESS
    assert result.request_id and result.tenant_id == "ERE"
    assert result.policy.tenant_authorised and result.policy.capability_allowed
    assert result.policy.data_approved
    assert result.audit is not None and result.audit.duration_ms is not None


def test_snapshot_metadata_identifies_the_dataset(monkeypatch, tmp_path):
    result = _mi_result(monkeypatch, tmp_path)
    snap = result.snapshot
    assert snap is not None
    assert snap.content_hash and snap.content_hash.startswith("sha256:")
    assert snap.snapshot_id
    assert snap.row_count and snap.row_count > 0
    assert snap.source_base and snap.source_kind


def test_blocked_envelope_carries_a_stable_error_and_policy(monkeypatch, tmp_path):
    from mi_agent_api import mi_service
    from mi_agent_api.dependencies import build_dependencies

    result = mi_service.execute_governed_mi_query(
        mi_service.MiQueryRequest(question="total balance",
                                  portfolio_id="other_tenant"),
        _context("ERE"),
        build_dependencies(tenant_registry=_registry("ERE", "other_tenant"),
                           mode="test"))
    assert result.status == STATUS_BLOCKED
    assert result.error.code == ErrorCode.TENANT_MISMATCH
    assert result.error.retryable is False
    assert result.policy.data_approved is False
    # A refusal still renders in the channels: the analytical payload shape holds.
    assert result.result["ok"] is False and result.result["answer"]


def test_envelope_is_json_serialisable_without_interface_types(monkeypatch, tmp_path):
    result = _mi_result(monkeypatch, tmp_path)
    blob = json.dumps(result.to_dict(), default=str)
    round_tripped = json.loads(blob)
    assert round_tripped["capability"] == "mi.question.answer"
    assert round_tripped["policy"]["data_approved"] is True
    assert "snapshot" in round_tripped and "provenance" in round_tripped


def test_react_payload_stays_backward_compatible(monkeypatch, tmp_path):
    """Every pre-existing key survives; governance is purely additive."""
    from mi_agent_api import presenters

    result = _mi_result(monkeypatch, tmp_path)
    payload = presenters.to_react_payload(result)
    for key in ("ok", "question", "answer", "interpreted", "spec", "validation",
                "artifacts", "warnings", "assumptions", "diagnostics",
                "sourceNotes", "metadata"):
        assert key in payload, f"React payload lost {key!r}"
    assert payload["governance"]["capability"] == "mi.question.answer"
    assert payload["metadata"]["snapshotId"] == result.snapshot.snapshot_id


# --------------------------------------------------------------------------- #
# 36-39 — errors and audit
# --------------------------------------------------------------------------- #
def test_error_codes_classify_consistently():
    """One table drives HTTP status, category and retryability for every channel."""
    from trakt_core.errors import TraktError, http_status_for

    assert http_status_for(ErrorCode.TENANT_MISMATCH) == 403
    assert http_status_for(ErrorCode.DATA_SOURCE_NOT_APPROVED) == 503
    assert http_status_for(ErrorCode.ARTEFACT_NOT_FOUND) == 404
    # Capability outcomes stay 200 with ok=false — the existing channel contract.
    assert http_status_for(ErrorCode.UNSUPPORTED_QUESTION) == 200

    assert TraktError(ErrorCode.STORAGE_UNAVAILABLE, "x").retryable is True
    assert TraktError(ErrorCode.PORTFOLIO_NOT_AUTHORISED, "x").retryable is False


def test_error_payload_leaks_nothing_sensitive():
    from trakt_core.errors import TraktError

    err = TraktError(ErrorCode.INTERNAL_ERROR, "Something failed.",
                     request_id="req_1", cause=ValueError("/secret/path/tape.csv"))
    blob = json.dumps(err.to_dict())
    assert "/secret/path" not in blob
    assert "Traceback" not in blob
    assert set(err.to_dict()) <= {"code", "message", "category", "retryable",
                                  "request_id", "details"}


def test_audit_event_carries_the_required_identifiers(monkeypatch, tmp_path):
    result = _mi_result(monkeypatch, tmp_path)
    event = audit_event_from_result(result)
    for key in ("capability", "request_id", "tenant_id", "actor_id", "actor_type",
                "channel", "outcome", "started_at", "duration_ms"):
        assert key in event, f"audit event missing {key!r}"
    assert event["outcome"] == STATUS_SUCCESS


def test_audit_event_omits_payloads_and_urls(monkeypatch, tmp_path):
    result = _mi_result(monkeypatch, tmp_path)
    event = audit_event_from_result(result)
    for forbidden in ("answer", "result", "rows", "artifacts", "token",
                      "downloadUrl", "signed_url", "path"):
        assert forbidden not in event
    blob = json.dumps(event, default=str)
    assert "synthetic_demo" not in blob or "source_kind" in event


def test_audit_emission_never_breaks_the_request(monkeypatch, tmp_path, caplog):
    result = _mi_result(monkeypatch, tmp_path)
    with caplog.at_level(logging.INFO, logger="trakt.audit"):
        event = emit_audit_event(result)
    assert event["request_id"] == result.request_id
    assert any("trakt.audit" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Enterprise-agent boundary proof
# --------------------------------------------------------------------------- #
def enterprise_agent_endpoint(payload, authenticated_service, dependencies):
    """The whole of a future client-agent adapter.

    It knows nothing about storage layout, manifest selection, synthetic-data
    rejection, tenant authorisation, metric calculation, provenance assembly or
    envelope construction — every one of those is behind the capability call.
    """
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    context = ExecutionContext(
        tenant_id=authenticated_service["tenant_id"],
        actor_id=authenticated_service["service_id"],
        actor_type=ACTOR_SERVICE,
        channel=CHANNEL_ENTERPRISE_AGENT,
        scopes=DEFAULT_MI_SCOPES,
        correlation_id=payload.get("correlation_id"),
    )
    request = MiQueryRequest(question=payload["question"],
                             portfolio_id=payload.get("portfolio_id"),
                             filters=payload.get("filters"))
    return execute_governed_mi_query(request, context, dependencies)


def test_enterprise_agent_adapter_is_transport_and_identity_only(monkeypatch, tmp_path):
    from mi_agent_api import data_source
    from mi_agent_api.dependencies import build_dependencies

    demo = sorted(_REPO_ROOT.glob("synthetic_demo/**/*canonical_typed.csv"))[0]
    monkeypatch.setenv("MI_AGENT_DATA_CSV", str(demo))
    monkeypatch.setenv("MI_AGENT_DATA_CACHE_TTL", "0")
    monkeypatch.setenv("MI_AGENT_LLM_PARSER", "off")
    data_source.reset_cache()

    result = enterprise_agent_endpoint(
        {"question": "What is the total current balance?",
         "correlation_id": "client-trace-42"},
        {"tenant_id": "ERE", "service_id": "client-agent-123"},
        build_dependencies(tenant_registry=_registry("ERE"), mode="test"))

    assert result.status == STATUS_SUCCESS
    assert result.audit.channel == CHANNEL_ENTERPRISE_AGENT
    assert result.audit.actor_type == ACTOR_SERVICE
    assert result.correlation_id == "client-trace-42"
    # The same governed facts a browser gets, with no browser involved.
    assert result.snapshot.content_hash and result.policy.data_approved


def test_enterprise_agent_adapter_inherits_every_control(monkeypatch, tmp_path):
    """The adapter wrote no governance code, yet gets all of it."""
    from mi_agent_api.dependencies import build_dependencies

    deps = build_dependencies(tenant_registry=_registry("ERE", "other_tenant"),
                              mode="test")
    blocked = enterprise_agent_endpoint(
        {"question": "total balance", "portfolio_id": "other_tenant"},
        {"tenant_id": "ERE", "service_id": "client-agent-123"}, deps)
    assert blocked.status == STATUS_BLOCKED
    assert blocked.error.code == ErrorCode.TENANT_MISMATCH

    # …and the production source-approval policy, without opting in.
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "production")
    from mi_agent_api import data_source
    data_source.reset_cache()
    refused = enterprise_agent_endpoint(
        {"question": "total balance"},
        {"tenant_id": "ERE", "service_id": "client-agent-123"},
        build_dependencies(tenant_registry=_registry("ERE"), mode="production"))
    assert refused.status == STATUS_BLOCKED
    assert refused.error.code in (ErrorCode.DATA_SOURCE_NOT_APPROVED,
                                 ErrorCode.DATA_SOURCE_UNAVAILABLE)
