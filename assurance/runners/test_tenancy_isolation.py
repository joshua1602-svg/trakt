"""Adversarial tenant-isolation tests for the governed MI query capability.

These exercise the full governed capability (`execute_governed_mi_query`) — not
just the `trakt_core.tenancy` unit — because the confirmed cross-tenant read was
in the *consumer* layer: `mi_service`/`datasets` re-parsed the caller's raw
`portfolio_id` and used its first segment as a client directory for storage
resolution, so an open-namespace deployment (no `config/tenancy.yaml`) served
another client's tape from a shared onboarding root.

Defect: ASSURE-001 (Critical). Fix: storage resolution is bound to the
authorised tenant (`datasets._resolve_query_frame(..., tenant_id=...)`).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

BASE_TAPE = Path("synthetic_demo/output/multibook/platform_2026-06-30_canonical_typed.csv")
FOREIGN_MARKER = 999_999.0


def _lay_down_roots(tmp_path: Path) -> Path:
    """A shared onboarding root holding two clients' central tapes.

    ``client_002`` carries a distinctive balance so a leak is unmistakable.
    """
    root = tmp_path / "onboarding_root"
    if root.exists():
        shutil.rmtree(root)

    foreign = pd.read_csv(BASE_TAPE).head(10).copy()
    foreign["current_outstanding_balance"] = FOREIGN_MARKER
    dst = root / "client_002" / "2026_06" / "output" / "central"
    dst.mkdir(parents=True)
    foreign.to_csv(dst / "18_central_lender_tape.csv", index=False)

    own = pd.read_csv(BASE_TAPE).copy()
    d1 = root / "client_001" / "2026_06" / "output" / "central"
    d1.mkdir(parents=True)
    own.to_csv(d1 / "18_central_lender_tape.csv", index=False)
    return root


@pytest.fixture()
def governed(tmp_path, monkeypatch):
    root = _lay_down_roots(tmp_path)
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    monkeypatch.setenv("MI_AGENT_PLATFORM_CANONICAL", str(BASE_TAPE))
    monkeypatch.setenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", str(root))
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", "client_001")
    monkeypatch.setenv("MI_AGENT_LLM_ENABLED", "0")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from mi_agent_api import data_source, datasets
    data_source.reset_cache()
    datasets._CLIENT_CURRENCY_CACHE.clear()

    from trakt_core.context import ExecutionContext
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    def ask(portfolio_id=None, tenant="client_001"):
        ctx = ExecutionContext.for_internal(tenant)
        return execute_governed_mi_query(
            MiQueryRequest(question="What is the total outstanding balance?",
                           portfolio_id=portfolio_id), ctx)

    yield ask
    data_source.reset_cache()
    datasets._CLIENT_CURRENCY_CACHE.clear()


def _balance(result) -> float:
    arts = (result.result or {}).get("artifacts") or []
    assert arts, "expected at least one artifact"
    return (arts[0].get("reconciliation") or {}).get("total_balance")


def test_foreign_client_portfolio_id_cannot_read_another_tenants_tape(governed):
    """A caller authenticated as client_001 must never receive client_002's
    tape by naming it in ``portfolio_id`` — even under an open namespace."""
    result = governed(portfolio_id="client_002/2026_06")
    balance = _balance(result)
    # The foreign tape totals 10 * 999,999 = 9,999,990. The client's own book is
    # an order of magnitude larger. A leak would surface the foreign figure.
    assert balance != pytest.approx(10 * FOREIGN_MARKER), (
        "cross-tenant read: client_001 received client_002's tape")
    assert balance > 10 * FOREIGN_MARKER, "expected the tenant's own (larger) book"


def test_legitimate_own_run_still_resolves(governed):
    result = governed(portfolio_id="client_001/2026_06")
    assert result.status == "success"
    assert _balance(result) > 10 * FOREIGN_MARKER


def test_default_portfolio_resolves_active_dataset(governed):
    result = governed(portfolio_id=None)
    assert result.status == "success"
    assert _balance(result) > 10 * FOREIGN_MARKER


def test_envelope_never_stamps_foreign_balance(governed):
    """Defence in depth: whatever the caller names, the balance returned is the
    tenant's own, so the governance envelope cannot certify a foreign figure."""
    for pid in ("client_002/2026_06", "client_002", "client_999/2026_06"):
        assert _balance(governed(portfolio_id=pid)) != pytest.approx(10 * FOREIGN_MARKER)
