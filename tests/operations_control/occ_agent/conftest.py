"""Fixtures for the OCC Agent test suite.

Everything runs file-backed in a tmp directory, with no Azure credentials and no
live containers in reach. Two properties of this setup are themselves part of
what is being tested:

* the synthetic store is pointed at its own container and its own sandbox root,
  so a test that accidentally wrote to the live container would fail;
* no ``TRAKT_STORAGE_BACKEND=blob`` and no credentials are set anywhere, so the
  whole feature is exercised in exactly the configuration
  :func:`operations_control.occ_agent.policy.assert_no_live_credentials_required`
  promises works.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.blob_trigger_app.storage import Storage
from operations_control.occ_agent.service import OccAgentService
from operations_control.occ_agent.store import SyntheticCaseStore

SYNTHETIC_CONTAINER = "operations-control-synthetic"
LIVE_CONTAINER = "operations-control"

TENANT_A = "client_a"
TENANT_B = "client_b"
ACTOR = "Alice"

OPERATORS = {
    "tok-a": {"name": "Alice", "clients": [TENANT_A]},
    "tok-b": {"name": "Bob", "clients": [TENANT_B]},
    "tok-all": {"name": "Root", "clients": ["*"]},
}


@pytest.fixture()
def agent_env(tmp_path, monkeypatch):
    """A credential-free, file-backed environment for the whole feature."""
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blob"))
    monkeypatch.setenv("TRAKT_OCC_AGENT_SANDBOX_ROOT", str(tmp_path / "sandbox"))
    monkeypatch.setenv("TRAKT_OCC_AGENT_SYNTHETIC_CONTAINER", SYNTHETIC_CONTAINER)
    monkeypatch.setenv("OCC_AGENT_SYNTHETIC_ENABLED", "true")
    monkeypatch.setenv("TRAKT_OPS_OPERATORS", json.dumps(OPERATORS))
    # Deliberately absent: every Azure credential variable. Nothing in the
    # feature may need one.
    for name in ("AZURE_STORAGE_CONNECTION_STRING", "AZURE_CLIENT_ID",
                 "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID",
                 "AZURE_STORAGE_ACCOUNT"):
        monkeypatch.delenv(name, raising=False)
    return {"tmp": tmp_path, "blob_root": tmp_path / "blob",
            "sandbox": tmp_path / "sandbox"}


@pytest.fixture()
def storage(agent_env) -> Storage:
    return Storage(agent_env["blob_root"])


@pytest.fixture()
def synthetic_store(storage, agent_env) -> SyntheticCaseStore:
    return SyntheticCaseStore(storage, container=SYNTHETIC_CONTAINER,
                              sandbox=agent_env["sandbox"])


@pytest.fixture()
def service(storage, agent_env) -> OccAgentService:
    return OccAgentService(storage, container=SYNTHETIC_CONTAINER,
                           sandbox=agent_env["sandbox"])


@pytest.fixture()
def blob_root(agent_env) -> Path:
    return agent_env["blob_root"]


def live_container_paths(blob_root: Path) -> list:
    """Every file under the LIVE operations container, if any exists."""
    live = Path(blob_root) / LIVE_CONTAINER
    if not live.exists():
        return []
    return sorted(p for p in live.rglob("*") if p.is_file())
