#!/usr/bin/env python3
"""mi_agent_api/tests/test_decks.py

Investor PPTX deck discovery + download (MI API surface, Task 1). Uses the local
deck-root mode (MI_AGENT_DECK_ROOT) so the tests need no blob backend. Verifies:
  * list_decks surfaces the latest pointer + dated periods, no raw paths;
  * resolve_deck_local resolves latest and a specific period;
  * the /mi/decks and /mi/decks/download endpoints behave (200 + 404 graceful);
  * a client with no decks yields a disabled/empty listing (never a 500).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
from fastapi.testclient import TestClient


DECK_NAME = "investor_pack.pptx"
POINTER_NAME = "latest_investor_pack.json"


def _write_deck(path: Path, content: bytes = b"PK\x03\x04 fake pptx") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.fixture()
def deck_root(tmp_path, monkeypatch):
    client = "client_001"
    root = tmp_path / "decks"
    _write_deck(root / client / "latest" / DECK_NAME)
    _write_deck(root / client / "2026-01" / DECK_NAME)
    _write_deck(root / client / "2025-12" / DECK_NAME)
    (root / client / "latest" / POINTER_NAME).write_text(json.dumps({
        "client_id": client, "reporting_period": "2026-01",
        "generated_at": "2026-01-12T09:00:00+00:00"}))
    monkeypatch.setenv("MI_AGENT_DECK_ROOT", str(root))
    return root


def test_list_decks_surfaces_latest_and_periods(deck_root):
    from mi_agent_api import decks as decks_mod
    out = decks_mod.list_decks("client_001")
    assert out["available"] is True
    assert out["latest"]["period"] == "2026-01"
    assert out["latest"]["generatedAt"].startswith("2026-01-12")
    periods = [d["period"] for d in out["decks"]]
    assert periods == ["2026-01", "2025-12"]  # newest first
    # No raw filesystem/blob path leaks into the payload.
    assert "investor_pack.pptx" not in json.dumps(out)


def test_list_decks_empty_for_unknown_client(deck_root):
    from mi_agent_api import decks as decks_mod
    out = decks_mod.list_decks("client_999")
    assert out["available"] is False
    assert out["latest"] is None
    assert out["decks"] == []


def test_resolve_latest_and_period(deck_root):
    from mi_agent_api import decks as decks_mod
    latest = decks_mod.resolve_deck_local("client_001", None)
    assert latest is not None
    path, name = latest
    assert path.exists() and name.endswith(".pptx")
    period = decks_mod.resolve_deck_local("client_001", "2025-12")
    assert period is not None and period[0].exists()
    assert decks_mod.resolve_deck_local("client_001", "1999-01") is None


def test_endpoints_download_and_404(deck_root, monkeypatch):
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "false")
    from mi_agent_api.app import app
    client = TestClient(app)

    listed = client.get("/mi/decks", params={"portfolioId": "client_001/mi_2026_01"})
    assert listed.status_code == 200
    assert listed.json()["available"] is True

    dl = client.get("/mi/decks/download", params={"portfolioId": "client_001/mi_2026_01"})
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument")

    missing = client.get("/mi/decks/download",
                         params={"client_id": "client_001", "period": "1999-01"})
    assert missing.status_code == 404
    assert missing.json()["ok"] is False


def test_persist_then_discover_roundtrip(tmp_path, monkeypatch):
    """persist_investor_deck (orchestration) → list_decks/resolve (MI API) via the
    filesystem-backed durable store, proving the publish→serve chain end-to-end."""
    monkeypatch.delenv("MI_AGENT_DECK_ROOT", raising=False)  # force blob mode
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blobstore"))
    monkeypatch.setenv("TRAKT_INVESTOR_PPTX_PERSIST", "true")

    from apps.blob_trigger_app import pptx_stage
    from mi_agent_api import decks as decks_mod

    deck = tmp_path / "run" / "reports" / DECK_NAME
    _write_deck(deck)
    published = pptx_stage.persist_investor_deck(
        deck, client_id="ERE", period="2026-01-31")
    assert published is not None and "latest_uri" in published

    out = decks_mod.list_decks("ERE")
    assert out["available"] is True
    assert out["latest"]["period"] == "2026-01-31"
    assert [d["period"] for d in out["decks"]] == ["2026-01"]

    resolved = decks_mod.resolve_deck_local("ERE", None)
    assert resolved is not None and resolved[0].exists()


def test_persist_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("TRAKT_INVESTOR_PPTX_PERSIST", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("TRAKT_BLOB_CONNECTION", raising=False)
    from apps.blob_trigger_app import pptx_stage
    deck = tmp_path / "reports" / DECK_NAME
    _write_deck(deck)
    assert pptx_stage.persist_investor_deck(deck, client_id="ERE") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
