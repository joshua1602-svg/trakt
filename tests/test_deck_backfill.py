#!/usr/bin/env python3
"""tests/test_deck_backfill.py

The explicit admin backfill that promotes already-generated investor decks
(``apps.blob_trigger_app.deck_backfill``) — the mechanism that makes an existing
run artifact such as::

    processed-v2/out/_blob_trigger/orun_ere_20260703T224438/reports/investor_pack.pptx

discoverable through the governed deck resolver (and therefore through Copilot).

Covered: dry run, successful promotion, duplicate no-op, malformed artifact
rejection, missing metadata handling, failed/incomplete runs never selected,
newest valid period wins, client filtering and the CLI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

DECK_NAME = "investor_pack.pptx"
POINTER_NAME = "latest_investor_pack.json"
PPTX_BYTES = b"PK\x03\x04" + b"historic investor pack" * 4


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A filesystem-backed processed container with an orchestration run tree."""
    root = tmp_path / "blobstore"
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(root))
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("TRAKT_BLOB_CONNECTION", raising=False)
    monkeypatch.delenv("TRAKT_RUN_OUTPUT_ROOT", raising=False)
    # Publication is OFF: the backfill must publish on its own explicit authority.
    monkeypatch.setenv("TRAKT_INVESTOR_PPTX_PERSIST", "false")
    (root / "processed-v2" / "out" / "_blob_trigger").mkdir(parents=True)
    return root


def _run_dir(store: Path, orun: str) -> Path:
    return store / "processed-v2" / "out" / "_blob_trigger" / orun


def _seed_run(store: Path, orun: str, *, client_id="ERE", status="done",
              period="2026-07-31", deck=PPTX_BYTES, artifact_status="available",
              blockers=None, manifest=True, artifact=True, run_id=None) -> Path:
    run = _run_dir(store, orun)
    (run / "reports").mkdir(parents=True, exist_ok=True)
    if deck is not None:
        (run / "reports" / DECK_NAME).write_bytes(deck)
    if manifest:
        data = {"run_id": run_id or f"run_{orun}", "client_id": client_id,
                "status": status, "blockers": blockers or []}
        if artifact:
            data["artifacts"] = {"investor_pack_pptx": {
                "type": "pptx", "status": artifact_status,
                "path": "reports/investor_pack.pptx",
                "client_id": client_id, "reporting_period": period}}
        (run / "run_state.json").write_text(json.dumps(data, indent=2))
    return run


def _decks(store: Path, client: str, period: str) -> Path:
    return store / "processed-v2" / "decks" / client / period


def _pointer(store: Path, client: str) -> dict:
    return json.loads((_decks(store, client, "latest") / POINTER_NAME)
                      .read_text(encoding="utf-8"))


def _backfill(**kw):
    from apps.blob_trigger_app.deck_backfill import run_backfill
    return run_backfill(**kw)


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #
def test_dry_run_reports_without_changing_anything(store):
    _seed_run(store, "orun_ere_20260703T224438")

    result = _backfill(dry_run=True)

    assert [r["action"] for r in result.records] == ["would_promote"]
    assert result.records[0]["client_id"] == "ERE"
    assert result.records[0]["reporting_period"] == "2026-07"
    assert result.records[0]["checksum"].startswith("sha256:")
    assert not (store / "processed-v2" / "decks").exists(), "dry run wrote something"


# --------------------------------------------------------------------------- #
# Promotion
# --------------------------------------------------------------------------- #
def test_successful_promotion_publishes_dated_latest_and_pointer(store):
    _seed_run(store, "orun_ere_20260703T224438", run_id="run_ere_202607")

    result = _backfill()

    assert [r["action"] for r in result.records] == ["promoted"]
    assert (_decks(store, "ERE", "2026-07") / DECK_NAME).read_bytes() == PPTX_BYTES
    assert (_decks(store, "ERE", "latest") / DECK_NAME).read_bytes() == PPTX_BYTES
    ptr = _pointer(store, "ERE")
    assert ptr["reporting_period"] == "2026-07"
    assert ptr["orchestration_run_id"] == "orun_ere_20260703T224438"
    assert ptr["source_run_id"] == "run_ere_202607"
    assert ptr["checksum"].startswith("sha256:")
    # The immutable run artifact is untouched.
    assert (_run_dir(store, "orun_ere_20260703T224438") / "reports" / DECK_NAME).exists()


def test_promoted_deck_is_discoverable_through_the_governed_resolver(store, monkeypatch):
    """The end of the chain: a backfilled deck resolves exactly like a deck the
    orchestration published, so the Copilot action finds it."""
    _seed_run(store, "orun_ere_20260703T224438")
    _backfill()

    monkeypatch.delenv("MI_AGENT_DECK_ROOT", raising=False)   # force blob mode
    from mi_agent_api import decks as decks_mod

    listing = decks_mod.list_decks("ERE")
    assert listing["available"] is True
    assert listing["latest"]["period"] == "2026-07"
    assert listing["latest"]["orchestrationRunId"] == "orun_ere_20260703T224438"
    resolved = decks_mod.resolve_deck_local("ERE", None)
    assert resolved is not None and resolved[0].exists()


def test_duplicate_backfill_is_a_no_op(store):
    _seed_run(store, "orun_ere_20260703T224438")
    first = _backfill()
    second = _backfill()

    assert [r["action"] for r in first.records] == ["promoted"]
    assert [r["action"] for r in second.records] == ["already_published"]


def test_newest_valid_period_wins_regardless_of_scan_order(store):
    _seed_run(store, "orun_z_older", period="2026-03-31", deck=PPTX_BYTES + b"MAR")
    _seed_run(store, "orun_a_newer", period="2026-07-31", deck=PPTX_BYTES + b"JUL")

    result = _backfill()

    assert {r["action"] for r in result.records} == {"promoted"}
    assert _pointer(store, "ERE")["reporting_period"] == "2026-07"
    assert (_decks(store, "ERE", "latest") / DECK_NAME).read_bytes() == PPTX_BYTES + b"JUL"
    # Both dated archives exist.
    assert (_decks(store, "ERE", "2026-03") / DECK_NAME).exists()
    assert (_decks(store, "ERE", "2026-07") / DECK_NAME).exists()


# --------------------------------------------------------------------------- #
# Governed selection — incomplete / failed / malformed runs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs,expected", [
    ({"status": "running"}, "not a completed run"),
    ({"status": "halted"}, "not a completed run"),
    ({"status": "failed"}, "not a completed run"),
    ({"blockers": ["gate 4 projection blocked"]}, "unresolved blockers"),
    ({"artifact_status": "failed"}, "artifact status"),
    ({"manifest": False}, "not a governed orchestration run"),
    ({"client_id": None}, "no client_id"),
])
def test_ineligible_runs_are_skipped_with_a_reason(store, kwargs, expected):
    _seed_run(store, "orun_bad", **kwargs)

    result = _backfill()

    assert len(result.records) == 1
    record = result.records[0]
    assert record["action"] == "skipped"
    assert expected in record["reason"]
    assert not (store / "processed-v2" / "decks").exists()


def test_malformed_artifact_is_rejected(store):
    _seed_run(store, "orun_bad_bytes", deck=b"this is not a pptx")

    result = _backfill()

    assert result.records[0]["action"] == "rejected"
    assert "invalid artifact" in result.records[0]["reason"]
    assert not (store / "processed-v2" / "decks").exists()


def test_missing_period_is_skipped_until_supplied_explicitly(store):
    _seed_run(store, "orun_no_period", period=None)

    skipped = _backfill()
    assert skipped.records[0]["action"] == "skipped"
    assert "no reporting period" in skipped.records[0]["reason"]

    promoted = _backfill(period="2026-05")
    assert promoted.records[0]["action"] == "promoted"
    assert _pointer(store, "ERE")["reporting_period"] == "2026-05"


def test_explicit_period_does_not_override_a_known_one(store):
    _seed_run(store, "orun_known", period="2026-07-31")

    result = _backfill(period="2020-01")

    assert result.records[0]["reporting_period"] == "2026-07"
    assert _pointer(store, "ERE")["reporting_period"] == "2026-07"


# --------------------------------------------------------------------------- #
# Scoping
# --------------------------------------------------------------------------- #
def test_client_filter_scopes_the_backfill(store):
    _seed_run(store, "orun_ere", client_id="ERE")
    _seed_run(store, "orun_other", client_id="OTHER")

    result = _backfill(client_id="ERE")

    assert [r["client_id"] for r in result.records] == ["ERE"]
    assert (_decks(store, "ERE", "latest") / DECK_NAME).exists()
    assert not (store / "processed-v2" / "decks" / "OTHER").exists()


def test_run_filter_scopes_the_backfill(store):
    _seed_run(store, "orun_one", period="2026-06-30")
    _seed_run(store, "orun_two", period="2026-07-31")

    result = _backfill(run_id="orun_one")

    assert [r["orchestration_run_id"] for r in result.records] == ["orun_one"]
    assert _pointer(store, "ERE")["reporting_period"] == "2026-06"


def test_empty_root_is_handled(store):
    result = _backfill()
    assert result.records == []
    assert result.to_dict()["scanned"] == 0


def test_run_output_root_is_configurable(store, monkeypatch, tmp_path):
    from apps.blob_trigger_app.deck_backfill import run_output_root
    from apps.blob_trigger_app.layout import Layout

    layout = Layout.from_env()
    assert run_output_root(layout).endswith("processed-v2/out/_blob_trigger")
    monkeypatch.setenv("TRAKT_RUN_OUTPUT_ROOT", "blob://processed-v2/elsewhere")
    assert run_output_root(layout) == "blob://processed-v2/elsewhere"
    assert run_output_root(layout, "blob://x/y/") == "blob://x/y"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_dry_run_reports_and_changes_nothing(store, capsys):
    from apps.blob_trigger_app.deck_backfill import main

    _seed_run(store, "orun_ere_20260703T224438")
    assert main(["--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "would promote 1" in out
    assert "orun_ere_20260703T224438" in out
    assert not (store / "processed-v2" / "decks").exists()


def test_cli_json_output_is_machine_readable(store, capsys):
    from apps.blob_trigger_app.deck_backfill import main

    _seed_run(store, "orun_ere_20260703T224438")
    assert main(["--json", "--client", "ERE"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["promoted"] == 1 and payload["skipped"] == 0
    assert payload["records"][0]["action"] == "promoted"
    # No storage credentials in the audit record.
    assert "accountkey" not in json.dumps(payload).lower()
