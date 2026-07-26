#!/usr/bin/env python3
"""tests/test_deck_publication.py

The durable investor-deck publication contract
(``apps.blob_trigger_app.pptx_stage.persist_investor_deck``).

  * the immutable run artifact is retained (only ever read);
  * the dated deck is published to decks/{client}/{YYYY-MM}/investor_pack.pptx;
  * the latest deck is published to decks/{client}/latest/investor_pack.pptx;
  * the latest pointer is written with checksum, size, content type, run ids and
    generator version;
  * a retry is idempotent;
  * an OLDER reporting period never replaces a newer latest pointer;
  * an existing dated publication is not overwritten by a different deck;
  * an invalid or failed-run artifact is not published.

Runs against the filesystem-backed storage backend, so the publish -> discover
chain is exercised end to end without Azure.
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
PPTX_BYTES = b"PK\x03\x04" + b"fake pptx payload" * 8


@pytest.fixture()
def blob_store(tmp_path, monkeypatch):
    """A filesystem-backed durable store with publication forced on."""
    root = tmp_path / "blobstore"
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(root))
    monkeypatch.setenv("TRAKT_INVESTOR_PPTX_PERSIST", "true")
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("TRAKT_BLOB_CONNECTION", raising=False)
    return root


def _write_deck(path: Path, content: bytes = PPTX_BYTES) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _deck_dir(root: Path, client: str, period: str) -> Path:
    return root / "processed-v2" / "decks" / client / period


def _pointer(root: Path, client: str) -> dict:
    return json.loads((_deck_dir(root, client, "latest") / POINTER_NAME)
                      .read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Publication
# --------------------------------------------------------------------------- #
def test_run_artifact_is_retained_and_all_three_targets_are_written(blob_store, tmp_path):
    from apps.blob_trigger_app import pptx_stage

    run_deck = _write_deck(tmp_path / "orun_x" / "reports" / DECK_NAME)
    published = pptx_stage.persist_investor_deck(
        run_deck, client_id="ERE", period="2026-06-30",
        orchestration_run_id="orun_x", source_run_id="run_ere_202606")

    assert published is not None
    # 1. the immutable run artifact is untouched
    assert run_deck.exists() and run_deck.read_bytes() == PPTX_BYTES
    # 2. dated copy
    dated = _deck_dir(blob_store, "ERE", "2026-06") / DECK_NAME
    assert dated.exists() and dated.read_bytes() == PPTX_BYTES
    # 3. latest copy
    latest = _deck_dir(blob_store, "ERE", "latest") / DECK_NAME
    assert latest.exists() and latest.read_bytes() == PPTX_BYTES
    # 4. latest pointer
    assert (_deck_dir(blob_store, "ERE", "latest") / POINTER_NAME).exists()


def test_pointer_carries_the_full_governed_metadata(blob_store, tmp_path):
    import hashlib

    from apps.blob_trigger_app import pptx_stage

    run_deck = _write_deck(tmp_path / "orun_x" / "reports" / DECK_NAME)
    pptx_stage.persist_investor_deck(
        run_deck, client_id="ERE", period="2026-06-30",
        orchestration_run_id="orun_ere_20260703T224438",
        source_run_id="run_ere_202606")

    ptr = _pointer(blob_store, "ERE")
    assert ptr["client_id"] == "ERE"
    assert ptr["reporting_period"] == "2026-06"
    assert ptr["as_of_date"] == "2026-06-30"
    assert ptr["orchestration_run_id"] == "orun_ere_20260703T224438"
    assert ptr["source_run_id"] == "run_ere_202606"
    assert ptr["checksum"] == "sha256:" + hashlib.sha256(PPTX_BYTES).hexdigest()
    assert ptr["size_bytes"] == len(PPTX_BYTES)
    assert ptr["content_type"] == pptx_stage.PPTX_CONTENT_TYPE
    assert ptr["generator"] == "mi_agent_pptx"
    assert ptr["generator_version"]
    assert ptr["generated_at"]
    # No credentials or connection strings leak into the pointer.
    blob = json.dumps(ptr).lower()
    assert "accountkey" not in blob and "sharedaccesssignature" not in blob


def test_publication_is_idempotent_on_retry(blob_store, tmp_path):
    from apps.blob_trigger_app import pptx_stage

    run_deck = _write_deck(tmp_path / "orun_x" / "reports" / DECK_NAME)
    kw = dict(client_id="ERE", period="2026-06-30")
    first = pptx_stage.persist_investor_deck(run_deck, **kw)
    second = pptx_stage.persist_investor_deck(run_deck, **kw)

    assert first is not None and second is not None
    assert second["period_skipped"]           # dated copy is write-once
    assert second["latest_uri"] == first["latest_uri"]
    assert second["checksum"] == first["checksum"]
    dated_dir = _deck_dir(blob_store, "ERE", "2026-06")
    assert sorted(p.name for p in dated_dir.iterdir()) == [DECK_NAME]


def test_existing_dated_publication_is_not_overwritten(blob_store, tmp_path):
    """The dated copy is the immutable record for a reporting period — a second,
    different deck for the same period must not silently replace it."""
    from apps.blob_trigger_app import pptx_stage

    original = _write_deck(tmp_path / "run_a" / "reports" / DECK_NAME)
    pptx_stage.persist_investor_deck(original, client_id="ERE", period="2026-06-30")

    different = _write_deck(tmp_path / "run_b" / "reports" / DECK_NAME,
                            content=PPTX_BYTES + b"DIFFERENT")
    published = pptx_stage.persist_investor_deck(
        different, client_id="ERE", period="2026-06-30")

    assert published["period_skipped"]
    dated = _deck_dir(blob_store, "ERE", "2026-06") / DECK_NAME
    assert dated.read_bytes() == PPTX_BYTES, "the dated publication was overwritten"


def test_older_period_does_not_replace_a_newer_latest(blob_store, tmp_path):
    from apps.blob_trigger_app import pptx_stage

    june = _write_deck(tmp_path / "run_june" / "reports" / DECK_NAME)
    pptx_stage.persist_investor_deck(june, client_id="ERE", period="2026-06-30")

    march = _write_deck(tmp_path / "run_march" / "reports" / DECK_NAME,
                        content=PPTX_BYTES + b"MARCH")
    published = pptx_stage.persist_investor_deck(
        march, client_id="ERE", period="2026-03-31")

    assert published["latest_skipped"]
    # latest still points at June...
    assert _pointer(blob_store, "ERE")["reporting_period"] == "2026-06"
    latest = _deck_dir(blob_store, "ERE", "latest") / DECK_NAME
    assert latest.read_bytes() == PPTX_BYTES
    # ...but the older period is still archived under its own date.
    assert (_deck_dir(blob_store, "ERE", "2026-03") / DECK_NAME).exists()


def test_newer_period_does_replace_latest(blob_store, tmp_path):
    from apps.blob_trigger_app import pptx_stage

    march = _write_deck(tmp_path / "run_march" / "reports" / DECK_NAME)
    pptx_stage.persist_investor_deck(march, client_id="ERE", period="2026-03-31")

    june = _write_deck(tmp_path / "run_june" / "reports" / DECK_NAME,
                       content=PPTX_BYTES + b"JUNE")
    pptx_stage.persist_investor_deck(june, client_id="ERE", period="2026-06-30")

    assert _pointer(blob_store, "ERE")["reporting_period"] == "2026-06"
    latest = _deck_dir(blob_store, "ERE", "latest") / DECK_NAME
    assert latest.read_bytes() == PPTX_BYTES + b"JUNE"


def test_clients_are_isolated(blob_store, tmp_path):
    from apps.blob_trigger_app import pptx_stage

    deck_a = _write_deck(tmp_path / "a" / "reports" / DECK_NAME, b"PK\x03\x04A")
    deck_b = _write_deck(tmp_path / "b" / "reports" / DECK_NAME, b"PK\x03\x04B")
    pptx_stage.persist_investor_deck(deck_a, client_id="ERE", period="2026-06-30")
    pptx_stage.persist_investor_deck(deck_b, client_id="OTHER", period="2026-06-30")

    assert (_deck_dir(blob_store, "ERE", "latest") / DECK_NAME).read_bytes() == b"PK\x03\x04A"
    assert (_deck_dir(blob_store, "OTHER", "latest") / DECK_NAME).read_bytes() == b"PK\x03\x04B"


# --------------------------------------------------------------------------- #
# Validation / refusal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("content,reason", [
    (b"", "empty"),
    (b"not a pptx at all", "magic"),
])
def test_invalid_artifact_is_not_published(blob_store, tmp_path, content, reason):
    from apps.blob_trigger_app import pptx_stage

    bad = _write_deck(tmp_path / "run" / "reports" / DECK_NAME, content=content)
    assert pptx_stage.persist_investor_deck(
        bad, client_id="ERE", period="2026-06-30") is None, reason
    assert not (_deck_dir(blob_store, "ERE", "latest") / DECK_NAME).exists()


def test_missing_artifact_is_not_published(blob_store, tmp_path):
    from apps.blob_trigger_app import pptx_stage

    assert pptx_stage.persist_investor_deck(
        tmp_path / "nope" / DECK_NAME, client_id="ERE", period="2026-06") is None


def test_failed_run_never_reaches_publication(blob_store, tmp_path, monkeypatch):
    """A generator failure records a failed artifact and publishes nothing."""
    from apps.blob_trigger_app import pptx_stage

    run_dir = tmp_path / "orun_failed"
    monkeypatch.setattr(pptx_stage, "_invoke_generator",
                        lambda _argv: (_ for _ in ()).throw(RuntimeError("boom")))
    artifact = pptx_stage.generate_investor_pptx(
        run_dir, client_name="ERE", as_of_date="2026-06-30")

    assert artifact["status"] == "failed"
    assert "published" not in artifact
    assert not (_deck_dir(blob_store, "ERE", "latest") / DECK_NAME).exists()


def test_period_key_normalisation():
    from apps.blob_trigger_app.pptx_stage import period_key

    assert period_key("2026-06-30") == "2026-06"
    assert period_key("2026-06") == "2026-06"
    assert period_key("202606") == "2026-06"
    assert period_key("2026/06/30") == "2026-06"
    assert period_key("") is None
    assert period_key("latest") is None
    assert period_key("2026-13") is None      # not a month


def test_validate_deck_file_reasons(tmp_path):
    from apps.blob_trigger_app.pptx_stage import validate_deck_file

    assert validate_deck_file(tmp_path / "missing.pptx") == "file does not exist"
    empty = _write_deck(tmp_path / "empty.pptx", b"")
    assert validate_deck_file(empty) == "file is empty"
    bad = _write_deck(tmp_path / "bad.pptx", b"nope")
    assert validate_deck_file(bad) == "not a PPTX (OOXML) file"
    good = _write_deck(tmp_path / "good.pptx", PPTX_BYTES)
    assert validate_deck_file(good) is None
