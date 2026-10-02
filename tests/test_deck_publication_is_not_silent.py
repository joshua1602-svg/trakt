#!/usr/bin/env python3
"""tests/test_deck_publication_is_not_silent.py

A deck that was generated but never published must not report success, and a
deck that WAS published must be the one that downloads.

The dashboard's "Investor deck" menu downloads whatever the deck store holds.
Three faults could each leave it serving a stale pack while "Generate a new
pack" reported a fresh one:

  * the write gate kept its own list of connection variables while the READ
    path resolved blob storage from an Azure marker alone — so an App Service
    configured as the blob runbook documents, with only ``AzureWebJobsStorage``,
    read the store perfectly and published nothing into it;
  * the artifact status was set to "available" BEFORE the publish and never
    downgraded, so nothing downstream could tell an upload from a no-op;
  * the download route preferred a "local mirror" on any storage that happened
    to have a ``_local_path``, which the blob backend inherits — so a stray
    local tree shadowed the blob.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from apps.blob_trigger_app import pptx_stage as PS       # noqa: E402
from apps.blob_trigger_app import storage as ST          # noqa: E402

_ENV = ("TRAKT_INVESTOR_PPTX_PERSIST", "AZURE_STORAGE_CONNECTION_STRING",
        "TRAKT_BLOB_CONNECTION", "AzureWebJobsStorage",
        "WEBSITE_SITE_NAME", "WEBSITE_INSTANCE_ID", "TRAKT_STORAGE_BACKEND")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _ENV:
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
# The write gate asks the same question the read path asks.
# --------------------------------------------------------------------------- #

def test_an_app_service_with_only_the_functions_connection_publishes(monkeypatch):
    """THE STALE-DECK DEFECT. Reads worked, writes did not, nothing said so."""
    monkeypatch.setenv("WEBSITE_SITE_NAME", "trakt-mi-api")
    monkeypatch.setenv("AzureWebJobsStorage", "UseDevelopmentStorage=true")

    assert ST.decide_backend()["backend"] == "azure_blob", "the read path"
    assert PS.pptx_persist_enabled() is True, "the write path must agree"


def test_the_two_paths_agree_on_every_configuration(monkeypatch):
    """Stated as the invariant, because the defect WAS two lists drifting."""
    cases = [
        {},
        {"TRAKT_BLOB_CONNECTION": "conn"},
        {"WEBSITE_SITE_NAME": "site", "AzureWebJobsStorage": "conn"},
        {"WEBSITE_INSTANCE_ID": "id", "AzureWebJobsStorage": "conn"},
        {"WEBSITE_SITE_NAME": "site"},
        {"TRAKT_STORAGE_BACKEND": "file", "WEBSITE_SITE_NAME": "site"},
    ]
    for env in cases:
        for var in _ENV:
            monkeypatch.delenv(var, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        durable = ST.decide_backend()["backend"] == "azure_blob"
        assert PS.pptx_persist_enabled() is durable, env


def test_local_development_still_does_not_publish():
    assert PS.pptx_persist_enabled() is False
    assert PS.publication_expected() is False


def test_the_explicit_override_still_wins_both_ways(monkeypatch):
    monkeypatch.setenv("TRAKT_INVESTOR_PPTX_PERSIST", "true")
    assert PS.pptx_persist_enabled() is True
    monkeypatch.setenv("TRAKT_INVESTOR_PPTX_PERSIST", "false")
    monkeypatch.setenv("TRAKT_BLOB_CONNECTION", "conn")
    assert PS.pptx_persist_enabled() is False


def test_an_unreadable_storage_configuration_does_not_fail_the_run(monkeypatch):
    def _boom():
        raise RuntimeError("storage configuration is unreadable")
    monkeypatch.setattr(ST, "decide_backend", _boom)
    assert PS.publication_expected() is False


# --------------------------------------------------------------------------- #
# A non-publish is reported, not swallowed.
# --------------------------------------------------------------------------- #

def test_persistence_off_returns_nothing_which_used_to_read_as_success(tmp_path,
                                                                     monkeypatch):
    monkeypatch.setenv("TRAKT_INVESTOR_PPTX_PERSIST", "false")
    deck = tmp_path / "investor_pack.pptx"
    deck.write_bytes(b"PK\x03\x04" + b"0" * 64)
    assert PS.persist_investor_deck(deck, client_id="c", period="2026-06") is None


def test_the_stage_records_a_skipped_publication(monkeypatch):
    """Read from the source, because the branch sits inside a function that
    needs a generated deck and a run directory to reach."""
    src = (_ROOT / "apps" / "blob_trigger_app" / "pptx_stage.py").read_text()
    assert 'artifact["publication_skipped"]' in src
    assert "elif publication_expected():" in src


def test_the_job_reports_blocked_rather_than_completed():
    """THE READER-FACING HALF — and the dashboard already renders BLOCKED as
    "Pack withheld" with this message, so no frontend change is needed."""
    from mi_agent_api.deck_generation import STATE_BLOCKED, outcome_of

    state, message, code = outcome_of(
        {"status": "available", "publication_skipped": "not published"})
    assert state == STATE_BLOCKED
    assert "still serve the previously published deck" in message
    assert code is None, "this is not a generation failure"


def test_a_successful_publication_still_completes():
    from mi_agent_api.deck_generation import STATE_COMPLETED, outcome_of

    state, message, _ = outcome_of(
        {"status": "available", "published": {"latest_uri": "blob://x"}})
    assert state == STATE_COMPLETED and message is None


def test_a_withheld_pack_and_an_unpublished_pack_say_different_things():
    """One is the pack's problem, the other the deployment's."""
    from mi_agent_api.deck_generation import STATE_BLOCKED, outcome_of

    gated = outcome_of({"status": "generated_not_published"})
    stored = outcome_of({"status": "available", "publication_skipped": "x"})
    assert gated[0] == stored[0] == STATE_BLOCKED
    assert "publication checks" in gated[1]
    assert "storage configuration" in stored[1]


def test_a_generation_that_produced_nothing_is_still_a_failure():
    from mi_agent_api.deck_generation import STATE_FAILED, outcome_of

    state, _message, code = outcome_of({"status": "missing"})
    assert state == STATE_FAILED and code is not None


# --------------------------------------------------------------------------- #
# The deck that downloads is the deck that was published.
# --------------------------------------------------------------------------- #

def test_blob_storage_is_not_treated_as_a_local_mirror():
    """The blob backend extends the filesystem class, so ``isinstance`` alone
    would say yes for both — which is how a stray local tree shadowed it."""
    from mi_agent_api.decks import _is_filesystem_storage

    assert issubclass(ST.BlobStorage, ST.Storage), "premise of the defect"
    blob = ST.BlobStorage.__new__(ST.BlobStorage)
    assert _is_filesystem_storage(blob) is False


def test_filesystem_storage_still_serves_its_own_files(tmp_path):
    from mi_agent_api.decks import _is_filesystem_storage

    assert _is_filesystem_storage(ST.Storage(local_root=tmp_path)) is True


def test_anything_else_is_never_a_local_mirror():
    from mi_agent_api.decks import _is_filesystem_storage

    assert _is_filesystem_storage(object()) is False
