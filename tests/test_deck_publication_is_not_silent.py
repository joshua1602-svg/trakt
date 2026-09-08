#!/usr/bin/env python3
"""tests/test_deck_publication_is_not_silent.py

A deck that was generated but never published must not report success.

The PPTX button in the dashboard downloads whatever the deck store holds. When
generation succeeds and publication does not, the store still holds the
PREVIOUS deck — so a green "completed" sends the reader to a stale pack and
tells them it is new. That is the defect these pin, in both of the places it
was possible:

  * the write gate kept its own list of connection variables while the READ
    path resolved blob storage from an Azure marker alone. An App Service with
    only ``AzureWebJobsStorage`` read the store perfectly and published nothing
    into it, forever;
  * the artifact status was set to "available" BEFORE the publish and never
    downgraded, so nothing downstream could tell the two cases apart.
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


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Neither the override nor any connection variable, unless a test sets one."""
    for var in ("TRAKT_INVESTOR_PPTX_PERSIST", "AZURE_STORAGE_CONNECTION_STRING",
                "TRAKT_BLOB_CONNECTION", "AzureWebJobsStorage",
                "WEBSITE_SITE_NAME", "WEBSITE_INSTANCE_ID",
                "TRAKT_STORAGE_BACKEND"):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
# The write gate asks the same question the read path asks.
# --------------------------------------------------------------------------- #

def test_an_app_service_with_only_the_functions_connection_still_publishes(monkeypatch):
    """THE BUG. Reads worked, writes did not, and nothing said so.

    ``decide_backend`` resolves blob storage from the Azure marker and falls
    back to ``AzureWebJobsStorage`` for the connection string. The write gate
    did not look at either, so this deployment served the old deck and
    published nothing into the store it was reading from.
    """
    monkeypatch.setenv("WEBSITE_SITE_NAME", "trakt-mi-api")
    monkeypatch.setenv("AzureWebJobsStorage", "UseDevelopmentStorage=true")

    assert ST.decide_backend()["backend"] == "azure_blob", "the read path"
    assert PS.pptx_persist_enabled() is True, "the write path must agree"


def test_the_two_paths_agree_on_every_configuration(monkeypatch):
    """Whatever resolves a durable store for reads must enable writes.

    Stated as the invariant rather than as a list of cases, because the defect
    was precisely that the two lists had drifted apart.
    """
    cases = [
        {},
        {"TRAKT_BLOB_CONNECTION": "conn"},
        {"WEBSITE_SITE_NAME": "site", "AzureWebJobsStorage": "conn"},
        {"WEBSITE_INSTANCE_ID": "id", "AzureWebJobsStorage": "conn"},
        {"WEBSITE_SITE_NAME": "site"},
    ]
    for env in cases:
        for var in ("TRAKT_BLOB_CONNECTION", "WEBSITE_SITE_NAME",
                    "WEBSITE_INSTANCE_ID", "AzureWebJobsStorage"):
            monkeypatch.delenv(var, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        durable = ST.decide_backend()["backend"] == "azure_blob"
        assert PS.pptx_persist_enabled() is durable, env


def test_local_development_still_does_not_publish():
    """No Azure, no connection: the scratch deck is enough and nothing uploads."""
    assert PS.pptx_persist_enabled() is False
    assert PS.publication_expected() is False


def test_the_explicit_override_still_wins_both_ways(monkeypatch):
    monkeypatch.setenv("TRAKT_INVESTOR_PPTX_PERSIST", "true")
    assert PS.pptx_persist_enabled() is True
    monkeypatch.setenv("TRAKT_INVESTOR_PPTX_PERSIST", "false")
    monkeypatch.setenv("TRAKT_BLOB_CONNECTION", "conn")
    assert PS.pptx_persist_enabled() is False


def test_an_unreadable_storage_configuration_does_not_fail_the_run(monkeypatch):
    """A publication hint must never be the thing that breaks a generated deck."""
    def _boom():
        raise RuntimeError("storage configuration is unreadable")
    monkeypatch.setattr(ST, "decide_backend", _boom)
    assert PS.publication_expected() is False


# --------------------------------------------------------------------------- #
# A non-publish is reported, not swallowed.
# --------------------------------------------------------------------------- #

def test_a_generated_deck_that_did_not_publish_says_so(tmp_path, monkeypatch):
    """The artifact must carry the fact, or nothing downstream can report it.

    Driven through the real ``persist_investor_deck``: with persistence off it
    returns None, which is exactly the state that used to be indistinguishable
    from a successful upload.
    """
    monkeypatch.setenv("TRAKT_INVESTOR_PPTX_PERSIST", "false")
    deck = tmp_path / "investor_pack.pptx"
    deck.write_bytes(b"PK\x03\x04" + b"0" * 64)
    assert PS.persist_investor_deck(deck, client_id="c", period="2026-06") is None


def test_the_job_reports_blocked_rather_than_completed():
    """THE READER-FACING HALF.

    "Completed" on a generation that changed nothing sends someone to a button
    that hands back the previous pack, and tells them it is new.
    """
    from mi_agent_api.deck_generation import outcome_of, STATE_BLOCKED

    state, message, code = outcome_of(
        {"status": "available", "publication_skipped": "not published"})
    assert state == STATE_BLOCKED
    assert "still serve the previously published deck" in message
    assert code is None, "this is not a generation failure"


def test_a_successful_publication_still_completes():
    """The fix must not turn every healthy generation amber."""
    from mi_agent_api.deck_generation import outcome_of, STATE_COMPLETED

    state, message, _code = outcome_of(
        {"status": "available", "published": {"latest_uri": "blob://x"}})
    assert state == STATE_COMPLETED
    assert message is None


def test_a_withheld_pack_keeps_its_own_distinct_reason():
    """Failing publication GATES and failing to reach the STORE are different
    facts, and a reader acts on them differently — one is the pack's problem,
    the other is the deployment's."""
    from mi_agent_api.deck_generation import outcome_of, STATE_BLOCKED

    gated = outcome_of({"status": "generated_not_published"})
    stored = outcome_of({"status": "available", "publication_skipped": "x"})
    assert gated[0] == stored[0] == STATE_BLOCKED
    assert gated[1] != stored[1]
    assert "publication checks" in gated[1]
    assert "storage configuration" in stored[1]


def test_a_generation_that_produced_nothing_is_still_a_failure():
    from mi_agent_api.deck_generation import outcome_of, STATE_FAILED

    state, _message, code = outcome_of({"status": "missing"})
    assert state == STATE_FAILED
    assert code is not None
