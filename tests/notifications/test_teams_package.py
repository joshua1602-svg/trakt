"""The Teams app package and the bot registration endpoint.

The package tests exist because losing a capability during an edit is silent:
the manifest still validates and still installs, and the loss only shows up
when a client's Copilot queries stop working or every proactive send fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

sys.path.insert(0, str(_REPO / "deploy" / "copilot-agent"))
import package_agent  # noqa: E402

AGENT_DIR = _REPO / "deploy" / "copilot-agent"


@pytest.fixture
def manifest() -> dict:
    return json.loads((AGENT_DIR / "manifest.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# The existing Copilot capability must survive
# --------------------------------------------------------------------------- #
def test_the_declarative_agent_is_still_declared(manifest):
    agents = manifest["copilotAgents"]["declarativeAgents"]
    assert agents[0]["file"] == "declarativeAgent.json"
    assert agents[0]["id"] == "traktDeclarativeAgent"


def test_the_copilot_actions_are_unchanged():
    plugin = json.loads((AGENT_DIR / "ai-plugin.json").read_text(encoding="utf-8"))
    assert [f["name"] for f in plugin["functions"]] == ["askTraktMi",
                                                        "getArtifact"]
    assert plugin["runtimes"][0]["auth"]["type"] == "OAuthPluginVault"


def test_the_app_identity_and_branding_are_unchanged(manifest):
    assert manifest["id"] == "89c9db43-8ae0-45f6-800f-df9e8fb20dda"
    assert manifest["name"]["short"] == "Trakt"
    assert manifest["accentColor"] == "#1F3A5F"


# --------------------------------------------------------------------------- #
# The new bot capability
# --------------------------------------------------------------------------- #
def test_a_bot_is_declared_in_the_same_package(manifest):
    """One app, two capabilities — not a second Teams application."""
    assert len(manifest["bots"]) == 1
    assert "copilotAgents" in manifest


def test_the_bot_is_personal_scope_and_notification_only(manifest):
    bot = manifest["bots"][0]
    assert bot["scopes"] == ["personal"]
    assert bot["isNotificationOnly"] is True
    for unsupported in ("team", "groupChat"):
        assert unsupported not in bot["scopes"]


def test_the_manifest_schema_supports_both_capabilities(manifest):
    # v1.19 carries `bots` and `copilotAgents` side by side.
    assert manifest["manifestVersion"] == "1.19"
    assert "v1.19" in manifest["$schema"]


# --------------------------------------------------------------------------- #
# Packaging guards
# --------------------------------------------------------------------------- #
def test_a_release_build_refuses_an_unresolved_bot_app_id(manifest):
    with pytest.raises(SystemExit) as excinfo:
        package_agent._validate_manifest(manifest, require_resolved=True)
    assert "TEAMS_BOT_APP_ID" in str(excinfo.value)


def test_packaging_refuses_a_manifest_that_lost_the_declarative_agent(manifest):
    manifest.pop("copilotAgents")
    with pytest.raises(SystemExit) as excinfo:
        package_agent._validate_manifest(manifest, require_resolved=False)
    assert "declarative agent" in str(excinfo.value)


def test_packaging_refuses_a_manifest_with_no_bot(manifest):
    manifest.pop("bots")
    with pytest.raises(SystemExit) as excinfo:
        package_agent._validate_manifest(manifest, require_resolved=False)
    assert "no bot" in str(excinfo.value)


def test_packaging_refuses_a_group_or_channel_scope(manifest):
    manifest["bots"][0]["scopes"] = ["personal", "team"]
    with pytest.raises(SystemExit) as excinfo:
        package_agent._validate_manifest(manifest, require_resolved=False)
    assert "does not implement" in str(excinfo.value)


def test_the_package_builds_and_carries_every_artefact(tmp_path):
    zip_path = package_agent.build(tmp_path, require_resolved=False)
    import zipfile
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert {"manifest.json", "declarativeAgent.json", "ai-plugin.json",
            "trakt-copilot-openapi.yaml", "color.png",
            "outline.png"} <= names


# --------------------------------------------------------------------------- #
# The bot messaging endpoint
# --------------------------------------------------------------------------- #
def _activity(**overrides) -> dict:
    activity = {
        "type": "conversationUpdate",
        "id": "activity-1",
        "channelId": "msteams",
        "serviceUrl": "https://smba.trafficmanager.net/emea/",
        "from": {"id": "29:abc", "aadObjectId": "user-object-id",
                 "name": "Pilot User"},
        "recipient": {"id": "28:bot"},
        "conversation": {"id": "a:1conversation",
                         "conversationType": "personal"},
        "channelData": {"tenant": {"id": "ms-tenant-1"}},
    }
    activity.update(overrides)
    return activity


@pytest.fixture
def bot_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAKT_TEAMS_BOT_ENABLED", "1")
    monkeypatch.setenv("TRAKT_TEAMS_TRAKT_TENANT", "ERE")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path))
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    return tmp_path


def test_registration_captures_the_conversation_without_authorising(bot_env):
    from mi_agent_api import teams_bot
    result = teams_bot.handle_activity(_activity())
    assert result["registered"] is True
    assert result["notifications_enabled"] is False
    assert result["portfolio_contexts"] == []
    assert "an operator must map this recipient" in result["next_step"]


def test_registration_never_returns_the_conversation_reference(bot_env):
    from mi_agent_api import teams_bot
    blob = json.dumps(teams_bot.handle_activity(_activity()))
    assert "serviceUrl" not in blob
    assert "smba.trafficmanager.net" not in blob


def test_a_group_chat_activity_is_not_stored_as_a_destination(bot_env):
    from mi_agent_api import teams_bot
    result = teams_bot.handle_activity(_activity(
        conversation={"id": "19:group", "conversationType": "groupChat"}))
    assert result["registered"] is False
    assert "personal" in result["reason"]


def test_an_activity_without_a_directory_identity_is_refused(bot_env):
    from mi_agent_api import teams_bot
    result = teams_bot.handle_activity(_activity(channelData={}, conversation={
        "id": "a:1", "conversationType": "personal"}))
    assert result["registered"] is False
    assert "directory identity" in result["reason"]


def test_removal_disables_delivery_and_keeps_the_record(bot_env):
    from mi_agent_api import teams_bot
    from trakt_notifications.recipients import RecipientStore
    from apps.blob_trigger_app.storage import open_storage

    teams_bot.handle_activity(_activity())
    store = RecipientStore(open_storage())
    rid = store.list("ERE")[0].recipient_id
    store.authorise("ERE", rid, portfolio_contexts=["total"], actor="op")

    result = teams_bot.handle_activity(_activity(
        type="installationUpdate", action="remove"))
    assert result["registered"] is False
    record = store.load("ERE", rid)
    assert record.notifications_enabled is False
    assert record.installation_status == "removed"


def test_no_configured_tenant_means_no_registration(bot_env, monkeypatch):
    from mi_agent_api import teams_bot
    monkeypatch.delenv("TRAKT_TEAMS_TRAKT_TENANT")
    result = teams_bot.handle_activity(_activity())
    assert result["registered"] is False
    assert "no Trakt tenant is configured" in result["reason"]


def test_the_endpoint_fails_closed_when_auth_is_unconfigured(monkeypatch):
    from fastapi import HTTPException
    from mi_agent_api import teams_bot

    monkeypatch.delenv("TRAKT_TEAMS_BOT_APP_ID", raising=False)
    monkeypatch.setenv("TRAKT_TEAMS_BOT_AUTH_MODE", "botframework")
    with pytest.raises(HTTPException) as excinfo:
        teams_bot.validate_activity_token("Bearer anything")
    assert excinfo.value.status_code == 503


def test_a_missing_bearer_token_is_rejected(monkeypatch):
    from fastapi import HTTPException
    from mi_agent_api import teams_bot

    monkeypatch.setenv("TRAKT_TEAMS_BOT_APP_ID", "bot-app-id")
    monkeypatch.setenv("TRAKT_TEAMS_BOT_AUTH_MODE", "botframework")
    with pytest.raises(HTTPException) as excinfo:
        teams_bot.validate_activity_token(None)
    assert excinfo.value.status_code == 401


def test_the_router_is_not_mounted_unless_switched_on(monkeypatch):
    from mi_agent_api import teams_bot
    monkeypatch.delenv("TRAKT_TEAMS_BOT_ENABLED", raising=False)
    assert teams_bot.enabled() is False
    monkeypatch.setenv("TRAKT_TEAMS_BOT_ENABLED", "1")
    assert teams_bot.enabled() is True


def test_the_bot_prefix_bypasses_the_easy_auth_guard_only_for_itself():
    from mi_agent_api import auth
    assert auth.TEAMS_BOT_PATH_PREFIX == "/v1/teams"
    assert not "/mi/pipeline".startswith(auth.TEAMS_BOT_PATH_PREFIX)
