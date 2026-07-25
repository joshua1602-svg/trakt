#!/usr/bin/env python3
"""mi_agent_api/tests/test_copilot_package.py

Structural validation of the Microsoft 365 Copilot agent package
(deploy/copilot-agent/): the templates are valid, the config-driven builder
produces a complete client-specific zip with NO placeholder or unresolved
template variable remaining, the app id is deterministic per client, the
OpenAPI document stays in lock-step with the implemented FastAPI routes, and
ONLY the three intended actions are exposed to Copilot.

Microsoft's official validation (Agents Toolkit `atk validate` + the published
declarative-agent v1.7 / plugin v2.4 JSON schemas) is run out-of-band — see
docs/copilot_client_deployment_runbook.md; these tests are the in-repo gate.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import yaml

PKG = _REPO_ROOT / "deploy" / "copilot-agent"
SAMPLE_CONFIG = PKG / "client-config.sample.json"

EXPECTED_OPERATIONS = {
    ("post", "/v1/copilot/mi/query"): "askTraktMi",
    ("get", "/v1/copilot/artifacts/latest/investor-deck"): "getLatestInvestorDeck",
    ("get", "/v1/copilot/artifacts/latest/canonical-tape"): "getLatestCanonicalTape",
}

# The scan the distributable must survive — kept deliberately independent of the
# builder's own scan so a regression in one is caught by the other.
PLACEHOLDER_PATTERNS = [
    re.compile(r"\{\{[^}]*\}\}"),
    re.compile(r"\$\{\{[^}]*\}\}"),
    re.compile(r"REPLACE-WITH", re.IGNORECASE),
    re.compile(r"TRAKT-COPILOT-APP-ID"),
    re.compile(r"<[a-z0-9-]*(guid|app-id|tenant|secret|placeholder)[a-z0-9->]*>",
               re.IGNORECASE),
]


def _builder():
    sys.path.insert(0, str(PKG))
    try:
        import package_agent
        return package_agent
    finally:
        sys.path.remove(str(PKG))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One sample-config build shared by the read-only assertions."""
    out = tmp_path_factory.mktemp("copilot_pkg")
    zip_path = _builder().build(SAMPLE_CONFIG, out)
    files = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            files[name] = zf.read(name)
    return {"zip_path": zip_path, "files": files,
            "config": json.loads(SAMPLE_CONFIG.read_text())}


def _load_json(name: str) -> dict:
    return json.loads((PKG / name).read_text(encoding="utf-8"))


def _load_spec_template() -> dict:
    # Template tokens are quoted strings, so the template itself parses as YAML.
    return yaml.safe_load((PKG / "trakt-copilot-openapi.yaml").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Built package: completeness + no placeholders
# --------------------------------------------------------------------------- #
def test_built_package_contains_all_required_files(built):
    assert {"manifest.json", "declarativeAgent.json", "ai-plugin.json",
            "trakt-copilot-openapi.yaml", "color.png", "outline.png"} <= set(built["files"])
    assert built["files"]["color.png"].startswith(b"\x89PNG")
    assert built["files"]["outline.png"].startswith(b"\x89PNG")


def test_no_placeholder_survives_in_distributable(built):
    """FAILS if any placeholder or unresolved template variable remains in ANY
    packaged file — the distributability guarantee."""
    for name, data in built["files"].items():
        if name.endswith(".png"):
            continue
        text = data.decode("utf-8")
        for pattern in PLACEHOLDER_PATTERNS:
            hit = pattern.search(text)
            assert hit is None, f"{name} still contains placeholder {hit.group(0)!r}"


def test_builder_refuses_unresolved_tokens(tmp_path, monkeypatch):
    """A template token the config does not resolve must abort the build."""
    pa = _builder()
    # A config that omits nothing (valid) but a template that grew a new token.
    poisoned = tmp_path / "templates"
    poisoned.mkdir()
    for name in pa.TEMPLATE_FILES:
        text = (PKG / name).read_text(encoding="utf-8")
        if name == "manifest.json":
            text = text.replace('"Trakt"', '"Trakt {{NEW_UNRESOLVED_TOKEN}}"', 1)
        (poisoned / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(pa, "HERE", poisoned)
    with pytest.raises(SystemExit, match="NOT DISTRIBUTABLE"):
        pa.build(SAMPLE_CONFIG, tmp_path / "dist")


def test_builder_rejects_incomplete_config(tmp_path):
    pa = _builder()
    cfg = json.loads(SAMPLE_CONFIG.read_text())
    del cfg["sso_auth_config_id"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="sso_auth_config_id"):
        pa.build(bad, tmp_path / "dist")


def test_app_guid_is_deterministic_per_client(tmp_path, built):
    """Rebuilding the same client updates the SAME app id; a different client
    gets a different id (so re-issues update rather than duplicate)."""
    pa = _builder()
    manifest = json.loads(built["files"]["manifest.json"])
    zip2 = pa.build(SAMPLE_CONFIG, tmp_path / "again")
    with zipfile.ZipFile(zip2) as zf:
        again = json.loads(zf.read("manifest.json"))
    assert again["id"] == manifest["id"]

    cfg = json.loads(SAMPLE_CONFIG.read_text())
    cfg["client_id"] = "otherclient"
    other_cfg = tmp_path / "other.json"
    other_cfg.write_text(json.dumps(cfg))
    zip3 = pa.build(other_cfg, tmp_path / "other")
    with zipfile.ZipFile(zip3) as zf:
        other = json.loads(zf.read("manifest.json"))
    assert other["id"] != manifest["id"]


# --------------------------------------------------------------------------- #
# Built package: substitution correctness + internal consistency
# --------------------------------------------------------------------------- #
def test_substitutions_and_cross_consistency(built):
    cfg = built["config"]
    manifest = json.loads(built["files"]["manifest.json"])
    plugin = json.loads(built["files"]["ai-plugin.json"])
    spec = yaml.safe_load(built["files"]["trakt-copilot-openapi.yaml"])

    host = cfg["api_base_url"].split("//", 1)[1].rstrip("/")
    assert manifest["validDomains"] == [host]
    assert manifest["version"] == cfg["package_version"]
    assert spec["servers"][0]["url"] == cfg["api_base_url"].rstrip("/")

    # Entra SSO wiring: the plugin carries the auth-config id; the OpenAPI
    # security scheme scopes reference the Application ID URI.
    auth = plugin["runtimes"][0]["auth"]
    assert auth == {"type": "OAuthPluginVault",
                    "reference_id": cfg["sso_auth_config_id"]}
    scopes = spec["components"]["securitySchemes"]["entraBearer"]["flows"][
        "authorizationCode"]["scopes"]
    assert list(scopes) == [f"{cfg['app_id_uri']}/{cfg['scope_name']}"]


def test_schema_versions_are_current(built):
    da = json.loads(built["files"]["declarativeAgent.json"])
    plugin = json.loads(built["files"]["ai-plugin.json"])
    assert da["version"] == "v1.7"
    assert da["$schema"].endswith("/declarative-agent/v1.7/schema.json")
    assert plugin["schema_version"] == "v2.4"
    assert plugin["$schema"].endswith("/plugin/v2.4/schema.json")


# --------------------------------------------------------------------------- #
# Agent surface: exactly the three actions
# --------------------------------------------------------------------------- #
def test_declarative_agent_structurally_valid(built):
    da = json.loads(built["files"]["declarativeAgent.json"])
    assert da["name"] == "Trakt"
    assert "never invent" in da["instructions"].lower()
    assert len(da["conversation_starters"]) == 4
    assert da["actions"] == [{"id": "traktActions", "file": "ai-plugin.json"}]


def test_plugin_exposes_exactly_three_functions(built):
    plugin = json.loads(built["files"]["ai-plugin.json"])
    names = [f["name"] for f in plugin["functions"]]
    assert names == ["askTraktMi", "getLatestInvestorDeck", "getLatestCanonicalTape"]
    runtimes = plugin["runtimes"]
    assert len(runtimes) == 1
    assert runtimes[0]["spec"]["url"] == "trakt-copilot-openapi.yaml"
    assert sorted(runtimes[0]["run_for_functions"]) == sorted(names)
    assert runtimes[0]["auth"]["type"] == "OAuthPluginVault"  # Entra SSO auth config


def test_openapi_covers_exactly_the_three_actions(built):
    spec = yaml.safe_load(built["files"]["trakt-copilot-openapi.yaml"])
    ops = {}
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            ops[(method, path)] = op.get("operationId")
    assert ops == EXPECTED_OPERATIONS
    assert not any(p.startswith("/mi/") or p == "/health" for _m, p in ops)
    assert spec["security"] == [{"entraBearer": []}]


def test_openapi_operation_ids_match_the_implemented_routes():
    from mi_agent_api.app import app

    generated = app.openapi()
    implemented = {}
    for path, methods in generated["paths"].items():
        if not path.startswith("/v1/copilot"):
            continue
        for method, op in methods.items():
            if method in ("head", "options"):
                continue
            implemented[(method, path)] = op.get("operationId")
    assert implemented == EXPECTED_OPERATIONS


def test_teams_manifest_structurally_valid(built):
    m = json.loads(built["files"]["manifest.json"])
    assert m["manifestVersion"]
    assert m["name"]["short"] == "Trakt"
    agents = m["copilotAgents"]["declarativeAgents"]
    assert len(agents) == 1 and agents[0]["file"] == "declarativeAgent.json"
    assert m["icons"] == {"color": "color.png", "outline": "outline.png"}
    # Deterministic UUIDv5 (RFC 4122 version 5) app id.
    assert re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
                    m["id"])
