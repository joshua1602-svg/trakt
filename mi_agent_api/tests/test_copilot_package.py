#!/usr/bin/env python3
"""mi_agent_api/tests/test_copilot_package.py

Structural validation of the Microsoft 365 Copilot agent package
(deploy/copilot-agent/): the manifests are valid, the OpenAPI document is
valid and stays in lock-step with the implemented FastAPI routes, and ONLY the
three intended actions are exposed to Copilot.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

PKG = _REPO_ROOT / "deploy" / "copilot-agent"

EXPECTED_OPERATIONS = {
    ("post", "/v1/copilot/mi/query"): "askTraktMi",
    ("get", "/v1/copilot/artifacts/latest/investor-deck"): "getLatestInvestorDeck",
    ("get", "/v1/copilot/artifacts/latest/canonical-tape"): "getLatestCanonicalTape",
}


def _load_json(name: str) -> dict:
    return json.loads((PKG / name).read_text(encoding="utf-8"))


def _load_spec() -> dict:
    return yaml.safe_load((PKG / "trakt-copilot-openapi.yaml").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Manifests
# --------------------------------------------------------------------------- #
def test_teams_manifest_structurally_valid():
    m = _load_json("manifest.json")
    assert m["manifestVersion"]
    assert m["name"]["short"] == "Trakt"
    agents = m["copilotAgents"]["declarativeAgents"]
    assert len(agents) == 1 and agents[0]["file"] == "declarativeAgent.json"
    assert m["icons"] == {"color": "color.png", "outline": "outline.png"}


def test_declarative_agent_structurally_valid():
    da = _load_json("declarativeAgent.json")
    assert da["name"] == "Trakt"
    assert "Never invent" in da["instructions"] or "never invent" in da["instructions"].lower()
    assert len(da["conversation_starters"]) == 4
    assert da["actions"] == [{"id": "traktActions", "file": "ai-plugin.json"}]


def test_plugin_exposes_exactly_three_functions():
    plugin = _load_json("ai-plugin.json")
    names = [f["name"] for f in plugin["functions"]]
    assert names == ["askTraktMi", "getLatestInvestorDeck", "getLatestCanonicalTape"]
    runtimes = plugin["runtimes"]
    assert len(runtimes) == 1
    assert runtimes[0]["spec"]["url"] == "trakt-copilot-openapi.yaml"
    assert sorted(runtimes[0]["run_for_functions"]) == sorted(names)
    # Auth is OAuth (Entra) — never anonymous.
    assert runtimes[0]["auth"]["type"] == "OAuthPluginVault"


# --------------------------------------------------------------------------- #
# OpenAPI document
# --------------------------------------------------------------------------- #
def test_openapi_document_structurally_valid():
    spec = _load_spec()
    assert spec["openapi"].startswith("3.0")
    assert spec["info"]["title"] == "Trakt Copilot Actions"
    assert spec["security"] == [{"entraBearer": []}]
    assert "entraBearer" in spec["components"]["securitySchemes"]


def test_openapi_covers_exactly_the_three_actions():
    spec = _load_spec()
    ops = {}
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            ops[(method, path)] = op.get("operationId")
    assert ops == EXPECTED_OPERATIONS
    # No internal routes leak into the plugin surface.
    assert not any(p.startswith("/mi/") or p == "/health" for _m, p in ops)


def test_openapi_operation_ids_match_the_implemented_routes():
    from mi_agent_api.app import app

    # The app's own generated schema is the authoritative view of what is
    # implemented (and include_in_schema=False keeps the signed-download
    # redemption route out of it by design).
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


def test_plugin_functions_match_openapi_operation_ids():
    plugin = _load_json("ai-plugin.json")
    spec_ids = {op.get("operationId")
                for methods in _load_spec()["paths"].values()
                for op in methods.values()}
    assert {f["name"] for f in plugin["functions"]} == spec_ids


# --------------------------------------------------------------------------- #
# Packaging script
# --------------------------------------------------------------------------- #
def test_package_builder_produces_valid_zip(tmp_path):
    sys.path.insert(0, str(PKG))
    try:
        import package_agent
        zip_path = package_agent.build(tmp_path)
    finally:
        sys.path.remove(str(PKG))
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert {"manifest.json", "declarativeAgent.json", "ai-plugin.json",
                "trakt-copilot-openapi.yaml", "color.png", "outline.png"} <= names
        assert zf.read("color.png").startswith(b"\x89PNG")
        assert zf.read("outline.png").startswith(b"\x89PNG")
