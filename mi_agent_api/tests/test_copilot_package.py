#!/usr/bin/env python3
"""mi_agent_api/tests/test_copilot_package.py

Structural validation of the Microsoft 365 Copilot agent package
(deploy/copilot-agent/): the manifests are valid, the OpenAPI document is valid
and stays in lock-step with the implemented FastAPI routes, the surface is the
intended TWO generic actions (askTraktMi + getArtifact), the downloadable
artifact axis is driven by the server-side registry (not per-artifact manifest
functions), and package-version telemetry is wired consistently.
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

from mi_agent_api import copilot_artifacts, copilot_package

PKG = _REPO_ROOT / "deploy" / "copilot-agent"

# The generic surface: ONE MI action + ONE generic artifact action. Artifact
# types are NOT enumerated as manifest functions/paths any more.
EXPECTED_OPERATIONS = {
    ("post", "/v1/copilot/mi/query"): "askTraktMi",
    ("get", "/v1/copilot/artifacts/latest"): "getArtifact",
}

# Artifact types this release must ship (server-side registry), including the
# three migrated off the declarativeAgent rule-10 denylist.
REQUIRED_ARTIFACT_TYPES = {
    "investor_deck", "canonical_tape",
    "mapping_report", "validation_report", "esma_xml",
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


def test_declarative_agent_no_longer_denylists_registry_artifacts():
    """The migrated artifact types must NOT be refused as unavailable — they are
    now first-class registry entries reachable via getArtifact."""
    instructions = _load_json("declarativeAgent.json")["instructions"].lower()
    # The instructions must reference the generic action and the new types.
    assert "getartifact" in instructions
    for token in ("mapping_report", "validation_report", "esma_xml"):
        assert token in instructions
    # The old blanket denial that named these artifacts as unavailable is gone.
    assert "esma xml, mapping or validation reports" not in instructions
    assert "only these three capabilities exist" not in instructions


def test_plugin_exposes_the_two_generic_actions():
    plugin = _load_json("ai-plugin.json")
    names = [f["name"] for f in plugin["functions"]]
    assert names == ["askTraktMi", "getArtifact"]
    runtimes = plugin["runtimes"]
    assert len(runtimes) == 1
    assert runtimes[0]["spec"]["url"] == "trakt-copilot-openapi.yaml"
    assert sorted(runtimes[0]["run_for_functions"]) == sorted(names)
    # Auth is OAuth (Entra) — never anonymous.
    assert runtimes[0]["auth"]["type"] == "OAuthPluginVault"
    # askTraktMi surfaces the Workspace deep link as its card URL.
    ask = next(f for f in plugin["functions"] if f["name"] == "askTraktMi")
    assert ask["capabilities"]["response_semantics"]["properties"]["url"] == "$.workspaceUrl"


# --------------------------------------------------------------------------- #
# OpenAPI document
# --------------------------------------------------------------------------- #
def test_openapi_document_structurally_valid():
    spec = _load_spec()
    assert spec["openapi"].startswith("3.0")
    assert spec["info"]["title"] == "Trakt Copilot Actions"
    assert spec["security"] == [{"entraBearer": []}]
    assert "entraBearer" in spec["components"]["securitySchemes"]


def test_openapi_covers_exactly_the_two_generic_actions():
    spec = _load_spec()
    ops = {}
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            ops[(method, path)] = op.get("operationId")
    assert ops == EXPECTED_OPERATIONS
    # No internal routes leak into the plugin surface.
    assert not any(p.startswith("/mi/") or p == "/health" for _m, p in ops)


def test_openapi_artifact_type_is_open_string_not_enumerated():
    """The whole point of the generic axis: artifactType is an OPEN string, so a
    new artifact type needs a server deploy only — never a package change."""
    spec = _load_spec()
    get_artifact = spec["paths"]["/v1/copilot/artifacts/latest"]["get"]
    params = {p["name"]: p for p in get_artifact["parameters"] if "name" in p}
    art = params["artifactType"]
    assert art["in"] == "query" and art["required"] is True
    assert art["schema"]["type"] == "string"
    # No enum anywhere would re-introduce per-type manifest hardcoding.
    assert "enum" not in art["schema"]


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


def test_plugin_functions_match_openapi_operation_ids():
    plugin = _load_json("ai-plugin.json")
    spec_ids = {op.get("operationId")
                for methods in _load_spec()["paths"].values()
                for op in methods.values()}
    assert {f["name"] for f in plugin["functions"]} == spec_ids


# --------------------------------------------------------------------------- #
# Registry completeness — the artifact axis is server-driven
# --------------------------------------------------------------------------- #
def test_registry_contains_the_required_artifact_types():
    registered = set(copilot_artifacts.available_types())
    missing = REQUIRED_ARTIFACT_TYPES - registered
    assert not missing, f"registry missing artifact types: {sorted(missing)}"


def test_every_registered_artifact_has_a_callable_resolver():
    for art_type, spec in copilot_artifacts.registry().items():
        assert spec.artifact_type == art_type
        assert callable(spec.resolve)
        assert spec.content_type
        assert spec.label


def test_registry_lookup_normalises_aliases_and_casing():
    for raw in ("investor_deck", "Investor Deck", "investor-deck", "deck"):
        assert copilot_artifacts.lookup(raw).artifact_type == "investor_deck"
    for raw in ("esma_xml", "ESMA XML", "annex 2", "annex2"):
        assert copilot_artifacts.lookup(raw).artifact_type == "esma_xml"
    assert copilot_artifacts.lookup("does_not_exist") is None
    assert copilot_artifacts.lookup("") is None


def test_no_manifest_level_artifact_hardcoding_remains():
    """Adding a NEW artifact type must not require touching the package: no
    per-artifact function, no per-artifact path, no artifactType enum."""
    plugin = _load_json("ai-plugin.json")
    fn_names = {f["name"] for f in plugin["functions"]}
    # No function is named after a specific artifact type.
    assert not any(art in name.lower()
                   for name in fn_names
                   for art in ("deck", "tape", "mapping", "validation", "esma"))
    spec = _load_spec()
    # No path is dedicated to a specific artifact type.
    for path in spec["paths"]:
        for art in ("investor-deck", "canonical-tape", "mapping", "validation", "esma"):
            assert art not in path


# --------------------------------------------------------------------------- #
# Package-version telemetry
# --------------------------------------------------------------------------- #
def test_package_version_is_consistent_across_the_package():
    manifest = _load_json("manifest.json")
    spec = _load_spec()
    assert manifest["version"] == copilot_package.COPILOT_PACKAGE_VERSION
    assert spec["info"]["version"] == copilot_package.COPILOT_PACKAGE_VERSION


def test_openapi_sends_the_package_version_header_on_every_operation():
    spec = _load_spec()
    header = copilot_package.PACKAGE_VERSION_HEADER
    param_def = spec["components"]["parameters"]["PackageVersion"]
    assert param_def["name"] == header and param_def["in"] == "header"
    assert param_def["schema"]["default"] == copilot_package.COPILOT_PACKAGE_VERSION
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            refs = [p.get("$ref", "") for p in op.get("parameters", [])]
            assert any("PackageVersion" in r for r in refs), (
                f"{method} {path} does not send the package-version header")


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
