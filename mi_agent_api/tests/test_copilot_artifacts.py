#!/usr/bin/env python3
"""mi_agent_api/tests/test_copilot_artifacts.py — the generic artifact registry.

Unit coverage of the registry itself (normalisation, aliases, completeness) and
the resolvers for the three registry-added types, independent of the HTTP layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from mi_agent_api import copilot_artifacts as ca


REQUIRED = {"investor_deck", "canonical_tape",
            "mapping_report", "validation_report", "esma_xml"}


def test_all_required_types_registered_with_resolvers():
    types = set(ca.available_types())
    assert REQUIRED <= types
    for t in REQUIRED:
        spec = ca.lookup(t)
        assert spec is not None and callable(spec.resolve)
        assert spec.content_type and spec.label


def test_lookup_normalises_casing_spaces_and_hyphens():
    assert ca.lookup("investor_deck").artifact_type == "investor_deck"
    assert ca.lookup("Investor Deck").artifact_type == "investor_deck"
    assert ca.lookup("investor-deck").artifact_type == "investor_deck"
    assert ca.lookup("canonical loan tape").artifact_type == "canonical_tape"
    assert ca.lookup("ANNEX 2").artifact_type == "esma_xml"
    assert ca.lookup(None) is None
    assert ca.lookup("unknown") is None


@pytest.mark.parametrize("art_type,env,filename,ctype", [
    ("mapping_report", "MI_AGENT_MAPPING_REPORT", "m.csv", "text/csv"),
    ("validation_report", "MI_AGENT_VALIDATION_REPORT", "v.json", "application/json"),
    ("esma_xml", "MI_AGENT_ESMA_XML", "a.xml", "application/xml"),
])
def test_new_type_resolver_uses_explicit_env_override(
        art_type, env, filename, ctype, tmp_path, monkeypatch):
    for var in ("MI_AGENT_PLATFORM_URI", "MI_AGENT_PLATFORM_DIR",
                "MI_AGENT_CLIENT_ID"):
        monkeypatch.delenv(var, raising=False)
    f = tmp_path / filename
    f.write_text("data")
    monkeypatch.setenv(env, str(f))
    spec = ca.lookup(art_type)
    assert spec.content_type == ctype
    resolved = spec.resolve("client_001")
    assert resolved is not None and resolved.path == f


def test_new_type_resolver_returns_none_when_absent(monkeypatch):
    for var in ("MI_AGENT_MAPPING_REPORT", "MI_AGENT_PLATFORM_URI",
                "MI_AGENT_PLATFORM_DIR"):
        monkeypatch.delenv(var, raising=False)
    assert ca.lookup("mapping_report").resolve("client_001") is None


def test_registering_a_new_type_is_a_one_line_server_change():
    """Adding an artifact is register(...) — no package artefact is touched."""
    before = set(ca.available_types())
    ca.register(ca.ArtifactSpec(
        artifact_type="_test_probe", content_type="text/plain",
        label="test probe", resolve=lambda _c: None, aliases=("probe",)))
    try:
        assert ca.lookup("probe").artifact_type == "_test_probe"
        assert "_test_probe" in ca.available_types()
    finally:
        # keep the shared registry clean for other tests
        ca._REGISTRY.pop("_test_probe", None)
        ca._ALIAS_INDEX.pop("_test_probe", None)
        ca._ALIAS_INDEX.pop("probe", None)
    assert set(ca.available_types()) == before
