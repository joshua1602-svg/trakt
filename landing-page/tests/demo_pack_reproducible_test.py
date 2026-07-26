"""Proves the committed demo pack is reproducible from the committed synthetic data.

Run with the repository's pytest:

    python -m pytest landing-page/tests/demo_pack_reproducible_test.py -q

Skipped automatically when the Trakt Python engine's dependencies are not
installed, so a Node-only contributor is not blocked by it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_LANDING_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _LANDING_ROOT.parent
_GENERATOR = _LANDING_ROOT / "scripts" / "build_demo_pack.py"
_PACK = _LANDING_ROOT / "data" / "demo-pack.json"

pytest.importorskip("pandas", reason="Trakt engine dependencies not installed")
pytest.importorskip("plotly", reason="Trakt engine dependencies not installed")


@pytest.fixture(scope="module")
def pack() -> dict:
    return json.loads(_PACK.read_text(encoding="utf-8"))


def test_committed_pack_matches_a_fresh_build():
    """The pack must be exactly what the engine produces today.

    A drift here means either the synthetic dataset or the deterministic engine
    changed and the public demo is quoting stale figures.
    """
    result = subprocess.run(
        [sys.executable, str(_GENERATOR), "--check"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_pack_describes_the_repository_synthetic_client(pack):
    assert pack["client"]["id"] == "synthetic_demo"
    assert pack["client"]["name"] == "Synthetic Demo Lender"
    assert pack["client"]["synthetic"] is True
    assert pack["portfolio"]["id"] == "SYNTHETIC_ERE_Portfolio_012026"
    assert pack["portfolio"]["asOfDate"] == "2025-11-30"


def test_headline_figures_match_the_canonical_dataset(pack):
    """The published totals must equal the governed canonical, to the penny."""
    pd = pytest.importorskip("pandas")
    canonical = _REPO_ROOT / (
        "synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv")
    frame = pd.read_csv(canonical, low_memory=False)

    assert pack["portfolio"]["loanCount"] == len(frame)
    expected = round(float(frame["current_outstanding_balance"].sum()), 2)
    assert pack["portfolio"]["totalBalance"] == expected


def test_no_exposure_level_column_reaches_the_pack(pack):
    forbidden = {
        "loan_identifier", "unique_identifier", "borrower_identifier", "postcode",
        "underlying_exposure_identifier", "youngest_borrower_age",
        "original_obligor_identifier", "new_obligor_identifier",
    }

    artifacts = [a for intent in pack["intents"] for a in intent["artifacts"]]
    artifacts += [
        block
        for report in pack["reports"]
        for page in report["pages"]
        for block in page["blocks"]
    ]

    seen_rows = 0
    for artifact in artifacts:
        for row in artifact.get("rows", []):
            seen_rows += 1
            assert not (forbidden & set(row)), row

    assert seen_rows > 0, "expected the pack to publish at least some rows"


def test_no_internal_path_or_engine_identifier_is_published(pack):
    """Provenance is described in prose; concrete paths stay server-side."""
    published = json.dumps({"intents": pack["intents"], "reports": pack["reports"]})
    for leak in ("/home/", "synthetic_demo/output", "mi_agent.workflow", "querySpec",
                 "blob.core.windows.net", "AZURE_", "MI_AGENT_"):
        assert leak not in published, leak


def test_every_intent_is_answerable_and_labelled(pack):
    for intent in pack["intents"]:
        assert intent["id"] and intent["label"] and intent["answer"]
        assert intent["phrases"], intent["id"]
        assert intent["artifacts"], intent["id"]
        # A narrative that failed substitution would show as "n/a".
        assert "n/a" not in intent["answer"], intent["id"]
        assert "{" not in intent["answer"], intent["id"]


def test_controlled_unsupported_topics_explain_themselves(pack):
    ids = {topic["id"] for topic in pack["unsupported"]}
    assert {"temporal_movement", "pipeline", "arrears", "loan_level"} <= ids
    for topic in pack["unsupported"]:
        assert topic["reason"] and topic["productionNote"]
