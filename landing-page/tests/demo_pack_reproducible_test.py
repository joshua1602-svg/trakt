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


def _generator():
    """Import the generator module under its own name (it uses dataclasses)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("bdp", str(_GENERATOR))
    module = importlib.util.module_from_spec(spec)
    sys.modules["bdp"] = module
    spec.loader.exec_module(module)
    return module


def test_the_resolved_dataset_is_the_pinned_demo_source():
    """The generator publishes from exactly one dataset, named explicitly."""
    m = _generator()
    engine = m.Engine(m.DEMO_SOURCE)
    assert engine.source.client_id == "synthetic_demo"
    assert engine.source.portfolio_id == "SYNTHETIC_ERE_Portfolio_012026"
    assert engine.as_of == m.DEMO_SOURCE.expected_reporting_date
    assert (m.DEMO_SOURCE.expected_min_balance
            <= engine.total_balance
            <= m.DEMO_SOURCE.expected_max_balance)


@pytest.mark.parametrize("field,value,expected", [
    ("expected_min_balance", 1_800_000_000, "outside the expected range"),
    ("client_id", "client_001", "client id"),
    ("expected_reporting_date", "2026-01-31", "reporting date"),
    ("expected_currency", "EUR", "currency"),
    ("expected_min_exposures", 1000, "below the expected minimum"),
    ("expected_asset_class", "auto_loan", "asset class"),
])
def test_a_mismatched_expectation_fails_closed(field, value, expected):
    """Every identity axis refuses rather than publishing the wrong figures.

    The balance case is the one that matters most: it is exactly the scenario
    where a landing page pinned to a large portfolio would otherwise silently
    publish a small one.
    """
    import dataclasses
    m = _generator()
    source = dataclasses.replace(
        m.DEMO_SOURCE,
        **{field: value},
        # The fingerprint would fire first; this isolates the axis under test.
        expected_sha256="0" * 64,
    )
    if field == "expected_min_balance":
        source = dataclasses.replace(source, expected_max_balance=2_000_000_000)

    with pytest.raises(m.DemoSourceMismatch) as excinfo:
        m.Engine(source)
    assert expected in str(excinfo.value)
    assert "does not match" in str(excinfo.value)


def test_a_substituted_dataset_is_caught_before_it_is_parsed():
    """A replaced or edited canonical fails on its fingerprint, not on a
    downstream schema error — so the message is diagnosable."""
    import dataclasses
    import shutil
    import tempfile
    m = _generator()
    other = Path(tempfile.mkdtemp()) / "substituted.csv"
    shutil.copy(_REPO_ROOT / "canonical_snapshot_demo.csv", other)

    with pytest.raises(m.DemoSourceMismatch) as excinfo:
        m.Engine(dataclasses.replace(m.DEMO_SOURCE, canonical_path=other))
    assert "fingerprint" in str(excinfo.value)


def test_the_generator_has_no_silent_fallback():
    """There is no code path from a failed source check to a published pack.

    Checked against the parsed module rather than its text, so the prose that
    *explains* the absent fallback does not itself trip the assertion.
    """
    import ast
    tree = ast.parse(_GENERATOR.read_text(encoding="utf-8"))

    # The data-source resolver — the thing that would silently substitute the
    # bundled demo dataset — is never imported.
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any("data_source" in name for name in imported), imported

    # The identity assertions are unconditional statements in Engine.__init__,
    # not guarded by a flag or a try/except.
    engine = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "Engine"
    )
    init = next(
        node for node in engine.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    called = {
        node.func.id
        for node in ast.walk(init)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_assert_source_file" in called
    assert "_assert_source_identity" in called
    assert not any(isinstance(node, (ast.Try, ast.If)) for node in init.body), (
        "source verification must not be conditional")


def test_pack_carries_its_source_identity(pack):
    """The runtime validates the pack it is served, so the pack must say what
    it was built from."""
    source = pack["source"]
    assert source["clientId"] == "synthetic_demo"
    assert source["portfolioId"] == "SYNTHETIC_ERE_Portfolio_012026"
    assert source["currency"] == "GBP"
    assert source["reportingDate"] == "2025-11-30"
    assert len(source["canonicalSha256"]) == 64
    assert source["totalBalance"] == pack["portfolio"]["totalBalance"]
    assert source["exposures"] == pack["portfolio"]["loanCount"]


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
