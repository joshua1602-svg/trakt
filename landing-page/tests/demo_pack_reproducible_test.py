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


def test_committed_source_extracts_match_a_fresh_generation():
    """The dataset BENEATH the pack must be reproducible too.

    The pack is only as trustworthy as the canonical it is built from, and the
    canonical is only as trustworthy as the source extracts. Guarding the pack
    alone leaves the layer that actually carries the figures unchecked.

    The generator is pure stdlib and seeded, so this is version-independent in
    a way the pack build is not.
    """
    generator = _REPO_ROOT / "synthetic_demo/build_multibook_input.py"
    result = subprocess.run(
        [sys.executable, str(generator), "--check"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_canonical_carries_book_identity_on_every_row():
    """Book identity is a governed attribute, not a query-time filter.

    Gate 2 stamps it. A row that cannot name its book would silently corrupt
    every per-book total downstream, so it is asserted at the dataset level
    rather than trusted.
    """
    pd = pytest.importorskip("pandas")
    for period in ("2026-05-31", "2026-06-30"):
        frame = pd.read_csv(
            _REPO_ROOT / f"synthetic_demo/output/multibook/platform_{period}"
            "_canonical_typed.csv", low_memory=False)
        for column in ("source_portfolio_id", "source_portfolio_type"):
            assert column in frame.columns, f"{period}: {column} missing"
            blank = frame[column].astype(str).str.strip() == ""
            assert not blank.any(), f"{period}: {int(blank.sum())} blank {column}"
        assert set(frame["source_portfolio_id"]) == {
            "alp_origination", "alp_acquired", "spv1_sponsored"}


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
    assert engine.source.client_id == "alderbridge_demo"
    assert engine.source.portfolio_id == "ALP_Platform_202606"
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
    assert source["clientId"] == "alderbridge_demo"
    assert source["portfolioId"] == "ALP_Platform_202606"
    assert source["currency"] == "GBP"
    assert source["reportingDate"] == "2026-06-30"
    assert len(source["canonicalSha256"]) == 64
    assert source["totalBalance"] == pack["portfolio"]["totalBalance"]
    assert source["exposures"] == pack["portfolio"]["loanCount"]


def test_pack_describes_the_repository_synthetic_client(pack):
    assert pack["client"]["id"] == "alderbridge_demo"
    assert pack["client"]["name"] == "Alderbridge Lending Platform"
    assert pack["client"]["synthetic"] is True
    assert pack["portfolio"]["id"] == "ALP_Platform_202606"
    assert pack["portfolio"]["asOfDate"] == "2026-06-30"
    assert pack["portfolio"]["priorAsOfDate"] == "2026-05-31"


def test_headline_figures_match_the_canonical_dataset(pack):
    """The published totals must equal the governed canonical, to the penny."""
    pd = pytest.importorskip("pandas")
    canonical = _REPO_ROOT / (
        "synthetic_demo/output/multibook/platform_2026-06-30_canonical_typed.csv")
    frame = pd.read_csv(canonical, low_memory=False)

    assert pack["portfolio"]["loanCount"] == len(frame)
    expected = round(float(frame["current_outstanding_balance"].sum()), 2)
    assert pack["portfolio"]["totalBalance"] == expected


def test_book_totals_reconcile_to_both_governed_scopes(pack):
    """Per-book figures must sum to the totals the page publishes.

    Two scopes are legitimate at once: the platform is what sits on the
    sponsor's balance sheet, the sponsor total adds the SPV they sold but still
    service and report on. Both are published, so both are asserted — a page
    that showed books and totals which did not add up would be worse than one
    that showed neither.
    """
    pd = pytest.importorskip("pandas")
    canonical = _REPO_ROOT / (
        "synthetic_demo/output/multibook/platform_2026-06-30_canonical_typed.csv")
    frame = pd.read_csv(canonical, low_memory=False)
    balances = frame.groupby("source_portfolio_id")["current_outstanding_balance"].sum()

    books = {b["id"]: b for b in pack["portfolio"]["books"]}
    assert set(books) == {"alp_origination", "alp_acquired", "spv1_sponsored"}
    # SPV1 is sold and derecognised; the other two are warehoused. The page
    # marks that distinction, so the pack must carry it.
    assert books["spv1_sponsored"]["balanceSheetStatus"] == "sold"
    assert {books[b]["balanceSheetStatus"] for b in
            ("alp_origination", "alp_acquired")} == {"warehoused"}

    platform = round(float(balances[["alp_origination", "alp_acquired"]].sum()), 2)
    sponsor = round(float(balances.sum()), 2)
    assert sponsor == pack["portfolio"]["totalBalance"]
    assert round(sum(float(balances[b]) for b in books), 2) == sponsor

    def _money(text):
        return float(text.replace("\u00a3", "").replace(",", ""))

    # Displayed to the nearest pound, so compare at that resolution.
    assert abs(_money(pack["portfolio"]["platformBalanceDisplay"]) - platform) < 1
    assert abs(_money(pack["portfolio"]["totalBalanceDisplay"]) - sponsor) < 1


def test_period_movement_is_differenced_not_stored(pack):
    """Movement must be the difference of two governed snapshots."""
    pd = pytest.importorskip("pandas")
    movement = next(i for i in pack["intents"] if i["id"] == "period_movement")

    current = pd.read_csv(_REPO_ROOT / (
        "synthetic_demo/output/multibook/platform_2026-06-30_canonical_typed.csv"),
        low_memory=False)
    prior = pd.read_csv(_REPO_ROOT / (
        "synthetic_demo/output/multibook/platform_2026-05-31_canonical_typed.csv"),
        low_memory=False)

    platform_books = ["alp_origination", "alp_acquired"]
    now = current[current["source_portfolio_id"].isin(platform_books)]
    was = prior[prior["source_portfolio_id"].isin(platform_books)]
    delta = round(float(now["current_outstanding_balance"].sum()
                        - was["current_outstanding_balance"].sum()), 2)

    # The answer states the movement, so it must equal the differenced figure.
    assert f"{abs(delta):,.0f}" in movement["answer"].replace("\u00a3", "")
    # And it must name both dates, so the comparison basis is explicit.
    assert "31 May 2026" in movement["answer"]
    assert "30 June 2026" in movement["answer"]


def test_annex_exceptions_carry_their_reasoning(pack):
    """A pass/fail verdict is not actionable; each exception must explain itself."""
    exceptions = next(i for i in pack["intents"] if i["id"] == "annex_exceptions")
    rows = [row for artifact in exceptions["artifacts"]
            for row in artifact.get("rows", [])]
    assert rows, "expected the reconciliation to publish at least one exception"

    for row in rows:
        assert row["disposition"], row
        assert row["field"], row
        assert row["resolution"], row

    dispositions = {row["disposition"] for row in rows}
    # The three seeded failures are different failure modes caught at different
    # gates. If they collapse to one, the demonstration has stopped showing
    # that validation reasons rather than merely runs.
    assert "Blocks submission" in dispositions
    assert "Defaults to a no-data code" in dispositions
    assert "Outside Annex 2 scope" in dispositions


def test_every_published_breakdown_covers_the_whole_book(pack):
    """A breakdown that says "of the funded book" must actually cover it.

    This is the control for OI-1. A portfolio-scope filter was silently applied
    to a question that had asked for a breakdown, so the channel answer covered
    67.9% of the platform while describing itself as a share of the funded book.
    The figures were individually correct and the total was wrong, which is the
    hardest kind of error to see by reading.

    Coverage is the engine's own reconciliation figure, so this asserts against
    what the engine believes it included — not against a number recomputed here.
    """
    for intent in pack["intents"]:
        for artifact in intent.get("artifacts") or []:
            coverage = artifact.get("coverage")
            if coverage is None:
                continue  # composite answers carry their own reconciliation
            assert coverage == pytest.approx(100.0, abs=0.05), (
                f"{intent['id']}: breakdown covers {coverage}% of the book. "
                "Either the answer is scoped and must say so, or a scope filter "
                "is being applied to a question that did not ask for one."
            )


def test_channel_breakdown_reconciles_to_the_sponsor_total(pack):
    """The specific figure OI-1 got wrong, pinned."""
    channel = next(i for i in pack["intents"] if i["id"] == "channel")
    rows = channel["artifacts"][0]["rows"]
    assert {r["origination_channel"] for r in rows} == {"Direct", "Broker", "IFA"}, (
        "all three books' channels must appear — the acquired book's 'Broker' "
        "was the one silently dropped"
    )
    total = round(sum(r["current_outstanding_balance_sum"] for r in rows), 2)
    assert total == pack["portfolio"]["totalBalance"]


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
    assert {"pipeline", "arrears", "loan_level"} <= ids
    # temporal_movement was a controlled refusal until a second governed
    # snapshot existed. It is answerable now, so it must NOT be refused —
    # a page that both answers and declines the same question is incoherent.
    assert "temporal_movement" not in ids
    assert any(i["id"] == "period_movement" for i in pack["intents"])
    # The two the page surfaces are the first two, and both must stay
    # genuinely underivable.
    assert [t["id"] for t in pack["unsupported"]][:2] == ["loan_level", "pipeline"]
    for topic in pack["unsupported"]:
        assert topic["reason"] and topic["productionNote"]
