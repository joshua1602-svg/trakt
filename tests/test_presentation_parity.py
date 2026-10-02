#!/usr/bin/env python3
"""tests/test_presentation_parity.py

React and the investor PPTX, compared as PRESENTATION — not as payloads.

``tests/mi_agent_pptx/test_channel_parity.py`` already proves the two channels
agree on the NUMBERS: it drives the real ``/mi/*`` routes on one side and
``mi_agent_pptx.mi_api.build_dashboard_data`` on the other. What it cannot see is
everything decided after that payload, because it never builds a deck — and both
halves of the two defects this suite exists to prevent lived exactly there:

  * the deck formatted money with a literal pound sign, so a book whose governed
    currency was EUR reported EUR on the dashboard and GBP in the pack;
  * the browser re-sorted ordinal bands into their natural ladder and the deck
    drew them in the balance order the payload arrived in, so the same LTV
    stratification read two different ways.

Neither is visible from a payload comparison. Both are visible here, because
this suite drives BOTH production paths end to end:

    React      TestClient(mi_agent_api.app)  -> GET /mi/snapshot, /mi/multidim, …
    PPTX       POST /mi/decks/generate -> poll -> GET /mi/decks/download

and then reads the PowerPoint that comes back.

WHAT IS INSPECTED, AND WHY IT IS SPLIT IN TWO
---------------------------------------------
Text that reaches a slide as text — titles, KPI tile values, tables, straplines —
is read straight out of the downloaded ``.pptx`` with python-pptx. That covers
currency, titles, labels, RAG status, reporting period and client identity.

Bar-list CATEGORY ORDER cannot be: the deck draws a bar list as a matplotlib PNG,
so the labels are pixels by the time the file exists. For those, the renderers
record what they drew AT THE MOMENT THEY DREW IT (``render.record_renders``), and
the record travels into the deck's own preflight sidecar — a production artefact
the publishing stage already writes and gates on, not a test fixture and not the
pre-render payload. That record is what the ordering assertions read.

Each test states which defect it would catch.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pd = pytest.importorskip("pandas")
pptx = pytest.importorskip("pptx")

CLIENT = "parity"
_CENTRAL = "18_central_lender_tape.csv"

#: LTV bands chosen so BAND order and BALANCE order disagree. A renderer that
#: sorts by balance (the order the governed stratification payload arrives in,
#: because ``analytics_lib.stratify`` ranks by materiality) draws
#: 40-50% | 80-90% | 20-30%; the governed ladder is 20-30% | 40-50% | 80-90%.
#: Any test asserting the ladder therefore fails the moment the old behaviour
#: returns, which is the whole point of picking these numbers.
_LTV_SPREAD = (
    # (ltv %, balance, region, age, rate)
    (25.0, 100_000.0, "Wales", 62, 6.4),
    (45.0, 400_000.0, "London", 71, 7.2),
    (85.0, 250_000.0, "South East", 78, 5.4),
    (45.0, 380_000.0, "London", 83, 7.1),
    (25.0, 90_000.0, "Wales", 57, 6.2),
)


def _loan(i, pid, ptype, *, ltv, balance, region, age, rate, cut, origination):
    return {
        "unique_identifier": f"{pid}_L{i:04d}",
        "source_portfolio_id": pid,
        "source_portfolio_type": ptype,
        "source_portfolio_label": pid.replace("_", " ").title(),
        "current_outstanding_balance": balance,
        "current_principal_balance": balance,
        "original_principal_balance": balance * 1.05,
        "current_valuation_amount": balance / (ltv / 100.0),
        "original_valuation_amount": balance / (ltv / 100.0),
        "current_loan_to_value": ltv,
        "original_loan_to_value": (ltv / 100.0) - 0.02,
        "current_interest_rate": rate,
        "youngest_borrower_age": age,
        "geographic_region_collateral": region,
        "collateral_geography": region,
        "origination_channel": "Direct",
        "broker_channel": "Direct",
        "product_type": "Lifetime Mortgage",
        "origination_date": origination,
        "data_cut_off_date": cut,
    }


def _rows(cut, *, scale=1.0, origination="2021-04-01"):
    out = []
    for i, (ltv, balance, region, age, rate) in enumerate(_LTV_SPREAD, start=1):
        out.append(_loan(i, "direct_001", "direct", ltv=ltv, balance=balance * scale,
                         region=region, age=age, rate=rate, cut=cut,
                         origination=origination))
    return out


def _write_book(root: Path, client: str = CLIENT):
    """Two reporting periods, in the layout run discovery actually walks."""
    for run_id, date, scale in (("mi_2026_05", "2026-05-31", 0.94),
                                ("mi_2026_06", "2026-06-30", 1.0)):
        central = root / client / run_id / "central"
        central.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(_rows(date, scale=scale)).to_csv(central / _CENTRAL, index=False)
    return root


def _client_config(path: Path, code: str) -> Path:
    """A governed client configuration declaring the reporting currency.

    This is the real mechanism: ``currency.governed_currency_code`` reads
    ``portfolio.base_currency`` from the approved client configuration, and that
    outranks anything inferred from the tape.
    """
    path.write_text(f"portfolio:\n  base_currency: {code}\n", encoding="utf-8")
    return path


@pytest.fixture()
def deployment(tmp_path, monkeypatch):
    """One tenant, two runs, a filesystem deck store, auth off.

    Parameterised by the governed reporting currency through
    ``request.param``-free indirection: tests that care set
    ``TRAKT_MI_CLIENT_CONFIG`` themselves via :func:`use_currency_config`.
    """
    root = _write_book(tmp_path / "runs")
    monkeypatch.setenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", str(root))
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", CLIENT)
    monkeypatch.setenv("MI_AGENT_PIPELINE_ROOT", str(root))
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blobstore"))
    monkeypatch.setenv("TRAKT_INVESTOR_PPTX_PERSIST", "true")
    monkeypatch.setenv("TRAKT_INVESTOR_PPTX_ON_DEMAND", "true")
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "false")
    monkeypatch.delenv("MI_AGENT_DECK_ROOT", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("TRAKT_BLOB_CONNECTION", raising=False)
    monkeypatch.delenv("TRAKT_MI_CLIENT_CONFIG", raising=False)

    from mi_agent_api import deck_generation, datasets
    deck_generation.reset_jobs()
    # The API caches a resolved currency per client for the life of the process;
    # a test that changes the governed configuration must not inherit the last
    # test's answer.
    datasets._CLIENT_CURRENCY_CACHE.clear()
    yield tmp_path
    deck_generation.reset_jobs()
    datasets._CLIENT_CURRENCY_CACHE.clear()


def use_currency_config(tmp_path, monkeypatch, code: str) -> None:
    """Put a governed reporting currency in force for this test."""
    from mi_agent_api import currency, datasets
    config = _client_config(tmp_path / f"client_{code}.yaml", code)
    monkeypatch.setenv("TRAKT_MI_CLIENT_CONFIG", str(config))
    currency._load_client_config.cache_clear()
    datasets._CLIENT_CURRENCY_CACHE.clear()


# --------------------------------------------------------------------------- #
# Driving both production channels.
# --------------------------------------------------------------------------- #

def react(path: str, **params):
    """A real React HTTP call against the real app."""
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app

    client = TestClient(app)
    res = client.get(path, params={"portfolioId": f"{CLIENT}/mi_2026_06", **params})
    assert res.status_code == 200, f"{path} -> {res.status_code}: {res.text}"
    return res.json()


def generate_and_download(timeout_s: int = 300) -> bytes:
    """The React button's own path: generate, poll, discover, download.

    Returns the bytes of the PowerPoint the download route served — the same file
    a user would receive. Nothing is mocked and no builder is called directly:
    certifying against a direct builder call would bypass the job service, the
    orchestration stage and the publication gates the button actually goes
    through.
    """
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app

    client = TestClient(app)
    accepted = client.post("/mi/decks/generate", json={})
    assert accepted.status_code == 202, accepted.text
    job_id = accepted.json()["jobId"]

    deadline = time.time() + timeout_s
    body = None
    while time.time() < deadline:
        status = client.get(f"/mi/decks/generate/{job_id}")
        assert status.status_code in (200, 202), status.text
        body = status.json()
        if body["state"] in ("completed", "blocked", "failed"):
            break
        time.sleep(0.4)
    assert body and body["state"] == "completed", body
    assert body["failedGates"] == [], body

    got = client.get("/mi/decks/download")
    assert got.status_code == 200, got.text
    assert got.content[:4] == b"PK\x03\x04", "not a PowerPoint file"
    return got.content


def deck_text(content: bytes) -> str:
    """All text that reached a slide as text, from the downloaded file."""
    deck = pptx.Presentation(io.BytesIO(content))
    parts = []
    for slide in deck.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


def deck_slide_titles(content: bytes) -> list:
    """The first text shape on each slide — the slide title band."""
    deck = pptx.Presentation(io.BytesIO(content))
    titles = []
    for slide in deck.slides:
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                titles.append(shape.text_frame.text.strip().split("\n")[0])
                break
    return titles


def sidecar(tmp_path) -> dict:
    """The deck's own preflight sidecar — a production artefact.

    ``pptx_stage`` writes it beside the generated deck, reads it to decide
    whether to publish, and records parts of it in the run manifest. It carries
    the render record: what each drawing function actually drew.
    """
    matches = sorted((tmp_path / "runs").glob("**/investor_pack.pptx.preflight.json"))
    assert matches, "the generator wrote no preflight sidecar"
    return json.loads(matches[-1].read_text(encoding="utf-8"))


def drawn_barlists(side: dict) -> dict:
    """``{dimension: [categories drawn]}`` from the render record."""
    out = {}
    for entry in side.get("rendered") or ():
        if entry.get("kind") == "barlist" and entry.get("dimension"):
            out[entry["dimension"]] = [str(c) for c in entry.get("categories") or ()]
    return out


# --------------------------------------------------------------------------- #
# 1. CURRENCY.
#
# Catches: the deck hard-coding a pound sign. A build with EUR in force used to
# render "£124.6MM" on every tile while /mi/snapshot said EUR.
# --------------------------------------------------------------------------- #

def test_a_euro_book_renders_euro_on_both_surfaces(deployment, monkeypatch):
    use_currency_config(deployment, monkeypatch, "EUR")

    snapshot = react("/mi/snapshot")
    assert snapshot["currencyCode"] == "EUR", "the dashboard did not adopt EUR"
    tiles = "  ".join(str(k.get("value")) for k in snapshot["kpis"])
    assert "€" in tiles, f"the dashboard rendered no euro amounts: {tiles}"

    text = deck_text(generate_and_download())
    assert "€" in text, "the deck rendered no euro amounts"
    assert "£" not in text, (
        "the deck rendered a POUND symbol for a EUR book — the governed currency "
        "defect has returned")


def test_a_sterling_book_still_renders_sterling(deployment, monkeypatch):
    """The fix must not have simply swapped one hard-coded symbol for another."""
    use_currency_config(deployment, monkeypatch, "GBP")

    snapshot = react("/mi/snapshot")
    assert snapshot["currencyCode"] == "GBP"

    text = deck_text(generate_and_download())
    assert "£" in text, "the deck rendered no sterling amounts"
    assert "€" not in text and "$" not in text


def test_the_deck_records_the_currency_it_reported_in(deployment, monkeypatch):
    use_currency_config(deployment, monkeypatch, "EUR")
    generate_and_download()
    assert sidecar(deployment).get("currency_code") == "EUR"


def test_both_channels_agree_on_the_headline_balance_and_its_currency(
        deployment, monkeypatch):
    """Same number, same symbol, on both surfaces."""
    use_currency_config(deployment, monkeypatch, "EUR")
    snapshot = react("/mi/snapshot")
    headline = next(k for k in snapshot["kpis"] if k["id"] == "balance")

    text = deck_text(generate_and_download())
    assert headline["value"] in text, (
        f"the dashboard's headline balance {headline['value']!r} does not appear "
        f"in the deck")


# --------------------------------------------------------------------------- #
# 2. BUCKET / CATEGORY ORDER.
#
# Catches: the deck drawing ordinal bands in balance order. The fixture is built
# so balance order and band order genuinely disagree.
# --------------------------------------------------------------------------- #

def test_the_dashboard_serves_bands_in_the_governed_ladder(deployment):
    strat = {s["key"]: s for s in react("/mi/snapshot")["stratifications"]}
    ltv = strat["ltv"]
    labels = [b["label"] for b in ltv["bars"]]
    assert ltv.get("displayOrder") == "governed", (
        "the payload no longer declares a governed order, so the browser would "
        "fall back to deciding it locally")
    assert labels == sorted(labels, key=_ltv_rank), (
        f"the dashboard served LTV bands out of the governed ladder: {labels}")


def test_the_deck_draws_bands_in_the_same_order_the_dashboard_serves(
        deployment):
    api_labels = [b["label"] for s in react("/mi/snapshot")["stratifications"]
                  if s["key"] == "ltv" for b in s["bars"]]
    generate_and_download()
    drawn = drawn_barlists(sidecar(deployment))
    assert "ltv" in drawn, "the deck drew no LTV bar list to compare"
    assert drawn["ltv"] == api_labels, (
        f"the deck drew LTV bands {drawn['ltv']} where the dashboard serves "
        f"{api_labels} — the bucket-order defect has returned")


def test_the_drawn_ltv_order_is_the_ladder_and_not_the_balance_ranking(
        deployment):
    """The fixture's balance ranking differs from its band ladder on purpose.

    This is the assertion that would have failed before the fix and passes after
    it. A renderer that draws the payload in arrival order (balance descending)
    produces a different sequence, and this names both.
    """
    generate_and_download()
    drawn = drawn_barlists(sidecar(deployment))["ltv"]
    assert drawn == sorted(drawn, key=_ltv_rank), (
        f"LTV bands drawn out of the governed ladder: {drawn}")

    balance_order = [b["label"] for b in sorted(
        (b for s in react("/mi/snapshot")["stratifications"] if s["key"] == "ltv"
         for b in s["bars"]),
        key=lambda b: b["balance"], reverse=True)]
    assert drawn != balance_order, (
        "the fixture no longer distinguishes band order from balance order, so "
        "this test can no longer catch the defect it exists for")


def _ltv_rank(label):
    from mi_agent_api import presentation
    return presentation.order_key("ltv")(label if isinstance(label, str)
                                         else label["label"])


def test_every_banded_bar_list_passed_the_publication_gate(deployment):
    """The product enforces the order, not only this suite."""
    generate_and_download()
    gates = {g: True for g in (sidecar(deployment)["preflight"].get("failed_gates") or [])}
    assert "governed_bucket_order" not in gates
    assert "governed_currency" not in gates


# --------------------------------------------------------------------------- #
# 3. SERIES ORDER, TITLES, LABELS.
# --------------------------------------------------------------------------- #

def test_evolution_series_are_named_as_the_dashboard_names_them(deployment):
    """The deck's funded-evolution series carry the dashboard's measure names."""
    generate_and_download()
    lines = [e for e in sidecar(deployment)["rendered"] if e["kind"] == "lines"]
    assert lines, "the deck drew no line charts"
    names = {n for e in lines for n in e.get("series") or ()}
    assert {"Funded balance", "WA LTV"} & names, (
        f"no recognisable funded measure among the drawn series: {sorted(names)}")


def test_slide_titles_are_present_and_unique(deployment):
    titles = deck_slide_titles(generate_and_download())
    assert titles, "the deck has no titles"
    duplicates = {t for t in titles if titles.count(t) > 1}
    assert not duplicates, f"duplicated slide titles: {sorted(duplicates)}"


def test_the_stratification_labels_match_the_dashboards(deployment):
    api = {s["key"]: s["label"] for s in react("/mi/snapshot")["stratifications"]}
    text = deck_text(generate_and_download())
    for key in ("ltv", "age"):
        assert api[key] in text, (
            f"the deck does not carry the dashboard's label for {key}: {api[key]!r}")


# --------------------------------------------------------------------------- #
# 4. REPORTING PERIOD, SCOPE AND CLIENT IDENTITY.
# --------------------------------------------------------------------------- #

def test_both_channels_report_the_same_period(deployment):
    reporting = react("/mi/snapshot")["portfolio"]["reporting_date"]
    assert reporting == "2026-06-30"
    text = deck_text(generate_and_download())
    assert ("30 June 2026" in text or reporting in text), (
        "the deck does not state the dashboard's reporting date")


def test_the_deck_names_the_tenant_it_was_generated_for(deployment):
    text = deck_text(generate_and_download())
    assert CLIENT.lower() in text.lower(), "the deck does not identify its client"


# --------------------------------------------------------------------------- #
# 5. THE SHARED MULTIDIMENSIONAL RESULT.
#
# Catches: the cross-tab going back to being PPTX-only.
# --------------------------------------------------------------------------- #

def test_react_can_reach_the_multidimensional_analysis(deployment):
    payload = react("/mi/multidim")
    assert payload["available"] is True, payload.get("reason")
    assert payload["pairs"], "no cross-tab resolved for a book that supports one"


def test_the_deck_and_react_cross_tab_share_axes_and_totals(deployment):
    payload = react("/mi/multidim")
    pair_key = next(iter(payload["pairs"]))
    api_pair = payload["pairs"][pair_key]

    generate_and_download()
    heatmaps = {e["chart"]: e for e in sidecar(deployment)["rendered"]
                if e["kind"] == "heatmap"}
    drawn = heatmaps.get(f"multidim_{pair_key}")
    if drawn is None:              # composition may not have selected this pair
        pytest.skip(f"the deck did not draw {pair_key} for this book")
    assert drawn["categories"] == api_pair["xLabels"]
    assert drawn["rows"] == api_pair["yLabels"]


def test_the_cross_tab_axis_order_is_the_governed_ladder(deployment):
    pairs = react("/mi/multidim")["pairs"]
    ltv_pair = next((p for k, p in pairs.items() if p.get("xDimension") == "ltv"), None)
    assert ltv_pair, "no LTV-keyed cross-tab to check"
    labels = ltv_pair["xLabels"]
    assert labels == sorted(labels, key=_ltv_rank), (
        f"the cross-tab's LTV axis is out of the governed ladder: {labels}")


# --------------------------------------------------------------------------- #
# 6. RAG SEMANTICS.
# --------------------------------------------------------------------------- #

def test_rag_status_vocabulary_is_the_governed_one(deployment):
    """Whatever statuses the deck draws must be the approved vocabulary."""
    generate_and_download()
    approved = {"pass", "warning", "breach", "unavailable", ""}
    for entry in sidecar(deployment)["rendered"]:
        if entry["kind"] != "utilisation":
            continue
        for status in entry.get("statuses") or ():
            assert str(status).strip().lower() in approved, (
                f"the deck drew an ungoverned RAG status: {status!r}")
