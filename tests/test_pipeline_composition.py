#!/usr/bin/env python3
"""tests/test_pipeline_composition.py

The pipeline pages, on the pack a funder receives.

Two defects motivate these. The pipeline had no four-panel stratification page
at all on the representative book — it was gated on the pipeline being large
relative to the FUNDED book, so a small pipeline's shape could not be shown
even though shape is what a funder asks about origination whatever its size.
And Pipeline Overview's second chart was broker/channel, which on a
direct-only book drew one bar labelled "Direct": the pipeline total from the
tile above it, redrawn as a chart.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("pandas")
pptx = pytest.importorskip("pptx")


def _page(content: bytes, title: str):
    deck = pptx.Presentation(io.BytesIO(content))
    for slide in deck.slides:
        lines = [sh.text_frame.text.strip() for sh in slide.shapes
                 if sh.has_text_frame and sh.text_frame.text.strip()]
        if lines and title in lines[0]:
            return lines
    return None


# --------------------------------------------------------------------------- #
# The selection rule, at the level the pages consume it.
# --------------------------------------------------------------------------- #

def test_the_preferred_order_yields_to_informativeness():
    """Catches: broker/channel holding a panel at 100% Direct while a
    dimension that distributes goes undrawn."""
    from mi_agent_api import presentation as P

    strats = [
        {"key": "broker", "label": "By broker / channel",
         "bars": [{"label": "Direct", "balance": 7_800_000.0}]},
        {"key": "ltv", "label": "By LTV band",
         "bars": [{"label": "20-30%", "balance": 3.0},
                  {"label": "30-40%", "balance": 2.0},
                  {"label": "40-50%", "balance": 2.0}]},
        {"key": "region", "label": "By region",
         "bars": [{"label": "London", "balance": 4.0},
                  {"label": "Wales", "balance": 3.0}]},
    ]
    chosen = P.select_dimensions(strats, want=2, value_key="balance",
                                 preferred=("broker", "ltv", "region"))
    keys = [e["key"] for e in chosen["selected"]]
    assert keys == ["ltv", "region"], keys
    assert any(r["key"] == "broker" for r in chosen["rejected"])


def test_a_book_where_nothing_distributes_still_gets_a_page():
    """On a book where every dimension is concentrated, that IS the finding —
    an empty page is not an improvement on a flat one."""
    from mi_agent_api import presentation as P

    flat = [{"key": k, "label": k, "bars": [{"label": "One", "balance": 10.0}]}
            for k in ("ltv", "region")]
    chosen = P.select_dimensions(flat, want=4, value_key="balance",
                                 preferred=("ltv", "region"))
    assert chosen["selected"] == []
    assert len(chosen["rejected"]) == 2


# --------------------------------------------------------------------------- #
# On the rendered pack.
# --------------------------------------------------------------------------- #

def test_the_pipeline_gets_its_own_four_panel_page(qa_deck):
    """Catches: a pipeline section with no stratification page at all."""
    lines = _page(qa_deck, "Pipeline Stratifications")
    assert lines, "no Pipeline Stratifications page in the pack"
    panels = [l for l in lines if l.startswith("By ") and "single band" not in l]
    assert len(panels) == 4, panels


def test_the_pipeline_panels_are_the_informative_dimensions(qa_deck):
    lines = _page(qa_deck, "Pipeline Stratifications")
    panels = {l for l in lines if l.startswith("By ") and "single band" not in l}
    assert "By broker / channel" not in panels, panels


def test_a_dropped_pipeline_dimension_says_why_on_the_page(qa_deck):
    """Nothing is dropped silently: the page names what it did not chart."""
    lines = _page(qa_deck, "Pipeline Stratifications")
    notes = [l for l in lines if "single band" in l]
    assert notes, lines
    assert "broker" in notes[0].lower(), notes


def test_the_pipeline_overview_second_chart_is_not_one_bar(qa_deck):
    """Catches: "Pipeline amount by broker / channel — Direct £7.8MM"."""
    lines = _page(qa_deck, "Pipeline Overview")
    assert lines, "no Pipeline Overview page in the pack"
    titles = [l for l in lines if l.startswith("Pipeline amount by")]
    assert len(titles) == 2, titles
    assert "broker" not in titles[1].lower(), titles


@pytest.fixture(scope="module")
def qa_deck(tmp_path_factory):
    """A multi-book growing pack WITH a governed pipeline, through the real
    route: ``POST /mi/decks/generate`` -> poll -> ``GET /mi/decks/download``.

    The book writers come from the visual-QA harness, which is the same
    representative fixture the sprint reviews by eye — so what these tests
    assert and what a reader looks at are the same pack.
    """
    import os
    import sys as _sys
    import time

    _sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    import pptx_visual_qa as Q

    tmp = tmp_path_factory.mktemp("pipecomp")
    client = "pipecomp1"
    root = Q.write_book(tmp / "runs", client, "multi_growing")
    Q.write_pipeline(root, client)
    config = tmp / "client.yaml"
    config.write_text("portfolio:\n  base_currency: GBP\n", encoding="utf-8")
    # The environment is RESTORED afterwards. A module-scoped fixture that
    # leaves a deck root and a client id set changes what every later test in
    # the run resolves, which is how a passing test starts failing only in
    # company.
    _saved = dict(os.environ)
    os.environ.update({
        "MI_AGENT_ONBOARDING_OUTPUT_ROOT": str(root),
        "MI_AGENT_CLIENT_ID": client,
        "MI_AGENT_PIPELINE_ROOT": str(root),
        "TRAKT_LOCAL_BLOB_ROOT": str(tmp / "blob"),
        "TRAKT_INVESTOR_PPTX_PERSIST": "true",
        "TRAKT_INVESTOR_PPTX_ON_DEMAND": "true",
        "MI_AGENT_AUTH_ENABLED": "false",
        "TRAKT_MI_CLIENT_CONFIG": str(config),
        "TRAKT_STORAGE_BACKEND": "file",
        "TRAKT_RUNTIME_MODE": "test",
    })
    for key in ("MI_AGENT_DECK_ROOT", "AZURE_STORAGE_CONNECTION_STRING",
                "TRAKT_BLOB_CONNECTION"):
        os.environ.pop(key, None)

    from mi_agent_api import currency as currency_mod, data_source, datasets
    from mi_agent_api import deck_generation
    currency_mod._load_client_config.cache_clear()
    datasets._CLIENT_CURRENCY_CACHE.clear()
    data_source.reset_cache()
    deck_generation.reset_jobs()

    from fastapi.testclient import TestClient
    from mi_agent_api.app import app

    api = TestClient(app)
    accepted = api.post("/mi/decks/generate", json={})
    assert accepted.status_code == 202, accepted.text
    job = accepted.json()["jobId"]
    deadline, body = time.time() + 420, None
    while time.time() < deadline:
        body = api.get(f"/mi/decks/generate/{job}").json()
        if body["state"] in ("completed", "blocked", "failed"):
            break
        time.sleep(0.4)
    assert body and body["state"] == "completed", body
    got = api.get("/mi/decks/download")
    assert got.status_code == 200
    yield got.content
    os.environ.clear()
    os.environ.update(_saved)
    currency_mod._load_client_config.cache_clear()
    datasets._CLIENT_CURRENCY_CACHE.clear()
    data_source.reset_cache()
    deck_generation.reset_jobs()
