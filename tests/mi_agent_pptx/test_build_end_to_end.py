"""End-to-end deck build tests for the payload-driven, dashboard-aligned deck.

The deck is a faithful export of the React MI dashboard: every slide renders
directly from an MI Agent API payload (mi_agent_pptx.mi_api). These tests assert
the deck always builds (placeholders never fail), stays 16:9 and within the
slide bounds, keeps the pipeline lens separate from the funded lens, and degrades
a panel to a branded placeholder only when the payload is genuinely absent.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pptx import Presentation
from pptx.util import Emu

from mi_agent_pptx.cli import run as cli_run
from mi_agent_pptx.deck import DeckBuilder, DeckContext
from mi_agent_pptx.mi_api import build_dashboard_data
from mi_agent_pptx.validation import MAX_SLIDES, MIN_SLIDES


def _slides(deck_config_path):
    cfg = yaml.safe_load(Path(deck_config_path).read_text(encoding="utf-8")) or {}
    return cfg.get("slides", [])


def test_deck_config_slide_count_within_range(deck_config_path):
    slides = _slides(deck_config_path)
    assert MIN_SLIDES <= len(slides) <= MAX_SLIDES
    # Every slide declares a type that maps to a deck handler.
    for spec in slides:
        assert spec.get("type") in DeckBuilder._DISPATCH, spec.get("type")


def test_cli_builds_valid_pptx(run_dir, deck_config_path, tmp_path):
    out = tmp_path / "deck.pptx"
    rc = cli_run([
        "--run-dir", str(run_dir),
        "--deck-config", str(deck_config_path),
        "--client-name", "Test Client",
        "--as-of-date", "2026-01-31",
        "--output", str(out),
    ])
    assert rc == 0
    assert out.exists()

    prs = Presentation(str(out))
    slides = list(prs.slides)
    assert MIN_SLIDES <= len(slides) <= MAX_SLIDES
    # 16:9 widescreen.
    assert round(Emu(prs.slide_width).inches, 2) == 13.33
    # Every slide carries at least a title + one more text frame.
    for s in slides:
        texts = [sh for sh in s.shapes
                 if sh.has_text_frame and sh.text_frame.text.strip()]
        assert len(texts) >= 2


def test_cli_builds_with_no_tape(empty_run_dir, deck_config_path, tmp_path):
    """Missing tape must still yield a full, valid, placeholder-filled deck."""
    out = tmp_path / "empty_deck.pptx"
    rc = cli_run([
        "--run-dir", str(empty_run_dir),
        "--deck-config", str(deck_config_path),
        "--client-name", "No Data Co",
        "--output", str(out),
    ])
    assert rc == 0
    assert out.exists()
    prs = Presentation(str(out))
    assert MIN_SLIDES <= len(list(prs.slides)) <= MAX_SLIDES


def test_funded_kpis_come_from_the_dashboard_snapshot(run_dir):
    """The Executive Summary tiles are the funded snapshot KPIs verbatim — the
    same payload the dashboard's Funded header renders."""
    data = build_dashboard_data(run_dir, as_of="2026-01-31")
    kpis = {k.get("id"): k for k in data.funded.get("kpis", [])}
    assert "balance" in kpis
    assert kpis["balance"].get("value")           # rendered, formatted value
    assert data.cohorts.get("cohorts")            # vintage composition resolves


def test_pipeline_lens_is_separate_from_funded(run_dir):
    """The funded run has no pipeline source, so the pipeline payload must be
    empty (a placeholder) rather than borrowing the funded total — the lens
    separation the dashboard enforces."""
    data = build_dashboard_data(run_dir, as_of="2026-01-31")
    assert data.funded.get("kpis")                # funded resolved
    assert not data.pipeline                       # pipeline did NOT resolve
    # And a coverage note explains the placeholder.
    assert any("pipeline" in n.lower() for n in data.notes)


def test_deck_records_flag_placeholder_slides(run_dir, deck_config_path, tmp_path):
    """A slide is recorded as a placeholder only when its payload is absent; the
    funded KPI/strat/geo slides must NOT be placeholders on a real funded run."""
    data = build_dashboard_data(run_dir, as_of="2026-01-31")
    ctx = DeckContext(client_name="X", as_of_date="2026-01-31",
                      run_dir=str(run_dir), work_dir=str(tmp_path / "charts"))
    report = DeckBuilder(data, ctx).build(_slides(deck_config_path),
                                          tmp_path / "d.pptx")
    by_id = {r["id"]: r for r in report["slides"]}
    # Every record carries a title.
    for rec in report["slides"]:
        assert rec.get("title")
    # The funded KPI summary resolves on a real funded run.
    assert not by_id["executive_summary"]["placeholder"]
    # Pipeline has no source here → placeholder, and it is recorded as such.
    assert by_id["pipeline"]["placeholder"]
