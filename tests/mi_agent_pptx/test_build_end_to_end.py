"""End-to-end deck build + validation tests (acceptance criteria 1-11)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pptx import Presentation
from pptx.util import Emu

from mi_agent_pptx.cli import run as cli_run
from mi_agent_pptx.deck_config import load_deck_config
from mi_agent_pptx.validation import (MAX_SLIDES, MIN_SLIDES,
                                      validate_deck_config)


def test_deck_config_slide_count_within_range(deck_config_path):
    cfg = load_deck_config(deck_config_path)
    assert MIN_SLIDES <= cfg.slide_count <= MAX_SLIDES
    report = validate_deck_config(cfg)
    assert report.ok, report.errors


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
    # Every slide carries at least a title + strapline (>= 2 text frames).
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
    # rc may be non-zero only if build validation fails; deck must still exist.
    assert out.exists()
    prs = Presentation(str(out))
    assert MIN_SLIDES <= len(list(prs.slides)) <= MAX_SLIDES


def test_broker_suppressed_at_consolidated(run_dir, deck_config_path, tmp_path):
    """Broker charts must be suppressed (placeholder) at consolidated funded."""
    out = tmp_path / "consolidated.pptx"
    work = tmp_path / "charts"
    rc = cli_run([
        "--run-dir", str(run_dir),
        "--deck-config", str(deck_config_path),
        "--client-name", "Consolidated Co",
        "--consolidated",
        "--work-dir", str(work),
        "--output", str(out),
    ])
    assert out.exists()
    # A broker chart should have been written as a suppressed placeholder.
    suppressed = list(Path(work).glob("*_suppressed.png"))
    assert suppressed, "expected a broker suppression placeholder"


def test_straplines_populated_on_every_slide(run_dir, deck_config_path, tmp_path):
    from mi_agent_pptx.artifact_loader import load_run_artifacts
    from mi_agent_pptx.chart_resolver import ChartResolver
    from mi_agent_pptx.data_resolver import resolve_data
    from mi_agent_pptx.insight_resolver import StraplineResolver
    from mi_agent_pptx.metric_resolver import MetricResolver
    from mi_agent_pptx.placeholders import AppendixNotes
    from mi_agent_pptx.pptx_builder import BuildContext, DeckBuilder
    from mi_agent_pptx.registry_loader import RegistryLoader

    reg = RegistryLoader()
    art = load_run_artifacts(run_dir)
    rd = resolve_data(art.tape, reg, as_of_date="2026-01-31")
    cfg = load_deck_config(deck_config_path)
    mr = MetricResolver(rd, reg)
    metrics = {k: mr.resolve(cfg.metric_spec(k)) for k in cfg.metrics}
    sr = StraplineResolver(metrics=metrics)
    cr = ChartResolver(rd, reg, tmp_path / "c")
    ctx = BuildContext(client_name="X", as_of_date="2026-01-31",
                       run_dir=str(run_dir))
    builder = DeckBuilder(cfg, ctx, mr, cr, sr, AppendixNotes())
    report = builder.build(tmp_path / "d.pptx")
    for rec in report["slides"]:
        assert rec["strapline"], f"slide {rec['id']} missing strapline"
    assert report["coverage_notes"]  # missing broker/pipeline recorded
