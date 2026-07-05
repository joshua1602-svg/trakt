"""Tests for chart rendering, placeholder creation and straplines."""

from __future__ import annotations

from PIL import Image

from mi_agent_pptx.chart_resolver import ChartResolver
from mi_agent_pptx.data_resolver import resolve_data
from mi_agent_pptx.insight_resolver import StraplineResolver, MAX_WORDS
from mi_agent_pptx.metric_resolver import MetricResult
from mi_agent_pptx.placeholders import AppendixNotes, render_placeholder_png
from mi_agent_pptx.pptx_theme import THEME


def _panel_corner(path):
    return Image.open(path).convert("RGB").getpixel((3, 3))


def test_barlist_renders_at_requested_size(sample_tape, registries, tmp_path):
    rd = resolve_data(sample_tape, registries)
    cr = ChartResolver(rd, registries, tmp_path, lens="funded")
    res = cr.resolve({"id": "ltv", "title": "Balance by LTV", "type": "barlist",
                     "dimension": "current_loan_to_value", "bucket": "ltv_bucket",
                     "measure": "balance"}, 6.0, 4.5)
    assert res.ok
    assert res.path.exists()
    # Background matches the panel colour (no white box on navy).
    assert _panel_corner(res.path) == THEME.rgb(THEME.bg_panel)
    # Rendered at the requested aspect ratio (no distortion when placed).
    im = Image.open(res.path)
    assert abs(im.size[0] / im.size[1] - 6.0 / 4.5) < 0.02


def test_heatmap_renders(sample_tape, registries, tmp_path):
    rd = resolve_data(sample_tape, registries)
    cr = ChartResolver(rd, registries, tmp_path, lens="funded")
    res = cr.resolve({"id": "hm", "title": "LTV x Age", "type": "heatmap",
                     "dimension": "current_loan_to_value", "bucket": "ltv_bucket",
                     "dimension2": "youngest_borrower_age",
                     "bucket2": "borrower_age_bucket"}, 12.0, 4.8)
    assert res.ok
    assert res.path.exists()


def test_missing_dimension_produces_placeholder(sample_tape, registries, tmp_path):
    rd = resolve_data(sample_tape, registries)
    cr = ChartResolver(rd, registries, tmp_path, lens="funded")
    res = cr.resolve({"id": "broker", "title": "Broker", "type": "barlist",
                     "dimension": "broker_channel", "measure": "balance"}, 6.0, 4.5)
    assert not res.ok
    assert res.placeholder
    assert res.path.exists()
    assert "placeholder" in res.path.name


def test_lens_unavailable_resolver_none():
    # A None resolver (lens frame absent) is handled by the builder, but the
    # resolver itself must also degrade if constructed with empty data.
    from mi_agent_pptx.chart_resolver import ChartResolver as CR
    import pandas as pd
    from mi_agent_pptx.registry_loader import RegistryLoader
    import tempfile
    rd = None
    cr = CR(rd, RegistryLoader(), tempfile.mkdtemp(), lens="pipeline")
    res = cr.resolve({"id": "x", "title": "X", "type": "barlist",
                     "dimension": "pipeline_stage"}, 6.0, 4.5)
    assert res.placeholder


def test_render_placeholder_png(tmp_path):
    p = render_placeholder_png(tmp_path / "ph.png", "Title", "Message",
                               theme=THEME)
    assert p.exists()
    assert _panel_corner(p) == THEME.rgb(THEME.bg_panel)


def test_appendix_notes_dedupe():
    a = AppendixNotes()
    a.add("note one")
    a.add("note one")
    a.add("note two")
    assert a.as_list() == ["note one", "note two"]


# ----------------------------------------------------------------- straplines
def _m(key, label, value, fmt="currency"):
    from mi_agent_pptx.metric_resolver import format_value
    return MetricResult(key=key, label=label, value=value, fmt=fmt,
                        available=True, basis="registry_computed",
                        display=format_value(value, fmt))


def test_deterministic_strapline_from_metrics():
    metrics = {
        "funded_balance": _m("funded_balance", "Funded Balance", 5_400_000),
        "loan_count": _m("loan_count", "Loan Count", 36, "integer"),
        "wa_current_ltv": _m("wa_current_ltv", "WA Current LTV", 0.435, "percent"),
    }
    sr = StraplineResolver(metrics=metrics)
    text = sr.resolve("executive_summary", "kpi")
    assert "£5.4MM" in text
    assert len(text.split()) <= MAX_WORDS


def test_llm_artifact_strapline_wins():
    sr = StraplineResolver(metrics={}, llm_artifact={
        "straplines": {"cover": "Bespoke funder commentary supplied by MI Agent."}})
    assert sr.resolve("cover") == "Bespoke funder commentary supplied by MI Agent."


def test_strapline_never_empty_without_data():
    sr = StraplineResolver(metrics={})
    text = sr.resolve("pipeline_conversion", "charts")
    assert text and isinstance(text, str)
    assert len(text.split()) <= MAX_WORDS


def test_waterfall_bridge_renders_from_explicit_rows(sample_tape, registries, tmp_path):
    rd = resolve_data(sample_tape, registries)
    cr = ChartResolver(rd, registries, tmp_path, lens="funded")
    rows = [
        {"label": "2025-10", "value": 600000, "type": "total"},
        {"label": "South East", "value": 300000, "type": "delta"},
        {"label": "Wales", "value": -100000, "type": "delta"},
        {"label": "2026-03 (latest)", "value": 800000, "type": "total"},
    ]
    res = cr.resolve({"id": "bridge", "title": "Balance bridge", "type": "waterfall",
                      "rows": rows}, 8.0, 3.6)
    assert res.ok and res.path.exists()
    assert res.kind == "waterfall"
    assert _panel_corner(res.path) == THEME.rgb(THEME.bg_panel)
    im = Image.open(res.path)
    assert abs(im.size[0] / im.size[1] - 8.0 / 3.6) < 0.02


def test_waterfall_buildup_from_dimension(sample_tape, registries, tmp_path):
    rd = resolve_data(sample_tape, registries)
    cr = ChartResolver(rd, registries, tmp_path, lens="funded")
    # No explicit rows → within-run build-up decomposed by a present dimension.
    res = cr.resolve({"id": "buildup", "title": "Balance by LTV build-up",
                      "type": "waterfall", "dimension": "current_loan_to_value",
                      "bucket": "ltv_bucket", "measure": "balance"}, 8.0, 3.6)
    assert res.ok and res.path.exists()


def test_waterfall_missing_dimension_placeholder(sample_tape, registries, tmp_path):
    rd = resolve_data(sample_tape, registries)
    cr = ChartResolver(rd, registries, tmp_path, lens="funded")
    res = cr.resolve({"id": "wf_missing", "title": "Bridge", "type": "waterfall",
                      "dimension": "broker_channel", "measure": "balance"}, 8.0, 3.6)
    assert not res.ok and res.placeholder and res.path.exists()
