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


def test_bar_chart_renders(sample_tape, registries, tmp_path):
    rd = resolve_data(sample_tape, registries)
    cr = ChartResolver(rd, registries, tmp_path)
    res = cr.resolve({"id": "ltv", "title": "Balance by LTV", "type": "bar",
                     "dimension": "current_loan_to_value", "bucket": "ltv_bucket",
                     "measure": "balance"})
    assert res.ok
    assert res.path.exists()
    # Background matches the panel colour (no white box on navy).
    assert _panel_corner(res.path) == THEME.rgb(THEME.bg_panel)


def test_heatmap_renders(sample_tape, registries, tmp_path):
    rd = resolve_data(sample_tape, registries)
    cr = ChartResolver(rd, registries, tmp_path)
    res = cr.resolve({"id": "hm", "title": "LTV x Age", "type": "heatmap",
                     "dimension": "current_loan_to_value", "bucket": "ltv_bucket",
                     "dimension2": "youngest_borrower_age",
                     "bucket2": "borrower_age_bucket"})
    assert res.ok
    assert res.path.exists()


def test_missing_dimension_produces_placeholder(sample_tape, registries, tmp_path):
    rd = resolve_data(sample_tape, registries)
    cr = ChartResolver(rd, registries, tmp_path)
    res = cr.resolve({"id": "broker", "title": "Broker", "type": "bar",
                     "dimension": "broker_channel", "measure": "balance"})
    assert not res.ok
    assert res.placeholder
    assert res.path.exists()
    assert "placeholder" in res.path.name


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
    assert "£5.4m" in text
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
