"""mi_agent_pptx.pptx_builder — assemble the institutional investor PPTX pack.

Assembles a 16:9 widescreen deck with python-pptx from the resolved metrics,
rendered charts and straplines. The visual system mirrors the MI Agent React
dark dashboard (navy surfaces, periwinkle accents, Inter typography, tabular
figures). Every content slide has a title, a strapline, a footer and a page
number; charts sit on a panel that matches their rendered background so there
are no white boxes; missing content is rendered as a branded placeholder and
recorded as an appendix coverage note.

The builder is driven entirely by the :class:`~mi_agent_pptx.deck_config.DeckConfig`
— it renders whatever slides the config declares and never invents analytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

from .chart_resolver import ChartResolver, ChartResult
from .deck_config import DeckConfig, SlideSpec
from .insight_resolver import StraplineResolver
from .metric_resolver import MetricResolver, MetricResult
from .placeholders import AppendixNotes
from .pptx_theme import PptxTheme, THEME

# 16:9 widescreen.
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

BROKER_DIMENSIONS = {"broker_channel", "broker"}


@dataclass
class BuildContext:
    """Runtime context threaded through the build."""

    client_name: str
    as_of_date: str
    run_dir: str
    lens: str = "total"
    consolidated: bool = False
    generated_by: str = "trakt MI Agent"
    logo_path: Optional[str] = None
    source_artifacts: List[str] = field(default_factory=list)


class DeckBuilder:
    """Assemble the deck from config + resolvers."""

    def __init__(
        self,
        config: DeckConfig,
        context: BuildContext,
        metric_resolver: MetricResolver,
        chart_resolver: ChartResolver,
        strapline_resolver: StraplineResolver,
        appendix: AppendixNotes,
        theme: PptxTheme = THEME,
    ):
        self.config = config
        self.ctx = context
        self.metrics = metric_resolver
        self.charts = chart_resolver
        self.straplines = strapline_resolver
        self.appendix = appendix
        self.theme = theme
        self.prs = Presentation()
        self.prs.slide_width = SLIDE_W
        self.prs.slide_height = SLIDE_H
        self._blank = self.prs.slide_layouts[6]
        self._page = 0
        self.slide_records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ colours
    def _rgb(self, hex_str: str) -> RGBColor:
        r, g, b = self.theme.rgb(hex_str)
        return RGBColor(r, g, b)

    # ------------------------------------------------------------------- public
    def build(self, output_path: str | Path) -> Dict[str, Any]:
        for slide_spec in self.config.slides:
            handler = getattr(self, f"_slide_{slide_spec.type}", self._slide_charts)
            record = handler(slide_spec)
            self.slide_records.append(record)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(out))
        return {
            "output": str(out),
            "slides": self.slide_records,
            "coverage_notes": self.appendix.as_list(),
        }

    # ------------------------------------------------------------- slide scaffold
    def _new_slide(self):
        slide = self.prs.slides.add_slide(self._blank)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = self._rgb(self.theme.bg_page)
        return slide

    def _add_text(self, slide, left, top, width, height, text, *, size=14,
                  color=None, bold=False, align=PP_ALIGN.LEFT, italic=False,
                  anchor=MSO_ANCHOR.TOP, font=None):
        box = slide.shapes.add_textbox(left, top, width, height)
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = 0
        tf.margin_right = 0
        tf.margin_top = 0
        tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = align
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.name = font or self.theme.font_sans
        run.font.color.rgb = self._rgb(color or self.theme.ink_100)
        return box

    def _add_panel(self, slide, left, top, width, height, *, fill=None,
                   line=None, radius=True):
        shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
        shape = slide.shapes.add_shape(shape_type, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = self._rgb(fill or self.theme.bg_panel)
        shape.line.color.rgb = self._rgb(line or self.theme.line)
        shape.line.width = Pt(0.75)
        shape.shadow.inherit = False
        return shape

    def _header(self, slide, title: str, strapline: str):
        # Navy accent rail on the left.
        rail = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.0),
                                      Inches(0.0), Inches(0.14), SLIDE_H)
        rail.fill.solid()
        rail.fill.fore_color.rgb = self._rgb(self.theme.peri)
        rail.line.fill.background()
        rail.shadow.inherit = False

        self._add_text(slide, Inches(0.55), Inches(0.33), Inches(12.2),
                       Inches(0.7), title, size=26, bold=True,
                       color=self.theme.ink_100)
        self._add_text(slide, Inches(0.57), Inches(1.02), Inches(12.2),
                       Inches(0.5), strapline, size=12.5,
                       color=self.theme.peri, italic=True)

    def _footer(self, slide):
        self._page += 1
        self._add_text(slide, Inches(0.55), Inches(7.06), Inches(9.0),
                       Inches(0.3), self.config.footer, size=8.5,
                       color=self.theme.ink_500)
        self._add_text(slide, Inches(12.2), Inches(7.06), Inches(0.9),
                       Inches(0.3), str(self._page), size=8.5,
                       color=self.theme.ink_500, align=PP_ALIGN.RIGHT)

    def _strapline_for(self, spec: SlideSpec) -> str:
        text = self.straplines.resolve(spec.id, spec.type)
        return text

    def _record(self, spec: SlideSpec, strapline: str, *, placeholder=False,
                extra=None) -> Dict[str, Any]:
        rec = {
            "id": spec.id,
            "type": spec.type,
            "title": spec.title,
            "strapline": strapline,
            "mandatory": spec.mandatory,
            "placeholder": placeholder,
        }
        if extra:
            rec.update(extra)
        return rec

    # ------------------------------------------------------------------ COVER
    def _slide_cover(self, spec: SlideSpec) -> Dict[str, Any]:
        slide = self._new_slide()
        # Full navy cover with periwinkle band.
        band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0),
                                      Inches(2.7), SLIDE_W, Inches(0.06))
        band.fill.solid()
        band.fill.fore_color.rgb = self._rgb(self.theme.peri)
        band.line.fill.background()
        band.shadow.inherit = False

        self._add_text(slide, Inches(0.9), Inches(1.5), Inches(11.5),
                       Inches(1.1), self.ctx.client_name, size=40, bold=True,
                       color=self.theme.ink_100)
        self._add_text(slide, Inches(0.92), Inches(2.85), Inches(11.5),
                       Inches(0.6), self.config.name, size=20,
                       color=self.theme.peri)
        strap = self._strapline_for(spec)
        self._add_text(slide, Inches(0.92), Inches(3.6), Inches(11.0),
                       Inches(0.6), strap, size=13, color=self.theme.ink_300,
                       italic=True)
        self._add_text(slide, Inches(0.92), Inches(5.4), Inches(11.0),
                       Inches(0.4),
                       f"Data cut-off: {self.ctx.as_of_date or 'n/a'}",
                       size=13, color=self.theme.ink_100)
        self._add_text(slide, Inches(0.92), Inches(5.85), Inches(11.0),
                       Inches(0.4), f"Generated by {self.ctx.generated_by}",
                       size=12, color=self.theme.ink_400)

        if self.ctx.logo_path and Path(self.ctx.logo_path).exists():
            try:
                slide.shapes.add_picture(self.ctx.logo_path, Inches(10.8),
                                         Inches(0.6), height=Inches(0.9))
            except Exception:
                pass
        return self._record(spec, strap)

    # ------------------------------------------------------------------ KPI
    def _slide_kpi(self, spec: SlideSpec) -> Dict[str, Any]:
        slide = self._new_slide()
        strap = self._strapline_for(spec)
        self._header(slide, spec.title or "Executive Summary", strap)

        specs = self.config.metric_specs(spec.kpis)
        results: List[MetricResult] = self.metrics.resolve_all(specs)

        # Grid: up to 5 across, 2 rows.
        cols = 5
        tile_w = Inches(2.36)
        tile_h = Inches(1.5)
        gap_x = Inches(0.15)
        gap_y = Inches(0.28)
        left0 = Inches(0.55)
        top0 = Inches(1.75)
        n_placeholder = 0
        for i, res in enumerate(results):
            r = i // cols
            c = i % cols
            left = Emu(int(left0) + c * (int(tile_w) + int(gap_x)))
            top = Emu(int(top0) + r * (int(tile_h) + int(gap_y)))
            self._kpi_tile(slide, left, top, tile_w, tile_h, res)
            if not res.ok:
                n_placeholder += 1
                self.appendix.add(
                    f"Executive KPI '{res.label}' unavailable: {res.note}")

        self._footer(slide)
        return self._record(spec, strap, placeholder=(n_placeholder == len(results)),
                            extra={"kpis_missing": n_placeholder})

    def _kpi_tile(self, slide, left, top, width, height, res: MetricResult):
        panel = self._add_panel(slide, left, top, width, height,
                                fill=self.theme.bg_panel_alt,
                                line=self.theme.line)
        # Value.
        val_color = self.theme.ink_100 if res.ok else self.theme.ink_500
        self._add_text(slide, left + Inches(0.16), top + Inches(0.22),
                       width - Inches(0.3), Inches(0.7), res.display,
                       size=22, bold=True, color=val_color)
        # Label.
        self._add_text(slide, left + Inches(0.16), top + Inches(0.98),
                       width - Inches(0.3), Inches(0.4), res.label.upper(),
                       size=9, color=self.theme.ink_400)

    # --------------------------------------------------------------- CHART SLIDE
    def _slide_charts(self, spec: SlideSpec) -> Dict[str, Any]:
        slide = self._new_slide()
        strap = self._strapline_for(spec)
        self._header(slide, spec.title, strap)

        chart_specs = list(spec.charts)
        rendered: List[ChartResult] = []
        for cspec in chart_specs:
            rendered.append(self._resolve_chart(spec, cspec))

        self._place_charts(slide, rendered)
        self._footer(slide)

        all_placeholder = bool(rendered) and all(c.placeholder for c in rendered)
        for c in rendered:
            if c.placeholder:
                self.appendix.add(f"[{spec.title}] {c.title}: {c.note}")
        return self._record(spec, strap, placeholder=all_placeholder,
                            extra={"charts": len(rendered)})

    def _resolve_chart(self, slide_spec: SlideSpec, cspec: Dict[str, Any]) -> ChartResult:
        title = cspec.get("title", cspec.get("id", "chart").replace("_", " ").title())
        cid = cspec.get("id", "chart")
        dim = cspec.get("dimension")
        # Broker suppression at consolidated funded level (lens-aware).
        is_broker = (dim in BROKER_DIMENSIONS) or bool(cspec.get("broker"))
        suppress = (self.config.suppress_broker_consolidated
                    or slide_spec.suppress_broker_consolidated)
        if is_broker and suppress and self.ctx.consolidated:
            note = ("Broker channel suppressed at consolidated funded level — "
                    "acquired portfolios do not carry broker data.")
            self.appendix.add(f"[{slide_spec.title}] {title}: {note}")
            from .placeholders import render_placeholder_png
            path = self.charts.out_dir / f"{cid}_suppressed.png"
            render_placeholder_png(path, title, "Broker channel suppressed "
                                   "(consolidated funded lens)", theme=self.theme)
            return ChartResult(chart_id=cid, title=title, path=path,
                               available=False, placeholder=True, kind="suppressed",
                               note=note)
        return self.charts.resolve(cspec)

    def _place_charts(self, slide, rendered: List[ChartResult]):
        if not rendered:
            return
        n = len(rendered)
        top = Inches(1.7)
        if n == 1:
            self._chart_panel(slide, rendered[0], Inches(0.55), top,
                              Inches(12.2), Inches(4.9))
        else:
            # Two charts side by side (simple charts / two-up layout).
            self._chart_panel(slide, rendered[0], Inches(0.55), top,
                              Inches(6.0), Inches(4.9))
            self._chart_panel(slide, rendered[1], Inches(6.78), top,
                              Inches(6.0), Inches(4.9))

    def _chart_panel(self, slide, chart: ChartResult, left, top, width, height):
        # Panel matches the chart PNG background -> no white box on navy.
        title_h = Inches(0.42)
        self._add_panel(slide, left, top, width, height,
                        fill=self.theme.bg_panel, line=self.theme.line)
        # Title banner.
        banner = self._add_panel(slide, left, top, width, title_h,
                                 fill=self.theme.navy, line=self.theme.navy)
        banner.shadow.inherit = False
        self._add_text(slide, left + Inches(0.16), top + Inches(0.05),
                       width - Inches(0.3), title_h, chart.title, size=11.5,
                       bold=True, color=self.theme.ink_100)
        if chart.path and Path(chart.path).exists():
            img_top = top + title_h + Inches(0.08)
            img_h = height - title_h - Inches(0.2)
            try:
                slide.shapes.add_picture(str(chart.path), left + Inches(0.12),
                                         img_top, width=width - Inches(0.24),
                                         height=img_h)
            except Exception:
                pass

    # -------------------------------------------------------------- RISK MONITOR
    def _slide_risk_monitor(self, spec: SlideSpec) -> Dict[str, Any]:
        slide = self._new_slide()
        strap = self._strapline_for(spec)
        self._header(slide, spec.title or "Risk Monitor", strap)

        risk = self.metrics.analytics.get("risk_monitor") if self.metrics.analytics else None
        placeholder = False
        if isinstance(risk, dict) and risk.get("tests"):
            self._risk_rag_tiles(slide, risk)
        else:
            placeholder = True
            from .placeholders import render_placeholder_png
            path = self.charts.out_dir / "risk_monitor_placeholder.png"
            render_placeholder_png(
                path, "Risk Monitor",
                "Risk limit artifacts not present for this run — "
                "concentration monitor pending", theme=self.theme,
                width_in=11.5)
            slide.shapes.add_picture(str(path), Inches(0.9), Inches(2.0),
                                     width=Inches(11.5))
            self.appendix.add(
                "Risk monitor rendered as placeholder: no risk-limit artifact "
                "in run directory (v1 acceptable).")
        self._footer(slide)
        return self._record(spec, strap, placeholder=placeholder)

    def _risk_rag_tiles(self, slide, risk: Dict[str, Any]):
        summary = risk.get("summary", {}) or {}
        tiles = [
            ("Limits", summary.get("total", "—"), self.theme.navy),
            ("Breaches", summary.get("breaches", 0), self.theme.rag["red"]),
            ("Warnings", summary.get("warnings", 0), self.theme.rag["amber"]),
            ("Within limit", summary.get("testsPassed", 0), self.theme.rag["green"]),
        ]
        tile_w = Inches(2.9)
        for i, (label, value, color) in enumerate(tiles):
            left = Emu(int(Inches(0.55)) + i * int(Inches(3.05)))
            panel = self._add_panel(slide, left, Inches(1.85), tile_w,
                                    Inches(1.4), fill=self.theme.bg_panel_alt,
                                    line=color)
            self._add_text(slide, left + Inches(0.2), Inches(2.05), tile_w,
                           Inches(0.7), str(value), size=26, bold=True,
                           color=color)
            self._add_text(slide, left + Inches(0.2), Inches(2.85), tile_w,
                           Inches(0.4), label.upper(), size=10,
                           color=self.theme.ink_400)
        # Observations list.
        obs = risk.get("observations", []) or []
        y = 3.5
        for line in obs[:5]:
            self._add_text(slide, Inches(0.6), Inches(y), Inches(12.0),
                           Inches(0.35), f"•  {line}", size=12,
                           color=self.theme.ink_300)
            y += 0.42

    # -------------------------------------------------------------- METHODOLOGY
    def _slide_methodology(self, spec: SlideSpec) -> Dict[str, Any]:
        slide = self._new_slide()
        strap = self._strapline_for(spec)
        self._header(slide, spec.title or "Methodology & Notes", strap)

        lines = [
            f"Client: {self.ctx.client_name}",
            f"Data cut-off date: {self.ctx.as_of_date or 'n/a'}",
            f"Run directory: {self.ctx.run_dir}",
            f"Lens basis: {self.ctx.lens}"
            + (" (consolidated funded)" if self.ctx.consolidated else ""),
            "Balance basis: current outstanding balance (registry-authorised).",
            "Forecast method: deterministic baseline (v1).",
            "Aggregations: MI Agent semantic registry + analytics_lib "
            "(config-driven buckets).",
        ]
        if self.ctx.source_artifacts:
            lines.append("Source artifacts:")
            lines.extend(f"   – {a}" for a in self.ctx.source_artifacts[:8])

        self._bullets(slide, lines, top=1.75, size=12.5)
        self._footer(slide)
        return self._record(spec, strap)

    # ------------------------------------------------------------------ APPENDIX
    def _slide_appendix(self, spec: SlideSpec) -> Dict[str, Any]:
        slide = self._new_slide()
        strap = self._strapline_for(spec)
        self._header(slide, spec.title or "Appendix — Data Coverage", strap)
        notes = self.appendix.as_list()
        if not notes:
            notes = ["All configured metrics and charts resolved from the "
                     "current pipeline run; no coverage gaps."]
        self._bullets(slide, [f"•  {n}" for n in notes[:12]], top=1.75, size=11)
        self._footer(slide)
        return self._record(spec, strap)

    def _bullets(self, slide, lines: List[str], *, top=1.75, size=12):
        box = slide.shapes.add_textbox(Inches(0.6), Inches(top), Inches(12.1),
                                       Inches(5.0))
        tf = box.text_frame
        tf.word_wrap = True
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            run = p.add_run()
            run.text = line
            run.font.size = Pt(size)
            run.font.name = self.theme.font_sans
            run.font.color.rgb = self._rgb(self.theme.ink_300)
            p.space_after = Pt(6)
