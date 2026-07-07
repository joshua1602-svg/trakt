"""mi_agent_pptx.deck — payload-driven deck assembly (dashboard-aligned).

Renders each slide directly from the MI API payloads (:mod:`mi_api`) so the pack
is a faithful export of the React dashboard: the Executive Summary is the funded
KPI tile grid, stratifications are the funded BarLists, the pipeline slide is the
pipeline snapshot, the forecast slide is the funded→forecast bridge, evolution
slides are the time series, geography/cohorts/risk mirror their tabs. Numbers are
taken verbatim from the payloads — never recomputed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

from .chart_resolver import render_bridge_waterfall
from .mi_api import DashboardData
from .placeholders import render_placeholder_png
from .pptx_theme import PptxTheme, THEME
from . import render as R

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
EMU_IN = 914400


@dataclass
class DeckContext:
    client_name: str
    as_of_date: str = ""
    run_dir: str = ""
    generated_by: str = "trakt MI Agent"
    footer: str = "trakt MI Agent · Confidential — for institutional funders/investors"
    deck_name: str = "Investor & Funder MI Pack"
    work_dir: str = ""
    logo_path: Optional[str] = None


class DeckBuilder:
    def __init__(self, data: DashboardData, ctx: DeckContext,
                 theme: PptxTheme = THEME):
        self.d = data
        self.ctx = ctx
        self.theme = theme
        self.prs = Presentation()
        self.prs.slide_width = SLIDE_W
        self.prs.slide_height = SLIDE_H
        self._blank = self.prs.slide_layouts[6]
        self._page = 0
        self.work = Path(ctx.work_dir or (Path(ctx.run_dir) / "_pptx_charts"))
        self.work.mkdir(parents=True, exist_ok=True)
        self.appendix: List[str] = list(data.notes)
        self.records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------- pptx scaffold
    def _rgb(self, hx):
        r, g, b = self.theme.rgb(hx)
        return RGBColor(r, g, b)

    def _slide(self):
        s = self.prs.slides.add_slide(self._blank)
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = self._rgb(self.theme.bg_page)
        return s

    def _text(self, slide, l, t, w, h, text, *, size=14, color=None, bold=False,
              align=PP_ALIGN.LEFT, italic=False, anchor=MSO_ANCHOR.TOP, spacing=None):
        box = slide.shapes.add_textbox(l, t, w, h)
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, 0)
        p = tf.paragraphs[0]
        p.alignment = align
        if spacing:
            p.line_spacing = spacing
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.name = self.theme.font_sans
        run.font.color.rgb = self._rgb(color or self.theme.ink_100)
        return box

    def _panel(self, slide, l, t, w, h, *, fill=None, line=None, radius=True, lw=1.0):
        shp = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, l, t, w, h)
        try:
            shp.adjustments[0] = 0.045
        except Exception:
            pass
        shp.fill.solid()
        shp.fill.fore_color.rgb = self._rgb(fill or self.theme.bg_panel)
        if line is None:
            shp.line.fill.background()
        else:
            shp.line.color.rgb = self._rgb(line)
            shp.line.width = Pt(lw)
        shp.shadow.inherit = False
        return shp

    def _header(self, slide, title, strap, *, accent=None):
        rail = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                                      Inches(0.12), SLIDE_H)
        rail.fill.solid()
        rail.fill.fore_color.rgb = self._rgb(accent or self.theme.peri)
        rail.line.fill.background()
        rail.shadow.inherit = False
        self._text(slide, Inches(0.55), Inches(0.34), Inches(12.2), Inches(0.6),
                   title, size=25, bold=True)
        if strap:
            self._text(slide, Inches(0.57), Inches(1.0), Inches(12.4), Inches(0.5),
                       strap, size=12, color=self.theme.peri, italic=True)

    def _footer(self, slide):
        self._page += 1
        self._text(slide, Inches(0.55), Inches(7.08), Inches(10), Inches(0.3),
                   self.ctx.footer, size=8, color=self.theme.ink_500)
        self._text(slide, Inches(12.3), Inches(7.08), Inches(0.8), Inches(0.3),
                   str(self._page), size=8, color=self.theme.ink_500,
                   align=PP_ALIGN.RIGHT)

    def _place(self, slide, path, l, t, w_in, h_in):
        try:
            slide.shapes.add_picture(str(path), l, t, width=Inches(w_in),
                                     height=Inches(h_in))
        except Exception:
            pass

    def _card(self, slide, l, t, w, h, title):
        """A dashboard-style card: panel + title, returns the inner image box."""
        self._panel(slide, l, t, w, h, fill=self.theme.bg_panel, line=self.theme.line)
        self._text(slide, l + Inches(0.22), t + Inches(0.16), w - Inches(0.4),
                   Inches(0.34), title, size=12.5, bold=True)
        img_l = l + Inches(0.16)
        img_t = t + Inches(0.62)
        img_w = (int(w) - 2 * int(Inches(0.16))) / EMU_IN
        img_h = (int(h) - int(Inches(0.62)) - int(Inches(0.16))) / EMU_IN
        return img_l, img_t, img_w, img_h

    # ------------------------------------------------------------------- tiles
    def _tile(self, slide, l, t, w, h, tile: Dict[str, Any]):
        self._panel(slide, l, t, w, h, fill=self.theme.bg_panel_alt,
                    line=self.theme.line_soft, lw=1.0)
        pad = Inches(0.16)
        iw = Emu(int(w) - 2 * int(pad))
        avail = bool(tile.get("available", True)) and tile.get("value") not in (None, "")
        self._text(slide, l + pad, t + Inches(0.14), iw, Inches(0.3),
                   str(tile.get("label", "")).upper(), size=8.5,
                   color=self.theme.ink_400, bold=True)
        val = tile.get("value") if avail else "—"
        self._text(slide, l + pad, t + Inches(0.44), iw, Inches(0.55), str(val),
                   size=20, bold=True,
                   color=self.theme.ink_100 if avail else self.theme.ink_500)
        y = t + Inches(1.02)
        delta, intent = tile.get("delta"), tile.get("deltaIntent")
        if delta:
            color = {"positive": self.theme.mint, "negative": self.theme.rose}.get(
                intent, self.theme.ink_400)
            arrow = {"positive": "▲ ", "negative": "▼ "}.get(intent, "")
            self._text(slide, l + pad, y, iw, Inches(0.3), f"{arrow}{delta}",
                       size=9.5, color=color, bold=True)
        elif tile.get("hint"):
            self._text(slide, l + pad, y, iw, Inches(0.3), str(tile["hint"]),
                       size=9, color=self.theme.ink_400)

    def _tile_grid(self, slide, tiles: List[Dict[str, Any]], *, top=1.62, cols=5):
        rows = max(1, (len(tiles) + cols - 1) // cols)
        gx, gy = Inches(0.16), Inches(0.22)
        left0, top0 = Inches(0.55), Inches(top)
        tile_w = Emu(int((int(Inches(12.25)) - (cols - 1) * int(gx)) / cols))
        tile_h = Inches(1.62) if rows <= 2 else Inches(1.3)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            l = Emu(int(left0) + c * (int(tile_w) + int(gx)))
            t = Emu(int(top0) + r * (int(tile_h) + int(gy)))
            self._tile(slide, l, t, tile_w, tile_h, tile)

    def _chart_boxes(self, n):
        top = Inches(1.62)
        h = Inches(4.95)
        if n <= 1:
            return [(Inches(0.55), top, Inches(12.25), h)]
        return [(Inches(0.55), top, Inches(6.0), h),
                (Inches(6.78), top, Inches(6.0), h)]

    def _barlist_card(self, slide, box, title, rows, value_key, *, currency=True,
                      cid="bl", label_key="label"):
        il, it, iw, ih = self._card(slide, *box, title)
        path = self.work / f"{cid}.png"
        if rows:
            R.draw_barlist(path, rows, value_key, iw, ih, theme=self.theme,
                           currency=currency, label_key=label_key)
        else:
            render_placeholder_png(path, "", "No data for this run",
                                   theme=self.theme, width_in=iw, height_in=ih)
        self._place(slide, path, il, it, iw, ih)
        return bool(rows)

    # =====================================================================
    # SLIDE HANDLERS
    # =====================================================================
    def _record(self, sid, title, strap, *, placeholder=False):
        self.records.append({"id": sid, "title": title, "strapline": strap,
                             "placeholder": placeholder})

    def slide_cover(self, spec):
        s = self._slide()
        # Full-height left accent rail (replaces the old overlapping corner panel).
        rail = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                                  Inches(0.16), SLIDE_H)
        rail.fill.solid()
        rail.fill.fore_color.rgb = self._rgb(self.theme.peri)
        rail.line.fill.background()
        rail.shadow.inherit = False
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(3.02),
                                 Inches(2.2), Inches(0.07))
        bar.fill.solid()
        bar.fill.fore_color.rgb = self._rgb(self.theme.peri)
        bar.line.fill.background()
        bar.shadow.inherit = False
        self._text(s, Inches(0.9), Inches(0.7), Inches(6), Inches(0.4),
                   "TRAKT · MI AGENT", size=12, color=self.theme.peri, bold=True)
        self._text(s, Inches(0.86), Inches(1.7), Inches(11.5), Inches(1.4),
                   self.ctx.client_name, size=42, bold=True)
        self._text(s, Inches(0.92), Inches(3.25), Inches(11), Inches(0.6),
                   self.ctx.deck_name, size=19, color=self.theme.peri)
        fb = (self.d.forecast.get("forecastBridge") or {})
        strap = self._cover_strapline()
        self._text(s, Inches(0.92), Inches(3.95), Inches(10.5), Inches(0.7),
                   strap, size=13, color=self.theme.ink_300, italic=True, spacing=1.15)
        self._text(s, Inches(0.92), Inches(5.7), Inches(6), Inches(0.4),
                   f"Data cut-off   {self.ctx.as_of_date or 'n/a'}", size=12.5)
        self._text(s, Inches(0.92), Inches(6.12), Inches(8), Inches(0.4),
                   f"Generated automatically by {self.ctx.generated_by}", size=11,
                   color=self.theme.ink_400)
        self._record("cover", self.ctx.client_name, strap)

    def _cover_strapline(self):
        kpis = {k.get("id"): k for k in self.d.funded.get("kpis", [])}
        bal = kpis.get("balance", {}).get("value")
        loans = kpis.get("loans", {}).get("value")
        ltv = kpis.get("wa_current_ltv", {}).get("value")
        if bal:
            return (f"Funded book of {bal} across {loans} loans at {ltv} weighted "
                    f"current LTV.")
        return "Automated MI pack generated from the latest pipeline run."

    def slide_kpi_summary(self, spec):
        s = self._slide()
        self._header(s, spec.get("title", "Executive Summary"),
                     "Funded book snapshot" + (
                         f" · reporting {self.d.reporting_date}" if self.d.reporting_date else ""))
        tiles = list(self.d.funded.get("kpis", []))[:10]
        if not tiles:
            self._placeholder_body(s, "Funded book unavailable for this run.")
            self._footer(s)
            return self._record("executive_summary", spec.get("title"), "", placeholder=True)
        self._tile_grid(s, tiles, top=1.62, cols=5)
        self._footer(s)
        self._record("executive_summary", spec.get("title", "Executive Summary"),
                     "Funded KPIs (dashboard-aligned).")

    def slide_strat(self, spec):
        s = self._slide()
        self._header(s, spec.get("title", "Funded Stratifications"),
                     "Balance by dimension", accent=self.theme.peri)
        strats = self.d.funded.get("stratifications", [])
        # up to two per slide (config may split); here show the first two.
        keys = spec.get("keys")
        if keys:
            strats = [st for st in strats if st.get("key") in keys]
        strats = strats[:2]
        boxes = self._chart_boxes(len(strats) or 1)
        ph = True
        for st, box in zip(strats, boxes):
            rows = st.get("bars", [])
            ok = self._barlist_card(s, box, st.get("label", st.get("key", "")), rows,
                                    "balance", cid=f"strat_{st.get('key')}")
            ph = ph and not ok
        if not strats:
            il, it, iw, ih = self._card(s, *boxes[0], "Stratifications")
            path = self.work / "strat_none.png"
            render_placeholder_png(path, "", "No funded stratifications for this run",
                                   theme=self.theme, width_in=iw, height_in=ih)
            self._place(s, path, il, it, iw, ih)
        self._footer(s)
        self._record(spec.get("id", "strat"), spec.get("title"), "", placeholder=ph)

    def slide_geo(self, spec):
        s = self._slide()
        self._header(s, spec.get("title", "Geographic Exposure"),
                     "Funded exposure by ITL3 area")
        geo = self.d.geo
        areas = sorted(geo.get("areas", []), key=lambda a: a.get("balance", 0),
                       reverse=True)
        # concentration tiles
        total = geo.get("total") or sum(a.get("balance", 0) for a in areas)
        from .metric_resolver import compact_currency, format_percent
        top = areas[0] if areas else {}
        top5 = sum(a.get("balance", 0) for a in areas[:5])
        tiles = [
            {"label": "Total funded exposure", "value": compact_currency(total),
             "hint": f"{geo.get('areaCount', len(areas))} ITL3 areas"},
            {"label": "Top area", "value": top.get("itl3_name", "—"),
             "hint": compact_currency(top.get("balance", 0)) if top else ""},
            {"label": "Top-5 concentration",
             "value": format_percent(top5 / total) if total else "—"},
            {"label": "Postcode coverage",
             "value": (f"{geo.get('coveragePct')}%" if geo.get("coveragePct") is not None else "—")},
        ]
        # 4 tiles across the top
        tw = Inches(2.98)
        for i, tile in enumerate(tiles):
            l = Emu(int(Inches(0.55)) + i * int(Inches(3.08)))
            self._tile(s, l, Inches(1.6), tw, Inches(1.5), tile)
        # region BarList below
        box = (Inches(0.55), Inches(3.35), Inches(12.25), Inches(3.25))
        rows = [{"label": a.get("itl3_name", a.get("itl3_code", "")),
                 "balance": a.get("balance", 0)} for a in areas[:12]]
        self._barlist_card(s, box, "Top areas by funded exposure", rows, "balance",
                           cid="geo_bars")
        self._footer(s)
        self._record("geography", spec.get("title"), "", placeholder=not areas)

    def _evolution_lines(self, s, spec, evo, chart_specs, accent=None):
        """Render N line-chart cards from an evolution payload's periods[].

        A time series needs ≥2 reporting periods; the dashboard flags a single cut
        with ``singlePeriod`` and shows an insufficient-history state rather than a
        lone point — so do the same (a one-dot 'trend' reads as broken)."""
        periods = evo.get("periods", [])
        single = bool(evo.get("singlePeriod")) or len(periods) < 2
        x = [str(p.get("period") or p.get("reporting_date") or p.get("run_id"))
             for p in periods]
        boxes = self._chart_boxes(len(chart_specs))
        for cs, box in zip(chart_specs, boxes):
            il, it, iw, ih = self._card(s, *box, cs["title"])
            series = [{"name": ser.get("name", ""),
                       "values": [(p.get("metrics") or {}).get(ser["key"]) for p in periods],
                       "color": ser.get("color")}
                      for ser in cs["series"]]
            path = self.work / f"{cs['id']}.png"
            if not single:
                R.draw_lines(path, x, series, iw, ih, theme=self.theme,
                             currency=cs.get("currency", True),
                             percent=cs.get("percent", False),
                             area=cs.get("area", False))
            else:
                render_placeholder_png(path, "", "Insufficient reporting history "
                                       "(needs ≥2 periods)", theme=self.theme,
                                       width_in=iw, height_in=ih)
            self._place(s, path, il, it, iw, ih)
        return single

    def slide_funded_evolution(self, spec):
        s = self._slide()
        self._header(s, spec.get("title", "Funded Evolution"), "Funded book over time")
        ph = self._evolution_lines(s, spec, self.d.funded_evolution, [
            {"id": "evo_bal", "title": "Funded balance by month",
             "series": [{"name": "Funded balance", "key": "funded_balance"}], "currency": True},
            {"id": "evo_ltv", "title": "WA current LTV by month",
             "series": [{"name": "WA LTV", "key": "wa_ltv"}], "currency": False, "percent": True},
        ])
        self._footer(s)
        self._record("funded_evolution", spec.get("title"), "", placeholder=ph)

    def slide_cohorts(self, spec):
        s = self._slide()
        self._header(s, spec.get("title", "Vintage Cohorts"),
                     "Static-pool profile compared across origination vintages")
        rows = sorted(self.d.cohorts.get("cohorts", []), key=lambda r: str(r.get("cohort")))
        x = [str(r.get("cohort")) for r in rows]
        boxes = self._chart_boxes(2)
        # Left: funded balance & book share by vintage. Right: WA LTV & WA rate.
        def _rate(v):
            if v is None:
                return None
            return v * 100 if v <= 1.5 else v
        il, it, iw, ih = self._card(s, *boxes[0], "Funded balance by vintage")
        p1 = self.work / "cohort_balance.png"
        if rows:
            R.draw_lines(p1, x, [{"name": "Funded balance",
                                  "values": [r.get("balance") for r in rows]}],
                         iw, ih, theme=self.theme, currency=True, area=True)
        else:
            render_placeholder_png(p1, "", "No cohort composition for this run",
                                   theme=self.theme, width_in=iw, height_in=ih)
        self._place(s, p1, il, it, iw, ih)

        il, it, iw, ih = self._card(s, *boxes[1], "WA current LTV & WA rate by vintage")
        p2 = self.work / "cohort_metrics.png"
        if rows:
            R.draw_lines(p2, x, [
                {"name": "WA current LTV", "values": [_rate(r.get("waLtv")) for r in rows],
                 "color": "#7c9cf0"},
                {"name": "WA rate", "values": [_rate(r.get("waRate")) for r in rows],
                 "color": "#5ec6b8"}],
                iw, ih, theme=self.theme, currency=False, percent=True)
        else:
            render_placeholder_png(p2, "", "No cohort composition for this run",
                                   theme=self.theme, width_in=iw, height_in=ih)
        self._place(s, p2, il, it, iw, ih)
        self._footer(s)
        self._record("cohorts", spec.get("title"), "", placeholder=not rows)

    def slide_pipeline(self, spec):
        s = self._slide()
        p = self.d.pipeline
        self._header(s, spec.get("title", "Pipeline"),
                     "Origination pipeline (pre-funded)", accent=self.theme.peri)
        if not p:
            self._placeholder_body(s, "No pipeline source resolved for this run.")
            self._footer(s)
            return self._record("pipeline", spec.get("title"), "", placeholder=True)
        from .metric_resolver import compact_currency, compact_number
        pw = p.get("priorWeek") or {}
        def delta(cur, prv, cur_key):
            if not pw or prv is None:
                return None, None
            diff = (cur or 0) - (prv or 0)
            intent = "positive" if diff > 0 else ("negative" if diff < 0 else "neutral")
            return (compact_currency(diff) if cur_key == "amt" else
                    ("+" if diff >= 0 else "−") + compact_number(abs(diff))) + " vs prior wk", intent
        d1, i1 = delta(p.get("pipelineAmount"), pw.get("pipelineAmount"), "amt")
        d2, i2 = delta(p.get("pipelineRowCount"), pw.get("pipelineRowCount"), "cnt")
        cases = p.get("pipelineRowCount") or 0
        amount = p.get("pipelineAmount") or 0
        avg_case = (amount / cases) if cases else 0
        tiles = [
            {"label": "Pipeline cases", "value": compact_number(cases),
             "delta": d2, "deltaIntent": i2},
            {"label": "Total pipeline amount", "value": compact_currency(amount),
             "delta": d1, "deltaIntent": i1},
            {"label": "Average case amount", "value": compact_currency(avg_case),
             "hint": "total ÷ cases"},
            {"label": "Weighted expected funded",
             "value": compact_currency(p.get("weightedExpectedFundedAmount")),
             "hint": "probability-weighted"},
        ]
        tw = Inches(2.92)
        for i, tile in enumerate(tiles):
            l = Emu(int(Inches(0.55)) + i * int(Inches(3.0)))
            self._tile(s, l, Inches(1.6), tw, Inches(1.45), tile)
        # two BarLists: stage + broker
        box1 = (Inches(0.55), Inches(3.28), Inches(6.0), Inches(3.35))
        box2 = (Inches(6.78), Inches(3.28), Inches(6.0), Inches(3.35))
        self._barlist_card(s, box1, "Pipeline amount by stage",
                           self._stage_rows(p.get("stageBreakdown", [])), "pipelineAmount",
                           cid="pipe_stage")
        # broker/region breakdown rows are keyed `key` (not `label`) and cap_breakdown
        # appends an aggregated "Other" row last — sort by amount so the BarList reads
        # largest-first, and bind the label to `key`.
        broker = list(p.get("brokerBreakdown", []) or p.get("regionBreakdown", []))
        broker.sort(key=lambda r: r.get("pipelineAmount", 0), reverse=True)
        broker_title = ("Pipeline amount by broker / channel"
                        if p.get("brokerBreakdown") else "Pipeline amount by region")
        self._barlist_card(s, box2, broker_title, broker, "pipelineAmount",
                           cid="pipe_broker", label_key="key")
        self._footer(s)
        self._record("pipeline", spec.get("title"), "", placeholder=False)

    def _stage_rows(self, rows, value_key="pipelineAmount"):
        order = {"KFI": 0, "APPLICATION": 1, "OFFER": 2, "COMPLETED": 3, "WITHDRAWN": 4}
        pretty = {"KFI": "KFI", "APPLICATION": "Application", "OFFER": "Offer",
                  "COMPLETED": "Completed", "WITHDRAWN": "Withdrawn", "UNKNOWN": "Other"}
        rows = sorted(rows, key=lambda r: order.get(str(r.get("stage", "")).upper(), 9))
        return [{"label": pretty.get(str(r.get("stage", "")).upper(),
                                     str(r.get("stage", "")).title()),
                 "pipelineAmount": r.get("pipelineAmount", 0),
                 "caseCount": r.get("caseCount", 0)} for r in rows]

    _STAGE_PRETTY = {"KFI": "KFI", "APPLICATION": "Application", "OFFER": "Offer",
                     "COMPLETED": "Completed", "WITHDRAWN": "Withdrawn"}
    _STAGE_COLOR = {"APPLICATION": "#7c9cf0", "OFFER": "#5ec6b8",
                    "COMPLETED": "#e0a458", "WITHDRAWN": "#eb6f6f"}

    def slide_pipeline_evolution(self, spec):
        s = self._slide()
        self._header(s, spec.get("title", "Pipeline Evolution"),
                     "Pipeline stock over time", accent=self.theme.peri)
        evo = self.d.pipeline_evolution or {}
        periods = evo.get("periods", [])
        single = bool(evo.get("singlePeriod")) or len(periods) < 2
        boxes = self._chart_boxes(2)
        x = [str(p.get("week") or p.get("period")) for p in periods]

        il, it, iw, ih = self._card(s, *boxes[0], "Pipeline amount by week")
        p1 = self.work / "pevo_amt.png"
        if not single:
            R.draw_lines(p1, x, [{"name": "Pipeline amount",
                                  "values": [(p.get("metrics") or {}).get("pipeline_amount")
                                             for p in periods]}],
                         iw, ih, theme=self.theme, currency=True, area=True)
        else:
            render_placeholder_png(p1, "", "Insufficient reporting history (needs ≥2 weeks)",
                                   theme=self.theme, width_in=iw, height_in=ih)
        self._place(s, p1, il, it, iw, ih)

        # Pipeline by stage over time — EXCLUDING the KFI line (dashboard view).
        il, it, iw, ih = self._card(s, *boxes[1], "Pipeline by stage over time")
        p2 = self.work / "pevo_stage.png"
        by_stage = evo.get("byStage", [])
        if not single and by_stage:
            weeks = sorted({str(r.get("week") or r.get("period")) for r in by_stage})
            lut = {(str(r.get("stage", "")).upper(), str(r.get("week") or r.get("period"))):
                   r.get("value") for r in by_stage}
            series = []
            for st in ("APPLICATION", "OFFER", "COMPLETED", "WITHDRAWN"):
                vals = [lut.get((st, w)) for w in weeks]
                if any(v for v in vals):
                    series.append({"name": self._STAGE_PRETTY[st], "values": vals,
                                   "color": self._STAGE_COLOR[st]})
            R.draw_lines(p2, weeks, series, iw, ih, theme=self.theme, currency=True)
        else:
            render_placeholder_png(p2, "", "Insufficient reporting history (needs ≥2 weeks)",
                                   theme=self.theme, width_in=iw, height_in=ih)
        self._place(s, p2, il, it, iw, ih)
        self._footer(s)
        self._record("pipeline_evolution", spec.get("title"), "", placeholder=single)

    def slide_origination_flow(self, spec):
        """KFI and Completion weekly-flow panels (bars) with a cumulative line —
        the dashboard's pipeline→origination flow view."""
        s = self._slide()
        self._header(s, spec.get("title", "Origination Flow — KFIs & Completions"),
                     "Weekly flow with cumulative build", accent=self.theme.peri)
        f = self.d.funnel or {}
        series = f.get("series", {}) or {}
        summary = f.get("summary", {}) or {}
        single = bool(f.get("singlePeriod")) or not series
        boxes = self._chart_boxes(2)
        for box, stage in zip(boxes, ("KFI", "COMPLETED")):
            label = self._STAGE_PRETTY.get(stage, stage)
            il, it, iw, ih = self._card(s, *box, f"{label}s · weekly flow")
            path = self.work / f"flow_{stage.lower()}.png"
            pts = series.get(stage) or []
            if not single and pts:
                weeks = [str(pt.get("week")) for pt in pts]
                vals = [pt.get("value") for pt in pts]
                cum, run = [], 0.0
                for v in vals:
                    run += float(v or 0)
                    cum.append(run)
                avg = (summary.get(stage) or {}).get("fiveWeekAvgFlowValue")
                R.draw_bars_with_line(path, weeks, vals, cum, iw, ih, theme=self.theme,
                                      avg=avg, line_label="Cumulative")
            else:
                render_placeholder_png(path, "", "Insufficient reporting history "
                                       "(needs ≥2 weeks)", theme=self.theme,
                                       width_in=iw, height_in=ih)
            self._place(s, path, il, it, iw, ih)
        self._footer(s)
        self._record("origination_flow", spec.get("title"), "", placeholder=single)

    def slide_multidim(self, spec):
        """Three multi-dimension funded views: LTV×Age bubble (left half) + two
        LTV heatmaps (borrower type, region) stacked on the right half."""
        s = self._slide()
        self._header(s, spec.get("title", "Multi-Dimensional Risk Analytics"),
                     "Funded balance across paired dimensions", accent=self.theme.peri)
        md = self.d.multidim or {}
        # Left half — bubble.
        il, it, iw, ih = self._card(s, Inches(0.55), Inches(1.62), Inches(6.4),
                                    Inches(4.95), "Balance by LTV × Borrower Age")
        bub = md.get("ltv_age")
        p0 = self.work / "md_bubble.png"
        if bub and bub.get("points"):
            R.draw_bubble(p0, bub["points"], bub["xLabels"], bub["yLabels"], iw, ih,
                          theme=self.theme)
        else:
            render_placeholder_png(p0, "", "LTV × age not available on this tape",
                                   theme=self.theme, width_in=iw, height_in=ih)
        self._place(s, p0, il, it, iw, ih)
        # Right half — two stacked heatmaps.
        rboxes = [(Inches(7.15), Inches(1.62), Inches(5.65), Inches(2.42)),
                  (Inches(7.15), Inches(4.15), Inches(5.65), Inches(2.42))]
        specs = [("ltv_borrower_type", "Balance by LTV × Borrower Type"),
                 ("ltv_region", "Balance by LTV × Region")]
        for box, (key, title) in zip(rboxes, specs):
            il, it, iw, ih = self._card(s, *box, title)
            hm = md.get(key)
            path = self.work / f"md_{key}.png"
            if hm and hm.get("matrix"):
                R.draw_heatmap(path, hm["xLabels"], hm["yLabels"], hm["matrix"], iw, ih,
                               theme=self.theme)
            else:
                render_placeholder_png(path, "", "Not available on this tape",
                                       theme=self.theme, width_in=iw, height_in=ih)
            self._place(s, path, il, it, iw, ih)
        self._footer(s)
        self._record("multidim", spec.get("title"), "", placeholder=not md)

    def slide_funnel(self, spec):
        s = self._slide()
        self._header(s, spec.get("title", "Origination Funnel"),
                     "Weekly origination funnel by stage", accent=self.theme.peri)
        summary = self.d.funnel.get("summary", {}) or {}
        stages = self.d.funnel.get("stages", []) or ["KFI", "APPLICATION", "OFFER", "COMPLETED"]
        pretty = {"KFI": "KFI", "APPLICATION": "Application", "OFFER": "Offer",
                  "COMPLETED": "Completed", "WITHDRAWN": "Withdrawn"}
        rows = [{"label": pretty.get(st, st),
                 "v": (summary.get(st) or {}).get("latestFlowValue", 0)}
                for st in stages]
        box = (Inches(0.55), Inches(1.62), Inches(12.25), Inches(4.95))
        title = "Latest weekly origination flow by stage"
        # Weekly flow needs ≥2 pipeline extracts. With a single extract, fall back to
        # the CURRENT pipeline funnel — case counts by stage — so the slide still
        # carries real data (matching the dashboard's single-period funnel).
        if not any(r["v"] for r in rows):
            stage_rows = self._stage_rows(self.d.pipeline.get("stageBreakdown", []),
                                          value_key="caseCount")
            rows = [{"label": r["label"], "v": r.get("caseCount", 0)} for r in stage_rows]
            title = "Current pipeline cases by stage"
        ok = self._barlist_card(s, box, title, [r for r in rows if r.get("v")], "v",
                                currency=False, cid="funnel")
        self._footer(s)
        self._record("funnel", spec.get("title"), "", placeholder=not ok)

    def slide_forecast_bridge(self, spec):
        s = self._slide()
        self._header(s, spec.get("title", "Forecast Bridge"),
                     "Funded + weighted pipeline → forecast funded",
                     accent=self.theme.mint)
        fb = self.d.forecast.get("forecastBridge") or {}
        if not fb or fb.get("forecastFundedBalance") is None:
            self._placeholder_body(s, "Forecast bridge requires funded + pipeline data.")
            self._footer(s)
            return self._record("forecast_bridge", spec.get("title"), "", placeholder=True)
        # Clarify the forecast is the CURRENT book's expected completions only.
        rd = self.d.reporting_date or "the reporting date"
        self._text(s, Inches(0.57), Inches(1.16), Inches(12.4), Inches(0.4),
                   f"Expected completions from the current book only (pipeline as of "
                   f"{rd}), weighted by historical stage conversion — not future new business.",
                   size=11, color=self.theme.ink_400, italic=True)
        # Full-width waterfall: split the weighted-pipeline block across expected
        # completion months (byCompletionMonth), Funded → +months → Forecast.
        funded = float(fb.get("fundedBalance") or 0)
        brk = (self.d.forecast.get("forecastBreakdowns") or {})
        months = [(str(m.get("month")), float(m.get("weightedExpectedFundedAmount") or 0))
                  for m in (brk.get("byCompletionMonth") or [])
                  if m.get("weightedExpectedFundedAmount")]
        months.sort()
        steps = [("Funded", funded, "base")]
        head, tail = months[:8], months[8:]
        for mth, val in head:
            steps.append((f"+ {mth}", val, "add"))
        if tail:
            steps.append(("+ Later", sum(v for _, v in tail), "add"))
        if len(steps) == 1:  # no monthly breakdown — single weighted block
            steps.append(("+ Weighted Pipeline",
                          float(fb.get("weightedExpectedFundedAmount") or 0), "add"))
        steps.append(("Forecast Funded", float(fb.get("forecastFundedBalance") or 0), "total"))
        box = (Inches(0.55), Inches(1.95), Inches(12.25), Inches(4.6))
        il, it, iw, ih = self._card(s, *box,
                                    "Funded + weighted pipeline (by expected completion month) → Forecast")
        path = self.work / "bridge.png"
        render_bridge_waterfall(path, steps, iw, ih, theme=self.theme)
        self._place(s, path, il, it, iw, ih)
        self._footer(s)
        self._record("forecast_bridge", spec.get("title"), "", placeholder=False)

    def slide_forecast_projection(self, spec):
        s = self._slide()
        self._header(s, spec.get("title", "Forecast Projection"),
                     "Run-rate scale-up (downside / base / upside)",
                     accent=self.theme.mint)
        ex = self.d.extrapolation or {}
        model = {}
        for key in ("completionRunRateForecast", "kfiConversionForecast"):
            cand = ex.get(key) or {}
            if cand.get("available") and cand.get("projectedBalances"):
                model = cand
                break
        proj = model.get("projectedBalances", [])
        # Projection chart (top ~62%) + milestone table (bottom) when available.
        chart_box = ((Inches(0.55), Inches(1.62), Inches(12.25), Inches(3.05)) if proj
                     else (Inches(0.55), Inches(1.62), Inches(12.25), Inches(4.95)))
        il, it, iw, ih = self._card(s, *chart_box, "Projected funded balance")
        path = self.work / "projection.png"
        if proj:
            x = [str(p.get("month")) for p in proj]
            series = [
                {"name": "Downside", "values": [p.get("downside") for p in proj], "color": "#eb6f6f"},
                {"name": "Base", "values": [p.get("base") for p in proj], "color": "#7c9cf0"},
                {"name": "Upside", "values": [p.get("upside") for p in proj], "color": "#5ec6b8"},
            ]
            R.draw_lines(path, x, series, iw, ih, theme=self.theme, currency=True)
        else:
            render_placeholder_png(path, "", "Insufficient run-rate history for a "
                                   "scale-up projection", theme=self.theme,
                                   width_in=iw, height_in=ih)
        self._place(s, path, il, it, iw, ih)
        # Milestone dates to funding thresholds.
        milestones = model.get("milestones", [])
        if proj and milestones:
            box2 = (Inches(0.55), Inches(4.85), Inches(12.25), Inches(1.7))
            il, it, iw, ih = self._card(s, *box2, "Milestone dates to funding thresholds")
            cols = ["Threshold", "Downside", "Base", "Upside"]
            def _d(m, k):
                v = m.get(f"{k}Date")
                return "reached" if v == "reached" else (str(v) if v else "—")
            trows = [[m.get("thresholdLabel", ""), _d(m, "downside"), _d(m, "base"),
                      _d(m, "upside")] for m in milestones[:6]]
            tpath = self.work / "milestones.png"
            R.draw_table(tpath, cols, trows, iw, ih, theme=self.theme)
            self._place(s, tpath, il, it, iw, ih)
        self._footer(s)
        self._record("forecast_projection", spec.get("title"), "", placeholder=not proj)

    def slide_forecast_evolution(self, spec):
        s = self._slide()
        self._header(s, spec.get("title", "Forecast Evolution"),
                     "Forecast funded balance across reporting runs", accent=self.theme.mint)
        ph = self._evolution_lines(s, spec, self.d.forecast_evolution, [
            {"id": "fevo", "title": "Forecast funded balance by run",
             "series": [
                 {"name": "Funded actual", "key": "funded_balance", "color": "#7c9cf0"},
                 {"name": "Weighted pipeline", "key": "weighted_expected_pipeline", "color": "#5ec6b8"},
                 {"name": "Forecast", "key": "forecast_funded_balance", "color": "#e0a458"}],
             "currency": True}])
        self._footer(s)
        self._record("forecast_evolution", spec.get("title"), "", placeholder=ph)

    def slide_risk(self, spec):
        s = self._slide()
        self._header(s, spec.get("title", "Risk Limits"),
                     "Concentration versus limits", accent=self.theme.peri)
        risk = self.d.risk or {}
        summary = risk.get("summary") or {}
        if not risk.get("available", True) and not risk.get("tests"):
            self._placeholder_body(s, risk.get("limitsReason")
                                   or "Risk-limit artifacts not present for this run.")
            self._footer(s)
            self.appendix.append("Risk monitor: no risk-limit artifact for this run.")
            return self._record("risk", spec.get("title"), "", placeholder=True)
        tiles = [
            {"label": "Tests passed", "value": summary.get("testsPassed", "—")},
            {"label": "Warnings", "value": summary.get("warnings", "—")},
            {"label": "Breaches", "value": summary.get("breaches", "—")},
            {"label": "Needs review", "value": summary.get("needsReview", "—")},
        ]
        tw = Inches(2.98)
        colors = [self.theme.rag["green"], self.theme.rag["amber"], self.theme.rag["red"],
                  self.theme.ink_400]
        for i, (tile, col) in enumerate(zip(tiles, colors)):
            l = Emu(int(Inches(0.55)) + i * int(Inches(3.08)))
            self._panel(s, l, Inches(1.6), tw, Inches(1.4), fill=self.theme.bg_panel_alt,
                        line=col, lw=1.2)
            self._text(s, l + Inches(0.2), Inches(1.82), tw, Inches(0.7),
                       str(tile["value"]), size=26, bold=True, color=col)
            self._text(s, l + Inches(0.2), Inches(2.62), tw, Inches(0.3),
                       tile["label"].upper(), size=9.5, color=self.theme.ink_400, bold=True)
        # tests table
        tests = risk.get("tests", [])[:10]
        box = (Inches(0.55), Inches(3.2), Inches(12.25), Inches(3.4))
        il, it, iw, ih = self._card(s, *box, "Limit tests")
        if tests:
            cols = ["Limit", "Actual", "Limit", "Headroom", "Status"]
            trows = [[str(t.get("label", "")), str(t.get("actualValue", "")),
                      str(t.get("limitValue", "")), str(t.get("headroom", "")),
                      str(t.get("status", ""))] for t in tests]
            path = self.work / "risk.png"
            R.draw_table(path, cols, trows, iw, ih, theme=self.theme, status_col=4)
            self._place(s, path, il, it, iw, ih)
        self._footer(s)
        self._record("risk", spec.get("title"), "", placeholder=False)

    def slide_methodology(self, spec):
        s = self._slide()
        self._header(s, spec.get("title", "Methodology & Notes"), "")
        lines = [
            f"Client:  {self.ctx.client_name}",
            f"Reporting date:  {self.d.reporting_date or 'n/a'}",
            f"Run:  {self.d.client_id}/{self.d.run_id}",
            "Data source:  MI Agent API computations (identical to the React dashboard).",
            "Funded KPIs / stratifications:  /mi/snapshot (compute_funded_snapshot).",
            "Pipeline & forecast:  /mi/forecast/snapshot (pipeline snapshot + forecast bridge).",
            "Evolution / cohorts / geography / risk:  /mi/evolution/*, /mi/cohorts, /mi/geo, /mi/risk-limits.",
            "Forecast method:  funded + Σ(weighted pipeline); scale-up via run-rate extrapolation.",
        ]
        if self.d.source_files:
            lines.append("Pipeline source:  " + ", ".join(self.d.source_files[:4]))
        self._bullets(s, lines, size=12.5)
        self._footer(s)
        self._record("methodology", spec.get("title"), "")

    def slide_appendix(self, spec):
        s = self._slide()
        self._header(s, spec.get("title", "Appendix — Data Coverage"), "")
        d = self.d.diagnostics or {}
        ts = d.get("timeSeries", {})
        pretty = {"funded_evolution": "Funded evolution",
                  "pipeline_evolution": "Pipeline evolution",
                  "funnel": "Origination funnel",
                  "forecast_projection": "Forecast projection", "risk": "Risk limits"}

        def _fmt(v, dash="not resolved"):
            return str(v) if v not in (None, "") else dash

        lines = ["RESOLVED SOURCES",
                 f"   Funded current source:  {_fmt(d.get('fundedCurrentSource'))}",
                 f"   Pipeline current source:  {_fmt(d.get('pipelineCurrentSource'))}",
                 "HISTORICAL DISCOVERY",
                 f"   Funded history root:  {_fmt(d.get('fundedHistoryRoot'))}",
                 f"      dated funded cuts found:  {d.get('fundedCutsFound', 0)}",
                 f"   Pipeline history root:  {_fmt(d.get('pipelineHistoryRoot'))}",
                 f"      dated pipeline snapshots found:  {d.get('pipelineSnapshotsFound', 0)}",
                 "TIME-SERIES SLIDE COVERAGE"]
        for key, label in pretty.items():
            info = ts.get(key, {})
            if info.get("placeholder"):
                lines.append(f"   {label}:  placeholder — {info.get('reason') or 'insufficient history'}")
            else:
                extra = f" ({info['periods']} periods)" if info.get("periods") else ""
                lines.append(f"   {label}:  rendered{extra}")
        extra_notes = [n for n in self.appendix
                       if "placeholder" not in n.lower() and "render" not in n.lower()]
        if extra_notes:
            lines.append("NOTES")
            lines += [f"   •  {n}" for n in extra_notes[:6]]
        self._bullets(s, lines, size=10.5)
        self._footer(s)
        self._record("appendix", spec.get("title"), "")

    # ------------------------------------------------------------------ helpers
    def _placeholder_body(self, slide, msg):
        path = self.work / f"ph_{self._page}.png"
        render_placeholder_png(path, "", msg, theme=self.theme, width_in=12.2,
                               height_in=4.9)
        self._place(slide, path, Inches(0.55), Inches(1.62), 12.2, 4.9)

    def _bullets(self, slide, lines, *, size=12):
        box = slide.shapes.add_textbox(Inches(0.6), Inches(1.62), Inches(12.1), Inches(5.2))
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

    # ------------------------------------------------------------------- build
    _DISPATCH = {
        "cover": "slide_cover", "kpi_summary": "slide_kpi_summary",
        "strat_barlists": "slide_strat", "multidim": "slide_multidim", "geo": "slide_geo",
        "funded_evolution": "slide_funded_evolution", "cohorts": "slide_cohorts",
        "pipeline_summary": "slide_pipeline", "pipeline_evolution": "slide_pipeline_evolution",
        "funnel": "slide_funnel", "origination_flow": "slide_origination_flow",
        "forecast_bridge": "slide_forecast_bridge",
        "forecast_projection": "slide_forecast_projection",
        "forecast_evolution": "slide_forecast_evolution", "risk": "slide_risk",
        "methodology": "slide_methodology", "appendix": "slide_appendix",
    }

    def build(self, slides: List[Dict[str, Any]], output: str | Path) -> Dict[str, Any]:
        for spec in slides:
            handler = getattr(self, self._DISPATCH.get(spec.get("type"), ""), None)
            if handler is None:
                continue
            handler(spec)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(out))
        return {"output": str(out), "slides": self.records,
                "coverage_notes": self.appendix}
