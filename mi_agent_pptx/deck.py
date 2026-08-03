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

_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")


def _pretty_date(value: Any) -> str:
    """``2026-06-30`` -> ``30 June 2026``, for a client-facing page."""
    text = str(value or "").strip()
    parts = text.split("-")
    if len(parts) == 3 and len(parts[0]) == 4:
        try:
            return f"{int(parts[2])} {_MONTHS[int(parts[1]) - 1]} {parts[0]}"
        except (ValueError, IndexError):
            return text
    return text


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
        #: Slides this portfolio did not justify, with reasons (rendered in the
        #: appendix so an omission is never silent).
        self.omissions: List[Any] = []
        self.facts: Dict[str, Any] = {}

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

    def scope_footnote(self) -> str:
        """The one-line scope + date stamp every slide carries.

        A slide read on its own — screenshotted, pasted into a memo — must still
        say which book it describes and as at when.
        """
        p = self.d.portfolio
        date = self.ctx.as_of_date or self.d.reporting_date
        if p is None:
            return f"Funded as at {date}" if date else ""
        label = f"{p.scope_label} portfolio"
        if p.has_mixed_reporting_dates:
            return f"{label} · mixed reporting dates (see cover)"
        return f"{label} · funded as at {date}" if date else label

    def _footer(self, slide):
        self._page += 1
        self._text(slide, Inches(0.55), Inches(7.08), Inches(6.6), Inches(0.3),
                   self.ctx.footer, size=8, color=self.theme.ink_500)
        stamp = self.scope_footnote()
        if stamp:
            self._text(slide, Inches(7.25), Inches(7.08), Inches(4.95), Inches(0.3),
                       stamp, size=8, color=self.theme.ink_400,
                       align=PP_ALIGN.RIGHT)
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
        val = str(tile.get("value") if avail else "—")
        # A KPI value may be a long label (an area name), not just a number. Step
        # the size down so it fits the tile instead of being clipped — a value the
        # reader cannot see is worse than a smaller one.
        width_in = int(iw) / EMU_IN
        size = 20
        for limit, candidate in ((width_in * 2.6, 20), (width_in * 3.4, 15),
                                 (width_in * 4.6, 12)):
            if len(val) <= limit:
                size = candidate
                break
        else:
            size = 10
        self._text(slide, l + pad, t + Inches(0.44), iw, Inches(0.58), val,
                   size=size, bold=True,
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
        """Cover — states, unambiguously, what this report is a report ABOUT.

        Scope, constituent books and every reporting date are rendered from the
        governed portfolio context, never from an operator-typed name, so a deck
        covering one book can never be mistaken for a total-portfolio deck.
        """
        s = self._slide()
        # Full-height left accent rail (replaces the old overlapping corner panel).
        rail = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                                  Inches(0.16), SLIDE_H)
        rail.fill.solid()
        rail.fill.fore_color.rgb = self._rgb(self.theme.peri)
        rail.line.fill.background()
        rail.shadow.inherit = False
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(2.92),
                                 Inches(2.2), Inches(0.07))
        bar.fill.solid()
        bar.fill.fore_color.rgb = self._rgb(self.theme.peri)
        bar.line.fill.background()
        bar.shadow.inherit = False
        self._text(s, Inches(0.9), Inches(0.62), Inches(6), Inches(0.4),
                   "TRAKT · MI AGENT", size=12, color=self.theme.peri, bold=True)
        self._text(s, Inches(0.86), Inches(1.42), Inches(11.5), Inches(1.1),
                   self._entity_name(), size=38, bold=True)
        self._text(s, Inches(0.92), Inches(3.12), Inches(11), Inches(0.5),
                   self.ctx.deck_name, size=18, color=self.theme.peri)
        strap = self._cover_strapline()
        self._text(s, Inches(0.92), Inches(3.7), Inches(10.5), Inches(0.6),
                   strap, size=12.5, color=self.theme.ink_300, italic=True,
                   spacing=1.12)
        self._cover_scope_block(s)
        self._text(s, Inches(0.92), Inches(6.72), Inches(8), Inches(0.35),
                   f"Generated automatically by {self.ctx.generated_by}", size=10,
                   color=self.theme.ink_400)
        self._record("cover", self._entity_name(), strap)

    def _entity_name(self):
        """The reporting entity — the operator-supplied client name."""
        return self.ctx.client_name

    def _cover_scope_block(self, s):
        """Reporting scope + constituent books + every reporting date."""
        p = self.d.portfolio
        left, top = Inches(0.92), Inches(4.42)
        self._text(s, left, top, Inches(5.6), Inches(0.3), "REPORTING SCOPE",
                   size=9, color=self.theme.peri, bold=True)
        if p is None:
            self._text(s, left, top + Inches(0.3), Inches(6), Inches(0.4),
                       "Scope unavailable — no governed portfolio context resolved.",
                       size=11.5, color=self.theme.ink_300)
            return
        self._text(s, left, top + Inches(0.28), Inches(5.6), Inches(0.4),
                   f"{p.scope_label} portfolio", size=15, bold=True)
        # Constituent books, each with its type.
        y = top + Inches(0.68)
        for book in p.portfolios[:5]:
            self._text(s, left, y, Inches(5.6), Inches(0.3),
                       f"·  {book.label}  —  {book.type_label}", size=10.5,
                       color=self.theme.ink_300)
            y = Emu(int(y) + int(Inches(0.245)))
        if len(p.portfolios) > 5:
            self._text(s, left, y, Inches(5.6), Inches(0.3),
                       f"·  and {len(p.portfolios) - 5} further book(s)", size=10.5,
                       color=self.theme.ink_400)

        # Reporting dates — per type when they differ, else one line.
        rleft = Inches(7.1)
        self._text(s, rleft, top, Inches(5.3), Inches(0.3), "REPORTING DATES",
                   size=9, color=self.theme.peri, bold=True)
        ry = top + Inches(0.28)
        dates = p.type_reporting_dates
        if dates:
            for ptype, date in sorted(dates.items()):
                from .deck_context import type_label
                self._text(s, rleft, ry, Inches(5.3), Inches(0.32),
                           f"{type_label(ptype)}:  {date}", size=11.5,
                           color=self.theme.ink_100)
                ry = Emu(int(ry) + int(Inches(0.3)))
        else:
            self._text(s, rleft, ry, Inches(5.3), Inches(0.32),
                       f"Funded portfolio:  {self.ctx.as_of_date or 'n/a'}",
                       size=11.5)
            ry = Emu(int(ry) + int(Inches(0.3)))
        if p.has_mixed_reporting_dates:
            self._text(s, rleft, Emu(int(ry) + int(Inches(0.06))), Inches(5.3),
                       Inches(0.5),
                       "⚠  Constituent books are reported as at different dates; "
                       "the total combines them.",
                       size=9.5, color=self.theme.amber, italic=True)

    def _cover_strapline(self):
        p = self.d.portfolio
        kpis = {k.get("id"): k for k in self.d.funded.get("kpis", [])}
        bal = kpis.get("balance", {}).get("value")
        loans = kpis.get("loans", {}).get("value")
        ltv = kpis.get("wa_current_ltv", {}).get("value")
        if bal and p is not None and p.is_mixed:
            return (f"Funded book of {bal} across {loans} loans, reported as a total "
                    f"and split by portfolio type.")
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

    # ------------------------------------------------- investor narrative
    def slide_exec_insights(self, spec):
        """Executive Summary — *what changed, and why*.

        Deterministic observations from :mod:`mi_agent_pptx.insights`. Every
        sentence is a template over a governed figure; nothing is generated.
        """
        s = self._slide()
        brief = self.d.insights or {}
        items = brief.get("insights") or []
        self._header(s, spec.get("title", "Executive Summary"),
                     "Governed observations for the period")
        if not items:
            self._placeholder_body(s, "No governed observations for this run.")
            self._footer(s)
            return self._record(spec.get("id", "executive_summary"),
                                spec.get("title"), "", placeholder=True)

        # Severity accent per observation — a caveat must not read as an aside.
        accent_for = {"concern": self.theme.rose, "attention": self.theme.amber}
        items = items[:8]
        top = 1.58
        # The card grid must FIT. Previously the row height was fixed, so a
        # seventh observation was laid out below the bottom of the slide and the
        # reader simply never saw it. Height is now derived from the available
        # band and the number of rows, so the grid always fits by construction.
        two_col = len(items) > 4
        col_w = Inches(6.0) if two_col else Inches(12.25)
        per_col = -(-len(items) // 2) if two_col else len(items)
        gap = 0.12
        band = 6.92 - top                      # bottom of the content area
        row_h = max(0.62, min(1.62 if two_col else 0.92,
                              (band - gap * (per_col - 1)) / max(per_col, 1)))
        for i, ins in enumerate(items):
            col, row = (i // per_col, i % per_col) if two_col else (0, i)
            l = Inches(0.55) if col == 0 else Inches(6.78)
            t = Inches(top + row * (row_h + gap))
            h = Inches(row_h)
            accent = accent_for.get(getattr(ins, "severity", "info"), self.theme.peri)
            self._panel(s, l, t, col_w, h, fill=self.theme.bg_panel_alt,
                        line=self.theme.line_soft)
            chip = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, Inches(0.045), h)
            chip.fill.solid()
            chip.fill.fore_color.rgb = self._rgb(accent)
            chip.line.fill.background()
            chip.shadow.inherit = False
            self._text(s, l + Inches(0.24), t + Inches(0.14),
                       col_w - Inches(0.46), Inches(0.42),
                       str(getattr(ins, "headline", "")), size=12.5, bold=True)
            self._text(s, l + Inches(0.24), t + Inches(0.56),
                       col_w - Inches(0.46), h - Inches(0.68),
                       str(getattr(ins, "summary", "")), size=10,
                       color=self.theme.ink_300, spacing=1.1)
        self._footer(s)
        self._record(spec.get("id", "executive_summary"), spec.get("title"),
                     f"{len(items)} governed observations.")

    def slide_portfolio_composition(self, spec):
        """Portfolio Composition — *what do I own today?*

        The anchor slide: total, then each portfolio type side by side on the
        same governed measures, so a blended average can never hide a divergence.
        """
        s = self._slide()
        p = self.d.portfolio
        self._header(s, spec.get("title", "Portfolio Composition"),
                     "Funded book by portfolio type")
        if p is None:
            self._placeholder_body(s, "No governed portfolio context resolved.")
            self._footer(s)
            return self._record(spec.get("id", "portfolio_composition"),
                                spec.get("title"), "", placeholder=True)
        from .metric_resolver import compact_currency, compact_number

        total_bal = p.total_balance or 0.0
        # Headline band: the total.
        self._panel(s, Inches(0.55), Inches(1.58), Inches(12.25), Inches(1.16),
                    fill=self.theme.bg_panel_alt, line=self.theme.line_soft)
        self._text(s, Inches(0.8), Inches(1.74), Inches(4.6), Inches(0.3),
                   "TOTAL FUNDED PORTFOLIO", size=9, color=self.theme.ink_400,
                   bold=True)
        self._text(s, Inches(0.8), Inches(2.04), Inches(4.6), Inches(0.55),
                   compact_currency(total_bal), size=26, bold=True)
        self._text(s, Inches(5.6), Inches(1.74), Inches(3.2), Inches(0.3),
                   "LOANS", size=9, color=self.theme.ink_400, bold=True)
        self._text(s, Inches(5.6), Inches(2.04), Inches(3.2), Inches(0.5),
                   compact_number(p.loan_count or 0), size=22, bold=True)
        self._text(s, Inches(8.9), Inches(1.74), Inches(3.7), Inches(0.3),
                   "CONSTITUENT BOOKS", size=9, color=self.theme.ink_400, bold=True)
        self._text(s, Inches(8.9), Inches(2.04), Inches(3.7), Inches(0.5),
                   f"{p.portfolio_count} · {len(p.type_slices) or 1} type(s)",
                   size=22, bold=True)

        # Per-type columns on identical governed measures.
        slices = list(p.type_slices)
        if not slices:
            self._text(s, Inches(0.55), Inches(3.1), Inches(12.25), Inches(0.5),
                       "Single-portfolio book — no type split applies.", size=12,
                       color=self.theme.ink_400, italic=True)
            self._footer(s)
            return self._record(spec.get("id", "portfolio_composition"),
                                spec.get("title"), "Single portfolio.")

        rows = [
            ("Funded balance", lambda sl: compact_currency(sl.balance)),
            ("Share of book", lambda sl: (f"{(sl.balance or 0) / total_bal * 100:.1f}%"
                                          if total_bal else "—")),
            ("Loans", lambda sl: compact_number(sl.loan_count or 0)),
            ("Average balance", lambda sl: compact_currency(sl.avg_balance)),
            ("WA current LTV", lambda sl: self._pct_display(sl, "wa_current_ltv")),
            ("WA interest rate", lambda sl: self._pct_display(sl, "wa_rate")),
            ("Period movement", lambda sl: self._signed_currency(sl.balance_movement)),
            ("Reporting date", lambda sl: sl.reporting_date or "—"),
        ]
        n = len(slices)
        gap = Inches(0.2)
        label_w = Inches(2.9)
        avail = int(Inches(12.25)) - int(label_w) - (n - 1) * int(gap)
        col_w = Emu(int(avail / n))
        top0 = Inches(3.02)
        row_h = Inches(0.44)

        # Column headers.
        for j, sl in enumerate(slices):
            l = Emu(int(Inches(0.55)) + int(label_w) + j * (int(col_w) + int(gap)))
            self._panel(s, l, top0, col_w, Inches(0.5), fill=self.theme.bg_panel,
                        line=self.theme.line)
            self._text(s, l + Inches(0.16), top0 + Inches(0.12),
                       col_w - Inches(0.3), Inches(0.3), sl.label.upper(),
                       size=9.5, bold=True, color=self.theme.peri)
        for i, (label, fn) in enumerate(rows):
            t = Emu(int(top0) + int(Inches(0.58)) + i * int(row_h))
            self._text(s, Inches(0.6), t + Inches(0.08), label_w, Inches(0.3),
                       label, size=10.5, color=self.theme.ink_400)
            for j, sl in enumerate(slices):
                l = Emu(int(Inches(0.55)) + int(label_w) + j * (int(col_w) + int(gap)))
                try:
                    value = fn(sl)
                except Exception:  # noqa: BLE001 — one cell must not break the slide
                    value = "—"
                self._text(s, l + Inches(0.16), t + Inches(0.08),
                           col_w - Inches(0.3), Inches(0.3), str(value),
                           size=11, bold=True)
        self._footer(s)
        self._record(spec.get("id", "portfolio_composition"), spec.get("title"),
                     f"{n} portfolio type(s).")

    def _pct_display(self, sl, kpi_id):
        """A weighted-average percentage as the governed snapshot formatted it."""
        return sl.display(kpi_id) or "—"

    @staticmethod
    def _signed_currency(value):
        """A movement with an explicit sign — ``+£851K`` / ``-£296K``."""
        from .metric_resolver import compact_currency
        if value is None:
            return "—"
        text = compact_currency(value)
        return f"+{text}" if float(value) > 0 else text

    def slide_portfolio_comparison(self, spec):
        """Direct vs Acquired — *why did the total move?*

        Movement attribution plus the mix differences that a blended total hides.
        """
        s = self._slide()
        p = self.d.portfolio
        self._header(s, spec.get("title", "Direct vs Acquired"),
                     "Period movement attribution and portfolio differences")
        if p is None or len(p.type_slices) < 2:
            self._placeholder_body(s, "Only one portfolio type is in scope.")
            self._footer(s)
            return self._record(spec.get("id", "portfolio_comparison"),
                                spec.get("title"), "", placeholder=True)
        from .metric_resolver import compact_currency

        slices = list(p.type_slices)
        # Left: movement attribution waterfall (total = Σ type movements).
        il, it, iw, ih = self._card(s, Inches(0.55), Inches(1.58), Inches(6.0),
                                    Inches(4.98), "What moved the total")
        movers = [(sl, sl.balance_movement) for sl in slices
                  if sl.balance_movement is not None]
        path = self.work / "cmp_attrib.png"
        if movers:
            opening = (p.total_balance or 0) - sum(v for _s, v in movers)
            steps = [("Opening", float(opening), "base")]
            for sl, v in movers:
                steps.append((sl.label.replace(" portfolio", "").replace(
                    " originations", ""), float(v), "add"))
            steps.append(("Closing", float(p.total_balance or 0), "total"))
            render_bridge_waterfall(path, steps, iw, ih, theme=self.theme)
        else:
            render_placeholder_png(path, "", "No prior reporting period to "
                                   "attribute movement against", theme=self.theme,
                                   width_in=iw, height_in=ih)
        self._place(s, path, il, it, iw, ih)

        # Right: side-by-side differences on identical governed measures.
        il, it, iw, ih = self._card(s, Inches(6.78), Inches(1.58), Inches(6.02),
                                    Inches(4.98), "How the books differ")
        cols = ["Measure"] + [sl.label for sl in slices]
        measures = [
            ("Funded balance", lambda sl: compact_currency(sl.balance)),
            ("Loans", lambda sl: f"{int(sl.loan_count or 0):,}"),
            ("Average balance", lambda sl: compact_currency(sl.avg_balance)),
            ("WA current LTV", lambda sl: self._pct_display(sl, "wa_current_ltv")),
            ("WA interest rate", lambda sl: self._pct_display(sl, "wa_rate")),
            ("WA borrower age", lambda sl: self._pct_display(sl, "wa_age")),
            ("Period movement", lambda sl: self._signed_currency(sl.balance_movement)),
            ("Loan movement", lambda sl: (f"{int(sl.loan_movement):+,}"
                                          if sl.loan_movement is not None else "—")),
        ]
        trows = []
        for label, fn in measures:
            row = [label]
            for sl in slices:
                try:
                    row.append(str(fn(sl)))
                except Exception:  # noqa: BLE001
                    row.append("—")
            trows.append(row)
        tpath = self.work / "cmp_table.png"
        R.draw_table(tpath, cols, trows, iw, ih, theme=self.theme)
        self._place(s, tpath, il, it, iw, ih)
        self._footer(s)
        self._record(spec.get("id", "portfolio_comparison"), spec.get("title"),
                     "Movement attribution by portfolio type.")

    def slide_movement_drivers(self, spec):
        """Portfolio Movement and Drivers — *why did funded AuM change?*

        Two governed views side by side: the movement by portfolio type (who
        moved it) and the movement by a chosen dimension (where it moved). Both
        reconcile exactly to the headline, because both come from governed
        decompositions whose parts sum to the whole by construction.
        """
        from . import movement as MV
        from .metric_resolver import compact_currency

        s = self._slide()
        p = self.d.portfolio
        bridges = self.d.movement or {}
        primary = next((bridges[k] for k in ("region", "broker", "ltv", "ticket")
                        if k in bridges and bridges[k].available), None)
        self._header(s, spec.get("title", "Portfolio Movement and Drivers"),
                     self._movement_window(primary), accent=self.theme.peri)

        # Left: movement by portfolio type (the attribution that matters most).
        il, it, iw, ih = self._card(s, Inches(0.55), Inches(1.62), Inches(6.02),
                                    Inches(3.62), "Movement by portfolio type")
        movers = [(sl, sl.balance_movement) for sl in (p.type_slices if p else ())
                  if sl.balance_movement is not None]
        path = self.work / "mv_type.png"
        if movers:
            opening = (p.total_balance or 0) - sum(v for _s, v in movers)
            steps = [("Opening", float(opening), "base")]
            for sl, v in movers:
                steps.append((sl.label.replace(" portfolio", "")
                              .replace(" originations", ""), float(v), "add"))
            steps.append(("Closing", float(p.total_balance or 0), "total"))
            render_bridge_waterfall(path, steps, iw, ih, theme=self.theme)
        elif primary is not None:
            steps = [("Opening", float(primary.opening or 0), "base"),
                     ("Net change", float(primary.total_delta or 0), "add"),
                     ("Closing", float(primary.closing or 0), "total")]
            render_bridge_waterfall(path, steps, iw, ih, theme=self.theme)
        self._place(s, path, il, it, iw, ih)

        # Right: movement by the leading governed dimension.
        if primary is not None:
            il, it, iw, ih = self._card(s, Inches(6.78), Inches(1.62), Inches(6.02),
                                        Inches(3.62), f"Movement by {primary.label.lower()}")
            rows = [{"category": c.category, "delta": c.delta, "is_other": c.is_other}
                    for c in sorted(primary.contributors,
                                    key=lambda c: -abs(c.delta))[:7]]
            dpath = self.work / "mv_dim.png"
            R.draw_diverging(dpath, rows, iw, ih, theme=self.theme)
            self._place(s, dpath, il, it, iw, ih)

        # Deterministic takeaways across the governed dimensions.
        lines: List[str] = []
        if p is not None and len(p.type_slices) > 1 and movers:
            ups = [(sl, v) for sl, v in movers if v > 0]
            downs = [(sl, v) for sl, v in movers if v < 0]
            if ups and downs:
                lines.append(
                    f"{ups[0][0].label} added {compact_currency(abs(ups[0][1]))}, "
                    f"partly offset by a {compact_currency(abs(downs[0][1]))} "
                    f"reduction in the {downs[0][0].label.lower()}.")
        for key in ("region", "broker", "ticket", "ltv"):
            b = bridges.get(key)
            if b is not None and b.available:
                head = MV.headline(b)
                if head:
                    lines.append(head)
            if len(lines) >= 3:
                break
        self._takeaway_strip(s, lines[:3], top=5.42)
        self._footer(s)
        self._record(spec.get("id", "movement_drivers"), spec.get("title"),
                     "Governed movement attribution.")

    def _movement_window(self, bridge) -> str:
        """The period a movement was measured over — never left implicit."""
        if bridge is None or not getattr(bridge, "available", False):
            return "Period movement attribution"
        if bridge.start_date and bridge.end_date:
            return (f"Movement from {_pretty_date(bridge.start_date)} to "
                    f"{_pretty_date(bridge.end_date)}")
        if bridge.start_period and bridge.end_period:
            return f"Movement from {bridge.start_period} to {bridge.end_period}"
        return "Period movement attribution"

    def _takeaway_strip(self, slide, lines, *, top: float, width: float = 12.25):
        """The slide's conclusions — the sentence an investor actually keeps."""
        lines = [l for l in lines if l]
        if not lines:
            return
        self._panel(slide, Inches(0.55), Inches(top), Inches(width),
                    Inches(0.34 + 0.26 * len(lines)),
                    fill=self.theme.bg_panel_alt, line=self.theme.line_soft)
        for i, line in enumerate(lines):
            self._text(slide, Inches(0.78), Inches(top + 0.14 + i * 0.26),
                       Inches(width - 0.46), Inches(0.26), f"·  {line}",
                       size=10, color=self.theme.ink_300)

    def slide_strat(self, spec):
        """Stratifications — composition, PLUS what changed.

        The composition view is legitimate investor content and is kept. What it
        could not answer is whether the shape of the book is moving, so where a
        comparable prior period exists the slide pairs each dimension with its
        governed marginal change and states the movement in words.
        """
        from . import movement as MV

        s = self._slide()
        strats = self.d.funded.get("stratifications", [])
        keys = spec.get("keys")
        if keys:
            strats = [st for st in strats if st.get("key") in keys]
        strats = strats[:2]

        bridges = self.d.movement or {}
        moved = [bridges[st.get("key")] for st in strats
                 if bridges.get(st.get("key")) is not None
                 and bridges[st.get("key")].available]
        window = self._movement_window(moved[0]) if moved else "Balance by dimension"
        self._header(s, spec.get("title", "Funded Stratifications"),
                     ("Composition and period movement" if moved
                      else "Balance by dimension"), accent=self.theme.peri)

        has_takeaways = bool(moved)
        chart_h = 3.62 if has_takeaways else 4.95
        ph = True
        if len(strats) == 1:
            boxes = [(Inches(0.55), Inches(1.62), Inches(12.25), Inches(chart_h))]
        else:
            boxes = [(Inches(0.55), Inches(1.62), Inches(6.02), Inches(chart_h)),
                     (Inches(6.78), Inches(1.62), Inches(6.02), Inches(chart_h))]

        for st, box in zip(strats, boxes):
            key = st.get("key")
            rows = st.get("bars", [])
            ok = self._barlist_card(s, box, st.get("label", key or ""), rows,
                                    "balance", cid=f"strat_{key}")
            ph = ph and not ok

        # A single marginal-change panel beneath, for the dimension that moved
        # most — one clear change view rather than a grid of small ones.
        lines: List[str] = []
        if moved:
            for b in moved:
                lines.extend(MV.takeaways(b, limit=1))
            self._takeaway_strip(s, lines[:3], top=5.42)
        if not strats:
            self._placeholder_body(s, "No funded stratifications for this run.")
        self._footer(s)
        self._record(spec.get("id", "strat"), spec.get("title"),
                     window if moved else "", placeholder=ph)

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
                     # NOT static-pool: this is a point-in-time cross-section of
                     # the CURRENT book grouped by origination year. A static
                     # pool tracks a fixed cohort forward through time, which
                     # this release does not render — so it must not be claimed.
                     "Point-in-time composition of the current book by origination year")
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
        # Only panels that actually resolved are drawn, and the layout adapts to
        # how many there are. Rendering an empty card labelled "not available"
        # tells an investor nothing and makes the pack look unfinished; the
        # composition guard omits the slide entirely when none resolve.
        bubble = md.get("ltv_age") if (md.get("ltv_age") or {}).get("points") else None
        heatmaps = [(key, title) for key, title in
                    (("ltv_borrower_type", "Balance by LTV × Borrower Type"),
                     ("ltv_region", "Balance by LTV × Region"))
                    if (md.get(key) or {}).get("matrix")]

        if bubble and heatmaps:
            # Bubble on the left, heatmaps stacked on the right.
            il, it, iw, ih = self._card(s, Inches(0.55), Inches(1.62), Inches(6.02),
                                        Inches(4.95), "Balance by LTV × Borrower Age")
            p0 = self.work / "md_bubble.png"
            R.draw_bubble(p0, bubble["points"], bubble["xLabels"], bubble["yLabels"],
                          iw, ih, theme=self.theme)
            self._place(s, p0, il, it, iw, ih)
            if len(heatmaps) == 1:
                boxes = [(Inches(6.78), Inches(1.62), Inches(6.02), Inches(4.95))]
            else:
                boxes = [(Inches(6.78), Inches(1.62), Inches(6.02), Inches(2.42)),
                         (Inches(6.78), Inches(4.15), Inches(6.02), Inches(2.42))]
            for box, (key, title) in zip(boxes, heatmaps):
                il, it, iw, ih = self._card(s, *box, title)
                hm = md[key]
                path = self.work / f"md_{key}.png"
                R.draw_heatmap(path, hm["xLabels"], hm["yLabels"], hm["matrix"],
                               iw, ih, theme=self.theme)
                self._place(s, path, il, it, iw, ih)
        elif bubble:
            il, it, iw, ih = self._card(s, Inches(0.55), Inches(1.62), Inches(12.25),
                                        Inches(4.95), "Balance by LTV × Borrower Age")
            p0 = self.work / "md_bubble.png"
            R.draw_bubble(p0, bubble["points"], bubble["xLabels"], bubble["yLabels"],
                          iw, ih, theme=self.theme)
            self._place(s, p0, il, it, iw, ih)
        else:
            boxes = self._chart_boxes(len(heatmaps))
            for box, (key, title) in zip(boxes, heatmaps):
                il, it, iw, ih = self._card(s, *box, title)
                hm = md[key]
                path = self.work / f"md_{key}.png"
                R.draw_heatmap(path, hm["xLabels"], hm["yLabels"], hm["matrix"],
                               iw, ih, theme=self.theme)
                self._place(s, path, il, it, iw, ih)
        self._footer(s)
        self._record("multidim", spec.get("title"), "", placeholder=False)

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
        self._text(s, Inches(0.57), Inches(1.54), Inches(12.4), Inches(0.30),
                   f"Expected completions from the current book only (pipeline as of "
                   f"{rd}), weighted by historical stage conversion — not future new business.",
                   size=10, color=self.theme.ink_400, italic=True)
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
        box = (Inches(0.55), Inches(1.92), Inches(12.25), Inches(4.64))
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

    def slide_watchlist(self, spec):
        """Portfolio Health and Watch Items — *what needs attention next?*

        At most five watch items and three observations, ranked by governed
        severity. No management actions are proposed: recommending what to do
        about a breach is a decision, and nothing in the evidence authorises the
        deck to make it.
        """
        s = self._slide()
        wl = self.d.watchlist or {}
        watch = wl.get("watch") or []
        observations = wl.get("observations") or []
        self._header(s, spec.get("title", "Portfolio Health and Watch Items"),
                     "Governed items requiring attention before the next period",
                     accent=self.theme.amber if watch else self.theme.peri)

        colour = {"concern": self.theme.rag["red"],
                  "attention": self.theme.rag["amber"]}
        top = 1.62
        if watch:
            for i, item in enumerate(watch[:5]):
                t = Inches(top + i * 0.86)
                accent = colour.get(item.severity, self.theme.ink_400)
                self._panel(s, Inches(0.55), t, Inches(7.7), Inches(0.76),
                            fill=self.theme.bg_panel_alt, line=self.theme.line_soft)
                chip = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), t,
                                          Inches(0.05), Inches(0.76))
                chip.fill.solid()
                chip.fill.fore_color.rgb = self._rgb(accent)
                chip.line.fill.background()
                chip.shadow.inherit = False
                self._text(s, Inches(0.78), t + Inches(0.09), Inches(7.2),
                           Inches(0.28), item.headline, size=11, bold=True)
                self._text(s, Inches(0.78), t + Inches(0.38), Inches(7.2),
                           Inches(0.32), item.summary, size=8.5,
                           color=self.theme.ink_400)
        else:
            # An empty list is a finding, and must be stated rather than leaving
            # the reader to wonder whether the check ran at all.
            self._panel(s, Inches(0.55), Inches(1.62), Inches(7.7), Inches(1.1),
                        fill=self.theme.bg_panel_alt, line=self.theme.line_soft)
            self._text(s, Inches(0.85), Inches(1.86), Inches(7.1), Inches(0.34),
                       "No material watch items identified for this period.",
                       size=13, bold=True, color=self.theme.rag["green"])
            self._text(s, Inches(0.85), Inches(2.22), Inches(7.1), Inches(0.3),
                       "Every governed check ran; none cleared its materiality "
                       "threshold.", size=9.5, color=self.theme.ink_400)

        # Observations column.
        self._panel(s, Inches(8.5), Inches(1.62), Inches(4.3), Inches(4.9),
                    fill=self.theme.bg_panel, line=self.theme.line)
        self._text(s, Inches(8.72), Inches(1.78), Inches(3.9), Inches(0.3),
                   "OBSERVATIONS", size=9, bold=True, color=self.theme.peri)
        y = 2.14
        for item in observations[:3]:
            self._text(s, Inches(8.72), Inches(y), Inches(3.9), Inches(0.46),
                       item.headline, size=10, bold=True)
            self._text(s, Inches(8.72), Inches(y + 0.5), Inches(3.9), Inches(0.72),
                       item.summary[:150], size=8.5, color=self.theme.ink_400)
            y += 1.42
        if not observations:
            self._text(s, Inches(8.72), Inches(2.14), Inches(3.9), Inches(0.3),
                       "None recorded.", size=10, color=self.theme.ink_400)
        self._footer(s)
        self._record(spec.get("id", "watchlist"), spec.get("title"),
                     f"{len(watch)} watch item(s).")

    def slide_concentration(self, spec):
        """Concentration Tests and Headroom — *am I within my limits?*

        Three states, kept visually and verbally distinct: CURRENT funded (the
        only actual), EXPECTED forecast, and the ALL-PIPELINE-CONVERTS stress.
        Presenting the stress as an expectation would misstate the risk, so it
        is labelled a stress everywhere it appears.
        """
        from . import concentration as C

        s = self._slide()
        env = self.d.concentration or {}
        rows = C.adapt_tests(env)
        self._header(s, spec.get("title", "Concentration Tests and Headroom"),
                     "Utilisation of contractual limits — current, expected and stress",
                     accent=self.theme.peri)
        if not rows:
            self._placeholder_body(s, "No governed concentration tests configured.")
            self._footer(s)
            return self._record(spec.get("id", "concentration"), spec.get("title"),
                                "", placeholder=True)

        summary = C.summarise(env, rows)
        top = C.select_tests(rows)
        forward = C.forward_states_available(env)

        # -- summary strip --------------------------------------------------
        tiles = [
            ("TESTS", str(summary["tests"]), self.theme.ink_400),
            ("IN BREACH", str(summary["breaches"]),
             self.theme.rag["red"] if summary["breaches"] else self.theme.rag["green"]),
            ("WARNING", str(summary["warnings"]),
             self.theme.rag["amber"] if summary["warnings"] else self.theme.ink_400),
            ("FORECAST BREACH", str(summary["expected_breaches"]) if forward else "—",
             self.theme.rag["amber"] if summary["expected_breaches"] else self.theme.ink_400),
            ("STRESS BREACH", str(summary["stress_breaches"]) if forward else "—",
             self.theme.ink_400),
        ]
        tw = Inches(2.35)
        for i, (label, value, colour) in enumerate(tiles):
            l = Emu(int(Inches(0.55)) + i * int(Inches(2.45)))
            self._panel(s, l, Inches(1.56), tw, Inches(0.92),
                        fill=self.theme.bg_panel_alt, line=self.theme.line_soft)
            self._text(s, l + Inches(0.18), Inches(1.66), tw - Inches(0.3),
                       Inches(0.28), label, size=8, color=self.theme.ink_400,
                       bold=True)
            self._text(s, l + Inches(0.18), Inches(1.92), tw - Inches(0.3),
                       Inches(0.42), value, size=19, bold=True, color=colour)

        # -- utilisation bars ------------------------------------------------
        bars = [{"label": r["label"], "utilisation": r["utilisation"] or 0,
                 "status": r["status"],
                 "expectedUtilisation": r["expected_utilisation"] if forward else None,
                 "stressUtilisation": r["stress_utilisation"] if forward else None}
                for r in top]
        il, it, iw, ih = self._card(s, Inches(0.55), Inches(2.66), Inches(6.5),
                                    Inches(3.62), "Utilisation of limit")
        path = self.work / "conc_util.png"
        R.draw_utilisation_tests(path, bars, iw, ih, theme=self.theme)
        self._place(s, path, il, it, iw, ih)

        # -- the numbers behind the bars -------------------------------------
        # Rendered as NATIVE PowerPoint text, not a chart image: a covenant table
        # is the page an investor reads closely, copies figures out of and zooms
        # into, and an image is none of those things.
        self._panel(s, Inches(7.28), Inches(2.66), Inches(5.52), Inches(3.62),
                    fill=self.theme.bg_panel, line=self.theme.line)
        self._text(s, Inches(7.5), Inches(2.82), Inches(5.1), Inches(0.34),
                   "Current position against limit", size=12.5, bold=True)
        cols = [("Test", 0.0, PP_ALIGN.LEFT), ("Current", 2.55, PP_ALIGN.RIGHT),
                ("Limit", 3.55, PP_ALIGN.RIGHT), ("Headroom", 4.6, PP_ALIGN.RIGHT)]
        if forward:
            cols = [("Test", 0.0, PP_ALIGN.LEFT), ("Current", 2.15, PP_ALIGN.RIGHT),
                    ("Limit", 3.05, PP_ALIGN.RIGHT), ("Headroom", 3.95, PP_ALIGN.RIGHT),
                    ("Expected", 5.05, PP_ALIGN.RIGHT)]
        head_y = Inches(3.3)
        for label, dx, align in cols:
            self._text(s, Inches(7.5 + dx), head_y, Inches(1.0 if dx else 2.4),
                       Inches(0.26), label, size=8.5, color=self.theme.ink_400,
                       bold=True, align=align)
        row_h = 0.56
        for i, r in enumerate(top):
            y = Inches(3.62 + i * row_h)
            status_colour = self.theme.rag.get(
                {"breach": "red", "warning": "amber"}.get(r["status"], "green"),
                self.theme.ink_300)
            cells = [
                (r["label"][:30], 0.0, PP_ALIGN.LEFT, self.theme.ink_100, 2.4),
                (C.format_measure(r["value"], r["unit"]),
                 cols[1][1], PP_ALIGN.RIGHT, status_colour, 1.0),
                (C.format_measure(r["limit"], r["unit"]),
                 cols[2][1], PP_ALIGN.RIGHT, self.theme.ink_300, 1.0),
                (f"{r['headroom']:.1f}" if r["headroom"] is not None else "—",
                 cols[3][1], PP_ALIGN.RIGHT, self.theme.ink_300, 1.0),
            ]
            if forward:
                cells.append((C.format_measure(r["expected_value"], r["unit"])
                              if r["expected_value"] is not None else "—",
                              cols[4][1], PP_ALIGN.RIGHT, self.theme.peri, 1.0))
            for value, dx, align, colour, w in cells:
                self._text(s, Inches(7.5 + dx), y, Inches(w), Inches(0.3),
                           str(value), size=10, color=colour, align=align,
                           bold=(align == PP_ALIGN.RIGHT and colour is status_colour))
            self._text(s, Inches(7.5), Emu(int(y) + int(Inches(0.28))),
                       Inches(5.1), Inches(0.22),
                       f"{r['status'].upper()}"
                       + (f" · breaches {r['breach_horizon']}" if r.get("breach_horizon")
                          else ""),
                       size=7.5, color=status_colour)

        # -- deterministic takeaway + source disclosure -----------------------
        takeaway = self._concentration_takeaway(summary, top, forward)
        self._text(s, Inches(0.57), Inches(6.42), Inches(9.4), Inches(0.4),
                   takeaway, size=10, color=self.theme.ink_300, italic=True)
        disclosure = C.source_disclosure(env)
        if disclosure:
            self._text(s, Inches(0.57), Inches(6.72), Inches(9.4), Inches(0.3),
                       disclosure, size=8.5, color=self.theme.ink_500)
        self._footer(s)
        self._record(spec.get("id", "concentration"), spec.get("title"),
                     f"{summary['tests']} governed tests.")

    def _concentration_takeaway(self, summary, rows, forward) -> str:
        """One deterministic sentence over the governed evidence — no inference."""
        from . import concentration as C
        if summary["breaches"]:
            worst = next((r for r in rows if r["status"] == C.STATUS_BREACH), None)
            lead = (f"{summary['breaches']} limit"
                    f"{'s are' if summary['breaches'] != 1 else ' is'} in breach at the "
                    f"reporting date")
            if worst:
                lead += f", led by {worst['label']}"
            lead += "."
        elif summary["warnings"]:
            lead = (f"All limits are within contractual thresholds; "
                    f"{summary['warnings']} "
                    f"{'are' if summary['warnings'] != 1 else 'is'} in warning range.")
        else:
            lead = "All governed concentration limits are within their contractual thresholds."
        tail = ""
        if forward and summary["expected_breaches"]:
            horizon = next((r.get("breach_horizon") for r in rows
                            if r.get("expected_breach") and r.get("breach_horizon")), None)
            tail = (f" The expected forecast breaches {summary['expected_breaches']} "
                    f"limit{'s' if summary['expected_breaches'] != 1 else ''}"
                    + (f", crossing around {horizon}" if horizon else "") + ".")
        elif forward and summary["stress_breaches"]:
            tail = (f" The all-pipeline-converts stress would breach "
                    f"{summary['stress_breaches']} "
                    f"limit{'s' if summary['stress_breaches'] != 1 else ''}; this is a "
                    f"stress, not the expected outcome.")
        return lead + tail

    def slide_methodology(self, spec):
        """Kept as an alias of the investor-safe Data and Methodology page.

        The former version of this slide listed endpoint paths, internal compute
        function names and resolved source filenames. Those are implementation
        details, not methodology, and had no place in a client-facing pack — the
        single page below states the basis of preparation in business language.
        """
        return self.slide_appendix(spec)

    def slide_appendix(self, spec):
        """Data and Methodology — investor-safe.

        The previous version of this page printed the generator's own
        diagnostics: discovery roots (absolute filesystem paths), resolved
        source filenames and internal compute-function names. None of that
        belongs in a client-facing document. This page states the same facts in
        business language — what the report covers, as at when, what was
        excluded and why — and never where the bytes live.
        """
        s = self._slide()
        self._header(s, spec.get("title", "Data and Methodology"),
                     "Scope, source dates, coverage and basis of preparation")
        p = self.d.portfolio
        d = self.d.diagnostics or {}

        left = []
        left.append("REPORTING SCOPE")
        if p is not None:
            left.append(f"   {p.scope_label} portfolio")
            for book in p.portfolios[:4]:
                left.append(f"   ·  {book.label} — {book.type_label}")
            if len(p.portfolios) > 4:
                left.append(f"   ·  and {len(p.portfolios) - 4} further book(s)")
        else:
            left.append("   Scope unavailable for this run.")

        left.append("")
        left.append("SOURCE REPORTING DATES")
        if p is not None and p.type_reporting_dates:
            from .deck_context import type_label
            for ptype, date in sorted(p.type_reporting_dates.items()):
                left.append(f"   {type_label(ptype)} funded data as at {_pretty_date(date)}.")
        elif self.d.reporting_date:
            left.append(f"   Funded portfolio data as at {_pretty_date(self.d.reporting_date)}.")
        pipe_date = (self.d.pipeline or {}).get("pipelineAsOfDate")
        if pipe_date:
            left.append(f"   Pipeline data as at {_pretty_date(pipe_date)}.")
        if p is not None and p.has_mixed_reporting_dates:
            left.append("   Constituent books are reported as at different dates;")
            left.append("   the total combines them.")

        right = ["BASIS OF PREPARATION",
                 "   Figures are produced by the governed MI calculations and are",
                 "   identical to the management dashboard for the same portfolio",
                 "   and reporting date.",
                 "   Commentary is generated deterministically from those figures.",
                 "   No language model is used in its production."]
        conc = self.d.concentration or {}
        if conc.get("tests"):
            from . import concentration as C
            disclosure = C.source_disclosure(conc)
            if disclosure:
                right.append(f"   Concentration limits: {disclosure.lower()}.")
        right.append("")
        right.append("COVERAGE")
        cuts = d.get("fundedCutsFound") or 0
        if cuts:
            right.append(f"   {cuts} funded reporting period(s) available.")
        snaps = d.get("pipelineSnapshotsFound") or 0
        right.append(f"   {snaps} weekly pipeline extract(s) available."
                     if snaps else "   No weekly pipeline extracts available.")

        if self.omissions:
            right.append("")
            right.append("SECTIONS NOT INCLUDED")
            for o in self.omissions[:6]:
                right.append(f"   {o.title}: {o.reason}.")
            if len(self.omissions) > 6:
                right.append(f"   and {len(self.omissions) - 6} further section(s).")

        self._column_text(s, left, Inches(0.6), Inches(6.0))
        self._column_text(s, right, Inches(6.95), Inches(5.85))
        self._footer(s)
        self._record(spec.get("id", "appendix"), spec.get("title"), "")

    def _column_text(self, slide, lines, left, width):
        """A column of the methodology page; section headings pick up the accent."""
        box = slide.shapes.add_textbox(left, Inches(1.6), width, Inches(5.3))
        tf = box.text_frame
        tf.word_wrap = True
        for i, line in enumerate(lines):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            run = para.add_run()
            run.text = line
            heading = bool(line) and line == line.upper() and not line.startswith(" ")
            run.font.size = Pt(9 if heading else 10)
            run.font.bold = heading
            run.font.name = self.theme.font_sans
            run.font.color.rgb = self._rgb(
                self.theme.peri if heading else self.theme.ink_300)
            para.space_after = Pt(5 if heading else 2)

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
        "exec_insights": "slide_exec_insights",
        "portfolio_composition": "slide_portfolio_composition",
        "portfolio_comparison": "slide_portfolio_comparison",
        "movement_drivers": "slide_movement_drivers",
        "watchlist": "slide_watchlist",
        "strat_barlists": "slide_strat", "multidim": "slide_multidim", "geo": "slide_geo",
        "funded_evolution": "slide_funded_evolution", "cohorts": "slide_cohorts",
        "pipeline_summary": "slide_pipeline", "pipeline_evolution": "slide_pipeline_evolution",
        "funnel": "slide_funnel", "origination_flow": "slide_origination_flow",
        "forecast_bridge": "slide_forecast_bridge",
        "forecast_projection": "slide_forecast_projection",
        "forecast_evolution": "slide_forecast_evolution", "risk": "slide_risk",
        "concentration": "slide_concentration",
        "methodology": "slide_methodology", "appendix": "slide_appendix",
    }

    def build(self, slides: List[Dict[str, Any]], output: str | Path) -> Dict[str, Any]:
        """Render the slides this portfolio justifies, and record the rest.

        Composition runs BEFORE any rendering, so a slide that would have had
        nothing to show never reaches the deck — an investor pack contains no
        "no data available" pages. Everything dropped is carried into the
        appendix with its reason, because a silent omission is indistinguishable
        from a book that had nothing to report.
        """
        from .composition import build_facts, select_slides

        facts = build_facts(self.d)
        selected, omissions = select_slides(slides, self.d, facts)
        self.omissions = list(omissions)
        self.facts = dict(facts)

        for spec in selected:
            handler = getattr(self, self._DISPATCH.get(spec.get("type"), ""), None)
            if handler is None:
                continue
            handler(spec)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(out))
        return {"output": str(out), "slides": self.records,
                "coverage_notes": self.appendix,
                "omitted_slides": [o.to_dict() for o in self.omissions],
                "facts": self.facts,
                "portfolio_context": (self.d.portfolio.to_dict()
                                      if self.d.portfolio else None),
                "insights": self._insight_records()}

    def _insight_records(self) -> Dict[str, Any]:
        """The executive summary in a serialisable form, for the run manifest."""
        brief = self.d.insights or {}
        return {
            "insight_version": brief.get("insight_version"),
            "status": brief.get("status"),
            "count": brief.get("count", 0),
            "headlines": [str(getattr(i, "headline", ""))
                          for i in (brief.get("insights") or [])],
            "omitted": [o.to_dict() for o in (brief.get("omitted") or [])],
        }
