"""mi_agent_pptx.render — draw charts from explicit dashboard payload rows/series.

These are the low-level renderers the payload-driven deck uses. They take the
data *verbatim from the MI API payloads* (BarList rows, evolution series, bridge
steps, risk tables) — no aggregation — so the visual is a faithful export of the
dashboard's Recharts/BarList/stat-tile components. Each renders at the exact
width×height of its slide panel and onto the theme panel background.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402

from .metric_resolver import compact_currency, compact_number  # noqa: E402
from .pptx_theme import PptxTheme, THEME  # noqa: E402

_SANS = next((f for f in ("Inter", "Liberation Sans", "DejaVu Sans")
              if f in {ff.name for ff in fm.fontManager.ttflist}), "DejaVu Sans")
_MONO = next((f for f in ("Liberation Mono", "DejaVu Sans Mono")
              if f in {ff.name for ff in fm.fontManager.ttflist}), "DejaVu Sans Mono")
plt.rcParams.update({"font.family": _SANS, "font.size": 11,
                     "axes.unicode_minus": False})
_MONO_FP = fm.FontProperties(family=_MONO)

# Dashboard evolution palette (EvolutionPanel PALETTE).
EVO_PALETTE = ["#7c9cf0", "#5ec6b8", "#e0a458", "#c98bdb", "#6fcf97", "#eb6f6f"]


def _fig(w, h, theme, dpi=220):
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    fig.patch.set_facecolor(theme.bg_panel)
    return fig


def _save(fig, path, theme, dpi=220):
    fig.savefig(Path(path), facecolor=theme.bg_panel, dpi=dpi)
    plt.close(fig)
    return Path(path)


def _truncate(label: str, max_chars: int) -> str:
    return label if len(label) <= max_chars else label[:max_chars - 1].rstrip() + "…"


def draw_barlist(path, rows: Sequence[Dict[str, Any]], value_key: str, w: float,
                 h: float, *, theme: PptxTheme = THEME, currency: bool = True,
                 label_key: str = "label", count_key: Optional[str] = "count",
                 dpi: int = 220) -> Path:
    """Dashboard BarList: label left, periwinkle bar ∝ max, mono value right."""
    rows = [r for r in rows if r is not None]
    fig = _fig(w, h, theme, dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(theme.bg_panel)
    ax.set_xlim(0, 1)
    ax.axis("off")
    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                color=theme.ink_500, fontsize=12)
        return _save(fig, path, theme, dpi)

    fmt: Callable = compact_currency if currency else compact_number
    values = [float(r.get(value_key) or 0) for r in rows]
    labels = [str(r.get(label_key, "")) for r in rows]
    n = len(rows)
    vmax = max(max(values), 1.0)
    pad_top, pad_bot = 0.10, 0.05
    band = (1.0 - pad_top - pad_bot) / max(n, 1)
    bar_h = min(band * 0.62, 0.135)
    label_x, tx0, tx1 = 0.005, 0.335, 0.85
    tw = tx1 - tx0
    max_chars = max(10, int((tx0 - label_x) * w * 72 / (10.5 * 0.56)))
    for i, (lab, val) in enumerate(zip(labels, values)):
        yc = 1.0 - pad_top - (i + 0.5) * band
        y0 = yc - bar_h / 2
        ax.add_patch(mpatches.FancyBboxPatch(
            (tx0, y0), tw, bar_h, boxstyle="round,pad=0,rounding_size=0.012",
            linewidth=0, facecolor=theme.bg_panel_alt, alpha=0.7,
            mutation_aspect=h / w, zorder=1))
        frac = max(val / vmax, 0.012)
        ax.add_patch(mpatches.FancyBboxPatch(
            (tx0, y0), tw * frac, bar_h, boxstyle="round,pad=0,rounding_size=0.012",
            linewidth=0, facecolor=theme.peri, alpha=0.9,
            mutation_aspect=h / w, zorder=2))
        ax.text(label_x, yc, _truncate(lab, max_chars), va="center", ha="left",
                color=theme.ink_300, fontsize=10.5, zorder=3)
        ax.text(0.995, yc, fmt(val), va="center", ha="right", color=theme.ink_100,
                fontsize=10.5, fontproperties=_MONO_FP, zorder=3)
    return _save(fig, path, theme, dpi)


def draw_bars_with_line(path, x_labels: Sequence[str], bars: Sequence[Optional[float]],
                        line: Sequence[Optional[float]], w: float, h: float, *,
                        theme: PptxTheme = THEME, bar_currency: bool = True,
                        avg: Optional[float] = None, line_label: str = "Cumulative",
                        dpi: int = 220) -> Path:
    """Weekly-flow bars (periwinkle, left axis) + a cumulative line (mint, right
    axis), with an optional dashed 5-week-average marker — the dashboard's
    KFI/Completions weekly-flow panel."""
    fig = _fig(w, h, theme, dpi)
    ax = fig.add_axes([0.09, 0.16, 0.82, 0.78])
    ax.set_facecolor(theme.bg_panel)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(theme.line_soft)
    ax.tick_params(colors=theme.ink_500, labelsize=8.5, length=0)
    ax.grid(axis="y", color=theme.line_soft, linewidth=0.7, linestyle=(0, (3, 3)), alpha=0.9)
    ax.set_axisbelow(True)
    n = len(x_labels)
    x = list(range(n))
    if not n:
        ax.text(0.5, 0.5, "Insufficient history", ha="center", va="center",
                transform=ax.transAxes, color=theme.ink_500, fontsize=12)
        ax.axis("off")
        return _save(fig, path, theme, dpi)
    bvals = [0.0 if v is None else float(v) for v in bars]
    ax.bar(x, bvals, width=0.62, color=theme.peri, alpha=0.85, zorder=2)
    if avg:
        ax.axhline(float(avg), color=theme.rag.get("amber", "#e0a458"), linewidth=1.2,
                   linestyle=(0, (5, 4)), zorder=3)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, p: compact_currency(v) if bar_currency else compact_number(v)))
    ax2 = ax.twinx()
    for s in ("top", "left", "right"):
        ax2.spines[s].set_visible(False)
    lvals = [None if v is None else float(v) for v in line]
    ax2.plot(x, lvals, color=theme.mint, linewidth=2.4, marker="o", markersize=3,
             zorder=4, solid_capstyle="round")
    ax2.tick_params(colors=theme.ink_500, labelsize=8.5, length=0)
    ax2.yaxis.set_major_formatter(FuncFormatter(
        lambda v, p: compact_currency(v) if bar_currency else compact_number(v)))
    step = max(1, n // 7)
    idx = sorted(set(list(range(0, n, step)) + [n - 1]))
    ax.set_xticks([x[i] for i in idx])
    ax.set_xticklabels([str(x_labels[i]) for i in idx], fontsize=8, color=theme.ink_500)
    return _save(fig, path, theme, dpi)


def draw_bubble(path, points: Sequence[Dict[str, Any]], x_labels: Sequence[str],
                y_labels: Sequence[str], w: float, h: float, *, theme: PptxTheme = THEME,
                dpi: int = 220) -> Path:
    """Balance bubble grid: x/y are ordered band labels, bubble area ∝ balance.
    *points* = ``[{x, y, value}]`` where x/y are indices into the label lists."""
    fig = _fig(w, h, theme, dpi)
    ax = fig.add_axes([0.16, 0.14, 0.80, 0.80])
    ax.set_facecolor(theme.bg_panel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(theme.line_soft)
    ax.tick_params(colors=theme.ink_400, labelsize=9, length=0)
    ax.grid(True, color=theme.line_soft, linewidth=0.6, linestyle=(0, (2, 3)), alpha=0.7)
    ax.set_axisbelow(True)
    pts = [p for p in points if p.get("value")]
    if not pts:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes,
                color=theme.ink_500, fontsize=12)
        ax.axis("off")
        return _save(fig, path, theme, dpi)
    vmax = max(float(p["value"]) for p in pts) or 1.0
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    sizes = [80 + 2600 * (float(p["value"]) / vmax) for p in pts]
    ax.scatter(xs, ys, s=sizes, c=theme.peri, alpha=0.62, edgecolors=theme.mint,
               linewidths=0.8, zorder=3)
    for p in pts:
        ax.text(p["x"], p["y"], compact_currency(p["value"]), ha="center", va="center",
                color=theme.ink_100, fontsize=7.2, zorder=4)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(list(x_labels), fontsize=8.5, color=theme.ink_400, rotation=0)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(list(y_labels), fontsize=8.5, color=theme.ink_400)
    ax.set_xlim(-0.6, len(x_labels) - 0.4)
    ax.set_ylim(-0.6, len(y_labels) - 0.4)
    return _save(fig, path, theme, dpi)


def draw_heatmap(path, x_labels: Sequence[str], y_labels: Sequence[str],
                 matrix: Sequence[Sequence[float]], w: float, h: float, *,
                 theme: PptxTheme = THEME, dpi: int = 220) -> Path:
    """Balance heatmap: rows=y_labels, cols=x_labels, cell shade ∝ balance, with
    the £ value annotated. Uses the periwinkle→mint brand ramp."""
    from matplotlib.colors import LinearSegmentedColormap
    fig = _fig(w, h, theme, dpi)
    ax = fig.add_axes([0.20, 0.16, 0.78, 0.78])
    ax.set_facecolor(theme.bg_panel)
    mat = np.array([[float(c or 0) for c in row] for row in matrix], dtype=float) \
        if matrix else np.zeros((len(y_labels), len(x_labels)))
    if mat.size == 0 or mat.max() == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes,
                color=theme.ink_500, fontsize=12)
        ax.axis("off")
        return _save(fig, path, theme, dpi)
    cmap = LinearSegmentedColormap.from_list(
        "brand", [theme.bg_panel_alt, theme.peri, theme.mint])
    ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=mat.max())
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(list(x_labels), fontsize=8, color=theme.ink_400)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(list(y_labels), fontsize=8, color=theme.ink_400)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j]:
                ax.text(j, i, compact_currency(mat[i, j]), ha="center", va="center",
                        color=theme.ink_100, fontsize=6.8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return _save(fig, path, theme, dpi)


def draw_lines(path, x_labels: Sequence[str], series: Sequence[Dict[str, Any]],
               w: float, h: float, *, theme: PptxTheme = THEME,
               currency: bool = True, percent: bool = False, area: bool = False,
               dpi: int = 220) -> Path:
    """Dashboard line/area chart. *series* = [{name, values, color?}]."""
    fig = _fig(w, h, theme, dpi)
    ax = fig.add_axes([0.10, 0.16, 0.88, 0.74 if len(series) > 1 else 0.80])
    ax.set_facecolor(theme.bg_panel)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(theme.line_soft)
    ax.tick_params(colors=theme.ink_500, labelsize=9, length=0)
    ax.grid(axis="y", color=theme.line_soft, linewidth=0.7,
            linestyle=(0, (3, 3)), alpha=0.9)
    ax.set_axisbelow(True)

    n = len(x_labels)
    x = list(range(n))
    if not n or not series:
        ax.text(0.5, 0.5, "Insufficient history", ha="center", va="center",
                transform=ax.transAxes, color=theme.ink_500, fontsize=12)
        ax.axis("off")
        return _save(fig, path, theme, dpi)

    for i, s in enumerate(series):
        vals = [None if v is None else float(v) for v in s.get("values", [])]
        color = s.get("color") or EVO_PALETTE[i % len(EVO_PALETTE)]
        ax.plot(x, vals, color=color, linewidth=2.4, marker="o", markersize=3,
                label=s.get("name", ""), zorder=3, solid_capstyle="round")
        if area and len(series) == 1:
            ax.fill_between(x, [v or 0 for v in vals], color=color, alpha=0.16, zorder=2)

    if currency:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: compact_currency(v)))
    elif percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v * 100:.0f}%"
                                                   if v <= 1.5 else f"{v:.0f}%"))
    step = max(1, n // 7)
    idx = sorted(set(list(range(0, n, step)) + [n - 1]))
    ax.set_xticks([x[i] for i in idx])
    ax.set_xticklabels([str(x_labels[i]) for i in idx], fontsize=8.5,
                       color=theme.ink_500)
    if len(series) > 1:
        leg = ax.legend(loc="upper left", fontsize=8.5, frameon=False,
                        ncol=min(len(series), 3), handlelength=1.4)
        for t in leg.get_texts():
            t.set_color(theme.ink_300)
    return _save(fig, path, theme, dpi)


def draw_table(path, columns: Sequence[str], rows: Sequence[Sequence[Any]],
               w: float, h: float, *, theme: PptxTheme = THEME,
               status_col: Optional[int] = None, dpi: int = 220) -> Path:
    """Compact dark table (risk category tables). *rows* are pre-formatted str
    cells; ``status_col`` colours a RAG status cell."""
    fig = _fig(w, h, theme, dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(theme.bg_panel)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ncol = len(columns)
    rag = {"green": theme.rag["green"], "amber": theme.rag["amber"],
           "red": theme.rag["red"], "needs_review": theme.ink_400,
           "unavailable": theme.ink_500}
    # column x positions: first column wide (label), rest even.
    xs = [0.02] + list(np.linspace(0.42, 0.98, ncol - 1)) if ncol > 1 else [0.02]
    header_y = 0.94
    for c, col in enumerate(columns):
        ha = "left" if c == 0 else "right"
        ax.text(xs[c], header_y, col, ha=ha, va="center", color=theme.ink_400,
                fontsize=9, fontweight="bold")
    ax.plot([0.02, 0.98], [0.90, 0.90], color=theme.line, linewidth=0.8)
    rh = 0.85 / max(len(rows), 1)
    for r, row in enumerate(rows):
        y = 0.86 - (r + 0.5) * rh
        for c, cell in enumerate(row):
            ha = "left" if c == 0 else "right"
            color = theme.ink_200 if hasattr(theme, "ink_200") else theme.ink_300
            fp = None if c == 0 else _MONO_FP
            if status_col is not None and c == status_col:
                color = rag.get(str(cell).lower(), theme.ink_300)
            ax.text(xs[c], y, str(cell), ha=ha, va="center", color=color,
                    fontsize=9.5, fontproperties=fp)
    return _save(fig, path, theme, dpi)
