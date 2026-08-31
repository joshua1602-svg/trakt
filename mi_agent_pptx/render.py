"""mi_agent_pptx.render — draw charts from explicit dashboard payload rows/series.

These are the low-level renderers the payload-driven deck uses. They take the
data *verbatim from the MI API payloads* (BarList rows, evolution series, bridge
steps, risk tables) — no aggregation — so the visual is a faithful export of the
dashboard's Recharts/BarList/stat-tile components. Each renders at the exact
width×height of its slide panel and onto the theme panel background.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
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


# --------------------------------------------------------------------------- #
# Render record
#
# A bar list is drawn as a PNG, so the category labels it drew are not text in
# the finished .pptx and cannot be read back out of the file. That made the one
# thing most worth checking — did the deck draw the bands in the governed order?
# — the one thing no test could see, which is exactly how the deck and the
# dashboard came to disagree about it.
#
# Each renderer therefore records WHAT IT DREW, at the moment it draws it. This
# is not the payload the deck intended to render: it is the sequence the drawing
# function actually walked, captured inside that function. The record travels
# into the deck's preflight sidecar, where a publication gate checks it and any
# reader can audit it.
# --------------------------------------------------------------------------- #

_RENDER_RECORD: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = \
    contextvars.ContextVar("pptx_render_record", default=None)


@contextmanager
def record_renders():
    """Collect a record of every chart drawn inside the block."""
    entries: List[Dict[str, Any]] = []
    token = _RENDER_RECORD.set(entries)
    try:
        yield entries
    finally:
        _RENDER_RECORD.reset(token)


def _record(kind: str, chart_id: Optional[str], **fields: Any) -> None:
    """Append one drawn-chart entry, when a recorder is active."""
    entries = _RENDER_RECORD.get()
    if entries is None:
        return
    entry: Dict[str, Any] = {"kind": kind, "chart": chart_id}
    entry.update(fields)
    entries.append(entry)


def _fig(w, h, theme, dpi=220):
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    fig.patch.set_facecolor(theme.bg_panel)
    return fig


def _save(fig, path, theme, dpi=220):
    fig.savefig(Path(path), facecolor=theme.bg_panel, dpi=dpi)
    plt.close(fig)
    return Path(path)


# --------------------------------------------------------------------------- #
# Plot geometry.
#
# A left margin expressed as a FRACTION of the figure scales with the figure,
# and the thing it has to clear does not. The widest y tick — "£800.0MM" — is
# about seven tenths of an inch whether it sits beside a 5.8in panel or a
# 12.25in full-width chart, but 0.145 of the figure reserves 0.84in on the
# first and 1.78in on the second. That is where the empty left-hand band on
# Funded Stock and Funded Balance Movement came from: not a chart drawn too
# small, a gutter sized for a figure three times narrower.
#
# So margins are computed from the INCHES the labels actually need and then
# expressed as a fraction, which keeps a narrow panel exactly as it was and
# hands the width back on a wide one.
# --------------------------------------------------------------------------- #

#: Average glyph width as a fraction of the font's point size, for the sans
#: face the theme uses. Deliberately generous: under-reserving clips a tick,
#: which is a defect, while over-reserving costs a little width.
_GLYPH_EM = 0.62

#: Clear air between the longest tick label and the plot's left edge.
_TICK_GAP_IN = 0.14


def _text_in(text: str, pt: float) -> float:
    """Roughly how wide *text* draws at *pt*, in inches."""
    return len(str(text or "")) * pt * _GLYPH_EM / 72.0


def axis_left(w: float, tick_samples: Sequence[Any], *, pt: float = 9.0,
              floor_in: float = 0.30, cap_frac: float = 0.34) -> float:
    """The left margin, as a fraction of *w*, that these tick labels need.

    ``floor_in`` keeps a small chart from crowding its axis; ``cap_frac`` stops
    a pathological label from eating the plot. Both are in the units they
    describe — inches for the floor, a fraction for the cap — because that is
    what each one is actually protecting.
    """
    widest = max((_text_in(t, pt) for t in tick_samples if t not in (None, "")),
                 default=0.0)
    needed = max(floor_in, widest + _TICK_GAP_IN)
    return min(cap_frac, needed / max(float(w), 1e-6))


def _money_ticks(values: Sequence[Any], fmt) -> List[str]:
    """Sample tick labels for a value range, formatted the way the axis will.

    The axis formatter runs after the axes exist, so the widest label cannot be
    measured before choosing the margin. The extremes of the data formatted the
    same way are what the widest tick will look like.
    """
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return []
    lo, hi = min(nums), max(nums)
    try:
        return [str(fmt(lo)), str(fmt(hi)), str(fmt((lo + hi) / 2.0))]
    except Exception:  # noqa: BLE001 - a sample must never break a chart
        return []


def _truncate(label: str, max_chars: int) -> str:
    return label if len(label) <= max_chars else label[:max_chars - 1].rstrip() + "…"


def draw_barlist(path, rows: Sequence[Dict[str, Any]], value_key: str, w: float,
                 h: float, *, theme: PptxTheme = THEME, currency: bool = True,
                 label_key: str = "label", count_key: Optional[str] = "count",
                 dpi: int = 220, chart_id: Optional[str] = None,
                 dimension: Optional[str] = None) -> Path:
    """Dashboard BarList: label left, periwinkle bar ∝ max, mono value right.

    Bars are drawn in the order given. The ORDER IS NOT DECIDED HERE — it is
    decided once, upstream, by ``mi_agent_api.presentation`` against the governed
    bucket ladder, and both this renderer and the React bar list consume it. The
    sequence drawn is recorded (see :func:`record_renders`) so a publication gate
    can check it against that ladder.
    """
    rows = [r for r in rows if r is not None]
    _record("barlist", chart_id, dimension=dimension,
            categories=[str(r.get(label_key, "")) for r in rows],
            values=[r.get(value_key) for r in rows], currency=currency)
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
    # The label column carries real region and broker names. At 0.335 the widest
    # standard UK region ("Yorkshire and The Humber", 24 characters) truncated on
    # every deck; a truncated dimension label is a legibility defect, not a
    # styling choice.
    label_x, tx0, tx1 = 0.005, 0.385, 0.86
    tw = tx1 - tx0

    # TYPE SIZE FOLLOWS THE ROW BAND. At a fixed 10.5pt, a panel with more rows
    # than vertical room drew its labels on top of one another — which is what
    # the forecast-by-region cut did at seven rows in an inch. The size is
    # derived from the height each row actually gets, and floored at the
    # smallest size that is still readable on a projected slide; below that the
    # panel is genuinely too small for this many rows, and the caller (not the
    # renderer) should be showing fewer.
    row_in = band * h
    font = max(7.5, min(10.5, row_in * 72.0 * 0.52))
    max_chars = max(10, int((tx0 - label_x) * w * 72 / (font * 0.56)))
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
                color=theme.ink_300, fontsize=font, zorder=3)
        ax.text(0.995, yc, fmt(val), va="center", ha="right", color=theme.ink_100,
                fontsize=font, fontproperties=_MONO_FP, zorder=3)
    return _save(fig, path, theme, dpi)


def draw_stacked_barlist(path, rows: Sequence[Dict[str, Any]],
                         segments: Sequence[Dict[str, Any]], w: float, h: float,
                         *, theme: PptxTheme = THEME, currency: bool = True,
                         label_key: str = "label", total_key: str = "total",
                         dpi: int = 220, chart_id: Optional[str] = None,
                         dimension: Optional[str] = None) -> Path:
    """A bar list whose bar is BUILT from its parts.

    ``segments`` is ``[{key, label, color}]`` in stacking order; each row
    carries a value under every segment key, and the row's ``total_key`` is
    what the parts must sum to. The right-hand figure is that total.

    A forecast bar drawn as one block shows the destination and hides the
    journey: the reader cannot see how much of a category's forecast exposure
    is already funded and how much is expected to arrive. Those are different
    facts with different certainty, and a funder is buying one of them.

    Nothing is summed here beyond drawing: the caller supplies the parts and
    the total, both from the governed payload, and a row whose parts do not
    reach its total is drawn short rather than rescaled — a chart must not
    hide a reconciliation failure.
    """
    rows = [r for r in rows if r is not None]
    _record("stacked_barlist", chart_id, dimension=dimension,
            categories=[str(r.get(label_key, "")) for r in rows],
            values=[r.get(total_key) for r in rows],
            segments=[str(sg.get("key")) for sg in segments], currency=currency)
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
    totals = [float(r.get(total_key) or 0.0) for r in rows]
    labels = [str(r.get(label_key, "")) for r in rows]
    n = len(rows)
    vmax = max(max(totals), 1.0)
    pad_top, pad_bot = 0.16, 0.05
    band = (1.0 - pad_top - pad_bot) / max(n, 1)
    bar_h = min(band * 0.62, 0.135)
    label_x, tx0, tx1 = 0.005, 0.385, 0.86
    tw = tx1 - tx0
    row_in = band * h
    font = max(7.5, min(10.5, row_in * 72.0 * 0.52))
    max_chars = max(10, int((tx0 - label_x) * w * 72 / (font * 0.56)))

    for i, (lab, total) in enumerate(zip(labels, totals)):
        yc = 1.0 - pad_top - (i + 0.5) * band
        y0 = yc - bar_h / 2
        ax.add_patch(mpatches.FancyBboxPatch(
            (tx0, y0), tw, bar_h, boxstyle="round,pad=0,rounding_size=0.012",
            linewidth=0, facecolor=theme.bg_panel_alt, alpha=0.7,
            mutation_aspect=h / w, zorder=1))
        cursor = tx0
        for sg in segments:
            value = float(rows[i].get(sg["key"]) or 0.0)
            if value <= 0:
                continue
            width = tw * max(value / vmax, 0.0)
            ax.add_patch(mpatches.Rectangle(
                (cursor, y0), width, bar_h, linewidth=0,
                facecolor=sg.get("color") or theme.peri, alpha=0.92, zorder=2))
            cursor += width
        ax.text(label_x, yc, _truncate(lab, max_chars), va="center", ha="left",
                color=theme.ink_300, fontsize=font, zorder=3)
        ax.text(0.995, yc, fmt(total), va="center", ha="right",
                color=theme.ink_100, fontsize=font, fontproperties=_MONO_FP,
                zorder=3)

    # The key, above the bars: two colours mean nothing without it.
    x = tx0
    for sg in segments:
        ax.add_patch(mpatches.Rectangle(
            (x, 1.0 - pad_top + 0.035), 0.018, 0.045, linewidth=0,
            facecolor=sg.get("color") or theme.peri, alpha=0.92, zorder=3))
        ax.text(x + 0.026, 1.0 - pad_top + 0.058, str(sg.get("label", "")),
                va="center", ha="left", color=theme.ink_400,
                fontsize=max(7.5, font - 1.0), zorder=3)
        x += 0.026 + len(str(sg.get("label", ""))) * 0.0105 + 0.03
    return _save(fig, path, theme, dpi)



def _tick_indices(x_labels: Sequence[str], axis_width_in: float,
                  fontsize: float = 8.5) -> List[int]:
    """Which x positions can carry a label WITHOUT colliding.

    The old rule was ``n // 7``, which ignores how wide the labels are: ten
    weekly extracts labelled ``2026-04-24`` produced ten ISO dates in six inches
    and rendered as one illegible band. This derives the count from the widest
    label and the axis it has to fit in, and always keeps the first and last so
    the reader can see the window the series covers.
    """
    n = len(x_labels)
    if n <= 1:
        return list(range(n))
    widest = max((len(str(l)) for l in x_labels), default=1)
    label_in = widest * fontsize * 0.55 / 72.0
    fits = max(2, int(axis_width_in / (label_in * 1.35))) if label_in else n
    if fits >= n:
        return list(range(n))
    step = max(1, -(-n // fits))
    idx = list(range(0, n, step))
    # Keeping the last position is what lets a reader see where the series ends,
    # but it lands wherever it lands — often one place after a stepped tick, and
    # the two then print on top of each other. The stepped neighbour goes.
    while idx and n - 1 - idx[-1] < step:
        idx.pop()
    return idx + [n - 1]


def draw_bars_with_line(path, x_labels: Sequence[str], bars: Sequence[Optional[float]],
                        line: Sequence[Optional[float]], w: float, h: float, *,
                        theme: PptxTheme = THEME, bar_currency: bool = True,
                        avg: Optional[float] = None, line_label: str = "Cumulative",
                        dpi: int = 220) -> Path:
    """Weekly-flow bars (periwinkle, left axis) + a cumulative line (mint, right
    axis), with an optional dashed 5-week-average marker — the dashboard's
    KFI/Completions weekly-flow panel."""
    fig = _fig(w, h, theme, dpi)
    # Left margin fits a full compact-currency tick ('£800.0MM'), measured
    # rather than guessed at a fraction of the figure.
    _fmt = compact_currency if bar_currency else compact_number
    left = axis_left(w, _money_ticks([v for v in list(bars) + list(line)
                                      if v is not None], _fmt), pt=8.5)
    ax = fig.add_axes([left, 0.16, 0.955 - left, 0.78])
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
    idx = _tick_indices(x_labels, w * (0.955 - left), fontsize=8)
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
                 theme: PptxTheme = THEME, dpi: int = 220,
                 chart_id: Optional[str] = None,
                 x_dimension: Optional[str] = None,
                 y_dimension: Optional[str] = None) -> Path:
    """Balance heatmap: rows=y_labels, cols=x_labels, cell shade ∝ balance, with
    the £ value annotated. Uses the periwinkle→mint brand ramp."""
    from matplotlib.colors import LinearSegmentedColormap
    _record("heatmap", chart_id, categories=[str(x) for x in x_labels],
            rows=[str(y) for y in y_labels],
            dimension=x_dimension, row_dimension=y_dimension)
    fig = _fig(w, h, theme, dpi)
    # The row labels are real category names — "Yorkshire and The Humber" is 24
    # characters — and a truncated dimension label is a legibility defect on a
    # slide whose whole purpose is comparing categories. The left margin is
    # derived from the widest label rather than fixed at 0.20, which clipped it.
    widest = max((len(str(l)) for l in y_labels), default=6)
    label_pt = 8.5 if widest <= 20 else 7.5
    left = min(0.34, max(0.20, widest * label_pt * 0.55 / 72.0 / max(w, 1.0) + 0.03))
    ax = fig.add_axes([left, 0.16, 0.975 - left, 0.78])
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


def _currency_tick_formatter(ax, series, stack):
    """A money tick formatter that cannot label two gridlines the same.

    Compact currency rounds to one decimal at millions, so its finest step is
    0.1MM. A series living between 109.05m and 109.14m therefore labels every
    gridline "£109.1MM" — four identical labels, which reads as a rendering
    fault rather than as a flat series.

    The test is the compact notation's own RESOLUTION against the axis range,
    not an absolute range: where the range spans fewer than about five compact
    steps, the axis carries more decimal places instead. Anything wider keeps
    the compact form every other money label in the pack uses.
    """
    values = [float(v) for sr in series for v in (sr.get("values") or ())
              if v is not None]
    if not values:
        return lambda v, p: compact_currency(v)
    if stack:
        columns = zip(*[[float(v or 0.0) for v in (sr.get("values") or ())]
                        for sr in series])
        totals = [sum(col) for col in columns] or [0.0]
        low, high = 0.0, max(totals)
    else:
        low, high = min(values), max(values)
    span, magnitude = high - low, max(abs(low), abs(high))
    unit, suffix = ((1e9, "BN") if magnitude >= 1e9 else
                    (1e6, "MM") if magnitude >= 1e6 else
                    (1e3, "K") if magnitude >= 1e3 else (1.0, ""))
    step = unit * (0.01 if suffix == "BN" else 0.1 if suffix == "MM" else 1.0)
    if span <= 0 or span >= step * 5:
        return lambda v, p: compact_currency(v)
    # Enough decimals for five ticks to be distinct at this unit.
    dp = 2
    while dp < 6 and span / unit < 5 * (10 ** -dp):
        dp += 1
    from mi_agent_api import currency as _cur
    return lambda v, p: f"{_cur.current_symbol()}{v / unit:,.{dp}f}{suffix}"


def draw_lines(path, x_labels: Sequence[str], series: Sequence[Dict[str, Any]],
               w: float, h: float, *, theme: PptxTheme = THEME,
               currency: bool = True, percent: bool = False, area: bool = False,
               dpi: int = 220, chart_id: Optional[str] = None,
               stack: bool = False, zero_based: Optional[bool] = None) -> Path:
    """Dashboard line/area chart. *series* = [{name, values, color?}].

    ``stack`` draws the series as a stacked area — the right grammar for a STOCK
    split into parts that sum to a total, where a set of separate lines would
    make the reader add them up by eye.

    ``zero_based`` forces the value axis to include zero. Default (``None``)
    decides it: a stock or a stacked series is anchored at zero, because a
    magnitude read off a floating baseline exaggerates every movement; a rate
    or a narrow-range series is not, because zero-anchoring it would flatten the
    only variation it has.
    """
    _record("lines", chart_id, categories=[str(x) for x in x_labels],
            series=[str(s.get("name", "")) for s in series],
            currency=currency, percent=percent)
    fig = _fig(w, h, theme, dpi)
    # Left margin fits the widest tick this data will actually draw; the axes
    # top leaves a clear band for the legend, which is drawn ABOVE the plot
    # rather than inside it — placed inside, it landed on the series it
    # described.
    _vals = [v for sr in series for v in (sr.get("values") or ()) if v is not None]
    if stack and _vals:
        # A stacked chart's axis reaches the SUM of the series at a point, not
        # the largest single value, so the tick it has to clear is wider.
        _cols = zip(*[[float(v or 0.0) for v in (sr.get("values") or ())]
                      for sr in series]) if len(series) > 1 else ()
        _vals = _vals + [sum(col) for col in _cols]
    if percent:
        _samples = ["100.0%"]
    elif currency:
        _samples = _money_ticks(_vals, compact_currency)
    else:
        _samples = _money_ticks(_vals, compact_number)
    left = axis_left(w, _samples, pt=9.0)
    ax = fig.add_axes([left, 0.16, 0.965 - left,
                       0.70 if len(series) > 1 else 0.78])
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

    if stack and len(series) > 1:
        # Stacked area: parts of one total. Drawn bottom-up in the order given,
        # so the caller's ordering (largest book first) is what the reader sees.
        stacked = [[float(v or 0.0) for v in s.get("values", [])] for s in series]
        colours = [s.get("color") or EVO_PALETTE[i % len(EVO_PALETTE)]
                   for i, s in enumerate(series)]
        ax.stackplot(x, *stacked, colors=colours, alpha=0.88,
                     labels=[s.get("name", "") for s in series],
                     edgecolor=theme.bg_panel, linewidth=0.6, zorder=2)
    else:
        for i, s in enumerate(series):
            vals = [None if v is None else float(v) for v in s.get("values", [])]
            color = s.get("color") or EVO_PALETTE[i % len(EVO_PALETTE)]
            ax.plot(x, vals, color=color, linewidth=2.4, marker="o", markersize=3,
                    label=s.get("name", ""), zorder=3, solid_capstyle="round")
            if area and len(series) == 1:
                ax.fill_between(x, [v or 0 for v in vals], color=color,
                                alpha=0.16, zorder=2)

    # AXIS MATERIALITY. A stock chart on a floating baseline turns a fractional
    # move into a cliff. Anchor a currency/stacked axis at zero unless the caller
    # says otherwise; leave a rate axis alone, where zero-anchoring would flatten
    # the only variation there is.
    anchor = zero_based if zero_based is not None else (stack or (currency and not percent))
    if anchor:
        shown = [float(v) for sr in series for v in (sr.get("values") or ())
                 if v is not None]
        if shown and min(shown) >= 0:
            top = max(sum(vals) for vals in zip(*[[float(v or 0.0) for v in
                      (sr.get("values") or ())] for sr in series])) if stack and len(series) > 1 \
                  else max(shown)
            ax.set_ylim(0, top * 1.08 if top else 1.0)

    if currency:
        # DUPLICATE TICKS. Compact currency rounds to one decimal at millions,
        # so a series that lives between 109.05m and 109.14m labels every
        # gridline "£109.1MM" — four identical labels, which reads as a
        # rendering fault rather than as a flat series. Where the ticks would
        # collide the axis carries a scaled label and states its unit once.
        ax.yaxis.set_major_formatter(FuncFormatter(
            _currency_tick_formatter(ax, series, stack)))
    elif percent:
        # Decimals follow the RANGE, not a constant. A weighted LTV that moves
        # between 45.8% and 47.2% produced five ticks all reading "46%" — an
        # axis that labels four distinct gridlines identically is worse than no
        # axis, because it reads as a rendering fault rather than a flat series.
        shown = [float(v) for sr in series for v in (sr.get("values") or ())
                 if v is not None]
        as_points = [v * 100 if abs(v) <= 1.5 else v for v in shown]
        spread = (max(as_points) - min(as_points)) if as_points else 0.0
        dp = 0 if spread >= 6 else (1 if spread >= 0.6 else 2)
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, p: f"{v * 100:.{dp}f}%" if abs(v) <= 1.5 else f"{v:.{dp}f}%"))
    idx = _tick_indices(x_labels, w * (0.965 - left), fontsize=8.5)
    ax.set_xticks([x[i] for i in idx])
    ax.set_xticklabels([str(x_labels[i]) for i in idx], fontsize=8.5,
                       color=theme.ink_500)
    if len(series) > 1:
        # ONE row. At ncol=3 a fourth series wrapped onto a second row that the
        # axes' headroom did not allow for, and the wrapped entry printed over
        # the row above it.
        leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02),
                        fontsize=8.5 if len(series) <= 4 else 7.5, frameon=False,
                        ncol=len(series), handlelength=1.4, columnspacing=1.4)
        for t in leg.get_texts():
            t.set_color(theme.ink_300)
    return _save(fig, path, theme, dpi)


#: Status -> RAG key, for both the approved concentration vocabulary
#: ("pass"/"warning"/"breach") and the legacy monitor ("green"/"amber"/"red").
_STATUS_RAG = {
    "pass": "green", "green": "green", "ok": "green",
    "warning": "amber", "amber": "amber", "warn": "amber",
    "breach": "red", "red": "red", "fail": "red",
}


def draw_diverging(path, rows: Sequence[Dict[str, Any]], w: float, h: float, *,
                   theme: PptxTheme = THEME, currency: bool = True,
                   dpi: int = 220) -> Path:
    """Movement by category as diverging bars around zero.

    Increases right, reductions left, ordered by magnitude. This is the shape
    that answers "what moved" at a glance — a stacked composition chart shows
    the level and hides the change, which is the opposite of what a period
    report needs.
    """
    fig = _fig(w, h, theme, dpi)
    ax = fig.add_axes([0.34, 0.08, 0.62, 0.88])
    ax.set_facecolor(theme.bg_panel)
    for spine in ax.spines.values():
        spine.set_visible(False)
    rows = list(rows)
    n = max(len(rows), 1)
    values = [float(r.get("delta") or 0.0) for r in rows]
    # 1.35 left no room for the value label on the LONGEST bar: drawn outside
    # the bar end, it ran into the category name beside it. Widen when any
    # value is negative, which is the side the category names sit on.
    headroom = 1.75 if any(v < 0 for v in values) else 1.45
    span = max((abs(v) for v in values), default=1.0) * headroom or 1.0
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.6, n - 0.4)
    ax.invert_yaxis()
    for i, (row, value) in enumerate(zip(rows, values)):
        colour = theme.mint if value >= 0 else theme.rose
        if row.get("is_other"):
            colour = theme.ink_500
        ax.barh(i, value, height=0.56, color=colour, edgecolor="none")
        label = (_fmt_money(value, signed=True) if currency
                 else f"{value:+,.1f}")
        offset = span * 0.02
        ax.text(value + (offset if value >= 0 else -offset), i, label,
                ha="left" if value >= 0 else "right", va="center",
                color=theme.ink_100, fontsize=9, fontproperties=_MONO_FP)
    ax.axvline(0, color=theme.ink_400, linewidth=1.0, alpha=0.85)
    ax.set_yticks(list(range(len(rows))))
    ax.set_yticklabels([_truncate(str(r.get("category", "")), 26) for r in rows])
    ax.tick_params(axis="y", colors=theme.ink_300, labelsize=9, length=0, pad=6)
    ax.set_xticks([])
    ax.grid(axis="x", color=theme.line_soft, linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)
    return _save(fig, path, theme, dpi)


def _fmt_money(value: float, *, signed: bool = False) -> str:
    """Chart labels in the SAME notation as the KPI tiles and the dashboard.

    These used to render ``+£111.6m`` while the axis beside them — formatted by
    ``compact_currency`` — rendered ``£800.0MM``, so one chart carried two
    conventions. ``compact_currency`` is the dashboard's own notation
    (``formatGBP`` in the React client), so delegating to it makes the deck
    internally consistent AND consistent with the screen it mirrors.
    """
    from .metric_resolver import compact_currency, signed_currency
    return signed_currency(value) if signed else compact_currency(value)


def draw_utilisation_tests(path, tests: Sequence[Dict[str, Any]], w: float, h: float,
                           *, theme: PptxTheme = THEME, dpi: int = 220, chart_id: Optional[str] = None) -> Path:
    """Concentration tests as horizontal utilisation bars against their limit.

    One row per test. The bar is utilisation of the contractual limit, so the
    100% gridline IS the limit and proximity is readable at a glance — the thing
    an investor actually asks of a covenant table.

    Each row can carry up to three marks, and they are deliberately visually
    distinct because conflating them would be the whole failure mode:

      * the filled bar  — CURRENT funded, the only actual;
      * a hollow caret  — EXPECTED forecast;
      * a hatched tick  — the ALL-PIPELINE-CONVERTS stress (never an expectation).
    """
    _record("utilisation", chart_id,
            categories=[str(x.get("label", "")) for x in tests],
            statuses=[str(x.get("status", "")) for x in tests])
    fig = _fig(w, h, theme, dpi)
    ax = fig.add_axes([0.30, 0.10, 0.62, 0.84])
    ax.set_facecolor(theme.bg_panel)
    for spine in ax.spines.values():
        spine.set_visible(False)
    n = max(len(tests), 1)
    # Scale so a breach is visible beyond the limit line without dwarfing the rest.
    vals = [t.get("utilisation") or 0 for t in tests]
    vals += [(t.get("expectedUtilisation") or 0) for t in tests]
    vals += [(t.get("stressUtilisation") or 0) for t in tests]
    top = max(120.0, min(200.0, (max(vals) if vals else 100) * 1.12))
    ax.set_xlim(0, top)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()

    for i, t in enumerate(tests):
        util = float(t.get("utilisation") or 0)
        rag = _STATUS_RAG.get(str(t.get("status", "")).lower(), "green")
        colour = theme.rag.get(rag, theme.neutral)
        # Track, then the filled current bar.
        ax.barh(i, top, height=0.52, color=theme.bg_panel_alt, edgecolor="none")
        ax.barh(i, min(util, top), height=0.52, color=colour, edgecolor="none")
        ax.text(min(util, top) + 0.012 * top, i, f"{util:.0f}%", ha="left",
                va="center", color=theme.ink_100, fontsize=9,
                fontproperties=_MONO_FP)
        exp = t.get("expectedUtilisation")
        if exp is not None:
            ax.plot([min(float(exp), top)], [i], marker="v", markersize=6,
                    markerfacecolor="none",
                    markeredgecolor=theme.peri, markeredgewidth=1.4)
        stress = t.get("stressUtilisation")
        if stress is not None:
            ax.plot([min(float(stress), top)], [i - 0.30], marker="|",
                    markersize=9, color=theme.ink_400, markeredgewidth=1.6)

    # The limit.
    ax.axvline(100, color=theme.ink_400, linewidth=1.1, linestyle="--", alpha=0.9)
    ax.text(100, -0.62, "limit", ha="center", va="bottom", color=theme.ink_400,
            fontsize=8.5)
    # Test names as tick labels — a bar an investor cannot name is not evidence.
    ax.set_yticks(list(range(len(tests))))
    ax.set_yticklabels([_truncate(str(t.get("label", "")), 30) for t in tests])
    ax.tick_params(axis="y", colors=theme.ink_300, labelsize=9, length=0, pad=6)
    ax.tick_params(axis="x", colors=theme.ink_500, labelsize=8.5, length=0)
    ax.set_xticks([0, 50, 100] + ([150] if top > 150 else []))
    ax.set_xticklabels([f"{v}%" for v in ([0, 50, 100] + ([150] if top > 150 else []))])
    ax.grid(axis="x", color=theme.line_soft, linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    return _save(fig, path, theme, dpi)


def draw_table(path, columns: Sequence[str], rows: Sequence[Sequence[Any]],
               w: float, h: float, *, theme: PptxTheme = THEME,
               status_col: Optional[int] = None, dpi: int = 220,
               chart_id: Optional[str] = None) -> Path:
    """Compact dark table (risk category tables). *rows* are pre-formatted str
    cells; ``status_col`` colours a RAG status cell."""
    _record("table", chart_id, columns=[str(c) for c in columns],
            cells=[[str(c) for c in row] for row in rows],
            status_col=status_col)
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
