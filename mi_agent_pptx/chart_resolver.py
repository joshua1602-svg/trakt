"""mi_agent_pptx.chart_resolver — dashboard-faithful static chart rendering.

Renders the deck's charts as PNGs that reproduce the MI Agent **React dashboard**
visual language (Recharts / BarList / heatmap), so the pack reads as a direct,
automated export of the dashboard rather than generic plotting output:

* the signature breakdown visual is the dashboard **BarList** — a horizontal bar
  with the category label on the left, a periwinkle fill proportional to the
  max, and a right-aligned mono value (``£X.XMM``);
* time series use monotone lines with a top→bottom gradient area fill;
* the heatmap uses the dashboard's navy→periwinkle→mint ramp with contrast-
  flipping cell values;
* every figure is rendered at the **exact width×height of its slide panel** so
  python-pptx never stretches it (the root cause of the earlier distortion).

All aggregation is delegated to the registry-authorised ``analytics_lib``. Each
resolver is bound to a single lens's frame; a slide routes to the resolver for
its lens, so pipeline charts never render funded data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data_resolver import ResolvedData  # noqa: E402
from .metric_resolver import compact_currency, compact_number, format_percent  # noqa: E402
from .placeholders import render_placeholder_png  # noqa: E402
from .pptx_theme import PptxTheme, THEME  # noqa: E402
from .registry_loader import RegistryLoader  # noqa: E402

# ---- fonts: prefer a clean humanist sans; mono for numeric values ----------
_SANS = next((f for f in ("Inter", "Liberation Sans", "DejaVu Sans")
              if f in {ff.name for ff in fm.fontManager.ttflist}), "DejaVu Sans")
_MONO = next((f for f in ("Liberation Mono", "DejaVu Sans Mono")
              if f in {ff.name for ff in fm.fontManager.ttflist}), "DejaVu Sans Mono")
plt.rcParams.update({
    "font.family": _SANS,
    "font.size": 11,
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
})
_MONO_FP = fm.FontProperties(family=_MONO)


@dataclass
class ChartResult:
    chart_id: str
    title: str
    path: Optional[Path] = None
    available: bool = False
    placeholder: bool = False
    note: str = ""
    kind: str = "barlist"

    @property
    def ok(self) -> bool:
        return self.available and self.path is not None


def _currency_axis(_x, _pos):
    return compact_currency(_x)


def render_bridge_waterfall(out_path, steps, width_in, height_in, theme=THEME,
                            dpi=220):
    """Render a forecast-bridge waterfall (funded → +weighted pipeline → forecast).

    *steps* is an ordered list of ``(label, value, kind)`` where kind is
    ``base`` / ``add`` / ``sub`` / ``total``. Mirrors the dashboard waterfall
    colours (base navy, add periwinkle, total mint).
    """
    from pathlib import Path as _P
    colors = {"base": theme.navy, "add": theme.peri, "sub": theme.negative,
              "total": theme.mint}
    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    fig.patch.set_facecolor(theme.bg_panel)
    ax = fig.add_axes([0.11, 0.14, 0.86, 0.80])
    ax.set_facecolor(theme.bg_panel)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(theme.line_soft)
    ax.tick_params(colors=theme.ink_300, labelsize=9.5, length=0)
    ax.grid(axis="y", color=theme.line_soft, linewidth=0.7,
            linestyle=(0, (3, 3)), alpha=0.9)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(_currency_axis))

    running = 0.0
    xs = list(range(len(steps)))
    prev_top = None
    for i, (label, value, kind) in enumerate(steps):
        if kind in ("base", "total"):
            bottom, top = 0.0, value
            running = value
        elif kind == "add":
            bottom, top = running, running + value
            running += value
        else:  # sub
            bottom, top = running - value, running
            running -= value
        ax.bar(i, top - bottom, bottom=bottom, width=0.62,
               color=colors.get(kind, theme.peri),
               edgecolor=theme.bg_panel, linewidth=0.5, zorder=3)
        # connector line
        if prev_top is not None and kind not in ("total",):
            ax.plot([i - 1 + 0.31, i - 0.31], [prev_top, prev_top],
                    color=theme.ink_500, linewidth=0.8, linestyle=(0, (2, 2)),
                    zorder=2)
        elif prev_top is not None and kind == "total":
            ax.plot([i - 1 + 0.31, i - 0.31], [prev_top, prev_top],
                    color=theme.ink_500, linewidth=0.8, linestyle=(0, (2, 2)),
                    zorder=2)
        ax.text(i, top + (max(v for _, v, _ in steps) * 0.02), compact_currency(value),
                ha="center", va="bottom", color=theme.ink_100, fontsize=10.5,
                fontproperties=_MONO_FP, zorder=4)
        prev_top = top
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for lab, _, _ in steps], fontsize=9.5,
                       color=theme.ink_300)
    ax.set_ylim(0, max(v for _, v, k in steps if k in ("base", "total", "add")) * 1.16)
    fig.savefig(_P(out_path), facecolor=theme.bg_panel, dpi=dpi)
    plt.close(fig)
    return _P(out_path)


class ChartResolver:
    """Render a single lens's chart specs into dashboard-faithful PNGs."""

    def __init__(
        self,
        data: Optional[ResolvedData],
        registries: RegistryLoader,
        out_dir: str | Path,
        theme: PptxTheme = THEME,
        dpi: int = 220,
        lens: str = "funded",
    ):
        self.data = data
        self.reg = registries
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.theme = theme
        self.dpi = dpi
        self.lens = lens

    # ------------------------------------------------------------------ public
    def resolve(self, spec: Dict[str, Any], width_in: float, height_in: float) -> ChartResult:
        chart_id = spec.get("id", "chart")
        title = spec.get("title", chart_id.replace("_", " ").title())
        kind = self._normalise_kind(spec.get("type", "barlist"))
        if self.data is None or self.data.df is None or self.data.df.empty:
            return self._placeholder(
                chart_id, title, kind, width_in, height_in,
                f"{self.lens.title()} lens data not available for this run.")
        try:
            handler = getattr(self, f"_render_{kind}", None)
            if handler is None:
                return self._placeholder(chart_id, title, kind, width_in, height_in,
                                         f"Unsupported chart type '{kind}'.")
            return handler(spec, chart_id, title, width_in, height_in)
        except Exception as exc:  # pragma: no cover - defensive
            return self._placeholder(chart_id, title, kind, width_in, height_in,
                                     f"Chart render error: {exc}")

    @staticmethod
    def _normalise_kind(kind: str) -> str:
        # Map legacy/simple kinds onto the dashboard renderers.
        return {
            "bar": "barlist", "dual_bar": "barlist", "hbar": "barlist",
            "cohort": "area", "line": "area", "bridge": "waterfall",
        }.get(kind, kind)

    # --------------------------------------------------------------- figures
    def _fig(self, width_in, height_in):
        fig = plt.figure(figsize=(width_in, height_in), dpi=self.dpi)
        fig.patch.set_facecolor(self.theme.bg_panel)
        return fig

    def _save(self, fig, chart_id, kind, title) -> ChartResult:
        path = self.out_dir / f"{self.lens}_{chart_id}.png"
        fig.savefig(path, facecolor=self.theme.bg_panel, dpi=self.dpi)
        plt.close(fig)
        return ChartResult(chart_id=chart_id, title=title, path=path,
                           available=True, placeholder=False, kind=kind,
                           note="Rendered from the latest pipeline run.")

    def _placeholder(self, chart_id, title, kind, w, h, note) -> ChartResult:
        path = self.out_dir / f"{self.lens}_{chart_id}_placeholder.png"
        # The slide card header already carries the title — keep the placeholder
        # image title-less so it does not duplicate.
        render_placeholder_png(path, "", note, theme=self.theme,
                               width_in=w, height_in=h, dpi=self.dpi)
        return ChartResult(chart_id=chart_id, title=title, path=path,
                           available=False, placeholder=True, kind=kind, note=note)

    # ---------------------------------------------------------- dimension prep
    def _dimension_column(self, spec: Dict[str, Any]) -> Optional[str]:
        bucket = spec.get("bucket")
        if bucket and bucket != "categorical":
            col = self.data.bucket_column(bucket)
            if col and col in self.data.df.columns:
                return col
        dim = spec.get("dimension")
        if dim and dim in self.data.df.columns:
            return dim
        if dim:
            sem = self.reg.dimension_semantic_field(dim)
            if sem and sem in self.data.df.columns:
                return sem
        return None

    def _bucket_order(self, spec: Dict[str, Any]) -> Optional[List[str]]:
        """Natural label order for an ordered bucket (LTV, age, …)."""
        bucket = spec.get("bucket")
        if not bucket or bucket == "categorical":
            return spec.get("order")
        bspec = self.reg.bucket_spec(bucket)
        if bspec and isinstance(bspec.get("labels"), list):
            return list(bspec["labels"])
        return spec.get("order")

    def _is_datelike(self, col: str) -> bool:
        s = self.data.df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            return True
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            return False
        # Only date-typed field names are treated as dates (avoids parsing
        # arbitrary string columns and the associated dateutil warning).
        if not any(tok in col.lower() for tok in ("date", "_dt", "period")):
            return False
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce")
        return parsed.notna().mean() > 0.6 and s.notna().any()

    def _breakdown(self, spec: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Aggregate one dimension into an ordered label/value/count table."""
        dim_col = self._dimension_column(spec)
        if not dim_col:
            return None
        df = self.data.df
        chronological = False
        # Date dimensions: bucket into period labels (YYYY-MM) so a barlist shows
        # months, not one bar per raw date, ordered chronologically.
        if spec.get("period") or (dim_col not in (spec.get("bucket"),) and self._is_datelike(dim_col)):
            period = spec.get("period", "M")
            df = df.copy()
            per = pd.to_datetime(df[dim_col], errors="coerce").dt.to_period(period)
            df["_period_label"] = per.astype(str)
            df.loc[per.isna(), "_period_label"] = "Unknown"
            dim_col = "_period_label"
            chronological = True
        from analytics_lib.stratify import stratify
        # An explicit measure_field (e.g. weighted_expected_funded_amount) lets a
        # forecast chart aggregate a registry-declared amount other than balance.
        mfield = spec.get("measure_field")
        bal = mfield if (mfield and mfield in df.columns) else self.data.balance_col
        table = stratify(df, dim_col, bal, loan_id_col=self.data.loan_id_col,
                         sort_by="balance_sum" if bal else "loan_count")
        if table is None or table.empty:
            return None
        table = table.rename(columns={dim_col: "label"})
        if chronological:
            table = table[table["label"] != "Unknown"]
            table = table.sort_values("label")
            top_n = spec.get("top_n")
            if top_n:
                table = table.tail(int(top_n))
            return table.reset_index(drop=True)
        order = self._bucket_order(spec)
        if order:
            table["label"] = table["label"].astype(str)
            table = (table.set_index("label").reindex([str(o) for o in order])
                     .dropna(how="all").reset_index())
        else:
            table = table.sort_values(
                "balance_sum" if bal else "loan_count", ascending=False)
            top_n = spec.get("top_n")
            if top_n:
                table = table.head(int(top_n))
        return table.reset_index(drop=True)

    # ------------------------------------------------------------- BARLIST
    def _render_barlist(self, spec, chart_id, title, w, h) -> ChartResult:
        table = self._breakdown(spec)
        if table is None or table.empty:
            return self._placeholder(chart_id, title, "barlist", w, h,
                                     f"Dimension '{spec.get('dimension')}' unavailable.")
        measure = spec.get("measure", "balance")
        use_balance = (measure == "balance" and "balance_sum" in table
                       and self.data.balance_col is not None)
        values = (table["balance_sum"] if use_balance else table["loan_count"]).astype(float)
        labels = table["label"].astype(str).tolist()
        counts = table["loan_count"].astype(int).tolist() if "loan_count" in table else None
        fmt = compact_currency if use_balance else compact_number

        fig = self._fig(w, h)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_facecolor(self.theme.bg_panel)
        ax.set_xlim(0, 1)
        ax.axis("off")

        n = len(labels)
        vmax = max(float(values.max()), 1.0)
        pad_top, pad_bot = 0.10, 0.05
        band = (1.0 - pad_top - pad_bot) / max(n, 1)
        bar_h = min(band * 0.62, 0.135)
        label_x, track_x0, track_x1 = 0.005, 0.335, 0.85
        track_w = track_x1 - track_x0
        # Truncate labels to the label column so they never overrun the track.
        max_chars = max(10, int((track_x0 - label_x) * w * 72 / (10.5 * 0.56)))
        labels = [(lab if len(lab) <= max_chars else lab[:max_chars - 1].rstrip() + "…")
                  for lab in labels]
        peri = self.theme.peri
        track_c = self.theme.bg_panel_alt

        for i, (lab, val) in enumerate(zip(labels, values)):
            yc = 1.0 - pad_top - (i + 0.5) * band
            y0 = yc - bar_h / 2
            # track
            ax.add_patch(mpatches.FancyBboxPatch(
                (track_x0, y0), track_w, bar_h,
                boxstyle="round,pad=0,rounding_size=0.012",
                linewidth=0, facecolor=track_c, alpha=0.7,
                mutation_aspect=h / w, zorder=1))
            frac = max(float(val) / vmax, 0.012)
            ax.add_patch(mpatches.FancyBboxPatch(
                (track_x0, y0), track_w * frac, bar_h,
                boxstyle="round,pad=0,rounding_size=0.012",
                linewidth=0, facecolor=peri, alpha=0.9,
                mutation_aspect=h / w, zorder=2))
            # label (left)
            ax.text(label_x, yc, lab, va="center", ha="left",
                    color=self.theme.ink_300, fontsize=10.5, zorder=3)
            # value (right, mono)
            vtxt = fmt(val)
            if use_balance and counts:
                vtxt = f"{vtxt}"
            ax.text(0.995, yc, vtxt, va="center", ha="right",
                    color=self.theme.ink_100, fontsize=10.5,
                    fontproperties=_MONO_FP, zorder=3)
        return self._save(fig, chart_id, "barlist", title)

    # ------------------------------------------------------------- AREA / LINE
    def _render_area(self, spec, chart_id, title, w, h) -> ChartResult:
        date_field = spec.get("date_field", "origination_date")
        df = self.data.df
        mfield = spec.get("measure_field")
        bal = mfield if (mfield and mfield in df.columns) else self.data.balance_col
        if date_field not in df.columns or bal is None:
            return self._placeholder(chart_id, title, "area", w, h,
                                     f"Date field '{date_field}' or balance unavailable.")
        try:
            from analytics_lib.cohort import cohort_table
            table, _ = cohort_table(df, date_field, bal,
                                    period=spec.get("period", "M"),
                                    loan_id_col=self.data.loan_id_col)
        except Exception as exc:  # pragma: no cover
            return self._placeholder(chart_id, title, "area", w, h, f"Cohort failed: {exc}")
        cohort_col = f"{date_field}_cohort"
        if table is None or table.empty or cohort_col not in table.columns:
            return self._placeholder(chart_id, title, "area", w, h, "No time-series data.")
        table = table.sort_values(cohort_col)
        x = list(range(len(table)))
        xlabels = table[cohort_col].astype(str).tolist()
        measure = spec.get("measure", "balance")
        use_balance = measure == "balance" and "balance_sum" in table
        y = (table["balance_sum"] if use_balance else table["loan_count"]).astype(float).tolist()
        cumulative = bool(spec.get("cumulative"))
        if cumulative:
            y = list(np.cumsum(y))

        series_color = self.theme.mint if self.lens == "forecast" else self.theme.peri
        fig = self._fig(w, h)
        ax = fig.add_axes([0.085, 0.16, 0.895, 0.80])
        self._axis_style(ax, use_balance)
        ax.plot(x, y, color=series_color, linewidth=2.4, zorder=3,
                solid_capstyle="round")
        ax.fill_between(x, y, color=series_color, alpha=0.16, zorder=2)
        ax.scatter([x[-1]], [y[-1]], s=26, color=series_color, zorder=4,
                   edgecolors=self.theme.bg_panel, linewidths=1.2)
        self._time_xticks(ax, x, xlabels)
        ax.set_ylim(bottom=0)
        return self._save(fig, chart_id, "area", title)

    # ------------------------------------------------------------- WATERFALL
    def _waterfall_items(self, spec) -> Optional[List[Tuple[str, float, str]]]:
        """(label, value, kind∈{total,add,sub}) for the waterfall.

        Uses explicit ``spec['rows']`` (a period BRIDGE: opening total → signed
        deltas → closing total) when present; otherwise a within-run BUILD-UP —
        the funded balance decomposed by a dimension into additive segments from
        zero to the book total (top contributors + an aggregated 'Other')."""
        rows = spec.get("rows")
        if rows:
            out: List[Tuple[str, float, str]] = []
            for r in rows:
                val = float(r.get("value", 0) or 0)
                kind = "total" if str(r.get("type")) == "total" else ("sub" if val < 0 else "add")
                out.append((str(r.get("label", "")), val, kind))
            return out or None
        table = self._breakdown(spec)
        if (table is None or table.empty or "balance_sum" not in table
                or self.data.balance_col is None):
            return None
        labels = table["label"].astype(str).tolist()
        vals = [float(v) for v in table["balance_sum"].tolist()]
        items = [(lab, v, "add") for lab, v in zip(labels, vals)]
        total = float(pd.to_numeric(self.data.df[self.data.balance_col],
                                    errors="coerce").sum())
        residual = total - sum(vals)
        if residual > max(total * 0.005, 1.0):
            items.append(("Other", residual, "add"))
        items.append(("Total", total, "total"))
        return items

    def _render_waterfall(self, spec, chart_id, title, w, h) -> ChartResult:
        items = self._waterfall_items(spec)
        if not items:
            return self._placeholder(chart_id, title, "waterfall", w, h,
                                     f"Dimension '{spec.get('dimension')}' unavailable for a bridge.")
        # Colours mirror the React waterfall (base / inflow / fallout / total).
        col = {"add": self.theme.peri, "sub": self.theme.negative,
               "total": self.theme.mint, "base": "#3d4a82"}
        n = len(items)
        bar_w = 0.62
        levels: List[float] = []
        rects: List[Tuple[float, float]] = []  # (y0, top) per bar
        lvl = 0.0
        for _lab, val, kind in items:
            if kind == "total":
                y0, top, lvl = 0.0, val, val
            elif val >= 0:
                y0, top = lvl, lvl + val
                lvl += val
            else:
                y0, top = lvl + val, lvl
                lvl += val
            rects.append((y0, top))
            levels.append(lvl)

        ymax = max(top for _, top in rects)
        fig = self._fig(w, h)
        ax = fig.add_axes([0.085, 0.20, 0.895, 0.76])
        self._axis_style(ax, currency_y=True)
        for i, (lab, val, kind) in enumerate(items):
            y0, top = rects[i]
            ax.bar(i, top - y0, bottom=y0, width=bar_w, color=col[kind],
                   linewidth=0, zorder=3)
            vtxt = ("" if kind == "total" else ("+" if val >= 0 else "−")) + compact_currency(abs(val))
            ax.text(i, top + ymax * 0.015, vtxt, ha="center", va="bottom",
                    fontsize=8.5, color=self.theme.ink_300,
                    fontproperties=_MONO_FP, zorder=4)
        # Step connectors between consecutive bars at the running level.
        for i in range(n - 1):
            ax.plot([i + bar_w / 2, i + 1 - bar_w / 2], [levels[i], levels[i]],
                    color=self.theme.line, linewidth=1.0, zorder=2)
        ax.set_xticks(list(range(n)))
        ax.set_xticklabels([lab for lab, _, _ in items], rotation=20, ha="right",
                           fontsize=8.5, color=self.theme.ink_400)
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(bottom=0, top=max(top for _, top in rects) * 1.12)
        return self._save(fig, chart_id, "waterfall", title)

    # ------------------------------------------------------------- HEATMAP
    def _render_heatmap(self, spec, chart_id, title, w, h) -> ChartResult:
        d1 = self._dimension_column({"dimension": spec.get("dimension"),
                                     "bucket": spec.get("bucket")})
        d2 = self._dimension_column({"dimension": spec.get("dimension2"),
                                     "bucket": spec.get("bucket2")})
        bal = self.data.balance_col
        if not d1 or not d2 or not bal:
            return self._placeholder(chart_id, title, "heatmap", w, h,
                                     "Two dimensions + balance required.")
        df = self.data.df
        pivot = pd.pivot_table(df, index=d1, columns=d2, values=bal,
                               aggfunc="sum", fill_value=0.0)
        o1 = self._bucket_order({"bucket": spec.get("bucket"),
                                 "dimension": spec.get("dimension")})
        o2 = self._bucket_order({"bucket": spec.get("bucket2"),
                                 "dimension": spec.get("dimension2")})
        if o1:
            pivot = pivot.reindex([x for x in o1 if x in pivot.index])
        if o2:
            pivot = pivot.reindex(columns=[x for x in o2 if x in pivot.columns])
        if pivot.empty:
            return self._placeholder(chart_id, title, "heatmap", w, h, "No data.")

        ramp = LinearSegmentedColormap.from_list(
            "trakt_ramp", [(0.0, "#1b2240"), (0.55, "#919dd1"), (1.0, "#36c2a8")])
        vals = pivot.values.astype(float)
        vmax = max(vals.max(), 1.0)

        fig = self._fig(w, h)
        ax = fig.add_axes([0.14, 0.20, 0.83, 0.74])
        ax.set_facecolor(self.theme.bg_panel)
        nrows, ncols = vals.shape
        gap = 0.06
        for i in range(nrows):
            for j in range(ncols):
                t = vals[i, j] / vmax
                ax.add_patch(mpatches.Rectangle(
                    (j + gap / 2, (nrows - 1 - i) + gap / 2), 1 - gap, 1 - gap,
                    facecolor=ramp(t), edgecolor="none"))
                if vals[i, j] > 0:
                    txt_c = "#0c1024" if t > 0.45 else self.theme.ink_100
                    ax.text(j + 0.5, (nrows - 1 - i) + 0.5, compact_currency(vals[i, j]),
                            ha="center", va="center", color=txt_c, fontsize=8.5,
                            fontproperties=_MONO_FP)
        ax.set_xlim(0, ncols)
        ax.set_ylim(0, nrows)
        ax.set_xticks([j + 0.5 for j in range(ncols)])
        ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=9,
                           color=self.theme.ink_300, rotation=0)
        ax.set_yticks([(nrows - 1 - i) + 0.5 for i in range(nrows)])
        ax.set_yticklabels([str(r) for r in pivot.index], fontsize=9,
                           color=self.theme.ink_300)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel(self.reg.label_for(spec.get("dimension2", d2)),
                      color=self.theme.ink_400, fontsize=9.5, labelpad=6)
        ax.set_ylabel(self.reg.label_for(spec.get("dimension", d1)),
                      color=self.theme.ink_400, fontsize=9.5, labelpad=6)
        return self._save(fig, chart_id, "heatmap", title)

    # ------------------------------------------------------------- BUBBLE
    def _render_bubble(self, spec, chart_id, title, w, h) -> ChartResult:
        x_field, y_field = spec.get("x"), spec.get("y")
        df = self.data.df
        bal = self.data.balance_col
        if not x_field or not y_field or x_field not in df.columns or y_field not in df.columns:
            return self._placeholder(chart_id, title, "bubble", w, h,
                                     "Bubble requires x and y fields present.")
        x = pd.to_numeric(df[x_field], errors="coerce")
        y = pd.to_numeric(df[y_field], errors="coerce")
        size = pd.to_numeric(df[bal], errors="coerce") if bal else None
        mask = x.notna() & y.notna()
        if mask.sum() == 0:
            return self._placeholder(chart_id, title, "bubble", w, h, "No plottable points.")
        s = None
        if size is not None:
            s = (size[mask] / max(size[mask].max(), 1)) * 520 + 24
        fig = self._fig(w, h)
        ax = fig.add_axes([0.085, 0.16, 0.895, 0.80])
        self._axis_style(ax, False)
        pct_x = self.reg.format_for(x_field) == "percent"
        xv = x[mask] * 100 if pct_x else x[mask]
        ax.scatter(xv, y[mask], s=s, alpha=0.55, color=self.theme.peri,
                   edgecolors=self.theme.navy, linewidths=0.6, zorder=3)
        ax.set_xlabel(self.reg.label_for(x_field), color=self.theme.ink_400, fontsize=9.5)
        ax.set_ylabel(self.reg.label_for(y_field), color=self.theme.ink_400, fontsize=9.5)
        if pct_x:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}%"))
        return self._save(fig, chart_id, "bubble", title)

    # ------------------------------------------------------------- helpers
    def _axis_style(self, ax, currency_y: bool) -> None:
        ax.set_facecolor(self.theme.bg_panel)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(self.theme.line_soft)
        ax.tick_params(colors=self.theme.ink_500, labelsize=9, length=0)
        ax.grid(axis="y", color=self.theme.line_soft, linewidth=0.7,
                linestyle=(0, (3, 3)), alpha=0.9)
        ax.set_axisbelow(True)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(self.theme.ink_500)
        if currency_y:
            ax.yaxis.set_major_formatter(FuncFormatter(_currency_axis))

    def _time_xticks(self, ax, x, xlabels) -> None:
        n = len(x)
        if n == 0:
            return
        # Evenly spaced ticks that always include the first and last, with no
        # collision at the right edge.
        want = min(7, n)
        idx = sorted(set(int(round(k)) for k in np.linspace(0, n - 1, want)))
        ax.set_xticks([x[i] for i in idx])
        ax.set_xticklabels([xlabels[i] for i in idx], fontsize=8.5,
                           color=self.theme.ink_500, rotation=0)
