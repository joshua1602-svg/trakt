"""mi_agent_pptx.chart_resolver — registry-authorised static chart rendering.

Renders the deck's charts as high-resolution PNGs styled to the MI Agent React
dark theme, so a chart drops onto its slide panel with no white pasted box. All
aggregation is delegated to the registry-authorised shared analytics library
(:mod:`analytics_lib` — ``stratify`` / ``concentration`` / ``cohort``); this
module only shapes the already-authorised numbers into a figure. When a chart's
source dimension/measure is unavailable it returns a *branded placeholder* result
(via :mod:`placeholders`) plus a coverage note, never a crash.

Chart kinds supported (v1): ``bar``, ``dual_bar`` (balance + count),
``hbar``, ``heatmap`` (2-dim balance matrix), ``bubble``, ``line`` and
``cohort`` (vintage/origination). Each renders onto the theme panel background.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data_resolver import ResolvedData  # noqa: E402
from .metric_resolver import compact_currency, compact_number  # noqa: E402
from .placeholders import render_placeholder_png  # noqa: E402
from .pptx_theme import PptxTheme, THEME  # noqa: E402
from .registry_loader import RegistryLoader  # noqa: E402


@dataclass
class ChartResult:
    """Outcome of resolving/rendering a single chart spec."""

    chart_id: str
    title: str
    path: Optional[Path] = None
    available: bool = False
    placeholder: bool = False
    note: str = ""
    kind: str = "bar"

    @property
    def ok(self) -> bool:
        return self.available and self.path is not None


def _apply_axes_style(ax, theme: PptxTheme) -> None:
    ax.set_facecolor(theme.bg_panel)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(theme.line)
    ax.tick_params(colors=theme.ink_300, labelsize=9)
    ax.grid(axis="y", color=theme.line_soft, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(theme.ink_300)


def _new_fig(theme: PptxTheme, width_in=9.6, height_in=4.6, dpi=200, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(width_in, height_in), dpi=dpi)
    fig.patch.set_facecolor(theme.bg_panel)
    if ncols == 1:
        axes = [axes]
    for ax in axes:
        _apply_axes_style(ax, theme)
    return fig, axes


def _currency_fmt(_x, _pos):
    return compact_currency(_x)


class ChartResolver:
    """Resolve deck chart specs into rendered PNGs or branded placeholders."""

    def __init__(
        self,
        data: ResolvedData,
        registries: RegistryLoader,
        out_dir: str | Path,
        theme: PptxTheme = THEME,
        dpi: int = 200,
    ):
        self.data = data
        self.reg = registries
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.theme = theme
        self.dpi = dpi

    # ------------------------------------------------------------------ public
    def resolve(self, spec: Dict[str, Any]) -> ChartResult:
        chart_id = spec.get("id", "chart")
        title = spec.get("title", chart_id.replace("_", " ").title())
        kind = spec.get("type", "bar")
        try:
            handler = getattr(self, f"_render_{kind}", None)
            if handler is None:
                return self._placeholder(chart_id, title, kind,
                                         f"Unsupported chart type '{kind}'.")
            return handler(spec, chart_id, title)
        except Exception as exc:  # pragma: no cover - defensive
            return self._placeholder(chart_id, title, kind,
                                     f"Chart render error: {exc}")

    # -------------------------------------------------------------- dimension
    def _dimension_column(self, spec: Dict[str, Any]) -> Optional[str]:
        """Resolve the physical column for a chart's dimension.

        A ``bucket`` key resolves to the materialised registry bucket column;
        otherwise the categorical dimension column is used directly.
        """
        bucket = spec.get("bucket")
        if bucket and bucket != "categorical":
            col = self.data.bucket_column(bucket)
            if col and col in self.data.df.columns:
                return col
        dim = spec.get("dimension")
        if dim and dim in self.data.df.columns:
            return dim
        # Fall back to the dimension's registry semantic field.
        if dim:
            sem = self.reg.dimension_semantic_field(dim)
            if sem and sem in self.data.df.columns:
                return sem
        return None

    def _stratify(self, dim_col: str, spec: Dict[str, Any]) -> Optional[pd.DataFrame]:
        from analytics_lib.stratify import stratify
        bal = self.data.balance_col
        table = stratify(
            self.data.df, dim_col, bal,
            loan_id_col=self.data.loan_id_col,
            sort_by="balance_sum" if bal else "loan_count",
        )
        if table is None or table.empty:
            return None
        top_n = spec.get("top_n")
        if top_n:
            table = table.head(int(top_n))
        return table.reset_index(drop=True)

    # ----------------------------------------------------------------- renders
    def _render_bar(self, spec, chart_id, title) -> ChartResult:
        dim_col = self._dimension_column(spec)
        if not dim_col:
            return self._placeholder(chart_id, title, "bar",
                                     f"Dimension '{spec.get('dimension')}' unavailable.")
        table = self._stratify(dim_col, spec)
        if table is None:
            return self._placeholder(chart_id, title, "bar", "No data to aggregate.")
        measure = spec.get("measure", "balance")
        y = table["balance_sum"] if (measure == "balance" and "balance_sum" in table) \
            else table["loan_count"]
        fig, (ax,) = _new_fig(self.theme, dpi=self.dpi)
        ax.bar(table[dim_col].astype(str), y, color=self.theme.navy,
               edgecolor=self.theme.peri, linewidth=0.4)
        if measure == "balance":
            ax.yaxis.set_major_formatter(FuncFormatter(_currency_fmt))
        ax.set_title(title, color=self.theme.ink_100, fontsize=13,
                     fontweight="bold", loc="left", pad=10)
        self._rotate_if_needed(ax, table[dim_col])
        return self._save(fig, chart_id, title, "bar")

    def _render_hbar(self, spec, chart_id, title) -> ChartResult:
        dim_col = self._dimension_column(spec)
        if not dim_col:
            return self._placeholder(chart_id, title, "hbar",
                                     f"Dimension '{spec.get('dimension')}' unavailable.")
        table = self._stratify(dim_col, spec)
        if table is None:
            return self._placeholder(chart_id, title, "hbar", "No data to aggregate.")
        table = table.iloc[::-1]
        measure = spec.get("measure", "balance")
        x = table["balance_sum"] if (measure == "balance" and "balance_sum" in table) \
            else table["loan_count"]
        fig, (ax,) = _new_fig(self.theme, dpi=self.dpi)
        ax.barh(table[dim_col].astype(str), x, color=self.theme.navy,
                edgecolor=self.theme.peri, linewidth=0.4)
        if measure == "balance":
            ax.xaxis.set_major_formatter(FuncFormatter(_currency_fmt))
        ax.set_title(title, color=self.theme.ink_100, fontsize=13,
                     fontweight="bold", loc="left", pad=10)
        return self._save(fig, chart_id, title, "hbar")

    def _render_dual_bar(self, spec, chart_id, title) -> ChartResult:
        dim_col = self._dimension_column(spec)
        if not dim_col:
            return self._placeholder(chart_id, title, "dual_bar",
                                     f"Dimension '{spec.get('dimension')}' unavailable.")
        table = self._stratify(dim_col, spec)
        if table is None:
            return self._placeholder(chart_id, title, "dual_bar", "No data to aggregate.")
        fig, axes = _new_fig(self.theme, width_in=9.6, height_in=4.4,
                             dpi=self.dpi, ncols=2)
        cats = table[dim_col].astype(str)
        if "balance_sum" in table:
            axes[0].bar(cats, table["balance_sum"], color=self.theme.navy,
                        edgecolor=self.theme.peri, linewidth=0.4)
            axes[0].yaxis.set_major_formatter(FuncFormatter(_currency_fmt))
        axes[0].set_title("Balance", color=self.theme.ink_300, fontsize=11,
                          loc="left")
        axes[1].bar(cats, table["loan_count"], color=self.theme.peri,
                    edgecolor=self.theme.navy, linewidth=0.4)
        axes[1].set_title("Loan count", color=self.theme.ink_300, fontsize=11,
                          loc="left")
        for ax in axes:
            self._rotate_if_needed(ax, table[dim_col])
        fig.suptitle(title, color=self.theme.ink_100, fontsize=13,
                     fontweight="bold", x=0.06, ha="left")
        return self._save(fig, chart_id, title, "dual_bar")

    def _render_heatmap(self, spec, chart_id, title) -> ChartResult:
        d1 = self._dimension_column({"dimension": spec.get("dimension"),
                                     "bucket": spec.get("bucket")})
        d2 = self._dimension_column({"dimension": spec.get("dimension2"),
                                     "bucket": spec.get("bucket2")})
        bal = self.data.balance_col
        if not d1 or not d2 or not bal:
            return self._placeholder(chart_id, title, "heatmap",
                                     "Two dimensions + balance required for heatmap.")
        df = self.data.df
        pivot = pd.pivot_table(
            df, index=d1, columns=d2,
            values=bal, aggfunc="sum", fill_value=0.0,
        )
        if pivot.empty:
            return self._placeholder(chart_id, title, "heatmap", "No data for heatmap.")
        fig, (ax,) = _new_fig(self.theme, width_in=9.6, height_in=5.0, dpi=self.dpi)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "trakt_seq", self.theme.sequential)
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        ax.set_xlabel(self.reg.label_for(spec.get("dimension2", d2)),
                      color=self.theme.ink_300, fontsize=10)
        ax.set_ylabel(self.reg.label_for(spec.get("dimension", d1)),
                      color=self.theme.ink_300, fontsize=10)
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=self.theme.ink_300, labelsize=8)
        cbar.outline.set_edgecolor(self.theme.line)
        ax.set_title(title, color=self.theme.ink_100, fontsize=13,
                     fontweight="bold", loc="left", pad=10)
        return self._save(fig, chart_id, title, "heatmap")

    def _render_bubble(self, spec, chart_id, title) -> ChartResult:
        x_field = spec.get("x")
        y_field = spec.get("y")
        df = self.data.df
        bal = self.data.balance_col
        if not x_field or not y_field or x_field not in df.columns \
                or y_field not in df.columns:
            return self._placeholder(chart_id, title, "bubble",
                                     "Bubble requires x and y fields present.")
        x = pd.to_numeric(df[x_field], errors="coerce")
        y = pd.to_numeric(df[y_field], errors="coerce")
        size = pd.to_numeric(df[bal], errors="coerce") if bal else None
        mask = x.notna() & y.notna()
        if mask.sum() == 0:
            return self._placeholder(chart_id, title, "bubble", "No plottable points.")
        s = None
        if size is not None:
            s = (size[mask] / max(size[mask].max(), 1)) * 600 + 15
        fig, (ax,) = _new_fig(self.theme, dpi=self.dpi)
        ax.scatter(x[mask], y[mask], s=s, alpha=0.6, color=self.theme.peri,
                   edgecolors=self.theme.navy, linewidths=0.5)
        ax.set_xlabel(self.reg.label_for(x_field), color=self.theme.ink_300)
        ax.set_ylabel(self.reg.label_for(y_field), color=self.theme.ink_300)
        ax.set_title(title, color=self.theme.ink_100, fontsize=13,
                     fontweight="bold", loc="left", pad=10)
        return self._save(fig, chart_id, title, "bubble")

    def _render_line(self, spec, chart_id, title) -> ChartResult:
        return self._render_cohort(spec, chart_id, title, as_line=True)

    def _render_cohort(self, spec, chart_id, title, as_line=True) -> ChartResult:
        date_field = spec.get("date_field", "origination_date")
        df = self.data.df
        bal = self.data.balance_col
        if date_field not in df.columns or not bal:
            return self._placeholder(chart_id, title, "cohort",
                                     f"Date field '{date_field}' or balance unavailable.")
        try:
            from analytics_lib.cohort import cohort_table
            period = spec.get("period", "M")
            table, _issues = cohort_table(
                df, date_field, bal, period=period,
                loan_id_col=self.data.loan_id_col)
        except Exception as exc:  # pragma: no cover
            return self._placeholder(chart_id, title, "cohort", f"Cohort failed: {exc}")
        cohort_col = f"{date_field}_cohort"
        if table is None or table.empty or cohort_col not in table.columns:
            return self._placeholder(chart_id, title, "cohort", "No cohort data.")
        table = table.sort_values(cohort_col)
        x = table[cohort_col].astype(str)
        measure = spec.get("measure", "balance")
        y = table["balance_sum"] if (measure == "balance" and "balance_sum" in table) \
            else table["loan_count"]
        fig, (ax,) = _new_fig(self.theme, dpi=self.dpi)
        if as_line:
            ax.plot(x, y, color=self.theme.peri, marker="o", markersize=4,
                    linewidth=2)
            ax.fill_between(range(len(x)), y, color=self.theme.navy, alpha=0.35)
        else:
            ax.bar(x, y, color=self.theme.navy, edgecolor=self.theme.peri,
                   linewidth=0.4)
        if measure == "balance":
            ax.yaxis.set_major_formatter(FuncFormatter(_currency_fmt))
        self._rotate_if_needed(ax, x)
        ax.set_title(title, color=self.theme.ink_100, fontsize=13,
                     fontweight="bold", loc="left", pad=10)
        return self._save(fig, chart_id, title, "cohort")

    # ------------------------------------------------------------------ helpers
    def _rotate_if_needed(self, ax, labels) -> None:
        n = len(labels)
        longest = max((len(str(x)) for x in labels), default=0)
        if n > 6 or longest > 8:
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(40)
                lbl.set_ha("right")

    def _save(self, fig, chart_id, title, kind) -> ChartResult:
        fig.tight_layout()
        path = self.out_dir / f"{chart_id}.png"
        fig.savefig(path, facecolor=self.theme.bg_panel, dpi=self.dpi,
                    bbox_inches="tight")
        plt.close(fig)
        return ChartResult(chart_id=chart_id, title=title, path=path,
                           available=True, placeholder=False, kind=kind,
                           note="Rendered from latest pipeline run.")

    def _placeholder(self, chart_id, title, kind, note) -> ChartResult:
        path = self.out_dir / f"{chart_id}_placeholder.png"
        render_placeholder_png(path, title, "Chart unavailable — " + note,
                               theme=self.theme)
        return ChartResult(chart_id=chart_id, title=title, path=path,
                           available=False, placeholder=True, kind=kind, note=note)
