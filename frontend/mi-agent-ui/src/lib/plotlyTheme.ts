/**
 * Trakt theme adapter for Plotly figures.
 *
 * Plotly is only used as a last-resort fallback (see ArtifactRenderer). When it
 * is, the backend figure carries the chart factory's light theme; this adapter
 * re-skins it to match the dark Trakt dashboard: transparent background, Inter
 * typography, navy/periwinkle palette, soft gridlines, and dark hover styling.
 */
import { THEME } from "./theme";

const FONT = 'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const GRID = "#1c2440"; // line-soft
const LINE = "#232b48"; // line
const INK_100 = "#eef1f8";
const INK_300 = "#b8c0d6";
const INK_400 = "#8c95b0";
const NAVY_950 = "#0c1024";

/** Dark sequential colourscale for heatmap-style traces (navy → periwinkle). */
export const TRAKT_SEQUENTIAL: Array<[number, string]> = [
  [0, "#11162e"],
  [0.5, "#3d4a82"],
  [1, THEME.peri],
];

type AnyObj = Record<string, unknown>;

function themeAxis(ax: AnyObj = {}): AnyObj {
  const title = (ax.title as AnyObj) ?? {};
  const titleFont = (title.font as AnyObj) ?? {};
  const tickFont = (ax.tickfont as AnyObj) ?? {};
  return {
    ...ax,
    gridcolor: GRID,
    zerolinecolor: LINE,
    linecolor: LINE,
    tickfont: { ...tickFont, color: INK_400, size: 11, family: FONT },
    title: { ...title, font: { ...titleFont, color: INK_400, size: 12, family: FONT } },
  };
}

export interface ThemedFigure {
  data: unknown[];
  layout: AnyObj;
  config: AnyObj;
}

export function applyTraktTheme(figure: { data?: unknown[]; layout?: AnyObj }): ThemedFigure {
  const layout = (figure.layout as AnyObj) ?? {};

  // Re-skin heatmap-like traces with the Trakt sequential scale.
  const data = (figure.data ?? []).map((t) => {
    const trace = t as AnyObj;
    if (trace.type === "heatmap" || trace.type === "histogram2d") {
      return { ...trace, colorscale: TRAKT_SEQUENTIAL, colorbar: { ...(trace.colorbar as AnyObj), tickfont: { color: INK_400 }, outlinewidth: 0 } };
    }
    return trace;
  });

  return {
    data,
    layout: {
      ...layout,
      title: undefined, // the ArtifactCard header already shows the title
      autosize: true,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: FONT, size: 12, color: INK_300 },
      colorway: THEME.categorical,
      margin: { l: 56, r: 16, t: 12, b: 44, ...((layout.margin as AnyObj) ?? {}) },
      xaxis: themeAxis(layout.xaxis as AnyObj),
      yaxis: themeAxis(layout.yaxis as AnyObj),
      legend: {
        ...((layout.legend as AnyObj) ?? {}),
        font: { color: INK_300, size: 11, family: FONT },
        bgcolor: "rgba(0,0,0,0)",
      },
      hoverlabel: {
        bgcolor: NAVY_950,
        bordercolor: LINE,
        font: { color: INK_100, family: FONT, size: 12 },
      },
      coloraxis: { ...((layout.coloraxis as AnyObj) ?? {}), colorscale: TRAKT_SEQUENTIAL },
    },
    config: { displayModeBar: false, responsive: true },
  };
}
