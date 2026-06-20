/**
 * Heavy Plotly module — imported only via React.lazy() so plotly.js lands in a
 * separate async chunk and never inflates the initial bundle.
 *
 * Plotly is a *fallback* renderer (see ArtifactRenderer): standard MI charts
 * render natively via Recharts / custom components. When Plotly is used, the
 * figure is pre-themed by `applyTraktTheme` so it matches the dark dashboard.
 *
 * We use `plotly.js-dist-min` + `react-plotly.js/factory` (rather than the
 * default `react-plotly.js` entry, which pulls the full, larger `plotly.js`
 * build) to keep the lazy chunk as small as Plotly allows.
 */
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const Plot = createPlotlyComponent(Plotly);

export interface ThemedPlotlyFigure {
  data: unknown[];
  layout?: Record<string, unknown>;
  config?: Record<string, unknown>;
}

export default function PlotlyLazy({ figure }: { figure: ThemedPlotlyFigure }) {
  return (
    <Plot
      data={figure.data}
      layout={figure.layout ?? {}}
      config={figure.config ?? { displayModeBar: false, responsive: true }}
      useResizeHandler
      style={{ width: "100%", height: "320px" }}
    />
  );
}
