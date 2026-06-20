/**
 * Heavy Plotly module — imported only via React.lazy() so plotly.js lands in a
 * separate async chunk and never inflates the initial bundle.
 *
 * We use `plotly.js-dist-min` + `react-plotly.js/factory` (rather than the
 * default `react-plotly.js` entry, which pulls the full, larger `plotly.js`
 * build) to keep the lazy chunk as small as Plotly allows.
 */
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const Plot = createPlotlyComponent(Plotly);

export interface PlotlyFigure {
  data: unknown[];
  layout?: Record<string, unknown>;
  config?: Record<string, unknown>;
}

export default function PlotlyLazy({ figure }: { figure: PlotlyFigure }) {
  // Preserve the backend-provided layout/config, but make the figure responsive
  // and suppress its own title (the ArtifactCard header already shows it).
  const layout = {
    ...(figure.layout ?? {}),
    title: undefined,
    autosize: true,
    height: 320,
    margin: { l: 56, r: 16, t: 12, b: 44, ...(((figure.layout ?? {}) as Record<string, unknown>).margin as object) },
  };

  return (
    <Plot
      data={figure.data}
      layout={layout}
      useResizeHandler
      style={{ width: "100%", height: "320px" }}
      config={{ displayModeBar: false, responsive: true, ...(figure.config ?? {}) }}
    />
  );
}
