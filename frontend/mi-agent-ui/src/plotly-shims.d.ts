// Minimal type shims for the Plotly packages we lazy-load.
// `plotly.js-dist-min` ships no types; `react-plotly.js/factory` is a subpath
// not covered by @types/react-plotly.js.

declare module "plotly.js-dist-min" {
  const Plotly: unknown;
  export default Plotly;
}

declare module "react-plotly.js/factory" {
  import type * as React from "react";
  export interface PlotParams {
    data: unknown[];
    layout?: Record<string, unknown>;
    config?: Record<string, unknown>;
    style?: React.CSSProperties;
    useResizeHandler?: boolean;
    className?: string;
  }
  export default function createPlotlyComponent(
    plotly: unknown,
  ): React.ComponentType<PlotParams>;
}
