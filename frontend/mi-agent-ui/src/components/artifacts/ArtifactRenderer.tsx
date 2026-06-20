/**
 * ArtifactRenderer — dispatches an artifact to its view by type, using the
 * domain type guards for safe narrowing. Adding a new artifact type means
 * adding a guard + a view here and nothing else in the canvas/card layer.
 */

import type { Artifact, ChartType } from "@/domain";
import {
  isChartArtifact,
  isKPIArtifact,
  isRiskArtifact,
  isScenarioArtifact,
  isTableArtifact,
  isUnsupportedArtifact,
  isValidationArtifact,
} from "@/domain";
import { KPIArtifactView } from "./KPIArtifactView";
import { ChartArtifactView } from "./ChartArtifactView";
import { HeatmapArtifactView } from "./HeatmapArtifactView";
import { TreemapArtifactView } from "./TreemapArtifactView";
import { PlotlyArtifactView, hasPlotlyFigure } from "./PlotlyArtifactView";
import { TableArtifactView } from "./TableArtifactView";
import { ValidationArtifactView } from "./ValidationArtifactView";
import { RiskArtifactView } from "./RiskArtifactView";
import { ScenarioArtifactView } from "./ScenarioArtifactView";
import { UnsupportedArtifactView } from "./UnsupportedArtifactView";

/** Chart types the native Recharts path renders with full Trakt styling. */
const RECHARTS_TYPES = new Set<ChartType>(["bar", "line", "area", "scatter", "bubble", "waterfall"]);

/**
 * Chart routing is NATIVE-FIRST. Standard MI charts render through Recharts /
 * custom Trakt-themed components; Plotly is used only as an explicit fallback
 * for a chart type with no acceptable native renderer (and is itself
 * re-skinned to the dark theme by PlotlyArtifactView).
 */
function renderChart(artifact: Extract<Artifact, { type: "chart" }>) {
  const ct = artifact.chartType;
  const figure = artifact.source.figure;

  // 1. Standard charts → native Recharts.
  if (RECHARTS_TYPES.has(ct)) return <ChartArtifactView artifact={artifact} />;

  // 2. heatmap → native custom grid when the matrix data is present.
  if (ct === "heatmap") {
    if (artifact.xKey && artifact.yKey && artifact.valueKey && artifact.rows.length) {
      return <HeatmapArtifactView artifact={artifact} />;
    }
    if (hasPlotlyFigure(figure)) return <PlotlyArtifactView artifact={artifact} />;
  }

  // 3. treemap → native Recharts Treemap when the data is present.
  if (ct === "treemap") {
    if (artifact.xKey && artifact.valueKey && artifact.rows.length) {
      return <TreemapArtifactView artifact={artifact} />;
    }
    if (hasPlotlyFigure(figure)) return <PlotlyArtifactView artifact={artifact} />;
  }

  // 4. Anything else → themed Plotly fallback if a figure exists.
  if (hasPlotlyFigure(figure)) return <PlotlyArtifactView artifact={artifact} />;

  return (
    <UnsupportedArtifactView
      artifact={{
        ...artifact,
        type: "unsupported",
        reason: `No native renderer is available for "${ct}" and no chart figure was provided.`,
      }}
    />
  );
}

export function ArtifactRenderer({ artifact }: { artifact: Artifact }) {
  if (isKPIArtifact(artifact)) return <KPIArtifactView artifact={artifact} />;
  if (isChartArtifact(artifact)) return renderChart(artifact);
  if (isTableArtifact(artifact)) return <TableArtifactView artifact={artifact} />;
  if (isValidationArtifact(artifact)) return <ValidationArtifactView artifact={artifact} />;
  if (isRiskArtifact(artifact)) return <RiskArtifactView artifact={artifact} />;
  if (isScenarioArtifact(artifact)) return <ScenarioArtifactView artifact={artifact} />;
  if (isUnsupportedArtifact(artifact)) return <UnsupportedArtifactView artifact={artifact} />;
  // Defensive fallback for an artifact type the renderer doesn't know yet.
  const unknown = artifact as Artifact;
  return (
    <UnsupportedArtifactView
      artifact={{
        ...unknown,
        type: "unsupported",
        reason: `No renderer is registered for artifact type "${unknown.type}".`,
      }}
    />
  );
}
