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
import { PlotlyArtifactView, hasPlotlyFigure } from "./PlotlyArtifactView";
import { TableArtifactView } from "./TableArtifactView";
import { ValidationArtifactView } from "./ValidationArtifactView";
import { RiskArtifactView } from "./RiskArtifactView";
import { ScenarioArtifactView } from "./ScenarioArtifactView";
import { UnsupportedArtifactView } from "./UnsupportedArtifactView";

/** Chart types the Recharts path can reconstruct from normalized table data. */
const RECHARTS_TYPES = new Set<ChartType>(["bar", "line", "area", "scatter", "bubble", "waterfall"]);

export function ArtifactRenderer({ artifact }: { artifact: Artifact }) {
  if (isKPIArtifact(artifact)) return <KPIArtifactView artifact={artifact} />;
  if (isChartArtifact(artifact)) {
    // Routing: (1) backend Plotly figure → faithful Plotly renderer;
    // (2) normalized data + Recharts-supported type → Recharts; (3) otherwise
    // an explicit unsupported state (e.g. heatmap/treemap with no figure).
    if (hasPlotlyFigure(artifact.source.figure)) {
      return <PlotlyArtifactView artifact={artifact} />;
    }
    if (RECHARTS_TYPES.has(artifact.chartType)) {
      return <ChartArtifactView artifact={artifact} />;
    }
    return (
      <UnsupportedArtifactView
        artifact={{
          ...artifact,
          type: "unsupported",
          reason: `"${artifact.chartType}" charts require a Plotly figure, which was not provided.`,
        }}
      />
    );
  }
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
