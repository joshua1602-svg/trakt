/**
 * ArtifactRenderer — dispatches an artifact to its view by type, using the
 * domain type guards for safe narrowing. Adding a new artifact type means
 * adding a guard + a view here and nothing else in the canvas/card layer.
 */

import type { Artifact } from "@/domain";
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
import { TableArtifactView } from "./TableArtifactView";
import { ValidationArtifactView } from "./ValidationArtifactView";
import { RiskArtifactView } from "./RiskArtifactView";
import { ScenarioArtifactView } from "./ScenarioArtifactView";
import { UnsupportedArtifactView } from "./UnsupportedArtifactView";

export function ArtifactRenderer({ artifact }: { artifact: Artifact }) {
  if (isKPIArtifact(artifact)) return <KPIArtifactView artifact={artifact} />;
  if (isChartArtifact(artifact)) return <ChartArtifactView artifact={artifact} />;
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
