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
  isValidationArtifact,
} from "@/domain";
import { KPIArtifactView } from "./KPIArtifactView";
import { ChartArtifactView } from "./ChartArtifactView";
import { TableArtifactView } from "./TableArtifactView";
import { ValidationArtifactView } from "./ValidationArtifactView";
import { RiskArtifactView } from "./RiskArtifactView";
import { ScenarioArtifactView } from "./ScenarioArtifactView";

export function ArtifactRenderer({ artifact }: { artifact: Artifact }) {
  if (isKPIArtifact(artifact)) return <KPIArtifactView artifact={artifact} />;
  if (isChartArtifact(artifact)) return <ChartArtifactView artifact={artifact} />;
  if (isTableArtifact(artifact)) return <TableArtifactView artifact={artifact} />;
  if (isValidationArtifact(artifact)) return <ValidationArtifactView artifact={artifact} />;
  if (isRiskArtifact(artifact)) return <RiskArtifactView artifact={artifact} />;
  if (isScenarioArtifact(artifact)) return <ScenarioArtifactView artifact={artifact} />;
  return (
    <div className="rounded-lg border border-[var(--color-line-soft)] bg-navy-850/40 p-4 text-sm text-ink-400">
      Unsupported artifact type.
    </div>
  );
}
