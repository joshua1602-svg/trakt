/**
 * Type guards for the artifact union. Used by the renderer and tests to narrow
 * artifacts safely without casts.
 */

import type {
  Artifact,
  ChartArtifact,
  KPIArtifact,
  RiskArtifact,
  ScenarioArtifact,
  TableArtifact,
  ValidationArtifact,
} from "./artifacts";

export const isKPIArtifact = (a: Artifact): a is KPIArtifact => a.type === "kpi";
export const isChartArtifact = (a: Artifact): a is ChartArtifact => a.type === "chart";
export const isTableArtifact = (a: Artifact): a is TableArtifact => a.type === "table";
export const isValidationArtifact = (a: Artifact): a is ValidationArtifact =>
  a.type === "validation";
export const isRiskArtifact = (a: Artifact): a is RiskArtifact => a.type === "risk";
export const isScenarioArtifact = (a: Artifact): a is ScenarioArtifact => a.type === "scenario";

/** Runtime validation that an object is a well-formed artifact envelope. */
export function isArtifact(value: unknown): value is Artifact {
  if (typeof value !== "object" || value === null) return false;
  const a = value as Record<string, unknown>;
  return (
    typeof a.id === "string" &&
    typeof a.type === "string" &&
    typeof a.title === "string" &&
    typeof a.createdAt === "string" &&
    typeof a.source === "object" &&
    a.source !== null
  );
}
