/**
 * Suggested next questions — deterministic, grounded follow-up chips shown after
 * a SUCCESSFUL chart/table result. Built from the executed spec + the catalogue
 * (never the LLM, never an unavailable field), so every suggestion is real and
 * routes back through the normal MI Agent flow (with context).
 */

import type { Artifact, MIQuerySpec, SuggestedAction } from "@/domain";
import { isChartArtifact, isTableArtifact } from "@/domain";
import { MEASURES } from "@/data/catalog";
import { buildDrillModel, type DrillArtifact } from "@/lib/drill";
import { cleanLabel, dimensionLabel, measureLabel } from "@/lib/analysisContext";

const MEASURE_BY_KEY = new Map(MEASURES.map((m) => [m.key, m]));

/** Common, broadly-available dimensions to offer as "Split by …". */
const SUGGESTED_DIMENSIONS = [
  "broker_channel",
  "geographic_region_obligor",
  "erm_product_type",
  "pipeline_stage",
  "vintage_year",
];

function aggregationWord(measureKey?: string): string {
  const fmt = measureKey ? MEASURE_BY_KEY.get(measureKey)?.format : undefined;
  return fmt === "pct" || fmt === "number" ? "average " : "";
}

/** The dimension value with the largest primary measure, for a drill chip. */
function largestValue(artifact: DrillArtifact): string | undefined {
  const model = buildDrillModel(artifact);
  if (!model?.primary) return undefined;
  let best: string | undefined;
  let bestSum = -Infinity;
  for (const v of model.values) {
    const sum = (model.rowsByValue.get(v) ?? []).reduce((acc, r) => {
      const n = Number(r[model.primary!.key]);
      return acc + (Number.isFinite(n) ? n : 0);
    }, 0);
    if (sum > bestSum) {
      bestSum = sum;
      best = v;
    }
  }
  return best;
}

/**
 * Build up to 5 grounded suggestions for an MI result. Returns [] when the spec
 * can't be grounded (no recognised measure/dimension) so nothing hallucinated is
 * ever shown.
 */
export function buildSuggestedActions(
  spec: Partial<MIQuerySpec> | undefined,
  artifact: Artifact,
): SuggestedAction[] {
  if (!spec) return [];
  if (!isChartArtifact(artifact) && !isTableArtifact(artifact)) return [];

  const measureKey = spec.metric;
  const measureLbl = measureLabel(measureKey) ?? "Balance";
  const baseMeasureKey = measureKey ?? "current_outstanding_balance";
  const dimKey = spec.dimensions?.[0] ?? spec.dimension;
  const dimLbl = dimensionLabel(dimKey);
  if (!dimLbl) return []; // ungrounded result — show nothing rather than guess

  // Parser-friendly forms for question strings (chip labels stay readable).
  const measureQ = cleanLabel(measureLbl) ?? measureLbl;
  const dimQ = cleanLabel(dimLbl) ?? dimLbl;

  const out: SuggestedAction[] = [];
  const seen = new Set<string>();
  const push = (a: SuggestedAction) => {
    const key = a.question.toLowerCase();
    if (!seen.has(key) && out.length < 5) {
      seen.add(key);
      out.push(a);
    }
  };

  // change_dimension — offer 2 other common dimensions.
  for (const cand of SUGGESTED_DIMENSIONS) {
    if (out.filter((a) => a.kind === "change_dimension").length >= 2) break;
    if (cand === dimKey) continue;
    const lbl = dimensionLabel(cand);
    if (!lbl) continue;
    push({ label: `Split by ${lbl}`, question: `${measureQ} by ${cleanLabel(lbl) ?? lbl}`, kind: "change_dimension" });
  }

  // change_measure — one alternative measure (LTV when on balance; balance else).
  const altMeasureKey =
    baseMeasureKey === "current_loan_to_value" ? "current_outstanding_balance" : "current_loan_to_value";
  const altLbl = measureLabel(altMeasureKey);
  if (altLbl) {
    push({
      label: `Show ${altLbl}`,
      question: `${aggregationWord(altMeasureKey)}${cleanLabel(altLbl) ?? altLbl} by ${dimQ}`,
      kind: "change_measure",
    });
  }

  // drill — into the largest current dimension value (routes via the follow-up
  // resolver as a value filter on the active dimension).
  const top = largestValue(artifact);
  if (top) push({ label: `Drill into ${top}`, question: `only ${top}`, kind: "drill" });

  // refine — offer the other output mode.
  if (isChartArtifact(artifact)) {
    push({ label: "Show as Table", question: `${measureQ} by ${dimQ} as a table`, kind: "refine" });
  }

  return out.slice(0, 5);
}
