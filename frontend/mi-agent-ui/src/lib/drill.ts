/**
 * Drill-through model — derives a categorical dimension and its measures from a
 * chart or table artifact so the UI can let a user select one dimension value
 * (e.g. "South East", "Broker Channel A", "2024") and inspect its detail.
 *
 * This deliberately REUSES the artifact's own result rows rather than running a
 * parallel analytics engine: every metric shown is already present in the
 * payload. Missing measures degrade to N/A in the renderer.
 */

import type { ChartArtifact, TableArtifact, ValueFormat } from "@/domain";
import { formatUiTitle, type PercentScale } from "@/lib/utils";

export type DrillArtifact = ChartArtifact | TableArtifact;

export interface DrillMeasure {
  key: string;
  label: string;
  format?: ValueFormat;
  scale?: PercentScale;
  /** Sum-able (gbp/number) so a "share of total" is meaningful. */
  additive: boolean;
}

export interface DrillModel {
  dimensionKey: string;
  dimensionLabel: string;
  /** Distinct categorical values, in first-seen row order. */
  values: string[];
  /** value → matching rows (summary tables have one; loan-level may have many). */
  rowsByValue: Map<string, Array<Record<string, string | number>>>;
  measures: DrillMeasure[];
  /** First additive measure — used for the derived share-of-total metric. */
  primary?: DrillMeasure;
  /** Sum of each additive measure across ALL rows (for share + context). */
  totals: Record<string, number>;
}

const ADDITIVE_FORMATS = new Set<ValueFormat>(["gbp", "number"]);

/** Beyond this many distinct numeric values we treat the axis as continuous. */
const MAX_NUMERIC_CATEGORIES = 24;

function isChart(a: DrillArtifact): a is ChartArtifact {
  return a.type === "chart";
}

/** Build the drill model, or null when the artifact has no drillable dimension. */
export function buildDrillModel(artifact: DrillArtifact): DrillModel | null {
  const rows = artifact.rows ?? [];
  if (rows.length === 0) return null;

  let dimensionKey: string | undefined;
  let dimensionLabel = "";
  const measures: DrillMeasure[] = [];

  if (isChart(artifact)) {
    // Scatter/bubble are loan-level, continuous — not a grouped dimension.
    if (artifact.chartType === "scatter" || artifact.chartType === "bubble") return null;
    dimensionKey = artifact.xKey;
    if (!dimensionKey) return null;
    dimensionLabel = formatUiTitle(artifact.xLabel ?? dimensionKey);

    const hints = artifact.displayHints ?? {};
    const labelFor = (key: string): string => {
      const s = artifact.series.find((ser) => ser.key === key);
      if (s) return s.label;
      if (key === artifact.yKey) return artifact.yLabel ?? formatUiTitle(key);
      if (key === artifact.sizeKey) return artifact.sizeLabel ?? formatUiTitle(key);
      return formatUiTitle(key);
    };

    const keys = [
      ...artifact.series.map((s) => s.key),
      artifact.valueKey,
      artifact.yKey,
      artifact.sizeKey,
    ].filter((k): k is string => !!k && k !== dimensionKey);

    for (const key of Array.from(new Set(keys))) {
      const hasNumeric = rows.some((r) => Number.isFinite(Number(r[key])));
      if (!hasNumeric) continue;
      const format = hints[key]?.format ?? artifact.valueFormat;
      measures.push({
        key,
        label: labelFor(key),
        format,
        scale: hints[key]?.scale,
        additive: !!format && ADDITIVE_FORMATS.has(format),
      });
    }
  } else {
    const dimCol = artifact.columns.find((c) => c.format === "text") ?? artifact.columns[0];
    if (!dimCol) return null;
    dimensionKey = dimCol.key;
    dimensionLabel = formatUiTitle(dimCol.label);
    for (const col of artifact.columns) {
      if (col.key === dimensionKey) continue;
      measures.push({
        key: col.key,
        label: col.label,
        format: col.format,
        scale: col.scale,
        additive: !!col.format && ADDITIVE_FORMATS.has(col.format),
      });
    }
  }

  if (!dimensionKey) return null;

  const rowsByValue = new Map<string, Array<Record<string, string | number>>>();
  const values: string[] = [];
  let numericValues = 0;
  for (const r of rows) {
    const raw = r[dimensionKey];
    if (raw === undefined || raw === null || raw === "") continue;
    if (typeof raw === "number") numericValues += 1;
    const v = String(raw);
    if (!rowsByValue.has(v)) {
      rowsByValue.set(v, []);
      values.push(v);
    }
    rowsByValue.get(v)!.push(r);
  }

  if (values.length < 2) return null;
  // A numeric dimension with too many distinct values is a continuous axis.
  if (numericValues === rows.length && values.length > MAX_NUMERIC_CATEGORIES) return null;

  const totals: Record<string, number> = {};
  for (const m of measures) {
    if (!m.additive) continue;
    totals[m.key] = rows.reduce((acc, r) => {
      const n = Number(r[m.key]);
      return acc + (Number.isFinite(n) ? n : 0);
    }, 0);
  }

  return {
    dimensionKey,
    dimensionLabel,
    values,
    rowsByValue,
    measures,
    primary: measures.find((m) => m.additive),
    totals,
  };
}

/**
 * Aggregate the rows for one selected dimension value into a single metric row.
 * Additive measures are summed; non-additive measures (averages, ratios) are
 * only reported when the selection is a single row — otherwise null (N/A).
 */
export function aggregateSelection(
  model: DrillModel,
  value: string,
): { records: number; values: Record<string, number | null> } | null {
  const matched = model.rowsByValue.get(value);
  if (!matched || matched.length === 0) return null;
  const out: Record<string, number | null> = {};
  for (const m of model.measures) {
    if (m.additive) {
      out[m.key] = matched.reduce((acc, r) => {
        const n = Number(r[m.key]);
        return acc + (Number.isFinite(n) ? n : 0);
      }, 0);
    } else if (matched.length === 1) {
      const n = Number(matched[0][m.key]);
      out[m.key] = Number.isFinite(n) ? n : null;
    } else {
      out[m.key] = null; // can't aggregate an average across rows
    }
  }
  return { records: matched.length, values: out };
}
