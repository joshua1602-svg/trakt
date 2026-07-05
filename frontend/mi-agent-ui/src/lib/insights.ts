/**
 * Insight Engine — a completely ADDITIVE, pure-function layer that analyses an
 * already-executed MI result (chart/table) and produces deterministic
 * observations + stat-driven suggested investigations.
 *
 * It NEVER requeries, calls an LLM, or touches the parser / MIQuerySpec /
 * execution engine. It operates only on the artifact rows already returned. If
 * anything is unsupported it returns null and the caller hides the panel — the
 * chart/table is unaffected.
 */

import type { Artifact, MIQuerySpec, SuggestedAction } from "@/domain";
import { isChartArtifact, isTableArtifact } from "@/domain";
import { buildDrillModel, type DrillArtifact, type DrillMeasure, type DrillModel } from "@/lib/drill";
import { formatValue, toPercentPoints } from "@/lib/utils";

export type Severity = "info" | "watch" | "significant";

export type ObservationKind =
  | "concentration"
  | "ranking"
  | "outlier"
  | "spread"
  | "movement";

export interface Observation {
  id: string;
  kind: ObservationKind;
  text: string;
  severity: Severity;
}

export interface InsightStatistics {
  measureKey?: string;
  measureLabel?: string;
  count: number;
  total: number;
  max: number;
  min: number;
  mean: number;
  median: number;
  stdev: number;
  spread: number;
  topLabel?: string;
  topValue?: number;
  topShare?: number;
  bottomLabel?: string;
  bottomValue?: number;
  top3Share?: number;
}

export interface InsightSummary {
  headline?: string;
  observations: Observation[];
  suggestions: SuggestedAction[];
  statistics: InsightStatistics;
}

/** Optional prior values for movement observations, keyed by dimension value. */
export interface InsightOptions {
  /** Prior total of the focus measure (for an aggregate movement statement). */
  previousTotal?: number;
  previousLabel?: string;
}

/* --------------------------------- math --------------------------------- */

function sum(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0);
}
function mean(xs: number[]): number {
  return xs.length ? sum(xs) / xs.length : 0;
}
function median(xs: number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
function stdev(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
}

/* ---------------------------- aggregation ---------------------------- */

function valueOf(model: DrillModel, value: string, measure: DrillMeasure): number {
  const rows = model.rowsByValue.get(value) ?? [];
  if (measure.additive) {
    return rows.reduce((acc, r) => {
      const n = Number(r[measure.key]);
      return acc + (Number.isFinite(n) ? n : 0);
    }, 0);
  }
  for (const r of rows) {
    const n = Number(r[measure.key]);
    if (Number.isFinite(n)) return n;
  }
  return NaN;
}

/** Format a measure value for prose (honours gbp/pct/number + scale). */
function fmt(value: number, measure?: DrillMeasure): string {
  if (!measure) return Number.isInteger(value) ? value.toLocaleString("en-GB") : value.toFixed(1);
  return formatValue(value, measure.format, measure.scale);
}

function pct(share: number): string {
  return `${Math.round(share * 100)}%`;
}

/** A clean prose noun for a measure label ("Balance (£MM)" → "Balance", "Share %" → "Share"). */
function measureNoun(label?: string): string {
  if (!label) return "value";
  return (
    label
      .replace(/\s*\(.*?\)\s*/g, " ")
      .replace(/\s*%\s*$/, "")
      .replace(/\s+/g, " ")
      .trim() || "value"
  );
}

function signedPct(ratio: number): string {
  const p = ratio * 100;
  return `${p >= 0 ? "+" : ""}${p.toFixed(1)}%`;
}

/** Dimension names that denote a point-in-time SNAPSHOT series (reporting month,
 * weekly extract, etc.) — deliberately excludes cross-sectional cohorts like
 * `vintage_year`, whose balances DO sum to the book total. */
const SNAPSHOT_DIM_RE = /(period|month|week|quarter|reporting|as[_\s]?of|extract|snapshot)/i;
/** A YYYY-MM or YYYY-MM-DD period value (a bare YYYY is treated as a cohort, not
 * a snapshot series, so vintage-year breakdowns keep their part-to-whole shares). */
const PERIOD_VALUE_RE = /^\d{4}-\d{2}(-\d{2})?$/;

/**
 * True when the dimension is a time-ordered STOCK snapshot series (e.g. funded
 * balance by reporting month). Summing a stock across snapshots double-counts the
 * overlapping book, so "% of total" / concentration / ranking observations are
 * meaningless here and must be suppressed.
 */
function isSnapshotTimeSeries(model: DrillModel): boolean {
  if (SNAPSHOT_DIM_RE.test(model.dimensionKey) || SNAPSHOT_DIM_RE.test(model.dimensionLabel)) {
    return true;
  }
  const periodLike = model.values.filter((v) => PERIOD_VALUE_RE.test(String(v).trim()));
  return periodLike.length >= 2 && periodLike.length >= Math.ceil(model.values.length * 0.8);
}

/* --------------------------- the engine --------------------------- */

/**
 * Analyse an artifact into an InsightSummary, or null when there's nothing to
 * say (not a chart/table, fewer than two categories, or no numeric measure).
 * Pure + synchronous; safe to call inside a try/catch on every result.
 */
export function computeInsights(
  artifact: Artifact,
  _spec?: Partial<MIQuerySpec>,
  opts?: InsightOptions,
): InsightSummary | null {
  if (!isChartArtifact(artifact) && !isTableArtifact(artifact)) return null;

  const model = buildDrillModel(artifact as DrillArtifact);
  if (!model || model.values.length < 2) return null;

  // Focus measure: the additive primary (balance/amount) if present, else the
  // first numeric measure (e.g. an average).
  const focus = model.primary ?? model.measures[0];
  if (!focus) return null;

  const series = model.values
    .map((label) => ({ label, value: valueOf(model, label, focus) }))
    .filter((s) => Number.isFinite(s.value));
  if (series.length < 2) return null;

  const sorted = [...series].sort((a, b) => b.value - a.value);
  const values = series.map((s) => s.value);
  const total = sum(values);
  const top = sorted[0];
  const bottom = sorted[sorted.length - 1];
  const sd = stdev(values);
  const m = mean(values);
  const top3 = sorted.slice(0, 3);

  // A stock snapshot series (funded balance by month) is NOT part-to-whole:
  // summing balances across snapshots double-counts the overlapping book, so
  // "% of total" concentration / ranking shares would be false. Only compute
  // additive shares for genuine cross-sectional breakdowns.
  const timeSeries = isSnapshotTimeSeries(model);
  const shareable = focus.additive && !timeSeries && total !== 0;

  const statistics: InsightStatistics = {
    measureKey: focus.key,
    measureLabel: focus.label,
    count: series.length,
    total,
    max: top.value,
    min: bottom.value,
    mean: m,
    median: median(values),
    stdev: sd,
    spread: top.value - bottom.value,
    topLabel: top.label,
    topValue: top.value,
    topShare: shareable ? top.value / total : undefined,
    bottomLabel: bottom.label,
    bottomValue: bottom.value,
    top3Share: shareable ? sum(top3.map((t) => t.value)) / total : undefined,
  };

  const mNoun = measureNoun(focus.label);
  const mLow = mNoun.toLowerCase();
  const observations: Observation[] = [];

  // 1. Concentration — share of the top bucket (additive measures only).
  if (statistics.topShare !== undefined) {
    const s = statistics.topShare;
    const severity: Severity = s > 0.6 ? "significant" : s > 0.4 ? "watch" : "info";
    observations.push({
      id: "concentration",
      kind: "concentration",
      severity,
      text: `${top.label} has the largest ${mLow}, at ${pct(s)} of the total.`,
    });
  }

  // 2. Ranking — share of the top 3.
  if (statistics.top3Share !== undefined && series.length >= 4) {
    const s = statistics.top3Share;
    observations.push({
      id: "ranking",
      kind: "ranking",
      severity: s > 0.8 ? "watch" : "info",
      text: `The top ${Math.min(3, series.length)} account for ${pct(s)} of ${mLow}.`,
    });
  }

  // 3. Outlier — a secondary average/rate measure far from its mean.
  const secondary = model.measures.find(
    (x) => x.key !== focus.key && !x.additive && (x.format === "pct" || x.format === "number"),
  );
  if (secondary) {
    const pairs = model.values
      .map((label) => ({ label, value: valueOf(model, label, secondary) }))
      .filter((p) => Number.isFinite(p.value));
    if (pairs.length >= 2) {
      const vals = pairs.map((p) => p.value);
      const avg = mean(vals);
      const sd2 = stdev(vals);
      const out = pairs.reduce((best, p) =>
        Math.abs(p.value - avg) > Math.abs(best.value - avg) ? p : best,
      );
      const dev = out.value - avg;
      if (sd2 > 0 && Math.abs(dev) >= sd2) {
        const severity: Severity = Math.abs(dev) > 2 * sd2 ? "significant" : "watch";
        const direction = dev >= 0 ? "above" : "below";
        const sLow = measureNoun(secondary.label).toLowerCase();
        const deltaText =
          secondary.format === "pct"
            ? `${Math.abs(toPercentPoints(out.value, secondary.scale) - toPercentPoints(avg, secondary.scale)).toFixed(1)}pp`
            : fmt(Math.abs(dev), secondary);
        observations.push({
          id: "outlier",
          kind: "outlier",
          severity,
          text: `${out.label}'s ${sLow} is ${deltaText} ${direction} the average (${fmt(out.value, secondary)} vs ${fmt(avg, secondary)}).`,
        });
      }
    }
  }

  // 4. Spread — range across buckets.
  if (statistics.spread > 0) {
    const aboveMean = top.value - m;
    const severity: Severity = sd > 0 && aboveMean > 2 * sd ? "significant" : sd > 0 && aboveMean > sd ? "watch" : "info";
    observations.push({
      id: "spread",
      kind: "spread",
      severity,
      text: `${mNoun} ranges from ${fmt(bottom.value, focus)} (${bottom.label}) to ${fmt(top.value, focus)} (${top.label}).`,
    });
  }

  // 4b. Trend — for a stock snapshot series, the meaningful movement is the
  // change between the earliest and latest snapshot (NOT a share of a summed
  // total). Uses the chronological endpoints of the series.
  if (timeSeries) {
    const chrono = [...series].sort((a, b) => String(a.label).localeCompare(String(b.label)));
    const firstPt = chrono[0];
    const lastPt = chrono[chrono.length - 1];
    if (firstPt && lastPt && firstPt.label !== lastPt.label && firstPt.value !== 0) {
      const ratio = (lastPt.value - firstPt.value) / Math.abs(firstPt.value);
      const severity: Severity = Math.abs(ratio) > 0.25 ? "significant" : Math.abs(ratio) > 0.1 ? "watch" : "info";
      observations.push({
        id: "trend",
        kind: "movement",
        severity,
        text: `${mNoun} moved from ${fmt(firstPt.value, focus)} (${firstPt.label}) to ${fmt(lastPt.value, focus)} (${lastPt.label}), ${signedPct(ratio)}.`,
      });
    }
  }

  // 5. Movement — only when a prior total is supplied (never invented).
  if (!timeSeries && opts?.previousTotal !== undefined && opts.previousTotal !== 0) {
    const ratio = (total - opts.previousTotal) / Math.abs(opts.previousTotal);
    const severity: Severity = Math.abs(ratio) > 0.25 ? "significant" : Math.abs(ratio) > 0.1 ? "watch" : "info";
    observations.push({
      id: "movement",
      kind: "movement",
      severity,
      text: `${mNoun} is ${signedPct(ratio)} versus ${opts.previousLabel ?? "the prior period"}.`,
    });
  }

  // Priority order: concentration, outlier, movement, spread, ranking.
  const priority: ObservationKind[] = ["concentration", "outlier", "movement", "spread", "ranking"];
  observations.sort((a, b) => priority.indexOf(a.kind) - priority.indexOf(b.kind));
  const trimmed = observations.slice(0, 5);

  const suggestions = buildInvestigations(statistics, secondary ? topOutlier(model, secondary, focus) : undefined);

  const headline = pickHeadline(trimmed);

  return { headline, observations: trimmed, suggestions, statistics };
}

/** Re-find the outlier label cheaply for the suggestion builder. */
function topOutlier(model: DrillModel, secondary: DrillMeasure, focus: DrillMeasure): string | undefined {
  if (secondary.key === focus.key) return undefined;
  const pairs = model.values
    .map((label) => ({ label, value: valueOf(model, label, secondary) }))
    .filter((p) => Number.isFinite(p.value));
  if (pairs.length < 2) return undefined;
  const avg = mean(pairs.map((p) => p.value));
  return pairs.reduce((best, p) => (Math.abs(p.value - avg) > Math.abs(best.value - avg) ? p : best)).label;
}

/**
 * Stat-driven investigations. Each routes through the existing follow-up
 * resolver as a value filter on the active dimension ("only X"), so it is always
 * executable. Capped at 4; deduplicated by question.
 */
export function buildInvestigations(stats: InsightStatistics, outlierLabel?: string): SuggestedAction[] {
  const out: SuggestedAction[] = [];
  const seen = new Set<string>();
  const push = (a: SuggestedAction) => {
    const k = a.question.toLowerCase();
    if (!seen.has(k) && out.length < 4) {
      seen.add(k);
      out.push(a);
    }
  };

  if (stats.topLabel && (stats.topShare ?? 0) >= 0.4) {
    push({ label: `Investigate ${stats.topLabel}`, question: `only ${stats.topLabel}`, kind: "drill" });
  }
  if (outlierLabel) {
    push({ label: `Analyse ${outlierLabel}`, question: `only ${outlierLabel}`, kind: "drill" });
  }
  // High variance → look at the smallest bucket too.
  if (stats.bottomLabel && stats.stdev > 0 && stats.spread > 2 * stats.stdev) {
    push({ label: `Investigate ${stats.bottomLabel}`, question: `only ${stats.bottomLabel}`, kind: "drill" });
  }
  return out;
}

function pickHeadline(obs: Observation[]): string | undefined {
  if (!obs.length) return undefined;
  const order: Severity[] = ["significant", "watch", "info"];
  const ranked = [...obs].sort((a, b) => order.indexOf(a.severity) - order.indexOf(b.severity));
  return ranked[0].text;
}

/** Convenience used by the renderer's empty-state guard. */
export function hasInsights(summary: InsightSummary | null): summary is InsightSummary {
  return !!summary && summary.observations.length > 0;
}
