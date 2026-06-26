/**
 * Analysis Context — a lightweight, spec-level memory of the last SUCCESSFUL MI
 * query so short follow-up prompts ("split by broker", "only South East", "now
 * average LTV") can be resolved against it. This is an ADDITIVE wrapper around
 * the existing NLQ flow: when a prompt doesn't look like a follow-up, or can't
 * be resolved, callers fall back to sending the raw question unchanged.
 *
 * It stores only spec metadata (measure / dimensions / filters) — never raw rows.
 */

import type { MIQuerySpec } from "@/domain";
import { DIMENSIONS, MEASURES } from "@/data/catalog";

export interface AnalysisContext {
  portfolioId?: string;
  lastQuestion?: string;
  lastSuccessfulSpec?: Partial<MIQuerySpec>;
  lastArtifactId?: string;
  /** Active measure semantic key (e.g. current_outstanding_balance). */
  activeMeasure?: string;
  /** Active dimension semantic keys (e.g. ["geographic_region_obligor"]). */
  activeDimensions?: string[];
  /** Active filters keyed by semantic field. */
  activeFilters?: Record<string, unknown>;
  activeTimeContext?: string;
  updatedAt?: string;
}

const MEASURE_BY_KEY = new Map(MEASURES.map((m) => [m.key, m]));
const DIMENSION_BY_KEY = new Map(DIMENSIONS.map((d) => [d.key, d]));

/** Synonyms → catalogue dimension key (grounded; never invents a field). */
const DIMENSION_SYNONYMS: Record<string, string> = {
  region: "geographic_region_obligor",
  geography: "geographic_region_obligor",
  broker: "broker_channel",
  channel: "broker_channel",
  originator: "broker_channel",
  product: "erm_product_type",
  "product type": "erm_product_type",
  stage: "pipeline_stage",
  status: "account_status",
  "risk grade": "internal_risk_grade",
  grade: "internal_risk_grade",
  vintage: "vintage_year",
  year: "vintage_year",
  spv: "spv_id",
  portfolio: "portfolio_id",
  "age band": "age_bucket",
  age: "age_bucket",
  "ltv band": "ltv_bucket",
  "rate type": "interest_rate_type",
};

/** Synonyms → catalogue measure key. */
const MEASURE_SYNONYMS: Record<string, string> = {
  balance: "current_outstanding_balance",
  "outstanding balance": "current_outstanding_balance",
  principal: "current_principal_balance",
  valuation: "current_valuation_amount",
  value: "current_valuation_amount",
  ltv: "current_loan_to_value",
  "loan to value": "current_loan_to_value",
  coupon: "current_interest_rate",
  rate: "current_interest_rate",
  "interest rate": "current_interest_rate",
  age: "youngest_borrower_age",
  arrears: "arrears_balance",
  default: "default_amount",
  redemptions: "redemptions_received_in_period",
  "forecast funded": "forecast_funded_balance",
};

export function measureLabel(key?: string): string | undefined {
  return key ? MEASURE_BY_KEY.get(key)?.label : undefined;
}
export function dimensionLabel(key?: string): string | undefined {
  return key ? DIMENSION_BY_KEY.get(key)?.label : undefined;
}

/** A parser-friendly label: drop a parenthetical qualifier ("Region (Obligor)" → "Region"). */
export function cleanLabel(label?: string): string | undefined {
  return label
    ?.replace(/\s*\(.*?\)\s*/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function norm(s: string): string {
  return s.trim().toLowerCase().replace(/\s+/g, " ");
}

/** Best-effort match of free text to a catalogue dimension (grounded). */
export function matchDimension(text: string): { key: string; label: string } | undefined {
  const t = norm(text);
  for (const [syn, key] of Object.entries(DIMENSION_SYNONYMS)) {
    if (t.includes(syn)) return { key, label: DIMENSION_BY_KEY.get(key)!.label };
  }
  const hit = DIMENSIONS.find((d) => t.includes(norm(d.label)) || norm(d.label).includes(t) || d.key.includes(t));
  return hit ? { key: hit.key, label: hit.label } : undefined;
}

/** Best-effort match of free text to a catalogue measure (grounded). */
export function matchMeasure(text: string): { key: string; label: string } | undefined {
  const t = norm(text);
  for (const [syn, key] of Object.entries(MEASURE_SYNONYMS)) {
    if (t.includes(syn)) return { key, label: MEASURE_BY_KEY.get(key)!.label };
  }
  const hit = MEASURES.find((m) => t.includes(norm(m.label)) || norm(m.label).includes(t) || m.key.includes(t));
  return hit ? { key: hit.key, label: hit.label } : undefined;
}

/** Derive context from a SUCCESSFUL query's spec (callers gate on res.ok). */
export function deriveContext(args: {
  spec?: Partial<MIQuerySpec>;
  question: string;
  artifactId?: string;
  portfolioId?: string;
  now?: string;
}): AnalysisContext {
  const spec = args.spec ?? {};
  const dims = spec.dimensions?.length ? spec.dimensions : spec.dimension ? [spec.dimension] : undefined;
  const filters = spec.filters && Object.keys(spec.filters).length ? spec.filters : {};
  return {
    portfolioId: args.portfolioId,
    lastQuestion: args.question,
    lastSuccessfulSpec: spec,
    lastArtifactId: args.artifactId,
    activeMeasure: spec.metric,
    activeDimensions: dims,
    activeFilters: filters,
    updatedAt: args.now,
  };
}

/** A short, unobtrusive label for the active context indicator. */
export function contextSummary(ctx: AnalysisContext | null | undefined): string | null {
  if (!ctx?.lastSuccessfulSpec) return null;
  const parts: string[] = [];
  const m = cleanLabel(measureLabel(ctx.activeMeasure));
  if (m) parts.push(m);
  for (const d of ctx.activeDimensions ?? []) {
    parts.push(cleanLabel(dimensionLabel(d)) ?? d);
  }
  for (const v of Object.values(ctx.activeFilters ?? {})) {
    parts.push(String(v));
  }
  return parts.length ? parts.join(" · ") : null;
}

// Connective phrasings that unambiguously signal a contextual follow-up.
const FOLLOWUP_TRIGGERS: RegExp[] = [
  /^\s*(?:split|group|broken?\s*down|break\s*(?:it\s*)?down)\s+by\b/i,
  /^\s*(?:now|then|instead|switch\s+to|change\s+to)\b/i,
  /^\s*(?:only|just|filter(?:\s+to)?)\b/i,
  /^\s*(?:drill\s+(?:in|into)|zoom\s+(?:in|into))\b/i,
  /^\s*(?:exclude|without)\b/i,
  /^\s*(?:what|how)\s+about\b/i,
  /\bas\s+(?:a\s+)?(?:table|chart|bar|line|grid)\b/i,
  /^\s*compare\s+(?:to|with|against)\b/i,
];

function hasFullStructure(q: string): boolean {
  // A complete "measure by dimension" question is a standalone, not a follow-up.
  return /\bby\b/i.test(q) && !!matchMeasure(q) && !!matchDimension(q.replace(/^.*\bby\b/i, ""));
}

/** True when the prompt looks like a follow-up that should use prior context. */
export function looksLikeFollowUp(question: string, context?: AnalysisContext | null): boolean {
  if (!context?.lastSuccessfulSpec) return false;
  const q = question.trim();
  if (!q) return false;
  if (hasFullStructure(q)) return false; // already a complete question
  return FOLLOWUP_TRIGGERS.some((p) => p.test(q));
}

export interface ResolvedFollowUp {
  /** The standalone question to send through the existing /mi/query flow. */
  question: string;
  /** Optional filters merged from context (reuses the backend filter seam). */
  filters?: Record<string, unknown>;
  /** Short human note for the Query Logic panel (e.g. "dimension → Broker"). */
  note: string;
}

function baseQuestion(ctx: AnalysisContext): string {
  const m = cleanLabel(measureLabel(ctx.activeMeasure)) ?? "balance";
  const d = cleanLabel(dimensionLabel(ctx.activeDimensions?.[0]));
  return d ? `${m} by ${d}` : m;
}

/**
 * Resolve a follow-up prompt into a standalone question (+ filters) using
 * context. Returns null when it can't be resolved — the caller then falls back
 * to sending the raw question through the unchanged flow.
 *
 * Supported (robust, deterministic): change dimension, change measure, add a
 * value filter (only / drill into / what about), output mode (as table/chart).
 * Not resolved (→ fallback): exclude (categorical not-equal), temporal compare.
 */
export function resolveFollowUp(question: string, ctx: AnalysisContext): ResolvedFollowUp | null {
  if (!ctx.lastSuccessfulSpec) return null;
  const q = question.trim();
  const lower = q.toLowerCase();

  // 1. Change dimension: "split by X" / "group by X" / "break down by X".
  let m = lower.match(/^(?:split|group|broken?\s*down|break\s*(?:it\s*)?down)\s+by\s+(.+)$/);
  if (m) {
    const dim = matchDimension(m[1]);
    if (!dim) return null;
    const measure = cleanLabel(measureLabel(ctx.activeMeasure)) ?? "balance";
    const dimLbl = cleanLabel(dim.label) ?? dim.label;
    return { question: `${measure} by ${dimLbl}`, filters: ctx.activeFilters, note: `dimension → ${dim.label}` };
  }

  // 2. Change measure: "now X" / "then X" / "switch to X".
  m = lower.match(/^(?:now|then|instead|switch\s+to|change\s+to)\s+(.+)$/);
  if (m) {
    const phrase = m[1].replace(/^(?:show|display|view)\s+(?:me\s+)?/, "").trim();
    const meas = matchMeasure(phrase);
    if (meas) {
      const d = cleanLabel(dimensionLabel(ctx.activeDimensions?.[0]));
      // Reuse the user's words so aggregation hints ("average ltv") survive.
      return { question: d ? `${phrase} by ${d}` : phrase, filters: ctx.activeFilters, note: `measure → ${meas.label}` };
    }
    // "now as a table" etc. → fall through to output-mode handling.
  }

  // 3. Output mode: "show as table" / "as a chart".
  m = lower.match(/\bas\s+(?:a\s+)?(table|chart|bar|line|grid)\b/);
  if (m) {
    const mode = m[1] === "grid" ? "table" : m[1];
    const stem = ctx.lastQuestion ?? baseQuestion(ctx);
    return { question: `${stem} as a ${mode}`, filters: ctx.activeFilters, note: `output → ${mode}` };
  }

  // 4. Add a value filter on the active dimension: "only X" / "drill into X" /
  //    "what about X". Defaults to the current dimension (the common sequence).
  m = lower.match(/^(?:only|just|filter(?:\s+to)?|drill\s+(?:in|into)|zoom\s+(?:in|into)|what\s+about|how\s+about)\s+(.+)$/);
  if (m) {
    const dimKey = ctx.activeDimensions?.[0];
    if (!dimKey) return null;
    const value = q.slice(q.length - m[1].length).trim().replace(/[?.!]+$/, "");
    if (!value) return null;
    const filters = { ...(ctx.activeFilters ?? {}), [dimKey]: value };
    return { question: baseQuestion(ctx), filters, note: `filter ${dimensionLabel(dimKey) ?? dimKey} = ${value}` };
  }

  // exclude / compare → not expressible safely; fall back to the raw flow.
  return null;
}
