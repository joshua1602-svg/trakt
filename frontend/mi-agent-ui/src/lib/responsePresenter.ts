/**
 * Response presenter — turns a query result into a plain-English, analyst-style
 * sentence for the chat. Deterministic (no LLM). It NEVER shows parser /
 * validation / aggregation / metric / dimension internals; those live only in
 * the collapsed Query Logic disclosure.
 *
 * It is defensive about the upstream narrative: some backends return the parser
 * interpretation as the "answer" (e.g. "Chart: Bar · Metric: Balance · … ·
 * Validation: Passed"). Such debug-style text is detected and discarded in
 * favour of a grounded sentence built from the result's own statistics.
 */

import type { Artifact, MIQuerySpec } from "@/domain";
import { isChartArtifact, isKPIArtifact, isTableArtifact, isValidationArtifact } from "@/domain";
import { computeInsights } from "@/lib/insights";
import { cleanLabel, dimensionLabel, measureLabel } from "@/lib/analysisContext";
import { MEASURES } from "@/data/catalog";
import { formatValue } from "@/lib/utils";

const MEASURE_FORMAT = new Map(MEASURES.map((m) => [m.key, m.format]));

const DEBUG_RE = /\b(parser|validation|aggregation|metric|dimension|chart\s*type|intent)\s*[:=]/i;

/** True when a narrative is parser/debug-style rather than human prose. */
export function isDebugNarrative(text?: string | null): boolean {
  return !!text && DEBUG_RE.test(text);
}

// Substance threshold: a genuinely analytical narrative (the rich mock prose, or
// an LLM-polished answer) is kept; short generic filler is replaced by the
// data-grounded sentence.
const RICH_MIN = 70;
function isRichProse(text?: string | null): boolean {
  return !!text && !isDebugNarrative(text) && text.trim().length >= RICH_MIN;
}

function fmtMeasure(value: number | undefined, measureKey?: string): string {
  if (value === undefined || !Number.isFinite(value)) return "n/a";
  return formatValue(value, measureKey ? MEASURE_FORMAT.get(measureKey) : undefined);
}

function noun(measureKey?: string, fallback = "the result"): string {
  const lbl = measureLabel(measureKey);
  return lbl ? (cleanLabel(lbl) ?? fallback).toLowerCase() : fallback;
}

/** A grounded sentence from a chart/table's own distribution. */
function groundedSentence(artifact: Artifact, spec?: Partial<MIQuerySpec>): string | null {
  let insights = null;
  try {
    insights = computeInsights(artifact, spec);
  } catch {
    insights = null;
  }
  const measureKey = spec?.metric;
  const mLow = noun(measureKey, "the result");
  const dimKey = spec?.dimensions?.[0] ?? spec?.dimension;
  const dLow = dimensionLabel(dimKey) ? (cleanLabel(dimensionLabel(dimKey)) ?? "").toLowerCase() : "";
  const byPart = dLow ? ` by ${dLow}` : "";

  if (insights?.statistics.topLabel) {
    const st = insights.statistics;
    const topVal = fmtMeasure(st.topValue, measureKey);
    const share =
      st.topShare !== undefined ? ` — ${Math.round(st.topShare * 100)}% of the ${fmtMeasure(st.total, measureKey)} total` : "";
    return `${st.topLabel} has the largest ${mLow} at ${topVal}${share}. I've shown ${mLow}${byPart} below so you can compare it with the rest of the portfolio.`;
  }
  return `Here is ${mLow}${byPart}, shown below.`;
}

export interface PresentArgs {
  question: string;
  ok: boolean;
  spec?: Partial<MIQuerySpec>;
  artifacts: Artifact[];
  narrative?: string;
  error?: string;
}

/** Produce the plain-English assistant message for a result. */
export function presentAnswer(args: PresentArgs): string {
  if (!args.ok) {
    if (args.error && !isDebugNarrative(args.error)) return args.error;
    return "I couldn't complete that query. Try rephrasing it, or ask about balances, concentration, pipeline, stratifications, risk or validation.";
  }

  // Keep a genuinely analytical narrative (rich mock prose / LLM-polished text).
  if (isRichProse(args.narrative)) return args.narrative!;

  // For a chart/table, prefer the data-grounded sentence (value + share + intro)
  // over short generic or parser/debug "answers".
  const primary = args.artifacts.find(isChartArtifact) ?? args.artifacts.find(isTableArtifact);
  if (primary) {
    const sentence = groundedSentence(primary, args.spec);
    if (sentence) return sentence;
  }

  // No groundable chart/table — keep any non-debug narrative before falling back.
  if (args.narrative && !isDebugNarrative(args.narrative)) return args.narrative;

  if (args.artifacts.some(isKPIArtifact)) return "Here are the headline portfolio metrics, shown below.";
  if (args.artifacts.some(isValidationArtifact)) return "I've run the validation checks — the results are below.";
  return "Here's what I found, shown below.";
}
