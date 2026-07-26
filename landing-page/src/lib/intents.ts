import "server-only";

import { demoPack } from "@/lib/demo-pack";

/**
 * Allow-listed intent resolution for the public demo.
 *
 * This is the whole "understanding" layer, and it is deliberately dumb: a
 * deterministic scorer over a fixed phrase table. Visitor text is never sent to
 * a model, never sent to the Trakt MI engine, and never used to build a query —
 * it only ever *selects* one of a fixed set of pre-computed answers, or selects
 * nothing. There is no code path from visitor input to portfolio computation.
 */

export type Resolution =
  | { kind: "intent"; id: string; score: number }
  | { kind: "report"; id: string; score: number }
  | { kind: "unsupported"; id: string; score: number }
  | { kind: "unmatched" };

interface Candidate {
  kind: "intent" | "report" | "unsupported";
  id: string;
  phrases: string[];
  label: string;
}

const CANDIDATES: Candidate[] = [
  ...demoPack.intents.map((i) => ({
    kind: "intent" as const,
    id: i.id,
    phrases: i.phrases,
    label: i.label,
  })),
  ...demoPack.reports.map((r) => ({
    kind: "report" as const,
    id: r.id,
    phrases: r.phrases,
    label: r.label,
  })),
  ...demoPack.unsupported.map((u) => ({
    kind: "unsupported" as const,
    id: u.id,
    phrases: u.phrases,
    label: u.label,
  })),
];

/** Words too common to carry intent on their own. */
const STOP_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
  "for", "from", "give", "how", "i", "in", "is", "it", "just", "me", "much",
  "my", "of", "on", "or", "our", "please", "show", "so", "some", "tell", "that",
  "the", "their", "there", "these", "this", "to", "us", "was", "we", "what",
  "whats", "when", "where", "which", "who", "why", "will", "with", "would",
  "you", "your",
]);

function normalise(text: string): string {
  return text
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9%\s-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokens(text: string): string[] {
  return normalise(text)
    .split(" ")
    .filter((t) => t.length > 1 && !STOP_WORDS.has(t));
}

/**
 * Score one candidate against the normalised question.
 *
 * A whole-phrase hit dominates; otherwise the score is the share of the
 * phrase's significant tokens that appear in the question, so "what's the ltv
 * looking like" still reaches `wa_ltv` while unrelated text reaches nothing.
 */
function score(question: string, questionTokens: Set<string>, phrases: string[]): number {
  let best = 0;
  for (const phrase of phrases) {
    const normalisedPhrase = normalise(phrase);
    if (!normalisedPhrase) continue;

    if (question.includes(normalisedPhrase)) {
      // Longer phrase matches are stronger evidence than short ones.
      best = Math.max(best, 1 + normalisedPhrase.length / 100);
      continue;
    }

    const phraseTokens = tokens(phrase);
    if (phraseTokens.length === 0) continue;
    const hits = phraseTokens.filter((t) => questionTokens.has(t)).length;
    if (hits === 0) continue;
    // Require every significant token of a one- or two-word phrase.
    const coverage = hits / phraseTokens.length;
    if (phraseTokens.length <= 2 && coverage < 1) continue;
    if (coverage < 0.6) continue;
    best = Math.max(best, 0.4 + coverage * 0.5);
  }
  return best;
}

/** Anything below this is treated as "we did not understand you". */
const MATCH_THRESHOLD = 0.7;

/**
 * Resolve free text to exactly one allow-listed outcome.
 *
 * `unsupported` candidates are scored alongside supported ones and win ties,
 * because refusing honestly is better than answering the adjacent question: a
 * visitor asking about arrears should be told there are none, not handed a
 * balance.
 */
export function resolveQuestion(raw: string): Resolution {
  const question = normalise(raw);
  if (!question) return { kind: "unmatched" };
  const questionTokens = new Set(tokens(raw));

  let winner: { candidate: Candidate; score: number } | null = null;
  for (const candidate of CANDIDATES) {
    const value = score(question, questionTokens, candidate.phrases);
    if (value < MATCH_THRESHOLD) continue;
    if (
      !winner ||
      value > winner.score ||
      (value === winner.score &&
        candidate.kind === "unsupported" &&
        winner.candidate.kind !== "unsupported")
    ) {
      winner = { candidate, score: value };
    }
  }

  if (!winner) return { kind: "unmatched" };
  return { kind: winner.candidate.kind, id: winner.candidate.id, score: winner.score };
}

/** Direct selection of a suggested chip. Never falls back to text matching. */
export function resolveId(id: string): Resolution {
  const candidate = CANDIDATES.find((c) => c.id === id);
  if (!candidate) return { kind: "unmatched" };
  return { kind: candidate.kind, id: candidate.id, score: 1 };
}
