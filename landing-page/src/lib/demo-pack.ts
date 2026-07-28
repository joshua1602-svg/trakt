import "server-only";

import packJson from "@data/demo-pack.json";
import type {
  DemoIntent,
  DemoMetaResponse,
  DemoPack,
  DemoReport,
  UnsupportedTopic,
} from "@/types/demo";
import { LIMITS } from "@/lib/env";

/** The pack shape this build understands. A mismatch is a build-time failure. */
const EXPECTED_PACK_VERSION = 1;

const pack = packJson as unknown as DemoPack;

if (pack.packVersion !== EXPECTED_PACK_VERSION) {
  throw new Error(
    `demo-pack.json is version ${pack.packVersion}; this build expects ` +
      `${EXPECTED_PACK_VERSION}. Re-run scripts/build_demo_pack.py.`,
  );
}

export const demoPack: DemoPack = pack;

const intentsById = new Map<string, DemoIntent>(pack.intents.map((i) => [i.id, i]));
const reportsById = new Map<string, DemoReport>(pack.reports.map((r) => [r.id, r]));
const unsupportedById = new Map<string, UnsupportedTopic>(
  pack.unsupported.map((u) => [u.id, u]),
);

/** The complete allow-list of answerable intent ids. */
export const ALLOWED_INTENT_IDS: readonly string[] = Object.freeze([...intentsById.keys()]);
export const ALLOWED_REPORT_IDS: readonly string[] = Object.freeze([...reportsById.keys()]);

/** The single portfolio this demo will ever answer for. */
export const ALLOWED_PORTFOLIO_IDS: readonly string[] = Object.freeze([pack.portfolio.id]);

export function getIntent(id: string): DemoIntent | null {
  return intentsById.get(id) ?? null;
}

export function getReport(id: string): DemoReport | null {
  return reportsById.get(id) ?? null;
}

export function getUnsupported(id: string): UnsupportedTopic | null {
  return unsupportedById.get(id) ?? null;
}

export function labelFor(id: string): string | null {
  return intentsById.get(id)?.label ?? reportsById.get(id)?.label ?? null;
}

/**
 * Four questions plus one report action — five choices, in one row.
 *
 * Five, not eleven: the previous set demonstrated a single capability, a
 * stratification, five different ways. Each of these shows something the others
 * do not, and the report action is folded in beside them rather than sitting
 * under its own heading, so the row reads as five choices and not two groups.
 *
 * Concentration and the stratifications stay in the pack — reachable as
 * follow-ups and by typing — but they are no longer what the page leads with.
 */
const SUGGESTED_ORDER = [
  "balance_by_book",    // the headline figure, and the differentiator
  "period_movement",    // movement, differenced from two governed snapshots
  "annex_exceptions",   // regulatory exceptions, with their reasoning
  "data_quality",       // "how do I know these numbers are right?"
] as const;

/** The single report action offered alongside them. */
const SUGGESTED_REPORT_ID = "investor_report";

export function buildMeta(): DemoMetaResponse {
  const suggested = SUGGESTED_ORDER.map((id) => intentsById.get(id)).filter(
    (i): i is DemoIntent => Boolean(i),
  );

  return {
    scope: {
      client: pack.client.name,
      clientDescription: pack.client.description,
      portfolio: pack.portfolio.id,
      portfolioName: pack.portfolio.name,
      assetClass: pack.portfolio.assetClass,
      asOfDate: pack.portfolio.asOfDate,
      asOfDisplay: pack.portfolio.asOfDisplay,
      loanCount: pack.portfolio.loanCount,
      totalBalanceDisplay: pack.portfolio.totalBalanceDisplay,
      platformBalanceDisplay: pack.portfolio.platformBalanceDisplay,
      books: pack.portfolio.books,
      currency: pack.portfolio.currency,
      regulatoryRegime: pack.portfolio.regulatoryRegime,
      synthetic: true,
    },
    suggestedQuestions: suggested.map((i) => ({
      id: i.id,
      label: i.label,
      category: i.category,
    })),
    reportActions: pack.reports
      .filter((r) => r.id === SUGGESTED_REPORT_ID)
      .map((r) => ({
        id: r.id,
        label: r.label,
        documentTitle: r.documentTitle,
      })),
    // The first two are the two the page shows. Order is set in the pack:
    // loan_level refuses on governance grounds and pipeline on a genuine data
    // gap — both remain underivable now that movement is answerable.
    exampleUnsupported: pack.unsupported
      .slice(0, 2)
      .map((u) => ({ id: u.id, label: u.label })),
    limits: {
      questionsPerSession: LIMITS.questionsPerSession,
      reportsPerSession: LIMITS.reportsPerSession,
      maxQuestionLength: LIMITS.maxQuestionLength,
    },
    provenanceNote: pack.provenance.note,
  };
}

export function followUpsFor(ids: string[]): { id: string; label: string }[] {
  return ids
    .map((id) => {
      const label = labelFor(id);
      return label ? { id, label } : null;
    })
    .filter((x): x is { id: string; label: string } => Boolean(x));
}
