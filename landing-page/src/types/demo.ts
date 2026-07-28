/**
 * The public demo-pack contract.
 *
 * These types describe `landing-page/data/demo-pack.json`, which is produced by
 * `scripts/build_demo_pack.py` from the Trakt deterministic MI engine. They are
 * the only shape the browser ever sees — the engine's internal query spec,
 * diagnostics and provenance paths are stripped at generation time.
 */

export type ValueFormat = "text" | "number" | "gbp" | "pct" | "date";

export interface DemoColumn {
  key: string;
  label: string;
  align: "left" | "right" | "center";
  format: ValueFormat | string;
  scale: string | null;
}

export type DemoRow = Record<string, string | number | boolean | null>;

export interface KpiArtifact {
  kind: "kpi";
  title: string | null;
  kpis: { label: string; value: string }[];
  coverage: number | null;
}

export interface TableArtifact {
  kind: "table";
  title: string | null;
  columns: DemoColumn[];
  rows: DemoRow[];
  totalRows: number;
  coverage: number | null;
}

export interface ChartArtifact extends Omit<TableArtifact, "kind"> {
  kind: "chart";
  chartType: string;
  xKey: string;
  valueKey: string | null;
  valueFormat: string | null;
}

export type DemoArtifact = KpiArtifact | TableArtifact | ChartArtifact;

export interface DemoIntent {
  id: string;
  label: string;
  category: string;
  /** Visitor phrasings the runtime matcher accepts. Never sent anywhere. */
  phrases: string[];
  answer: string;
  interpreted: string | null;
  artifacts: DemoArtifact[];
  followUps: string[];
}

export interface ReportPage {
  title: string;
  subtitle: string | null;
  blocks: DemoArtifact[];
  note: string | null;
}

export interface DemoReport {
  id: string;
  label: string;
  category: "report";
  phrases: string[];
  documentTitle: string;
  documentSubtitle: string;
  answer: string;
  pages: ReportPage[];
  followUps: string[];
}

export interface UnsupportedTopic {
  id: string;
  label: string;
  phrases: string[];
  reason: string;
  productionNote: string;
}

export interface DemoPack {
  packVersion: number;
  client: {
    id: string;
    name: string;
    originator: string;
    description: string;
    synthetic: true;
  };
  portfolio: {
    id: string;
    name: string;
    assetClass: string;
    currency: string;
    country: string;
    asOfDate: string;
    asOfDisplay: string;
    loanCount: number;
    totalBalance: number;
    /** Sponsor scope: every book, including the sold SPV. */
    totalBalanceDisplay: string;
    /** Warehoused scope: the books still on the sponsor's balance sheet. */
    platformBalanceDisplay: string;
    books: DemoBookInfo[];
    priorAsOfDate: string;
    regulatoryRegime: string;
    deliveryPreflight: string | null;
  };
  provenance: { sourceDataset: string; engine: string[]; note: string };
  intents: DemoIntent[];
  reports: DemoReport[];
  unsupported: UnsupportedTopic[];
}

/* -------------------------------------------------------------------------- */
/* Wire contracts (browser ⇄ /api/demo/*)                                      */
/* -------------------------------------------------------------------------- */

/** One governed book, as the pack describes it. */
export interface DemoBookInfo {
  id: string;
  label: string;
  shortLabel: string;
  /** "warehoused" (on balance sheet) or "sold" (derecognised). */
  balanceSheetStatus: string;
  statusDetail: string;
  balanceDisplay: string;
}

export interface DemoScopeInfo {
  client: string;
  clientDescription: string;
  portfolio: string;
  portfolioName: string;
  assetClass: string;
  asOfDate: string;
  asOfDisplay: string;
  loanCount: number;
  /** Sponsor scope: every book, including the sold SPV. */
  totalBalanceDisplay: string;
  /** Warehoused scope: the books still on the sponsor's balance sheet. */
  platformBalanceDisplay: string;
  books: DemoBookInfo[];
  currency: string;
  regulatoryRegime: string;
  synthetic: true;
}

export interface DemoMetaResponse {
  scope: DemoScopeInfo;
  suggestedQuestions: { id: string; label: string; category: string }[];
  reportActions: { id: string; label: string; documentTitle: string }[];
  exampleUnsupported: { id: string; label: string }[];
  limits: { questionsPerSession: number; reportsPerSession: number; maxQuestionLength: number };
  provenanceNote: string;
}

export type DemoAnswerStatus = "answered" | "unsupported" | "limit_reached" | "error";

export interface DemoAnswerResponse {
  status: DemoAnswerStatus;
  intentId: string | null;
  question: string;
  answer: string;
  /** Present when status === "answered". */
  interpreted?: string | null;
  artifacts?: DemoArtifact[];
  asOfDate?: string;
  asOfDisplay?: string;
  portfolioScope?: string;
  followUps?: { id: string; label: string }[];
  /** Present when status === "unsupported". */
  productionNote?: string;
  synthetic: true;
  usage: { questionsUsed: number; questionsRemaining: number };
}

export interface DemoReportResponse {
  status: "ready" | "limit_reached" | "error";
  reportId: string | null;
  documentTitle?: string;
  documentSubtitle?: string;
  answer?: string;
  pages?: ReportPage[];
  followUps?: { id: string; label: string }[];
  synthetic: true;
  usage: { reportsUsed: number; reportsRemaining: number };
}
