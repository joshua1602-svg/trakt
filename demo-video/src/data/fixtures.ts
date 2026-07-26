/**
 * The film's data layer.
 *
 * Every figure the video shows is read from the fixtures written by
 * `python -m demo_platform.run_demo`. This module does not compute, derive or
 * reformat any business value — it selects. If a number is not in the fixture, it
 * does not appear on screen.
 *
 * The fixtures are imported (not fetched), so the bundle is self-contained and a
 * render never touches the network or the filesystem at frame time.
 */

import metricsFixture from "../../public/fixtures/demo_metrics.json";
import artefactFixture from "../../public/fixtures/artefact_catalogue.json";
import assertionFixture from "../../public/fixtures/assertion_report.json";
import safetyFixture from "../../public/fixtures/safety_report.json";
import manifestFixture from "../../public/fixtures/demo_manifest.json";
import { STATED_ARRIVAL } from "../claims";

// --------------------------------------------------------------------------- //
// Shapes (only the fields the film reads are typed)
// --------------------------------------------------------------------------- //
export interface PortfolioNarrative {
  key: string;
  source_portfolio_id: string;
  source_portfolio_type: string;
  display_id: string;
  label: string;
  short_label: string;
  description: string;
  schema_name: string;
  acquisition_date: string | null;
  seller_name: string | null;
  /** "warehoused" (held) or "sold" (derecognised). Not the same as the type. */
  balance_sheet_status: string;
  /** One-line plain-English status for the consolidation lanes. */
  status_detail: string;
}

/**
 * The two scopes, and the rule that keeps them apart.
 *
 *   PLATFORM — the warehoused books the assembler consolidates. £1.96bn / 11,035
 *              loans. EVERY figure in the film is this scope, without exception,
 *              apart from the one beat below.
 *   SPONSOR  — the platform plus the securitisation the sponsor sold. £2.81bn /
 *              15,215 loans. Used ONCE, in S3's consolidation beat, because it is the
 *              number no single system can produce today.
 *
 * Confusing the two is the main risk in the film. A unit test asserts that the sponsor
 * figure appears in exactly one scene, and the scope stamp on S4's result band names
 * which scope the viewer is looking at.
 */
export interface Scope {
  label: string;
  portfolios: string[];
  fundedBalance: number;
  loanCount: number;
}

export interface SponsoredScope {
  sourcePortfolioId: string;
  displayId: string;
  label: string;
  status: string;
  fundedBalance: number;
  loanCount: number;
  reportingDate: string;
}

export interface PeriodNarrative {
  run_id: string;
  reporting_date: string;
  period: string;
  label: string;
  short_label: string;
  role: string;
}

export interface SourceColumn {
  header: string;
  canonical_field: string | null;
  resolution: string;
  note: string;
}

export interface SourceSchema {
  name: string;
  system_of_record: string;
  description: string;
  headers: string[];
  columns: SourceColumn[];
}

export interface LensMetrics {
  funded_balance: number | null;
  loan_count: number | null;
  wa_ltv_points: number | null;
  wa_interest_rate: number | null;
  avg_borrower_age: number | null;
}

export interface RegionExposure {
  region: string;
  balance: number;
  share: number | null;
}

export interface RegionContribution {
  region: string;
  start: number;
  end: number;
  delta: number;
}

export interface CohortMovement {
  id: string;
  label: string;
  current: number;
  prior: number;
  delta: number;
}

export interface MovementBlock {
  available: boolean;
  currentPeriod: string;
  priorPeriod: string;
  currentReportingDate: string;
  priorReportingDate: string;
  current: LensMetrics;
  prior: LensMetrics;
  delta: LensMetrics;
  regionContributions: RegionContribution[];
  primaryRegion: RegionContribution | null;
  runnerUpRegion: RegionContribution | null;
  primaryRegionShare: number | null;
  primaryRegionLead: number | null;
  /** True when the primary region leads the runner-up decisively enough for the
   *  answer to say "primarily driven by" rather than naming a largest mover. */
  primaryRegionIsDominant: boolean;
  completionsBalanceInMonth: number | null;
  completionsBalanceInPrimaryRegion: number | null;
  completionsShareOfPrimaryRegion: number | null;
  completionsEvidenced: boolean;
  cohortMovements: CohortMovement[];
  topRegions: RegionExposure[];
}

export interface SummaryBlock {
  available: boolean;
  period: string;
  reportingDate: string;
  metrics: LensMetrics;
  topRegions: RegionExposure[];
  cohorts: { id: string; label: string }[];
  cohortBalances: Record<string, number>;
  periodCount: number;
}

export interface SeriesPoint {
  period: string;
  reportingDate: string;
  loanCount: number | null;
  fundedBalance: number | null;
  waCurrentLtvPoints: number | null;
  waInterestRate: number | null;
  avgBorrowerAge: number | null;
}

export interface Kpi {
  id: string;
  label: string;
  value: string;
  format: string;
  raw: number | null;
  available: boolean;
}

export interface ChatArtifact {
  type: string;
  title?: string;
  description?: string;
  kpis?: { label: string; value: string }[];
  chartType?: string;
  xKey?: string;
  rows?: Record<string, unknown>[];
  series?: { key: string; label: string; color?: string }[];
  columns?: { key: string; label: string; align?: string; format?: string }[];
}

export interface ChatTurn {
  question: string;
  answer: string;
  artifacts: ChatArtifact[];
  route?: string | null;
}

export interface ArtefactEntry {
  available: boolean;
  kind?: string;
  title?: string;
  fileName?: string;
  format?: string;
  rows?: number;
  columns?: number;
  slides?: number;
  sizeBytes?: number;
  reason?: string | null;
  note?: string;
  [key: string]: unknown;
}

// --------------------------------------------------------------------------- //
// Accessors
// --------------------------------------------------------------------------- //
const F = metricsFixture as unknown as {
  synthetic_notice: string;
  narrative: {
    synthetic_banner: string;
    client: { client_id: string; display_name: string; short_name: string };
    portfolios: PortfolioNarrative[];
    periods: PeriodNarrative[];
    primary_movement_region: string;
  };
  schemas: Record<string, SourceSchema>;
  metrics: {
    client: { id: string; displayName: string; shortName: string };
    scopes: { platform: Scope; sponsor: Scope; sponsored: SponsoredScope };
    currentPeriod: { period: string; reportingDate: string; label: string };
    priorPeriod: { period: string; reportingDate: string; label: string };
    lenses: Record<
      string,
      { label: string; summary: SummaryBlock; movement: MovementBlock; series: SeriesPoint[] }
    >;
  };
  dashboard: {
    snapshots: { portfolios: { client_id: string; label: string; runs: { run_id: string; reporting_date: string }[] }[] };
    sourcePortfolios: { lenses: { id: string; kind: string; label: string }[] };
    fundedSnapshot: { kpis: Kpi[]; loan_count: number; current_outstanding_balance: number };
    geoExposure: { areas: { itl3_name?: string; balance?: number; share?: number }[]; areaCount: number };
    dataSource: { kind: string; name: string };
  };
  miAgent: { questions: string[]; turns: ChatTurn[] };
  copilot: {
    agentName: string;
    actions: string[];
    answers: { question?: string; answer: string; reportingDate?: string | null }[];
    artefactAction: {
      fileName: string;
      reportingPeriod: string;
      downloadUrl: string;
      expiresInSeconds: number;
    } | null;
    artefactQuestion: string;
  };
};

export const SYNTHETIC_NOTICE = F.synthetic_notice;
export const CLIENT = F.metrics.client;
export const PORTFOLIOS = F.narrative.portfolios;
export const PERIODS = F.narrative.periods;
export const SCHEMAS = F.schemas;
export const CURRENT_PERIOD = F.metrics.currentPeriod;
export const PRIOR_PERIOD = F.metrics.priorPeriod;
export const PRIMARY_REGION = F.narrative.primary_movement_region;

/** PLATFORM scope — the default, and the scope of every figure but one. */
export const PLATFORM_SCOPE = F.metrics.scopes.platform;

/** SPONSOR scope — S3's consolidation beat only. Never propagate this. */
export const SPONSOR_SCOPE = F.metrics.scopes.sponsor;

/** The sold securitisation's own governed figures. */
export const SPONSORED = F.metrics.scopes.sponsored;

export const TOTAL = F.metrics.lenses.total;
export const SUMMARY = TOTAL.summary;
export const MOVEMENT = TOTAL.movement;
export const SERIES = TOTAL.series;

export const DASHBOARD = F.dashboard;
export const MI_TURNS = F.miAgent.turns;
export const COPILOT = F.copilot;

export const ARTEFACTS = artefactFixture as unknown as Record<string, ArtefactEntry>;
export const ASSERTIONS = assertionFixture as unknown as {
  ok: boolean;
  checksRun: number;
  checksPassed: number;
  metrics: Record<string, unknown>;
};
export const SAFETY = safetyFixture as unknown as {
  ok: boolean;
  filesScanned: number;
  findingCount: number;
};

// --------------------------------------------------------------------------- //
// Onboarding and orchestration (from the demo manifest)
// --------------------------------------------------------------------------- //
export interface MappingDecision {
  source_header: string;
  canonical_field: string | null;
  tier: string;
  tier_label: string;
  confidence: number;
  low_confidence: boolean;
  resolved_by_client_contract: boolean;
  note: string;
}

export interface OnboardingMapping {
  mapped_count: number;
  unmapped_count: number;
  unmapped_headers: string[];
  low_confidence_count: number;
  client_contract_count: number;
}

export interface OrchestrationRun {
  portfolio: string;
  source_portfolio_id: string;
  period: string;
  reporting_date: string;
  rows: number;
  total_outstanding_balance: number;
  elapsed_seconds: number;
  gate_summary: string[];
}

const M = manifestFixture as unknown as {
  onboarding: Record<
    string,
    { mapping: OnboardingMapping; mappingDecisions: MappingDecision[] }
  >;
  orchestration: { runs: OrchestrationRun[] };
};

/** One portfolio's Gate 1 mapping result, by storyboard key. */
export const onboarding = (key: string) => {
  const p = portfolio(key);
  const found = M.onboarding[p.source_portfolio_id];
  if (!found) throw new Error(`No onboarding record for ${p.source_portfolio_id}`);
  return found;
};

/** The recorded mapping decision for one source header, by storyboard key. */
export const mappingDecision = (key: string, header: string): MappingDecision => {
  const found = onboarding(key).mappingDecisions.find((d) => d.source_header === header);
  if (!found) throw new Error(`No mapping decision for ${key}:${header}`);
  return found;
};

export const ORCHESTRATION_RUNS = M.orchestration.runs;

/** The recurring run for one portfolio in the current reporting period. */
export const currentRun = (key: string): OrchestrationRun => {
  const p = portfolio(key);
  const found = ORCHESTRATION_RUNS.find(
    (r) => r.source_portfolio_id === p.source_portfolio_id && r.period === CURRENT_PERIOD.period,
  );
  if (!found) throw new Error(`No current-period run for ${p.source_portfolio_id}`);
  return found;
};

/**
 * Measured wall-clock for the current period across both portfolios: source
 * extract received to governed canonical published. This is the only elapsed time
 * the demonstration measures, so it is the only one the film may state.
 */
export const currentPeriodElapsedSeconds = (): number =>
  ORCHESTRATION_RUNS.filter((r) => r.period === CURRENT_PERIOD.period).reduce(
    (total, r) => total + r.elapsed_seconds,
    0,
  );

/**
 * The three lanes S3's consolidation beat draws, in the order it draws them: the deal
 * that was sold first, because it is the one the sponsor's own ledger has lost.
 *
 * Every figure is read from the fixtures — the two warehoused books from their own MI
 * lenses, SPV1 from its own governed canonical.
 */
export interface PortfolioLane {
  displayId: string;
  status: string;
  statusDetail: string;
  fundedBalance: number;
  loanCount: number;
}

export const portfolioLanes = (): PortfolioLane[] => {
  const narrative = (id: string) => {
    const found = PORTFOLIOS.find((p) => p.display_id === id);
    if (!found) throw new Error(`No narrative record for ${id}`);
    return found;
  };
  const sponsored = narrative(SPONSORED.displayId);
  const lanes: PortfolioLane[] = [
    {
      displayId: SPONSORED.displayId,
      status: "Sponsored · sold",
      statusDetail: sponsored.status_detail,
      fundedBalance: SPONSORED.fundedBalance,
      loanCount: SPONSORED.loanCount,
    },
  ];
  for (const key of ["A", "B"]) {
    const p = portfolio(key);
    const metrics = lens(key).summary.metrics;
    lanes.push({
      displayId: p.display_id,
      status: p.source_portfolio_type === "acquired" ? "Acquired back book" : "Direct origination",
      statusDetail: p.status_detail,
      fundedBalance: metrics.funded_balance ?? 0,
      loanCount: metrics.loan_count ?? 0,
    });
  }
  return lanes;
};

/**
 * What arrives, before anything is done to it.
 *
 * Five artefact types, five owners, three reporting cycles — which is the whole point of
 * the beat, and why it needs no caption. Four rows are read from the demonstration
 * itself: the three source schemas the generator writes, plus the risk limit schedule
 * the concentration monitor reads. The fifth is stated copy (see `src/claims.ts`).
 */
export interface Arrival {
  title: string;
  format: string;
  owner: string;
  frequency: string;
}

const ARRIVAL_META: Record<string, { title: string; owner: string; frequency: string }> = {
  origination_extract: {
    title: "Origination extract",
    owner: "internal system",
    frequency: "monthly",
  },
  servicer_account_extract: {
    title: "Servicer account extract",
    owner: "third-party servicer",
    frequency: "monthly",
  },
  trustee_deal_extract: {
    title: "SPV1 trustee asset schedule",
    owner: "deal trustee",
    frequency: "quarterly",
  },
};

export const ARRIVALS: Arrival[] = [
  ...Object.entries(ARRIVAL_META).map(([schemaName, meta]) => {
    if (!SCHEMAS[schemaName]) {
      throw new Error(`No source schema ${schemaName} in the fixtures`);
    }
    return { ...meta, format: "CSV" };
  }),
  {
    title: "Risk limit schedule",
    format: "PDF",
    owner: String(
      (artefactFixture as unknown as Record<string, ArtefactEntry>).riskMonitor?.limitsSource ??
        "facility agent",
    ).replace(/ document$/, ""),
    frequency: "quarterly",
  },
  STATED_ARRIVAL,
];

/** One portfolio's narrative record, by storyboard key ("A" | "B"). */
export const portfolio = (key: string): PortfolioNarrative => {
  const found = PORTFOLIOS.find((p) => p.key === key);
  if (!found) throw new Error(`No demo portfolio with key ${key}`);
  return found;
};

/** One portfolio's source schema. */
export const schemaFor = (key: string): SourceSchema => {
  const p = portfolio(key);
  const schema = SCHEMAS[p.schema_name];
  if (!schema) throw new Error(`No source schema ${p.schema_name}`);
  return schema;
};

/** The per-portfolio lens block, by storyboard key. */
export const lens = (key: string) => {
  const p = portfolio(key);
  const found = F.metrics.lenses[p.source_portfolio_id];
  if (!found) throw new Error(`No lens metrics for ${p.source_portfolio_id}`);
  return found;
};

/** The movement contribution for one portfolio. */
export const contribution = (key: string): CohortMovement => {
  const p = portfolio(key);
  const found = MOVEMENT.cohortMovements.find((c) => c.id === p.source_portfolio_id);
  if (!found) throw new Error(`No movement for ${p.source_portfolio_id}`);
  return found;
};

/** A named MI Agent conversation turn (0-indexed). */
export const miTurn = (index: number): ChatTurn => {
  const turn = MI_TURNS[index];
  if (!turn) throw new Error(`No MI Agent turn ${index}`);
  return turn;
};

/** The artifact of a given type from a turn, if present. */
export const turnArtifact = (
  turn: ChatTurn,
  type: string,
  titleContains?: string,
): ChatArtifact | undefined =>
  turn.artifacts.find(
    (a) =>
      a.type === type &&
      (!titleContains ||
        (a.title ?? "").toLowerCase().includes(titleContains.toLowerCase())),
  );

/** The Copilot answer for a question index. */
export const copilotAnswer = (index: number): string => {
  const a = COPILOT.answers[index];
  if (!a) throw new Error(`No Copilot answer ${index}`);
  return a.answer;
};
