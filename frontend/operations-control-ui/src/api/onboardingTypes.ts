/**
 * Client Onboarding — the governed standing-configuration capability.
 *
 * These types mirror `operations_control/onboarding/`. Nothing here is a
 * decision: every option list arrives from the backend (`vocabularies`), and
 * every validation message is the backend's wording. React renders; it does not
 * decide what is valid, what applies, or what will be written.
 */

export type OnboardingStep =
  | "client"
  | "portfolio"
  | "reporting"
  | "regime"
  | "static_reporting"
  | "sources"
  | "review";

export interface ClientStanding {
  client_id: string;
  client_name: string;
  legal_entity_name: string;
  lei: string;
  jurisdiction: string;
  reporting_currency: string;
  time_zone: string;
  environment: string;
  primary_reporting_contact: string;
  reporting_email: string;
  operational_contact: string;
  operational_email: string;
}

export interface PortfolioStanding {
  portfolio_id: string;
  display_name: string;
  portfolio_type: string;
  asset_class: string;
  structure: string;
  funded_cadence: string;
  pipeline_cadence: string;
  originates: boolean | null;
  warehouse_lender: string;
  source_system: string;
}

export interface ReportingStanding {
  products: string[];
  derivation: Record<string, string>;
}

export interface StaticReportingStanding {
  app_title: string;
  primary_color: string;
  logo_uri: string;
  disclaimer: string;
  investor_contact: string;
  investor_email: string;
  trustee_name: string;
  warehouse_lender: string;
  payment_convention: string;
  day_count_convention: string;
  reporting_convention: string;
}

export interface SourceRegistrationStanding {
  portfolio_id: string;
  dataset: string;
  frequency: string;
  source_system: string;
  expected_files: string[];
  regime_required: boolean;
  status: string;
  blob_prefix: string;
}

export interface OnboardingProfile {
  client_id: string;
  client: ClientStanding;
  portfolios: PortfolioStanding[];
  reporting: ReportingStanding;
  regime: Record<string, Record<string, string | boolean>>;
  static_reporting: StaticReportingStanding;
  sources: SourceRegistrationStanding[];
}

/** One field of one reporting product, as declared by the governed spec. */
export interface StandingField {
  key: string;
  label: string;
  regime_code?: string;
  classification: string;
  required?: boolean;
  format: string;
  help?: string;
  max_length?: number;
  options?: (string | { value: string; label: string })[];
  writes_to?: string;
}

export interface UnrepresentedField {
  key: string;
  label: string;
  reason: string;
  product?: string;
}

export interface StandingProduct {
  label: string;
  regime_id?: string;
  always_applies?: boolean;
  derived_from?: string;
  requires_dataset?: string;
  supported_by_assets_declaring?: string;
  fields: StandingField[];
  unrepresented?: UnrepresentedField[];
}

export interface StandingFieldSpec {
  version: number;
  products: Record<string, StandingProduct>;
}

export interface AssetChoice {
  value: string;
  label: string;
  supports_regimes: string[];
}

export interface ProductChoice {
  key: string;
  label: string;
  selectable: boolean;
  always_applies: boolean;
  derived_from: string;
  requires_dataset: string;
  supported_by_assets_declaring: string;
  field_count: number;
  unrepresented: UnrepresentedField[];
}

export interface Vocabularies {
  asset_classes: AssetChoice[];
  portfolio_types: string[];
  portfolio_structures: string[];
  datasets: string[];
  regime_capable_datasets: string[];
  cadences: string[];
  registration_statuses: string[];
  products: ProductChoice[];
  sources: Record<string, string>;
}

export interface OnboardingReference {
  vocabularies: Vocabularies;
  standing_fields: StandingFieldSpec;
}

export interface OnboardingProblem {
  step: string;
  field: string;
  message: string;
}

export interface DerivedProduct {
  applies: boolean;
  eligible: boolean;
  derived: boolean;
  reason: string;
}

export interface AdoptionGap {
  step: string;
  field: string;
  label: string;
  why: string;
}

export interface OnboardingDraft {
  draft_id: string;
  client_id: string;
  /** new | adopted | current */
  origin: string;
  based_on_version: number | null;
  step: OnboardingStep;
  steps: { key: OnboardingStep; label: string; problems: number }[];
  profile: OnboardingProfile;
  derived_reporting: Record<string, DerivedProduct>;
  derived_sources: SourceRegistrationStanding[];
  /** Where an adopted value came from, in operator wording. */
  provenance: Record<string, string>;
  gaps: AdoptionGap[];
  sources_read: string[];
  problems: OnboardingProblem[];
  problems_by_step: Record<string, OnboardingProblem[]>;
  ready: boolean;
  created_by: string;
  created_at: string;
}

export interface PlannedSourceRecord extends SourceRegistrationStanding {
  action: string;
  expected_location: string;
  portfolio_label: string;
  client_id: string;
  source_portfolio_id: string;
}

export interface PlannedArtefact {
  kind: string;
  rel: string;
  label: string;
  /** create | update | unchanged */
  action: string;
  text: string;
  previous_text: string;
  records: PlannedSourceRecord[];
  summary: string[];
}

export interface ProfileChange {
  path: string;
  label: string;
  before: string;
  after: string;
  action: string;
}

export interface OnboardingReview {
  draft_id: string;
  client_id: string;
  profile: OnboardingProfile;
  artefacts: PlannedArtefact[];
  changes: ProfileChange[];
  problems: OnboardingProblem[];
  ready: boolean;
  current_version: number;
  next_version: number;
  unrepresented: UnrepresentedField[];
}

export interface OnboardingClientRow {
  client_id: string;
  /** onboarded | legacy */
  status: string;
  display_name: string;
  version: number;
  portfolios: number;
  products: string[];
  updated_at: string;
  updated_by: string;
  open_drafts: number;
  summary: string;
}

export interface WrittenArtefact {
  kind: string;
  target: string;
  action: string;
  detail: Record<string, unknown>;
}

export interface ProfileVersionRow {
  version: number;
  status: string;
  approved_by: string;
  approved_at: string;
  reason: string;
  based_on_version: number | null;
  change_count: number;
  changes: ProfileChange[];
  artefacts: WrittenArtefact[];
  content_hash: string;
}

export interface OnboardingClientDetail {
  client_id: string;
  status: string;
  version: number;
  approved_by?: string;
  approved_at?: string;
  reason?: string;
  content_hash?: string;
  profile: OnboardingProfile | null;
  derived_reporting?: Record<string, DerivedProduct>;
  live_source_registrations: Record<string, unknown>[];
  artefacts: WrittenArtefact[];
  unrepresented?: UnrepresentedField[];
  history: ProfileVersionRow[];
  summary?: string;
}

export interface ApprovalResult {
  version: number;
  client_id: string;
  artefacts: WrittenArtefact[];
  changes: ProfileChange[];
}
