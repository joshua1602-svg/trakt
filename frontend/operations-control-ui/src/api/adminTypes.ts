/**
 * Payload types for the administrator configuration area
 * (`/ops/me` and `/ops/admin/config/…`).
 *
 * These mirror the backend read models exactly. The browser renders what the
 * backend says and never derives configuration meaning of its own.
 */

export type ConfigLayer = "system" | "regime" | "asset";

export type PackageStatus = "draft" | "active" | "superseded";

export interface Principal {
  name: string;
  role: string;
  is_admin: boolean;
  all_clients: boolean;
  clients: string[];
}

export interface ValidationResult {
  ok: boolean;
  problems: string[];
  validated_at?: string;
}

export interface ValidationProblem {
  summary: string;
  file: string;
  technical: string;
}

export interface Dependency {
  name: string;
  relationship: string;
}

export interface ValidationSummary {
  state: "passed" | "failed" | "not_checked";
  ok: boolean;
  headline: string;
  detail: string;
  problems: ValidationProblem[];
  warnings: string[];
  dependencies: Dependency[];
  checked_at: string | null;
}

export interface ChangedFile {
  path: string;
  label: string;
}

export interface PackageFile {
  path: string;
  label: string;
  sha256: string;
  size: number;
}

export interface VersionRow {
  layer: ConfigLayer;
  version: number;
  status: PackageStatus;
  notes: string;
  created_by: string;
  created_at: string;
  validated: boolean;
  activated_by: string | null;
  activated_at: string | null;
}

export interface DraftSummary {
  version: number;
  based_on_version: number | null;
  created_by: string;
  created_at: string;
  notes: string;
  validated: boolean;
  validation: ValidationResult | null;
  validation_summary: ValidationSummary;
  changed_files: ChangedFile[];
  package_hash: string;
  file_count: number;
}

export interface LayerOverview {
  layer: ConfigLayer;
  label: string;
  description: string;
  active_version: number;
  activated_at: string | null;
  activated_by: string | null;
  status: PackageStatus;
  validated: boolean;
  validation: ValidationResult | null;
  validation_summary: ValidationSummary;
  package_hash: string;
  file_count: number;
  files: PackageFile[];
  draft: DraftSummary | null;
  versions: VersionRow[];
  dependencies: Dependency[];
}

export interface RegimeReference {
  id: string;
  label: string;
}

export interface AssetPackage {
  id: string;
  label: string;
  layer: ConfigLayer;
  active_version: number;
  status: PackageStatus;
  validated: boolean;
  draft_version: number | null;
  supported_regimes: RegimeReference[];
  source_semantics: {
    profile_count: number;
    capabilities: string[];
    profile_version: string;
  };
  mapping_defaults: { setting_count: number; source: string };
  taxonomies: string[];
  issue_policy: { rule_count: number };
  dependencies: Dependency[];
  last_activation: { at: string | null; by: string | null };
}

export interface RegimePackage {
  id: string;
  label: string;
  annex: string;
  regulator: string;
  description: string;
  layer: ConfigLayer;
  active_version: number;
  package_version: number | string | null;
  status: PackageStatus;
  validated: boolean;
  draft_version: number | null;
  schema: {
    file?: string;
    namespace?: string;
    is_draft?: boolean;
    state?: string;
  };
  field_universe: { count: number; source: string };
  code_order: { version?: string; entry_count?: number; source?: string };
  validation_policy: {
    unknown_values: string;
    apply_defaults_only_if_explicit: boolean;
    deferred_fields: string[];
    rule_count: number;
  };
  no_data_policy: { codes_allowed: string[]; fields_allowing_no_data: number };
  compatible_assets: RegimeReference[];
  dependencies: Dependency[];
  last_activation: { at: string | null; by: string | null };
}

export interface CompatibilityCell {
  regime: string;
  regime_label: string;
  supported: boolean;
  label: string;
}

export interface CompatibilityRow {
  asset: string;
  asset_label: string;
  cells: CompatibilityCell[];
}

export interface CompatibilityMatrix {
  regimes: RegimeReference[];
  rows: CompatibilityRow[];
}

export interface CompatibilityIssue {
  layer: ConfigLayer;
  asset?: string;
  regime?: string;
  message: string;
}

export interface AuditEntry {
  seq: number;
  event: string;
  event_label: string;
  actor: string;
  at: string;
  layer: string;
  layer_label: string;
  version: number | null;
  notes: string;
  outcome: "ok" | "failed";
  description: string;
}

export interface AuditTrail {
  entries: AuditEntry[];
  chain_intact: boolean;
}

export interface ConfigOverview {
  layers: Record<ConfigLayer, LayerOverview>;
  assets: AssetPackage[];
  regimes: RegimePackage[];
  compatibility: CompatibilityMatrix;
  attention: {
    drafts_awaiting_action: {
      layer: ConfigLayer;
      label: string;
      version: number;
      validated: boolean;
      created_by: string;
      created_at: string;
    }[];
    validation_failures: {
      layer: ConfigLayer;
      label: string;
      version: number;
      problems: string[];
    }[];
    packages_requiring_review: {
      layer: ConfigLayer;
      label: string;
      version: number;
    }[];
    compatibility_issues: CompatibilityIssue[];
  };
  recent_activations: AuditEntry[];
  recent_rollbacks: AuditEntry[];
}

export interface ConfigCatalogue {
  assets: AssetPackage[];
  regimes: RegimePackage[];
  compatibility: CompatibilityMatrix;
  issues: CompatibilityIssue[];
}

export interface ActivationBlocker {
  kind: string;
  message: string;
}

export interface ConfigVersion {
  layer: ConfigLayer;
  version: number;
  status: PackageStatus;
  notes: string;
  based_on_version?: number | null;
  created_by: string;
  created_at: string;
  validated: boolean;
  validation: ValidationResult | null;
  validation_summary: ValidationSummary;
  activated_by: string | null;
  activated_at: string | null;
  package_hash: string;
  dependencies: Dependency[];
  activation_blockers: ActivationBlocker[];
  file_list: PackageFile[];
  changed_files?: ChangedFile[];
  file_content?: string;
  file_label?: string;
}

export type DiffLineKind = "added" | "removed" | "context" | "hunk";

export interface DiffLine {
  kind: DiffLineKind;
  text: string;
}

export type FileChange = "added" | "changed" | "removed" | "unchanged";

export interface ComparisonFile {
  path: string;
  label: string;
  change: FileChange;
  from_hash: string;
  to_hash: string;
  lines: DiffLine[];
  truncated?: boolean;
}

export interface VersionRef {
  version: number;
  status: PackageStatus;
  created_by: string;
  created_at: string;
  activated_at: string | null;
  activated_by: string | null;
  notes: string;
  package_hash: string;
}

export interface Comparison {
  layer: ConfigLayer;
  from: VersionRef;
  to: VersionRef;
  summary: { added: number; changed: number; removed: number; unchanged: number };
  files: ComparisonFile[];
  dependencies: { shared: Dependency[]; note: string };
  validation: { from: ValidationResult | null; to: ValidationResult | null };
}

export interface PinnedWorkflow {
  workflow_id: string;
  client_id: string;
  portfolio_id: string;
  status: string;
  reporting_period: string;
  pinned_version: string;
  still_running: boolean;
}

export interface ImpactAnalysis {
  layer: ConfigLayer;
  version: number;
  clients: number;
  portfolios: number;
  pinned_workflows: number;
  running_pinned_workflows: number;
  future_workflows_affected: boolean;
  workflows: PinnedWorkflow[];
  explanation: string[];
}

export interface CreateDraftInput {
  from_version?: number;
  edits?: Record<string, string>;
  notes?: string;
}
