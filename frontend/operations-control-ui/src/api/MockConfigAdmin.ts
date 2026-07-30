import { OpsError } from "./OpsClient";
import type {
  AuditEntry,
  AuditTrail,
  Comparison,
  ComparisonFile,
  ConfigCatalogue,
  ConfigLayer,
  ConfigOverview,
  ConfigVersion,
  CreateDraftInput,
  DiffLine,
  ImpactAnalysis,
  LayerOverview,
  PackageFile,
  ValidationResult,
  ValidationSummary,
  VersionRow,
} from "./adminTypes";

/**
 * A stateful stand-in for the administrator configuration API, mirroring the
 * backend's governed lifecycle: draft -> validate -> activate, and rollback to
 * a prior version. Used by `VITE_OPS_MODE=mock` and by the tests.
 *
 * It enforces the same refusals the backend enforces, so a test that activates
 * an unvalidated draft fails here for the same reason it would fail for real.
 */

interface StoredVersion {
  layer: ConfigLayer;
  version: number;
  status: "draft" | "active" | "superseded";
  notes: string;
  based_on_version: number | null;
  created_by: string;
  created_at: string;
  validated: boolean;
  validation: ValidationResult | null;
  activated_by: string | null;
  activated_at: string | null;
  files: Record<string, string>;
}

const SEED_FILES: Record<ConfigLayer, Record<string, string>> = {
  system: {
    "config/system/fields_registry.yaml": "fields:\n  loan_identifier:\n    type: text\n",
    "config/system/enum_mapping.yaml": "repayment_type:\n  interest_only: Interest only\n",
    "config/system/run_context_policy.yaml": "version: 1\nallow_partial: false\n",
  },
  regime: {
    "config/regime/annex2_field_universe.yaml":
      "regime: ESMA_Annex2\nfield_count: 107\nfields:\n  RREL1:\n    field_name: Unique Identifier\n",
    "config/regime/annex2_delivery_rules.yaml":
      "regime: ESMA_Annex2\nversion: 2\ndefault_policy:\n  unknown_values: reject\n",
  },
  asset: {
    "config/asset/product_profiles.yaml": "version: 2\ncapabilities:\n  - base_mi\n",
    "config/asset/product_defaults_ERM.yaml": "amortisation_type: none\n",
    "config/asset/issue_policy.yaml": "on_missing_value: review\n",
  },
};

const FILE_LABELS: Record<string, string> = {
  "config/system/fields_registry.yaml": "Field registry",
  "config/system/enum_mapping.yaml": "Accepted values",
  "config/system/run_context_policy.yaml": "Run policy",
  "config/regime/annex2_field_universe.yaml": "Annex 2 field list",
  "config/regime/annex2_delivery_rules.yaml": "Annex 2 delivery rules",
  "config/asset/product_profiles.yaml": "Product profiles",
  "config/asset/product_defaults_ERM.yaml": "Equity release defaults",
  "config/asset/issue_policy.yaml": "Issue handling policy",
};

const LAYER_LABELS: Record<ConfigLayer, string> = {
  system: "System package",
  regime: "Regime packages",
  asset: "Asset packages",
};

const LAYER_DESCRIPTIONS: Record<ConfigLayer, string> = {
  system: "The platform-wide field registry, wording, and run policy every client shares.",
  regime: "The regulatory reporting packages, one per annex.",
  asset: "The product packages that describe how each asset type behaves.",
};

const DEPENDENCIES: Record<ConfigLayer, { name: string; relationship: string }[]> = {
  system: [
    {
      name: "Client configuration",
      relationship:
        "Client and portfolio settings are layered on top of this package during onboarding.",
    },
  ],
  regime: [
    {
      name: "System package",
      relationship: "Regime reporting reads its field names and code order from the system package.",
    },
  ],
  asset: [
    { name: "System package", relationship: "Asset defaults are expressed in system field names." },
  ],
};

function label(path: string): string {
  return FILE_LABELS[path] ?? path.split("/").pop() ?? path;
}

function hash(text: string): string {
  let h = 0;
  for (let i = 0; i < text.length; i += 1) {
    h = (h * 31 + text.charCodeAt(i)) | 0;
  }
  return `sha256:${(h >>> 0).toString(16).padStart(8, "0")}${"0".repeat(56)}`;
}

function explain(layer: ConfigLayer, validation: ValidationResult | null): ValidationSummary {
  const dependencies = DEPENDENCIES[layer];
  if (validation === null) {
    return {
      state: "not_checked",
      ok: false,
      headline: "Not checked yet",
      detail: "This version has not been checked. Check it before activating.",
      problems: [],
      warnings: [],
      dependencies,
      checked_at: null,
    };
  }
  if (validation.ok) {
    return {
      state: "passed",
      ok: true,
      headline: "Checks passed",
      detail: "Every check passed. This version can be activated.",
      problems: [],
      warnings: [],
      dependencies,
      checked_at: validation.validated_at ?? null,
    };
  }
  const count = validation.problems.length;
  return {
    state: "failed",
    ok: false,
    headline: "Validation failed",
    detail: `This draft cannot be activated because ${count} configuration ${
      count === 1 ? "check" : "checks"
    } failed.`,
    problems: validation.problems.map((problem) => {
      const path = problem.split(": ")[0];
      const name = path.includes("/") ? label(path) : path;
      return {
        summary: `${name} could not be read. Its formatting is broken.`,
        file: name,
        technical: problem,
      };
    }),
    warnings: [],
    dependencies,
    checked_at: validation.validated_at ?? null,
  };
}

export class MockConfigAdmin {
  private versions: StoredVersion[] = [];
  private audit: AuditEntry[] = [];
  private seq = 0;
  private actor = "Administrator";

  constructor() {
    for (const layer of ["system", "regime", "asset"] as ConfigLayer[]) {
      this.versions.push({
        layer,
        version: 1,
        status: "active",
        notes: "seeded from the platform's own configuration",
        based_on_version: null,
        created_by: "system",
        created_at: "2026-01-12T09:00:00Z",
        validated: true,
        validation: { ok: true, problems: [], validated_at: "2026-01-12T09:00:00Z" },
        activated_by: "Administrator",
        activated_at: "2026-01-12T09:00:00Z",
        files: { ...SEED_FILES[layer] },
      });
    }
  }

  private of(layer: ConfigLayer): StoredVersion[] {
    return this.versions.filter((v) => v.layer === layer).sort((a, b) => a.version - b.version);
  }

  private active(layer: ConfigLayer): StoredVersion {
    const found = this.of(layer).find((v) => v.status === "active");
    if (!found) throw new OpsError("No active version could be found.");
    return found;
  }

  private find(layer: ConfigLayer, version: number): StoredVersion {
    const found = this.of(layer).find((v) => v.version === version);
    if (!found) throw new OpsError("That could not be found.", "OPS_NOT_FOUND");
    return found;
  }

  private draftOf(layer: ConfigLayer): StoredVersion | undefined {
    return [...this.of(layer)].reverse().find((v) => v.status === "draft");
  }

  private record(entry: Omit<AuditEntry, "seq">): void {
    this.seq += 1;
    this.audit.unshift({ seq: this.seq, ...entry });
  }

  private row(v: StoredVersion): VersionRow {
    return {
      layer: v.layer,
      version: v.version,
      status: v.status,
      notes: v.notes,
      created_by: v.created_by,
      created_at: v.created_at,
      validated: v.validated,
      activated_by: v.activated_by,
      activated_at: v.activated_at,
    };
  }

  private fileList(v: StoredVersion): PackageFile[] {
    return Object.entries(v.files)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([path, content]) => ({
        path,
        label: label(path),
        sha256: hash(content),
        size: content.length,
      }));
  }

  private packageHash(v: StoredVersion): string {
    return hash(
      Object.entries(v.files)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([path, content]) => `${path}:${hash(content)}`)
        .join("\n"),
    );
  }

  private changedFiles(v: StoredVersion): { path: string; label: string }[] {
    if (v.based_on_version === null) return [];
    const base = this.of(v.layer).find((x) => x.version === v.based_on_version);
    if (!base) return [];
    const paths = new Set([...Object.keys(base.files), ...Object.keys(v.files)]);
    return [...paths]
      .filter((p) => base.files[p] !== v.files[p])
      .sort()
      .map((path) => ({ path, label: label(path) }));
  }

  private layerOverview(layer: ConfigLayer): LayerOverview {
    const active = this.active(layer);
    const draft = this.draftOf(layer);
    return {
      layer,
      label: LAYER_LABELS[layer],
      description: LAYER_DESCRIPTIONS[layer],
      active_version: active.version,
      activated_at: active.activated_at,
      activated_by: active.activated_by,
      status: active.status,
      validated: active.validated,
      validation: active.validation,
      validation_summary: explain(layer, active.validation),
      package_hash: this.packageHash(active),
      file_count: Object.keys(active.files).length,
      files: this.fileList(active),
      draft: draft
        ? {
            version: draft.version,
            based_on_version: draft.based_on_version,
            created_by: draft.created_by,
            created_at: draft.created_at,
            notes: draft.notes,
            validated: draft.validated,
            validation: draft.validation,
            validation_summary: explain(layer, draft.validation),
            changed_files: this.changedFiles(draft),
            package_hash: this.packageHash(draft),
            file_count: Object.keys(draft.files).length,
          }
        : null,
      versions: this.of(layer).map((v) => this.row(v)),
      dependencies: DEPENDENCIES[layer],
    };
  }

  catalogue(): ConfigCatalogue {
    const assetActive = this.active("asset");
    const regimeActive = this.active("regime");
    const assetDraft = this.draftOf("asset");
    const regimeDraft = this.draftOf("regime");
    const assets: ConfigCatalogue["assets"] = [
      {
        id: "equity_release",
        label: "Equity Release",
        readiness: {
          state: "attention",
          label: "Usable, with things to know",
          statement:
            `Equity Release v${assetActive.version} is active and valid for MI reporting ` +
            "and ESMA Annex 2. There is something to know before you rely on it.",
          can_process: true,
          warnings: ["ESMA Annex 2 uses a draft schema published by the regulator."],
        },
        layer: "asset",
        active_version: assetActive.version,
        status: assetActive.status,
        validated: assetActive.validated,
        draft_version: assetDraft?.version ?? null,
        supported_regimes: [{ id: "ESMA_Annex2", label: "ESMA Annex 2" }],
        source_semantics: { profile_count: 3, capabilities: ["base_mi"], profile_version: "2" },
        mapping_defaults: { setting_count: 18, source: "config/asset/product_defaults_ERM.yaml" },
        taxonomies: ["repayment", "valuation"],
        issue_policy: { rule_count: 6 },
        dependencies: DEPENDENCIES.asset,
        last_activation: { at: assetActive.activated_at, by: assetActive.activated_by },
      },
    ];
    const regimes: ConfigCatalogue["regimes"] = [
      {
        id: "ESMA_Annex2",
        label: "ESMA Annex 2",
        annex: "Annex 2",
        regulator: "ESMA",
        description: "Underlying exposure reporting for non-ABCP securitisations.",
        layer: "regime",
        active_version: regimeActive.version,
        package_version: 2,
        status: regimeActive.status,
        validated: regimeActive.validated,
        draft_version: regimeDraft?.version ?? null,
        schema: {
          file: "DRAFT1auth.099.001.04_1.3.0.xsd",
          namespace: "urn:esma:xsd:DRAFT1auth.099.001.04",
          is_draft: true,
          state: "present_but_draft",
        },
        field_universe: { count: 107, source: "annex2_template_workbook.xlsx" },
        code_order: { version: "1.0", entry_count: 107, source: "config/system/esma_code_order.yaml" },
        validation_policy: {
          unknown_values: "reject",
          apply_defaults_only_if_explicit: true,
          deferred_fields: ["RREC22"],
          rule_count: 96,
        },
        no_data_policy: { codes_allowed: ["ND5"], fields_allowing_no_data: 41 },
        compatible_assets: [{ id: "equity_release", label: "Equity Release" }],
        dependencies: DEPENDENCIES.regime,
        last_activation: { at: regimeActive.activated_at, by: regimeActive.activated_by },
      },
    ];
    const compatibility = {
      regimes: regimes.map((r) => ({ id: r.id, label: r.label })),
      rows: assets.map((a) => ({
        asset: a.id,
        asset_label: a.label,
        cells: regimes.map((r) => {
          const supported = a.supported_regimes.some((s) => s.id === r.id);
          return {
            regime: r.id,
            regime_label: r.label,
            supported,
            label: supported ? "Supported" : "Not supported",
          };
        }),
      })),
    };
    const issues = regimes
      .filter((r) => r.schema.is_draft)
      .map((r) => ({
        layer: "regime" as ConfigLayer,
        regime: r.id,
        message: `${r.label} uses a draft regulatory schema. Confirm the published schema before relying on this for a real submission.`,
      }));
    return { assets, regimes, compatibility, issues };
  }

  overview(): ConfigOverview {
    const layers = {
      system: this.layerOverview("system"),
      regime: this.layerOverview("regime"),
      asset: this.layerOverview("asset"),
    };
    const cat = this.catalogue();
    const drafts = (Object.values(layers) as LayerOverview[])
      .filter((l) => l.draft)
      .map((l) => ({
        layer: l.layer,
        label: l.label,
        version: l.draft!.version,
        validated: l.draft!.validated,
        created_by: l.draft!.created_by,
        created_at: l.draft!.created_at,
      }));
    return {
      layers,
      assets: cat.assets,
      regimes: cat.regimes,
      compatibility: cat.compatibility,
      attention: {
        drafts_awaiting_action: drafts,
        validation_failures: (Object.values(layers) as LayerOverview[])
          .filter((l) => l.draft?.validation && !l.draft.validation.ok)
          .map((l) => ({
            layer: l.layer,
            label: l.label,
            version: l.draft!.version,
            problems: l.draft!.validation!.problems,
          })),
        packages_requiring_review: (Object.values(layers) as LayerOverview[])
          .filter((l) => l.draft && l.draft.validation === null)
          .map((l) => ({ layer: l.layer, label: l.label, version: l.draft!.version })),
        compatibility_issues: cat.issues,
      },
      recent_activations: this.audit.filter((a) => a.event === "config_activated").slice(0, 5),
      recent_rollbacks: this.audit.filter((a) => a.event === "config_rolled_back").slice(0, 5),
    };
  }

  version(layer: ConfigLayer, version: number, file?: string): ConfigVersion {
    const v = this.find(layer, version);
    return {
      layer,
      version: v.version,
      status: v.status,
      notes: v.notes,
      based_on_version: v.based_on_version,
      created_by: v.created_by,
      created_at: v.created_at,
      validated: v.validated,
      validation: v.validation,
      validation_summary: explain(layer, v.validation),
      activated_by: v.activated_by,
      activated_at: v.activated_at,
      package_hash: this.packageHash(v),
      dependencies: DEPENDENCIES[layer],
      activation_blockers: [],
      file_list: this.fileList(v),
      changed_files: this.changedFiles(v),
      ...(file && v.files[file] !== undefined
        ? { file_content: v.files[file], file_label: label(file) }
        : {}),
    };
  }

  compare(layer: ConfigLayer, from: number, to: number): Comparison {
    const left = this.find(layer, from);
    const right = this.find(layer, to);
    const paths = [...new Set([...Object.keys(left.files), ...Object.keys(right.files)])].sort();
    let added = 0;
    let changed = 0;
    let removed = 0;
    const files: ComparisonFile[] = paths.map((path) => {
      const before = left.files[path];
      const after = right.files[path];
      let change: ComparisonFile["change"] = "unchanged";
      if (before === undefined) {
        change = "added";
        added += 1;
      } else if (after === undefined) {
        change = "removed";
        removed += 1;
      } else if (before !== after) {
        change = "changed";
        changed += 1;
      }
      const lines: DiffLine[] =
        change === "unchanged"
          ? []
          : [
              { kind: "hunk", text: "@@ -1 +1 @@" },
              ...(before ?? "")
                .split("\n")
                .filter(Boolean)
                .map((text) => ({ kind: "removed" as const, text: `-${text}` })),
              ...(after ?? "")
                .split("\n")
                .filter(Boolean)
                .map((text) => ({ kind: "added" as const, text: `+${text}` })),
            ];
      return {
        path,
        label: label(path),
        change,
        from_hash: before === undefined ? "" : hash(before),
        to_hash: after === undefined ? "" : hash(after),
        lines,
      };
    });
    const ref = (v: StoredVersion) => ({
      version: v.version,
      status: v.status,
      created_by: v.created_by,
      created_at: v.created_at,
      activated_at: v.activated_at,
      activated_by: v.activated_by,
      notes: v.notes,
      package_hash: this.packageHash(v),
    });
    return {
      layer,
      from: ref(left),
      to: ref(right),
      summary: {
        added,
        changed,
        removed,
        unchanged: files.filter((f) => f.change === "unchanged").length,
      },
      files,
      dependencies: {
        shared: DEPENDENCIES[layer],
        note: "Both versions depend on the same layers; only the file contents differ.",
      },
      validation: { from: left.validation, to: right.validation },
    };
  }

  impact(layer: ConfigLayer, version?: number): ImpactAnalysis {
    const v = version ?? this.active(layer).version;
    const isActive = this.active(layer).version === v;
    const workflows = isActive
      ? [
          {
            workflow_id: "wf-1001",
            client_id: "Alpine Capital",
            portfolio_id: "European Growth",
            status: "needs_review",
            reporting_period: "2026-06-30",
            pinned_version: `v${v}`,
            still_running: true,
          },
          {
            workflow_id: "wf-1003",
            client_id: "Cedar Rock Advisors",
            portfolio_id: "Real Assets",
            status: "published",
            reporting_period: "2026-06-30",
            pinned_version: `v${v}`,
            still_running: false,
          },
        ]
      : [];
    return {
      layer,
      version: v,
      clients: isActive ? 2 : 0,
      portfolios: isActive ? 2 : 0,
      pinned_workflows: workflows.length,
      running_pinned_workflows: workflows.filter((w) => w.still_running).length,
      future_workflows_affected: true,
      workflows,
      explanation: [
        "Workflows that have already started stay on the version they began with.",
        "Only workflows started after activation will use the new version.",
        "Reports that have already been published are unchanged.",
      ],
    };
  }

  auditTrail(): AuditTrail {
    return { entries: [...this.audit], chain_intact: true };
  }

  createDraft(layer: ConfigLayer, input?: CreateDraftInput): { version: number; status: string } {
    const base = input?.from_version ? this.find(layer, input.from_version) : this.active(layer);
    const next = Math.max(...this.of(layer).map((v) => v.version)) + 1;
    const files = { ...base.files, ...(input?.edits ?? {}) };
    const created_at = new Date().toISOString();
    this.versions.push({
      layer,
      version: next,
      status: "draft",
      notes: input?.notes ?? "",
      based_on_version: base.version,
      created_by: this.actor,
      created_at,
      validated: false,
      validation: null,
      activated_by: null,
      activated_at: null,
      files,
    });
    this.record({
      event: "config_draft_created",
      event_label: "Draft created",
      actor: this.actor,
      at: created_at,
      layer,
      layer_label: LAYER_LABELS[layer],
      version: next,
      notes: input?.notes ?? "",
      outcome: "ok",
      description: `${LAYER_LABELS[layer]} draft v${next} created by ${this.actor}, based on v${base.version}. A draft changes nothing until it is validated and activated.`,
    });
    return { version: next, status: "draft" };
  }

  validate(
    layer: ConfigLayer,
    version: number,
  ): { validation: ValidationResult; validation_summary: ValidationSummary } {
    const v = this.find(layer, version);
    const problems = Object.entries(v.files)
      .filter(([, content]) => content.includes("[") && !content.includes("]"))
      .map(([path]) => `${path}: not valid YAML (ScannerError)`);
    const at = new Date().toISOString();
    v.validation = { ok: problems.length === 0, problems, validated_at: at };
    v.validated = problems.length === 0;
    this.record({
      event: "config_validated",
      event_label: "Validation completed",
      actor: this.actor,
      at,
      layer,
      layer_label: LAYER_LABELS[layer],
      version,
      notes: "",
      outcome: v.validated ? "ok" : "failed",
      description: v.validated
        ? `${LAYER_LABELS[layer]} v${version} passed its checks when ${this.actor} asked Trakt to check it.`
        : `${LAYER_LABELS[layer]} v${version} did not pass: ${problems.length} ${
            problems.length === 1 ? "check" : "checks"
          } failed. It cannot be activated until they are fixed.`,
    });
    return { validation: v.validation, validation_summary: explain(layer, v.validation) };
  }

  activate(layer: ConfigLayer, version: number): { active_version: number } {
    const v = this.find(layer, version);
    if (v.status === "active") return { active_version: version };
    if (!v.validated) {
      throw new OpsError(
        "This version has not passed its checks, so it cannot be activated. Check it first.",
        "OPS_CONFIG_NOT_READY",
      );
    }
    const current = this.active(layer);
    if (current.version !== version) current.status = "superseded";
    const at = new Date().toISOString();
    v.status = "active";
    v.activated_by = this.actor;
    v.activated_at = at;
    this.record({
      event: "config_activated",
      event_label: "Activated",
      actor: this.actor,
      at,
      layer,
      layer_label: LAYER_LABELS[layer],
      version,
      notes: "",
      outcome: "ok",
      description: `${LAYER_LABELS[layer]} v${version} activated by ${this.actor}. Future workflows will use this version. Existing workflows remain pinned.`,
    });
    return { active_version: version };
  }

  rollback(layer: ConfigLayer, toVersion: number): { active_version: number } {
    const target = this.find(layer, toVersion);
    target.validated = true;
    const current = this.active(layer);
    if (current.version !== toVersion) current.status = "superseded";
    const at = new Date().toISOString();
    target.status = "active";
    target.activated_by = this.actor;
    target.activated_at = at;
    this.record({
      event: "config_rolled_back",
      event_label: "Rolled back",
      actor: this.actor,
      at,
      layer,
      layer_label: LAYER_LABELS[layer],
      version: toVersion,
      notes: "",
      outcome: "ok",
      description: `${LAYER_LABELS[layer]} rolled back to v${toVersion} by ${this.actor}. The version it replaced has been kept.`,
    });
    return { active_version: toVersion };
  }
}
