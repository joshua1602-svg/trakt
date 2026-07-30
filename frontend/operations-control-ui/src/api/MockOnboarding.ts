/**
 * In-memory Client Onboarding, for demonstration and for the browser tests.
 *
 * It mirrors the real service closely enough to exercise the screens — the same
 * governed vocabularies, the same standing-field declaration, the same
 * derivation rules, the same immutable versioning — but it is a fixture, not a
 * second implementation: the real decisions live in
 * `operations_control/onboarding/`, and the HTTP client talks to those.
 */

import type {
  ApprovalResult,
  OnboardingClientDetail,
  OnboardingClientRow,
  OnboardingDraft,
  OnboardingProblem,
  OnboardingProfile,
  OnboardingReference,
  OnboardingReview,
  OnboardingStep,
  PlannedArtefact,
  ProfileChange,
  ProfileVersionRow,
  SourceRegistrationStanding,
  StandingProduct,
  UnrepresentedField,
} from "./onboardingTypes";

const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

const STEPS: { key: OnboardingStep; label: string }[] = [
  { key: "client", label: "Client" },
  { key: "portfolio", label: "Portfolios" },
  { key: "reporting", label: "Reporting" },
  { key: "regime", label: "Regime configuration" },
  { key: "static_reporting", label: "Investor and static reporting" },
  { key: "sources", label: "Source registration" },
  { key: "review", label: "Review" },
];

const PRODUCTS: Record<string, StandingProduct> = {
  mi: {
    label: "Management Information",
    always_applies: true,
    derived_from:
      "Every governed delivery produces MI. The workflow outcome vocabulary has no MI-less option.",
    fields: [],
  },
  esma_annex2: {
    label: "ESMA Annex 2",
    requires_dataset: "funded",
    supported_by_assets_declaring: "ESMA_Annex2",
    fields: [
      {
        key: "originator_name",
        label: "Originator name",
        regime_code: "RREL82",
        classification: "standing_client",
        required: true,
        format: "text",
        max_length: 100,
        help: "The full legal name of the originator. Must match the name held against the LEI in the GLEIF database.",
        writes_to: "client_config:defaults.originator_name",
      },
      {
        key: "originator_legal_entity_identifier",
        label: "Originator LEI",
        regime_code: "RREL83",
        classification: "standing_client",
        required: true,
        format: "lei",
        help: "The originator's 20-character Legal Entity Identifier.",
        writes_to: "client_config:defaults.originator_legal_entity_identifier",
      },
      {
        key: "originator_establishment_country",
        label: "Originator country of establishment",
        regime_code: "RREL84",
        classification: "standing_client",
        required: true,
        format: "country",
        writes_to: "client_config:defaults.originator_establishment_country",
      },
      {
        key: "original_lender_legal_entity_identifier",
        label: "Original lender LEI",
        regime_code: "RREL80",
        classification: "standing_client",
        required: false,
        format: "lei",
        help: "Only where the original lender differs from the originator and is the same for the whole book. Left blank, the delivery rules apply ND5.",
        writes_to: "client_config:defaults.original_lender_legal_entity_identifier",
      },
      {
        key: "original_lender_establishment_country",
        label: "Original lender country of establishment",
        regime_code: "RREL81",
        classification: "standing_client",
        required: false,
        format: "country",
        writes_to: "client_config:defaults.original_lender_establishment_country",
      },
      {
        key: "nuts_classification_year",
        label: "NUTS classification year",
        classification: "standing_portfolio",
        required: false,
        format: "enum",
        options: ["2021", "2024"],
        help: "The NUTS vintage used for the regulatory geographic region fields.",
        writes_to: "client_config:nuts_classification_year",
      },
      {
        key: "uk_geography_override",
        label: "Report UK geography as GBZZZ",
        classification: "standing_portfolio",
        required: false,
        format: "boolean",
        help: "Delivers GBZZZ for UK obligor and collateral region (RREL11 / RREC6) while retaining granular ITL3 in canonical for MI.",
        writes_to: "client_config:regime_overrides.ESMA_Annex2.uk_geography.enabled",
      },
    ],
    unrepresented: [
      {
        key: "sponsor_name",
        label: "Sponsor",
        reason:
          "Annex 2 is the underlying-exposure report; it carries no sponsor field. Sponsor appears only as a risk-retention holder value in Annex 12.",
      },
      {
        key: "sspe_name",
        label: "SSPE",
        reason:
          "No SSPE field exists in the Annex 2 field universe. The securitisation entity is named in Annex 12.",
      },
      {
        key: "national_competent_authority",
        label: "National competent authority",
        reason: "Trakt produces the report; it does not file it. No NCA field exists.",
      },
      {
        key: "sts_status",
        label: "STS status and notification identifier",
        reason:
          "No STS fields exist in the current Annex 2 model. Adding them extends the field universe, which is an administrator change.",
      },
      {
        key: "servicer_name",
        label: "Servicer",
        reason:
          "A source registration records who supplies the data. There is no regulatory servicer field in the Annex 2 model.",
      },
      {
        key: "calculation_conventions",
        label: "Calculation conventions",
        reason:
          "Day-count and payment-frequency conventions are captured as static reporting information, but they are not Annex 2 fields.",
      },
    ],
  },
  investor_reporting: {
    label: "Investor Reporting",
    regime_id: "ESMA_Annex12",
    fields: [
      {
        key: "IVSS3",
        label: "Securitisation name",
        classification: "standing_client",
        format: "text",
        writes_to: "annex12_config:annex12.deal.IVSS3",
      },
      {
        key: "IVSS4",
        label: "Reporting entity name",
        classification: "standing_client",
        format: "text",
        writes_to: "annex12_config:annex12.deal.IVSS4",
      },
      {
        key: "IVSS5",
        label: "Reporting entity contact person",
        classification: "standing_client",
        required: true,
        format: "text",
        writes_to: "annex12_config:annex12.deal.IVSS5",
      },
      {
        key: "IVSS6",
        label: "Reporting entity contact telephone",
        classification: "standing_client",
        required: true,
        format: "text",
        writes_to: "annex12_config:annex12.deal.IVSS6",
      },
      {
        key: "IVSS7",
        label: "Reporting entity contact email",
        classification: "standing_client",
        required: true,
        format: "email",
        writes_to: "annex12_config:annex12.deal.IVSS7",
      },
      {
        key: "IVSS8",
        label: "Risk retention method",
        classification: "standing_client",
        required: true,
        format: "enum",
        options: [
          { value: "VSLC", label: "Vertical slice" },
          { value: "SLLS", label: "Seller's share" },
          { value: "RSEX", label: "Randomly-selected exposures" },
          { value: "FLTR", label: "First loss tranche" },
          { value: "FLEX", label: "First loss exposure in each asset" },
          { value: "NCOM", label: "No compliance" },
          { value: "OTHR", label: "Other" },
        ],
        writes_to: "annex12_config:annex12.deal.IVSS8",
      },
      {
        key: "IVSS9",
        label: "Risk retention holder",
        classification: "standing_client",
        required: true,
        format: "enum",
        options: [
          { value: "ORIG", label: "Originator" },
          { value: "SPON", label: "Sponsor" },
          { value: "OLND", label: "Original lender" },
          { value: "SELL", label: "Seller" },
          { value: "NCOM", label: "No compliance" },
          { value: "OTHR", label: "Other" },
        ],
        writes_to: "annex12_config:annex12.deal.IVSS9",
      },
      {
        key: "IVSS10",
        label: "Underlying exposure type",
        classification: "standing_portfolio",
        required: true,
        format: "enum",
        options: [
          { value: "RMRT", label: "Residential mortgage" },
          { value: "CMRT", label: "Commercial mortgage" },
          { value: "CONL", label: "Consumer loan" },
          { value: "ALOL", label: "Automobile loan or lease" },
          { value: "SMEL", label: "SME" },
          { value: "MIXD", label: "Mixed" },
          { value: "OTHR", label: "Other" },
        ],
        writes_to: "annex12_config:annex12.deal.IVSS10",
      },
      {
        key: "IVSS11",
        label: "Risk transfer method applied",
        classification: "standing_client",
        format: "yes_no",
        writes_to: "annex12_config:annex12.deal.IVSS11",
      },
      {
        key: "IVSS13",
        label: "Revolving / ramp-up end date",
        classification: "standing_client",
        format: "date_or_nd",
        help: "A date where one applies, otherwise ND5.",
        writes_to: "annex12_config:annex12.deal.IVSS13",
      },
    ],
    unrepresented: [
      {
        key: "trustee_name",
        label: "Trustee",
        reason:
          "Captured as static reporting information, but Annex 12 has no trustee field, so it is not written into the regime configuration.",
      },
      {
        key: "IVSR_triggers",
        label: "Trigger definitions (IVSR1-IVSR10)",
        reason:
          "Deal-structure information, not client standing information. Left where it is until a governed trigger model exists.",
      },
    ],
  },
  static_pools: {
    label: "Static Pools",
    requires_dataset: "funded",
    fields: [],
    unrepresented: [
      {
        key: "cohort_definition",
        label: "Cohort definition",
        reason:
          "Held as asset configuration and administered through the governed asset package, not per client.",
      },
    ],
  },
};

const REFERENCE: OnboardingReference = {
  vocabularies: {
    asset_classes: [
      { value: "equity_release", label: "Equity Release", supports_regimes: ["ESMA_Annex2"] },
    ],
    portfolio_types: ["direct", "acquired"],
    portfolio_structures: ["on_balance_sheet", "warehouse", "spv", "managed"],
    datasets: ["funded", "pipeline"],
    regime_capable_datasets: ["funded"],
    cadences: ["monthly", "weekly", "daily", "adhoc", "ad_hoc"],
    registration_statuses: ["pending_review", "active"],
    products: Object.entries(PRODUCTS).map(([key, product]) => ({
      key,
      label: product.label,
      selectable: key !== "mi",
      always_applies: Boolean(product.always_applies),
      derived_from: product.derived_from ?? "",
      requires_dataset: product.requires_dataset ?? "",
      supported_by_assets_declaring: product.supported_by_assets_declaring ?? "",
      field_count: product.fields.length,
      unrepresented: product.unrepresented ?? [],
    })),
    sources: {
      asset_classes: "operations_control.configuration.packages.ASSET_MODEL",
      portfolio_types: "apps.blob_trigger_app.source_registry.SourceRecord",
      datasets: "operations_control.contracts.BATCH_DATASETS",
      cadences: "apps.blob_trigger_app.path_parser.VALID_FREQUENCIES",
      products: "config/regime/onboarding_standing_fields.yaml",
    },
  },
  standing_fields: { version: 1, products: PRODUCTS },
};

function emptyProfile(clientId = ""): OnboardingProfile {
  return {
    client_id: clientId,
    client: {
      client_id: clientId,
      client_name: "",
      legal_entity_name: "",
      lei: "",
      jurisdiction: "",
      reporting_currency: "",
      time_zone: "Europe/London",
      environment: "production",
      primary_reporting_contact: "",
      reporting_email: "",
      operational_contact: "",
      operational_email: "",
    },
    portfolios: [],
    reporting: { products: [], derivation: {} },
    regime: {},
    static_reporting: {
      app_title: "",
      primary_color: "",
      logo_uri: "",
      disclaimer: "",
      investor_contact: "",
      investor_email: "",
      trustee_name: "",
      warehouse_lender: "",
      payment_convention: "",
      day_count_convention: "",
      reporting_convention: "",
    },
    sources: [],
  };
}

/** The ERE client as it exists today, before onboarding has adopted it. */
function adoptedEreProfile(): OnboardingProfile {
  const profile = emptyProfile("ERE");
  profile.client = {
    client_id: "ERE",
    client_name: "ERE Funding - Equity Release Mortgages",
    legal_entity_name: "ERE Funding Limited",
    lei: "213800ABCDE123456701",
    jurisdiction: "GB",
    reporting_currency: "GBP",
    time_zone: "Europe/London",
    environment: "production",
    primary_reporting_contact: "",
    reporting_email: "",
    operational_contact: "",
    operational_email: "",
  };
  profile.portfolios = [
    {
      portfolio_id: "direct_001",
      display_name: "ERE Direct Originations",
      portfolio_type: "direct",
      asset_class: "equity_release",
      structure: "on_balance_sheet",
      funded_cadence: "monthly",
      pipeline_cadence: "weekly",
      originates: true,
      warehouse_lender: "",
      source_system: "ERE core lending platform",
    },
    {
      portfolio_id: "acquired_001",
      display_name: "BigBank Legacy Book",
      portfolio_type: "acquired",
      asset_class: "equity_release",
      structure: "spv",
      funded_cadence: "ad_hoc",
      pipeline_cadence: "",
      originates: false,
      warehouse_lender: "",
      source_system: "BigBank plc (seller)",
    },
  ];
  profile.reporting = {
    products: ["esma_annex2"],
    derivation: {
      esma_annex2:
        "pipeline.esma_enabled / supported_regimes in the client configuration, and regime_required source registrations",
    },
  };
  profile.regime = {
    esma_annex2: {
      originator_name: "ERE Funding Limited",
      originator_legal_entity_identifier: "213800ABCDE123456701",
      originator_establishment_country: "GB",
      nuts_classification_year: "2021",
      uk_geography_override: true,
    },
  };
  profile.static_reporting = {
    ...profile.static_reporting,
    app_title: "Portfolio MI Dashboard",
    primary_color: "#0B1F3B",
    day_count_convention: "ACT365",
    payment_convention: "QUARTERLY",
  };
  profile.sources = deriveSources(profile);
  return profile;
}

function deriveSources(profile: OnboardingProfile): SourceRegistrationStanding[] {
  const annex2 = profile.reporting.products.includes("esma_annex2");
  const out: SourceRegistrationStanding[] = [];
  for (const p of profile.portfolios) {
    const supports = REFERENCE.vocabularies.asset_classes
      .find((a) => a.value === p.asset_class)
      ?.supports_regimes.includes("ESMA_Annex2");
    out.push({
      portfolio_id: p.portfolio_id,
      dataset: "funded",
      frequency: p.funded_cadence,
      source_system: p.source_system,
      expected_files: [],
      regime_required: Boolean(annex2 && supports),
      status: "pending_review",
      blob_prefix: "",
    });
    if (p.pipeline_cadence) {
      out.push({
        portfolio_id: p.portfolio_id,
        dataset: "pipeline",
        frequency: p.pipeline_cadence,
        source_system: p.source_system,
        expected_files: [],
        regime_required: false,
        status: "pending_review",
        blob_prefix: "",
      });
    }
  }
  return out;
}

const LEI_RE = /^[A-Z0-9]{18}[0-9]{2}$/;
const CCY_RE = /^[A-Z]{3}$/;
const COUNTRY_RE = /^[A-Z]{2}$/;
const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

function validate(profile: OnboardingProfile): OnboardingProblem[] {
  const out: OnboardingProblem[] = [];
  const push = (step: string, field: string, message: string) =>
    out.push({ step, field, message });
  const c = profile.client;
  if (!c.client_id.trim()) push("client", "client_id", "The client needs an identifier.");
  if (!c.client_name.trim()) push("client", "client_name", "The client needs a name.");
  if (!c.legal_entity_name.trim())
    push("client", "legal_entity_name", "The legal entity name has not been given.");
  if (!c.lei.trim()) push("client", "lei", "The legal entity identifier has not been given.");
  else if (!LEI_RE.test(c.lei.trim().toUpperCase()))
    push("client", "lei", "That is not a valid 20-character LEI.");
  if (!COUNTRY_RE.test(c.jurisdiction.trim().toUpperCase()))
    push("client", "jurisdiction", "Give the jurisdiction as a two-letter country code.");
  if (!CCY_RE.test(c.reporting_currency.trim().toUpperCase()))
    push("client", "reporting_currency", "Give the reporting currency as a three-letter code.");
  if (!c.time_zone.trim())
    push("client", "time_zone", "The reporting time zone has not been given.");
  if (!c.primary_reporting_contact.trim())
    push("client", "primary_reporting_contact", "The primary reporting contact has not been named.");
  if (!EMAIL_RE.test(c.reporting_email.trim()))
    push("client", "reporting_email", "The reporting email address has not been given.");
  if (!c.operational_contact.trim())
    push("client", "operational_contact", "The operational contact has not been named.");
  if (!EMAIL_RE.test(c.operational_email.trim()))
    push("client", "operational_email", "The operational email address has not been given.");

  if (profile.portfolios.length === 0)
    push("portfolio", "portfolios", "A client needs at least one portfolio.");
  for (const p of profile.portfolios) {
    const where = p.portfolio_id || "this portfolio";
    if (!p.portfolio_id.trim())
      push("portfolio", "portfolio_id", "Every portfolio needs an identifier.");
    if (!p.display_name.trim())
      push("portfolio", "display_name", `${where} needs a display name.`);
    if (!REFERENCE.vocabularies.asset_classes.some((a) => a.value === p.asset_class))
      push("portfolio", "asset_class", `${where} has no recognised asset class.`);
  }

  for (const key of profile.reporting.products) {
    const product = PRODUCTS[key];
    if (!product) continue;
    const answers = profile.regime[key] ?? {};
    for (const field of product.fields) {
      const value = answers[field.key];
      const text = value === undefined || value === null ? "" : String(value).trim();
      if (!text) {
        if (field.required)
          push("regime", `${key}.${field.key}`, `${field.label} is needed for ${product.label}.`);
        continue;
      }
      if (field.format === "lei" && !LEI_RE.test(text.toUpperCase()))
        push("regime", `${key}.${field.key}`, `${field.label} is not a valid 20-character LEI.`);
      if (field.format === "country" && !COUNTRY_RE.test(text.toUpperCase()))
        push("regime", `${key}.${field.key}`, `${field.label} must be a two-letter country code.`);
      if (field.format === "email" && !EMAIL_RE.test(text))
        push("regime", `${key}.${field.key}`, `${field.label} does not look like an email address.`);
    }
  }
  return out;
}

function flatten(node: unknown, prefix = "", out: Record<string, string> = {}) {
  if (Array.isArray(node)) {
    if (node.every((v) => typeof v !== "object" || v === null)) {
      out[prefix] = node.join(", ");
    } else {
      node.forEach((v, i) => {
        const key =
          v && typeof v === "object" && "portfolio_id" in (v as object)
            ? String((v as { portfolio_id: string }).portfolio_id)
            : String(i);
        flatten(v, `${prefix}[${key}]`, out);
      });
    }
  } else if (node && typeof node === "object") {
    for (const [k, v] of Object.entries(node)) flatten(v, prefix ? `${prefix}.${k}` : k, out);
  } else {
    out[prefix] = node === null || node === undefined ? "" : String(node);
  }
  return out;
}

const FIELD_LABELS: Record<string, string> = {
  "client.client_name": "Client name",
  "client.legal_entity_name": "Legal entity name",
  "client.lei": "Legal entity identifier",
  "client.jurisdiction": "Jurisdiction",
  "client.reporting_currency": "Reporting currency",
  "client.time_zone": "Time zone",
  "client.primary_reporting_contact": "Primary reporting contact",
  "client.reporting_email": "Reporting email",
  "client.operational_contact": "Operational contact",
  "client.operational_email": "Operational email",
  "reporting.products": "Reporting products",
};

/** Paths that restate another path — the root client id mirrors client.client_id. */
const MIRRORED_PATHS = new Set(["client_id"]);

function diff(before: OnboardingProfile | null, after: OnboardingProfile): ProfileChange[] {
  const old = before ? flatten(before) : {};
  const now = flatten(after);
  const paths = Array.from(new Set([...Object.keys(old), ...Object.keys(now)]))
    .filter((path) => !MIRRORED_PATHS.has(path))
    .sort();
  const changes: ProfileChange[] = [];
  for (const path of paths) {
    const was = old[path] ?? "";
    const next = now[path] ?? "";
    if (was === next || (!was && !next)) continue;
    changes.push({
      path,
      label: FIELD_LABELS[path] ?? path.replace(/_/g, " ").replace(/\./g, " → "),
      before: was,
      after: next,
      action: !was ? "added" : !next ? "removed" : "changed",
    });
  }
  return changes;
}

interface StoredDraft {
  draft_id: string;
  client_id: string;
  origin: string;
  based_on_version: number | null;
  step: OnboardingStep;
  profile: OnboardingProfile;
  provenance: Record<string, string>;
  gaps: OnboardingDraft["gaps"];
  sources_read: string[];
  created_by: string;
  created_at: string;
}

export class MockOnboarding {
  private drafts = new Map<string, StoredDraft>();
  private versions = new Map<string, ProfileVersionRow[]>();
  private profiles = new Map<string, OnboardingProfile>();
  private artefacts = new Map<string, Map<string, string>>();
  private counter = 0;

  reference(): OnboardingReference {
    return clone(REFERENCE);
  }

  clients(): OnboardingClientRow[] {
    const rows: OnboardingClientRow[] = [];
    const known = new Set<string>(["ERE", ...this.profiles.keys()]);
    for (const clientId of Array.from(known).sort()) {
      const profile = this.profiles.get(clientId);
      const history = this.versions.get(clientId) ?? [];
      if (!profile) {
        rows.push({
          client_id: clientId,
          status: "legacy",
          display_name: clientId,
          version: 0,
          portfolios: 0,
          products: [],
          updated_at: "",
          updated_by: "",
          open_drafts: 0,
          summary:
            "Configured before Client Onboarding existed. Its current values can be adopted without re-entering them.",
        });
        continue;
      }
      rows.push({
        client_id: clientId,
        status: "onboarded",
        display_name: profile.client.client_name || clientId,
        version: history[0]?.version ?? 0,
        portfolios: profile.portfolios.length,
        products: [...profile.reporting.products],
        updated_at: history[0]?.approved_at ?? "",
        updated_by: history[0]?.approved_by ?? "",
        open_drafts: 0,
        summary: "",
      });
    }
    return rows;
  }

  startDraft(input: { client_id?: string; adopt?: boolean }, by: string): OnboardingDraft {
    const clientId = input.client_id ?? "";
    const existing = clientId ? this.profiles.get(clientId) : undefined;
    let profile: OnboardingProfile;
    let origin = "new";
    let gaps: OnboardingDraft["gaps"] = [];
    let provenance: Record<string, string> = {};
    let sourcesRead: string[] = [];
    if (existing) {
      profile = clone(existing);
      origin = "current";
    } else if (input.adopt && clientId === "ERE") {
      profile = adoptedEreProfile();
      origin = "adopted";
      provenance = {
        "client.client_name": "client.display_name in the client configuration",
        "client.legal_entity_name": "defaults.originator_name (RREL82) in the client configuration",
        "client.lei":
          "defaults.originator_legal_entity_identifier (RREL83) in the client configuration",
        "client.jurisdiction": "portfolio.country",
        "client.reporting_currency": "portfolio.base_currency",
        sources: "the durable source registry",
        "static_reporting.branding": "mi.branding in the client configuration",
      };
      sourcesRead = [
        "config/client/config_client_ERM_UK.yaml",
        "config/client/portfolio_registry.yaml",
        "config/tenancy.yaml",
        "source registry",
      ];
      gaps = [
        {
          step: "client",
          field: "reporting_contact",
          label: "Primary reporting contact",
          why: "The client configuration has never carried reporting contacts.",
        },
        {
          step: "client",
          field: "operational_contact",
          label: "Operational contact",
          why: "The client configuration has never carried operational contacts.",
        },
      ];
    } else {
      profile = emptyProfile(clientId);
    }
    const draft: StoredDraft = {
      draft_id: `onb_${++this.counter}`,
      client_id: profile.client_id,
      origin,
      based_on_version: (this.versions.get(clientId) ?? [])[0]?.version ?? null,
      step: "client",
      profile,
      provenance,
      gaps,
      sources_read: sourcesRead,
      created_by: by,
      created_at: new Date().toISOString(),
    };
    this.drafts.set(draft.draft_id, draft);
    return this.present(draft);
  }

  draft(draftId: string): OnboardingDraft {
    return this.present(this.require(draftId));
  }

  saveStep(draftId: string, step: OnboardingStep, payload: Record<string, unknown>) {
    const draft = this.require(draftId);
    const profile = draft.profile;
    if (step === "client") {
      profile.client = { ...profile.client, ...(payload as Partial<OnboardingProfile["client"]>) };
      profile.client_id = profile.client.client_id;
      draft.client_id = profile.client_id;
    } else if (step === "portfolio") {
      profile.portfolios = (payload.portfolios ?? []) as OnboardingProfile["portfolios"];
      profile.sources = deriveSources(profile);
    } else if (step === "reporting") {
      profile.reporting = {
        products: (payload.products ?? []) as string[],
        derivation: profile.reporting.derivation,
      };
      profile.sources = deriveSources(profile);
    } else if (step === "regime") {
      const incoming = (payload.regime ?? {}) as Record<string, Record<string, string | boolean>>;
      for (const [key, answers] of Object.entries(incoming)) {
        profile.regime[key] = { ...(profile.regime[key] ?? {}), ...answers };
      }
    } else if (step === "static_reporting") {
      profile.static_reporting = {
        ...profile.static_reporting,
        ...(payload as Partial<OnboardingProfile["static_reporting"]>),
      };
    } else if (step === "sources") {
      profile.sources = (payload.sources ?? []) as SourceRegistrationStanding[];
    }
    draft.step = step;
    return this.present(draft);
  }

  review(draftId: string): OnboardingReview {
    const draft = this.require(draftId);
    const profile = draft.profile;
    const current = this.profiles.get(profile.client_id) ?? null;
    const history = this.versions.get(profile.client_id) ?? [];
    const unrepresented: UnrepresentedField[] = [];
    for (const key of profile.reporting.products) {
      for (const entry of PRODUCTS[key]?.unrepresented ?? []) {
        unrepresented.push({ ...entry, product: PRODUCTS[key].label });
      }
    }
    return {
      draft_id: draftId,
      client_id: profile.client_id,
      profile: clone(profile),
      artefacts: this.plan(profile),
      changes: diff(current, profile),
      problems: validate(profile),
      ready: validate(profile).length === 0,
      current_version: history[0]?.version ?? 0,
      next_version: (history[0]?.version ?? 0) + 1,
      unrepresented,
    };
  }

  approve(draftId: string, reason: string, by: string): ApprovalResult {
    const draft = this.require(draftId);
    const profile = draft.profile;
    const clientId = profile.client_id;
    const history = this.versions.get(clientId) ?? [];
    const previous = this.profiles.get(clientId) ?? null;
    const artefacts = this.plan(profile);
    const store = this.artefacts.get(clientId) ?? new Map<string, string>();
    for (const artefact of artefacts) {
      if (artefact.text) store.set(artefact.rel, artefact.text);
    }
    this.artefacts.set(clientId, store);
    const version: ProfileVersionRow = {
      version: (history[0]?.version ?? 0) + 1,
      status: "active",
      approved_by: by,
      approved_at: new Date().toISOString(),
      reason,
      based_on_version: history[0]?.version ?? null,
      changes: diff(previous, profile),
      change_count: diff(previous, profile).length,
      artefacts: artefacts.map((a) => ({
        kind: a.kind,
        target: a.rel,
        action: a.action,
        detail: {},
      })),
      content_hash: `sha256:mock${version_hash(profile)}`,
    };
    for (const row of history) row.status = "superseded";
    this.versions.set(clientId, [version, ...history]);
    this.profiles.set(clientId, clone(profile));
    this.drafts.delete(draftId);
    return {
      version: version.version,
      client_id: clientId,
      artefacts: version.artefacts,
      changes: version.changes,
    };
  }

  discard(draftId: string) {
    this.drafts.delete(draftId);
  }

  client(clientId: string): OnboardingClientDetail {
    const profile = this.profiles.get(clientId);
    const history = this.versions.get(clientId) ?? [];
    if (!profile) {
      return {
        client_id: clientId,
        status: "legacy",
        version: 0,
        profile: null,
        live_source_registrations: [],
        artefacts: [],
        history: [],
        summary:
          "This client has not been through Client Onboarding. Adopt it to manage its configuration here.",
      };
    }
    const unrepresented: UnrepresentedField[] = [];
    for (const key of profile.reporting.products) {
      for (const entry of PRODUCTS[key]?.unrepresented ?? []) {
        unrepresented.push({ ...entry, product: PRODUCTS[key].label });
      }
    }
    return {
      client_id: clientId,
      status: "onboarded",
      version: history[0]?.version ?? 0,
      approved_by: history[0]?.approved_by,
      approved_at: history[0]?.approved_at,
      reason: history[0]?.reason,
      content_hash: history[0]?.content_hash,
      profile: clone(profile),
      derived_reporting: this.derived(profile),
      live_source_registrations: profile.sources.map((s) => ({
        client_id: clientId,
        source_portfolio_id: s.portfolio_id,
        dataset: s.dataset,
        frequency: s.frequency,
        source_system: s.source_system,
        regime_required: s.regime_required,
        status: s.status,
      })),
      artefacts: history[0]?.artefacts ?? [],
      unrepresented,
      history,
    };
  }

  version(clientId: string, version: number): ProfileVersionRow {
    const row = (this.versions.get(clientId) ?? []).find((v) => v.version === version);
    if (!row) throw new Error("That could not be found.");
    return clone(row);
  }

  // -- internals ---------------------------------------------------------- //

  private require(draftId: string): StoredDraft {
    const draft = this.drafts.get(draftId);
    if (!draft) throw new Error("That could not be found.");
    return draft;
  }

  private derived(profile: OnboardingProfile): Record<string, DerivedShape> {
    const out: Record<string, DerivedShape> = {};
    for (const [key, product] of Object.entries(PRODUCTS)) {
      if (product.always_applies) {
        out[key] = {
          applies: true,
          eligible: true,
          derived: true,
          reason: product.derived_from ?? "",
        };
        continue;
      }
      const needs = product.supported_by_assets_declaring;
      let eligible: boolean;
      let reason: string;
      if (needs) {
        const supporting = profile.portfolios
          .filter((p) =>
            REFERENCE.vocabularies.asset_classes
              .find((a) => a.value === p.asset_class)
              ?.supports_regimes.includes(needs),
          )
          .map((p) => p.portfolio_id);
        eligible = supporting.length > 0;
        reason = eligible
          ? `Supported by ${supporting.join(", ")}.`
          : "No portfolio's asset class declares support for this regime.";
      } else {
        eligible = profile.portfolios.length > 0;
        reason = eligible ? "Available for the funded book." : "No portfolio has been added yet.";
      }
      out[key] = {
        applies: profile.reporting.products.includes(key),
        eligible,
        derived: false,
        reason,
      };
    }
    return out;
  }

  private plan(profile: OnboardingProfile): PlannedArtefact[] {
    const clientId = profile.client_id || "CLIENT";
    const stored = this.artefacts.get(clientId);
    const artefacts: PlannedArtefact[] = [];

    const clientRel = `config/client/config_client_${clientId}.yaml`;
    const clientText = renderClientConfig(profile);
    artefacts.push({
      kind: "client_config",
      rel: clientRel,
      label: "Client configuration",
      action: action(stored?.get(clientRel), clientText),
      text: clientText,
      previous_text: stored?.get(clientRel) ?? "",
      records: [],
      summary: [
        `Client identity for ${profile.client.client_name || clientId}.`,
        "Portfolio jurisdiction and reporting currency.",
        `Reporting products: ${profile.reporting.products.join(", ") || "MI only"}.`,
        ...(profile.regime.esma_annex2
          ? ["Annex 2 standing values injected into every reported row."]
          : []),
      ],
    });

    if (profile.reporting.products.includes("investor_reporting")) {
      const rel = `config/client/config_client_${clientId}_annex12.yaml`;
      const text = renderAnnex12Config(profile);
      artefacts.push({
        kind: "annex12_config",
        rel,
        label: "Investor report configuration",
        action: action(stored?.get(rel), text),
        text,
        previous_text: stored?.get(rel) ?? "",
        records: [],
        summary: [
          `${Object.keys(profile.regime.investor_reporting ?? {}).length} standing investor-report fields.`,
        ],
      });
    }

    const pfRel = `config/client/portfolio_registry_${clientId}.yaml`;
    const pfText = renderPortfolioRegistry(profile);
    artefacts.push({
      kind: "portfolio_registry",
      rel: pfRel,
      label: "Portfolio metadata",
      action: action(stored?.get(pfRel), pfText),
      text: pfText,
      previous_text: stored?.get(pfRel) ?? "",
      records: [],
      summary: [`${profile.portfolios.length} portfolios described.`],
    });

    const records = profile.sources.map((s) => {
      const portfolio = profile.portfolios.find((p) => p.portfolio_id === s.portfolio_id);
      return {
        ...s,
        client_id: clientId,
        source_portfolio_id: s.portfolio_id,
        action: "create",
        portfolio_label: portfolio?.display_name ?? "",
        expected_location: `raw-v2/${clientId}/${portfolio?.portfolio_type ?? "direct"}/${s.dataset}/${s.frequency}/${s.portfolio_id}/{reporting_period}/`,
      };
    });
    artefacts.push({
      kind: "source_registry",
      rel: "source registry",
      label: "Source registrations",
      action: records.length ? "create" : "unchanged",
      text: "",
      previous_text: "",
      records,
      summary: records.map(
        (r) => `${clientId}/${r.portfolio_id}/${r.dataset}/${r.frequency} registered, awaiting its first pack.`,
      ),
    });
    return artefacts;
  }

  private present(draft: StoredDraft): OnboardingDraft {
    const problems = validate(draft.profile);
    const byStep: Record<string, OnboardingProblem[]> = {};
    for (const step of STEPS) byStep[step.key] = [];
    for (const problem of problems) (byStep[problem.step] ??= []).push(problem);
    return {
      draft_id: draft.draft_id,
      client_id: draft.client_id,
      origin: draft.origin,
      based_on_version: draft.based_on_version,
      step: draft.step,
      steps: STEPS.map((s) => ({ ...s, problems: (byStep[s.key] ?? []).length })),
      profile: clone(draft.profile),
      derived_reporting: this.derived(draft.profile),
      derived_sources: deriveSources(draft.profile),
      provenance: draft.provenance,
      gaps: draft.gaps,
      sources_read: draft.sources_read,
      problems,
      problems_by_step: byStep,
      ready: problems.length === 0,
      created_by: draft.created_by,
      created_at: draft.created_at,
    };
  }
}

type DerivedShape = { applies: boolean; eligible: boolean; derived: boolean; reason: string };

function action(previous: string | undefined, text: string): string {
  if (previous === undefined) return "create";
  return previous === text ? "unchanged" : "update";
}

function version_hash(profile: OnboardingProfile): string {
  return String(JSON.stringify(profile).length);
}

const BANNER = `# GENERATED by the Trakt Operations Control Centre — Client Onboarding.
# This file is a governed artefact. Change it through Client Onboarding,
# not by hand: a hand edit is not versioned, not attributed and not
# audited, and the next approved change will overwrite it.
`;

function renderClientConfig(profile: OnboardingProfile): string {
  const c = profile.client;
  const annex2 = (profile.regime.esma_annex2 ?? {}) as Record<string, string | boolean>;
  const lines = [
    BANNER,
    "client:",
    `  client_id: ${c.client_id}`,
    `  display_name: ${c.client_name}`,
    `  environment: ${c.environment}`,
    `  legal_entity_name: ${c.legal_entity_name}`,
    `  time_zone: ${c.time_zone}`,
    "  reporting_contact:",
    `    name: ${c.primary_reporting_contact}`,
    `    email: ${c.reporting_email}`,
    "  operational_contact:",
    `    name: ${c.operational_contact}`,
    `    email: ${c.operational_email}`,
    "portfolio:",
    `  asset_class: ${profile.portfolios[0]?.asset_class ?? ""}`,
    `  country: ${c.jurisdiction}`,
    `  base_currency: ${c.reporting_currency}`,
  ];
  if (profile.reporting.products.includes("esma_annex2")) {
    lines.push("supported_regimes:", "  - ESMA_Annex2", "default_regime: ESMA_Annex2", "regime: ESMA_Annex2");
  }
  lines.push(
    "pipeline:",
    "  loan_engine_enabled: false",
    "  mi_enabled: true",
    `  esma_enabled: ${profile.reporting.products.includes("esma_annex2")}`,
  );
  if (Object.keys(annex2).length) {
    lines.push("defaults:");
    if (annex2.originator_name) lines.push(`  originator_name: ${annex2.originator_name}`);
    if (annex2.originator_legal_entity_identifier)
      lines.push(
        `  originator_legal_entity_identifier: ${annex2.originator_legal_entity_identifier}`,
      );
    if (annex2.originator_establishment_country)
      lines.push(`  originator_establishment_country: ${annex2.originator_establishment_country}`);
    if (annex2.nuts_classification_year)
      lines.push(`nuts_classification_year: ${annex2.nuts_classification_year}`);
    if (annex2.uk_geography_override) {
      lines.push(
        "regime_overrides:",
        "  ESMA_Annex2:",
        "    uk_geography:",
        "      enabled: true",
        "      override_value: GBZZZ",
        "      target_fields:",
        "        - geographic_region_obligor",
        "        - geographic_region_collateral",
      );
    }
  }
  const sr = profile.static_reporting;
  if (sr.app_title || sr.primary_color) {
    lines.push("mi:", "  branding:");
    if (sr.app_title) lines.push(`    app_title: ${sr.app_title}`);
    if (sr.primary_color) lines.push("    theme:", `      primary_color: '${sr.primary_color}'`);
  }
  if (sr.day_count_convention || sr.payment_convention) {
    lines.push("loan_engine:");
    if (sr.day_count_convention) lines.push(`  day_count_convention: ${sr.day_count_convention}`);
    if (sr.payment_convention)
      lines.push(`  interest_payment_frequency: ${sr.payment_convention}`);
  }
  return lines.join("\n") + "\n";
}

function renderAnnex12Config(profile: OnboardingProfile): string {
  const answers = (profile.regime.investor_reporting ?? {}) as Record<string, string>;
  const lines = [
    BANNER,
    "annex12:",
    "  meta:",
    "    template: ESMA Annex 12 (Investor Report)",
    "  assumptions:",
    `    reporting_currency: ${profile.client.reporting_currency}`,
    "  deal:",
  ];
  for (const [key, value] of Object.entries(answers)) lines.push(`    ${key}: ${value}`);
  return lines.join("\n") + "\n";
}

function renderPortfolioRegistry(profile: OnboardingProfile): string {
  const lines = [BANNER, "clients:", `  ${profile.client_id}:`, "    portfolios:"];
  for (const p of profile.portfolios) {
    lines.push(
      `      - source_portfolio_id: ${p.portfolio_id}`,
      `        source_portfolio_type: ${p.portfolio_type}`,
      `        source_portfolio_label: ${p.display_name}`,
      `        originates: ${p.originates ?? p.portfolio_type === "direct"}`,
      `        pipeline_data_available: ${Boolean(p.pipeline_cadence)}`,
    );
  }
  return lines.join("\n") + "\n";
}
