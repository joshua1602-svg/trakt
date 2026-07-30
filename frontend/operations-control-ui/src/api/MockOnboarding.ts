/**
 * In-memory Client Onboarding, for demonstration and browser tests.
 *
 * A fixture, not a second implementation: the governed decisions live in
 * `operations_control/onboarding/`, and the HTTP client talks to those. What is
 * mirrored here is only enough to exercise the screens — the same catalogue
 * shape, the same statuses, the same conditional requirements, the same
 * blank-start behaviour.
 *
 * It carries NO client values. A new case starts empty, exactly as the product
 * requires; the legacy fixture used by the migration path is a deliberately
 * synthetic client, not a real one.
 */

import type {
  ActivationResult,
  CaseAnswers,
  CaseKind,
  CasePreview,
  CaseProblem,
  CaseRow,
  CaseStatus,
  CatalogueField,
  ChecklistRow,
  ConfigurationVersionRow,
  InformationRequest,
  OnboardingCase,
  OnboardingClientDetail,
  OnboardingHome,
  OnboardingReference,
  OpenQuestion,
  PlannedArtefact,
} from "./onboardingTypes";
import { CATALOGUE, STEPS, STATUS_LABELS, KIND_LABELS } from "./mockCatalogue";

const clone = <T,>(value: T): T =>
  value === undefined ? (undefined as T) : (JSON.parse(JSON.stringify(value)) as T);

const LEI_RE = /^[A-Z0-9]{18}[0-9]{2}$/;
const CCY_RE = /^[A-Z]{3}$/;
const COUNTRY_RE = /^[A-Z]{2}$/;
const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;
const IDENT_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$/;
const COLOUR_RE = /^#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$/;

function present(value: unknown): boolean {
  if (value === null || value === undefined) return false;
  if (typeof value === "boolean") return true;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value).length > 0;
  return String(value).trim() !== "";
}

/** The same tiny conditional grammar the backend evaluates. */
export function evaluate(
  expression: string,
  answers: CaseAnswers,
  item?: Record<string, unknown>,
): boolean {
  const text = (expression ?? "").trim();
  if (!text) return false;
  if (text === "always") return true;
  const m = /^\s*([\w.]+)\s+(==|!=|in|contains)\s+(.+?)\s*$/.exec(text);
  if (!m) return false;
  const [, path, op, rawValue] = m;
  const actual = resolvePath(path, answers, item);
  const expected = literal(rawValue);
  const scalar = (v: unknown) => (typeof v === "boolean" ? v : String(v ?? "").trim());
  if (op === "==") return scalar(actual) === scalar(expected);
  if (op === "!=") return scalar(actual) !== scalar(expected);
  if (op === "in") {
    const values = Array.isArray(expected) ? expected : [expected];
    return values.map(scalar).includes(scalar(actual));
  }
  if (op === "contains") {
    if (Array.isArray(actual)) return actual.map(scalar).includes(scalar(expected));
    return String(actual ?? "").includes(String(expected));
  }
  return false;
}

function resolvePath(path: string, answers: CaseAnswers, item?: Record<string, unknown>) {
  if (!path.includes(".") && item && path in item) return item[path];
  let node: unknown = answers;
  for (const part of path.split(".")) {
    if (node && typeof node === "object") node = (node as Record<string, unknown>)[part];
    else return undefined;
  }
  return node;
}

function literal(raw: string): unknown {
  const text = raw.trim();
  if (text.startsWith("[") && text.endsWith("]")) {
    return text
      .slice(1, -1)
      .split(",")
      .map((p) => p.trim().replace(/^['"]|['"]$/g, ""))
      .filter(Boolean);
  }
  if (text.toLowerCase() === "true") return true;
  if (text.toLowerCase() === "false") return false;
  return text.replace(/^['"]|['"]$/g, "");
}

export function isRequired(
  field: CatalogueField,
  answers: CaseAnswers,
  item?: Record<string, unknown>,
): boolean {
  if (field.required) return true;
  if (!field.required_when) return false;
  return evaluate(field.required_when, answers, item);
}

function validateValue(field: CatalogueField, value: unknown): string {
  if (!present(value)) return "";
  const text = String(value).trim();
  const rule = field.validation || field.type;
  if (rule === "lei" && !LEI_RE.test(text.toUpperCase()))
    return `${field.label} is not a valid 20-character identifier.`;
  if (rule === "country" && !COUNTRY_RE.test(text.toUpperCase()))
    return `${field.label} must be a two-letter country code.`;
  if (rule === "currency" && !CCY_RE.test(text.toUpperCase()))
    return `${field.label} must be a three-letter currency code.`;
  if (rule === "email" && !EMAIL_RE.test(text))
    return `${field.label} does not look like an email address.`;
  if (rule === "identifier" && !IDENT_RE.test(text))
    return `${field.label} may use letters, numbers, hyphens and underscores only, and must start with a letter or number.`;
  if (rule === "colour" && !COLOUR_RE.test(text))
    return `${field.label} must be a colour such as #1F3B5C.`;
  if (field.max_length && text.length > field.max_length)
    return `${field.label} is longer than ${field.max_length} characters.`;
  if (rule === "enum" && field.options.length) {
    const allowed = new Set(field.options.map((o) => o.value));
    const chosen = Array.isArray(value) ? value : [value];
    for (const item of chosen) {
      if (String(item).trim() && !allowed.has(String(item).trim()))
        return `${field.label} is not one of the accepted values.`;
    }
  }
  return "";
}

interface StoredCase {
  case_id: string;
  kind: CaseKind;
  status: CaseStatus;
  client_id: string;
  client_name: string;
  answers: CaseAnswers;
  information_requests: InformationRequest[];
  open_questions: OpenQuestion[];
  provenance: Record<string, string>;
  based_on_version: number | null;
  events: OnboardingCase["events"];
  created_by: string;
  updated_by: string;
  updated_at: string;
  approved_by: string;
  approved_at: string;
  approval_reason: string;
  activated_version: number | null;
  withdrawal_reason: string;
}

const TRANSITIONS: Record<CaseStatus, CaseStatus[]> = {
  draft: ["information_requested", "in_review", "ready_for_approval", "withdrawn"],
  information_requested: ["awaiting_client", "draft", "in_review", "withdrawn"],
  awaiting_client: ["in_review", "draft", "changes_required", "withdrawn"],
  in_review: ["changes_required", "ready_for_approval", "draft", "withdrawn"],
  changes_required: ["draft", "in_review", "information_requested", "withdrawn"],
  ready_for_approval: ["approved", "changes_required", "draft", "withdrawn"],
  approved: ["activated", "changes_required", "withdrawn"],
  activated: [],
  withdrawn: [],
};

export class MockOnboardingError extends Error {
  constructor(
    message: string,
    readonly errorCode: string,
  ) {
    super(message);
  }
}

export class MockOnboarding {
  private cases = new Map<string, StoredCase>();
  private versions = new Map<string, ConfigurationVersionRow[]>();
  private active = new Map<string, CaseAnswers>();
  private artefactStore = new Map<string, Map<string, string>>();
  private sequence = 0;

  /** A synthetic legacy client, so the migration path is demonstrable without
   *  making any real client part of the product. */
  private legacy = new Map<string, CaseAnswers>([
    [
      "LEGACYCO",
      {
        client: {
          client_id: "LEGACYCO",
          client_name: "Legacy Lending",
          jurisdiction: "GB",
          reporting_currency: "GBP",
          environment: "production",
        },
        entities: [
          {
            entity_id: "ent_legacy_1",
            legal_name: "Legacy Lending Limited",
            roles: ["originator", "reporting_entity"],
            lei: "213800ABCDE123456701N202501",
            country_of_establishment: "GB",
          },
        ],
        contacts: {},
        portfolios: [
          {
            portfolio_id: "direct_001",
            display_name: "Legacy Direct",
            portfolio_type: "direct",
            asset_class: "equity_release",
            owning_entity: "ent_legacy_1",
          },
        ],
        sources: [
          {
            source_key: "direct_001/funded",
            portfolio_id: "direct_001",
            dataset: "funded",
            cadence: "monthly",
            source_party: "Legacy core",
            regime_required: true,
          },
        ],
        reporting: { products: ["esma_annex2"] },
        regime: {
          esma_annex2: {
            originator_name: "Legacy Lending Limited",
            originator_legal_entity_identifier: "213800ABCDE123456701N202501",
            originator_establishment_country: "GB",
          },
        },
        presentation: { report_title: "Legacy MI" },
      },
    ],
  ]);

  reference(): OnboardingReference {
    return {
      catalogue: clone(CATALOGUE),
      steps: STEPS.map((s) => ({ ...s })),
      statuses: (Object.keys(STATUS_LABELS) as CaseStatus[]).map((key) => ({
        key,
        label: STATUS_LABELS[key],
      })),
      kinds: (Object.keys(KIND_LABELS) as CaseKind[]).map((key) => ({
        key,
        label: KIND_LABELS[key],
      })),
    };
  }

  // -- opening ------------------------------------------------------------ //

  private open(kind: CaseKind, answers: CaseAnswers, by: string, extra?: Partial<StoredCase>) {
    this.sequence += 1;
    const stored: StoredCase = {
      case_id: `ONB-2026-${String(this.sequence).padStart(4, "0")}`,
      kind,
      status: "draft",
      client_id: String((answers.client as Record<string, string>)?.client_id ?? ""),
      client_name: String((answers.client as Record<string, string>)?.client_name ?? ""),
      answers,
      information_requests: [],
      open_questions: [],
      provenance: {},
      based_on_version: null,
      events: [],
      created_by: by,
      updated_by: by,
      updated_at: new Date().toISOString(),
      approved_by: "",
      approved_at: "",
      approval_reason: "",
      activated_version: null,
      withdrawal_reason: "",
      ...extra,
    };
    this.syncDerived(stored);
    this.record(stored, "opened", by, {});
    this.cases.set(stored.case_id, stored);
    return stored;
  }

  startNewClient(by: string): OnboardingCase {
    return this.present(this.open("new_client", { sources: [] }, by));
  }

  startMigration(clientId: string, by: string): OnboardingCase {
    const answers = clone(this.legacy.get(clientId) ?? { sources: [] });
    const stored = this.open("migration", answers, by, {
      provenance: {
        "client.client_name": "the client configuration",
        "entities.originator": "the originator defaults in the client configuration (RREL82/83/84)",
        sources: "the durable source registrations",
      },
    });
    // Legacy values today's rules refuse surface immediately as questions.
    for (const problem of this.validate(stored)) {
      if (problem.message.includes("not a valid")) {
        stored.open_questions.push({
          question_id: `q_${stored.open_questions.length + 1}`,
          question: `${problem.message} (carried over from the existing configuration)`,
          origin: "migration",
          raised_at: new Date().toISOString(),
          raised_by: "Trakt",
        });
      }
    }
    return this.present(stored);
  }

  startAmendment(clientId: string, by: string): OnboardingCase {
    const current = this.active.get(clientId);
    if (!current) {
      throw new MockOnboardingError(
        "This client has no approved configuration to amend.",
        "OPS_CLIENT_NOT_ONBOARDED",
      );
    }
    const history = this.versions.get(clientId) ?? [];
    return this.present(
      this.open("amendment", clone(current), by, {
        based_on_version: history[0]?.version ?? null,
      }),
    );
  }

  // -- answering ---------------------------------------------------------- //

  saveStep(caseId: string, step: string, payload: Record<string, unknown>): OnboardingCase {
    const stored = this.require(caseId);
    this.requireEditable(stored);
    const section = CATALOGUE.sections.find((s) => s.key === step);
    const before = clone(stored.answers[step]);

    if (section?.from_regime) {
      const held = (stored.answers.regime ?? {}) as Record<string, Record<string, unknown>>;
      for (const [product, answers] of Object.entries(
        (payload.regime ?? {}) as Record<string, Record<string, unknown>>,
      )) {
        held[product] = { ...(held[product] ?? {}), ...answers };
      }
      stored.answers.regime = held;
    } else if (section?.repeatable) {
      stored.answers[step] = (payload[section.key] ?? payload.items ?? []) as unknown[];
    } else if (section) {
      stored.answers[step] = { ...((stored.answers[step] ?? {}) as object), ...payload };
    }

    if (step === "client") {
      const client = (stored.answers.client ?? {}) as Record<string, string>;
      stored.client_id = client.client_id ?? "";
      stored.client_name = client.client_name ?? "";
    }
    this.syncDerived(stored);
    this.record(stored, `answered_${step}`, stored.updated_by, {
      before: { [step]: before },
      after: { [step]: clone(stored.answers[step]) },
    });
    return this.present(stored);
  }

  addPipelineBook(caseId: string, portfolioId: string): OnboardingCase {
    const stored = this.require(caseId);
    this.requireEditable(stored);
    const sources = (stored.answers.sources ?? []) as Record<string, unknown>[];
    if (!sources.some((s) => s.portfolio_id === portfolioId && s.dataset === "pipeline")) {
      sources.push({
        source_key: `${portfolioId}/pipeline`,
        portfolio_id: portfolioId,
        dataset: "pipeline",
        cadence: "",
        source_party: "",
        regime_required: false,
      });
    }
    stored.answers.sources = sources;
    this.syncDerived(stored);
    return this.present(stored);
  }

  removeSource(caseId: string, portfolioId: string, dataset: string): OnboardingCase {
    const stored = this.require(caseId);
    this.requireEditable(stored);
    stored.answers.sources = ((stored.answers.sources ?? []) as Record<string, unknown>[]).filter(
      (s) => !(s.portfolio_id === portfolioId && s.dataset === dataset),
    );
    return this.present(stored);
  }

  private syncDerived(stored: StoredCase) {
    const entities = (stored.answers.entities ?? []) as Record<string, unknown>[];
    entities.forEach((e, i) => {
      if (!e.entity_id) e.entity_id = `ent_${stored.case_id}_${i + 1}`;
    });
    const portfolios = (stored.answers.portfolios ?? []) as Record<string, unknown>[];
    portfolios.forEach((p) => {
      if (p.originates === undefined || p.originates === null) {
        p.originates = p.portfolio_type === "direct";
      }
    });
    const existing = (stored.answers.sources ?? []) as Record<string, unknown>[];
    const annex2 = this.products(stored).includes("esma_annex2");
    const kept: Record<string, unknown>[] = [];
    for (const p of portfolios) {
      const pid = String(p.portfolio_id ?? "");
      if (!pid) continue;
      for (const dataset of ["funded", "pipeline"]) {
        const previous = existing.find((s) => s.portfolio_id === pid && s.dataset === dataset);
        if (dataset === "pipeline" && !previous) continue;
        kept.push({
          ...(previous ?? {}),
          source_key: `${pid}/${dataset}`,
          portfolio_id: pid,
          dataset,
          cadence: previous?.cadence ?? "",
          source_party: previous?.source_party ?? "",
          regime_required: dataset === "funded" && annex2,
          expected_location:
            `raw-v2/${stored.client_id || "{client}"}/${p.portfolio_type ?? "direct"}/` +
            `${dataset}/${previous?.cadence || "{cadence}"}/${pid}/{reporting_period}/`,
        });
      }
    }
    stored.answers.sources = kept;
  }

  private products(stored: StoredCase): string[] {
    return ((stored.answers.reporting ?? {}) as { products?: string[] }).products ?? [];
  }

  // -- information requests ----------------------------------------------- //

  checklist(caseId: string): ChecklistRow[] {
    const stored = this.require(caseId);
    const already = new Set(
      stored.information_requests.flatMap((r) =>
        r.items.map((i) => `${i.section}/${i.field}/${i.index}`),
      ),
    );
    return this.outstandingForClient(stored).filter(
      (row) => !already.has(`${row.section}/${row.field}/${row.index}`),
    );
  }

  private outstandingForClient(stored: StoredCase): ChecklistRow[] {
    const chosen = new Set(this.products(stored));
    const out: ChecklistRow[] = [];
    for (const section of CATALOGUE.sections) {
      if (section.repeatable) {
        const items = (stored.answers[section.key] ?? []) as Record<string, unknown>[];
        items.forEach((item, index) => {
          for (const field of section.fields) {
            if (field.source !== "client_supplied") continue;
            if (!isRequired(field, stored.answers, item)) continue;
            if (present(item[field.key])) continue;
            const name = String(item[section.item_label_field] ?? "").trim();
            out.push({
              section: section.key,
              section_label: section.label,
              field: field.key,
              label: `${field.label} — ${name || `item ${index + 1}`}`,
              help: field.help,
              index,
              scope: field.scope,
              evidence_required: field.evidence_required,
              sensitive: field.sensitive,
            });
          }
        });
        continue;
      }
      const block = (stored.answers[section.key] ?? {}) as Record<string, unknown>;
      for (const field of section.fields) {
        if (field.source !== "client_supplied") continue;
        if (section.from_regime && field.product && !chosen.has(field.product)) continue;
        const value = section.from_regime
          ? ((block[field.product] ?? {}) as Record<string, unknown>)[field.key]
          : block[field.key];
        if (!isRequired(field, stored.answers, block)) continue;
        if (present(value)) continue;
        out.push({
          section: section.key,
          section_label: section.label,
          field: field.key,
          label: field.label,
          help: field.help,
          index: null,
          scope: field.scope,
          evidence_required: field.evidence_required,
          sensitive: field.sensitive,
        });
      }
    }
    return out;
  }

  createRequest(
    caseId: string,
    items: ChecklistRow[],
    options?: { responsible_party?: string; due_date?: string; note?: string },
  ): OnboardingCase {
    const stored = this.require(caseId);
    if (!items.length) {
      throw new MockOnboardingError(
        "Choose at least one item to ask for.",
        "OPS_NOTHING_TO_REQUEST",
      );
    }
    stored.information_requests.push({
      request_id: `req_${stored.information_requests.length + 1}`,
      items,
      responsible_party: options?.responsible_party ?? "client",
      status: "open",
      requested_by: stored.updated_by,
      requested_at: new Date().toISOString(),
      sent_at: "",
      due_date: options?.due_date ?? "",
      response_note: "",
      responded_at: "",
      responded_by: "",
      evidence: [],
      reviewed_by: "",
      reviewed_at: "",
      review_note: options?.note ?? "",
    });
    this.transition(stored, "information_requested");
    return this.present(stored);
  }

  markSent(caseId: string, requestId: string): OnboardingCase {
    const stored = this.require(caseId);
    const request = stored.information_requests.find((r) => r.request_id === requestId);
    if (request) {
      request.status = "sent";
      request.sent_at = new Date().toISOString();
    }
    this.transition(stored, "awaiting_client");
    return this.present(stored);
  }

  recordResponse(
    caseId: string,
    requestId: string,
    body: {
      note?: string;
      answers?: Record<string, unknown>;
      evidence?: { name: string; reference: string }[];
    },
  ): OnboardingCase {
    const stored = this.require(caseId);
    const request = stored.information_requests.find((r) => r.request_id === requestId);
    if (request) {
      request.status = "answered";
      request.response_note = body.note ?? "";
      request.responded_at = new Date().toISOString();
      request.evidence.push(
        ...(body.evidence ?? []).map((e) => ({ ...e, received_at: new Date().toISOString() })),
      );
    }
    for (const [section, values] of Object.entries(body.answers ?? {})) {
      if (Array.isArray(values)) stored.answers[section] = values;
      else
        stored.answers[section] = {
          ...((stored.answers[section] ?? {}) as object),
          ...(values as object),
        };
    }
    this.syncDerived(stored);
    this.transition(stored, "in_review");
    return this.present(stored);
  }

  addQuestion(caseId: string, question: string): OnboardingCase {
    const stored = this.require(caseId);
    stored.open_questions.push({
      question_id: `q_${stored.open_questions.length + 1}`,
      question,
      origin: "operator",
      raised_at: new Date().toISOString(),
      raised_by: stored.updated_by,
    });
    return this.present(stored);
  }

  resolveQuestion(caseId: string, questionId: string, resolution: string): OnboardingCase {
    const stored = this.require(caseId);
    for (const q of stored.open_questions) {
      if (q.question_id === questionId) {
        q.resolved_at = new Date().toISOString();
        q.resolution = resolution;
      }
    }
    return this.present(stored);
  }

  // -- approval ------------------------------------------------------------ //

  preview(caseId: string): CasePreview {
    const stored = this.require(caseId);
    const current = this.active.get(stored.client_id) ?? null;
    const history = this.versions.get(stored.client_id) ?? [];
    return {
      ...this.present(stored),
      artefacts: this.plan(stored),
      changes: diff(current, stored.answers),
      current_version: history[0]?.version ?? 0,
      next_version: (history[0]?.version ?? 0) + 1,
      unrepresented: this.unrepresented(stored),
      generated_identifiers: this.generatedIdentifiers(stored),
      defaults_used: this.defaultsUsed(stored),
    };
  }

  submit(caseId: string): OnboardingCase {
    const stored = this.require(caseId);
    this.requireReady(stored);
    this.transition(stored, "ready_for_approval");
    return this.present(stored);
  }

  approve(caseId: string, reason: string, by: string): OnboardingCase {
    const stored = this.require(caseId);
    if (stored.status === "activated" || stored.status === "withdrawn") {
      throw new MockOnboardingError(
        `This onboarding is ${STATUS_LABELS[stored.status].toLowerCase()} and can no longer be approved.`,
        "OPS_ONBOARDING_BAD_TRANSITION",
      );
    }
    if (!reason.trim()) {
      throw new MockOnboardingError(
        "Please say why this is being approved.",
        "OPS_REASON_REQUIRED",
      );
    }
    this.requireReady(stored);
    this.transition(stored, "ready_for_approval");
    stored.approved_by = by;
    stored.approved_at = new Date().toISOString();
    stored.approval_reason = reason;
    this.transition(stored, "approved");
    return this.present(stored);
  }

  activate(caseId: string, by: string): ActivationResult {
    const stored = this.require(caseId);
    if (stored.status !== "approved") {
      throw new MockOnboardingError(
        "This onboarding has not been approved yet.",
        "OPS_ONBOARDING_NOT_APPROVED",
      );
    }
    const clientId = stored.client_id;
    const history = this.versions.get(clientId) ?? [];
    const previous = this.active.get(clientId) ?? null;
    const planned = this.plan(stored);
    const store = this.artefactStore.get(clientId) ?? new Map<string, string>();
    for (const artefact of planned) if (artefact.text) store.set(artefact.rel, artefact.text);
    this.artefactStore.set(clientId, store);

    const version: ConfigurationVersionRow = {
      version: (history[0]?.version ?? 0) + 1,
      status: "active",
      approved_by: stored.approved_by,
      approved_at: stored.approved_at,
      activated_by: by,
      activated_at: new Date().toISOString(),
      reason: stored.approval_reason,
      case_id: stored.case_id,
      case_kind: stored.kind,
      based_on_version: history[0]?.version ?? null,
      change_count: diff(previous, stored.answers).length,
      changes: diff(previous, stored.answers),
      artefacts: planned.map((a) => ({ kind: a.kind, target: a.rel, action: a.action })),
      content_hash: `sha256:mock${JSON.stringify(stored.answers).length}`,
    };
    for (const row of history) row.status = "superseded";
    this.versions.set(clientId, [version, ...history]);
    this.active.set(clientId, clone(stored.answers));
    stored.activated_version = version.version;
    this.transition(stored, "activated");
    return {
      case_id: stored.case_id,
      client_id: clientId,
      version: version.version,
      artefacts: version.artefacts,
      changes: version.changes,
    };
  }

  withdraw(caseId: string, reason: string): OnboardingCase {
    const stored = this.require(caseId);
    if (!reason.trim()) {
      throw new MockOnboardingError(
        "Please say why this is being withdrawn.",
        "OPS_REASON_REQUIRED",
      );
    }
    stored.withdrawal_reason = reason;
    this.transition(stored, "withdrawn");
    return this.present(stored);
  }

  // -- views --------------------------------------------------------------- //

  home(): OnboardingHome {
    const all = [...this.cases.values()].sort((a, b) => b.case_id.localeCompare(a.case_id));
    const rows = (...statuses: CaseStatus[]) =>
      all.filter((c) => statuses.includes(c.status)).map((c) => this.row(c));
    const activeClients = [...this.active.entries()].map(([clientId, answers]) => {
      const history = this.versions.get(clientId) ?? [];
      const client = (answers.client ?? {}) as Record<string, string>;
      return {
        client_id: clientId,
        display_name: client.client_name || clientId,
        version: history[0]?.version ?? 0,
        portfolios: ((answers.portfolios ?? []) as unknown[]).length,
        products: ((answers.reporting ?? {}) as { products?: string[] }).products ?? [],
        activated_at: history[0]?.activated_at ?? "",
        activated_by: history[0]?.activated_by ?? "",
      };
    });
    return {
      drafts: rows("draft", "changes_required"),
      awaiting_client: rows("information_requested", "awaiting_client"),
      in_review: rows("in_review", "ready_for_approval"),
      approved: rows("approved"),
      active_clients: activeClients,
      legacy_clients: [...this.legacy.keys()]
        .filter((id) => !this.active.has(id))
        .map((id) => ({ client_id: id, display_name: id })),
      recently_completed: rows("activated", "withdrawn").slice(0, 10),
    };
  }

  case(caseId: string): OnboardingCase {
    return this.present(this.require(caseId));
  }

  client(clientId: string): OnboardingClientDetail {
    const answers = this.active.get(clientId);
    const history = this.versions.get(clientId) ?? [];
    const cases = [...this.cases.values()]
      .filter((c) => c.client_id === clientId)
      .map((c) => this.row(c));
    if (!answers) {
      return {
        client_id: clientId,
        status: "not_onboarded",
        version: 0,
        answers: null,
        live_source_registrations: [],
        cases,
        history: [],
        summary: "This client has no approved configuration in Trakt.",
      };
    }
    return {
      client_id: clientId,
      status: "active",
      version: history[0]?.version ?? 0,
      activated_by: history[0]?.activated_by,
      activated_at: history[0]?.activated_at,
      reason: history[0]?.reason,
      answers: clone(answers),
      live_source_registrations: ((answers.sources ?? []) as Record<string, unknown>[]).map(
        (s) => ({
          client_id: clientId,
          source_portfolio_id: s.portfolio_id,
          dataset: s.dataset,
          frequency: s.cadence,
          source_system: s.source_party,
          regime_required: s.regime_required,
          status: "pending_review",
        }),
      ),
      artefacts: history[0]?.artefacts ?? [],
      cases,
      history,
    };
  }

  version(clientId: string, version: number): ConfigurationVersionRow {
    const row = (this.versions.get(clientId) ?? []).find((v) => v.version === version);
    if (!row) throw new MockOnboardingError("That could not be found.", "OPS_NOT_FOUND");
    return clone(row);
  }

  // -- internals ----------------------------------------------------------- //

  private require(caseId: string): StoredCase {
    const stored = this.cases.get(caseId);
    if (!stored) throw new MockOnboardingError("That could not be found.", "OPS_NOT_FOUND");
    return stored;
  }

  private requireEditable(stored: StoredCase) {
    if (["activated", "withdrawn", "approved"].includes(stored.status)) {
      throw new MockOnboardingError(
        `This onboarding is ${STATUS_LABELS[stored.status].toLowerCase()} and can no longer be edited.`,
        "OPS_ONBOARDING_LOCKED",
      );
    }
  }

  private requireReady(stored: StoredCase) {
    const blocking = this.validate(stored).filter((p) => p.severity === "blocking");
    const outstanding = stored.information_requests.filter((r) =>
      ["open", "sent"].includes(r.status),
    );
    if (blocking.length || outstanding.length) {
      throw new MockOnboardingError(
        blocking[0]?.message ?? "Information is still outstanding from the client.",
        "OPS_ONBOARDING_INCOMPLETE",
      );
    }
  }

  private transition(stored: StoredCase, to: CaseStatus) {
    if (stored.status === to) return;
    if (!TRANSITIONS[stored.status].includes(to)) {
      throw new MockOnboardingError(
        `This onboarding cannot move from ${STATUS_LABELS[stored.status].toLowerCase()} to ${STATUS_LABELS[to].toLowerCase()}.`,
        "OPS_ONBOARDING_BAD_TRANSITION",
      );
    }
    const before = stored.status;
    stored.status = to;
    this.record(stored, `status_${to}`, stored.updated_by, {
      before: { status: before },
      after: { status: to },
    });
  }

  private record(
    stored: StoredCase,
    event: string,
    actor: string,
    detail: { before?: Record<string, unknown>; after?: Record<string, unknown> },
  ) {
    stored.events.push({
      event,
      actor,
      at: new Date().toISOString(),
      reason: "",
      before: detail.before ?? {},
      after: detail.after ?? {},
      detail: {},
    });
    stored.updated_at = new Date().toISOString();
  }

  private validate(stored: StoredCase): CaseProblem[] {
    const out: CaseProblem[] = [];
    const chosen = new Set(this.products(stored));
    const push = (
      section: string,
      field: string,
      message: string,
      severity: "blocking" | "advisory" = "blocking",
      index: number | null = null,
    ) => out.push({ section, field, message, severity, index, owner: "operator" });

    for (const section of CATALOGUE.sections) {
      if (section.repeatable) {
        const items = (stored.answers[section.key] ?? []) as Record<string, unknown>[];
        if (section.min_items && items.length < section.min_items) {
          push(section.key, section.key, `At least one entry is needed under ${section.label.toLowerCase()}.`);
        }
        items.forEach((item, index) => {
          for (const field of section.fields) {
            if (!["client_supplied", "operator_supplied", "trakt_default"].includes(field.source))
              continue;
            const value = item[field.key];
            const name = String(item[section.item_label_field] ?? "").trim();
            const where = name ? ` for ${name}` : "";
            if (!present(value)) {
              if (isRequired(field, stored.answers, item))
                push(section.key, field.key, `${field.label}${where} is needed.`, "blocking", index);
              continue;
            }
            const problem = validateValue(field, value);
            if (problem) push(section.key, field.key, problem, "blocking", index);
          }
        });
        continue;
      }
      const block = (stored.answers[section.key] ?? {}) as Record<string, unknown>;
      for (const field of section.fields) {
        if (!["client_supplied", "operator_supplied", "trakt_default"].includes(field.source))
          continue;
        if (section.from_regime && field.product && !chosen.has(field.product)) continue;
        const value = section.from_regime
          ? ((block[field.product] ?? {}) as Record<string, unknown>)[field.key]
          : block[field.key];
        const key = section.from_regime ? `${field.product}.${field.key}` : field.key;
        if (!present(value)) {
          if (isRequired(field, stored.answers, block))
            push(section.key, key, `${field.label} is needed.`);
          continue;
        }
        const problem = validateValue(field, value);
        if (problem) push(section.key, key, problem);
      }
    }

    // Structural rules.
    const clientId = String(((stored.answers.client ?? {}) as Record<string, string>).client_id ?? "");
    if (
      clientId &&
      stored.kind === "new_client" &&
      stored.based_on_version === null &&
      this.active.has(clientId)
    ) {
      push("client", "client_id", `'${clientId}' is already in use by another client.`);
    }
    const seen = new Set<string>();
    ((stored.answers.portfolios ?? []) as Record<string, unknown>[]).forEach((p, index) => {
      const pid = String(p.portfolio_id ?? "");
      if (!pid) return;
      if (seen.has(pid)) push("portfolios", "portfolio_id", `'${pid}' is listed twice.`, "blocking", index);
      seen.add(pid);
    });
    if (chosen.has("esma_annex2")) {
      const entities = (stored.answers.entities ?? []) as Record<string, unknown>[];
      if (!entities.some((e) => ((e.roles ?? []) as string[]).includes("originator"))) {
        push(
          "entities",
          "roles",
          "Regulatory reporting names an originator. Give one entity the originator role.",
        );
      }
    }
    for (const q of stored.open_questions.filter((q) => !q.resolved_at)) {
      push("review", "open_questions", `An open question is unanswered: ${q.question}`, "advisory");
    }
    return out;
  }

  private present(stored: StoredCase): OnboardingCase {
    const problems = this.validate(stored);
    const blocking = problems.filter((p) => p.severity === "blocking");
    const byStep: Record<string, CaseProblem[]> = {};
    for (const step of STEPS) byStep[step.key] = [];
    for (const p of problems) (byStep[p.section] ??= []).push(p);
    const outstanding = stored.information_requests.filter((r) =>
      ["open", "sent"].includes(r.status),
    );
    return {
      ...this.row(stored),
      answers: clone(stored.answers),
      provenance: stored.provenance,
      based_on_version: stored.based_on_version,
      information_requests: stored.information_requests,
      open_questions: stored.open_questions,
      events: stored.events,
      product_eligibility: this.eligibility(stored),
      steps: STEPS.map((s) => ({ ...s, problems: (byStep[s.key] ?? []).length })),
      problems,
      blocking,
      by_step: byStep,
      client_checklist: this.checklist(stored.case_id),
      outstanding_requests: outstanding,
      unresolved_questions: stored.open_questions.filter((q) => !q.resolved_at),
      ready: blocking.length === 0 && outstanding.length === 0,
      approved_by: stored.approved_by,
      approved_at: stored.approved_at,
      approval_reason: stored.approval_reason,
      activated_version: stored.activated_version,
      withdrawal_reason: stored.withdrawal_reason,
    };
  }

  private row(stored: StoredCase): CaseRow {
    return {
      case_id: stored.case_id,
      kind: stored.kind,
      kind_label: KIND_LABELS[stored.kind],
      status: stored.status,
      status_label: STATUS_LABELS[stored.status],
      client_id: stored.client_id,
      client_name: stored.client_name || stored.client_id || "Not yet named",
      portfolios: ((stored.answers.portfolios ?? []) as unknown[]).length,
      outstanding_requests: stored.information_requests.filter((r) =>
        ["open", "sent"].includes(r.status),
      ).length,
      unresolved_questions: stored.open_questions.filter((q) => !q.resolved_at).length,
      updated_at: stored.updated_at,
      updated_by: stored.updated_by,
      created_by: stored.created_by,
    };
  }

  private eligibility(stored: StoredCase) {
    const chosen = new Set(this.products(stored));
    const portfolios = (stored.answers.portfolios ?? []) as Record<string, unknown>[];
    const out: OnboardingCase["product_eligibility"] = {};
    for (const [key, product] of Object.entries(CATALOGUE.regime_products)) {
      if (product.always_applies) {
        out[key] = {
          label: product.label,
          applies: true,
          eligible: true,
          derived: true,
          reason: product.derived_from ?? "",
        };
        continue;
      }
      const eligible = portfolios.length > 0;
      out[key] = {
        label: product.label,
        applies: chosen.has(key),
        eligible,
        derived: false,
        reason: eligible
          ? `Supported by ${portfolios.map((p) => p.portfolio_id).join(", ")}.`
          : "No portfolio has been added yet.",
      };
    }
    return out;
  }

  private unrepresented(stored: StoredCase) {
    const out: CasePreview["unrepresented"] = [];
    for (const section of CATALOGUE.sections) {
      for (const field of section.fields) {
        if (field.writes_to !== "onboarding_record") continue;
        if (!["client_supplied", "operator_supplied", "trakt_default"].includes(field.source))
          continue;
        out.push({
          key: field.key,
          label: field.label,
          reason:
            "Held in the governed onboarding record. No existing configuration artefact represents it.",
        });
      }
    }
    for (const key of this.products(stored)) {
      for (const entry of CATALOGUE.regime_products[key]?.unrepresented ?? []) {
        out.push({ ...entry, product: CATALOGUE.regime_products[key].label });
      }
    }
    return out;
  }

  private generatedIdentifiers(stored: StoredCase) {
    const out = [{ label: "Onboarding reference", value: stored.case_id }];
    for (const e of (stored.answers.entities ?? []) as Record<string, unknown>[]) {
      out.push({
        label: `Entity reference — ${String(e.legal_name ?? "")}`,
        value: String(e.entity_id ?? ""),
      });
    }
    for (const s of (stored.answers.sources ?? []) as Record<string, unknown>[]) {
      if (s.expected_location)
        out.push({
          label: `Delivery location — ${String(s.portfolio_id)} ${String(s.dataset)}`,
          value: String(s.expected_location),
        });
    }
    return out;
  }

  private defaultsUsed(stored: StoredCase) {
    const out: CasePreview["defaults_used"] = [];
    for (const section of CATALOGUE.sections) {
      if (section.repeatable || section.from_regime) continue;
      const block = (stored.answers[section.key] ?? {}) as Record<string, unknown>;
      for (const field of section.fields) {
        if (field.default === null || field.default === undefined) continue;
        if (!present(block[field.key]))
          out.push({ label: field.label, value: String(field.default), section: section.label });
      }
    }
    return out;
  }

  private plan(stored: StoredCase): PlannedArtefact[] {
    const clientId = stored.client_id || "CLIENT";
    const store = this.artefactStore.get(clientId);
    const products = this.products(stored);
    const out: PlannedArtefact[] = [];

    const add = (kind: string, rel: string, label: string, text: string, summary: string[]) => {
      const previous = store?.get(rel);
      out.push({
        kind,
        rel,
        label,
        action: previous === undefined ? "create" : previous === text ? "unchanged" : "update",
        text,
        previous_text: previous ?? "",
        records: [],
        summary,
      });
    };

    add(
      "client_config",
      `config/client/config_client_${clientId}.yaml`,
      "Client configuration",
      renderClientConfig(stored.answers, products),
      [
        `Client identity for ${stored.client_name || clientId}.`,
        "Portfolio jurisdiction and reporting currency.",
        `Reporting products: ${products.join(", ") || "management information only"}.`,
      ],
    );
    if (products.includes("investor_reporting")) {
      add(
        "annex12_config",
        `config/client/config_client_${clientId}_annex12.yaml`,
        "Investor report configuration",
        "# GENERATED\nannex12:\n  deal: {}\n",
        ["Standing investor-report fields."],
      );
    }
    add(
      "portfolio_registry",
      `config/client/portfolio_registry_${clientId}.yaml`,
      "Portfolio metadata",
      renderPortfolios(stored.answers, clientId),
      [`${((stored.answers.portfolios ?? []) as unknown[]).length} portfolios described.`],
    );
    add(
      "tenancy",
      `config/client/client_index_${clientId}.yaml`,
      "Client index",
      renderIndex(stored.answers, clientId),
      ["Client index entry."],
    );

    const records = ((stored.answers.sources ?? []) as Record<string, unknown>[]).map((s) => {
      const portfolio = ((stored.answers.portfolios ?? []) as Record<string, unknown>[]).find(
        (p) => p.portfolio_id === s.portfolio_id,
      );
      return {
        client_id: clientId,
        source_portfolio_id: String(s.portfolio_id),
        dataset: String(s.dataset),
        frequency: String(s.cadence ?? ""),
        source_system: String(s.source_party ?? ""),
        regime_required: Boolean(s.regime_required),
        status: "pending_review",
        action: "create",
        portfolio_label: String(portfolio?.display_name ?? ""),
        expected_location: String(s.expected_location ?? ""),
      };
    });
    out.push({
      kind: "source_registry",
      rel: "source registrations",
      label: "Source registrations",
      action: records.length ? "create" : "unchanged",
      text: "",
      previous_text: "",
      records,
      summary: records.map(
        (r) =>
          `${clientId}/${r.source_portfolio_id}/${r.dataset}/${r.frequency} registered, awaiting its first pack.`,
      ),
    });
    return out;
  }
}

// --------------------------------------------------------------------------- //

const BANNER = `# GENERATED by the Trakt Operations Control Centre — Client Onboarding.
# This file is a governed artefact. Change it through Client Onboarding,
# not by hand: a hand edit is not versioned, not attributed and not
# audited, and the next approved change will overwrite it.
`;

function renderClientConfig(answers: CaseAnswers, products: string[]): string {
  const client = (answers.client ?? {}) as Record<string, string>;
  const contacts = (answers.contacts ?? {}) as Record<string, string>;
  const portfolios = (answers.portfolios ?? []) as Record<string, unknown>[];
  const regime = ((answers.regime ?? {}) as Record<string, Record<string, string>>).esma_annex2 ?? {};
  const lines = [
    BANNER,
    "client:",
    `  display_name: ${client.client_name ?? ""}`,
    `  client_id: ${client.client_id ?? ""}`,
    `  time_zone: ${client.time_zone ?? ""}`,
    `  environment: ${client.environment ?? "production"}`,
    "  reporting_contact:",
    `    name: ${contacts.reporting_contact_name ?? ""}`,
    `    email: ${contacts.reporting_contact_email ?? ""}`,
    "  operational_contact:",
    `    name: ${contacts.operational_contact_name ?? ""}`,
    `    email: ${contacts.operational_contact_email ?? ""}`,
    "portfolio:",
    `  country: ${client.jurisdiction ?? ""}`,
    `  base_currency: ${client.reporting_currency ?? ""}`,
    `  asset_class: ${String(portfolios[0]?.asset_class ?? "")}`,
  ];
  if (Object.keys(regime).length) {
    lines.push("defaults:");
    for (const [key, value] of Object.entries(regime)) lines.push(`  ${key}: ${value}`);
  }
  if (products.includes("esma_annex2")) {
    lines.push("supported_regimes:", "  - ESMA_Annex2", "default_regime: ESMA_Annex2", "regime: ESMA_Annex2");
  }
  lines.push(
    "pipeline:",
    "  loan_engine_enabled: false",
    "  mi_enabled: true",
    `  esma_enabled: ${products.includes("esma_annex2")}`,
  );
  return lines.join("\n") + "\n";
}

function renderPortfolios(answers: CaseAnswers, clientId: string): string {
  const lines = [BANNER, "clients:", `  ${clientId}:`, "    portfolios:"];
  for (const p of (answers.portfolios ?? []) as Record<string, unknown>[]) {
    lines.push(
      `      - source_portfolio_id: ${String(p.portfolio_id)}`,
      `        source_portfolio_type: ${String(p.portfolio_type ?? "")}`,
      `        source_portfolio_label: ${String(p.display_name ?? "")}`,
      `        originates: ${String(p.originates ?? "")}`,
    );
  }
  return lines.join("\n") + "\n";
}

function renderIndex(answers: CaseAnswers, clientId: string): string {
  const client = (answers.client ?? {}) as Record<string, string>;
  const portfolios = (answers.portfolios ?? []) as Record<string, unknown>[];
  return (
    [
      BANNER,
      "tenants:",
      `  ${clientId}:`,
      `    display_name: ${client.client_name ?? ""}`,
      `    default_portfolio: ${clientId}`,
      "    portfolios:",
      `      - ${clientId}`,
      ...portfolios.map((p) => `      - ${String(p.portfolio_id)}`),
    ].join("\n") + "\n"
  );
}

function flatten(node: unknown, prefix = "", out: Record<string, string> = {}) {
  if (Array.isArray(node)) {
    if (node.every((v) => typeof v !== "object" || v === null)) out[prefix] = node.join(", ");
    else
      node.forEach((v, i) => {
        const key =
          v && typeof v === "object"
            ? String(
                (v as Record<string, unknown>).portfolio_id ??
                  (v as Record<string, unknown>).source_key ??
                  (v as Record<string, unknown>).legal_name ??
                  i,
              )
            : String(i);
        flatten(v, `${prefix}[${key}]`, out);
      });
  } else if (node && typeof node === "object") {
    for (const [k, v] of Object.entries(node)) flatten(v, prefix ? `${prefix}.${k}` : k, out);
  } else {
    out[prefix] = node === null || node === undefined ? "" : String(node);
  }
  return out;
}

const SUPPRESSED = ["entity_id", "source_key", "expected_location"];

const LABELS: Record<string, string> = {
  "client.client_name": "Client name",
  "client.client_id": "Client identifier",
  "client.jurisdiction": "Jurisdiction",
  "client.reporting_currency": "Reporting currency",
  "reporting.products": "Reporting products",
};

export function diff(before: CaseAnswers | null, after: CaseAnswers) {
  const old = before ? flatten(before) : {};
  const now = flatten(after);
  const paths = Array.from(new Set([...Object.keys(old), ...Object.keys(now)]))
    .filter((p) => !SUPPRESSED.some((s) => p === s || p.endsWith(`.${s}`)))
    .sort();
  const changes = [];
  for (const path of paths) {
    const was = old[path] ?? "";
    const next = now[path] ?? "";
    if (was === next || (!was && !next)) continue;
    changes.push({
      path,
      label: LABELS[path] ?? path.replace(/_/g, " ").replace(/\./g, " → "),
      before: was,
      after: next,
      action: !was ? "added" : !next ? "removed" : "changed",
    });
  }
  return changes;
}
