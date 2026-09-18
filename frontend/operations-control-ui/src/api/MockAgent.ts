import { CATALOGUE } from "./mockCatalogue";
import { MockOnboarding } from "./MockOnboarding";
import { OpsError } from "./OpsClient";
import type {
  ActivationIntent,
  ActivationResult,
  AgentActivation,
  AgentClassification,
  AgentAudit,
  AgentChecklist,
  AgentMeta,
  AgentPack,
  AgentPreview,
  AgentProposal,
  AgentReadinessPackage,
  AgentReviewPackage,
  AgentStatus,
  AgentTurn,
  CaseSummary,
  ClientFormGroup,
  ClientFormStep,
  ClientFormView,
  DecisionCard,
  ExecutionFacts,
  FieldCategory,
  FieldClassification,
  LifecycleState,
  MappingOverview,
  MappingRow,
  PackQuestion,
  PackReceipt,
  PackSection,
  ScenarioSummary,
  SyntheticRunDoc,
  AgentMail,
  AgentMailMessage,
  MailIngestOutcome,
} from "./agentTypes";
import type { ChecklistRow, OnboardingCase } from "./onboardingTypes";

/**
 * A stateful stand-in for the OCC Agent API, mirroring the backend's structure
 * as well as its behaviour. Used by `VITE_OPS_MODE=mock` and by the tests.
 *
 * The important thing about it is what it does NOT do: it has no onboarding
 * model of its own. It drives its own `MockOnboarding` instance — exactly as the
 * server drives the real `OnboardingService` against an isolated container — so
 * a practice case here is a genuine onboarding case with a genuine status, and
 * the two lifecycles are as separate in the mock as they are in the product.
 *
 * It never activates. Activation is the one call that creates a client
 * configuration, and a practice case must reach readiness having created
 * nothing.
 */

const S = {
  AWAITING_ONBOARDING: "AWAITING_ONBOARDING",
  PACK_DRAFTED: "PACK_DRAFTED",
  PACK_REVIEW_REQUIRED: "PACK_REVIEW_REQUIRED",
  PACK_APPROVED_TO_SEND: "PACK_APPROVED_TO_SEND",
  PACK_SENT: "PACK_SENT",
  READY_TO_RUN: "READY_TO_RUN",
  SYNTHETIC_ONBOARDING_RUNNING: "SYNTHETIC_ONBOARDING_RUNNING",
  EXCEPTIONS_REQUIRE_INPUT: "EXCEPTIONS_REQUIRE_INPUT",
  SYNTHETIC_ONBOARDING_PASSED: "SYNTHETIC_ONBOARDING_PASSED",
  ORCHESTRATION_PLAN_GENERATED: "ORCHESTRATION_PLAN_GENERATED",
  EXECUTION_APPROVAL_REQUIRED: "EXECUTION_APPROVAL_REQUIRED",
  READY_FOR_EXECUTION: "READY_FOR_EXECUTION",
  READY_FOR_REVIEW: "READY_FOR_REVIEW",
  APPROVED_FOR_ACTIVATION: "APPROVED_FOR_ACTIVATION",
  ACTIVATION_CONFIRMATION_REQUIRED: "ACTIVATION_CONFIRMATION_REQUIRED",
  ACTIVATING: "ACTIVATING",
  INGESTION_STARTED: "INGESTION_STARTED",
  ACTIVATION_FAILED: "ACTIVATION_FAILED",
  BLOCKED: "BLOCKED",
  CANCELLED: "CANCELLED",
} as const;

const STATE_LABELS: Record<string, string> = {
  [S.AWAITING_ONBOARDING]: "Working through onboarding",
  [S.PACK_DRAFTED]: "Onboarding pack drafted",
  [S.PACK_REVIEW_REQUIRED]: "Pack needs your review",
  [S.PACK_APPROVED_TO_SEND]: "Pack approved to send",
  [S.PACK_SENT]: "Pack issued to the client",
  [S.READY_TO_RUN]: "Ready to run",
  [S.SYNTHETIC_ONBOARDING_RUNNING]: "Synthetic onboarding running",
  [S.EXCEPTIONS_REQUIRE_INPUT]: "Exceptions need your input",
  [S.SYNTHETIC_ONBOARDING_PASSED]: "Synthetic onboarding passed",
  [S.ORCHESTRATION_PLAN_GENERATED]: "Execution plan prepared",
  [S.EXECUTION_APPROVAL_REQUIRED]: "Readiness needs your approval",
  [S.READY_FOR_EXECUTION]: "READY_FOR_EXECUTION",
  [S.READY_FOR_REVIEW]: "Ready for review and approval",
  [S.APPROVED_FOR_ACTIVATION]: "Approved for activation",
  [S.ACTIVATION_CONFIRMATION_REQUIRED]: "Confirm activation",
  [S.ACTIVATING]: "Activating",
  [S.INGESTION_STARTED]: "Ingestion started",
  [S.ACTIVATION_FAILED]: "Activation failed",
  [S.BLOCKED]: "Blocked",
  [S.CANCELLED]: "Cancelled",
};

const ORDER = Object.values(S);

/** Onboarding actions are legal wherever the CASE allows them, so the execution
 *  table does not gate them — exactly as in `occ_agent.states`. */
const ONBOARDING_ACTIONS = [
  "answer_onboarding_question",
  "request_client_information",
  "record_client_response",
  "submit_for_approval",
  "approve_onboarding",
  "request_changes",
  "withdraw_case",
];

const PACK_ACTIONS = ["draft_onboarding_pack", "register_synthetic_artefact", "cancel_run"];

const ALLOWED: Record<string, string[]> = {
  [S.AWAITING_ONBOARDING]: PACK_ACTIONS,
  [S.PACK_DRAFTED]: [...PACK_ACTIONS, "approve_pack_to_send"],
  [S.PACK_REVIEW_REQUIRED]: [...PACK_ACTIONS, "approve_pack_to_send"],
  [S.PACK_APPROVED_TO_SEND]: [
    "send_onboarding_pack",
    "register_synthetic_artefact",
    "cancel_run",
  ],
  [S.PACK_SENT]: PACK_ACTIONS,
  [S.READY_TO_RUN]: ["run_synthetic_onboarding", "register_synthetic_artefact", "cancel_run"],
  [S.SYNTHETIC_ONBOARDING_RUNNING]: ["cancel_run"],
  [S.EXCEPTIONS_REQUIRE_INPUT]: [
    "resolve_decision",
    "acknowledge_exception",
    "run_synthetic_onboarding",
    "cancel_run",
  ],
  [S.SYNTHETIC_ONBOARDING_PASSED]: ["generate_orchestration_plan", "cancel_run"],
  [S.ORCHESTRATION_PLAN_GENERATED]: ["approve_execution_readiness", "cancel_run"],
  [S.EXECUTION_APPROVAL_REQUIRED]: ["approve_execution_readiness", "cancel_run"],
  // A waypoint, not the finish: what follows is the human decision about the
  // real thing.
  [S.READY_FOR_EXECUTION]: ["request_activation", "cancel_run"],
  [S.READY_FOR_REVIEW]: ["approve_activation", "cancel_run"],
  [S.APPROVED_FOR_ACTIVATION]: ["confirm_activation", "cancel_run"],
  [S.ACTIVATION_CONFIRMATION_REQUIRED]: ["confirm_activation", "cancel_run"],
  [S.ACTIVATING]: [],
  [S.INGESTION_STARTED]: [],
  [S.ACTIVATION_FAILED]: ["confirm_activation", "cancel_run"],
  [S.BLOCKED]: [
    "resolve_decision",
    "register_synthetic_artefact",
    "run_synthetic_onboarding",
    "draft_onboarding_pack",
    "cancel_run",
  ],
  [S.CANCELLED]: [],
};

const POLICY = {
  runtime_mode: "synthetic",
  allow_external_email: false,
  allow_live_blob_write: false,
  allow_live_pipeline_trigger: false,
  allow_production_config_write: false,
  allow_publish: false,
  allow_live_case_access: false,
  allow_activate_configuration: false,
};

const LEI = "894500SYNTHETIC00042";

/** What a practice client sends back, keyed by the catalogue's section.field. */
function clientResponse(domain: string): Record<string, string> {
  return {
    "entities.lei": LEI,
    "entities.country_of_establishment": "GB",
    "contacts.reporting_contact_name": "Practice Reporting Contact",
    "contacts.reporting_contact_email": `reporting@${domain}`,
    "contacts.operational_contact_name": "Practice Operations Contact",
    "contacts.operational_contact_email": `operations@${domain}`,
    "portfolios.portfolio_type": "direct",
    // The business conventions behind the numbers. Trakt cannot read these out
    // of a file, so onboarding asks and a scenario answers.
    "data_semantics.balance_definition":
      "Current principal only. Accrued interest is reported separately.",
    "data_semantics.gross_net_convention": "Gross of fees and charges.",
    "data_semantics.units_and_currency": "Units, GBP.",
    "data_semantics.cut_off_convention": "Calendar month end.",
    "data_semantics.measure_basis": "point_in_time",
  };
}

const SCENARIOS: ScenarioSummary[] = [
  {
    fixture_id: "scenario_a_clean",
    label: "A — Clean onboarding",
    description:
      "Complete artefacts, valid configuration and high-confidence mappings. Reaches READY_FOR_EXECUTION with no blocking control.",
    instruction:
      "Onboard Northstar Lending. It is a UK equity-release portfolio. They require monthly management information. Portfolio id direct_101. First reporting date 2026-06-30.",
    files: [
      { filename: "northstar_loan_extract_202606.csv", artefact_type: "loan_extract", bytes: 4096 },
    ],
    client_response: Object.keys(clientResponse("northstar.example")),
    reporting_period: "2026-06-30",
    expected_state: S.READY_FOR_EXECUTION,
    expected_onboarding_status: "approved",
    expected_human_steps: [
      "ask the client for what is outstanding",
      "record their response",
      "approve the onboarding",
      "approve readiness",
    ],
    demonstrates: "The whole operating process end to end with nothing in the way.",
  },
  {
    fixture_id: "scenario_b_ambiguous_mapping",
    label: "B — Ambiguous mapping",
    description:
      "Two source columns resolve to the same canonical field. The run halts for a human decision, then reaches READY_FOR_EXECUTION.",
    instruction:
      "Onboard Harbour Point Capital. UK equity release. Monthly management information. Portfolio id direct_102. First reporting date 2026-06-30.",
    files: [
      {
        filename: "harbourpoint_loan_extract_202606.csv",
        artefact_type: "loan_extract",
        bytes: 4400,
      },
    ],
    client_response: Object.keys(clientResponse("harbourpoint.example")),
    reporting_period: "2026-06-30",
    expected_state: S.READY_FOR_EXECUTION,
    expected_onboarding_status: "approved",
    expected_human_steps: ["settle the ambiguous mapping"],
    demonstrates: "A control the engine cannot settle alone.",
  },
  {
    fixture_id: "scenario_c_missing_artefact",
    label: "C — Missing mandatory artefact",
    description:
      "A required input role is absent, so the practice run is blocked. The onboarding itself still completes.",
    instruction:
      "Onboard Kestrel Mutual. UK equity release. Monthly management information. Portfolio id direct_103. First reporting date 2026-06-30.",
    files: [
      {
        filename: "kestrel_property_extract_202606.csv",
        artefact_type: "property_extract",
        bytes: 1200,
      },
    ],
    client_response: Object.keys(clientResponse("kestrel.example")),
    reporting_period: "2026-06-30",
    expected_state: S.BLOCKED,
    expected_onboarding_status: "approved",
    expected_human_steps: ["provide the missing loan tape"],
    demonstrates: "The configured input requirements refusing an incomplete pack.",
  },
  {
    fixture_id: "scenario_d_product_information_gap",
    label: "D — Product information gap",
    description:
      "Core onboarding proceeds, but the selected product asks a question the client has not answered, so approval is refused.",
    instruction:
      "Onboard Aldermere Advances. UK equity release. Management information and investor reporting. Portfolio id direct_104. First reporting date 2026-06-30.",
    files: [
      { filename: "aldermere_loan_extract_202606.csv", artefact_type: "loan_extract", bytes: 4096 },
    ],
    client_response: Object.keys(clientResponse("aldermere.example")),
    reporting_period: "2026-06-30",
    expected_state: S.AWAITING_ONBOARDING,
    expected_onboarding_status: "in_review",
    expected_human_steps: ["chase the product question they left unanswered"],
    demonstrates: "A product-specific requirement, loaded through the product framework.",
  },
  {
    fixture_id: "scenario_e_business_rule_failure",
    label: "E — Material business-rule failure",
    description:
      "Canonical transformation succeeds; the deterministic business rules then fail at BLOCKING materiality.",
    instruction:
      "Onboard Brackenfield Finance. UK equity release. Monthly management information. Portfolio id direct_105. First reporting date 2026-06-30.",
    files: [
      {
        filename: "brackenfield_loan_extract_202606.csv",
        artefact_type: "loan_extract",
        bytes: 4096,
      },
    ],
    client_response: Object.keys(clientResponse("brackenfield.example")),
    reporting_period: "2026-06-30",
    expected_state: S.BLOCKED,
    expected_onboarding_status: "approved",
    expected_human_steps: [],
    demonstrates: "A deterministic blocker that natural language cannot bypass.",
  },
];

/** Which domain each scenario's practice client answers from. */
const SCENARIO_DOMAINS: Record<string, string> = {
  scenario_a_clean: "northstar.example",
  scenario_b_ambiguous_mapping: "harbourpoint.example",
  scenario_c_missing_artefact: "kestrel.example",
  scenario_d_product_information_gap: "aldermere.example",
  scenario_e_business_rule_failure: "brackenfield.example",
};

function lifecycle(current?: string, reached?: Set<string>): LifecycleState[] {
  return ORDER.map((state) => ({
    state,
    label: STATE_LABELS[state],
    permitted_prior: [],
    required_inputs: [],
    automatic_actions: [],
    deterministic_controls: [],
    required_approvals: [],
    allowed_human_actions: ALLOWED[state] ?? [],
    next_states: [],
    blocking_conditions: [],
    occ_stage: "",
    terminal: state === S.READY_FOR_EXECUTION || state === S.CANCELLED,
    reached: reached?.has(state) ?? false,
    current: state === current,
  }));
}

/**
 * What became of each source column, as the server projects it.
 *
 * A double for `occ_agent/mapping_view.overview`, kept in step with it by
 * `MappingTable.test.tsx` rather than by hope: the trusted-tier set and the
 * confidence threshold are the engine's, and a screen that classified rows
 * differently from the engine would tell an operator a mapping had been
 * checked when it had not.
 */
function mappingOverview(doc: SyntheticRunDoc): MappingOverview {
  // {column: [decision_id, decision_type]} — faithful to
  // `mapping_view._open_decision_by_column`. The TYPE matters: a proposal is
  // approved with the set, a question is answered on its own, and a table that
  // cannot tell them apart puts seventy clean columns in the same queue as the
  // three that are genuinely unresolved.
  const byColumn = new Map<string, [string, string]>();
  for (const decision of doc.open_decisions) {
    if (decision.status !== "open") continue;
    const subject = decision.subject as {
      source_column?: string;
      source_columns?: string[];
      decision_type?: string;
    };
    const entry: [string, string] = [decision.decision_id,
                                     String(subject?.decision_type ?? "")];
    for (const name of [subject?.source_column, ...(subject?.source_columns ?? [])]) {
      const key = String(name ?? "").trim().toLowerCase();
      if (key && !byColumn.has(key)) byColumn.set(key, entry);
    }
  }

  const rows: MappingRow[] = (doc.mapping_report ?? []).map((raw) => {
    const r = raw as Record<string, unknown>;
    const tier = String(r.tier ?? "");
    const canonical = String(r.canonical_field ?? "");
    const rawConfidence = Number(r.confidence);
    const confidence = Number.isFinite(rawConfidence) ? rawConfidence : null;
    const column = String(r.source_column ?? "");
    const [decisionId, decisionType] = byColumn.get(column.toLowerCase()) ?? ["", ""];
    const primary = r.primary !== false;
    // What a model proposed for a column the deterministic tiers could not
    // place. Never a mapping — it stands until a person confirms it.
    const suggested = String(r.llm_field ?? "");
    let state: MappingRow["state"];
    // An open decision wins over the tier: an ambiguity is raised after both
    // rows are written, at whatever tier they matched at, so reading the tier
    // alone would call a blocked column "matched automatically".
    if (decisionId) state = decisionType === "mapping_proposal" ? "proposed" : "needs_you";
    else if (tier === "operator_approved") state = "confirmed";
    else if (tier === "unreadable") state = "unreadable";
    else if (!canonical) state = "unused";
    else if (TRUSTED_TIERS.has(tier) || (confidence !== null && confidence >= LOW_CONFIDENCE))
      state = "automatic";
    // A weak match outside the primary tape raised no question, so calling it
    // "Needs you" would point at a decision that does not exist.
    else state = primary ? "needs_you" : "unchecked";
    return {
      source_file: String(r.source_file ?? ""),
      source_column: column,
      canonical_field: canonical,
      field_label: canonical.replace(/_/g, " "),
      tier,
      tier_label: TIER_LABELS[tier] ?? tier.replace(/_/g, " "),
      confidence,
      note: String(r.note ?? ""),
      state,
      state_label: MAPPING_STATE_LABELS[state],
      decision_id: decisionId,
      primary,
      // Faithful to `operations_control.occ_agent.mapping_view.overview`: an
      // operator's own answer, then Trakt's deterministic matching, then a
      // model's proposal for a column nothing matched. A mock that omits the
      // basis lets a screen be built against a field the server sends and the
      // double does not, which is how the last mock/server drift got through.
      basis: tier === "operator_approved"
        ? "you"
        : canonical
          ? "deterministic"
          : suggested
            ? "model"
            : "",
      basis_label: tier === "operator_approved"
        ? "You confirmed it"
        : canonical
          ? "Trakt's own matching"
          : suggested
            ? "Suggested by a model, not yet confirmed"
            : "",
      suggested_field: suggested,
      suggested_label: suggested.replace(/_/g, " "),
      suggested_reason: String(r.llm_reasoning ?? ""),
    };
  });

  rows.sort(
    (a, b) =>
      Number(!a.primary) - Number(!b.primary) ||
      a.source_file.localeCompare(b.source_file) ||
      MAPPING_STATE_ORDER.indexOf(a.state) - MAPPING_STATE_ORDER.indexOf(b.state) ||
      a.source_column.toLowerCase().localeCompare(b.source_column.toLowerCase()),
  );

  const counts: Record<string, number> = {};
  for (const state of MAPPING_STATE_ORDER) {
    counts[state] = rows.filter((r) => r.state === state).length;
  }
  counts.columns = rows.length;
  counts.mapped = counts.confirmed + counts.automatic;

  const files: MappingOverview["files"] = [];
  for (const row of rows) {
    if (!row.source_file || files.some((f) => f.name === row.source_file)) continue;
    files.push({
      name: row.source_file,
      primary: row.primary,
      columns: rows.filter((r) => r.source_file === row.source_file).length,
    });
  }
  return {
    rows,
    counts,
    files,
    proposed: counts.proposed ?? 0,
    blocking_questions: counts.needs_you ?? 0,
  };
}

interface StoredRun {
  doc: SyntheticRunDoc;
  reached: Set<string>;
  scenario: string;
  audit: Record<string, unknown>[];
}

/**
 * What the header mapper made of a small but representative tape.
 *
 * Deliberately covers every state the table can show — matched exactly,
 * matched by alias, too weak to use, nothing matched, and one an operator
 * confirmed — because a fixture that only contains clean rows lets a screen
 * that renders only clean rows pass.
 */
const MAPPING_REPORT: Record<string, unknown>[] = [
  { source_file: "loan_tape.csv", source_column: "loan_id", canonical_field: "loan_id",
    tier: "exact", confidence: 1.0, note: "", primary: true },
  { source_file: "loan_tape.csv", source_column: "Current Balance",
    canonical_field: "current_principal_balance", tier: "alias", confidence: 1.0,
    note: "", primary: true },
  { source_file: "loan_tape.csv", source_column: "Int Rate",
    canonical_field: "interest_rate", tier: "normalized", confidence: 1.0,
    note: "", primary: true },
  { source_file: "loan_tape.csv", source_column: "Prop Val",
    canonical_field: "property_value", tier: "operator_approved", confidence: 1.0,
    note: "confirmed by an operator", primary: true },
  { source_file: "loan_tape.csv", source_column: "Val Dt",
    canonical_field: "valuation_date", tier: "fuzz_token_set", confidence: 0.62,
    note: "below the confidence threshold", primary: true },
  // Nothing matched it, and a model proposed something. That is the ordinary
  // shape of a real tape now the model is wired into the mapping stage, and a
  // fixture without one lets a table that drops every proposal pass.
  { source_file: "loan_tape.csv", source_column: "Internal Ref", canonical_field: "",
    tier: "unmapped", confidence: 0.0, note: "", primary: true,
    llm_field: "loan_identifier", llm_confidence: 0.81,
    llm_reasoning: "An internal loan reference." },
  // A second file in the pack. The canonical tape is not built from it, so a
  // weak match here raises no question — and a fixture with only one file lets
  // a table that silently drops the others pass.
  { source_file: "property_tape.csv", source_column: "property_value",
    canonical_field: "property_value", tier: "exact", confidence: 1.0,
    note: "", primary: false },
  { source_file: "property_tape.csv", source_column: "Prp Ref",
    canonical_field: "property_reference", tier: "fuzz_ratio_norm", confidence: 0.58,
    note: "below the confidence threshold", primary: false },
];

/** Mirrors `occ_agent/mapping_view.TIER_LABELS`. */
const TIER_LABELS: Record<string, string> = {
  exact: "The column is named exactly as the field is",
  normalized: "The names match once case and punctuation are ignored",
  alias: "A known alias for this field",
  token_set: "The words overlap, but the names are not the same",
  fuzz_token_set: "The words are similar, not the same",
  fuzz_ratio_norm: "The names are spelled similarly",
  unmapped: "Nothing Trakt reports on resembles this column",
  empty: "The column has no name",
  operator_approved: "You said so",
  unreadable: "The file could not be read",
};

const MAPPING_STATE_LABELS: Record<string, string> = {
  needs_you: "Needs you",
  proposed: "Proposed",
  unreadable: "Could not be read",
  unchecked: "Weak match, nothing asked",
  unused: "Not used",
  confirmed: "You confirmed it",
  automatic: "Matched automatically",
};

const MAPPING_STATE_ORDER = [
  "needs_you",
  "proposed",
  "unreadable",
  "unchecked",
  "unused",
  "confirmed",
  "automatic",
];

/** Mirrors `execution._TRUSTED_TIERS` and `execution.LOW_CONFIDENCE`. */
const TRUSTED_TIERS = new Set(["exact", "normalized", "alias"]);
const LOW_CONFIDENCE = 0.9;

/**
 * One proposed mapping, as a first onboarding raises it.
 *
 * Faithful to `execution._mapping_proposal`: the source column is fixed and
 * the answer names the field, the subject carries the decision type so
 * promotion and the table can both read it, and it BLOCKS — a first delivery
 * does not move until a person has said these are right for this client.
 */
function proposalFor(row: { source_column: string; canonical_field: string;
                            tier: string; confidence: number }): DecisionCard {
  const field = row.canonical_field.replace(/_/g, " ");
  return {
    decision_id: `map_${row.source_column.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
    kind: "field_mapping",
    title: `Confirm '${row.source_column}' reads as ${field}`,
    question: `'${row.source_column}' reads as ${field}. This is the first delivery from this client, so it is proposed rather than applied.`,
    blocking: true,
    status: "open",
    issue: `'${row.source_column}' reads as ${field}.`,
    evidence: [],
    recommendation: row.canonical_field,
    recommendation_source: "deterministic",
    confidence: row.confidence,
    materiality: "BLOCKING",
    downstream_consequence:
      "The practice run cannot continue until the mappings are approved.",
    options: [
      { value: "accept_mapping", label: "Accept this mapping" },
      { value: "mark_unavailable", label: "Do not use this column" },
    ],
    subject: {
      decision_type: "mapping_proposal",
      source_column: row.source_column,
      target_field: row.canonical_field,
    },
  };
}

/**
 * One weak match, raised as its own question.
 *
 * Faithful to `execution._mapping_decision`. The mock used to seed none of
 * these, so a column classified "Needs you" from its tier alone had no card to
 * answer — which was harmless while nothing waited on it, and became a dead
 * end the moment approval of the set was gated on the questions being cleared.
 */
function confirmationFor(row: { source_column: string; canonical_field: string;
                                confidence: number }): DecisionCard {
  const field = row.canonical_field.replace(/_/g, " ");
  return {
    decision_id: `map_${row.source_column.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
    kind: "field_mapping",
    title: `Confirm how '${row.source_column}' should be read`,
    question: `'${row.source_column}' looks like ${field}, but not clearly enough to use without confirmation.`,
    blocking: true,
    status: "open",
    issue: `'${row.source_column}' looks like ${field}.`,
    evidence: [],
    recommendation: row.canonical_field,
    recommendation_source: "deterministic",
    confidence: row.confidence,
    materiality: "BLOCKING",
    downstream_consequence:
      "The practice run cannot continue until this is answered.",
    options: [
      { value: "accept_mapping", label: "Accept this mapping" },
      { value: "mark_unavailable", label: "Do not use this column" },
    ],
    subject: {
      decision_type: "mapping_confirmation",
      source_column: row.source_column,
      target_field: row.canonical_field,
    },
  };
}

const AMBIGUOUS_DECISION: DecisionCard = {
  decision_id: "amb_current_principal_balance",
  kind: "field_mapping",
  title: "Confirm where 'Current Balance' belongs",
  question:
    "Two source fields could represent current outstanding balance. Confirm this mapping or provide a correction.",
  blocking: true,
  status: "open",
  issue: "2 source fields could represent current principal balance.",
  evidence: [
    {
      label: "What Trakt found",
      kind: "text",
      data: {
        issue: "Two columns claim the same canonical field.",
        detail:
          "'Current Balance' carries values for 24 of 24 records; 'Principal Balance' carries values for 21 of 24 records.",
      },
    },
  ],
  recommendation: "Current Balance",
  recommendation_source: "deterministic",
  confidence: 0.5,
  materiality: "BLOCKING",
  downstream_consequence: "The practice run cannot continue until this is answered.",
  options: [
    { value: "Current Balance", label: "Use Current Balance" },
    { value: "Principal Balance", label: "Use Principal Balance" },
  ],
  subject: {
    // Carried, as `OccAgentService._decision_card` now carries it. Without the
    // type nothing downstream can tell an ambiguity from a proposal: promotion
    // reads it to decide which side an amended answer belongs on, the table
    // reads it to tell "Change this" from "Answer this", and the scripted walk
    // reads it to know this is the one halt it must not settle.
    decision_type: "mapping_ambiguity",
    source_column: "Current Balance",
    source_columns: ["Current Balance", "Principal Balance"],
    target_field: "current_principal_balance",
  },
};

const TENANT = "Alpine Capital";
const ACTOR = "Operator";

/** Test seam: whether the mocked environment offers a REAL onboarding, and
 *  what the last create actually asked for. A mock deployment is a rehearsal
 *  by default, exactly as a real one is. */
export const mockAgentLive: { available: boolean; lastCreateLive?: boolean } = {
  available: false,
};

export class MockAgent {
  /** The agent's OWN onboarding store, isolated from the one the `/onboarding`
   *  screens use — the mock's equivalent of the synthetic container. */
  private readonly onboarding = new MockOnboarding();
  private runs = new Map<string, StoredRun>();
  private nextId = 1;

  meta(): AgentMeta {
    return {
      enabled: true,
      flag: "OCC_AGENT_SYNTHETIC_ENABLED",
      runtime_mode: "synthetic",
      live_available: mockAgentLive.available,
      policy: POLICY,
      lifecycle: lifecycle(),
      onboarding_reference: this.onboarding.reference(),
      scenarios: SCENARIOS,
      tenant: TENANT,
    };
  }

  list(state?: string): CaseSummary[] {
    const rows = [...this.runs.values()].map((r) => this.summary(r));
    return state ? rows.filter((r) => r.state === state) : rows;
  }

  /**
   * Open a case. `amendClient` opens an AMENDMENT to that client's active
   * configuration instead of a new onboarding: the answers start from the
   * version in force, so adding a reporting product later is a difference to
   * review rather than a fresh set of answers that happen to mostly match.
   */
  create(instruction: string, fixtureId = "", amendClient = "",
         live = false): AgentStatus {
    const scenario = fixtureId || this.scenarioFor(instruction);
    if (amendClient) {
      // An amendment already carries the client's answers. Re-reading a
      // (usually empty) instruction over them would overwrite what is in
      // force with whatever a sentence happened to mention.
      const amended = this.onboarding.startAmendment(amendClient, ACTOR);
      const amendedRun: StoredRun = {
        doc: this.blankRun(amended.case_id, instruction, scenario, ""),
        reached: new Set([S.AWAITING_ONBOARDING]),
        scenario,
        audit: [],
      };
      this.runs.set(amended.case_id, amendedRun);
      this.record(amendedRun, "practice_case_opened",
                  `an operator opened an amendment to ${amendClient}`);
      return this.status(amended.case_id);
    }
    const opened = this.onboarding.startNewClient(ACTOR);
    const facts = interpret(instruction);
    this.onboarding.saveStep(opened.case_id, "client", {
      client_name: facts.clientName,
      jurisdiction: facts.jurisdiction,
    });
    this.onboarding.saveStep(opened.case_id, "entities", {
      entities: [{ legal_name: facts.clientName, roles: ["originator"] }],
    });
    this.onboarding.saveStep(opened.case_id, "portfolios", {
      portfolios: [
        {
          portfolio_id: facts.portfolioId,
          display_name: `${facts.clientName} portfolio`,
          asset_class: "equity_release",
          portfolio_type: "direct",
        },
      ],
    });
    this.onboarding.saveStep(opened.case_id, "reporting", { products: facts.products });
    if (facts.pipeline) {
      this.onboarding.addPipelineBook(opened.case_id, facts.portfolioId);
    }

    const doc = this.blankRun(opened.case_id, instruction, scenario,
                              facts.reportingPeriod, live);
    const stored: StoredRun = {
      doc,
      reached: new Set([S.AWAITING_ONBOARDING]),
      scenario,
      audit: [],
    };
    this.runs.set(opened.case_id, stored);
    this.record(stored, "practice_case_opened", "an operator opened a practice case");
    this.record(stored, "onboarding_answered", "structured answers read from the instruction");
    // Read the case back: what Trakt worked out for itself only exists after the
    // answers have been through `saveStep`, and that is what gets shown back.
    stored.doc.messages.push({
      role: "agent",
      text: this.describe(this.onboardingCase(opened.case_id)),
      at: nowIso(),
      refs: [],
    });
    return this.status(opened.case_id);
  }

  get(caseRef: string): StoredRun {
    const stored = this.runs.get(caseRef);
    if (!stored) {
      throw new OpsError("That practice case could not be found.", "OCC_AGENT_RUN_NOT_FOUND");
    }
    return stored;
  }

  private onboardingCase(caseRef: string): OnboardingCase {
    return this.onboarding.case(caseRef);
  }

  status(caseRef: string): AgentStatus {
    const stored = this.get(caseRef);
    const onboarding = this.onboardingCase(caseRef);
    const facts = this.facts(onboarding, stored);
    stored.doc.facts = facts;
    return {
      case_ref: caseRef,
      run: stored.doc,
      summary: this.summary(stored),
      onboarding,
      facts,
      streams: this.streamSummaries(onboarding),
      state: lifecycle(stored.doc.state).find((s) => s.state === stored.doc.state)!,
      lifecycle: lifecycle(stored.doc.state, stored.reached),
      stage_outcomes: stored.doc.stage_outcomes,
      readiness: this.readiness(stored, onboarding),
      policy: POLICY,
      open_decisions: stored.doc.open_decisions,
      mapping: mappingOverview(stored.doc),
      observations: stored.doc.observations,
      blockers: stored.doc.blockers,
      occ_links: [
        // Faithful to the server: the onboarding link is offered only once the
        // case has ACTIVATED and so exists in the governed store, and it points
        // at the real route. It used to be offered always, at
        // `/onboarding/{ref}` — which is not a route, for a case the governed
        // wizard would 404 on anyway. Leaving that here would let a UI test
        // pass against a link production no longer emits.
        ...(onboarding.status === "activated"
          ? [
              {
                label: "Client onboarding",
                to: `/onboarding/cases/${caseRef}`,
                why: "The activated onboarding case, in the screens an operator normally works it in.",
              },
            ]
          : []),
        {
          label: "Platform configuration",
          to: "/admin/config",
          why: "The asset, regime and system packages this case resolved against.",
        },
        {
          label: "Rules",
          to: "/rules",
          why: "The approved mapping and alias rules the platform applies.",
        },
      ],
      pack: this.packView(stored),
      review_package_ref: stored.doc.review_package_ref,
      activation: {
        mode: stored.doc.mode,
        adapter: "synthetic",
        live_enabled: false,
        intent: stored.doc.activation_intent as ActivationIntent | Record<string, never>,
        result: stored.doc.activation_result as ActivationResult | Record<string, never>,
        approval:
          (stored.doc.approvals.find((a) => a.subject === "configuration") as Record<
            string,
            unknown
          >) ?? {},
      },
      running: false,
      anything_simulated: false,
      anything_blocked: stored.doc.state === S.BLOCKED,
      configuration_written: onboarding.status === "activated",
    };
  }

  /**
   * The streams projection, from the onboarding case's own source rows —
   * the same shape the server derives, with the same regime rule: pipeline
   * never feeds a regime, funded may where the products permit.
   */
  private streamSummaries(onboarding: OnboardingCase): AgentStatus["streams"] {
    const sources = (onboarding.answers.sources ?? []) as Record<string, unknown>[];
    const products = ((onboarding.answers.reporting ?? {}) as { products?: string[] })
      .products ?? [];
    const regimeChosen = products.includes("esma_annex2");
    return sources.map((source) => {
      const dataset = String(source.dataset ?? "");
      const capable = dataset === "funded";
      return {
        source_key: String(source.source_key ?? ""),
        portfolio_id: String(source.portfolio_id ?? ""),
        dataset,
        label: dataset ? dataset[0].toUpperCase() + dataset.slice(1) : "Stream",
        purpose:
          dataset === "pipeline"
            ? "Pipeline MI only"
            : "Funded MI, risk monitoring and reporting as configured",
        cadence: String(source.cadence ?? ""),
        cadence_confirmed: false,
        required_file: dataset === "pipeline" ? "Pipeline Report" : "Primary loan tape",
        expected_files: (source.expected_files as string[] | undefined) ?? [],
        regime_capable: capable,
        regime_status: !capable
          ? "not_applicable"
          : regimeChosen
            ? "configured"
            : "potential",
        regime_note: !capable
          ? "Pipeline data feeds pipeline MI only. It never feeds a regulatory regime."
          : regimeChosen
            ? "Regime reporting is configured."
            : "Potentially applicable, subject to product selection.",
      };
    });
  }

  // -- the client pack ----------------------------------------------------- //

  /**
   * The pack, projected from the onboarding reference's own catalogue.
   *
   * The mock builds it the same way the server does — every question is a
   * catalogue field — so a test that asserts "no question was invented" is
   * meaningful against the mock too.
   */
  pack(caseRef: string): AgentPack {
    const stored = this.get(caseRef);
    const sections = this.packSections(caseRef);
    return {
      pack: { sections, mapping_statement: MAPPING_STATEMENT },
      document: packDocument(this.onboardingCase(caseRef).client_name, sections),
      status: stored.doc.pack_status,
      history: stored.doc.pack_history,
      receipt: stored.doc.pack_receipt as PackReceipt | Record<string, never>,
    };
  }

  /**
   * Every catalogue field, in one of the five categories.
   *
   * The same rules the server applies, against the mock's own catalogue: an
   * answered field is already known; a section that needs something else to
   * exist first is deferred; an operator decision is internal; and what is left
   * is decided by the field's own `source`.
   */
  private classify(caseRef: string): FieldClassification[] {
    const onboarding = this.onboardingCase(caseRef);
    const catalogue = this.onboarding.reference().catalogue;
    const answers = onboarding.answers as Record<string, unknown>;
    const products = new Set(
      ((answers.reporting as Record<string, unknown>)?.products as string[]) ?? [],
    );
    const out: FieldClassification[] = [];
    for (const section of catalogue.sections ?? []) {
      const items = section.repeatable
        ? (((answers[section.key] as Record<string, unknown>[]) ?? [{}]))
        : [(answers[section.key] as Record<string, unknown>) ?? {}];
      items.forEach((holder, index) => {
        for (const f of section.fields ?? []) {
          const path = `${section.key}.${f.key}`;
          const value = holder?.[f.key] ?? null;
          const asked = ASKED_WHEN[path];
          let category: FieldCategory;
          let reason: string;
          if (INTERNAL_FIELDS.has(path)) {
            category = "internal";
            reason = "an operator decision, not a client question";
          } else if (present(value)) {
            category = "known";
            reason = `already answered (${ORIGIN[f.source] ?? f.source})`;
          } else if (asked && !asked(products)) {
            category = "not_applicable";
            reason = "does not apply here";
          } else if (DEFERRED_SECTIONS.has(section.key)) {
            category = "first_delivery";
            reason = "asked once the client has listed the files they will send";
          } else {
            category = BY_SOURCE[f.source] ?? "internal";
            reason = REASONS[category];
          }
          out.push({
            section: section.key,
            field: f.key,
            label: f.label,
            category,
            number: CATEGORY_NUMBERS[category],
            category_label: CATEGORY_LABELS[category],
            reason,
            source: f.source,
            required: Boolean(f.required),
            confirm:
              category === "known" &&
              Boolean(f.required) &&
              !NEVER_CONFIRMED_SOURCES.has(f.source),
            value,
            provenance: "",
            index: section.repeatable ? index : null,
            item: "",
          });
        }
      });
    }
    return out;
  }

  clientForm(caseRef: string): ClientFormView {
    const onboarding = this.onboardingCase(caseRef);
    const catalogue = this.onboarding.reference().catalogue;
    // Answered client questions sit BESIDE the steps, never among them: a
    // client is not re-asked what they have answered, and the pack is built
    // from the steps. They are carried so an operator can see what an answer
    // saved as and correct a typo in it.
    const isAnsweredClient = (r: FieldClassification): boolean => {
      if (r.category !== "known") return false;
      const section = (catalogue.sections ?? []).find((s2) => s2.key === r.section);
      const f = (section?.fields ?? []).find((x) => x.key === r.field);
      return (
        (f as { source?: string } | undefined)?.source === "client_supplied" &&
        r.value !== null &&
        r.value !== undefined &&
        r.value !== ""
      );
    };
    const rows = this.classify(caseRef);
    const asked = rows.filter((r) => r.category === "client");
    const steps: ClientFormStep[] = [];
    for (const spec of FORM_STEPS) {
      const groups: ClientFormGroup[] = [];
      for (const key of spec.sections) {
        const section = (catalogue.sections ?? []).find((s) => s.key === key);
        if (!section) continue;
        const rows = asked.filter((r) => r.section === key);
        if (rows.length === 0) continue;
        groups.push({
          key,
          label: section.label,
          help: section.help ?? "",
          repeatable: Boolean(section.repeatable),
          index: rows[0].index,
          item: rows[0].item,
          required: rows.filter((r) => r.required).length,
          fields: rows.map((r) => {
            const f = (section.fields ?? []).find((x) => x.key === r.field);
            return {
              key: r.index === null ? `${key}.${r.field}` : `${key}[${r.index}].${r.field}`,
              section: key,
              field: r.field,
              label: r.label,
              help: f?.help ?? "",
              type: f?.type ?? "text",
              options: (f?.options ?? []) as { value: string; label: string }[],
              required: r.required,
              sensitive: Boolean(f?.sensitive),
              evidence_required: false,
              max_length: null,
              validation: f?.validation ?? "",
              value: r.value,
              answered: false,
              index: r.index,
              item: r.item,
            };
          }),
        });
      }
      if (groups.length > 0) {
        steps.push({
          key: spec.key,
          label: spec.label,
          help: spec.help,
          unlocked_by: "",
          groups,
          questions: groups.reduce((n, g) => n + g.fields.length, 0),
          required: groups.reduce((n, g) => n + g.required, 0),
        });
      }
    }
    return {
      case_ref: caseRef,
      client_name: onboarding.client_name,
      steps,
      answered: rows.filter(isAnsweredClient).map((r) => {
        const section = (catalogue.sections ?? []).find((s2) => s2.key === r.section);
        const f = (section?.fields ?? []).find((x) => x.key === r.field);
        return {
          key: r.index === null ? `${r.section}.${r.field}` : `${r.section}[${r.index}].${r.field}`,
          section: r.section,
          field: r.field,
          label: r.label,
          help: f?.help ?? "",
          type: f?.type ?? "text",
          options: (f?.options ?? []) as { value: string; label: string }[],
          required: r.required,
          sensitive: Boolean(f?.sensitive),
          evidence_required: false,
          max_length: null,
          validation: f?.validation ?? "",
          value: r.value,
          answered: true,
          index: r.index,
          item: r.item,
        };
      }),
      locked: (catalogue.sections ?? [])
        .filter((s: { deferred_until?: string }) => Boolean(s.deferred_until))
        .map((s: { key: string; label: string; deferred_until?: string }) => ({
          step: "data",
          label: s.label,
          unlocked_by: `Asked once ${s.deferred_until}.`,
        })),
      questions: steps.reduce((n, s) => n + s.questions, 0),
      required: steps.reduce((n, s) => n + s.required, 0),
      content_hash: "sha-mock-form",
    };
  }

  private packSections(caseRef: string): PackSection[] {
    const form = this.clientForm(caseRef);
    const out: PackSection[] = [];
    for (const step of form.steps) {
      for (const group of step.groups) {
        out.push({
          key: group.key,
          label: group.label,
          help: group.help,
          repeatable: group.repeatable,
          step: step.key,
          step_label: step.label,
          index: group.index,
          item: group.item,
          questions: group.fields.map((f) => ({
            section: f.section,
            field: f.field,
            label: f.label,
            help: f.help,
            status: present(f.value) ? "answered" : "outstanding",
            value: f.value ?? null,
            provenance: "",
            index: f.index,
            item: f.item,
            required: f.required,
            evidence_required: f.evidence_required,
            sensitive: f.sensitive,
            writes_to: "",
            step: step.key,
            step_label: step.label,
          })),
          outstanding: group.fields.filter((f) => !present(f.value)).length,
        });
      }
    }
    return out;
  }

  /**
   * Persist a structured response, verbatim.
   *
   * The same refusals the server applies: a key the catalogue does not declare,
   * and a key this client was not asked, are both refused with nothing saved.
   */
  /**
   * The whole concentration-test decision, with the server's own refusals.
   *
   * A blank answer can never be recorded as supplied, and only the four
   * declared statuses are accepted — the two controls this route exists to
   * enforce, so a screen tested against the mock meets them here too.
   */
  recordConcentration(
    caseRef: string,
    input: { status: string; response_text?: string; reason?: string },
  ): AgentStatus {
    const stored = this.get(caseRef);
    const allowed = ["supplied", "not_applicable", "deferred_with_reason", "pending_client_response"];
    if (!allowed.includes(input.status)) {
      throw new OpsError(
        `'${input.status}' is not a concentration-test status.`,
        "OCC_AGENT_INVALID_STATUS",
      );
    }
    if (input.status === "supplied" && !(input.response_text ?? "").trim()) {
      throw new OpsError(
        "A blank answer cannot be recorded as supplied.",
        "OCC_AGENT_BLANK_SUPPLIED",
      );
    }
    const payload: Record<string, unknown> = { concentration_tests_status: input.status };
    if ((input.response_text ?? "").trim()) payload.concentration_tests = input.response_text;
    if ((input.reason ?? "").trim()) payload.concentration_tests_status_reason = input.reason;
    this.onboarding.saveStep(caseRef, "risk_limits", payload);
    this.record(stored, "concentration_outcome_recorded", "an operator recorded the decision");
    return this.status(caseRef);
  }

  submitClientForm(
    caseRef: string,
    answers: Record<string, unknown>,
    options: { request_id?: string; strict?: boolean } = {},
  ): AgentStatus {
    const stored = this.get(caseRef);
    const form = this.clientForm(caseRef);
    // Asked OR already answered: a correction to an answer is submitted under
    // the key it was first saved with, and refusing it would mean a typo could
    // never be fixed through the form it was typed into.
    const served = new Set([
      ...form.steps.flatMap((s) => s.groups.flatMap((g) => g.fields.map((f) => f.key))),
      ...form.answered.map((f) => f.key),
    ]);
    // A key that is not a catalogue field at all, and a key that IS one but
    // was not put to this client, are different refusals — an operator needs
    // to know which, and so would a client portal.
    const catalogue = this.onboarding.reference().catalogue;
    const declared = new Set(
      (catalogue.sections ?? []).flatMap((section) =>
        (section.fields ?? []).map((f) => `${section.key}.${f.key}`),
      ),
    );
    const unknown: string[] = [];
    const notAsked: string[] = [];
    for (const key of Object.keys(answers)) {
      const match = /^([a-z_]+)(?:\[\d+\])?\.([a-z0-9_]+)$/.exec(key);
      if (!match || !declared.has(`${match[1]}.${match[2]}`)) {
        unknown.push(key);
      } else if (!served.has(key)) {
        notAsked.push(key);
      }
    }
    if (unknown.length > 0) {
      throw new OpsError(
        `These answers do not correspond to anything Trakt asks: ${unknown.join(", ")}. Nothing was saved.`,
        "OCC_AGENT_UNKNOWN_ANSWER_KEY",
      );
    }
    if (notAsked.length > 0 && options.strict !== false) {
      throw new OpsError(
        `These are not questions Trakt puts to a client: ${notAsked.join(", ")}. Nothing was saved.`,
        "OCC_AGENT_NOT_A_CLIENT_QUESTION",
      );
    }

    const bySection: Record<string, Record<string, unknown>> = {};
    const repeatable: Record<string, Record<string, unknown>[]> = {};
    const onboarding = this.onboardingCase(caseRef);
    for (const [key, value] of Object.entries(answers)) {
      const match = /^([a-z_]+)(?:\[(\d+)\])?\.([a-z0-9_]+)$/.exec(key);
      if (!match) continue;
      const [, section, index, fieldKey] = match;
      if (index === undefined) {
        (bySection[section] ??= {})[fieldKey] = value;
        continue;
      }
      const rows =
        repeatable[section] ??
        (((onboarding.answers[section] as Record<string, unknown>[]) ?? []).map((r) => ({
          ...r,
        })) as Record<string, unknown>[]);
      while (rows.length <= Number(index)) rows.push({});
      rows[Number(index)][fieldKey] = value;
      repeatable[section] = rows;
    }
    for (const [section, payload] of Object.entries(bySection)) {
      this.onboarding.saveStep(caseRef, section, payload);
    }
    for (const [section, rows] of Object.entries(repeatable)) {
      this.onboarding.saveStep(caseRef, section, { [section]: rows });
    }
    this.record(stored, "client_response_submitted",
                "structured answers written straight through Client Onboarding");
    return this.status(caseRef);
  }

  classification(caseRef: string): AgentClassification {
    const fields = this.classify(caseRef);
    const counts = Object.fromEntries(
      (Object.keys(CATEGORY_NUMBERS) as FieldCategory[]).map((k) => [
        k,
        fields.filter((f) => f.category === k).length,
      ]),
    ) as Record<FieldCategory, number>;
    return {
      fields,
      summary: {
        total: fields.length,
        counts,
        labels: CATEGORY_LABELS,
        numbers: CATEGORY_NUMBERS,
        client_facing: counts.client,
        confirmations: fields.filter((f) => f.confirm).length,
      },
    };
  }

  private packView(stored: StoredRun): AgentStatus["pack"] {
    const receipt = stored.doc.pack_receipt as PackReceipt | Record<string, never>;
    const pack = stored.doc.pack as Record<string, unknown>;
    const classified = this.classification(stored.doc.case_ref);
    return {
      status: stored.doc.pack_status,
      confirmations: (pack.confirmations as PackQuestion[]) ?? [],
      not_asked: (pack.not_asked as AgentStatus["pack"]["not_asked"]) ?? [],
      summary: classified.summary,
      history: stored.doc.pack_history,
      outstanding: Number(pack.outstanding ?? 0),
      questions: Number(pack.questions ?? 0),
      sections: (pack.sections as PackSection[]) ?? [],
      email: (pack.email as AgentStatus["pack"]["email"]) ?? {
        to: [],
        cc: [],
        subject: "",
        body: "",
      },
      artefacts: (pack.artefacts as { name: string; ref: string }[]) ?? [],
      mapping_statement: MAPPING_STATEMENT,
      receipt,
      sent: Boolean((receipt as PackReceipt).sent),
    };
  }

  // -- the review package --------------------------------------------------- //

  review(caseRef: string): AgentReviewPackage {
    const onboarding = this.onboardingCase(caseRef);
    const stored = this.get(caseRef);
    const pkg = {
      case_ref: caseRef,
      client_name: onboarding.client_name,
      sections: this.packSections(caseRef).map((s) => ({
        key: s.key,
        label: s.label,
        rows: s.questions
          .filter((q) => present(q.value))
          .map((q) => ({
            label: q.label,
            value: q.value,
            item: q.item,
            provenance: "client_supplied",
            provenance_label: "the client told Trakt",
          })),
      })),
      outstanding: [],
      access_requirements: [],
      operator_actions: [],
      mapping_note: MAPPING_NOTE,
      access_note: ACCESS_NOTE,
      activation: stored.doc.activation_intent,
      readiness: this.readiness(stored, onboarding),
      content_hash: "sha-mock-review",
    };
    return { package: pkg, document: reviewDocument(pkg) };
  }

  activation(caseRef: string): AgentActivation {
    const stored = this.get(caseRef);
    const intent = this.intent(stored);
    return {
      intent,
      preconditions: {
        mode: stored.doc.mode,
        flag_enabled: false,
        case_ref: caseRef,
        onboarding_status: this.onboardingCase(caseRef).status,
        configuration_approved: stored.doc.approvals.some(
          (a) => a.subject === "configuration",
        ),
        readiness_passed: stored.doc.readiness_status === S.READY_FOR_EXECUTION,
        tenant: TENANT,
        client_id: intent.client_id,
        portfolio_id: intent.portfolio_id,
        configuration_valid: true,
        configuration_problems: [],
        artefacts_present: stored.doc.received_artefacts.length,
        required_artefacts_satisfied: true,
        approval_audited: true,
        already_activated: false,
        confirmed: false,
      },
      refusals: [
        "This case is in rehearsal mode.",
        "Live execution is not switched on in this environment.",
      ],
      mode: "synthetic",
      live_enabled: false,
    };
  }

  private intent(stored: StoredRun): ActivationIntent {
    const facts = this.facts(this.onboardingCase(stored.doc.case_ref), stored);
    const files = stored.doc.received_artefacts.map((a) => ({
      name: a.source_file,
      target: a.intended_live_uri,
      sha256: a.sha256,
    }));
    return {
      client_id: facts.client_id,
      client_name: facts.client_name,
      portfolio_id: facts.portfolio_id,
      dataset: facts.dataset,
      reporting_period: stored.doc.reporting_period,
      files,
      target_locations: [...new Set(files.map((f) => f.target).filter(Boolean))],
      actions: [
        `Write 1 configuration artefact(s) for ${facts.client_id}, as a new governed version.`,
        "Register the expected source deliveries in the production source registry.",
        `Place ${files.length} file(s) in the production raw location.`,
        "Start the existing Onboarding Agent, which will profile, map, transform, validate and assemble the delivery.",
      ],
      configuration_artefacts: [`clients/${facts.client_id}.yaml`],
      statement: `Confirming activates ${facts.client_name || facts.client_id}'s configuration and starts ingestion of the files listed. This is production. It cannot be undone from this tab.`,
    };
  }

  step(caseRef: string, step: string, body: Record<string, unknown> = {}): AgentStatus {
    const stored = this.get(caseRef);
    const action = STEP_ACTIONS[step];
    if (!action) throw new OpsError("That is not something Trakt can do.", "OCC_AGENT_UNKNOWN_STEP");
    this.requireAction(stored, action);
    this.applyStep(stored, step, body);
    return this.status(caseRef);
  }

  saveStep(caseRef: string, step: string, payload: Record<string, unknown>): AgentStatus {
    this.get(caseRef);
    this.onboarding.saveStep(caseRef, step, payload);
    return this.status(caseRef);
  }

  checklist(caseRef: string): AgentChecklist {
    this.get(caseRef);
    return {
      checklist: this.onboarding.checklist(caseRef),
      requests: this.onboardingCase(caseRef).information_requests,
    };
  }

  preview(caseRef: string): AgentPreview {
    this.get(caseRef);
    return {
      preview: this.onboarding.preview(caseRef),
      written: false,
      execution_status: "not_activated",
    };
  }

  recordClientResponse(
    caseRef: string,
    input: { request_id: string; answers: Record<string, unknown>; note?: string },
  ): AgentStatus {
    const stored = this.get(caseRef);
    const request = this.onboardingCase(caseRef).information_requests.find(
      (r) => r.request_id === input.request_id,
    );
    if (!request) {
      throw new OpsError(
        "That information request could not be found.",
        "OCC_AGENT_REQUEST_NOT_FOUND",
      );
    }
    if (request.status === "open") this.onboarding.markSent(caseRef, input.request_id);
    this.onboarding.recordResponse(caseRef, input.request_id, {
      note: input.note ?? "",
      answers: input.answers,
    });
    this.record(stored, "client_response_recorded", "the operator recorded the client's response");
    return this.status(caseRef);
  }

  instruct(caseRef: string, text: string, confirm: boolean): AgentTurn {
    const stored = this.get(caseRef);
    const lower = text.trim().toLowerCase();
    stored.doc.messages.push({ role: "operator", text, at: nowIso(), refs: [] });

    if (lower.endsWith("?") || /^(what|why|which|how|when|who)\b/.test(lower)) {
      const reply = this.answer(stored, lower);
      stored.doc.messages.push({ role: "agent", text: reply, at: nowIso(), refs: [] });
      return { ...this.status(caseRef), reply, applied: false, proposal: null };
    }

    const step = this.stepForInstruction(lower, stored, this.onboardingCase(caseRef));
    if (!step) {
      throw new OpsError(
        "Trakt could not tell what to do with that. Try naming the change directly.",
        "OCC_AGENT_NOT_UNDERSTOOD",
      );
    }
    this.requireAction(stored, STEP_ACTIONS[step]);

    if (MATERIAL_STEPS.has(step) && !confirm) {
      const proposal: AgentProposal = {
        proposal_id: `prop-${this.nextId++}`,
        action: STEP_ACTIONS[step],
        payload: { step },
        summary: PROPOSAL_SUMMARIES[step] ?? "Apply this change.",
        basis: "explicit instruction",
        material: true,
        confidence: 1,
      };
      stored.doc.messages.push({
        role: "agent",
        text: `Proposed: ${proposal.summary} Confirm to apply.`,
        at: nowIso(),
        refs: [proposal.proposal_id],
      });
      return { ...this.status(caseRef), reply: proposal.summary, applied: false, proposal };
    }

    this.applyStep(stored, step, {});
    const reply = this.statusSentence(stored, this.onboardingCase(caseRef));
    stored.doc.messages.push({ role: "agent", text: reply, at: nowIso(), refs: [] });
    // An APPLIED turn still carries a proposal object. This mock used to
    // return null here, which is why a screen that leaves the proposal banner
    // up over an already-applied change passed every test: the fake was kinder
    // than the server. `OccAgentService.instruct` returns
    // `proposal={"disclosure": plan.disclosure()}` on the applied path, so the
    // value is TRUTHY and a caller keying off its presence is misled.
    //
    // The server sends only the disclosure; the shape here is fuller because
    // the type demands it. Neither matters to a correct caller — an applied
    // turn has no pending proposal, and this field must not be read.
    return {
      ...this.status(caseRef),
      reply,
      applied: true,
      proposal: {
        proposal_id: `applied-${this.nextId++}`,
        action: STEP_ACTIONS[step],
        payload: { step },
        summary: "",
        basis: "applied",
        material: false,
        confidence: 1,
        disclosure: { understood: [reply], proposed: "", questions: [], unrecognised: [] },
      },
    };
  }

  /**
   * Approve every mapping still proposed on this delivery.
   *
   * Faithful to `OccAgentService.approve_proposed_mappings`: one act, and one
   * resolved record per column — not one record for "the mappings" — because
   * that is what promotion turns into governed rules. Anything already
   * answered keeps its answer.
   */
  approveMappings(caseRef: string, reason = ""): AgentStatus {
    const stored = this.get(caseRef);
    this.requireAction(stored, "resolve_decision");
    const pending = stored.doc.open_decisions.filter(
      (d) =>
        (d.subject as { decision_type?: string })?.decision_type === "mapping_proposal" &&
        d.status === "open",
    );
    if (pending.length === 0) {
      throw new OpsError(
        "There are no proposed mappings waiting for approval.",
        "OCC_AGENT_NO_PROPOSED_MAPPINGS",
      );
    }
    for (const decision of pending) {
      decision.status = "approved";
      decision.resolved_value = String(
        (decision.subject as { target_field?: string })?.target_field ?? "",
      );
      decision.resolved_by = ACTOR;
      this.applyAnswer(
        stored,
        String((decision.subject as { source_column?: string })?.source_column ?? ""),
        decision.resolved_value,
        true,
      );
    }
    this.record(
      stored,
      "mapping_set_approved",
      reason || "the operator approved the proposed mappings for this delivery",
    );
    if (!stored.doc.open_decisions.some((d) => d.blocking && d.status === "open")) {
      this.move(stored, S.SYNTHETIC_ONBOARDING_RUNNING);
      stored.doc.stage_outcomes = COMPLETED_STAGES;
      this.move(stored, S.SYNTHETIC_ONBOARDING_PASSED);
      stored.doc.blockers = [];
    }
    return this.status(caseRef);
  }

  /**
   * Record an answered mapping on the report itself.
   *
   * The server reruns the whole onboard stage after a decision, and the rerun
   * resolves that column from `approved_mappings` at tier `operator_approved`.
   * Without this the mock re-derives the row from its ORIGINAL tier, so a weak
   * match an operator has just answered goes straight back to "Needs you" and
   * the approval it gates can never be reached.
   */
  private applyAnswer(stored: StoredRun, column: string, field: string,
                      used: boolean): void {
    for (const raw of stored.doc.mapping_report) {
      const row = raw as Record<string, unknown>;
      if (String(row.source_column ?? "") !== column) continue;
      row.tier = "operator_approved";
      row.note = used ? "confirmed by an operator" : "an operator set this aside";
      row.canonical_field = used ? field : "";
      row.confidence = 1.0;
    }
  }

  /**
   * The canonical field an answer names, or "" if it names an ACTION.
   *
   * The decision card sends the chosen option's value as the amended field, and
   * the real options are actions — so "accept_mapping" would otherwise be
   * written into the tape as a field name. `OccAgentService._approved_mappings`
   * guards exactly this list; without the same guard here the mock let an
   * action string reach the mapping report, and a column came out reading
   * "approve".
   */
  private static fieldFromAnswer(value: string, fallback: string): string {
    const ACTIONS = new Set([
      "accept_mapping", "confirm_selected", "approve", "amend", "",
    ]);
    return ACTIONS.has(value) ? fallback : value;
  }

  answerDecision(
    caseRef: string,
    input: { decision_id: string; action: string; value?: string; reason?: string },
  ): AgentStatus {
    const stored = this.get(caseRef);
    this.requireAction(stored, "resolve_decision");
    const decision = stored.doc.open_decisions.find((d) => d.decision_id === input.decision_id);
    if (!decision) {
      throw new OpsError(
        "That decision could not be found on this case.",
        "OCC_AGENT_DECISION_NOT_FOUND",
      );
    }
    decision.status = input.action === "reject" ? "rejected" : "approved";
    decision.resolved_value = input.value || decision.recommendation;
    decision.resolved_by = ACTOR;
    const subject = decision.subject as { source_column?: string; target_field?: string };
    this.applyAnswer(
      stored,
      String(subject?.source_column ?? ""),
      MockAgent.fieldFromAnswer(String(decision.resolved_value ?? ""),
                                String(subject?.target_field ?? "")),
      input.action !== "reject" && input.value !== "mark_unavailable",
    );
    this.record(
      stored,
      "human_decision_recorded",
      input.reason || `operator chose '${input.action}'`,
    );
    if (!stored.doc.open_decisions.some((d) => d.blocking && d.status === "open")) {
      // The affected controls rerun, exactly as the backend reruns them.
      this.move(stored, S.SYNTHETIC_ONBOARDING_RUNNING);
      stored.doc.stage_outcomes = COMPLETED_STAGES;
      this.move(stored, S.SYNTHETIC_ONBOARDING_PASSED);
      stored.doc.blockers = [];
    }
    return this.status(caseRef);
  }

  /**
   * The live URI an artefact WOULD occupy, or "" while it cannot be derived.
   *
   * Empty until the client, the portfolio AND the period are all known, which
   * is what `ArtefactService.intended_uri` does. The mock used to interpolate
   * an empty period into the path and hand back a URI with a hole in it, so a
   * screen that showed a destination in the mock showed "—" against the real
   * server.
   */
  private intendedUri(
    facts: { client_id: string; portfolio_id: string; cadence: string },
    doc: { reporting_period: string; dataset: string },
    name: string,
  ): string {
    if (!facts.client_id || !facts.portfolio_id || !doc.reporting_period) return "";
    const book = doc.dataset || "funded";
    return `blob://raw-v2/${facts.client_id}/direct/${book}/${facts.cadence}/${facts.portfolio_id}/${doc.reporting_period}/${name}`;
  }

  setRunTarget(
    caseRef: string,
    input: { portfolio_id?: string; dataset?: string; reporting_period?: string },
  ): AgentStatus {
    const stored = this.get(caseRef);
    if (input.portfolio_id) stored.doc.portfolio_id = input.portfolio_id;
    if (input.dataset) stored.doc.dataset = input.dataset;
    if (input.reporting_period) stored.doc.reporting_period = input.reporting_period;
    // Files already uploaded are re-derived, as the server does: the period is
    // usually named AFTER the files arrive, and a destination computed once at
    // upload time would stay empty forever.
    const facts = this.facts(this.onboardingCase(caseRef), stored);
    for (const artefact of stored.doc.received_artefacts) {
      artefact.intended_live_uri = this.intendedUri(facts, stored.doc, artefact.source_file);
    }
    this.record(stored, "run_target_set", "an operator named the delivery this run is for");
    return this.status(caseRef);
  }

  /** Take one file back out of the pack. The bytes are not deleted. */
  removeArtefact(caseRef: string, artefactId: string): AgentStatus {
    const stored = this.get(caseRef);
    this.requireAction(stored, "register_synthetic_artefact");
    const before = stored.doc.received_artefacts.length;
    stored.doc.received_artefacts = stored.doc.received_artefacts.filter(
      (a) => a.artefact_id !== artefactId,
    );
    if (stored.doc.received_artefacts.length === before) {
      throw new OpsError(
        "That file is not in this case's pack.",
        "OCC_AGENT_ARTEFACT_NOT_FOUND",
      );
    }
    this.record(stored, "synthetic_artefact_removed", "an operator removed the file from the pack");
    this.onboarding.registerSample(
      caseRef,
      stored.doc.received_artefacts.map((a) => ({ name: a.source_file, headers: [] })),
    );
    this.record(stored, "artefacts_classified", "apps.blob_trigger_app.file_roles");
    return this.status(caseRef);
  }

  uploadArtefacts(caseRef: string, names: string[]): AgentStatus {
    const stored = this.get(caseRef);
    this.requireAction(stored, "register_synthetic_artefact");
    const facts = this.facts(this.onboardingCase(caseRef), stored);
    for (const name of names) {
      stored.doc.received_artefacts.push({
        artefact_id: `sart-${this.nextId++}`,
        source_file: name,
        artefact_type: name.toLowerCase().includes("loan") ? "loan_extract" : "property_extract",
        synthetic_location: `practice_cases/${caseRef}/artefacts/${name}`,
        intended_live_uri: this.intendedUri(facts, stored.doc, name),
        execution_status: "simulated_only",
        sha256: "sha256:mock",
        size: 4096,
        columns: [],
        row_count: 24,
        recognition_confidence: 1,
        recognition_basis: "filename_keyword",
        provided_by: ACTOR,
        provided_at: nowIso(),
        fixture_id: stored.scenario,
      });
    }
    this.record(stored, "synthetic_artefact_registered", "stored in the practice sandbox");
    // Recognition also answers the onboarding's file-format question, exactly as
    // registering a sample pack does on the server.
    this.onboarding.registerSample(
      caseRef,
      stored.doc.received_artefacts.map((a) => ({ name: a.source_file, headers: [] })),
    );
    this.record(stored, "artefacts_classified", "apps.blob_trigger_app.file_roles");
    return this.status(caseRef);
  }

  /**
   * Answer the outstanding CLIENT QUESTIONS, synthetically.
   *
   * Mirrors the server's `/responses/generate`: values are derived from the
   * served form's own declared type and options, then submitted through the
   * ordinary `submitClientForm` path so the same refusals apply.
   *
   * Distinct from `generateResponse`, which makes up the data FILES.
   */
  generateAnswers(caseRef: string): AgentStatus {
    const form = this.clientForm(caseRef);
    const slug =
      (this.onboardingCase(caseRef).client_name || "practice")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "") || "practice";
    const answers: Record<string, unknown> = {};
    for (const step of form.steps) {
      for (const group of step.groups) {
        for (const f of group.fields) {
          const value = syntheticAnswer(f, `${slug}.example`);
          if (value !== undefined) answers[f.key] = value;
        }
      }
    }
    if (Object.keys(answers).length === 0) {
      throw new OpsError(
        "There is nothing outstanding for Trakt to answer on this case.",
        "OCC_AGENT_NOTHING_TO_ANSWER",
      );
    }
    return this.submitClientForm(caseRef, answers);
  }

  generateResponse(caseRef: string): AgentStatus {
    const stored = this.get(caseRef);
    const facts = this.facts(this.onboardingCase(caseRef), stored);
    return this.uploadArtefacts(caseRef, [
      `${facts.portfolio_id || "portfolio"}_loan_extract_202606.csv`,
    ]);
  }

  loadFixtureArtefacts(caseRef: string, fixtureId: string): AgentStatus {
    const scenario = scenarioById(fixtureId);
    const stored = this.get(caseRef);
    stored.scenario = fixtureId;
    stored.doc.fixture_id = fixtureId;
    return this.uploadArtefacts(
      caseRef,
      scenario.files.map((f) => f.filename),
    );
  }

  runScenario(fixtureId: string): AgentStatus {
    const scenario = scenarioById(fixtureId);
    const created = this.create(scenario.instruction, fixtureId);
    const caseRef = created.case_ref;
    const stored = this.get(caseRef);
    stored.doc.reporting_period = scenario.reporting_period;

    // The process starts with the client: draft the pack, have a human read
    // it, approve it and record it as issued.
    this.step(caseRef, "pack/draft");
    this.step(caseRef, "pack/approve");
    this.step(caseRef, "pack/send", {
      to: [clientResponse(SCENARIO_DOMAINS[fixtureId] ?? "practice.example")[
        "contacts.reporting_contact_email"
      ]],
    });

    this.loadFixtureArtefacts(caseRef, fixtureId);

    // The onboarding half, in the order a human would work it.
    const outstanding = this.onboarding.checklist(caseRef);
    if (outstanding.length > 0) {
      this.step(caseRef, "information-requests");
      const request = this.onboardingCase(caseRef).information_requests.at(-1);
      if (request) {
        this.recordClientResponse(caseRef, {
          request_id: request.request_id,
          answers: answersFor(
            this.onboardingCase(caseRef),
            request.items,
            clientResponse(SCENARIO_DOMAINS[fixtureId] ?? "practice.example"),
          ),
        });
      }
    }
    if (!this.onboardingCase(caseRef).ready) return this.status(caseRef);
    this.step(caseRef, "submit");
    this.step(caseRef, "approve");

    // The execution half, then the human decision about the real thing. The
    // drive stops at the confirmation: that act is a person's, and in a
    // rehearsal it is refused anyway.
    for (const step of ["run", "plan", "readiness/approve", "review", "activation/approve"]) {
      let now = this.get(caseRef).doc.state;
      // A FIRST ONBOARDING STOPS ONCE TO HAVE ITS MAPPINGS APPROVED, and a
      // prepared example is a scripted walk of what a person would do — so it
      // does that too, exactly as the Python scenario harness settles the
      // decisions a run raises.
      //
      // Only what is merely PROPOSED. A genuine question — an ambiguity, a
      // weak match — is what scenario B exists to halt on, and a walk that
      // answered those would be a walk with nothing left to demonstrate.
      if (now === S.EXCEPTIONS_REQUIRE_INPUT && this.nothingAmbiguousIsOpen(caseRef)) {
        this.settleStraightforwardMappings(caseRef);
        now = this.get(caseRef).doc.state;
      }
      if (now === S.BLOCKED || now === S.EXCEPTIONS_REQUIRE_INPUT) break;
      this.step(caseRef, step);
    }
    return this.status(caseRef);
  }

  /** True when nothing genuinely AMBIGUOUS is waiting.
   *
   *  A proposal and a weak match both have an obvious answer — accept what
   *  Trakt read — and a scripted walk gives it. Two columns claiming one field
   *  do not, and that is the halt scenario B exists to show.
   */
  private nothingAmbiguousIsOpen(caseRef: string): boolean {
    const open = this.get(caseRef).doc.open_decisions.filter(
      (d) => d.blocking && d.status === "open",
    );
    return (
      open.length > 0 &&
      open.every(
        (d) =>
          (d.subject as { decision_type?: string })?.decision_type !==
          "mapping_ambiguity",
      )
    );
  }

  /** Answer the weak matches, then approve the set — what a person would do. */
  private settleStraightforwardMappings(caseRef: string): void {
    const stored = this.get(caseRef);
    const questions = stored.doc.open_decisions.filter(
      (d) =>
        d.blocking &&
        d.status === "open" &&
        (d.subject as { decision_type?: string })?.decision_type ===
          "mapping_confirmation",
    );
    for (const question of questions) {
      this.answerDecision(caseRef, {
        decision_id: question.decision_id,
        action: "approve",
        value: question.recommendation,
        reason: "accepted with the prepared example",
      });
    }
    if (
      this.get(caseRef).doc.open_decisions.some(
        (d) =>
          d.blocking &&
          d.status === "open" &&
          (d.subject as { decision_type?: string })?.decision_type ===
            "mapping_proposal",
      )
    ) {
      this.approveMappings(caseRef, "approved with the prepared example");
    }
  }

  readinessPackage(caseRef: string): AgentReadinessPackage {
    const stored = this.get(caseRef);
    const onboarding = this.onboardingCase(caseRef);
    const readiness = this.readiness(stored, onboarding);
    return {
      readiness,
      package: readiness.ready
        ? {
            case_summary: {
              case_ref: caseRef,
              onboarding_status: onboarding.status,
              status: S.READY_FOR_EXECUTION,
            },
            configuration_that_would_be_created: {
              written: false,
              execution_status: "not_activated",
            },
            execution_manifest: {
              runtime_mode: "synthetic",
              execution_performed: false,
              configuration_activated: false,
              published: false,
              live_writes: [],
              emails_sent: [],
              content_hash: "sha-mock",
            },
            statement: {
              headline: "Practice case ready for execution.",
              not_done: [
                "No live files were written.",
                "No production pipeline was triggered.",
                "No external email was sent.",
                "No client configuration was activated.",
                "Nothing was published.",
              ],
            },
          }
        : null,
    };
  }

  audit(caseRef: string): AgentAudit {
    return {
      events: this.get(caseRef).audit,
      chain_intact: true,
      onboarding_events: this.onboardingCase(caseRef).events as unknown as Record<
        string,
        unknown
      >[],
    };
  }

  // -- internals ---------------------------------------------------------- //

  private applyStep(stored: StoredRun, step: string, body: Record<string, unknown>): void {
    const caseRef = stored.doc.case_ref;
    switch (step) {
      case "information-requests": {
        const items = this.onboarding.checklist(caseRef);
        if (items.length === 0) {
          throw new OpsError(
            "There is nothing outstanding to ask the client for.",
            "OCC_AGENT_NOTHING_OUTSTANDING",
          );
        }
        this.onboarding.createRequest(caseRef, items);
        this.record(stored, "client_information_requested", "the checklist became a request");
        break;
      }
      case "submit":
        this.onboarding.submit(caseRef);
        this.record(stored, "onboarding_submitted_for_approval", "the onboarding reported ready");
        break;
      case "pack/draft": {
        const sections = this.packSections(caseRef);
        const onboarding = this.onboardingCase(caseRef);
        const contacts = (onboarding.answers.contacts ?? {}) as Record<string, string>;
        const outstanding = sections.reduce((n, sec) => n + sec.outstanding, 0);
        const classified = this.classification(caseRef);
        stored.doc.pack = {
          sections,
          confirmations: classified.fields
            .filter((f) => f.confirm)
            .map((f) => ({
              section: f.section, field: f.field, label: f.label, help: "",
              status: "answered", value: f.value, provenance: f.provenance,
              index: f.index, item: f.item, required: f.required,
              evidence_required: false, sensitive: false, writes_to: "",
              step: "", step_label: "",
            })),
          not_asked: classified.fields
            .filter((f) => !["client", "known"].includes(f.category))
            .map((f) => ({
              key: `${f.section}.${f.field}`, label: f.label,
              category: f.category, number: f.number, reason: f.reason,
            })),
          outstanding,
          questions: sections.reduce((n, sec) => n + sec.questions.length, 0),
          content_hash: "sha-mock-pack",
          email: {
            to: [contacts.reporting_contact_email].filter(Boolean),
            cc: [],
            subject: `Trakt onboarding — ${onboarding.client_name} (${caseRef})`,
            body: packEmailBody(onboarding.client_name, outstanding),
          },
          artefacts: [
            { name: "onboarding_pack.md", ref: `blob://practice/${caseRef}/onboarding_pack.md` },
            { name: "covering_email.txt", ref: `blob://practice/${caseRef}/covering_email.txt` },
          ],
        };
        this.setPackStatus(stored, "DRAFTED", "the agent drafted the pack from the catalogue");
        this.setPackStatus(stored, "HUMAN_REVIEW_REQUIRED", "a human must read it first");
        if (stored.doc.state === S.AWAITING_ONBOARDING || stored.doc.state === S.PACK_SENT) {
          this.move(stored, S.PACK_DRAFTED);
        }
        this.move(stored, S.PACK_REVIEW_REQUIRED);
        this.record(stored, "onboarding_pack_drafted", "every question is a catalogue field");
        break;
      }
      case "pack/approve": {
        if (!stored.doc.pack.sections) {
          throw new OpsError("There is no pack to approve. Draft one first.", "OCC_AGENT_NO_PACK");
        }
        stored.doc.approvals.push({
          subject: "client_pack",
          decision: "approved",
          actor: ACTOR,
          reason: String(body.reason ?? ""),
        });
        this.setPackStatus(stored, "APPROVED_TO_SEND", "approved by a human");
        this.move(stored, S.PACK_APPROVED_TO_SEND);
        this.record(stored, "onboarding_pack_approved", "a human approved the pack for issue");
        break;
      }
      case "pack/send": {
        const email = (stored.doc.pack.email ?? {}) as { to?: string[]; subject?: string };
        const to = (body.to as string[] | null) ?? email.to ?? [];
        if (to.length === 0) {
          throw new OpsError(
            "There is no contact address on this case to issue the pack to. Record one first.",
            "OCC_AGENT_NO_RECIPIENT",
          );
        }
        stored.doc.pack_receipt = {
          adapter: "record_only",
          sent: false,
          at: nowIso(),
          receipt_id: `com_${caseRef}`,
          to,
          subject: String(email.subject ?? ""),
          artefacts: (stored.doc.pack.artefacts as { name: string; ref: string }[]) ?? [],
          content_hash: "sha-mock-pack",
          statement: RECORD_ONLY_STATEMENT,
        };
        this.setPackStatus(stored, "SENT", RECORD_ONLY_STATEMENT);
        this.move(stored, S.PACK_SENT);
        this.record(stored, "onboarding_pack_issued", RECORD_ONLY_STATEMENT);
        break;
      }
      case "review": {
        stored.doc.activation_intent = this.intent(stored) as unknown as Record<string, unknown>;
        stored.doc.review_package_ref = `blob://practice/${caseRef}/review_package.json`;
        this.move(stored, S.READY_FOR_REVIEW);
        this.record(stored, "review_package_assembled", "derived from the case and the run");
        break;
      }
      case "activation/approve": {
        if (!stored.doc.review_package_ref) {
          throw new OpsError(
            "There is no review package to approve. Submit the case for review first.",
            "OCC_AGENT_NO_REVIEW_PACKAGE",
          );
        }
        stored.doc.approvals.push({
          subject: "configuration",
          decision: "approved",
          actor: ACTOR,
          reason: String(body.reason ?? "") || "Configuration approved for activation.",
        });
        this.move(stored, S.APPROVED_FOR_ACTIVATION);
        stored.doc.activation_intent = this.intent(stored) as unknown as Record<string, unknown>;
        this.move(stored, S.ACTIVATION_CONFIRMATION_REQUIRED);
        this.record(
          stored,
          "activation_approved",
          "a human approved the configuration; nothing was started",
        );
        break;
      }
      case "activation/confirm": {
        // The rehearsal's whole point. The refusal is the feature, and it is
        // audited exactly as the server audits it.
        this.record(stored, "activation_refused", "live execution is not available here");
        throw new OpsError(
          "This cannot be activated yet. This case is in rehearsal mode. Live execution is not switched on in this environment.",
          "OCC_AGENT_ACTIVATION_REFUSED",
        );
      }
      case "approve":
        this.onboarding.approve(
          caseRef,
          String(body.reason ?? "") || "Approved in a practice case.",
          ACTOR,
        );
        if (PACK_OR_START.has(stored.doc.state)) this.move(stored, S.READY_TO_RUN);
        this.record(
          stored,
          "onboarding_approved",
          "the operator approved the onboarding; no configuration was created",
        );
        break;
      case "request-changes": {
        const reason = String(body.reason ?? "") || "Changes requested.";
        this.onboarding.requestChanges(caseRef, reason);
        this.record(stored, "onboarding_changes_requested", reason);
        break;
      }
      case "run": {
        const hasLoanTape = stored.doc.received_artefacts.some(
          (a) => a.artefact_type === "loan_extract",
        );
        if (!hasLoanTape) {
          this.block(stored, ["Trakt still needs the Primary loan tape."]);
          break;
        }
        this.move(stored, S.SYNTHETIC_ONBOARDING_RUNNING);
        stored.doc.stage_outcomes = COMPLETED_STAGES;
        // Reading the files is what produces the mapping report, so it is
        // written here rather than seeded at case creation.
        stored.doc.mapping_report = MAPPING_REPORT.map((row) => ({ ...row }));
        this.record(stored, "synthetic_onboarding_started", "the conductor ran over the adapter");
        // A FIRST ONBOARDING PROPOSES; IT DOES NOT DECIDE. Every column the
        // mapper settled on its own is raised for approval, because a governed
        // alias says the NAME is familiar and not that this client means the
        // same thing by it. An amendment's mappings are already governed and
        // already answered, so it keeps matching as before.
        const proposals = MAPPING_REPORT.filter((raw) => {
          const r = raw as { primary?: boolean; canonical_field?: string;
                             tier?: string; confidence?: number };
          return (
            r.primary !== false &&
            Boolean(r.canonical_field) &&
            // A column a person has already answered is never re-proposed:
            // `execution.onboard` resolves it on its own branch before the
            // mapper is consulted. Re-asking would undo the approval this
            // whole design exists to capture.
            r.tier !== "operator_approved" &&
            (TRUSTED_TIERS.has(String(r.tier)) ||
              Number(r.confidence ?? 0) >= LOW_CONFIDENCE)
          );
        }).map((raw) => {
          const r = raw as { source_column: string; canonical_field: string;
                             tier: string; confidence: number };
          return proposalFor(r);
        });
        // A weak match is a QUESTION, not a proposal: it is answered on its
        // own and it gates the approval of the set, so it has to have a card.
        const questions = MAPPING_REPORT.filter((raw) => {
          const r = raw as { primary?: boolean; canonical_field?: string;
                             tier?: string; confidence?: number };
          return (
            r.primary !== false &&
            Boolean(r.canonical_field) &&
            r.tier !== "operator_approved" &&
            !TRUSTED_TIERS.has(String(r.tier)) &&
            Number(r.confidence ?? 0) < LOW_CONFIDENCE
          );
        }).map((raw) => {
          const r = raw as { source_column: string; canonical_field: string;
                             confidence: number };
          return confirmationFor(r);
        });
        if (stored.scenario === "scenario_b_ambiguous_mapping") {
          stored.doc.stage_outcomes = { onboard: "human_input_required" };
          // The clash is asked FIRST and its two columns stop being proposals:
          // "is this right?" has no answer while two columns claim one field.
          const contested = new Set(
            ((AMBIGUOUS_DECISION.subject as { source_columns?: string[] })
              ?.source_columns ?? []).map((c) => String(c)),
          );
          stored.doc.open_decisions = [
            ...questions,
            ...proposals.filter(
              (p) =>
                !contested.has(
                  String((p.subject as { source_column?: string }).source_column),
                ),
            ),
            { ...AMBIGUOUS_DECISION },
          ];
          this.move(stored, S.EXCEPTIONS_REQUIRE_INPUT);
        } else if (stored.scenario === "scenario_e_business_rule_failure") {
          stored.doc.stage_outcomes = {
            onboard: "deterministic_execution_completed",
            transform: "deterministic_execution_completed",
            validate: "hard_blocked",
          };
          this.move(stored, S.EXCEPTIONS_REQUIRE_INPUT);
          this.block(stored, [
            "PORTFOLIO: DAT002 affects 24 record(s) (100.0%) — materiality BLOCKING",
          ]);
        } else if (proposals.length + questions.length > 0) {
          // The ordinary first delivery: nothing is ambiguous and nothing is
          // wrong, and it still waits. "Clean" means no exceptions, not no
          // approval.
          stored.doc.stage_outcomes = { onboard: "human_input_required" };
          stored.doc.open_decisions = [...questions, ...proposals];
          this.move(stored, S.EXCEPTIONS_REQUIRE_INPUT);
        } else {
          this.move(stored, S.SYNTHETIC_ONBOARDING_PASSED);
          this.record(stored, "synthetic_onboarding_passed", "every control passed");
        }
        break;
      }
      case "plan":
        stored.doc.orchestration_plan = MOCK_PLAN;
        stored.doc.assembler_plan = {
          satisfied: true,
          summary: "Assembler prerequisites are satisfied.",
        };
        this.move(stored, S.ORCHESTRATION_PLAN_GENERATED);
        this.move(stored, S.EXECUTION_APPROVAL_REQUIRED);
        this.record(stored, "orchestration_plan_generated", "nothing was executed");
        break;
      case "readiness/approve": {
        stored.doc.approvals.push({ subject: "execution_readiness", decision: "approved" });
        const verdict = this.readiness(stored, this.onboardingCase(caseRef));
        if (!verdict.ready) {
          this.block(
            stored,
            verdict.outstanding.map((c) => c.remedy || c.detail),
          );
          break;
        }
        this.move(stored, S.READY_FOR_EXECUTION);
        stored.doc.readiness_status = S.READY_FOR_EXECUTION;
        this.record(stored, "ready_for_execution", "every readiness criterion passed");
        break;
      }
      case "cancel": {
        // Faithful to the server: the operator's reason is what the audit and
        // the withdrawal record, and cancelling the run WITHDRAWS the case
        // under it (OccAgentService.cancel chains into withdraw). Hardcoding
        // "cancelled by the operator" here and leaving the case open would
        // have let a UI test pass against behaviour production does not have.
        const why = String(body.reason ?? "").trim() || "The practice case was cancelled.";
        this.move(stored, S.CANCELLED);
        this.record(stored, "practice_case_cancelled", why);
        this.onboarding.withdraw(caseRef, why);
        break;
      }
      default:
        throw new OpsError("That is not something Trakt can do.", "OCC_AGENT_UNKNOWN_STEP");
    }
  }

  private facts(onboarding: OnboardingCase, stored: StoredRun): ExecutionFacts {
    const answers = onboarding.answers as Record<string, Record<string, unknown>>;
    const portfolios = ((onboarding.answers.portfolios ?? []) as Record<string, string>[]) ?? [];
    const portfolio =
      portfolios.find((p) => p.portfolio_id === stored.doc.portfolio_id) ?? portfolios[0] ?? {};
    const sources = ((onboarding.answers.sources ?? []) as Record<string, string>[]) ?? [];
    const source = sources.find((s) => s.portfolio_id === portfolio.portfolio_id);
    const products = ((answers.reporting?.products ?? []) as string[]) ?? [];
    return {
      client_id: onboarding.client_id,
      client_name: onboarding.client_name,
      portfolio_id: String(portfolio.portfolio_id ?? ""),
      portfolio_name: String(portfolio.display_name ?? ""),
      asset_class: String(portfolio.asset_class ?? ""),
      dataset: stored.doc.dataset,
      cadence: String(source?.cadence ?? "monthly"),
      jurisdiction: String(answers.client?.jurisdiction ?? ""),
      products,
      product_labels: products.map(
        (id) =>
          (this.onboarding.reference().catalogue.regime_products?.[id]?.label as string) ?? id,
      ),
      outcome: products.includes("esma_annex2") ? "mi_annex2" : "mi",
      regime: products.includes("esma_annex2")
        ? "ESMA_Annex2"
        : products.includes("investor_reporting")
          ? "ESMA_Annex12"
          : "",
      basis: {},
    };
  }

  private readiness(stored: StoredRun, onboarding: OnboardingCase): AgentStatus["readiness"] {
    const doc = stored.doc;
    const criteria: AgentStatus["readiness"]["criteria"] = [
      {
        key: "onboarding_approved",
        label: "Onboarding approved",
        passed: onboarding.status === "approved",
        detail: `The onboarding is ${onboarding.status_label.toLowerCase()}.`,
        remedy: "Work the onboarding through to approval.",
        stage: "onboarding",
      },
      {
        key: "onboarding_clean",
        label: "Nothing outstanding on the onboarding",
        passed: onboarding.blocking.length === 0,
        detail: "The onboarding reports no problems.",
        remedy: "Clear the onboarding's own problems.",
        stage: "onboarding",
      },
      {
        key: "artefacts_present",
        label: "Required source files received",
        passed: doc.received_artefacts.some((a) => a.artefact_type === "loan_extract"),
        detail: "All required files were provided.",
        remedy: "Provide the missing file.",
        stage: "execution",
      },
      {
        key: "exceptions_cleared",
        label: "Blocking exceptions cleared",
        passed: !doc.open_decisions.some((d) => d.blocking && d.status === "open"),
        detail: "No blocking exceptions remain.",
        remedy: "Resolve each blocking decision.",
        stage: "execution",
      },
      {
        key: "pipeline_contracts",
        label: "Pipeline input contracts satisfied",
        passed: ["onboard", "transform", "validate"].every(
          (s) => doc.stage_outcomes[s] === "deterministic_execution_completed",
        ),
        detail: "Every deterministic control completed.",
        remedy: "Run the practice onboarding.",
        stage: "execution",
      },
      {
        key: "orchestration_valid",
        label: "Orchestration sequencing valid",
        passed: Object.keys(doc.orchestration_plan).length > 0,
        detail: "The execution plan is prepared.",
        remedy: "Generate the orchestration plan.",
        stage: "execution",
      },
      {
        key: "execution_approved",
        label: "Readiness approved",
        passed: doc.approvals.some((a) => a.subject === "execution_readiness"),
        detail: "An operator approved readiness for execution.",
        remedy: "Approve readiness once the plan is right.",
        stage: "execution",
      },
      {
        key: "no_configuration_written",
        label: "No configuration was created",
        passed: onboarding.status !== "activated",
        detail: "This practice case created no client configuration.",
        remedy: "This case cannot proceed; report it to your administrator.",
        stage: "boundary",
      },
      {
        key: "runtime_controls_intact",
        label: "Practice controls intact",
        passed: true,
        detail: "The practice boundary held for the whole case.",
        remedy: "",
        stage: "boundary",
      },
    ];
    const outstanding = criteria.filter((c) => !c.passed);
    return {
      ready: outstanding.length === 0,
      status: outstanding.length === 0 ? S.READY_FOR_EXECUTION : "NOT_READY",
      criteria,
      outstanding,
    };
  }

  private describe(onboarding: OnboardingCase): string {
    const client = (onboarding.answers.client ?? {}) as Record<string, string>;
    const lines = [`Onboarding ${onboarding.case_id}.`];
    if (client.client_name) lines.push(`Client name: ${client.client_name}`);
    if (client.client_id) lines.push(`Client identifier: ${client.client_id}`);
    if (client.jurisdiction) lines.push(`Jurisdiction: ${client.jurisdiction}`);
    const outstanding = this.onboarding.checklist(onboarding.case_id);
    if (outstanding.length > 0) {
      lines.push("Still needed from the client:");
      for (const row of outstanding.slice(0, 8)) lines.push(`- ${row.label}`);
    }
    return lines.join("\n");
  }

  private answer(stored: StoredRun, lower: string): string {
    const onboarding = this.onboardingCase(stored.doc.case_ref);
    if (/client/.test(lower) && /(ask|need|send|outstanding|waiting|chase)/.test(lower)) {
      const checklist = this.onboarding.checklist(stored.doc.case_ref);
      if (checklist.length === 0) return "There is nothing outstanding from the client.";
      return `The client still has to tell us:\n${checklist
        .map((row) => `- ${row.label}`)
        .join("\n")}`;
    }
    if (/(left|remain|outstanding|still)/.test(lower)) {
      const verdict = this.readiness(stored, onboarding);
      if (verdict.ready) {
        return "Every readiness criterion is satisfied. Approve readiness to reach READY_FOR_EXECUTION.";
      }
      return `Still outstanding:\n${verdict.outstanding
        .map((c) => `- ${c.label}: ${c.remedy || c.detail}`)
        .join("\n")}`;
    }
    if (lower.startsWith("why") && stored.doc.open_decisions.length > 0) {
      const d = stored.doc.open_decisions[0];
      return `${d.title}\n${d.question}`;
    }
    return this.statusSentence(stored, onboarding);
  }

  private statusSentence(stored: StoredRun, onboarding: OnboardingCase): string {
    const parts = [
      `The onboarding is ${onboarding.status_label.toLowerCase()}; the practice run is at ${
        STATE_LABELS[stored.doc.state]
      }.`,
    ];
    if (stored.doc.blockers.length > 0) {
      parts.push(`In the way: ${stored.doc.blockers.slice(0, 3).join("; ")}`);
    }
    const blocking = stored.doc.open_decisions.filter((d) => d.blocking && d.status === "open");
    if (blocking.length > 0) {
      parts.push(`${blocking.length} decision${blocking.length === 1 ? "" : "s"} need you.`);
    }
    const checklist = this.onboarding.checklist(stored.doc.case_ref);
    if (checklist.length > 0) {
      parts.push(
        `${checklist.length} item${checklist.length === 1 ? "" : "s"} still outstanding from the client.`,
      );
    }
    return parts.join(" ");
  }

  private stepForInstruction(
    lower: string,
    stored: StoredRun,
    onboarding: OnboardingCase,
  ): string | null {
    if (/\bcancel\b/.test(lower)) return "cancel";
    if (/\breadiness\b|\bready for execution\b/.test(lower)) return "readiness/approve";
    if (/\b(approve|accept)\b.*\bonboarding\b|\bapprove the case\b/.test(lower)) return "approve";
    if (/\b(submit|send)\b.*\bapproval\b/.test(lower)) return "submit";
    if (/\b(ask|request)\b.*\b(client|information|checklist)\b/.test(lower))
      return "information-requests";
    if (/\b(generate|prepare)\b.*\b(plan|orchestration)\b/.test(lower)) return "plan";
    if (/\b(run|start|re-?run)\b.*\b(onboarding|practice run|controls)\b/.test(lower)) return "run";
    // A bare "yes" means whatever the case is currently waiting on.
    if (/^(yes|confirm|confirmed|approve|approved|go ahead|proceed)\.?$/.test(lower)) {
      if (["draft", "in_review", "changes_required"].includes(onboarding.status)) return "submit";
      if (onboarding.status === "ready_for_approval") return "approve";
      if (onboarding.status !== "approved") return null;
      return PENDING_CONFIRMATION[stored.doc.state] ?? null;
    }
    return null;
  }

  private requireAction(stored: StoredRun, action: string): void {
    if (ONBOARDING_ACTIONS.includes(action)) return; // the case's own table decides
    if (!(ALLOWED[stored.doc.state] ?? []).includes(action)) {
      throw new OpsError(
        `'${action.replace(/_/g, " ")}' is not something you can do while this practice case is at ${
          STATE_LABELS[stored.doc.state]
        }.`,
        "OCC_AGENT_ACTION_NOT_ALLOWED",
      );
    }
  }

  private move(stored: StoredRun, state: string): void {
    stored.doc.state = state;
    stored.doc.version += 1;
    stored.doc.updated_at = nowIso();
    stored.reached.add(state);
  }

  private block(stored: StoredRun, blockers: string[]): void {
    stored.doc.blockers = blockers.filter(Boolean);
    this.move(stored, S.BLOCKED);
    this.record(stored, "run_blocked", "a deterministic control blocked the run");
  }

  // -- the client's reply ---------------------------------------------------- //

  /**
   * What is "in the mailbox" for this case.
   *
   * The mock's own send is record-only, so there is no real conversation to
   * correlate against. What is reproduced here is the SHAPE the server
   * returns, including the part a happy-path fixture would leave untested: an
   * unmatched message. The screen has to render one, and an operator has to
   * see that Trakt will not guess.
   */
  mail(caseRef: string): AgentMail {
    const stored = this.get(caseRef);
    if (stored.doc.pack_status !== "SENT") {
      return {
        case_ref: caseRef,
        mail: {
          messages: [],
          matched: 0,
          unmatched: 0,
          for_this_case: [],
          note: "Nothing has been issued to this client yet.",
        },
      };
    }
    const messages = this.mailbox(stored);
    const mine = messages.filter((m) => m.correlation.matched);
    return {
      case_ref: caseRef,
      mail: {
        mailbox: "onboarding@traktinfra.io",
        folder: "inbox",
        messages,
        matched: mine.length,
        unmatched: messages.length - mine.length,
        for_this_case: mine,
      },
    };
  }

  private mailbox(stored: StoredRun): AgentMailMessage[] {
    const caseRef = stored.doc.case_ref;
    const receipt = stored.doc.pack_receipt as { to?: string[] };
    const to = (receipt.to ?? [])[0] ?? "client@example.com";
    const taken = new Set(
      (stored.doc.ingested_mail ?? []).map((m) => String(m)),
    );
    return [
      {
        graph_id: `mail-${caseRef}-1`,
        internet_message_id: `<reply-1@${to.split("@")[1] ?? "example.com"}>`,
        conversation_id: `conv-${caseRef}`,
        subject: `RE: Onboarding information request (${caseRef})`,
        sender: to,
        sender_name: "Client contact",
        received_at: nowIso(),
        body_text:
          "Answers below, and the funded book is attached.\n" +
          "LEI: 5493001KJTIIGC8Y1R12\n" +
          "Country of establishment: United Kingdom",
        has_attachments: true,
        attachments: [
          {
            name: "funded_book.csv",
            content_type: "text/csv",
            size: 20480,
            inline: false,
            oversize: false,
            readable: true,
          },
        ],
        correlation: {
          case_ref: caseRef,
          tenant: stored.doc.tenant,
          bases: ["conversation", "in_reply_to", "case_ref", "sender"],
          candidates: [],
          matched: true,
          note: "Matched on conversation, in_reply_to, case_ref, sender.",
        },
        already_ingested: taken.has(`mail-${caseRef}-1`),
      },
      {
        graph_id: `mail-${caseRef}-2`,
        internet_message_id: "<newsletter@example.net>",
        conversation_id: "conv-unrelated",
        subject: "Some files for you",
        sender: to,
        sender_name: "Client contact",
        received_at: nowIso(),
        body_text: "Sending these over as discussed.",
        has_attachments: false,
        attachments: [],
        correlation: {
          case_ref: "",
          tenant: "",
          bases: [],
          candidates: [caseRef],
          matched: false,
          note:
            "The sender is a contact on this case, but nothing else in the message " +
            "ties it there. An address alone is not enough to record a client's " +
            "answer against a case.",
        },
        already_ingested: false,
      },
    ];
  }

  /**
   * Take named replies in. Files are registered through the same path an
   * upload uses; the client's words become a `client` turn and change no
   * answer — the property the server holds, reproduced here so a test driving
   * the mock cannot pass while the real one would refuse.
   */
  ingestMail(
    caseRef: string,
    messageIds: string[],
  ): AgentStatus & { ingested: MailIngestOutcome[] } {
    const stored = this.get(caseRef);
    const known = new Map(this.mailbox(stored).map((m) => [m.graph_id, m]));
    const ingested: MailIngestOutcome[] = [];

    for (const id of messageIds) {
      const message = known.get(id);
      if (!message) {
        throw new OpsError(
          `Message ${id} is no longer in the mailbox folder this deployment reads.`,
          "OCC_AGENT_MAIL_NOT_FOUND",
        );
      }
      if (!message.correlation.matched) {
        throw new OpsError(
          "This message could not be tied to this case by anything the mail system " +
            "carries, so it was not ingested. " + message.correlation.note,
          "OCC_AGENT_MAIL_UNMATCHED",
        );
      }
      const already = (stored.doc.ingested_mail ?? []).includes(id);
      if (already) {
        ingested.push({
          case_ref: caseRef,
          graph_id: id,
          registered: [],
          skipped: [],
          recorded_text: false,
          recognised: false,
          recognition_note: "",
          already: true,
          statement: "This reply has already been taken in.",
        });
        continue;
      }
      const files = message.attachments.filter((a) => !a.inline && a.readable);
      if (files.length > 0) this.uploadArtefacts(caseRef, files.map((f) => f.name));
      if (message.body_text) {
        stored.doc.messages.push({
          role: "client",
          text: message.body_text,
          at: message.received_at,
          refs: [],
          author: message.sender_name || message.sender,
        });
      }
      stored.doc.ingested_mail = [...(stored.doc.ingested_mail ?? []), id];
      this.record(
        stored,
        "client_reply_ingested",
        "a reply in the OCC mailbox, matched to this case on " +
          message.correlation.bases.join(", "),
      );
      ingested.push({
        case_ref: caseRef,
        graph_id: id,
        registered: files.map((f) => f.name),
        skipped: [],
        recorded_text: Boolean(message.body_text),
        // uploadArtefacts above already registers the sample with Client
        // Onboarding, which is what "read" means here — the mock reproduces
        // the server's behaviour rather than a happier version of it.
        recognised: files.length > 0,
        recognition_note: "",
        already: false,
        statement:
          `Reply from ${message.sender} received ${message.received_at}. ` +
          (files.length > 0 ? `Registered: ${files.map((f) => f.name).join(", ")}. ` : "") +
          "The client's message was recorded on the case. It has NOT been applied to " +
          "any answer — read it and instruct the agent if it should change the case.",
      });
    }
    return { ...this.status(caseRef), ingested };
  }

  private record(stored: StoredRun, action: string, basis: string): void {
    stored.audit.push({
      event_id: `cae-${stored.audit.length + 1}`,
      case_ref: stored.doc.case_ref,
      at: nowIso(),
      action,
      decision_basis: basis,
      runtime_mode: "synthetic",
      resulting_state: stored.doc.state,
    });
  }

  private summary(stored: StoredRun): CaseSummary {
    const doc = stored.doc;
    const onboarding = this.onboardingCase(doc.case_ref);
    return {
      case_ref: doc.case_ref,
      tenant: doc.tenant,
      state: doc.state,
      state_label: STATE_LABELS[doc.state],
      readiness_status: doc.readiness_status,
      runtime_mode: "synthetic",
      synthetic: true,
      open_decisions: doc.open_decisions.filter((d) => d.blocking && d.status === "open").length,
      fixture_id: doc.fixture_id,
      created_at: doc.created_at,
      updated_at: doc.updated_at,
      onboarding_status: onboarding.status,
      onboarding_status_label: onboarding.status_label,
      client_id: onboarding.client_id,
      client_name: onboarding.client_name || onboarding.client_id || "Not yet named",
      onboarding_missing: false,
    };
  }

  private scenarioFor(instruction: string): string {
    const lower = instruction.toLowerCase();
    const hit = SCENARIOS.find((s) =>
      lower.includes(s.instruction.split(".")[0].replace("Onboard ", "").toLowerCase()),
    );
    return hit?.fixture_id ?? "scenario_a_clean";
  }

  private setPackStatus(stored: StoredRun, status: string, note: string): void {
    stored.doc.pack_status = status;
    stored.doc.pack_history = [
      ...stored.doc.pack_history,
      { status, actor: ACTOR, at: nowIso(), note },
    ];
  }

  private blankRun(
    caseRef: string,
    instruction: string,
    scenario: string,
    reportingPeriod: string,
    live = false,
  ): SyntheticRunDoc {
    return {
      case_ref: caseRef,
      tenant: TENANT,
      initiating_user: ACTOR,
      state: S.AWAITING_ONBOARDING,
      runtime_mode: "synthetic",
      version: 1,
      synthetic: true,
      portfolio_id: "",
      dataset: "funded",
      reporting_period: reportingPeriod,
      facts: {},
      received_artefacts: [],
      stage_outcomes: {},
      mapping_report: [],
      open_decisions: [],
      control_results: [],
      planned_pipeline_actions: [],
      orchestration_plan: {},
      assembler_plan: {},
      readiness: {},
      readiness_status: "not_evaluated",
      readiness_package_ref: "",
      review_package_ref: "",
      // The mode an operator CHOSE, which the mock used to discard: every case
      // it made was a rehearsal, so no screen was ever rendered against a real
      // onboarding and a heading reading "Practice run" on a live case went
      // unnoticed until a client onboarding met it. The runtime mode above
      // stays synthetic on both, and correctly: the dry run writes nothing
      // either way, and activation is the one crossing that does.
      mode: live ? "live" : "synthetic",
      pack: {},
      pack_status: "",
      pack_history: [],
      pack_receipt: {},
      activation_intent: {},
      activation_result: {},
      approvals: [],
      blockers: [],
      observations: [],
      messages: [{ role: "operator", text: instruction, at: nowIso(), refs: [] }],
      fixture_id: scenario,
      created_at: nowIso(),
      updated_at: nowIso(),
    };
  }
}

const COMPLETED_STAGES: Record<string, string> = {
  onboard: "deterministic_execution_completed",
  transform: "deterministic_execution_completed",
  validate: "deterministic_execution_completed",
  stamp: "deterministic_execution_completed",
  assemble: "deterministic_execution_completed",
};

/** From which states approving the onboarding releases the execution half. */
const PACK_OR_START = new Set<string>([
  S.AWAITING_ONBOARDING,
  S.PACK_DRAFTED,
  S.PACK_REVIEW_REQUIRED,
  S.PACK_APPROVED_TO_SEND,
  S.PACK_SENT,
]);

const STEP_ACTIONS: Record<string, string> = {
  "information-requests": "request_client_information",
  "pack/draft": "draft_onboarding_pack",
  "pack/approve": "approve_pack_to_send",
  "pack/send": "send_onboarding_pack",
  review: "request_activation",
  "activation/approve": "approve_activation",
  "activation/confirm": "confirm_activation",
  submit: "submit_for_approval",
  approve: "approve_onboarding",
  "request-changes": "request_changes",
  run: "run_synthetic_onboarding",
  plan: "generate_orchestration_plan",
  "readiness/approve": "approve_execution_readiness",
  cancel: "cancel_run",
};

const MATERIAL_STEPS = new Set([
  "submit",
  "approve",
  "readiness/approve",
  "pack/approve",
  "pack/send",
  "activation/approve",
  "activation/confirm",
  "cancel",
]);

const PROPOSAL_SUMMARIES: Record<string, string> = {
  submit: "Submit the onboarding for approval.",
  "pack/approve": "Approve the pack for sending.",
  "pack/send": "Record the pack as issued to the client.",
  "activation/approve": "Approve the configuration for activation. This starts nothing.",
  "activation/confirm": "Confirm activation. This is production.",
  approve: "Approve the onboarding.",
  "readiness/approve": "Approve readiness for execution.",
  cancel: "Cancel this practice case.",
};

/** What a bare "yes" means once the onboarding is approved. */
const PENDING_CONFIRMATION: Record<string, string> = {
  [S.READY_TO_RUN]: "run",
  [S.SYNTHETIC_ONBOARDING_PASSED]: "plan",
  [S.ORCHESTRATION_PLAN_GENERATED]: "readiness/approve",
  [S.EXECUTION_APPROVAL_REQUIRED]: "readiness/approve",
  [S.READY_FOR_EXECUTION]: "review",
  [S.READY_FOR_REVIEW]: "activation/approve",
  // ACTIVATION_CONFIRMATION_REQUIRED is deliberately absent: a bare "yes" must
  // never start production.
};

const MOCK_PLAN = {
  target: "mi",
  outcome: "mi",
  valid: true,
  execution_status: "not_executed",
  steps: [
    { step: "onboard", agent: "Onboarding Agent", produces: ["18_central_lender_tape.csv"] },
    { step: "assemble", agent: "Assembler Agent", produces: ["platform_canonical_typed.csv"] },
  ],
};

/** The mock interpreter: what one instruction says, and nothing more. */
function interpret(instruction: string) {
  const clientName =
    /onboard\s+([A-Z][\w ]+?)[.,]/i.exec(instruction)?.[1]?.trim() ?? "New client";
  // `.` and `-` are legal INSIDE an identifier and only noise at the very end,
  // so the pattern keeps them and the trim drops the sentence punctuation.
  const portfolioId = (
    /portfolio id\s+([\w_.-]+)/i.exec(instruction)?.[1] ?? "direct_001"
  ).replace(/[.,;:_-]+$/, "");
  const lower = instruction.toLowerCase();
  const products = ["mi"];
  if (/investor reporting|annex ?12/.test(lower)) products.push("investor_reporting");
  if (/annex ?2|regulatory reporting/.test(lower)) products.push("esma_annex2");
  return {
    clientName,
    portfolioId,
    // A declared pipeline stream is a SEPARATE registration, mirroring the
    // server: "a pipeline and a funded book" is two streams, never one.
    pipeline: /\bpipeline\b/.test(lower),
    products,
    jurisdiction: /\buk\b|united kingdom|british/.test(lower) ? "GB" : "",
    reportingPeriod:
      /(?:first\s+)?report(?:ing)?\s+(?:date|period)[^.]{0,40}?(\d{4}-\d{2}-\d{2})/i.exec(
        instruction,
      )?.[1] ?? "",
  };
}

/**
 * One plausible, valid answer for one question, from its own declaration.
 *
 * `undefined` means leave it unanswered — used where nothing sensible can be
 * made up, which is better than filling a box with noise a reviewer has to
 * unpick. Kept in step with `fixtures._answer_for` on the server.
 */
function syntheticAnswer(
  f: { key: string; label?: string; type?: string; validation?: string;
       options?: { value?: string }[] },
  domain: string,
): unknown {
  const kind = (f.type ?? "text").toLowerCase();
  const rule = (f.validation ?? "").toLowerCase();
  const label = f.label ?? "this";
  const slug = (s: string) =>
    s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  const options = (f.options ?? []).map((o) => o.value).filter(Boolean) as string[];

  if (options.length > 0) return kind === "multi_enum" ? [options[0]] : options[0];
  if (rule === "lei" || kind === "lei") return "894500SYNTHETIC00042";
  if (rule === "email" || kind === "email") return `${slug(label) || "contact"}@${domain}`;
  if (rule === "colour" || kind === "colour") return "#1F3B5C";
  if (rule === "country" || kind === "country") return "GB";
  if (rule === "currency" || kind === "currency") return "GBP";
  if (kind === "boolean") return true;
  if (kind === "date") return "2026-06-30";
  if (kind === "number" || kind === "integer" || kind === "decimal") return 0;
  if (kind === "multiline")
    return `Practice answer for ${label.toLowerCase()}. Synthetic content for a rehearsal; no client supplied this.`;
  if (kind === "text" || kind === "identifier") return `Practice ${label.toLowerCase()}`;
  return undefined;
}

/** Turn `{"section.field": value}` into a `recordResponse` payload. */
function answersFor(
  onboarding: OnboardingCase,
  items: ChecklistRow[],
  response: Record<string, string>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const item of items) {
    const key = `${item.section}.${item.field}`;
    if (!(key in response)) continue;
    if (item.index === null || item.index === undefined) {
      const block = (out[item.section] ?? {}) as Record<string, unknown>;
      block[item.field] = response[key];
      out[item.section] = block;
      continue;
    }
    let rows = out[item.section] as Record<string, unknown>[] | undefined;
    if (!Array.isArray(rows)) {
      rows = ((onboarding.answers[item.section] ?? []) as Record<string, unknown>[]).map((r) => ({
        ...r,
      }));
      out[item.section] = rows;
    }
    if (rows[item.index]) rows[item.index][item.field] = response[key];
  }
  return out;
}

function scenarioById(fixtureId: string): ScenarioSummary {
  const scenario = SCENARIOS.find((s) => s.fixture_id === fixtureId);
  if (!scenario) {
    throw new OpsError("That practice scenario could not be found.", "OCC_AGENT_FIXTURE_NOT_FOUND");
  }
  return scenario;
}

function nowIso(): string {
  return new Date().toISOString();
}


/** What the pack tells a client about mappings. A governed decision, stated. */
const MAPPING_STATEMENT =
  "Trakt does not ask you to map your fields to ours. Send a representative " +
  "file and Trakt will propose the mapping itself; an operator reviews and " +
  "approves it during the first ingestion, and it is then fixed for every " +
  "later delivery.";

/** What the approver is told about mappings. */
const MAPPING_NOTE =
  "Field mappings are NOT part of this configuration and were not collected. " +
  "They are proposed by Trakt from the first representative delivery, " +
  "reviewed and approved by an operator during that first ingestion, and then " +
  "fingerprinted and fixed. Approving this activation does not approve any " +
  "mapping.";

/** What the approver is told about user access. */
const ACCESS_NOTE =
  "User access recorded during onboarding is a REQUIREMENT, not a grant. " +
  "Trakt reads its operators from environment configuration in this " +
  "environment, so nothing below has been provisioned.";

/** Never "queued" or "pending": the pack is not going anywhere on its own. */
const RECORD_ONLY_STATEMENT =
  "Recorded as issued. Trakt did not send it: no email integration is enabled " +
  "in this environment. Send the approved pack and covering email from the " +
  "record.";

function present(value: unknown): boolean {
  if (value === null || value === undefined) return false;
  if (typeof value === "boolean") return true;
  if (Array.isArray(value)) return value.length > 0;
  return String(value).trim() !== "";
}

function packEmailBody(clientName: string, outstanding: number): string {
  return [
    `Dear ${clientName || "there"},`,
    "",
    "We are setting your portfolio up on Trakt. Attached is the onboarding " +
      "pack: it lists what we still need from you, and the files to send with it.",
    "",
    outstanding > 0
      ? `There are ${outstanding} question${outstanding === 1 ? "" : "s"} outstanding.`
      : "We have everything we need — please confirm the details in the pack are right.",
    "",
    "Kind regards,",
    "Trakt Operations",
  ].join("\n");
}

function packDocument(clientName: string, sections: PackSection[]): string {
  const lines = [`# Onboarding — ${clientName}`, ""];
  for (const section of sections) {
    lines.push(`## ${section.label}`, "");
    for (const question of section.questions) {
      lines.push(`- [${question.status === "answered" ? "answered" : "  "}] ${question.label}`);
    }
    lines.push("");
  }
  lines.push("## About field mappings", "", MAPPING_STATEMENT, "");
  return lines.join("\n");
}

function reviewDocument(pkg: {
  client_name: string;
  sections: { label: string; rows: { label: string; value: unknown }[] }[];
  mapping_note: string;
}): string {
  const lines = [`# Review — ${pkg.client_name}`, "", "## What Trakt holds", ""];
  for (const section of pkg.sections) {
    if (section.rows.length === 0) continue;
    lines.push(`### ${section.label}`, "");
    for (const row of section.rows) lines.push(`- **${row.label}**: ${String(row.value)}`);
    lines.push("");
  }
  lines.push("## Field mappings", "", pkg.mapping_note, "");
  return lines.join("\n");
}


/** The five categories, mirroring `occ_agent.classification`. */
const CATEGORY_NUMBERS: Record<FieldCategory, number> = {
  not_applicable: 0,
  known: 1,
  client: 2,
  derived: 3,
  first_delivery: 4,
  internal: 5,
};

const CATEGORY_LABELS: Record<FieldCategory, string> = {
  not_applicable: "Does not apply to this client",
  known: "Already known",
  client: "Only the client can answer",
  derived: "Trakt works it out",
  first_delivery: "Learned from the first delivery",
  internal: "Internal operator decision",
};

/** The catalogue's `source` axis, mapped onto the categories. */
const BY_SOURCE: Record<string, FieldCategory> = {
  client_supplied: "client",
  operator_supplied: "internal",
  trakt_default: "derived",
  derived: "derived",
  system_generated: "derived",
  // "read from data the client uploads" — the Onboarding Agent's job.
  inferred: "first_delivery",
};

const REASONS: Record<FieldCategory, string> = {
  client: "only the client knows it",
  derived: "Trakt works it out",
  first_delivery: "the Onboarding Agent reads it from the first delivery",
  internal: "an operator decision, not a client question",
  known: "already answered",
  not_applicable: "does not apply here",
};

const ORIGIN: Record<string, string> = {
  client_supplied: "from the client",
  operator_supplied: "set by an operator",
  inferred: "read from a delivery",
  derived: "derived by Trakt",
  trakt_default: "a Trakt default",
  system_generated: "minted by Trakt",
};

/** Fields whose `source` is right for generation but wrong for who to ask. */
const INTERNAL_FIELDS = new Set([
  "client.client_id",
  "portfolios.portfolio_id",
  "sources.sample_provided",
  "sources.mapping_complete",
  "sources.dataset",
]);

/**
 * Sections asked once something else exists.
 *
 * Read from the catalogue's own `deferred_until`, so which questions wait for
 * a delivery is a governed decision rather than a constant here.
 */
const DEFERRED_SECTIONS = new Set(
  (CATALOGUE.sections ?? [])
    .filter((s: { deferred_until?: string }) => Boolean(s.deferred_until))
    .map((s: { key: string }) => s.key),
);

/** An identifier a client never saw is not put up for confirmation. */
const NEVER_CONFIRMED_SOURCES = new Set([
  "operator_supplied",
  "system_generated",
  "derived",
]);

/** The catalogue's own `asked_when` conditions, as predicates over products. */
const ASKED_WHEN: Record<string, (products: Set<string>) => boolean> = {
  "contacts.investor_report_recipients": (p) => p.has("investor_reporting"),
  "presentation.brand_colour": (p) => p.has("mi"),
  "presentation.logo_uri": (p) => p.has("mi"),
  "presentation.disclaimer": (p) => p.has("mi"),
  "presentation.reporting_calendar_note": (p) => p.has("mi"),
  "data_semantics.cashflow_basis": (p) => p.has("mi"),
  "data_semantics.valuation_basis": (p) => p.has("mi"),
  "data_semantics.accrued_interest_treatment": (p) => p.has("mi"),
};

/** The client-facing steps, in the order a client meets them. */
const FORM_STEPS: { key: string; label: string; help: string; sections: string[] }[] = [
  {
    key: "about_you",
    label: "About your business",
    help: "The legal entity behind the portfolio, and who we should talk to.",
    sections: ["client", "entities", "contacts"],
  },
  {
    key: "portfolios",
    label: "Your portfolios",
    help: "One set of answers per book. Everything above is asked once.",
    sections: ["portfolios"],
  },
  {
    key: "deliveries",
    label: "Sending us your data",
    help: "How each delivery reaches Trakt.",
    sections: ["sources"],
  },
  {
    key: "reporting",
    label: "Your reports",
    help: "What you receive, and how it should look.",
    sections: ["reporting", "regime", "presentation"],
  },
  {
    key: "access",
    label: "Who needs access",
    help: "People at your end who need Trakt, or who receive reports.",
    sections: ["access"],
  },
  {
    key: "anything_else",
    label: "Anything else we need to know",
    help:
      "Anything specific to your business, your asset class or this portfolio " +
      "that would help Trakt read your data correctly. Leave it blank if " +
      "nothing comes to mind.",
    sections: ["additional_context"],
  },
  // No "what your numbers mean" step. `data_semantics` is deferred by the
  // catalogue until a representative file has arrived, so those questions are
  // put to a client against what was actually found in their data rather than
  // asked in the abstract. A demo that still asked them would show a product
  // that no longer exists.
];
