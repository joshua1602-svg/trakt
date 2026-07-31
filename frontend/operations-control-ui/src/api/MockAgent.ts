import { MockOnboarding } from "./MockOnboarding";
import { OpsError } from "./OpsClient";
import type {
  AgentAudit,
  AgentChecklist,
  AgentMeta,
  AgentPreview,
  AgentProposal,
  AgentReadinessPackage,
  AgentStatus,
  AgentTurn,
  CaseSummary,
  DecisionCard,
  ExecutionFacts,
  LifecycleState,
  ScenarioSummary,
  SyntheticRunDoc,
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
  READY_TO_RUN: "READY_TO_RUN",
  SYNTHETIC_ONBOARDING_RUNNING: "SYNTHETIC_ONBOARDING_RUNNING",
  EXCEPTIONS_REQUIRE_INPUT: "EXCEPTIONS_REQUIRE_INPUT",
  SYNTHETIC_ONBOARDING_PASSED: "SYNTHETIC_ONBOARDING_PASSED",
  ORCHESTRATION_PLAN_GENERATED: "ORCHESTRATION_PLAN_GENERATED",
  EXECUTION_APPROVAL_REQUIRED: "EXECUTION_APPROVAL_REQUIRED",
  READY_FOR_EXECUTION: "READY_FOR_EXECUTION",
  BLOCKED: "BLOCKED",
  CANCELLED: "CANCELLED",
} as const;

const STATE_LABELS: Record<string, string> = {
  [S.AWAITING_ONBOARDING]: "Working through onboarding",
  [S.READY_TO_RUN]: "Ready to run",
  [S.SYNTHETIC_ONBOARDING_RUNNING]: "Synthetic onboarding running",
  [S.EXCEPTIONS_REQUIRE_INPUT]: "Exceptions need your input",
  [S.SYNTHETIC_ONBOARDING_PASSED]: "Synthetic onboarding passed",
  [S.ORCHESTRATION_PLAN_GENERATED]: "Execution plan prepared",
  [S.EXECUTION_APPROVAL_REQUIRED]: "Readiness needs your approval",
  [S.READY_FOR_EXECUTION]: "READY_FOR_EXECUTION",
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

const ALLOWED: Record<string, string[]> = {
  [S.AWAITING_ONBOARDING]: ["register_synthetic_artefact", "cancel_run"],
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
  [S.READY_FOR_EXECUTION]: [],
  [S.BLOCKED]: [
    "resolve_decision",
    "register_synthetic_artefact",
    "run_synthetic_onboarding",
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

interface StoredRun {
  doc: SyntheticRunDoc;
  reached: Set<string>;
  scenario: string;
  audit: Record<string, unknown>[];
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
    source_column: "Current Balance",
    source_columns: ["Current Balance", "Principal Balance"],
    target_field: "current_principal_balance",
  },
};

const TENANT = "Alpine Capital";
const ACTOR = "Operator";

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

  create(instruction: string, fixtureId = ""): AgentStatus {
    const scenario = fixtureId || this.scenarioFor(instruction);
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

    const doc = this.blankRun(opened.case_id, instruction, scenario, facts.reportingPeriod);
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
      state: lifecycle(stored.doc.state).find((s) => s.state === stored.doc.state)!,
      lifecycle: lifecycle(stored.doc.state, stored.reached),
      stage_outcomes: stored.doc.stage_outcomes,
      readiness: this.readiness(stored, onboarding),
      policy: POLICY,
      open_decisions: stored.doc.open_decisions,
      observations: stored.doc.observations,
      blockers: stored.doc.blockers,
      occ_links: [
        {
          label: "Client onboarding",
          to: `/onboarding/${caseRef}`,
          why: "The onboarding case itself, in the screens an operator normally works it in.",
        },
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
      anything_simulated: false,
      anything_blocked: stored.doc.state === S.BLOCKED,
      configuration_written: onboarding.status === "activated",
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
    return { ...this.status(caseRef), reply, applied: true, proposal: null };
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

  setRunTarget(
    caseRef: string,
    input: { portfolio_id?: string; dataset?: string; reporting_period?: string },
  ): AgentStatus {
    const stored = this.get(caseRef);
    if (input.portfolio_id) stored.doc.portfolio_id = input.portfolio_id;
    if (input.dataset) stored.doc.dataset = input.dataset;
    if (input.reporting_period) stored.doc.reporting_period = input.reporting_period;
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
        intended_live_uri: `blob://raw-v2/${facts.client_id}/direct/funded/${facts.cadence}/${facts.portfolio_id}/${stored.doc.reporting_period}/${name}`,
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

    // The execution half.
    for (const step of ["run", "plan", "readiness/approve"]) {
      const now = this.get(caseRef).doc.state;
      if (now === S.BLOCKED || now === S.EXCEPTIONS_REQUIRE_INPUT) break;
      this.step(caseRef, step);
    }
    return this.status(caseRef);
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
      case "approve":
        this.onboarding.approve(
          caseRef,
          String(body.reason ?? "") || "Approved in a practice case.",
          ACTOR,
        );
        if (stored.doc.state === S.AWAITING_ONBOARDING) this.move(stored, S.READY_TO_RUN);
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
        this.record(stored, "synthetic_onboarding_started", "the conductor ran over the adapter");
        if (stored.scenario === "scenario_b_ambiguous_mapping") {
          stored.doc.stage_outcomes = { onboard: "human_input_required" };
          stored.doc.open_decisions = [{ ...AMBIGUOUS_DECISION }];
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
      case "cancel":
        this.move(stored, S.CANCELLED);
        this.record(stored, "practice_case_cancelled", "cancelled by the operator");
        break;
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

  private blankRun(
    caseRef: string,
    instruction: string,
    scenario: string,
    reportingPeriod: string,
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

const STEP_ACTIONS: Record<string, string> = {
  "information-requests": "request_client_information",
  submit: "submit_for_approval",
  approve: "approve_onboarding",
  "request-changes": "request_changes",
  run: "run_synthetic_onboarding",
  plan: "generate_orchestration_plan",
  "readiness/approve": "approve_execution_readiness",
  cancel: "cancel_run",
};

const MATERIAL_STEPS = new Set(["submit", "approve", "readiness/approve", "cancel"]);

const PROPOSAL_SUMMARIES: Record<string, string> = {
  submit: "Submit the onboarding for approval.",
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
    products,
    jurisdiction: /\buk\b|united kingdom|british/.test(lower) ? "GB" : "",
    reportingPeriod:
      /(?:first\s+)?report(?:ing)?\s+(?:date|period)[^.]{0,40}?(\d{4}-\d{2}-\d{2})/i.exec(
        instruction,
      )?.[1] ?? "",
  };
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
