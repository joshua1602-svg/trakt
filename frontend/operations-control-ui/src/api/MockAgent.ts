import { OpsError } from "./OpsClient";
import type {
  AgentAudit,
  AgentMeta,
  AgentPack,
  AgentProposal,
  AgentReadinessPackage,
  AgentStatus,
  AgentTurn,
  CaseSummary,
  DecisionCard,
  LifecycleState,
  ScenarioSummary,
  SyntheticCaseDoc,
} from "./agentTypes";

/**
 * A stateful stand-in for the OCC Agent API, mirroring the backend's lifecycle
 * and its refusals. Used by `VITE_OPS_MODE=mock` and by the tests.
 *
 * It is a fixture, not a second implementation: it moves a case through the
 * same states in the same order and refuses the same things (an action the
 * current state does not allow; readiness before every criterion passes), so a
 * test that passes here would pass against the real API for the same reason.
 * All the real deterministic work — mapping, validation, materiality — happens
 * server-side and is simply reflected here as outcomes.
 */

const S = {
  CASE_CREATED: "CASE_CREATED",
  REQUIREMENTS_EXTRACTED: "REQUIREMENTS_EXTRACTED",
  REQUIREMENTS_CONFIRMATION_REQUIRED: "REQUIREMENTS_CONFIRMATION_REQUIRED",
  REQUIREMENTS_CONFIRMED: "REQUIREMENTS_CONFIRMED",
  ONBOARDING_PACK_GENERATED: "ONBOARDING_PACK_GENERATED",
  ONBOARDING_PACK_APPROVAL_REQUIRED: "ONBOARDING_PACK_APPROVAL_REQUIRED",
  ONBOARDING_PACK_ISSUED_SYNTHETICALLY: "ONBOARDING_PACK_ISSUED_SYNTHETICALLY",
  AWAITING_CLIENT_INPUT: "AWAITING_CLIENT_INPUT",
  ARTEFACTS_RECEIVED: "ARTEFACTS_RECEIVED",
  ARTEFACTS_CLASSIFIED: "ARTEFACTS_CLASSIFIED",
  CONFIG_DRAFTED: "CONFIG_DRAFTED",
  CONFIG_APPROVAL_REQUIRED: "CONFIG_APPROVAL_REQUIRED",
  CONFIG_APPROVED: "CONFIG_APPROVED",
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
  [S.CASE_CREATED]: "Case created",
  [S.REQUIREMENTS_EXTRACTED]: "Interpretation proposed",
  [S.REQUIREMENTS_CONFIRMATION_REQUIRED]: "Interpretation needs your confirmation",
  [S.REQUIREMENTS_CONFIRMED]: "Requirements confirmed",
  [S.ONBOARDING_PACK_GENERATED]: "Onboarding pack drafted",
  [S.ONBOARDING_PACK_APPROVAL_REQUIRED]: "Onboarding pack needs your approval",
  [S.ONBOARDING_PACK_ISSUED_SYNTHETICALLY]: "Pack recorded as issued (synthetically)",
  [S.AWAITING_CLIENT_INPUT]: "Waiting for the client response",
  [S.ARTEFACTS_RECEIVED]: "Synthetic artefacts received",
  [S.ARTEFACTS_CLASSIFIED]: "Artefacts recognised",
  [S.CONFIG_DRAFTED]: "Configuration proposed",
  [S.CONFIG_APPROVAL_REQUIRED]: "Configuration needs your approval",
  [S.CONFIG_APPROVED]: "Configuration approved",
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

const ALLOWED: Record<string, string[]> = {
  [S.CASE_CREATED]: ["correct_requirements", "cancel_case"],
  [S.REQUIREMENTS_EXTRACTED]: ["confirm_requirements", "correct_requirements", "cancel_case"],
  [S.REQUIREMENTS_CONFIRMATION_REQUIRED]: [
    "confirm_requirements",
    "correct_requirements",
    "cancel_case",
  ],
  [S.REQUIREMENTS_CONFIRMED]: [
    "generate_onboarding_pack",
    "correct_requirements",
    "return_to_stage",
    "cancel_case",
  ],
  [S.ONBOARDING_PACK_GENERATED]: [
    "approve_onboarding_pack",
    "generate_onboarding_pack",
    "correct_requirements",
    "cancel_case",
  ],
  [S.ONBOARDING_PACK_APPROVAL_REQUIRED]: [
    "approve_onboarding_pack",
    "generate_onboarding_pack",
    "correct_requirements",
    "cancel_case",
  ],
  [S.ONBOARDING_PACK_ISSUED_SYNTHETICALLY]: ["register_synthetic_artefact", "cancel_case"],
  [S.AWAITING_CLIENT_INPUT]: ["register_synthetic_artefact", "return_to_stage", "cancel_case"],
  [S.ARTEFACTS_RECEIVED]: ["register_synthetic_artefact", "classify_artefacts", "cancel_case"],
  [S.ARTEFACTS_CLASSIFIED]: [
    "draft_client_config",
    "register_synthetic_artefact",
    "resolve_decision",
    "cancel_case",
  ],
  [S.CONFIG_DRAFTED]: [
    "approve_client_config",
    "correct_client_config",
    "draft_client_config",
    "cancel_case",
  ],
  [S.CONFIG_APPROVAL_REQUIRED]: [
    "approve_client_config",
    "correct_client_config",
    "cancel_case",
  ],
  [S.CONFIG_APPROVED]: [
    "run_synthetic_onboarding",
    "correct_client_config",
    "return_to_stage",
    "cancel_case",
  ],
  [S.SYNTHETIC_ONBOARDING_RUNNING]: ["cancel_case"],
  [S.EXCEPTIONS_REQUIRE_INPUT]: [
    "resolve_decision",
    "acknowledge_exception",
    "run_synthetic_onboarding",
    "correct_client_config",
    "return_to_stage",
    "cancel_case",
  ],
  [S.SYNTHETIC_ONBOARDING_PASSED]: [
    "generate_orchestration_plan",
    "return_to_stage",
    "cancel_case",
  ],
  [S.ORCHESTRATION_PLAN_GENERATED]: [
    "approve_execution_readiness",
    "return_to_stage",
    "cancel_case",
  ],
  [S.EXECUTION_APPROVAL_REQUIRED]: [
    "approve_execution_readiness",
    "return_to_stage",
    "cancel_case",
  ],
  [S.READY_FOR_EXECUTION]: [],
  [S.BLOCKED]: [
    "resolve_decision",
    "register_synthetic_artefact",
    "correct_requirements",
    "correct_client_config",
    "return_to_stage",
    "cancel_case",
  ],
  [S.CANCELLED]: [],
};

const RETURNABLE = [
  S.REQUIREMENTS_EXTRACTED,
  S.ONBOARDING_PACK_GENERATED,
  S.AWAITING_CLIENT_INPUT,
  S.ARTEFACTS_RECEIVED,
  S.CONFIG_DRAFTED,
];

const POLICY = {
  runtime_mode: "synthetic",
  allow_external_email: false,
  allow_live_blob_write: false,
  allow_live_pipeline_trigger: false,
  allow_production_config_write: false,
  allow_publish: false,
  allow_live_case_access: false,
};

const SCENARIOS: ScenarioSummary[] = [
  {
    fixture_id: "scenario_a_clean",
    label: "A — Clean onboarding",
    description:
      "Complete artefacts, valid configuration and high-confidence mappings. Reaches READY_FOR_EXECUTION with no blocking control.",
    instruction:
      "Onboard Northstar Lending. It is a UK equity-release portfolio. They require monthly portfolio MI. Portfolio id direct_101. First reporting date 2026-06-30.",
    files: [
      { filename: "northstar_loan_extract_202606.csv", artefact_type: "loan_extract", bytes: 4096 },
    ],
    expected_state: S.READY_FOR_EXECUTION,
    expected_human_steps: ["confirm the interpretation", "approve the pack", "approve readiness"],
    demonstrates: "The whole operating process end to end with nothing in the way.",
  },
  {
    fixture_id: "scenario_b_ambiguous_mapping",
    label: "B — Ambiguous mapping",
    description:
      "Two source columns resolve to the same canonical field. The run halts for a human decision, then reaches READY_FOR_EXECUTION.",
    instruction:
      "Onboard Harbour Point Capital. UK equity release. Monthly portfolio MI. Portfolio id direct_102. First reporting date 2026-06-30.",
    files: [
      {
        filename: "harbourpoint_loan_extract_202606.csv",
        artefact_type: "loan_extract",
        bytes: 4400,
      },
    ],
    expected_state: S.READY_FOR_EXECUTION,
    expected_human_steps: ["settle the ambiguous mapping"],
    demonstrates: "A control the engine cannot settle alone.",
  },
  {
    fixture_id: "scenario_c_missing_artefact",
    label: "C — Missing mandatory artefact",
    description: "A required input role is absent, so the case stays blocked.",
    instruction:
      "Onboard Kestrel Mutual. UK equity release. Monthly portfolio MI. Portfolio id direct_103. First reporting date 2026-06-30.",
    files: [
      {
        filename: "kestrel_property_extract_202606.csv",
        artefact_type: "property_extract",
        bytes: 1200,
      },
    ],
    expected_state: S.BLOCKED,
    expected_human_steps: [],
    demonstrates: "The configured input requirements refusing an incomplete pack.",
  },
  {
    fixture_id: "scenario_d_product_config_gap",
    label: "D — Product configuration gap",
    description:
      "Core onboarding proceeds, but the selected product needs configuration the client has not supplied.",
    instruction:
      "Onboard Aldermere Advances. UK equity release. Portfolio MI and regulatory reporting. Portfolio id direct_104. First reporting date 2026-06-30.",
    files: [
      { filename: "aldermere_loan_extract_202606.csv", artefact_type: "loan_extract", bytes: 4096 },
    ],
    expected_state: S.BLOCKED,
    expected_human_steps: ["supply the product configuration"],
    demonstrates: "A product-specific configuration requirement.",
  },
  {
    fixture_id: "scenario_e_business_rule_failure",
    label: "E — Material business-rule failure",
    description:
      "Canonical transformation succeeds; the deterministic business rules then fail at BLOCKING materiality.",
    instruction:
      "Onboard Brackenfield Finance. UK equity release. Monthly portfolio MI. Portfolio id direct_105. First reporting date 2026-06-30.",
    files: [
      {
        filename: "brackenfield_loan_extract_202606.csv",
        artefact_type: "loan_extract",
        bytes: 4096,
      },
    ],
    expected_state: S.BLOCKED,
    expected_human_steps: [],
    demonstrates: "A deterministic blocker that natural language cannot bypass.",
  },
];

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

interface StoredCase {
  doc: SyntheticCaseDoc;
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
  downstream_consequence: "The synthetic run cannot continue until this is answered.",
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

export class MockAgent {
  private cases = new Map<string, StoredCase>();
  private nextId = 1;

  meta(): AgentMeta {
    return {
      enabled: true,
      flag: "OCC_AGENT_SYNTHETIC_ENABLED",
      runtime_mode: "synthetic",
      policy: POLICY,
      lifecycle: lifecycle(),
      returnable_states: [...RETURNABLE],
      scenarios: SCENARIOS,
      tenant: "Alpine Capital",
    };
  }

  list(state?: string): CaseSummary[] {
    const rows = [...this.cases.values()].map((c) => this.summary(c));
    return state ? rows.filter((r) => r.state === state) : rows;
  }

  create(instruction: string, fixtureId = ""): AgentStatus {
    const id = `CASE-MOCK${String(this.nextId++).padStart(4, "0")}`;
    const scenario = fixtureId || this.scenarioFor(instruction);
    const doc = this.blankDoc(id, instruction, scenario);
    const stored: StoredCase = {
      doc,
      reached: new Set([S.CASE_CREATED, S.REQUIREMENTS_EXTRACTED]),
      scenario,
      audit: [],
    };
    this.cases.set(id, stored);
    this.record(stored, "case_created", "an operator opened a synthetic case");
    this.move(stored, S.REQUIREMENTS_CONFIRMATION_REQUIRED);
    return this.status(id);
  }

  get(caseId: string): StoredCase {
    const stored = this.cases.get(caseId);
    if (!stored) {
      throw new OpsError("That synthetic case could not be found.", "OCC_AGENT_CASE_NOT_FOUND");
    }
    return stored;
  }

  status(caseId: string): AgentStatus {
    const stored = this.get(caseId);
    const readiness = this.readiness(stored);
    return {
      case: stored.doc,
      summary: this.summary(stored),
      state: lifecycle(stored.doc.state).find((s) => s.state === stored.doc.state)!,
      lifecycle: lifecycle(stored.doc.state, stored.reached),
      stage_outcomes: stored.doc.stage_outcomes,
      readiness,
      policy: POLICY,
      open_decisions: stored.doc.open_decisions,
      observations: stored.doc.observations,
      blockers: stored.doc.blockers,
      occ_links: [
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
    };
  }

  step(caseId: string, step: string): AgentStatus {
    const stored = this.get(caseId);
    const action = STEP_ACTIONS[step];
    if (!action) throw new OpsError("That is not something Trakt can do.", "OCC_AGENT_UNKNOWN_STEP");
    this.requireAction(stored, action);
    this.applyStep(stored, step);
    return this.status(caseId);
  }

  instruct(caseId: string, text: string, confirm: boolean): AgentTurn {
    const stored = this.get(caseId);
    const lower = text.trim().toLowerCase();
    stored.doc.messages.push({ role: "operator", text, at: nowIso(), refs: [] });

    if (lower.endsWith("?") || /^(what|why|which|how|when|who)\b/.test(lower)) {
      const reply = this.answer(stored, lower);
      stored.doc.messages.push({ role: "agent", text: reply, at: nowIso(), refs: [] });
      return { ...this.status(caseId), reply, applied: false, proposal: null };
    }

    const step = this.stepForInstruction(lower, stored);
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
      return { ...this.status(caseId), reply: proposal.summary, applied: false, proposal };
    }

    this.applyStep(stored, step);
    const reply = this.statusSentence(stored);
    stored.doc.messages.push({ role: "agent", text: reply, at: nowIso(), refs: [] });
    return { ...this.status(caseId), reply, applied: true, proposal: null };
  }

  answerDecision(
    caseId: string,
    input: { decision_id: string; action: string; value?: string; reason?: string },
  ): AgentStatus {
    const stored = this.get(caseId);
    this.requireAction(stored, "resolve_decision");
    const decision = stored.doc.open_decisions.find((d) => d.decision_id === input.decision_id);
    if (!decision) {
      throw new OpsError("That decision could not be found on this case.", "OCC_AGENT_DECISION_NOT_FOUND");
    }
    decision.status = input.action === "reject" ? "rejected" : "approved";
    decision.resolved_value = input.value || decision.recommendation;
    decision.resolved_by = "Operator";
    stored.doc.approval_history.push({
      subject: "decision",
      decision: decision.status,
      reference: decision.decision_id,
    });
    this.record(stored, "human_decision_recorded", input.reason || `operator chose '${input.action}'`);
    if (!stored.doc.open_decisions.some((d) => d.blocking && d.status === "open")) {
      // The affected controls rerun, exactly as the backend reruns them.
      this.move(stored, S.SYNTHETIC_ONBOARDING_RUNNING);
      this.move(stored, S.SYNTHETIC_ONBOARDING_PASSED);
      stored.doc.blockers = [];
    }
    return this.status(caseId);
  }

  uploadArtefacts(caseId: string, names: string[]): AgentStatus {
    const stored = this.get(caseId);
    this.requireAction(stored, "register_synthetic_artefact");
    for (const name of names) {
      stored.doc.received_artefacts.push({
        artefact_id: `sart-${this.nextId++}`,
        source_file: name,
        artefact_type: name.toLowerCase().includes("loan") ? "loan_extract" : "property_extract",
        synthetic_location: `synthetic_cases/${stored.doc.case_id}/artefacts/${name}`,
        intended_live_uri: `blob://raw-v2/${stored.doc.client_id}/direct/funded/monthly/${stored.doc.portfolio_id}/2026-06-30/${name}`,
        execution_status: "simulated_only",
        sha256: "sha256:mock",
        size: 4096,
        columns: [],
        row_count: 24,
        recognition_confidence: 1,
        recognition_basis: "filename_keyword",
        provided_by: "Operator",
        provided_at: nowIso(),
        fixture_id: stored.scenario,
      });
    }
    this.move(stored, S.ARTEFACTS_RECEIVED);
    this.record(stored, "synthetic_artefact_registered", "stored in the synthetic case sandbox");
    return this.status(caseId);
  }

  loadFixtureArtefacts(caseId: string, fixtureId: string): AgentStatus {
    const scenario = SCENARIOS.find((s) => s.fixture_id === fixtureId);
    if (!scenario) {
      throw new OpsError("That synthetic scenario could not be found.", "OCC_AGENT_FIXTURE_NOT_FOUND");
    }
    const stored = this.get(caseId);
    stored.scenario = fixtureId;
    stored.doc.fixture_id = fixtureId;
    return this.uploadArtefacts(caseId, scenario.files.map((f) => f.filename));
  }

  returnToStage(caseId: string, targetState: string): AgentStatus {
    const stored = this.get(caseId);
    this.requireAction(stored, "return_to_stage");
    if (!RETURNABLE.includes(targetState as (typeof RETURNABLE)[number])) {
      throw new OpsError(
        `This case cannot move to ${STATE_LABELS[targetState] ?? targetState}.`,
        "OCC_AGENT_ILLEGAL_TRANSITION",
      );
    }
    this.move(stored, targetState);
    this.record(stored, "returned_to_earlier_stage", "the operator reopened a stage");
    return this.status(caseId);
  }

  runScenario(fixtureId: string): AgentStatus {
    const scenario = SCENARIOS.find((s) => s.fixture_id === fixtureId);
    if (!scenario) {
      throw new OpsError("That synthetic scenario could not be found.", "OCC_AGENT_FIXTURE_NOT_FOUND");
    }
    const created = this.create(scenario.instruction, fixtureId);
    const id = created.case.case_id;
    for (const step of [
      "requirements/confirm",
      "pack/generate",
      "pack/approve",
    ]) {
      this.step(id, step);
    }
    this.loadFixtureArtefacts(id, fixtureId);
    const stored = this.get(id);
    for (const step of [
      "artefacts/classify",
      "configuration/draft",
      "configuration/approve",
      "run",
      "plan",
      "readiness/approve",
    ]) {
      if (stored.doc.state === S.BLOCKED || stored.doc.state === S.EXCEPTIONS_REQUIRE_INPUT) break;
      this.step(id, step);
    }
    return this.status(id);
  }

  pack(caseId: string): AgentPack {
    const stored = this.get(caseId);
    return {
      pack: stored.doc.onboarding_pack as AgentPack["pack"],
      required_artefacts: stored.doc.required_artefacts,
      issued_at: stored.doc.pack_issued_synthetically_at,
    };
  }

  readinessPackage(caseId: string): AgentReadinessPackage {
    const stored = this.get(caseId);
    const readiness = this.readiness(stored);
    return {
      readiness,
      package: readiness.ready
        ? {
            case_summary: { case_id: stored.doc.case_id, status: S.READY_FOR_EXECUTION },
            execution_manifest: {
              runtime_mode: "synthetic",
              execution_performed: false,
              published: false,
              live_writes: [],
              emails_sent: [],
              content_hash: "sha-mock",
            },
            statement: {
              headline: "Synthetic case ready for execution.",
              not_done: [
                "No live files were written.",
                "No production pipeline was triggered.",
                "No external email was sent.",
                "Nothing was published.",
              ],
            },
          }
        : null,
    };
  }

  audit(caseId: string): AgentAudit {
    return { events: this.get(caseId).audit, chain_intact: true };
  }

  // -- internals ---------------------------------------------------------- //

  private applyStep(stored: StoredCase, step: string): void {
    switch (step) {
      case "requirements/confirm":
        stored.doc.confirmed_requirements = { ...stored.doc.extracted_requirements };
        stored.doc.approval_history.push({ subject: "requirements", decision: "approved" });
        this.move(stored, S.REQUIREMENTS_CONFIRMED);
        this.record(stored, "requirements_confirmed", "the operator confirmed the interpretation");
        break;
      case "pack/generate":
        stored.doc.onboarding_pack = MOCK_PACK;
        stored.doc.required_artefacts = [
          { role: "loan_extract", label: "Primary loan tape", required: true },
          { role: "cashflow_extract", label: "Cash-flow tape", required: false },
        ];
        this.move(stored, S.ONBOARDING_PACK_GENERATED);
        this.move(stored, S.ONBOARDING_PACK_APPROVAL_REQUIRED);
        this.record(stored, "onboarding_pack_generated", "built from the onboarding configuration");
        break;
      case "pack/approve":
        stored.doc.approval_history.push({ subject: "onboarding_pack", decision: "approved" });
        stored.doc.pack_issued_synthetically_at = nowIso();
        this.move(stored, S.ONBOARDING_PACK_ISSUED_SYNTHETICALLY);
        this.move(stored, S.AWAITING_CLIENT_INPUT);
        this.record(
          stored,
          "onboarding_pack_issued_synthetically",
          "recorded as issued; no email was sent",
        );
        break;
      case "artefacts/classify": {
        this.move(stored, S.ARTEFACTS_CLASSIFIED);
        this.record(stored, "artefacts_classified", "apps.blob_trigger_app.file_roles");
        const hasLoanTape = stored.doc.received_artefacts.some(
          (a) => a.artefact_type === "loan_extract",
        );
        if (!hasLoanTape) {
          this.block(stored, ["Trakt still needs the Primary loan tape."]);
        }
        break;
      }
      case "configuration/draft":
        stored.doc.proposed_configuration = MOCK_CONFIG;
        stored.doc.configuration_provenance = MOCK_PROVENANCE;
        this.move(stored, S.CONFIG_DRAFTED);
        this.record(stored, "configuration_drafted", "resolved through the configuration hierarchy");
        if (stored.scenario === "scenario_d_product_config_gap") {
          this.block(stored, ["The reporting entity's LEI has not been configured."]);
        } else {
          this.move(stored, S.CONFIG_APPROVAL_REQUIRED);
        }
        break;
      case "configuration/approve":
        stored.doc.confirmed_configuration = { client: { client_id: stored.doc.client_id } };
        stored.doc.configuration_provenance = stored.doc.configuration_provenance.map((p) => ({
          ...p,
          confirmed: true,
          requires_human_confirmation: false,
        }));
        stored.doc.approval_history.push({ subject: "client_configuration", decision: "approved" });
        this.move(stored, S.CONFIG_APPROVED);
        this.record(stored, "configuration_approved", "the operator approved the configuration");
        break;
      case "run":
        this.move(stored, S.SYNTHETIC_ONBOARDING_RUNNING);
        stored.doc.stage_outcomes = {
          onboard: "deterministic_execution_completed",
          transform: "deterministic_execution_completed",
          validate: "deterministic_execution_completed",
          stamp: "deterministic_execution_completed",
          assemble: "deterministic_execution_completed",
        };
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
      case "plan":
        stored.doc.orchestration_plan = MOCK_PLAN;
        stored.doc.assembler_plan = { satisfied: true, summary: "Assembler prerequisites are satisfied." };
        this.move(stored, S.ORCHESTRATION_PLAN_GENERATED);
        this.move(stored, S.EXECUTION_APPROVAL_REQUIRED);
        this.record(stored, "orchestration_plan_generated", "nothing was executed");
        break;
      case "readiness/approve": {
        stored.doc.approval_history.push({ subject: "execution_readiness", decision: "approved" });
        const verdict = this.readiness(stored);
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
        this.record(stored, "case_cancelled", "cancelled by the operator");
        break;
      default:
        throw new OpsError("That is not something Trakt can do.", "OCC_AGENT_UNKNOWN_STEP");
    }
  }

  private readiness(stored: StoredCase): AgentStatus["readiness"] {
    const doc = stored.doc;
    const criteria = [
      {
        key: "facts_confirmed",
        label: "Required facts confirmed",
        passed: Object.keys(doc.confirmed_requirements).length > 0,
        detail: "The interpretation has been confirmed.",
        remedy: "Confirm the interpretation.",
      },
      {
        key: "artefacts_present",
        label: "Required source files received",
        passed: doc.received_artefacts.some((a) => a.artefact_type === "loan_extract"),
        detail: "All required files were provided.",
        remedy: "Provide the missing file.",
      },
      {
        key: "configuration_validates",
        label: "Configuration validates",
        passed: Object.keys(doc.confirmed_configuration).length > 0,
        detail: "Configuration resolved.",
        remedy: "Approve the configuration.",
      },
      {
        key: "exceptions_cleared",
        label: "Blocking exceptions cleared",
        passed: !doc.open_decisions.some((d) => d.blocking && d.status === "open"),
        detail: "No blocking exceptions remain.",
        remedy: "Resolve each blocking decision.",
      },
      {
        key: "orchestration_valid",
        label: "Orchestration sequencing valid",
        passed: Object.keys(doc.orchestration_plan).length > 0,
        detail: "The execution plan is prepared.",
        remedy: "Generate the orchestration plan.",
      },
      {
        key: "approvals_complete",
        label: "Required approvals complete",
        passed: ["requirements", "onboarding_pack", "client_configuration", "execution_readiness"].every(
          (subject) => doc.approval_history.some((a) => a.subject === subject),
        ),
        detail: "All required approvals are recorded.",
        remedy: "Give the outstanding approvals.",
      },
      {
        key: "runtime_controls_intact",
        label: "Synthetic controls intact",
        passed: true,
        detail: "The synthetic boundary held for the whole case.",
        remedy: "",
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

  private answer(stored: StoredCase, lower: string): string {
    if (/(left|remain|outstanding|still)/.test(lower)) {
      const verdict = this.readiness(stored);
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
    return this.statusSentence(stored);
  }

  private statusSentence(stored: StoredCase): string {
    const parts = [`This case is at ${STATE_LABELS[stored.doc.state]}.`];
    if (stored.doc.blockers.length > 0) {
      parts.push(`In the way: ${stored.doc.blockers.slice(0, 3).join("; ")}`);
    }
    const blocking = stored.doc.open_decisions.filter((d) => d.blocking && d.status === "open");
    if (blocking.length > 0) {
      parts.push(`${blocking.length} decision${blocking.length === 1 ? "" : "s"} need you.`);
    }
    return parts.join(" ");
  }

  private stepForInstruction(lower: string, stored: StoredCase): string | null {
    if (/\bcancel\b/.test(lower)) return "cancel";
    if (/\breadiness\b|\bready for execution\b/.test(lower)) return "readiness/approve";
    if (/\b(approve|accept)\b.*\b(config)/.test(lower)) return "configuration/approve";
    if (/\b(approve|accept)\b.*\b(pack|questionnaire|email)\b/.test(lower)) return "pack/approve";
    if (/\b(generate|draft|regenerate)\b.*\b(pack|questionnaire|email)\b/.test(lower))
      return "pack/generate";
    if (/\b(generate|draft)\b.*\bconfig/.test(lower)) return "configuration/draft";
    if (/\b(generate|prepare)\b.*\b(plan|orchestration)\b/.test(lower)) return "plan";
    if (/\b(run|start)\b.*\b(onboarding|controls)\b/.test(lower)) return "run";
    if (/\bclassify\b/.test(lower)) return "artefacts/classify";
    if (/\bconfirm\b.*\b(interpretation|requirements|facts)\b/.test(lower))
      return "requirements/confirm";
    // A bare "yes" means whatever the case is currently waiting on.
    if (/^(yes|confirm|confirmed|approve|approved|go ahead|proceed)\.?$/.test(lower)) {
      return PENDING_CONFIRMATION[stored.doc.state] ?? null;
    }
    return null;
  }

  private requireAction(stored: StoredCase, action: string): void {
    if (!(ALLOWED[stored.doc.state] ?? []).includes(action)) {
      throw new OpsError(
        `'${action.replace(/_/g, " ")}' is not something you can do while this case is at ${
          STATE_LABELS[stored.doc.state]
        }.`,
        "OCC_AGENT_ACTION_NOT_ALLOWED",
      );
    }
  }

  private move(stored: StoredCase, state: string): void {
    stored.doc.state = state;
    stored.doc.case_version += 1;
    stored.doc.updated_at = nowIso();
    stored.reached.add(state);
  }

  private block(stored: StoredCase, blockers: string[]): void {
    stored.doc.blockers = blockers.filter(Boolean);
    this.move(stored, S.BLOCKED);
    this.record(stored, "case_blocked", "a deterministic control blocked the case");
  }

  private record(stored: StoredCase, action: string, basis: string): void {
    stored.audit.push({
      event_id: `cae-${stored.audit.length + 1}`,
      case_id: stored.doc.case_id,
      at: nowIso(),
      action,
      decision_basis: basis,
      runtime_mode: "synthetic",
      resulting_state: stored.doc.state,
    });
  }

  private summary(stored: StoredCase): CaseSummary {
    const doc = stored.doc;
    return {
      case_id: doc.case_id,
      tenant: doc.tenant,
      client_id: doc.client_id,
      client_name: doc.client_name,
      portfolio_id: doc.portfolio_id,
      portfolio_name: doc.portfolio_name,
      asset_type: doc.asset_type,
      state: doc.state,
      state_label: STATE_LABELS[doc.state],
      readiness_status: doc.readiness_status,
      runtime_mode: "synthetic",
      synthetic: true,
      open_decisions: doc.open_decisions.filter((d) => d.blocking && d.status === "open").length,
      fixture_id: doc.fixture_id,
      created_at: doc.created_at,
      updated_at: doc.updated_at,
    };
  }

  private scenarioFor(instruction: string): string {
    const lower = instruction.toLowerCase();
    const hit = SCENARIOS.find((s) =>
      lower.includes(s.instruction.split(".")[0].replace("Onboard ", "").toLowerCase()),
    );
    return hit?.fixture_id ?? "scenario_a_clean";
  }

  private blankDoc(id: string, instruction: string, scenario: string): SyntheticCaseDoc {
    const clientName = /onboard\s+([A-Z][\w ]+?)[.,]/i.exec(instruction)?.[1]?.trim() ?? "New client";
    const portfolioId = /portfolio id\s+([\w_.-]+)/i.exec(instruction)?.[1] ?? "direct_001";
    const extracted = {
      client_name: clientName,
      proposed_client_id: clientName.toLowerCase().replace(/\W+/g, "_"),
      asset_type: "equity_release",
      jurisdiction: "GB",
      required_products: ["portfolio_mi"],
      reporting_frequency: "monthly",
      reporting_date: "2026-06-30",
      expected_artefacts: ["loan_extract"],
      proposed_portfolio_id: portfolioId,
      unresolved_questions: [] as string[],
    };
    return {
      case_id: id,
      tenant: "Alpine Capital",
      initiating_user: "Operator",
      state: S.REQUIREMENTS_EXTRACTED,
      runtime_mode: "synthetic",
      case_version: 1,
      synthetic: true,
      client_id: extracted.proposed_client_id,
      client_name: clientName,
      portfolio_id: portfolioId,
      portfolio_name: "",
      asset_type: "equity_release",
      extracted_requirements: extracted,
      confirmed_requirements: {},
      unresolved_questions: [],
      onboarding_pack: {},
      pack_issued_synthetically_at: "",
      required_artefacts: [],
      received_artefacts: [],
      proposed_configuration: {},
      confirmed_configuration: {},
      configuration_provenance: [],
      mapping_decisions: [],
      open_decisions: [],
      control_results: [],
      stage_outcomes: {},
      blockers: [],
      observations: [],
      planned_pipeline_actions: [],
      orchestration_plan: {},
      assembler_plan: {},
      readiness_status: "not_evaluated",
      readiness_package_ref: "",
      approval_history: [],
      messages: [
        { role: "operator", text: instruction, at: nowIso(), refs: [] },
        {
          role: "agent",
          text: [
            `Client: ${clientName}`,
            "Asset type: Equity Release",
            "Products:",
            "- Portfolio MI",
            "Frequency: Monthly",
            "First reporting date: 2026-06-30",
          ].join("\n"),
          at: nowIso(),
          refs: [],
        },
      ],
      fixture_id: scenario,
      created_at: nowIso(),
      updated_at: nowIso(),
    };
  }
}

const STEP_ACTIONS: Record<string, string> = {
  "requirements/confirm": "confirm_requirements",
  "pack/generate": "generate_onboarding_pack",
  "pack/approve": "approve_onboarding_pack",
  "artefacts/classify": "classify_artefacts",
  "configuration/draft": "draft_client_config",
  "configuration/approve": "approve_client_config",
  run: "run_synthetic_onboarding",
  plan: "generate_orchestration_plan",
  "readiness/approve": "approve_execution_readiness",
  cancel: "cancel_case",
};

const MATERIAL_STEPS = new Set([
  "requirements/confirm",
  "pack/approve",
  "configuration/approve",
  "readiness/approve",
  "cancel",
]);

const PROPOSAL_SUMMARIES: Record<string, string> = {
  "requirements/confirm": "Confirm the proposed interpretation.",
  "pack/approve": "Approve the onboarding pack.",
  "configuration/approve": "Approve the proposed configuration.",
  "readiness/approve": "Approve readiness for execution.",
  cancel: "Cancel this synthetic case.",
};

/** What a bare "yes" means, given where the case is. */
const PENDING_CONFIRMATION: Record<string, string> = {
  [S.REQUIREMENTS_EXTRACTED]: "requirements/confirm",
  [S.REQUIREMENTS_CONFIRMATION_REQUIRED]: "requirements/confirm",
  [S.ONBOARDING_PACK_GENERATED]: "pack/approve",
  [S.ONBOARDING_PACK_APPROVAL_REQUIRED]: "pack/approve",
  [S.CONFIG_DRAFTED]: "configuration/approve",
  [S.CONFIG_APPROVAL_REQUIRED]: "configuration/approve",
  [S.ORCHESTRATION_PLAN_GENERATED]: "readiness/approve",
  [S.EXECUTION_APPROVAL_REQUIRED]: "readiness/approve",
};

const MOCK_PACK = {
  content_hash: "mock-pack-hash",
  outcome: "mi",
  required_roles: ["loan_extract"],
  optional_roles: ["cashflow_extract"],
  sections: [
    {
      key: "client_email",
      title: "Draft client email",
      body: "Dear team,\n\nWe are setting up your portfolio on Trakt.\n\nKind regards,\nTrakt Operations",
      items: [],
      depends_on: ["client_name"],
    },
    {
      key: "questionnaire",
      title: "Onboarding questionnaire",
      body: "",
      items: [
        { id: "q_client_legal_name", question: "What is the client's full legal name?", why: "" },
      ],
      depends_on: ["client_name"],
    },
  ],
};

const MOCK_CONFIG = {
  status: "READY",
  content_hash: "mock-config-hash",
  candidate: { client: { client_id: "mock_client" }, portfolio: { asset_class: "equity_release" } },
  blockers: [],
  warnings: [],
};

const MOCK_PROVENANCE = [
  {
    key: "portfolio.base_currency",
    value: "GBP",
    source: "deterministic_default",
    evidence: "standard currency for GB",
    confidence: null,
    downstream_impact: "All amounts are validated and presented in it.",
    material: true,
    requires_human_confirmation: true,
    confirmed: false,
    confirmed_by: "",
  },
];

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

function nowIso(): string {
  return new Date().toISOString();
}
