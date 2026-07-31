import { MockAgent } from "./MockAgent";
import { MockConfigAdmin } from "./MockConfigAdmin";
import { MockOnboarding } from "./MockOnboarding";
import { OpsError, type OpsClient } from "./OpsClient";
import type {
  AgentAudit,
  AgentMeta,
  AgentPack,
  AgentReadinessPackage,
  AgentStatus,
  AgentTurn,
  CaseSummary,
} from "./agentTypes";
import type {
  AuditTrail,
  Comparison,
  ConfigCatalogue,
  ConfigLayer,
  ConfigOverview,
  ConfigVersion,
  CreateDraftInput,
  ImpactAnalysis,
  Principal,
  ValidationResult,
  ValidationSummary,
} from "./adminTypes";
import type {
  ActivationResult,
  CasePreview,
  ChecklistRow,
  ConfigurationVersionRow,
  OnboardingCase,
  OnboardingClientDetail,
  OnboardingHome,
  OnboardingReference,
} from "./onboardingTypes";
import type {
  Batch,
  BatchInputRole,
  CreateBatchInput,
  Dashboard,
  DecisionInput,
  DecisionResult,
  Delivery,
  Publication,
  RegisterDeliveryInput,
  Review,
  Rule,
  RulesQuery,
  StartWorkflowInput,
  StepDecision,
  Workflow,
  WorkflowRow,
  WorkflowStep,
  WorkflowType,
} from "./types";

/** Plain labels for the delivery workflow step states. */
const STEP_STATUS_LABELS: Record<WorkflowStep["status"], string> = {
  complete: "Complete",
  current: "Current",
  pending: "Pending",
  blocked: "Blocked",
  not_applicable: "Not applicable",
};

const TYPE_LABELS: Record<WorkflowType, string> = {
  new_client: "New client onboarding",
  new_portfolio: "New portfolio onboarding",
  recurring: "Recurring reporting",
  backfill: "Historical backfill",
};

const OUTCOME_LABELS = {
  mi: "MI Reporting",
  mi_annex2: "MI Reporting + ESMA Annex 2 delivery",
};

const SCOPE_EXPLANATIONS: Record<string, string> = {
  file: "Trakt will use this answer for this set of files only, and ask again next time.",
  portfolio: "Trakt will remember this answer whenever it prepares reports for this portfolio.",
  client: "Trakt will remember this answer for everything it prepares for this client.",
  global: "Trakt will remember this answer for every client, everywhere.",
};

/** Expected inputs for an onboarding batch, in the order the mock assigns them. */
const BATCH_ROLE_SPECS: { role: string; label: string; required: boolean }[] = [
  { role: "holdings", label: "Holdings statement", required: true },
  { role: "loan_tape", label: "Primary loan tape", required: true },
  { role: "prior_period", label: "Prior period comparison", required: false },
];

/** `2026-06-30` -> `June 2026`, mirroring the backend's own wording. */
function periodWords(period: string): string {
  const [year, month] = (period ?? "").split("-");
  const names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
  ];
  const index = Number(month) - 1;
  return names[index] ? `${names[index]} ${year}` : period;
}

function deepCopy<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

/**
 * Canned, stateful data so the app runs with no backend
 * (`VITE_OPS_MODE=mock npm run dev`).
 */
export class MockOpsClient implements OpsClient {
  private readonly delayMs: number;
  private readonly principal: Principal;
  private readonly config = new MockConfigAdmin();
  private readonly onboarding = new MockOnboarding();
  private readonly agent = new MockAgent();
  private nextId = 6000;
  /** Counts reads of a running workflow so it visibly "finishes" while polling. */
  private runningTicks = new Map<string, number>();

  /** Counts reads of a running batch so it visibly "finishes" while polling. */
  private batchTicks = new Map<string, number>();

  private clients = ["Alpine Capital", "Birchwood Partners", "Cedar Rock Advisors"];

  private deliveries: Delivery[] = [
    {
      delivery_id: "del-3001",
      client_id: "Alpine Capital",
      portfolio_id: "European Growth",
      reporting_period: "2026-06-30",
      dataset: "Holdings and transactions",
      frequency: "Monthly",
      files: ["Holdings June 2026.xlsx", "Transactions June 2026.xlsx"],
      classification: "recurring",
      classification_label: "Recurring reporting",
      classification_sentence:
        "This looks like the regular monthly reporting for Alpine Capital — European Growth, covering June 2026.",
      registered_at: "2026-07-24T09:12:00Z",
    },
    {
      delivery_id: "del-3002",
      client_id: "Alpine Capital",
      portfolio_id: "Nordic Bond",
      reporting_period: "2026-06-30",
      dataset: "Holdings",
      frequency: "Monthly",
      files: ["Nordic Bond June 2026.xlsx"],
      classification: "new_portfolio",
      classification_label: "New portfolio onboarding",
      classification_sentence:
        "This is the first time Trakt has seen the Nordic Bond portfolio for Alpine Capital, so it will treat this as onboarding a new portfolio.",
      registered_at: "2026-07-25T14:40:00Z",
    },
  ];

  private batches: Batch[] = [
    {
      batch_id: "batch-7001",
      client_id: "Alpine Capital",
      portfolio_id: "Nordic Bond",
      reporting_date: "2026-06-30",
      workflow_type: "mi",
      dataset: "funded",
      dataset_label: "Funded book",
      status: "review_required",
      status_label: "Needs your review",
      status_sentence:
        "One of the files could be more than one thing. Confirm what it is so Trakt can continue.",
      auto_start_when_ready: true,
      files: [
        {
          source_file_id: "sf-101",
          filename: "nordic-bond-loan-tape-june.xlsx",
          role: "loan_tape",
          role_label: "Primary loan tape",
          status: "recognised",
          status_sentence: "Received and recognised as the primary loan tape.",
          confidence: 0.97,
        },
        {
          source_file_id: "sf-102",
          filename: "nordic-extract-06.csv",
          role: "",
          role_label: "Not yet decided",
          status: "ambiguous",
          status_sentence:
            "This could be either the holdings statement or a transactions extract. Trakt needs your confirmation before it can continue.",
          confidence: 0.52,
        },
      ],
      input_roles: [
        { role: "holdings", label: "Holdings statement", required: true, satisfied: false },
        { role: "loan_tape", label: "Primary loan tape", required: true, satisfied: true },
        { role: "prior_period", label: "Prior period comparison", required: false, satisfied: false },
      ],
      missing_roles: ["holdings"],
      configuration_ready: true,
      blocking_decisions: ["dec-9101"],
      source_prefix: "",
      workflow_id: null,
      created_at: "2026-07-27T08:20:00Z",
      updated_at: "2026-07-27T08:24:00Z",
    },
    {
      // The pack behind wf-1002, so its workflow can show the files it ran on.
      batch_id: "batch-7002",
      client_id: "Birchwood Partners",
      portfolio_id: "Global Credit",
      reporting_date: "2026-06-30",
      workflow_type: "mi",
      dataset: "funded",
      dataset_label: "Funded book",
      status: "completed",
      status_label: "Done",
      status_sentence: "Trakt has finished with this input pack.",
      auto_start_when_ready: true,
      files: [
        {
          source_file_id: "sf-201",
          filename: "global-credit-loan-tape-june.xlsx",
          role: "loan_tape",
          role_label: "Primary loan tape",
          status: "recognised",
          status_sentence: "Received and recognised as the primary loan tape.",
          confidence: 0.98,
        },
        {
          source_file_id: "sf-202",
          filename: "global-credit-holdings-june.xlsx",
          role: "holdings",
          role_label: "Holdings statement",
          status: "recognised",
          status_sentence: "Received and recognised as the holdings statement.",
          confidence: 0.96,
        },
      ],
      input_roles: [
        { role: "holdings", label: "Holdings statement", required: true, satisfied: true },
        { role: "loan_tape", label: "Primary loan tape", required: true, satisfied: true },
        { role: "prior_period", label: "Prior period comparison", required: false, satisfied: false },
      ],
      missing_roles: [],
      configuration_ready: true,
      blocking_decisions: [],
      source_prefix: "Birchwood Partners/direct/funded/monthly/Global Credit/2026-06-30",
      workflow_id: "wf-1002",
      created_at: "2026-07-26T07:10:00Z",
      updated_at: "2026-07-26T07:40:00Z",
    },
  ];

  private workflows: Workflow[] = [
    {
      workflow_id: "wf-1001",
      client_id: "Alpine Capital",
      portfolio_id: "European Growth",
      reporting_period: "2026-06-30",
      outcome: "mi",
      workflow_type: "recurring",
      workflow_type_label: "Recurring reporting",
      status: "needs_review",
      open_decisions: 2,
      created_at: "2026-07-24T09:15:00Z",
      updated_at: "2026-07-24T09:32:00Z",
      outcome_label: OUTCOME_LABELS.mi,
      status_sentence:
        "Trakt has prepared most of this report, but two questions need your answer before it can continue.",
      interrupted: false,
      blockers: [],
      rerun_count: 0,
      stages: [
        {
          stage: "receive",
          label: "Files received",
          status: "completed",
          status_label: "Done",
          summary: "Two files arrived for June 2026 and both were readable.",
        },
        {
          stage: "understand",
          label: "Understanding the files",
          status: "completed",
          status_label: "Done",
          summary:
            "Trakt recognised both files and matched almost everything to what it already knows about this portfolio.",
          why_it_matters:
            "Getting this right means every figure lands in the right place in the report.",
          evidence: [
            {
              label: "How the columns were matched",
              kind: "table",
              data: {
                columns: ["The file says", "Trakt read it as"],
                rows: [
                  ["Security Name", "Holding name"],
                  ["Qty", "Quantity held"],
                  ["Book Cost", "Purchase cost"],
                  ["Ccy", "Currency"],
                ],
              },
            },
          ],
        },
        {
          stage: "check",
          label: "Checking the numbers",
          status: "needs_review",
          status_label: "Needs review",
          summary: "Two names in the files are new to Trakt and need your confirmation.",
          why_it_matters:
            "Until these are confirmed, those holdings could be placed in the wrong section of the report.",
          warnings: ["Two new names need your confirmation before the report can continue."],
          decision_count: 2,
          evidence: [
            {
              label: "What Trakt checked",
              kind: "list",
              data: [
                "Every holding adds up to the portfolio total",
                "All values are in the expected currency",
                "No dates fall outside June 2026",
              ],
            },
          ],
        },
        {
          stage: "prepare",
          label: "Preparing the report",
          status: "waiting",
          status_label: "Waiting",
        },
        {
          stage: "publish",
          label: "Publishing",
          status: "waiting",
          status_label: "Waiting",
        },
      ],
    },
    {
      workflow_id: "wf-1002",
      client_id: "Birchwood Partners",
      portfolio_id: "Global Credit",
      reporting_period: "2026-06-30",
      outcome: "mi_annex2",
      workflow_type: "recurring",
      workflow_type_label: "Recurring reporting",
      status: "awaiting_publication",
      open_decisions: 0,
      created_at: "2026-07-22T08:05:00Z",
      updated_at: "2026-07-23T10:20:00Z",
      outcome_label: OUTCOME_LABELS.mi_annex2,
      status_sentence: "This report is ready. Look it over and publish when you're happy.",
      interrupted: false,
      blockers: [],
      rerun_count: 0,
      stages: [
        {
          stage: "receive",
          label: "Files received",
          status: "completed",
          status_label: "Done",
          summary: "Three files arrived for June 2026.",
        },
        {
          stage: "understand",
          label: "Understanding the files",
          status: "completed",
          status_label: "Done",
          summary: "Everything matched what Trakt already knows about this portfolio.",
        },
        {
          stage: "check",
          label: "Checking the numbers",
          status: "approved",
          status_label: "Approved",
          summary: "All checks passed. One earlier question was answered by your team.",
        },
        {
          stage: "prepare",
          label: "Preparing the report",
          status: "completed",
          status_label: "Done",
          summary: "The report and the regulatory annex have been prepared.",
        },
        {
          stage: "publish",
          label: "Publishing",
          status: "ready",
          status_label: "Ready",
          summary: "Waiting for your approval to publish this as the official version.",
        },
      ],
    },
    {
      workflow_id: "wf-1003",
      client_id: "Cedar Rock Advisors",
      portfolio_id: "Real Assets",
      reporting_period: "2026-06-30",
      outcome: "mi",
      workflow_type: "recurring",
      workflow_type_label: "Recurring reporting",
      status: "published",
      open_decisions: 0,
      created_at: "2026-07-18T09:00:00Z",
      updated_at: "2026-07-21T16:45:00Z",
      outcome_label: OUTCOME_LABELS.mi,
      status_sentence: "This report was published on 21 July 2026.",
      interrupted: false,
      blockers: [],
      rerun_count: 0,
      stages: [
        {
          stage: "receive",
          label: "Files received",
          status: "completed",
          status_label: "Done",
        },
        {
          stage: "understand",
          label: "Understanding the files",
          status: "completed",
          status_label: "Done",
        },
        {
          stage: "check",
          label: "Checking the numbers",
          status: "completed",
          status_label: "Done",
        },
        {
          stage: "prepare",
          label: "Preparing the report",
          status: "completed",
          status_label: "Done",
        },
        {
          stage: "publish",
          label: "Publishing",
          status: "completed",
          status_label: "Done",
          summary: "Published as version 2 on 21 July 2026.",
        },
      ],
    },
    {
      workflow_id: "wf-1004",
      client_id: "Alpine Capital",
      portfolio_id: "Nordic Bond",
      reporting_period: "2026-06-30",
      outcome: "mi",
      workflow_type: "new_portfolio",
      workflow_type_label: "New portfolio onboarding",
      status: "blocked",
      open_decisions: 0,
      created_at: "2026-07-25T14:45:00Z",
      updated_at: "2026-07-25T14:50:00Z",
      outcome_label: OUTCOME_LABELS.mi,
      status_sentence:
        "Trakt could not open one of the files. It may still be uploading — try running this again.",
      interrupted: false,
      blockers: ["One of the files could not be opened."],
      rerun_count: 1,
      stages: [
        {
          stage: "receive",
          label: "Files received",
          status: "blocked",
          status_label: "Blocked",
          summary: "One file arrived but could not be opened.",
          blockers: ["The file may be incomplete. Ask the sender to save it again, then run again."],
        },
        {
          stage: "understand",
          label: "Understanding the files",
          status: "waiting",
          status_label: "Waiting",
        },
        {
          stage: "check",
          label: "Checking the numbers",
          status: "waiting",
          status_label: "Waiting",
        },
        {
          stage: "prepare",
          label: "Preparing the report",
          status: "waiting",
          status_label: "Waiting",
        },
        {
          stage: "publish",
          label: "Publishing",
          status: "waiting",
          status_label: "Waiting",
        },
      ],
    },
  ];

  private reviews: Review[] = [
    {
      decision_id: "rev-2001",
      workflow_id: "wf-1001",
      client_id: "Alpine Capital",
      stage: "check",
      kind: "mapping",
      title: "A new column in the holdings file",
      question:
        "The holdings file has a column called “Mkt Val (Loc)” that Trakt hasn't seen before. What should it be treated as?",
      blocking: true,
      status: "open",
      recommendation: {
        value: "market_value_local",
        label: "Market value in local currency",
        confidence: 0.92,
      },
      options: [
        { value: "market_value_local", label: "Market value in local currency" },
        { value: "market_value_reporting", label: "Market value in reporting currency" },
        { value: "purchase_cost", label: "Purchase cost" },
      ],
      allowed_scopes: ["file", "portfolio", "client", "global"],
      default_scope: "portfolio",
      scope_explanations: SCOPE_EXPLANATIONS,
      created_at: "2026-07-24T09:30:00Z",
    },
    {
      decision_id: "rev-2002",
      workflow_id: "wf-1001",
      client_id: "Alpine Capital",
      stage: "check",
      kind: "mapping",
      title: "A holding type Trakt doesn't recognise",
      question:
        "Some holdings are described as “Perp FRN”. What kind of holding is this?",
      blocking: true,
      status: "open",
      recommendation: {
        value: "floating_rate_bond",
        label: "A bond with a variable interest rate",
        confidence: 0.81,
      },
      options: [
        { value: "floating_rate_bond", label: "A bond with a variable interest rate" },
        { value: "fixed_rate_bond", label: "A bond with a fixed interest rate" },
        { value: "equity", label: "A share in a company" },
      ],
      allowed_scopes: ["file", "portfolio", "client", "global"],
      default_scope: "client",
      scope_explanations: SCOPE_EXPLANATIONS,
      created_at: "2026-07-24T09:31:00Z",
    },
  ];

  private rules: Rule[] = [
    {
      rule_id: "rule-4001",
      version: 3,
      kind: "mapping",
      scope: "portfolio",
      client_id: "Alpine Capital",
      portfolio_id: "European Growth",
      status: "approved",
      source_term: "Bk Cost",
      approved_meaning: "Purchase cost",
      description:
        "The Alpine holdings files shorten “purchase cost” to “Bk Cost”. Confirmed by the operations team.",
      approved_by: "J. Whitfield",
      approved_at: "2026-06-12T10:00:00Z",
    },
    {
      rule_id: "rule-4002",
      version: 1,
      kind: "mapping",
      scope: "client",
      client_id: "Birchwood Partners",
      portfolio_id: "",
      status: "approved",
      source_term: "Cpn",
      approved_meaning: "Interest rate paid by a bond",
      description: "Birchwood files use “Cpn” for the interest a bond pays.",
      approved_by: "M. Ellison",
      approved_at: "2026-05-30T15:20:00Z",
    },
    {
      rule_id: "rule-4003",
      version: 2,
      kind: "format",
      scope: "global",
      client_id: "",
      portfolio_id: "",
      status: "approved",
      source_term: "31.12.2025",
      approved_meaning: "Dates written with the day first",
      description:
        "When a file writes dates with dots, read them with the day first. Applies everywhere.",
      approved_by: "J. Whitfield",
      approved_at: "2026-04-08T11:05:00Z",
    },
    {
      rule_id: "rule-4004",
      version: 1,
      kind: "mapping",
      scope: "client",
      client_id: "Cedar Rock Advisors",
      portfolio_id: "",
      status: "approved",
      source_term: "NAV/Sh",
      approved_meaning: "Value of one share of the fund",
      description: "Cedar Rock statements shorten this in their monthly files.",
      approved_by: "M. Ellison",
      approved_at: "2026-07-01T09:45:00Z",
    },
  ];

  private publications: Publication[] = [
    {
      publication_id: "pub-5001",
      client_id: "Cedar Rock Advisors",
      workflow_id: "wf-1003",
      workflow_type_label: "Recurring reporting",
      reporting_period: "2026-06-30",
      status: "published",
      version: 2,
      rule_version_count: 14,
      previous_publication_id: "pub-4900",
      approved_by: "M. Ellison",
      published_at: "2026-07-21T16:45:00Z",
      prepared_at: "2026-07-21T15:30:00Z",
    },
    {
      publication_id: "pub-4900",
      client_id: "Cedar Rock Advisors",
      workflow_id: "wf-0903",
      workflow_type_label: "Recurring reporting",
      reporting_period: "2026-05-31",
      status: "published",
      version: 1,
      rule_version_count: 12,
      previous_publication_id: null,
      approved_by: "M. Ellison",
      published_at: "2026-06-19T14:10:00Z",
      prepared_at: "2026-06-19T13:00:00Z",
    },
  ];

  constructor(delayMs = 120, role: "admin" | "operator" = "admin") {
    this.delayMs = delayMs;
    this.principal = {
      name: role === "admin" ? "Administrator" : "Operator",
      role,
      is_admin: role === "admin",
      all_clients: true,
      clients: [],
    };
  }

  /** Every administrator call is refused for a non-administrator, exactly as
   *  the backend refuses it — the browser is never the security boundary. */
  private requireAdmin(): void {
    if (!this.principal.is_admin) {
      throw new OpsError("This area is for administrators.", "OPS_ADMIN_REQUIRED");
    }
  }

  private async wait(): Promise<void> {
    if (this.delayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, this.delayMs));
    }
  }

  private id(prefix: string): string {
    this.nextId += 1;
    return `${prefix}-${this.nextId}`;
  }

  /** How far each published approval was recorded as reaching. */
  private readonly approvalScopes = new Map<string, string>();

  /**
   * The delivery case file, in the same shape the backend serves: completed
   * steps stay visible, the current step is where the operator acts, and the
   * approval decision lives inside the workflow rather than beside it.
   */
  private caseFile(workflow: Workflow): WorkflowStep[] {
    const batch = this.batches.find((b) => b.workflow_id === workflow.workflow_id);
    const decisions = this.reviews.filter((d) => d.workflow_id === workflow.workflow_id);
    const openDecisions = decisions.filter((d) => !d.resolved_at && d.status !== "resolved");
    const resolved = decisions.filter((d) => Boolean(d.resolved_at) || d.status === "resolved");
    const publication = this.publications.find((p) => p.workflow_id === workflow.workflow_id);
    const published = workflow.status === "published";
    const awaiting = workflow.status === "awaiting_publication";
    const blocked = workflow.status === "blocked";

    const toDecision = (d: Review): StepDecision => ({
      decision_id: d.decision_id,
      title: d.title,
      question: d.question,
      blocking: d.blocking,
      status: d.status,
      answer: "",
      scope: "",
      actor: d.resolved_by ?? "",
      at: d.resolved_at ?? "",
      overridden: false,
    });

    const files = (batch?.files ?? []).map((f) => ({
      filename: f.filename,
      size: "24.0 KB",
      received_at: batch?.created_at ?? workflow.created_at,
      received_from: "You",
      checksum: f.source_file_id,
      role_label: f.role_label,
    }));

    const step = (
      key: string,
      label: string,
      status: WorkflowStep["status"],
      summary: string,
      extra: Partial<WorkflowStep> = {},
    ): WorkflowStep => ({
      key,
      label,
      status,
      status_label: STEP_STATUS_LABELS[status],
      summary,
      facts: [],
      warnings: [],
      blockers: [],
      ...extra,
    });

    // On a cancelled delivery, a step it never reached is not "pending" — it is
    // never going to happen. Matches the backend's case-file view.
    const asCaseFile = (steps: WorkflowStep[]): WorkflowStep[] =>
      workflow.status !== "cancelled"
        ? steps
        : steps.map((s) =>
            s.status === "pending"
              ? { ...s, status: "not_applicable", status_label: STEP_STATUS_LABELS.not_applicable }
              : s,
          );

    return asCaseFile([
      step(
        "files_received",
        "Files received",
        files.length > 0 ? "complete" : "pending",
        files.length > 0
          ? `${files.length} files received. The pack was complete.`
          : "No files have been received yet.",
        {
          files,
          facts: [
            { label: "Files received", value: String(files.length) },
            { label: "Source location", value: batch?.source_prefix ?? "" },
          ].filter((f) => f.value),
        },
      ),
      step(
        "delivery_identified",
        "Delivery identified",
        "complete",
        `${workflow.workflow_type_label} for ${workflow.client_id} — ${workflow.portfolio_id}.`,
        {
          facts: [
            { label: "Client", value: workflow.client_id },
            { label: "Portfolio", value: workflow.portfolio_id },
            { label: "Reporting period", value: workflow.reporting_period },
            { label: "Book", value: batch?.dataset_label ?? "Funded book" },
            { label: "Delivery type", value: workflow.workflow_type_label },
          ],
        },
      ),
      step(
        "configuration_selected",
        "Configuration selected",
        "complete",
        "The configuration for this delivery was selected and locked to it.",
        {
          facts: [
            { label: "Asset package", value: "v1" },
            {
              label: "Regime",
              value:
                workflow.outcome === "mi_annex2"
                  ? "ESMA Annex 2"
                  : "Management information only",
            },
          ],
        },
      ),
      step(
        "data_assessed",
        "Data assessed",
        blocked ? "blocked" : published || awaiting ? "complete" : "current",
        "1,284 loans were read and checked against the approved rules.",
        {
          warnings: workflow.stages.flatMap((s) => s.warnings ?? []),
          blockers: workflow.blockers,
          facts: [
            { label: "Loans assessed", value: "1,284" },
            {
              label: "Output eligibility",
              value: blocked ? "Held by a blocking problem" : "Ready for publication",
            },
          ],
        },
      ),
      step(
        "issues_reviewed",
        "Issues reviewed",
        openDecisions.length > 0 ? "current" : resolved.length > 0 ? "complete" : "not_applicable",
        openDecisions.length > 0
          ? `${openDecisions.length} questions still need an answer.`
          : resolved.length > 0
            ? `${resolved.length} questions answered.`
            : "Trakt had no questions about this delivery.",
        {
          resolved: resolved.map(toDecision),
          unresolved: openDecisions.map(toDecision),
        },
      ),
      step(
        "publication_approval",
        "Publication approval",
        published ? "complete" : awaiting ? "current" : "pending",
        published
          ? "Published."
          : awaiting
            ? "The delivery is ready. It will only be published when you approve it."
            : "Nothing is waiting for your approval yet.",
        {
          approval: {
            available: awaiting,
            headline: "Ready to publish",
            evidence_lines: [
              `${files.length} files received`,
              "1,284 loans assessed",
              "0 blocking issues",
              "Asset configuration v1 applied",
              ...(workflow.outcome === "mi_annex2" ? ["ESMA Annex 2 enabled"] : []),
            ],
            question: "Publish this delivery as the latest official version?",
            scope_question: "Should Trakt remember this decision for future deliveries?",
            scope_note:
              "Trakt records your answer so it is on the delivery's record. It does not " +
              "publish anything on its own — every delivery is approved by a person.",
            scopes: [
              {
                value: "delivery",
                label: "No — this delivery only",
                explanation: "Nothing is carried forward. Trakt will ask again next time.",
              },
              {
                value: "portfolio",
                label: "Yes — future deliveries for this portfolio",
                explanation:
                  "Your answer is recorded against this portfolio. Someone still approves " +
                  "each delivery.",
              },
              {
                value: "client",
                label: "Yes — future deliveries for this client",
                explanation:
                  "Your answer is recorded against this client. Someone still approves " +
                  "each delivery.",
              },
            ],
            default_scope: "delivery",
            consequence:
              `This will publish the ${periodWords(workflow.reporting_period)} ` +
              `${workflow.client_id} ${workflow.portfolio_id} delivery as the latest ` +
              "official version.",
            scope_consequences: {
              delivery: "This decision applies only to this delivery.",
              portfolio:
                `Your answer is also recorded against future deliveries for ` +
                `${workflow.portfolio_id}, which are still approved one at a time.`,
              client:
                `Your answer is also recorded against future deliveries for ` +
                `${workflow.client_id}, which are still approved one at a time.`,
            },
            version: publication?.version ?? null,
          },
        },
      ),
      step(
        "published",
        "Published",
        published ? "complete" : "pending",
        published
          ? "Published. This is now the latest official version."
          : "Nothing has been published for this delivery yet.",
        {
          outputs: published ? ["Management information (latest)"] : [],
          facts: published
            ? [
                { label: "Published", value: publication?.published_at ?? "" },
                { label: "Published by", value: publication?.approved_by ?? "" },
                {
                  label: "Decision recorded for",
                  value:
                    this.approvalScopes.get(workflow.workflow_id) === "portfolio"
                      ? "Noted for this portfolio"
                      : this.approvalScopes.get(workflow.workflow_id) === "client"
                        ? "Noted for this client"
                        : "This delivery only",
                },
                { label: "Audit reference", value: publication?.publication_id ?? "" },
              ]
            : [],
        },
      ),
    ]);
  }

  private findWorkflow(workflowId: string): Workflow {
    const workflow = this.workflows.find((w) => w.workflow_id === workflowId);
    if (!workflow) {
      throw new OpsError("That workflow could not be found.");
    }
    return workflow;
  }

  async health(): Promise<void> {
    await this.wait();
  }

  async getDashboard(): Promise<Dashboard> {
    await this.wait();
    const needsAttention = this.workflows.filter(
      (w) => w.status === "needs_review" || w.status === "blocked",
    );
    return deepCopy({
      tiles: {
        new_deliveries: this.deliveries.length,
        needs_attention: this.workflows.filter((w) => w.status === "needs_review").length,
        blocked: this.workflows.filter((w) => w.status === "blocked").length,
        ready_to_publish: this.workflows.filter((w) => w.status === "awaiting_publication").length,
        recently_published: this.publications.filter((p) => p.status === "published").length,
      },
      needs_attention: needsAttention,
      recently_published: this.publications.filter((p) => p.status === "published").slice(0, 5),
    });
  }

  async getClients(): Promise<string[]> {
    await this.wait();
    return [...this.clients];
  }

  async getDeliveries(client?: string): Promise<Delivery[]> {
    await this.wait();
    return deepCopy(client ? this.deliveries.filter((d) => d.client_id === client) : this.deliveries);
  }

  async registerDelivery(input: RegisterDeliveryInput): Promise<Delivery> {
    await this.wait();
    const isKnownClient = this.clients.includes(input.client_id);
    const delivery: Delivery = {
      delivery_id: this.id("del"),
      client_id: input.client_id,
      portfolio_id: input.portfolio_id,
      reporting_period: input.reporting_period,
      dataset: input.dataset ?? "Holdings and transactions",
      frequency: input.frequency ?? "Monthly",
      files: ["Holdings.xlsx", "Transactions.xlsx"],
      classification: isKnownClient ? "new_portfolio" : "new_client",
      classification_label: isKnownClient ? "New portfolio onboarding" : "New client onboarding",
      classification_sentence: isKnownClient
        ? `This is the first time Trakt has seen the ${input.portfolio_id} portfolio for ${input.client_id}, so it will treat this as onboarding a new portfolio.`
        : `${input.client_id} is new to Trakt, so it will treat this as onboarding a new client.`,
      registered_at: new Date().toISOString(),
    };
    this.deliveries.push(delivery);
    if (!isKnownClient) {
      this.clients.push(input.client_id);
    }
    return deepCopy(delivery);
  }

  private findBatch(batchId: string): Batch {
    const batch = this.batches.find((b) => b.batch_id === batchId);
    if (!batch) {
      throw new OpsError("That input pack could not be found.");
    }
    return batch;
  }

  /** Re-derives a not-yet-started batch's status from its files and roles. */
  private refreshBatch(batch: Batch): void {
    const missing = batch.input_roles.filter((r) => r.required && !r.satisfied);
    batch.missing_roles = missing.map((r) => r.role);
    batch.updated_at = new Date().toISOString();

    if (batch.files.some((f) => f.status === "ambiguous")) {
      batch.status = "review_required";
      batch.status_label = "Needs your review";
      batch.status_sentence =
        "One of the files could be more than one thing. Confirm what it is so Trakt can continue.";
    } else if (batch.files.length === 0) {
      batch.status = "receiving";
      batch.status_label = "Receiving files";
      batch.status_sentence = "Trakt is watching for the files to arrive.";
    } else if (missing.length > 0) {
      batch.status = "incomplete";
      batch.status_label = "Waiting for files";
      batch.status_sentence = `Waiting for required input: ${missing
        .map((r) => r.label)
        .join(", ")}`;
    } else {
      batch.configuration_ready = true;
      batch.status = "ready";
      batch.status_label = "Ready to start";
      batch.status_sentence =
        "Everything Trakt needs has arrived. Start the onboarding when you're ready.";
    }
  }

  private satisfyRole(batch: Batch, role: BatchInputRole, filename: string): void {
    role.satisfied = true;
    batch.files.push({
      source_file_id: this.id("sf"),
      filename,
      role: role.role,
      role_label: role.label,
      status: "recognised",
      status_sentence: `Received and recognised as the ${role.label.toLowerCase()}.`,
      confidence: 0.95,
    });
  }

  async createBatch(input: CreateBatchInput): Promise<Batch> {
    await this.wait();
    const now = new Date().toISOString();
    const batch: Batch = {
      batch_id: this.id("batch"),
      client_id: input.client_id,
      portfolio_id: input.portfolio_id,
      reporting_date: input.reporting_date,
      workflow_type: input.workflow_type,
      dataset: input.dataset,
      dataset_label: input.dataset === "pipeline" ? "Pipeline" : "Funded book",
      status: "receiving",
      status_label: "Receiving files",
      status_sentence: "Trakt is watching for the files to arrive.",
      auto_start_when_ready: input.auto_start_when_ready,
      files: [],
      input_roles: BATCH_ROLE_SPECS.map((spec) => ({
        role: spec.role,
        label: spec.label,
        required: spec.required,
        satisfied: false,
      })),
      missing_roles: BATCH_ROLE_SPECS.filter((s) => s.required).map((s) => s.role),
      configuration_ready: false,
      blocking_decisions: [],
      source_prefix: "",
      workflow_id: null,
      created_at: now,
      updated_at: now,
    };
    this.batches.unshift(batch);
    if (!this.clients.includes(input.client_id)) {
      this.clients.push(input.client_id);
    }
    return deepCopy(batch);
  }

  async listBatches(client?: string): Promise<Batch[]> {
    await this.wait();
    return deepCopy(client ? this.batches.filter((b) => b.client_id === client) : this.batches);
  }

  async getBatch(batchId: string, _client?: string): Promise<Batch> {
    await this.wait();
    const batch = this.findBatch(batchId);
    if (batch.status === "running") {
      const ticks = (this.batchTicks.get(batchId) ?? 0) + 1;
      this.batchTicks.set(batchId, ticks);
      if (ticks >= 3) {
        this.batchTicks.delete(batchId);
        batch.status = "completed";
        batch.status_label = "Done";
        batch.status_sentence = "The onboarding run has finished. You can open the workflow to see the result.";
        batch.updated_at = new Date().toISOString();
      }
    }
    return deepCopy(batch);
  }

  /** Mirrors the real route: content in, server-derived destination, no path. */
  async uploadBatchFiles(batchId: string, files: File[], _client?: string): Promise<Batch> {
    await this.wait();
    const batch = this.findBatch(batchId);
    if (batch.status === "running" || batch.status === "completed") {
      throw new OpsError("This onboarding has already started, so new files can't be added to it.");
    }
    if (files.length === 0) {
      throw new OpsError("Choose at least one file to send.");
    }

    batch.source_prefix =
      `${batch.client_id}/direct/${batch.dataset}/monthly/` +
      `${batch.portfolio_id}/${batch.reporting_date}`;

    for (const file of files) {
      const unsatisfied = batch.input_roles.filter((r) => !r.satisfied);
      const nextRequired = unsatisfied.find((r) => r.required) ?? unsatisfied[0];
      if (nextRequired) {
        this.satisfyRole(batch, nextRequired, file.name);
      } else {
        batch.files.push({
          source_file_id: this.id("sf"),
          filename: file.name,
          role: "",
          role_label: "Additional file",
          status: "recognised",
          status_sentence: "Received. Trakt will keep this alongside the pack.",
          confidence: 0.9,
        });
      }
    }

    this.refreshBatch(batch);
    // The intake path opens the workflow itself once a pack set to start on
    // its own is complete — the same rule an automated arrival follows.
    if (batch.status === "ready" && batch.auto_start_when_ready) {
      return this.startBatch(batchId);
    }
    return deepCopy(batch);
  }

  async startBatch(batchId: string, _client?: string): Promise<Batch> {
    await this.wait();
    const batch = this.findBatch(batchId);
    if (batch.status !== "ready") {
      throw new OpsError("This pack is not ready to start yet.");
    }
    const now = new Date().toISOString();
    const workflow: Workflow = {
      workflow_id: this.id("wf"),
      client_id: batch.client_id,
      portfolio_id: batch.portfolio_id,
      reporting_period: batch.reporting_date,
      outcome: batch.workflow_type,
      workflow_type: "new_portfolio",
      workflow_type_label: TYPE_LABELS.new_portfolio,
      status: "running",
      open_decisions: 0,
      created_at: now,
      updated_at: now,
      outcome_label: OUTCOME_LABELS[batch.workflow_type],
      status_sentence: "Trakt is working through the files now. This usually takes a few minutes.",
      interrupted: false,
      blockers: [],
      rerun_count: 0,
      stages: [
        {
          stage: "receive",
          label: "Files received",
          status: "completed",
          status_label: "Done",
          summary: `${batch.files.length} files were received.`,
        },
        {
          stage: "understand",
          label: "Understanding the files",
          status: "running",
          status_label: "In progress",
        },
        { stage: "check", label: "Checking the numbers", status: "waiting", status_label: "Waiting" },
        { stage: "prepare", label: "Preparing the report", status: "waiting", status_label: "Waiting" },
        { stage: "publish", label: "Publishing", status: "waiting", status_label: "Waiting" },
      ],
    };
    this.workflows.unshift(workflow);
    batch.workflow_id = workflow.workflow_id;
    batch.status = "running";
    batch.status_label = "In progress";
    batch.status_sentence = "Trakt is working through the files now. This usually takes a few minutes.";
    batch.updated_at = now;
    this.batchTicks.set(batchId, 0);
    return deepCopy(batch);
  }

  async startWorkflow(input: StartWorkflowInput): Promise<Workflow> {
    await this.wait();
    const delivery = this.deliveries.find((d) => d.delivery_id === input.delivery_id);
    if (!delivery) {
      throw new OpsError("That delivery could not be found. Register the files again.");
    }
    const workflowType = input.workflow_type ?? delivery.classification;
    const now = new Date().toISOString();
    const workflow: Workflow = {
      workflow_id: this.id("wf"),
      client_id: input.client_id,
      portfolio_id: delivery.portfolio_id,
      reporting_period: delivery.reporting_period,
      outcome: input.outcome,
      workflow_type: workflowType,
      workflow_type_label: TYPE_LABELS[workflowType],
      status: "running",
      open_decisions: 0,
      created_at: now,
      updated_at: now,
      outcome_label: OUTCOME_LABELS[input.outcome],
      status_sentence: "Trakt is working through the files now. This usually takes a few minutes.",
      interrupted: false,
      blockers: [],
      rerun_count: 0,
      stages: [
        {
          stage: "receive",
          label: "Files received",
          status: "completed",
          status_label: "Done",
          summary: `${delivery.files.length} files were received.`,
        },
        {
          stage: "understand",
          label: "Understanding the files",
          status: "running",
          status_label: "In progress",
        },
        { stage: "check", label: "Checking the numbers", status: "waiting", status_label: "Waiting" },
        { stage: "prepare", label: "Preparing the report", status: "waiting", status_label: "Waiting" },
        { stage: "publish", label: "Publishing", status: "waiting", status_label: "Waiting" },
      ],
    };
    this.workflows.unshift(workflow);
    return deepCopy(workflow);
  }

  async getWorkflows(params?: { client?: string; status?: string }): Promise<WorkflowRow[]> {
    await this.wait();
    let rows = this.workflows as WorkflowRow[];
    if (params?.client) {
      rows = rows.filter((w) => w.client_id === params.client);
    }
    if (params?.status) {
      rows = rows.filter((w) => w.status === params.status);
    } else {
      // Kept, but off the working list — it needs nothing from anybody.
      rows = rows.filter((w) => w.status !== "cancelled");
    }
    return deepCopy(rows);
  }

  async getWorkflow(workflowId: string): Promise<Workflow> {
    await this.wait();
    const workflow = this.findWorkflow(workflowId);
    if (workflow.status === "running") {
      const ticks = (this.runningTicks.get(workflowId) ?? 0) + 1;
      this.runningTicks.set(workflowId, ticks);
      if (ticks >= 3) {
        this.runningTicks.delete(workflowId);
        workflow.status = "awaiting_publication";
        workflow.status_sentence = "This report is ready. Look it over and publish when you're happy.";
        workflow.updated_at = new Date().toISOString();
        for (const stage of workflow.stages) {
          if (stage.stage === "publish") {
            stage.status = "ready";
            stage.status_label = "Ready";
            stage.summary = "Waiting for your approval to publish this as the official version.";
          } else {
            stage.status = "completed";
            stage.status_label = "Done";
          }
        }
      }
    }
    const full = deepCopy(workflow);
    full.steps = this.caseFile(workflow);
    // A blocked step wins; otherwise the furthest step still in progress.
    full.current_step =
      full.steps.find((s) => s.status === "blocked")?.key ??
      [...full.steps].reverse().find((s) => s.status === "current")?.key ??
      [...full.steps].reverse().find((s) => s.status === "complete")?.key ??
      full.steps[0].key;
    return full;
  }

  async rerunWorkflow(workflowId: string): Promise<Workflow> {
    await this.wait();
    const workflow = this.findWorkflow(workflowId);
    workflow.status = "running";
    workflow.status_sentence = "Trakt is running this again now.";
    workflow.blockers = [];
    workflow.interrupted = false;
    workflow.rerun_count += 1;
    workflow.updated_at = new Date().toISOString();
    this.runningTicks.set(workflowId, 0);
    for (const stage of workflow.stages) {
      if (stage.status === "blocked" || stage.status === "needs_review") {
        stage.status = "running";
        stage.status_label = "In progress";
        stage.blockers = [];
      }
    }
    return deepCopy(workflow);
  }

  async cancelWorkflow(workflowId: string, reason: string): Promise<Workflow> {
    await this.wait();
    const workflow = this.findWorkflow(workflowId);
    workflow.status = "cancelled";
    workflow.status_sentence = "This workflow was cancelled.";
    workflow.updated_at = new Date().toISOString();
    // A cancelled delivery has nothing left to decide, so its open questions
    // stop being asked. Superseded, not approved or rejected: neither answer
    // was given.
    for (const review of this.reviews) {
      if (review.workflow_id === workflowId && review.status === "open") {
        review.status = "superseded";
        review.resolved_at = workflow.updated_at;
      }
    }
    const row = (this.workflows as WorkflowRow[]).find((w) => w.workflow_id === workflowId);
    if (row) {
      row.status = "cancelled";
      row.open_decisions = 0;
    }
    void reason;
    return deepCopy(workflow);
  }

  async publishWorkflow(workflowId: string, rememberScope?: string): Promise<Publication> {
    await this.wait();
    const workflow = this.findWorkflow(workflowId);
    if (workflow.status !== "awaiting_publication") {
      throw new OpsError("This report is not ready to publish yet.");
    }
    this.approvalScopes.set(workflowId, rememberScope ?? "delivery");
    workflow.status = "published";
    workflow.status_sentence = "This report has been published.";
    workflow.updated_at = new Date().toISOString();
    for (const stage of workflow.stages) {
      stage.status = "completed";
      stage.status_label = "Done";
    }
    const publication: Publication = {
      publication_id: this.id("pub"),
      client_id: workflow.client_id,
      workflow_id: workflow.workflow_id,
      workflow_type_label: workflow.workflow_type_label,
      reporting_period: workflow.reporting_period,
      status: "published",
      version: 1,
      rule_version_count: 9,
      previous_publication_id: null,
      approved_by: "You",
      published_at: new Date().toISOString(),
      prepared_at: workflow.updated_at,
    };
    this.publications.unshift(publication);
    return deepCopy(publication);
  }

  async holdWorkflow(workflowId: string, _reason: string): Promise<Publication> {
    await this.wait();
    const workflow = this.findWorkflow(workflowId);
    workflow.status = "held";
    workflow.status_sentence = "This report is on hold. Run it again when you're ready.";
    workflow.updated_at = new Date().toISOString();
    return deepCopy({
      publication_id: this.id("pub"),
      client_id: workflow.client_id,
      workflow_id: workflow.workflow_id,
      workflow_type_label: workflow.workflow_type_label,
      reporting_period: workflow.reporting_period,
      status: "prepared",
      version: 1,
      rule_version_count: 9,
      previous_publication_id: null,
      approved_by: "",
      published_at: null,
      prepared_at: workflow.updated_at,
    });
  }

  async getReviews(params?: { client?: string; workflow_id?: string }): Promise<Review[]> {
    await this.wait();
    // The queue is what is still to be answered, matching the API, which lists
    // open decisions only.
    let reviews = this.reviews.filter((r) => r.status === "open");
    if (params?.client) {
      reviews = reviews.filter((r) => r.client_id === params.client);
    }
    if (params?.workflow_id) {
      reviews = reviews.filter((r) => r.workflow_id === params.workflow_id);
    }
    return deepCopy(reviews);
  }

  async getReview(decisionId: string): Promise<Review> {
    await this.wait();
    const review = this.reviews.find((r) => r.decision_id === decisionId);
    if (!review) {
      throw new OpsError("That question could not be found.");
    }
    return deepCopy(review);
  }

  async submitDecision(decisionId: string, input: DecisionInput): Promise<DecisionResult> {
    await this.wait();
    const review = this.reviews.find((r) => r.decision_id === decisionId);
    if (!review) {
      throw new OpsError("That question could not be found.");
    }
    review.status = "resolved";
    review.resolved_by = "You";
    review.resolved_at = new Date().toISOString();

    const workflow = this.workflows.find((w) => w.workflow_id === review.workflow_id);
    let rerunScheduled = false;
    if (workflow && workflow.open_decisions > 0) {
      workflow.open_decisions -= 1;
      if (workflow.open_decisions === 0 && input.action !== "reject") {
        workflow.status = "running";
        workflow.status_sentence = "Thanks — Trakt is re-running the affected step now.";
        this.runningTicks.set(workflow.workflow_id, 0);
        for (const stage of workflow.stages) {
          if (stage.status === "needs_review") {
            stage.status = "running";
            stage.status_label = "In progress";
            stage.decision_count = 0;
          }
        }
        rerunScheduled = true;
      }
      workflow.updated_at = new Date().toISOString();
    }

    let rule: Rule | null = null;
    if (input.action !== "reject") {
      const chosen = review.options.find((o) => o.value === input.value);
      rule = {
        rule_id: this.id("rule"),
        version: 1,
        kind: review.kind,
        scope: input.scope,
        client_id: review.client_id,
        portfolio_id: "",
        status: "approved",
        source_term: review.title,
        approved_meaning: chosen?.label ?? input.value,
        description: input.reason || "Approved from a review decision.",
        approved_by: "You",
        approved_at: new Date().toISOString(),
      };
      this.rules.unshift(rule);
    }

    return deepCopy({ review, rule, rerun_scheduled: rerunScheduled });
  }

  async getRules(params?: RulesQuery): Promise<Rule[]> {
    await this.wait();
    let rules = this.rules;
    if (params?.client) {
      rules = rules.filter((r) => r.client_id === params.client);
    }
    if (params?.kind) {
      rules = rules.filter((r) => r.kind === params.kind);
    }
    if (params?.scope) {
      rules = rules.filter((r) => r.scope === params.scope);
    }
    if (params?.q) {
      const q = params.q.toLowerCase();
      rules = rules.filter(
        (r) =>
          r.source_term.toLowerCase().includes(q) ||
          r.approved_meaning.toLowerCase().includes(q) ||
          r.client_id.toLowerCase().includes(q),
      );
    }
    return deepCopy(rules);
  }

  async getRuleHistory(ruleId: string): Promise<Rule[]> {
    await this.wait();
    const rule = this.rules.find((r) => r.rule_id === ruleId);
    if (!rule) {
      throw new OpsError("That rule could not be found.");
    }
    const history: Rule[] = [deepCopy(rule)];
    for (let v = rule.version - 1; v >= 1; v -= 1) {
      history.push({
        ...deepCopy(rule),
        version: v,
        approved_meaning: v === 1 ? `${rule.approved_meaning} (first version)` : rule.approved_meaning,
        approved_at: "2026-03-14T10:00:00Z",
        approved_by: "J. Whitfield",
      });
    }
    return history;
  }

  async getHistory(client?: string): Promise<Publication[]> {
    await this.wait();
    return deepCopy(
      client ? this.publications.filter((p) => p.client_id === client) : this.publications,
    );
  }

  async getPrincipal(): Promise<Principal> {
    await this.wait();
    return deepCopy(this.principal);
  }

  async getConfigOverview(): Promise<ConfigOverview> {
    await this.wait();
    this.requireAdmin();
    return deepCopy(this.config.overview());
  }

  async getConfigCatalogue(): Promise<ConfigCatalogue> {
    await this.wait();
    this.requireAdmin();
    return deepCopy(this.config.catalogue());
  }

  async getConfigVersion(
    layer: ConfigLayer,
    version: number,
    file?: string,
  ): Promise<ConfigVersion> {
    await this.wait();
    this.requireAdmin();
    return deepCopy(this.config.version(layer, version, file));
  }

  async compareConfigVersions(layer: ConfigLayer, from: number, to: number): Promise<Comparison> {
    await this.wait();
    this.requireAdmin();
    return deepCopy(this.config.compare(layer, from, to));
  }

  async getConfigImpact(layer: ConfigLayer, version?: number): Promise<ImpactAnalysis> {
    await this.wait();
    this.requireAdmin();
    return deepCopy(this.config.impact(layer, version));
  }

  async getConfigAudit(): Promise<AuditTrail> {
    await this.wait();
    this.requireAdmin();
    return deepCopy(this.config.auditTrail());
  }

  async createConfigDraft(
    layer: ConfigLayer,
    input?: CreateDraftInput,
  ): Promise<{ version: number; status: string }> {
    await this.wait();
    this.requireAdmin();
    return this.config.createDraft(layer, input);
  }

  async validateConfigVersion(
    layer: ConfigLayer,
    version: number,
  ): Promise<{ validation: ValidationResult; validation_summary: ValidationSummary }> {
    await this.wait();
    this.requireAdmin();
    return deepCopy(this.config.validate(layer, version));
  }

  async activateConfigVersion(
    layer: ConfigLayer,
    version: number,
  ): Promise<{ active_version: number }> {
    await this.wait();
    this.requireAdmin();
    return this.config.activate(layer, version);
  }

  async rollbackConfig(
    layer: ConfigLayer,
    toVersion: number,
  ): Promise<{ active_version: number }> {
    await this.wait();
    this.requireAdmin();
    return this.config.rollback(layer, toVersion);
  }

  // -- Client Onboarding ---------------------------------------------------- //

  async getOnboardingReference(): Promise<OnboardingReference> {
    await this.wait();
    return this.onboarding.reference();
  }

  async getOnboardingHome(): Promise<OnboardingHome> {
    await this.wait();
    return this.onboarding.home();
  }

  async startNewClientCase(): Promise<OnboardingCase> {
    await this.wait();
    return this.onboarding.startNewClient(this.principal.name);
  }

  async startMigrationCase(clientId: string): Promise<OnboardingCase> {
    await this.wait();
    return this.onboarding.startMigration(clientId, this.principal.name);
  }

  async startAmendmentCase(clientId: string): Promise<OnboardingCase> {
    await this.wait();
    return this.onboarding.startAmendment(clientId, this.principal.name);
  }

  async getCase(caseId: string): Promise<OnboardingCase> {
    await this.wait();
    return this.onboarding.case(caseId);
  }

  async saveCaseStep(
    caseId: string,
    step: string,
    payload: Record<string, unknown>,
  ): Promise<OnboardingCase> {
    await this.wait();
    return this.onboarding.saveStep(caseId, step, payload);
  }

  async addPipelineBook(caseId: string, portfolioId: string): Promise<OnboardingCase> {
    await this.wait();
    return this.onboarding.addPipelineBook(caseId, portfolioId);
  }

  async registerSample(
    caseId: string,
    files: { name: string; headers?: string[] }[],
  ): Promise<OnboardingCase> {
    await this.wait();
    return this.onboarding.registerSample(caseId, files);
  }

  async removeSource(
    caseId: string,
    portfolioId: string,
    dataset: string,
  ): Promise<OnboardingCase> {
    await this.wait();
    return this.onboarding.removeSource(caseId, portfolioId, dataset);
  }

  async getCaseChecklist(caseId: string): Promise<ChecklistRow[]> {
    await this.wait();
    return this.onboarding.checklist(caseId);
  }

  async createInformationRequest(
    caseId: string,
    items: ChecklistRow[],
    options?: { responsible_party?: string; due_date?: string; note?: string },
  ): Promise<OnboardingCase> {
    await this.wait();
    return this.onboarding.createRequest(caseId, items, options);
  }

  async markRequestSent(caseId: string, requestId: string): Promise<OnboardingCase> {
    await this.wait();
    return this.onboarding.markSent(caseId, requestId);
  }

  async recordRequestResponse(
    caseId: string,
    requestId: string,
    body: {
      note?: string;
      answers?: Record<string, unknown>;
      evidence?: { name: string; reference: string }[];
    },
  ): Promise<OnboardingCase> {
    await this.wait();
    return this.onboarding.recordResponse(caseId, requestId, body);
  }

  async addCaseQuestion(caseId: string, question: string): Promise<OnboardingCase> {
    await this.wait();
    return this.onboarding.addQuestion(caseId, question);
  }

  async resolveCaseQuestion(
    caseId: string,
    questionId: string,
    resolution: string,
  ): Promise<OnboardingCase> {
    await this.wait();
    return this.onboarding.resolveQuestion(caseId, questionId, resolution);
  }

  async getCasePreview(caseId: string): Promise<CasePreview> {
    await this.wait();
    return this.onboarding.preview(caseId);
  }

  async submitCase(caseId: string): Promise<OnboardingCase> {
    await this.wait();
    return this.onboarding.submit(caseId);
  }

  /** Approval is an administrator act, exactly as the server enforces. */
  async approveCase(caseId: string, reason: string): Promise<OnboardingCase> {
    await this.wait();
    this.requireAdmin();
    return this.onboarding.approve(caseId, reason, this.principal.name);
  }

  async activateCase(caseId: string): Promise<ActivationResult> {
    await this.wait();
    this.requireAdmin();
    return this.onboarding.activate(caseId, this.principal.name);
  }

  async withdrawCase(caseId: string, reason: string): Promise<OnboardingCase> {
    await this.wait();
    return this.onboarding.withdraw(caseId, reason);
  }

  async getOnboardingClient(clientId: string): Promise<OnboardingClientDetail> {
    await this.wait();
    return this.onboarding.client(clientId);
  }

  async getOnboardingClientVersion(
    clientId: string,
    version: number,
  ): Promise<ConfigurationVersionRow> {
    await this.wait();
    return this.onboarding.version(clientId, version);
  // -- OCC Agent ----------------------------------------------------------- //

  async getAgentMeta(): Promise<AgentMeta> {
    await this.wait();
    return this.agent.meta();
  }

  async listAgentCases(state?: string): Promise<CaseSummary[]> {
    await this.wait();
    return this.agent.list(state);
  }

  async createAgentCase(instruction: string, fixtureId?: string): Promise<AgentStatus> {
    await this.wait();
    return this.agent.create(instruction, fixtureId);
  }

  async getAgentCase(caseId: string): Promise<AgentStatus> {
    await this.wait();
    return this.agent.status(caseId);
  }

  async instructAgent(caseId: string, text: string, confirm = false): Promise<AgentTurn> {
    await this.wait();
    return this.agent.instruct(caseId, text, confirm);
  }

  async answerAgentDecision(
    caseId: string,
    input: { decision_id: string; action: string; value?: string; reason?: string },
  ): Promise<AgentStatus> {
    await this.wait();
    return this.agent.answerDecision(caseId, input);
  }

  async runAgentStep(caseId: string, step: string): Promise<AgentStatus> {
    await this.wait();
    return this.agent.step(caseId, step);
  }

  async uploadAgentArtefacts(caseId: string, files: File[]): Promise<AgentStatus> {
    await this.wait();
    return this.agent.uploadArtefacts(
      caseId,
      files.map((f) => f.name),
    );
  }

  async loadAgentFixtureArtefacts(caseId: string, fixtureId: string): Promise<AgentStatus> {
    await this.wait();
    return this.agent.loadFixtureArtefacts(caseId, fixtureId);
  }

  async returnAgentCaseToStage(caseId: string, targetState: string): Promise<AgentStatus> {
    await this.wait();
    return this.agent.returnToStage(caseId, targetState);
  }

  async runAgentScenario(fixtureId: string): Promise<AgentStatus> {
    await this.wait();
    return this.agent.runScenario(fixtureId);
  }

  async getAgentPack(caseId: string): Promise<AgentPack> {
    await this.wait();
    return this.agent.pack(caseId);
  }

  async getAgentReadiness(caseId: string): Promise<AgentReadinessPackage> {
    await this.wait();
    return this.agent.readinessPackage(caseId);
  }

  async getAgentAudit(caseId: string): Promise<AgentAudit> {
    await this.wait();
    return this.agent.audit(caseId);
  }
}
