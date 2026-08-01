/**
 * MockConcentration — a stateful, in-memory stand-in for the concentration
 * review API, mirroring the backend's governance rules exactly: extraction
 * yields matched / ambiguous / unsupported proposals; approval is refused
 * while anything is unresolved (same wording shape as the server's 409);
 * activation only reads approved proposals; approved proposals are immutable
 * except through supersede. Used by `VITE_OPS_MODE=mock` and by the tests —
 * nothing intercepts fetch.
 */

import { OpsError } from "./OpsClient";
import type {
  ActiveConfigurationDoc,
  ConcentrationReviewPackage,
  LibraryMetricDoc,
  ProposalUpdatePayload,
  TestProposal,
  VersionHeader,
} from "./concentrationTypes";

const NET_WAC_QUESTION =
  "Does 'Net WAC' mean customer coupon net of servicing fee only, or net of "
  + "servicing, hedging and funding costs? Please give the deduction in "
  + "percentage points.";

const LIBRARY: LibraryMetricDoc[] = [
  {
    metric_id: "geo_region_share",
    display_name: "Regional exposure",
    category: "geography",
    description:
      "Balance-weighted share of the portfolio secured on properties in one or more named regions.",
    unit: "percent",
    supported_operators: ["max", "min"],
    parameters: {
      regions: { type: "list_string", required: true },
      denominator: { type: "enum", values: ["current_balance", "original_balance"] },
    },
    implementation_status: "implemented",
  },
  {
    metric_id: "rate_net_wac",
    display_name: "Net WAC",
    category: "rate_product",
    description: "Balance-weighted average coupon net of a confirmed deduction.",
    unit: "percent",
    supported_operators: ["max", "min"],
    parameters: {
      deduction_percent: { type: "number", required: true },
      deduction_basis: {
        type: "enum",
        values: ["servicing_fee_only", "servicing_hedging_funding", "other_confirmed"],
        required: true,
      },
    },
    implementation_status: "implemented",
  },
];

let seq = 0;

function proposal(partial: Partial<TestProposal>): TestProposal {
  seq += 1;
  return {
    proposal_id: `ctp_mock_${seq}`,
    onboarding_run_id: "",
    client_id: "",
    portfolio_id: "",
    evidence: { source_reference: "", source_text: "", locator: "", supplied_at: "" },
    extracted_test_name: "",
    extracted_category: "",
    proposed_metric_id: "",
    match_outcome: "unsupported",
    extracted_operator: "",
    extracted_threshold: null,
    extracted_unit: "percent",
    numerator_basis: "",
    denominator_basis: "",
    weighting_basis: "",
    parameters: {},
    segment_filters: {},
    bucket_boundaries: [],
    effective_date: "",
    expiry_date: "",
    cure_or_waiver_notes: "",
    extraction_confidence: 0.9,
    mapping_confidence: 0.9,
    concern_codes: [],
    concern_messages: [],
    confirmation_questions: [],
    confirmation_answers: [],
    status: "pending_approval",
    approval: {},
    severity: "high",
    operator_notes: [],
    implementation_request: {},
    created_at: "2026-01-10T09:00:00+00:00",
    updated_at: "2026-01-10T09:00:00+00:00",
    version: 1,
    ...partial,
  };
}

export class MockConcentration {
  private proposals = new Map<string, TestProposal[]>();
  private versions = new Map<string, ActiveConfigurationDoc[]>();

  constructor(private readonly operator: string = "Administrator") {}

  private list(client: string): TestProposal[] {
    if (!this.proposals.has(client)) this.proposals.set(client, []);
    return this.proposals.get(client)!;
  }

  private find(client: string, proposalId: string): TestProposal {
    const found = this.list(client).find((p) => p.proposal_id === proposalId);
    if (!found) throw new OpsError("That could not be found.", "OPS_NOT_FOUND");
    return found;
  }

  review(client: string): ConcentrationReviewPackage {
    const proposals = this.list(client);
    const counts: Record<string, number> = {};
    for (const p of proposals) counts[p.status] = (counts[p.status] ?? 0) + 1;
    const versions = this.versions.get(client) ?? [];
    const current = versions.find((v) => v.status === "active") ?? null;
    const headers: VersionHeader[] = versions.map((v) => ({
      version: v.version,
      status: v.status,
      activated_by: v.activated_by,
      activated_at: v.activated_at,
      test_count: v.tests.length,
      content_hash: v.content_hash,
      reason: v.reason,
    }));
    return {
      clientId: client,
      proposals,
      countsByStatus: counts,
      currentVersion: current?.version ?? null,
      currentTestCount: current?.tests.length ?? 0,
      versions: headers,
      libraryVersion: "1.0.0",
    };
  }

  library(): { libraryVersion: string; metrics: LibraryMetricDoc[] } {
    return { libraryVersion: "1.0.0", metrics: LIBRARY };
  }

  extract(client: string, sourceReference: string, text: string): TestProposal[] {
    if (!text.trim()) {
      throw new OpsError(
        "Nothing to extract: provide the client's response text or a limits table.",
        "CONCENTRATION_REFUSED",
      );
    }
    const created: TestProposal[] = [];
    const lowered = text.toLowerCase();
    if (lowered.includes("east anglia")) {
      created.push(proposal({
        client_id: client,
        evidence: {
          source_reference: sourceReference,
          source_text:
            "The aggregate Current Balance of Loans secured on Properties located "
            + "in East Anglia must not exceed 25% of the Portfolio.",
          locator: "clause 1.1",
          supplied_at: "",
        },
        extracted_test_name: "East Anglia exposure",
        extracted_category: "geography",
        proposed_metric_id: "geo_region_share",
        match_outcome: "matched",
        extracted_operator: "max",
        extracted_threshold: 25,
        parameters: { regions: ["East Anglia"] },
        status: "pending_approval",
      }));
    }
    if (lowered.includes("net wac")) {
      created.push(proposal({
        client_id: client,
        evidence: {
          source_reference: sourceReference,
          source_text: "The Net WAC of the Portfolio must be at least 3.75%.",
          locator: "clause 5.1",
          supplied_at: "",
        },
        extracted_test_name: "Net WAC floor",
        extracted_category: "rate_product",
        proposed_metric_id: "rate_net_wac",
        match_outcome: "ambiguous",
        extracted_operator: "min",
        extracted_threshold: 3.75,
        concern_codes: ["net_wac_definition_uncertain"],
        concern_messages: [NET_WAC_QUESTION],
        confirmation_questions: [NET_WAC_QUESTION],
        status: "pending_confirmation",
        mapping_confidence: 0.5,
      }));
    }
    if (lowered.includes("duration")) {
      created.push(proposal({
        client_id: client,
        evidence: {
          source_reference: sourceReference,
          source_text: "The portfolio duration must not exceed 7 years.",
          locator: "clause 9.1",
          supplied_at: "",
        },
        extracted_test_name: "Portfolio duration cap",
        match_outcome: "unsupported",
        extracted_operator: "max",
        concern_codes: ["unsupported_metric"],
        concern_messages: ["unsupported metric"],
        status: "unsupported",
        mapping_confidence: 0,
        implementation_request: {
          requested_capability: "Portfolio duration cap",
          reason: "No governed metric represents this test.",
        },
      }));
    }
    this.list(client).push(...created);
    return created;
  }

  answer(client: string, proposalId: string, question: string, answer: string): TestProposal {
    const p = this.find(client, proposalId);
    p.confirmation_answers.push({
      question, answer, answered_by: this.operator,
      answered_at: "2026-01-10T10:00:00+00:00",
    });
    p.confirmation_questions = p.confirmation_questions.filter((q) => q !== question);
    p.operator_notes.push({
      at: "2026-01-10T10:00:00+00:00", actor: this.operator,
      note: `Answered: ${question}`,
    });
    return p;
  }

  update(client: string, proposalId: string, payload: ProposalUpdatePayload): TestProposal {
    const p = this.find(client, proposalId);
    if (p.status === "approved" || p.status === "superseded") {
      throw new OpsError(
        "An approved proposal is immutable; supersede it instead.",
        "CONCENTRATION_REFUSED",
      );
    }
    if (payload.parameters) p.parameters = payload.parameters;
    if (payload.threshold !== undefined) p.extracted_threshold = payload.threshold;
    if (payload.operator) p.extracted_operator = payload.operator;
    if (payload.effective_date) p.effective_date = payload.effective_date;
    for (const code of payload.resolved_concerns ?? []) {
      p.concern_codes = p.concern_codes.filter((c) => c !== code);
    }
    p.version += 1;
    if (p.status === "pending_confirmation" && p.concern_codes.length === 0
        && p.confirmation_questions.length === 0) {
      p.status = "pending_approval";
    }
    p.operator_notes.push({
      at: "2026-01-10T10:05:00+00:00", actor: this.operator,
      note: payload.note || "Edited the proposal.",
    });
    return p;
  }

  approvalBlockers(p: TestProposal): string[] {
    const blockers: string[] = [];
    if (p.match_outcome === "unsupported" || p.status === "unsupported") {
      blockers.push("This test has no supported implementation; mark it unsupported instead.");
    }
    if (p.concern_codes.length > 0) {
      blockers.push(`Unresolved concerns remain: ${p.concern_codes.join(", ")}.`);
    }
    if (p.confirmation_questions.length > 0) {
      blockers.push("Unanswered confirmation questions remain.");
    }
    if (p.extracted_threshold === null) blockers.push("The proposal has no threshold.");
    if (p.extracted_operator !== "max" && p.extracted_operator !== "min") {
      blockers.push("The comparison direction is unresolved.");
    }
    return blockers;
  }

  approve(client: string, proposalId: string,
          payload: { effective_date?: string; comments?: string }): TestProposal {
    const p = this.find(client, proposalId);
    const blockers = this.approvalBlockers(p);
    if (p.status !== "pending_approval" && p.status !== "pending_confirmation") {
      blockers.unshift(`A proposal in status '${p.status}' cannot be approved.`);
    }
    if (blockers.length) {
      throw new OpsError(blockers.join(" "), "CONCENTRATION_REFUSED");
    }
    p.status = "approved";
    p.approval = {
      decision: "approved", operator: this.operator,
      decided_at: "2026-01-10T11:00:00+00:00",
      approved_metric_id: p.proposed_metric_id,
      approved_parameters: p.parameters,
      source_version: p.version,
      effective_date: payload.effective_date ?? p.effective_date,
      comments: payload.comments ?? "",
      prior_active_test_id: "",
    };
    if (payload.effective_date) p.effective_date = payload.effective_date;
    p.operator_notes.push({
      at: "2026-01-10T11:00:00+00:00", actor: this.operator,
      note: `Approved. ${payload.comments ?? ""}`.trim(),
    });
    return p;
  }

  decide(client: string, proposalId: string,
         status: "rejected" | "unsupported" | "not_applicable" | "clarification_requested",
         note: string): TestProposal {
    const p = this.find(client, proposalId);
    if (p.status === "approved" || p.status === "superseded") {
      throw new OpsError(
        "An approved proposal is immutable; supersede it instead.",
        "CONCENTRATION_REFUSED",
      );
    }
    p.status = status;
    if (status === "clarification_requested" && !p.confirmation_questions.includes(note)) {
      p.confirmation_questions.push(note);
    }
    p.operator_notes.push({
      at: "2026-01-10T11:10:00+00:00", actor: this.operator, note,
    });
    return p;
  }

  supersede(client: string, proposalId: string, reason: string): TestProposal {
    const p = this.find(client, proposalId);
    if (p.status !== "approved") {
      throw new OpsError(
        "Only an approved proposal can be superseded.", "CONCENTRATION_REFUSED",
      );
    }
    p.status = "superseded";
    p.operator_notes.push({
      at: "2026-01-10T11:20:00+00:00", actor: this.operator,
      note: `Superseded: ${reason}`,
    });
    return p;
  }

  activate(client: string, reason = ""): ActiveConfigurationDoc {
    const approved = this.list(client).filter((p) => p.status === "approved");
    if (!approved.length) {
      throw new OpsError(
        "No approved proposals — there is nothing to activate.",
        "CONCENTRATION_REFUSED",
      );
    }
    const versions = this.versions.get(client) ?? [];
    for (const v of versions) v.status = "superseded";
    const config: ActiveConfigurationDoc = {
      client_id: client,
      version: versions.length + 1,
      status: "active",
      tests: approved.map((p, i) => ({
        test_id: `ct_mock_${versions.length + 1}_${i}`,
        proposal_id: p.proposal_id,
        metric_id: p.proposed_metric_id,
        display_name: p.extracted_test_name,
        category: p.extracted_category,
        operator: p.extracted_operator,
        threshold: p.extracted_threshold,
        unit: p.extracted_unit,
        parameters: p.parameters,
        effective_date: p.effective_date,
        evidence: p.evidence,
        approval: p.approval as ActiveConfigurationDoc["tests"][number]["approval"],
        severity: p.severity,
      })),
      activated_by: this.operator,
      activated_at: "2026-01-10T12:00:00+00:00",
      based_on_version: versions.length || null,
      reason,
      content_hash: `hash_${versions.length + 1}`,
    };
    versions.push(config);
    this.versions.set(client, versions);
    return config;
  }

  version(client: string, versionNumber: number): ActiveConfigurationDoc {
    const found = (this.versions.get(client) ?? []).find(
      (v) => v.version === versionNumber,
    );
    if (!found) throw new OpsError("That could not be found.", "OPS_NOT_FOUND");
    return found;
  }
}
