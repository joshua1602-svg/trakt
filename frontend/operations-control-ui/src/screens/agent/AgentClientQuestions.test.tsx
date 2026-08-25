import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MockOpsClient } from "@/api/MockOpsClient";
import { OpsClientProvider } from "@/api/context";
import type { PackQuestion } from "@/api/agentTypes";
import type { ChecklistRow, InformationRequest } from "@/api/onboardingTypes";
import { ToastProvider } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { ClientQuestionsPanel } from "./AgentClientQuestions";

/**
 * The panel's own contract, tested with its props supplied directly.
 *
 * Driving the whole app to produce already-known values means drafting a pack
 * and waiting on the stage machine, which tests the mock's timeline rather than
 * this component. What matters here is narrower and worth pinning exactly:
 * already-known values are SHOWN and never offered an input, because a value
 * Trakt derived is corrected in the conversation — where the change is read,
 * put to the operator, and recorded with who said it.
 */
function known(): PackQuestion[] {
  const row = (field: string, label: string, value: unknown, item = ""): PackQuestion => ({
    section: "client",
    field,
    label,
    help: "",
    status: "answered",
    value,
    provenance: "",
    index: null,
    item,
    required: true,
    evidence_required: false,
    sensitive: false,
    writes_to: "",
    step: "",
    step_label: "",
  });
  return [
    row("client_name", "Client name", "Northstar Lending"),
    row("jurisdiction", "Jurisdiction", "GB"),
    row("cadence", "Expected cadence", "monthly", "funded"),
  ];
}

function outstanding(): ChecklistRow[] {
  const row = (section: string, field: string, label: string): ChecklistRow => ({
    section,
    section_label: section,
    field,
    label,
    help: "",
    index: null,
    scope: "client_supplied",
    evidence_required: false,
    sensitive: false,
  });
  return [
    row("contacts", "reporting_contact_email", "Reporting email"),
    row("identity", "legal_entity_identifier", "Legal entity identifier"),
  ];
}

function mount(options: {
  confirmations?: PackQuestion[];
  checklist?: ChecklistRow[];
  requests?: InformationRequest[];
} = {}) {
  const client = new MockOpsClient();
  const user = userEvent.setup();
  const created = client.createAgentCase(
    "Onboard Northstar Lending. UK equity release. Monthly management information.",
  );
  return created.then((made) => {
    render(
      <ToastProvider>
        <OpsClientProvider client={client}>
          <ClientQuestionsPanel
            caseId={made.case_ref}
            version={1}
            confirmations={options.confirmations ?? []}
            checklist={options.checklist ?? outstanding()}
            requests={options.requests ?? []}
            busy={false}
            onSaved={vi.fn()}
          />
        </OpsClientProvider>
      </ToastProvider>,
    );
    return user;
  });
}

async function open(confirmations: PackQuestion[]) {
  const user = await mount({ confirmations });
  await user.click(screen.getByRole("button", { name: copy.agent.questionsShow }));
  await screen.findByLabelText(/Reporting email/);
  return user;
}

describe("the client questions panel", () => {
  it("shows what Trakt already knows", async () => {
    await open(known());
    expect(screen.getByText(copy.agent.questionsKnownHeading)).toBeInTheDocument();
    expect(screen.getByText("Northstar Lending")).toBeInTheDocument();
    expect(screen.getByText("GB")).toBeInTheDocument();
  });

  it("never offers an input for a value Trakt already holds", async () => {
    await open(known());
    const block = screen.getByText(copy.agent.questionsKnownHeading).closest("div");
    expect(block).toBeTruthy();
    expect(block!.querySelectorAll("input, select, textarea")).toHaveLength(0);
  });

  it("says where an already-known value IS corrected", async () => {
    await open(known());
    expect(screen.getByText(copy.agent.questionsKnownHelp)).toBeInTheDocument();
  });

  it("distinguishes two rows that share a label", async () => {
    await open(known());
    expect(screen.getByText(/funded/)).toBeInTheDocument();
  });

  it("says nothing about already-known values when there are none", async () => {
    await open([]);
    expect(screen.queryByText(copy.agent.questionsKnownHeading)).not.toBeInTheDocument();
  });
});

/**
 * The outstanding list and the form used to be two separate panels — Client
 * Onboarding's checklist in the rail, the answerable form inside a timeline
 * stage. So the list of what the client still owes was in one place and the
 * way to satisfy it was in another, and the second one disappeared when its
 * stage completed. They are the same subject and are now one panel.
 */
describe("the client questions panel — what is still outstanding", () => {
  it("lists what the client still owes without opening the form", async () => {
    await mount();
    expect(screen.getByText(copy.agent.questionsOutstanding(2))).toBeInTheDocument();
    expect(screen.getByText("Legal entity identifier")).toBeInTheDocument();
  });

  it("withdraws the list once the form is open", async () => {
    const user = await mount();
    await user.click(screen.getByRole("button", { name: copy.agent.questionsShow }));
    await screen.findByLabelText(/Reporting email/);
    // The form itself lists every unanswered question. The same list twice on
    // one screen is noise an operator has to reconcile.
    expect(screen.queryByText("Legal entity identifier")).not.toBeInTheDocument();
    expect(screen.getByText(copy.agent.questionsOutstanding(2))).toBeInTheDocument();
  });

  it("says so plainly when the client owes nothing", async () => {
    await mount({ checklist: [] });
    expect(screen.getByText(copy.agent.checklistEmpty)).toBeInTheDocument();
  });

  /**
   * Chasing a client is a step with a place in the sequence, and it stays on
   * the timeline where the sequence is. A second ask button here would be a
   * third one on the page.
   */
  it("does not offer to chase the client from the rail", async () => {
    await mount();
    expect(screen.queryByRole("button", { name: copy.agent.checklistAsk }))
      .not.toBeInTheDocument();
  });

  it("says what has been asked for and whether it came back", async () => {
    await mount({
      requests: [
        {
          request_id: "REQ-1",
          items: outstanding(),
          responsible_party: "client",
          status: "sent",
          requested_by: "ops",
          requested_at: "",
          sent_at: "",
        } as InformationRequest,
      ],
    });
    expect(screen.getByText(copy.agent.requestsHeading)).toBeInTheDocument();
    expect(screen.getByText(copy.agent.requestOutstanding)).toBeInTheDocument();
  });
});
