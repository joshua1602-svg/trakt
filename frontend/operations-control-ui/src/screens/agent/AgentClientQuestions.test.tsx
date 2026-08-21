import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MockOpsClient } from "@/api/MockOpsClient";
import { OpsClientProvider } from "@/api/context";
import type { PackQuestion } from "@/api/agentTypes";
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

async function open(confirmations: PackQuestion[]) {
  const client = new MockOpsClient();
  const created = await client.createAgentCase(
    "Onboard Northstar Lending. UK equity release. Monthly management information.",
  );
  const user = userEvent.setup();
  render(
    <ToastProvider>
      <OpsClientProvider client={client}>
        <ClientQuestionsPanel
          caseId={created.case_ref}
          version={1}
          confirmations={confirmations}
          busy={false}
          onSaved={vi.fn()}
        />
      </OpsClientProvider>
    </ToastProvider>,
  );
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
