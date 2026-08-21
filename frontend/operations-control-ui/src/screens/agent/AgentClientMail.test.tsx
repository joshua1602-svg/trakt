import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MockOpsClient } from "@/api/MockOpsClient";
import { OpsClientProvider } from "@/api/context";
import { ToastProvider } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { ClientMailPanel } from "./AgentClientMail";

/**
 * The panel's contract, driven through the mock client.
 *
 * What is pinned here is the part that is a safety property rather than a
 * layout choice: an unmatched message is SHOWN and has no button. Hiding it
 * would leave a client's reply unread; offering a button for it would invite
 * exactly the mistake the correlator refuses to make.
 */
async function panel(options: { issue?: boolean; canRegister?: boolean } = {}) {
  const client = new MockOpsClient();
  const created = await client.createAgentCase(
    "Onboard Northstar Lending. UK equity release. Monthly management information.",
  );
  const caseRef = created.case_ref;
  if (options.issue !== false) {
    await client.draftAgentPack(caseRef);
    await client.approveAgentPack(caseRef);
    await client.sendAgentPack(caseRef, ["ops@northstar.example"]);
  }
  const onIngested = vi.fn();
  const user = userEvent.setup();
  render(
    <ToastProvider>
      <OpsClientProvider client={client}>
        <ClientMailPanel
          caseId={caseRef}
          busy={false}
          canRegister={options.canRegister ?? true}
          onIngested={onIngested}
        />
      </OpsClientProvider>
    </ToastProvider>,
  );
  return { user, client, caseRef, onIngested };
}

async function opened(options: { issue?: boolean; canRegister?: boolean } = {}) {
  const context = await panel(options);
  await context.user.click(screen.getByRole("button", { name: copy.agent.mailShow }));
  return context;
}

describe("the client mail panel", () => {
  it("reads nothing until the operator asks", async () => {
    const { client } = await panel();
    const spy = vi.spyOn(client, "getAgentMail");
    expect(spy).not.toHaveBeenCalled();
    expect(screen.getByText(copy.agent.mailClosed)).toBeInTheDocument();
  });

  it("shows a reply that belongs to this case", async () => {
    await opened();
    expect(await screen.findByText(/RE: Onboarding information request/))
      .toBeInTheDocument();
    expect(screen.getByText(/funded_book\.csv/)).toBeInTheDocument();
  });

  it("says why the reply was tied to this case", async () => {
    await opened();
    const heading = await screen.findByText(/RE: Onboarding information request/);
    const row = heading.closest("li") as HTMLElement;
    // The evidence, in words an operator reads — not the correlator's own
    // vocabulary. Taking a client's file onto a case should not require
    // opening an audit trail to see the basis for it. Scoped to the row,
    // because the unmatched section explains the same rule in general terms.
    expect(within(row).getByText(/the conversation it belongs to/))
      .toBeInTheDocument();
    expect(within(row).getByText(/the reference in its subject/))
      .toBeInTheDocument();
  });

  it("shows an unmatched message rather than hiding it", async () => {
    await opened();
    expect(await screen.findByText(copy.agent.mailUnmatchedHeading))
      .toBeInTheDocument();
    expect(screen.getByText("Some files for you")).toBeInTheDocument();
  });

  it("offers no way to take an unmatched message into the case", async () => {
    await opened();
    await screen.findByText(copy.agent.mailUnmatchedHeading);
    const block = screen
      .getByText(copy.agent.mailUnmatchedHeading)
      .closest("div") as HTMLElement;
    expect(within(block).queryByRole("button")).not.toBeInTheDocument();
  });

  it("says an ingested reply has not been applied to the case", async () => {
    await opened();
    await screen.findByText(/RE: Onboarding information request/);
    expect(screen.getByText(copy.agent.mailNotApplied)).toBeInTheDocument();
  });

  it("takes a reply in and reports what it did", async () => {
    const { user, onIngested } = await opened();
    await screen.findByText(/RE: Onboarding information request/);
    await user.click(screen.getByRole("button", { name: copy.agent.mailTake }));
    expect(await screen.findByText(copy.agent.mailAlready)).toBeInTheDocument();
    expect(onIngested).toHaveBeenCalled();
  });

  it("registers the client's file on the case", async () => {
    const { user, client, caseRef } = await opened();
    await screen.findByText(/RE: Onboarding information request/);
    await user.click(screen.getByRole("button", { name: copy.agent.mailTake }));
    await screen.findByText(copy.agent.mailAlready);
    const status = await client.getAgentCase(caseRef);
    const files = status.run.received_artefacts.map((a) => a.source_file);
    expect(files).toContain("funded_book.csv");
  });

  it("records the client's words as a client turn, not as Trakt's", async () => {
    const { user, client, caseRef } = await opened();
    await screen.findByText(/RE: Onboarding information request/);
    await user.click(screen.getByRole("button", { name: copy.agent.mailTake }));
    await screen.findByText(copy.agent.mailAlready);
    const status = await client.getAgentCase(caseRef);
    const turns = status.run.messages.filter((m) => m.role === "client");
    expect(turns).toHaveLength(1);
    expect(turns[0].text).toContain("5493001KJTIIGC8Y1R12");
    expect(turns[0].author).toBeTruthy();
  });

  it("changes no answer on the case", async () => {
    const { user, client, caseRef } = await opened();
    const before = JSON.stringify((await client.getAgentCase(caseRef)).facts);
    await screen.findByText(/RE: Onboarding information request/);
    await user.click(screen.getByRole("button", { name: copy.agent.mailTake }));
    await screen.findByText(copy.agent.mailAlready);
    const after = JSON.stringify((await client.getAgentCase(caseRef)).facts);
    expect(after).toEqual(before);
  });

  it("says nothing has arrived when the pack has not gone out", async () => {
    await opened({ issue: false });
    expect(await screen.findByText(copy.agent.mailNone)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: copy.agent.mailTake }))
      .not.toBeInTheDocument();
  });

  it("does not offer to add a file to a case that is past taking one", async () => {
    await opened({ canRegister: false });
    await screen.findByText(/RE: Onboarding information request/);
    // Offering it would produce a refusal on click, which reads as a broken
    // button rather than as a rule being enforced.
    expect(screen.queryByRole("button", { name: copy.agent.mailTake }))
      .not.toBeInTheDocument();
    expect(screen.getByText(copy.agent.mailCannotRegister)).toBeInTheDocument();
  });

  it("names a mailbox that cannot be read rather than reporting it empty", async () => {
    const { user, client } = await panel();
    vi.spyOn(client, "getAgentMail").mockRejectedValue(
      new Error("The mailbox could not be read."),
    );
    await user.click(screen.getByRole("button", { name: copy.agent.mailShow }));
    expect(await screen.findByText(/could not be read/)).toBeInTheDocument();
    expect(screen.queryByText(copy.agent.mailNone)).not.toBeInTheDocument();
  });
});
