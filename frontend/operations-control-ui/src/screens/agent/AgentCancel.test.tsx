import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
import { MockOpsClient } from "@/api/MockOpsClient";
import { copy } from "@/lib/copy";

/**
 * Cancelling a case from the case screen.
 *
 * WHY THIS IS AN INTEGRATION TEST AND NOT A COMPONENT ONE
 *
 * The defect was never that cancelling did not work — the API, the service and
 * the state machine all had it, and a unit test of any of them would have
 * passed. The defect was that an operator looking at the screen could not find
 * it: the only trace was a bullet reading "cancel run" in a grey list of things
 * you could "ask for in conversation". So the assertion that matters is made by
 * driving the real screen and looking for what a person would look for.
 *
 * Mounting the dialog directly would prove the dialog renders, which was never
 * in doubt, and would have gone on passing while the screen offered no way to
 * open it.
 */

function renderApp(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <App />
    </MemoryRouter>,
  );
}

async function createCase() {
  const user = userEvent.setup();
  renderApp("/agent");
  const box = await screen.findByLabelText(copy.agent.newCaseHeading);
  await user.type(
    box,
    "Onboard Northstar Lending. UK equity release. Monthly management information.",
  );
  await user.click(screen.getByRole("button", { name: copy.agent.createButton }));
  await screen.findByText(copy.agent.conversationHeading);
  return user;
}

describe("OCC Agent — ending a case", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("the case screen offers a way to cancel", async () => {
    await createCase();
    expect(
      await screen.findByRole("button", { name: copy.agent.cancelLink }),
    ).toBeInTheDocument();
  });

  it("cancelling is not offered as a thing to ask for in conversation", async () => {
    /* It used to appear there, as "cancel run", and that was the only mention
       of it anywhere on the screen. Now that it has a real control, listing it
       again would offer the same act twice — once with a required reason and
       once without. */
    await createCase();
    await screen.findByRole("button", { name: copy.agent.cancelLink });
    expect(screen.queryByText("cancel run")).not.toBeInTheDocument();
  });

  it("asks why, and will not proceed until it is told", async () => {
    const user = await createCase();
    await user.click(await screen.findByRole("button", { name: copy.agent.cancelLink }));

    await screen.findByText(copy.agent.cancelHeading);
    const confirm = screen.getByRole("button", { name: copy.agent.cancelConfirm });
    expect(confirm).toBeDisabled();

    await user.type(screen.getByLabelText(/why is this being cancelled/i), "Superseded.");
    expect(confirm).toBeEnabled();
  });

  it("the confirm button cannot be mistaken for the link that opened it", async () => {
    /* Both read "Cancel this case" at first. On an irreversible dialog that is
       a coin toss, and to a screen reader the two are identical. */
    const user = await createCase();
    await user.click(await screen.findByRole("button", { name: copy.agent.cancelLink }));
    await screen.findByText(copy.agent.cancelHeading);

    expect(copy.agent.cancelConfirm).not.toBe(copy.agent.cancelLink);
    expect(screen.getAllByRole("button", { name: copy.agent.cancelLink })).toHaveLength(1);
    expect(screen.getAllByRole("button", { name: copy.agent.cancelConfirm })).toHaveLength(1);
  });

  it("can be backed out of without ending anything", async () => {
    const user = await createCase();
    await user.click(await screen.findByRole("button", { name: copy.agent.cancelLink }));
    await screen.findByText(copy.agent.cancelHeading);

    await user.click(screen.getByRole("button", { name: copy.agent.cancelKeep }));

    await waitFor(() =>
      expect(screen.queryByText(copy.agent.cancelHeading)).not.toBeInTheDocument(),
    );
    // Still offered, because nothing ended.
    expect(screen.getByRole("button", { name: copy.agent.cancelLink })).toBeInTheDocument();
  });

  it("sends the operator's reason, verbatim", async () => {
    const said = "Superseded by a fresh onboarding so elapsed time is measurable.";
    const step = vi.spyOn(MockOpsClient.prototype, "runAgentStep");

    const user = await createCase();
    await user.click(await screen.findByRole("button", { name: copy.agent.cancelLink }));
    await screen.findByText(copy.agent.cancelHeading);
    await user.type(screen.getByLabelText(/why is this being cancelled/i), said);
    await user.click(screen.getByRole("button", { name: copy.agent.cancelConfirm }));

    await waitFor(() => expect(step).toHaveBeenCalled());
    const call = step.mock.calls.find(([, name]) => name === "cancel");
    expect(call).toBeDefined();
    expect(call?.[2]).toEqual({ reason: said });
  });

  it("the reason reaches the withdrawn case, not just the request", async () => {
    /* The point of collecting it. A reason that stops at the API call is the
       same defect the server-side fix removed, one layer up. */
    const said = "Superseded after platform remediation.";
    const client = new MockOpsClient();
    const created = await client.createAgentCase(
      "Onboard Northstar Lending. Monthly management information.",
    );
    const ref = created.run.case_ref;

    await client.runAgentStep(ref, "cancel", { reason: said });

    const after = await client.getAgentCase(ref);
    expect(after.onboarding.status).toBe("withdrawn");
    expect(after.onboarding.withdrawal_reason).toBe(said);
  });

  it("a cancelled case no longer offers to be cancelled", async () => {
    /* Driven entirely through the screen: the app builds its own mock client,
       so a case created against a separate instance would not be there to
       open. Cancelling in the UI is also the only version of this that proves
       the screen re-reads the case after the act. */
    const user = await createCase();
    await user.click(await screen.findByRole("button", { name: copy.agent.cancelLink }));
    await screen.findByText(copy.agent.cancelHeading);
    await user.type(screen.getByLabelText(/why is this being cancelled/i), "Superseded.");
    await user.click(screen.getByRole("button", { name: copy.agent.cancelConfirm }));

    await waitFor(() =>
      expect(screen.queryByText(copy.agent.cancelHeading)).not.toBeInTheDocument(),
    );
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: copy.agent.cancelLink }),
      ).not.toBeInTheDocument(),
    );
  });
});
