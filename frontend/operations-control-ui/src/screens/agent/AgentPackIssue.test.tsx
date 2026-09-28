import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
import { MockOpsClient } from "@/api/MockOpsClient";
import { copy } from "@/lib/copy";

/**
 * Issuing the pack: why the button is unavailable, and whether it actually sent.
 *
 * TWO SILENCES, ONE SYMPTOM
 *
 * An operator approved a pack, pressed "Record it as issued", and nothing
 * happened. Both halves of that were the screen's fault, and neither was the
 * send path, which works with a typed address whatever the case holds.
 *
 * 1. The button disables itself when there is no address, in silence. The
 *    address is missing for a reason that reads as a contradiction — the
 *    reporting contact is one of the questions the pack is going out to ASK —
 *    so the operator's reasonable conclusion is that the button is broken, and
 *    their next move is to press it again rather than to fill in the box beside
 *    it. Issuing has never depended on that catalogue answer; it needed
 *    somewhere to send it.
 *
 * 2. A successful issue confirmed nothing. `act` toasts on failure and is
 *    silent on success, so "it sent" and "nothing happened" looked identical
 *    on screen — and this is the one action in the tab that emails a client.
 *    The receipt already carries the honest answer; it just was not said at
 *    the moment the operator was asking the question.
 */

function renderApp(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <App />
    </MemoryRouter>,
  );
}

/** A case walked to the point where the pack can be issued. */
async function readyToIssue() {
  const user = userEvent.setup();
  renderApp("/agent");
  const box = await screen.findByLabelText(copy.agent.newCaseHeading);
  await user.type(
    box,
    "Onboard Northstar Lending. UK equity release. Monthly management information.",
  );
  await user.click(screen.getByRole("button", { name: copy.agent.createButton }));
  await screen.findByText(copy.agent.conversationHeading);

  await user.click(await screen.findByRole("button", { name: copy.agent.packDraft }));
  await user.click(await screen.findByRole("button", { name: copy.agent.packApprove }));
  await screen.findByRole("button", { name: copy.agent.packSend });
  return user;
}

describe("OCC Agent — issuing the pack", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("says why it cannot be issued yet, instead of going quiet", async () => {
    await readyToIssue();
    expect(screen.getByRole("button", { name: copy.agent.packSend })).toBeDisabled();
    expect(screen.getByText(copy.agent.packNeedsAddress)).toBeInTheDocument();
  });

  it("the reason goes away as soon as an address is typed", async () => {
    const user = await readyToIssue();
    await user.type(
      screen.getByLabelText(copy.agent.packRecipients),
      "ops@northstar.example",
    );
    await waitFor(() =>
      expect(screen.queryByText(copy.agent.packNeedsAddress)).not.toBeInTheDocument(),
    );
    expect(screen.getByRole("button", { name: copy.agent.packSend })).toBeEnabled();
  });

  it("issues to a typed address, with no contact recorded on the case", async () => {
    /* The point of the whole panel: sending is gated by having somewhere to
       send to, never by the client having answered the contact question. */
    const send = vi.spyOn(MockOpsClient.prototype, "sendAgentPack");
    const user = await readyToIssue();
    await user.type(
      screen.getByLabelText(copy.agent.packRecipients),
      "ops@northstar.example",
    );
    await user.click(screen.getByRole("button", { name: copy.agent.packSend }));

    await waitFor(() => expect(send).toHaveBeenCalled());
    expect(send.mock.calls[0][1]).toEqual(["ops@northstar.example"]);
  });

  it("says what happened, rather than leaving the screen unchanged", async () => {
    const user = await readyToIssue();
    await user.type(
      screen.getByLabelText(copy.agent.packRecipients),
      "ops@northstar.example",
    );
    await user.click(screen.getByRole("button", { name: copy.agent.packSend }));

    // The mock has no outbound transport, so the honest answer is the one that
    // says nothing was sent. Which of the two appears is the receipt's call —
    // that either appears at all is this test's.
    await waitFor(() =>
      expect(
        screen.queryByText(copy.agent.packRecordedToast)
          ?? screen.queryByText(copy.agent.packIssuedToast),
      ).toBeInTheDocument(),
    );
  });

  it("the two outcomes are not the same sentence", () => {
    /* "It left Trakt" and "nothing was sent" must never be interchangeable:
       one reached a client and the other is a record that it did not. */
    expect(copy.agent.packIssuedToast).not.toBe(copy.agent.packRecordedToast);
    expect(copy.agent.packRecordedToast.toLowerCase()).toContain("nothing was sent");
  });
});

describe("OCC Agent — providing the files", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("can be done while the case is still waiting on the client's answers", async () => {
    /* `register_synthetic_artefact` is permitted from PACK_SENT onwards, but
       the panel lived only on the artefacts stage — which stays FUTURE, and a
       future stage renders no panel, until the responses stage completes. So
       an operator already holding the loan tape had to wait for the client to
       answer contact questions before Trakt would take it. */
    const user = await readyToIssue();
    await user.type(
      screen.getByLabelText(copy.agent.packRecipients),
      "ops@northstar.example",
    );
    await user.click(screen.getByRole("button", { name: copy.agent.packSend }));

    // Now on the responses stage, with the client yet to answer anything.
    expect(
      await screen.findByText(copy.agent.artefactsHeading),
    ).toBeInTheDocument();
  });
});
