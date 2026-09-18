import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
import { MockOpsClient } from "@/api/MockOpsClient";
import { copy } from "@/lib/copy";

/**
 * Recording the concentration-test decision, and seeing an answer that saved.
 *
 * WHAT HAPPENED TO AN OPERATOR
 *
 * They pasted a client's concentration limits into the box on the case,
 * pressed save, and the screen looked unchanged. The server had returned 200.
 * There was no way to tell by looking whether it had landed, because an
 * answered client question left the form — so a saved answer and an unsaved
 * one looked identical, and neither could be edited.
 *
 * Separately, the decision itself was split in two. `record_concentration_
 * outcome` takes the status, the client's wording and the reason together, and
 * refuses a blank answer recorded as supplied. No screen called it. So the
 * status went in through the conversation — which truncates prose at the first
 * clause — and the limits through the client form, with nothing checking that
 * the two agreed.
 *
 * These are integration tests because every layer underneath already worked.
 * What is asserted is what a person can see and do.
 */

function renderApp(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <App />
    </MemoryRouter>,
  );
}

async function openCase() {
  const user = userEvent.setup();
  renderApp("/agent");
  const box = await screen.findByLabelText(copy.agent.newCaseHeading);
  await user.type(
    box,
    "Onboard Northstar Lending. UK equity release. Monthly management information. " +
      "Portfolio id direct_101.",
  );
  await user.click(screen.getByRole("button", { name: copy.agent.createButton }));
  await screen.findByText(copy.agent.conversationHeading);
  await screen.findByText(copy.agent.concentrationHeading);
  return user;
}

function panel(): HTMLElement {
  return screen.getByText(copy.agent.concentrationHeading).closest("section") as HTMLElement;
}

const LIMITS =
  "Maximum 10% of the portfolio to any one postcode district.\n" +
  "Maximum LTV 55% at origination.\n" +
  "No more than 5% of loans above GBP 750,000.";

describe("OCC Agent — the concentration decision is one act", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("offers the status, the limits and the reason together", async () => {
    await openCase();
    expect(within(panel()).getByLabelText(copy.agent.concentrationStatus)).toBeInTheDocument();
    expect(within(panel()).getByLabelText(copy.agent.concentrationText)).toBeInTheDocument();
    expect(within(panel()).getByLabelText(copy.agent.concentrationReason)).toBeInTheDocument();
  });

  it("sends the whole decision in one call", async () => {
    const record = vi.spyOn(MockOpsClient.prototype, "recordAgentConcentration");
    const user = await openCase();
    await user.selectOptions(
      within(panel()).getByLabelText(copy.agent.concentrationStatus),
      "supplied",
    );
    await user.type(within(panel()).getByLabelText(copy.agent.concentrationText), LIMITS);
    await user.click(within(panel()).getByRole("button", { name: copy.agent.concentrationSave }));
    await waitFor(() => expect(record).toHaveBeenCalledTimes(1));
    expect(record.mock.calls[0][1]).toMatchObject({ status: "supplied", response_text: LIMITS });
  });

  it("keeps every line of what was pasted", async () => {
    /* The conversation truncates prose at the first clause. This must not. */
    const record = vi.spyOn(MockOpsClient.prototype, "recordAgentConcentration");
    const user = await openCase();
    await user.selectOptions(
      within(panel()).getByLabelText(copy.agent.concentrationStatus),
      "supplied",
    );
    await user.type(within(panel()).getByLabelText(copy.agent.concentrationText), LIMITS);
    await user.click(within(panel()).getByRole("button", { name: copy.agent.concentrationSave }));
    await waitFor(() => expect(record).toHaveBeenCalled());
    const sent = record.mock.calls[0][1].response_text ?? "";
    expect(sent.split("\n")).toHaveLength(3);
    expect(sent).toContain("750,000");
  });

  it("says why it will not record a blank answer as supplied", async () => {
    /* The server refuses it. The screen says so before the button is pressed
       rather than after it is rejected. */
    const user = await openCase();
    await user.selectOptions(
      within(panel()).getByLabelText(copy.agent.concentrationStatus),
      "supplied",
    );
    expect(within(panel()).getByText(copy.agent.concentrationNeedsText)).toBeInTheDocument();
    expect(
      within(panel()).getByRole("button", { name: copy.agent.concentrationSave }),
    ).toBeDisabled();
  });

  it("asks for a reason before deferring", async () => {
    const user = await openCase();
    await user.selectOptions(
      within(panel()).getByLabelText(copy.agent.concentrationStatus),
      "deferred_with_reason",
    );
    expect(within(panel()).getByText(copy.agent.concentrationNeedsReason)).toBeInTheDocument();
    expect(
      within(panel()).getByRole("button", { name: copy.agent.concentrationSave }),
    ).toBeDisabled();
  });

  it("records a deferral once the reason is given", async () => {
    const record = vi.spyOn(MockOpsClient.prototype, "recordAgentConcentration");
    const user = await openCase();
    await user.selectOptions(
      within(panel()).getByLabelText(copy.agent.concentrationStatus),
      "deferred_with_reason",
    );
    await user.type(
      within(panel()).getByLabelText(copy.agent.concentrationReason),
      "The client has not yet provided its limits.",
    );
    await user.click(within(panel()).getByRole("button", { name: copy.agent.concentrationSave }));
    await waitFor(() => expect(record).toHaveBeenCalled());
    expect(record.mock.calls[0][1]).toMatchObject({
      status: "deferred_with_reason",
      reason: "The client has not yet provided its limits.",
    });
  });

  it("shows what was recorded when the case is reopened", async () => {
    /* The whole point: an operator can see what they saved. */
    const user = await openCase();
    await user.selectOptions(
      within(panel()).getByLabelText(copy.agent.concentrationStatus),
      "supplied",
    );
    await user.type(within(panel()).getByLabelText(copy.agent.concentrationText), LIMITS);
    await user.click(within(panel()).getByRole("button", { name: copy.agent.concentrationSave }));
    await waitFor(() =>
      expect(within(panel()).getByLabelText(copy.agent.concentrationText)).toHaveValue(LIMITS),
    );
  });
});

describe("OCC Agent — a saved answer can be seen and corrected", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  async function answerAQuestion() {
    const user = await openCase();
    await user.click(await screen.findByRole("button", { name: copy.agent.questionsShow }));
    const box = await screen.findByLabelText(/Company registration number/i);
    await user.type(box, "09876543");
    await user.click(screen.getByRole("button", { name: copy.agent.questionsSave }));
    return user;
  }

  it("keeps the answer on screen instead of making the box vanish", async () => {
    /* The defect: after saving, the question left the form entirely, so "it
       saved" and "it did not save" looked exactly the same. */
    await answerAQuestion();
    await waitFor(() =>
      expect(screen.getByLabelText(/Company registration number/i)).toHaveValue("09876543"),
    );
    expect(screen.getByText(copy.agent.questionsAnsweredHeading)).toBeInTheDocument();
  });

  it("lets a typo be corrected in the box it was typed into", async () => {
    const user = await answerAQuestion();
    await waitFor(() =>
      expect(screen.getByLabelText(/Company registration number/i)).toHaveValue("09876543"),
    );
    const box = await screen.findByLabelText(/Company registration number/i);
    // WAIT FOR THE BOX TO BE EDITABLE, not just for the value to arrive.
    //
    // These are two different moments and the gap between them is real: the
    // value lands when the reloaded status renders, and the form stays
    // disabled until the action that reloaded it settles its `busy` flag in a
    // `finally` AFTER that render. Asserting only on the value, the test
    // reached for a box the screen was still holding shut and `clear()` threw
    // "only supported on editable elements" — intermittently, because it is a
    // race, which is why it passed locally and failed in CI.
    await waitFor(() => expect(box).toBeEnabled());
    await user.clear(box);
    await user.type(box, "12345678");
    await user.click(screen.getByRole("button", { name: copy.agent.questionsSave }));
    await waitFor(() =>
      expect(screen.getByLabelText(/Company registration number/i)).toHaveValue("12345678"),
    );
  });

  /* The separation itself — an answered question sits in `answered` and NOT
     among the steps a client is asked — is asserted against the real
     implementation in
     `tests/operations_control/occ_agent/test_an_answer_can_be_seen_and_corrected.py`
     (`test_it_is_still_there_afterwards`). Asserting it here would be
     asserting the mock's own classification of one repeatable-section field,
     which is a statement about the double rather than about the platform. */
});
