import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
import { copy } from "@/lib/copy";

/**
 * Naming the delivery, and taking a file back out.
 *
 * TWO THINGS THE SCREEN COULD NOT DO, WITH ONE VISIBLE SYMPTOM
 *
 * A file card read "Where this would be filed: —" and nothing an operator did
 * changed it. The intended location needs the client, the portfolio and the
 * reporting period; the first two came from the onboarding answers and the
 * third had no box. `POST /cases/{ref}/target` and `setAgentRunTarget` both
 * existed — grep for the second returned its declaration in the interface, its
 * implementation in the HTTP client, its implementation in the mock, and no
 * caller anywhere.
 *
 * The other half: uploading was a one-way door. `received_artefacts` was only
 * ever appended to, so an operator who attached the wrong file — an encrypted
 * workbook, last month's tape — could re-upload over the bytes and leave the
 * case recording two files where one arrived, or cancel the case.
 *
 * These are integration tests for the same reason the cancel ones are: every
 * layer underneath already worked, and a unit test of any of them would have
 * passed while the screen offered no way in. What is asserted is what a person
 * can find and do.
 */

function renderApp(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <App />
    </MemoryRouter>,
  );
}

/**
 * A case walked to where the operator in question actually stands: the pack
 * issued, the client yet to answer, files in hand. That is the stage the
 * artefacts panel renders under, and the stage at which the missing period
 * was noticed.
 */
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

  await user.click(await screen.findByRole("button", { name: copy.agent.packDraft }));
  await user.click(await screen.findByRole("button", { name: copy.agent.packApprove }));
  await user.type(
    await screen.findByLabelText(copy.agent.packRecipients),
    "ops@northstar.example",
  );
  await user.click(screen.getByRole("button", { name: copy.agent.packSend }));
  await screen.findByText(copy.agent.artefactsHeading);
  return user;
}

async function attach(user: ReturnType<typeof userEvent.setup>, ...names: string[]) {
  const input = document.querySelector('input[type="file"]') as HTMLInputElement;
  await user.upload(
    input,
    names.map((name) => new File(["loan_id,balance\nL1,100\n"], name, { type: "text/csv" })),
  );
  for (const name of names) await screen.findByText(name);
}

function cardFor(name: string): HTMLElement {
  return screen.getByText(name).closest("li") as HTMLElement;
}

describe("OCC Agent — naming the delivery", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("offers somewhere to put the reporting period", async () => {
    await openCase();
    expect(await screen.findByLabelText(copy.agent.targetPeriodLabel)).toBeInTheDocument();
  });

  it("gives an already-uploaded file a destination once the period is named", async () => {
    const user = await openCase();
    await attach(user, "LoanExtract.csv");
    expect(within(cardFor("LoanExtract.csv")).getByText(/Where this would be filed: —/)).
      toBeInTheDocument();

    await user.type(await screen.findByLabelText(copy.agent.targetPeriodLabel), "2026-04");
    await user.click(screen.getByRole("button", { name: copy.agent.targetSave }));

    await waitFor(() =>
      expect(within(cardFor("LoanExtract.csv")).getByText(/2026-04/)).toBeInTheDocument(),
    );
  });

  it("says what is missing rather than leaving a bare dash", async () => {
    const user = await openCase();
    await attach(user, "LoanExtract.csv");
    expect(screen.getByText(copy.agent.artefactNoDestination)).toBeInTheDocument();
  });

  it("keeps the period after it is saved", async () => {
    const user = await openCase();
    await user.type(await screen.findByLabelText(copy.agent.targetPeriodLabel), "2026-04");
    await user.click(screen.getByRole("button", { name: copy.agent.targetSave }));
    await waitFor(() =>
      expect(screen.getByLabelText(copy.agent.targetPeriodLabel)).toHaveValue("2026-04"),
    );
  });

  it("will not save nothing", async () => {
    await openCase();
    await screen.findByLabelText(copy.agent.targetPeriodLabel);
    expect(screen.getByRole("button", { name: copy.agent.targetSave })).toBeDisabled();
  });
});

describe("OCC Agent — taking a file back out", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("offers a way to remove each file", async () => {
    const user = await openCase();
    await attach(user, "LoanExtract.csv");
    expect(
      within(cardFor("LoanExtract.csv")).getByRole("button", {
        name: copy.agent.artefactRemove,
      }),
    ).toBeInTheDocument();
  });

  it("removes the file it was asked to and leaves the others", async () => {
    const user = await openCase();
    await attach(user, "LoanExtract.csv", "encrypted.csv");
    await user.click(
      within(cardFor("encrypted.csv")).getByRole("button", { name: copy.agent.artefactRemove }),
    );
    await user.click(screen.getByRole("button", { name: copy.agent.artefactRemoveConfirm }));

    await waitFor(() => expect(screen.queryByText("encrypted.csv")).not.toBeInTheDocument());
    expect(screen.getByText("LoanExtract.csv")).toBeInTheDocument();
  });

  it("asks before removing, and keeping it changes nothing", async () => {
    const user = await openCase();
    await attach(user, "LoanExtract.csv");
    await user.click(
      within(cardFor("LoanExtract.csv")).getByRole("button", { name: copy.agent.artefactRemove }),
    );
    await user.click(screen.getByRole("button", { name: copy.agent.artefactRemoveKeep }));
    expect(screen.getByText("LoanExtract.csv")).toBeInTheDocument();
  });

  it("says what removing does before it is confirmed", async () => {
    /* "Remove" alone reads as "delete", and the bytes are not deleted. */
    const user = await openCase();
    await attach(user, "LoanExtract.csv");
    await user.click(
      within(cardFor("LoanExtract.csv")).getByRole("button", { name: copy.agent.artefactRemove }),
    );
    expect(screen.getByText(copy.agent.artefactRemoveExplain)).toBeInTheDocument();
  });

  it("the confirmation does not collide with the button that opened it", async () => {
    const user = await openCase();
    await attach(user, "LoanExtract.csv");
    await user.click(
      within(cardFor("LoanExtract.csv")).getByRole("button", { name: copy.agent.artefactRemove }),
    );
    expect(
      screen.getAllByRole("button", { name: copy.agent.artefactRemoveConfirm }),
    ).toHaveLength(1);
  });

  it("gets back to no files at all", async () => {
    const user = await openCase();
    await attach(user, "LoanExtract.csv");
    await user.click(
      within(cardFor("LoanExtract.csv")).getByRole("button", { name: copy.agent.artefactRemove }),
    );
    await user.click(screen.getByRole("button", { name: copy.agent.artefactRemoveConfirm }));
    await waitFor(() => expect(screen.queryByText("LoanExtract.csv")).not.toBeInTheDocument());
  });
});
