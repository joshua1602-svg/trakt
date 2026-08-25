import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
import { MockOpsClient } from "@/api/MockOpsClient";
import { OpsError } from "@/api/OpsClient";
import { copy } from "@/lib/copy";

/**
 * The OCC Agent tab, tested through the REAL application root.
 *
 * These render `App`, not the screens in isolation, because the thing most
 * worth proving is where the tab lives: inside the existing OCC, alongside the
 * existing navigation, behind a flag — not as a second application. A test that
 * mounted `AgentCasesScreen` directly would pass even if the tab were never
 * wired into the shell at all.
 */

function renderApp(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <App />
    </MemoryRouter>,
  );
}

/** Every live nav entry, which must survive the new tab unchanged. */
const LIVE_NAV = [
  copy.nav.home,
  copy.nav.review,
  copy.nav.workflows,
  copy.nav.rules,
  copy.nav.history,
];

describe("OCC Agent tab — integration with the existing OCC", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("appears in the existing navigation when the flag is on", async () => {
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
    renderApp("/");
    expect((await screen.findAllByText(copy.nav.agent)).length).toBeGreaterThan(0);
  });

  it("is hidden when the flag is off", async () => {
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "");
    renderApp("/");
    // Wait for the shell to settle before asserting an absence.
    await screen.findByText(copy.home.subtitle);
    expect(screen.queryByText(copy.nav.agent)).not.toBeInTheDocument();
  });

  it("is hidden for an unrecognised flag value (fails closed)", async () => {
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "maybe");
    renderApp("/");
    await screen.findByText(copy.home.subtitle);
    expect(screen.queryByText(copy.nav.agent)).not.toBeInTheDocument();
  });

  it("leaves the existing navigation intact", async () => {
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
    renderApp("/");
    for (const label of LIVE_NAV) {
      expect(
        (await screen.findAllByText(label)).length,
        `${label} disappeared from the navigation`,
      ).toBeGreaterThan(0);
    }
  });

  it("leaves the existing screens working", async () => {
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
    const view = renderApp("/workflows");
    expect(await screen.findByText(copy.workflows.subtitle)).toBeInTheDocument();
    view.unmount();

    renderApp("/rules");
    expect(await screen.findByText(copy.rules.subtitle)).toBeInTheDocument();
  });

  it("refuses the route itself when the flag is off", async () => {
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "");
    renderApp("/agent");
    expect(await screen.findByText(copy.agent.disabled)).toBeInTheDocument();
  });
});

describe("OCC Agent tab — case navigation", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("says plainly that everything here is a practice case", async () => {
    renderApp("/agent");
    expect(await screen.findByText(copy.agent.syntheticBanner)).toBeInTheDocument();
  });

  it("offers the prepared scenarios with their expected outcome", async () => {
    renderApp("/agent");
    expect(await screen.findByText("A — Clean onboarding")).toBeInTheDocument();
    expect(screen.getByTestId("expected-scenario_a_clean")).toHaveTextContent(
      "READY_FOR_EXECUTION",
    );
    expect(screen.getByTestId("expected-scenario_c_missing_artefact")).toHaveTextContent(
      "BLOCKED",
    );
  });

  it("creates a case from a natural-language instruction", async () => {
    const user = userEvent.setup();
    renderApp("/agent");
    const box = await screen.findByLabelText(copy.agent.newCaseHeading);
    await user.type(
      box,
      "Onboard Northstar Lending. UK equity release. Monthly management information. Portfolio id direct_101.",
    );
    await user.click(screen.getByRole("button", { name: copy.agent.createButton }));
    expect(await screen.findByText(copy.agent.conversationHeading)).toBeInTheDocument();
    // What Trakt now holds is shown back, in the catalogue's own labels.
    expect(await screen.findByText(/Client name: Northstar Lending/)).toBeInTheDocument();
  });

  it("opens a REAL onboarding case, and says what the client still owes", async () => {
    const user = userEvent.setup();
    renderApp("/agent");
    const box = await screen.findByLabelText(copy.agent.newCaseHeading);
    await user.type(
      box,
      "Onboard Northstar Lending. UK equity release. Monthly management information. Portfolio id direct_101.",
    );
    await user.click(screen.getByRole("button", { name: copy.agent.createButton }));
    await screen.findByText(copy.agent.conversationHeading);

    // A Client Onboarding reference, not an identifier this feature invented.
    expect((await screen.findAllByText(/ONB-\d{4}-\d{4}/)).length).toBeGreaterThan(0);
    // And Client Onboarding's own outstanding-for-client list, in the one
    // panel that both lists what the client owes and lets it be answered.
    const panel = (await screen.findByText(copy.agent.questionsHeading)).closest("section");
    expect(
      within(panel as HTMLElement).getByText(/Legal Entity Identifier/),
    ).toBeInTheDocument();
  });

  it("filters cases by status", async () => {
    const user = userEvent.setup();
    renderApp("/agent");
    const card = (await screen.findByText("A — Clean onboarding")).closest("li");
    await user.click(
      within(card as HTMLElement).getByRole("button", { name: copy.agent.scenarioRun }),
    );
    await screen.findByText(copy.agent.conversationHeading);

    await user.click(screen.getByRole("link", { name: new RegExp(copy.agent.casesHeading) }));
    await screen.findByText(copy.agent.casesHeading);
    await user.click(screen.getByRole("button", { name: copy.agent.filterReady }));
    expect(await screen.findByText("Northstar Lending")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: copy.agent.filterBlocked }));
    await waitFor(() => expect(screen.getByText(copy.agent.caseEmpty)).toBeInTheDocument());
  });
});

describe("OCC Agent tab — the operating loop", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  async function runScenario(label: string) {
    const user = userEvent.setup();
    renderApp("/agent");
    const card = (await screen.findByText(label)).closest("li");
    expect(card).not.toBeNull();
    await user.click(within(card as HTMLElement).getByRole("button"));
    await screen.findByText(copy.agent.conversationHeading);
    return user;
  }

  it("a clean case reaches READY_FOR_EXECUTION and states what did not happen", async () => {
    await runScenario("A — Clean onboarding");
    // The headline appears on the banner and again on the readiness panel.
    expect((await screen.findAllByText(copy.agent.readyHeadline)).length).toBeGreaterThan(0);
    for (const line of copy.agent.readyNotDone) {
      expect(screen.getByText(line)).toBeInTheDocument();
    }
    // The status word itself: on the stage chip and on the readiness field.
    expect(screen.getAllByText(copy.agent.readyStatus).length).toBeGreaterThan(0);
    // Never the forbidden vocabulary.
    expect(screen.queryByText(/production successful/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Published$/)).not.toBeInTheDocument();
  });

  it("an incomplete case stays blocked and says what is missing", async () => {
    await runScenario("C — Missing mandatory artefact");
    expect(await screen.findByText(copy.agent.blockersHeading)).toBeInTheDocument();
    // Named in the blocker AND on the stream summary's required-file row.
    expect(screen.getAllByText(/Primary loan tape/).length).toBeGreaterThan(0);
    expect(screen.queryByText(copy.agent.readyHeadline)).not.toBeInTheDocument();
  });

  it("an ambiguous mapping shows a decision card with its evidence", async () => {
    await runScenario("B — Ambiguous mapping");
    expect(await screen.findByText(copy.agent.decisionsHeading)).toBeInTheDocument();
    expect(screen.getByText(/Confirm where 'Current Balance' belongs/)).toBeInTheDocument();
    expect(screen.getByText(copy.agent.decisionRecommendation)).toBeInTheDocument();
    expect(screen.getByText(copy.agent.decisionMateriality)).toBeInTheDocument();
    expect(screen.getByText(copy.agent.decisionConsequence)).toBeInTheDocument();
    expect(screen.getByText(/carries values for 24 of 24 records/)).toBeInTheDocument();
  });

  it("a material business-rule failure cannot be talked past in the conversation", async () => {
    const user = await runScenario("E — Material business-rule failure");
    expect(await screen.findByText(/materiality BLOCKING/)).toBeInTheDocument();

    const box = screen.getByLabelText(copy.agent.conversationHeading);
    await user.type(box, "approve readiness{Enter}");
    // The state machine refuses it; the case does not become ready.
    await waitFor(() =>
      expect(screen.queryByText(copy.agent.readyHeadline)).not.toBeInTheDocument(),
    );
  });

  it("shows a material change as a proposal before applying it", async () => {
    const user = userEvent.setup();
    renderApp("/agent");
    const box = await screen.findByLabelText(copy.agent.newCaseHeading);
    await user.type(box, "Onboard Northstar Lending. Monthly management information.");
    await user.click(screen.getByRole("button", { name: copy.agent.createButton }));
    await screen.findByText(copy.agent.conversationHeading);

    const message = screen.getByLabelText(copy.agent.conversationHeading);
    await user.type(message, "cancel this case{Enter}");
    expect(await screen.findByText(copy.agent.proposalHeading)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: copy.agent.proposalConfirm }));
    await waitFor(() =>
      expect(screen.queryByText(copy.agent.proposalHeading)).not.toBeInTheDocument(),
    );
  });

  it("links out to the existing OCC views rather than reproducing them", async () => {
    await runScenario("A — Clean onboarding");
    const panel = (await screen.findByText(copy.agent.occLinksHeading)).closest("section");
    expect(panel).not.toBeNull();
    expect(
      within(panel as HTMLElement).getByRole("link", { name: /Platform configuration/ }),
    ).toHaveAttribute("href", "/admin/config");
    // The tab links to the approvals screen; it does not rebuild it.
    expect(within(panel as HTMLElement).getByRole("link", { name: /Rules/ })).toHaveAttribute(
      "href",
      "/rules",
    );
  });

  it("never says there is nothing to do while a blocked case still has options", async () => {
    // A blocked case allows corrections and a return to an earlier stage —
    // none of which is a no-argument button, so the panel must point at the
    // conversation rather than claim the case is finished.
    await runScenario("E — Material business-rule failure");
    const panel = (await screen.findByText(copy.agent.actionsHeading)).closest("section");
    expect(panel).not.toBeNull();
    expect(
      within(panel as HTMLElement).getByText(copy.agent.actionsInConversation),
    ).toBeInTheDocument();
    expect(
      within(panel as HTMLElement).queryByText(copy.agent.actionsNone),
    ).not.toBeInTheDocument();
  });

  it("a rehearsal that has passed still has one thing left to decide", async () => {
    // READY_FOR_EXECUTION is a waypoint, not a finish: what remains is the
    // human decision about the real thing, and the tab must offer it rather
    // than declaring the case done.
    await runScenario("A — Clean onboarding");
    const panel = (await screen.findByText(copy.agent.activationHeading)).closest("section");
    expect(panel).not.toBeNull();
    expect(
      within(panel as HTMLElement).getByRole("button", { name: copy.agent.activationConfirm }),
    ).toBeInTheDocument();
  });

  it("shows the intended storage location and says it was not written", async () => {
    await runScenario("A — Clean onboarding");
    expect(await screen.findByText(copy.agent.artefactsHeading)).toBeInTheDocument();
    expect(screen.getAllByText(copy.agent.artefactNotWritten).length).toBeGreaterThan(0);
  });

  it("shows the configuration it would have created, and says it did not", async () => {
    await runScenario("A — Clean onboarding");
    const panel = (await screen.findByText(copy.agent.previewHeading)).closest("section");
    expect(panel).not.toBeNull();
    expect(
      within(panel as HTMLElement).getByText(copy.agent.previewNothingWritten),
    ).toBeInTheDocument();
    // There IS a configuration to show — the preview is real, not a placeholder.
    await waitFor(() =>
      expect(
        within(panel as HTMLElement).queryByText(copy.agent.previewNone),
      ).not.toBeInTheDocument(),
    );
  });

  it("shows both lifecycles, and states that activation is refused here", async () => {
    await runScenario("A — Clean onboarding");
    const panel = (await screen.findByText(copy.agent.statusHeading)).closest("section");
    expect(panel).not.toBeNull();
    // The onboarding reached approved; the practice run reached readiness.
    expect(within(panel as HTMLElement).getByText("Approved")).toBeInTheDocument();
    expect(
      within(panel as HTMLElement).getAllByText(copy.agent.readyStatus).length,
    ).toBeGreaterThan(0);
    // Activation IS offered — and the tab says outright that it is refused
    // here, and why, rather than hiding the control and explaining nothing.
    expect(await screen.findByText(copy.agent.activationRefusedHeading)).toBeInTheDocument();
    expect(screen.getByText(copy.agent.activationDisabled)).toBeInTheDocument();
    expect(screen.getByText(/rehearsal mode/i)).toBeInTheDocument();
  });

  it("confirming activation is refused, and says why", async () => {
    const user = userEvent.setup();
    await runScenario("A — Clean onboarding");
    const input = await screen.findByLabelText(copy.agent.activationConfirmLabel);
    await user.type(input, "go ahead");
    await user.click(screen.getByRole("button", { name: copy.agent.activationConfirm }));
    expect(await screen.findByText(/rehearsal mode/i)).toBeInTheDocument();
  });

  it("approving the configuration starts nothing", async () => {
    // Asserted against the client rather than the screen: the lifecycle panel
    // NAMES every state including "Ingestion started", so its presence as text
    // says nothing about where the case actually is.
    const client = new MockOpsClient();
    const status = await client.runAgentScenario("scenario_a_clean");
    expect(
      status.run.approvals.some((a) => (a as { subject?: string }).subject === "configuration"),
    ).toBe(true);
    expect(status.run.state).toBe("ACTIVATION_CONFIRMATION_REQUIRED");
    expect(status.run.activation_result).toEqual({});
    expect(status.configuration_written).toBe(false);
    expect(status.onboarding.status).toBe("approved");
  });

  it("the pack is drafted, approved and issued — and never claims it was sent", async () => {
    await runScenario("A — Clean onboarding");
    const panel = (await screen.findByText(copy.agent.packHeading)).closest("section");
    expect(panel).not.toBeNull();
    expect(within(panel as HTMLElement).getByText(copy.agent.packNotSent)).toBeInTheDocument();
    // And it states the governed decision about mappings to the client.
    expect(
      within(panel as HTMLElement).getByText(/does not ask you to map/i),
    ).toBeInTheDocument();
  });

  it("the mailbox is reachable from the case, and reads nothing on its own", async () => {
    await runScenario("A — Clean onboarding");
    const panel = (await screen.findByText(copy.agent.mailHeading)).closest("section");
    expect(panel).not.toBeNull();
    // Present, and inert: opening the case must not reach a mailbox.
    expect(
      within(panel as HTMLElement).getByText(copy.agent.mailClosed),
    ).toBeInTheDocument();
    expect(
      within(panel as HTMLElement).queryByRole("button", { name: copy.agent.mailTake }),
    ).not.toBeInTheDocument();
  });

  it("the client is only asked what the client can answer", async () => {
    const client = new MockOpsClient();
    const created = await client.runAgentScenario("scenario_a_clean");
    const classification = await client.getAgentClassification(created.case_ref);
    const form = await client.getAgentClientForm(created.case_ref);

    // Only category 2 becomes a question.
    const asked = new Set(
      form.form.steps.flatMap((s) => s.groups.flatMap((g) => g.fields.map((f) => f.key))),
    );
    for (const field of classification.fields) {
      const key =
        field.index === null
          ? `${field.section}.${field.field}`
          : `${field.section}[${field.index}].${field.field}`;
      if (asked.has(key)) expect(field.category).toBe("client");
    }
    // And the form is materially smaller than the catalogue.
    expect(form.form.questions).toBeLessThan(classification.summary.total);
    expect(form.form.required).toBeLessThan(form.form.questions);
  });

  it("nothing the Onboarding Agent learns is asked of the client", async () => {
    const client = new MockOpsClient();
    const created = await client.runAgentScenario("scenario_a_clean");
    const form = await client.getAgentClientForm(created.case_ref);
    const asked = new Set(
      form.form.steps.flatMap((s) => s.groups.flatMap((g) => g.fields.map((f) => f.key))),
    );
    for (const deferred of [
      "sources[0].file_format",
      "sources[0].expected_files",
      "client.client_id",
    ]) {
      expect(asked.has(deferred)).toBe(false);
    }
    expect(form.form.locked.some((l) => l.step === "data")).toBe(true);
  });

  it("a structured answer is persisted verbatim, and a bad key is refused", async () => {
    const client = new MockOpsClient();
    const created = await client.createAgentCase(
      "Onboard Northstar Lending. UK equity release. Monthly portfolio MI. " +
        "Portfolio id direct_101.",
    );
    const after = await client.submitAgentClientForm(created.case_ref, {
      "contacts.reporting_contact_name": "Dana Fox",
    });
    expect(
      (after.onboarding.answers.contacts as Record<string, string>).reporting_contact_name,
    ).toBe("Dana Fox");

    await expect(
      client.submitAgentClientForm(created.case_ref, {
        "contacts.favourite_colour": "blue",
      }),
    ).rejects.toMatchObject({ errorCode: "OCC_AGENT_UNKNOWN_ANSWER_KEY" });

    await expect(
      client.submitAgentClientForm(created.case_ref, { "client.client_id": "HIJACKED" }),
    ).rejects.toMatchObject({ errorCode: "OCC_AGENT_NOT_A_CLIENT_QUESTION" });
  });

  it("every pack question is a field the governed catalogue declares", async () => {
    const client = new MockOpsClient();
    const created = await client.runAgentScenario("scenario_a_clean");
    const pack = await client.getAgentPack(created.case_ref);
    const reference = (await client.getAgentMeta()).onboarding_reference;
    const declared = new Set<string>();
    for (const section of reference.catalogue.sections ?? []) {
      for (const field of section.fields ?? []) declared.add(`${section.key}.${field.key}`);
    }
    const questions = pack.pack.sections.flatMap((s) => s.questions);
    expect(questions.length).toBeGreaterThan(0);
    for (const question of questions) {
      expect(declared.has(`${question.section}.${question.field}`)).toBe(true);
    }
  });

  it("the review package says mappings are learned at first ingestion", async () => {
    const client = new MockOpsClient();
    const created = await client.runAgentScenario("scenario_a_clean");
    const review = await client.getAgentReview(created.case_ref);
    expect(review.document).toMatch(/were not collected/i);
    expect(review.document).toMatch(/first representative delivery/i);
  });

  it("links to the onboarding case in the screens it normally lives in", async () => {
    await runScenario("A — Clean onboarding");
    const panel = (await screen.findByText(copy.agent.occLinksHeading)).closest("section");
    const link = within(panel as HTMLElement).getByRole("link", {
      name: new RegExp(copy.nav.onboarding),
    });
    expect(link.getAttribute("href")).toMatch(/^\/onboarding\/ONB-\d{4}-\d{4}$/);
  });
});

describe("OCC Agent tab — confirming a proposal", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  /**
   * Confirming used to re-send the PROPOSAL'S SUMMARY rather than the
   * operator's own words. The server re-reads the message to decide what to
   * apply, and a summary is prose ABOUT a change, not an instruction to make
   * one — so "Answer the onboarding." came back as "Trakt could not tell what
   * to do with that", with the proposal still on screen describing exactly
   * what it could no longer do.
   *
   * The mock's own summaries happen to contain the verbs its router matches on
   * ("Submit the onboarding for approval." still reads as "submit"), which is
   * why the defect survived: the mock never exercised the failing case. So
   * this asserts what the COMPONENT sends, which is where the defect lived.
   */
  it("re-sends the operator's own words, never the proposal summary", async () => {
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
    const sent = vi.spyOn(MockOpsClient.prototype, "instructAgent");
    const user = userEvent.setup();
    renderApp("/agent");

    const box = await screen.findByLabelText(copy.agent.newCaseHeading);
    await user.type(box, "Onboard Northstar Lending. Monthly management information.");
    await user.click(screen.getByRole("button", { name: copy.agent.createButton }));
    await screen.findByText(copy.agent.conversationHeading);

    const said = "submit for approval";
    const message = screen.getByLabelText(copy.agent.conversationHeading);
    await user.type(message, `${said}{Enter}`);

    const confirmButton = await screen.findByRole("button", {
      name: copy.agent.proposalConfirm,
    });
    await user.click(confirmButton);

    await waitFor(() => expect(sent.mock.calls.length).toBeGreaterThanOrEqual(2));
    const [, proposeText, proposeConfirm] = sent.mock.calls[0];
    const [, confirmText, confirmConfirm] = sent.mock.calls[1];
    expect(proposeConfirm).toBeFalsy();
    expect(proposeText).toBe(said);
    expect(confirmConfirm).toBe(true);
    // The whole point: the same words, not a restatement of them.
    expect(confirmText).toBe(said);
  });
});

describe("OCC Agent tab — the client questions", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  /**
   * An operator could see that a pack contained "22 questions" and could not
   * read them, could not change one, and could not correct a value Trakt had
   * already decided. Every capability existed and was tested; no screen
   * rendered any of them.
   */
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

  async function openCase() {
    const user = await createCase();
    await issuePack(user);
    return user;
  }

  /**
   * Walk the case to the stage where client answers are recorded.
   *
   * This is not scaffolding around the panel — it is the workflow the panel
   * now belongs to. Recording a client's answers comes AFTER the request has
   * gone out, so a test that reaches the form without issuing the pack would
   * be asserting an order the screen no longer offers.
   */
  async function issuePack(user: ReturnType<typeof userEvent.setup>) {
    await user.click(await screen.findByRole("button", { name: copy.agent.packDraft }));
    await user.click(await screen.findByRole("button", { name: copy.agent.packApprove }));
    const to = await screen.findByLabelText(copy.agent.packRecipients);
    await user.type(to, "ops@northstar.example");
    await user.click(screen.getByRole("button", { name: copy.agent.packSend }));
    await waitFor(() =>
      expect(
        document.querySelector('[data-stage="responses"][data-stage-status="current"]'),
      ).not.toBeNull(),
    );
  }

  it("lets an operator read every question the client is asked", async () => {
    const user = await openCase();

    // Closed by default: the case page is a workflow, not a form.
    expect(screen.queryByText(copy.agent.questionsRequired)).not.toBeInTheDocument();

    await user.click(await screen.findByRole("button", { name: copy.agent.questionsShow }));

    // Real catalogue questions, with their obligation shown.
    expect(await screen.findByLabelText(/Reporting email/)).toBeInTheDocument();
    expect(screen.getAllByText(copy.agent.questionsRequired).length).toBeGreaterThan(0);
    expect(screen.getAllByText(copy.agent.questionsOptional).length).toBeGreaterThan(0);
  });

  it("records an answer against the authoritative catalogue key", async () => {
    const submitted = vi.spyOn(MockOpsClient.prototype, "submitAgentClientForm");
    const user = await openCase();
    await user.click(await screen.findByRole("button", { name: copy.agent.questionsShow }));

    const email = await screen.findByLabelText(/Reporting email/);
    await user.type(email, "dana@northstar.example");
    await user.click(screen.getByRole("button", { name: copy.agent.questionsSave }));

    await waitFor(() => expect(submitted).toHaveBeenCalled());
    const [, answers] = submitted.mock.calls[0];
    expect(answers).toEqual({
      "contacts.reporting_contact_email": "dana@northstar.example",
    });
  });

  it("saves nothing until something is changed", async () => {
    const submitted = vi.spyOn(MockOpsClient.prototype, "submitAgentClientForm");
    const user = await openCase();
    await user.click(await screen.findByRole("button", { name: copy.agent.questionsShow }));
    await screen.findByLabelText(/Reporting email/);   // the form has loaded

    const save = screen.getByRole("button", { name: copy.agent.questionsSave });
    expect(save).toBeDisabled();
    await user.click(save);
    expect(submitted).not.toHaveBeenCalled();
  });

  /**
   * Where recording an answer lives, pinned.
   *
   * It sat inside a timeline stage, and the predictable thing happened: the
   * stage completed, collapsed, and the only place to record an answer went
   * with it. Onboarding is not a linear pass — answers arrive by email, by
   * phone, as a correction three stages later — so this is a reference view of
   * the case, reachable at every stage, not a step an operator walks past.
   */
  it("can record an answer at any stage, before the pack and after it", async () => {
    const user = await createCase();
    expect(await screen.findByRole("button", { name: copy.agent.questionsShow }))
      .toBeInTheDocument();

    await issuePack(user);
    expect(screen.getByRole("button", { name: copy.agent.questionsShow }))
      .toBeInTheDocument();
    // And it is NOT inside the stage that completes and collapses.
    const stage = document.querySelector('[data-stage="responses"]') as HTMLElement;
    expect(stage).not.toBeNull();
    expect(within(stage).queryByRole("button", { name: copy.agent.questionsShow }))
      .not.toBeInTheDocument();
  });

  /** The timeline keeps the two things that ARE steps: receive, and chase. */
  it("keeps receiving and chasing on the timeline", async () => {
    const user = await createCase();
    await issuePack(user);
    const stage = document.querySelector('[data-stage="responses"]') as HTMLElement;
    expect(within(stage).getByText(copy.agent.mailHeading)).toBeInTheDocument();
    expect(within(stage).getByRole("button", { name: copy.agent.checklistAsk }))
      .toBeInTheDocument();
  });

  it("surfaces a refused answer instead of swallowing it", async () => {
    vi.spyOn(MockOpsClient.prototype, "submitAgentClientForm").mockRejectedValue(
      new OpsError("That is not a question Trakt puts to a client.", "OCC_AGENT_X"),
    );
    const user = await openCase();
    await user.click(await screen.findByRole("button", { name: copy.agent.questionsShow }));

    const email = await screen.findByLabelText(/Reporting email/);
    await user.type(email, "dana@northstar.example");
    await user.click(screen.getByRole("button", { name: copy.agent.questionsSave }));

    expect(
      await screen.findByText(/not a question Trakt puts to a client/),
    ).toBeInTheDocument();
  });
});
