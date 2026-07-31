import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
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
    // And Client Onboarding's own outstanding-for-client list.
    const panel = (await screen.findByText(copy.agent.checklistHeading)).closest("section");
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
    expect(screen.getByText(/Primary loan tape/)).toBeInTheDocument();
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

  it("says a finished case is finished", async () => {
    await runScenario("A — Clean onboarding");
    const panel = (await screen.findByText(copy.agent.actionsHeading)).closest("section");
    expect(within(panel as HTMLElement).getByText(copy.agent.actionsNone)).toBeInTheDocument();
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

  it("shows both lifecycles, and never offers to activate", async () => {
    await runScenario("A — Clean onboarding");
    const panel = (await screen.findByText(copy.agent.statusHeading)).closest("section");
    expect(panel).not.toBeNull();
    // The onboarding reached approved; the practice run reached readiness.
    expect(within(panel as HTMLElement).getByText("Approved")).toBeInTheDocument();
    expect(
      within(panel as HTMLElement).getAllByText(copy.agent.readyStatus).length,
    ).toBeGreaterThan(0);
    // Nothing anywhere offers activation.
    expect(screen.queryByRole("button", { name: /activate/i })).not.toBeInTheDocument();
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
