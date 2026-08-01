import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
import { MockOpsClient } from "@/api/MockOpsClient";
import { OpsClientProvider } from "@/api/context";
import { RouteErrorBoundary } from "@/components/ErrorBoundary";
import { ToastProvider } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { AgentCaseScreen } from "./AgentCase";

/**
 * The completion pass, tested through the real root:
 *
 * 1. the OCC Agent is the primary entry point when the flag is on, and the
 *    manual-first experience is preserved untouched when it is off;
 * 2. the exact Capital 123 instruction produces two separately-rendered
 *    streams with the regime rule stated;
 * 3. the chronological timeline drives the case page;
 * 4. a failed action or a rendering fault never blanks the page.
 */

function renderApp(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <App />
    </MemoryRouter>,
  );
}

describe("OCC Agent as the primary entry point", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => vi.unstubAllEnvs());

  it("makes Start with OCC Agent the primary shell action", async () => {
    renderApp("/");
    const links = await screen.findAllByRole("link", { name: new RegExp(copy.nav.startAgent) });
    expect(links.length).toBeGreaterThan(0);
    expect(links[0]).toHaveAttribute("href", "/agent");
  });

  it("keeps manual delivery available as the specialist route", async () => {
    renderApp("/");
    const manual = await screen.findAllByRole("link", { name: new RegExp(copy.nav.manual) });
    expect(manual.length).toBeGreaterThan(0);
    expect(manual[0]).toHaveAttribute("href", "/new");
    // The specialist grouping is labelled, so manual is secondary — not hidden.
    expect(screen.getAllByText(copy.nav.specialistHeading).length).toBeGreaterThan(0);
  });

  it("shows the agent-first home hero with quick actions and the manual note", async () => {
    renderApp("/");
    expect(await screen.findByText(copy.home.heroHeading)).toBeInTheDocument();
    expect(screen.getByText(copy.home.heroBody)).toBeInTheDocument();
    // The hero CTA and the shell's start button both lead to the OCC Agent.
    for (const link of screen.getAllByRole("link", { name: new RegExp(copy.home.heroCta) })) {
      expect(link).toHaveAttribute("href", "/agent");
    }
    for (const label of Object.values(copy.home.quick)) {
      expect(screen.getByRole("link", { name: label })).toBeInTheDocument();
    }
    expect(screen.getByText(new RegExp(copy.home.manualNote.slice(0, 40)))).toBeInTheDocument();
    expect(screen.getByRole("link", { name: copy.home.manualCta })).toHaveAttribute(
      "href",
      "/new",
    );
  });

  it("a home quick action opens the OCC Agent with a matching starting point", async () => {
    const user = userEvent.setup();
    renderApp("/");
    await user.click(await screen.findByRole("link", { name: copy.home.quick.onboardClient }));
    const box = await screen.findByLabelText(copy.agent.newCaseHeading);
    expect((box as HTMLTextAreaElement).value).toMatch(/^Onboard /);
  });

  it("preserves the manual-first experience when the flag is off", async () => {
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "");
    renderApp("/");
    await screen.findByText(copy.home.subtitle);
    // The old primary action, unchanged.
    expect(
      screen.getAllByRole("link", { name: new RegExp(copy.nav.startNew) }).length,
    ).toBeGreaterThan(0);
    // No agent hero, no agent nav, no specialist grouping.
    expect(screen.queryByText(copy.home.heroHeading)).not.toBeInTheDocument();
    expect(screen.queryByText(copy.nav.agent)).not.toBeInTheDocument();
    expect(screen.queryByText(copy.nav.specialistHeading)).not.toBeInTheDocument();
  });
});

describe("Capital 123 — pipeline and funded as separate streams", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => vi.unstubAllEnvs());

  it("the exact instruction renders two streams with the regime rule", async () => {
    const user = userEvent.setup();
    renderApp("/agent");
    const box = await screen.findByLabelText(copy.agent.newCaseHeading);
    await user.type(
      box,
      "Onboard Capital 123. They will provide a pipeline and a funded book.",
    );
    await user.click(screen.getByRole("button", { name: copy.agent.createButton }));

    // The case summary names the client in full — digits included.
    expect((await screen.findAllByText(/Capital 123/)).length).toBeGreaterThan(0);

    const funded = await screen.findByTestId("stream-funded");
    const pipeline = await screen.findByTestId("stream-pipeline");
    expect(within(pipeline).getByText(/Pipeline MI only/)).toBeInTheDocument();
    expect(within(pipeline).getByText(copy.agent.streamRegimeNone)).toBeInTheDocument();
    expect(within(funded).getByText(/Funded MI/)).toBeInTheDocument();
    expect(
      within(funded).getByText(new RegExp(copy.agent.streamRegimePotential)),
    ).toBeInTheDocument();
  });
});

describe("the chronological case page", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => vi.unstubAllEnvs());

  it("a preset creates a case, opens it, and shows the timeline", async () => {
    const user = userEvent.setup();
    renderApp("/agent");
    const card = (await screen.findByText("A — Clean onboarding")).closest("li");
    await user.click(within(card as HTMLElement).getByRole("button"));

    // The created case is open: summary card, current stage, and timeline.
    expect(await screen.findByText(copy.agent.timelineHeading)).toBeInTheDocument();
    expect(screen.getAllByText(copy.agent.summaryStage).length).toBeGreaterThan(0);
    // Exactly one stage is current.
    await waitFor(() => {
      const current = document.querySelectorAll('[data-stage-status="current"]');
      expect(current.length).toBe(1);
    });
  });

  it("a fresh case starts at the beginning of the journey, with later stages closed", async () => {
    const user = userEvent.setup();
    renderApp("/agent");
    const box = await screen.findByLabelText(copy.agent.newCaseHeading);
    await user.type(box, "Onboard Northstar Lending. UK equity release. Monthly portfolio MI.");
    await user.click(screen.getByRole("button", { name: copy.agent.createButton }));
    await screen.findByText(copy.agent.timelineHeading);

    const current = document.querySelector('[data-stage-status="current"]');
    expect(current?.getAttribute("data-stage")).toBe("pack_prepare");
    // Future stages are rows only — the rehearsal button is not offered yet.
    const future = document.querySelector('[data-stage="rehearsal"]');
    expect(future?.getAttribute("data-stage-status")).toBe("future");
    expect(screen.queryByRole("button", { name: "Start rehearsal" })).not.toBeInTheDocument();
  });

  it("a completed preset advances the timeline to the activation decision", async () => {
    const user = userEvent.setup();
    renderApp("/agent");
    const card = (await screen.findByText("A — Clean onboarding")).closest("li");
    await user.click(within(card as HTMLElement).getByRole("button"));
    await screen.findByText(copy.agent.timelineHeading);

    await waitFor(() => {
      const current = document.querySelector('[data-stage-status="current"]');
      expect(current?.getAttribute("data-stage")).toBe("confirm_activation");
    });
    // Everything before it reads as done.
    expect(
      document.querySelector('[data-stage="rehearsal"]')?.getAttribute("data-stage-status"),
    ).toBe("done");
  });
});

describe("failure never blanks the OCC Agent", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => vi.unstubAllEnvs());

  function renderCase(client: MockOpsClient, caseRef: string) {
    return render(
      <MemoryRouter initialEntries={[`/agent/${caseRef}`]}>
        <OpsClientProvider client={client}>
          <ToastProvider>
            <Routes>
              <Route
                path="/agent/:caseId"
                element={
                  <RouteErrorBoundary>
                    <AgentCaseScreen />
                  </RouteErrorBoundary>
                }
              />
            </Routes>
          </ToastProvider>
        </OpsClientProvider>
      </MemoryRouter>,
    );
  }

  it("a failed pack draft shows an error and keeps the case page standing", async () => {
    const user = userEvent.setup();
    const client = new MockOpsClient();
    const created = await client.createAgentCase(
      "Onboard Northstar Lending. UK equity release. Monthly portfolio MI.",
    );
    vi.spyOn(client, "draftAgentPack").mockRejectedValue(
      new Error("The pack could not be drafted just now."),
    );

    renderCase(client, created.case_ref);
    const draft = await screen.findByRole("button", { name: copy.agent.packDraft });
    await user.click(draft);

    // The failure is visible…
    expect(
      await screen.findByText("The pack could not be drafted just now."),
    ).toBeInTheDocument();
    // …and the case page is still there, with the action available to retry.
    expect(screen.getByText(copy.agent.timelineHeading)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: copy.agent.packDraft })).toBeEnabled();
  });

  it("a rendering fault shows the failure card instead of a blank page", async () => {
    function Boom(): never {
      throw new Error("render fault");
    }
    render(
      <MemoryRouter>
        <RouteErrorBoundary>
          <Boom />
        </RouteErrorBoundary>
      </MemoryRouter>,
    );
    expect(screen.getByText(copy.agent.errorTitle)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: copy.agent.errorRetry })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: copy.agent.errorBack })).toHaveAttribute(
      "href",
      "/agent",
    );
  });

  it("a pack missing its optional lists still renders (deployed-shape tolerance)", async () => {
    const client = new MockOpsClient();
    const created = await client.createAgentCase(
      "Onboard Northstar Lending. UK equity release. Monthly portfolio MI.",
    );
    const real = client.getAgentCase.bind(client);
    vi.spyOn(client, "getAgentCase").mockImplementation(async (ref: string) => {
      const status = await real(ref);
      // The deployed defect: a drafted pack whose projection dropped the
      // confirmation and not-asked lists. The page must not crash on it.
      return {
        ...status,
        pack: {
          ...status.pack,
          sections: status.pack.sections ?? [],
          confirmations: undefined,
          not_asked: undefined,
          summary: undefined,
        },
      } as unknown as Awaited<ReturnType<typeof real>>;
    });

    renderCase(client, created.case_ref);
    expect(await screen.findByText(copy.agent.timelineHeading)).toBeInTheDocument();
    expect(screen.queryByText(copy.agent.errorTitle)).not.toBeInTheDocument();
  });
});
