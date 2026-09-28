import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
import { copy } from "@/lib/copy";

/**
 * A case being run through the OCC Agent is work, and the OCC says so.
 *
 * WHAT AN OPERATOR SAW
 *
 * A real client onboarding had been issued and was waiting on the client's
 * reply. Client Onboarding showed "Nothing in progress" and "Nothing is waiting
 * on a client"; Home showed five zeros under "Nothing needs your attention".
 *
 * None of that was wrong about the STORE. An Agent case lives in the synthetic
 * container and reaches the governed one only at activation — and promotion
 * refuses anything not already approved, so a case appears governed-side for
 * the first time already approved and is activated immediately after. It could
 * never appear in these queues. It went from absent to Active in one step.
 *
 * That is the isolation boundary working at the storage layer and failing at
 * the product layer. The doorway governs what may be WRITTEN; it was never
 * meant to govern what an operator may SEE. So the reader widened and nothing
 * else: the rows are read-only, they link to the Agent tab where the case is
 * actually worked, and no case crosses any earlier than it did before.
 */

function renderApp(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <App />
    </MemoryRouter>,
  );
}

async function createAgentCase(user: ReturnType<typeof userEvent.setup>) {
  const box = await screen.findByLabelText(copy.agent.newCaseHeading);
  await user.type(
    box,
    "Onboard Northstar Lending. UK equity release. Monthly management information.",
  );
  await user.click(screen.getByRole("button", { name: copy.agent.createButton }));
  await screen.findByText(copy.agent.conversationHeading);
}

describe("Client onboarding — Agent cases in the governed queues", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("a new Agent case shows as work in progress, not as an empty queue", async () => {
    const user = userEvent.setup();
    renderApp("/agent");
    await createAgentCase(user);

    // Two nav bars render the same links (desktop and mobile); either will do.
    await user.click(screen.getAllByRole("link", { name: copy.nav.onboarding })[0]);
    await screen.findByText(copy.onboarding.draftsHeading);

    // The heading whose emptiness was the lie.
    await waitFor(() =>
      expect(screen.queryByText(copy.onboarding.noDrafts)).not.toBeInTheDocument(),
    );
    expect(await screen.findByText("Northstar Lending")).toBeInTheDocument();
  });

  it("the row is marked as the Agent's, and links there rather than to the wizard", async () => {
    /* The case wizard reads the governed store, which has never heard of this
       case — following a link to it is the 404 this whole change is about. */
    const user = userEvent.setup();
    renderApp("/agent");
    await createAgentCase(user);

    // Two nav bars render the same links (desktop and mobile); either will do.
    await user.click(screen.getAllByRole("link", { name: copy.nav.onboarding })[0]);
    const row = await screen.findByTestId("agent-case-row");
    expect(within(row).getByText(copy.onboarding.agentChip)).toBeInTheDocument();
    expect(row.getAttribute("href")).toMatch(/^\/agent\//);
    expect(row.getAttribute("href")).not.toMatch(/\/onboarding\/cases\//);
  });

  it("Client onboarding still renders when the Agent tab is switched off", async () => {
    /* The tab is flag-gated and may not be mounted, in which case listing its
       cases fails. That is never this page's failure. */
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "false");
    renderApp("/onboarding");

    expect(await screen.findByText(copy.onboarding.draftsHeading)).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByText(copy.onboarding.noDrafts)).toBeInTheDocument(),
    );
    expect(screen.queryByText(copy.onboarding.unavailable)).not.toBeInTheDocument();
  });
});

describe("OCC Agent — the case list is not called practice", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => vi.unstubAllEnvs());

  it("the heading does not call every case a rehearsal", () => {
    /* The list holds rehearsals AND real client onboardings; the mode is per
       case. A real onboarding sat under a heading calling it practice. */
    expect(copy.agent.casesHeading.toLowerCase()).not.toContain("practice");
    expect(copy.agent.caseEmpty.toLowerCase()).not.toContain("practice");
    expect(copy.agent.caseCreated.toLowerCase()).not.toContain("practice");
  });

  it("but the per-case distinction is kept, because it is the true one", () => {
    expect(copy.agent.syntheticChip).toBe("Practice case");
    expect(copy.agent.modeLiveBadge).toBe("Real onboarding");
  });
});

describe("OCC Agent — the screen does not call a real onboarding practice", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => vi.unstubAllEnvs());

  it("a rehearsal still leads with the practice banner", async () => {
    const user = userEvent.setup();
    renderApp("/agent");
    await createAgentCase(user);
    expect(await screen.findByText(copy.agent.syntheticBanner)).toBeInTheDocument();
    expect(screen.queryByText(copy.agent.liveBanner)).not.toBeInTheDocument();
  });

  it("the practice banner never claims a real onboarding cannot send email", () => {
    /* It rendered unconditionally, so a case that had just emailed a client
       led with "Practice mode ... does not ... send email". A banner that
       contradicts what the operator did a minute ago invites them to doubt the
       thing that actually happened. */
    expect(copy.agent.syntheticBanner).toContain("send email");
    expect(copy.agent.liveBanner).not.toContain("does not");
    expect(copy.agent.liveBanner.toLowerCase()).toContain("real client onboarding");
  });

  it("the upload help does not call the client's own files practice files", () => {
    expect(copy.agent.uploadHelp.toLowerCase()).not.toContain("practice file");
    expect(copy.agent.uploadHelp.toLowerCase()).not.toContain("this practice case");
  });

  it("nothing an operator reads on any case calls it a practice case outright", () => {
    /* These render regardless of mode, so each was a standing false claim on a
       real onboarding. Strings that describe the REHEARSAL — which both modes
       run — are deliberately not in this list. */
    for (const s of [
      copy.agent.casesHeading,
      copy.agent.caseEmpty,
      copy.agent.caseCreated,
      copy.agent.newCase,
      copy.agent.uploadHelp,
      copy.agent.readyHeadline,
      copy.agent.notFound,
      copy.agent.previewNothingWritten,
      copy.agent.artefactNotWritten,
    ]) {
      expect(s.toLowerCase()).not.toContain("practice");
    }
  });
});
