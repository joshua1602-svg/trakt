import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
import { mockAgentLive } from "@/api/MockAgent";
import { copy } from "@/lib/copy";

/**
 * Starting a REAL onboarding, rather than a rehearsal.
 *
 * The Agent authors every case in an isolated container and can only reach a
 * client's live configuration through one gated crossing at activation. A case
 * marked live is what makes that crossing possible, so this is the control that
 * decides whether an onboarding is real — and the failure it must prevent is an
 * operator producing a real one without meaning to.
 *
 * Rendered through the real `App` so what is proven is the control an operator
 * actually meets, not a component in isolation.
 */

function renderAgent() {
  return render(
    <MemoryRouter initialEntries={["/agent"]}>
      <App />
    </MemoryRouter>,
  );
}

async function typeInstruction(user: ReturnType<typeof userEvent.setup>) {
  const box = await screen.findByLabelText(copy.agent.newCaseHeading);
  await user.type(box, "Onboard Northstar Lending. Monthly MI.");
}

describe("starting a real onboarding", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
    mockAgentLive.available = false;
    mockAgentLive.lastCreateLive = undefined;
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    mockAgentLive.available = false;
  });

  it("offers no live choice where the environment cannot do it", async () => {
    renderAgent();
    await screen.findByText(copy.agent.newCaseHeading);
    expect(screen.queryByText(copy.agent.modeHeading)).not.toBeInTheDocument();
  });

  it("creates a rehearsal when the choice is never touched", async () => {
    const user = userEvent.setup();
    mockAgentLive.available = true;
    renderAgent();
    await typeInstruction(user);
    await user.click(screen.getByRole("button", { name: copy.agent.createButton }));
    await waitFor(() => expect(mockAgentLive.lastCreateLive).toBe(false));
  });

  it("will not start a real case on the choice alone", async () => {
    const user = userEvent.setup();
    mockAgentLive.available = true;
    renderAgent();
    await typeInstruction(user);
    await user.click(await screen.findByRole("radio", { name: /Real onboarding/ }));

    // Chosen, but not confirmed: the button must refuse.
    expect(screen.getByRole("button", { name: copy.agent.createButton })).toBeDisabled();
    expect(mockAgentLive.lastCreateLive).toBeUndefined();
  });

  it("will not accept the wrong confirmation word", async () => {
    const user = userEvent.setup();
    mockAgentLive.available = true;
    renderAgent();
    await typeInstruction(user);
    await user.click(await screen.findByRole("radio", { name: /Real onboarding/ }));
    await user.type(screen.getByLabelText(copy.agent.modeLiveConfirmLabel), "yes");

    expect(screen.getByRole("button", { name: copy.agent.createButton })).toBeDisabled();
  });

  it("starts a real onboarding once it is chosen AND confirmed", async () => {
    const user = userEvent.setup();
    mockAgentLive.available = true;
    renderAgent();
    await typeInstruction(user);
    await user.click(await screen.findByRole("radio", { name: /Real onboarding/ }));
    await user.type(
      screen.getByLabelText(copy.agent.modeLiveConfirmLabel),
      copy.agent.modeLiveConfirmWord,
    );

    const button = screen.getByRole("button", { name: /Start/ });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);
    await waitFor(() => expect(mockAgentLive.lastCreateLive).toBe(true));
  });

  it("going back to rehearsal clears the confirmation", async () => {
    const user = userEvent.setup();
    mockAgentLive.available = true;
    renderAgent();
    await typeInstruction(user);
    await user.click(await screen.findByRole("radio", { name: /Real onboarding/ }));
    await user.type(
      screen.getByLabelText(copy.agent.modeLiveConfirmLabel),
      copy.agent.modeLiveConfirmWord,
    );
    await user.click(screen.getByRole("radio", { name: /Rehearsal/ }));

    // The typed word must not survive, or re-selecting live would be armed
    // already and the second step would have been skipped.
    await user.click(screen.getByRole("radio", { name: /Real onboarding/ }));
    expect(screen.getByLabelText(copy.agent.modeLiveConfirmLabel)).toHaveValue("");
    expect(screen.getByRole("button", { name: copy.agent.createButton })).toBeDisabled();
  });
});

/**
 * A real client onboarding must never be labelled practice.
 *
 * Reported, in those words, from a live case: "Why do you keep referring to
 * this as a Rehearsal when this is a live onboarding?" The heading beside the
 * case status read "Practice run" on every case, rehearsal and real alike,
 * because the string was fixed rather than chosen. It sits directly next to
 * what the case has got to, so an operator reads it as a statement about the
 * case in front of them — and on a real onboarding it was untrue.
 *
 * The same defect had already been found once on this screen and fixed for the
 * case-list heading. This proves it for every remaining place the word appears.
 */
describe("a real onboarding is not called practice", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
    mockAgentLive.available = true;
    mockAgentLive.lastCreateLive = undefined;
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    mockAgentLive.available = false;
  });

  async function openARealCase(user: ReturnType<typeof userEvent.setup>) {
    renderAgent();
    await typeInstruction(user);
    await user.click(await screen.findByRole("radio", { name: /Real onboarding/ }));
    await user.type(
      screen.getByLabelText(copy.agent.modeLiveConfirmLabel),
      copy.agent.modeLiveConfirmWord,
    );
    const button = screen.getByRole("button", { name: /Start/ });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);
    await screen.findByText(copy.agent.conversationHeading);
  }

  it("says onboarding run, not practice run, beside the case status", async () => {
    const user = userEvent.setup();
    await openARealCase(user);
    expect(await screen.findByText(copy.agent.stageHeading(true))).toBeInTheDocument();
    expect(screen.queryByText(copy.agent.stageHeading(false))).not.toBeInTheDocument();
  });

  it("uses no practice wording anywhere on the open case", async () => {
    const user = userEvent.setup();
    await openARealCase(user);
    await screen.findByText(copy.agent.stageHeading(true));

    // Deliberately a sweep of the rendered page rather than a list of the
    // strings known to be wrong today: the defect was a fixed string nobody
    // had thought about, and the next one will be too.
    const body = document.body.textContent ?? "";
    for (const word of ["Practice run", "practice run", "practice boundary"]) {
      expect(body, `a real onboarding is described as "${word}"`).not.toContain(word);
    }
  });

  it("still says practice on a rehearsal", async () => {
    const user = userEvent.setup();
    renderAgent();
    await typeInstruction(user);
    await user.click(screen.getByRole("button", { name: copy.agent.createButton }));
    await screen.findByText(copy.agent.conversationHeading);

    expect(await screen.findByText(copy.agent.stageHeading(false))).toBeInTheDocument();
  });
});
