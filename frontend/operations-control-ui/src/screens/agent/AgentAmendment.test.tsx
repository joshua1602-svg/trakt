import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
import { copy } from "@/lib/copy";

/**
 * Changing a client that is already live.
 *
 * The OCC Agent could only ever open a NEW onboarding. That is a problem for
 * the ordinary shape of this work: a client goes live on management
 * information, and the regulatory return is added weeks later when their
 * reference data arrives. Doing that in conversation on the live case does not
 * work — `regime_required` lives on the source registry and is written when a
 * configuration is ACTIVATED, so the book stays registered as it was and the
 * engine refuses the delivery rather than splitting it across two incomplete
 * ones.
 *
 * So the amendment has to be reachable, and it has to be its OWN act. A
 * checkbox on the new-case box would put "onboard them" and "change what is in
 * force for them" behind one control, which is how the wrong one gets done.
 */

function renderApp(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <App />
    </MemoryRouter>,
  );
}

describe("amending a client already live", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("offers the amendment as a separate act from opening a new case", async () => {
    renderApp("/agent");
    await screen.findByText(copy.agent.newCaseHeading);

    const panel = await screen.findByRole("group", {
      name: copy.agent.amendHeading,
    });
    expect(panel).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: copy.agent.amendStart }),
    ).toBeInTheDocument();
  });

  it("says why this is not an edit", async () => {
    renderApp("/agent");
    expect(await screen.findByText(copy.agent.amendPrompt)).toBeInTheDocument();
  });

  it("will not open an amendment without naming a client", async () => {
    renderApp("/agent");
    const button = await screen.findByRole("button", {
      name: copy.agent.amendStart,
    });
    expect(button).toBeDisabled();
  });

  it("refuses a client that has no configuration to amend, and says so", async () => {
    const user = userEvent.setup();
    renderApp("/agent");

    await user.type(
      await screen.findByLabelText(copy.agent.amendLabel),
      "NOBODY",
    );
    await user.click(
      await screen.findByRole("button", { name: copy.agent.amendStart }),
    );

    await waitFor(async () =>
      expect(
        await screen.findByText(/no approved configuration to amend/i),
      ).toBeInTheDocument(),
    );
  });

  it("leaves the new-case box alone", async () => {
    const user = userEvent.setup();
    renderApp("/agent");

    // Typing a client to amend must not arm the create button, or the two
    // acts are one control after all.
    await user.type(
      await screen.findByLabelText(copy.agent.amendLabel),
      "ERE",
    );
    expect(
      screen.getByRole("button", { name: new RegExp(copy.agent.createButton, "i") }),
    ).toBeDisabled();
  });
});
