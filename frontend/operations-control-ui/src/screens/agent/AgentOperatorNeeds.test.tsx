import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import type { CaseProblem } from "@/api/onboardingTypes";
import { copy } from "@/lib/copy";
import { OperatorNeedsPanel } from "./AgentCase";

/**
 * The panel that names what the operator owes, and where to answer it.
 *
 * A case reached "Generate configuration" with two required source fields
 * unanswered. Every panel said something was missing; none said where to put
 * it. The only thing that looked like a route was "use the conversation",
 * which could not set those fields — so the case could not move and nothing on
 * screen explained why.
 */
function problem(field: string, message: string): CaseProblem {
  return {
    section: "sources",
    field,
    message,
    severity: "blocking",
    index: null,
    owner: "operator",
  };
}

function panel(problems: CaseProblem[], to = "/onboarding/ONB-2026-0008") {
  render(
    <MemoryRouter>
      <OperatorNeedsPanel problems={problems} to={to} />
    </MemoryRouter>,
  );
}

describe("what Trakt still needs from you", () => {
  it("lists the problems and points at where to answer them", () => {
    panel([problem("file_format", "File format for direct_001/funded is needed.")]);
    expect(screen.getByText(/File format for direct_001\/funded/))
      .toBeInTheDocument();
    expect(screen.getByRole("link", { name: new RegExp(copy.agent.missingWhere) }))
      .toHaveAttribute("href", "/onboarding/ONB-2026-0008");
  });

  it("says these are not the client's, so they are not chased", () => {
    panel([problem("file_format", "File format is needed.")]);
    expect(screen.getByText(copy.agent.missingHelp)).toBeInTheDocument();
  });

  it("does not render at all when the operator owes nothing", () => {
    const { container } = render(
      <MemoryRouter>
        <OperatorNeedsPanel problems={[]} to="/onboarding/X" />
      </MemoryRouter>,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("still lists the problems when there is no case to link to", () => {
    // Degrading to "no link" is right; degrading to "no panel" would hide a
    // blocker because a link was missing.
    panel([problem("file_format", "File format is needed.")], "");
    expect(screen.getByText(/File format is needed/)).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });
});
