import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
import { copy } from "@/lib/copy";

/**
 * Every source column, and what Trakt read it as.
 *
 * WHAT WAS INVISIBLE
 *
 * The decisions panel shows the columns the header mapper could NOT settle.
 * Everything it settled on its own had no screen at all: `run.mapping_report`
 * reached the API in the readiness and review packages, was typed in the
 * frontend as `mapping_report: Record<string, unknown>[]`, and was rendered
 * nowhere — the mock returned an empty list. So an operator could answer the
 * twenty-nine questions asked and still had no way to check the seventy-eight
 * answers nobody asked about.
 *
 * WHY THESE ASSERTIONS
 *
 * Two things are easy to build and useless. A table that shows only the rows
 * needing attention is the decisions panel again under a new heading — so the
 * tests check that a column matched automatically is ON SCREEN. And a table
 * that says "needs you" without saying which question — so the test checks the
 * row carries the link to it.
 *
 * The classification itself is the server's, and `MockAgent.mappingOverview`
 * is a double for it. The last group pins that double to the engine's own
 * rule: trusted tier OR confidence at the threshold. If the two drift, a
 * screen starts telling an operator a mapping was checked when it was not.
 */

function renderApp(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <App />
    </MemoryRouter>,
  );
}

/**
 * A case walked to the point where the files have been read.
 *
 * Driven through a prepared example rather than by hand: reaching the run
 * means drafting, approving and issuing the pack, answering the client's
 * questions, submitting and approving — which is the scenario runner's whole
 * job, and re-typing it here would test the walk rather than the table.
 */
async function afterTheRun(fixtureId = "scenario_a_clean") {
  const user = userEvent.setup();
  renderApp("/agent");
  const marker = await screen.findByTestId(`expected-${fixtureId}`);
  const card = marker.closest("li") as HTMLElement;
  await user.click(within(card).getByRole("button", { name: copy.agent.scenarioRun }));
  await screen.findByText(copy.agent.mappingHeading);
  return user;
}

function table(): HTMLElement {
  return screen.getByText(copy.agent.mappingHeading).closest("section") as HTMLElement;
}

function rowFor(column: string): HTMLElement {
  return within(table()).getByText(column).closest("tr") as HTMLElement;
}

describe("OCC Agent — every column is accounted for", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("shows the columns nobody was asked about, not just the exceptions", async () => {
    /* The defect in one assertion: `loan_id` matched on its own, so it raised
       no decision, so before this panel it appeared on no screen. */
    await afterTheRun();
    expect(within(table()).getByText("loan_id")).toBeInTheDocument();
  });

  it("says what it read each column as", async () => {
    await afterTheRun();
    expect(within(rowFor("Current Balance")).getByText(/current principal balance/)).
      toBeInTheDocument();
  });

  it("says on what evidence, in words rather than the mapper's vocabulary", async () => {
    await afterTheRun();
    const basis = within(rowFor("Current Balance")).getByText(/alias/i);
    expect(basis).toBeInTheDocument();
    expect(basis.textContent).not.toMatch(/fuzz|token_set|normalized/);
  });

  it("shows the confidence behind a weak match", async () => {
    await afterTheRun();
    expect(within(rowFor("Val Dt")).getByText("62%")).toBeInTheDocument();
  });

  it("counts what is feeding a field, not what was proposed", async () => {
    /* Five of eight. `Val Dt` is a proposal waiting on somebody and `Prp Ref`
       is a weak match nobody was asked about, so neither counts. */
    await afterTheRun();
    expect(within(table()).getByText(copy.agent.mappingCount(5, 8))).toBeInTheDocument();
  });

  it("distinguishes what an operator confirmed from what Trakt decided", async () => {
    await afterTheRun();
    expect(within(rowFor("Prop Val")).getByText("You confirmed it")).toBeInTheDocument();
    expect(within(rowFor("loan_id")).getByText("Matched automatically")).toBeInTheDocument();
  });

  it("shows a column nothing matched rather than dropping it", async () => {
    await afterTheRun();
    expect(within(rowFor("Internal Ref")).getByText("Not used")).toBeInTheDocument();
  });

  it("puts the rows that need a person first", async () => {
    await afterTheRun();
    const columns = within(table())
      .getAllByRole("row")
      .slice(1)
      .map((row) => row.querySelector("td")?.textContent ?? "");
    expect(columns[0]).toContain("Val Dt");
  });

  it("filters to one kind and back", async () => {
    const user = await afterTheRun();
    await user.click(within(table()).getByRole("button", { name: /Needs you 1/ }));
    await waitFor(() => expect(within(table()).queryByText("loan_id")).not.toBeInTheDocument());
    expect(within(table()).getByText("Val Dt")).toBeInTheDocument();

    await user.click(within(table()).getByRole("button", { name: /All 8/ }));
    await waitFor(() => expect(within(table()).getByText("loan_id")).toBeInTheDocument());
  });

  it("offers no filter for a kind with no rows", async () => {
    /* A chip reading "Could not be read 0" is a question nobody asked. */
    await afterTheRun();
    expect(within(table()).queryByRole("button", { name: /Could not be read/ })).toBeNull();
  });

  it("shows every file in the pack, not only the one the tape is built from", async () => {
    /* A pack is three or four tapes. The report used to cover the primary tape
       alone, so the property and cash-flow tapes appeared nowhere. */
    await afterTheRun();
    expect(within(table()).getByText("loan_tape.csv")).toBeInTheDocument();
    expect(within(table()).getByText("property_tape.csv")).toBeInTheDocument();
  });

  it("says which file the canonical tape is built from", async () => {
    await afterTheRun();
    expect(within(table()).getByText(copy.agent.mappingPrimaryFile)).toBeInTheDocument();
    expect(within(table()).getByText(copy.agent.mappingSecondaryFile)).toBeInTheDocument();
  });

  it("does not claim a question is waiting on a file the tape is not built from", async () => {
    /* `Prp Ref` matches weakly, but no decision is raised for a secondary
       file — so "Needs you" would point at a question that does not exist. */
    await afterTheRun();
    expect(within(rowFor("Prp Ref")).getByText("Weak match, nothing asked")).
      toBeInTheDocument();
    expect(within(rowFor("Prp Ref")).queryByRole("link")).toBeNull();
  });

  it("says nothing has been read yet before the run", async () => {
    const user = userEvent.setup();
    renderApp("/agent");
    const box = await screen.findByLabelText(copy.agent.newCaseHeading);
    await user.type(box, "Onboard Northstar Lending. Monthly management information.");
    await user.click(screen.getByRole("button", { name: copy.agent.createButton }));
    await screen.findByText(copy.agent.conversationHeading);
    // A case created from this screen is a rehearsal until an operator says
    // otherwise, so this is the practice wording.
    expect(await screen.findByText(copy.agent.mappingEmpty(false))).toBeInTheDocument();
  });
});

describe("OCC Agent — a blocked column does not read as settled", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("links a needs-you row to the decision that settles it", async () => {
    /* Without this the table says "needs you" and sends an operator hunting
       for the matching question in a different list. */
    await afterTheRun("scenario_b_ambiguous_mapping");
    const link = within(rowFor("Current Balance")).getByRole("link", {
      name: copy.agent.mappingAnswer,
    });
    expect(link).toHaveAttribute("href", "#decision-amb_current_principal_balance");
    expect(document.getElementById("decision-amb_current_principal_balance")).
      toBeInTheDocument();
  });

  it("calls an ambiguous column unsettled, however well its name matched", async () => {
    /* The trap: an ambiguity is raised AFTER both rows are written to the
       report, at whatever tier they matched at — often an alias or exact
       match, because that is how both came to claim the same field. Reading
       the tier alone, the table would say "matched automatically" about a
       column the run is blocked on. */
    await afterTheRun("scenario_b_ambiguous_mapping");
    expect(within(rowFor("Current Balance")).getByText("Needs you")).toBeInTheDocument();
    expect(within(rowFor("Current Balance")).queryByText("Matched automatically")).toBeNull();
  });
});

