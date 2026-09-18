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
/**
 * Scenario B, because this file is about the table BEFORE it is settled.
 *
 * A prepared example walks as far as it can, and on scenario A that is now all
 * the way: the proposals get approved and the weak match answered, exactly as
 * a person would. Scenario B halts on a genuine ambiguity, so its case holds
 * the whole range at once — proposals waiting on one approval, a weak match
 * waiting on its own answer, and two columns claiming one field. That is the
 * table this screen exists for.
 */
async function afterTheRun(fixtureId = "scenario_b_ambiguous_mapping") {
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
    /* Two of eight. On a first delivery the three confident matches are
       PROPOSED and feed nothing until approved; `Val Dt` is a question and
       `Prp Ref` a weak match nobody was asked about. What is left is the one
       column an operator confirmed and the one on a secondary file. */
    await afterTheRun();
    expect(within(table()).getByText(copy.agent.mappingCount(2, 8))).toBeInTheDocument();
  });

  it("distinguishes what an operator confirmed from what Trakt decided", async () => {
    await afterTheRun();
    expect(within(rowFor("Prop Val")).getByText("You confirmed it")).toBeInTheDocument();
    // On a first delivery the primary tape's confident matches are proposed,
    // so what remains "matched automatically" is the secondary file — which
    // the canonical tape is not built from and so raises nothing to approve.
    expect(within(rowFor("property_value")).getByText("Matched automatically")).
      toBeInTheDocument();
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
    // Both unanswered questions come before anything settled.
    expect(columns.slice(0, 2).join(" ")).toContain("Val Dt");
    expect(columns.slice(0, 2).join(" ")).toContain("Current Balance");
  });

  it("filters to one kind and back", async () => {
    const user = await afterTheRun();
    await user.click(within(table()).getByRole("button", { name: /Needs you 2/ }));
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

  /* One row, one line.
   *
   * The reported defect, in the operator's words: "it is too narrow so one row
   * falls onto two lines". The cause was structural — the table sat in a
   * 1024px reading measure, inside a two-column grid whose rail took 24rem,
   * leaving 528px for five columns of a hundred-column tape.
   *
   * jsdom does no layout, so the rendered height cannot be asserted here; that
   * was measured against a real browser across 1024–2560px, where every row
   * comes out at a uniform 46px with nothing clipped. What IS assertable here
   * is the structure that produces it, and the structure is what a careless
   * edit would undo: a fixed layout, and one fact per cell.
   */
  it("gives every fact its own cell rather than stacking them", async () => {
    /* The status used to be a pill appended to the column name. Two facts in
       one cell is what made the longest rows wrap, and a status is the thing
       an operator scans down a column — it has to line up. */
    await afterTheRun();
    const cells = within(rowFor("Val Dt")).getAllByRole("cell");
    expect(cells).toHaveLength(5);
    expect(cells[0]).toHaveTextContent("Val Dt");
    expect(cells[1]).toHaveTextContent("Needs you");
    expect(cells[0]).not.toHaveTextContent("Needs you");
  });

  it("lays the columns out to a fixed width rather than to their content", async () => {
    /* Without this one long evidence sentence re-flows the whole table and
       every row wraps — which is the defect, arriving by a different door. */
    await afterTheRun();
    const el = within(table()).getAllByRole("table")[0];
    expect(el.className).toContain("table-fixed");
    expect(el.querySelectorAll("colgroup col")).toHaveLength(5);
  });

  /* What a model proposed for a column nothing matched.
   *
   * With the model wired into the mapping stage this is the row an operator
   * most needs, and the table rendered "—" for it: the proposal was on the run
   * and visible nowhere. */
  it("shows what a model proposed for a column nothing matched", async () => {
    await afterTheRun();
    expect(within(rowFor("Internal Ref")).getByText(/loan identifier/)).
      toBeInTheDocument();
  });

  it("marks a proposal as a proposal, not as a mapping", async () => {
    /* A suggestion set in the same type as a contract-backed match is the
       model writing mappings by another route. */
    await afterTheRun();
    expect(within(rowFor("Internal Ref")).getByText(copy.agent.mappingProposed)).
      toBeInTheDocument();
    expect(within(rowFor("Internal Ref")).getByText("Not used")).toBeInTheDocument();
  });

  /* THE TABLE IS THE APPROVAL SURFACE.
   *
   * "This is a first time onboarding so once human approves the initial
   * onboarding then it will match every month thereafter. It shouldn't auto
   * match on initial onboarding."
   *
   * A governed alias says the NAME is one the platform has seen before; it does
   * not say this lender means the same thing by it. So on a first delivery every
   * confident match is PROPOSED, and what a person approves here is what gets
   * promoted into governed rules and applied every month after.
   *
   * One act, because seventy proposals answered one at a time is the same
   * approval seventy times over — and the table, because seventy cards beside a
   * table already listing the same seventy columns is the friction this removes.
   */
  it("proposes rather than deciding on a first delivery", async () => {
    await afterTheRun();
    expect(within(rowFor("Int Rate")).getByText("Proposed")).toBeInTheDocument();
  });

  it("offers one act for the whole set, and says what it would settle", async () => {
    await afterTheRun();
    expect(within(table()).getByRole("button", { name: /Approve 2 mappings/ })).
      toBeInTheDocument();
  });

  it("does not also raise a card for every proposed column", async () => {
    /* The whole point of approving the set. The one genuine question keeps its
       card; the three proposals do not, because seventy cards beside a table
       listing the same seventy columns is the friction, not the governance. */
    await afterTheRun();
    const panel = screen.getByText(copy.agent.decisionsHeading).closest("section");
    // Two genuine questions keep their cards — the weak match and the
    // ambiguity — and the two proposals do not.
    expect(within(panel as HTMLElement).getAllByRole("listitem")).toHaveLength(2);
  });

  it("will not approve the set while a real question is unanswered", async () => {
    /* A weak match is answered on its own. Offering to settle the set while one
       waits would promise a run that cannot move. */
    await afterTheRun();
    expect(within(table()).getByRole("button", { name: /Approve 2 mappings/ })).
      toBeDisabled();
    expect(within(table()).getByText(/need an answer first/)).toBeInTheDocument();
  });

  it("offers to change a proposal, not to answer it", async () => {
    /* Different acts, different words: a proposal is already right or it is
       not, and a question has no answer yet. */
    await afterTheRun();
    expect(within(rowFor("Int Rate")).getByText(copy.agent.mappingChange)).
      toBeInTheDocument();
    expect(within(rowFor("Val Dt")).getByText(copy.agent.mappingAnswer)).
      toBeInTheDocument();
  });

  it("does not re-propose a column a person already confirmed", async () => {
    await afterTheRun();
    expect(within(rowFor("Prop Val")).getByText("You confirmed it")).
      toBeInTheDocument();
  });

  it("keeps the word for a model's suggestion distinct from a proposal", async () => {
    /* Two different claims sharing one word on one screen is how an operator
       comes to think a model wrote something a person is being asked to sign. */
    await afterTheRun();
    expect(within(rowFor("Internal Ref")).getByText(copy.agent.mappingProposed)).
      toBeInTheDocument();
    expect(copy.agent.mappingProposed).not.toBe("Proposed");
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

