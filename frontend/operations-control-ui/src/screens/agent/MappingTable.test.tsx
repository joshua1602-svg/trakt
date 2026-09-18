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

/** The section for one file of the pack. */
function fileSection(name: string): HTMLElement {
  return within(table()).getByText(name).closest("section") as HTMLElement;
}

/**
 * One row, named the way the server names a mapping: by FILE and column.
 *
 * A pack carries the same column name in more than one file — "Val Dt" is in
 * the loan tape and the property extract — so a helper that looked a row up by
 * name alone found two and threw. That is the same collision the decision
 * lookup had, arriving in the tests.
 */
function rowFor(column: string, file = "loan_tape.csv"): HTMLElement {
  return within(fileSection(file)).getByText(column).closest("tr") as HTMLElement;
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
    /* One of nine. On a first delivery EVERY confident match is proposed and
       feeds nothing until approved — in the property extract as much as in the
       tape, because what an operator approves becomes a rule for the whole
       book. `Val Dt` and `Prp Ref` are questions. What is left feeding a field
       is the one column an operator has already confirmed. */
    await afterTheRun();
    expect(within(table()).getByText(copy.agent.mappingCount(1, 9))).toBeInTheDocument();
  });

  it("distinguishes what an operator confirmed from what Trakt decided", async () => {
    await afterTheRun();
    expect(within(rowFor("Prop Val")).getByText("You confirmed it")).toBeInTheDocument();
  });

  it("settles nothing on its own on a first delivery, in any file", async () => {
    /* The reported defect, in the operator's words: "Why are some fields being
       proposed, while other fields are still automatically matched? ALL fields
       should either be proposed, or unmapped."

       The cause was that only the primary tape's columns were put to a person;
       every other file's were matched and reported as settled. So a hundred and
       fifty three columns came back reading two different ways, and the
       thirty-six that read "matched automatically" were settled by exactly the
       alias registry a first onboarding exists to stop trusting unread. */
    await afterTheRun();
    expect(within(table()).queryByText("Matched automatically")).toBeNull();
    expect(within(rowFor("property_value", "property_tape.csv")).getByText("Proposed")).
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
    await user.click(within(table()).getByRole("button", { name: /Needs you 3/ }));
    await waitFor(() => expect(within(table()).queryByText("loan_id")).not.toBeInTheDocument());
    expect(within(fileSection("loan_tape.csv")).getByText("Val Dt")).toBeInTheDocument();

    await user.click(within(table()).getByRole("button", { name: /All 9/ }));
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

  it("asks about a weak match in a file the tape is not built from", async () => {
    /* It used to read "Weak match, nothing asked" and have nothing to click.
       The canonical tape is not built from the property extract, but a mapping
       an operator approves is promoted to a rule scoped to the BOOK, and
       production consolidates a loan-domain field whichever file carries it —
       so a column here is worth a person's answer exactly as much as one in
       the tape. */
    await afterTheRun();
    const row = rowFor("Prp Ref", "property_tape.csv");
    expect(within(row).getByText("Needs you")).toBeInTheDocument();
    expect(within(row).getByText(copy.agent.mappingAnswer)).toBeInTheDocument();
  });

  it("gives the same column name in two files its own question", async () => {
    /* The decision lookup keyed on the column NAME alone, so a question raised
       about the tape's "Val Dt" marked the property extract's "Val Dt" too:
       two rows pointing at one question, one of which it was not about. */
    await afterTheRun();
    expect(within(rowFor("Val Dt", "loan_tape.csv")).getByText("Needs you")).
      toBeInTheDocument();
    expect(within(rowFor("Val Dt", "property_tape.csv")).getByText("Proposed")).
      toBeInTheDocument();
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
    expect(within(table()).getByRole("button", { name: /Approve 4 mappings/ })).
      toBeInTheDocument();
  });

  it("does not also raise a card for every proposed column", async () => {
    /* The whole point of approving the set. The one genuine question keeps its
       card; the three proposals do not, because seventy cards beside a table
       listing the same seventy columns is the friction, not the governance. */
    await afterTheRun();
    const panel = screen.getByText(copy.agent.decisionsHeading).closest("section");
    // Three genuine questions keep their cards — the two weak matches and the
    // ambiguity — and the four proposals do not.
    expect(within(panel as HTMLElement).getAllByRole("listitem")).toHaveLength(3);
  });

  it("will not approve the set while a real question is unanswered", async () => {
    /* A weak match is answered on its own. Offering to settle the set while one
       waits would promise a run that cannot move. */
    await afterTheRun();
    expect(within(table()).getByRole("button", { name: /Approve 4 mappings/ })).
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

/* A COLUMN THAT MATCHED NOTHING USED TO BE A DEAD END.
 *
 * "For unmapped fields, there should still be an option to i) add to field
 * registry as a new entry, or ii) add to an existing field in the field
 * registry as an alias."
 *
 * The table labelled it "Not used" and stopped there. On a first delivery that
 * is most of the tape — eighty-nine of a hundred and fifty three for the first
 * client through — and an operator who KNEW what the column was had nowhere to
 * say so.
 *
 * The two acts are not the same act, and these tests hold them apart. Naming a
 * field Trakt already has settles the column here and now. Asking for a field
 * it does not have changes what every client's report is written in, so it is
 * recorded as a request and the column stays unused.
 */
describe("OCC Agent — a column that matched nothing has somewhere to go", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
    vi.stubEnv("VITE_OCC_AGENT_SYNTHETIC_ENABLED", "true");
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  async function openTheDialog() {
    const user = await afterTheRun();
    await user.click(
      within(rowFor("Internal Ref")).getByRole("button", {
        name: copy.agent.mappingUnmappedAction,
      }),
    );
    await screen.findByText(copy.agent.mappingUnmappedHeading("Internal Ref"));
    return user;
  }

  it("offers an unused column somewhere to go", async () => {
    await afterTheRun();
    expect(
      within(rowFor("Internal Ref")).getByRole("button", {
        name: copy.agent.mappingUnmappedAction,
      }),
    ).toBeInTheDocument();
  });

  it("offers nothing of the sort on a column that already has a field", async () => {
    /* The action belongs to a column with no field. On one that has one it
       would be a second, ungoverned way to change a mapping — the governed way
       is answering its decision. */
    await afterTheRun();
    expect(
      within(rowFor("Int Rate")).queryByRole("button", {
        name: copy.agent.mappingUnmappedAction,
      }),
    ).toBeNull();
  });

  it("maps it to a field Trakt already has", async () => {
    const user = await openTheDialog();
    // The picker holds still until the field list has arrived: a box you can
    // type a field name into before the list exists is a box that cannot tell
    // you the name is not one.
    const picker = screen.getByLabelText(copy.agent.mappingPickField);
    await waitFor(() => expect(picker).toBeEnabled());
    await user.type(picker, "borrower_date_of_birth");
    await user.click(
      screen.getByRole("button", { name: copy.agent.mappingUseExistingConfirm }),
    );
    await waitFor(() =>
      expect(
        within(rowFor("Internal Ref")).getByText(/borrower date of birth/),
      ).toBeInTheDocument(),
    );
    expect(within(rowFor("Internal Ref")).getByText("You confirmed it")).
      toBeInTheDocument();
  });

  it("refuses a field Trakt does not report on", async () => {
    /* A free-text box would let a name that matches nothing be promoted into a
       governed rule that silently matches nothing every month. */
    const user = await openTheDialog();
    const picker = screen.getByLabelText(copy.agent.mappingPickField);
    await waitFor(() => expect(picker).toBeEnabled());
    await user.type(picker, "made_up_field");
    expect(
      screen.getByRole("button", { name: copy.agent.mappingUseExistingConfirm }),
    ).toBeDisabled();
  });

  it("records an ask for a new field without mapping anything", async () => {
    /* The distinction the whole design turns on: a request is not a field, and
       the column has to keep saying so. */
    const user = await openTheDialog();
    await user.click(screen.getByLabelText(copy.agent.mappingRequestNew));
    await user.type(screen.getByLabelText(copy.agent.mappingNewFieldName), "broker_code");
    await user.type(
      screen.getByLabelText(copy.agent.mappingNewFieldWhat),
      "The intermediary who introduced the case.",
    );
    await user.click(
      screen.getByRole("button", { name: copy.agent.mappingRequestConfirm }),
    );
    await waitFor(() =>
      expect(
        within(rowFor("Internal Ref")).getByText(
          copy.agent.mappingRequestedChip("broker_code"),
        ),
      ).toBeInTheDocument(),
    );
    // Still unused: nothing was mapped, and a screen that implied otherwise
    // would have an operator believe a field exists that does not.
    expect(within(rowFor("Internal Ref")).getByText("Not used")).toBeInTheDocument();
  });

  it("says who is going to act on the ask, and lets it be taken back", async () => {
    const user = await openTheDialog();
    await user.click(screen.getByLabelText(copy.agent.mappingRequestNew));
    await user.type(screen.getByLabelText(copy.agent.mappingNewFieldName), "broker_code");
    await user.click(
      screen.getByRole("button", { name: copy.agent.mappingRequestConfirm }),
    );
    const asks = await screen.findByText(copy.agent.mappingRequestsHeading);
    const panel = asks.closest("section") as HTMLElement;
    expect(within(panel).getByText(copy.agent.mappingRequestNewHelp)).toBeInTheDocument();
    // The control holds still while the previous act is in flight, so the walk
    // waits for it rather than clicking at a button that is not listening.
    const withdraw = within(panel).getByRole("button", {
      name: copy.agent.mappingWithdrawRequest,
    });
    await waitFor(() => expect(withdraw).toBeEnabled());
    await user.click(withdraw);
    await waitFor(() =>
      expect(screen.queryByText(copy.agent.mappingRequestsHeading)).toBeNull(),
    );
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

