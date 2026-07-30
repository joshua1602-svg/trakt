import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { MockOpsClient } from "@/api/MockOpsClient";
import type { CreateBatchInput } from "@/api/types";
import { OpsClientProvider } from "@/api/context";
import { ToastProvider } from "@/components/Toast";
import { NewWorkflowScreen } from "./NewWorkflow";

/** Renders the screen and reports every createBatch payload it sends, so the
 * assertions are about what actually reaches the backend rather than about
 * which pixels changed. */
function renderScreen() {
  const client = new MockOpsClient(0);
  const sent: CreateBatchInput[] = [];
  const create = client.createBatch.bind(client);
  vi.spyOn(client, "createBatch").mockImplementation(async (input) => {
    sent.push(input);
    return create(input);
  });
  render(
    <OpsClientProvider client={client}>
      <ToastProvider>
        <MemoryRouter initialEntries={["/new"]}>
          <NewWorkflowScreen />
        </MemoryRouter>
      </ToastProvider>
    </OpsClientProvider>,
  );
  return { sent, client };
}

function bookOption(value: string) {
  const label = document.querySelector(`[data-dataset="${value}"]`);
  if (!label) throw new Error(`no book option for ${value}`);
  return label as HTMLElement;
}

/** Chooses a client, then one of that client's registered portfolios. */
async function chooseBook(
  user: ReturnType<typeof userEvent.setup>,
  clientName = "Alpine Capital",
  portfolioId = "direct_001",
) {
  await waitFor(() => expect(screen.getByLabelText("Client")).toBeTruthy());
  await user.selectOptions(screen.getByLabelText("Client"), clientName);
  await waitFor(() =>
    expect(
      (screen.getByLabelText("Portfolio") as HTMLSelectElement).options.length,
    ).toBeGreaterThan(2),
  );
  await user.selectOptions(screen.getByLabelText("Portfolio"), portfolioId);
}

async function fillPeriod(user: ReturnType<typeof userEvent.setup>) {
  const period = screen.getByLabelText("Reporting period") as HTMLInputElement;
  await user.clear(period);
  await user.type(period, "2026-06");
}

describe("choosing who the delivery is for", () => {
  it("offers only the chosen client's portfolios", async () => {
    const user = userEvent.setup();
    renderScreen();
    await waitFor(() => expect(screen.getByLabelText("Client")).toBeTruthy());
    await user.selectOptions(screen.getByLabelText("Client"), "Alpine Capital");

    const select = screen.getByLabelText("Portfolio") as HTMLSelectElement;
    await waitFor(() => expect(select.options.length).toBeGreaterThan(2));
    const values = Array.from(select.options).map((o) => o.value);
    expect(values).toContain("direct_001");
    expect(values).toContain("acquired_002");
    // Another client's book is never in this client's list.
    expect(values).not.toContain("direct_010");
  });

  it("re-scopes the list when the client changes", async () => {
    const user = userEvent.setup();
    renderScreen();
    await chooseBook(user);
    await user.selectOptions(screen.getByLabelText("Client"), "Birchwood Partners");

    const select = screen.getByLabelText("Portfolio") as HTMLSelectElement;
    await waitFor(() =>
      expect(Array.from(select.options).map((o) => o.value)).toContain("direct_010"),
    );
    expect(Array.from(select.options).map((o) => o.value)).not.toContain("direct_001");
    // The previous client's choice does not survive the change.
    expect(select.value).toBe("");
  });

  it("has no free-text portfolio field", async () => {
    const user = userEvent.setup();
    renderScreen();
    await chooseBook(user);
    const portfolio = screen.getByLabelText("Portfolio");
    expect(portfolio.tagName).toBe("SELECT");
  });

  it("shows the book's display name, identifier and asset class separately", async () => {
    const user = userEvent.setup();
    renderScreen();
    await chooseBook(user);

    const context = await waitFor(() => {
      const el = document.querySelector('[data-portfolio-context="direct_001"]');
      if (!el) throw new Error("no portfolio context");
      return el as HTMLElement;
    });
    expect(within(context).getByText("Alpine Direct Originations")).toBeInTheDocument();
    expect(within(context).getByText("direct_001")).toBeInTheDocument();
    // Asset class is its own fact, not part of the portfolio's identity.
    expect(within(context).getByText("Asset class")).toBeInTheDocument();
    expect(within(context).getByText("Equity Release")).toBeInTheDocument();
  });

  it("lets an operator declare a book Trakt has never seen", async () => {
    const user = userEvent.setup();
    const { sent } = renderScreen();
    await waitFor(() => expect(screen.getByLabelText("Client")).toBeTruthy());
    await user.selectOptions(screen.getByLabelText("Client"), "Alpine Capital");
    await waitFor(() =>
      expect(
        (screen.getByLabelText("Portfolio") as HTMLSelectElement).options.length,
      ).toBeGreaterThan(2),
    );
    await user.selectOptions(screen.getByLabelText("Portfolio"), "__new_portfolio__");
    await user.type(screen.getByLabelText("New portfolio identifier"), "direct_099");
    await fillPeriod(user);
    await user.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => expect(sent).toHaveLength(1));
    // The assertion is explicit, so the API can refuse a typo.
    expect(sent[0]).toMatchObject({ portfolio_id: "direct_099", new_portfolio: true });
  });
});

describe("choosing the book", () => {
  it("defaults to the funded book so existing behaviour is unchanged", async () => {
    const user = userEvent.setup();
    const { sent } = renderScreen();
    await chooseBook(user);
    await fillPeriod(user);
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0].dataset).toBe("funded");
  });

  it("sends the pipeline dataset when the operator chooses it", async () => {
    const user = userEvent.setup();
    const { sent } = renderScreen();
    await chooseBook(user);
    await user.click(within(bookOption("pipeline")).getByRole("radio"));
    await fillPeriod(user);
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0].dataset).toBe("pipeline");
  });

  it("offers only the books this portfolio actually delivers", async () => {
    const user = userEvent.setup();
    renderScreen();
    // acquired_002 delivers a funded book only.
    await chooseBook(user, "Alpine Capital", "acquired_002");
    await waitFor(() =>
      expect(
        (within(bookOption("pipeline")).getByRole("radio") as HTMLInputElement).disabled,
      ).toBe(true),
    );
    expect(within(bookOption("pipeline")).getByText(/not registered for this portfolio/))
      .toBeInTheDocument();
    expect(
      (within(bookOption("funded")).getByRole("radio") as HTMLInputElement).disabled,
    ).toBe(false);
  });

  it("says plainly that funded and pipeline are separate deliveries", async () => {
    renderScreen();
    expect(
      screen.getByText(/separate deliveries with their own files, workflow and approval/),
    ).toBeInTheDocument();
    expect(screen.getByText(/Prepare the monthly delivery for funded loans/)).toBeInTheDocument();
    expect(screen.getByText(/Prepare the monthly delivery for pipeline cases/)).toBeInTheDocument();
  });
});

describe("what Trakt will prepare", () => {
  it("is derived from the book, not chosen by the operator", async () => {
    const user = userEvent.setup();
    const { sent } = renderScreen();
    await chooseBook(user, "Alpine Capital", "direct_001");

    // direct_001's funded book is regime-required.
    await waitFor(() =>
      expect(document.querySelector('[data-derived-outcome="mi_annex2"]')).not.toBeNull(),
    );
    expect(screen.getByText(/Trakt decides this from what this book is registered to report/))
      .toBeInTheDocument();
    // There is no control to override it.
    expect(screen.queryByRole("radio", { name: /ESMA Annex 2/ })).toBeNull();

    await fillPeriod(user);
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0].workflow_type).toBe("mi_annex2");
  });

  it("never asks for the regulatory annex on a pipeline delivery", async () => {
    const user = userEvent.setup();
    const { sent } = renderScreen();
    await chooseBook(user, "Alpine Capital", "direct_001");
    await user.click(within(bookOption("pipeline")).getByRole("radio"));

    await waitFor(() =>
      expect(document.querySelector('[data-derived-outcome="mi"]')).not.toBeNull(),
    );
    await fillPeriod(user);
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(sent).toHaveLength(1));
    // The combination the backend refuses must never be sent at all.
    expect(sent[0]).toMatchObject({ dataset: "pipeline", workflow_type: "mi" });
  });

  it("prepares management information only for a book with no regime record", async () => {
    const user = userEvent.setup();
    renderScreen();
    await chooseBook(user, "Alpine Capital", "acquired_002");
    await waitFor(() =>
      expect(document.querySelector('[data-derived-outcome="mi"]')).not.toBeNull(),
    );
  });
});
