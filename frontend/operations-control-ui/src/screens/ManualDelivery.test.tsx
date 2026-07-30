import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { MockOpsClient } from "@/api/MockOpsClient";
import { OpsClientProvider } from "@/api/context";
import { ToastProvider } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { NewWorkflowScreen } from "./NewWorkflow";

function renderScreen(client = new MockOpsClient(0)) {
  render(
    <OpsClientProvider client={client}>
      <ToastProvider>
        <MemoryRouter initialEntries={["/new"]}>
          <Routes>
            <Route path="/new" element={<NewWorkflowScreen />} />
            <Route path="/workflows/:id" element={<p>workflow page</p>} />
            <Route path="/batches/:id" element={<p>input pack page</p>} />
          </Routes>
        </MemoryRouter>
      </ToastProvider>
    </OpsClientProvider>,
  );
  return client;
}

async function fillDetails(user: ReturnType<typeof userEvent.setup>) {
  await waitFor(() => expect(screen.getByLabelText("Client")).toBeTruthy());
  await user.selectOptions(screen.getByLabelText("Client"), "Alpine Capital");
  await waitFor(() =>
    expect(
      (screen.getByLabelText("Portfolio") as HTMLSelectElement).options.length,
    ).toBeGreaterThan(2),
  );
  await user.selectOptions(screen.getByLabelText("Portfolio"), "direct_001");
  const period = screen.getByLabelText("Reporting period") as HTMLInputElement;
  await user.clear(period);
  await user.type(period, "2026-06");
  await user.click(screen.getByRole("button", { name: "Continue" }));
}

describe("manual delivery", () => {
  it("names itself for what it is, and says when to use it", () => {
    renderScreen();
    expect(screen.getByText("Create a manual delivery")).toBeInTheDocument();
    expect(
      screen.getByText(/files are not arriving through the normal automated intake process/),
    ).toBeInTheDocument();
  });

  it("asks for the governing facts first and derives the outcome after", () => {
    renderScreen();
    // The order is the dependency order: who, which book, what Trakt will
    // prepare (derived), which period.
    const headings = Array.from(document.querySelectorAll("section h2")).map(
      (h) => h.textContent,
    );
    expect(headings).toEqual([
      copy.newWorkflow.detailsHeading,
      copy.newWorkflow.bookHeading,
      copy.newWorkflow.outcomeHeading,
      copy.newWorkflow.periodHeading,
    ]);
  });

  it("never offers a place to type a storage location", async () => {
    const user = userEvent.setup();
    renderScreen();
    await fillDetails(user);

    const upload = await screen.findByLabelText(copy.newWorkflow.uploadLabel);
    expect(upload).toHaveAttribute("type", "file");
    expect(upload).toHaveAttribute("multiple");
    // The old free-text location field is gone, not merely hidden.
    expect(screen.queryByLabelText(/where are the files/i)).toBeNull();
    expect(
      document.querySelector('input[type="text"][id="folder"]'),
    ).toBeNull();
  });

  it("sends the files and lands on the resulting delivery workflow", async () => {
    const user = userEvent.setup();
    renderScreen();
    await fillDetails(user);

    const upload = await screen.findByLabelText(copy.newWorkflow.uploadLabel);
    await user.upload(upload, [
      new File(["a"], "holdings-june.xlsx"),
      new File(["b"], "loan-tape-june.xlsx"),
    ]);

    // The chosen files are listed before anything is sent. (The confirm step
    // also counts them, so scope to the upload step's own heading.)
    const chosen = screen.getAllByText(copy.newWorkflow.chosenFiles)[0]
      .parentElement as HTMLElement;
    expect(within(chosen).getByText("holdings-june.xlsx")).toBeInTheDocument();
    expect(within(chosen).getByText("loan-tape-june.xlsx")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: copy.newWorkflow.uploadButton }));

    // The existing intake path opened a workflow, so that is where we land.
    expect(await screen.findByText("workflow page")).toBeInTheDocument();
  });

  it("confirms what is being submitted before sending", async () => {
    const user = userEvent.setup();
    renderScreen();
    await fillDetails(user);

    const confirm = (await screen.findByText(copy.newWorkflow.confirmHeading)).closest(
      "section",
    ) as HTMLElement;
    // Everything that governs the delivery, restated before it is sent.
    expect(within(confirm).getByText("Alpine Capital")).toBeInTheDocument();
    expect(within(confirm).getByText("Alpine Direct Originations")).toBeInTheDocument();
    expect(within(confirm).getByText("direct_001")).toBeInTheDocument();
    expect(within(confirm).getByText("Equity Release")).toBeInTheDocument();
    expect(within(confirm).getByText("2026-06")).toBeInTheDocument();
    expect(within(confirm).getByText("Funded book")).toBeInTheDocument();
    expect(
      within(confirm).getByText(copy.newWorkflow.outcomeAnnex),
    ).toBeInTheDocument();
    // Nothing can be sent until files are chosen.
    expect(screen.getByRole("button", { name: copy.newWorkflow.uploadButton })).toBeDisabled();
  });
});
