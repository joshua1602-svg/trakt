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
  const clientSelect = screen.getByLabelText("Client") as HTMLSelectElement;
  await user.selectOptions(clientSelect, clientSelect.options[1].value);
  await user.type(screen.getByLabelText("Portfolio"), "European Growth");
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

  it("walks the operator through the six governed steps", () => {
    renderScreen();
    for (const heading of [
      copy.newWorkflow.outcomeHeading,
      copy.newWorkflow.bookHeading,
      copy.newWorkflow.detailsHeading,
      copy.newWorkflow.periodHeading,
    ]) {
      expect(screen.getByText(heading)).toBeInTheDocument();
    }
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

    // The chosen files are listed before anything is sent.
    const chosen = screen.getByText(copy.newWorkflow.chosenFiles).parentElement as HTMLElement;
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
    expect(within(confirm).getByText("European Growth")).toBeInTheDocument();
    expect(within(confirm).getByText("2026-06")).toBeInTheDocument();
    expect(within(confirm).getByText("Funded book")).toBeInTheDocument();
    // Nothing can be sent until files are chosen.
    expect(screen.getByRole("button", { name: copy.newWorkflow.uploadButton })).toBeDisabled();
  });
});
