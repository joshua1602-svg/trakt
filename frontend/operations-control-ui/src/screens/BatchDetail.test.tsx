import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { MockOpsClient } from "@/api/MockOpsClient";
import type { OpsClient } from "@/api/OpsClient";
import { OpsClientProvider } from "@/api/context";
import { ToastProvider } from "@/components/Toast";
import { BatchDetailScreen } from "./BatchDetail";

function renderBatch(client: OpsClient, batchId: string) {
  return render(
    <OpsClientProvider client={client}>
      <ToastProvider>
        <MemoryRouter initialEntries={[`/batches/${batchId}?client=Alpine%20Capital`]}>
          <Routes>
            <Route path="/batches/:id" element={<BatchDetailScreen />} />
          </Routes>
        </MemoryRouter>
      </ToastProvider>
    </OpsClientProvider>,
  );
}

describe("BatchDetail", () => {
  it("lists every expected input and offers the start button once everything is ready", async () => {
    const mock = new MockOpsClient(0);
    const batch = await mock.createBatch({
      client_id: "Alpine Capital",
      portfolio_id: "European Growth",
      reporting_date: "2026-06",
      workflow_type: "mi",
      dataset: "funded",
      auto_start_when_ready: false,
    });
    // Two uploads satisfy both required inputs. The browser sends file
    // content only — there is no location to name.
    await mock.uploadBatchFiles(batch.batch_id, [
      new File(["a"], "holdings-june.xlsx"),
      new File(["b"], "loan-tape-june.xlsx"),
    ]);

    renderBatch(mock, batch.batch_id);

    expect(await screen.findByText("Input pack")).toBeInTheDocument();

    // Every expected role renders with its state.
    expect(screen.getByText("Primary loan tape")).toBeInTheDocument();
    expect(screen.getByText("Holdings statement")).toBeInTheDocument();
    expect(screen.getByText("Prior period comparison")).toBeInTheDocument();
    expect(screen.getAllByText("✓ Received and recognised").length).toBe(2);
    expect(screen.getByText("Optional — not provided")).toBeInTheDocument();

    // Configuration and blocking decisions rows.
    expect(screen.getByText("Configuration")).toBeInTheDocument();
    expect(screen.getByText("✓ Ready")).toBeInTheDocument();
    expect(screen.getByText("Blocking decisions")).toBeInTheDocument();
    expect(screen.getByText("None")).toBeInTheDocument();

    // Ready + not auto-starting → the operator gets the start button.
    expect(screen.getByRole("button", { name: "Start onboarding" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add files" })).toBeInTheDocument();

    // The only way to add files is to send them; no path field exists.
    expect(screen.getByLabelText("Choose the files to send")).toHaveAttribute("type", "file");
    expect(screen.queryByLabelText(/where are the files/i)).not.toBeInTheDocument();
  });

  it("shows the waiting role, highlights the unclear file and links to its decisions", async () => {
    renderBatch(new MockOpsClient(0), "batch-7001");

    // The required role that has not arrived yet.
    expect(await screen.findByText("Waiting")).toBeInTheDocument();

    // The unclear file's own sentence is shown.
    expect(
      screen.getByText(/Trakt needs your confirmation before it can continue/),
    ).toBeInTheDocument();

    // Blocking decisions link points at the reviews list for this pack.
    expect(screen.getByText("1 decision").closest("a")).toHaveAttribute(
      "href",
      "/reviews?workflow=batch-7001",
    );

    // Not ready, so no start button.
    expect(screen.queryByRole("button", { name: "Start onboarding" })).not.toBeInTheDocument();
  });
});
