import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { MockOpsClient } from "@/api/MockOpsClient";
import type { CreateBatchInput } from "@/api/types";
import { OpsClientProvider } from "@/api/context";
import { copy } from "@/lib/copy";
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
  return { sent };
}

function bookOption(value: string) {
  const label = document.querySelector(`[data-dataset="${value}"]`);
  if (!label) throw new Error(`no book option for ${value}`);
  return label as HTMLElement;
}

async function fillDetails(user: ReturnType<typeof userEvent.setup>) {
  await waitFor(() => expect(screen.getByLabelText("Client")).toBeTruthy());
  const clientSelect = screen.getByLabelText("Client") as HTMLSelectElement;
  await user.selectOptions(clientSelect, clientSelect.options[1].value);
  await user.type(screen.getByLabelText("Portfolio"), "European Growth");
  const period = screen.getByLabelText("Reporting period") as HTMLInputElement;
  await user.clear(period);
  await user.type(period, "2026-06");
}

describe("NewWorkflow — choosing the book", () => {
  it("defaults to the funded book so existing behaviour is unchanged", async () => {
    const user = userEvent.setup();
    const { sent } = renderScreen();
    await fillDetails(user);
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0].dataset).toBe("funded");
    expect(sent[0].workflow_type).toBe("mi");
  });

  it("sends the pipeline dataset when the operator chooses it", async () => {
    const user = userEvent.setup();
    const { sent } = renderScreen();
    await user.click(within(bookOption("pipeline")).getByRole("radio"));
    await fillDetails(user);
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0].dataset).toBe("pipeline");
  });

  it("makes the regulatory annex unreachable for pipeline", async () => {
    const user = userEvent.setup();
    renderScreen();
    const annex = screen.getByRole("radio", {
      name: /ESMA Annex 2/,
    }) as HTMLInputElement;
    expect(annex.disabled).toBe(false);

    await user.click(within(bookOption("pipeline")).getByRole("radio"));
    expect(annex.disabled).toBe(true);
    expect(
      screen.getByText(/management information only, so the regulatory annex is not available/i),
    ).toBeTruthy();
  });

  it("falls back to MI if the annex was already selected when pipeline is chosen", async () => {
    const user = userEvent.setup();
    const { sent } = renderScreen();
    await user.click(screen.getByRole("radio", { name: /ESMA Annex 2/ }));
    await user.click(within(bookOption("pipeline")).getByRole("radio"));
    await fillDetails(user);
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(sent).toHaveLength(1));
    // The combination the backend refuses must never be sent at all.
    expect(sent[0]).toMatchObject({ dataset: "pipeline", workflow_type: "mi" });
  });

  it("re-enables the annex when the operator goes back to the funded book", async () => {
    const user = userEvent.setup();
    renderScreen();
    await user.click(within(bookOption("pipeline")).getByRole("radio"));
    await user.click(within(bookOption("funded")).getByRole("radio"));
    const annex = screen.getByRole("radio", { name: /ESMA Annex 2/ }) as HTMLInputElement;
    expect(annex.disabled).toBe(false);
    expect(
      screen.queryByText(/regulatory annex is not available/i),
    ).toBeNull();
  });

  it("explains each book in operator language, not dataset identifiers", async () => {
    renderScreen();
    expect(screen.getByText("Which book?")).toBeTruthy();
    expect(
      screen.getByText(/Loans already advanced\. Regulatory reporting is prepared from this book\./),
    ).toBeTruthy();
    expect(screen.getByText(/Cases not yet funded\./)).toBeTruthy();
  });
});

/**
 * Sending a delivery that is not monthly.
 *
 * TWO CONTROLS, ONE CONSEQUENCE
 *
 * `CreateBatch` has carried a frequency since the API door was fixed, and
 * `intake.create_batch` stores `frequency or BATCH_FREQUENCY_DEFAULT` — which
 * is `monthly`. This screen never sent one, so EVERY manually created delivery
 * landed under `/monthly/` whatever it actually was. The frequency is a path
 * segment and part of the pack key, so a weekly pipeline tape filed as monthly
 * is in the wrong folder under the wrong key.
 *
 * And the period was `<input type="month">`, which cannot express a pipeline
 * snapshot's own date (`2026-09-14`), an ISO week or a quarter — all of which
 * Trakt files under. Between them, the one screen an operator has could only
 * ever create a monthly funded delivery.
 *
 * The assertions are on the payload rather than the markup: what reaches the
 * server is what decides where the files go.
 */
describe("NewWorkflow — a delivery that is not monthly", () => {
  it("offers a frequency", async () => {
    renderScreen();
    await waitFor(() =>
      expect(screen.getByLabelText(copy.newWorkflow.frequencyLabel)).toBeTruthy(),
    );
  });

  it("sends the frequency the operator chose", async () => {
    const user = userEvent.setup();
    const { sent } = renderScreen();
    await user.click(within(bookOption("pipeline")).getByRole("radio"));
    await fillDetails(user);
    await user.selectOptions(
      screen.getByLabelText(copy.newWorkflow.frequencyLabel),
      "adhoc",
    );
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0].frequency).toBe("adhoc");
  });

  it("still sends monthly when nobody chooses", async () => {
    /* Unchanged for the funded pack, which is every existing caller. */
    const user = userEvent.setup();
    const { sent } = renderScreen();
    await fillDetails(user);
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0].frequency).toBe("monthly");
  });

  it("accepts a snapshot's own date, which a month picker cannot express", async () => {
    const user = userEvent.setup();
    const { sent } = renderScreen();
    await user.click(within(bookOption("pipeline")).getByRole("radio"));
    await waitFor(() => expect(screen.getByLabelText("Client")).toBeTruthy());
    const clientSelect = screen.getByLabelText("Client") as HTMLSelectElement;
    await user.selectOptions(clientSelect, clientSelect.options[1].value);
    await user.type(screen.getByLabelText("Portfolio"), "direct_001");
    await user.type(screen.getByLabelText("Reporting period"), "2026-09-14");
    await user.selectOptions(
      screen.getByLabelText(copy.newWorkflow.frequencyLabel),
      "adhoc",
    );
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0]).toMatchObject({
      reporting_date: "2026-09-14",
      frequency: "adhoc",
      dataset: "pipeline",
    });
  });

  it("the period box is not a month picker", async () => {
    /* `type="month"` silently refuses a day or a week, so the control itself
       is the constraint — no payload assertion can catch it. */
    renderScreen();
    await waitFor(() => expect(screen.getByLabelText("Reporting period")).toBeTruthy());
    const period = screen.getByLabelText("Reporting period") as HTMLInputElement;
    expect(period.type).not.toBe("month");
  });

  it("will not send an empty period", async () => {
    const user = userEvent.setup();
    const { sent } = renderScreen();
    await waitFor(() => expect(screen.getByLabelText("Client")).toBeTruthy());
    const clientSelect = screen.getByLabelText("Client") as HTMLSelectElement;
    await user.selectOptions(clientSelect, clientSelect.options[1].value);
    await user.type(screen.getByLabelText("Portfolio"), "direct_001");
    await user.type(screen.getByLabelText("Reporting period"), "   ");
    expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
    expect(sent).toHaveLength(0);
  });
});
