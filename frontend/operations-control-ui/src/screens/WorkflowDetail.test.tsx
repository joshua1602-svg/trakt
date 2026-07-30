import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { MockOpsClient } from "@/api/MockOpsClient";
import { OpsClientProvider } from "@/api/context";
import { ToastProvider } from "@/components/Toast";
import { WorkflowDetailScreen } from "./WorkflowDetail";

function renderWorkflow(client: MockOpsClient, entry: string) {
  render(
    <OpsClientProvider client={client}>
      <ToastProvider>
        <MemoryRouter initialEntries={[entry]}>
          <Routes>
            <Route path="/workflows/:id" element={<WorkflowDetailScreen />} />
          </Routes>
        </MemoryRouter>
      </ToastProvider>
    </OpsClientProvider>,
  );
}

function step(key: string): HTMLElement {
  const el = document.querySelector(`li[data-step="${key}"]`);
  if (!el) throw new Error(`no step ${key}`);
  return el as HTMLElement;
}

/** Waits for the case file to load, then returns the approval step. */
async function approvalStep(): Promise<HTMLElement> {
  await waitFor(() =>
    expect(document.querySelector('li[data-step="publication_approval"]')).not.toBeNull(),
  );
  return step("publication_approval");
}

const ALL_STEPS = [
  "files_received",
  "delivery_identified",
  "configuration_selected",
  "data_assessed",
  "issues_reviewed",
  "publication_approval",
  "published",
];

describe("delivery workflow", () => {
  it("shows every governed step, in order, with a state each", async () => {
    renderWorkflow(new MockOpsClient(0), "/workflows/wf-1002");

    await waitFor(() => expect(document.querySelector("li[data-step]")).not.toBeNull());
    const keys = Array.from(document.querySelectorAll("li[data-step]")).map((el) =>
      el.getAttribute("data-step"),
    );
    expect(keys).toEqual(ALL_STEPS);

    // Every step declares a state; none is left blank.
    for (const key of ALL_STEPS) {
      expect(step(key).getAttribute("data-step-status")).toBeTruthy();
    }
  });

  it("opens at the step the delivery is actually at", async () => {
    renderWorkflow(new MockOpsClient(0), "/workflows/wf-1002");

    // wf-1002 is awaiting publication, so that is the step in progress…
    await waitFor(() =>
      expect(step("publication_approval").getAttribute("data-step-status")).toBe("current"),
    );
    expect(
      step("publication_approval").querySelector("button")?.getAttribute("aria-expanded"),
    ).toBe("true");
    // …and it is the step that is open.
    expect(within(step("publication_approval")).getByText("Ready to publish")).toBeInTheDocument();
  });

  it("keeps completed steps visible and expandable", async () => {
    const user = userEvent.setup();
    renderWorkflow(new MockOpsClient(0), "/workflows/wf-1002");

    await waitFor(() =>
      expect(step("delivery_identified").getAttribute("data-step-status")).toBe("complete"),
    );
    const header = within(step("delivery_identified")).getByRole("button");
    expect(header).toHaveAttribute("aria-expanded", "false");

    await user.click(header);
    expect(header).toHaveAttribute("aria-expanded", "true");
    // The evidence for a step already behind us is still there.
    expect(within(step("delivery_identified")).getByText("Client")).toBeInTheDocument();
    expect(within(step("delivery_identified")).getByText("Reporting period")).toBeInTheDocument();
  });

  it("shows the received files with their arrival detail", async () => {
    const user = userEvent.setup();
    renderWorkflow(new MockOpsClient(0), "/workflows/wf-1002");

    await waitFor(() => expect(document.querySelector("li[data-step]")).not.toBeNull());
    await user.click(within(step("files_received")).getByRole("button"));
    const received = step("files_received");
    expect(within(received).getByText("Source location")).toBeInTheDocument();
    expect(within(received).getByText(/files received\. The pack was complete\./)).toBeInTheDocument();
  });

  it("marks a blocked delivery as blocked at the step that is stuck", async () => {
    renderWorkflow(new MockOpsClient(0), "/workflows/wf-1004");

    await waitFor(() =>
      expect(step("data_assessed").getAttribute("data-step-status")).toBe("blocked"),
    );
    expect(
      step("data_assessed").querySelector("button")?.getAttribute("aria-expanded"),
    ).toBe("true");
  });
});

describe("publication approval inside the workflow", () => {
  it("shows the evidence, the question and the consequence before confirming", async () => {
    renderWorkflow(new MockOpsClient(0), "/workflows/wf-1002");

    const approval = await approvalStep();
    expect(within(approval).getByText("Ready to publish")).toBeInTheDocument();
    expect(
      within(approval).getByText("Publish this delivery as the latest official version?"),
    ).toBeInTheDocument();
    expect(within(approval).getByText(/0 blocking issues/)).toBeInTheDocument();
    expect(
      within(approval).getByText(/as the latest official version\..*applies only to this delivery/s),
    ).toBeInTheDocument();
  });

  it("defaults the remembered scope to this delivery only", async () => {
    renderWorkflow(new MockOpsClient(0), "/workflows/wf-1002");

    const approval = await approvalStep();
    expect(
      within(approval).getByText("Should Trakt remember this decision for future deliveries?"),
    ).toBeInTheDocument();
    const chosen = within(approval).getByRole("radio", {
      name: /No — this delivery only/,
    }) as HTMLInputElement;
    expect(chosen.checked).toBe(true);
  });

  it("never offers a platform-wide scope on a single delivery", async () => {
    renderWorkflow(new MockOpsClient(0), "/workflows/wf-1002");

    const approval = await approvalStep();
    expect(within(approval).getAllByRole("radio").length).toBe(3);
    expect(within(approval).queryByText(/All of Trakt/)).toBeNull();
    expect(within(approval).queryByRole("radio", { name: /every client/i })).toBeNull();
  });

  it("sends the scope the operator chose", async () => {
    const user = userEvent.setup();
    const client = new MockOpsClient(0);
    const publish = vi.spyOn(client, "publishWorkflow");
    renderWorkflow(client, "/workflows/wf-1002");

    const approval = await approvalStep();
    await user.click(
      within(approval).getByRole("radio", { name: /future deliveries for this portfolio/i }),
    );
    // The consequence updates with the scope, before anything is confirmed.
    expect(within(approval).getByText(/also apply to future deliveries for/)).toBeInTheDocument();

    await user.click(within(approval).getByRole("button", { name: "Approve and publish" }));
    await user.click(await screen.findByRole("button", { name: "Publish this delivery" }));

    await waitFor(() => expect(publish).toHaveBeenCalledWith("wf-1002", "portfolio"));
  });
});

describe("workflow layout on a small screen", () => {
  it("lays the step evidence out one column first and never fixes a width", async () => {
    const user = userEvent.setup();
    renderWorkflow(new MockOpsClient(0), "/workflows/wf-1002");

    await waitFor(() => expect(document.querySelector("li[data-step]")).not.toBeNull());
    await user.click(within(step("delivery_identified")).getByRole("button"));

    const facts = step("delivery_identified").querySelector("dl");
    expect(facts?.className).toContain("grid-cols-1");
    expect(facts?.className).toContain("sm:grid-cols-2");

    // Step headers pad tighter on small screens and widen from `sm` up.
    const header = within(step("delivery_identified")).getByRole("button");
    expect(header.className).toContain("px-4");
    expect(header.className).toContain("sm:px-6");
  });
});
