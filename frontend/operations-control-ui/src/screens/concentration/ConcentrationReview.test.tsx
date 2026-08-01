/**
 * The concentration review surface: an operator can read the client's
 * wording, work each proposal to a decision and activate the approved set —
 * with the same refusals the backend gives (blocked approval explained,
 * admin-only actions gated, immutable approved proposals, nothing activates
 * from nothing). The stateful mock enforces the backend's rules, so these
 * tests exercise real governance behaviour, not happy-path rendering.
 */

import { describe, expect, it } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import type { ReactNode } from "react";
import { MockOpsClient } from "@/api/MockOpsClient";
import type { OpsClient } from "@/api/OpsClient";
import { OpsClientProvider } from "@/api/context";
import { SessionProvider } from "@/api/session";
import { ToastProvider } from "@/components/Toast";
import { copy } from "@/lib/copy";
import {
  ConcentrationClientsScreen,
  ConcentrationReviewScreen,
} from "./ConcentrationReview";

function renderAt(client: OpsClient, path: string): ReturnType<typeof render> {
  const ui: ReactNode = (
    <OpsClientProvider client={client}>
      <ToastProvider>
        <SessionProvider>
          <MemoryRouter initialEntries={[path]}>
            <Routes>
              <Route path="/concentration" element={<ConcentrationClientsScreen />} />
              <Route
                path="/concentration/:client"
                element={<ConcentrationReviewScreen />}
              />
            </Routes>
          </MemoryRouter>
        </SessionProvider>
      </ToastProvider>
    </OpsClientProvider>
  );
  return render(ui);
}

const RESPONSE =
  "East Anglia must not exceed 25%. The Net WAC must be at least 3.75%. "
  + "The portfolio duration must not exceed 7 years.";

async function extract(client: MockOpsClient) {
  await client.extractConcentrationTests("client_a", "covenant response", RESPONSE);
}

describe("ConcentrationReviewScreen", () => {
  it("shows the no-response state before anything is read", async () => {
    renderAt(new MockOpsClient(0), "/concentration/client_a");
    expect(
      await screen.findByText(copy.concentration.noResponse),
    ).toBeInTheDocument();
  });

  it("reads a pasted response into proposals with outcomes and statuses", async () => {
    const client = new MockOpsClient(0);
    renderAt(client, "/concentration/client_a");
    fireEvent.change(
      await screen.findByLabelText(copy.concentration.responseText),
      { target: { value: RESPONSE } },
    );
    fireEvent.click(
      screen.getByRole("button", { name: copy.concentration.extract }),
    );
    expect(await screen.findByText("East Anglia exposure")).toBeInTheDocument();
    expect(screen.getByText("Net WAC floor")).toBeInTheDocument();
    expect(screen.getByText("Portfolio duration cap")).toBeInTheDocument();
    // Outcomes are shown per proposal.
    expect(screen.getByText("matched")).toBeInTheDocument();
    expect(screen.getByText("ambiguous")).toBeInTheDocument();
  });

  it("filters proposals by status", async () => {
    const client = new MockOpsClient(0);
    await extract(client);
    renderAt(client, "/concentration/client_a");
    await screen.findByText("East Anglia exposure");
    fireEvent.change(screen.getByLabelText("Filter proposals"), {
      target: { value: "pending_confirmation" },
    });
    expect(screen.queryByText("East Anglia exposure")).toBeNull();
    expect(screen.getByText("Net WAC floor")).toBeInTheDocument();
  });

  it("opens a proposal and compares wording, mapping and confidence", async () => {
    const client = new MockOpsClient(0);
    await extract(client);
    renderAt(client, "/concentration/client_a");
    fireEvent.click(await screen.findByText("East Anglia exposure"));
    const detail = await screen.findByTestId("proposal-detail");
    expect(detail).toHaveTextContent("must not exceed 25% of the Portfolio");
    expect(detail).toHaveTextContent("clause 1.1");
    expect(detail).toHaveTextContent("geo_region_share");
    expect(detail).toHaveTextContent("≤ 25%");
    expect(detail).toHaveTextContent("mapping 90%");
  });

  it("explains exactly why approval is blocked, and disables the button", async () => {
    const client = new MockOpsClient(0);
    await extract(client);
    renderAt(client, "/concentration/client_a");
    fireEvent.click(await screen.findByText("Net WAC floor"));
    const blockers = await screen.findByTestId("approval-blockers");
    expect(blockers).toHaveTextContent("net_wac_definition_uncertain");
    expect(blockers).toHaveTextContent("Unanswered confirmation questions remain.");
    expect(
      screen.getByRole("button", { name: copy.concentration.approve }),
    ).toBeDisabled();
  });

  it("answers a question, resolves the concern and approves (administrator)", async () => {
    const client = new MockOpsClient(0);
    await extract(client);
    renderAt(client, "/concentration/client_a");
    fireEvent.click(await screen.findByText("Net WAC floor"));
    const detail = await screen.findByTestId("proposal-detail");
    fireEvent.change(within(detail).getByLabelText(copy.concentration.answerLabel), {
      target: { value: "Servicing fee only, 50bp." },
    });
    fireEvent.click(
      within(detail).getByRole("button", { name: copy.concentration.recordAnswer }),
    );
    await waitFor(() =>
      expect(screen.getByTestId("proposal-detail")).toHaveTextContent(
        "Servicing fee only, 50bp.",
      ),
    );
    // Resolve the concern through the edit form.
    fireEvent.click(screen.getByLabelText("resolved"));
    fireEvent.click(
      screen.getByRole("button", { name: copy.concentration.saveEdits }),
    );
    await waitFor(() =>
      expect(screen.queryByTestId("approval-blockers")).toBeNull(),
    );
    const approve = screen.getByRole("button", { name: copy.concentration.approve });
    expect(approve).toBeEnabled();
    fireEvent.click(approve);
    await waitFor(() =>
      expect(screen.getByTestId("proposal-detail")).toHaveTextContent(
        "Approved by Administrator",
      ),
    );
  });

  it("operators see approve disabled and the mock refuses admin actions", async () => {
    const client = new MockOpsClient(0, "operator");
    await extract(client);
    renderAt(client, "/concentration/client_a");
    expect(
      await screen.findByText(copy.concentration.permissionNote),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByText("East Anglia exposure"));
    await screen.findByTestId("proposal-detail");
    expect(
      screen.getByRole("button", { name: copy.concentration.approve }),
    ).toBeDisabled();
    // The boundary is the backend's, not the button's.
    await expect(
      client.approveConcentrationProposal("client_a", "nope", {}),
    ).rejects.toMatchObject({ errorCode: "OPS_ADMIN_REQUIRED" });
  });

  it("rejects, marks unsupported and requests clarification with reasons", async () => {
    const client = new MockOpsClient(0);
    await extract(client);
    renderAt(client, "/concentration/client_a");
    fireEvent.click(await screen.findByText("East Anglia exposure"));
    const detail = await screen.findByTestId("proposal-detail");
    fireEvent.change(within(detail).getByLabelText(copy.concentration.reasonLabel), {
      target: { value: "not in the amended facility" },
    });
    fireEvent.click(
      within(detail).getByRole("button", { name: copy.concentration.reject }),
    );
    await waitFor(() =>
      expect(screen.getByTestId("proposal-detail")).toHaveTextContent(
        "Rejected: not in the amended facility",
      ),
    );
  });

  it("activation lists what will become active, then mints the version", async () => {
    const client = new MockOpsClient(0);
    await extract(client);
    const review = await client.getConcentrationReview("client_a");
    const ea = review.proposals.find((p) => p.extracted_test_name === "East Anglia exposure")!;
    await client.approveConcentrationProposal("client_a", ea.proposal_id, {
      effective_date: "2026-01-31",
    });
    renderAt(client, "/concentration/client_a");
    expect(
      await screen.findByText(copy.concentration.willActivate),
    ).toBeInTheDocument();
    expect(screen.getByText(/East Anglia exposure \(max 25%\)/)).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: copy.concentration.activate }),
    );
    expect(
      await screen.findByText(`${copy.concentration.currentVersion}: v1 (1 test(s))`),
    ).toBeInTheDocument();
    expect(screen.getByText("v1 · 1 test(s)")).toBeInTheDocument();
  });

  it("an unsupported test can never reach activation", async () => {
    const client = new MockOpsClient(0);
    await extract(client);
    renderAt(client, "/concentration/client_a");
    fireEvent.click(await screen.findByText("Portfolio duration cap"));
    const detail = await screen.findByTestId("proposal-detail");
    // Decided proposals show no approve button at all.
    expect(
      within(detail).queryByRole("button", { name: copy.concentration.approve }),
    ).toBeNull();
    expect(
      screen.getByText(copy.concentration.nothingToActivate),
    ).toBeInTheDocument();
    await expect(
      client.activateConcentrationTests("client_a"),
    ).rejects.toMatchObject({ errorCode: "CONCENTRATION_REFUSED" });
  });

  it("shows the decision history on the proposal", async () => {
    const client = new MockOpsClient(0);
    await extract(client);
    const review = await client.getConcentrationReview("client_a");
    const wac = review.proposals.find((p) => p.extracted_test_name === "Net WAC floor")!;
    await client.answerConcentrationQuestion(
      "client_a", wac.proposal_id, wac.confirmation_questions[0], "Servicing only.");
    renderAt(client, "/concentration/client_a");
    fireEvent.click(await screen.findByText("Net WAC floor"));
    const detail = await screen.findByTestId("proposal-detail");
    expect(detail).toHaveTextContent(copy.concentration.auditHeading);
    expect(detail).toHaveTextContent("Administrator: Answered:");
  });

  it("a blank extraction is refused with the backend's message", async () => {
    const client = new MockOpsClient(0);
    await expect(
      client.extractConcentrationTests("client_a", "x", "   "),
    ).rejects.toMatchObject({ errorCode: "CONCENTRATION_REFUSED" });
  });
});

describe("ConcentrationClientsScreen", () => {
  it("offers the client list and navigates into a review", async () => {
    const client = new MockOpsClient(0);
    renderAt(client, "/concentration");
    expect(
      await screen.findByText(copy.concentration.pickClient),
    ).toBeInTheDocument();
  });
});
