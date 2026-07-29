import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { MockOpsClient } from "@/api/MockOpsClient";
import type { OpsClient } from "@/api/OpsClient";
import type { Dashboard } from "@/api/types";
import { OpsClientProvider } from "@/api/context";
import { ToastProvider } from "@/components/Toast";
import { HomeScreen } from "./Home";

function renderHome(client: OpsClient) {
  return render(
    <OpsClientProvider client={client}>
      <ToastProvider>
        <MemoryRouter>
          <HomeScreen />
        </MemoryRouter>
      </ToastProvider>
    </OpsClientProvider>,
  );
}

describe("Home", () => {
  it("shows the five tiles and the attention list from the mock data", async () => {
    renderHome(new MockOpsClient(0));

    expect(await screen.findByText("New deliveries")).toBeInTheDocument();
    expect(screen.getAllByText("Blocked").length).toBeGreaterThan(0);
    expect(screen.getByText("Ready to publish")).toBeInTheDocument();
    expect(screen.getAllByText("Needs your attention").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Recently published").length).toBeGreaterThan(0);

    // Attention rows link to workflows and show decision counts.
    expect(screen.getByText("Alpine Capital · European Growth")).toBeInTheDocument();
    expect(screen.getByText("2 decisions")).toBeInTheDocument();
    expect(
      screen.getByText("Alpine Capital · European Growth").closest("a"),
    ).toHaveAttribute("href", "/workflows/wf-1001");

    // Recently published list.
    expect(screen.getAllByText("Cedar Rock Advisors").length).toBeGreaterThan(0);
  });

  it("shows the empty state when nothing needs attention", async () => {
    const emptyDashboard: Dashboard = {
      tiles: {
        new_deliveries: 0,
        needs_attention: 0,
        blocked: 0,
        ready_to_publish: 0,
        recently_published: 0,
      },
      needs_attention: [],
      recently_published: [],
    };
    const stub = { getDashboard: async () => emptyDashboard } as unknown as OpsClient;
    renderHome(stub);

    expect(await screen.findByText("Nothing needs your attention.")).toBeInTheDocument();
  });
});
