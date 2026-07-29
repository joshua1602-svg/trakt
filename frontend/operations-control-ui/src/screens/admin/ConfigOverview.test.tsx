import { screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ConfigOverview } from "@/api/adminTypes";
import type { OpsClient } from "@/api/OpsClient";
import { copy } from "@/lib/copy";
import { ConfigOverviewScreen } from "./ConfigOverview";
import { mockClient, renderAdmin } from "./testUtils";

const ADMIN_PRINCIPAL = {
  name: "Administrator",
  role: "admin",
  is_admin: true,
  all_clients: true,
  clients: [],
};

function stubWith(overview: ConfigOverview): OpsClient {
  return {
    getPrincipal: async () => ADMIN_PRINCIPAL,
    getConfigOverview: async () => overview,
  } as unknown as OpsClient;
}

describe("configuration overview", () => {
  it("shows the active version of every package layer", async () => {
    renderAdmin(<ConfigOverviewScreen />);

    expect(await screen.findByText("System package")).toBeInTheDocument();
    expect(screen.getAllByText("Asset packages").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Regime packages").length).toBeGreaterThan(0);
    // Each card states its active version.
    expect(screen.getAllByText("v1").length).toBeGreaterThanOrEqual(3);
    expect(screen.getAllByText("Active").length).toBeGreaterThan(0);
  });

  it("lists the asset and regime packages with their compatibility", async () => {
    renderAdmin(<ConfigOverviewScreen />);

    expect((await screen.findAllByText("Equity Release")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("ESMA Annex 2").length).toBeGreaterThan(0);
    expect(screen.getByText("ESMA · Annex 2")).toBeInTheDocument();
    expect(screen.getAllByText(copy.admin.compatibility.supported).length).toBeGreaterThan(0);
  });

  it("shows a draft awaiting action once one is created", async () => {
    const client = mockClient();
    await client.createConfigDraft("system", { notes: "tidy up" });
    renderAdmin(<ConfigOverviewScreen />, client);

    expect(await screen.findByText(copy.admin.overview.draftsAwaiting)).toBeInTheDocument();
    expect(screen.getByText(/System package v2 · Administrator/)).toBeInTheDocument();
    // The draft is marked as not yet checked, so it cannot be mistaken for valid.
    const draftRow = screen.getByText(/System package v2 · Administrator/).closest("a");
    expect(draftRow).not.toBeNull();
    expect(within(draftRow!).getByText(copy.statusLabels.not_checked)).toBeInTheDocument();
  });

  it("shows a warning returned by the backend without hiding it", async () => {
    renderAdmin(<ConfigOverviewScreen />);
    expect(
      await screen.findByText(/uses a draft regulatory schema/),
    ).toBeInTheDocument();
  });

  it("shows the empty state when nothing needs an administrator", async () => {
    const client = mockClient();
    const overview = await client.getConfigOverview();
    overview.attention = {
      drafts_awaiting_action: [],
      validation_failures: [],
      packages_requiring_review: [],
      compatibility_issues: [],
    };
    overview.recent_activations = [];
    overview.recent_rollbacks = [];
    renderAdmin(<ConfigOverviewScreen />, stubWith(overview));

    expect(await screen.findByText(copy.admin.overview.noAttention)).toBeInTheDocument();
    expect(screen.getAllByText(copy.admin.overview.noRecent).length).toBe(2);
  });

  it("shows a failed check on the overview", async () => {
    const client = mockClient();
    await client.createConfigDraft("system", {
      edits: { "config/system/run_context_policy.yaml": "not: [valid" },
    });
    await client.validateConfigVersion("system", 2);
    renderAdmin(<ConfigOverviewScreen />, client);

    expect(await screen.findByText(copy.admin.overview.validationFailures)).toBeInTheDocument();
    expect(
      screen.getByText(`System package v2 — ${copy.admin.validation.fail}`),
    ).toBeInTheDocument();
  });
});
