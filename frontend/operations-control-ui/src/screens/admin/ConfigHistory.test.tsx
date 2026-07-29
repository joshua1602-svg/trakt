import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { copy } from "@/lib/copy";
import { ConfigHistoryScreen } from "./ConfigHistory";
import { mockClient, renderAdmin } from "./testUtils";

describe("configuration history", () => {
  it("shows the empty state before anything has been changed", async () => {
    renderAdmin(<ConfigHistoryScreen />, mockClient(), { route: "/admin/config/history" });
    expect(await screen.findByText(copy.admin.audit.empty)).toBeInTheDocument();
  });

  it("describes every lifecycle step in plain English, newest first", async () => {
    const client = mockClient();
    await client.createConfigDraft("regime", { notes: "annex refresh" });
    await client.validateConfigVersion("regime", 2);
    await client.activateConfigVersion("regime", 2);
    await client.rollbackConfig("regime", 1);

    renderAdmin(<ConfigHistoryScreen />, client, { route: "/admin/config/history" });

    await screen.findByText(copy.admin.audit.chainIntact);
    const events = [...document.querySelectorAll("li[data-event]")].map((li) =>
      li.getAttribute("data-event"),
    );
    expect(events).toEqual([
      "config_rolled_back",
      "config_activated",
      "config_validated",
      "config_draft_created",
    ]);

    expect(
      screen.getByText(/Future workflows will use this version. Existing workflows remain pinned./),
    ).toBeInTheDocument();
    expect(screen.getByText(/rolled back to v1/)).toBeInTheDocument();
    expect(screen.getByText(/The version it replaced has been kept./)).toBeInTheDocument();
    expect(screen.getByText(/A draft changes nothing until it is validated/)).toBeInTheDocument();
    expect(screen.getByText("annex refresh")).toBeInTheDocument();
  });

  it("marks a failed check as a failure", async () => {
    const client = mockClient();
    await client.createConfigDraft("system", {
      edits: { "config/system/run_context_policy.yaml": "not: [valid" },
    });
    await client.validateConfigVersion("system", 2);

    renderAdmin(<ConfigHistoryScreen />, client, { route: "/admin/config/history" });
    expect(await screen.findByText(/did not pass: 1 check failed/)).toBeInTheDocument();
    expect(screen.getByText(/cannot be activated until they are fixed/)).toBeInTheDocument();
  });
});
