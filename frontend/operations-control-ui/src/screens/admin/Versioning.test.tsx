import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { copy } from "@/lib/copy";
import { ConfigSystemScreen } from "./ConfigSystem";
import { mockClient, renderAdmin } from "./testUtils";

async function renderSystem(client = mockClient()) {
  renderAdmin(<ConfigSystemScreen />, client, { route: "/admin/config/system" });
  await screen.findByText(copy.admin.system.title);
  return client;
}

/** The section tab strip, so a tab is never confused with an action button. */
function sectionTab(name: string): HTMLElement {
  const nav = screen.getByRole("navigation", { name: copy.admin.sections.summary });
  return within(nav).getByRole("button", { name });
}

/** A client with three versions: v1 (superseded), v2 (active), v3 (draft). */
async function clientWithHistory() {
  const client = mockClient();
  await client.createConfigDraft("system", {
    edits: { "config/system/run_context_policy.yaml": "version: 2\nallow_partial: false\n" },
  });
  await client.validateConfigVersion("system", 2);
  await client.activateConfigVersion("system", 2);
  await client.createConfigDraft("system", {
    edits: { "config/system/run_context_policy.yaml": "version: 3\nallow_partial: true\n" },
  });
  return client;
}

describe("versions", () => {
  it("lists historical versions and highlights the active one", async () => {
    const user = userEvent.setup();
    await renderSystem(await clientWithHistory());

    await user.click(sectionTab(copy.admin.sections.history));

    const table = await screen.findByRole("table");
    expect(within(table).getByText("v1")).toBeInTheDocument();
    expect(within(table).getByText("v2")).toBeInTheDocument();
    expect(within(table).getByText("v3")).toBeInTheDocument();

    const activeRow = table.querySelector('tr[data-version="2"]');
    expect(activeRow).toHaveAttribute("data-active", "true");
    expect(table.querySelector('tr[data-version="1"]')).not.toHaveAttribute("data-active");
    expect(within(activeRow as HTMLElement).getByText(copy.statusLabels.active)).toBeInTheDocument();
    expect(
      within(table.querySelector('tr[data-version="1"]') as HTMLElement).getByText(
        copy.statusLabels.superseded,
      ),
    ).toBeInTheDocument();
  });

  it("compares two versions and groups the changes", async () => {
    const user = userEvent.setup();
    await renderSystem(await clientWithHistory());

    await user.click(sectionTab(copy.admin.sections.compare));

    // Defaults to active versus draft.
    expect(await screen.findByText("v2 → v3", { exact: false })).toBeInTheDocument();
    const changedTile = document.querySelector('[data-summary="changed"]') as HTMLElement;
    expect(within(changedTile).getByText("1")).toBeInTheDocument();
    expect(
      within(document.querySelector('[data-summary="added"]') as HTMLElement).getByText("0"),
    ).toBeInTheDocument();

    // The changed part opens to a line-by-line comparison.
    await user.click(screen.getByRole("button", { name: /Run policy/ }));
    expect(screen.getByText("-version: 2")).toBeInTheDocument();
    expect(screen.getByText("+version: 3")).toBeInTheDocument();
  });

  it("says so when two versions are identical", async () => {
    const user = userEvent.setup();
    await renderSystem();

    await user.click(sectionTab(copy.admin.sections.compare));
    expect(await screen.findByText(copy.admin.compare.identical)).toBeInTheDocument();
  });

  it("inspects a part read-only and never offers to edit it", async () => {
    const user = userEvent.setup();
    await renderSystem();

    await user.click(sectionTab(copy.admin.sections.files));
    expect(await screen.findByText(copy.admin.files.readOnly)).toBeInTheDocument();
    expect(screen.getByText("Field registry")).toBeInTheDocument();

    const registryRow = screen.getByText("Field registry").closest("div")!.parentElement!;
    await user.click(within(registryRow).getByRole("button", { name: copy.admin.actions.viewPart }));
    expect(await screen.findByText(/loan_identifier/)).toBeInTheDocument();
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: copy.common.save })).not.toBeInTheDocument();
  });

  it("shows which workflows are pinned to the active version", async () => {
    const user = userEvent.setup();
    await renderSystem();

    await user.click(sectionTab(copy.admin.sections.impact));

    expect(await screen.findByText(copy.admin.impact.heading)).toBeInTheDocument();
    expect(screen.getByText(copy.admin.impact.clients)).toBeInTheDocument();
    expect(screen.getByText(copy.admin.impact.running)).toBeInTheDocument();
    expect(
      screen.getAllByText(copy.admin.activate.pinnedNote).length,
    ).toBeGreaterThan(0);
    expect(screen.getByText("Alpine Capital · European Growth")).toBeInTheDocument();
  });

  it("shows what the package depends on", async () => {
    const user = userEvent.setup();
    await renderSystem();

    await user.click(sectionTab(copy.admin.sections.dependencies));
    expect(await screen.findByText("Client configuration")).toBeInTheDocument();
    expect(screen.getByText(/layered on top of this package/)).toBeInTheDocument();
  });
});
