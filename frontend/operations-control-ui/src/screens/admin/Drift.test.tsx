import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { copy } from "@/lib/copy";
import { PackageWorkspace } from "./PackageWorkspace";
import { mockClient, renderAdmin } from "./testUtils";

/**
 * A package layer is taken from the repository ONCE, when it is first used,
 * and never again. So a later deployment can carry a changed configuration
 * file while the platform goes on using the version it snapshotted — and
 * until now nothing on any screen said so. The active-version panel was
 * truthful, the file list was truthful, and the whole was misleading.
 *
 * These tests are about the one thing an administrator has to be able to see:
 * that what is deployed is NOT what is in force, and what to do about it.
 */

const GEOGRAPHY = "config/asset/mi_geography.yaml";

async function renderAssets(client = mockClient()) {
  renderAdmin(
    <PackageWorkspace
      layer="asset"
      title={copy.admin.assets.title}
      subtitle={copy.admin.assets.subtitle}
    />,
    client,
    { route: "/admin/config/assets" },
  );
  await screen.findByText(copy.admin.assets.title);
  return client;
}

/** A deployment whose geography file differs from the version in force. */
function clientWithALaterDeployment() {
  const client = mockClient();
  client.setDeployedConfigFiles("asset", {
    [GEOGRAPHY]: "primary_basis_by_asset_class:\n  equity_release: collateral\n",
  });
  return client;
}

describe("what this deployment carries", () => {
  it("says nothing when the deployed files are the version in force", async () => {
    await renderAssets();

    expect(screen.queryByText(copy.admin.drift.heading)).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: copy.admin.drift.adopt }),
    ).not.toBeInTheDocument();
  });

  it("says which file differs, and that the deployed one is not in force", async () => {
    await renderAssets(clientWithALaterDeployment());

    await screen.findByText(copy.admin.drift.heading);
    const sentence = screen.getByText(/is not in force/i);
    expect(sentence).toHaveTextContent(/geography/i);
    expect(sentence).toHaveTextContent(/version 1/i);
  });

  it("explains that a package is taken from the repository only once", async () => {
    await renderAssets(clientWithALaterDeployment());

    expect(await screen.findByText(copy.admin.drift.explain)).toBeInTheDocument();
  });

  it("drafts a version from the deployed files, leaving the active one alone", async () => {
    const user = userEvent.setup();
    const client = await renderAssets(clientWithALaterDeployment());
    const before = await client.getConfigOverview();
    expect(before.layers.asset.draft).toBeNull();

    await user.click(await screen.findByRole("button", { name: copy.admin.drift.adopt }));
    await screen.findByText(copy.admin.drift.adopted);

    const after = await client.getConfigOverview();
    expect(after.layers.asset.draft?.version).toBe(before.layers.asset.active_version + 1);
    expect(after.layers.asset.active_version).toBe(before.layers.asset.active_version);
  });

  it("holds the drafted version back until it is checked and activated", async () => {
    const user = userEvent.setup();
    const client = await renderAssets(clientWithALaterDeployment());

    await user.click(await screen.findByRole("button", { name: copy.admin.drift.adopt }));
    await screen.findByText(copy.admin.drift.adopted);

    const after = await client.getConfigOverview();
    expect(after.layers.asset.draft?.validated).toBe(false);
    expect(after.layers.asset.drift.differs).toBe(true);
  });
});
