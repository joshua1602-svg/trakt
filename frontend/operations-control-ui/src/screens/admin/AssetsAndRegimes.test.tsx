import { screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ConfigCatalogue } from "@/api/adminTypes";
import type { OpsClient } from "@/api/OpsClient";
import { copy } from "@/lib/copy";
import { ConfigAssetsScreen } from "./ConfigAssets";
import { ConfigRegimesScreen } from "./ConfigRegimes";
import { mockClient, renderAdmin } from "./testUtils";

/** Waits for a package card and returns it, so repeated labels elsewhere on
 *  the page (the compatibility table, the workspace below) cannot be matched
 *  by accident. */
async function card(selector: string): Promise<HTMLElement> {
  await waitFor(() => expect(document.querySelector(selector)).not.toBeNull());
  return document.querySelector(selector) as HTMLElement;
}

describe("asset configuration", () => {
  it("shows each asset package with its version and supported regimes", async () => {
    renderAdmin(<ConfigAssetsScreen />, mockClient(), { route: "/admin/config/assets" });

    const asset = await card('article[data-asset="equity_release"]');
    expect(within(asset).getByText("Equity Release")).toBeInTheDocument();
    expect(within(asset).getByText("equity_release")).toBeInTheDocument();
    expect(within(asset).getByText("v1")).toBeInTheDocument();
    expect(within(asset).getByText(copy.statusLabels.active)).toBeInTheDocument();
    expect(within(asset).getByText(copy.statusLabels.valid)).toBeInTheDocument();
    expect(within(asset).getByText("ESMA Annex 2")).toBeInTheDocument();
    expect(within(asset).getByText(copy.admin.assets.supportedRegimes)).toBeInTheDocument();
  });

  it("shows the source semantics, defaults and dependencies of an asset", async () => {
    renderAdmin(<ConfigAssetsScreen />, mockClient(), { route: "/admin/config/assets" });

    const asset = await card('article[data-asset="equity_release"]');
    expect(within(asset).getByText(copy.admin.assets.profiles)).toBeInTheDocument();
    expect(within(asset).getByText(copy.admin.assets.mappingDefaults)).toBeInTheDocument();
    expect(within(asset).getByText("18 settings")).toBeInTheDocument();
    expect(within(asset).getByText("System package")).toBeInTheDocument();
  });

  it("keeps client and portfolio configuration out of the administrator area", async () => {
    renderAdmin(<ConfigAssetsScreen />, mockClient(), { route: "/admin/config/assets" });
    expect(await screen.findByText(copy.admin.assets.clientNote)).toBeInTheDocument();
  });

  it("shows the compatibility matrix with only declared relationships", async () => {
    renderAdmin(<ConfigAssetsScreen />, mockClient(), { route: "/admin/config/assets" });

    await screen.findByText(copy.admin.compatibility.heading);
    const supported = document.querySelector('td[data-supported="true"]');
    expect(supported).not.toBeNull();
    expect(within(supported as HTMLElement).getByText(copy.admin.compatibility.supported))
      .toBeInTheDocument();
    // Nothing is described as "planned" — that is not a backend-declared state.
    expect(screen.queryByText(/planned/i)).not.toBeInTheDocument();
  });

  it("shows a compatibility blocker returned by the backend", async () => {
    const catalogue: ConfigCatalogue = await mockClient().getConfigCatalogue();
    catalogue.issues = [
      {
        layer: "asset",
        asset: "consumer_lending",
        regime: "ESMA_Annex9",
        message:
          "Consumer Lending is set up to report under Annex 9, but that regulatory package is not configured.",
      },
    ];
    const stub = {
      getPrincipal: async () => ({
        name: "Administrator",
        role: "admin",
        is_admin: true,
        all_clients: true,
        clients: [],
      }),
      getConfigCatalogue: async () => catalogue,
      getConfigOverview: async () => {
        throw new Error("not needed for this test");
      },
    } as unknown as OpsClient;

    renderAdmin(<ConfigAssetsScreen />, stub, { route: "/admin/config/assets" });
    expect(
      await screen.findByText(/that regulatory package is not configured/),
    ).toBeInTheDocument();
  });
});

describe("regime configuration", () => {
  it("shows the regulator, annex, schema and policies of a regime package", async () => {
    renderAdmin(<ConfigRegimesScreen />, mockClient(), { route: "/admin/config/regimes" });

    const regime = await card('article[data-regime="ESMA_Annex2"]');
    expect(within(regime).getByText("ESMA Annex 2")).toBeInTheDocument();
    expect(within(regime).getByText("ESMA")).toBeInTheDocument();
    expect(within(regime).getByText("Annex 2")).toBeInTheDocument();
    expect(within(regime).getByText("v1")).toBeInTheDocument();
    expect(within(regime).getByText("DRAFT1auth.099.001.04_1.3.0.xsd")).toBeInTheDocument();
    expect(within(regime).getByText("107")).toBeInTheDocument();
    expect(within(regime).getByText("reject")).toBeInTheDocument();
    expect(within(regime).getByText("RREC22")).toBeInTheDocument();
    expect(within(regime).getByText("ND5")).toBeInTheDocument();
  });

  it("shows the assets compatible with a regime", async () => {
    renderAdmin(<ConfigRegimesScreen />, mockClient(), { route: "/admin/config/regimes" });

    const regime = await card('article[data-regime="ESMA_Annex2"]');
    expect(within(regime).getByText(copy.admin.regimes.compatibleAssets)).toBeInTheDocument();
    expect(within(regime).getByText("Equity Release")).toBeInTheDocument();
  });

  it("does not hide the draft-schema warning", async () => {
    renderAdmin(<ConfigRegimesScreen />, mockClient(), { route: "/admin/config/regimes" });
    expect(await screen.findByText(copy.admin.regimes.draftSchema)).toBeInTheDocument();
  });
});
