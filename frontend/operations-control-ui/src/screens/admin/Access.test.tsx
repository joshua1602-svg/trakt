import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { OpsError } from "@/api/OpsClient";
import type { OpsClient } from "@/api/OpsClient";
import { OpsClientProvider } from "@/api/context";
import { SessionProvider } from "@/api/session";
import { Shell } from "@/components/Shell";
import { ToastProvider } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { AdminOnly } from "./AdminLayout";
import { ConfigOverviewScreen } from "./ConfigOverview";
import { mockClient, renderAdmin } from "./testUtils";

function renderShell(client: OpsClient) {
  return render(
    <OpsClientProvider client={client}>
      <ToastProvider>
        <SessionProvider>
          <MemoryRouter>
            <Shell>
              <p>content</p>
            </Shell>
          </MemoryRouter>
        </SessionProvider>
      </ToastProvider>
    </OpsClientProvider>,
  );
}

describe("administrator access", () => {
  it("shows the configuration entry in the navigation for an administrator", async () => {
    renderShell(mockClient("admin"));
    expect((await screen.findAllByText(copy.nav.admin)).length).toBeGreaterThan(0);
  });

  it("hides the configuration entry from an ordinary operator", async () => {
    renderShell(mockClient("operator"));
    // Wait for the shell to settle on a nav item that is always present.
    expect((await screen.findAllByText(copy.nav.home)).length).toBeGreaterThan(0);
    expect(screen.queryByText(copy.nav.admin)).not.toBeInTheDocument();
  });

  it("refuses an operator who reaches the route directly", async () => {
    renderAdmin(
      <AdminOnly>
        <ConfigOverviewScreen />
      </AdminOnly>,
      mockClient("operator"),
    );
    expect(await screen.findByText(copy.admin.accessDenied)).toBeInTheDocument();
    expect(screen.queryByText(copy.admin.overview.assetHeading)).not.toBeInTheDocument();
  });

  it("shows the access-denied state when the backend answers 403", async () => {
    const stub = {
      getPrincipal: async () => ({
        name: "Someone",
        role: "admin",
        is_admin: true,
        all_clients: true,
        clients: [],
      }),
      getConfigOverview: async () => {
        throw new OpsError("This area is for administrators.", "OPS_ADMIN_REQUIRED");
      },
    } as unknown as OpsClient;

    renderAdmin(<ConfigOverviewScreen />, stub);
    expect(await screen.findByText(copy.admin.accessDenied)).toBeInTheDocument();
  });

  it("shows the unavailable message when the backend cannot be reached", async () => {
    const stub = {
      getPrincipal: async () => ({
        name: "Administrator",
        role: "admin",
        is_admin: true,
        all_clients: true,
        clients: [],
      }),
      getConfigOverview: async () => {
        throw new OpsError(copy.errors.network);
      },
    } as unknown as OpsClient;

    renderAdmin(<ConfigOverviewScreen />, stub);
    expect(await screen.findByText(copy.admin.unavailable)).toBeInTheDocument();
    expect(screen.queryByText(copy.admin.accessDenied)).not.toBeInTheDocument();
  });
});
