import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { HeaderBar } from "./HeaderBar";
import type { AgentClient } from "@/api";
import type { UserIdentity } from "@/lib/identity";

function client(): AgentClient {
  return {
    id: "test", mock: true,
    ask: vi.fn(), getSnapshots: vi.fn(), getSourcePortfolios: vi.fn(), getSnapshot: vi.fn(),
    getForecastSnapshot: vi.fn(), getFundedEvolution: vi.fn(), getPipelineEvolution: vi.fn(),
    getForecastEvolution: vi.fn(), getFunnelEvolution: vi.fn(), getRiskLimits: vi.fn(),
    getForecastExtrapolation: vi.fn(),
    getMe: vi.fn(),
    getDecks: vi.fn(async () => ({ available: false, latest: null, decks: [], client_id: "" })),
    deckDownloadUrl: vi.fn(() => null),
    getCohorts: vi.fn(),
  } as unknown as AgentClient;
}

function renderHeader(identity: UserIdentity | null) {
  return render(
    <HeaderBar
      portfolios={[]} runs={[]} selectedClientId={null} selectedRunId={null}
      onPortfolioChange={() => {}} onRunChange={() => {}} mock={false}
      client={client()} portfolioId="client_001/mi_2025_11" reportingPeriod="2025-11"
      identity={identity} onRefresh={() => {}}
    />,
  );
}

describe("HeaderBar identity + role gating (Task 8)", () => {
  it("renders the Entra-derived display name, not a hardcoded label", () => {
    renderHeader({ authenticated: true, user: "Joshua Hall", isOperator: true, roles: ["operator"] });
    expect(screen.getByTestId("user-identity").textContent).toContain("J. Hall");
    expect(screen.queryByText("J. Analyst")).toBeNull();
  });

  it("shows settings + notifications for operators", () => {
    renderHeader({ authenticated: true, user: "Joshua Hall", isOperator: true, roles: ["operator"] });
    expect(screen.getByLabelText("Settings")).toBeInTheDocument();
    expect(screen.getByLabelText("Notifications")).toBeInTheDocument();
  });

  it("hides settings + notifications for client users", () => {
    renderHeader({ authenticated: true, user: "Jane Smith", isOperator: false, roles: ["client"] });
    expect(screen.queryByLabelText("Settings")).toBeNull();
    expect(screen.queryByLabelText("Notifications")).toBeNull();
    expect(screen.getByTestId("user-identity").textContent).toContain("J. Smith");
  });

  it("hides admin controls and identity when unauthenticated (fail-closed)", () => {
    renderHeader({ authenticated: false });
    expect(screen.queryByLabelText("Settings")).toBeNull();
    expect(screen.queryByTestId("user-identity")).toBeNull();
  });

  it("keeps a manual refresh control", () => {
    renderHeader({ authenticated: true, user: "Joshua Hall", isOperator: true });
    expect(screen.getByTestId("refresh-button")).toBeInTheDocument();
  });
});
