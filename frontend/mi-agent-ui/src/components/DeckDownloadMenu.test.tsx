import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { DeckDownloadMenu } from "./DeckDownloadMenu";
import type { AgentClient } from "@/api";
import type { DeckIndex } from "@/domain";

function client(over: Partial<AgentClient>): AgentClient {
  return {
    id: "test", mock: false,
    ask: vi.fn(), getSnapshots: vi.fn(), getSourcePortfolios: vi.fn(), getSnapshot: vi.fn(),
    getForecastSnapshot: vi.fn(), getFundedEvolution: vi.fn(), getPipelineEvolution: vi.fn(),
    getForecastEvolution: vi.fn(), getFunnelEvolution: vi.fn(), getRiskLimits: vi.fn(),
    getForecastExtrapolation: vi.fn(), getMe: vi.fn(), getCohorts: vi.fn(),
    getDecks: vi.fn(async () => ({ available: false, latest: null, decks: [], client_id: "" }) as DeckIndex),
    deckDownloadUrl: vi.fn(() => "http://api/mi/decks/download?portfolioId=client_001/mi_2025_11"),
    ...over,
  } as unknown as AgentClient;
}

describe("DeckDownloadMenu (Task 1)", () => {
  it("disables the action when no deck is available", async () => {
    const c = client({});
    render(<DeckDownloadMenu client={c} portfolioId="client_001/mi_2025_11" />);
    const btn = await screen.findByTestId("deck-download");
    expect(btn).toBeDisabled();
    expect(btn.textContent).toContain("No deck");
  });

  it("disables when the client cannot serve decks (e.g. mock returns null URL)", async () => {
    const c = client({
      getDecks: vi.fn(async () => ({ available: true, latest: { period: "2025-11" }, decks: [], client_id: "c" })),
      deckDownloadUrl: vi.fn(() => null),
    });
    render(<DeckDownloadMenu client={c} portfolioId="client_001/mi_2025_11" />);
    expect(await screen.findByTestId("deck-download")).toBeDisabled();
  });

  it("offers the latest deck and a disabled state for a reporting date without a deck", async () => {
    const c = client({
      getDecks: vi.fn(async () => ({
        available: true, latest: { period: "2025-11", generatedAt: "2025-11-28T09:00:00Z" },
        decks: [{ period: "2025-11" }, { period: "2025-10" }], client_id: "client_001",
      })),
    });
    render(<DeckDownloadMenu client={c} portfolioId="client_001/mi_2025_11" reportingPeriod="2025-09" />);
    // Wait for getDecks to resolve → the button becomes the enabled "Investor deck".
    const btn = await screen.findByText("Investor deck");
    fireEvent.click(btn);
    // Latest deck action is present.
    expect(await screen.findByText("Latest deck")).toBeInTheDocument();
    // The selected reporting date (2025-09) has no deck → the disabled note shows.
    expect(screen.getByText("No deck available for this reporting date")).toBeInTheDocument();
  });

  it("triggers a download via the client's deckDownloadUrl", async () => {
    const dl = vi.fn(() => "http://api/mi/decks/download?portfolioId=client_001/mi_2025_11");
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    const c = client({
      getDecks: vi.fn(async () => ({ available: true, latest: { period: "2025-11" }, decks: [{ period: "2025-11" }], client_id: "c" })),
      deckDownloadUrl: dl,
    });
    render(<DeckDownloadMenu client={c} portfolioId="client_001/mi_2025_11" />);
    fireEvent.click(await screen.findByText("Investor deck"));
    fireEvent.click(await screen.findByText("Latest deck"));
    await waitFor(() => expect(clickSpy).toHaveBeenCalled());
    clickSpy.mockRestore();
  });
});
