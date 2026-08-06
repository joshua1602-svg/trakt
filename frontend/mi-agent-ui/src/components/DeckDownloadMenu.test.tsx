import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { DeckDownloadMenu } from "./DeckDownloadMenu";
import type { AgentClient } from "@/api";
import type { DeckGenerationJob, DeckIndex } from "@/domain";

function client(over: Partial<AgentClient>): AgentClient {
  return {
    id: "test", mock: false,
    ask: vi.fn(), getSnapshots: vi.fn(), getSourcePortfolios: vi.fn(), getSnapshot: vi.fn(),
    getPortfolioContext: vi.fn(async () => ({ available: false, client_id: null, default_context_id: "total", contexts: [], portfolios: [], portfolio_types: [], pipeline_portfolios: null })),
    getForecastSnapshot: vi.fn(), getFundedEvolution: vi.fn(), getPipelineEvolution: vi.fn(),
    getForecastEvolution: vi.fn(), getFunnelEvolution: vi.fn(), getRiskLimits: vi.fn(),
    getForecastExtrapolation: vi.fn(), getMe: vi.fn(), getCohorts: vi.fn(),
    getDecks: vi.fn(async () => ({ available: false, latest: null, decks: [], client_id: "" }) as DeckIndex),
    deckDownloadUrl: vi.fn(() => "http://api/mi/decks/download?portfolioId=client_001/mi_2025_11"),
    generateDeck: null,
    getDeckGeneration: vi.fn(),
    ...over,
  } as unknown as AgentClient;
}

const PUBLISHED: DeckIndex = {
  available: true, latest: { period: "2025-11", generatedAt: "2025-11-28T09:00:00Z" },
  decks: [{ period: "2025-11" }, { period: "2025-10" }], client_id: "client_001",
};

function job(over: Partial<DeckGenerationJob> = {}): DeckGenerationJob {
  return {
    ok: true, jobId: "deckgen_1", state: "queued", portfolioId: "client_001",
    requestedAt: "2026-06-30T10:00:00Z", failedGates: [], ...over,
  };
}

describe("DeckDownloadMenu — download", () => {
  it("disables the action when no deck is available and none can be generated", async () => {
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
    const c = client({ getDecks: vi.fn(async () => PUBLISHED) });
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

describe("DeckDownloadMenu — generation", () => {
  it("offers generation even when nothing has ever been published", async () => {
    // The old control was a dead 'No deck' button here. That is exactly the
    // state in which a user most wants to ask for one.
    const c = client({ generateDeck: vi.fn(async () => job()) });
    render(<DeckDownloadMenu client={c} portfolioId="client_001/mi_2025_11" />);
    fireEvent.click(await screen.findByText("Investor deck"));
    expect(await screen.findByTestId("deck-generate")).toBeInTheDocument();
    expect(screen.getByText(/No investor pack has been published/)).toBeInTheDocument();
  });

  it("sends the governed workspace scope with the request", async () => {
    const generateDeck = vi.fn(async () => job());
    const c = client({ getDecks: vi.fn(async () => PUBLISHED), generateDeck });
    render(<DeckDownloadMenu client={c} portfolioId="client_001/mi_2025_11"
                             portfolioContext="acquired" />);
    fireEvent.click(await screen.findByText("Investor deck"));
    fireEvent.click(await screen.findByTestId("deck-generate"));
    await waitFor(() => expect(generateDeck).toHaveBeenCalledWith({
      portfolioId: "client_001/mi_2025_11", portfolioContext: "acquired",
    }));
  });

  it("shows progress while generating, then offers the new deck", async () => {
    const getDecks = vi.fn(async () => PUBLISHED);
    const getDeckGeneration = vi.fn()
      .mockResolvedValueOnce(job({ state: "generating" }))
      .mockResolvedValue(job({ state: "completed", reportingPeriod: "2026-06-30" }));
    const c = client({
      getDecks, generateDeck: vi.fn(async () => job({ state: "queued" })),
      getDeckGeneration,
    });
    render(<DeckDownloadMenu client={c} portfolioId="client_001/mi_2025_11" />);
    fireEvent.click(await screen.findByText("Investor deck"));
    fireEvent.click(await screen.findByTestId("deck-generate"));

    expect(await screen.findByText("Queued…")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Generating…")).toBeInTheDocument(),
                  { timeout: 4000 });
    // Settles back into the normal control...
    await waitFor(() => expect(screen.getByText("Investor deck")).toBeInTheDocument(),
                  { timeout: 6000 });
    // ...and the deck index was re-read, so the new pack is offered without a reload.
    await waitFor(() => expect(getDecks.mock.calls.length).toBeGreaterThan(1));
  }, 15000);

  it("reports a gate-blocked pack as withheld, not as a failure", async () => {
    // Blocked means the pack was BUILT and not released. Calling that 'failed'
    // would send an operator looking for a crash that did not happen.
    const c = client({
      getDecks: vi.fn(async () => PUBLISHED),
      generateDeck: vi.fn(async () => job({ state: "queued" })),
      getDeckGeneration: vi.fn(async () => job({
        state: "blocked", failedGates: ["totals_reconcile"],
        message: "The pack was generated but did not pass its publication checks, "
          + "so it has not been released.",
      })),
    });
    render(<DeckDownloadMenu client={c} portfolioId="client_001/mi_2025_11" />);
    fireEvent.click(await screen.findByText("Investor deck"));
    fireEvent.click(await screen.findByTestId("deck-generate"));

    const problem = await screen.findByTestId("deck-generate-problem", {}, { timeout: 6000 });
    expect(problem.textContent).toContain("Pack withheld");
    expect(problem.getAttribute("title")).toContain("has not been released");
  }, 15000);

  it("surfaces the API's own reason when the request is refused", async () => {
    const c = client({
      getDecks: vi.fn(async () => PUBLISHED),
      generateDeck: vi.fn(async () => {
        throw new Error("No completed reporting run is available for 2019-01.");
      }),
    });
    render(<DeckDownloadMenu client={c} portfolioId="client_001/mi_2025_11" />);
    fireEvent.click(await screen.findByText("Investor deck"));
    fireEvent.click(await screen.findByTestId("deck-generate"));

    const problem = await screen.findByTestId("deck-generate-problem");
    expect(problem.getAttribute("title")).toContain("No completed reporting run");
  });

  it("hides the action entirely when the client cannot generate", async () => {
    const c = client({ getDecks: vi.fn(async () => PUBLISHED), generateDeck: null });
    render(<DeckDownloadMenu client={c} portfolioId="client_001/mi_2025_11" />);
    fireEvent.click(await screen.findByText("Investor deck"));
    await screen.findByText("Latest deck");
    expect(screen.queryByTestId("deck-generate")).toBeNull();
  });
});
