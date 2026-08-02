/**
 * Phase 1A item 8 — the dashboard must stop re-fetching data it already has.
 *
 * Three behaviours are pinned:
 *   1. switching sub-tabs and coming back does NOT refetch (the pane is kept
 *      mounted, so its loaded data survives);
 *   2. enlarging a chart issues NO network call at all;
 *   3. a repeated request while one is in flight is de-duplicated.
 *
 * These assert on the AgentClient call counts rather than on internals, so they
 * keep working if the panels are restructured.
 */

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { KeepMounted } from "@/components/KeepMounted";
import { EvolutionPanel } from "@/components/EvolutionPanel";
import { withCache } from "@/api/CachingAgentClient";
import { MockAgentClient } from "@/api/MockAgentClient";
import type { AgentClient } from "@/api";

function countingClient(): { client: AgentClient; calls: Record<string, number> } {
  const calls: Record<string, number> = {};
  // No simulated latency: these tests assert CALL COUNTS, not timing.
  const base = new MockAgentClient({ latencyMs: 0 }) as unknown as AgentClient;
  const mutable = base as unknown as Record<string, unknown>;
  const wrap = (name: keyof AgentClient) => {
    const original = mutable[name as string];
    if (typeof original !== "function") return;
    const fn = original as (...a: unknown[]) => unknown;
    mutable[name as string] = (...args: unknown[]) => {
      calls[name as string] = (calls[name as string] ?? 0) + 1;
      return fn.apply(base, args);
    };
  };
  (["getFundedEvolution", "getPipelineEvolution", "getFunnelEvolution",
    "getForecastEvolution", "getCohorts", "getCohortProgression",
    "getSnapshot", "getForecastSnapshot"] as Array<keyof AgentClient>).forEach(wrap);
  return { client: withCache(base), calls };
}

describe("KeepMounted", () => {
  it("does not mount children until first activated", () => {
    const spy = vi.fn();
    function Probe() { spy(); return <div>child</div>; }
    const { rerender } = render(
      <KeepMounted active={false}><Probe /></KeepMounted>);
    expect(spy).not.toHaveBeenCalled();
    rerender(<KeepMounted active><Probe /></KeepMounted>);
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("keeps children mounted once activated, hiding them instead", () => {
    const unmount = vi.fn();
    function Probe() {
      // Runs on unmount only.
      return <div data-testid="probe" ref={() => unmount} >child</div>;
    }
    const { rerender, container } = render(
      <KeepMounted active testId="pane"><Probe /></KeepMounted>);
    expect(screen.getByTestId("probe")).toBeInTheDocument();
    rerender(<KeepMounted active={false} testId="pane"><Probe /></KeepMounted>);
    // Still in the DOM (state preserved), but hidden from view and a11y.
    const pane = container.querySelector('[data-testid="pane"]');
    expect(pane).toBeTruthy();
    expect(pane).toHaveAttribute("hidden");
    expect(pane).toHaveAttribute("aria-hidden", "true");
  });
});

describe("tab retention", () => {
  it("returning to a sub-view does not refetch its series", async () => {
    const { client, calls } = countingClient();
    // Mount the funded series, switch away, and come back — the pane is kept
    // mounted, so there is nothing to re-request.
    const { rerender } = render(
      <>
        <KeepMounted active>
          <EvolutionPanel client={client} portfolioId="c/r" tabs={["funded"]} heading={false} />
        </KeepMounted>
        <KeepMounted active={false}>
          <div>other</div>
        </KeepMounted>
      </>,
    );
    await waitFor(() => expect(calls.getFundedEvolution).toBe(1));

    // Switch away and back.
    rerender(
      <>
        <KeepMounted active={false}>
          <EvolutionPanel client={client} portfolioId="c/r" tabs={["funded"]} heading={false} />
        </KeepMounted>
        <KeepMounted active><div>other</div></KeepMounted>
      </>,
    );
    rerender(
      <>
        <KeepMounted active>
          <EvolutionPanel client={client} portfolioId="c/r" tabs={["funded"]} heading={false} />
        </KeepMounted>
        <KeepMounted active={false}><div>other</div></KeepMounted>
      </>,
    );
    await new Promise((r) => setTimeout(r, 20));
    expect(calls.getFundedEvolution).toBe(1);
  });
});

describe("chart expansion", () => {
  it("enlarging a chart issues no new API calls", async () => {
    const { client, calls } = countingClient();
    render(
      <EvolutionPanel client={client} portfolioId="c/r"
        tabs={["pipeline", "origination"]} heading={false} />,
    );
    await waitFor(() => expect(calls.getPipelineEvolution).toBe(1));

    // Move to the origination funnel, which hosts the expandable stage cards.
    fireEvent.click(screen.getByRole("tab", { name: /origination/i }));
    await waitFor(() => expect(calls.getFunnelEvolution).toBe(1));
    const afterTab = JSON.stringify(calls);

    // Assert (not skip) that there is something to enlarge — a test that
    // silently passes when the fixture has no stages would prove nothing.
    const expandButtons = await screen.findAllByTestId(/^funnel-expand-/);
    expect(expandButtons.length).toBeGreaterThan(0);

    fireEvent.click(expandButtons[0]);
    await waitFor(() =>
      expect(screen.getByTestId("funnel-focus-modal")).toBeInTheDocument());
    await new Promise((r) => setTimeout(r, 20));

    // THE assertion: enlarging re-renders from data already held. No refetch,
    // and therefore no backend recalculation.
    expect(JSON.stringify(calls)).toBe(afterTab);
  });
});

describe("in-flight de-duplication", () => {
  it("concurrent identical GETs issue one underlying request", async () => {
    const { client, calls } = countingClient();
    await Promise.all([
      client.getSnapshot("c/r", "total"),
      client.getSnapshot("c/r", "total"),
      client.getSnapshot("c/r", "total"),
    ]);
    expect(calls.getSnapshot).toBe(1);
  });

  it("a different portfolio scope is not de-duplicated into the wrong entry", async () => {
    const { client, calls } = countingClient();
    await Promise.all([
      client.getSnapshot("c/r", "total"),
      client.getSnapshot("c/r", "direct_001"),
    ]);
    expect(calls.getSnapshot).toBe(2);
  });
});
