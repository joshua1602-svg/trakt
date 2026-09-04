/**
 * Tests for the UX-enhancement pass: inline disabled-tab reasons, chat
 * collapse / stop / mid-conversation suggestions, stale workspace links,
 * drill-filter chips, and workspace artifact accumulation.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ChartArtifact } from "@/domain";
import { ViewToggle } from "./ViewToggle";
import { AgentChatPanel } from "./AgentChatPanel";
import { ChatResult } from "./ChatResult";
import { ArtifactCard } from "./ArtifactCard";

beforeEach(() => {
  localStorage.clear();
});

function chart(over: Partial<ChartArtifact> = {}): ChartArtifact {
  return {
    id: "c1",
    type: "chart",
    title: "Balance by Region",
    source: {
      engine: "mi_agent.workflow",
      label: "MI Agent · bar",
      spec: { metric: "current_outstanding_balance", dimension: "geographic_region_obligor" },
      question: "balance by region",
    },
    createdAt: "2026-05-31T08:00:00Z",
    mock: false,
    chartType: "bar",
    xKey: "geographic_region_obligor",
    series: [{ key: "current_outstanding_balance", label: "Balance", color: "#000" }],
    valueFormat: "gbp",
    rows: [
      { geographic_region_obligor: "London", current_outstanding_balance: 700 },
      { geographic_region_obligor: "South East", current_outstanding_balance: 300 },
    ],
    ...over,
  };
}

describe("ViewToggle — inline disabled reason", () => {
  it("reveals the backend's reason inline when a disabled tab is clicked, without firing onChange", () => {
    const onChange = vi.fn();
    const reason = "No portfolio in this scope originates new lending.";
    render(<ViewToggle active="funded" onChange={onChange}
      disabledViews={["pipeline"]} disabledReasons={{ pipeline: reason }} />);
    expect(screen.queryByTestId("disabled-view-reason")).toBeNull();
    // The click lands on the wrapper (the disabled button swallows it).
    fireEvent.click(screen.getByRole("tab", { name: /Pipeline/ }).parentElement!);
    const note = screen.getByTestId("disabled-view-reason");
    expect(note.textContent).toContain(reason);
    expect(onChange).not.toHaveBeenCalled();
    // Dismissible.
    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
    expect(screen.queryByTestId("disabled-view-reason")).toBeNull();
  });
});

describe("AgentChatPanel — collapse / stop / suggestions", () => {
  const baseProps = {
    messages: [],
    isWorking: false,
    mock: true,
    onSubmit: vi.fn(),
    onOpenArtifact: vi.fn(),
    onRetry: vi.fn(),
  };

  it("collapses to a slim rail and expands back (surface marker retained)", () => {
    const { container } = render(<AgentChatPanel {...baseProps} context={null} />);
    fireEvent.click(screen.getByRole("button", { name: /collapse chat/i }));
    expect(container.querySelector('[data-surface="ai-chat"]')).not.toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /expand chat/i }));
    expect(screen.getByRole("button", { name: /collapse chat/i })).toBeInTheDocument();
  });

  it("shows a Stop control while working and dispatches onStop", () => {
    const onStop = vi.fn();
    render(<AgentChatPanel {...baseProps} isWorking onStop={onStop} context={null} />);
    fireEvent.click(screen.getByRole("button", { name: /stop/i }));
    expect(onStop).toHaveBeenCalledOnce();
  });

  it("keeps suggested questions reachable mid-conversation", () => {
    const onSubmit = vi.fn();
    const messages = [
      { id: "m1", role: "assistant" as const, content: "hello", createdAt: "2026-05-31T08:00:00Z" },
      { id: "m2", role: "user" as const, content: "balance", createdAt: "2026-05-31T08:01:00Z" },
    ];
    render(<AgentChatPanel {...baseProps} messages={messages} onSubmit={onSubmit} context={null} />);
    fireEvent.click(screen.getByRole("button", { name: /suggested questions/i }));
    expect(screen.getByText(/Suggested questions/i)).toBeInTheDocument();
  });
});

// CHANGED DELIBERATELY. These pinned a disabled "(cleared)" BUTTON and a live
// one beside it. There is no longer a button in either state: the workspace
// reveals a new artifact itself, so the live link had nowhere to take anyone,
// and the cleared case is something to say rather than something to click. What
// the pair was really protecting — that a reader is never left clicking at an
// artifact that is gone — is what these now assert.
describe("ChatResult — a result cleared from the workspace", () => {
  it("says so, in words, and offers nothing to click", () => {
    const onOpenArtifact = vi.fn();
    render(<ChatResult artifacts={[chart()]} onOpenArtifact={onOpenArtifact}
      workspaceArtifactIds={new Set(["something-else"])} />);
    expect(screen.getByTestId("chat-result-cleared").textContent)
      .toMatch(/no longer in the workspace — ask again to regenerate it/i);
    expect(screen.queryByRole("button")).toBeNull();
    expect(onOpenArtifact).not.toHaveBeenCalled();
  });

  it("stays silent while the artifact is present", () => {
    const { container } = render(<ChatResult artifacts={[chart()]} onOpenArtifact={vi.fn()}
      workspaceArtifactIds={new Set(["c1"])} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("stays silent when it cannot know — no workspace ids supplied", () => {
    const { container } = render(<ChatResult artifacts={[chart()]} onOpenArtifact={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("ArtifactCard — drill-filter chips (the way back)", () => {
  it("renders active drill filters as chips and re-runs unfiltered on removal", () => {
    const onDrill = vi.fn();
    const drilled = chart({ drillFilters: { geographic_region_obligor: "London" } });
    render(<ArtifactCard artifact={drilled} onTogglePin={vi.fn()} onDrill={onDrill} />);
    const chips = screen.getByTestId("drill-filter-chips");
    expect(chips.textContent).toContain("London");
    fireEvent.click(screen.getByRole("button", { name: /remove .* filter/i }));
    // Removing the only chip re-runs the originating query with no filters.
    expect(onDrill).toHaveBeenCalledWith(drilled, {});
  });

  it("shows no chip strip on an unfiltered artifact", () => {
    render(<ArtifactCard artifact={chart()} onTogglePin={vi.fn()} />);
    expect(screen.queryByTestId("drill-filter-chips")).toBeNull();
  });
});
