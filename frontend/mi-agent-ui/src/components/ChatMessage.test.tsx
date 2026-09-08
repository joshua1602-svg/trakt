import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { ChartArtifact, ChatMessage as ChatMessageType, TableArtifact } from "@/domain";
import { ChatMessage } from "./ChatMessage";

function regionChart(): ChartArtifact {
  return {
    id: "c1",
    type: "chart",
    title: "Balance by Region",
    source: { engine: "mi_agent.workflow", label: "MI Agent · bar", spec: { metric: "current_outstanding_balance", dimension: "geographic_region_obligor" } },
    createdAt: "2026-05-31T08:00:00Z",
    mock: false,
    chartType: "bar",
    xKey: "geographic_region_obligor",
    series: [{ key: "current_outstanding_balance", label: "Balance", color: "#000" }],
    valueFormat: "gbp",
    displayHints: { current_outstanding_balance: { format: "gbp", scale: null } },
    rows: [
      { geographic_region_obligor: "London", current_outstanding_balance: 700 },
      { geographic_region_obligor: "South East", current_outstanding_balance: 300 },
    ],
  };
}

function regionTable(): TableArtifact {
  return {
    id: "t1",
    type: "table",
    title: "Balance by Region",
    source: { engine: "mi_agent.workflow", label: "MI Agent · table" },
    createdAt: "2026-05-31T08:00:00Z",
    mock: false,
    columns: [
      { key: "geographic_region_obligor", label: "Region", format: "text" },
      { key: "current_outstanding_balance", label: "Balance", format: "gbp" },
    ],
    rows: [
      { geographic_region_obligor: "London", current_outstanding_balance: 700 },
      { geographic_region_obligor: "South East", current_outstanding_balance: 300 },
    ],
  };
}

const errorMessage: ChatMessageType = {
  id: "m1",
  role: "assistant",
  content: "Could not reach the MI Agent API. Is the backend running?",
  createdAt: "2026-05-31T08:00:00Z",
  error: true,
};

describe("ChatMessage error state", () => {
  it("renders the error text and a retry control", () => {
    const onRetry = vi.fn();
    render(<ChatMessage message={errorMessage} onRetry={onRetry} />);
    expect(screen.getByText(/Could not reach the MI Agent API/)).toBeInTheDocument();
    const retry = screen.getByRole("button", { name: /retry/i });
    retry.click();
    expect(onRetry).toHaveBeenCalledOnce();
  });
});

const answeredMessage: ChatMessageType = {
  id: "m2",
  role: "assistant",
  content: "Average LTV is highest in London.",
  createdAt: "2026-05-31T08:00:00Z",
  interpreted: "Concentration · total_funded · average_ltv by region",
  spec: { metric: "average_ltv", dimensions: ["region"], aggregation: "weighted_avg" },
  confidence: 0.92,
  diagnostics: ["resolved region via NUTS 2024"],
};

describe("ChatMessage embedded result", () => {
  const msg: ChatMessageType = {
    id: "m9",
    role: "assistant",
    content: "London has the largest balance. I've shown balance by region below.",
    createdAt: "2026-05-31T08:00:00Z",
    artifacts: [regionChart(), regionTable()],
    artifactRefs: [
      { id: "c1", title: "Balance by Region", type: "chart" },
      { id: "t1", title: "Balance by Region", type: "table" },
    ],
  };

  const inWorkspace = new Set(["c1", "t1"]);

  it("says nothing about a result that is already on screen", () => {
    // CHANGED DELIBERATELY. This asserted that the chat offered "Open chart in
    // workspace" / "Open table in workspace" buttons. The workspace expands
    // itself, scrolls to the new card and flash-highlights it on arrival, so
    // those buttons navigated to something already open, already scrolled to
    // and already glowing. The chat states; the workspace shows.
    const { container } = render(
      <ChatMessage message={msg} onTogglePin={vi.fn()} onOpenArtifact={vi.fn()}
        workspaceArtifactIds={inWorkspace} />,
    );
    expect(screen.getByTestId("assistant-bubble").textContent).toMatch(/London has the largest balance/i);
    expect(screen.queryByText(/Parser|Validation: Passed|Aggregation/)).not.toBeInTheDocument();
    // The chart is not duplicated in the chat — that part is unchanged.
    expect(container.querySelector(".recharts-responsive-container")).toBeNull();
    // And no navigation to a panel the reader is already looking at.
    expect(screen.queryByRole("button", { name: /Open chart in workspace/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /Open table in workspace/i })).toBeNull();
    expect(screen.queryByTestId("chat-result-cleared")).toBeNull();
  });

  it("says so, once, when the result has been cleared from the workspace", () => {
    render(
      <ChatMessage message={msg} onTogglePin={vi.fn()} onOpenArtifact={vi.fn()}
        workspaceArtifactIds={new Set<string>()} />,
    );
    expect(screen.getByTestId("chat-result-cleared").textContent)
      .toMatch(/no longer in the workspace/i);
    // A sentence, not a control.
    expect(screen.queryByRole("button", { name: /workspace/i })).toBeNull();
  });

  it("repeats none of the result's own figures", () => {
    // The chips read "Groups: 2" and "Coverage: 100%" — both already in the
    // answer's execution receipt and the artifact's reconciliation footer.
    render(
      <ChatMessage message={msg} onTogglePin={vi.fn()} workspaceArtifactIds={inWorkspace} />,
    );
    expect(screen.queryByText(/^Groups: /)).toBeNull();
    expect(screen.queryByText(/^Coverage: /)).toBeNull();
  });

  it("never renders the full chart inline — outputs live in the workspace", () => {
    const { container } = render(<ChatMessage message={msg} onTogglePin={vi.fn()} />);
    // No inline chart, and no 'Show here' affordance that would duplicate it.
    expect(container.querySelector(".recharts-responsive-container")).toBeNull();
    expect(screen.queryByRole("button", { name: /Show here/i })).toBeNull();
  });
});

describe("ChatMessage suggestions", () => {
  it("renders no suggestion chips — they belong beside the result", () => {
    // CHANGED DELIBERATELY. This asserted the chat rendered the chips and
    // dispatched on click. Two engines were building follow-ups independently
    // — buildSuggestedActions here and buildInvestigations on the artifact card
    // — so one chart put six offers on screen in two visual languages. Both now
    // feed the card's single "Investigate next" row; see
    // InsightPanel.suggestions.test.tsx for the surface that renders them.
    const msg: ChatMessageType = {
      id: "m3",
      role: "assistant",
      content: "Balance by region.",
      createdAt: "2026-05-31T08:00:00Z",
      suggestions: [
        { label: "Split by Broker", question: "Balance by Broker", kind: "change_dimension" },
        { label: "Drill into London", question: "only London", kind: "drill" },
      ],
    };
    render(<ChatMessage message={msg} />);
    expect(screen.queryByRole("button", { name: "Split by Broker" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Drill into London" })).toBeNull();
    // The message still carries them for anything reading the transcript.
    expect(msg.suggestions).toHaveLength(2);
  });
});

describe("ChatMessage clean default view", () => {
  it("shows only the conversational narrative — no interpretation, spec or diagnostics", () => {
    render(<ChatMessage message={answeredMessage} />);
    // Narrative is visible.
    expect(screen.getByText(/Average LTV is highest in London/)).toBeInTheDocument();
    // The client chat stays clean: no interpretation line, no query-logic
    // controls, no raw diagnostics (that provenance stays backend-side).
    expect(screen.queryByRole("button", { name: /query logic/i })).toBeNull();
    expect(screen.queryByText(/interpreted as/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/total_funded/)).not.toBeInTheDocument();
    expect(screen.queryByText(/resolved region via NUTS 2024/)).not.toBeInTheDocument();
  });

  it("shows only a minimal dataset badge (which book answered), nothing else", () => {
    const msg: ChatMessageType = {
      ...answeredMessage,
      interpreted:
        "Chart: Bar · Metric: Current Outstanding Balance · Dimension: Region · Parser: deterministic · Validation: Passed",
      datasetContext: "pipeline",
    };
    render(<ChatMessage message={msg} />);
    // The book that answered IS shown (it materially changes the number's meaning).
    expect(screen.getByText(/^pipeline$/i)).toBeInTheDocument();
    // But the interpretation / parse internals are NOT rendered client-side.
    expect(screen.queryByText(/interpreted as/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Parser:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Validation:/)).not.toBeInTheDocument();
  });
});
