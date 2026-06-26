import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
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

  it("renders the chart inline with a Chart/Table toggle and no debug text", () => {
    const { container } = render(<ChatMessage message={msg} onTogglePin={vi.fn()} />);
    // Conversational, in a teal-tinted assistant bubble.
    expect(screen.getByTestId("assistant-bubble").textContent).toMatch(/London has the largest balance/i);
    expect(screen.queryByText(/Parser|Validation: Passed|Aggregation/)).not.toBeInTheDocument();
    // The chart renders directly in the conversation.
    expect(container.querySelector(".recharts-responsive-container")).not.toBeNull();
    // Chart/Table toggle present.
    expect(screen.getByRole("button", { name: "Chart" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Table" })).toBeInTheDocument();
    // Key observations embedded under the result.
    expect(screen.getByText(/Key observations/i)).toBeInTheDocument();
    // The bare "CHART →" navigation links are NOT the only output.
    expect(screen.queryByText(/CHART →/)).not.toBeInTheDocument();
  });
});

describe("ChatMessage suggestions", () => {
  it("renders suggestion chips and dispatches the question on click", () => {
    const onAsk = vi.fn();
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
    render(<ChatMessage message={msg} onAsk={onAsk} />);
    fireEvent.click(screen.getByRole("button", { name: "Split by Broker" }));
    expect(onAsk).toHaveBeenCalledWith("Balance by Broker");
  });
});

describe("ChatMessage clean default view", () => {
  it("hides interpretation and technical details until Query logic is expanded", () => {
    render(<ChatMessage message={answeredMessage} />);
    // Narrative is visible.
    expect(screen.getByText(/Average LTV is highest in London/)).toBeInTheDocument();
    // Interpretation / diagnostics are NOT in the default view.
    expect(screen.queryByText(/interpreted as/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/total_funded/)).not.toBeInTheDocument();
    expect(screen.queryByText(/resolved region via NUTS 2024/)).not.toBeInTheDocument();
    // Expanding the single Query logic panel reveals them.
    fireEvent.click(screen.getByRole("button", { name: /query logic/i }));
    expect(screen.getByText(/total_funded/)).toBeInTheDocument();
    expect(screen.getByText(/resolved region via NUTS 2024/)).toBeInTheDocument();
  });
});
