import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { AnalysisContext } from "@/lib/analysisContext";
import { AgentChatPanel } from "./AgentChatPanel";

const baseProps = {
  messages: [],
  isWorking: false,
  mock: true,
  onSubmit: vi.fn(),
  onOpenArtifact: vi.fn(),
  onRetry: vi.fn(),
};

const context: AnalysisContext = {
  lastSuccessfulSpec: { metric: "current_outstanding_balance", dimension: "geographic_region_obligor" },
  activeMeasure: "current_outstanding_balance",
  activeDimensions: ["geographic_region_obligor"],
  activeFilters: { geographic_region_obligor: "South East" },
};

describe("AgentChatPanel surface", () => {
  it("marks the chat as a distinct AI surface with teal styling", () => {
    const { container } = render(<AgentChatPanel {...baseProps} context={null} />);
    const surface = container.querySelector('[data-surface="ai-chat"]');
    expect(surface).not.toBeNull();
    // Distinct teal palette, not the navy analytics card colour.
    expect(surface!.className).toMatch(/teal/);
  });

  it("carries a Beta tag beside the MI Agent title", () => {
    render(<AgentChatPanel {...baseProps} context={null} />);
    expect(screen.getByText("MI Agent")).toBeInTheDocument();
    expect(screen.getByText("Beta")).toBeInTheDocument();
  });

  it("invites a portfolio question, not a list of capability keywords", () => {
    render(<AgentChatPanel {...baseProps} context={null} />);
    expect(screen.getByPlaceholderText("Ask me about your portfolio…")).toBeInTheDocument();
  });
});

describe("AgentChatPanel context indicator", () => {
  it("is hidden when no context is active", () => {
    render(<AgentChatPanel {...baseProps} context={null} />);
    expect(screen.queryByText(/Context:/)).not.toBeInTheDocument();
  });

  // CHANGED DELIBERATELY. The context bar used to be permanently visible, which
  // made it a third printing of the measure and dimensions the answer's
  // execution receipt and the artifact's own title already carry. It answers
  // exactly one question — "what will a follow-up attach to?" — so it now
  // appears when that question is live.
  it("shows the active context summary once the composer has focus", () => {
    const onClearContext = vi.fn();
    render(<AgentChatPanel {...baseProps} context={context} onClearContext={onClearContext} />);
    expect(screen.queryByTestId("chat-context-bar")).toBeNull();
    fireEvent.focus(screen.getByRole("textbox"));
    expect(screen.getByText(/Balance · Region · South East/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /clear context/i }));
    expect(onClearContext).toHaveBeenCalledOnce();
  });

  it("shows it unprompted once what is typed reads as a follow-up", () => {
    render(<AgentChatPanel {...baseProps} context={context} onClearContext={vi.fn()} />);
    const box = screen.getByRole("textbox");
    fireEvent.change(box, { target: { value: "split by broker" } });
    fireEvent.blur(box);
    expect(screen.getByTestId("chat-context-bar")).toBeInTheDocument();
  });

  it("stays out of the way for a standalone question", () => {
    render(<AgentChatPanel {...baseProps} context={context} onClearContext={vi.fn()} />);
    const box = screen.getByRole("textbox");
    fireEvent.change(box, { target: { value: "What is the funded balance by region?" } });
    fireEvent.blur(box);
    expect(screen.queryByTestId("chat-context-bar")).toBeNull();
  });
});
