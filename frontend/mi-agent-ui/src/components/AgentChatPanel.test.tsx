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
});

describe("AgentChatPanel context indicator", () => {
  it("is hidden when no context is active", () => {
    render(<AgentChatPanel {...baseProps} context={null} />);
    expect(screen.queryByText(/Context:/)).not.toBeInTheDocument();
  });

  it("shows the active context summary and clears on demand", () => {
    const onClearContext = vi.fn();
    render(<AgentChatPanel {...baseProps} context={context} onClearContext={onClearContext} />);
    expect(screen.getByText(/Balance · Region · South East/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /clear context/i }));
    expect(onClearContext).toHaveBeenCalledOnce();
  });
});
