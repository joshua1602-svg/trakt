import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ChatMessage as ChatMessageType } from "@/domain";
import { ChatMessage } from "./ChatMessage";

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
