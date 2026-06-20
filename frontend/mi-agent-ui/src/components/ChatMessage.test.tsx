import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
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
