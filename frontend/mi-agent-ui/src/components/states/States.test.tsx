import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ErrorState } from "./States";
import { NOT_PROVISIONED_MESSAGE } from "@/domain/accessErrors";

describe("ErrorState", () => {
  it("renders an ordinary failure as a retryable fault", () => {
    const onRetry = vi.fn();
    render(<ErrorState message="MI Agent API returned 500" onRetry={onRetry} />);
    expect(screen.getByText("Something went wrong")).toBeTruthy();
    expect(screen.getByRole("button", { name: /retry/i })).toBeTruthy();
  });

  it("renders an unprovisioned account as an explanation, not a fault", () => {
    render(
      <ErrorState
        message={`The MI Agent API could not be reached — ${NOT_PROVISIONED_MESSAGE}`}
        onRetry={vi.fn()}
      />,
    );
    expect(screen.getByText("Access not provisioned")).toBeTruthy();
    expect(screen.queryByText("Something went wrong")).toBeNull();
  });

  it("offers no Retry for an access refusal", () => {
    // Retrying cannot provision an account; offering the button sends the user
    // round a loop with no exit, which is what the old panel did.
    render(<ErrorState message={NOT_PROVISIONED_MESSAGE} onRetry={vi.fn()} />);
    expect(screen.queryByRole("button", { name: /retry/i })).toBeNull();
  });

  it("does not leak the API plumbing into the provisioning message", () => {
    render(
      <ErrorState
        message={`The MI Agent API could not be reached — ${NOT_PROVISIONED_MESSAGE}`}
      />,
    );
    expect(screen.queryByText(/could not be reached/i)).toBeNull();
    expect(screen.getByText(/provisioned for Trakt access/i)).toBeTruthy();
  });
});
