import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StatusChip } from "./StatusChip";

describe("StatusChip", () => {
  it("prefers the server-provided label", () => {
    render(<StatusChip status="needs_review" label="Needs review" />);
    expect(screen.getByText("Needs review")).toBeInTheDocument();
  });

  it("falls back to a plain-English label for known statuses", () => {
    render(<StatusChip status="awaiting_publication" />);
    expect(screen.getByText("Ready to publish")).toBeInTheDocument();
  });

  it("humanises unknown statuses instead of showing raw codes", () => {
    render(<StatusChip status="some_new_state" />);
    expect(screen.getByText("Some new state")).toBeInTheDocument();
  });

  it("colour-codes by status", () => {
    render(<StatusChip status="blocked" />);
    const chip = screen.getByText("Blocked");
    expect(chip.className).toContain("rose");
  });
});
