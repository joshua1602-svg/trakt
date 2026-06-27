import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ArtifactCanvas } from "./ArtifactCanvas";
import { kpiArtifact } from "@/data/mockArtifacts";

const ctx = { asOf: "2026-05-31", portfolio: "erm-uk-master" };

describe("ArtifactCanvas declutter controls (A8)", () => {
  it("clears artifacts via the Clear button (view-only)", () => {
    const onClear = vi.fn();
    render(
      <ArtifactCanvas
        artifacts={[kpiArtifact(ctx)]}
        onTogglePin={() => {}}
        isWorking={false}
        portfolioName="ERM UK"
        onClear={onClear}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /Clear artifacts/i }));
    expect(onClear).toHaveBeenCalledTimes(1);
  });

  it("collapses and expands the workspace body", () => {
    render(
      <ArtifactCanvas
        artifacts={[kpiArtifact(ctx)]}
        onTogglePin={() => {}}
        isWorking={false}
        portfolioName="ERM UK"
        onClear={() => {}}
      />,
    );
    // Expanded: the KPI artifact title renders.
    expect(screen.getByText("Executive Summary")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Collapse artifact workspace/i }));
    expect(screen.queryByText("Executive Summary")).toBeNull();
    expect(screen.getByText(/Workspace collapsed/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Expand artifact workspace/i }));
    expect(screen.getByText("Executive Summary")).toBeInTheDocument();
  });
});
