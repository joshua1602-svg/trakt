import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  kpiArtifact,
  concentrationTableArtifact,
  regionChartArtifact,
  riskConcentrationArtifact,
  riskMigrationArtifact,
  scenarioArtifact,
  validationArtifact,
} from "@/data/mockArtifacts";
import { ArtifactRenderer } from "./ArtifactRenderer";

const ctx = { asOf: "2026-05-31", portfolio: "erm-uk-master" };

describe("ArtifactRenderer", () => {
  it("renders KPI values", () => {
    render(<ArtifactRenderer artifact={kpiArtifact(ctx)} />);
    expect(screen.getByText("Portfolio Balance")).toBeInTheDocument();
    expect(screen.getByText("£842.6MM")).toBeInTheDocument();
  });

  it("renders a table with region rows", () => {
    render(<ArtifactRenderer artifact={concentrationTableArtifact(ctx)} />);
    expect(screen.getByText("London")).toBeInTheDocument();
    expect(screen.getByText("Wtd. LTV %")).toBeInTheDocument();
  });

  it("renders a chart container without throwing", () => {
    const { container } = render(<ArtifactRenderer artifact={regionChartArtifact(ctx)} />);
    expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
  });

  it("renders the validation summary", () => {
    render(<ArtifactRenderer artifact={validationArtifact(ctx)} />);
    expect(screen.getByText("Blockers")).toBeInTheDocument();
    expect(screen.getByText(/Missing valuation date/)).toBeInTheDocument();
  });

  it("renders risk concentration limits", () => {
    render(<ArtifactRenderer artifact={riskConcentrationArtifact(ctx)} />);
    expect(screen.getByText(/London \(UKI\)/)).toBeInTheDocument();
    expect(screen.getAllByText(/Breach|Within limit|Approaching/).length).toBeGreaterThan(0);
  });

  it("renders the risk migration matrix", () => {
    const { container } = render(<ArtifactRenderer artifact={riskMigrationArtifact(ctx)} />);
    expect(container.querySelector("table")).toBeTruthy();
    expect(screen.getByText(/From . To/)).toBeInTheDocument();
  });

  it("renders the scenario assumptions and chart", () => {
    const { container } = render(<ArtifactRenderer artifact={scenarioArtifact(ctx)} />);
    expect(screen.getByText("HPI growth")).toBeInTheDocument();
    expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
  });
});
