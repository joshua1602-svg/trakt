import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { PortfolioCapabilities, PortfolioContextOption } from "@/domain";
import { PortfolioContextSelector } from "./PortfolioContextSelector";
import { PortfolioScopeBanner } from "./PortfolioScopeBanner";
import { ViewToggle } from "./ViewToggle";

const EMPTY_CAPS = {} as PortfolioCapabilities;

function ctx(over: Partial<PortfolioContextOption>): PortfolioContextOption {
  return {
    context_id: "total", context_kind: "total", label: "Total",
    parent_id: null, depth: 0, portfolio_ids: [], portfolio_types: [],
    capabilities: EMPTY_CAPS,
    ...over,
  };
}

/** Four source portfolios: two direct, two acquired. */
const CONTEXTS: PortfolioContextOption[] = [
  ctx({ context_id: "total", label: "Total", depth: 0,
        portfolio_ids: ["direct_001", "direct_002", "acquired_001", "acquired_002"] }),
  ctx({ context_id: "direct", context_kind: "type", label: "Direct", parent_id: "total",
        depth: 1, portfolio_ids: ["direct_001", "direct_002"], portfolio_types: ["direct"] }),
  ctx({ context_id: "direct_001", context_kind: "portfolio", label: "Direct 001",
        parent_id: "direct", depth: 2, portfolio_ids: ["direct_001"],
        portfolio_types: ["direct"] }),
  ctx({ context_id: "direct_002", context_kind: "portfolio", label: "Direct 002",
        parent_id: "direct", depth: 2, portfolio_ids: ["direct_002"],
        portfolio_types: ["direct"] }),
  ctx({ context_id: "acquired", context_kind: "type", label: "Acquired", parent_id: "total",
        depth: 1, portfolio_ids: ["acquired_001", "acquired_002"],
        portfolio_types: ["acquired"] }),
  ctx({ context_id: "acquired_001", context_kind: "portfolio", label: "Acquired 001",
        parent_id: "acquired", depth: 2, portfolio_ids: ["acquired_001"],
        portfolio_types: ["acquired"] }),
  ctx({ context_id: "acquired_002", context_kind: "portfolio", label: "Acquired 002",
        parent_id: "acquired", depth: 2, portfolio_ids: ["acquired_002"],
        portfolio_types: ["acquired"] }),
];

describe("PortfolioContextSelector", () => {
  it("renders the governed hierarchy including every individual portfolio", () => {
    render(<PortfolioContextSelector contexts={CONTEXTS} value="total" onChange={vi.fn()} />);
    fireEvent.click(screen.getByTestId("portfolio-context-selector"));
    const options = screen.getAllByRole("option");
    expect(options.map((o) => o.getAttribute("data-context-id"))).toEqual([
      "total", "direct", "direct_001", "direct_002",
      "acquired", "acquired_001", "acquired_002",
    ]);
  });

  it("renders the backend's depth, so groups nest without any client-side grouping", () => {
    render(<PortfolioContextSelector contexts={CONTEXTS} value="total" onChange={vi.fn()} />);
    fireEvent.click(screen.getByTestId("portfolio-context-selector"));
    const depth = (id: string) =>
      screen.getAllByRole("option").find((o) => o.getAttribute("data-context-id") === id)!
        .getAttribute("data-depth");
    expect(depth("total")).toBe("0");
    expect(depth("acquired")).toBe("1");
    expect(depth("acquired_002")).toBe("2");
  });

  it("selects an individual portfolio by its governed context id", () => {
    const onChange = vi.fn();
    render(<PortfolioContextSelector contexts={CONTEXTS} value="total" onChange={onChange} />);
    fireEvent.click(screen.getByTestId("portfolio-context-selector"));
    fireEvent.click(screen.getByText("Acquired 002"));
    expect(onChange).toHaveBeenCalledWith("acquired_002");
  });

  it("summarises a group by its member count, not by a hard-coded description", () => {
    render(<PortfolioContextSelector contexts={CONTEXTS} value="total" onChange={vi.fn()} />);
    fireEvent.click(screen.getByTestId("portfolio-context-selector"));
    expect(screen.getByText("4 portfolios")).toBeInTheDocument();
    expect(screen.getByText("2 direct portfolios")).toBeInTheDocument();
    expect(screen.getByText("2 acquired portfolios")).toBeInTheDocument();
  });

  it("offers no choice when the platform has a single book", () => {
    render(<PortfolioContextSelector contexts={[CONTEXTS[0]]} value="total" onChange={vi.fn()} />);
    fireEvent.click(screen.getByTestId("portfolio-context-selector"));
    expect(screen.queryByRole("listbox")).toBeNull();
  });
});

describe("ViewToggle — capability gating", () => {
  it("disables a view and shows the backend's business reason", () => {
    const reason = "No portfolio in this scope originates new lending, so there is "
      + "no origination pipeline to report.";
    render(<ViewToggle active="funded" onChange={vi.fn()}
      disabledViews={["pipeline"]} disabledReasons={{ pipeline: reason }} />);
    const tab = screen.getByRole("tab", { name: /Pipeline/ });
    expect(tab).toBeDisabled();
    expect(tab).toHaveAttribute("title", reason);
  });

  it("does not fire onChange for a disabled view", () => {
    const onChange = vi.fn();
    render(<ViewToggle active="funded" onChange={onChange} disabledViews={["pipeline"]} />);
    fireEvent.click(screen.getByRole("tab", { name: /Pipeline/ }));
    expect(onChange).not.toHaveBeenCalled();
  });

  it("leaves every view enabled when the backend gates nothing", () => {
    render(<ViewToggle active="funded" onChange={vi.fn()} />);
    for (const name of [/Funded/, /Pipeline/, /Forecast/, /Risk Limits/]) {
      expect(screen.getByRole("tab", { name })).not.toBeDisabled();
    }
  });
});

describe("PortfolioScopeBanner", () => {
  const capability = (over = {}) => ({
    capability: "pipeline" as const,
    enabled: true, reason_code: null, detail: null,
    contributing_portfolios: [], excluded_portfolios: [], partial: false,
    ...over,
  });

  it("shows nothing when the scope is fully applicable", () => {
    const { container } = render(
      <PortfolioScopeBanner context={CONTEXTS[0]} capability={capability()} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("discloses originating-only pipeline while keeping the Total context", () => {
    render(<PortfolioScopeBanner context={CONTEXTS[0]} capability={capability({
      partial: true,
      detail: "Pipeline is sourced from originating portfolios only.",
      contributing_portfolios: ["direct_001", "direct_002"],
      excluded_portfolios: ["acquired_001", "acquired_002"],
    })} />);
    const banner = screen.getByTestId("portfolio-scope-banner");
    expect(banner).toHaveAttribute("data-partial", "true");
    // The workspace still reads as Total — no silent switch to Direct.
    expect(banner.textContent).toContain("Total workspace");
    expect(banner.textContent).toContain("Pipeline is sourced from originating portfolios only.");
    expect(banner.textContent).toContain("Contributing: direct_001, direct_002");
    expect(banner.textContent).toContain("Excluded: acquired_001, acquired_002");
  });

  it("explains a disabled capability rather than showing an empty analysis", () => {
    render(<PortfolioScopeBanner context={CONTEXTS[5]} capability={capability({
      enabled: false,
      reason_code: "NON_ORIGINATING",
      detail: "acquired_001 does not originate new lending.",
      excluded_portfolios: ["acquired_001"],
    })} />);
    const banner = screen.getByTestId("portfolio-scope-banner");
    expect(banner).toHaveAttribute("data-enabled", "false");
    expect(banner.textContent).toContain("acquired_001 does not originate new lending.");
  });
});
