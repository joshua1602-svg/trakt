import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ViewToggle } from "./ViewToggle";

describe("ViewToggle", () => {
  it("renders Funded / Pipeline / Forecast tabs with the active one selected", () => {
    render(<ViewToggle active="funded" onChange={() => {}} />);
    expect(screen.getByRole("tab", { name: /Funded/ })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: /Pipeline/ })).toHaveAttribute("aria-selected", "false");
    expect(screen.getByRole("tab", { name: /Forecast/ })).toHaveAttribute("aria-selected", "false");
  });

  it("exposes the Eligibility & Concentrations tab and fires onChange for it", () => {
    const onChange = vi.fn();
    render(<ViewToggle active="funded" onChange={onChange} />);
    // The tab is LABELLED "Eligibility & Concentrations"; its view id stays
    // `risk_limits`, because that id is in shared links and saved state.
    const tab = screen.getByRole("tab", { name: /Eligibility & Concentrations/ });
    expect(tab).toBeInTheDocument();
    fireEvent.click(tab);
    expect(onChange).toHaveBeenCalledWith("risk_limits");
  });

  it("fires onChange with the chosen view", () => {
    const onChange = vi.fn();
    render(<ViewToggle active="funded" onChange={onChange} />);
    fireEvent.click(screen.getByRole("tab", { name: /Forecast/ }));
    expect(onChange).toHaveBeenCalledWith("forecast");
  });
});
