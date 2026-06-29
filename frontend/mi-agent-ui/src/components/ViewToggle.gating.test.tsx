import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ViewToggle } from "./ViewToggle";

/** Acquired-only source portfolios have no origination pipeline, so Pipeline /
 * Forecast must be disabled and non-interactive while Funded stays active. */
describe("ViewToggle gating", () => {
  it("disables Pipeline and Forecast when listed in disabledViews", () => {
    const onChange = vi.fn();
    render(
      <ViewToggle active="funded" onChange={onChange} disabledViews={["pipeline", "forecast"]} />,
    );
    const pipeline = screen.getByRole("tab", { name: /Pipeline/ });
    const forecast = screen.getByRole("tab", { name: /Forecast/ });
    expect(pipeline).toBeDisabled();
    expect(forecast).toBeDisabled();
    fireEvent.click(pipeline);
    expect(onChange).not.toHaveBeenCalled();
    // Funded remains clickable.
    fireEvent.click(screen.getByRole("tab", { name: /Funded/ }));
    expect(onChange).toHaveBeenCalledWith("funded");
  });

  it("enables all views when nothing is disabled", () => {
    const onChange = vi.fn();
    render(<ViewToggle active="funded" onChange={onChange} />);
    const pipeline = screen.getByRole("tab", { name: /Pipeline/ });
    expect(pipeline).not.toBeDisabled();
    fireEvent.click(pipeline);
    expect(onChange).toHaveBeenCalledWith("pipeline");
  });
});
