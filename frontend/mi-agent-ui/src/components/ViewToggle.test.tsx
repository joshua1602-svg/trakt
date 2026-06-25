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

  it("fires onChange with the chosen view", () => {
    const onChange = vi.fn();
    render(<ViewToggle active="funded" onChange={onChange} />);
    fireEvent.click(screen.getByRole("tab", { name: /Forecast/ }));
    expect(onChange).toHaveBeenCalledWith("forecast");
  });
});
