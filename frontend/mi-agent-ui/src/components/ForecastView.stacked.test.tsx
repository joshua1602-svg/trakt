import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BarList, type BarDatum } from "@/components/pipeline/bits";

/**
 * A forecast bar drawn as one block shows the destination and hides the
 * journey. These pin that the bar is drawn from the parts the engine supplies,
 * and that a payload without parts still renders — a book with no pipeline is
 * all funded, which is honest rather than a rendering fault.
 */
describe("stacked forecast bars", () => {
  const stacked: BarDatum[] = [
    {
      label: "London", value: 12_000_000,
      parts: [
        { label: "Current funded", value: 11_000_000, className: "bg-peri-400/70" },
        { label: "Expected additions", value: 1_000_000, className: "bg-mint-400/80" },
      ],
    },
    {
      label: "Wales", value: 8_000_000,
      parts: [
        { label: "Current funded", value: 7_500_000, className: "bg-peri-400/70" },
        { label: "Expected additions", value: 500_000, className: "bg-mint-400/80" },
      ],
    },
  ];

  it("draws one segment per part", () => {
    const { container } = render(<BarList data={stacked} format="gbp" />);
    expect(container.querySelectorAll(".bg-peri-400\\/70")).toHaveLength(2);
    expect(container.querySelectorAll(".bg-mint-400\\/80")).toHaveLength(2);
  });

  it("names each part on hover, so the colours are readable", () => {
    render(<BarList data={stacked} format="gbp" />);
    expect(screen.getAllByTitle(/Current funded: /)).toHaveLength(2);
    expect(screen.getAllByTitle(/Expected additions: /)).toHaveLength(2);
    expect(screen.getByTitle("Current funded: £11.0MM")).toBeInTheDocument();
  });

  it("still shows the forecast total as the row value", () => {
    render(<BarList data={stacked} format="gbp" />);
    expect(screen.getByText("£12.0MM")).toBeInTheDocument();
  });

  it("falls back to a single bar when the payload carries no parts", () => {
    const plain: BarDatum[] = [{ label: "London", value: 12_000_000 }];
    const { container } = render(<BarList data={plain} format="gbp" />);
    expect(container.querySelectorAll(".bg-mint-400\\/80")).toHaveLength(0);
    expect(container.querySelectorAll(".bg-peri-400\\/70")).toHaveLength(1);
  });

  it("sizes the segments against the same maximum as every other row", () => {
    // Bars stay apples-to-apples: a part's width is its share of the LIST max,
    // not of its own row, or two rows of different size would look alike.
    const { container } = render(<BarList data={stacked} format="gbp" />);
    const widths = [...container.querySelectorAll<HTMLElement>(".bg-peri-400\\/70")]
      .map((el) => parseFloat(el.style.width));
    expect(widths[0]).toBeGreaterThan(widths[1]);
    expect(widths[0]).toBeCloseTo((11 / 12) * 100, 0);
  });
});
