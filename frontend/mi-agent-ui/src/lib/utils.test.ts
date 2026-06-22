import { describe, expect, it } from "vitest";
import { formatValue, toPercentPoints } from "./utils";

describe("percent display honours the dataset contract scale", () => {
  it("formats a fraction (0.51) as 51.0%", () => {
    expect(formatValue(0.51, "pct", "percent_fraction")).toBe("51.0%");
  });

  it("leaves points (51) as 51.0%", () => {
    expect(formatValue(51, "pct", "percent_points")).toBe("51.0%");
  });

  it("LTV range 0.29..0.56 (fraction) displays 29.0%..56.0%, not 0.3%..0.6%", () => {
    expect(formatValue(0.29, "pct", "percent_fraction")).toBe("29.0%");
    expect(formatValue(0.56, "pct", "percent_fraction")).toBe("56.0%");
  });

  it("without a scale, a points value is unchanged (back-compat)", () => {
    expect(formatValue(31.4, "pct")).toBe("31.4%");
  });

  it("toPercentPoints only scales fractions", () => {
    expect(toPercentPoints(0.51, "percent_fraction")).toBeCloseTo(51);
    expect(toPercentPoints(51, "percent_points")).toBe(51);
    expect(toPercentPoints(51, null)).toBe(51);
  });
});
