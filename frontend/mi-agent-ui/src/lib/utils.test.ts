import { describe, expect, it } from "vitest";
import { formatHeading, formatUiTitle, formatValue, toFilenameStem, toPercentPoints } from "./utils";

describe("formatHeading", () => {
  it("leaves curated prose titles untouched", () => {
    expect(formatHeading("Pipeline Bridge to £100MM")).toBe("Pipeline Bridge to £100MM");
    expect(formatHeading("Funded vs. Pipeline Volume")).toBe("Funded vs. Pipeline Volume");
  });
  it("polishes raw snake_case keys", () => {
    expect(formatHeading("average_ltv_by_region")).toBe("Average LTV By Region");
  });
});

describe("formatUiTitle", () => {
  it("title-cases snake_case and spaced input", () => {
    expect(formatUiTitle("average_ltv by region by age_bucket")).toBe("Average LTV By Region By Age Bucket");
  });

  it("keeps known acronyms capitalised", () => {
    expect(formatUiTitle("wa_ltv")).toBe("WA LTV");
    expect(formatUiTitle("spv_balance")).toBe("SPV Balance");
    expect(formatUiTitle("nneg_loss")).toBe("NNEG Loss");
  });

  it("collapses whitespace and trims, returns empty for nullish", () => {
    expect(formatUiTitle("  region   mix  ")).toBe("Region Mix");
    expect(formatUiTitle(undefined)).toBe("");
    expect(formatUiTitle("")).toBe("");
  });
});

describe("toFilenameStem", () => {
  it("slugifies a polished title to snake_case", () => {
    expect(toFilenameStem("Average LTV By Region")).toBe("average_ltv_by_region");
  });
  it("falls back to a default stem", () => {
    expect(toFilenameStem("")).toBe("export");
    expect(toFilenameStem("***")).toBe("export");
  });
});

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
